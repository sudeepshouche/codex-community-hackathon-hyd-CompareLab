"""Minimal HTTP server for TRIBE v2 inference."""

from __future__ import annotations

import logging
import json
import os
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
import tempfile
import threading
import time
import typing as tp
import uuid
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="cgi")
import cgi

from backend.app import (
    CACHE_DIR,
    analyze_asset,
    analyze_assets_compare,
    is_model_loaded,
    start_prewarm,
)

HOST = os.environ.get("TRIBEV2_HOST", "127.0.0.1")
PORT = int(os.environ.get("TRIBEV2_PORT", "8002"))
UPLOAD_DIR = CACHE_DIR / "uploads"
LOG_DIR = CACHE_DIR / "logs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOGGER = logging.getLogger("tribev2.api")
if not LOGGER.handlers:
    LOGGER.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(LOG_DIR / "backend.log")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)
    LOGGER.addHandler(file_handler)
    LOGGER.propagate = False

JOBS_LOCK = threading.Lock()
JOBS: dict[str, "JobState"] = {}
ACTIVE_JOB_STATUSES = {"queued", "warming", "preparing", "scoring", "comparing", "running"}


@dataclass
class JobState:
    job_id: str
    status: str
    progress: int
    message: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    result: dict[str, tp.Any] | None = None
    error: str | None = None
    detail: str | None = None
    error_code: str | None = None

    def to_dict(self) -> dict[str, tp.Any]:
        payload = {
            "job_id": self.job_id,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "created_at": round(self.created_at, 3),
            "updated_at": round(self.updated_at, 3),
        }
        if self.result is not None:
            payload["result"] = self.result
        if self.error is not None:
            payload["error"] = self.error
        if self.detail is not None:
            payload["detail"] = self.detail
        if self.error_code is not None:
            payload["code"] = self.error_code
        return payload


class TribeHandler(BaseHTTPRequestHandler):
    server_version = "tribev2-api/1.0"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "service": "tribev2-api",
                    "model_cached": is_model_loaded(),
                    "active_jobs": _active_job_count(),
                },
            )
            return
        if self.path.startswith("/jobs/"):
            job_id = self.path.removeprefix("/jobs/").strip("/")
            if not job_id:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            job = _get_job(job_id)
            if job is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "job not found"})
                return
            self._send_json(HTTPStatus.OK, job.to_dict())
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self._cors_headers()
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/analyze":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return

        try:
            uploads = self._read_video_uploads()
            job = _create_job()
            LOGGER.info("Accepted job %s", job.job_id)
            worker = threading.Thread(
                target=_run_job,
                args=(job.job_id, uploads),
                daemon=True,
            )
            worker.start()
            self._send_json(
                HTTPStatus.ACCEPTED,
                {
                    "job_id": job.job_id,
                    "status": job.status,
                    "progress": job.progress,
                    "message": job.message,
                    "status_url": f"/jobs/{job.job_id}",
                },
            )
        except ValueError as exc:
            LOGGER.warning("Rejected request: %s", exc)
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {
                    "error": "Invalid analysis request.",
                    "detail": str(exc),
                    "code": "INVALID_REQUEST",
                },
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Unhandled request error")
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {
                    "error": "Analysis request failed.",
                    "detail": str(exc),
                    "code": "REQUEST_FAILED",
                },
            )

    def _read_video_uploads(self) -> dict[str, tp.Any]:
        content_type = self.headers.get("Content-Type", "")
        if content_type.startswith("application/json"):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            data = json.loads(raw.decode("utf-8"))
            video_path_a = Path(data["video_path_a"] if "video_path_a" in data else data["video_path"]).expanduser()
            if not video_path_a.is_file():
                raise ValueError(f"Asset file does not exist: {video_path_a}")

            payload = {
                "video_path_a": video_path_a,
                "video_name_a": video_path_a.name,
                "video_path_b": None,
                "video_name_b": None,
                "cleanup_dir": None,
            }
            if "video_path_b" in data and data["video_path_b"]:
                video_path_b = Path(data["video_path_b"]).expanduser()
                if not video_path_b.is_file():
                    raise ValueError(f"Asset file does not exist: {video_path_b}")
                payload["video_path_b"] = video_path_b
                payload["video_name_b"] = video_path_b.name
            return payload

        if not content_type.startswith("multipart/form-data"):
            raise ValueError("Expected multipart/form-data with a video, audio, text, or image file.")

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
            },
        )
        temp_dir = Path(tempfile.mkdtemp(prefix="tribev2-", dir=str(UPLOAD_DIR)))
        upload_a = self._save_upload_field(form, field_name="video_a", fallback_field="video", directory=temp_dir)
        upload_b = self._save_upload_field(form, field_name="video_b", directory=temp_dir, required=False)
        return {
            "video_path_a": upload_a[0],
            "video_name_a": upload_a[1],
            "video_path_b": upload_b[0] if upload_b else None,
            "video_name_b": upload_b[1] if upload_b else None,
            "cleanup_dir": temp_dir,
        }

    def _save_upload_field(
        self,
        form: cgi.FieldStorage,
        *,
        field_name: str,
        directory: Path,
        fallback_field: str | None = None,
        required: bool = True,
    ) -> tuple[Path, str] | None:
        field: tp.Any = None
        if field_name in form:
            field = form[field_name]
        elif fallback_field and fallback_field in form:
            field = form[fallback_field]

        if field is None:
            if required:
                raise ValueError(f"Missing '{field_name}' file upload.")
            return None

        if not getattr(field, "filename", None):
            if required:
                raise ValueError(f"Uploaded file for '{field_name}' is missing a filename.")
            return None

        original_name = Path(field.filename or "upload.mp4").name
        suffix = Path(original_name).suffix or ".mp4"
        temp_path = directory / f"{field_name}{suffix}"
        with temp_path.open("wb") as handle:
            shutil.copyfileobj(field.file, handle)
        return temp_path, original_name

    def _send_json(self, status: HTTPStatus, payload: dict[str, tp.Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format: str, *args: tp.Any) -> None:  # noqa: A003
        message = format % args
        LOGGER.info("%s - %s", self.address_string(), message)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), TribeHandler)
    LOGGER.info("TRIBE v2 API listening on http://%s:%s", HOST, PORT)
    start_prewarm()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down TRIBE v2 API")
    finally:
        server.server_close()


def _create_job() -> JobState:
    job_id = uuid.uuid4().hex
    job = JobState(
        job_id=job_id,
        status="queued",
        progress=0,
        message="Queued for analysis",
    )
    with JOBS_LOCK:
        JOBS[job_id] = job
    return job


def _get_job(job_id: str) -> JobState | None:
    with JOBS_LOCK:
        return JOBS.get(job_id)


def _active_job_count() -> int:
    with JOBS_LOCK:
        return sum(1 for job in JOBS.values() if job.status not in {"complete", "failed"})


def _update_job(
    job_id: str,
    *,
    status: str | None = None,
    progress: int | None = None,
    message: str | None = None,
    result: dict[str, tp.Any] | None = None,
    error: str | None = None,
    detail: str | None = None,
    error_code: str | None = None,
) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        if status is not None:
            job.status = status
        if progress is not None:
            job.progress = max(0, min(progress, 100))
        if message is not None:
            job.message = message
        if result is not None:
            job.result = result
        if error is not None:
            job.error = error
        if detail is not None:
            job.detail = detail
        if error_code is not None:
            job.error_code = error_code
        job.updated_at = time.time()


def _bump_running_job_progress(job_id: str) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None or job.status not in ACTIVE_JOB_STATUSES:
            return
        max_progress = _heartbeat_progress_cap(job)
        if job.progress >= max_progress:
            return
        job.progress += 1
        job.updated_at = time.time()


def _run_job(
    job_id: str,
    uploads: dict[str, tp.Any],
) -> None:
    heartbeat_stop = threading.Event()

    def report(progress: int, message: str) -> None:
        _update_job(
            job_id,
            status=_status_from_message(message),
            progress=progress,
            message=message,
        )

    def heartbeat() -> None:
        while not heartbeat_stop.wait(2):
            _bump_running_job_progress(job_id)

    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()

    try:
        video_path_a = uploads["video_path_a"]
        video_name_a = uploads["video_name_a"]
        video_path_b = uploads["video_path_b"]
        video_name_b = uploads["video_name_b"]
        if video_path_b is not None:
            result = analyze_assets_compare(
                video_path_a,
                video_path_b,
                asset_name_a=video_name_a,
                asset_name_b=video_name_b,
                progress_callback=report,
            )
        else:
            result = analyze_asset(
                video_path_a,
                asset_name=video_name_a,
                progress_callback=report,
            )
        _update_job(
            job_id,
            status="complete",
            progress=100,
            message="Complete",
            result=result,
        )
        diagnostics = result.get("diagnostics", {}) if isinstance(result, dict) else {}
        LOGGER.info(
            "Job %s complete mode=%s profile=%s device=%s total_ms=%s model_load_ms=%s event_build_ms=%s predict_ms=%s summarize_ms=%s",
            job_id,
            result.get("mode") if isinstance(result, dict) else "unknown",
            diagnostics.get("profile"),
            diagnostics.get("device_resolved"),
            diagnostics.get("total_ms"),
            diagnostics.get("model_load_ms"),
            diagnostics.get("event_build_ms"),
            diagnostics.get("predict_ms"),
            diagnostics.get("summarize_ms"),
        )
    except Exception as exc:  # noqa: BLE001
        error_code, error_message, error_detail = _public_error_payload(exc)
        LOGGER.exception("Job %s failed", job_id)
        _update_job(
            job_id,
            status="failed",
            progress=100,
            message="Analysis failed",
            error=error_message,
            detail=error_detail,
            error_code=error_code,
        )
    finally:
        heartbeat_stop.set()
        cleanup_dir = uploads.get("cleanup_dir")
        if cleanup_dir is not None:
            shutil.rmtree(cleanup_dir, ignore_errors=True)


def _public_error_payload(exc: Exception) -> tuple[str, str, str]:
    detail = str(exc).strip() or exc.__class__.__name__
    if isinstance(exc, ValueError):
        return "INVALID_INPUT", "Unable to analyze the uploaded asset.", detail
    if isinstance(exc, FileNotFoundError):
        return "MISSING_FILE", "The uploaded file could not be found for analysis.", detail
    return "ANALYSIS_FAILED", "Analysis failed while processing the asset.", detail


def _status_from_message(message: str | None) -> str:
    normalized = str(message or "").strip().lower()
    if not normalized:
        return "running"
    if "queued" in normalized:
        return "queued"
    if "loading model" in normalized or "warming" in normalized:
        return "warming"
    if "preparing" in normalized:
        return "preparing"
    if "scoring" in normalized:
        return "scoring"
    if "comparing" in normalized:
        return "comparing"
    if "summarizing" in normalized:
        return "comparing"
    if normalized == "complete":
        return "complete"
    return "running"


def _heartbeat_progress_cap(job: JobState) -> int:
    normalized = str(job.message or "").strip().lower()
    if job.status == "queued":
        return 4
    if job.status == "warming":
        return 15
    if job.status == "preparing":
        if "analyzing b" in normalized:
            return 72
        return 45
    if job.status == "scoring":
        if "analyzing b" in normalized:
            return 88
        return 80
    if job.status == "comparing" or "summarizing" in normalized:
        return 96
    return 90


if __name__ == "__main__":
    main()
