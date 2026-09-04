"""TRIBE v2 analysis helpers."""

from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import logging
import os
import platform
import tempfile
import threading
import time
from pathlib import Path
import typing as tp

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import pandas as pd

from .summary import (
    DEFAULT_TIE_THRESHOLD_PCT,
    build_fixture_result,
    build_stimulus_analysis,
    summarize_comparison,
    summarize_single,
)

LOGGER = logging.getLogger("tribev2.api")

MODEL_ID = os.environ.get("TRIBEV2_MODEL_ID", "facebook/tribev2")
CACHE_DIR = Path(os.environ.get("TRIBEV2_CACHE_DIR", "cache/tribev2"))
CLUSTER = os.environ.get("TRIBEV2_CLUSTER") or None
FIXTURE_MODE = os.environ.get("TRIBE_FIXTURE_MODE", "0") == "1"
FIXTURE_FALLBACK = os.environ.get("TRIBE_FIXTURE_FALLBACK", "0") == "1"
TIE_THRESHOLD_PCT = float(os.environ.get("TRIBE_TIE_THRESHOLD_PCT", DEFAULT_TIE_THRESHOLD_PCT))
TIE_THRESHOLD_BASIS = os.environ.get(
    "TRIBE_TIE_THRESHOLD_BASIS",
    "Fallback default of 7% pending Hour 0 test-retest measurement.",
)

DEFAULT_DEVICE = "auto"
DEFAULT_PROFILE = "fast"
DEFAULT_FAST_BATCH_SIZE = 8
FAST_NUM_WORKERS = 0
DEFAULT_FAST_VIDEO_SAMPLING_HZ = 0.5
DEFAULT_FAST_VIDEO_NUM_FRAMES = 8
DEFAULT_FAST_VIDEO_MAX_IMSIZE = 384
DEFAULT_FAST_VIDEO_MAX_DURATION_SEC = 30.0
DEFAULT_MODEL_IDLE_TTL_SEC = 300.0
DEFAULT_IMAGE_PROXY_DURATION_SEC = 1.0
DEFAULT_IMAGE_PROXY_FPS = 8

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".avi", ".mkv"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg"}
TEXT_SUFFIXES = {".txt"}

_MODEL_CACHE: dict[tuple[str, str, str], tp.Any] = {}
_MODEL_LOCK = threading.RLock()
_INFERENCE_LOCK = threading.Lock()
_PREWARM_LOCK = threading.Lock()
_PREWARM_STARTED = False
_MPS_DISABLED = False
_ACTIVE_MODEL_USERS = 0
_IDLE_UNLOAD_TIMER: threading.Timer | None = None

_RESULT_CACHE_ENABLED = os.environ.get("TRIBEV2_RESULT_CACHE", "1") != "0"
_RESULT_CACHE_DIR = CACHE_DIR / "results"


def _result_cache_key(asset_paths: list[Path], asset_names: list[str], profile: str) -> str:
    """Cache key from file contents, display names, and profile settings."""
    digest = hashlib.sha256()
    digest.update(profile.encode())
    for name, path in zip(asset_names, asset_paths):
        digest.update(name.encode("utf-8", "replace"))
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _result_cache_get(key: str) -> dict[str, tp.Any] | None:
    if not _RESULT_CACHE_ENABLED:
        return None
    try:
        with open(_RESULT_CACHE_DIR / f"{key}.json", encoding="utf-8") as fh:
            cached = json.load(fh)
    except (OSError, ValueError):
        return None
    cached.setdefault("diagnostics", {})["cached"] = True
    return cached


def _result_cache_put(key: str, result: dict[str, tp.Any]) -> None:
    if not _RESULT_CACHE_ENABLED:
        return
    try:
        _RESULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = dict(result)
        payload.setdefault("diagnostics", {})["cached"] = False
        tmp_path = _RESULT_CACHE_DIR / f"{key}.{os.getpid()}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp_path, _RESULT_CACHE_DIR / f"{key}.json")
    except (OSError, TypeError, ValueError):
        LOGGER.debug("Result cache write failed for %s", key, exc_info=True)


def is_model_loaded() -> bool:
    """Return whether the process has already materialized at least one model."""
    with _MODEL_LOCK:
        return bool(_MODEL_CACHE)


def clear_model_cache(*, reset_mps_fallback: bool = False) -> None:
    """Clear cached models for benchmark runs or explicit resets."""
    global _IDLE_UNLOAD_TIMER, _MPS_DISABLED
    with _MODEL_LOCK:
        if _IDLE_UNLOAD_TIMER is not None:
            _IDLE_UNLOAD_TIMER.cancel()
            _IDLE_UNLOAD_TIMER = None
        _MODEL_CACHE.clear()
        if reset_mps_fallback:
            _MPS_DISABLED = False
    _release_accelerator_memory()


def start_prewarm() -> None:
    """Prewarm the default local model in the background."""
    global _PREWARM_STARTED
    if FIXTURE_MODE or not _prewarm_enabled():
        return

    with _PREWARM_LOCK:
        if _PREWARM_STARTED:
            return
        _PREWARM_STARTED = True

    thread = threading.Thread(target=_prewarm_default_model, name="tribev2-prewarm", daemon=True)
    thread.start()


def analyze_asset(
    asset_path: Path,
    *,
    asset_name: str | None = None,
    progress_callback: tp.Callable[[int, str], None] | None = None,
) -> dict[str, tp.Any]:
    """Run TRIBE v2 on one uploaded asset and return the unified schema."""
    asset_name = asset_name or asset_path.name
    upload_modality = detect_modality(asset_path)
    profile = _requested_profile()
    result_modality = _result_modality(upload_modality)

    cache_key = _result_cache_key(
        [asset_path],
        [asset_name],
        f"{profile}:fixture" if FIXTURE_MODE else profile,
    )
    cached = _result_cache_get(cache_key)
    if cached is not None:
        _report_progress(progress_callback, 100, "Complete (cached)")
        return cached

    if FIXTURE_MODE:
        fixture_result = build_fixture_result(
            asset_name_a=asset_name,
            modality=result_modality,
            tie_threshold_pct=TIE_THRESHOLD_PCT,
            threshold_basis=TIE_THRESHOLD_BASIS,
            analysis_diagnostics=_fixture_analysis_diagnostics(profile=profile),
        )
        _result_cache_put(cache_key, fixture_result)
        _report_progress(progress_callback, 100, "Complete")
        return fixture_result

    _begin_model_use()
    total_start = time.perf_counter()
    model_meta: dict[str, tp.Any] | None = None
    event_build_ms = 0.0
    predict_ms = 0.0
    stimulus_build_ms = 0.0
    cleanup_paths: list[Path] = []

    try:
        runtime_asset_path, model_modality, generated_paths = _prepare_runtime_asset(
            asset_path,
            modality=upload_modality,
        )
        cleanup_paths.extend(generated_paths)
        _report_progress(progress_callback, 5, "Analyzing A · loading model")
        model, model_meta = get_model(
            profile=profile,
            modality_group=model_modality,
            source_modality=upload_modality,
            progress_callback=progress_callback,
            phase_label="Analyzing A",
            queue_progress=2,
        )

        stimulus, event_build_ms, predict_ms, stimulus_build_ms, model, model_meta = (
            _analyze_runtime_asset(
                model,
                model_meta=model_meta,
                asset_path=asset_path,
                asset_name=asset_name,
                runtime_asset_path=runtime_asset_path,
                source_modality=upload_modality,
                runtime_modality=model_modality,
                result_modality=result_modality,
                profile=profile,
                timeline="A",
                progress_callback=progress_callback,
                prepare_progress=20,
                score_progress=55,
                phase_label="Analyzing A",
            )
        )

        _report_progress(progress_callback, 85, "Analyzing A · summarizing")
        summarize_start = time.perf_counter()
        result = summarize_single(
            stimulus,
            tie_threshold_pct=TIE_THRESHOLD_PCT,
            threshold_basis=TIE_THRESHOLD_BASIS,
            analysis_diagnostics=_analysis_diagnostics(
                model_meta=model_meta,
                event_build_ms=event_build_ms,
                predict_ms=predict_ms,
                summarize_ms=0.0,
                total_ms=0.0,
            ),
        )
        summarize_ms = stimulus_build_ms + _elapsed_ms(summarize_start)
        result["diagnostics"]["summarize_ms"] = round(summarize_ms, 3)
        result["diagnostics"]["total_ms"] = round(_elapsed_ms(total_start), 3)
        _report_progress(progress_callback, 100, "Complete")
        _result_cache_put(cache_key, result)
        return result
    except Exception:
        if not FIXTURE_FALLBACK:
            raise
        _report_progress(progress_callback, 95, "Switching to fixture mode")
        return build_fixture_result(
            asset_name_a=asset_name,
            modality=result_modality,
            tie_threshold_pct=TIE_THRESHOLD_PCT,
            threshold_basis=TIE_THRESHOLD_BASIS,
            analysis_diagnostics=_fixture_analysis_diagnostics(profile=profile),
        )
    finally:
        _cleanup_generated_paths(cleanup_paths)
        _end_model_use()


def _analyze_runtime_asset(
    model: tp.Any,
    *,
    model_meta: dict[str, tp.Any],
    asset_path: Path,
    asset_name: str,
    runtime_asset_path: Path,
    source_modality: str,
    runtime_modality: str,
    result_modality: str,
    profile: str,
    timeline: str,
    progress_callback: tp.Callable[[int, str], None] | None,
    prepare_progress: int,
    score_progress: int,
    phase_label: str,
) -> tuple[dict[str, tp.Any], float, float, float, tp.Any, dict[str, tp.Any]]:
    _report_progress(progress_callback, prepare_progress, f"{phase_label} · preparing {result_modality}")
    event_build_start = time.perf_counter()
    events = _events_for_asset(
        model,
        asset_path=runtime_asset_path,
        source_modality=source_modality,
        runtime_modality=runtime_modality,
        profile=profile,
        timeline=timeline,
    )
    event_build_ms = _elapsed_ms(event_build_start)

    _report_progress(progress_callback, score_progress, f"{phase_label} · scoring")
    preds, segments, predict_ms, model, model_meta = _predict_with_retry(
        model=model,
        events=events,
        runtime_modality=runtime_modality,
        source_modality=source_modality,
        profile=profile,
        model_meta=model_meta,
        progress_callback=progress_callback,
        phase_label=phase_label,
        queue_progress=max(score_progress - 5, 0),
    )

    stimulus_build_start = time.perf_counter()
    stimulus = build_stimulus_analysis(
        preds=preds,
        segments=segments,
        asset_path=asset_path,
        asset_name=asset_name,
        model_id=MODEL_ID,
        device=model_meta["device_resolved"],
        label=timeline,
        modality=result_modality,
        runtime_diagnostics={
            "device_requested": model_meta["device_requested"],
            "device_resolved": model_meta["device_resolved"],
            "profile": model_meta["profile_requested"],
        },
    )
    stimulus_build_ms = _elapsed_ms(stimulus_build_start)
    return stimulus, event_build_ms, predict_ms, stimulus_build_ms, model, model_meta


def analyze_assets_compare(
    asset_path_a: Path,
    asset_path_b: Path,
    *,
    asset_name_a: str | None = None,
    asset_name_b: str | None = None,
    progress_callback: tp.Callable[[int, str], None] | None = None,
) -> dict[str, tp.Any]:
    """Run TRIBE v2 on two same-modality assets and return the unified compare schema."""
    asset_name_a = asset_name_a or asset_path_a.name
    asset_name_b = asset_name_b or asset_path_b.name
    upload_modality_a = detect_modality(asset_path_a)
    upload_modality_b = detect_modality(asset_path_b)
    if upload_modality_a != upload_modality_b:
        raise ValueError("Compare requires both uploads to use the same modality.")
    result_modality_a = _result_modality(upload_modality_a)
    result_modality_b = _result_modality(upload_modality_b)

    profile = _requested_profile()

    cache_key = _result_cache_key(
        [asset_path_a, asset_path_b],
        [asset_name_a, asset_name_b],
        f"{profile}:fixture" if FIXTURE_MODE else profile,
    )
    cached = _result_cache_get(cache_key)
    if cached is not None:
        _report_progress(progress_callback, 100, "Complete (cached)")
        return cached

    if FIXTURE_MODE:
        fixture_result = build_fixture_result(
            asset_name_a=asset_name_a,
            asset_name_b=asset_name_b,
            modality=result_modality_a,
            tie_threshold_pct=TIE_THRESHOLD_PCT,
            threshold_basis=TIE_THRESHOLD_BASIS,
            analysis_diagnostics=_fixture_analysis_diagnostics(profile=profile),
        )
        _result_cache_put(cache_key, fixture_result)
        _report_progress(progress_callback, 100, "Complete")
        return fixture_result

    _begin_model_use()
    total_start = time.perf_counter()
    model_meta: dict[str, tp.Any] | None = None
    event_build_ms = 0.0
    predict_ms = 0.0
    stimulus_build_ms = 0.0
    cleanup_paths: list[Path] = []

    try:
        runtime_asset_path_a, model_modality_a, generated_paths_a = _prepare_runtime_asset(
            asset_path_a,
            modality=upload_modality_a,
        )
        runtime_asset_path_b, model_modality_b, generated_paths_b = _prepare_runtime_asset(
            asset_path_b,
            modality=upload_modality_b,
        )
        cleanup_paths.extend(generated_paths_a)
        cleanup_paths.extend(generated_paths_b)
        if model_modality_a != model_modality_b:
            raise ValueError("Compare requires both uploads to resolve to the same runtime modality.")

        _report_progress(progress_callback, 5, "Analyzing A · loading model")
        model, model_meta = get_model(
            profile=profile,
            modality_group=model_modality_a,
            source_modality=upload_modality_a,
            progress_callback=progress_callback,
            phase_label="Analyzing A",
            queue_progress=2,
        )

        stimulus_a, event_build_ms_a, predict_ms_a, stimulus_build_ms_a, model, model_meta = (
            _analyze_runtime_asset(
                model,
                model_meta=model_meta,
                asset_path=asset_path_a,
                asset_name=asset_name_a,
                runtime_asset_path=runtime_asset_path_a,
                source_modality=upload_modality_a,
                runtime_modality=model_modality_a,
                result_modality=result_modality_a,
                profile=profile,
                timeline="A",
                progress_callback=progress_callback,
                prepare_progress=20,
                score_progress=40,
                phase_label="Analyzing A",
            )
        )
        event_build_ms += event_build_ms_a
        predict_ms += predict_ms_a
        stimulus_build_ms += stimulus_build_ms_a

        stimulus_b, event_build_ms_b, predict_ms_b, stimulus_build_ms_b, model, model_meta = (
            _analyze_runtime_asset(
                model,
                model_meta=model_meta,
                asset_path=asset_path_b,
                asset_name=asset_name_b,
                runtime_asset_path=runtime_asset_path_b,
                source_modality=upload_modality_b,
                runtime_modality=model_modality_b,
                result_modality=result_modality_b,
                profile=profile,
                timeline="B",
                progress_callback=progress_callback,
                prepare_progress=55,
                score_progress=75,
                phase_label="Analyzing B",
            )
        )
        event_build_ms += event_build_ms_b
        predict_ms += predict_ms_b
        stimulus_build_ms += stimulus_build_ms_b

        _report_progress(progress_callback, 90, "Comparing")
        summarize_start = time.perf_counter()
        result = summarize_comparison(
            stimulus_a,
            stimulus_b,
            tie_threshold_pct=TIE_THRESHOLD_PCT,
            threshold_basis=TIE_THRESHOLD_BASIS,
            analysis_diagnostics=_analysis_diagnostics(
                model_meta=model_meta,
                event_build_ms=event_build_ms,
                predict_ms=predict_ms,
                summarize_ms=0.0,
                total_ms=0.0,
            ),
        )
        summarize_ms = stimulus_build_ms + _elapsed_ms(summarize_start)
        result["diagnostics"]["summarize_ms"] = round(summarize_ms, 3)
        result["diagnostics"]["total_ms"] = round(_elapsed_ms(total_start), 3)
        _report_progress(progress_callback, 100, "Complete")
        _result_cache_put(cache_key, result)
        return result
    except Exception:
        if not FIXTURE_FALLBACK:
            raise
        _report_progress(progress_callback, 95, "Switching to fixture mode")
        return build_fixture_result(
            asset_name_a=asset_name_a,
            asset_name_b=asset_name_b,
            modality=result_modality_a,
            tie_threshold_pct=TIE_THRESHOLD_PCT,
            threshold_basis=TIE_THRESHOLD_BASIS,
            analysis_diagnostics=_fixture_analysis_diagnostics(profile=profile),
        )
    finally:
        _cleanup_generated_paths(cleanup_paths)
        _end_model_use()


def get_model(
    *,
    profile: str,
    modality_group: str,
    source_modality: str | None = None,
    progress_callback: tp.Callable[[int, str], None] | None = None,
    phase_label: str = "Analyzing A",
    queue_progress: int = 2,
) -> tuple[tp.Any, dict[str, tp.Any]]:
    """Load or reuse a profile-specific model."""
    requested_profile = _requested_profile(profile)
    effective_profile = _effective_profile(requested_profile, source_modality or modality_group)
    total_model_load_ms = 0.0

    while True:
        device_requested = _requested_device()
        device_resolved = _resolve_device(
            device_requested,
            profile=effective_profile,
            source_modality=source_modality or modality_group,
        )
        key = (effective_profile, device_resolved, modality_group)

        with _MODEL_LOCK:
            cached_model = _MODEL_CACHE.get(key)
        if cached_model is not None:
            return cached_model, _model_metadata(
                cached_model,
                device_requested=device_requested,
                device_resolved=device_resolved,
                profile_requested=requested_profile,
                profile_effective=effective_profile,
                cache_warm=True,
                model_load_ms=total_model_load_ms,
            )

        with _inference_guard(
            progress_callback=progress_callback,
            queue_progress=queue_progress,
            queue_message=f"{phase_label} · queued",
        ):
            with _MODEL_LOCK:
                cached_model = _MODEL_CACHE.get(key)
                if cached_model is not None:
                    return cached_model, _model_metadata(
                        cached_model,
                        device_requested=device_requested,
                        device_resolved=device_resolved,
                        profile_requested=requested_profile,
                        profile_effective=effective_profile,
                        cache_warm=True,
                        model_load_ms=total_model_load_ms,
                    )

                _prepare_runtime_env()
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                config_update = _config_update_for_model(
                    profile=effective_profile,
                    modality_group=modality_group,
                )

                load_start = time.perf_counter()
                try:
                    from tribev2 import TribeModel

                    model = TribeModel.from_pretrained(
                        MODEL_ID,
                        cache_folder=str(CACHE_DIR),
                        device=device_resolved,
                        cluster=CLUSTER,
                        config_update=config_update,
                    )
                    _force_device(model, device_resolved)
                except Exception as exc:
                    total_model_load_ms += _elapsed_ms(load_start)
                    if device_resolved == "mps" and _activate_cpu_fallback(exc, phase="model load"):
                        continue
                    raise

                total_model_load_ms += _elapsed_ms(load_start)
                _MODEL_CACHE[key] = model
                LOGGER.info(
                    "Loaded model profile=%s source_modality=%s runtime_modality=%s device=%s cache_warm=%s batch_size=%s duration_trs=%s num_workers=%s model_load_ms=%.1f",
                    requested_profile,
                    source_modality or modality_group,
                    modality_group,
                    device_resolved,
                    False,
                    getattr(model.data, "batch_size", "n/a"),
                    getattr(model.data, "duration_trs", "n/a"),
                    getattr(model.data, "num_workers", "n/a"),
                    total_model_load_ms,
                )
                return model, _model_metadata(
                    model,
                    device_requested=device_requested,
                    device_resolved=device_resolved,
                    profile_requested=requested_profile,
                    profile_effective=effective_profile,
                    cache_warm=False,
                    model_load_ms=total_model_load_ms,
                )


def detect_modality(asset_path: Path) -> str:
    suffix = asset_path.suffix.lower()
    if suffix in VIDEO_SUFFIXES:
        return "video"
    if suffix in AUDIO_SUFFIXES:
        return "audio"
    if suffix in TEXT_SUFFIXES:
        return "text"
    if suffix in IMAGE_SUFFIXES:
        return "image"
    raise ValueError(f"Unsupported asset type: {suffix or 'unknown'}")


def _result_modality(upload_modality: str) -> str:
    return upload_modality


def _prepare_runtime_asset(asset_path: Path, *, modality: str) -> tuple[Path, str, list[Path]]:
    if modality == "image":
        proxy_path = _create_video_proxy_for_image(asset_path)
        return proxy_path, "video", [proxy_path]
    return asset_path, modality, []


def _create_video_proxy_for_image(asset_path: Path) -> Path:
    from moviepy import ImageClip

    duration_sec = _effective_image_proxy_duration_sec()
    fps = _image_proxy_fps()
    temp_dir = Path(tempfile.mkdtemp(prefix="tribev2-image-proxy-", dir=str(CACHE_DIR / "uploads")))
    output_path = temp_dir / f"{asset_path.stem}.mp4"
    clip = ImageClip(str(asset_path), duration=duration_sec)
    with (
        open(os.devnull, "w") as devnull,
        contextlib.redirect_stdout(devnull),
        contextlib.redirect_stderr(devnull),
    ):
        clip.write_videofile(str(output_path), codec="libx264", audio=False, fps=fps, logger=None)
    clip.close()
    LOGGER.info(
        "Created synthetic video proxy for image %s duration=%.2fs fps=%d path=%s",
        asset_path.name,
        duration_sec,
        fps,
        output_path,
    )
    return output_path


def _cleanup_generated_paths(paths: list[Path]) -> None:
    for path in paths:
        with contextlib.suppress(Exception):
            if path.exists():
                path.unlink()
        with contextlib.suppress(Exception):
            parent = path.parent
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()


def _events_for_asset(
    model: tp.Any,
    *,
    asset_path: Path,
    source_modality: str,
    runtime_modality: str,
    profile: str,
    timeline: str,
) -> pd.DataFrame:
    effective_profile = _effective_profile(profile, source_modality)
    if runtime_modality == "video" and effective_profile == "fast":
        source_duration_sec = _probe_video_duration_seconds(asset_path)
        analysis_duration_sec = _fast_video_analysis_duration_sec(source_duration_sec)
        if source_duration_sec is not None and analysis_duration_sec is not None:
            if analysis_duration_sec < source_duration_sec:
                LOGGER.info(
                    "Fast video path capped %s from %.2fs to %.2fs at %.2fHz with %d frames and max_imsize=%s",
                    asset_path.name,
                    source_duration_sec,
                    analysis_duration_sec,
                    _fast_video_sampling_hz(),
                    _fast_video_num_frames(),
                    _fast_video_max_imsize(),
                )
            else:
                LOGGER.info(
                    "Fast video path analyzing %s at %.2fHz with %d frames and max_imsize=%s",
                    asset_path.name,
                    _fast_video_sampling_hz(),
                    _fast_video_num_frames(),
                    _fast_video_max_imsize(),
                )
        event = {
            "type": "Video",
            "filepath": str(asset_path),
            "start": 0.0,
            "timeline": timeline,
            "subject": timeline,
        }
        if analysis_duration_sec is not None:
            event["duration"] = analysis_duration_sec
        return pd.DataFrame([event])

    if runtime_modality == "video":
        events = model.get_events_dataframe(video_path=str(asset_path)).copy()
        events.loc[:, "timeline"] = timeline
        events.loc[:, "subject"] = timeline
        return events

    if runtime_modality == "audio":
        events = model.get_events_dataframe(audio_path=str(asset_path), audio_only=True).copy()
        events.loc[:, "timeline"] = timeline
        events.loc[:, "subject"] = timeline
        return events

    if runtime_modality == "text":
        events = model.get_events_dataframe(text_path=str(asset_path)).copy()
        events.loc[:, "timeline"] = timeline
        events.loc[:, "subject"] = timeline
        return events

    return pd.DataFrame(
        [
            {
                "type": "Image",
                "filepath": str(asset_path),
                "start": 0.0,
                "duration": float(model.data.TR),
                "timeline": timeline,
                "subject": timeline,
            }
        ]
    )


def _predict_with_retry(
    *,
    model: tp.Any,
    events: pd.DataFrame,
    runtime_modality: str,
    source_modality: str,
    profile: str,
    model_meta: dict[str, tp.Any],
    progress_callback: tp.Callable[[int, str], None] | None,
    phase_label: str,
    queue_progress: int,
) -> tuple[np.ndarray, list[tp.Any], float, tp.Any, dict[str, tp.Any]]:
    total_predict_ms = 0.0
    current_model = model
    current_meta = dict(model_meta)

    while True:
        try:
            with _inference_guard(
                progress_callback=progress_callback,
                queue_progress=queue_progress,
                queue_message=f"{phase_label} · queued",
            ):
                predict_start = time.perf_counter()
                preds, segments = current_model.predict(events=events, verbose=False)
                total_predict_ms += _elapsed_ms(predict_start)
            return np.asarray(preds), list(segments), total_predict_ms, current_model, current_meta
        except Exception as exc:
            if current_meta.get("device_resolved") == "mps" and _activate_cpu_fallback(exc, phase="predict"):
                current_model, retry_meta = get_model(
                    profile=profile,
                    modality_group=runtime_modality,
                    source_modality=source_modality,
                    progress_callback=progress_callback,
                    phase_label=phase_label,
                    queue_progress=queue_progress,
                )
                retry_meta["model_load_ms"] = round(
                    float(current_meta.get("model_load_ms", 0.0))
                    + float(retry_meta.get("model_load_ms", 0.0)),
                    3,
                )
                current_meta = retry_meta
                continue
            raise


def _analysis_diagnostics(
    *,
    model_meta: dict[str, tp.Any],
    event_build_ms: float,
    predict_ms: float,
    summarize_ms: float,
    total_ms: float,
) -> dict[str, tp.Any]:
    return {
        "device_requested": model_meta["device_requested"],
        "device_resolved": model_meta["device_resolved"],
        "profile": model_meta["profile_requested"],
        "cache_warm": bool(model_meta["cache_warm"]),
        "model_load_ms": round(float(model_meta["model_load_ms"]), 3),
        "event_build_ms": round(event_build_ms, 3),
        "predict_ms": round(predict_ms, 3),
        "summarize_ms": round(summarize_ms, 3),
        "total_ms": round(total_ms, 3),
        "batch_size": int(model_meta["batch_size"]),
        "duration_trs": int(model_meta["duration_trs"]),
        "num_workers": int(model_meta["num_workers"]),
        "video_sampling_hz": round(float(model_meta.get("video_sampling_hz", 0.0)), 3),
        "video_num_frames": int(model_meta.get("video_num_frames", 0)),
        "video_max_imsize": model_meta.get("video_max_imsize"),
        "video_clip_duration": model_meta.get("video_clip_duration"),
    }


def _fixture_analysis_diagnostics(*, profile: str) -> dict[str, tp.Any]:
    return {
        "device_requested": "fixture",
        "device_resolved": "fixture",
        "profile": profile,
        "cache_warm": True,
        "model_load_ms": 0.0,
        "event_build_ms": 0.0,
        "predict_ms": 0.0,
        "summarize_ms": 0.0,
        "total_ms": 0.0,
        "batch_size": 0,
        "duration_trs": 0,
        "num_workers": 0,
        "video_sampling_hz": 0.0,
        "video_num_frames": 0,
        "video_max_imsize": None,
        "video_clip_duration": None,
    }


def _model_metadata(
    model: tp.Any,
    *,
    device_requested: str,
    device_resolved: str,
    profile_requested: str,
    profile_effective: str,
    cache_warm: bool,
    model_load_ms: float,
) -> dict[str, tp.Any]:
    data = getattr(model, "data", None)
    batch_size = getattr(data, "batch_size", 0) if data is not None else 0
    duration_trs = getattr(data, "duration_trs", 0) if data is not None else 0
    num_workers = getattr(data, "num_workers", 0) if data is not None else 0
    video_feature = getattr(data, "video_feature", None) if data is not None else None
    video_sampling_hz = getattr(video_feature, "frequency", 0) if video_feature is not None else 0
    video_num_frames = getattr(video_feature, "num_frames", 0) if video_feature is not None else 0
    video_max_imsize = getattr(video_feature, "max_imsize", None) if video_feature is not None else None
    video_clip_duration = (
        getattr(video_feature, "clip_duration", None) if video_feature is not None else None
    )
    return {
        "device_requested": device_requested,
        "device_resolved": device_resolved,
        "profile_requested": profile_requested,
        "profile_effective": profile_effective,
        "cache_warm": cache_warm,
        "model_load_ms": round(model_load_ms, 3),
        "batch_size": int(batch_size or 0),
        "duration_trs": int(duration_trs or 0),
        "num_workers": int(num_workers or 0),
        "video_sampling_hz": float(video_sampling_hz or 0.0),
        "video_num_frames": int(video_num_frames or 0),
        "video_max_imsize": int(video_max_imsize) if video_max_imsize is not None else None,
        "video_clip_duration": (
            round(float(video_clip_duration), 3) if video_clip_duration is not None else None
        ),
    }


def _config_update_for_model(*, profile: str, modality_group: str) -> dict[str, tp.Any]:
    if modality_group == "audio":
        return {
            "data.features_to_use": ["audio"],
            "data.num_workers": FAST_NUM_WORKERS,
            "data.batch_size": _fast_batch_size(),
        }

    if modality_group == "text":
        batch_size = _fast_batch_size()
        return {
            "data.features_to_use": ["text"],
            "data.num_workers": FAST_NUM_WORKERS,
            "data.batch_size": batch_size,
            "data.text_feature.batch_size": min(batch_size, 4),
        }

    if profile == "fast":
        sampling_hz = _fast_video_sampling_hz()
        return {
            "data.features_to_use": ["video"],
            "data.frequency": sampling_hz,
            "data.num_workers": FAST_NUM_WORKERS,
            "data.batch_size": _fast_batch_size(),
            "data.video_feature.use_audio": False,
            "data.video_feature.clip_duration": round(1.0 / sampling_hz, 3),
            "data.video_feature.max_imsize": _fast_video_max_imsize(),
            "data.video_feature.num_frames": _fast_video_num_frames(),
        }

    return {}


def _prepare_runtime_env() -> None:
    """Set conservative defaults for OpenMP-heavy inference runtimes."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


def _requested_device() -> str:
    return (os.environ.get("TRIBEV2_DEVICE", DEFAULT_DEVICE) or DEFAULT_DEVICE).strip().lower()


def _requested_profile(profile: str | None = None) -> str:
    value = (profile or os.environ.get("TRIBEV2_PROFILE", DEFAULT_PROFILE) or DEFAULT_PROFILE).strip().lower()
    return value if value in {"fast", "full"} else DEFAULT_PROFILE


def _prewarm_enabled() -> bool:
    return os.environ.get("TRIBEV2_PREWARM", "0") == "1"


def _model_idle_ttl_sec() -> float:
    raw_value = os.environ.get(
        "TRIBEV2_MODEL_IDLE_TTL_SEC",
        str(DEFAULT_MODEL_IDLE_TTL_SEC),
    )
    try:
        return max(0.0, float(raw_value))
    except ValueError:
        return DEFAULT_MODEL_IDLE_TTL_SEC


def _fast_batch_size() -> int:
    raw_value = os.environ.get("TRIBEV2_BATCH_SIZE_FAST", str(DEFAULT_FAST_BATCH_SIZE))
    try:
        return max(1, int(raw_value))
    except ValueError:
        return DEFAULT_FAST_BATCH_SIZE


def _fast_video_sampling_hz() -> float:
    raw_value = os.environ.get(
        "TRIBEV2_FAST_VIDEO_SAMPLING_HZ",
        str(DEFAULT_FAST_VIDEO_SAMPLING_HZ),
    )
    try:
        return max(0.125, float(raw_value))
    except ValueError:
        return DEFAULT_FAST_VIDEO_SAMPLING_HZ


def _fast_video_num_frames() -> int:
    raw_value = os.environ.get(
        "TRIBEV2_FAST_VIDEO_NUM_FRAMES",
        str(DEFAULT_FAST_VIDEO_NUM_FRAMES),
    )
    try:
        return max(1, int(raw_value))
    except ValueError:
        return DEFAULT_FAST_VIDEO_NUM_FRAMES


def _fast_video_max_imsize() -> int:
    raw_value = os.environ.get(
        "TRIBEV2_FAST_VIDEO_MAX_IMSIZE",
        str(DEFAULT_FAST_VIDEO_MAX_IMSIZE),
    )
    try:
        return max(64, int(raw_value))
    except ValueError:
        return DEFAULT_FAST_VIDEO_MAX_IMSIZE


def _fast_video_max_duration_sec() -> float:
    raw_value = os.environ.get(
        "TRIBEV2_FAST_VIDEO_MAX_DURATION_SEC",
        str(DEFAULT_FAST_VIDEO_MAX_DURATION_SEC),
    )
    try:
        return max(0.0, float(raw_value))
    except ValueError:
        return DEFAULT_FAST_VIDEO_MAX_DURATION_SEC


def _image_proxy_duration_sec() -> float:
    raw_value = os.environ.get(
        "TRIBEV2_IMAGE_PROXY_DURATION_SEC",
        str(DEFAULT_IMAGE_PROXY_DURATION_SEC),
    )
    try:
        return max(1.0, float(raw_value))
    except ValueError:
        return DEFAULT_IMAGE_PROXY_DURATION_SEC


def _effective_image_proxy_duration_sec() -> float:
    minimum_duration_sec = 1.0 / _fast_video_sampling_hz()
    return round(max(_image_proxy_duration_sec(), minimum_duration_sec), 3)


def _image_proxy_fps() -> int:
    raw_value = os.environ.get(
        "TRIBEV2_IMAGE_PROXY_FPS",
        str(DEFAULT_IMAGE_PROXY_FPS),
    )
    try:
        return max(1, int(raw_value))
    except ValueError:
        return DEFAULT_IMAGE_PROXY_FPS


def _fast_video_analysis_duration_sec(source_duration_sec: float | None) -> float | None:
    if source_duration_sec is None:
        return None
    max_duration_sec = _fast_video_max_duration_sec()
    if max_duration_sec <= 0:
        return round(source_duration_sec, 3)
    return round(min(source_duration_sec, max_duration_sec), 3)


def _probe_video_duration_seconds(asset_path: Path) -> float | None:
    try:
        from moviepy import VideoFileClip

        clip = VideoFileClip(str(asset_path))
        try:
            return float(clip.duration)
        finally:
            clip.close()
    except Exception as exc:
        LOGGER.warning("Unable to probe duration for %s: %s", asset_path, exc)
        return None


def _effective_profile(profile: str, source_modality: str) -> str:
    if source_modality == "image":
        return "fast"
    return profile


def _resolve_device(
    requested_device: str,
    *,
    profile: str,
    source_modality: str,
) -> str:
    """Return a device string that is safe for this local machine."""
    requested = (requested_device or DEFAULT_DEVICE).strip().lower()
    if requested == "auto":
        import torch

        if source_modality == "image":
            return "cpu"

        # Competitor implementation and local logs both show MPS instability for this
        # stack on Apple Silicon. Keep GPU auto-selection only for the fast video path.
        if (
            not _MPS_DISABLED
            and platform.system() == "Darwin"
            and platform.machine() == "arm64"
            and profile == "fast"
            and source_modality == "video"
        ):
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is not None and mps_backend.is_built() and mps_backend.is_available():
                return "mps"

        if getattr(torch.version, "cuda", None) and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    if requested == "mps":
        if source_modality == "image":
            return "cpu"
        if _MPS_DISABLED:
            return "cpu"

        import torch

        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_built() and mps_backend.is_available():
            return "mps"
        return "cpu"

    if requested == "cuda":
        import torch

        if getattr(torch.version, "cuda", None) and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    return requested or "cpu"


def _force_device(model: tp.Any, device: str) -> None:
    """Keep nested feature extractors on one explicit device."""
    data = getattr(model, "data", None)
    if data is None:
        return

    for feature_name in ("text_feature", "audio_feature", "video_feature", "image_feature"):
        feature = getattr(data, feature_name, None)
        if feature is None:
            continue
        if hasattr(feature, "device"):
            try:
                feature.device = device
            except Exception:
                pass
        nested_image = getattr(feature, "image", None)
        if nested_image is not None and hasattr(nested_image, "device"):
            try:
                nested_image.device = device
            except Exception:
                pass


def _activate_cpu_fallback(exc: Exception, *, phase: str) -> bool:
    global _MPS_DISABLED
    if not _is_mps_runtime_error(exc):
        return False

    with _MODEL_LOCK:
        if _MPS_DISABLED:
            return False
        _MPS_DISABLED = True
        stale_keys = [key for key in _MODEL_CACHE if key[1] == "mps"]
        for key in stale_keys:
            _MODEL_CACHE.pop(key, None)

    LOGGER.warning("MPS %s failed; downgrading this process to CPU: %s", phase, exc)
    return True


def _is_mps_runtime_error(exc: Exception) -> bool:
    message = str(exc).strip().lower()
    if not message:
        return False
    tokens = (
        "mps",
        "metal",
        "placeholder storage",
        "not compiled with mps",
        "mps backend",
        "mps device",
        "mpsgraph",
    )
    return any(token in message for token in tokens)


@contextlib.contextmanager
def _inference_guard(
    *,
    progress_callback: tp.Callable[[int, str], None] | None,
    queue_progress: int,
    queue_message: str,
) -> tp.Iterator[None]:
    if _INFERENCE_LOCK.locked():
        _report_progress(progress_callback, queue_progress, queue_message)
    _INFERENCE_LOCK.acquire()
    try:
        yield
    finally:
        _INFERENCE_LOCK.release()


def _prewarm_default_model() -> None:
    try:
        profile = _requested_profile()
        _begin_model_use()
        model, metadata = get_model(profile=profile, modality_group="video")
        LOGGER.info(
            "Prewarm complete profile=%s device=%s cache_warm=%s model_load_ms=%.1f batch_size=%s",
            metadata["profile_requested"],
            metadata["device_resolved"],
            metadata["cache_warm"],
            metadata["model_load_ms"],
            metadata["batch_size"],
        )
        del model
    except Exception:
        LOGGER.exception("Background prewarm failed")
    finally:
        _end_model_use()


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _report_progress(
    progress_callback: tp.Callable[[int, str], None] | None,
    progress: int,
    message: str,
) -> None:
    """Best-effort progress updates for long-running jobs."""
    if progress_callback is None:
        return

    try:
        progress_callback(max(0, min(progress, 100)), message)
    except Exception:
        pass


def _begin_model_use() -> None:
    global _ACTIVE_MODEL_USERS, _IDLE_UNLOAD_TIMER
    with _MODEL_LOCK:
        _ACTIVE_MODEL_USERS += 1
        if _IDLE_UNLOAD_TIMER is not None:
            _IDLE_UNLOAD_TIMER.cancel()
            _IDLE_UNLOAD_TIMER = None


def _end_model_use() -> None:
    global _ACTIVE_MODEL_USERS
    with _MODEL_LOCK:
        _ACTIVE_MODEL_USERS = max(0, _ACTIVE_MODEL_USERS - 1)
        if _ACTIVE_MODEL_USERS == 0:
            _schedule_idle_unload_locked()


def _schedule_idle_unload_locked() -> None:
    global _IDLE_UNLOAD_TIMER
    ttl_sec = _model_idle_ttl_sec()
    if ttl_sec <= 0:
        return
    if _IDLE_UNLOAD_TIMER is not None:
        _IDLE_UNLOAD_TIMER.cancel()
    timer = threading.Timer(ttl_sec, _unload_models_if_idle)
    timer.daemon = True
    _IDLE_UNLOAD_TIMER = timer
    timer.start()


def _unload_models_if_idle() -> None:
    global _IDLE_UNLOAD_TIMER
    released = False
    with _MODEL_LOCK:
        _IDLE_UNLOAD_TIMER = None
        if _ACTIVE_MODEL_USERS > 0 or not _MODEL_CACHE:
            if _ACTIVE_MODEL_USERS == 0 and _MODEL_CACHE:
                _schedule_idle_unload_locked()
            return
        _MODEL_CACHE.clear()
        released = True
    if released:
        _release_accelerator_memory()
        LOGGER.info("Unloaded cached models after %.0fs of inactivity", _model_idle_ttl_sec())


def _release_accelerator_memory() -> None:
    gc.collect()
    with contextlib.suppress(Exception):
        import torch

        mps_module = getattr(torch, "mps", None)
        if mps_module is not None and hasattr(mps_module, "empty_cache"):
            mps_module.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
