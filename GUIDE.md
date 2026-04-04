# TRIBE Compare Lab Guide

This repo now ships a compare-first product on top of the existing TRIBE v2 inference code. The current app accepts one or two video, audio, text, or image uploads, runs TRIBE-derived analysis, and returns:

- modality-aware structural metrics
- an overlaid response-curve view
- a per-metric comparison table with plain-English insight lines
- numeric observations
- a visible disclaimer

Primary output is the table, not a recommendation banner. The banner is a summary layer, while the table is the evidence layer.

## Fastest Way To Run

From the repo root:

```bash
./run.sh
```

`run.sh` now owns the full lifecycle:

- loads Node `22.16.0` through `nvm` when available
- creates `.venv` if it does not exist
- upgrades `pip`
- installs Python dependencies with `pip install -e .` when needed
- installs `web/` dependencies with `npm install` when needed
- starts the backend on `http://127.0.0.1:8002`
- starts the frontend on `http://127.0.0.1:3000`
- waits for both health checks to pass
- stops both services and their child processes on `Ctrl-C`

If either target port is already occupied, the script exits instead of attaching to an unknown process.

## Prerequisites

- Python `3.10+`
- Node.js `22.16.0`
- network access for first-time dependency installation and Hugging Face model downloads

## Runtime Defaults

- Backend port: `TRIBEV2_PORT=8002`
- Frontend port: `PORT=3000`
- Frontend-to-backend URL: `TRIBE_API_URL=http://127.0.0.1:8002/analyze`
- Device selection: `TRIBEV2_DEVICE=auto`
- Inference profile: `TRIBEV2_PROFILE=fast`
- Backend prewarm: `TRIBEV2_PREWARM=0`
- Idle model unload: `TRIBEV2_MODEL_IDLE_TTL_SEC=300`
- Default tie threshold: `TRIBE_TIE_THRESHOLD_PCT=7`
- Default tie-threshold note: `TRIBE_TIE_THRESHOLD_BASIS="Fallback default of 7% pending Hour 0 test-retest measurement."`

You can override these before launch:

```bash
TRIBEV2_PORT=8010 PORT=3010 ./run.sh
```

## Fixture Mode

Use fixture mode when the live model path is too slow or unreliable for demo conditions.

```bash
TRIBE_FIXTURE_MODE=1 ./run.sh
```

Relevant flags:

- `TRIBE_FIXTURE_MODE=1`: skip live inference and return deterministic demo payloads
- `TRIBE_FIXTURE_FALLBACK=1`: try live inference first, then fall back to fixtures on failure
- `TRIBEV2_DEVICE=auto|mps|cpu|cuda`: choose the local inference device
- `TRIBEV2_PROFILE=fast|full`: choose the speed-first or full multimodal path
- `TRIBEV2_PREWARM=1`: opt into loading the default local model in the background on backend boot
- `TRIBEV2_MODEL_IDLE_TTL_SEC=<seconds>`: unload cached models after inactivity to release RAM and GPU memory
- `TRIBEV2_BATCH_SIZE_FAST=<n>`: override the fast-profile batch size after benchmarking
- `TRIBEV2_FAST_VIDEO_SAMPLING_HZ=<hz>`: override fast video temporal sampling rate, default `0.5`
- `TRIBEV2_FAST_VIDEO_NUM_FRAMES=<n>`: override fast video frames per clip, default `8`
- `TRIBEV2_FAST_VIDEO_MAX_IMSIZE=<px>`: override fast video max frame dimension, default `384`
- `TRIBEV2_FAST_VIDEO_MAX_DURATION_SEC=<seconds>`: cap fast video analysis duration, default `30`
- `TRIBEV2_IMAGE_PROXY_DURATION_SEC=<seconds>`: requested duration for synthetic image-to-video proxy clips, default `1`
- `TRIBEV2_IMAGE_PROXY_FPS=<fps>`: fps for synthetic image-to-video proxy clips, default `8`
- `TRIBEV2_WHISPERX_CPU_COMPUTE_TYPE=<type>`: CPU transcription compute type, default `int8`
- `TRIBEV2_WHISPERX_CUDA_COMPUTE_TYPE=<type>`: CUDA transcription compute type, default `float16`
- `TRIBEV2_WHISPERX_MODEL=<name>`: WhisperX model name, default `large-v3`
- `TRIBEV2_WHISPERX_BATCH_SIZE=<n>`: WhisperX batch size, default `16`

## Manual Run

If you want to run both processes yourself:

Backend:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
python -m backend.server
```

Frontend:

```bash
source ~/.nvm/nvm.sh
nvm use 22.16.0
cd web
npm install
npm run dev
```

## What To Expect

- The first backend start can be slow because the TRIBE model and feature extractors are heavy.
- The default local path is now `fast`, which uses a video-only scoring profile for video uploads and avoids audio/text preprocessing.
- The fast video path is now aggressively trimmed for local Apple Silicon use: audio is disabled, sampling runs at `0.5 Hz`, clips use `8` frames, frames are capped at `384px`, and analysis is capped to the first `30s` unless you override it.
- Static image uploads are converted into `1s` static MP4 proxies at `8fps` and then analyzed through the standard video event path instead of the raw fast-video shortcut.
- Audio uploads use TRIBE's audio event path.
- Audio transcription now uses device-aware WhisperX defaults. On this Mac, CPU paths use `int8` instead of `float16`, which avoids the failure mode shown in your logs.
- Text uploads use TRIBE's text event path.
- `full` keeps the heavier multimodal TRIBE path as a fallback and verification mode.
- On Apple Silicon, `TRIBEV2_DEVICE=auto` is selective rather than blindly MPS-first: fast video may use MPS, while image jobs stay on CPU because the current TRIBE image path is not stable on MPS.
- The backend no longer prewarms by default, and cached models are unloaded after idle time to reduce background impact on your Mac.
- If MPS is unavailable or downgraded, fast video still works, but CPU inference can remain slow on longer uploads.
- A single upload returns a single stimulus profile.
- Two uploads return the compare view with winners/ties per metric.
- Each comparison-table row now includes a one-line layman insight that explains why one version is ahead in practical terms.
- Some rows can still look numerically close. When the backend assigns a winner, that winner remains highlighted even if the insight notes the gap is a close call.
- The app is descriptive only. It does not claim to predict business outcomes or measured cognition.

## Quick Start Modal

The in-app quick start modal is now designed as a compact onboarding panel rather than a plain help dialog. It should explain:

- who the product is for: a marketing lead or founder making a fast pre-spend creative choice
- what goes in: one or two uploads in the same modality
- what comes out: structural metrics, response curve, comparison rows, and observations
- how to read it: start with the table, then use the banner and deeper evidence modules for support
- what not to assume: this is not a virality, CTR, ROAS, sales, or neuroscience claim

Keep that modal dense, visual, and compact enough to fit the existing dialog footprint without turning into a long-form doc.

## Benchmarking

Use the benchmark harness to compare cold and warm timings on the same asset pair.

```bash
source .venv/bin/activate
python -m backend.benchmark sample.mp4 sample.mp4
```

By default it runs `fast` and `full`, and for `fast` it benchmarks batch sizes `8`, `12`, and `16`.

## Output Schema

The compare flow returns:

```json
{
  "stimulus_a": {
    "modality": "video",
    "curve": [],
    "structural_metrics": {}
  },
  "stimulus_b": {
    "modality": "video",
    "curve": [],
    "structural_metrics": {}
  },
  "comparison": {
    "metrics": [],
    "summary": {}
  },
  "observations": [],
  "diagnostics": {},
  "disclaimer": ""
}
```

## Structural Metrics

- `Opening`: mean activation over the first 20% of windows
- `Middle`: mean activation over the middle 60% of windows
- `Closing`: mean activation over the last 20% of windows
- `Peak`: highest response point
- `Spread`: `max - min`
- `Consistency`: `mean / std` with epsilon guard

In the UI, every technical row should also have a plain-English sentence that tells a non-technical operator why that lead matters.

## Common Failure Modes

- `Port 8002 is already in use`: stop the old backend or set a different `TRIBEV2_PORT`.
- `Port 3000 is already in use`: stop the old frontend or set a different `PORT`.
- slow first analysis on CPU: expected
- progress reaching the 90s now means the job is actually near the end; extractor-heavy preparation stages no longer heartbeat all the way to `95%`
- model download or dependency install stalls: expected on first machine setup
- invalid file type or empty upload: rejected by the backend with a structured error payload
- proxy upload timeouts: the Next route now streams the multipart body directly to the backend instead of reparsing and rebuilding the upload

## Development Notes

- Product constraints live in [`notes.md`](/Users/Sudeep.Shouche/Desktop/My%20Code/CodexHackathon/notes.md).
- Ongoing changelog, explicit user preferences, and implementation insights now live in [`TRACKER.MD`](/Users/Sudeep.Shouche/Desktop/My%20Code/CodexHackathon/TRACKER.MD).
- Keep `TRACKER.MD` updated during development instead of scattering this information across ad hoc notes.
