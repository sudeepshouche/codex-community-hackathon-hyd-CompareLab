# TRACKER

## Purpose

Persistent development tracker for:

- changelog
- explicit user preferences
- implementation insights

Update this file as part of ongoing development work.

## Changelog

### 2026-04-04

- Reworked the product from a single-video response lab into a compare-first TRIBE Compare Lab.
- Standardized frontend/backend defaults on port `8002` for the backend API.
- Added unified single/compare result schema with `stimulus_a`, `stimulus_b`, `comparison`, `observations`, `diagnostics`, and `disclaimer`.
- Added structural metric computation: Opening, Middle, Closing, Peak, Spread, Consistency.
- Added thresholded comparison logic with default fallback tie threshold of `7%`.
- Added deterministic observation generation with numeric citations only.
- Added fixture mode and fallback fixture mode for demo reliability.
- Replaced the frontend with a compare-first desktop UI: two uploads, phased progress, overlaid curves, comparison table, observations, and adjacent disclaimer.
- Updated Next/Tailwind/PostCSS config so the frontend builds cleanly under the current toolchain.
- Replaced `run.sh` with a full lifecycle bootstrap script that creates `.venv`, installs dependencies, launches both services, waits for health checks, and cleans up child processes on exit.
- Updated `GUIDE.md` to reflect the current product and launcher behavior.
- Fixed broken UI styling by exporting design tokens into Tailwind v4's `@theme inline` layer so shadcn utility classes like `bg-primary`, `bg-muted`, and `bg-card` resolve again.
- Added image ad support on the same compare flow, with same-modality enforcement and honest metric support for static assets.
- Fixed stuck-at-5% progress behavior by adding backend heartbeat progress updates during long-running jobs, richer stage messages, backend error logging, and frontend toast-based error surfacing.
- Implemented Apple M3 performance path defaults: `TRIBEV2_DEVICE=auto`, `TRIBEV2_PROFILE=fast`, and background prewarm.
- Replaced the single global model cache with profile-aware cached model variants keyed by profile, resolved device, and modality group.
- Added MPS-first device resolution with one-time CPU downgrade on MPS runtime failure.
- Reworked compare analysis to build one combined events dataframe and score A/B in a single `model.predict` pass.
- Added per-request timing diagnostics for model load, event build, predict, summarize, and total wall time.
- Added a local benchmark harness at `python -m backend.benchmark` for cold/warm profile and batch-size timing runs.
- Added local machine safety rails: prewarm is now opt-in, cached models unload after idle, accelerator caches are explicitly released on unload, and the backend launches at lower CPU priority from `run.sh`.
- Replaced the Next upload proxy's form-data rebuild with direct multipart streaming to the backend to avoid long local proxy stalls and 5-minute client timeouts.
- Changed Apple Silicon `auto` device policy to be selective: fast video can still use MPS, but image jobs no longer auto-pick MPS because that path is unstable on this model stack.
- Tightened the fast video path for local Apple Silicon: audio disabled, sampling reduced to `0.5 Hz`, frames-per-clip reduced to `8`, frame size capped at `384px`, and fast analysis capped to the first `30s` by default.
- Fixed misleading long-job progress behavior so heartbeat progress now stays within stage-specific ceilings instead of drifting to `95%` during extractor-heavy preparation.
- Fixed image-model config merging so `data.image_feature` is now replaced wholesale instead of partially mutating TRIBE's video-shaped default image extractor config.
- Tightened the image extractor override again to use dotted `data.image_feature.=replace=` keys, because the first nested replacement attempt was still being merged by `ConfDict` and left stale video-only keys behind.
- Replaced the unsupported direct-image checkpoint path with a synthetic image-to-video proxy path so static image uploads now run through the checkpoint's supported video flow instead of forcing a pure image extractor into a video-trained config.
- Dropped separate image-mode semantics. Image uploads are now normalized to one-frame video proxies, and the supported live modalities are video, audio, and text.
- Fixed image-proxy runtime length so proxy clips are now clamped to at least one fast-video sampling interval; otherwise TRIBE produced zero sample points and failed with `shape=(0,)`.
- Switched image proxies to `1s` static videos at `8fps` and excluded them from the raw fast-video event shortcut, because the shortcut still produced downstream indexing mismatches for image-backed proxies.
- Fixed WhisperX launch defaults for audio uploads so CPU paths use `int8` instead of the previously hard-coded `float16`.
- Replaced the unsafe merged compare `predict()` pass with separate per-asset scoring, so short clips and image-backed proxies no longer rely on timeline partitioning after inference.
- Fixed audio uploads to stay on the audio-only path by calling TRIBE's event builder with `audio_only=True`, which removes the WhisperX dependency from local audio analysis.
- Restored image result semantics while keeping the video runtime adapter: image uploads now preserve `modality="image"` for UI and device policy, but still run through the supported video checkpoint path.
- Corrected the image proxy duration clamp in code to at least one fast-video sampling interval; the earlier tracker note was ahead of the implementation.
- Fixed the upload UI contract for non-video assets: text files now render a proper ready-for-analysis state, and image result cards report the actual peak point from proxy-derived curves.
- Removed the fast-video `data.duration_trs=40` override after confirming the shipped checkpoint still predicts `100` timesteps; the mixed `40`-window dataloader and `100`-step model output caused the boolean-mask `IndexError` on video uploads.
- Reframed the results layer for non-technical users: the backend now emits a user-facing engagement scorecard, clearer system labels, laterality summary, peak-moment callout, and schematic brain-view metadata on top of the existing TRIBE metrics.
- Reworked the frontend from a technical metrics dashboard into a scorecard-first workspace with overview cards, engagement-systems tables, schematic brain scans, plain-language takeaways, and an advanced disclosure for the original low-level metrics.

## Explicit User Preferences

- Concise, direct, zero-fluff communication.
- Answer first, then compact reasoning.
- Challenge assumptions and flag weak logic.
- Default to realistic, cost-aware MVP choices.
- Avoid over-engineering.
- Tests only when asked, but verification should still be explicit when useful.
- Audit calculations carefully and state uncertainty and edge cases.
- For this product:
- Desktop-first.
- Light mode only.
- Monochrome, dense, high-contrast UI.
- No gradients, shadows, or decorative UI.
- Follow design tokens instead of hardcoded visual values when working inside the product design system.
- Product scope is compare-first video analysis, not strategy advice.
- Output must be numeric, cited, and non-prescriptive.
- For performance work, optimize for the current Apple M3 laptop first, with warm compare latency as the primary target.
- Default to the speed-first path over maximum multimodal fidelity unless explicitly overridden.
- Protect local machine responsiveness over shaving a few seconds off startup when the backend is idle.
- When local logs contradict the optimization plan, follow the logs. A fast path that falls back or times out is slower than a narrower path that actually completes.

## Insights

- The existing backend is intentionally small and job-based, so schema and orchestration changes are cheaper than architectural rewrites.
- Demo reliability matters more than modality breadth. Fixture mode is not optional polish; it is part of the delivery strategy.
- The old UI and docs were still framed around "neural activation" and a single-upload lab. That language conflicts with the current product positioning and had to be removed from the main user path.
- The frontend had pre-existing build issues unrelated to the compare feature: missing `node_modules`, outdated PostCSS wiring for Tailwind v4, and a TypeScript deprecation gate.
- The current launcher should own startup and cleanup. Reusing unknown healthy processes is convenient, but it weakens reproducibility and violates explicit lifecycle ownership.
- The live TRIBE model path is still operationally risky for short hackathon timelines because first-run CPU inference remains slow.
- Tailwind v4 theme tokens must be declared in CSS for this stack. Defining colors only in the legacy JS config left shadcn component utilities unresolved at runtime.
- Non-temporal modalities need a modality-aware metric contract. For static images, only `Peak` is exposed; temporal metrics are suppressed instead of being fabricated.
- Long model loads need visible progress movement even when the underlying inference library does not expose fine-grained callbacks. A lightweight backend heartbeat is enough to avoid the appearance of a frozen job.
- TRIBE's convenience helper for `video_path` is expensive because it triggers audio extraction and text/context preprocessing before scoring. The fast local path must bypass that helper and construct raw `Video` events directly.
- On one local Apple GPU, serialized inference is a better default than trying to run multiple model predictions concurrently.
- Keeping an MPS-backed model resident after work completes is a real system-impact risk on a laptop. Idle eviction is the right default for local development.
- The working Hugging Face Space this user referenced keeps TRIBE extractors on CPU specifically because MPS is brittle for this model path. Our local app should use that as a stability constraint, not ignore it.
- The earlier "fast" video profile was still too heavy because it kept TRIBE's native video extractor near default workload. The real speed win comes from reducing extractor work itself: fewer timepoints, fewer frames, no audio, smaller images, and shorter local analysis windows.
- TRIBE's shipped default config defines `image_feature` as a video-style wrapper around an image model. If we swap that extractor class at runtime, the override must replace the whole subconfig, not just individual keys, or pydantic validation will fail on leftover video-only fields.
- In this config system, nested replacement is more reliable through dotted keys like `data.image_feature.=replace=` than by assigning a whole sub-dict value, because whole-value assignment can still merge into the existing nested object.
- The Hugging Face Space evidence points to the real constraint: this runtime is operationally proven for video, not for a swapped-in standalone image extractor. The practical fix is to adapt images into the supported video path and keep the UI/output semantics image-specific.
- The simpler and more defensible contract is supported runtime modalities only: video, audio, and text. Images are just a convenience input that gets converted into a minimal video stimulus before inference.
- When image proxies are shorter than the active video sampling interval, TRIBE's video extractor produces no timepoints. Proxy duration therefore has to be constrained by the sampler, not just by UI intent.
- Image-backed proxies do better through TRIBE's standard video event builder than through the stripped-down raw fast-video event path. The convenience cost is acceptable here because image uploads are now just a compatibility path.
- Audio failures on this Mac were caused by the subprocess wrapper, not by TRIBE scoring itself. WhisperX defaults must be device-aware, or CPU transcription crashes before inference starts.
- Preserving the original upload modality separately from the runtime modality is necessary for stability. Image inputs need the video checkpoint, but they still need image-specific CPU policy and UI semantics.
- Tracker notes can drift from the code during rapid iteration. For fragile runtime paths like image proxies, verification has to happen against the actual implementation, not the intended change list.
- TRIBE checkpoint shape assumptions matter more than local speed tweaks. Changing loader-side `duration_trs` without a matching checkpoint rebuild breaks `predict()` even when the rest of the pipeline is correct.
- The target user for this product is not a neuroscience operator. Raw `Opening/Middle/Spread/Consistency` labels can stay in the payload for analysts, but the default UX has to translate them into business-readable concepts first.
- Brain views do not need to be photorealistic to be useful. A stable schematic layer driven by predicted systems is safer for this product than binding the UX to optional 3D plotting dependencies that are not installed by default.
- For this product, first-run clarity matters as much as inference correctness. The default screen should teach the three-layer reading order upfront: decision first, explanation second, analyst detail last.
- Progressive disclosure needs to exist at both empty state and results state. Marketing users should not land directly inside timelines, brain views, or raw TRIBE tables before they understand the headline call.
- The saved design brief in `.impeccable.md` is now part of the working context and should anchor future UI changes toward a calm Firecrawl/Vercel/OpenAI-style SaaS presentation.
- The stimulus contract is now explicit in the UI: each side can switch between video, audio, or pasted text. Text should be treated as a first-class input mode in the product, not just a hidden `.txt` file edge case.

## Open Items

- Measure Hour 0 test-retest variance and replace the fallback `7%` threshold if the measurement is reliable.
- Validate the full compare flow end-to-end against live model inference, not just fixture mode and build checks.
- Run the new benchmark harness on the target machine and lock the fastest stable `TRIBEV2_BATCH_SIZE_FAST` value.
- Decide whether audio remains out of scope or becomes a follow-up after the video path is stable.
