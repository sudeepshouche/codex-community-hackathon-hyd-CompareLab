"""Compact TRIBE result shaping for the compare lab UI."""

from __future__ import annotations

from pathlib import Path
import hashlib
import math
import typing as tp

import numpy as np

DEFAULT_TIE_THRESHOLD_PCT = 7.0
CONSISTENCY_EPSILON = 1e-6
DISPLAY_METRICS = (
    "Opening",
    "Middle",
    "Closing",
    "Peak",
    "Spread",
    "Consistency",
)
DISCLAIMER = (
    "Descriptive TRIBE-based comparison only. Not a predictor of virality, CTR, ROAS, "
    "sales, or measured cognition."
)
SYSTEM_DEFINITIONS = (
    {
        "key": "visual_pull",
        "label": "Visual pull",
        "category": "Visual Processing",
        "start": 0.00,
        "end": 0.15,
        "weight": 0.22,
        "description": "How much the visual layer carries the response.",
        "high_text": "The visual layer is doing real work through motion, framing, contrast, or faces.",
        "low_text": "The visual layer is present, but it is not carrying much of the response yet.",
    },
    {
        "key": "voice_meaning",
        "label": "Voice & meaning",
        "category": "Auditory & Language",
        "start": 0.15,
        "end": 0.35,
        "weight": 0.21,
        "description": "How strongly words, narration, or audio cues are landing.",
        "high_text": "Speech, wording, or audio cues are contributing clear signal.",
        "low_text": "The response is not being driven much by words, narration, or sound.",
    },
    {
        "key": "focus_retention",
        "label": "Focus retention",
        "category": "Attention & Spatial",
        "start": 0.35,
        "end": 0.55,
        "weight": 0.22,
        "description": "How well the creative keeps attention organized and on-track.",
        "high_text": "The piece keeps attention locked and spatially organized.",
        "low_text": "Attention is active, but the hold is lighter and easier to lose.",
    },
    {
        "key": "thinking_load",
        "label": "Thinking load",
        "category": "Executive & Motor",
        "start": 0.55,
        "end": 0.80,
        "weight": 0.17,
        "description": "How much the content triggers active thinking and mental effort.",
        "high_text": "The viewer is doing more active processing rather than passively consuming.",
        "low_text": "The content is being taken in more passively than analytically.",
    },
    {
        "key": "emotion_intent",
        "label": "Emotion & intent",
        "category": "Emotion & Decision",
        "start": 0.80,
        "end": 1.00,
        "weight": 0.18,
        "description": "How much the content taps emotion, reward, and action readiness.",
        "high_text": "The response has more emotional and action-oriented weight.",
        "low_text": "The response is lighter on emotion and action-readiness than the stronger systems.",
    },
)
FRIENDLY_METRIC_LABELS = {
    "Opening": {
        "label": "Hook strength",
        "description": "How strong the first impression is.",
    },
    "Middle": {
        "label": "Mid-sequence hold",
        "description": "How well the response holds once the piece settles in.",
    },
    "Closing": {
        "label": "Finish strength",
        "description": "How much signal is still there by the end.",
    },
    "Peak": {
        "label": "Best moment",
        "description": "The strongest response point in the run.",
    },
    "Spread": {
        "label": "Variation",
        "description": "How much the response rises and falls across the run.",
    },
    "Consistency": {
        "label": "Steadiness",
        "description": "How even the response feels from moment to moment.",
    },
}
BRAIN_VIEW_TEMPLATES = (
    {
        "key": "left",
        "label": "Left hemisphere",
        "description": "Language, sequencing, and analytical load.",
        "hotspots": (
            {"system_key": "voice_meaning", "label": "Voice & meaning", "x": 38, "y": 34, "r": 14},
            {"system_key": "thinking_load", "label": "Thinking load", "x": 57, "y": 22, "r": 12},
            {"system_key": "focus_retention", "label": "Focus retention", "x": 54, "y": 48, "r": 10},
        ),
    },
    {
        "key": "right",
        "label": "Right hemisphere",
        "description": "Emotion, imagery, and spatial awareness.",
        "hotspots": (
            {"system_key": "visual_pull", "label": "Visual pull", "x": 37, "y": 28, "r": 14},
            {"system_key": "emotion_intent", "label": "Emotion & intent", "x": 57, "y": 24, "r": 12},
            {"system_key": "focus_retention", "label": "Focus retention", "x": 53, "y": 47, "r": 10},
        ),
    },
    {
        "key": "dorsal",
        "label": "Top-down view",
        "description": "Focus, pacing, and coordination.",
        "hotspots": (
            {"system_key": "focus_retention", "label": "Focus retention", "x": 50, "y": 22, "r": 14},
            {"system_key": "thinking_load", "label": "Thinking load", "x": 34, "y": 42, "r": 12},
            {"system_key": "visual_pull", "label": "Visual pull", "x": 66, "y": 42, "r": 12},
        ),
    },
    {
        "key": "anterior",
        "label": "Front view",
        "description": "Decision, reward, and social appraisal.",
        "hotspots": (
            {"system_key": "emotion_intent", "label": "Emotion & intent", "x": 50, "y": 20, "r": 14},
            {"system_key": "voice_meaning", "label": "Voice & meaning", "x": 36, "y": 44, "r": 11},
            {"system_key": "thinking_load", "label": "Thinking load", "x": 64, "y": 44, "r": 11},
        ),
    },
)


def _safe_float(value: tp.Any) -> float:
    return float(np.asarray(value).item())


def _clamp_score(value: float) -> int:
    return int(round(min(100.0, max(0.0, value))))


def _grade(score: float) -> str:
    if score >= 90:
        return "A+"
    if score >= 85:
        return "A"
    if score >= 80:
        return "A-"
    if score >= 75:
        return "B+"
    if score >= 70:
        return "B"
    if score >= 65:
        return "B-"
    if score >= 60:
        return "C+"
    if score >= 55:
        return "C"
    if score >= 50:
        return "C-"
    if score >= 40:
        return "D"
    return "F"


def _score_band(score: int) -> str:
    if score >= 82:
        return "Very strong"
    if score >= 68:
        return "Strong"
    if score >= 54:
        return "Moderate"
    if score >= 40:
        return "Light"
    return "Weak"


def _score_band_summary(score: int) -> str:
    band = _score_band(score)
    if band == "Very strong":
        return "The response is clearly above the rest of the profile."
    if band == "Strong":
        return "This system is contributing meaningful signal."
    if band == "Moderate":
        return "This system is present, but it is not the main driver."
    if band == "Light":
        return "This system is showing only a lighter signal."
    return "This system is the least active part of the profile."


def _paired_vertex_indices(
    start_ratio: float,
    end_ratio: float,
    *,
    n_vertices: int,
) -> np.ndarray:
    if n_vertices <= 0:
        return np.asarray([], dtype=int)
    half = max(n_vertices // 2, 1)
    left_start = min(int(half * start_ratio), half)
    left_end = max(left_start + 1, min(int(math.ceil(half * end_ratio)), half))
    left = np.arange(left_start, left_end, dtype=int)
    right = np.arange(half + left_start, min(half + left_end, n_vertices), dtype=int)
    if right.size == 0:
        return left
    return np.concatenate([left, right])


def _system_score_rows(preds: np.ndarray) -> tuple[list[dict[str, tp.Any]], dict[str, tp.Any]]:
    avg_activation = np.mean(np.abs(preds), axis=0)
    global_mean = float(avg_activation.mean()) if avg_activation.size else 0.0
    n_vertices = int(avg_activation.size)
    half = max(n_vertices // 2, 1)
    left_activation = float(avg_activation[:half].mean()) if half else 0.0
    right_activation = float(avg_activation[half:].mean()) if n_vertices > half else left_activation
    peaks: list[float] = []
    activations: list[float] = []
    rows: list[dict[str, tp.Any]] = []
    for definition in SYSTEM_DEFINITIONS:
        indices = _paired_vertex_indices(
            definition["start"],
            definition["end"],
            n_vertices=n_vertices,
        )
        values = avg_activation[indices] if indices.size else np.asarray([0.0], dtype=float)
        activation = float(values.mean())
        peak = float(values.max())
        peaks.append(peak)
        activations.append(activation)
        rows.append(
            {
                "key": definition["key"],
                "label": definition["label"],
                "category": definition["category"],
                "description": definition["description"],
                "activation": activation,
                "peak": peak,
                "weight": definition["weight"],
                "high_text": definition["high_text"],
                "low_text": definition["low_text"],
            }
        )

    max_peak = max(peaks) if peaks else 0.0
    for row in rows:
        normalized_mean = (row["activation"] / max(global_mean, CONSISTENCY_EPSILON)) * 50.0
        normalized_peak = (row["peak"] / max(max_peak, CONSISTENCY_EPSILON)) * 100.0
        score = _clamp_score(normalized_mean * 0.4 + normalized_peak * 0.6)
        row["score"] = score
        row["grade"] = _grade(score)
        row["band"] = _score_band(score)
        row["readout"] = row["high_text"] if score >= 60 else row["low_text"]

    laterality_index = (left_activation - right_activation) / (
        abs(left_activation) + abs(right_activation) + CONSISTENCY_EPSILON
    )
    total_lateral = max(left_activation + right_activation, CONSISTENCY_EPSILON)
    left_share = round((left_activation / total_lateral) * 100.0, 1)
    right_share = round((right_activation / total_lateral) * 100.0, 1)
    if laterality_index > 0.05:
        laterality_label = "Left-leaning"
        laterality_text = "More of the response leans toward language and analytical processing."
    elif laterality_index < -0.05:
        laterality_label = "Right-leaning"
        laterality_text = "More of the response leans toward imagery, emotion, and spatial processing."
    else:
        laterality_label = "Balanced"
        laterality_text = "The response is balanced between analytical and emotional/spatial processing."

    return rows, {
        "left_share_pct": left_share,
        "right_share_pct": right_share,
        "laterality_index": round(laterality_index, 3),
        "label": laterality_label,
        "text": laterality_text,
    }


def _build_brain_views(
    systems: list[dict[str, tp.Any]],
    *,
    laterality_index: float,
) -> list[dict[str, tp.Any]]:
    system_scores = {system["key"]: float(system["score"]) for system in systems}
    left_bias = 1.0 + max(laterality_index, 0.0) * 1.2
    right_bias = 1.0 + max(-laterality_index, 0.0) * 1.2
    views = []
    for template in BRAIN_VIEW_TEMPLATES:
        hotspots = []
        for hotspot in template["hotspots"]:
            system_score = system_scores.get(hotspot["system_key"], 50.0)
            bias = 1.0
            if template["key"] == "left":
                bias = left_bias
            elif template["key"] == "right":
                bias = right_bias
            intensity = min(1.0, max(0.12, (system_score / 100.0) * bias))
            hotspots.append(
                {
                    "label": hotspot["label"],
                    "system_key": hotspot["system_key"],
                    "x": hotspot["x"],
                    "y": hotspot["y"],
                    "r": hotspot["r"],
                    "intensity": round(float(intensity), 3),
                }
            )
        strongest_hotspot = max(hotspots, key=lambda item: item["intensity"])
        strongest_system = next(
            system for system in systems if system["key"] == strongest_hotspot["system_key"]
        )
        views.append(
            {
                "key": template["key"],
                "label": template["label"],
                "description": template["description"],
                "annotation": f"Most visible signal: {strongest_system['label']}",
                "hotspots": hotspots,
            }
        )
    return views


def _build_scorecard(
    *,
    preds: np.ndarray,
    curve: list[dict[str, tp.Any]],
    structural_metrics: dict[str, float],
    modality: str,
) -> dict[str, tp.Any]:
    systems, laterality = _system_score_rows(preds)
    weighted_total = sum(system["score"] * float(system["weight"]) for system in systems)
    overall_score = _clamp_score(weighted_total)
    dominant_system = max(systems, key=lambda system: system["score"])
    weakest_system = min(systems, key=lambda system: system["score"])
    peak_point = max(curve, key=lambda point: point["score"])
    hook_delta_pct = (
        round(_relative_pct(structural_metrics["Opening"], structural_metrics["Middle"]), 1)
        if "Opening" in structural_metrics and "Middle" in structural_metrics
        else None
    )
    finish_delta_pct = (
        round(_relative_pct(structural_metrics["Closing"], structural_metrics["Middle"]), 1)
        if "Closing" in structural_metrics and "Middle" in structural_metrics
        else None
    )
    if modality == "image":
        shape_text = "Static asset. The readout reflects one short proxy window rather than a time-evolving sequence."
    elif hook_delta_pct is not None and finish_delta_pct is not None:
        hook_text = "opens above its mid-run level" if hook_delta_pct >= 0 else "opens below its mid-run level"
        finish_text = (
            "finishes above its mid-run level" if finish_delta_pct >= 0 else "finishes below its mid-run level"
        )
        shape_text = (
            f"It {hook_text} ({abs(hook_delta_pct):.1f}%) and {finish_text} ({abs(finish_delta_pct):.1f}%)."
        )
    else:
        shape_text = "Sequence-level shape is limited for this modality."

    return {
        "overall_score": overall_score,
        "overall_grade": _grade(overall_score),
        "overall_band": _score_band(overall_score),
        "headline": f"{_score_band(overall_score)} {dominant_system['label'].lower()} profile",
        "summary": (
            f"Strongest signal is {dominant_system['label']} ({dominant_system['score']}/100). "
            f"Weakest signal is {weakest_system['label']} ({weakest_system['score']}/100)."
        ),
        "shape_summary": shape_text,
        "peak_moment": {
            "label": "Best moment",
            "at_s": round(float(peak_point["midpoint_s"]), 3),
            "score": round(float(peak_point["score"]), 6),
        },
        "dominant_system": {
            "key": dominant_system["key"],
            "label": dominant_system["label"],
            "score": dominant_system["score"],
        },
        "weakest_system": {
            "key": weakest_system["key"],
            "label": weakest_system["label"],
            "score": weakest_system["score"],
        },
        "systems": [
            {
                "key": system["key"],
                "label": system["label"],
                "score": system["score"],
                "grade": system["grade"],
                "band": system["band"],
                "description": system["description"],
                "readout": system["readout"],
            }
            for system in systems
        ],
        "laterality": laterality,
        "brain_views": _build_brain_views(
            systems,
            laterality_index=float(laterality["laterality_index"]),
        ),
        "friendly_metrics": [
            {
                "key": name.lower(),
                "metric": name,
                "label": FRIENDLY_METRIC_LABELS[name]["label"],
                "description": FRIENDLY_METRIC_LABELS[name]["description"],
                "value": round(float(structural_metrics[name]), 6),
            }
            for name in DISPLAY_METRICS
            if name in structural_metrics
        ],
    }


def _build_fixture_scorecard(
    *,
    asset_name: str,
    label: str,
    modality: str,
    curve: list[dict[str, tp.Any]],
    structural_metrics: dict[str, float],
) -> dict[str, tp.Any]:
    digest = hashlib.sha256(f"{label}:{asset_name}:{modality}:scorecard".encode("utf-8")).digest()
    systems = []
    for index, definition in enumerate(SYSTEM_DEFINITIONS):
        score = 40 + digest[index] % 45
        systems.append(
            {
                "key": definition["key"],
                "label": definition["label"],
                "score": score,
                "grade": _grade(score),
                "band": _score_band(score),
                "description": definition["description"],
                "readout": definition["high_text"] if score >= 60 else definition["low_text"],
            }
        )
    overall_score = _clamp_score(
        sum(system["score"] * float(definition["weight"]) for system, definition in zip(systems, SYSTEM_DEFINITIONS))
    )
    dominant_system = max(systems, key=lambda system: system["score"])
    weakest_system = min(systems, key=lambda system: system["score"])
    peak_point = max(curve, key=lambda point: point["score"])
    laterality_index = round(((digest[7] / 255.0) - 0.5) * 0.24, 3)
    laterality = {
        "left_share_pct": round(50 + laterality_index * 100, 1),
        "right_share_pct": round(50 - laterality_index * 100, 1),
        "laterality_index": laterality_index,
        "label": "Balanced" if abs(laterality_index) <= 0.05 else ("Left-leaning" if laterality_index > 0 else "Right-leaning"),
        "text": "Fixture summary for development builds.",
    }
    return {
        "overall_score": overall_score,
        "overall_grade": _grade(overall_score),
        "overall_band": _score_band(overall_score),
        "headline": f"{_score_band(overall_score)} {dominant_system['label'].lower()} profile",
        "summary": (
            f"Strongest signal is {dominant_system['label']} ({dominant_system['score']}/100). "
            f"Weakest signal is {weakest_system['label']} ({weakest_system['score']}/100)."
        ),
        "shape_summary": "Fixture summary for development builds.",
        "peak_moment": {
            "label": "Best moment",
            "at_s": round(float(peak_point["midpoint_s"]), 3),
            "score": round(float(peak_point["score"]), 6),
        },
        "dominant_system": {
            "key": dominant_system["key"],
            "label": dominant_system["label"],
            "score": dominant_system["score"],
        },
        "weakest_system": {
            "key": weakest_system["key"],
            "label": weakest_system["label"],
            "score": weakest_system["score"],
        },
        "systems": systems,
        "laterality": laterality,
        "brain_views": _build_brain_views(systems, laterality_index=float(laterality_index)),
        "friendly_metrics": [
            {
                "key": name.lower(),
                "metric": name,
                "label": FRIENDLY_METRIC_LABELS[name]["label"],
                "description": FRIENDLY_METRIC_LABELS[name]["description"],
                "value": round(float(structural_metrics[name]), 6),
            }
            for name in DISPLAY_METRICS
            if name in structural_metrics
        ],
    }


def build_stimulus_analysis(
    *,
    preds: np.ndarray,
    segments: list[tp.Any],
    asset_path: Path,
    asset_name: str,
    model_id: str,
    device: str,
    label: str,
    modality: str,
    runtime_diagnostics: dict[str, tp.Any] | None = None,
) -> dict[str, tp.Any]:
    """Convert raw predictions into one stimulus payload for the UI."""
    if preds.ndim != 2:
        raise ValueError(f"Expected a 2D prediction array, got shape {preds.shape}")
    if len(preds) != len(segments):
        raise ValueError(
            "Prediction rows must match the number of segments "
            f"({len(preds)} != {len(segments)})"
        )
    if len(preds) == 0:
        raise ValueError("No response segments were produced for the uploaded asset.")

    response_level = np.abs(preds).mean(axis=1)
    curve = [
        _curve_point(index=index, segment=segment, score=float(score))
        for index, (segment, score) in enumerate(zip(segments, response_level))
    ]
    structural_metrics, supported_metrics = _metrics_for_modality(modality, curve)
    duration_s = _video_duration(segments) if modality == "video" else None

    diagnostics = {
        "model_id": model_id,
        "device": device,
        "prediction_shape": [int(dim) for dim in preds.shape],
        "source": "tribev2",
        "segments_kept": len(curve),
        "response_definition": "mean absolute activation across vertices",
    }
    if runtime_diagnostics:
        diagnostics.update(runtime_diagnostics)

    return {
        "label": label,
        "modality": modality,
        "asset": {
            "name": asset_name,
            "path": asset_path.name,
            "size_bytes": asset_path.stat().st_size,
            "duration_s": round(duration_s, 3) if duration_s is not None else None,
        },
        "curve": curve,
        "structural_metrics": structural_metrics,
        "supported_metrics": supported_metrics,
        "scorecard": _build_scorecard(
            preds=preds,
            curve=curve,
            structural_metrics=structural_metrics,
            modality=modality,
        ),
        "diagnostics": diagnostics,
    }


def summarize_single(
    stimulus: dict[str, tp.Any],
    *,
    tie_threshold_pct: float = DEFAULT_TIE_THRESHOLD_PCT,
    threshold_basis: str = "Fallback default pending test-retest measurement.",
    analysis_diagnostics: dict[str, tp.Any] | None = None,
) -> dict[str, tp.Any]:
    """Return the unified result schema for a single stimulus."""
    observations = _single_observations(
        scorecard=stimulus["scorecard"],
        metrics=stimulus["structural_metrics"],
        modality=stimulus["modality"],
        supported_metrics=stimulus["supported_metrics"],
    )
    diagnostics = {
        "tie_threshold_pct": round(tie_threshold_pct, 3),
        "tie_threshold_basis": threshold_basis,
    }
    if analysis_diagnostics:
        diagnostics.update(analysis_diagnostics)

    return {
        "mode": "single",
        "stimulus_a": stimulus,
        "stimulus_b": None,
        "comparison": None,
        "observations": observations,
        "diagnostics": diagnostics,
        "disclaimer": DISCLAIMER,
    }


def summarize_comparison(
    stimulus_a: dict[str, tp.Any],
    stimulus_b: dict[str, tp.Any],
    *,
    tie_threshold_pct: float = DEFAULT_TIE_THRESHOLD_PCT,
    threshold_basis: str = "Fallback default pending test-retest measurement.",
    analysis_diagnostics: dict[str, tp.Any] | None = None,
) -> dict[str, tp.Any]:
    """Return the unified compare schema for two stimuli."""
    comparison = _compare_metrics(
        stimulus_a["structural_metrics"],
        stimulus_b["structural_metrics"],
        supported_metrics_a=stimulus_a["supported_metrics"],
        supported_metrics_b=stimulus_b["supported_metrics"],
        threshold_pct=tie_threshold_pct,
    )
    comparison["engagement"] = _compare_engagement_profiles(
        stimulus_a["scorecard"],
        stimulus_b["scorecard"],
        threshold_pct=tie_threshold_pct,
    )
    diagnostics = {
        "tie_threshold_pct": round(tie_threshold_pct, 3),
        "tie_threshold_basis": threshold_basis,
    }
    if analysis_diagnostics:
        diagnostics.update(analysis_diagnostics)

    return {
        "mode": "compare",
        "stimulus_a": stimulus_a,
        "stimulus_b": stimulus_b,
        "comparison": comparison,
        "observations": _comparison_observations(
            comparison["metrics"],
            engagement=comparison["engagement"],
        ),
        "diagnostics": diagnostics,
        "disclaimer": DISCLAIMER,
    }


def build_fixture_result(
    *,
    asset_name_a: str,
    asset_name_b: str | None = None,
    modality: str = "video",
    tie_threshold_pct: float = DEFAULT_TIE_THRESHOLD_PCT,
    threshold_basis: str = "Fallback default pending test-retest measurement.",
    analysis_diagnostics: dict[str, tp.Any] | None = None,
) -> dict[str, tp.Any]:
    """Create a deterministic demo payload without model inference."""
    stimulus_a = _fixture_stimulus(asset_name_a, label="A", modality=modality)
    if not asset_name_b:
        return summarize_single(
            stimulus_a,
            tie_threshold_pct=tie_threshold_pct,
            threshold_basis=threshold_basis,
            analysis_diagnostics=analysis_diagnostics,
        )

    stimulus_b = _fixture_stimulus(asset_name_b, label="B", modality=modality)
    return summarize_comparison(
        stimulus_a,
        stimulus_b,
        tie_threshold_pct=tie_threshold_pct,
        threshold_basis=threshold_basis,
        analysis_diagnostics=analysis_diagnostics,
    )


def _fixture_stimulus(asset_name: str, *, label: str, modality: str) -> dict[str, tp.Any]:
    digest = hashlib.sha256(f"{label}:{asset_name}:{modality}".encode("utf-8")).digest()
    n_points = 18 if modality == "video" else 1
    duration_s = 18.0 if modality == "video" else None
    points: list[dict[str, tp.Any]] = []
    for index in range(n_points):
        start_s = index * ((duration_s or 1.0) / n_points)
        midpoint_s = start_s + ((duration_s or 1.0) / n_points) / 2
        baseline = 0.22 + (digest[index % len(digest)] / 255) * 0.18
        wave = 0.08 * math.sin((index + 1) * 0.55 + digest[0] / 32)
        slope = 0.05 * (index / max(n_points - 1, 1))
        score = max(0.02, baseline + wave + slope)
        points.append(
            {
                "index": index,
                "start_s": round(start_s, 3),
                "midpoint_s": round(midpoint_s, 3),
                "duration_s": round(((duration_s or 1.0) / n_points), 3),
                "score": round(score, 6),
            }
        )

    metrics, supported_metrics = _metrics_for_modality(modality, points)

    return {
        "label": label,
        "modality": modality,
        "asset": {
            "name": asset_name,
            "path": asset_name,
            "size_bytes": 0,
            "duration_s": duration_s,
        },
        "curve": points,
        "structural_metrics": metrics,
        "supported_metrics": supported_metrics,
        "scorecard": _build_fixture_scorecard(
            asset_name=asset_name,
            label=label,
            modality=modality,
            curve=points,
            structural_metrics=metrics,
        ),
        "diagnostics": {
            "model_id": "fixture",
            "device": "fixture",
            "prediction_shape": [n_points, 0],
            "source": "fixture",
            "segments_kept": n_points,
            "response_definition": "demo fixture",
        },
    }


def _curve_point(index: int, segment: tp.Any, score: float) -> dict[str, tp.Any]:
    start = _safe_float(getattr(segment, "start", 0.0))
    duration = _safe_float(getattr(segment, "duration", 0.0))
    midpoint = start + duration / 2.0
    return {
        "index": index,
        "start_s": round(start, 3),
        "midpoint_s": round(midpoint, 3),
        "duration_s": round(duration, 3),
        "score": round(score, 6),
    }


def _video_duration(segments: list[tp.Any]) -> float:
    if not segments:
        return 0.0
    return max(
        _safe_float(getattr(segment, "start", 0.0))
        + _safe_float(getattr(segment, "duration", 0.0))
        for segment in segments
    )


def _metrics_for_modality(
    modality: str,
    curve: list[dict[str, tp.Any]],
) -> tuple[dict[str, float], list[str]]:
    if modality == "image":
        peak = round(float(max(point["score"] for point in curve)), 6)
        return {"Peak": peak}, ["Peak"]
    return _structural_metrics(curve), list(DISPLAY_METRICS)


def _structural_metrics(curve: list[dict[str, tp.Any]]) -> dict[str, float]:
    scores = np.asarray([point["score"] for point in curve], dtype=float)
    if scores.size == 0:
        raise ValueError("Cannot compute structural metrics from an empty curve.")

    opening_slice, middle_slice, closing_slice = _segment_slices(scores)
    mean_score = float(scores.mean())
    std_score = float(scores.std(ddof=0))

    return {
        "Opening": round(float(scores[opening_slice].mean()), 6),
        "Middle": round(float(scores[middle_slice].mean()), 6),
        "Closing": round(float(scores[closing_slice].mean()), 6),
        "Peak": round(float(scores.max()), 6),
        "Spread": round(float(scores.max() - scores.min()), 6),
        "Consistency": round(mean_score / max(std_score, CONSISTENCY_EPSILON), 6),
    }


def _segment_slices(scores: np.ndarray) -> tuple[slice, slice, slice]:
    count = int(scores.size)
    edge = max(1, int(math.ceil(count * 0.2)))
    opening_end = min(edge, count)
    closing_start = max(count - edge, 0)
    if closing_start <= opening_end:
        middle_start = opening_end
        middle_end = max(opening_end + 1, closing_start)
    else:
        middle_start = opening_end
        middle_end = closing_start
    if middle_start >= middle_end:
        middle_start = min(max(count // 2, 0), max(count - 1, 0))
        middle_end = min(middle_start + 1, count)

    return slice(0, opening_end), slice(middle_start, middle_end), slice(closing_start, count)


def _compare_metrics(
    metrics_a: dict[str, float],
    metrics_b: dict[str, float],
    *,
    supported_metrics_a: list[str],
    supported_metrics_b: list[str],
    threshold_pct: float,
) -> dict[str, tp.Any]:
    metric_rows = []
    a_wins = 0
    b_wins = 0
    ties = 0
    metric_names = [name for name in DISPLAY_METRICS if name in supported_metrics_a and name in supported_metrics_b]
    if not metric_names:
        raise ValueError("No shared metrics available for comparison.")

    for name in metric_names:
        a_value = float(metrics_a[name])
        b_value = float(metrics_b[name])
        delta_pct = _relative_to_b_pct(a_value, b_value)
        gap_pct = _symmetric_gap_pct(a_value, b_value)
        if gap_pct <= threshold_pct:
            winner = "tie"
            ties += 1
        elif a_value > b_value:
            winner = "A"
            a_wins += 1
        else:
            winner = "B"
            b_wins += 1

        metric_rows.append(
            {
                "name": name,
                "label": FRIENDLY_METRIC_LABELS[name]["label"],
                "description": FRIENDLY_METRIC_LABELS[name]["description"],
                "a": round(a_value, 6),
                "b": round(b_value, 6),
                "delta_pct": round(delta_pct, 1),
                "gap_pct": round(gap_pct, 1),
                "winner": winner,
            }
        )

    global_mean_a = round(sum(float(metrics_a[name]) for name in metric_names) / len(metric_names), 6)
    global_mean_b = round(sum(float(metrics_b[name]) for name in metric_names) / len(metric_names), 6)

    return {
        "metrics": metric_rows,
        "summary": {
            "a_wins": a_wins,
            "b_wins": b_wins,
            "ties": ties,
            "summary_line": f"A leads on {a_wins} shape metrics, B on {b_wins}, {ties} are tied.",
            "headline_basis": (
                "Per-metric counts. If needed, break ties with global mean activation."
            ),
            "global_mean_activation": {
                "a": global_mean_a,
                "b": global_mean_b,
            },
        },
    }


def _compare_engagement_profiles(
    scorecard_a: dict[str, tp.Any],
    scorecard_b: dict[str, tp.Any],
    *,
    threshold_pct: float,
) -> dict[str, tp.Any]:
    overall_a = float(scorecard_a["overall_score"])
    overall_b = float(scorecard_b["overall_score"])
    overall_gap_pct = _symmetric_gap_pct(overall_a, overall_b)
    if overall_gap_pct <= threshold_pct:
        overall_winner = "tie"
        headline = (
            f"Both creatives land in the same overall range ({int(round(overall_a))} vs {int(round(overall_b))})."
        )
    elif overall_a > overall_b:
        overall_winner = "A"
        headline = f"A leads overall ({int(round(overall_a))} vs {int(round(overall_b))})."
    else:
        overall_winner = "B"
        headline = f"B leads overall ({int(round(overall_b))} vs {int(round(overall_a))})."

    systems_a = {item["key"]: item for item in scorecard_a["systems"]}
    systems_b = {item["key"]: item for item in scorecard_b["systems"]}
    system_rows = []
    for definition in SYSTEM_DEFINITIONS:
        key = definition["key"]
        system_a = systems_a[key]
        system_b = systems_b[key]
        a_score = float(system_a["score"])
        b_score = float(system_b["score"])
        gap_pct = _symmetric_gap_pct(a_score, b_score)
        if gap_pct <= threshold_pct:
            winner = "tie"
        elif a_score > b_score:
            winner = "A"
        else:
            winner = "B"
        system_rows.append(
            {
                "key": key,
                "label": definition["label"],
                "description": definition["description"],
                "a": int(round(a_score)),
                "b": int(round(b_score)),
                "winner": winner,
                "gap_pct": round(gap_pct, 1),
                "delta_pct": round(_relative_to_b_pct(a_score, b_score), 1),
            }
        )

    ranked_edges = [row for row in system_rows if row["winner"] != "tie"]
    ranked_edges.sort(key=lambda row: row["gap_pct"], reverse=True)
    if ranked_edges:
        top_edge = ranked_edges[0]
        if top_edge["winner"] == "A":
            secondary = f"A's clearest edge is {top_edge['label'].lower()}."
        else:
            secondary = f"B's clearest edge is {top_edge['label'].lower()}."
    else:
        secondary = "No single engagement system breaks away from the other."

    return {
        "overall": {
            "a": int(round(overall_a)),
            "b": int(round(overall_b)),
            "winner": overall_winner,
            "gap_pct": round(overall_gap_pct, 1),
            "headline": headline,
            "secondary": secondary,
        },
        "systems": system_rows,
    }


def _comparison_observations(
    metrics: list[dict[str, tp.Any]],
    *,
    engagement: dict[str, tp.Any],
) -> list[str]:
    observations = []
    overall = engagement["overall"]
    if overall["winner"] == "tie":
        observations.append(
            f"Overall, the two creatives land in the same range ({overall['a']} vs {overall['b']})."
        )
    elif overall["winner"] == "A":
        observations.append(f"A leads the overall engagement scorecard, {overall['a']} vs {overall['b']}.")
    else:
        observations.append(f"B leads the overall engagement scorecard, {overall['b']} vs {overall['a']}.")

    lead_systems = [row for row in engagement["systems"] if row["winner"] != "tie"]
    if lead_systems:
        lead_systems.sort(key=lambda row: row["gap_pct"], reverse=True)
        top_system = lead_systems[0]
        if top_system["winner"] == "A":
            observations.append(
                f"A's strongest edge is {top_system['label']} ({top_system['a']} vs {top_system['b']})."
            )
        else:
            observations.append(
                f"B's strongest edge is {top_system['label']} ({top_system['b']} vs {top_system['a']})."
            )

    tied_systems = [row for row in engagement["systems"] if row["winner"] == "tie"]
    if tied_systems:
        closest = min(tied_systems, key=lambda row: row["gap_pct"])
        observations.append(
            f"Both creatives are effectively tied on {closest['label']} (gap {closest['gap_pct']:.1f}%)."
        )

    if metrics:
        peak_row = next((metric for metric in metrics if metric["name"] == "Peak"), None)
        if peak_row is not None and peak_row["winner"] != "tie":
            if peak_row["winner"] == "A":
                observations.append(
                    f"A owns the single strongest moment, with Peak {peak_row['a']:.3f} vs {peak_row['b']:.3f}."
                )
            else:
                observations.append(
                    f"B owns the single strongest moment, with Peak {peak_row['b']:.3f} vs {peak_row['a']:.3f}."
                )
    return observations[:4]


def _single_observations(
    *,
    scorecard: dict[str, tp.Any],
    metrics: dict[str, float],
    modality: str,
    supported_metrics: list[str],
) -> list[str]:
    observations = [
        f"Overall engagement score is {scorecard['overall_score']}/100 ({scorecard['overall_band'].lower()}).",
        (
            f"Strongest signal is {scorecard['dominant_system']['label']} "
            f"({scorecard['dominant_system']['score']}/100)."
        ),
        (
            f"Weakest signal is {scorecard['weakest_system']['label']} "
            f"({scorecard['weakest_system']['score']}/100)."
        ),
    ]
    if modality == "image" or supported_metrics == ["Peak"]:
        observations.append(f"Best moment score is {metrics['Peak']:.3f}.")
        return observations

    observations.append(
        f"Best moment lands at {scorecard['peak_moment']['at_s']:.1f}s with score {scorecard['peak_moment']['score']:.3f}."
    )
    if "Opening" in metrics and "Middle" in metrics and "Closing" in metrics:
        opening_vs_middle = _relative_pct(metrics["Opening"], metrics["Middle"])
        closing_vs_middle = _relative_pct(metrics["Closing"], metrics["Middle"])
        observations.append(
            f"The piece opens {abs(opening_vs_middle):.1f}% "
            f"{'above' if opening_vs_middle >= 0 else 'below'} its mid-sequence level and finishes "
            f"{abs(closing_vs_middle):.1f}% {'above' if closing_vs_middle >= 0 else 'below'} it."
        )
    return observations[:5]


def _relative_to_b_pct(a_value: float, b_value: float) -> float:
    if abs(b_value) <= CONSISTENCY_EPSILON:
        return 0.0 if abs(a_value) <= CONSISTENCY_EPSILON else 100.0
    return ((a_value - b_value) / abs(b_value)) * 100.0


def _relative_pct(value: float, baseline: float) -> float:
    if abs(baseline) <= CONSISTENCY_EPSILON:
        return 0.0 if abs(value) <= CONSISTENCY_EPSILON else 100.0
    return ((value - baseline) / abs(baseline)) * 100.0


def _symmetric_gap_pct(a_value: float, b_value: float) -> float:
    denominator = max((abs(a_value) + abs(b_value)) / 2.0, CONSISTENCY_EPSILON)
    return abs(a_value - b_value) / denominator * 100.0
