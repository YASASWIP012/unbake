"""Score a render against the recipe it came from, one layer at a time.

The fixer reads these scores and changes only the layer that fell short, so a
good camera move survives a re-render for a bad grade.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel

from .recipe import Recipe

PASS_SCORE = 70.0
LAYER_WEIGHTS = {"camera": 0.4, "transform": 0.25, "look": 0.2, "pacing": 0.1, "sound": 0.05}


class LayerScore(BaseModel):
    score: float
    detail: str


class MatchReport(BaseModel):
    overall: float
    layers: dict[str, LayerScore]
    failing: list[str]


def shape_similarity(a: list[float], b: list[float]) -> float:
    """Pearson correlation mapped to 0..1, with opposite shapes scoring 0."""
    x, y = np.asarray(a), np.asarray(b)
    if len(x) == 0 or len(y) == 0 or x.std() < 1e-9 or y.std() < 1e-9:
        return 0.0
    return float(max(np.corrcoef(x, y)[0, 1], 0.0))


def score_camera(ref: Recipe, out: Recipe) -> LayerScore:
    axes = ["x", "y", "zoom", "roll"]
    thresholds = {"x": 0.02, "y": 0.02, "zoom": 0.03, "roll": math.radians(1.5)}
    parts: list[float] = []
    notes: list[str] = []
    for axis in axes:
        wanted = getattr(ref.camera, axis)
        got = getattr(out.camera, axis)
        travel = abs(wanted[-1]) if wanted else 0.0
        if travel < thresholds[axis]:
            continue
        ratio = abs(got[-1]) / travel if got else 0.0
        magnitude = min(ratio, 1 / ratio) if ratio > 0 else 0.0
        shape = shape_similarity(wanted, got)
        # Multiplied so a right-shaped move at a third of the size can't pass.
        parts.append(100 * shape * math.sqrt(magnitude))
        notes.append(f"{axis}: {ratio:.0%} of the travel")
    if not parts:
        drift = max(abs(v[-1]) for v in (out.camera.x, out.camera.y, out.camera.zoom) if v)
        score = 100 * math.exp(-drift / 0.05)
        return LayerScore(score=score, detail=f"reference is locked off, render drifts {drift:.3f}")
    return LayerScore(score=float(np.mean(parts)), detail=", ".join(notes))


def score_transform(ref: Recipe, out: Recipe) -> LayerScore | None:
    if not ref.transform.present:
        return None
    ratio = out.transform.amount / max(ref.transform.amount, 1e-6)
    score = 100 * min(ratio, 1 / ratio) if ratio > 0 else 0.0
    return LayerScore(score=score, detail=f"{ratio:.0%} of the reference's change")


def score_look(ref: Recipe, out: Recipe) -> LayerScore:
    delta = float(np.linalg.norm(np.subtract(ref.look.lab_mean, out.look.lab_mean)))
    wanted, got = ref.look.contrast, out.look.contrast
    contrast = min(wanted, got) / max(wanted, got) if max(wanted, got) > 0 else 1.0
    return LayerScore(
        score=100 * (0.7 * math.exp(-delta / 25) + 0.3 * contrast),
        detail=f"color distance {delta:.1f}, contrast {contrast:.0%}",
    )


def score_pacing(ref: Recipe, out: Recipe) -> LayerScore:
    similarity = shape_similarity(ref.pacing.energy, out.pacing.energy)
    return LayerScore(score=100 * similarity, detail=f"motion timing correlation {similarity:.2f}")


def score_sound(ref: Recipe, out: Recipe) -> LayerScore | None:
    if not ref.sound.present:
        return None
    similarity = shape_similarity(ref.sound.envelope, out.sound.envelope)
    return LayerScore(score=100 * similarity, detail=f"loudness correlation {similarity:.2f}")


def compare(ref: Recipe, out: Recipe) -> MatchReport:
    candidates = {
        "camera": score_camera(ref, out) if ref.camera.enabled else None,
        "transform": score_transform(ref, out) if ref.transform.enabled else None,
        "look": score_look(ref, out) if ref.look.enabled else None,
        "pacing": score_pacing(ref, out) if ref.pacing.enabled else None,
        "sound": score_sound(ref, out) if ref.sound.enabled else None,
    }
    layers = {name: score for name, score in candidates.items() if score is not None}
    total = sum(LAYER_WEIGHTS[name] for name in layers) or 1.0
    overall = sum(LAYER_WEIGHTS[name] * s.score for name, s in layers.items()) / total
    return MatchReport(
        overall=overall,
        layers=layers,
        failing=[name for name, s in layers.items() if s.score < PASS_SCORE],
    )
