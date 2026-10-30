"""Measure a clip and write down what it does.

Everything here runs on a CPU. Geometry and timing come from tracking pixels,
so a recipe's numbers can be checked against a render later.
"""

from __future__ import annotations

import json
import math
import subprocess
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks

from .recipe import (
    CURVE_POINTS,
    CameraLayer,
    LookLayer,
    PacingLayer,
    Recipe,
    SoundLayer,
    SourceInfo,
    TransformLayer,
)

ANALYSIS_WIDTH = 320
ANALYSIS_FPS = 12.0
CUT_THRESHOLD = 0.45
MIN_CAMERA_CONFIDENCE = 0.7
MAX_COMPARABLE_TRAVEL = 0.4
VERIFY_SPAN = 8
VERIFY_MIN_ALIGNMENT = 0.75
AUDIO_RATE = 16000


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True,
        check=True,
        text=True,
    )
    return json.loads(out.stdout)


def read_frames(path: Path) -> tuple[list[np.ndarray], float, tuple[int, int, float]]:
    """Return small BGR frames at about ANALYSIS_FPS, their rate, and the source size."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot read video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, round(fps / ANALYSIS_FPS))
    small_h = max(2, round(height * ANALYSIS_WIDTH / width))
    frames: list[np.ndarray] = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            frames.append(cv2.resize(frame, (ANALYSIS_WIDTH, small_h), interpolation=cv2.INTER_AREA))
        index += 1
    cap.release()
    if len(frames) < 2:
        raise ValueError("clip is too short to analyze")
    return frames, fps / step, (width, height, fps)


def find_shots(frames: list[np.ndarray], rate: float) -> list[tuple[int, int]]:
    """Split on large jumps between color histograms."""
    hists = []
    for frame in frames:
        hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
        hists.append(cv2.normalize(hist, hist).flatten())
    cuts = [0]
    for i in range(1, len(hists)):
        distance = cv2.compareHist(hists[i - 1], hists[i], cv2.HISTCMP_BHATTACHARYYA)
        if distance > CUT_THRESHOLD and i - cuts[-1] >= rate / 2:
            cuts.append(i)
    cuts.append(len(frames))
    return list(pairwise(cuts))


def resample(values: np.ndarray, points: int = CURVE_POINTS) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(points)
    if len(values) == 1:
        return np.full(points, float(values[0]))
    return np.interp(np.linspace(0, 1, points), np.linspace(0, 1, len(values)), values)


def track_motion(frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit global motion between frames.

    Returns steps [dx, dy, dzoom, droll], flow energy, and how far the fitted
    motion can be trusted (see verify_motion).

    RANSAC keeps the background's motion and rejects subjects moving through it.
    Steps are stated from the camera's point of view: pixels sliding left mean
    the camera moved right.
    """
    height, width = frames[0].shape[:2]
    center = np.array([width / 2, height / 2])
    steps = np.zeros((len(frames) - 1, 4))
    energy = np.zeros(len(frames) - 1)
    matrices = [np.eye(2, 3) for _ in range(len(frames) - 1)]
    previous = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for i, frame in enumerate(frames[1:]):
        current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        points = cv2.goodFeaturesToTrack(previous, 300, 0.01, 7)
        if points is not None and len(points) >= 8:
            moved, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None)
            keep = status.flatten() == 1
            source, target = points[keep], moved[keep]
            if len(source) >= 8:
                energy[i] = float(np.median(np.linalg.norm(target - source, axis=-1))) / width
                matrix, _ = cv2.estimateAffinePartial2D(
                    source, target, method=cv2.RANSAC, ransacReprojThreshold=2.0
                )
                if matrix is not None:
                    matrices[i] = matrix
                    a, b = matrix[0, 0], matrix[1, 0]
                    shift = matrix[:, :2] @ center + matrix[:, 2] - center
                    steps[i] = (
                        -shift[0] / width,
                        -shift[1] / width,
                        math.log(max(math.hypot(a, b), 1e-6)),
                        math.atan2(b, a),
                    )
        previous = current
    return steps, energy, verify_motion(frames, matrices)


def verify_motion(frames: list[np.ndarray], matrices: list[np.ndarray]) -> float:
    """Share of windows where the fitted motion really lines the picture up.

    Points on drifting clouds or water agree from one frame to the next, so a
    per-frame fit looks healthy. Chaining the fit over several frames and
    warping the picture exposes it: real camera motion lands, a drifting
    texture has changed shape by then.
    """
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    height, width = grays[0].shape
    box = (height // 4, 3 * height // 4, width // 4, 3 * width // 4)
    checked = landed = 0
    for start in range(0, len(matrices) - VERIFY_SPAN, VERIFY_SPAN // 2):
        chain = np.eye(3)
        for matrix in matrices[start : start + VERIFY_SPAN]:
            chain = np.vstack([matrix, [0, 0, 1]]) @ chain
        if np.abs(chain[:2, 2]).max() < 1 and np.abs(chain[:2, :2] - np.eye(2)).max() < 0.01:
            continue  # nothing moved, so there is nothing to verify
        warped = cv2.warpAffine(grays[start], chain[:2], (width, height))
        a = warped[box[0] : box[1], box[2] : box[3]].astype(np.float32)
        b = grays[start + VERIFY_SPAN][box[0] : box[1], box[2] : box[3]].astype(np.float32)
        checked += 1
        if a.std() > 1 and b.std() > 1 and np.corrcoef(a.ravel(), b.ravel())[0, 1] >= VERIFY_MIN_ALIGNMENT:
            landed += 1
    return landed / checked if checked else 1.0


def classify_camera(steps: np.ndarray) -> CameraLayer:
    path = np.vstack([np.zeros(4), np.cumsum(steps, axis=0)])
    x, y, zoom, roll = path.T
    parts: list[str] = []
    if abs(zoom[-1]) > 0.06:
        parts.append("push_in" if zoom[-1] > 0 else "pull_out")
    if abs(x[-1]) > 0.05:
        parts.append("pan_right" if x[-1] > 0 else "pan_left")
    if abs(y[-1]) > 0.05:
        parts.append("tilt_down" if y[-1] > 0 else "tilt_up")
    if abs(roll[-1]) > math.radians(3):
        parts.append("roll_ccw" if roll[-1] > 0 else "roll_cw")

    jitter = float(np.std(np.diff(steps[:, :2], axis=0))) if len(steps) > 2 else 0.0
    if not parts:
        parts.append("handheld" if jitter > 0.003 else "static")

    dominant = np.abs(np.array([x[-1], y[-1], zoom[-1], roll[-1]])).argmax()
    return CameraLayer(
        move="+".join(parts),
        easing=classify_easing(path[:, dominant]),
        x=resample(x).tolist(),
        y=resample(y).tolist(),
        zoom=resample(zoom).tolist(),
        roll=resample(roll).tolist(),
        jitter=jitter,
    )


def classify_easing(curve: np.ndarray) -> str:
    span = curve[-1] - curve[0]
    if abs(span) < 1e-6:
        return "linear"
    progress = resample((curve - curve[0]) / span)
    t = np.linspace(0, 1, CURVE_POINTS)
    shapes = {
        "linear": t,
        "ease_in": t**2,
        "ease_out": 1 - (1 - t) ** 2,
        "ease_in_out": t * t * (3 - 2 * t),
    }
    return min(shapes, key=lambda name: float(np.abs(progress - shapes[name]).mean()))


def measure_look(frames: list[np.ndarray]) -> LookLayer:
    picks = frames[:: max(1, len(frames) // 8)]
    pixels = np.concatenate(
        [cv2.cvtColor(f.astype(np.float32) / 255, cv2.COLOR_BGR2LAB).reshape(-1, 3) for f in picks]
    )
    mean, std = pixels.mean(axis=0), pixels.std(axis=0)

    sample = np.concatenate([f.reshape(-1, 3) for f in picks]).astype(np.float32)[::37]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
    _, labels, centers = cv2.kmeans(sample, 5, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
    order = np.argsort(-np.bincount(labels.flatten(), minlength=5))
    palette = ["#{:02x}{:02x}{:02x}".format(*centers[i][::-1].astype(int)) for i in order]

    return LookLayer(
        lab_mean=mean.tolist(),
        lab_std=std.tolist(),
        palette=palette,
        saturation=float(np.hypot(pixels[:, 1], pixels[:, 2]).mean()),
        contrast=float(std[0]),
    )


def measure_transform(first: np.ndarray, last: np.ndarray, duration: float) -> TransformLayer:
    """How much the picture's content changes across the shot, 0 to 1."""
    channels, bins, ranges = [0, 1], [24, 16], [0, 180, 0, 256]
    hists = [
        cv2.normalize(cv2.calcHist([cv2.cvtColor(f, cv2.COLOR_BGR2HSV)], channels, None, bins, ranges), None).flatten()
        for f in (first, last)
    ]
    color = cv2.compareHist(hists[0], hists[1], cv2.HISTCMP_BHATTACHARYYA)

    a = cv2.GaussianBlur(cv2.cvtColor(first, cv2.COLOR_BGR2GRAY), (0, 0), 3).astype(np.float32)
    b = cv2.GaussianBlur(cv2.cvtColor(last, cv2.COLOR_BGR2GRAY), (0, 0), 3).astype(np.float32)
    structure = 1 - max(float(cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)[0, 0]), 0.0)

    amount = float(np.clip(0.6 * color + 0.4 * structure, 0, 1))
    return TransformLayer(present=amount > 0.35, amount=amount, end=duration)


def measure_pacing(energy: np.ndarray, duration: float) -> PacingLayer:
    curve = resample(energy)
    peak = curve.max()
    if peak > 0:
        curve = curve / peak
    median = float(np.median(curve))
    ramps: list[tuple[float, float]] = []
    if median > 0:
        fast = curve > max(2 * median, 0.5)
        start = None
        for i, on in enumerate([*fast, False]):
            if on and start is None:
                start = i
            elif not on and start is not None:
                ramps.append((start / CURVE_POINTS * duration, i / CURVE_POINTS * duration))
                start = None
    return PacingLayer(energy=curve.tolist(), ramps=ramps)


def measure_sound(path: Path, has_audio: bool, duration: float) -> SoundLayer:
    if not has_audio:
        return SoundLayer()
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(AUDIO_RATE),
         "-f", "s16le", "-"],
        capture_output=True,
        check=True,
    ).stdout
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
    hop = AUDIO_RATE // 20
    usable = len(samples) // hop * hop
    if usable == 0:
        return SoundLayer()
    rms = np.sqrt((samples[:usable].reshape(-1, hop) ** 2).mean(axis=1))
    if rms.max() < 1e-3:
        return SoundLayer()

    normalized = rms / rms.max()
    peaks, _ = find_peaks(normalized, prominence=0.3, distance=4)
    onsets = [round(p / len(rms) * duration, 2) for p in peaks]
    if not onsets:
        words = "a steady bed of sound"
    elif len(onsets) == 1:
        words = f"a single hit at {onsets[0]:.1f}s"
    else:
        words = f"{len(onsets)} sharp hits, first at {onsets[0]:.1f}s"
    return SoundLayer(
        present=True,
        envelope=resample(normalized).tolist(),
        onsets=onsets,
        description=words,
    )


def analyze(path: str | Path, name: str = "Untitled effect") -> Recipe:
    path = Path(path)
    frames, rate, (width, height, fps) = read_frames(path)
    info = probe(path)
    has_audio = any(s.get("codec_type") == "audio" for s in info["streams"])
    duration = float(info["format"].get("duration", len(frames) / rate))

    shots = find_shots(frames, rate)
    chosen = max(range(len(shots)), key=lambda i: shots[i][1] - shots[i][0])
    start, end = shots[chosen]
    shot_frames = frames[start:end]
    if len(shot_frames) < 2:
        shot_frames = frames
    shot_duration = len(shot_frames) / rate

    steps, energy, confidence = track_motion(shot_frames)
    camera = classify_camera(steps).model_copy(update={"confidence": confidence})
    transform = measure_transform(shot_frames[0], shot_frames[-1], shot_duration)
    unsupported = []
    if max(abs(camera.zoom[-1]), abs(camera.x[-1]), abs(camera.y[-1])) > MAX_COMPARABLE_TRAVEL:
        # The ends of the shot no longer show the same things, so a difference
        # between them says nothing about whether the subject changed.
        transform.present = False
        unsupported.append("the camera travels too far to measure a transform; describe one by hand")
    if confidence < MIN_CAMERA_CONFIDENCE:
        unsupported.append(
            f"camera read from only {confidence:.0%} of the shot; "
            "the picture moves too freely to separate the camera from the scene"
        )
    if len(shots) > 1:
        unsupported.append(f"{len(shots)} shots found; only the longest is in this recipe")

    return Recipe(
        name=name,
        source=SourceInfo(
            duration=duration,
            fps=fps,
            width=width,
            height=height,
            shots=[(a / rate, b / rate) for a, b in shots],
            analyzed_shot=chosen,
        ),
        camera=camera,
        transform=transform,
        look=measure_look(shot_frames),
        pacing=measure_pacing(energy, shot_duration),
        sound=measure_sound(path, has_audio, duration),
        unsupported=unsupported,
    )
