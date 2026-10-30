"""Turn a recipe's camera curves into a guide video.

The guide is the user's own photo moved along the recipe's path. The video
model follows it, which is why a recipe can drive a render with the reference
clip long gone, and why editing a curve changes the shot.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from .recipe import CameraLayer


def fit_frame(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Center-crop to the render's aspect ratio, then resize."""
    width, height = size
    h, w = image.shape[:2]
    scale = max(width / w, height / h)
    resized = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    top = (resized.shape[0] - height) // 2
    left = (resized.shape[1] - width) // 2
    return resized[top : top + height, left : left + width]


def overscan_for(camera: CameraLayer, size: tuple[int, int]) -> float:
    """Smallest base zoom that keeps the frame's edges covered for the whole move.

    A push-in covers its own edges, so the guide can open on the full photo.
    """
    if not camera.x:
        return 1.0
    width, height = size
    x, y, zoom, roll = (
        np.asarray(curve) * camera.gain for curve in (camera.x, camera.y, camera.zoom, camera.roll)
    )
    tilt_margin = np.abs(np.sin(roll)) * max(width / height, height / width)
    needed = 1 + 2 * np.maximum(np.abs(x), np.abs(y) * width / height) + tilt_margin
    return float(max(1.0, (needed / np.exp(zoom)).max()))


def camera_matrices(camera: CameraLayer, frames: int, size: tuple[int, int]) -> list[np.ndarray]:
    width, height = size
    center = np.array([width / 2, height / 2])
    base = overscan_for(camera, size)
    t = np.linspace(0, 1, frames)
    grid = np.linspace(0, 1, len(camera.x))
    x, y, zoom, roll = (
        np.interp(t, grid, np.asarray(curve) * camera.gain)
        for curve in (camera.x, camera.y, camera.zoom, camera.roll)
    )
    matrices = []
    for i in range(frames):
        scale = base * np.exp(zoom[i])
        a, b = scale * np.cos(roll[i]), scale * np.sin(roll[i])
        linear = np.array([[a, -b], [b, a]])
        offset = center - linear @ center - np.array([x[i] * width, y[i] * width])
        matrices.append(np.column_stack([linear, offset]))
    return matrices


def guide_frames(image: np.ndarray, camera: CameraLayer, frames: int, size: tuple[int, int]) -> Iterator[np.ndarray]:
    source = cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)
    for matrix in camera_matrices(camera, frames, size):
        yield cv2.warpAffine(source, matrix, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def render_guide(
    image: np.ndarray,
    camera: CameraLayer,
    out: str | Path,
    frames: int,
    fps: float,
    size: tuple[int, int],
) -> Path:
    width, height = size
    out = Path(out)
    encoder = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(out)],
        stdin=subprocess.PIPE,
    )
    assert encoder.stdin is not None
    for frame in guide_frames(image, camera, frames, size):
        encoder.stdin.write(frame.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("ffmpeg failed while writing the guide video")
    return out
