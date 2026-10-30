"""Known motion in, same numbers out. No GPU or network needed."""

import cv2
import numpy as np
import pytest

from unbake.analyze import analyze
from unbake.guide import render_guide
from unbake.match import compare
from unbake.recipe import CURVE_POINTS, CameraLayer


def textured_image(size=(960, 540)):
    width, height = size
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 255, (height // 8, width // 8, 3), dtype=np.uint8)
    image = cv2.resize(noise, size, interpolation=cv2.INTER_CUBIC)
    for _ in range(120):
        center = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.circle(image, center, int(rng.integers(6, 40)), color, -1)
    return image


def camera_with(x=0.0, y=0.0, zoom=0.0, roll=0.0):
    t = np.linspace(0, 1, CURVE_POINTS)
    return CameraLayer(
        x=(x * t).tolist(), y=(y * t).tolist(), zoom=(zoom * t).tolist(), roll=(roll * t).tolist()
    )


CASES = [
    ("push_in", camera_with(zoom=0.3)),
    ("pull_out", camera_with(zoom=-0.3)),
    ("pan_right", camera_with(x=0.2)),
    ("pan_left", camera_with(x=-0.2)),
    ("tilt_down", camera_with(y=0.15)),
    ("tilt_up", camera_with(y=-0.15)),
    ("roll_ccw", camera_with(roll=0.15)),
]


@pytest.mark.parametrize("label,camera", CASES, ids=[c[0] for c in CASES])
def test_motion_is_recovered(tmp_path, label, camera):
    clip = render_guide(textured_image(), camera, tmp_path / "clip.mp4", 72, 24, (960, 540))
    recipe = analyze(clip)
    assert label in recipe.camera.move


def test_recovered_magnitude_is_close(tmp_path):
    camera = camera_with(x=0.2, zoom=0.25)
    clip = render_guide(textured_image(), camera, tmp_path / "clip.mp4", 72, 24, (960, 540))
    recipe = analyze(clip)
    assert recipe.camera.x[-1] == pytest.approx(0.2, abs=0.03)
    assert recipe.camera.zoom[-1] == pytest.approx(0.25, abs=0.03)


def test_guide_rebuilt_from_a_recipe_matches_it(tmp_path):
    camera = camera_with(x=0.15, zoom=0.2)
    clip = render_guide(textured_image(), camera, tmp_path / "ref.mp4", 72, 24, (960, 540))
    reference = analyze(clip)

    # A different picture, same recipe.
    other = cv2.flip(textured_image(), 0)
    rendered = render_guide(other, reference.camera, tmp_path / "out.mp4", 72, 24, (960, 540))
    report = compare(reference, analyze(rendered))
    assert report.layers["camera"].score > 85


def test_a_wrong_move_scores_low(tmp_path):
    reference = analyze(
        render_guide(textured_image(), camera_with(x=0.2), tmp_path / "a.mp4", 72, 24, (960, 540))
    )
    wrong = analyze(
        render_guide(textured_image(), camera_with(x=-0.2), tmp_path / "b.mp4", 72, 24, (960, 540))
    )
    assert compare(reference, wrong).layers["camera"].score < 50


def test_preview_scores_itself_on_camera_only(tmp_path, monkeypatch):
    monkeypatch.setenv("UNBAKE_DATA", str(tmp_path / "data"))
    from fastapi.testclient import TestClient

    from unbake import server

    monkeypatch.setattr(server, "RUNS", tmp_path)
    clip = render_guide(textured_image(), camera_with(zoom=0.3), tmp_path / "ref.mp4", 72, 24, (960, 540))
    recipe = analyze(clip)
    monkeypatch.setattr(server, "load_recipe", lambda _: recipe)

    cv2.imwrite(str(tmp_path / "photo.png"), cv2.flip(textured_image(), 1))
    with (tmp_path / "photo.png").open("rb") as photo:
        response = TestClient(server.app).post("/api/recipes/x/preview", files={"photo": photo})
    report = response.json()["report"]
    assert list(report["layers"]) == ["camera"]
    assert report["overall"] > 80
