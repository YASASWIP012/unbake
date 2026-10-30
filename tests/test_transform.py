"""A camera move alone is not a transform."""

import cv2
import numpy as np
from test_roundtrip import camera_with, textured_image

from unbake.analyze import analyze
from unbake.guide import render_guide


def test_zoom_alone_is_not_a_transform(tmp_path):
    clip = render_guide(textured_image(), camera_with(zoom=0.5), tmp_path / "zoom.mp4", 72, 24, (960, 540))
    assert not analyze(clip).transform.present


def test_changing_picture_is_a_transform(tmp_path):
    image = textured_image()
    path = tmp_path / "change.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (960, 540))
    gold = np.full_like(image, (40, 180, 230))
    for i in range(48):
        writer.write(cv2.addWeighted(image, 1 - i / 47, gold, i / 47, 0))
    writer.release()
    assert analyze(path).transform.present
