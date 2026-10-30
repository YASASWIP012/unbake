"""Run a recipe on a photo and keep going until the match holds.

One local step sits around the render: grading the result toward the recipe's
look. Everything else goes through the Comfy SDK, so pointing COMFY_BASE_URL at
a local comfy-api-proxy runs the same code on your own GPU.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np
from comfy_sdk import Comfy

from .analyze import analyze
from .compile import (
    RenderSettings,
    bake_graph,
    describe_graph,
    keyframe_graph,
    prompt_for,
    valid_frames,
)
from .guide import fit_frame, render_guide
from .match import LayerScore, MatchReport, compare
from .recipe import CameraLayer, LookLayer, Recipe

MAX_PASSES = 3
MIN_OPENING_MATCH = 0.6
SAVE_VIDEO_NODE = "30"  # job.outputs also lists the uploaded guide, so always ask for this node
Notify = Callable[[str, dict], None]


def _quiet(_: str, __: dict) -> None:
    return None


def grade(path: Path, look: LookLayer, out: Path) -> Path:
    """Move a clip's Lab statistics toward the recipe's, and keep its audio."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_mean, target_std = np.array(look.lab_mean), np.array(look.lab_std)
    encoder = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-i", str(path),
         "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "16",
         "-pix_fmt", "yuv420p", "-c:a", "aac", str(out)],
        stdin=subprocess.PIPE,
    )
    assert encoder.stdin is not None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        lab = cv2.cvtColor(frame.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
        mean, std = lab.reshape(-1, 3).mean(axis=0), lab.reshape(-1, 3).std(axis=0)
        moved = (lab - mean) * (target_std / np.maximum(std, 1e-3)) + target_mean
        blended = lab + look.strength * (moved - lab)
        bgr = cv2.cvtColor(blended.astype(np.float32), cv2.COLOR_LAB2BGR)
        encoder.stdin.write((np.clip(bgr, 0, 1) * 255).astype(np.uint8).tobytes())
    cap.release()
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("ffmpeg failed while grading")
    return out


def opening_match(clip: Path, guide: Path) -> float:
    """How well the render's first frame matches the guide's. A broken render fails here."""
    frames = []
    for path in (clip, guide):
        cap = cv2.VideoCapture(str(path))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return 0.0
        frames.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36)).astype(np.float32).ravel())
    return float(np.corrcoef(frames[0], frames[1])[0, 1])


def fix(recipe: Recipe, report: MatchReport, rendered: Recipe, settings: RenderSettings
        ) -> tuple[CameraLayer, RenderSettings, LookLayer, list[str]]:
    """Change only what the failing layers call for."""
    camera, look, notes = recipe.camera.model_copy(), recipe.look.model_copy(), []
    new = replace(settings)
    if "opening" in report.failing:
        new.first_frame_strength = min(settings.first_frame_strength + 0.15, 0.9)
        new.seed = settings.seed + 1
        notes.append(f"opening: first frame strength {new.first_frame_strength:.2f}, new seed {new.seed}")
        return camera, new, look, notes
    if "camera" in report.failing:
        wanted = max(abs(recipe.camera.x[-1]), abs(recipe.camera.y[-1]), abs(recipe.camera.zoom[-1]))
        got = max(abs(rendered.camera.x[-1]), abs(rendered.camera.y[-1]), abs(rendered.camera.zoom[-1]))
        if got > 1e-4:
            camera.gain = float(np.clip(camera.gain * wanted / got, 0.5, 2.0))
        new.guide_strength = min(settings.guide_strength + 0.15, 1.0)
        new.first_frame_strength = max(settings.first_frame_strength - 0.1, 0.3)
        notes.append(f"camera: gain {camera.gain:.2f}, guide strength {new.guide_strength:.2f}")
    if "transform" in report.failing:
        new.end_frame_strength = min(settings.end_frame_strength + 0.15, 1.0)
        notes.append(f"transform: end frame strength {new.end_frame_strength:.2f}")
    if "look" in report.failing:
        look.strength = min(look.strength + 0.15, 1.0)
        notes.append(f"look: grade strength {look.strength:.2f}")
    if "pacing" in report.failing or "sound" in report.failing:
        new.seed = settings.seed + 1
        notes.append(f"timing: new seed {new.seed}")
    return camera, new, look, notes


def rank(result: dict) -> tuple[bool, float]:
    """A clean render beats a broken one, whatever the broken one scored."""
    return "opening" not in result["report"]["failing"], result["report"]["overall"]


class Baker:
    def __init__(self, workdir: Path, client: Comfy | None = None, notify: Notify = _quiet) -> None:
        self.client = client or Comfy(client_info="unbake")
        self.workdir = workdir
        self.notify = notify
        workdir.mkdir(parents=True, exist_ok=True)

    def _run(self, graph: dict, inputs: dict[str, tuple[str, Path]]):
        workflow = self.client.workflows.from_json(deepcopy(graph))
        for node, (field, path) in inputs.items():
            workflow.set_input(node, field, self.client.assets.from_file(path))
        job = self.client.submit(workflow)
        return job.result()

    def describe(self, clip: Path, recipe: Recipe) -> None:
        """Fill the transform layer's words. Numbers stay local; only a contact sheet is sent."""
        cap = cv2.VideoCapture(str(clip))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sheet = []
        for position in (0, total // 2, max(total - 1, 0)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            ok, frame = cap.read()
            if ok:
                sheet.append(cv2.resize(frame, (480, 270)))
        cap.release()
        path = self.workdir / f"{recipe.id}-sheet.png"
        cv2.imwrite(str(path), np.hstack(sheet))
        question = (
            "These three frames are the start, middle and end of a video effect. "
            'Reply with JSON only: {"before": "<the subject at the start>", '
            '"after": "<what the subject becomes at the end, as an edit instruction>", '
            '"name": "<a two word name for the effect>"}'
        )
        job = self._run(describe_graph(question), {"2": ("image", path)})
        text = job.get_outputs("4")[0].to_bytes().decode("utf-8", "ignore")
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            words = json.loads(text[start : end + 1])
            recipe.transform.before = words.get("before", "")
            recipe.transform.after = words.get("after", "")
            recipe.name = words.get("name", recipe.name)

    def _keyframe(self, photo: Path, instruction: str, seed: int) -> Path:
        job = self._run(keyframe_graph(instruction, seed), {"11": ("image", photo)})
        out = self.workdir / f"keyframe-{seed}.png"
        job.get_outputs("9")[0].to_file(out)
        return out

    def bake(
        self,
        recipe: Recipe,
        photo: Path,
        subject: str = "",
        settings: RenderSettings | None = None,
    ) -> dict:
        """Render and score. Pass settings to fit a smaller GPU."""
        settings = settings or RenderSettings(
            frames=valid_frames(round(min(recipe.source.duration, 6.0) * RenderSettings.fps))
        )
        size = (settings.width, settings.height)
        image = fit_frame(cv2.imread(str(photo)), size)
        first = self.workdir / "first.png"
        cv2.imwrite(str(first), image)

        camera, look = recipe.camera, recipe.look
        use_end = recipe.transform.enabled and bool(recipe.transform.after)
        end_frame = None
        if use_end:
            self.notify("keyframe", {"instruction": recipe.transform.after})
            end_frame = self._keyframe(first, recipe.transform.after, settings.seed)

        best: dict | None = None
        passes: list[dict] = []
        for number in range(1, MAX_PASSES + 1):
            self.notify("pass", {"number": number, "settings": asdict(settings)})
            guide = render_guide(
                image, camera, self.workdir / f"guide-{number}.mp4", settings.frames, settings.fps, size
            )
            graph = bake_graph(
                prompt_for(recipe.model_copy(update={"camera": camera}), subject),
                settings,
                has_end_frame=use_end,
            )
            inputs = {"1": ("image", first), "2": ("file", guide)}
            if end_frame:
                inputs["3"] = ("image", end_frame)
            job = self._run(graph, inputs)
            raw = self.workdir / f"raw-{number}.mp4"
            job.get_outputs(SAVE_VIDEO_NODE)[0].to_file(raw)

            final = self.workdir / f"bake-{number}.mp4"
            if recipe.look.enabled:
                grade(raw, look, final)
            else:
                final = raw

            rendered = analyze(final)
            report = compare(recipe, rendered)
            opening = opening_match(final, guide)
            if opening < MIN_OPENING_MATCH:
                report.layers["opening"] = LayerScore(
                    score=max(opening, 0) * 100, detail=f"first frame match {opening:.2f}"
                )
                report.failing.append("opening")
            result = {"pass": number, "video": final.name, "report": report.model_dump()}
            passes.append(result)
            self.notify("scored", result)
            if best is None or rank(result) > rank(best):
                best = result
                best["graph"] = graph
            if not report.failing:
                break
            camera, settings, look, notes = fix(
                recipe.model_copy(update={"camera": camera, "look": look}), report, rendered, settings
            )
            self.notify("fix", {"notes": notes})
        return {"best": best, "passes": passes}
