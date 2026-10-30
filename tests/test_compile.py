"""Graphs must hang together, and the fix loop must touch only what failed."""

from pathlib import Path

import numpy as np

from unbake.bake import Baker, fix
from unbake.compile import (
    RenderSettings,
    bake_graph,
    clip_graph,
    describe_graph,
    keyframe_graph,
    node_graph,
    valid_frames,
)
from unbake.guide import overscan_for, render_guide
from unbake.match import LayerScore, MatchReport, compare
from unbake.recipe import (
    CURVE_POINTS,
    CameraLayer,
    LookLayer,
    PacingLayer,
    Recipe,
    SoundLayer,
    SourceInfo,
    TransformLayer,
)


def links(graph):
    for node in graph.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                yield value[0]


def test_every_link_points_at_a_node():
    graphs = [
        bake_graph("p", RenderSettings()),
        bake_graph("p", RenderSettings(), has_end_frame=True),
        keyframe_graph("make it gold"),
        describe_graph("what is this"),
    ]
    for graph in graphs:
        assert all(target in graph for target in links(graph))


def test_node_graph_builds_the_guide_inside_comfy():
    graph = node_graph("{}", "p", RenderSettings())
    assert all(target in graph for target in links(graph))
    assert "LoadVideo" not in {node["class_type"] for node in graph.values()}


def test_clip_graph_starts_from_a_video():
    graph = clip_graph("clip.mp4", "p", RenderSettings())
    assert graph["40"]["class_type"] == "UnbakeClip"
    assert all(target in graph for target in links(graph))


def test_frame_counts_fit_ltx():
    for wanted in (9, 10, 97, 100, 144):
        assert (valid_frames(wanted) - 1) % 8 == 0


def make_recipe(**camera) -> Recipe:
    t = np.linspace(0, 1, CURVE_POINTS)
    flat = [0.0] * CURVE_POINTS
    curves = {k: (camera.get(k, 0.0) * t).tolist() for k in ("x", "y", "zoom", "roll")}
    return Recipe(
        source=SourceInfo(duration=4, fps=24, width=960, height=540, shots=[(0, 4)]),
        camera=CameraLayer(**curves),
        transform=TransformLayer(),
        look=LookLayer(lab_mean=[50, 0, 0], lab_std=[20, 5, 5], contrast=20),
        pacing=PacingLayer(energy=flat),
        sound=SoundLayer(),
    )


def report_failing(*names) -> MatchReport:
    return MatchReport(
        overall=40,
        layers={n: LayerScore(score=40, detail="") for n in names},
        failing=list(names),
    )


def test_fix_raises_gain_when_the_move_came_up_short():
    wanted, got = make_recipe(zoom=0.3), make_recipe(zoom=0.15)
    camera, settings, _, _ = fix(wanted, report_failing("camera"), got, RenderSettings())
    assert camera.gain == 2.0
    assert settings.guide_strength == 1.0
    assert settings.first_frame_strength < RenderSettings.first_frame_strength


def test_fix_leaves_passing_layers_alone():
    recipe = make_recipe(zoom=0.3)
    camera, settings, look, _ = fix(recipe, report_failing("look"), recipe, RenderSettings())
    assert camera.gain == 1.0
    assert settings == RenderSettings()
    assert look.strength > recipe.look.strength


def test_broken_render_keeps_the_first_frame_and_changes_seed():
    recipe = make_recipe(zoom=0.3)
    report = report_failing("camera", "opening")
    camera, settings, _, _ = fix(recipe, report, recipe, RenderSettings())
    assert camera.gain == 1.0
    assert settings.first_frame_strength > RenderSettings.first_frame_strength
    assert settings.seed != RenderSettings.seed


def test_push_in_guide_opens_on_the_full_photo():
    assert overscan_for(make_recipe(zoom=1.0).camera, (960, 544)) == 1.0
    assert overscan_for(make_recipe(x=0.2).camera, (960, 544)) > 1.0


class FakeOutput:
    def __init__(self, source: Path):
        self.source = source

    def to_file(self, path):
        Path(path).write_bytes(self.source.read_bytes())


class FakeJob:
    """Lists the uploaded guide first, the way a real job does."""

    def __init__(self, source):
        self.outputs = [FakeOutput(Path("guide-that-must-not-be-saved")), FakeOutput(source)]

    def result(self):
        return self

    def get_outputs(self, node):
        return [self.outputs[1]] if node == "30" else []


class FakeWorkflow:
    def __init__(self, graph):
        self.json = graph

    def set_input(self, node, field, value):
        self.json[node]["inputs"][field] = value


class FakeClient:
    """Answers each render with a clip whose camera falls short at first, then lands."""

    def __init__(self, clips):
        self.clips = iter(clips)
        self.workflows = self
        self.assets = self

    def from_json(self, graph):
        return FakeWorkflow(graph)

    def from_file(self, path):
        return str(path)

    def submit(self, workflow):
        return FakeJob(next(self.clips))


def test_loop_stops_once_the_camera_lands(tmp_path):
    import cv2

    rng = np.random.default_rng(1)
    image = cv2.resize(rng.integers(0, 255, (68, 120, 3), dtype=np.uint8), (960, 544))
    photo = tmp_path / "photo.png"
    cv2.imwrite(str(photo), image)

    from unbake.analyze import analyze

    reference = analyze(
        render_guide(image, make_recipe(zoom=0.3).camera, tmp_path / "ref.mp4", 97, 24, (960, 544))
    )
    reference.look.enabled = False
    reference.pacing.enabled = False
    short = render_guide(image, make_recipe(zoom=0.1).camera, tmp_path / "short.mp4", 97, 24, (960, 544))
    good = render_guide(image, make_recipe(zoom=0.3).camera, tmp_path / "good.mp4", 97, 24, (960, 544))

    baker = Baker(tmp_path / "run", client=FakeClient([short, good]))
    result = baker.bake(reference, photo)

    assert len(result["passes"]) == 2
    assert result["best"]["pass"] == 2
    assert result["best"]["report"]["overall"] > compare(reference, analyze(short)).overall
