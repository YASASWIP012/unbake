"""The app's HTTP API and static files."""

from __future__ import annotations

import io
import json
import os
import queue
import shutil
import threading
import uuid
import zipfile
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .analyze import analyze
from .bake import Baker, fit_frame
from .compile import RenderSettings, bake_graph, prompt_for, valid_frames
from .guide import render_guide
from .match import compare
from .recipe import Recipe

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("UNBAKE_DATA", str(ROOT / "data")))
RECIPES = DATA / "recipes"
RUNS = DATA / "runs"
WEB = ROOT / "web"
EXAMPLES = ROOT / "examples" / "recipes"
MAX_CLIP_SECONDS = 20

for folder in (RECIPES, RUNS):
    folder.mkdir(parents=True, exist_ok=True)


def load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        key, _, value = line.partition("=")
        if key.strip() and value.strip() and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


load_env()
app = FastAPI(title="Unbake")
jobs: dict[str, queue.Queue] = {}


def recipe_path(recipe_id: str) -> Path:
    if not recipe_id.isalnum():
        raise HTTPException(404, "no such recipe")
    return RECIPES / f"{recipe_id}.json"


def load_recipe(recipe_id: str) -> Recipe:
    path = recipe_path(recipe_id)
    if not path.exists():
        example = EXAMPLES / f"{recipe_id}.json"
        if not example.exists():
            raise HTTPException(404, "no such recipe")
        path = example
    return Recipe.model_validate_json(path.read_text())


def save_recipe(recipe: Recipe) -> Recipe:
    recipe_path(recipe.id).write_text(recipe.model_dump_json(indent=2))
    return recipe


def keep_upload(upload: UploadFile, folder: Path, default_suffix: str) -> Path:
    suffix = Path(upload.filename or "").suffix.lower() or default_suffix
    path = folder / f"{uuid.uuid4().hex}{suffix}"
    with path.open("wb") as out:
        shutil.copyfileobj(upload.file, out)
    return path


@app.get("/api/status")
def status() -> dict:
    return {"can_render": bool(os.environ.get("COMFY_API_KEY") or os.environ.get("COMFY_BASE_URL"))}


@app.get("/api/recipes")
def list_recipes() -> list[Recipe]:
    found = {p.stem: p for p in EXAMPLES.glob("*.json")} if EXAMPLES.exists() else {}
    found.update({p.stem: p for p in RECIPES.glob("*.json")})
    ordered = sorted(found.values(), key=lambda p: p.stat().st_mtime, reverse=True)
    return [Recipe.model_validate_json(p.read_text()) for p in ordered]


@app.post("/api/unbake")
def unbake(clip: UploadFile = File(...)) -> Recipe:
    """Measure a clip and throw it away. The recipe is all that stays."""
    run = RUNS / uuid.uuid4().hex
    run.mkdir()
    path = keep_upload(clip, run, ".mp4")
    try:
        try:
            recipe = analyze(path, Path(clip.filename or "effect").stem)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        if recipe.source.duration > MAX_CLIP_SECONDS:
            recipe.unsupported.append(f"clip is over {MAX_CLIP_SECONDS}s, so timing is coarse")
        if status()["can_render"]:
            try:
                Baker(run).describe(path, recipe)
            except Exception as error:  # the numbers are still good without the words
                recipe.unsupported.append(f"could not describe the transform: {error}")
        return save_recipe(recipe)
    finally:
        shutil.rmtree(run, ignore_errors=True)


@app.get("/api/recipes/{recipe_id}")
def get_recipe(recipe_id: str) -> Recipe:
    return load_recipe(recipe_id)


@app.put("/api/recipes/{recipe_id}")
def put_recipe(recipe_id: str, recipe: Recipe) -> Recipe:
    recipe.id = recipe_id
    return save_recipe(recipe)


class SpliceRequest(BaseModel):
    base: str
    layers: dict[str, str]


@app.post("/api/splice")
def splice(request: SpliceRequest) -> Recipe:
    """Build a new recipe from layers of others, such as one's camera and another's look."""
    allowed = {"camera", "transform", "look", "pacing", "sound"}
    if not set(request.layers) <= allowed:
        raise HTTPException(422, f"layers must be among {sorted(allowed)}")
    base = load_recipe(request.base)
    update = {name: getattr(load_recipe(source), name) for name, source in request.layers.items()}
    names = sorted({base.name, *(load_recipe(s).name for s in request.layers.values())})
    child = base.model_copy(
        update={
            **update,
            "id": uuid.uuid4().hex[:10],
            "name": " x ".join(names),
            "lineage": [request.base, *request.layers.values()],
        }
    )
    return save_recipe(child)


def prepared_photo(photo: UploadFile, run: Path):
    path = keep_upload(photo, run, ".png")
    image = cv2.imread(str(path))
    if image is None:
        raise HTTPException(422, "that file is not an image")
    return path, image


@app.post("/api/recipes/{recipe_id}/preview")
def preview(recipe_id: str, photo: UploadFile = File(...)) -> dict:
    """Move the photo along the recipe's path. Instant, local, free."""
    recipe = load_recipe(recipe_id)
    run_id = uuid.uuid4().hex
    run = RUNS / run_id
    run.mkdir()
    _, image = prepared_photo(photo, run)
    size = (RenderSettings.width, RenderSettings.height)
    frames = valid_frames(round(min(recipe.source.duration, 6.0) * RenderSettings.fps))
    out = render_guide(fit_frame(image, size), recipe.camera, run / "preview.mp4", frames, RenderSettings.fps, size)
    # A preview only moves the camera, so only the camera is scored.
    camera_only = recipe.model_copy(deep=True)
    for layer in (camera_only.transform, camera_only.look, camera_only.pacing, camera_only.sound):
        layer.enabled = False
    report = compare(camera_only, analyze(out))
    return {"url": f"/api/runs/{run_id}/preview.mp4", "report": report.model_dump()}


@app.post("/api/recipes/{recipe_id}/bake")
def bake(recipe_id: str, photo: UploadFile = File(...), subject: str = Form("")) -> dict:
    if not status()["can_render"]:
        raise HTTPException(409, "set COMFY_API_KEY in .env to render")
    recipe = load_recipe(recipe_id)
    run_id = uuid.uuid4().hex
    run = RUNS / run_id
    run.mkdir()
    path, _ = prepared_photo(photo, run)
    events: queue.Queue = queue.Queue()
    jobs[run_id] = events

    def work() -> None:
        def notify(kind: str, data: dict) -> None:
            events.put({"kind": kind, **data})

        try:
            result = Baker(run, notify=notify).bake(recipe, path, subject)
            (run / "workflow_api.json").write_text(json.dumps(result["best"]["graph"], indent=2))
            events.put({"kind": "done", "run": run_id, "best": result["best"]})
        except Exception as error:
            events.put({"kind": "error", "message": str(error)})
        events.put(None)

    threading.Thread(target=work, daemon=True).start()
    return {"run": run_id}


@app.get("/api/jobs/{run_id}/events")
def job_events(run_id: str) -> StreamingResponse:
    events = jobs.get(run_id)
    if events is None:
        raise HTTPException(404, "no such job")

    def stream():
        while True:
            try:
                item = events.get(timeout=15)
            except queue.Empty:
                yield ": still working\n\n"
                continue
            if item is None:
                return
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/runs/{run_id}/{name}")
def run_file(run_id: str, name: str) -> FileResponse:
    if not run_id.isalnum() or Path(name).name != name:
        raise HTTPException(404, "no such file")
    path = RUNS / run_id / name
    if not path.exists():
        raise HTTPException(404, "no such file")
    return FileResponse(path)


@app.get("/api/recipes/{recipe_id}/export")
def export(recipe_id: str) -> Response:
    """A recipe, its compiled graph and a note on running it, ready to share."""
    recipe = load_recipe(recipe_id)
    frames = valid_frames(round(min(recipe.source.duration, 6.0) * RenderSettings.fps))
    use_end = recipe.transform.enabled and bool(recipe.transform.after)
    graph = bake_graph(prompt_for(recipe), RenderSettings(frames=frames), has_end_frame=use_end)
    note = (
        f"# {recipe.name}\n\n"
        f"Camera: {recipe.camera.move}, {recipe.camera.easing}.\n\n"
        "workflow_api.json is a ComfyUI API graph. Set node 1 to your image and node 2 to a guide "
        "video, which is what Preview camera produces in Unbake.\n"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{recipe.id}.recipe.json", recipe.model_dump_json(indent=2))
        archive.writestr("workflow_api.json", json.dumps(graph, indent=2))
        archive.writestr("README.md", note)
    return Response(
        buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{recipe.id}.zip"'},
    )


app.mount("/", StaticFiles(directory=WEB, html=True), name="web")


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8420")))
