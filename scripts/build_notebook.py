"""Build notebooks/unbake_free_gpu.ipynb with this repo's code packed inside it.

The notebook needs nothing but a GPU and a Hugging Face token, so it can run
before the repo is public.
"""

from __future__ import annotations

import base64
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKED = ["src/unbake", "custom_nodes/comfyui_unbake", "examples/recipes", "examples/photos/lighthouse.jpg"]


def pack() -> str:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in PACKED:
            archive.add(ROOT / path, arcname=path, filter=lambda t: None if "__pycache__" in t.name else t)
    return base64.b64encode(buffer.getvalue()).decode()


SETUP = r"""
import base64, io, os, subprocess, sys, tarfile, time
from pathlib import Path

HOME = Path("/tmp/unbake_run")
HOME.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(PACKED))) as archive:
    archive.extractall(HOME / "unbake")

print(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; free -g | head -2; df -h /tmp | tail -1",
                     shell=True, capture_output=True, text=True).stdout)

try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    from getpass import getpass
    token = getpass("Hugging Face token: ")
os.environ["HF_TOKEN"] = token
"""

INSTALL = r"""
def sh(command):
    subprocess.run(command, shell=True, check=True)

sh(f"git clone --depth 1 https://github.com/comfyanonymous/ComfyUI {HOME}/ComfyUI")
sh(f"git clone https://github.com/Lightricks/ComfyUI-LTXVideo {HOME}/ComfyUI/custom_nodes/ComfyUI-LTXVideo")
sh(f"git -C {HOME}/ComfyUI/custom_nodes/ComfyUI-LTXVideo checkout 44c8a36b7c31ee00b4271daed11dc2a4aa481140")
sh(f"cp -r {HOME}/unbake/custom_nodes/comfyui_unbake {HOME}/ComfyUI/custom_nodes/")
sh(f"pip install -q -r {HOME}/ComfyUI/requirements.txt -r {HOME}/ComfyUI/custom_nodes/ComfyUI-LTXVideo/requirements.txt")
sh("pip install -q comfy-sdk comfy-api-proxy opencv-python-headless scipy pydantic huggingface_hub")
"""

MODELS = r"""
from huggingface_hub import hf_hub_download

FILES = [
    ("Lightricks/LTX-2.5", "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors", "diffusion_models"),
    ("Lightricks/LTX-2.5", "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors", "text_encoders"),
    ("Lightricks/LTX-2.5", "vae/ltx-2.5-audio-vae-bf16.safetensors", "vae"),
    ("Lightricks/LTX-2.5", "vae/ltx-2.5-video-vae-bf16.safetensors", "vae"),
    ("Cseti/LTX2.3-22B_IC-LoRA-Cameraman_v2", "LTX2.3-22B_IC-LoRA-Cameraman_v2_14000.safetensors", "loras"),
]
for repo, file, folder in FILES:
    target = HOME / "ComfyUI/models" / folder
    target.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(repo, file, local_dir=HOME / "hf", token=os.environ["HF_TOKEN"])
    link = target / Path(file).name
    if not link.exists():
        link.symlink_to(path)
    print("ready:", link.name)
"""

SERVERS = r"""
import urllib.request, json

env = {**os.environ, "PYTHONPATH": f"{HOME}/unbake/src"}
log = open(HOME / "comfy.log", "w")
comfy = subprocess.Popen([sys.executable, "main.py", "--port", "8188", "--lowvram"],
                         cwd=HOME / "ComfyUI", env=env, stdout=log, stderr=log)
for _ in range(120):
    try:
        urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=2)
        break
    except Exception:
        time.sleep(2)
proxy = subprocess.Popen(["comfy-api-proxy"], stdout=log, stderr=log)
time.sleep(5)

nodes = json.load(urllib.request.urlopen("http://127.0.0.1:8188/object_info"))
for name in ["UnbakeClip", "UnbakeGuide", "UnbakeMatch", "LTXICLoRALoaderModelOnly",
             "LTXAddVideoICLoRAGuide", "LTXVImgToVideoInplace"]:
    print(("OK      " if name in nodes else "MISSING ") + name)
"""

RENDER = r"""
import traceback
sys.path.insert(0, f"{HOME}/unbake/src")
os.environ["COMFY_BASE_URL"] = "http://127.0.0.1:8189"
from comfy_sdk import Comfy
from unbake.bake import Baker
from unbake.compile import RenderSettings
from unbake.recipe import Recipe

recipe = Recipe.model_validate_json((HOME / "unbake/examples/recipes/crashzoom.json").read_text())
photo = HOME / "unbake/examples/photos/lighthouse.jpg"
settings = RenderSettings(width=960, height=544, frames=97)
out = Path("/kaggle/working/unbake_out") if Path("/kaggle/working").exists() else HOME / "out"

started = time.time()
try:
    result = Baker(out, client=Comfy(client_info="unbake"), notify=lambda kind, data: print(kind, data)).bake(
        recipe, photo, "a white lighthouse on a green headland above a harbor at dusk", settings
    )
    print("best:", result["best"]["video"], round(result["best"]["report"]["overall"]), "of 100")
    (out / "result.json").write_text(json.dumps(result, indent=2))
except Exception:
    traceback.print_exc()
print(f"{(time.time() - started) / 60:.1f} minutes")
print("".join(open(HOME / "comfy.log").readlines()[-40:]))
"""

FINISH = r"""
import shutil
shutil.copy(HOME / "comfy.log", out / "comfy.log")
archive = shutil.make_archive(str(out.parent / "unbake_results"), "zip", out)
print("Download this file from the Output panel:", archive)
"""


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    }


def note(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def main() -> None:
    cells = [
        note(
            "# Unbake on a free GPU\n\n"
            "Turn on a GPU and internet in the notebook settings, add your Hugging Face token as a "
            "secret named `HF_TOKEN`, then run every cell. It downloads about 39 GB and renders one "
            "small clip."
        ),
        code(f'PACKED = "{pack()}"'),
        code(SETUP),
        code(INSTALL),
        code(MODELS),
        code(SERVERS),
        code(RENDER),
        code(FINISH),
    ]
    notebook = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = ROOT / "notebooks" / "unbake_free_gpu.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(notebook))
    print(out, f"{out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
