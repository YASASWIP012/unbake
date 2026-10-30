"""Unbake inside any ComfyUI graph.

Install the unbake package into ComfyUI's Python, then link this folder into
ComfyUI/custom_nodes. Recipes travel between nodes as JSON strings.
"""

from __future__ import annotations

import uuid

import cv2
import folder_paths
import numpy as np
import torch

from unbake.analyze import analyze
from unbake.compile import prompt_for
from unbake.guide import fit_frame, guide_frames, render_guide
from unbake.match import compare
from unbake.recipe import Recipe


class UnbakeClip:
    """Measure a video file and output its recipe."""

    CATEGORY = "unbake"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("recipe", "prompt")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video_path": ("STRING", {"default": ""})}}

    def run(self, video_path):
        recipe = analyze(video_path)
        return recipe.model_dump_json(), prompt_for(recipe)


class UnbakeRecipe:
    """Hold a recipe's JSON in the graph. Paste a file from examples/recipes, or any export."""

    CATEGORY = "unbake"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("recipe", "prompt")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"recipe_json": ("STRING", {"multiline": True, "default": ""})}}

    def run(self, recipe_json):
        recipe = Recipe.model_validate_json(recipe_json)
        return recipe.model_dump_json(), prompt_for(recipe)


def to_bgr(image):
    return (image.cpu().numpy()[..., ::-1] * 255).clip(0, 255).astype("uint8").copy()


class UnbakeGuideFrames:
    """Move an image along a recipe's camera path. Outputs the guide frames and the first frame."""

    CATEGORY = "unbake"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("guide", "first_frame")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "recipe": ("STRING", {"forceInput": True}),
                "width": ("INT", {"default": 960, "min": 64, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 544, "min": 64, "max": 4096, "step": 32}),
                "frames": ("INT", {"default": 97, "min": 9, "max": 481, "step": 8}),
            }
        }

    def run(self, image, recipe, width, height, frames):
        size = (width, height)
        picture = fit_frame(to_bgr(image[0]), size)
        camera = Recipe.model_validate_json(recipe).camera
        moved = [frame[..., ::-1].astype(np.float32) / 255 for frame in guide_frames(picture, camera, frames, size)]
        guide = torch.from_numpy(np.stack(moved))
        return guide, guide[:1]


class UnbakeGuide:
    """Render a recipe's camera path over an image, as a video file for an IC-LoRA guide."""

    CATEGORY = "unbake"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("guide_path",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "recipe": ("STRING", {"forceInput": True}),
                "frames": ("INT", {"default": 97, "min": 9, "max": 481, "step": 8}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
            }
        }

    def run(self, image, recipe, frames, fps):
        picture = (image[0].cpu().numpy()[..., ::-1] * 255).clip(0, 255).astype("uint8").copy()
        height, width = picture.shape[:2]
        size = (width - width % 2, height - height % 2)
        out = folder_paths.get_temp_directory() + f"/unbake_guide_{id(picture)}.mp4"
        render_guide(picture, Recipe.model_validate_json(recipe).camera, out, frames, fps, size)
        return (out,)


class UnbakeMatch:
    """Score a rendered video file against a recipe, layer by layer."""

    CATEGORY = "unbake"
    FUNCTION = "run"
    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("overall", "report")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "recipe": ("STRING", {"forceInput": True}),
                "video_path": ("STRING", {"default": ""}),
            }
        }

    def run(self, recipe, video_path):
        report = compare(Recipe.model_validate_json(recipe), analyze(video_path))
        return report.overall, report.model_dump_json(indent=2)


class UnbakeMatchFrames:
    """Score rendered frames against a recipe, inside the graph."""

    CATEGORY = "unbake"
    FUNCTION = "run"
    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("overall", "report")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "recipe": ("STRING", {"forceInput": True}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
            }
        }

    def run(self, images, recipe, fps):
        path = folder_paths.get_temp_directory() + f"/unbake_match_{uuid.uuid4().hex}.mp4"
        height, width = images.shape[1:3]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        for frame in images:
            writer.write(to_bgr(frame))
        writer.release()
        report = compare(Recipe.model_validate_json(recipe), analyze(path))
        return {"ui": {"text": [report.model_dump_json()]}, "result": (report.overall, report.model_dump_json(indent=2))}


NODE_CLASS_MAPPINGS = {
    "UnbakeRecipe": UnbakeRecipe,
    "UnbakeGuideFrames": UnbakeGuideFrames,
    "UnbakeMatchFrames": UnbakeMatchFrames,
    "UnbakeClip": UnbakeClip,
    "UnbakeGuide": UnbakeGuide,
    "UnbakeMatch": UnbakeMatch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "UnbakeRecipe": "Unbake Recipe",
    "UnbakeGuideFrames": "Unbake Guide Frames",
    "UnbakeMatchFrames": "Unbake Match Frames",
    "UnbakeClip": "Unbake Clip",
    "UnbakeGuide": "Unbake Camera Guide",
    "UnbakeMatch": "Unbake Match Score",
}
