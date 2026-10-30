"""Compile a recipe into ComfyUI API graphs.

The bake graph is LTX-2.5 with the Cameraman IC-LoRA, following the layout of
the V2V IC-LoRA template in ComfyUI-LTXVideo. The guide video comes from the
recipe's curves, and an optional end keyframe carries the transform layer.
"""

from __future__ import annotations

from dataclasses import dataclass

from .recipe import CameraLayer, Recipe

LTX_UNET = "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
LTX_VIDEO_VAE = "ltx-2.5-video-vae-bf16.safetensors"
LTX_AUDIO_VAE = "ltx-2.5-audio-vae-bf16.safetensors"
LTX_TEXT = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
CAMERAMAN_LORA = "LTX2.3-22B_IC-LoRA-Cameraman_v2_14000.safetensors"
DISTILLED_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"

KLEIN_UNET = "flux3-image-klein-int8-convrot.safetensors"
KLEIN_TEXT = "qwen3vl_4b_int8_convrot.safetensors"
KLEIN_VAE = "flux2-vae.safetensors"

DESCRIBE_TEXT = "qwen3vl_4b_fp8_scaled.safetensors"

NEGATIVE = "static, frozen frame, warped face, melting, text, watermark, flicker"

MOVE_WORDS = {
    "push_in": "the camera pushes in",
    "pull_out": "the camera pulls back",
    "pan_left": "the camera pans left",
    "pan_right": "the camera pans right",
    "tilt_up": "the camera tilts up",
    "tilt_down": "the camera tilts down",
    "roll_cw": "the camera rolls clockwise",
    "roll_ccw": "the camera rolls counterclockwise",
    "handheld": "handheld camera with natural shake",
    "static": "locked-off camera",
}
EASING_WORDS = {
    "linear": "at an even speed",
    "ease_in": "starting slow and accelerating",
    "ease_out": "starting fast and settling",
    "ease_in_out": "easing in and out smoothly",
}


@dataclass
class RenderSettings:
    width: int = 960
    height: int = 544
    frames: int = 97
    fps: float = 24.0
    seed: int = 42
    guide_strength: float = 1.0
    first_frame_strength: float = 0.6
    end_frame_strength: float = 0.7


def valid_frames(frames: int) -> int:
    """LTX wants 8n + 1 frames."""
    return max(9, (frames - 1) // 8 * 8 + 1)


def camera_words(camera: CameraLayer) -> str:
    """Name the one move that dominates. Listing every small axis dilutes the instruction."""
    gain = camera.gain
    zoom = (camera.zoom[-1] if camera.zoom else 0.0) * gain
    x = (camera.x[-1] if camera.x else 0.0) * gain
    y = (camera.y[-1] if camera.y else 0.0) * gain
    size = max(abs(zoom), abs(x) * 2, abs(y) * 2)
    if size < 0.06:
        return MOVE_WORDS["handheld" if "handheld" in camera.move else "static"]
    speed = "a very fast, dramatic" if size > 0.7 else "a steady" if size > 0.25 else "a slow, subtle"
    if abs(zoom) >= max(abs(x), abs(y)) * 2:
        shot = "dolly forward deep into the scene" if zoom > 0 else "dolly backward out of the scene"
    elif abs(x) >= abs(y):
        shot = f"pan to the {'right' if x > 0 else 'left'}"
    else:
        shot = f"tilt {'down' if y > 0 else 'up'}"
    return f"Camera movement: {speed} {shot}, {EASING_WORDS[camera.easing]}, the whole time"


def prompt_for(recipe: Recipe, subject: str = "") -> str:
    parts = [subject.strip()] if subject.strip() else []
    if recipe.camera.enabled:
        parts.append(camera_words(recipe.camera))
    if recipe.transform.enabled and recipe.transform.after:
        parts.append(f"the scene transforms: {recipe.transform.after}")
    if recipe.sound.enabled and recipe.sound.present and recipe.sound.description:
        parts.append(f"sound: {recipe.sound.description}")
    return ". ".join(parts) + "."


def bake_graph(
    prompt: str,
    settings: RenderSettings,
    *,
    has_end_frame: bool = False,
) -> dict:
    """LTX-2.5 render. Nodes 1, 2 and 3 take the first frame, guide and end frame."""
    s = settings
    guided = ["14", 0], ["14", 1], ["14", 2]
    graph: dict[str, dict] = {
        "1": {"class_type": "LoadImage", "inputs": {"image": ""}},
        "2": {"class_type": "LoadVideo", "inputs": {"file": ""}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": LTX_AUDIO_VAE}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": LTX_VIDEO_VAE}},
        "6": {"class_type": "UNETLoader", "inputs": {"unet_name": LTX_UNET, "weight_dtype": "default"}},
        "7": {"class_type": "CLIPLoader", "inputs": {"clip_name": LTX_TEXT, "type": "ltxv", "device": "default"}},
        "8": {
            "class_type": "LTXICLoRALoaderModelOnly",
            "inputs": {"model": ["6", 0], "lora_name": CAMERAMAN_LORA, "strength_model": 1},
        },
        "9": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 0], "text": prompt}},
        "10": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["7", 0], "text": NEGATIVE}},
        "11": {
            "class_type": "LTXVConditioning",
            "inputs": {"positive": ["9", 0], "negative": ["10", 0], "frame_rate": s.fps},
        },
        "12": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {"width": s.width, "height": s.height, "length": s.frames, "batch_size": 1},
        },
        "13": {
            "class_type": "LTXVImgToVideoInplace",
            "inputs": {
                "vae": ["5", 0],
                "image": ["1", 0],
                "latent": ["12", 0],
                "bypass": False,
                "strength": s.first_frame_strength,
            },
        },
        "15": {"class_type": "GetVideoComponents", "inputs": {"video": ["2", 0]}},
        "14": {
            "class_type": "LTXAddVideoICLoRAGuide",
            "inputs": {
                "positive": ["11", 0],
                "negative": ["11", 1],
                "vae": ["5", 0],
                "latent": ["13", 0],
                "image": ["15", 0],
                "latent_downscale_factor": ["8", 1],
                "frame_idx": 0,
                "strength": s.guide_strength,
                "crop": "disabled",
                "use_tiled_encode": False,
                "tile_size": 256,
                "tile_overlap": 64,
                "use_streaming": False,
                "tile_frames": 97,
            },
        },
    }

    if has_end_frame:
        graph["3"] = {"class_type": "LoadImage", "inputs": {"image": ""}}
        graph["16"] = {"class_type": "LTXVPreprocess", "inputs": {"image": ["3", 0], "img_compression": 18}}
        graph["17"] = {
            "class_type": "LTXVAddGuide",
            "inputs": {
                "positive": ["14", 0],
                "negative": ["14", 1],
                "vae": ["5", 0],
                "latent": ["14", 2],
                "image": ["16", 0],
                "frame_idx": -1,
                "strength": s.end_frame_strength,
            },
        }
        guided = ["17", 0], ["17", 1], ["17", 2]

    positive, negative, video_latent = guided
    graph.update(
        {
            "18": {
                "class_type": "LTXVEmptyLatentAudio",
                "inputs": {
                    "frames_number": s.frames,
                    "frame_rate": s.fps,
                    "batch_size": 1,
                    "audio_vae": ["4", 0],
                },
            },
            "19": {
                "class_type": "LTXVConcatAVLatent",
                "inputs": {"video_latent": video_latent, "audio_latent": ["18", 0]},
            },
            "20": {
                "class_type": "CFGGuider",
                "inputs": {"model": ["8", 0], "positive": positive, "negative": negative, "cfg": 1},
            },
            "21": {"class_type": "RandomNoise", "inputs": {"noise_seed": s.seed}},
            "22": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
            "23": {"class_type": "ManualSigmas", "inputs": {"sigmas": DISTILLED_SIGMAS}},
            "24": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["21", 0],
                    "guider": ["20", 0],
                    "sampler": ["22", 0],
                    "sigmas": ["23", 0],
                    "latent_image": ["19", 0],
                },
            },
            "25": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["24", 0]}},
            "26": {
                "class_type": "LTXVCropGuides",
                "inputs": {"positive": positive, "negative": negative, "latent": ["25", 0]},
            },
            "27": {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["25", 1], "audio_vae": ["4", 0]}},
            "28": {
                "class_type": "VAEDecodeTiled",
                "inputs": {
                    "samples": ["26", 2],
                    "vae": ["5", 0],
                    "tile_size": 512,
                    "overlap": 64,
                    "temporal_size": 64,
                    "temporal_overlap": 8,
                },
            },
            "29": {
                "class_type": "CreateVideo",
                "inputs": {
                    "images": ["28", 0],
                    "audio": ["27", 0],
                    "fps": s.fps,
                    "bit_depth": 8,
                    "color_space": "sRGB",
                    "codec": "none",
                },
            },
            "30": {
                "class_type": "SaveVideo",
                "inputs": {
                    "video": ["29", 0],
                    "filename_prefix": "unbake/bake",
                    "format": "auto",
                    "format.codec": "auto",
                    "codec": "auto",
                },
            },
        }
    )
    return graph


def node_graph(recipe_json: str, prompt: str, settings: RenderSettings) -> dict:
    """The bake graph with the guide built by Unbake's own nodes, so it runs inside ComfyUI."""
    s = settings
    graph = bake_graph(prompt, s)
    del graph["2"], graph["15"]
    graph["40"] = {"class_type": "UnbakeRecipe", "inputs": {"recipe_json": recipe_json}}
    graph["41"] = {
        "class_type": "UnbakeGuideFrames",
        "inputs": {
            "image": ["1", 0],
            "recipe": ["40", 0],
            "width": s.width,
            "height": s.height,
            "frames": s.frames,
        },
    }
    graph["13"]["inputs"]["image"] = ["41", 1]
    graph["14"]["inputs"]["image"] = ["41", 0]
    graph["42"] = {
        "class_type": "UnbakeMatchFrames",
        "inputs": {"images": ["28", 0], "recipe": ["40", 0], "fps": s.fps},
    }
    return graph


def clip_graph(video_path: str, prompt: str, settings: RenderSettings) -> dict:
    """The whole loop in one queue: measure a clip, then render its camera on a photo."""
    graph = node_graph("", prompt, settings)
    graph["40"] = {"class_type": "UnbakeClip", "inputs": {"video_path": video_path}}
    return graph


def keyframe_graph(instruction: str, seed: int = 42) -> dict:
    """FLUX 3 Klein edit: the user's photo plus the recipe's end state. Node 11 takes the image."""
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": KLEIN_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": KLEIN_TEXT, "type": "flux3", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": KLEIN_VAE}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": instruction}},
        "32": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "11": {"class_type": "LoadImage", "inputs": {"image": ""}},
        "17": {
            "class_type": "ReferenceMultiLatent",
            "inputs": {
                "positive": ["4", 0],
                "negative": ["32", 0],
                "vae": ["3", 0],
                "images.image0": ["11", 0],
                "resolution": "resize",
                "resolution.megapixels": 1,
            },
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["17", 0],
                "negative": ["17", 1],
                "latent_image": ["17", 2],
                "seed": seed,
                "steps": 16,
                "cfg": 1,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1,
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "unbake/keyframe"}},
    }


def describe_graph(question: str) -> dict:
    """Qwen3-VL over a contact sheet of the shot. Node 2 takes the sheet."""
    return {
        "1": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": DESCRIBE_TEXT, "type": "ideogram4", "device": "default"},
        },
        "2": {"class_type": "LoadImage", "inputs": {"image": ""}},
        "3": {
            "class_type": "TextGenerate",
            "inputs": {
                "clip": ["1", 0],
                "image": ["2", 0],
                "prompt": question,
                "max_length": 300,
                "sampling_mode": "on",
                "sampling_mode.temperature": 0.2,
                "sampling_mode.top_k": 64,
                "sampling_mode.top_p": 0.95,
                "sampling_mode.min_p": 0.05,
                "sampling_mode.repetition_penalty": 1.05,
                "sampling_mode.seed": 0,
                "sampling_mode.presence_penalty": 0,
                "thinking": False,
                "use_default_template": True,
            },
        },
        "4": {"class_type": "PreviewAny", "inputs": {"source": ["3", 0]}},
    }
