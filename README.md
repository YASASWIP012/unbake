# Unbake

Unbake turns a video effect into a recipe you can read, edit and run on your own photo.

Drop in a short clip. Unbake measures what the camera did, how the picture changes, the color, the pacing and the sound, then deletes the clip. What stays is a small JSON recipe with no pixels from the source. The camera preview moves your photo along the recipe's path and measures the result the same way it measured the clip. That runs free on a laptop CPU.

**Status: real renders work on a free GPU.** LTX-2.5 with the Cameraman IC-LoRA has run through ComfyUI, `comfy-api-proxy` and the Comfy SDK on a free Kaggle T4. A 4-second clip at 960x544 takes about 21 minutes. Analysis, recipes, splicing and the camera preview run on a laptop CPU. See [what has and hasn't been tested](#what-has-and-hasnt-been-tested).

Try the camera preview in your browser: [yasaswip012.github.io/unbake](https://yasaswip012.github.io/unbake/). No install, no AI model, nothing uploaded.

To reproduce a render, import [`notebooks/unbake_free_gpu.ipynb`](notebooks/unbake_free_gpu.ipynb) into Kaggle, turn on a T4 GPU and internet, add a Hugging Face token as `HF_TOKEN`, and run every cell. The clip-to-render graph is [`unbake_clip_to_render_api.json`](examples/workflows/unbake_clip_to_render_api.json).

Entry for Comfy's [Open Source Higgsfield](https://comfy.org/open-source-higgsfield) challenge.

## How it works

```
clip ──▶ analyze ──▶ recipe.json ──▶ compile ──▶ LTX-2.5 render ──▶ match report
         (local CPU)   camera         ComfyUI      through the        layer by layer
                       transform      API graph    Comfy SDK               │
                       look                             ▲                  │
                       pacing                           └── fix what missed┘
                       sound
```

The camera layer is what makes this different from copying a reference clip. Unbake tracks the clip's background, fits the camera's motion, and stores it as curves. To render, it moves your photo along those curves to make a guide video and hands that to the Cameraman IC-LoRA. The reference is gone by then, so you can edit the curves, scale the move, or take the camera from one recipe and the look from another.

| Layer | How it's measured | How it's used |
|---|---|---|
| Camera | Feature tracking and a RANSAC motion fit, verified by chaining the fit over 8 frames and checking the picture lines up | Guide video for the IC-LoRA |
| Transform | Color and structure change between the shot's ends, words from Qwen3-VL | End keyframe from FLUX 3 Klein |
| Look | Lab color statistics and a palette | Grade applied to the render |
| Pacing | Motion energy over time | Compared in the match report |
| Sound | Loudness envelope and hits | Described in the LTX prompt |

Unbake says when it can't tell. A clip whose camera can't be separated from the scene, like drifting clouds, gets a warning on the recipe. When the camera travels too far for the first and last frames to show the same things, Unbake won't report a transform and asks you to describe one.

## Run it

You need [uv](https://docs.astral.sh/uv/) and ffmpeg.

```bash
git clone https://github.com/YASASWIP012/unbake
cd unbake
cp .env.example .env
uv run unbake
```

Open http://localhost:8420.

Analyzing, editing, splicing, exporting and the camera preview all run on your CPU with no account or key. Rendering needs a Comfy API key in `.env`, or `COMFY_BASE_URL` pointed at a local [comfy-api-proxy](https://github.com/Comfy-Org/comfy-api-proxy) to use your own GPU. The code is the same either way.

## Models

All open weights: LTX-2.5 22B distilled with the Cameraman v2 IC-LoRA, FLUX 3 Image Klein, and Qwen3-VL. The render graph follows the V2V IC-LoRA template in ComfyUI-LTXVideo. The Cameraman LoRA isn't in this repo, and its model card says it's for research only and not for commercial use, so check that before using it in paid work.

## Inside a ComfyUI graph

`custom_nodes/comfyui_unbake` adds six nodes. Unbake Recipe holds a recipe's JSON, Unbake Guide Frames moves an image along its camera path, and Unbake Match Frames scores the render inside the graph. Unbake Clip, Unbake Camera Guide and Unbake Match Score do the same jobs with video files. Install this package into ComfyUI's Python and link the folder into `ComfyUI/custom_nodes`. `examples/workflows/unbake_clip_to_render_api.json` runs the whole loop in one queue, from a source video through Unbake Clip to a scored render, and made `lake_clip_to_render.mp4`. `examples/workflows/unbake_nodes_api.json` starts from a recipe embedded in the graph. Put `examples/photos/chainbridge.jpg` in ComfyUI's input folder, load the workflow and queue it to reproduce `bridge_sweeppan.mp4`.

## What has and hasn't been tested

- `uv run pytest` covers 20 tests with no GPU or network: known camera moves are recovered, a recipe rebuilt on another picture still matches, a wrong move scores low, and the fix loop stops when a render lands.
- On the four example recipes, the camera preview scored 79 (crash zoom), 67 (road tracking), 94 (city time-lapse) and 50 (cloud drift, which Unbake flags as unreliable).
- The analyzer has been run on the four real clips below. The camera confidence cutoff of 0.7 was chosen on those four, so treat it as a starting value.
- **Real guided renders.** All of these are in `examples/renders`, at 960x544, and none has been graded. The first four have 97 frames, the last three 65. Scores were taken on the full clips, before any trimming.

  | Render | Recipe | Photo | Camera travel landed | Overall |
  |---|---|---|---|---|
  | `bridge_sweeppan.mp4` | Sweeping pan | Chain Bridge | x 100% | 99 |
  | `road_roadtrack.mp4` | Road tracking | Road | zoom 100%, roll 104%, x 154%, y 50% | 61 |
  | `lighthouse_roadtrack.mp4` | Road tracking | Lighthouse | zoom 102%, roll 98%, x 160%, y 50% | 59 |
  | `lighthouse_crashzoom.mp4` | Crash zoom | Lighthouse | zoom 102%, x 40%, y 2%, roll 27% | 40 |
  | `forest_sweeppan.mp4` | Sweeping pan | Misty forest | x 100% | 99 |
  | `lake_clip_to_render.mp4` | Measured in the graph | Mountain lake | zoom 112%, x 75%, roll 71% | 72 |
  | `wave_slowpush.mp4` | Slow push-in | Ocean wave | zoom 105%, x 12%, roll 59% | 51 |

  Renders of still scenes stay close to the guide, so a large move can look like a repainted zoom, and the road render goes soft at the end. Scenes the prompt sets in motion do better: the surf in the wave clip is generated. The final frames of two clips broke down, so `lake_clip_to_render.mp4` stops at frame 54 and `wave_slowpush.mp4` at frame 57. Water in the bridge render changes over time, which a warped still wouldn't do. `bridge_sweeppan.mp4` was built and scored inside one graph by the Unbake nodes, using `examples/workflows/unbake_nodes_api.json`. The scores come from Unbake's own measurement, which is weak on low-contrast pictures like the lighthouse sky.
- **Prompt alone, for comparison.** `examples/renders/road_ltx25.mp4` came from the [Lightricks/LTX-2.5](https://huggingface.co/spaces/Lightricks/LTX-2.5) public demo with a compiled prompt and no guide video. It landed 17% of the zoom and scored 21 for camera and 19 overall.
- The fix loop hasn't produced a valid result on a real render yet. A first-frame check rejects broken renders, and that check has only run in tests.
- Unbake Camera Guide and Unbake Match Score load in ComfyUI, but no graph has used them. I haven't run the Comfy Cloud path, or confirmed that it hosts the Cameraman LoRA. The FLUX 3 Klein and Qwen3-VL graphs haven't run, and renders have no matched sound.
- Comfy publishes workflows from inside its own UI, so Export gives you the recipe and graph as a zip to import there.

## Example clips

The recipes in `examples/recipes` were measured from these Wikimedia Commons clips. The clips themselves aren't in this repo.

- Crash zoom: [Crash zoom on cigarette](https://commons.wikimedia.org/wiki/File:Crash_zoom_on_cigarette.webm), Sarrah Anderson, CC BY 3.0
- Road tracking shot: [Tracking shot Moelltal Strasse](https://commons.wikimedia.org/wiki/File:Tracking_shot_Moelltal_Strasse.ogv), Jarlhelm (author as assumed by Commons), CC BY-SA 3.0
- City time-lapse: [Manaus Time-Lapse](https://commons.wikimedia.org/wiki/File:Manaus_Time-Lapse.webm), Landsat / Copernicus, CC BY 3.0
- Cloud drift: [Cloud time lapse 1280](https://commons.wikimedia.org/wiki/File:Cloud_time_lapse_1280.ogv), Error406, public domain
- Sweeping pan: [Rendszerváltó Nagykoncert panning shot](https://commons.wikimedia.org/wiki/File:Rendszerv%C3%A1lt%C3%B3_Nagykoncert_panning_shot.webm), CC BY 4.0. The original travels five frame widths, so the recipe keeps 5% of the measured move.
- Slow push-in: [Sequence 4 NW Road pan right to fence](https://commons.wikimedia.org/wiki/File:Sequence_4_NW_Road_pan_right_to_fence_(49717705892).webm), public domain

Photos: [Misty coniferous forest on a slope](https://commons.wikimedia.org/wiki/File:Misty_coniferous_forest_on_a_slope_(Unsplash).jpg), [Big ocean wave splash](https://commons.wikimedia.org/wiki/File:Big_ocean_wave_splash_(Unsplash).jpg), [Mountain lake dam](https://commons.wikimedia.org/wiki/File:Mountain_lake_dam.jpg), [Cihou Lighthouse in Evening](https://commons.wikimedia.org/wiki/File:Cihou_Lighthouse_in_Evening_20140508.jpg) and [Széchenyi Chain Bridge in Budapest at night](https://commons.wikimedia.org/wiki/File:Sz%C3%A9chenyi_Chain_Bridge_in_Budapest_at_night.jpg), all CC0.

`examples/renders/lighthouse_splice.mp4` is the lighthouse render graded with the city time-lapse's look. `web/example.jpg` is a frame from the Manaus Time-Lapse clip above. `examples/photos/road.png` is a frame of the Moelltal tracking shot, and `examples/renders/road_ltx25.mp4` and `road_roadtrack.mp4` start from it, so all three carry that clip's CC BY-SA 3.0 license.

## License

MIT
