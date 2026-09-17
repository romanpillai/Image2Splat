# Image2Splat

**One photo in, a 3D Gaussian splat out.** A local web tool that lifts a single
photograph into a point cloud, flies a camera around it, has a video model turn
that fly-around into a photoreal clip, and trains a Gaussian splat from the clip
with the exact camera poses it was authored with.

> **Beta, for developers and tinkerers.** Tested on Windows with an NVIDIA GPU.
> Video generation runs on [fal.ai](https://fal.ai) and is **paid per clip** on
> your own account.

## How it works

The app walks you through six steps in a left-hand rail, with a 3D viewport on
the right that previews everything live.

| Step | What happens |
|---|---|
| **1 Source** | Drop a photo. MoGe-2 measures the lens automatically. |
| **2 Cloud** | The photo is lifted into a metric point cloud (MoGe-2). Clean it up or isolate the subject; it re-lifts as you change settings. |
| **3 Shot** | Author the camera orbit (length, sweep, radius, aim height, lens). The server renders a *control video* of the point cloud from exactly those cameras. |
| **4 Generate** | The control video and your photo go to a fal.ai video model (LTX 2.3, Wan 2.2 VACE, Wan 3.0, MiniMax H3). An approval window shows exactly what will be sent before anything is paid for. |
| **5 Review** | Watch the clip, cut a matte of the subject (BiRefNet, SAM 2.1, RMBG-1.4, optional MatAnyone), retime if needed, approve. |
| **6 Train** | A COLMAP dataset is built with the authored poses and [Brush](https://github.com/ArthurBrussee/brush) trains the splat. Checkpoints appear in the viewport as they land. |

A second **Passes** tab renders extra orbits (above and below) from the trained
splat, to generate more views and refine it.

Every file the tool reads, writes or uploads is logged with its full path in the
console at the bottom.

The full story, with the dead ends, the measurements and the lessons, is in the
write-up: **[One image in, a Gaussian splat out](blog-post/img2splat-one-image-to-gaussian-splat.md)**.

## Requirements

- **Windows 10 or 11.** Other platforms are untested (Brush paths and some
  defaults assume Windows).
- **NVIDIA GPU**, 12 GB of VRAM recommended, with a recent driver.
- **Python 3.12**
- **Git** (two packages install straight from GitHub)
- **A fal.ai account and API key** -> https://fal.ai/dashboard/keys
- **Brush**, the Gaussian splat trainer -> https://github.com/ArthurBrussee/brush/releases
- Several GB of free disk space for Python packages and model downloads.

## Install

In PowerShell:

```powershell
git clone https://github.com/romanpillai/image2SPLAT.git
cd image2SPLAT

py -3.12 -m venv .venv
.venv\Scripts\activate

# 1. PyTorch with CUDA. If your driver is older, pick a matching build at
#    https://pytorch.org/get-started/locally/
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu129

# 2. Everything else
pip install -r requirements.txt

# 3. Optional: MatAnyone matting and Rerun training telemetry
pip install -r requirements-optional.txt
```

## Set up

**1. Your fal.ai key.** Copy the template and put your key in it:

```powershell
copy .env.example .env
notepad .env
```

`.env` stays on your machine; it is listed in `.gitignore`. A `FAL_KEY`
environment variable works too, and takes precedence.

**2. Brush.** Download the Windows build from the
[Brush releases](https://github.com/ArthurBrussee/brush/releases) and put the
executable at `brush\brush_app.exe` inside this folder (rename it if the release
uses a different file name). Keeping it somewhere else? Copy
`config.example.json` to `config.json` and set `"brush"` to its path.

## Run

```powershell
.venv\Scripts\activate
python run.py
```

Open **http://127.0.0.1:8771**. The **Environment** chip in the header turns red
if the fal key or Brush is missing.

The first lift is slow: MoGe-2 downloads and loads once. After that, depth is
cached per photo and re-lifts take well under a second.

The server has no auto-reload. Restart it after changing any `.py` file, and
hard-refresh the browser (Ctrl+F5) after changing anything in `web/`.

## Using it

- Drop a photo in **Source**. The lens is measured and the points are lifted
  without any button presses; **Cloud** opens by itself.
- Anything that runs shows a working card over the viewport with its progress,
  then a tick, a warning or a cross when it ends. Buttons show busy / done /
  failed on themselves.
- Playhead under the viewport: **← →** step a frame (Shift for ten), **Space**
  plays, **Home / End** jump.
- **Generate** re-renders an out-of-date control video for you, then opens the
  approval window. Nothing is uploaded until you press *Approve & send* there.
- Projects live in `projects\<name>\` (not committed). Use the **⋯** menu to
  rename, copy, export or import a project.

## Configuration

`config.json` is optional. Every key has a default:

| Key | Default | Meaning |
|---|---|---|
| `brush` | `./brush/brush_app.exe` | The Brush executable |
| `projects` | `./projects` | Where projects are stored |
| `python` | the running interpreter | Shown in the Environment panel |
| `host` | `127.0.0.1` | Keep it local: the server has no authentication |
| `port` | `8771` | Change it if the port is taken |
| `log_ring` | `5000` | Console lines kept in memory |

## Models and licences

No model weights are stored in this repository. Each is downloaded from its
original source the first time a feature needs it, and each has **its own
licence**. Read them before any commercial use.

| Model | Used for | Source |
|---|---|---|
| MoGe-2 | Depth, point cloud, lens | [microsoft/MoGe](https://github.com/microsoft/MoGe), `Ruicheng/moge-2-vitl-normal` |
| BiRefNet / BiRefNet-HR | Subject matte | `ZhengPeng7/BiRefNet`, `ZhengPeng7/BiRefNet_HR` |
| SAM 2.1 + Grounding DINO | Tracked and text-prompted matte | `facebook/sam2.1-hiera-small`, `IDEA-Research/grounding-dino-tiny` |
| RMBG-1.4 | Fast matte | `briaai/RMBG-1.4` |
| MatAnyone *(optional)* | Soft-edged video matte | `PeiqingYang/MatAnyone`. **Non-commercial** (NTU S-Lab 1.0) |
| fal.ai video models | Photoreal clip from the control video | Remote, billed by fal.ai under its terms |
| Brush | Splat training | Separate download |

## Troubleshooting

- **"FAL_KEY is not set"**: check `.env` is next to `run.py`, then restart the server.
- **"Brush not found"**: check the path shown in the Environment panel.
- **CUDA out of memory**: close other GPU apps, or use **⋯ → Free VRAM**.
- **Port already in use**: set another `port` in `config.json`.
- **Something failed**: the console at the bottom has the full error and every
  file path involved. The `error` filter shows only failures.

## Project layout

```
run.py              starts the server
server/             FastAPI routes, job runner, config
steps.py            geometry, rendering, depth, matting, datasets
falclient.py        fal.ai engines and prompts
web/                the browser app (plain ES modules, no build step)
web/vendor/         three.js, PlayCanvas, GSAP
blog-post/          the write-up and its screenshots
projects/           your work (created on first run, not committed)
```

## Credits

Built on [three.js](https://threejs.org) (MIT),
[PlayCanvas](https://github.com/playcanvas/engine) (MIT),
[GSAP](https://gsap.com) (GreenSock standard no-charge licence),
[FastAPI](https://fastapi.tiangolo.com),
[Hugging Face Transformers](https://github.com/huggingface/transformers),
[MoGe](https://github.com/microsoft/MoGe) and
[Brush](https://github.com/ArthurBrussee/brush).

## Licence

The code in this repository is released under the [MIT licence](LICENSE). That
covers this project's own code only, not the models it downloads or the
third-party libraries in `web/vendor/`, which keep their own licences.
