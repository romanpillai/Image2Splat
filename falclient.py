"""fal.ai LTX-2.3 render-to-real wrapper.

Endpoint: fal-ai/ltx-2.3-quality/render-to-real

Its schema is exactly the shape of this pipeline:
  video_url            required — the CG / 3D render to make photoreal (our orbit)
  image_url            optional — reference for FRAME 0, defines the photoreal look
  prompt               must carry the 3DREAL trigger; fal guarantees it is present
  intensity            light | strong | strong-v2   (light keeps more of the render)
  enable_detail_refine extra 2x spatial upscale + detailer pass

Credentials come from the FAL_KEY environment variable only — nothing is written
to disk by this module.
"""
from __future__ import annotations

import os
import time as _time
from pathlib import Path

ENDPOINT = "fal-ai/ltx-2.3-quality/render-to-real"
TRIGGER = "3DREAL"

# Documented ceilings on OUTPUT DURATION, which is num_frames / frames_per_second.
# They are caps, not fixed lengths: 100 frames at 24 fps is 4.17 s and sits
# comfortably inside either. Frame count is the meaningful setting, since every
# frame is one camera pose.
MAX_SECONDS = {"720p": 6.0, "480p": 15.0}

# --- second engine -------------------------------------------------------
# Wan 3.0 Prime reference-to-video. NOT pose-exact: see this module's notes.
WAN_ENDPOINT = "alibaba/wan-3.0-prime/reference-to-video"
# Wan 2.2 VACE, depth task. The one endpoint here that takes a per-frame
# CONTROL signal rather than a loose reference, and the only one that can be
# told to match the input's frame count -- which is what makes pose i and
# frame i line up without a retime.
VACE_ENDPOINT = "fal-ai/wan-22-vace-fun-a14b/depth"
# The schema declares no maxItems; 8 is a self-imposed sanity cap.
VACE_MAX_REF_IMAGES = 8
VACE_RESOLUTIONS = ("auto", "240p", "360p", "480p", "580p", "720p")
VACE_SAMPLERS = ("unipc", "dpm++", "euler")
VACE_FRAME_RANGE = (81, 241)
WAN_RESOLUTIONS = ("480p", "720p", "1080p")
WAN_ASPECTS = ("adaptive", "16:9", "4:3", "1:1", "3:4", "9:16")
WAN_MAX_REF_SECONDS = 15.0   # total across reference_video_urls
WAN_DURATION_RANGE = (2, 30)
WAN_MAX_REF_IMAGES = 10   # schema maxItems
# The endpoint exposes no fps input. MEASURED from real returns: 5s came back
# as 150 frames and 7s as 210 -- both exactly 30 fps, with the duration
# honoured precisely. Rendering the orbit at this fps makes the returned frame
# count match the pose count with no retiming at all.
WAN_OUTPUT_FPS = 30

# MiniMax Hailuo H3, reference-to-video. Same CLASS as Wan 3.0 Prime: the
# control video is a motion REFERENCE, not a driving signal, so frame i is not
# authored pose i. Its reason to exist here is resolution -- it is the only
# engine offering 2K and 4K, and low output resolution has been a real limit on
# splat detail (a 448x640 clip left the subject at 27% of the pixels).
MINIMAX_ENDPOINT = "minimax/h3/reference-to-video"
MINIMAX_RESOLUTIONS = ("480P", "768P", "2K", "4K")   # the API is uppercase
MINIMAX_ASPECTS = ("adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16")
MINIMAX_DURATION_RANGE = (5, 15)          # whole seconds
MINIMAX_MAX_REF_VIDEOS = 3
MINIMAX_MAX_REF_IMAGES = 9          # schema maxItems
MINIMAX_EXPANSION = ("fast", "balanced", "quality")
# Not documented as a contract; recorded when the first clip comes back.
MINIMAX_OUTPUT_FPS = None


def wan_exact_duration_options(frames: int, fps: int) -> dict:
    """Can this orbit be asked for as a whole number of seconds?

    Wan takes integer seconds, so frames/fps must already be an integer for
    the request to match the control video exactly. When it does not, offer
    the two ways to fix it: keep the frame count and change fps, or keep fps
    and change the frame count.
    """
    secs = frames / max(fps, 1)
    exact = abs(secs - round(secs)) < 1e-9
    out = {"seconds": round(secs, 4), "exact": exact,
           "request": max(WAN_DURATION_RANGE[0],
                          min(WAN_DURATION_RANGE[1], int(round(secs))))}
    if not exact:
        target = out["request"]
        out["fps_fix"] = (round(frames / target, 4)
                          if target and frames / target == int(frames / target)
                          else None)
        # frame counts near the current one that divide evenly by fps
        out["frames_fix"] = target * fps
        out["error_pct"] = round(100 * abs(secs - target) / max(secs, 1e-9), 2)
    return out

ENGINES = {
    "ltx": {
        "label": "LTX 2.3 render-to-real",
        "endpoint": ENDPOINT,
        "pose_exact": True,
        "output_fps": None,      # honours the fps we send
        "note": "Follows the control video frame by frame.",
    },
    "wan22vace": {
        "label": "Wan 2.2 VACE (depth control)",
        "endpoint": VACE_ENDPOINT,
        "pose_exact": True,
        "output_fps": None,
        "note": "Follows the control video frame by frame.",
    },
    "wan": {
        "label": "Wan 3.0 Prime reference-to-video",
        "endpoint": WAN_ENDPOINT,
        "pose_exact": False,
        "output_fps": WAN_OUTPUT_FPS,   # measured; not a documented contract
        "note": "Uses the control video loosely.",
    },
    "minimax": {
        "label": "MiniMax H3 reference-to-video (2K/4K)",
        "endpoint": MINIMAX_ENDPOINT,
        "pose_exact": False,
        "output_fps": MINIMAX_OUTPUT_FPS,
        "note": "Uses the control video loosely. Up to 4K.",
    },
}


def submit_vace(video_path: Path, prompt: str,
                ref_image: Path | None = None,
                first_frame: Path | None = None,
                num_frames: int | None = None,
                match_input_frames: bool = True,
                fps: int = 16, resolution: str = "720p",
                steps: int = 30, guidance: float = 5.0,
                sampler: str = "unipc", seed: int | None = None,
                negative_prompt: str = "", scene: bool = False,
                extra_images=None, pass_view: str = "",
                on_log=None) -> dict:
    """Blocking submit to Wan 2.2 VACE (depth task).

    Unlike the other two engines the control video drives EVERY frame, so the
    output follows the authored camera rather than reinterpreting it. Frame
    count is matched to the input, which is the property the retime step
    exists to repair on the other engines.
    """
    if resolution not in VACE_RESOLUTIONS:
        raise ValueError(f"unknown resolution {resolution!r} for VACE "
                         f"(expected one of {VACE_RESOLUTIONS})")
    if sampler not in VACE_SAMPLERS:
        raise ValueError(f"unknown sampler {sampler!r} "
                         f"(expected one of {VACE_SAMPLERS})")
    lo, hi = VACE_FRAME_RANGE
    if num_frames is not None and not (lo <= int(num_frames) <= hi):
        raise ValueError(
            f"VACE takes {lo}-{hi} frames; this orbit is {num_frames}. "
            f"Change the frame count in Orbit.")

    fc = _client()
    args = {
        "prompt": _log_prompt(on_log, prompt, scene, pass_view),
        "video_url": upload(video_path, on_log, "control video"),
        "resolution": resolution,
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance),
        "sampler": sampler,
        "match_input_num_frames": bool(match_input_frames),
        "frames_per_second": int(max(5, min(30, fps))),
    }
    if not match_input_frames and num_frames is not None:
        args["num_frames"] = int(num_frames)
    if negative_prompt.strip():
        args["negative_prompt"] = negative_prompt.strip()
    if seed is not None:
        args["seed"] = int(seed)
    if ref_image is not None and Path(ref_image).exists():
        args["ref_image_urls"] = _ref_urls(ref_image, extra_images, on_log,
                                           VACE_MAX_REF_IMAGES)
    if first_frame is not None and Path(first_frame).exists():
        args["first_frame_url"] = upload(Path(first_frame), on_log,
                                         "first frame")

    seen = {"key": None}

    def _on_q(update):
        if on_log is None:
            return
        key = (type(update).__name__, getattr(update, "position", None))
        if key != seen["key"]:
            seen["key"] = key
            pos = key[1]
            where = f", queue position {pos}" if pos is not None else ""
            on_log(f"fal: {key[0]}{where}")

    if on_log:
        on_log(f"fal: {VACE_ENDPOINT} | {resolution} | {steps} steps | "
               f"guidance {guidance} | {sampler}"
               + (" | matching input frame count" if match_input_frames
                  else f" | {num_frames} frames"))
    return fc.subscribe(VACE_ENDPOINT, arguments=args,
                        with_logs=True, on_queue_update=_on_q)


def load_dotenv(path: Path | None = None) -> str | None:
    """Populate FAL_KEY from a local .env if it is not already in the env.

    A real environment variable always wins, so exporting the key still works
    and never gets silently overridden by a stale file. Deliberately dependency
    free -- KEY=value lines, # comments, optional quotes.
    """
    p = Path(path) if path else Path(__file__).parent / ".env"
    if not p.exists():
        return None
    # utf-8-sig, not utf-8: Windows editors and PowerShell's Out-File write a
    # BOM, which would otherwise make the first key parse as "﻿FAL_KEY"
    # and never match.
    for raw in p.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and not os.environ.get(k):
            os.environ[k] = v
    return str(p)


def have_key() -> bool:
    return bool(os.environ.get("FAL_KEY"))


def key_source() -> str:
    if not have_key():
        return "not set"
    return "environment" if _ENV_PRELOADED else ".env"


_ENV_PRELOADED = bool(os.environ.get("FAL_KEY"))
load_dotenv()


def _client():
    if not have_key():
        raise RuntimeError(
            "FAL_KEY is not set. Set it in the shell that launches this server, "
            "e.g.  $env:FAL_KEY = \"...\"  then restart."
        )
    import fal_client
    return fal_client


def _ref_urls(main: Path | None, extras, on_log, cap: int) -> list[str]:
    """Upload the reference images, MAIN FIRST, and return their URLs.

    Order is load-bearing: these endpoints treat the first entry as the
    primary appearance anchor, so the deliberately-chosen reference has to
    lead and the extras follow. Capped here as well as server-side because
    this function is what actually builds the request.
    """
    urls = []
    if main is not None and Path(main).exists():
        urls.append(upload(Path(main), on_log, "reference image"))
    elif main is not None and on_log:
        on_log(f"fal: WARNING reference {main} missing - "
               "sending no appearance anchor")
    for i, q in enumerate(extras or []):
        if len(urls) >= cap:
            if on_log:
                on_log(f"fal: {len(extras) - i} extra reference(s) dropped - "
                       f"this endpoint takes {cap}")
            break
        if Path(q).exists():
            urls.append(upload(Path(q), on_log, f"extra reference {i + 1}"))
        elif on_log:
            on_log(f"fal: WARNING extra reference {q} missing - skipped")
    return urls


def _pass_clause(pass_view: str) -> str:
    """The camera-angle clause for an orbit pass, or "" for a normal render.

    Placed SECOND, immediately after the trigger, because the angle is the one
    thing distinguishing a top pass from a bottom one -- until this existed the
    two sent identical camera language and the model had no way to know which
    it was making.

    Deliberately vague about height rather than quoting a computed elevation:
    "from above, looking down" is a thing a video model has seen a million
    examples of, where "orbiting at 31 degrees" is not.
    """
    v = (pass_view or "").lower()
    if v == "top":
        return ", the camera orbiting from above, looking down"
    if v == "bottom":
        return ", the camera orbiting from below, looking up"
    return ""


def build_prompt(user_prompt: str, scene: bool = False,
                 pass_view: str = "") -> str:
    """Ensure the 3DREAL trigger leads, then the caller's description.

    Two boilerplates, because they say opposite things and using the wrong one
    fights the control video:

    grey  -- forbids scenery outright. A first attempt sent the RGBA cutout as
             image_url; fal flattened the alpha, LTX saw the photograph's
             original background and reproduced it for the whole clip. A
             background that stays fixed while the camera orbits has no 3D
             solution, so it must not appear at all.

    scene -- the control video now contains a floor and marker geometry that
             DOES move with correct parallax. Telling the model "no scenery,
             no floor detail" here would contradict what it is being shown.

    Both now name the FLASH. The grey text used to ask for "even soft studio
    lighting" while the description it was concatenated with called for a hard
    direct frontal flash -- one prompt asking for two incompatible lightings,
    with the model free to split the difference. The scenery rules are left
    exactly as they are: they are already conditional, and "no scenery" is
    correct for a grey backdrop precisely because a background that stays fixed
    while the view orbits has no 3D solution.
    """
    p = (user_prompt or "").strip()
    if p.upper().startswith(TRIGGER):
        return p
    # LIGHTING LEADS. It used to sit at clause 12 of 23, behind six
    # consecutive negatives, and it did not survive into the output. A video
    # model weights what comes first, so the flash is now the second thing the
    # prompt says and the negatives are last, where a negation belongs.
    #
    # The word "studio" is deliberately gone from the grey backdrop: it named
    # the BACKDROP but reads as studio LIGHTING, which is soft and even and the
    # opposite of what is wanted. The soft-light negations are explicit for the
    # same reason -- there is no fill, the flash is the whole lighting setup.
    if scene:
        base = (f"{TRIGGER} photorealistic 360 degree turntable"
                f"{_pass_clause(pass_view)}"
                ", the subject "
                "lit by a single harsh direct on-axis flash as its dominant "
                "light source, the flash highlight centred on the subject, "
                "blown-out speculars on every glossy surface, deep shadow "
                "falling away immediately behind it, hard contrast, the "
                "camera orbits the subject at constant speed, consistent "
                "identity, costume and materials from every angle, the "
                "subject stands on a solid ground plane in a coherent "
                "physical environment, the ground and surroundings hold still "
                "in the world and sweep past with correct perspective as the "
                "camera moves around, ambient light from the environment only "
                "behind the subject, no soft fill on the subject, no diffused "
                "light on the subject, no camera shake, smooth constant-speed "
                "rotation")
    else:
        base = (f"{TRIGGER} photorealistic 360 degree turntable"
                f"{_pass_clause(pass_view)}"
                ", lit by a single harsh direct on-axis flash as the "
                "ONLY light source, the flash highlight centred on the "
                "subject, blown-out speculars on every glossy surface, deep "
                "black shadow falling away immediately behind, extreme "
                "contrast between the brightly lit subject and the dark "
                "background, flash photography look, consistent identity, "
                "costume and materials from every angle, isolated on a "
                "completely plain seamless flat neutral grey backdrop, empty "
                "background, no scenery, no landscape, no sky, no horizon, "
                "no floor detail, no props, no soft lighting, no studio "
                "lighting, no ambient fill, no diffused light, no rim light, "
                "no camera shake, smooth constant-speed rotation")
    return f"{base}. {p}" if p else base


def _log_prompt(on_log, prompt: str, scene: bool,
                pass_view: str = "") -> str:
    """Build the prompt AND log the exact string fal receives.

    The boilerplate is invisible from the UI -- the description box shows your
    words, but what is actually sent is a fixed preamble with your text glued
    on the end. Two things depend on knowing the difference: whether the
    boilerplate contradicts the description (it asked for "even soft studio
    lighting" while the description asked for a hard flash), and WHERE in the
    string each instruction sits, since a video model weights what comes first.

    Logged clause by clause and numbered, because order is the thing being
    inspected and a single 90-word line hides it.
    """
    full = build_prompt(prompt, scene=scene, pass_view=pass_view)
    if on_log:
        kind = "scene/panorama" if scene else "grey"
        if pass_view:
            kind += f" + {pass_view} pass"
        parts = [c.strip() for c in full.split(",") if c.strip()]
        on_log(f"fal prompt [{kind} boilerplate + your description] "
               f"{len(full.split())} words, {len(parts)} clauses, EXACTLY as "
               f"sent:")
        for i, c in enumerate(parts, 1):
            mark = ""
            low = c.lower()
            if "flash" in low or "lighting" in low or "lit by" in low:
                mark = "   <-- LIGHTING"
            on_log(f"    {i:2d}. {c}{mark}")
        on_log(f"fal prompt: full string -> {full}")
    return full


# Every file this module hands to fal, recorded where the handing actually
# happens. Anything reconstructed from the UI's intentions could disagree with
# what went; this cannot, because it IS the call.
SENT: list = []


def sent_reset():
    SENT.clear()


def sent_list() -> list:
    return list(SENT)


def upload(path: Path, on_log=None, role: str = "") -> str:
    fc = _client()
    mb = path.stat().st_size / 1e6
    t0 = _time.time()
    url = fc.upload_file(str(path))
    el = _time.time() - t0
    SENT.append({"role": role or "input", "name": Path(path).name,
                 "path": str(path), "url": url,
                 "bytes": int(Path(path).stat().st_size)})
    if on_log:
        on_log(f"file: sending {path} -> fal ({mb:.2f} MB in {el:.1f}s "
               f"= {mb * 8 / max(el, 1e-6):.1f} Mbit/s)")
        # The URL is what fal actually reads. When a render comes back wrong
        # this is the only way to confirm which bytes it was given.
        on_log(f"file: fal now holds it at {url}")
    return url


# --------------------------------------------------------- auto-describe ---
# fal-ai/any-llm/vision is deprecated; openrouter/router/vision is the current
# vision endpoint and fronts Gemini / GPT / Claude / Qwen etc.
VISION_ENDPOINT = "openrouter/router/vision"
# Every one of these must accept IMAGE input -- this endpoint is handed a
# picture, and a text-only model returns a confident description of nothing.
# The two Claude 5 ids were checked against OpenRouter's own endpoint listing
# rather than guessed from the naming pattern; both report
# input_modalities ["text", "image", "file"].
VISION_MODELS = [
    "google/gemini-2.5-flash",          # cheap, fast, good at materials
    "google/gemini-2.0-flash-001",
    "openai/gpt-4o-mini",
    "anthropic/claude-3.5-haiku",
    "anthropic/claude-sonnet-5",        # stronger on materials and lighting
    "anthropic/claude-opus-5",          # strongest, and the dearest per call
]

# The boilerplate in build_prompt() already owns motion, backdrop and framing.
# If the model also describes those it will contradict them -- so it is
# constrained to the one thing it is actually needed for: what the subject is
# made of, which is what render-to-real has to keep consistent from every angle.
_VISION_RULES_TAIL = (
    "- Never mention rotation, turning, turntable, camera, angle or motion. "
    "Those are supplied separately.\n"
    "- Never write 'the image shows', 'a photo of', 'this is' or similar.\n"
    "- Be concrete about MATERIALS (brushed chrome, crushed velvet, matte skin, "
    "woven silk, worn leather), because material consistency from every angle "
    "is what the video model most needs to hold on to.\n"
    "- No bullet points, no preamble, no trailing commentary. Output the "
    "description text only."
)

# Subject only. Paired with the grey boilerplate, which forbids scenery.
# Every description carries the same lighting, deliberately. A flash at the
# viewpoint keeps its highlight in the MIDDLE of the subject and lands on the
# same surfaces from every angle -- the one setup that stays self-consistent
# as the view moves round. A light fixed somewhere in the world would sweep
# across the subject instead, and every frame would disagree with the last
# about which side is bright.
#
# Phrasing constraint: _VISION_RULES_TAIL forbids the OUTPUT naming the camera
# or any motion, so this has to read as a fixed condition of the shot rather
# than as a light that follows something.
_FLASH_LIGHTING = (
    "LIGHTING - the same on every description, without exception:\n"
    "The subject is lit by ONE harsh flash square in front of it and by nothing "
    "else. It is the only light source: there is no fill, no ambient, no "
    "soft box, no rim light. Contrast is harsh. A bright frontal burst "
    "lands its highlight in the CENTRE of the subject and falls away fast, "
    "so whatever is behind drops to near black, with a hard-edged shadow "
    "stacked immediately behind.\n"
    "- Say WHERE the light lands, naming the actual materials in front of "
    "you: which surfaces take a small bright specular point (anything "
    "rounded, glossy, wet, beaded or faceted takes one each), which blow out "
    "to pure white (chrome, crystal, polished metal, wet lips, gemstones), "
    "which light up only at their fibre tips (fur, knitwear, feathers, fuzzy "
    "fabric), and which stay matte and swallow it.\n"
    "- Frontal flash flattens the shadows across a face and pushes colour to "
    "high saturation. Say so where it applies.\n"
    "- Call it 'direct frontal flash' or 'flash photography lighting'. "
    "Do NOT use the word camera, and do not describe the light moving, "
    "following or tracking anything - it is a fixed condition of the shot.\n"
)

VISION_SYSTEM_SUBJECT = (
    "You write the subject description for a video model that turns a grey "
    "clay render into photoreal footage of a 360 degree turntable.\n\n"
    "Describe ONLY the subject: what it is, its materials, colours, surface "
    "finish, costume and distinguishing features.\n\n"
    "Rules:\n"
    "- Never mention the background, environment, scenery or floor. They are "
    "supplied separately and your text must not contradict them. LIGHTING is "
    "the exception and is now REQUIRED - see below.\n"
    + _VISION_RULES_TAIL +
    "\n\n" + _FLASH_LIGHTING +
    "\n- Two or three flowing clauses, under 90 words. The extra room "
    "over the old 60 is for the lighting; do not spend it on the subject."
)

# Subject AND setting. Paired with the scene boilerplate: the control video
# carries floor and marker geometry, so the model needs to be told what that
# geometry should look like once it is made photoreal. Saying nothing leaves it
# to invent an environment with no relation to the source image.
# Used whenever the project has a PANORAMA backdrop. The render then already
# contains the environment -- a photographic dome the subject stands inside --
# rather than the grey floor and marker blocks this prompt used to describe.
# That set was removed, and the stale wording was telling the vision model
# about scenery that is no longer in the frame.
# Used whenever the project has a PANORAMA backdrop. The render then already
# contains the environment -- a photographic dome the subject stands inside --
# rather than the grey floor and marker blocks this prompt used to describe.
# That set was removed, and the stale wording was telling the vision model
# about scenery that is no longer in the frame.
VISION_SYSTEM_SCENE = (
    "You write the description for a video model that turns a rough render "
    "into photoreal footage of a camera orbiting a subject. The render shows "
    "the subject standing inside a real environment: a photographic panorama "
    "surrounds it and lights it, and that environment is part of the shot, "
    "not a backdrop to be replaced.\n\n"
    "Write TWO parts in one flowing passage:\n"
    "1. The subject: what it is, materials, colours, surface finish, costume.\n"
    "2. The place it is standing in, as shown: the ground underfoot, what "
    "surrounds it, the time of day and the quality of the light, so the "
    "render's rough surroundings resolve into that same place rather than "
    "something invented.\n\n"
    "Rules:\n"
    "- Describe the setting as solid physical surroundings the subject stands "
    "in, never as a backdrop, wall, photograph or painted scene.\n"
    "- Name the ground surface explicitly (packed earth, wet cobbles, dry "
    "grass, studio floor), since the render's floor has to become that.\n"
    "- Say where the AMBIENT light comes from and what colour it is; the "
    "panorama lights the setting, so the two have to agree.\n"
    "- The subject on top of that carries the flash described below. Both "
    "are true at once: the environment supplies the ambient, the flash "
    "supplies the highlights on the subject itself.\n"
    + _VISION_RULES_TAIL +
    "\n\n" + _FLASH_LIGHTING +
    "\n- Under 140 words."
)


def describe_image(image_path: Path, model: str = VISION_MODELS[0],
                   extra: str = "", scene: bool = False, on_log=None) -> dict:
    """Caption an image into a description for the generation prompt.

    scene=True also describes the SETTING, because the control video's floor and
    marker geometry has to be rendered as something. Left undescribed the model
    invents an environment unrelated to the source image.
    """
    fc = _client()
    url = upload(Path(image_path), role="image to describe")
    ask = ("Describe this image for the video prompt, following the rules."
           if scene else
           "Write the subject description for this render, following the rules.")
    if extra.strip():
        # Framed as a BRIEF, not as context to honour. The old wording --
        # "the user adds this context, honour it" -- let the model treat the
        # text as material to reproduce, so when the box still held a previous
        # caption it re-described the previous subject while looking at a new
        # photo. Describing what is actually in front of it is stated first
        # and stated as the job; the brief only steers emphasis and intent.
        ask += ("\n\nDescribe what you SEE in this image. That is the job, "
                "and nothing below changes it.\n"
                "The user has also written what they want from the result. "
                "Fold it in where it applies -- emphasis, mood, materials, "
                "what matters to them -- but never let it replace or "
                "contradict what is actually in the image. If it mentions "
                "something that is not there, ignore that part.\n"
                f"User's brief: {extra.strip()}")

    def _log(update):
        if on_log is None:
            return
        for entry in getattr(update, "logs", None) or []:
            msg = entry.get("message") if isinstance(entry, dict) else str(entry)
            if msg:
                on_log(msg)

    res = fc.subscribe(VISION_ENDPOINT, arguments={
        "image_urls": [url],
        "prompt": ask,
        "system_prompt": VISION_SYSTEM_SCENE if scene else VISION_SYSTEM_SUBJECT,
        "model": model,
        "temperature": 0.3,          # description, not creative writing
        "max_tokens": 320 if scene else 220,
    }, with_logs=True, on_queue_update=_log)

    text = (res.get("output") or "").strip()
    # models sometimes wrap output in quotes or add a lead-in despite the rules
    for lead in ("Subject description:", "Description:"):
        if text.lower().startswith(lead.lower()):
            text = text[len(lead):].strip()
    text = text.strip('"').strip()
    usage = res.get("usage") or {}
    return {"text": text, "model": model,
            "cost": usage.get("cost"), "tokens": usage.get("total_tokens")}


def submit(video_path: Path, image_path: Path | None, prompt: str,
           intensity: str = "strong-v2", detail_refine: bool = False,
           num_frames: int | None = None, fps: int = 24,
           resolution: str = "720p", video_quality: str = "high",
           scene: bool = False, pass_view: str = "", on_log=None) -> dict:
    """Blocking submit + wait. Returns the raw fal result dict.

    num_frames MUST be passed. It defaults to 121 server-side, while our orbit
    is typically 100 frames -- and every pose is bound to a specific frame
    index, so a returned clip of a different length silently misaligns the
    entire reconstruction.

    resolution is the SHORT SIDE ("480p" | "720p"); the source aspect ratio is
    preserved, which is why the control video's aspect must be a standard one.
    720p allows roughly 6s, 480p roughly 15s.
    """
    if num_frames:
        secs = num_frames / max(fps, 1)
        cap = MAX_SECONDS.get(resolution, 6.0)
        if secs > cap:
            raise ValueError(
                f"{num_frames} frames at {fps} fps is {secs:.2f}s, over the "
                f"{cap:g}s limit for {resolution} (max {int(cap * fps)} frames). "
                "Reduce the orbit frame count or switch resolution — do not let "
                "fal silently return a different frame count, because every "
                "frame is bound to a camera pose."
            )

    fc = _client()
    args = {
        "video_url": upload(video_path, on_log, "control video"),
        "prompt": _log_prompt(on_log, prompt, scene, pass_view),
        "intensity": intensity,
        "enable_detail_refine": bool(detail_refine),
        "frames_per_second": fps,
        "resolution": resolution,
        "video_quality": video_quality,
    }
    if num_frames:
        args["num_frames"] = int(num_frames)
    if image_path is not None and Path(image_path).exists():
        args["image_url"] = upload(Path(image_path), on_log,
                                   "first frame")

    # A QUEUED job produces no log entries, so forwarding only `logs` left the
    # UI silent for the entire queue wait -- which reads as "the upload is
    # slow" when the upload already finished in seconds. Measured: 4.71 MB
    # video uploads in 4.3s. Report the STATUS itself, with elapsed time and
    # queue position when fal supplies one.
    t_sub = _time.time()
    seen = {"state": None}

    def _log(update):
        if on_log is None:
            return
        state = type(update).__name__
        if state != seen["state"]:
            seen["state"] = state
            pos = getattr(update, "position", None)
            where = f", queue position {pos}" if pos is not None else ""
            on_log(f"fal: {state}{where} "
                   f"(+{_time.time() - t_sub:.0f}s since submit)")
        for entry in getattr(update, "logs", None) or []:
            msg = entry.get("message") if isinstance(entry, dict) else str(entry)
            if msg:
                on_log(msg)

    return fc.subscribe(ENDPOINT, arguments=args, with_logs=True,
                        on_queue_update=_log)


def submit_wan(video_path: Path, image_path: Path | None, prompt: str,
               duration_s: int | None, resolution: str = "720p",
               aspect_ratio: str = "adaptive", audio: bool = False,
               prompt_expansion: bool = False, enable_thinking: bool = False,
               seed: int | None = None, scene: bool = False,
               extra_images=None, pass_view: str = "", on_log=None) -> dict:
    """Blocking submit to Wan 3.0 Prime reference-to-video.

    The control video goes in as a REFERENCE, which is the only slot this
    endpoint has for it -- there is no driving-video input and no frame-count
    input. duration is whole seconds (2-30); the returned clip's frame count
    is the model's choice, so the caller must reconcile it against the pose
    count before building a splat.

    audio defaults False and prompt_expansion defaults False here, against
    fal's own defaults of True: generated audio is dead weight for a control
    signal, and prompt rewriting would discard the turntable instructions
    build_prompt() is careful to attach.
    """
    lo, hi = WAN_DURATION_RANGE
    if duration_s is None:
        # fal's "smart duration": the model picks a length from the prompt
        # and the reference media.
        pass
    else:
        # The schema is anyOf[integer 2..30, null] -- whole seconds only, so
        # a fractional request would 422. Round here and let the caller
        # report the difference.
        duration_s = int(round(float(duration_s)))
        if not (lo <= duration_s <= hi):
            raise ValueError(f"duration {duration_s}s is outside Wan's "
                             f"{lo}-{hi}s range")
    if duration_s is not None and duration_s > WAN_MAX_REF_SECONDS:
        raise ValueError(
            f"the control video is {duration_s}s, over Wan's "
            f"{WAN_MAX_REF_SECONDS:g}s reference limit - shorten the orbit")
    if resolution not in WAN_RESOLUTIONS:
        raise ValueError(f"unknown resolution {resolution!r} for Wan "
                         f"(expected one of {WAN_RESOLUTIONS})")
    if aspect_ratio not in WAN_ASPECTS:
        raise ValueError(f"unknown aspect_ratio {aspect_ratio!r} for Wan")

    fc = _client()
    args = {
        "prompt": _log_prompt(on_log, prompt, scene, pass_view),
        "resolution": resolution,
        "aspect_ratio": aspect_ratio,
        "duration": duration_s,
        "audio": bool(audio),
        "enable_prompt_expansion": bool(prompt_expansion),
        "enable_thinking": bool(enable_thinking),
        "reference_video_urls": [upload(video_path, on_log,
                                        "reference video")],
    }
    if seed is not None:
        args["seed"] = int(seed)
    refs = _ref_urls(image_path, extra_images, on_log, WAN_MAX_REF_IMAGES)
    if refs:
        args["reference_image_urls"] = refs

    seen = {"key": None}

    def _on_q(update):
        if on_log is None:
            return
        key = (type(update).__name__, getattr(update, "position", None))
        if key != seen["key"]:
            seen["key"] = key
            pos = key[1]
            where = f", queue position {pos}" if pos is not None else ""
            on_log(f"fal: {key[0]}{where}")

    if on_log:
        _dur = "smart" if duration_s is None else f"{duration_s}s"
        on_log(f"fal: {WAN_ENDPOINT} | {_dur} | {resolution} | "
               f"aspect {aspect_ratio}")
    return fc.subscribe(WAN_ENDPOINT, arguments=args,
                        with_logs=True, on_queue_update=_on_q)


def submit_minimax(video_path: Path, image_path: Path | None, prompt: str,
                   duration_s: int | None, resolution: str = "768P",
                   aspect_ratio: str = "adaptive",
                   prompt_expansion: str = "balanced",
                   seed: int | None = None, scene: bool = False,
                   extra_images=None, pass_view: str = "", on_log=None) -> dict:
    """Blocking submit to MiniMax H3 reference-to-video.

    The control video goes into reference_video_urls, which the model uses as a
    MOTION reference. That is not the same thing as driving each frame: it does
    not take a frame count, and the clip that comes back is the model's own
    length and cadence. Frame i is therefore NOT authored pose i, and a splat
    built straight from it can be misaligned -- retime it against the poses
    first, exactly as with Wan 3.0 Prime.

    prompt_expansion is a MODE here rather than a boolean, and there is no way
    to switch it off. "fast" rewrites the prompt least, so it is the default
    that best preserves the turntable instructions build_prompt() attaches;
    fal's own default is "balanced".
    """
    lo, hi = MINIMAX_DURATION_RANGE
    if duration_s is not None:
        duration_s = int(round(float(duration_s)))
        if not (lo <= duration_s <= hi):
            raise ValueError(f"duration {duration_s}s is outside MiniMax's "
                             f"{lo}-{hi}s range - shorten or lengthen the "
                             f"orbit, or drop the fps")
    if resolution not in MINIMAX_RESOLUTIONS:
        raise ValueError(f"unknown resolution {resolution!r} for MiniMax "
                         f"(expected one of {MINIMAX_RESOLUTIONS})")
    if aspect_ratio not in MINIMAX_ASPECTS:
        raise ValueError(f"unknown aspect_ratio {aspect_ratio!r} for MiniMax")
    if prompt_expansion not in MINIMAX_EXPANSION:
        raise ValueError(f"unknown prompt_expansion_mode {prompt_expansion!r}")

    fc = _client()
    args = {
        "prompt": _log_prompt(on_log, prompt, scene, pass_view),
        "resolution": resolution,
        "aspect_ratio": aspect_ratio,
        "prompt_expansion_mode": prompt_expansion,
        "reference_video_urls": [upload(video_path, on_log,
                                        "reference video (motion)")],
    }
    if duration_s is not None:
        args["duration"] = duration_s
    if seed is not None:
        args["seed"] = int(seed)
    refs = _ref_urls(image_path, extra_images, on_log, MINIMAX_MAX_REF_IMAGES)
    if refs:
        args["reference_image_urls"] = refs

    seen = {"key": None}

    def _on_q(update):
        if on_log is None:
            return
        key = (type(update).__name__, getattr(update, "position", None))
        if key != seen["key"]:
            seen["key"] = key
            pos = key[1]
            where = f", queue position {pos}" if pos is not None else ""
            on_log(f"fal: {key[0]}{where}")

    if on_log:
        _dur = "model's choice" if duration_s is None else f"{duration_s}s"
        on_log(f"fal: {MINIMAX_ENDPOINT} | {_dur} | {resolution} | "
               f"aspect {aspect_ratio} | expansion {prompt_expansion}")
    return fc.subscribe(MINIMAX_ENDPOINT, arguments=args,
                        with_logs=True, on_queue_update=_on_q)


def result_video_url(result: dict) -> str | None:
    """fal returns either {'video': {'url': ...}} or {'videos': [{'url': ...}]}."""
    if not isinstance(result, dict):
        return None
    v = result.get("video")
    if isinstance(v, dict) and v.get("url"):
        return v["url"]
    if isinstance(v, str):
        return v
    vs = result.get("videos")
    if isinstance(vs, list) and vs and isinstance(vs[0], dict):
        return vs[0].get("url")
    for k in ("url", "output", "video_url"):
        if isinstance(result.get(k), str):
            return result[k]
    return None


def download(url: str, dst: Path) -> Path:
    import urllib.request
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return dst
