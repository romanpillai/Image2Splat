"""Img2Splat pipeline steps.

Image -> cutout -> orbit control render -> (fal LTX render-to-real) -> matte ->
COLMAP dataset with EXACT poses -> Brush.

The premise, carried over from V3: when the camera path is authored rather than
solved, pose error is removed entirely. Part 2 measured that as the difference
between 25.06 dB (solved, 4.5 deg RMS) and 29.58 dB (exact). Here the orbit is
defined by this tool, so the poses are exact by construction -- no Blender scene
and no pose solver anywhere in the geometry path.

The control video is rendered HERE, from the same camera model that is written
to COLMAP, so the video LTX conditions on and the poses Brush trains against can
never drift apart.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time as _time
import sys
from dataclasses import dataclass, asdict, replace
from pathlib import Path

import cv2
import numpy as np

BG_GREY = 60           # matches the backdrop of the clips that mattes cleanly
# Flat clay: one neutral grey for every point, no light. Kept well clear of
# BG_GREY -- the mattes separate subject from backdrop by exactly this contrast.
CLAY_GREY = 150


# ------------------------------------------------------------- aspect fit ---
# Standard video aspect ratios, with canvas sizes whose BOTH sides are multiples
# of 32 and whose ratio is exact. Diffusion video models bucket to standard
# aspects; feeding an odd one is what produced pillarbox bars on the 704x960
# clip (aspect 0.7333 = 11:15), where 99/100 frames carried black side bars.
#
# Honest caveat: the root cause is not documented by fal. The 448x640 clip
# (0.70 = 7:10) was equally non-standard and came back CLEAN, and both sizes are
# divisible by 32 -- so divisibility alone does not explain it. Sticking to
# standard ratios is a mitigation, not a proven fix, and matte() still strips
# bars defensively.
#
# fal's ltx-2.3 image-to-video sibling only accepts auto | 16:9 | 9:16, which is
# a strong hint about which buckets the model is actually happy in.
ASPECTS = [
    # name          w     h     ratio
    ("9:16 tall",   576, 1024),
    ("2:3 portrait", 640, 960),
    ("3:4 portrait", 768, 1024),
    ("1:1 square",  1024, 1024),
    ("4:3 landscape", 1024, 768),
    ("3:2 landscape", 960, 640),
    ("16:9 wide",   1024, 576),
]


def aspect_presets():
    return [{"name": n, "width": w, "height": h, "ratio": round(w / h, 5)}
            for n, w, h in ASPECTS]


def suggest_aspect(img_w: int, img_h: int):
    """Closest standard preset to an image, by log-ratio distance.

    Log distance so that being 10% too wide and 10% too tall are penalised
    equally -- a linear difference is biased toward the landscape end.
    """
    r = img_w / max(img_h, 1)
    best = min(ASPECTS, key=lambda a: abs(math.log(r) - math.log(a[1] / a[2])))
    n, w, h = best
    return {"name": n, "width": w, "height": h, "ratio": round(w / h, 5),
            "source_ratio": round(r, 5),
            "off_by_pct": round(100 * abs(r - w / h) / (w / h), 2)}


def fit_card_height(cut_w: int, cut_h: int, orbit: "Orbit",
                    fill: float = 0.86) -> float:
    """Card height (world units) so the subject fills `fill` of the frame.

    This is the "fill the image to that aspect ratio" step: rather than padding
    pixels into the cutout, the card is scaled in 3D so the chosen canvas is
    filled by the subject with a margin. Whichever axis binds first wins, so the
    subject never spills out of a wide or a tall canvas.
    """
    d = math.hypot(orbit.radius, orbit.aim_z - orbit.cam_z)
    tan_v = math.tan(math.radians(orbit.vfov_deg) / 2.0)
    card_h = 2.0 * fill * d * tan_v                      # height-limited

    f = orbit.focal()
    card_w = card_h * (cut_w / max(cut_h, 1))
    proj_w = f * card_w / d
    if proj_w > fill * orbit.width:                      # width binds instead
        card_h *= (fill * orbit.width) / proj_w
    return round(card_h, 4)


# ----------------------------------------------------------------- camera ---
@dataclass
class Orbit:
    frames: int = 100
    sweep_deg: float = 360.0
    radius: float = 9.0
    elev_deg: float = -15.0       # camera elevation w.r.t. the aim point.
                                  # negative = camera BELOW the aim, looking up.
                                  # -15 reproduces the proven Blender rig
                                  # (cam_z -0.1312, aim_z 2.2768, radius 8.96).
    # Elevation sweep. 0 keeps the old flat ring; ~20 gives the +-20 degree
    # spread SplatFormer measured at +1.88 dB over +-10. Sinusoidal, so frame 0
    # and the final frame both sit at elev_deg -- source framing and loop
    # closure are untouched.
    elev_sweep_deg: float = 0.0
    elev_cycles: float = 1.0      # up-down cycles per full revolution
    aim_z: float = 2.28           # where the optical axes converge
    vfov_deg: float = 22.62       # 4800px focal at 1920 -- the proven rig
    width: int = 640
    height: int = 960
    ccw: bool = True
    # Degrees the orbit is rotated away from the photo viewpoint. 0 puts the
    # first camera exactly where the photo was taken, which makes that frame
    # blind to depth error -- see angles().
    phase_deg: float = 0.0

    # ---- camera path ------------------------------------------------------
    # "ring"  : constant elevation, optionally swept by a sinusoid that
    #           returns to elev_deg at frame 0 and at closure.
    # "helix" : height ramps linearly between two absolute world heights while
    #           the camera circles, turning at each end -- a triangle wave, so
    #           the motion is continuous and the full height range is covered
    #           no matter where it starts.
    #
    # Ring is parameterised by ANGLE because it is calibrated from the photo;
    # helix is parameterised by HEIGHT because that is what the operator is
    # choosing when they say "between here and there". Height is also the
    # safer input: elevation derived through atan2 can never run away, whereas
    # radius*tan(elev) explodes near +-90.
    path: str = "ring"
    helix_min_z: float = 0.0      # lowest camera height, world units
    helix_max_z: float = 3.0      # highest camera height, world units
    # Where in the min->max->min travel frame 0 sits. 0 = at the bottom
    # climbing, 0.5 = at the top descending, 0.25 = halfway up. The camera
    # does NOT return to its starting height, and that is fine: the poses are
    # authored, so nothing downstream needs the path to close.
    helix_start: float = 0.0

    # ---- three-ring pass --------------------------------------------------
    # path = "rings": three complete orbits in sequence, at three heights and
    # three radii. This is the direct answer to the flat-ring problem -- one
    # elevation gives no vertical parallax at all, so nothing constrains how
    # tall a gaussian may grow, which is what produces needles. Three
    # separated heights constrain it from three directions.
    #
    # The MIDDLE ring is first and uses radius/elev_deg unchanged, so frame 0
    # is still the photo viewpoint at the nominal radius and everything
    # anchored to frame 0 -- the card, the ray origin, the fal reference --
    # behaves exactly as it does on a plain ring.
    #
    # A radius of 0 means "same as the middle", so only what differs is set.
    ring_top_z: float = 4.0
    ring_top_r: float = 0.0
    ring_bot_z: float = 0.0
    ring_bot_r: float = 0.0

    def __post_init__(self):
        """Keep every camera parameter inside a range that produces a real
        picture. tan() near +-90 degrees does not overflow to infinity, it
        returns ~1.6e16, so an out-of-range elevation yields a finite but
        absurd camera that passes every downstream sanity check."""
        self.frames = max(1, int(self.frames))
        self.radius = max(1e-3, float(self.radius))
        self.vfov_deg = min(170.0, max(1.0, float(self.vfov_deg)))
        self.elev_deg = min(85.0, max(-85.0, float(self.elev_deg)))
        self.width = max(16, int(self.width))
        self.height = max(16, int(self.height))
        self.elev_cycles = max(0.0, float(self.elev_cycles))
        _p = str(self.path).lower()
        self.path = _p if _p in ("helix", "rings") else "ring"
        self.ring_top_r = max(0.0, float(self.ring_top_r))
        self.ring_bot_r = max(0.0, float(self.ring_bot_r))
        # A reversed pair is an obvious slip, not an error worth refusing.
        lo, hi = float(self.helix_min_z), float(self.helix_max_z)
        self.helix_min_z, self.helix_max_z = min(lo, hi), max(lo, hi)
        self.helix_start = float(self.helix_start) % 1.0
        # the sweep must not push any frame past the same limit
        head = min(85.0 - self.elev_deg, self.elev_deg + 85.0)
        self.elev_sweep_deg = min(max(0.0, head),
                                  max(0.0, float(self.elev_sweep_deg)))

    @property
    def cam_z(self) -> float:
        """Height of the FRAME 0 camera.

        Frame 0 is the lifting viewpoint -- every pixel's ray starts there --
        so this is the value the depth lift, the framing and the preview all
        read. Taking it from cam_zs()[0] keeps it true for both paths; on a
        ring it is exactly the old aim_z + radius*tan(elev_deg).
        """
        return self.cam_zs()[0]

    def focal(self) -> float:
        return (self.height / 2.0) / math.tan(math.radians(self.vfov_deg) / 2.0)

    def intrinsics(self):
        f = self.focal()
        return f, f, self.width / 2.0, self.height / 2.0

    def _helix_zs(self):
        """Per-frame camera height along the helix: a triangle wave.

        A plain ramp cannot honour a start offset -- it would either use only
        part of the range or jump when it wrapped. Turning at each end keeps
        the motion continuous, covers min..max in full from any start, and
        makes elev_cycles read naturally: 0.5 is a single climb, 1.0 is up and
        back down, 2.0 is two round trips per revolution.
        """
        n = self.frames
        lo, hi = self.helix_min_z, self.helix_max_z
        revs = abs(self.sweep_deg) / 360.0
        span = float(self.elev_cycles) * revs
        out = []
        for i in range(n):
            p = (self.helix_start + span * (i / n)) % 1.0
            tri = 1.0 - 2.0 * abs(p - 0.5)      # 0 at p=0, 1 at p=0.5
            out.append(lo + (hi - lo) * tri)
        return out

    def elevs(self):
        """Per-frame elevation, in degrees, for whichever path is selected."""
        n = self.frames
        if self.path == "helix":
            return [math.degrees(math.atan2(z - self.aim_z, self.radius))
                    for z in self._helix_zs()]
        if not self.elev_sweep_deg:
            return [self.elev_deg] * n
        w = 2.0 * math.pi * float(self.elev_cycles)
        return [self.elev_deg + self.elev_sweep_deg * math.sin(w * i / n)
                for i in range(n)]

    def rings(self):
        """The three rings as (height, radius, frame_count), middle FIRST.

        Frames divide as evenly as the count allows, with the remainder given
        to the earlier rings, so 100 frames become 34/33/33 rather than
        33/33/33 and a dropped frame.
        """
        n = self.frames
        base = n // 3
        counts = [base + (1 if i < n - base * 3 else 0) for i in range(3)]
        mid_z = self.aim_z + self.radius * math.tan(math.radians(self.elev_deg))
        return [
            (mid_z, self.radius, counts[0]),
            (float(self.ring_top_z), self.ring_top_r or self.radius, counts[1]),
            (float(self.ring_bot_z), self.ring_bot_r or self.radius, counts[2]),
        ]

    def radii(self):
        """Per-frame orbit radius.

        Only the three-ring path varies it; every other path returns the one
        radius repeated. poses() reads THIS rather than self.radius, which is
        what lets the rings differ in size -- self.radius stays the nominal
        value that scene-scale heuristics and the frame-0 card are built on.
        """
        if self.path != "rings":
            return [self.radius] * self.frames
        out = []
        for _z, r, k in self.rings():
            out.extend([r] * k)
        return out[:self.frames] or [self.radius]

    def cam_zs(self):
        """Per-frame camera height."""
        if self.path == "helix":
            return self._helix_zs()
        if self.path == "rings":
            out = []
            for z, _r, k in self.rings():
                out.extend([z] * k)
            return out[:self.frames] or [self.aim_z]
        return [self.aim_z + self.radius * math.tan(math.radians(e))
                for e in self.elevs()]

    def elev_span(self):
        e = self.elevs()
        return (min(e), max(e))

    # The azimuth the PHOTO was taken from. Every pixel's ray starts here,
    # because that is where the light actually came from, so this is physics
    # and not a choice.
    RAY_AZIMUTH = -math.pi / 2

    def ray_angle(self) -> float:
        """Azimuth of the viewpoint that defines every pixel's ray."""
        return self.RAY_AZIMUTH

    def angles(self):
        """Camera azimuth per frame.

        phase_deg rotates the whole orbit away from the photo viewpoint. At 0
        the first camera sits exactly where the photo was taken, and that
        camera is structurally BLIND to depth error: every point was placed
        along a ray from there, so sliding a point up or down its own ray
        cannot change what it sees. The one frame looked at hardest is the one
        frame that cannot report a mistake, which is how a 3.6% focal error
        and a 20-degree ground tilt both survived review.

        Any non-zero phase moves every camera off that degenerate viewpoint,
        so depth error becomes visible in frame 0 like it is everywhere else.
        The cost is honest: the photo has no data for what the offset reveals,
        so disocclusion holes appear from frame 0 onward instead of opening
        up gradually.
        """
        n = self.frames
        span = math.radians(self.sweep_deg) * (1 if self.ccw else -1)
        ph = math.radians(self.phase_deg)
        if self.path == "rings":
            # Each ring sweeps the FULL angle in its own share of the frames,
            # so all three cover the subject completely rather than each doing
            # a third of a turn.
            out = []
            for _z, _r, k in self.rings():
                out.extend(self.RAY_AZIMUTH + ph + span * j / max(k, 1)
                           for j in range(k))
            return out[:n] or [self.RAY_AZIMUTH + ph]
        # last frame lands exactly one step short of closing, so frame 0 is not
        # duplicated -- the V3 rig wasted a training view on an exact repeat
        return [self.RAY_AZIMUTH + ph + span * i / n for i in range(n)]

    def poses(self):
        """World-to-camera (R, t) per frame, COLMAP convention (+Z forward, +Y down)."""
        out = []
        aim = np.array([0.0, 0.0, self.aim_z])
        for a, cz, rr in zip(self.angles(), self.cam_zs(), self.radii()):
            C = np.array([rr * math.cos(a),
                          rr * math.sin(a),
                          cz])
            fwd = aim - C
            fwd /= np.linalg.norm(fwd)
            world_up = np.array([0.0, 0.0, 1.0])
            right = np.cross(fwd, world_up)
            right /= np.linalg.norm(right)
            down = np.cross(fwd, right)
            R = np.stack([right, down, fwd], axis=0)   # rows = camera axes
            t = -R @ C
            out.append((R, t))
        return out


def project(R, t, P, fx, fy, cx, cy):
    X = (R @ P.T).T + t
    z = X[:, 2]
    u = fx * X[:, 0] / z + cx
    v = fy * X[:, 1] / z + cy
    return np.stack([u, v], axis=1), z


# ----------------------------------------------------------------- cutout ---
def cutout(src: Path, dst: Path, size: int = 1024) -> dict:
    """RMBG-1.4 background removal -> RGBA PNG."""
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    mp = hf_hub_download("briaai/RMBG-1.4", "onnx/model.onnx")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    prov = ort.get_available_providers()
    use = (["CUDAExecutionProvider", "CPUExecutionProvider"]
           if "CUDAExecutionProvider" in prov else ["CPUExecutionProvider"])
    sess = ort.InferenceSession(mp, so, providers=use)
    inp = sess.get_inputs()[0]

    im = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"cannot read image: {src}")
    H, W = im.shape[:2]

    x = cv2.resize(im, (size, size), interpolation=cv2.INTER_LINEAR)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = np.transpose((x - 0.5), (2, 0, 1))[None].astype(np.float32)

    out = sess.run(None, {inp.name: x})[0]
    m = out[0, 0] if out.ndim == 4 else out[0]
    m = (m - m.min()) / max(float(m.max() - m.min()), 1e-8)
    alpha = np.clip(cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR), 0, 1)

    # Keep the FULL frame. Trimming to the subject bbox used to happen here,
    # and it silently destroyed the correspondence with the source image: the
    # trimmed subject was then rescaled to fill 86% of the canvas and centred on
    # the aim point, so control frame 0 showed the subject at a different size
    # and position than the reference image sent as frame 0. The video model
    # reconciled the two by warping the subject.
    #
    # The bbox is recorded instead, so "fit subject to frame" can still be
    # offered as an explicit choice rather than being forced on everyone.
    ys, xs = np.nonzero(alpha > 0.5)
    if len(ys):
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    else:
        bbox = [0, 0, int(W), int(H)]

    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), np.dstack([im, (alpha * 255).astype(np.uint8)]),
                [cv2.IMWRITE_PNG_COMPRESSION, 6])
    return {"width": int(W), "height": int(H), "bbox": bbox,
            "coverage": float((alpha > 0.5).mean()),
            "providers": sess.get_providers()}


def _rmbg_session(size: int = 1024):
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    mp = hf_hub_download("briaai/RMBG-1.4", "onnx/model.onnx")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    prov = ort.get_available_providers()
    use = (["CUDAExecutionProvider", "CPUExecutionProvider"]
           if "CUDAExecutionProvider" in prov else ["CPUExecutionProvider"])
    sess = ort.InferenceSession(mp, so, providers=use)
    return sess, sess.get_inputs()[0], size


def _rmbg_alpha(sess, inp, size, im):
    H, W = im.shape[:2]
    x = cv2.resize(im, (size, size), interpolation=cv2.INTER_LINEAR)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = np.transpose((x - 0.5), (2, 0, 1))[None].astype(np.float32)
    out = sess.run(None, {inp.name: x})[0]
    m = out[0, 0] if out.ndim == 4 else out[0]
    m = (m - m.min()) / max(float(m.max() - m.min()), 1e-8)
    return np.clip(cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR), 0, 1)


# Matte models, in the order they are offered. "gated" ones need a Hugging
# Face token and a licence accepted on the model page, so they are listed but
# will fail with a clear message rather than silently.
MATTE_MODELS = {
    "rmbg14": {"label": "RMBG-1.4 (fast, 2024)", "repo": "briaai/RMBG-1.4",
               "kind": "onnx", "mb": 177, "gated": False,
               "note": "The original. Quick, and the weakest on hair and fur."},
    "birefnet": {"label": "BiRefNet", "repo": "ZhengPeng7/BiRefNet",
                 "kind": "hf", "mb": 444, "gated": False,
                 "note": "The open workhorse behind several commercial "
                         "removers. Much stronger edges than RMBG-1.4."},
    "birefnet_hr": {"label": "BiRefNet-HR (best on hair)",
                    "repo": "ZhengPeng7/BiRefNet_HR",
                    "kind": "hf", "mb": 444, "gated": False,
                    "note": "High-resolution variant; benchmarks best on hair "
                            "specifically. Start here for fur and dark coats."},
    "ben2": {"label": "BEN2 (refines uncertain pixels)",
             "repo": "PramaLLC/BEN2", "kind": "ben2", "mb": 1135,
             "gated": False,
             "note": "A refiner network reprocesses only the pixels the base "
                     "was unsure about -- which is where hair lives."},
    "matanyone": {"label": "MatAnyone (video matting, soft edges)",
                  "repo": "PeiqingYang/MatAnyone", "kind": "matanyone",
                  "mb": 250, "gated": False,
                  "note": "Video MATTING rather than segmentation: it "
                          "propagates one first-frame mask through the clip "
                          "with a memory bank and returns fractional alpha, "
                          "so hair keeps partial coverage instead of being "
                          "cut hard. Seeded from BiRefNet-HR. Licence is "
                          "NTU S-Lab 1.0 - NON-COMMERCIAL."},
    "sam2": {"label": "SAM 2.1 (tracked through the clip)",
             "repo": "facebook/sam2.1-hiera-small", "kind": "sam2", "mb": 184,
             "gated": False,
             "note": "Segments the subject ONCE and tracks it, instead of "
                     "guessing again every frame. Seeded from BiRefNet's "
                     "first frame. Use it when a per-frame model keeps "
                     "latching onto the wrong object."},
    "sam2_text": {"label": "SAM 2.1 + text prompt (say what to keep)",
                  "repo": "facebook/sam2.1-hiera-small", "kind": "sam2_text",
                  "mb": 184 + 692, "gated": False,
                  "note": "You name the objects and it segments exactly those, "
                          "however many. GroundingDINO finds each one on frame "
                          "0, SAM 2 tracks them all. Use it when the automatic "
                          "models keep picking the wrong thing."},
    "rmbg20": {"label": "RMBG-2.0 (needs an HF token)",
               "repo": "briaai/RMBG-2.0", "kind": "hf", "mb": 885,
               "gated": True,
               "note": "Best benchmark of the salient-object set, but the "
                       "repo is gated: accept the licence and set HF_TOKEN."},
}

_MATTE_CACHE: dict = {}


def _hf_matte_model(repo: str):
    """A transformers image-segmentation model, held on the CPU between runs.

    Same policy as the depth models: this box shares its VRAM with ComfyUI,
    Ollama and Brush, so nothing squats on the GPU between calls.
    """
    if repo not in _MATTE_CACHE:
        import torch
        from transformers import AutoModelForImageSegmentation
        m = AutoModelForImageSegmentation.from_pretrained(
            repo, trust_remote_code=True)
        _MATTE_CACHE[repo] = m.eval()
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    return _MATTE_CACHE[repo]


def _hf_alpha_batch(model, images, size: int = 1024):
    """Alpha for a list of BGR frames from a transformers segmentation model."""
    import torch
    dev = ("cuda" if torch.cuda.is_available()
           and torch.cuda.mem_get_info()[0] > 2.5e9 else "cpu")
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    batch = []
    for im in images:
        x = cv2.resize(im, (size, size), interpolation=cv2.INTER_LINEAR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        batch.append(np.transpose((x - mean) / std, (2, 0, 1)))
    t = torch.from_numpy(np.stack(batch)).float()
    # BiRefNet ships fp16 weights. Half convolutions are unsupported on most
    # CPUs, so the model is promoted to fp32 there and the INPUT is matched to
    # the weights on GPU -- otherwise "Input type (float) and bias type
    # (c10::Half) should be the same".
    model.to(dev)
    if dev == "cpu":
        model.float()
    dt = next(model.parameters()).dtype
    try:
        with torch.no_grad():
            out = model(t.to(dev, dtype=dt))
        pred = out[-1] if isinstance(out, (list, tuple)) else out
        if isinstance(pred, (list, tuple)):
            pred = pred[-1]
        pred = torch.sigmoid(pred).float().cpu().numpy()
    finally:
        model.to("cpu")
        import torch as _t
        if dev == "cuda":
            _t.cuda.empty_cache()
    outs = []
    for im, p in zip(images, pred):
        m = p[0] if p.ndim == 3 else p
        outs.append(np.clip(cv2.resize(m, (im.shape[1], im.shape[0]),
                                       interpolation=cv2.INTER_LINEAR), 0, 1))
    return outs


def matte_dir(frames: Path, out: Path, model: str = "rmbg14",
              log=None, batch: int = 4, prompt: str = "") -> int:
    """Run a matte model over a frame directory -> RGBA PNGs carrying alpha.

    Every model here is per-frame: it has no idea frame 47 is the same subject
    as frame 46, which is why silhouettes can flicker. That is a property of
    the model class, not of any particular one.
    """
    spec = MATTE_MODELS.get(model)
    if spec is None:
        raise RuntimeError(f"unknown matte model {model!r} "
                           f"(have {list(MATTE_MODELS)})")
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("frame_*.png"):
        stale.unlink()          # a shorter re-run must not leave a tail behind
    files = sorted(frames.glob("frame_*.png"))
    if not files:
        raise RuntimeError(f"no frames in {frames}")

    if spec["kind"] == "onnx":
        sess, inp, size = _rmbg_session()
        for k, f in enumerate(files):
            im = cv2.imread(str(f))
            if im is None:
                continue
            a = (_rmbg_alpha(sess, inp, size, im) * 255).astype(np.uint8)
            cv2.imwrite(str(out / f.name), np.dstack([im, a]),
                        [cv2.IMWRITE_PNG_COMPRESSION, 3])
            if log and (k % 20 == 0 or k == len(files) - 1):
                log(f"  {model} {k + 1}/{len(files)}")
        return len(files)

    if spec["kind"] == "matanyone":
        return _matanyone(files, out, log)

    if spec["kind"] == "sam2":
        return _sam2_track(files, out, spec["repo"], log)

    if spec["kind"] == "sam2_text":
        return _sam2_track(files, out, spec["repo"], log, prompt=prompt)

    if spec["kind"] == "ben2":
        raise RuntimeError(
            "BEN2 ships its own loader rather than a transformers class; "
            "not wired yet. BiRefNet-HR is the closest available option.")

    try:
        mdl = _hf_matte_model(spec["repo"])
    except Exception as e:
        if spec["gated"]:
            raise RuntimeError(
                f"{spec['label']} is gated: accept the licence at "
                f"huggingface.co/{spec['repo']} and set HF_TOKEN in the "
                f"server's environment. ({type(e).__name__})") from e
        raise
    for i in range(0, len(files), max(1, batch)):
        chunk = files[i:i + max(1, batch)]
        ims = [cv2.imread(str(f)) for f in chunk]
        keep = [(f, im) for f, im in zip(chunk, ims) if im is not None]
        if not keep:
            continue
        alphas = _hf_alpha_batch(mdl, [im for _, im in keep])
        for (f, im), a in zip(keep, alphas):
            cv2.imwrite(str(out / f.name),
                        np.dstack([im, (a * 255).astype(np.uint8)]),
                        [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if log:
            log(f"  {model} {min(i + batch, len(files))}/{len(files)}")
    return len(files)


DINO_REPO = "IDEA-Research/grounding-dino-tiny"


def _dino_boxes(rgb, prompt: str, log=None):
    """Text -> boxes on one frame, via GroundingDINO.

    Phrases are period-separated and lowercase, which is what the model was
    trained on: "a horse. a person. a car." Anything it cannot find is simply
    absent from the result rather than an error, so a prompt naming ten things
    and matching three yields three tracked objects.
    """
    import torch
    from PIL import Image
    from transformers import (GroundingDinoForObjectDetection,
                              GroundingDinoProcessor)
    txt = prompt.strip().lower()
    if not txt:
        return [], []
    if not txt.endswith("."):
        txt += "."
    dev = ("cuda" if torch.cuda.is_available()
           and torch.cuda.mem_get_info()[0] > 3e9 else "cpu")
    proc = GroundingDinoProcessor.from_pretrained(DINO_REPO)
    det = GroundingDinoForObjectDetection.from_pretrained(DINO_REPO).to(dev).eval()
    try:
        H, W = rgb.shape[:2]
        inp = proc(images=Image.fromarray(rgb), text=txt,
                   return_tensors="pt").to(dev)
        with torch.no_grad():
            out = det(**inp)
        # 0.25 rather than the usual 0.35: a phrase that half-matches is
        # still worth tracking, and a wrong box is obvious in the preview.
        res = proc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=0.25, text_threshold=0.25,
            target_sizes=[(H, W)])[0]
        boxes = res["boxes"].cpu().numpy().tolist()
        labels = list(res.get("text_labels") or res.get("labels") or [])
        scores = res["scores"].cpu().numpy().tolist()
    finally:
        det.to("cpu")
        if dev == "cuda":
            torch.cuda.empty_cache()
    if log:
        log(f"  dino: found {len(boxes)} object(s) for {txt!r}")
        for l, sc in zip(labels, scores):
            log(f"    - {l}  (confidence {sc:.2f})")
        if len(boxes) < len([q for q in txt.split(".") if q.strip()]):
            log("  dino: some phrases matched nothing. Plain nouns work best "
                "('a horse', not 'the black horse in the middle'), and each "
                "one needs to be visible in FRAME 0.")
    return boxes, labels


def _matanyone(files, out: Path, log=None, max_size: int = 1024) -> int:
    """MatAnyone: propagate ONE first-frame mask through the clip as alpha.

    Different in kind from everything else here. The per-frame models decide
    the silhouette again on every frame, so a strand of hair that is 40%
    covered gets rounded to in-or-out differently each time. MatAnyone carries
    a memory of the subject and returns FRACTIONAL alpha, which is what hair
    actually needs.

    Seeded from BiRefNet-HR's first frame, the same choice _sam2_track makes
    and for the same reason: the authored coverage is not a silhouette.

    It writes its own output tree (pha/ and fgr/, plus mp4s) under a temp dir,
    so the alpha is read back from the LOSSLESS pha PNGs -- never the mp4,
    which is 8-bit and lossy exactly where the soft edges are.
    """
    import shutil
    import tempfile
    from matanyone import InferenceCore

    if not files:
        raise RuntimeError("no frames to matte")
    work = Path(tempfile.mkdtemp(prefix="matanyone_"))
    try:
        # seed: BiRefNet-HR on frame 0, reduced to a single-channel mask
        seed_in = work / "seed_in"
        seed_in.mkdir()
        shutil.copy(files[0], seed_in / files[0].name)
        seed_out = work / "seed_out"
        if log:
            log("  matanyone: seeding from BiRefNet-HR on frame 0")
        matte_dir(seed_in, seed_out, model="birefnet_hr", log=None)
        seeded = sorted(seed_out.glob("frame_*.png"))
        if not seeded:
            raise RuntimeError("could not build a first-frame mask to seed from")
        a0 = cv2.imread(str(seeded[0]), cv2.IMREAD_UNCHANGED)
        if a0 is None or a0.shape[2] < 4:
            raise RuntimeError("seed mask has no alpha channel")
        mask_p = work / "seed.png"
        cv2.imwrite(str(mask_p), a0[:, :, 3])

        if log:
            log(f"  matanyone: propagating through {len(files)} frames "
                f"(max_size={max_size})")
        proc = InferenceCore("PeiqingYang/MatAnyone")
        proc.process_video(input_path=str(files[0].parent),
                           mask_path=str(mask_p),
                           output_path=str(work / "out"),
                           save_image=True, max_size=max_size)
        pha = sorted((work / "out" / files[0].parent.name / "pha").glob("*.png"))
        if not pha:
            raise RuntimeError(
                f"MatAnyone wrote no per-frame alpha under {work / 'out'}")
        if len(pha) != len(files) and log:
            log(f"  matanyone: WARNING got {len(pha)} alphas for "
                f"{len(files)} frames - pairing by order")

        n = 0
        for f, ap in zip(files, pha):
            im = cv2.imread(str(f))
            a = cv2.imread(str(ap), cv2.IMREAD_GRAYSCALE)
            if a is None:
                continue
            if a.shape[:2] != im.shape[:2]:
                # max_size downscales; put the alpha back on the frame's grid
                a = cv2.resize(a, (im.shape[1], im.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(out / f.name), np.dstack([im, a]),
                        [cv2.IMWRITE_PNG_COMPRESSION, 3])
            n += 1
            if log and (n % 20 == 0 or n == len(files)):
                log(f"  matanyone {n}/{len(files)}")
        return n
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _sam2_track(files, out: Path, repo: str, log=None, prompt: str = "") -> int:
    """SAM 2.1 video propagation: segment once, then TRACK.

    Every other model here decides afresh on each frame, which is why their
    silhouettes flicker. SAM 2 carries the subject's identity through the clip
    with a memory bank.

    Seeded from BiRefNet's first frame rather than from the authored coverage.
    The authored mask is not a subject silhouette at all -- it marks every
    pixel a point landed on, so at the photo viewpoint it covers ~88% of the
    frame and a quarter-turn away only ~16%. Tracking that would track "the
    visible part of the point cloud", which is not the thing we want.

    Measured on this footage it did NOT beat BiRefNet: 10.1% ambiguous edge
    pixels against 2.1%, for a frame-to-frame change of 5.14% against 5.37%.
    It earns its place when a salient-object model latches onto the wrong
    subject, not as a general upgrade.
    """
    import torch
    from transformers import Sam2VideoModel, Sam2VideoProcessor
    imgs = [cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in files]
    imgs = [i for i in imgs if i is not None]
    if not imgs:
        raise RuntimeError("no readable frames for SAM 2")
    H, W = imgs[0].shape[:2]

    # SAM 2 resizes to 1024 internally, so handing it full-resolution frames
    # buys nothing and costs a great deal: the session keeps EVERY frame as a
    # tensor, and 150 frames of 1174x1762 in float32 is ~4.6 GB before the
    # model has done anything. That filled a 12 GB card and the run stalled at
    # 99% utilisation with 42 MB free, which looks exactly like a hang.
    #
    # Track at 1024 on the long side and put the masks back at full size
    # afterwards -- post_process_masks upscales to whatever it is given.
    SAM_LONG = 1024
    scale = min(1.0, SAM_LONG / float(max(H, W)))
    if scale < 1.0:
        tw, th = int(round(W * scale)), int(round(H * scale))
        imgs = [cv2.resize(i, (tw, th), interpolation=cv2.INTER_AREA)
                for i in imgs]
        if log:
            log(f"  sam2: tracking at {tw}x{th} (from {W}x{H}); masks come "
                f"back at full size")
    else:
        tw, th = W, H

    boxes, labels = ([], [])
    if prompt:
        # DINO runs on the RESIZED frame, so its boxes are already in the
        # session's coordinate space -- no rescaling to get wrong.
        boxes, labels = _dino_boxes(imgs[0], prompt, log)
        if not boxes:
            raise RuntimeError(
                f"nothing in the first frame matched {prompt!r}. Try plainer "
                "nouns, separated by full stops: 'a horse. a person. a car.'")
    else:
        if log:
            log("  sam2: no prompt, seeding from BiRefNet on frame 0")
        seed_model = _hf_matte_model(MATTE_MODELS["birefnet"]["repo"])
        seed = _hf_alpha_batch(seed_model, [cv2.cvtColor(imgs[0],
                                                         cv2.COLOR_RGB2BGR)])[0]
        seed = seed > 0.5
        if not seed.any():
            raise RuntimeError("the seed mask is empty; nothing to track")

    dev = ("cuda" if torch.cuda.is_available()
           and torch.cuda.mem_get_info()[0] > 3e9 else "cpu")
    model = Sam2VideoModel.from_pretrained(repo).to(dev).eval()
    proc = Sam2VideoProcessor.from_pretrained(repo)
    try:
        # Keep the frames in system RAM and stream them in. The memory bank
        # still lives on the GPU, which is the part that needs to be fast.
        sess = proc.init_video_session(
            video=imgs, inference_device=dev,
            video_storage_device="cpu",
            inference_state_device=dev,
            max_vision_features_cache_size=1,
            dtype=torch.float32)
        if boxes:
            # All objects in ONE call. Adding them one at a time clears the
            # previous object's conditioning, and propagation then dies with
            # "maskmem_features ... cannot be empty".
            proc.add_inputs_to_inference_session(
                inference_session=sess, frame_idx=0,
                obj_ids=list(range(1, len(boxes) + 1)),
                input_boxes=[[[float(v) for v in b] for b in boxes]],
                original_size=(th, tw))
        else:
            proc.add_inputs_to_inference_session(
                inference_session=sess, frame_idx=0, obj_ids=1,
                input_masks=[torch.from_numpy(seed.astype(np.float32))],
                original_size=(th, tw))
        n = 0
        if log:
            free = (torch.cuda.mem_get_info()[0] / 1e9
                    if dev == "cuda" else 0.0)
            log(f"  sam2: propagating {len(files)} frames on {dev}"
                + (f", {free:.1f} GB VRAM free" if dev == "cuda" else ""))
        t_start = _time.time()
        with torch.no_grad():
            for res in model.propagate_in_video_iterator(sess,
                                                         start_frame_idx=0):
                m = proc.post_process_masks([res.pred_masks],
                                            original_sizes=[[H, W]],
                                            binarize=False)[0]
                # One channel per tracked object; the matte is their union,
                # so naming several things keeps all of them.
                a = torch.sigmoid(m[:, 0].float()).cpu().numpy()
                a = a.max(axis=0) if a.ndim == 3 else a
                # imgs may be the resized copies; the OUTPUT must carry the
                # original pixels, so re-read rather than upscaling them back.
                bgr = cv2.imread(str(files[res.frame_idx]))
                cv2.imwrite(str(out / files[res.frame_idx].name),
                            np.dstack([bgr, (a * 255).astype(np.uint8)]),
                            [cv2.IMWRITE_PNG_COMPRESSION, 3])
                n += 1
                # Every FIVE frames, with a rate and an estimate. At twenty
                # the first line could be minutes away, and silence is
                # indistinguishable from a hang -- which is exactly how this
                # was read.
                if log and (n % 5 == 0 or n == 1 or n == len(files)):
                    el = max(_time.time() - t_start, 1e-6)
                    rate = n / el
                    left = (len(files) - n) / rate if rate > 0 else 0
                    log(f"  sam2 {n}/{len(files)}  {rate:.1f} fps"
                        + (f", ~{left:.0f}s left" if n < len(files) else
                           f", {el:.0f}s total"))
    finally:
        model.to("cpu")
        if dev == "cuda":
            torch.cuda.empty_cache()
    return n


def authored_alpha_mp4(mattes: Path, dst: Path, fps: int = 24) -> Path:
    """The authored coverage as a viewable video: white subject on black.

    Not a model output -- this is what the renderer actually drew, so it is
    the reference the segmenters are trying to reproduce.
    """
    files = sorted(mattes.glob("frame_*.png"))
    if not files:
        raise RuntimeError("no authored mattes; re-render the control video")
    first = cv2.imread(str(files[0]), cv2.IMREAD_UNCHANGED)
    H, W = first.shape[:2]
    dst.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(
        [_ffmpeg(), "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-framerate", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
         "-preset", "veryfast", str(dst)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    for f in files:
        g = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if g is None:
            continue
        if g.ndim == 3:
            g = g[:, :, 3] if g.shape[2] == 4 else g[:, :, 0]
        enc.stdin.write(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR).tobytes())
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError("ffmpeg failed writing the authored-alpha preview")
    return dst


def rmbg_dir(frames: Path, out: Path, log=None) -> int:
    """RMBG-1.4 over a frame directory -> RGBA PNGs carrying the alpha.

    Used on the AI frames as a matte source. Unlike the authored coverage this
    is a per-frame segmentation with no temporal coupling, so its silhouette
    can flicker -- but it is dense, and it segments the exact images that will
    be trained on rather than a stand-in for them.
    """
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("frame_*.png"):
        stale.unlink()          # a shorter re-run must not leave a tail behind
    sess, inp, size = _rmbg_session()
    files = sorted(frames.glob("frame_*.png"))
    for k, f in enumerate(files):
        im = cv2.imread(str(f))
        if im is None:
            continue
        a = (_rmbg_alpha(sess, inp, size, im) * 255).astype(np.uint8)
        cv2.imwrite(str(out / f.name), np.dstack([im, a]),
                    [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if log and (k % 20 == 0 or k == len(files) - 1):
            log(f"  rmbg {k + 1}/{len(files)}")
    return len(files)


def rmbg_preview_mp4(rmbg: Path, dst: Path, fps: int = 24,
                     bg: int = 0) -> Path:
    """The cut-out frames flattened onto a flat colour, so they can be viewed.

    Alpha does not survive an mp4, so the subject is composited on a constant
    background -- black by default, which reads clearly against most subjects
    and makes a bad matte obvious rather than plausible.
    """
    files = sorted(rmbg.glob("frame_*.png"))
    if not files:
        raise RuntimeError("no cut-out frames to preview")
    first = cv2.imread(str(files[0]), cv2.IMREAD_UNCHANGED)
    H, W = first.shape[:2]
    dst.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(
        [_ffmpeg(), "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-framerate", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
         "-preset", "veryfast", str(dst)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    for f in files:
        im = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if im is None or im.ndim != 3 or im.shape[2] != 4:
            continue
        a = (im[:, :, 3:4].astype(np.float32) / 255.0)
        out = (im[:, :, :3].astype(np.float32) * a + bg * (1.0 - a))
        enc.stdin.write(np.clip(out, 0, 255).astype(np.uint8).tobytes())
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError("ffmpeg failed writing the cut-out preview")
    return dst


def copy_frames_opaque(frames: Path, out: Path) -> int:
    """Copy frames as fully-opaque RGBA, for training with the background kept.

    The dataset format stays RGBA so nothing downstream has to special-case it;
    alpha is simply 255 everywhere, and brush_cmd drops --match-alpha-weight
    since there is no matte to supervise against.
    """
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("frame_*.png"):
        f.unlink()
    files = sorted(frames.glob("frame_*.png"))
    for f in files:
        im = cv2.imread(str(f), cv2.IMREAD_COLOR)
        a = np.full(im.shape[:2] + (1,), 255, np.uint8)
        cv2.imwrite(str(out / f.name), np.dstack([im, a]),
                    [cv2.IMWRITE_PNG_COMPRESSION, 4])
    return len(files)


# --------------------------------------------------- orbit control render ---
# ------------------------------------------------------------- backdrop ----
BACKDROP_MODES = ("grey",)          # the panorama dome was removed in beta
_LIGHT = np.array([-0.45, -0.35, 0.82])


def _box_faces(cx, cy, cz, sx, sy, sz, yaw):
    """Six quads of an axis-aligned-then-yawed box, in world space."""
    c, s = math.cos(yaw), math.sin(yaw)
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    corners = []
    for dx in (-hx, hx):
        for dy in (-hy, hy):
            for dz in (-hz, hz):
                corners.append([cx + dx * c - dy * s, cy + dx * s + dy * c, cz + dz])
    P = np.array(corners)          # index: (dx,dy,dz) bits -> 4*i+2*j+k
    idx = [(0, 2, 6, 4), (1, 5, 7, 3),      # -z, +z
           (0, 1, 3, 2), (4, 6, 7, 5),      # -x', +x'
           (0, 4, 5, 1), (2, 3, 7, 6)]      # -y', +y'
    out = []
    for f in idx:
        q = P[list(f)]
        n = np.cross(q[1] - q[0], q[2] - q[0])
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        n /= ln
        # Ambient term: pure |n.L| sends faces perpendicular to the light to
        # zero, which rendered markers as solid black rectangles that read as
        # holes punched in the frame rather than as objects.
        out.append((q, 0.38 + 0.62 * float(abs(np.dot(n, _LIGHT)))))
    return out


def _camera_rays(orbit: "Orbit") -> np.ndarray:
    """Unit ray directions in CAMERA space, one per pixel. Pose-independent, so
    this is computed once and rotated per frame."""
    fx, fy, cx, cy = orbit.intrinsics()
    W, H = orbit.width, orbit.height
    uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], axis=-1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def scene_polys(orbit: "Orbit", ground_z: float, seed: int = 0):
    """Floor disc plus a ring of marker blocks, as (poly3d, shade) pairs.

    The ring sits OUTSIDE the camera orbit (1.6x the radius) so the camera
    always travels inside it. Markers within the orbit would swing between the
    camera and the subject and occlude it, which would wreck the reconstruction;
    from inside the ring they can only ever be background.

    The floor is built as annulus segments rather than one disc because the
    camera stands inside it: a single polygon would have vertices behind the
    camera and get culled whole. Small quads let only the behind-camera ones go.
    """
    rng = np.random.default_rng(seed)
    R = orbit.radius
    ring_r = R * 1.6
    floor_r = ring_r * 1.9      # push the horizon well past the marker ring
    polys = []

    # Floor as a grid of XY tiles, not annulus segments. Radial segments make a
    # checker that converges to a bullseye directly under the subject, which
    # reads as a dartboard rather than a floor. A cartesian grid stays uniform
    # and gives the video model a texture whose perspective sweep is
    # unambiguous.
    cell = max(floor_r / 11.0, 0.75)
    nt = int(math.ceil(floor_r / cell))
    for ix in range(-nt, nt):
        for iy in range(-nt, nt):
            x0, x1 = ix * cell, (ix + 1) * cell
            y0, y1 = iy * cell, (iy + 1) * cell
            # keep the floor roughly circular so there is no square silhouette
            if math.hypot((x0 + x1) / 2, (y0 + y1) / 2) > floor_r:
                continue
            q = np.array([[x0, y0, ground_z], [x1, y0, ground_z],
                          [x1, y1, ground_z], [x0, y1, ground_z]])
            shade = 0.60 + (0.09 if (ix + iy) % 2 else 0.0)
            polys.append((q, shade))

    # Marker blocks on the ring. 24 rather than 16: the horizontal FOV is only
    # ~16 deg, so at any instant just two or three fall inside the frame, and
    # too few leaves stretches of the orbit with no parallax reference at all.
    n_mark = 24
    for i in range(n_mark):
        a = 2 * math.pi * i / n_mark + float(rng.uniform(-0.05, 0.05))
        rr = ring_r * float(rng.uniform(0.92, 1.10))
        h = float(rng.uniform(0.55, 1.6)) * (orbit.aim_z - ground_z + 1.0)
        w = float(rng.uniform(0.8, 2.0))
        d = float(rng.uniform(0.8, 2.0))
        cx, cy = rr * math.cos(a), rr * math.sin(a)
        polys.extend(_box_faces(cx, cy, ground_z + h / 2, w, d, h, a))
    return polys


def _draw_polys(canvas, polys, R, t, orbit: "Orbit", card_depth: float,
                base=BG_GREY):
    """Painter's-algorithm fill. Returns (far_done, near_polys_for_after_card)."""
    fx, fy, cx, cy = orbit.intrinsics()
    W, H = orbit.width, orbit.height
    items = []
    for q, shade in polys:
        X = (R @ q.T).T + t
        if (X[:, 2] <= 0.05).any():          # any vertex behind the camera
            continue
        u = fx * X[:, 0] / X[:, 2] + cx
        v = fy * X[:, 1] / X[:, 2] + cy
        pts = np.stack([u, v], 1)
        if (pts[:, 0].max() < 0 or pts[:, 0].min() > W
                or pts[:, 1].max() < 0 or pts[:, 1].min() > H):
            continue
        items.append((float(X[:, 2].mean()), pts.astype(np.int32), shade))
    items.sort(key=lambda it: -it[0])        # far first

    after = []
    for depth, pts, shade in items:
        if depth < card_depth:
            after.append((pts, shade))
            continue
        cv2.fillConvexPoly(canvas, pts, (base * shade * 2.6,) * 3, cv2.LINE_AA)
    return after


def scene_init_points(orbit: "Orbit", ground_z: float, n: int = 140000,
                      seed: int = 0):
    """Init cloud for a scene splat, seeded from the geometry we authored.

    With the background kept there is no silhouette to carve a visual hull
    from, so instead of falling back to a random blob this samples the parts of
    the scene we know exist: the subject volume, the floor, and the marker ring.
    """
    rng = np.random.default_rng(seed)
    R = orbit.radius
    ring_r = R * 1.6
    floor_r = ring_r * 1.3

    n_sub = int(n * 0.55)
    n_floor = int(n * 0.30)
    n_mark = n - n_sub - n_floor

    rr = R * 0.16 * np.sqrt(rng.random(n_sub))
    aa = rng.random(n_sub) * 2 * np.pi
    zz = rng.uniform(ground_z, orbit.aim_z + (orbit.aim_z - ground_z), n_sub)
    sub = np.stack([rr * np.cos(aa), rr * np.sin(aa), zz], 1)

    fr = floor_r * np.sqrt(rng.random(n_floor))
    fa = rng.random(n_floor) * 2 * np.pi
    flo = np.stack([fr * np.cos(fa), fr * np.sin(fa),
                    np.full(n_floor, ground_z)], 1)

    ma = rng.random(n_mark) * 2 * np.pi
    mr = ring_r * rng.uniform(0.88, 1.14, n_mark)
    mz = rng.uniform(ground_z, ground_z + (orbit.aim_z - ground_z) * 1.6, n_mark)
    mar = np.stack([mr * np.cos(ma), mr * np.sin(ma), mz], 1)

    return np.concatenate([sub, flo, mar], 0)


def clay(bgr: np.ndarray, alpha: np.ndarray, mode: str = "shaded",
         light=(-0.5, -0.7, 0.6)) -> np.ndarray:
    """Replace the photograph with a neutral grey clay material.

    The point is to hand LTX FORM and MOTION without handing it TEXTURE. If the
    control video carries the source texture, render-to-real has little left to
    invent and tends to copy the card rather than build a believable turnaround.

    A flat card has no real normals, so 'shaded' fakes them: a distance
    transform inward from the silhouette edge gives a dome-like height field,
    and its gradient gives a pseudo-normal to light. The result reads as an
    inflated clay maquette -- volume cues, zero surface detail.

    'flat' is a uniform grey silhouette: no texture and no form either. Useful
    when even the pseudo-relief is misleading (very thin or filigree subjects).
    """
    a = alpha[:, :, 0] if alpha.ndim == 3 else alpha
    m = (a > 0.5).astype(np.uint8)
    if mode == "off":
        return bgr
    if mode == "flat" or not m.any():
        out = np.full_like(bgr, 168.0)
        return out

    # (a) silhouette relief -- an inflated dome from the outer boundary
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    scale = max(float(np.percentile(dt[m > 0], 92)), 1.0)
    h = np.sqrt(np.clip(dt / scale, 0.0, 1.0))
    h = cv2.GaussianBlur(h, (0, 0), max(scale * 0.10, 1.0))

    gx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=5)
    strength = 2.2
    nx, ny, nz = -gx * strength, -gy * strength, np.ones_like(h)
    n = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
    nx, ny, nz = nx / n, ny / n, nz / n
    L = np.array(light, np.float32)
    L /= np.linalg.norm(L)
    ndl = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0.0, 1.0)

    # (b) internal structure -- where the arms, head and drape folds are.
    # Relief alone knows only the outline and renders an anonymous blob, which
    # is poor guidance. Luminance carries that structure, but it also carries
    # texture, which is exactly what must not reach LTX. So: smooth hard with an
    # edge-preserving filter at a scale tied to subject size, then crush the
    # contrast into a narrow band. Large forms survive; brocade and print do not.
    lum = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    sig = max(scale * 0.55, 3.0)
    sm = cv2.bilateralFilter(lum, d=0, sigmaColor=90.0, sigmaSpace=sig)
    sm = cv2.GaussianBlur(sm, (0, 0), max(sig * 0.5, 2.0))
    inside = sm[m > 0]
    lo, hi = np.percentile(inside, 4), np.percentile(inside, 96)
    form = np.clip((sm - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    form = 0.5 + (form - 0.5) * 0.55                    # crushed contrast

    grey = 40.0 + 118.0 * ndl + 22.0 * h + 74.0 * form
    return np.repeat(np.clip(grey, 0, 255)[:, :, None], 3, axis=2)


def _card_geometry(cw: int, ch: int, orbit: Orbit, card_height: float):
    """Card corners in world space, plus the source quad for the homography.

    The card faces the FRAME-0 view ray rather than standing upright -- see
    render_orbit for why that matters once elevation is non-zero.
    """
    card_w = card_height * cw / ch
    hw, hh = card_w / 2.0, card_height / 2.0

    aim = np.array([0.0, 0.0, orbit.aim_z])
    a0 = orbit.ray_angle()
    C0 = np.array([orbit.radius * math.cos(a0), orbit.radius * math.sin(a0),
                   orbit.cam_z])
    fwd0 = aim - C0
    fwd0 /= np.linalg.norm(fwd0)
    right = np.cross(fwd0, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd0)
    up /= np.linalg.norm(up)

    corners = np.array([
        aim - right * hw + up * hh,
        aim + right * hw + up * hh,
        aim + right * hw - up * hh,
        aim - right * hw - up * hh,
    ])
    src = np.array([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]], np.float32)
    return corners, src, card_w


def _compose(bgr, al, corners, src, R, t, orbit: Orbit, polys=None):
    """Warp the card into one camera view, over the backdrop.

    Scene geometry is depth-sorted around the card rather than simply drawn
    underneath it: floor quads nearer to the camera than the subject must paint
    over the card's base, or the subject looks like it is floating in front of
    the floor instead of standing on it.
    """
    fx, fy, cx, cy = orbit.intrinsics()
    W, H = orbit.width, orbit.height
    canvas = np.full((H, W, 3), float(BG_GREY), np.float32)

    uv, z = project(R, t, corners, fx, fy, cx, cy)
    card_depth = float(z.mean()) if (z > 1e-3).all() else 1e9

    after = []
    if polys:
        after = _draw_polys(canvas, polys, R, t, orbit, card_depth)

    area = 0.0
    if (z > 1e-3).all():
        x_, y_ = uv[:, 0], uv[:, 1]
        area = 0.5 * abs(np.dot(x_, np.roll(y_, -1)) - np.dot(y_, np.roll(x_, -1)))
    if area > 25.0:
        Hm = cv2.getPerspectiveTransform(src, uv.astype(np.float32))
        wc = cv2.warpPerspective(bgr, Hm, (W, H), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT)
        wa = cv2.warpPerspective(al, Hm, (W, H), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT)[..., None]
        canvas = wc * wa + canvas * (1.0 - wa)

    for pts, shade in after:                 # nearer than the card
        cv2.fillConvexPoly(canvas, pts, (BG_GREY * shade * 2.6,) * 3, cv2.LINE_AA)

    return np.clip(canvas, 0, 255).astype(np.uint8), area


def card_height_full(orbit: Orbit, src_w: int = 0, src_h: int = 0,
                     src_vfov_deg: float = 0.0) -> float:
    """Card height that makes the card fill the frame-0 view ("contain").

    DECOUPLED from the orbit camera. The card is what every source pixel is
    projected onto, so its angular size decides which 3D DIRECTION each pixel
    is treated as having come from -- that is a property of the lens that took
    the photograph, not of the lens we have chosen to orbit with. Sizing it by
    orbit.vfov_deg meant changing the orbit fov silently reshaped the point
    cloud, and any hand-set fov built the scene through the wrong lens.

    src_vfov_deg is the MEASURED vertical fov of the source image. When it is
    supplied the card subtends exactly that, so the cloud is correct whatever
    the orbit is doing. The orbit still owns the render camera, the canvas and
    the COLMAP intrinsics -- this only fixes what the pixels are unprojected
    through. Falls back to the orbit fov when nothing was measured, which is
    the old behaviour.

    The consequence is deliberate: when the two fovs differ, control frame 0 no
    longer reproduces the source at exactly the same scale. That equality was
    only ever a side effect of forcing one lens to stand in for the other.

    The cutout is the full source frame, so a frustum-filling card makes
    control frame 0 reproduce the source image at its true placement and scale.
    Anything else (the old fill-86%-and-centre behaviour) shows the video model
    a first frame that disagrees with the image_url reference, and it
    reconciles the two by warping the subject.

    When the source is wider than the canvas, height-filling would overflow and
    crop the sides (~7% on a 0.805-aspect source in a 3:4 canvas), so the card
    is scaled to CONTAIN: the whole source stays visible, letterboxed on grey.
    """
    d = math.hypot(orbit.radius, orbit.aim_z - orbit.cam_z)
    vf = float(src_vfov_deg) if src_vfov_deg and src_vfov_deg > 0         else orbit.vfov_deg
    h = 2.0 * d * math.tan(math.radians(vf) / 2.0)
    if src_w and src_h:
        ar_src = src_w / src_h
        ar_canvas = orbit.width / orbit.height
        if ar_src > ar_canvas:
            h *= ar_canvas / ar_src
    return h


def ground_z_for(orbit: Orbit, card_height: float,
                 feet_frac: float = 0.0) -> float:
    """Floor height in world units.

    feet_frac is how far up the card the subject's lowest pixel sits (0 = the
    card's bottom edge). With a full-frame card the subject's feet are rarely
    at the frame's bottom edge, so without this the floor would sit below the
    feet and the subject would appear to hover above it.
    """
    bottom = orbit.aim_z - card_height / 2.0
    return bottom + card_height * max(0.0, min(0.9, feet_frac))


def render_reference(cut: Path, orbit: Orbit, dst: Path,
                     card_height: float = 3.6, backdrop: str = "grey",
                     feet_frac: float = 0.0) -> dict:
    """Frame 0 in COLOUR on the flat grey backdrop -- the fal `image_url`.

    Why this exists: cutout.png is RGBA and still carries the ORIGINAL image in
    its RGB channels, with alpha only marking the subject. fal flattens the
    alpha away, so sending the cutout handed LTX the original background and it
    faithfully reproduced it in the generated video.

    Rendering frame 0 instead gives a reference that (a) has the same flat grey
    backdrop as the control video, and (b) is framed identically to control
    frame 0 -- which is what the endpoint wants, since image_url anchors the
    first frame.
    """
    rgba = cv2.imread(str(cut), cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.shape[2] < 4:
        raise RuntimeError("cutout must be RGBA")
    ch, cw = rgba.shape[:2]
    al = rgba[:, :, 3:4].astype(np.float32) / 255.0
    bgr = rgba[:, :, :3].astype(np.float32)

    corners, src, _ = _card_geometry(cw, ch, orbit, card_height)
    polys = (scene_polys(orbit, ground_z_for(orbit, card_height, feet_frac))
             if backdrop == "scene" else None)
    R, t = orbit.poses()[0]
    img, area = _compose(bgr, al, corners, src, R, t, orbit, polys)

    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_PNG_COMPRESSION, 4])

    flat = img.reshape(-1, 3)
    corner_px = np.concatenate([img[:8, :8].reshape(-1, 3),
                                img[:8, -8:].reshape(-1, 3),
                                img[-8:, :8].reshape(-1, 3),
                                img[-8:, -8:].reshape(-1, 3)])
    return {"size": [orbit.width, orbit.height],
            "backdrop_grey": int(BG_GREY),
            "corner_std": round(float(corner_px.std()), 3),
            "mean": [round(float(v), 1) for v in flat.mean(axis=0)]}


def _clear_frames(out_dir: Path) -> int:
    """Delete frame_*.png left over from a previous render.

    frames_to_mp4 globs the directory, so a shorter run would otherwise inherit
    the tail of a longer one and hand fal a clip whose frame count disagrees
    with the pose count -- misaligning every camera.
    """
    stale = sorted(out_dir.glob("frame_*.png"))
    for f in stale:
        f.unlink()
    return len(stale)


def render_orbit(cut: Path, orbit: Orbit, out_dir: Path,
                 card_height: float = 3.6, clay_mode: str = "shaded",
                 backdrop: str = "grey", feet_frac: float = 0.0) -> dict:
    """Render the cutout as a flat card in 3D, orbited by the camera.

    The card is oriented PERPENDICULAR TO THE FRAME-0 VIEW RAY, not simply stood
    upright. That matters as soon as elevation is non-zero: the source image
    already contains the perspective of the camera that shot it, so standing the
    card vertically and then viewing it from 15 degrees below would apply that
    foreshortening a second time. Facing the card at camera 0 makes frame 0
    reproduce the source image, whatever the elevation.

    Deliberately crude beyond that: a flat card compresses to a sliver at 90 and
    270 degrees and its back is the mirrored front. The homography reproduces
    both without special-casing.

    Rendered from the SAME camera model written to COLMAP, so the conditioning
    video and the training poses cannot disagree.
    """
    rgba = cv2.imread(str(cut), cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.shape[2] < 4:
        raise RuntimeError("cutout must be RGBA")
    ch, cw = rgba.shape[:2]
    al = (rgba[:, :, 3:4].astype(np.float32)) / 255.0
    bgr = clay(rgba[:, :, :3].astype(np.float32), al, clay_mode)

    card_w = card_height * cw / ch
    hw, hh = card_w / 2.0, card_height / 2.0

    # basis perpendicular to the frame-0 view ray
    aim = np.array([0.0, 0.0, orbit.aim_z])
    a0 = orbit.ray_angle()
    C0 = np.array([orbit.radius * math.cos(a0), orbit.radius * math.sin(a0),
                   orbit.cam_z])
    fwd0 = aim - C0
    fwd0 /= np.linalg.norm(fwd0)
    right = np.cross(fwd0, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd0)
    up /= np.linalg.norm(up)

    corners = np.array([
        aim - right * hw + up * hh,
        aim + right * hw + up * hh,
        aim + right * hw - up * hh,
        aim - right * hw - up * hh,
    ])
    src = np.array([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]], np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    _clear_frames(out_dir)
    W, H = orbit.width, orbit.height
    ground = ground_z_for(orbit, card_height, feet_frac)
    polys = scene_polys(orbit, ground) if backdrop == "scene" else None
    # A floor is only meaningful when the camera is above it. At strongly
    # negative elevation the camera sits below the subject's feet and would be
    # looking at the underside of the plane.
    floor_ok = orbit.cam_z > ground + 0.05

    skipped = 0
    areas = []
    for i, (R, t) in enumerate(orbit.poses(), start=1):
        # _compose owns the card warp, the backdrop, and the depth ordering
        # between them, so the reference frame and the video cannot diverge.
        img, area = _compose(bgr, al, corners, src, R, t, orbit, polys)
        if area <= 25.0:
            skipped += 1
        areas.append(area)
        cv2.imwrite(str(out_dir / f"frame_{i:04d}.png"), img,
                    [cv2.IMWRITE_PNG_COMPRESSION, 3])

    thin = int(sum(1 for a in areas if 25.0 < a < 0.02 * W * H))
    return {"frames": orbit.frames, "blank_edge_on": skipped, "thin_frames": thin,
            "card_w": card_w, "card_h": card_height, "clay": clay_mode,
            "backdrop": backdrop, "ground_z": round(ground, 3),
            "floor_visible": bool(floor_ok),
            "scene_polys": len(polys) if polys else 0,
            "elev_deg": orbit.elev_deg, "cam_z": round(orbit.cam_z, 4),
            "elev_sweep_deg": float(orbit.elev_sweep_deg),
            "elev_span": [round(v, 2) for v in orbit.elev_span()],
            "quad_area_min": float(min(areas)), "quad_area_max": float(max(areas))}


# ----------------------------------------------------- depth point cloud ---
_DEPTH_MODEL = None


def _depth_model(size: str = "small", device: str = "cpu"):
    """Depth-Anything-V2 via transformers.

    Held on the CPU between calls, like every other model here. It used to
    load onto the GPU and stay there for the life of the process, which cost
    6.7 GB of dedicated VRAM that nothing could reclaim -- torch had freed it
    internally, so mem_get_info reported it available, but the caching
    allocator kept it reserved from Windows and Task Manager rightly showed it
    as used.
    """
    global _DEPTH_MODEL
    tag = f"depth-anything/Depth-Anything-V2-{size.capitalize()}-hf"
    if _DEPTH_MODEL is None or _DEPTH_MODEL[3] != tag:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        proc = AutoImageProcessor.from_pretrained(tag)
        mod = AutoModelForDepthEstimation.from_pretrained(tag)
        _DEPTH_MODEL = (mod.eval(), proc, "cpu", tag)
    return _DEPTH_MODEL


def estimate_depth(img_bgr: np.ndarray, size: str = "small") -> np.ndarray:
    """Relative depth map at image resolution. Larger value = closer."""
    import torch
    from PIL import Image
    mod, proc, _dev, _ = _depth_model(size)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mod = mod.to(dev)
    pil = Image.fromarray(img_bgr[..., ::-1])
    try:
        with torch.no_grad():
            inputs = proc(images=pil, return_tensors="pt").to(dev)
            pred = mod(**inputs).predicted_depth[0].float().cpu().numpy()
    finally:
        # Give the card back. Same policy as the matte and metric-depth paths.
        mod.to("cpu")
        if dev == "cuda":
            torch.cuda.empty_cache()
    return cv2.resize(pred, (img_bgr.shape[1], img_bgr.shape[0]),
                      interpolation=cv2.INTER_LINEAR)


_MOGE_MODEL = None
MOGE_TAG = "Ruicheng/moge-2-vitl-normal"

# Beyond this many orbit radii behind the card plane, metric distances are
# compressed toward a "far dome". A ridge 2 km away contributes essentially no
# parallax to an orbit a few units wide, so placing it at its literal distance
# would only waste the point budget and shrink it to nothing. True scale is
# kept where parallax actually matters -- near the subject -- and the far field
# drifts slowly, which is all the video model needs from it.
FAR_DOME_RADII = 6.0

# Depth engines that return real meters, and so go through the metric lift
# (subject anchored to the card plane, far field tanh-compressed) rather than
# the percentile-normalised relative path.
METRIC_MODELS = ("moge2", "sharp", "da3-metric")

# The depth models the author workspace offers, by the id the client sends.
#   moge2, da3-metric -- metric: through the anchored metric lift.
#   da2, da2-large, da3-mono -- relative: through the percentile-normalised
#     path in depth_cloud_points, where relief is world units across the photo.
# Licences: Depth Anything V2 Small, DA3 Metric-Large and DA3 Mono-Large are
# Apache-2.0; V2 Base and Large are CC-BY-NC-4.0, which is why V2 Large is
# labelled non-commercial in the UI.
DEPTH_MODELS = {
    "moge2": "MoGe-2",
    "da2": "Depth Anything V2 Small",
    "da2-large": "Depth Anything V2 Large",
    "da3-metric": "Depth Anything 3 Metric Large",
    "da3-mono": "Depth Anything 3 Mono Large",
}
_DA2_SIZE = {"da2": "small", "da2-large": "large"}
DA3_TAGS = {"da3-metric": "depth-anything/DA3METRIC-LARGE",
            "da3-mono": "depth-anything/DA3MONO-LARGE"}
# Longest side DA3 runs at (rounded to its 14 px patch). 1008 measured 0.4 s
# and a 3.2 GB peak on the 5070; the default 504 loses fine relief.
DA3_RES = 1008
DA3_VRAM_BYTES = 4.5e9
_DA3_MODEL = None


def da3_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("depth_anything_3") is not None


def _da3_model(tag: str):
    """Depth Anything 3 (ByteDance Seed), held on the CPU between calls.

    DA3's api module imports its export tools -- moviepy 1.x, open3d, a
    pinned gsplat -- at load time, and pins numpy<2. A depth-only lift never
    exports, so it is installed with --no-deps and, when those tools are
    missing, the export module is replaced by a placeholder that refuses.
    """
    global _DA3_MODEL
    if _DA3_MODEL is not None and _DA3_MODEL[1] == tag:
        return _DA3_MODEL[0]
    if not da3_available():
        raise RuntimeError("Depth Anything 3 is not installed - see "
                           "requirements-optional.txt, or pick another depth model")
    import importlib
    import sys
    import types
    name = "depth_anything_3.utils.export"
    if name not in sys.modules:
        try:
            importlib.import_module(name)
        except Exception:
            stub = types.ModuleType(name)

            def _no_export(*a, **k):
                raise RuntimeError("DA3 export is not available in this build")
            stub.export = _no_export
            sys.modules[name] = stub
    from depth_anything_3.api import DepthAnything3
    _DA3_MODEL = None                 # drop the other size before loading
    _DA3_MODEL = (DepthAnything3.from_pretrained(tag).eval(), tag)
    return _DA3_MODEL[0]


def estimate_depth_da3(src: Path, img_bgr: np.ndarray, model: str, log=None):
    """Depth Anything 3. Returns (depth, valid) at image resolution.

    da3-metric: DA3METRIC-LARGE's output is metric once multiplied by
    focal/300 (focal in pixels at the size it ran at). The measured lens is
    used for that; the lift anchors the subject to the card plane, so the
    scale only sets the meters in the log, never the geometry.
    da3-mono: DA3MONO-LARGE, relative DEPTH (larger = farther) -- not
    disparity like V2. valid is False where DA3 sees sky.
    """
    import torch
    m = _da3_model(DA3_TAGS[model])
    dev = "cpu"
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        if free > DA3_VRAM_BYTES:
            dev = "cuda"
        elif log:
            log(f"depth: only {free / 1e9:.1f} GB of VRAM free - Depth Anything 3 "
                f"runs on the CPU, which is slow")
    m = m.to(dev)
    try:
        p = m.inference([np.ascontiguousarray(img_bgr[..., ::-1])],
                        process_res=DA3_RES)
    finally:
        if dev == "cuda":
            m.to("cpu")
            torch.cuda.empty_cache()
    ih, iw = img_bgr.shape[:2]
    raw = p.depth[0].astype(np.float32)
    ph, pw = raw.shape
    depth = cv2.resize(raw, (iw, ih), interpolation=cv2.INTER_LINEAR)
    valid = depth > 1e-6
    if p.sky is not None:
        sky = cv2.resize(p.sky[0].astype(np.float32), (iw, ih),
                         interpolation=cv2.INTER_LINEAR)
        valid &= sky < 0.3            # DA3's own non-sky threshold
    if model == "da3-metric":
        f_px, _ = _source_focal_px(Path(src), iw, ih, None)
        if f_px > 0:
            depth = depth * (f_px * (pw / iw) / 300.0)
        elif log:
            log("depth: no measured lens - DA3 metric depth is in relative "
                "units until the lens is measured (the lift is unaffected)")
    return depth, valid


def depth_model_id(value) -> str:
    """A depth model id this build can run; anything else falls back to MoGe-2."""
    v = str(value or "").strip().lower()
    return v if v in DEPTH_MODELS else "moge2"


def _moge_model():
    """MoGe-2 (Microsoft, NeurIPS 2025): METRIC depth from a single image.

    Held on the CPU between calls and moved to the GPU only for the inference
    itself. This box shares its VRAM with ComfyUI, Ollama and Brush, and a
    once-per-render hop costs far less than squatting on 1.3 GB.
    """
    global _MOGE_MODEL
    if _MOGE_MODEL is None:
        from moge.model.v2 import MoGeModel
        _MOGE_MODEL = MoGeModel.from_pretrained(MOGE_TAG).eval()
    return _MOGE_MODEL


_DEPTH_MEM: dict = {}


def _depth_metric_cached(src: Path, bgr: np.ndarray, log=None):
    """MoGe-2 depth for THIS source file -- see _depth_cached."""
    return _depth_cached(src, bgr, "moge2", log=log)


def _depth_cached(src: Path, bgr: np.ndarray, model: str, log=None):
    """Depth for THIS source file and model: memory, then disk, then the model.

    Returns (depth, valid). valid is None for Depth Anything, which has no
    sky mask. Every model's output depends on the pixels alone, so it is keyed
    by the file's modification time and size. Every relift used to re-run the
    network -- seconds of GPU per slider release -- and a server restart threw
    the result away. The disk copy lives in <project>/cache/depth_<model>.npz
    and a new upload changes the key, so a stale depth map can never be read
    back for a different photo.
    """
    model = depth_model_id(model)
    name = DEPTH_MODELS[model]

    def compute():
        if model == "moge2":
            return estimate_depth_metric(bgr)
        if log:
            log(f"depth: {name} on {Path(src).name}")
        if model in DA3_TAGS:
            return estimate_depth_da3(src, bgr, model, log=log)
        return estimate_depth(bgr, _DA2_SIZE[model]), None

    src = Path(src)
    try:
        stt = src.stat()
        key = f"{src.name}|{stt.st_mtime_ns}|{stt.st_size}|{model}"
    except OSError:
        return compute()
    mem = f"{src.resolve()}|{model}"
    hit = _DEPTH_MEM.get(mem)
    if hit and hit[0] == key:
        return hit[1], hit[2]

    def remember(depth, valid):
        # a few entries, so flipping between models on one photo is instant
        while len(_DEPTH_MEM) >= 4:
            _DEPTH_MEM.pop(next(iter(_DEPTH_MEM)))
        _DEPTH_MEM[mem] = (key, depth, valid)

    disk = src.parent / "cache" / f"depth_{model}.npz"
    if disk.exists():
        try:
            z = np.load(disk, allow_pickle=False)
            if str(z["key"]) == key:
                depth = z["depth"]
                valid = z["valid"].astype(bool) if "valid" in z.files else None
                remember(depth, valid)
                if log:
                    log(f"depth: {name} depth read from the cache  @ {disk.resolve()}")
                return depth, valid
        except Exception:
            pass                          # unreadable cache: recompute below
    depth, valid = compute()
    try:
        disk.parent.mkdir(parents=True, exist_ok=True)
        tmp = disk.with_name(f"depth_{model}.tmp.npz")
        extra = {} if valid is None else {"valid": valid.astype(np.uint8)}
        np.savez_compressed(tmp, key=np.array(key),
                            depth=depth.astype(np.float32), **extra)
        os.replace(tmp, disk)
        if log:
            log(f"file: wrote cache/{disk.name} ({name} depth, "
                f"{disk.stat().st_size / 1e6:.2f} MB)  @ {disk.resolve()}")
    except Exception as e:
        if log:
            log(f"depth: could not write the depth cache ({e})")
    remember(depth, valid)
    return depth, valid


def estimate_depth_metric(img_bgr: np.ndarray):
    """Metric depth via MoGe-2. Returns (depth_m, valid) at image resolution.

    depth_m is in METERS -- MoGe-2 carries an explicit metric-scale head -- and
    valid is False where depth is undefined, which in practice means sky. This
    is what scene-scale displacement needs: Depth-Anything returns relative
    inverse depth, which collapses a 50 m tree line and a 5 km ridge onto
    nearly the same value once it is percentile-normalised.
    """
    import torch
    mod = _moge_model()
    dev = "cpu"
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        if free > 3e9:
            dev = "cuda"
    mod = mod.to(dev)
    img = torch.tensor(np.ascontiguousarray(img_bgr[..., ::-1]),
                       dtype=torch.float32, device=dev).permute(2, 0, 1) / 255.0
    with torch.no_grad():
        out = mod.infer(img)
    depth = out["depth"].float().cpu().numpy()
    valid = out["mask"].cpu().numpy().astype(bool)
    if dev == "cuda":
        mod.to("cpu")
        torch.cuda.empty_cache()
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    valid = valid & (depth > 1e-6)
    return depth, valid


_SHARP_MODEL = None

# SHARP peaked at 7.4 GB on a 913x735 image (it runs a ViT at 1536x1536
# internally). Below this much free VRAM it goes to the CPU rather than
# OOM-ing mid-render and taking the server's other work down with it.
SHARP_VRAM_BYTES = 8.5e9


def free_vram(log=None) -> dict:
    """Hand every cached model back to the CPU and release the allocator.

    torch's caching allocator keeps freed blocks RESERVED from the OS, so a
    process that has finished its GPU work still shows the peak in Task
    Manager. Nothing else on the machine can use that until it is released.
    """
    import torch
    before = 0.0
    if torch.cuda.is_available():
        before = torch.cuda.memory_reserved() / 1e9
    for holder in ("_DEPTH_MODEL", "_MOGE_MODEL", "_SHARP_MODEL", "_DA3_MODEL"):
        obj = globals().get(holder)
        if obj is None:
            continue
        m = obj[0] if isinstance(obj, tuple) else obj
        try:
            m.to("cpu")
        except Exception:
            pass
    # Every dict-shaped model cache in this module, found by name so a cache
    # added later is covered without editing this list.
    for name, obj in list(globals().items()):
        if not (name.endswith("_CACHE") and isinstance(obj, dict)):
            continue
        for m in list(obj.values()):
            try:
                m.to("cpu")
            except Exception:
                pass
    _PREVIEW_CACHE.clear()
    import gc
    gc.collect()
    after = 0.0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        after = torch.cuda.memory_reserved() / 1e9
        free, total = torch.cuda.mem_get_info()
    else:
        free = total = 0
    if log:
        log(f"vram: released {max(0.0, before - after):.2f} GB "
            f"(reserved {before:.2f} -> {after:.2f} GB); "
            f"device now {free / 1e9:.2f} GB free of {total / 1e9:.2f} GB")
    return {"released_gb": round(max(0.0, before - after), 3),
            "reserved_gb": round(after, 3),
            "free_gb": round(free / 1e9, 3), "total_gb": round(total / 1e9, 3)}


def sharp_available() -> bool:
    try:
        import sharp.cli.predict  # noqa: F401
        return True
    except Exception:
        return False


def _sharp_model():
    """Apple SHARP (ml-sharp): metric per-pixel geometry from one image.

    SHARP regresses a full 3D Gaussian representation, but we only want its
    GEOMETRY -- the metric mean vectors -- to displace our own point cloud.
    V1 rejected SHARP as a FINAL splat ("falls apart the moment you orbit");
    as a control signal that failure mode is acceptable, since LTX replaces
    appearance and only needs plausible parallax.
    """
    global _SHARP_MODEL
    if _SHARP_MODEL is None:
        import torch
        from sharp.cli.predict import DEFAULT_MODEL_URL
        from sharp.models import PredictorParams, create_predictor
        sd = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL,
                                                progress=False)
        m = create_predictor(PredictorParams())
        m.load_state_dict(sd)
        _SHARP_MODEL = m.eval()
    return _SHARP_MODEL


def estimate_depth_sharp(img_bgr: np.ndarray, f_px: float = 0.0, log=None):
    """Metric depth via SHARP. Returns (depth_m, valid), same contract as MoGe.

    SHARP hands back an unordered set of metric points rather than a depth
    image, and the ordering of its mean vectors does not reshape cleanly onto a
    pixel grid. So instead of guessing a layout we PROJECT the points back
    through the intrinsics predict_image unprojected them with -- the original
    f_px about the image centre -- and z-buffer them into a map. That is
    ordering-independent and lands 95%+ of the points in frame.

    f_px comes from EXIF when the file has it and falls back to a 30 mm
    equivalent. It matters: SHARP's scale is metric *given* the focal, so a
    wrong focal rescales the whole scene. It does not affect the ORBIT though,
    because the subject's median depth is normalised onto the card plane
    downstream -- only the relative near/far spread survives, which is the part
    we actually use.
    """
    import torch
    from sharp.cli.predict import predict_image

    mod = _sharp_model()
    dev = "cpu"
    free = 0.0
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        if free > SHARP_VRAM_BYTES:
            dev = "cuda"
    if dev == "cpu" and log:
        # This fallback used to be silent, and on the CPU a ViT at 1536x1536
        # takes minutes -- indistinguishable from a hang. Anything else on the
        # GPU (ComfyUI, a game, another trainer) is enough to trigger it, so
        # say the numbers and name the cure.
        log(f"depth: SHARP has only {free / 1e9:.1f} GB of VRAM free and wants "
            f"{SHARP_VRAM_BYTES / 1e9:.1f} GB -- running on the CPU instead, "
            f"which takes MINUTES rather than seconds. Close whatever else is "
            f"holding the GPU (ComfyUI is the usual one) and re-run.")
    elif log:
        log(f"depth: SHARP on the GPU ({free / 1e9:.1f} GB free)")
    mod = mod.to(dev)

    H, W = img_bgr.shape[:2]
    if not f_px:
        f_px = 0.75 * max(W, H)      # ~30 mm equivalent, SHARP's own fallback
    rgb = np.ascontiguousarray(img_bgr[..., ::-1])
    with torch.no_grad():
        g = predict_image(mod, rgb, float(f_px), torch.device(dev))
    P = g.mean_vectors.reshape(-1, 3).float().cpu().numpy()
    if dev == "cuda":
        mod.to("cpu")
        torch.cuda.empty_cache()

    X, Y, Z = P[:, 0], P[:, 1], P[:, 2]
    ok = np.isfinite(Z) & (Z > 1e-6)
    u = f_px * X[ok] / Z[ok] + W / 2.0
    v = f_px * Y[ok] / Z[ok] + H / 2.0
    z = Z[ok]
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    ui, vi, zi = u[inb].astype(np.int32), v[inb].astype(np.int32), z[inb]

    # z-buffer by painter's order: write far first so the nearest point wins
    depth = np.zeros((H, W), np.float32)
    order = np.argsort(-zi)
    depth[vi[order], ui[order]] = zi[order]
    valid = depth > 0

    # Close the scatter's pinholes. A hole is a missing SAMPLE, not a real
    # depth discontinuity, and a zero there would fling that pixel into the
    # camera during the lift. Grey-dilate pulls each gap to its nearest filled
    # neighbour, which is the right guess for a one-pixel gap and converges in
    # a couple of passes at ~90% initial coverage.
    for _ in range(4):
        if valid.all():
            break
        grown = cv2.dilate(depth, np.ones((3, 3), np.uint8))
        depth = np.where(valid, depth, grown)
        valid = depth > 0
    if not valid.all():
        depth[~valid] = float(np.median(depth[valid])) if valid.any() else 1.0
        valid = np.ones_like(valid)
    return depth, valid


def estimate_fov(src: Path) -> dict:
    """Camera field of view from MoGe-2's predicted intrinsics.

    MoGe-2 returns NORMALISED intrinsics -- fx by image width, fy by height --
    so the vertical FOV is 2*atan(0.5/fy). This is the number GeoCalib cannot
    give reliably: its focal uncertainty is routinely larger than its estimate,
    which is why the pipeline never auto-applied it.
    """
    import torch
    bgr = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise RuntimeError(f"cannot read {src}")
    bgr = bgr[:, :, :3]
    h, w = bgr.shape[:2]

    mod = _moge_model()
    dev = "cpu"
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        if free > 3e9:
            dev = "cuda"
    mod = mod.to(dev)
    img = torch.tensor(np.ascontiguousarray(bgr[..., ::-1]),
                       dtype=torch.float32, device=dev).permute(2, 0, 1) / 255.0
    with torch.no_grad():
        out = mod.infer(img)
    K = out["intrinsics"].detach().cpu().numpy()
    if dev == "cuda":
        mod.to("cpu")
        torch.cuda.empty_cache()

    fx_n, fy_n = float(K[0, 0]), float(K[1, 1])
    vfov = 2.0 * math.degrees(math.atan(0.5 / max(fy_n, 1e-6)))
    hfov = 2.0 * math.degrees(math.atan(0.5 / max(fx_n, 1e-6)))
    return {"vfov_deg": round(vfov, 2), "hfov_deg": round(hfov, 2),
            "focal_px": round(fy_n * h, 1), "width": w, "height": h,
            "model": MOGE_TAG, "device": dev}


def _anchor_depth(src: Path, depth: np.ndarray, valid: np.ndarray) -> float:
    """Median metric depth of the SUBJECT.

    This is the depth that maps onto the card plane, so the subject orbits
    where the flat card used to sit and the rest of the scene lands in
    proportion around it. Uses the cutout alpha when one exists, since it
    outlines the subject exactly; falls back to a central window otherwise.
    """
    h, w = depth.shape
    mask = None
    cut = Path(src).parent / "cutout.png"
    if cut.exists():
        a = cv2.imread(str(cut), cv2.IMREAD_UNCHANGED)
        if a is not None and a.ndim == 3 and a.shape[2] == 4:
            al = cv2.resize(a[:, :, 3], (w, h), interpolation=cv2.INTER_NEAREST)
            m = (al > 128) & valid
            if int(m.sum()) > 200:
                mask = m
    if mask is None:
        m = np.zeros_like(valid)
        m[int(h * 0.30):int(h * 0.85), int(w * 0.30):int(w * 0.70)] = True
        mask = m & valid
        if int(mask.sum()) < 50:
            mask = valid
    return float(np.median(depth[mask]))


# How many points the last lift's prune removed, for logging. The lift
# returns a fixed tuple that several callers unpack positionally, so widening
# it would break them all.
_PRUNE_STAT = {"dropped": 0, "kept": 0}


_GROUND_STAT = {"moved": 0, "z0": 0.0, "tilt_deg": 0.0}


def bend_span(C0: np.ndarray, P: np.ndarray) -> tuple:
    """(rNear, rFar): the horizontal distance range the bend is defined over.

    Percentiles, not min/max, so one stray point cannot define the pivot and
    collapse the whole ramp. The viewport computes this identically from the
    same points, which is what keeps the preview honest.
    """
    r = np.hypot(P[:, 0] - C0[0], P[:, 1] - C0[1])
    return float(np.percentile(r, 2.0)), float(np.percentile(r, 98.0))


def bend_scene(C0: np.ndarray, rays: np.ndarray, P: np.ndarray,
               strength: float, falloff: float = 1.0,
               coverage: float = 1.0) -> np.ndarray:
    """Hinge the scene at its far edge and lift/drop the front.

    strength is in world units at full deflection; sign chooses up or down.
    Each point is moved ALONG ITS OWN RAY to the target height, so camera 0
    cannot see the point move and frame 0 keeps its geometry exactly.

    One honest caveat, measured: what CAN change in frame 0 is which splat
    wins a pixel, because moving points by different amounts along their rays
    reorders them in depth. At ordinary settings this is 0.06-0.08% of pixels;
    at an extreme (2 units, falloff 0.4, coverage 0.25) it reached 2.9%,
    spread over 1,672 separate regions whose largest was 16 px -- a single
    splat's footprint. So it is scattered z-order flipping, not a warp of the
    subject. Nothing shifts or stretches; individual points swap places.
    """
    if not strength or not len(P):
        return P
    rNear, rFar = bend_span(C0, P)
    span = max(rFar - rNear, 1e-6) * max(min(float(coverage), 1.0), 1e-3)
    r = np.hypot(P[:, 0] - C0[0], P[:, 1] - C0[1])
    u = np.clip((rFar - r) / span, 0.0, 1.0)
    w = np.power(u, max(float(falloff), 0.05))
    dz = float(strength) * w

    # Solve the ray parameter that reaches the target height, then CLAMP it to
    # stay in front of the camera.
    #
    # The clamp earns its place at strong settings. A ray pointing downward can
    # only gain height by travelling BACKWARDS along itself -- through the
    # camera and out the other side -- where the direction inverts and the
    # point projects to a mirrored pixel instead of its own. Measured, points
    # the unclamped formula would send behind camera 0:
    #     bend 0.5      0        bend 2.0    5,468
    #     bend 1.0      0        bend 3.0   29,000
    # so it is inert for gentle bends and load-bearing for strong ones.
    #
    # Clamped, a point that cannot reach the target height simply travels as
    # far along its ray as it safely can. It stays on its own line of sight, so
    # frame 0 is still exact, and the bend degrades smoothly instead of
    # destroying half the scene.
    t_now = np.einsum("ij,ij->i", P - C0[None, :], rays)
    dzr = rays[:, 2]
    safe = np.where(np.abs(dzr) < 0.08, np.where(dzr < 0, -0.08, 0.08), dzr)
    t_new = t_now + dz / safe
    t_new = np.clip(t_new, 0.05 * np.maximum(t_now, 1e-6),
                    4.0 * np.maximum(t_now, 1e-6))
    return C0[None, :] + rays * t_new[:, None]


def level_ground(C0: np.ndarray, rays: np.ndarray, P: np.ndarray,
                 strength: float, band: float = 0.45) -> np.ndarray:
    """Re-place floor points on a level plane, along their own rays.

    Monocular depth cannot get a receding floor right, and the error is
    invisible from the camera that produced it -- which is the property this
    pipeline is built on. From any other angle it shows as a floor that tilts
    and droops away. Measured on a real lift the dominant ground plane came
    out with normal [0, 0.35, -0.94]: tilted ~20 degrees and sloping toward
    the front. Every depth engine here does it, so it is the problem and not
    the model.

    The correction is free because of the same invariant. Each point sits at
    some distance along its own ray from camera 0, so for floor pixels that
    distance is simply re-solved as the ray's intersection with a level
    plane. The point lands where the floor really is and camera 0 cannot tell,
    because motion along a view ray is invisible to the camera that defines
    it. Frame 0 still reproduces the source exactly, at any strength.

    The target is HORIZONTAL at the robust median height of the floor band.
    Snapping to the best-fit plane instead would faithfully preserve the tilt
    that is the artifact.
    """
    if strength <= 0 or not len(P):
        return P
    z = P[:, 2]
    # The floor is the low tail, but not its extreme: the very lowest points
    # are usually stray depth errors below the real surface.
    lo, hi = np.percentile(z, 2.0), np.percentile(z, 35.0)
    seed = z[(z >= lo) & (z <= hi)]
    if len(seed) < 200:
        return P
    z0 = float(np.median(seed))

    dz = rays[:, 2]
    hits = np.abs(dz) > 1e-3                 # a ray parallel to the plane
    on_floor = hits & (np.abs(z - z0) <= band)   # never meets it stably
    if int(on_floor.sum()) < 200:
        return P
    t_now = np.linalg.norm(P - C0[None, :], axis=1)
    t_plane = np.where(hits, (z0 - C0[2]) / np.where(hits, dz, 1.0), t_now)
    # A solution behind the camera, or absurdly far, means this ray does not
    # really meet the floor ahead; leave those points where they are.
    sane = on_floor & (t_plane > 0) & (t_plane < np.percentile(t_now, 99.5) * 4)
    if not sane.any():
        return P
    k = float(min(1.0, strength))
    t_new = t_now * (1.0 - k) + t_plane * k
    out = P.copy()
    out[sane] = C0[None, :] + rays[sane] * t_new[sane][:, None]
    _GROUND_STAT["moved"] = int(sane.sum())
    _GROUND_STAT["z0"] = round(z0, 4)
    return out


def prune_isolated(P: np.ndarray, origin: np.ndarray, strength: float,
                   k: int = 8) -> np.ndarray:
    """Keep-mask dropping points with no neighbours (depth-edge flying pixels).

    strength is 0..1; 0 disables. Returns a boolean mask over P.

    Scale-aware by construction: a pinhole lift spaces points in proportion to
    their distance from the camera, so the raw neighbour distance is larger
    for far surfaces and a global threshold would shave the background off
    every time. Dividing by the distance from the lifting camera gives an
    ANGULAR spacing, which is ~constant over any smooth surface at any depth.

    The cut is median + t*MAD of that ratio. Median/MAD rather than mean/std
    because the outliers being removed would otherwise inflate the very
    statistic used to detect them, and because a clean cloud should lose
    nothing -- a percentile rule always removes its quota whether or not
    there is anything wrong.
    """
    n = len(P)
    if strength <= 0 or n < k + 2:
        return np.ones(n, bool)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(P).query(P, k=k + 1, workers=-1)
    knn = d[:, 1:].mean(axis=1)                      # column 0 is the point
    r = np.linalg.norm(P - origin[None, :], axis=1)
    ratio = knn / np.maximum(r, 1e-6)
    med = float(np.median(ratio))
    mad = float(np.median(np.abs(ratio - med))) + 1e-12
    # Range chosen by measurement, not taste. On a cloud with 300 known
    # flying pixels among 8,000 clean points: t=12 removes 251 with zero
    # collateral, t=4 removes 272 for 94 (1.2%). Below t=4 the collateral
    # climbs ~10x for three more outliers, so 4 is where the slider stops.
    t = 12.0 - 8.0 * min(1.0, float(strength))
    return ratio <= med + t * mad


def _source_focal_px(src: Path, iw: int, ih: int, orbit) -> tuple:
    """Focal length of the lens that TOOK src, in source pixels.

    Returns (f_px, where_it_came_from). The measured estimate wins: it is a
    property of the image, and it survives the orbit fov being changed for
    creative reasons. The estimate is stored at whatever size it was measured
    on, so it is rescaled if the source has since been resized -- focal in
    pixels is proportional to pixel count, not an intrinsic of the lens.
    """
    st = src.parent / "state.json"
    if st.exists():
        try:
            fov = (json.loads(st.read_text(encoding="utf-8")) or {}).get("fov")
            if fov:
                f = float(fov.get("focal_px") or 0.0)
                fh = fov.get("height")
                if f > 0:
                    if fh and int(fh) > 0 and int(fh) != ih:
                        f *= ih / float(fh)
                    return f, "measured"
        except Exception:
            pass                      # a missing or broken estimate is normal
    # No measurement: hand back 0.0 so SHARP keeps its own 0.75*max(W,H)
    # fallback. Deriving one from the orbit was tried and is WRONG here -- on a
    # project with no estimate the orbit fov is itself a default, and it gave
    # 3800px where sibling projects of the same source measured 1657-1774.
    # An unmeasured guess replacing a calibrated fallback is a regression, so
    # this path deliberately changes nothing.
    return 0.0, "unmeasured - left to SHARP's own fallback"


def depth_cloud_points(src: Path, orbit: Orbit, card_height: float,
                       feet_frac: float, depth_strength: float,
                       clay_mode: str = "shaded", max_points_w: int = 640,
                       with_normals: bool = False,
                       depth_model: str = "da2",
                       depth_cutoff: float = 0.0,
                       isolate_prune: float = 0.0,
                       ground_level: float = 0.0,
                       bend: float = 0.0, bend_falloff: float = 1.0,
                       bend_coverage: float = 1.0,
                       subject: tuple = (0.0, 0.0, 0.0, 0.0),
                       crop_sphere: tuple = None,
                       with_rays: bool = False, log=None):
    """The lift itself: pixels -> displaced 3D points + colours.

    Shared by the control-video renderer and the live viewport preview, so what
    you see live is the same cloud the video is rendered from.

    Two kinds of depth engine:

      da2, da2-large, da3-mono -- Depth Anything V2 Small / Large (inverse
               depth) and Depth Anything 3 Mono (depth). RELATIVE, percentile
               normalised, so depth_strength is world units of relief across
               the whole image. Fine for an isolated subject; it squashes a
               real environment into a thin shell, because scale and shift are
               both unknown and the normalisation throws the range away.

      moge2, da3-metric -- MoGe-2 / Depth Anything 3 Metric. METRIC meters
               (DA3 via the measured lens). The subject's median depth is anchored to
               the card plane and everything else keeps its true proportion
               around it, so a character at 5 m and trees at 60 m end up
               twelve times further out instead of side by side. The far field
               is then tanh-compressed toward a dome (FAR_DOME_RADII) and sky
               is pinned to it. depth_strength is a parallax multiplier here,
               where 1.0 means true metric scale.

    Either way, every pixel is displaced ALONG ITS OWN RAY FROM CAMERA 0, so
    frame 0 reproduces the source image exactly at any strength -- motion along
    a view ray is invisible to the camera that defines it.
    """
    im = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise RuntimeError(f"cannot read {src}")
    bgr = im[:, :, :3]
    ih, iw = bgr.shape[:2]

    anchor_m = None
    if depth_model in METRIC_MODELS:
        # SHARP is METRIC: the geometry it hands back is computed FROM the
        # focal it is given, and it was being given nothing -- so it fell back
        # to its own 0.75*max(W,H) guess, 2016px on a 2688x1520 source against
        # a measured 1657. SHARP therefore unprojected with one lens while
        # every later stage used another.
        #
        # The right number is the lens that TOOK the photo, which is the
        # measured estimate -- NOT the orbit's vfov. Those two are the same
        # only until the orbit fov is overridden by hand, and then the orbit
        # says what we want to RENDER with, which has no bearing on how the
        # source image was formed. Measured first, orbit only as a fallback.
        f_px, f_src = _source_focal_px(src, iw, ih, orbit)
        if depth_model == "sharp" and log:
            log(f"depth: SHARP focal "
                + (f"{f_px:.1f}px ({f_src})" if f_px > 0
                   else f"{0.75 * max(iw, ih):.0f}px ({f_src})")
                + f" at {iw}x{ih}")
        depth, valid = (estimate_depth_sharp(bgr, f_px=f_px, log=log)
                        if depth_model == "sharp"
                        else _depth_cached(src, bgr, depth_model, log=log))
        # Fill sky with a large finite depth BEFORE resizing: area averaging
        # over a NaN/zero sky would bleed near-camera depth into the skyline
        # and tear the silhouette apart.
        far_fill = (float(np.percentile(depth[valid], 99)) * 10.0
                    if valid.any() else 1e4)
        depth = depth.copy()
        depth[~valid] = far_fill
    elif depth_model in _DA2_SIZE:
        depth, valid = _depth_cached(src, bgr, depth_model, log=log)
    elif depth_model == "da3-mono":
        # DA3 Mono is depth (larger = farther); this path wants larger = closer
        depth, valid = _depth_cached(src, bgr, depth_model, log=log)
        depth = -depth
    else:
        raise RuntimeError(f"unknown depth model {depth_model!r} - "
                           f"choose one of {', '.join(DEPTH_MODELS)}")

    # Resample the sampling grid UP as well as down. Downsampling caps cost;
    # upsampling matters because the cloud's speckle is a SAMPLING gap, not a
    # missing-information one -- one point per source pixel spread across a
    # larger canvas leaves holes between them, worse wherever the depth
    # gradient pulls neighbours apart. Interpolating depth (smooth by nature)
    # and colour onto a finer grid fills those without inventing geometry.
    # Without this, asking for more than the source width silently did nothing.
    if iw != max_points_w:
        interp = cv2.INTER_AREA if iw > max_points_w else cv2.INTER_LINEAR
        s = max_points_w / iw
        tgt = (max_points_w, max(1, int(round(ih * s))))
        bgr = cv2.resize(bgr, tgt, interpolation=interp)
        depth = cv2.resize(depth, tgt, interpolation=interp)
        if valid is not None:
            valid = cv2.resize(valid.astype(np.uint8), tgt,
                               interpolation=cv2.INTER_NEAREST).astype(bool)
    ch_, cw_ = bgr.shape[:2]

    corners, _, card_w = _card_geometry(cw_, ch_, orbit, card_height)
    origin = corners[0]
    ex = (corners[1] - corners[0]) / cw_
    ey = (corners[3] - corners[0]) / ch_
    uu, vv = np.meshgrid(np.arange(cw_) + 0.5, np.arange(ch_) + 0.5)
    P = (origin[None, None, :] + uu[..., None] * ex[None, None, :]
         + vv[..., None] * ey[None, None, :]).reshape(-1, 3)

    a0 = orbit.ray_angle()
    C0 = np.array([orbit.radius * math.cos(a0), orbit.radius * math.sin(a0),
                   orbit.cam_z])
    rays = P - C0[None, :]
    t_card = np.linalg.norm(rays, axis=1)
    rays = rays / t_card[:, None]

    if depth_model in METRIC_MODELS:
        zm = depth.reshape(-1).astype(np.float64)
        vmask = valid.reshape(-1)
        anchor_m = _anchor_depth(src, depth, valid)

        # meters -> world units, scaled so the subject's depth lands exactly on
        # the card plane; distances behind it stay in true proportion
        tt = t_card * (zm / max(anchor_m, 1e-6))

        # parallax dial about the card plane; 1.0 = true metric scale
        tt = t_card + (tt - t_card) * depth_strength

        # Soft knee behind the card: identity slope at the card plane (so near
        # geometry keeps its true relative depth) saturating at the far dome.
        far = t_card + FAR_DOME_RADII * orbit.radius
        span = far - t_card
        tt = np.where(tt > t_card,
                      t_card + span * np.tanh((tt - t_card) / span), tt)

        tt[~vmask] = far[~vmask]                # sky sits on the dome
        tt = np.maximum(tt, 0.05 * t_card)      # never behind the camera
        P = C0[None, :] + rays * tt[:, None]

        # From here dnorm only drives the clay shading. Use normalised
        # disparity so "nearer is brighter" still holds, and so the shading
        # does not wash out when a single far object stretches the range.
        disp = 1.0 / np.maximum(depth, 1e-6)
        dvals = disp[valid] if valid.any() else disp
        lo, hi = np.percentile(dvals, 2), np.percentile(dvals, 98)
        dnorm = np.clip((disp - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    else:
        lo, hi = np.percentile(depth, 2), np.percentile(depth, 98)
        dnorm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        off = (0.5 - dnorm.reshape(-1)) * 2.0 * depth_strength
        P = P + rays * off[:, None]

    if ground_level > 0:
        P = level_ground(C0, rays, P, ground_level)
    if bend:
        P = bend_scene(C0, rays, P, bend, bend_falloff, bend_coverage)

    if clay_mode == "shaded":
        # FLAT clay. "shaded" is kept as the value so saved projects still pick
        # clay, but there is no light any more: no N.L, no depth brightening.
        # The control video carries outline and motion; appearance comes from
        # the prompt and the reference image.
        cols = np.full((ch_ * cw_, 3), CLAY_GREY, np.uint8)
    else:
        cols = bgr.reshape(-1, 3)

    normals = None
    if with_normals:
        G = P.reshape(ch_, cw_, 3)
        du = np.gradient(G, axis=1)
        dv_ = np.gradient(G, axis=0)
        n = np.cross(du.reshape(-1, 3), dv_.reshape(-1, 3))
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        n = n / np.maximum(ln, 1e-9)
        flip = np.einsum("ij,ij->i", n, C0[None, :] - P) < 0
        n[flip] = -n[flip]
        normals = n

    # Signed displacement along each pixel's ray: positive = pushed behind
    # the card plane. One number per point, shared by the cutoff below and by
    # the viewport shader that rescales strength live.
    dt = np.einsum("ij,ij->i",
                   P - (C0[None, :] + rays * t_card[:, None]), rays)

    rays_k, t_card_k = rays, t_card
    if depth_cutoff and depth_cutoff > 0:
        # Drop what the estimate pushed too far back -- usually background the
        # subject does not need. A hole is more honest than a wrong wall.
        keep = dt <= float(depth_cutoff)
        P, cols, dt = P[keep], cols[keep], dt[keep]
        rays_k, t_card_k = rays[keep], t_card[keep]
        if normals is not None:
            normals = normals[keep]

    if isolate_prune and isolate_prune > 0 and len(P):
        # AFTER the cutoff: the cutoff removes what is too far, this removes
        # what is nowhere near anything. They catch different things.
        keep = prune_isolated(P, C0, float(isolate_prune))
        dropped = int((~keep).sum())
        P, cols, dt = P[keep], cols[keep], dt[keep]
        rays_k, t_card_k = rays_k[keep], t_card_k[keep]
        if normals is not None:
            normals = normals[keep]
        _PRUNE_STAT["dropped"] = dropped
        _PRUNE_STAT["kept"] = int(len(P))
    else:
        _PRUNE_STAT["dropped"] = 0
        _PRUNE_STAT["kept"] = int(len(P))

    # Isolate a volume: keep only what is inside a sphere the operator placed.
    # Applied HERE, inside the lift, rather than as a one-off edit of a loaded
    # cloud -- the cloud is rebuilt from the depth model every time anything
    # upstream changes, so an edit made to the points alone would silently
    # disappear on the next relift. As a lift parameter it survives, and the
    # control render and the training cloud get the same treatment as the
    # viewport preview.
    if crop_sphere and len(P):
        cx, cy, cz, cr = (float(v) for v in crop_sphere)
        if cr > 0:
            d2 = ((P[:, 0] - cx) ** 2 + (P[:, 1] - cy) ** 2
                  + (P[:, 2] - cz) ** 2)
            keep = d2 <= cr * cr
            n_in = int(keep.sum())
            if n_in == 0:
                # Refuse rather than hand back an empty cloud: an empty lift
                # renders a blank video and trains on nothing, and the cause
                # (a sphere parked off the subject) is invisible downstream.
                raise RuntimeError(
                    f"the isolate sphere at ({cx:.2f}, {cy:.2f}, {cz:.2f}) "
                    f"radius {cr:.2f} contains no points - move it onto the "
                    f"subject or widen it")
            if log:
                log(f"crop: isolate sphere r={cr:.2f} at "
                    f"({cx:.2f}, {cy:.2f}, {cz:.2f}) kept {n_in:,} of "
                    f"{len(P):,} points ({100.0 * n_in / len(P):.1f}%)")
            P, cols, dt = P[keep], cols[keep], dt[keep]
            rays_k, t_card_k = rays_k[keep], t_card_k[keep]
            if normals is not None:
                normals = normals[keep]
            _PRUNE_STAT["kept"] = int(len(P))

    # Manual placement: rotate about the vertical axis through the origin,
    # then translate. Note this deliberately breaks frame-0 exactness -- it
    # exists for composing the cloud against the dome, and the operator can
    # see both while doing it.
    sdx, sdy, sdz, srot = subject[:4]
    srx, sry = (subject[4], subject[5]) if len(subject) >= 6 else (0.0, 0.0)
    Rm = _subject_rotation(srx, sry, srot)
    if Rm is not None:
        P = P @ Rm.T
        if normals is not None:
            normals = normals @ Rm.T
    if sdx or sdy or sdz:
        P = P + np.array([sdx, sdy, sdz], np.float64)[None, :]

    out = (P.astype(np.float32), np.ascontiguousarray(cols), normals,
           (cw_, ch_, card_w, anchor_m))
    if with_rays:
        card_pts = (C0[None, :]
                    + rays_k * t_card_k[:, None]).astype(np.float32)
        return out + (card_pts, rays_k.astype(np.float32),
                      dt.astype(np.float32))
    return out


def cloud_packed(src: Path, orbit: Orbit, card_height: float,
                 feet_frac: float = 0.0, depth_model: str = "moge2",
                 max_points_w: int = 288,
                 isolate_prune: float = 0.0,
                 ground_level: float = 0.0,
                 crop_sphere: tuple = None, log=None) -> bytes:
    """The lift, packed for the browser viewport.

    Strength is baked at 1.0; each point ships its card-plane position, its
    ray, and its signed displacement, so the viewport rescales strength and
    applies the cutoff in a vertex shader -- the sliders are instant and MoGe
    never re-runs. Layout, little-endian:
        u32 N | f32 N*3 cardPos | f32 N*3 ray | f32 N dt | u8 N*3 rgb(BGR)
    """
    import struct
    _P, cols, _n, _extras, card_pts, rays, dt = depth_cloud_points(
        src, orbit, card_height, feet_frac, 1.0, clay_mode="off",
        max_points_w=max_points_w, with_normals=False,
        depth_model=depth_model, isolate_prune=isolate_prune,
        ground_level=ground_level, crop_sphere=crop_sphere, with_rays=True,
        log=log)
    # Trailing header: camera-0 origin and the bend's distance range, so the
    # vertex shader evaluates exactly the expression bend_scene() does.
    a0 = orbit.ray_angle()
    C0 = np.array([orbit.radius * math.cos(a0), orbit.radius * math.sin(a0),
                   orbit.cam_z], np.float32)
    P_now = card_pts + rays * dt[:, None]
    rn, rf = bend_span(C0, P_now) if len(P_now) else (0.0, 1.0)
    return (struct.pack("<I", len(card_pts)) + card_pts.tobytes()
            + rays.tobytes() + dt.tobytes()
            + np.ascontiguousarray(cols).tobytes()
            + struct.pack("<5f", float(C0[0]), float(C0[1]), float(C0[2]),
                          rn, rf))


def _splat_points(canvas: np.ndarray, u: np.ndarray, v: np.ndarray,
                  zz: np.ndarray, cc: np.ndarray,
                  cover: np.ndarray | None = None) -> None:
    """Far-to-near 2x2 splat with ONE global ordering.

    Sorting once and replaying that order in four separate full passes let a
    far point's bottom-right corner overwrite a near point's centre (the
    fourth pass ran wholly after the first) -- a one-pixel background fringe
    over every near silhouette edge. Writing each point's four offsets
    consecutively in one sorted pass makes the painter order globally true.
    The sort runs on quantised int32 depth: numpy's stable sort is radix for
    ints (~3x faster than float argsort), and a 2^31-step quantisation over
    the depth span cannot reorder anything visible.
    """
    lo = float(zz.min()) if len(zz) else 0.0
    span = float(zz.max() - lo) if len(zz) else 1.0
    q = ((zz - lo) * ((2 ** 31 - 1) / max(span, 1e-12))).astype(np.int32)
    o = np.argsort(-q, kind="stable")            # far first, near last
    u, v, cc = u[o], v[o], cc[o]
    # Corner spills first, each far-to-near; the (0, 0) centre LAST. A pixel's
    # own point must always beat a neighbour's 2x2 spill: deciding that fight
    # by z instead made the winner flip with depth_strength (z scales with it,
    # u and v do not), and frame 0 stopped being strength-invariant. Centre-
    # over-spill keeps frame 0 = the source at any strength, and still fixes
    # the old fringe, where a background corner overwrote a subject centre.
    for du, dv in ((0, 1), (1, 0), (1, 1), (0, 0)):
        canvas[v + dv, u + du] = cc
        if cover is not None:
            # Exactly the pixels a point was written to -- the authored
            # silhouette, not an estimate of it.
            cover[v + dv, u + du] = 255


_PREVIEW_CACHE: dict = {}


def render_orbit_depthcloud(src: Path, orbit: Orbit, out_dir: Path,
                            card_height: float = 3.6, feet_frac: float = 0.0,
                            depth_strength: float = 1.5,
                            backdrop: str = "scene",
                            clay_mode: str = "shaded",
                            backface_cull: bool = True,
                            max_points_w: int = 640,
                            depth_model: str = "da2",
                            depth_cutoff: float = 0.0,
                            isolate_prune: float = 0.0,
                            ground_level: float = 0.0,
                            bend: float = 0.0, bend_falloff: float = 1.0,
                            bend_coverage: float = 1.0,
                            subject_x: float = 0.0, subject_y: float = 0.0,
                            subject_z: float = 0.0,
                            subject_rot_deg: float = 0.0,
                            subject_rot_x: float = 0.0,
                            subject_rot_y: float = 0.0,
                            mp4_out: Path | None = None,
                            fps: int = 24,
                            first_frame_out: Path | None = None,
                            matte_out: Path | None = None,
                            crop_sphere: tuple = None,
                            log=None, progress=None) -> dict:
    """Control video from a DEPTH-DISPLACED point cloud instead of a flat card.

    Lift every pixel to a 3D point at its estimated depth, colour it from the
    image (or shade it as clay), and orbit that. Points rather than a displaced
    mesh, so there are no stretched triangles at depth discontinuities;
    disocclusions come out as honest backdrop-coloured holes, which the video
    model fills.

    The lift itself lives in depth_cloud_points() -- see there for the two
    depth engines and why frame 0 is exact at any strength.
    """
    P, cols, normals, (cw_, ch_, card_w, anchor_m) = depth_cloud_points(
        src, orbit, card_height, feet_frac, depth_strength,
        clay_mode=clay_mode, max_points_w=max_points_w,
        with_normals=backface_cull, depth_model=depth_model,
        depth_cutoff=depth_cutoff, isolate_prune=isolate_prune,
        ground_level=ground_level, bend=bend, bend_falloff=bend_falloff,
        bend_coverage=bend_coverage,
        subject=(subject_x, subject_y, subject_z, subject_rot_deg,
                 subject_rot_x, subject_rot_y),
        crop_sphere=crop_sphere, log=log)

    ground = ground_z_for(orbit, card_height, feet_frac)
    polys = scene_polys(orbit, ground) if backdrop == "scene" else None

    # The backdrop is flat grey, full stop. The panorama/ground-projection dome
    # was removed in this build; `backdrop` survives only so LEGACY projects
    # whose state.json still says "scene" keep rendering their marker ring.

    fx, fy, cx, cy = orbit.intrinsics()
    W, H = orbit.width, orbit.height
    out_dir.mkdir(parents=True, exist_ok=True)
    _clear_frames(out_dir)

    if matte_out is not None:
        matte_out.mkdir(parents=True, exist_ok=True)
        for stale in matte_out.glob("frame_*.png"):
            stale.unlink()

    enc = None
    if mp4_out is not None:
        enc = subprocess.Popen(
            [_ffmpeg(), "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{W}x{H}", "-framerate", str(fps), "-i", "-",
             # same encode settings as frames_to_mp4 -- see the CRF note there
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
             "-preset", "veryfast", str(mp4_out)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)

    for i, (R, t) in enumerate(orbit.poses(), start=1):
        if progress is not None:
            try:
                progress(i, orbit.frames)
            except Exception:
                pass                      # a progress report never fails a render
        canvas = np.full((H, W, 3), float(BG_GREY), np.float32)
        if polys:
            leftover = _draw_polys(canvas, polys, R, t, orbit, -1e9)
            for pts, shade in leftover:
                cv2.fillConvexPoly(canvas, pts, (BG_GREY * shade * 2.6,) * 3,
                                   cv2.LINE_AA)

        X = (R @ P.T).T + t
        z = X[:, 2]
        ok = z > 0.05
        if normals is not None:
            C = -R.T @ t                        # camera centre, world space
            view = C[None, :] - P
            facing = np.einsum("ij,ij->i", normals, view) > 0.0
            ok = ok & facing
        u = (fx * X[ok, 0] / z[ok] + cx).astype(np.int32)
        v = (fy * X[ok, 1] / z[ok] + cy).astype(np.int32)
        inb = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
        u, v = u[inb], v[inb]
        zz = z[ok][inb]
        cc = cols[ok][inb]

        cover = (np.zeros((H, W), np.uint8)
                 if matte_out is not None else None)
        _splat_points(canvas, u, v, zz, cc, cover)
        if cover is not None:
            # Close the single-pixel gaps between neighbouring splats so the
            # silhouette is solid rather than stippled; the 2x2 splat leaves
            # them wherever the projection lands off-grid.
            cover = cv2.morphologyEx(
                cover, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            cv2.imwrite(str(matte_out / f"frame_{i:04d}.png"), cover,
                        [cv2.IMWRITE_PNG_COMPRESSION, 6])
        img = np.clip(canvas, 0, 255).astype(np.uint8)
        if i == 1 and first_frame_out is not None:
            # Saved BEFORE the encoder sees it: this frame becomes the
            # appearance reference, so it must not be an x264 copy of itself.
            first_frame_out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(first_frame_out), img,
                        [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if enc is not None:
            # straight into x264: no PNG encode, no re-read pass, and the
            # video encode overlaps the render in another process
            enc.stdin.write(img.tobytes())
        else:
            cv2.imwrite(str(out_dir / f"frame_{i:04d}.png"), img,
                        [cv2.IMWRITE_PNG_COMPRESSION, 3])

    if enc is not None:
        enc.stdin.close()
        if enc.wait() != 0:
            raise RuntimeError("ffmpeg failed while encoding the control "
                               "video from the piped frames")

    return {"frames": orbit.frames, "mode": "depthcloud",
            "depth_model": depth_model,
            "depth_cutoff": float(depth_cutoff),
            "isolate_prune": float(isolate_prune),
            "ground_level": float(ground_level),
            "bend": float(bend), "bend_falloff": float(bend_falloff),
            "bend_coverage": float(bend_coverage),
            "ground_levelled_points": int(_GROUND_STAT["moved"]),
            "pruned_points": int(_PRUNE_STAT["dropped"]),
            "subject_x": float(subject_x), "subject_y": float(subject_y),
            "subject_z": float(subject_z),
            "subject_rot_deg": float(subject_rot_deg),
            "subject_rot_x": float(subject_rot_x),
            "subject_rot_y": float(subject_rot_y),
            "anchor_depth_m": round(anchor_m, 3) if anchor_m else None,
            "points": int(len(P)), "points_w": int(max_points_w),
            "depth_strength": float(depth_strength),
            "backface_cull": bool(backface_cull),
            "blank_edge_on": 0, "thin_frames": 0,
            "card_w": card_w, "card_h": card_height,
            "clay": clay_mode if clay_mode != "off" else "colour",
            "backdrop": backdrop, "ground_z": round(ground, 3),
            "floor_visible": bool(orbit.cam_z > ground + 0.05),
            "scene_polys": len(polys) if polys else 0,
            "elev_deg": orbit.elev_deg, "cam_z": round(orbit.cam_z, 4),
            "elev_sweep_deg": float(orbit.elev_sweep_deg),
            "elev_span": [round(v, 2) for v in orbit.elev_span()],
            "quad_area_min": 0.0, "quad_area_max": 0.0}


# ------------------------------------------------------------------ video ---
def _ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def frames_to_mp4(frames: Path, dst: Path, fps: int = 24) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([_ffmpeg(), "-y", "-framerate", str(fps),
                    "-i", str(frames / "frame_%04d.png"),
                    # CRF 20 / veryfast, not 16 / medium: this is a CONTROL
                    # signal, not a deliverable. Point-cloud renders are full
                    # of high-frequency speckle that x264 cannot predict, so a
                    # near-lossless CRF inflates the file (4.71 MB -> 3.58 MB
                    # here) and every megabyte is upload time before fal.
                    # yuv420p needs EVEN dimensions and x264 simply fails on
                    # odd ones -- a 1332x629 capture returned a bare non-zero
                    # exit and an empty file. Rounding down by a pixel is
                    # invisible and cannot fail.
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                    "-preset", "veryfast",
                    str(dst)], check=True, capture_output=True)
    return dst


def video_frame_count(src: Path) -> tuple:
    """(frame count, fps) without decoding the whole clip."""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {src}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if n <= 0:                      # some containers do not carry a count
        n = 0
        while cap.grab():
            n += 1
    cap.release()
    return n, fps


def soften_video(src: Path, dst: Path, factor: float = 2.0,
                 blur: float = 0.0, fps: int = 24, log=None) -> dict:
    """Throw away detail from the control video, keep its structure.

    The control render's texture is not real: it is a point cloud sampled from
    one photo, so its high-frequency detail is an artefact of the lift rather
    than anything the subject actually has. Handed that at full sharpness, a
    video model tends to preserve it. Softening first leaves the model the
    shapes and the motion while inviting it to invent plausible detail.

    The output really is smaller: W/f x H/f, not scaled back up. An earlier
    version kept the original dimensions to stay clear of fal's crop, but
    that crop keys on ASPECT, not size -- a uniform downscale preserves the
    aspect exactly, so there is nothing to protect against, and a clip that
    reports the same resolution as its source looks like the setting did
    nothing.

    Dimensions are rounded to EVEN numbers because libx264 requires it; 640/3
    is 213.3, and 213 would be rejected. The log reports the size actually
    produced rather than the one asked for.

    INTER_AREA to shrink: it averages, which is what removes detail cleanly
    rather than aliasing it into false texture.
    """
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {src}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    factor = max(1.0, float(factor))
    even = lambda v: max(8, int(round(v / 2.0)) * 2)   # libx264 wants even
    sw, sh = even(W / factor), even(H / factor)
    # odd kernel, and 0 means no blur at all rather than a 1px no-op
    k = int(round(float(blur)))
    k = 0 if k <= 0 else (k * 2 + 1)

    dst.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(
        [_ffmpeg(), "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{sw}x{sh}", "-framerate", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
         "-preset", "veryfast", str(dst)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    n = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out = cv2.resize(fr, (sw, sh), interpolation=cv2.INTER_AREA)
        # Blur at the OUTPUT size, so the slider means pixels in the file that
        # is actually sent.
        if k:
            out = cv2.GaussianBlur(out, (k, k), 0)
        enc.stdin.write(out.tobytes())
        n += 1
    cap.release()
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError("ffmpeg failed writing the softened control video")
    ar_in, ar_out = W / max(H, 1), sw / max(sh, 1)
    if log:
        log(f"soften: {W}x{H} -> {sw}x{sh} ({W / max(sw, 1):.2f}x smaller)"
            + (f", blur {k}px" if k else ", no blur")
            + f", {n} frames. Aspect {ar_in:.4f} -> {ar_out:.4f}"
            + (" (unchanged)" if abs(ar_in - ar_out) < 0.005
               else " -- CHANGED, fal will crop"))
    return {"frames": n, "width": sw, "height": sh,
            "src_w": W, "src_h": H, "factor": factor, "blur_px": k,
            "aspect_in": round(ar_in, 4), "aspect_out": round(ar_out, 4)}


def retime_to_frames(src: Path, dst: Path, n_frames: int,
                     fps: int = 24) -> dict:
    """Resample a clip to EXACTLY n_frames by uniform index selection.

    For clips this pipeline did not author the frame count of -- Wan output,
    or a video the user uploaded. It maps output frame i to input frame
    round(i * (m-1) / (n-1)), i.e. it assumes the camera sweep is linear in
    time and follows the authored path. That assumption is what makes pose i
    address image i afterwards; if the source clip does not actually follow
    the orbit, retiming makes the frame COUNT agree while the geometry still
    disagrees, and the splat will be wrong in a way nothing downstream can
    detect. Use it knowingly.

    Two passes so memory stays flat: count, then stream.
    """
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")
    m, src_fps = video_frame_count(src)
    if m < 1:
        raise RuntimeError(f"{src.name} holds no decodable frames")
    want = [round(i * (m - 1) / max(n_frames - 1, 1))
            for i in range(n_frames)]
    need = set(want)

    cap = cv2.VideoCapture(str(src))
    ok, first = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"cannot decode {src.name}")
    H, W = first.shape[:2]
    cache = {0: first} if 0 in need else {}
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        i += 1
        if i in need:
            cache[i] = fr
    cap.release()

    dst.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(
        [_ffmpeg(), "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-framerate", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
         "-preset", "veryfast", str(dst)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    last = None
    for idx in want:
        fr = cache.get(idx, last)
        if fr is None:
            fr = first
        enc.stdin.write(fr.tobytes())
        last = fr
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError("ffmpeg failed while retiming")
    return {"from_frames": m, "to_frames": n_frames,
            "from_fps": round(src_fps, 3), "to_fps": fps,
            "size": [W, H]}


class MergedOrbit:
    """Several rings presented as ONE orbit, for write_colmap.

    A pass is rendered from the trained splat, which lives in the author's
    COLMAP world, and the pass viewport uses the same camera formula as
    Orbit.poses(). So the rings are already in one coordinate system -- there
    is nothing to register, only to concatenate. Verified against the pass
    viewport: 360 camera centres agreed to 9.4e-15 world units.

    Intrinsics come from the FIRST ring, and every clip in the set must have
    been delivered at the same size for that to be legitimate. The caller
    checks it; this class would happily write a lie.

    Only what write_colmap touches is implemented, deliberately: this is a
    view over real Orbits, not a fourth camera path to keep in step.
    """

    def __init__(self, parts):
        if not parts:
            raise ValueError("a merged orbit needs at least one ring")
        self.parts = list(parts)
        b = self.parts[0]
        self.width, self.height = b.width, b.height
        self.vfov_deg, self.aim_z = b.vfov_deg, b.aim_z
        self.radius = b.radius
        self.frames = sum(p.frames for p in self.parts)

    def intrinsics(self):
        return self.parts[0].intrinsics()

    def poses(self):
        for p in self.parts:
            yield from p.poses()


def ring_orbit(base: Orbit, cam_z: float, radius: float, frames: int) -> Orbit:
    """A copy of `base` flown at one height and radius.

    Elevation is DERIVED from the height rather than typed, because Orbit is
    parameterised by angle: elev = atan2(z - aim_z, r) reproduces cam_z exactly
    through the same tan() the class uses, so the ring lands where the pass
    viewport put it instead of somewhere close to it.
    """
    r = max(1e-3, float(radius))
    elev = math.degrees(math.atan2(float(cam_z) - base.aim_z, r))
    return replace(base, frames=int(frames), radius=r, elev_deg=elev,
                   path="ring", elev_sweep_deg=0.0)


def conform_video(src: Path, dst: Path, width: int, height: int,
                  n_frames: int, fps: int, log=None) -> dict:
    """Force a clip to EXACTLY width x height x n_frames at fps.

    Written for orbit passes, which must be interchangeable with the author's
    AI video and with each other. When every clip in a dataset is the same
    size, the whole set is described by ONE COLMAP camera; let them differ and
    each needs its own intrinsics, because focal length in PIXELS depends on
    the pixel grid even when the lens is identical (measured: the same 29.8
    degree lens is 1172.6 px at 624 and 1804.0 px at 960).

    Resizing is a plain RESIZE to exactly width x height. Nothing is cropped:
    everything fal drew is still in the frame. When the aspects already match
    -- which is the whole point of the aspect guard on the send -- this is a
    uniform scale and is exact.

    When they do NOT match it is an anamorphic stretch, and the honest cost is
    this: a stretched image is still a perfectly good pinhole view, but only
    with fx and fy scaled independently, so that clip needs its own camera
    line in COLMAP. Cropping was tried first and is worse -- measured, an
    832x480 return cropped to 624x624 kept 58% of the width and cut a limb off
    the subject. Deleting picture to save a camera line is the wrong trade.
    Matching the aspect at generation time avoids both.

    Frames are resampled the way retime_to_frames does, by uniform index
    selection, and it carries retime_to_frames' warning with it: this makes
    the COUNT agree, not the geometry. A clip that does not follow the
    authored path is still wrong afterwards, just wrong at the right length.
    """
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")
    width, height = int(width) - int(width) % 2, int(height) - int(height) % 2
    m, src_fps = video_frame_count(src)
    if m < 1:
        raise RuntimeError(f"{src.name} holds no decodable frames")
    want = [round(i * (m - 1) / max(n_frames - 1, 1)) for i in range(n_frames)]
    need = set(want)

    cap = cv2.VideoCapture(str(src))
    ok, first = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"cannot decode {src.name}")
    sh, sw = first.shape[:2]

    def fit(fr):
        if (fr.shape[1], fr.shape[0]) == (width, height):
            return fr
        # INTER_AREA downsizes without aliasing; it is wrong for upscaling,
        # where CUBIC is the right one. Judged on the AVERAGE of the two axis
        # scales, since a stretch can shrink one axis while growing the other.
        sc = 0.5 * (width / fr.shape[1] + height / fr.shape[0])
        interp = cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC
        return cv2.resize(fr, (width, height), interpolation=interp)

    cache = {0: first} if 0 in need else {}
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        i += 1
        if i in need:
            cache[i] = fr
    cap.release()

    dst.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(
        [_ffmpeg(), "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{width}x{height}", "-framerate", str(int(fps)), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
         "-preset", "veryfast", str(dst)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    last = None
    for idx in want:
        fr = cache.get(idx, last)
        if fr is None:
            fr = first
        fr = fit(fr)
        enc.stdin.write(fr.tobytes())
        last = cache.get(idx, last)
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError("ffmpeg failed while conforming")
    # Nothing is discarded, so the only thing worth reporting is whether the
    # pixels were stretched -- which is what decides if this clip can share a
    # COLMAP camera with the others.
    src_ar, dst_ar = sw / max(sh, 1), width / max(height, 1)
    stretched = abs(src_ar - dst_ar) > 0.002 * dst_ar
    info = {"from_size": [sw, sh], "to_size": [width, height],
            "from_frames": m, "to_frames": n_frames,
            "from_fps": round(src_fps, 3), "to_fps": int(fps),
            "resized": (sw, sh) != (width, height),
            "resampled": m != n_frames,
            "from_aspect": round(src_ar, 4), "to_aspect": round(dst_ar, 4),
            "stretched": bool(stretched)}
    if log:
        log(f"conform: {sw}x{sh} {m}f @{src_fps:g} -> "
            f"{width}x{height} {n_frames}f @{fps}"
            + ("" if info["resized"] or info["resampled"]
               else "  (already matched, re-encoded only)"))
        if stretched:
            log(f"conform: WARNING aspect {src_ar:.3f} -> {dst_ar:.3f}. "
                f"Nothing is cropped - the whole frame is kept - but the "
                f"pixels are STRETCHED, so this clip is only geometrically "
                f"correct with its own fx/fy and cannot share one COLMAP "
                f"camera with the author clip. Set the engine's aspect to "
                f"match the author clip and regenerate to avoid it.")
    return info


# --------------------------------------------------------------- staleness --
# Derived directories were reused whenever they were merely non-empty. That is
# a guess: ai_frames/ from a previous AI video, or a matte cut against it,
# looks exactly as valid as one cut against the current clip. So every derived
# directory records what produced it, and is rebuilt when that no longer
# matches. Cheap to write, and it removes a whole class of "why is it showing
# the old one".

def _fingerprint(p: Path) -> dict:
    """Identity of a source file: name, size and mtime. Rewriting a file with
    the same name changes at least one of the last two."""
    q = Path(p)
    if not q.exists():
        return {"name": q.name, "missing": True}
    st = q.stat()
    return {"name": q.name, "size": st.st_size, "mtime": int(st.st_mtime)}


def stamp_write(out: Path, **facts) -> None:
    """Record what a derived directory was built from."""
    out.mkdir(parents=True, exist_ok=True)
    (out / ".built_from.json").write_text(
        json.dumps(facts, sort_keys=True, indent=1), encoding="utf-8")


def stamp_ok(out: Path, **facts) -> bool:
    """True when out/ exists, holds frames, and was built from exactly `facts`."""
    q = out / ".built_from.json"
    if not (out.is_dir() and any(out.glob("frame_*.png")) and q.exists()):
        return False
    try:
        have = json.loads(q.read_text(encoding="utf-8"))
    except Exception:
        return False
    return have == json.loads(json.dumps(facts, sort_keys=True))


def stamp_write_file(out: Path, **facts) -> None:
    """Record what a derived FILE was built from, in a sidecar."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(out.suffix + ".built_from.json").write_text(
        json.dumps(facts, sort_keys=True, indent=1), encoding="utf-8")


def stamp_ok_file(out: Path, **facts) -> bool:
    """True when the file exists and was built from exactly `facts`."""
    q = out.with_suffix(out.suffix + ".built_from.json")
    if not (out.is_file() and q.exists()):
        return False
    try:
        return json.loads(q.read_text(encoding="utf-8")) ==             json.loads(json.dumps(facts, sort_keys=True))
    except Exception:
        return False


def mp4_to_frames(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    for f in dst.glob("frame_*.png"):
        f.unlink()
    subprocess.run([_ffmpeg(), "-y", "-i", str(src), "-fps_mode", "passthrough",
                    "-start_number", "1", str(dst / "frame_%04d.png")],
                   check=True, capture_output=True)
    return len(list(dst.glob("frame_*.png")))


# ------------------------------------------------------------------ matte ---
def _subject_rotation(rx_deg: float, ry_deg: float, rz_deg: float):
    """Rotation applied to the subject cloud, or None when it is identity.

    Order is X, then Y, then Z (R = Rz @ Ry @ Rx), about the cloud's own
    origin and before translation. The viewport applies the same order, so
    the preview and the render agree once more than one angle is non-zero --
    with any other pairing they would diverge silently.
    """
    if not (rx_deg or ry_deg or rz_deg):
        return None
    ax, ay, az = (math.radians(rx_deg), math.radians(ry_deg),
                  math.radians(rz_deg))
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], np.float64)
    return Rz @ Ry @ Rx


def _fit_cover_crop(mask: np.ndarray, shape) -> np.ndarray:
    """Map a control-canvas mask into AI-frame space the way fal transformed
    the pixels: scale to COVER, then centre-crop.

    A plain resize stretches instead, which is the same mistake the COLMAP
    intrinsics made. On an 800x992 mask going to 448x576 the two differ by
    3.7% horizontally -- about 17px across the frame, so the mask edge lands
    up to 8px off the real silhouette on each side. Every one of those pixels
    tells the trainer the wrong thing about transparency.
    """
    H, W = shape
    mh, mw = mask.shape[:2]
    s = max(W / mw, H / mh)
    rw, rh = int(round(mw * s)), int(round(mh * s))
    r = cv2.resize(mask, (rw, rh), interpolation=cv2.INTER_NEAREST)
    x0, y0 = (rw - W) // 2, (rh - H) // 2
    return r[y0:y0 + H, x0:x0 + W]


def matte(frames: Path, rmbg_dir: Path | None, out: Path,
          lo: float = 10.0, hi: float = 38.0, unmix_min: float = 0.25,
          use_distance: bool = False) -> dict:
    """alpha from distance-to-backdrop, bars stripped, per-frame backdrop.

    rmbg_dir holds the AUTHORED coverage written by the control render -- the
    exact pixels a point was splatted to, one 8-bit mask per frame. It is not
    a segmentation guess: the renderer knows the silhouette because it drew it.

    Distance alone is not enough. Measured over 8 frames of a dark subject on
    grey, by fraction of pixels with alpha strictly between 0 and 1:
        distance only          16.45%
        RMBG + distance         4.80%
        authored + distance     2.59%
    --match-alpha-weight supervises against that alpha, so the fuzz is learned
    as semi-transparent hair around every edge.

    Note the advantage is edge SHARPNESS, not temporal stability: measured
    frame-to-frame silhouette change was comparable to RMBG's (6.8% vs 6.1%
    mean, 8.3% vs 10.6% worst), and at these rotation rates that metric is
    dominated by real motion rather than jitter.

    Legacy RGBA cutouts are still accepted; None falls back to distance only.

    Unmix only above unmix_min -- dividing by a small alpha amplifies noise
    ~20x and blows out translucency.
    """
    out.mkdir(parents=True, exist_ok=True)
    # Every other frame producer clears first. Without this, re-running at a
    # lower frame count leaves the tail of the previous run behind and the
    # dataset ends up with more images than the rig has poses.
    for stale in out.glob("frame_*.png"):
        stale.unlink()
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    files = sorted(frames.glob("frame_*.png"))
    cov, par, bars_px = [], [], []
    missing = []

    for f in files:
        im = cv2.imread(str(f)).astype(np.float32)
        H, W = im.shape[:2]
        g = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY)

        dark = g < 24
        col, row = dark.mean(axis=0), dark.mean(axis=1)
        L = int(np.argmax(col < 0.97)) if (col >= 0.97).any() else 0
        R = int(np.argmax(col[::-1] < 0.97)) if (col >= 0.97).any() else 0
        T = int(np.argmax(row < 0.97)) if (row >= 0.97).any() else 0
        B = int(np.argmax(row[::-1] < 0.97)) if (row >= 0.97).any() else 0
        L, R, T, B = (min(v + 2, s // 4) if v else 0
                      for v, s in ((L, W), (R, W), (T, H), (B, H)))

        inner = im[T:H - B or H, L:W - R or W]
        q = [inner[:20, :20], inner[:20, -20:], inner[-20:, :20], inner[-20:, -20:]]
        bg = np.median(np.concatenate([p.reshape(-1, 3) for p in q]), axis=0)

        a_dist = np.clip((np.linalg.norm(im - bg, axis=2) - lo) / (hi - lo), 0, 1)
        rf = (rmbg_dir / f.name) if rmbg_dir is not None else None
        a_r = np.zeros_like(a_dist)
        if rf is not None and not rf.exists():
            missing.append(f.name)
        if rf is not None and rf.exists():
            rr = cv2.imread(str(rf), cv2.IMREAD_UNCHANGED)
            if rr is not None:
                # Single channel = authored coverage from the control render.
                # Four channels = a legacy RGBA cutout. Anything else is not
                # a mask and is ignored rather than indexed blindly.
                ch = rr[:, :, 3] if rr.ndim == 3 and rr.shape[2] == 4 else (
                    rr if rr.ndim == 2 else None)
                if ch is not None:
                    if ch.shape != a_dist.shape:
                        ch = _fit_cover_crop(ch, a_dist.shape)
                    a_r = ch.astype(np.float32) / 255.0
        # The distance term was written for the CONTROL render, where the
        # backdrop really is one flat colour the renderer chose. On a fal
        # clip that assumption is false -- Wan invents a studio floor a
        # couple of dozen levels lighter than the wall, and distance calls
        # the whole bottom half of the frame subject. Measured on
        # project1-10: BiRefNet 30.5% coverage, unioned 57.8%, and the
        # difference was rows 349-699 of a 700-tall frame. So when a mask
        # source is chosen, that source IS the matte.
        alpha = a_dist if rmbg_dir is None else (
            np.maximum(a_r, a_dist) if use_distance else a_r)

        if L:
            alpha[:, :L] = 0
        if R:
            alpha[:, W - R:] = 0
        if T:
            alpha[:T, :] = 0
        if B:
            alpha[H - B:, :] = 0

        solid = (alpha > 0.5).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(solid, 8)
        if n > 2:
            keep = np.zeros((n,), bool)
            for i in range(1, n):
                keep[i] = st[i, cv2.CC_STAT_AREA] >= 0.001 * H * W
            alpha = alpha * keep[lab]
        alpha = cv2.erode(alpha, ke, iterations=1)

        a3 = alpha[..., None]
        um = (im - (1 - a3) * bg) / np.maximum(a3, 1e-3)
        fg = np.clip(np.where(a3 >= unmix_min, um, im), 0, 255)

        cv2.imwrite(str(out / f.name),
                    np.dstack([fg.astype(np.uint8), (alpha * 255).astype(np.uint8)]),
                    [cv2.IMWRITE_PNG_COMPRESSION, 6])
        cov.append(float((alpha > 0.5).mean()))
        par.append(float(((alpha > .02) & (alpha < .98)).mean()))
        bars_px.append(L + R + T + B)

    if missing and not use_distance:
        # Silently mattes to nothing otherwise: no mask file means alpha 0
        # for the whole frame, and the trainer would be told to delete the
        # subject there. Loud is the only safe behaviour.
        raise RuntimeError(
            f"{len(missing)} of {len(files)} frames have no mask in "
            f"{rmbg_dir.name}/ (first: {missing[0]}). Re-run the matte model "
            f"over the AI frames - they are out of step.")
    return {"frames": len(files), "coverage": float(np.mean(cov)),
            "coverage_sd": float(np.std(cov)), "partial_alpha": float(np.mean(par)),
            "bar_px_max": int(max(bars_px)) if bars_px else 0}


# ----------------------------------------------------------------- COLMAP ---
def visual_hull(mattes: Path, orbit: Orbit, n_points: int = 120000,
                grid: int = 96) -> np.ndarray:
    """Carve an init cloud from the silhouettes using the known poses.

    Far better than a random blob: gaussians start on the actual occupied volume.
    Doubles as a pose check -- wrong angles carve the subject away, so a tiny
    survivor count means something upstream is wrong.
    """
    files = sorted(mattes.glob("frame_*.png"))
    if not files:
        raise RuntimeError("no mattes to carve from")
    a0 = cv2.imread(str(files[0]), cv2.IMREAD_UNCHANGED)
    H, W = a0.shape[:2]
    sx = W / orbit.width
    sy = H / orbit.height
    fx, fy, cx, cy = orbit.intrinsics()
    fx, cx = fx * sx, cx * sx
    fy, cy = fy * sy, cy * sy

    r = orbit.radius * 0.34
    zs = np.linspace(orbit.aim_z - r * 1.5, orbit.aim_z + r * 1.5, grid)
    xs = np.linspace(-r, r, grid)
    ys = np.linspace(-r, r, grid)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    keep = np.ones(len(P), bool)

    poses = orbit.poses()
    step = max(1, len(files) // 40)          # 40 views is plenty to carve
    for i in range(0, len(files), step):
        al = cv2.imread(str(files[i]), cv2.IMREAD_UNCHANGED)[:, :, 3] > 96
        R, t = poses[i]
        uv, z = project(R, t, P[keep], fx, fy, cx, cy)
        u = np.round(uv[:, 0]).astype(int)
        v = np.round(uv[:, 1]).astype(int)
        ok = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        inside = np.zeros(ok.shape, bool)
        inside[ok] = al[v[ok], u[ok]]
        idx = np.nonzero(keep)[0]
        keep[idx[~inside]] = False
        if keep.sum() < 500:
            break

    pts = P[keep]
    if len(pts) == 0:
        raise RuntimeError("visual hull carved everything away — poses suspect")
    rng = np.random.default_rng(0)
    if len(pts) < n_points:
        cell = (xs[1] - xs[0])
        reps = int(np.ceil(n_points / len(pts)))
        pts = np.repeat(pts, reps, axis=0)[:n_points]
        pts = pts + rng.uniform(-cell / 2, cell / 2, pts.shape)
    else:
        pts = pts[rng.choice(len(pts), n_points, replace=False)]
    return pts


def write_colmap(orbit: Orbit, out: Path, names, img_w: int, img_h: int,
                 points: np.ndarray):
    """COLMAP text model. Intrinsics rescaled to the delivered video size.

    Axes are already COLMAP convention (+Z forward, +Y down) because Orbit.poses
    builds them that way, so there is no Blender-style flip to get wrong here --
    and no scaled object matrix to decompose, which is what silently broke V3.
    """
    out.mkdir(parents=True, exist_ok=True)
    fx, fy, cx, cy = orbit.intrinsics()

    # The delivered video is a UNIFORM SCALE + CENTRE CROP of the authored
    # canvas, not an anamorphic resize. Measured by correlating control frame 0
    # against AI frame 0 under each hypothesis (edge NCC):
    #     scale + centre-crop   0.93 / 0.99
    #     anamorphic stretch    0.64 / 0.32
    #     letterbox pad         0.38 / 0.16
    # Scaling fx and fy by img_w/canvas_w and img_h/canvas_h independently --
    # which is what this did -- models the stretch. Whenever the delivered
    # aspect differs from the authored one that put a systematic error into
    # the focal length (7.1% on a 3:4 canvas returned at 0.70), so every ray
    # was mis-angled and no amount of training could reconcile the views. It
    # showed up as smearing and holes.
    #
    # A cover scale with a centred crop is the correct model: one scale for
    # both axes, and the principal point shifted by the crop offset.
    s = max(img_w / orbit.width, img_h / orbit.height)
    x0 = (orbit.width * s - img_w) / 2.0
    y0 = (orbit.height * s - img_h) / 2.0
    fx, fy = fx * s, fy * s
    cx, cy = cx * s - x0, cy * s - y0

    with open(out / "cameras.txt", "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n# Number of cameras: 1\n")
        f.write(f"1 PINHOLE {img_w} {img_h} {fx:.10f} {fy:.10f} {cx:.10f} {cy:.10f}\n")

    _poses = list(orbit.poses())
    with open(out / "images.txt", "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(names)}, mean observations per image: 0\n")
        # zip() would silently truncate to the shorter side while the
        # header above still claims len(names) -- a mislabelled dataset that
        # trains for hours before anyone notices.
        if len(_poses) != len(names):
            raise RuntimeError(
                f"{len(names)} images but {len(_poses)} poses - refusing to "
                "write a COLMAP set whose images and poses disagree")
        for i, ((R, t), nm) in enumerate(zip(_poses, names), start=1):
            q = _mat2quat(R)
            f.write(f"{i} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f} {q[3]:.10f} "
                    f"{t[0]:.10f} {t[1]:.10f} {t[2]:.10f} 1 {nm}\n\n")

    with open(out / "points3D.txt", "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f.write(f"# Number of points: {len(points)}, mean track length: 0\n")
        for k, p in enumerate(points, start=1):
            f.write(f"{k} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} 128 128 128 0\n")
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "width": img_w, "height": img_h, "points": int(len(points))}


def _mat2quat(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return (w, x, y, z)


def verify(orbit: Orbit, points: np.ndarray, img_w: int, img_h: int) -> dict:
    """Depth-positive and in-frame checks. Catches an inverted or flipped rig.

    Internal consistency only -- a self-consistent but wrong rig still passes.
    """
    fx, fy, cx, cy = orbit.intrinsics()
    kx, ky = img_w / orbit.width, img_h / orbit.height
    fx, cx, fy, cy = fx * kx, cx * kx, fy * ky, cy * ky
    rng = np.random.default_rng(0)
    S = points[rng.choice(len(points), min(3000, len(points)), replace=False)]
    pos, ins = [], []
    for R, t in orbit.poses():
        uv, z = project(R, t, S, fx, fy, cx, cy)
        pos.append(float((z > 0).mean()))
        ins.append(float(((uv[:, 0] >= 0) & (uv[:, 0] < img_w) &
                          (uv[:, 1] >= 0) & (uv[:, 1] < img_h) & (z > 0)).mean()))
    return {"depth_positive_min": min(pos), "in_frame_min": min(ins),
            "in_frame_mean": float(np.mean(ins)),
            "pass": bool(min(pos) > 0.999 and min(ins) > 0.80)}


# ------------------------------------------------------------------ brush ---
# How hard to densify. Every preset is a point on one axis: how many gaussians
# the optimiser is allowed to split the subject into.
#
#   broad    -- Brush's own growth defaults, with a splat budget sized for a
#               single subject. Fewer, larger, more opaque gaussians. Solid.
#   balanced -- roughly half way; keeps some fine detail without needling.
#   fine     -- the old aggressive setup. Most detail, and the one that goes
#               thin and see-through by 10k steps.
# "stock" is handled before this table is consulted -- it means "pass Brush
# nothing", which no set of values here can express.
SPLAT_DETAIL = {
    "broad":    {"grad": "4e-5",   "frac": "0.1", "refine": "100",
                 "max": "350000",  "opac": "1e-9", "stop": 0.6},
    "balanced": {"grad": "1.5e-5", "frac": "0.25", "refine": "100",
                 "max": "700000",  "opac": "1e-9", "stop": 0.6},
    # quality: moderate growth so gaussians are not shredded, a budget high
    # enough that it should never bind, and a longer growth window followed by
    # a long refinement tail. Meant for 40-60k steps, where the learning-rate
    # schedule (which decays over total-steps) has room to settle shapes
    # instead of stopping mid-thin.
    "quality":  {"grad": "2e-5",   "frac": "0.15", "refine": "100",
                 "max": "1200000", "opac": "1e-9", "stop": 0.75},
    "fine":     {"grad": "3e-6",   "frac": "0.6", "refine": "120",
                 "max": "1500000", "opac": "1e-9", "stop": 0.6},
}


RERUN_PORT = 9876          # Rerun's default gRPC port, which Brush dials


def rerun_viewer_exe() -> Path | None:
    """The Rerun viewer shipped with rerun-sdk, if it is installed."""
    exe = Path(sys.executable).parent / (
        "rerun.exe" if os.name == "nt" else "rerun")
    return exe if exe.exists() else None


def rerun_listening(port: int = RERUN_PORT) -> bool:
    """Is a viewer already accepting connections on the gRPC port?"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sk:
        sk.settimeout(0.35)
        return sk.connect_ex(("127.0.0.1", port)) == 0


def brush_cmd(brush_exe: Path, dataset: Path, out: Path, steps: int = 50000,
              max_res: int = 960, alpha_supervision: bool = True,
              detail: str = "balanced", sh_degree: int = 0,
              rerun: bool = False, rerun_stats_every: int = 50,
              rerun_splats_every: int = 0, rerun_max_img: int = 512):
    # STOCK: hand Brush the dataset and nothing else. Every tuned setting
    # below was added to fix a specific artefact, and measured against the
    # artefact it targeted -- but a side-by-side against an untuned run showed
    # the accumulated result was worse overall: soft, ghosted, flat. Chasing
    # low anisotropy in particular is satisfied perfectly by fat blurry blobs,
    # which is what it produced. Defaults are the honest baseline to compare
    # against, so they are reachable in one click.
    #
    # How far the tuning had drifted, for the record:
    #   max-splats            10,000,000  ->    700,000   (14x lower)
    #   lr-scale-end                6e-3  ->       5e-4   (12x lower)
    #   lr-scale                    1e-2  ->       2e-3    (5x lower)
    #   match-alpha-weight           0.1  ->        0.4    (4x higher)
    #   sh-degree                      3  ->          0   (no view-dependent
    #                                                      colour at all)
    if detail == "stock":
        return [str(brush_exe),
                "--total-steps", str(steps),
                "--max-resolution", str(max_res),
                "--eval-split-every", "6",
                "--eval-every", "2500",
                "--eval-save-to-disk",
                "--export-every", str(max(steps // 4, 1)),
                "--export-path", str(out)] + (
               ["--rerun-enabled",
                "--rerun-log-train-stats-every", str(max(1, rerun_stats_every)),
                "--rerun-max-img-size", str(max(64, rerun_max_img))]
               + (["--rerun-log-splats-every", str(int(rerun_splats_every))]
                  if rerun_splats_every and rerun_splats_every > 0 else [])
               if rerun else []) + [str(dataset)]

    d = SPLAT_DETAIL.get(detail, SPLAT_DETAIL["balanced"])
    cmd = [str(brush_exe),
           "--total-steps", str(steps),
           "--max-splats", d["max"],
           "--max-resolution", str(max_res),
           # SH degree 0 by default. On a constant-elevation ring the view
           # directions form a 1-D circle, so 27 of the 48 SH parameters per
           # splat are unidentifiable: they drift freely during training and
           # then evaluate to garbage at any elevation the cameras never saw.
           # That is the coloured-blob artefact. Degree 0 has no null space.
           "--sh-degree", str(max(0, min(3, int(sh_degree)))),
           # Aux losses decay to zero at this fraction of training. The 0.9
           # default leaves the final 10% completely unregularised, which is
           # exactly when gaussians were seen to needle hardest.
           "--aux-loss-time", "1.0",
           # THE anti-needle setting, measured rather than assumed. Ablation
           # at 20k steps, identical seed, this flag alone:
           #   lr-scale 1e-2 (Brush default)  median aspect 162:1, 41.8% of
           #                                  splats degenerate, opacity 0.030
           #   lr-scale 2e-3                  median aspect   4.8:1, 9.3%
           #                                  degenerate, opacity 0.064
           # A 34x reduction in anisotropy and double the opacity -- so it
           # fixes the see-through surfaces too, since those were slivers too
           # thin to occlude. It also trains 27% FASTER, because degenerate
           # splats are expensive to sort for nothing. Upstream halves this to
           # 5e-3 for the same reason; on a single-elevation ring, where
           # nothing constrains vertical extent, it has to go further.
           "--lr-scale", "2e-3",
           "--lr-scale-end", "5e-4",
           "--growth-grad-threshold", d["grad"],
           "--growth-select-fraction", d["frac"],
           "--refine-every", d["refine"],
           "--growth-stop-iter", str(int(steps * d.get("stop", 0.6)))]
    # Only meaningful when the frames carry a matte. With the background kept
    # every pixel is opaque, so this would supervise against a constant and do
    # nothing but cost a loss term.
    if alpha_supervision:
        cmd += ["--match-alpha-weight", "0.4"]
    if rerun:
        cmd += ["--rerun-enabled",
                "--rerun-log-train-stats-every", str(max(1, rerun_stats_every)),
                "--rerun-max-img-size", str(max(64, rerun_max_img))]
        # Brush's own help flags this as heavy: it ships every gaussian over
        # gRPC at each interval, so it stays opt-in and separate from stats.
        if rerun_splats_every and rerun_splats_every > 0:
            cmd += ["--rerun-log-splats-every", str(int(rerun_splats_every))]
    return cmd + [
            "--scale-loss-weight", "1e-7",
            "--opac-loss-weight", d["opac"],
            "--ssim-weight", "0.35",
            "--eval-split-every", "6",
            "--eval-every", "2500",
            "--eval-save-to-disk",
            "--export-every", str(max(steps // 4, 1)),
            "--export-path", str(out),
            str(dataset)]


# --------------------------------------------------------- splat preview ---
SH_C0 = 0.28209479177387814


def read_ply_splats(ply: Path):
    """Parse a Brush/INRIA gaussian PLY into plain arrays.

    Conventions confirmed against a known-good export: opacity is stored as a
    LOGIT and scales as LOGS, so both need activating. Colour is spherical
    harmonic DC only -- view-dependent terms are dropped, which is right for a
    preview and costs little at SH degree 2 on a single camera ring.
    """
    raw = ply.read_bytes()
    end = raw.find(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode("ascii", "replace")
    if "binary_little_endian" not in header:
        raise RuntimeError("only binary_little_endian PLY is supported")

    count, props = 0, []
    sizes = {"float": 4, "double": 8, "uchar": 1, "char": 1, "ushort": 2,
             "short": 2, "uint": 4, "int": 4}
    codes = {"float": "f4", "double": "f8", "uchar": "u1", "char": "i1",
             "ushort": "u2", "short": "i2", "uint": "u4", "int": "i4"}
    for line in header.splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] == "element" and t[1] == "vertex":
            count = int(t[2])
        elif t[0] == "property" and len(t) == 3:
            props.append((t[1], t[2]))

    dt = np.dtype([(n, "<" + codes[ty]) for ty, n in props])
    stride = sum(sizes[ty] for ty, _ in props)
    arr = np.frombuffer(raw[end:end + count * stride], dtype=dt, count=count)

    xyz = np.stack([arr["x"], arr["y"], arr["z"]], 1).astype(np.float32)
    dc = np.stack([arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"]], 1).astype(np.float32)
    rgb = np.clip(0.5 + SH_C0 * dc, 0.0, 1.0)
    op = 1.0 / (1.0 + np.exp(-arr["opacity"].astype(np.float32)))
    scale = np.exp(np.stack([arr["scale_0"], arr["scale_1"], arr["scale_2"]],
                            1).astype(np.float32))
    rot = np.stack([arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"]],
                   1).astype(np.float32)
    n = np.linalg.norm(rot, axis=1, keepdims=True)
    rot = rot / np.maximum(n, 1e-8)
    return xyz, rgb, op, scale, rot


def splat_stats(ply: Path) -> dict:
    """Measure an exported splat: the numbers that separate failure modes.

    Reads the binary PLY directly rather than going through a viewer, so the
    result describes what was trained rather than what a renderer chose to
    show. Everything here is cheap: one pass over the arrays.
    """
    raw = ply.read_bytes()
    end = raw.find(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode("latin-1")
    if "binary_little_endian" not in header:
        return {"error": "not a binary_little_endian PLY"}
    sizes = {"float": 4, "double": 8, "uchar": 1, "char": 1, "ushort": 2,
             "short": 2, "uint": 4, "int": 4}
    np_of = {"float": "<f4", "double": "<f8", "uchar": "u1", "char": "i1",
             "ushort": "<u2", "short": "<i2", "uint": "<u4", "int": "<i4"}
    count, props = 0, []
    for line in header.splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "element" and p[1] == "vertex":
            count = int(p[2])
        elif p[0] == "property" and p[1] in sizes:
            props.append((p[2], p[1]))
    names = {n for n, _ in props}
    if not {"x", "y", "z"} <= names or count == 0:
        return {"error": "not a 3DGS export"}
    dt = np.dtype([(n, np_of[t]) for n, t in props])
    a = np.frombuffer(raw, dt, count=count, offset=end)
    out = {"splats": int(count), "mb": round(len(raw) / 1e6, 2)}

    xyz = np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float32)
    finite = np.isfinite(xyz).all(axis=1)
    out["non_finite"] = int((~finite).sum())
    xyz = xyz[finite]
    if not len(xyz):
        return {**out, "error": "no finite positions"}

    # Robust centre/extent: a handful of floaters would otherwise define the
    # bounding box and make every other number meaningless.
    med = np.median(xyz, axis=0)
    d = np.linalg.norm(xyz - med, axis=1)
    p50, p99 = float(np.percentile(d, 50)), float(np.percentile(d, 99))
    out["radius_p50"] = round(p50, 3)
    out["radius_p99"] = round(p99, 3)
    out["floaters_beyond_5x_p99"] = int((d > 5 * max(p99, 1e-6)).sum())

    if {"scale_0", "scale_1", "scale_2"} <= names:
        sc = np.exp(np.stack([a["scale_0"], a["scale_1"], a["scale_2"]],
                             1).astype(np.float32)[finite])
        hi = sc.max(axis=1)
        lo = np.maximum(sc.min(axis=1), 1e-12)
        agg = hi / lo
        out["aniso_median"] = round(float(np.median(agg)), 2)
        out["aniso_p90"] = round(float(np.percentile(agg, 90)), 2)
        out["degenerate_pct"] = round(100.0 * float((agg > 50).mean()), 2)
        out["scale_median"] = round(float(np.median(hi)), 5)
    if "opacity" in names:
        op = 1.0 / (1.0 + np.exp(-a["opacity"].astype(np.float32)[finite]))
        out["opacity_mean"] = round(float(op.mean()), 4)
        out["opacity_median"] = round(float(np.median(op)), 4)
        out["below_0_1_pct"] = round(100.0 * float((op < 0.1).mean()), 2)

    # Occupancy: voxelise the centres and ask how solid the occupied region
    # is. Gaps and thin coverage are invisible in every other number here --
    # a run can have plenty of splats, healthy opacity and still be full of
    # holes if they are not spread over the surface.
    core = xyz[d <= p99]
    if len(core) > 1000:
        vox = max(float(p50) / 24.0, 1e-4)
        idx = np.floor((core - core.min(axis=0)) / vox).astype(np.int64)
        dims = idx.max(axis=0) + 1
        if dims.prod() < 40_000_000:
            flat = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]
            occ = np.unique(flat)
            out["voxel_size"] = round(vox, 4)
            out["occupied_voxels"] = int(len(occ))
            out["splats_per_occupied_voxel"] = round(len(core) / len(occ), 2)
            # Of the voxels adjacent to an occupied one, how many are also
            # occupied? A solid shell scores high; a holey one does not.
            oc = np.zeros(int(dims.prod()), bool)
            oc[occ] = True
            oc3 = oc.reshape(tuple(int(x) for x in dims))
            nb = np.zeros_like(oc3, np.uint8)
            for ax in (0, 1, 2):
                nb[:-1] += oc3[1:].astype(np.uint8) if ax == 0 else 0
                nb[1:] += oc3[:-1].astype(np.uint8) if ax == 0 else 0
            # cheap 6-neighbour count along each axis
            nb = np.zeros_like(oc3, np.uint8)
            nb[:-1] |= oc3[1:]
            nb[1:] |= oc3[:-1]
            nb[:, :-1] |= oc3[:, 1:]
            nb[:, 1:] |= oc3[:, :-1]
            nb[:, :, :-1] |= oc3[:, :, 1:]
            nb[:, :, 1:] |= oc3[:, :, :-1]
            shell = int((nb & ~oc3).sum())
            out["surface_gap_voxels"] = shell
            out["gap_ratio"] = round(shell / max(len(occ), 1), 3)
    return out


def write_splat_report(project_dir: Path, run: dict) -> Path:
    """Append one run to the project's report, JSON and Markdown.

    History rather than last-run-only: the question is almost always "what
    changed since the run that looked right", and that needs both.
    """
    import datetime
    import json as _json
    rp = project_dir / "splat_report.json"
    hist = []
    if rp.exists():
        try:
            hist = _json.loads(rp.read_text(encoding="utf-8-sig"))
            if not isinstance(hist, list):
                hist = []
        except Exception:
            hist = []          # a corrupt report must not lose the new run
    hist.append(run)
    hist = hist[-40:]
    rp.write_text(_json.dumps(hist, indent=1), encoding="utf-8")

    def fmt(v):
        return "-" if v is None else (f"{v:,}" if isinstance(v, int) else str(v))

    L = [f"# Splat training report - {project_dir.name}", ""]
    for r in reversed(hist):
        L.append(f"## {r.get('when','?')}  -  {r.get('outcome','?')}")
        st = r.get("settings", {})
        L.append(f"- steps **{fmt(st.get('steps'))}** | detail "
                 f"`{st.get('detail')}` | SH `{st.get('sh_degree')}` | "
                 f"alpha supervision `{st.get('alpha_supervision')}`")
        ds = r.get("dataset", {})
        L.append(f"- dataset {fmt(ds.get('frames'))} frames at "
                 f"{ds.get('width')}x{ds.get('height')}, init "
                 f"{fmt(ds.get('points'))} points")
        mt = r.get("matte", {})
        if mt:
            L.append(f"- matte `{mt.get('source')}` - coverage "
                     f"{mt.get('coverage')}, ambiguous "
                     f"{mt.get('partial_alpha')}")
        cks = r.get("checkpoints", [])
        if cks:
            L.append("")
            L.append("| step | splats | MB | aniso med | aniso p90 | "
                     "degen % | opacity mean | <0.1 % | gap ratio |")
            L.append("|---|---|---|---|---|---|---|---|---|")
            for c in cks:
                L.append(f"| {fmt(c.get('step'))} | {fmt(c.get('splats'))} | "
                         f"{c.get('mb')} | {c.get('aniso_median')} | "
                         f"{c.get('aniso_p90')} | {c.get('degenerate_pct')} | "
                         f"{c.get('opacity_mean')} | {c.get('below_0_1_pct')} "
                         f"| {c.get('gap_ratio')} |")
        if r.get("brush_log_tail"):
            L.append("")
            L.append("<details><summary>Brush log tail</summary>")
            L.append("")
            L.append("```")
            L.append(r["brush_log_tail"][-1200:])
            L.append("```")
            L.append("</details>")
        if r.get("notes"):
            L.append("")
            for n in r["notes"]:
                L.append(f"> {n}")
        L.append("")
    (project_dir / "splat_report.md").write_text("\n".join(L),
                                                 encoding="utf-8")
    return rp


def clean_splat_ply(src: Path, dst: Path, min_opacity: float = 0.06,
                    max_aniso: float = 12.0, max_extent: float = 0.0,
                    centre: tuple = (0.0, 0.0, 0.0),
                    scale_gain: float = 1.0,
                    opacity_gain: float = 1.0) -> dict:
    """Post-training cleanup: prune dead splats, fatten needles, cut floaters.

    Brush has no anisotropy constraint, and over a long run the optimiser
    collapses one axis of each gaussian toward zero -- measured on a 50k run,
    the median splat ended 526x longer than it was thin, 57% at effectively
    zero volume, and 65% below 0.1 opacity. That is what reads as needles and
    as see-through surfaces: a surface built from thousands of nearly invisible
    slivers only looks solid where enough of them happen to stack up.

    Three passes, all on the exported PLY so nothing about training changes:

    prune     drop splats below min_opacity. They contribute almost no colour
              but do contribute sorting cost and haze.
    isotropy  raise each splat's shortest axes so no axis is more than
              max_aniso times shorter than the longest. This is the constraint
              the trainer never applied; it fattens slivers back into
              disc-like gaussians without moving or recolouring anything.
    extent    drop splats further than max_extent from centre. Floaters end up
              flung far outside the subject -- the 50k run spanned z -90..77
              for a subject about 2 units tall.
    gapfill   scale_gain grows EVERY splat by a constant factor. The isotropy
              pass above only rescues needles; a splat that is already round
              but simply too small to touch its neighbours is left alone by
              it, and that is what a hole is made of. Growing everything makes
              neighbours overlap and the shell closes.
    opacity   opacity_gain multiplies each splat's alpha. A shell built from
              near-invisible splats reads see-through however well shaped it
              is; measured mean opacity on a real export was 0.116.

    Returns counts so the effect is auditable rather than a black box.
    """
    import re as _re
    raw = src.read_bytes()
    end = raw.find(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode("ascii", "replace")
    if "binary_little_endian" not in header:
        raise RuntimeError("only binary_little_endian PLY is supported")

    count, props = 0, []
    sizes = {"float": 4, "double": 8, "uchar": 1, "char": 1, "ushort": 2,
             "short": 2, "uint": 4, "int": 4}
    for line in header.splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "element" and p[1] == "vertex":
            count = int(p[2])
        elif p[0] == "property" and p[1] in sizes:
            props.append((p[2], p[1]))
    np_of = {"float": "<f4", "double": "<f8", "uchar": "u1", "char": "i1",
             "ushort": "<u2", "short": "<i2", "uint": "<u4", "int": "<i4"}
    dt = np.dtype([(n, np_of[t]) for n, t in props])
    arr = np.frombuffer(raw, dt, count=count, offset=end).copy()

    keep = np.ones(count, bool)
    opac = 1.0 / (1.0 + np.exp(-arr["opacity"].astype(np.float64)))
    if min_opacity > 0:
        keep &= opac >= min_opacity
    n_after_opac = int(keep.sum())

    if max_extent > 0:
        c = np.asarray(centre, np.float64)
        d = np.sqrt((arr["x"] - c[0]) ** 2 + (arr["y"] - c[1]) ** 2
                    + (arr["z"] - c[2]) ** 2)
        keep &= d <= max_extent
    n_after_extent = int(keep.sum())

    arr = arr[keep]
    # scales are stored as logs, so the anisotropy clamp is a floor in log
    # space -- no exp/log round trip on the long axis, which stays exact
    s = np.stack([arr["scale_0"], arr["scale_1"], arr["scale_2"]],
                 1).astype(np.float64)
    fixed = 0
    if max_aniso > 0 and len(arr):
        floor = s.max(axis=1, keepdims=True) - math.log(max_aniso)
        touched = (s < floor).any(axis=1)
        fixed = int(touched.sum())
        s = np.maximum(s, floor)
    # Gap fill. The clamp above only rescues NEEDLES -- it lifts a short axis
    # toward the long one and never touches a splat that is already round, so
    # a hole between two well-shaped splats survives it untouched. A uniform
    # gain grows every splat until neighbours overlap. Scales are stored as
    # logs, so multiplying is adding, and the RATIO between the three axes --
    # the shape of the gaussian -- is preserved exactly.
    if scale_gain and scale_gain != 1.0 and len(arr):
        s = s + math.log(scale_gain)
    if len(arr):
        arr["scale_0"] = s[:, 0].astype(np.float32)
        arr["scale_1"] = s[:, 1].astype(np.float32)
        arr["scale_2"] = s[:, 2].astype(np.float32)

    # Opacity is stored as a logit. Going through probability is what keeps
    # "gain" the intuitive quantity -- 2.0 really is twice as opaque -- and the
    # clamp stops a large gain saturating to exactly 1.0, which would write an
    # infinity into the PLY and render as a black splat.
    boosted = 0
    if opacity_gain and opacity_gain != 1.0 and len(arr):
        pr = 1.0 / (1.0 + np.exp(-arr["opacity"].astype(np.float64)))
        pr = np.clip(pr * opacity_gain, 1e-6, 1.0 - 1e-6)
        arr["opacity"] = np.log(pr / (1.0 - pr)).astype(np.float32)
        boosted = int(len(arr))

    if len(arr) == 0:
        raise RuntimeError(
            f"cleanup would remove every splat (min_opacity={min_opacity}, "
            f"max_extent={max_extent}) - refusing to write an empty PLY. "
            f"Lower the opacity threshold or widen the extent.")

    new_header = _re.sub(r"element vertex \d+",
                         f"element vertex {len(arr)}", header, count=1)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        f.write(new_header.encode("ascii"))
        f.write(arr.tobytes())

    return {"in": int(count), "out": int(len(arr)),
            "dropped_low_opacity": int(count - n_after_opac),
            "dropped_far": int(n_after_opac - n_after_extent),
            "fattened": fixed,
            "boosted": boosted,
            "min_opacity": float(min_opacity),
            "max_aniso": float(max_aniso),
            "scale_gain": float(scale_gain),
            "opacity_gain": float(opacity_gain),
            "bytes": dst.stat().st_size}


def ply_splat_count(path: Path) -> int:
    """Splat count from a PLY header, without loading the body."""
    try:
        with open(path, "rb") as f:
            head = f.read(2048)
        m = re.search(rb"element vertex (\d+)", head)
        return int(m.group(1)) if m else 0
    except Exception:
        return 0


# The properties a 3DGS export must carry. Anything missing them will render
# as noise in the viewer rather than fail, so it is rejected at the door.
GS_REQUIRED_PROPS = ("x", "y", "z", "opacity",
                     "scale_0", "scale_1", "scale_2",
                     "rot_0", "rot_1", "rot_2", "rot_3",
                     "f_dc_0", "f_dc_1", "f_dc_2")


def inspect_splat_ply(path: Path) -> dict:
    """Validate a PLY as a 3DGS export and report what it holds.

    Reads only the header. Raises with a specific reason rather than letting
    a mesh or a point cloud through to the viewer as noise.
    """
    with open(path, "rb") as f:
        head = f.read(65536)
    end = head.find(b"end_header")
    if not head.startswith(b"ply") or end < 0:
        raise RuntimeError("not a PLY file (no header found)")
    text = head[:end].decode("latin-1")          # byte-exact, never raises
    fmt = re.search(r"format\s+(\S+)", text)
    m = re.search(r"element vertex (\d+)", text)
    if not m:
        raise RuntimeError("PLY has no vertex element")
    props = re.findall(r"property\s+\S+\s+(\S+)", text)
    missing = [q for q in GS_REQUIRED_PROPS if q not in props]
    if missing:
        raise RuntimeError(
            "not a 3D Gaussian Splat PLY - missing "
            + ", ".join(missing[:6])
            + (" ..." if len(missing) > 6 else "")
            + ". A splat export carries position, opacity, scale, rotation "
              "and SH colour per point.")
    sh = len([q for q in props if q.startswith("f_rest_")])
    return {"splats": int(m.group(1)),
            "format": fmt.group(1) if fmt else "unknown",
            "sh_rest": sh,
            "sh_degree": {0: 0, 9: 1, 24: 2, 45: 3}.get(sh, -1),
            "properties": len(props)}


def list_exports(out_dir: Path):
    """Checkpoint PLYs, newest step last. Brush names them export_<step>.ply.

    Uploaded splats are named upload_<unix-ts>.ply, so their digits sort after
    any real step count and "follow latest" picks up a fresh upload.
    """
    items = []
    for p in sorted(out_dir.glob("*.ply")):
        digits = "".join(c for c in p.stem if c.isdigit())
        items.append({"name": p.name, "step": int(digits or 0),
                      "uploaded": p.name.startswith("upload_"),
                      "bytes": p.stat().st_size,
                      "mtime": p.stat().st_mtime})
    items.sort(key=lambda x: (x["step"], x["mtime"]))
    return items


def save_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))


def orbit_from_dict(d) -> Orbit:
    f = {k: v for k, v in (d or {}).items() if k in Orbit.__annotations__}
    return Orbit(**f)


def orbit_to_dict(o: Orbit):
    d = asdict(o)
    d["cam_z"] = round(o.cam_z, 4)      # derived, included for display only
    return d
