"""The author workspace: the lens, the lifted cloud, the control render.

The control video is rendered SERVER-SIDE, from the identical camera model that
`steps.write_colmap` later writes out. If the browser rendered the video and
the server wrote the poses, the two could disagree in a way nothing downstream
would catch -- and V3 proved how expensive that class of bug is.

GeoCalib is gone from this build. The camera is front-on at elevation 0 and
the canvas matches the photo, so a pitch/roll measurement has nothing left to
drive; MoGe-2 measures the lens and lifts the cloud. calib.py stays on disk,
unused, and an old project's saved `calib` block is simply ignored.
"""
from __future__ import annotations

import json
import time
from types import SimpleNamespace
from urllib.parse import quote

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response

import steps

from .common import _soft_stamp, _soften_on, soft_path
from .jobs import _busy_msg, _io, _log, _run, progress, refuse, rx
from .models import CloudReq, NewProject, OrbitReq
from .state import _crop, load_state, pdir, save_state, source_images

router = APIRouter()

# The last lift the viewport showed, kept on disk so a page refresh or a server
# restart puts the same points back on screen instead of an empty stage.
CLOUD_BIN = "cloud_view.bin"
CLOUD_META = "cloud_view.json"


@router.get("/api/aspects")
def aspects():
    return {"presets": steps.aspect_presets()}


@router.post("/api/cloud")
def cloud(r: CloudReq):
    """The lifted point cloud, packed for the browser viewport.

    Strength baked at 1.0 with per-point rays, so the viewport rescales
    strength and cutoff in a shader -- MoGe runs once here and never again
    while the sliders move. The MoGe depth itself is cached per source file
    (steps._depth_metric_cached), so a relift after a slider release costs the
    geometry, not the network.

    The packed bytes are also written to cloud_view.bin with the client's lift
    key beside them, which is what /api/cloud/saved reads back after a refresh.
    """
    d = pdir(r.project)
    srcs = source_images(d)
    if not srcs:
        return JSONResponse({"error": "no source image"}, status_code=400)
    orbit = steps.orbit_from_dict({**(load_state(d).get("orbit") or {}),
                                   **r.orbit})
    t0 = time.time()
    try:
        data = steps.cloud_packed(srcs[0], orbit, r.card_height,
                                  depth_model="moge2",
                                  # same ceiling as the render: the viewport is
                                  # meant to predict it, not approximate it
                                  max_points_w=max(96, min(2048, r.points_w)),
                                  isolate_prune=r.isolate_prune,
                                  ground_level=0.0,
                                  crop_sphere=_crop(r.crop_sphere))
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    _log(f"cloud: packed {len(data) / 1e6:.2f} MB for the viewport from "
         f"{srcs[0].name} (moge2, width {r.points_w}) in "
         f"{time.time() - t0:.1f}s", source="cloud")
    try:
        (d / CLOUD_BIN).write_bytes(data)
        (d / CLOUD_META).write_text(json.dumps({
            "key": r.key or "", "time": time.time(), "source": srcs[0].name,
            "bytes": len(data)}), encoding="utf-8")
        _io("wrote", d / CLOUD_BIN, source="cloud",
            note="the viewport's point cloud, read back after a refresh")
    except Exception as e:
        _log(f"cloud: could not keep a copy for the next refresh ({e})",
             level="warn", source="cloud")
    return Response(content=data, media_type="application/octet-stream",
                    headers={"Cache-Control": "no-store"})


@router.get("/api/cloud/saved")
def cloud_saved(project: str):
    """The last lifted cloud for this project, or 404.

    Refused when the source photo is newer than the saved cloud -- that copy
    was lifted from a different image and showing it would be a lie.
    """
    d = pdir(project)
    b, m = d / CLOUD_BIN, d / CLOUD_META
    srcs = source_images(d)
    # 204, not 404: "nothing saved yet" is a normal answer, not an error the
    # browser console should shout about.
    if not (b.exists() and m.exists() and srcs):
        return Response(status_code=204)
    try:
        meta = json.loads(m.read_text(encoding="utf-8"))
    except Exception:
        return Response(status_code=204)
    if srcs[0].stat().st_mtime > b.stat().st_mtime or \
            meta.get("source") not in (None, srcs[0].name):
        _log("cloud: the saved viewport cloud predates the source photo - "
             "ignoring it", source="cloud")
        return Response(status_code=204)
    _io("reading", b, source="cloud", note="saved viewport cloud, after a refresh")
    return Response(content=b.read_bytes(),
                    media_type="application/octet-stream",
                    headers={"Cache-Control": "no-store",
                             "X-Lift-Key": quote(meta.get("key") or ""),
                             "X-Lift-Time": str(meta.get("time") or 0)})


@router.post("/api/fov")
def measure_fov(p: NewProject):
    """Measure the source photo's vertical FOV with MoGe-2.

    Runs automatically after an upload. The client applies it to the shot as a
    starting value; it never overwrites a lens set by hand.
    """
    rx("/api/fov", p.name)
    d = pdir(p.name)
    srcs = source_images(d)
    if not srcs:
        return JSONResponse({"error": "upload an image first"},
                            status_code=400)

    def work():
        progress(phase="MoGe-2 is working out the lens from the photo")
        _log(f"fov: MoGe-2 intrinsics on {srcs[0].name}")
        _io("reading", srcs[0], note="source photo, for intrinsics")
        f = steps.estimate_fov(srcs[0])
        _log(f"fov: MoGe-2 {f['vfov_deg']}deg vertical "
             f"({f['hfov_deg']}deg horizontal, focal {f['focal_px']}px)")
        save_state(d, fov=f, fov_time=time.time())

    if not _run("fov", work, d.name):
        refuse("fov", p.name, _busy_msg())
        return JSONResponse({"error": _busy_msg()}, status_code=409)
    return {"started": True}


@router.post("/api/orbit")
def do_orbit(r: OrbitReq):
    rx("/api/orbit", r.project, clay=r.clay, depth_model="moge2",
       depth_strength=r.depth_strength, fps=r.fps, points_w=r.points_w,
       auto_fit=r.auto_fit, card_height=r.card_height,
       soften=f"{r.soften_downsample:g}x blur {r.soften_blur:g}",
       frames=(r.orbit or {}).get("frames"),
       radius=(r.orbit or {}).get("radius"),
       canvas=f"{(r.orbit or {}).get('width')}x{(r.orbit or {}).get('height')}")
    d = pdir(r.project)
    # The backdrop is flat grey. "scene" and "panorama" are legacy values from
    # removed features; both are normalised away here so nothing downstream
    # has to reason about a backdrop this build cannot render.
    if r.backdrop != "grey":
        _log(f"orbit: backdrop {r.backdrop!r} is not available in this build - "
             f"rendering flat grey", level="warn")
        r.backdrop = "grey"
    if not source_images(d):
        return JSONResponse({"error": "upload an image first"},
                            status_code=400)
    # Front-on, always, in the author workspace: elevation 0 and no offset from
    # the photo. The passes workspace is where the camera leaves this plane.
    orb = dict(r.orbit or {})
    if float(orb.get("elev_deg") or 0) != 0 or float(orb.get("phase_deg") or 0) != 0:
        _log("orbit: elevation and camera offset are locked at 0 in the author "
             "workspace - ignoring the values sent", level="warn")
    orb["elev_deg"] = 0.0
    orb["phase_deg"] = 0.0
    orbit = steps.orbit_from_dict(orb)
    s = load_state(d)
    ci = {"width": s.get("source_w"), "height": s.get("source_h")}
    soft = SimpleNamespace(ref_downsample=max(1.0, float(r.soften_downsample)),
                           ref_blur=max(0.0, float(r.soften_blur)))

    def work():
        card_h = r.card_height
        feet_frac = 0.0
        # AUTO-FIT NO LONGER OVERRIDES YOU. The fit is only SEEDED, on a
        # project that has no orbit recorded yet and therefore has nothing of
        # yours to destroy. After that the field is yours.
        seeding = not (s.get("orbit") or {})
        if r.auto_fit and ci.get("width") and seeding:
            src_vf = float(((s.get("fov") or {}).get("vfov_deg")) or 0.0)
            card_h = round(steps.card_height_full(orbit, ci["width"],
                                                  ci["height"], src_vf), 4)
            _log(f"orbit: card height {card_h} fitted from the "
                 + (f"measured {src_vf:.2f} deg lens" if src_vf > 0
                    else f"shot's {orbit.vfov_deg:.2f} deg lens"))

        srcs = source_images(d)
        if not srcs:
            raise RuntimeError(
                "no source image in this project - re-upload it")
        progress(phase="lifting points")
        _log(f"orbit: DEPTH CLOUD from {srcs[0].name} via MOGE2, strength "
             f"{r.depth_strength} x metric scale, clay={r.clay}, "
             f"{orbit.frames} frames, canvas {orbit.width}x{orbit.height}")
        _io("reading", srcs[0], note="source photo, lifted to a point cloud")

        def tick(i, n):
            progress(i, n, phase="rendering frames")

        info = steps.render_orbit_depthcloud(
            srcs[0], orbit, d / "control_frames", card_height=card_h,
            feet_frac=feet_frac, depth_strength=r.depth_strength,
            backdrop=r.backdrop, clay_mode=r.clay,
            backface_cull=r.backface_cull, depth_model="moge2",
            max_points_w=max(160, min(2048, r.points_w)),
            depth_cutoff=r.depth_cutoff,
            isolate_prune=r.isolate_prune,
            crop_sphere=_crop(r.crop_sphere),
            ground_level=0.0, bend=0.0, bend_falloff=1.0, bend_coverage=1.0,
            subject_x=r.subject_x, subject_y=r.subject_y,
            subject_z=r.subject_z, subject_rot_deg=r.subject_rot_deg,
            subject_rot_x=r.subject_rot_x, subject_rot_y=r.subject_rot_y,
            mp4_out=d / "control.mp4", fps=r.fps,
            first_frame_out=d / "control_frame0.png",
            matte_out=d / "control_mattes", log=_log, progress=tick)
        _log(f"orbit: {info['points']} points; floor visible: "
             f"{info['floor_visible']}")
        if info.get("pruned_points"):
            _log(f"orbit: isolated-point cleanup dropped "
                 f"{info['pruned_points']:,} points with no neighbours")
        if info.get("anchor_depth_m"):
            _log(f"orbit: MoGe-2 metric, subject median depth "
                 f"{info['anchor_depth_m']} m anchored to the card plane")
        _io("rendered", srcs[0], d / "control.mp4",
            note=f"{orbit.frames} frames at {r.fps} fps - kept on disk as the "
                 f"source of the softened clip")
        _io("wrote", d / "control_frame0.png", note="control frame 0")
        _io("wrote", d / "control_mattes",
            note="authored coverage: where points actually landed")

        # ---- soften, in the SAME job. The softened clip is the one you see
        # and the one that is sent; control.mp4 stays on disk underneath it
        # because the soften stamp, the Wan/MiniMax duration and recovery all
        # read it.
        dst = soft_path(d)
        soft_state = {"control_soft": None, "control_soft_factor": None,
                      "control_soft_blur": None}
        if _soften_on(soft):
            progress(0, 0, phase="softening")
            sinfo = steps.soften_video(d / "control.mp4", dst,
                                       factor=float(soft.ref_downsample),
                                       blur=float(soft.ref_blur),
                                       fps=int(r.fps), log=_log)
            steps.stamp_write_file(dst, **_soft_stamp(soft, d))
            _io("softened", d / "control.mp4", dst,
                note=f"{sinfo['src_w']}x{sinfo['src_h']} -> "
                     f"{sinfo['width']}x{sinfo['height']}, blur "
                     f"{sinfo['blur_px']}px - this is the clip that is sent")
            soft_state = {"control_soft": dst.name,
                          "control_soft_factor": float(soft.ref_downsample),
                          "control_soft_blur": float(soft.ref_blur)}
        else:
            for q in (dst, dst.with_suffix(dst.suffix + ".built_from.json")):
                if q.exists():
                    _io("deleting", q, note="softening is off - the sharp "
                                            "render is what gets sent")
                    q.unlink()
        save_state(d, stage="generate", orbit_time=time.time(),
                   fps=r.fps,
                   control_video="control.mp4",
                   reference=None, reference_info=None,
                   orbit=steps.orbit_to_dict(orbit), orbit_info=info,
                   backdrop=r.backdrop, card_height=card_h,
                   crop_sphere=r.crop_sphere, **soft_state)

    if not _run("orbit", work, d.name):
        refuse("orbit", r.project, _busy_msg())
        return JSONResponse({"error": _busy_msg()}, status_code=409)
    return {"started": True}
