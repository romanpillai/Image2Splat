"""Request bodies.

One module because the route packages share them -- GenReq is read by the
generate routes AND by the passes routes, and a model defined next to one of
them would make the other import a route module for a type, which is how
import cycles start.

The comments here are the load-bearing part. Most of these fields encode a
measurement or a fal quirk, and the reasoning is worth more than the default.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class NewProject(BaseModel):
    name: str


class UIState(BaseModel):
    project: str
    ui: dict = {}


class CloneReq(BaseModel):
    project: str
    name: str = ""      # optional "save as" name; default <project>-copy


class RenameReq(BaseModel):
    project: str          # current folder name
    name: str             # what to call it


class RefDel(BaseModel):
    project: str
    name: str


class OrbitReq(BaseModel):
    project: str
    orbit: dict = {}
    card_height: float = 3.6
    auto_fit: bool = True        # scale the card so the subject fills the frame
    clay: str = "shaded"         # shaded | off
    backdrop: str = "grey"       # grey is the only backdrop in this build
    depth: bool = False          # depth-displaced point cloud instead of card
    depth_strength: float = 1.5  # world-units of relief across the depth range
    backface_cull: bool = True   # hide points seen from behind the shell
    depth_model: str = "moge2"   # see steps.DEPTH_MODELS: moge2 | da2 | da2-large | da3-metric | da3-mono
    fps: int = 24                # playback speed; poses are unchanged, only how fast they run
    points_w: int = 640          # depth-cloud sampling width; point count scales with its square
    # Composing the lifted cloud:
    depth_cutoff: float = 0.0         # drop points > N units behind the card
    isolate_prune: float = 0.0        # 0..1, drop points with no neighbours
    ground_level: float = 0.0         # 0..1, re-place floor pixels on a plane
    # Isolate volume: [x, y, z, radius] in WORLD units, or null for off. Kept
    # as a lift parameter rather than an edit of the loaded points, because the
    # cloud is rebuilt from the depth model whenever anything upstream changes
    # and a points-only edit would vanish on the next relift.
    crop_sphere: list | None = None
    bend: float = 0.0                 # world units of deflection at the front
    bend_falloff: float = 1.0
    bend_coverage: float = 1.0
    subject_x: float = 0.0            # translate / rotate the whole cloud
    subject_y: float = 0.0
    subject_z: float = 0.0
    subject_rot_deg: float = 0.0      # about world Z (vertical)
    subject_rot_x: float = 0.0        # about world X
    subject_rot_y: float = 0.0        # about world Y
    # Softening runs in the SAME job as the render, so pressing Render leaves
    # one clip to look at: the softened one. 1 and 0 mean off (sharp is sent).
    soften_downsample: float = 1.0
    soften_blur: float = 0.0


class CloudReq(BaseModel):
    """The lifted point cloud for the viewport. Same lift parameters as the
    render, so the viewport predicts the render rather than approximating it."""
    project: str
    orbit: dict = {}
    card_height: float = 3.6
    depth_strength: float = 1.0
    depth_model: str = "moge2"
    clay: str = "off"
    frame: int = 0
    depth_cutoff: float = 0.0
    isolate_prune: float = 0.0
    ground_level: float = 0.0
    crop_sphere: list | None = None
    bend: float = 0.0
    bend_falloff: float = 1.0
    bend_coverage: float = 1.0
    subject_x: float = 0.0
    subject_y: float = 0.0
    subject_z: float = 0.0
    subject_rot_deg: float = 0.0
    subject_rot_x: float = 0.0
    subject_rot_y: float = 0.0
    points_w: int = 640
    # The client's description of the lift settings. Stored beside the saved
    # cloud so a refresh can tell whether what comes back is still current.
    key: str = ""


class GenReq(BaseModel):
    # populate_by_name so BOTH "pass" (the wire name) and pass_ (the python
    # one) work -- "pass" is a keyword and cannot be an attribute.
    model_config = ConfigDict(populate_by_name=True)
    # An orbit PASS instead of the authored control video. "top"/"bottom"
    # swap in pass_top.mp4 / pass_bottom.mp4 -- the pass is an ordinary
    # generation whose control video happens to come from the splat rather
    # than from the point cloud.
    pass_: str = Field("", alias="pass")
    project: str
    prompt: str = ""
    # ---- reference softening --------------------------------------------
    # The control render's detail is an artefact of the point lift, not the
    # subject. Softening the CONTROL VIDEO leaves structure and motion while
    # inviting the model to invent real detail. The reference image and first
    # frame are deliberately NOT softened -- they are what the subject looks
    # like, and that is the one thing worth keeping sharp.
    ref_downsample: float = 1.0    # 1 = off, 2/3/4 = detail thrown away
    ref_blur: float = 0.0          # extra blur, in pixels, on top
    intensity: str = "strong-v2"
    detail_refine: bool = False
    resolution: str = "720p"
    video_quality: str = "high"
    # control = the control video's own frame 0: same camera, same canvas,
    # same framing as the clip being asked for, so nothing has to be
    # reconciled. auto | source | composited remain as alternatives.
    reference_mode: str = "control"
    engine: str = "ltx"            # ltx (pose-exact) | wan (reference)
    # Wan-only. Its endpoint has no frame-count input, so duration is derived
    # from the orbit unless overridden here.
    wan_resolution: str = "720p"   # 480p | 720p | 1080p
    # ---- Wan 2.2 VACE (depth control) ------------------------------------
    # The depth PASS was removed in this build, so the control video is always
    # the point render. vace_control is kept so an old client sending it does
    # not 422, and "depth" now resolves to the point render with a warning.
    vace_control: str = "control"      # control (the only real option now)
    vace_reference: str = "source"     # source | frame0 | none
    vace_first_frame: str = "frame0"   # source | frame0 | none
    vace_resolution: str = "720p"
    vace_steps: int = 30
    vace_guidance: float = 5.0
    vace_sampler: str = "unipc"
    vace_negative: str = ""
    vace_seed: int | None = None
    wan_aspect: str = "adaptive"
    # MiniMax H3. Its resolutions are UPPERCASE in the API ("768P"), and
    # prompt expansion is a mode rather than a switch -- "fast" rewrites the
    # prompt least, which is what protects the turntable boilerplate.
    mm_resolution: str = "768P"        # 480P | 768P | 2K | 4K
    mm_aspect: str = "adaptive"
    mm_duration: float = 0.0           # 0 = derive from the control video
    mm_smart_duration: bool = False    # let the model choose
    mm_expansion: str = "fast"         # fast | balanced | quality
    mm_seed: int | None = None
    # float, not int: the natural value is frames/fps (150/24 = 6.25s) and a
    # typed-in 6.25 used to 422 with an unreadable body. Wan needs whole
    # seconds, so this is rounded at the point of use and the UI says so.
    wan_duration: float = 0
    wan_smart_duration: bool = False   # True sends null ("smart duration")
    wan_audio: bool = False
    wan_prompt_expansion: bool = False
    wan_thinking: bool = False
    wan_seed: int | None = None


class DescribeReq(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    project: str
    model: str = ""            # "" -> falclient.VISION_MODELS[0] at the route
    extra: str = ""
    # Which workspace asked. The description itself is the same either way --
    # one subject -- but the prompt PREVIEW is not: a pass carries a camera
    # clause at position 2 and the author clip does not. Logging the author's
    # string under "EXACTLY as sent" while a pass was selected is a lie about
    # the one thing this log exists to show.
    pass_: str = Field("", alias="pass")


class RetimeReq(BaseModel):
    project: str


class AiRmbgReq(BaseModel):
    project: str
    model: str = "birefnet"
    # Only used by the text-prompted model. Period-separated nouns, e.g.
    # "a horse. a person. a car." -- every match becomes a tracked object.
    prompt: str = ""


class SplatReq(BaseModel):
    project: str
    # Where the training alpha comes from:
    #   authored - coverage recorded by the control render (dense as the cloud)
    #   rmbg     - RMBG-1.4 segmentation of the AI frames themselves
    #   distance - colour distance to the rendered backdrop only
    # NOT "authored". The control-render coverage marks every pixel a point
    # landed on, so it collapses as the camera orbits away from the photo --
    # measured 88.4% of the frame at the photo viewpoint, 15.6% a quarter turn
    # later. As a training alpha that instructs the trainer to delete most of
    # the subject on most frames.
    matte_source: str = "birefnet"
    # Build the COLMAP dataset and STOP, without launching Brush. Everything
    # up to this point -- frame extraction, matting, the init cloud, the
    # authored poses, the pose verification -- is identical to a real run, so
    # the folder that lands on disk is exactly the one training would have
    # used. That is the point: it is opened in Brush by hand.
    dataset_only: bool = False
    rerun: bool = False            # stream live telemetry to a Rerun viewer
    rerun_stats_every: int = 50
    rerun_splats_every: int = 0    # 0 = never; heavy, see brush_cmd
    rerun_max_img: int = 512
    steps: int = 50000
    detail: str = "stock"        # stock (Brush defaults) | broad | balanced
                                 # | quality | fine
    sh_degree: int = 3           # Brush's own default; ignored by "stock"
    # Off by default: the cleanup was written to fix needles, and it does,
    # but it also prunes and reshapes a result that may not need it. Stock
    # training plus no post-processing is the baseline everything else should
    # be judged against.
    clean: bool = False          # post-training prune / de-needle pass


# ------------------------------------------------------------ orbit passes --
class PassEncode(BaseModel):
    project: str
    which: str = "top"
    fps: int = 30


class PassSoften(BaseModel):
    project: str
    which: str = "top"
    ref_downsample: float = 1.0
    ref_blur: float = 0.0


class PassConform(BaseModel):
    project: str
    which: str = "top"


class PassRetime(BaseModel):
    project: str
    which: str = "top"


class DatasetComplete(BaseModel):
    project: str
    # The AUTHOR ring is not optional and has no flag. It is the ring the
    # subject was calibrated from, the one frame 0 anchors, and the only one
    # with a real photograph behind it -- a set without it is not a smaller
    # version of this, it is a different thing.
    top: bool = False
    bottom: bool = False
    matte_source: str = "sam2_text"
    # Init points from the trained splat, subsampled. This is what makes the
    # build a REFINEMENT of the splat you already have rather than a fresh
    # solve: Brush initialises from points3D.txt, so seeding it with the
    # trained geometry starts the optimiser where the last run finished.
    init_from_splat: bool = True
    max_init_points: int = 250000
