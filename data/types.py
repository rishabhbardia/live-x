from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from enum import StrEnum

import numpy as np

from utils.freeze import check_shape, frozen_array
from geometry.pose import rodrigues
from geometry.projections import project, unproject, world_ray_dirs

@dataclass(frozen=True)
class CameraIntrinsics:
    """One pinhole intrinsic: f, (cx, cy) and the image it belongs to.

    `per_frame` says if this was changing the calibration per-frame or not
    Our data right now does a rotation only and one focal length
    """

    f: float
    cx: float
    cy: float
    width: int
    height: int
    per_frame: bool = False
    source: str = "BA-Annotated"

    # Derived field: excluded from constructor, repr, and equality/hash.
    _K: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not np.isfinite([self.f, self.cx, self.cy]).all() or self.f <= 0:
            raise ValueError("Intrinsics must be finite, with f > 0")

        if self.width <= 0 or self.height <= 0:
            raise ValueError("Image dimensions must be positive")
        K = np.array(
            [
                [self.f, 0.0, self.cx],
                [0.0, self.f, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # Explicit initialization of a derived field in a frozen dataclass.
        object.__setattr__(self, "_K", frozen_array(K))

    @property
    def K(self) -> np.ndarray:
        """(3, 3) float64 calibration matrix, READ-ONLY and cached.

        Published, so frozen: it is handed to every consumer that projects
        anything, and a caller scaling it in place for a resized image would
        move the camera for all of them. `np.array(cam.K)` is the editable
        copy.
        """
        return self._K

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def hfov_deg(self) -> float:
        """Horizontal FOV assuming a centered principal point."""
        return float(
            np.degrees(2.0 * np.arctan(0.5 * self.width / self.f))
        )

    def __repr__(self) -> str:
        return (
            f"Intrinsics(f={self.f:.2f}, "
            f"c=({self.cx:.1f}, {self.cy:.1f}), "
            f"{self.width}x{self.height}, "
            f"hfov {self.hfov_deg:.1f} deg, "
            f"{'per-frame' if self.per_frame else 'per-clip'})"
        )

@dataclass(frozen=True, eq=False)  
class CameraPose:
    """

    The convention

    We assume that camera center is 0 in it's own coordinate system.
    Ccam = R @ Cworld + t, Ccam = 0
    t = -RCworld
    Cworld = -R.t @ t

    Now a point in world Xw in camera frame is as 
    Xcamera = R @ Xworld + t
    Xcamera = R @ (Xworld - Cworld)

    Pinhole projection : 

    u = f * X/Z + cx
    y = f * Y/Z + cy

    R maps the world into camera. Its rows are the camera's aces in world 
    coordinates, 
    R[0] is the right X
    R[1] is the down Y
    R[2] is the forward Z.
    omega is its Rodrigues vector, which is what both bundle adjustment and tracker
    write.

    Source says which process wrote the pose.

    `project` / `unproject` / `rays` / `ground_points` are methods here rather
    than functions a consumer calls

    TODO implement covariance

    """

    index: int
    omega: np.ndarray
    C: np.ndarray
    intr: CameraIntrinsics
    reproj_px: float
    samples: int
    covariance: np.ndarray | None = None
    source: str = "BA-Annotated"
    _R: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Own the input buffers so external aliases cannot modify this pose.
        check_shape(self.C, rows=3)
        object.__setattr__(self, "omega", frozen_array(self.omega))
        object.__setattr__(self, "C", frozen_array(self.C))
        if self.covariance is not None:
            covariance = np.array(
                self.covariance, dtype=np.float64, copy=True
            )
            check_shape(covariance, rows=6, cols=6)
            object.__setattr__(
                self, "covariance", frozen_array(covariance)
            )

        R = frozen_array(rodrigues(self.omega))
        check_shape(R, rows=3, cols=3)
        object.__setattr__(self, "_R", frozen_array(R))
        

    @property
    def R(self) -> np.ndarray:
        """
        (3,3) world-to-camera rotation, whose rows are camera axis in world coordinates
        """
        return self._R

    @property
    def f(self) -> float:
        return self.intr.f

    @property
    def K(self) -> np.ndarray:
        return self.intr.K

    @property
    def forward(self) -> np.ndarray:
        """Unit world direction the camera looks along (its +Z axis)."""
        return self.R[2]

    def project(self, Xw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ 
        World points (N,3) -> pixels (N,2) and camera-space Z (N,)
    
        Pixels for points at or behind the camera (z <= 0) are NaN.
    
        Pinhole projection model, square pixels, no skew or distortion.
        This is just for the broadcast projection and optimization not the
        renderer.
        """
        check_shape(Xw, cols=3)
        return project(Xw, self.R, self.C, self.intr.f, self.intr.cx, self.intr.cy)

    def unproject(self, uv: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Pixels (N, 2) and camera space depth (N,) in meters gives back world (N,3)
    
        Z is the depth along the camera's +Z axis, optical axis what `ClipBundle.metric_depth`
        returns -- not a range along the ray (common for monocular depth models like VDA)
        So the point in camera space would be Z * [x,y,1].
    
        For range it would be r * [x,y,1]/(||[x,y,1]||) i.e. normalized
    
        Xcamera = R @ (Xworld - Cworld)
        Xworld = R.T @ Xcamera + Cworld
        """
        check_shape(uv, cols=2)
        check_shape(Z, rows=uv.shape[0])
        return unproject(uv, Z, self.R, self.C, self.intr.f, self.intr.cx, self.intr.cy)
        
    def rays(self, uv: Any) -> np.ndarray:
        """
        Pixels (N, 2) -> unit world ray directions (N, 3) from `C`.
        """
        check_shape(uv, cols=2)
        return world_ray_dirs(
            uv, self.R, self.f, self.intr.cx, self.intr.cy, unit=True
        )

@dataclass(frozen=True)
class FighterPriors:

    def __init__(self):
        pass

@dataclass(frozen=True)
class Point:
    idx: int
    u: float
    v: float

    def __repr__(self) -> str:
        return (
            f"Id = {self.idx}, "
            f"pt=({self.u:.2f}, {self.v:.2f}), "
        )

@dataclass
class Annotation:
    frame_idx: int
    frame_key: str
    canvas_points: list[Point] = field(default_factory=list)
    bottom_pillar_points: list[Point] = field(default_factory=list)
    top_pillar_points: list[Point] = field(default_factory=list)

    def add_canvas_point(self, pt: Point) -> None:
        self.canvas_points.append(pt)

    def add_top_pillar_point(self, pt: Point) -> None:
        self.top_pillar_points.append(pt)

    def add_bottom_pillar_point(self, pt: Point) -> None:
        self.bottom_pillar_points.append(pt)

    def add_annotations(
        self,
        canvas: list[Point],
        top: list[Point],
        bottom: list[Point],
    ) -> None:
        # Own the containers; immutable Point instances can be shared.
        self.canvas_points = list(canvas)
        self.top_pillar_points = list(top)
        self.bottom_pillar_points = list(bottom)
    
    def to_dict(self):
        Q = "Q"
        P = "P"
        out = {}
        for p in self.canvas_points:
            out[f"{Q}{p.idx}"] = np.array([p.u, p.v], np.float64)
        
        for p in self.top_pillar_points:
            out[f"{P}{p.idx}t"] = np.array([p.u, p.v], np.float64)

        for p in self.bottom_pillar_points:
            out[f"{P}{p.idx}b"] = np.array([p.u, p.v], np.float64)

        return out


@dataclass(frozen=True)
class AnnotationPriors:
    """
    These are annotation and ONE FC specific for now
    """
    # These are annotation sigmas in pixels
    # and is very annotation specifc.
    pillar_base_sigma_px: float
    pillar_top_sigma_px: float
    landmark_sigma_px: float
    principal_point_sigma_px: float

    # VenueMeasurement priors/slack since these are not machine cut
    delta_R_m: float
    delta_H_m: float

    # pillar IDS
    pillar_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "pillar_ids", tuple(self.pillar_ids))

    def __repr__(self) -> str:
        return f"Pillar Base Sigma {self.pillar_base_sigma_px:.3f}, Pillar Top Sigma {self.pillar_top_sigma_px:.3f}, Landmark Sigma {self.landmark_sigma_px:.3f}, Principal Point Sigma {self.principal_point_sigma_px:.2f}, Radius sigma {self.delta_R_m}, height sigma {self.delta_H_m}"
        


class SolverParameters:
    """
        Ceres parameters for setup
    """
    huber: float = 2.0
    max_iters: int = 200
    verbose: bool = False
    focal_grid_rel: list[float] = [0.8, 1.2, 1.6, 2.0]



class _Choice(StrEnum):
    """StrEnum with strict parsing that names the alternatives when it fails.

    StrEnum rather than Enum so a member can still be written into json, used
    in a format string, or compared against a value read off disk without a
    conversion at every boundary -- while `parse` keeps the boundary strict.
    """

    @classmethod
    def parse(cls, value: Any, where: str = "") -> "_Choice":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            for member in cls:
                if member.value == value:
                    return member
        ctx = f" (in {where})" if where else ""
        raise ValueError(
            f"{cls.__name__} must be one of "
            f"{[m.value for m in cls]}, got {value!r}{ctx}")

    @classmethod
    def parse_or_none(cls, value: Any) -> "_Choice | None":
        """For values read off disk where absence is normal and a WRONG value
        should become a reported concern rather than a construction failure."""
        if value is None:
            return None
        try:
            return cls.parse(value)
        except ValueError:
            return None

class CameraLock(_Choice):
    """What the camera is allowed to do between frames.

    Measured by motoin model selector (not landed), never assumed: ROT is the default only
    when no measurement exists, and the bundle always says which it is.
    """
    ROT = "rot"     # t = 0, one focal for the clip
    ZOOM = "zoom"   # t = 0, f per frame against the measured ratio track
    MOVED = "moved" # t per frame: the operator did not rotate about the optical centre


class SolverConfig(_Choice):
    """
        SolverConfig name of the basic config principle solver ran on
    """
    PILLARS_ONLY = "pillars_only"    
    PILLARS_LANDMARKS = "pillars_landmarks"  
    JOINT = "joint_refined"
    RELAX = "relaxed_per_frame" # camera moving and or zooming


class SurfaceType(_Choice):
    """
        Type of surface on ONE FC
    """
    CANVAS = "canvas"    
    POST = "post"  
    BAND = "band"

class TrackStatus(_Choice):
    """
    """
    ACTIVE = "active" 
    LOST = "lost"
    TENTATIVE = "tentative"
    PROVISIONAL = "provisional"

class TrackerMode(_Choice):
    """
    """
    FIXED_CAMERA_CENTER = "fixed_camera_center"
    FREE_CAMERA_CENTER = "free_camera_center"