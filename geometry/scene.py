import numpy as np

from collections.abc import Mapping, Sequence
from typing import Dict, Any, Tuple

from geometry.pose import omega_of, look_at, rodrigues
from geometry.projections import floor_points

from data.types import CameraLock, SolverConfig, Annotation
from data.venue import VenueGeometryStructure, pillar_world

LOCK_MODES = [CameraLock.ROT, CameraLock.MOVED, CameraLock.ZOOM]

class Scene:
    """
        The parameter blokcs, that Ceres holds pointers to, never reallocated
        only written in place.

        The scene runs a few modes, for ex. we lock the camera to a fixed location
        i.e. Cworld (camera center in world is fixed). This is by design choice and 
        how the broadcast clips are shot and constructed for a sequence.

        The parameter block: 

            1. principal_point : cx, cy
            2. focal_length    : f (assume same fx and fy) if we don't lock this parameter as single for sequence it is 1 x n_frames
            3. Cworld          : cx, cy, cz (Camera center in the world)
            4. deltaRadius     : dR the correction of the measurement of the venue that is published
            5. deltaHeight     : dH the correction of the measurement of the height of venue that is published
            6. theta           : one pillar’s azimuth around the cage centre, measured from +x toward +y, in pillar_ids order. theta[0] is held constant to fix world's rotation about vertical.
            7. omega           : 3 x num_frames these are the rodrigues rvec that parametrizes the SO(3) R which Ceres optimizes, R is the world to camera transform
            7. tracked_points  : 2 x num_tracks these are the canvas points that are tracked from annotations
    """

    def __init__(self, n_frames: int, n_pillars: int, n_tracks: int, lock: CameraLock = CameraLock.ROT):
        """
            n_frames  : Number of annotated frames
            n_pillars : Number of annotated pillars in the frames 
            n_tracks  : Number of annotated canvas tracks

        """
        if lock not in LOCK_MODES:
            raise ValueError(f"lock must be one of {LOCK_MODES}, got {lock!r}")
        self.lock = lock
        self.n_frames = n_frames
        self.n_pillars = n_pillars
        self.n_tracks = n_tracks

        # Parameter blocks that ceres will touch
        self.principal_point = np.zeros(2)
        self.focal_length = [np.zeros(1) for _ in
                      range(n_frames if lock == CameraLock.ZOOM else 1)]
        self.camera_center_world = [np.zeros(3) for _ in
                       range(n_frames if lock == CameraLock.MOVED else 1)]
        self.delta_radius = np.zeros(1)
        self.delta_height = np.zeros(1)
        self.theta = [np.zeros(1) for _ in range(self.n_pillars)]
        self.omega = [np.zeros(3) for _ in range(n_frames)]
        self.tracked_points = [np.zeros(2) for _ in range(n_tracks)]

        # book keeping
        self.cost_history: Dict[SolverConfig, float] = {}

        # --------------- Accessors ---------------

    def focal_length_of(self, fi: int = 0):
        return self.focal_length[fi if len(self.focal_length) > 1 else 0]

    def camera_center_of(self, fi: int = 0):
        return self.camera_center_world[fi if len(self.camera_center_world) > 1 else 0]

    def intr_of(self, fi=0):
        """
            [f, cx, cy] for one frame, as a COPY -- the projection helpers take
            it positionally. Writing to the result changes nothing; use
            `set_intr`.
        """
        return np.array([self.focal_length_of(fi)[0], self.principal_point[0], self.principal_point[1]])

    def set_intr(self, f, cx, cy):
        """
            Set the focal on every frame and the shared principal point.
        """
        for b in self.focal_length:
            b[0] = f
        self.principal_point[:] = [cx, cy]

    def set_center(self, C):
        for b in self.camera_center_world:
            b[:] = C

    def set_delta_radius(self, r: float):
        self.delta_radius[0] = r

    def set_delta_height(self, h: float):
        self.delta_height[0] = h

    @property
    def f(self):
        """
            The reference focal (frame 0). Per-frame focals are `focal_length_of(fi)`.
        """
        return self.focal_length[0][0]

    @property
    def C(self):
        """
            The shared camera centre -- frame 0's in CameraLock.MOVED. Writable, since
            it is the real block, but in CameraLock.MOVED prefer `camera_center_of(fi)`.
        """
        return self.camera_center_world[0]

    
    def f_track(self):
        return np.array([self.focal_length_of(t)[0] for t in range(self.n_frames)])

    def c_track(self):
        return np.array([self.camera_center_of(t) for t in range(self.n_frames)])

    def thetas(self):
        return np.array([t[0] for t in self.theta])

    def blocks(self):
        return {"focal": self.focal_length, "pp": [self.principal_point], "center": self.camera_center_world,
                "delta": [self.delta_radius], "dh": [self.delta_height], "theta": self.theta,
                "omega": self.omega, "Q": self.tracked_points}

    def relax(self, lock: CameraLock) -> None:
        """
            Widen focal/center to `lock`'s block count in place, each new block a copy of the shared value.
        """
        if lock not in LOCK_MODES: raise ValueError(...)
        if lock == CameraLock.ZOOM:
            self.focal_length = [self.focal_length[0]] + [self.focal_length[0].copy() for _ in range(self.n_frames - 1)]
        elif lock == CameraLock.MOVED:
            self.camera_center_world = [self.camera_center_world[0]] + [self.camera_center_world[0].copy() for _ in range(self.n_frames - 1)]
        self.lock = lock


        


def initialize_scene(observations: Dict[str, Any], focal_init: float, 
                     venue_geometry: VenueGeometryStructure,
                     image_wh: tuple[int, int],
                     direction: float, # rough init grid search
                     spread: float, # rough init grid search
                     distance: float, # rough init grid search
                     lock: CameraLock = CameraLock.ROT,) -> Scene:
    """
        Defines the unique geometry of the broadcast camera and the structure
        of the stage observed.
    """
    W, H = image_wh

    scene = Scene(
        n_frames=len(observations["frames"]),
        n_pillars=len(observations["pillar_ids"]),
        n_tracks=len(observations["track_ids"]),
        lock=lock,
    )

    scene.set_intr(focal_init, W / 2.0, H / 2.0)

    # grid search params from the main caller
    theta_dir = direction
    spred_deg = spread
    dist = distance
    
    # height control where the broad cast camera is
    # in the world for initialization, we need this to be an
    # input configuration.
    height = 1.9 

    thetas = theta_dir * np.radians(spred_deg) * np.arange(len(observations["pillar_ids"]))
    for i, t in enumerate(thetas):
        scene.theta[i][0] = t
    bearing = thetas.mean() + np.pi
    scene.set_center(
        [dist * np.cos(bearing), dist * np.sin(bearing), height]
    )
    w = omega_of(look_at(scene.C))
    for om in scene.omega:
        om[:] = w

    return scene


def init_landmarks(scene: Scene, observations: Dict[str, Any], radius: float) -> None:
    """
        Back-project each track to Z = 0 and average over the frames that saw it.

        radius is the clip's nominal post radius, use only as the fallback seed for
        a track no grame could back project
    """
    acc = [[] for _ in scene.tracked_points]
    for k, (fi, qi) in enumerate(observations["pt_idx"]):
        f, cx, cy = scene.intr_of(fi)
        P = floor_points(
                observations["pt_uv"][k].reshape(-1, 2),
                rodrigues(scene.omega[fi]),
                scene.camera_center_of(fi),
                f, cx, cy)[0]
        if np.all(np.isfinite(P)):
            acc[qi].append(P[:2])
    fallback = [radius, 0.0]
    for j, a in enumerate(acc):
        scene.tracked_points[j][:] = np.mean(a, axis=0) if a else fallback
        
    return

def get_pillar_world(scene: Scene, structure: VenueGeometryStructure) -> tuple[np.ndarray, np.ndarray]:
    """
        Uses the scene solved thetas, the delta in radius and delta in height to correct the 
        original "measurements" of the stage, to return the pillars in the world coordinate
        frame.
    """
    base_w, top_w = pillar_world(
        theta=scene.thetas(),
        r_in=structure.measurements.radius,
        dR=scene.delta_radius[0],
        pillar_h=structure.measurements.pillar_height,
        dh=scene.delta_height[0] 
    )
    return base_w, top_w


def build_landmarks_all(
        scene:Scene,
        observations: Dict[str, Any],
        obs_all: Dict[str, Any],
        structure: VenueGeometryStructure,
) -> Tuple[Dict[str, Tuple[np.ndarray, Tuple]], Dict[int, float]]:
    """name -> (world XYZ, plane tag), plus {pillar id: theta_rad}.

    """

    P: Dict[str, Tuple[np.ndarray, Tuple]] = {}
    for tid, point in zip(observations["track_ids"], scene.tracked_points):
        P[f"Q{tid}"] = (np.array([point[0], point[1], 0.0]), ("canvas",))

    _, cx, cy = scene.intr_of(0)

    solved_r = structure.measurements.radius + scene.delta_radius[0]
    solved_h = structure.measurements.pillar_height + scene.delta_height[0]

    # For all the pillar ids
    all_ids = list(np.unique(observations["pil_idx"][:,0]))

    # BA gets only a subset of pillar ids
    known = {int(pid): float(th)
             for pid, th in zip(observations["pillar_ids"], scene.thetas())}

    acc: Dict[int, list] = {}
    for row, (fi, slot) in enumerate(obs_all["pil_idx"]):
        R = rodrigues(scene.omega[fi])
        C = scene.camera_center_of(fi)
        f = scene.focal_length_of(fi)[0]
        base_uv = obs_all["pil_base"][row]
        X = floor_points(uv=base_uv[None], R=R, C=C, f=f, cx=cx, cy=cy)[0]
        if np.all(np.isfinite(X)):
            pid = obs_all["pillar_ids"][slot]
            acc.setdefault(pid, []).append(
                np.arctan2(X[1], X[0]))

    thetas: Dict[int, float] = {}
    for pid, angs in sorted(acc.items()):
        th = known.get(pid)
        if th is None:
            a = np.asarray(angs)
            th = float(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a))))
        thetas[pid] = th

        base = np.array([solved_r * np.cos(th), solved_r * np.sin(th), 0.0])
        plane = ("post", pid)
        P[f"P{pid}b"] = (base, plane)
        P[f"P{pid}t"] = (base + np.array([0.0, 0.0, solved_h]), plane)
    

    return P, thetas


def pair_arc_widths(landmarks: Mapping[str, tuple[np.ndarray, tuple[Any, ...]]],
                    thetas: Mapping[int, float],
                    pairs: Sequence[tuple[int, int]]) -> dict[tuple[int, int], float]:
    """{(a, b): metres} arc between edges a and b at the cage radius.

    `landmarks` is `build_landmarks_all` output (`P{a}b` gives the radius); `thetas`
    is `thetas`: edge id -> angle (rad).
    """
    out: dict[tuple[int, int], float] = {}
    for a, b in pairs:
        if f"P{a}b" not in landmarks:
            raise KeyError(f"landmarks has no P{a}b")
        for i in (a, b):
            if i not in thetas:
                raise KeyError(f"thetas has no edge {i}")
        R = float(np.hypot(*np.asarray(landmarks[f"P{a}b"][0], float)[:2]))
        gap = (thetas[b] - thetas[a] + np.pi) % (2 * np.pi) - np.pi
        out[(a, b)] = R * abs(gap)
    return out
