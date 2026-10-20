from collections.abc import Mapping
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import numpy as np
import pyceres

from data.types import TrackStatus, SurfaceType
from geometry.costs import StaticPointCost
from geometry.pose import rodrigues, plane_warp
from geometry.projections import project, plane_points
from geometry.scene import Scene
from geometry.solver import solve
from geometry.surface import StageSurfaces

from tracker import constants as defaults
from tracker.features import (
    create_mser,
    gray_image,
    pick_replenish_candidates,
    reacquire,
    track_points,
    warp_reference,
)

"""
Track fixed world landmarks through a clip, starting from a calibrated seed.

Fixed world landmarks, fixed intrinsics, forward and backward pass from the calibrated seed.

The landmarks have the following states : 

1. ACTIVE : Trusted landmark with a current 2D position. Seed, promoted or successfully reacquired.
2. PROVISIONAL : New sruface landmark, age starts from 0. Tracked, but excluded from pose estimation.
3. TENTATIVE : LOST landmark matchd in the current image. Awating validation through a pose fit.
4. LOST : Keep fixed 3D + surface, remove live 2D. Retain for possible reacquistion.

   Track Active and Provisional points

    Previous frame + live 2D points , Current frame + person mask
                                |
                                |
            Forward and backward LK + mask gating
    
        |           |                  |                      |
        |           |                  |                      |

  Valid Active.  Valid Provisional.   Failed Active.          Failed Provisional. 
A = survived       P= surviving         Mark LOST, drop 2D.     Delete candidate, Drop 2D, 2D, stage and age


Solve 1 : Pose Fit using A

Validate P against pose from Sovle 1 (These candidates dont participate in Solve 1)
If succeeded the reprojection test then each passing candidate is aged +1

Also computes reprojection errors on A as a diagnostic.


Reacquire LOST Landmarks -- trial, fit, filter, refit and recheck

LOST pool (has the fixed world position and the surface it is on)
Predict pixel + compute plane H
Use seed/current poses on fixed plane basis

    A_j = K[R_jxe1 R_j x e2, R_j (O - C_j)]

    H = A_current x inv(A_seed)
Project fixed 3D with the working pose to rpedict pixels
Warp the seed image + template matching gives T candidates

Solve 2 : Pose solve on A + T, this is a candidate pose

Filter with reprojection error to get S candidates

Now Solve 3 : Pose solve with A + S

End with a replenish step if the tracking landmarks per surface are
running low with MSER.


"""


Landmark = tuple[np.ndarray, tuple[Any, ...]]


_SETTING_NAMES = (
    "DEFAULT_LEVELS",
    "WIN",
    "DEFAULT_DILATE",
    "DEFAULT_FB_TOL",
    "MIN_POSE_PTS",
    "SIGMA_MSER",
    "PROMOTE_FRAMES",
    "REPLENISH_TOL",
    "REACQ_WINDOW",
    "REACQ_NCC",
    "REACQ_PATCH",
    "PROMOTE_TOL",
    "REPLENISH_CANVAS_TARGET",
    "REPLENISH_POST_TARGET",
    "REPLENISH_BAND_TARGET",
    "REPLENISH_MIN_SEP",
    "REPLENISH_MAX_NEW",
)


def _plane_tag(plane) -> tuple:
    """Normalize enum/string surface tags to a common dictionary key."""
    return (SurfaceType(plane[0]), *plane[1:])


def _median(values) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else None



class Tracker:
    """Track fixed world landmarks through a calibrated clip.

    Owns records, landmarks and continuation state. Does not retain the clip,
    scene, surface object, Ceres objects or OpenCV detector objects.

    Intrinsics are fixed for the run, taken from the seed calibration frame.
    """

    def __init__(
        self,
        clip: Any,
        surface: StageSurfaces,
        scene: Scene,
        observations: Mapping[str, Any],
        landmarks: Mapping[str, Landmark],
        start_annotation_frame_idx: int,
        id_offset: int = 0,
        step: int = 1,
        free_center: bool = False,
        replenish: bool = True,
        post_basis_ids: Mapping[int, int] | None = None,
    ):
        if step == 0:
            raise ValueError("step must be nonzero")

        self.records: list[dict[str, Any]] = []
        self.landmarks: dict[str, Landmark] = {
            name: (
                np.asarray(point, dtype=np.float64).reshape(3).copy(),
                _plane_tag(plane),
            )
            for name, (point, plane) in landmarks.items()
        }
        self.base_names = tuple(sorted(self.landmarks))
        self.calibration_frames = tuple(observations["frames"])

        self.seed_frame_key = self.calibration_frames[
            start_annotation_frame_idx
        ]
        self.current_index = clip.frame_key_index(self.seed_frame_key)
        self.current_frame_key = self.seed_frame_key
        self.frame_count = clip.size()

        self.step = step
        self.free_center = free_center
        self.replenish = replenish
        self.next_id = id_offset

        # Snapshot settings so loading does not silently adopt changed defaults.
        self.settings = SimpleNamespace(
            **{
                name: deepcopy(getattr(defaults, name))
                for name in _SETTING_NAMES
            }
        )
        self.landmark_sigma = float(
            clip.get_clip().annotation_priors.landmark_sigma_px
        )
        if (
            not np.isfinite(self.landmark_sigma)
            or self.landmark_sigma <= 0
        ):
            raise ValueError("landmark_sigma_px must be finite and positive")

        self.intrinsics = np.asarray(
            scene.intr_of(start_annotation_frame_idx),
            dtype=np.float64,
        ).copy()
        f, cx, cy = self.intrinsics
        self.K = np.array(
            [[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        self.reference_omega = np.asarray(
            scene.omega[start_annotation_frame_idx],
            dtype=np.float64,
        ).copy()
        self.reference_center = np.asarray(
            scene.camera_center_of(start_annotation_frame_idx),
            dtype=np.float64,
        ).copy()
        self.omega = self.reference_omega.copy()
        self.center = self.reference_center.copy()

        self.reference_gray = gray_image(
            clip.get_frame(self.current_index).broadcast_frame
        )
        self.previous_gray = self.reference_gray.copy()

        annotations = clip.get_clip().annotations
        if self.seed_frame_key not in annotations:
            raise ValueError(
                f"No annotations for seed frame {self.seed_frame_key!r}"
            )

        seed_positions = annotations[self.seed_frame_key].to_dict()
        self.positions = {
            name: np.asarray(uv, dtype=np.float64).reshape(2).copy()
            for name, uv in seed_positions.items()
            if name in self.landmarks
        }
        self.status = {
            name: (
                TrackStatus.ACTIVE
                if name in self.positions
                else TrackStatus.LOST
            )
            for name in self.base_names
        }
        self.ages: dict[str, int] = {}

        self.canvas_basis = deepcopy(surface.canvas_basis())
        self.post_bases = deepcopy(surface.post_basis())
        self.band_bases = deepcopy(surface.band_basis())
        self.post_basis_ids = dict(
            post_basis_ids
            if post_basis_ids is not None
            else {0: 0, 1: 0, 2: 2, 3: 2, 4: 4, 5: 4}
        )
        self.seed_frame_index = self.current_index
        self._seed_positions = {
            name: uv.copy() for name, uv in self.positions.items()
        }
        self._seed_status = self.status.copy()
        self._record(
            stem=self.seed_frame_key,
            index=self.seed_frame_index,
            errors_before=self._errors(self._names(TrackStatus.ACTIVE)),
            reacquired=[],
            promoted=[],
            replenished=[],
            pose_solved=False,  # Supplied by calibration, not solved by this tracker.
        )
        self.records[-1]["leg"] = "seed"
        self.records[-1]["pose_source"] = "calibration"

    def _names(self, status: TrackStatus) -> list[str]:
        return [
            name for name, value in self.status.items()
            if value == status
        ]

    def _observed(self, names):
        return [(name, self.positions[name]) for name in names]

    def _forget(self, name: str) -> None:
        """Erase a failed provisional landmark completely."""
        self.status.pop(name, None)
        self.positions.pop(name, None)
        self.ages.pop(name, None)
        self.landmarks.pop(name, None)

    def _lose(self, name: str) -> None:
        self.status[name] = TrackStatus.LOST
        self.positions.pop(name, None)

    def _new_name(self, prefix: str) -> str:
        while True:
            self.next_id += 1
            name = f"{prefix}{self.next_id}"
            if name not in self.landmarks:
                return name

    def _project(self, names, omega=None, center=None):
        points = np.asarray(
            [self.landmarks[name][0] for name in names],
            dtype=np.float64,
        ).reshape(-1, 3)

        if not len(points):
            return np.empty((0, 2)), np.empty(0)

        f, cx, cy = self.intrinsics
        return project(
            points,
            rodrigues(self.omega if omega is None else omega),
            self.center if center is None else center,
            f,
            cx,
            cy,
        )

    def _errors(self, names, omega=None, center=None):
        if not names:
            return np.empty(0)

        predicted, depths = self._project(names, omega, center)
        observed = np.asarray([self.positions[name] for name in names])
        errors = np.linalg.norm(predicted - observed, axis=1)
        errors[(depths <= 0) | ~np.isfinite(depths)] = np.nan
        return errors

    def _solve_pose(self, names):
        if len(names) < self.settings.MIN_POSE_PTS:
            return None

        intrinsics = self.intrinsics.copy()
        center = self.center.copy()
        omega = self.omega.copy()

        problem = pyceres.Problem()
        loss = pyceres.HuberLoss(2.0)
        keep = [loss]

        for name, uv in self._observed(names):
            sigma = (
                self.settings.SIGMA_MSER
                if name.startswith("M")
                else self.landmark_sigma
            )
            cost = StaticPointCost(self.landmarks[name][0], uv, sigma)
            keep.append(cost)
            problem.add_residual_block(
                cost, loss, [intrinsics, center, omega]
            )

        problem.set_parameter_block_constant(intrinsics)
        if not self.free_center:
            problem.set_parameter_block_constant(center)

        summary = solve(problem, max_iter=200)
        if (
            not summary.IsSolutionUsable()
            or not np.isfinite(omega).all()
            or not np.isfinite(center).all()
            or not np.isfinite(summary.final_cost)
        ):
            return None

        return omega, center

    def _accept_pose(self, solution) -> None:
        self.omega, self.center = solution

    def _flow(self, gray, person):
        """Track active and provisional points in one LK call."""
        names = [
            name for name, status in self.status.items()
            if status in (TrackStatus.ACTIVE, TrackStatus.PROVISIONAL)
        ]
        if not names:
            return [], []

        positions, good = track_points(
            self.previous_gray,
            gray,
            np.asarray([self.positions[name] for name in names]),
            person,
            levels=self.settings.DEFAULT_LEVELS,
            win=(self.settings.WIN, self.settings.WIN),
            fb_tolerance=self.settings.DEFAULT_FB_TOL,
        )

        active, provisional = [], []
        for name, uv, valid in zip(names, positions, good):
            is_provisional = self.status[name] == TrackStatus.PROVISIONAL

            if valid:
                self.positions[name] = uv.copy()
                (provisional if is_provisional else active).append(name)
            elif is_provisional:
                self._forget(name)
            else:
                self._lose(name)

        return active, provisional

    def _promote_provisional(self, names, pose_solved):
        # A stale pose must not count as another successful validation frame.
        if not pose_solved:
            for name in names:
                self.ages[name] = 0
            return []

        promoted = []
        for name, error in zip(names, self._errors(names)):
            if not np.isfinite(error) or error > self.settings.REPLENISH_TOL:
                self._forget(name)
                continue

            self.ages[name] = self.ages.get(name, 0) + 1
            if self.ages[name] >= self.settings.PROMOTE_FRAMES:
                self.status[name] = TrackStatus.ACTIVE
                self.ages.pop(name, None)
                promoted.append(name)

        return promoted

    def _basis(self, plane):
        kind = plane[0]
        if kind == SurfaceType.CANVAS:
            return self.canvas_basis
        if kind == SurfaceType.POST:
            post_id = plane[1]
            basis_id = self.post_basis_ids.get(post_id, post_id)
            return self.post_bases[basis_id]
        if kind == SurfaceType.BAND:
            return self.band_bases[plane[1]]
        raise ValueError(f"Unsupported surface: {plane!r}")

    def _reacquire(self, gray, person, active):
        warps = {}
        tentative = []

        lost = self._names(TrackStatus.LOST)
        predicted, depths = self._project(lost)

        for name, uv, depth in zip(lost, predicted, depths):
            if depth <= 0 or not np.isfinite(uv).all():
                continue

            plane = self.landmarks[name][1]
            if plane not in warps:
                try:
                    homography = plane_warp(
                        self.K,
                        self.reference_omega,
                        self.reference_center,
                        self.omega,
                        self.center,
                        self._basis(plane),
                    )
                except np.linalg.LinAlgError:
                    warps[plane] = None
                else:
                    warps[plane] = (
                        warp_reference(
                            self.reference_gray, homography, gray.shape
                        )
                        if np.isfinite(homography).all()
                        else None
                    )

            if warps[plane] is None:
                continue

            match = reacquire(
                gray,
                warps[plane],
                uv,
                window=self.settings.REACQ_WINDOW,
                ncc_min=self.settings.REACQ_NCC,
                patch=self.settings.REACQ_PATCH,
                person=person,
            )
            if match is not None:
                self.positions[name] = match[0]
                self.status[name] = TrackStatus.TENTATIVE
                tentative.append(name)

        if not tentative:
            return [], False

        solution = self._solve_pose(active + tentative)
        accepted = []

        if solution is not None:
            omega, center = solution
            errors = self._errors(tentative, omega, center)
            accepted = [
                name for name, error in zip(tentative, errors)
                if np.isfinite(error) and error <= self.settings.PROMOTE_TOL
            ]

        # Refit without rejected tentative points: their observations should
        # not influence the pose we retain.
        final_solution = (
            self._solve_pose(active + accepted) if accepted else None
        )
        if final_solution is None:
            accepted = []
        else:
            errors = self._errors(accepted, *final_solution)
            if np.any(
                ~np.isfinite(errors)
                | (errors > self.settings.PROMOTE_TOL)
            ):
                accepted = []
                final_solution = None

        accepted_set = set(accepted)
        for name in tentative:
            if name in accepted_set:
                self.status[name] = TrackStatus.ACTIVE
            else:
                self._lose(name)

        if final_solution is not None:
            self._accept_pose(final_solution)

        return accepted, final_solution is not None

    def _surface_spec(self, surface, plane):
        """Return population target, sampling plane and name prefix."""
        kind = plane[0]
        if kind == SurfaceType.CANVAS:
            return (
                self.settings.REPLENISH_CANVAS_TARGET,
                np.zeros(3),
                np.array([0.0, 0.0, 1.0]),
                "MQ",
            )

        if kind == SurfaceType.POST:
            index = plane[1]
            origin, normal, _ = surface._post_geometry(
                surface.post_theta_centers[index],
                surface.post_half_width[index],
            )
            return (
                self.settings.REPLENISH_POST_TARGET,
                origin,
                normal,
                f"MP{index}_",
            )

        if kind == SurfaceType.BAND:
            panel = plane[1]
            origin, normal, _ = surface._band_geometry(
                *surface.band_panels[panel]
            )
            return (
                self.settings.REPLENISH_BAND_TARGET,
                origin,
                normal,
                f"MB{panel[0]}_{panel[1]}_",
            )

        raise ValueError(f"Unsupported surface: {plane!r}")

    def _replenish(self, clip, surface, index, gray, person, mser):
        masks = surface.all_masks(self.omega, self.center, index, clip)
        existing = list(self.positions.values())
        added = []

        f, cx, cy = self.intrinsics
        R = rodrigues(self.omega)

        for key, mask in masks.items():
            plane = _plane_tag(key)
            target, origin, normal, prefix = self._surface_spec(
                surface, plane
            )

            have = sum(
                name.startswith("M")
                and status in (
                    TrackStatus.ACTIVE,
                    TrackStatus.PROVISIONAL,
                )
                and self.landmarks[name][1] == plane
                for name, status in self.status.items()
            )
            need = min(
                target - have, self.settings.REPLENISH_MAX_NEW
            )
            if need <= 0:
                continue

            mask = np.asarray(mask, dtype=bool).copy()
            if person is not None:
                mask &= ~np.asarray(person, dtype=bool)

            candidates = pick_replenish_candidates(
                gray,
                mask,
                existing,
                mser,
                min_sep=self.settings.REPLENISH_MIN_SEP,
                max_n=need,
            )

            for uv in candidates:
                point = plane_points(
                    uv.reshape(1, 2),
                    R,
                    self.center,
                    f,
                    cx,
                    cy,
                    origin,
                    normal,
                )[0]

                if not np.isfinite(point).all():
                    continue
                if (R @ (point - self.center))[2] <= 0:
                    continue

                name = self._new_name(prefix)
                self.landmarks[name] = (point.copy(), plane)
                self.positions[name] = uv.copy()
                self.status[name] = TrackStatus.PROVISIONAL
                self.ages[name] = 0

                existing.append(uv)
                added.append(name)

        return added

    def sort_order(self) -> None:
        self.sorted_records = sorted(self.records, key=lambda record: record["stem"])
    
    def to_dict(self) -> dict:
        return {
            "frames": sorted(self.records, key=lambda record: record["stem"])
        }
    
    def _record(
        self,
        stem,
        index,
        errors_before,
        reacquired,
        promoted,
        replenished,
        pose_solved,
    ):
        active = self._names(TrackStatus.ACTIVE)
        errors = self._errors(active)
        _, depths = self._project(list(self.positions))

        record = {
            "stem": stem,
            "frame_index": index,
            "leg": "fwd" if self.step > 0 else "bwd",
            "calibration": stem in self.calibration_frames,
            "n_active": len(active),
            "n_provisional": len(self._names(TrackStatus.PROVISIONAL)),
            "n_lost": sum(
                self.status[name] == TrackStatus.LOST
                for name in self.base_names
            ),
            "n_lost_total": len(self._names(TrackStatus.LOST)),
            "n_reacquired": len(reacquired),
            "reacq": list(reacquired),
            "n_promoted_new": len(promoted),
            "promoted_new": list(promoted),
            "n_replenished": len(replenished),
            "replenished": list(replenished),
            "obs": {
                name: self.positions[name].tolist() for name in active
            },
            "omega": self.omega.tolist(),
            "C": self.center.tolist(),
            "t": (-rodrigues(self.omega) @ self.center).tolist(),
            "dC_m": float(
                np.linalg.norm(self.center - self.reference_center)
            ),
            "reproj_before_px": _median(errors_before),
            "reproj_after_px": _median(errors),
            "behind_camera": int(np.count_nonzero(depths <= 0)),
            "pose_solved": pose_solved,
        }

        for label, prefix in (
            ("canvas", "Q"),
            ("pillar", "P"),
            ("mser", "M"),
        ):
            selected = [
                error for name, error in zip(active, errors)
                if name.startswith(prefix)
            ]
            record[f"n_{label}"] = len(selected)
            record[f"reproj_{label}_px"] = _median(selected)

        self.records.append(record)

    
    def start_pass(self, step: int, id_offset: int = 0) -> None:
        """Restart at the seed; preserve records and the landmark archive."""
        if step == 0:
            raise ValueError("step must be nonzero")

        self.step = step
        self.next_id = id_offset

        self.current_index = self.seed_frame_index
        self.current_frame_key = self.seed_frame_key

        self.omega = self.reference_omega.copy()
        self.center = self.reference_center.copy()
        self.previous_gray = self.reference_gray.copy()

        self.positions = {
            name: uv.copy() for name, uv in self._seed_positions.items()
        }
        self.status = self._seed_status.copy()
        self.ages = {}
    
    def run(
        self,
        clip: Any,
        surface: StageSurfaces,
        max_frames: int | None = None,
        verbose: bool = True,
    ) -> list[dict[str, Any]]:
        """Continue from the current frame; return all accumulated records."""
        if max_frames is not None and max_frames < 0:
            raise ValueError("max_frames must be nonnegative")

        if clip.size() != self.frame_count:
            raise ValueError("Clip frame count differs from the saved run")
        if clip.frame_index_key(self.current_index) != self.current_frame_key:
            raise ValueError("Clip frame keys differ from the saved run")

        mser = create_mser() if self.replenish else None
        processed = 0
        index = self.current_index + self.step

        while 0 <= index < clip.size():
            if max_frames is not None and processed >= max_frames:
                break

            stem = clip.frame_index_key(index)
            gray = gray_image(clip.get_frame(index).broadcast_frame)
            if gray.shape != self.reference_gray.shape:
                raise ValueError(f"Image dimensions changed at {stem}")

            person = clip.get_person_mask(
                index, dilate=self.settings.DEFAULT_DILATE
            )
            if person is not None:
                person = np.asarray(person, dtype=bool)
                if person.shape != gray.shape:
                    raise ValueError(f"Person-mask dimensions differ at {stem}")

            active, provisional = self._flow(gray, person)

            solution = self._solve_pose(active)
            pose_solved = solution is not None
            if pose_solved:
                self._accept_pose(solution)

            errors_before = self._errors(active)
            promoted = self._promote_provisional(
                provisional, pose_solved
            )

            reacquired, reacquired_pose = self._reacquire(
                gray, person, active + promoted
            )
            pose_solved |= reacquired_pose

            # Do not manufacture new world points from an unverified pose.
            replenished = []
            if self.replenish and pose_solved:
                replenished = self._replenish(
                    clip, surface, index, gray, person, mser
                )

            self._record(
                stem,
                index,
                errors_before,
                reacquired,
                promoted,
                replenished,
                pose_solved,
            )

            self.previous_gray = gray
            self.current_index = index
            self.current_frame_key = stem

            if verbose:
                record = self.records[-1]
                print(
                    f"[{stem}] active={record['n_active']} "
                    f"provisional={record['n_provisional']} "
                    f"reacquired={len(reacquired)} "
                    f"pose_solved={pose_solved}"
                )

            processed += 1
            index += self.step

        return self.records
