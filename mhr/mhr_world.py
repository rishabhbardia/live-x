from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any, Dict

import cv2
import numpy as np

from data.types import BodyPart
from geometry.pose import rodrigues
from mhr.constants import (
    MHR_MODEL_NPZ,
    MHR_PARTS_NPZ,
    MHR_FACES_KEY,
    ERODE_PX,
    MIN_PTS,
    JOINT_HEAD,
    JOINT_NECK
)


class MHRWorld:
    """Place predicted MHR bodies into the calibrated world coordinate system.

    Placement separates image alignment from metric scale:

        X_camera = scale * (vertices + tau)
        translation = scale * tau

    1. Solve body-unit translation tau from 3D/2D keypoint correspondences.
       Scale cancels from projection.
    2. Estimate scale from metric camera-Z depth at projected mesh vertices.
    3. Transform the placed vertices into world coordinates:

        X_world = X_camera @ R + C

    Assumes predicted vertices and 3D keypoints share the same body coordinate
    system, already oriented consistently with the camera axes.

    Visibility uses the nearest projected vertex per pixel. This is an
    approximation to triangle rasterization, not a complete mesh z-buffer.
    """

    def __init__(
        self,
        faces: np.ndarray | None = None,
        model_path: Path = MHR_MODEL_NPZ,
        erode_px: int = ERODE_PX,
        min_pts: int = MIN_PTS,
    ):
        if erode_px < 0:
            raise ValueError("erode_px must be nonnegative")
        if min_pts < 1:
            raise ValueError("min_pts must be positive")

        if faces is None:
            with np.load(model_path, allow_pickle=False) as data:
                faces = data[MHR_FACES_KEY]

        faces = np.asarray(faces)
        self.path_base = Path(model_path).parent / MHR_PARTS_NPZ
        self.mhr_parts = np.load(self.path_base, allow_pickle=False) 
            
        print(f"Shape faces {faces.shape}")
        if (
            faces.ndim != 2
            or faces.shape[1] != 3
            or not np.issubdtype(faces.dtype, np.integer)
            or np.any(faces < 0)
        ):
            raise ValueError("faces must contain nonnegative integer triangles")

        self.faces = faces.astype(np.int32, copy=True)
        self.faces.setflags(write=False)
        self.erode_px = erode_px
        self.min_pts = min_pts

    @staticmethod
    def _vertex_normals(vertices, faces):
        face_normals = np.cross(
            vertices[faces[:, 1]] - vertices[faces[:, 0]],
            vertices[faces[:, 2]] - vertices[faces[:, 0]],
        )
        normals = np.zeros_like(vertices)

        for corner in range(3):
            np.add.at(normals, faces[:, corner], face_normals)

        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        return normals / np.maximum(lengths, 1e-12)

    @staticmethod
    def _intrinsics(K):
        K = np.asarray(K, dtype=np.float64)
        if K.shape != (3, 3) or not np.isfinite(K).all():
            raise ValueError("K must be a finite (3, 3) matrix")
        if K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError("Focal lengths must be positive")
        if not np.allclose(K[2], [0, 0, 1]) or not np.allclose(
            [K[0, 1], K[1, 0]], 0
        ):
            raise ValueError("Expected pinhole K without skew")

        return K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    @staticmethod
    def _project(points, intrinsics):
        fx, fy, cx, cy = intrinsics
        return np.column_stack((
            fx * points[:, 0] / points[:, 2] + cx,
            fy * points[:, 1] / points[:, 2] + cy,
        ))

    def solve_tau(
        self,
        joints,
        keypoints,
        K,
        tau0,
        iterations: int = 25,
        damping: float = 1e-6,
    ):
        """
        Solve body-unit translation tau from 2D/3D keypoints using Damped Gauss–Newton
        with three translation DOFs.

        Writing metric translation as t = s * tau makes scale cancel in projection:

            u = f * (s * Jx + s * tau_x) / (s * Jz + s * tau_z) + cx
             = f * (Jx + tau_x) / (Jz + tau_z) + cx

        Thus keypoints determine tau, while metric depth subsequently determines s.
        An object twice as large at twice the distance has the same projection.

        tau_x and tau_y control image position; tau_z controls apparent size.

        Assumes the predicted 3D keypoints are already oriented in camera axes.
        Using our calibrated K instead of the prediction model's camera may leave
        reprojection errors that translation alone cannot resolve.

        Returns (tau, median_reprojection_error_px).
        """
        intrinsics = self._intrinsics(K)
        fx, fy, cx, cy = intrinsics

        joints = np.asarray(joints, dtype=np.float64)
        keypoints = np.asarray(keypoints, dtype=np.float64)
        tau = np.asarray(tau0, dtype=np.float64).reshape(3).copy()

        if joints.ndim != 2 or joints.shape[1] != 3:
            raise ValueError("joints must have shape (N, 3)")
        if keypoints.shape != (len(joints), 2):
            raise ValueError("keypoints must have shape (N, 2)")
        if not np.isfinite(tau).all():
            return tau, float("nan")

        finite = (
            np.isfinite(joints).all(axis=1)
            & np.isfinite(keypoints).all(axis=1)
        )
        joints = joints[finite]
        keypoints = keypoints[finite]

        for _ in range(iterations):
            camera_points = joints + tau
            valid = camera_points[:, 2] > 1e-6
            if valid.sum() < 4:
                return tau, float("nan")

            points = camera_points[valid]
            predicted = self._project(points, intrinsics)
            errors = predicted - keypoints[valid]
            depth = points[:, 2]
            zeros = np.zeros(len(points))

            Ju = np.column_stack((
                fx / depth,
                zeros,
                -(predicted[:, 0] - cx) / depth,
            ))
            Jv = np.column_stack((
                zeros,
                fy / depth,
                -(predicted[:, 1] - cy) / depth,
            ))
            A = np.vstack((Ju, Jv))
            residual = np.concatenate((errors[:, 0], errors[:, 1]))

            normal = A.T @ A
            normal += (
                damping
                * max(float(np.trace(normal)), 1.0)
                * np.eye(3)
            )

            try:
                step = np.linalg.solve(normal, -A.T @ residual)
            except np.linalg.LinAlgError:
                return tau, float("nan")

            if not np.isfinite(step).all():
                return tau, float("nan")

            tau += step
            if np.linalg.norm(step) < 1e-9:
                break

        camera_points = joints + tau
        valid = camera_points[:, 2] > 1e-6
        if valid.sum() < 4:
            return tau, float("nan")

        predicted = self._project(camera_points[valid], intrinsics)
        errors = np.linalg.norm(predicted - keypoints[valid], axis=1)
        return tau, float(np.median(errors))

    def place_by_depth(
        self,
        person: Mapping[str, Any],
        K: np.ndarray,
        metric_depth: np.ndarray,
        target_height: float | None = None,
    ):
        """Return (scale, camera_translation, info), or (None, None, info).

        metric_depth must contain camera-space Z in metres, aligned with
        the person's mask and the image coordinates used by K/keypoints.

        target_height preserves the original optional override: scale the
        mesh's Y extent to this value. For a posed mesh, Y extent is not
        generally anatomical standing height.
        """
        info = {
            "placed": False,
            "n_anchor": 0,
            "reason": "",
            "scale": None,
            "tau_reproj_px": float("nan"),
        }

        def failed(reason):
            info["reason"] = reason
            return None, None, info

        required = (
            "pred_vertices",
            "mask",
            "pred_keypoints_3d",
            "pred_keypoints_2d",
            "pred_cam_t",
        )
        missing = [key for key in required if key not in person]
        if missing:
            return failed(f"missing fields: {', '.join(missing)}")

        vertices = np.asarray(person["pred_vertices"], dtype=np.float64)
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or not len(vertices)
            or not np.isfinite(vertices).all()
        ):
            return failed("invalid predicted vertices")

        if self.faces.size and self.faces.max() >= len(vertices):
            return failed("mesh topology does not match predicted vertices")

        # Accept binary masks represented as bool, 0/1, or 0/255.
        mask = np.squeeze(np.asarray(person["mask"])) > 0
        depth = np.asarray(metric_depth, dtype=np.float64)

        if mask.ndim != 2 or depth.ndim != 2:
            raise ValueError("mask and metric_depth must be 2-D")
        if depth.shape != mask.shape:
            raise ValueError(
                f"Depth shape {depth.shape} != mask shape {mask.shape}; "
                "align inputs and intrinsics before placement"
            )

        interior = mask
        if self.erode_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.erode_px, self.erode_px),
            )
            interior = cv2.erode(mask.astype(np.uint8), kernel) > 0

        if interior.sum() < self.min_pts:
            return failed("mask interior too small")

        intrinsics = self._intrinsics(K)
        tau, error = self.solve_tau(
            person["pred_keypoints_3d"],
            person["pred_keypoints_2d"],
            K,
            person["pred_cam_t"],
        )
        info["tau_reproj_px"] = error

        if not np.isfinite(error):
            return failed("keypoint translation solve failed")

        camera_points = vertices + tau
        normals = self._vertex_normals(vertices, self.faces)

        # Perspective-facing test; assumes outward mesh winding.
        front = (
            (np.einsum("ij,ij->i", normals, camera_points) < 0)
            & (camera_points[:, 2] > 1e-6)
        )
        indices = np.flatnonzero(front)
        info["n_front"] = len(indices)

        if not len(indices):
            return failed("no front-facing vertices")

        pixels = self._project(camera_points[indices], intrinsics)
        h, w = depth.shape

        finite = np.isfinite(pixels).all(axis=1)
        indices = indices[finite]
        pixels = np.rint(pixels[finite])

        inside = (
            (pixels[:, 0] >= 0) & (pixels[:, 0] < w)
            & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
        )
        indices = indices[inside]
        pixels = pixels[inside].astype(np.int64)

        hit = interior[pixels[:, 1], pixels[:, 0]]
        indices = indices[hit]
        pixels = pixels[hit]

        if len(indices) < self.min_pts:
            return failed("too few vertices on mask interior")

        # Keep one nearest projected vertex per pixel.
        pixel_ids = pixels[:, 1] * w + pixels[:, 0]
        body_depths = camera_points[indices, 2]
        order = np.lexsort((body_depths, pixel_ids))

        sorted_ids = pixel_ids[order]
        first = np.r_[True, sorted_ids[1:] != sorted_ids[:-1]]
        visible = order[first]

        pixels = pixels[visible]
        body_depths = body_depths[visible]
        observed_depths = depth[pixels[:, 1], pixels[:, 0]]

        info["n_visible"] = len(visible)

        valid = (
            np.isfinite(observed_depths)
            & (observed_depths > 0)
            & np.isfinite(body_depths)
            & (body_depths > 1e-6)
        )
        if valid.sum() < self.min_pts:
            return failed("too few valid depth anchors")

        ratios = observed_depths[valid] / body_depths[valid]
        scale = float(np.median(ratios))

        if not np.isfinite(scale) or scale <= 0:
            return failed("degenerate scale")

        depth_scale = scale
        y_extent = float(np.ptp(vertices[:, 1]))

        if target_height is not None:
            if not np.isfinite(target_height) or target_height <= 0:
                raise ValueError("target_height must be finite and positive")
            if y_extent <= 1e-9:
                return failed("degenerate mesh Y extent")
            scale = float(target_height) / y_extent

        info.update(
            placed=True,
            n_anchor=int(valid.sum()),
            scale=scale,
            depth_scale=depth_scale,
            scale_source="depth" if target_height is None else "y_extent",
            scale_iqr=float(np.subtract(*np.percentile(ratios, [75, 25]))),
            implied_height_m=scale * y_extent,
        )
        return scale, scale * tau, info

    
    @staticmethod
    def _unit(vector, eps=1e-12):
        vector = np.asarray(vector, dtype=np.float64)
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm <= eps:
            raise ValueError("Cannot normalize a nonfinite or near-zero vector")
        return vector / norm

    def _pca_axis(self, points):
        centered = points - points.mean(axis=0)
        values, vectors = np.linalg.eigh(centered.T @ centered)
        if values[-1] <= 0:
            raise ValueError("Cannot find an axis for coincident points")
        return vectors[:, -1]

    def compute_ends(self, points, fraction=0.12, *, axis=None):
        """Mean positions in the two end caps selected along an axis.

        With axis=None, use the dominant PCA axis.
        """
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
            raise ValueError("Expected at least two points with shape (N, 3)")
        if not np.isfinite(points).all():
            raise ValueError("Points must be finite")
        if not 0 < fraction < 0.5:
            raise ValueError("fraction must be between 0 and 0.5")

        axis = self._pca_axis(points) if axis is None else self._unit(axis)
        projected = (points - points.mean(axis=0)) @ axis
        lo, hi = np.quantile(projected, [fraction, 1.0 - fraction])

        return (
            points[projected <= lo].mean(axis=0),
            points[projected >= hi].mean(axis=0),
        )

    def height_check(
        self,
        world_vertices: np.ndarray,
        joints: np.ndarray,
        info: Dict[str, Any],
        target_height: float,
    ):
        """Compare a sum of body-part lengths with known height in metres.

        world_vertices are already placed by build(); joints are the original
        pred_keypoints_3d values. This is a pose-dependent height proxy.
        """
        vertices = np.asarray(world_vertices, dtype=np.float64)
        joints = np.asarray(joints, dtype=np.float64)
        names = list(self.mhr_parts["names"])
        labels = np.asarray(self.mhr_parts["vertex_labels"])

        if labels.shape != (len(vertices),):
            raise ValueError("Part labels must match this person's vertices")
        if not np.isfinite(target_height) or target_height <= 0:
            raise ValueError("target_height must be finite and positive")

        def part(*body_parts):
            indices = [names.index(name) for name in body_parts]
            return vertices[np.isin(labels, indices)]

        # Translation cancels in the difference; normalization removes scale.
        head_axis = (joints[JOINT_HEAD] - joints[JOINT_NECK]) @ info["R"]

        segments = {
            "head_neck": (part(BodyPart.HEAD, BodyPart.NECK), head_axis),
            "torso": (part(BodyPart.TORSO), None),
            "left_thigh": (part(BodyPart.LEFT_THIGH), None),
            "left_shin": (part(BodyPart.LEFT_SHIN), None),
            "left_foot": (part(BodyPart.LEFT_FOOT), [0.0, 0.0, 1.0]),
        }

        lengths = {}
        for name, (points, axis) in segments.items():
            lo, hi = self.compute_ends(points, axis=axis)
            if "foot" in name:
                lengths[name] =  abs(hi[2] - lo[2])
            else:
                lengths[name] = float(np.linalg.norm(hi - lo))
        height = sum(lengths.values())
        error = height - float(target_height)

        return {
            "height_m": height,
            "target_height_m": float(target_height),
            "error_m": error,
            "segment_lengths_m": lengths,
        }
    
    def build(
        self,
        metric_depth: np.ndarray,
        persons: Sequence[Mapping[str, Any]],
        K: np.ndarray,
        omega: np.ndarray,
        C: np.ndarray,
    ):
        """Return combined (vertices_world, faces, per-person diagnostics).

        Returns (None, None, diagnostics) if no body could be placed.
        """
        if metric_depth is None:
            return None, None, []

        R = rodrigues(np.asarray(omega, dtype=np.float64))
        center = np.asarray(C, dtype=np.float64).reshape(3)

        vertices_out = {}
        faces_out = {}
        diagnostics = {}
        offset = 0

        for index, person in enumerate(persons):
            scale, translation, info = self.place_by_depth(
                person, K, metric_depth
            )
            info["person_index"] = index
            if scale is None:
                continue

            vertices = np.asarray(
                person["pred_vertices"], dtype=np.float64
            )
            camera_vertices = scale * vertices + translation
            world_vertices = camera_vertices @ R + center

            info["vertex_start"] = offset
            info["vertex_count"] = len(vertices)
            info["translation"] = translation
            info["scale"] = scale
            info["R"] = R
            info["C"] = C

            diagnostics[index] = info
            vertices_out[index] = world_vertices
            faces_out[index] = self.faces + offset
            offset += len(vertices)

        if not vertices_out:
            return None, None, diagnostics

        return (
            vertices_out,
            faces_out,
            diagnostics,
        )