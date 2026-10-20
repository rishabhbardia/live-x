
import numpy as np
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from data.types import (
    CameraIntrinsics, CameraPose, FighterPriors,
    Annotation, AnnotationPriors
)

from data.venue import (
    VenueMeasurements
)

from geometry.scene import Scene
from geometry.surface import StageSurfaces
from geometry.depth import DepthMetricFit

from tracker.tracker import Tracker


@dataclass(frozen=True, eq=False)
class FramePayload:
    broadcast_frame: np.ndarray
    inverse_depth: np.ndarray
    segmentation_masks: Mapping[Any, Any]
    body_poses: Any  # Tighten once the actual schema is known.

    metric_depth: np.ndarray | None = None
    camera_intrinsics: CameraIntrinsics | None = None
    camera_pose: CameraPose | None = None


@dataclass(eq=False)
class ClipPayload:
    fighter_priors: FighterPriors | None = None
    venue_measurements: VenueMeasurements | None = None
    depth_metric_fit: DepthMetricFit | None = None
    annotations: dict[str, Annotation] = field(default_factory=dict)
    annotation_priors: AnnotationPriors | None = None
    trajectory: list[CameraPose] = field(default_factory=list)
    intrinsics: list[CameraIntrinsics] = field(default_factory=list)
    calibrated_scene: Scene | None = None
    stage_surfaces: StageSurfaces | None = None
    tracker: Tracker | None = None


def organize_annotations(annotations: dict[str, Annotation], pillar_ids: list[int]):
    """
    Returns one dict. 

        "frames"      list[str], -- frame names (RGB broadcast), sorted
        "pillar_ids"  list[int], -- unique pillar ids seen in images, gated by input pillar_ids
        "track_ids"   list[int], -- unique point-track ids.
        "pil_idx"     int   (N_p, 2) -- [frame index into `frames`, post index into `pillar_ids`]
        "pil_base"    float (N_p, 2) -- [u, v] px, post base, row-parallel to `pil_idx`
        "pil_top"     float (N_p, 2) -- [u, v] px, post top, row-parallel to `pil_idx`
        "pt_idx"      int   (N_t, 2) -- [frame index into `frames`, track index into `track_ids`]
        "pt_uv"       float (N_t, 2) -- [u, v] px, row-parallel to `pt_idx`

    NOTE : frame index is of the annotated frames and not actual sequence index
    The four arrays are reshaped to (-1, 2), so an empty set is (0, 2) rather
    than (0,).
    """
    i = 0
    frames = []
    pil_idx, pil_base, pil_top, seen = [], [], [], []
    pt_idx, pt_uv = [], []
    tracks = []
    for key, ann in annotations.items():
        frames.append(key)
        assert len(ann.top_pillar_points) == len(ann.bottom_pillar_points)
        num_annotated_pillars = len(ann.top_pillar_points)
        # print(f"Key : {key}, annotated idx {i}, # tracked canvas points {len(ann.canvas_points)}, # tracked top pillar points {len(ann.top_pillar_points)}, # tracked bottom pillar points {len(ann.bottom_pillar_points)}")
        for pid in pillar_ids: # (selected ones 0, 2, 4 from annotation file)
            if pid >= num_annotated_pillars:
                continue
            if pid not in seen:
                seen.append(pid)
            for pt_top, pt_bottom in zip(ann.top_pillar_points, ann.bottom_pillar_points):
                if pt_top.idx == pid:
                    # print("Top : ", pt_top)
                    # print("Bottom : ", pt_bottom)
                    pil_idx.append([i, seen.index(pid)])
                    pil_base.append([pt_bottom.u, pt_bottom.v])
                    pil_top.append([pt_top.u, pt_top.v])
 
        for pt_canvas in ann.canvas_points:
                # print("Canvas : ", pt_canvas)
                if pt_canvas.idx not in tracks:
                    tracks.append(pt_canvas.idx)
                qi = tracks.index(pt_canvas.idx)
                pt_idx.append([i, qi])
                pt_uv.append([pt_canvas.u, pt_canvas.v])

        i += 1
    return {"frames": frames, "pillar_ids": seen, "track_ids": tracks,
            "pil_idx": np.asarray(pil_idx, int).reshape(-1, 2),
            "pil_base": np.asarray(pil_base, float).reshape(-1, 2),
            "pil_top": np.asarray(pil_top, float).reshape(-1, 2),
            "pt_idx": np.asarray(pt_idx, int).reshape(-1, 2),
            "pt_uv": np.asarray(pt_uv, float).reshape(-1, 2)}