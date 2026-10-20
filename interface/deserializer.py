import json
import pickle
import cv2
import numpy as np

from pathlib import Path
from typing import Any, Mapping

from utils.freeze import freeze_array, freeze_data
from data.types import (
    Annotation, Point, AnnotationPriors) 
from data.venue import VenueMeasurements
from geometry.scene import Scene
from geometry.depth import DepthMetricFit

from tracker.tracker import Tracker


def load_scene(path: Path) -> Scene:
    """Load a Scene from a trusted pickle file."""
    with Path(path).open("rb") as fh:
        scene = pickle.load(fh)

    if not isinstance(scene, Scene):
        raise TypeError(f"Expected Scene, got {type(scene).__name__}")

    return scene

def _load_one_fc_pillar(data: Any) -> tuple[list[Point], list[Point]]:
    """
        While annotating we made sure that each frame's top and bottom
        annotations were synchronized ordering
    """
    base = []
    top = []
    counter_base = 0
    counter_top = 0
    for d in data:
        pt_top = d["top"]
        pt_base = d["base"]
        pt0 = Point(counter_base, pt_base[0], pt_base[1])
        pt1 = Point(counter_top, pt_top[0], pt_top[1])
        base.append(pt0)
        top.append(pt1)
        counter_base += 1
        counter_top += 1
    return base, top

def _load_one_fc_points(data: Any) -> list[Point]:
    points = []
    for idx, dl in data.items():
        pt0 = Point(int(idx), dl[0], dl[1])
        points.append(pt0)
    return points
    
def _load_annotation_priors(data: Any) -> AnnotationPriors:
    return AnnotationPriors(
        pillar_base_sigma_px=data["sigmas_px"]["pillar_base"],
        pillar_top_sigma_px=data["sigmas_px"]["pillar_top"],
        landmark_sigma_px=data["sigmas_px"]["landmark"],
        principal_point_sigma_px=data["sigmas_px"]["principal_point"],
        delta_R_m=data["priors_m"]["dR"],
        delta_H_m=data["priors_m"]["dH"],
        pillar_ids=data["pillar_ids"],
    )
    
def load_annotation_priors(path: Path) -> AnnotationPriors:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return _load_annotation_priors(data=data)


def load_annotation(path: Path) -> Mapping[str, Annotation]:
    annotation_load = {}

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    for frame_name, structures in data.items():
        frame_idx = int(frame_name.removeprefix("frame_"))
        annotation = Annotation(frame_idx, frame_name)

        canvas_pts: list[Point] = []
        top_pts: list[Point] = []
        bottom_pts: list[Point] = []

        for structure_type, structure_data in structures.items():
            if structure_type == "pillars": # TODO : move this to constants
                bottom_pts, top_pts = _load_one_fc_pillar(structure_data)
            elif structure_type == "points": # TODO : move this to constants
                canvas_pts = _load_one_fc_points(structure_data)

        annotation.add_annotations(canvas_pts, top_pts, bottom_pts)
        annotation_load[frame_name] = annotation

    return annotation_load



def load_depth_metric_fit(path: Path) -> DepthMetricFit:
    with Path(path).open("rb") as fh:
        fit = pickle.load(fh)

    if not isinstance(fit, DepthMetricFit):
        raise TypeError(
            f"Expected DepthMetricFit, got {type(fit).__name__}"
        )

    return fit

def load_venue_measurements(path: Path) -> VenueMeasurements:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return VenueMeasurements(
        diameter=data["cage"]["size_m"],
        pillar_height=data["cage"]["post_height_m"],
        shape=data["cage"]["shape"],
        total_posts=data["cage"]["n_posts"]
    )


def load_npz_file(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        depth = data["depths"]

    if depth.ndim != 2:
        raise ValueError(
            f"Expected one (H, W) depth frame in {path}, got {depth.shape}"
        )

    return freeze_array(depth)


def load_image_file(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise OSError(f"Could not read image: {path}")

    # Preserves OpenCV's existing BGR/BGRA/grayscale convention.
    return freeze_array(image)


def load_pickle_file(path: Path) -> Any:
    with path.open("rb") as fh:
        data = pickle.load(fh)

    return freeze_data(data)


def load_tracker(path: Path) -> Tracker:
    with Path(path).open("rb") as stream:
        payload = pickle.load(stream)

    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("Unsupported tracker file")

    tracker = payload.get("tracker")
    if not isinstance(tracker, Tracker):
        raise TypeError("File does not contain a Tracker")

    return tracker