import numpy as np
import cv2

from typing import Tuple
from collections.abc import Mapping
from pathlib import Path

from data.payloads import ClipPayload, FramePayload
from interface.deserializer import (
    load_annotation,
    load_annotation_priors,
    load_venue_measurements,
    load_image_file,
    load_npz_file,
    load_pickle_file,
    load_scene,
    load_tracker,
    load_depth_metric_fit
)
from interface.path_handler import PathHandler

class ClipBundle:
    """
    ClipBundle is the one interface for a clip that reads/writes all the data that is needed to run the pipeline. 
    
    """
    def __init__(self, directory: Path, force_recompute:bool=False):
        self.directory = Path(directory)
        self.force_recompute = force_recompute
        self.path_handler = PathHandler(self.directory)
        self.clip_payload = self._load_clip_data()
        self.segmentation_cache = {}
        

    def _load_clip_data(self) -> ClipPayload:
        payload = ClipPayload()
        path = self.path_handler.get_annotations_file()

        if path is not None:
            payload.annotations = dict(load_annotation(path))

        path = self.path_handler.get_annotations_prior_file()
        if path is not None:
            payload.annotation_priors = load_annotation_priors(path)

        path = self.path_handler.get_venue_file()
        if path is not None:
            payload.venue_measurements = load_venue_measurements(path)

        if self.force_recompute is False:
            path = self.path_handler.get_calibrated_stage_file()
            if path is not None:
                payload.calibrated_scene = load_scene(path)

            path = self.path_handler.get_tracker_file()
            if path is not None:
                payload.tracker = load_tracker(path)
                payload.tracker.sort_order()

            path = self.path_handler.get_depth_metric_file()
            if path is not None:
                payload.depth_metric_fit = load_depth_metric_fit(path)
        
        return payload

    def _load_frame_data(self, index: int) -> FramePayload:
        paths = self.path_handler

        image = load_image_file(paths.get_frame_path(index))
        depth = load_npz_file(paths.get_depth_path(index))

        if depth.shape != image.shape[:2]:
            raise ValueError(
                f"Frame {paths.frame_ids[index]}: "
                f"image size {image.shape[:2]} != depth size {depth.shape}"
            )

        segmentation = load_pickle_file(paths.get_segmentation_path(index))
        poses = load_pickle_file(paths.get_body_pose_path(index))

        if not isinstance(segmentation, Mapping):
            raise TypeError(
                f"Expected segmentation mapping, got "
                f"{type(segmentation).__name__}"
            )

        return FramePayload(
            broadcast_frame=image,
            inverse_depth=depth,
            segmentation_masks=segmentation,
            body_poses=poses,
        )

    def get_frame(self, frame_index: int) -> FramePayload:
        return self._load_frame_data(frame_index)

    def get_pose(self, frame_index: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.get_clip().tracker is None:
            raise ValueError("Tracker is not run")
        r = self.get_clip().tracker.sorted_records[frame_index]
        fk = self.frame_index_key(frame_index)
        assert r["stem"] == fk
        return np.asarray(r["omega"], np.float64), np.asarray(r["C"], np.float64)
    
    def get_clip(self) -> ClipPayload:
        return self.clip_payload

    def save_clip(self, clip: ClipPayload) -> None:
        raise NotImplementedError("Clip serialization is not implemented")

    def frame_key_index(self, key: str) -> int:
        # TODO : check bounds
        return self.path_handler.frame_key_to_index(key)

    def frame_index_key(self, idx: int) -> str:
        # TODO : check bounds
        return self.path_handler.frame_idx_key[idx]

    def size(self,) -> int:
        return self.path_handler.len_src_data()

    def get_person_mask(self, idx: int, dilate: int, cache_size: int = 6):

        if (idx, dilate) in self.segmentation_cache:
            return self.segmentation_cache[(idx, dilate)]
        
        k = np.ones((2 * dilate + 1,) * 2, np.uint8) if dilate > 0 else None
        m = None
        for person in self.get_frame(idx).body_poses:
            pm = person.get("mask")
            if pm is None:
                continue
            pm = np.squeeze(np.asarray(pm)) > 0
            m = pm if m is None else (m | pm)
        if m is None:
            seg = self.get_frame(idx).segmentation_masks
            if seg is not None:
                    masks = np.asarray(seg["out_binary_masks"])
                    if len(masks):
                        m = masks.any(axis=0)
        if m is not None and k is not None:
            m = cv2.dilate(m.astype(np.uint8), k) > 0
        if len(self.segmentation_cache) > cache_size:
            self.segmentation_cache.clear()
        self.segmentation_cache[(idx, dilate)] = m
        return m
