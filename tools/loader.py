from interface.clip import ClipBundle
from data.venue import VenueGeometryStructure
from data.payloads import organize_annotations

from pathlib import Path


if __name__ == "__main__":

    verbose = True
    clip_path = Path("/Users/rishabhbardia/Documents/workspace/vr-live-x/deploy/data/superbon/")
    clip_bundle = ClipBundle(clip_path)

    frame_load = clip_bundle.get_frame(0)
    clip_load = clip_bundle.get_clip()

    annotations = clip_load.annotations
    annotation_priors = clip_load.annotation_priors
    venue_measurements = clip_load.venue_measurements

    structure = VenueGeometryStructure(
         measurements=venue_measurements,
         priors=annotation_priors
    )

    observations = organize_annotations(annotations=annotations, 
                                           pillar_ids=annotation_priors.pillar_ids)


    inverse_depth = frame_load.inverse_depth
    rgb_frame = frame_load.broadcast_frame
    segmentation_masks = frame_load.segmentation_masks
    body_poses = frame_load.body_poses

    for i, d in segmentation_masks.items():
        print(type(i), type(d))

    for p in body_poses:
        print(type(p))

    for i in range(clip_bundle.size()):
        f = clip_bundle.get_frame(i)
        print(f"Frame {i}")