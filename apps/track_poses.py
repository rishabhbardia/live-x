import argparse
from interface.clip import ClipBundle
from interface.serializer import save_tracker
from pathlib import Path

from geometry.scene import Scene, build_landmarks_all, pair_arc_widths
from geometry.surface import StageSurfaces

from data.venue import VenueGeometryStructure
from data.payloads import organize_annotations
from data.types import TrackerMode

from visualization.key_frame_overlay import render_surfaces
from visualization.tracker_visuals import visualize_tracks

from tracker.tracker import Tracker

from interface.path_handler import TRACKER_FILE

import json

# /Users/rishabhbardia/Documents/workspace/vr-live-x/deploy/data/superbon/
# /Users/rishabhbardia/Documents/workspace/refactor/surfaces

if __name__ == "__main__":

    ap = argparse.ArgumentParser(description="Runs the tracker step of the pipeline")
    ap.add_argument("--clip_dir",  default=None, required=True, help="Base directory with data")
    ap.add_argument("--out_dir",  default=None, required=False, help="Uses clip_dir if not given")
    ap.add_argument("--recompute", action="store_true", help="Forces a recompute")
    args = ap.parse_args()

    verbose = True
    clip_path = Path(args.clip_dir)
    clip_bundle = ClipBundle(clip_path, args.recompute)

    frame_load = clip_bundle.get_frame(0)
    clip_load = clip_bundle.get_clip()

    scene: Scene = clip_load.calibrated_scene

    annotation_priors = clip_load.annotation_priors
    venue_measurements = clip_load.venue_measurements
    annotations = clip_load.annotations

    structure = VenueGeometryStructure(
            measurements=venue_measurements,
            priors=annotation_priors
    )
    r = structure.measurements.radius + float(scene.delta_radius)
    print(f"Computed radius was {r:.4f}")

    observations = organize_annotations(annotations=annotations, 
                                        pillar_ids=annotation_priors.pillar_ids)
    # For all the pillar ids
    all_ids = sorted({
        point.idx
        for annotation in annotations.values()
        for point in annotation.bottom_pillar_points
    })
    obs_all = organize_annotations(annotations=annotations, pillar_ids=all_ids)

    image_wh = (frame_load.broadcast_frame.shape[1], frame_load.broadcast_frame.shape[0])

    P_all, thetas_all = build_landmarks_all(
        scene=scene,
        observations=observations,
        structure=structure,
        obs_all=obs_all,
    )

    post_widths = pair_arc_widths(landmarks=P_all, 
                                  thetas=thetas_all, 
                                  pairs={(0,1), (2,3), (4,5)})
    mean_width = 0
    for p, w in post_widths.items():
        mean_width += w
    mean_width /= float(len(post_widths))

    print(f"Post width {mean_width}")


    stage_surface = StageSurfaces(
        clip=clip_bundle,
        scene=scene,
        tracked_points=P_all,
        post_thetas=thetas_all,
        post_widths=post_widths,
        post_half_width=mean_width/2.0,
        image_wh=image_wh
    )
    visualize = True
    out_dir = Path(args.out_dir) if args.out_dir is not None else Path(args.clip_dir)
    if visualize:
        render_surfaces(
            surf=stage_surface,
            observations=observations,
            clip=clip_bundle,
            out_dir=out_dir / "surfaces"
        )

    tracker_mode = TrackerMode.FIXED_CAMERA_CENTER

    force_retrack = False
    p = clip_path / TRACKER_FILE

    if not force_retrack and clip_load.tracker is not None:
        print("Use loaded the tracker")
        data = clip_load.tracker.to_dict()
        p0 = out_dir / "tracker/poses.json"
        p0.parent.mkdir(parents=True, exist_ok=True)
        with open(p0, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=4)
        if visualize:
            visualize_tracks(
                clip=clip_bundle,
                tracker=clip_load.tracker,
                output_dir=out_dir / "tracker"
            )
    else:
        tracker = Tracker(
            clip=clip_bundle,
            surface=stage_surface,
            scene=scene,
            observations=observations,
            landmarks=P_all,
            start_annotation_frame_idx=len(observations["frames"]) - 1,
        )

        for step, id_offset in ((1, 0), (-1, 100000)):
            tracker.start_pass(step=step, id_offset=id_offset)
            tracker.run(
                clip=clip_bundle,
                surface=stage_surface,
                max_frames=None,
            )

        save_tracker(path=p, tracker=tracker)

        records_both_way = sorted(
            tracker.records,
            key=lambda record: record["frame_index"],
        )
        all_points = tracker.landmarks
  


    