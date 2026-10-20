from interface.clip import ClipBundle
from interface.serializer import save_depth_metric_fit
from interface.path_handler import DEPTH_METRIC_FILE

from pathlib import Path

from geometry.scene import Scene, build_landmarks_all, pair_arc_widths
from geometry.surface import StageSurfaces

from data.venue import VenueGeometryStructure
from data.payloads import organize_annotations

from geometry.depth import DepthMetricFit

import argparse

# "/Users/rishabhbardia/Documents/workspace/vr-live-x/deploy/data/superbon/"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Runs the metric depth part of the pipeline")
    ap.add_argument("--clip_dir", type=str, default=None, required=True, help="Base directory with data")
    ap.add_argument("--recompute", action="store_true", help="Forces a recompute")
    args = ap.parse_args()
    
    verbose = True
    clip_path = Path(args.clip_dir)
    clip_bundle = ClipBundle(clip_path, args.recompute)
    clip_load = clip_bundle.get_clip()
    frame_load = clip_bundle.get_frame(0)

    if clip_load.calibrated_scene is not None:
        print("Scene was calibrated on stage")

    annotation_priors = clip_load.annotation_priors
    venue_measurements = clip_load.venue_measurements
    annotations = clip_load.annotations

    structure = VenueGeometryStructure(
            measurements=venue_measurements,
            priors=annotation_priors
    )

    r = structure.measurements.radius + float(clip_load.calibrated_scene.delta_radius)
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
        scene=clip_load.calibrated_scene,
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
        scene=clip_load.calibrated_scene,
        tracked_points=P_all,
        post_thetas=thetas_all,
        post_widths=post_widths,
        post_half_width=mean_width/2.0,
        image_wh=image_wh
    )

    print("StageSurfaces Reconstructed")

    if clip_load.tracker is not None:
        print("Tracker was already run")

   
    depth_fitter = DepthMetricFit(num_frames=clip_bundle.size())

    depth_fitter.run(clip=clip_bundle, 
                     surface=stage_surface, 
                     tracker=clip_load.tracker, 
                     subsampling=-1)

    save_depth_metric_fit(
        path= clip_path / DEPTH_METRIC_FILE,
        fit=depth_fitter,
    )