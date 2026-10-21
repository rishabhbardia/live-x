from interface.clip import ClipBundle
from pathlib import Path

from rendering.renderer import Renderer, make_view, View, stereo_views
from rendering.exporter import SBSVideoExporter

from mhr.mhr_world import MHRWorld

from geometry.depth import corrected_depth

import cv2
import json
import numpy as np

import argparse

def load_view(path, scale=1.0):
    """
        Loads the target view for the seat.
    """
    with open(path) as fh:
        d = json.load(fh)
    K = np.asarray(d["K"], np.float64).copy()
    K[:2] *= scale
    return View(K, np.asarray(d["R"], np.float64),
                np.asarray(d["C"], np.float64),
                (int(round(d["width"] * scale)),
                 int(round(d["height"] * scale)))), d.get("note", "")

# These are not measured exact these are tale of the tapes
# which every fighter knows is a big ass lie that your
# promotoer tells. Use this as a rough check
HEIGHT_CHECKS = {
     0 : 1.78, # I don't know refree's height, I dont know his name
     1 : 1.78, # Petrosyan 
     2 : 1.77, # Superbon
}

if __name__ == "__main__":

    ap = argparse.ArgumentParser(description="Runs the rendering part of the pipeline")
    ap.add_argument("--clip_dir", type=str, default=None, required=True, help="Base directory with data")
    ap.add_argument("--src_video", type=str, default=None, required=True, help="Original broadcast video")
    ap.add_argument("--tgt_view", type=str, default=None, required=True, help="Novel view")
    ap.add_argument("--out_dir", type=str, default=None, required=False, help="Uses clip_dir if not given")
    ap.add_argument("--mhr_model", type=str, default=None, required=False, help="MHR npz file")
    ap.add_argument("--recompute", action="store_true", help="Forces a recompute")
    args = ap.parse_args()

    source_video_path = args.src_video
    verbose = True
    clip_path = Path(args.clip_dir)
    clip_bundle = ClipBundle(clip_path, args.recompute)

    frame_load = clip_bundle.get_frame(0)
    clip_load = clip_bundle.get_clip()

    # Optional MHR, this is a validation check right now
    # but eventually we will build the body texture atlas
    # to cover the missing data in frame that is present in
    # the sequence
    mhr_placer = MHRWorld(model_path=args.mhr_model)

    if clip_load.calibrated_scene is not None:
        print("Scene was calibrated on stage")

    if clip_load.tracker is not None:
        print("Tracker was already run")

    if clip_load.depth_metric_fit is not None:
            print("Metric fitting was already run")

    tgt_view = args.tgt_view
    view_input, _ = load_view(tgt_view)
    view_render = make_view(
        projection="equirect",
        R=view_input.R,
        C=view_input.C,
        size=2048,
    )

    left_view, right_view = stereo_views(
        R=view_input.R,
        C=view_input.C,
        ipd=0.064,       # Eye separation in metres; configurable.
        size=2048,       # Per-eye resolution.
        fov=180.0,
        projection="equirect",
    )

    left_renderer = Renderer(left_view, world_radius=0.012, max_px=6)
    right_renderer = Renderer(right_view, world_radius=0.012, max_px=6)

    out_dir = Path(args.out_dir) if args.out_dir is not None else Path(args.clip_dir)
    out = out_dir / "output_stereo"
    out.mkdir(parents=True, exist_ok=True)

    p = np.zeros(3)
    p[0] = clip_load.depth_metric_fit.alpha
    p[1] = clip_load.depth_metric_fit.beta
    p[2] = clip_load.depth_metric_fit.gamma
    b_t = clip_load.depth_metric_fit.beta_t
    source_frame_indices = []
    do_height_check = True
    heights_accum = {}
    # Each item supplies one source frame and its calibrated camera data.
    for i in range(0, clip_bundle.size()):
        frame_load = clip_bundle.get_frame(i)
        
        f, cx, cy = clip_load.calibrated_scene.intr_of(i)
        K = np.array(
                [[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]],
                    dtype=np.float64,
        )
        
        omega, C = clip_bundle.get_pose(i)
        
        person_mask = clip_bundle.get_person_mask(i, 0)
        image_bgr = frame_load.broadcast_frame
        inverse_depth = frame_load.inverse_depth
        metric_depth = corrected_depth(inverse_depth, p, b_t[i])

        vertices_world, faces, diag = mhr_placer.build(metric_depth=metric_depth, 
                                                       persons=frame_load.body_poses, 
                                                       K=K, 
                                                       omega=omega, 
                                                       C=C)
        if do_height_check:
            print(f"Len vertices {len(vertices_world)}")
            print(f"Shape vertices {vertices_world[0].shape}")
            for pidx, mhr_d in vertices_world.items():
                heights = mhr_placer.height_check(mhr_d, 
                                        frame_load.body_poses[pidx]["pred_joint_coords"],
                                        diag[pidx],
                                        HEIGHT_CHECKS[pidx])
                if pidx not in heights_accum:
                    heights_accum[pidx] = []
                else:
                    heights_accum[pidx].append(heights["height_m"])
            
        
        points, rgb, stats = left_renderer.fighter_cloud(
            person_mask=person_mask,
            image=image_bgr,
            metric_depth=metric_depth,
            K=K,
            omega=omega,
            C=C,
        )


        if points is None:
            points = np.empty((0, 3))
            rgb = np.empty((0, 3))

        left_rgb, left_depth, left_coverage = left_renderer.render(points, rgb)
        right_rgb, right_depth, right_coverage = right_renderer.render(points, rgb)
        print(f"Rendered image {i}")

        sbs_rgb = np.hstack([left_rgb, right_rgb])
        sbs_u8 = np.round(np.clip(sbs_rgb, 0.0, 1.0) * 255).astype(np.uint8)
        sbs_bgr = cv2.cvtColor(sbs_u8, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(out / f"{i:05d}.png"), sbs_bgr)
        source_frame_indices.append(i)

    for pidx, hdata in heights_accum.items():
        print(f"{pidx} Mean height {np.mean(hdata)} and std {np.std(hdata)}")
        
    exporter = SBSVideoExporter(out / "render.mp4")
    exporter.export(source_video_path, out)

    # Optional: indices must refer to the source video, not renumbered filenames.
    ok, message = exporter.mux_audio(
        source_video=source_video_path,
        fps=cv2.VideoCapture(source_video_path).get(cv2.CAP_PROP_FPS) or 25.0,
        frame_indices=source_frame_indices,
    )
    print(message)

    vr_path, error = exporter.inject_spatial_metadata()
