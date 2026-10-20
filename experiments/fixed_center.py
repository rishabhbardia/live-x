from interface.clip import ClipBundle
from data.venue import VenueGeometryStructure
from data.payloads import organize_annotations
from data.types import SolverParameters, CameraLock, AnnotationPriors
from data.venue import VenueMeasurements


from geometry.scene import Scene
from geometry.solver import staged_solve, build_problem, solve
from geometry.sensitivity import report, jacobian_analysis, parameter_uncertainty

from pathlib import Path
from typing import Dict, Any

import numpy as np
import argparse

def fixed_center_test(scene: Scene, 
                      observations: Dict[str, Any], 
                      image_wh: tuple[int,int], 
                      priors: AnnotationPriors, 
                      venue_measurements: VenueMeasurements, 
                      params: SolverParameters):
    """
        This is an experiment to signal if fixing the camera center was the right call.
        It frees one camera center at a time against the frame's observations, which
        measures how well each frame pins the center.

        This is differnt from CameraLock.MOVED which frees the camera center in the
        parameter block for the whole tracjectory for the joint solve.

        This comes from the observation that the broadcast shot for this test instance
        was a simple rotation on the camera's optical axis to cover the highlight moment.
    """
    
    out = []
    for t in range(len(observations["frames"])):
        sel = observations["pil_idx"][:, 0] == t
        selq = observations["pt_idx"][:, 0] == t
        sub = {"frames": [observations["frames"][t]], "pillar_ids": observations["pillar_ids"],
               "track_ids": observations["track_ids"],
               "pil_idx": np.c_[np.zeros(sel.sum(), int), observations["pil_idx"][sel, 1]],
               "pil_base": observations["pil_base"][sel], "pil_top": observations["pil_top"][sel],
               "pt_idx": np.c_[np.zeros(selq.sum(), int), observations["pt_idx"][selq, 1]],
               "pt_uv": observations["pt_uv"][selq]}

        s1 = Scene(
            n_frames=1,
            n_pillars=len(observations["pillar_ids"]),
            n_tracks=len(observations["track_ids"])
        )
     
        f, cx, cy = scene.intr_of(t)
        s1.set_intr(f, cx, cy)
        s1.set_center(scene.camera_center_of(t))
        s1.delta_radius[:] = scene.delta_radius
        s1.delta_height[:] = scene.delta_height

        for i, th in enumerate(scene.theta):
            s1.theta[i][:] = th
        s1.omega[0][:] = scene.omega[t]
        for j, q in enumerate(scene.tracked_points):
            s1.tracked_points[j][:] = q
        p, k = build_problem(scene=s1, 
                             observations=sub, 
                             image_wh=image_wh, 
                             use_points=True, 
                             free=set({"center", "omega"}),
                             priors=annotation_priors,
                             venue_measurements=venue_measurements,
                             params=params
        )
        
        solve(prob=p, max_iter=200)
        out.append(s1.C - scene.camera_center_of(t))
    return np.array(out)

if __name__ == "__main__":

    verbose = True
    ap = argparse.ArgumentParser(description="Runs the calibration step of the pipeline")
    ap.add_argument("--clip_dir", type=str, default=None, required=True, help="Base directory with data")
    args = ap.parse_args()
    clip_path = Path(args.clip_dir)
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

    rgb_frame = frame_load.broadcast_frame
    params = SolverParameters()
    image_wh = (rgb_frame.shape[1], rgb_frame.shape[0])
    focal_grid = [r * image_wh[0] for r in params.focal_grid_rel]
    
    use_delta = True
    use_dh = False
    lock = CameraLock.ROT
    f_ratios = None # for stage D we need this

    scene, problem, keep, summary = staged_solve(
         params=params,
         focal_grid=focal_grid,
         observations=observations,
         structure=structure,
         venue_measurements=venue_measurements,
         annotation_priors=annotation_priors,
         image_wh=image_wh,
         use_delta=use_delta,
         use_dh=use_dh,
         verbose=True,
         lock=lock,
         f_ratios=f_ratios
    )

    print(report(
         scene=scene,
         observations=observations,
         image_wh=image_wh,
         structure=structure
    ))


    jacobian_analysis(
          problem=problem,
          scene=scene,
          observations=observations
    )

    sigmas, chi2 = parameter_uncertainty(
          problem=problem,
          summary=summary,
          scene=scene,
          observation=observations
    )

    # fixed center test
    eps = fixed_center_test(
        scene=scene,
        observations=observations,
        image_wh=image_wh,
        priors=annotation_priors,
        venue_measurements=venue_measurements,
        params=params
    )

    print("\nfixed-centre test (per-frame centre offset, m):")
    for stem, e in zip(observations["frames"], eps):
        print(f"  {stem}: [{e[0]:+.3f}, {e[1]:+.3f}, {e[2]:+.3f}]  "
                f"|eps| {np.linalg.norm(e):.3f}")
    print(f"  median |eps| {np.median(np.linalg.norm(eps, axis=1)):.3f} m")

