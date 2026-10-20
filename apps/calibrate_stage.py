from interface.clip import ClipBundle
from interface.serializer import save_scene
from interface.path_handler import STAGE_FILE, CALIBRATION_SIGMA

from data.venue import VenueGeometryStructure
from data.payloads import organize_annotations
from data.types import SolverParameters, CameraLock

from geometry.costs import _project_jac
from geometry.solver import staged_solve
from geometry.sensitivity import report, jacobian_analysis, parameter_uncertainty, build_uncertainty, subcov, major_radius_cm, conditional_q
from visualization.key_frame_overlay import batch_overlay, plot_world_q_uncertainty

from pathlib import Path

import json
import argparse
import numpy as np

def print_shape(value):
        print(value.shape)

if __name__ == "__main__":

    ap = argparse.ArgumentParser(description="Runs the calibration step of the pipeline")
    ap.add_argument("--clip_dir", type=str, default=None, required=True, help="Base directory with data")
    ap.add_argument("--recompute", action="store_true", help="Forces a recompute")
    args = ap.parse_args()

    verbose = True
    clip_path = Path(args.clip_dir)
    clip_bundle = ClipBundle(clip_path, force_recompute=args.recompute)

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

    print(annotation_priors)


    last_weakest = jacobian_analysis(
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

    with open(clip_path / CALIBRATION_SIGMA, "w", encoding="utf-8") as fh:
        json.dump(sigmas, fh, indent=4)


    output = clip_path / STAGE_FILE

    save_scene(
          path=output,
          scene=scene
    )
                

    full_cov, names, name_idx = build_uncertainty(
        problem=problem,
        scene=scene,
        observation=observations
    )

    # This report uses shared intrinsics and camera center (CameraLock.ROT).
    if scene.lock != CameraLock.ROT:
        raise ValueError("Uncertainty report currently supports CameraLock.ROT only")

    # Retain the weakest parameter group from the Jacobian analysis.
    # Expected: last_weakest[0] == ["f", "cy", "cx"] for this solve.
    if not last_weakest:
        raise ValueError("jacobian_analysis must return weakest parameter-name groups")
    weak_names = list(last_weakest[0])
    center_names = ["C.x", "C.y", "C.z"]
    both_names = list(dict.fromkeys(weak_names + center_names))

    # The largest marginal covariance mode of the selected group.
    # Its ranking depends on parameter units (pixels vs meters).
    E_II = subcov(full_cov, both_names, name_idx)
    eigenvalues, eigenvectors = np.linalg.eigh(E_II)
    delta_I = np.sqrt(max(eigenvalues[-1], 0)) * eigenvectors[:, -1]
    print(f"Weak Jacobian parameters: {weak_names}")
    print(f"Joint [weak + C] 1-sigma covariance mode: {dict(zip(both_names, delta_I))}")
    print("\nQ   original  | weak known | C known | weak+C known | dQ for joint mode (cm)")

    world_Q_uncertainty = {}
    analysis = {}
    for ti, tid in enumerate(observations["track_ids"]):
        q_names = [f"Q[track {tid}].X", f"Q[track {tid}].Y"]
        E_Q = subcov(full_cov, q_names, name_idx)
        E_Q_weak = conditional_q(q_names, weak_names, full_cov, name_idx)
        E_Q_C = conditional_q(q_names, center_names, full_cov, name_idx)
        E_Q_both = conditional_q(q_names, both_names, full_cov, name_idx)

        # Conditional-mean movement for a perturbation along the joint mode.
        E_QI = subcov(full_cov, q_names + both_names, name_idx)[:2, 2:]
        delta_Q = E_QI @ np.linalg.solve(E_II, delta_I)

        world_Q_uncertainty[tid] = {"E": E_Q, "pt": scene.tracked_points[ti]}
        analysis[tid] = {
            "E_Q": E_Q,
            "E_Q_given_weak": E_Q_weak,
            "E_Q_given_C": E_Q_C,
            "E_Q_given_weak_C": E_Q_both,
            "delta_Q": delta_Q,
        }
        print(f"{str(tid):>2}   {major_radius_cm(E_Q):8.2f}   "
              f"{major_radius_cm(E_Q_weak):8.2f}   "
              f"{major_radius_cm(E_Q_C):7.2f}   "
              f"{major_radius_cm(E_Q_both):10.2f}     "
              f"{np.round(100 * delta_Q, 2)}")

    # Predicted pixel covariance for every Q in every annotated frame.
    visualization_data = {}
    for fi, stem in enumerate(observations["frames"]):
        intr = scene.intr_of(fi)
        C = scene.camera_center_of(fi)
        omega = scene.omega[fi]
        camera_names = [
            "f", "cx", "cy", "C.x", "C.y", "C.z",
            *(f"omega[{stem}].{a}" for a in "xyz"),
        ]
        visualization_data[stem] = []
        for ti, tid in enumerate(observations["track_ids"]):
            q_names = [f"Q[track {tid}].X", f"Q[track {tid}].Y"]
            q = scene.tracked_points[ti]
            Xw = np.array([q[0], q[1], 0.0])
            proj, J_intr, J_Cw, J_omega, J_Xw, z = _project_jac(Xw, intr, C, omega)
            if z <= 1e-6:
                continue

            J_proj = np.hstack((J_intr, J_Cw, J_omega, J_Xw[:, :2]))
            E_proj = subcov(full_cov, camera_names + q_names, name_idx)
            E_uv = J_proj @ E_proj @ J_proj.T
            E_uv = (E_uv + E_uv.T) / 2
            visualization_data[stem].append({"proj": proj, "E_uv": E_uv})

    output = clip_path / "calibration_overlay"
    batch_overlay(
        output=output,
        scene=scene,
        observations=observations,
        clip=clip_bundle,
        structure=structure,
        q_uncertainty=visualization_data,
        interactive=False,
    )
    plot_world_q_uncertainty(
        world_Q_uncertainty=world_Q_uncertainty,
        output=output / "uncertainty_Q.png",
    )