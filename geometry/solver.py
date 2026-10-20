import cv2
import pyceres
import numpy as np

from typing import Dict, Any, Set

from data.types import CameraLock, AnnotationPriors, SolverParameters, SolverConfig
from data.venue import VenueMeasurements, VenueGeometryStructure

from geometry.costs import PillarCost, LandmarkCost, HeightPrior, PrincipalPointPrior, DeltaPrior
from geometry.scene import Scene, initialize_scene, init_landmarks


def solve(prob, max_iter=200, verbose=False):
    opts = pyceres.SolverOptions()
    opts.max_num_iterations = max_iter
    opts.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
    opts.minimizer_progress_to_stdout = verbose
    opts.logging_type = pyceres.LoggingType.SILENT
    summary = pyceres.SolverSummary()
    pyceres.solve(opts, prob, summary)
    return summary


def build_problem(scene: Scene, observations: Dict[str, Any], image_wh: tuple[int, int], 
                  use_points: bool, free: Set[str], priors: AnnotationPriors, 
                  venue_measurements: VenueMeasurements, params: SolverParameters,
                  f_ratios: float | None=None):
    """
        Builds a ceres problem based on the Scene, and annotated observations, 
        
        scene : Contains the whole parameter block that we are optimizing, depending on it's configuration certain parameters can be frozen
        observations : This is a dict of the annotations which tells us what's observed 
        image_wh : the image dimensions
        use_points : enables the landmark cost of the canvas points
        free : is a set of parameter names which are kept free to be optimized
        priors : used to get the observation's sigmas and other priors
        f_ratios : TODO : Motion model selector work

        Ceres stores raw pointers to python cost and loss objects. Letting them fall out of 
        scope segfaults the moment solver evaluates, so it is returned as a keepalive list
        and must outlive the Problem. 

        A residual block whose parameter blocks are all constant segfaults inside. That is why
        priors are added only when the thing they constrain is free.

        Freezing happens at build time rather than later through a freeze() pass, so a stage can
        never produce an all-constant block
    
    """

    sigma_base = priors.pillar_base_sigma_px
    sigma_top = priors.pillar_top_sigma_px
    sigma_landmark = priors.landmark_sigma_px
    sigma_principal_pt = priors.principal_point_sigma_px

    radius = venue_measurements.radius
    height = venue_measurements.pillar_height

    if "intr" in free:
        free |= {"focal", "pp"}

    problem = pyceres.Problem()
    loss = pyceres.HuberLoss(params.huber)

    keep = [loss]

    for k, (fi, pi) in enumerate(observations["pil_idx"]):
        for uv, is_top, s in ((observations["pil_base"][k], False, sigma_base),
                              (observations["pil_top"][k], True, sigma_top)):
            cost = PillarCost(uv, is_top, s, radius, height)
            keep.append(cost)
            problem.add_residual_block(
                cost, loss, 
                [
                    scene.focal_length_of(fi),
                    scene.principal_point,
                    scene.camera_center_of(fi),
                    scene.omega[fi],
                    scene.theta[pi],
                    scene.delta_radius,
                    scene.delta_height
                ]
            )
    if use_points:
        for k, (fi, qi) in enumerate(observations["pt_idx"]):
            cost = LandmarkCost(observations["pt_uv"][k], sigma_landmark)
            keep.append(cost)
            problem.add_residual_block(
                cost, loss, 
                [
                    scene.focal_length_of(fi), scene.principal_point, 
                    scene.camera_center_of(fi),
                    scene.omega[fi], 
                    scene.tracked_points[qi]
                ]
            )
    if "pp" in free:
        pp = PrincipalPointPrior(
            image_wh[0], image_wh[1], sigma_principal_pt
        )
        keep.append(pp)
        problem.add_residual_block(
            pp,
            None,
            [scene.principal_point]
        )

    if "delta" in free:
        dp = DeltaPrior(
            priors.delta_R_m
        )
        keep.append(dp)
        problem.add_residual_block(
            dp, None, 
            [scene.delta_radius]
        )

    if "dh" in free:
        hp = HeightPrior(priors.delta_H_m)
        keep.append(hp)
        problem.add_residual_block(
            hp, None, 
            [scene.delta_height]
        )

    if scene.lock == CameraLock.ZOOM and f_ratios and "focal" in free:
        print("Special case")

    for name, blocks, in scene.blocks().items():
        for i, b in enumerate(blocks):
            if not problem.has_parameter_block(b):
                continue
            if name in free and not (name == "theta" and i == 0): # gauge theta_0
                problem.set_parameter_block_variable(b)
            else:
                problem.set_parameter_block_constant(b)

    return problem, keep


def staged_solve(params: SolverParameters, 
                 focal_grid: np.ndarray,
                 observations: Dict[str, Any],
                 structure: VenueGeometryStructure,
                 venue_measurements: VenueMeasurements,
                 annotation_priors: AnnotationPriors,
                 image_wh: tuple[int, int], 
                 use_delta: bool, 
                 use_dh: bool,
                 verbose: bool = True,
                 lock: CameraLock = CameraLock.ROT, 
                 f_ratios: float | None = None):
    """
        Estimate camera intrinsics, poses, and venue geometry from annotated frames.

        Stages A–C share one focal length and one world-space camera center across
        frames, with a separate rotation for each frame (CameraLock.ROT).

        A — Pillars only:
            Search initial focal lengths, pillar-angle directions and spreads, and
            camera distances. For each seed, optimize rotations first, then jointly
            refine camera parameters and pillar angles, plus enabled radius/height
            corrections. Keep the usable solution with the lowest cost.

        B — Canvas landmarks:
            Initialize tracked points on the floor plane, then optimize their XY
            positions and per-frame camera rotations.

        C — Joint refinement:
            Refine camera parameters, pillar angles, and canvas landmarks together,
            including enabled radius/height corrections. The first pillar angle
            remains fixed to anchor the world's azimuth.

        D — Optional motion-model relaxation:
            For ZOOM, allow per-frame focal lengths. For MOVED, allow per-frame
            camera centers. Rebuild the problem and refine under the requested model.

        Returns:
            scene: Optimized parameter state.
            problem: Final Ceres problem.
            keep: Cost/loss references that must remain alive while using problem.
            summary: Final solver summary.
    """

    best = None
    for f0 in focal_grid:
        for direction in (+1.0, -1.0):
            for spread in (25.0, 45.0):
                for dist in (4.5, 8.0):
                    scene: Scene = initialize_scene(
                        observations=observations,
                        focal_init=f0,
                        venue_geometry=structure,
                        image_wh=image_wh,
                        direction=direction,
                        spread=spread,
                        distance=dist,
                        lock=CameraLock.ROT
                    )
                    problem, keep = build_problem(
                        scene=scene,
                        observations=observations,
                        image_wh=image_wh,
                        use_points=False,
                        free=set({"omega"}),
                        priors=annotation_priors,
                        venue_measurements=venue_measurements,
                        params=params,
                        f_ratios=None
                    )

                    # Just a rotation solve first
                    _ = solve(prob=problem, max_iter=params.max_iters)

                    free_stage_1 = ({"focal", "pp", "center", "theta", "omega"} |
                                ({"delta"} if use_delta else set()) |
                                ({"dh"} if use_dh else set()))

                    problem, keep = build_problem(
                        scene=scene,
                        observations=observations,
                        image_wh=image_wh,
                        use_points=False,
                        free=free_stage_1,
                        priors=annotation_priors,
                        venue_measurements=venue_measurements,
                        params=params
                    )
                    sumA = solve(problem, max_iter=400)
                    if not sumA.IsSolutionUsable():
                        continue

                    if best is None or sumA.final_cost < best[0]:
                        best = (sumA.final_cost, scene, f0, direction, spread, dist)
    
    if best is None:
        raise RuntimeError("No initialization produced a usable solution")
    costA, scene, f0, direction, spread, dist = best
    scene.cost_history[SolverConfig.PILLARS_ONLY] = float(costA)

    if verbose:
        print(f"[stage A] pillars only: best of {len(focal_grid)*8} starts "
                f"(f0={f0:.0f} dir={direction:+.0f} spread={spread:.0f} dist={dist:.1f})  "
                f"cost={costA:.4f}")

    init_landmarks(scene=scene, observations=observations, radius=venue_measurements.radius)
    problem, keep = build_problem(scene=scene, observations=observations, image_wh=image_wh,
                                    use_points=True, free=set({"Q", "omega"}), 
                                    priors=annotation_priors, venue_measurements=venue_measurements, 
                                    params=params)
    sumB = solve(prob=problem, max_iter=params.max_iters)
    
    if not sumB.IsSolutionUsable():
        raise ValueError("Stage 2's solution not usable")
    scene.cost_history[SolverConfig.PILLARS_LANDMARKS] = float(sumB.final_cost)
    
    if verbose:
        print(f"[stage B] landmarks added: cost={sumB.final_cost:.4f}")

    
    free_stage_3 = ({"focal", "pp", "center", "theta", "omega", "Q"} |
                ({"delta"} if use_delta else set()) | ({"dh"} if use_dh else set()))
    
    problem, keep = build_problem(
        scene=scene,
        observations=observations,
        image_wh=image_wh,
        use_points=True,
        free=free_stage_3,
        priors=annotation_priors,
        venue_measurements=venue_measurements,
        params=params
    )
    sumC = solve(problem, max_iter=500)

    if not sumC.IsSolutionUsable():
        raise ValueError("Stage 3's solution not usable")
    
    scene.cost_history[SolverConfig.JOINT] = float(sumC.final_cost)
    if verbose:
        print(f"[stage C] joint refine:   cost={sumC.final_cost:.4f}   "
                f"{sumC.BriefReport()}")

    summary = sumC

    if lock != CameraLock.ROT:
        print("Relaxing to per-frame focal and per-frame center")
        scene.relax(lock=lock)
        problem, keep = build_problem(
            scene=scene,
            observations=observations,
            image_wh=image_wh,
            use_points=True,
            free=free_stage_3,
            priors=annotation_priors,
            params=params,
            venue_measurements=venue_measurements,
            f_ratios=f_ratios
        )
        summary = solve(problem, max_iter=600)
        scene.cost_history[SolverConfig.RELAX] = float(summary.final_cost)
        if verbose:
            n_extra = ((len(scene.focal_length) - 1) if lock == CameraLock.ZOOM
                        else 3 * (len(scene.camera_center_world) - 1))
            print(f"[stage D] lock={lock}: cost={summary.final_cost:.4f}   "
                    f"(+{n_extra} free parameters"
                    + (f", {len(f_ratios)} focal-ratio priors" if f_ratios else "")
                    + f")   {summary.BriefReport()}")
    
    return scene, problem, keep, summary