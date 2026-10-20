import numpy as np
import pyceres

from scipy.sparse import csr_matrix

from typing import Dict, Any

from data.venue import VenueGeometryStructure
from data.types import CameraLock

from geometry.scene import Scene, get_pillar_world
from geometry.projections import project
from geometry.pose import rodrigues

def reprojection_errors(scene: Scene, 
                        structure: VenueGeometryStructure, 
                        observations: Dict[str, Any]) -> tuple[np.array, np.array, np.array]:
    """
        Constructs the pillar structure from the venue specifications in world coordinates.
        Projects them into the image and computes the errors from the observations.

    """
    base_w, top_w = get_pillar_world(
        scene=scene,
        structure=structure
    )
    Rs = [rodrigues(w) for w in scene.omega]

    def px(Xw, fi):
        f, cx, cy = scene.intr_of(fi)
        return project(Xw[None], Rs[fi], scene.camera_center_of(fi), f, cx, cy)[0][0]

    err_base, err_top, err_can = [], [], []

    for k, (fi, pi) in enumerate(observations["pil_idx"]):
        err_base.append(
            np.linalg.norm(
                px(base_w[pi], fi) - observations["pil_base"][k]
            )
        )
        err_top.append(
            np.linalg.norm(
                px(top_w[pi], fi) - observations["pil_top"][k]
            )
        )
    p_canvas = np.array([[q[0], q[1], 0.0] for q in scene.tracked_points]) if scene.tracked_points else np.zeros((0, 3))
    for k, (fi, qi) in enumerate(observations["pt_idx"]):
        err_can.append(
            np.linalg.norm(
                px(p_canvas[qi], fi) - observations["pt_uv"][k]
            )
        )
    return np.array(err_base), np.array(err_top), np.array(err_can)


def report(scene: Scene, observations: Dict[str, Any], image_wh: tuple[int, int], structure: VenueGeometryStructure):
    """
        Returns a string for reporting the solution.
    """
    f, cx, cy = scene.intr_of(0)
    err_base, err_top, err_can = reprojection_errors(
        scene=scene,
        structure=structure,
        observations=observations
    )

    th = scene.thetas()
    lines = [
        "",
        f"lock = {scene.lock}   "
        + {CameraLock.ROT: "t = 0, one f for the clip",
           CameraLock.ZOOM: "t = 0, f per frame",
           CameraLock.MOVED: "t per frame"}[scene.lock],
        f"f  = {f:.1f} px      cx, cy = {cx:.1f}, {cy:.1f} "
        f"(image centre {image_wh[0]/2:.0f}, {image_wh[1]/2:.0f})",
        f"C  = [{scene.C[0]:+.3f}, {scene.C[1]:+.3f}, {scene.C[2]:+.3f}] m"
        f"   -> {np.linalg.norm(scene.C[:2]):.2f} m from cage centre, "
        f"{scene.C[2]:.2f} m up",
        f"stage = {structure.measurements.shape} ({structure.measurements.total_posts} posts)   "
        f"R_p = {structure.measurements.radius + scene.delta_radius[0]:.3f} m (nominal "
        f"{structure.measurements.radius:.3f}, dR {scene.delta_radius[0]:+.3f})"
        f"   H_p = {structure.measurements.pillar_height + scene.delta_height[0]:.3f} m (dH {scene.delta_height[0]:+.3f})",
        "pillar angles: " + "  ".join(
            f"{pid}:{np.degrees(t):+.1f}d" for pid, t in zip(observations["pillar_ids"], th)),
    ]

    lines += [
        "",
        f"reprojection px   base: med {np.nanmedian(err_base):.2f} max {np.nanmax(err_base):.2f}"
        f"   top: med {np.nanmedian(err_top):.2f} max {np.nanmax(err_top):.2f}"
        + (f"   landmark: med {np.nanmedian(err_can):.2f} max {np.nanmax(err_can):.2f}"
           if len(err_can) else ""),
        "",
        f"{'frame':14s} {'pil':>4s} {'pts':>4s} {'base px':>9s} {'top px':>8s} {'lmk px':>8s}",
    ]
    for t, stem in enumerate(observations["frames"]):
        m = observations["pil_idx"][:, 0] == t
        mq = observations["pt_idx"][:, 0] == t
        lines.append(f"{stem:14s} {int(m.sum()):4d} {int(mq.sum()):4d} "
                    f"{np.nanmedian(err_base[m]) if m.any() else float('nan'):9.2f} "
                    f"{np.nanmedian(err_top[m]) if m.any() else float('nan'):8.2f} "
                    f"{np.nanmedian(err_can[mq]) if mq.any() else float('nan'):8.2f}")
        
    behind = int(np.isnan(err_base).sum() + np.isnan(err_top).sum() + np.isnan(err_can).sum())
    if behind:
        lines.append(f"\nWARNING: {behind} observation(s) predict BEHIND the camera "
                    f"-- excluded above; the solve is in a mirrored configuration")
    return "\n".join(lines)


def parameter_names(scene: Scene, observations: Dict[str, Any]) -> tuple[list[Any], list[Any]]:
    """
        Gives back a tuple of lists. One contains the parameters that Scene held and the
        other has understandable names for them.
    """
    blocks, names = [], []
    stems = list(observations["frames"])
    if len(scene.focal_length) > 1:  # CameraLock.ZOOM is one focla per frame
        for t, stem in enumerate(stems):
            blocks.append(scene.focal_length[t])
            names += [f"f[{stem}]"]
    else:
        blocks.append(scene.focal_length[0])
        names += ["f"]

    blocks.append(scene.principal_point)
    names += ["cx", "cy"]

    if len(scene.camera_center_world) > 1: # CameraLock.MOVED: one centre per frame
        for t, stem in enumerate(stems):
            blocks.append(scene.camera_center_of(t))
            names += [f"C[{stem}].{a}" for a in "xyz"]
    else:
        blocks.append(scene.camera_center_world[0])
        names += ["C.x", "C.y", "C.z"]


    blocks.append(scene.delta_radius)
    names += ["deltaRadius"]
    blocks.append(scene.delta_height)
    names += ["deltaHeight"]

    for i, pid in enumerate(observations["pillar_ids"]):
        blocks.append(scene.theta[i])
        names += [f"theta[pillar {pid}]"]
    for t, stem in enumerate(observations["frames"]):
        blocks.append(scene.omega[t])
        names += [f"omega[{stem}].{a}" for a in "xyz"]
    for j, tid in enumerate(observations["track_ids"]):
        blocks.append(scene.tracked_points[j])
        names += [f"Q[track {tid}].X", f"Q[track {tid}].Y"]

    return blocks, names


def null_directions(s: np.ndarray, Vt: np.ndarray, names: list[str], k:int=3, min_weight: float=0.05) -> list[Any]:
    """
        The k weakest parameter-space directions, as (sigma, [(weight, name)]).

        We use min weight to keep the print meaningful, as many of these are close to 1e-3
        so they are practically not saying much

        `J = U S V^T`, so the rows of `V^t` are the directions in parameter space and the
        matching singular value says how hard the data pushes back along each one. These
        are a linear combination of parameters which is the direction described by the J.

        The last rows are what the observation barely constrain and reading their components
        tells you which parameters trade off against each other -- a scalar condition number 
        cannot.

        Components are in raw parameter units (px for f/cx/cy, meters for C and Q, radians for theta/omega),
        so a direction that mixes them is not a like for like blend, compare within a unit before across.
    """

    out = []
    for r in range(1, min(k, len(s)) + 1):
        v = Vt[-r]
        order = np.argsort(-np.abs(v))
        comps = [(float(v[i]), names[i]) for i in order if abs(v[i]) >= min_weight]
        out.append((float(s[-r]), comps or [(float(v[order[0]]), names[order[0]])]))

    return out

def strongest_direction(s: np.ndarray, Vt: np.ndarray, names: list[str], k:int=3, min_weight: float=0.1) -> list[Any]:
    """
        This is reverse of `null_directions`
    """
    out = []
    for r in range(1, min(k, len(s)) + 1):
        v = Vt[r-1]
        order = np.argsort(-np.abs(v))
        comps = [(float(v[i]), names[i]) for i in order if abs(v[i]) >= min_weight]
        out.append((float(s[r-1]), comps or [(float(v[order[0]]), names[order[0]])]))

    return out


def _crs(J):
    t = J.to_tuple() if hasattr(J, "to_tuple") else None
    if t is not None and len(t) == 5:
        return t
    return (np.asarray(J.rows), np.asarray(J.cols), np.asarray(J.values),
            J.num_rows, J.num_cols)


def form_jacobian(problem: pyceres.Problem, scene: Scene, observations: Dict[str, Any]) :
    """
        Rank and near-null directions of the full problem Jacobian.

        A tiny singular value means some combination of parameters the data does not
        pin down -- far more useful than a cost number, because it names which
        combination is unobservable.

        Returns : 
            dense : Jacobian matrix (this is ok for these small problems now)
            s     : Singular values of the SVD of the dense J
            Vt    : The vector matrix of the SVD on right, columns are linear 
                    combinations of the parameter directions.
            names : names of the parameters that went into the Jacobian
            n     : number of columns that are null directions.
    """

    blocks, names = parameter_names(scene=scene, observations=observations)
    options = pyceres.EvaluateOptions()
    options.set_parameter_blocks(blocks)
    CRS_J = problem.evaluate_jacobian(options) # compressed row structure
    rows, cols, values, num_rows, num_cols = _crs(CRS_J)
    
    dense = np.zeros((num_rows, num_cols))
    for r in range(num_rows):
        for k in range(rows[r], rows[r + 1]):
            dense[r, cols[k]] = values[k]

    
    # Columns for blocks held constant are identically zero. This is when we 
    # freeze a parameter block. Leaving them in manufactures null directions
    # that say nothing about observability, so they
    # are dropped before rank and conditioning are computed.
    live = np.abs(dense).max(axis=0) > 0
    dense = dense[:, live]

    names = [n for n, keep in zip(names, live) if keep]
    _, s, Vt = np.linalg.svd(dense, full_matrices=False)

    return dense, s, Vt, names, int((~live).sum())



def jacobian_analysis(problem: pyceres.Problem, scene: Scene, observations: Dict[str, Any]):
    """
        Forms the full jacobian of the problem, excludes the frozen blocks from the analysis.
        This prints out :
            1. Jacobian rank
            2. Condition number -> largest singular value / smallest singular value
            3. 3 largest singular values
            4. 3 smallest singular values
            5. Weakest 3 parameter direction
            6. Strongest 5 parameter direction
    """
    
    dense, s, Vt, names, n_frozen = form_jacobian(problem, scene, observations)
    rank = int((s > s[0] * 1e-10).sum())
    cond_value = s[0] / max(s[-1], 1e-300)
    rows = dense.shape[0]
    cols = dense.shape[1]
    deficient = rank - cols

    print(f"\njacobian {rows}x{cols} free params "
          f"\n number of names {len(names)}\n"
          f"({n_frozen} frozen columns dropped)   rank {rank}"
          f"   cond {cond_value:.3e}")

    print("  singular values, largest: " + " ".join(f"{v:.3e}" for v in s[:3]))
    print("  smallest:                 " + " ".join(f"{v:.3e}" for v in s[-3:]))

    print("  full rank -- every free parameter is constrained by the data"
              if rank == cols else
              f"  RANK DEFICIENT by {deficient}: some parameter "
              f"combination the data does not pin down")
    
    
    print(f"\nweakest parameter directions (rows of V^T, J = U S V^T)"
          f"   [px | m | rad -- mixed units, compare within a unit]")

    last_3_weakest = []
    for n, (sigma, comps) in enumerate(
        null_directions(s=s, Vt=Vt, names=names, k=3, min_weight=0.05), start = 1):
        print(f" v_{n} (sigma = {sigma:.4e}):")
        lw = []
        for w, name in comps:
            print(f"      {w:+.4f}  {name}")
            lw.append(name)
        last_3_weakest.append(lw)

    strongest = False
    if strongest:
        print(f"\nstrongest parameter directions (rows of V^T, J = U S V^T)"
            f"   [px | m | rad -- mixed units, compare within a unit]")
        for n, (sigma, comps) in enumerate(
            strongest_direction(s=s, Vt=Vt, names=names, k=2, min_weight=0.1), start = 1):
            print(f"v_{n} (sigma = {sigma:.4e}):")
            for w, name in comps:
                print(f"      {w:+.4f}  {name}")
    
    
    return last_3_weakest


def subcov(full_cov, names, name_to_idx):
    idx = [name_to_idx[name] for name in names]
    return full_cov[np.ix_(idx, idx)]

def build_uncertainty(problem: pyceres.Problem, 
                      scene: Scene, 
                      observation: Dict[str, Any]):
    blocks, names = parameter_names(
        scene=scene,
        observations=observation
    )

    free_blocks, free_names = [], []
    offset = 0

    for b in blocks:
        group = names[offset:offset + b.size]
        offset += b.size

        if problem.has_parameter_block(b) and not problem.is_parameter_block_constant(b):
            free_blocks.append(b)
            free_names.extend(group)

    blocks = free_blocks
    names = free_names

    names_idx = {name: i for i, name in enumerate(names)}

    total_params = 0
    for b in blocks:
        total_params += len(b)
    assert total_params == len(names)
    
    options = pyceres.CovarianceOptions()
    options.algorithm_type = pyceres.CovarianceAlgorithmType.DENSE_SVD
    options.apply_loss_function = True
    covariance = pyceres.Covariance(options)

    pairs = [
        (blocks[i], blocks[j])
        for i in range(len(blocks))
        for j in range(i, len(blocks))
    ]
    
    if not covariance.compute(pairs, problem):
        raise RuntimeError("Ceres covariance computation failed")

    sizes = [b.size for b in blocks]
    offsets = np.cumsum([0] + sizes)

    full_cov = np.zeros((offsets[-1], offsets[-1]))

    for i, a in enumerate(blocks):
        si = slice(offsets[i], offsets[i + 1])
        for j in range(i, len(blocks)):
            b = blocks[j]
            sj = slice(offsets[j], offsets[j + 1])

            result = covariance.get_covariance_block(a, b)
            if result is None:
                raise RuntimeError(f"Covariance unavailable for {i}, {j}")

            block = np.asarray(result, dtype=np.float64).reshape(
                sizes[i], sizes[j]
            )

            full_cov[si, sj] = block
            full_cov[sj, si] = block.T
    return full_cov, names, names_idx

def parameter_uncertainty(problem: pyceres.Problem, 
                          summary:pyceres.SolverSummary,
                          scene: Scene, 
                          observation: Dict[str, Any]):
    """
        ({name : 1-sigma}, chi2_reduced) from the covariance (J^T J)^-1.

    The singular values say which directions are stiff relative to each other;
    this converts that into an actual error bar per parameter, in the parameter's
    own units.

    Two things it assumes, both reported.
        * The residuals really do have the sigmas used to whiten them. Ceres returns 
          (J^T J)^-1 with unit variance residual assumed, so if the assumed pixel noise is wrong
          every error bar is wrong by the same factor. chi2_reduced is that factor square --
          near 1 means the sigmas were about right, below 1 means they were pessimistic.
        * The problem is full rank, but a DENSE_SVD is used rather than a SPARSE_QR so a rank
          deficient case degreades into a readable answer instead of a failure.
    """

    blocks, names = parameter_names(
        scene=scene,
        observations=observation
    )
    options = pyceres.CovarianceOptions()
    options.algorithm_type = pyceres.CovarianceAlgorithmType.DENSE_SVD
    options.apply_loss_function = True
    covariance = pyceres.Covariance(options)

    free = [b for b in blocks if problem.has_parameter_block(b) and not problem.is_parameter_block_constant(b)]
    # identity, not equality: `b in free` compares numpy arrays element-wise
    # and raises on ambiguous truth value

    free_ids = {id(b) for b in free}
    if not covariance.compute([(b,b) for b in free], problem):
        return None, float("nan")

    sigmas, i = {}, 0
    for b in blocks:
        n = len(b)
        group = names[i:i+n]
        i += n
        if id(b) not in free_ids:
            continue
        block = np.asarray(
            covariance.get_covariance_block(b,b), np.float64
        ).reshape(n, n)
        print(f"Group name : {group} has shape {block.shape}")
        for k, nm in enumerate(group):
            sigmas[nm] = float(np.sqrt(max(block[k, k], 0.0)))

    # The residual i.e. final cost needs to be scaled by 
    # the degree of freedom in Gaussian noise assumption
    # E[chi-sq] = m - n
    # chi-sq_reduced = chi-sq / (m - n)

    def diagnostic(chi_sq_reduced: float):
        if np.linalg.norm(chi_sq_reduced - 1) < 1e-3:
            return "Residuals are consistent with assumed noise variance."
        elif chi_sq_reduced > 1 :
            return "Noise is under estimated, model mismatch or outliers."
        elif chi_sq_reduced < 1 :
            return "Noise is over estimated, correlated residual, overfitting."
    
    dof = max(problem.num_residuals() - len(sigmas), 1)
    chi2 = float(2.0 * summary.final_cost / dof)

    print(f"chi-squared is {chi2:.4f}")
    print(diagnostic(chi2))
    return sigmas, chi2


def conditional_q(q_names, given_names, full_cov, name_idx):
    """Covariance of Q after conditioning on the specified parameters."""
    E = subcov(full_cov, q_names + given_names, name_idx)
    E_QQ, E_QI, E_II = E[:2, :2], E[:2, 2:], E[2:, 2:]

    result = E_QQ - E_QI @ np.linalg.solve(E_II, E_QI.T)
    return (result + result.T) / 2

def major_radius_cm(E):
    """95% confidence-ellipse major semiaxis, in centimeters."""
    scale95 = np.sqrt(-2.0 * np.log(0.05))
    return 100 * scale95 * np.sqrt(max(np.linalg.eigvalsh(E)[-1], 0.0))