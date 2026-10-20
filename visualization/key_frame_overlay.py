import cv2
import numpy as np

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from interface.clip import ClipBundle

from data.venue import VenueGeometryStructure
from data.types import SurfaceType

from geometry.projections import project
from geometry.pose import rodrigues
from geometry.scene import Scene, get_pillar_world
from geometry.surface import StageSurfaces

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def stage_outline(n_posts, r_post, theta0=0.0, n_arc=361, fixed_vertices=False):
    
    if not fixed_vertices:
        a = np.linspace(0.0, 2.0 * np.pi, n_arc)
        return np.stack([r_post * np.cos(a), r_post * np.sin(a),
                         np.zeros_like(a)], 1)
    k = np.arange(n_posts + 1)            # +1 closes the polygon
    a = theta0 + k * (2.0 * np.pi / n_posts)
    return np.stack([r_post * np.cos(a), r_post * np.sin(a),
                     np.zeros_like(a)], 1)


PRED_A_COLOR = (255, 170, 60)     # 
PRED_B_COLOR = (60, 170, 255)     # 
CAGE_COLOR = (90, 200, 255)
CAGE_TOP_COLOR = (200, 140, 60)
SURF_CANVAS_COLOR = (200, 160, 0)               # StageSurfaces canvas fill
SURF_POST_COLORS = [(0, 90, 200), (170, 0, 200), (0, 200, 90)]  # cycle by post id
SURF_BAND_COLORS = [(200, 200, 60), (60, 200, 200)]             # cycle by panel
PERSON_COLOR = (0, 0, 220)                      # raw person silhouette
MSER_COLOR = (255, 255, 255) 

def _blend(img, mask, color, alpha):
    """In-place alpha blend of `color` over `img` wherever `mask` is True."""
    if mask is not None and mask.any():
        img[mask] = (img[mask].astype(np.float32) * (1.0 - alpha)
                     + np.array(color, np.float32) * alpha).astype(img.dtype)

def covariance_ellipse(cov, confidence_scale=2.448):
    """
        2.448 gives the approximate 95% joint confidence ellipse
        for 2D
    """
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    axes = confidence_scale * np.sqrt(np.maximum(eigenvalues, 0))
    angle = np.degrees(np.arctan2(
        eigenvectors[1, 0], eigenvectors[0, 0]
    ))

    return axes, angle


def render_surfaces(surf: StageSurfaces, 
                    clip: ClipBundle,
                    observations: Dict[str,Any],
                    out_dir: Path, 
                    alpha: float = 0.4) -> list[str]:
    """Save every annotated frame with `surf`'s visible surfaces alpha-blended over it.

    """
    if not out_dir.is_dir():
        out_dir.mkdir(parents=True, exist_ok=True)

    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha!r}")
    
    frames = observations["frames"]
    calibrated_scene = clip.get_clip().calibrated_scene

    for i, stem in enumerate(frames):
        idx = clip.frame_key_index(stem)
        img = deepcopy(clip.get_frame(idx).broadcast_frame).copy()
        C = calibrated_scene.camera_center_of(i)
        omega = calibrated_scene.omega[i]
        
        if img is None:
            raise FileNotFoundError(f"no frame image for annotated stem {stem!r}")

        legend: list[tuple[str, tuple[int, int, int]]] = [(stem, (255, 255, 255))]
        for key, m in surf.all_masks(omega, C, idx, clip).items():
            if not m.any():
                continue
            if key[0] == SurfaceType.CANVAS:
                color, name = SURF_CANVAS_COLOR, "canvas"
            elif key[0] == SurfaceType.BAND:
                color = SURF_BAND_COLORS[hash(key[1]) % len(SURF_BAND_COLORS)]
                name = f"band {key[1]}"
            else:
                color = SURF_POST_COLORS[key[1] % len(SURF_POST_COLORS)]
                name = f"post {key[1]}"
            _blend(img, m, color, alpha)
            legend.append((name, color))

        scale = img.shape[0] / 1080.0
        for row, (text, color) in enumerate(legend):
            org = (int(20 * scale), int((40 + 36 * row) * scale))
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale,
                        (0, 0, 0), max(1, int(4 * scale)), cv2.LINE_AA)
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale,
                        color, max(1, int(2 * scale)), cv2.LINE_AA)

        path = out_dir / f"{stem}_surfaces.png"
        if not cv2.imwrite(path, img):
            raise OSError(f"could not write {path}")
    return


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2


def plot_world_q_uncertainty(world_Q_uncertainty, output: Path, confidence=0.95):
    fig, ax = plt.subplots(figsize=(9, 9))
    scale = np.sqrt(chi2.ppf(confidence, df=2))

    for name, data in world_Q_uncertainty.items():
        pt = np.asarray(data["pt"]).reshape(2)
        cov = np.asarray(data["E"]).reshape(2, 2)

        # Covariance eigenvectors define ellipse orientation.
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 0)
        eigvecs = eigvecs[:, order]

        axes = scale * np.sqrt(eigvals)
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

        ax.add_patch(Ellipse(
            xy=pt,
            width=2 * axes[0],
            height=2 * axes[1],
            angle=angle,
            fill=False,
            linewidth=1.5,
        ))

        ax.plot(*pt, "o", markersize=4)
        ax.annotate(str(name), pt, xytext=(5, 5),
                    textcoords="offset points")

    ax.set_aspect("equal")
    ax.autoscale()
    ax.margins(0.15)
    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Y (m)")
    ax.set_title(f"Q World Position — {confidence:.0%} Confidence Ellipses")
    ax.grid(alpha=0.3)

    fig.savefig(output, dpi=300, bbox_inches="tight")
    return fig, ax

def batch_overlay(output: Path,
                  scene:Scene,
                  observations: Dict[str,Any],
                  clip: ClipBundle,
                  structure: VenueGeometryStructure,
                  q_uncertainty: dict[str, list[dict[str, np.ndarray]]] | None = None,
                  interactive:bool = True):
    print("Writing otuput")
    if not output.is_dir():
        output.mkdir(parents=True, exist_ok=True)

    base_w, top_w = get_pillar_world(
            scene=scene,
            structure=structure
        )

    r = structure.measurements.radius + float(scene.delta_radius)
    h_p = structure.measurements.pillar_height + float(scene.delta_height)

    circle = stage_outline(n_posts=structure.measurements.total_posts,
                  r_post=r)
    Qw = np.array([[q[0], q[1], 0.0] for q in scene.tracked_points]) if scene.tracked_points else np.zeros((0, 3))

    written = []
    for t, stem in enumerate(observations["frames"]):
        i = clip.frame_key_index(stem)
        frame = clip.get_frame(i)
        img = deepcopy(frame.broadcast_frame).copy()
        if img is None:
            continue
        R = rodrigues(scene.omega[t])
        fl, cx, cy = scene.intr_of(t)
        C_t = scene.camera_center_of(t)
        for pts, col in ((circle, (90, 200, 255)),
                         (circle + [0, 0, h_p], (200, 140, 60))):
            uv, z = project(pts, R, C_t, fl, cx, cy)
            ok = np.where(z > 1e-6)[0]
            for a, b in zip(ok[:-1], ok[1:]):
                if b == a + 1:
                    cv2.line(img, tuple(uv[a].astype(int)), tuple(uv[b].astype(int)),
                             col, 1, cv2.LINE_AA)
        for i in range(len(scene.theta)):
            ua, za = project(base_w[i][None], R, C_t, fl, cx, cy)
            ub, zb = project(top_w[i][None], R, C_t, fl, cx, cy)
            if za[0] <= 0 or zb[0] <= 0:      # behind the camera; drawing the
                continue                      # mirrored pixel would be a lie
            a = tuple(ua[0].astype(int))
            b = tuple(ub[0].astype(int))
            cv2.line(img, a, b, (60, 180, 255), 2, cv2.LINE_AA)
            cv2.circle(img, a, 6, (60, 180, 255), 2)
            cv2.putText(img, str(observations["pillar_ids"][i]), (a[0] + 8, a[1] + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 180, 255), 2)
        uvq, zq = project(Qw, R, C_t, fl, cx, cy)
        for uv, z in zip(uvq, zq):
            if z > 1e-6:
                cv2.circle(img, tuple(uv.astype(int)), 5, (80, 230, 80), 2)
        
        if interactive == False:
            print("Write uncertainties")
            for item in (q_uncertainty or {}).get(stem, []):
                proj = np.asarray(item["proj"], dtype=np.float64).reshape(2)
                E_uv = np.asarray(item["E_uv"], dtype=np.float64).reshape(2, 2)

                if not np.all(np.isfinite(proj)) or not np.all(np.isfinite(E_uv)):
                    continue

                # Returns 95% ellipse semi-axes (pixels) and angle (degrees)
                axes, angle = covariance_ellipse(E_uv)

                center = tuple(np.rint(proj).astype(int))
                radii = tuple(np.maximum(1, np.rint(axes)).astype(int))

                cv2.ellipse(
                    img,
                    center,
                    radii,
                    float(angle),
                    0,
                    360,
                    (255, 255, 0),  # Cyan, BGR
                    1,
                    cv2.LINE_AA,
                )
        else:
            fig, ax = plt.subplots(figsize=(16, 9))
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            for item in q_uncertainty[stem]:
                proj = item["proj"]
                axes, angle = covariance_ellipse(item["E_uv"])

                ax.add_patch(Ellipse(
                    xy=proj,
                    width=2 * axes[0],
                    height=2 * axes[1],
                    angle=angle,
                    fill=False,
                    edgecolor="cyan",
                    linewidth=1.5,
                ))
                ax.plot(*proj, "r+", markersize=6)

            plt.show()
        
        for k, (fi, _) in enumerate(observations["pil_idx"]):
            if fi == t:
                cv2.circle(img, tuple(observations["pil_base"][k].astype(int)), 3, (255, 255, 255), -1)
                cv2.circle(img, tuple(observations["pil_top"][k].astype(int)), 3, (255, 255, 255), -1)
        for k, (fi, _) in enumerate(observations["pt_idx"]):
            if fi == t:
                cv2.circle(img, tuple(observations["pt_uv"][k].astype(int)), 3, (255, 255, 255), -1)
        path = output / f"{stem}_overlay.png"
        cv2.imwrite(path, img)
        written.append(path)
    return written
