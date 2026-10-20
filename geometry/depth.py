
import numpy as np
import cv2
import pyceres

from typing import Dict, Tuple
from copy import deepcopy

from data.types import SurfaceType

from geometry.costs import InvDepthCost
from geometry.solver import solve
from geometry.projections import plane_ray_depth
from geometry.pose import rodrigues

# Experiments on 1 frame had rough height validation
ALPHA_V = 0.00047590
BETA_V  = 0.092232
GAMMA_V = 0.85471

# Constants for fitting
SURFACES_FOR_METRIC = (SurfaceType.CANVAS, SurfaceType.POST)

SAMPLES_PER_SURFACE = 400   # pixels sampled per surface per frame
ERODE_PX = 6                # shrink each surface mask before sampling
MIN_DEPTH_M = 1.0           # reject geometry closer than this (grazing rays)
MAX_DEPTH_M = 25.0          # ... and further than the cage can possibly be
HUBER_INV = 0.02            # robust loss in 1/m, ~ the spread of a good fit
SIGMA_INV = 0.01            # nominal 1/m sigma on an inverse-depth residual


def inverse_depth(d, alpha, beta, gamma):
    """
        d is the disparity which is fit to metric against 1/Z_geom
        So this is effectively 1/Z
    """
    return alpha * np.power(np.maximum(d, 1e-6), gamma) + beta

def corrected_depth(d, p, b):
    """
        The per frame correction version
    """
    return 1.0 / (inverse_depth(d, *p) + b + 1e-6)

def fitglobal(d: np.ndarray, 
              z: np.ndarray, 
              init:np.ndarray, 
              huber:float, 
              sigma:float, 
              verbose=False):
    """
        Robust fit of [alpha, beta, gamma] to (disparity, geometric depth).

        d : 1-D array of inverse depth from the model
        z : 1-D array of inverse depth of the analytical points from geometry 
        init : Previously fitted params from a single frame (src manual experiment)
               Ordered as [`alpha`, `beta`, `gamma`]
        huber : huber curve param
        sigma : the 1/m std for whitening
         
    """
    p = np.array([init[0], init[1], init[2]], np.float64)
    problem = pyceres.Problem()
    loss = pyceres.HuberLoss(huber / sigma)
    keep = [loss]
    for di, zi in zip(d, z):
        cost = InvDepthCost(di, zi, sigma)
        keep.append(cost)
        problem.add_residual_block(cost, loss, [p])
    summary = solve(problem, max_iter=200, verbose=verbose)
    return p, summary, keep


def fit_per_frame(
        d: np.ndarray, 
        z: np.ndarray, 
        init:np.ndarray, # global fit
        huber:float,
        iterations:int = 20,
):
    """
        Robust per-frame inverse-depth offset.

        The residual is linear in b, so this is a robust location estimate,
        not a NLLS. `b` is whatever constant best explains `1/Z_geom - q(d)`
        Initialized at the median and refined by Huber IRLS.
    """
    target = 1.0 / z - inverse_depth(d, *init)
    b = float(np.median(target))
    for _ in range(iterations):
        s = np.abs(target - b)
        w = np.where(s <= huber, 1.0, huber / np.maximum(s, 1e-12))
        nb = float(np.sum(w * target) / np.sum(w))
        if abs(nb - b) < 1e-12:
            return nb
        b = nb
    return b


class DepthMetricFit:
    """
        Fits the network depth to inverse depth.
    """

    def __init__(self, num_frames: int):
        self.num_frames = num_frames
        
        self.huber: float = HUBER_INV
        self.sigma: float = SIGMA_INV

        self.seed : int = 0

        self.alpha = np.array([ALPHA_V])
        self.beta = np.array([BETA_V])
        self.gamma = np.array([GAMMA_V])
        self.beta_t = [np.zeros(1) for _ in range(num_frames)]

    
    def _sample_frame(self, surface, disparity, masks, K, om, C, rng, n_per_surface, erode):
        """
            Sample (disparity, geometric depth) pairs from one frame

            Erode is a gating on edge pixels, a SAM3 mask would be better here.
            A monocular depth network smooths across depth discontinuities, so the pixels
            within a few px of a surface's silhouette carry a blend of that surface and
            whatever is behind it. That is worst exactly where this fit is most
            valuable: a ~24 cm post at ~8 m is only tens of px wide against a far
            background, so an un-eroded post sample is substantially background.


            Returns a dict of arrays.
        """

        # Generate the surface origin and normals
        canvas_basis = deepcopy(surface.canvas_basis())
        post_origin_normals = deepcopy(surface.post_normals())

        cO, e1, e2 = canvas_basis
        planes = {
            (SurfaceType.CANVAS,) : (cO, np.cross(e1, e2)),
        }
        for k, v in post_origin_normals.items():
            pO, pN = v
            planes[(SurfaceType.POST, k)] = (pO, pN)

        d_out, z_out, surf_out, uv_out = [], [], [], []

        d_canvas, z_canvas = [], []

        k = (np.ones((2 * erode + 1,) * 2, np.uint8) if erode > 0 else None)

        for key, (O, n) in planes.items():
            m = masks.get(key)
            if m is None or not m.any():
                continue
            if k is not None:
                m = cv2.erode(m.astype(np.uint8), k) > 0
                if not m.any():
                    continue
            ys, xs = np.nonzero(m)
            if len(xs) > n_per_surface:
                pick = rng.choice(len(xs), n_per_surface, replace=False)
                ys, xs = ys[pick], xs[pick]
            uv = np.stack([xs, ys], 1).astype(np.float64)

            z = plane_ray_depth(
                uv=uv,
                R=rodrigues(omega=om),
                C=C,
                f=K[0,0],
                cx=K[0,2],
                cy=K[1,2],
                O=O,
                n_world=n
            )
            d = disparity[ys, xs].astype(np.float64)
            ok = (np.isfinite(z) & (z > MIN_DEPTH_M) & (z < MAX_DEPTH_M)
                    & np.isfinite(d) & (d > 0))
            if not ok.any():
                continue
            d_out.append(d[ok])
            z_out.append(z[ok])
            uv_out.append(uv[ok])
            surf_out += [f"{key[0]}{'' if len(key) == 1 else key[1]}"] * int(ok.sum())

            if key == (SurfaceType.CANVAS,):
                d_canvas.append(d[ok])
                z_canvas.append(z[ok])

        if not d_out:
            return None, None

        return {"d": np.concatenate(d_out), "z": np.concatenate(z_out),
            "uv": np.concatenate(uv_out), "surface": np.array(surf_out)}, {
                "d_canvas" : d_canvas,
                "z_canvas" : z_canvas,
            }
    
    def _collect(self, clip, 
                surface, 
                tracker,
                rng,
                subsampling=-1):
        """
            Gathers all the data needed to fit the depth
        """
        canvas_all = {
            
        }
        parts = []
        records = {
            record["stem"]: record
            for record in tracker.records
        }
        for i in range(0, clip.size()):
            if subsampling > 0:
                if i % subsampling != 0:
                    continue
            print(f"Collecting frame {i}")
            f, cx, cy = clip.get_clip().calibrated_scene.intr_of(i)
            K = np.array(
                    [[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]],
                        dtype=np.float64,
            )
            stem = clip.frame_index_key(i)
            record = records.get(stem)
            if record is None:
                continue
            assert stem == record["stem"]
            assert i == record["frame_index"]
            if record["pose_solved"] is False and record.get("pose_source") != "calibration":
                print("Current pose was not solved, skipping")
                continue
            omega = np.asarray(record["omega"], np.float64)
            C = np.asarray(record["C"], np.float64)

            disparity = clip.get_frame(i).inverse_depth
            if disparity is None:
                continue
            masks = surface.all_masks(omega, C, i, clip)

            print(f"Pose --> {C}, {omega}")

            rec, canvas = self._sample_frame(surface, disparity, masks, K, omega, C, rng, SAMPLES_PER_SURFACE, ERODE_PX)
            if rec is None:
                continue

            rec["frame"] = np.full(len(rec["d"]), i)

            if len(canvas["d_canvas"]) > 0:
                canvas_all[i] = {"d" : np.concatenate(canvas["d_canvas"]), 
                                "z" : np.concatenate(canvas["z_canvas"])}


            parts.append(rec)
        if not parts:
            return None, None
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}, canvas_all

    def _filter_surface(self, data, surface_type):
        ds, zs = [], []
        for d, z, surface in zip(data["d"], data["z"], data["surface"]):
            if surface_type in surface:
                ds.append(d)
                zs.append(z)
        return np.asarray(ds), np.asarray(zs)
    
    def run(self, 
            clip, 
            surface, 
            tracker,
            subsampling=-1):
        rng = np.random.default_rng(self.seed)
        data, canvas_all = self._collect(clip=clip, surface=surface, tracker=tracker, rng=rng, subsampling=subsampling)
        if data is None:
            raise ValueError("No valid disparity/geometric-depth correspondences")
        print(f"A total of {len(data["d"])} correspondences have been found")
        print(f"Surface data {len(data["surface"])}")
        print(f"Parameters init {self.alpha[0]} {self.beta[0]} {self.gamma[0]}")
        p, summary, keep = fitglobal(
            d=data["d"],
            z=data["z"],
            init=np.asarray([self.alpha[0], self.beta[0], self.gamma[0]]),
            huber=HUBER_INV,
            sigma=SIGMA_INV,
        )
        self.alpha[0], self.beta[0], self.gamma[0] = p
        print(f"Global parameters final {self.alpha[0]} {self.beta[0]} {self.gamma[0]}")

        for fi, canvas in canvas_all.items():
            d = canvas["d"]
            z = canvas["z"]

            bt = fit_per_frame(
                d=d,
                z=z,
                init=np.asarray([self.alpha[0], self.beta[0], self.gamma[0]]),
                huber=HUBER_INV,
            )

            self.beta_t[fi] = bt
    
        return


 