import numpy as np
from utils.freeze import check_shape

EPS = 1e-9

def ray_dirs(
    uv: np.ndarray,
    f: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Pixels (N, 2) -> camera-space directions (N, 3), with z=1."""
    if not np.isfinite([f, cx, cy]).all() or f <= 0:
        raise ValueError("Intrinsics must be finite, with f > 0")

    x = (uv[:, 0] - cx) / f
    y = (uv[:, 1] - cy) / f

    return np.stack([x, y, np.ones_like(x)], axis=1)


def world_ray_dirs(uv: np.ndarray, R: np.ndarray, f: float, cx: float, cy: float, unit: bool = False) -> np.ndarray:
    """
    Pixels (N, 2) -> WORLD ray directions (N, 3) from the camera centre.

    `unit=False` keeps the z == 1 camera-space scaling (so a plane parameter
    along the ray is still camera depth)
    `unit=True` normalises, for a caller that wants a direction rather than 
    a parameterisation.
    """
    d = ray_dirs(uv, f, cx, cy) @ R
    if not unit:
        return d
    return d / np.linalg.norm(d, axis=1, keepdims=True)

def project(Xw: np.ndarray, R: np.ndarray, C: np.ndarray, f: float, cx: float, cy: float) -> tuple[np.ndarray, np.ndarray]:

    """ 
    World points (N,3) -> pixels (N,2) and camera-space Z (N,)

    Pixels for points at or behind the camera (z <= 0) are NaN.

    Pinhole projection model, square pixels, no skew or distortion.
    This is just for the broadcast projection and optimization not the
    renderer.
    """

    Xc = (Xw - C) @ R.T
    z = Xc[:, 2]
    safe = np.where(np.abs(z) < EPS, EPS, z)
    uv = np.stack([f * Xc[:, 0] / safe + cx,
                   f * Xc[:, 1] / safe + cy], 1)

    return np.where((z > EPS)[:, None], uv, np.nan), z


def unproject(uv: np.ndarray, Z: np.ndarray, R: np.ndarray, C: np.ndarray, f: float, cx: float, cy: float) -> np.ndarray:
    """
    Pixels (N, 2) and camera space depth (N,) in meters gives back world (N,3)

    Z is the depth along the camera's +Z axis, optical axis what `ClipBundle.metric_depth`
    returns -- not a range along the ray (common for monocular depth models like VDA)
    So the point in camera space would be Z * [x,y,1].

    For range it would be r * [x,y,1]/(||[x,y,1]||) i.e. normalized

    Xcamera = R @ (Xworld - Cworld)
    Xworld = R.T @ Xcamera + Cworld

        d * Z[:, none] 
        (N,3 .x N,1)
        (N,3) x (3,3) + (3,)
        N,3
    """

    d = ray_dirs(uv, f, cx, cy)
    Xworld = d * Z[:, None] @ R + C
    return Xworld
    

def floor_points(uv: np.ndarray, R: np.ndarray, C: np.ndarray, f
                    : float, cx: float, cy: float, plane_z: float = 0.0) -> np.ndarray:
    """
        Transforms the pixels (N, 2) to world points (N, 3) on the
        horizontal plane Z = plane_z. 

        The canvas is Z = 0 by construction for all combat sports in the world frame,
        so with `plane_z` left alone this is the exact metric position of any canvas
        pixels. No depth is involved, which is why the metric fit uses it as a ground
        truth and why it initializes the canvas landmarks

        Rays that do not reach the plane away from the camera come back NaN rather
        than reporting the intersection behind it, and so do rays with `EPS` of 
        horizontal, whole intersection is arbitrarily far away and numerically meaningless

    """
    z0 = plane_z
    d = world_ray_dirs(uv, R, f, cx, cy)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(np.abs(d[:, 2]) > EPS, (z0 - C[2]) / d[:, 2], np.nan)
        X = C[None, :] + t[:, None] * d
    return np.where((t > 0.0)[:, None], X, np.nan)



def plane_ray_depth(uv: np.ndarray, R: np.ndarray, C: np.ndarray, f: float, cx: float, cy: float,
                    O: np.ndarray, n_world: np.ndarray) -> np.ndarray:
    """
        Camera-space depth (N,) at which each pixel's ray meets the world plane
        through `O` with normal `n_world`.

        Exact for a point on the INFINITE plane -- restricting to the actual
        bounded surface (a polygon, a disk) is the caller's job. `np.inf` where the
        ray is parallel to the plane; a negative or small positive value means the
        plane is behind the camera along that ray, which callers filter the same
        way they filter `project`'s z <= 0.

        This is the operation the metric fit rests on: with the camera solved,
        every pixel on a known stage surface has an exact depth by intersection,
        which is thousands of correspondences that never involve fighter height.
    """
    Rm = np.asarray(R, np.float64)
    check_shape(n_world, rows=3)
    check_shape(O, rows=3)
    check_shape(C, rows=3)
    
    n_cam = Rm @ n_world
    Oc = Rm @ (O - C)
    d = ray_dirs(uv, f, cx, cy)
    denom = d @ n_cam
    num = float(n_cam @ Oc)
    return np.where(np.abs(denom) > EPS, num / denom, np.inf)

def plane_points(uv: np.ndarray, R: np.ndarray, C: np.ndarray, f: float, cx: float, cy: float,
                 O: np.ndarray, n_world: np.ndarray) -> np.ndarray:
    """World XYZ (N, 3) where each pixel's ray meets the plane (O, n_world).

    The inverse of `project` for a point known to lie on a given plane. What
    lets a brand-new detection get a world position from a TRUSTED pose instead
    of guessing one: back-project through the surface it was found on.
    """
    t = plane_ray_depth(uv, R, C, f, cx, cy, O, n_world)
    Xc = t[:, None] * ray_dirs(uv, f, cx, cy)
    return C + Xc @ np.asarray(R, np.float64)