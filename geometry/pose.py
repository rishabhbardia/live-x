import cv2
import numpy as np

from typing import Any

from utils.freeze import check_shape

EPS: float = 1e-9

def rodrigues(omega: np.ndarray) -> np.ndarray:
    """
    Axis-angle (3,) -> (3, 3) world-to-camera rotation.
    """
    omega = np.asarray(omega, dtype=np.float64)
    check_shape(omega, rows=3)
    return cv2.Rodrigues(omega)[0]


def omega_of(R: np.ndarray) -> np.ndarray:
    """
    (3, 3) rotation -> its axis-angle (3,) vector. The inverse of
    `rodrigues`, for code that builds a rotation geometrically and has to write
    it out in the form every file on disk uses.
    """
    R = np.asarray(R, dtype=np.float64)
    check_shape(R, rows=3, cols=3)

    if (
        not np.allclose(R.T @ R, np.eye(3), atol=1e-6, rtol=0)
        or not np.isclose(np.linalg.det(R), 1.0, atol=1e-6, rtol=0)
    ):
        raise ValueError("R must be a proper rotation matrix")

    return cv2.Rodrigues(R)[0].reshape(3)

def look_at(C: Any, target: Any = (0.0, 0.0, 0.0),
            up: Any = (0.0, 0.0, 1.0)) -> np.ndarray:
    """World-to-camera rotation for a camera at `C` looking at `target`.

    `up` is WORLD up; the row that receives it is the camera's +Y, which points
    DOWN, so the cross products are ordered to produce a right-handed frame
    with Y descending.

    Degenerate inputs return something usable rather than NaN: a target at the
    camera gives identity, and looking straight along `up` picks an arbitrary
    horizontal right vector. Both are cases where no rotation is determined, so
    the alternative is silently propagating NaN into a solve.
    """
    Cv = np.asarray(C, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    check_shape(Cv, rows=3)
    check_shape(target, rows=3)
    check_shape(up, rows=3)

    fwd = target - Cv
    n = np.linalg.norm(fwd)
    if n < EPS:
        return np.eye(3)
    fwd = fwd / n
    right = np.cross(fwd, up)
    if np.linalg.norm(right) < 1e-6:          # looking straight up or down
        right = np.cross(fwd, np.array([1.0, 0.0, 0.0]))
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(fwd, right), fwd])


def plane_warp(K, omega_ref, C_ref, omega, C, basis):
    """Reference-to-current homography for a fixed world plane.
       Basis is a tuple of Origin, basis vector 1 and basis vector 2
    
       `Xw(a, b) = O + a*e1 + b*e2`, a world point on a plane can be written as
       origin plus scaled basis1 and scaled basis 2

       It's projection in an image is thus : 
       `p = K * R * (Xw - C)` this is the same notation we are using throughout,
       `C` is the camera center in world space and `R` is the rotation

       `p = K [R * e1 R * e2 R * (O - C)] * [a, b, 1].T`

       `p_ref = K [R_ref * e1, R_ref * e2, R_ref * (O - C)]  * x.T`
       `p_t   = K [R_t * e1, R_t * e2, R_t * (O - C)]  * x.T`

       `p_ref = Aref x`
       `p_t = At x`

       `x = Aref^-1 p_ref`
       `p_t = At * Aref^-1 p_ref`
       `p_t = H p_ref`
    """
    origin, e1, e2 = (
        np.asarray(value, dtype=np.float64) for value in basis
    )

    def plane_to_image(rotation, center):
        R = rodrigues(rotation)
        return K @ np.column_stack(
            (R @ e1, R @ e2, R @ (origin - center))
        )

    reference = plane_to_image(omega_ref, C_ref)
    current = plane_to_image(omega, C)

    # current @ inv(reference), without explicitly forming the inverse.
    return np.linalg.solve(reference.T, current.T).T