
import open3d as o3d
import numpy as np
import cv2


GRAD_REL = 0.035         # |grad Z| / Z above this is a depth discontinuity
MIN_FACING = 0.26        # cos(75 deg); below this the surface is grazing
SOR_NEIGHBOURS = 16
SOR_STD = 2.0

from geometry.projections import ray_dirs

def sor(xyz, rgb, k=SOR_NEIGHBOURS, std=SOR_STD):
    """Statistical outlier removal, the last line of defence.

    Catches what the image-space filters cannot see: a point that looks locally
    smooth in the depth map but sits alone in space, which is what a whole
    interpolated edge SEGMENT looks like (it is smooth along the edge).
    """
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    pc.colors = o3d.utility.Vector3dVector(rgb)
    pc, idx = pc.remove_statistical_outlier(nb_neighbors=k, std_ratio=std)
    return np.asarray(pc.points), np.asarray(pc.colors), len(idx)


def depth_normals(Z, K):
    """Unit camera-space normals estimated from the depth map.

    From the unprojected neighbourhood, not from image intensity: the cross
    product of the two surface tangents. Central differences, so the result is
    one pixel blurrier than the depth -- which is fine here, because it is used
    to REJECT grazing pixels and a slightly smoothed normal is the conservative
    direction for that test.
    """
    H, W = Z.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    uv = np.stack([u.ravel(), v.ravel()], 1).astype(np.float64)
    P = (Z.ravel()[:, None] * ray_dirs(uv, K[0,0], K[0,2], K[1,2])).reshape(H, W, 3)
    dx = np.zeros_like(P)
    dy = np.zeros_like(P)
    dx[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dy[1:-1, :] = P[2:, :] - P[:-2, :]
    n = np.cross(dx, dy)
    ln = np.linalg.norm(n, axis=2, keepdims=True)
    n = n / np.maximum(ln, 1e-12)
    view = P / np.maximum(np.linalg.norm(P, axis=2, keepdims=True), 1e-12)
    facing = np.abs(np.sum(n * view, axis=2))      # |cos| to the view ray
    return n, facing


def clean_pixel_mask(Z, K, keep, grad_rel=GRAD_REL, min_facing=MIN_FACING):
    """Drop pixels that straddle a depth discontinuity or graze the view.

    Returns a mask, not a filtered cloud, so the caller can see what each stage
    cost -- a filter that silently removes 80% of the fighter is a bug, and a
    count is the only way that shows up before the render does.
    """
    stats = {"in": int(keep.sum())}
    gx = cv2.Sobel(Z, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(Z, cv2.CV_64F, 0, 1, ksize=3)
    rel = np.hypot(gx, gy) / np.maximum(Z, 1e-6)
    keep = keep & (rel < grad_rel)
    stats["after_gradient"] = int(keep.sum())
    _, facing = depth_normals(Z, K)
    keep = keep & (facing > min_facing)
    stats["after_grazing"] = int(keep.sum())
    return keep, stats