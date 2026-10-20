import cv2
import numpy as np

from rendering.constants import ERODE_PX
from geometry.pose import rodrigues
from geometry.projections import ray_dirs
from geometry.filtering import clean_pixel_mask, sor


class View:
    """Pinhole camera in world coordinates: X_camera = R @ (X_world - C).

    R is world-to-camera and C is the camera center in world coordinates.
    size is (width, height).
    """

    def __init__(self, K, R, C, size):
        self.K = np.asarray(K, np.float64)
        self.R = np.asarray(R, np.float64)
        self.C = np.asarray(C, np.float64)
        self.W, self.H = size

    @property
    def f(self):
        return float(self.K[0, 0])

    def project(self, Xw):
        """Return pixels and camera-axis depth; z <= 0 means behind."""
        Xc = (np.asarray(Xw, np.float64) - self.C) @ self.R.T
        z = Xc[:, 2]
        safe = np.where(np.abs(z) < 1e-9, 1e-9, z)

        uv = np.stack([
            self.K[0, 0] * Xc[:, 0] / safe + self.K[0, 2],
            self.K[1, 1] * Xc[:, 1] / safe + self.K[1, 2],
        ], axis=1)

        return uv, z


class FisheyeView:
    """Equidistant fisheye with the same world-to-camera convention as View.

    Projects directly into the fisheye image, avoiding an intermediate
    pinhole render and resampling. A single pinhole image cannot cover
    a full 180-degree field of view.

        radius_px = theta / (fov / 2) * image_radius

    f is pixels per radian. project() returns range for depth ordering
    and splat sizing, with negative depth outside the requested field.

    Circular splats approximate the projection footprint away from the axis.
    """

    def __init__(self, R, C, size=2048, fov_deg=180.0):
        self.R = np.asarray(R, np.float64)
        self.C = np.asarray(C, np.float64)
        self.S = int(size)
        self.W = self.H = self.S
        self.half = np.radians(float(fov_deg)) / 2.0
        self.cc = (self.S - 1) / 2.0
        self.radius_px = self.S / 2.0

    @property
    def f(self):
        return self.radius_px / self.half

    def project(self, Xw):
        Xc = (np.asarray(Xw, np.float64) - self.C) @ self.R.T
        rng = np.linalg.norm(Xc, axis=1)
        safe = np.maximum(rng, 1e-9)

        theta = np.arccos(np.clip(Xc[:, 2] / safe, -1.0, 1.0))
        phi = np.arctan2(Xc[:, 1], Xc[:, 0])
        rr = theta / self.half * self.radius_px

        uv = np.stack([
            self.cc + rr * np.cos(phi),
            self.cc + rr * np.sin(phi),
        ], axis=1)

        return uv, np.where(theta <= self.half, rng, -1.0)


class EquirectView:
    """Per-eye half-equirectangular projection for the front hemisphere.

    Longitude and latitude each cover +/-90 degrees. Camera axes are
    +X right, +Y down, +Z forward.

        direction = (
            cos(lat) * sin(lon),
            -sin(lat),
            cos(lat) * cos(lon),
        )
        u = (lon + pi/2) / pi * size
        v = (pi/2 - lat) / pi * size

    project() returns range, with negative depth outside the front hemisphere.

    f is pixels per radian at the equator. Circular splats underestimate
    horizontal coverage toward the poles because longitude stretches by
    1 / cos(latitude).

    Playback metadata must match the projection and angular coverage.
    Tagging fisheye pixels as equirectangular does not convert them.

    fov_deg is retained for interface compatibility; this view always
    covers 180 degrees.
    """

    def __init__(self, R, C, size=2048, fov_deg=180.0):
        self.R = np.asarray(R, np.float64)
        self.C = np.asarray(C, np.float64)
        self.S = int(size)
        self.W = self.H = self.S

    @property
    def f(self):
        return self.S / np.pi

    def project(self, Xw):
        Xc = (np.asarray(Xw, np.float64) - self.C) @ self.R.T
        rng = np.linalg.norm(Xc, axis=1)
        d = Xc / np.maximum(rng, 1e-9)[:, None]

        lat = np.arcsin(np.clip(-d[:, 1], -1.0, 1.0))
        lon = np.arctan2(d[:, 0], d[:, 2])

        uv = np.stack([
            (lon + np.pi / 2) / np.pi * self.S,
            (np.pi / 2 - lat) / np.pi * self.S,
        ], axis=1)

        return uv, np.where(Xc[:, 2] > 0, rng, -1.0)


def make_view(projection, R, C, size, fov_deg=180.0):
    """Create a spherical target view independently of the renderer."""
    if projection == "fisheye":
        return FisheyeView(R, C, size, fov_deg)
    if projection == "equirect":
        return EquirectView(R, C, size, fov_deg)

    raise ValueError(f"unknown projection {projection!r}")


def stereo_views(R, C, ipd, size, fov, projection="equirect"):
    """Create parallel left/right views separated by ipd in world units.

    R[0] is camera-right in world coordinates. Both eyes share the same orientation 
    and angular coordinate frame. Separate their centers along camera-right 
    to produce stereo. Do not independently aim each eye at the subject.

    Stereo disparity comes from the reconstructed geometry. A single-view
    surface can expose holes near silhouettes when viewed from either eye,
    producing disagreement between the eye images.

    ipd=0 produces coincident cameras and should reproduce the mono render.
    """
    right = np.asarray(R, np.float64)[0]
    C = np.asarray(C, np.float64)

    return (
        make_view(projection, R, C - 0.5 * ipd * right, size, fov),
        make_view(projection, R, C + 0.5 * ipd * right, size, fov),
    )


def to_sbs(img):
    """Duplicate a mono image into side-by-side layout."""
    return np.hstack([img, img])


class Renderer:
    """Extract fighter clouds and rasterize world points into a target view.

    The target view and splat settings are fixed at construction.
    Source images, depth, masks and poses are supplied per frame.

    fighter_cloud():
        BGR source image -> world points and RGB float colors.

    render():
        World points and RGB colors -> RGB float image, depth and coverage.

    Rendering uses two passes: resolve nearest depth first, then write
    colors at that depth. Disc size follows f * world_radius / depth,
    clipped to the configured pixel limit.
    """

    def __init__(
        self,
        view,
        world_radius=0.012,
        max_px=6,
        erode_px=ERODE_PX,
        clean=True,
    ):
        self.view = view
        self.world_radius = world_radius
        self.max_px = max_px
        self.erode_px = erode_px
        self.clean = clean

    def fighter_cloud(self, person_mask, image, metric_depth, K, omega, C):
        """Return (world_points, rgb_colors, stats) for one source frame.

        image is BGR. metric_depth is camera-axis Z in metres.
        K and person_mask must correspond to the source image grid.

        Depth is resized with nearest-neighbor sampling when dimensions
        differ, assuming it covers the same image extent.

        Mask erosion, pixel filtering and statistical outlier removal
        reduce boundary contamination and isolated depth artifacts.
        """
        if metric_depth is None or image is None:
            return None, None, {}

        if metric_depth.shape[:2] != image.shape[:2]:
            metric_depth = cv2.resize(
                metric_depth,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        if person_mask is None:
            return None, None, {"reason": "no person mask"}

        if self.erode_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * self.erode_px + 1,) * 2,
            )
            person_mask = cv2.erode(
                person_mask.astype(np.uint8), kernel
            ) > 0

        keep = (
            person_mask
            & np.isfinite(metric_depth)
            & (metric_depth > 0.5)
            & (metric_depth < 30.0)
        )
        stats = {"masked": int(keep.sum())}

        if self.clean:
            keep, cleaning_stats = clean_pixel_mask(metric_depth, K, keep)
            stats.update(cleaning_stats)

        if not keep.any():
            return None, None, stats

        ys, xs = np.nonzero(keep)
        uv = np.stack([xs, ys], axis=1).astype(np.float64)

        Xc = metric_depth[ys, xs][:, None] * ray_dirs(
            uv, K[0, 0], K[0, 2], K[1, 2]
        )
        Xw = np.asarray(C, np.float64) + Xc @ rodrigues(omega)
        rgb = image[ys, xs, ::-1].astype(np.float64) / 255.0

        if self.clean:
            Xw, rgb, count = sor(Xw, rgb)
            stats["after_sor"] = count

        stats["out"] = len(Xw)
        return Xw, rgb, stats

    @staticmethod
    def _disc(radius):
        r = int(np.ceil(radius))
        offsets = [
            (dx, dy)
            for dy in range(-r, r + 1)
            for dx in range(-r, r + 1)
            if dx * dx + dy * dy <= radius * radius
        ]
        return offsets or [(0, 0)]

    def render(self, Xw, rgb):
        """Return (color, depth, coverage) for one point cloud.

        Xw:  (N, 3) world points.
        rgb: (N, 3) RGB float colors in [0, 1].

        color is an RGB float image with black uncovered pixels.
        Uncovered depth is infinity. Depth means camera Z for View and
        range for FisheyeView/EquirectView.

        Empty clouds must have shape (0, 3).
        """
        view = self.view
        world_radius = self.world_radius
        max_px = self.max_px

        H, W = view.H, view.W
        depth = np.full((H, W), np.inf)
        color = np.zeros((H, W, 3))
        cover = np.zeros((H, W), bool)

        uv, z = view.project(Xw)
        ok = z > 1e-6
        if not ok.any():
            return color, depth, cover

        uv, z, rgb = uv[ok], z[ok], np.asarray(rgb)[ok]
        rpx = np.clip(view.f * world_radius / z, 0.7, max_px)
        x0 = np.round(uv[:, 0]).astype(int)
        y0 = np.round(uv[:, 1]).astype(int)

        order = np.argsort(np.round(rpx).astype(int))
        groups = np.split(
            order,
            np.flatnonzero(
                np.diff(np.round(rpx[order]).astype(int))
            ) + 1,
        )

        # Pass 1: nearest depth at every covered pixel.
        for g in groups:
            if not len(g):
                continue

            radius = max(float(np.round(rpx[g[0]])), 0.7)
            for dx, dy in self._disc(radius):
                X, Y = x0[g] + dx, y0[g] + dy
                inb = (X >= 0) & (X < W) & (Y >= 0) & (Y < H)
                if not inb.any():
                    continue

                np.minimum.at(depth, (Y[inb], X[inb]), z[g][inb])

        # Pass 2: write only points that match the nearest depth.
        for g in groups:
            if not len(g):
                continue

            radius = max(float(np.round(rpx[g[0]])), 0.7)
            for dx, dy in self._disc(radius):
                X, Y = x0[g] + dx, y0[g] + dy
                inb = (X >= 0) & (X < W) & (Y >= 0) & (Y < H)
                if not inb.any():
                    continue

                Xi, Yi, zi = X[inb], Y[inb], z[g][inb]
                win = zi <= depth[Yi, Xi] + 1e-9

                color[Yi[win], Xi[win]] = rgb[g][inb][win]
                cover[Yi[win], Xi[win]] = True

        return color, depth, cover