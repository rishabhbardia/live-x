import numpy as np
import cv2

from typing import Any, Dict, Tuple
from data.types import SurfaceType

from geometry.scene import Scene
from geometry.projections import plane_ray_depth, project
from geometry.pose import rodrigues

from tracker.constants import BAND_TOP_FRAC, BAND_HEIGHT_FRAC


from collections.abc import Mapping

DILATE = 12

def post_centres_and_half_widths(
        thetas: Mapping[int, float],
        post_widths: Mapping[tuple[int, int], float],
) -> tuple[dict[int, float], dict[int, float]]:
    """
        ({a: centre_rad}, {a: half_width_m}) per edge pair (a, b), keyed by `a`.

        Centre is the midpoint angle of the pair's two edges; half-width is half of
        the pair's arc width. 
    """
    centres: dict[int, float] = {}
    half: dict[int, float] = {}
    for (a, b), w in post_widths.items():
        gap = (thetas[b] - thetas[a] + np.pi) % (2 * np.pi) - np.pi
        centres[a] = thetas[a] + gap / 2.0
        half[a] = w / 2.0
    return centres, half



class StageSurfaces:

    def __init__(self, 
                 clip:Any, # really a clip bundle
                 scene: Scene, 
                 tracked_points: Tuple[Dict[str, Tuple[np.ndarray, Tuple]]],
                 post_thetas:  Mapping[int, float],
                 post_widths: Mapping[tuple[int, int], float],
                 post_half_width: float,
                 image_wh: tuple[int,int]):

        self.radius = clip.clip_payload.venue_measurements.radius + scene.delta_radius[0]
        self.height = clip.clip_payload.venue_measurements.pillar_height + scene.delta_height[0]
        self.W, self.H = image_wh
        self.scene = scene
        self.mean_post_half_width = post_half_width
        self.n_posts = clip.clip_payload.venue_measurements.total_posts

        # radians   , meters
        self.post_theta_centers, self.post_half_width = post_centres_and_half_widths(
            thetas=post_thetas,
            post_widths=post_widths
        )
        for tc, hw in zip(self.post_theta_centers, self.post_half_width):
            print(f"Theta mid post {self.post_theta_centers[tc]:.4f}, half width {self.post_half_width[hw]:.4f}")

        # We are constructing the band where there's labels on the cage top end
        # this is so that the features can be used reliably
        self.band_radius = self.radius # assume top ring to be the same
        self.band_z1 = float(BAND_TOP_FRAC) * self.height
        self.band_z0 = self.band_z1 - float(BAND_HEIGHT_FRAC) * self.height

        order = sorted(self.post_theta_centers.items(), key=lambda kv: kv[1])
        self.band_panels = {(a, b): (self.post_theta_centers[a], self.post_theta_centers[b])
                             for (a, _), (b, _) in zip(order[:-1], order[1:])}

        self._cache_key = None
        self._cache = None

    # ----------------- Static World Geometry ----------------
    
    def canvas_boundary(self, n=256):
        """
            Canvas points, due to the structure of the world they are
            at z=0 and the cage is a circle which is the only thing we care about.
        """
        a = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        return np.stack([self.radius * np.cos(a), self.radius * np.sin(a),
                         np.zeros_like(a)], axis=1)

    
    def canvas_basis(self):
        return (np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))
    
    def _post_geometry(self, theta, half_width):
        """
            (O, n_world, quad) for a post face at angle `theta`
            O is the base of the post's face on the boundary circle.
            `e1` (tangential) and `e2` (vertical) are the same basis `post_basis / band_basis`
            builds for the per-edge landmark system, so a post face here and a
            pillar-plane landmark there describe the same physical surface.
            
        """

        O = np.array([self.radius * np.cos(theta), self.radius * np.sin(theta), 0.0])
        e1 = np.array([-np.sin(theta), np.cos(theta), 0.0])
        e2 = np.array([0.0, 0.0, 1.0])
        n = np.cross(e2, e1)

        quad = np.stack([O - half_width * e1,
                         O + half_width * e1,
                         O + half_width * e1 + self.height * e2,
                         O - half_width * e1 + self.height * e2])
        return O, n, quad


    def post_quad(self, i, theta=None, half_width=None):
        """
            (4, 3) world corners of a post face, base-to-top, one tangential
            side to the other.
            `i` is one of `self.post_theta_centers`'s calibrated posts by default, and
            then also supplies its real derived half-width.
        """
        if theta is None:
            if i not in self.post_theta_centers:
                raise KeyError(f"post {i!r} has no calibrated theta -- pass "
                               f"theta= explicitly for a predicted post")
            theta = self.post_theta_centers[i]
            if half_width is None:
                half_width = self.post_half_width[i]
        hw = self.mean_post_half_width if half_width is None else half_width
        return self._post_geometry(theta, hw)[2]

    def _band_geometry(self, theta_a, theta_b):
        """
            (O, n_world, quad) for the flat band panel spanning two posts.

        """
        ca = np.array([self.radius * np.cos(theta_a), self.radius * np.sin(theta_a), 0.0])
        cb = np.array([self.radius * np.cos(theta_b), self.radius * np.sin(theta_b), 0.0])
        chord = cb - ca
        e1 = chord / (np.linalg.norm(chord) or 1.0)
        e2 = np.array([0.0, 0.0, 1.0])
        n = np.cross(e2, e1)
        O = ca + self.band_z0 * e2
        quad = np.stack([O, O + chord, O + chord + (self.band_z1 - self.band_z0) * e2,
                         O + (self.band_z1 - self.band_z0) * e2])
        return O, n, quad
    
    def band_basis(self):
        """
            {(post_a, post_b): (O, e1, e2)} -- center, plane basis 1 and 2 
            (see `plane_warp` for useage).
        """
        out = {}
        for key, (ta, tb) in self.band_panels.items():
            O, _, quad = self._band_geometry(ta, tb)
            chord = quad[1] - quad[0]
            out[key] = (O, chord / (np.linalg.norm(chord) or 1.0),
                        np.array([0.0, 0.0, 1.0]))
        return out

    def post_basis(self) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
            {post_id: (O, e1, e2)} for each calibrated post's face: O is the face's base
            centre on the cage circle, e1 runs tangentially, e2 is vertical.
        """
        out = {}
        for i, theta in self.post_theta_centers.items():
            O, _, quad = self._post_geometry(theta, self.post_half_width[i])
            tangent = quad[1] - quad[0]
            out[i] = (O, tangent / (np.linalg.norm(tangent) or 1.0), np.array([0.0, 0.0, 1.0]))
        return out
    
    def post_normals(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        out = {}
        for i, theta in self.post_theta_centers.items():
            O, N, _ = self._post_geometry(theta, self.post_half_width[i])
            out[i] = (O, N)
        return out


    def predicted_posts(self, n_posts: int|None = None):
        """
            {k: theta} for every post around the cage, k = 0 at the first
            calibrated post, extrapolated by the measured angular spacing.
        """
        if n_posts is None:
            n_posts = self.n_posts
        ids_thetas = sorted(self.post_theta_centers.items())
        thetas = np.array([th for _, th in ids_thetas])
        dtheta = (float(np.mean(np.diff(thetas))) if len(thetas) > 1 else -2 * np.pi / n_posts)
        th0 = thetas[0]
        return {k: th0 + k * dtheta for k in range(n_posts)}

    # ----------------- Per-Frame ----------------
    def _facing(self, n_world, O, C):
        """
            True if the pad-normal at `O` is turned toward the camera centre
            `C` -- False means this post's pad is turned away and cannot be seen
            from `C`, regardless of which side of the ring either sits on.
        """
        return float(np.dot(n_world, np.asarray(C, np.float64) - O)) > 0.0

    def _fill(self, uv):
        """
            Rasterise one projected polygon into a bool (H, W) mask. Fewer
            than 3 finite points (off-image, behind-camera, degenerate) fills
            nothing rather than raising -- a surface that cannot be drawn this
            frame is simply not visible this frame.
        """
        m = np.zeros((self.H, self.W), np.uint8)
        uv = np.asarray(uv, np.float64)
        if len(uv) < 3 or not np.all(np.isfinite(uv)):
            return m.astype(bool)
        pts = np.round(uv).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(m, [pts], 1)
        return m.astype(bool)

    def _plane_ray_t(self, f, cx, cy, omega, C, O, n_world, uv):
        """
            Camera-space depth at which each pixel's ray meets the world plane
            through `O` with normal `n_world`.
        """
        return plane_ray_depth(uv, rodrigues(omega), C, f, cx, cy, O, n_world)
    
    def _depth_image(self, mask, O, n_world, omega, C):
        """
            (H, W) camera-space depth, `_plane_ray_t` evaluated at every pixel
            inside `mask` and `np.inf` elsewhere -- a surface with no depth entry
            at a pixel never wins a nearer-than comparison there.
        """
        depth = np.full((self.H, self.W), np.inf, np.float64)
        ys, xs = np.nonzero(mask)
        if len(xs):
            uv = np.stack([xs, ys], axis=1).astype(np.float64)
            f, cx, cy = self.scene.intr_of(0) # Note just using 1 focal length here could bite
            depth[ys, xs] = self._plane_ray_t(f, cx, cy, omega, C, O, n_world, uv)
        return depth

    def _key(self, omega, C, frame_idx):
        return (frame_idx, tuple(np.round(np.asarray(omega, np.float64), 6)),
                tuple(np.round(np.asarray(C, np.float64), 6)))

    def _project(self, f, cx, cy, omega, C, X):
        R = rodrigues(omega=omega)
        return project(np.atleast_2d(X), R, C, f, cx, cy)
    
    def _state(self, omega, C, frame_idx, clip):
        
        key = self._key(omega, C, frame_idx)
        if self._cache_key == key:
            return self._cache

        f, cx, cy = self.scene.intr_of(0) # Note just using 1 focal length here could bite

        # (key, footprint, depth) for every surface that renders at all
        surfaces = {}
        canvas_O, canvas_n = np.zeros(3), np.array([0.0, 0.0, 1.0])
        ring = self.canvas_boundary()

        uv, z = self._project(f, cx, cy, omega, C, ring)
        valid = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1]) & (z > 0)

        if valid.sum() >= 3:
            fp = self._fill(uv[valid])
            if fp.any():
                surfaces[(SurfaceType.CANVAS,)] = (
                    fp, self._depth_image(fp, canvas_O, canvas_n, omega, C))
        
        def add_quad(skey, O, n, quad):
            if not self._facing(n, O, C):
                return                        # pad turned away from the camera
            uvq, zq = self._project(f, cx, cy, omega, C, quad)
            if not (np.all(zq > 0) and np.all(np.isfinite(uvq))):
                return
            fp = self._fill(uvq)
            if fp.any():
                surfaces[skey] = (fp, self._depth_image(fp, O, n, omega, C))
        
        for i, theta in self.post_theta_centers.items():
            add_quad(
                (SurfaceType.POST, i),
                *self._post_geometry(theta=theta, half_width=self.post_half_width[i])
            )
        for pkey, (ta, tb) in self.band_panels.items():
            add_quad(
                (SurfaceType.BAND,  pkey),
                *self._band_geometry(ta, tb)
            )
        person = clip.get_person_mask(frame_idx, DILATE)
        final = {}
        for skey, (fp, depth) in surfaces.items():
            occ = np.zeros_like(fp)
            for okey, (ofp, odepth) in surfaces.items():
                if okey != skey:
                    occ |= ofp & (odepth < depth)
            m = fp & ~occ
            if person is not None:
                m = m & ~person
            final[skey] = m
        self._cache_key = key
        self._cache = final 
        return self._cache

    def _empty(self):
        return np.zeros((self.H, self.W), bool)
    
    def canvas_mask(self, omega, C, frame_idx, clip):
        return self._state(omega=omega, C=C, frame_idx=frame_idx, clip=clip).get((SurfaceType.CANVAS,), self._empty())

    def post_mask(self, i, omega, C, frame_idx, clip):
        return self._state(omega, C, frame_idx, clip=clip).get((SurfaceType.POST, i), self._empty())

    def band_mask(self, panel, omega, C, frame_idx, clip):
        return self._state(omega, C, frame_idx, clip=clip).get((SurfaceType.BAND, panel),
                                                    self._empty())

    def mask(self, surface, omega, C, frame_idx, clip, **kw):
        """
            Dispatch on `surface` (CANVAS / POST / BAND); POST takes `i=`,
            BAND takes `panel=`.
        """
        if surface == SurfaceType.CANVAS:
            return self.canvas_mask(omega, C, frame_idx, clip)
        if surface == SurfaceType.POST:
            return self.post_mask(kw["i"], omega, C, frame_idx, clip)
        if surface == SurfaceType.BAND:
            return self.band_mask(kw["panel"], omega, C, frame_idx, clip)
        raise ValueError(f"unknown surface {surface!r}")

    def all_masks(self, omega, C, frame_idx, clip):
        return dict(self._state(omega, C, frame_idx, clip))