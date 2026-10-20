
import numpy as np
from dataclasses import dataclass, field

from data.types import (
    AnnotationPriors
)



@dataclass(frozen=True)
class VenueMeasurements:
    """
    This is highly specialized to ONE FC venue for Kickboxing.
    It contains the published stage/cage dimensions.
    TODO : create appropriate abstraction later
    """

    diameter: float
    pillar_height: float
    shape: str
    total_posts: int
    radius: float = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "radius", self.diameter / 2.0)



def pillar_world(theta: np.ndarray, r_in: float, dR: float, 
                pillar_h: float, dh: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Builds world points of the pillar base and top for the theoretical cage/stage.
    Only tested for ONE FC

    Parameters
    ----------
    theta : float (P,)
        Each post's azimuth in the world frame, radians, measured in the x-y
        (canvas) plane from +x toward +y. P = posts, in `obs["pillar_ids"]`
        order. Free parameters on a circle.
    r_in : float
            Centre-to-post radius, in meters -- off the published figures.
    dR : float
        A prior on radius correction dR, in meters, added to `r_in` for every post at
        once (one number for the stage, not per post).
    pillar_h : float
            Published post height, in meters.This is the gauge length everything 
            metric inherits.
    dh : float
        A prior on post-height correction dH, in meters, added to `pillar_h` for
        every post at once. Pinned at 0 unless the solver frees it.

    Returns
    -------
    base : float (P, 3)
        Post bases in world metres, row i for `theta[i]`:
        [R_p cos theta_i, R_p sin theta_i, 0]. z is exactly 0 -- the bases
        define the canvas plane rather than being fitted to it.
    top : float (P, 3)
        Post tops, `base` + [0, 0, H_p]: same x, y, with z = H_p. Posts are
        vertical by construction, so no lean is modelled.
    """
    r = r_in + dR
    c, s = np.cos(theta), np.sin(theta)
    base = np.stack([r * c, r * s, np.zeros_like(c)], axis=1)
    return base, base + np.array([0.0, 0.0, pillar_h + dh])


class VenueGeometryStructure:
    """
    This is highly specialized to ONE FC venue for Kickboxing.
    It contains the geometrical layout of the stage so we can solve
    for the camera-pose.
    TODO : create appropriate abstraction later
    """

    def __init__(self, measurements: VenueMeasurements, priors: AnnotationPriors) -> None:
        self.measurements: VenueMeasurements = measurements
        self.priors: AnnotationPriors = priors
        self.theta_dir: float = +1.0
        self.pillar_spread_deg: float = 25.0 # it’s a starting value for free per-post angles, not cage geometry,
        self.dist: float = 8.0
        self.theta = [np.zeros(1) for _ in range(len(priors.pillar_ids))]
        th = self.theta_dir * np.radians(self.pillar_spread_deg) * np.arange(len(priors.pillar_ids))
        for i, t in enumerate(th):
            self.theta[i][0] = t
        return

    def thetas(self):
        return np.array([t[0] for t in self.theta])