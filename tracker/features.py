import cv2
import numpy as np

from tracker.constants import LK


def gray_image(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def create_mser():
    return cv2.MSER_create()


def on_person(
    person: np.ndarray | None,
    points: np.ndarray,
) -> np.ndarray:
    """Return whether each finite, in-image point hits the person mask."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    result = np.zeros(len(points), dtype=bool)

    if person is None or not len(points):
        return result

    h, w = person.shape
    finite = np.isfinite(points).all(axis=1)
    indices = np.flatnonzero(finite)

    xy = np.rint(points[indices]).astype(np.int64)
    inside = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < w)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < h)
    )

    indices = indices[inside]
    xy = xy[inside]
    result[indices] = person[xy[:, 1], xy[:, 0]].astype(bool)
    return result


def lk_hop(
    previous: np.ndarray,
    current: np.ndarray,
    points: np.ndarray,
    levels: int,
    win: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward/backward LK, returning positions, round-trip errors and status."""
    p0 = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    count = len(p0)

    positions = np.full((count, 2), np.nan, dtype=np.float64)
    errors = np.full(count, np.inf, dtype=np.float64)
    valid = np.zeros(count, dtype=bool)

    if not count:
        return positions, errors, valid

    options = dict(LK, maxLevel=levels, winSize=win)
    p1, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        previous, current, p0, None, **options
    )
    if p1 is None or forward_status is None:
        return positions, errors, valid

    positions[:] = p1.reshape(-1, 2)
    forward_ok = (
        forward_status.ravel().astype(bool)
        & np.isfinite(positions).all(axis=1)
    )
    indices = np.flatnonzero(forward_ok)

    if not len(indices):
        return positions, errors, valid

    # Do not pass failed/nonfinite forward tracks into the backward solve.
    backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current, previous, p1[indices], None, **options
    )
    if backward is None or backward_status is None:
        return positions, errors, valid

    round_trip = np.linalg.norm(
        p0[indices].reshape(-1, 2) - backward.reshape(-1, 2),
        axis=1,
    )
    errors[indices] = round_trip
    valid[indices] = (
        backward_status.ravel().astype(bool)
        & np.isfinite(round_trip)
    )
    return positions, errors, valid


def track_points(
    previous: np.ndarray,
    current: np.ndarray,
    points: np.ndarray,
    person: np.ndarray | None,
    levels: int,
    win: tuple[int, int],
    fb_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """LK with round-trip, image-boundary and person-mask filtering."""
    positions, errors, valid = lk_hop(
        previous, current, points, levels, win
    )
    h, w = current.shape

    valid &= (
        np.isfinite(positions).all(axis=1)
        & (errors <= fb_tolerance)
        & (positions[:, 0] >= 0)
        & (positions[:, 0] < w)
        & (positions[:, 1] >= 0)
        & (positions[:, 1] < h)
        & ~on_person(person, positions)
    )
    return positions, valid


def detect_mser(gray, mask=None, mser=None):
    """Detect MSER regions. Optional mask filter"""
    if mser is None:
        mser = create_mser()

    regions, _ = mser.detectRegions(gray)
    if mask is None:
        return list(regions)

    h, w = mask.shape
    kept = []

    for points in regions:
        x, y = np.rint(points.mean(axis=0)).astype(int)
        if 0 <= x < w and 0 <= y < h and mask[y, x]:
            kept.append(points)

    return kept


def pick_replenish_candidates(
    gray,
    mask,
    existing_uv,
    mser,
    min_sep,
    max_n,
    quality=0.01,
):
    """Pick separated corners inside masked MSER regions."""
    if max_n <= 0:
        return []

    mask = np.asarray(mask, dtype=bool)
    regions = detect_mser(gray, mask, mser)
    if not regions:
        return []

    blob = np.zeros(gray.shape, dtype=np.uint8)
    for points in regions:
        hull = cv2.convexHull(points.astype(np.int32))
        cv2.fillPoly(blob, [hull], 1)

    blob[~mask] = 0
    if not blob.any():
        return []

    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max(4 * max_n, 8),
        qualityLevel=quality,
        minDistance=float(min_sep),
        mask=blob,
    )
    if corners is None:
        return []

    occupied = [
        uv.copy()
        for uv in np.asarray(existing_uv, dtype=np.float64).reshape(-1, 2)
    ]
    accepted = []

    for uv in corners.reshape(-1, 2).astype(np.float64):
        if occupied:
            distances = np.linalg.norm(np.asarray(occupied) - uv, axis=1)
            if np.any(distances < min_sep):
                continue

        accepted.append(uv)
        occupied.append(uv)

        if len(accepted) >= max_n:
            break

    return accepted


def warp_reference(
    reference: np.ndarray,
    homography: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    h, w = shape
    return cv2.warpPerspective(reference, homography, (w, h))


def reacquire(
    gray,
    warped_reference,
    uv_pred,
    window,
    ncc_min,
    patch,
    person=None,
):
    """Match a warped reference patch near its predicted image position."""
    if not np.isfinite(uv_pred).all():
        return None

    h, w = gray.shape
    cx, cy = np.rint(uv_pred).astype(int)

    tx0, ty0 = cx - patch, cy - patch
    tx1, ty1 = cx + patch + 1, cy + patch + 1

    if tx0 < 0 or ty0 < 0 or tx1 > w or ty1 > h:
        return None

    template = warped_reference[ty0:ty1, tx0:tx1]
    if not template.size or float(template.std()) < 3.0:
        return None

    x0 = max(cx - window - patch, 0)
    y0 = max(cy - window - patch, 0)
    x1 = min(cx + window + patch + 1, w)
    y1 = min(cy + window + patch + 1, h)

    roi = gray[y0:y1, x0:x1]
    if (
        roi.shape[0] < template.shape[0]
        or roi.shape[1] < template.shape[1]
    ):
        return None

    scores = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    scores[~np.isfinite(scores)] = -1.0

    if person is not None:
        rows, cols = scores.shape
        blocked = person[
            y0 + patch:y0 + patch + rows,
            x0 + patch:x0 + patch + cols,
        ].astype(bool)
        scores[blocked] = -1.0

    _, best, _, location = cv2.minMaxLoc(scores)
    if best < ncc_min:
        return None

    uv = np.array(
        [x0 + location[0] + patch, y0 + location[1] + patch],
        dtype=np.float64,
    )
    return uv, float(best)