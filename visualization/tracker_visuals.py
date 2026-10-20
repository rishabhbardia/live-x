from collections import deque
from pathlib import Path

import cv2
import numpy as np

from interface.clip import ClipBundle
from tracker.tracker import Tracker


def visualize_tracks(
    clip: ClipBundle,
    tracker: Tracker,
    output_dir: Path,
    *,
    trail_frames: int = 25,
    name_prefix: str | None = None,
    include_seed: bool = True,
) -> None:
    """Write tracking overlays in chronological frame order.

    Green: active tracked observations and their recent image-space trails.
    Magenta: manual annotations, where available.
    Drift: tracked-to-annotated pixel distance for matching landmark names.

    name_prefix="Q" shows only original canvas landmarks.
    name_prefix=None shows all active landmarks.

    Does not modify the clip or tracker.
    """
    if trail_frames < 1:
        raise ValueError("trail_frames must be positive")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations = clip.get_clip().annotations

    # A repeated pass may contain the same frame more than once.
    # Use its most recently recorded result.
    records = {
        record["stem"]: record
        for record in tracker.records
    }

    if include_seed and tracker.seed_frame_key not in records:
        records[tracker.seed_frame_key] = {
            "stem": tracker.seed_frame_key,
            "frame_index": tracker.seed_frame_index,
            "obs": tracker._seed_positions,
        }

    ordered = sorted(
        records.values(),
        key=lambda record: record["frame_index"],
    )

    def selected(name):
        return name_prefix is None or name.startswith(name_prefix)

    def valid_points(points):
        result = {}
        for name, value in points.items():
            if not selected(name):
                continue

            uv = np.asarray(value, dtype=np.float64)
            if uv.shape == (2,) and np.isfinite(uv).all():
                result[name] = uv

        return result

    def pixel(uv):
        return tuple(np.rint(uv).astype(int))

    def text(image, label, position, color, scale=0.65):
        # Dark outline keeps text readable over both canvas and fighters.
        cv2.putText(
            image, label, position,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale, (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            image, label, position,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale, color, 1, cv2.LINE_AA,
        )

    green = (60, 230, 60)
    magenta = (255, 0, 255)
    white = (255, 255, 255)

    base_names = {
        name for name in tracker.base_names if selected(name)
    }
    trails = {}
    previous_index = None
    stride = abs(tracker.step)

    for record in ordered:
        stem = record["stem"]
        index = record["frame_index"]

        if clip.frame_index_key(index) != stem:
            raise ValueError(
                f"Frame mapping mismatch: index {index}, record {stem!r}"
            )

        rgb = clip.get_frame(index).broadcast_frame
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        observed = valid_points(record["obs"])
        annotation = annotations.get(stem)
        annotated = valid_points(
            annotation.to_dict() if annotation is not None else {}
        )

        # Do not draw a line across unrecorded sections of the clip.
        if (
            previous_index is not None
            and index - previous_index != stride
        ):
            trails.clear()

        # A lost track ends its trail. Reacquisition starts a new one.
        for name in list(trails):
            if name not in observed:
                del trails[name]

        for name, uv in observed.items():
            history = trails.setdefault(name, deque())
            history.append((index, uv.copy()))

            while history and history[0][0] < index - trail_frames + 1:
                history.popleft()

            if len(history) >= 2:
                path = np.asarray(
                    [point for _, point in history],
                    dtype=np.float64,
                )
                path = np.rint(path).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(
                    image, [path], False, green, 1, cv2.LINE_AA
                )

            x, y = pixel(uv)
            cv2.circle(image, (x, y), 5, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(image, (x, y), 4, green, -1, cv2.LINE_AA)

            # Keep the full name: Q3, P0, MQ17, etc.
            text(image, name, (x + 8, y - 8), green, scale=0.45)

        for uv in annotated.values():
            cv2.drawMarker(
                image,
                pixel(uv),
                magenta,
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=17,
                thickness=2,
                line_type=cv2.LINE_AA,
            )

        matched = observed.keys() & annotated.keys()
        drift = [
            np.linalg.norm(observed[name] - annotated[name])
            for name in matched
        ]

        n_base = len(observed.keys() & base_names)
        n_new = len(observed.keys() - base_names)

        heading = f"{stem}   tracked {n_base}/{len(base_names)}"
        if n_new:
            heading += f"   new {n_new}"

        if annotated:
            heading += f"   annotated {len(annotated)}"
        if drift:
            heading += f"   drift median {np.median(drift):.2f} px"

        text(image, heading, (18, 28), white)
        text(
            image,
            f"green = tracked (trail = last {trail_frames} frames)"
            "   magenta = annotation",
            (18, 57),
            white,
        )

        destination = output_dir / f"{stem}_tracks.png"
        if not cv2.imwrite(str(destination), image):
            raise OSError(f"Could not write {destination}")

        previous_index = index