import pickle
import numpy as np
from pathlib import Path
from geometry.scene import Scene
from geometry.depth import DepthMetricFit

from tracker.tracker import Tracker



def save_scene(path: Path, scene: Scene) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as fh:
        pickle.dump(scene, fh, protocol=pickle.HIGHEST_PROTOCOL)

def save_tracker(path: Path, tracker: Tracker) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": 1,
        "tracker": tracker,
    }
    with path.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)


def save_depth_metric_fit(path: Path, fit: DepthMetricFit) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as fh:
        pickle.dump(fit, fh, protocol=pickle.HIGHEST_PROTOCOL)


