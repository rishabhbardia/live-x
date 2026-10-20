from pathlib import Path
from types import MappingProxyType

STAGE_FILE = "calibrated_stage.pkl"
TRACKER_FILE = "tracked.pkl"
DEPTH_METRIC_FILE = "depth_metric.pkl"
CALIBRATION_SIGMA = "calibration_sigmas.json"

def _index_files(folder: Path, suffixes: set[str]) -> dict[int, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(folder)

    indexed = {}

    for path in folder.iterdir():
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in suffixes:
            continue

        digits = path.stem.removeprefix("frame_")
        if not path.stem.startswith("frame_") or not digits.isdecimal():
            raise ValueError(f"Expected frame_<number> filename: {path}")

        frame_id = int(digits)
        if frame_id in indexed:
            raise ValueError(f"Duplicate frame {frame_id} in {folder}")

        indexed[frame_id] = path

    return indexed


class PathHandler:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)

        self.FRAMES_FOLDER = self.output_dir / "frames"
        self.DEPTH_FOLDER = self.output_dir / "depth"
        self.SEGMENTATION_FOLDER = self.output_dir / "segmentation"
        self.BODY_POSE_FOLDER = self.output_dir / "body-pose"
        self.ANNOTATION_FILE = (
            self.output_dir / "annotation" / "rigid_geometry.json"
        )

        self.ANNOTATION_PRIORS = self.output_dir / "stage.json"
        self.VENUE_FILE = self.output_dir / "venue.json"

        self.CALIBRATED_STAGE_FILE = self.output_dir / STAGE_FILE
        self.TRACKER_STATE_FILE = self.output_dir / TRACKER_FILE

        self.DEPTH_METRIC_FILE = self.output_dir / DEPTH_METRIC_FILE
        
        frames = _index_files(
            self.FRAMES_FOLDER, {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        )
        depths = _index_files(self.DEPTH_FOLDER, {".npz"})
        segments = _index_files(self.SEGMENTATION_FOLDER, {".pkl"})
        poses = _index_files(self.BODY_POSE_FOLDER, {".pkl"})

        if not frames:
            raise ValueError(f"No broadcast frames in {self.FRAMES_FOLDER}")

        expected = set(frames)

        for name, files in (
            ("depth", depths),
            ("segmentation", segments),
            ("body-pose", poses),
        ):
            actual = set(files)
            if actual != expected:
                raise ValueError(
                    f"{name} frame IDs do not match broadcast frames: "
                    f"missing={sorted(expected - actual)}, "
                    f"extra={sorted(actual - expected)}"
                )

    
        self.frame_ids = tuple(sorted(expected))
        self.frame_files = tuple(frames[i] for i in self.frame_ids)
        self.depth_files = tuple(depths[i] for i in self.frame_ids)
        self.segmentation_files = tuple(segments[i] for i in self.frame_ids)
        self.body_pose_files = tuple(poses[i] for i in self.frame_ids)
        self.length = len(self.frame_ids)
        self.frame_idx_key = MappingProxyType({
                    frame_idx: frames[frame_idx].stem
                    for frame_idx in self.frame_ids
                })

    def _get_path(self, files: tuple[Path, ...], idx: int) -> Path:
        if not 0 <= idx < self.length:
            raise IndexError(
                f"Frame position {idx} outside [0, {self.length})"
            )
        return files[idx]

    @staticmethod
    def frame_key_to_index(frame_key: str) -> int:
        digits = frame_key.removeprefix("frame_")
        if not frame_key.startswith("frame_") or not digits.isdecimal():
            raise ValueError(f"Invalid frame key: {frame_key!r}")
        return int(digits)

    def get_frame_path(self, idx: int) -> Path:
        return self._get_path(self.frame_files, idx)

    def get_depth_path(self, idx: int) -> Path:
        return self._get_path(self.depth_files, idx)

    def get_segmentation_path(self, idx: int) -> Path:
        return self._get_path(self.segmentation_files, idx)

    def get_body_pose_path(self, idx: int) -> Path:
        return self._get_path(self.body_pose_files, idx)

    def get_annotations_file(self) -> Path | None:
        return self.ANNOTATION_FILE if self.ANNOTATION_FILE.is_file() else None

    def get_annotations_prior_file(self) -> Path | None:
        return self.ANNOTATION_PRIORS if self.ANNOTATION_PRIORS.is_file() else None

    def get_venue_file(self) -> Path | None:
            return self.VENUE_FILE if self.VENUE_FILE.is_file() else None
    
    def get_calibrated_stage_file(self) -> Path | None:
            return self.CALIBRATED_STAGE_FILE if self.CALIBRATED_STAGE_FILE.is_file() else None

    def get_tracker_file(self) -> Path | None:
            return self.TRACKER_STATE_FILE if self.TRACKER_STATE_FILE.is_file() else None

    def get_depth_metric_file(self) -> Path | None:
                return self.DEPTH_METRIC_FILE if self.DEPTH_METRIC_FILE.is_file() else None
    
    
    def len_src_data(self) -> int:
        return self.length

    def write(self, stage_name: str):
        raise NotImplementedError("Stage output allocation is not implemented")