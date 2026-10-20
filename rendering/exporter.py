import os
import subprocess
import sys
from pathlib import Path

import cv2


class SBSVideoExporter:
    """Encode rendered SBS frames, then optionally add audio and metadata.

    Frames are already arranged as [left | right]. Filenames must sort
    chronologically, such as 000000.png or frame_000000.png.

    FPS comes from the source video. OpenCV handles video encoding;
    ffmpeg adds audio without re-encoding the video stream.
    """

    def __init__(self, output_video_path, codec="avc1"):
        self.output_video_path = str(output_video_path)
        self.codec = codec

    @staticmethod
    def _source_metadata(original_video_path):
        cap = cv2.VideoCapture(str(original_video_path))
        try:
            if not cap.isOpened():
                raise RuntimeError(
                    f"Could not open source video: {original_video_path}"
                )

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()

        return fps, frame_count

    @staticmethod
    def _frame_paths(frames_dir):
        paths = sorted(
            Path(frames_dir) / name
            for name in os.listdir(frames_dir)
            if name.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        if not paths:
            raise RuntimeError(f"No frames found in {frames_dir}")
        return paths

    def export(self, original_video_path, frames_dir):
        """Encode the frame directory using source FPS.

        """
        fps, source_count = self._source_metadata(original_video_path)
        frame_paths = self._frame_paths(frames_dir)

        print(f"Source FPS: {fps:.3f}")
        print(f"Source Frames: {source_count}")
        print(f"Found {len(frame_paths)} rendered frames")

        first = cv2.imread(str(frame_paths[0]))
        if first is None:
            raise RuntimeError(f"Could not read {frame_paths[0]}")

        h, w = first.shape[:2]
        print(f"Output Resolution: {w} x {h}")

        writer = cv2.VideoWriter(
            self.output_video_path,
            cv2.VideoWriter_fourcc(*self.codec),
            fps,
            (w, h),
        )
        try:
            if not writer.isOpened():
                raise RuntimeError(
                    f"Could not open writer: {self.output_video_path}"
                )

            for idx, path in enumerate(frame_paths):
                frame = cv2.imread(str(path))
                if frame is None:
                    print(f"Skipping unreadable frame: {path}")
                    continue

                writer.write(frame)

                if idx % 100 == 0:
                    print(f"Written {idx}/{len(frame_paths)}")
        finally:
            writer.release()

        print(f"Finished: {self.output_video_path}")

    def mux_audio(self, source_video, fps, frame_indices):
        """Attach source audio for the rendered frame interval.

        Audio starts at first_frame / fps and lasts n_frames / fps.
        Indices must refer to the supplied source video's timeline.

        The video stream is copied; audio is encoded as AAC. The output
        file is replaced only after ffmpeg completes successfully.

        Returns (ok, message). Noncontiguous indices retain the original
        behavior: mux the interval and warn that synchronization will drift.
        """
        if not frame_indices:
            return False, "no frames"

        first = frame_indices[0]
        count = len(frame_indices)
        contiguous = list(frame_indices) == list(
            range(first, first + count)
        )
        start = first / float(fps)
        duration = count / float(fps)

        source_video = str(source_video)
        temporary = f"{self.output_video_path}.mux.mp4"

        command = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", self.output_video_path,
            "-ss", f"{start:.6f}",
            "-t", f"{duration:.6f}",
            "-i", source_video,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            temporary,
        ]

        try:
            subprocess.run(command, check=True)
        except FileNotFoundError:
            return False, "ffmpeg not found"
        except subprocess.CalledProcessError as exc:
            if os.path.isfile(temporary):
                os.remove(temporary)
            return False, f"ffmpeg failed ({exc.returncode})"

        os.replace(temporary, self.output_video_path)

        message = (
            f"audio {start:.2f}s +{duration:.2f}s from "
            f"{os.path.basename(source_video)}"
        )
        if not contiguous:
            message += (
                "   WARNING frames are not contiguous, so sound drifts "
                "against picture after the first gap"
            )

        return True, message

    def inject_spatial_metadata(self, stereo="left-right"):
        """Run the existing spherical metadata command; write *_vr.mp4.

        This tags the video as equirectangular; it does not reproject pixels.
        The encoded projection and angular coverage must match playback
        expectations. Do not use this command to label fisheye imagery.

        Returns (output_path, None) or (None, error_message).
        """
        output = os.path.splitext(self.output_video_path)[0] + "_vr.mp4"
        command = [
            sys.executable, "-m", "spatialmedia",
            "-i",
            f"--stereo={stereo}",
            self.output_video_path,
            output,
        ]

        try:
            subprocess.run(command, check=True, capture_output=True)
            return output, None
        except FileNotFoundError:
            return None, "spatialmedia not installed (pip install spatialmedia)"
        except subprocess.CalledProcessError as exc:
            message = (exc.stderr or b"").decode()[:200]
            return None, message or "spatialmedia failed"