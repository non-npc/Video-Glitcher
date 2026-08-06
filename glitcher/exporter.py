from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable

import cv2

from .engine import RenderEngine
from .models import Project


ProgressCallback = Callable[[float, str], None]


class ExportCancelled(Exception):
    pass


class VideoExporter:
    def __init__(self, project: Project) -> None:
        self.project = project
        self.cancel_event = threading.Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    def _run_ffmpeg(self, command: list[str]) -> None:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        while True:
            try:
                _stdout, stderr = process.communicate(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                if self.cancel_event.is_set():
                    process.terminate()
                    try:
                        process.communicate(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.communicate()
                    raise ExportCancelled("Export cancelled.")
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command, stderr=stderr)

    def export(self, output_path: str | Path, progress: ProgressCallback | None = None) -> None:
        output = Path(output_path).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if self.project.duration <= 0:
            raise ValueError("The project has no video or generator clips.")
        frame_count = max(1, int(round(self.project.duration * self.project.fps)))
        temp_handle, temp_name = tempfile.mkstemp(prefix="video-glitcher-", suffix=".mp4", dir=output.parent)
        os.close(temp_handle)
        temp_video = Path(temp_name)
        engine = RenderEngine(self.project)
        writer = cv2.VideoWriter(
            str(temp_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.project.fps,
            (self.project.width, self.project.height),
        )
        if not writer.isOpened():
            engine.close()
            temp_video.unlink(missing_ok=True)
            raise RuntimeError("OpenCV could not create the temporary video stream.")
        cancelled_during_frames = False
        try:
            for index in range(frame_count):
                if self.cancel_event.is_set():
                    raise ExportCancelled("Export cancelled.")
                writer.write(engine.render(index / self.project.fps))
                if progress and (index % max(1, self.project.fps // 2) == 0 or index == frame_count - 1):
                    progress((index + 1) / frame_count * 0.90, f"Rendering frame {index + 1:,} of {frame_count:,}")
        except ExportCancelled:
            cancelled_during_frames = True
        finally:
            writer.release()
            engine.close()

        if cancelled_during_frames or self.cancel_event.is_set():
            temp_video.unlink(missing_ok=True)
            raise ExportCancelled("Export cancelled.")

        try:
            if self.project.audio_path:
                if progress:
                    progress(0.93, "Muxing the audio track")
                command = [
                    "ffmpeg", "-y", "-v", "error",
                    "-i", str(temp_video),
                    "-ss", f"{max(0.0, self.project.audio_source_start):.6f}",
                    "-i", str(Path(self.project.audio_path).resolve()),
                    "-filter_complex", f"[1:a]adelay=delays={int(max(0.0, self.project.audio_timeline_start) * 1000)}:all=1[audio]",
                    "-map", "0:v:0", "-map", "[audio]",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-c:a", "aac", "-b:a", "192k", "-t", f"{self.project.duration:.6f}", str(output),
                ]
                self._run_ffmpeg(command)
                temp_video.unlink(missing_ok=True)
            else:
                if progress:
                    progress(0.93, "Encoding H.264 video")
                command = [
                    "ffmpeg", "-y", "-v", "error", "-i", str(temp_video),
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-an", str(output),
                ]
                self._run_ffmpeg(command)
                temp_video.unlink(missing_ok=True)
        except ExportCancelled:
            temp_video.unlink(missing_ok=True)
            output.unlink(missing_ok=True)
            raise
        except (OSError, subprocess.CalledProcessError):
            # The intermediate MP4 is still playable if FFmpeg is unavailable.
            shutil.move(str(temp_video), str(output))
        if progress:
            progress(1.0, f"Exported {output.name}")
