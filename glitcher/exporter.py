from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable

from .engine import RenderEngine
from .models import Project


ProgressCallback = Callable[[float, str], None]


class ExportCancelled(Exception):
    pass


@dataclass(frozen=True, slots=True)
class EncoderSpec:
    key: str
    label: str
    codec: str
    arguments: tuple[str, ...]
    hardware: bool = True


CPU_ENCODER = EncoderSpec(
    "cpu",
    "CPU (x264)",
    "libx264",
    ("-preset", "veryfast", "-crf", "18"),
    hardware=False,
)

HARDWARE_ENCODERS = (
    EncoderSpec(
        "nvenc",
        "NVIDIA GPU (NVENC)",
        "h264_nvenc",
        ("-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "20", "-b:v", "0"),
    ),
    EncoderSpec(
        "qsv",
        "Intel GPU (Quick Sync)",
        "h264_qsv",
        ("-preset", "medium", "-global_quality", "20"),
    ),
    EncoderSpec(
        "amf",
        "AMD GPU (AMF)",
        "h264_amf",
        ("-quality", "quality", "-rc", "cqp", "-qp_i", "20", "-qp_p", "20"),
    ),
)


def _creation_flags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _probe_encoder(spec: EncoderSpec) -> bool:
    """Confirm that FFmpeg can actually open and use a hardware encoder."""
    command = [
        "ffmpeg", "-hide_banner", "-v", "error",
        "-f", "lavfi", "-i", "color=size=64x64:rate=1:duration=0.1",
        "-frames:v", "1", "-c:v", spec.codec, *spec.arguments,
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            creationflags=_creation_flags(),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


@lru_cache(maxsize=1)
def detect_h264_encoders() -> tuple[EncoderSpec, ...]:
    """Return working encoders in automatic preference order, then CPU."""
    available = [spec for spec in HARDWARE_ENCODERS if _probe_encoder(spec)]
    available.append(CPU_ENCODER)
    return tuple(available)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:d}:{seconds:02d}"


class VideoExporter:
    def __init__(self, project: Project, encoder: str = "auto") -> None:
        self.project = project
        self.encoder = encoder
        self.cancel_event = threading.Event()
        self.active_process: subprocess.Popen[bytes] | None = None

    def cancel(self) -> None:
        self.cancel_event.set()
        process = self.active_process
        if process is not None and process.poll() is None:
            process.terminate()

    def _selected_encoder(self) -> EncoderSpec:
        available = detect_h264_encoders()
        if self.encoder == "auto":
            return available[0]
        return next((spec for spec in available if spec.key == self.encoder), CPU_ENCODER)

    def _command(self, output: Path, encoder: EncoderSpec) -> list[str]:
        command = [
            "ffmpeg", "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-video_size", f"{self.project.width}x{self.project.height}",
            "-framerate", str(self.project.fps), "-i", "pipe:0",
        ]
        if self.project.audio_path:
            command.extend([
                "-ss", f"{max(0.0, self.project.audio_source_start):.6f}",
                "-i", str(Path(self.project.audio_path).resolve()),
                "-filter_complex",
                f"[1:a]adelay=delays={int(max(0.0, self.project.audio_timeline_start) * 1000)}:all=1[audio]",
                "-map", "0:v:0", "-map", "[audio]",
            ])
        else:
            command.extend(["-map", "0:v:0", "-an"])
        command.extend([
            "-c:v", encoder.codec, *encoder.arguments,
            "-pix_fmt", "yuv420p",
        ])
        if self.project.audio_path:
            command.extend(["-c:a", "aac", "-b:a", "192k"])
        command.extend([
            "-t", f"{self.project.duration:.6f}",
            "-movflags", "+faststart", str(output),
        ])
        return command

    @staticmethod
    def _stop_process(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def _stream_frames(
        self,
        output: Path,
        encoder: EncoderSpec,
        progress: ProgressCallback | None,
    ) -> None:
        frame_count = max(1, int(round(self.project.duration * self.project.fps)))
        command = self._command(output, encoder)
        process: subprocess.Popen[bytes] | None = None
        engine = RenderEngine(self.project)
        started = time.perf_counter()
        last_sample_time = started
        last_sample_frame = 0
        smoothed_rate: float | None = None
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=0,
                creationflags=_creation_flags(),
            )
            self.active_process = process
            if progress:
                progress(0.0, f"Rendering on CPU · Encoding with {encoder.label} · Estimating time remaining…")
            assert process.stdin is not None
            update_interval = max(1, self.project.fps // 2)
            for index in range(frame_count):
                if self.cancel_event.is_set():
                    raise ExportCancelled("Export cancelled.")
                frame = engine.render(index / self.project.fps)
                process.stdin.write(frame.tobytes())
                completed = index + 1
                if progress and (completed % update_interval == 0 or completed == frame_count):
                    now = time.perf_counter()
                    interval = now - last_sample_time
                    if interval > 0:
                        recent_rate = (completed - last_sample_frame) / interval
                        smoothed_rate = recent_rate if smoothed_rate is None else smoothed_rate * 0.72 + recent_rate * 0.28
                    elapsed = now - started
                    if elapsed >= 2.0 and smoothed_rate and completed < frame_count:
                        eta = format_duration((frame_count - completed) / smoothed_rate)
                        timing = f"About {eta} remaining"
                    elif completed == frame_count:
                        timing = "Finishing export…"
                    else:
                        timing = "Estimating time remaining…"
                    text = (
                        f"Rendering on CPU · Encoding with {encoder.label}\n"
                        f"Frame {completed:,} of {frame_count:,} · {format_duration(elapsed)} elapsed · {timing}"
                    )
                    progress(completed / frame_count * 0.99, text)
                    last_sample_time = now
                    last_sample_frame = completed
            process.stdin.close()
            stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
            return_code = process.wait()
            if self.cancel_event.is_set():
                raise ExportCancelled("Export cancelled.")
            if return_code:
                raise subprocess.CalledProcessError(return_code, command, stderr=stderr)
        except BrokenPipeError as error:
            stderr = ""
            if process is not None:
                if process.stderr:
                    stderr = process.stderr.read().decode("utf-8", errors="replace")
                process.wait()
            if self.cancel_event.is_set():
                raise ExportCancelled("Export cancelled.") from error
            raise subprocess.CalledProcessError(
                process.returncode if process and process.returncode is not None else 1,
                command,
                stderr=stderr,
            ) from error
        finally:
            engine.close()
            if process is not None and process.poll() is None:
                self._stop_process(process)
            if process is not None:
                if process.stdin is not None and not process.stdin.closed:
                    process.stdin.close()
                if process.stderr is not None and not process.stderr.closed:
                    process.stderr.close()
            self.active_process = None

    def export(self, output_path: str | Path, progress: ProgressCallback | None = None) -> None:
        output = Path(output_path).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if self.project.duration <= 0:
            raise ValueError("The project has no video or generator clips.")
        if self.cancel_event.is_set():
            raise ExportCancelled("Export cancelled.")

        temp_handle, temp_name = tempfile.mkstemp(prefix="video-glitcher-", suffix=".mp4", dir=output.parent)
        os.close(temp_handle)
        temporary_output = Path(temp_name)
        selected = self._selected_encoder()
        attempts = [selected]
        if selected.hardware:
            attempts.append(CPU_ENCODER)

        try:
            for attempt_index, encoder in enumerate(attempts):
                temporary_output.unlink(missing_ok=True)
                if attempt_index and progress:
                    progress(0.0, f"{selected.label} failed · Retrying with {CPU_ENCODER.label}")
                try:
                    self._stream_frames(temporary_output, encoder, progress)
                    break
                except ExportCancelled:
                    raise
                except (OSError, subprocess.CalledProcessError):
                    if attempt_index == len(attempts) - 1:
                        raise
            if self.cancel_event.is_set():
                raise ExportCancelled("Export cancelled.")
            os.replace(temporary_output, output)
        except Exception:
            temporary_output.unlink(missing_ok=True)
            raise
        if progress:
            progress(1.0, f"Exported {output.name} with {encoder.label}")
