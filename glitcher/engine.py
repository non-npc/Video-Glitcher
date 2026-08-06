from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .effects import add_crt_finish, apply_effect
from .generators import render_generator
from .models import Clip, Project


class RenderEngine:
    def __init__(self, project: Project, preview_size: tuple[int, int] | None = None) -> None:
        self.project = project
        self.preview_size = preview_size
        self._captures: dict[str, cv2.VideoCapture] = {}

    @property
    def size(self) -> tuple[int, int]:
        return self.preview_size or (self.project.width, self.project.height)

    def close(self) -> None:
        for capture in self._captures.values():
            capture.release()
        self._captures.clear()

    def _video_frame(self, clip: Clip, local_time: float, width: int, height: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        if not clip.path or not Path(clip.path).exists():
            return self._missing_frame(width, height, clip.name), (0, 0, width, height)
        capture = self._captures.get(clip.id)
        if capture is None:
            capture = cv2.VideoCapture(clip.path)
            self._captures[clip.id] = capture
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, local_time) * 1000.0)
        ok, frame = capture.read()
        if not ok:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = capture.read()
        if not ok:
            return self._missing_frame(width, height, clip.name), (0, 0, width, height)
        return self._fit_frame_and_rect(frame, width, height, self.project.source_sizing)

    @staticmethod
    def _fit_frame(frame: np.ndarray, width: int, height: int, sizing: str = "Fit entire frame") -> np.ndarray:
        fitted, _rect = RenderEngine._fit_frame_and_rect(frame, width, height, sizing)
        return fitted

    @staticmethod
    def _fit_frame_and_rect(
        frame: np.ndarray,
        width: int,
        height: int,
        sizing: str = "Fit entire frame",
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Scale a source and return its visible rectangle in the output frame."""
        source_height, source_width = frame.shape[:2]
        if sizing == "Stretch to frame":
            return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA), (0, 0, width, height)
        scale = max(width / source_width, height / source_height) if sizing == "Fill frame (crop)" else min(width / source_width, height / source_height)
        resized = cv2.resize(frame, (max(1, int(source_width * scale)), max(1, int(source_height * scale))), interpolation=cv2.INTER_AREA)
        if sizing == "Fit entire frame":
            output = np.zeros((height, width, 3), dtype=np.uint8)
            y = max(0, (height - resized.shape[0]) // 2)
            x = max(0, (width - resized.shape[1]) // 2)
            copy_height = min(height, resized.shape[0])
            copy_width = min(width, resized.shape[1])
            output[y:y + copy_height, x:x + copy_width] = resized[:copy_height, :copy_width]
            return output, (x, y, copy_width, copy_height)
        y = max(0, (resized.shape[0] - height) // 2)
        x = max(0, (resized.shape[1] - width) // 2)
        return resized[y:y + height, x:x + width].copy(), (0, 0, width, height)

    @staticmethod
    def _missing_frame(width: int, height: int, label: str) -> np.ndarray:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (22, 12, 30)
        cv2.putText(frame, f"OFFLINE: {label}", (24, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (190, 240, 240), 2, cv2.LINE_AA)
        return frame

    def render(self, time_seconds: float) -> np.ndarray:
        width, height = self.size
        clip, local_time = self.project.clip_at(max(0.0, time_seconds))
        if clip is None:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            content_rect = (0, 0, width, height)
        elif clip.kind == "generator":
            frame = render_generator(clip.generator or "Neon Grid", width, height, local_time, clip.seed)
            content_rect = (0, 0, width, height)
        else:
            frame, content_rect = self._video_frame(clip, local_time, width, height)
        x, y, content_width, content_height = content_rect
        content = frame[y:y + content_height, x:x + content_width].copy()
        for event in self.project.effects:
            if event.start <= time_seconds <= event.end:
                content = apply_effect(content, event, time_seconds, self.project.fps)
        content = add_crt_finish(content, time_seconds)
        frame[y:y + content_height, x:x + content_width] = content
        return frame
