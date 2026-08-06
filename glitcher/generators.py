from __future__ import annotations

import math

import cv2
import numpy as np


def render_generator(name: str, width: int, height: int, time_seconds: float, seed: int) -> np.ndarray:
    if name == "Synth Sun":
        return _synth_sun(width, height, time_seconds, seed)
    if name == "Plasma Field":
        return _plasma(width, height, time_seconds, seed)
    if name == "Analog Static":
        return _static(width, height, time_seconds, seed)
    return _neon_grid(width, height, time_seconds, seed)


def _background(width: int, height: int) -> np.ndarray:
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    frame = np.zeros((height, width, 3), dtype=np.float32)
    frame[..., 0] = 30 + 38 * y
    frame[..., 1] = 4 + 9 * y
    frame[..., 2] = 14 + 30 * y
    return frame.astype(np.uint8)


def _neon_grid(width: int, height: int, time_seconds: float, seed: int) -> np.ndarray:
    frame = _background(width, height)
    horizon = int(height * 0.47)
    cv2.line(frame, (0, horizon), (width, horizon), (210, 60, 255), 2)
    center = width // 2 + int(math.sin(time_seconds * 0.3 + seed) * width * 0.03)
    for index in range(-16, 17):
        bottom_x = center + index * max(22, width // 14)
        cv2.line(frame, (center, horizon), (bottom_x, height), (220, 235, 30), 1)
    phase = (time_seconds * 0.34) % 1.0
    for index in range(18):
        depth = (index + phase) / 18.0
        curve = depth * depth
        y = int(horizon + curve * (height - horizon))
        cv2.line(frame, (0, y), (width, y), (220, 235, 30), 1)
    return frame


def _synth_sun(width: int, height: int, time_seconds: float, seed: int) -> np.ndarray:
    frame = _neon_grid(width, height, time_seconds, seed)
    center = (width // 2, int(height * 0.31))
    radius = max(24, int(height * 0.19))
    cv2.circle(frame, center, radius, (65, 92, 255), -1, lineType=cv2.LINE_AA)
    for y in range(center[1] - radius, center[1] + radius, max(5, height // 55)):
        if y > center[1]:
            cv2.line(frame, (center[0] - radius, y), (center[0] + radius, y), (33, 8, 62), max(1, height // 100))
    return frame


def _plasma(width: int, height: int, time_seconds: float, seed: int) -> np.ndarray:
    x = np.linspace(-3.0, 3.0, width, dtype=np.float32)[None, :]
    y = np.linspace(-2.0, 2.0, height, dtype=np.float32)[:, None]
    phase = seed % 997 / 997.0 * math.tau
    value = (
        np.sin(x * 2.1 + time_seconds * 1.5 + phase)
        + np.sin(y * 3.2 - time_seconds * 1.1)
        + np.sin((x + y) * 2.4 + time_seconds * 0.8)
        + np.sin(np.sqrt(x * x + y * y) * 4.8 - time_seconds * 2.0)
    ) / 4.0
    hue = ((value + 1.0) * 82 + time_seconds * 14) % 180
    hsv = np.empty((height, width, 3), dtype=np.uint8)
    hsv[..., 0] = hue.astype(np.uint8)
    hsv[..., 1] = 225
    hsv[..., 2] = np.clip(130 + (value + 1.0) * 62, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _static(width: int, height: int, time_seconds: float, seed: int) -> np.ndarray:
    frame_index = int(time_seconds * 30.0)
    rng = np.random.default_rng((seed + frame_index * 65537) & 0xFFFFFFFF)
    luminance = rng.integers(0, 256, (height, width), dtype=np.uint8)
    frame = cv2.cvtColor(luminance, cv2.COLOR_GRAY2BGR)
    chroma = rng.integers(-30, 31, (height, width), dtype=np.int16)
    frame[..., 0] = np.clip(frame[..., 0].astype(np.int16) + chroma, 0, 255).astype(np.uint8)
    frame[..., 2] = np.clip(frame[..., 2].astype(np.int16) - chroma, 0, 255).astype(np.uint8)
    return frame

