from __future__ import annotations

import math

import cv2
import numpy as np

from .models import EffectEvent


def _smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def event_envelope(local_time: float, duration: float, meld: float = 0.0) -> float:
    """Smooth attack/peak/recovery envelope for effects of any duration."""
    if duration <= 0:
        return 0.0
    progress = min(1.0, max(0.0, local_time / duration))
    attack_end = min(0.28, max(0.08, 0.16 - meld * 0.05))
    release_start = min(0.94, max(0.55, 0.74 + meld * 0.16))
    if progress < attack_end:
        return _smoothstep(progress / attack_end)
    if progress > release_start:
        return _smoothstep((1.0 - progress) / (1.0 - release_start))
    return 1.0


def _blend(original: np.ndarray, changed: np.ndarray, amount: float) -> np.ndarray:
    return cv2.addWeighted(original, 1.0 - amount, changed, amount, 0.0)


def _rng_for(event: EffectEvent, frame_index: int) -> np.random.Generator:
    return np.random.default_rng((event.seed + frame_index * 104729) & 0xFFFFFFFF)


def apply_effect(frame: np.ndarray, event: EffectEvent, absolute_time: float, fps: float) -> np.ndarray:
    local = absolute_time - event.start
    amount = event_envelope(local, event.duration, event.meld) * min(1.0, max(0.0, event.intensity))
    if amount <= 0.001:
        return frame
    frame_index = int(round(absolute_time * fps))
    name = event.effect
    if name == "VHS Tracking Failure":
        changed = _tracking(frame, event, local, frame_index)
    elif name == "RGB Ghost":
        changed = _rgb_ghost(frame, event, local)
    elif name == "Neon Signal Collapse":
        changed = _neon_collapse(frame, event, local, frame_index)
    elif name == "Static Reconstruction":
        changed = _static_rebuild(frame, event, local, frame_index)
    elif name == "Vertical Sync Roll":
        changed = _vertical_roll(frame, event, local)
    elif name == "Video Feedback":
        changed = _feedback(frame, event, local)
    else:
        return frame
    return _blend(frame, changed, amount)


def _rgb_ghost(frame: np.ndarray, event: EffectEvent, local: float) -> np.ndarray:
    height, width = frame.shape[:2]
    phase = math.sin(local * 5.2 + event.seed * 0.01)
    offset = max(2, int(width * (0.008 + 0.018 * abs(phase))))
    blue, green, red = cv2.split(frame)
    blue = np.roll(blue, offset, axis=1)
    red = np.roll(red, -offset, axis=1)
    green = np.roll(green, int(offset * phase * 0.35), axis=1)
    return cv2.merge((blue, green, red))


def _tracking(frame: np.ndarray, event: EffectEvent, local: float, frame_index: int) -> np.ndarray:
    output = _rgb_ghost(frame, event, local)
    height, width = frame.shape[:2]
    rng = _rng_for(event, frame_index // 2)
    band_count = int(rng.integers(2, 7))
    for _ in range(band_count):
        y = int(rng.integers(0, max(1, height - 2)))
        band_height = int(rng.integers(max(2, height // 90), max(4, height // 10)))
        shift = int(rng.integers(-max(3, width // 10), max(4, width // 10)))
        end = min(height, y + band_height)
        output[y:end] = np.roll(output[y:end], shift, axis=1)
    switch_height = max(3, height // 26)
    output[-switch_height:] = np.roll(output[-switch_height:], int(width * 0.045), axis=1)
    return output


def _neon_collapse(frame: np.ndarray, event: EffectEvent, local: float, frame_index: int) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + 18 + 28 * math.sin(local * 2.7)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.75 + 30, 0, 255)
    hsv[..., 2] = np.clip((hsv[..., 2] // 42) * 42 + 20, 0, 255)
    output = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    edges = cv2.Canny(frame, 80, 150)
    edge_color = np.zeros_like(frame)
    edge_color[..., 0] = edges
    edge_color[..., 2] = np.roll(edges, 3, axis=1)
    output = cv2.addWeighted(output, 0.82, edge_color, 0.65, 0)
    return _tracking(output, event, local, frame_index)


def _static_rebuild(frame: np.ndarray, event: EffectEvent, local: float, frame_index: int) -> np.ndarray:
    height, width = frame.shape[:2]
    rng = _rng_for(event, frame_index)
    noise = rng.integers(0, 256, frame.shape, dtype=np.uint8)
    progress = min(1.0, max(0.0, local / max(0.001, event.duration)))
    threshold = int(height * (1.0 - progress))
    mask = np.zeros((height, width), dtype=np.float32)
    feather = max(8, height // 12)
    for y in range(height):
        mask[y] = min(1.0, max(0.0, (threshold - y + feather) / feather))
    mask = mask[..., None]
    return np.clip(frame * (1.0 - mask) + noise * mask, 0, 255).astype(np.uint8)


def _vertical_roll(frame: np.ndarray, event: EffectEvent, local: float) -> np.ndarray:
    height = frame.shape[0]
    progress = local / max(event.duration, 0.001)
    shift = int(height * (0.5 - 0.5 * math.cos(progress * math.tau)))
    output = np.roll(frame, shift, axis=0)
    seam = shift % height
    output[max(0, seam - 2):min(height, seam + 3)] = 8
    return output


def _feedback(frame: np.ndarray, event: EffectEvent, local: float) -> np.ndarray:
    height, width = frame.shape[:2]
    output = frame.astype(np.float32)
    center = (width / 2, height / 2)
    for index in range(1, 6):
        scale = 1.0 - index * (0.035 + 0.006 * math.sin(local * 2.0))
        matrix = cv2.getRotationMatrix2D(center, index * 0.35 * math.sin(local), scale)
        echo = cv2.warpAffine(frame, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
        output += echo.astype(np.float32) * (0.22 / index)
    return np.clip(output / 1.28, 0, 255).astype(np.uint8)


def add_crt_finish(frame: np.ndarray, time_seconds: float, strength: float = 0.22) -> np.ndarray:
    output = frame.astype(np.float32)
    height, width = frame.shape[:2]
    output[1::3] *= 1.0 - strength
    output[2::3] *= 1.0 - strength * 0.45
    yy, xx = np.ogrid[-1:1:complex(height), -1:1:complex(width)]
    vignette = np.clip(1.08 - 0.30 * (xx * xx + yy * yy), 0.62, 1.0)[..., None]
    output *= vignette
    flicker = 0.985 + 0.015 * math.sin(time_seconds * 57.0)
    return np.clip(output * flicker, 0, 255).astype(np.uint8)

