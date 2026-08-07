from __future__ import annotations

import math

import cv2
import numpy as np

from .effect_catalog import parameters_for_preset
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
    parameters = parameters_for_preset(name, event.preset)
    parameters.update(event.parameters)
    if name == "Glow":
        changed = _glow(frame, parameters, local)
    elif name == "Chromatic Aberration":
        changed = _chromatic_aberration(frame, parameters, local)
    elif name == "Color Adjustment":
        changed = _color_adjustment(frame, parameters, local)
    elif name == "Gaussian Blur":
        changed = _gaussian_blur(frame, parameters, local)
    elif name == "Pixelate":
        changed = _pixelate(frame, parameters)
    elif name == "Scanlines":
        changed = _scanlines(frame, parameters, frame_index)
    elif name == "TV Reception":
        changed = _tv_reception(frame, parameters, event, local, frame_index)
    elif name == "Row Glitch":
        changed = _row_glitch(frame, parameters, event, frame_index)
    elif name == "JPEG Glitch":
        changed = _jpeg_glitch(frame, parameters, event, frame_index)
    elif name == "Posterize":
        changed = _posterize(frame, parameters)
    elif name == "Sine Modulation":
        changed = _sine_modulation(frame, parameters, local)
    elif name == "Halftone":
        changed = _halftone(frame, parameters)
    elif name == "CMYK Halftone":
        changed = _cmyk_halftone(frame, parameters)
    elif name == "Declarative Shader":
        changed = _declarative_shader(frame, parameters, event, local, frame_index)
    elif name == "VHS Tracking Failure":
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


def _glow(frame: np.ndarray, parameters: dict, local: float) -> np.ndarray:
    radius = max(0.1, float(parameters.get("radius", 3.5)))
    strength = max(0.0, float(parameters.get("strength", 0.55)))
    strength *= 0.86 + 0.14 * math.sin(local * math.tau)
    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=radius, sigmaY=radius)
    return np.clip(frame.astype(np.float32) + blurred.astype(np.float32) * strength * 0.38, 0, 255).astype(np.uint8)


def _chromatic_aberration(frame: np.ndarray, parameters: dict, local: float) -> np.ndarray:
    offset = max(0, int(parameters.get("offset", 3)))
    pulse = max(0.0, float(parameters.get("pulse", 0.25)))
    animated_offset = max(1, round(offset * (1.0 - pulse + pulse * abs(math.sin(local * 4.7)))))
    blue, green, red = cv2.split(frame)
    return cv2.merge((np.roll(blue, animated_offset, axis=1), green, np.roll(red, -animated_offset, axis=1)))


def _color_adjustment(frame: np.ndarray, parameters: dict, local: float) -> np.ndarray:
    brightness = max(0.0, float(parameters.get("brightness", 1.0)))
    contrast = max(0.0, float(parameters.get("contrast", 1.0)))
    saturation = max(0.0, float(parameters.get("saturation", 1.0)))
    hue = float(parameters.get("hue", 0.0)) + math.sin(local * 1.7) * 2.0
    adjusted = np.clip((frame.astype(np.float32) - 127.5) * contrast + 127.5, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + hue / 2.0) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * brightness, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _gaussian_blur(frame: np.ndarray, parameters: dict, local: float) -> np.ndarray:
    radius = max(0.1, float(parameters.get("radius", 2.0)))
    pulse = max(0.0, float(parameters.get("pulse", 0.0)))
    radius *= 1.0 - pulse * 0.5 + pulse * (0.5 + 0.5 * math.sin(local * math.tau))
    return cv2.GaussianBlur(frame, (0, 0), sigmaX=max(0.1, radius), sigmaY=max(0.1, radius))


def _pixelate(frame: np.ndarray, parameters: dict) -> np.ndarray:
    height, width = frame.shape[:2]
    size = max(1, int(parameters.get("pixel_size", 6)))
    small = cv2.resize(frame, (max(1, width // size), max(1, height // size)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)


def _scanlines(frame: np.ndarray, parameters: dict, frame_index: int) -> np.ndarray:
    spacing = max(2, int(parameters.get("spacing", 3)))
    thickness = max(1, min(spacing, int(parameters.get("thickness", 1))))
    darkness = min(1.0, max(0.0, float(parameters.get("darkness", 0.35))))
    output = frame.copy()
    phase = frame_index % spacing
    for offset in range(thickness):
        output[(phase + offset) % spacing::spacing] = np.rint(
            output[(phase + offset) % spacing::spacing].astype(np.float32) * (1.0 - darkness)
        ).astype(np.uint8)
    return output


def _row_glitch(frame: np.ndarray, parameters: dict, event: EffectEvent, frame_index: int) -> np.ndarray:
    output = frame.copy()
    height = frame.shape[0]
    amount = max(0, int(parameters.get("amount", 12)))
    slice_height = max(1, int(parameters.get("slice_height", 5)))
    density = min(1.0, max(0.0, float(parameters.get("density", 0.2))))
    rng = _rng_for(event, frame_index)
    for y in range(0, height, slice_height):
        if rng.random() <= density:
            shift = int(rng.integers(-amount, amount + 1)) if amount else 0
            output[y:min(height, y + slice_height)] = np.roll(output[y:min(height, y + slice_height)], shift, axis=1)
    return output


def _tv_reception(frame: np.ndarray, parameters: dict, event: EffectEvent, local: float, frame_index: int) -> np.ndarray:
    output = frame.copy()
    height, width = frame.shape[:2]
    mode = str(parameters.get("mode", "Weak Reception"))
    distortion = max(0, int(parameters.get("distortion", 8)))
    band_height = max(1, min(height, int(parameters.get("band_height", 7))))
    static = min(1.0, max(0.0, float(parameters.get("static", 0.22))))
    ghost = min(1.0, max(0.0, float(parameters.get("ghost", 0.12))))
    rng = _rng_for(event, frame_index)

    for y in range(0, height, band_height):
        active = mode == "Horizontal Sync" or rng.random() < (0.38 if mode == "Signal Dropouts" else 0.18)
        if active:
            scale = 2.0 if mode == "Horizontal Sync" else 1.0
            shift = int(rng.integers(-distortion, distortion + 1) * scale) if distortion else 0
            output[y:min(height, y + band_height)] = np.roll(output[y:min(height, y + band_height)], shift, axis=1)

    if mode == "Rolling Bands":
        roll = int(height * ((local * 0.32) % 1.0))
        output = np.roll(output, roll, axis=0)
        rows = np.arange(height, dtype=np.float32)
        center = (local * 0.28 % 1.0) * height
        distance = np.minimum(np.abs(rows - center), height - np.abs(rows - center))
        band = np.clip(1.0 - distance / max(1.0, height * 0.14), 0.0, 1.0)
        output = np.clip(output.astype(np.float32) * (1.0 - band[:, None, None] * 0.45), 0, 255).astype(np.uint8)

    if ghost:
        ghost_frame = np.roll(output, max(1, width // 130), axis=1)
        output = cv2.addWeighted(output, 1.0 - ghost, ghost_frame, ghost, 0)

    if mode == "Color Interference":
        blue, green, red = cv2.split(output)
        offset = max(2, width // 90)
        output = cv2.merge((np.roll(blue, offset, axis=1), green, np.roll(red, -offset, axis=1)))

    noise = rng.integers(0, 256, frame.shape, dtype=np.uint8)
    if mode != "Color Interference":
        gray = cv2.cvtColor(noise, cv2.COLOR_BGR2GRAY)
        noise = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    static_scale = 1.65 if mode == "Heavy Snow" else 0.75 if mode == "Weak Reception" else 1.0
    noise_amount = min(1.0, static * static_scale * (1.0 if rng.random() < 0.34 else 0.42))
    output = cv2.addWeighted(output, 1.0 - noise_amount, noise, noise_amount, 0)

    if mode == "Signal Dropouts":
        for _ in range(max(1, height // 120)):
            y = int(rng.integers(0, height))
            end = min(height, y + band_height)
            output[y:end] = rng.integers(0, 256, (end - y, width, 3), dtype=np.uint8)
    return output


def _jpeg_glitch(frame: np.ndarray, parameters: dict, event: EffectEvent, frame_index: int) -> np.ndarray:
    quality = max(1, min(95, int(parameters.get("quality", 18))))
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return frame.copy()
    output = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    shift = max(0, int(parameters.get("block_shift", 8)))
    density = min(1.0, max(0.0, float(parameters.get("density", 0.35))))
    rng = _rng_for(event, frame_index)
    if shift:
        for y in range(0, output.shape[0], 8):
            if rng.random() < density:
                output[y:y + 8] = np.roll(output[y:y + 8], int(rng.integers(-shift, shift + 1)), axis=1)
    return output


def _posterize(frame: np.ndarray, parameters: dict) -> np.ndarray:
    levels = max(2, min(64, int(parameters.get("levels", 8))))
    source = frame
    if parameters.get("monochrome", False):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        source = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    step = 255.0 / (levels - 1)
    return np.clip(np.rint(source.astype(np.float32) / step) * step, 0, 255).astype(np.uint8)


def _sine_modulation(frame: np.ndarray, parameters: dict, local: float) -> np.ndarray:
    height = frame.shape[0]
    amplitude = float(parameters.get("amplitude", 8.0))
    frequency = max(0.1, float(parameters.get("frequency", 3.0)))
    speed = float(parameters.get("speed", 1.0))
    output = np.empty_like(frame)
    for y in range(height):
        offset = round(math.sin(y / max(1, height) * math.tau * frequency + local * speed * math.tau) * amplitude)
        output[y] = np.roll(frame[y], offset, axis=0)
    return output


def _parse_bgr(value: str) -> tuple[int, int, int]:
    text = str(value).removeprefix("#")[:6]
    if len(text) != 6:
        return 0, 0, 0
    red, green, blue = (int(text[index:index + 2], 16) for index in (0, 2, 4))
    return blue, green, red


def _halftone(frame: np.ndarray, parameters: dict) -> np.ndarray:
    height, width = frame.shape[:2]
    cell = max(2, int(parameters.get("cell_size", 8)))
    angle = float(parameters.get("angle", 45.0))
    foreground = _parse_bgr(str(parameters.get("foreground", "#101018")))
    background = _parse_bgr(str(parameters.get("background", "#F4E9D8")))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    gray = cv2.warpAffine(gray, rotation, (width, height), flags=cv2.INTER_LINEAR, borderValue=255)
    output = np.full_like(frame, background)
    grid_width = math.ceil(width / cell)
    grid_height = math.ceil(height / cell)
    darkness = 1.0 - cv2.resize(gray, (grid_width, grid_height), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    output[_dot_mask(darkness, height, width, cell, 1.04)] = foreground
    inverse = cv2.getRotationMatrix2D((width / 2, height / 2), -angle, 1.0)
    return cv2.warpAffine(output, inverse, (width, height), flags=cv2.INTER_LINEAR, borderValue=background)


def _dot_mask(amounts: np.ndarray, height: int, width: int, cell: int, gain: float) -> np.ndarray:
    radii = np.sqrt(np.clip(amounts, 0.0, 1.0)) * cell * 0.5 * gain
    expanded = np.repeat(np.repeat(radii, cell, axis=0), cell, axis=1)[:height, :width]
    center = (cell - 1) / 2.0
    y_distance = (np.arange(height, dtype=np.float32) % cell - center)[:, None] ** 2
    x_distance = (np.arange(width, dtype=np.float32) % cell - center)[None, :] ** 2
    return y_distance + x_distance <= expanded * expanded


def _cmyk_halftone(frame: np.ndarray, parameters: dict) -> np.ndarray:
    height, width = frame.shape[:2]
    cell = max(3, int(parameters.get("cell_size", 10)))
    gain = max(0.1, float(parameters.get("dot_gain", 1.0)))
    rgb = frame[..., ::-1].astype(np.float32) / 255.0
    key = 1.0 - np.max(rgb, axis=2)
    denominator = np.maximum(0.001, 1.0 - key)
    cmy = (1.0 - rgb - key[..., None]) / denominator[..., None]
    channels = (cmy[..., 0], cmy[..., 1], cmy[..., 2], key)
    inks = ((255, 180, 0), (140, 0, 220), (0, 210, 255), (0, 0, 0))
    output = np.full_like(frame, 255)
    offsets = ((0, 0), (cell // 3, 0), (0, cell // 3), (cell // 4, cell // 4))
    for channel, ink, (shift_y, shift_x) in zip(channels, inks, offsets):
        grid_width = math.ceil(width / cell)
        grid_height = math.ceil(height / cell)
        amounts = cv2.resize(channel, (grid_width, grid_height), interpolation=cv2.INTER_AREA)
        mask = _dot_mask(amounts, height, width, cell, gain)
        mask = np.roll(mask, (shift_y, shift_x), axis=(0, 1))
        output[mask] = np.rint(output[mask].astype(np.float32) * 0.42 + np.asarray(ink, dtype=np.float32) * 0.58).astype(np.uint8)
    return output


def _declarative_shader(frame: np.ndarray, parameters: dict, event: EffectEvent, local: float, frame_index: int) -> np.ndarray:
    output = frame.copy()
    offset = int(parameters.get("channel_offset", 0))
    if offset:
        output = _chromatic_aberration(output, {"offset": offset, "pulse": 0.15}, local)
    noise_strength = float(parameters.get("noise", 0.0))
    if noise_strength:
        rng = _rng_for(event, frame_index)
        noise = rng.normal(0.0, noise_strength, output.shape).astype(np.float32)
        output = np.clip(output.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    spacing = int(parameters.get("scanline_spacing", 0))
    if spacing:
        output = _scanlines(output, {"spacing": spacing, "thickness": 1, "darkness": parameters.get("scanline_darkness", 0.25)}, frame_index)
    if "levels" in parameters:
        output = _posterize(output, {"levels": parameters["levels"], "monochrome": False})
    glow = float(parameters.get("glow", 0.0))
    if glow:
        output = cv2.addWeighted(output, 1.0, cv2.GaussianBlur(output, (0, 0), 4.0), glow * 0.35, 0)
    vignette_strength = float(parameters.get("vignette", 0.0))
    if vignette_strength:
        height, width = output.shape[:2]
        yy, xx = np.ogrid[-1:1:complex(height), -1:1:complex(width)]
        vignette = np.clip(1.0 - vignette_strength * 0.65 * (xx * xx + yy * yy), 0.35, 1.0)[..., None]
        output = np.clip(output.astype(np.float32) * vignette, 0, 255).astype(np.uint8)
    return output


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
    # Build a new field for every frame so the effect reads as temporal TV
    # snow.  The event envelope handles the uniform fade in and out; a moving
    # spatial mask here used to create an unintended upward wipe.
    luminance = rng.integers(0, 256, (height, width), dtype=np.uint8)
    noise = cv2.cvtColor(luminance, cv2.COLOR_GRAY2BGR).astype(np.float32)
    row_gain = rng.uniform(0.72, 1.18, (height, 1, 1)).astype(np.float32)
    noise *= row_gain

    # A few short horizontal streaks keep the texture recognizably analog
    # without introducing a single hard boundary that travels through frame.
    for _ in range(max(2, height // 180)):
        y = int(rng.integers(0, height))
        streak_height = int(rng.integers(1, max(2, height // 120 + 1)))
        noise[y:min(height, y + streak_height)] *= float(rng.uniform(0.25, 1.65))

    noise = np.clip(noise, 0, 255).astype(np.uint8)
    return cv2.addWeighted(noise, 0.94, frame, 0.06, 0.0)


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
