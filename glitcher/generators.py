from __future__ import annotations

import math

import cv2
import numpy as np

from .retrowave_drive import render_retrowave_drive


GEOMETRIC_TRAIL_PRESETS: dict[str, dict] = {
    "Green Circle Flow": {
        "shape": "Circle", "path": "Figure Eight", "head_color": "#78FF87",
        "tail_color": "#207844", "copies": 52, "span": 1.15,
        "tail_scale": 1.35, "size": 0.16, "speed": 0.14,
    },
    "Blue Square Cascade": {
        "shape": "Square", "path": "Sine Wave", "head_color": "#3678FF",
        "tail_color": "#172D80", "copies": 42, "span": 1.3,
        "tail_scale": 1.5, "size": 0.17, "speed": 0.12,
    },
    "Triangle Orbit": {
        "shape": "Triangle", "path": "Orbit", "head_color": "#FFE55C",
        "tail_color": "#FF3CAC", "copies": 36, "span": 0.82,
        "trail_rotation": 11.0, "rotation_speed": 0.5, "size": 0.18,
    },
    "Static Concentric Signal": {
        "shape": "Circle", "path": "Static Point", "head_color": "#20E3FF",
        "tail_color": "#FF3CAC", "copies": 24, "span": 0.0,
        "tail_scale": 4.0, "tail_opacity": 0.65, "animate": False,
        "pulse": 0.0, "size": 0.08,
    },
    "Rainbow Lissajous": {
        "shape": "Ellipse", "path": "Lissajous", "head_color": "#7067FF",
        "tail_color": "#20E3FF", "copies": 72, "span": 1.0,
        "tail_scale": 0.8, "size": 0.14, "aspect": 1.8, "lobes_x": 3,
        "lobes_y": 2, "speed": 0.1,
    },
    "Diamond Pulse": {
        "shape": "Diamond", "path": "Diamond Circuit", "head_color": "#FF3CAC",
        "tail_color": "#754BFF", "copies": 38, "span": 0.85,
        "pulse": 0.4, "pulse_speed": 0.45, "rotation_speed": -0.35,
        "size": 0.16,
    },
    "Filled Triangle Wave": {
        "shape": "Triangle", "path": "Sine Wave", "head_color": "#FF6A3D",
        "tail_color": "#FFE66D", "copies": 30, "span": 1.0,
        "filled": True, "tail_opacity": 0.05, "trail_rotation": 16.0,
        "size": 0.17, "speed": 0.13,
    },
    "Solo Flashing Square": {
        "shape": "Square", "path": "Random Drift", "head_color": "#31E7FF",
        "tail_color": "#31E7FF", "copies": 1, "span": 0.0,
        "tail_scale": 1.0, "size": 0.22, "speed": 0.18,
        "pulse": 0.32, "pulse_speed": 0.3, "flash_speed": 2.0,
        "line_width": 4, "glow": 1.0,
    },
}

GEOMETRIC_GENERATOR_NAMES = tuple(GEOMETRIC_TRAIL_PRESETS)

RETROWAVE_DRIVE_DEFAULTS = {
    "terrain_style": "Rolling Hills",
    "travel_speed": 1.0,
    "horizon_y": 0.44,
    "vanishing_x": 0.5,
    "camera_height": 1.2,
    "road_width": 2.3,
    "road_lanes": 4,
    "grid_spacing": 1.35,
    "depth_lines": 28,
    "terrain_width": 6.5,
    "terrain_height": 1.7,
    "terrain_roughness": 0.75,
    "terrain_columns": 11,
    "terrain_samples": 48,
    "terrain_morph": 0.0,
    "morph_cycles": 1,
    "sun_visible": True,
    "sun_x": 0.5,
    "sun_y": 0.28,
    "sun_size": 0.14,
    "sun_stripes": 6,
    "sun_stripe_gap": 0.38,
    "line_width": 2,
    "glow_strength": 0.9,
    "reflection_strength": 0.28,
    "sky_top_color": "#05000FFF",
    "sky_horizon_color": "#351068FF",
    "ground_color": "#07000FFF",
    "road_color": "#09000FFF",
    "road_grid_color": "#FF43D1FF",
    "terrain_color": "#FF4FD8FF",
    "sun_top_color": "#FFF36BFF",
    "sun_bottom_color": "#FF3AAEFF",
    "transparent_background": False,
    "seed": 1986,
    "loop_duration": 6.0,
}

RETROWAVE_DRIVE_PRESETS = {
    "Neon Sunset Drive": {},
    "Jagged Night Highway": {
        "terrain_style": "Jagged Mountains", "terrain_height": 2.4,
        "terrain_roughness": 1.25, "sky_top_color": "#010008FF",
        "sky_horizon_color": "#16003FFF", "road_grid_color": "#28DFFFFF",
        "terrain_color": "#FF3AAEFF",
    },
    "Neon Canyon": {
        "terrain_style": "Canyon", "road_width": 1.7,
        "terrain_height": 2.8, "terrain_width": 5.2,
        "terrain_color": "#FF643CFF", "road_grid_color": "#FFE05CFF",
        "sun_bottom_color": "#FF4828FF",
    },
    "Symmetrical Valley": {
        "terrain_style": "Symmetrical Valley", "terrain_height": 2.1,
        "terrain_roughness": 0.55, "road_width": 2.8,
        "terrain_color": "#B84DFFFF", "road_grid_color": "#36E7FFFF",
    },
    "Morphing Dreamscape": {
        "terrain_style": "Sparse Peaks", "terrain_height": 3.0,
        "terrain_morph": 0.55, "morph_cycles": 2,
        "glow_strength": 1.35, "sky_horizon_color": "#5A126CFF",
        "terrain_color": "#FF70E6FF",
    },
    "Flat Infinite Grid": {
        "terrain_style": "Flat Infinite Grid", "terrain_height": 0.0,
        "terrain_columns": 16, "terrain_width": 8.0, "road_width": 2.8,
        "sun_size": 0.17, "road_grid_color": "#756DFFFF",
        "terrain_color": "#756DFFFF",
    },
    "Transparent Wireframe": {
        "transparent_background": True, "sun_visible": False,
        "reflection_strength": 0.0, "terrain_style": "Rolling Hills",
        "terrain_color": "#FF4FD8D8", "road_grid_color": "#20E3FFD8",
    },
}

GENERATOR_NAMES = (
    "Neon Grid",
    "Synth Sun",
    "Plasma Field",
    "Analog Static",
    "Geometric Trails",
    "Retrowave Drive",
)

GENERATOR_DEFAULTS: dict[str, dict] = {
    "Neon Grid": {
        "horizon": 0.47, "speed": 0.34, "ray_count": 33,
        "depth_lines": 18, "grid_color": "#1EEBDC",
        "horizon_color": "#FF3CD2", "sky_top": "#0E041E",
        "sky_bottom": "#44261E",
    },
    "Synth Sun": {
        "horizon": 0.47, "speed": 0.34, "ray_count": 33,
        "depth_lines": 18, "grid_color": "#1EEBDC",
        "horizon_color": "#FF3CD2", "sky_top": "#0E041E",
        "sky_bottom": "#44261E", "sun_size": 0.19,
        "sun_color": "#FF5C41", "stripe_color": "#3E0821",
        "stripe_spacing": 0.014,
    },
    "Plasma Field": {
        "speed": 1.0, "scale": 1.0, "hue_speed": 14.0,
        "saturation": 225, "brightness": 1.0,
    },
    "Analog Static": {
        "color": True, "grain_size": 1, "contrast": 1.0,
    },
    "Geometric Trails": dict(GEOMETRIC_TRAIL_PRESETS["Green Circle Flow"]),
    "Retrowave Drive": dict(RETROWAVE_DRIVE_DEFAULTS),
}

GENERATOR_PRESETS: dict[str, dict[str, dict]] = {
    "Neon Grid": {
        "Classic Neon": {},
        "Cyan Highway": {"grid_color": "#20E3FF", "horizon_color": "#20E3FF", "sky_bottom": "#081C32", "speed": 0.5},
        "Magenta Night": {"grid_color": "#FF3CAC", "horizon_color": "#A94BFF", "sky_top": "#030008", "sky_bottom": "#260027", "depth_lines": 24},
    },
    "Synth Sun": {
        "Classic Sunset": {},
        "Golden Horizon": {"sun_color": "#FFE36B", "stripe_color": "#B02D45", "grid_color": "#FF43D1", "sky_bottom": "#5A1230"},
        "Electric Blue": {"sun_color": "#61E8FF", "stripe_color": "#142F68", "grid_color": "#756DFF", "horizon_color": "#20E3FF", "sky_bottom": "#071C45"},
    },
    "Plasma Field": {
        "Classic Plasma": {},
        "Fast Acid": {"speed": 1.8, "scale": 1.35, "hue_speed": 26.0, "saturation": 255},
        "Deep Ocean": {"speed": 0.55, "scale": 0.72, "hue_speed": 5.0, "saturation": 175, "brightness": 0.72},
    },
    "Analog Static": {
        "Color Static": {},
        "Monochrome Snow": {"color": False, "grain_size": 1, "contrast": 1.15},
        "Coarse Antenna": {"color": False, "grain_size": 4, "contrast": 1.35},
    },
    "Geometric Trails": GEOMETRIC_TRAIL_PRESETS,
    "Retrowave Drive": RETROWAVE_DRIVE_PRESETS,
}

GENERATOR_SETTINGS: dict[str, tuple[dict, ...]] = {
    "Neon Grid": (
        {"id": "horizon", "name": "Horizon", "type": "float"},
        {"id": "speed", "name": "Travel Speed", "type": "float"},
        {"id": "ray_count", "name": "Ray Count", "type": "integer"},
        {"id": "depth_lines", "name": "Depth Lines", "type": "integer"},
        {"id": "grid_color", "name": "Grid Color", "type": "color"},
        {"id": "horizon_color", "name": "Horizon Color", "type": "color"},
        {"id": "sky_top", "name": "Sky Top", "type": "color"},
        {"id": "sky_bottom", "name": "Sky Bottom", "type": "color"},
    ),
    "Synth Sun": (
        {"id": "speed", "name": "Travel Speed", "type": "float"},
        {"id": "sun_size", "name": "Sun Size", "type": "float"},
        {"id": "stripe_spacing", "name": "Stripe Spacing", "type": "float"},
        {"id": "sun_color", "name": "Sun Color", "type": "color"},
        {"id": "stripe_color", "name": "Stripe Color", "type": "color"},
        {"id": "grid_color", "name": "Grid Color", "type": "color"},
        {"id": "sky_top", "name": "Sky Top", "type": "color"},
        {"id": "sky_bottom", "name": "Sky Bottom", "type": "color"},
    ),
    "Plasma Field": (
        {"id": "speed", "name": "Motion Speed", "type": "float"},
        {"id": "scale", "name": "Pattern Scale", "type": "float"},
        {"id": "hue_speed", "name": "Hue Speed", "type": "float"},
        {"id": "saturation", "name": "Saturation", "type": "integer"},
        {"id": "brightness", "name": "Brightness", "type": "float"},
    ),
    "Analog Static": (
        {"id": "color", "name": "Color Static", "type": "boolean"},
        {"id": "grain_size", "name": "Grain Size", "type": "integer"},
        {"id": "contrast", "name": "Contrast", "type": "float"},
    ),
    "Geometric Trails": (
        {"id": "shape", "name": "Shape", "type": "choice", "choices": ("Circle", "Ellipse", "Square", "Rectangle", "Diamond", "Triangle")},
        {"id": "path", "name": "Motion Path", "type": "choice", "choices": ("Static Point", "Horizontal Sweep", "Vertical Sweep", "Orbit", "Sine Wave", "Figure Eight", "Lissajous", "Diamond Circuit", "Random Drift")},
        {"id": "animate", "name": "Animate", "type": "boolean"},
        {"id": "copies", "name": "Trail Copies", "type": "integer"},
        {"id": "span", "name": "Trail Span", "type": "float"},
        {"id": "size", "name": "Shape Size", "type": "float"},
        {"id": "tail_scale", "name": "Tail Scale", "type": "float"},
        {"id": "speed", "name": "Motion Speed", "type": "float"},
        {"id": "pulse", "name": "Size Pulse Amount", "type": "float"},
        {"id": "pulse_speed", "name": "Size Pulse Speed", "type": "float"},
        {"id": "flash_speed", "name": "Flash Speed", "type": "float"},
        {"id": "center_x", "name": "Center X", "type": "float"},
        {"id": "center_y", "name": "Center Y", "type": "float"},
        {"id": "range_x", "name": "Horizontal Range", "type": "float"},
        {"id": "range_y", "name": "Vertical Range", "type": "float"},
        {"id": "trail_rotation", "name": "Rotation Per Copy", "type": "float"},
        {"id": "head_color", "name": "Head Color", "type": "color"},
        {"id": "tail_color", "name": "Tail Color", "type": "color"},
        {"id": "filled", "name": "Filled Shapes", "type": "boolean"},
        {"id": "glow", "name": "Glow", "type": "float"},
    ),
    "Retrowave Drive": (
        {"id": "terrain_style", "name": "Terrain Style", "type": "choice", "choices": ("Rolling Hills", "Jagged Mountains", "Canyon", "Symmetrical Valley", "Sparse Peaks", "Flat Infinite Grid")},
        {"id": "travel_speed", "name": "Travel Speed", "type": "float"},
        {"id": "horizon_y", "name": "Horizon Y", "type": "float"},
        {"id": "vanishing_x", "name": "Vanishing Point X", "type": "float"},
        {"id": "road_width", "name": "Road Width", "type": "float"},
        {"id": "road_lanes", "name": "Road Grid Columns", "type": "integer"},
        {"id": "depth_lines", "name": "Depth Lines", "type": "integer"},
        {"id": "terrain_width", "name": "Terrain Width", "type": "float"},
        {"id": "terrain_height", "name": "Terrain Height", "type": "float"},
        {"id": "terrain_roughness", "name": "Terrain Roughness", "type": "float"},
        {"id": "terrain_morph", "name": "Terrain Morph", "type": "float"},
        {"id": "sun_visible", "name": "Show Sun", "type": "boolean"},
        {"id": "sun_size", "name": "Sun Size", "type": "float"},
        {"id": "sun_stripes", "name": "Sun Stripes", "type": "integer"},
        {"id": "line_width", "name": "Grid Line Width", "type": "integer"},
        {"id": "glow_strength", "name": "Glow Strength", "type": "float"},
        {"id": "reflection_strength", "name": "Sun Reflection", "type": "float"},
        {"id": "road_grid_color", "name": "Road Grid Color", "type": "color"},
        {"id": "terrain_color", "name": "Terrain Color", "type": "color"},
        {"id": "sun_top_color", "name": "Sun Top Color", "type": "color"},
        {"id": "sun_bottom_color", "name": "Sun Bottom Color", "type": "color"},
        {"id": "loop_duration", "name": "Loop Duration", "type": "float"},
    ),
}


def generator_preset_names(name: str) -> tuple[str, ...]:
    return tuple(GENERATOR_PRESETS.get(name, {"Original": {}}))


def default_generator_preset(name: str) -> str:
    presets = generator_preset_names(name)
    return presets[0]


def generator_parameters_for_preset(name: str, preset: str | None = None) -> dict:
    defaults = dict(GENERATOR_DEFAULTS.get(name, {}))
    presets = GENERATOR_PRESETS.get(name, {})
    selected = preset if preset in presets else next(iter(presets), None)
    if selected is not None:
        defaults.update(presets[selected])
    return defaults


def _resolve_legacy_generator(
    name: str,
    parameters: dict | None,
) -> tuple[str, dict]:
    if name in GEOMETRIC_TRAIL_PRESETS:
        merged = generator_parameters_for_preset("Geometric Trails", name)
        merged.update(parameters or {})
        return "Geometric Trails", merged
    merged = dict(GENERATOR_DEFAULTS.get(name, {}))
    merged.update(parameters or {})
    return name, merged


def render_generator(
    name: str,
    width: int,
    height: int,
    time_seconds: float,
    seed: int,
    parameters: dict | None = None,
) -> np.ndarray:
    name, parameters = _resolve_legacy_generator(name, parameters)
    parameters.setdefault("seed", seed)
    if name == "Retrowave Drive":
        image = render_retrowave_drive(
            (width, height),
            time_seconds,
            parameters,
            {},
        )
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        background = np.zeros((height, width, 3), dtype=np.float32)
        alpha = rgba[..., 3:4].astype(np.float32) / 255.0
        rgb = rgba[..., :3].astype(np.float32)
        composited = background * (1.0 - alpha) + rgb * alpha
        return np.ascontiguousarray(composited[..., ::-1].astype(np.uint8))
    if name == "Geometric Trails":
        overlay, alpha = geometric_trail_layer(
            width,
            height,
            time_seconds,
            parameters,
        )
        background = np.zeros((height, width, 3), dtype=np.uint8)
        background[:] = (11, 7, 15)
        return np.clip(
            background.astype(np.float32) * (1.0 - alpha[..., None])
            + overlay.astype(np.float32) * alpha[..., None],
            0,
            255,
        ).astype(np.uint8)
    if name == "Synth Sun":
        return _synth_sun(width, height, time_seconds, seed, parameters)
    if name == "Plasma Field":
        return _plasma(width, height, time_seconds, seed, parameters)
    if name == "Analog Static":
        return _static(width, height, time_seconds, seed, parameters)
    return _neon_grid(width, height, time_seconds, seed, parameters)


def render_generator_overlay(
    name: str,
    width: int,
    height: int,
    time_seconds: float,
    seed: int,
    parameters: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Render a generator as BGR color plus a reusable floating alpha mask."""
    name, parameters = _resolve_legacy_generator(name, parameters)
    parameters.setdefault("seed", seed)
    if name == "Retrowave Drive":
        overlay_parameters = dict(parameters)
        overlay_parameters["transparent_background"] = True
        image = render_retrowave_drive(
            (width, height),
            time_seconds,
            overlay_parameters,
            {},
        )
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        return (
            np.ascontiguousarray(rgba[..., :3][..., ::-1]),
            rgba[..., 3].astype(np.float32) / 255.0,
        )
    if name == "Geometric Trails":
        return geometric_trail_layer(
            width,
            height,
            time_seconds,
            parameters,
        )

    frame = render_generator(
        name,
        width,
        height,
        time_seconds,
        seed,
        parameters,
    )
    if name in {"Neon Grid", "Synth Sun"}:
        background = _background(width, height, parameters)
        difference = np.max(
            np.abs(frame.astype(np.int16) - background.astype(np.int16)),
            axis=2,
        ).astype(np.float32)
        alpha = np.clip(difference / 72.0, 0.0, 1.0)
    else:
        alpha = np.ones((height, width), dtype=np.float32)
    return frame, alpha


def _background(width: int, height: int, parameters: dict | None = None) -> np.ndarray:
    parameters = parameters or {}
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    top = np.asarray(_hex_bgr(str(parameters.get("sky_top", "#0E041E"))), dtype=np.float32)
    bottom = np.asarray(_hex_bgr(str(parameters.get("sky_bottom", "#44261E"))), dtype=np.float32)
    rows = top[None, None, :] * (1.0 - y[..., None]) + bottom[None, None, :] * y[..., None]
    frame = np.repeat(rows, width, axis=1)
    return frame.astype(np.uint8)


def _vanishing_point_x(width: int) -> int:
    """Keep the road, horizon, and Synth Sun on one exact center line."""
    return width // 2


def _neon_grid(width: int, height: int, time_seconds: float, seed: int, parameters: dict | None = None) -> np.ndarray:
    parameters = parameters or {}
    frame = _background(width, height, parameters)
    horizon = int(height * float(parameters.get("horizon", 0.47)))
    horizon_color = _hex_bgr(str(parameters.get("horizon_color", "#FF3CD2")))
    grid_color = _hex_bgr(str(parameters.get("grid_color", "#1EEBDC")))
    cv2.line(frame, (0, horizon), (width, horizon), horizon_color, 2)
    center = _vanishing_point_x(width)
    ray_count = max(3, int(parameters.get("ray_count", 33)))
    half = ray_count // 2
    for index in range(-half, half + 1):
        bottom_x = center + index * max(8, width // max(4, half - 2))
        cv2.line(frame, (center, horizon), (bottom_x, height), grid_color, 1)
    phase = (time_seconds * float(parameters.get("speed", 0.34))) % 1.0
    depth_lines = max(4, int(parameters.get("depth_lines", 18)))
    for index in range(depth_lines):
        depth = (index + phase) / depth_lines
        curve = depth * depth
        y = int(horizon + curve * (height - horizon))
        cv2.line(frame, (0, y), (width, y), grid_color, 1)
    return frame


def _synth_sun(width: int, height: int, time_seconds: float, seed: int, parameters: dict | None = None) -> np.ndarray:
    parameters = parameters or {}
    frame = _neon_grid(width, height, time_seconds, seed, parameters)
    center = (_vanishing_point_x(width), int(height * 0.31))
    radius = max(4, int(height * float(parameters.get("sun_size", 0.19))))
    sun_color = _hex_bgr(str(parameters.get("sun_color", "#FF5C41")))
    stripe_color = _hex_bgr(str(parameters.get("stripe_color", "#3E0821")))
    cv2.circle(frame, center, radius, sun_color, -1, lineType=cv2.LINE_AA)
    spacing = max(2, int(height * float(parameters.get("stripe_spacing", 0.014))))
    for y in range(center[1] - radius, center[1] + radius, spacing):
        if y > center[1]:
            vertical_offset = y - center[1]
            half_width = int(math.sqrt(max(0, radius * radius - vertical_offset * vertical_offset)))
            cv2.line(frame, (center[0] - half_width, y), (center[0] + half_width, y), stripe_color, max(1, height // 100))
    return frame


def _plasma(width: int, height: int, time_seconds: float, seed: int, parameters: dict | None = None) -> np.ndarray:
    parameters = parameters or {}
    scale = max(0.1, float(parameters.get("scale", 1.0)))
    speed = float(parameters.get("speed", 1.0))
    x = np.linspace(-3.0, 3.0, width, dtype=np.float32)[None, :] * scale
    y = np.linspace(-2.0, 2.0, height, dtype=np.float32)[:, None] * scale
    phase = seed % 997 / 997.0 * math.tau
    value = (
        np.sin(x * 2.1 + time_seconds * 1.5 * speed + phase)
        + np.sin(y * 3.2 - time_seconds * 1.1 * speed)
        + np.sin((x + y) * 2.4 + time_seconds * 0.8 * speed)
        + np.sin(np.sqrt(x * x + y * y) * 4.8 - time_seconds * 2.0 * speed)
    ) / 4.0
    hue = ((value + 1.0) * 82 + time_seconds * float(parameters.get("hue_speed", 14.0))) % 180
    hsv = np.empty((height, width, 3), dtype=np.uint8)
    hsv[..., 0] = hue.astype(np.uint8)
    hsv[..., 1] = np.clip(int(parameters.get("saturation", 225)), 0, 255)
    brightness = max(0.0, float(parameters.get("brightness", 1.0)))
    hsv[..., 2] = np.clip((130 + (value + 1.0) * 62) * brightness, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _static(width: int, height: int, time_seconds: float, seed: int, parameters: dict | None = None) -> np.ndarray:
    parameters = parameters or {}
    frame_index = int(time_seconds * 30.0)
    rng = np.random.default_rng((seed + frame_index * 65537) & 0xFFFFFFFF)
    grain = max(1, int(parameters.get("grain_size", 1)))
    low_height = math.ceil(height / grain)
    low_width = math.ceil(width / grain)
    luminance = rng.integers(0, 256, (low_height, low_width), dtype=np.uint8)
    if grain > 1:
        luminance = cv2.resize(luminance, (width, height), interpolation=cv2.INTER_NEAREST)
    frame = cv2.cvtColor(luminance, cv2.COLOR_GRAY2BGR)
    if parameters.get("color", True):
        chroma = rng.integers(-30, 31, (height, width), dtype=np.int16)
        frame[..., 0] = np.clip(frame[..., 0].astype(np.int16) + chroma, 0, 255).astype(np.uint8)
        frame[..., 2] = np.clip(frame[..., 2].astype(np.int16) - chroma, 0, 255).astype(np.uint8)
    contrast = max(0.0, float(parameters.get("contrast", 1.0)))
    frame = np.clip((frame.astype(np.float32) - 127.5) * contrast + 127.5, 0, 255).astype(np.uint8)
    return frame


def _hex_bgr(value: str) -> tuple[int, int, int]:
    text = str(value).removeprefix("#")[:6]
    if len(text) != 6:
        return 255, 255, 255
    try:
        red, green, blue = (
            int(text[index:index + 2], 16) for index in (0, 2, 4)
        )
    except ValueError:
        return 255, 255, 255
    return blue, green, red


def _trail_path(
    path: str,
    angle: float,
    lobes_x: int,
    lobes_y: int,
    seed: int = 1,
) -> tuple[float, float]:
    if path == "Static Point":
        return 0.0, 0.0
    if path == "Horizontal Sweep":
        return math.sin(angle), 0.0
    if path == "Vertical Sweep":
        return 0.0, math.sin(angle)
    if path == "Orbit":
        return math.cos(angle), math.sin(angle)
    if path == "Sine Wave":
        return math.sin(angle), math.sin(angle * max(1, lobes_y))
    if path == "Figure Eight":
        return math.sin(angle), math.sin(angle * 2.0) * 0.72
    if path == "Lissajous":
        return (
            math.sin(angle * max(1, lobes_x) + math.pi / 2.0),
            math.sin(angle * max(1, lobes_y)),
        )
    if path == "Random Drift":
        node_count = 12
        rng = np.random.default_rng(seed & 0xFFFFFFFF)
        nodes = rng.uniform(-1.0, 1.0, (node_count, 2))
        position = (angle / math.tau % 1.0) * node_count
        index = int(math.floor(position))
        amount = position - index
        amount = amount * amount * (3.0 - 2.0 * amount)
        first = nodes[index % node_count]
        second = nodes[(index + 1) % node_count]
        point = first * (1.0 - amount) + second * amount
        return float(point[0]), float(point[1])
    cosine, sine = math.cos(angle), math.sin(angle)
    scale = 1.0 / max(abs(cosine) + abs(sine), 1e-6)
    return cosine * scale, sine * scale


def _trail_points(
    shape: str,
    center: tuple[float, float],
    width: float,
    height: float,
    rotation: float,
) -> np.ndarray:
    half_width = max(0.5, width / 2.0)
    half_height = max(0.5, height / 2.0)
    if shape in {"Circle", "Ellipse"}:
        count = max(24, min(96, round(max(width, height) / 3)))
        points = [
            (
                math.cos(index * math.tau / count) * half_width,
                math.sin(index * math.tau / count) * half_height,
            )
            for index in range(count)
        ]
    elif shape in {"Square", "Rectangle"}:
        points = [
            (-half_width, -half_height), (half_width, -half_height),
            (half_width, half_height), (-half_width, half_height),
        ]
    elif shape == "Diamond":
        points = [
            (0.0, -half_height), (half_width, 0.0),
            (0.0, half_height), (-half_width, 0.0),
        ]
    else:
        points = [(0.0, -half_height), (half_width, half_height), (-half_width, half_height)]
    cosine, sine = math.cos(rotation), math.sin(rotation)
    center_x, center_y = center
    return np.asarray(
        [
            (
                round(center_x + x * cosine - y * sine),
                round(center_y + x * sine + y * cosine),
            )
            for x, y in points
        ],
        dtype=np.int32,
    )


def geometric_trail_layer(
    width: int,
    height: int,
    time_seconds: float,
    parameters: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Render a reusable BGR shape-trail layer and a floating-point alpha mask."""
    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    alpha = np.zeros((height, width), dtype=np.float32)
    shape = str(parameters.get("shape", "Circle"))
    path = str(parameters.get("path", "Figure Eight"))
    copies = max(1, min(512, int(parameters.get("copies", 36))))
    span = max(0.0, float(parameters.get("span", 0.8)))
    animate = bool(parameters.get("animate", True))
    phase = time_seconds * float(parameters.get("speed", 0.16)) if animate else 0.0
    center_x = float(parameters.get("center_x", 0.5)) * width
    center_y = float(parameters.get("center_y", 0.5)) * height
    range_x = float(parameters.get("range_x", 0.34)) * width
    range_y = float(parameters.get("range_y", 0.27)) * height
    size = max(1.0, float(parameters.get("size", 0.18)) * min(width, height))
    aspect = max(0.05, float(parameters.get("aspect", 1.5)))
    tail_scale = max(0.02, float(parameters.get("tail_scale", 0.7)))
    tail_opacity = min(1.0, max(0.0, float(parameters.get("tail_opacity", 0.16))))
    pulse = max(0.0, float(parameters.get("pulse", 0.12)))
    pulse_speed = float(parameters.get("pulse_speed", 0.24))
    rotation_speed = float(parameters.get("rotation_speed", 0.0))
    trail_rotation = math.radians(float(parameters.get("trail_rotation", 2.0)))
    line_width = max(1, int(parameters.get("line_width", 2)))
    filled = bool(parameters.get("filled", False))
    lobes_x = max(1, int(parameters.get("lobes_x", 2)))
    lobes_y = max(1, int(parameters.get("lobes_y", 3)))
    seed = int(parameters.get("seed", 1))
    flash_speed = max(0.0, float(parameters.get("flash_speed", 0.0)))
    flash = (
        1.0
        if flash_speed <= 0.0 or (time_seconds * flash_speed) % 1.0 < 0.5
        else 0.0
    )
    head = np.asarray(_hex_bgr(str(parameters.get("head_color", "#72FF8B"))), dtype=np.float32)
    tail = np.asarray(_hex_bgr(str(parameters.get("tail_color", "#20E3FF"))), dtype=np.float32)

    for echo_index in reversed(range(copies)):
        age = echo_index / max(1, copies - 1)
        angle = (phase - age * span) * math.tau
        path_x, path_y = _trail_path(path, angle, lobes_x, lobes_y, seed)
        scale = 1.0 + (tail_scale - 1.0) * age
        if pulse:
            scale *= max(0.02, 1.0 + pulse * math.sin((time_seconds * pulse_speed - age * span) * math.tau))
        shape_width = size * scale
        shape_height = size * scale
        if shape in {"Ellipse", "Rectangle"}:
            shape_width *= aspect
        points = _trail_points(
            shape,
            (center_x + path_x * range_x, center_y + path_y * range_y),
            shape_width,
            shape_height,
            time_seconds * rotation_speed * math.tau + echo_index * trail_rotation,
        )
        color = tuple(int(value) for value in np.rint(head + (tail - head) * age))
        opacity = float(1.0 + (tail_opacity - 1.0) * age) * flash
        if filled:
            cv2.fillPoly(overlay, [points], color, lineType=cv2.LINE_AA)
            cv2.fillPoly(alpha, [points], opacity * 0.5, lineType=cv2.LINE_AA)
        else:
            cv2.polylines(overlay, [points], True, color, line_width, cv2.LINE_AA)
            cv2.polylines(alpha, [points], True, opacity, line_width, cv2.LINE_AA)

    glow = max(0.0, float(parameters.get("glow", 0.65)))
    if glow:
        radius = max(0.1, float(parameters.get("glow_radius", 5.0)))
        blurred_alpha = cv2.GaussianBlur(alpha, (0, 0), radius)
        blurred_color = cv2.GaussianBlur(overlay, (0, 0), radius)
        overlay = np.clip(
            overlay.astype(np.float32) + blurred_color.astype(np.float32) * glow,
            0,
            255,
        ).astype(np.uint8)
        alpha = np.clip(alpha + blurred_alpha * glow * 0.7, 0.0, 1.0)
    return overlay, alpha
