from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ProgressCallback = Callable[[int, str], None]

TERRAIN_STYLES = (
    "Rolling Hills",
    "Jagged Mountains",
    "Canyon",
    "Symmetrical Valley",
    "Sparse Peaks",
    "Flat Infinite Grid",
)


def _rgba(value: str, default: str = "#FFFFFFFF") -> tuple[int, int, int, int]:
    text = str(value or default).removeprefix("#")
    if len(text) == 6:
        text += "FF"
    if len(text) != 8:
        text = default.removeprefix("#")
    return tuple(int(text[index:index + 2], 16) for index in range(0, 8, 2))


def _vertical_gradient(
    size: tuple[int, int],
    top: tuple[int, int, int, int],
    bottom: tuple[int, int, int, int],
) -> Image.Image:
    width, height = size
    amount = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    first = np.asarray(top, dtype=np.float32)[None, None, :]
    second = np.asarray(bottom, dtype=np.float32)[None, None, :]
    rows = np.rint(first * (1.0 - amount) + second * amount).astype(np.uint8)
    array = np.repeat(rows, width, axis=1)
    return Image.fromarray(array, mode="RGBA")


def _project(
    x: float,
    elevation: float,
    depth: float,
    *,
    vanishing_x: float,
    horizon_y: float,
    focal_length: float,
    camera_height: float,
) -> tuple[float, float]:
    safe_depth = max(0.001, float(depth))
    return (
        vanishing_x + float(x) * focal_length / safe_depth,
        horizon_y
        + (camera_height - float(elevation)) * focal_length / safe_depth,
    )


def _terrain_height(
    x: float,
    world_depth: float,
    *,
    road_half_width: float,
    terrain_width: float,
    maximum_height: float,
    roughness: float,
    style: str,
    loop_length: float,
    phases: np.ndarray,
    morph_phase: float,
    morph_amount: float,
) -> float:
    distance = max(0.0, abs(x) - road_half_width)
    span = max(0.001, terrain_width - road_half_width)
    edge = np.clip(distance / span, 0.0, 1.0)
    edge = edge * edge * (3.0 - 2.0 * edge)
    if edge <= 0.0 or style == "Flat Infinite Grid":
        return 0.0

    side = -1.0 if x < 0 else 1.0
    side_phase = 0.0 if style == "Symmetrical Valley" else side * phases[4]
    depth_angle = world_depth / max(0.001, loop_length) * np.pi * 2.0
    cross_angle = distance / span * np.pi
    first = np.sin(
        depth_angle * 2.0
        + cross_angle * 1.25
        + phases[0]
        + side_phase
    )
    second = np.sin(
        depth_angle * 5.0
        - cross_angle * 2.1
        + phases[1]
        - side_phase * 0.7
    )
    third = np.sin(
        depth_angle * 9.0
        + cross_angle * 3.7
        + phases[2]
        + side_phase * 1.3
    )
    broad = np.sin(
        depth_angle
        - cross_angle * 0.6
        + phases[3]
    )
    value = (
        0.48
        + first * 0.24
        + second * 0.16 * roughness
        + third * 0.09 * roughness
        + broad * 0.18
    )

    if style == "Rolling Hills":
        shaped = np.clip(value, 0.0, 1.0) ** 1.45
    elif style == "Jagged Mountains":
        ridge = 1.0 - abs(first * 0.58 + second * 0.3 + third * 0.12)
        shaped = np.clip(value * 0.55 + ridge ** 3.0 * 0.8, 0.0, 1.35)
    elif style == "Canyon":
        wall = np.clip((edge - 0.12) / 0.88, 0.0, 1.0) ** 0.7
        shaped = np.clip(value * 0.45 + wall * (0.48 + broad * 0.2), 0.0, 1.3)
    elif style == "Symmetrical Valley":
        shaped = np.clip(value, 0.0, 1.0) ** 1.25
    else:
        shaped = np.clip((value - 0.48) / 0.52, 0.0, 1.0) ** 2.2

    if morph_amount:
        morph = np.sin(
            morph_phase
            + depth_angle * 3.0
            + cross_angle * 1.7
            + phases[5]
        )
        shaped *= max(0.0, 1.0 + morph * morph_amount)
    return float(maximum_height * edge ** 0.55 * shaped)


def _depth_color(
    color: tuple[int, int, int, int],
    depth: float,
    near: float,
    far: float,
) -> tuple[int, int, int, int]:
    fade = 1.0 - np.clip((depth - near) / max(0.001, far - near), 0.0, 1.0)
    strength = 0.2 + fade * 0.8
    return (
        round(color[0] * (0.55 + strength * 0.45)),
        round(color[1] * (0.55 + strength * 0.45)),
        round(color[2] * (0.55 + strength * 0.45)),
        round(color[3] * strength),
    )


def _near_terrain_fade(
    depth: float,
    near_depth: float,
    grid_spacing: float,
) -> float:
    amount = np.clip(
        (depth - near_depth) / max(0.001, grid_spacing * 2.75),
        0.0,
        1.0,
    )
    return float(amount * amount * (3.0 - 2.0 * amount))


def _draw_sun(
    size: tuple[int, int],
    parameters: dict,
    horizon_y: float,
) -> tuple[Image.Image, Image.Image]:
    width, height = size
    sun = Image.new("RGBA", size, (0, 0, 0, 0))
    glow = Image.new("RGBA", size, (0, 0, 0, 0))
    if not parameters.get("sun_visible", True):
        return sun, glow

    center_x = float(parameters.get("sun_x", 0.5)) * width
    center_y = float(parameters.get("sun_y", 0.28)) * height
    radius = max(
        2.0,
        float(parameters.get("sun_size", 0.14)) * min(width, height),
    )
    stripe_count = max(0, int(parameters.get("sun_stripes", 6)))
    stripe_ratio = max(
        0.05,
        min(0.9, float(parameters.get("sun_stripe_gap", 0.38))),
    )
    top = _rgba(parameters.get("sun_top_color", "#FFF36BFF"))
    bottom = _rgba(parameters.get("sun_bottom_color", "#FF3AAEFF"))
    draw = ImageDraw.Draw(sun)
    glow_draw = ImageDraw.Draw(glow)
    glow_draw.ellipse(
        (
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
        ),
        fill=bottom,
    )
    stripe_height = radius * 0.92 / max(1, stripe_count)
    striped_start = center_y - radius * 0.02
    for y in range(
        max(0, int(np.floor(center_y - radius))),
        min(height, int(np.ceil(center_y + radius)) + 1),
    ):
        relative_y = (y - center_y) / radius
        extent = radius * np.sqrt(max(0.0, 1.0 - relative_y * relative_y))
        if stripe_count and y >= striped_start:
            stripe_position = (y - striped_start) / max(0.001, stripe_height)
            if stripe_position % 1.0 < stripe_ratio:
                continue
        amount = np.clip((y - (center_y - radius)) / (radius * 2.0), 0.0, 1.0)
        color = tuple(
            round(top[channel] * (1.0 - amount) + bottom[channel] * amount)
            for channel in range(4)
        )
        draw.line(
            (center_x - extent, y, center_x + extent, y),
            fill=color,
            width=1,
        )
    if center_y + radius > horizon_y:
        mask = Image.new("L", size, 255)
        ImageDraw.Draw(mask).rectangle(
            (0, int(horizon_y), width, height),
            fill=0,
        )
        sun.putalpha(Image.composite(sun.getchannel("A"), mask, mask))
    return sun, glow


def render_retrowave_drive(
    size: tuple[int, int],
    time_seconds: float,
    parameters: dict,
    _inputs: dict,
    progress: ProgressCallback | None = None,
) -> Image.Image:
    width, height = size
    horizon_y = float(parameters.get("horizon_y", 0.44)) * height
    vanishing_x = float(parameters.get("vanishing_x", 0.5)) * width
    horizon_y = np.clip(horizon_y, 1.0, max(1.0, height - 2.0))
    vanishing_x = np.clip(vanishing_x, 0.0, max(0.0, width - 1.0))
    camera_height = max(0.2, float(parameters.get("camera_height", 1.2)))
    near_depth = 1.0
    focal_length = (
        max(1.0, height - horizon_y) * near_depth / camera_height
    )
    road_half_width = max(0.2, float(parameters.get("road_width", 2.3)) / 2.0)
    terrain_width = max(
        road_half_width + 0.5,
        float(parameters.get("terrain_width", 6.5)),
    )
    grid_spacing = max(0.15, float(parameters.get("grid_spacing", 1.35)))
    depth_line_count = max(6, int(parameters.get("depth_lines", 28)))
    loop_length = grid_spacing * depth_line_count
    far_depth = loop_length
    loop_duration = max(0.1, float(parameters.get("loop_duration", 6.0)))
    phase = (float(time_seconds) % loop_duration) / loop_duration
    travel_speed = float(parameters.get("travel_speed", 1.0))
    travel_cycles = max(1, round(abs(travel_speed)))
    direction = 1.0 if travel_speed >= 0.0 else -1.0
    camera_depth = (
        phase * travel_cycles * loop_length * direction
    ) % loop_length
    morph_amount = max(
        0.0,
        min(1.0, float(parameters.get("terrain_morph", 0.0))),
    )
    morph_cycles = max(1, int(parameters.get("morph_cycles", 1)))
    morph_phase = phase * morph_cycles * np.pi * 2.0

    transparent = bool(parameters.get("transparent_background", False))
    if transparent:
        output = Image.new("RGBA", size, (0, 0, 0, 0))
    else:
        output = _vertical_gradient(
            size,
            _rgba(parameters.get("sky_top_color", "#05000FFF")),
            _rgba(parameters.get("sky_horizon_color", "#351068FF")),
        )
    draw = ImageDraw.Draw(output)
    ground_color = (
        (0, 0, 0, 0)
        if transparent
        else _rgba(parameters.get("ground_color", "#07000FFF"))
    )
    draw.rectangle((0, horizon_y, width, height), fill=ground_color)
    if progress is not None:
        progress(15, "Drawing retrowave sky")

    sun, sun_glow_source = _draw_sun(size, parameters, horizon_y)
    glow_strength = max(
        0.0,
        min(3.0, float(parameters.get("glow_strength", 0.9))),
    )
    if glow_strength:
        sun_glow = sun_glow_source.filter(
            ImageFilter.GaussianBlur(radius=max(2.0, min(width, height) * 0.035))
        )
        alpha = np.asarray(sun_glow.getchannel("A"), dtype=np.float32)
        sun_glow.putalpha(
            Image.fromarray(
                np.clip(alpha * glow_strength * 0.58, 0, 255).astype(np.uint8)
            )
        )
        output = Image.alpha_composite(output, sun_glow)
    output = Image.alpha_composite(output, sun)

    road_surface = Image.new("RGBA", size, (0, 0, 0, 0))
    road_draw = ImageDraw.Draw(road_surface)
    left_near = _project(
        -road_half_width,
        0.0,
        near_depth,
        vanishing_x=vanishing_x,
        horizon_y=horizon_y,
        focal_length=focal_length,
        camera_height=camera_height,
    )
    right_near = _project(
        road_half_width,
        0.0,
        near_depth,
        vanishing_x=vanishing_x,
        horizon_y=horizon_y,
        focal_length=focal_length,
        camera_height=camera_height,
    )
    road_color = (
        (0, 0, 0, 0)
        if transparent
        else _rgba(parameters.get("road_color", "#09000FFF"))
    )
    road_draw.polygon(
        (
            (vanishing_x, horizon_y),
            left_near,
            (left_near[0], height),
            (right_near[0], height),
            right_near,
        ),
        fill=road_color,
    )

    reflection_strength = max(
        0.0,
        min(1.0, float(parameters.get("reflection_strength", 0.28))),
    )
    if reflection_strength:
        reflection = Image.new("RGBA", size, (0, 0, 0, 0))
        reflection_draw = ImageDraw.Draw(reflection)
        reflection_color = _rgba(
            parameters.get("sun_bottom_color", "#FF3AAEFF")
        )
        reflection_draw.polygon(
            (
                (vanishing_x - width * 0.012, horizon_y),
                (vanishing_x + width * 0.012, horizon_y),
                (vanishing_x + width * 0.12, height),
                (vanishing_x - width * 0.12, height),
            ),
            fill=reflection_color[:3]
            + (round(150 * reflection_strength),),
        )
        reflection = reflection.filter(
            ImageFilter.GaussianBlur(radius=max(2.0, width * 0.022))
        )
        road_surface = Image.alpha_composite(road_surface, reflection)
    output = Image.alpha_composite(output, road_surface)

    lines = Image.new("RGBA", size, (0, 0, 0, 0))
    line_draw = ImageDraw.Draw(lines)
    road_grid_color = _rgba(
        parameters.get("road_grid_color", "#FF43D1FF")
    )
    terrain_color = _rgba(
        parameters.get("terrain_color", "#FF4FD8FF")
    )
    line_width = max(1, int(parameters.get("line_width", 2)))
    road_lanes = max(1, int(parameters.get("road_lanes", 4)))
    for lane in range(road_lanes + 1):
        x = -road_half_width + lane / road_lanes * road_half_width * 2.0
        near_point = _project(
            x,
            0.0,
            near_depth,
            vanishing_x=vanishing_x,
            horizon_y=horizon_y,
            focal_length=focal_length,
            camera_height=camera_height,
        )
        line_draw.line(
            (near_point, (vanishing_x, horizon_y)),
            fill=road_grid_color,
            width=line_width,
        )

    first_grid = np.floor(
        (camera_depth + near_depth) / grid_spacing
    ) * grid_spacing
    world_grid_depths = np.arange(
        first_grid,
        camera_depth + far_depth + grid_spacing,
        grid_spacing,
    )
    relative_grid_depths = [
        float(world_depth - camera_depth)
        for world_depth in world_grid_depths
        if near_depth <= world_depth - camera_depth <= far_depth
    ]
    relative_grid_depths.sort(reverse=True)
    for depth in relative_grid_depths:
        left = _project(
            -road_half_width,
            0.0,
            depth,
            vanishing_x=vanishing_x,
            horizon_y=horizon_y,
            focal_length=focal_length,
            camera_height=camera_height,
        )
        right = _project(
            road_half_width,
            0.0,
            depth,
            vanishing_x=vanishing_x,
            horizon_y=horizon_y,
            focal_length=focal_length,
            camera_height=camera_height,
        )
        line_draw.line(
            (left, right),
            fill=_depth_color(
                road_grid_color,
                depth,
                near_depth,
                far_depth,
            ),
            width=line_width,
        )

    style = str(parameters.get("terrain_style", "Rolling Hills"))
    if style not in TERRAIN_STYLES:
        style = "Rolling Hills"
    maximum_height = max(0.0, float(parameters.get("terrain_height", 1.7)))
    roughness = max(
        0.0,
        min(2.0, float(parameters.get("terrain_roughness", 0.75))),
    )
    terrain_columns = max(3, int(parameters.get("terrain_columns", 11)))
    terrain_samples = max(12, int(parameters.get("terrain_samples", 48)))
    rng = np.random.default_rng(int(parameters.get("seed", 1986)) % (2 ** 32))
    phases = rng.uniform(0.0, np.pi * 2.0, 6)
    left_x = np.linspace(-terrain_width, -road_half_width, terrain_columns)
    right_x = np.linspace(road_half_width, terrain_width, terrain_columns)
    terrain_x_groups = (left_x, right_x)

    for depth in relative_grid_depths:
        world_depth = (camera_depth + depth) % loop_length
        for x_values in terrain_x_groups:
            points = []
            for x in x_values:
                elevation = _terrain_height(
                    float(x),
                    world_depth,
                    road_half_width=road_half_width,
                    terrain_width=terrain_width,
                    maximum_height=maximum_height,
                    roughness=roughness,
                    style=style,
                    loop_length=loop_length,
                    phases=phases,
                    morph_phase=morph_phase,
                    morph_amount=morph_amount,
                )
                elevation *= _near_terrain_fade(
                    depth,
                    near_depth,
                    grid_spacing,
                )
                points.append(
                    _project(
                        float(x),
                        elevation,
                        depth,
                        vanishing_x=vanishing_x,
                        horizon_y=horizon_y,
                        focal_length=focal_length,
                        camera_height=camera_height,
                    )
                )
            line_draw.line(
                points,
                fill=_depth_color(
                    terrain_color,
                    depth,
                    near_depth,
                    far_depth,
                ),
                width=line_width,
            )

    sample_depths = np.geomspace(
        near_depth,
        far_depth,
        terrain_samples,
    )
    for x_values in terrain_x_groups:
        for x in x_values:
            previous = None
            previous_depth = None
            for depth in reversed(sample_depths):
                world_depth = (camera_depth + float(depth)) % loop_length
                elevation = _terrain_height(
                    float(x),
                    world_depth,
                    road_half_width=road_half_width,
                    terrain_width=terrain_width,
                    maximum_height=maximum_height,
                    roughness=roughness,
                    style=style,
                    loop_length=loop_length,
                    phases=phases,
                    morph_phase=morph_phase,
                    morph_amount=morph_amount,
                )
                elevation *= _near_terrain_fade(
                    float(depth),
                    near_depth,
                    grid_spacing,
                )
                point = _project(
                    float(x),
                    elevation,
                    float(depth),
                    vanishing_x=vanishing_x,
                    horizon_y=horizon_y,
                    focal_length=focal_length,
                    camera_height=camera_height,
                )
                if previous is not None and previous_depth is not None:
                    line_draw.line(
                        (previous, point),
                        fill=_depth_color(
                            terrain_color,
                            min(float(depth), previous_depth),
                            near_depth,
                            far_depth,
                        ),
                        width=line_width,
                    )
                previous = point
                previous_depth = float(depth)
    if progress is not None:
        progress(80, f"Drawing {style.lower()}")

    if glow_strength:
        line_glow = lines.filter(
            ImageFilter.GaussianBlur(radius=max(1.0, line_width * 2.2))
        )
        alpha = np.asarray(line_glow.getchannel("A"), dtype=np.float32)
        line_glow.putalpha(
            Image.fromarray(
                np.clip(alpha * glow_strength * 0.72, 0, 255).astype(np.uint8)
            )
        )
        output = Image.alpha_composite(output, line_glow)
    output = Image.alpha_composite(output, lines)
    if progress is not None:
        progress(100, "Retrowave drive complete")
    return output
