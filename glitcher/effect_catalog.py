from __future__ import annotations

from copy import deepcopy
from typing import Any

from .generators import GEOMETRIC_TRAIL_PRESETS


EffectParameters = dict[str, Any]


EFFECT_PRESETS: dict[str, dict[str, EffectParameters]] = {
    "VHS Tracking Failure": {
        "Original": {"channel_scale": 1.0, "band_min": 2, "band_max": 6, "shift_scale": 0.10, "bottom_shift": 0.045},
        "Loose Tracking": {"channel_scale": 0.7, "band_min": 3, "band_max": 8, "shift_scale": 0.065, "bottom_shift": 0.025},
        "Tracking Storm": {"channel_scale": 1.45, "band_min": 6, "band_max": 13, "shift_scale": 0.18, "bottom_shift": 0.085},
    },
    "RGB Ghost": {
        "Original": {"offset_base": 0.008, "offset_range": 0.018, "green_drift": 0.35},
        "Soft Echo": {"offset_base": 0.003, "offset_range": 0.009, "green_drift": 0.18},
        "Wide Separation": {"offset_base": 0.018, "offset_range": 0.040, "green_drift": 0.60},
    },
    "Neon Signal Collapse": {
        "Original": {"hue_base": 18, "hue_swing": 28, "saturation": 1.75, "posterize_step": 42, "edge_strength": 0.65, "tracking_scale": 1.0},
        "Candy Breakdown": {"hue_base": 35, "hue_swing": 42, "saturation": 2.10, "posterize_step": 32, "edge_strength": 0.48, "tracking_scale": 0.55},
        "Total Collapse": {"hue_base": 70, "hue_swing": 65, "saturation": 2.55, "posterize_step": 58, "edge_strength": 0.90, "tracking_scale": 1.65},
    },
    "Static Reconstruction": {
        "Original": {"color": False, "grain_size": 1, "source_mix": 0.06, "row_variation": 0.23, "streak_density": 1.0},
        "Color Snow": {"color": True, "grain_size": 1, "source_mix": 0.10, "row_variation": 0.16, "streak_density": 0.65},
        "Coarse Rebuild": {"color": False, "grain_size": 4, "source_mix": 0.18, "row_variation": 0.32, "streak_density": 1.8},
    },
    "Vertical Sync Roll": {
        "Original": {"roll_scale": 1.0, "speed": 1.0, "seam_height": 5},
        "Slow Roll": {"roll_scale": 0.72, "speed": 0.45, "seam_height": 3},
        "Hard Flip": {"roll_scale": 1.0, "speed": 2.2, "seam_height": 11},
    },
    "Video Feedback": {
        "Original": {"echoes": 5, "scale_step": 0.035, "rotation": 0.35, "echo_strength": 0.22, "normalization": 1.28},
        "Soft Tunnel": {"echoes": 7, "scale_step": 0.022, "rotation": 0.18, "echo_strength": 0.14, "normalization": 1.20},
        "Deep Spiral": {"echoes": 9, "scale_step": 0.052, "rotation": 0.90, "echo_strength": 0.31, "normalization": 1.42},
    },
    "Glow": {
        "Soft Bloom": {"radius": 3.5, "strength": 0.55},
        "Neon Halo": {"radius": 8.0, "strength": 1.1},
        "Dream Bloom": {"radius": 15.0, "strength": 0.82},
    },
    "Chromatic Aberration": {
        "Subtle Fringe": {"offset": 2, "pulse": 0.25},
        "Signal Split": {"offset": 7, "pulse": 0.55},
        "Prism Tear": {"offset": 16, "pulse": 0.9},
    },
    "Color Adjustment": {
        "Faded Tape": {"brightness": 0.96, "contrast": 0.82, "saturation": 0.72, "hue": -5},
        "Cold CRT": {"brightness": 0.92, "contrast": 1.18, "saturation": 0.82, "hue": 12},
        "Acid Neon": {"brightness": 1.08, "contrast": 1.35, "saturation": 1.75, "hue": 28},
    },
    "Gaussian Blur": {
        "Soft Focus": {"radius": 1.5, "pulse": 0.15},
        "Analog Smear": {"radius": 4.0, "pulse": 0.35},
        "Defocus": {"radius": 9.0, "pulse": 0.5},
    },
    "Pixelate": {
        "Fine Pixels": {"pixel_size": 4},
        "8-Bit Blocks": {"pixel_size": 9},
        "Chunky Signal": {"pixel_size": 20},
    },
    "Scanlines": {
        "Fine Display": {"spacing": 3, "thickness": 1, "darkness": 0.22},
        "Classic CRT": {"spacing": 4, "thickness": 1, "darkness": 0.38},
        "Heavy Raster": {"spacing": 6, "thickness": 2, "darkness": 0.55},
    },
    "TV Reception": {
        "Weak Reception": {"mode": "Weak Reception", "static": 0.22, "distortion": 8, "band_height": 7, "ghost": 0.12},
        "Heavy Snow": {"mode": "Heavy Snow", "static": 0.72, "distortion": 5, "band_height": 5, "ghost": 0.08},
        "Horizontal Sync": {"mode": "Horizontal Sync", "static": 0.18, "distortion": 28, "band_height": 6, "ghost": 0.16},
        "Rolling Bands": {"mode": "Rolling Bands", "static": 0.20, "distortion": 10, "band_height": 12, "ghost": 0.18},
        "Color Interference": {"mode": "Color Interference", "static": 0.34, "distortion": 12, "band_height": 8, "ghost": 0.28},
        "Signal Dropouts": {"mode": "Signal Dropouts", "static": 0.30, "distortion": 18, "band_height": 15, "ghost": 0.16},
    },
    "Row Glitch": {
        "Light Tearing": {"amount": 8, "slice_height": 4, "density": 0.14},
        "Tracking Failure": {"amount": 28, "slice_height": 7, "density": 0.34},
        "Digital Break": {"amount": 64, "slice_height": 12, "density": 0.62},
    },
    "JPEG Glitch": {
        "Compression Haze": {"quality": 42, "block_shift": 3, "density": 0.16},
        "Corrupted Tape": {"quality": 18, "block_shift": 10, "density": 0.34},
        "Data Collapse": {"quality": 6, "block_shift": 28, "density": 0.58},
    },
    "Posterize": {
        "Retro Color": {"levels": 8, "monochrome": False},
        "Limited Palette": {"levels": 4, "monochrome": False},
        "Monochrome Terminal": {"levels": 4, "monochrome": True},
    },
    "Sine Modulation": {
        "Gentle Ripple": {"amplitude": 4.0, "frequency": 2.0, "speed": 0.7},
        "Tape Wave": {"amplitude": 11.0, "frequency": 4.0, "speed": 1.2},
        "Signal Melt": {"amplitude": 26.0, "frequency": 7.0, "speed": 0.45},
    },
    "Halftone": {
        "Fine Print": {"cell_size": 6, "angle": 45.0, "foreground": "#101018", "background": "#F4E9D8"},
        "Newspaper": {"cell_size": 10, "angle": 30.0, "foreground": "#151515", "background": "#E8E1D2"},
        "Pop Art": {"cell_size": 15, "angle": 45.0, "foreground": "#1420A0", "background": "#F4D84A"},
    },
    "CMYK Halftone": {
        "Fine Registration": {"cell_size": 6, "dot_gain": 0.78},
        "Classic Print": {"cell_size": 10, "dot_gain": 1.0},
        "Bold Misprint": {"cell_size": 15, "dot_gain": 1.38},
    },
    "Declarative Shader": {
        "CRT": {"recipe": "CRT", "channel_offset": 2, "scanline_spacing": 3, "scanline_darkness": 0.30, "vignette": 0.45},
        "VHS": {"recipe": "VHS", "channel_offset": 4, "noise": 10.0, "scanline_spacing": 4, "scanline_darkness": 0.18, "levels": 32},
        "Arcade Monitor": {"recipe": "Arcade Monitor", "channel_offset": 1, "scanline_spacing": 3, "scanline_darkness": 0.42, "vignette": 0.30, "glow": 0.45},
    },
    "Geometric Trails": GEOMETRIC_TRAIL_PRESETS,
}


EFFECT_NAMES = tuple(EFFECT_PRESETS)


def preset_names(effect: str) -> tuple[str, ...]:
    return tuple(EFFECT_PRESETS.get(effect, ()))


def default_preset(effect: str) -> str:
    presets = preset_names(effect)
    return presets[0] if presets else "Legacy"


def parameters_for_preset(effect: str, preset: str | None = None) -> EffectParameters:
    presets = EFFECT_PRESETS.get(effect)
    if not presets:
        return {}
    selected = preset if preset in presets else next(iter(presets))
    return deepcopy(presets[selected])
