from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import random
from pathlib import Path
from typing import Any
from uuid import uuid4

from .effect_catalog import EFFECT_NAMES, default_preset, parameters_for_preset, preset_names


GENERATOR_NAMES = (
    "Neon Grid",
    "Synth Sun",
    "Plasma Field",
    "Analog Static",
)

SOURCE_SIZING_MODES = (
    "Fit entire frame",
    "Fill frame (crop)",
    "Stretch to frame",
)


@dataclass(slots=True)
class Clip:
    name: str
    duration: float
    kind: str = "video"
    path: str | None = None
    generator: str | None = None
    seed: int = 1
    id: str = field(default_factory=lambda: uuid4().hex)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Clip":
        allowed = {"name", "duration", "kind", "path", "generator", "seed", "id"}
        return cls(**{key: value for key, value in data.items() if key in allowed})


@dataclass(slots=True)
class EffectEvent:
    start: float
    duration: float
    effect: str
    intensity: float = 0.7
    meld: float = 0.35
    seed: int = 1
    preset: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        if not self.preset:
            self.preset = default_preset(self.effect)
        if not self.parameters:
            self.parameters = parameters_for_preset(self.effect, self.preset)

    @property
    def end(self) -> float:
        return self.start + self.duration

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EffectEvent":
        allowed = {"start", "duration", "effect", "intensity", "meld", "seed", "preset", "parameters", "id"}
        return cls(**{key: value for key, value in data.items() if key in allowed})


@dataclass
class Project:
    name: str = "Untitled Signal"
    clips: list[Clip] = field(default_factory=list)
    effects: list[EffectEvent] = field(default_factory=list)
    audio_path: str | None = None
    audio_duration: float = 0.0
    audio_source_start: float = 0.0
    audio_timeline_start: float = 0.0
    fps: int = 30
    width: int = 1280
    height: int = 720
    seed: int = 847201
    source_sizing: str = "Fit entire frame"

    @property
    def duration(self) -> float:
        return sum(max(0.0, clip.duration) for clip in self.clips)

    def clip_at(self, time_seconds: float) -> tuple[Clip | None, float]:
        cursor = 0.0
        for clip in self.clips:
            if time_seconds < cursor + clip.duration or clip is self.clips[-1]:
                return clip, max(0.0, time_seconds - cursor)
            cursor += clip.duration
        return None, 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 2,
            "name": self.name,
            "clips": [asdict(clip) for clip in self.clips],
            "effects": [asdict(event) for event in self.effects],
            "audio_path": self.audio_path,
            "audio_duration": self.audio_duration,
            "audio_source_start": self.audio_source_start,
            "audio_timeline_start": self.audio_timeline_start,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "source_sizing": self.source_sizing,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=data.get("name", "Untitled Signal"),
            clips=[Clip.from_dict(item) for item in data.get("clips", [])],
            effects=[EffectEvent.from_dict(item) for item in data.get("effects", [])],
            audio_path=data.get("audio_path"),
            audio_duration=float(data.get("audio_duration", 0.0)),
            audio_source_start=float(data.get("audio_source_start", 0.0)),
            audio_timeline_start=float(data.get("audio_timeline_start", 0.0)),
            fps=int(data.get("fps", 30)),
            width=int(data.get("width", 1280)),
            height=int(data.get("height", 720)),
            seed=int(data.get("seed", 847201)),
            source_sizing=data.get("source_sizing", "Fit entire frame"),
        )

    def generate_effect_sequence(
        self,
        minimum: float,
        maximum: float,
        density: float,
        intensity: float,
        meld: float,
        seed: int | None = None,
    ) -> list[EffectEvent]:
        """Create a deterministic, varied effect arrangement across the project."""
        total = self.duration
        if total <= 0:
            self.effects = []
            return []
        low = max(0.2, min(minimum, maximum))
        high = max(low, max(minimum, maximum))
        density = min(1.0, max(0.0, density))
        rng = random.Random(self.seed if seed is None else seed)
        events: list[EffectEvent] = []
        cursor = rng.uniform(0.25, max(0.3, 2.4 - density * 1.8))
        previous = ""
        while cursor < total:
            # Triangular sampling favors musical, medium-length events.
            duration = min(rng.triangular(low, high, (low + high) / 2), total - cursor)
            choices = [name for name in EFFECT_NAMES if name != previous]
            effect = rng.choice(choices)
            preset = rng.choice(preset_names(effect))
            events.append(
                EffectEvent(
                    start=round(cursor, 3),
                    duration=round(max(0.2, duration), 3),
                    effect=effect,
                    preset=preset,
                    parameters=parameters_for_preset(effect, preset),
                    intensity=intensity,
                    meld=meld,
                    seed=rng.randrange(1, 2_147_483_647),
                )
            )
            previous = effect
            gap = rng.uniform(0.35, max(0.45, 4.6 - density * 4.0))
            overlap = duration * meld * rng.uniform(0.0, 0.55)
            cursor += duration + gap - overlap
        self.effects = events
        return events

    def fill_clips_with_random_effects(
        self,
        clip_indices: list[int] | None,
        minimum: float,
        maximum: float,
        intensity: float,
        meld: float,
        seed: int | None = None,
    ) -> list[EffectEvent]:
        """Continuously cover selected clips with randomized effect phrases.

        Events never cross a clip boundary. A small overlap derived from the
        meld control keeps adjacent phrases active during each handoff.
        """
        if not self.clips:
            return []
        targets = list(range(len(self.clips))) if clip_indices is None else sorted(set(clip_indices))
        targets = [index for index in targets if 0 <= index < len(self.clips)]
        if not targets:
            return []
        low = max(0.2, min(minimum, maximum))
        high = max(low, max(minimum, maximum))
        intensity = min(1.0, max(0.0, intensity))
        meld = min(1.0, max(0.0, meld))
        rng = random.Random(self.seed if seed is None else seed)
        bounds: list[tuple[int, float, float]] = []
        cursor = 0.0
        for index, clip in enumerate(self.clips):
            end = cursor + max(0.0, clip.duration)
            if index in targets:
                bounds.append((index, cursor, end))
            cursor = end

        def overlaps_target(event: EffectEvent) -> bool:
            return any(event.start < end and event.end > start for _, start, end in bounds)

        retained = [event for event in self.effects if not overlaps_target(event)]
        created: list[EffectEvent] = []
        previous = ""
        for _index, clip_start, clip_end in bounds:
            position = clip_start
            previous = ""
            while position < clip_end - 0.0001:
                remaining = clip_end - position
                sampled = rng.triangular(low, high, (low + high) / 2)
                duration = min(sampled, remaining)
                choices = [name for name in EFFECT_NAMES if name != previous]
                effect = rng.choice(choices)
                preset = rng.choice(preset_names(effect))
                event = EffectEvent(
                    start=round(position, 3),
                    duration=round(max(0.001, duration), 3),
                    effect=effect,
                    preset=preset,
                    parameters=parameters_for_preset(effect, preset),
                    intensity=intensity,
                    meld=meld,
                    seed=rng.randrange(1, 2_147_483_647),
                )
                # Rounding must not allow an event to leak into the next clip.
                event.duration = min(event.duration, round(clip_end - event.start, 3))
                created.append(event)
                previous = effect
                if duration >= remaining - 0.0001:
                    break
                overlap = min(duration * meld * 0.25, max(0.0, duration - 0.2))
                position += max(0.2, duration - overlap)
        self.effects = sorted(retained + created, key=lambda item: item.start)
        return created
