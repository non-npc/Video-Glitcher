from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import subprocess
import wave

import cv2
import numpy as np

from glitcher.effects import apply_effect, event_envelope
from glitcher.engine import RenderEngine
from glitcher.exporter import ExportCancelled, VideoExporter
from glitcher.generators import render_generator
from glitcher.media import probe_media
from glitcher.models import Clip, EFFECT_NAMES, EffectEvent, Project


class ModelTests(unittest.TestCase):
    def test_project_round_trip(self) -> None:
        project = Project(name="Test", clips=[Clip(name="Grid", duration=5, kind="generator", generator="Neon Grid")])
        project.source_sizing = "Fill frame (crop)"
        project.audio_duration = 12.5
        project.audio_source_start = 2.25
        project.audio_timeline_start = 1.5
        project.effects.append(EffectEvent(start=1, duration=2.5, effect="RGB Ghost", seed=99))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.glitcher.json"
            project.save(path)
            loaded = Project.load(path)
        self.assertEqual(loaded.name, "Test")
        self.assertEqual(loaded.clips[0].generator, "Neon Grid")
        self.assertEqual(loaded.effects[0].seed, 99)
        self.assertEqual(loaded.source_sizing, "Fill frame (crop)")
        self.assertEqual(loaded.audio_duration, 12.5)
        self.assertEqual(loaded.audio_source_start, 2.25)
        self.assertEqual(loaded.audio_timeline_start, 1.5)

    def test_seeded_sequence_is_reproducible(self) -> None:
        first = Project(clips=[Clip(name="Grid", duration=30, kind="generator", generator="Neon Grid")], seed=42)
        second = Project(clips=[Clip(name="Grid", duration=30, kind="generator", generator="Neon Grid")], seed=42)
        first.generate_effect_sequence(1.5, 4.0, 0.6, 0.7, 0.35)
        second.generate_effect_sequence(1.5, 4.0, 0.6, 0.7, 0.35)
        left = [(item.start, item.duration, item.effect, item.seed) for item in first.effects]
        right = [(item.start, item.duration, item.effect, item.seed) for item in second.effects]
        self.assertEqual(left, right)
        self.assertTrue(all(item.effect in EFFECT_NAMES for item in first.effects))

    def test_random_fill_covers_each_clip_without_crossing_boundaries(self) -> None:
        project = Project(
            clips=[
                Clip(name="First", duration=5),
                Clip(name="Second", duration=7),
            ],
            seed=51,
        )
        created = project.fill_clips_with_random_effects(None, 1.0, 2.0, 0.75, 0.4)
        first = [event for event in created if event.start < 5]
        second = [event for event in created if event.start >= 5]
        self.assertAlmostEqual(first[0].start, 0.0, places=3)
        self.assertAlmostEqual(first[-1].end, 5.0, places=3)
        self.assertAlmostEqual(second[0].start, 5.0, places=3)
        self.assertAlmostEqual(second[-1].end, 12.0, places=3)
        self.assertTrue(all(event.end <= 5.0 for event in first))
        self.assertTrue(all(event.end <= 12.0 for event in second))
        for events in (first, second):
            self.assertTrue(all(right.start <= left.end for left, right in zip(events, events[1:])))

    def test_random_fill_selected_clip_preserves_other_effects(self) -> None:
        existing = EffectEvent(start=0.5, duration=1.0, effect="RGB Ghost", id="keep-me")
        project = Project(clips=[Clip(name="First", duration=4), Clip(name="Second", duration=4)], effects=[existing])
        created = project.fill_clips_with_random_effects([1], 1.0, 2.0, 0.7, 0.3, seed=7)
        self.assertIn("keep-me", [event.id for event in project.effects])
        self.assertTrue(created)
        self.assertTrue(all(4.0 <= event.start and event.end <= 8.0 for event in created))


class RenderTests(unittest.TestCase):
    def test_envelope_is_smooth_and_bounded(self) -> None:
        values = [event_envelope(index / 20 * 4, 4, 0.4) for index in range(21)]
        self.assertEqual(values[0], 0)
        self.assertEqual(values[-1], 0)
        self.assertTrue(all(0 <= value <= 1 for value in values))
        self.assertGreater(max(values), 0.99)

    def test_generator_is_deterministic(self) -> None:
        first = render_generator("Plasma Field", 160, 90, 1.25, 123)
        second = render_generator("Plasma Field", 160, 90, 1.25, 123)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first.shape, (90, 160, 3))

    def test_every_effect_preserves_frame_contract(self) -> None:
        frame = render_generator("Synth Sun", 160, 90, 0.5, 1)
        for effect in EFFECT_NAMES:
            event = EffectEvent(start=0, duration=4, effect=effect, intensity=0.8, seed=5)
            output = apply_effect(frame, event, 2.0, 30)
            self.assertEqual(output.shape, frame.shape, effect)
            self.assertEqual(output.dtype, np.uint8, effect)

    def test_render_engine_combines_generator_and_effect(self) -> None:
        project = Project(width=160, height=90, clips=[Clip(name="Grid", duration=5, kind="generator", generator="Neon Grid")])
        project.effects = [EffectEvent(start=0, duration=4, effect="VHS Tracking Failure")]
        engine = RenderEngine(project)
        try:
            frame = engine.render(2.0)
        finally:
            engine.close()
        self.assertEqual(frame.shape, (90, 160, 3))

    def test_fit_entire_frame_preserves_full_portrait_image(self) -> None:
        source = np.full((200, 100, 3), 255, dtype=np.uint8)
        fitted = RenderEngine._fit_frame(source, 160, 90, "Fit entire frame")
        self.assertEqual(fitted.shape, (90, 160, 3))
        # Portrait footage is fully visible with pillarboxes on both sides.
        self.assertTrue(np.all(fitted[:, :50] == 0))
        self.assertTrue(np.all(fitted[:, 58:102] == 255))

    def test_fill_frame_remains_available_as_explicit_crop_mode(self) -> None:
        source = np.full((200, 100, 3), 255, dtype=np.uint8)
        filled = RenderEngine._fit_frame(source, 160, 90, "Fill frame (crop)")
        self.assertEqual(filled.shape, (90, 160, 3))
        self.assertTrue(np.all(filled == 255))

    def test_effects_are_clipped_to_fitted_video_rectangle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "portrait.mp4"
            writer = cv2.VideoWriter(str(source_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (40, 80))
            self.assertTrue(writer.isOpened())
            for _ in range(5):
                writer.write(np.full((80, 40, 3), 210, dtype=np.uint8))
            writer.release()
            project = Project(
                width=160,
                height=90,
                fps=10,
                source_sizing="Fit entire frame",
                clips=[Clip(name="Portrait", path=str(source_path), duration=0.5)],
                effects=[EffectEvent(start=0, duration=0.5, effect="Static Reconstruction", intensity=1.0, seed=8)],
            )
            engine = RenderEngine(project)
            try:
                rendered = engine.render(0.25)
            finally:
                engine.close()
        self.assertTrue(np.all(rendered[:, :50] == 0))
        self.assertTrue(np.all(rendered[:, 110:] == 0))
        self.assertGreater(int(rendered[:, 58:102].mean()), 0)

    def test_short_video_export(self) -> None:
        project = Project(
            width=160,
            height=90,
            fps=10,
            clips=[Clip(name="Plasma", duration=0.5, kind="generator", generator="Plasma Field")],
        )
        project.effects = [EffectEvent(start=0, duration=0.5, effect="RGB Ghost")]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke.mp4"
            VideoExporter(project).export(output)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 1000)
            self.assertTrue(probe_media(output)["has_video"])

    def test_separate_audio_is_muxed_into_export(self) -> None:
        project = Project(
            width=160,
            height=90,
            fps=10,
            clips=[Clip(name="Grid", duration=0.5, kind="generator", generator="Neon Grid")],
        )
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "tone.wav"
            sample_rate = 8000
            samples = (np.sin(np.arange(sample_rate // 2) * 2 * np.pi * 220 / sample_rate) * 12000).astype(np.int16)
            with wave.open(str(audio), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(sample_rate)
                stream.writeframes(samples.tobytes())
            project.audio_path = str(audio)
            project.audio_source_start = 0.1
            project.audio_timeline_start = 0.2
            output = Path(directory) / "with-audio.mp4"
            VideoExporter(project).export(output)
            info = probe_media(output)
            self.assertTrue(info["has_video"])
            self.assertTrue(info["has_audio"])
            decode = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(output), "-map", "0:a:0", "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "-"],
                capture_output=True,
                check=True,
            )
            decoded = np.frombuffer(decode.stdout, dtype=np.int16)
            self.assertLess(float(np.abs(decoded[: int(sample_rate * 0.12)]).mean()), 500)
            self.assertGreater(float(np.abs(decoded[int(sample_rate * 0.27): int(sample_rate * 0.38)]).mean()), 1000)

    def test_cancelled_export_does_not_leave_partial_output(self) -> None:
        project = Project(
            width=160,
            height=90,
            fps=10,
            clips=[Clip(name="Static", duration=1, kind="generator", generator="Analog Static")],
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cancelled.mp4"
            exporter = VideoExporter(project)
            exporter.cancel()
            with self.assertRaises(ExportCancelled):
                exporter.export(output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
