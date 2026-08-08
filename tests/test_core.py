from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import subprocess
import wave

import cv2
import numpy as np

from glitcher.app import GlitcherApp, parse_timecode
from glitcher.effects import apply_effect, event_envelope
from glitcher.effect_catalog import EFFECT_PRESETS, parameters_for_preset
from glitcher.engine import RenderEngine
from glitcher.exporter import CPU_ENCODER, HARDWARE_ENCODERS, ExportCancelled, VideoExporter, detect_h264_encoders, format_duration
from glitcher.generators import (
    GENERATOR_NAMES,
    GEOMETRIC_GENERATOR_NAMES,
    RETROWAVE_DRIVE_PRESETS,
    _vanishing_point_x,
    generator_parameters_for_preset,
    generator_preset_names,
    render_generator,
    render_generator_overlay,
)
from glitcher.media import probe_media
from glitcher.models import Clip, EFFECT_NAMES, EffectEvent, GeneratorOverlay, Project


class ModelTests(unittest.TestCase):
    def test_project_round_trip(self) -> None:
        project = Project(
            name="Test",
            clips=[
                Clip(
                    name="Grid",
                    duration=5,
                    kind="generator",
                    generator="Neon Grid",
                    generator_preset="Cyan Highway",
                    generator_parameters=generator_parameters_for_preset("Neon Grid", "Cyan Highway"),
                )
            ],
        )
        project.source_sizing = "Fill frame (crop)"
        project.audio_duration = 12.5
        project.audio_source_start = 2.25
        project.audio_timeline_start = 1.5
        project.effects.append(EffectEvent(start=1, duration=2.5, effect="RGB Ghost", seed=99))
        project.effects.append(EffectEvent(start=4, duration=1, effect="Posterize", preset="Limited Palette"))
        project.generator_overlays.append(
            GeneratorOverlay(
                start=0.5,
                duration=3.0,
                generator="Neon Grid",
                generator_preset="Magenta Night",
                generator_parameters=generator_parameters_for_preset("Neon Grid", "Magenta Night"),
                opacity=0.6,
                blend_mode="Add",
                seed=44,
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.glitcher.json"
            project.save(path)
            loaded = Project.load(path)
            data = path.read_text(encoding="utf-8")
        self.assertEqual(loaded.name, "Test")
        self.assertEqual(loaded.clips[0].generator, "Neon Grid")
        self.assertEqual(loaded.clips[0].generator_preset, "Cyan Highway")
        self.assertEqual(loaded.clips[0].generator_parameters["grid_color"], "#20E3FF")
        self.assertEqual(loaded.effects[0].seed, 99)
        self.assertEqual(loaded.effects[1].preset, "Limited Palette")
        self.assertEqual(loaded.effects[1].parameters, parameters_for_preset("Posterize", "Limited Palette"))
        self.assertEqual(loaded.generator_overlays[0].generator, "Neon Grid")
        self.assertEqual(loaded.generator_overlays[0].generator_preset, "Magenta Night")
        self.assertEqual(loaded.generator_overlays[0].generator_parameters["grid_color"], "#FF3CAC")
        self.assertEqual(loaded.generator_overlays[0].blend_mode, "Add")
        self.assertEqual(loaded.generator_overlays[0].opacity, 0.6)
        self.assertEqual(loaded.source_sizing, "Fill frame (crop)")
        self.assertEqual(loaded.audio_duration, 12.5)
        self.assertEqual(loaded.audio_source_start, 2.25)
        self.assertEqual(loaded.audio_timeline_start, 1.5)
        self.assertIn('"version": 3', data)

    def test_older_project_without_generator_overlays_still_loads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.glitcher.json"
            path.write_text(
                '{"version": 2, "name": "Legacy", "clips": []}',
                encoding="utf-8",
            )
            loaded = Project.load(path)
        self.assertEqual(loaded.generator_overlays, [])

    def test_seeded_sequence_is_reproducible(self) -> None:
        first = Project(clips=[Clip(name="Grid", duration=30, kind="generator", generator="Neon Grid")], seed=42)
        second = Project(clips=[Clip(name="Grid", duration=30, kind="generator", generator="Neon Grid")], seed=42)
        first.generate_effect_sequence(1.5, 4.0, 0.6, 0.7, 0.35)
        second.generate_effect_sequence(1.5, 4.0, 0.6, 0.7, 0.35)
        left = [(item.start, item.duration, item.effect, item.preset, item.parameters, item.seed) for item in first.effects]
        right = [(item.start, item.duration, item.effect, item.preset, item.parameters, item.seed) for item in second.effects]
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


class AppCallbackTests(unittest.TestCase):
    def test_generator_duration_accepts_timeline_timecodes(self) -> None:
        self.assertEqual(parse_timecode("4:32"), 272.0)
        self.assertEqual(parse_timecode("4;32"), 272.0)
        self.assertEqual(parse_timecode("1:02:03"), 3723.0)
        self.assertEqual(parse_timecode("8.5"), 8.5)
        with self.assertRaises(ValueError):
            parse_timecode("4:72")

    def test_inspector_scroll_ignores_native_combobox_popdown(self) -> None:
        class AppStub:
            def winfo_containing(self, _x: int, _y: int) -> object:
                raise KeyError("popdown")

        class EventStub:
            x_root = 10
            y_root = 20

        self.assertIsNone(GlitcherApp._scroll_inspector(AppStub(), EventStub()))

    def test_library_scroll_ignores_native_combobox_popdown(self) -> None:
        class AppStub:
            def winfo_containing(self, _x, _y):
                raise KeyError("popdown")

        class EventStub:
            x_root = 10
            y_root = 20

        self.assertIsNone(GlitcherApp._scroll_library(AppStub(), EventStub()))


class RenderTests(unittest.TestCase):
    def test_catalog_keeps_six_originals_and_adds_fifteen_effects(self) -> None:
        self.assertEqual(len(EFFECT_NAMES), 21)
        for original in (
            "VHS Tracking Failure",
            "RGB Ghost",
            "Neon Signal Collapse",
            "Static Reconstruction",
            "Vertical Sync Roll",
            "Video Feedback",
        ):
            self.assertIn(original, EFFECT_NAMES)
            self.assertIn("Original", EFFECT_PRESETS[original])
        self.assertIn("Posterize", EFFECT_NAMES)
        self.assertNotIn("Bitmap Posterize", EFFECT_NAMES)
        self.assertNotIn("Rain Overlay", EFFECT_NAMES)
        self.assertNotIn("Snow Overlay", EFFECT_NAMES)
        self.assertTrue(all(len(EFFECT_PRESETS[name]) >= 3 for name in EFFECT_NAMES))
        self.assertEqual(sum(len(presets) for presets in EFFECT_PRESETS.values()), 71)

    def test_export_duration_formatting(self) -> None:
        self.assertEqual(format_duration(0), "0:00")
        self.assertEqual(format_duration(83), "1:23")
        self.assertEqual(format_duration(3723), "1:02:03")

    def test_encoder_detection_always_includes_cpu_fallback(self) -> None:
        self.assertEqual(detect_h264_encoders()[-1], CPU_ENCODER)

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

    def test_geometric_video_generators_render_and_animate(self) -> None:
        self.assertEqual(len(GEOMETRIC_GENERATOR_NAMES), 8)
        for name in GEOMETRIC_GENERATOR_NAMES:
            with self.subTest(generator=name):
                first = render_generator(name, 160, 90, 0.5, 123)
                duplicate = render_generator(name, 160, 90, 0.5, 123)
                later = render_generator(name, 160, 90, 1.5, 123)
                self.assertTrue(np.array_equal(first, duplicate))
                self.assertEqual(first.shape, (90, 160, 3))
                self.assertEqual(first.dtype, np.uint8)
                self.assertGreater(int(first.max()), 20)
                if name != "Static Concentric Signal":
                    self.assertFalse(np.array_equal(first, later))

    def test_every_generator_can_render_as_an_overlay(self) -> None:
        names = GENERATOR_NAMES + GEOMETRIC_GENERATOR_NAMES
        for name in names:
            with self.subTest(generator=name):
                layer, alpha = render_generator_overlay(name, 160, 90, 0.1, 9)
                self.assertEqual(layer.shape, (90, 160, 3))
                self.assertEqual(alpha.shape, (90, 160))
                self.assertEqual(layer.dtype, np.uint8)
                self.assertGreaterEqual(float(alpha.min()), 0.0)
                self.assertLessEqual(float(alpha.max()), 1.0)
                self.assertGreater(float(alpha.max()), 0.0)

    def test_every_generator_exposes_renderable_presets(self) -> None:
        for generator in GENERATOR_NAMES:
            presets = generator_preset_names(generator)
            self.assertGreaterEqual(len(presets), 3, generator)
            for preset in presets:
                with self.subTest(generator=generator, preset=preset):
                    parameters = generator_parameters_for_preset(generator, preset)
                    frame = render_generator(generator, 160, 90, 0.4, 19, parameters)
                    self.assertEqual(frame.shape, (90, 160, 3))
                    self.assertEqual(frame.dtype, np.uint8)

    def test_retrowave_drive_has_all_seven_presets_and_animates(self) -> None:
        self.assertEqual(len(RETROWAVE_DRIVE_PRESETS), 7)
        for preset in RETROWAVE_DRIVE_PRESETS:
            with self.subTest(preset=preset):
                parameters = generator_parameters_for_preset("Retrowave Drive", preset)
                first = render_generator("Retrowave Drive", 160, 90, 0.25, 7, parameters)
                duplicate = render_generator("Retrowave Drive", 160, 90, 0.25, 7, parameters)
                later = render_generator("Retrowave Drive", 160, 90, 1.25, 7, parameters)
                self.assertTrue(np.array_equal(first, duplicate))
                self.assertFalse(np.array_equal(first, later))

    def test_synth_sun_and_road_share_the_exact_canvas_center(self) -> None:
        for width in (160, 640, 641):
            self.assertEqual(_vanishing_point_x(width), width // 2)
        frame = render_generator("Synth Sun", 640, 360, 0.5, 1)
        sun_pixels = np.all(frame == (65, 92, 255), axis=2)
        _y, x = np.nonzero(sun_pixels)
        self.assertTrue(x.size)
        self.assertAlmostEqual(float(x.mean()), _vanishing_point_x(640), delta=1.0)

    def test_synth_sun_stripes_are_clipped_to_the_circle(self) -> None:
        width, height = 640, 360
        grid = render_generator("Neon Grid", width, height, 0.5, 1)
        sun = render_generator("Synth Sun", width, height, 0.5, 1)
        center_x = _vanishing_point_x(width)
        center_y = int(height * 0.31)
        radius = max(24, int(height * 0.19))
        yy, xx = np.ogrid[:height, :width]
        safely_outside_sun = (xx - center_x) ** 2 + (yy - center_y) ** 2 > (radius + 2) ** 2
        self.assertTrue(np.array_equal(sun[safely_outside_sun], grid[safely_outside_sun]))

    def test_every_effect_preserves_frame_contract(self) -> None:
        frame = render_generator("Synth Sun", 160, 90, 0.5, 1)
        for effect in EFFECT_NAMES:
            event = EffectEvent(start=0, duration=4, effect=effect, intensity=0.8, seed=5)
            output = apply_effect(frame, event, 2.0, 30)
            self.assertEqual(output.shape, frame.shape, effect)
            self.assertEqual(output.dtype, np.uint8, effect)

    def test_every_effect_preset_renders_and_changes_pixels(self) -> None:
        frame = render_generator("Synth Sun", 160, 90, 0.5, 1)
        for effect, presets in EFFECT_PRESETS.items():
            for preset, parameters in presets.items():
                event = EffectEvent(
                    start=0,
                    duration=4,
                    effect=effect,
                    preset=preset,
                    parameters=parameters,
                    intensity=1.0,
                    seed=17,
                )
                output = apply_effect(frame, event, 2.0, 30)
                self.assertEqual(output.shape, frame.shape, f"{effect} / {preset}")
                self.assertEqual(output.dtype, np.uint8, f"{effect} / {preset}")
                self.assertFalse(np.array_equal(output, frame), f"{effect} / {preset}")

    def test_halftone_covers_the_full_canvas_without_rotated_frame_edges(self) -> None:
        frame = np.full((180, 100, 3), 128, dtype=np.uint8)
        parameters = parameters_for_preset("Halftone", "Fine Print")
        event = EffectEvent(0, 4, "Halftone", 1.0, preset="Fine Print", parameters=parameters)
        output = apply_effect(frame, event, 2.0, 30)
        foreground = np.array((24, 16, 16), dtype=np.uint8)
        background = np.array((216, 233, 244), dtype=np.uint8)
        valid = np.all(output == foreground, axis=2) | np.all(output == background, axis=2)
        self.assertTrue(np.all(valid))
        self.assertTrue(np.any(np.all(output[:20] == foreground, axis=2)))
        self.assertTrue(np.any(np.all(output[-20:] == foreground, axis=2)))

    def test_original_glitcher_effects_still_render(self) -> None:
        frame = render_generator("Synth Sun", 160, 90, 0.5, 1)
        legacy = (
            "VHS Tracking Failure",
            "RGB Ghost",
            "Neon Signal Collapse",
            "Static Reconstruction",
            "Vertical Sync Roll",
            "Video Feedback",
        )
        for effect in legacy:
            output = apply_effect(frame, EffectEvent(start=0, duration=4, effect=effect, intensity=0.8, seed=5), 2.0, 30)
            self.assertEqual(output.shape, frame.shape, effect)

    def test_static_reconstruction_affects_the_full_frame_without_a_wipe(self) -> None:
        frame = np.full((120, 160, 3), 96, dtype=np.uint8)
        event = EffectEvent(start=0, duration=4, effect="Static Reconstruction", intensity=1.0, seed=5)
        output = apply_effect(frame, event, 2.0, 30)
        difference = np.abs(output.astype(np.int16) - frame.astype(np.int16)).mean(axis=2)
        self.assertGreater(float(difference[:30].mean()), 30.0)
        self.assertGreater(float(difference[-30:].mean()), 30.0)

    def test_render_engine_combines_generator_and_effect(self) -> None:
        project = Project(width=160, height=90, clips=[Clip(name="Grid", duration=5, kind="generator", generator="Neon Grid")])
        project.effects = [EffectEvent(start=0, duration=4, effect="VHS Tracking Failure")]
        engine = RenderEngine(project)
        try:
            frame = engine.render(2.0)
        finally:
            engine.close()
        self.assertEqual(frame.shape, (90, 160, 3))

    def test_render_engine_superimposes_generator_overlay(self) -> None:
        project = Project(
            width=160,
            height=90,
            clips=[
                Clip(
                    name="Grid",
                    duration=5,
                    kind="generator",
                    generator="Neon Grid",
                )
            ],
        )
        plain_engine = RenderEngine(project)
        try:
            plain = plain_engine.render(2.0)
        finally:
            plain_engine.close()
        project.generator_overlays.append(
            GeneratorOverlay(
                start=0,
                duration=5,
                generator="Triangle Orbit",
                opacity=0.9,
                blend_mode="Screen",
            )
        )
        overlay_engine = RenderEngine(project)
        try:
            composited = overlay_engine.render(2.0)
        finally:
            overlay_engine.close()
        self.assertFalse(np.array_equal(plain, composited))

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
        project.effects = [EffectEvent(start=0, duration=0.5, effect="Chromatic Aberration", preset="Signal Split")]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke.mp4"
            updates: list[tuple[float, str]] = []
            VideoExporter(project, encoder="cpu").export(output, lambda amount, text: updates.append((amount, text)))
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 1000)
            info = probe_media(output)
            self.assertTrue(info["has_video"])
            self.assertEqual((info["width"], info["height"]), (160, 90))
            self.assertTrue(any("Encoding with CPU (x264)" in text for _amount, text in updates))
            self.assertTrue(any("Estimating time remaining" in text for _amount, text in updates))
            self.assertEqual(updates[-1][0], 1.0)

    def test_streaming_command_reads_raw_frames_from_stdin(self) -> None:
        project = Project(width=640, height=360, fps=30, clips=[Clip(name="Grid", duration=1, kind="generator")])
        command = VideoExporter(project, encoder="cpu")._command(Path("output.mp4"), CPU_ENCODER)
        self.assertIn("rawvideo", command)
        self.assertIn("640x360", command)
        self.assertIn("pipe:0", command)
        self.assertNotIn("mp4v", command)

    def test_hardware_failure_retries_with_cpu(self) -> None:
        project = Project(width=160, height=90, fps=10, clips=[Clip(name="Grid", duration=0.1, kind="generator")])
        exporter = VideoExporter(project)
        attempted: list[str] = []

        def fake_stream(output: Path, encoder: object, progress: object) -> None:
            attempted.append(encoder.key)
            if encoder.hardware:
                raise subprocess.CalledProcessError(1, ["ffmpeg"])
            output.write_bytes(b"cpu fallback")

        exporter._selected_encoder = lambda: HARDWARE_ENCODERS[0]
        exporter._stream_frames = fake_stream
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fallback.mp4"
            exporter.export(output)
            self.assertEqual(output.read_bytes(), b"cpu fallback")
        self.assertEqual(attempted, ["nvenc", "cpu"])

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
