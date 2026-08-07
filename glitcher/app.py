from __future__ import annotations

import copy
from pathlib import Path
import queue
import random
import subprocess
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from .effect_catalog import default_preset, parameters_for_preset, preset_names
from .engine import RenderEngine
from .exporter import CPU_ENCODER, EncoderSpec, ExportCancelled, VideoExporter, detect_h264_encoders
from .media import probe_media
from .models import Clip, EFFECT_NAMES, EffectEvent, GENERATOR_NAMES, Project, SOURCE_SIZING_MODES
from .timeline import TimelineCanvas


BG = "#080c0f"
PANEL = "#12191d"
PANEL_2 = "#0d1317"
LINE = "#2a363c"
TEXT = "#ecf5f5"
MUTED = "#93a2a8"
CYAN = "#38e5df"
MAGENTA = "#f04fb8"


class GlitcherApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Video Glitcher v0.2")
        self.geometry("1280x820")
        self.minsize(980, 680)
        self.configure(bg=BG)
        self.project = Project()
        self.project_path: Path | None = None
        self.engine = RenderEngine(self.project, self._preview_render_size())
        self.position = 0.0
        self.playing = False
        self.last_tick = time.perf_counter()
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.exporter: VideoExporter | None = None
        self.audio_process: subprocess.Popen[str] | None = None
        self.export_progress_dialog: tk.Toplevel | None = None
        self.export_progress_var = tk.DoubleVar(value=0.0)
        self.export_progress_text = tk.StringVar(value="Preparing export…")
        self.export_progress_percent = tk.StringVar(value="0%")
        self.available_encoders: tuple[EncoderSpec, ...] = (CPU_ENCODER,)
        self.encoder_probe_complete = False
        self.worker_messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self._configure_style()
        self._build_ui()
        self.bind("<Home>", lambda _event: self.jump_to_start())
        self._refresh_all()
        self.after(15, self._tick)
        self.after(100, self._poll_worker)
        threading.Thread(target=self._encoder_probe_worker, daemon=True).start()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_style(self) -> None:
        # ttk uses a separate native Listbox for Combobox pop-downs on Windows.
        # Configure it explicitly so the popup and readonly field share the
        # application's dark palette instead of inheriting system off-white.
        self.option_add("*TCombobox*Listbox.background", PANEL_2)
        self.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.option_add("*TCombobox*Listbox.selectBackground", CYAN)
        self.option_add("*TCombobox*Listbox.selectForeground", "#071113")
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background=PANEL)
        style.configure("Dark.TFrame", background=BG)
        style.configure("TLabel", background=PANEL, foreground=TEXT)
        style.configure("Muted.TLabel", background=PANEL, foreground=MUTED)
        style.configure("TButton", background=PANEL_2, foreground=TEXT, bordercolor=LINE, padding=(9, 6))
        style.map("TButton", background=[("active", "#1d2a2f")], bordercolor=[("focus", CYAN), ("active", CYAN)])
        style.configure("Accent.TButton", background=CYAN, foreground="#071113", bordercolor=CYAN, padding=(10, 6))
        style.map("Accent.TButton", background=[("active", "#66f2ed")])
        style.configure(
            "TCombobox",
            fieldbackground=PANEL_2,
            background=PANEL_2,
            foreground=TEXT,
            selectbackground=PANEL_2,
            selectforeground=TEXT,
            arrowcolor=TEXT,
            bordercolor=LINE,
            lightcolor=LINE,
            darkcolor=LINE,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", PANEL_2), ("disabled", "#20282d")],
            foreground=[("readonly", TEXT), ("disabled", MUTED)],
            selectbackground=[("readonly", PANEL_2)],
            selectforeground=[("readonly", TEXT)],
            background=[("readonly", PANEL_2), ("active", "#1d2a2f")],
            arrowcolor=[("readonly", TEXT), ("active", CYAN)],
            bordercolor=[("focus", CYAN), ("active", CYAN), ("readonly", LINE)],
        )
        style.configure("TEntry", fieldbackground=PANEL_2, foreground=TEXT, insertcolor=TEXT, bordercolor=LINE)
        style.configure("TScale", background=PANEL)
        style.configure("Horizontal.TProgressbar", background=CYAN, troughcolor=PANEL_2)
        style.configure("Vertical.TScrollbar", background="#253137", troughcolor=PANEL_2, bordercolor=LINE, arrowcolor=TEXT)
        style.map("Vertical.TScrollbar", background=[("active", "#34454c")])

    def _build_ui(self) -> None:
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self._build_toolbar()
        self._build_library()
        self._build_preview()
        self._build_inspector()
        self.timeline = TimelineCanvas(self, self.seek)
        self.timeline.grid(row=2, column=0, columnspan=3, sticky="ew")
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status_var, style="Muted.TLabel", padding=(10, 5)).grid(row=3, column=0, columnspan=3, sticky="ew")

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self, style="Dark.TFrame", padding=(10, 8))
        bar.grid(row=0, column=0, columnspan=3, sticky="ew")
        bar.grid_columnconfigure(5, weight=1)
        logo = tk.Label(bar, text="G", bg=CYAN, fg="#071113", font=("Segoe UI", 13, "bold"), padx=8, pady=3)
        logo.grid(row=0, column=0, padx=(0, 8))
        tk.Label(bar, text="VIDEO GLITCHER", bg=BG, fg=TEXT, font=("Segoe UI", 12, "bold")).grid(row=0, column=1, padx=(0, 18))
        ttk.Button(bar, text="New", command=self.new_project).grid(row=0, column=2, padx=3)
        ttk.Button(bar, text="Open", command=self.open_project).grid(row=0, column=3, padx=3)
        ttk.Button(bar, text="Save", command=self.save_project).grid(row=0, column=4, padx=3)
        self.project_name_var = tk.StringVar(value=self.project.name)
        tk.Label(bar, textvariable=self.project_name_var, bg=BG, fg=MUTED).grid(row=0, column=5)
        ttk.Button(bar, text="Export MP4", style="Accent.TButton", command=self.export_video).grid(row=0, column=6, padx=(8, 0))

    def _build_library(self) -> None:
        panel = ttk.Frame(self, padding=12)
        panel.grid(row=1, column=0, sticky="nsew")
        panel.configure(width=230)
        ttk.Label(panel, text="SOURCES").pack(anchor="w", pady=(0, 7))
        ttk.Button(panel, text="Import video clips", style="Accent.TButton", command=self.import_video).pack(fill="x", pady=3)
        ttk.Button(panel, text="Choose audio source", command=self.import_audio).pack(fill="x", pady=3)
        self.audio_var = tk.StringVar(value="No audio selected")
        ttk.Label(panel, textvariable=self.audio_var, style="Muted.TLabel", wraplength=205).pack(anchor="w", pady=(3, 5))
        audio_timing = ttk.Frame(panel)
        audio_timing.pack(fill="x", pady=(0, 5))
        audio_timing.grid_columnconfigure(1, weight=1)
        ttk.Label(audio_timing, text="Audio in-point", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        self.audio_source_start_var = tk.DoubleVar(value=0.0)
        ttk.Entry(audio_timing, textvariable=self.audio_source_start_var, width=7).grid(row=0, column=1, sticky="e")
        ttk.Label(audio_timing, text="Timeline start", style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(3, 0))
        self.audio_timeline_start_var = tk.DoubleVar(value=0.0)
        ttk.Entry(audio_timing, textvariable=self.audio_timeline_start_var, width=7).grid(row=1, column=1, sticky="e", pady=(3, 0))
        ttk.Button(panel, text="Apply audio timing", command=self.apply_audio_timing).pack(fill="x", pady=(0, 10))
        ttk.Label(panel, text="SOURCE SIZING", style="Muted.TLabel").pack(anchor="w")
        self.source_sizing_var = tk.StringVar(value=self.project.source_sizing)
        sizing = ttk.Combobox(panel, textvariable=self.source_sizing_var, values=SOURCE_SIZING_MODES, state="readonly")
        sizing.pack(fill="x", pady=(3, 10))
        sizing.bind("<<ComboboxSelected>>", self._change_source_sizing)
        ttk.Label(panel, text="CLIP ORDER", style="Muted.TLabel").pack(anchor="w")
        self.clip_list = tk.Listbox(panel, height=10, bg=PANEL_2, fg=TEXT, selectbackground="#1c5f63", selectforeground=TEXT, borderwidth=1, relief="solid", highlightthickness=0, activestyle="none", exportselection=False)
        self.clip_list.pack(fill="both", expand=True, pady=(5, 4))
        clip_actions = ttk.Frame(panel)
        clip_actions.pack(fill="x")
        ttk.Button(clip_actions, text="↑", width=3, command=lambda: self.move_clip(-1)).pack(side="left", padx=(0, 3))
        ttk.Button(clip_actions, text="↓", width=3, command=lambda: self.move_clip(1)).pack(side="left", padx=3)
        ttk.Button(clip_actions, text="Remove", command=self.remove_clip).pack(side="right")
        ttk.Separator(panel).pack(fill="x", pady=12)
        ttk.Label(panel, text="PROCEDURAL GENERATOR").pack(anchor="w")
        self.generator_var = tk.StringVar(value=GENERATOR_NAMES[0])
        ttk.Combobox(panel, textvariable=self.generator_var, values=GENERATOR_NAMES, state="readonly").pack(fill="x", pady=5)
        gen_row = ttk.Frame(panel)
        gen_row.pack(fill="x")
        ttk.Label(gen_row, text="Duration", style="Muted.TLabel").pack(side="left")
        self.generator_duration_var = tk.DoubleVar(value=8.0)
        ttk.Entry(gen_row, textvariable=self.generator_duration_var, width=7).pack(side="right")
        ttk.Button(panel, text="Add generator clip", command=self.add_generator).pack(fill="x", pady=(5, 0))

    def _build_preview(self) -> None:
        center = ttk.Frame(self, style="Dark.TFrame", padding=14)
        center.grid(row=1, column=1, sticky="nsew")
        center.grid_rowconfigure(0, weight=1)
        center.grid_columnconfigure(0, weight=1)
        self.preview = tk.Canvas(center, bg="#030407", highlightbackground=LINE, highlightthickness=1)
        self.preview.grid(row=0, column=0, sticky="nsew")
        self.preview.bind("<Configure>", lambda _event: self._show_frame())
        controls = ttk.Frame(center, style="Dark.TFrame", padding=(0, 10, 0, 0))
        controls.grid(row=1, column=0)
        ttk.Button(controls, text="|◀ Start", command=self.jump_to_start).pack(side="left", padx=3)
        ttk.Button(controls, text="◀ Frame", command=lambda: self.seek(self.position - 1 / self.project.fps)).pack(side="left", padx=3)
        self.play_button = ttk.Button(controls, text="▶ Play", style="Accent.TButton", command=self.toggle_play)
        self.play_button.pack(side="left", padx=3)
        ttk.Button(controls, text="Frame ▶", command=lambda: self.seek(self.position + 1 / self.project.fps)).pack(side="left", padx=3)
        self.time_var = tk.StringVar(value="00:00.00 / 00:00.00")
        ttk.Label(controls, textvariable=self.time_var, style="Muted.TLabel", padding=(12, 0)).pack(side="left")

    def _build_inspector(self) -> None:
        host = ttk.Frame(self)
        self.inspector_host = host
        host.grid(row=1, column=2, sticky="nsew")
        host.grid_rowconfigure(0, weight=1)
        host.grid_columnconfigure(0, weight=1)
        self.inspector_canvas = tk.Canvas(host, width=270, bg=PANEL, highlightthickness=0, borderwidth=0)
        self.inspector_canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(host, orient="vertical", command=self.inspector_canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.inspector_canvas.configure(yscrollcommand=scrollbar.set)
        panel = ttk.Frame(self.inspector_canvas, padding=12)
        inspector_window = self.inspector_canvas.create_window((0, 0), window=panel, anchor="nw")
        panel.bind(
            "<Configure>",
            lambda _event: self.inspector_canvas.configure(scrollregion=self.inspector_canvas.bbox("all")),
        )
        self.inspector_canvas.bind(
            "<Configure>",
            lambda event: self.inspector_canvas.itemconfigure(inspector_window, width=event.width),
        )
        self.bind_all("<MouseWheel>", self._scroll_inspector, add="+")
        ttk.Label(panel, text="GLITCH SEQUENCER").pack(anchor="w")
        ttk.Label(panel, text="Effect", style="Muted.TLabel").pack(anchor="w", pady=(9, 2))
        self.effect_var = tk.StringVar(value=EFFECT_NAMES[0])
        self.effect_combo = ttk.Combobox(panel, textvariable=self.effect_var, values=EFFECT_NAMES, state="readonly")
        self.effect_combo.pack(fill="x")
        self.effect_combo.bind("<<ComboboxSelected>>", self._effect_choice_changed)
        ttk.Label(panel, text="Preset", style="Muted.TLabel").pack(anchor="w", pady=(9, 2))
        self.preset_var = tk.StringVar(value=default_preset(EFFECT_NAMES[0]))
        self.preset_combo = ttk.Combobox(panel, textvariable=self.preset_var, values=preset_names(EFFECT_NAMES[0]), state="readonly")
        self.preset_combo.pack(fill="x")
        self.preset_combo.bind("<<ComboboxSelected>>", self._preset_choice_changed)
        self.preset_summary_var = tk.StringVar(value="")
        ttk.Label(panel, textvariable=self.preset_summary_var, style="Muted.TLabel", wraplength=238).pack(anchor="w", pady=(4, 0))
        ttk.Label(panel, text="Timing mode", style="Muted.TLabel").pack(anchor="w", pady=(9, 2))
        self.timing_var = tk.StringVar(value="Random range")
        timing = ttk.Combobox(panel, textvariable=self.timing_var, values=("Fixed duration", "Random range"), state="readonly")
        timing.pack(fill="x")
        timing.bind("<<ComboboxSelected>>", lambda _event: self._sync_timing_fields())
        self.fixed_frame = ttk.Frame(panel)
        ttk.Label(self.fixed_frame, text="Duration (seconds)", style="Muted.TLabel").pack(side="left")
        self.fixed_duration_var = tk.DoubleVar(value=3.8)
        ttk.Entry(self.fixed_frame, textvariable=self.fixed_duration_var, width=7).pack(side="right")
        self.range_frame = ttk.Frame(panel)
        ttk.Label(self.range_frame, text="Duration range", style="Muted.TLabel").grid(row=0, column=0, columnspan=3, sticky="w")
        self.min_duration_var = tk.DoubleVar(value=1.5)
        self.max_duration_var = tk.DoubleVar(value=6.0)
        ttk.Entry(self.range_frame, textvariable=self.min_duration_var, width=7).grid(row=1, column=0, sticky="ew")
        ttk.Label(self.range_frame, text="to", style="Muted.TLabel").grid(row=1, column=1, padx=6)
        ttk.Entry(self.range_frame, textvariable=self.max_duration_var, width=7).grid(row=1, column=2, sticky="ew")
        self.range_frame.columnconfigure(0, weight=1)
        self.range_frame.columnconfigure(2, weight=1)
        self.range_frame.pack(fill="x", pady=(9, 0))
        self.intensity_var = tk.DoubleVar(value=0.72)
        self.density_var = tk.DoubleVar(value=0.55)
        self.meld_var = tk.DoubleVar(value=0.40)
        self.first_scale_frame = self._scale_field(panel, "Intensity", self.intensity_var)
        self._scale_field(panel, "Sequence density", self.density_var)
        self._scale_field(panel, "Meld probability", self.meld_var)
        seed_row = ttk.Frame(panel)
        seed_row.pack(fill="x", pady=(10, 4))
        ttk.Label(seed_row, text="Seed", style="Muted.TLabel").pack(side="left")
        self.seed_var = tk.IntVar(value=self.project.seed)
        ttk.Entry(seed_row, textvariable=self.seed_var, width=10).pack(side="right")
        ttk.Button(panel, text="Add effect at playhead", command=self.add_effect).pack(fill="x", pady=3)
        ttk.Button(panel, text="Generate full sequence", style="Accent.TButton", command=self.generate_sequence).pack(fill="x", pady=3)
        ttk.Button(panel, text="Randomize seed + sequence", command=self.randomize_sequence).pack(fill="x", pady=3)
        ttk.Label(panel, text="Continuous random coverage", style="Muted.TLabel").pack(anchor="w", pady=(9, 2))
        self.coverage_var = tk.StringVar(value="All clips")
        ttk.Combobox(panel, textvariable=self.coverage_var, values=("Selected clip", "All clips"), state="readonly").pack(fill="x")
        ttk.Button(panel, text="Fill target with random effects", style="Accent.TButton", command=self.fill_random_effects).pack(fill="x", pady=(4, 3))
        ttk.Label(panel, text="EVENTS", style="Muted.TLabel").pack(anchor="w", pady=(12, 3))
        self.effect_list = tk.Listbox(panel, height=7, bg=PANEL_2, fg=TEXT, selectbackground="#1c5f63", selectforeground=TEXT, borderwidth=1, relief="solid", highlightthickness=0, activestyle="none", exportselection=False)
        self.effect_list.pack(fill="both", expand=True)
        self.effect_list.bind("<<ListboxSelect>>", self._select_effect_event)
        ttk.Button(panel, text="Replace with chosen effect", style="Accent.TButton", command=self.replace_selected_effect).pack(fill="x", pady=(5, 3))
        ttk.Button(panel, text="Apply chosen preset", command=self.apply_selected_preset).pack(fill="x", pady=3)
        ttk.Button(panel, text="Reroll selected effect", command=self.reroll_selected_effect).pack(fill="x", pady=3)
        ttk.Button(panel, text="Remove selected event", command=self.remove_effect).pack(fill="x", pady=(5, 0))
        self._update_preset_summary()

    def _effect_choice_changed(self, _event: tk.Event | None = None) -> None:
        effect = self.effect_var.get()
        presets = preset_names(effect)
        self.preset_combo.configure(values=presets)
        self.preset_var.set(default_preset(effect))
        self._update_preset_summary()

    def _preset_choice_changed(self, _event: tk.Event | None = None) -> None:
        self._update_preset_summary()

    def _update_preset_summary(self) -> None:
        parameters = parameters_for_preset(self.effect_var.get(), self.preset_var.get())
        summary = " · ".join(f"{key.replace('_', ' ')}: {value}" for key, value in parameters.items())
        self.preset_summary_var.set(summary)

    def _show_effect_preset(self, effect: str, preset: str) -> None:
        self.effect_var.set(effect)
        presets = preset_names(effect)
        self.preset_combo.configure(values=presets)
        self.preset_var.set(preset if preset in presets else default_preset(effect))
        self._update_preset_summary()

    def _scroll_inspector(self, event: tk.Event) -> str | None:
        widget = self.winfo_containing(event.x_root, event.y_root)
        while widget is not None and widget is not self.inspector_host:
            widget = widget.master
        if widget is None:
            return None
        direction = -1 if event.delta > 0 else 1
        self.inspector_canvas.yview_scroll(direction * 3, "units")
        return "break"

    def _scale_field(self, parent: ttk.Frame, label: str, variable: tk.DoubleVar) -> ttk.Frame:
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=(9, 0))
        value = ttk.Label(frame, text=f"{int(variable.get() * 100)}%", style="Muted.TLabel")
        ttk.Label(frame, text=label, style="Muted.TLabel").pack(side="left")
        value.pack(side="right")
        scale = ttk.Scale(parent, from_=0, to=1, variable=variable, command=lambda current, target=value: target.configure(text=f"{int(float(current) * 100)}%"))
        scale.pack(fill="x")
        return frame

    def _sync_timing_fields(self) -> None:
        if self.timing_var.get() == "Fixed duration":
            self.range_frame.pack_forget()
            self.fixed_frame.pack(fill="x", pady=(9, 0), before=self.first_scale_frame)
        else:
            self.fixed_frame.pack_forget()
            if not self.range_frame.winfo_manager():
                self.range_frame.pack(fill="x", pady=(9, 0), before=self.first_scale_frame)

    def _rebuild_engine(self) -> None:
        self.engine.close()
        self.engine = RenderEngine(self.project, self._preview_render_size())

    def _preview_render_size(self) -> tuple[int, int]:
        """Fit the project canvas inside the preview while preserving its aspect ratio."""
        width = max(1, self.project.width)
        height = max(1, self.project.height)
        scale = min(720 / width, 405 / height, 1.0)
        return max(1, round(width * scale)), max(1, round(height * scale))

    def _change_source_sizing(self, _event: tk.Event | None = None) -> None:
        self.project.source_sizing = self.source_sizing_var.get()
        self._rebuild_engine()
        self._show_frame()
        self.status_var.set(f"Source sizing: {self.project.source_sizing}")

    def _refresh_all(self) -> None:
        self.project_name_var.set(self.project.name)
        self.source_sizing_var.set(self.project.source_sizing)
        self.audio_var.set(Path(self.project.audio_path).name if self.project.audio_path else "No audio selected")
        self.audio_source_start_var.set(self.project.audio_source_start)
        self.audio_timeline_start_var.set(self.project.audio_timeline_start)
        self.clip_list.delete(0, tk.END)
        for clip in self.project.clips:
            marker = "GEN" if clip.kind == "generator" else "VID"
            self.clip_list.insert(tk.END, f"{marker}  {clip.name}  ·  {clip.duration:.1f}s")
        self.effect_list.delete(0, tk.END)
        for event in sorted(self.project.effects, key=lambda item: item.start):
            self.effect_list.insert(tk.END, f"{event.start:05.1f}s  {event.effect} / {event.preset}  ·  {event.duration:.1f}s")
        self.timeline.set_project(self.project)
        self.position = min(self.position, max(0.0, self.project.duration))
        self._show_frame()

    def _show_frame(self) -> None:
        if not hasattr(self, "preview"):
            return
        frame = self.engine.render(self.position)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        canvas_width = max(80, self.preview.winfo_width())
        canvas_height = max(45, self.preview.winfo_height())
        image.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(image)
        self.preview.delete("all")
        self.preview.create_image(canvas_width // 2, canvas_height // 2, image=self.preview_photo, anchor="center")
        self.preview.create_text(10, canvas_height - 10, text="VIDEO GLITCHER PREVIEW", anchor="sw", fill="#e7ffff", font=("Consolas", 9))
        self.timeline.set_position(self.position)
        self.time_var.set(f"{self._format_time(self.position)} / {self._format_time(self.project.duration)}")

    def _tick(self) -> None:
        now = time.perf_counter()
        delta = now - self.last_tick
        self.last_tick = now
        if self.playing and self.project.duration > 0:
            self.position += delta
            if self.position >= self.project.duration:
                self.position = 0.0
                self._stop_audio_preview()
            if self.audio_process is not None and self.audio_process.poll() is not None:
                self.audio_process = None
            if self.audio_process is None:
                self._start_audio_preview()
            self._show_frame()
        self.after(15, self._tick)

    def seek(self, seconds: float) -> None:
        self.position = min(max(0.0, seconds), max(0.0, self.project.duration))
        if self.playing:
            self._restart_audio_preview()
        self._show_frame()

    def jump_to_start(self) -> None:
        """Return the playhead to the beginning of the video sequence."""
        self.seek(0.0)

    def toggle_play(self) -> None:
        if self.project.duration <= 0:
            return
        self.playing = not self.playing
        self.last_tick = time.perf_counter()
        self.play_button.configure(text="❚❚ Pause" if self.playing else "▶ Play")
        if self.playing:
            self._start_audio_preview()
        else:
            self._stop_audio_preview()

    def _audio_source_position(self) -> float | None:
        if not self.project.audio_path or self.position < self.project.audio_timeline_start:
            return None
        source_position = self.project.audio_source_start + self.position - self.project.audio_timeline_start
        if source_position < 0:
            return None
        if self.project.audio_duration > 0 and source_position >= self.project.audio_duration:
            return None
        return source_position

    def _start_audio_preview(self) -> None:
        if self.audio_process is not None and self.audio_process.poll() is None:
            return
        source_position = self._audio_source_position()
        if source_position is None:
            return
        command = [
            "ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
            "-ss", f"{source_position:.6f}", str(Path(self.project.audio_path).resolve()),
        ]
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self.audio_process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creation_flags,
            )
        except OSError:
            self.audio_process = None
            self.status_var.set("Audio preview requires ffplay on PATH")

    def _stop_audio_preview(self) -> None:
        process = self.audio_process
        self.audio_process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=0.3)
        except subprocess.TimeoutExpired:
            process.kill()

    def _restart_audio_preview(self) -> None:
        self._stop_audio_preview()
        if self.playing:
            self._start_audio_preview()

    def import_video(self) -> None:
        paths = filedialog.askopenfilenames(title="Import silent video clips", filetypes=[("Video files", "*.mp4 *.mov *.mkv *.avi *.webm *.m4v"), ("All files", "*.*")])
        if not paths:
            return
        for path in paths:
            info = probe_media(path)
            if not info["has_video"] or float(info["duration"]) <= 0:
                messagebox.showwarning("Unsupported clip", f"Could not read video from:\n{path}")
                continue
            if not self.project.clips:
                source_width = int(info["width"])
                source_height = int(info["height"])
                if source_width > 0 and source_height > 0:
                    self.project.width = source_width
                    self.project.height = source_height
            self.project.clips.append(Clip(name=Path(path).name, path=str(Path(path).resolve()), duration=float(info["duration"])))
        self._rebuild_engine()
        self._refresh_all()
        self.status_var.set(f"Imported {len(paths)} clip(s)")

    def import_audio(self) -> None:
        path = filedialog.askopenfilename(title="Choose a separate audio source", filetypes=[("Audio or video", "*.mp3 *.ogg *.wav *.flac *.m4a *.aac *.mp4 *.mov *.mkv *.webm"), ("All files", "*.*")])
        if not path:
            return
        info = probe_media(path)
        if not info["has_audio"]:
            messagebox.showwarning("No audio stream", "The selected file does not contain a readable audio stream.")
            return
        self.project.audio_path = str(Path(path).resolve())
        self.project.audio_duration = float(info["duration"])
        self.project.audio_source_start = 0.0
        self.project.audio_timeline_start = 0.0
        self._restart_audio_preview()
        self._refresh_all()

    def apply_audio_timing(self) -> None:
        if not self.project.audio_path:
            messagebox.showinfo("No audio source", "Choose an audio source first.")
            return
        try:
            source_start = max(0.0, float(self.audio_source_start_var.get()))
            timeline_start = max(0.0, float(self.audio_timeline_start_var.get()))
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid audio timing", "Enter valid times in seconds.")
            return
        if self.project.audio_duration > 0:
            source_start = min(source_start, max(0.0, self.project.audio_duration - 0.01))
        self.project.audio_source_start = source_start
        self.project.audio_timeline_start = timeline_start
        self.audio_source_start_var.set(source_start)
        self.audio_timeline_start_var.set(timeline_start)
        self.timeline.redraw()
        self._restart_audio_preview()
        self.status_var.set(f"Audio starts at video {timeline_start:.2f}s from source {source_start:.2f}s")

    def add_generator(self) -> None:
        try:
            duration = max(0.5, float(self.generator_duration_var.get()))
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid duration", "Generator duration must be a number.")
            return
        name = self.generator_var.get()
        self.project.clips.append(Clip(name=name, duration=duration, kind="generator", generator=name, seed=self.seed_var.get()))
        self._rebuild_engine()
        self._refresh_all()

    def move_clip(self, direction: int) -> None:
        selection = self.clip_list.curselection()
        if not selection:
            return
        index = selection[0]
        target = index + direction
        if target < 0 or target >= len(self.project.clips):
            return
        self.project.clips[index], self.project.clips[target] = self.project.clips[target], self.project.clips[index]
        self._refresh_all()
        self.clip_list.selection_set(target)

    def remove_clip(self) -> None:
        selection = self.clip_list.curselection()
        if not selection:
            return
        self.project.clips.pop(selection[0])
        self.project.effects = [event for event in self.project.effects if event.start < self.project.duration]
        self._rebuild_engine()
        self._refresh_all()

    def _duration_sample(self) -> float:
        if self.timing_var.get() == "Fixed duration":
            return max(0.2, float(self.fixed_duration_var.get()))
        low = max(0.2, float(self.min_duration_var.get()))
        high = max(low, float(self.max_duration_var.get()))
        return random.Random(self.seed_var.get() + int(self.position * 1000)).triangular(low, high, (low + high) / 2)

    def add_effect(self) -> None:
        if self.project.duration <= 0:
            messagebox.showinfo("No source footage", "Import a video or add a generator first.")
            return
        try:
            duration = min(self._duration_sample(), self.project.duration - self.position)
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid timing", "Enter valid numeric duration values.")
            return
        effect = self.effect_var.get()
        preset = self.preset_var.get()
        self.project.effects.append(EffectEvent(start=self.position, duration=max(0.2, duration), effect=effect, intensity=self.intensity_var.get(), meld=self.meld_var.get(), seed=self.seed_var.get(), preset=preset, parameters=parameters_for_preset(effect, preset)))
        self._refresh_all()

    def generate_sequence(self) -> None:
        if self.project.duration <= 0:
            messagebox.showinfo("No source footage", "Import a video or add a generator first.")
            return
        try:
            if self.timing_var.get() == "Fixed duration":
                low = high = max(0.2, float(self.fixed_duration_var.get()))
            else:
                low = float(self.min_duration_var.get())
                high = float(self.max_duration_var.get())
            self.project.seed = self.seed_var.get()
            self.project.generate_effect_sequence(low, high, self.density_var.get(), self.intensity_var.get(), self.meld_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid settings", "Check the seed and duration values.")
            return
        self._refresh_all()
        self.status_var.set(f"Generated {len(self.project.effects)} seeded glitch events")

    def randomize_sequence(self) -> None:
        self.seed_var.set(random.SystemRandom().randrange(100000, 1000000))
        self.generate_sequence()

    def fill_random_effects(self) -> None:
        if self.project.duration <= 0:
            messagebox.showinfo("No source footage", "Import a video or add a generator first.")
            return
        targets: list[int] | None = None
        if self.coverage_var.get() == "Selected clip":
            selection = self.clip_list.curselection()
            if not selection:
                messagebox.showinfo("Select a clip", "Select a clip in the Clip Order list first.")
                return
            targets = [selection[0]]
        try:
            if self.timing_var.get() == "Fixed duration":
                low = high = max(0.2, float(self.fixed_duration_var.get()))
            else:
                low = float(self.min_duration_var.get())
                high = float(self.max_duration_var.get())
            self.project.seed = self.seed_var.get()
            created = self.project.fill_clips_with_random_effects(
                targets,
                low,
                high,
                self.intensity_var.get(),
                self.meld_var.get(),
            )
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid settings", "Check the seed and duration values.")
            return
        self._refresh_all()
        scope = "selected clip" if targets is not None else "all clips"
        self.status_var.set(f"Filled {scope} with {len(created)} continuous random effects")

    def remove_effect(self) -> None:
        target = self._selected_effect_event()
        if target is None:
            return
        self.project.effects = [event for event in self.project.effects if event.id != target.id]
        self._refresh_all()

    def _selected_effect_event(self) -> EffectEvent | None:
        selection = self.effect_list.curselection()
        if not selection:
            return None
        ordered = sorted(self.project.effects, key=lambda item: item.start)
        index = selection[0]
        return ordered[index] if index < len(ordered) else None

    def _select_effect_event(self, _event: tk.Event | None = None) -> None:
        target = self._selected_effect_event()
        if target is None:
            return
        self._show_effect_preset(target.effect, target.preset)
        self.seek(target.start)
        self.status_var.set(f"Selected {target.effect} / {target.preset} at {target.start:.1f}s")

    def _refresh_and_reselect_effect(self, event_id: str) -> None:
        self._refresh_all()
        ordered = sorted(self.project.effects, key=lambda item: item.start)
        for index, event in enumerate(ordered):
            if event.id == event_id:
                self.effect_list.selection_set(index)
                self.effect_list.see(index)
                break

    def replace_selected_effect(self) -> None:
        target = self._selected_effect_event()
        if target is None:
            messagebox.showinfo("Select an event", "Select an event in the Events list first.")
            return
        replacement = self.effect_var.get()
        preset = self.preset_var.get()
        previous = target.effect
        target.effect = replacement
        target.preset = preset
        target.parameters = parameters_for_preset(replacement, preset)
        self._refresh_and_reselect_effect(target.id)
        self._show_frame()
        self.status_var.set(f"Changed {previous} to {replacement}; timing preserved")

    def apply_selected_preset(self) -> None:
        target = self._selected_effect_event()
        if target is None:
            messagebox.showinfo("Select an event", "Select an event in the Events list first.")
            return
        if target.effect != self.effect_var.get():
            messagebox.showinfo("Effect differs", "Use Replace with chosen effect before applying this preset.")
            return
        target.preset = self.preset_var.get()
        target.parameters = parameters_for_preset(target.effect, target.preset)
        self._refresh_and_reselect_effect(target.id)
        self._show_frame()
        self.status_var.set(f"Applied {target.preset} to {target.effect}")

    def reroll_selected_effect(self) -> None:
        target = self._selected_effect_event()
        if target is None:
            messagebox.showinfo("Select an event", "Select an event in the Events list first.")
            return
        choices = [effect for effect in EFFECT_NAMES if effect != target.effect]
        previous = target.effect
        target.effect = random.SystemRandom().choice(choices)
        target.preset = random.SystemRandom().choice(preset_names(target.effect))
        target.parameters = parameters_for_preset(target.effect, target.preset)
        target.seed = random.SystemRandom().randrange(1, 2_147_483_647)
        self._show_effect_preset(target.effect, target.preset)
        self._refresh_and_reselect_effect(target.id)
        self._show_frame()
        self.status_var.set(f"Rerolled {previous} as {target.effect}; timing preserved")

    def new_project(self) -> None:
        if self.project.clips and not messagebox.askyesno("New project", "Start a new project? Unsaved changes will be discarded."):
            return
        self._stop_audio_preview()
        self.playing = False
        self.play_button.configure(text="▶ Play")
        self.project = Project()
        self.project_path = None
        self.position = 0.0
        self.seed_var.set(self.project.seed)
        self.source_sizing_var.set(self.project.source_sizing)
        self._show_effect_preset(EFFECT_NAMES[0], default_preset(EFFECT_NAMES[0]))
        self._rebuild_engine()
        self._refresh_all()

    def open_project(self) -> None:
        path = filedialog.askopenfilename(title="Open Video Glitcher project", filetypes=[("Video Glitcher project", "*.glitcher.json"), ("JSON", "*.json")])
        if not path:
            return
        try:
            self.project = Project.load(path)
        except (OSError, ValueError) as error:
            messagebox.showerror("Could not open project", str(error))
            return
        self._stop_audio_preview()
        self.playing = False
        self.play_button.configure(text="▶ Play")
        if self.project.audio_path and self.project.audio_duration <= 0 and Path(self.project.audio_path).exists():
            self.project.audio_duration = float(probe_media(self.project.audio_path)["duration"])
        self.project_path = Path(path)
        self.position = 0.0
        self.seed_var.set(self.project.seed)
        self._rebuild_engine()
        self._refresh_all()
        self.status_var.set(f"Opened {Path(path).name}")

    def save_project(self) -> None:
        if self.project_path is None:
            path = filedialog.asksaveasfilename(title="Save Video Glitcher project", defaultextension=".glitcher.json", filetypes=[("Video Glitcher project", "*.glitcher.json")])
            if not path:
                return
            self.project_path = Path(path)
            self.project.name = self.project_path.name.removesuffix(".glitcher.json")
        try:
            self.project.seed = self.seed_var.get()
            self.project.save(self.project_path)
            self._refresh_all()
            self.status_var.set(f"Saved {self.project_path.name}")
        except (OSError, tk.TclError) as error:
            messagebox.showerror("Could not save project", str(error))

    def export_video(self) -> None:
        if self.project.duration <= 0:
            messagebox.showinfo("Nothing to export", "Import a video or add a generator first.")
            return
        path = filedialog.asksaveasfilename(title="Export MP4", defaultextension=".mp4", filetypes=[("MP4 video", "*.mp4")])
        if not path:
            return
        settings = self._export_dialog()
        if settings is None:
            return
        snapshot = copy.deepcopy(self.project)
        snapshot.width, snapshot.height, snapshot.fps, encoder = settings
        self.exporter = VideoExporter(snapshot, encoder=encoder)
        self.status_var.set("Starting export…")
        self._show_export_progress(Path(path).name)
        thread = threading.Thread(target=self._export_worker, args=(path,), daemon=True)
        thread.start()

    def _show_export_progress(self, output_name: str) -> None:
        self._close_export_progress()
        dialog = tk.Toplevel(self)
        self.export_progress_dialog = dialog
        dialog.title("Exporting video")
        dialog.configure(bg=PANEL)
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.protocol("WM_DELETE_WINDOW", self.cancel_export)
        body = ttk.Frame(dialog, padding=20)
        body.grid(row=0, column=0, sticky="nsew")
        ttk.Label(body, text="EXPORTING MP4", font=("Segoe UI", 12, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(body, text=output_name, style="Muted.TLabel", wraplength=390).grid(row=1, column=0, sticky="w", pady=(3, 18))
        progress_row = ttk.Frame(body)
        progress_row.grid(row=2, column=0, sticky="ew")
        progress_row.grid_columnconfigure(0, weight=1)
        ttk.Label(progress_row, textvariable=self.export_progress_text, wraplength=330).grid(row=0, column=0, sticky="w")
        ttk.Label(progress_row, textvariable=self.export_progress_percent, style="Muted.TLabel").grid(row=0, column=1, sticky="e", padx=(12, 0))
        ttk.Progressbar(body, variable=self.export_progress_var, maximum=100, length=420).grid(row=3, column=0, sticky="ew", pady=(8, 18))
        self.export_cancel_button = ttk.Button(body, text="Cancel export", command=self.cancel_export)
        self.export_cancel_button.grid(row=4, column=0, sticky="e")
        self.export_progress_var.set(0.0)
        self.export_progress_text.set("Preparing frames…")
        self.export_progress_percent.set("0%")
        dialog.update_idletasks()
        x = self.winfo_rootx() + max(0, (self.winfo_width() - dialog.winfo_width()) // 2)
        y = self.winfo_rooty() + max(0, (self.winfo_height() - dialog.winfo_height()) // 2)
        dialog.geometry(f"+{x}+{y}")
        dialog.grab_set()
        dialog.focus_force()

    def _close_export_progress(self) -> None:
        dialog = self.export_progress_dialog
        if dialog is not None:
            try:
                dialog.grab_release()
                dialog.destroy()
            except tk.TclError:
                pass
        self.export_progress_dialog = None

    def cancel_export(self) -> None:
        if self.exporter is None:
            self._close_export_progress()
            return
        self.exporter.cancel()
        self.export_progress_text.set("Cancelling after the current frame…")
        self.export_cancel_button.configure(state="disabled", text="Cancelling…")
        self.status_var.set("Cancelling export…")

    def _export_dialog(self) -> tuple[int, int, int, str] | None:
        dialog = tk.Toplevel(self)
        dialog.title("Export settings")
        dialog.configure(bg=PANEL)
        dialog.transient(self)
        dialog.grab_set()
        result: list[tuple[int, int, int, str]] = []
        resolution = tk.StringVar(value=f"{self.project.width}×{self.project.height}")
        fps = tk.IntVar(value=self.project.fps)
        preferred = self.available_encoders[0]
        auto_label = f"Auto — {preferred.label}" if self.encoder_probe_complete else "Auto — detecting hardware…"
        encoder_labels = {auto_label: "auto"}
        encoder_labels.update({spec.label: spec.key for spec in self.available_encoders})
        encoder = tk.StringVar(value=auto_label)
        ttk.Label(dialog, text="Resolution", padding=(14, 12, 14, 2)).grid(row=0, column=0, sticky="w")
        current_resolution = f"{self.project.width}×{self.project.height}"
        resolution_choices = tuple(dict.fromkeys((current_resolution, "854×480", "1280×720", "1920×1080")))
        ttk.Combobox(dialog, textvariable=resolution, values=resolution_choices, state="readonly", width=18).grid(row=1, column=0, padx=14, sticky="ew")
        ttk.Label(dialog, text="Frames per second", padding=(14, 12, 14, 2)).grid(row=2, column=0, sticky="w")
        ttk.Combobox(dialog, textvariable=fps, values=(24, 25, 30, 60), state="readonly", width=18).grid(row=3, column=0, padx=14, sticky="ew")
        ttk.Label(dialog, text="Video encoder", padding=(14, 12, 14, 2)).grid(row=4, column=0, sticky="w")
        ttk.Combobox(dialog, textvariable=encoder, values=tuple(encoder_labels), state="readonly", width=28).grid(row=5, column=0, padx=14, sticky="ew")
        ttk.Label(dialog, text="Auto uses the fastest available GPU encoder.", style="Muted.TLabel", padding=(14, 5, 14, 0)).grid(row=6, column=0, sticky="w")
        buttons = ttk.Frame(dialog, padding=14)
        buttons.grid(row=7, column=0, sticky="ew")
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="right")
        def accept() -> None:
            width_text, height_text = resolution.get().split("×")
            result.append((int(width_text), int(height_text), int(fps.get()), encoder_labels[encoder.get()]))
            dialog.destroy()
        ttk.Button(buttons, text="Export", style="Accent.TButton", command=accept).pack(side="right", padx=6)
        dialog.wait_window()
        return result[0] if result else None

    def _export_worker(self, path: str) -> None:
        try:
            assert self.exporter is not None
            self.exporter.export(path, lambda amount, text: self.worker_messages.put(("progress", (amount, text))))
            self.worker_messages.put(("done", path))
        except ExportCancelled:
            self.worker_messages.put(("cancelled", None))
        except Exception as error:
            self.worker_messages.put(("error", str(error)))

    def _encoder_probe_worker(self) -> None:
        self.worker_messages.put(("encoders", detect_h264_encoders()))

    def _poll_worker(self) -> None:
        try:
            while True:
                kind, payload = self.worker_messages.get_nowait()
                if kind == "progress":
                    amount, text = payload
                    self.status_var.set(f"{text} · {float(amount) * 100:.0f}%")
                    self.export_progress_var.set(float(amount) * 100)
                    self.export_progress_text.set(str(text))
                    self.export_progress_percent.set(f"{float(amount) * 100:.0f}%")
                elif kind == "encoders":
                    self.available_encoders = tuple(payload)
                    self.encoder_probe_complete = True
                elif kind == "done":
                    self.status_var.set(f"Export complete: {Path(str(payload)).name}")
                    self._close_export_progress()
                    messagebox.showinfo("Export complete", f"Saved to:\n{payload}")
                    self.exporter = None
                elif kind == "cancelled":
                    self.status_var.set("Export cancelled")
                    self._close_export_progress()
                    self.exporter = None
                elif kind == "error":
                    self._close_export_progress()
                    messagebox.showerror("Export failed", str(payload))
                    self.status_var.set("Export failed")
                    self.exporter = None
        except queue.Empty:
            pass
        self.after(100, self._poll_worker)

    def _on_close(self) -> None:
        if self.exporter and not messagebox.askyesno("Export in progress", "Cancel the export and close?"):
            return
        if self.exporter:
            self.exporter.cancel()
        self._stop_audio_preview()
        self._close_export_progress()
        self.engine.close()
        self.destroy()

    @staticmethod
    def _format_time(seconds: float) -> str:
        minutes = int(seconds // 60)
        return f"{minutes:02d}:{seconds - minutes * 60:05.2f}"


def main() -> None:
    app = GlitcherApp()
    app.mainloop()


if __name__ == "__main__":
    main()
