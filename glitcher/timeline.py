from __future__ import annotations

import tkinter as tk
from collections.abc import Callable

from .models import Project


class TimelineCanvas(tk.Canvas):
    TRACK_LABEL_WIDTH = 70

    def __init__(self, master: tk.Misc, seek_callback: Callable[[float], None]) -> None:
        super().__init__(master, height=170, highlightthickness=0, cursor="hand2")
        self.project = Project()
        self.position = 0.0
        self.seek_callback = seek_callback
        self.bind("<Configure>", lambda _event: self.redraw())
        self.bind("<Button-1>", self._seek)

    def set_project(self, project: Project) -> None:
        self.project = project
        self.redraw()

    def set_position(self, seconds: float) -> None:
        self.position = seconds
        self.redraw()

    def _time_to_x(self, seconds: float) -> float:
        usable = max(1, self.winfo_width() - self.TRACK_LABEL_WIDTH - 10)
        return self.TRACK_LABEL_WIDTH + seconds / max(0.001, self.project.duration) * usable

    def _seek(self, event: tk.Event) -> None:
        if event.x < self.TRACK_LABEL_WIDTH or self.project.duration <= 0:
            return
        usable = max(1, self.winfo_width() - self.TRACK_LABEL_WIDTH - 10)
        seconds = (event.x - self.TRACK_LABEL_WIDTH) / usable * self.project.duration
        self.seek_callback(min(self.project.duration, max(0.0, seconds)))

    def redraw(self) -> None:
        self.delete("all")
        width = max(1, self.winfo_width())
        bg = "#0d1216"
        panel = "#151c20"
        line = "#2d383d"
        text = "#94a4aa"
        cyan = "#37ded9"
        magenta = "#ec55b5"
        lime = "#aade4e"
        amber = "#eaa64a"
        self.configure(background=bg)
        self.create_rectangle(0, 0, width, 170, fill=bg, outline="")
        for y in (27, 72, 117, 162):
            self.create_line(0, y, width, y, fill=line)
        self.create_text(8, 49, text="VIDEO", anchor="w", fill=text, font=("Segoe UI", 9))
        self.create_text(8, 94, text="GLITCH", anchor="w", fill=text, font=("Segoe UI", 9))
        self.create_text(8, 139, text="AUDIO", anchor="w", fill=text, font=("Segoe UI", 9))
        if self.project.duration <= 0:
            self.create_text(self.TRACK_LABEL_WIDTH + 12, 84, text="Import a clip or add a generator to begin", anchor="w", fill=text)
            return
        for index in range(7):
            seconds = self.project.duration * index / 6
            x = self._time_to_x(seconds)
            self.create_line(x, 20, x, 162, fill=line)
            self.create_text(x + 3, 11, text=self._format_time(seconds), anchor="w", fill=text, font=("Consolas", 8))
        cursor = 0.0
        for index, clip in enumerate(self.project.clips):
            x1 = self._time_to_x(cursor) + 1
            x2 = self._time_to_x(cursor + clip.duration) - 1
            color = cyan if clip.kind == "video" else magenta
            self.create_rectangle(x1, 34, x2, 65, fill=color, outline="")
            self.create_text(x1 + 5, 49, text=clip.name, anchor="w", fill="#071113", width=max(1, int(x2 - x1 - 8)), font=("Segoe UI", 9))
            cursor += clip.duration
        for index, effect in enumerate(self.project.effects):
            x1 = self._time_to_x(effect.start) + 1
            x2 = self._time_to_x(min(self.project.duration, effect.end)) - 1
            color = lime if index % 2 == 0 else amber
            self.create_rectangle(x1, 79, x2, 110, fill=color, outline="")
            self.create_text(x1 + 4, 94, text=effect.effect, anchor="w", fill="#11160b", width=max(1, int(x2 - x1 - 6)), font=("Segoe UI", 8))
        if self.project.audio_path:
            audio_start = max(0.0, self.project.audio_timeline_start)
            available = max(0.0, self.project.audio_duration - self.project.audio_source_start)
            visible_duration = min(max(0.0, self.project.duration - audio_start), available or self.project.duration)
            if visible_duration > 0:
                x1 = self._time_to_x(audio_start) + 1
                x2 = self._time_to_x(min(self.project.duration, audio_start + visible_duration)) - 1
                self.create_rectangle(x1, 124, x2, 155, fill="#a63d82", outline="")
                name = self.project.audio_path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
                label = f"{name} · in {self.project.audio_source_start:.1f}s"
                self.create_text(x1 + 5, 139, text=label, anchor="w", fill="#ffffff", width=max(1, int(x2 - x1 - 8)), font=("Segoe UI", 9))
        x = self._time_to_x(min(self.position, self.project.duration))
        self.create_line(x, 20, x, 163, fill="#ffffff", width=2)
        self.create_polygon(x - 5, 20, x + 5, 20, x, 27, fill="#ffffff", outline="")

    @staticmethod
    def _format_time(seconds: float) -> str:
        minutes = int(seconds // 60)
        return f"{minutes:02d}:{seconds - minutes * 60:04.1f}"
