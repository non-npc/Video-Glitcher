# Video Glitcher v1

![Video Glitcher version](https://img.shields.io/badge/Video_Glitcher-v0.1.0-38e5df)
![Python version](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)

Video Glitcher is a Python desktop editor for combining silent video clips, a separate audio track, procedural retro visuals, and smooth seeded glitch events.

![Video Glitcher interface](screenshot.png)

## Included in v1

- Import and reorder multiple video clips.
- Preserve each source's full aspect ratio by default, with optional crop-to-fill or stretch sizing.
- Confine glitches and CRT processing to the visible source rectangle, leaving letterbox and pillarbox areas untouched.
- Use MP3, OGG, WAV, FLAC, AAC, M4A, or a video's audio stream as the soundtrack.
- Hear the soundtrack during editor playback, including after pausing or seeking.
- Set both the soundtrack's source in-point and its start position on the video timeline.
- Add infinite procedural Neon Grid, Synth Sun, Plasma Field, and Analog Static clips.
- Apply six animated retro effects with smooth attack and recovery envelopes.
- Add individual effects at the playhead or generate an entire seeded sequence.
- Continuously fill either the selected clip or every clip with randomized effects that stay inside clip boundaries.
- Replace or reroll individual generated events without changing their position or duration.
- Use fixed duration or a randomized duration range, density, intensity, and meld controls.
- Preview the actual rendering pipeline used by export.
- Save and reopen `.glitcher.json` projects.
- Export H.264 MP4 with AAC audio through FFmpeg.

## Requirements

- Python 3.11 or newer.
- Tkinter with Tcl/Tk 8.6 or newer.
- FFmpeg, FFprobe, and FFplay available on the system `PATH`.
- The Python packages listed in `requirements.txt`: OpenCV, NumPy, and Pillow.

## Run

Install the Python dependencies, then launch the application:

```powershell
python -m pip install -r requirements.txt
python run_glitcher.py
```

## First workflow

1. Import one or more video clips, or add a procedural generator clip.
2. Optionally choose a separate audio source.
3. Choose fixed or randomized timing and adjust intensity, density, meld, and seed.
4. Generate a full sequence, then preview or edit individual events.
5. Save the project and export an MP4.

## V1 boundaries

- Video clips are concatenated in the displayed order. Source sizing can fit with letterboxing/pillarboxing, crop to fill, or stretch.
- Audio timing is controlled by an audio source in-point and a separate video-timeline start value.
- Clip trimming, draggable events, undo/redo, and GPU shaders are planned beyond v1.
- Rendering is CPU-based; 1080p or 60 fps exports may take longer than real time.
