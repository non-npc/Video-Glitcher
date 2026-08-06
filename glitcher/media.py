from __future__ import annotations

import json
from pathlib import Path
import subprocess

import cv2


def probe_media(path: str | Path) -> dict[str, object]:
    """Return duration and stream information using ffprobe, with OpenCV fallback."""
    source = str(Path(path).resolve())
    command = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration:stream=codec_type,width,height,r_frame_rate",
        "-of", "json", source,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=20)
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
        has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
        duration = float(data.get("format", {}).get("duration") or 0.0)
        return {
            "duration": duration,
            "has_video": video is not None,
            "has_audio": has_audio,
            "width": int(video.get("width", 0)) if video else 0,
            "height": int(video.get("height", 0)) if video else 0,
        }
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError):
        capture = cv2.VideoCapture(source)
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        info = {
            "duration": frames / fps if fps else 0.0,
            "has_video": capture.isOpened(),
            "has_audio": False,
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        }
        capture.release()
        return info

