from datetime import datetime
from pathlib import Path
from typing import Tuple

import cv2
import sounddevice as sd
import soundfile as sf


def record_audio_video(
    duration_s: int,
    fps: int,
    width: int,
    height: int,
    out_dir: Path,
) -> Tuple[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = out_dir / f"video_{timestamp}.mp4"
    audio_filename = out_dir / f"audio_{timestamp}.wav"

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_filename), fourcc, fps, (width, height))

    print("Recording audio, then video...")

    audio_data = sd.rec(int(duration_s * 44100), samplerate=44100, channels=1)
    sd.wait()

    frame_count = 0
    while frame_count < duration_s * fps:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    sf.write(str(audio_filename), audio_data, 44100)

    return str(video_filename), str(audio_filename)
