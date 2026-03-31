import argparse

from src.application.paths import raw_dir
from src.infrastructure.capture_io import record_audio_video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=10)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    video_path, audio_path = record_audio_video(
        duration_s=args.duration,
        fps=args.fps,
        width=args.width,
        height=args.height,
        out_dir=raw_dir(),
    )
    print(f"Saved {video_path} and {audio_path}")


if __name__ == "__main__":
    main()
