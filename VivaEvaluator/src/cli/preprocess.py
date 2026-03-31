import argparse
from pathlib import Path

from src.application.paths import features_dir, raw_dir
from src.application.preprocess_service import preprocess_sample, save_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="")
    parser.add_argument("--audio", type=str, default="")
    parser.add_argument("--out", type=str, default="sample_features.json")
    args = parser.parse_args()

    video_path = args.video or str(next(raw_dir().glob("*.mp4"), ""))
    audio_path = args.audio or str(next(raw_dir().glob("*.wav"), ""))

    if not video_path or not audio_path:
        raise FileNotFoundError("No video or audio found in data/raw")

    output_path = str(Path(features_dir()) / args.out)
    features = preprocess_sample(video_path, audio_path)
    save_features(features, output_path)
    print(f"Saved features to {output_path}")


if __name__ == "__main__":
    main()
