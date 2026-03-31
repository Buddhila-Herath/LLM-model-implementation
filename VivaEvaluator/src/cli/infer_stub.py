from src.application.inference_stub import infer_probability
from src.application.paths import models_dir


def main() -> None:
    model_path = models_dir() / "audio_model.pth"
    prob, _ = infer_probability(str(model_path))
    print(f"Predicted confidence probability: {prob:.2f}")


if __name__ == "__main__":
    main()
