from src.application.paths import models_dir
from src.application.training_stub import train_dummy_model
from src.application.facial_expression_model import train_face_model
import torch


def main() -> None:
    model, loss = train_dummy_model()
    audio_output_path = models_dir() / "audio_model.pth"
    torch.save(model.state_dict(), audio_output_path)

    features_path = models_dir().parent / "features" / "sample_features.json"
    face_model, face_loss, face_source = train_face_model(str(features_path))
    face_output_path = models_dir() / "face_model.pth"
    torch.save(face_model.state_dict(), face_output_path)

    print(f"Trained dummy model. Loss: {loss:.4f}")
    print(f"Saved model to {audio_output_path}")
    print(f"Trained facial model ({face_source}). Loss: {face_loss:.4f}")
    print(f"Saved model to {face_output_path}")


if __name__ == "__main__":
    main()
