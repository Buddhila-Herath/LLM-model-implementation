from src.application.inference_stub import infer_probability
from src.application.paths import models_dir
from src.application.facial_expression_model import infer_face_probability


def main() -> None:
    audio_model_path = models_dir() / "audio_model.pth"
    audio_prob, _ = infer_probability(str(audio_model_path))

    face_model_path = models_dir() / "face_model.pth"
    features_path = models_dir().parent / "features" / "sample_features.json"
    face_prob, face_source = infer_face_probability(str(face_model_path), str(features_path))

    # --- Hand gesture inference ---
    from src.application.hand_gesture_model import infer_hand_probability
    hand_model_path = models_dir() / "hand_model.pth"
    hand_prob, hand_source = infer_hand_probability(str(hand_model_path), str(features_path))

    # Fused (simple average)
    fused_prob = (audio_prob + face_prob + hand_prob) / 3.0

    print(f"Audio confidence probability: {audio_prob:.2f}")
    print(f"Face confidence probability ({face_source}): {face_prob:.2f}")
    print(f"Hand gesture confidence probability ({hand_source}): {hand_prob:.2f}")
    print(f"Fused confidence probability: {fused_prob:.2f}")


if __name__ == "__main__":
    main()
