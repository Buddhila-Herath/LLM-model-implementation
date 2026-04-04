from typing import Any, Dict, List, Tuple

import cv2
import mediapipe as mp


def extract_vision_features(video_path: str) -> Dict[str, Any]:
    """Face Mesh, Hands, Pose, and DeepFace emotion (every 10th frame)."""
    from deepface import DeepFace

    mp_face = mp.solutions.face_mesh.FaceMesh()
    mp_hands = mp.solutions.hands.Hands()
    mp_pose = mp.solutions.pose.Pose()

    face_landmarks: List = []
    hand_landmarks: List = []
    pose_features: List = []
    emotion_results: List = []
    frame_count = 0

    cap = cv2.VideoCapture(video_path)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_results = mp_face.process(rgb)
            if face_results.multi_face_landmarks:
                face_lm = face_results.multi_face_landmarks[0]
                coords = [[lm.x, lm.y, lm.z] for lm in face_lm.landmark]
                face_landmarks.append(coords)

            hand_results = mp_hands.process(rgb)
            if hand_results.multi_hand_landmarks:
                for hand in hand_results.multi_hand_landmarks:
                    coords = [[lm.x, lm.y, lm.z] for lm in hand.landmark]
                    hand_landmarks.append(coords)

            pose_res = mp_pose.process(rgb)
            if pose_res.pose_landmarks:
                coords = [
                    [lm.x, lm.y, lm.z]
                    for lm in pose_res.pose_landmarks.landmark
                ]
                pose_features.append(coords)

            if frame_count % 10 == 0:
                try:
                    result = DeepFace.analyze(
                        frame,
                        actions=["emotion"],
                        enforce_detection=False,
                    )
                    if isinstance(result, list) and result:
                        emotion_results.append(result[0].get("dominant_emotion", "unknown"))
                    elif isinstance(result, dict):
                        emotion_results.append(result.get("dominant_emotion", "unknown"))
                except Exception:
                    pass

            frame_count += 1
    finally:
        cap.release()
        mp_face.close()
        mp_hands.close()
        mp_pose.close()

    return {
        "face_landmarks": face_landmarks,
        "hand_landmarks": hand_landmarks,
        "pose_features": pose_features,
        "emotion_results": emotion_results,
    }


def extract_face_hand_landmarks(video_path: str) -> Tuple[List, List]:
    """Backward-compatible: returns only face and hand landmark lists."""
    v = extract_vision_features(video_path)
    return v["face_landmarks"], v["hand_landmarks"]
