from typing import List, Tuple

import cv2
import mediapipe as mp


def extract_face_hand_landmarks(video_path: str) -> Tuple[List, List]:
    mp_face = mp.solutions.face_mesh.FaceMesh()
    mp_hands = mp.solutions.hands.Hands()

    face_landmarks = []
    hand_landmarks = []

    cap = cv2.VideoCapture(video_path)
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

    cap.release()
    return face_landmarks, hand_landmarks
