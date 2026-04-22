from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path


USE_SOLUTIONS_API = False
_mp_hands_module = None
_mp_tasks_python = None
_mp_tasks_vision = None

try:
    _mp_hands_module = mp.solutions.hands
    USE_SOLUTIONS_API = True
except AttributeError:
    from mediapipe.tasks import python as _mp_tasks_python
    from mediapipe.tasks.python import vision as _mp_tasks_vision


class LiveHandLandmarkExtractor:
    def __init__(self):
        if USE_SOLUTIONS_API:
            self._hands = _mp_hands_module.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._use_tasks = False
        else:
            model_path = (
                Path(__file__).resolve().parents[2] / "mediapipe/models/hand_landmarker.task"
            )
            options = _mp_tasks_vision.HandLandmarkerOptions(
                base_options=_mp_tasks_python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=_mp_tasks_vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._hands = _mp_tasks_vision.HandLandmarker.create_from_options(options)
            self._use_tasks = True

    def close(self) -> None:
        self._hands.close()

    @staticmethod
    def _normalize_hand(hand_vec: np.ndarray) -> np.ndarray:
        out = hand_vec.copy()
        if not np.any(out):
            return out

        coords = out.reshape(21, 3)
        wrist_xy = coords[0, :2].copy()
        coords[:, :2] -= wrist_xy
        scale = np.max(np.linalg.norm(coords[:, :2], axis=1))
        if scale > 1e-6:
            coords[:, :2] /= scale
            coords[:, 2] /= scale
        return coords.reshape(-1)

    @classmethod
    def _normalize_vector(cls, vec: np.ndarray) -> np.ndarray:
        left = cls._normalize_hand(vec[:63])
        right = cls._normalize_hand(vec[63:])
        return np.concatenate([left, right]).astype(np.float32)

    def extract(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self._use_tasks:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = self._hands.detect(mp_image)
        else:
            results = self._hands.process(rgb)

        left = np.zeros(63, dtype=np.float32)
        right = np.zeros(63, dtype=np.float32)

        if hasattr(results, "multi_hand_landmarks"):
            if results.multi_hand_landmarks and results.multi_handedness:
                for landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness,
                ):
                    hand_name = handedness.classification[0].label.lower()
                    target = left if hand_name == "left" else right
                    for idx, lm in enumerate(landmarks.landmark):
                        offset = idx * 3
                        target[offset] = lm.x
                        target[offset + 1] = lm.y
                        target[offset + 2] = lm.z
            return self._normalize_vector(np.concatenate([left, right]))

        if getattr(results, "hand_landmarks", None):
            for landmarks, handedness in zip(results.hand_landmarks, results.handedness):
                hand_name = "right"
                if handedness:
                    hand_name = handedness[0].category_name.lower()

                target = left if hand_name == "left" else right
                for idx, lm in enumerate(landmarks):
                    offset = idx * 3
                    target[offset] = lm.x
                    target[offset + 1] = lm.y
                    target[offset + 2] = lm.z

        return self._normalize_vector(np.concatenate([left, right]))
