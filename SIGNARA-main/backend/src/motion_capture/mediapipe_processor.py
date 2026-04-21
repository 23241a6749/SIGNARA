import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional, Dict
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os


class MediaPipeProcessor:
    """MediaPipe processor using the new Task API for hand landmark detection"""

    # Class-level model path (will download automatically)
    MODEL_PATH = None

    def __init__(self, model_path: str = "mediapipe/models/hand_landmarker.task"):
        # Create hand landmarker using downloaded model
        base_options = python.BaseOptions(model_asset_path=model_path)

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.detector = vision.HandLandmarker.create_from_options(options)

        # Buffer for temporal processing
        self.buffer_size = 50
        self.landmark_buffer = []

        # Landmark info
        self.num_hand_landmarks = 21

        # Drawing utils
        self.mp_drawing = mp.Image
        self._setup_drawing()

    def _get_hand_model_bytes(self):
        """Get hand landmarker model - use built-in or download"""
        # For the Task API, we need model files
        # Use a workaround - create model from bytes
        # MediaPipe will download model automatically on first use
        return None  # Will use default model

    def _setup_drawing(self):
        """Setup drawing utilities"""
        # Create blank module for drawing
        pass

    def process_frame(self, frame: np.ndarray) -> Dict:
        """Process a single frame and extract hand landmarks"""
        # Convert to RGB and create MediaPipe image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Process with MediaPipe
        results = self.detector.detect_for_video(mp_image, int(time.time() * 1000))

        # Extract keypoints
        keypoints = self._extract_keypoints(results)

        # Draw landmarks on frame
        annotated_frame = self._draw_landmarks(frame.copy(), results)

        # Check handedness
        has_left = False
        has_right = False
        if results.handedness:
            for hand_list in results.handedness:
                for hand in hand_list:
                    if hand.category_name == "Left":
                        has_left = True
                    elif hand.category_name == "Right":
                        has_right = True

        return {
            "keypoints": keypoints,
            "annotated_frame": annotated_frame,
            "has_left_hand": has_left,
            "has_right_hand": has_right,
            "has_any_hand": results.hand_landmarks is not None
            and len(results.hand_landmarks) > 0,
            "hand_landmarks": results.hand_landmarks,
            "hand_handedness": results.handedness,
        }

    def _extract_keypoints(self, results) -> np.ndarray:
        """Extract hand keypoints into flat array"""
        keypoints = []

        # Both hands: 21 points each * 3 (x, y, z) = 126
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                for landmark in hand_landmarks:
                    keypoints.extend([landmark.x, landmark.y, landmark.z])
        else:
            # No hands detected - zeros for 2 hands
            keypoints = [0.0] * (21 * 2 * 3)

        return np.array(keypoints, dtype=np.float32)

    def _draw_landmarks(self, frame: np.ndarray, results) -> np.ndarray:
        """Draw hand landmarks on frame"""
        if not results.hand_landmarks:
            return frame

        h, w = frame.shape[:2]

        # Define hand connections
        HAND_CONNECTIONS = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),  # Thumb
            (0, 5),
            (5, 6),
            (6, 7),
            (7, 8),  # Index finger
            (0, 9),
            (9, 10),
            (10, 11),
            (11, 12),  # Middle finger
            (0, 13),
            (13, 14),
            (14, 15),
            (15, 16),  # Ring finger
            (0, 17),
            (17, 18),
            (18, 19),
            (19, 20),  # Pinky
            (5, 9),
            (9, 13),
            (13, 17),  # Palm connections
        ]

        # Colors for each hand
        colors = [(0, 255, 0), (0, 0, 255)]

        for hand_idx, hand_landmarks in enumerate(results.hand_landmarks):
            color = colors[hand_idx % 2] if hand_idx < 2 else (255, 255, 0)

            # Draw connections
            for start_idx, end_idx in HAND_CONNECTIONS:
                if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                    x1 = int(hand_landmarks[start_idx].x * w)
                    y1 = int(hand_landmarks[start_idx].y * h)
                    x2 = int(hand_landmarks[end_idx].x * w)
                    y2 = int(hand_landmarks[end_idx].y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), color, 2)

            # Draw landmarks
            for landmark in hand_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 3, color, -1)

        return frame

    def get_hand_keypoints(self, results, hand_side: str = "Right") -> np.ndarray:
        """Extract keypoints for specific hand"""
        keypoints = []

        if results.hand_landmarks and results.handedness:
            for hand_landmarks, handedness_list in zip(
                results.hand_landmarks, results.handedness
            ):
                for handedness in handedness_list:
                    if handedness.category_name == hand_side:
                        for landmark in hand_landmarks:
                            keypoints.extend([landmark.x, landmark.y])
                        break

        # Pad to 21 * 2 = 42 values
        while len(keypoints) < 42:
            keypoints.extend([0.0, 0.0])

        return np.array(keypoints[:42], dtype=np.float32)

    def add_to_buffer(self, keypoints: np.ndarray):
        """Add keypoints to temporal buffer"""
        self.landmark_buffer.append(keypoints)

        if len(self.landmark_buffer) > self.buffer_size:
            self.landmark_buffer.pop(0)

    def get_buffer_as_array(self) -> np.ndarray:
        """Get buffer as array for model input"""
        if not self.landmark_buffer:
            return np.zeros((21, 100), dtype=np.float32)

        while len(self.landmark_buffer) < self.buffer_size:
            self.landmark_buffer.insert(0, self.landmark_buffer[0].copy())

        buffer_array = np.array(self.landmark_buffer[-self.buffer_size :])
        result = buffer_array.T.reshape(21, -1)

        return result

    def clear_buffer(self):
        """Clear the landmark buffer"""
        self.landmark_buffer = []

    def close(self):
        """Clean up MediaPipe resources"""
        self.detector.close()


class CameraProcessor:
    """Real-time camera processor for sign language recognition"""

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
        self.mediapipe = MediaPipeProcessor()
        self.is_running = False
        self.fps = 0
        self.frame_count = 0
        self.start_time = None

    def start(self):
        """Start camera capture"""
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.is_running = True
        self.start_time = time.time()
        self.frame_count = 0

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read next frame"""
        if self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        self.frame_count += 1

        if self.start_time and self.frame_count % 30 == 0:
            elapsed = time.time() - self.start_time
            self.fps = self.frame_count / elapsed

        return ret, frame

    def process_frame(self, frame: np.ndarray) -> Dict:
        """Process frame with MediaPipe"""
        return self.mediapipe.process_frame(frame)

    def stop(self):
        """Stop camera and release resources"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.mediapipe.close()

    def get_fps(self) -> float:
        """Get current FPS"""
        return self.fps


def test_camera():
    """Test camera and MediaPipe"""
    print("Starting camera test...")

    # Check for available camera
    test_cap = cv2.VideoCapture(0)
    if not test_cap.isOpened():
        print("No camera available - testing with dummy frame")
        test_cap.release()

        # Test with dummy frame
        processor = MediaPipeProcessor()
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        try:
            result = processor.process_frame(test_frame)
            print(f"Keypoints shape: {result['keypoints'].shape}")
            print("MediaPipe test with dummy frame PASSED!")
        except Exception as e:
            print(f"Error: {e}")

        processor.close()
        return

    test_cap.release()

    # Test with real camera
    processor = CameraProcessor()
    processor.start()

    print("Camera test - Press 'q' to quit")

    while processor.is_running:
        ret, frame = processor.read_frame()
        if not ret:
            print("Failed to read frame")
            break

        result = processor.process_frame(frame)

        fps_text = f"FPS: {processor.get_fps():.1f}"
        cv2.putText(
            result["annotated_frame"],
            fps_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        status = f"L: {result['has_left_hand']} | R: {result['has_right_hand']}"
        cv2.putText(
            result["annotated_frame"],
            status,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Signara Test", result["annotated_frame"])

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    processor.stop()
    cv2.destroyAllWindows()
    print("Camera test complete")


if __name__ == "__main__":
    test_camera()
