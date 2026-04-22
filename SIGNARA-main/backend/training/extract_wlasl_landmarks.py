import argparse
import csv
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


USE_SOLUTIONS_API = False
mp_hands = None
mp_tasks_python = None
mp_tasks_vision = None

try:
    mp_hands = mp.solutions.hands
    USE_SOLUTIONS_API = True
except AttributeError:
    from mediapipe.tasks import python as mp_tasks_python
    from mediapipe.tasks.python import vision as mp_tasks_vision


def _extract_two_hand_vector(results) -> np.ndarray:
    left = np.zeros(63, dtype=np.float32)
    right = np.zeros(63, dtype=np.float32)

    if hasattr(results, "multi_hand_landmarks"):
        if not results.multi_hand_landmarks or not results.multi_handedness:
            return np.concatenate([left, right])

        for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_name = handedness.classification[0].label.lower()
            target = left if hand_name == "left" else right
            for idx, lm in enumerate(landmarks.landmark):
                base = idx * 3
                target[base] = lm.x
                target[base + 1] = lm.y
                target[base + 2] = lm.z
        return np.concatenate([left, right])

    if not getattr(results, "hand_landmarks", None):
        return np.concatenate([left, right])

    for landmarks, handedness in zip(results.hand_landmarks, results.handedness):
        hand_name = "right"
        if handedness:
            hand_name = handedness[0].category_name.lower()

        target = left if hand_name == "left" else right
        for idx, lm in enumerate(landmarks):
            base = idx * 3
            target[base] = lm.x
            target[base + 1] = lm.y
            target[base + 2] = lm.z

    return np.concatenate([left, right])


def _normalize_hand(hand_vec: np.ndarray) -> np.ndarray:
    out = hand_vec.copy()
    if not np.any(out):
        return out

    coords = out.reshape(21, 3)
    wrist = coords[0, :2].copy()
    coords[:, :2] -= wrist
    scale = np.max(np.linalg.norm(coords[:, :2], axis=1))
    if scale > 1e-6:
        coords[:, :2] /= scale
        coords[:, 2] /= scale
    return coords.reshape(-1)


def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    left = _normalize_hand(vec[:63])
    right = _normalize_hand(vec[63:])
    return np.concatenate([left, right]).astype(np.float32)


def _sample_frame_indices(total_frames: int, sequence_length: int) -> np.ndarray:
    if total_frames <= 0:
        return np.zeros(sequence_length, dtype=np.int32)
    if total_frames == 1:
        return np.zeros(sequence_length, dtype=np.int32)
    return np.linspace(0, total_frames - 1, sequence_length).astype(np.int32)


def _extract_video_sequence(video_path: Path, hands, sequence_length: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = _sample_frame_indices(total_frames, sequence_length)
    needed = set(frame_indices.tolist())
    extracted: dict[int, np.ndarray] = {}

    frame_idx = 0
    while cap.isOpened() and needed:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in needed:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if USE_SOLUTIONS_API:
                results = hands.process(rgb)
            else:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                results = hands.detect(mp_image)
            vec = _extract_two_hand_vector(results)
            extracted[frame_idx] = _normalize_vector(vec)
            needed.remove(frame_idx)
        frame_idx += 1

    cap.release()

    if not extracted:
        return None

    sequence = []
    last_valid = np.zeros(126, dtype=np.float32)
    for idx in frame_indices:
        current = extracted.get(int(idx), last_valid)
        sequence.append(current)
        if np.any(current):
            last_valid = current

    return np.stack(sequence).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract WLASL hand landmark sequences")
    parser.add_argument("--manifest", required=True, help="CSV from wlasl_prepare_subset.py")
    parser.add_argument("--output", required=True, help="Output NPZ path")
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means use all")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_path = Path(args.output)

    records = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(row)

    if args.max_samples > 0:
        records = records[: args.max_samples]

    labels = sorted({record["gloss"] for record in records})
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    X = []
    y = []
    splits = []

    if USE_SOLUTIONS_API:
        hands_ctx = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        hands = hands_ctx
        close_hands = hands_ctx.close
    else:
        model_path = Path(__file__).resolve().parents[1] / "mediapipe/models/hand_landmarker.task"
        options = mp_tasks_vision.HandLandmarkerOptions(
            base_options=mp_tasks_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_tasks_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        hands = mp_tasks_vision.HandLandmarker.create_from_options(options)
        close_hands = hands.close

    try:
        for idx, record in enumerate(records, start=1):
            sequence = _extract_video_sequence(
                video_path=Path(record["video_path"]),
                hands=hands,
                sequence_length=args.sequence_length,
            )
            if sequence is None:
                continue

            X.append(sequence)
            y.append(label_to_index[record["gloss"]])
            splits.append(record.get("split", "train"))

            if idx % 100 == 0:
                print(f"Processed {idx}/{len(records)}")
    finally:
        close_hands()

    if not X:
        raise RuntimeError("No sequences extracted. Check manifest and video paths.")

    X_arr = np.stack(X).astype(np.float32)
    y_arr = np.array(y, dtype=np.int64)
    splits_arr = np.array(splits)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=X_arr,
        y=y_arr,
        splits=splits_arr,
        labels=np.array(labels),
    )

    metadata_path = output_path.with_suffix(".json")
    metadata = {
        "sequence_length": args.sequence_length,
        "num_samples": int(X_arr.shape[0]),
        "num_classes": len(labels),
        "feature_size": int(X_arr.shape[-1]),
        "labels": labels,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved dataset to {output_path}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
