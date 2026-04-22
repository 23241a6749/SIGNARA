from __future__ import annotations

from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple
import json

import joblib
import numpy as np

try:
    import torch
    from src.model.pose_transformer import PoseTransformerClassifier
except Exception:
    torch = None
    PoseTransformerClassifier = None


class WlaslSequenceService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        self._initialized = True
        self._model = None
        self._backend_type = "none"
        self._labels: List[str] = []
        self.sequence_length = 32
        self.feature_size = 126
        self._buffers: Dict[str, Deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=self.sequence_length)
        )
        self._recent_labels: Dict[str, Deque[str]] = defaultdict(lambda: deque(maxlen=5))
        self._ema_probs: Dict[str, np.ndarray] = {}
        self._alpha = 0.45
        self._min_confidence = 0.55
        self._min_margin = 0.12
        self._class_thresholds: Dict[str, float] = {}
        self._vote_window = 5
        self._vote_min_count = 3

        self._load_artifacts()
        self._load_runtime_policy()

    def _load_artifacts(self) -> None:
        repo_dir = Path(__file__).resolve().parents[3]
        backend_dir = Path(__file__).resolve().parents[2]
        root_models_dir = repo_dir / "models" / "wlasl_v1"
        backend_models_dir = backend_dir / "models" / "wlasl_v1"

        if (backend_models_dir / "transformer_model.pt").exists() or (
            backend_models_dir / "model.joblib"
        ).exists():
            models_dir = backend_models_dir
        else:
            models_dir = root_models_dir

        transformer_path = models_dir / "transformer_model.pt"
        artifact_path = models_dir / "model.joblib"

        if transformer_path.exists() and torch is not None and PoseTransformerClassifier is not None:
            try:
                checkpoint = torch.load(transformer_path, map_location="cpu")
                labels = [str(label).upper() for label in checkpoint["labels"]]
                self.sequence_length = int(checkpoint.get("sequence_length", 32))
                self.feature_size = int(checkpoint.get("feature_size", 126))
                hidden_dim = int(checkpoint.get("hidden_dim", 256))
                num_heads = int(checkpoint.get("num_heads", 4))
                num_layers = int(checkpoint.get("num_layers", 3))
                dropout = float(checkpoint.get("dropout", 0.2))

                model = PoseTransformerClassifier(
                    input_dim=self.feature_size,
                    num_classes=len(labels),
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    dropout=dropout,
                )
                model.load_state_dict(checkpoint["state_dict"])
                model.eval()

                self._model = model
                self._labels = labels
                self._backend_type = "pose_transformer"
                self._buffers = defaultdict(lambda: deque(maxlen=self.sequence_length))
                self._recent_labels = defaultdict(lambda: deque(maxlen=5))
                print(
                    "WLASL transformer loaded: "
                    f"classes={len(self._labels)}, seq={self.sequence_length}"
                )
                return
            except Exception as exc:
                print(f"Failed to load WLASL transformer model: {exc}")

        if not artifact_path.exists():
            print(f"WLASL model artifact not found: {artifact_path}")
            return

        try:
            artifact = joblib.load(artifact_path)
            self._model = artifact["model"]
            self._labels = [str(label).upper() for label in artifact["labels"]]
            self.sequence_length = int(artifact.get("sequence_length", 32))
            self.feature_size = int(artifact.get("feature_size", 126))
            self._backend_type = "sklearn_trees"
            self._buffers = defaultdict(lambda: deque(maxlen=self.sequence_length))
            self._recent_labels = defaultdict(lambda: deque(maxlen=5))
            print(
                f"WLASL sequence model loaded: classes={len(self._labels)}, seq={self.sequence_length}"
            )
        except Exception as exc:
            print(f"Failed to load WLASL sequence model: {exc}")
            self._model = None
            self._backend_type = "none"

    def _load_runtime_policy(self) -> None:
        repo_dir = Path(__file__).resolve().parents[3]
        backend_dir = Path(__file__).resolve().parents[2]
        root_policy = repo_dir / "models" / "wlasl_v1" / "runtime_policy.json"
        backend_policy = backend_dir / "models" / "wlasl_v1" / "runtime_policy.json"
        policy_path = backend_policy if backend_policy.exists() else root_policy
        if not policy_path.exists():
            return

        try:
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
            self._alpha = float(policy.get("ema_alpha", self._alpha))
            self._min_confidence = float(
                policy.get("default_min_confidence", self._min_confidence)
            )
            self._min_margin = float(policy.get("min_margin", self._min_margin))
            self._vote_window = int(policy.get("vote_window", self._vote_window))
            self._vote_min_count = int(
                policy.get("vote_min_count", self._vote_min_count)
            )

            raw_thresholds = policy.get("class_thresholds", {})
            self._class_thresholds = {
                str(label).upper(): float(value)
                for label, value in raw_thresholds.items()
                if str(label).upper() in set(self._labels)
            }

            self._recent_labels = defaultdict(lambda: deque(maxlen=self._vote_window))
            print(
                "WLASL runtime policy loaded: "
                f"thresholds={len(self._class_thresholds)}, "
                f"alpha={self._alpha}, margin={self._min_margin}"
            )
        except Exception as exc:
            print(f"Failed to load runtime policy: {exc}")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def labels(self) -> List[str]:
        return self._labels

    @property
    def backend_type(self) -> str:
        return self._backend_type

    @property
    def policy(self) -> Dict[str, float]:
        return {
            "ema_alpha": self._alpha,
            "default_min_confidence": self._min_confidence,
            "min_margin": self._min_margin,
            "vote_window": self._vote_window,
            "vote_min_count": self._vote_min_count,
            "class_threshold_count": len(self._class_thresholds),
        }

    def _predict_probabilities(self, sequence: np.ndarray) -> np.ndarray:
        if self._backend_type == "pose_transformer":
            with torch.no_grad():
                tensor = torch.from_numpy(sequence.astype(np.float32)).unsqueeze(0)
                logits = self._model(tensor)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            return probs.astype(np.float32)

        return self._model.predict_proba(sequence.reshape(1, -1))[0].astype(np.float32)

    def _predict_with_hand_swap(self, sequence: np.ndarray) -> np.ndarray:
        """
        Run inference on original + left/right-swapped hands and keep
        the distribution with stronger top-1 confidence.
        """
        probs_main = self._predict_probabilities(sequence)

        if self.feature_size < 126:
            return probs_main

        swapped = sequence.copy()
        swapped[:, :63], swapped[:, 63:126] = sequence[:, 63:126], sequence[:, :63]
        probs_swapped = self._predict_probabilities(swapped)

        if float(np.max(probs_swapped)) > float(np.max(probs_main)):
            return probs_swapped
        return probs_main

    def _apply_rejection(self, probs: np.ndarray) -> Tuple[str, float, List[Tuple[str, float]]]:
        sorted_idx = np.argsort(probs)[-5:][::-1]
        top5 = [(self._labels[idx], float(probs[idx])) for idx in sorted_idx]
        best_idx = int(sorted_idx[0])
        best_score = float(probs[best_idx])
        second_score = float(probs[sorted_idx[1]]) if len(sorted_idx) > 1 else 0.0
        best_label = self._labels[best_idx]
        class_threshold = self._class_thresholds.get(best_label, self._min_confidence)

        if best_score < class_threshold:
            return "UNKNOWN", best_score, top5
        if (best_score - second_score) < self._min_margin:
            return "UNKNOWN", best_score, top5
        return best_label, best_score, top5

    def _majority_vote(self, stream_id: str, candidate_label: str) -> str:
        history = self._recent_labels[stream_id]
        history.append(candidate_label)
        counts = Counter(history)
        best_label, best_count = counts.most_common(1)[0]
        if best_count < self._vote_min_count and len(history) >= self._vote_min_count:
            return history[-1]
        return best_label

    def _normalize_frame(self, keypoints: np.ndarray) -> np.ndarray:
        vec = keypoints.astype(np.float32).flatten()
        if len(vec) < self.feature_size:
            vec = np.pad(vec, (0, self.feature_size - len(vec)))
        elif len(vec) > self.feature_size:
            vec = vec[: self.feature_size]
        return vec

    def predict_sequence(
        self, sequence: np.ndarray
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        if not self.is_loaded:
            return "MODEL_NOT_LOADED", 0.0, []

        seq = np.asarray(sequence, dtype=np.float32)
        if seq.ndim != 2:
            raise ValueError("Sequence must be 2D [time, features]")

        if seq.shape[0] < self.sequence_length:
            pad_count = self.sequence_length - seq.shape[0]
            pad = np.repeat(seq[:1], repeats=pad_count, axis=0)
            seq = np.concatenate([pad, seq], axis=0)
        elif seq.shape[0] > self.sequence_length:
            seq = seq[-self.sequence_length :]

        if seq.shape[1] != self.feature_size:
            fixed = np.zeros((seq.shape[0], self.feature_size), dtype=np.float32)
            width = min(self.feature_size, seq.shape[1])
            fixed[:, :width] = seq[:, :width]
            seq = fixed

        logits = self._predict_with_hand_swap(seq)
        return self._apply_rejection(logits)

    def predict_from_frame(
        self, stream_id: str, frame_keypoints: np.ndarray
    ) -> Tuple[str, float, List[Tuple[str, float]], bool]:
        """
        Returns: gloss, confidence, top5, is_buffering
        """
        if not self.is_loaded:
            return "MODEL_NOT_LOADED", 0.0, [], False

        normalized = self._normalize_frame(frame_keypoints)
        buffer = self._buffers[stream_id]
        buffer.append(normalized)

        if len(buffer) < self.sequence_length:
            return "BUFFERING", 0.0, [], True

        sequence = np.stack(list(buffer), axis=0)
        logits = self._predict_with_hand_swap(sequence)

        previous = self._ema_probs.get(stream_id)
        if previous is None or len(previous) != len(logits):
            smoothed = logits
        else:
            smoothed = self._alpha * logits + (1.0 - self._alpha) * previous

        self._ema_probs[stream_id] = smoothed
        candidate_label, _, smoothed_top5 = self._apply_rejection(smoothed)
        voted_label = self._majority_vote(stream_id, candidate_label)

        if voted_label == "UNKNOWN":
            return voted_label, 0.0, smoothed_top5, False

        label_idx = self._labels.index(voted_label)
        return voted_label, float(smoothed[label_idx]), smoothed_top5, False


def get_wlasl_sequence_service() -> WlaslSequenceService:
    return WlaslSequenceService()
