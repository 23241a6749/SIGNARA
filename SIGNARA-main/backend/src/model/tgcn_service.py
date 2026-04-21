import torch
import torch.nn as nn
import numpy as np
from huggingface_hub import hf_hub_download
import os
import json
from typing import Optional, Tuple, List


class Config:
    """Configuration for TGCN model"""

    def __init__(self, config_path: str = None):
        self.num_samples = 50  # Number of frames
        self.hidden_size = 256  # For asl300
        self.drop_p = 0.3
        self.num_stages = 24
        self.num_class = 300

        if config_path and os.path.exists(config_path):
            self._load_from_file(config_path)

    def _load_from_file(self, config_path: str):
        """Load config from ini file"""
        with open(config_path, "r") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=")
                    if key == "num_samples":
                        self.num_samples = int(value)
                    elif key == "hidden_size":
                        self.hidden_size = int(value)
                    elif key == "drop_p":
                        self.drop_p = float(value)
                    elif key == "num_stages":
                        self.num_stages = int(value)
                    elif key == "num_class":
                        self.num_class = int(value)


class SpatialGraphConv(nn.Module):
    """Spatial Graph Convolutional Layer"""

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size, 1)
        self.kernel_size = kernel_size

    def forward(self, x, adj):
        x = self.conv(x)
        x = x.view(x.size(0), self.kernel_size, -1, x.size(2))
        return torch.einsum("nkct,kct->nct", x, adj)


class TGCNModel(nn.Module):
    """Temporal Graph Convolutional Network"""

    def __init__(self, input_feature, hidden_feature, num_class, p_dropout, num_stage):
        super().__init__()

        self.gc1 = SpatialGraphConv(input_feature, hidden_feature, 3)
        self.gc2 = SpatialGraphConv(hidden_feature, hidden_feature, 3)
        self.gc3 = SpatialGraphConv(hidden_feature, hidden_feature, 3)

        self.bn1 = nn.BatchNorm2d(hidden_feature)
        self.bn2 = nn.BatchNorm2d(hidden_feature)
        self.bn3 = nn.BatchNorm2d(hidden_feature)

        self.dropout = nn.Dropout(p_dropout)
        self.relu = nn.ReLU(inplace=True)

        self.fc = nn.Linear(hidden_feature, num_class)

    def forward(self, x, adj):
        # x: (batch, nodes, features) -> (batch, features, nodes)
        x = x.permute(0, 2, 1).unsqueeze(-1)

        x = self.gc1(x, adj)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.gc2(x, adj)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.gc3(x, adj)
        x = self.bn3(x)
        x = self.relu(x)

        x = x.squeeze(-1).permute(0, 2, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class TGCNService:
    """Service for TGCN model inference"""

    _instance = None
    _model = None
    _config = None
    _labels = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            self._load_model()

    def _load_model(self):
        """Load TGCN model from HuggingFace"""
        print("Loading TGCN-WLASL model...")

        try:
            # Try to download from HuggingFace
            checkpoint_path = hf_hub_download(
                repo_id="sharonn18/tgcn-wlasl",
                filename="checkpoints/asl300/pytorch_model.bin",
            )

            config_path = hf_hub_download(
                repo_id="sharonn18/tgcn-wlasl", filename="checkpoints/asl300/config.ini"
            )

            self._config = Config(config_path)

            # Create model
            input_feature = self._config.num_samples * 2
            self._model = TGCNModel(
                input_feature=input_feature,
                hidden_feature=self._config.hidden_size,
                num_class=self._config.num_class,
                p_dropout=self._config.drop_p,
                num_stage=self._config.num_stages,
            )

            # Load weights
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            self._model.load_state_dict(state_dict, strict=False)
            self._model.eval()

            # Load labels
            self._load_labels()

            print(f"Model loaded successfully! Classes: {self._config.num_class}")

        except Exception as e:
            print(f"Error loading model from HuggingFace: {e}")
            print("Using fallback simple model...")
            self._create_fallback_model()

    def _load_labels(self):
        """Load class labels"""
        try:
            labels_path = hf_hub_download(
                repo_id="sharonn18/tgcn-wlasl", filename="checkpoints/asl300/label.txt"
            )
            with open(labels_path, "r") as f:
                self._labels = [line.strip() for line in f.readlines()]
        except:
            # Generate placeholder labels
            self._labels = [f"word_{i}" for i in range(self._config.num_class)]

    def _create_fallback_model(self):
        """Create a simple fallback model for testing"""
        self._config = Config()
        input_feature = self._config.num_samples * 2

        self._model = nn.Sequential(
            nn.Linear(input_feature, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self._config.num_class),
        )
        self._model.eval()

        # Simple placeholder labels (common ASL words)
        common_words = [
            "HELLO",
            "THANK",
            "YOU",
            "YES",
            "NO",
            "PLEASE",
            "SORRY",
            "GOOD",
            "MORNING",
            "NIGHT",
            "WATER",
            "FOOD",
            "HELP",
            "HOME",
            "WORK",
            "FRIEND",
            "FAMILY",
            "LOVE",
            "HAPPY",
            "SAD",
            "ANGRY",
            "TIRED",
            "HUNGRY",
            "THIRSTY",
            "GO",
            "COME",
            "STOP",
            "WAIT",
            "LOOK",
            "SEE",
            "HEAR",
            "UNDERSTAND",
            "KNOW",
            "THINK",
            "WANT",
            "NEED",
            "LIKE",
            "DISLIKE",
            "BIG",
            "SMALL",
            "GOOD",
            "BAD",
            "HOT",
            "COLD",
            "NEW",
            "OLD",
            "YOUNG",
            "MAN",
            "WOMAN",
            "BOY",
            "GIRL",
            "CHILD",
        ]

        # Expand to 300
        self._labels = common_words * 7 + common_words[:12]
        self._labels = self._labels[: self._config.num_class]

        print(f"Fallback model created with {len(self._labels)} labels")

    def predict(
        self, keypoints: np.ndarray
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Predict sign from keypoints

        Args:
            keypoints: numpy array of shape (55, 100) - 55 keypoints, 50 frames x 2 (x,y)

        Returns:
            predicted_word, confidence, top5_predictions
        """
        if keypoints.shape != (55, self._config.num_samples * 2):
            # Resize if needed
            keypoints = self._resize_keypoints(keypoints)

        # Convert to tensor
        x = torch.FloatTensor(keypoints).unsqueeze(0)

        # Create adjacency matrix (simplified)
        adj = torch.ones(3, 55, 55)  # Simplified adjacency

        # Predict
        with torch.no_grad():
            output = self._model(x, adj)
            probs = torch.softmax(output, dim=1)

        # Get top 5
        top5_probs, top5_idx = torch.topk(probs, min(5, len(self._labels)), k=5)

        predicted_idx = top5_idx[0][0].item()
        confidence = top5_probs[0][0].item()

        predicted_word = (
            self._labels[predicted_idx]
            if predicted_idx < len(self._labels)
            else f"word_{predicted_idx}"
        )

        top5 = [
            (
                self._labels[idx.item()]
                if idx.item() < len(self._labels)
                else f"word_{idx.item()}",
                prob.item(),
            )
            for idx, prob in zip(top5_idx[0], top5_probs[0])
        ]

        return predicted_word, confidence, top5

    def _resize_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """Resize keypoints to expected shape"""
        # Simple interpolation
        target_len = self._config.num_samples * 2
        current_len = keypoints.shape[1]

        if current_len == target_len:
            return keypoints

        # Flatten and interpolate
        result = np.zeros((55, target_len))
        for i in range(55):
            result[i] = np.interp(
                np.linspace(0, 1, target_len),
                np.linspace(0, 1, current_len),
                keypoints[i],
            )
        return result

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def num_classes(self) -> int:
        return self._config.num_class if self._config else 0


def get_model_service() -> TGCNService:
    """Get singleton model service"""
    return TGCNService()
