import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os
from typing import Tuple, List
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class SignModelService:
    """Lightweight sign language recognition using scikit-learn"""

    _instance = None
    _model = None
    _scaler = None
    _labels = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._model is None:
            self._load_or_create_model()
            self._load_tf_model()

    def _load_or_create_model(self):
        """Load pre-trained model or create a simple one"""
        print("Initializing Signara model...")

        # Common ASL words for 300-class model
        self._labels = [
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
            "TIME",
            "DAY",
            "WEEK",
            "MONTH",
            "YEAR",
            "NOW",
            "THEN",
            "TODAY",
            "TOMORROW",
            "YESTERDAY",
            "ALWAYS",
            "NEVER",
            "MAYBE",
            "QUITE",
            "VERY",
            "REALLY",
            "JUST",
            "STILL",
            "ALREADY",
            "YET",
            "AGAIN",
            "HERE",
            "THERE",
            "WHERE",
            "WHEN",
            "WHY",
            "HOW",
            "WHAT",
            "WHO",
            "WHICH",
            "THAT",
            "THIS",
            "THESE",
            "THOSE",
            "SOME",
            "ANY",
            "ALL",
            "EVERY",
            "MUCH",
            "MANY",
            "MORE",
            "LESS",
            "FEW",
            "LITTLE",
            "BOTH",
            "EACH",
            "OTHER",
            "ANOTHER",
            "SUCH",
            "ONLY",
            "EVEN",
            "ALSO",
            "TOO",
            "VERY",
            "SO",
            "NOW",
            "THEN",
            "THERE",
            "HERE",
            "AWAY",
            "BACK",
            "DOWN",
            "UP",
            "OUT",
            "IN",
            "ON",
            "OFF",
            "OVER",
            "UNDER",
            "WITH",
            "WITHOUT",
            "FROM",
            "INTO",
            "ABOUT",
            "AGAIN",
            "AWAY",
            "AFTER",
            "BEFORE",
            "BEHIND",
            "BETWEEN",
            "THROUGH",
            "DURING",
            "UNTIL",
            "WHILE",
            "DURING",
            "BESIDE",
            "BEYOND",
            "AMONG",
            "WITH",
            "AGAINST",
            "ALONG",
            "AROUND",
            "WITH",
            "WITHIN",
            "WITHOUT",
            "CAN",
            "COULD",
            "WILL",
            "WOULD",
            "SHALL",
            "SHOULD",
            "MAY",
            "MIGHT",
            "MUST",
            "HAVE",
            "HAS",
            "HAD",
            "DO",
            "DOES",
            "DID",
            "AM",
            "IS",
            "ARE",
            "WAS",
            "WERE",
            "BE",
            "BEING",
            "BEEN",
            "GET",
            "GOT",
            "GETTING",
            "GIVE",
            "GAVE",
            "GIVEN",
            "TAKE",
            "TOOK",
            "TAKEN",
            "MAKE",
            "MADE",
            "SAY",
            "SAID",
            "SAYING",
            "TELL",
            "TOLD",
            "THINK",
            "THOUGHT",
            "KNOW",
            "KNEW",
            "KNOWN",
            "SEE",
            "SAW",
            "SEEN",
            "WANT",
            "WANTED",
            "USE",
            "USED",
            "USING",
            "FIND",
            "FOUND",
            "FINDING",
            "GIVE",
            "GAVE",
            "TELL",
            "TOLD",
            "CALL",
            "CALLED",
            "TRY",
            "TRYING",
            "ASK",
            "ASKED",
            "WORK",
            "WORKING",
            "SEEM",
            "SEEMED",
            "FEEL",
            "FEELING",
            "BECOME",
            "LEAVE",
            "PUT",
            "KEEP",
            "KEPT",
            "BEGIN",
            "BEGAN",
            "SHOW",
            "SHOWED",
            "HEAR",
            "HEARD",
            "PLAY",
            "PLAYING",
            "RUN",
            "RUNNING",
            "MOVE",
            "LIVE",
            "LIVING",
            "BELIEVE",
            "BRING",
            "HAPPEN",
            "WRITE",
            "WROTE",
            "PROVIDE",
            "SIT",
            "STAND",
            "LOSE",
            "PAY",
            "MEET",
            "INCLUDE",
            "CONTINUE",
            "SET",
            "LEARN",
            "CHANGE",
            "LEAD",
            "UNDERSTAND",
            "WATCH",
            "FOLLOW",
            "STOP",
            "CREATE",
            "SPEAK",
            "READ",
            "ALLOW",
            "ADD",
            "SPEND",
            "GROW",
            "OPEN",
            "WALK",
            "WIN",
            "OFFER",
            "REMEMBER",
            "LOVE",
            "CONSIDER",
            "APPEAR",
            "BUY",
            "WAIT",
            "SERVE",
            "DIE",
            "SEND",
            "EXPECT",
            "BUILD",
            "STAY",
            "FALL",
            "CUT",
            "REACH",
            "KILL",
            "REMAIN",
            "PERFORM",
            "APPEAR",
            "INCLUDE",
            "CONTINUE",
            "RECORD",
            "DEFINE",
        ]

        # Limit to 300
        self._labels = self._labels[:300]

        # Initialize scaler
        self._scaler = StandardScaler()

        # Create a simple Random Forest model
        # Input: 55 keypoints x 2 (x,y) x 50 frames = 5500 features
        self._model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )

        # Generate some random training data for demo
        # In production, this would be real training data
        self._generate_demo_model()

        print(f"Model initialized with {len(self._labels)} classes")

    def _generate_demo_model(self):
        """Generate a demo model with random data"""
        n_samples = 1000
        n_features = 5500  # 55 keypoints * 100 values (50 frames * 2)

        # Generate random training data
        X = np.random.randn(n_samples, n_features) * 0.1
        y = np.random.randint(0, len(self._labels), n_samples)

        # Fit scaler
        self._scaler.fit(X)
        X_scaled = self._scaler.transform(X)

        # Train model
        self._model.fit(X_scaled, y)
        print("Demo model trained with random data")

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
        # Reshape keypoints to flat array
        if keypoints.ndim == 2:
            keypoints_flat = keypoints.flatten()
        else:
            keypoints_flat = keypoints

        # Ensure correct size
        if len(keypoints_flat) < 5500:
            keypoints_flat = np.pad(keypoints_flat, (0, 5500 - len(keypoints_flat)))
        elif len(keypoints_flat) > 5500:
            keypoints_flat = keypoints_flat[:5500]

        # Scale and predict
        keypoints_scaled = self._scaler.transform(keypoints_flat.reshape(1, -1))
        probabilities = self._model.predict_proba(keypoints_scaled)[0]

        # Get top 5
        top5_indices = np.argsort(probabilities)[-5:][::-1]
        top5_probs = probabilities[top5_indices]

        predicted_word = self._labels[top5_indices[0]]
        confidence = top5_probs[0]

        top5 = [
            (self._labels[idx], float(prob))
            for idx, prob in zip(top5_indices, top5_probs)
        ]

        return predicted_word, float(confidence), top5

    def _load_tf_model(self):
        print("Loading text-to-sign model...")
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            h5_path = os.path.join(base_dir, "sign_model.h5")
            csv_path = os.path.join(base_dir, "sign_dataset.csv")

            # Load TF model
            print(f"Loading weights from {h5_path}...")
            self.tf_model = load_model(h5_path)
            
            # Load dataset to recreate tokenizers
            print(f"Loading dataset from {csv_path}...")
            df = pd.read_csv(csv_path)
            
            self.tokenizer_in = Tokenizer()
            self.tokenizer_in.fit_on_texts(df['english'].astype(str))
            
            self.tokenizer_out = Tokenizer()
            self.tokenizer_out.fit_on_texts(df['sign'].astype(str))
            
            self.is_tf_loaded = True
            print("Text-to-sign model loaded successfully!")
        except Exception as e:
            print(f"Error loading text-to-sign model: {e}")
            self.tf_model = None
            self.is_tf_loaded = False

    def predict_sign(self, text: str) -> List[str]:
        if not getattr(self, "is_tf_loaded", False) or not self.tf_model:
            print("TF Model not loaded.")
            return []
            
        try:
            text = text.lower()
            seqs = self.tokenizer_in.texts_to_sequences([text])
            if not seqs or len(seqs[0]) == 0:
                print("No tokens found in input.")
                return []
                
            padded = pad_sequences(seqs, maxlen=10, padding='post')
            preds = self.tf_model.predict(padded, verbose=0)
            
            # Get max from prediction distribution over vocabulary size
            out_seq = np.argmax(preds, axis=-1)[0]
            
            # Build reverse word map mapping index -> actual sign word
            reverse_word_map = dict(map(reversed, self.tokenizer_out.word_index.items()))
            
            signs = []
            for word_id in out_seq:
                if word_id != 0:
                    word = reverse_word_map.get(word_id, '')
                    if word:
                        signs.append(word.upper())
            print(f"Predicted signs for '{text}': {signs}")
            return signs
        except Exception as e:
            print(f"Error in predict_sign: {e}")
            return []

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def num_classes(self) -> int:
        return len(self._labels)


def get_model_service() -> SignModelService:
    """Get singleton model service"""
    return SignModelService()
