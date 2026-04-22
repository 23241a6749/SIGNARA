# Signara - AI-Powered Bi-Directional Sign Language Interpreter

An AI-powered sign language interpreter designed for inclusive education. Signara enables bi-directional communication between deaf/hearing-impaired individuals and others through:
- **Sign → Text**: Translates sign language gestures to text
- **Text → Sign**: Displays 3D avatar animations for text input

## Features

- Real-time sign language detection using keypoint-based ML model
- WebSocket support for live streaming
- 3D avatar animation for text-to-sign conversion
- 291 word vocabulary (ASL-based)
- Clean web interface with webcam, chat, and avatar panels
- CPU-friendly (works without GPU)

## Quick Start

### 1. Start Backend

```bash
cd SIGNARA-main/backend
python3 -m virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 2. Open Frontend

```bash
cd SIGNARA-main/frontend
npm install
cp .env.example .env.local
npm run dev
```

Then open `http://localhost:3000` in Chrome.

### 3. Usage

1. Click **Start** to begin sign detection
2. Make signs in front of your webcam
3. Type in the chat box to see avatar animation
4. The system will display detected signs in real-time

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/predict` | POST | Convert text to sign sequence |
| `/predict-keypoints` | POST | Predict sign from keypoints |
| `/convert-text-to-gloss` | POST | Convert text to gloss sequence |
| `/ws/stream/{session_id}` | WebSocket | Real-time sign detection |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    SIGNARA ARCHITECTURE                 │
├─────────────────────────────────────────────────────────┤
│  Input: Webcam → Keypoint Extraction → ML Model        │
│  Output: Text / Avatar Animation                        │
├─────────────────────────────────────────────────────────┤
│  Backend (Python/FastAPI):                              │
│  - src/api/main.py - FastAPI server                     │
│  - src/model/sign_model.py - ML model (291 classes)     │
│  - src/motion_capture/ - Hand keypoint detection        │
├─────────────────────────────────────────────────────────┤
│  Frontend (HTML/JS):                                     │
│  - Webcam capture                                        │
│  - Canvas avatar animation                              │
│  - WebSocket client                                      │
└─────────────────────────────────────────────────────────┘
```

## Tech Stack

- **Backend**: Python, FastAPI, scikit-learn, NumPy
- **Frontend**: HTML5, JavaScript, TailwindCSS, Canvas API
- **ML Model**: RandomForest classifier with 291 ASL word classes

## Requirements

- Python 3.12+
- Node.js 18+ (for frontend development only)
- Webcam
- 8GB RAM recommended

## Known Limitations

- MediaPipe requires GPU libraries (libGLES) - using fallback detector
- Demo model trained with random data (needs real training data)
- For production: train model with actual ASL sign data

## WLASL Sign Detection (No Personal Recording)

The repository now includes a WLASL-only training pipeline under `backend/training/`.

1. Build a filtered manifest from WLASL metadata
2. Extract MediaPipe hand landmark sequences from selected words
3. Train a low-latency sequence classifier and save artifacts to `backend/models/wlasl_v1/`

Runtime behavior:

- If `backend/models/wlasl_v1/model.joblib` exists, `/predict-keypoints` and `/ws/stream/{session_id}` use the sequence model with temporal smoothing.
- If `backend/models/wlasl_v1/transformer_model.pt` exists, runtime prefers this transformer model.
- If `backend/models/wlasl_v1/runtime_policy.json` exists, runtime loads per-class confidence thresholds and rejection margin automatically.
- The runtime applies confidence and margin rejection and returns `UNKNOWN` when uncertain.
- If no artifact exists, backend falls back to the legacy keypoint model.

## License

For educational purposes.
