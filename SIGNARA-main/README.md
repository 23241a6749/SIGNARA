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
cd signara/backend
source venv/bin/activate
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8001
```

### 2. Open Frontend

Open `signara/index.html` in a web browser (Chrome recommended).

### 3. Usage

1. Click **Start** to begin sign detection
2. Make signs in front of your webcam
3. Type in the chat box to see avatar animation
4. The system will display detected signs in real-time

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
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

## License

For educational purposes.
