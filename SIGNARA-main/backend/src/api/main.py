from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import base64
import json
import numpy as np
import time
import cv2

app = FastAPI(
    title="Signara API",
    description="AI-Powered Bi-Directional Sign Language Interpreter",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model_service = None
sequence_service = None
active_connections = 0


class PredictionRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    signs: List[str]


class KeypointPredictionRequest(BaseModel):
    keypoints: List
    sequence_id: Optional[str] = None


class TopPrediction(BaseModel):
    gloss: str
    confidence: float = Field(ge=0.0, le=1.0)


class KeypointPredictionResponse(BaseModel):
    gloss: str
    confidence: float = Field(ge=0.0, le=1.0)
    top5: List[TopPrediction]
    timestamp: int
    latency_ms: float


class GlossInterpretationRequest(BaseModel):
    input: List[List[str]]


class GlossInterpretationResponse(BaseModel):
    sentence: str


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global model_service, sequence_service
    print("Initializing Signara API...")
    try:
        from src.model.sign_model import get_model_service
        from src.model.wlasl_sequence_service import get_wlasl_sequence_service

        model_service = get_model_service()
        sequence_service = get_wlasl_sequence_service()
        print(
            f"Model loaded: {model_service.is_loaded}, Classes: {model_service.num_classes}"
        )
        print(f"Sequence model loaded: {sequence_service.is_loaded}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model_service = None
        sequence_service = None


@app.get("/")
async def root():
    return {"message": "Signara API - AI-Powered Sign Language Interpreter"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_service.is_loaded if model_service else False,
        "text_model_loaded": getattr(model_service, "is_tf_loaded", False)
        if model_service
        else False,
        "gpu_available": False,
        "active_connections": active_connections,
        "num_classes": model_service.num_classes if model_service else 0,
        "sequence_model_loaded": sequence_service.is_loaded if sequence_service else False,
        "sequence_backend": sequence_service.backend_type if sequence_service else "none",
        "sequence_policy": sequence_service.policy if sequence_service else {},
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """REST endpoint for text-to-sign prediction"""
    if model_service is None:
        return PredictionResponse(signs=[])

    signs = model_service.predict_sign(request.text)
    if not signs:
        fallback = await convert_text_to_gloss({"text": request.text})
        signs = fallback.get("glosses", [])

    return PredictionResponse(signs=signs)


def _empty_keypoint_response(gloss: str, start_time: float) -> KeypointPredictionResponse:
    return KeypointPredictionResponse(
        gloss=gloss,
        confidence=0.0,
        top5=[],
        timestamp=int(time.time() * 1000),
        latency_ms=(time.time() - start_time) * 1000,
    )


@app.post("/predict-keypoints", response_model=KeypointPredictionResponse)
async def predict_keypoints(request: KeypointPredictionRequest):
    """Predict from pre-extracted keypoints"""
    start_time = time.time()

    keypoints = request.keypoints
    if not keypoints or len(keypoints) == 0:
        return _empty_keypoint_response("NO_KEYPOINTS", start_time)

    try:
        keypoints_array = np.array(keypoints)
    except Exception:
        return _empty_keypoint_response("INVALID_KEYPOINTS", start_time)

    if sequence_service and sequence_service.is_loaded:
        if keypoints_array.ndim == 2:
            word, confidence, top5 = sequence_service.predict_sequence(keypoints_array)
        else:
            word, confidence, top5, _ = sequence_service.predict_from_frame(
                stream_id=request.sequence_id or "rest-default",
                frame_keypoints=keypoints_array.flatten(),
            )
    elif model_service and model_service.is_loaded:
        if keypoints_array.ndim > 1:
            keypoints_array = keypoints_array.flatten()
        word, confidence, top5 = model_service.predict(keypoints_array)
    else:
        return _empty_keypoint_response("MODEL_NOT_LOADED", start_time)

    serialized_top5 = [
        TopPrediction(gloss=entry[0], confidence=float(entry[1])) for entry in top5
    ]

    return KeypointPredictionResponse(
        gloss=word,
        confidence=float(confidence),
        top5=serialized_top5,
        timestamp=int(time.time() * 1000),
        latency_ms=(time.time() - start_time) * 1000,
    )


@app.post("/interpret-glosses", response_model=GlossInterpretationResponse)
async def interpret_glosses(request: GlossInterpretationRequest):
    """Convert gloss sequences to natural language sentences"""
    glosses = []
    for chunk in request.input:
        if chunk:
            glosses.append(chunk[0])

    sentence = " ".join(glosses).lower()

    return GlossInterpretationResponse(sentence=sentence)


# Simple word-to-gloss mapping for text-to-sign
WORD_TO_GLOSS = {
    "hello": "HELLO",
    "hi": "HELLO",
    "hey": "HELLO",
    "thank": "THANK",
    "thanks": "THANK",
    "thankyou": "THANK",
    "you": "YOU",
    "u": "YOU",
    "yes": "YES",
    "yeah": "YES",
    "yep": "YES",
    "no": "NO",
    "nope": "NO",
    "please": "PLEASE",
    "sorry": "SORRY",
    "apologies": "SORRY",
    "good": "GOOD",
    "great": "GOOD",
    "nice": "GOOD",
    "bad": "BAD",
    "terrible": "BAD",
    "morning": "MORNING",
    "am": "MORNING",
    "night": "NIGHT",
    "evening": "NIGHT",
    "water": "WATER",
    "drink": "WATER",
    "food": "FOOD",
    "eat": "FOOD",
    "help": "HELP",
    "helpme": "HELP",
    "home": "HOME",
    "house": "HOME",
    "work": "WORK",
    "job": "WORK",
    "friend": "FRIEND",
    "buddy": "FRIEND",
    "family": "FAMILY",
    "relatives": "FAMILY",
    "love": "LOVE",
    "like": "LIKE",
    "happy": "HAPPY",
    "glad": "HAPPY",
    "sad": "SAD",
    "unhappy": "SAD",
    "go": "GO",
    "going": "GO",
    "come": "COME",
    "coming": "COME",
    "stop": "STOP",
    "wait": "WAIT",
    "see": "SEE",
    "watch": "SEE",
    "want": "WANT",
    "need": "NEED",
    "know": "KNOW",
    "understand": "UNDERSTAND",
}


@app.post("/convert-text-to-gloss")
async def convert_text_to_gloss(data: dict):
    """Convert text to gloss sequence for avatar animation"""
    text = data.get("text", "").lower()
    words = text.split()

    glosses = []
    for word in words:
        # Remove punctuation
        clean_word = "".join(c for c in word if c.isalnum())
        if clean_word in WORD_TO_GLOSS:
            glosses.append(WORD_TO_GLOSS[clean_word])
        else:
            # Try to find partial match
            for key, gloss in WORD_TO_GLOSS.items():
                if key in clean_word or clean_word in key:
                    glosses.append(gloss)
                    break

    return {"glosses": glosses}


@app.websocket("/ws/stream/{session_id}")
async def websocket_stream(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time sign language detection"""
    global active_connections
    from src.motion_capture.live_landmark_extractor import LiveHandLandmarkExtractor

    await websocket.accept()
    active_connections += 1
    print(f"Client connected: {session_id}, Active: {active_connections}")
    landmark_extractor = LiveHandLandmarkExtractor()

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "frame":
                keypoints = message.get("keypoints")
                image_data_url = message.get("image")

                start_time = time.time()
                keypoints_array = None

                if image_data_url:
                    try:
                        encoded = image_data_url.split(",", maxsplit=1)[1]
                        jpg_bytes = base64.b64decode(encoded)
                        jpg_array = np.frombuffer(jpg_bytes, dtype=np.uint8)
                        frame = cv2.imdecode(jpg_array, cv2.IMREAD_COLOR)
                        if frame is not None:
                            keypoints_array = landmark_extractor.extract(frame)
                    except Exception:
                        keypoints_array = None

                if keypoints_array is None and keypoints is not None:
                    keypoints_array = np.array(keypoints).flatten()

                if keypoints_array is not None and len(keypoints_array) > 0:
                    if sequence_service and sequence_service.is_loaded:
                        word, confidence, top5, buffering = sequence_service.predict_from_frame(
                            stream_id=session_id,
                            frame_keypoints=keypoints_array,
                        )
                    elif model_service and model_service.is_loaded:
                        word, confidence, top5 = model_service.predict(keypoints_array)
                        buffering = False
                    else:
                        word, confidence, top5, buffering = (
                            "MODEL_NOT_LOADED",
                            0.0,
                            [],
                            False,
                        )

                    response = {
                        "type": "prediction",
                        "data": {
                            "gloss": word,
                            "confidence": float(confidence),
                            "top5": top5,
                            "buffering": buffering,
                            "timestamp": int(time.time() * 1000),
                            "latency_ms": (time.time() - start_time) * 1000,
                        },
                    }
                else:
                    response = {
                        "type": "prediction",
                        "data": {
                            "gloss": "NO_KEYPOINTS",
                            "confidence": 0.0,
                            "top5": [],
                            "buffering": False,
                            "timestamp": int(time.time() * 1000),
                            "latency_ms": (time.time() - start_time) * 1000,
                        },
                    }

                await websocket.send_json(response)

            elif message.get("type") == "stop":
                response = {"type": "stopped", "data": {"message": "Streaming stopped"}}
                await websocket.send_json(response)

    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")
    finally:
        landmark_extractor.close()
        active_connections -= 1


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
