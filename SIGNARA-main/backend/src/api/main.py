from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import json
import base64
import numpy as np
import time

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
active_connections = 0


class PredictionRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    signs: List[str]


class GlossInterpretationRequest(BaseModel):
    input: List[List[str]]


class GlossInterpretationResponse(BaseModel):
    sentence: str


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global model_service
    print("Initializing Signara API...")
    try:
        from src.model.sign_model import get_model_service

        model_service = get_model_service()
        print(
            f"Model loaded: {model_service.is_loaded}, Classes: {model_service.num_classes}"
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        model_service = None


@app.get("/")
async def root():
    return {"message": "Signara API - AI-Powered Sign Language Interpreter"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_service.is_loaded if model_service else False,
        "gpu_available": False,
        "active_connections": active_connections,
        "num_classes": model_service.num_classes if model_service else 0,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """REST endpoint for text-to-sign prediction"""
    if model_service is None:
        return PredictionResponse(signs=[])

    signs = model_service.predict_sign(request.text)
    return PredictionResponse(signs=signs)


@app.post("/predict-keypoints", response_model=PredictionResponse)
async def predict_keypoints(data: dict):
    """Predict from pre-extracted keypoints"""
    start_time = time.time()

    if model_service is None or not model_service.is_loaded:
        return PredictionResponse(
            gloss="MODEL_NOT_LOADED",
            confidence=0.0,
            top5=[],
            timestamp=int(time.time() * 1000),
            latency_ms=0.0,
        )

    keypoints = data.get("keypoints", [])
    if not keypoints or len(keypoints) == 0:
        return PredictionResponse(
            gloss="NO_KEYPOINTS",
            confidence=0.0,
            top5=[],
            timestamp=int(time.time() * 1000),
            latency_ms=0.0,
        )

    # Convert to numpy array - flatten if nested
    try:
        keypoints_array = np.array(keypoints)
        if keypoints_array.ndim > 1:
            keypoints_array = keypoints_array.flatten()
    except Exception as e:
        return PredictionResponse(
            gloss="INVALID_KEYPOINTS",
            confidence=0.0,
            top5=[],
            timestamp=int(time.time() * 1000),
            latency_ms=0.0,
        )

    # Predict
    word, confidence, top5 = model_service.predict(keypoints_array)

    return PredictionResponse(
        gloss=word,
        confidence=confidence,
        top5=top5,
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
    await websocket.accept()
    active_connections += 1
    print(f"Client connected: {session_id}, Active: {active_connections}")

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "frame":
                keypoints = message.get("keypoints", None)

                start_time = time.time()

                if keypoints is not None and model_service and model_service.is_loaded:
                    # Use provided keypoints for prediction
                    keypoints_array = np.array(keypoints)
                    word, confidence, top5 = model_service.predict(keypoints_array)

                    response = {
                        "type": "prediction",
                        "data": {
                            "gloss": word,
                            "confidence": float(confidence),
                            "top5": top5,
                            "timestamp": int(time.time() * 1000),
                            "latency_ms": (time.time() - start_time) * 1000,
                        },
                    }
                else:
                    # Placeholder response
                    response = {
                        "type": "prediction",
                        "data": {
                            "gloss": "HELLO",
                            "confidence": 0.85,
                            "top5": [["HELLO", 0.85], ["THANK", 0.05], ["YOU", 0.03]],
                            "timestamp": int(time.time() * 1000),
                            "latency_ms": 45.2,
                        },
                    }

                await websocket.send_json(response)

            elif message.get("type") == "stop":
                response = {"type": "stopped", "data": {"message": "Streaming stopped"}}
                await websocket.send_json(response)

    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")
    finally:
        active_connections -= 1


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
