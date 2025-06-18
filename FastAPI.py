"""
FastAPI.py

FastAPI application for image classification using a pre-trained ResNet50 model (Keras/TensorFlow).
- Provides REST API endpoints for health checks and image prediction.
- Accepts image uploads and returns top-5 predicted classes with confidence scores.
- Loads and manages the ResNet50 model in memory for efficient inference.
- Designed for scalable, production-ready deployment (local, Docker, or cloud).
"""

import os
import io
import logging
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input,
    decode_predictions,
)
from tensorflow.keras.preprocessing import image as keras_image

# --- Use FastAPI lifespan for startup/shutdown events ---
from contextlib import asynccontextmanager

class Settings(BaseSettings):
    """
    Configuration settings loaded from environment or defaults.
    """
    ALLOWED_ORIGINS: List[str] = [
        # Restrict to trusted domains in production
        "https://yourdomain.com",
        "http://localhost",
        "http://127.0.0.1",
    ]
    MAX_UPLOAD_SIZE: int = 5 * 1024 * 1024  # 5 MB
    MODEL_WEIGHTS: str = "imagenet"

    class Config:
        env_file = ".env"


# Initialize settings and logging
settings = Settings()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

model = None  # Will hold the loaded ResNet50 model

# --- Use FastAPI lifespan event for model loading ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = ResNet50(weights=settings.MODEL_WEIGHTS)
        logger.info("ResNet50 model loaded successfully.")
    except Exception as e:
        logger.critical(f"Failed to load ResNet50 model: {e}")
        import sys
        sys.exit(1)  # Exit if model fails to load
    yield
   

app = FastAPI(
    title="ResNet50 Image Classification API",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware with restricted origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class PredictionResponse(BaseModel):
    """
    Schema for prediction response.
    """
    predicted_class: str
    confidence: float
    top_5: list = None  # Optional: include top-5 predictions

@app.get("/health")
async def health() -> JSONResponse:
    """
    Health check endpoint to verify the service is running.
    """
    return JSONResponse(status_code=200, content={"status": "ok"})

def preprocess_image(img: Image.Image) -> np.ndarray:
    """
    Resize and preprocess a PIL Image for ResNet50 inference.

    Args:
        img (PIL.Image.Image): Input image.

    Returns:
        np.ndarray: Preprocessed image tensor.
    """
    img = img.resize((224, 224))
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

@app.post("/predict/", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """
    Endpoint to classify an uploaded image.

    Validates content type and file size, preprocesses the image, runs inference,
    and returns the top-1 ImageNet prediction.
    """
    if model is None:
        logger.error("Model is not loaded.")
        raise HTTPException(status_code=500, detail="Model not loaded.")

    # Validate file type
    if not file.content_type.startswith("image/"):
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    # Enforce upload size limit (read in chunks to avoid loading large files into memory)
    size = 0
    chunk_size = 1024 * 1024  # 1 MB
    contents = b""
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        size += len(chunk)
        if size > settings.MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="File too large.")
        contents += chunk
    await file.seek(0)

    try:
        # Read and load image
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        # Offload CPU-bound preprocessing to a thread pool
        img_tensor = await run_in_threadpool(preprocess_image, img)

        # Offload inference to thread pool to avoid blocking the event loop
        preds = await run_in_threadpool(model.predict, img_tensor)

        # Decode the top-5 predictions
        decoded = decode_predictions(preds, top=5)[0]
        class_name = decoded[0][1]
        confidence = float(decoded[0][2])
        top_5 = [{"class": c[1], "confidence": float(c[2])} for c in decoded]

        logger.info(f"Prediction: class={class_name}, confidence={confidence}")
        return PredictionResponse(predicted_class=class_name, confidence=confidence, top_5=top_5)

    except HTTPException:
        # Propagate HTTP errors directly
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed.")