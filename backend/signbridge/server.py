"""FastAPI HTTP & WebSocket server bridging React camera capture to Python MediaPipe."""

import asyncio
from contextlib import asynccontextmanager
import os
import threading

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import (
    DEFAULT_HOLISTIC_MODEL_PATH,
    DEFAULT_MODEL_PATH,
    GESTURE_TRANSLATIONS,
    RUNNING_MODES,
    config_from_env,
    stt_config_from_env,
)
from .model_manager import ensure_model_asset
from .recognizer import SignGestureRecognizer
from .speech_to_text import WhisperSpeechToText
from .stt_router import get_stt_service, stt_router

_IMAGE_LOCK = threading.Lock()


@asynccontextmanager
async def lifespan(application: FastAPI):
    cfg = config_from_env(running_mode="IMAGE")
    ensure_model_asset(cfg.model_path, cfg.model_url)
    if cfg.enable_holistic:
        ensure_model_asset(cfg.holistic_model_path, cfg.holistic_model_url)
    application.state.image_recognizer = SignGestureRecognizer(cfg)
    application.state.image_recognizers = {cfg.enable_holistic: application.state.image_recognizer}

    stt_service = WhisperSpeechToText(stt_config_from_env())
    if os.getenv("SIGNBRIDGE_PRELOAD_STT", "").strip().lower() in {"1", "true", "yes"}:
        await asyncio.to_thread(stt_service.load)
    application.state.stt_service = stt_service

    try:
        yield
    finally:
        cache = getattr(application.state, "image_recognizers", None) or {}
        for rec in cache.values():
            if rec is not None:
                rec.close()
        application.state.image_recognizers = {}
        application.state.image_recognizer = None
        stt = getattr(application.state, "stt_service", None)
        if stt is not None:
            stt.close()
            application.state.stt_service = None


app = FastAPI(title="SignBridge MediaPipe Backend", version="0.3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(stt_router)


def _get_image_recognizer(enable_holistic: bool | None = None) -> SignGestureRecognizer:
    cfg = config_from_env(running_mode="IMAGE", enable_holistic=enable_holistic)
    cache = getattr(app.state, "image_recognizers", None)
    if cache is None:
        cache = {}
        app.state.image_recognizers = cache

    rec = cache.get(cfg.enable_holistic)
    if rec is None:
        rec = SignGestureRecognizer(cfg)
        cache[cfg.enable_holistic] = rec
        if cfg.enable_holistic:
            app.state.image_recognizer = rec
    return rec


def _recognize_image_sync(
    raw_bytes: bytes,
    include_world_landmarks: bool,
    include_face_mesh: bool,
    enable_holistic: bool | None,
) -> dict[str, object]:
    with _IMAGE_LOCK:
        rec = _get_image_recognizer(enable_holistic=enable_holistic)
        pred = rec.process_encoded_bytes(raw_bytes)
    payload = pred.to_ws_dict(
        include_world_landmarks=include_world_landmarks,
        include_face_mesh=include_face_mesh,
    )
    payload["mode"] = "IMAGE"
    return payload


@app.get("/api/health")
def health_check() -> dict[str, object]:
    """Verify backend readiness and local MediaPipe + Holistic + Whisper STT availability."""
    cfg = config_from_env()
    model_path = ensure_model_asset(cfg.model_path, cfg.model_url)
    holistic_path = (
        ensure_model_asset(cfg.holistic_model_path, cfg.holistic_model_url)
        if cfg.enable_holistic
        else DEFAULT_HOLISTIC_MODEL_PATH
    )
    stt = get_stt_service(app.state)
    return {
        "status": "ok",
        "model": model_path.name,
        "custom_model": model_path.resolve() != DEFAULT_MODEL_PATH.resolve(),
        "holistic_enabled": cfg.enable_holistic,
        "holistic_model": holistic_path.name if cfg.enable_holistic else None,
        "holistic_landmarks_total": 553 if cfg.enable_holistic else 0,
        "spatial_grammar_features": (
            [
                "torso_relative_signing_space",
                "non_manual_markers",
                "eyebrow_question_mood",
                "head_nod_shake_tilt",
                "eye_gaze_referent_focus",
                "directional_verb_trajectories",
                "body_role_shift",
            ]
            if cfg.enable_holistic
            else ["frame_zone_fallback"]
        ),
        "running_modes": list(RUNNING_MODES),
        "supported_baseline_gestures": list(GESTURE_TRANSLATIONS.keys()),
        "stt": {
            "model_id": stt.config.model_id,
            "loaded": stt.is_loaded,
            "device": stt.device,
        },
    }


@app.get("/api/signs")
def list_signs() -> dict[str, object]:
    """Expose baseline gesture translation mappings."""
    return {"gestures": GESTURE_TRANSLATIONS}


@app.post("/api/recognize-image")
async def recognize_image(
    file: UploadFile = File(...),
    include_world_landmarks: bool = False,
    include_face_mesh: bool = False,
    enable_holistic: bool | None = None,
) -> dict[str, object]:
    """Run MediaPipe GestureRecognizer + HolisticLandmarker in IMAGE mode."""
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty.")

    try:
        return await asyncio.to_thread(
            _recognize_image_sync,
            raw_bytes,
            include_world_landmarks,
            include_face_mesh,
            enable_holistic,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.websocket("/ws/recognize")
async def recognize_stream(
    websocket: WebSocket,
    include_world_landmarks: bool = False,
    include_face_mesh: bool = False,
    mirrored: bool = False,
    enable_holistic: bool | None = None,
) -> None:
    """Receive JPEG frames from the React camera UI and stream back holistic landmarks + signs."""
    await websocket.accept()
    cfg = config_from_env(running_mode="VIDEO", enable_holistic=enable_holistic)
    cfg.invert_handedness = mirrored
    recognizer = SignGestureRecognizer(cfg)
    try:
        while True:
            frame_bytes = await websocket.receive_bytes()
            # Run synchronous C++ MediaPipe inference off the event loop.
            prediction = await asyncio.to_thread(
                recognizer.process_encoded_bytes,
                frame_bytes,
            )
            await websocket.send_json(
                prediction.to_ws_dict(
                    include_world_landmarks=include_world_landmarks,
                    include_face_mesh=include_face_mesh,
                )
            )
    except WebSocketDisconnect:
        pass
    finally:
        recognizer.close()
