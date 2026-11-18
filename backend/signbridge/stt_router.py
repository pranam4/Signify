"""FastAPI HTTP and WebSocket routes for `openai/whisper-tiny` speech-to-text."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from transformers.models.whisper.tokenization_whisper import LANGUAGES

from .config import STT_TASKS, stt_config_from_env
from .speech_to_text import (
    AudioDecodeError,
    WhisperSpeechToText,
    normalize_stt_language,
    normalize_stt_task,
)

stt_router = APIRouter(tags=["speech-to-text"])
_PCM_FORMATS = {"pcm_s16le", "pcm_f32le"}


def get_stt_service(app_state: Any) -> WhisperSpeechToText:
    """Retrieve or lazily initialize the shared WhisperSpeechToText service."""
    service = getattr(app_state, "stt_service", None)
    if service is None:
        service = WhisperSpeechToText(stt_config_from_env())
        app_state.stt_service = service
    return service


def _parse_optional_bool(raw: str | None) -> bool | None:
    if raw is None:
        return None
    val = raw.strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {raw!r}.")


@stt_router.get("/api/stt/health")
def stt_health(request: Request) -> dict[str, object]:
    """Report Whisper speech-to-text configuration and readiness."""
    stt = get_stt_service(request.app.state)
    return {
        "status": "ok",
        "model_id": stt.config.model_id,
        "loaded": stt.is_loaded,
        "device": stt.device,
        "sample_rate": stt.config.sample_rate,
        "supported_tasks": list(STT_TASKS),
        "signbridge_languages": ["en", "hi", "kn"],
        "total_whisper_languages": len(LANGUAGES),
    }


@stt_router.post("/api/stt/transcribe")
async def transcribe_audio(
    request: Request,
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
    task: str | None = Form(default=None),
    return_timestamps: bool | None = Form(default=None),
    audio_format: str | None = Form(default=None),
    sample_rate: int | None = Form(default=None),
    channels: int = Form(default=1),
) -> dict[str, object]:
    """Transcribe or translate an uploaded audio clip using `openai/whisper-tiny`."""
    stt = get_stt_service(request.app.state)
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
    if len(raw_bytes) > stt.config.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Audio upload exceeds maximum size of {stt.config.max_upload_bytes} bytes.",
        )

    qp = request.query_params
    resolved_language = language if language is not None else qp.get("language")
    resolved_task = task if task is not None else qp.get("task")
    resolved_format = (
        audio_format
        if audio_format is not None
        else qp.get("audio_format")
        or (Path(file.filename).suffix.lstrip(".") if file.filename else None)
    )

    try:
        resolved_timestamps = (
            return_timestamps
            if return_timestamps is not None
            else _parse_optional_bool(qp.get("return_timestamps"))
        )
        resolved_sr = (
            sample_rate
            if sample_rate is not None
            else (int(qp["sample_rate"]) if "sample_rate" in qp else None)
        )
        resolved_channels = (
            int(qp["channels"]) if "channels" in qp else channels
        )

        result = await asyncio.to_thread(
            stt.transcribe_bytes,
            raw_bytes,
            language=resolved_language,
            task=resolved_task,
            return_timestamps=resolved_timestamps,
            audio_format=resolved_format,
            input_sample_rate=resolved_sr,
            channels=resolved_channels,
        )
        return result.to_dict()
    except (AudioDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@stt_router.websocket("/ws/stt")
async def stt_websocket(
    websocket: WebSocket,
    language: str | None = None,
    task: str = "transcribe",
    audio_format: str = "auto",
    sample_rate: int = 16000,
    channels: int = 1,
    return_timestamps: bool = True,
) -> None:
    """WebSocket endpoint for utterance-by-utterance or buffered PCM speech-to-text."""
    await websocket.accept()
    stt = get_stt_service(websocket.app.state)

    try:
        state = {
            "language": normalize_stt_language(language),
            "task": normalize_stt_task(task),
            "audio_format": (audio_format or "auto").strip().lower().lstrip("."),
            "sample_rate": int(sample_rate),
            "channels": int(channels),
            "return_timestamps": bool(return_timestamps),
        }
        if state["sample_rate"] <= 0 or state["channels"] < 1:
            raise ValueError("`sample_rate` must be > 0 and `channels` must be >= 1.")
    except ValueError as exc:
        await websocket.send_json({"type": "error", "detail": str(exc)})
        await websocket.close(code=1008)
        return

    pcm_buffer = bytearray()

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            if message.get("bytes") is not None:
                chunk: bytes = message["bytes"]
                if not chunk:
                    await websocket.send_json(
                        {"type": "error", "detail": "Audio chunk is empty."}
                    )
                    continue

                if state["audio_format"] in _PCM_FORMATS:
                    bytes_per_sample = 2 if state["audio_format"] == "pcm_s16le" else 4
                    max_pcm_bytes = int(
                        stt.config.max_audio_seconds
                        * state["sample_rate"]
                        * state["channels"]
                        * bytes_per_sample
                    )
                    if len(pcm_buffer) + len(chunk) > max_pcm_bytes:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "detail": "Buffered PCM audio exceeds maximum allowed duration.",
                            }
                        )
                        continue
                    pcm_buffer.extend(chunk)
                    continue

                try:
                    result = await asyncio.to_thread(
                        stt.transcribe_bytes,
                        chunk,
                        language=state["language"],
                        task=state["task"],
                        return_timestamps=state["return_timestamps"],
                        audio_format=state["audio_format"],
                        input_sample_rate=state["sample_rate"],
                        channels=state["channels"],
                    )
                    await websocket.send_json(
                        {"type": "transcription", **result.to_dict()}
                    )
                except (AudioDecodeError, ValueError) as exc:
                    await websocket.send_json({"type": "error", "detail": str(exc)})
                continue

            if message.get("text") is not None:
                try:
                    cmd = json.loads(message["text"])
                    if not isinstance(cmd, dict):
                        raise ValueError("JSON control message must be an object.")
                except ValueError as exc:
                    await websocket.send_json({"type": "error", "detail": str(exc)})
                    continue

                cmd_type = str(cmd.get("type", "")).strip().lower()
                if cmd_type == "ping":
                    await websocket.send_json({"type": "pong"})
                elif cmd_type == "reset":
                    pcm_buffer.clear()
                    await websocket.send_json({"type": "reset", "buffered_bytes": 0})
                elif cmd_type == "config":
                    try:
                        if "language" in cmd:
                            state["language"] = normalize_stt_language(cmd["language"])
                        if "task" in cmd:
                            state["task"] = normalize_stt_task(cmd["task"])
                        if "audio_format" in cmd:
                            state["audio_format"] = (
                                str(cmd["audio_format"] or "auto")
                                .strip()
                                .lower()
                                .lstrip(".")
                            )
                        if "sample_rate" in cmd:
                            sr = int(cmd["sample_rate"])
                            if sr <= 0:
                                raise ValueError("`sample_rate` must be > 0.")
                            state["sample_rate"] = sr
                        if "channels" in cmd:
                            ch = int(cmd["channels"])
                            if ch < 1:
                                raise ValueError("`channels` must be >= 1.")
                            state["channels"] = ch
                        if "return_timestamps" in cmd:
                            state["return_timestamps"] = bool(cmd["return_timestamps"])
                        await websocket.send_json({"type": "configured", **state})
                    except ValueError as exc:
                        await websocket.send_json({"type": "error", "detail": str(exc)})
                elif cmd_type in {"flush", "transcribe"}:
                    if not pcm_buffer:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "detail": "No buffered audio available to transcribe.",
                            }
                        )
                        continue
                    buffered = bytes(pcm_buffer)
                    if not cmd.get("keep_buffer", False):
                        pcm_buffer.clear()
                    try:
                        result = await asyncio.to_thread(
                            stt.transcribe_bytes,
                            buffered,
                            language=state["language"],
                            task=state["task"],
                            return_timestamps=state["return_timestamps"],
                            audio_format=state["audio_format"],
                            input_sample_rate=state["sample_rate"],
                            channels=state["channels"],
                        )
                        await websocket.send_json(
                            {"type": "transcription", **result.to_dict()}
                        )
                    except (AudioDecodeError, ValueError) as exc:
                        await websocket.send_json({"type": "error", "detail": str(exc)})
                else:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "detail": f"Unsupported control command: {cmd_type!r}.",
                        }
                    )
    except WebSocketDisconnect:
        pass
