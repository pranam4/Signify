"""Hugging Face Whisper (`openai/whisper-tiny`) speech-to-text service."""

from __future__ import annotations

from dataclasses import dataclass, field
import io
from pathlib import Path
import threading
import time
from typing import Any

import av
import numpy as np
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.tokenization_whisper import (
    LANGUAGES,
    TO_LANGUAGE_CODE,
)

from .config import (
    STT_LANGUAGE_ALIASES,
    STT_TASKS,
    SpeechToTextConfig,
    stt_config_from_env,
)

_WHISPER_CHUNK_SECONDS = 30.0
_WHISPER_ENCODER_MAX_FRAMES = 3000


class AudioDecodeError(ValueError):
    """Raised when incoming audio bytes cannot be decoded into samples."""


@dataclass(slots=True)
class TranscriptionSegment:
    """Single timestamped segment emitted by Whisper."""

    start: float
    end: float
    text: str


@dataclass(slots=True)
class TranscriptionResult:
    """Structured speech-to-text output from `openai/whisper-tiny`."""

    text: str
    language: str | None
    task: str
    model_id: str
    duration_seconds: float
    processing_ms: float
    is_speech: bool
    segments: list[TranscriptionSegment] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize transcription output into JSON-safe primitives."""
        return {
            "text": self.text,
            "language": self.language,
            "task": self.task,
            "model_id": self.model_id,
            "duration_seconds": self.duration_seconds,
            "processing_ms": self.processing_ms,
            "is_speech": self.is_speech,
            "segments": [
                {"start": seg.start, "end": seg.end, "text": seg.text}
                for seg in self.segments
            ],
        }


def normalize_stt_language(language: str | None) -> str | None:
    """Normalize a locale or language name (e.g. 'en-IN', 'Hindi') into a Whisper code."""
    if language is None:
        return None
    raw = language.strip().lower()
    if not raw or raw == "auto":
        return None
    if raw in LANGUAGES:
        return raw
    if raw in TO_LANGUAGE_CODE:
        return TO_LANGUAGE_CODE[raw]
    if raw in STT_LANGUAGE_ALIASES:
        return STT_LANGUAGE_ALIASES[raw]

    base = raw.replace("_", "-").split("-", 1)[0]
    if base in LANGUAGES:
        return base
    if base in TO_LANGUAGE_CODE:
        return TO_LANGUAGE_CODE[base]
    if base in STT_LANGUAGE_ALIASES:
        return STT_LANGUAGE_ALIASES[base]

    raise ValueError(f"Unsupported Whisper language: {language!r}.")


def normalize_stt_task(task: str | None) -> str:
    """Validate and normalize Whisper task ('transcribe' or 'translate')."""
    resolved = (task or "transcribe").strip().lower()
    if resolved not in STT_TASKS:
        raise ValueError(
            f"Unsupported Whisper task {task!r}. Expected one of {STT_TASKS}."
        )
    return resolved


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    """Convert numeric numpy audio (1D or 2D frames x channels) to contiguous mono float32."""
    arr = np.asarray(audio)
    if arr.size == 0:
        raise AudioDecodeError("Audio array is empty.")

    if arr.ndim == 2:
        # Collapse channels regardless of (samples, channels) or (channels, samples)
        channel_axis = 1 if arr.shape[0] >= arr.shape[1] else 0
        arr = arr.astype(np.float32).mean(axis=channel_axis)
    elif arr.ndim != 1:
        raise AudioDecodeError(f"Expected 1D or 2D audio array, got shape {arr.shape}.")

    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        scale = float(max(abs(info.min), info.max))
        arr = arr.astype(np.float32) / scale
    else:
        arr = arr.astype(np.float32, copy=False)

    return np.ascontiguousarray(np.clip(arr, -1.0, 1.0), dtype=np.float32)


def resample_audio(
    audio: np.ndarray,
    orig_sample_rate: int,
    target_sample_rate: int = 16000,
) -> np.ndarray:
    """Resample mono float32 audio to `target_sample_rate` using PyAV."""
    if orig_sample_rate <= 0 or target_sample_rate <= 0:
        raise ValueError("Sample rates must be positive integers.")

    mono = _to_mono_float32(audio)
    if orig_sample_rate == target_sample_rate:
        return mono

    frame = av.AudioFrame.from_ndarray(
        mono[np.newaxis, :],
        format="flt",
        layout="mono",
    )
    frame.sample_rate = orig_sample_rate

    resampler = av.AudioResampler(
        format="flt",
        layout="mono",
        rate=target_sample_rate,
    )
    chunks = [out.to_ndarray().reshape(-1) for out in resampler.resample(frame)]
    chunks.extend(out.to_ndarray().reshape(-1) for out in resampler.resample(None))
    if not chunks:
        raise AudioDecodeError("Resampler produced no audio samples.")

    resampled = np.concatenate(chunks).astype(np.float32, copy=False)
    return np.clip(resampled, -1.0, 1.0)


def _decode_pcm_bytes(
    raw_bytes: bytes,
    *,
    pcm_format: str,
    input_sample_rate: int | None,
    channels: int,
    target_sample_rate: int,
) -> np.ndarray:
    if input_sample_rate is None or input_sample_rate <= 0:
        raise AudioDecodeError(
            "Positive `input_sample_rate` is required for raw PCM audio."
        )
    if channels < 1:
        raise AudioDecodeError("`channels` must be >= 1 for raw PCM audio.")

    dtype = np.dtype("<i2") if pcm_format == "pcm_s16le" else np.dtype("<f4")
    frame_width = dtype.itemsize * channels
    if len(raw_bytes) % frame_width != 0:
        raise AudioDecodeError(
            f"Raw PCM byte length {len(raw_bytes)} is not aligned to {frame_width}-byte frames."
        )

    samples = np.frombuffer(raw_bytes, dtype=dtype)
    if channels > 1:
        samples = samples.reshape(-1, channels)
    return resample_audio(samples, input_sample_rate, target_sample_rate)


def _decode_container_with_pyav(
    raw_bytes: bytes,
    *,
    target_sample_rate: int,
    format_hint: str | None = None,
) -> np.ndarray:
    open_kwargs: dict[str, Any] = {}
    if format_hint:
        open_kwargs["format"] = format_hint

    with av.open(io.BytesIO(raw_bytes), **open_kwargs) as container:
        audio_stream = next(
            (stream for stream in container.streams if stream.type == "audio"),
            None,
        )
        if audio_stream is None:
            raise AudioDecodeError("No audio stream found in media container.")

        resampler = av.AudioResampler(
            format="flt",
            layout="mono",
            rate=target_sample_rate,
        )
        chunks: list[np.ndarray] = []
        for frame in container.decode(audio_stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for flushed in resampler.resample(None):
            chunks.append(flushed.to_ndarray().reshape(-1))

    if not chunks:
        raise AudioDecodeError("Decoded audio stream contains zero samples.")

    audio = np.concatenate(chunks).astype(np.float32, copy=False)
    return np.clip(audio, -1.0, 1.0)


def decode_audio_bytes(
    raw_bytes: bytes,
    *,
    target_sample_rate: int = 16000,
    audio_format: str | None = None,
    input_sample_rate: int | None = None,
    channels: int = 1,
) -> tuple[np.ndarray, float]:
    """Decode encoded audio (WAV/WebM/OGG/MP3/MP4/FLAC) or raw PCM to mono 16kHz float32."""
    if not raw_bytes:
        raise AudioDecodeError("Audio payload is empty.")

    fmt = audio_format.strip().lower().lstrip(".") if audio_format else None
    if fmt in {"pcm_s16le", "pcm_f32le"}:
        audio = _decode_pcm_bytes(
            raw_bytes,
            pcm_format=fmt,
            input_sample_rate=input_sample_rate,
            channels=channels,
            target_sample_rate=target_sample_rate,
        )
        return audio, float(audio.shape[0] / target_sample_rate)

    audio: np.ndarray | None = None
    last_err: Exception | None = None

    # Auto-probe container headers first; retry with explicit format hint if provided.
    hints: list[str | None] = [None]
    if fmt and fmt != "auto":
        hints.append(fmt)

    for hint in hints:
        try:
            audio = _decode_container_with_pyav(
                raw_bytes,
                target_sample_rate=target_sample_rate,
                format_hint=hint,
            )
            break
        except Exception as exc:
            last_err = exc

    if audio is None:
        try:
            sf_audio, sf_rate = sf.read(
                io.BytesIO(raw_bytes),
                dtype="float32",
                always_2d=True,
            )
            audio = resample_audio(sf_audio, sf_rate, target_sample_rate)
        except Exception as exc:
            detail = last_err or exc
            raise AudioDecodeError(f"Failed to decode audio payload: {detail}") from exc

    if audio.size == 0:
        raise AudioDecodeError("Decoded audio contains zero samples.")

    duration_seconds = float(audio.shape[0] / target_sample_rate)
    return audio, duration_seconds


class WhisperSpeechToText:
    """Thread-safe wrapper around Hugging Face `openai/whisper-tiny`."""

    def __init__(self, config: SpeechToTextConfig | None = None) -> None:
        self.config = config or stt_config_from_env()
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._processor: WhisperProcessor | None = None
        self._model: WhisperForConditionalGeneration | None = None
        self._torch_device: torch.device | None = None
        self._torch_dtype: torch.dtype = torch.float32
        self._id_to_lang: dict[int, str] = {}

    @property
    def is_loaded(self) -> bool:
        """Whether the Whisper processor and model weights are loaded in memory."""
        return self._model is not None and self._processor is not None

    @property
    def device(self) -> str:
        """Resolved PyTorch device string ('cpu' or 'cuda')."""
        if self._torch_device is not None:
            return str(self._torch_device)
        return self._resolve_device_str()

    def _resolve_device_str(self) -> str:
        requested = (self.config.device or "").strip().lower()
        if requested and requested != "auto":
            return requested
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _resolve_dtype(self, device_str: str) -> torch.dtype:
        requested = (self.config.dtype or "").strip().lower()
        if requested:
            mapping = {
                "float32": torch.float32,
                "fp32": torch.float32,
                "float16": torch.float16,
                "fp16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
            }
            if requested not in mapping:
                raise ValueError(f"Unsupported torch dtype: {self.config.dtype!r}.")
            return mapping[requested]
        return torch.float16 if device_str.startswith("cuda") else torch.float32

    def load(self) -> None:
        """Load `openai/whisper-tiny` processor and model weights once (thread-safe)."""
        if self.is_loaded:
            return

        with self._load_lock:
            if self.is_loaded:
                return

            device_str = self._resolve_device_str()
            torch_dtype = self._resolve_dtype(device_str)
            torch_device = torch.device(device_str)

            hub_kwargs: dict[str, Any] = {
                "local_files_only": self.config.local_files_only,
            }
            if self.config.cache_dir is not None:
                hub_kwargs["cache_dir"] = str(self.config.cache_dir)

            processor = WhisperProcessor.from_pretrained(
                self.config.model_id,
                **hub_kwargs,
            )
            model = WhisperForConditionalGeneration.from_pretrained(
                self.config.model_id,
                dtype=torch_dtype,
                **hub_kwargs,
            )
            model.to(torch_device)
            model.eval()

            lang_to_id = getattr(model.generation_config, "lang_to_id", None) or {}
            self._id_to_lang = {
                int(token_id): token.strip("<|>")
                for token, token_id in lang_to_id.items()
            }
            self._processor = processor
            self._model = model
            self._torch_device = torch_device
            self._torch_dtype = torch_dtype

    def close(self) -> None:
        """Release model and processor references."""
        with self._load_lock:
            self._model = None
            self._processor = None
            self._torch_device = None
            self._id_to_lang = {}

    def __enter__(self) -> "WhisperSpeechToText":
        self.load()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _extract_detected_language(
        self,
        outputs: Any,
        input_features: torch.Tensor,
    ) -> str | None:
        candidate_seqs: list[torch.Tensor] = []
        if isinstance(outputs, dict):
            segments = outputs.get("segments")
            if segments and segments[0]:
                first_result = segments[0][0].get("result")
                if isinstance(first_result, dict) and "sequences" in first_result:
                    candidate_seqs.append(first_result["sequences"])
            if "sequences" in outputs:
                candidate_seqs.append(outputs["sequences"][0])
        elif hasattr(outputs, "sequences"):
            candidate_seqs.append(outputs.sequences[0])

        for seq in candidate_seqs:
            for token_id in seq[:4].tolist():
                if int(token_id) in self._id_to_lang:
                    return self._id_to_lang[int(token_id)]

        if self._model is not None and hasattr(self._model, "detect_language"):
            lang_ids = self._model.detect_language(
                input_features[:, :, :_WHISPER_ENCODER_MAX_FRAMES]
            )
            return self._id_to_lang.get(int(lang_ids[0]))
        return None

    def _extract_segments(self, outputs: Any) -> list[TranscriptionSegment]:
        if not isinstance(outputs, dict) or self._processor is None:
            return []
        raw_batch_segments = outputs.get("segments")
        if not raw_batch_segments or not raw_batch_segments[0]:
            return []

        segments: list[TranscriptionSegment] = []
        for seg in raw_batch_segments[0]:
            seg_text = self._processor.decode(
                seg["tokens"],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            if not seg_text:
                continue
            segments.append(
                TranscriptionSegment(
                    start=round(float(seg["start"]), 2),
                    end=round(float(seg["end"]), 2),
                    text=seg_text,
                )
            )
        return segments

    def transcribe_array(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int | None = None,
        language: str | None = None,
        task: str | None = None,
        return_timestamps: bool | None = None,
    ) -> TranscriptionResult:
        """Transcribe or translate a numpy audio waveform using `openai/whisper-tiny`."""
        t0 = time.perf_counter()
        target_sr = self.config.sample_rate
        src_sr = sample_rate or target_sr
        mono = resample_audio(audio, src_sr, target_sr)
        duration_seconds = float(mono.shape[0] / target_sr)

        if duration_seconds > self.config.max_audio_seconds:
            raise ValueError(
                f"Audio duration {duration_seconds:.1f}s exceeds limit of "
                f"{self.config.max_audio_seconds:.1f}s."
            )

        norm_lang = normalize_stt_language(
            language if language is not None else self.config.default_language
        )
        resolved_task = normalize_stt_task(task or self.config.default_task)
        use_timestamps = (
            self.config.return_timestamps
            if return_timestamps is None
            else bool(return_timestamps)
        )

        # Gate pure silence so Whisper does not hallucinate phantom words.
        rms = float(np.sqrt(np.mean(np.square(mono))))
        if rms < self.config.min_rms_energy:
            return TranscriptionResult(
                text="",
                language=norm_lang,
                task=resolved_task,
                model_id=self.config.model_id,
                duration_seconds=round(duration_seconds, 3),
                processing_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                is_speech=False,
                segments=[],
            )

        self.load()
        assert self._processor is not None
        assert self._model is not None
        assert self._torch_device is not None

        is_longform = duration_seconds > _WHISPER_CHUNK_SECONDS
        processor_kwargs: dict[str, Any] = {
            "sampling_rate": target_sr,
            "return_attention_mask": True,
            "return_tensors": "pt",
        }
        if is_longform:
            processor_kwargs["truncation"] = False
            processor_kwargs["padding"] = "longest"

        inputs = self._processor(mono, **processor_kwargs)
        input_features = inputs.input_features.to(
            device=self._torch_device,
            dtype=self._torch_dtype,
        )
        attention_mask = inputs.attention_mask.to(device=self._torch_device)

        gen_kwargs: dict[str, Any] = {
            "task": resolved_task,
            "return_timestamps": use_timestamps,
            "return_dict_in_generate": True,
        }
        if norm_lang is not None:
            gen_kwargs["language"] = norm_lang

        with self._infer_lock:
            with torch.inference_mode():
                outputs = self._model.generate(
                    input_features,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )
                detected_lang = norm_lang or self._extract_detected_language(
                    outputs,
                    input_features,
                )

        sequences = (
            outputs["sequences"] if isinstance(outputs, dict) else outputs.sequences
        )
        text = self._processor.batch_decode(
            sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        segments = self._extract_segments(outputs) if use_timestamps else []
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return TranscriptionResult(
            text=text,
            language=detected_lang,
            task=resolved_task,
            model_id=self.config.model_id,
            duration_seconds=round(duration_seconds, 3),
            processing_ms=round(elapsed_ms, 2),
            is_speech=bool(text),
            segments=segments,
        )

    def transcribe_bytes(
        self,
        raw_bytes: bytes,
        *,
        language: str | None = None,
        task: str | None = None,
        return_timestamps: bool | None = None,
        audio_format: str | None = None,
        input_sample_rate: int | None = None,
        channels: int = 1,
    ) -> TranscriptionResult:
        """Decode audio bytes (WAV/WebM/OGG/MP3/MP4/FLAC or raw PCM) and transcribe."""
        audio, _ = decode_audio_bytes(
            raw_bytes,
            target_sample_rate=self.config.sample_rate,
            audio_format=audio_format,
            input_sample_rate=input_sample_rate,
            channels=channels,
        )
        return self.transcribe_array(
            audio,
            sample_rate=self.config.sample_rate,
            language=language,
            task=task,
            return_timestamps=return_timestamps,
        )

    def transcribe_file(
        self,
        file_path: str | Path,
        *,
        language: str | None = None,
        task: str | None = None,
        return_timestamps: bool | None = None,
    ) -> TranscriptionResult:
        """Read an audio file from disk and transcribe it with `openai/whisper-tiny`."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")
        return self.transcribe_bytes(
            path.read_bytes(),
            language=language,
            task=task,
            return_timestamps=return_timestamps,
            audio_format=path.suffix.lstrip(".") or None,
        )
