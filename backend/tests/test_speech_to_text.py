"""Tests for SignBridge `openai/whisper-tiny` speech-to-text service and API."""

from __future__ import annotations

import io
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

from signbridge.config import DEFAULT_STT_MODEL_ID, SpeechToTextConfig
from signbridge.server import app
from signbridge.speech_to_text import (
    AudioDecodeError,
    WhisperSpeechToText,
    decode_audio_bytes,
    normalize_stt_language,
    normalize_stt_task,
)

FIXTURE_WAV = Path(__file__).resolve().parent / "fixtures" / "hello_thank_you.wav"


@pytest.fixture(scope="module")
def stt_service() -> WhisperSpeechToText:
    cfg = SpeechToTextConfig(model_id=DEFAULT_STT_MODEL_ID, local_files_only=True)
    with WhisperSpeechToText(cfg) as service:
        yield service


def test_normalize_stt_language_and_task() -> None:
    assert normalize_stt_language(None) is None
    assert normalize_stt_language("") is None
    assert normalize_stt_language("auto") is None
    assert normalize_stt_language("en") == "en"
    assert normalize_stt_language("en-IN") == "en"
    assert normalize_stt_language("EN_US") == "en"
    assert normalize_stt_language("English") == "en"
    assert normalize_stt_language("hi-IN") == "hi"
    assert normalize_stt_language("Hindi") == "hi"
    assert normalize_stt_language("kn-IN") == "kn"
    assert normalize_stt_language("Kannada") == "kn"

    with pytest.raises(ValueError, match="Unsupported Whisper language"):
        normalize_stt_language("xx-invalid")

    assert normalize_stt_task(None) == "transcribe"
    assert normalize_stt_task("TRANSLATE") == "translate"
    with pytest.raises(ValueError, match="Unsupported Whisper task"):
        normalize_stt_task("summarize")


def test_decode_audio_bytes_formats_and_errors() -> None:
    # 1. Stereo 44.1kHz WAV -> mono 16kHz float32
    sr_in = 44100
    t = np.linspace(0.0, 0.5, int(sr_in * 0.5), dtype=np.float32)
    tone = 0.25 * np.sin(2.0 * np.pi * 440.0 * t)
    stereo = np.stack([tone, tone * 0.5], axis=1)

    wav_buf = io.BytesIO()
    sf.write(wav_buf, stereo, sr_in, format="WAV", subtype="PCM_16")
    decoded_wav, dur_wav = decode_audio_bytes(wav_buf.getvalue(), target_sample_rate=16000)
    assert decoded_wav.dtype == np.float32
    assert decoded_wav.ndim == 1
    assert abs(dur_wav - 0.5) < 0.03

    # 2. OGG/Opus container at 48kHz (browser WebRTC/Opus rate) -> mono 16kHz
    sr_opus = 48000
    t_opus = np.linspace(0.0, 0.5, int(sr_opus * 0.5), dtype=np.float32)
    tone_opus = 0.25 * np.sin(2.0 * np.pi * 440.0 * t_opus)
    ogg_buf = io.BytesIO()
    sf.write(ogg_buf, tone_opus, sr_opus, format="OGG", subtype="OPUS")
    decoded_ogg, dur_ogg = decode_audio_bytes(ogg_buf.getvalue(), target_sample_rate=16000)
    assert decoded_ogg.dtype == np.float32
    assert decoded_ogg.ndim == 1
    assert abs(dur_ogg - 0.5) < 0.08

    # 3. Raw PCM S16LE & F32LE
    pcm16 = (np.sin(2.0 * np.pi * 300.0 * np.linspace(0, 0.25, 12000)) * 16000).astype("<i2")
    decoded_pcm, dur_pcm = decode_audio_bytes(
        pcm16.tobytes(),
        target_sample_rate=16000,
        audio_format="pcm_s16le",
        input_sample_rate=48000,
        channels=1,
    )
    assert decoded_pcm.dtype == np.float32
    assert abs(dur_pcm - 0.25) < 0.02

    pcm32 = np.sin(2.0 * np.pi * 300.0 * np.linspace(0, 0.25, 4000)).astype("<f4")
    decoded_f32, dur_f32 = decode_audio_bytes(
        pcm32.tobytes(),
        target_sample_rate=16000,
        audio_format="pcm_f32le",
        input_sample_rate=16000,
        channels=1,
    )
    assert decoded_f32.dtype == np.float32
    assert abs(dur_f32 - 0.25) < 0.01

    # 4. Error handling
    with pytest.raises(AudioDecodeError, match="empty"):
        decode_audio_bytes(b"")
    with pytest.raises(AudioDecodeError, match="Failed to decode"):
        decode_audio_bytes(b"not-valid-audio-data")
    with pytest.raises(AudioDecodeError, match="aligned"):
        decode_audio_bytes(
            b"\x01\x02\x03",
            audio_format="pcm_s16le",
            input_sample_rate=16000,
        )


def test_whisper_tiny_transcribes_speech_and_gates_silence(
    stt_service: WhisperSpeechToText,
) -> None:
    # Pure silence must be gated cleanly without phantom hallucinations.
    silence = np.zeros(16000, dtype=np.float32)
    silent_res = stt_service.transcribe_array(silence, sample_rate=16000, language="en")
    assert silent_res.text == ""
    assert silent_res.is_speech is False
    assert silent_res.segments == []

    # Real speech WAV fixture with automatic language detection
    result = stt_service.transcribe_file(FIXTURE_WAV, language=None)
    assert result.model_id == DEFAULT_STT_MODEL_ID
    assert result.is_speech is True
    assert result.language == "en"
    lowered = result.text.lower()
    assert "hello" in lowered
    assert "thank" in lowered
    assert len(result.segments) >= 1
    assert 0.0 <= result.segments[0].start <= result.segments[0].end

    # Explicit locale without timestamps
    no_ts = stt_service.transcribe_file(
        FIXTURE_WAV,
        language="en-IN",
        return_timestamps=False,
    )
    assert no_ts.language == "en"
    assert no_ts.segments == []
    assert "hello" in no_ts.text.lower()


def test_whisper_tiny_rejects_excess_duration(stt_service: WhisperSpeechToText) -> None:
    too_long = np.zeros(int(16000 * (stt_service.config.max_audio_seconds + 1.0)), dtype=np.float32)
    with pytest.raises(ValueError, match="exceeds limit"):
        stt_service.transcribe_array(too_long, sample_rate=16000)


def test_whisper_tiny_longform_over_30s(stt_service: WhisperSpeechToText) -> None:
    speech, sr = sf.read(FIXTURE_WAV, dtype="float32")
    assert sr == 16000
    long_audio = np.concatenate([np.zeros(30 * sr, dtype=np.float32), speech])
    res = stt_service.transcribe_array(
        long_audio,
        sample_rate=sr,
        language="en",
        return_timestamps=True,
    )
    assert res.duration_seconds > 31.0
    assert res.is_speech is True
    assert "hello" in res.text.lower()


def test_fastapi_stt_endpoints_and_websocket(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIGNBRIDGE_STT_LOCAL_ONLY", "1")
    wav_bytes = FIXTURE_WAV.read_bytes()
    speech, sr = sf.read(FIXTURE_WAV, dtype="float32")

    ogg_buf = io.BytesIO()
    sf.write(ogg_buf, speech, sr, format="OGG", subtype="OPUS")
    ogg_bytes = ogg_buf.getvalue()

    pcm16_bytes = (np.clip(speech, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()

    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["stt"]["model_id"] == DEFAULT_STT_MODEL_ID

        stt_health = client.get("/api/stt/health")
        assert stt_health.status_code == 200
        stt_info = stt_health.json()
        assert stt_info["model_id"] == DEFAULT_STT_MODEL_ID
        assert stt_info["signbridge_languages"] == ["en", "hi", "kn"]

        # Multipart WAV transcription
        resp_wav = client.post(
            "/api/stt/transcribe",
            files={"file": ("hello.wav", wav_bytes, "audio/wav")},
        )
        assert resp_wav.status_code == 200
        wav_json = resp_wav.json()
        assert wav_json["model_id"] == DEFAULT_STT_MODEL_ID
        assert wav_json["language"] == "en"
        assert "hello" in wav_json["text"].lower()
        assert "thank" in wav_json["text"].lower()

        # Multipart OGG/Opus transcription with explicit locale
        resp_ogg = client.post(
            "/api/stt/transcribe?language=en-IN",
            files={"file": ("speech.ogg", ogg_bytes, "audio/ogg")},
        )
        assert resp_ogg.status_code == 200
        assert resp_ogg.json()["language"] == "en"
        assert "hello" in resp_ogg.json()["text"].lower()

        # Error cases
        assert (
            client.post(
                "/api/stt/transcribe",
                files={"file": ("empty.wav", b"", "audio/wav")},
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/api/stt/transcribe",
                files={"file": ("bad.wav", b"not-audio", "audio/wav")},
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/api/stt/transcribe",
                files={"file": ("hello.wav", wav_bytes, "audio/wav")},
                data={"language": "invalid-lang"},
            ).status_code
            == 400
        )

        # WebSocket utterance mode (container bytes)
        with client.websocket_connect("/ws/stt?language=en") as ws:
            ws.send_bytes(wav_bytes)
            msg = ws.receive_json()
            assert msg["type"] == "transcription"
            assert msg["language"] == "en"
            assert "hello" in msg["text"].lower()

        # WebSocket buffered PCM streaming mode
        with client.websocket_connect(
            "/ws/stt?audio_format=pcm_s16le&sample_rate=16000&channels=1"
        ) as ws:
            ws.send_json({"type": "ping"})
            assert ws.receive_json() == {"type": "pong"}

            ws.send_json({"type": "config", "language": "en-IN"})
            cfg_ack = ws.receive_json()
            assert cfg_ack["type"] == "configured"
            assert cfg_ack["language"] == "en"

            midpoint = len(pcm16_bytes) // 2
            midpoint -= midpoint % 2
            ws.send_bytes(pcm16_bytes[:midpoint])
            ws.send_bytes(pcm16_bytes[midpoint:])
            ws.send_json({"type": "flush"})
            flush_reply = ws.receive_json()
            assert flush_reply["type"] == "transcription"
            assert "hello" in flush_reply["text"].lower()

            # Buffer cleared after flush -> flushing again returns clean error
            ws.send_json({"type": "flush"})
            err_reply = ws.receive_json()
            assert err_reply["type"] == "error"

            ws.send_json({"type": "reset"})
            assert ws.receive_json() == {"type": "reset", "buffered_bytes": 0}
