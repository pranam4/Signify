"""Tests for SignBridge MediaPipe GestureRecognizer backend."""

from types import SimpleNamespace

import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest

from signbridge.config import RecognizerConfig, translate_gesture_label
from signbridge.recognizer import (
    SignGestureRecognizer,
    classify_spatial_zone,
    extract_observations,
)
from signbridge.server import app
from signbridge.stabilizer import GestureStabilizer


def test_spatial_zone_classification() -> None:
    assert classify_spatial_zone(0.15) == "left"
    assert classify_spatial_zone(0.50) == "center"
    assert classify_spatial_zone(0.85) == "right"


def test_extract_observations_invert_handedness() -> None:
    lm = SimpleNamespace(x=0.25, y=0.5, z=0.0)
    fake_result = SimpleNamespace(
        hand_landmarks=[[lm] * 21, [lm] * 21, [lm] * 21],
        hand_world_landmarks=None,
        handedness=[
            [SimpleNamespace(category_name="Left", display_name="Left", score=0.95)],
            [SimpleNamespace(category_name="Right", display_name="Right", score=0.92)],
            [SimpleNamespace(category_name="Unknown", display_name="Unknown", score=0.5)],
        ],
        gestures=[
            [SimpleNamespace(category_name="Open_Palm", display_name="Open_Palm", score=0.9)],
            [SimpleNamespace(category_name="Thumb_Up", display_name="Thumb_Up", score=0.88)],
            [],
        ],
    )

    normal = extract_observations(fake_result, RecognizerConfig(invert_handedness=False))
    assert [h.handedness for h in normal] == ["Left", "Right", "Unknown"]

    inverted = extract_observations(fake_result, RecognizerConfig(invert_handedness=True))
    assert [h.handedness for h in inverted] == ["Right", "Left", "Unknown"]


def test_translate_gesture_label_stock_and_custom() -> None:
    assert translate_gesture_label("ILoveYou")["en"] == "I love you"
    assert translate_gesture_label("None") == {}
    assert translate_gesture_label(None) == {}

    custom = translate_gesture_label("thank_you")
    assert custom["kind"] == "custom_sign"
    assert custom["en"] == "Thank you"

    camel = translate_gesture_label("GoodMorning")
    assert camel["en"] == "Good morning"


def test_gesture_stabilizer_majority_and_cooldown() -> None:
    stab = GestureStabilizer(
        window_size=4,
        min_votes=3,
        min_score=0.6,
        cooldown_seconds=2.0,
    )

    # First two frames build progress without locking
    assert stab.update("Open_Palm", 0.9, now=1.0).locked_label is None
    second = stab.update("Open_Palm", 0.9, now=1.1)
    assert second.locked_label is None
    assert second.progress == 2 / 3

    # Third vote reaches min_votes and emits locked event
    third = stab.update("Open_Palm", 0.9, now=1.2)
    assert third.locked_label == "Open_Palm"

    # Holding during cooldown should not re-fire
    fourth = stab.update("Open_Palm", 0.9, now=1.5)
    assert fourth.locked_label is None

    # After cooldown expires, same gesture can fire again
    after_cd = stab.update("Open_Palm", 0.9, now=3.5)
    assert after_cd.locked_label == "Open_Palm"


def test_classifier_options_mutual_exclusion() -> None:
    bad_cfg = RecognizerConfig(
        canned_category_allowlist=["Thumb_Up"],
        canned_category_denylist=["Closed_Fist"],
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        SignGestureRecognizer(bad_cfg)


def test_mediapipe_recognizer_initializes_and_processes_frame() -> None:
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    with SignGestureRecognizer(RecognizerConfig()) as recognizer:
        pred = recognizer.process_rgb(blank, timestamp_ms=100)
        assert pred.timestamp_ms == 100
        assert pred.hands == []
        assert pred.locked_label is None
        payload = pred.to_ws_dict()
        assert payload["hands"] == []
        assert payload["stability_progress"] == 0.0
        assert payload["holistic"]["enabled"] is True
        assert payload["spatial_grammar"]["signing_space"]["source"] == "frame_fallback"

    with SignGestureRecognizer(RecognizerConfig(enable_holistic=False)) as hand_only:
        pred = hand_only.process_rgb(blank, timestamp_ms=100)
        payload = pred.to_ws_dict()
        assert payload["holistic"]["enabled"] is False
        assert "spatial_grammar" not in payload


def test_image_and_live_stream_running_modes() -> None:
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    img_cfg = RecognizerConfig(
        running_mode="IMAGE",
        canned_score_threshold=0.4,
        canned_max_results=3,
    )
    with SignGestureRecognizer(img_cfg) as img_rec:
        pred = img_rec.process_rgb(blank, timestamp_ms=42)
        assert pred.timestamp_ms == 42
        assert pred.hands == []
        assert pred.locked_label is None

    with pytest.raises(ValueError, match="result_callback"):
        SignGestureRecognizer(RecognizerConfig(running_mode="LIVE_STREAM"))

    received = []
    live_cfg = RecognizerConfig(running_mode="LIVE_STREAM")
    with SignGestureRecognizer(live_cfg, result_callback=received.append) as live_rec:
        with pytest.raises(ValueError, match="process_async_rgb"):
            live_rec.process_rgb(blank)
        ts = live_rec.process_async_rgb(blank, timestamp_ms=200)
        assert ts == 200


def test_fastapi_health_signs_and_websocket() -> None:
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", blank)
    assert ok
    jpeg_bytes = encoded.tobytes()

    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        data = health.json()
        assert data["status"] == "ok"
        assert "ILoveYou" in data["supported_baseline_gestures"]
        assert "IMAGE" in data["running_modes"]
        assert data["holistic_enabled"] is True
        assert data["holistic_landmarks_total"] == 553
        assert "torso_relative_signing_space" in data["spatial_grammar_features"]

        signs = client.get("/api/signs")
        assert signs.status_code == 200
        assert "Pointing_Up" in signs.json()["gestures"]

        img_resp = client.post(
            "/api/recognize-image",
            files={"file": ("blank.jpg", jpeg_bytes, "image/jpeg")},
        )
        assert img_resp.status_code == 200
        img_data = img_resp.json()
        assert img_data["mode"] == "IMAGE"
        assert img_data["hands"] == []
        assert img_data["holistic"]["enabled"] is True
        assert "signing_space" in img_data["spatial_grammar"]

        img_no_hol_resp = client.post(
            "/api/recognize-image?enable_holistic=false",
            files={"file": ("blank.jpg", jpeg_bytes, "image/jpeg")},
        )
        assert img_no_hol_resp.status_code == 200
        assert img_no_hol_resp.json()["holistic"]["enabled"] is False

        bad_resp = client.post(
            "/api/recognize-image",
            files={"file": ("corrupt.jpg", b"not-an-image", "image/jpeg")},
        )
        assert bad_resp.status_code == 400

        with client.websocket_connect("/ws/recognize") as ws:
            ws.send_bytes(jpeg_bytes)
            reply = ws.receive_json()
            assert reply["hands"] == []
            assert reply["locked_label"] is None
            assert reply["spatial_zone"] == "center"
            assert reply["holistic"]["enabled"] is True
            assert "non_manual_markers" in reply["spatial_grammar"]

        with client.websocket_connect("/ws/recognize?enable_holistic=false") as ws:
            ws.send_bytes(jpeg_bytes)
            reply = ws.receive_json()
            assert reply["holistic"]["enabled"] is False
