"""Tests for SignBridge MediaPipe GestureRecognizer backend."""

from types import SimpleNamespace

import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest

from signbridge.config import (
    DEFAULT_DYNAMIC_CLASSIFIER_PATH,
    RecognizerConfig,
    translate_gesture_label,
)
from signbridge.dynamic_recognizer import (
    DynamicSignClassifier,
    TemporalSignTracker,
    extract_kinematic_frame,
    train_dynamic_classifier,
)
from signbridge.recognizer import (
    SignGestureRecognizer,
    build_prediction,
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
        assert "dynamic_classifier" in data
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
            assert reply["motion_state"] == "idle"
            assert reply["recognition_source"] == "none"
            assert reply["holistic"]["enabled"] is True
            assert "non_manual_markers" in reply["spatial_grammar"]

        with client.websocket_connect("/ws/recognize?enable_holistic=false") as ws:
            ws.send_bytes(jpeg_bytes)
            reply = ws.receive_json()
            assert reply["holistic"]["enabled"] is False


def test_merge_holistic_hands_deduplicates_overlapping_wrists() -> None:
    def _hand_lms(wx: float, wy: float) -> list[SimpleNamespace]:
        pts = [SimpleNamespace(x=wx, y=wy, z=0.0) for _ in range(21)]
        pts[9] = SimpleNamespace(x=wx, y=wy - 0.10, z=0.0)
        return pts

    pose = [SimpleNamespace(x=0.5, y=0.25, z=0.0, visibility=1.0) for _ in range(33)]
    pose[11] = SimpleNamespace(x=0.62, y=0.48, z=0.0, visibility=1.0)
    pose[12] = SimpleNamespace(x=0.38, y=0.48, z=0.0, visibility=1.0)

    # GestureRecognizer detected Right hand at (0.35, 0.52); Holistic duplicated nearly same wrist under Left!
    g_res = SimpleNamespace(
        hand_landmarks=[_hand_lms(0.35, 0.52)],
        hand_world_landmarks=[],
        handedness=[[SimpleNamespace(category_name="Right", display_name="Right", score=0.9)]],
        gestures=[[SimpleNamespace(category_name="Open_Palm", display_name="Open_Palm", score=0.8)]],
    )
    h_overlap = SimpleNamespace(
        pose_landmarks=pose,
        pose_world_landmarks=[],
        face_landmarks=[],
        face_blendshapes=[],
        left_hand_landmarks=_hand_lms(0.36, 0.53),
        left_hand_world_landmarks=[],
        right_hand_landmarks=[],
        right_hand_world_landmarks=[],
    )
    pred_dedup = build_prediction(
        g_res,
        timestamp_ms=100,
        config=RecognizerConfig(enable_custom_classifier=False, enable_dynamic_recognizer=False),
        stabilizer=GestureStabilizer(),
        immediate=True,
        holistic_result=h_overlap,
    )
    assert len(pred_dedup.hands) == 1

    # Distant second Left hand at (0.70, 0.55) SHOULD be merged cleanly.
    h_two = SimpleNamespace(
        pose_landmarks=pose,
        pose_world_landmarks=[],
        face_landmarks=[],
        face_blendshapes=[],
        left_hand_landmarks=_hand_lms(0.70, 0.55),
        left_hand_world_landmarks=[],
        right_hand_landmarks=[],
        right_hand_world_landmarks=[],
    )
    pred_two = build_prediction(
        g_res,
        timestamp_ms=120,
        config=RecognizerConfig(enable_custom_classifier=False, enable_dynamic_recognizer=False),
        stabilizer=GestureStabilizer(),
        immediate=True,
        holistic_result=h_two,
    )
    assert len(pred_two.hands) == 2


def test_realtime_dynamic_moving_sign_streaming_and_idle_rejection(tmp_path) -> None:
    from signbridge.avatar_dataset import (
        DEFAULT_SIGN_ANIMATIONS_PATH,
        _fk_hand_landmarks,
        _fk_pose_landmarks,
        _sample_track_pose,
        load_avatar_pose_library,
    )

    if not DEFAULT_SIGN_ANIMATIONS_PATH.is_file():
        pytest.skip("signAnimations.js not found.")

    dyn_path = (
        DEFAULT_DYNAMIC_CLASSIFIER_PATH
        if DEFAULT_DYNAMIC_CLASSIFIER_PATH.is_file()
        else tmp_path / "dyn_test.pt"
    )
    if not dyn_path.is_file():
        train_dynamic_classifier(output_path=dyn_path, epochs=25)

    lib = load_avatar_pose_library(DEFAULT_SIGN_ANIMATIONS_PATH)
    dyn_clf = DynamicSignClassifier(dyn_path)
    assert dyn_clf.is_loaded
    tracker = TemporalSignTracker(dyn_clf, score_threshold=0.60, min_similarity=0.48)
    pose = _fk_pose_landmarks(0.0)

    for track_id, expected in (("goodbye", "Goodbye"), ("thank_you", "Thank You"), ("water", "Water")):
        tracker.reset()
        track = lib["tracks"][track_id]
        fired = None
        for step in range(15):
            ts = 1000 + step * 75
            t_eval = min(1.0, step / 12.0)
            ps = _sample_track_pose(track, t_eval, lib["handshapes"], lib["idle"])
            sp = ps["right"]
            hands = [
                SimpleNamespace(
                    handedness="Right",
                    landmarks=_fk_hand_landmarks(sp["wrist"], sp["palm"], sp["finger"], sp["joints"], -1),
                )
            ]
            kf = extract_kinematic_frame(hands, pose_landmarks=pose)
            trig, _, _ = tracker.update(ts, kf)
            if trig is not None:
                fired = trig.label
                break
        assert fired == expected

    # Relaxed low-chest wandering idle motion must not fire a dynamic sign.
    tracker.reset()
    false_fired = None
    for step in range(14):
        u = step / 13.0
        hands = [
            SimpleNamespace(
                handedness="Right",
                landmarks=_fk_hand_landmarks(
                    [-0.25 + 0.10 * u, 0.98 - 0.08 * np.sin(u * 2.0), 0.18],
                    [0.4, -0.35, 0.75],
                    [0.1, -0.85, 0.25],
                    lib["handshapes"]["relaxed"],
                    -1,
                ),
            )
        ]
        trig, _, _ = tracker.update(3000 + step * 75, extract_kinematic_frame(hands, pose_landmarks=pose))
        if trig is not None:
            false_fired = trig.label
            break
    assert false_fired is None


def test_stabilizer_clears_on_missing_hands_and_fast_switches() -> None:
    stab = GestureStabilizer(
        window_size=5,
        min_votes=3,
        min_score=0.50,
        cooldown_seconds=0.8,
    )
    # 3 votes lock "Book"
    for i in range(2):
        st = stab.update("Book", 0.85, now=1.0 + i * 0.1)
        assert st.locked_label is None
    st = stab.update("Book", 0.85, now=1.2)
    assert st.locked_label == "Book"

    # Two consecutive missing/None frames immediately clear stale candidate history.
    st_miss1 = stab.update(None, 0.0, now=1.3)
    assert st_miss1.candidate_label is None or st_miss1.candidate_label == "Book"
    st_miss2 = stab.update("None", 0.0, now=1.4)
    assert st_miss2.candidate_label is None
    assert st_miss2.progress == 0.0

    # Switching signs without dropping hands flushes previous sign votes after 2 frames.
    stab.update("Book", 0.82, now=2.5)
    stab.update("Book", 0.82, now=2.6)
    stab.update("Please", 0.84, now=2.7)
    st_switch = stab.update("Please", 0.84, now=2.8)
    assert st_switch.candidate_label == "Please"
    st_lock = stab.update("Please", 0.84, now=2.9)
    assert st_lock.locked_label == "Please"
