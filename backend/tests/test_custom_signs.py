"""Tests for custom ISL landmark dataset, PyTorch classifier, and dataset API."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest

from signbridge.avatar_dataset import (
    DEFAULT_SIGN_ANIMATIONS_PATH,
    _fk_hand_landmarks,
    _fk_pose_landmarks,
    _sample_track_pose,
    load_avatar_pose_library,
    sync_voice_to_sign_dataset,
    track_id_to_label,
)
from signbridge.config import (
    DEFAULT_CUSTOM_CLASSIFIER_PATH,
    DEFAULT_DATASET_DIR,
    RecognizerConfig,
    translate_gesture_label,
)
from signbridge.custom_classifier import (
    ENRICHED_FEATURE_DIM,
    FEATURE_DIM,
    CustomSignClassifier,
    enrich_sign_features,
    extract_sign_features,
    mirror_feature_vector,
    train_landmark_classifier,
)
from signbridge.dynamic_recognizer import DynamicSignClassifier
from signbridge.recognizer import SignGestureRecognizer, _select_static_custom_predictions
from signbridge.server import app
from signbridge.sign_dataset import SignDatasetStore
from signbridge.sign_planner import text_to_sign_plan


def _synthetic_hand(
    handedness: str,
    wrist_xy: tuple[float, float],
    tip_scale: float = 0.14,
) -> SimpleNamespace:
    wx, wy = wrist_xy
    pts = []
    for idx in range(21):
        angle = (idx / 21.0) * 1.4
        dist = 0.02 + (idx % 5) * (tip_scale / 4.0)
        pts.append(
            SimpleNamespace(
                x=wx + dist * np.cos(angle),
                y=wy - dist * np.sin(angle),
                z=0.0,
            )
        )
    return SimpleNamespace(handedness=handedness, landmarks=pts)


def test_extract_sign_features_and_pytorch_train_predict(tmp_path: Path) -> None:
    pose = [SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=1.0) for _ in range(33)]
    pose[11] = SimpleNamespace(x=0.65, y=0.6, z=0.0, visibility=1.0)
    pose[12] = SimpleNamespace(x=0.35, y=0.6, z=0.0, visibility=1.0)

    help_vec = extract_sign_features(
        [_synthetic_hand("Left", (0.48, 0.55), 0.12), _synthetic_hand("Right", (0.52, 0.48), 0.06)],
        pose_landmarks=pose,
    )
    hug_vec = extract_sign_features(
        [_synthetic_hand("Left", (0.6, 0.45), 0.05), _synthetic_hand("Right", (0.4, 0.45), 0.05)],
        pose_landmarks=pose,
    )
    assert help_vec.shape == (FEATURE_DIM,)
    assert hug_vec.shape == (FEATURE_DIM,)

    ckpt_path = tmp_path / "custom_clf.pt"
    metrics = train_landmark_classifier(
        {"Help": [help_vec], "Hug": [hug_vec]},
        output_path=ckpt_path,
        epochs=25,
    )
    assert metrics["num_classes"] == 2
    assert metrics["train_accuracy"] == 1.0

    clf = CustomSignClassifier(ckpt_path)
    assert clf.is_loaded
    preds_help = clf.predict(help_vec, top_k=2)
    assert preds_help[0].label == "Help"
    assert preds_help[0].score > 0.7

    preds_hug = clf.predict(hug_vec, top_k=2)
    assert preds_hug[0].label == "Hug"
    assert preds_hug[0].score > 0.7


def test_custom_translations_and_sign_planner() -> None:
    for word in ("Change", "Correct", "Goodbye", "Home", "Hug", "Meet", "Touch", "Please", "Sorry"):
        tr = translate_gesture_label(word)
        assert tr["en"] == word
        assert tr["hi"]
        assert tr["kn"]

    plan = text_to_sign_plan("Please meet me at home")
    glosses = [step.gloss for step in plan.steps]
    assert "PLEASE" in glosses
    assert "MEET" in glosses
    assert "HOME" in glosses


def test_bootstrapped_dataset_and_live_classifier() -> None:
    if not DEFAULT_DATASET_DIR.is_dir() or not DEFAULT_CUSTOM_CLASSIFIER_PATH.is_file():
        pytest.skip("Bootstrapped dataset artifacts not generated.")

    store = SignDatasetStore(DEFAULT_DATASET_DIR)
    summary = store.get_summary()
    assert summary["total_samples"] >= 79
    assert summary["total_classes"] >= 46
    assert summary["source_totals"].get("image2_labeled", 0) >= 15
    assert summary["source_totals"].get("image1_inferred", 0) >= 64

    clf = CustomSignClassifier(DEFAULT_CUSTOM_CLASSIFIER_PATH)
    assert clf.is_loaded
    assert clf.status()["num_classes"] >= 46

    # Verify SignGestureRecognizer recognizes a stored sample tile when custom classifier is active.
    sample_img = DEFAULT_DATASET_DIR / "images" / "Meet" / "img2_meet.jpg"
    bgr = cv2.imread(str(sample_img))
    assert bgr is not None
    bgr_up = cv2.copyMakeBorder(
        cv2.resize(bgr, (bgr.shape[1] * 3, bgr.shape[0] * 3), interpolation=cv2.INTER_CUBIC),
        40,
        40,
        40,
        40,
        cv2.BORDER_REPLICATE,
    )
    cfg = RecognizerConfig(
        running_mode="IMAGE",
        enable_holistic=True,
        enable_custom_classifier=True,
        min_hand_detection_confidence=0.15,
        min_hand_presence_confidence=0.15,
        min_hand_landmarks_confidence=0.15,
    )
    with SignGestureRecognizer(cfg) as rec:
        pred = rec.process_bgr(bgr_up)
        assert pred.locked_label == "Meet"
        assert pred.phrases["en"] == "Meet"


def test_dataset_api_crud_import_and_train(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_store = SignDatasetStore(tmp_path / "ds")
    tmp_clf = CustomSignClassifier(tmp_path / "clf.pt")

    # Prepare two synthetic folder images using real extracted sample images if present.
    ref_change = DEFAULT_DATASET_DIR / "images" / "Change" / "img2_change.jpg"
    ref_home = DEFAULT_DATASET_DIR / "images" / "Home" / "img2_home.jpg"
    if not ref_change.is_file() or not ref_home.is_file():
        pytest.skip("Reference sample images not found.")

    import_root = tmp_path / "import_src"
    (import_root / "Change").mkdir(parents=True)
    (import_root / "Home").mkdir(parents=True)
    (import_root / "Change" / "c1.jpg").write_bytes(ref_change.read_bytes())
    (import_root / "Home" / "h1.jpg").write_bytes(ref_home.read_bytes())

    with TestClient(app) as client:
        monkeypatch.setattr(app.state, "dataset_store", tmp_store, raising=False)
        monkeypatch.setattr(app.state, "custom_classifier", tmp_clf, raising=False)

        overview = client.get("/api/dataset")
        assert overview.status_code == 200
        assert overview.json()["total_samples"] == 0

        imp = client.post(
            "/api/dataset/import-folder",
            json={"folder_path": str(import_root), "max_per_class": 5},
        )
        assert imp.status_code == 200
        assert imp.json()["imported"] == 2

        upload = client.post(
            "/api/dataset/samples",
            files={"file": ("c2.jpg", ref_change.read_bytes(), "image/jpeg")},
            data={"label": "Change", "source": "webcam"},
        )
        assert upload.status_code == 200
        sid = upload.json()["sample"]["sample_id"]

        patched = client.patch(
            f"/api/dataset/samples/{sid}",
            json={"label": "Switch", "needs_review": False},
        )
        assert patched.status_code == 200
        assert patched.json()["sample"]["label"] == "Switch"

        trained = client.post("/api/dataset/train", json={"epochs": 15})
        assert trained.status_code == 200
        train_json = trained.json()
        assert train_json["status"] == "trained"
        assert train_json["num_classes"] == 3
        assert train_json["classifier"]["loaded"] is True

        deleted = client.delete(f"/api/dataset/samples/{sid}")
        assert deleted.status_code == 200
        assert deleted.json()["deleted"] is True

        signs_resp = client.get("/api/signs")
        assert signs_resp.status_code == 200
        assert "Change" in signs_resp.json()["custom_signs"]


def _build_fk_sample_for_track(lib: dict, track_id: str, t_val: float = 0.55) -> tuple[np.ndarray, int]:
    track = lib["tracks"][track_id]
    pose_state = _sample_track_pose(track, t_val, lib["handshapes"], lib["idle"])
    pose_lms = _fk_pose_landmarks(pose_state.get("torsoYaw", 0.0))
    hands = []
    if "right" in pose_state:
        r = pose_state["right"]
        hands.append(
            SimpleNamespace(
                handedness="Right",
                landmarks=_fk_hand_landmarks(r["wrist"], r["palm"], r["finger"], r["joints"], -1),
            )
        )
    if "left" in pose_state:
        l = pose_state["left"]
        hands.append(
            SimpleNamespace(
                handedness="Left",
                landmarks=_fk_hand_landmarks(l["wrist"], l["palm"], l["finger"], l["joints"], 1),
            )
        )
    return extract_sign_features(hands, pose_landmarks=pose_lms), len(hands)


def _build_fk_sample_for_char(lib: dict, ch: str) -> np.ndarray:
    spec = lib["letters"][ch]
    joints = lib["handshapes"].get(spec["shape"]) or lib["handshapes"]["relaxed"]
    pose_lms = _fk_pose_landmarks(0.0)
    rh = SimpleNamespace(
        handedness="Right",
        landmarks=_fk_hand_landmarks([-0.16, 1.32, 0.30], spec["palm"], spec["finger"], joints, -1),
    )
    return extract_sign_features([rh], pose_landmarks=pose_lms)


def test_mirror_feature_vector_swaps_handedness() -> None:
    pose = [SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=1.0) for _ in range(33)]
    pose[11] = SimpleNamespace(x=0.65, y=0.6, z=0.0, visibility=1.0)
    pose[12] = SimpleNamespace(x=0.35, y=0.6, z=0.0, visibility=1.0)

    vec = extract_sign_features([_synthetic_hand("Right", (0.28, 0.45), 0.14)], pose_landmarks=pose)
    assert np.allclose(vec[126:128], [0.0, 1.0])

    mirrored = mirror_feature_vector(vec)
    assert mirrored.shape == (FEATURE_DIM,)
    assert np.allclose(mirrored[126:128], [1.0, 0.0])
    assert np.allclose(mirrored[63:126], 0.0)
    assert not np.allclose(mirrored[0:63], 0.0)
    assert np.allclose(mirror_feature_vector(mirrored), vec, atol=1e-5)


def test_sync_voice_to_sign_dataset_isolated(tmp_path: Path) -> None:
    if not DEFAULT_SIGN_ANIMATIONS_PATH.is_file():
        pytest.skip("signAnimations.js not found.")

    tmp_store = SignDatasetStore(tmp_path / "ds")
    tmp_clf_path = tmp_path / "voice_sign_clf.pt"
    res = sync_voice_to_sign_dataset(
        store=tmp_store,
        model_path=tmp_clf_path,
        epochs=30,
    )
    assert res["added_samples"] > 0
    for expected in ("Water", "Good Morning", "Sleep", "Wait", "Danger", "0", "7", "8"):
        assert expected in res["added_classes"]
    assert res["summary"]["total_classes"] >= 80
    assert res["train"]["train_accuracy"] >= 0.85

    clf = CustomSignClassifier(tmp_clf_path)
    assert clf.is_loaded
    lib = load_avatar_pose_library(DEFAULT_SIGN_ANIMATIONS_PATH)

    for track_id, expected_label in (("water", "Water"), ("sleep", "Sleep"), ("good_morning", "Good Morning")):
        vec, num_hands = _build_fk_sample_for_track(lib, track_id)
        assert clf.predict(vec, top_k=1)[0].label == expected_label
        if num_hands == 1:
            assert clf.predict(mirror_feature_vector(vec), top_k=1)[0].label == expected_label

    for ch in ("0", "7", "8"):
        vec = _build_fk_sample_for_char(lib, ch)
        assert clf.predict(vec, top_k=1)[0].label == ch
        assert clf.predict(mirror_feature_vector(vec), top_k=1)[0].label == ch


def test_sync_voice_signs_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not DEFAULT_SIGN_ANIMATIONS_PATH.is_file():
        pytest.skip("signAnimations.js not found.")

    tmp_store = SignDatasetStore(tmp_path / "ds")
    tmp_clf = CustomSignClassifier(tmp_path / "clf.pt")
    with TestClient(app) as client:
        monkeypatch.setattr(app.state, "dataset_store", tmp_store, raising=False)
        monkeypatch.setattr(app.state, "custom_classifier", tmp_clf, raising=False)

        resp = client.post("/api/dataset/sync-voice-signs", json={"epochs": 20})
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["added_samples"] > 0
        assert "Water" in payload["added_classes"]
        assert payload["classifier"]["loaded"] is True

        signs_resp = client.get("/api/signs")
        assert signs_resp.status_code == 200
        assert "Water" in signs_resp.json()["custom_signs"]
        assert "Good Morning" in signs_resp.json()["custom_signs"]


def test_bootstrapped_voice_to_sign_coverage() -> None:
    if not DEFAULT_DATASET_DIR.is_dir() or not DEFAULT_CUSTOM_CLASSIFIER_PATH.is_file():
        pytest.skip("Bootstrapped dataset artifacts not generated.")

    store = SignDatasetStore(DEFAULT_DATASET_DIR)
    summary = store.get_summary()
    by_label_lower = {c["label"].lower(): c for c in summary["classes"]}

    signs_json = DEFAULT_SIGN_ANIMATIONS_PATH.parents[1] / "data" / "signs.json"
    phrases = json.loads(signs_json.read_text(encoding="utf-8")).get("speechToSignPhrases", [])
    for phrase in phrases:
        pid = phrase["id"]
        if pid == "give_a_to_b":
            assert "give" in by_label_lower
        elif pid == "what_name":
            assert all(tok in by_label_lower for tok in ("what", "your", "name"))
        else:
            assert track_id_to_label(pid).lower() in by_label_lower, f"Missing phrase sign: {pid}"

    expected_new_tracks = (
        "water",
        "good_morning",
        "sleep",
        "wait",
        "danger",
        "welcome",
        "good",
        "name",
        "who",
        "your",
        "my",
        "me",
        "come",
        "go",
        "wash",
        "work",
        "give_a_to_b",
    )
    lib = load_avatar_pose_library(DEFAULT_SIGN_ANIMATIONS_PATH)
    clf = CustomSignClassifier(DEFAULT_CUSTOM_CLASSIFIER_PATH)

    for track_id in expected_new_tracks:
        label = track_id_to_label(track_id)
        assert label.lower() in by_label_lower
        tr = translate_gesture_label(label)
        assert tr.get("hi") and tr.get("kn")
        vec, num_hands = _build_fk_sample_for_track(lib, track_id)
        assert clf.predict(vec, top_k=1)[0].label == label
        if num_hands == 1:
            assert clf.predict(mirror_feature_vector(vec), top_k=1)[0].label == label

    for ch in ("0", "7", "8"):
        assert ch.lower() in by_label_lower
        vec = _build_fk_sample_for_char(lib, ch)
        assert clf.predict(vec, top_k=1)[0].label == ch
        assert clf.predict(mirror_feature_vector(vec), top_k=1)[0].label == ch


def test_enriched_features_ood_and_offhand_fallback(tmp_path: Path) -> None:
    if not DEFAULT_SIGN_ANIMATIONS_PATH.is_file():
        pytest.skip("signAnimations.js not found.")

    lib = load_avatar_pose_library(DEFAULT_SIGN_ANIMATIONS_PATH)
    pose = _fk_pose_landmarks(0.0)
    water_vec, _ = _build_fk_sample_for_track(lib, "water")
    sleep_vec, _ = _build_fk_sample_for_track(lib, "sleep")

    enriched = enrich_sign_features(water_vec)
    assert enriched.shape == (ENRICHED_FEATURE_DIM,)

    ckpt = tmp_path / "ood_clf.pt"
    train_landmark_classifier({"Water": [water_vec], "Sleep": [sleep_vec]}, output_path=ckpt, epochs=30)
    clf = CustomSignClassifier(ckpt)

    # Empty / all-zero feature vector must yield strict zero similarity & confidence.
    zero_preds = clf.predict(np.zeros(FEATURE_DIM, dtype=np.float32), top_k=2, min_similarity=0.58)
    assert zero_preds[0].similarity == 0.0
    assert zero_preds[0].score == 0.0

    # Active Water hand paired with a resting low lap hand must still recover Water via dominant-hand fallback.
    ps_w = _sample_track_pose(lib["tracks"]["water"], 0.55, lib["handshapes"], lib["idle"])
    sp_w = ps_w["right"]
    active_hand = SimpleNamespace(
        handedness="Right",
        spatial_zone="center",
        height_level="head",
        wrist_xy=(0.40, 0.34),
        landmarks=_fk_hand_landmarks(sp_w["wrist"], sp_w["palm"], sp_w["finger"], sp_w["joints"], -1),
    )
    resting_hand = SimpleNamespace(
        handedness="Left",
        spatial_zone="left",
        height_level="low",
        wrist_xy=(0.75, 0.88),
        landmarks=_fk_hand_landmarks([0.30, 0.96, 0.16], [-0.7, 0.1, 0.7], [-0.12, -0.95, 0.25], lib["handshapes"]["relaxed"], 1),
    )
    preds = _select_static_custom_predictions(
        [active_hand, resting_hand],
        pose,
        clf,
        RecognizerConfig(custom_classifier_threshold=0.62, custom_classifier_min_similarity=0.58),
    )
    assert preds and preds[0].label == "Water"
    assert preds[0].score >= 0.62


def test_dataset_sequences_api_crud_and_train(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ref_goodbye = DEFAULT_DATASET_DIR / "images" / "Goodbye" / "img2_goodbye.jpg"
    if not ref_goodbye.is_file():
        pytest.skip("Reference Goodbye image not found.")
    img = cv2.imread(str(ref_goodbye))
    assert img is not None
    # Upscale & border-pad so MediaPipe reliably detects hand on all frames of synthetic clip.
    big = cv2.copyMakeBorder(cv2.resize(img, (320, 320)), 40, 40, 40, 40, cv2.BORDER_REPLICATE)
    ok, enc = cv2.imencode(".jpg", big)
    assert ok
    frame_bytes = enc.tobytes()

    tmp_store = SignDatasetStore(tmp_path / "ds")
    tmp_clf = CustomSignClassifier(tmp_path / "clf.pt")
    tmp_dyn = DynamicSignClassifier(tmp_path / "dyn.pt")

    with TestClient(app) as client:
        monkeypatch.setattr(app.state, "dataset_store", tmp_store, raising=False)
        monkeypatch.setattr(app.state, "custom_classifier", tmp_clf, raising=False)
        monkeypatch.setattr(app.state, "dynamic_classifier", tmp_dyn, raising=False)

        files = [("files", (f"f{i}.jpg", frame_bytes, "image/jpeg")) for i in range(6)]
        resp = client.post(
            "/api/dataset/sequences",
            files=files,
            data={"label": "Goodbye", "timestamps_ms": "0,120,240,360,480,600"},
        )
        assert resp.status_code == 200
        seq_id = resp.json()["sequence"]["sequence_id"]
        assert resp.json()["sequence"]["num_frames"] == 6

        listed = client.get("/api/dataset/sequences")
        assert listed.status_code == 200
        assert any(s["sequence_id"] == seq_id for s in listed.json()["sequences"])

        # Add one static sample and trigger /api/dataset/train (trains both static & dynamic!).
        up = client.post(
            "/api/dataset/samples",
            files={"file": ("g0.jpg", frame_bytes, "image/jpeg")},
            data={"label": "Goodbye", "source": "webcam"},
        )
        assert up.status_code == 200
        tr = client.post("/api/dataset/train", json={"epochs": 15})
        assert tr.status_code == 200
        assert tr.json()["dynamic_classifier"]["loaded"] is True
        assert tr.json()["dynamic_classifier"]["num_classes"] >= 40

        deleted = client.delete(f"/api/dataset/sequences/{seq_id}")
        assert deleted.status_code == 200
        assert deleted.json()["deleted"] is True


def test_aspect_ratio_invariant_feature_and_kinematic_extraction() -> None:
    from signbridge.avatar_dataset import _fk_hand_landmarks, _fk_pose_landmarks
    from signbridge.dynamic_recognizer import extract_kinematic_frame

    pose_sq = _fk_pose_landmarks(0.0)
    joints = {
        "thumb": [0.1, 0.0, 0.2, 0.15],
        "index": [0.05, 0.0, 0.0, 0.0],
        "middle": [0.05, 0.0, 0.0, 0.0],
        "ring": [1.3, 1.2, 0.9, 0.0],
        "pinky": [1.3, 1.2, 0.9, 0.0],
    }
    lms_sq = _fk_hand_landmarks(
        [-0.25, 0.25, 0.18],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        joints,
        -1,
    )
    ar = 16.0 / 9.0

    def _squish_lms(lms: list[SimpleNamespace]) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                x=0.5 + (p.x - 0.5) / ar,
                y=p.y,
                z=p.z / ar,
                visibility=getattr(p, "visibility", 1.0),
            )
            for p in lms
        ]

    hand_sq = SimpleNamespace(handedness="Right", landmarks=lms_sq)
    hand_wide = SimpleNamespace(handedness="Right", landmarks=_squish_lms(lms_sq))
    pose_wide = _squish_lms(pose_sq)

    feat_sq = extract_sign_features([hand_sq], pose_landmarks=pose_sq, aspect_ratio=1.0)
    feat_wide = extract_sign_features([hand_wide], pose_landmarks=pose_wide, aspect_ratio=ar)
    assert np.allclose(feat_sq, feat_wide, atol=1e-4)

    kf_sq = extract_kinematic_frame([hand_sq], pose_landmarks=pose_sq, aspect_ratio=1.0)
    kf_wide = extract_kinematic_frame([hand_wide], pose_landmarks=pose_wide, aspect_ratio=ar)
    assert np.allclose(kf_sq, kf_wide, atol=1e-4)


def test_real_human_goodbye_in_640x480_video_and_no_stale_output() -> None:
    from signbridge.recognizer import SignGestureRecognizer

    ref_goodbye = DEFAULT_DATASET_DIR / "images" / "Goodbye" / "img2_goodbye.jpg"
    if not ref_goodbye.is_file() or not DEFAULT_CUSTOM_CLASSIFIER_PATH.is_file():
        pytest.skip("Trained custom model or Goodbye photo not found.")

    img = cv2.imread(str(ref_goodbye))
    assert img is not None
    # Simulate placing a tall portrait crop in a 640x480 webcam frame.
    scene = np.full((480, 640, 3), 235, dtype=np.uint8)
    h, w = img.shape[:2]
    scale = 360.0 / float(h)
    nw, nh = int(w * scale), 360
    resized = cv2.resize(img, (nw, nh))
    x0 = (640 - nw) // 2
    y0 = (480 - nh) // 2
    scene[y0 : y0 + nh, x0 : x0 + nw] = resized

    blank = np.full((480, 640, 3), 20, dtype=np.uint8)
    locked = None
    with SignGestureRecognizer(RecognizerConfig(running_mode="VIDEO")) as rec:
        for i in range(6):
            pred = rec.process_bgr(scene, timestamp_ms=100 + i * 100)
            if pred.locked_label:
                locked = pred.locked_label
        assert locked == "Goodbye"

        # Two blank frames must immediately flush stale candidate votes.
        rec.process_bgr(blank, timestamp_ms=800)
        after_blank = rec.process_bgr(blank, timestamp_ms=900)
        assert after_blank.hands == []
        assert after_blank.candidate_label is None
        assert after_blank.stability_progress == 0.0


def test_duplicate_handedness_slot_resolution() -> None:
    from signbridge.avatar_dataset import _fk_hand_landmarks, _fk_pose_landmarks

    pose = _fk_pose_landmarks(0.0)
    joints = {
        "thumb": [0.1, 0.0, 0.2, 0.15],
        "index": [0.05, 0.0, 0.0, 0.0],
        "middle": [0.05, 0.0, 0.0, 0.0],
        "ring": [1.3, 1.2, 0.9, 0.0],
        "pinky": [1.3, 1.2, 0.9, 0.0],
    }
    h_right_side = SimpleNamespace(
        handedness="Right",
        landmarks=_fk_hand_landmarks([-0.22, 0.28, 0.18], [0, 0, 1], [0, -1, 0], joints, -1),
    )
    h_left_side = SimpleNamespace(
        handedness="Left",
        landmarks=_fk_hand_landmarks([0.22, 0.28, 0.18], [0, 0, 1], [0, -1, 0], joints, 1),
    )
    f_distinct = extract_sign_features([h_left_side, h_right_side], pose_landmarks=pose)

    # Even if MediaPipe reports both hands as "Right" in reversed list order,
    # anatomical wrist x resolves Left vs Right into identical slots.
    h_dup_r1 = SimpleNamespace(handedness="Right", landmarks=h_right_side.landmarks)
    h_dup_r2 = SimpleNamespace(handedness="Right", landmarks=h_left_side.landmarks)
    f_dup = extract_sign_features([h_dup_r1, h_dup_r2], pose_landmarks=pose)
    assert np.allclose(f_distinct, f_dup, atol=1e-5)
