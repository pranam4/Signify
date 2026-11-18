"""Tests for custom ISL landmark dataset, PyTorch classifier, and dataset API."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest

from signbridge.config import (
    DEFAULT_CUSTOM_CLASSIFIER_PATH,
    DEFAULT_DATASET_DIR,
    RecognizerConfig,
    translate_gesture_label,
)
from signbridge.custom_classifier import (
    FEATURE_DIM,
    CustomSignClassifier,
    extract_sign_features,
    train_landmark_classifier,
)
from signbridge.recognizer import SignGestureRecognizer
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
