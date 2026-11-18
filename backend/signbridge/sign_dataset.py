"""Dataset management for custom ISL landmark samples and image thumbnails."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import threading
from typing import Any
import uuid

import cv2
import numpy as np

from .config import DEFAULT_DATASET_DIR, RecognizerConfig, translate_gesture_label
from .custom_classifier import extract_sign_features
from .recognizer import SignGestureRecognizer

_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def sanitize_label_dir(label: str) -> str:
    """Convert a sign label into a safe single-segment directory name."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip()).strip("._")
    if not cleaned:
        raise ValueError("Sign label must contain at least one alphanumeric character.")
    return cleaned


def normalize_sign_label(label: str) -> str:
    """Normalize user-supplied sign label whitespace while preserving casing."""
    cleaned = " ".join((label or "").strip().split())
    if not cleaned:
        raise ValueError("Sign label cannot be empty.")
    if len(cleaned) > 64:
        raise ValueError("Sign label must be 64 characters or fewer.")
    return cleaned


class SignDatasetStore:
    """Thread-safe JSON + JPEG store for extracted sign landmarks and samples."""

    def __init__(self, dataset_dir: Path = DEFAULT_DATASET_DIR) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / "images"
        self.manifest_path = self.dataset_dir / "manifest.json"
        self._lock = threading.Lock()
        self._ensure_initialized()

    def _ensure_initialized(self) -> None:
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.is_file():
            self._write_manifest_unlocked(
                {
                    "version": 1,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "samples": [],
                }
            )

    def _read_manifest_unlocked(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            return {"version": 1, "updated_at": None, "samples": []}
        with self.manifest_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict) or not isinstance(data.get("samples"), list):
            return {"version": 1, "updated_at": None, "samples": []}
        return data

    def _write_manifest_unlocked(self, data: dict[str, Any]) -> None:
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        tmp_path = self.manifest_path.with_name(
            f"{self.manifest_path.name}.{threading.get_ident()}.tmp"
        )
        try:
            with tmp_path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            tmp_path.replace(self.manifest_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    @staticmethod
    def _build_extractor_recognizer() -> SignGestureRecognizer:
        cfg = RecognizerConfig(
            running_mode="IMAGE",
            enable_holistic=True,
            enable_custom_classifier=False,
            min_hand_detection_confidence=0.12,
            min_hand_presence_confidence=0.12,
            min_hand_landmarks_confidence=0.12,
        )
        return SignGestureRecognizer(cfg)

    def _extract_prediction(
        self,
        bgr: np.ndarray,
        recognizer: SignGestureRecognizer | None = None,
    ):
        own_rec = recognizer is None
        rec = recognizer or self._build_extractor_recognizer()
        try:
            pred = rec.process_bgr(bgr)
            if pred.hands:
                return pred

            # Retry with upscaling + replicate padding for tight/low-contrast crops.
            h, w = bgr.shape[:2]
            for scale in (2, 3):
                resized = cv2.resize(
                    bgr,
                    (w * scale, h * scale),
                    interpolation=cv2.INTER_CUBIC,
                )
                padded = cv2.copyMakeBorder(
                    resized, 40, 40, 40, 40, cv2.BORDER_REPLICATE
                )
                retry_pred = rec.process_bgr(padded)
                if retry_pred.hands:
                    return retry_pred
            return pred
        finally:
            if own_rec:
                rec.close()

    @staticmethod
    def _public_sample(entry: dict[str, Any]) -> dict[str, Any]:
        rel = str(entry["image_relpath"]).replace("\\", "/")
        return {
            "sample_id": entry["sample_id"],
            "label": entry["label"],
            "source": entry.get("source", "webcam"),
            "needs_review": bool(entry.get("needs_review", False)),
            "num_hands": int(entry.get("num_hands", 0)),
            "has_pose": bool(entry.get("has_pose", False)),
            "created_at": entry.get("created_at"),
            "image_url": f"/api/dataset/image/{rel}",
            "metadata": entry.get("metadata", {}),
        }

    def add_sample_from_bgr(
        self,
        bgr: np.ndarray,
        label: str,
        *,
        source: str = "webcam",
        needs_review: bool = False,
        metadata: dict[str, Any] | None = None,
        recognizer: SignGestureRecognizer | None = None,
        sample_id: str | None = None,
    ) -> dict[str, Any]:
        """Extract landmarks from a BGR frame, store image + features, and return sample info."""
        if bgr is None or bgr.size == 0:
            raise ValueError("Input image is empty.")

        clean_label = normalize_sign_label(label)
        safe_dir = sanitize_label_dir(clean_label)
        pred = self._extract_prediction(bgr, recognizer=recognizer)
        if not pred.hands:
            raise ValueError(
                "No hand landmarks detected in frame. Center your hand(s) in good lighting and try again."
            )

        pose_lms = pred.holistic.get("pose_landmarks") if pred.holistic else None
        features = extract_sign_features(pred.hands, pose_landmarks=pose_lms)

        sid = sample_id or uuid.uuid4().hex[:12]
        rel_path = Path(safe_dir) / f"{sid}.jpg"
        abs_path = self.images_dir / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            raise ValueError("Failed to encode sample JPEG image.")
        abs_path.write_bytes(encoded.tobytes())

        entry: dict[str, Any] = {
            "sample_id": sid,
            "label": clean_label,
            "source": source,
            "needs_review": bool(needs_review),
            "num_hands": len(pred.hands),
            "has_pose": bool(pose_lms),
            "image_relpath": rel_path.as_posix(),
            "features": [round(float(x), 6) for x in features.tolist()],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }

        with self._lock:
            manifest = self._read_manifest_unlocked()
            # Replace existing sample if deterministic sample_id was supplied (idempotent bootstrap).
            samples = [
                s for s in manifest["samples"] if s.get("sample_id") != sid
            ]
            samples.append(entry)
            manifest["samples"] = samples
            self._write_manifest_unlocked(manifest)

        return self._public_sample(entry)

    def add_sample_from_bytes(
        self,
        raw_bytes: bytes,
        label: str,
        *,
        source: str = "webcam",
        needs_review: bool = False,
        metadata: dict[str, Any] | None = None,
        recognizer: SignGestureRecognizer | None = None,
    ) -> dict[str, Any]:
        """Decode uploaded image bytes and store extracted sample."""
        if not raw_bytes:
            raise ValueError("Uploaded image file is empty.")
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image bytes.")
        return self.add_sample_from_bgr(
            bgr,
            label,
            source=source,
            needs_review=needs_review,
            metadata=metadata,
            recognizer=recognizer,
        )

    def list_samples(
        self,
        *,
        label: str | None = None,
        source: str | None = None,
        needs_review: bool | None = None,
    ) -> list[dict[str, Any]]:
        """List stored samples matching optional filters."""
        with self._lock:
            manifest = self._read_manifest_unlocked()
            items = manifest["samples"]

        out: list[dict[str, Any]] = []
        for entry in items:
            if label and entry.get("label") != label:
                continue
            if source and entry.get("source") != source:
                continue
            if needs_review is not None and bool(entry.get("needs_review")) != needs_review:
                continue
            out.append(self._public_sample(entry))
        return out

    def get_summary(self) -> dict[str, Any]:
        """Return per-class sample counts, sources, and review counts."""
        with self._lock:
            manifest = self._read_manifest_unlocked()
            samples = manifest["samples"]
            updated_at = manifest.get("updated_at")

        by_class: dict[str, dict[str, Any]] = {}
        source_totals: dict[str, int] = {}
        review_total = 0

        for entry in samples:
            lbl = entry["label"]
            src = entry.get("source", "webcam")
            rev = bool(entry.get("needs_review", False))
            source_totals[src] = source_totals.get(src, 0) + 1
            if rev:
                review_total += 1

            cls_info = by_class.get(lbl)
            if cls_info is None:
                cls_info = {
                    "label": lbl,
                    "count": 0,
                    "needs_review_count": 0,
                    "sources": {},
                    "translations": translate_gesture_label(lbl),
                    "preview_url": self._public_sample(entry)["image_url"],
                }
                by_class[lbl] = cls_info

            cls_info["count"] += 1
            if rev:
                cls_info["needs_review_count"] += 1
            cls_info["sources"][src] = cls_info["sources"].get(src, 0) + 1

        classes_sorted = sorted(by_class.values(), key=lambda c: c["label"].lower())
        return {
            "dataset_dir": str(self.dataset_dir),
            "updated_at": updated_at,
            "total_samples": len(samples),
            "total_classes": len(classes_sorted),
            "needs_review_total": review_total,
            "source_totals": source_totals,
            "classes": classes_sorted,
        }

    def update_sample(
        self,
        sample_id: str,
        *,
        label: str | None = None,
        needs_review: bool | None = None,
    ) -> dict[str, Any]:
        """Relabel or mark a sample as reviewed."""
        with self._lock:
            manifest = self._read_manifest_unlocked()
            target = next(
                (s for s in manifest["samples"] if s.get("sample_id") == sample_id),
                None,
            )
            if target is None:
                raise KeyError(f"Sample {sample_id!r} not found.")

            if label is not None:
                new_label = normalize_sign_label(label)
                if new_label != target["label"]:
                    old_abs = self.images_dir / target["image_relpath"]
                    new_dir = sanitize_label_dir(new_label)
                    new_rel = Path(new_dir) / Path(target["image_relpath"]).name
                    new_abs = self.images_dir / new_rel
                    new_abs.parent.mkdir(parents=True, exist_ok=True)
                    if old_abs.is_file():
                        old_abs.replace(new_abs)
                    target["label"] = new_label
                    target["image_relpath"] = new_rel.as_posix()

            if needs_review is not None:
                target["needs_review"] = bool(needs_review)

            self._write_manifest_unlocked(manifest)
            return self._public_sample(target)

    def delete_sample(self, sample_id: str) -> bool:
        """Delete a sample from manifest and remove its image file."""
        with self._lock:
            manifest = self._read_manifest_unlocked()
            remaining: list[dict[str, Any]] = []
            removed: dict[str, Any] | None = None
            for entry in manifest["samples"]:
                if entry.get("sample_id") == sample_id and removed is None:
                    removed = entry
                else:
                    remaining.append(entry)

            if removed is None:
                return False

            manifest["samples"] = remaining
            self._write_manifest_unlocked(manifest)

        img_path = self.images_dir / removed["image_relpath"]
        img_path.unlink(missing_ok=True)
        return True

    def get_training_data(self) -> dict[str, list[np.ndarray]]:
        """Return mapping of label -> list of 155-dim feature vectors."""
        with self._lock:
            manifest = self._read_manifest_unlocked()
            samples = manifest["samples"]

        by_label: dict[str, list[np.ndarray]] = {}
        for entry in samples:
            feats = entry.get("features")
            if not feats:
                continue
            by_label.setdefault(entry["label"], []).append(
                np.asarray(feats, dtype=np.float32)
            )
        return by_label

    def resolve_image_path(self, rel_path: str) -> Path:
        """Safely resolve a relative image path inside `self.images_dir`."""
        base = self.images_dir.resolve()
        candidate = (base / rel_path).resolve()
        if not candidate.is_relative_to(base) or not candidate.is_file():
            raise FileNotFoundError(f"Sample image not found: {rel_path!r}")
        return candidate

    def import_folder(
        self,
        folder_path: str | Path,
        *,
        max_per_class: int = 40,
        source: str = "imported",
    ) -> dict[str, Any]:
        """Import a `<root>/<label>/*.jpg` directory of sign images into the dataset."""
        root = Path(folder_path)
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {root}")

        subdirs = sorted(p for p in root.iterdir() if p.is_dir())
        if not subdirs:
            raise ValueError(
                f"Expected labeled class subdirectories inside {root} (e.g. A/, B/, Help/)."
            )

        imported = 0
        skipped = 0
        imported_classes: dict[str, int] = {}

        with self._build_extractor_recognizer() as rec:
            for class_dir in subdirs:
                label = normalize_sign_label(class_dir.name)
                files = sorted(
                    f
                    for f in class_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in _ALLOWED_EXTENSIONS
                )[: max(1, max_per_class)]
                for img_file in files:
                    bgr = cv2.imread(str(img_file))
                    if bgr is None:
                        skipped += 1
                        continue
                    try:
                        self.add_sample_from_bgr(
                            bgr,
                            label,
                            source=source,
                            needs_review=False,
                            metadata={"origin_file": img_file.name},
                            recognizer=rec,
                        )
                        imported += 1
                        imported_classes[label] = imported_classes.get(label, 0) + 1
                    except ValueError:
                        skipped += 1

        return {
            "folder": str(root),
            "imported": imported,
            "skipped": skipped,
            "classes": imported_classes,
        }
