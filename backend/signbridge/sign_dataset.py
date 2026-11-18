"""Dataset management for custom ISL landmark samples and image thumbnails."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import threading
import time
from typing import TYPE_CHECKING, Any
import uuid

import cv2
import numpy as np

from .config import DEFAULT_DATASET_DIR, DEFAULT_DB_PATH, RecognizerConfig, translate_gesture_label
from .custom_classifier import extract_sign_features
from .database import SignBridgeDatabase

if TYPE_CHECKING:
    from .recognizer import SignGestureRecognizer

_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_SINGLE_HAND_IMAGE2_LABELS: set[str] = {
    "Correct",
    "Goodbye",
    "Nice",
    "No",
    "Please",
    "Sorry",
    "Yes",
    "You",
}
_TWO_HAND_PREFERRED_LABELS: set[str] = {
    "Book",
    "Change",
    "Cook",
    "Cracker",
    "Finish",
    "Friend",
    "Help",
    "Home",
    "Hug",
    "Meet",
    "More",
    "Play",
    "Same",
    "Stop",
    "Touch",
    "Wait",
    "Welcome",
    "With",
    "Work",
}


def _validated_pose_landmarks(
    pose_lms: list[dict[str, Any]] | None,
    *,
    source: str,
) -> list[dict[str, Any]] | None:
    """Reject hallucinated holistic shoulders on hand-only / graphic crops."""
    if source in {"image1_inferred", "poster_illustrated"}:
        return None
    if not pose_lms or len(pose_lms) <= 12:
        return None
    ls, rs = pose_lms[11], pose_lms[12]
    ls_v = float(ls.get("v", ls.get("visibility", 0.0)) or 0.0)
    rs_v = float(rs.get("v", rs.get("visibility", 0.0)) or 0.0)
    if ls_v < 0.45 or rs_v < 0.45:
        return None
    lx, ly = float(ls.get("x", 0.0)), float(ls.get("y", 0.0))
    rx, ry = float(rs.get("x", 0.0)), float(rs.get("y", 0.0))
    sw = abs(lx - rx)
    if not (0.12 <= sw <= 0.88):
        return None
    if not (0.02 <= lx <= 0.98 and 0.02 <= rx <= 0.98 and 0.02 <= ly <= 0.98 and 0.02 <= ry <= 0.98):
        return None
    return pose_lms


def _filter_static_hands(
    hands: list[Any],
    *,
    label: str,
    source: str,
) -> list[Any]:
    """Drop low-border resting fists/crop-edge false detections on single-hand collages."""
    if len(hands) <= 1:
        return hands
    if source == "image2_labeled" and label in _SINGLE_HAND_IMAGE2_LABELS:
        # Pick the primary signing hand (highest in frame, i.e. smallest wrist y).
        return [min(hands, key=lambda h: float(h.wrist_xy[1]))]
    return hands


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

    def __init__(
        self,
        dataset_dir: Path = DEFAULT_DATASET_DIR,
        *,
        db: SignBridgeDatabase | None = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / "images"
        self.manifest_path = self.dataset_dir / "manifest.json"
        self.sequences_path = self.dataset_dir / "sequences.json"
        self._lock = threading.Lock()
        if db is not None:
            self._db: SignBridgeDatabase | None = db
        elif self.dataset_dir.resolve() == DEFAULT_DATASET_DIR.resolve():
            self._db = SignBridgeDatabase(DEFAULT_DB_PATH)
        else:
            self._db = None
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
        if self._db is not None:
            manifest = self._read_manifest_unlocked()
            samples = manifest.get("samples", [])
            if self._db.get_stats()["dataset_samples"] != len(samples):
                self._db.sync_dataset_samples(samples)

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
            for attempt in range(8):
                try:
                    tmp_path.replace(self.manifest_path)
                    break
                except PermissionError:
                    if attempt == 7:
                        raise
                    time.sleep(0.1)
        finally:
            tmp_path.unlink(missing_ok=True)

    @staticmethod
    def _build_extractor_recognizer() -> SignGestureRecognizer:
        from .recognizer import SignGestureRecognizer

        cfg = RecognizerConfig(
            running_mode="IMAGE",
            enable_holistic=True,
            enable_custom_classifier=False,
            enable_dynamic_recognizer=False,
            min_hand_detection_confidence=0.22,
            min_hand_presence_confidence=0.22,
            min_hand_landmarks_confidence=0.22,
            min_pose_detection_confidence=0.45,
        )
        return SignGestureRecognizer(cfg)

    def _extract_prediction_with_aspect(
        self,
        bgr: np.ndarray,
        recognizer: SignGestureRecognizer | None = None,
        *,
        prefer_multi_hand: bool = False,
    ) -> tuple[Any, float]:
        own_rec = recognizer is None
        rec = recognizer or self._build_extractor_recognizer()
        try:
            h, w = bgr.shape[:2]
            raw_ar = float(w) / float(max(1, h))
            pred = rec.process_bgr(bgr)
            if pred.hands and (not prefer_multi_hand or len(pred.hands) >= 2):
                return pred, raw_ar

            best_pred = pred
            best_ar = raw_ar

            # Retry with upscaling + replicate padding for tight/low-contrast or 2-hand crops.
            for scale in (2, 3):
                resized = cv2.resize(
                    bgr,
                    (w * scale, h * scale),
                    interpolation=cv2.INTER_CUBIC,
                )
                padded = cv2.copyMakeBorder(
                    resized, 40, 40, 40, 40, cv2.BORDER_REPLICATE
                )
                ph, pw = padded.shape[:2]
                pad_ar = float(pw) / float(max(1, ph))
                retry_pred = rec.process_bgr(padded)
                if len(retry_pred.hands) > len(best_pred.hands):
                    best_pred = retry_pred
                    best_ar = pad_ar
                if retry_pred.hands and (
                    not prefer_multi_hand or len(retry_pred.hands) >= 2
                ):
                    return retry_pred, pad_ar
            return best_pred, best_ar
        finally:
            if own_rec:
                rec.close()

    def _extract_prediction(
        self,
        bgr: np.ndarray,
        recognizer: SignGestureRecognizer | None = None,
    ):
        pred, _ = self._extract_prediction_with_aspect(bgr, recognizer=recognizer)
        return pred

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

    def _store_sample_record(
        self,
        bgr: np.ndarray,
        label: str,
        *,
        features: np.ndarray,
        num_hands: int,
        has_pose: bool,
        source: str,
        needs_review: bool,
        metadata: dict[str, Any] | None,
        sample_id: str | None,
    ) -> dict[str, Any]:
        """Persist preview JPEG, manifest entry, and SQLite metadata atomically."""
        clean_label = normalize_sign_label(label)
        safe_dir = sanitize_label_dir(clean_label)

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
            "num_hands": int(num_hands),
            "has_pose": bool(has_pose),
            "image_relpath": rel_path.as_posix(),
            "features": [round(float(x), 6) for x in np.asarray(features, dtype=np.float32).tolist()],
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
            if self._db is not None:
                self._db.upsert_dataset_sample(entry)

        return self._public_sample(entry)

    def add_samples_batch_from_features(
        self,
        items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Persist multiple precomputed landmark samples in a single manifest/DB write."""
        if not items:
            return []

        entries: list[dict[str, Any]] = []
        replace_ids: set[str] = set()
        now_iso = datetime.now(timezone.utc).isoformat()

        for item in items:
            bgr = item["bgr"]
            if bgr is None or bgr.size == 0:
                raise ValueError("Input preview image is empty.")
            clean_label = normalize_sign_label(item["label"])
            safe_dir = sanitize_label_dir(clean_label)
            sid = item.get("sample_id") or uuid.uuid4().hex[:12]
            rel_path = Path(safe_dir) / f"{sid}.jpg"
            abs_path = self.images_dir / rel_path
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            if not ok:
                raise ValueError("Failed to encode sample JPEG image.")
            abs_path.write_bytes(encoded.tobytes())

            feats = np.asarray(item["features"], dtype=np.float32)
            entries.append(
                {
                    "sample_id": sid,
                    "label": clean_label,
                    "source": item.get("source", "voice_to_sign_avatar"),
                    "needs_review": bool(item.get("needs_review", False)),
                    "num_hands": int(item["num_hands"]),
                    "has_pose": bool(item.get("has_pose", False)),
                    "image_relpath": rel_path.as_posix(),
                    "features": [round(float(x), 6) for x in feats.tolist()],
                    "created_at": now_iso,
                    "metadata": item.get("metadata") or {},
                }
            )
            replace_ids.add(sid)

        with self._lock:
            manifest = self._read_manifest_unlocked()
            kept = [
                s for s in manifest["samples"] if s.get("sample_id") not in replace_ids
            ]
            manifest["samples"] = kept + entries
            self._write_manifest_unlocked(manifest)
            if self._db is not None:
                for entry in entries:
                    self._db.upsert_dataset_sample(entry)

        return [self._public_sample(entry) for entry in entries]

    def add_sample_from_features(
        self,
        bgr: np.ndarray,
        label: str,
        *,
        features: np.ndarray,
        num_hands: int,
        has_pose: bool = False,
        source: str = "voice_to_sign_avatar",
        needs_review: bool = False,
        metadata: dict[str, Any] | None = None,
        sample_id: str | None = None,
    ) -> dict[str, Any]:
        """Store a precomputed 155-dim landmark feature vector alongside a preview JPEG."""
        if bgr is None or bgr.size == 0:
            raise ValueError("Input preview image is empty.")
        return self._store_sample_record(
            bgr,
            label,
            features=features,
            num_hands=num_hands,
            has_pose=has_pose,
            source=source,
            needs_review=needs_review,
            metadata=metadata,
            sample_id=sample_id,
        )

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
        prefer_two = clean_label in _TWO_HAND_PREFERRED_LABELS
        pred, used_ar = self._extract_prediction_with_aspect(
            bgr,
            recognizer=recognizer,
            prefer_multi_hand=prefer_two,
        )
        if not pred.hands:
            raise ValueError(
                "No hand landmarks detected in frame. Center your hand(s) in good lighting and try again."
            )

        clean_hands = _filter_static_hands(pred.hands, label=clean_label, source=source)
        raw_pose = pred.holistic.get("pose_landmarks") if pred.holistic else None
        pose_lms = _validated_pose_landmarks(raw_pose, source=source)
        features = extract_sign_features(
            clean_hands,
            pose_landmarks=pose_lms,
            aspect_ratio=used_ar,
        )
        return self._store_sample_record(
            bgr,
            clean_label,
            features=features,
            num_hands=len(clean_hands),
            has_pose=bool(pose_lms),
            source=source,
            needs_review=needs_review,
            metadata=metadata,
            sample_id=sample_id,
        )

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
            if self._db is not None:
                self._db.upsert_dataset_sample(target)
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
            if self._db is not None:
                self._db.delete_dataset_sample(sample_id)

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

    def _read_sequences_unlocked(self) -> dict[str, Any]:
        if not self.sequences_path.is_file():
            return {"version": 1, "updated_at": None, "sequences": []}
        data = json.loads(self.sequences_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("sequences"), list):
            return {"version": 1, "updated_at": None, "sequences": []}
        return data

    def _write_sequences_unlocked(self, data: dict[str, Any]) -> None:
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        tmp_path = self.sequences_path.with_name(
            f"{self.sequences_path.name}.{threading.get_ident()}.tmp"
        )
        try:
            tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            for attempt in range(8):
                try:
                    tmp_path.replace(self.sequences_path)
                    break
                except PermissionError:
                    if attempt == 7:
                        raise
                    time.sleep(0.1)
        finally:
            tmp_path.unlink(missing_ok=True)

    def add_sequence_from_frames(
        self,
        frame_bytes_list: list[bytes],
        label: str,
        *,
        timestamps_ms: list[int] | None = None,
        source: str = "webcam_sequence",
        recognizer: SignGestureRecognizer | None = None,
    ) -> dict[str, Any]:
        """Extract 46-dim kinematic frames across a recorded moving sign clip and persist."""
        if len(frame_bytes_list) < 4:
            raise ValueError("Need at least 4 frames to record a moving sign sequence.")
        clean_label = normalize_sign_label(label)
        safe_dir = sanitize_label_dir(clean_label)

        own_rec = recognizer is None
        rec = recognizer or self._build_extractor_recognizer()
        extracted_seq: list[tuple[int, list[float]]] = []
        mid_bgr: np.ndarray | None = None

        from .dynamic_recognizer import extract_kinematic_frame

        try:
            for idx, raw in enumerate(frame_bytes_list):
                arr = np.frombuffer(raw, dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                if idx == len(frame_bytes_list) // 2:
                    mid_bgr = bgr
                pred, used_ar = self._extract_prediction_with_aspect(
                    bgr,
                    recognizer=rec,
                    prefer_multi_hand=clean_label in _TWO_HAND_PREFERRED_LABELS,
                )
                if not pred.hands:
                    continue
                pose_lms = pred.holistic.get("pose_landmarks") if pred.holistic else None
                face_lms = pred.holistic.get("face_landmarks") if pred.holistic else None
                kf = extract_kinematic_frame(
                    pred.hands,
                    pose_landmarks=pose_lms,
                    face_landmarks=face_lms,
                    aspect_ratio=used_ar,
                )
                ts = (
                    int(timestamps_ms[idx])
                    if timestamps_ms and idx < len(timestamps_ms)
                    else idx * 100
                )
                extracted_seq.append((ts, [round(float(x), 5) for x in kf.tolist()]))
        finally:
            if own_rec:
                rec.close()

        if len(extracted_seq) < 4:
            raise ValueError(
                "Could not detect hands across enough frames in the clip. Keep hands inside frame and retry."
            )

        seq_id = f"seq_{uuid.uuid4().hex[:10]}"
        rel_path = Path(safe_dir) / f"{seq_id}.jpg"
        abs_path = self.images_dir / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        if mid_bgr is not None:
            ok, encoded = cv2.imencode(".jpg", mid_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            if ok:
                abs_path.write_bytes(encoded.tobytes())

        entry: dict[str, Any] = {
            "sequence_id": seq_id,
            "label": clean_label,
            "source": source,
            "num_frames": len(extracted_seq),
            "duration_ms": int(extracted_seq[-1][0] - extracted_seq[0][0]),
            "image_relpath": rel_path.as_posix(),
            "frames": [{"ts": ts, "kframe": kf} for ts, kf in extracted_seq],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            payload = self._read_sequences_unlocked()
            payload["sequences"].append(entry)
            self._write_sequences_unlocked(payload)

        return {
            "sequence_id": seq_id,
            "label": clean_label,
            "source": source,
            "num_frames": len(extracted_seq),
            "duration_ms": entry["duration_ms"],
            "created_at": entry["created_at"],
        }

    def list_sequences(self, *, label: str | None = None) -> list[dict[str, Any]]:
        """List stored dynamic motion sequences."""
        with self._lock:
            seqs = self._read_sequences_unlocked().get("sequences", [])
        out: list[dict[str, Any]] = []
        for s in seqs:
            if label and s.get("label") != label:
                continue
            out.append(
                {
                    "sequence_id": s["sequence_id"],
                    "label": s["label"],
                    "source": s.get("source", "webcam_sequence"),
                    "num_frames": int(s.get("num_frames", 0)),
                    "duration_ms": int(s.get("duration_ms", 0)),
                    "created_at": s.get("created_at"),
                }
            )
        return out

    def delete_sequence(self, sequence_id: str) -> bool:
        """Delete a stored moving-sign sequence."""
        with self._lock:
            payload = self._read_sequences_unlocked()
            seqs = payload.get("sequences", [])
            kept = [s for s in seqs if s.get("sequence_id") != sequence_id]
            if len(kept) == len(seqs):
                return False
            payload["sequences"] = kept
            self._write_sequences_unlocked(payload)
            return True

    def get_dynamic_training_data(self) -> dict[str, list[list[tuple[int, np.ndarray]]]]:
        """Return user-recorded moving-sign sequences keyed by label."""
        with self._lock:
            seqs = self._read_sequences_unlocked().get("sequences", [])
        out: dict[str, list[list[tuple[int, np.ndarray]]]] = {}
        for s in seqs:
            frames_raw = s.get("frames") or []
            if len(frames_raw) < 4:
                continue
            parsed = [
                (int(f["ts"]), np.asarray(f["kframe"], dtype=np.float32))
                for f in frames_raw
            ]
            out.setdefault(s["label"], []).append(parsed)
        return out

    def reextract_clean_image_samples(self) -> dict[str, Any]:
        """Re-extract landmarks on stored image samples with deduplicated hands & valid pose."""
        from .recognizer import SignGestureRecognizer

        low_cfg = RecognizerConfig(
            running_mode="IMAGE",
            enable_holistic=True,
            enable_custom_classifier=False,
            enable_dynamic_recognizer=False,
            min_hand_detection_confidence=0.12,
            min_hand_presence_confidence=0.12,
            min_hand_landmarks_confidence=0.12,
            min_pose_detection_confidence=0.45,
        )
        updated = 0
        kept_fallback = 0
        re_pose_cleared = 0

        with self._lock:
            manifest = self._read_manifest_unlocked()
            samples = manifest.get("samples", [])

        with self._build_extractor_recognizer() as rec_hi, SignGestureRecognizer(low_cfg) as rec_lo:
            for entry in samples:
                src = entry.get("source", "webcam")
                if src == "voice_to_sign_avatar":
                    continue
                img_path = self.images_dir / str(entry.get("image_relpath", ""))
                if not img_path.is_file():
                    continue
                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    continue

                prefer_two = entry["label"] in _TWO_HAND_PREFERRED_LABELS
                pred, used_ar = self._extract_prediction_with_aspect(
                    bgr,
                    recognizer=rec_hi,
                    prefer_multi_hand=prefer_two,
                )
                if not pred.hands or (prefer_two and len(pred.hands) < 2):
                    pred_lo, ar_lo = self._extract_prediction_with_aspect(
                        bgr,
                        recognizer=rec_lo,
                        prefer_multi_hand=prefer_two,
                    )
                    if len(pred_lo.hands) > len(pred.hands):
                        pred, used_ar = pred_lo, ar_lo

                if not pred.hands:
                    # Preserve existing features if re-extraction missed, but strip hallucinated pose.
                    feats = np.asarray(entry["features"], dtype=np.float32)
                    if src in {"image1_inferred", "poster_illustrated"} and feats[132] > 0.5:
                        feats[132:137] = 0.0
                        entry["features"] = [round(float(x), 6) for x in feats.tolist()]
                        entry["has_pose"] = False
                        re_pose_cleared += 1
                    kept_fallback += 1
                    continue

                clean_hands = _filter_static_hands(
                    pred.hands,
                    label=entry["label"],
                    source=src,
                )
                raw_pose = pred.holistic.get("pose_landmarks") if pred.holistic else None
                pose_lms = _validated_pose_landmarks(raw_pose, source=src)
                feats = extract_sign_features(
                    clean_hands,
                    pose_landmarks=pose_lms,
                    aspect_ratio=used_ar,
                )
                entry["features"] = [round(float(x), 6) for x in feats.tolist()]
                entry["num_hands"] = len(clean_hands)
                entry["has_pose"] = bool(pose_lms)
                updated += 1

        with self._lock:
            self._write_manifest_unlocked(manifest)
            if self._db is not None:
                self._db.sync_dataset_samples(manifest["samples"])

        return {
            "updated": updated,
            "kept_fallback": kept_fallback,
            "re_pose_cleared": re_pose_cleared,
        }
