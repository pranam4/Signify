"""Poster collage cropper, visual reference catalog, and dataset ingestion."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import threading
from typing import Any

import cv2

from .config import (
    BACKEND_DIR,
    DEFAULT_CUSTOM_CLASSIFIER_PATH,
    DEFAULT_DB_PATH,
    DEFAULT_POSTER_REFERENCES_PATH,
    RecognizerConfig,
    translate_gesture_label,
)
from .custom_classifier import train_landmark_classifier
from .database import SignBridgeDatabase
from .sign_dataset import SignDatasetStore

REPO_ROOT = BACKEND_DIR.parent
DEFAULT_REF_PUBLIC_DIR = REPO_ROOT / "frontend" / "public" / "signs" / "ref"


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


POSTER_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "p1_classroom",
        "title": "ASL Classroom Words",
        "category": "Classroom",
        "style": "line_art",
        "sample_source": None,
        "needs_review": True,
        "default_filename": "media_1794986576256_c38705fc.jpg",
        "cells": [
            {"label": lbl, "box": (x0 / 960, y0 / 960, x1 / 960, y1 / 960)}
            for (y0, y1), row_labels in zip(
                [(228, 340), (344, 456), (460, 574), (578, 690), (694, 806)],
                [
                    ["Answer", "Done", "Help", "I'm Sorry", "Ignore"],
                    ["Me Too", "No", "Now", "Pencil", "Please"],
                    ["Question", "Show Me", "Sit", "Stand", "Stop"],
                    ["Thank You", "Tissue", "Toilet", "Wash", "Water"],
                    ["Work", "Yes"],
                ],
            )
            for (x0, x1), lbl in zip(
                [(52, 220), (224, 392), (396, 564), (568, 736), (740, 908)],
                row_labels,
            )
        ]
        + [
            {"label": "You're Welcome", "box": (396 / 960, 694 / 960, 736 / 960, 806 / 960)},
        ],
    },
    {
        "id": "p2_essentials",
        "title": "Signs Everyone Should Know",
        "category": "Everyday Essentials",
        "style": "illustrated_hands",
        "sample_source": "poster_illustrated",
        "needs_review": False,
        "default_filename": "media_1794986576356_62f1f47f.jpg",
        "cells": [
            {"label": "Hello", "box": (0.02, 0.135, 0.27, 0.295)},
            {"label": "My Name Is", "box": (0.28, 0.135, 0.72, 0.295)},
            {"label": "Thank You", "box": (0.73, 0.135, 0.98, 0.295)},
            *[
                {"label": lbl, "box": (x0, y0, x1, y1)}
                for (y0, y1), labels in [
                    ((0.300, 0.460), ["Yes", "No", "Help", "I Love You"]),
                    ((0.465, 0.625), ["Please", "Sorry", "Stop", "Pain"]),
                    ((0.630, 0.795), ["Eat", "Drink", "More", "Bathroom"]),
                    ((0.800, 0.975), ["How", "What", "Where", "When"]),
                ]
                for (x0, x1), lbl in zip(
                    [(0.02, 0.255), (0.265, 0.500), (0.505, 0.745), (0.750, 0.985)],
                    labels,
                )
            ],
        ],
    },
    {
        "id": "p3_actions",
        "title": "Everyday Actions (Early Years)",
        "category": "Everyday Actions",
        "style": "line_art",
        "sample_source": None,
        "needs_review": True,
        "default_filename": "media_1794986576357_58275acd.jpg",
        "cells": [
            {"label": lbl, "box": (x0, y0, x1, y1)}
            for (y0, y1), labels in [
                ((0.095, 0.305), ["Drink", "Eat", "Climb", "Run"]),
                ((0.315, 0.525), ["Fall", "Give", "Push", "Cry"]),
                ((0.535, 0.750), ["Sleep", "Sit Down", "Kick", "Wait"]),
                ((0.755, 0.955), ["Walk", "Wash", "Look", "Jump"]),
            ]
            for (x0, x1), lbl in zip(
                [(0.02, 0.265), (0.265, 0.500), (0.500, 0.735), (0.735, 0.980)],
                labels,
            )
        ],
    },
    {
        "id": "p4_first_signs",
        "title": "First Signs (Real Photos)",
        "category": "First Signs",
        "style": "photo",
        "sample_source": "poster_photo",
        "needs_review": False,
        "default_filename": "media_1794986576360_5376db9d.jpg",
        "cells": [
            {"label": lbl, "box": (x0 / 454, y0 / 675, x1 / 454, y1 / 675)}
            for (y0, y1), labels in zip(
                [(10, 129), (135, 254), (260, 378), (384, 503), (509, 628)],
                [
                    ["More", "Finish", "Eat"],
                    ["Drink", "Milk", "Want"],
                    ["Cookie", "Cracker", "Book"],
                    ["Home", "Bed", "Bath"],
                    ["Mom", "Dad", "Thank You"],
                ],
            )
            for (x0, x1), lbl in zip([(14, 150), (160, 293), (303, 440)], labels)
        ],
    },
    {
        "id": "p5_emergency",
        "title": "Workplace & Emergency Signs",
        "category": "Emergency",
        "style": "line_art",
        "sample_source": None,
        "needs_review": True,
        "default_filename": "media_1794986576367_3994b3a6.jpg",
        "cells": [
            {"label": "Ambulance", "box": (0.010, 0.140, 0.215, 0.355)},
            {"label": "Police", "box": (0.215, 0.125, 0.400, 0.355)},
            {"label": "Fire", "box": (0.400, 0.115, 0.590, 0.340)},
            {"label": "Sick", "box": (0.595, 0.130, 0.790, 0.355)},
            {"label": "Water", "box": (0.790, 0.145, 0.990, 0.355)},
            {"label": "Up", "box": (0.055, 0.385, 0.245, 0.555)},
            {"label": "Down", "box": (0.250, 0.375, 0.450, 0.545)},
            {"label": "Over", "box": (0.450, 0.355, 0.630, 0.520)},
            {"label": "Under", "box": (0.630, 0.375, 0.805, 0.550)},
            {"label": "Outside", "box": (0.805, 0.385, 0.985, 0.560)},
            {"label": "Come", "box": (0.036, 0.595, 0.250, 0.765)},
            {"label": "Go", "box": (0.255, 0.570, 0.450, 0.745)},
            {"label": "Stop", "box": (0.450, 0.555, 0.625, 0.730)},
            {"label": "Wait", "box": (0.625, 0.565, 0.795, 0.745)},
            {"label": "Careful", "box": (0.795, 0.585, 0.985, 0.770)},
            {"label": "Hot", "box": (0.070, 0.785, 0.310, 0.965)},
            {"label": "Stay Calm", "box": (0.315, 0.760, 0.535, 0.945)},
            {"label": "Wet", "box": (0.540, 0.765, 0.745, 0.950)},
            {"label": "Danger", "box": (0.750, 0.785, 0.985, 0.968)},
        ],
    },
)


def _crop_normalized(bgr: Any, box: tuple[float, float, float, float]) -> Any:
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = box
    px0 = max(0, min(w - 1, int(round(x0 * w))))
    py0 = max(0, min(h - 1, int(round(y0 * h))))
    px1 = max(px0 + 1, min(w, int(round(x1 * w))))
    py1 = max(py0 + 1, min(h, int(round(y1 * h))))
    return bgr[py0:py1, px0:px1]


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{threading.get_ident()}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def ingest_posters(
    *,
    source_dir: Path,
    store: SignDatasetStore | None = None,
    ref_public_dir: Path = DEFAULT_REF_PUBLIC_DIR,
    references_path: Path = DEFAULT_POSTER_REFERENCES_PATH,
    train_model: bool = True,
    model_output_path: Path = DEFAULT_CUSTOM_CLASSIFIER_PATH,
) -> dict[str, Any]:
    """Crop all 5 sign posters into reference cards and extract landmarks where supported."""
    store = store or SignDatasetStore()
    ref_public_dir.mkdir(parents=True, exist_ok=True)

    cfg = RecognizerConfig(
        running_mode="IMAGE",
        enable_holistic=True,
        enable_custom_classifier=False,
        min_hand_detection_confidence=0.12,
        min_hand_presence_confidence=0.12,
        min_hand_landmarks_confidence=0.12,
    )

    items: list[dict[str, Any]] = []
    poster_summaries: list[dict[str, Any]] = []
    landmark_extracted = 0
    landmark_skipped = 0

    from .recognizer import SignGestureRecognizer

    with SignGestureRecognizer(cfg) as rec:
        for spec in POSTER_SPECS:
            img_path = source_dir / spec["default_filename"]
            if not img_path.is_file():
                continue
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue

            poster_extracted = 0
            for idx, cell in enumerate(spec["cells"]):
                label = cell["label"]
                slug = _slugify(label)
                ref_filename = f"{spec['id']}_{idx:02d}_{slug}.jpg"
                ref_file_path = ref_public_dir / ref_filename
                crop = _crop_normalized(bgr, cell["box"])
                ok, encoded = cv2.imencode(
                    ".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92]
                )
                if not ok:
                    continue
                ref_file_path.write_bytes(encoded.tobytes())

                translations = translate_gesture_label(label)
                has_landmarks = False
                num_hands = 0
                sample_id = f"{spec['id']}_{slug}"

                if spec["sample_source"]:
                    try:
                        saved = store.add_sample_from_bgr(
                            crop,
                            label,
                            source=spec["sample_source"],
                            needs_review=bool(spec["needs_review"]),
                            metadata={
                                "poster": spec["title"],
                                "poster_id": spec["id"],
                                "category": spec["category"],
                                "style": spec["style"],
                            },
                            recognizer=rec,
                            sample_id=sample_id,
                        )
                        has_landmarks = True
                        num_hands = int(saved.get("num_hands", 0))
                        landmark_extracted += 1
                        poster_extracted += 1
                    except ValueError:
                        landmark_skipped += 1

                items.append(
                    {
                        "id": sample_id,
                        "label": label,
                        "poster_id": spec["id"],
                        "poster_title": spec["title"],
                        "category": spec["category"],
                        "style": spec["style"],
                        "en": translations["en"],
                        "hi": translations["hi"],
                        "kn": translations["kn"],
                        "image_url": f"/signs/ref/{ref_filename}",
                        "has_landmarks": has_landmarks,
                        "num_hands": num_hands,
                    }
                )

            poster_summaries.append(
                {
                    "id": spec["id"],
                    "title": spec["title"],
                    "category": spec["category"],
                    "style": spec["style"],
                    "count": len(spec["cells"]),
                    "landmarks_extracted": poster_extracted,
                }
            )

    payload = {
        "version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_cards": len(items),
        "posters": poster_summaries,
        "items": items,
    }
    _atomic_write_json(references_path, payload)

    db = SignBridgeDatabase(DEFAULT_DB_PATH)
    try:
        db.upsert_poster_references(items)
    finally:
        db.close()

    train_info = None
    if train_model:
        training_data = store.get_training_data()
        if training_data:
            train_info = train_landmark_classifier(
                training_data,
                output_path=model_output_path,
                epochs=55,
            )

    return {
        "total_reference_cards": len(items),
        "posters": poster_summaries,
        "landmarks_extracted": landmark_extracted,
        "landmarks_skipped": landmark_skipped,
        "dataset_summary": store.get_summary(),
        "train": train_info,
    }


class PosterReferenceStore:
    """SQLite-backed search & filter store over the cropped poster reference catalog."""

    def __init__(
        self,
        references_path: Path = DEFAULT_POSTER_REFERENCES_PATH,
        *,
        db: SignBridgeDatabase | None = None,
    ) -> None:
        self.references_path = Path(references_path)
        self._lock = threading.Lock()
        self._payload: dict[str, Any] | None = None
        self._seeded = False
        if db is not None:
            self.db = db
        else:
            db_path = (
                DEFAULT_DB_PATH
                if self.references_path.resolve() == DEFAULT_POSTER_REFERENCES_PATH.resolve()
                else self.references_path.with_suffix(".db")
            )
            self.db = SignBridgeDatabase(db_path)

    def reload(self) -> None:
        """Invalidate cached metadata so subsequent queries resync from disk/DB."""
        with self._lock:
            self._payload = None
            self._seeded = False

    def _load_unlocked(self) -> dict[str, Any]:
        if self._payload is not None:
            return self._payload
        if not self.references_path.is_file():
            self._payload = {
                "version": 1,
                "updated_at": None,
                "total_cards": 0,
                "posters": [],
                "items": [],
            }
        else:
            self._payload = json.loads(
                self.references_path.read_text(encoding="utf-8")
            )
        if not self._seeded:
            if self.db.get_stats()["poster_cards"] == 0 and self._payload.get("items"):
                self.db.upsert_poster_references(self._payload["items"])
            self._seeded = True
        return self._payload

    def list_references(
        self,
        *,
        poster_id: str = "all",
        query: str = "",
    ) -> dict[str, Any]:
        """Return matching poster reference cards from SQLite and poster metadata."""
        with self._lock:
            payload = self._load_unlocked()
            posters = payload.get("posters", [])

        filtered = self.db.query_poster_references(poster_id=poster_id, query=query)
        return {
            "total": len(filtered),
            "posters": posters,
            "items": filtered,
        }


def main() -> None:
    """CLI entrypoint to ingest the 5 sign posters into reference cards + classifier."""
    parser = argparse.ArgumentParser(description="Ingest uploaded sign language posters.")
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Directory containing the 5 uploaded poster JPEG files.",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Skip retraining the PyTorch landmark classifier.",
    )
    args = parser.parse_args()
    result = ingest_posters(
        source_dir=Path(args.source_dir),
        train_model=not args.no_train,
    )
    print(
        f"Ingested {result['total_reference_cards']} poster cards "
        f"({result['landmarks_extracted']} landmark samples extracted). "
        f"Dataset now has {result['dataset_summary']['total_classes']} classes / "
        f"{result['dataset_summary']['total_samples']} samples."
    )
    if result["train"]:
        print(
            f"Retrained model -> {result['train']['model_path']} "
            f"(acc={result['train']['train_accuracy']}, classes={result['train']['num_classes']})"
        )


if __name__ == "__main__":
    main()
