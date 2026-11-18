"""Bootstrap custom sign dataset & PyTorch model from sign reference collages."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2

from .config import BACKEND_DIR, DEFAULT_CUSTOM_CLASSIFIER_PATH, DEFAULT_DATASET_DIR
from .custom_classifier import train_landmark_classifier
from .sign_dataset import SignDatasetStore

IMAGE2_LABELS: tuple[str, ...] = (
    "Change",
    "Correct",
    "Goodbye",
    "Help",
    "Home",
    "Hug",
    "Meet",
    "Nice",
    "No",
    "Please",
    "Sorry",
    "Stop",
    "Touch",
    "Yes",
    "You",
)

# Best-effort visual mapping for the unlabeled 8x8 ISL alphanumeric collage.
# Marked with `needs_review=True` so users can verify/relabel tiles in the UI.
IMAGE1_GRID_LABELS: tuple[tuple[str, ...], ...] = (
    ("A", "9", "6", "5", "V", "9", "L", "V"),
    ("J", "E", "O", "E", "T", "K", "E", "L"),
    ("Z", "D", "K", "H", "P", "B", "U", "F"),
    ("H", "Q", "R", "X", "C", "P", "M", "4"),
    ("3", "Z", "I", "B", "V", "6", "2", "I"),
    ("X", "T", "4", "Q", "2", "2", "U", "E"),
    ("H", "4", "H", "G", "G", "J", "3", "E"),
    ("S", "O", "1", "Y", "C", "E", "1", "J"),
)


def bootstrap_from_collages(
    *,
    image1_path: Path | None = None,
    image2_path: Path | None = None,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    model_path: Path = DEFAULT_CUSTOM_CLASSIFIER_PATH,
    ref_dir: Path | None = None,
    train: bool = True,
) -> dict[str, Any]:
    """Slice labeled/inferred tiles from sign collage images, extract landmarks, and train."""
    store = SignDatasetStore(dataset_dir)
    extracted = 0
    skipped = 0
    details: dict[str, int] = {"image1": 0, "image2": 0}

    with store._build_extractor_recognizer() as rec:
        if image2_path is not None and image2_path.is_file():
            img2 = cv2.imread(str(image2_path))
            if img2 is not None:
                h, w = img2.shape[:2]
                if ref_dir is not None:
                    ref_dir.mkdir(parents=True, exist_ok=True)
                for idx, label in enumerate(IMAGE2_LABELS):
                    r, c = divmod(idx, 5)
                    y0, y1 = int(r * h / 3) + 2, int((r + 1) * h / 3) - 34
                    x0, x1 = int(c * w / 5) + 4, int((c + 1) * w / 5) - 4
                    crop = img2[y0:y1, x0:x1]
                    if ref_dir is not None:
                        cv2.imwrite(str(ref_dir / f"{label.lower()}.jpg"), crop)
                    try:
                        store.add_sample_from_bgr(
                            crop,
                            label,
                            source="image2_labeled",
                            needs_review=False,
                            metadata={"grid_index": idx, "grid_row": r, "grid_col": c},
                            recognizer=rec,
                            sample_id=f"img2_{label.lower()}",
                        )
                        extracted += 1
                        details["image2"] += 1
                    except ValueError:
                        skipped += 1

        if image1_path is not None and image1_path.is_file():
            img1 = cv2.imread(str(image1_path))
            if img1 is not None:
                h1, w1 = img1.shape[:2]
                for r in range(8):
                    for c in range(8):
                        label = IMAGE1_GRID_LABELS[r][c]
                        y0, y1 = int(r * h1 / 8) + 2, int((r + 1) * h1 / 8) - 2
                        x0, x1 = int(c * w1 / 8) + 2, int((c + 1) * w1 / 8) - 2
                        tile = img1[y0:y1, x0:x1]
                        try:
                            store.add_sample_from_bgr(
                                tile,
                                label,
                                source="image1_inferred",
                                needs_review=True,
                                metadata={"grid_row": r, "grid_col": c},
                                recognizer=rec,
                                sample_id=f"img1_r{r}c{c}",
                            )
                            extracted += 1
                            details["image1"] += 1
                        except ValueError:
                            skipped += 1

    train_result: dict[str, Any] | None = None
    if train:
        training_data = store.get_training_data()
        if training_data:
            train_result = train_landmark_classifier(
                training_data,
                output_path=model_path,
            )

    return {
        "extracted": extracted,
        "skipped": skipped,
        "details": details,
        "summary": store.get_summary(),
        "train": train_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap SignBridge custom sign dataset.")
    parser.add_argument("--image1", type=Path, default=None, help="Path to 8x8 ISL grid image.")
    parser.add_argument("--image2", type=Path, default=None, help="Path to 3x5 labeled word grid image.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_CUSTOM_CLASSIFIER_PATH)
    parser.add_argument(
        "--ref-dir",
        type=Path,
        default=BACKEND_DIR.parent / "frontend" / "public" / "signs" / "ref",
    )
    parser.add_argument("--no-train", action="store_true")
    args = parser.parse_args()

    result = bootstrap_from_collages(
        image1_path=args.image1,
        image2_path=args.image2,
        dataset_dir=args.dataset_dir,
        model_path=args.model_path,
        ref_dir=args.ref_dir,
        train=not args.no_train,
    )
    print(
        f"Extracted={result['extracted']} skipped={result['skipped']} "
        f"classes={result['summary']['total_classes']} "
        f"samples={result['summary']['total_samples']}"
    )
    if result["train"]:
        print(
            f"Trained model -> {result['train']['model_path']} "
            f"(acc={result['train']['train_accuracy']}, classes={result['train']['num_classes']})"
        )


if __name__ == "__main__":
    main()
