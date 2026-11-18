"""FastAPI endpoints for custom sign dataset collection, review, and PyTorch training."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import (
    DEFAULT_CUSTOM_CLASSIFIER_PATH,
    DEFAULT_DATASET_DIR,
    RecognizerConfig,
    config_from_env,
)
from .custom_classifier import CustomSignClassifier, train_landmark_classifier
from .recognizer import SignGestureRecognizer
from .sign_dataset import SignDatasetStore

dataset_router = APIRouter(tags=["sign-dataset"])
_DATASET_LOCK = threading.Lock()
_TRAIN_LOCK = threading.Lock()


class SamplePatchPayload(BaseModel):
    label: str | None = None
    needs_review: bool | None = None


class ImportFolderPayload(BaseModel):
    folder_path: str = Field(..., min_length=1)
    max_per_class: int = Field(default=40, ge=1, le=500)


class TrainPayload(BaseModel):
    epochs: int = Field(default=45, ge=5, le=200)


def get_dataset_store(app_state: Any) -> SignDatasetStore:
    store = getattr(app_state, "dataset_store", None)
    if store is None:
        store = SignDatasetStore(DEFAULT_DATASET_DIR)
        app_state.dataset_store = store
    return store


def get_custom_classifier(app_state: Any) -> CustomSignClassifier:
    clf = getattr(app_state, "custom_classifier", None)
    if clf is None:
        cfg = config_from_env(running_mode="IMAGE")
        clf = CustomSignClassifier(cfg.custom_classifier_path)
        app_state.custom_classifier = clf
    return clf


def _get_dataset_extractor(app_state: Any) -> SignGestureRecognizer:
    rec = getattr(app_state, "dataset_extractor", None)
    if rec is None:
        rec = SignGestureRecognizer(
            RecognizerConfig(
                running_mode="IMAGE",
                enable_holistic=True,
                enable_custom_classifier=False,
                min_hand_detection_confidence=0.25,
                min_hand_presence_confidence=0.25,
                min_hand_landmarks_confidence=0.25,
            )
        )
        app_state.dataset_extractor = rec
    return rec


def _add_sample_sync(
    app_state: Any,
    raw_bytes: bytes,
    label: str,
    source: str,
    needs_review: bool,
) -> dict[str, Any]:
    store = get_dataset_store(app_state)
    with _DATASET_LOCK:
        extractor = _get_dataset_extractor(app_state)
        return store.add_sample_from_bytes(
            raw_bytes,
            label,
            source=source,
            needs_review=needs_review,
            recognizer=extractor,
        )


def _train_sync(app_state: Any, epochs: int) -> dict[str, Any]:
    store = get_dataset_store(app_state)
    classifier = get_custom_classifier(app_state)
    with _TRAIN_LOCK:
        training_data = store.get_training_data()
        if not training_data:
            raise ValueError(
                "Dataset has no landmark samples yet. Capture webcam frames or import images first."
            )
        metrics = train_landmark_classifier(
            training_data,
            output_path=classifier.model_path or DEFAULT_CUSTOM_CLASSIFIER_PATH,
            epochs=epochs,
        )
        classifier.maybe_reload()
        return {
            "status": "trained",
            **metrics,
            "classifier": classifier.status(),
        }


@dataset_router.get("/api/dataset")
def dataset_overview(request: Request) -> dict[str, Any]:
    """Return custom dataset class summary, review counts, and classifier status."""
    store = get_dataset_store(request.app.state)
    classifier = get_custom_classifier(request.app.state)
    summary = store.get_summary()
    summary["classifier"] = classifier.status()
    return summary


@dataset_router.get("/api/dataset/samples")
def list_dataset_samples(
    request: Request,
    label: str | None = None,
    source: str | None = None,
    needs_review: bool | None = None,
) -> dict[str, Any]:
    """List extracted dataset samples with optional label/source/review filters."""
    store = get_dataset_store(request.app.state)
    return {
        "samples": store.list_samples(
            label=label,
            source=source,
            needs_review=needs_review,
        )
    }


@dataset_router.get("/api/dataset/image/{label_dir}/{filename}")
def get_dataset_image(request: Request, label_dir: str, filename: str) -> FileResponse:
    """Serve a stored dataset sample thumbnail."""
    store = get_dataset_store(request.app.state)
    try:
        img_path = store.resolve_image_path(f"{label_dir}/{filename}")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(img_path, media_type="image/jpeg")


@dataset_router.post("/api/dataset/samples")
async def upload_dataset_sample(
    request: Request,
    file: UploadFile = File(...),
    label: str | None = Form(default=None),
    source: str = Form(default="webcam"),
    needs_review: bool = Form(default=False),
) -> dict[str, Any]:
    """Extract hand + holistic landmarks from an uploaded frame and save to dataset."""
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded frame is empty.")

    resolved_label = label if label is not None else request.query_params.get("label")
    if not resolved_label:
        raise HTTPException(status_code=400, detail="Sign `label` is required.")

    try:
        sample = await asyncio.to_thread(
            _add_sample_sync,
            request.app.state,
            raw_bytes,
            resolved_label,
            source,
            needs_review,
        )
        return {"sample": sample}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@dataset_router.patch("/api/dataset/samples/{sample_id}")
def patch_dataset_sample(
    request: Request,
    sample_id: str,
    payload: SamplePatchPayload,
) -> dict[str, Any]:
    """Relabel or mark a dataset sample as reviewed."""
    store = get_dataset_store(request.app.state)
    try:
        updated = store.update_sample(
            sample_id,
            label=payload.label,
            needs_review=payload.needs_review,
        )
        return {"sample": updated}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@dataset_router.delete("/api/dataset/samples/{sample_id}")
def delete_dataset_sample(request: Request, sample_id: str) -> dict[str, Any]:
    """Remove a dataset sample and its stored image."""
    store = get_dataset_store(request.app.state)
    if not store.delete_sample(sample_id):
        raise HTTPException(status_code=404, detail=f"Sample {sample_id!r} not found.")
    return {"deleted": True, "sample_id": sample_id}


@dataset_router.post("/api/dataset/import-folder")
async def import_dataset_folder(
    request: Request,
    payload: ImportFolderPayload,
) -> dict[str, Any]:
    """Import labeled subdirectories (`<folder>/<label>/*.jpg`) into the landmark dataset."""
    store = get_dataset_store(request.app.state)
    try:
        return await asyncio.to_thread(
            store.import_folder,
            payload.folder_path,
            max_per_class=payload.max_per_class,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@dataset_router.post("/api/dataset/train")
async def train_custom_signs(
    request: Request,
    payload: TrainPayload | None = None,
) -> dict[str, Any]:
    """Train the PyTorch landmark classifier on all stored samples and hot-reload it."""
    epochs = payload.epochs if payload is not None else 45
    try:
        return await asyncio.to_thread(_train_sync, request.app.state, epochs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
