"""Normalized holistic + two-hand landmark feature extractor and PyTorch classifier."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
import threading
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from .config import DEFAULT_CUSTOM_CLASSIFIER_PATH

FEATURE_DIM = 155

_FINGERTIP_INDICES: tuple[int, ...] = (4, 8, 12, 16, 20)
_MCP_INDICES: tuple[int, ...] = (2, 5, 9, 13, 17)
_PINCH_TIP_INDICES: tuple[int, ...] = (8, 12, 16, 20)
_POSE_LEFT_SHOULDER = 11
_POSE_RIGHT_SHOULDER = 12


@dataclass(slots=True)
class CustomPrediction:
    """Single predicted class from the custom landmark MLP."""

    label: str
    score: float
    similarity: float


def _pt_xyz(pt: Any) -> tuple[float, float, float]:
    if isinstance(pt, dict):
        return (
            float(pt.get("x", 0.0) or 0.0),
            float(pt.get("y", 0.0) or 0.0),
            float(pt.get("z", 0.0) or 0.0),
        )
    return (
        float(getattr(pt, "x", 0.0) or 0.0),
        float(getattr(pt, "y", 0.0) or 0.0),
        float(getattr(pt, "z", 0.0) or 0.0),
    )


def _hand_landmarks_array(hand: Any) -> np.ndarray | None:
    lms = hand.get("landmarks") if isinstance(hand, dict) else getattr(hand, "landmarks", None)
    if not lms or len(lms) < 21:
        return None
    coords = np.asarray([_pt_xyz(p) for p in lms[:21]], dtype=np.float32)
    return coords


def _hand_handedness(hand: Any) -> str:
    val = hand.get("handedness") if isinstance(hand, dict) else getattr(hand, "handedness", "Unknown")
    return str(val or "Unknown")


def _assign_hand_slots(
    hands: Sequence[Any],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Assign up to two detected hands into deterministic (Left, Right) slots."""
    parsed: list[tuple[str, np.ndarray]] = []
    for hand in hands:
        arr = _hand_landmarks_array(hand)
        if arr is not None:
            parsed.append((_hand_handedness(hand), arr))
    if not parsed:
        return None, None

    if len(parsed) == 1:
        side, arr = parsed[0]
        return (arr, None) if side == "Left" else (None, arr)

    left_arr: np.ndarray | None = None
    right_arr: np.ndarray | None = None
    unplaced: list[np.ndarray] = []

    for side, arr in parsed[:2]:
        if side == "Left" and left_arr is None:
            left_arr = arr
        elif side == "Right" and right_arr is None:
            right_arr = arr
        else:
            unplaced.append(arr)

    for arr in unplaced:
        if left_arr is None and right_arr is not None:
            left_arr = arr
        elif right_arr is None and left_arr is not None:
            right_arr = arr
        else:
            # Both unassigned: order left-to-right in frame coordinates.
            unplaced_sorted = sorted([arr] + [a for a in unplaced if a is not arr], key=lambda a: float(a[0, 0]))
            left_arr = unplaced_sorted[0]
            right_arr = unplaced_sorted[1] if len(unplaced_sorted) > 1 else None
            break

    return left_arr, right_arr


def _normalize_single_hand(
    coords: np.ndarray | None,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Normalize 21x3 hand landmarks around wrist and extract extension/pinch ratios."""
    if coords is None:
        return (
            np.zeros(63, dtype=np.float32),
            1.0,
            np.zeros(5, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
        )

    wrist = coords[0]
    centered = coords - wrist
    palm_span = float(np.linalg.norm(centered[9, :2]))
    max_span = float(np.max(np.linalg.norm(centered[:, :2], axis=1)))
    scale = max(palm_span, max_span * 0.5, 1e-4)

    normed = np.clip(centered / scale, -4.0, 4.0)
    flat_coords = normed.reshape(-1).astype(np.float32)

    ext_ratios = np.zeros(5, dtype=np.float32)
    for idx, (tip_i, mcp_i) in enumerate(zip(_FINGERTIP_INDICES, _MCP_INDICES)):
        tip_d = float(np.linalg.norm(centered[tip_i, :2]))
        mcp_d = float(np.linalg.norm(centered[mcp_i, :2]))
        ext_ratios[idx] = min(4.0, tip_d / max(mcp_d, 1e-4)) / 4.0

    pinch_dists = np.zeros(4, dtype=np.float32)
    thumb_tip = centered[4, :2]
    for idx, tip_i in enumerate(_PINCH_TIP_INDICES):
        d = float(np.linalg.norm(centered[tip_i, :2] - thumb_tip))
        pinch_dists[idx] = min(3.0, d / scale) / 3.0

    return flat_coords, scale, ext_ratios, pinch_dists


def extract_sign_features(
    hands: Sequence[Any],
    pose_landmarks: Sequence[Any] | None = None,
) -> np.ndarray:
    """Build a 155-dim invariant feature vector from MediaPipe hands + optional pose."""
    left_arr, right_arr = _assign_hand_slots(hands)
    left_flat, left_scale, left_ext, left_pinch = _normalize_single_hand(left_arr)
    right_flat, right_scale, right_ext, right_pinch = _normalize_single_hand(right_arr)

    has_left = 1.0 if left_arr is not None else 0.0
    has_right = 1.0 if right_arr is not None else 0.0
    presence = np.asarray([has_left, has_right], dtype=np.float32)

    inter_hand = np.zeros(4, dtype=np.float32)
    if left_arr is not None and right_arr is not None:
        joint_scale = max(0.5 * (left_scale + right_scale), 1e-4)
        dx_w = (right_arr[0, 0] - left_arr[0, 0]) / joint_scale
        dy_w = (right_arr[0, 1] - left_arr[0, 1]) / joint_scale
        dx_i = (right_arr[8, 0] - left_arr[8, 0]) / joint_scale
        dy_i = (right_arr[8, 1] - left_arr[8, 1]) / joint_scale
        inter_hand = np.clip(
            np.asarray([dx_w, dy_w, dx_i, dy_i], dtype=np.float32) / 4.0,
            -1.0,
            1.0,
        )

    pose_feat = np.zeros(5, dtype=np.float32)
    if pose_landmarks and len(pose_landmarks) > _POSE_RIGHT_SHOULDER:
        ls_x, ls_y, _ = _pt_xyz(pose_landmarks[_POSE_LEFT_SHOULDER])
        rs_x, rs_y, _ = _pt_xyz(pose_landmarks[_POSE_RIGHT_SHOULDER])
        shoulder_width = math.hypot(ls_x - rs_x, ls_y - rs_y)
        if shoulder_width > 1e-4:
            cx = 0.5 * (ls_x + rs_x)
            cy = 0.5 * (ls_y + rs_y)
            pose_feat[0] = 1.0
            if left_arr is not None:
                pose_feat[1] = np.clip((left_arr[0, 0] - cx) / shoulder_width, -3.0, 3.0) / 3.0
                pose_feat[2] = np.clip((left_arr[0, 1] - cy) / shoulder_width, -3.0, 3.0) / 3.0
            if right_arr is not None:
                pose_feat[3] = np.clip((right_arr[0, 0] - cx) / shoulder_width, -3.0, 3.0) / 3.0
                pose_feat[4] = np.clip((right_arr[0, 1] - cy) / shoulder_width, -3.0, 3.0) / 3.0

    vec = np.concatenate(
        [
            left_flat,
            right_flat,
            presence,
            inter_hand,
            pose_feat,
            left_ext,
            right_ext,
            left_pinch,
            right_pinch,
        ]
    ).astype(np.float32)
    assert vec.shape == (FEATURE_DIM,)
    return vec


def augment_feature_vector(vec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply small geometric perturbations to normalized landmark features."""
    out = vec.copy()
    angle = float(rng.uniform(-0.17, 0.17))  # ~ +/-10 deg
    scale = float(rng.uniform(0.93, 1.07))
    ca, sa = math.cos(angle), math.sin(angle)

    for offset, present_idx in ((0, 126), (63, 127)):
        if out[present_idx] > 0.5:
            block = out[offset : offset + 63].reshape(21, 3)
            x = block[:, 0].copy()
            y = block[:, 1].copy()
            block[:, 0] = (ca * x - sa * y) * scale + rng.normal(0.0, 0.015, size=21)
            block[:, 1] = (sa * x + ca * y) * scale + rng.normal(0.0, 0.015, size=21)
            block[:, 2] = block[:, 2] * scale + rng.normal(0.0, 0.015, size=21)
            out[offset : offset + 63] = np.clip(block.reshape(-1), -4.0, 4.0)

    if out[126] > 0.5 and out[127] > 0.5:
        for idx in (128, 130):
            dx = out[idx]
            dy = out[idx + 1]
            out[idx] = np.clip(ca * dx - sa * dy + rng.normal(0.0, 0.01), -1.0, 1.0)
            out[idx + 1] = np.clip(sa * dx + ca * dy + rng.normal(0.0, 0.01), -1.0, 1.0)

    # Randomly mask pose anchor so hands-only images and full-webcam frames both generalize.
    if rng.random() < 0.45:
        out[132:137] = 0.0
    elif out[132] > 0.5:
        out[133:137] = np.clip(
            out[133:137] + rng.normal(0.0, 0.02, size=4),
            -1.0,
            1.0,
        )

    out[137:155] = np.clip(out[137:155] + rng.normal(0.0, 0.015, size=18), 0.0, 1.0)
    return out


class LandmarkSignMLP(nn.Module):
    """Lightweight PyTorch MLP classifier over normalized 155-dim sign landmarks."""

    def __init__(self, input_dim: int = FEATURE_DIM, num_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, max(1, num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_landmark_classifier(
    samples_by_label: dict[str, list[np.ndarray]],
    output_path: Path = DEFAULT_CUSTOM_CLASSIFIER_PATH,
    *,
    epochs: int = 45,
    batch_size: int = 64,
    learning_rate: float = 2e-3,
    seed: int = 42,
) -> dict[str, Any]:
    """Train `LandmarkSignMLP` on extracted landmark vectors and save checkpoint."""
    clean_classes = {
        label: [np.asarray(v, dtype=np.float32) for v in vecs]
        for label, vecs in samples_by_label.items()
        if vecs
    }
    if not clean_classes:
        raise ValueError("Cannot train custom classifier with zero valid samples.")

    labels = sorted(clean_classes.keys())
    label_to_idx = {lbl: idx for idx, lbl in enumerate(labels)}
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # Balance class representation via landmark augmentation.
    target_per_class = max(48, min(160, max(len(v) for v in clean_classes.values()) * 6))
    raw_x: list[np.ndarray] = []
    raw_y: list[int] = []
    orig_x: list[np.ndarray] = []
    orig_y: list[int] = []

    for lbl in labels:
        idx = label_to_idx[lbl]
        vecs = clean_classes[lbl]
        for vec in vecs:
            orig_x.append(vec)
            orig_y.append(idx)
            raw_x.append(vec)
            raw_y.append(idx)

        needed = max(0, target_per_class - len(vecs))
        for k in range(needed):
            base = vecs[k % len(vecs)]
            raw_x.append(augment_feature_vector(base, rng))
            raw_y.append(idx)

    x_train_np = np.stack(raw_x, axis=0).astype(np.float32)
    y_train_np = np.asarray(raw_y, dtype=np.int64)
    x_orig_np = np.stack(orig_x, axis=0).astype(np.float32)
    y_orig_np = np.asarray(orig_y, dtype=np.int64)

    mean = torch.from_numpy(x_train_np.mean(axis=0)).float()
    std = torch.from_numpy(np.maximum(x_train_np.std(axis=0), 1e-2)).float()

    x_train = (torch.from_numpy(x_train_np) - mean) / std
    y_train = torch.from_numpy(y_train_np)
    x_orig = (torch.from_numpy(x_orig_np) - mean) / std
    y_orig = torch.from_numpy(y_orig_np)

    # Compute standardized L2-normalized class centroids for OOD gating.
    centroids = []
    for idx in range(len(labels)):
        cls_vecs = x_orig[y_orig == idx]
        c = cls_vecs.mean(dim=0)
        c = c / c.norm().clamp(min=1e-6)
        centroids.append(c)
    centroid_tensor = torch.stack(centroids, dim=0)

    model = LandmarkSignMLP(input_dim=FEATURE_DIM, num_classes=len(labels))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    model.train()
    n_samples = x_train.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n_samples)
        for start in range(0, n_samples, batch_size):
            batch_idx = perm[start : start + batch_size]
            xb = x_train[batch_idx]
            yb = y_train[batch_idx]
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.inference_mode():
        orig_logits = model(x_orig)
        preds = orig_logits.argmax(dim=1)
        train_accuracy = float((preds == y_orig).float().mean().item())

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f"{target.name}.{threading.get_ident()}.tmp")
    trained_at = datetime.now(timezone.utc).isoformat()
    sample_counts = {lbl: len(clean_classes[lbl]) for lbl in labels}

    checkpoint = {
        "state_dict": model.state_dict(),
        "labels": labels,
        "feature_dim": FEATURE_DIM,
        "feature_mean": mean,
        "feature_std": std,
        "class_centroids": centroid_tensor,
        "trained_at": trained_at,
        "sample_counts": sample_counts,
        "train_accuracy": round(train_accuracy, 4),
        "epochs": epochs,
    }
    try:
        torch.save(checkpoint, tmp_path)
        tmp_path.replace(target)
    finally:
        tmp_path.unlink(missing_ok=True)

    return {
        "model_path": str(target),
        "num_classes": len(labels),
        "labels": labels,
        "total_original_samples": len(orig_y),
        "total_augmented_samples": int(n_samples),
        "train_accuracy": round(train_accuracy, 4),
        "epochs": epochs,
        "trained_at": trained_at,
        "sample_counts": sample_counts,
    }


class CustomSignClassifier:
    """Thread-safe hot-reloading wrapper around a trained `LandmarkSignMLP` checkpoint."""

    def __init__(self, model_path: Path = DEFAULT_CUSTOM_CLASSIFIER_PATH) -> None:
        self.model_path = Path(model_path)
        self._lock = threading.Lock()
        self._mtime_ns: int | None = None
        self._model: LandmarkSignMLP | None = None
        self._labels: list[str] = []
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None
        self._centroids: torch.Tensor | None = None
        self._metadata: dict[str, Any] = {}
        self.maybe_reload()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and bool(self._labels)

    def status(self) -> dict[str, Any]:
        """Return lightweight status metadata for health and dataset panels."""
        self.maybe_reload()
        with self._lock:
            return {
                "loaded": self.is_loaded,
                "model_path": str(self.model_path),
                "num_classes": len(self._labels),
                "labels": list(self._labels),
                **self._metadata,
            }

    def maybe_reload(self) -> bool:
        """Load or hot-reload checkpoint if it appeared or changed on disk."""
        if not self.model_path.is_file():
            with self._lock:
                self._model = None
                self._labels = []
                self._mtime_ns = None
                self._metadata = {}
            return False

        stat_mtime = self.model_path.stat().st_mtime_ns
        with self._lock:
            if self._mtime_ns == stat_mtime and self._model is not None:
                return True

            ckpt = torch.load(self.model_path, map_location="cpu", weights_only=True)
            labels = list(ckpt["labels"])
            feature_dim = int(ckpt.get("feature_dim", FEATURE_DIM))
            model = LandmarkSignMLP(input_dim=feature_dim, num_classes=len(labels))
            model.load_state_dict(ckpt["state_dict"])
            model.eval()

            self._model = model
            self._labels = labels
            self._mean = ckpt["feature_mean"].float()
            self._std = ckpt["feature_std"].float().clamp(min=1e-3)
            self._centroids = ckpt["class_centroids"].float()
            self._mtime_ns = stat_mtime
            self._metadata = {
                "trained_at": ckpt.get("trained_at"),
                "train_accuracy": ckpt.get("train_accuracy"),
                "epochs": ckpt.get("epochs"),
                "sample_counts": ckpt.get("sample_counts", {}),
            }
            return True

    def predict(
        self,
        features: np.ndarray,
        *,
        top_k: int = 3,
        min_similarity: float = 0.0,
    ) -> list[CustomPrediction]:
        """Predict top-K custom sign candidates from a 155-dim feature vector."""
        if not self.maybe_reload():
            return []

        with self._lock:
            if (
                self._model is None
                or self._mean is None
                or self._std is None
                or self._centroids is None
                or not self._labels
            ):
                return []

            vec = torch.from_numpy(np.asarray(features, dtype=np.float32))
            normed = (vec - self._mean) / self._std
            unit = normed / normed.norm().clamp(min=1e-6)
            sims = torch.matmul(self._centroids, unit)

            with torch.inference_mode():
                logits = self._model(normed.unsqueeze(0)).squeeze(0)
                if len(self._labels) == 1:
                    probs = torch.tensor([float(max(0.0, sims[0].item()))])
                else:
                    probs = torch.softmax(logits, dim=0)

            k = min(max(1, top_k), len(self._labels))
            top_probs, top_indices = torch.topk(probs, k=k)
            results: list[CustomPrediction] = []
            for p, idx in zip(top_probs.tolist(), top_indices.tolist()):
                sim = float(sims[idx].item())
                score = float(p) if sim >= min_similarity else min(float(p), max(0.0, sim))
                results.append(
                    CustomPrediction(
                        label=self._labels[idx],
                        score=score,
                        similarity=sim,
                    )
                )
            return results
