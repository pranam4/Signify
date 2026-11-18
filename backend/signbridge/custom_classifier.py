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
ENRICHED_HAND_DIM = 24
ENRICHED_EXTRA_DIM = 4
ENRICHED_FEATURE_DIM = FEATURE_DIM + (ENRICHED_HAND_DIM * 2) + ENRICHED_EXTRA_DIM  # 207

_FINGERTIP_INDICES: tuple[int, ...] = (4, 8, 12, 16, 20)
_MCP_INDICES: tuple[int, ...] = (2, 5, 9, 13, 17)
_PINCH_TIP_INDICES: tuple[int, ...] = (8, 12, 16, 20)
_POSE_LEFT_SHOULDER = 11
_POSE_RIGHT_SHOULDER = 12
_FINGER_CHAINS: tuple[tuple[int, int, int, int], ...] = (
    (1, 2, 3, 4),      # Thumb
    (5, 6, 7, 8),      # Index
    (9, 10, 11, 12),   # Middle
    (13, 14, 15, 16),  # Ring
    (17, 18, 19, 20),  # Pinky
)

# Slice groups corresponding to Left-only, Right-only, Two-hand, and Pose features.
_LEFT_ONLY_SLICES: tuple[slice, ...] = (
    slice(0, 63),
    slice(137, 142),
    slice(147, 151),
    slice(155, 155 + ENRICHED_HAND_DIM),
)
_RIGHT_ONLY_SLICES: tuple[slice, ...] = (
    slice(63, 126),
    slice(142, 147),
    slice(151, 155),
    slice(155 + ENRICHED_HAND_DIM, 155 + 2 * ENRICHED_HAND_DIM),
)
_BOTH_HANDS_SLICES: tuple[slice, ...] = (
    slice(128, 132),
    slice(155 + 2 * ENRICHED_HAND_DIM, ENRICHED_FEATURE_DIM),
)
_POSE_SLICES: tuple[slice, ...] = (slice(132, 137),)


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


def mirror_feature_vector(vec: np.ndarray) -> np.ndarray:
    """Horizontally mirror a 155-dim feature vector (swap Left/Right slots and negate x)."""
    out = vec.copy()
    left_block = out[0:63].reshape(21, 3).copy()
    right_block = out[63:126].reshape(21, 3).copy()
    left_block[:, 0] *= -1.0
    right_block[:, 0] *= -1.0
    out[0:63] = right_block.reshape(-1)
    out[63:126] = left_block.reshape(-1)

    out[126], out[127] = out[127], out[126]
    # Swapping Left and Right reverses (right - left) sign in both x and y!
    out[128:132] *= -1.0

    # Pose anchor: swap left/right wrist relative coords and negate x.
    lw_x, lw_y = out[133], out[134]
    rw_x, rw_y = out[135], out[136]
    out[133], out[134] = -rw_x, rw_y
    out[135], out[136] = -lw_x, lw_y

    # Swap extension ratios (137..142 <-> 142..147) and pinch distances (147..151 <-> 151..155).
    l_ext, r_ext = out[137:142].copy(), out[142:147].copy()
    out[137:142], out[142:147] = r_ext, l_ext
    l_pin, r_pin = out[147:151].copy(), out[151:155].copy()
    out[147:151], out[151:155] = r_pin, l_pin
    return out


def _hand_geometric_extras(flat_block: np.ndarray, present: bool) -> np.ndarray:
    """Compute 24 rotation/scale-invariant finger flexion, spread, and palm orientation cues."""
    out = np.zeros(ENRICHED_HAND_DIM, dtype=np.float32)
    if not present:
        return out

    pts = flat_block.reshape(21, 3).astype(np.float32).copy()
    # Soften noisy camera Z depth relative to 2D projected hand geometry.
    pts[:, 2] *= 0.45

    # 10 flexion bend cosines across proximal & distal finger links (5 * 2)
    idx = 0
    for a, b, c, d in _FINGER_CHAINS:
        base_dir = pts[b] - pts[0] if a == 1 else pts[b] - pts[0]
        seg1 = pts[c] - pts[b]
        seg2 = pts[d] - pts[c]
        n0 = float(np.linalg.norm(base_dir))
        n1 = float(np.linalg.norm(seg1))
        n2 = float(np.linalg.norm(seg2))
        cos1 = float(np.dot(base_dir, seg1) / (n0 * n1)) if n0 > 1e-4 and n1 > 1e-4 else 0.0
        cos2 = float(np.dot(seg1, seg2) / (n1 * n2)) if n1 > 1e-4 and n2 > 1e-4 else 0.0
        out[idx] = np.clip(cos1, -1.0, 1.0)
        out[idx + 1] = np.clip(cos2, -1.0, 1.0)
        idx += 2

    # 4 adjacent fingertip gaps + 2 cross gaps (thumb-middleMCP, indexTip-pinkyTip)
    palm_span = max(float(np.linalg.norm(pts[9, :2])), 0.25)
    tip_pairs = ((4, 8), (8, 12), (12, 16), (16, 20), (4, 9), (8, 20))
    for offset_i, (p_a, p_b) in enumerate(tip_pairs):
        d = float(np.linalg.norm(pts[p_a, :2] - pts[p_b, :2])) / palm_span
        out[idx + offset_i] = min(3.0, d) / 3.0
    idx += 6

    # Palm normal (cross product of wrist->index_mcp and wrist->pinky_mcp) + middle finger axis
    v_idx = pts[5] - pts[0]
    v_pnk = pts[17] - pts[0]
    palm_norm = np.cross(v_idx, v_pnk)
    pn_len = float(np.linalg.norm(palm_norm))
    if pn_len > 1e-5:
        out[idx : idx + 3] = (palm_norm / pn_len).astype(np.float32)
    idx += 3

    v_mid = pts[9] - pts[0]
    vm_len = float(np.linalg.norm(v_mid))
    if vm_len > 1e-5:
        out[idx : idx + 3] = (v_mid / vm_len).astype(np.float32)
    idx += 3

    # Compact spread & thumb-across-palm opposition scalar
    tips_2d = pts[list(_FINGERTIP_INDICES), :2]
    tip_radii = np.linalg.norm(tips_2d, axis=1) / palm_span
    out[idx] = float(np.clip(np.std(tip_radii), 0.0, 1.5)) / 1.5
    out[idx + 1] = float(np.clip(np.linalg.norm(pts[4, :2] - pts[17, :2]) / (2.5 * palm_span), 0.0, 1.0))
    return out


def enrich_sign_features(vec_155: np.ndarray) -> np.ndarray:
    """Expand a 155-dim base vector into a 207-dim domain-invariant feature vector."""
    base = np.asarray(vec_155, dtype=np.float32).reshape(-1)
    if base.shape[0] != FEATURE_DIM:
        raise ValueError(f"Expected {FEATURE_DIM}-dim vector, got {base.shape[0]}.")

    out_base = base.copy()
    has_left = bool(out_base[126] > 0.5)
    has_right = bool(out_base[127] > 0.5)

    # Attenuate depth Z coordinates slightly to reduce webcam depth scale drift.
    if has_left:
        lb = out_base[0:63].reshape(21, 3)
        lb[:, 2] *= 0.5
        out_base[0:63] = lb.reshape(-1)
    if has_right:
        rb = out_base[63:126].reshape(21, 3)
        rb[:, 2] *= 0.5
        out_base[63:126] = rb.reshape(-1)

    left_extra = _hand_geometric_extras(base[0:63], has_left)
    right_extra = _hand_geometric_extras(base[63:126], has_right)

    pair_extra = np.zeros(ENRICHED_EXTRA_DIM, dtype=np.float32)
    if has_left and has_right:
        lw_dist = float(math.hypot(base[128], base[129]))
        li_dist = float(math.hypot(base[130], base[131]))
        pair_extra[0] = min(1.0, lw_dist)
        pair_extra[1] = min(1.0, li_dist)
        pair_extra[2] = float(np.dot(left_extra[16:19], right_extra[16:19]))  # Palm alignment
        pair_extra[3] = float(np.dot(left_extra[19:22], right_extra[19:22]))  # Finger axis alignment

    enriched = np.concatenate([out_base, left_extra, right_extra, pair_extra]).astype(np.float32)
    assert enriched.shape == (ENRICHED_FEATURE_DIM,)
    return enriched


def _active_slot_mask_from_base(vec_155: np.ndarray) -> np.ndarray:
    """Zero out standardized features belonging to absent hands or absent torso pose."""
    mask = np.ones(ENRICHED_FEATURE_DIM, dtype=np.float32)
    has_left = bool(vec_155[126] > 0.5)
    has_right = bool(vec_155[127] > 0.5)
    has_pose = bool(vec_155[132] > 0.5)

    if not has_left:
        for sl in _LEFT_ONLY_SLICES:
            mask[sl] = 0.0
    if not has_right:
        for sl in _RIGHT_ONLY_SLICES:
            mask[sl] = 0.0
    if not (has_left and has_right):
        for sl in _BOTH_HANDS_SLICES:
            mask[sl] = 0.0
    if not has_pose:
        for sl in _POSE_SLICES:
            mask[sl] = 0.0
    if not has_left and not has_right:
        mask[126:128] = 0.0
    return mask


def augment_feature_vector(vec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply camera-realistic geometric perturbations, 3D tilt, z-jitter, and mirroring."""
    out = mirror_feature_vector(vec) if rng.random() < 0.45 else vec.copy()
    angle = float(rng.uniform(-0.38, 0.38))  # ~ +/-22 deg wrist tilt
    scale_x = float(rng.uniform(0.88, 1.09))
    scale_y = float(rng.uniform(0.88, 1.09))
    z_scale = float(rng.uniform(0.50, 1.40))
    ca, sa = math.cos(angle), math.sin(angle)

    for offset, present_idx in ((0, 126), (63, 127)):
        if out[present_idx] > 0.5:
            block = out[offset : offset + 63].reshape(21, 3)
            x = block[:, 0].copy() * scale_x
            y = block[:, 1].copy() * scale_y
            block[:, 0] = (ca * x - sa * y) + rng.normal(0.0, 0.022, size=21)
            block[:, 1] = (sa * x + ca * y) + rng.normal(0.0, 0.022, size=21)
            block[:, 2] = block[:, 2] * z_scale + rng.normal(0.0, 0.025, size=21)
            out[offset : offset + 63] = np.clip(block.reshape(-1), -4.0, 4.0)

    if out[126] > 0.5 and out[127] > 0.5:
        for idx in (128, 130):
            dx = out[idx] * scale_x
            dy = out[idx + 1] * scale_y
            out[idx] = np.clip(ca * dx - sa * dy + rng.normal(0.0, 0.02), -1.0, 1.0)
            out[idx + 1] = np.clip(sa * dx + ca * dy + rng.normal(0.0, 0.02), -1.0, 1.0)

    # Prevent pose_valid from becoming a dataset-source shortcut:
    # - Add synthetic webcam torso offsets to close-up/hands-only samples (55%)
    # - Randomly mask torso pose on full-body samples (40%)
    if out[132] < 0.5:
        if rng.random() < 0.55:
            out[132] = 1.0
            if out[126] > 0.5:
                out[133] = float(rng.uniform(-0.15, 0.35))
                out[134] = float(rng.uniform(-0.20, 0.25))
            if out[127] > 0.5:
                out[135] = float(rng.uniform(-0.35, 0.15))
                out[136] = float(rng.uniform(-0.20, 0.25))
    elif rng.random() < 0.40:
        out[132:137] = 0.0
    else:
        shift = rng.normal(0.0, 0.06, size=4)
        out[133:137] = np.clip(out[133:137] + shift, -1.0, 1.0)

    out[137:155] = np.clip(out[137:155] + rng.normal(0.0, 0.022, size=18), 0.0, 1.0)
    return out


class LandmarkSignMLP(nn.Module):
    """Lightweight PyTorch MLP classifier over normalized/enriched sign landmarks."""

    def __init__(
        self,
        input_dim: int = ENRICHED_FEATURE_DIM,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 320),
            nn.LayerNorm(320),
            nn.GELU(),
            nn.Dropout(0.22),
            nn.Linear(320, 160),
            nn.LayerNorm(160),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(160, max(1, num_classes)),
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
    """Train `LandmarkSignMLP` on enriched landmark vectors and save checkpoint."""
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

    # Balance class representation via camera-domain landmark augmentation.
    target_per_class = max(56, min(180, max(len(v) for v in clean_classes.values()) * 6))
    raw_base_x: list[np.ndarray] = []
    raw_y: list[int] = []
    orig_base_x: list[np.ndarray] = []
    orig_y: list[int] = []

    for lbl in labels:
        idx = label_to_idx[lbl]
        vecs = clean_classes[lbl]
        for vec in vecs:
            orig_base_x.append(vec)
            orig_y.append(idx)
            raw_base_x.append(vec)
            raw_y.append(idx)

        needed = max(0, target_per_class - len(vecs))
        for k in range(needed):
            base = vecs[k % len(vecs)]
            raw_base_x.append(augment_feature_vector(base, rng))
            raw_y.append(idx)

    raw_enriched = [enrich_sign_features(v) for v in raw_base_x]
    orig_enriched = [enrich_sign_features(v) for v in orig_base_x]

    x_train_np = np.stack(raw_enriched, axis=0).astype(np.float32)
    y_train_np = np.asarray(raw_y, dtype=np.int64)
    x_orig_np = np.stack(orig_enriched, axis=0).astype(np.float32)
    y_orig_np = np.asarray(orig_y, dtype=np.int64)
    orig_masks_np = np.stack([_active_slot_mask_from_base(v) for v in orig_base_x], axis=0).astype(np.float32)

    mean = torch.from_numpy(x_train_np.mean(axis=0)).float()
    std = torch.from_numpy(np.maximum(x_train_np.std(axis=0), 0.04)).float()

    x_train = (torch.from_numpy(x_train_np) - mean) / std
    y_train = torch.from_numpy(y_train_np)
    x_orig = (torch.from_numpy(x_orig_np) - mean) / std
    y_orig = torch.from_numpy(y_orig_np)
    orig_masks = torch.from_numpy(orig_masks_np)

    # Compute active-slot masked L2-normalized class centroids for unbiased OOD gating.
    centroids = []
    for idx in range(len(labels)):
        sel = y_orig == idx
        cls_vecs = x_orig[sel] * orig_masks[sel]
        # Average unit vectors across real + mirrored variants so both L and R match cleanly.
        cls_norms = cls_vecs.norm(dim=1, keepdim=True).clamp(min=1e-6)
        c = (cls_vecs / cls_norms).mean(dim=0)
        c = c / c.norm().clamp(min=1e-6)
        centroids.append(c)
    centroid_tensor = torch.stack(centroids, dim=0)

    model = LandmarkSignMLP(input_dim=ENRICHED_FEATURE_DIM, num_classes=len(labels))
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
    # Measure original (unmirrored) accuracy over input base samples
    real_orig_x = torch.stack(
        [(torch.from_numpy(enrich_sign_features(v)) - mean) / std for lbl in labels for v in clean_classes[lbl]],
        dim=0,
    )
    real_orig_y = torch.tensor(
        [label_to_idx[lbl] for lbl in labels for _ in clean_classes[lbl]],
        dtype=torch.long,
    )
    with torch.inference_mode():
        orig_logits = model(real_orig_x)
        preds = orig_logits.argmax(dim=1)
        train_accuracy = float((preds == real_orig_y).float().mean().item())

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f"{target.name}.{threading.get_ident()}.tmp")
    trained_at = datetime.now(timezone.utc).isoformat()
    sample_counts = {lbl: len(clean_classes[lbl]) for lbl in labels}

    checkpoint = {
        "state_dict": model.state_dict(),
        "labels": labels,
        "feature_dim": ENRICHED_FEATURE_DIM,
        "feature_mode": "enriched_v2",
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
        import time as _time
        for attempt in range(8):
            try:
                tmp_path.replace(target)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                _time.sleep(0.1)
    finally:
        tmp_path.unlink(missing_ok=True)

    return {
        "model_path": str(target),
        "num_classes": len(labels),
        "labels": labels,
        "total_original_samples": int(real_orig_y.shape[0]),
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
        self._feature_mode: str = "raw_155"
        self._feature_dim: int = FEATURE_DIM
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
                "feature_mode": self._feature_mode,
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
            feature_mode = str(ckpt.get("feature_mode", "raw_155"))
            model = LandmarkSignMLP(input_dim=feature_dim, num_classes=len(labels))
            model.load_state_dict(ckpt["state_dict"])
            model.eval()

            self._model = model
            self._labels = labels
            self._feature_dim = feature_dim
            self._feature_mode = feature_mode
            self._mean = ckpt["feature_mean"].float()
            self._std = ckpt["feature_std"].float().clamp(min=0.04)
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

            base = np.asarray(features, dtype=np.float32).reshape(-1)
            # Empty frame (no hands detected at all) -> strict zero similarity/scores.
            if base[126] < 0.5 and base[127] < 0.5:
                return [
                    CustomPrediction(label=self._labels[0], score=0.0, similarity=0.0)
                ]

            if self._feature_mode == "enriched_v2" and self._feature_dim == ENRICHED_FEATURE_DIM:
                vec_np = enrich_sign_features(base)
                mask = torch.from_numpy(_active_slot_mask_from_base(base))
                mir_base = mirror_feature_vector(base)
                mir_np = enrich_sign_features(mir_base)
                mir_mask = torch.from_numpy(_active_slot_mask_from_base(mir_base))
            else:
                vec_np = base
                mask = torch.ones_like(torch.from_numpy(base))
                mir_np = None
                mir_mask = None

            vec = torch.from_numpy(vec_np)
            normed = (vec - self._mean) / self._std
            masked_normed = normed * mask
            mn_norm = float(masked_normed.norm().item())
            if mn_norm <= 1e-6:
                sims = torch.zeros(len(self._labels), dtype=torch.float32)
            else:
                unit = masked_normed / mn_norm
                sims = torch.matmul(self._centroids, unit)
                if mir_np is not None and mir_mask is not None:
                    mir_normed = ((torch.from_numpy(mir_np) - self._mean) / self._std) * mir_mask
                    m_norm = float(mir_normed.norm().item())
                    if m_norm > 1e-6:
                        sims_mir = torch.matmul(self._centroids, mir_normed / m_norm)
                        sims = torch.maximum(sims, sims_mir)

            with torch.inference_mode():
                logits = self._model(normed.unsqueeze(0)).squeeze(0)
                if len(self._labels) == 1:
                    probs = torch.tensor([float(max(0.0, sims[0].item()))])
                else:
                    mlp_probs = torch.softmax(logits, dim=0)
                    if len(self._labels) >= 6:
                        proto_probs = torch.softmax(sims * 9.0, dim=0)
                        probs = 0.78 * mlp_probs + 0.22 * proto_probs
                    else:
                        probs = mlp_probs

            k = min(max(1, top_k), len(self._labels))
            top_probs, top_indices = torch.topk(probs, k=k)
            results: list[CustomPrediction] = []
            for p, idx in zip(top_probs.tolist(), top_indices.tolist()):
                sim = float(sims[idx].item())
                if sim >= min_similarity:
                    score = float(p)
                else:
                    # Smoothly damp out-of-distribution hands well below confidence thresholds.
                    score = min(float(p), max(0.0, sim * 0.55))
                results.append(
                    CustomPrediction(
                        label=self._labels[idx],
                        score=score,
                        similarity=sim,
                    )
                )
            return results
