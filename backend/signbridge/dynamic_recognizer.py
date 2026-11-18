"""Real-time spatio-temporal recognizer for moving / dynamic ISL signs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from .config import BACKEND_DIR, DEFAULT_DYNAMIC_CLASSIFIER_PATH
from .custom_classifier import _assign_hand_slots, _normalize_single_hand

DEFAULT_SIGN_ANIMATIONS_PATH = (
    BACKEND_DIR.parent / "frontend" / "src" / "utils" / "signAnimations.js"
)

K_FRAME_DIM = 46
TEMPORAL_STEPS = 8
TRAJ_SUMMARY_DIM = 28
DYNAMIC_FEATURE_DIM = (TEMPORAL_STEPS * K_FRAME_DIM) + TRAJ_SUMMARY_DIM  # 396
IDLE_MOTION_LABEL = "_idle"

_POSE_NOSE = 0
_POSE_LEFT_SHOULDER = 11
_POSE_RIGHT_SHOULDER = 12
_FACE_NOSE_TIP = 1


@dataclass(slots=True)
class DynamicPrediction:
    """Single top-K class candidate from `DynamicSignClassifier`."""

    label: str
    score: float
    similarity: float


@dataclass(slots=True)
class DynamicTrigger:
    """Committed real-time moving-sign event emitted by `TemporalSignTracker`."""

    label: str
    score: float
    similarity: float
    window_ms: int
    candidates: list[DynamicPrediction]


def _pt_xyv(pt: Any) -> tuple[float, float, float]:
    if isinstance(pt, dict):
        x = float(pt.get("x", 0.0) or 0.0)
        y = float(pt.get("y", 0.0) or 0.0)
        v_raw = pt.get("v", pt.get("visibility", 1.0))
    else:
        x = float(getattr(pt, "x", 0.0) or 0.0)
        y = float(getattr(pt, "y", 0.0) or 0.0)
        v_raw = getattr(pt, "visibility", 1.0)
    vis = 1.0 if v_raw is None else float(v_raw)
    return x, y, vis


def _hand_kinematics_block(
    coords: np.ndarray | None,
    cx: float,
    cy: float,
    nose_x: float,
    nose_y: float,
    sh_width: float,
) -> tuple[np.ndarray, float, float]:
    """Build 18-dim kinematic block + image wrist (x, y) for one hand."""
    block = np.zeros(18, dtype=np.float32)
    if coords is None:
        return block, 0.0, 0.0

    wrist = coords[0]
    rel_sx = float(np.clip((wrist[0] - cx) / sh_width, -2.5, 2.5))
    rel_sy = float(np.clip((wrist[1] - cy) / sh_width, -2.5, 2.5))
    rel_fx = float(np.clip((wrist[0] - nose_x) / sh_width, -2.5, 2.5))
    rel_fy = float(np.clip((wrist[1] - nose_y) / sh_width, -2.5, 2.5))

    _, palm_span, ext_ratios, pinch_dists = _normalize_single_hand(coords)
    hand_scale = float(np.clip(palm_span / sh_width, 0.0, 2.0))

    v_idx = coords[5] - wrist
    v_pnk = coords[17] - wrist
    palm_norm = np.cross(v_idx, v_pnk)
    pn = float(np.linalg.norm(palm_norm))
    if pn > 1e-5:
        palm_norm = palm_norm / pn
    else:
        palm_norm = np.zeros(3, dtype=np.float32)

    v_mid = coords[9] - wrist
    vm = float(np.linalg.norm(v_mid))
    if vm > 1e-5:
        finger_axis = v_mid / vm
    else:
        finger_axis = np.zeros(3, dtype=np.float32)

    block[0] = rel_sx
    block[1] = rel_sy
    block[2] = rel_fx
    block[3] = rel_fy
    block[4] = hand_scale
    block[5:10] = ext_ratios
    block[10:12] = pinch_dists[:2]
    block[12:15] = np.clip(palm_norm, -1.0, 1.0)
    block[15:18] = np.clip(finger_axis, -1.0, 1.0)
    return block, float(wrist[0]), float(wrist[1])


def extract_kinematic_frame(
    hands: Sequence[Any],
    *,
    pose_landmarks: Sequence[Any] | None = None,
    face_landmarks: Sequence[Any] | None = None,
    head_yaw_ratio: float = 0.0,
    head_pitch_ratio: float = 0.0,
) -> np.ndarray:
    """Extract a 46-dim torso+face-anchored kinematic frame descriptor."""
    left_arr, right_arr = _assign_hand_slots(hands)
    has_left = left_arr is not None
    has_right = right_arr is not None

    cx, cy, sh_width = 0.5, 0.52, 0.34
    pose_valid = 0.0
    nose_x, nose_y = 0.5, 0.30

    if pose_landmarks and len(pose_landmarks) > _POSE_RIGHT_SHOULDER:
        ls_x, ls_y, ls_v = _pt_xyv(pose_landmarks[_POSE_LEFT_SHOULDER])
        rs_x, rs_y, rs_v = _pt_xyv(pose_landmarks[_POSE_RIGHT_SHOULDER])
        sw = abs(ls_x - rs_x)
        if ls_v >= 0.35 and rs_v >= 0.35 and 0.08 <= sw <= 0.90:
            cx = 0.5 * (ls_x + rs_x)
            cy = 0.5 * (ls_y + rs_y)
            sh_width = max(0.08, sw)
            pose_valid = 1.0
            nx, ny, nv = _pt_xyv(pose_landmarks[_POSE_NOSE])
            if nv >= 0.35:
                nose_x, nose_y = nx, ny
            else:
                nose_x, nose_y = cx, cy - 0.55 * sh_width

    if face_landmarks and len(face_landmarks) > _FACE_NOSE_TIP:
        fn_x, fn_y, _ = _pt_xyv(face_landmarks[_FACE_NOSE_TIP])
        if 0.01 <= fn_x <= 0.99 and 0.01 <= fn_y <= 0.99:
            nose_x, nose_y = fn_x, fn_y

    left_blk, lwx, lwy = _hand_kinematics_block(
        left_arr, cx, cy, nose_x, nose_y, sh_width
    )
    right_blk, rwx, rwy = _hand_kinematics_block(
        right_arr, cx, cy, nose_x, nose_y, sh_width
    )

    kframe = np.zeros(K_FRAME_DIM, dtype=np.float32)
    kframe[0] = 1.0 if has_left else 0.0
    kframe[1] = 1.0 if has_right else 0.0
    kframe[2:20] = left_blk
    kframe[20:38] = right_blk

    if has_left and has_right:
        dx = float(np.clip((lwx - rwx) / sh_width, -2.5, 2.5))
        dy = float(np.clip((lwy - rwy) / sh_width, -2.5, 2.5))
        kframe[38] = dx
        kframe[39] = dy
        kframe[40] = min(3.0, math.hypot(dx, dy)) / 3.0

    kframe[41] = float(np.clip(head_yaw_ratio, -1.0, 1.0))
    kframe[42] = float(np.clip(head_pitch_ratio, -1.0, 1.0))
    kframe[43] = pose_valid

    active_ys = []
    if has_left:
        active_ys.append(left_blk[1])
    if has_right:
        active_ys.append(right_blk[1])
    if active_ys:
        kframe[44] = float(sum(active_ys) / len(active_ys))
    if has_left and has_right:
        kframe[45] = abs(left_blk[1] - right_blk[1])
    return kframe


def mirror_kinematic_frame(kframe: np.ndarray) -> np.ndarray:
    """Horizontally mirror one 46-dim kinematic frame (swap Left <-> Right, flip x)."""
    out = kframe.copy()
    l_blk = out[2:20].copy()
    r_blk = out[20:38].copy()
    # Flip shoulder rel_x, face rel_x, palm normal x, and finger axis x.
    for blk in (l_blk, r_blk):
        blk[0] *= -1.0
        blk[2] *= -1.0
        blk[12] *= -1.0
        blk[15] *= -1.0
    out[0], out[1] = out[1], out[0]
    out[2:20] = r_blk
    out[20:38] = l_blk
    out[38] *= -1.0
    out[41] *= -1.0
    return out


def _count_1d_reversals(vals: np.ndarray, min_step: float = 0.08) -> int:
    if len(vals) < 3:
        return 0
    rev = 0
    direction = 0
    extreme = float(vals[0])
    for v_raw in vals[1:]:
        v = float(v_raw)
        diff = v - extreme
        if direction == 0:
            if abs(diff) >= min_step:
                direction = 1 if diff > 0 else -1
                extreme = v
        elif direction == 1:
            if v > extreme:
                extreme = v
            elif extreme - v >= min_step:
                rev += 1
                direction = -1
                extreme = v
        else:
            if v < extreme:
                extreme = v
            elif v - extreme >= min_step:
                rev += 1
                direction = 1
                extreme = v
    return rev


def _hand_trajectory_summary(
    seq: np.ndarray,
    *,
    present_col: int,
    start_col: int,
) -> tuple[np.ndarray, float, float]:
    """Compute 9-dim trajectory summary for one hand slot over resampled sequence."""
    out = np.zeros(9, dtype=np.float32)
    mask = seq[:, present_col] > 0.5
    if int(mask.sum()) < 2:
        return out, 0.0, 0.0

    xs = seq[mask, start_col]
    ys = seq[mask, start_col + 1]
    fx = seq[mask, start_col + 2]
    fy = seq[mask, start_col + 3]
    exts = seq[mask, start_col + 5 : start_col + 10]

    net_dx = float(xs[-1] - xs[0])
    net_dy = float(ys[-1] - ys[0])
    steps = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
    path_len = float(steps.sum())
    net_dist = math.hypot(net_dx, net_dy)
    eff = net_dist / max(path_len, 1e-4) if path_len > 1e-4 else 0.0

    x_rev = _count_1d_reversals(xs, min_step=0.075)
    y_rev = _count_1d_reversals(ys, min_step=0.075)
    nose_dists = np.sqrt(fx * fx + fy * fy)
    min_nose = float(nose_dists.min())
    mean_y = float(ys.mean())
    shape_delta = float(np.linalg.norm(exts[-1] - exts[0]))

    out[0] = float(np.clip(net_dx, -2.5, 2.5))
    out[1] = float(np.clip(net_dy, -2.5, 2.5))
    out[2] = float(np.clip(path_len, 0.0, 5.0)) / 5.0
    out[3] = float(np.clip(eff, 0.0, 1.0))
    out[4] = min(1.0, x_rev / 4.0)
    out[5] = min(1.0, y_rev / 4.0)
    out[6] = min(1.0, min_nose / 2.5)
    out[7] = float(np.clip(mean_y, -2.0, 2.0)) / 2.0
    out[8] = min(1.0, shape_delta / 1.5)
    span_x = float(xs.max() - xs.min())
    span_y = float(ys.max() - ys.min())
    return out, span_x, span_y


def build_temporal_descriptor(
    frames: Sequence[tuple[int, np.ndarray]],
) -> tuple[np.ndarray, dict[str, float]]:
    """Resample a timestamped sequence to 8 steps + 28 kinematics -> 396-dim descriptor."""
    if len(frames) < 2:
        return np.zeros(DYNAMIC_FEATURE_DIM, dtype=np.float32), {
            "duration_s": 0.0,
            "path_len": 0.0,
            "max_speed": 0.0,
            "end_speed": 0.0,
            "shape_delta": 0.0,
            "osc_count": 0.0,
            "head_span": 0.0,
        }

    ts_arr = np.asarray([float(ts) for ts, _ in frames], dtype=np.float64)
    feats_arr = np.stack([np.asarray(f, dtype=np.float32) for _, f in frames], axis=0)
    t0, t1 = float(ts_arr[0]), float(ts_arr[-1])
    duration_s = max((t1 - t0) / 1000.0, 1e-3)

    target_ts = np.linspace(t0, t1, TEMPORAL_STEPS, dtype=np.float64)
    resampled = np.zeros((TEMPORAL_STEPS, K_FRAME_DIM), dtype=np.float32)
    for col in range(K_FRAME_DIM):
        resampled[:, col] = np.interp(target_ts, ts_arr, feats_arr[:, col]).astype(np.float32)
    # Snap interpolated presence flags cleanly.
    resampled[:, 0] = (resampled[:, 0] >= 0.5).astype(np.float32)
    resampled[:, 1] = (resampled[:, 1] >= 0.5).astype(np.float32)
    resampled[:, 43] = (resampled[:, 43] >= 0.5).astype(np.float32)

    l_sum, l_sx, l_sy = _hand_trajectory_summary(resampled, present_col=0, start_col=2)
    r_sum, r_sx, r_sy = _hand_trajectory_summary(resampled, present_col=1, start_col=20)

    summary = np.zeros(TRAJ_SUMMARY_DIM, dtype=np.float32)
    summary[0:9] = l_sum
    summary[9:18] = r_sum

    both_mask = (resampled[:, 0] > 0.5) & (resampled[:, 1] > 0.5)
    if int(both_mask.sum()) >= 2:
        dists = resampled[both_mask, 40] * 3.0
        summary[18] = float(dists[0]) / 3.0
        summary[19] = float(dists[-1]) / 3.0
        summary[20] = float(np.clip((dists[-1] - dists[0]) / 2.0, -1.0, 1.0))
        summary[21] = float(dists.min()) / 3.0
        summary[22] = min(1.0, _count_1d_reversals(dists, min_step=0.07) / 4.0)

    # Speed metrics in shoulder-widths per second across raw timestamps.
    step_speeds: list[float] = []
    for idx in range(1, len(frames)):
        dt_i = max((ts_arr[idx] - ts_arr[idx - 1]) / 1000.0, 0.016)
        disp = 0.0
        if feats_arr[idx, 1] > 0.5 and feats_arr[idx - 1, 1] > 0.5:
            disp = max(
                disp,
                math.hypot(
                    float(feats_arr[idx, 20] - feats_arr[idx - 1, 20]),
                    float(feats_arr[idx, 21] - feats_arr[idx - 1, 21]),
                ),
            )
        if feats_arr[idx, 0] > 0.5 and feats_arr[idx - 1, 0] > 0.5:
            disp = max(
                disp,
                math.hypot(
                    float(feats_arr[idx, 2] - feats_arr[idx - 1, 2]),
                    float(feats_arr[idx, 3] - feats_arr[idx - 1, 3]),
                ),
            )
        step_speeds.append(disp / dt_i)

    max_speed = max(step_speeds) if step_speeds else 0.0
    end_speed = (
        float(sum(step_speeds[-2:]) / len(step_speeds[-2:])) if step_speeds else 0.0
    )
    summary[23] = min(1.0, max_speed / 4.0)
    summary[24] = min(1.0, end_speed / 4.0)

    span_x = max(l_sx, r_sx)
    span_y = max(l_sy, r_sy)
    summary[25] = span_x / max(span_x + span_y, 1e-4)
    osc_score = max(float(l_sum[4] + l_sum[5]), float(r_sum[4] + r_sum[5]))
    summary[26] = min(1.0, osc_score)

    yaw_span = float(resampled[:, 41].max() - resampled[:, 41].min())
    pitch_span = float(resampled[:, 42].max() - resampled[:, 42].min())
    head_span = max(yaw_span, pitch_span)
    summary[27] = min(1.0, head_span / 0.5)

    desc = np.concatenate([resampled.reshape(-1), summary]).astype(np.float32)
    assert desc.shape == (DYNAMIC_FEATURE_DIM,)
    total_path = max(float(l_sum[2]) * 5.0, float(r_sum[2]) * 5.0)
    shape_delta = max(float(l_sum[8]) * 1.5, float(r_sum[8]) * 1.5)
    metrics = {
        "duration_s": duration_s,
        "path_len": total_path,
        "max_speed": max_speed,
        "end_speed": end_speed,
        "shape_delta": shape_delta,
        "osc_count": osc_score * 4.0,
        "head_span": head_span,
    }
    return desc, metrics


def _fk_kframe_from_pose(
    pose_state: dict[str, Any],
    *,
    hand_scale_jitter: float = 1.0,
    wrist_jitter: tuple[float, float, float] = (0.0, 0.0, 0.0),
    amp_scale: float = 1.0,
    amp_center: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    from .avatar_dataset import _fk_hand_landmarks, _fk_pose_landmarks

    torso_yaw = float(pose_state.get("torsoYaw", 0.0) or 0.0)
    pose_lms = _fk_pose_landmarks(torso_yaw)
    hands: list[SimpleNamespace] = []

    for side_key, mp_label, sign in (("right", "Right", -1), ("left", "Left", 1)):
        if side_key not in pose_state:
            continue
        spec = pose_state[side_key]
        wrist = np.asarray(spec["wrist"], dtype=np.float32)
        if amp_center and side_key in amp_center:
            wrist = amp_center[side_key] + (wrist - amp_center[side_key]) * amp_scale
        wrist = wrist + np.asarray(wrist_jitter, dtype=np.float32) * hand_scale_jitter
        hands.append(
            SimpleNamespace(
                handedness=mp_label,
                landmarks=_fk_hand_landmarks(
                    wrist.tolist(),
                    spec["palm"],
                    spec["finger"],
                    spec["joints"],
                    sign,
                ),
            )
        )

    head = pose_state.get("head") or {}
    return extract_kinematic_frame(
        hands,
        pose_landmarks=pose_lms,
        head_yaw_ratio=float(head.get("yaw", 0.0) or 0.0) * 1.5,
        head_pitch_ratio=float(head.get("pitch", 0.0) or 0.0) * 1.5,
    )


def _track_motion_span(
    track: list[dict[str, Any]],
    handshapes: dict[str, Any],
    idle_pose: dict[str, Any],
) -> float:
    """Measure wrist displacement + shape change across track."""
    from .avatar_dataset import _sample_track_pose

    frames: list[tuple[int, np.ndarray]] = []
    for i in range(6):
        t_val = i / 5.0
        ps = _sample_track_pose(track, t_val, handshapes, idle_pose)
        frames.append((i * 150, _fk_kframe_from_pose(ps)))
    _, metrics = build_temporal_descriptor(frames)
    return metrics["path_len"] + metrics["shape_delta"] * 0.6 + metrics["head_span"] * 1.2


def _synthesize_char_stroke_sequence(
    ch: str,
    lib: dict[str, Any],
    duration_ms: float,
    rng: np.random.Generator,
) -> list[tuple[int, np.ndarray]]:
    """Synthesize dynamic trajectory for letters 'J' (pinky curve) and 'Z' (zigzag)."""
    spec = lib["letters"][ch]
    joints = lib["handshapes"].get(spec["shape"]) or lib["handshapes"]["relaxed"]
    amp = float(rng.uniform(0.75, 1.30))
    jx, jy = float(rng.normal(0.0, 0.03)), float(rng.normal(0.0, 0.03))
    n_pts = 10
    frames: list[tuple[int, np.ndarray]] = []

    for i in range(n_pts):
        u = i / (n_pts - 1)
        if ch == "J":
            # Downward stroke curving inward/left then hooking slightly up.
            wx = -0.22 + amp * (0.14 * max(0.0, u - 0.35) / 0.65) + jx
            wy = 1.44 - amp * 0.24 * math.sin(u * math.pi * 0.65) + jy
        else:
            # 'Z': left -> right (u<0.33), diag back left-down (0.33..0.67), right (0.67..1.0)
            if u < 0.33:
                p = u / 0.33
                wx, wy = -0.26 + amp * 0.18 * p, 1.44
            elif u < 0.67:
                p = (u - 0.33) / 0.34
                wx, wy = (-0.26 + amp * 0.18) - amp * 0.20 * p, 1.44 - amp * 0.18 * p
            else:
                p = (u - 0.67) / 0.33
                wx, wy = (-0.26 - amp * 0.02) + amp * 0.20 * p, 1.44 - amp * 0.18
            wx += jx
            wy += jy

        pose_state = {
            "torsoYaw": 0.0,
            "right": {
                "wrist": [wx, wy, 0.28],
                "palm": spec["palm"],
                "finger": spec["finger"],
                "joints": joints,
            },
        }
        kf = _fk_kframe_from_pose(pose_state)
        kf += rng.normal(0.0, 0.015, size=K_FRAME_DIM).astype(np.float32)
        kf[0:2] = (kf[0:2] >= 0.5).astype(np.float32)
        frames.append((int(round(u * duration_ms)), kf))
    return frames


_SYNTH_BASE_LOCK = threading.Lock()
_SYNTH_BASE_CACHE: dict[tuple[str, int, int], dict[str, list[np.ndarray]]] = {}


def synthesize_dynamic_training_sequences(
    animations_path: Path = DEFAULT_SIGN_ANIMATIONS_PATH,
    *,
    user_sequences: dict[str, list[list[tuple[int, np.ndarray]]]] | None = None,
    seed: int = 42,
) -> dict[str, list[np.ndarray]]:
    """Build 396-dim temporal sequences for moving signs + negative `_idle` motions."""
    from .avatar_dataset import _sample_track_pose, load_avatar_pose_library, track_id_to_label

    resolved_path = Path(animations_path).resolve()
    mtime_ns = resolved_path.stat().st_mtime_ns if resolved_path.is_file() else 0
    cache_key = (str(resolved_path), int(mtime_ns), int(seed))

    with _SYNTH_BASE_LOCK:
        cached = _SYNTH_BASE_CACHE.get(cache_key)

    if cached is not None:
        by_label: dict[str, list[np.ndarray]] = {
            lbl: [v.copy() for v in vecs] for lbl, vecs in cached.items()
        }
    else:
        by_label = _build_base_synthetic_sequences(
            animations_path,
            seed=seed,
            _load_lib=load_avatar_pose_library,
            _sample_pose=_sample_track_pose,
            _id_to_label=track_id_to_label,
        )
        with _SYNTH_BASE_LOCK:
            _SYNTH_BASE_CACHE[cache_key] = {
                lbl: [v.copy() for v in vecs] for lbl, vecs in by_label.items()
            }

    if user_sequences:
        for lbl, seqs in user_sequences.items():
            for seq in seqs:
                if len(seq) >= 4:
                    desc, _ = build_temporal_descriptor(seq)
                    by_label.setdefault(lbl, []).append(desc)
                    # Add mirrored sequence if 1-handed user clip
                    if (seq[0][1][0] > 0.5) != (seq[0][1][1] > 0.5):
                        m_seq = [(ts, mirror_kinematic_frame(kf)) for ts, kf in seq]
                        m_desc, _ = build_temporal_descriptor(m_seq)
                        by_label.setdefault(lbl, []).append(m_desc)

    return by_label


def _build_base_synthetic_sequences(
    animations_path: Path,
    *,
    seed: int,
    _load_lib: Any,
    _sample_pose: Any,
    _id_to_label: Any,
) -> dict[str, list[np.ndarray]]:
    lib = _load_lib(animations_path)
    handshapes = lib["handshapes"]
    idle_pose = lib["idle"]
    tracks = lib["tracks"]
    rng = np.random.default_rng(seed)

    by_label: dict[str, list[np.ndarray]] = {IDLE_MOTION_LABEL: []}

    shape_names = list(handshapes.keys())
    all_track_ids = [tid for tid in tracks if tid != "unknown_gloss"]

    # 1. Moving word/phrase tracks from SIGN_TRACKS
    cyclic_tracks = {
        "hello", "goodbye", "please", "sorry", "eat", "more", "wash", "work", "yes", "wait", "who"
    }
    for tid in all_track_ids:
        track = tracks[tid]
        if _track_motion_span(track, handshapes, idle_pose) < 0.14:
            continue
        label = _id_to_label(tid)
        vecs = by_label.setdefault(label, [])

        # Compute track center for amplitude scaling
        centers: dict[str, np.ndarray] = {}
        for side_key in ("right", "left"):
            pts = [
                np.asarray(kf[side_key]["wrist"], dtype=np.float32)
                for kf in track
                if side_key in kf and kf[side_key].get("wrist")
            ]
            if pts:
                centers[side_key] = np.mean(pts, axis=0)

        for aug_i in range(36):
            dur_ms = float(rng.uniform(520.0, 1280.0))
            t0 = float(rng.uniform(0.0, 0.16))
            t1 = float(rng.uniform(0.84, 1.0))
            amp = float(rng.uniform(0.68, 1.40))
            w_jit = (
                float(rng.normal(0.0, 0.035)),
                float(rng.normal(0.0, 0.035)),
                float(rng.normal(0.0, 0.02)),
            )
            do_repeat = tid in cyclic_tracks and aug_i % 3 == 1
            raw_seq: list[tuple[int, np.ndarray]] = []
            for step_i in range(10):
                u = step_i / 9.0
                if do_repeat:
                    t_eval = 0.20 + 0.70 * (0.5 - 0.5 * math.cos(u * 2.0 * math.pi))
                else:
                    t_eval = t0 + (t1 - t0) * u
                ps = _sample_pose(track, t_eval, handshapes, idle_pose)
                # Interpolate optional head cues when present in adjacent keyframes.
                ps["head"] = track[min(len(track) - 1, int(round(t_eval * (len(track) - 1))))].get("head", {})
                kf = _fk_kframe_from_pose(
                    ps,
                    wrist_jitter=w_jit,
                    amp_scale=amp,
                    amp_center=centers,
                )
                kf += rng.normal(0.0, 0.016, size=K_FRAME_DIM).astype(np.float32)
                kf[0:2] = (kf[0:2] >= 0.5).astype(np.float32)
                raw_seq.append((int(round(u * dur_ms)), kf))

            # Random mirroring for 1-handed dynamic signs
            if (raw_seq[0][1][0] > 0.5) != (raw_seq[0][1][1] > 0.5) and (aug_i % 2 == 1):
                raw_seq = [(ts, mirror_kinematic_frame(kf)) for ts, kf in raw_seq]

            desc, _ = build_temporal_descriptor(raw_seq)
            vecs.append(desc)

    # 2. Dynamic alphabet letters 'J' and 'Z'
    for ch in ("J", "Z"):
        vecs = by_label.setdefault(ch, [])
        for aug_i in range(32):
            dur_ms = float(rng.uniform(550.0, 1200.0))
            raw_seq = _synthesize_char_stroke_sequence(ch, lib, dur_ms, rng)
            if aug_i % 2 == 1:
                raw_seq = [(ts, mirror_kinematic_frame(kf)) for ts, kf in raw_seq]
            desc, _ = build_temporal_descriptor(raw_seq)
            vecs.append(desc)

    # 3. Balanced `_idle` negative motion sequences (relaxed hover, lowering arm to lap, relaxed transit)
    idle_vecs = by_label[IDLE_MOTION_LABEL]
    for n_i in range(96):
        dur_ms = float(rng.uniform(450.0, 1200.0))
        kind = n_i % 3
        raw_seq = []
        mode_2hand = n_i % 4 == 0
        w_start = [float(rng.uniform(-0.36, -0.08)), float(rng.uniform(0.94, 1.40)), float(rng.uniform(0.14, 0.30))]
        if kind == 0:
            # Gentle hover / breathing jitter around relaxed posture
            w_end = [w_start[0] + float(rng.normal(0, 0.035)), w_start[1] + float(rng.normal(0, 0.035)), w_start[2]]
        elif kind == 1:
            # Lowering arm back down below chest toward lap/waist
            w_start = [float(rng.uniform(-0.28, -0.10)), float(rng.uniform(1.22, 1.46)), 0.24]
            w_end = [float(rng.uniform(-0.32, -0.18)), float(rng.uniform(0.88, 1.02)), 0.14]
        else:
            # Unstructured relaxed hand drift across lower torso / waist
            w_start = [float(rng.uniform(-0.32, -0.12)), float(rng.uniform(0.96, 1.18)), 0.20]
            w_end = [w_start[0] + float(rng.uniform(-0.18, 0.18)), float(rng.uniform(0.92, 1.14)), 0.20]

        for step_i in range(9):
            u = step_i / 8.0
            wrist_r = [
                w_start[j] + (w_end[j] - w_start[j]) * u + float(rng.normal(0, 0.015))
                for j in range(3)
            ]
            ps: dict[str, Any] = {
                "torsoYaw": 0.0,
                "right": {
                    "wrist": wrist_r,
                    "palm": [float(rng.uniform(-0.5, 0.5)), -0.35, 0.75],
                    "finger": [0.15, -0.85 if kind >= 1 else 0.4, 0.25],
                    "joints": handshapes["relaxed"],
                },
            }
            if mode_2hand:
                ps["left"] = {
                    "wrist": [-wrist_r[0] + 0.05, wrist_r[1], wrist_r[2]],
                    "palm": [-0.4, -0.35, 0.75],
                    "finger": [-0.15, -0.85, 0.25],
                    "joints": handshapes["relaxed"],
                }
            kf = _fk_kframe_from_pose(ps)
            kf += rng.normal(0.0, 0.02, size=K_FRAME_DIM).astype(np.float32)
            kf[0:2] = (kf[0:2] >= 0.5).astype(np.float32)
            raw_seq.append((int(round(u * dur_ms)), kf))

        if (not mode_2hand) and (n_i % 2 == 1):
            raw_seq = [(ts, mirror_kinematic_frame(kf)) for ts, kf in raw_seq]
        desc, _ = build_temporal_descriptor(raw_seq)
        idle_vecs.append(desc)

    return by_label


def _dynamic_active_mask(desc: np.ndarray) -> np.ndarray:
    """Zero out absent hand slots inside a 396-dim temporal descriptor for unbiased similarity."""
    mask = np.ones(DYNAMIC_FEATURE_DIM, dtype=np.float32)
    seq = desc[: TEMPORAL_STEPS * K_FRAME_DIM].reshape(TEMPORAL_STEPS, K_FRAME_DIM)
    has_left = float(seq[:, 0].mean()) > 0.25
    has_right = float(seq[:, 1].mean()) > 0.25

    seq_mask = np.ones((TEMPORAL_STEPS, K_FRAME_DIM), dtype=np.float32)
    if not has_left:
        seq_mask[:, 2:20] = 0.0
    if not has_right:
        seq_mask[:, 20:38] = 0.0
    if not (has_left and has_right):
        seq_mask[:, 38:41] = 0.0
        seq_mask[:, 45] = 0.0

    mask[: TEMPORAL_STEPS * K_FRAME_DIM] = seq_mask.reshape(-1)
    sum_off = TEMPORAL_STEPS * K_FRAME_DIM
    if not has_left:
        mask[sum_off : sum_off + 9] = 0.0
    if not has_right:
        mask[sum_off + 9 : sum_off + 18] = 0.0
    if not (has_left and has_right):
        mask[sum_off + 18 : sum_off + 23] = 0.0
    if not has_left and not has_right:
        mask[:] = 0.0
    return mask


class DynamicSignMLP(nn.Module):
    """Lightweight temporal MLP for 396-dim moving-sign sequence descriptors."""

    def __init__(
        self,
        input_dim: int = DYNAMIC_FEATURE_DIM,
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


def train_dynamic_classifier(
    sequences_by_label: dict[str, list[np.ndarray]] | None = None,
    output_path: Path = DEFAULT_DYNAMIC_CLASSIFIER_PATH,
    *,
    animations_path: Path = DEFAULT_SIGN_ANIMATIONS_PATH,
    user_sequences: dict[str, list[list[tuple[int, np.ndarray]]]] | None = None,
    epochs: int = 45,
    batch_size: int = 64,
    learning_rate: float = 2e-3,
    seed: int = 42,
) -> dict[str, Any]:
    """Train `DynamicSignMLP` over spatio-temporal sign trajectories and save checkpoint."""
    data = (
        sequences_by_label
        if sequences_by_label is not None
        else synthesize_dynamic_training_sequences(
            animations_path=animations_path,
            user_sequences=user_sequences,
            seed=seed,
        )
    )
    clean_classes = {
        lbl: [np.asarray(v, dtype=np.float32) for v in vecs]
        for lbl, vecs in data.items()
        if vecs
    }
    if not clean_classes:
        raise ValueError("Cannot train dynamic classifier with zero sequences.")

    labels = sorted(clean_classes.keys())
    label_to_idx = {lbl: idx for idx, lbl in enumerate(labels)}
    torch.manual_seed(seed)

    raw_x: list[np.ndarray] = []
    raw_y: list[int] = []
    rng = np.random.default_rng(seed)
    target_per_class = 56
    for lbl in labels:
        idx = label_to_idx[lbl]
        vecs = clean_classes[lbl]
        for k in range(max(len(vecs), target_per_class)):
            vec = vecs[k % len(vecs)]
            if k >= len(vecs):
                vec = (vec + rng.normal(0.0, 0.012, size=DYNAMIC_FEATURE_DIM)).astype(np.float32)
            raw_x.append(vec)
            raw_y.append(idx)

    x_np = np.stack(raw_x, axis=0).astype(np.float32)
    y_np = np.asarray(raw_y, dtype=np.int64)
    masks_np = np.stack([_dynamic_active_mask(v) for v in raw_x], axis=0).astype(np.float32)

    mean = torch.from_numpy(x_np.mean(axis=0)).float()
    # Floor std at 0.08 so near-constant columns (head pitch/pose flags) cannot blow up under sensor noise.
    std = torch.from_numpy(np.maximum(x_np.std(axis=0), 0.08)).float()
    x_tensor = (torch.from_numpy(x_np) - mean) / std
    y_tensor = torch.from_numpy(y_np)
    masks_tensor = torch.from_numpy(masks_np)

    centroids = []
    for idx in range(len(labels)):
        sel = y_tensor == idx
        cls_vecs = x_tensor[sel] * masks_tensor[sel]
        norms = cls_vecs.norm(dim=1, keepdim=True).clamp(min=1e-6)
        c = (cls_vecs / norms).mean(dim=0)
        c = c / c.norm().clamp(min=1e-6)
        centroids.append(c)
    centroid_tensor = torch.stack(centroids, dim=0)

    model = DynamicSignMLP(input_dim=DYNAMIC_FEATURE_DIM, num_classes=len(labels))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    model.train()
    n_samples = x_tensor.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n_samples)
        for start in range(0, n_samples, batch_size):
            b_idx = perm[start : start + batch_size]
            xb = x_tensor[b_idx]
            yb = y_tensor[b_idx]
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.inference_mode():
        preds = model(x_tensor).argmax(dim=1)
        train_accuracy = float((preds == y_tensor).float().mean().item())

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f"{target.name}.{threading.get_ident()}.tmp")
    trained_at = datetime.now(timezone.utc).isoformat()
    sign_labels = [lbl for lbl in labels if lbl != IDLE_MOTION_LABEL]
    sample_counts = {lbl: len(clean_classes[lbl]) for lbl in labels}

    checkpoint = {
        "state_dict": model.state_dict(),
        "labels": labels,
        "feature_dim": DYNAMIC_FEATURE_DIM,
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
        for attempt in range(8):
            try:
                tmp_path.replace(target)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.1)
    finally:
        tmp_path.unlink(missing_ok=True)

    return {
        "model_path": str(target),
        "num_classes": len(sign_labels),
        "total_classes_with_idle": len(labels),
        "labels": sign_labels,
        "total_sequences": int(n_samples),
        "train_accuracy": round(train_accuracy, 4),
        "epochs": epochs,
        "trained_at": trained_at,
    }


class DynamicSignClassifier:
    """Thread-safe hot-reloading wrapper around `DynamicSignMLP`."""

    def __init__(self, model_path: Path = DEFAULT_DYNAMIC_CLASSIFIER_PATH) -> None:
        self.model_path = Path(model_path)
        self._lock = threading.Lock()
        self._mtime_ns: int | None = None
        self._model: DynamicSignMLP | None = None
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
        """Return lightweight status metadata for API health and training panels."""
        self.maybe_reload()
        with self._lock:
            sign_labels = [lbl for lbl in self._labels if lbl != IDLE_MOTION_LABEL]
            return {
                "loaded": self.is_loaded,
                "model_path": str(self.model_path),
                "num_classes": len(sign_labels),
                "labels": sign_labels,
                **self._metadata,
            }

    def maybe_reload(self) -> bool:
        """Load or reload checkpoint cleanly when modified on disk."""
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
            f_dim = int(ckpt.get("feature_dim", DYNAMIC_FEATURE_DIM))
            model = DynamicSignMLP(input_dim=f_dim, num_classes=len(labels))
            model.load_state_dict(ckpt["state_dict"])
            model.eval()

            self._model = model
            self._labels = labels
            self._mean = ckpt["feature_mean"].float()
            self._std = ckpt["feature_std"].float().clamp(min=0.08)
            self._centroids = ckpt["class_centroids"].float()
            self._mtime_ns = stat_mtime
            self._metadata = {
                "trained_at": ckpt.get("trained_at"),
                "train_accuracy": ckpt.get("train_accuracy"),
                "epochs": ckpt.get("epochs"),
            }
            return True

    def predict_sequence(
        self,
        descriptor: np.ndarray,
        *,
        top_k: int = 3,
        min_similarity: float = 0.45,
    ) -> list[DynamicPrediction]:
        """Score a 396-dim sequence descriptor and return sorted candidates."""
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

            vec_np = np.asarray(descriptor, dtype=np.float32).reshape(-1)
            mask = torch.from_numpy(_dynamic_active_mask(vec_np))
            vec = torch.from_numpy(vec_np)
            normed = (vec - self._mean) / self._std
            masked_normed = normed * mask
            mn_norm = float(masked_normed.norm().item())
            if mn_norm <= 1e-6:
                sims = torch.zeros(len(self._labels), dtype=torch.float32)
            else:
                unit = masked_normed / mn_norm
                sims = torch.matmul(self._centroids, unit)

            with torch.inference_mode():
                logits = self._model(normed.unsqueeze(0)).squeeze(0)
                probs = torch.softmax(logits, dim=0)

            k = min(max(1, top_k), len(self._labels))
            top_probs, top_indices = torch.topk(probs, k=k)
            out: list[DynamicPrediction] = []
            for p, idx in zip(top_probs.tolist(), top_indices.tolist()):
                lbl = self._labels[idx]
                sim = float(sims[idx].item())
                score = float(p) if sim >= min_similarity else min(float(p), max(0.0, sim * 0.6))
                out.append(DynamicPrediction(label=lbl, score=score, similarity=sim))
            return out


class TemporalSignTracker:
    """Real-time sliding-window moving-sign recognizer over live webcam streams."""

    def __init__(
        self,
        classifier: DynamicSignClassifier | None = None,
        *,
        max_buffer_ms: int = 1500,
        windows_ms: tuple[int, ...] = (480, 780, 1120),
        min_path_shoulders: float = 0.18,
        score_threshold: float = 0.60,
        min_similarity: float = 0.48,
        cooldown_seconds: float = 1.4,
    ) -> None:
        self.classifier = classifier
        self.max_buffer_ms = max_buffer_ms
        self.windows_ms = windows_ms
        self.min_path_shoulders = min_path_shoulders
        self.score_threshold = score_threshold
        self.min_similarity = min_similarity
        self.cooldown_seconds = cooldown_seconds

        self._buffer: deque[tuple[int, np.ndarray]] = deque()
        self._recent_votes: deque[tuple[int, str, float]] = deque(maxlen=4)
        self._last_fired_label: str | None = None
        self._last_fired_at_ms: int = -10_000_000

    def reset(self) -> None:
        self._buffer.clear()
        self._recent_votes.clear()
        self._last_fired_label = None
        self._last_fired_at_ms = -10_000_000

    def update(
        self,
        timestamp_ms: int,
        kframe: np.ndarray,
    ) -> tuple[DynamicTrigger | None, list[DynamicPrediction], float]:
        """Push one frame and return `(triggered_event, live_candidates, instant_speed)`."""
        self._buffer.append((int(timestamp_ms), np.asarray(kframe, dtype=np.float32)))
        cutoff = timestamp_ms - self.max_buffer_ms
        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.popleft()

        # Instant speed over latest two frames (shoulder widths / sec).
        instant_speed = 0.0
        if len(self._buffer) >= 2:
            t_prev, f_prev = self._buffer[-2]
            dt_s = max((timestamp_ms - t_prev) / 1000.0, 0.016)
            for p_col, x_col in ((0, 2), (1, 20)):
                if kframe[p_col] > 0.5 and f_prev[p_col] > 0.5:
                    instant_speed = max(
                        instant_speed,
                        math.hypot(
                            float(kframe[x_col] - f_prev[x_col]),
                            float(kframe[x_col + 1] - f_prev[x_col + 1]),
                        )
                        / dt_s,
                    )

        if self.classifier is None or not self.classifier.is_loaded or len(self._buffer) < 4:
            return None, [], instant_speed

        raw_list = list(self._buffer)
        best_cands: list[DynamicPrediction] = []
        best_pred: DynamicPrediction | None = None
        best_metrics: dict[str, float] = {}
        best_win_ms = 0

        for win_ms in self.windows_ms:
            win_start = timestamp_ms - win_ms
            win_frames = [item for item in raw_list if item[0] >= win_start]
            if len(win_frames) < 4:
                continue
            if (win_frames[-1][0] - win_frames[0][0]) < max(260, int(win_ms * 0.45)):
                continue

            desc, metrics = build_temporal_descriptor(win_frames)
            has_motion = (
                metrics["path_len"] >= self.min_path_shoulders
                or metrics["shape_delta"] >= 0.22
                or metrics["head_span"] >= 0.10
            )
            if not has_motion:
                continue

            cands = self.classifier.predict_sequence(
                desc,
                top_k=4,
                min_similarity=self.min_similarity,
            )
            non_idle = [c for c in cands if c.label != IDLE_MOTION_LABEL]
            idle_score = next((c.score for c in cands if c.label == IDLE_MOTION_LABEL), 0.0)
            if not non_idle:
                continue
            top = non_idle[0]
            if idle_score >= top.score:
                continue

            if best_pred is None or top.score > best_pred.score:
                best_pred = top
                best_cands = non_idle[:3]
                best_metrics = metrics
                best_win_ms = int(win_frames[-1][0] - win_frames[0][0])

        if best_pred is None or best_pred.score < self.score_threshold:
            return None, best_cands, instant_speed

        second_score = best_cands[1].score if len(best_cands) > 1 else 0.0
        if (best_pred.score - second_score) < 0.10:
            return None, best_cands, instant_speed

        self._recent_votes.append((timestamp_ms, best_pred.label, best_pred.score))
        recent_same = sum(
            1
            for ts, lbl, _ in self._recent_votes
            if lbl == best_pred.label and (timestamp_ms - ts) <= 450
        )

        # Fire when 2+ overlapping windows agree, on clean stroke completion, or with very high margin/confidence.
        stroke_settled = (
            best_metrics.get("max_speed", 0.0) >= 0.55
            and best_metrics.get("end_speed", 1.0) <= 0.48 * best_metrics.get("max_speed", 1.0)
            and best_pred.score >= 0.68
        )
        ultra_confident = best_pred.score >= 0.88 and (best_pred.score - second_score) >= 0.45
        if recent_same < 2 and not stroke_settled and not ultra_confident:
            return None, best_cands, instant_speed

        in_cooldown = (
            best_pred.label == self._last_fired_label
            and (timestamp_ms - self._last_fired_at_ms) < int(self.cooldown_seconds * 1000)
        )
        if in_cooldown:
            return None, best_cands, instant_speed

        self._last_fired_label = best_pred.label
        self._last_fired_at_ms = timestamp_ms
        self._recent_votes.clear()
        trigger = DynamicTrigger(
            label=best_pred.label,
            score=best_pred.score,
            similarity=best_pred.similarity,
            window_ms=best_win_ms,
            candidates=best_cands,
        )
        return trigger, best_cands, instant_speed
