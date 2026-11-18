"""Bridge Voice -> Sign 3D avatar keyframe poses into Sign -> Text/Voice landmarks."""

from __future__ import annotations

import math
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

from .config import (
    BACKEND_DIR,
    DEFAULT_CUSTOM_CLASSIFIER_PATH,
    DEFAULT_DATASET_DIR,
    HAND_CONNECTIONS,
)
from .custom_classifier import (
    extract_sign_features,
    mirror_feature_vector,
    train_landmark_classifier,
)
from .sign_dataset import SignDatasetStore

DEFAULT_SIGN_ANIMATIONS_PATH = (
    BACKEND_DIR.parent / "frontend" / "src" / "utils" / "signAnimations.js"
)

SIGN_TRACK_LABEL_OVERRIDES: dict[str, str] = {
    "give_a_to_b": "Give",
    "i_love_you": "I Love You",
    "thank_you": "Thank You",
    "good_morning": "Good Morning",
}

_HAND_BONE_SPECS: tuple[tuple[str, int, tuple[float, float, float]], ...] = (
    ("index", -1, (0.036, 0.024, 0.018)),
    ("middle", -1, (0.040, 0.027, 0.019)),
    ("ring", 1, (0.036, 0.024, 0.018)),
    ("pinky", 1, (0.028, 0.020, 0.015)),
)
_FINGER_X_OFFSETS: dict[str, float] = {
    "index": -0.025,
    "middle": -0.008,
    "ring": 0.009,
    "pinky": 0.025,
}


def _extract_js_object(src: str, name: str) -> dict[str, Any]:
    """Parse a top-level JS const object literal (`HANDSHAPES`, `SIGN_TRACKS`, etc.)."""
    import json

    match = re.search(rf"(?:export\s+)?const\s+{re.escape(name)}\s*=\s*\{{", src)
    if match is None:
        raise ValueError(f"Could not locate `{name}` in {DEFAULT_SIGN_ANIMATIONS_PATH}.")

    start = match.end() - 1
    depth = 0
    in_str: str | None = None
    escaped = False
    end = -1

    for idx in range(start, len(src)):
        ch = src[idx]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == in_str:
                in_str = None
        else:
            if ch in ('"', "'"):
                in_str = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = idx
                    break

    if end == -1:
        raise ValueError(f"Unbalanced braces while parsing `{name}`.")

    block = src[start : end + 1]
    block = re.sub(r"//[^\n]*", "", block)
    block = re.sub(r'([{\[,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', block)
    block = re.sub(r",\s*([}\]])", r"\1", block)
    return json.loads(block)


def load_avatar_pose_library(
    animations_path: Path = DEFAULT_SIGN_ANIMATIONS_PATH,
) -> dict[str, Any]:
    """Load `HANDSHAPES`, `LETTER_HANDSHAPE`, `IDLE_POSE`, and `SIGN_TRACKS` from frontend JS."""
    src = Path(animations_path).read_text(encoding="utf-8")
    return {
        "handshapes": _extract_js_object(src, "HANDSHAPES"),
        "letters": _extract_js_object(src, "LETTER_HANDSHAPE"),
        "idle": _extract_js_object(src, "IDLE_POSE"),
        "tracks": _extract_js_object(src, "SIGN_TRACKS"),
    }


def track_id_to_label(track_id: str) -> str:
    """Convert a `SIGN_TRACKS` key into its canonical SignBridge dataset class label."""
    if track_id in SIGN_TRACK_LABEL_OVERRIDES:
        return SIGN_TRACK_LABEL_OVERRIDES[track_id]
    return " ".join(part.capitalize() for part in track_id.split("_") if part)


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _euler_xyz(a: float, b: float, c: float) -> np.ndarray:
    return _rx(a) @ _ry(b) @ _rz(c)


def _orient_basis(finger_arr: list[float], palm_arr: list[float]) -> np.ndarray:
    f = np.asarray(finger_arr, dtype=np.float32)
    f = f / max(float(np.linalg.norm(f)), 1e-8)
    p = np.asarray(palm_arr, dtype=np.float32)
    p = p / max(float(np.linalg.norm(p)), 1e-8)
    side = np.cross(f, p)
    if float(np.dot(side, side)) < 1e-6:
        side = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        side = side / float(np.linalg.norm(side))
    palm = np.cross(side, f)
    palm = palm / max(float(np.linalg.norm(palm)), 1e-8)
    return np.column_stack([side, f, palm])


def _fk_hand_landmarks(
    wrist_world: list[float],
    palm_arr: list[float],
    finger_arr: list[float],
    joints: dict[str, list[float]],
    side: int,
) -> list[SimpleNamespace]:
    """Compute 21 MediaPipe-compatible landmarks via 3D forward kinematics of `SignAvatar3D`."""
    basis = _orient_basis(finger_arr, palm_arr)
    w_pos = np.asarray(wrist_world, dtype=np.float32)

    local_pts: list[np.ndarray] = [np.zeros(3, dtype=np.float32)]
    ta = joints["thumb"]
    local_pts.append(np.array([side * -0.020, 0.014, 0.004], dtype=np.float32))
    t_base = np.array([side * -0.034, 0.026, 0.006], dtype=np.float32)
    local_pts.append(t_base)
    r_tb = _euler_xyz(ta[0] + ta[3] * 0.45, -side * ta[3], side * ta[2])
    t_ip = t_base + r_tb @ np.array([0.0, 0.035, 0.0], dtype=np.float32)
    local_pts.append(t_ip)
    r_tt = _euler_xyz(ta[1], 0.0, -side * ta[3] * 0.25)
    t_tip = t_ip + (r_tb @ r_tt) @ np.array([0.0, 0.030, 0.0], dtype=np.float32)
    local_pts.append(t_tip)

    for finger_name in ("index", "middle", "ring", "pinky"):
        ang = joints[finger_name]
        xp = side * _FINGER_X_OFFSETS[finger_name]
        lens = next(item[2] for item in _HAND_BONE_SPECS if item[0] == finger_name)
        mcp = np.array([xp, 0.084, 0.0], dtype=np.float32)
        r_m = _euler_xyz(ang[0], 0.0, -side * ang[3])
        r_p = _euler_xyz(ang[1], 0.0, 0.0)
        r_d = _euler_xyz(ang[2], 0.0, 0.0)
        pip = mcp + r_m @ np.array([0.0, lens[0], 0.0], dtype=np.float32)
        dip = pip + (r_m @ r_p) @ np.array([0.0, lens[1], 0.0], dtype=np.float32)
        tip = dip + (r_m @ r_p @ r_d) @ np.array([0.0, lens[2], 0.0], dtype=np.float32)
        local_pts.extend([mcp, pip, dip, tip])

    pts_world = w_pos[None, :] + (np.asarray(local_pts, dtype=np.float32) @ basis.T)
    return [
        SimpleNamespace(
            x=float(0.5 + p[0] * 0.8),
            y=float(0.5 - (p[1] - 1.32) * 0.8),
            z=float(-p[2] * 0.8),
        )
        for p in pts_world
    ]


def _fk_pose_landmarks(torso_yaw: float = 0.0) -> list[SimpleNamespace]:
    """Generate 33 minimal pose landmarks with accurate shoulder anchors (11, 12)."""
    pts = [SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=0.0) for _ in range(33)]
    c, s = math.cos(torso_yaw), math.sin(torso_yaw)
    for idx, wx in ((11, 0.2), (12, -0.2)):
        rx = wx * c
        rz = -wx * s
        pts[idx] = SimpleNamespace(
            x=float(0.5 + rx * 0.8),
            y=float(0.5 - (1.38 - 1.32) * 0.8),
            z=float(-rz * 0.8),
            visibility=1.0,
        )
    return pts


def _lerp_vec(a: list[float], b: list[float], s: float) -> list[float]:
    return [float(x + (y - x) * s) for x, y in zip(a, b)]


def _normalize_vec(v: list[float]) -> list[float]:
    n = math.sqrt(sum(c * c for c in v)) or 1.0
    return [c / n for c in v]


def _sample_track_pose(
    track: list[dict[str, Any]],
    t: float,
    handshapes: dict[str, dict[str, list[float]]],
    idle_pose: dict[str, Any],
) -> dict[str, Any]:
    """Sample interpolated avatar hand & torso state at normalized time `t` in [0, 1]."""
    has_left = any("left" in kf for kf in track)
    has_right = any("right" in kf for kf in track)

    def resolve_kf(kf: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {"torsoYaw": float(kf.get("torsoYaw", 0.0) or 0.0)}
        for side_key in ("right", "left"):
            if (side_key == "right" and has_right) or (side_key == "left" and has_left):
                spec = kf.get(side_key) or idle_pose[side_key]
                shape_name = spec.get("shape") or idle_pose[side_key]["shape"]
                out[side_key] = {
                    "wrist": list(spec.get("wrist") or idle_pose[side_key]["wrist"]),
                    "palm": _normalize_vec(spec.get("palm") or idle_pose[side_key]["palm"]),
                    "finger": _normalize_vec(spec.get("finger") or idle_pose[side_key]["finger"]),
                    "joints": handshapes.get(shape_name) or handshapes["relaxed"],
                }
        return out

    if len(track) == 1 or t <= float(track[0].get("t", 0.0)):
        return resolve_kf(track[0])
    if t >= float(track[-1].get("t", 1.0)):
        return resolve_kf(track[-1])

    for i in range(len(track) - 1):
        t0 = float(track[i].get("t", 0.0))
        t1 = float(track[i + 1].get("t", 1.0))
        if t0 <= t <= t1:
            span = max(1e-4, t1 - t0)
            u = (t - t0) / span
            s = u * u * (3.0 - 2.0 * u)
            p0 = resolve_kf(track[i])
            p1 = resolve_kf(track[i + 1])
            blended: dict[str, Any] = {
                "torsoYaw": p0["torsoYaw"] + (p1["torsoYaw"] - p0["torsoYaw"]) * s
            }
            for side_key in ("right", "left"):
                if side_key in p0 and side_key in p1:
                    j0 = p0[side_key]["joints"]
                    j1 = p1[side_key]["joints"]
                    blended[side_key] = {
                        "wrist": _lerp_vec(p0[side_key]["wrist"], p1[side_key]["wrist"], s),
                        "palm": _normalize_vec(_lerp_vec(p0[side_key]["palm"], p1[side_key]["palm"], s)),
                        "finger": _normalize_vec(_lerp_vec(p0[side_key]["finger"], p1[side_key]["finger"], s)),
                        "joints": {
                            k: _lerp_vec(j0[k], j1[k], s)
                            for k in ("thumb", "index", "middle", "ring", "pinky")
                        },
                    }
            return blended
    return resolve_kf(track[-1])


def _render_skeleton_preview(
    label: str,
    hands: list[SimpleNamespace],
    pose: list[SimpleNamespace],
    size: int = 256,
) -> np.ndarray:
    """Render a crisp 256x256 hand + shoulder skeleton thumbnail for dataset review."""
    canvas = np.full((size, size, 3), (24, 18, 12), dtype=np.uint8)

    # Subtle shoulder bar
    ls = (int(pose[11].x * size), int(pose[11].y * size))
    rs = (int(pose[12].x * size), int(pose[12].y * size))
    cv2.line(canvas, rs, ls, (71, 85, 105), 3, cv2.LINE_AA)

    colors = {"Right": (248, 189, 56), "Left": (129, 212, 16)}
    for hand in hands:
        col = colors.get(hand.handedness, (248, 189, 56))
        pts = [
            (
                max(6, min(size - 6, int(round(p.x * size)))),
                max(6, min(size - 6, int(round(p.y * size)))),
            )
            for p in hand.landmarks
        ]
        for a, b in HAND_CONNECTIONS:
            cv2.line(canvas, pts[a], pts[b], col, 2, cv2.LINE_AA)
        for px, py in pts:
            cv2.circle(canvas, (px, py), 3, (248, 250, 252), -1, cv2.LINE_AA)

    cv2.putText(
        canvas,
        f"Avatar: {label}",
        (10, size - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (226, 232, 240),
        1,
        cv2.LINE_AA,
    )
    return canvas


def sync_voice_to_sign_dataset(
    *,
    store: SignDatasetStore | None = None,
    animations_path: Path = DEFAULT_SIGN_ANIMATIONS_PATH,
    model_path: Path = DEFAULT_CUSTOM_CLASSIFIER_PATH,
    include_existing: bool = False,
    train_model: bool = True,
    epochs: int = 55,
) -> dict[str, Any]:
    """Populate Sign -> Text/Voice dataset with FK landmarks from Voice -> Sign 3D poses."""
    store = store or SignDatasetStore(DEFAULT_DATASET_DIR)
    lib = load_avatar_pose_library(animations_path)
    handshapes = lib["handshapes"]
    letters = lib["letters"]
    idle_pose = lib["idle"]
    tracks = lib["tracks"]

    existing_labels = {
        c["label"].lower(): c["label"] for c in store.get_summary().get("classes", [])
    }

    added_samples = 0
    added_classes: list[str] = []
    skipped_existing: list[str] = []
    pending_items: list[dict[str, Any]] = []

    # 1. Sync SIGN_TRACKS word/phrase signs
    for track_id, track in tracks.items():
        if track_id == "unknown_gloss":
            continue
        label = track_id_to_label(track_id)
        if not include_existing and label.lower() in existing_labels:
            skipped_existing.append(label)
            continue

        added_classes.append(label)
        slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        for kf_idx, t_val in enumerate((0.25, 0.55, 0.85)):
            pose_state = _sample_track_pose(track, t_val, handshapes, idle_pose)
            pose_lms = _fk_pose_landmarks(pose_state.get("torsoYaw", 0.0))
            hands: list[SimpleNamespace] = []
            if "right" in pose_state:
                r = pose_state["right"]
                hands.append(
                    SimpleNamespace(
                        handedness="Right",
                        landmarks=_fk_hand_landmarks(
                            r["wrist"], r["palm"], r["finger"], r["joints"], -1
                        ),
                    )
                )
            if "left" in pose_state:
                l = pose_state["left"]
                hands.append(
                    SimpleNamespace(
                        handedness="Left",
                        landmarks=_fk_hand_landmarks(
                            l["wrist"], l["palm"], l["finger"], l["joints"], 1
                        ),
                    )
                )
            if not hands:
                continue

            feats = extract_sign_features(hands, pose_landmarks=pose_lms)
            preview = _render_skeleton_preview(label, hands, pose_lms)
            pending_items.append(
                {
                    "bgr": preview,
                    "label": label,
                    "features": feats,
                    "num_hands": len(hands),
                    "has_pose": True,
                    "source": "voice_to_sign_avatar",
                    "needs_review": False,
                    "metadata": {"track_id": track_id, "t": t_val},
                    "sample_id": f"avatar_{slug}_kf{kf_idx}",
                }
            )
            added_samples += 1

    # 2. Sync missing fingerspelling digits/letters (e.g. '0', '7', '8')
    for ch, spec in letters.items():
        if not include_existing and ch.lower() in existing_labels:
            skipped_existing.append(ch)
            continue

        added_classes.append(ch)
        joints = handshapes.get(spec["shape"]) or handshapes["relaxed"]
        pose_lms = _fk_pose_landmarks(0.0)
        right_hand = SimpleNamespace(
            handedness="Right",
            landmarks=_fk_hand_landmarks(
                [-0.16, 1.32, 0.30], spec["palm"], spec["finger"], joints, -1
            ),
        )
        feats_r = extract_sign_features([right_hand], pose_landmarks=pose_lms)
        preview_r = _render_skeleton_preview(ch, [right_hand], pose_lms)
        pending_items.append(
            {
                "bgr": preview_r,
                "label": ch,
                "features": feats_r,
                "num_hands": 1,
                "has_pose": True,
                "source": "voice_to_sign_avatar",
                "needs_review": False,
                "metadata": {"fingerspell_char": ch, "handedness": "Right"},
                "sample_id": f"avatar_char_{ch.lower()}_r",
            }
        )
        added_samples += 1

        # Store mirrored Left-hand sample too so single-hand fingerspelling works with either hand.
        feats_l = mirror_feature_vector(feats_r)
        left_hand = SimpleNamespace(
            handedness="Left",
            landmarks=[
                SimpleNamespace(x=1.0 - p.x, y=p.y, z=p.z) for p in right_hand.landmarks
            ],
        )
        preview_l = _render_skeleton_preview(ch, [left_hand], pose_lms)
        pending_items.append(
            {
                "bgr": preview_l,
                "label": ch,
                "features": feats_l,
                "num_hands": 1,
                "has_pose": True,
                "source": "voice_to_sign_avatar",
                "needs_review": False,
                "metadata": {"fingerspell_char": ch, "handedness": "Left"},
                "sample_id": f"avatar_char_{ch.lower()}_l",
            }
        )
        added_samples += 1

    if pending_items:
        store.add_samples_batch_from_features(pending_items)

    train_info = None
    if train_model:
        training_data = store.get_training_data()
        if training_data:
            train_info = train_landmark_classifier(
                training_data,
                output_path=model_path,
                epochs=epochs,
            )

    return {
        "added_classes": added_classes,
        "added_samples": added_samples,
        "skipped_existing": skipped_existing,
        "summary": store.get_summary(),
        "train": train_info,
    }


if __name__ == "__main__":
    res = sync_voice_to_sign_dataset()
    print(
        f"Added {len(res['added_classes'])} classes ({res['added_samples']} samples): "
        f"{res['added_classes']}"
    )
    if res["train"]:
        print(
            f"Retrained -> {res['train']['model_path']} "
            f"(classes={res['train']['num_classes']}, acc={res['train']['train_accuracy']})"
        )
