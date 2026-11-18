"""Holistic spatial grammar & non-manual marker engine for SignBridge.

Transforms raw MediaPipe HolisticLandmarker outputs (33 pose landmarks, 478 face
landmarks, 52 ARKit-compatible face blendshapes, and 21-point hand skeletons)
into body-anchored signing space coordinates, non-manual grammatical markers
(eyebrow mood, head nod/shake/tilt, eye gaze, cheek puff, body role-shift),
and cross-zone directional verb trajectories (e.g., Locus A -> Locus B).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any, Sequence

from .config import DIRECTIONAL_PHRASES

# Canonical MediaPipe Pose landmark indices.
POSE_NOSE = 0
POSE_LEFT_EYE = 2
POSE_RIGHT_EYE = 5
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_HIP = 23
POSE_RIGHT_HIP = 24

# Canonical MediaPipe 478 Face Mesh landmark indices.
FACE_FOREHEAD = 10
FACE_CHIN = 152
FACE_NOSE_TIP = 1
FACE_RIGHT_CHEEK = 234
FACE_LEFT_CHEEK = 454
FACE_RIGHT_EYE_OUTER = 33
FACE_RIGHT_EYE_INNER = 133
FACE_RIGHT_EYE_TOP = 159
FACE_RIGHT_EYE_BOTTOM = 145
FACE_LEFT_EYE_INNER = 362
FACE_LEFT_EYE_OUTER = 263
FACE_LEFT_EYE_TOP = 386
FACE_LEFT_EYE_BOTTOM = 374
FACE_RIGHT_BROW_MID = 105
FACE_LEFT_BROW_MID = 334
FACE_RIGHT_IRIS_CENTER = 468
FACE_LEFT_IRIS_CENTER = 473
FACE_UPPER_LIP_INNER = 13
FACE_LOWER_LIP_INNER = 14

# Ordered polyline landmark indices verified against MediaPipe FaceLandmarksConnections.
FACE_CONTOUR_PATHS: dict[str, tuple[int, ...]] = {
    "face_oval": (
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10,
    ),
    "left_eyebrow": (300, 293, 334, 296, 336),
    "right_eyebrow": (70, 63, 105, 66, 107),
    "left_eye": (
        263, 249, 390, 373, 374, 380, 381, 382, 362,
        398, 384, 385, 386, 387, 388, 466, 263,
    ),
    "right_eye": (
        33, 7, 163, 144, 145, 153, 154, 155, 133,
        173, 157, 158, 159, 160, 161, 246, 33,
    ),
    "left_iris": (474, 475, 476, 477, 474),
    "right_iris": (469, 470, 471, 472, 469),
    "lips_outer": (
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
        375, 321, 405, 314, 17, 84, 181, 91, 146, 61,
    ),
    "lips_inner": (
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
        324, 318, 402, 317, 14, 87, 178, 88, 95, 78,
    ),
    "nose_bridge": (168, 6, 197, 195, 5, 4, 1, 2),
}

_KEY_NMM_BLENDSHAPES: tuple[str, ...] = (
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "browDownLeft",
    "browDownRight",
    "cheekPuff",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "jawOpen",
    "mouthFunnel",
    "mouthPucker",
    "mouthSmileLeft",
    "mouthSmileRight",
)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _lm_xyv(lm: Any) -> tuple[float, float, float]:
    """Read (x, y, visibility) safely from a MediaPipe landmark or SimpleNamespace."""
    x = float(getattr(lm, "x", 0.0) or 0.0)
    y = float(getattr(lm, "y", 0.0) or 0.0)
    vis_raw = getattr(lm, "visibility", None)
    vis = 1.0 if vis_raw is None else float(vis_raw)
    return x, y, vis


@dataclass(slots=True)
class SigningSpaceFrame:
    """Body-anchored coordinate system for Left / Center / Right signing space."""

    source: str
    torso_center_x: float
    shoulder_y: float
    shoulder_width: float
    left_boundary: float
    right_boundary: float
    head_zone_max_y: float
    waist_zone_min_y: float
    body_shift: str = "center"
    shoulder_tilt_deg: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "torso_center_x": round(self.torso_center_x, 4),
            "shoulder_y": round(self.shoulder_y, 4),
            "shoulder_width": round(self.shoulder_width, 4),
            "left_boundary": round(self.left_boundary, 4),
            "right_boundary": round(self.right_boundary, 4),
            "head_zone_max_y": round(self.head_zone_max_y, 4),
            "waist_zone_min_y": round(self.waist_zone_min_y, 4),
            "body_shift": self.body_shift,
            "shoulder_tilt_deg": round(self.shoulder_tilt_deg, 2),
        }


@dataclass(slots=True)
class NonManualMarkers:
    """Grammatical facial expressions, eye gaze, and head kinematics."""

    brow_state: str = "neutral"
    brow_raise_score: float = 0.0
    brow_furrow_score: float = 0.0
    head_roll_deg: float = 0.0
    head_yaw_ratio: float = 0.0
    head_yaw: str = "center"
    head_pitch_ratio: float = 0.0
    head_gesture: str = "none"
    eye_gaze: str = "center"
    mouth_state: str = "neutral"
    cheek_puff: bool = False
    sentence_type: str = "statement"
    modifiers: list[str] = field(default_factory=list)
    blendshapes: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "brow_state": self.brow_state,
            "brow_raise_score": round(self.brow_raise_score, 3),
            "brow_furrow_score": round(self.brow_furrow_score, 3),
            "head_roll_deg": round(self.head_roll_deg, 2),
            "head_yaw_ratio": round(self.head_yaw_ratio, 3),
            "head_yaw": self.head_yaw,
            "head_pitch_ratio": round(self.head_pitch_ratio, 3),
            "head_gesture": self.head_gesture,
            "eye_gaze": self.eye_gaze,
            "mouth_state": self.mouth_state,
            "cheek_puff": self.cheek_puff,
            "sentence_type": self.sentence_type,
            "modifiers": list(self.modifiers),
            "blendshapes": {k: round(v, 3) for k, v in self.blendshapes.items()},
        }


@dataclass(slots=True)
class DirectionalEvent:
    """Cross-zone directional verb arc (e.g., Locus A -> Locus B)."""

    action: str
    handedness: str
    from_zone: str
    to_zone: str
    displacement_shoulders: float
    phrases: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "handedness": self.handedness,
            "from_zone": self.from_zone,
            "to_zone": self.to_zone,
            "displacement_shoulders": round(self.displacement_shoulders, 3),
            "phrases": dict(self.phrases),
        }


@dataclass(slots=True)
class SpatialGrammarState:
    """Complete holistic spatial grammar snapshot for a frame."""

    signing_space: SigningSpaceFrame
    non_manual_markers: NonManualMarkers
    referent_focus: str = "center"
    active_heights: dict[str, str] = field(default_factory=dict)
    directional_event: DirectionalEvent | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "signing_space": self.signing_space.to_dict(),
            "non_manual_markers": self.non_manual_markers.to_dict(),
            "referent_focus": self.referent_focus,
            "active_heights": dict(self.active_heights),
            "directional_event": (
                self.directional_event.to_dict()
                if self.directional_event is not None
                else None
            ),
        }


def compute_signing_space(
    pose_landmarks: Sequence[Any] | None,
    *,
    baseline_center_x: float = 0.5,
    left_fallback: float = 0.35,
    right_fallback: float = 0.65,
) -> SigningSpaceFrame:
    """Compute torso-anchored signing space boundaries from Holistic pose landmarks."""
    if not pose_landmarks or len(pose_landmarks) <= POSE_RIGHT_SHOULDER:
        return SigningSpaceFrame(
            source="frame_fallback",
            torso_center_x=0.5,
            shoulder_y=0.45,
            shoulder_width=max(0.18, (right_fallback - left_fallback) / 1.1),
            left_boundary=left_fallback,
            right_boundary=right_fallback,
            head_zone_max_y=0.35,
            waist_zone_min_y=0.72,
            body_shift="center",
            shoulder_tilt_deg=0.0,
        )

    ls_x, ls_y, ls_vis = _lm_xyv(pose_landmarks[POSE_LEFT_SHOULDER])
    rs_x, rs_y, rs_vis = _lm_xyv(pose_landmarks[POSE_RIGHT_SHOULDER])
    raw_shoulder_dx = abs(ls_x - rs_x)

    if ls_vis < 0.2 or rs_vis < 0.2 or raw_shoulder_dx < 0.05:
        return SigningSpaceFrame(
            source="frame_fallback",
            torso_center_x=0.5,
            shoulder_y=0.45,
            shoulder_width=max(0.18, (right_fallback - left_fallback) / 1.1),
            left_boundary=left_fallback,
            right_boundary=right_fallback,
            head_zone_max_y=0.35,
            waist_zone_min_y=0.72,
            body_shift="center",
            shoulder_tilt_deg=0.0,
        )

    torso_center_x = (ls_x + rs_x) / 2.0
    shoulder_y = (ls_y + rs_y) / 2.0
    shoulder_width = max(0.08, raw_shoulder_dx)

    # Dynamic Left / Center / Right signing space scaled around shoulder width.
    half_neutral = 0.55 * shoulder_width
    left_boundary = _clamp(torso_center_x - half_neutral, 0.08, 0.65)
    right_boundary = _clamp(torso_center_x + half_neutral, 0.35, 0.92)
    if right_boundary - left_boundary < 0.14:
        mid = (left_boundary + right_boundary) / 2.0
        left_boundary = _clamp(mid - 0.07, 0.08, 0.80)
        right_boundary = _clamp(left_boundary + 0.14, 0.22, 0.92)

    head_zone_max_y = _clamp(shoulder_y - 0.12 * shoulder_width, 0.05, 0.85)
    waist_zone_min_y = _clamp(shoulder_y + 0.90 * shoulder_width, head_zone_max_y + 0.08, 0.95)

    # Signed shoulder roll in image space (positive = image-right shoulder lower).
    if ls_x >= rs_x:
        shoulder_tilt_deg = math.degrees(math.atan2(ls_y - rs_y, max(ls_x - rs_x, 1e-4)))
    else:
        shoulder_tilt_deg = math.degrees(math.atan2(rs_y - ls_y, max(rs_x - ls_x, 1e-4)))

    # Role-shift detection: use hip midpoint when visible, else neutral torso baseline.
    lean_norm = (torso_center_x - baseline_center_x) / shoulder_width
    if len(pose_landmarks) > POSE_RIGHT_HIP:
        lh_x, lh_y, lh_vis = _lm_xyv(pose_landmarks[POSE_LEFT_HIP])
        rh_x, rh_y, rh_vis = _lm_xyv(pose_landmarks[POSE_RIGHT_HIP])
        if (
            lh_vis >= 0.35
            and rh_vis >= 0.35
            and 0.0 <= lh_y <= 1.05
            and 0.0 <= rh_y <= 1.05
        ):
            hip_center_x = (lh_x + rh_x) / 2.0
            lean_norm = (torso_center_x - hip_center_x) / shoulder_width

    if lean_norm <= -0.18 or (lean_norm <= -0.10 and shoulder_tilt_deg <= -6.0):
        body_shift = "left"
    elif lean_norm >= 0.18 or (lean_norm >= 0.10 and shoulder_tilt_deg >= 6.0):
        body_shift = "right"
    else:
        body_shift = "center"

    return SigningSpaceFrame(
        source="holistic_pose",
        torso_center_x=torso_center_x,
        shoulder_y=shoulder_y,
        shoulder_width=shoulder_width,
        left_boundary=left_boundary,
        right_boundary=right_boundary,
        head_zone_max_y=head_zone_max_y,
        waist_zone_min_y=waist_zone_min_y,
        body_shift=body_shift,
        shoulder_tilt_deg=shoulder_tilt_deg,
    )


def classify_torso_relative_position(
    wrist_x: float,
    wrist_y: float,
    space: SigningSpaceFrame,
) -> tuple[str, str, float, float]:
    """Classify hand wrist into (spatial_zone, height_level, rel_x, rel_y)."""
    if wrist_x < space.left_boundary:
        zone = "left"
    elif wrist_x > space.right_boundary:
        zone = "right"
    else:
        zone = "center"

    if wrist_y < space.head_zone_max_y:
        height_level = "head"
    elif wrist_y > space.waist_zone_min_y:
        height_level = "low"
    else:
        height_level = "chest"

    rel_x = (wrist_x - space.torso_center_x) / max(space.shoulder_width, 0.08)
    rel_y = (wrist_y - space.shoulder_y) / max(space.shoulder_width, 0.08)
    return zone, height_level, rel_x, rel_y


def extract_face_contours(face_landmarks: Sequence[Any] | None) -> dict[str, list[list[float]]]:
    """Extract compact named polyline contours from 478 MediaPipe face landmarks."""
    if not face_landmarks:
        return {}
    n_lms = len(face_landmarks)
    contours: dict[str, list[list[float]]] = {}
    for name, indices in FACE_CONTOUR_PATHS.items():
        if any(idx >= n_lms for idx in indices):
            continue
        contours[name] = [
            [
                round(float(face_landmarks[idx].x), 4),
                round(float(face_landmarks[idx].y), 4),
            ]
            for idx in indices
        ]
    return contours


def extract_blendshape_scores(
    face_blendshapes: Sequence[Any] | None,
) -> dict[str, float]:
    """Extract NMM-relevant ARKit blendshape coefficients from Holistic results."""
    if not face_blendshapes:
        return {}
    scores: dict[str, float] = {}
    key_set = set(_KEY_NMM_BLENDSHAPES)
    for cat in face_blendshapes:
        name = getattr(cat, "category_name", None) or getattr(cat, "display_name", None)
        if name in key_set:
            scores[name] = float(getattr(cat, "score", 0.0) or 0.0)
    return scores


def extract_non_manual_markers(
    face_landmarks: Sequence[Any] | None,
    face_blendshapes: Sequence[Any] | None = None,
    pose_landmarks: Sequence[Any] | None = None,
    *,
    temporal_head_gesture: str = "none",
) -> NonManualMarkers:
    """Compute facial & head non-manual grammatical markers from Holistic output."""
    blend = extract_blendshape_scores(face_blendshapes)
    n_face = len(face_landmarks) if face_landmarks else 0

    head_roll_deg = 0.0
    head_yaw_ratio = 0.0
    head_pitch_ratio = 0.0

    if n_face > FACE_LEFT_CHEEK:
        re_x, re_y, _ = _lm_xyv(face_landmarks[FACE_RIGHT_EYE_OUTER])
        le_x, le_y, _ = _lm_xyv(face_landmarks[FACE_LEFT_EYE_OUTER])
        head_roll_deg = math.degrees(math.atan2(le_y - re_y, max(le_x - re_x, 1e-4)))

        rc_x, _, _ = _lm_xyv(face_landmarks[FACE_RIGHT_CHEEK])
        lc_x, _, _ = _lm_xyv(face_landmarks[FACE_LEFT_CHEEK])
        nose_x, nose_y, _ = _lm_xyv(face_landmarks[FACE_NOSE_TIP])
        face_w = max(lc_x - rc_x, 1e-4)
        head_yaw_ratio = ((nose_x - rc_x) / face_w) - 0.5

        fh_x, fh_y, _ = _lm_xyv(face_landmarks[FACE_FOREHEAD])
        ch_x, ch_y, _ = _lm_xyv(face_landmarks[FACE_CHIN])
        del fh_x, ch_x
        face_h = max(ch_y - fh_y, 1e-4)
        head_pitch_ratio = ((nose_y - fh_y) / face_h) - 0.55
    elif pose_landmarks and len(pose_landmarks) > POSE_RIGHT_EYE:
        le_x, le_y, le_v = _lm_xyv(pose_landmarks[POSE_LEFT_EYE])
        re_x, re_y, re_v = _lm_xyv(pose_landmarks[POSE_RIGHT_EYE])
        nx, ny, n_v = _lm_xyv(pose_landmarks[POSE_NOSE])
        del ny
        if le_v >= 0.3 and re_v >= 0.3 and abs(le_x - re_x) > 1e-3:
            head_roll_deg = math.degrees(math.atan2(le_y - re_y, max(le_x - re_x, 1e-4)))
            if n_v >= 0.3:
                mid_x = (le_x + re_x) / 2.0
                head_yaw_ratio = _clamp((nx - mid_x) / max(abs(le_x - re_x) * 2.0, 1e-4), -0.5, 0.5)

    if head_yaw_ratio <= -0.10:
        head_yaw = "left"
    elif head_yaw_ratio >= 0.10:
        head_yaw = "right"
    else:
        head_yaw = "center"

    # Eyebrow state (Y/N Question raise vs WH-Question furrow)
    if face_blendshapes is not None:
        brow_raise_score = max(
            blend.get("browInnerUp", 0.0),
            (blend.get("browOuterUpLeft", 0.0) + blend.get("browOuterUpRight", 0.0)) / 2.0,
        )
        brow_furrow_score = (
            blend.get("browDownLeft", 0.0) + blend.get("browDownRight", 0.0)
        ) / 2.0
    elif n_face > FACE_LEFT_BROW_MID:
        # Geometric fallback: normalized brow-to-eye gap relative to inter-ocular span
        re_x, _, _ = _lm_xyv(face_landmarks[FACE_RIGHT_EYE_OUTER])
        le_x, _, _ = _lm_xyv(face_landmarks[FACE_LEFT_EYE_OUTER])
        iod = max(abs(le_x - re_x), 1e-4)
        _, rb_y, _ = _lm_xyv(face_landmarks[FACE_RIGHT_BROW_MID])
        _, re_top_y, _ = _lm_xyv(face_landmarks[FACE_RIGHT_EYE_TOP])
        _, lb_y, _ = _lm_xyv(face_landmarks[FACE_LEFT_BROW_MID])
        _, le_top_y, _ = _lm_xyv(face_landmarks[FACE_LEFT_EYE_TOP])
        gap = (((re_top_y - rb_y) + (le_top_y - lb_y)) / 2.0) / iod
        brow_raise_score = _clamp((gap - 0.21) / 0.11, 0.0, 1.0)
        brow_furrow_score = _clamp((0.15 - gap) / 0.07, 0.0, 1.0)
    else:
        brow_raise_score = 0.0
        brow_furrow_score = 0.0

    if brow_furrow_score >= 0.32 and brow_furrow_score >= brow_raise_score:
        brow_state = "furrowed"
    elif brow_raise_score >= 0.38:
        brow_state = "raised"
    else:
        brow_state = "neutral"

    # Eye gaze: iris center inside eye corners + ARKit gaze blendshapes + head yaw cue
    eye_gaze = "center"
    iris_ratios: list[float] = []
    if n_face > FACE_LEFT_IRIS_CENTER:
        r_outer_x, _, _ = _lm_xyv(face_landmarks[FACE_RIGHT_EYE_OUTER])
        r_inner_x, _, _ = _lm_xyv(face_landmarks[FACE_RIGHT_EYE_INNER])
        r_iris_x, _, _ = _lm_xyv(face_landmarks[FACE_RIGHT_IRIS_CENTER])
        if r_inner_x - r_outer_x > 1e-4:
            iris_ratios.append((r_iris_x - r_outer_x) / (r_inner_x - r_outer_x))

        l_inner_x, _, _ = _lm_xyv(face_landmarks[FACE_LEFT_EYE_INNER])
        l_outer_x, _, _ = _lm_xyv(face_landmarks[FACE_LEFT_EYE_OUTER])
        l_iris_x, _, _ = _lm_xyv(face_landmarks[FACE_LEFT_IRIS_CENTER])
        if l_outer_x - l_inner_x > 1e-4:
            iris_ratios.append((l_iris_x - l_inner_x) / (l_outer_x - l_inner_x))

    if iris_ratios:
        avg_iris = sum(iris_ratios) / len(iris_ratios)
        if avg_iris <= 0.41:
            eye_gaze = "left"
        elif avg_iris >= 0.59:
            eye_gaze = "right"

    if eye_gaze == "center" and blend:
        look_img_left = (
            blend.get("eyeLookOutRight", 0.0) + blend.get("eyeLookInLeft", 0.0)
        ) / 2.0
        look_img_right = (
            blend.get("eyeLookOutLeft", 0.0) + blend.get("eyeLookInRight", 0.0)
        ) / 2.0
        if look_img_left >= 0.35 and look_img_left > look_img_right + 0.1:
            eye_gaze = "left"
        elif look_img_right >= 0.35 and look_img_right > look_img_left + 0.1:
            eye_gaze = "right"

    if eye_gaze == "center" and head_yaw != "center":
        eye_gaze = head_yaw

    # Mouth & cheek adverbial markers
    cheek_puff = blend.get("cheekPuff", 0.0) >= 0.35
    if face_blendshapes is not None:
        jaw_open = blend.get("jawOpen", 0.0)
        pucker = max(blend.get("mouthPucker", 0.0), blend.get("mouthFunnel", 0.0))
        smile = (
            blend.get("mouthSmileLeft", 0.0) + blend.get("mouthSmileRight", 0.0)
        ) / 2.0
        if jaw_open >= 0.35:
            mouth_state = "open"
        elif pucker >= 0.38:
            mouth_state = "pucker"
        elif smile >= 0.40:
            mouth_state = "smile"
        else:
            mouth_state = "neutral"
    elif n_face > FACE_RIGHT_EYE_OUTER and n_face > FACE_CHIN:
        _, u_lip_y, _ = _lm_xyv(face_landmarks[FACE_UPPER_LIP_INNER])
        _, l_lip_y, _ = _lm_xyv(face_landmarks[FACE_LOWER_LIP_INNER])
        _, fh_y, _ = _lm_xyv(face_landmarks[FACE_FOREHEAD])
        _, ch_y, _ = _lm_xyv(face_landmarks[FACE_CHIN])
        mouth_ratio = (l_lip_y - u_lip_y) / max(ch_y - fh_y, 1e-4)
        mouth_state = "open" if mouth_ratio >= 0.09 else "neutral"
    else:
        mouth_state = "neutral"

    # Head gesture combines temporal nod/shake with static roll tilt.
    head_gesture = temporal_head_gesture
    if head_gesture == "none":
        if head_roll_deg <= -12.0:
            head_gesture = "tilt_left"
        elif head_roll_deg >= 12.0:
            head_gesture = "tilt_right"

    if head_gesture == "shake":
        sentence_type = "negation"
    elif brow_state == "furrowed":
        sentence_type = "wh_question"
    elif brow_state == "raised":
        sentence_type = "yes_no_question"
    elif head_gesture == "nod":
        sentence_type = "affirmation"
    else:
        sentence_type = "statement"

    modifiers: list[str] = []
    if brow_state == "raised":
        modifiers.append("brow_raise")
    elif brow_state == "furrowed":
        modifiers.append("brow_furrow")
    if head_gesture != "none":
        modifiers.append(f"head_{head_gesture}")
    if cheek_puff:
        modifiers.append("cheek_puff")
    if mouth_state != "neutral":
        modifiers.append(f"mouth_{mouth_state}")
    if eye_gaze != "center":
        modifiers.append(f"gaze_{eye_gaze}")

    return NonManualMarkers(
        brow_state=brow_state,
        brow_raise_score=brow_raise_score,
        brow_furrow_score=brow_furrow_score,
        head_roll_deg=head_roll_deg,
        head_yaw_ratio=head_yaw_ratio,
        head_yaw=head_yaw,
        head_pitch_ratio=head_pitch_ratio,
        head_gesture=head_gesture,
        eye_gaze=eye_gaze,
        mouth_state=mouth_state,
        cheek_puff=cheek_puff,
        sentence_type=sentence_type,
        modifiers=modifiers,
        blendshapes=blend,
    )


def apply_nmm_to_phrase(
    phrase: str,
    nmm: NonManualMarkers | None,
    *,
    lang: str = "en",
) -> str:
    """Optionally inflect a translated sign phrase with active NMM clause markers."""
    if not phrase or nmm is None:
        return phrase
    if nmm.sentence_type == "negation":
        prefix = {"en": "Not: ", "hi": "नहीं: ", "kn": "ಇಲ್ಲ: "}.get(lang, "Not: ")
        return phrase if phrase.lower().startswith("no") else f"{prefix}{phrase}"
    if nmm.sentence_type in {"yes_no_question", "wh_question"}:
        return phrase if phrase.endswith("?") else f"{phrase}?"
    return phrase


@dataclass(slots=True)
class _HandSample:
    timestamp_s: float
    rel_x: float
    rel_y: float
    zone: str
    gesture: str


def _count_oscillation_reversals(values: Sequence[float], min_step: float) -> int:
    """Count significant direction reversals in a 1D signal (for nod/shake)."""
    if len(values) < 3:
        return 0
    reversals = 0
    direction = 0
    extreme = values[0]
    for v in values[1:]:
        diff = v - extreme
        if direction == 0:
            if abs(diff) >= min_step:
                direction = 1 if diff > 0 else -1
                extreme = v
        elif direction == 1:
            if v > extreme:
                extreme = v
            elif extreme - v >= min_step:
                reversals += 1
                direction = -1
                extreme = v
        else:
            if v < extreme:
                extreme = v
            elif v - extreme >= min_step:
                reversals += 1
                direction = 1
                extreme = v
    return reversals


class SpatialGrammarTracker:
    """Temporal tracker for torso baseline, head nod/shake, and directional verb arcs."""

    def __init__(
        self,
        *,
        head_window_seconds: float = 1.1,
        hand_window_seconds: float = 1.5,
        min_arc_shoulders: float = 0.70,
        trajectory_cooldown_seconds: float = 2.0,
    ) -> None:
        self.head_window_seconds = head_window_seconds
        self.hand_window_seconds = hand_window_seconds
        self.min_arc_shoulders = min_arc_shoulders
        self.trajectory_cooldown_seconds = trajectory_cooldown_seconds

        self._baseline_torso_x: float = 0.5
        self._has_torso_baseline: bool = False
        self._head_history: deque[tuple[float, float, float]] = deque()
        self._hand_history: dict[str, deque[_HandSample]] = {}
        self._last_trajectory_at: dict[str, float] = {}

    def reset(self) -> None:
        self._baseline_torso_x = 0.5
        self._has_torso_baseline = False
        self._head_history.clear()
        self._hand_history.clear()
        self._last_trajectory_at.clear()

    def _detect_temporal_head_gesture(
        self,
        ts_s: float,
        yaw_ratio: float,
        pitch_ratio: float,
    ) -> str:
        self._head_history.append((ts_s, yaw_ratio, pitch_ratio))
        cutoff = ts_s - self.head_window_seconds
        while self._head_history and self._head_history[0][0] < cutoff:
            self._head_history.popleft()

        if len(self._head_history) < 4:
            return "none"

        yaws = [item[1] for item in self._head_history]
        pitches = [item[2] for item in self._head_history]
        yaw_span = max(yaws) - min(yaws)
        pitch_span = max(pitches) - min(pitches)

        shake_reversals = _count_oscillation_reversals(yaws, min_step=0.065)
        nod_reversals = _count_oscillation_reversals(pitches, min_step=0.048)

        if shake_reversals >= 2 and yaw_span >= 0.11 and yaw_span >= pitch_span:
            return "shake"
        if nod_reversals >= 2 and pitch_span >= 0.08:
            return "nod"
        return "none"

    def _detect_directional_event(
        self,
        ts_s: float,
        handedness: str,
        rel_x: float,
        rel_y: float,
        zone: str,
        gesture: str,
    ) -> DirectionalEvent | None:
        hist = self._hand_history.setdefault(handedness, deque())
        hist.append(
            _HandSample(
                timestamp_s=ts_s,
                rel_x=rel_x,
                rel_y=rel_y,
                zone=zone,
                gesture=gesture,
            )
        )
        cutoff = ts_s - self.hand_window_seconds
        while hist and hist[0].timestamp_s < cutoff:
            hist.popleft()

        if len(hist) < 3:
            return None

        last_fired = self._last_trajectory_at.get(handedness, -1e9)
        if (ts_s - last_fired) < self.trajectory_cooldown_seconds:
            return None

        latest = hist[-1]
        prev = hist[-2]
        is_settling = abs(latest.rel_x - prev.rel_x) < 0.12

        # Look back for the strongest coherent directional arc ending at the current zone.
        best_candidate: tuple[int, float, _HandSample] | None = None
        samples = list(hist)
        for idx, start in enumerate(samples[:-2]):
            dt = latest.timestamp_s - start.timestamp_s
            if dt < 0.18 or dt > self.hand_window_seconds:
                continue
            if start.zone == latest.zone:
                continue

            is_cross_space = {start.zone, latest.zone} == {"left", "right"}
            # Let cross-space A <-> B sweeps finish crossing center before triggering *_TO_SELF.
            if not is_cross_space and not is_settling:
                continue

            net_dx = latest.rel_x - start.rel_x
            if abs(net_dx) < self.min_arc_shoulders:
                continue

            # Require path coherence so jitter does not masquerade as an agreement verb.
            sub_path = sum(
                abs(samples[j + 1].rel_x - samples[j].rel_x)
                for j in range(idx, len(samples) - 1)
            )
            if sub_path <= 1e-4 or (abs(net_dx) / sub_path) < 0.55:
                continue

            priority = 1 if is_cross_space else 0
            if best_candidate is None or (priority, abs(net_dx)) > (
                best_candidate[0],
                best_candidate[1],
            ):
                best_candidate = (priority, abs(net_dx), start)

        if best_candidate is None:
            return None

        _, displacement, start = best_candidate
        action_map = {
            ("left", "right"): "GIVE_A_TO_B",
            ("right", "left"): "GIVE_B_TO_A",
            ("center", "left"): "SELF_TO_A",
            ("center", "right"): "SELF_TO_B",
            ("left", "center"): "A_TO_SELF",
            ("right", "center"): "B_TO_SELF",
        }
        action = action_map.get((start.zone, latest.zone))
        if action is None:
            return None

        self._last_trajectory_at[handedness] = ts_s
        hist.clear()
        return DirectionalEvent(
            action=action,
            handedness=handedness,
            from_zone=start.zone,
            to_zone=latest.zone,
            displacement_shoulders=displacement,
            phrases=dict(DIRECTIONAL_PHRASES.get(action, {})),
        )

    def update(
        self,
        *,
        timestamp_ms: int,
        pose_landmarks: Sequence[Any] | None = None,
        face_landmarks: Sequence[Any] | None = None,
        face_blendshapes: Sequence[Any] | None = None,
        hand_entries: Sequence[tuple[str, float, float, str]] = (),
        left_fallback: float = 0.35,
        right_fallback: float = 0.65,
        temporal: bool = True,
    ) -> SpatialGrammarState:
        """Update spatial grammar state for one video/live frame or single image."""
        space = compute_signing_space(
            pose_landmarks,
            baseline_center_x=self._baseline_torso_x,
            left_fallback=left_fallback,
            right_fallback=right_fallback,
        )

        if temporal and space.source == "holistic_pose":
            if not self._has_torso_baseline:
                self._baseline_torso_x = space.torso_center_x
                self._has_torso_baseline = True
            elif abs(space.shoulder_tilt_deg) < 6.0 and space.body_shift == "center":
                self._baseline_torso_x = (
                    0.97 * self._baseline_torso_x + 0.03 * space.torso_center_x
                )

        raw_nmm = extract_non_manual_markers(
            face_landmarks,
            face_blendshapes,
            pose_landmarks,
        )
        ts_s = timestamp_ms / 1000.0
        temporal_head = (
            self._detect_temporal_head_gesture(
                ts_s, raw_nmm.head_yaw_ratio, raw_nmm.head_pitch_ratio
            )
            if temporal
            else "none"
        )
        nmm = (
            extract_non_manual_markers(
                face_landmarks,
                face_blendshapes,
                pose_landmarks,
                temporal_head_gesture=temporal_head,
            )
            if temporal_head != "none"
            else raw_nmm
        )

        active_heights: dict[str, str] = {}
        directional_event: DirectionalEvent | None = None
        for handedness, wrist_x, wrist_y, gesture in hand_entries:
            zone, height_level, rel_x, _rel_y = classify_torso_relative_position(
                wrist_x, wrist_y, space
            )
            active_heights[handedness] = height_level
            if temporal and directional_event is None:
                directional_event = self._detect_directional_event(
                    ts_s, handedness, rel_x, _rel_y, zone, gesture
                )

        # Combine gaze, head turn, and body shift into active referent locus focus.
        if space.body_shift in {"left", "right"}:
            referent_focus = space.body_shift
        elif nmm.head_yaw in {"left", "right"}:
            referent_focus = nmm.head_yaw
        elif nmm.eye_gaze in {"left", "right"}:
            referent_focus = nmm.eye_gaze
        else:
            referent_focus = "center"

        return SpatialGrammarState(
            signing_space=space,
            non_manual_markers=nmm,
            referent_focus=referent_focus,
            active_heights=active_heights,
            directional_event=directional_event,
        )
