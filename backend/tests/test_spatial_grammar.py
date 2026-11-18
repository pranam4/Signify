"""Unit and integration tests for Holistic Spatial Grammar & Non-Manual Markers."""

from types import SimpleNamespace

from signbridge.config import RecognizerConfig
from signbridge.recognizer import build_prediction
from signbridge.spatial_grammar import (
    SpatialGrammarTracker,
    apply_nmm_to_phrase,
    classify_torso_relative_position,
    compute_signing_space,
    extract_face_contours,
    extract_non_manual_markers,
)
from signbridge.stabilizer import GestureStabilizer


def _pt(x: float, y: float, z: float = 0.0, visibility: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z, visibility=visibility)


def _cat(name: str, score: float) -> SimpleNamespace:
    return SimpleNamespace(category_name=name, display_name=name, score=score)


def _make_pose(
    *,
    ls_x: float = 0.62,
    ls_y: float = 0.48,
    rs_x: float = 0.38,
    rs_y: float = 0.48,
    lh_x: float | None = None,
    rh_x: float | None = None,
) -> list[SimpleNamespace]:
    lms = [_pt(0.5, 0.25) for _ in range(33)]
    lms[11] = _pt(ls_x, ls_y)
    lms[12] = _pt(rs_x, rs_y)
    if lh_x is not None and rh_x is not None:
        lms[23] = _pt(lh_x, 0.85)
        lms[24] = _pt(rh_x, 0.85)
    else:
        lms[23] = _pt(ls_x, 1.15, visibility=0.1)
        lms[24] = _pt(rs_x, 1.15, visibility=0.1)
    return lms


def _make_face(
    *,
    nose_x: float = 0.50,
    nose_y: float = 0.27,
    roll_dy: float = 0.0,
    r_iris_offset: float = 0.5,
    l_iris_offset: float = 0.5,
    brow_gap: float = 0.017,
) -> list[SimpleNamespace]:
    lms = [_pt(0.5, 0.25) for _ in range(478)]
    lms[10] = _pt(0.50, 0.14)   # Forehead
    lms[152] = _pt(0.50, 0.38)  # Chin
    lms[234] = _pt(0.42, 0.26)  # Image-left cheek
    lms[454] = _pt(0.58, 0.26)  # Image-right cheek
    lms[1] = _pt(nose_x, nose_y)

    # Eyes & brows
    lms[33] = _pt(0.45, 0.23 - roll_dy)
    lms[133] = _pt(0.48, 0.23 - roll_dy)
    lms[159] = _pt(0.465, 0.225 - roll_dy)
    lms[105] = _pt(0.465, 0.225 - brow_gap - roll_dy)

    lms[362] = _pt(0.52, 0.23 + roll_dy)
    lms[263] = _pt(0.55, 0.23 + roll_dy)
    lms[386] = _pt(0.535, 0.225 + roll_dy)
    lms[334] = _pt(0.535, 0.225 - brow_gap + roll_dy)

    # Irises inside eye spans
    lms[468] = _pt(0.45 + r_iris_offset * (0.48 - 0.45), 0.23)
    lms[473] = _pt(0.52 + l_iris_offset * (0.55 - 0.52), 0.23)

    # Lips inner
    lms[13] = _pt(0.50, 0.32)
    lms[14] = _pt(0.50, 0.33)
    return lms


def test_torso_relative_signing_space_and_fallback() -> None:
    fallback = compute_signing_space(None)
    assert fallback.source == "frame_fallback"
    assert fallback.left_boundary == 0.35
    assert fallback.right_boundary == 0.65

    # Signer standing off-center on the left (center=0.30, shoulder_w=0.20)
    left_shifted_pose = _make_pose(ls_x=0.40, rs_x=0.20, ls_y=0.50, rs_y=0.50)
    space = compute_signing_space(left_shifted_pose, baseline_center_x=0.30)
    assert space.source == "holistic_pose"
    assert abs(space.torso_center_x - 0.30) < 1e-4
    assert space.left_boundary < 0.25
    assert space.right_boundary < 0.45

    # Hand at x=0.48 is to the RIGHT of a signer centered at x=0.30!
    zone, height, rel_x, _ = classify_torso_relative_position(0.48, 0.32, space)
    assert zone == "right"
    assert height == "head"
    assert rel_x > 0.5

    # Role shift toward right relative to hips
    lean_right_pose = _make_pose(
        ls_x=0.68, rs_x=0.48, lh_x=0.60, rh_x=0.40
    )
    shifted = compute_signing_space(lean_right_pose)
    assert shifted.body_shift == "right"


def test_non_manual_markers_blendshapes_and_geometry() -> None:
    face = _make_face()
    yn_nmm = extract_non_manual_markers(
        face,
        [_cat("browInnerUp", 0.72), _cat("cheekPuff", 0.55)],
    )
    assert yn_nmm.brow_state == "raised"
    assert yn_nmm.cheek_puff is True
    assert yn_nmm.sentence_type == "yes_no_question"
    assert "brow_raise" in yn_nmm.modifiers
    assert "cheek_puff" in yn_nmm.modifiers
    assert apply_nmm_to_phrase("I love you", yn_nmm) == "I love you?"

    wh_nmm = extract_non_manual_markers(
        face,
        [_cat("browDownLeft", 0.62), _cat("browDownRight", 0.64)],
    )
    assert wh_nmm.brow_state == "furrowed"
    assert wh_nmm.sentence_type == "wh_question"

    # Gaze left via iris position inside eye corners + tilted head
    gaze_left_face = _make_face(r_iris_offset=0.20, l_iris_offset=0.22, roll_dy=0.02)
    gaze_nmm = extract_non_manual_markers(gaze_left_face, [])
    assert gaze_nmm.eye_gaze == "left"
    assert gaze_nmm.head_gesture == "tilt_right"

    # Geometric brow-raise fallback when blendshapes list is omitted
    high_brow_face = _make_face(brow_gap=0.032)
    geom_nmm = extract_non_manual_markers(high_brow_face, None)
    assert geom_nmm.brow_state == "raised"


def test_temporal_head_shake_nod_and_directional_verb_arc() -> None:
    tracker = SpatialGrammarTracker(
        min_arc_shoulders=0.70,
        trajectory_cooldown_seconds=2.0,
    )
    pose = _make_pose(ls_x=0.62, rs_x=0.38)

    # Head shake oscillation -> negation
    yaw_wave = [0.50, 0.46, 0.54, 0.46, 0.53]
    last_state = None
    for idx, nx in enumerate(yaw_wave):
        last_state = tracker.update(
            timestamp_ms=100 + idx * 120,
            pose_landmarks=pose,
            face_landmarks=_make_face(nose_x=nx),
            face_blendshapes=[],
            temporal=True,
        )
    assert last_state is not None
    assert last_state.non_manual_markers.head_gesture == "shake"
    assert last_state.non_manual_markers.sentence_type == "negation"
    assert apply_nmm_to_phrase("Yes / Confirm", last_state.non_manual_markers) == "Not: Yes / Confirm"

    # Head nod oscillation -> affirmation
    tracker.reset()
    pitch_wave = [0.27, 0.24, 0.30, 0.24, 0.29]
    for idx, ny in enumerate(pitch_wave):
        last_state = tracker.update(
            timestamp_ms=1000 + idx * 120,
            pose_landmarks=pose,
            face_landmarks=_make_face(nose_y=ny),
            face_blendshapes=[],
            temporal=True,
        )
    assert last_state is not None
    assert last_state.non_manual_markers.head_gesture == "nod"
    assert last_state.non_manual_markers.sentence_type == "affirmation"

    # Directional verb trajectory across Left -> Center -> Right (GIVE_A_TO_B)
    tracker.reset()
    xs = [0.22, 0.32, 0.48, 0.64, 0.78]
    fired = None
    for idx, wx in enumerate(xs):
        state = tracker.update(
            timestamp_ms=2000 + idx * 150,
            pose_landmarks=pose,
            face_landmarks=_make_face(),
            face_blendshapes=[],
            hand_entries=[("Right", wx, 0.55, "Open_Palm")],
            temporal=True,
        )
        if state.directional_event is not None:
            fired = state.directional_event
    assert fired is not None
    assert fired.action == "GIVE_A_TO_B"
    assert fired.from_zone == "left"
    assert fired.to_zone == "right"
    assert "Person A gives to Person B" in fired.phrases["en"]

    # Immediate re-crossing during cooldown should not spam
    during_cd = tracker.update(
        timestamp_ms=2800,
        pose_landmarks=pose,
        face_landmarks=_make_face(),
        face_blendshapes=[],
        hand_entries=[("Right", 0.20, 0.55, "Open_Palm")],
        temporal=True,
    )
    assert during_cd.directional_event is None


def test_face_contours_and_holistic_hand_merge_in_prediction() -> None:
    face = _make_face()
    contours = extract_face_contours(face)
    assert "face_oval" in contours
    assert "left_eyebrow" in contours
    assert "right_iris" in contours
    assert len(contours["left_eyebrow"]) == 5

    # GestureRecognizer returned no hands, while Holistic detected Left hand in left zone
    empty_gesture_res = SimpleNamespace(
        hand_landmarks=[],
        hand_world_landmarks=[],
        handedness=[],
        gestures=[],
    )
    holistic_res = SimpleNamespace(
        pose_landmarks=_make_pose(ls_x=0.62, rs_x=0.38),
        pose_world_landmarks=[],
        face_landmarks=face,
        face_blendshapes=[_cat("browInnerUp", 0.65)],
        left_hand_landmarks=[_pt(0.20, 0.52) for _ in range(21)],
        left_hand_world_landmarks=[],
        right_hand_landmarks=[],
        right_hand_world_landmarks=[],
    )

    pred = build_prediction(
        empty_gesture_res,
        timestamp_ms=500,
        config=RecognizerConfig(enable_holistic=True),
        stabilizer=GestureStabilizer(),
        holistic_result=holistic_res,
        spatial_tracker=SpatialGrammarTracker(),
    )
    assert len(pred.hands) == 1
    assert pred.hands[0].handedness == "Left"
    assert pred.hands[0].spatial_zone == "left"
    payload = pred.to_ws_dict()
    assert payload["holistic"]["enabled"] is True
    assert payload["holistic"]["landmark_counts"]["face"] == 478
    assert "face_landmarks" not in payload["holistic"]  # Compact by default
    assert payload["spatial_grammar"]["non_manual_markers"]["sentence_type"] == "yes_no_question"

    full_payload = pred.to_ws_dict(include_face_mesh=True)
    assert len(full_payload["holistic"]["face_landmarks"]) == 478
