"""Standalone OpenCV webcam recognizer using MediaPipe LIVE_STREAM + Holistic Landmarker."""

import threading
import time

import cv2
import mediapipe as mp

from .config import GESTURE_TRANSLATIONS, POSE_CONNECTIONS, RecognizerConfig
from .model_manager import ensure_model_asset
from .recognizer import FramePrediction, build_prediction
from .spatial_grammar import SpatialGrammarTracker
from .stabilizer import GestureStabilizer

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
HolisticLandmarkerResult = mp.tasks.vision.HolisticLandmarkerResult
RunningMode = mp.tasks.vision.RunningMode
HandLandmarksConnections = mp.tasks.vision.HandLandmarksConnections
drawing_utils = mp.tasks.vision.drawing_utils
drawing_styles = mp.tasks.vision.drawing_styles


def draw_hud(frame, prediction: FramePrediction | None) -> None:
    """Draw holistic pose/face/hand landmarks, body-anchored zones, and NMM banner."""
    h, w = frame.shape[:2]

    left_b = 0.35
    right_b = 0.65
    if prediction is not None and prediction.spatial_grammar is not None:
        left_b = prediction.spatial_grammar.signing_space.left_boundary
        right_b = prediction.spatial_grammar.signing_space.right_boundary

    # Dynamic body-anchored spatial zone dividers (Left / Center / Right)
    lx = int(left_b * w)
    rx = int(right_b * w)
    cv2.line(frame, (lx, 0), (lx, h), (100, 116, 139), 1, cv2.LINE_AA)
    cv2.line(frame, (rx, 0), (rx, h), (100, 116, 139), 1, cv2.LINE_AA)

    if prediction is not None:
        if prediction.holistic and prediction.holistic.get("enabled"):
            pose_lms = prediction.holistic.get("pose_landmarks") or []
            for a_idx, b_idx in POSE_CONNECTIONS:
                if a_idx < len(pose_lms) and b_idx < len(pose_lms):
                    pa, pb = pose_lms[a_idx], pose_lms[b_idx]
                    if pa.get("v", 1.0) >= 0.25 and pb.get("v", 1.0) >= 0.25:
                        cv2.line(
                            frame,
                            (int(pa["x"] * w), int(pa["y"] * h)),
                            (int(pb["x"] * w), int(pb["y"] * h)),
                            (129, 140, 248),
                            2,
                            cv2.LINE_AA,
                        )
            contours = prediction.holistic.get("face_contours") or {}
            for key, pts in contours.items():
                color = (250, 204, 21) if "eyebrow" in key else (148, 163, 184)
                for idx in range(len(pts) - 1):
                    cv2.line(
                        frame,
                        (int(pts[idx][0] * w), int(pts[idx][1] * h)),
                        (int(pts[idx + 1][0] * w), int(pts[idx + 1][1] * h)),
                        color,
                        1,
                        cv2.LINE_AA,
                    )

        for hand in prediction.hands:
            pts = [
                mp.tasks.components.containers.NormalizedLandmark(
                    x=p.x, y=p.y, z=p.z
                )
                for p in hand.landmarks
            ]
            drawing_utils.draw_landmarks(
                frame,
                pts,
                HandLandmarksConnections.HAND_CONNECTIONS,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style(),
            )
            wx = int(hand.wrist_xy[0] * w)
            wy = max(28, int(hand.wrist_xy[1] * h) - 14)
            label = (
                f"{hand.handedness}: {hand.gesture} ({hand.gesture_score:.0%}) "
                f"[{hand.spatial_zone.upper()}/{hand.height_level}]"
            )
            cv2.putText(
                frame,
                label,
                (max(10, wx - 60), wy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (56, 189, 248),
                2,
                cv2.LINE_AA,
            )

        banner = "Show a sign..."
        if prediction.locked_label:
            phrase = GESTURE_TRANSLATIONS.get(prediction.locked_label, {}).get(
                "en", prediction.locked_label
            )
            banner = f"LOCKED: {phrase} ({prediction.locked_score:.0%})"
        elif prediction.candidate_label:
            banner = f"Holding: {prediction.candidate_label} ({prediction.stability_progress:.0%})"

        if prediction.spatial_grammar is not None:
            nmm = prediction.spatial_grammar.non_manual_markers
            banner = f"{banner} | Mood: {nmm.sentence_type} | Focus: {prediction.spatial_grammar.referent_focus.upper()}"

        cv2.rectangle(frame, (0, h - 48), (w, h), (15, 23, 42), -1)
        cv2.putText(
            frame,
            banner,
            (16, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (248, 250, 252),
            2,
            cv2.LINE_AA,
        )


def run_webcam_demo(camera_index: int = 0) -> None:
    """Open local webcam and run MediaPipe GestureRecognizer + HolisticLandmarker."""
    config = RecognizerConfig(invert_handedness=True)
    model_path = ensure_model_asset(config.model_path, config.model_url)
    holistic_path = ensure_model_asset(
        config.holistic_model_path, config.holistic_model_url
    )
    stabilizer = GestureStabilizer(
        window_size=config.stability_window,
        min_votes=config.stability_min_votes,
        min_score=config.stability_min_score,
        cooldown_seconds=config.cooldown_seconds,
    )
    spatial_tracker = SpatialGrammarTracker()

    lock = threading.Lock()
    latest: dict[str, FramePrediction | None] = {"prediction": None}
    latest_holistic: dict[str, tuple[int, HolisticLandmarkerResult] | None] = {
        "value": None
    }

    def on_holistic(
        result: HolisticLandmarkerResult,
        _output_image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        with lock:
            latest_holistic["value"] = (timestamp_ms, result)

    def on_result(
        result: GestureRecognizerResult,
        _output_image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        with lock:
            hol_pair = latest_holistic["value"]
            hol_res = (
                hol_pair[1]
                if hol_pair is not None and abs(timestamp_ms - hol_pair[0]) <= 350
                else None
            )
        pred = build_prediction(
            result,
            timestamp_ms,
            config,
            stabilizer,
            holistic_result=hol_res,
            spatial_tracker=spatial_tracker,
        )
        with lock:
            latest["prediction"] = pred

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.LIVE_STREAM,
        num_hands=config.num_hands,
        min_hand_detection_confidence=config.min_hand_detection_confidence,
        min_hand_presence_confidence=config.min_hand_presence_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
        result_callback=on_result,
    )
    holistic_options = HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(holistic_path)),
        running_mode=RunningMode.LIVE_STREAM,
        output_face_blendshapes=config.output_face_blendshapes,
        result_callback=on_holistic,
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open webcam index {camera_index}.")

    last_ts = 0
    try:
        with (
            GestureRecognizer.create_from_options(options) as recognizer,
            HolisticLandmarker.create_from_options(holistic_options) as holistic,
        ):
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break

                # Mirror for natural selfie signing orientation
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                ts = int(time.monotonic() * 1000)
                if ts <= last_ts:
                    ts = last_ts + 1
                last_ts = ts

                recognizer.recognize_async(mp_image, ts)
                holistic.detect_async(mp_image, ts)

                with lock:
                    pred = latest["prediction"]

                draw_hud(frame, pred)
                cv2.imshow("Signify Python Holistic MediaPipe Demo (ESC / Q to quit)", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam_demo()
