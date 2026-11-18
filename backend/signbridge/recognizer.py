"""Core MediaPipe GestureRecognizer + HolisticLandmarker wrapper and result extraction."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
import threading
import time
from typing import Any, Callable

import cv2
import mediapipe as mp
import numpy as np

from .config import (
    RUNNING_MODES,
    RecognizerConfig,
    translate_gesture_label,
)
from .model_manager import ensure_model_asset
from .spatial_grammar import (
    SpatialGrammarState,
    SpatialGrammarTracker,
    classify_torso_relative_position,
    compute_signing_space,
    extract_face_contours,
)
from .stabilizer import GestureStabilizer

BaseOptions = mp.tasks.BaseOptions
ClassifierOptions = mp.tasks.components.processors.ClassifierOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
HolisticLandmarker = mp.tasks.vision.HolisticLandmarker
HolisticLandmarkerOptions = mp.tasks.vision.HolisticLandmarkerOptions
HolisticLandmarkerResult = mp.tasks.vision.HolisticLandmarkerResult
RunningMode = mp.tasks.vision.RunningMode

_RUNNING_MODE_MAP: dict[str, Any] = {
    "IMAGE": RunningMode.IMAGE,
    "VIDEO": RunningMode.VIDEO,
    "LIVE_STREAM": RunningMode.LIVE_STREAM,
}


@dataclass(slots=True)
class LandmarkPoint:
    x: float
    y: float
    z: float
    visibility: float | None = None


@dataclass(slots=True)
class GestureCandidate:
    category_name: str
    score: float
    display_name: str = ""


@dataclass(slots=True)
class HandObservation:
    handedness: str
    handedness_score: float
    gesture: str
    gesture_score: float
    spatial_zone: str
    wrist_xy: tuple[float, float]
    landmarks: list[LandmarkPoint]
    world_landmarks: list[LandmarkPoint]
    top_gestures: list[GestureCandidate] = field(default_factory=list)
    height_level: str = "chest"
    rel_xy: tuple[float, float] = (0.0, 0.0)


@dataclass(slots=True)
class FramePrediction:
    timestamp_ms: int
    hands: list[HandObservation]
    candidate_label: str | None
    stability_progress: float
    locked_label: str | None
    locked_score: float
    spatial_zone: str
    phrases: dict[str, str]
    holistic: dict[str, Any] | None = None
    spatial_grammar: SpatialGrammarState | None = None

    def to_ws_dict(
        self,
        include_world_landmarks: bool = False,
        include_face_mesh: bool = False,
    ) -> dict[str, Any]:
        """Serialize prediction for low-latency WebSocket streaming and REST responses."""
        hands_payload = []
        for hand in self.hands:
            item: dict[str, Any] = {
                "handedness": hand.handedness,
                "handedness_score": round(hand.handedness_score, 3),
                "gesture": hand.gesture,
                "gesture_score": round(hand.gesture_score, 3),
                "top_gestures": [
                    {
                        "category": g.category_name,
                        "score": round(g.score, 3),
                    }
                    for g in hand.top_gestures
                ],
                "spatial_zone": hand.spatial_zone,
                "height_level": hand.height_level,
                "rel_xy": [round(hand.rel_xy[0], 3), round(hand.rel_xy[1], 3)],
                "wrist_xy": [round(hand.wrist_xy[0], 4), round(hand.wrist_xy[1], 4)],
                "landmarks": [
                    {"x": round(p.x, 4), "y": round(p.y, 4), "z": round(p.z, 4)}
                    for p in hand.landmarks
                ],
            }
            if include_world_landmarks:
                item["world_landmarks"] = [
                    {"x": p.x, "y": p.y, "z": p.z} for p in hand.world_landmarks
                ]
            hands_payload.append(item)

        payload: dict[str, Any] = {
            "timestamp_ms": self.timestamp_ms,
            "hands": hands_payload,
            "candidate_label": self.candidate_label,
            "stability_progress": round(self.stability_progress, 3),
            "locked_label": self.locked_label,
            "locked_score": round(self.locked_score, 3),
            "spatial_zone": self.spatial_zone,
            "phrases": self.phrases,
        }

        if self.holistic is not None:
            holistic_out = dict(self.holistic)
            if not include_face_mesh:
                holistic_out.pop("face_landmarks", None)
            if not include_world_landmarks:
                holistic_out.pop("pose_world_landmarks", None)
            payload["holistic"] = holistic_out

        if self.spatial_grammar is not None:
            payload["spatial_grammar"] = self.spatial_grammar.to_dict()

        return payload


def classify_spatial_zone(
    wrist_x: float,
    left_max_x: float = 0.35,
    right_min_x: float = 0.65,
) -> str:
    """Map normalized wrist X coordinate into Left / Center / Right signing space."""
    if wrist_x < left_max_x:
        return "left"
    if wrist_x > right_min_x:
        return "right"
    return "center"


def _build_classifier_options(
    *,
    display_names_locale: str | None = None,
    max_results: int | None = None,
    score_threshold: float | None = None,
    category_allowlist: list[str] | None = None,
    category_denylist: list[str] | None = None,
) -> ClassifierOptions:
    """Create MediaPipe ClassifierOptions while validating allowlist/denylist."""
    if category_allowlist and category_denylist:
        raise ValueError(
            "category_allowlist and category_denylist are mutually exclusive."
        )
    return ClassifierOptions(
        display_names_locale=display_names_locale,
        max_results=max_results,
        score_threshold=score_threshold,
        category_allowlist=list(category_allowlist) if category_allowlist else None,
        category_denylist=list(category_denylist) if category_denylist else None,
    )


def _swap_handedness(name: str) -> str:
    if name == "Left":
        return "Right"
    if name == "Right":
        return "Left"
    return name


def _landmarks_to_points(raw_lms: Any) -> list[LandmarkPoint]:
    return [
        LandmarkPoint(
            x=float(p.x),
            y=float(p.y),
            z=float(getattr(p, "z", 0.0) or 0.0),
            visibility=(
                float(p.visibility)
                if getattr(p, "visibility", None) is not None
                else None
            ),
        )
        for p in (raw_lms or [])
    ]


def extract_observations(
    result: GestureRecognizerResult,
    config: RecognizerConfig,
    *,
    signing_space: Any = None,
) -> list[HandObservation]:
    """Convert MediaPipe GestureRecognizerResult into typed HandObservation items."""
    observations: list[HandObservation] = []
    num_detected = len(result.hand_landmarks or [])

    for idx in range(num_detected):
        raw_lms = result.hand_landmarks[idx]
        landmarks = _landmarks_to_points(raw_lms)

        raw_world = (
            result.hand_world_landmarks[idx]
            if result.hand_world_landmarks and idx < len(result.hand_world_landmarks)
            else []
        )
        world_landmarks = _landmarks_to_points(raw_world)

        handedness_name = "Unknown"
        handedness_score = 0.0
        if result.handedness and idx < len(result.handedness) and result.handedness[idx]:
            top_hand = result.handedness[idx][0]
            handedness_name = top_hand.category_name or top_hand.display_name or "Unknown"
            handedness_score = float(top_hand.score or 0.0)
            if config.invert_handedness:
                handedness_name = _swap_handedness(handedness_name)

        top_gestures: list[GestureCandidate] = []
        if result.gestures and idx < len(result.gestures) and result.gestures[idx]:
            for cat in result.gestures[idx]:
                name = cat.category_name or cat.display_name or "None"
                top_gestures.append(
                    GestureCandidate(
                        category_name=name,
                        score=float(cat.score or 0.0),
                        display_name=cat.display_name or "",
                    )
                )

        gesture_name = top_gestures[0].category_name if top_gestures else "None"
        gesture_score = top_gestures[0].score if top_gestures else 0.0

        wrist = landmarks[0] if landmarks else LandmarkPoint(0.5, 0.5, 0.0)
        if signing_space is not None:
            zone, height_level, rel_x, rel_y = classify_torso_relative_position(
                wrist.x, wrist.y, signing_space
            )
        else:
            zone = classify_spatial_zone(
                wrist.x,
                left_max_x=config.left_zone_max_x,
                right_min_x=config.right_zone_min_x,
            )
            height_level = "chest"
            rel_x = (wrist.x - 0.5) / 0.3
            rel_y = (wrist.y - 0.5) / 0.3

        observations.append(
            HandObservation(
                handedness=handedness_name,
                handedness_score=handedness_score,
                gesture=gesture_name,
                gesture_score=gesture_score,
                spatial_zone=zone,
                wrist_xy=(wrist.x, wrist.y),
                landmarks=landmarks,
                world_landmarks=world_landmarks,
                top_gestures=top_gestures,
                height_level=height_level,
                rel_xy=(rel_x, rel_y),
            )
        )

    return observations


def _merge_holistic_hands(
    hands: list[HandObservation],
    holistic_result: HolisticLandmarkerResult,
    config: RecognizerConfig,
    signing_space: Any,
) -> list[HandObservation]:
    """Fill in any missing hand skeleton detected by HolisticLandmarker."""
    if len(hands) >= config.num_hands:
        return hands

    present = {h.handedness for h in hands}
    candidates = [
        (
            "Left",
            getattr(holistic_result, "left_hand_landmarks", None) or [],
            getattr(holistic_result, "left_hand_world_landmarks", None) or [],
        ),
        (
            "Right",
            getattr(holistic_result, "right_hand_landmarks", None) or [],
            getattr(holistic_result, "right_hand_world_landmarks", None) or [],
        ),
    ]

    merged = list(hands)
    for raw_label, raw_lms, raw_world in candidates:
        if not raw_lms or len(merged) >= config.num_hands:
            continue
        label = _swap_handedness(raw_label) if config.invert_handedness else raw_label
        if label in present:
            continue

        landmarks = _landmarks_to_points(raw_lms)
        world_landmarks = _landmarks_to_points(raw_world)
        wrist = landmarks[0]
        if signing_space is not None:
            zone, height_level, rel_x, rel_y = classify_torso_relative_position(
                wrist.x, wrist.y, signing_space
            )
        else:
            zone = classify_spatial_zone(
                wrist.x,
                left_max_x=config.left_zone_max_x,
                right_min_x=config.right_zone_min_x,
            )
            height_level = "chest"
            rel_x = (wrist.x - 0.5) / 0.3
            rel_y = (wrist.y - 0.5) / 0.3

        merged.append(
            HandObservation(
                handedness=label,
                handedness_score=0.85,
                gesture="None",
                gesture_score=0.0,
                spatial_zone=zone,
                wrist_xy=(wrist.x, wrist.y),
                landmarks=landmarks,
                world_landmarks=world_landmarks,
                top_gestures=[],
                height_level=height_level,
                rel_xy=(rel_x, rel_y),
            )
        )
        present.add(label)

    return merged


def _serialize_holistic_payload(
    holistic_result: HolisticLandmarkerResult,
) -> dict[str, Any]:
    """Convert HolisticLandmarkerResult into compact JSON-ready landmarks & counts."""
    pose_lms = getattr(holistic_result, "pose_landmarks", None) or []
    face_lms = getattr(holistic_result, "face_landmarks", None) or []
    left_lms = getattr(holistic_result, "left_hand_landmarks", None) or []
    right_lms = getattr(holistic_result, "right_hand_landmarks", None) or []
    pose_world = getattr(holistic_result, "pose_world_landmarks", None) or []

    pose_payload = [
        {
            "x": round(float(p.x), 4),
            "y": round(float(p.y), 4),
            "z": round(float(getattr(p, "z", 0.0) or 0.0), 4),
            "v": round(
                float(p.visibility) if getattr(p, "visibility", None) is not None else 1.0,
                3,
            ),
        }
        for p in pose_lms
    ]

    face_contours = extract_face_contours(face_lms)
    face_mesh_payload = [
        {
            "x": round(float(p.x), 4),
            "y": round(float(p.y), 4),
            "z": round(float(getattr(p, "z", 0.0) or 0.0), 4),
        }
        for p in face_lms
    ]

    total_detected = len(pose_lms) + len(face_lms) + len(left_lms) + len(right_lms)
    payload: dict[str, Any] = {
        "enabled": True,
        "landmark_counts": {
            "pose": len(pose_lms),
            "face": len(face_lms),
            "left_hand": len(left_lms),
            "right_hand": len(right_lms),
            "total": total_detected,
        },
        "pose_landmarks": pose_payload,
        "face_contours": face_contours,
        "face_landmarks": face_mesh_payload,
    }
    if pose_world:
        payload["pose_world_landmarks"] = [
            {
                "x": float(p.x),
                "y": float(p.y),
                "z": float(getattr(p, "z", 0.0) or 0.0),
            }
            for p in pose_world
        ]
    return payload


def build_prediction(
    result: GestureRecognizerResult,
    timestamp_ms: int,
    config: RecognizerConfig,
    stabilizer: GestureStabilizer,
    *,
    immediate: bool = False,
    holistic_result: HolisticLandmarkerResult | None = None,
    spatial_tracker: SpatialGrammarTracker | None = None,
) -> FramePrediction:
    """Extract hand + holistic observations, compute spatial grammar, and smooth signs."""
    signing_space = None
    if holistic_result is not None:
        baseline_x = (
            getattr(spatial_tracker, "_baseline_torso_x", 0.5)
            if spatial_tracker is not None
            else 0.5
        )
        signing_space = compute_signing_space(
            getattr(holistic_result, "pose_landmarks", None),
            baseline_center_x=baseline_x,
            left_fallback=config.left_zone_max_x,
            right_fallback=config.right_zone_min_x,
        )

    hands = extract_observations(result, config, signing_space=signing_space)
    if holistic_result is not None:
        hands = _merge_holistic_hands(hands, holistic_result, config, signing_space)

    spatial_state: SpatialGrammarState | None = None
    holistic_payload: dict[str, Any] | None = None
    if holistic_result is not None:
        tracker = spatial_tracker or SpatialGrammarTracker()
        hand_entries = [
            (h.handedness, h.wrist_xy[0], h.wrist_xy[1], h.gesture)
            for h in hands
        ]
        spatial_state = tracker.update(
            timestamp_ms=timestamp_ms,
            pose_landmarks=getattr(holistic_result, "pose_landmarks", None),
            face_landmarks=getattr(holistic_result, "face_landmarks", None),
            face_blendshapes=getattr(holistic_result, "face_blendshapes", None),
            hand_entries=hand_entries,
            left_fallback=config.left_zone_max_x,
            right_fallback=config.right_zone_min_x,
            temporal=not immediate,
        )
        holistic_payload = _serialize_holistic_payload(holistic_result)
    elif not config.enable_holistic:
        holistic_payload = {"enabled": False}

    primary_hand: HandObservation | None = None
    best_score = -1.0
    for hand in hands:
        if hand.gesture != "None" and hand.gesture_score > best_score:
            best_score = hand.gesture_score
            primary_hand = hand

    if primary_hand is None and hands:
        primary_hand = hands[0]

    if immediate:
        # IMAGE mode runs statelessly on a single image without multi-frame buffering.
        has_valid_gesture = (
            primary_hand is not None
            and primary_hand.gesture != "None"
            and primary_hand.gesture_score >= config.stability_min_score
        )
        active_label = primary_hand.gesture if has_valid_gesture else None
        locked_score = primary_hand.gesture_score if has_valid_gesture else 0.0
        return FramePrediction(
            timestamp_ms=timestamp_ms,
            hands=hands,
            candidate_label=active_label,
            stability_progress=1.0 if has_valid_gesture else 0.0,
            locked_label=active_label,
            locked_score=locked_score,
            spatial_zone=primary_hand.spatial_zone if primary_hand else "center",
            phrases=translate_gesture_label(active_label),
            holistic=holistic_payload,
            spatial_grammar=spatial_state,
        )

    stability = stabilizer.update(
        label=primary_hand.gesture if primary_hand else None,
        score=primary_hand.gesture_score if primary_hand else 0.0,
        now=timestamp_ms / 1000.0,
    )

    active_label = stability.locked_label or stability.candidate_label
    phrases = translate_gesture_label(active_label)

    return FramePrediction(
        timestamp_ms=timestamp_ms,
        hands=hands,
        candidate_label=stability.candidate_label,
        stability_progress=stability.progress,
        locked_label=stability.locked_label,
        locked_score=stability.locked_score,
        spatial_zone=primary_hand.spatial_zone if primary_hand else "center",
        phrases=phrases,
        holistic=holistic_payload,
        spatial_grammar=spatial_state,
    )


class SignGestureRecognizer:
    """MediaPipe GestureRecognizer + HolisticLandmarker wrapper (IMAGE/VIDEO/LIVE_STREAM)."""

    def __init__(
        self,
        config: RecognizerConfig | None = None,
        result_callback: Callable[[FramePrediction], None] | None = None,
    ) -> None:
        self.config = config or RecognizerConfig()
        mode_key = self.config.running_mode.upper()
        if mode_key not in _RUNNING_MODE_MAP:
            raise ValueError(
                f"Unsupported running_mode {self.config.running_mode!r}; expected one of {RUNNING_MODES}."
            )
        if mode_key == "LIVE_STREAM" and result_callback is None:
            raise ValueError("result_callback is required when running_mode is 'LIVE_STREAM'.")

        self._mode_key = mode_key
        self._result_callback = result_callback
        model_path = ensure_model_asset(self.config.model_path, self.config.model_url)

        self._stabilizer = GestureStabilizer(
            window_size=self.config.stability_window,
            min_votes=self.config.stability_min_votes,
            min_score=self.config.stability_min_score,
            cooldown_seconds=self.config.cooldown_seconds,
        )
        self._spatial_tracker = SpatialGrammarTracker()
        self._last_timestamp_ms = 0
        self._closed = False
        self._holistic_lock = threading.Lock()
        self._latest_holistic: tuple[int, HolisticLandmarkerResult] | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._holistic_landmarker: HolisticLandmarker | None = None

        canned_opts = _build_classifier_options(
            display_names_locale=self.config.display_names_locale,
            max_results=self.config.canned_max_results,
            score_threshold=self.config.canned_score_threshold,
            category_allowlist=self.config.canned_category_allowlist,
            category_denylist=self.config.canned_category_denylist,
        )
        custom_opts = _build_classifier_options(
            display_names_locale=self.config.display_names_locale,
            max_results=self.config.custom_max_results,
            score_threshold=self.config.custom_score_threshold,
            category_allowlist=self.config.custom_category_allowlist,
            category_denylist=self.config.custom_category_denylist,
        )

        mp_callback = None
        holistic_callback = None
        if mode_key == "LIVE_STREAM":
            def _on_holistic_live(
                h_result: HolisticLandmarkerResult,
                _output_image: mp.Image,
                timestamp_ms: int,
            ) -> None:
                with self._holistic_lock:
                    self._latest_holistic = (timestamp_ms, h_result)

            def _on_live_result(
                result: GestureRecognizerResult,
                _output_image: mp.Image,
                timestamp_ms: int,
            ) -> None:
                holistic_res = None
                if self.config.enable_holistic:
                    with self._holistic_lock:
                        if self._latest_holistic is not None:
                            h_ts, h_val = self._latest_holistic
                            if abs(timestamp_ms - h_ts) <= 350:
                                holistic_res = h_val
                pred = build_prediction(
                    result,
                    timestamp_ms,
                    self.config,
                    self._stabilizer,
                    holistic_result=holistic_res,
                    spatial_tracker=self._spatial_tracker,
                )
                if self._result_callback is not None:
                    self._result_callback(pred)

            mp_callback = _on_live_result
            holistic_callback = _on_holistic_live

        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=_RUNNING_MODE_MAP[mode_key],
            num_hands=self.config.num_hands,
            min_hand_detection_confidence=self.config.min_hand_detection_confidence,
            min_hand_presence_confidence=self.config.min_hand_presence_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            canned_gestures_classifier_options=canned_opts,
            custom_gestures_classifier_options=custom_opts,
            result_callback=mp_callback,
        )
        self._recognizer = GestureRecognizer.create_from_options(options)

        if self.config.enable_holistic:
            try:
                holistic_path = ensure_model_asset(
                    self.config.holistic_model_path,
                    self.config.holistic_model_url,
                )
                holistic_options = HolisticLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(holistic_path)),
                    running_mode=_RUNNING_MODE_MAP[mode_key],
                    min_face_detection_confidence=self.config.min_face_detection_confidence,
                    min_face_suppression_threshold=self.config.min_face_suppression_threshold,
                    min_face_presence_confidence=self.config.min_face_presence_confidence,
                    min_pose_detection_confidence=self.config.min_pose_detection_confidence,
                    min_pose_suppression_threshold=self.config.min_pose_suppression_threshold,
                    min_pose_presence_confidence=self.config.min_pose_presence_confidence,
                    min_hand_landmarks_confidence=self.config.min_hand_landmarks_confidence,
                    output_face_blendshapes=self.config.output_face_blendshapes,
                    output_pose_segmentation_masks=self.config.output_pose_segmentation_masks,
                    result_callback=holistic_callback,
                )
                self._holistic_landmarker = HolisticLandmarker.create_from_options(
                    holistic_options
                )
                if mode_key != "LIVE_STREAM":
                    self._executor = ThreadPoolExecutor(
                        max_workers=2,
                        thread_name_prefix="signbridge-mp",
                    )
            except Exception:
                self.close()
                raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        if getattr(self, "_recognizer", None) is not None:
            self._recognizer.close()
        if self._holistic_landmarker is not None:
            self._holistic_landmarker.close()
            self._holistic_landmarker = None

    def __enter__(self) -> "SignGestureRecognizer":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _next_timestamp_ms(self, timestamp_ms: int | None = None) -> int:
        raw = int(time.monotonic() * 1000) if timestamp_ms is None else int(timestamp_ms)
        if raw <= self._last_timestamp_ms:
            raw = self._last_timestamp_ms + 1
        self._last_timestamp_ms = raw
        return raw

    def process_rgb(
        self,
        rgb_frame: np.ndarray,
        timestamp_ms: int | None = None,
    ) -> FramePrediction:
        """Run synchronous gesture + holistic landmark recognition (IMAGE or VIDEO mode)."""
        if self._mode_key == "LIVE_STREAM":
            raise ValueError(
                "Use process_async_rgb() when running_mode is 'LIVE_STREAM'."
            )

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        if self._mode_key == "IMAGE":
            ts = int(time.monotonic() * 1000) if timestamp_ms is None else int(timestamp_ms)
            if self._holistic_landmarker is not None and self._executor is not None:
                fut_g = self._executor.submit(self._recognizer.recognize, mp_image)
                fut_h = self._executor.submit(self._holistic_landmarker.detect, mp_image)
                result = fut_g.result()
                holistic_result = fut_h.result()
            else:
                result = self._recognizer.recognize(mp_image)
                holistic_result = None
            return build_prediction(
                result,
                ts,
                self.config,
                self._stabilizer,
                immediate=True,
                holistic_result=holistic_result,
                spatial_tracker=self._spatial_tracker,
            )

        ts = self._next_timestamp_ms(timestamp_ms)
        if self._holistic_landmarker is not None and self._executor is not None:
            fut_g = self._executor.submit(self._recognizer.recognize_for_video, mp_image, ts)
            fut_h = self._executor.submit(
                self._holistic_landmarker.detect_for_video, mp_image, ts
            )
            result = fut_g.result()
            holistic_result = fut_h.result()
        else:
            result = self._recognizer.recognize_for_video(mp_image, ts)
            holistic_result = None

        return build_prediction(
            result,
            ts,
            self.config,
            self._stabilizer,
            holistic_result=holistic_result,
            spatial_tracker=self._spatial_tracker,
        )

    def process_async_rgb(
        self,
        rgb_frame: np.ndarray,
        timestamp_ms: int | None = None,
    ) -> int:
        """Submit an RGB uint8 frame asynchronously in LIVE_STREAM mode."""
        if self._mode_key != "LIVE_STREAM":
            raise ValueError(
                "process_async_rgb() requires running_mode='LIVE_STREAM'."
            )
        ts = self._next_timestamp_ms(timestamp_ms)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self._recognizer.recognize_async(mp_image, ts)
        if self._holistic_landmarker is not None:
            self._holistic_landmarker.detect_async(mp_image, ts)
        return ts

    def process_bgr(
        self,
        bgr_frame: np.ndarray,
        timestamp_ms: int | None = None,
    ) -> FramePrediction:
        """Convert OpenCV BGR frame to RGB and run synchronous recognition."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        return self.process_rgb(rgb, timestamp_ms=timestamp_ms)

    def process_encoded_bytes(
        self,
        encoded_image: bytes,
        timestamp_ms: int | None = None,
    ) -> FramePrediction:
        """Decode JPEG/PNG bytes from a client and run synchronous recognition."""
        arr = np.frombuffer(encoded_image, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image bytes.")
        return self.process_bgr(bgr, timestamp_ms=timestamp_ms)
