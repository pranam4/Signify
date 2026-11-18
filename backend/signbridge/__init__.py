"""SignBridge Python MediaPipe recognition package."""

from .config import (
    CUSTOM_SIGN_TRANSLATIONS,
    DEFAULT_STT_MODEL_ID,
    DIRECTIONAL_PHRASES,
    GESTURE_TRANSLATIONS,
    HAND_CONNECTIONS,
    POSE_CONNECTIONS,
    RUNNING_MODES,
    RecognizerConfig,
    SpeechToTextConfig,
    config_from_env,
    stt_config_from_env,
    translate_gesture_label,
)
from .custom_classifier import (
    CustomPrediction,
    CustomSignClassifier,
    LandmarkSignMLP,
    extract_sign_features,
    train_landmark_classifier,
)
from .model_manager import ensure_model_asset
from .recognizer import (
    FramePrediction,
    GestureCandidate,
    HandObservation,
    SignGestureRecognizer,
)
from .sign_dataset import SignDatasetStore
from .sign_planner import SignPlan, SignStep, text_to_sign_plan
from .spatial_grammar import (
    DirectionalEvent,
    NonManualMarkers,
    SigningSpaceFrame,
    SpatialGrammarState,
    SpatialGrammarTracker,
    apply_nmm_to_phrase,
    classify_torso_relative_position,
    compute_signing_space,
    extract_face_contours,
    extract_non_manual_markers,
)
from .speech_to_text import (
    AudioDecodeError,
    TranscriptionResult,
    TranscriptionSegment,
    WhisperSpeechToText,
)
from .stabilizer import GestureStabilizer, StabilityUpdate

__all__ = [
    "AudioDecodeError",
    "CUSTOM_SIGN_TRANSLATIONS",
    "CustomPrediction",
    "CustomSignClassifier",
    "DEFAULT_STT_MODEL_ID",
    "DIRECTIONAL_PHRASES",
    "DirectionalEvent",
    "FramePrediction",
    "GESTURE_TRANSLATIONS",
    "GestureCandidate",
    "GestureStabilizer",
    "HAND_CONNECTIONS",
    "HandObservation",
    "LandmarkSignMLP",
    "NonManualMarkers",
    "POSE_CONNECTIONS",
    "RUNNING_MODES",
    "RecognizerConfig",
    "SignDatasetStore",
    "SignGestureRecognizer",
    "SignPlan",
    "SignStep",
    "SigningSpaceFrame",
    "SpatialGrammarState",
    "SpatialGrammarTracker",
    "SpeechToTextConfig",
    "StabilityUpdate",
    "TranscriptionResult",
    "TranscriptionSegment",
    "WhisperSpeechToText",
    "apply_nmm_to_phrase",
    "classify_torso_relative_position",
    "compute_signing_space",
    "config_from_env",
    "ensure_model_asset",
    "extract_face_contours",
    "extract_non_manual_markers",
    "extract_sign_features",
    "stt_config_from_env",
    "text_to_sign_plan",
    "train_landmark_classifier",
    "translate_gesture_label",
]
