"""Configuration, defaults, and sign/gesture phrase mappings for SignBridge."""

from dataclasses import dataclass, field
import os
from pathlib import Path
import re

DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
)
DEFAULT_HOLISTIC_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task"
)
BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BACKEND_DIR / "models" / "gesture_recognizer.task"
DEFAULT_HOLISTIC_MODEL_PATH = BACKEND_DIR / "models" / "holistic_landmarker.task"
RUNNING_MODES: tuple[str, ...] = ("IMAGE", "VIDEO", "LIVE_STREAM")

DEFAULT_STT_MODEL_ID = "openai/whisper-tiny"
DEFAULT_STT_SAMPLE_RATE = 16000
STT_TASKS: tuple[str, ...] = ("transcribe", "translate")
STT_LANGUAGE_ALIASES: dict[str, str] = {
    "english": "en",
    "hindi": "hi",
    "kannada": "kn",
}

# Standard MediaPipe 21-landmark hand skeleton connections (wrist -> fingers).
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17),                               # Palm base
)

# Upper-body & torso MediaPipe 33-landmark pose connections relevant to signing space.
POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (11, 12),                              # Shoulders
    (11, 13), (13, 15),                    # Left arm
    (12, 14), (14, 16),                    # Right arm
    (11, 23), (12, 24), (23, 24),          # Torso box
    (15, 17), (15, 19), (15, 21), (17, 19),# Left wrist anchor
    (16, 18), (16, 20), (16, 22), (18, 20),# Right wrist anchor
)


@dataclass(slots=True)
class RecognizerConfig:
    """Runtime settings for MediaPipe GestureRecognizer + HolisticLandmarker."""

    model_path: Path = field(default_factory=lambda: DEFAULT_MODEL_PATH)
    model_url: str = DEFAULT_MODEL_URL
    enable_holistic: bool = True
    holistic_model_path: Path = field(default_factory=lambda: DEFAULT_HOLISTIC_MODEL_PATH)
    holistic_model_url: str = DEFAULT_HOLISTIC_MODEL_URL
    running_mode: str = "VIDEO"
    num_hands: int = 2
    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    min_face_detection_confidence: float = 0.5
    min_face_suppression_threshold: float = 0.3
    min_face_presence_confidence: float = 0.5
    min_pose_detection_confidence: float = 0.5
    min_pose_suppression_threshold: float = 0.3
    min_pose_presence_confidence: float = 0.5
    min_hand_landmarks_confidence: float = 0.5
    output_face_blendshapes: bool = True
    output_pose_segmentation_masks: bool = False
    display_names_locale: str | None = None
    canned_score_threshold: float | None = None
    canned_max_results: int | None = 3
    canned_category_allowlist: list[str] | None = None
    canned_category_denylist: list[str] | None = None
    custom_score_threshold: float | None = None
    custom_max_results: int | None = None
    custom_category_allowlist: list[str] | None = None
    custom_category_denylist: list[str] | None = None
    stability_window: int = 6
    stability_min_votes: int = 4
    stability_min_score: float = 0.55
    cooldown_seconds: float = 1.5
    left_zone_max_x: float = 0.35
    right_zone_min_x: float = 0.65
    invert_handedness: bool = False


def config_from_env(
    *,
    running_mode: str = "VIDEO",
    enable_holistic: bool | None = None,
) -> RecognizerConfig:
    """Build a RecognizerConfig, honoring optional SIGNBRIDGE_* env overrides."""
    override_model = os.getenv("SIGNBRIDGE_MODEL_PATH", "").strip()
    model_path = Path(override_model) if override_model else DEFAULT_MODEL_PATH
    override_holistic = os.getenv("SIGNBRIDGE_HOLISTIC_MODEL_PATH", "").strip()
    holistic_path = (
        Path(override_holistic) if override_holistic else DEFAULT_HOLISTIC_MODEL_PATH
    )
    if enable_holistic is None:
        raw_holistic = os.getenv("SIGNBRIDGE_ENABLE_HOLISTIC", "1").strip().lower()
        resolved_holistic = raw_holistic not in {"0", "false", "no", "off"}
    else:
        resolved_holistic = bool(enable_holistic)

    return RecognizerConfig(
        model_path=model_path,
        enable_holistic=resolved_holistic,
        holistic_model_path=holistic_path,
        running_mode=running_mode,
    )


@dataclass(slots=True)
class SpeechToTextConfig:
    """Runtime settings for Hugging Face Whisper (`openai/whisper-tiny`) speech-to-text."""

    model_id: str = DEFAULT_STT_MODEL_ID
    sample_rate: int = DEFAULT_STT_SAMPLE_RATE
    device: str | None = None
    dtype: str | None = None
    cache_dir: Path | None = None
    default_language: str | None = None
    default_task: str = "transcribe"
    return_timestamps: bool = True
    min_rms_energy: float = 1e-4
    max_audio_seconds: float = 300.0
    max_upload_bytes: int = 25 * 1024 * 1024
    local_files_only: bool = False


def stt_config_from_env() -> SpeechToTextConfig:
    """Build a SpeechToTextConfig from optional SIGNBRIDGE_STT_* environment variables."""
    model_id = os.getenv("SIGNBRIDGE_STT_MODEL", "").strip() or DEFAULT_STT_MODEL_ID
    device = os.getenv("SIGNBRIDGE_STT_DEVICE", "").strip() or None
    cache_raw = os.getenv("SIGNBRIDGE_STT_CACHE_DIR", "").strip()
    cache_dir = Path(cache_raw) if cache_raw else None
    local_only = os.getenv("SIGNBRIDGE_STT_LOCAL_ONLY", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    return SpeechToTextConfig(
        model_id=model_id,
        device=device,
        cache_dir=cache_dir,
        local_files_only=local_only,
    )


def translate_gesture_label(label: str | None) -> dict[str, str]:
    """Return phrase translations for stock gestures or format custom Model Maker labels."""
    if not label or label == "None":
        return {}
    if label in GESTURE_TRANSLATIONS:
        return dict(GESTURE_TRANSLATIONS[label])

    # Format custom model labels like "thank_you" or "ThankYou" into readable text.
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", label.replace("_", " ").replace("-", " "))
    humanized = " ".join(spaced.split()).strip().capitalize()
    if not humanized:
        return {}
    return {
        "kind": "custom_sign",
        "en": humanized,
        "hi": humanized,
        "kn": humanized,
    }


# Baseline labels returned by the stock MediaPipe GestureRecognizer model,
# mapped cleanly with English/Hindi/Kannada captions for early UI testing.
# Replace or extend with verified ISL/ASL sign classifiers in the next step.
GESTURE_TRANSLATIONS: dict[str, dict[str, str]] = {
    "ILoveYou": {
        "kind": "sign",
        "en": "I love you",
        "hi": "मैं तुमसे प्यार करता हूँ",
        "kn": "ನಾನು ನಿನ್ನನ್ನು ಪ್ರೀತಿಸುತ್ತೇನೆ",
    },
    "Pointing_Up": {
        "kind": "spatial_cue",
        "en": "Spatial point",
        "hi": "दिशा संकेत",
        "kn": "ದಿಕ್ಕು ಸೂಚನೆ",
    },
    "Open_Palm": {
        "kind": "control",
        "en": "Hello / Open palm",
        "hi": "नमस्ते / रुको",
        "kn": "ನಮಸ್ಕಾರ / ನಿಲ್ಲು",
    },
    "Closed_Fist": {
        "kind": "control",
        "en": "Hold / Fist",
        "hi": "रुको / मुट्ठी",
        "kn": "ಹಿಡಿ / ಮುಷ್ಟಿ",
    },
    "Thumb_Up": {
        "kind": "control",
        "en": "Yes / Confirm",
        "hi": "हाँ / ठीक है",
        "kn": "ಹೌದು / ಸರಿ",
    },
    "Thumb_Down": {
        "kind": "control",
        "en": "No / Cancel",
        "hi": "नहीं / रद्द करें",
        "kn": "ಇಲ್ಲ / ರದ್ದು",
    },
    "Victory": {
        "kind": "control",
        "en": "Two / Peace",
        "hi": "दो / शांति",
        "kn": "ಎರಡು / ಶಾಂತಿ",
    },
}

# Directional spatial-agreement phrases inferred from cross-zone hand trajectories.
DIRECTIONAL_PHRASES: dict[str, dict[str, str]] = {
    "GIVE_A_TO_B": {
        "kind": "spatial_verb",
        "en": "Person A gives to Person B",
        "hi": "व्यक्ति A व्यक्ति B को देता है",
        "kn": "ವ್ಯಕ್ತಿ A ವ್ಯಕ್ತಿ B ಗೆ ಕೊಡುತ್ತಾರೆ",
    },
    "GIVE_B_TO_A": {
        "kind": "spatial_verb",
        "en": "Person B gives to Person A",
        "hi": "व्यक्ति B व्यक्ति A को देता है",
        "kn": "ವ್ಯಕ್ತಿ B ವ್ಯಕ್ತಿ A ಗೆ ಕೊಡುತ್ತಾರೆ",
    },
    "SELF_TO_A": {
        "kind": "spatial_verb",
        "en": "I give / direct to Person A",
        "hi": "मैं व्यक्ति A को देता हूँ",
        "kn": "ನಾನು ವ್ಯಕ್ತಿ A ಗೆ ಕೊಡುತ್ತೇನೆ",
    },
    "SELF_TO_B": {
        "kind": "spatial_verb",
        "en": "I give / direct to Person B",
        "hi": "मैं व्यक्ति B को देता हूँ",
        "kn": "ನಾನು ವ್ಯಕ್ತಿ B ಗೆ ಕೊಡುತ್ತೇನೆ",
    },
    "A_TO_SELF": {
        "kind": "spatial_verb",
        "en": "Person A gives to me",
        "hi": "व्यक्ति A मुझे देता है",
        "kn": "ವ್ಯಕ್ತಿ A ನನಗೆ ಕೊಡುತ್ತಾರೆ",
    },
    "B_TO_SELF": {
        "kind": "spatial_verb",
        "en": "Person B gives to me",
        "hi": "व्यक्ति B मुझे देता है",
        "kn": "ವ್ಯಕ್ತಿ B ನನಗೆ ಕೊಡುತ್ತಾರೆ",
    },
}
