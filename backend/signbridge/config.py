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
DEFAULT_DATASET_DIR = BACKEND_DIR / "data" / "sign_dataset"
DEFAULT_DB_PATH = BACKEND_DIR / "data" / "signbridge.db"
DEFAULT_ISLRTC_DICTIONARY_PATH = BACKEND_DIR / "data" / "islrtc_dictionary.json"
DEFAULT_POSTER_REFERENCES_PATH = BACKEND_DIR / "data" / "poster_references.json"
DEFAULT_CUSTOM_CLASSIFIER_PATH = BACKEND_DIR / "models" / "custom_sign_classifier.pt"
DEFAULT_DYNAMIC_CLASSIFIER_PATH = BACKEND_DIR / "models" / "dynamic_sign_classifier.pt"
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
    enable_custom_classifier: bool = True
    custom_classifier_path: Path = field(
        default_factory=lambda: DEFAULT_CUSTOM_CLASSIFIER_PATH
    )
    custom_classifier_threshold: float = 0.62
    custom_classifier_min_similarity: float = 0.52
    enable_dynamic_recognizer: bool = True
    dynamic_classifier_path: Path = field(
        default_factory=lambda: DEFAULT_DYNAMIC_CLASSIFIER_PATH
    )
    dynamic_classifier_threshold: float = 0.60
    dynamic_classifier_min_similarity: float = 0.48


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
    override_custom = os.getenv("SIGNBRIDGE_CUSTOM_CLASSIFIER_PATH", "").strip()
    custom_path = (
        Path(override_custom) if override_custom else DEFAULT_CUSTOM_CLASSIFIER_PATH
    )
    override_dynamic = os.getenv("SIGNBRIDGE_DYNAMIC_CLASSIFIER_PATH", "").strip()
    dynamic_path = (
        Path(override_dynamic) if override_dynamic else DEFAULT_DYNAMIC_CLASSIFIER_PATH
    )
    raw_custom = os.getenv("SIGNBRIDGE_ENABLE_CUSTOM_CLASSIFIER", "1").strip().lower()
    enable_custom = raw_custom not in {"0", "false", "no", "off"}
    raw_dynamic = os.getenv("SIGNBRIDGE_ENABLE_DYNAMIC_RECOGNIZER", "1").strip().lower()
    enable_dynamic = raw_dynamic not in {"0", "false", "no", "off"}

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
        enable_custom_classifier=enable_custom,
        custom_classifier_path=custom_path,
        enable_dynamic_recognizer=enable_dynamic,
        dynamic_classifier_path=dynamic_path,
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
    """Return phrase translations for stock gestures or custom ISL sign labels."""
    if not label or label == "None":
        return {}
    if label in GESTURE_TRANSLATIONS:
        return dict(GESTURE_TRANSLATIONS[label])
    if label in CUSTOM_SIGN_TRANSLATIONS:
        return dict(CUSTOM_SIGN_TRANSLATIONS[label])
    if len(label) == 1 and label.isalnum():
        upper = label.upper()
        return {
            "kind": "fingerspell",
            "en": upper,
            "hi": upper,
            "kn": upper,
        }

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


CUSTOM_SIGN_TRANSLATIONS: dict[str, dict[str, str]] = {
    "Ambulance": {"kind": "custom_sign", "en": "Ambulance", "hi": "एम्बुलेंस", "kn": "ಆಂಬ್ಯುಲೆನ್ಸ್"},
    "Answer": {"kind": "custom_sign", "en": "Answer", "hi": "उत्तर", "kn": "ಉತ್ತರ"},
    "Bath": {"kind": "custom_sign", "en": "Bath", "hi": "नहाना", "kn": "ಸ್ನಾನ"},
    "Bathroom": {"kind": "custom_sign", "en": "Bathroom", "hi": "बाथरूम", "kn": "ಸ್ನಾನಗೃಹ"},
    "Bed": {"kind": "custom_sign", "en": "Bed", "hi": "बिस्तर", "kn": "ಹಾಸಿಗೆ"},
    "Book": {"kind": "custom_sign", "en": "Book", "hi": "किताब", "kn": "ಪುಸ್ತಕ"},
    "Careful": {"kind": "custom_sign", "en": "Careful", "hi": "सावधान", "kn": "ಜಾಗ್ರತೆ"},
    "Change": {"kind": "custom_sign", "en": "Change", "hi": "बदलें", "kn": "ಬದಲಾಯಿಸಿ"},
    "Climb": {"kind": "custom_sign", "en": "Climb", "hi": "चढ़ना", "kn": "ಹತ್ತು"},
    "Come": {"kind": "custom_sign", "en": "Come", "hi": "आओ", "kn": "ಬನ್ನಿ"},
    "Cookie": {"kind": "custom_sign", "en": "Cookie", "hi": "कुकी / बिस्कुट", "kn": "ಬಿಸ್ಕೆಟ್"},
    "Correct": {"kind": "custom_sign", "en": "Correct", "hi": "सही", "kn": "ಸರಿ"},
    "Cracker": {"kind": "custom_sign", "en": "Cracker", "hi": "क्रैकर", "kn": "ಕ್ರ್ಯಾಕರ್"},
    "Cry": {"kind": "custom_sign", "en": "Cry", "hi": "रोना", "kn": "ಅಳು"},
    "Dad": {"kind": "custom_sign", "en": "Dad", "hi": "पापा", "kn": "ಅಪ್ಪ"},
    "Danger": {"kind": "custom_sign", "en": "Danger", "hi": "खतरा", "kn": "ಅಪಾಯ"},
    "Done": {"kind": "custom_sign", "en": "Done", "hi": "हो गया", "kn": "ಮುಗಿಯಿತು"},
    "Down": {"kind": "custom_sign", "en": "Down", "hi": "नीचे", "kn": "ಕೆಳಗೆ"},
    "Drink": {"kind": "custom_sign", "en": "Drink", "hi": "पीना", "kn": "ಕುಡಿ"},
    "Eat": {"kind": "custom_sign", "en": "Eat", "hi": "खाना", "kn": "ತಿನ್ನು"},
    "Fall": {"kind": "custom_sign", "en": "Fall", "hi": "गिरना", "kn": "ಬೀಳು"},
    "Finish": {"kind": "custom_sign", "en": "Finish", "hi": "समाप्त", "kn": "ಮುಗಿಸು"},
    "Fire": {"kind": "custom_sign", "en": "Fire", "hi": "आग", "kn": "ಬೆಂಕಿ"},
    "Give": {"kind": "custom_sign", "en": "Give", "hi": "देना", "kn": "ಕೊಡು"},
    "Go": {"kind": "custom_sign", "en": "Go", "hi": "जाओ", "kn": "ಹೋಗು"},
    "Good": {"kind": "custom_sign", "en": "Good", "hi": "अच्छा", "kn": "ಒಳ್ಳೆಯದು"},
    "Good Morning": {"kind": "custom_sign", "en": "Good morning", "hi": "सुप्रभात", "kn": "ಶುಭೋದಯ"},
    "Goodbye": {"kind": "custom_sign", "en": "Goodbye", "hi": "अलविदा", "kn": "ವಿದಾಯ"},
    "Hello": {"kind": "custom_sign", "en": "Hello", "hi": "नमस्ते", "kn": "ನಮಸ್ಕಾರ"},
    "Help": {"kind": "custom_sign", "en": "Help", "hi": "मदद", "kn": "ಸಹಾಯ"},
    "Home": {"kind": "custom_sign", "en": "Home", "hi": "घर", "kn": "ಮನೆ"},
    "Hot": {"kind": "custom_sign", "en": "Hot", "hi": "गरम", "kn": "ಬಿಸಿ"},
    "How": {"kind": "custom_sign", "en": "How", "hi": "कैसे", "kn": "ಹೇಗೆ"},
    "Hug": {"kind": "custom_sign", "en": "Hug", "hi": "गले लगाना", "kn": "ಅಪ್ಪಿಕೊಳ್ಳಿ"},
    "I Love You": {"kind": "custom_sign", "en": "I love you", "hi": "मैं तुमसे प्यार करता हूँ", "kn": "ನಾನು ನಿನ್ನನ್ನು ಪ್ರೀತಿಸುತ್ತೇನೆ"},
    "I'm Sorry": {"kind": "custom_sign", "en": "I'm sorry", "hi": "मुझे माफ़ कीजिए", "kn": "ದಯವಿಟ್ಟು ಕ್ಷಮಿಸಿ"},
    "Ignore": {"kind": "custom_sign", "en": "Ignore", "hi": "अनदेखा करना", "kn": "ನಿರ್ಲಕ್ಷಿಸು"},
    "Jump": {"kind": "custom_sign", "en": "Jump", "hi": "कूदना", "kn": "ಜಿಗಿ"},
    "Kick": {"kind": "custom_sign", "en": "Kick", "hi": "लात मारना", "kn": "ಒದೆ"},
    "Look": {"kind": "custom_sign", "en": "Look", "hi": "देखो", "kn": "ನೋಡು"},
    "Me": {"kind": "custom_sign", "en": "Me", "hi": "मैं / मुझे", "kn": "ನಾನು / ನನಗೆ"},
    "Me Too": {"kind": "custom_sign", "en": "Me too", "hi": "मैं भी", "kn": "ನಾನೂ ಸಹ"},
    "Meet": {"kind": "custom_sign", "en": "Meet", "hi": "मिलना", "kn": "ಭೇಟಿಯಾಗಿ"},
    "Milk": {"kind": "custom_sign", "en": "Milk", "hi": "दूध", "kn": "ಹಾಲು"},
    "Mom": {"kind": "custom_sign", "en": "Mom", "hi": "माँ", "kn": "ಅಮ್ಮ"},
    "More": {"kind": "custom_sign", "en": "More", "hi": "और अधिक", "kn": "ಇನ್ನೂ ಹೆಚ್ಚು"},
    "My": {"kind": "custom_sign", "en": "My", "hi": "मेरा", "kn": "ನನ್ನ"},
    "My Name Is": {"kind": "custom_sign", "en": "My name is", "hi": "मेरा नाम है", "kn": "ನನ್ನ ಹೆಸರು"},
    "Name": {"kind": "custom_sign", "en": "Name", "hi": "नाम", "kn": "ಹೆಸರು"},
    "Nice": {"kind": "custom_sign", "en": "Nice", "hi": "अच्छा", "kn": "ಚೆನ್ನಾಗಿದೆ"},
    "No": {"kind": "custom_sign", "en": "No", "hi": "नहीं", "kn": "ಇಲ್ಲ"},
    "Now": {"kind": "custom_sign", "en": "Now", "hi": "अभी", "kn": "ಈಗ"},
    "Outside": {"kind": "custom_sign", "en": "Outside", "hi": "बाहर", "kn": "ಹೊರಗೆ"},
    "Over": {"kind": "custom_sign", "en": "Over", "hi": "ऊपर / पार", "kn": "ಮೇಲೆ"},
    "Pain": {"kind": "custom_sign", "en": "Pain", "hi": "दर्द", "kn": "ನೋವು"},
    "Pencil": {"kind": "custom_sign", "en": "Pencil", "hi": "पेंसिल", "kn": "ಪೆನ್ಸಿಲ್"},
    "Please": {"kind": "custom_sign", "en": "Please", "hi": "कृपया", "kn": "ದಯವಿಟ್ಟು"},
    "Police": {"kind": "custom_sign", "en": "Police", "hi": "पुलिस", "kn": "ಪೊಲೀಸ್"},
    "Push": {"kind": "custom_sign", "en": "Push", "hi": "धक्का देना", "kn": "ತಳ್ಳು"},
    "Question": {"kind": "custom_sign", "en": "Question", "hi": "प्रश्न", "kn": "ಪ್ರಶ್ನೆ"},
    "Run": {"kind": "custom_sign", "en": "Run", "hi": "दौड़ना", "kn": "ಓಡು"},
    "Show Me": {"kind": "custom_sign", "en": "Show me", "hi": "मुझे दिखाओ", "kn": "ನನಗೆ ತೋರಿಸು"},
    "Sick": {"kind": "custom_sign", "en": "Sick", "hi": "बीमार", "kn": "ಅನಾರೋಗ್ಯ"},
    "Sit": {"kind": "custom_sign", "en": "Sit", "hi": "बैठो", "kn": "ಕುಳಿತುಕೊ"},
    "Sit Down": {"kind": "custom_sign", "en": "Sit down", "hi": "बैठ जाइए", "kn": "ಕೆಳಗೆ ಕುಳಿತುಕೊಳ್ಳಿ"},
    "Sleep": {"kind": "custom_sign", "en": "Sleep", "hi": "सोना", "kn": "ನಿದ್ರೆ"},
    "Sorry": {"kind": "custom_sign", "en": "Sorry", "hi": "माफ़ कीजिए", "kn": "ಕ್ಷಮಿಸಿ"},
    "Stand": {"kind": "custom_sign", "en": "Stand", "hi": "खड़े हो जाओ", "kn": "ನಿಲ್ಲಿ"},
    "Stay Calm": {"kind": "custom_sign", "en": "Stay calm", "hi": "शांत रहें", "kn": "ಶಾಂತವಾಗಿರಿ"},
    "Stop": {"kind": "custom_sign", "en": "Stop", "hi": "रुको", "kn": "ನಿಲ್ಲು"},
    "Thank You": {"kind": "custom_sign", "en": "Thank you", "hi": "धन्यवाद", "kn": "ಧನ್ಯವಾದಗಳು"},
    "Tissue": {"kind": "custom_sign", "en": "Tissue", "hi": "टिश्यू", "kn": "ಟಿಶ್ಯೂ"},
    "Toilet": {"kind": "custom_sign", "en": "Toilet", "hi": "शौचालय", "kn": "ಶೌಚಾಲಯ"},
    "Touch": {"kind": "custom_sign", "en": "Touch", "hi": "छूना", "kn": "ಸ್ಪರ್ಶಿಸಿ"},
    "Under": {"kind": "custom_sign", "en": "Under", "hi": "नीचे", "kn": "ಕೆಳಗೆ"},
    "Up": {"kind": "custom_sign", "en": "Up", "hi": "ऊपर", "kn": "ಮೇಲೆ"},
    "Wait": {"kind": "custom_sign", "en": "Wait", "hi": "इंतज़ार करें", "kn": "ಕಾಯಿರಿ"},
    "Walk": {"kind": "custom_sign", "en": "Walk", "hi": "चलना", "kn": "ನಡೆ"},
    "Want": {"kind": "custom_sign", "en": "Want", "hi": "चाहिए", "kn": "ಬೇಕು"},
    "Wash": {"kind": "custom_sign", "en": "Wash", "hi": "धोना", "kn": "ತೊಳೆ"},
    "Water": {"kind": "custom_sign", "en": "Water", "hi": "पानी", "kn": "ನೀರು"},
    "Welcome": {"kind": "custom_sign", "en": "Welcome", "hi": "स्वागत है", "kn": "ಸ್ವಾಗತ"},
    "Wet": {"kind": "custom_sign", "en": "Wet", "hi": "गीला", "kn": "ಒದ್ದೆ"},
    "What": {"kind": "custom_sign", "en": "What", "hi": "क्या", "kn": "ಏನು"},
    "When": {"kind": "custom_sign", "en": "When", "hi": "कब", "kn": "ಯಾವಾಗ"},
    "Where": {"kind": "custom_sign", "en": "Where", "hi": "कहाँ", "kn": "ಎಲ್ಲಿ"},
    "Who": {"kind": "custom_sign", "en": "Who", "hi": "कौन", "kn": "ಯಾರು"},
    "Work": {"kind": "custom_sign", "en": "Work", "hi": "काम", "kn": "ಕೆಲಸ"},
    "Yes": {"kind": "custom_sign", "en": "Yes", "hi": "हाँ", "kn": "ಹೌದು"},
    "You": {"kind": "custom_sign", "en": "You", "hi": "आप", "kn": "ನೀವು"},
    "You're Welcome": {"kind": "custom_sign", "en": "You're welcome", "hi": "आपका स्वागत है", "kn": "ಸ್ವಾಗತ"},
    "Your": {"kind": "custom_sign", "en": "Your", "hi": "तुम्हारा / आपका", "kn": "ನಿಮ್ಮ"},
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
