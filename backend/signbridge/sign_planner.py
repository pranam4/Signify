"""Spoken text -> Sign Language gloss planner with NMM & fingerspelling support."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

_COPULA_AND_FILLERS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "am",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "to",
        "of",
        "for",
        "in",
        "on",
        "at",
        "with",
        "and",
        "do",
        "does",
        "did",
        "will",
        "would",
        "shall",
        "should",
        "can",
        "could",
        "has",
        "have",
        "had",
        "it",
        "that",
        "this",
        "है",
        "हैं",
        "था",
        "थे",
        "को",
        "से",
        "में",
        "और",
        "का",
        "की",
        "के",
        "एक",
        "ಇದೆ",
        "ಆಗಿದೆ",
        "ಮತ್ತು",
        "ಒಂದು",
    }
)

_WH_TRIGGERS: frozenset[str] = frozenset(
    {
        "what",
        "where",
        "who",
        "when",
        "why",
        "how",
        "which",
        "क्या",
        "कहाँ",
        "कौन",
        "कब",
        "क्यों",
        "कैसे",
        "ಏನು",
        "ಎಲ್ಲಿ",
        "ಯಾರು",
        "ಯಾವಾಗ",
        "ಏಕೆ",
        "ಹೇಗೆ",
    }
)

_YES_NO_STARTERS: frozenset[str] = frozenset(
    {
        "do",
        "does",
        "did",
        "can",
        "could",
        "is",
        "are",
        "will",
        "would",
        "have",
        "has",
        "क्या",
    }
)

_NEGATION_TOKENS: frozenset[str] = frozenset(
    {
        "no",
        "not",
        "never",
        "none",
        "nope",
        "dont",
        "don't",
        "doesnt",
        "doesn't",
        "didnt",
        "didn't",
        "cant",
        "can't",
        "wont",
        "won't",
        "nahi",
        "nahin",
        "illa",
        "नहीं",
        "ना",
        "मत",
        "ಇಲ್ಲ",
        "ಬೇಡ",
    }
)

_AFFIRMATION_TOKENS: frozenset[str] = frozenset(
    {
        "yes",
        "yeah",
        "yep",
        "ok",
        "okay",
        "sure",
        "haan",
        "houdu",
        "हाँ",
        "हां",
        "जी",
        "ಹೌದು",
        "ಸರಿ",
    }
)


@dataclass(frozen=True, slots=True)
class SignEntry:
    """Dictionary definition for one animated sign gloss."""

    sign_id: str
    gloss: str
    cue: str
    triggers: tuple[str, ...]
    duration_ms: int = 1150
    is_wh: bool = False
    spatial_loci: tuple[str, str] | None = None


SIGN_LEXICON: tuple[SignEntry, ...] = (
    SignEntry(
        sign_id="give_a_to_b",
        gloss="GIVE (A → B)",
        cue="Cupped hand starts at Locus A (left) and arcs smoothly to Locus B (right).",
        triggers=(
            "person a gives something to person b",
            "person a gives to person b",
            "give to",
            "give",
            "pass",
            "देता",
            "दें",
            "दो",
            "ಕೊಡು",
            "ಕೊಡುತ್ತಾರೆ",
        ),
        duration_ms=1550,
        spatial_loci=("left", "right"),
    ),
    SignEntry(
        sign_id="i_love_you",
        gloss="I LOVE YOU",
        cue="Thumb, index finger, and pinky extended upward together toward the listener.",
        triggers=("i love you", "love you", "love", "प्यार", "ಪ್ರೀತಿ"),
        duration_ms=1350,
    ),
    SignEntry(
        sign_id="thank_you",
        gloss="THANK YOU",
        cue="Flat fingertips touch near the chin, then move smoothly forward toward the listener.",
        triggers=(
            "thank you",
            "thanks",
            "thank",
            "dhanyawad",
            "dhanyavadagalu",
            "धन्यवाद",
            "शुक्रिया",
            "ಧನ್ಯವಾದ",
            "ಧನ್ಯವಾದಗಳು",
        ),
        duration_ms=1250,
    ),
    SignEntry(
        sign_id="good_morning",
        gloss="GOOD MORNING",
        cue="Flat hand touches chin, then right palm rises smoothly from the crook of the left arm.",
        triggers=("good morning", "morning", "सुप्रभात", "ಶುಭೋದಯ"),
        duration_ms=1500,
    ),
    SignEntry(
        sign_id="hello",
        gloss="HELLO",
        cue="Open palm raised near the temple, sweeping outward in a welcoming wave.",
        triggers=(
            "hello",
            "hi",
            "hey",
            "namaste",
            "namaskara",
            "नमस्ते",
            "नमस्कार",
            "ನಮಸ್ಕಾರ",
        ),
        duration_ms=1250,
    ),
    SignEntry(
        sign_id="please",
        gloss="PLEASE",
        cue="Flat open palm circles gently over the center of the chest.",
        triggers=("please", "kindly", "कृपया", "ದಯವಿಟ್ಟು"),
        duration_ms=1200,
    ),
    SignEntry(
        sign_id="sorry",
        gloss="SORRY",
        cue="Closed fist circles calmly over the heart with a sympathetic head tilt.",
        triggers=("sorry", "apologize", "apologies", "excuse me", "माफ़", "क्षमा", "ಕ್ಷಮಿಸಿ"),
        duration_ms=1250,
    ),
    SignEntry(
        sign_id="help",
        gloss="HELP",
        cue="Thumbs-up fist resting on the opposite open palm, lifting upward together.",
        triggers=("help", "assist", "madad", "sahaya", "मदद", "सहायता", "ಸಹಾಯ"),
        duration_ms=1300,
    ),
    SignEntry(
        sign_id="water",
        gloss="WATER",
        cue="W-handshape taps gently near the chin twice.",
        triggers=("water", "paani", "neeru", "drink", "पानी", "जल", "ನೀರು"),
        duration_ms=1200,
    ),
    SignEntry(
        sign_id="yes",
        gloss="YES",
        cue="Closed fist nods cleanly forward at the wrist twice.",
        triggers=(
            "yes",
            "yeah",
            "yep",
            "okay",
            "ok",
            "haan",
            "houdu",
            "हाँ",
            "हां",
            "ठीक है",
            "ಹೌದು",
            "ಸರಿ",
        ),
        duration_ms=1150,
    ),
    SignEntry(
        sign_id="no",
        gloss="NO",
        cue="Index and middle fingers snap shut against the thumb cleanly in front of the chest.",
        triggers=("no", "nope", "not", "nahi", "illa", "नहीं", "ना", "ಇಲ್ಲ"),
        duration_ms=1100,
    ),
    SignEntry(
        sign_id="welcome",
        gloss="WELCOME",
        cue="Open palm sweeps gracefully inward toward the chest.",
        triggers=("welcome", "स्वागत", "ಸ್ವಾಗತ"),
        duration_ms=1200,
    ),
    SignEntry(
        sign_id="good",
        gloss="GOOD",
        cue="Flat hand moves cleanly from chin down onto the opposite open palm.",
        triggers=("good", "great", "fine", "nice", "अच्छा", "बढ़िया", "ಚೆನ್ನಾಗಿದೆ", "ಒಳ್ಳೆಯದು"),
        duration_ms=1150,
    ),
    SignEntry(
        sign_id="name",
        gloss="NAME",
        cue="Two-finger H-handshapes tap together cleanly in front of the chest.",
        triggers=("name", "called", "naam", "नाम", "ಹೆಸರು"),
        duration_ms=1150,
    ),
    SignEntry(
        sign_id="what",
        gloss="WHAT",
        cue="Both open palms face upward and shake gently side-to-side with furrowed brows.",
        triggers=("what", "क्या", "ಏನು"),
        duration_ms=1150,
        is_wh=True,
    ),
    SignEntry(
        sign_id="where",
        gloss="WHERE",
        cue="Index finger points up and waves side-to-side with furrowed brows.",
        triggers=("where", "कहाँ", "ಎಲ್ಲಿ"),
        duration_ms=1100,
        is_wh=True,
    ),
    SignEntry(
        sign_id="who",
        gloss="WHO",
        cue="Index finger circles lightly near the chin with furrowed brows.",
        triggers=("who", "कौन", "ಯಾರು"),
        duration_ms=1100,
        is_wh=True,
    ),
    SignEntry(
        sign_id="want",
        gloss="WANT",
        cue="Curved open hands pull gently inward toward the torso.",
        triggers=("want", "need", "wish", "चाहिए", "ಬೇಕು"),
        duration_ms=1150,
    ),
    SignEntry(
        sign_id="stop",
        gloss="STOP",
        cue="Vertical flat hand chops cleanly onto the opposite horizontal palm.",
        triggers=("stop", "halt", "wait", "रुको", "ನಿಲ್ಲು"),
        duration_ms=1100,
    ),
    SignEntry(
        sign_id="your",
        gloss="YOUR",
        cue="Flat palm extends smoothly forward toward the listener.",
        triggers=("your", "yours", "तुम्हारा", "आपका", "ನಿಮ್ಮ"),
        duration_ms=950,
    ),
    SignEntry(
        sign_id="you",
        gloss="YOU",
        cue="Index finger points smoothly forward toward the listener.",
        triggers=("you", "tum", "aap", "तुम", "आप", "ನೀವು"),
        duration_ms=950,
    ),
    SignEntry(
        sign_id="my",
        gloss="MY",
        cue="Flat palm rests gently against the center of the chest.",
        triggers=("my", "mine", "मेरा", "मेरी", "ನನ್ನ"),
        duration_ms=950,
    ),
    SignEntry(
        sign_id="me",
        gloss="ME",
        cue="Index finger points gently inward to the chest.",
        triggers=("i", "me", "myself", "मैं", "मुझे", "ನಾನು", "ನನಗೆ"),
        duration_ms=950,
    ),
    SignEntry(
        sign_id="change",
        gloss="CHANGE",
        cue="Both hooked hands twist smoothly over one another in front of the chest.",
        triggers=("change", "switch", "modify", "बदल", "बदलें", "ಬದಲಾಯಿಸಿ"),
        duration_ms=1150,
    ),
    SignEntry(
        sign_id="correct",
        gloss="CORRECT",
        cue="Pinched fingertips tap cleanly onto the opposite upward open palm.",
        triggers=("correct", "right", "exact", "सही", "ಸರಿ"),
        duration_ms=1100,
    ),
    SignEntry(
        sign_id="goodbye",
        gloss="GOODBYE",
        cue="Open palm raised near shoulder height, waving gently forward.",
        triggers=("goodbye", "bye", "bye bye", "farewell", "अलविदा", "ವಿದಾಯ"),
        duration_ms=1200,
    ),
    SignEntry(
        sign_id="home",
        gloss="HOME",
        cue="Both flat palms meet at fingertips forming a peaked roof in front of the chest.",
        triggers=("home", "house", "घर", "ಮನೆ"),
        duration_ms=1150,
    ),
    SignEntry(
        sign_id="hug",
        gloss="HUG",
        cue="Both fists cross snugly over the chest in an embracing posture.",
        triggers=("hug", "embrace", "गले", "अಪ್ಪಿಕೊಳ್ಳಿ", "ಅಪ್ಪಿಕೊಳ್ಳಿ"),
        duration_ms=1250,
    ),
    SignEntry(
        sign_id="meet",
        gloss="MEET",
        cue="Both upright index fingers move smoothly together side-by-side at chest height.",
        triggers=("meet", "meeting", "मिलना", "भेंट", "ಭೇಟಿ", "ಭೇಟಿಯಾಗಿ"),
        duration_ms=1150,
    ),
    SignEntry(
        sign_id="touch",
        gloss="TOUCH",
        cue="Flat fingertips gently tap onto the back of the opposite horizontal hand.",
        triggers=("touch", "tap", "contact", "छूना", "स्पर्श", "ಸ್ಪರ್ಶಿಸಿ"),
        duration_ms=1100,
    ),
)

# Build fast trigger lookup sorted longest-first for multi-word phrase matching.
_PHRASE_TRIGGERS: list[tuple[str, SignEntry]] = sorted(
    (
        (trig.lower(), entry)
        for entry in SIGN_LEXICON
        for trig in entry.triggers
    ),
    key=lambda pair: len(pair[0]),
    reverse=True,
)


@dataclass(slots=True)
class SignStep:
    """Single playable step in the 3D signing timeline."""

    type: str
    sign_id: str
    gloss: str
    cue: str
    duration_ms: int
    letters: list[str] = field(default_factory=list)
    spatial_loci: tuple[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": self.type,
            "id": self.sign_id,
            "gloss": self.gloss,
            "cue": self.cue,
            "duration_ms": self.duration_ms,
        }
        if self.letters:
            payload["letters"] = list(self.letters)
        if self.spatial_loci is not None:
            payload["spatial_loci"] = list(self.spatial_loci)
        return payload


@dataclass(slots=True)
class SignPlan:
    """Complete spoken-text -> sign-gloss + NMM animation plan."""

    input_text: str
    language: str
    dialect: str
    sentence_type: str
    nmm: dict[str, Any]
    gloss_line: str
    steps: list[SignStep]

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_text": self.input_text,
            "language": self.language,
            "dialect": self.dialect,
            "sentence_type": self.sentence_type,
            "nmm": dict(self.nmm),
            "gloss_line": self.gloss_line,
            "steps": [step.to_dict() for step in self.steps],
        }


def _tokenize(text: str) -> list[str]:
    return [tok for tok in re.findall(r"[^\s,.;:!?\"'()\[\]{}]+", text) if tok]


def _classify_clause(raw_text: str, raw_tokens_lower: list[str]) -> tuple[str, dict[str, Any]]:
    has_question_mark = "?" in raw_text
    has_wh = any(tok in _WH_TRIGGERS for tok in raw_tokens_lower)
    has_neg = any(tok in _NEGATION_TOKENS for tok in raw_tokens_lower)
    has_aff = any(tok in _AFFIRMATION_TOKENS for tok in raw_tokens_lower)
    starts_yn = bool(raw_tokens_lower and raw_tokens_lower[0] in _YES_NO_STARTERS)

    if has_wh:
        return (
            "wh_question",
            {
                "brow_state": "furrowed",
                "head_nod": False,
                "head_shake": False,
                "head_tilt": True,
            },
        )
    if has_question_mark or starts_yn:
        return (
            "yes_no_question",
            {
                "brow_state": "raised",
                "head_nod": False,
                "head_shake": False,
                "head_tilt": True,
            },
        )
    if has_neg:
        return (
            "negation",
            {
                "brow_state": "neutral",
                "head_nod": False,
                "head_shake": True,
                "head_tilt": False,
            },
        )
    if has_aff:
        return (
            "affirmation",
            {
                "brow_state": "neutral",
                "head_nod": True,
                "head_shake": False,
                "head_tilt": False,
            },
        )
    return (
        "statement",
        {
            "brow_state": "neutral",
            "head_nod": False,
            "head_shake": False,
            "head_tilt": False,
        },
    )


def _match_trigger_at(tokens_lower: list[str], idx: int) -> tuple[int, SignEntry] | None:
    for trig, entry in _PHRASE_TRIGGERS:
        trig_parts = trig.split()
        span = len(trig_parts)
        if idx + span <= len(tokens_lower) and tokens_lower[idx : idx + span] == trig_parts:
            return span, entry
    return None


def text_to_sign_plan(
    text: str,
    *,
    language: str = "en",
    dialect: str = "isl-south",
) -> SignPlan:
    """Convert spoken/typed text into ordered sign steps, NMM markers, and fingerspelling."""
    cleaned = (text or "").strip()
    if not cleaned:
        return SignPlan(
            input_text="",
            language=language,
            dialect=dialect,
            sentence_type="statement",
            nmm={
                "brow_state": "neutral",
                "head_nod": False,
                "head_shake": False,
                "head_tilt": False,
            },
            gloss_line="",
            steps=[],
        )

    raw_tokens = _tokenize(cleaned)
    tokens_lower = [tok.lower() for tok in raw_tokens]
    sentence_type, nmm = _classify_clause(cleaned, tokens_lower)

    main_steps: list[SignStep] = []
    wh_steps: list[SignStep] = []

    idx = 0
    while idx < len(tokens_lower):
        matched = _match_trigger_at(tokens_lower, idx)
        if matched is not None:
            span, entry = matched
            step = SignStep(
                type="sign",
                sign_id=entry.sign_id,
                gloss=entry.gloss,
                cue=entry.cue,
                duration_ms=entry.duration_ms,
                spatial_loci=entry.spatial_loci,
            )
            if entry.is_wh:
                wh_steps.append(step)
            else:
                main_steps.append(step)
            idx += span
            continue

        word = raw_tokens[idx]
        low = tokens_lower[idx]
        idx += 1
        if low in _COPULA_AND_FILLERS:
            continue

        # Fingerspell ASCII alphanumeric words; non-ASCII unknown words show as gloss cues.
        ascii_letters = [ch.upper() for ch in word if ch.isascii() and ch.isalnum()]
        if ascii_letters:
            letters = ascii_letters[:12]
            main_steps.append(
                SignStep(
                    type="fingerspell",
                    sign_id="fingerspell",
                    gloss=word.upper(),
                    cue=f"Fingerspell {'-'.join(letters)}",
                    duration_ms=max(900, len(letters) * 460 + 250),
                    letters=letters,
                )
            )
        else:
            main_steps.append(
                SignStep(
                    type="sign",
                    sign_id="unknown_gloss",
                    gloss=word.upper(),
                    cue=f"Sign gloss cue: {word}",
                    duration_ms=1050,
                )
            )

    # In ISL / ASL grammar, WH-question words move to the end of the clause.
    ordered_steps = main_steps + wh_steps
    gloss_line = " ".join(step.gloss for step in ordered_steps)

    return SignPlan(
        input_text=cleaned,
        language=language,
        dialect=dialect,
        sentence_type=sentence_type,
        nmm=nmm,
        gloss_line=gloss_line,
        steps=ordered_steps,
    )
