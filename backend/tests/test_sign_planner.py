"""Tests for SignBridge spoken text -> sign gloss & NMM planner."""

from fastapi.testclient import TestClient

from signbridge.server import app
from signbridge.sign_planner import text_to_sign_plan


def test_text_to_sign_plan_wh_question_reordering() -> None:
    plan = text_to_sign_plan("What is your name?")
    assert plan.sentence_type == "wh_question"
    assert plan.nmm["brow_state"] == "furrowed"
    assert plan.nmm["head_tilt"] is True
    assert [s.sign_id for s in plan.steps] == ["your", "name", "what"]
    assert plan.gloss_line == "YOUR NAME WHAT"


def test_text_to_sign_plan_multi_phrase_and_fingerspell() -> None:
    plan = text_to_sign_plan("Hello, my name is Pranam, please help!")
    assert [s.sign_id for s in plan.steps] == [
        "hello",
        "my",
        "name",
        "fingerspell",
        "please",
        "help",
    ]
    assert plan.steps[3].letters == ["P", "R", "A", "N", "A", "M"]
    assert plan.gloss_line == "HELLO MY NAME PRANAM PLEASE HELP"


def test_text_to_sign_plan_hindi_kannada_and_negation() -> None:
    hi_plan = text_to_sign_plan("नमस्ते धन्यवाद पानी", language="hi")
    assert [s.sign_id for s in hi_plan.steps] == ["hello", "thank_you", "water"]

    kn_neg = text_to_sign_plan("ಇಲ್ಲ ಸಹಾಯ", language="kn")
    assert kn_neg.sentence_type == "negation"
    assert kn_neg.nmm["head_shake"] is True
    assert [s.sign_id for s in kn_neg.steps] == ["no", "help"]


def test_text_to_sign_endpoint() -> None:
    with TestClient(app) as client:
        resp = client.post(
            "/api/text-to-sign",
            json={"text": "Do you want water?", "language": "en", "dialect": "isl-south"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["sentence_type"] == "yes_no_question"
        assert data["nmm"]["brow_state"] == "raised"
        assert [s["id"] for s in data["steps"]] == ["you", "want", "water"]
