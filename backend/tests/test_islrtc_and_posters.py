"""Tests for the ISLRTC multilingual dictionary, 5 uploaded sign posters, and expanded lexicon."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from signbridge.config import (
    DEFAULT_DATASET_DIR,
    DEFAULT_ISLRTC_DICTIONARY_PATH,
    DEFAULT_POSTER_REFERENCES_PATH,
    translate_gesture_label,
)
from signbridge.islrtc_dataset import (
    IslrtcDictionaryStore,
    clean_islrtc_term,
    extract_youtube_embed_id,
    parse_listpage_entries,
)
from signbridge.poster_dataset import PosterReferenceStore
from signbridge.server import app
from signbridge.sign_dataset import SignDatasetStore
from signbridge.sign_planner import text_to_sign_plan


def test_islrtc_html_parsers() -> None:
    sample_html = """
    <a href='searchpage.php?type=list&id=8854&search=1 One'> 1 One </a>
    <a href="searchpage.php?type=list&id=101">  _Namaste &amp; Hello  </a>
    <a href="searchpage.php?type=list&id=8854">Duplicate</a>
    """
    rows = parse_listpage_entries(sample_html)
    assert rows == [("8854", "1 One"), ("101", "Namaste & Hello")]
    assert clean_islrtc_term("  __<AB  ") == "AB"
    assert (
        extract_youtube_embed_id(
            '<iframe src="https://www.youtube.com/embed/lH6pT7nQb9A?rel=0"></iframe>'
        )
        == "lH6pT7nQb9A"
    )


def test_harvested_islrtc_dictionary_store() -> None:
    assert DEFAULT_ISLRTC_DICTIONARY_PATH.is_file()
    store = IslrtcDictionaryStore(DEFAULT_ISLRTC_DICTIONARY_PATH)
    summary = store.get_summary()
    assert summary["available"] is True
    assert summary["total_entries"] >= 15000
    assert summary["with_hindi"] >= 15000
    assert summary["with_kannada"] >= 10000
    assert len(summary["categories"]) == 12
    assert len(summary["tutorials"]) == 6

    res_water = store.search(query="Water", category="1", limit=10)
    assert res_water["total"] > 0
    assert res_water["entries"][0]["en"].lower().startswith("water")
    assert res_water["entries"][0]["hi"]

    res_hi = store.search(query="धन्यवाद", limit=10)
    assert res_hi["total"] > 0


def test_poster_references_and_extracted_samples() -> None:
    assert DEFAULT_POSTER_REFERENCES_PATH.is_file()
    ref_store = PosterReferenceStore(DEFAULT_POSTER_REFERENCES_PATH)
    all_refs = ref_store.list_references()
    assert all_refs["total"] == 92
    assert len(all_refs["posters"]) == 5

    photo_refs = ref_store.list_references(poster_id="p4_first_signs")
    assert photo_refs["total"] == 15

    emerg_refs = ref_store.list_references(poster_id="p5_emergency", query="ambulance")
    assert emerg_refs["total"] == 1
    assert emerg_refs["items"][0]["hi"]

    ds_summary = SignDatasetStore(DEFAULT_DATASET_DIR).get_summary()
    assert ds_summary["total_samples"] >= 113
    assert ds_summary["total_classes"] >= 69
    assert ds_summary["source_totals"].get("poster_photo", 0) == 15
    assert ds_summary["source_totals"].get("poster_illustrated", 0) == 19

    for label in ("Milk", "Ambulance", "I'm Sorry", "You're Welcome", "Bathroom"):
        tr = translate_gesture_label(label)
        assert tr["hi"]
        assert tr["kn"]


def test_expanded_sign_planner_vocabulary() -> None:
    plan = text_to_sign_plan("Mom please give more milk and eat now")
    ids = [step.sign_id for step in plan.steps]
    for expected in ("mom", "please", "give_a_to_b", "more", "milk", "eat"):
        assert expected in ids

    wh_plan = text_to_sign_plan("When do we go home?")
    assert wh_plan.sentence_type == "wh_question"
    assert wh_plan.steps[-1].sign_id == "when"


def test_islrtc_and_poster_endpoints_with_mock_video(tmp_path: Path, monkeypatch) -> None:
    sample_dict = {
        "version": 1,
        "source_url": "https://divyangjan.depwd.gov.in/islrtc/",
        "drive_url": "https://drive.google.com/",
        "harvested_at": "2026-11-18T00:00:00Z",
        "total_entries": 1,
        "categories": [{"id": "1", "en": "Everyday", "hi": "रोज़मर्रा", "kn": "ಪ್ರತಿದಿನ", "count": 1}],
        "tutorials": [],
        "language_portals": [],
        "entries": [
            {
                "id": "29",
                "category_id": "1",
                "category_en": "Everyday",
                "en": "Water",
                "hi": "पानी",
                "kn": "ನೀರು",
                "youtube_id": None,
                "portal_url": "https://divyangjan.depwd.gov.in/islrtc/searchpage.php?type=list&id=29",
            }
        ],
    }
    dict_path = tmp_path / "islrtc_test.json"
    dict_path.write_text(json.dumps(sample_dict), encoding="utf-8")

    mock_resp = MagicMock()
    mock_resp.text = '<iframe src="https://www.youtube.com/embed/Ykv_2L4-7RY?si=abc"></iframe>'
    mock_resp.raise_for_status.return_value = None
    mock_client = MagicMock()
    mock_client.__enter__.return_value.get.return_value = mock_resp

    with TestClient(app) as client:
        monkeypatch.setattr(app.state, "islrtc_store", IslrtcDictionaryStore(dict_path), raising=False)

        sum_resp = client.get("/api/islrtc/summary")
        assert sum_resp.status_code == 200
        assert sum_resp.json()["total_entries"] == 1

        search_resp = client.get("/api/islrtc/dictionary?q=water")
        assert search_resp.status_code == 200
        assert search_resp.json()["entries"][0]["hi"] == "पानी"

        with patch("signbridge.islrtc_dataset.httpx.Client", return_value=mock_client):
            entry_resp = client.get("/api/islrtc/entry/29")
        assert entry_resp.status_code == 200
        assert entry_resp.json()["entry"]["youtube_id"] == "Ykv_2L4-7RY"

        refs_resp = client.get("/api/dataset/references?poster_id=p1_classroom")
        assert refs_resp.status_code == 200
        assert refs_resp.json()["total"] == 23
