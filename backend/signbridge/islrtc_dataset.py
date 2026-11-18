"""Official ISLRTC (DEPwD) Indian Sign Language dictionary importer & search store."""

from __future__ import annotations

from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re
import ssl
import threading
from typing import Any
from urllib.parse import quote_plus

import httpx

try:
    import truststore
except ImportError:  # pragma: no cover - optional fallback when truststore unavailable
    truststore = None  # type: ignore[assignment]

from .config import DEFAULT_ISLRTC_DICTIONARY_PATH

ISLRTC_BASE_URL = "https://divyangjan.depwd.gov.in/islrtc"
ISLRTC_DRIVE_URL = (
    "https://drive.google.com/drive/folders/1WQnYirJsKGVOL8Z-SjH7ip3AVE9wqwGO?usp=sharing"
)

ISLRTC_CATEGORIES: tuple[dict[str, str], ...] = (
    {"id": "1", "en": "Everyday", "hi": "रोज़मर्रा", "kn": "ಪ್ರತಿದಿನ"},
    {"id": "2", "en": "Academic", "hi": "शैक्षिक", "kn": "ಶೈಕ್ಷಣಿಕ"},
    {"id": "3", "en": "Agriculture", "hi": "कृषि", "kn": "ಕೃಷಿ"},
    {"id": "4", "en": "Countries", "hi": "देश", "kn": "ದೇಶಗಳು"},
    {"id": "5", "en": "Finance and Banking", "hi": "वित्त और बैंकिंग", "kn": "ಹಣಕಾಸು ಮತ್ತು ಬ್ಯಾಂಕಿಂಗ್"},
    {"id": "6", "en": "Idioms", "hi": "मुहावरें", "kn": "ನಾಣ್ಣುಡಿಗಳು"},
    {"id": "7", "en": "Legal", "hi": "कानूनी", "kn": "ಕಾನೂನುಬದ್ಧ"},
    {"id": "8", "en": "Medical", "hi": "चिकित्सा", "kn": "ವೈದ್ಯಕೀಯ"},
    {"id": "9", "en": "Numbers", "hi": "संख्या", "kn": "ಸಂಖ್ಯೆಗಳು"},
    {"id": "10", "en": "Regional", "hi": "क्षेत्रीय", "kn": "ಪ್ರಾದೇಶಿಕ"},
    {"id": "11", "en": "States And Cities", "hi": "राज्य और शहर", "kn": "ರಾಜ್ಯಗಳು ಮತ್ತು ನಗರಗಳು"},
    {"id": "12", "en": "Technical", "hi": "तकनीकी", "kn": "ತಾಂತ್ರಿಕ"},
)
_CATEGORY_BY_ID: dict[str, dict[str, str]] = {c["id"]: c for c in ISLRTC_CATEGORIES}

ISLRTC_TUTORIAL_PLAYLISTS: tuple[dict[str, str], ...] = (
    {
        "youtube_id": "HT5VidjmJ9o",
        "en": "Greetings and Communicative Phrases",
        "hi": "अभिवादन और सम्प्रेषण सम्बन्धी अन्य वाक्यांश",
        "embed_url": "https://www.youtube.com/embed/HT5VidjmJ9o",
        "watch_url": "https://www.youtube.com/watch?v=HT5VidjmJ9o",
    },
    {
        "youtube_id": "vtFLWKxUX4A",
        "en": "People and Relations",
        "hi": "लोग और सम्बन्ध",
        "embed_url": "https://www.youtube.com/embed/vtFLWKxUX4A",
        "watch_url": "https://www.youtube.com/watch?v=vtFLWKxUX4A",
    },
    {
        "youtube_id": "SgVLj-CHvZ4",
        "en": "Food Related",
        "hi": "भोजन सम्बन्धी",
        "embed_url": "https://www.youtube.com/embed/SgVLj-CHvZ4",
        "watch_url": "https://www.youtube.com/watch?v=SgVLj-CHvZ4",
    },
    {
        "youtube_id": "FXliBRJMltY",
        "en": "Daily Routines & Everyday Items",
        "hi": "दैनिक दिनचर्या और रोज़मर्रा की चीजें",
        "embed_url": "https://www.youtube.com/embed/FXliBRJMltY",
        "watch_url": "https://www.youtube.com/watch?v=FXliBRJMltY",
    },
    {
        "youtube_id": "HaMafb3Bv_U",
        "en": "Everyday Items",
        "hi": "रोज़मर्रा की वस्तुएं",
        "embed_url": "https://www.youtube.com/embed/HaMafb3Bv_U",
        "watch_url": "https://www.youtube.com/watch?v=HaMafb3Bv_U",
    },
    {
        "youtube_id": "BtOQjIrcwUU",
        "en": "Mixed Words",
        "hi": "मिश्रित शब्द",
        "embed_url": "https://www.youtube.com/embed/BtOQjIrcwUU",
        "watch_url": "https://www.youtube.com/watch?v=BtOQjIrcwUU",
    },
)

ISLRTC_LANGUAGE_PORTALS: tuple[dict[str, str], ...] = (
    {"lang": "English", "url": f"{ISLRTC_BASE_URL}/index.php"},
    {"lang": "Hindi", "url": f"{ISLRTC_BASE_URL}/hindi/index.php"},
    {"lang": "Kannada", "url": f"{ISLRTC_BASE_URL}/kannada/index.php"},
    {"lang": "Assamese", "url": f"{ISLRTC_BASE_URL}/assamese/index.php"},
    {"lang": "Bengali", "url": f"{ISLRTC_BASE_URL}/bengali/index.php"},
    {"lang": "Gujarati", "url": f"{ISLRTC_BASE_URL}/gujarthi/index.php"},
    {"lang": "Malayalam", "url": f"{ISLRTC_BASE_URL}/malayalam/index.php"},
    {"lang": "Marathi", "url": f"{ISLRTC_BASE_URL}/marathi/index.php"},
    {"lang": "Odia", "url": f"{ISLRTC_BASE_URL}/odia/index.php"},
    {"lang": "Punjabi", "url": f"{ISLRTC_BASE_URL}/punjabi/index.php"},
    {"lang": "Tamil", "url": f"{ISLRTC_BASE_URL}/tamil/"},
    {"lang": "Telugu", "url": f"{ISLRTC_BASE_URL}/telugu/"},
)

_LINK_RE = re.compile(
    r"<a[^>]*href=['\"]([^'\"]*searchpage\.php\?type=list&id=(\d+)[^'\"]*)['\"][^>]*>(.*?)</a>",
    re.IGNORECASE | re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")
_YOUTUBE_EMBED_RE = re.compile(r"youtube\.com/embed/([A-Za-z0-9_-]{6,20})")
_LEADING_ARTIFACT_RE = re.compile(r"^[\s_<>]+")


def build_islrtc_ssl_context() -> ssl.SSLContext:
    """Create a TLS client context using OS trust certificates for Indian .gov.in chains."""
    if truststore is not None:
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return ssl.create_default_context()


def clean_islrtc_term(raw: str) -> str:
    """Clean HTML markup and accidental leading artifacts from a scraped dictionary term."""
    text = html.unescape(_TAG_RE.sub(" ", raw or ""))
    text = " ".join(text.split())
    cleaned = _LEADING_ARTIFACT_RE.sub("", text).strip()
    return cleaned or text


def parse_listpage_entries(html_text: str) -> list[tuple[str, str]]:
    """Extract ordered `(entry_id, term)` pairs from an ISLRTC `listpage2.php` HTML response."""
    seen: set[str] = set()
    rows: list[tuple[str, str]] = []
    for _href, entry_id, raw_label in _LINK_RE.findall(html_text or ""):
        eid = entry_id.strip()
        if not eid or eid in seen:
            continue
        term = clean_islrtc_term(raw_label)
        if not term:
            continue
        seen.add(eid)
        rows.append((eid, term))
    return rows


def extract_youtube_embed_id(html_text: str) -> str | None:
    """Extract the embedded YouTube video ID from an ISLRTC `searchpage.php` page."""
    match = _YOUTUBE_EMBED_RE.search(html_text or "")
    return match.group(1) if match else None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{threading.get_ident()}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _fetch_category_terms(
    client: httpx.Client,
    endpoint_url: str,
    category_id: str,
    submit_label: str,
) -> list[tuple[str, str]]:
    response = client.post(
        endpoint_url,
        data={"category": category_id, "submit": submit_label},
    )
    response.raise_for_status()
    return parse_listpage_entries(response.text)


def harvest_islrtc_dictionary(
    output_path: Path = DEFAULT_ISLRTC_DICTIONARY_PATH,
    *,
    category_ids: list[str] | None = None,
    timeout_seconds: float = 45.0,
) -> dict[str, Any]:
    """Fetch ISLRTC categories in English, Hindi, and Kannada and write a unified JSON index."""
    existing_videos: dict[str, str] = {}
    if output_path.is_file():
        try:
            old_data = json.loads(output_path.read_text(encoding="utf-8"))
            for item in old_data.get("entries", []):
                eid = str(item.get("id") or "")
                yid = item.get("youtube_id")
                if eid and yid:
                    existing_videos[eid] = str(yid)
        except (OSError, json.JSONDecodeError):
            pass

    target_cats = (
        [c for c in ISLRTC_CATEGORIES if c["id"] in set(category_ids)]
        if category_ids
        else list(ISLRTC_CATEGORIES)
    )

    entries_by_id: dict[str, dict[str, Any]] = {}
    ssl_ctx = build_islrtc_ssl_context()

    endpoints = (
        ("en", f"{ISLRTC_BASE_URL}/listpage2.php", "Search"),
        ("hi", f"{ISLRTC_BASE_URL}/hindi/listpage2.php", "खोज"),
        ("kn", f"{ISLRTC_BASE_URL}/kannada/listpage2.php", "ಹುಡುಕು"),
    )

    with httpx.Client(verify=ssl_ctx, timeout=timeout_seconds, follow_redirects=True) as client:
        for cat in target_cats:
            cid = cat["id"]
            for lang_code, url, submit_val in endpoints:
                rows = _fetch_category_terms(client, url, cid, submit_val)
                for eid, term in rows:
                    record = entries_by_id.get(eid)
                    if record is None:
                        yid = existing_videos.get(eid)
                        record = {
                            "id": eid,
                            "category_id": cid,
                            "category_en": cat["en"],
                            "en": "",
                            "hi": "",
                            "kn": "",
                            "youtube_id": yid,
                            "portal_url": (
                                f"{ISLRTC_BASE_URL}/searchpage.php?type=list&id={eid}"
                            ),
                        }
                        entries_by_id[eid] = record
                    if not record[lang_code]:
                        record[lang_code] = term
                    if lang_code == "en" and term:
                        record["portal_url"] = (
                            f"{ISLRTC_BASE_URL}/searchpage.php"
                            f"?type=list&id={eid}&search={quote_plus(term)}"
                        )

    ordered = sorted(
        entries_by_id.values(),
        key=lambda e: (int(e["category_id"]), (e["en"] or e["hi"] or e["kn"]).lower()),
    )

    category_counts: dict[str, int] = {}
    for item in ordered:
        cid = item["category_id"]
        category_counts[cid] = category_counts.get(cid, 0) + 1

    payload: dict[str, Any] = {
        "version": 1,
        "source_url": f"{ISLRTC_BASE_URL}/",
        "drive_url": ISLRTC_DRIVE_URL,
        "harvested_at": datetime.now(timezone.utc).isoformat(),
        "total_entries": len(ordered),
        "categories": [
            {**cat, "count": category_counts.get(cat["id"], 0)}
            for cat in ISLRTC_CATEGORIES
        ],
        "tutorials": list(ISLRTC_TUTORIAL_PLAYLISTS),
        "language_portals": list(ISLRTC_LANGUAGE_PORTALS),
        "entries": ordered,
    }
    _atomic_write_json(output_path, payload)
    return {
        "output_path": str(output_path),
        "total_entries": len(ordered),
        "with_hindi": sum(1 for e in ordered if e.get("hi")),
        "with_kannada": sum(1 for e in ordered if e.get("kn")),
        "category_counts": category_counts,
    }


class IslrtcDictionaryStore:
    """Thread-safe indexed reader & YouTube embed resolver for the ISLRTC dataset."""

    def __init__(self, dictionary_path: Path = DEFAULT_ISLRTC_DICTIONARY_PATH) -> None:
        self.dictionary_path = Path(dictionary_path)
        self._lock = threading.Lock()
        self._payload: dict[str, Any] | None = None
        self._entries_by_id: dict[str, dict[str, Any]] = {}
        self._search_index: list[tuple[str, str, str, str, dict[str, Any]]] = []

    def _load_unlocked(self) -> dict[str, Any]:
        if self._payload is not None:
            return self._payload

        if not self.dictionary_path.is_file():
            self._payload = {
                "version": 1,
                "source_url": f"{ISLRTC_BASE_URL}/",
                "drive_url": ISLRTC_DRIVE_URL,
                "harvested_at": None,
                "total_entries": 0,
                "categories": [{**c, "count": 0} for c in ISLRTC_CATEGORIES],
                "tutorials": list(ISLRTC_TUTORIAL_PLAYLISTS),
                "language_portals": list(ISLRTC_LANGUAGE_PORTALS),
                "entries": [],
            }
        else:
            self._payload = json.loads(
                self.dictionary_path.read_text(encoding="utf-8")
            )

        self._rebuild_index_unlocked()
        return self._payload

    def _rebuild_index_unlocked(self) -> None:
        entries = self._payload.get("entries", []) if self._payload else []
        by_id: dict[str, dict[str, Any]] = {}
        search_rows: list[tuple[str, str, str, str, dict[str, Any]]] = []
        for entry in entries:
            eid = str(entry.get("id") or "")
            if not eid:
                continue
            by_id[eid] = entry
            en_l = (entry.get("en") or "").lower()
            hi_l = (entry.get("hi") or "").lower()
            kn_l = (entry.get("kn") or "").lower()
            cat_l = (entry.get("category_en") or "").lower()
            search_rows.append((en_l, hi_l, kn_l, cat_l, entry))
        self._entries_by_id = by_id
        self._search_index = search_rows

    def reload(self) -> None:
        """Clear in-memory cache so next read reloads from disk."""
        with self._lock:
            self._payload = None
            self._entries_by_id = {}
            self._search_index = []

    def get_summary(self) -> dict[str, Any]:
        """Return dataset statistics, categories, tutorials, and portals."""
        with self._lock:
            payload = self._load_unlocked()
            entries = payload.get("entries", [])
            return {
                "available": self.dictionary_path.is_file(),
                "source_url": payload.get("source_url", f"{ISLRTC_BASE_URL}/"),
                "drive_url": payload.get("drive_url", ISLRTC_DRIVE_URL),
                "harvested_at": payload.get("harvested_at"),
                "total_entries": len(entries),
                "with_hindi": sum(1 for e in entries if e.get("hi")),
                "with_kannada": sum(1 for e in entries if e.get("kn")),
                "cached_videos": sum(1 for e in entries if e.get("youtube_id")),
                "categories": payload.get("categories", []),
                "tutorials": payload.get("tutorials", []),
                "language_portals": payload.get("language_portals", []),
            }

    @staticmethod
    def _decorate_entry(entry: dict[str, Any]) -> dict[str, Any]:
        cat = _CATEGORY_BY_ID.get(str(entry.get("category_id") or ""), {})
        yid = entry.get("youtube_id")
        return {
            "id": entry["id"],
            "category_id": entry.get("category_id", ""),
            "category_en": entry.get("category_en") or cat.get("en", ""),
            "category_hi": cat.get("hi", ""),
            "category_kn": cat.get("kn", ""),
            "en": entry.get("en", ""),
            "hi": entry.get("hi", ""),
            "kn": entry.get("kn", ""),
            "youtube_id": yid,
            "embed_url": f"https://www.youtube.com/embed/{yid}" if yid else None,
            "watch_url": f"https://www.youtube.com/watch?v={yid}" if yid else None,
            "portal_url": entry.get("portal_url")
            or f"{ISLRTC_BASE_URL}/searchpage.php?type=list&id={entry['id']}",
        }

    def search(
        self,
        *,
        query: str = "",
        category: str = "all",
        limit: int = 36,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Search ISLRTC entries across EN/HI/KN with relevance ranking and pagination."""
        clean_q = " ".join((query or "").strip().lower().split())
        cat_filter = (category or "all").strip().lower()
        safe_limit = max(1, min(100, int(limit)))
        safe_offset = max(0, int(offset))

        with self._lock:
            self._load_unlocked()
            exact_hits: list[dict[str, Any]] = []
            prefix_hits: list[dict[str, Any]] = []
            contains_hits: list[dict[str, Any]] = []

            for en_l, hi_l, kn_l, cat_l, entry in self._search_index:
                if cat_filter not in ("", "all"):
                    cid_match = str(entry.get("category_id") or "").lower() == cat_filter
                    name_match = cat_l == cat_filter
                    if not (cid_match or name_match):
                        continue

                if not clean_q:
                    contains_hits.append(entry)
                    continue

                if clean_q == en_l or clean_q == hi_l or clean_q == kn_l:
                    exact_hits.append(entry)
                elif (
                    en_l.startswith(clean_q)
                    or hi_l.startswith(clean_q)
                    or kn_l.startswith(clean_q)
                    or f" {clean_q}" in en_l
                ):
                    prefix_hits.append(entry)
                elif clean_q in en_l or clean_q in hi_l or clean_q in kn_l:
                    contains_hits.append(entry)

            matched = exact_hits + prefix_hits + contains_hits
            total = len(matched)
            page = matched[safe_offset : safe_offset + safe_limit]

        return {
            "query": query.strip(),
            "category": category,
            "total": total,
            "limit": safe_limit,
            "offset": safe_offset,
            "entries": [self._decorate_entry(item) for item in page],
        }

    def get_entry(self, entry_id: str) -> dict[str, Any] | None:
        """Look up a single ISLRTC entry by numeric ID."""
        eid = str(entry_id).strip()
        with self._lock:
            self._load_unlocked()
            entry = self._entries_by_id.get(eid)
            return self._decorate_entry(entry) if entry is not None else None

    def resolve_entry_video(
        self,
        entry_id: str,
        *,
        timeout_seconds: float = 20.0,
    ) -> dict[str, Any]:
        """Return an ISLRTC entry with its official YouTube video ID resolved & cached."""
        eid = str(entry_id).strip()
        with self._lock:
            self._load_unlocked()
            entry = self._entries_by_id.get(eid)
            if entry is None:
                raise KeyError(f"ISLRTC entry {eid!r} not found.")
            if entry.get("youtube_id"):
                return self._decorate_entry(entry)
            portal_url = entry.get("portal_url") or (
                f"{ISLRTC_BASE_URL}/searchpage.php?type=list&id={eid}"
            )

        ssl_ctx = build_islrtc_ssl_context()
        with httpx.Client(verify=ssl_ctx, timeout=timeout_seconds, follow_redirects=True) as client:
            resp = client.get(portal_url)
            resp.raise_for_status()
            youtube_id = extract_youtube_embed_id(resp.text)

        with self._lock:
            payload = self._load_unlocked()
            entry = self._entries_by_id.get(eid)
            if entry is None:
                raise KeyError(f"ISLRTC entry {eid!r} not found.")
            if youtube_id and entry.get("youtube_id") != youtube_id:
                entry["youtube_id"] = youtube_id
                if self.dictionary_path.is_file():
                    _atomic_write_json(self.dictionary_path, payload)
            return self._decorate_entry(entry)


def main() -> None:
    """CLI entrypoint to harvest the official ISLRTC dictionary into `backend/data/`."""
    summary = harvest_islrtc_dictionary()
    print(
        f"ISLRTC dictionary harvested -> {summary['output_path']} "
        f"({summary['total_entries']} entries, HI={summary['with_hindi']}, KN={summary['with_kannada']})"
    )


if __name__ == "__main__":
    main()
