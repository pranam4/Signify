"""SQLite database storage for ISLRTC dictionary signs, videos, poster cards, and dataset samples."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
import threading
from typing import Any

from .config import DEFAULT_DB_PATH

_MATCH_STOPWORDS: frozenset[str] = frozenset(
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
        "or",
        "do",
        "does",
        "did",
        "i",
        "me",
        "my",
        "you",
        "your",
        "it",
        "this",
        "that",
        "we",
        "they",
        "what",
        "is",
    }
)


class SignBridgeDatabase:
    """Thread-safe SQLite repository backing ISLRTC entries, tutorials, posters, and samples."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=30.0,
        )
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS islrtc_categories (
                    id TEXT PRIMARY KEY,
                    en TEXT NOT NULL,
                    hi TEXT NOT NULL DEFAULT '',
                    kn TEXT NOT NULL DEFAULT '',
                    entry_count INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS islrtc_signs (
                    id TEXT PRIMARY KEY,
                    category_id TEXT NOT NULL,
                    category_en TEXT NOT NULL,
                    en TEXT NOT NULL DEFAULT '',
                    en_lower TEXT NOT NULL DEFAULT '',
                    hi TEXT NOT NULL DEFAULT '',
                    kn TEXT NOT NULL DEFAULT '',
                    youtube_id TEXT,
                    portal_url TEXT NOT NULL,
                    updated_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_islrtc_en_lower ON islrtc_signs(en_lower);
                CREATE INDEX IF NOT EXISTS idx_islrtc_cat ON islrtc_signs(category_id);
                CREATE INDEX IF NOT EXISTS idx_islrtc_yt ON islrtc_signs(youtube_id);

                CREATE TABLE IF NOT EXISTS islrtc_tutorials (
                    youtube_id TEXT PRIMARY KEY,
                    en TEXT NOT NULL,
                    hi TEXT NOT NULL DEFAULT '',
                    embed_url TEXT NOT NULL,
                    watch_url TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS poster_references (
                    id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    poster_id TEXT NOT NULL,
                    poster_title TEXT NOT NULL,
                    category TEXT NOT NULL,
                    style TEXT NOT NULL,
                    en TEXT NOT NULL DEFAULT '',
                    hi TEXT NOT NULL DEFAULT '',
                    kn TEXT NOT NULL DEFAULT '',
                    image_url TEXT NOT NULL,
                    has_landmarks INTEGER NOT NULL DEFAULT 0,
                    num_hands INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS dataset_samples (
                    sample_id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    source TEXT NOT NULL,
                    needs_review INTEGER NOT NULL DEFAULT 0,
                    num_hands INTEGER NOT NULL DEFAULT 0,
                    has_pose INTEGER NOT NULL DEFAULT 0,
                    image_relpath TEXT NOT NULL,
                    created_at TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                """
            )
            self._conn.commit()

    def close(self) -> None:
        """Close the SQLite connection cleanly."""
        with self._lock:
            self._conn.close()

    def get_stats(self) -> dict[str, Any]:
        """Return record counts across all tables in the SignBridge SQLite database."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN hi != '' THEN 1 ELSE 0 END) AS with_hi,
                    SUM(CASE WHEN kn != '' THEN 1 ELSE 0 END) AS with_kn,
                    SUM(CASE WHEN youtube_id IS NOT NULL AND youtube_id != '' THEN 1 ELSE 0 END) AS with_yt
                FROM islrtc_signs
                """
            ).fetchone()
            tut_cnt = self._conn.execute(
                "SELECT COUNT(*) FROM islrtc_tutorials"
            ).fetchone()[0]
            ref_cnt = self._conn.execute(
                "SELECT COUNT(*) FROM poster_references"
            ).fetchone()[0]
            ds_cnt = self._conn.execute(
                "SELECT COUNT(*) FROM dataset_samples"
            ).fetchone()[0]

        return {
            "db_path": str(self.db_path),
            "islrtc_entries": int(row["total"] or 0),
            "islrtc_with_hindi": int(row["with_hi"] or 0),
            "islrtc_with_kannada": int(row["with_kn"] or 0),
            "islrtc_cached_videos": int(row["with_yt"] or 0),
            "islrtc_tutorials": int(tut_cnt or 0),
            "poster_cards": int(ref_cnt or 0),
            "dataset_samples": int(ds_cnt or 0),
        }

    def list_islrtc_categories(self) -> list[dict[str, Any]]:
        """Return ISLRTC categories sorted numerically."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, en, hi, kn, entry_count
                FROM islrtc_categories
                ORDER BY CAST(id AS INTEGER)
                """
            ).fetchall()
        return [
            {
                "id": r["id"],
                "en": r["en"],
                "hi": r["hi"],
                "kn": r["kn"],
                "count": int(r["entry_count"]),
            }
            for r in rows
        ]

    def list_islrtc_tutorials(self) -> list[dict[str, str]]:
        """Return official ISLRTC tutorial playlist records."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT youtube_id, en, hi, embed_url, watch_url FROM islrtc_tutorials"
            ).fetchall()
        return [dict(r) for r in rows]

    def upsert_islrtc_bulk(
        self,
        categories: list[dict[str, Any]],
        tutorials: list[dict[str, Any]],
        entries: list[dict[str, Any]],
    ) -> None:
        """Insert or update ISLRTC categories, tutorials, and signs while preserving cached YouTube IDs."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._conn:
                self._conn.executemany(
                    """
                    INSERT INTO islrtc_categories (id, en, hi, kn, entry_count)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        en = excluded.en,
                        hi = excluded.hi,
                        kn = excluded.kn,
                        entry_count = excluded.entry_count
                    """,
                    [
                        (
                            str(c["id"]),
                            str(c.get("en") or ""),
                            str(c.get("hi") or ""),
                            str(c.get("kn") or ""),
                            int(c.get("count") or 0),
                        )
                        for c in categories
                    ],
                )
                self._conn.executemany(
                    """
                    INSERT INTO islrtc_tutorials (youtube_id, en, hi, embed_url, watch_url)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(youtube_id) DO UPDATE SET
                        en = excluded.en,
                        hi = excluded.hi,
                        embed_url = excluded.embed_url,
                        watch_url = excluded.watch_url
                    """,
                    [
                        (
                            str(t["youtube_id"]),
                            str(t.get("en") or ""),
                            str(t.get("hi") or ""),
                            str(t.get("embed_url") or ""),
                            str(t.get("watch_url") or ""),
                        )
                        for t in tutorials
                    ],
                )
                self._conn.executemany(
                    """
                    INSERT INTO islrtc_signs (
                        id, category_id, category_en, en, en_lower, hi, kn, youtube_id, portal_url, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        category_id = excluded.category_id,
                        category_en = excluded.category_en,
                        en = excluded.en,
                        en_lower = excluded.en_lower,
                        hi = excluded.hi,
                        kn = excluded.kn,
                        youtube_id = COALESCE(NULLIF(excluded.youtube_id, ''), islrtc_signs.youtube_id),
                        portal_url = excluded.portal_url,
                        updated_at = excluded.updated_at
                    """,
                    [
                        (
                            str(e["id"]),
                            str(e.get("category_id") or ""),
                            str(e.get("category_en") or ""),
                            str(e.get("en") or ""),
                            str(e.get("en") or "").lower(),
                            str(e.get("hi") or ""),
                            str(e.get("kn") or ""),
                            e.get("youtube_id"),
                            str(e.get("portal_url") or ""),
                            now,
                        )
                        for e in entries
                    ],
                )

    def get_islrtc_entry(self, entry_id: str) -> dict[str, Any] | None:
        """Fetch a single ISLRTC sign entry row by ID."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT id, category_id, category_en, en, hi, kn, youtube_id, portal_url
                FROM islrtc_signs
                WHERE id = ?
                """,
                (str(entry_id).strip(),),
            ).fetchone()
        return dict(row) if row is not None else None

    def set_islrtc_youtube_id(self, entry_id: str, youtube_id: str | None) -> None:
        """Persist a resolved ISLRTC YouTube video ID in SQLite."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "UPDATE islrtc_signs SET youtube_id = ?, updated_at = ? WHERE id = ?",
                    (youtube_id, now, str(entry_id).strip()),
                )

    def search_islrtc(
        self,
        *,
        query: str = "",
        category: str = "all",
        limit: int = 36,
        offset: int = 0,
    ) -> tuple[int, list[dict[str, Any]]]:
        """Search SQLite `islrtc_signs` across EN/HI/KN with exact/prefix ranking."""
        clean_q = " ".join((query or "").strip().lower().split())
        cat_filter = (category or "all").strip().lower()
        safe_limit = max(1, min(100, int(limit)))
        safe_offset = max(0, int(offset))

        where_clauses: list[str] = []
        params: list[Any] = []
        if cat_filter not in ("", "all"):
            where_clauses.append("(LOWER(category_id) = ? OR LOWER(category_en) = ?)")
            params.extend([cat_filter, cat_filter])

        if clean_q:
            like_q = f"%{clean_q}%"
            where_clauses.append("(en_lower LIKE ? OR hi LIKE ? OR kn LIKE ?)")
            params.extend([like_q, like_q, like_q])

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        with self._lock:
            total = int(
                self._conn.execute(
                    f"SELECT COUNT(*) FROM islrtc_signs {where_sql}",
                    params,
                ).fetchone()[0]
            )
            if clean_q:
                order_sql = """
                ORDER BY
                    CASE
                        WHEN en_lower = ? OR LOWER(hi) = ? OR LOWER(kn) = ? THEN 0
                        WHEN en_lower LIKE ? OR en_lower LIKE ? OR hi LIKE ? OR kn LIKE ? THEN 1
                        ELSE 2
                    END,
                    CASE WHEN youtube_id IS NOT NULL AND youtube_id != '' THEN 0 ELSE 1 END,
                    LENGTH(en),
                    en_lower
                """
                order_params: list[Any] = [
                    clean_q,
                    clean_q,
                    clean_q,
                    f"{clean_q}%",
                    f"% {clean_q}%",
                    f"{clean_q}%",
                    f"{clean_q}%",
                ]
            else:
                order_sql = """
                ORDER BY
                    CASE WHEN youtube_id IS NOT NULL AND youtube_id != '' THEN 0 ELSE 1 END,
                    CAST(category_id AS INTEGER),
                    en_lower
                """
                order_params = []

            rows = self._conn.execute(
                f"""
                SELECT id, category_id, category_en, en, hi, kn, youtube_id, portal_url
                FROM islrtc_signs
                {where_sql}
                {order_sql}
                LIMIT ? OFFSET ?
                """,
                [*params, *order_params, safe_limit, safe_offset],
            ).fetchall()

        return total, [dict(r) for r in rows]

    def match_text_islrtc(self, text: str, *, limit: int = 6) -> list[dict[str, Any]]:
        """Find best matching ISLRTC database entries for tokens/phrases in spoken or typed text."""
        cleaned = " ".join((text or "").strip().split())
        if not cleaned:
            return []

        tokens = [
            tok.lower()
            for tok in re.findall(r"[^\s,.;:!?\"'()\[\]{}]+", cleaned)
            if tok and tok.lower() not in _MATCH_STOPWORDS
        ]
        if not tokens:
            return []

        candidates: list[str] = []
        # Full phrase first, then bigrams, then individual content tokens.
        full_clean = cleaned.lower().rstrip("?!., ")
        if full_clean:
            candidates.append(full_clean)
        for idx in range(len(tokens) - 1):
            candidates.append(f"{tokens[idx]} {tokens[idx + 1]}")
        for tok in tokens:
            if tok not in candidates:
                candidates.append(tok)

        seen_ids: set[str] = set()
        matches: list[dict[str, Any]] = []

        with self._lock:
            for cand in candidates:
                if len(matches) >= limit:
                    break
                rows = self._conn.execute(
                    """
                    SELECT id, category_id, category_en, en, hi, kn, youtube_id, portal_url
                    FROM islrtc_signs
                    WHERE en_lower = ? OR hi = ? OR kn = ? OR en_lower LIKE ?
                    ORDER BY
                        CASE WHEN en_lower = ? OR hi = ? OR kn = ? THEN 0 ELSE 1 END,
                        CASE WHEN youtube_id IS NOT NULL AND youtube_id != '' THEN 0 ELSE 1 END,
                        CASE WHEN category_id IN ('1', '8', '9') THEN 0 ELSE 1 END,
                        LENGTH(en)
                    LIMIT 2
                    """,
                    (cand, cand, cand, f"{cand} (%", cand, cand, cand),
                ).fetchall()
                for row in rows:
                    if row["id"] in seen_ids:
                        continue
                    seen_ids.add(row["id"])
                    matches.append(dict(row))
                    if len(matches) >= limit:
                        break

        return matches

    def upsert_poster_references(self, items: list[dict[str, Any]]) -> None:
        """Replace or update poster reference cards inside SQLite."""
        with self._lock:
            with self._conn:
                self._conn.executemany(
                    """
                    INSERT INTO poster_references (
                        id, label, poster_id, poster_title, category, style,
                        en, hi, kn, image_url, has_landmarks, num_hands
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        label = excluded.label,
                        poster_id = excluded.poster_id,
                        poster_title = excluded.poster_title,
                        category = excluded.category,
                        style = excluded.style,
                        en = excluded.en,
                        hi = excluded.hi,
                        kn = excluded.kn,
                        image_url = excluded.image_url,
                        has_landmarks = excluded.has_landmarks,
                        num_hands = excluded.num_hands
                    """,
                    [
                        (
                            str(item["id"]),
                            str(item["label"]),
                            str(item["poster_id"]),
                            str(item["poster_title"]),
                            str(item["category"]),
                            str(item["style"]),
                            str(item.get("en") or ""),
                            str(item.get("hi") or ""),
                            str(item.get("kn") or ""),
                            str(item["image_url"]),
                            1 if item.get("has_landmarks") else 0,
                            int(item.get("num_hands") or 0),
                        )
                        for item in items
                    ],
                )

    def query_poster_references(
        self,
        *,
        poster_id: str = "all",
        query: str = "",
    ) -> list[dict[str, Any]]:
        """Query poster reference cards from SQLite with optional poster and keyword filter."""
        clean_q = " ".join((query or "").strip().lower().split())
        pid = (poster_id or "all").strip().lower()
        where_clauses: list[str] = []
        params: list[Any] = []
        if pid not in ("", "all"):
            where_clauses.append("LOWER(poster_id) = ?")
            params.append(pid)
        if clean_q:
            like = f"%{clean_q}%"
            where_clauses.append(
                "(LOWER(label) LIKE ? OR LOWER(en) LIKE ? OR LOWER(hi) LIKE ? OR LOWER(kn) LIKE ? OR LOWER(category) LIKE ?)"
            )
            params.extend([like, like, like, like, like])
        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT id, label, poster_id, poster_title, category, style,
                       en, hi, kn, image_url, has_landmarks, num_hands
                FROM poster_references
                {where_sql}
                ORDER BY poster_id, id
                """,
                params,
            ).fetchall()

        return [
            {
                **dict(r),
                "has_landmarks": bool(r["has_landmarks"]),
                "num_hands": int(r["num_hands"]),
            }
            for r in rows
        ]

    def sync_dataset_samples(self, samples: list[dict[str, Any]]) -> None:
        """Synchronize landmark sample metadata rows into SQLite `dataset_samples`."""
        with self._lock:
            with self._conn:
                self._conn.execute("DELETE FROM dataset_samples")
                self._conn.executemany(
                    """
                    INSERT INTO dataset_samples (
                        sample_id, label, source, needs_review, num_hands, has_pose,
                        image_relpath, created_at, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            str(s["sample_id"]),
                            str(s["label"]),
                            str(s.get("source") or "webcam"),
                            1 if s.get("needs_review") else 0,
                            int(s.get("num_hands") or 0),
                            1 if s.get("has_pose") else 0,
                            str(s.get("image_relpath") or ""),
                            s.get("created_at"),
                            json.dumps(s.get("metadata") or {}, ensure_ascii=False),
                        )
                        for s in samples
                    ],
                )

    def upsert_dataset_sample(self, entry: dict[str, Any]) -> None:
        """Insert or update a single sample row in `dataset_samples`."""
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO dataset_samples (
                        sample_id, label, source, needs_review, num_hands, has_pose,
                        image_relpath, created_at, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(sample_id) DO UPDATE SET
                        label = excluded.label,
                        source = excluded.source,
                        needs_review = excluded.needs_review,
                        num_hands = excluded.num_hands,
                        has_pose = excluded.has_pose,
                        image_relpath = excluded.image_relpath,
                        created_at = excluded.created_at,
                        metadata_json = excluded.metadata_json
                    """,
                    (
                        str(entry["sample_id"]),
                        str(entry["label"]),
                        str(entry.get("source") or "webcam"),
                        1 if entry.get("needs_review") else 0,
                        int(entry.get("num_hands") or 0),
                        1 if entry.get("has_pose") else 0,
                        str(entry.get("image_relpath") or ""),
                        entry.get("created_at"),
                        json.dumps(entry.get("metadata") or {}, ensure_ascii=False),
                    ),
                )

    def delete_dataset_sample(self, sample_id: str) -> None:
        """Remove a sample row from `dataset_samples`."""
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "DELETE FROM dataset_samples WHERE sample_id = ?",
                    (sample_id,),
                )
