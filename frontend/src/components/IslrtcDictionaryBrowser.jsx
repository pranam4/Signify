import { useCallback, useEffect, useState } from "react";
import {
  BookOpen,
  ExternalLink,
  FolderOpen,
  Play,
  Search,
  Video,
} from "lucide-react";

export default function IslrtcDictionaryBrowser({ compact = false, initialQuery = "" }) {
  const [summary, setSummary] = useState(null);
  const [query, setQuery] = useState(initialQuery);
  const [activeCategory, setActiveCategory] = useState("all");
  const [results, setResults] = useState([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(false);
  const [activeEntry, setActiveEntry] = useState(null);
  const [resolvingId, setResolvingId] = useState("");
  const [errorMsg, setErrorMsg] = useState("");
  const limit = compact ? 12 : 36;

  useEffect(() => {
    setQuery(initialQuery || "");
    setOffset(0);
  }, [initialQuery]);

  useEffect(() => {
    let cancelled = false;
    fetch("/api/islrtc/summary")
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (!cancelled && data) setSummary(data);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  const runSearch = useCallback(
    async (qText, catId, pageOffset) => {
      setLoading(true);
      setErrorMsg("");
      try {
        const params = new URLSearchParams({
          q: qText || "",
          category: catId || "all",
          limit: String(limit),
          offset: String(pageOffset),
        });
        const res = await fetch(`/api/islrtc/dictionary?${params.toString()}`);
        if (!res.ok) throw new Error("Could not load ISLRTC dictionary.");
        const data = await res.json();
        setResults(data.entries || []);
        setTotal(data.total || 0);
      } catch (err) {
        setErrorMsg(err?.message || "Failed to search ISLRTC dictionary.");
      } finally {
        setLoading(false);
      }
    },
    [limit]
  );

  useEffect(() => {
    const handle = setTimeout(() => {
      void runSearch(query, activeCategory, offset);
    }, 180);
    return () => clearTimeout(handle);
  }, [query, activeCategory, offset, runSearch]);

  const handleWatchVideo = async (item) => {
    setResolvingId(item.id);
    setErrorMsg("");
    try {
      const res = await fetch(`/api/islrtc/entry/${encodeURIComponent(item.id)}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || "Could not load sign video.");
      setActiveEntry(data.entry || item);
      if (data.video_error) {
        setErrorMsg(
          `Video embed lookup issue (${data.video_error}) — use Official Portal link below.`
        );
      }
    } catch (err) {
      setErrorMsg(err?.message || "Could not resolve ISLRTC YouTube video.");
      setActiveEntry(item);
    } finally {
      setResolvingId("");
    }
  };

  const pageEnd = Math.min(total, offset + results.length);

  return (
    <div className="islrtc-browser-shell">
      <div className="islrtc-top-bar">
        <div className="islrtc-search-wrap">
          <Search size={15} className="muted-icon" />
          <input
            type="text"
            className="dataset-text-input"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setOffset(0);
            }}
            placeholder="Search 15,602 ISLRTC signs in English, हिंदी, or ಕನ್ನಡ (e.g. water, नमस्ते, ambulance)..."
          />
        </div>

        {!compact && summary && (
          <div className="islrtc-meta-links">
            <span className="status-pill neutral">
              {summary.total_entries?.toLocaleString?.() || summary.total_entries} Signs · HI:{" "}
              {summary.with_hindi?.toLocaleString?.()} · KN:{" "}
              {summary.with_kannada?.toLocaleString?.()}
            </span>
            {summary.drive_url && (
              <a
                href={summary.drive_url}
                target="_blank"
                rel="noreferrer"
                className="icon-btn"
              >
                <FolderOpen size={14} />
                <span>ISLRTC Archive</span>
                <ExternalLink size={12} />
              </a>
            )}
            {summary.source_url && (
              <a
                href={summary.source_url}
                target="_blank"
                rel="noreferrer"
                className="icon-btn"
              >
                <BookOpen size={14} />
                <span>ISLRTC Portal</span>
                <ExternalLink size={12} />
              </a>
            )}
          </div>
        )}
      </div>

      {!compact && summary?.categories?.length > 0 && (
        <div className="islrtc-cat-pills">
          <button
            type="button"
            className={`chip-btn ${activeCategory === "all" ? "selected" : ""}`}
            onClick={() => {
              setActiveCategory("all");
              setOffset(0);
            }}
          >
            All Categories ({summary.total_entries})
          </button>
          {summary.categories.map((cat) => (
            <button
              key={cat.id}
              type="button"
              className={`chip-btn ${activeCategory === cat.id ? "selected" : ""}`}
              onClick={() => {
                setActiveCategory(cat.id);
                setOffset(0);
              }}
            >
              {cat.en} ({cat.count})
            </button>
          ))}
        </div>
      )}

      {errorMsg && <div className="inline-warn">{errorMsg}</div>}

      {activeEntry && (
        <div className="islrtc-player-card">
          <div className="card-header">
            <span>
              <Video size={14} /> {activeEntry.en || activeEntry.label}
              {activeEntry.hi ? ` · ${activeEntry.hi}` : ""}
              {activeEntry.kn ? ` · ${activeEntry.kn}` : ""}
            </span>
            <div className="transcript-actions">
              {activeEntry.portal_url && (
                <a
                  href={activeEntry.portal_url}
                  target="_blank"
                  rel="noreferrer"
                  className="text-btn"
                >
                  Open ISLRTC Page <ExternalLink size={12} />
                </a>
              )}
              <button
                type="button"
                className="text-btn"
                onClick={() => setActiveEntry(null)}
              >
                Close Video
              </button>
            </div>
          </div>

          {activeEntry.embed_url ? (
            <div className="islrtc-iframe-wrap">
              <iframe
                key={activeEntry.embed_url}
                src={activeEntry.embed_url}
                title={`ISLRTC Sign: ${activeEntry.en || activeEntry.id}`}
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
              />
            </div>
          ) : (
            <p className="empty-log">
              Video embed not resolved for this entry. Use &ldquo;Open ISLRTC Page&rdquo; above.
            </p>
          )}
        </div>
      )}

      <div className="islrtc-results-grid">
        {loading && results.length === 0 ? (
          <p className="empty-log">Searching official ISLRTC dictionary...</p>
        ) : results.length === 0 ? (
          <p className="empty-log">No ISLRTC entries matched &ldquo;{query}&rdquo;.</p>
        ) : (
          results.map((item) => (
            <div key={item.id} className="islrtc-entry-card">
              <div className="islrtc-entry-top">
                <span className="gloss-badge">{item.category_en}</span>
                {item.youtube_id && <span className="cached-dot">Cached Video</span>}
              </div>
              <strong className="islrtc-entry-title">{item.en || "—"}</strong>
              <div className="islrtc-entry-langs">
                {item.hi && <span>HI: {item.hi}</span>}
                {item.kn && <span>KN: {item.kn}</span>}
              </div>
              <div className="dataset-sample-actions">
                <button
                  type="button"
                  className="icon-btn"
                  disabled={resolvingId === item.id}
                  onClick={() => handleWatchVideo(item)}
                >
                  <Play size={13} />
                  <span>
                    {resolvingId === item.id ? "Loading..." : "Watch Sign"}
                  </span>
                </button>
                <a
                  href={item.portal_url}
                  target="_blank"
                  rel="noreferrer"
                  className="text-btn"
                  title="View on ISLRTC portal"
                >
                  <ExternalLink size={12} />
                </a>
              </div>
            </div>
          ))
        )}
      </div>

      {total > limit && (
        <div className="islrtc-pagination">
          <button
            type="button"
            className="icon-btn"
            disabled={offset === 0}
            onClick={() => setOffset((o) => Math.max(0, o - limit))}
          >
            Previous
          </button>
          <span>
            Showing {total === 0 ? 0 : offset + 1}–{pageEnd} of{" "}
            {total.toLocaleString()} signs
          </span>
          <button
            type="button"
            className="icon-btn"
            disabled={offset + limit >= total}
            onClick={() => setOffset((o) => o + limit)}
          >
            Next
          </button>
        </div>
      )}

      {!compact && summary?.tutorials?.length > 0 && (
        <div className="islrtc-tutorials-section">
          <div className="card-header">
            <span>
              <Video size={14} /> Official ISLRTC Topic Playlists (DEPwD)
            </span>
          </div>
          <div className="islrtc-tutorial-grid">
            {summary.tutorials.map((tut) => (
              <button
                key={tut.youtube_id}
                type="button"
                className="islrtc-tutorial-card"
                onClick={() =>
                  setActiveEntry({
                    id: tut.youtube_id,
                    en: tut.en,
                    hi: tut.hi,
                    embed_url: tut.embed_url,
                    portal_url: tut.watch_url,
                  })
                }
              >
                <Play size={15} />
                <div>
                  <strong>{tut.en}</strong>
                  <small>{tut.hi}</small>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
