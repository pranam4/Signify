import { useEffect, useState } from "react";
import { Camera, Search, Sparkles } from "lucide-react";

export default function PosterReferencesBrowser({ onSelectForTraining }) {
  const [posters, setPosters] = useState([]);
  const [items, setItems] = useState([]);
  const [activePoster, setActivePoster] = useState("all");
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const handle = setTimeout(async () => {
      setLoading(true);
      try {
        const params = new URLSearchParams({
          poster_id: activePoster,
          q: query,
        });
        const res = await fetch(`/api/dataset/references?${params.toString()}`);
        if (!res.ok) return;
        const data = await res.json();
        if (!cancelled) {
          setPosters(data.posters || []);
          setItems(data.items || []);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }, 150);

    return () => {
      cancelled = true;
      clearTimeout(handle);
    };
  }, [activePoster, query]);

  return (
    <div className="poster-refs-shell">
      <div className="islrtc-top-bar">
        <div className="islrtc-search-wrap">
          <Search size={15} className="muted-icon" />
          <input
            type="text"
            className="dataset-text-input"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Filter 92 poster cards (e.g. ambulance, milk, toilet, thank you, पानी)..."
          />
        </div>
        <span className="status-pill neutral">
          Showing {items.length} visual sign cards
        </span>
      </div>

      <div className="islrtc-cat-pills">
        <button
          type="button"
          className={`chip-btn ${activePoster === "all" ? "selected" : ""}`}
          onClick={() => setActivePoster("all")}
        >
          All 5 Posters
        </button>
        {posters.map((p) => (
          <button
            key={p.id}
            type="button"
            className={`chip-btn ${activePoster === p.id ? "selected" : ""}`}
            onClick={() => setActivePoster(p.id)}
          >
            {p.title} ({p.count})
          </button>
        ))}
      </div>

      <div className="poster-cards-grid">
        {loading && items.length === 0 ? (
          <p className="empty-log">Loading poster reference cards...</p>
        ) : items.length === 0 ? (
          <p className="empty-log">No sign cards matched &ldquo;{query}&rdquo;.</p>
        ) : (
          items.map((card) => (
            <div key={card.id} className="poster-ref-card">
              <img
                src={card.image_url}
                alt={`${card.label} sign`}
                className="poster-ref-thumb"
                loading="lazy"
              />
              <div className="poster-ref-body">
                <div className="poster-ref-title-row">
                  <strong>{card.label}</strong>
                  {card.has_landmarks && (
                    <span className="trained-pill" title="MediaPipe landmarks trained">
                      <Sparkles size={10} /> Trained
                    </span>
                  )}
                </div>
                <small className="poster-ref-langs">
                  HI: {card.hi} · KN: {card.kn}
                </small>
                <small className="poster-ref-source">{card.category}</small>
                {onSelectForTraining && (
                  <button
                    type="button"
                    className="text-btn"
                    onClick={() => onSelectForTraining(card.label)}
                  >
                    <Camera size={12} /> Capture Webcam Reps
                  </button>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
