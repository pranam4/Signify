import { useCallback, useMemo, useRef, useState } from "react";
import {
  Check,
  Copy,
  FileText,
  Globe2,
  Hand,
  MessageSquareText,
  RotateCcw,
  Trash2,
  Volume2,
  VolumeX,
} from "lucide-react";
import CameraCaptureStage from "./components/CameraCaptureStage";
import SpeechToSignPanel from "./components/SpeechToSignPanel";
import signsData from "./data/signs.json";
import { speakText } from "./utils/speech";

export default function App() {
  const [mode, setMode] = useState("sign-to-speech");
  const [activeLanguage, setActiveLanguage] = useState("en");
  const [activeDialect, setActiveDialect] = useState(signsData.dialects[0].id);
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [latestOutput, setLatestOutput] = useState(null);
  const [history, setHistory] = useState([]);
  const [activeLoci, setActiveLoci] = useState({ left: null, right: null });
  const [voiceNotice, setVoiceNotice] = useState("");
  const [transcriptText, setTranscriptText] = useState("");
  const [autoAppendTranscript, setAutoAppendTranscript] = useState(true);
  const [gestureShortcuts, setGestureShortcuts] = useState(false);
  const [copiedTranscript, setCopiedTranscript] = useState(false);
  const transcriptUndoRef = useRef([]);

  const activeLangConfig = useMemo(
    () =>
      signsData.languages.find((l) => l.code === activeLanguage) ||
      signsData.languages[0],
    [activeLanguage]
  );

  const activeDialectConfig = useMemo(
    () =>
      signsData.dialects.find((d) => d.id === activeDialect) ||
      signsData.dialects[0],
    [activeDialect]
  );

  const triggerSpeech = useCallback(
    (text) => {
      if (!voiceEnabled || !text) return;
      const res = speakText(text, activeLangConfig.ttsLocale);
      if (res.spoken && !res.hasNativeVoice) {
        setVoiceNotice(
          `No native ${activeLangConfig.label} TTS voice found in this browser; using default voice.`
        );
      } else {
        setVoiceNotice("");
      }
    },
    [voiceEnabled, activeLangConfig]
  );

  const pushTranscriptSnapshot = useCallback((current) => {
    transcriptUndoRef.current = [...transcriptUndoRef.current, current].slice(-50);
  }, []);

  const appendToTranscript = useCallback(
    (segment) => {
      if (!segment) return;
      setTranscriptText((prev) => {
        pushTranscriptSnapshot(prev);
        const trimmed = prev.replace(/[ \t]+$/, "");
        if (!trimmed) return segment;
        const sep = trimmed.endsWith("\n") ? "" : " ";
        return `${trimmed}${sep}${segment}`;
      });
    },
    [pushTranscriptSnapshot]
  );

  const handleUndoTranscript = useCallback(() => {
    setTranscriptText((prev) => {
      if (transcriptUndoRef.current.length === 0) return prev;
      const nextStack = [...transcriptUndoRef.current];
      const previousValue = nextStack.pop();
      transcriptUndoRef.current = nextStack;
      return previousValue ?? "";
    });
  }, []);

  const handleCopyTranscript = useCallback(async () => {
    if (!transcriptText.trim()) return;
    try {
      await navigator.clipboard.writeText(transcriptText);
      setCopiedTranscript(true);
      setTimeout(() => setCopiedTranscript(false), 1500);
    } catch {
      // Ignore clipboard failures in restricted contexts
    }
  }, [transcriptText]);

  const handleClearTranscript = useCallback(() => {
    setTranscriptText((prev) => {
      if (prev) pushTranscriptSnapshot(prev);
      return "";
    });
  }, [pushTranscriptSnapshot]);

  const handleRecognizedEvent = useCallback(
    (payload) => {
      const text =
        payload.phrases?.[activeLanguage] ||
        payload.phrases?.en ||
        payload.locked_label;
      if (!text) return;

      // Update spatial loci when Pointing_Up occurs in Left or Right lane.
      if (payload.locked_label === "Pointing_Up") {
        if (payload.spatial_zone === "left") {
          setActiveLoci((prev) => ({ ...prev, left: "Person A" }));
        } else if (payload.spatial_zone === "right") {
          setActiveLoci((prev) => ({ ...prev, right: "Person B" }));
        }
      }

      const primaryHand =
        (payload.hands || []).find((h) => h.gesture === payload.locked_label) ||
        payload.hands?.[0];

      const entry = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
        label: payload.locked_label,
        text,
        phrases: payload.phrases || {},
        score: payload.locked_score || 0,
        zone: payload.spatial_zone || "center",
        handedness: primaryHand?.handedness || "Unknown",
        candidates: primaryHand?.top_gestures || [],
        time: new Date().toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
        }),
      };

      setLatestOutput(entry);
      setHistory((prev) => [entry, ...prev].slice(0, 20));

      if (gestureShortcuts && payload.locked_label === "Thumb_Down") {
        handleUndoTranscript();
        return;
      }
      if (gestureShortcuts && payload.locked_label === "Thumb_Up") {
        const toSpeak = transcriptText.trim() || text;
        triggerSpeech(toSpeak);
        return;
      }

      if (autoAppendTranscript) {
        appendToTranscript(text);
      }
      triggerSpeech(text);
    },
    [
      activeLanguage,
      appendToTranscript,
      autoAppendTranscript,
      gestureShortcuts,
      handleUndoTranscript,
      transcriptText,
      triggerSpeech,
    ]
  );

  const displayedLatestText = latestOutput
    ? latestOutput.phrases?.[activeLanguage] || latestOutput.text
    : "Show a sign in front of the camera";

  return (
    <div className="app-shell">
      <header className="top-bar">
        <div className="brand">
          <div className="brand-icon">
            <Hand size={20} />
          </div>
          <div>
            <h1>SignBridge · Prototype</h1>
            <p>Two-Way Camera &amp; Speech Translation Studio</p>
          </div>
        </div>

        <nav className="mode-tabs" aria-label="Translation Mode">
          <button
            type="button"
            className={`mode-tab ${mode === "sign-to-speech" ? "active" : ""}`}
            onClick={() => setMode("sign-to-speech")}
          >
            Sign → Text &amp; Speech
          </button>
          <button
            type="button"
            className={`mode-tab ${mode === "speech-to-sign" ? "active" : ""}`}
            onClick={() => setMode("speech-to-sign")}
          >
            Speech → Sign
          </button>
        </nav>

        <div className="top-controls">
          <label className="select-pill" title="Sign Dialect">
            <span>Dialect</span>
            <select
              value={activeDialect}
              onChange={(e) => setActiveDialect(e.target.value)}
            >
              {signsData.dialects.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.label}
                </option>
              ))}
            </select>
          </label>

          <label className="select-pill" title="Spoken Output Language">
            <Globe2 size={15} />
            <select
              value={activeLanguage}
              onChange={(e) => setActiveLanguage(e.target.value)}
            >
              {signsData.languages.map((lang) => (
                <option key={lang.code} value={lang.code}>
                  {lang.label}
                </option>
              ))}
            </select>
          </label>

          <button
            type="button"
            className={`icon-btn ${voiceEnabled ? "active" : ""}`}
            onClick={() => setVoiceEnabled((v) => !v)}
            title="Toggle Speech Output"
          >
            {voiceEnabled ? <Volume2 size={16} /> : <VolumeX size={16} />}
            <span>{voiceEnabled ? "Voice On" : "Muted"}</span>
          </button>
        </div>
      </header>

      <main className="workspace-grid">
        <div className="primary-stage">
          {mode === "sign-to-speech" ? (
            <CameraCaptureStage
              activeLanguage={activeLanguage}
              activeDialect={activeDialectConfig.label}
              onRecognizedEvent={handleRecognizedEvent}
              activeLoci={activeLoci}
            />
          ) : (
            <SpeechToSignPanel
              phrases={signsData.speechToSignPhrases}
              activeLanguage={activeLanguage}
              activeLangConfig={activeLangConfig}
            />
          )}
        </div>

        <aside className="side-panel">
          <section className="output-card">
            <div className="card-header">
              <span>Recognized Output ({activeLangConfig.label})</span>
              {latestOutput && (
                <button
                  type="button"
                  className="text-btn"
                  onClick={() => triggerSpeech(displayedLatestText)}
                >
                  <Volume2 size={14} /> Speak
                </button>
              )}
            </div>

            <div className="output-phrase">{displayedLatestText}</div>

            {latestOutput && (
              <>
                <div className="output-meta">
                  <span>Raw label: {latestOutput.label}</span>
                  <span>Hand: {latestOutput.handedness}</span>
                  <span>Zone: {latestOutput.zone.toUpperCase()}</span>
                  <span>{(latestOutput.score * 100).toFixed(0)}% conf</span>
                </div>
                {latestOutput.candidates?.length > 1 && (
                  <div className="candidate-pills">
                    {latestOutput.candidates.map((c) => (
                      <span key={c.category} className="candidate-pill">
                        {c.category} {(c.score * 100).toFixed(0)}%
                      </span>
                    ))}
                  </div>
                )}
              </>
            )}

            {voiceNotice && <p className="voice-note">{voiceNotice}</p>}
          </section>

          <section className="transcript-card">
            <div className="card-header">
              <span>
                <FileText size={14} /> Sign → Text Transcript
              </span>
              <div className="transcript-actions">
                <button
                  type="button"
                  className="text-btn"
                  onClick={handleUndoTranscript}
                  title="Undo last sign"
                >
                  <RotateCcw size={13} /> Undo
                </button>
                <button
                  type="button"
                  className="text-btn"
                  onClick={handleCopyTranscript}
                  title="Copy text"
                >
                  {copiedTranscript ? <Check size={13} /> : <Copy size={13} />}
                  {copiedTranscript ? "Copied" : "Copy"}
                </button>
                {transcriptText && (
                  <button
                    type="button"
                    className="text-btn"
                    onClick={handleClearTranscript}
                    title="Clear transcript"
                  >
                    <Trash2 size={13} /> Clear
                  </button>
                )}
              </div>
            </div>

            <textarea
              className="transcript-textarea"
              rows={3}
              value={transcriptText}
              onChange={(e) => setTranscriptText(e.target.value)}
              placeholder="Locked signs accumulate here as editable text..."
            />

            <div className="transcript-footer">
              <div className="transcript-toggles">
                <label className="check-pill">
                  <input
                    type="checkbox"
                    checked={autoAppendTranscript}
                    onChange={(e) => setAutoAppendTranscript(e.target.checked)}
                  />
                  <span>Auto-append</span>
                </label>
                <label
                  className="check-pill"
                  title="When enabled: Thumb_Up speaks transcript, Thumb_Down undoes last sign"
                >
                  <input
                    type="checkbox"
                    checked={gestureShortcuts}
                    onChange={(e) => setGestureShortcuts(e.target.checked)}
                  />
                  <span>Gesture shortcuts</span>
                </label>
              </div>
              <button
                type="button"
                className="text-btn"
                onClick={() => triggerSpeech(transcriptText.trim())}
                disabled={!transcriptText.trim()}
              >
                <Volume2 size={13} /> Speak Sentence
              </button>
            </div>
          </section>

          <section className="spatial-card">
            <div className="card-header">
              <span>Spatial Grammar Loci</span>
              <button
                type="button"
                className="text-btn"
                onClick={() => setActiveLoci({ left: null, right: null })}
              >
                Reset Loci
              </button>
            </div>
            <div className="loci-grid">
              <div className={`locus-box ${activeLoci.left ? "filled" : ""}`}>
                <small>LEFT ZONE</small>
                <strong>{activeLoci.left || "Unassigned"}</strong>
              </div>
              <div className={`locus-box ${activeLoci.right ? "filled" : ""}`}>
                <small>RIGHT ZONE</small>
                <strong>{activeLoci.right || "Unassigned"}</strong>
              </div>
            </div>
            <p className="spatial-hint">
              Point upward in the Left or Right lane to anchor Person A / Person B
              in signing space.
            </p>
          </section>

          <section className="history-card">
            <div className="card-header">
              <span>
                <MessageSquareText size={14} /> Translation Log
              </span>
              {history.length > 0 && (
                <button
                  type="button"
                  className="text-btn"
                  onClick={() => {
                    setHistory([]);
                    setLatestOutput(null);
                  }}
                >
                  <Trash2 size={13} /> Clear
                </button>
              )}
            </div>

            <div className="history-list">
              {history.length === 0 ? (
                <p className="empty-log">
                  Locked signs and spoken translations appear here.
                </p>
              ) : (
                history.map((item) => (
                  <div key={item.id} className="history-item">
                    <div>
                      <strong>
                        {item.phrases?.[activeLanguage] || item.text}
                      </strong>
                      <small>
                        {item.label} · {item.zone.toUpperCase()} ·{" "}
                        {(item.score * 100).toFixed(0)}%
                      </small>
                    </div>
                    <time>{item.time}</time>
                  </div>
                ))
              )}
            </div>
          </section>
        </aside>
      </main>
    </div>
  );
}
