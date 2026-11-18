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
  ArrowUpRight,
  AudioLines,
  BookOpen,
  ChevronRight,
  Sparkles,
  Heart,
} from "lucide-react";
import CameraCaptureStage from "./components/CameraCaptureStage";
import SignDatasetPanel from "./components/SignDatasetPanel";
import SpeechToSignPanel from "./components/SpeechToSignPanel";
import PromoVideo from "./components/PromoVideo";
import signsData from "./data/signs.json";
import { speakText } from "./utils/speech";

function formatNmmInflection(text, nmm, lang) {
  if (!text || !nmm) return text;
  if (nmm.sentence_type === "negation") {
    const prefix = { en: "Not: ", hi: "नहीं: ", kn: "ಇಲ್ಲ: " }[lang] || "Not: ";
    return text.toLowerCase().startsWith("no") ? text : `${prefix}${text}`;
  }
  if (nmm.sentence_type === "yes_no_question" || nmm.sentence_type === "wh_question") {
    return text.endsWith("?") ? text : `${text}?`;
  }
  return text;
}

function formatBrowLabel(browState) {
  if (browState === "raised") return "Raised (Y/N ?)";
  if (browState === "furrowed") return "Furrowed (WH ?)";
  return "Neutral";
}

function formatSentenceLabel(sentenceType) {
  return (
    {
      statement: "Statement",
      yes_no_question: "Yes/No Question",
      wh_question: "WH-Question",
      negation: "Negation",
      affirmation: "Affirmation",
    }[sentenceType] || "Statement"
  );
}

export default function App() {
  const [mode, setMode] = useState("sign-to-speech");
  const [showGuide, setShowGuide] = useState(false);
  const [activeLanguage, setActiveLanguage] = useState("en");
  const [activeDialect, setActiveDialect] = useState(signsData.dialects[0].id);
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [latestOutput, setLatestOutput] = useState(null);
  const [livePreview, setLivePreview] = useState(null);
  const [history, setHistory] = useState([]);
  const [activeLoci, setActiveLoci] = useState({ left: null, right: null });
  const [spatialGrammar, setSpatialGrammar] = useState(null);
  const [lastDirectionalEvent, setLastDirectionalEvent] = useState(null);
  const [applyNmmGrammar, setApplyNmmGrammar] = useState(true);
  const [voiceNotice, setVoiceNotice] = useState("");
  const [transcriptText, setTranscriptText] = useState("");
  const [autoAppendTranscript, setAutoAppendTranscript] = useState(true);
  const [gestureShortcuts, setGestureShortcuts] = useState(false);
  const [copiedTranscript, setCopiedTranscript] = useState(false);
  const transcriptUndoRef = useRef([]);
  const lastSpatialUpdateRef = useRef(0);
  const lastHandSeenAtRef = useRef(0);

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

  const handleSpatialUpdate = useCallback((sg, _holistic, liveInfo) => {
    const now = performance.now();
    const handCount = liveInfo?.handCount ?? 0;
    const hasActiveSignal = handCount > 0 || Boolean(liveInfo?.candidateLabel);

    if (hasActiveSignal) {
      lastHandSeenAtRef.current = now;
    } else {
      setLivePreview(null);
      if (
        lastHandSeenAtRef.current > 0 &&
        now - lastHandSeenAtRef.current >= 1800
      ) {
        setLatestOutput(null);
        lastHandSeenAtRef.current = 0;
      }
    }

    if (now - lastSpatialUpdateRef.current < 120) return;
    lastSpatialUpdateRef.current = now;
    setSpatialGrammar(sg);
    if (liveInfo !== undefined) {
      setLivePreview(hasActiveSignal ? liveInfo : null);
    }
  }, []);

  const handleDirectionalEvent = useCallback(
    (dirEvent, sg) => {
      if (!dirEvent) return;
      const isCrossLocus =
        dirEvent.action === "GIVE_A_TO_B" || dirEvent.action === "GIVE_B_TO_A";
      const hasAnchoredLoci = Boolean(activeLoci.left || activeLoci.right);
      if (!isCrossLocus && !hasAnchoredLoci) return;

      setLastDirectionalEvent(dirEvent);
      const nmm = sg?.non_manual_markers;
      const rawText = dirEvent.phrases?.[activeLanguage] || dirEvent.phrases?.en || dirEvent.action;
      const text = applyNmmGrammar
        ? formatNmmInflection(rawText, nmm, activeLanguage)
        : rawText;

      const inflectedPhrases = { ...(dirEvent.phrases || {}) };
      if (applyNmmGrammar && nmm) {
        for (const code of ["en", "hi", "kn"]) {
          if (inflectedPhrases[code]) {
            inflectedPhrases[code] = formatNmmInflection(
              inflectedPhrases[code],
              nmm,
              code
            );
          }
        }
      }

      const entry = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
        label: dirEvent.action,
        text,
        phrases: inflectedPhrases,
        score: 0.9,
        zone: `${dirEvent.from_zone}→${dirEvent.to_zone}`,
        handedness: dirEvent.handedness || "Unknown",
        sentenceType: nmm?.sentence_type || "statement",
        candidates: [],
        time: new Date().toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
        }),
      };

      setLatestOutput(entry);
      setHistory((prev) => [entry, ...prev].slice(0, 20));
      if (autoAppendTranscript) {
        appendToTranscript(text);
      }
      triggerSpeech(text);
    },
    [
      activeLanguage,
      activeLoci.left,
      activeLoci.right,
      appendToTranscript,
      applyNmmGrammar,
      autoAppendTranscript,
      triggerSpeech,
    ]
  );

  const handleRecognizedEvent = useCallback(
    (payload) => {
      const rawText =
        payload.phrases?.[activeLanguage] ||
        payload.phrases?.en ||
        payload.locked_label;
      if (!rawText) return;

      // Update spatial loci when Pointing_Up occurs in Left or Right lane.
      if (payload.locked_label === "Pointing_Up") {
        if (payload.spatial_zone === "left") {
          setActiveLoci((prev) => ({ ...prev, left: "Person A" }));
        } else if (payload.spatial_zone === "right") {
          setActiveLoci((prev) => ({ ...prev, right: "Person B" }));
        }
      }

      const nmm = payload.spatial_grammar?.non_manual_markers;
      const shouldInflect =
        applyNmmGrammar &&
        payload.locked_label !== "Pointing_Up" &&
        !(gestureShortcuts && (payload.locked_label === "Thumb_Up" || payload.locked_label === "Thumb_Down"));

      const text = shouldInflect
        ? formatNmmInflection(rawText, nmm, activeLanguage)
        : rawText;

      const inflectedPhrases = { ...(payload.phrases || {}) };
      if (shouldInflect && nmm) {
        for (const code of ["en", "hi", "kn"]) {
          if (inflectedPhrases[code]) {
            inflectedPhrases[code] = formatNmmInflection(
              inflectedPhrases[code],
              nmm,
              code
            );
          }
        }
      }

      const primaryHand =
        (payload.hands || []).find((h) => h.gesture === payload.locked_label) ||
        payload.hands?.[0];

      const entry = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
        label: payload.locked_label,
        text,
        phrases: inflectedPhrases,
        score: payload.locked_score || 0,
        zone: payload.spatial_zone || "center",
        heightLevel: primaryHand?.height_level || "chest",
        handedness: primaryHand?.handedness || "Unknown",
        recognitionSource: payload.recognition_source || "static_pose",
        sentenceType: nmm?.sentence_type || "statement",
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
      applyNmmGrammar,
      autoAppendTranscript,
      gestureShortcuts,
      handleUndoTranscript,
      transcriptText,
      triggerSpeech,
    ]
  );

  const isAwaitingLiveSign = !latestOutput && !livePreview?.candidateLabel;
  const displayedLatestText = latestOutput
    ? latestOutput.phrases?.[activeLanguage] || latestOutput.text
    : livePreview?.candidateLabel || "Show a sign in front of the camera";

  const space = spatialGrammar?.signing_space;
  const nmm = spatialGrammar?.non_manual_markers;
  const referentFocus = spatialGrammar?.referent_focus || "center";

  return (
    <div className={`app-shell mode-${mode}`}>
      <a className="skip-link" href="#workspace">Skip to workspace</a>
      <aside className="app-rail">
        <a className="rail-brand" href="#workspace" aria-label="Signify workspace"><span className="brand-icon"><Hand size={24} /></span><span>Signify<span className="brand-period">.</span><small>A little closer, together.</small></span></a>
        <span className="rail-label">YOUR WORKSPACE</span>
        <nav className="rail-nav" aria-label="Translation mode">
          {[
            { id: "sign-to-speech", icon: Hand, title: "Sign to speech", detail: "Let your hands speak" },
            { id: "speech-to-sign", icon: AudioLines, title: "Speech to sign", detail: "Say it with Ray" },
            { id: "dataset", icon: BookOpen, title: "Sign library", detail: "Explore, capture & train" },
          ].map(({ id, icon: Icon, title, detail }) => <button type="button" key={id} className={`rail-link ${mode === id ? "active" : ""}`} aria-current={mode === id ? "page" : undefined} onClick={() => setMode(id)}><Icon size={21} /><span>{title}<small>{detail}</small></span><ChevronRight size={15} /></button>)}
        </nav>
        <div className="rail-ray-card"><div className="ray-monogram">R<span /></div><span className="rail-label">MEET YOUR SIGNING COMPANION</span><h3>A little help from Ray.</h3><p>Turn everyday words into a more connected conversation.</p><button type="button" onClick={() => setMode("speech-to-sign")}>Say hello to Ray <ArrowUpRight size={16} /></button></div>
        <div className="rail-bottom"><Heart size={15} /><span>Made for connection.<small>Designed for everyone.</small></span></div>
      </aside>
      <div className="app-content">
      <header className="top-bar">
        <div className="brand">
          <div className="brand-icon">
            <Hand size={20} />
          </div>
          <div>
            <span className="breadcrumb">Workspace <ChevronRight size={13} /> <strong>{mode === "dataset" ? "Sign library" : "Translation studio"}</strong></span>
          </div>
        </div>

        <nav className="mode-tabs legacy-tabs" aria-label="Translation Mode">
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
            Voice → Text &amp; Sign
          </button>
          <button
            type="button"
            className={`mode-tab ${mode === "dataset" ? "active" : ""}`}
            onClick={() => setMode("dataset")}
          >
            Dataset &amp; Trainer
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
              aria-label="Output language"
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
            aria-pressed={voiceEnabled}
          >
            {voiceEnabled ? <Volume2 size={16} /> : <VolumeX size={16} />}
            <span>{voiceEnabled ? "Voice On" : "Muted"}</span>
          </button>

          <PromoVideo />
        </div>
      </header>
      <section className="workspace-intro">
        <div><span className="eyebrow"><span /> EVERY CONVERSATION MATTERS</span><h1>{mode === "sign-to-speech" ? <>Your hands. <em>Your voice.</em></> : mode === "speech-to-sign" ? <>Your words. <em>Ray’s hands.</em></> : <>Small signs. <em>Endless possibilities.</em></>}</h1><p>{mode === "sign-to-speech" ? "Express yourself naturally. Turn your signs into words that connect." : mode === "speech-to-sign" ? "Speak, type, or choose a phrase. Ray brings your words to life." : "Build your vocabulary, explore signs, and make recognition your own."}</p></div>
        <div className="intro-actions"><button type="button" className="guide-button" aria-expanded={showGuide} onClick={() => setShowGuide(v => !v)}><Sparkles size={16} /> Quick guide <ChevronRight size={15} /></button></div>
      </section>
      {showGuide && <div className="quick-guide">{(mode === "sign-to-speech" ? ["Allow camera access and keep your hands in frame.", "Hold a sign steady until the progress bar fills.", "Edit, copy, or listen to your translated words."] : mode === "speech-to-sign" ? ["Record your voice, type a sentence, or choose a phrase.", "Watch Ray sign. Slow down or replay any movement.", "Try the front or hands view for a closer look."] : ["Explore reference posters or choose a sign to train.", "Capture webcam samples or import your own images.", "Review labels, then train your custom model."]).map((tip, i) => <div key={tip}><span>0{i + 1}</span><p>{tip}</p></div>)}</div>}
      <main id="workspace" className="workspace-grid">
        <div className="primary-stage">
          {mode === "sign-to-speech" ? (
            <CameraCaptureStage
              activeLanguage={activeLanguage}
              activeDialect={activeDialectConfig.label}
              onRecognizedEvent={handleRecognizedEvent}
              onDirectionalEvent={handleDirectionalEvent}
              onSpatialUpdate={handleSpatialUpdate}
              activeLoci={activeLoci}
            />
          ) : mode === "speech-to-sign" ? (
            <SpeechToSignPanel
              phrases={signsData.speechToSignPhrases}
              activeLanguage={activeLanguage}
              activeLangConfig={activeLangConfig}
              activeDialect={activeDialect}
            />
          ) : (
            <SignDatasetPanel />
          )}
        </div>

        {mode === "sign-to-speech" && <aside className="side-panel">
          <section className="output-card">
            <div className="card-header">
              <span><Sparkles size={15} /> Live translation</span>
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

            <div className={`output-phrase ${isAwaitingLiveSign ? "awaiting-sign" : ""}`} aria-live="polite">
              {isAwaitingLiveSign && <Hand size={28} />}
              {displayedLatestText}
            </div>

            {livePreview?.candidateLabel && (
              <div className="candidate-pills" style={{ marginTop: "6px" }}>
                <span className="candidate-pill">
                  {livePreview.recognitionSource === "dynamic_motion"
                    ? `Live Motion: ${livePreview.candidateLabel}`
                    : `Live Pose: ${livePreview.candidateLabel}`}
                </span>
              </div>
            )}

            {latestOutput && (
              <>
                <div className="output-meta">
                  <span>Raw label: {latestOutput.label}</span>
                  <span>
                    {latestOutput.recognitionSource === "dynamic_motion"
                      ? "Moving Sign"
                      : "Static Sign"}
                  </span>
                  <span>Hand: {latestOutput.handedness}</span>
                  <span>Zone: {latestOutput.zone.toUpperCase()}</span>
                  {latestOutput.heightLevel && (
                    <span>Level: {latestOutput.heightLevel}</span>
                  )}
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
              aria-label="Sign to speech transcript"
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
                  title="Inflect transcript using Holistic Non-Manual Markers (? on brow raise, Not on head shake)"
                >
                  <input
                    type="checkbox"
                    checked={applyNmmGrammar}
                    onChange={(e) => setApplyNmmGrammar(e.target.checked)}
                  />
                  <span>NMM grammar</span>
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
              <span>Holistic Spatial Grammar</span>
              <button
                type="button"
                className="text-btn"
                onClick={() => {
                  setActiveLoci({ left: null, right: null });
                  setLastDirectionalEvent(null);
                }}
              >
                Reset Loci
              </button>
            </div>

            <div className="loci-grid">
              <div
                className={`locus-box ${activeLoci.left ? "filled" : ""} ${
                  referentFocus === "left" ? "focused" : ""
                }`}
              >
                <small>
                  LEFT ZONE {referentFocus === "left" ? "· FOCUS" : ""}
                </small>
                <strong>{activeLoci.left || "Unassigned"}</strong>
              </div>
              <div
                className={`locus-box ${activeLoci.right ? "filled" : ""} ${
                  referentFocus === "right" ? "focused" : ""
                }`}
              >
                <small>
                  RIGHT ZONE {referentFocus === "right" ? "· FOCUS" : ""}
                </small>
                <strong>{activeLoci.right || "Unassigned"}</strong>
              </div>
            </div>

            {spatialGrammar && (
              <div className="nmm-grid">
                <div className="nmm-item">
                  <small>ANCHOR</small>
                  <strong>
                    {space?.source === "holistic_pose" ? "Torso Pose" : "Frame"}
                  </strong>
                </div>
                <div className="nmm-item">
                  <small>CLAUSE MOOD</small>
                  <strong>{formatSentenceLabel(nmm?.sentence_type)}</strong>
                </div>
                <div className="nmm-item">
                  <small>BROWS</small>
                  <strong>{formatBrowLabel(nmm?.brow_state)}</strong>
                </div>
                <div className="nmm-item">
                  <small>GAZE / SHIFT</small>
                  <strong>
                    {nmm?.eye_gaze?.toUpperCase() || "CENTER"}
                    {space?.body_shift && space.body_shift !== "center"
                      ? ` · Shift ${space.body_shift.toUpperCase()}`
                      : ""}
                  </strong>
                </div>
              </div>
            )}

            {nmm?.modifiers?.length > 0 && (
              <div className="nmm-pills">
                {nmm.modifiers.map((mod) => (
                  <span key={mod} className="nmm-pill">
                    {mod.replace(/_/g, " ")}
                  </span>
                ))}
              </div>
            )}

            {lastDirectionalEvent && (
              <div className="directional-banner">
                <strong>
                  {lastDirectionalEvent.from_zone.toUpperCase()} →{" "}
                  {lastDirectionalEvent.to_zone.toUpperCase()}
                </strong>
                <span>
                  {lastDirectionalEvent.phrases?.[activeLanguage] ||
                    lastDirectionalEvent.phrases?.en ||
                    lastDirectionalEvent.action}
                </span>
              </div>
            )}

            <p className="spatial-hint">
              Point up in Left/Right to anchor Locus A/B, sweep across zones for
              directional verbs (A → B), raise brows for questions (?), or shake
              head for negation.
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
                        {item.sentenceType && item.sentenceType !== "statement"
                          ? ` · ${formatSentenceLabel(item.sentenceType)}`
                          : ""}
                      </small>
                    </div>
                    <time>{item.time}</time>
                  </div>
                ))
              )}
            </div>
          </section>
        </aside>}
      </main>
      <footer className="workspace-footer"><span><Hand size={14} /> A bridge between signs and words.</span><span>Signify studio <span className="footer-dot">·</span> {activeLangConfig.label}</span></footer>
      </div>
    </div>
  );
}
