import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Box,
  Check,
  Copy,
  FileAudio,
  Hand,
  Mic,
  MicOff,
  Radio,
  Send,
  Trash2,
  Video,
} from "lucide-react";
import SignAvatar3D from "./SignAvatar3D";
import { buildLocalSignPlan } from "../utils/signAnimations";
import { getSpeechRecognitionCtor } from "../utils/speech";

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function findTriggerIndex(queryLower, trigger) {
  const clean = trigger.trim().toLowerCase();
  if (!clean) return -1;
  if (/^[a-z0-9\s]+$/.test(clean)) {
    const match = new RegExp(`\\b${escapeRegExp(clean)}\\b`, "i").exec(
      queryLower
    );
    return match ? match.index : -1;
  }
  return queryLower.indexOf(clean);
}

function matchPhraseSequence(phrases, text) {
  const query = text.trim().toLowerCase();
  if (!query) return [];
  const hits = [];
  for (const item of phrases) {
    let bestIdx = -1;
    const candidates = [
      ...(item.triggers || []),
      ...Object.values(item.translations || {}),
    ];
    for (const trig of candidates) {
      const idx = findTriggerIndex(query, trig);
      if (idx !== -1 && (bestIdx === -1 || idx < bestIdx)) {
        bestIdx = idx;
      }
    }
    if (bestIdx !== -1) {
      hits.push({ item, index: bestIdx });
    }
  }
  hits.sort((a, b) => a.index - b.index);
  return hits.map((h) => h.item);
}

function pickSupportedRecorderMime() {
  if (typeof MediaRecorder === "undefined") return "";
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/mp4",
  ];
  return candidates.find((type) => MediaRecorder.isTypeSupported?.(type)) || "";
}

function extForMime(mimeType) {
  const lower = (mimeType || "").toLowerCase();
  if (lower.includes("ogg")) return "ogg";
  if (lower.includes("mp4") || lower.includes("m4a")) return "m4a";
  if (lower.includes("wav")) return "wav";
  return "webm";
}

export default function SpeechToSignPanel({
  phrases,
  activeLanguage,
  activeLangConfig,
  activeDialect = "isl-south",
}) {
  const recognitionRef = useRef(null);
  const recorderRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);
  const audioFileInputRef = useRef(null);

  const [recordingWhisper, setRecordingWhisper] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [transcribing, setTranscribing] = useState(false);
  const [liveListening, setLiveListening] = useState(false);
  const [transcript, setTranscript] = useState("Hello, thank you");
  const [manualText, setManualText] = useState("");
  const [micError, setMicError] = useState("");
  const [sttStatus, setSttStatus] = useState(null);
  const [copied, setCopied] = useState(false);
  const [stageMode, setStageMode] = useState("avatar3d");
  const [selectedPhraseId, setSelectedPhraseId] = useState(null);
  const [videoAvailable, setVideoAvailable] = useState(false);
  const [signPlan, setSignPlan] = useState(() =>
    buildLocalSignPlan("Hello, thank you", phrases)
  );
  const [playSignal, setPlaySignal] = useState(0);

  const matchedSequence = useMemo(
    () => matchPhraseSequence(phrases, transcript),
    [phrases, transcript]
  );

  const matchedPhrase = useMemo(() => {
    if (!transcript.trim()) {
      return (
        phrases.find((p) => p.id === selectedPhraseId) || phrases[0] || null
      );
    }
    if (matchedSequence.length === 0) return null;
    return (
      matchedSequence.find((p) => p.id === selectedPhraseId) ||
      matchedSequence[0]
    );
  }, [matchedSequence, phrases, selectedPhraseId, transcript]);

  const requestSignPlan = useCallback(
    async (textToSign, { replay = true } = {}) => {
      const cleaned = (textToSign || "").trim();
      if (!cleaned) {
        setSignPlan(buildLocalSignPlan("", phrases));
        return;
      }
      try {
        const res = await fetch("/api/text-to-sign", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: cleaned,
            language: activeLanguage || "en",
            dialect: activeDialect || "isl-south",
          }),
        });
        if (!res.ok) throw new Error("Planner request failed");
        const plan = await res.json();
        setSignPlan(plan);
        if (replay) setPlaySignal((n) => n + 1);
      } catch {
        setSignPlan(buildLocalSignPlan(cleaned, phrases));
        if (replay) setPlaySignal((n) => n + 1);
      }
    },
    [activeDialect, activeLanguage, phrases]
  );

  // Sync initial / edited transcript with the backend Sign Grammar Planner (debounced).
  useEffect(() => {
    const handle = setTimeout(() => {
      void requestSignPlan(transcript, { replay: true });
    }, 260);
    return () => clearTimeout(handle);
  }, [transcript, requestSignPlan]);

  // Cleanup active mic / speech recognition on unmount.
  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (recorderRef.current && recorderRef.current.state !== "inactive") {
        recorderRef.current.stop();
      }
      if (streamRef.current) {
        for (const track of streamRef.current.getTracks()) track.stop();
      }
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, []);

  // Only render <video> if an actual video file exists at videoUrl (avoids Vite HTML fallback).
  useEffect(() => {
    let cancelled = false;
    setVideoAvailable(false);
    if (!matchedPhrase?.videoUrl) return undefined;

    fetch(matchedPhrase.videoUrl, { method: "HEAD" })
      .then((res) => {
        if (cancelled) return;
        const ctype = res.headers.get("content-type") || "";
        setVideoAvailable(res.ok && ctype.startsWith("video/"));
      })
      .catch(() => {
        if (!cancelled) setVideoAvailable(false);
      });

    return () => {
      cancelled = true;
    };
  }, [matchedPhrase]);

  const sendAudioToWhisper = async (blob, filename) => {
    setTranscribing(true);
    setMicError("");
    try {
      const formData = new FormData();
      formData.append("file", blob, filename);
      formData.append("language", activeLanguage || "en");
      formData.append("task", "transcribe");

      const res = await fetch("/api/stt/transcribe", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data?.detail || "Whisper transcription failed.");
      }

      const spokenText = (data.text || "").trim();
      setSttStatus({
        source: "Whisper Tiny",
        duration: data.duration_seconds,
        processingMs: Math.round(data.processing_ms || 0),
        isSpeech: Boolean(data.is_speech),
      });

      if (!spokenText) {
        setMicError(
          "No speech detected in audio — speak closer to the mic and try again."
        );
        return;
      }

      setTranscript(spokenText);
      setSelectedPhraseId(null);
    } catch (err) {
      setMicError(
        err?.message ||
          "Could not reach Python Whisper backend (/api/stt/transcribe)."
      );
    } finally {
      setTranscribing(false);
    }
  };

  const stopWhisperRecording = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    }
    setRecordingWhisper(false);
  };

  const handleToggleWhisperMic = async () => {
    setMicError("");
    if (recordingWhisper) {
      stopWhisperRecording();
      return;
    }

    if (liveListening && recognitionRef.current) {
      recognitionRef.current.stop();
      setLiveListening(false);
    }

    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
      setMicError(
        "Microphone audio recording is unavailable in this browser. Use Live Browser Mic, Upload Audio, or type below."
      );
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      chunksRef.current = [];

      const mimeType = pickSupportedRecorderMime();
      const recorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);

      recorder.ondataavailable = (evt) => {
        if (evt.data && evt.data.size > 0) {
          chunksRef.current.push(evt.data);
        }
      };

      recorder.onstop = () => {
        for (const track of stream.getTracks()) track.stop();
        streamRef.current = null;
        const finalType = recorder.mimeType || mimeType || "audio/webm";
        const blob = new Blob(chunksRef.current, { type: finalType });
        chunksRef.current = [];
        if (blob.size > 0) {
          void sendAudioToWhisper(blob, `mic.${extForMime(finalType)}`);
        }
      };

      recorderRef.current = recorder;
      recorder.start();
      setRecordingSeconds(0);
      setRecordingWhisper(true);
      timerRef.current = setInterval(() => {
        setRecordingSeconds((s) => s + 1);
      }, 1000);
    } catch (err) {
      setMicError(
        err?.message || "Microphone permission denied or no microphone found."
      );
    }
  };

  const handleToggleLiveMic = () => {
    setMicError("");
    if (liveListening && recognitionRef.current) {
      recognitionRef.current.stop();
      setLiveListening(false);
      return;
    }

    if (recordingWhisper) {
      stopWhisperRecording();
    }

    const SpeechCtor = getSpeechRecognitionCtor();
    if (!SpeechCtor) {
      setMicError(
        "Browser Web Speech API requires Chrome/Edge. Use Record Voice (Whisper AI) instead!"
      );
      return;
    }

    const rec = new SpeechCtor();
    rec.lang = activeLangConfig?.sttLocale || "en-IN";
    rec.interimResults = true;
    rec.continuous = false;

    rec.onresult = (event) => {
      const text = Array.from(event.results)
        .map((r) => r[0].transcript)
        .join(" ");
      setTranscript(text);
      setSelectedPhraseId(null);
      setSttStatus({ source: "Browser Speech API", isSpeech: true });
    };

    rec.onerror = (event) => {
      const hint =
        event.error === "network"
          ? "Browser Speech API blocked/offline — use 'Record Voice (Whisper AI)' which runs locally."
          : `Microphone error: ${event.error || "unavailable"}`;
      setMicError(hint);
      setLiveListening(false);
    };

    rec.onend = () => {
      setLiveListening(false);
    };

    recognitionRef.current = rec;
    setLiveListening(true);
    rec.start();
  };

  const handleAudioFileUpload = async (event) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    await sendAudioToWhisper(file, file.name);
  };

  const handleSelectPhrase = (phraseItem) => {
    const text =
      phraseItem.translations?.[activeLanguage] || phraseItem.translations.en;
    setTranscript(text);
    setSelectedPhraseId(phraseItem.id);
    setMicError("");
    setPlaySignal((n) => n + 1);
  };

  const handleSubmitText = (e) => {
    e.preventDefault();
    if (!manualText.trim()) return;
    const text = manualText.trim();
    setTranscript(text);
    setSelectedPhraseId(null);
    setManualText("");
    setPlaySignal((n) => n + 1);
  };

  const handleCopyTranscript = async () => {
    if (!transcript.trim()) return;
    try {
      await navigator.clipboard.writeText(transcript);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Ignore clipboard failures
    }
  };

  return (
    <section className="speech-card">
      <div className="speech-controls">
        <button
          type="button"
          className={`mic-btn ${recordingWhisper ? "listening" : ""}`}
          onClick={handleToggleWhisperMic}
          disabled={transcribing}
          title="Record microphone audio and transcribe locally with Whisper AI"
        >
          {recordingWhisper ? <MicOff size={18} /> : <Mic size={18} />}
          <span>
            {transcribing
              ? "Transcribing..."
              : recordingWhisper
              ? `Stop & Transcribe (${recordingSeconds}s)`
              : "Record Voice (Whisper AI)"}
          </span>
        </button>

        <button
          type="button"
          className={`icon-btn ${liveListening ? "active" : ""}`}
          onClick={handleToggleLiveMic}
          title="Real-time browser streaming captions (Chrome/Edge online)"
        >
          <Radio size={15} />
          <span>{liveListening ? "Stop Live Mic" : "Live Browser Mic"}</span>
        </button>

        <input
          ref={audioFileInputRef}
          type="file"
          accept="audio/*"
          hidden
          onChange={handleAudioFileUpload}
        />
        <button
          type="button"
          className="icon-btn"
          onClick={() => audioFileInputRef.current?.click()}
          disabled={transcribing}
          title="Transcribe an audio file (.wav, .mp3, .webm, .m4a) via Whisper"
        >
          <FileAudio size={15} />
          <span>Upload Audio</span>
        </button>

        <form className="text-fallback-form" onSubmit={handleSubmitText}>
          <input
            type="text"
            value={manualText}
            onChange={(e) => setManualText(e.target.value)}
            placeholder="Try: What is your name? / My name is Pranam / Please help..."
          />
          <button
            type="submit"
            className="icon-btn active"
            title="Sign with 3D Avatar"
          >
            <Send size={15} />
          </button>
        </form>
      </div>

      {micError && <div className="inline-warn">{micError}</div>}

      <div className="voice-transcript-box">
        <div className="card-header">
          <span>
            Voice → Text ({activeLangConfig?.label || "English"})
            {sttStatus && (
              <small className="stt-meta-tag">
                {sttStatus.source}
                {sttStatus.duration ? ` · ${sttStatus.duration}s` : ""}
                {sttStatus.processingMs ? ` · ${sttStatus.processingMs}ms` : ""}
              </small>
            )}
          </span>
          <div className="transcript-actions">
            <button
              type="button"
              className="text-btn"
              onClick={handleCopyTranscript}
              disabled={!transcript.trim()}
            >
              {copied ? <Check size={13} /> : <Copy size={13} />}
              {copied ? "Copied" : "Copy"}
            </button>
            {transcript && (
              <button
                type="button"
                className="text-btn"
                onClick={() => {
                  setTranscript("");
                  setSelectedPhraseId(null);
                  setSttStatus(null);
                }}
              >
                <Trash2 size={13} /> Clear
              </button>
            )}
          </div>
        </div>
        <textarea
          className="transcript-textarea"
          rows={2}
          value={transcript}
          onChange={(e) => {
            setTranscript(e.target.value);
            setSelectedPhraseId(null);
          }}
          placeholder="Speak via Whisper AI or type any sentence — the 3D avatar signs dictionary phrases and fingerspells names!"
        />
        {signPlan?.gloss_line && (
          <div className="gloss-plan-line">
            <span className="gloss-badge">GLOSS PLAN</span>
            <strong>{signPlan.gloss_line}</strong>
          </div>
        )}
      </div>

      <div className="phrase-chips">
        {phrases.map((item) => (
          <button
            key={item.id}
            type="button"
            className={`chip-btn ${
              matchedPhrase?.id === item.id ? "selected" : ""
            }`}
            onClick={() => handleSelectPhrase(item)}
          >
            {item.translations[activeLanguage] || item.translations.en}
          </button>
        ))}
      </div>

      <div className="stage-mode-bar">
        <button
          type="button"
          className={`icon-btn ${stageMode === "avatar3d" ? "active" : ""}`}
          onClick={() => setStageMode("avatar3d")}
        >
          <Box size={15} />
          <span>3D Signing Avatar</span>
        </button>
        <button
          type="button"
          className={`icon-btn ${stageMode === "cue" ? "active" : ""}`}
          onClick={() => setStageMode("cue")}
        >
          <Video size={15} />
          <span>Video / Cue Card</span>
        </button>
      </div>

      {stageMode === "avatar3d" ? (
        <SignAvatar3D signPlan={signPlan} playSignal={playSignal} />
      ) : (
        <div className="sign-stage">
          {matchedPhrase ? (
            <>
              <div className="sign-stage-header">
                <span className="gloss-badge">GLOSS: {matchedPhrase.gloss}</span>
                <span className="spoken-echo">
                  {matchedPhrase.translations[activeLanguage] ||
                    matchedPhrase.translations.en}
                </span>
              </div>

              {videoAvailable ? (
                <video
                  key={matchedPhrase.id}
                  src={matchedPhrase.videoUrl}
                  controls
                  autoPlay
                  loop
                  muted
                  playsInline
                  className="sign-video-player"
                />
              ) : (
                <div className="sign-cue-fallback">
                  {matchedPhrase.imageUrl ? (
                    <img
                      src={matchedPhrase.imageUrl}
                      alt={`${matchedPhrase.gloss} reference sign`}
                      className="sign-ref-photo"
                    />
                  ) : (
                    <Hand size={36} />
                  )}
                  <h4>{matchedPhrase.gloss}</h4>
                  <p>{matchedPhrase.cue}</p>
                  <small>
                    Optional video: place a clip at{" "}
                    <code>frontend/public{matchedPhrase.videoUrl}</code> to play
                    prerecorded sign video automatically.
                  </small>
                </div>
              )}
            </>
          ) : (
            <div className="sign-cue-fallback">
              <Video size={32} />
              <h4>{signPlan?.gloss_line || "Speak or choose a phrase chip above"}</h4>
              <p>
                Switch back to <strong>3D Signing Avatar</strong> to watch the
                animated avatar sign &amp; fingerspell this utterance in 3D.
              </p>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
