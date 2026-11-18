import { useMemo, useRef, useState } from "react";
import { Mic, MicOff, Play, Send, Video } from "lucide-react";
import { getSpeechRecognitionCtor } from "../utils/speech";

export default function SpeechToSignPanel({
  phrases,
  activeLanguage,
  activeLangConfig,
}) {
  const recognitionRef = useRef(null);
  const [listening, setListening] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [manualText, setManualText] = useState("");
  const [micError, setMicError] = useState("");
  const [videoMissing, setVideoMissing] = useState(false);

  const matchedPhrase = useMemo(() => {
    const query = transcript.trim().toLowerCase();
    if (!query) return phrases[0] || null;
    return (
      phrases.find((item) =>
        item.triggers.some((t) => query.includes(t.toLowerCase()))
      ) || null
    );
  }, [phrases, transcript]);

  const handleToggleMic = () => {
    setMicError("");
    if (listening && recognitionRef.current) {
      recognitionRef.current.stop();
      setListening(false);
      return;
    }

    const SpeechCtor = getSpeechRecognitionCtor();
    if (!SpeechCtor) {
      setMicError(
        "Web Speech API microphone capture requires Chrome/Edge. Use the quick chips or text input below."
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
      setVideoMissing(false);
    };

    rec.onerror = (event) => {
      setMicError(`Microphone error: ${event.error || "unavailable"}`);
      setListening(false);
    };

    rec.onend = () => {
      setListening(false);
    };

    recognitionRef.current = rec;
    setListening(true);
    rec.start();
  };

  const handleSelectPhrase = (phraseText) => {
    setTranscript(phraseText);
    setVideoMissing(false);
  };

  const handleSubmitText = (e) => {
    e.preventDefault();
    if (!manualText.trim()) return;
    handleSelectPhrase(manualText.trim());
    setManualText("");
  };

  return (
    <section className="speech-card">
      <div className="speech-controls">
        <button
          type="button"
          className={`mic-btn ${listening ? "listening" : ""}`}
          onClick={handleToggleMic}
        >
          {listening ? <MicOff size={18} /> : <Mic size={18} />}
          <span>{listening ? "Stop Listening" : "Speak Into Microphone"}</span>
        </button>

        <form className="text-fallback-form" onSubmit={handleSubmitText}>
          <input
            type="text"
            value={manualText}
            onChange={(e) => setManualText(e.target.value)}
            placeholder="Or type a phrase (e.g. Thank you, Help, Water)..."
          />
          <button type="submit" className="icon-btn active" title="Translate to Sign">
            <Send size={15} />
          </button>
        </form>
      </div>

      {micError && <div className="inline-warn">{micError}</div>}

      <div className="phrase-chips">
        {phrases.map((item) => (
          <button
            key={item.id}
            type="button"
            className={`chip-btn ${
              matchedPhrase?.id === item.id ? "selected" : ""
            }`}
            onClick={() => handleSelectPhrase(item.translations.en)}
          >
            {item.translations[activeLanguage] || item.translations.en}
          </button>
        ))}
      </div>

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

            {!videoMissing ? (
              <video
                key={matchedPhrase.id}
                src={matchedPhrase.videoUrl}
                controls
                autoPlay
                loop
                muted
                playsInline
                className="sign-video-player"
                onError={() => setVideoMissing(true)}
              />
            ) : (
              <div className="sign-cue-fallback">
                <Video size={36} />
                <h4>{matchedPhrase.gloss}</h4>
                <p>{matchedPhrase.cue}</p>
                <small>
                  Drop a clip at <code>frontend/public{matchedPhrase.videoUrl}</code>{" "}
                  to play prerecorded sign video automatically.
                </small>
              </div>
            )}
          </>
        ) : (
          <div className="sign-cue-fallback">
            <Play size={32} />
            <h4>No matching phrase in local dictionary yet</h4>
            <p>
              Spoken: &ldquo;{transcript}&rdquo;. Try one of the quick phrase
              chips above or add the word to <code>src/data/signs.json</code>.
            </p>
          </div>
        )}
      </div>
    </section>
  );
}
