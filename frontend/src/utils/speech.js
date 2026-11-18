/**
 * Browser Web Speech API wrappers (SpeechSynthesis + SpeechRecognition).
 */

export function isTtsSupported() {
  return typeof window !== "undefined" && "speechSynthesis" in window;
}

export function getAvailableVoices() {
  if (!isTtsSupported()) return [];
  return window.speechSynthesis.getVoices();
}

export function speakText(text, locale = "en-IN") {
  if (!isTtsSupported() || !text) {
    return { spoken: false, reason: "SpeechSynthesis not supported in this browser." };
  }

  try {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = locale;
    utterance.rate = 0.98;

    const voices = window.speechSynthesis.getVoices();
    const langPrefix = locale.split("-")[0].toLowerCase();
    const exactMatch = voices.find(
      (v) => v.lang.toLowerCase() === locale.toLowerCase()
    );
    const prefixMatch = voices.find((v) =>
      v.lang.toLowerCase().startsWith(langPrefix)
    );

    if (exactMatch) {
      utterance.voice = exactMatch;
    } else if (prefixMatch) {
      utterance.voice = prefixMatch;
    }

    window.speechSynthesis.speak(utterance);
    return {
      spoken: true,
      hasNativeVoice: Boolean(exactMatch || prefixMatch),
    };
  } catch (err) {
    return { spoken: false, reason: err?.message || "Failed to trigger speech." };
  }
}

export function getSpeechRecognitionCtor() {
  if (typeof window === "undefined") return null;
  return window.SpeechRecognition || window.webkitSpeechRecognition || null;
}
