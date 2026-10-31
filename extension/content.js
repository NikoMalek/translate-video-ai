let subtitle;
let status;
let savedVideo;
let speechQueue = Promise.resolve();

function createOverlay() {
  if (subtitle) return;
  status = document.createElement("div");
  subtitle = document.createElement("div");
  const container = document.createElement("div");
  Object.assign(container.style, {
    position: "fixed", bottom: "12%", left: "50%", transform: "translateX(-50%)",
    zIndex: "2147483647", width: "min(90vw, 900px)", textAlign: "center",
    pointerEvents: "none", fontFamily: "Arial, sans-serif"
  });
  Object.assign(status.style, {
    color: "#b9d8ff", fontSize: "13px", marginBottom: "6px",
    textShadow: "0 1px 3px #000"
  });
  Object.assign(subtitle.style, {
    color: "white", background: "rgba(0, 0, 0, .78)", borderRadius: "6px",
    padding: "9px 14px", fontSize: "22px", lineHeight: "1.3",
    textShadow: "0 1px 2px #000"
  });
  container.append(status, subtitle);
  document.body.appendChild(container);
}

function setStatus(text) {
  createOverlay();
  status.textContent = text;
}

function prepareDownload() {
  const video = document.querySelector("video");
  if (!video) return { ok: false, error: "No se encontró el reproductor de YouTube." };
  video.pause();
  savedVideo = video;
  setStatus("Video pausado. Descargando solo el audio...");
  return { ok: true, url: window.location.href };
}

function speak(text, voiceName) {
  speechQueue = speechQueue.then(() => new Promise((resolve) => {
    if (!("speechSynthesis" in window) || !text.trim()) {
      resolve();
      return;
    }
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = voiceName;
    utterance.rate = 1;
    utterance.onend = resolve;
    utterance.onerror = resolve;
    window.speechSynthesis.speak(utterance);
  }));
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "prepareDownload") {
    sendResponse(prepareDownload());
    return true;
  }
  if (message.action === "translationStatus") setStatus(message.text);
  if (message.action === "translatedVideo") {
    setStatus("Traducción lista. Reproduciendo...");
    savedVideo?.play().catch(() => {});
    speak(message.text, message.voice);
  }
  if (message.action === "stopProcessing") {
    window.speechSynthesis?.cancel();
    subtitle && (subtitle.textContent = "");
    status && (status.textContent = "");
  }
});

createOverlay();
