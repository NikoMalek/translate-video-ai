let subtitle;
let status;
let savedVideo;
let speechRun = 0;

function log(message, details) {
  console.log("[Translate Video AI][content]", message, details || "");
}

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
  log("Video preparado.", { url: window.location.href });
  return { ok: true, url: window.location.href };
}

function speak(text, lang) {
  const run = ++speechRun;
  window.speechSynthesis.cancel();

  const sentences = text.match(/[^.!?…]+[.!?…]*\s*/g) || [text];
  const parts = [];
  let buffer = "";
  for (const sentence of sentences) {
    if (buffer && (buffer + sentence).length > 200) {
      parts.push(buffer);
      buffer = "";
    }
    buffer += sentence;
  }
  if (buffer) parts.push(buffer);

  let index = 0;
  const next = () => {
    if (run !== speechRun || index >= parts.length) return;
    const utterance = new SpeechSynthesisUtterance(parts[index++].trim());
    utterance.lang = lang;
    utterance.rate = 1;
    utterance.onend = next;
    utterance.onerror = next;
    window.speechSynthesis.speak(utterance);
  };
  next();
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "prepareDownload") {
    log("Solicitud de preparación recibida.");
    sendResponse(prepareDownload());
    return true;
  }
  if (message.action === "translationStatus") {
    log("Estado recibido.", message.text);
    setStatus(message.text);
  }
  if (message.action === "translatedVideo") {
    log("Traducción recibida.", { textLength: message.text?.length || 0 });
    setStatus("Traducción lista. Reproduciendo...");
    if (!savedVideo) {
      log("No hay video guardado para reproducir.");
    } else {
      savedVideo.muted = true;
      savedVideo.currentTime = 0;
      savedVideo.play().then(
        () => log("Video reproducido."),
        (error) => {
          console.error("[Translate Video AI][content] No se pudo reproducir el video.", error);
          setStatus("Traducción lista, pero YouTube bloqueó la reproducción automática.");
        }
      );
    }
    speak(message.text, message.voice);
  }
  if (message.action === "stopProcessing") {
    log("Solicitud de detención recibida.");
    speechRun++;
    window.speechSynthesis?.cancel();
    if (savedVideo) savedVideo.muted = false;
    subtitle && (subtitle.textContent = "");
    status && (status.textContent = "");
  }
});

createOverlay();
log("Content script cargado.", window.location.href);