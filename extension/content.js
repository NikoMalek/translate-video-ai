let savedVideo;
let dubbedAudio;
let dubbedAudioUrl;
let audioReady = false;
let syncing = false;
let originalVolumeControl;
let customVolumeControl;
let previousVideoMuted = false;
let previousVideoVolume = 1;
let volumeObserver;
let volumeMountTimer = null;
let volumeChromeBottom = null;
const hiddenVolumeControls = new Set();
let dubVideoId = null;
let lastVolume = 1;
let mutedIconMarkup = null;
let unmutedIconMarkup = null;

function log(message, details) {
  console.log("[Translate Video AI][content]", message, details || "");
}

function setStatus(text) {
  log("Estado de interfaz.", text);
}

function prepareDownload() {
  const video = document.querySelector("video");
  if (!video) return { ok: false, error: "No se encontró el reproductor de YouTube." };
  video.pause();
  savedVideo = video;
  dubVideoId = new URLSearchParams(window.location.search).get("v");
  previousVideoMuted = video.muted;
  previousVideoVolume = video.volume;
  // Guardamos el icono sin silenciar antes de silenciar el video, para poder
  // alternar el icono del clon entre los dos estados.
  const nativeIcon = document.querySelector(".ytp-volume-icon");
  if (!unmutedIconMarkup && nativeIcon?.querySelector(".ytp-svg-volume-animation-speaker")) {
    unmutedIconMarkup = nativeIcon.innerHTML;
  }
  video.muted = true;
  bindVideoControls(video);
  setStatus("Video pausado. Descargando solo el audio...");
  log("Video preparado.", { url: window.location.href });
  return { ok: true, url: window.location.href };
}
function adIsShowing() {
  return !!document.querySelector(".html5-video-player.ad-showing");
}

function bindVideoControls(video) {
  if (video.dataset.translateVideoAiBound === "true") return;
  video.dataset.translateVideoAiBound = "true";
  video.addEventListener("volumechange", () => {
    if (audioReady && !video.muted) video.muted = true;
  });
  video.addEventListener("play", () => {
    if (!audioReady || syncing || adIsShowing()) return;
    dubbedAudio.play().catch((error) => console.warn("[Translate Video AI][content] No se pudo reproducir el audio.", error));
  });
  video.addEventListener("pause", () => {
    if (!audioReady || syncing) return;
    dubbedAudio.pause();
  });
  video.addEventListener("seeking", () => {
    if (audioReady && Number.isFinite(video.currentTime)) syncAudioTime(video.currentTime);
  });
  video.addEventListener("ratechange", () => {
    if (audioReady) dubbedAudio.playbackRate = video.playbackRate;
  });
  video.addEventListener("timeupdate", () => {
    if (audioReady && adIsShowing()) {
      dubbedAudio.pause();
      return;
    }
    if (audioReady && dubbedAudio.paused && !video.paused) {
      dubbedAudio.play().catch(() => {});
    }
    if (audioReady && Number.isFinite(video.currentTime) && Math.abs(dubbedAudio.currentTime - video.currentTime) > 0.35) {
      syncAudioTime(video.currentTime);
    }
  });
  video.addEventListener("ended", () => {
    if (!audioReady) return;
    dubbedAudio.pause();
    dubbedAudio.currentTime = 0;
  });
}

function syncAudioTime(time) {
  if (!dubbedAudio || !Number.isFinite(time)) return;
  syncing = true;
  dubbedAudio.currentTime = time;
  syncing = false;
}

function updateVolumeIcon(icon, muted) {
  const label = muted ? "Activar audio traducido" : "Silenciar audio traducido";
  const elements = [customVolumeControl?.querySelector(".ytp-mute-button"), icon].filter(Boolean);
  for (const element of elements) {
    element.setAttribute("aria-label", label);
    element.setAttribute("data-tooltip-title", label);
    element.setAttribute("data-title-no-tooltip", label);
    element.setAttribute("data-translate-muted", String(muted));
  }
  if (!icon) return;
  const markup = muted ? mutedIconMarkup : unmutedIconMarkup;
  if (markup && icon.innerHTML !== markup) icon.innerHTML = markup;
}

function clearVolumeMountTimer() {
  if (!volumeMountTimer) return;
  clearInterval(volumeMountTimer);
  volumeMountTimer = null;
}

function scheduleVolumeControlMount() {
  if (volumeMountTimer || customVolumeControl || !audioReady) return;
  const deadline = Date.now() + 30000;
  volumeMountTimer = setInterval(() => {
    if (!audioReady || customVolumeControl || Date.now() > deadline) {
      clearVolumeMountTimer();
      return;
    }
    mountCustomVolumeControl();
  }, 500);
}

function hideNativeVolumeControl(control) {
  if (!control || hiddenVolumeControls.has(control)) return;
  control.style.display = "none";
  hiddenVolumeControls.add(control);
}

// YouTube expande el slider añadiendo la clase ytp-volume-slider-active al
// chrome bottom. Como ese evento está atado al control nativo (oculto), hay que
// replicarlo en el clon al pasar el ratón, enfocar o arrastrar.
function setVolumeSliderOpen(open) {
  if (!customVolumeControl) return;
  volumeChromeBottom = customVolumeControl.closest(".ytp-chrome-bottom") || volumeChromeBottom;
  volumeChromeBottom?.classList.toggle("ytp-volume-slider-active", open);
  const panel = customVolumeControl.querySelector(".ytp-volume-panel");
  if (!panel) return;
  const handle = customVolumeControl.querySelector(".ytp-volume-slider-handle");
  const bigMode = (handle?.offsetWidth || 12) >= 18;
  panel.classList.toggle("ytp-volume-control-hover", open);
  panel.style.width = open ? (bigMode ? "78px" : "52px") : "";
  panel.style.marginRight = open ? (bigMode ? "5px" : "3px") : "";
}

function removeCustomVolumeControl() {
  clearVolumeMountTimer();
  volumeObserver?.disconnect();
  volumeObserver = null;
  volumeChromeBottom?.classList.remove("ytp-volume-slider-active");
  volumeChromeBottom = null;
  for (const control of hiddenVolumeControls) control.style.display = "";
  hiddenVolumeControls.clear();
  originalVolumeControl = null;
  customVolumeControl?.remove();
  customVolumeControl = null;
}

function mountCustomVolumeControl() {
  if (!audioReady || customVolumeControl) return;
  const nativeControl = document.querySelector(".ytp-volume-area:not(.ytp-translate-volume-clone)");
  if (!nativeControl?.parentNode) {
    log("Los controles de YouTube aún no están disponibles; se reintentará el volumen.");
    scheduleVolumeControlMount();
    return;
  }

  // Se clona antes de ocultar el control nativo: si se oculta primero, el clon
  // hereda el style="display: none" y nunca se muestra.
  const clone = nativeControl.cloneNode(true);
  clone.classList.add("ytp-translate-volume-clone");
  clone.removeAttribute("id");
  clone.style.display = "";

  const button = clone.querySelector(".ytp-mute-button") || clone;
  const icon = clone.querySelector(".ytp-volume-icon") || button;
  const slider = clone.querySelector(".ytp-volume-slider");
  const handle = clone.querySelector(".ytp-volume-slider-handle");
  const panel = clone.querySelector(".ytp-volume-panel");
  if (!slider) {
    clone.remove();
    log("No se encontró la estructura de volumen de YouTube.");
    return;
  }

  const nativeIcon = nativeControl.querySelector(".ytp-volume-icon");
  if (nativeIcon) {
    if (nativeIcon.querySelector(".ytp-svg-volume-animation-speaker")) {
      unmutedIconMarkup ||= nativeIcon.innerHTML;
    } else {
      mutedIconMarkup = nativeIcon.innerHTML;
    }
  }

  const setVolume = (value) => {
    const volume = Math.max(0, Math.min(1, Number(value)));
    if (!Number.isFinite(volume)) return;
    if (volume > 0) lastVolume = volume;
    dubbedAudio.volume = volume;
    dubbedAudio.muted = volume === 0;
    if (handle) {
      // El tirador nativo se mueve entre 0 y (ancho del slider - ancho del
      // tirador). En modo grande (pantalla completa) el tirador mide 18 px.
      const maxLeft = (handle.offsetWidth || 12) >= 18 ? 60 : 40;
      handle.style.left = `${volume * maxLeft}px`;
    }
    if (panel) {
      const percent = Math.round(volume * 100);
      panel.setAttribute("aria-valuenow", String(percent));
      panel.setAttribute("aria-valuetext", `${percent}% volumen`);
    }
    updateVolumeIcon(icon, dubbedAudio.muted);
  };
  const setVolumeFromPointer = (event) => {
    const bounds = slider.getBoundingClientRect();
    if (bounds.width) setVolume((event.clientX - bounds.left) / bounds.width);
  };

  let sliderHovered = false;
  let sliderDragging = false;
  const updateSliderOpen = () => setVolumeSliderOpen(sliderHovered || sliderDragging);

  button.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    setVolume(dubbedAudio.muted ? lastVolume || 1 : 0);
  });
  clone.addEventListener("mouseenter", () => {
    sliderHovered = true;
    updateSliderOpen();
  });
  clone.addEventListener("mouseleave", () => {
    sliderHovered = false;
    updateSliderOpen();
  });
  clone.addEventListener("focusin", () => {
    sliderHovered = true;
    updateSliderOpen();
  });
  clone.addEventListener("focusout", () => {
    setTimeout(() => {
      if (!clone.contains(document.activeElement)) {
        sliderHovered = false;
        updateSliderOpen();
      }
    }, 0);
  });
  slider.addEventListener("mousedown", (event) => {
    sliderDragging = true;
    updateSliderOpen();
    setVolumeFromPointer(event);
    const stopDragging = () => {
      sliderDragging = false;
      updateSliderOpen();
      document.removeEventListener("mouseup", stopDragging);
    };
    document.addEventListener("mouseup", stopDragging);
  });
  slider.addEventListener("mousemove", (event) => {
    if (event.buttons & 1) setVolumeFromPointer(event);
  });
  nativeControl.parentNode.insertBefore(clone, nativeControl);
  originalVolumeControl = nativeControl;
  hideNativeVolumeControl(nativeControl);
  customVolumeControl = clone;
  clearVolumeMountTimer();
  setVolume(dubbedAudio.volume);

  volumeObserver = new MutationObserver(() => {
    // Se excluye el clon: como se inserta antes que el control nativo, un
    // querySelector normal devolvería el clon y lo ocultaría a él.
    for (const control of document.querySelectorAll(".ytp-volume-area:not(.ytp-translate-volume-clone)")) {
      hideNativeVolumeControl(control);
      originalVolumeControl = control;
    }
    if (customVolumeControl && !document.contains(customVolumeControl)) {
      volumeObserver?.disconnect();
      volumeObserver = null;
      customVolumeControl = null;
      mountCustomVolumeControl();
    }
  });
  volumeObserver.observe(nativeControl.parentNode, { childList: true });
  log("Control de volumen clonado y montado en la posición nativa.");
}

async function loadDubbedAudio(url) {
  log("Descargando audio traducido.", { url });
  setStatus("Descargando el audio traducido...");
  const response = await fetch(url);
  if (!response.ok) throw new Error(`El backend respondió con HTTP ${response.status}.`);
  const blob = await response.blob();
  if (!blob.size) throw new Error("El backend devolvió un audio vacío.");

  if (dubbedAudioUrl) URL.revokeObjectURL(dubbedAudioUrl);
  dubbedAudioUrl = URL.createObjectURL(blob);
  if (!dubbedAudio) {
    dubbedAudio = document.createElement("audio");
    dubbedAudio.preload = "auto";
    dubbedAudio.style.display = "none";
    document.body.appendChild(dubbedAudio);
  }
  dubbedAudio.src = dubbedAudioUrl;
  dubbedAudio.volume = 1;
  dubbedAudio.load();
  audioReady = true;
  if (savedVideo) {
    bindVideoControls(savedVideo);
    savedVideo.muted = true;
    syncAudioTime(savedVideo.currentTime);
    dubbedAudio.playbackRate = savedVideo.playbackRate;
  }
  mountCustomVolumeControl();
  const audioPlay = dubbedAudio.play();
  const videoPlay = savedVideo?.play();
  await Promise.allSettled([audioPlay, videoPlay]);
  setStatus("Audio traducido listo. Usa los controles de YouTube para controlarlo.");
  log("Audio cargado y reproducción solicitada.", { bytes: blob.size, type: blob.type });
}

function stopDubbing() {
  audioReady = false;
  if (savedVideo) {
    savedVideo.muted = previousVideoMuted;
    savedVideo.volume = previousVideoVolume;
  }
  removeCustomVolumeControl();
  if (dubbedAudio) {
    syncing = true;
    dubbedAudio.pause();
    dubbedAudio.removeAttribute("src");
    dubbedAudio.load();
    syncing = false;
  }
  if (dubbedAudioUrl) {
    URL.revokeObjectURL(dubbedAudioUrl);
    dubbedAudioUrl = null;
  }
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
  if (message.action === "audioReady") {
    if (new URLSearchParams(window.location.search).get("v") !== dubVideoId) {
      log("El video cambió mientras se procesaba; se ignora el audio.");
      stopDubbing();
      return;
    }
    loadDubbedAudio(message.audioUrl).catch((error) => {
      console.error("[Translate Video AI][content] No se pudo cargar el audio traducido.", error);
      setStatus(`Error descargando el audio: ${error.message}`);
      stopDubbing();
    });
  }
  if (message.action === "stopProcessing") {
    log("Solicitud de detención recibida.");
    stopDubbing();
  }
});

document.addEventListener("yt-navigate-start", () => {
  if (savedVideo) stopDubbing();
});

log("Content script cargado.", window.location.href);
