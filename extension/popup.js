const message = document.getElementById("message");

const UI_TEXT = {
  es: {
    title: "Traductor de YouTube",
    source: "Idioma del video",
    target: "Traducir a",
    autoVoice: "La voz se elige automáticamente según el hablante.",
    model: "Modelo Whisper local",
    start: "Procesar y traducir",
    stop: "Detener",
    openYoutube: "Abre primero un video de YouTube.",
    preparing: "Pausando el video y preparando la traducción...",
    ready: "Listo.",
    stopped: "Detenido.",
    error: "Error"
  },
  en: {
    title: "YouTube Translator",
    source: "Video language",
    target: "Translate to",
    autoVoice: "The voice is chosen automatically based on the speaker.",
    model: "Local Whisper model",
    start: "Process and translate",
    stop: "Stop",
    openYoutube: "Open a YouTube video first.",
    preparing: "Pausing the video and preparing the translation...",
    ready: "Ready.",
    stopped: "Stopped.",
    error: "Error"
  }
};
const LANGUAGE_OPTIONS = {
  spa_Latn: { es: "Español", en: "Spanish" },
  eng_Latn: { es: "Inglés", en: "English" },
  fra_Latn: { es: "Francés", en: "French" },
  deu_Latn: { es: "Alemán", en: "German" },
  ita_Latn: { es: "Italiano", en: "Italian" },
  por_Latn: { es: "Portugués", en: "Portuguese" },
  jpn_Jpan: { es: "Japonés", en: "Japanese" },
  kor_Hang: { es: "Coreano", en: "Korean" }
};
const MODEL_OPTIONS = {
  tiny: { es: "Tiny: más rápido, menos preciso", en: "Tiny: faster, less accurate" },
  base: { es: "Base: recomendado", en: "Base: recommended" },
  small: { es: "Small: más preciso, más lento", en: "Small: more accurate, slower" },
  medium: { es: "Medium: alta precisión, lento", en: "Medium: high accuracy, slow" },
  "large-v3": { es: "Large-v3: máxima precisión, muy lento", en: "Large-v3: highest accuracy, very slow" }
};

let uiLanguage = localStorage.getItem("translate-video-ai-language") || "es";

function log(message, details) {
  console.log("[Translate Video AI][popup]", message, details || "");
}

function text() {
  return UI_TEXT[uiLanguage];
}

function applyLanguage() {
  const labels = text();
  document.documentElement.lang = uiLanguage;
  document.getElementById("title").textContent = labels.title;
  document.getElementById("sourceLanguageLabel").textContent = labels.source;
  document.getElementById("targetLanguageLabel").textContent = labels.target;
  document.getElementById("voiceHint").textContent = labels.autoVoice;
  document.getElementById("modelLabel").textContent = labels.model;
  document.getElementById("start").textContent = labels.start;
  document.getElementById("stop").textContent = labels.stop;
  document.getElementById("languageSwitch").textContent = uiLanguage === "es" ? "ES / EN" : "EN / ES";
  for (const option of document.querySelectorAll("#sourceLanguage option, #targetLanguage option")) {
    option.textContent = LANGUAGE_OPTIONS[option.value][uiLanguage];
  }
  for (const option of document.querySelectorAll("#model option")) {
    option.textContent = MODEL_OPTIONS[option.value][uiLanguage];
  }
}

applyLanguage();
document.getElementById("languageSwitch").addEventListener("click", () => {
  uiLanguage = uiLanguage === "es" ? "en" : "es";
  localStorage.setItem("translate-video-ai-language", uiLanguage);
  applyLanguage();
});

document.getElementById("start").addEventListener("click", async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  log("Botón procesar pulsado.", { tabId: tab?.id, url: tab?.url });

  if (!tab?.id || !/^https:\/\/(www\.)?youtube\.com\//.test(tab.url || "")) {
    message.textContent = text().openYoutube;
    return;
  }

  message.textContent = text().preparing;
  try {
    const request = {
      action: "startTranslation",
      tabId: tab.id,
      sourceLanguage: document.getElementById("sourceLanguage").value,
      targetLanguage: document.getElementById("targetLanguage").value,
      model: document.getElementById("model").value
    };
    log("Enviando solicitud al background.", request);
    const response = await chrome.runtime.sendMessage(request);
    log("Respuesta del background recibida.", response);
    message.textContent = response?.ok ? text().ready : response?.error || "Could not start.";
  } catch (error) {
    console.error("[Translate Video AI][popup] Error enviando solicitud.", error);
    message.textContent = `${text().error}: ${error.message}`;
  }
});

document.getElementById("stop").addEventListener("click", async () => {
  log("Botón detener pulsado.");
  await chrome.runtime.sendMessage({ action: "stopTranslation" });
  message.textContent = text().stopped;
});
