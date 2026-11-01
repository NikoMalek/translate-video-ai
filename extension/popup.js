const message = document.getElementById("message");

const VOICE_LANGUAGE_NAMES = {
  spa_Latn: { es: "Español", en: "Spanish" },
  eng_Latn: { es: "Inglés", en: "English" },
  fra_Latn: { es: "Francés", en: "French" },
  deu_Latn: { es: "Alemán", en: "German" },
  ita_Latn: { es: "Italiano", en: "Italian" },
  por_Latn: { es: "Portugués", en: "Portuguese" }
};

const UI_TEXT = {
  es: {
    title: "Traductor de YouTube",
    source: "Idioma del video",
    target: "Traducir a",
    voice: "Voz",
    model: "Modelo Whisper local",
    start: "Procesar y traducir",
    stop: "Detener",
    loading: "Cargando voces...",
    noVoices: "No hay voces disponibles",
    voicesError: "No se pudieron cargar las voces",
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
    voice: "Voice",
    model: "Local Whisper model",
    start: "Process and translate",
    stop: "Stop",
    loading: "Loading voices...",
    noVoices: "No voices available",
    voicesError: "Could not load voices",
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
let voiceCatalog = null;

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
  document.getElementById("voiceLabel").textContent = labels.voice;
  document.getElementById("modelLabel").textContent = labels.model;
  document.getElementById("start").textContent = labels.start;
  document.getElementById("stop").textContent = labels.stop;
  document.getElementById("languageSwitch").textContent = uiLanguage === "es" ? "ES / EN" : "EN / ES";
  if (!voiceCatalog) {
    document.getElementById("voice").firstElementChild.textContent = labels.loading;
  }
  for (const option of document.querySelectorAll("#sourceLanguage option, #targetLanguage option")) {
    option.textContent = LANGUAGE_OPTIONS[option.value][uiLanguage];
  }
  for (const option of document.querySelectorAll("#model option")) {
    option.textContent = MODEL_OPTIONS[option.value][uiLanguage];
  }
}

async function loadVoices() {
  const response = await fetch(chrome.runtime.getURL("voices.json"));
  if (!response.ok) {
    throw new Error(`No se pudo cargar el catálogo de voces (HTTP ${response.status}).`);
  }
  return response.json();
}

function renderVoices(catalog, language) {
  const voiceSelect = document.getElementById("voice");
  const voices = catalog[language] || [];
  voiceSelect.replaceChildren();
  for (const voice of voices) {
    const option = document.createElement("option");
    option.value = voice.shortName;
    const displayName = uiLanguage === "en"
      ? voice.displayNameEn || voice.displayName
      : voice.displayName;
    option.textContent = `${VOICE_LANGUAGE_NAMES[language][uiLanguage]} - ${displayName || voice.shortName}`;
    voiceSelect.appendChild(option);
  }
  voiceSelect.disabled = voices.length === 0;
  if (!voices.length) {
    voiceSelect.appendChild(new Option(text().noVoices, ""));
  }
}

const voicesReady = loadVoices()
  .then((catalog) => {
    voiceCatalog = catalog;
    const targetLanguage = document.getElementById("targetLanguage");
    renderVoices(catalog, targetLanguage.value);
    targetLanguage.addEventListener("change", () => {
      renderVoices(catalog, targetLanguage.value);
    });
    return catalog;
  })
  .catch((error) => {
    console.error("[Translate Video AI][popup] Error cargando voces.", error);
    document.getElementById("voice").replaceChildren(
      new Option(text().voicesError, "")
    );
    message.textContent = `${text().error}: ${error.message}`;
    throw error;
  });

applyLanguage();
document.getElementById("languageSwitch").addEventListener("click", () => {
  uiLanguage = uiLanguage === "es" ? "en" : "es";
  localStorage.setItem("translate-video-ai-language", uiLanguage);
  applyLanguage();
  if (voiceCatalog) {
    renderVoices(voiceCatalog, document.getElementById("targetLanguage").value);
  }
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
    await voicesReady;
    const request = {
      action: "startTranslation",
      tabId: tab.id,
      sourceLanguage: document.getElementById("sourceLanguage").value,
      targetLanguage: document.getElementById("targetLanguage").value,
      voice: document.getElementById("voice").value,
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
