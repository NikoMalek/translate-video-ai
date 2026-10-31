const OFFSCREEN_URL = "offscreen.html";
const BACKEND_URL = "http://localhost:8000";
let processing = false;

async function ensureOffscreenDocument() {
  const contexts = await chrome.runtime.getContexts({
    contextTypes: ["OFFSCREEN_DOCUMENT"],
    documentUrls: [chrome.runtime.getURL(OFFSCREEN_URL)]
  });

  if (contexts.length === 0) {
    await chrome.offscreen.createDocument({
      url: OFFSCREEN_URL,
      reasons: ["AUDIO_PLAYBACK"],
      justification: "Traducir y reproducir el resultado del procesamiento local"
    });
  }
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "startTranslation") {
    startTranslation(message).then(
      () => sendResponse({ ok: true }),
      (error) => sendResponse({ ok: false, error: error.message })
    );
    return true;
  }

  if (message.action === "stopTranslation") {
    processing = false;
    chrome.runtime.sendMessage({ action: "stopProcessing" }).catch(() => {});
  }
});

async function startTranslation({ tabId, sourceLanguage, targetLanguage, voice, model }) {
  if (!tabId || !sourceLanguage || !targetLanguage) {
    throw new Error("Faltan la pestaña o los idiomas.");
  }

  const preparation = await chrome.tabs.sendMessage(tabId, {
    action: "prepareDownload"
  });
  if (!preparation?.ok) {
    throw new Error(preparation?.error || "No se pudo preparar el video.");
  }

  await ensureOffscreenDocument();
  processing = true;
  notifyTab(tabId, "Enviando el video al backend...");
  const response = await fetch(`${BACKEND_URL}/api/transcribe`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      videoUrl: preparation.url,
      model: model || "base"
    })
  });

  let result;
  try {
    result = await response.json();
  } catch {
    throw new Error(`El backend respondió con HTTP ${response.status}.`);
  }
  if (!response.ok || !result.ok || !result.transcript) {
    throw new Error(result.detail || result.error || "No se pudo transcribir el video.");
  }
  if (!processing) return;

  chrome.runtime.sendMessage({
    action: "startProcessing",
    tabId,
    sourceLanguage,
    targetLanguage,
    voice,
    transcript: result.transcript
  });
}

function notifyTab(tabId, text) {
  chrome.tabs.sendMessage(tabId, {
    action: "translationStatus",
    text
  }).catch(() => {});
}
