const BACKEND_URL = "http://localhost:8000";
let processing = false;
let controller = null;
let activeTabId = null;
let keepAlive = null;

function log(message, details) {
  console.log("[Translate Video AI][background]", message, details || "");
}

function stopKeepAlive() {
  clearInterval(keepAlive);
  keepAlive = null;
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "startTranslation") {
    log("Solicitud recibida desde popup.", {
      tabId: message.tabId,
      sourceLanguage: message.sourceLanguage,
      targetLanguage: message.targetLanguage,
      model: message.model
    });
    startTranslation(message).then(
      () => {
        log("Transcripción y traducción completadas en el backend.");
        sendResponse({ ok: true });
      },
      (error) => {
        console.error("[Translate Video AI][background] Error iniciando traducción.", error);
        sendResponse({ ok: false, error: error.message });
      }
    );
    return true;
  }

  if (message.action === "stopTranslation") {
    log("Solicitud de detención recibida.");
    processing = false;
    controller?.abort();
    stopKeepAlive();
    if (activeTabId) {
      chrome.tabs.sendMessage(activeTabId, { action: "stopProcessing" }).catch(() => {});
    }
    sendResponse({ ok: true });
    return true;
  }
});

async function startTranslation({ tabId, sourceLanguage, targetLanguage, voice, model }) {
  if (!tabId || !sourceLanguage || !targetLanguage) {
    throw new Error("Faltan la pestaña o los idiomas.");
  }

  const preparation = await chrome.tabs.sendMessage(tabId, {
    action: "prepareDownload"
  });
  log("Respuesta de preparación recibida.", preparation);
  if (!preparation?.ok) {
    throw new Error(preparation?.error || "No se pudo preparar el video.");
  }

  processing = true;
  activeTabId = tabId;
  controller = new AbortController();
  keepAlive = setInterval(() => chrome.runtime.getPlatformInfo(), 20000);

  try {
    notifyTab(tabId, "Enviando el video al backend...");
    log("Enviando transcripción y traducción al backend.", {
      url: preparation.url,
      model: model || "base",
      sourceLanguage,
      targetLanguage
    });

    let response;
    try {
      response = await fetch(`${BACKEND_URL}/api/transcribe`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          videoUrl: preparation.url,
          model: model || "base",
          sourceLanguage,
          targetLanguage
        })
      });
    } catch (error) {
      if (error.name === "AbortError") throw new Error("Detenido.");
      console.error("[Translate Video AI][background] Error conectando con backend.", error);
      throw new Error(`No se pudo conectar con el backend: ${error.message}`);
    }

    let result;
    try {
      result = await response.json();
    } catch {
      throw new Error(`El backend respondió con HTTP ${response.status}.`);
    }
    log("Respuesta del backend recibida.", {
      status: response.status,
      ok: result?.ok,
      transcriptLength: result?.transcript?.length || 0,
      translationLength: result?.translation?.length || 0
    });
    if (!response.ok || !result.ok || !result.transcript || !result.translation) {
      throw new Error(result.detail || result.error || "No se pudo procesar el video.");
    }
    if (!processing) throw new Error("Detenido.");

    log("Enviando traducción a la pestaña.", {
      tabId,
      translationLength: result.translation.length
    });
    await chrome.tabs.sendMessage(tabId, {
      action: "translatedVideo",
      text: result.translation,
      voice
    });
    notifyTab(tabId, "Traducción lista. Reproduciendo video...");
  } finally {
    processing = false;
    stopKeepAlive();
  }
}

function notifyTab(tabId, text) {
  log("Actualizando estado en la pestaña.", text);
  chrome.tabs.sendMessage(tabId, {
    action: "translationStatus",
    text
  }).catch((error) => {
    console.warn("[Translate Video AI][background] No se pudo actualizar la pestaña.", error);
  });
}