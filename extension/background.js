const BACKEND_URL = "http://localhost:8000"; // Cambia esto a la URL de tu backend
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

function setBadge(tabId, text) {
  chrome.action.setBadgeText({ tabId, text }).catch(() => {});
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

async function startTranslation({ tabId, sourceLanguage, targetLanguage, model }) {
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
  let failed = false;
  try {
    notifyTab(tabId, "Creando trabajo de procesamiento...");
    log("Enviando transcripción y traducción al backend.", {
      url: preparation.url,
      model: model || "base",
      sourceLanguage,
      targetLanguage
    });

    let response;
    try {
      response = await fetch(`${BACKEND_URL}/api/jobs`, {
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
    log("Job creado.", {
      status: response.status,
      jobId: result?.jobId
    });
    if (!response.ok || !result.jobId) {
      throw new Error(result.detail || result.error || "No se pudo crear el trabajo.");
    }

    const completed = await waitForJob(tabId, result.jobId);
    if (!processing) throw new Error("Detenido.");
    await chrome.tabs.sendMessage(tabId, {
      action: "audioReady",
      audioUrl: new URL(completed.audioUrl, BACKEND_URL).toString(),
      jobId: result.jobId
    });
    notifyTab(tabId, "Audio traducido descargado. Usa el reproductor para escucharlo.");
  } catch (error) {
    failed = error.message !== "Detenido.";
    chrome.tabs.sendMessage(tabId, { action: "stopProcessing" }).catch(() => {});
    throw error;
  } finally {
    processing = false;
    stopKeepAlive();
    setBadge(tabId, failed ? "ERR" : "");
  }

  async function waitForJob(tabId, jobId) {
    while (processing) {
      let response;
      try {
        response = await fetch(`${BACKEND_URL}/api/jobs/${jobId}`, {
          signal: controller.signal
        });
      } catch (error) {
        if (error.name === "AbortError") throw new Error("Detenido.");
        throw new Error(`No se pudo consultar el job: ${error.message}`);
      }

      const result = await response.json();
      log("Estado del job recibido.", {
        jobId,
        status: result.status,
        stage: result.stage,
        progress: result.progress
      });
      if (!response.ok) {
        throw new Error(result.detail || "No se pudo consultar el job.");
      }
      if (result.status === "failed") {
        throw new Error(result.error || "El procesamiento falló.");
      }
      if (result.status === "completed" && result.audioUrl) return result;

      notifyTab(tabId, `Procesando: ${result.stage} (${Math.round(result.progress * 100)}%)`);
      setBadge(tabId, `${Math.round(result.progress * 100)}%`);
      await new Promise((resolve) => setTimeout(resolve, 2000));
    }
    throw new Error("Detenido.");
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