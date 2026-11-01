const message = document.getElementById("message");

const TARGET_VOICES = {
  spa_Latn: "es-ES", eng_Latn: "en-US", fra_Latn: "fr-FR",
  deu_Latn: "de-DE", ita_Latn: "it-IT", por_Latn: "pt-BR"
};

function log(message, details) {
  console.log("[Translate Video AI][popup]", message, details || "");
}

document.getElementById("start").addEventListener("click", async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  log("Botón procesar pulsado.", { tabId: tab?.id, url: tab?.url });

  if (!tab?.id || !/^https:\/\/(www\.)?youtube\.com\//.test(tab.url || "")) {
    message.textContent = "Abre primero un video de YouTube.";
    return;
  }

  message.textContent = "Pausando el video y obteniendo la transcripción...";
  try {
    const request = {
      action: "startTranslation",
      tabId: tab.id,
      sourceLanguage: document.getElementById("sourceLanguage").value,
      targetLanguage: document.getElementById("targetLanguage").value,
      voice: TARGET_VOICES[document.getElementById("targetLanguage").value],
      model: document.getElementById("model").value
    };
    log("Enviando solicitud al background.", request);
    const response = await chrome.runtime.sendMessage(request);
    log("Respuesta del background recibida.", response);
    message.textContent = response?.ok ? "Listo." : response?.error || "No se pudo iniciar.";
  } catch (error) {
    console.error("[Translate Video AI][popup] Error enviando solicitud.", error);
    message.textContent = `Error: ${error.message}`;
  }
});

document.getElementById("stop").addEventListener("click", async () => {
  log("Botón detener pulsado.");
  await chrome.runtime.sendMessage({ action: "stopTranslation" });
  message.textContent = "Detenido.";
});
