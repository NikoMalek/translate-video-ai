const message = document.getElementById("message");

document.getElementById("start").addEventListener("click", async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  if (!tab?.id || !/^https:\/\/(www\.)?youtube\.com\//.test(tab.url || "")) {
    message.textContent = "Abre primero un video de YouTube.";
    return;
  }

  message.textContent = "Pausando el video y obteniendo la transcripción...";
  const response = await chrome.runtime.sendMessage({
    action: "startTranslation",
    tabId: tab.id,
    sourceLanguage: document.getElementById("sourceLanguage").value,
    targetLanguage: document.getElementById("targetLanguage").value,
    voice: document.getElementById("voice").value,
    model: document.getElementById("model").value
  });
  message.textContent = response?.ok ? "Procesando el audio..." : response?.error || "No se pudo iniciar.";
});

document.getElementById("stop").addEventListener("click", async () => {
  await chrome.runtime.sendMessage({ action: "stopTranslation" });
  message.textContent = "Detenido.";
});
