import { loadModels, translateText } from "./ai.js";

let options;
let processing = false;

chrome.runtime.onMessage.addListener((message) => {
  if (message.action === "startProcessing") {
    startProcessing(message).catch((error) => reportStatus(`Error: ${error.message}`));
  }
  if (message.action === "stopProcessing") {
    processing = false;
  }
});

async function startProcessing(configuration) {
  processing = true;
  options = configuration;
  reportStatus("Cargando modelos de traducción...");
  await loadModels();

  const chunks = splitTranscript(options.transcript);
  const translatedChunks = [];

  for (let index = 0; index < chunks.length && processing; index++) {
    reportStatus(`Traduciendo video: ${index + 1}/${chunks.length}`);
    translatedChunks.push(await translateText(
      chunks[index],
      options.sourceLanguage,
      options.targetLanguage
    ));
  }

  if (!processing) return;
  await chrome.tabs.sendMessage(options.tabId, {
    action: "translatedVideo",
    text: translatedChunks.join(" "),
    voice: options.voice
  });
  reportStatus("Traducción lista. Reproduciendo video...");
  processing = false;
}

function splitTranscript(text) {
  const words = text.trim().split(/\s+/);
  const chunks = [];
  for (let index = 0; index < words.length; index += 350) {
    chunks.push(words.slice(index, index + 350).join(" "));
  }
  return chunks.filter(Boolean);
}

function reportStatus(text) {
  if (options?.tabId) {
    chrome.tabs.sendMessage(options.tabId, {
      action: "translationStatus",
      text
    }).catch(() => {});
  }
}
