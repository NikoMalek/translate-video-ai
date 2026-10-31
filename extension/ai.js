import { pipeline } from "@huggingface/transformers";

let translator;

export async function loadModels() {
  translator = await pipeline("translation", "Xenova/nllb-200-distilled-600M");
}

export async function translateText(text, sourceLanguage, targetLanguage) {
  const result = await translator(text, {
    src_lang: sourceLanguage,
    tgt_lang: targetLanguage
  });
  return result[0].translation_text.trim();
}
