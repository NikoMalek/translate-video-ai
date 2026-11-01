import logging
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import yt_dlp
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("translate-video-ai")

app = FastAPI(title="Translate Video AI API")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(chrome-extension://.*|https?://.*)$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranscribeRequest(BaseModel):
    videoUrl: HttpUrl
    model: str | None = None
    sourceLanguage: str = "eng_Latn"
    targetLanguage: str = "spa_Latn"


SUPPORTED_TRANSLATION_LANGUAGES = {
    "eng_Latn",
    "spa_Latn",
    "fra_Latn",
    "deu_Latn",
    "ita_Latn",
    "por_Latn",
    "jpn_Jpan",
    "kor_Hang",
}

WHISPER_LANGUAGES = {
    "eng_Latn": "en",
    "spa_Latn": "es",
    "fra_Latn": "fr",
    "deu_Latn": "de",
    "ita_Latn": "it",
    "por_Latn": "pt",
    "jpn_Jpan": "ja",
    "kor_Hang": "ko",
}


@lru_cache(maxsize=1)
def get_whisper_model(model_name: str):
    return WhisperModel(model_name, device="cpu", compute_type="int8")


@lru_cache(maxsize=1)
def get_nllb():
    logger.info("Cargando modelo de traducción NLLB.")
    name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    model.eval()
    return tokenizer, model


def translate_text(text: str, source_language: str, target_language: str) -> str:
    parts = split_words(text)
    if not parts:
        return ""

    tokenizer, model = get_nllb()
    tokenizer.src_lang = source_language
    target_id = tokenizer.convert_tokens_to_ids(target_language)

    translated = []
    batch_size = 4
    for start in range(0, len(parts), batch_size):
        batch = parts[start:start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=400,
        )
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=target_id,
                max_new_tokens=400,
                num_beams=4,
            )
        translated.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        logger.info("Traducidos %d/%d fragmentos.", min(start + batch_size, len(parts)), len(parts))

    return " ".join(item.strip() for item in translated).strip()

def split_words(text: str, size: int = 150) -> list[str]:
    words = text.split()
    return [" ".join(words[index:index + size]) for index in range(0, len(words), size)]

@app.post("/api/transcribe")
def transcribe(request: TranscribeRequest):
    model_name = request.model or os.getenv("WHISPER_MODEL", "base")
    logger.info(
        "Procesamiento solicitado: url=%s model=%s source=%s target=%s",
        request.videoUrl,
        model_name,
        request.sourceLanguage,
        request.targetLanguage,
    )
    if model_name not in {"tiny", "base", "small", "medium", "large-v3"}:
        raise HTTPException(status_code=400, detail="Modelo Whisper no permitido.")
    if (
        request.sourceLanguage not in SUPPORTED_TRANSLATION_LANGUAGES
        or request.targetLanguage not in SUPPORTED_TRANSLATION_LANGUAGES
    ):
        raise HTTPException(status_code=400, detail="Idioma de traducción no permitido.")

    with tempfile.TemporaryDirectory(prefix="translate-video-ai-") as temp_dir:
        output_template = str(Path(temp_dir) / "source.%(ext)s")
        options = {
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "extractor_args": {
                "youtube": {
                    "player_client": ["android", "ios"]
                }
            }
        }

        try:
            logger.info("Descargando audio del video.")
            with yt_dlp.YoutubeDL(options) as downloader:
                downloader.download([str(request.videoUrl)])

            audio_files = [
                path for path in Path(temp_dir).glob("source.*")
                if path.suffix not in {".part", ".ytdl"}
            ]
            if not audio_files:
                raise RuntimeError("yt-dlp no produjo un archivo de audio.")

            logger.info("Audio descargado: file=%s", audio_files[0].name)
            model = get_whisper_model(model_name)
            logger.info("Modelo Whisper cargado: model=%s", model_name)
            segments, _ = model.transcribe(str(audio_files[0]), beam_size=5, language=WHISPER_LANGUAGES[request.sourceLanguage])
            transcript = " ".join(segment.text.strip() for segment in segments).strip()
            logger.info("Transcripción completada: characters=%d", len(transcript))
            logger.info("Iniciando traducción.")
            translation = translate_text(
                transcript,
                request.sourceLanguage,
                request.targetLanguage,
            )
            logger.info("Traducción completada: characters=%d", len(translation))
            return {
                "ok": True,
                "transcript": transcript,
                "translation": translation,
            }
        except HTTPException:
            raise
        except Exception as error:
            logger.exception("Error transcribiendo el video.")
            raise HTTPException(status_code=502, detail=f"No se pudo transcribir el video: {error}") from error
