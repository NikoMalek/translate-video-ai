import logging
import os
import threading
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl

from .jobs import job_manager
from .pipeline import precargar_modelos

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

def _preload() -> None:
    try:
        precargar_modelos(os.getenv("WHISPER_MODEL", "small"))
    except Exception:
        logger.exception("No se pudieron precargar los modelos.")


@app.on_event("startup")
def start_preload() -> None:
    threading.Thread(target=_preload, daemon=True).start()


class TranscribeRequest(BaseModel):
    videoUrl: HttpUrl
    model: str | None = None
    sourceLanguage: str = "eng_Latn"
    targetLanguage: str = "spa_Latn"


class JobRequest(TranscribeRequest):
    voice: str = "es-ES-ElviraNeural"


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
VOICE_LANGUAGE_PREFIXES = {
    "spa_Latn": "es-",
    "eng_Latn": "en-",
    "fra_Latn": "fr-",
    "deu_Latn": "de-",
    "ita_Latn": "it-",
    "por_Latn": "pt-",
}

@app.post("/api/jobs", status_code=202)
def create_job(request: JobRequest):
    host = (urlparse(str(request.videoUrl)).hostname or "").lower()
    if host not in {"www.youtube.com", "youtube.com", "m.youtube.com", "youtu.be"}:
        raise HTTPException(status_code=400, detail="Solo se admiten videos de YouTube.")
    if request.model and request.model not in {"tiny", "base", "small", "medium", "large-v3"}:
        raise HTTPException(status_code=400, detail="Modelo Whisper no permitido.")
    if (
        request.sourceLanguage not in SUPPORTED_TRANSLATION_LANGUAGES
        or request.targetLanguage not in SUPPORTED_TRANSLATION_LANGUAGES
    ):
        raise HTTPException(status_code=400, detail="Idioma de traducción no permitido.")
    voice_prefix = VOICE_LANGUAGE_PREFIXES.get(request.targetLanguage)
    if voice_prefix and not request.voice.startswith(voice_prefix):
        raise HTTPException(
            status_code=400,
            detail="La voz seleccionada no corresponde al idioma de destino.",
        )
    return job_manager.create(
        url=str(request.videoUrl),
        source_language=request.sourceLanguage,
        target_language=request.targetLanguage,
        voice=request.voice,
        model=request.model or os.getenv("WHISPER_MODEL", "small"),
    )


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    try:
        return job_manager.public_status(job_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Job no encontrado.") from error


@app.get("/api/jobs/{job_id}/audio")
def get_job_audio(job_id: str):
    try:
        path = job_manager.audio_path(job_id)
    except (KeyError, FileNotFoundError) as error:
        raise HTTPException(status_code=404, detail="Audio del job no disponible.") from error
    return FileResponse(path, media_type="audio/mpeg", filename="dubbed.mp3")
