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

load_dotenv()

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


@lru_cache(maxsize=1)
def get_whisper_model(model_name: str):
    return WhisperModel(model_name, device="cpu", compute_type="int8")


@app.post("/api/transcribe")
def transcribe(request: TranscribeRequest):
    model_name = request.model or os.getenv("WHISPER_MODEL", "base")
    if model_name not in {"tiny", "base", "small", "medium", "large-v3"}:
        raise HTTPException(status_code=400, detail="Modelo Whisper no permitido.")

    with tempfile.TemporaryDirectory(prefix="translate-video-ai-") as temp_dir:
        output_template = str(Path(temp_dir) / "source.%(ext)s")
        options = {
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
        }

        try:
            with yt_dlp.YoutubeDL(options) as downloader:
                downloader.download([str(request.videoUrl)])

            audio_files = [
                path for path in Path(temp_dir).glob("source.*")
                if path.suffix not in {".part", ".ytdl"}
            ]
            if not audio_files:
                raise RuntimeError("yt-dlp no produjo un archivo de audio.")

            model = get_whisper_model(model_name)
            segments, _ = model.transcribe(str(audio_files[0]), beam_size=5)
            transcript = " ".join(segment.text.strip() for segment in segments).strip()
            return {"ok": True, "transcript": transcript.strip()}
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(status_code=502, detail=f"No se pudo transcribir el video: {error}") from error
