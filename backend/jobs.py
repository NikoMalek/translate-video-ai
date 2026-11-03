"""In-memory job queue for the local dubbing pipeline."""

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import shutil

from .pipeline import procesar_video

logger = logging.getLogger("translate-video-ai.jobs")

RANGES = {
    "download": (0.0, 0.08), "transcription": (0.08, 0.30),
    "diarization": (0.30, 0.36), "voices": (0.36, 0.38),
    "translation": (0.38, 0.46), "tts": (0.46, 0.68),
    "separation": (0.68, 0.90), "mixing": (0.90, 1.0),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipeline")
        self._root = Path(os.getenv("JOBS_DIR", "job_data"))
        self._root.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        url: str,
        source_language: str,
        target_language: str,
        voice: str | None,
        model: str,
    ) -> dict:
        job_id = uuid4().hex
        job = {
            "jobId": job_id,
            "status": "queued",
            "stage": "queued",
            "progress": 0.0,
            "error": None,
            "audioPath": None,
            "translation": None,
            "segments": None,
            "createdAt": _now(),
            "updatedAt": _now(),
        }
        with self._lock:
            self._jobs[job_id] = job
        self._executor.submit(
            self._run,
            job_id,
            url,
            source_language,
            target_language,
            voice,
            model,
        )
        logger.info("Job creado: id=%s status=queued", job_id)
        return self.public_status(job_id)

    def public_status(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise KeyError(job_id)
            result = {
                key: value
                for key, value in job.items()
                if key not in {"audioPath", "segments"}
            }
        if result["status"] == "completed":
            result["audioUrl"] = f"/api/jobs/{job_id}/audio"
        return result

    def audio_path(self, job_id: str) -> Path:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job["status"] != "completed" or not job["audioPath"]:
                raise KeyError(job_id)
            path = Path(job["audioPath"])
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def _update(self, job_id: str, **changes) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.update(changes, updatedAt=_now())

    def _progress(self, job_id: str, stage: str, progress: float) -> None:
        start, end = RANGES[stage]
        overall = start + (end - start) * progress
        self._update(job_id, status="processing", stage=stage, progress=overall)
        logger.info("Job %s: stage=%s progress=%.0f%%", job_id, stage, overall * 100)

    def _run(
        self,
        job_id: str,
        url: str,
        source_language: str,
        target_language: str,
        voice: str | None,
        model: str,
    ) -> None:
        output_dir = self._root / job_id
        try:
            def on_progress(stage: str, progress: float) -> None:
                self._progress(job_id, stage, progress)

            audio_path, segments = procesar_video(
                url,
                source_language=source_language,
                target_language=target_language,
                voice=voice,
                output_dir=output_dir,
                whisper_model=model,
                on_progress=on_progress,
            )
            for name in ("audio.wav", "dubbed.wav"):
                (output_dir / name).unlink(missing_ok=True)
            for source in output_dir.glob("source.*"):
                source.unlink(missing_ok=True)
            shutil.rmtree(output_dir / "clips", ignore_errors=True)
            translation = " ".join(
                segment.get("text_translated", "").strip()
                for segment in segments
                if segment.get("text_translated", "").strip()
            )
            self._update(
                job_id,
                status="completed",
                stage="finished",
                progress=1.0,
                audioPath=str(audio_path),
                translation=translation,
                segments=segments,
            )
            logger.info("Job completado: id=%s audio=%s", job_id, audio_path)
        except Exception as error:
            logger.exception("Job fallido: id=%s", job_id)
            shutil.rmtree(output_dir, ignore_errors=True)
            self._update(job_id, status="failed", stage="failed", error=str(error))

job_manager = JobManager()
