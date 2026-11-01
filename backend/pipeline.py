"""Reusable GPU/CPU pipeline for producing a dubbed audio track."""

import asyncio
import gc
import logging
import os
import shutil
import subprocess
import tempfile
import wave
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Callable
import importlib.util

import numpy as np

logger = logging.getLogger("translate-video-ai.pipeline")

def _add_cuda_dll_dirs() -> None:
    if os.name != "nt":
        return
    paths = []
    spec = importlib.util.find_spec("torch")
    if spec and spec.submodule_search_locations:
        paths.append(Path(list(spec.submodule_search_locations)[0]) / "lib")
    spec = importlib.util.find_spec("nvidia")
    if spec and spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            paths.extend(Path(location).glob("*/bin"))
    for path in paths:
        if path.is_dir():
            try:
                os.add_dll_directory(str(path))
            except OSError:
                pass
            os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")


_add_cuda_dll_dirs()

SAMPLE_RATE = 44100
MAX_TEMPO = 1.5
LANGUAGE_CODES = {
    "eng_Latn": "en",
    "spa_Latn": "es",
    "fra_Latn": "fr",
    "deu_Latn": "de",
    "ita_Latn": "it",
    "por_Latn": "pt",
    "jpn_Jpan": "ja",
    "kor_Hang": "ko",
}
DEFAULT_TTS_VOICES = {
    "es-ES": "es-CL-LorenzoNeural",
    "en-US": "en-US-JennyNeural",
    "fr-FR": "fr-FR-DeniseNeural",
    "de-DE": "de-DE-KatjaNeural",
    "pt-BR": "pt-BR-FranciscaNeural",
    "it-IT": "it-IT-ElsaNeural"
}
ProgressCallback = Callable[[str, float], None]


def _ffmpeg_command() -> str:
    executable = shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError(
            "FFmpeg no está instalado o no está disponible en PATH. "
            "Instálalo con 'winget install Gyan.FFmpeg' y reinicia la terminal."
        )
    return executable


def _progress(callback: ProgressCallback | None, stage: str, value: float) -> None:
    if callback:
        callback(stage, max(0.0, min(1.0, value)))


def _device() -> str:
    configured = os.getenv("DEVICE", "").lower()
    if not configured:
        try:
            import torch
            configured = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            configured = "cpu"
    if configured not in {"cuda", "cpu"}:
        raise ValueError("DEVICE debe ser 'cuda' o 'cpu'.")
    if configured == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("DEVICE=cuda pero PyTorch no detecta una GPU CUDA.")
    return configured


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / float(audio.getframerate())


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as audio:
        channels = audio.getnchannels()
        width = audio.getsampwidth()
        rate = audio.getframerate()
        if width != 2:
            raise ValueError(f"Se esperaba WAV de 16 bits, llegó {width * 8} bits.")
        data = np.frombuffer(audio.readframes(audio.getnframes()), dtype=np.int16)
    data = data.reshape(-1, channels).astype(np.float32) / 32768.0
    if channels == 1:
        data = np.repeat(data, 2, axis=1)
    return data, rate


def _write_wav(path: Path, data: np.ndarray) -> None:
    pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(SAMPLE_RATE)
        audio.writeframes(pcm.tobytes())


@lru_cache(maxsize=1)
def _get_whisper_model(model_name: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel

    logger.info("Cargando Whisper: model=%s device=%s", model_name, device)
    return WhisperModel(model_name, device=device, compute_type=compute_type)


@lru_cache(maxsize=1)
def _get_translation_model(device: str):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_name = "facebook/nllb-200-distilled-600M"
    logger.info("Cargando NLLB: device=%s", device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if device == "cuda":
        model = model.half().to(device)
    else:
        model = model.to(device)
    model.eval()
    return tokenizer, model, torch


@lru_cache(maxsize=1)
def _get_demucs_model():
    from demucs.pretrained import get_model

    logger.info("Cargando Demucs.")
    return get_model("htdemucs")


def _download_audio(url: str, output_dir: Path, max_seconds: int | None) -> tuple[Path, float]:
    import yt_dlp
    from yt_dlp.utils import download_range_func

    output_dir.mkdir(parents=True, exist_ok=True)
    options = {
        "format": "bestaudio/best",
        "outtmpl": str(output_dir / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    if max_seconds:
        options["download_ranges"] = download_range_func(None, [(0, max_seconds)])
    logger.info("Descargando audio: url=%s", url)
    max_minutes = int(os.getenv("MAX_VIDEO_MINUTES", "20"))
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(url, download=False)
        if info.get("is_live"):
            raise RuntimeError("No se pueden procesar transmisiones en vivo.")
        if (info.get("duration") or 0) > max_minutes * 60:
            raise RuntimeError(f"El video supera el máximo de {max_minutes} minutos.")
        downloader.download([url])
    sources = [
        path for path in output_dir.glob("source.*")
        if path.suffix not in {".part", ".ytdl"}
    ]
    if not sources:
        raise RuntimeError("yt-dlp no produjo ningún archivo de audio.")

    wav_path = output_dir / "audio.wav"
    command = [_ffmpeg_command(), "-y", "-v", "error", "-i", str(sources[0])]
    if max_seconds:
        command += ["-t", str(max_seconds)]
    command += ["-ac", "2", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(wav_path)]
    subprocess.run(command, check=True)
    return wav_path, _wav_seconds(wav_path)


def _transcribe(
    wav_path: Path,
    source_language: str,
    model_name: str,
    device: str,
    beam_size: int,
) -> list[dict]:
    from faster_whisper import BatchedInferencePipeline

    compute_type = os.getenv("WHISPER_COMPUTE", "float16" if device == "cuda" else "int8")
    model = _get_whisper_model(model_name, device, compute_type)
    pipeline = BatchedInferencePipeline(model=model)
    segments, _ = pipeline.transcribe(
        str(wav_path),
        batch_size=int(os.getenv("WHISPER_BATCH_SIZE", "16")),
        beam_size=beam_size,
        language=LANGUAGE_CODES[source_language],
    )
    return [
        {"start": float(segment.start), "end": float(segment.end), "text": segment.text.strip()}
        for segment in segments
        if segment.text.strip()
    ]


def _translate(
    segments: list[dict],
    source_language: str,
    target_language: str,
    device: str,
    batch_size: int,
) -> None:
    tokenizer, model, torch = _get_translation_model(device)
    tokenizer.src_lang = source_language
    target_id = tokenizer.convert_tokens_to_ids(target_language)
    texts = [segment["text"] for segment in segments]
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(device)
            generated = model.generate(
                **encoded,
                forced_bos_token_id=target_id,
                max_new_tokens=256,
                num_beams=int(os.getenv("NLLB_BEAMS", "2")),
            )
            translations = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for segment, translation in zip(segments[start:start + batch_size], translations):
                segment["text_translated"] = translation.strip()


def _tts(segments: list[dict], clips_dir: Path, voice: str, concurrency: int) -> None:
    import edge_tts

    resolved_voice = DEFAULT_TTS_VOICES.get(voice, voice)
    logger.info("Generando TTS: requested_voice=%s voice=%s", voice, resolved_voice)
    shutil.rmtree(clips_dir, ignore_errors=True)
    clips_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    async def render() -> None:
        semaphore = asyncio.Semaphore(concurrency)

        async def render_one(index: int, text: str) -> None:
            async with semaphore:
                last_error = None
                for attempt in range(3):
                    try:
                        await edge_tts.Communicate(text, resolved_voice).save(
                            str(clips_dir / f"{index:04d}.mp3")
                        )
                        return
                    except Exception as error:
                        last_error = error
                        await asyncio.sleep(0.6 * (attempt + 1))
                failures.append(f"{index}: {last_error}")

        await asyncio.gather(*(
            render_one(index, segment["text_translated"])
            for index, segment in enumerate(segments)
            if segment["text_translated"].strip()
        ))

    asyncio.run(render())
    total = sum(1 for segment in segments if segment["text_translated"].strip())
    if failures:
        logger.warning("TTS falló en %d de %d segmentos: %s", len(failures), total, failures[0])
    if len(failures) > max(2, int(total * 0.3)):
        raise RuntimeError(f"TTS falló en {len(failures)} de {total} segmentos: {failures[0]}")

def _decode_clip(path: Path, tempo: float = 1.0) -> np.ndarray:
    command = [_ffmpeg_command(), "-v", "error", "-i", str(path)]
    if tempo > 1.01:
        command += ["-filter:a", f"atempo={tempo:.4f}"]
    command += ["-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"]
    process = subprocess.run(command, capture_output=True)
    if process.returncode:
        raise RuntimeError(process.stderr.decode(errors="ignore")[:300])
    return np.frombuffer(process.stdout, dtype=np.float32).copy()


def _assemble_voice(segments: list[dict], clips_dir: Path, seconds: float, workers: int) -> np.ndarray:
    arrays: list[np.ndarray | None] = [None] * len(segments)

    def decode(index: int) -> np.ndarray | None:
        path = clips_dir / f"{index:04d}.mp3"
        return _decode_clip(path) if path.exists() else None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, audio in enumerate(pool.map(decode, range(len(segments)))):
            arrays[index] = audio

    redo: list[tuple[int, float]] = []
    for index, (segment, audio) in enumerate(zip(segments, arrays)):
        if audio is None or not len(audio):
            continue
        next_start = segments[index + 1]["start"] if index + 1 < len(segments) else segment["end"] + 1.5
        slot = max(0.3, next_start - segment["start"])
        factor = (len(audio) / SAMPLE_RATE) / slot
        if factor > 1.05:
            redo.append((index, min(factor, MAX_TEMPO)))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for (index, tempo), audio in zip(
            redo,
            pool.map(lambda item: _decode_clip(clips_dir / f"{item[0]:04d}.mp3", item[1]), redo),
        ):
            arrays[index] = audio

    voice = np.zeros(int(seconds * SAMPLE_RATE) + SAMPLE_RATE, dtype=np.float32)
    for segment, audio in zip(segments, arrays):
        if audio is None:
            continue
        start = int(segment["start"] * SAMPLE_RATE)
        end = min(len(voice), start + len(audio))
        if start < len(voice):
            voice[start:end] += audio[:end - start]
    return voice


def _separate_background(wav_path: Path, device: str) -> np.ndarray:
    import torch
    from demucs.apply import apply_model

    model = _get_demucs_model().to(device).eval()
    audio, rate = _read_wav(wav_path)
    if rate != model.samplerate:
        raise RuntimeError(f"Frecuencia {rate} != {model.samplerate}")
    mix = torch.from_numpy(audio.T.copy())
    reference = mix.mean(0)
    mean, std = reference.mean(), reference.std() + 1e-8
    mix = (mix - mean) / std
    with torch.inference_mode():
        sources = apply_model(
            model,
            mix[None],
            device=device,
            split=True,
            overlap=0.25,
            shifts=0,
            progress=False,
        )[0]
    sources = sources * std + mean
    keep = [index for index, name in enumerate(model.sources) if name != "vocals"]
    return sources[keep].sum(0).cpu().numpy().T.astype(np.float32)


def _write_mix(voice: np.ndarray, background: np.ndarray | None, output_dir: Path) -> Path:
    length = len(voice)
    background_stereo = np.zeros((length, 2), dtype=np.float32)
    if background is not None:
        background_stereo[:min(length, len(background))] = background[:length]
    mixed = np.stack([voice, voice], axis=1) + background_stereo * 0.85
    wav_path = output_dir / "dubbed.wav"
    mp3_path = output_dir / "dubbed.mp3"
    _write_wav(wav_path, mixed)
    subprocess.run(
        [
            _ffmpeg_command(), "-y", "-v", "error", "-i", str(wav_path),
            "-c:a", "libmp3lame", "-b:a", "96k", str(mp3_path),
        ],
        check=True,
    )
    return mp3_path


def procesar_video(
    url: str,
    source_language: str = "eng_Latn",
    target_language: str = "spa_Latn",
    voice: str = "es-ES-ElviraNeural",
    output_dir: str | Path | None = None,
    max_seconds: int | None = None,
    whisper_model: str = "small",
    device: str | None = None,
    on_progress: ProgressCallback | None = None,
) -> tuple[Path, list[dict]]:
    """Process a complete video and return its dubbed MP3 and timed segments.

    Models are cached by configuration, so subsequent calls reuse the same
    in-memory models. ``on_progress`` receives a stage name and a 0..1 value.
    """
    if source_language not in LANGUAGE_CODES or target_language not in LANGUAGE_CODES:
        raise ValueError("Idioma de origen o destino no soportado.")
    if not url:
        raise ValueError("La URL del video es obligatoria.")

    if device:
        os.environ["DEVICE"] = device
    active_device = _device()
    output_path = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="translate-video-ai-"))
    output_path.mkdir(parents=True, exist_ok=True)
    workers = int(os.getenv("PIPELINE_WORKERS", "8"))
    batch_size = int(os.getenv("NLLB_BATCH_SIZE", "16"))
    beam_size = int(os.getenv("WHISPER_BEAM_SIZE", "1"))

    logger.info("Iniciando pipeline: device=%s url=%s", active_device, url)
    _progress(on_progress, "download", 0.0)
    wav_path, seconds = _download_audio(url, output_path, max_seconds)
    _progress(on_progress, "download", 1.0)

    _progress(on_progress, "transcription", 0.0)
    segments = _transcribe(
        wav_path, source_language, whisper_model, active_device, beam_size
    )
    if not segments:
        raise RuntimeError("Whisper no devolvió segmentos.")
    _progress(on_progress, "transcription", 1.0)

    _progress(on_progress, "translation", 0.0)
    _translate(segments, source_language, target_language, active_device, batch_size)
    _progress(on_progress, "translation", 1.0)

    _progress(on_progress, "tts", 0.0)
    clips_dir = output_path / "clips"
    _tts(segments, clips_dir, voice, int(os.getenv("TTS_CONCURRENCY", "8")))
    _progress(on_progress, "tts", 1.0)

    voice_audio = _assemble_voice(segments, clips_dir, seconds, workers)
    background = None
    if os.getenv("SKIP_DEMUCS", "0").lower() not in {"1", "true", "yes"}:
        _progress(on_progress, "separation", 0.0)
        try:
            background = _separate_background(wav_path, active_device)
        except Exception:
            logger.exception("Demucs falló; se continúa sin música de fondo.")
        _progress(on_progress, "separation", 1.0)

    _progress(on_progress, "mixing", 0.0)
    mp3_path = _write_mix(voice_audio, background, output_path)
    _progress(on_progress, "mixing", 1.0)
    _free_gpu()
    logger.info("Pipeline completado: output=%s segments=%d", mp3_path, len(segments))
    return mp3_path, segments

def precargar_modelos(whisper_model: str = "small") -> None:
    device = _device()
    compute_type = os.getenv("WHISPER_COMPUTE", "float16" if device == "cuda" else "int8")
    _get_whisper_model(whisper_model, device, compute_type)
    _get_translation_model(device)
    _get_demucs_model().to(device).eval()
    logger.info("Modelos precargados en %s.", device)