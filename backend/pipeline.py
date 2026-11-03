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
from dotenv import load_dotenv

load_dotenv()

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
# Voces por idioma y género. Se usan en orden para que dos hablantes del
# mismo género no compartan la misma voz.
TTS_VOICE_POOLS = {
    "spa_Latn": {
        "male": ["es-ES-AlvaroNeural", "es-MX-JorgeNeural", "es-AR-TomasNeural"],
        "female": ["es-ES-ElviraNeural", "es-MX-DaliaNeural", "es-AR-ElenaNeural"],
    },
    "eng_Latn": {
        "male": ["en-US-GuyNeural", "en-US-ChristopherNeural", "en-GB-RyanNeural"],
        "female": ["en-US-JennyNeural", "en-US-AriaNeural", "en-GB-SoniaNeural"],
    },
    "fra_Latn": {
        "male": ["fr-FR-HenriNeural", "fr-CA-AntoineNeural"],
        "female": ["fr-FR-DeniseNeural", "fr-CA-SylvieNeural"],
    },
    "deu_Latn": {
        "male": ["de-DE-ConradNeural", "de-AT-JonasNeural"],
        "female": ["de-DE-KatjaNeural", "de-AT-IngridNeural"],
    },
    "ita_Latn": {
        "male": ["it-IT-DiegoNeural"],
        "female": ["it-IT-ElsaNeural", "it-IT-IsabellaNeural"],
    },
    "por_Latn": {
        "male": ["pt-BR-AntonioNeural", "pt-PT-DuarteNeural"],
        "female": ["pt-BR-FranciscaNeural", "pt-PT-RaquelNeural"],
    },
    "jpn_Jpan": {
        "male": ["ja-JP-KeitaNeural"],
        "female": ["ja-JP-NanamiNeural"],
    },
    "kor_Hang": {
        "male": ["ko-KR-InJoonNeural"],
        "female": ["ko-KR-SunHiNeural"],
    },
}
# Voz por defecto de cada idioma y género (primera del pool).
AUTO_TTS_VOICES = {
    language: {gender: voices[0] for gender, voices in pools.items()}
    for language, pools in TTS_VOICE_POOLS.items()
}
FALLBACK_TTS_VOICE = "es-ES-ElviraNeural"
MALE_PITCH_THRESHOLD_HZ = float(os.getenv("VOICE_PITCH_THRESHOLD", "155"))
DEFAULT_DIARIZATION_MODEL = os.getenv(
    "PYANNOTE_MODEL", "pyannote/speaker-diarization-community-1"
)
LEGACY_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
# Clasificador de género que se aplica al audio de cada hablante ya separado
# por pyannote. Usamos el modelo multilingüe de XLS-R, más robusto que el
# clasificador pequeño anterior, sobre todo con voces mayores o con acento.
DEFAULT_GENDER_MODEL = os.getenv(
    "GENDER_MODEL",
    "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech",
)
GENDER_SAMPLE_RATE = 16000
GENDER_MAX_SECONDS = float(os.getenv("GENDER_MAX_SECONDS", "40"))
GENDER_WINDOW_SECONDS = float(os.getenv("GENDER_WINDOW_SECONDS", "5"))
# Ajustes del reetiquetado de turnos por embeddings de hablante.
TURN_EMBEDDING_MIN_SECONDS = float(os.getenv("DIARIZATION_MIN_TURN_SECONDS", "0.4"))
TURN_RELABEL_MARGIN = float(os.getenv("DIARIZATION_RELABEL_MARGIN", "0.05"))
TURN_MERGE_GAP_SECONDS = float(os.getenv("DIARIZATION_MERGE_GAP", "0.4"))
TURN_INHERIT_MAX_SECONDS = float(os.getenv("DIARIZATION_INHERIT_MAX", "2.0"))
# Duración mínima de un trozo de palabra para conservar un cambio de hablante.
MIN_SPEAKER_RUN_SECONDS = float(os.getenv("MIN_SPEAKER_RUN_SECONDS", "0.25"))
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


def _read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as audio:
        channels = audio.getnchannels()
        width = audio.getsampwidth()
        rate = audio.getframerate()
        if width != 2:
            raise ValueError(f"Se esperaba WAV de 16 bits, llegó {width * 8} bits.")
        raw = np.frombuffer(audio.readframes(audio.getnframes()), dtype=np.int16)
    data = raw.reshape(-1, channels).astype(np.float32)
    mono = data.mean(axis=1, dtype=np.float32) / 32768.0
    return mono, rate


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


@lru_cache(maxsize=1)
def _get_gender_model(device: str):
    """Load the pretrained voice-gender classifier once and keep it in memory."""
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    model_name = DEFAULT_GENDER_MODEL
    logger.info("Cargando clasificador de género: %s", model_name)
    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForAudioClassification.from_pretrained(model_name)
    model = model.to(device).eval()
    logger.info("Clasificador de género listo: %s", model_name)
    return extractor, model, model.config.id2label


def _release_gender_model() -> None:
    """Free the gender classifier before loading NLLB/Demucs.

    The XLS-R classifier is large and is only needed once per job. Keeping it
    cached while translation and separation run can exhaust 8 GB GPUs.
    """
    if _get_gender_model.cache_info().currsize:
        _get_gender_model.cache_clear()
        _free_gpu()



def _huggingface_token() -> str | None:
    return (
        os.getenv("HUGGINGFACE_API_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
    )


@lru_cache(maxsize=1)
def _get_diarization_pipeline(device: str):
    """Load the pyannote diarization pipeline once and keep it in memory."""
    import torch
    from pyannote.audio import Pipeline

    token = _huggingface_token()
    if not token:
        raise RuntimeError(
            "Falta HUGGINGFACE_API_TOKEN en .env para usar pyannote.audio."
        )

    models = [DEFAULT_DIARIZATION_MODEL]
    if LEGACY_DIARIZATION_MODEL not in models:
        models.append(LEGACY_DIARIZATION_MODEL)

    last_error: Exception | None = None
    for model_name in models:
        try:
            logger.info("Cargando modelo de diarización: %s", model_name)
            pipeline = Pipeline.from_pretrained(model_name, token=token)
            pipeline.to(torch.device(device))
            logger.info("Modelo de diarización listo: %s", model_name)
            return pipeline, model_name
        except Exception as error:  # gated repo, red, versión incompatible...
            last_error = error
            logger.warning(
                "No se pudo cargar %s: %s: %s",
                model_name,
                type(error).__name__,
                str(error).splitlines()[0][:300],
            )

    raise RuntimeError(
        "No se pudo cargar ningún modelo de diarización de pyannote. "
        "Acepta las condiciones en "
        f"https://hf.co/{DEFAULT_DIARIZATION_MODEL}, "
        "https://hf.co/pyannote/segmentation-3.0 y comprueba el token."
    ) from last_error


def _diarize(wav_path: Path, device: str) -> list[dict] | None:
    """Return speaker turns sorted by time, or ``None`` if pyannote fails.

    The audio is passed in memory as ``{"waveform", "sample_rate"}`` so the
    pipeline does not need torchcodec/FFmpeg to decode the WAV file.
    """
    try:
        import torch

        pipeline, model_name = _get_diarization_pipeline(device)
        audio, rate = _read_wav_mono(wav_path)
        waveform = torch.from_numpy(audio).unsqueeze(0)
        logger.info(
            "Diarizando %.1fs de audio con %s (%s).",
            len(audio) / rate,
            model_name,
            device,
        )
        with torch.inference_mode():
            kwargs = {}
            if os.getenv("PYANNOTE_MIN_SPEAKERS"):
                kwargs["min_speakers"] = int(os.environ["PYANNOTE_MIN_SPEAKERS"])
            if os.getenv("PYANNOTE_MAX_SPEAKERS"):
                kwargs["max_speakers"] = int(os.environ["PYANNOTE_MAX_SPEAKERS"])
            output = pipeline({"waveform": waveform, "sample_rate": rate}, **kwargs)
        annotation = getattr(output, "exclusive_speaker_diarization", output)
        turns = [
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
            }
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        ]
        del audio, waveform
    except Exception as error:
        logger.warning(
            "Diarización con pyannote falló (%s: %s); se usará la detección de "
            "tono por segmento.",
            type(error).__name__,
            str(error).splitlines()[0][:300],
        )
        return None

    if not turns:
        logger.warning("pyannote no devolvió ningún turno de habla.")
        return None
    speakers = sorted({turn["speaker"] for turn in turns})
    logger.info(
        "Diarización completada: %d hablante(s) %s en %d turno(s).",
        len(speakers),
        ", ".join(speakers),
        len(turns),
    )
    return turns


def _turn_speaker_at(seconds: float, turns: list[dict]) -> str | None:
    """Return the speaker whose turn contains ``seconds``."""
    for turn in turns:
        if turn["start"] <= seconds <= turn["end"]:
            return turn["speaker"]
    return None


def _turn_speaker_for_span(start: float, end: float, turns: list[dict]) -> str | None:
    """Return the speaker with the largest overlap with ``[start, end]``."""
    best_speaker, best_overlap = None, 0.0
    for turn in turns:
        overlap = min(end, turn["end"]) - max(start, turn["start"])
        if overlap > best_overlap:
            best_speaker, best_overlap = turn["speaker"], overlap
    return best_speaker


def _relabel_turns(turns: list[dict], wav_path: Path, device: str) -> list[dict]:
    """Re-label pyannote turns using pyannote's own speaker embeddings.

    Diarization can emit very short turns with inconsistent labels, especially
    for soft or older voices. We anchor each speaker on its longest turns and
    reassign every other turn by cosine similarity. Short or ambiguous turns
    inherit the nearest reliable turn. Finally, adjacent turns of the same
    speaker separated by a small gap are merged.
    """
    if len({turn["speaker"] for turn in turns}) < 2:
        return turns

    try:
        import torch
        import torchaudio.functional as F

        pipeline, _ = _get_diarization_pipeline(device)
        embedding_model = getattr(pipeline, "_embedding", None)
        if embedding_model is None:
            logger.info("pyannote no expone embeddings; se conservan sus turnos.")
            return turns
        audio, rate = _read_wav_mono(wav_path)
    except Exception as error:  # modelo no disponible, memoria, etc.
        logger.warning(
            "No se pudo preparar el reetiquetado por embeddings (%s: %s); "
            "se conservan los turnos de pyannote.",
            type(error).__name__,
            str(error).splitlines()[0][:300],
        )
        return turns

    labels = sorted({turn["speaker"] for turn in turns})
    embeddings: dict[int, np.ndarray | None] = {}
    for index, turn in enumerate(turns):
        duration = turn["end"] - turn["start"]
        start = max(0, int(turn["start"] * rate))
        end = min(len(audio), int(turn["end"] * rate))
        if duration < TURN_EMBEDDING_MIN_SECONDS or end <= start:
            embeddings[index] = None
            continue
        tensor = torch.from_numpy(audio[start:end]).unsqueeze(0)
        if rate != GENDER_SAMPLE_RATE:
            tensor = F.resample(tensor, rate, GENDER_SAMPLE_RATE)
        try:
            with torch.inference_mode():
                vector = np.asarray(embedding_model(tensor.unsqueeze(0))[0]).ravel()
        except Exception as error:
            logger.warning(
                "No se pudo calcular el embedding del turno %.2f-%.2f (%s: %s).",
                turn["start"],
                turn["end"],
                type(error).__name__,
                str(error).splitlines()[0][:200],
            )
            embeddings[index] = None
            continue
        embeddings[index] = vector / (np.linalg.norm(vector) + 1e-9)

    # Referencia de cada hablante: sus turnos más largos (hasta 20 s).
    references: dict[str, np.ndarray] = {}
    for label in labels:
        items = sorted(
            (
                (embeddings[index], turn["end"] - turn["start"])
                for index, turn in enumerate(turns)
                if turn["speaker"] == label and embeddings.get(index) is not None
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        selected: list[np.ndarray] = []
        total = 0.0
        for vector, duration in items:
            selected.append(vector)
            total += duration
            if total >= 20.0 or len(selected) >= 8:
                break
        if selected:
            mean = np.mean(selected, axis=0)
            references[label] = mean / (np.linalg.norm(mean) + 1e-9)

    if len(references) < 2:
        logger.info("No hay suficientes turnos largos para reetiquetar hablantes.")
        return turns

    final_labels: dict[int, str] = {}
    reliable: list[int] = []
    changed = 0
    for index, turn in enumerate(turns):
        vector = embeddings.get(index)
        if vector is None:
            continue
        scores = {
            label: float(np.dot(vector, reference))
            for label, reference in references.items()
        }
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best, top = ordered[0]
        margin = top - ordered[1][1] if len(ordered) > 1 else 1.0
        if margin >= TURN_RELABEL_MARGIN:
            final_labels[index] = best
            reliable.append(index)
            if best != turn["speaker"]:
                changed += 1

    if not reliable:
        logger.info("Los embeddings no separaron con claridad; se conservan los turnos.")
        return turns

    for index, turn in enumerate(turns):
        if index in final_labels:
            continue
        middle = (turn["start"] + turn["end"]) / 2.0
        nearest_index, nearest_distance = None, float("inf")
        for other_index in reliable:
            other = turns[other_index]
            if other["start"] <= middle <= other["end"]:
                distance = 0.0
            else:
                distance = min(
                    abs(middle - other["start"]), abs(middle - other["end"])
                )
            if distance < nearest_distance:
                nearest_index, nearest_distance = other_index, distance
        if nearest_index is not None and nearest_distance <= TURN_INHERIT_MAX_SECONDS:
            final_labels[index] = final_labels[nearest_index]
        else:
            final_labels[index] = turn["speaker"]

    merged: list[dict] = []
    for index in sorted(range(len(turns)), key=lambda item: turns[item]["start"]):
        speaker = final_labels[index]
        start, end = turns[index]["start"], turns[index]["end"]
        if (
            merged
            and merged[-1]["speaker"] == speaker
            and start - merged[-1]["end"] <= TURN_MERGE_GAP_SECONDS
        ):
            merged[-1]["end"] = max(merged[-1]["end"], end)
        else:
            merged.append({"start": start, "end": end, "speaker": speaker})

    logger.info(
        "Turnos reetiquetados por embeddings: %d/%d corregidos, %d -> %d turnos.",
        changed,
        len(turns),
        len(turns),
        len(merged),
    )
    return merged


def _classify_speaker_genders(
    wav_path: Path,
    turns: list[dict],
    device: str,
) -> dict[str, str]:
    """Classify male/female for every diarized speaker using its own audio.

    This runs *after* pyannote has separated the speakers. For each speaker we
    concatenate the longest turns (up to ``GENDER_MAX_SECONDS``) and classify
    them in windows, so the prediction is based on a single clean voice instead
    of guessing from short, possibly mixed, segments.
    """
    if not turns:
        return {}

    try:
        import torch
        import torchaudio.functional as F

        extractor, model, labels = _get_gender_model(device)
        audio, rate = _read_wav_mono(wav_path)
    except Exception as error:
        logger.warning(
            "No se pudo cargar el clasificador de género (%s: %s); "
            "se usará el tono por segmento.",
            type(error).__name__,
            str(error).splitlines()[0][:300],
        )
        return {}

    label_items = sorted(labels.items())
    position_by_label = {
        str(name).lower(): position for position, (_, name) in enumerate(label_items)
    }
    female_index = position_by_label.get("female")
    male_index = position_by_label.get("male")
    if female_index is None or male_index is None:
        logger.warning("El clasificador de género no expone female/male: %s", labels)
        return {}

    by_speaker: dict[str, list[dict]] = {}
    for turn in turns:
        by_speaker.setdefault(turn["speaker"], []).append(turn)

    window_size = max(1, int(GENDER_SAMPLE_RATE * GENDER_WINDOW_SECONDS))
    minimum = max(1, int(GENDER_SAMPLE_RATE * 0.5))
    result: dict[str, str] = {}
    for speaker, items in sorted(by_speaker.items()):
        ordered = sorted(items, key=lambda turn: turn["end"] - turn["start"], reverse=True)
        chunks: list[np.ndarray] = []
        total = 0.0
        for turn in ordered:
            start = max(0, int(turn["start"] * rate))
            end = min(len(audio), int(turn["end"] * rate))
            if end - start < rate // 2:
                continue
            chunks.append(audio[start:end])
            total += (end - start) / rate
            if total >= GENDER_MAX_SECONDS:
                break
        if not chunks:
            logger.info(
                "Hablante %s sin audio suficiente para clasificar género.", speaker
            )
            continue

        samples = np.concatenate(chunks)
        if rate != GENDER_SAMPLE_RATE:
            samples = F.resample(
                torch.from_numpy(samples), rate, GENDER_SAMPLE_RATE
            ).numpy()
        probabilities = np.zeros(len(label_items), dtype=np.float64)
        used = 0.0
        for start in range(0, max(1, len(samples) - window_size + 1), window_size):
            chunk = samples[start:start + window_size]
            if len(chunk) < minimum:
                continue
            inputs = extractor(
                chunk, sampling_rate=GENDER_SAMPLE_RATE, return_tensors="pt"
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                batch = torch.softmax(model(**inputs).logits, dim=-1)[0].cpu().numpy()
            weight = len(chunk) / GENDER_SAMPLE_RATE
            probabilities += batch * weight
            used += weight
        if used <= 0:
            continue
        probabilities /= used
        gender = (
            "female"
            if probabilities[female_index] >= probabilities[male_index]
            else "male"
        )
        result[speaker] = gender
        detail = ", ".join(
            f"{name}={probabilities[position]:.3f}"
            for position, (_, name) in enumerate(label_items)
        )
        logger.info(
            "Hablante %s -> %s (%s, %.1fs clasificados)",
            speaker,
            gender,
            detail,
            used,
        )

    del audio
    return result


def _assign_speakers(segments: list[dict], turns: list[dict]) -> None:
    """Attach to each Whisper segment the pyannote speaker with most overlap."""
    for segment in segments:
        start, end = segment["start"], segment["end"]
        best_speaker: str | None = None
        best_overlap = 0.0
        for turn in turns:
            overlap = min(end, turn["end"]) - max(start, turn["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]
        segment["speaker"] = best_speaker

    # Segments between turns (or without overlap) take the nearest one.
    known = sorted(
        (index for index, segment in enumerate(segments) if segment.get("speaker")),
        key=lambda index: segments[index]["start"],
    )
    if not known:
        for segment in segments:
            segment["speaker"] = turns[0]["speaker"]
        return
    for index, segment in enumerate(segments):
        if segment.get("speaker"):
            continue
        middle = (segment["start"] + segment["end"]) / 2
        nearest = min(
            known,
            key=lambda other: abs(
                (segments[other]["start"] + segments[other]["end"]) / 2 - middle
            ),
        )
        segment["speaker"] = segments[nearest]["speaker"]


def _consolidate_runs(runs: list[dict]) -> list[dict]:
    """Join adjacent word runs that ended up with the same speaker."""
    result: list[dict] = []
    for run in runs:
        if result and result[-1]["speaker"] == run["speaker"]:
            result[-1]["words"].extend(run["words"])
        else:
            result.append(run)
    return result


def _smooth_short_runs(runs: list[dict], min_seconds: float) -> list[dict]:
    """Merge word runs shorter than ``min_seconds`` into a neighbouring run.

    A single spurious word at a turn boundary should not switch the TTS voice
    in the middle of a sentence. Genuine short interjections (e.g. a 0.4 s
    "Yeah") are longer than the default 0.25 s and are preserved.
    """
    while len(runs) > 1:
        merged = False
        for index, run in enumerate(runs):
            start = run["words"][0]["start"]
            end = run["words"][-1]["end"]
            if end - start >= min_seconds:
                continue
            previous = runs[index - 1] if index > 0 else None
            following = runs[index + 1] if index + 1 < len(runs) else None
            if previous is None and following is None:
                break
            if previous is not None and (
                following is None
                or len(previous["words"]) >= len(following["words"])
            ):
                previous["words"].extend(run["words"])
            else:
                following["words"][:0] = run["words"]
            runs.pop(index)
            merged = True
            break
        if not merged:
            break
    return _consolidate_runs(runs)


def _split_segments_by_speaker(
    segments: list[dict],
    turns: list[dict],
) -> list[dict]:
    """Split Whisper segments at pyannote turn boundaries.

    Whisper often returns long segments that contain several speakers. Without
    this step the whole segment gets a single voice, which is exactly what made
    a female voice read both sides of a conversation. Word timestamps let us
    cut the segment where the diarized speaker changes.
    """
    if not turns:
        return segments

    result: list[dict] = []
    for segment in segments:
        words = segment.get("words") or []
        if len(words) < 2:
            result.append(
                {key: value for key, value in segment.items() if key != "words"}
            )
            continue

        runs: list[dict] = []
        for word in words:
            middle = (word["start"] + word["end"]) / 2.0
            speaker = _turn_speaker_at(middle, turns)
            if speaker is None:
                speaker = _turn_speaker_for_span(word["start"], word["end"], turns)
            if speaker is None:
                speaker = runs[-1]["speaker"] if runs else segment.get("speaker")
            if not runs or runs[-1]["speaker"] != speaker:
                runs.append({"speaker": speaker, "words": [word]})
            else:
                runs[-1]["words"].append(word)

        runs = _smooth_short_runs(runs, MIN_SPEAKER_RUN_SECONDS)
        if len(runs) <= 1:
            item = {key: value for key, value in segment.items() if key != "words"}
            if runs and runs[0]["speaker"]:
                item["speaker"] = runs[0]["speaker"]
            result.append(item)
            continue

        for run in runs:
            text = "".join(word["word"] for word in run["words"]).strip()
            if not text:
                continue
            result.append(
                {
                    "start": float(run["words"][0]["start"]),
                    "end": float(run["words"][-1]["end"]),
                    "text": text,
                    "speaker": run["speaker"],
                }
            )

    logger.info(
        "Segmentos partidos por hablante: %d -> %d subsegmentos.",
        len(segments),
        len(result),
    )
    return result


def _apply_gender_map(
    segments: list[dict],
    speaker_genders: dict[str, str],
) -> None:
    """Apply one gender per diarized speaker to all of its segments."""
    for segment in segments:
        gender = speaker_genders.get(segment.get("speaker"))
        if gender:
            segment["gender"] = gender


def _apply_speaker_genders(segments: list[dict]) -> dict[str, str]:
    """Replace per-segment genders with one gender per diarized speaker."""
    pitches: dict[str, list[float]] = {}
    segment_genders: dict[str, list[str]] = {}
    for segment in segments:
        speaker = segment.get("speaker")
        if not speaker:
            continue
        if segment.get("pitch_hz"):
            pitches.setdefault(speaker, []).append(segment["pitch_hz"])
        if segment.get("gender"):
            segment_genders.setdefault(speaker, []).append(segment["gender"])

    speakers = sorted({segment["speaker"] for segment in segments if segment.get("speaker")})
    speaker_genders: dict[str, str] = {}
    for speaker in speakers:
        values = pitches.get(speaker) or []
        if values:
            median = float(np.median(values))
            gender = "male" if median < MALE_PITCH_THRESHOLD_HZ else "female"
            detail = f"pitch_mediana={median:.1f}Hz"
        else:
            genders = segment_genders.get(speaker) or []
            gender = max(set(genders), key=genders.count) if genders else "female"
            detail = "sin_pitch"
        speaker_genders[speaker] = gender
        logger.info("Hablante %s -> %s (%s)", speaker, gender, detail)

    for segment in segments:
        gender = speaker_genders.get(segment.get("speaker"))
        if gender:
            segment["gender"] = gender
    return speaker_genders


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
        word_timestamps=True,
    )
    result = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        words = [
            {
                "start": float(word.start),
                "end": float(word.end),
                "word": word.word,
            }
            for word in (segment.words or [])
            if word.start is not None and word.end is not None
        ]
        result.append(
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": text,
                "words": words,
            }
        )
    return result


def _estimate_pitch_hz(
    samples: np.ndarray,
    rate: int,
    min_hz: float = 75.0,
    max_hz: float = 350.0,
) -> float | None:
    """Estimate the median fundamental frequency of a short audio slice.

    Uses frame-wise autocorrelation (via FFT) and ignores unvoiced/silent
    frames, which is enough to separate typical male and female voices.
    """
    frame_len = int(rate * 0.04)
    if len(samples) < frame_len:
        return None
    hop = max(1, int(rate * 0.02))
    frame_count = 1 + (len(samples) - frame_len) // hop
    indexes = (
        np.linspace(0, frame_count - 1, 80).astype(int)
        if frame_count > 80
        else np.arange(frame_count)
    )
    frames = np.stack([
        samples[index * hop:index * hop + frame_len] for index in indexes
    ]).astype(np.float32)
    frames -= frames.mean(axis=1, keepdims=True)
    frames *= np.hanning(frame_len).astype(np.float32)

    energy = np.sqrt(np.mean(frames ** 2, axis=1))
    if not len(energy) or energy.max() < 1e-5:
        return None
    frames = frames[energy > max(1e-5, energy.max() * 0.05)]
    if not len(frames):
        return None

    size = 1 << (2 * frame_len - 1).bit_length()
    spectrum = np.fft.rfft(frames, n=size, axis=1)
    autocorrelation = np.fft.irfft(
        spectrum * np.conj(spectrum), n=size, axis=1
    )[:, :frame_len]
    autocorrelation /= autocorrelation[:, :1] + 1e-9

    min_lag = max(1, int(rate / max_hz))
    max_lag = min(frame_len - 1, int(rate / min_hz))
    if max_lag <= min_lag:
        return None
    window = autocorrelation[:, min_lag:max_lag + 1]
    peak_values = window.max(axis=1)
    peak_lags = window.argmax(axis=1) + min_lag

    voiced = peak_values >= 0.3
    if not voiced.any():
        return None
    frequencies = rate / peak_lags[voiced]
    frequencies = frequencies[(frequencies >= min_hz) & (frequencies <= max_hz)]
    if not len(frequencies):
        return None
    return float(np.median(frequencies))


def _analyze_voices(wav_path: Path, segments: list[dict]) -> None:
    """Add ``gender`` and ``pitch_hz`` to each segment from its own audio."""
    audio, rate = _read_wav_mono(wav_path)
    previous_gender: str | None = None
    for segment in segments:
        start = max(0, int(segment["start"] * rate))
        end = min(len(audio), int(segment["end"] * rate))
        pitch = _estimate_pitch_hz(audio[start:end], rate) if end > start else None
        if pitch is None:
            segment["pitch_hz"] = None
            segment["gender"] = previous_gender
        else:
            gender = "male" if pitch < MALE_PITCH_THRESHOLD_HZ else "female"
            segment["pitch_hz"] = round(pitch, 1)
            segment["gender"] = gender
            previous_gender = gender

    # Leading segments without pitch inherit the first detected gender.
    first_known = next((item["gender"] for item in segments if item["gender"]), None)
    for segment in segments:
        if not segment.get("gender"):
            segment["gender"] = first_known

    del audio
    males = sum(1 for item in segments if item.get("gender") == "male")
    females = sum(1 for item in segments if item.get("gender") == "female")
    unknowns = len(segments) - males - females
    logger.info(
        "Voces detectadas por segmento: male=%d female=%d unknown=%d", males, females, unknowns
    )


def _assign_voices(
    segments: list[dict],
    target_language: str,
    override_voice: str | None = None,
) -> None:
    """Choose a TTS voice per segment.

    Every diarized speaker gets its own voice inside the pool of its detected
    gender, so two male speakers do not share the same voice. If the gender
    classifier cannot be used, the pitch-based gender is the safety net.
    """
    if override_voice:
        resolved = DEFAULT_TTS_VOICES.get(override_voice, override_voice)
        for segment in segments:
            segment["tts_voice"] = resolved
        return

    pools = TTS_VOICE_POOLS.get(target_language) or {}

    def pool_for(gender: str | None) -> list[str]:
        return (
            pools.get(gender or "female")
            or pools.get("female")
            or [FALLBACK_TTS_VOICE]
        )

    speaker_genders: dict[str, str] = {}
    speaker_order: list[str] = []
    for segment in segments:
        speaker = segment.get("speaker")
        if not speaker:
            continue
        if speaker not in speaker_genders:
            speaker_genders[speaker] = segment.get("gender") or "female"
            speaker_order.append(speaker)

    used = {"male": 0, "female": 0}
    speaker_voices: dict[str, str] = {}
    for speaker in speaker_order:
        gender = speaker_genders[speaker]
        pool = pool_for(gender)
        index = used.get(gender, 0)
        speaker_voices[speaker] = pool[index % len(pool)]
        used[gender] = index + 1

    for segment in segments:
        speaker = segment.get("speaker")
        if speaker and speaker in speaker_voices:
            segment["tts_voice"] = speaker_voices[speaker]
        else:
            segment["tts_voice"] = pool_for(segment.get("gender"))[0]

    if speaker_voices:
        logger.info(
            "Voces por hablante: %s",
            ", ".join(f"{speaker}={voice}" for speaker, voice in speaker_voices.items()),
        )


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


def _tts(segments: list[dict], clips_dir: Path, concurrency: int) -> None:
    import edge_tts

    voices = sorted({segment.get("tts_voice") or FALLBACK_TTS_VOICE for segment in segments})
    logger.info("Generando TTS con %d voz(es): %s", len(voices), ", ".join(voices))
    shutil.rmtree(clips_dir, ignore_errors=True)
    clips_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    async def render() -> None:
        semaphore = asyncio.Semaphore(concurrency)

        async def render_one(index: int, text: str, voice: str) -> None:
            async with semaphore:
                last_error = None
                for attempt in range(3):
                    try:
                        await edge_tts.Communicate(text, voice).save(
                            str(clips_dir / f"{index:04d}.mp3")
                        )
                        return
                    except Exception as error:
                        last_error = error
                        await asyncio.sleep(0.6 * (attempt + 1))
                failures.append(f"{index}: {last_error}")

        await asyncio.gather(*(
            render_one(
                index,
                segment["text_translated"],
                segment.get("tts_voice") or FALLBACK_TTS_VOICE,
            )
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
    voice: str | None = None,
    output_dir: str | Path | None = None,
    max_seconds: int | None = None,
    whisper_model: str = "small",
    device: str | None = None,
    on_progress: ProgressCallback | None = None,
) -> tuple[Path, list[dict]]:
    """Process a complete video and return its dubbed MP3 and timed segments.

    Models are cached by configuration, so subsequent calls reuse the same
    in-memory models. ``on_progress`` receives a stage name and a 0..1 value.
    Diarization and gender classification always run: pyannote separates the
    speakers, XLS-R chooses the gender pool and every speaker receives a
    different voice. A pitch-based path exists only as a safety net if a model
    cannot be loaded.
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

    _progress(on_progress, "diarization", 0.0)
    turns = _diarize(wav_path, active_device)
    if turns:
        turns = _relabel_turns(turns, wav_path, active_device)
        _assign_speakers(segments, turns)
    _progress(on_progress, "diarization", 1.0)

    _progress(on_progress, "voices", 0.0)
    if turns:
        segments = _split_segments_by_speaker(segments, turns)
    _analyze_voices(wav_path, segments)
    if turns and not voice:
        speaker_genders = _classify_speaker_genders(wav_path, turns, active_device)
        if speaker_genders:
            _apply_gender_map(segments, speaker_genders)
        else:
            _apply_speaker_genders(segments)
    _assign_voices(segments, target_language, voice)
    # El clasificador XLS-R ya no se necesita; liberamos su VRAM antes de NLLB
    # y Demucs, que son los siguientes en cargarse.
    _release_gender_model()
    # Los timestamps por palabra solo se usan para partir por hablante; no
    # deben viajar en la respuesta de la API.
    for segment in segments:
        segment.pop("words", None)
    _progress(on_progress, "voices", 1.0)

    _progress(on_progress, "translation", 0.0)
    _translate(segments, source_language, target_language, active_device, batch_size)
    _progress(on_progress, "translation", 1.0)

    _progress(on_progress, "tts", 0.0)
    clips_dir = output_path / "clips"
    _tts(segments, clips_dir, int(os.getenv("TTS_CONCURRENCY", "8")))
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
    _get_gender_model(device)
    # La diarización también queda lista desde el arranque: en un despliegue
    # que se despierta con la primera petición, el usuario no espera a que se
    # descargue y cargue pyannote.
    _get_diarization_pipeline(device)
    logger.info("Modelos precargados en %s.", device)