# Translate Video AI

Monorepositorio con una extensión de Chrome (`extension/`) y una API FastAPI
(`backend/`). La API descarga temporalmente el audio con `yt-dlp`, lo
transcribe localmente con `faster-whisper`, lo traduce con NLLB y devuelve el
resultado a la extensión. La extensión reproduce la traducción con
`SpeechSynthesis`.

## Backend local

1. Crea un entorno virtual e instala las dependencias:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r backend\requirements.txt
   ```

2. Inicia la API:

   ```powershell
   python -m uvicorn backend.main:app --reload --port 8000
   ```

   La primera ejecución puede tardar mientras se descargan los modelos de
   Whisper y NLLB.

## Extensión

1. Abre `chrome://extensions`, activa **Developer mode** y selecciona la
   carpeta `extension/` con **Load unpacked**.
2. Abre un video de YouTube y pulsa **Procesar y traducir**.

### Ver logs de diagnóstico

- Backend: revisa la terminal donde ejecutaste `python -m uvicorn ...`.
- Background: en `chrome://extensions`, abre **Errors** o **Inspect views**
  para el service worker.
- Content script: abre DevTools (`F12`) en la pestaña de YouTube.

Los mensajes usan el prefijo `[Translate Video AI]`.
