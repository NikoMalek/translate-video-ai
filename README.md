# Translate Video AI

Monorepositorio con una extensión de Chrome (`extension/`) y una API FastAPI
(`backend/`). La API descarga temporalmente el audio con `yt-dlp`, lo
transcribe localmente con `faster-whisper` y devuelve el texto a la extensión.
La extensión traduce el texto con NLLB y lo reproduce con `SpeechSynthesis`.

## Backend local

1. Crea un entorno virtual e instala las dependencias:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r backend\requirements.txt
   ```

2. Inicia la API:

   ```powershell
   uvicorn backend.main:app --reload --port 8000
   ```

## Extensión

1. Instala y compila desde `extension/`:

   ```powershell
   npm install --prefix extension
   npm run build --prefix extension
   ```

2. Abre `chrome://extensions`, activa **Developer mode** y selecciona la
   carpeta `extension/` con **Load unpacked**.
3. Abre un video de YouTube y pulsa **Procesar y traducir**.

