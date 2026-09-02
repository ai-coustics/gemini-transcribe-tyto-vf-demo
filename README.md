# Gemini × Quail transcription lab

Compare speech through two paths:

1. original audio → Gemini 3.5 Transcribe
2. original audio → Quail Voice Focus 2.2 → Gemini 3.5 Transcribe

Live mode streams microphone PCM to two Gemini 3.5 Transcribe Live sessions while Quail Voice Focus processes the second path locally. Both transcript panels update as you speak. Optional Tyto Audio Insight scores the original microphone signal after a five-second warm-up and refreshes once per second. File mode accepts a WAV upload, runs both Gemini requests concurrently, and adds playback, optional WER scoring, diarization, timestamps, and Tyto analysis.

## Setup

Prerequisites: Python 3.11 to 3.13, a Gemini API key with access to Gemini 3.5 Transcribe, and an ai-coustics SDK license.

```bash
cp .env.example .env
# Edit .env with GOOGLE_API_KEY and AIC_SDK_LICENSE.
# Create the Google key at https://aistudio.google.com/apikey
set -a; source .env; set +a
uv sync --extra dev
uv run uvicorn app.main:app --reload
```

The app uses the Gemini Developer API (`generativelanguage.googleapis.com`) with models
`gemini-3.5-transcribe` for file compare and `gemini-3.5-transcribe-live` for `/ws/live`.
A single AI Studio API key covers both; no Google Cloud project, IAM role, or `gcloud` setup is needed.

Open <http://127.0.0.1:8000>.

If Google returns `API key not valid`, recreate the key at <https://aistudio.google.com/apikey>. If it returns `NOT_FOUND` for the model, your key does not yet have access to Gemini 3.5 Transcribe.

The first enhanced run downloads `quail-vf-2.2-l-16khz` (~20 MB). Enabling Tyto downloads `tyto-1.1-l-16khz` (~13 MB). Audio and results are stored under `data/jobs/` for local evaluation, ignored by Git, and automatically removed after 24 hours the next time the app starts.

## Notes

- Quail Voice Focus is intended for a single primary speaker. Its first few seconds are a warm-up period, so longer samples are more representative.
- Both paths use the same normalized 16 kHz mono control signal. The untouched uploaded WAV is retained alongside the run for auditability but is not sent directly to Gemini; this keeps Quail as the only treatment variable.
- Gemini's file endpoint currently limits recordings to 15 minutes when advanced transcription features are used. This app enforces that limit.
- VAD is deliberately not in this file-based path: VAD is useful for endpointing in a future live-streaming mode, but trimming a benchmark recording would make the comparison less fair.
