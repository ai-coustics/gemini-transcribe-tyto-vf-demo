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

The first enhanced run downloads `quail-vf-2.2-l-16khz` (~20 MB). Enabling Tyto downloads `tyto-1.1-l-16khz` (~13 MB). Uploaded audio is never written to disk. It is held in memory only long enough for the browser to play the A/B comparison back (10 minutes by default), then dropped.

## Notes

- Quail Voice Focus is intended for a single primary speaker. Its first few seconds are a warm-up period, so longer samples are more representative.
- Both paths use the same normalized 16 kHz mono control signal. The untouched uploaded WAV is retained alongside the run for auditability but is not sent directly to Gemini; this keeps Quail as the only treatment variable.
- Gemini's file endpoint currently limits recordings to 15 minutes when advanced transcription features are used. This app enforces that limit.
- VAD is deliberately not in this file-based path: VAD is useful for endpointing in a future live-streaming mode, but trimming a benchmark recording would make the comparison less fair.

## Privacy

Uploaded audio is never written to disk. `/api/compare` holds the decoded original and
the Quail-enhanced result in memory only so the browser can play the A/B comparison
back, and drops them after `AUDIO_TTL_SECONDS` (10 minutes by default). The bytes the
client uploaded are never re-served. Live mode persists nothing at all.

Audio is still sent to Google for transcription — twice per file comparison, and as a
stream in live mode. That is inherent to comparing Gemini output.

## Limits

Modal has no built-in per-IP rate limiting, so it lives in `app/limits.py`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `RATE_LIMIT_PER_IP` | 5 | Comparisons per address per window |
| `RATE_LIMIT_WINDOW_SECONDS` | 600 | Sliding window length |
| `RATE_LIMIT_DAILY` | 200 | Global daily budget across all callers |
| `AUDIO_TTL_SECONDS` | 600 | How long playback audio stays in memory |

Set any of them to `0` to disable it. Both `/api/compare` and `/ws/live` are limited —
a live session opens two Gemini Live streams, so it is not cheaper than a file run.

## Deploying to Modal

```bash
modal secret create aic-demo-secrets GOOGLE_API_KEY=... AIC_SDK_LICENSE=...
modal deploy modal_app.py
```

`modal_app.py` pins `max_containers=1`. The rate limiter keeps per-process state, so it
is only accurate while one container serves every request. If you raise that, move the
limiter to a `modal.Dict` first or the limits become per-container.
