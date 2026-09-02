from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from app.audio import AudioError, decode_wav, encode_wav
from app.services import (
    LiveQuailProcessor,
    LiveTytoAnalyzer,
    analyze_with_tyto,
    enhance_with_quail,
    gemini_client,
    gemini_live_client,
    transcribe_with_gemini,
)


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
JOBS_DIR = BASE_DIR / "data" / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = 150 * 1024 * 1024
RETENTION_SECONDS = 24 * 60 * 60
comparison_slots = asyncio.Semaphore(2)


def _cleanup_old_jobs() -> None:
    cutoff = time.time() - RETENTION_SECONDS
    for path in JOBS_DIR.iterdir():
        if path.is_dir() and path.stat().st_mtime < cutoff:
            shutil.rmtree(path)


_cleanup_old_jobs()

app = FastAPI(title="Gemini × Quail transcription lab")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/audio", StaticFiles(directory=JOBS_DIR), name="audio")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico")
def favicon():
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")


@app.get("/sw.js")
def neutral_service_worker():
    return Response(
        "self.addEventListener('install',()=>self.skipWaiting());"
        "self.addEventListener('activate',e=>e.waitUntil(self.registration.unregister()));",
        media_type="application/javascript",
        headers={"Service-Worker-Allowed": "/", "Cache-Control": "no-store"},
    )


@app.get("/api/status")
def status():
    google_api_key = bool(os.environ.get("GOOGLE_API_KEY"))
    return {
        "google": google_api_key,
        "live": google_api_key,
        "ai_coustics": bool(os.environ.get("AIC_SDK_LICENSE")),
        "project": "API key" if google_api_key else "Not configured",
    }


def _csv(value: str, limit: int) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()][:limit]


def _public_google_error(exc: Exception) -> str:
    message = str(exc)
    if "API key not valid" in message or "API_KEY_INVALID" in message:
        return (
            "GOOGLE_API_KEY is not a valid Gemini API key. "
            "Create one at https://aistudio.google.com/apikey."
        )
    if "NOT_FOUND" in message or "is not found for API version" in message:
        return (
            "Gemini 3.5 Transcribe is not available to this API key. "
            "Check the model name and that your key has access."
        )
    if "RESOURCE_EXHAUSTED" in message or "429" in message:
        return "Gemini rate limit or quota exceeded. Wait a moment and retry."
    if "PERMISSION_DENIED" in message:
        return "Google denied access. Check that the API key can use Gemini 3.5 Transcribe."
    return "Live transcription failed. Check the server log for details."


def _pcm16(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()


async def _receive_live_transcript(
    session: Any, websocket: WebSocket, path: str, send_lock: asyncio.Lock
) -> None:
    async for message in session.receive():
        server_content = getattr(message, "server_content", None)
        if not server_content:
            continue
        interim = getattr(server_content, "interim_input_transcription", None)
        if interim and getattr(interim, "text", None):
            async with send_lock:
                await websocket.send_json(
                    {"type": "transcript", "path": path, "final": False, "text": interim.text}
                )
        final = getattr(server_content, "input_transcription", None)
        if final and getattr(final, "text", None):
            async with send_lock:
                await websocket.send_json(
                    {"type": "transcript", "path": path, "final": True, "text": final.text}
                )


async def _publish_live_tyto(
    tyto: LiveTytoAnalyzer,
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    stop: asyncio.Event,
) -> None:
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=1)
            break
        except TimeoutError:
            pass
        if not tyto.ready:
            continue
        try:
            result = await asyncio.to_thread(tyto.analyze)
            if result:
                async with send_lock:
                    await websocket.send_json({"type": "tyto", **result})
        except Exception as exc:
            async with send_lock:
                await websocket.send_json(
                    {"type": "tyto_warning", "message": f"Tyto analysis paused: {exc}"}
                )


async def _run_tyto_only(
    websocket: WebSocket,
    tyto: LiveTytoAnalyzer,
    block_size: int,
    google_error: Exception,
) -> None:
    send_lock = asyncio.Lock()
    stop = asyncio.Event()
    task = asyncio.create_task(_publish_live_tyto(tyto, websocket, send_lock, stop))
    try:
        await websocket.send_json(
            {"type": "warning", "message": _public_google_error(google_error)}
        )
        await websocket.send_json(
            {
                "type": "status",
                "status": "ready",
                "text": "Tyto listening. First score after 5 seconds",
                "transcription_available": False,
            }
        )
        while True:
            incoming = await websocket.receive()
            if incoming["type"] == "websocket.disconnect":
                raise WebSocketDisconnect()
            if incoming.get("text"):
                command = json.loads(incoming["text"])
                if command.get("action") == "stop":
                    break
                continue
            audio_bytes = incoming.get("bytes")
            if not audio_bytes:
                continue
            samples = np.frombuffer(audio_bytes, dtype="<f4")
            if len(samples) == block_size:
                tyto.buffer(samples)
        await websocket.send_json({"type": "complete", "mode": "tyto"})
    finally:
        stop.set()
        await asyncio.gather(task, return_exceptions=True)


@app.websocket("/ws/live")
async def live_compare(websocket: WebSocket):
    from google.genai import types

    await websocket.accept()
    processor: LiveQuailProcessor | None = None
    tyto: LiveTytoAnalyzer | None = None
    tyto_task: asyncio.Task | None = None
    tyto_stop = asyncio.Event()
    receivers: list[asyncio.Task] = []
    try:
        options = await websocket.receive_json()
        sample_rate = int(options.get("sample_rate", 48_000))
        block_size = int(options.get("block_size", round(sample_rate * 0.015)))
        level = float(options.get("enhancement_level", 0.5))
        use_tyto = bool(options.get("tyto", True))
        if not 8_000 <= sample_rate <= 192_000 or not 1 <= block_size <= 8192:
            raise ValueError("Unsupported microphone audio format")
        if not 0 <= level <= 1:
            raise ValueError("Enhancement level must be between 0 and 1")

        loading = "Loading Quail and Tyto" if use_tyto else "Loading Quail"
        await websocket.send_json({"type": "status", "status": "loading", "text": loading})
        async with comparison_slots:
            processor = await asyncio.to_thread(
                LiveQuailProcessor, sample_rate, block_size, level
            )
            if use_tyto:
                tyto = await asyncio.to_thread(LiveTytoAnalyzer, sample_rate, block_size)
            client = gemini_live_client()
            transcription_options: dict[str, Any] = {}
            languages = _csv(str(options.get("language_codes", "")), 20)
            vocabulary = _csv(str(options.get("vocabulary", "")), 1_000)
            if languages:
                transcription_options["language_codes"] = languages
            if vocabulary:
                transcription_options["custom_vocabulary"] = vocabulary
            config = types.LiveConnectConfig(
                response_modalities=["TEXT"],
                input_audio_transcription=types.AudioTranscriptionConfig(
                    **transcription_options
                ),
            )
            model = "gemini-3.5-transcribe-live"
            try:
                async with (
                    client.aio.live.connect(model=model, config=config) as raw_session,
                    client.aio.live.connect(model=model, config=config) as enhanced_session,
                ):
                    if raw_session.setup_complete is None or enhanced_session.setup_complete is None:
                        raise RuntimeError("Gemini Live session setup did not complete")
                    send_lock = asyncio.Lock()
                    receivers = [
                        asyncio.create_task(
                            _receive_live_transcript(raw_session, websocket, "raw", send_lock)
                        ),
                        asyncio.create_task(
                            _receive_live_transcript(
                                enhanced_session, websocket, "enhanced", send_lock
                            )
                        ),
                    ]
                    if tyto is not None:
                        tyto_task = asyncio.create_task(
                            _publish_live_tyto(tyto, websocket, send_lock, tyto_stop)
                        )
                    await websocket.send_json(
                        {
                            "type": "status",
                            "status": "ready",
                            "text": "Listening",
                            "quail_delay_ms": processor.delay_ms,
                            "transcription_available": True,
                        }
                    )
                    mime_type = f"audio/pcm;rate={sample_rate}"
                    while True:
                        incoming = await websocket.receive()
                        if incoming["type"] == "websocket.disconnect":
                            raise WebSocketDisconnect()
                        if incoming.get("text"):
                            command = json.loads(incoming["text"])
                            if command.get("action") == "stop":
                                break
                            continue
                        audio_bytes = incoming.get("bytes")
                        if not audio_bytes:
                            continue
                        samples = np.frombuffer(audio_bytes, dtype="<f4")
                        if len(samples) != block_size:
                            continue
                        if tyto is not None:
                            tyto.buffer(samples)
                        enhanced = processor.process(samples)
                        await asyncio.gather(
                            raw_session.send_realtime_input(
                                audio=types.Blob(data=_pcm16(samples), mime_type=mime_type)
                            ),
                            enhanced_session.send_realtime_input(
                                audio=types.Blob(data=_pcm16(enhanced), mime_type=mime_type)
                            ),
                        )
                    await asyncio.gather(
                        raw_session.send_realtime_input(audio_stream_end=True),
                        enhanced_session.send_realtime_input(audio_stream_end=True),
                    )
                    tyto_stop.set()
                    if tyto_task is not None:
                        await asyncio.gather(tyto_task, return_exceptions=True)
                    done, pending = await asyncio.wait(receivers, timeout=5)
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*done, *pending, return_exceptions=True)
                    await websocket.send_json({"type": "complete"})
            except WebSocketDisconnect:
                raise
            except Exception as exc:
                if tyto is None or tyto_task is not None:
                    raise
                print(f"Gemini unavailable, continuing with Tyto: {type(exc).__name__}: {exc}")
                await _run_tyto_only(websocket, tyto, block_size, exc)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        print(f"Live transcription error: {type(exc).__name__}: {exc}")
        try:
            await websocket.send_json({"type": "error", "message": _public_google_error(exc)})
        except Exception:
            pass
    finally:
        tyto_stop.set()
        if tyto_task is not None and not tyto_task.done():
            tyto_task.cancel()
            await asyncio.gather(tyto_task, return_exceptions=True)
        for task in receivers:
            if not task.done():
                task.cancel()
        if processor is not None:
            try:
                await asyncio.to_thread(processor.close)
            except Exception:
                pass
        if tyto is not None:
            try:
                await asyncio.to_thread(tyto.close)
            except Exception:
                pass


async def _read_limited(upload: UploadFile) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while chunk := await upload.read(1024 * 1024):
        size += len(chunk)
        if size > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "WAV file is too large (150 MB maximum)")
        chunks.append(chunk)
    return b"".join(chunks)


@app.post("/api/compare")
async def compare(
    audio_file: UploadFile = File(...),
    enhancement_level: float = Form(0.5),
    language_codes: str = Form(""),
    vocabulary: str = Form(""),
    diarization: bool = Form(False),
    word_timestamps: bool = Form(False),
    tyto: bool = Form(False),
):
    _cleanup_old_jobs()
    if not 0 <= enhancement_level <= 1:
        raise HTTPException(400, "Enhancement level must be between 0 and 1")
    try:
        uploaded_bytes = await _read_limited(audio_file)
        raw_audio = decode_wav(uploaded_bytes)
    except AudioError as exc:
        raise HTTPException(400, str(exc)) from exc

    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir()
    source_path = job_dir / "source.wav"
    raw_path = job_dir / "original.wav"
    enhanced_path = job_dir / "quail.wav"
    source_path.write_bytes(uploaded_bytes)
    raw_path.write_bytes(encode_wav(raw_audio))

    try:
        async with comparison_slots, asyncio.timeout(180):
            enhanced_audio, quail = await asyncio.to_thread(
                enhance_with_quail, raw_audio, enhancement_level
            )
            enhanced_path.write_bytes(encode_wav(enhanced_audio))

            args = (
                _csv(language_codes, 20),
                _csv(vocabulary, 1_000),
                diarization,
                word_timestamps,
            )
            raw_task = asyncio.to_thread(transcribe_with_gemini, raw_audio, *args)
            enhanced_task = asyncio.to_thread(transcribe_with_gemini, enhanced_audio, *args)
            if tyto:
                raw_result, enhanced_result, tyto_result = await asyncio.gather(
                    raw_task, enhanced_task, asyncio.to_thread(analyze_with_tyto, raw_audio)
                )
            else:
                raw_result, enhanced_result = await asyncio.gather(raw_task, enhanced_task)
                tyto_result = None
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (job_dir / "error.txt").write_text(error)
        if "not configured" in str(exc):
            public_error = str(exc)
        elif isinstance(exc, TimeoutError):
            public_error = "Comparison timed out after 3 minutes"
        else:
            public_error = _public_google_error(exc)
        raise HTTPException(502, public_error) from exc

    result = {
        "job_id": job_id,
        "duration_seconds": round(raw_audio.duration_seconds, 2),
        "raw": {**raw_result, "audio_url": f"/audio/{job_id}/original.wav"},
        "enhanced": {**enhanced_result, "audio_url": f"/audio/{job_id}/quail.wav"},
        "quail": quail,
        "tyto": tyto_result,
    }
    (job_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result
