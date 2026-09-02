import io
import wave

import numpy as np
from fastapi.testclient import TestClient

import app.main as main


def _wav() -> bytes:
    target = io.BytesIO()
    with wave.open(target, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(np.zeros(8_000, dtype="<i2").tobytes())
    return target.getvalue()


def test_status_accepts_google_api_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("AIC_SDK_LICENSE", "test-license")

    response = TestClient(main.app).get("/api/status")

    assert response.json() == {
        "google": True,
        "live": True,
        "ai_coustics": True,
        "project": "API key",
    }


def test_status_reports_missing_google_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("AIC_SDK_LICENSE", "test-license")

    body = TestClient(main.app).get("/api/status").json()

    assert body["google"] is False
    assert body["live"] is False
    assert body["project"] == "Not configured"


def test_service_worker_is_neutralized():
    response = TestClient(main.app).get("/sw.js")

    assert response.status_code == 200
    assert "unregister" in response.text
    assert response.headers["cache-control"] == "no-store"


def test_invalid_api_key_error_is_actionable():
    error = RuntimeError("400 INVALID_ARGUMENT API key not valid. Please pass a valid API key.")

    result = main._public_google_error(error)

    assert "not a valid Gemini API key" in result
    assert "aistudio.google.com/apikey" in result


def test_model_not_found_error_is_actionable():
    error = RuntimeError(
        "404 NOT_FOUND models/gemini-3.5-transcribe is not found for API version v1beta"
    )

    assert "not available to this API key" in main._public_google_error(error)


def test_rate_limit_error_is_actionable():
    error = RuntimeError("429 RESOURCE_EXHAUSTED quota exceeded")

    assert "rate limit or quota" in main._public_google_error(error)


def test_permission_denied_error_is_actionable():
    error = RuntimeError("403 PERMISSION_DENIED")

    assert "Gemini 3.5 Transcribe" in main._public_google_error(error)


def test_unknown_error_falls_back():
    assert "server log" in main._public_google_error(RuntimeError("boom"))


def test_compare_runs_both_paths(monkeypatch, tmp_path):
    calls = []

    def fake_enhance(audio, level):
        return audio, {
            "model": "quail-vf-2.2-l-16khz",
            "enhancement_level": level,
            "audio_delay_ms": 30,
            "processing_ms": 4,
        }

    def fake_transcribe(audio, *options):
        calls.append(len(audio.samples))
        return {"text": "hello", "segments": [], "words": [], "elapsed_ms": 5}

    monkeypatch.setattr(main, "JOBS_DIR", tmp_path)
    monkeypatch.setattr(main, "enhance_with_quail", fake_enhance)
    monkeypatch.setattr(main, "transcribe_with_gemini", fake_transcribe)

    response = TestClient(main.app).post(
        "/api/compare",
        files={"audio_file": ("sample.wav", _wav(), "audio/wav")},
        data={"enhancement_level": "0.8"},
    )

    assert response.status_code == 200
    assert sorted(calls) == [8_000, 8_000]
    assert response.json()["raw"]["text"] == "hello"
    assert response.json()["quail"]["enhancement_level"] == 0.8
