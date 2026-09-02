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


def test_compare_runs_both_paths(monkeypatch):
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


def test_comparison_audio_is_served_from_memory_and_expires(monkeypatch):
    def fake_enhance(audio, level):
        return audio, {"model": "quail", "enhancement_level": level, "audio_delay_ms": 30, "processing_ms": 4}

    monkeypatch.setattr(main, "enhance_with_quail", fake_enhance)
    monkeypatch.setattr(main, "transcribe_with_gemini",
                        lambda audio, *a: {"text": "hi", "segments": [], "words": [], "elapsed_ms": 5})
    main.rate_limiter = main.RateLimiter(per_ip=0, daily=0)
    main.audio_store = main.AudioStore()
    client = TestClient(main.app)

    job_id = client.post("/api/compare", files={"audio_file": ("s.wav", _wav(), "audio/wav")}).json()["job_id"]

    for name in ("original.wav", "quail.wav"):
        served = client.get(f"/audio/{job_id}/{name}")
        assert served.status_code == 200
        assert served.headers["content-type"] == "audio/wav"

    # the uploaded source is never retrievable, and nothing survives expiry
    assert client.get(f"/audio/{job_id}/source.wav").status_code == 404
    main.audio_store = main.AudioStore(ttl=0)
    assert client.get(f"/audio/{job_id}/original.wav").status_code == 404


def test_compare_refuses_over_the_per_ip_limit(monkeypatch):
    monkeypatch.setattr(main, "enhance_with_quail",
                        lambda audio, level: (audio, {"model": "q", "enhancement_level": level,
                                                      "audio_delay_ms": 1, "processing_ms": 1}))
    monkeypatch.setattr(main, "transcribe_with_gemini",
                        lambda audio, *a: {"text": "hi", "segments": [], "words": [], "elapsed_ms": 1})
    main.rate_limiter = main.RateLimiter(per_ip=1, window=600, daily=0)
    client = TestClient(main.app)
    files = {"audio_file": ("s.wav", _wav(), "audio/wav")}

    assert client.post("/api/compare", files=files).status_code == 200
    second = client.post("/api/compare", files={"audio_file": ("s.wav", _wav(), "audio/wav")})
    assert second.status_code == 429
    assert "Too many comparisons" in second.json()["detail"]


def test_daily_budget_is_enforced_across_addresses():
    main.rate_limiter = main.RateLimiter(per_ip=0, daily=1)
    assert main.rate_limiter.check("1.1.1.1") is None
    assert "daily limit" in main.rate_limiter.check("2.2.2.2")


def test_client_ip_prefers_the_proxy_header():
    assert main.client_ip({"x-forwarded-for": "9.9.9.9, 10.0.0.1"}, "127.0.0.1") == "9.9.9.9"
    assert main.client_ip({}, "127.0.0.1") == "127.0.0.1"
