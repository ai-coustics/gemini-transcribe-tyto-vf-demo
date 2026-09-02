import io
import wave

import numpy as np
import pytest

from app.audio import Audio, AudioError, decode_wav, encode_wav


def test_wav_round_trip():
    samples = np.sin(np.linspace(0, 30, 16_000, dtype=np.float32)) * 0.5
    encoded = encode_wav(Audio(samples))
    decoded = decode_wav(encoded)

    assert decoded.sample_rate == 16_000
    assert decoded.duration_seconds == pytest.approx(1)
    assert np.max(np.abs(decoded.samples - samples)) < 0.0001


def test_stereo_is_mixed_and_resampled():
    left = np.ones(8_000, dtype=np.int16) * 8_000
    right = np.ones(8_000, dtype=np.int16) * -4_000
    stereo = np.column_stack([left, right]).astype("<i2")
    target = io.BytesIO()
    with wave.open(target, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(8_000)
        wav.writeframes(stereo.tobytes())

    decoded = decode_wav(target.getvalue())
    assert len(decoded.samples) == 16_000
    assert np.mean(decoded.samples) == pytest.approx(2_000 / 32_768, abs=0.001)


def test_invalid_wav_has_friendly_error():
    with pytest.raises(AudioError, match="valid PCM WAV"):
        decode_wav(b"not audio")


def test_downsampling_rejects_out_of_band_tone():
    sample_rate = 48_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    # 12 kHz is above the 8 kHz Nyquist limit of the 16 kHz output.
    source = np.sin(2 * np.pi * 12_000 * time) * 0.8
    target = io.BytesIO()
    with wave.open(target, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes((source * 32767).astype("<i2").tobytes())

    decoded = decode_wav(target.getvalue())
    assert decoded.duration_seconds == pytest.approx(1)
    assert np.sqrt(np.mean(decoded.samples**2)) < 0.02
