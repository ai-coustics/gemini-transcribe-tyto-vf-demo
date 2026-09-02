from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from math import gcd

import numpy as np
from scipy.signal import resample_poly


TARGET_SAMPLE_RATE = 16_000
MAX_DURATION_SECONDS = 15 * 60


class AudioError(ValueError):
    pass


@dataclass(frozen=True)
class Audio:
    samples: np.ndarray
    sample_rate: int = TARGET_SAMPLE_RATE

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / self.sample_rate


def _pcm_to_float(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
    if sample_width == 3:
        data = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        values = data[:, 0] | (data[:, 1] << 8) | (data[:, 2] << 16)
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return values.astype(np.float32) / 8388608
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648
    raise AudioError(f"Unsupported WAV sample width: {sample_width * 8}-bit")


def decode_wav(data: bytes) -> Audio:
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            if wav.getcomptype() != "NONE":
                raise AudioError("Compressed WAV files are not supported")
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            width = wav.getsampwidth()
            frames = wav.getnframes()
            raw = wav.readframes(frames)
    except (wave.Error, EOFError) as exc:
        raise AudioError("Please provide a valid PCM WAV file") from exc

    if channels < 1 or sample_rate < 8_000:
        raise AudioError("The WAV audio format is invalid")

    samples = _pcm_to_float(raw, width)
    if len(samples) % channels:
        raise AudioError("The WAV file contains an incomplete audio frame")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    duration = len(samples) / sample_rate
    if duration < 0.25:
        raise AudioError("Record at least a quarter second of audio")
    if duration > MAX_DURATION_SECONDS:
        raise AudioError("Audio must be 15 minutes or shorter")

    if sample_rate != TARGET_SAMPLE_RATE:
        divisor = gcd(sample_rate, TARGET_SAMPLE_RATE)
        samples = resample_poly(
            samples, TARGET_SAMPLE_RATE // divisor, sample_rate // divisor
        ).astype(np.float32)

    return Audio(np.ascontiguousarray(np.clip(samples, -1, 1), dtype=np.float32))


def encode_wav(audio: Audio) -> bytes:
    pcm = (np.clip(audio.samples, -1, 1) * 32767).astype("<i2").tobytes()
    target = io.BytesIO()
    with wave.open(target, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(audio.sample_rate)
        wav.writeframes(pcm)
    return target.getvalue()
