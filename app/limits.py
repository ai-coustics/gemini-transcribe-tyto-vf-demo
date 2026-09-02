from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict, deque


def _int_env(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, default)))
    except ValueError:
        return default


# 0 disables a limit. Tunable per deployment without touching code.
WINDOW_SECONDS = _int_env("RATE_LIMIT_WINDOW_SECONDS", 600)
PER_IP_LIMIT = _int_env("RATE_LIMIT_PER_IP", 5)
DAILY_LIMIT = _int_env("RATE_LIMIT_DAILY", 200)
AUDIO_TTL_SECONDS = _int_env("AUDIO_TTL_SECONDS", 600)
AUDIO_MAX_BYTES = _int_env("AUDIO_MAX_BYTES", 256 * 1024 * 1024)


class RateLimiter:
    """Per-IP sliding window plus a global daily budget.

    State is per process, so this is only accurate while the app runs as a
    single instance. On Modal, pair it with max_containers=1 so every request
    lands in the same process.
    """

    def __init__(
        self,
        per_ip: int = PER_IP_LIMIT,
        window: int = WINDOW_SECONDS,
        daily: int = DAILY_LIMIT,
    ) -> None:
        self.per_ip = per_ip
        self.window = window
        self.daily = daily
        self._hits: dict[str, deque[float]] = {}
        self._today: deque[float] = deque()
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        cutoff = now - self.window
        for ip in list(self._hits):
            hits = self._hits[ip]
            while hits and hits[0] < cutoff:
                hits.popleft()
            if not hits:
                del self._hits[ip]
        day_cutoff = now - 86_400
        while self._today and self._today[0] < day_cutoff:
            self._today.popleft()

    def check(self, ip: str) -> str | None:
        """Record a request. Returns None when allowed, else why it was refused."""
        now = time.time()
        with self._lock:
            self._prune(now)
            if self.daily and len(self._today) >= self.daily:
                return "This demo has reached its daily limit. Please try again tomorrow."
            hits = self._hits.setdefault(ip, deque())
            if self.per_ip and len(hits) >= self.per_ip:
                minutes = int((hits[0] + self.window - now) // 60) + 1
                return f"Too many comparisons from this address. Try again in about {minutes} minute(s)."
            hits.append(now)
            self._today.append(now)
            return None


class AudioStore:
    """Holds comparison audio in memory so nothing is ever written to disk.

    Entries expire on TTL and the whole store is bounded; the oldest job is
    dropped when the cap is exceeded.
    """

    def __init__(
        self, ttl: int = AUDIO_TTL_SECONDS, max_bytes: int = AUDIO_MAX_BYTES
    ) -> None:
        self.ttl = ttl
        self.max_bytes = max_bytes
        self._jobs: OrderedDict[str, tuple[float, dict[str, bytes]]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def _drop(self, job_id: str) -> None:
        _, files = self._jobs.pop(job_id)
        self._bytes -= sum(len(data) for data in files.values())

    def _evict(self, now: float) -> None:
        for job_id, (created, _) in list(self._jobs.items()):
            if now - created > self.ttl:
                self._drop(job_id)
        while self._jobs and self._bytes > self.max_bytes:
            self._drop(next(iter(self._jobs)))

    def put(self, job_id: str, files: dict[str, bytes]) -> None:
        now = time.time()
        with self._lock:
            if job_id in self._jobs:
                self._drop(job_id)
            self._jobs[job_id] = (now, files)
            self._bytes += sum(len(data) for data in files.values())
            self._evict(now)

    def get(self, job_id: str, name: str) -> bytes | None:
        now = time.time()
        with self._lock:
            self._evict(now)
            entry = self._jobs.get(job_id)
            return entry[1].get(name) if entry else None


def client_ip(headers, fallback: str | None) -> str:
    """Real caller address, honouring the proxy header Modal sits behind."""
    forwarded = headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return fallback or "unknown"
