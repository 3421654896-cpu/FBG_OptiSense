"""Thread-safe buffering for real-time spectrum frames.

Stress sensing values freshness over replaying every queued frame: when the UI
falls behind, it consumes the newest complete spectrum and records how many
older complete spectra were skipped.  Slow calibration modes can still consume
the same buffer in FIFO order.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import queue
import threading
import time
from typing import Optional


@dataclass(frozen=True)
class TimestampedFrame:
    payload: bytes
    received_ns: int

    def age_ms(self, now_ns: Optional[int] = None) -> float:
        """Return age since the complete frame reached the desktop parser."""

        if now_ns is None:
            now_ns = time.monotonic_ns()
        return max(0.0, (int(now_ns) - self.received_ns) / 1_000_000.0)


@dataclass(frozen=True)
class FrameBufferStats:
    received: int
    pending: int
    consumer_skipped: int
    overflow_dropped: int
    last_received_ns: int | None = None

    @property
    def dropped(self) -> int:
        return self.consumer_skipped + self.overflow_dropped


class LatestFrameBuffer:
    """Bounded frame buffer with selectable FIFO/latest consumption."""

    def __init__(self, maxlen: int = 200):
        if int(maxlen) <= 0:
            raise ValueError("maxlen must be positive")
        self._frames: deque[TimestampedFrame] = deque()
        self._maxlen = int(maxlen)
        self._lock = threading.Lock()
        self._received = 0
        self._consumer_skipped = 0
        self._overflow_dropped = 0
        self._last_received_ns = None
        self._taps = []

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

    def put(self, payload: bytes, received_ns: Optional[int] = None) -> None:
        frame = TimestampedFrame(
            payload=bytes(payload),
            received_ns=(
                # This timestamp is also used for dataset/event alignment.
                # On Windows Python 3.12 perf_counter (QPC) and monotonic
                # (GetTickCount64) are different clocks; never mix them.
                time.monotonic_ns() if received_ns is None else int(received_ns)
            ),
        )
        with self._lock:
            if len(self._frames) >= self._maxlen:
                self._frames.popleft()
                self._overflow_dropped += 1
            self._frames.append(frame)
            self._received += 1
            self._last_received_ns = frame.received_ns
            for tap in self._taps:
                tap._offer(frame)

    def subscribe(self, *, capacity=2048):
        """Independent bounded recording queue; UI latest-only drops do not affect it."""
        tap = FrameTap(capacity)
        with self._lock:
            self._taps.append(tap)
        return tap

    def unsubscribe(self, tap):
        # Synchronize with put(): no producer can offer after this returns.
        with self._lock:
            if tap in self._taps:
                self._taps.remove(tap)

    def take(self, *, latest_only: bool) -> Optional[TimestampedFrame]:
        """Take one frame, optionally discarding every older pending frame."""

        with self._lock:
            if not self._frames:
                return None
            if latest_only:
                frame = self._frames.pop()
                skipped = len(self._frames)
                if skipped:
                    self._frames.clear()
                    self._consumer_skipped += skipped
                return frame
            return self._frames.popleft()

    def clear(self, *, reset_counters: bool = False) -> None:
        with self._lock:
            self._frames.clear()
            self._last_received_ns = None
            if reset_counters:
                self._received = 0
                self._consumer_skipped = 0
                self._overflow_dropped = 0

    def stats(self) -> FrameBufferStats:
        with self._lock:
            return FrameBufferStats(
                received=self._received,
                pending=len(self._frames),
                consumer_skipped=self._consumer_skipped,
                overflow_dropped=self._overflow_dropped,
                last_received_ns=self._last_received_ns,
            )


class FrameTap:
    """Nonblocking producer side of a capture subscription. Loss is explicit."""
    def __init__(self, capacity):
        if int(capacity) <= 0:
            raise ValueError("tap capacity must be positive")
        self.queue = queue.Queue(maxsize=int(capacity))
        self.offered = 0
        self.dropped = 0
        self._lock = threading.Lock()

    def _offer(self, frame):
        with self._lock:
            self.offered += 1
            try:
                self.queue.put_nowait(frame)
            except queue.Full:
                self.dropped += 1

    def stats(self):
        with self._lock:
            return {"offered": self.offered, "dropped": self.dropped,
                    "pending": self.queue.qsize()}
