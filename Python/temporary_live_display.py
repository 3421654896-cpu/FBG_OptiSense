"""Bounded, latest-frame display handoff; never the acquisition/recording path.

The acquisition caller records and validates EVERY frame before calling offer.
Only obsolete paints are superseded. No Qt signal is queued for each frame.
"""

from collections import deque
import threading


class LatestFrameMailbox:
    def __init__(self):
        self._lock = threading.Lock()
        self._pending = None
        self._board_times = deque(maxlen=64)
        self.received = 0
        self.displayed = 0
        self.superseded = 0

    def offer(self, frame):
        with self._lock:
            self.received += 1
            if self._pending is not None:
                self.superseded += 1
            self._pending = frame
            if frame['sequence'] >= 16:
                # execute_device already unwraps and validates the board clock.
                self._board_times.append(frame['cycle_start_us'])

    def take(self):
        with self._lock:
            if self._pending is None:
                return None
            frame = dict(self._pending)  # Do not annotate the recorded raw object.
            self._pending = None
            self.displayed += 1
            hz = None
            if len(self._board_times) > 1:
                span = self._board_times[-1] - self._board_times[0]
                if span > 0:
                    hz = 1e6 * (len(self._board_times) - 1) / span
            frame['display_statistics'] = {
                'received_frames': self.received,
                'displayed_frames': self.displayed,
                'superseded_paints': self.superseded,
                'board_hz': hz,
                'raw_frames_discarded': 0,
            }
            return frame
