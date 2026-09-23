"""Loss-visible native transport recording independent of Qt rendering.

Bytes are preserved verbatim, not declared to be direct ADC or a full spectrum.
Consumers must decode per-point freshness/RC provenance before using values.
"""
import json
from pathlib import Path
import queue
import threading
import time


class NativeFrameRecorder:
    def __init__(self, buffer, output, *, capacity=2048):
        if int(capacity) <= 0:
            raise ValueError("capture capacity must be positive")
        self.buffer = buffer
        self.output = Path(output)
        self._file = self.output.open("x", encoding="utf-8")
        try:
            self.tap = buffer.subscribe(capacity=capacity)
        except Exception:
            self._file.close()
            raise
        self._stop = threading.Event()
        self._written = 0
        self._error = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="native-frame-recorder", daemon=True)
        try:
            self._thread.start()
        except Exception:
            buffer.unsubscribe(self.tap)
            self._file.close()
            raise

    def _run(self):
        try:
            while not self._stop.is_set() or not self.tap.queue.empty():
                try:
                    frame = self.tap.queue.get(timeout=.05)
                except queue.Empty:
                    continue
                try:
                    self._file.write(json.dumps({
                        "schema": "fbg-native-receive/v1",
                        "record_number": self._written + 1,
                        "host_received_monotonic_ns": frame.received_ns,
                        "payload_hex": frame.payload.hex(),
                    }, separators=(",", ":")) + "\n")
                    self._file.flush()
                    self._written += 1
                finally:
                    self.tap.queue.task_done()
        except Exception as exc:
            self._error = str(exc)
        finally:
            try:
                self._file.close()
            except Exception as exc:
                self._error = self._error or str(exc)

    def stats(self):
        return {**self.tap.stats(), "written": self._written, "error": self._error,
                "host_clock": time.get_clock_info("monotonic").implementation,
                "host_clock_resolution_s": time.get_clock_info("monotonic").resolution}

    def close(self):
        if not self._closed:
            self.buffer.unsubscribe(self.tap)
            self._stop.set()
            self._closed = True
        self._thread.join(timeout=3)
        if self._thread.is_alive():
            raise RuntimeError("native recorder did not finish; capture is incomplete")
        stats = self.stats()
        if stats["error"] or stats["dropped"] or stats["written"] != stats["offered"]:
            raise RuntimeError(f"native recording incomplete: {stats}")
        return stats
