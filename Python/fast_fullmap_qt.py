"""Qt worker for the validated all-fresh 45-point CH1 position stream."""

from __future__ import annotations

from pathlib import Path
import threading

from PyQt5 import QtCore

from fast_fullmap_live import execute_realtime_device
from fast_fullmap_realtime import FastFullMapRealtimeLocalizer


class FastFullMapStreamWorker(QtCore.QThread):
    result_ready = QtCore.pyqtSignal(object)
    progress = QtCore.pyqtSignal(int, int)
    completed = QtCore.pyqtSignal(object)
    cancelled = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        device,
        output: Path | str,
        *,
        cycles: int = 60_000,
        spacing_us: int = 50,
        model_path: Path | str | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.device = device
        self.output = Path(output)
        self.cycles = int(cycles)
        self.spacing_us = int(spacing_us)
        self.model_path = None if model_path is None else Path(model_path)
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def _stopping(self) -> bool:
        return self._stop_event.is_set()

    def run(self) -> None:
        try:
            localizer = (
                FastFullMapRealtimeLocalizer()
                if self.model_path is None
                else FastFullMapRealtimeLocalizer(self.model_path)
            )

            def publish(frame):
                result = localizer.update(frame)
                self.result_ready.emit(result)
                completed = int(frame["sequence"]) + 1
                if completed in {1, 16, 80, self.cycles} or completed % 4 == 0:
                    self.progress.emit(completed, self.cycles)

            report = execute_realtime_device(
                self.device,
                self.output,
                cycles=self.cycles,
                spacing_us=self.spacing_us,
                on_frame=publish,
                should_stop=self._stopping,
            )
        except InterruptedError:
            self.cancelled.emit(str(self.output))
        except BaseException as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.completed.emit(report)


__all__ = ["FastFullMapStreamWorker"]
