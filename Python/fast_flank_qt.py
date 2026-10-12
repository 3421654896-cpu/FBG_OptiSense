"""Qt worker for the bounded CH1 18-flank pressure-change stream.

The worker never commands the CNC.  It borrows the already-open local USB
transport, publishes only validated board frames and leaves final optical
shutdown to :func:`capture_fast_flank_stream.execute_device` on every exit.
"""

from __future__ import annotations

from pathlib import Path
import threading

from PyQt5 import QtCore

from capture_ch1_rc_diagnostic import load_installed_rows
from capture_fast_flank_stream import execute_device
from fast_flank_features import BASELINE_FRAMES
from fast_flank_realtime import FastFlankRealtimeLocalizer


class FastFlankStreamWorker(QtCore.QThread):
    """Run one finite stream without blocking or touching the CNC."""

    result_ready = QtCore.pyqtSignal(object)
    progress = QtCore.pyqtSignal(int, int)
    completed = QtCore.pyqtSignal(object)
    cancelled = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, device, output: Path | str, *, cycles: int = 1024, parent=None):
        super().__init__(parent)
        self.device = device
        self.output = Path(output)
        self.cycles = int(cycles)
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Request a causal stop; the capture layer still verifies shutdown."""

        self._stop_event.set()

    def _stopping(self) -> bool:
        return self._stop_event.is_set()

    def run(self) -> None:
        try:
            rows, _metadata = load_installed_rows()
            localizer = FastFlankRealtimeLocalizer(
                rows,
                baseline_frames=BASELINE_FRAMES,
            )

            def publish(frame):
                # Only the initial, explicitly displayed phase may teach the
                # baseline.  Freeze it immediately after 64 valid no-contact
                # frames so a later press can never be learned away.
                declared_no_contact = not localizer.baseline.frozen
                result = localizer.update(
                    frame,
                    declared_no_contact=declared_no_contact,
                )
                if result.baseline_ready and not result.baseline_frozen:
                    localizer.freeze_for_press()
                self.result_ready.emit(result)
                completed = int(frame["sequence"]) + 1
                # Progress text/progress-bar repaints do not need one signal
                # per 47 Hz acquisition frame.  Keep important phase edges and
                # otherwise publish at about 12 Hz; inference still consumes
                # every validated frame above.
                if completed in {1, 16, 80, self.cycles} or completed % 4 == 0:
                    self.progress.emit(completed, self.cycles)

            report = execute_device(
                self.device,
                self.output,
                cycles=self.cycles,
                on_frame=publish,
                should_stop=self._stopping,
            )
        except InterruptedError:
            self.cancelled.emit(str(self.output))
        except BaseException as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.completed.emit(report)


__all__ = ["FastFlankStreamWorker"]
