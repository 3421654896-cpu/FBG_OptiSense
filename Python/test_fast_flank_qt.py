import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtTest

import fast_flank_qt


class FastFlankQtWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])

    def test_baseline_is_frozen_before_the_next_published_frame(self):
        declarations = []

        class DummyBaseline:
            frozen = False

        class DummyLocalizer:
            def __init__(self, _rows, *, baseline_frames):
                self.baseline = DummyBaseline()
                self.baseline_frames = baseline_frames

            def update(self, frame, *, declared_no_contact):
                declarations.append(bool(declared_no_contact))
                return SimpleNamespace(
                    sequence=frame["sequence"],
                    baseline_ready=frame["sequence"] == 0,
                    baseline_frozen=self.baseline.frozen,
                )

            def freeze_for_press(self):
                self.baseline.frozen = True

        def fake_execute(_device, _output, *, cycles, on_frame, should_stop):
            self.assertFalse(should_stop())
            on_frame({"sequence": 0})
            on_frame({"sequence": 1})
            return {"analysis": {"board_rate_hz": 47.0}, "cycles": cycles}

        with (
            mock.patch.object(fast_flank_qt, "load_installed_rows", return_value=([], {})),
            mock.patch.object(fast_flank_qt, "FastFlankRealtimeLocalizer", DummyLocalizer),
            mock.patch.object(fast_flank_qt, "execute_device", side_effect=fake_execute),
        ):
            worker = fast_flank_qt.FastFlankStreamWorker(
                object(), Path("unused.json"), cycles=32
            )
            results = QtTest.QSignalSpy(worker.result_ready)
            progress = QtTest.QSignalSpy(worker.progress)
            completed = QtTest.QSignalSpy(worker.completed)
            worker.start()
            self.assertTrue(worker.wait(2000))
            self.app.processEvents()

        self.assertEqual(declarations, [True, False])
        self.assertEqual(len(results), 2)
        self.assertEqual(len(progress), 1)
        self.assertEqual(tuple(progress[0]), (1, 32))
        self.assertFalse(results[0][0].baseline_frozen)
        self.assertTrue(results[1][0].baseline_frozen)
        self.assertEqual(completed[0][0]["cycles"], 32)

    def test_stop_request_reaches_capture_and_reports_cancelled(self):
        def fake_execute(_device, _output, *, cycles, on_frame, should_stop):
            self.assertTrue(should_stop())
            raise InterruptedError("requested")

        with (
            mock.patch.object(fast_flank_qt, "load_installed_rows", return_value=([], {})),
            mock.patch.object(fast_flank_qt, "execute_device", side_effect=fake_execute),
        ):
            worker = fast_flank_qt.FastFlankStreamWorker(
                object(), Path("cancelled.json"), cycles=32
            )
            cancelled = QtTest.QSignalSpy(worker.cancelled)
            failed = QtTest.QSignalSpy(worker.failed)
            worker.stop()
            worker.start()
            self.assertTrue(worker.wait(2000))
            self.app.processEvents()

        self.assertEqual(len(cancelled), 1)
        self.assertEqual(cancelled[0][0], "cancelled.json")
        self.assertEqual(len(failed), 0)


if __name__ == "__main__":
    unittest.main()
