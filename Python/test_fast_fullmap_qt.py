import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtTest

import fast_fullmap_qt


class FastFullMapQtWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])

    def test_every_frame_is_inferred_and_capture_contract_is_forwarded(self):
        seen = []

        class DummyLocalizer:
            def update(self, frame):
                seen.append(frame["sequence"])
                return SimpleNamespace(sequence=frame["sequence"])

        def fake_execute(
            _device,
            _output,
            *,
            cycles,
            spacing_us,
            on_frame,
            should_stop,
        ):
            self.assertFalse(should_stop())
            self.assertEqual(spacing_us, 50)
            on_frame({"sequence": 0})
            on_frame({"sequence": 1})
            return {"cycles": cycles}

        with (
            mock.patch.object(
                fast_fullmap_qt,
                "FastFullMapRealtimeLocalizer",
                DummyLocalizer,
            ),
            mock.patch.object(
                fast_fullmap_qt, "execute_realtime_device", side_effect=fake_execute
            ),
        ):
            worker = fast_fullmap_qt.FastFullMapStreamWorker(
                object(), Path("unused.json"), cycles=32, spacing_us=50
            )
            results = QtTest.QSignalSpy(worker.result_ready)
            completed = QtTest.QSignalSpy(worker.completed)
            worker.start()
            self.assertTrue(worker.wait(2000))
            self.app.processEvents()

        self.assertEqual(seen, [0, 1])
        self.assertEqual(len(results), 2)
        self.assertEqual(completed[0][0]["cycles"], 32)


if __name__ == "__main__":
    unittest.main()
