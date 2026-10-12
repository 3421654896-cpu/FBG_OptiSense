import threading
import unittest
from unittest.mock import patch, Mock

from temporary_live_display import LatestFrameMailbox


def frame(n):
    return {'sequence': n, 'cycle_start_us': n * 10000, 'raw': 'unchanged'}


class MailboxTests(unittest.TestCase):
    def test_burst_bounded_latest_only_board_rate_not_paint_rate(self):
        mailbox = LatestFrameMailbox()
        raw = []
        for n in range(500):
            item = frame(n)
            raw.append(item)
            mailbox.offer(item)
        shown = mailbox.take()
        self.assertEqual(shown['sequence'], 499)
        self.assertEqual(shown['display_statistics']['board_hz'], 100)
        self.assertEqual(shown['display_statistics']['received_frames'], 500)
        self.assertEqual(shown['display_statistics']['displayed_frames'], 1)
        self.assertEqual(shown['display_statistics']['superseded_paints'], 499)
        self.assertEqual(shown['display_statistics']['raw_frames_discarded'], 0)
        self.assertIsNone(mailbox.take())
        self.assertTrue(all('display_statistics' not in item for item in raw))
        self.assertEqual(len(mailbox._board_times), 64)

    def test_warmup_not_counted_as_rate_and_no_duplicate_final_paint(self):
        mailbox = LatestFrameMailbox()
        for n in range(17):
            mailbox.offer(frame(n))
        self.assertIsNone(mailbox.take()['display_statistics']['board_hz'])
        mailbox.offer(frame(17))
        self.assertEqual(mailbox.take()['display_statistics']['board_hz'], 100)
        self.assertIsNone(mailbox.take())

    def test_thread_handoff_counts_and_final_value(self):
        mailbox = LatestFrameMailbox()
        done = threading.Event()
        def feed():
            for n in range(10000):
                mailbox.offer(frame(n))
            done.set()
        thread = threading.Thread(target=feed)
        thread.start()
        shown = []
        while not done.is_set():
            item = mailbox.take()
            if item is not None:
                shown.append(item)
        thread.join()
        final = mailbox.take()
        if final is not None:
            shown.append(final)
        self.assertEqual(shown[-1]['sequence'], 9999)
        self.assertEqual(mailbox.received, 10000)
        self.assertEqual(mailbox.displayed + mailbox.superseded, 10000)
        self.assertEqual(sorted({r['sequence'] for r in shown}), [r['sequence'] for r in shown])

    def test_worker_records_before_handoff_and_never_queues_per_frame_signal(self):
        import app_JDSU
        from temporary_test_widget import TemporaryWorker
        worker = TemporaryWorker(None, None, 'test.json', 32, coalesce_display=True)
        emitted = Mock()
        worker.frame.connect(emitted)
        stored = []
        def execute(*args, **kwargs):
            for n in range(100):
                item = frame(n)
                stored.append(item)
                kwargs['on_frame'](item)
            return dict(mean_hz=100, p99_period_ms=10)
        with patch('temporary_test_widget.execute_device', side_effect=execute):
            worker.run()
        self.assertEqual(len(stored), 100)
        emitted.assert_not_called()
        self.assertEqual(worker.take_display_frame()['sequence'], 99)
        self.assertIsNone(worker.take_display_frame())


if __name__ == '__main__':
    unittest.main()
