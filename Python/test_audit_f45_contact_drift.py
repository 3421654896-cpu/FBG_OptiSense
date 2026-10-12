import unittest
import numpy as np
from audit_f45_contact_drift import phase_masks
from select_f45_contact_fullcycle import represent


class PhaseMaskTests(unittest.TestCase):
    def test_history_features_do_not_use_future_frames(self):
        data = np.random.default_rng(19).normal(size=(100,135))
        complete = represent(data, 'history')
        for stop in (1,4,16,63,80):
            np.testing.assert_array_equal(represent(data[:stop], 'history'), complete[:stop])
        np.testing.assert_array_equal(complete[:,:135], data)

    def test_moving_and_release_transition_are_not_idle_labels(self):
        seq = np.arange(76, 90)
        times = [i * 100_000_000 for i in range(len(seq))]
        events = {
            'contact_descent_started': 500_000_000,
            'contact_reached_commanded_depth': 700_000_000,
            'release_started': 1_100_000_000,
            'safe_height_reached': 1_100_000_000,
        }
        data = {'commissioning': {
            'frame_receipts': [{'sequence': int(s), 'host_monotonic_ns': t}
                               for s,t in zip(seq,times)],
            'events': [{'kind': k, 'host_monotonic_ns': v} for k,v in events.items()],
        }}
        masks = phase_masks(data, seq)
        self.assertEqual(seq[masks['pre_descent_unseen']].tolist(), [80])
        self.assertEqual(seq[masks['held']].tolist(), [85,86])
        self.assertEqual(seq[masks['after_safe_retraction']].tolist(), [89])
        counts = sum(m.astype(int) for m in masks.values())
        self.assertTrue(np.all(counts <= 1))
        self.assertEqual(counts[seq == 87].item(), 0)


if __name__ == '__main__': unittest.main()
