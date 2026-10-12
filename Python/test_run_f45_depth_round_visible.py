from pathlib import Path
import tempfile
import unittest
import json

from run_f45_depth_round_visible import unused_retry_output, verify_capture


class RetryOutputTests(unittest.TestCase):
    def test_mixed_gain_frame_invalidates_complete_capture(self):
        data = {'complete': True, 'feedback_selectors': [0,1],
            'frames': [{'feedback_selectors':[0,1]} for _ in range(640)],
            'commissioning': {'target':{'point_index':17},'depth_mm':.12,
                'events':[{'kind':'final_safe_state_confirmed','status':{
                    'stopped':True,'x_homed':True,'y_homed':True,'z_homed':True}}]},
            'safety_cleanup':{'exact_disarm_ack':True,'exact_soa_shutter_ack':True},
            'analysis':{'fresh_point_count_per_cycle':45,'meets_sampling_rate_for_15hz':True,
                'board_rate_hz':31.3,'board_interval_max_us':32400}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'capture.json'
            path.write_text(json.dumps(data),'utf-8')
            self.assertTrue(verify_capture(path,17,.12,1)['checks']['gain_matches'])
            data['frames'][-1]['feedback_selectors'] = [0,2]
            path.write_text(json.dumps(data),'utf-8')
            with self.assertRaisesRegex(RuntimeError,'gain_matches'):
                verify_capture(path,17,.12,1)

    def test_skips_existing_base_and_retry_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "d0p35_p01").mkdir()
            self.assertEqual(
                unused_retry_output(root, "d0p35_p01"),
                root / "d0p35_p01_retry1",
            )
            (root / "d0p35_p01_retry1").mkdir()
            self.assertEqual(
                unused_retry_output(root, "d0p35_p01"),
                root / "d0p35_p01_retry2",
            )


if __name__ == "__main__":
    unittest.main()
