import copy
import unittest

from developer_diagnostics import measured_budget, nominal_budget


class DeveloperDiagnosticsTests(unittest.TestCase):
    def frames(self):
        frame = dict(sequence=16, firmware=0x10076, single_sample=True,
                     cycle_start_us=0, elapsed_us=2717,
                     records=[dict(write_end_us=486, sample_end_us=1358),
                              dict(write_end_us=1844, sample_end_us=2716)])
        following = {**frame, 'sequence': 17, 'cycle_start_us': 3717}
        return frame, following

    def test_nominal_45_and_63_budget_at_850_us(self):
        for count, wait, wire, total in ((45, 38.25, 20.25, 58.5),
                                         (63, 53.55, 28.35, 81.9)):
            result = nominal_budget([850] * count)
            self.assertAlmostEqual(result['wait_ms'], wait)
            self.assertAlmostEqual(result['dac_wire_ms'], wire)
            self.assertAlmostEqual(result['minimum_ms'], total)

    def test_board_budget_adds_up_and_does_not_add_temperature_again(self):
        result = measured_budget(*self.frames(), [850, 850])
        self.assertAlmostEqual(result['dac_and_interpoint_ms'], .972)
        self.assertAlmostEqual(result['adc_and_hold_overhead_ms'], .044)
        self.assertAlmostEqual(sum(result[k] for k in (
            'dac_and_interpoint_ms', 'prescribed_wait_ms',
            'adc_and_hold_overhead_ms', 'frame_tail_ms', 'frame_gap_ms')),
            result['period_ms'])

    def test_skipped_frames_and_mismatched_settings_are_not_measured_periods(self):
        for change in ({'sequence': 19}, {'firmware': 1}, {'adc_channel': 3},
                       {'i2c_errors': 1}, {'cycle_start_us': 100}):
            frame, following = self.frames()
            following.update(change)
            with self.assertRaises(ValueError):
                measured_budget(frame, following, [850, 850])

    def test_invalid_waits_and_point_stamps_rejected(self):
        for waits in ([], [float('nan')], [-1]):
            with self.assertRaises(ValueError):
                nominal_budget(waits)
        frame, following = self.frames()
        broken = copy.deepcopy(frame)
        broken['records'][1]['write_end_us'] = 1
        with self.assertRaises(ValueError):
            measured_budget(broken, following, [850, 850])


if __name__ == '__main__':
    unittest.main()
