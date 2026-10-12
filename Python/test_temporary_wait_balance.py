import copy
import json
import unittest

from temporary_wait_balance import analyze_wait_balance
from temporary_test_mode import select_points
from test_temporary_test_mode import reference


TIMES = [1, 56, 111, 167, 222, 277, 332, 387, 443, 500,
         650, 800, 1000, 1250, 1500, 2000, 2500, 3000,
         4000, 5000, 6000, 8000, 10000, 12000, 16000,
         20000, 25000, 30000, 40000, 50000, 60000, 80000, 100000]


def timing_report(route=None):
    if route is None:
        table, x, y = reference()
        route = select_points(table, x, y)['rows'][:5]
    records = []
    for repeat in range(3):
        for target in range(1, len(route)):
            values = [100 if t < 650 else 1000 for t in TIMES]
            records.append(dict(
                repeat=repeat, route_source_order=target - 1, route_target_order=target,
                source_fullband_index=route[target-1]['index'],
                target_fullband_index=route[target]['index'], fast_dac_writes=False,
                path_update_us=450, baseline_adc=[0, 100, 0, 0, 0, 0],
                sample_start_after_target_write_us=list(TIMES),
                direct_adc=[[0, value, 0, 0, 0, 0] for value in values],
            ))
    return dict(complete=True, exact_disarm_ack=True, exact_soa_shutter_ack=True,
                route=route, repeats=3, fast_dac_writes=False,
                ch1_feedback_selector=2, captures=records)


class BalanceTests(unittest.TestCase):
    def analyze(self, report):
        return analyze_wait_balance(report, [850] * len(report['route']))

    def test_steady_candidates_round_up_with_margin_preserve_first_not_certify_accuracy(self):
        result = self.analyze(timing_report())
        self.assertEqual(result['candidate_point_delays_us'], [850, 725, 725, 725, 725])
        self.assertTrue(result['candidate_available'])
        self.assertFalse(result['closed_loop_accuracy_verified'])
        self.assertFalse(result['absolute_wavelength_accuracy_verified'])
        self.assertFalse(result['reverse_wrap_tested'])
        self.assertTrue(result['first_point_unmeasured'])

    def test_real_numpy_comparisons_are_json_serializable_for_checkpoint_and_ui(self):
        report = timing_report()
        report['captures'][0]['baseline_adc'][1] = 995
        result = self.analyze(report)
        restored = json.loads(json.dumps(result, ensure_ascii=False, allow_nan=False))
        self.assertEqual(result, restored)
        for point in restored['points'][1:]:
            for repeat in point['individual_repeats']:
                self.assertIs(type(repeat['meaningful_step']), bool)

    def test_rebound_after_first_90pct_delays_candidate(self):
        report = timing_report()
        for item in report['captures']:
            if item['route_target_order'] == 1:
                for t, adc in zip(TIMES, item['direct_adc']):
                    if 1500 <= t < 2500:
                        adc[1] = 1400
        result = self.analyze(report)
        self.assertEqual(result['candidate_point_delays_us'][1], 2575)

    def test_one_slow_repeat_not_hidden_by_median(self):
        report = timing_report()
        for item in report['captures']:
            if item['repeat'] == 2 and item['route_target_order'] == 1:
                for t, adc in zip(TIMES, item['direct_adc']):
                    if t < 10000:
                        adc[1] = 100
        result = self.analyze(report)
        self.assertEqual(result['candidate_point_delays_us'][1], 10075)

    def test_small_step_with_detour_is_not_automatically_minimum_wait(self):
        report = timing_report()
        for item in report['captures']:
            item['baseline_adc'][1] = 995
            for t, adc in zip(TIMES, item['direct_adc']):
                adc[1] = 1400 if t == 1000 else 1000
        result = self.analyze(report)
        self.assertEqual(result['candidate_point_delays_us'][1], 1325)
        self.assertFalse(result['points'][1]['individual_repeats'][0]['meaningful_step'])

    def test_over_limit_is_unavailable_not_silently_capped(self):
        report = timing_report()
        for item in report['captures']:
            for t, adc in zip(TIMES, item['direct_adc']):
                if t < 20000:
                    adc[1] = 100
        result = self.analyze(report)
        self.assertFalse(result['candidate_available'])
        self.assertIsNone(result['candidate_point_delays_us'][1])
        self.assertGreater(result['points'][1]['required_wait_us'], 15000)
        self.assertIn('required_wait_exceeds_protocol_limit', result['points'][1]['reasons'])

    def test_tail_drift_clipping_and_different_repeat_level_veto(self):
        for mode in ('drift', 'clip', 'repeat_level'):
            with self.subTest(mode=mode):
                report = timing_report()
                item = report['captures'][-1]
                if mode == 'drift':
                    for adc, v in zip(item['direct_adc'][-3:], (800, 1000, 1200)):
                        adc[1] = v
                elif mode == 'clip':
                    item['direct_adc'][5][1] = 4095
                else:
                    for adc in item['direct_adc'][-3:]:
                        adc[1] = 1200
                result = self.analyze(report)
                self.assertFalse(result['candidate_available'])
                self.assertIsNone(result['candidate_point_delays_us'][-1])

    def test_invalid_missing_stale_reordered_and_shutdown_fail_closed(self):
        mutations = [
            lambda r: r.update(complete=False),
            lambda r: r.update(exact_soa_shutter_ack=False),
            lambda r: r['captures'].pop(),
            lambda r: r['captures'][0].update(repeat=2),
            lambda r: r['captures'][0].update(target_fullband_index=0),
            lambda r: r['captures'][0].update(fast_dac_writes=True),
            lambda r: r['captures'][0]['sample_start_after_target_write_us'].__setitem__(2, 1),
            lambda r: r['captures'][0]['direct_adc'][3].__setitem__(1, float('nan')),
            lambda r: r['captures'][0]['direct_adc'].pop(),
        ]
        for mutate in mutations:
            report = timing_report(); mutate(report)
            with self.assertRaises((ValueError, KeyError)):
                self.analyze(report)

    def test_does_not_modify_raw_report_or_waits_and_supports_seven_points(self):
        from test_temporary_symmetric_seven import selected
        report = timing_report(selected()[-1]['rows'][:7])
        before = copy.deepcopy(report)
        waits = [850] * 7
        result = analyze_wait_balance(report, waits)
        self.assertEqual(report, before)
        self.assertEqual(waits, [850] * 7)
        self.assertEqual(len(result['points']), 7)

    def test_all_signal_channels_use_their_own_adc_and_feedback(self):
        for channel in range(4):
            report = timing_report()
            selector = 2 if channel < 2 else 0
            report.update(signal_channel=channel, feedback_selector=selector)
            for item in report['captures']:
                item['baseline_adc'][channel] = 100
                for sample in item['direct_adc']:
                    value = sample[1]
                    if channel != 1:
                        sample[1] = 4095  # Other-channel clipping is not signal clipping.
                    sample[channel] = value
            result = self.analyze(report)
            self.assertEqual(result['signal_channel'], channel)
            self.assertEqual(result['feedback_selector'], selector)
            self.assertEqual(result['candidate_point_delays_us'][1], 725)


if __name__ == '__main__':
    unittest.main()
