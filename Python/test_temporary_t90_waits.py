import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from temporary_t90_waits import (
    analyze_t90_waits, checked_t90_waits, T90_SCHEMA, T90_CRITERION, t90_point_diagnostic_text,
    T90_GATE_POLICY, T90_VOLTAGE_THRESHOLD_V, T90_LOW_VOLTAGE_WAIT_US,
)
from temporary_test_widget import TemporaryTestWindow, SwitchTimingWorker
from test_temporary_symmetric_seven import selected
from test_temporary_wait_balance import TIMES
import test_temporary_test_widget as widget_tests


def voltage_probe(target, *, channel=1, adc=1000, fast_dac_writes=False):
    samples = [[0]*6 for _ in TIMES]
    for sample in samples:
        sample[channel] = adc
    return dict(source_fullband_index=target['index'],target_fullband_index=target['index'],
                fast_dac_writes=fast_dac_writes, path_update_us=450, baseline_adc=samples[0][:],
                sample_start_after_target_write_us=list(TIMES),direct_adc=samples)


def cycle_report(rows=None, *, channel=1, machine='machine_1', repeats=1):
    rows = rows or selected()[-1]['rows']
    records = []
    for repeat in range(repeats):
        for target in range(len(rows)):
            source = (target - 1) % len(rows)
            values = [100 if stamp < 650 else 1000 for stamp in TIMES]
            base = [0] * 6
            base[channel] = 100
            samples = [[0] * 6 for _ in TIMES]
            for sample, value in zip(samples, values):
                sample[channel] = value
            records.append(dict(
                repeat=repeat, route_source_order=source, route_target_order=target,
                source_fullband_index=rows[source]['index'], target_fullband_index=rows[target]['index'],
                fast_dac_writes=False, path_update_us=450, baseline_adc=base,
                measurement_kind='measured_transition', voltage_probe=voltage_probe(rows[target],channel=channel),
                sample_start_after_target_write_us=list(TIMES), direct_adc=samples))
    return dict(complete=True, exact_disarm_ack=True, exact_soa_shutter_ack=True,
                route=rows, repeats=repeats, reverse_wrap_tested=True, machine_id=machine,
                cycle_wrap_included=True, voltage_gate_policy=T90_GATE_POLICY,
                voltage_threshold_v=T90_VOLTAGE_THRESHOLD_V, low_voltage_wait_us=T90_LOW_VOLTAGE_WAIT_US,
                signal_channel=channel, signal_channel_name=f'CH{channel}',
                feedback_selector=2 if channel < 2 else 0,
                fast_dac_writes=False, captures=records)


def gated_report(levels, *, channel=1):
    report=cycle_report(channel=channel)
    for target,(record,adc) in enumerate(zip(report['captures'],levels)):
        record['voltage_probe']=voltage_probe(report['route'][target],channel=channel,adc=adc)
        if adc*2.5/4096 <= .2:
            for key in ('direct_adc','baseline_adc','sample_start_after_target_write_us','path_update_us'):
                record.pop(key)
            record['measurement_kind']='low_voltage_fixed_wait'
        else:
            for stamp,row in zip(TIMES,record['direct_adc']):
                row[channel]=100 if stamp<650 else adc
    return report


class T90AnalysisTests(unittest.TestCase):
    def test_voltage_gate_below_equal_above_threshold_and_exact_300_without_margin(self):
        for adc,expected in ((327,300),(327.68,300),(328,725)):
            with self.subTest(adc=adc):
                report=gated_report([adc]*63)
                result=analyze_t90_waits(report)
                self.assertEqual(result['candidate_point_delays_us'],[expected]*63)
                self.assertEqual(result['measured_point_count'],0 if expected==300 else 63)
                self.assertEqual(result['fixed_wait_point_count'],63 if expected==300 else 0)
                point=result['points'][0]
                self.assertEqual(point['t90_measurement_skipped'],expected==300)
                if expected==300:
                    self.assertIsNone(point['settled_90pct_after_target_write_us'])
                    self.assertIsNone(point['dac_write_us'])
                    self.assertEqual(point['individual_repeats'],[])
                self.assertEqual(checked_t90_waits(result,report['route'],channel=1,selector=2,
                                                 fast=False,machine_id='machine_1'),[expected]*63)

    def test_mixed_voltage_table_applies_atomically_in_all_four_channels(self):
        levels=[1000 if i%3==0 else 100 for i in range(63)]
        for channel in range(4):
            report=gated_report(levels,channel=channel)
            original=copy.deepcopy(report)
            result=analyze_t90_waits(report)
            waits=checked_t90_waits(result,report['route'],channel=channel,
                                    selector=2 if channel<2 else 0,fast=False,machine_id='machine_1')
            self.assertEqual(waits,[725 if adc>.2*4096/2.5 else 300 for adc in levels])
            self.assertEqual((result['measured_point_count'],result['fixed_wait_point_count']),(21,42))
            self.assertEqual(report,original)

    def test_unmeasured_low_voltage_wrap_has_explicit_fixed_wait_without_fake_wrap_timing(self):
        report=gated_report([100]*63)
        result=analyze_t90_waits(report)
        self.assertTrue(result['cycle_wrap_included'])
        self.assertFalse(result['reverse_wrap_tested'])
        self.assertTrue(result['first_point_unmeasured'])
        self.assertFalse(result['first_point_unhandled'])
        self.assertTrue(result['candidate_available'])
        self.assertFalse(result['absolute_wavelength_accuracy_verified'])

    def test_probe_corruption_cannot_be_treated_as_a_low_voltage_fixed_wait(self):
        for mode in ('nan','clipping','row','style','missing','kind'):
            report=gated_report([100]*63)
            record=report['captures'][0]
            if mode=='nan':record['voltage_probe']['direct_adc'][0][1]=float('nan')
            elif mode=='clipping':record['voltage_probe']['direct_adc'][0][1]=4080
            elif mode=='row':record['voltage_probe']['source_fullband_index']+=1
            elif mode=='style':record['voltage_probe']['fast_dac_writes']=True
            elif mode=='missing':record.pop('voltage_probe')
            else:record['measurement_kind']='measured_transition'
            with self.subTest(mode=mode),self.assertRaises(ValueError):analyze_t90_waits(report)

    def test_fixed_wait_tampering_is_rejected_even_if_all_summary_wait_fields_agree(self):
        report=gated_report([100]*63)
        result=analyze_t90_waits(report)
        result['candidate_point_delays_us'][0]=325
        result['points'][0]['candidate_wait_us']=325
        result['points'][0]['required_wait_us']=325
        with self.assertRaisesRegex(ValueError,'固定300'):
            checked_t90_waits(result,report['route'],channel=1,selector=2,fast=False,machine_id='machine_1')

    def test_old_raw_report_cannot_be_silently_reinterpreted_as_voltage_gated(self):
        report=cycle_report()
        report.pop('voltage_gate_policy')
        with self.assertRaisesRegex(ValueError,'重新测量'):analyze_t90_waits(report)

    def test_low_final_voltage_does_not_hide_actual_transient_adc_clipping(self):
        report=cycle_report()
        for row in report['captures'][0]['direct_adc']:row[1]=100
        report['captures'][0]['direct_adc'][0][1]=4080
        result=analyze_t90_waits(report)
        self.assertFalse(result['candidate_available'])
        self.assertIsNone(result['candidate_point_delays_us'][0])
        self.assertIn('ADC饱和',t90_point_diagnostic_text(result['points'][0]))

    def test_steady_probe_uses_identical_real_source_and_target_rows_not_a_fake_zero_start(self):
        from capture_temporary_half_height_transients import capture_target_level
        row=cycle_report()['route'][0]
        raw=voltage_probe(row,adc=100)
        with patch('capture_temporary_half_height_transients.capture_custom',return_value=raw) as capture:
            result=capture_target_level('port',row,123,fast_dac_writes=False)
        capture.assert_called_once_with('port',row,row,123,fast_dac_writes=False)
        self.assertEqual(result['source_fullband_index'],row['index'])
        self.assertEqual(result['target_fullband_index'],row['index'])

    def test_all_63_points_including_wrap_round_up_and_do_not_certify_wavelength(self):
        report = cycle_report()
        result = analyze_t90_waits(report)
        self.assertEqual(result['candidate_point_delays_us'], [725] * 63)
        self.assertEqual(result['candidate_wait_sum_us'], 45675)
        self.assertTrue(result['candidate_available'])
        self.assertEqual(result['points'][0]['source_order'], 62)
        self.assertEqual(result['points'][0]['category'], 'frame_wrap')
        self.assertEqual(result['points'][0]['settled_90pct_after_target_write_us'], 650)
        self.assertEqual(result['points'][0]['settled_90pct_from_switch_begin_us'], 1100)
        self.assertFalse(result['first_point_unmeasured'])
        self.assertFalse(result['absolute_wavelength_accuracy_verified'])
        self.assertEqual(result['schema'], T90_SCHEMA)
        self.assertEqual(result['criterion'], T90_CRITERION)
        self.assertEqual(result['tolerance_fraction'], .10)
        self.assertEqual(result['tolerance_floor_adc'], 0.)
        self.assertEqual(result['calculation_start_adc'], 0.)
        self.assertEqual(result['capture_mode'], 'adjacent_wavelength_transitions')
        self.assertEqual(result['measurement_mode'], 'single_voltage_gated_transition_per_point')
        self.assertEqual(result['repeats'], 1)
        self.assertEqual(len(report['captures']), 63)
        self.assertTrue(all(len(p['individual_repeats']) == 1 for p in result['points']))
        self.assertEqual(result['points'][0]['target_90pct_adc'], 900.)
        self.assertEqual(result['points'][0]['individual_repeats'][0]['raw_baseline_adc'], 100.)
        self.assertEqual(result, json.loads(json.dumps(result, allow_nan=False)))

    def test_single_measurement_still_checks_rebound_and_slow_settling(self):
        for mode, expected in [('rebound', 2575), ('slow', 10075)]:
            report = cycle_report()
            item = report['captures'][0]  # Single measurement, wrap point.
            for stamp, row in zip(TIMES, item['direct_adc']):
                if mode == 'rebound' and 1500 <= stamp < 2500:
                    row[1] = 1400
                if mode == 'slow' and stamp < 10000:
                    row[1] = 100
            result = analyze_t90_waits(report)
            self.assertEqual(result['candidate_point_delays_us'][0], expected)

    def test_low_final_after_high_probe_uses_fixed_300_not_50_or_measured_t90(self):
        report = cycle_report()
        for item in report['captures']:
            item['baseline_adc'][1] = 995
            for stamp, row in zip(TIMES, item['direct_adc']):
                row[1] = 108 if stamp == 1000 else 8
        result = analyze_t90_waits(report)
        self.assertEqual(result['candidate_point_delays_us'], [300] * 63)
        self.assertEqual(result['points'][0]['wait_source'], 'low_voltage_fixed_300us')
        self.assertIsNone(result['points'][0]['settled_90pct_after_target_write_us'])
        self.assertFalse(result['points'][0]['t90_measurement_skipped'])
        self.assertFalse(result['points'][0]['meaningful_step'])
        self.assertEqual(result['points'][0]['repeat_tolerances_adc'], [.8])

    def test_low_adc_uses_fixed_wait_while_preserving_raw_diagnostics(self):
        for final, low in ((8, 6), (20, 17)):
            report = cycle_report()
            for item in report['captures']:
                for stamp, row in zip(TIMES, item['direct_adc']):
                    row[1] = low if stamp < 10000 else final
            result = analyze_t90_waits(report)
            self.assertEqual(result['candidate_point_delays_us'], [300] * 63)
            self.assertEqual(result['points'][0]['repeat_tolerances_adc'], [final * .1])

    def test_low_adc_tail_noise_does_not_veto_the_user_specified_fixed_wait(self):
        report = cycle_report()
        for item in report['captures']:
            for row in item['direct_adc']:
                row[1] = 8
        report['captures'][0]['direct_adc'][-1][1] = 10
        result = analyze_t90_waits(report)
        self.assertEqual(result['points'][0]['target_adc'], 8.)
        self.assertEqual(result['points'][0]['repeat_tolerances_adc'], [.8])
        self.assertIn('tail_not_repeatably_flat', result['points'][0]['individual_repeats'][0]['reasons'])
        self.assertEqual(result['points'][0]['candidate_wait_us'],300)
        self.assertTrue(result['candidate_available'])

    def test_zero_adc_target_has_zero_tolerance_without_division_or_fabricated_noise_floor(self):
        report = cycle_report()
        for item in report['captures']:
            for stamp, row in zip(TIMES, item['direct_adc']):
                row[1] = 1 if stamp < 650 else 0
        result = analyze_t90_waits(report)
        self.assertEqual(result['candidate_point_delays_us'], [300] * 63)
        self.assertEqual(result['points'][0]['repeat_tolerances_adc'], [0.])
        self.assertEqual(result['points'][0]['target_90pct_adc'], 0.)
        self.assertEqual(result, json.loads(json.dumps(result, allow_nan=False)))

    def test_stable_sample_count_and_time_minima_remain_unchanged(self):
        report = cycle_report()
        # Preserve a full 33-sample capture, but the last 3 span only 2 ms.
        item = report['captures'][0]
        item['sample_start_after_target_write_us'][-3:] = [98000, 99000, 100000]
        result = analyze_t90_waits(report)
        self.assertIn('tail_not_repeatably_flat', result['points'][0]['reasons'])
        self.assertFalse(result['candidate_available'])
        from temporary_wait_balance import _repeat_metric
        item = cycle_report()['captures'][0]
        item['direct_adc'][-3][1] = 800
        metric = _repeat_metric(item, 1000., 100., 1, reference_baseline=0.)
        self.assertIn('insufficient_stable_tail', metric['reasons'])

    def test_90pct_criterion_is_ten_pct_target_relative_to_zero_not_previous_step(self):
        report = cycle_report()
        for item in report['captures']:
            for stamp, row in zip(TIMES, item['direct_adc']):
                row[1] = 100 if stamp < 650 else 905 if stamp < 10000 else 1000
        result = analyze_t90_waits(report)
        self.assertEqual(result['candidate_point_delays_us'], [725] * 63)
        self.assertEqual(result['points'][0]['repeat_tolerances_adc'], [100.])

    def test_previous_point_adc_does_not_change_threshold_and_raw_capture_is_preserved(self):
        for baseline in (0, 100, 995, 2000, 4090):
            report = cycle_report()
            for item in report['captures']:
                item['baseline_adc'][1] = baseline
                for stamp, row in zip(TIMES, item['direct_adc']):
                    row[1] = 100 if stamp < 650 else 905 if stamp < 10000 else 1000
            original = copy.deepcopy(report)
            with self.subTest(baseline=baseline):
                result = analyze_t90_waits(report)
                self.assertEqual(result['candidate_point_delays_us'], [725] * 63)
                point = result['points'][0]
                self.assertEqual(point['repeat_tolerances_adc'], [100.])
                self.assertTrue(point['meaningful_step'])
                self.assertEqual(point['calculation_start_adc'], 0.)
                self.assertEqual(point['target_90pct_adc'], 900.)
                self.assertEqual([r['raw_baseline_adc'] for r in point['individual_repeats']],
                                 [float(baseline)])
                self.assertEqual(report, original)  # No fabricated zero source ADC.

    def test_actual_target_adc_clipping_still_prevents_application(self):
        report = cycle_report()
        report['captures'][0]['direct_adc'][0][1] = 4080
        result = analyze_t90_waits(report)
        self.assertFalse(result['candidate_available'])
        self.assertIn('adc_clipped', result['points'][0]['reasons'])

    def test_single_measurement_keeps_tail_checks_without_claiming_repeatability(self):
        from temporary_t90_waits import REASON_LABELS
        report = cycle_report()
        report['captures'][0]['direct_adc'][-1][1] = 1400
        result = analyze_t90_waits(report)
        self.assertIn('tail_not_repeatably_flat', result['points'][0]['reasons'])
        self.assertEqual(REASON_LABELS['tail_not_repeatably_flat'], '末段不稳定')
        self.assertFalse(result['candidate_available'])

    def test_old_multiple_repeats_not_silently_reinterpreted_as_one_measurement(self):
        for repeats in (0, 2, 3, True):
            report = cycle_report(repeats=repeats)
            with self.subTest(repeats=repeats), self.assertRaisesRegex(ValueError, '每点只测量一次'):
                analyze_t90_waits(report)

    def test_falling_edge_supported_for_all_channels(self):
        for channel in range(4):
            report = cycle_report(channel=channel)
            for item in report['captures']:
                item['baseline_adc'][channel] = 1000
                for stamp, row in zip(TIMES, item['direct_adc']):
                    row[channel] = 1000 if stamp < 650 else 400
            self.assertEqual(analyze_t90_waits(report)['candidate_point_delays_us'], [725] * 63)

    def test_over_limit_not_capped_and_incomplete_wrap_not_applied(self):
        report = cycle_report()
        for item in report['captures']:
            if item['route_target_order'] == 0:
                for stamp, row in zip(TIMES, item['direct_adc']):
                    row[1] = 100 if stamp < 20000 else 1000
        result = analyze_t90_waits(report)
        self.assertFalse(result['candidate_available'])
        self.assertIsNone(result['candidate_point_delays_us'][0])
        self.assertEqual(result['points'][0]['required_wait_us'], 20075)
        self.assertEqual(result['points'][0]['reasons'], ['required_wait_exceeds_protocol_limit'])
        report = cycle_report()
        report['reverse_wrap_tested'] = False
        report['cycle_wrap_included'] = False
        report['captures'] = [r for r in report['captures'] if r['route_target_order']]
        result = analyze_t90_waits(report)
        self.assertFalse(result['candidate_available'])
        self.assertTrue(result['first_point_unmeasured'])

    def test_missing_shutter_duplicate_bad_order_invalid_samples_current_rejected(self):
        for mode in ('shutter', 'missing', 'order', 'index', 'style', 'nan', 'current'):
            report = cycle_report()
            if mode == 'shutter': report['exact_soa_shutter_ack'] = False
            elif mode == 'missing': report['captures'].pop()
            elif mode == 'order': report['captures'][0]['route_source_order'] = 0
            elif mode == 'index': report['captures'][0]['target_fullband_index'] += 1
            elif mode == 'style': report['captures'][0]['fast_dac_writes'] = True
            elif mode == 'nan': report['captures'][0]['direct_adc'][0][1] = float('nan')
            elif mode == 'current': report['route'][0]['codes'][0] = 63352
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                analyze_t90_waits(report)

    def test_atomic_apply_preflight_all_bindings_and_tampered_waits(self):
        report = cycle_report()
        result = analyze_t90_waits(report)
        options = dict(channel=1, selector=2, fast=False, machine_id='machine_1')
        self.assertEqual(checked_t90_waits(result, report['route'], **options), [725] * 63)
        for field, value in [('channel', 0), ('selector', 3), ('fast', True), ('machine_id', 'machine_3')]:
            changed = dict(options, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                checked_t90_waits(result, report['route'], **changed)
        result['candidate_point_delays_us'][0] = 50
        with self.assertRaises(ValueError): checked_t90_waits(result, report['route'], **options)

    def test_old_or_inconsistent_reference_summary_cannot_be_applied(self):
        report = cycle_report()
        result = analyze_t90_waits(report)
        options = dict(channel=1, selector=2, fast=False, machine_id='machine_1')
        for field, value in (
                ('schema', 'temporary_point_90pct_waits_v1'),
                ('schema', 'temporary_point_zero_reference_90pct_waits_v2'),
                ('schema', 'temporary_point_zero_reference_single_90pct_waits_v3'),
                ('schema', 'temporary_point_zero_reference_single_90pct_no_floor_waits_v4'),
                ('voltage_threshold_v', .1), ('low_voltage_wait_us', 325),
                ('criterion', 'all_repeat_observed_suffix_within_max_5_codes_or_10pct_step'),
                ('criterion', 'single_transition_observed_suffix_within_max_5_codes_or_10pct_target_adc'),
                ('tolerance_floor_adc', 5.), ('tolerance_fraction', .05),
                ('repeats', 3), ('repeats', True), ('measurement_mode', 'three_repeats_per_point'),
                ('calculation_start_adc', 100.), ('reference_model', 'previous_point_adc'),
                ('capture_mode', 'restart_from_zero')):
            changed = copy.deepcopy(result)
            changed[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, '请重新测量'):
                checked_t90_waits(changed, report['route'], **options)


class T90WidgetTests(unittest.TestCase):
    setUpClass = widget_tests.WidgetTests.__dict__['setUpClass']

    def setUp(self):
        widget_tests.WidgetTests.setUp(self)
        probe = patch('capture_temporary_half_height_transients.capture_target_level',
                      side_effect=lambda device,target,tag,**kwargs: voltage_probe(target,**kwargs))
        probe.start()
        self.addCleanup(probe.stop)

    def window(self):
        with patch.dict(os.environ, {'FBG_STUDIO_BETA': '1'}):
            window = TemporaryTestWindow(None)
        self.addCleanup(window.close)
        window.channel_combo.setCurrentIndex(window.channel_combo.findData(1))
        window.plan = selected()[-1]
        window.render_plan(ready=True)
        return window

    def measured(self, window, report=None):
        window.worker = SimpleNamespace(output='t90.json')
        window.switch_timing_ready(report or cycle_report(window.plan['rows']))
        window.worker = None
        window.refresh_wait_controls()

    def test_low_voltage_result_and_wait_editor_show_fixed_wait_not_failed_t90(self):
        from PyQt5 import QtWidgets
        window=self.window()
        self.measured(window,gated_report([100]*63))
        self.assertIn('0点×1次90%实测',window.status.text())
        self.assertTrue(window.apply_beta_t90_waits())
        self.assertEqual(window.point_waits,[300]*63)
        def inspect(dialog):
            table=dialog.findChild(QtWidgets.QTableWidget)
            self.assertEqual(table.item(0,1).text(),'不测（≤0.2 V）')
            self.assertEqual(table.item(0,3).text(),'300')
            self.assertIn('只做稳态电压预检',table.item(0,4).text())
            dialog.reject()
            return QtWidgets.QDialog.Rejected
        with patch.object(QtWidgets.QDialog,'exec_',inspect):window.show_switch_timing_results()
        def inspect_waits(dialog):
            table=dialog.findChild(QtWidgets.QTableWidget)
            self.assertIn('固定300',table.item(0,3).text())
            dialog.reject()
            return QtWidgets.QDialog.Rejected
        with patch.object(QtWidgets.QDialog,'exec_',inspect_waits):window.configure_point_waits()

    def test_worker_skips_all_low_voltage_adjacent_transients_but_saves_real_probes(self):
        import app_JDSU as app
        report=cycle_report()
        feedback=MagicMock();feedback._feedback_command_and_ack.return_value=True
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(app,'EqualIntervalWorker',return_value=feedback), \
             patch('benchmark_stress_realtime._safe_disarm',return_value=True), \
             patch('benchmark_stress_realtime._close_shutter',return_value=True), \
             patch('capture_temporary_half_height_transients.capture_target_level',
                   side_effect=lambda device,target,tag,**kw:voltage_probe(target,adc=327,**kw)) as probe, \
             patch('capture_temporary_half_height_transients.capture_custom') as capture:
            output=Path(folder)/'low.json'
            worker=SwitchTimingWorker(MagicMock(timeout=.03),{'rows':report['route']},output,32,
                                      point_delays_us=[850]*63)
            worker.run()
            saved=json.loads(output.read_text(encoding='utf-8'))
        capture.assert_not_called()
        self.assertEqual(probe.call_count,63)
        self.assertTrue(saved['complete'])
        self.assertTrue(saved['exact_soa_shutter_ack'])
        self.assertEqual(saved['t90_waits']['candidate_point_delays_us'],[300]*63)
        self.assertTrue(all('direct_adc' not in record for record in saved['captures']))
        self.assertTrue(all(len(record['voltage_probe']['direct_adc'])==33 for record in saved['captures']))

    def test_worker_only_measures_high_voltage_targets_and_keeps_original_predecessors(self):
        import app_JDSU as app
        report=cycle_report()
        rows=report['route'];order={row['index']:i for i,row in enumerate(rows)}
        high=[i for i in range(63) if i%2]
        feedback=MagicMock();feedback._feedback_command_and_ack.return_value=True
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(app,'EqualIntervalWorker',return_value=feedback), \
             patch('benchmark_stress_realtime._safe_disarm',return_value=True), \
             patch('benchmark_stress_realtime._close_shutter',return_value=True), \
             patch('capture_temporary_half_height_transients.capture_target_level',
                   side_effect=lambda device,target,tag,**kw:voltage_probe(
                       target,adc=1000 if order[target['index']] in high else 327,**kw)), \
             patch('capture_temporary_half_height_transients.capture_custom',
                   side_effect=[report['captures'][i] for i in high]) as capture:
            output=Path(folder)/'mixed.json'
            worker=SwitchTimingWorker(MagicMock(timeout=.03),{'rows':rows},output,32,
                                      point_delays_us=[850]*63)
            worker.run();saved=json.loads(output.read_text(encoding='utf-8'))
        self.assertEqual(capture.call_count,31)
        for call,i in zip(capture.call_args_list,high):
            self.assertEqual(call.args[1],rows[(i-1)%63]);self.assertEqual(call.args[2],rows[i])
        self.assertTrue(saved['complete'])
        self.assertEqual(saved['t90_waits']['candidate_point_delays_us'],[725 if i in high else 300 for i in range(63)])

    def test_measurement_alone_does_not_apply_then_one_click_enables_exact_63_waits(self):
        window = self.window()
        old = copy.deepcopy(window.point_waits)
        self.measured(window)
        self.assertFalse(window.use_point_waits.isChecked())
        self.assertEqual(window.point_waits, old)
        self.assertTrue(window.beta_apply_t90_waits.isEnabled())
        window.beta_apply_t90_waits.click()
        self.assertTrue(window.use_point_waits.isChecked())
        self.assertEqual(window.point_waits, [725] * 63)
        self.assertEqual(window.point_wait_route, window.point_route_key())
        self.assertFalse(window.first_delay.isEnabled())
        self.assertFalse(window.boundary_delay.isEnabled())
        self.assertIn('已一键应用63点', window.status.text())
        self.assertIsNone(window.worker)  # Application does not turn on the laser.

    def test_unreliable_point_blocks_all_application_without_modifying_old_table(self):
        window = self.window()
        report = cycle_report(window.plan['rows'])
        report['captures'][0]['direct_adc'][-1][1] = 3000
        self.measured(window, report)
        old = copy.deepcopy(window.point_waits)
        self.assertFalse(window.apply_beta_t90_waits())
        self.assertEqual(window.point_waits, old)
        self.assertFalse(window.use_point_waits.isChecked())
        self.assertIn('第1点', window.status.text())
        self.assertNotIn('重复测量', window.status.text())

    def test_restored_old_algorithm_result_is_labeled_and_cannot_modify_waits(self):
        from PyQt5 import QtWidgets
        window = self.window()
        self.measured(window)
        window.plan['switch_90pct_timing']['t90_waits']['schema'] = 'temporary_point_90pct_waits_v1'
        old = copy.deepcopy(window.point_waits)
        def inspect(dialog):
            texts = ' '.join(label.text() for label in dialog.findChildren(QtWidgets.QLabel))
            self.assertIn('旧的', texts)
            self.assertIn('需重新测量', texts)
            dialog.reject()
            return QtWidgets.QDialog.Rejected
        with patch.object(QtWidgets.QDialog, 'exec_', inspect): window.show_switch_timing_results()
        self.assertFalse(window.apply_beta_t90_waits())
        self.assertEqual(window.point_waits, old)
        self.assertFalse(window.use_point_waits.isChecked())
        self.assertIn('旧结果未应用', window.status.text())

    def test_route_gain_or_channel_change_rejects_stale_result(self):
        for mode in ('route', 'gain', 'channel', 'busy'):
            window = self.window()
            self.measured(window)
            if mode == 'route': window.plan['rows'][0]['codes'][0] -= 1
            elif mode == 'gain': window.analog_gain.setCurrentIndex(3)
            elif mode == 'channel': window.channel_combo.setCurrentIndex(0)
            elif mode == 'busy': window.worker = SimpleNamespace()
            old = copy.deepcopy(window.point_waits)
            self.assertFalse(window.apply_beta_t90_waits())
            self.assertEqual(window.point_waits, old)
            self.assertFalse(window.use_point_waits.isChecked())
            window.worker = None

    def test_applied_waits_are_used_by_real_scan_request_and_worker_not_just_display(self):
        import app_JDSU as app
        import serial
        from temporary_test_mode import command as encode
        window = self.window()
        self.measured(window)
        self.assertTrue(window.apply_beta_t90_waits())
        usb = serial.Serial(port=None)
        self.addCleanup(usb.close)
        with patch.object(app, 'ser', usb), patch.object(app, 'ser_open', False), \
             patch.object(app, 'switch_mode_enable', True), patch.object(app, 'ensure_serial_open'), \
             patch('temporary_test_widget.TemporaryWorker') as worker_type, \
             patch('temporary_test_widget.command', wraps=encode) as command:
            window.start_scan()
            self.assertEqual(command.call_args.kwargs['point_delays_us'], [725]*63)
            self.assertEqual(worker_type.call_args.kwargs['point_delays_us'], [725]*63)
            self.assertEqual(worker_type.call_args.kwargs['boundary_extra_us'], 0)
            self.assertTrue(worker_type.call_args.kwargs['single_sample'])
            worker_type.return_value.start.assert_called_once_with()
            window.finished()

    def test_partial_measurement_does_not_emit_usable_waits_and_still_closes_shutter(self):
        import app_JDSU as app
        report = cycle_report()
        feedback = MagicMock()
        feedback._feedback_command_and_ack.return_value = True
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(app, 'EqualIntervalWorker', return_value=feedback), \
             patch('benchmark_stress_realtime._safe_disarm', return_value=True) as disarm, \
             patch('benchmark_stress_realtime._close_shutter', return_value=True) as shutter, \
             patch('capture_temporary_half_height_transients.capture_custom',
                   side_effect=[report['captures'][0], TimeoutError('incomplete')]):
            worker = SwitchTimingWorker(MagicMock(timeout=.03), {'rows': report['route']},
                                        Path(folder)/'t90.json', 32, point_delays_us=[850]*63)
            worker.run()
            saved = json.loads((Path(folder)/'t90.json').read_text(encoding='utf-8'))
        self.assertFalse(saved['complete'])
        self.assertNotIn('t90_waits', saved)
        self.assertEqual(len(saved['captures']), 1)
        self.assertEqual(disarm.call_count, 2)
        self.assertEqual(shutter.call_count, 2)

    def test_dialog_all_63_including_first_wrap_and_apply_button(self):
        from PyQt5 import QtWidgets
        window = self.window()
        self.measured(window)
        def inspect(dialog):
            texts = ' '.join(label.text() for label in dialog.findChildren(QtWidgets.QLabel))
            self.assertIn('63点90%实测', texts)
            self.assertNotIn('3轮', texts)
            self.assertIn('允许偏差不设5码下限', texts)
            table = dialog.findChild(QtWidgets.QTableWidget)
            self.assertEqual(table.rowCount(), 63)
            self.assertEqual(table.columnCount(), 5)
            self.assertNotIn('目标点', [table.horizontalHeaderItem(i).text() for i in range(5)])
            self.assertNotIn('来源点', [table.horizontalHeaderItem(i).text() for i in range(5)])
            self.assertEqual(table.item(0, 0).text(), '450.0')
            self.assertEqual(table.item(0, 1).text(), '650.0')
            self.assertEqual(table.item(0, 3).text(), '725')
            self.assertIn('最终稳定电压 0.610352 V', table.item(0, 4).text())
            self.assertNotIn('最终ADC', table.item(0, 4).text())
            self.assertIn('允许偏差±100码', table.item(0, 4).text())
            self.assertIn('末段最大偏差 0码', table.item(0, 4).text())
            buttons = dialog.findChild(QtWidgets.QDialogButtonBox)
            apply = next(b for b in buttons.buttons() if b.text() == '一键应用到逐点等待')
            apply.click()
            self.assertTrue(window.use_point_waits.isChecked())
            return QtWidgets.QDialog.Accepted
        with patch.object(QtWidgets.QDialog, 'exec_', inspect): window.show_switch_timing_results()

    def test_result_table_shows_actual_failure_values_not_only_reason_labels(self):
        from PyQt5 import QtWidgets
        window = self.window()
        report = cycle_report()
        item = report['captures'][0]
        item['sample_start_after_target_write_us'][-3:] = [80000, 95000, 100000]
        for row, value in zip(item['direct_adc'][-3:], (850, 1000, 1100)):
            row[1] = value
        self.measured(window, report)
        before = copy.deepcopy(window.point_waits)
        def inspect(dialog):
            dialog.show()
            for _ in range(3): self.app.processEvents()
            table = dialog.findChild(QtWidgets.QTableWidget)
            self.assertTrue(table.wordWrap())
            self.assertGreater(table.columnWidth(4), 400)
            self.assertLess(table.rowHeight(0), 300)  # No pre-layout, screen-high row.
            details = table.item(0, 4).text()
            self.assertIn('末段最大偏差 150码（15.00%），超50码', details)
            self.assertIn('末段波动（最大−最小） 250码（25.00%），超150码', details)
            self.assertIn('稳定末段2个样本，要求≥3个，少1个', details)
            self.assertIn('稳定观察跨度5ms，要求≥20ms，差15ms', details)
            self.assertEqual(table.item(0, 4).toolTip(), details)
            dialog.reject()
            return QtWidgets.QDialog.Rejected
        with patch.object(QtWidgets.QDialog, 'exec_', inspect): window.show_switch_timing_results()
        self.assertEqual(window.point_waits, before)
        self.assertFalse(window.use_point_waits.isChecked())

    def test_old_current_formula_capture_can_fill_display_details_without_mutation(self):
        window = self.window()
        report = cycle_report()
        self.measured(window, report)
        timing = copy.deepcopy(window.plan['switch_90pct_timing'])
        for point in timing['points']:
            for metric in point['individual_repeats']:
                for key in ('tail_max_deviation_from_target_adc', 'tolerance_adc', 'tail_adc',
                            'tail_sample_times_after_target_write_us', 'tail_span_us',
                            'observed_stable_sample_count', 'observed_stable_span_us', 'signal_peak_adc'):
                    metric.pop(key)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'raw.json'
            path.write_text(json.dumps(report), encoding='utf-8')
            timing['source'] = str(path)
            before = copy.deepcopy(timing)
            displayed = window._t90_display_timing(timing)
            self.assertIn('tail_max_deviation_from_target_adc', displayed['points'][0]['individual_repeats'][0])
            self.assertEqual(timing, before)
            report['machine_id'] = 'machine_3'
            path.write_text(json.dumps(report), encoding='utf-8')
            self.assertIs(window._t90_display_timing(timing), timing)
        self.assertIs(window._t90_display_timing(timing), timing)  # Missing file has no invented detail.

    def test_wait_editor_shows_current_90pct_measurements_and_applied_waits_including_wrap(self):
        from PyQt5 import QtWidgets
        window = self.window()
        self.measured(window)
        self.assertTrue(window.apply_beta_t90_waits())
        def inspect(dialog):
            table = dialog.findChild(QtWidgets.QTableWidget)
            self.assertEqual(table.rowCount(), 63)
            self.assertEqual(table.item(0, 3).text(), '90%稳定 650 µs')
            self.assertIn('回绕', table.item(0, 1).text())
            self.assertEqual([s.value() for s in dialog.findChildren(QtWidgets.QSpinBox)], [725]*63)
            dialog.reject()
            return QtWidgets.QDialog.Rejected
        with patch.object(QtWidgets.QDialog, 'exec_', inspect): window.configure_point_waits()

    def test_worker_measures_all_63_once_and_safe_shutdown(self):
        import app_JDSU as app
        report = cycle_report()
        captures = [dict(r) for r in report['captures']]
        port = MagicMock(timeout=.03)
        feedback = MagicMock()
        feedback._feedback_command_and_ack.return_value = True
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(app, 'EqualIntervalWorker', return_value=feedback), \
             patch('benchmark_stress_realtime._safe_disarm', return_value=True), \
             patch('benchmark_stress_realtime._close_shutter', return_value=True), \
             patch('capture_temporary_half_height_transients.capture_custom', side_effect=captures) as capture:
            output = Path(folder)/'beta_data'/'outputs'/'temporary_test'/'t90.json'
            self.assertFalse(output.parent.exists())
            worker = SwitchTimingWorker(port, {'rows': report['route']}, output,
                                        32, point_delays_us=[850]*63)
            messages = []
            worker.message.connect(messages.append)
            worker.run()
            saved = json.loads(output.read_text(encoding='utf-8'))
        self.assertEqual(capture.call_count, 63)
        self.assertEqual(capture.call_args_list[0].args[1], report['route'][-1])
        self.assertEqual(capture.call_args_list[0].args[2], report['route'][0])
        self.assertTrue(saved['complete'])
        self.assertTrue(saved['exact_disarm_ack'])
        self.assertTrue(saved['exact_soa_shutter_ack'])
        self.assertEqual(saved['repeats'], 1)
        self.assertEqual(saved['t90_waits']['repeats'], 1)
        self.assertEqual(len(saved['captures']), 63)
        self.assertTrue(all(item['repeat'] == 0 for item in saved['captures']))
        self.assertIn('第1/1轮', messages[-1])
        self.assertNotIn('/3轮', ' '.join(messages))
        self.assertEqual(saved['t90_waits']['candidate_point_delays_us'], [725]*63)

    def test_legacy_forward_only_worker_keeps_its_original_three_repeats(self):
        import app_JDSU as app
        report = cycle_report(repeats=3)
        captures = [dict(r) for r in report['captures'] if r['route_target_order']]
        feedback = MagicMock()
        feedback._feedback_command_and_ack.return_value = True
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(app, 'EqualIntervalWorker', return_value=feedback), \
             patch('benchmark_stress_realtime._safe_disarm', return_value=True), \
             patch('benchmark_stress_realtime._close_shutter', return_value=True), \
             patch('capture_temporary_half_height_transients.capture_custom', side_effect=captures) as capture:
            output = Path(folder)/'legacy.json'
            worker = SwitchTimingWorker(MagicMock(timeout=.03), {'rows': report['route']}, output, 32)
            worker.run()
            saved = json.loads(output.read_text(encoding='utf-8'))
        self.assertEqual(capture.call_count, 186)
        self.assertEqual(saved['repeats'], 3)
        self.assertFalse(saved['reverse_wrap_tested'])
        self.assertNotIn('t90_waits', saved)
        self.assertTrue(saved['complete'])
        self.assertTrue(saved['exact_disarm_ack'])
        self.assertTrue(saved['exact_soa_shutter_ack'])

    def test_beta_control_and_status_texts_describe_single_measurement(self):
        window = self.window()
        self.assertIn('测量一次', window.test_switch_timing.toolTip())
        self.assertIn('允许偏差没有5码下限', window.test_switch_timing.toolTip())
        self.assertIn('每点只测量一次', window.beta_apply_t90_waits.toolTip())
        self.assertNotIn('3轮', window.beta_notice.text())
        self.measured(window)
        self.assertIn('63点×1次', window.status.text())

    def test_output_preflight_fails_before_optical_commands_and_still_cleans_up(self):
        report = cycle_report()
        port = MagicMock(timeout=.03)
        with tempfile.TemporaryDirectory() as folder, \
             patch('capture_ch1_teacher_spectrum.save_checkpoint', side_effect=PermissionError('read only')), \
             patch('benchmark_stress_realtime._safe_disarm', return_value=True) as disarm, \
             patch('benchmark_stress_realtime._close_shutter', return_value=True) as shutter, \
             patch('capture_temporary_half_height_transients.capture_custom') as capture:
            worker = SwitchTimingWorker(port, {'rows': report['route']}, Path(folder)/'t90.json',
                                        32, point_delays_us=[850]*63)
            messages, results = [], []
            worker.message.connect(messages.append)
            worker.timing.connect(results.append)
            worker.run()
        port.write.assert_not_called()  # No work-mode / light / DAC commands.
        capture.assert_not_called()
        disarm.assert_called_once_with(port)
        shutter.assert_called_once_with(port)
        self.assertFalse(results)
        self.assertIn('保存失败', messages[-1])
        self.assertIn('停采和关光均已确认', messages[-1])

    def test_periodic_and_final_disk_errors_do_not_hide_shutdown_failure(self):
        import app_JDSU as app
        report = cycle_report()
        feedback = MagicMock()
        feedback._feedback_command_and_ack.return_value = True
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(app, 'EqualIntervalWorker', return_value=feedback), \
             patch('capture_ch1_teacher_spectrum.save_checkpoint',
                   side_effect=[None, OSError('disk full'), OSError('disk full')]), \
             patch('benchmark_stress_realtime._safe_disarm', side_effect=[True, False]) as disarm, \
             patch('benchmark_stress_realtime._close_shutter', side_effect=[True, True]) as shutter, \
             patch('capture_temporary_half_height_transients.capture_custom',
                   side_effect=[dict(r) for r in report['captures']]) as capture:
            port = MagicMock(timeout=.03)
            worker = SwitchTimingWorker(port, {'rows': report['route']}, Path(folder)/'t90.json',
                                        32, point_delays_us=[850]*63)
            messages, results = [], []
            worker.message.connect(messages.append)
            worker.timing.connect(results.append)
            worker.run()
        self.assertEqual(capture.call_count, 10)
        self.assertEqual(disarm.call_count, 2)
        self.assertEqual(shutter.call_count, 2)
        self.assertFalse(results)
        self.assertEqual(port.timeout, .03)
        self.assertIn('保存失败', messages[-1])
        self.assertIn('未完整确认', messages[-1])

    def test_cleanup_exception_still_attempts_shutter_and_persists_failed_ack(self):
        import app_JDSU as app
        report = cycle_report()
        feedback = MagicMock()
        feedback._feedback_command_and_ack.return_value = True
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(app, 'EqualIntervalWorker', return_value=feedback), \
             patch('benchmark_stress_realtime._safe_disarm',
                   side_effect=[True, OSError('link lost')]), \
             patch('benchmark_stress_realtime._close_shutter', return_value=True) as shutter, \
             patch('capture_temporary_half_height_transients.capture_custom',
                   side_effect=[dict(r) for r in report['captures']]):
            output = Path(folder)/'new'/'temporary_test'/'t90.json'
            worker = SwitchTimingWorker(MagicMock(timeout=.03), {'rows': report['route']}, output,
                                        32, point_delays_us=[850]*63)
            results = []
            worker.timing.connect(results.append)
            worker.run()
            saved = json.loads(output.read_text(encoding='utf-8'))
        self.assertFalse(saved['complete'])
        self.assertFalse(saved['exact_disarm_ack'])
        self.assertTrue(saved['exact_soa_shutter_ack'])
        self.assertEqual(shutter.call_count, 2)
        self.assertFalse(results)


class T90DiagnosticTests(unittest.TestCase):
    def point(self, values=None, times=None):
        report = cycle_report()
        if values:
            for row, value in zip(report['captures'][0]['direct_adc'][-3:], values):
                row[1] = value
        if times:
            report['captures'][0]['sample_start_after_target_write_us'][-3:] = times
        return analyze_t90_waits(report)['points'][0]

    def test_final_stable_value_uses_shared_voltage_conversion_without_changing_adc(self):
        for code, voltage in ((0, '0.000000'), (1, '0.000610'), (309, '0.188599'),
                              (1000, '0.610352'), (2048, '1.250000'), (4095, '2.499390')):
            with self.subTest(adc=code):
                point = self.point((code, code, code))
                before = copy.deepcopy(point)
                text = t90_point_diagnostic_text(point)
                self.assertEqual(text.splitlines()[0], f'最终稳定电压 {voltage} V')
                self.assertNotIn('最终ADC', text)
                self.assertEqual(point['target_adc'], code)
                self.assertEqual(point, before)

    def test_peak_to_peak_excess_does_not_mislabel_maximum_deviation_as_exceeding(self):
        point = self.point((920, 1000, 1080))
        text = t90_point_diagnostic_text(point)
        self.assertIn('允许偏差±100码（10.00%）', text)
        self.assertIn('末段最大偏差 80码（8.00%），未超限', text)
        self.assertIn('末段波动（最大−最小） 160码（16.00%），超60码', text)
        self.assertIn('末3个ADC：920 / 1000 / 1080', text)
        self.assertEqual(point, json.loads(json.dumps(point, allow_nan=False)))

    def test_actual_sample_and_duration_deficits_are_displayed(self):
        text = t90_point_diagnostic_text(self.point((850, 1000, 1100), (80000, 95000, 100000)))
        self.assertIn('稳定末段2个样本，要求≥3个，少1个', text)
        self.assertIn('稳定观察跨度5ms，要求≥20ms，差15ms', text)
        text = t90_point_diagnostic_text(self.point(times=(98000, 99000, 100000)))
        self.assertIn('末3样本跨度2ms，要求≥20ms，差18ms', text)

    def test_wait_limit_and_clipping_report_actual_values(self):
        report = cycle_report()
        for stamp, row in zip(TIMES, report['captures'][0]['direct_adc']):
            row[1] = 100 if stamp < 20000 else 1000
        point = analyze_t90_waits(report)['points'][0]
        self.assertIn('所需等待20075µs，上限15000µs，超5075µs', t90_point_diagnostic_text(point))
        report = cycle_report()
        report['captures'][0]['direct_adc'][0][1] = 4090
        self.assertIn('ADC饱和：最大4090码，判定阈值4080码',
                      t90_point_diagnostic_text(analyze_t90_waits(report)['points'][0]))

    def test_zero_voltage_is_labeled_fixed_wait_without_fabricated_percentage(self):
        report = cycle_report()
        for row in report['captures'][0]['direct_adc']:
            row[1] = 0
        report['captures'][0]['direct_adc'][-1][1] = 1
        text = t90_point_diagnostic_text(analyze_t90_waits(report)['points'][0])
        self.assertIn('固定300 µs', text)
        self.assertIn('不是实测90%时间', text)
        self.assertNotIn('inf', text)

    def test_missing_old_fields_are_not_invented_from_available_range(self):
        point = self.point((920, 1000, 1080))
        for key in ('tail_max_deviation_from_target_adc', 'observed_stable_sample_count',
                    'observed_stable_span_us'):
            point['individual_repeats'][0].pop(key)
        text = t90_point_diagnostic_text(point)
        self.assertIn('160码', text)
        self.assertNotIn('末段最大偏差', text)
        self.assertIn('旧记录缺少部分', text)


if __name__ == '__main__':
    unittest.main()
