import copy
import unittest

from temporary_route_balance_validation import (compare_full_route, compare_pipeline_styles,
                                                 choose_fastest_comparison)
from test_temporary_symmetric_seven import selected
from test_temporary_test_mode import reference


def capture(wait=15000, shift=0, period_us=1000000):
    # This fixed synthetic ADC vector describes a uniform seven-row window;
    # don't couple its geometry to a changing automatic-selection policy.
    table=reference()[0]
    rows=[dict(group=1,index=point.index,target_nm=point.target_nm,
               measured_nm=point.measured_nm,codes=list(point.codes))
          for point in (table[100+offset] for offset in (-9,-6,-3,0,3,6,9))]
    values=(80,450,900,1500,900,450,80)
    return dict(complete=True, disarm_ack=True, shutter_ack=True,
                continuous=False, received_frames=32, plan=dict(rows=rows),
                single_sample=True, fast_dac_writes=True,
                hardware_identity=dict(serial_number='synthetic_test_fixture_not_a_board',
                                       boot_id=123,firmware_version=0x10073),
                stationary_setup_confirmed=True,
                signal_channel=0, feedback_selector=2, point_delays_us=[wait]*7,
                frames=[dict(sequence=n,kind='data',firmware=0x10073,
                             adc_channel=0,feedback_selector=2,i2c_errors=0,spi_errors=0,
                             cycle_start_us=n*period_us,single_sample=True,
                             records=[dict(adc_code=v+(shift if i==2 else 0))
                                      for i,v in enumerate(values)]) for n in range(32)])


class RouteBalanceTests(unittest.TestCase):
    def test_height_paired_route_with_coordinate_consistent_raw_signal_still_qualifies(self):
        _,_,values,plan=selected()
        before,fast,after=capture(),capture(850),capture()
        rows=plan['rows'][:7]
        for report in (before,fast,after):
            report['plan']['rows']=copy.deepcopy(rows)
            for frame in report['frames']:
                frame['records']=[dict(adc_code=float(values[row['index']])) for row in rows]
        self.assertTrue(compare_full_route(before,fast,after)['relative_accuracy_passed'])

    def test_equal_waveform_accepts_relative_only_includes_first_and_wrap(self):
        result=compare_full_route(capture(),capture(850,period_us=100000),capture())
        self.assertTrue(result['relative_accuracy_passed'])
        self.assertFalse(result['absolute_wavelength_accuracy_verified'])
        self.assertTrue(result['first_and_wrap_samples_included'])
        self.assertEqual(result['rate']['mean_hz'],10)

    def test_faster_distorted_waveform_not_chosen(self):
        okay=compare_full_route(capture(),capture(850,period_us=100000),capture())
        distorted=compare_full_route(capture(),capture(50,shift=400,period_us=20000),capture())
        self.assertFalse(distorted['relative_accuracy_passed'])
        self.assertIn('raw_waveform_differs_from_slow_reference',distorted['reasons'])
        self.assertIs(choose_fastest_comparison([okay,distorted]),okay)
        self.assertIsNone(choose_fastest_comparison([distorted]))

    def test_jittery_high_mean_does_not_beat_better_p99_throughput(self):
        steady=dict(relative_accuracy_passed=True,rate=dict(mean_hz=30,p99_period_ms=34))
        jittery=dict(relative_accuracy_passed=True,rate=dict(mean_hz=50,p99_period_ms=100))
        self.assertIs(choose_fastest_comparison([steady,jittery]),steady)

    def test_one_bad_frame_not_hidden_by_median_and_first_point_not_masked(self):
        fast=capture(850)
        fast['frames'][23]['records'][0]['adc_code']=200
        result=compare_full_route(capture(),fast,capture())
        self.assertFalse(result['relative_accuracy_passed'])
        self.assertGreater(result['max_raw_waveform_tolerance_ratio'],1)

    def test_moving_reference_refuses_recommendation(self):
        result=compare_full_route(capture(),capture(850),capture(shift=60))
        self.assertFalse(result['relative_accuracy_passed'])
        self.assertIn('reference_moving_or_noisy',result['reasons'])

    def test_large_center_difference_is_reported_without_a_pm_gate(self):
        from unittest.mock import patch
        import numpy as np
        import temporary_route_balance_validation as validation
        raw=np.ones((16,7))*1000
        rate=dict(mean_hz=10, p99_period_ms=100, measured_frames=16, firmware=0x10073)
        observed=[(raw, np.ones((16,1))*1530, True, rate),
                  (raw, np.ones((16,1))*1530.100, True, rate),
                  (raw, np.ones((16,1))*1530.040, True, rate)]
        with patch.object(validation, '_observations', side_effect=observed):
            result=compare_full_route(capture(),capture(850),capture())
        self.assertTrue(result['relative_accuracy_passed'])
        self.assertAlmostEqual(result['reference_center_drift_pm'],40)
        self.assertAlmostEqual(result['max_relative_center_error_pm'],100)
        self.assertNotIn('center_limit_pm', result['criteria'])
        self.assertTrue(result['criteria']['center_difference_is_report_only'])
        self.assertEqual(result['reasons'],[])

    def test_dark_capture_cannot_prove_optical_accuracy(self):
        dark=capture()
        for f in dark['frames']:
            for r in f['records']:r['adc_code']=2
        result=compare_full_route(dark,copy.deepcopy(dark),copy.deepcopy(dark))
        self.assertFalse(result['relative_accuracy_passed'])
        self.assertIn('reference_peak_shape_unresolved',result['reasons'])

    def test_incomplete_bus_errors_stale_scope_shutdown_and_reference_wait_rejected(self):
        for mode in ('missing','scope','gain','coords','style','bus','firmware','shutdown',
                     'reference_wait','identity','stationary'):
            with self.subTest(mode=mode):
                pre,fast,post=capture(),capture(850),capture()
                if mode=='missing':fast['frames'].pop()
                elif mode=='scope':fast['plan']['rows'][0]['codes'][0]-=1
                elif mode=='gain':fast['feedback_selector']=0
                elif mode=='coords':fast['plan']['rows'][0]['measured_nm']+=.001
                elif mode=='style':fast['fast_dac_writes']=False
                elif mode=='bus':fast['frames'][25]['spi_errors']=1
                elif mode=='firmware':post['frames'][25]['firmware']+=1
                elif mode=='shutdown':fast['shutter_ack']=False
                elif mode=='identity':fast['hardware_identity']['serial_number']='different_board'
                elif mode=='stationary':fast['stationary_setup_confirmed']=False
                else:pre['point_delays_us'][0]=850
                with self.assertRaises(ValueError):compare_full_route(pre,fast,post)


class LiveToolPreflightTests(unittest.TestCase):
    def test_wait_candidates_include_slower_accuracy_candidates_as_well_as_fast_ones(self):
        from benchmark_temporary_wait_balance_live import DEFAULT_DWELLS_US
        self.assertIn(850, DEFAULT_DWELLS_US)
        self.assertIn(15000, DEFAULT_DWELLS_US)
        self.assertTrue(any(850 < wait < 15000 for wait in DEFAULT_DWELLS_US))
        self.assertTrue(any(wait < 850 for wait in DEFAULT_DWELLS_US))
        self.assertEqual(len(DEFAULT_DWELLS_US), len(set(DEFAULT_DWELLS_US)))

    def test_no_light_and_stationary_flags_means_no_hardware_access(self):
        import contextlib,io
        from unittest.mock import patch
        import benchmark_temporary_wait_balance_live as live
        with patch('sys.argv',['benchmark','--session','missing.json','--output','unused']), \
                patch.object(live,'verify_usb') as verify, \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):live.main()
        verify.assert_not_called()

    def test_old_firmware_refuses_single_adc_before_opening_acquisition_or_output(self):
        import tempfile,json
        from pathlib import Path
        from unittest.mock import patch
        import benchmark_temporary_wait_balance_live as live
        with tempfile.TemporaryDirectory() as root:
            folder=Path(root)
            plan=selected()[-1];plan['signal_channel']=0
            session=folder/'session.json'
            session.write_text(json.dumps(dict(plan=plan,signal_channel=0,machine_id='machine_3')))
            output=folder/'trial'
            identity=dict(serial_number='207933893232',firmware_version=0x10072,
                          boot_id=123,max_image_size=229376,bootloader_ready=True)
            with patch('sys.argv',['benchmark','--session',str(session),'--output',str(output),
                                   '--allow-light','--stationary-setup']), \
                    patch.object(live,'verify_usb',return_value=identity), \
                    patch.object(live,'ThermalSerial') as serial_open:
                with self.assertRaisesRegex(RuntimeError,'v1.0.115'):live.main()
                serial_open.assert_not_called()
                self.assertFalse(output.exists())


class PipelineStyleTests(unittest.TestCase):
    def legacy(self):
        result = capture()
        result.update(single_sample=False, fast_dac_writes=False)
        for frame in result['frames']:
            frame.update(single_sample=False, fast_dac_writes=False)
            frame['records'] = [dict(first_code=r['adc_code'], second_code=r['adc_code'])
                                for r in frame['records']]
        return result

    def test_identical_signal_explicitly_qualifies_style_but_not_absolute_accuracy(self):
        result = compare_pipeline_styles(self.legacy(), capture(), self.legacy())
        self.assertTrue(result['relative_sampling_style_consistency_passed'])
        self.assertFalse(result['absolute_wavelength_accuracy_verified'])
        self.assertEqual(result['candidate_style'], dict(single_sample=True, fast_dac_writes=True))
        with self.assertRaises(ValueError):
            compare_full_route(self.legacy(), capture(), self.legacy())

    def test_style_change_that_shifts_waveform_cannot_be_qualified_by_same_style_references(self):
        changed = capture(shift=400)
        # Same-style references would hide the optimization's change.
        self.assertTrue(compare_full_route(changed, copy.deepcopy(changed),
                                          copy.deepcopy(changed))['relative_accuracy_passed'])
        result = compare_pipeline_styles(self.legacy(), changed, self.legacy())
        self.assertFalse(result['relative_sampling_style_consistency_passed'])
        self.assertIn('raw_waveform_differs_from_slow_reference', result['reasons'])

    def test_short_wait_cannot_be_misrepresented_as_sampling_style_qualification(self):
        with self.assertRaises(ValueError):
            compare_pipeline_styles(self.legacy(), capture(850), self.legacy())

    def test_route_gain_board_or_reference_style_change_is_rejected(self):
        for mode in ('route', 'gain', 'identity', 'reference_style'):
            with self.subTest(mode=mode):
                before, optimized, after = self.legacy(), capture(), self.legacy()
                if mode == 'route': optimized['plan']['rows'][0]['codes'][0] -= 1
                elif mode == 'gain': optimized['feedback_selector'] = 0
                elif mode == 'identity': optimized['hardware_identity']['boot_id'] += 1
                else: after['single_sample'] = True
                with self.assertRaises(ValueError):
                    compare_pipeline_styles(before, optimized, after)

    def test_dark_or_moving_legacy_reference_does_not_qualify_style(self):
        for mode in ('dark', 'moving'):
            before, optimized, after = self.legacy(), capture(), self.legacy()
            if mode == 'dark':
                for report in (before, optimized, after):
                    for frame in report['frames']:
                        for record in frame['records']:
                            for key in record: record[key] = 2
            else:
                for frame in after['frames']:
                    frame['records'][2]['second_code'] += 60
            self.assertFalse(compare_pipeline_styles(before, optimized, after)
                             ['relative_sampling_style_consistency_passed'])


class PipelineLiveWorkflowTests(unittest.TestCase):
    def exercise(self, *, shifted_style, firmware=0x10074):
        import contextlib
        import io
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import benchmark_temporary_wait_balance_live as live

        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            plan = capture()['plan']
            plan.update(peaks_nm=[plan['rows'][3]['measured_nm']], signal_channel=0)
            source = folder/'session.json'
            source.write_text(json.dumps(dict(plan=plan, signal_channel=0, machine_id='machine_3')))
            output = folder/'trial'
            optimized = capture(shift=400 if shifted_style else 0)
            samples = [PipelineStyleTests().legacy(), optimized, PipelineStyleTests().legacy(),
                       capture(), capture(850, period_us=100000), capture()]
            for sample in samples:
                sample['fast_dac_writes'] = False
            identity = dict(capture()['hardware_identity'], firmware_version=firmware)
            for sample in samples:
                sample['hardware_identity'] = dict(identity)
                for frame in sample['frames']:
                    frame['firmware'] = identity['firmware_version']
            with patch('sys.argv', ['benchmark', '--session', str(source), '--output', str(output),
                                   '--cycles', '32', '--dwells-us', '850',
                                   '--allow-light', '--stationary-setup']), \
                    patch.object(live, 'verify_usb', return_value=identity), \
                    patch.object(live, 'run_capture', side_effect=samples) as run, \
                    patch.object(live, 'ThermalSerial') as hardware_open, \
                    contextlib.redirect_stdout(io.StringIO()):
                if shifted_style:
                    with self.assertRaisesRegex(RuntimeError, 'Optimized sampling disagrees'):
                        live.main()
                else:
                    self.assertEqual(live.main(), 0)
                hardware_open.assert_not_called()
            summary = json.loads((output/'summary.json').read_text())
            return summary, run.call_args_list

    def test_failed_style_qualification_does_not_run_short_wait_sweep(self):
        summary, calls = self.exercise(shifted_style=True)
        self.assertEqual(len(calls), 3)
        self.assertFalse(summary['complete'])
        self.assertFalse(summary['sampling_style_qualification']
                         ['relative_sampling_style_consistency_passed'])
        self.assertEqual(summary['comparisons'], [])
        self.assertNotIn('fastest_relative_accuracy_candidate', summary)

    def test_approved_style_keeps_legacy_qualification_separate_from_wait_sweep(self):
        summary, calls = self.exercise(shifted_style=False)
        self.assertEqual(len(calls), 6)
        self.assertTrue(summary['complete'])
        for index in (0, 2):
            self.assertFalse(calls[index].kwargs['single_adc'])
            self.assertFalse(calls[index].kwargs['fast_dac_writes'])
        for index in (0, 1, 2, 3, 5):
            self.assertEqual(calls[index].args[2], [15000]*7)
        self.assertEqual(calls[4].args[2], [850]*7)
        self.assertTrue(summary['sampling_style_qualification']
                        ['relative_sampling_style_consistency_passed'])
        self.assertEqual(summary['fastest_relative_accuracy_candidate']['point_delays_us'], [850]*7)
        self.assertFalse(summary['sampling_style_qualification']['candidate_style']['fast_dac_writes'])
        self.assertFalse(summary['absolute_wavelength_accuracy_verified'])

    def test_v118_unmerged_profile_is_explicitly_compatible_without_applying_waits(self):
        summary, calls = self.exercise(shifted_style=False, firmware=0x10076)
        self.assertTrue(summary['complete'])
        self.assertEqual(summary['hardware_identity']['firmware_version'], 0x10076)
        self.assertFalse(summary['waits_applied'])
        self.assertEqual(len(calls), 6)

    def test_v118_merge_and_wrong_expected_version_fail_before_acquisition(self):
        import tempfile, json, contextlib, io
        from pathlib import Path
        from unittest.mock import patch
        import benchmark_temporary_wait_balance_live as live
        for options in (['--fast-dac-writes'], ['--expected-version', '0x10075']):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as root:
                folder = Path(root)
                plan = capture()['plan']
                plan.update(peaks_nm=[plan['rows'][3]['measured_nm']], signal_channel=0)
                source = folder/'session.json'
                source.write_text(json.dumps(dict(plan=plan, signal_channel=0, machine_id='machine_3')))
                output = folder/'trial'
                identity = dict(capture()['hardware_identity'], firmware_version=0x10076)
                with patch('sys.argv', ['benchmark', '--session', str(source), '--output', str(output),
                        '--cycles', '32', '--dwells-us', '850', '--allow-light', '--stationary-setup', *options]), \
                        patch.object(live, 'verify_usb', return_value=identity), \
                        patch.object(live, 'run_capture') as capture_hardware, \
                        contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, 'no light commanded'): live.main()
                    capture_hardware.assert_not_called()
                    self.assertFalse(output.exists())


if __name__=='__main__':unittest.main()
