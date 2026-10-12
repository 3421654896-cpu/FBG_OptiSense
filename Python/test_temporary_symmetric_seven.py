"""Offline seven-point selection, protocol, fitting and UI tests."""
import binascii
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import temporary_test_mode as t
from temporary_peak_fit_makima import fit_peak_points, fit_temporary_frame
from test_temporary_test_mode import reference, seal
import test_temporary_test_widget as widget_tests
from temporary_test_widget import TemporaryTestWindow


def selected(stride=1, jitter=False, sigma=.16):
    table, _, _ = reference()
    if jitter:
        for row in table:
            row.measured_nm = row.target_nm + .0019 * np.sin(row.index * 1.7)
    x = np.asarray([row.measured_nm for row in table])[::stride]
    y = 30 + sum(1800 * np.exp(-.5 * ((x-c)/sigma)**2) for c in np.linspace(1527, 1563, 9))
    return table, x, y, t.auto_select_dense_plan(table, x, y)


def unpack(packet):
    bits = ''.join(f'{byte:08b}' for byte in packet[24:t.V6_CRC_OFFSET])
    result, cursor = [], 0
    for _ in range(packet[5]):
        row = []
        for width in t.V6_FIELD_BITS:
            row.append(int(bits[cursor:cursor+width], 2))
            cursor += width
        result.append(row)
    return result


def seven_frame(plan, channel=1, selector=2):
    n = len(plan['rows'])
    packet = t.command(plan['rows'], 32, 123, first_delay_us=850,
                       point_delays_us=[850]*n, adc_channel=channel, feedback_selector=selector)
    crc = binascii.crc32(packet[:t.V6_CRC_OFFSET])
    wire = bytearray(46+n*13)
    wire[:4] = bytes((217, 157, 14, 3 | ((channel if channel != 1 else 0) << 2)))
    for at, value in ((4,123),(8,0x10071),(12,16),(16,100000),(20,n*1400),(24,crc)):
        wire[at:at+4] = value.to_bytes(4,'big')
    wire[28:34] = bytes((34,n,16,2,*t._feedback_pair(channel,selector)))
    wire[40:40+n] = bytes(range(n))
    for i in range(n):
        at = 40+n+i*12
        value = (80,350,1100,1800,1100,350,80)[i%7]
        wire[at:at+2] = value.to_bytes(2,'big')
        wire[at+2:at+4] = value.to_bytes(2,'big')
        wire[at+4:at+8] = (i*1400+100).to_bytes(4,'big')
        wire[at+8:at+12] = (i*1400+1100).to_bytes(4,'big')
    return crc, seal(wire)


class SevenSelectionTests(unittest.TestCase):
    def test_full_and_strided_dense_spectra_have_three_real_height_pairs(self):
        for stride, sigma in ((1,.16), (2,.16), (5,.35)):
            table,x,y,plan = selected(stride, sigma=sigma)
            self.assertEqual((plan['point_count'],plan['peak_count'],plan['points_per_peak']),(63,9,7))
            for group in range(9):
                rows = plan['rows'][group*7:group*7+7]
                indices = np.asarray([r['index'] for r in rows])
                self.assertTrue(np.all(np.diff(indices)>0))
                check = plan['symmetry_checks'][group]
                self.assertEqual(check['left_indices'], indices[2::-1].tolist())
                self.assertEqual(check['right_indices'], indices[4:].tolist())
                self.assertTrue(all(i%stride==0 for i in indices))
                band = plan['selection_band_checks'][group]
                self.assertGreaterEqual(rows[0]['measured_nm'], band['band_left_nm'])
                self.assertLessEqual(rows[-1]['measured_nm'], band['band_right_nm'])
                self.assertEqual([r['codes'] for r in rows],[list(table[i].codes) for i in indices])
            np.testing.assert_array_equal(plan['reference_values'],y)
            self.assertFalse(plan['optical_accuracy_verified'])

    def test_every_real_point_is_in_default_makima_visual_dbm_upper_half(self):
        from scipy.interpolate import Akima1DInterpolator
        from temporary_optical_units import adc_code_to_dbm
        table, x, y, plan = selected(2, jitter=True)
        spline = Akima1DInterpolator(x, y, method='makima', extrapolate=False)
        self.assertEqual(plan['selection_height_region'], 'dbm_visual_upper_half')
        self.assertEqual(plan['selection_height_fraction'], .5)
        self.assertNotIn('selection_peak_drop_db', plan)
        self.assertEqual(plan['symmetry_basis'], 'fitted_dbm_height')
        self.assertEqual(plan['selection_fit_method'], 'makima_piecewise_cubic_dense_adc')
        self.assertEqual(plan['selection_threshold_unit'], 'dbm')
        self.assertNotIn('selection_envelope_width_nm', plan)
        for group, band in enumerate(plan['selection_band_checks']):
            self.assertAlmostEqual(band['threshold_dbm'], .5*(band['fitted_peak_dbm'] + band['baseline_dbm']))
            self.assertEqual(band['baseline_dbm'], max(band['left_base_dbm'],band['right_base_dbm']))
            self.assertLess(band['threshold_dbm'], 0)
            self.assertAlmostEqual(band['fitted_peak_adc'], float(spline(band['fitted_peak_nm'])))
            self.assertAlmostEqual(band['fitted_peak_dbm'], float(adc_code_to_dbm(
                band['fitted_peak_adc'], 2)))
            rows = plan['rows'][group*7:group*7+7]
            sample_orders = [row['index']//2 for row in rows]
            raw_dbm = np.asarray(adc_code_to_dbm(y[sample_orders], 2))
            fit_dbm = np.asarray(adc_code_to_dbm(spline([row['measured_nm'] for row in rows]), 2))
            self.assertTrue(np.all(raw_dbm >= band['threshold_dbm'] - 1e-8))
            self.assertTrue(np.all(fit_dbm >= band['threshold_dbm'] - 1e-8))
            np.testing.assert_allclose(band['selected_raw_dbm'], raw_dbm)
            np.testing.assert_allclose(band['selected_fitted_dbm'], fit_dbm)
            self.assertTrue(band['all_selected_at_or_above_threshold'])
        np.testing.assert_array_equal(plan['reference_values'], y)
        self.assertTrue(all(row['codes'] == list(table[row['index']].codes) for row in plan['rows']))

    def test_band_scales_with_peak_width_instead_of_fixed_one_nm_window(self):
        *_, narrow = selected()
        *_, wide = selected(sigma=.6)
        span = lambda plan: plan['rows'][6]['target_nm'] - plan['rows'][0]['target_nm']
        self.assertLess(span(narrow), .8)
        self.assertGreater(span(wide), 1.0)

    def test_dense_apex_matches_default_makima_without_using_discrete_peak_as_threshold(self):
        table, _, _ = reference()
        x = np.asarray([row.measured_nm for row in table])
        y = 30 + sum(1800*np.exp(-.5*((x-center)/.16)**2)
                     for center in np.linspace(1527,1563,9)+.007)
        plan = t.auto_select_dense_plan(table, x, y)
        for group, band in enumerate(plan['selection_band_checks']):
            center_index = plan['rows'][group*7+3]['index']
            orders = np.arange(center_index-3, center_index+4)
            fit = fit_peak_points(x[orders], y[orders])
            self.assertEqual(fit['fit_method'], 'makima_piecewise_cubic_7point')
            self.assertAlmostEqual(band['fitted_peak_nm'], fit['center_nm'], places=9)
            self.assertAlmostEqual(band['fitted_peak_adc'], fit['peak_adc'], places=8)
            self.assertGreater(band['fitted_peak_adc'], float(np.max(y[orders])))

    def test_visual_dbm_midpoint_is_not_linear_power_half_or_fixed_five_db(self):
        from temporary_optical_units import adc_code_to_dbm
        table, _, _ = reference()
        x = np.asarray([row.measured_nm for row in table])
        y = 600 + sum(1800*np.exp(-.5*((x-c)/.16)**2) for c in np.linspace(1527,1563,9))
        plan = t.auto_select_dense_plan(table, x, y)
        for group, band in enumerate(plan['selection_band_checks']):
            baseline_adc = 600.
            self.assertAlmostEqual(band['baseline_dbm'], float(adc_code_to_dbm(baseline_adc,2)))
            self.assertAlmostEqual(band['threshold_adc'], np.sqrt(band['fitted_peak_adc']*baseline_adc))
            self.assertNotAlmostEqual(band['peak_drop_db'], 5)
            self.assertNotAlmostEqual(band['threshold_adc'], .5*(band['fitted_peak_adc']+baseline_adc))
            rows = plan['rows'][group*7:group*7+7]
            raw = y[[row['index'] for row in rows]]
            self.assertGreaterEqual(min(raw), band['threshold_adc']-1e-8)
            self.assertLess(min(raw), 600+.5*(band['fitted_peak_adc']-600))

    def test_asymmetric_peak_pairs_heights_instead_of_wavelength_distances(self):
        from temporary_optical_units import adc_code_to_dbm
        table, _, _ = reference()
        x = np.asarray([row.measured_nm for row in table])
        y = np.full_like(x, 30.)
        for center in np.linspace(1527,1563,9):
            sigma = np.where(x<center, .12, .26)
            y += 1800*np.exp(-.5*((x-center)/sigma)**2)
        plan = t.auto_select_dense_plan(table, x, y)
        for group, band in enumerate(plan['selection_band_checks']):
            rows = plan['rows'][group*7:group*7+7]
            indices = np.asarray([row['index'] for row in rows])
            self.assertFalse(np.array_equal(indices+indices[::-1], [2*indices[3]]*7))
            self.assertGreater(band['band_right_nm']-band['fitted_peak_nm'],
                               band['fitted_peak_nm']-band['band_left_nm'])
            check = plan['symmetry_checks'][group]
            mirrored_right = 2*indices[3]-indices[2::-1]
            mirrored_difference = abs(np.asarray(adc_code_to_dbm(y[indices[2::-1]],2))
                                      - np.asarray(adc_code_to_dbm(y[mirrored_right],2)))
            self.assertLess(max(check['fitted_pair_difference_db']), 1.0)
            self.assertLess(max(check['fitted_pair_difference_db']), .5*float(max(mirrored_difference)))
            np.testing.assert_allclose(check['target_drop_db'], np.linspace(band['peak_drop_db']/3,band['peak_drop_db'],3))
            self.assertTrue(np.all(np.asarray(band['selected_raw_dbm']) >= band['threshold_dbm']-1e-8))
        np.testing.assert_array_equal(plan['reference_values'], y)

    def test_coarse_scan_fails_without_admitting_points_below_cutoff(self):
        with self.assertRaisesRegex(ValueError, '左右不足三个实测点'):
            selected(5, sigma=.10)

    def test_full_visual_half_band_near_scan_edge_does_not_need_old_one_nm_window(self):
        table, x, y, _ = selected()
        subset = x >= 1526.54
        plan = t.auto_select_dense_plan(table, x[subset], y[subset])
        self.assertEqual(plan['point_count'], 63)
        self.assertGreaterEqual(plan['rows'][0]['measured_nm'], x[subset][0])

    def test_meter_axis_is_preserved_without_faking_mirrored_wavelengths(self):
        table,_,_,plan = selected(2,True)
        self.assertEqual(plan['symmetry_basis'], 'fitted_dbm_height')
        for check in plan['symmetry_checks']:
            self.assertNotIn('paired_offsets_indices', check)
            self.assertNotIn('measured_pair_midpoint_error_pm', check)
        for row in plan['rows']:
            self.assertEqual(row['measured_nm'],table[row['index']].measured_nm)

    def test_incomplete_visual_half_band_never_borrows_from_outside_scan(self):
        table,x,y,_ = selected()
        subset = x>=1526.8
        with self.assertRaises(ValueError):
            t.auto_select_dense_plan(table,x[subset],y[subset])

    def test_saved_plan_round_trip_and_inconsistent_grouping_rejected(self):
        *_,plan = selected()
        self.assertEqual(t.validate_plan_payload(plan)['points_per_peak'],7)
        wrong = json.loads(json.dumps(plan)); wrong['points_per_peak']=5
        with self.assertRaises(ValueError): t.validate_plan_payload(wrong)
        wrong = json.loads(json.dumps(plan)); wrong['rows'][6]['group']=2
        with self.assertRaises(ValueError): t.validate_plan_payload(wrong)

    def test_quantized_height_pairs_do_not_require_any_mirrored_physical_rows(self):
        table,_,_ = reference()
        axis = np.asarray([row.measured_nm for row in table])
        keep = np.arange(2001)%2 == 0
        for center in np.linspace(1527,1563,9):
            index = int(round((center-1525)/.02))
            positions = np.arange(index+1,index+41)
            keep[positions] = positions%2 == 1
        x = axis[keep]
        y = 30+sum(1800*np.exp(-.5*((x-center)/.16)**2) for center in np.linspace(1527,1563,9))
        plan = t.auto_select_dense_plan(table,x,y)
        self.assertEqual(t.validate_plan_payload(plan)['point_count'],63)
        for check in plan['symmetry_checks']:
            self.assertTrue(all(index%2 == 0 for index in check['left_indices']))
            self.assertTrue(all(index%2 == 1 for index in check['right_indices']))

    def test_saved_height_pair_metadata_cannot_claim_false_pair_differences(self):
        *_, plan = selected()
        for field in ('fitted_pair_difference_db','left_indices','fitted_left_dbm'):
            wrong = json.loads(json.dumps(plan))
            wrong['symmetry_checks'][0][field][0] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                t.validate_plan_payload(wrong)
        wrong = json.loads(json.dumps(plan))
        wrong.pop('selection_band_checks')
        with self.assertRaises(ValueError): t.validate_plan_payload(wrong)

    def test_legacy_wavelength_mirrored_route_keeps_original_validation(self):
        table,x,y,_ = selected()
        centers = [int(round((center-1525)/.02)) for center in np.linspace(1527,1563,9)]
        groups = [[center+offset for offset in (-6,-4,-2,0,2,4,6)] for center in centers]
        legacy = t.build_manual_dense_plan(table,groups,x,y)
        legacy.update(automatic_selection=True,symmetry_basis='calibrated_table_target_wavelength')
        self.assertEqual(t.validate_plan_payload(legacy)['points_per_peak'],7)
        moved = table[groups[0][0]-1]
        legacy['rows'][0].update(index=moved.index, target_nm=moved.target_nm,
                                 measured_nm=moved.measured_nm,codes=list(moved.codes))
        with self.assertRaisesRegex(ValueError,'左右不对称'): t.validate_plan_payload(legacy)

    def test_joint_height_pair_optimizer_matches_brute_force_and_uses_distinct_rows(self):
        from itertools import combinations
        left = np.asarray([-10.3,-11.3,-12.9,-14.1,-14.9])
        right = np.asarray([-10.4,-11.7,-13.0,-14.0,-14.7,-14.98])
        levels = np.linspace(5/3,5,3)
        score = lambda pairs: sum((-10-left[i]-level)**2+(-10-right[j]-level)**2
                                  +4*(left[i]-right[j])**2 for (i,j),level in zip(pairs,levels))
        choices = [list(zip(l,r)) for l in combinations(range(len(left)),3)
                   for r in combinations(range(len(right)),3)]
        pairs = t._three_height_pairs(left,right,-10,5)
        self.assertAlmostEqual(score(pairs),min(score(choice) for choice in choices))
        self.assertEqual(len(set(i for i,_ in pairs)),3)
        self.assertEqual(len(set(j for _,j in pairs)),3)

    def test_outer_pair_is_anchored_even_when_joint_mismatch_prefers_higher_points(self):
        left = np.asarray([-10.4,-11.7,-13.2,-14.0,-14.5,-14.99])
        right = np.asarray([-10.3,-11.6,-13.3,-14.0,-14.5])
        unconstrained = t._three_height_pairs(left,right,-10,5)
        self.assertNotEqual(unconstrained[-1],(5,4))
        anchored = t._three_height_pairs(left,right,-10,5,anchor_outer=True)
        self.assertEqual(anchored[-1],(5,4))
        self.assertTrue(all(i < 5 and j < 4 for i,j in anchored[:-1]))

    def test_three_target_heights_uniformly_include_half_height_boundary(self):
        *_, plan = selected()
        for band, check in zip(plan['selection_band_checks'], plan['symmetry_checks']):
            levels = [band['fitted_peak_dbm'], *check['target_height_dbm']]
            np.testing.assert_allclose(np.diff(levels), [-band['peak_drop_db']/3]*3)
            self.assertAlmostEqual(levels[-1],band['threshold_dbm'])
            np.testing.assert_allclose(band['boundary_height_gap_db'],
                np.asarray(band['selected_fitted_dbm'])[[0,6]] - band['threshold_dbm'])

    def test_noisy_flank_cannot_place_inner_samples_below_the_anchored_lowest_pair(self):
        left = np.asarray([-14.99,-14.98,-10.1,-11.6,-13.4,-14.8])
        right = np.asarray([-10.3,-11.6,-13.3,-14.0,-14.9])
        pairs = t._three_height_pairs(left,right,-10,5,anchor_outer=True)
        self.assertEqual(pairs[-1],(5,4))
        self.assertTrue(all(left[i] >= left[5] and right[j] >= right[4] for i,j in pairs[:-1]))

    def test_actual_outer_samples_are_nearest_eligible_heights_above_half(self):
        from temporary_optical_units import adc_code_to_dbm
        from scipy.interpolate import Akima1DInterpolator
        table,x,y,plan = selected(2, jitter=True)
        spline = Akima1DInterpolator(x,y,method='makima',extrapolate=False)
        sampled_indices = np.arange(0,2001,2)
        raw_dbm = np.asarray(adc_code_to_dbm(y,2))
        fit_dbm = np.asarray(adc_code_to_dbm(spline(x),2))
        for group,band in enumerate(plan['selection_band_checks']):
            rows = plan['rows'][7*group:7*group+7]
            center = rows[3]['index']
            allowed = ((x >= band['band_left_nm']) & (x <= band['band_right_nm'])
                       & (raw_dbm >= band['threshold_dbm']-1e-8)
                       & (fit_dbm >= band['threshold_dbm']-1e-8))
            for side,chosen in ((-1,rows[0]),(1,rows[-1])):
                orders = np.flatnonzero(allowed & ((sampled_indices-center)*side > 0))
                orders = sorted(orders, key=lambda order: abs(sampled_indices[order]-center))
                eligible = orders[2:]
                closest = min(eligible,key=lambda order: fit_dbm[order]-band['threshold_dbm'])
                self.assertEqual(chosen['index'],sampled_indices[closest])
                self.assertGreaterEqual(fit_dbm[closest],band['threshold_dbm']-1e-8)
                self.assertEqual(chosen['codes'], list(table[chosen['index']].codes))

    def test_visual_half_metadata_rejects_incorrect_baseline_and_boundary_gap(self):
        *_, plan = selected()
        for field in ('baseline_dbm','peak_drop_db','height_fraction'):
            wrong = json.loads(json.dumps(plan))
            wrong['selection_band_checks'][0][field] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                t.validate_plan_payload(wrong)
        wrong = json.loads(json.dumps(plan))
        wrong['selection_band_checks'][0]['boundary_height_gap_db'][0] += .1
        with self.assertRaises(ValueError): t.validate_plan_payload(wrong)
        wrong = json.loads(json.dumps(plan))
        wrong['symmetry_checks'][0]['target_height_dbm'][-1] += .1
        with self.assertRaises(ValueError): t.validate_plan_payload(wrong)
        wrong = json.loads(json.dumps(plan)); wrong.pop('selection_height_fraction')
        with self.assertRaises(ValueError): t.validate_plan_payload(wrong)

    def test_legacy_fixed_relative_height_metadata_still_loads_without_reselection(self):
        *_, current = selected()
        legacy = json.loads(json.dumps(current))
        for key in ('selection_height_region','selection_height_fraction','selection_baseline_basis',
                    'selection_boundary_rule'):
            legacy.pop(key)
        drop = max(band['peak_drop_db'] for band in legacy['selection_band_checks'])+1
        legacy['selection_peak_drop_db'] = drop
        legacy['selection_rule'] = 'seven_real_height_paired_points_above_fitted_peak_relative_cutoff'
        for band,check in zip(legacy['selection_band_checks'],legacy['symmetry_checks']):
            band['threshold_dbm'] = band['fitted_peak_dbm']-drop
            check['target_drop_db'] = np.linspace(drop/3,drop,3).tolist()
        loaded = t.validate_plan_payload(legacy)
        self.assertEqual(loaded['rows'],legacy['rows'])
        self.assertEqual(loaded['selection_peak_drop_db'],drop)
        self.assertNotIn('selection_height_region',loaded)

    def test_fixed_808_byte_packet_preserves_every_dac_code_and_15ms_wait(self):
        *_,plan = selected()
        rng = np.random.default_rng(7)
        for row in plan['rows']:
            row['codes']=[int(rng.integers(0,limit+1)) for limit in t.LIMITS]
        plan['rows'][-1]['codes']=list(t.LIMITS)
        waits=[int(v)*25 for v in rng.integers(2,601,size=63)]; waits[-1]=15000
        packet=t.command(plan['rows'],0,123,point_delays_us=waits)
        self.assertEqual(len(packet),808)
        self.assertEqual(packet[:6],bytes((255,255,3,10,6,63)))
        self.assertEqual(packet[12:16],b'T7S!')
        self.assertEqual(int.from_bytes(packet[796:800],'big'),binascii.crc32(packet[:796]))
        self.assertFalse(any(packet[800:]))
        for got,row,wait in zip(unpack(packet),plan['rows'],waits):
            self.assertEqual(got,[row['index'],*row['codes'],wait//25])

    def test_current_ceiling_and_invalid_waits_fail_before_io(self):
        *_,plan=selected()
        for channel,limit in enumerate(t.LIMITS):
            rows=json.loads(json.dumps(plan['rows'])); rows[0]['codes'][channel]=limit+1
            with self.assertRaises(ValueError): t.command(rows,32,123)
        for wait in (0,49,851,15025):
            with self.assertRaises(ValueError):
                t.command(plan['rows'],32,123,point_delays_us=[wait]*63)

    def test_all_channels_63_point_frames_crc_and_fragmented_stream(self):
        *_,plan=selected()
        for channel,selector in ((0,3),(1,2),(2,0),(3,0)):
            crc,wire=seven_frame(plan,channel,selector)
            parsed=t.decode(wire,123,crc,first_delay_us=850,point_delays_us=[850]*63,
                            point_count=63,adc_channel=channel,feedback_selector=selector,points_per_peak=7)
            self.assertEqual(len(parsed['records']),63)
            self.assertFalse(t.quality(parsed)['peak_at_edge'])
            buffer=bytearray(b'noise'+wire[:31]); self.assertFalse(list(t.extract_frames(buffer)))
            buffer.extend(wire[31:]); self.assertEqual(list(t.extract_frames(buffer)),[wire])
            damaged=bytearray(wire); damaged[-8]^=1
            with self.assertRaises(ValueError): t.decode(damaged,123,crc,point_count=63,points_per_peak=7)

    def test_makima_uses_all_seven_raw_samples(self):
        x=np.linspace(1540,1541,7); y=np.asarray([80,320,1200,2000,1300,520,100])
        fit=fit_peak_points(x,y)
        self.assertTrue(fit['fit_available']); self.assertEqual(fit['input_point_count'],7)
        np.testing.assert_allclose(np.interp(x,fit['curve_x_nm'],fit['curve_adc']),y)
        *_,plan=selected()
        self.assertEqual(len(fit_temporary_frame(plan['rows'],np.tile(y,9))),9)

    def test_selected_reference_acquires_all_63_raw_adc_points_at_10ms(self):
        import app_JDSU as app
        from temporary_reference import acquire_selected_reference
        table,_,_,plan = selected()
        port = MagicMock(timeout=.03)
        worker = MagicMock()
        worker._feedback_command_and_ack.return_value = True
        worker._acquire_point.side_effect = [
            dict(adc_codes=(0., float(i), 0., 0.), voltages=(0., .01, 0., 0.),
                 saturated_channels=(), monitor_codes=(0., 0.), settle_s=.01, elapsed_s=.01)
            for i in range(63)]
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(app, 'load_fullband_accuracy_table', return_value=table), \
             patch.object(app, 'EqualIntervalWorker', return_value=worker) as make_worker, \
             patch('benchmark_stress_realtime._safe_disarm', return_value=True), \
             patch('benchmark_stress_realtime._close_shutter', return_value=True), \
             patch('temporary_reference.acquire_checked_point') as stable_check, \
             patch('temporary_reference.time.sleep'):
            report = acquire_selected_reference(port, Path(directory)/'reference.json',
                                                selected_rows=plan['rows'], prefer_board_batch=False)
        self.assertTrue(report['complete'])
        self.assertEqual(len(report['rows']), 63)
        self.assertEqual([row['signal_adc_code'] for row in report['rows']], list(range(63)))
        self.assertFalse(report['stability_filter_applied'])
        self.assertEqual(make_worker.call_args.kwargs['settle_s'], .01)
        stable_check.assert_not_called()

    def test_complete_mocked_stream_logs_63_points_and_confirms_shutter(self):
        import app_JDSU as app
        *_,plan = selected()
        crc, sample = seven_frame(plan)
        frames = []
        for sequence in range(32):
            wire = bytearray(sample)
            wire[3] = 2 | int(sequence >= 16)
            wire[12:16] = sequence.to_bytes(4, 'big')
            wire[16:20] = (sequence * 100000).to_bytes(4, 'big')
            frames.append(seal(wire))
        terminal = bytearray(42)
        terminal[:4] = bytes((217,157,15,0))
        terminal[4:8] = (123).to_bytes(4,'big')
        terminal[8:12] = (0x10071).to_bytes(4,'big')
        terminal[12:16] = (32).to_bytes(2,'big') * 2
        terminal[18:22] = bytes((1,1,0,2))
        terminal[32:36] = crc.to_bytes(4,'big')
        port = MagicMock(timeout=.03)
        port.write.side_effect = lambda data: len(data)
        port.read.return_value = b''.join(frames) + seal(terminal)
        worker = MagicMock()
        worker._feedback_command_and_ack.return_value = True
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(app, 'EqualIntervalWorker', return_value=worker), \
             patch('benchmark_stress_realtime._safe_disarm', return_value=True), \
             patch('benchmark_stress_realtime._close_shutter', return_value=True), \
             patch('temporary_test_mode.time.time_ns', return_value=123):
            report = t.execute_device(port, plan, Path(directory)/'stream.json', cycles=32,
                                      first_delay_us=850, point_delays_us=[850]*63)
        self.assertTrue(report['complete'])
        self.assertTrue(report['shutter_ack'])
        self.assertEqual(report['points_per_peak'], 7)
        self.assertEqual(len(report['frames']), 32)
        self.assertTrue(all(len(frame['records']) == 63 for frame in report['frames']))

    def test_firmware_expands_buffers_but_preserves_legacy_bounds(self):
        root=Path(__file__).resolve().parents[1]
        header=(root/'JDSU/Core/Inc/candidate_route_protocol.h').read_text(encoding='utf-8')
        firmware=(root/'JDSU/Core/Src/ms5614t.c').read_text(encoding='utf-8')
        self.assertIn('#define TEMPORARY_ROUTE_COUNT 45U',header)
        self.assertIn('#define TEMPORARY_ROUTE_MAX_COUNT 63U',header)
        self.assertIn('{11U,16U,16U,15U,15U,15U,10U}',header)
        self.assertIn('CandidateRoute_ParseV6(packet, out)',header)
        self.assertIn('uint32_t secondEndUs[TEMPORARY_ROUTE_MAX_COUNT]',firmware)
        self.assertIn('selectedAdcChannel != 1U',firmware)


class SevenWidgetTests(unittest.TestCase):
    setUpClass=widget_tests.WidgetTests.__dict__['setUpClass']
    setUp=widget_tests.WidgetTests.setUp

    def window(self):
        window=TemporaryTestWindow(None); self.addCleanup(window.close)
        table,x,y,_=selected(2)
        window.dense_wavelengths,window.dense_values=x,y
        window.dense_channel=window.selected_channel()
        with patch('app_JDSU.load_fullband_accuracy_table',return_value=table):
            self.assertTrue(window.auto_select_points(),window.status.text())
        return window,table,x,y

    def test_63_point_ui_waits_plot_and_fit(self):
        window,_,_,_=self.window()
        self.assertEqual((window.point_count(),window.peak_count(),window.points_per_peak()),(63,9,7))
        self.assertEqual(window.table.rowCount(),63); self.assertEqual(len(window.selection_markers),63)
        self.assertIn('dBm视觉上半段按高度均匀分层',window.status.text())
        self.assertIn('按高度配对',window.status.text())
        self.assertIn('最大配对高度差',window.status.text())
        self.assertIn('最低点锚定半高线',window.status.text())
        self.assertIn('半高线',window.auto_select_button.toolTip())
        self.assertIn('63/63',window.manual_selection_label.text())
        self.assertEqual(window._uniform_point_waits(),[850]*63)
        window.boundary_delay.setValue(1000)
        self.assertEqual(window._uniform_point_waits(),[1000 if i%7==0 else 850 for i in range(63)])
        window.latest_values=np.tile([80,320,1200,2000,1300,520,100],9).tolist()
        window.redraw_values()
        self.assertTrue(all(len(c.getData()[0])==7 for c in window.curves))
        crc, wire = seven_frame(window.plan)
        data = t.decode(wire,123,crc,first_delay_us=850,point_delays_us=[850]*63,
                        point_count=63,points_per_peak=7)
        data['quality'] = t.quality(data)
        window.update_frame(data)
        self.assertTrue(all(f['input_point_count']==7 for f in window.peak_fits))

    def test_ready_63_point_session_restores_and_legacy_45_still_loads(self):
        window,_,_,_=self.window()
        window.reference_values=np.tile([80,320,1200,2000,1300,520,100],9).tolist()
        window.reference_wavelengths=[r['measured_nm'] for r in window.plan['rows']]
        window.plan['reference_acquisition_method']='equal_interval_single'
        window.plan['reference_stability_evaluated']=False; window.plan_ready=True
        window.persistence_enabled = True
        window._save_last_route_session(ready=True)
        restored=TemporaryTestWindow(None); self.addCleanup(restored.close)
        self.assertTrue(restored._load_last_route_session(), restored.status.text())
        self.assertEqual(restored.point_count(),63); self.assertEqual(restored.points_per_peak(),7)
        self.assertTrue(restored.plan_ready); self.assertEqual(restored.reference_values,window.reference_values)
        table,x,y=reference()
        self.assertEqual(t.validate_plan_payload(t.select_points(table,x,y))['points_per_peak'],5)

    def test_undo_preserves_seven_point_manual_grouping(self):
        window,table,_,_=self.window()
        missing=window.plan['rows'][-1]['index']
        self.assertTrue(window.undo_manual_selection()); self.assertEqual(len(window.manual_selection_indices),62)
        self.assertEqual(window.manual_point_count(),63); self.assertEqual(window.points_per_peak(),7)
        with patch('app_JDSU.load_fullband_accuracy_table',return_value=table):
            self.assertTrue(window.add_manual_selection_at(table[missing].measured_nm))
        self.assertEqual(window.point_count(),63); self.assertEqual(window.points_per_peak(),7)

    def test_failed_selection_preserves_existing_route(self):
        window,table,x,_=self.window(); old_plan=window.plan; window.dense_values=np.zeros_like(x)
        with patch('app_JDSU.load_fullband_accuracy_table',return_value=table):
            self.assertFalse(window.auto_select_points())
        self.assertIs(window.plan,old_plan)

    def test_too_coarse_visual_half_selection_preserves_plan_and_applied_waits(self):
        window, table, x, y = self.window()
        old_plan = window.plan
        window.point_waits = [725]*63
        old_waits = window.point_waits
        coarse_x = x[::2]
        coarse_y = 30+sum(1800*np.exp(-.5*((coarse_x-c)/.10)**2) for c in np.linspace(1527,1563,9))
        window.dense_wavelengths, window.dense_values = coarse_x, coarse_y
        with patch('app_JDSU.load_fullband_accuracy_table',return_value=table):
            self.assertFalse(window.auto_select_points())
        self.assertIs(window.plan, old_plan)
        self.assertIs(window.point_waits, old_waits)
        self.assertIn('原有点位保留', window.status.text())


if __name__=='__main__': unittest.main()
