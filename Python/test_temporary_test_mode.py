import binascii
from dataclasses import dataclass
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import temporary_test_mode as t


@dataclass
class Point:
    index: int
    target_nm: float
    measured_nm: float
    codes: tuple = (50000, 45000, 8000, 2000, 3000)


def reference():
    x = np.linspace(1525, 1565, 2001)
    y = sum(np.exp(-.5*((x-center)/.12)**2) for center in np.linspace(1527, 1563, 9))
    return [Point(i, float(v), float(v)+.001) for i, v in enumerate(x)], x, y


def frame(sequence=16):
    wire = bytearray(451)
    wire[:4] = bytes((217, 157, 14, int(sequence >= 16)))
    for offset, value in ((4, 123), (8, 0x10035), (12, sequence), (16, 1000000), (20, 45000), (24, 456)):
        wire[offset:offset+4] = value.to_bytes(4, 'big')
    wire[28:34] = bytes((24, 45, 16, 2, 0, 2))
    wire[40:85] = bytes(range(45))
    for i in range(45):
        for offset, value in zip((0, 2, 4, 6), (100, 102, i*1000+100, i*1000+800)):
            wire[85+i*8+offset:87+i*8+offset] = value.to_bytes(2, 'big')
    return seal(wire)


def variable_frame(point_count, sequence=16, *, route_crc=456):
    wire = bytearray(46 + point_count * 9)
    wire[:4] = bytes((217, 157, 14, int(sequence >= 16)))
    for offset, value in ((4, 123), (8, 0x1004c), (12, sequence),
                          (16, 1000000), (20, point_count * 1000),
                          (24, route_crc)):
        wire[offset:offset + 4] = value.to_bytes(4, 'big')
    wire[28:34] = bytes((24, point_count, 16, 2, 0, 2))
    wire[40:40 + point_count] = bytes(range(point_count))
    start = 40 + point_count
    for i in range(point_count):
        for offset, value in zip((0, 2, 4, 6),
                                 (100, 102, i * 1000 + 100, i * 1000 + 800)):
            at = start + i * 8 + offset
            wire[at:at + 2] = value.to_bytes(2, 'big')
    return seal(wire)


def seal(wire):
    wire = bytearray(wire)
    wire[-6:-2] = binascii.crc32(wire[:-6]).to_bytes(4, 'big')
    wire[-2:] = b'\x9d\xd9'
    return bytes(wire)


class TemporaryTests(unittest.TestCase):
    def test_three_detected_peaks_create_real_15_point_protocol(self):
        table, x, _ = reference()
        y = sum(np.exp(-.5 * ((x - center) / .12) ** 2)
                for center in (1530., 1542., 1554.))
        plan = t.select_points(table, x, y)
        self.assertEqual(plan['peak_count'], 3)
        self.assertEqual(plan['point_count'], 15)
        self.assertEqual(len(plan['rows']), 15)
        self.assertEqual(plan['schema'], 'temporary_test_variable_plan_v2')
        packet = t.command(plan['rows'], 32, 123,
                           point_delays_us=[650] * 15)
        self.assertEqual(packet[:6], bytes((255, 255, 3, 10, 4, 15)))
        self.assertEqual(packet[12:16], b'TVR!')
        self.assertEqual(packet[568:572], b'PWT1')
        self.assertFalse(any(packet[602:662]))
        route_crc = binascii.crc32(packet[:662])
        wire = variable_frame(15, route_crc=route_crc)
        parsed = t.decode(wire, 123, route_crc, point_delays_us=[650] * 15,
                          point_count=15)
        self.assertEqual(len(parsed['records']), 15)
        self.assertIn('peak_at_edge', t.quality(parsed))
        self.assertEqual(len(list(t.extract_frames(bytearray(wire)))), 1)

    def test_t45_uses_dedicated_145_ma_ceiling_only(self):
        table, x, y = reference()
        rows = t.select_points(table, x, y)['rows']
        for row in rows:
            row['codes'] = [63351, 63351, 32767, 24575, 24575]
        packet = t.command(rows, 32, 123)
        self.assertEqual(int.from_bytes(packet[26:28], 'big'), 63351)
        self.assertEqual(int.from_bytes(packet[28:30], 'big'), 63351)

        rows[0]['codes'][0] = 63352
        with self.assertRaisesRegex(ValueError, 'DAC current limit exceeded'):
            t.command(rows, 32, 123)

    def test_saved_default_plan_is_complete_and_offline_only(self):
        plan = t.load_saved_plan(
            Path(__file__).with_name('temporary_test_default_plan.json')
        )
        self.assertEqual(len(plan['rows']), 45)
        self.assertEqual(len(plan['peaks_nm']), 9)
        self.assertEqual(plan['hidden_points'], 0)
        self.assertFalse(plan['optical_accuracy_verified'])
        self.assertTrue(plan['selection_validation_required'])
        self.assertTrue(all(
            len(set(np.diff([
                row['index'] for row in plan['rows'][group * 5:group * 5 + 5]
            ]))) == 1
            for group in range(9)
        ))

    def test_all_equal_spaced_points_are_inside_prominence_half_height(self):
        table,x,y=reference()
        plan=t.select_points(table,x,y,spacing_nm=0,baseline_fraction=.1)
        self.assertEqual(len(plan['rows']),45)
        self.assertEqual(plan['selection_height_fraction'], .5)
        self.assertEqual(
            plan['selection_rule'],
            'five_equal_interval_points_at_or_above_half_height',
        )
        for group, check in enumerate(plan['endpoint_checks']):
            rows=plan['rows'][group*5:group*5+5]
            indices=[r['index'] for r in rows]
            self.assertEqual(len(set(np.diff(indices))),1)
            self.assertTrue(check['all_selected_at_or_above_half_height'])
            self.assertTrue(all(
                value >= check['half_height_adc'] - 1e-9
                for value in check['selected_envelope_adc']
            ))
            self.assertGreaterEqual(rows[0]['measured_nm'], check['fwhm_left_nm'])
            self.assertLessEqual(rows[-1]['measured_nm'], check['fwhm_right_nm'])

    def test_continuous_bounded_memory_clock_wrap_and_manual_stop(self):
        table, x, y = reference()
        plan = t.select_points(table, x, y)
        packet = t.command(plan['rows'], 0, 123)
        self.assertEqual(packet[16:18], b'\0\0')
        route_crc = binascii.crc32(packet[24:564])
        port = MagicMock(timeout=.03)
        port.write.return_value = 808
        def make_wire(seq):
            wire = bytearray(frame(seq))
            wire[16:20] = ((0xffff0000 + seq*45000) & 0xffffffff).to_bytes(4, 'big')
            wire[24:28] = route_crc.to_bytes(4, 'big')
            return seal(wire)
        port.read.side_effect = [make_wire(seq) for seq in range(520)]
        received = []
        with tempfile.TemporaryDirectory() as folder, \
             patch('temporary_test_mode.time.time_ns', return_value=123), \
             patch('app_JDSU.EqualIntervalWorker') as worker, \
             patch('benchmark_stress_realtime._safe_disarm', return_value=True), \
             patch('benchmark_stress_realtime._close_shutter', return_value=True):
            worker.return_value._feedback_command_and_ack.return_value = True
            result = t.execute_device(port, plan, Path(folder)/'run.json', cycles=0,
                                      on_frame=received.append, should_stop=lambda:len(received)>=520)
            self.assertTrue(result['complete'])
            self.assertTrue(result['stopped_by_user'])
            self.assertTrue(result['shutter_ack'])
            self.assertEqual(result['received_frames'], 520)
            self.assertEqual(len(result['frames']), 512)
            self.assertEqual(len(Path(result['frame_log']).read_text(encoding='utf-8').splitlines()), 520)
            self.assertGreater(received[-1]['cycle_start_us'], 0xffffffff)
            self.assertTrue(all(b['cycle_start_us'] > a['cycle_start_us'] for a,b in zip(received,received[1:])))

    def test_analog_gain_encoding_and_echo(self):
        table, x, y = reference()
        rows = t.select_points(table, x, y)['rows']
        for selector in range(4):
            packet = t.command(rows, 32, 123, feedback_selector=selector)
            self.assertEqual(packet[19], 0 if selector == 2 else selector+1)
            self.assertEqual(int.from_bytes(packet[564:568], 'big'), binascii.crc32(packet[:564]))
            wire = bytearray(frame())
            wire[33] = selector
            self.assertEqual(t.decode(seal(wire), 123, 456, feedback_selector=selector)['feedback_selector'], selector)
            with self.assertRaises(ValueError):
                t.decode(seal(wire), 123, 456, feedback_selector=(selector+1)%4)
        for selector in (-1, 4, True, 2.0):
            with self.assertRaises(ValueError):
                t.command(rows, 32, 123, feedback_selector=selector)

    def test_selected_adc_channel_uses_backward_compatible_v5(self):
        table, x, y = reference()
        rows = t.select_points(table, x, y)['rows']
        packet = t.command(rows, 32, 123, feedback_selector=3, adc_channel=0)
        self.assertEqual(packet[:6], bytes((255, 255, 3, 10, 5, 45)))
        self.assertEqual(packet[12:16], b'TVC!')
        self.assertEqual(packet[19], 3)

        wire = bytearray(frame())
        wire[3] = 1  # CH0 is encoded as zero in the v5 status high bits.
        wire[32:34] = bytes((3, 0))
        parsed = t.decode(seal(wire),
                          123, 456, feedback_selector=3, adc_channel=0)
        self.assertEqual(parsed['adc_channel'], 0)
        self.assertEqual(parsed['signal_channel_name'], 'CH0')

        packet = t.command(rows, 32, 123, feedback_selector=2, adc_channel=2)
        self.assertEqual(packet[19], 0x20)  # CH2, fixed 2 kOhm selector zero.
        wire = bytearray(frame())
        wire[3] = 1 | (2 << 2)
        wire[32:34] = b'\0\0'
        parsed = t.decode(seal(wire), 123, 456,
                          feedback_selector=0, adc_channel=2)
        self.assertEqual(parsed['adc_channel'], 2)
        with self.assertRaises(ValueError):
            t.decode(seal(wire), 123, 456)

    def test_v5_channel_selection_is_wired_into_firmware_source(self):
        root = Path(__file__).resolve().parents[1]
        header = (root / 'JDSU/Core/Inc/candidate_route_protocol.h').read_text(
            encoding='utf-8'
        )
        firmware = (root / 'JDSU/Core/Src/ms5614t.c').read_text(
            encoding='utf-8'
        )
        self.assertIn(
            'version == 3U || version == 4U || version == 5U', header
        )
        self.assertIn("packet[14] == 'C'", header)
        self.assertIn('out->adcChannel = adcChannel;', header)
        self.assertIn(
            'ADC_ReadMask((uint8_t)(1U << selectedAdcChannel), samples);',
            firmware,
        )
        self.assertIn('samples[selectedAdcChannel]', firmware)

    def test_matching_adc_pairs_never_prove_stability(self):
        parsed = t.decode(frame(), 123, 456)
        for row in parsed['records']:
            row['first_code'] = row['second_code']
        self.assertFalse(t.quality(parsed)['stability_verified'])
        parsed['records'][0]['second_code'] = 1000
        self.assertTrue(t.quality(parsed)['pair_difference_large'])
        self.assertEqual(parsed['records'][0]['second_code'], 1000)

    def test_calibration_import_requires_complete_matching_records(self):
        table,x,y=reference()
        plan=t.select_points(table,x,y)
        payload=dict(complete=True,plan=plan,rows=[dict(index=r['index'],success=True,
                     calibrated_codes=r['codes']) for r in plan['rows']])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'calibration.json'
            path.write_text(json.dumps(payload),encoding='utf-8')
            self.assertFalse(t.load_calibrated_plan(path)['optical_accuracy_verified'])
            payload['complete']=False
            path.write_text(json.dumps(payload),encoding='utf-8')
            with self.assertRaises(ValueError):t.load_calibrated_plan(path)
            payload['complete']=True
            payload['plan']['rows'][0]['measured_nm']+=.01
            path.write_text(json.dumps(payload),encoding='utf-8')
            with self.assertRaises(ValueError):t.load_calibrated_plan(path)

    def test_equal_five_point_selection_from_nine_peaks(self):
        table, x, y = reference()
        plan = t.select_points(table, x, y)
        self.assertEqual(len(plan['rows']), 45)
        for group in np.array([r['index'] for r in plan['rows']]).reshape(9, 5):
            self.assertEqual(len(set(np.diff(group))), 1)
        self.assertFalse(plan['optical_accuracy_verified'])
        self.assertFalse(plan['selection_validation_required'])
        self.assertEqual(len(plan['selection_fit_checks']), 9)
        self.assertTrue(all(check['passed'] and check['r_squared'] >= .95
                            for check in plan['selection_fit_checks']))
        # A Gaussian with sigma .12 has a half-width of about .141 nm.  On the
        # 0.02 nm table the widest centered five-point window has 0.06 nm
        # adjacent spacing and remains above half height at both ends.
        self.assertAlmostEqual(plan['rows'][1]['target_nm']-plan['rows'][0]['target_nm'], .06)
        for group in range(9):
            heights = [y[row['index']] for row in plan['rows'][5*group:5*group+5]]
            self.assertTrue(heights[0] < heights[1] < heights[2] > heights[3] > heights[4])

    def test_isolated_dense_spike_does_not_move_broad_peak_center(self):
        table, x, y = reference()
        true_center = int(np.argmin(np.abs(x - 1527.0)))
        spiked = y.copy()
        spiked[true_center + 12] += 2.0
        plan = t.select_points(table, x, spiked)
        self.assertEqual(plan['rows'][2]['index'], true_center)
        self.assertEqual(
            plan['selection_center_method'],
            'dbm_prominence_width_rank_then_linear_fwhm_inner_equal_spacing',
        )

    def test_dbm_detection_keeps_weak_ninth_peak_below_linear_global_gate(self):
        table, x, _ = reference()
        centers = np.asarray([1528.04, 1530.98, 1533.66, 1537.00, 1539.66,
                              1542.50, 1545.56, 1548.74, 1551.50])
        amplitudes = np.asarray([2000., 1500., 1700., 820., 660.,
                                 470., 250., 205., 135.])
        y = np.full_like(x, 10.0)
        for center, amplitude in zip(centers, amplitudes):
            y += amplitude * np.exp(-.5 * ((x - center) / .20) ** 2)
        # The old linear global prominence gate was 8% of the first peak and
        # therefore exceeded the entire ninth peak above baseline.
        self.assertLess(amplitudes[-1], .08 * np.ptp(y))
        plan = t.select_points(table, x, y, feedback_selector=2)
        self.assertEqual(plan['peak_count'], 9)
        self.assertEqual(plan['peak_detection_domain'],
                         'estimated_optical_power_dbm')
        self.assertAlmostEqual(plan['peaks_nm'][-1], centers[-1], places=2)
        self.assertEqual(len(plan['rows']), 45)

    def test_explicit_live_validated_route_is_rebuilt_from_active_table(self):
        table, x, y = reference()
        selected = t.select_points(table, x, y)
        groups = [[row['index'] for row in selected['rows'][group*5:group*5+5]]
                  for group in range(9)]
        values = [y[index] for indices in groups for index in indices]
        rebuilt = t.build_selected_plan(table, groups, values, source='unit-live-reference')
        self.assertEqual(groups, [[row['index'] for row in rebuilt['rows'][group*5:group*5+5]]
                                  for group in range(9)])
        self.assertEqual(rebuilt['reference_source'], 'unit-live-reference')
        self.assertEqual(len(rebuilt['selection_fit_checks']), 9)
        self.assertTrue(all(check['passed'] for check in rebuilt['selection_fit_checks']))
        with self.assertRaises(ValueError):
            t.build_selected_plan(table, groups, [1]*45)

    def test_manual_dense_route_accepts_independent_nonuniform_points(self):
        table, x, y = reference()
        automatic = t.select_points(table, x, y)
        groups = [[row['index'] for row in automatic['rows'][group*5:group*5+5]]
                  for group in range(9)]
        groups[3][1] += 1
        groups[3][3] -= 1
        rebuilt = t.build_manual_dense_plan(table, groups, x, y, source='manual-drag')
        self.assertEqual(
            groups[3],
            [row['index'] for row in rebuilt['rows'][15:20]],
        )
        self.assertTrue(rebuilt['manual_selection'])
        self.assertEqual(rebuilt['reference_source'], 'manual-drag')
        self.assertNotEqual(len(set(np.diff(groups[3]))), 1)
        packet = t.command(rebuilt['rows'], 32, 123)
        self.assertEqual(len(packet), 808)

    def test_observed_stale_peak_windows_fail_selection_margin(self):
        axis = np.arange(5, dtype=float) * .1
        for values in ([21, 216, 531, 74, 44], [26, 350, 984, 748, 64]):
            check = t.assess_five_point_shape(axis, values)
            self.assertTrue(check['passed'], check)
            self.assertGreaterEqual(check['r_squared'], .95)
        for values in ([68, 314, 211, 54, 180], [100, 864, 840, 529, 133]):
            check = t.assess_five_point_shape(axis, values)
            self.assertFalse(check['passed'], check)
            self.assertEqual(check['reason'], '第三点偏离峰顶')

    def test_bad_reference_rejected(self):
        table, x, y = reference()
        for bad in (y*0, np.full_like(y, np.nan)):
            with self.assertRaises(ValueError): t.select_points(table, x, bad)

    def test_noisy_raw_preview_does_not_veto_half_height_geometry(self):
        table, x, y = reference()
        noisy = y.copy()
        center = int(np.argmin(np.abs(x - 1539.0)))
        noisy[center - 8:center + 9:2] += np.asarray(
            [.35, -.20, .42, -.18, .38, -.16, .31, -.12, .25]
        )
        plan = t.select_points(table, x, noisy)
        self.assertEqual(len(plan['rows']), 45)
        self.assertTrue(all(
            check['all_selected_at_or_above_half_height']
            for check in plan['endpoint_checks']
        ))
        self.assertEqual(
            plan['selection_validation_required'],
            not all(check['passed'] for check in plan['selection_fit_checks']),
        )

    def test_weak_satellite_is_not_a_tenth_main_peak(self):
        table, x, y = reference()
        satellite = np.exp(-.5*((x-1528.8)/.08)**2)
        plan = t.select_points(table, x, y + .15*satellite)
        self.assertEqual(plan['discarded_weak_peaks'], 1)
        with self.assertRaisesRegex(ValueError, '无法可靠区分'):
            t.select_points(table, x, y + .8*satellite)
        with self.assertRaises(ValueError): t.select_points(table, x[::-1], y)

    def test_command_crc_and_bounds(self):
        table, x, y = reference()
        rows = t.select_points(table, x, y)['rows']
        packet = t.command(rows, 256, 123)
        self.assertEqual(packet[:6], b'\xff\xff\x03\x0a\x03\x2d')
        self.assertEqual(int.from_bytes(packet[564:568], 'big'), binascii.crc32(packet[:564]))
        self.assertFalse(any(packet[568:]))
        boundary = t.command(rows,128,123,350,50,1800)
        self.assertEqual(int.from_bytes(boundary[22:24],'big'),1800)
        t.command(rows,128,123,850,50,25)  # Slow scans are now permitted.
        with self.assertRaises(ValueError):t.command(rows,128,123,350,50,3025)
        for cycles in (31, 513, True):
            with self.assertRaises(ValueError): t.command(rows, cycles, 123)
        rows[2]['index'] += 1  # Manually edited points need not be equally spaced.
        self.assertEqual(len(t.command(rows, 32, 123)), 808)
        rows[2]['index'] = rows[1]['index']
        with self.assertRaises(ValueError): t.command(rows, 32, 123)

    def test_fragmented_crc_protected_45_records(self):
        wire = frame()
        buffer = bytearray(b'noise' + wire[:130])
        self.assertEqual(list(t.extract_frames(buffer)), [])
        buffer.extend(wire[130:])
        parsed = t.decode(list(t.extract_frames(buffer))[0], 123, 456)
        self.assertEqual(len(parsed['records']), 45)
        self.assertFalse(t.quality(parsed)['optical_accuracy_verified'])
        broken = bytearray(wire); broken[100] ^= 1
        with self.assertRaises(ValueError): t.decode(broken, 123, 456)
        with self.assertRaises(ValueError): t.decode(wire, 124, 456)

    def test_invalid_timing_order_rejected(self):
        broken = bytearray(frame()); broken[97:99] = b'\x00\x00'
        with self.assertRaises(ValueError): t.decode(seal(broken), 123, 456)
        with self.assertRaises(ValueError): t.decode(frame(),123,456,boundary_extra_us=75)


if __name__ == '__main__':
    unittest.main()
