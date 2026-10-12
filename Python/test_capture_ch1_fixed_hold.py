import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from concurrent.futures import Future

import capture_ch1_fixed_hold as hold


def fixture():
    rows = []
    for i in range(801):
        codes = [100, 101, 0, 40 + (10 if i < 100 else 0), 0, 0]
        raw = b'\xff\xff\x02\x02' + b''.join(c.to_bytes(2, 'big') for c in codes) + bytes(4)
        rows.append(dict(received_s=i/100, codes=codes, wire_hex=raw.hex()))
    return dict(complete=True, point=17, repeat=0, selector=2, elapsed_s=8., adc=rows, target_nm=1540.04,
                meter=[dict(start_s=.3, end_s=.9, reading=dict(wavelength_nm=1540.04, power_mw=1.7,
                                                             peak_count=1, side_mode_suppression_db=None))])


class FixedHoldTests(unittest.TestCase):
    def test_hold_body_cannot_rewrite_dac_or_purge_stream(self):
        source = inspect.getsource(hold.collect_hold)
        self.assertNotIn('_command_and_ack', source)
        self.assertNotIn('reset_input_buffer', source)
        self.assertNotIn('.write(', source)

    def test_dark_control_has_no_positive_dac_or_meter(self):
        source = inspect.getsource(hold.execute_dark)
        self.assertNotIn('checked_command(', source)
        self.assertNotIn('AQ6150B(', source)
        self.assertLess(source.index('_close_shutter(device)'), source.index('for selector in'))

    def test_summary_detects_early_offset_without_bandwidth_claim(self):
        result = hold.summarize_hold(fixture())
        self.assertEqual(result['early_minus_late_codes'], 10)
        self.assertEqual(result['ch1_after2s_halfsecond_median_span'], 0)
        self.assertFalse(result['physical_bandwidth_verified'])
        self.assertFalse(result['training_eligible'])

    def test_raw_redecode_and_incomplete_protection(self):
        for change in ('raw', 'complete', 'time', 'meter_time', 'meter_nan'):
            r = fixture()
            if change == 'raw': r['adc'][0]['codes'][3] = 1
            elif change == 'complete': r['complete'] = False
            elif change == 'time': r['adc'][10]['received_s'] = -1
            elif change == 'meter_time': r['meter'][0]['end_s'] = .1
            else: r['meter'][0]['reading']['power_mw'] = float('nan')
            with self.subTest(change=change), self.assertRaises(ValueError):
                hold.summarize_hold(r)

    def test_reader_keeps_draining_while_meter_future_is_pending(self):
        tick = 0
        def clock():
            nonlocal tick
            tick += 10000000
            return tick
        raw = bytes.fromhex(fixture()['adc'][0]['wire_hex'])
        reads = []
        def frames():
            reads.append(tick)
            return (raw,)
        class Deferred:
            def __init__(self): self.until = tick+500000000
            def done(self): return tick >= self.until
            def result(self): return {'test': 'completed'}
        executor = SimpleNamespace(submit=lambda *_: Deferred())
        record = dict(target_nm=1540.)
        with patch.object(hold.time, 'monotonic_ns', side_effect=clock), patch.object(hold.time, 'sleep'):
            hold.collect_hold(SimpleNamespace(_read_available_frames=frames), None, executor, record, 8.)
        self.assertGreater(len(reads), 350)
        self.assertGreaterEqual(len(record['meter']), 7)
        self.assertFalse(record['dac_rewritten_during_hold'])

    def test_stream_loss_aborts(self):
        ticks = iter(range(0, 10000000000, 300000000))
        with patch.object(hold.time, 'monotonic_ns', side_effect=lambda: next(ticks)), \
                patch.object(hold.time, 'sleep'):
            with self.assertRaisesRegex(TimeoutError, 'ADC stopped'):
                hold.collect_hold(SimpleNamespace(_read_available_frames=lambda: ()), None,
                                  SimpleNamespace(submit=lambda *_: Future()), dict(target_nm=1540.), 8.)

    def test_preflight_blocks_invalid_or_existing_output_before_hardware(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(hold.serial, 'Serial') as serial:
            p = Path(tmp)/'exists.json'
            p.touch()
            with self.assertRaises(FileExistsError): hold.execute(p)
            for duration in (0, 7.9, 21, float('nan')):
                with self.assertRaises(ValueError): hold.execute(Path(tmp)/'new.json', duration_s=duration)
            for points in ((1, 1, 3), (0, 17, 18), (16, 17, 45), (16, 17), (True, 17, 18)):
                with self.assertRaises(ValueError): hold.execute(Path(tmp)/'new.json', points=points)
            serial.assert_not_called()

    def test_failed_meter_setup_still_shutters(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(hold.serial, 'Serial'), \
                patch.object(hold, '_safe_disarm', return_value=True), \
                patch.object(hold, '_close_shutter', return_value=True) as shutter, \
                patch.object(hold, 'AQ6150B', side_effect=RuntimeError('offline')):
            p = Path(tmp)/'failure.json'
            with self.assertRaisesRegex(RuntimeError, 'offline'): hold.execute(p)
            shutter.assert_called_once()
            r = json.loads(p.read_text(encoding='utf-8'))
            self.assertFalse(r['complete'])
            self.assertTrue(r['safety_cleanup']['exact_soa_shutter_ack'])


if __name__ == '__main__': unittest.main()
