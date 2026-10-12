import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import binascii
import json
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import app_JDSU as app
import dense_scan_transport as dense


def points(count):
    return [app.FullbandAccuracyPoint(i, 1525 + i*.02, 1525 + i*.02,
                                     (100, 200, 300, 400, 500)) for i in range(count)]


def reply(packet, *, status=0):
    query = packet[3] == 12
    count = 0 if query else packet[14]
    result = bytearray(54 + count * 22)
    result[:4] = b'\xd6\x6d\x01' + bytes([0 if query else 1])
    struct.pack_into('>H', result, 4, len(result))
    result[6:10] = packet[8:12]
    result[10:12] = packet[12:14]
    result[12:14] = bytes([dense.MAX_OPS if query else count, count])
    struct.pack_into('>HI', result, 14, status, 0x1006a)
    if not query:
        result[20:24] = packet[804:808]
        result[28:32] = packet[16:20]
        result[32:34] = packet[20:22]
        result[42:44] = packet[22:24]
    result[34] = 1 if query or packet[15] & 2 else 0
    result[35] = 0x87
    elapsed = 0
    for i in range(count):
        index = int.from_bytes(packet[24 + i*16:26 + i*16], 'big')
        wait = int.from_bytes(packet[36 + i*16:40 + i*16], 'big')
        write_end = elapsed + 250
        elapsed = write_end + wait + 30
        adc = [0]*6 if index & dense.GUARD_FLAG else [index % 1000 + c*100 for c in range(6)]
        struct.pack_into('>7HII', result, 48 + i*22, index, *adc, write_end, elapsed)
    struct.pack_into('>I', result, 24, elapsed)
    result[-2:] = b'\x6d\xd6'
    struct.pack_into('>I', result, len(result)-6, binascii.crc32(result[:-6]) & 0xffffffff)
    return bytes(result)


class SimulatedBoard:
    def __init__(self):
        self.timeout = .03
        self.buffer = bytearray()
        self.writes = []

    @property
    def in_waiting(self):
        return len(self.buffer)

    def write(self, packet):
        self.writes.append(bytes(packet))
        if packet[3] in (12, 13):
            self.buffer.extend(reply(packet))
        return len(packet)

    def read(self, size):
        data = bytes(self.buffer[:min(size, 37)])  # Deliberate USB fragmentation.
        del self.buffer[:len(data)]
        return data

    def reset_input_buffer(self):
        self.buffer.clear()


class DenseScanTests(unittest.TestCase):
    def run_selected_reference(self, device, folder, **kwargs):
        from temporary_reference import acquire_selected_reference
        selected = points(45)
        rows = [dict(index=p.index, target_nm=p.target_nm,
                     measured_nm=p.measured_nm, codes=list(p.codes)) for p in selected]
        with patch.object(app, 'load_fullband_accuracy_table', return_value=selected), \
                patch.object(app, 'load_fullband_transition_guards', return_value={}), \
                patch('benchmark_stress_realtime._safe_disarm', return_value=True) as disarm, \
                patch('benchmark_stress_realtime._close_shutter', return_value=True) as shutter, \
                patch.object(app.EqualIntervalWorker, '_feedback_command_and_ack', return_value=True), \
                patch('temporary_reference.time.sleep'):
            try:
                return acquire_selected_reference(device, Path(folder)/'scan.json', selected_rows=rows, **kwargs)
            finally:
                self.assertEqual(disarm.call_count, 2)
                self.assertEqual(shutter.call_count, 2)
                self.assertEqual(device.timeout, .03)

    def test_old_firmware_selected_reference_uses_raw_single_sample_without_stability_retry(self):
        def raw(worker, point, channel, attempt):
            self.assertEqual(worker.settle_s, .010)
            return dict(index=point.index, dac_codes=list(point.codes), stable=False,
                        signal_adc_code=4095 if point.index % 2 else 0,
                        signal_saturated=bool(point.index % 2), sample_count=1)
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(dense, 'probe', return_value=None), \
                patch('temporary_reference._equal_interval_row', side_effect=raw) as read, \
                patch('temporary_reference.acquire_checked_point') as stable:
            result = self.run_selected_reference(SimulatedBoard(), folder)
        self.assertEqual(read.call_count, 45)
        stable.assert_not_called()
        self.assertTrue(result['complete'])
        self.assertEqual(result['failed_point_count'], 0)
        self.assertTrue(all(row['stability_not_evaluated'] for row in result['rows']))
        self.assertEqual(result['acquisition_backend'], 'host_single_monitor')

    def test_selected_reference_cancel_keeps_partial_raw_data_and_shuts_down(self):
        count = []
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_selected_reference(
                SimulatedBoard(), folder, on_row=lambda row, n, total: count.append(n),
                should_stop=lambda: bool(count),
            )
            self.assertTrue((Path(folder)/'scan.json').is_file())
        self.assertFalse(result['complete'])
        self.assertTrue(result['stopped_early'])
        self.assertEqual(len(result['rows']), 1)
        self.assertTrue(result['disarm_ack'] and result['shutter_ack'])

    def test_selected_batch_fault_never_replays_points_with_host_fallback(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(dense, 'acquire', side_effect=RuntimeError('CRC mismatch')), \
                patch('temporary_reference._equal_interval_row') as host:
            with self.assertRaisesRegex(RuntimeError, 'CRC mismatch'):
                self.run_selected_reference(SimulatedBoard(), folder)
            host.assert_not_called()
            report = json.loads((Path(folder)/'scan.json').read_text())
            self.assertFalse(report['complete'])
            self.assertTrue(report['disarm_ack'] and report['shutter_ack'])

    def test_selected_45_reference_is_one_10ms_batch_and_not_a_stability_window(self):
        from temporary_reference import acquire_selected_reference
        selected = points(45)
        selected_rows = [dict(index=p.index, target_nm=p.target_nm,
                             measured_nm=p.measured_nm, codes=list(p.codes)) for p in selected]
        device = SimulatedBoard()
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(app, 'load_fullband_accuracy_table', return_value=selected), \
                patch.object(app, 'load_fullband_transition_guards', return_value={}), \
                patch('benchmark_stress_realtime._safe_disarm', return_value=True) as disarm, \
                patch('benchmark_stress_realtime._close_shutter', return_value=True) as shutter, \
                patch.object(app.EqualIntervalWorker, '_feedback_command_and_ack', return_value=True), \
                patch('temporary_reference.acquire_checked_point') as stable, \
                patch('temporary_reference.time.sleep'):
            result = acquire_selected_reference(device, Path(folder)/'scan.json', selected_rows=selected_rows)
            stable.assert_not_called()
        commands = [p for p in device.writes if p[3] == 13]
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0][14:16], bytes([45, 3]))
        self.assertEqual(int.from_bytes(commands[0][16:20], 'big'), 10000)
        self.assertEqual([r['signal_adc_code'] for r in result['rows']], [100 + i for i in range(45)])
        self.assertTrue(all(r['sample_count'] == 1 and r['stable'] is None and not r['attempts']
                            for r in result['rows']))
        self.assertFalse(result['stability_filter_applied'])
        self.assertAlmostEqual(result['requested_point_dwell_s'], .45)
        self.assertAlmostEqual(result['board_acquisition_elapsed_s'], .4626)
        self.assertEqual(disarm.call_count, 2)
        self.assertEqual(shutter.call_count, 2)

    def test_command_limits_zero_padding_and_crc(self):
        ops = dense.operations(points(48), {}, 2000)
        packet = dense.command(1, 0, ops, flags=3, settle_us=2000, selectors=(0, 2), total=48)
        self.assertEqual(len(packet), 808)
        self.assertEqual(packet[792:804], bytes(12))
        self.assertEqual(int.from_bytes(packet[804:], 'big'), binascii.crc32(packet[:804]) & 0xffffffff)
        for codes in ((63352, 1, 1, 1, 1), (1, 63352, 1, 1, 1),
                      (1, 1, 32768, 1, 1), (1, 1, 1, 24576, 1),
                      (1, 1, 1, 1, 24576), (1.5, 1, 1, 1, 1)):
            with self.subTest(codes=codes), self.assertRaises(ValueError):
                dense.command(1, 0, [(0, codes, 2000)], flags=3, settle_us=2000,
                              selectors=(0, 2), total=1)

    def test_complete_2001_points_are_fresh_and_not_interpolated(self):
        device = SimulatedBoard()
        self.assertEqual(dense.probe(device)['firmware_version'], 0x1006a)
        rows = list(dense.acquire(device, points(2001), {}, settle_s=.002, selectors=(0, 2)))
        self.assertEqual([r['index'] for r in rows], list(range(2001)))
        self.assertEqual(len(device.writes)-1, 42)
        self.assertEqual(rows[1001]['adc_codes'], [1, 101, 201, 301])
        self.assertEqual(rows[1001]['monitor_codes'], [401, 501])
        self.assertEqual(device.writes[1][15], 1)
        self.assertEqual(device.writes[-1][15], 2)
        self.assertTrue(all(r['settle_s'] == .002 and r['single_sample'] for r in rows))

    def test_transition_guard_order_and_wait_preserved_not_plotted(self):
        guard = SimpleNamespace(hold_s_per_stage=.200001,
                                precondition_codes=((10, 20, 30, 40, 50), (20, 30, 40, 50, 60)))
        ops = dense.operations(points(4), {2: guard}, 2000)
        self.assertEqual([op[0] for op in ops], [0, 1, 0x8002, 0x8002, 2, 3])
        self.assertEqual(ops[2][2], 200001)
        rows = list(dense.acquire(SimulatedBoard(), points(4), {2: guard},
                                  settle_s=.002, selectors=(0, 2)))
        self.assertEqual([r['index'] for r in rows], [0, 1, 2, 3])

    def test_packet_crc_tag_sequence_timing_and_selector_checks(self):
        packet = dense.command(3, 0, dense.operations(points(2), {}, 2000), flags=3,
                               settle_us=2000, selectors=(0, 2), total=2)
        wire = reply(packet)
        for offset, value in ((6, 1), (11, 1), (32, 2), (35, 0xb7), (24, 0xff), (68, 0)):
            altered = bytearray(wire)
            altered[offset] = value
            struct.pack_into('>I', altered, len(altered)-6,
                             binascii.crc32(altered[:-6]) & 0xffffffff)
            with self.subTest(offset=offset), self.assertRaises((ValueError, RuntimeError)):
                dense.decode(bytes(altered), 3, 0, packet)
        corrupt = bytearray(wire)
        corrupt[50] ^= 1
        with self.assertRaises(ValueError):
            list(dense.extract_frames(corrupt))
        with self.assertRaises(RuntimeError):
            dense.decode(reply(packet, status=128), 3, 0, packet)

    def test_partial_frames_and_monitor_noise_resynchronize(self):
        packet = dense.command(2, 0, dense.operations(points(1), {}, 0), flags=3,
                               settle_us=0, selectors=(0, 0), total=1)
        wire = reply(packet)
        buffer = bytearray(b'\xff\xff\x02\x02' + bytes(16) + wire[:31])
        self.assertEqual(list(dense.extract_frames(buffer)), [])
        buffer.extend(wire[31:])
        self.assertEqual(list(dense.extract_frames(buffer)), [wire])

    def test_cancelled_batch_is_not_written_or_retried(self):
        device = SimulatedBoard()
        with self.assertRaises(InterruptedError):
            list(dense.acquire(device, points(100), {}, settle_s=.002,
                               selectors=(0, 2), should_stop=lambda: True))
        self.assertEqual(device.writes, [])

    def test_unsupported_firmware_receives_only_a_shuttered_capability_probe(self):
        device = SimulatedBoard()
        with patch.object(dense, '_receive', return_value=None):
            self.assertIsNone(dense.probe(device))
        self.assertEqual(len(device.writes), 1)
        self.assertEqual(device.writes[0][3], 12)
        self.assertEqual(device.writes[0][12:], bytes(796))

    def test_missing_batch_response_never_replays_dac_commands(self):
        device = SimulatedBoard()
        with patch.object(dense, '_receive', return_value=None), self.assertRaises(TimeoutError):
            list(dense.acquire(device, points(100), {}, settle_s=.002, selectors=(0, 2)))
        self.assertEqual(len(device.writes), 1)

    def test_board_fault_still_runs_exact_shutdown_and_saves_error(self):
        from temporary_reference import acquire_reference
        device = SimulatedBoard()
        original_write = device.write
        def fail_block(packet):
            original_write(packet)
            if packet[3] == 13:
                device.buffer[:] = reply(packet, status=128)
            return len(packet)
        device.write = fail_block
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(app, 'load_fullband_accuracy_table', return_value=points(99)), \
                patch.object(app, 'load_fullband_transition_guards', return_value={}), \
                patch('benchmark_stress_realtime._safe_disarm', return_value=True) as disarm, \
                patch('benchmark_stress_realtime._close_shutter', return_value=True) as shutter, \
                patch.object(app.EqualIntervalWorker, '_feedback_command_and_ack', return_value=True), \
                patch('temporary_reference.time.sleep'):
            output = Path(folder)/'failed.json'
            with self.assertRaises(RuntimeError):
                acquire_reference(device, output, acquisition_method='equal_interval_single', settle_s=.002)
            self.assertIn('Dense board acquisition failed', output.read_text(encoding='utf-8'))
        self.assertEqual(disarm.call_count, 2)
        self.assertEqual(shutter.call_count, 2)
        self.assertEqual(len(device.writes), 3)  # work mode, probe, one failed batch

    def test_dense_reference_integrates_board_backend_and_cleanup(self):
        from temporary_reference import acquire_reference
        device = SimulatedBoard()
        selected = points(99)
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(app, 'load_fullband_accuracy_table', return_value=selected), \
                patch.object(app, 'load_fullband_transition_guards', return_value={}), \
                patch('benchmark_stress_realtime._safe_disarm', return_value=True) as disarm, \
                patch('benchmark_stress_realtime._close_shutter', return_value=True) as shutter, \
                patch.object(app.EqualIntervalWorker, '_feedback_command_and_ack', return_value=True), \
                patch('temporary_reference.time.sleep'):
            result = acquire_reference(device, Path(folder)/'scan.json',
                                       acquisition_method='equal_interval_single', settle_s=.002,
                                       signal_channel=2)
        self.assertEqual(result['acquisition_backend'], 'stm32_dense_batch_v1')
        self.assertEqual(len(result['rows']), 99)
        self.assertEqual(result['rows'][7]['signal_adc_code'], 207)
        self.assertAlmostEqual(result['requested_point_dwell_s'], .198)
        self.assertTrue(result['complete'])
        self.assertEqual(disarm.call_count, 2)
        self.assertEqual(shutter.call_count, 2)
        self.assertEqual(device.timeout, .03)

    def test_checkpoint_disk_wait_does_not_block_next_scan_points(self):
        from temporary_reference import acquire_reference
        device = SimulatedBoard()
        release_disk = threading.Event()
        saved_counts = []
        def slow_save(_output, report):
            if len(report['rows']) < 501:
                self.assertTrue(release_disk.wait(2), 'Scan waited for disk before reaching its final point')
            saved_counts.append(len(report['rows']))
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(app, 'load_fullband_accuracy_table', return_value=points(501)), \
                patch.object(app, 'load_fullband_transition_guards', return_value={}), \
                patch('benchmark_stress_realtime._safe_disarm', return_value=True), \
                patch('benchmark_stress_realtime._close_shutter', return_value=True), \
                patch.object(app.EqualIntervalWorker, '_feedback_command_and_ack', return_value=True), \
                patch('capture_ch1_teacher_spectrum.save_checkpoint', side_effect=slow_save), \
                patch('temporary_reference.time.sleep'):
            result = acquire_reference(
                device, Path(folder)/'scan.json', acquisition_method='equal_interval_single',
                settle_s=.002, on_progress=lambda n, total: release_disk.set() if n == total else None)
        self.assertTrue(result['complete'])
        self.assertEqual(saved_counts, [240, 480, 501])

    def test_firmware_is_finite_current_bounded_and_fail_dark(self):
        root = Path(__file__).resolve().parents[1] / 'JDSU/Core'
        source = (root/'Src/ms5614t.c').read_text(encoding='utf-8')
        header = (root/'Inc/dense_scan_protocol.h').read_text(encoding='utf-8')
        self.assertIn('63351U,63351U,32767U,24575U,24575U', header)
        self.assertIn('out->crc != DenseScan_Crc', header)
        self.assertIn('out->waits[i] != out->settleUs', header)
        self.assertIn('nowTick-denseScanLastTick) > 5000U', source)
        self.assertIn('if(status || (!query && (request.flags & 2U)))', source)
        self.assertIn('if(!denseScanSend(length, overLan))', source)
        self.assertIn('Keep any newly arrived stop mailbox intact.', source)


if __name__ == '__main__':
    unittest.main()
