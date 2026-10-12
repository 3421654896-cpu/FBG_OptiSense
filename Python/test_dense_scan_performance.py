"""Deterministic timing model; never connects to or opens a physical laser."""

import os
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import app_JDSU as app
from fbg_lan_serial import FbgLanSerial


def monitor_frame(values=(111, 222, 1000, 2000, 3000, 4000)):
    frame = bytearray(app.SINGLE_VALUE_FRAME_SIZE)
    frame[:4] = b'\xff\xff\x02\x02'
    for index, value in enumerate(values):
        frame[4 + index * 2:6 + index * 2] = int(value).to_bytes(2, 'big')
    return bytes(frame)


class VirtualClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        self.now += max(0.0, duration)


class SimulatedMonitorSerial:
    """pyserial-like size blocking, 20 ms monitor cadence and 1 ms ACK delay."""

    def __init__(self, clock):
        self.clock = clock
        self.timeout = .1
        self.buffer = bytearray()
        self.next_monitor = .02
        self.pending_acks = []
        self.current_gain = 0
        self.reset_count = 0
        self.read_sizes = []

    def _pump(self):
        while True:
            ack_time = self.pending_acks[0][0] if self.pending_acks else float('inf')
            event = min(ack_time, self.next_monitor)
            if event > self.clock.now + 1e-12:
                break
            if ack_time <= self.next_monitor:
                self.buffer.extend(self.pending_acks.pop(0)[1])
            else:
                self.buffer.extend(monitor_frame(
                    (111, 222, 1000, 1000 + self.current_gain, 3000, 4000)
                ))
                self.next_monitor += .02

    @property
    def in_waiting(self):
        self._pump()
        return len(self.buffer)

    def write(self, command):
        self.current_gain = int.from_bytes(command[4:6], 'big')
        ack = bytearray(app.SINGLE_VALUE_FRAME_SIZE)
        ack[:4] = b'\xff\xff\x02\x00'
        ack[4:14] = command[4:14]
        ack[14] = app.PI11210_SOA_SOURCE_MODE
        ack[15] = 0x87  # Status present, identity OK, write accepted, no fault.
        self.pending_acks.append((self.clock.now + .001, bytes(ack)))
        return len(command)

    def read(self, size):
        self.read_sizes.append(size)
        self._pump()
        deadline = self.clock.now + self.timeout
        while len(self.buffer) < size and self.clock.now < deadline:
            ack_time = self.pending_acks[0][0] if self.pending_acks else float('inf')
            self.clock.now = min(ack_time, self.next_monitor, deadline)
            self._pump()
        result = bytes(self.buffer[:size])
        del self.buffer[:size]
        return result

    def reset_input_buffer(self):
        self._pump()
        self.buffer.clear()
        self.reset_count += 1


class LegacyReceiveWorker(app.EqualIntervalWorker):
    """The former read-64 path, kept only to benchmark the regression fix."""

    def _read_available_frames(self):
        waiting = self.port.in_waiting
        data = self.port.read(min(max(waiting, 64), 4096))
        if not data:
            return ()
        self._rx_buffer.extend(data)
        return tuple(app.extract_single_value_frames(self._rx_buffer))


def compare_receive_paths(point_count=200):
    summary = {
        'test_kind': 'offline_serial_timing_simulation',
        'physical_hardware_tested': False,
        'settle_s': .002,
        'monitor_interval_s': .02,
        'points': point_count,
    }
    all_values = []
    for name, worker_type in (
        ('before', LegacyReceiveWorker), ('after', app.EqualIntervalWorker)
    ):
        clock = VirtualClock()
        port = SimulatedMonitorSerial(clock)
        worker = worker_type(port, (), {}, settle_s=.002)
        results = []
        with patch.object(app.time, 'monotonic', clock.monotonic), \
                patch.object(app.time, 'sleep', clock.sleep):
            for index in range(point_count):
                point = app.FullbandAccuracyPoint(
                    index=index, target_nm=1525 + .02 * index,
                    measured_nm=1525 + .02 * index,
                    codes=(100 + index % 10, 200, 300, 400, 500),
                )
                result = worker._acquire_point(1, point)
                assert result['adc_codes'][1] == 1100 + index % 10
                results.append(result)
        summary[name] = {
            'mean_point_ms': 1000 * sum(r['elapsed_s'] for r in results) / point_count,
            'total_s': clock.now,
            'fresh_frame_flushes': port.reset_count,
        }
        all_values.append([r['adc_codes'] for r in results])
    summary['raw_values_identical'] = all_values[0] == all_values[1]
    return summary


class ReceivePerformanceTests(unittest.TestCase):
    def test_complete_20_byte_frame_does_not_wait_for_64_bytes(self):
        clock = VirtualClock()
        port = SimulatedMonitorSerial(clock)
        port.buffer.extend(monitor_frame())
        worker = app.EqualIntervalWorker(port, (), {}, settle_s=.002)
        self.assertEqual(worker._read_available_frames(), (monitor_frame(),))
        self.assertEqual(port.read_sizes, [20])
        self.assertEqual(clock.now, 0)

    def test_idle_reader_yields_without_a_blocking_serial_read(self):
        clock = VirtualClock()
        port = SimulatedMonitorSerial(clock)
        worker = app.EqualIntervalWorker(port, (), {}, settle_s=.002)
        with patch.object(app.time, 'sleep', clock.sleep):
            self.assertEqual(worker._read_available_frames(), ())
        self.assertEqual(port.read_sizes, [])
        self.assertAlmostEqual(clock.now, .001)

    def test_fragmented_frame_is_retained_until_complete(self):
        clock = VirtualClock()
        port = SimulatedMonitorSerial(clock)
        worker = app.EqualIntervalWorker(port, (), {}, settle_s=.002)
        payload = monitor_frame()
        port.buffer.extend(payload[:7])
        self.assertEqual(worker._read_available_frames(), ())
        port.buffer.extend(payload[7:])
        self.assertEqual(worker._read_available_frames(), (payload,))
        self.assertEqual(port.read_sizes, [7, 13])
        self.assertEqual(clock.now, 0)

    def test_lan_adapter_reads_the_same_native_frame(self):
        port = FbgLanSerial()
        try:
            port._receive_buffer.extend(monitor_frame())
            worker = app.EqualIntervalWorker(port, (), {}, settle_s=.002)
            self.assertEqual(worker._read_available_frames(), (monitor_frame(),))
        finally:
            port.close()

    def test_timing_model_speeds_up_without_reusing_old_adc_samples(self):
        result = compare_receive_paths(100)
        self.assertLess(result['after']['mean_point_ms'], 25)
        self.assertLess(result['after']['total_s'], result['before']['total_s'] / 3)
        self.assertTrue(result['raw_values_identical'])
        for name in ('before', 'after'):
            self.assertEqual(result[name]['fresh_frame_flushes'], 100)


if __name__ == '__main__':
    unittest.main()
