import binascii
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import capture_candidate27_guarded_stream as c
from capture_fast_flank_stream import FIRMWARE as ACTIVE_FIRMWARE


class FirmwareIdentityTest(unittest.TestCase):
    def test_exact_shared_firmware(self):
        self.assertEqual(c.FIRMWARE, ACTIVE_FIRMWARE)


def rows():
    return [dict(index=local * 10, codes=[50000, 45000, 8000, 2000, 3000])
            for local in range(27)]


def guard():
    return dict(index=205, codes=[50000, 45000, 8000, 2000, 3000],
                before_order=21, hold_us=1000)


def seal(wire):
    wire = bytearray(wire)
    wire[-6:-2] = binascii.crc32(wire[:-6]).to_bytes(4, 'big')
    wire[-2:] = b'\x9d\xd9'
    return bytes(wire)


def frame(sequence=16, route_crc=456, first_delay=400, spacing=50):
    wire = bytearray(297)
    wire[:4] = b'\xd9\x9d\x0c' + bytes((int(sequence >= 16),))
    for offset, value in ((4, 123), (8, c.FIRMWARE), (12, sequence),
                          (16, sequence * 27000), (20, 26000), (24, route_crc)):
        wire[offset:offset + 4] = value.to_bytes(4, 'big')
    wire[28:34] = bytes((first_delay // 25, 27, 16, spacing // 25, 0, 2))
    wire[40:67] = bytes(range(27))
    for local in range(27):
        write = 100 + local * 900 + (1000 if local >= 21 else 0)
        end = write + 600
        for offset, value in zip((0, 2, 4, 6),
                                 (20 + local, 30 + local, write, end)):
            at = 67 + local * 8 + offset
            wire[at:at + 2] = value.to_bytes(2, 'big')
    wire[283:285] = bytes((2, 21))
    wire[285:287] = (1000).to_bytes(2, 'big')
    wire[287:289] = (205).to_bytes(2, 'big')
    wire[289:291] = (19000).to_bytes(2, 'big')
    return seal(wire)


def terminal(route_crc=456, cycles=32):
    wire = bytearray(48)
    wire[:4] = b'\xd9\x9d\x0d\x00'
    for offset, value in ((4, 123), (8, c.FIRMWARE), (28, cycles * 27000),
                          (32, route_crc)):
        wire[offset:offset + 4] = value.to_bytes(4, 'big')
    wire[12:16] = cycles.to_bytes(2, 'big') * 2
    wire[18:22] = bytes((1, 1, 0, 2))
    wire[36:38] = bytes((2, 21))
    wire[38:40] = (1000).to_bytes(2, 'big')
    wire[40:42] = (205).to_bytes(2, 'big')
    return seal(wire)


class GuardedCandidate27Tests(unittest.TestCase):
    def test_v2_packet_layout_crcs_and_zero_tail(self):
        packet = c.command(rows(), guard(), 256, 123)
        self.assertEqual(len(packet), 808)
        self.assertEqual(packet[:6], b'\xff\xff\x03\x0a\x02\x1b')
        self.assertEqual(packet[22:24], bytes((21, 40)))
        self.assertEqual(packet[348:352], bytes(4))
        self.assertEqual(int.from_bytes(packet[352:354], 'big'), 205)
        self.assertEqual(int.from_bytes(packet[364:368], 'big'),
                         binascii.crc32(packet[:364]))
        self.assertFalse(any(packet[368:]))
        self.assertEqual(binascii.crc32(packet[24:364]),
                         binascii.crc32(packet[24:348] + bytes(4) + packet[352:364]))

    def test_guard_and_timing_bounds(self):
        for hold in (475, 5025, True):
            with self.assertRaises(ValueError):
                c.command(rows(), guard(), 32, 1, guard_hold_us=hold)
        for before, index in ((0, 5), (27, 265), (21, 200), (21, 210)):
            invalid = guard(); invalid['index'] = index
            with self.assertRaises(ValueError):
                c.command(rows(), invalid, 32, 1, guard_before=before)

    def test_exact_decode_exposes_guard_execution(self):
        decoded = c.decode(frame(), 123, 456, 400, 50, guard())
        self.assertEqual(decoded['sequence'], 16)
        self.assertEqual(decoded['guard_write_end_us'], 19000)
        self.assertEqual(len(decoded['records']), 27)
        self.assertEqual(decoded['records'][21]['write_end_us'], 20000)

    def test_missing_guard_hold_or_wrong_provenance_fails(self):
        wire = bytearray(frame()); wire[289:291] = (19550).to_bytes(2, 'big')
        with self.assertRaises(ValueError):
            c.decode(seal(wire), 123, 456, 400, 50, guard())
        wire = bytearray(frame()); wire[287:289] = (206).to_bytes(2, 'big')
        with self.assertRaises(ValueError):
            c.decode(seal(wire), 123, 456, 400, 50, guard())

    def test_terminal_requires_guard_and_shutdown_provenance(self):
        decoded = c.decode(terminal(), 123, 456, 400, 50, guard())
        self.assertTrue(decoded['firmware_shutter_asserted'])
        wire = bytearray(terminal()); wire[19] = 0
        with self.assertRaises(ValueError):
            c.decode(seal(wire), 123, 456, 400, 50, guard())

    def test_extractor_separates_new_kinds(self):
        data, end = frame(), terminal()
        buffer = bytearray(b'noise' + data + end)
        self.assertEqual(list(c.extract_frames(buffer)), [data, end])
        legacy = bytearray(data); legacy[2] = 10
        self.assertEqual(list(c.extract_frames(bytearray(legacy))), [])

    def test_device_path_checks_every_frame_and_cleans_up(self):
        packet = c.command(rows(), guard(), 32, 123)
        route_crc = binascii.crc32(packet[24:364])
        prepared = dict(command_hex=packet.hex(), route_rows=rows(), guard=guard(),
                        cycles_requested=32, first_delay_us=400, spacing_us=50,
                        frames=[], complete=False, route_crc32=route_crc,
                        expected_firmware=c.FIRMWARE)
        wires = [frame(index, route_crc) for index in range(32)] + [terminal(route_crc)]
        device = Mock(); device.read.return_value = b''.join(wires)
        saved, observed = [], []
        with ExitStack() as stack:
            stack.enter_context(patch.object(c, '_safe_disarm', return_value=True))
            stack.enter_context(patch.object(c, '_close_shutter', return_value=True))
            worker = stack.enter_context(patch.object(c.app, 'EqualIntervalWorker'))
            worker.return_value._feedback_command_and_ack.return_value = True
            stack.enter_context(patch.object(c, 'save_checkpoint',
                                             side_effect=lambda p, r: saved.append(deepcopy(r))))
            stack.enter_context(patch.object(c, 'print'))
            result = c.execute_device(device, Path('__unused_guarded_fixture__.json'),
                                      prepared=prepared,
                                      on_frame=lambda row: observed.append(row['sequence']))
        self.assertTrue(result['complete'])
        self.assertEqual(observed, list(range(32)))
        self.assertTrue(saved[-1]['disarm_ack'] and saved[-1]['shutter_ack'])


if __name__ == '__main__':
    unittest.main()
