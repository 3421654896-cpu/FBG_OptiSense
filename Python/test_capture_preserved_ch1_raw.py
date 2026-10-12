import unittest
from pathlib import Path

from capture_preserved_ch1_raw import command, decode, LENGTH
from frame_telemetry import MULTIRATE_SENTINEL_LOCAL_OFFSETS


def fixture():
    b = bytearray(LENGTH)
    b[:6] = b'\xd6\x6d\x02\x00\x2d\x02'
    b[6:10] = (11).to_bytes(4,'big')
    b[10:14] = (150).to_bytes(4,'big')
    b[14:18] = (123000).to_bytes(4,'big')
    b[18:22] = (30000).to_bytes(4,'big')
    b[22:28] = bytes((4,1,255,255,9,20))
    b[34:36] = bytes((2,2))
    b[36:40] = (0x1001b).to_bytes(4,'big')
    fresh = {g*5+offset for g,offset in enumerate(MULTIRATE_SENTINEL_LOCAL_OFFSETS)}
    for point in range(45):
        at = 40+point*13
        if point in fresh:
            b[28+point//8] |= 1 << (point%8)
            b[at:at+8] = b'\x00\x0a\x00\x0c\x00\x00\x00\x0d'
            b[at+9:at+13] = ((point+1)*600).to_bytes(4,'big')
        else:
            b[at+9:at+13] = b'\xff'*4
    b[-2:] = b'\x6d\xd6'
    return bytes(b)


class PreservedRawTests(unittest.TestCase):
    def test_explicit_opt_in_tag_and_fixed_command(self):
        packet = command(11)
        self.assertEqual(len(packet),808)
        self.assertEqual(packet[:5],b'\xff\xff\x03\x02\x01')
        self.assertEqual(int.from_bytes(packet[5:9],'big'),11)
        for value in [0,-1,1<<32]:
            with self.assertRaises(ValueError): command(value)

    def test_fresh_and_cached_are_distinguished(self):
        result = decode(fixture(),11,2)
        self.assertEqual(result['profile'],1)
        self.assertEqual(len(result['fresh_indices']),9)
        self.assertEqual(result['records'][36]['second_direct_adc'],12)
        self.assertIsNone(result['records'][35]['second_direct_adc'])
        self.assertEqual(result['frame_start_ms'],123000)

    def test_corruption_stale_unknown_and_missing_timestamp_rejected(self):
        for position,value in [(2,1),(9,12),(22,3),(26,8),(33,0xff),(35,1),
                                (40,1),(40+36*13+9,0xff),(40+36*13,0xff)]:
            packet = bytearray(fixture())
            packet[position] = value
            with self.assertRaises(ValueError): decode(bytes(packet),11,2)
        with self.assertRaises(ValueError): decode(fixture()[:-1],11,2)

    def test_firmware_preserves_sparse_and_clears_aborted_capture(self):
        source = (Path(__file__).resolve().parents[1]/'JDSU/Core/Src/ms5614t.c').read_text('utf-8')
        begin = source.split('static void stressBeginFrame(void)',1)[1].split('static void stressAppendCachedCh1',1)[0]
        self.assertIn('if(stressRawCaptureActive == 1U)', begin)
        self.assertEqual(begin.count('StressMultirate_ForceMap('),1)
        self.assertIn('stressRawCaptureActive = 0U;',begin)
        self.assertIn('stressRawActiveTag = stressRawRequestedTag;',begin)
        clear = source.split('static void stressMultirateDisarmAndClear(void)',1)[1].split('static PI11210_Status_t',1)[0]
        self.assertIn('stressRawCaptureRequested = 0U;',clear)
        self.assertIn('stressRawActiveTag = 0U;',clear)


if __name__ == '__main__':
    unittest.main()
