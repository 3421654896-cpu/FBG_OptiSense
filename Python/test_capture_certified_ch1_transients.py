import unittest
from pathlib import Path
from unittest.mock import patch
from capture_certified_ch1_transients import command, decode, OFFSETS_US, capture_stable_tail
import app_JDSU as app


def fixture():
    b = bytearray(b"\xd5\x5d\x02\x03\x00\x01\x00\x02")
    b.extend((5300).to_bytes(4,"big"))
    b.extend(bytes((33,6)))
    b.extend((9).to_bytes(4,"big"))
    b.extend(b"\x00\x7b"*6)
    for offset in OFFSETS_US:
        b.extend((5305+offset).to_bytes(4,"big"))
        b.extend(b"\x00\x64"*6)
    b.extend(b"\x5d\xd5")
    return bytes(b)


class TransientProtocolTests(unittest.TestCase):
    def test_stable_teacher_holds_target_without_dac_rewrite(self):
        class Port:
            def reset_input_buffer(self): pass
            def write(self, command): raise AssertionError('teacher must not rewrite DAC')
        class Worker:
            port = Port()
            running = True
            _rx_buffer = bytearray()
            def _interruptible_wait(self, duration):
                self.wait = duration
                return True
            def _read_available_frames(self):
                frame = b'\xff\xff\x02\x02' + b'\x00\x64' * 6 + b'\x00' * 4
                return [frame] * app.FULLBAND_ACCURACY_MIN_SAMPLES
        worker = Worker()
        with patch('capture_certified_ch1_transients.time.sleep'):
            result = capture_stable_tail(worker)
        self.assertEqual(worker.wait, .2)
        self.assertTrue(result['stable'])
        self.assertFalse(result['dac_rewritten'])
        self.assertEqual(result['ch1_adc_code'], 100)
        with self.assertRaises(ValueError): capture_stable_tail(worker, .1)

    def test_explicit_flag_and_request_tag(self):
        b = command(1,2,9)
        self.assertEqual(len(b),808)
        self.assertEqual(b[8],3)
        self.assertEqual(int.from_bytes(b[9:13],"big"),9)
        for s,t,n in [(-1,2,9),(1,45,9),(1,2,0)]:
            with self.assertRaises(ValueError): command(s,t,n)

    def test_valid_timestamped_direct_adc(self):
        result = decode(fixture(),1,2,9)
        self.assertEqual(result["path_update_us"],5300)
        self.assertEqual(result["sample_start_after_target_write_us"][-1],100005)
        self.assertEqual(len(result["direct_adc"]),33)

    def test_stale_truncated_invalid_rejected(self):
        for packet,s,t,n in [(fixture(),1,2,8),(fixture(),0,2,9),
                            (fixture()[:-1],1,2,9),
                            (fixture().replace(b"\x00\x64",b"\xff\xff"),1,2,9)]:
            with self.assertRaises(ValueError): decode(packet,s,t,n)

    def test_firmware_preserves_shared_clock_and_normal_path(self):
        source = (Path(__file__).resolve().parents[1]/"JDSU/Core/Src/ms5614t.c").read_text("utf-8")
        method = source.split("void runLaserSwitchTest(void)",1)[1].split("static void sendStressRawCapture",1)[0]
        self.assertNotIn("DWT->CYCCNT =",method)
        self.assertIn("PI11210_ApplyStressRowCalibrationOrder()",method)
        self.assertIn("Stress_Path_Precondition_DAC",method)
        self.assertIn("switchTestWriteU32(&position, diagnosticTag)",method)
        self.assertIn("sourceIndex >= STRESS_TABLE_POINT_COUNT",method)


if __name__ == "__main__": unittest.main()
