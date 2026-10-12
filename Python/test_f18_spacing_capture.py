import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch
import capture_fast_flank_stream as stream
from f18_spacing_protocol import FIRMWARE
from test_capture_fast_flank_stream import data_frame, end_frame, _finish_crc


def reply(raw, *, data=False, spacing=50):
    raw=bytearray(raw); raw[8:12]=FIRMWARE.to_bytes(4,'big')
    if data: raw[31]=spacing//25
    return _finish_crc(raw)


class CaptureTests(unittest.TestCase):
    def test_complete_explicit_receive(self):
        wire=b''.join(reply(data_frame(i,start_us=i*20000),data=True) for i in range(32))
        device=Mock(); device.read.return_value=wire+reply(end_frame())
        frames,terminal=stream.receive(device,cycles=32,tag=123,table_crc32=456,spacing_us=50)
        self.assertEqual(len(frames),32)
        self.assertTrue(terminal['firmware_shutter_asserted'])

    def test_wrong_spacing_is_rejected_before_callback(self):
        device=Mock(); device.read.return_value=reply(data_frame(),data=True,spacing=600)
        callback=Mock()
        with self.assertRaises(ValueError):
            stream.receive(device,cycles=32,tag=123,table_crc32=456,spacing_us=50,on_frame=callback)
        callback.assert_not_called()

    def test_cancel_before_read(self):
        device=Mock()
        with self.assertRaises(InterruptedError):
            stream.receive(device,cycles=32,tag=123,table_crc32=456,spacing_us=50,should_stop=lambda:True)
        device.read.assert_not_called()

    def test_receive_deadline_is_bounded(self):
        device=Mock(); device.read.return_value=b''
        with patch.object(stream.time,'monotonic',side_effect=[0,0,70]):
            with self.assertRaises(TimeoutError):
                stream.receive(device,cycles=32,tag=123,table_crc32=456,spacing_us=50)
        device.read.assert_called_once()

    def test_all_receive_failures_attempt_shutdown(self):
        for error in (TimeoutError('test timeout'), InterruptedError('test cancellation'), ValueError('test mismatch')):
            with self.subTest(error=type(error).__name__), ExitStack() as stack:
                def mock(name, **kwargs): return stack.enter_context(patch.object(stream,name,**kwargs))
                mock('load_installed_rows',return_value=([],None)); mock('codes_crc',return_value=456)
                mock('sha256_file',return_value='test'); save=mock('save_checkpoint')
                disarm=mock('_safe_disarm',return_value=True); shutter=mock('_close_shutter',return_value=True)
                mock('receive',side_effect=error)
                worker=stack.enter_context(patch.object(stream.app,'EqualIntervalWorker')).return_value
                worker._feedback_command_and_ack.return_value=True
                worker._interruptible_wait.return_value=True
                device=Mock()
                with self.assertRaises(type(error)):
                    stream.execute_device(device,Path('outputs/unit_test_no_file.json'),cycles=32,spacing_us=50)
                self.assertEqual(disarm.call_count,2); self.assertEqual(shutter.call_count,2)
                packet=device.write.call_args.args[0]
                self.assertEqual(packet[19:22],b'S\x00\x32')
                report=save.call_args.args[1]
                self.assertFalse(report['complete'])
                self.assertTrue(report['safety_cleanup']['exact_soa_shutter_ack'])

    def test_invalid_spacing_does_not_open_port(self):
        with patch.object(stream.serial,'Serial') as port:
            with self.assertRaises(ValueError): stream.execute(Path('unused'),spacing_us=51)
            port.assert_not_called()


if __name__=='__main__': unittest.main()
