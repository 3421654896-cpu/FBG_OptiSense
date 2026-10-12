import binascii
from copy import deepcopy
import unittest
from unittest.mock import Mock, patch
from pathlib import Path
from contextlib import ExitStack
import capture_candidate27_stream as c
from capture_fast_flank_stream import FIRMWARE as ACTIVE_FIRMWARE


class FirmwareIdentityTest(unittest.TestCase):
    def test_exact_shared_firmware(self):
        self.assertEqual(c.FIRMWARE, ACTIVE_FIRMWARE)


def rows():
    return [dict(index=i*10, codes=[50000,45000,8000,2000,3000]) for i in range(27)]


def frame(first_delay=600, spacing=50):
    wire=bytearray(291)
    wire[:4]=b'\xd9\x9d\x0a\x01'
    for offset,value in ((4,123),(8,c.FIRMWARE),(12,16),(16,1000000),(20,30000),(24,456)):
        wire[offset:offset+4]=value.to_bytes(4,'big')
    wire[28:34]=bytes((first_delay//25,27,16,spacing//25,0,2))
    wire[40:67]=bytes(range(27))
    for i in range(27):
        for offset,value in zip((0,2,4,6),(20+i,30+i,i*1000+100,i*1000+800)):
            at=67+i*8+offset
            wire[at:at+2]=value.to_bytes(2,'big')
    return seal(wire)


def seal(wire):
    wire=bytearray(wire)
    wire[-6:-2]=binascii.crc32(wire[:-6]).to_bytes(4,'big')
    wire[-2:]=b'\x9d\xd9'
    return bytes(wire)


class Candidate27Tests(unittest.TestCase):
    def test_bounded_command_and_both_crcs(self):
        data=c.command(rows(),256,123)
        self.assertEqual(len(data),808)
        self.assertEqual(data[:6],b'\xff\xff\x03\x0a\x01\x1b')
        self.assertEqual(data[12:16],b'C27!')
        self.assertFalse(any(data[352:]))
        self.assertEqual(int.from_bytes(data[348:352],'big'),binascii.crc32(data[:348]))
        self.assertEqual(int.from_bytes(data[24:26],'big'),0)
        self.assertEqual(int.from_bytes(data[336:338],'big'),260)

    def test_each_current_limit_and_order_rejected(self):
        for channel,limit in enumerate(c.LIMITS):
            r=rows(); r[3]['codes'][channel]=limit+1
            with self.assertRaises(ValueError): c.command(r,32,1)
        for index in (-1,2001,0):
            r=rows(); r[3]['index']=index
            with self.assertRaises(ValueError): c.command(r,32,1)

    def test_duration_and_timing_bounds(self):
        for cycles in (31,1025,True):
            with self.assertRaises(ValueError): c.command(rows(),cycles,1)
        for timing in ((25,600),(600,600),(51,50),(float('nan'),50)):
            with self.assertRaises(ValueError): c.command(rows(),32,1,*timing)

    def test_exact_decode_preserves_fresh_samples(self):
        data=c.decode(frame(),123,456,600,50)
        self.assertEqual(data['sequence'],16)
        self.assertEqual(len(data['records']),27)
        self.assertEqual(data['records'][26]['second_code'],56)
        self.assertTrue(data['minimum_warmup_complete'])

    def test_corrupted_wire_and_wrong_provenance_fail(self):
        wire=bytearray(frame()); wire[100]^=1
        with self.assertRaises(ValueError): c.decode(wire,123,456,600,50)
        for tag,crc,first,spacing in ((124,456,600,50),(123,457,600,50),(123,456,50,600)):
            with self.assertRaises(ValueError): c.decode(frame(),tag,crc,first,spacing)
        wire=bytearray(frame()); wire[8:12]=(c.FIRMWARE-1).to_bytes(4,'big')
        with self.assertRaises(ValueError): c.decode(seal(wire),123,456,600,50)

    def test_short_sample_timing_or_adc_overrange_fail(self):
        for at,value in ((67,4096),(73,701)):
            wire=bytearray(frame()); wire[at:at+2]=value.to_bytes(2,'big')
            with self.assertRaises(ValueError): c.decode(seal(wire),123,456,600,50)

    def test_partial_frames_and_legacy_are_not_candidate_data(self):
        wire=frame()
        buffer=bytearray(b'noise'+wire[:100])
        self.assertEqual(list(c.extract_frames(buffer)),[])
        buffer.extend(wire[100:])
        self.assertEqual(list(c.extract_frames(buffer)),[wire])
        self.assertEqual(buffer,bytearray())
        legacy=bytearray(wire); legacy[2]=3
        with self.assertRaises(ValueError): c.decode(seal(legacy),123,456,600,50)

    def test_terminal_requires_shutter_and_identity(self):
        wire=bytearray(42)
        wire[:4]=b'\xd9\x9d\x0b\x00'
        for at,value in ((4,123),(8,c.FIRMWARE),(32,456)):
            wire[at:at+4]=value.to_bytes(4,'big')
        wire[12:16]=b'\x00\x20\x00\x20'
        wire[18:22]=bytes((1,1,0,2))
        decoded=c.decode(seal(wire),123,456,600,50)
        self.assertEqual(decoded['completed_cycles'],32)
        self.assertTrue(decoded['firmware_shutter_asserted'])
        wire[19]=0
        with self.assertRaises(ValueError): c.decode(seal(wire),123,456,600,50)

    def test_live_malformed_cases_request_only_zero_current(self):
        from verify_candidate27_rejection import malformed
        for name in ('bad_crc','duplicate_index','nonzero_tail','too_many_cycles','bad_timing'):
            wire=malformed(name,123)
            self.assertEqual(len(wire),808)
            self.assertTrue(all(wire[26+i*12:36+i*12]==bytes(10) for i in range(27)))

    def test_rejection_requires_matching_tag_and_trailer(self):
        wire=bytearray(20)
        wire[:4]=b'\xd9\x9d\x0a\x7f'
        wire[4:8]=(123).to_bytes(4,'big')
        wire[-2:]=b'\x9d\xd9'
        with self.assertRaises(RuntimeError): c.decode(wire,123,456,600,50)
        with self.assertRaises(ValueError): c.decode(wire,124,456,600,50)
        wire[-1]=0
        with self.assertRaises(ValueError): c.decode(wire,123,456,600,50)

    def run_device_fixture(self, callback=None, stop=None):
        packet=c.command(rows(),32,123)
        crc=binascii.crc32(packet[24:348])
        prepared=dict(command_hex=packet.hex(),route_rows=rows(),cycles_requested=32,
                      first_delay_us=600,spacing_us=50,frames=[],complete=False,
                      route_crc32=crc,expected_firmware=c.FIRMWARE)
        wires=[]
        for i in range(32):
            wire=bytearray(frame()); wire[3]=int(i>=16)
            for at,value in ((12,i),(16,i*31000),(24,crc)):
                wire[at:at+4]=value.to_bytes(4,'big')
            wires.append(seal(wire))
        end=bytearray(42); end[:4]=b'\xd9\x9d\x0b\x00'
        for at,value in ((4,123),(8,c.FIRMWARE),(32,crc)):
            end[at:at+4]=value.to_bytes(4,'big')
        end[12:16]=b'\x00\x20\x00\x20'; end[18:22]=bytes((1,1,0,2))
        wires.append(seal(end))
        device=Mock(); device.read.return_value=b''.join(wires)
        saved=[]
        with ExitStack() as stack:
            stack.enter_context(patch.object(c,'_safe_disarm',return_value=True))
            stack.enter_context(patch.object(c,'_close_shutter',return_value=True))
            worker=stack.enter_context(patch.object(c.app,'EqualIntervalWorker'))
            worker.return_value._feedback_command_and_ack.return_value=True
            stack.enter_context(patch.object(c,'save_checkpoint',side_effect=lambda p,r:saved.append(deepcopy(r))))
            stack.enter_context(patch.object(c,'print'))
            try:
                result=c.execute_device(device,Path('__unused_candidate_fixture__.json'),
                                        prepared=prepared,on_frame=callback,should_stop=stop)
            except RuntimeError:
                result=None
        self.assertTrue(saved[-1]['disarm_ack'])
        self.assertTrue(saved[-1]['shutter_ack'])
        self.assertEqual(prepared['frames'],[])
        return result,saved[-1]

    def test_callbacks_receive_only_checked_fresh_frames(self):
        observed=[]
        result,saved=self.run_device_fixture(callback=lambda r:observed.append(r['sequence']))
        self.assertTrue(result['complete'])
        self.assertEqual(observed,list(range(32)))

    def test_callback_failure_and_external_stop_both_cleanup(self):
        def fail(_): raise RuntimeError('injected callback failure')
        result,saved=self.run_device_fixture(callback=fail)
        self.assertIsNone(result)
        self.assertFalse(saved['complete'])
        self.assertEqual(len(saved['frames']),1)
        result,saved=self.run_device_fixture(stop=lambda:True)
        self.assertIsNone(result)
        self.assertFalse(saved['complete'])
        self.assertEqual(len(saved['frames']),0)


if __name__=='__main__':
    unittest.main()
