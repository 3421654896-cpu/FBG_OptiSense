import binascii
import unittest
from unittest.mock import Mock

import capture_long_route_stability as probe
from test_firmware_scan_scheduler import _function, MS5614T


def fixture(tag=7):
    rows,_=probe.load_installed_rows();wire=bytearray(probe.packet_length());indices=probe.candidate_indices('g8_right_left')
    put=lambda at,value,size=4:wire.__setitem__(slice(at,at+size),int(value).to_bytes(size,'big'))
    wire[:4]=b'\xd9\x9d\x02\x00';put(4,tag);put(8,probe.FIRMWARE);wire[12:16]=bytes((1,32,18,0));put(16,1500,2)
    wire[18:24]=b'\x00\x02'+(576).to_bytes(2,'big')+b'\x00\x02';put(32,576,2);put(36,probe.codes_crc(rows));wire[44:62]=bytes(indices);wire[62]=1;wire[-2:]=b'\x9d\xd9'
    end=0
    for index in range(576):
        at=64+index*32;row=indices[index%18];put(at,row,2);wire[at+2:at+4]=bytes(divmod(index,18));start=end+1;write=end+101;first=write+50+(1500 if row==38 else 0);second=first+610;end=second+10
        for offset,value in ((4,start),(8,write),(12,first),(16,first+10),(22,second),(26,end)):put(at+offset,value)
        put(at+20,40+(row%5),2);put(at+30,50+(row%5),2)
    put(24,end+1);put(28,write);put(len(wire)-6,binascii.crc32(wire[:-6]));return bytes(wire)


class LongRouteTests(unittest.TestCase):
    def test_command_is_fixed_bounded_and_has_no_currents(self):
        command=probe.command('g8_right_left',1500,9)
        self.assertEqual(len(command),808);self.assertEqual(command[:4],b'\xff\xff\x03\x06');self.assertEqual(command[12:16],b'R32!');self.assertEqual(set(command[16:]),{0})
        for bad in ((True,1500,1),('bad',1500,1),('g8_right_left',2,1),('g8_right_left',1500,0)):
            with self.assertRaises(ValueError):probe.command(*bad)

    def test_decode_and_analysis_recheck_raw_structure(self):
        rows,_=probe.load_installed_rows();decoded=probe.decode(fixture(),'g8_right_left',1500,7,rows)
        self.assertEqual(len(decoded['points']),576);self.assertEqual(decoded['points'][-1]['cycle'],31)
        analysis=probe.analyze(decoded,rows);self.assertEqual(set(analysis['series']),{f'G{i}' for i in range(1,10)});self.assertFalse(analysis['deployment_eligible'])
        broken=bytearray(fixture());broken[100]^=1
        with self.assertRaises(ValueError):probe.decode(bytes(broken),'g8_right_left',1500,7,rows)

    def test_capture_recognizes_explicit_rejection(self):
        frame=b'\xd9\x9d\x00\x7f'+(11).to_bytes(4,'big')+(256).to_bytes(2,'big')+bytes(8)+b'\x9d\xd9';port=Mock();port.read.return_value=frame
        with self.assertRaises(probe.RouteRejected) as caught:probe.capture_packet(port,'g8_right_left',1500,11)
        self.assertEqual(caught.exception.reason_mask,256)

    def test_firmware_stream_is_diagnostic_only_and_shutters(self):
        source=MS5614T.read_text('utf-8');method=_function(source,'runShortRouteDiagnostic');dispatcher=_function(source,'write_ms5614t_extra')
        for text in ('ROUTE_STREAM_CYCLES','streamOnly ? 0U : ROUTE_TEST_TEACHERS','streamOnly ? 2U : 1U','PI11210_SetSOAShutter(1U)','Stress_Wave_DAC[row]'):
            self.assertIn(text,method)
        for forbidden in ('HAL_FLASH','Wave_DAC[','PI11210_SetSOAShutter(0'):
            self.assertNotIn(forbidden,method.replace('Stress_Wave_DAC[','InstalledRows['))
        self.assertIn('aRxBuffer[3] == 0x05U || aRxBuffer[3] == 0x06U',dispatcher)


if __name__=='__main__':unittest.main()
