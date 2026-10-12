import binascii
import copy
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import capture_short_route_teacher as probe
from test_firmware_scan_scheduler import _function, MS5614T


def crc(wire):
    wire[-6:-2] = binascii.crc32(wire[:-6]).to_bytes(4, 'big')
    return bytes(wire)


def fixture(route='g8_right_left', target=36, extra=0, tag=1):
    rows, _ = probe.load_installed_rows()
    request = probe.command(route, target, extra, tag)
    n = 144 + request[5] + 1
    wire = bytearray(probe.packet_length(route, target))
    def put(at, val, size=4): wire[at:at+size] = int(val).to_bytes(size, 'big')
    wire[:4] = b'\xd9\x9d\x01\x00'; put(4,tag); put(8,probe.FIRMWARE)
    wire[12:16] = bytes((request[4],8,18,request[5])); put(16,extra,2)
    wire[18:24] = b'\x00\x02' + n.to_bytes(2,'big') + b'\x19\x01'
    put(32,n,2); put(36,probe.codes_crc(rows)); wire[44:62] = bytes(probe.candidate_indices(route))
    wire[62]=1; wire[-2:]=b'\x9d\xd9'
    end = 0
    for index in range(n):
        at = 64+index*32
        row = probe.candidate_indices(route)[index%18]
        put(at,row,2); wire[at+2:at+4] = bytes(divmod(index,18))
        begin = end+1; write_end=begin+100; first=write_end+50+(extra if row==38 else 0)
        second = first+10+600; end=second+10
        for offset,value in ((4,begin),(8,write_end),(12,first),(16,first+10),(22,second),(26,end)):
            put(at+offset,value)
        put(at+20,40,2); put(at+30,50,2)
    put(28,write_end); put(24,write_end+780051)
    for index in range(25):
        at=64+n*32+index*20; put(at,300000+index*20000); put(at+4,300050+index*20000)
        for c,value in enumerate((0,52,0,0,40,1620)): put(at+8+c*2,value,2)
    return crc(wire)


def report_fixture():
    rows,_=probe.load_installed_rows()
    wire=fixture()
    return dict(schema='ch1_continuous_short_route_teacher_v1', complete=True, route='g8_right_left',
        targets=[36], repeats=1, extra_us=0, installed_rows=rows,
        captures=[dict(repeat=0,target=36,tag=1,wire_hex=wire.hex(),
                       decoded=probe.decode(wire,'g8_right_left',36,0,1,rows))])


class ShortRouteTests(unittest.TestCase):
    def test_protocol_has_no_host_currents_and_requires_explicit_magic(self):
        for route in probe.ROUTES:
            for extra in probe.DELAYS:
                for target in probe.candidate_indices(route):
                    data = probe.command(route,target,extra,1)
                    self.assertEqual(len(data),808)
                    self.assertEqual(data[:4],b'\xff\xff\x03\x05')
                    self.assertEqual(data[12:16],b'R18!')
                    self.assertEqual(set(data[16:]),{0})
                    rows,_=probe.load_installed_rows()
                    decoded=probe.decode(fixture(route,target,extra),route,target,extra,1,rows)
                    self.assertEqual(decoded['points'][-1]['target'],target)
                    self.assertEqual(len(decoded['teacher_samples']),25)
                    self.assertFalse(decoded['training_eligible'])

    def test_unlisted_parameters_rejected_before_any_serial_open(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(probe.serial,'Serial') as port:
            path=Path(directory)/'out.json'
            for args in (dict(route='g4_g8_peak_right'),dict(targets=(37,)),dict(extra_us=1),
                         dict(repeats=0),dict(repeats=True),dict(targets=(36,36)),dict(targets=())):
                with self.subTest(args=args),self.assertRaises(ValueError):probe.execute(path,**args)
            path.touch()
            with self.assertRaises(FileExistsError):probe.execute(path)
            port.assert_not_called()

    def test_raw_packet_provenance_crc_timing_and_abort_fail_closed(self):
        rows,_=probe.load_installed_rows()
        for at,value in ((3,1),(7,2),(11,31),(12,0),(13,7),(14,17),(15,14),
                         (17,1),(18,1),(19,1),(20,1),(22,24),(23,0),(34,1),(36,0),
                         (40,1),(44,0),(62,0),(63,1),(64,1),(66,8),(67,1),(84,255)):
            wire=bytearray(fixture());wire[at]=value
            with self.subTest(at=at),self.assertRaises(ValueError):
                probe.decode(crc(wire),'g8_right_left',36,0,1,rows)
        wire=bytearray(fixture());wire[85]^=1
        with self.assertRaises(ValueError):probe.decode(bytes(wire),'g8_right_left',36,0,1,rows)
        wire=bytearray(fixture());wire[76:80]=(100).to_bytes(4,'big')
        with self.assertRaises(ValueError):probe.decode(crc(wire),'g8_right_left',36,0,1,rows)
        wire=bytearray(fixture());at=64+160*32;wire[at:at+4]=(200000).to_bytes(4,'big')
        with self.assertRaises(ValueError):probe.decode(crc(wire),'g8_right_left',36,0,1,rows)

    def test_analysis_redecodes_and_preserves_same_trial_teacher_not_pressure_truth(self):
        result=probe.summarize(report_fixture())
        trial=result['trials'][0]
        self.assertEqual(trial['same_trial_second'],50)
        self.assertEqual(trial['same_trial_teacher'],52)
        self.assertEqual(trial['same_trial_error_codes'],-2)
        self.assertEqual(len(trial['per_cycle_scan_span_us']),8)
        self.assertEqual(set(map(len,trial['last_four_cycle_second_codes'].values())),{4})
        self.assertFalse(result['pressure_bandwidth_verified'])
        for kind in ('missing','duplicate','derived','count'):
            report=report_fixture()
            if kind=='missing':report['captures']=[]
            elif kind=='duplicate':report['captures']*=2
            elif kind=='count':report['repeats']=True
            else:report['captures'][0]['decoded']['teacher_ch1']=100
            with self.subTest(kind=kind),self.assertRaises(ValueError):probe.summarize(report)

    def test_capture_failure_saves_partial_record_and_still_closes_laser(self):
        port=Mock();port.__enter__=Mock(return_value=port);port.__exit__=Mock(return_value=False)
        with tempfile.TemporaryDirectory() as directory, patch.object(probe.serial,'Serial',return_value=port), \
                patch.object(probe,'_safe_disarm',return_value=True) as disarm, \
                patch.object(probe,'_close_shutter',return_value=True) as shutter, \
                patch.object(probe.app,'EqualIntervalWorker',autospec=True) as worker, \
                patch.object(probe,'capture_packet',side_effect=TimeoutError('missing')):
            path=Path(directory)/'partial.json'
            with self.assertRaises(TimeoutError):probe.execute(path,targets=(36,),repeats=1)
            self.assertEqual(shutter.call_count,2);self.assertEqual(disarm.call_count,2)
            report=json.loads(path.read_text('utf-8'))
            self.assertFalse(report['complete']);self.assertEqual(len(report['captures']),1)
            self.assertTrue(report['safety_cleanup']['exact_soa_shutter_ack'])

    def test_firmware_is_explicit_continuous_bounded_and_shutters_before_usb(self):
        source=MS5614T.read_text('utf-8')
        method=_function(source,'runShortRouteDiagnostic')
        for safety in ('workState != EXTRA_STATE','!usbCdcHostOpen','memcmp(&aRxBuffer[12], "R18!", 4U)',
                       'PI11210_SOA_SHUTTER','route >= 2U','targetPosition >= ROUTE_TEST_POINTS',
                       'ThermalControl_SafetyTick();','PI11210_Process();','WifiTransport_IsOtaActive()',
                       'ROUTE_TEST_TIMEOUT_US','ADC_LastTransferOk()'):
            self.assertIn(safety,method)
        self.assertLess(method.index('ClearRxBuff();'),method.index('ReceEndFlag = 0U;'))
        self.assertIn('completed % ROUTE_TEST_POINTS',method)
        self.assertEqual(method.count('ADC_ReadMask(0x02U, samples)'),2)
        self.assertEqual(method.count('PI11210_ApplyStressRowCalibrationOrder()'),1)
        after_loop=method.split('/* No DAC or feedback rewrites',1)[1]
        self.assertNotIn('PI11210_Apply',after_loop)
        self.assertLess(after_loop.index('PI11210_SetSOAShutter(1U)'),after_loop.index('CDC_Transmit_FS'))
        for forbidden in ('CYCCNT =','Wave_DAC[','PI11210_SetSOAShutter(0','HAL_FLASH','malloc('):
            self.assertNotIn(forbidden,method.replace('Stress_Wave_DAC[','InstalledRows['))
        dispatcher=_function(source,'write_ms5614t_extra')
        self.assertIn('aRxBuffer[3] == 0x05',dispatcher)
        self.assertIn('runShortRouteDiagnostic(aRxBuffer[3] == 0x06U);',dispatcher)
        self.assertIn('aRxBuffer[3] == 0x05U || aRxBuffer[3] == 0x06U',dispatcher)
        self.assertLess(dispatcher.index('runShortRouteDiagnostic('),dispatcher.index('checkRT();'))
        self.assertLess(dispatcher.index('if(!dma_transfer_complete) return;'),
                        dispatcher.index('runShortRouteDiagnostic('))
        # A command delivered by an IRQ during either monitor callback must be
        # left for the next invocation. No mailbox clearing below either call.
        after_monitors = dispatcher[dispatcher.index('checkTemp(workState);'):]
        self.assertNotIn('ReceEndFlag =', after_monitors)
        self.assertNotIn('ClearRxBuff(', after_monitors)
        self.assertNotIn('aRxBuffer[', after_monitors)
        self.assertNotIn('runShortRouteDiagnostic',_function(source,'write_ms5614t_table'))
        for forbidden in ('Mach3','move_','install_','flash_'):
            self.assertNotIn(forbidden,inspect.getsource(probe.execute))

    def test_previous_firmware_packet_remains_redecodable_with_actual_version(self):
        rows,_=probe.load_installed_rows()
        wire=bytearray(fixture());wire[8:12]=(0x10020).to_bytes(4,'big')
        self.assertEqual(probe.decode(crc(wire),'g8_right_left',36,0,1,rows)['firmware'],0x10020)
        self.assertEqual(probe.RouteCaptureTimeout(b'raw').received_wire_hex,b'raw'.hex())

    def test_explicit_rejection_keeps_reason_and_received_bytes(self):
        frame = (b'\xd9\x9d\x00\x7f' + (71).to_bytes(4, 'big')
                 + (2048).to_bytes(2, 'big') + bytes((0,2,3,0,1,1,1,4))
                 + b'\x9d\xd9')
        port = Mock()
        port.read.side_effect = [b'other' + frame[:12], frame[12:]]
        with self.assertRaises(probe.RouteRejected) as caught:
            probe.capture_packet(port, 'g8_right_left', 11, 1500, 71)
        self.assertEqual(caught.exception.reason_mask, 2048)
        self.assertEqual(caught.exception.received_wire_hex, (b'other'+frame).hex())
        port.write.assert_called_once_with(probe.command('g8_right_left',11,1500,71))


if __name__ == '__main__':unittest.main()
