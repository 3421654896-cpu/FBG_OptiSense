import unittest
from pathlib import Path

from capture_sparse_held_trajectory import command, decode, LENGTH, OFFSETS_US
from frame_telemetry import MULTIRATE_SENTINEL_LOCAL_OFFSETS


def fixture():
    b = bytearray(LENGTH)
    b[:4] = b'\xd7\x7d\x01\x00'
    b[4:8] = (99).to_bytes(4,'big')
    b[8:12] = (0x1001c).to_bytes(4,'big')
    b[12:14] = (36).to_bytes(2,'big')
    b[14:20] = bytes((4,1,255,255,9,50))
    b[20:24] = (105).to_bytes(4,'big')
    b[24:28] = (100000).to_bytes(4,'big')
    b[28:32] = (24000).to_bytes(4,'big')
    b[32:38] = bytes((2,2,33,25,6,1))
    for g,offset in enumerate(MULTIRATE_SENTINEL_LOCAL_OFFSETS):
        p = g*5+offset
        b[38+p//8] |= 1 << (p%8)
    end = 0
    for i,offset in enumerate(list(OFFSETS_US)+[300000+i*20000 for i in range(25)]):
        start = max(offset,end)+1
        end = start+55
        at = 44+i*20
        b[at:at+4] = start.to_bytes(4,'big')
        b[at+4:at+8] = end.to_bytes(4,'big')
        for c,value in enumerate((10,30,2,4,100,200)):
            b[at+8+c*2:at+10+c*2] = value.to_bytes(2,'big')
    b[-2:] = b'\x7d\xd7'
    return bytes(b)


class HeldTrajectoryTests(unittest.TestCase):
    def test_command_is_tagged_and_bounded(self):
        p = command(36,99)
        self.assertEqual(len(p),808)
        self.assertEqual(p[:4],b'\xff\xff\x03\x03')
        for point,tag in [(-1,1),(45,1),(36,0),(36,2**32)]:
            with self.assertRaises(ValueError): command(point,tag)

    def test_prefix_and_teacher_provenance(self):
        r = decode(fixture(),36,99,2)
        self.assertEqual(r['prefix_profile'],1)
        self.assertEqual(len(r['curve']),33)
        self.assertEqual(len(r['teacher_samples']),25)
        self.assertEqual(r['teacher']['median'],[100,200,10,30,2,4])
        self.assertTrue(r['teacher']['stable'])
        self.assertFalse(r['teacher']['dac_rewritten'])
        self.assertTrue(r['diagnostic_interrupts_scan'])
        self.assertTrue(r['target_would_normally_be_sampled'])

    def test_abort_stale_malformed_unknown_and_feedback_rejected(self):
        for position,value in [(0,0),(3,1),(7,98),(11,27),(13,35),
                               (14,3),(15,9),(16,1),(18,8),(33,1),
                               (34,32),(35,24),(37,0),(43,255),
                               (44+8,255),(44+4,255)]:
            b = bytearray(fixture()); b[position] = value
            with self.subTest(position=position), self.assertRaises(ValueError):
                decode(bytes(b),36,99,2)
        with self.assertRaises(ValueError): decode(fixture()[:-1],36,99,2)

    def test_early_teacher_and_overlapping_samples_rejected(self):
        for index,value in [(33,200000),(1,1)]:
            b = bytearray(fixture())
            b[44+index*20:48+index*20] = value.to_bytes(4,'big')
            with self.assertRaises(ValueError): decode(bytes(b),36,99,2)

    def test_drifting_teacher_not_accepted_as_stable(self):
        b = bytearray(fixture())
        for i in range(25):
            at = 44+(33+i)*20+10
            b[at:at+2] = (30+i*4).to_bytes(2,'big')
        self.assertFalse(decode(bytes(b),36,99,2)['teacher']['stable'])

    def test_firmware_has_bounded_hold_and_fail_closed_exit(self):
        src = (Path(__file__).resolve().parents[1]/'JDSU/Core/Src/ms5614t.c').read_text('utf-8')
        method = src.split('static void runStressHeldTrajectory(',1)[1].split('static void sendStressRawCapture(',1)[0]
        self.assertIn('PI11210_SetSOAShutter(1U)',method)
        self.assertIn('ApplyWorkState(EXTRA_STATE)',method)
        self.assertIn('300000U + (sample - SWITCH_TEST_SAMPLE_COUNT) * 20000U',method)
        self.assertNotIn('PI11210_Apply',method)
        self.assertNotIn('ForceMap',method)
        self.assertNotIn('CYCCNT =',method)
        self.assertIn('ADC_LastTransferOk()',method)
        wait = src.split('static uint8_t stressHoldWait(',1)[1].split('\n}\n',1)[0]
        self.assertIn('ReceEndFlag || !fastStreamControlActive()',wait)
        active = src.split('static uint8_t fastStreamControlActive(void)',1)[1].split('\n}\n',1)[0]
        self.assertIn('usbCdcHostOpen || WifiTransport_IsLanRawClientActive()',active)
        self.assertIn('PI11210_GetStatus().soaMode != PI11210_SOA_SOURCE',wait)
        self.assertIn('ThermalControl_SafetyTick()',wait)
        begin = src.split('static void stressBeginFrame(void)',1)[1].split('static void stressAppendCachedCh1',1)[0]
        self.assertIn('stressHoldActiveTag = stressHoldRequestedTag;',begin)
        self.assertIn('stressHoldRequestedTag = 0U;',begin)


if __name__ == '__main__': unittest.main()
