import unittest
from pathlib import Path

from capture_production_pair_teacher import command,decode,LENGTH,selected_points,validate_capture_profile
from frame_telemetry import sentinel_offsets_for_version


def fixture(version=5,recheck=False):
    b=bytearray(LENGTH); b[:4]=b'\xd8\x8d\x01\x00'
    b[4:8]=(99).to_bytes(4,'big'); b[8:12]=(0x1001e).to_bytes(4,'big')
    b[12:14]=(38 if version==5 else 36).to_bytes(2,'big')
    b[14:20]=bytes((version,1,255,255,9,40))
    b[20:24]=(100).to_bytes(4,'big'); b[24:28]=(12000).to_bytes(4,'big')
    b[32:38]=bytes((2,2,7 if recheck else 3,25,6,1))
    for g,o in enumerate(sentinel_offsets_for_version(version)):
        p=g*5+o; b[38+p//8]|=1<<(p%8)
    b[48:52]=bytes((0,59,2,1))
    for i,(start,end,value) in enumerate(((40,60,40),(680,700,53),(1350,1370,55))):
        if i==2 and not recheck: continue
        at=52+i*10; b[at:at+4]=start.to_bytes(4,'big'); b[at+4:at+8]=end.to_bytes(4,'big')
        b[at+8:at+10]=value.to_bytes(2,'big')
    for i in range(26):
        start=(1400 if recheck else 740) if i==0 else 300000+(i-1)*20000
        at=82+i*20; b[at:at+4]=start.to_bytes(4,'big'); b[at+4:at+8]=(start+55).to_bytes(4,'big')
        for c,v in enumerate((0,51,2,3,400,300)):
            b[at+8+c*2:at+10+c*2]=v.to_bytes(2,'big')
    b[-2:]=b'\x8d\xd8'
    return bytes(b)


def reference_fixture(point=38,version=4):
    b=bytearray(fixture(version))
    b[8:12]=(0x1001f).to_bytes(4,'big')
    b[12:14]=point.to_bytes(2,'big')
    b[15:20]=bytes((0,255,255,45,128))
    b[38:44]=b'\xff'*5+b'\x1f'
    return bytes(b)


class PairTeacherTests(unittest.TestCase):
    def test_full_reference_is_explicit_and_all_five_rows_are_selected(self):
        for g in range(1,10):
            for version in (4,5):
                self.assertEqual(len(selected_points(g,version)),1)
                self.assertEqual(selected_points(g,version,full_map_reference=True),
                                 list(range((g-1)*5,g*5)))
        for point in range(45):
            r=decode(reference_fixture(point),point,99,2,4)
            validate_capture_profile(r,full_map_reference=True)
            self.assertEqual(bytes.fromhex(r['wire_packet_hex']),reference_fixture(point))
            with self.assertRaises(ValueError):validate_capture_profile(r)
        with self.assertRaises(ValueError):
            validate_capture_profile(decode(fixture(),38,99,2,5),full_map_reference=True)

    def test_new_opcode_is_bounded_and_tagged(self):
        b=command(38,99); self.assertEqual(len(b),808)
        self.assertEqual(b[:4],b'\xff\xff\x03\x04')
        self.assertEqual(int.from_bytes(b[4:6],'big'),38)
        for point,tag in [(-1,1),(45,1),(38,0),(38,2**32)]:
            with self.assertRaises(ValueError): command(point,tag)

    def test_same_trial_teacher_and_causal_feature_availability(self):
        for version in (4,5):
            for recheck in (False,True):
                p=38 if version==5 else 36
                r=decode(fixture(version,recheck),p,99,2,version)
                self.assertEqual(len(r['production_reads']),3 if recheck else 2)
                self.assertTrue(r['same_trial']); self.assertFalse(r['teacher']['dac_rewritten'])
                self.assertTrue(r['teacher']['stable']); self.assertIsNone(r['pressure_ground_truth'])
                self.assertEqual(r['teacher']['median'],[400,300,0,51,2,3])
                self.assertLess(r['pair_available_us'],r['optical_state']['start_us'])
                self.assertLess(r['extended_features_available_us'],r['teacher_samples'][0]['start_us'])
                self.assertEqual(r['firmware_rc_estimate'],59)

    def test_stale_abort_version_feedback_provenance_and_values_rejected(self):
        for at,value in [(0,0),(3,1),(7,98),(11,29),(13,36),(14,4),
                         (15,9),(16,1),(18,8),(19,128),(32,1),(33,1),
                         (34,1),(35,24),(36,5),(37,0),(43,255),(48,255),
                         (50,3),(51,0),(60,255),(72,1),(90,255)]:
            b=bytearray(fixture()); b[at]=value
            with self.subTest(at=at),self.assertRaises(ValueError): decode(bytes(b),38,99,2,5)
        with self.assertRaises(ValueError): decode(fixture()[:-1],38,99,2,5)

    def test_timestamps_must_be_causal_and_teachers_not_late_bursts(self):
        for at,value in [(62,59),(82,699),(82+20,200000),(82+20,320000),(82+40,300000)]:
            b=bytearray(fixture()); b[at:at+4]=value.to_bytes(4,'big')
            with self.assertRaises(ValueError): decode(bytes(b),38,99,2,5)

    def test_unstable_teacher_is_preserved_but_not_qualified(self):
        b=bytearray(fixture())
        for i in range(25):
            at=102+i*20+10; b[at:at+2]=(51+i*4).to_bytes(2,'big')
        self.assertFalse(decode(bytes(b),38,99,2,5)['teacher']['stable'])

    def test_firmware_calls_production_reader_before_teacher_and_stops(self):
        src=(Path(__file__).resolve().parents[1]/'JDSU/Core/Src/ms5614t.c').read_text('utf-8')
        loop=src.split('void write_ms5614t_table(',1)[1].split('void ',1)[0]
        at=loop.index('sampleVoltageStable(i - 1)')
        self.assertIn('runStressProductionTeacher(&frameTimer);',loop[at:at+650])
        method=src.split('static void runStressProductionTeacher(',1)[1].split('static void sendStressRawCapture(',1)[0]
        self.assertNotIn('PI11210_Apply',method)
        self.assertIn('PI11210_SetSOAShutter(1U)',method)
        self.assertIn('ApplyWorkState(EXTRA_STATE)',method)
        self.assertIn('stressPairValidMask == 3U || stressPairValidMask == 7U',method)
        reader=src.split('void sampleVoltageStable(',1)[1].split('void sampleVoltagePrecision(',1)[0]
        self.assertNotIn('ADC_ReadSix(',reader)
        self.assertEqual(reader.count('ADC_ReadMask(sampleMask, first)'),1)
        self.assertEqual(reader.count('ADC_ReadMask(sampleMask, second)'),1)
        self.assertIn('if(pairedProbe && adc_idx == 1U)',reader)
        self.assertIn('stressPairEstimateValid = 1U;',reader)
        self.assertIn('cyclesPerUs && stressPairEstimateValid',method)


if __name__ == '__main__': unittest.main()
