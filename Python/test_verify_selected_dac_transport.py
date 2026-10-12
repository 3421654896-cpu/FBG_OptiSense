"""DAC/transport verification safety tests; never open hardware."""
from pathlib import Path
import struct
import unittest
from unittest.mock import MagicMock, patch

import verify_selected_dac_transport_live as live

OFFSETS = dict(indices=0, codes=126, pointDelayUs=762, pointCount=888,
               tableCrc=896, protocolVersion=900, adcChannel=901,
               feedbackSelector=902, pointsPerPeak=903, fastDacWrites=904)


def fixture():
    rows=[dict(index=i*2,codes=[i+1,i+2,i+3,i+4,i+5]) for i in range(63)]
    data=bytearray(920)
    struct.pack_into('<63H',data,0,*[r['index'] for r in rows])
    struct.pack_into('<315H',data,126,*[v for r in rows for v in r['codes']])
    struct.pack_into('<63H',data,762,*([850]*63))
    for key,value in dict(pointCount=63,protocolVersion=7,adcChannel=0,
                          feedbackSelector=2,pointsPerPeak=7,fastDacWrites=0).items():
        data[OFFSETS[key]]=value
    struct.pack_into('<I',data,896,123)
    return rows,bytes(data)


class DacAckTests(unittest.TestCase):
    def test_saved_145ma_profile_is_exact_not_rounded_or_truncated_to_135ma(self):
        expected=(63351,63350,32767,24575,12345)
        packet=live.build_selected_dac_packet(expected)
        self.assertEqual(struct.unpack('>5H',packet[4:14]),expected)
        self.assertEqual(len(packet),808)
        for invalid in ((63352,1,2,3,4),(1,2,32768,3,4),(True,1,2,3,4)):
            with self.assertRaises(ValueError): live.build_selected_dac_packet(invalid)

    def ack(self):
        codes=[1,0x1234,0x5678,0x2345,0x3456]
        return codes, b'\xff\xff\x02\x00'+struct.pack('>5H',*codes)+b'\0\x87\x0a\x10\0\0'

    def test_echo_and_real_i2c_write_status_are_separate_from_register_readback(self):
        codes,packet=self.ack()
        result=live.decode_dac_ack(packet,codes)
        self.assertEqual(result['echoed_codes'],codes)
        self.assertTrue(result['all_i2c_writes_acknowledged'])
        self.assertFalse(result['actual_dac_register_readback'])

    def test_wrong_codes_mode_write_status_fault_or_bus_error_are_rejected(self):
        for kind in ('code','mode','write_failed','offline','bad_part','hot','i2c_error'):
            with self.subTest(kind=kind):
                codes,original=self.ack();packet=bytearray(original)
                if kind=='code':packet[4]^=1
                elif kind=='mode':packet[14]=1
                elif kind=='write_failed':packet[15]&=~4
                elif kind=='offline':packet[15]&=~1
                elif kind=='bad_part':packet[15]&=~2
                elif kind=='hot':packet[15]|=0x10
                elif kind=='i2c_error':packet[19]=1
                with self.assertRaises(RuntimeError):live.decode_dac_ack(packet,codes)


class ActualRouteRamTests(unittest.TestCase):
    def test_all_315_codes_indices_and_waits_match(self):
        rows,data=fixture()
        result=live.validate_route_ram(data,OFFSETS,rows,[850]*63,123)
        self.assertEqual(result['individual_dac_codes_verified'],315)
        self.assertTrue(result['dac_codes_exactly_match_host'])
        self.assertFalse(result['actual_dac_register_readback'])

    def test_corruption_is_rejected_including_the_last_code(self):
        for location in (0,126,126+314*2,762,888,896,900,901,902,903,904):
            with self.subTest(location=location):
                rows,raw=fixture();data=bytearray(raw);data[location]^=1
                with self.assertRaises(RuntimeError):live.validate_route_ram(data,OFFSETS,rows,[850]*63,123)

    def test_swd_uses_uid_and_exact_flash_before_stable_ram_and_never_controls_cpu(self):
        for failure in (None,'uid','flash','moving_ram','halted'):
            with self.subTest(failure=failure):
                rows,raw=fixture()
                session=MagicMock();session.__enter__.return_value=session
                session.probe.unique_id='test_probe'
                target=session.target
                target.get_state.return_value=(live.Target.State.HALTED if failure=='halted'
                                               else live.Target.State.RUNNING)
                uid={0x1FFF7A10:0x100,0x1FFF7A14:0x32320000,0x1FFF7A18:0x20793289}
                if failure=='uid':uid[0x1FFF7A10]+=1
                target.read32.side_effect=lambda a:uid[a]
                reads=[]
                def read(address,size):
                    reads.append(address)
                    if address==0x08008000:return b'bad!' if failure=='flash' else b'CODE'
                    if failure=='moving_ram' and reads.count(0x20001000)>1:
                        return bytes([raw[0]^1])+raw[1:]
                    return raw
                target.read_memory_block8.side_effect=read
                with patch.object(live,'load_route_layout',return_value=(0x20001000,920,OFFSETS,[(0x08008000,b'CODE')])), \
                     patch.object(live.ConnectHelper,'session_with_chosen_probe',return_value=session):
                    if failure:
                        with self.assertRaises(RuntimeError):
                            live.inspect_route_readonly('test_probe',Path('fake.elf'),'207933893232',rows,[850]*63,123)
                    else:
                        result=live.inspect_route_readonly('test_probe',Path('fake.elf'),'207933893232',rows,[850]*63,123)
                        self.assertTrue(result['actual_flash_matches_elf'])
                    if failure in ('uid','halted'):self.assertEqual(reads,[])
                    if failure=='flash':self.assertNotIn(0x20001000,reads)
                for method in ('halt','resume','reset','reset_and_halt','write32','write_memory_block8'):
                    getattr(target,method).assert_not_called()

    def test_live_verifier_contains_no_optical_accuracy_gates(self):
        source=Path(live.__file__).read_text(encoding='utf-8')
        for forbidden in ('compare_full_route(', 'compare_pipeline_styles(', 'fit_temporary_frame(', 'center_limit_pm'):
            self.assertNotIn(forbidden,source)


if __name__=='__main__':unittest.main()
