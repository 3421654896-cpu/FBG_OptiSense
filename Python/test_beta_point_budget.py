import copy
import unittest

import temporary_test_mode as t
from test_beta_single_adc import single_frame
from test_temporary_symmetric_seven import selected
from validate_beta_single_adc_live import point_budget, validate_capture


def capture():
    plan=selected()[-1]
    records=[]
    for sequence in range(32):
        _,crc,wire=single_frame(plan,sequence=sequence)
        frame=t.decode(wire,123,crc,first_delay_us=850,point_delays_us=[850]*63,
                       point_count=63,points_per_peak=7,single_sample=True,fast_dac_writes=True)
        records.append(frame)
    return dict(complete=True,received_frames=32,disarm_ack=True,shutter_ack=True,
                point_count=63,signal_channel=1,feedback_selector=2,frames=records,
                point_delays_us=[850]*63,spacing_us=0,single_sample=True,
                samples_per_point=1,fast_dac_writes=True,mean_hz=10,p99_period_ms=100)


class PointBudgetTests(unittest.TestCase):
    def test_real_single_stamp_budget_sums_to_board_period(self):
        result=validate_capture(capture(),0x10073,32,63)
        self.assertEqual(result['prescribed_wait_ms'],53.55)
        self.assertAlmostEqual(result['adc_and_hold_overhead_ms'],.315)
        self.assertAlmostEqual(sum(result[k] for k in (
            'dac_and_interpoint_ms','prescribed_wait_ms','adc_and_hold_overhead_ms',
            'frame_tail_ms','frame_gap_ms')),result['period_ms'])

    def test_incomplete_bus_and_fake_pair_are_not_timing_acceptance(self):
        for mode in ('count','version','bus','pair','stop'):
            report=capture()
            if mode=='count':report['frames'].pop()
            elif mode=='version':report['frames'][3]['firmware']+=1
            elif mode=='bus':report['frames'][3]['spi_errors']=1
            elif mode=='pair':report['frames'][3]['records'][0]['second_code']=100
            else:report['shutter_ack']=False
            with self.assertRaises(RuntimeError):validate_capture(report,0x10073,32,63)

    def test_clock_inconsistent_with_sample_timing_rejected(self):
        report=capture()
        report['frames'][16]['elapsed_us']=1
        with self.assertRaises(ValueError):point_budget(report)


if __name__=='__main__':unittest.main()
