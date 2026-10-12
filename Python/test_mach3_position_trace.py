import unittest
from capture_mach3_position_trace import summarize


class TraceTests(unittest.TestCase):
    def samples(self):
        return [dict(host_request_perf_ns=i*10_000_000,host_received_perf_ns=i*10_000_000+1_000_000,
                     position=dict(machine_x=1,machine_y=2,machine_z=3)) for i in range(5)]

    def test_speed_does_not_prove_encoder_or_dynamic_response(self):
        result=summarize(self.samples())
        self.assertAlmostEqual(result['receive_rate_hz'],100)
        self.assertFalse(result['encoder_refresh_rate_verified'])
        self.assertFalse(result['dynamic_15hz_reference_verified'])
        self.assertFalse(result['synchronized_to_adc'])

    def test_time_overlap_rejected(self):
        samples=self.samples();samples[2]['host_request_perf_ns']=0
        with self.assertRaises(ValueError):summarize(samples)
