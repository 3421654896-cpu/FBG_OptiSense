import copy
import json
import unittest
from audit_held_pd_normalization import audit
from capture_sparse_held_trajectory import decode
from test_capture_sparse_held_trajectory import fixture


class MemorySource:
    def __init__(self, records): self.records = records
    def read_bytes(self):
        return json.dumps({'complete':True,'captures':self.records,
            'safety_cleanup':{'exact_disarm_ack':True,'exact_soa_shutter_ack':True}}).encode()


class NormalizationTests(unittest.TestCase):
    def records(self):
        a = decode(fixture(),36,99,2)
        b = copy.deepcopy(a)
        for record, scale in ((a,1),(b,2)):
            for sample in record['teacher_samples']:
                sample['direct_adc'] = [0,20*scale,0,0,100*scale,200*scale]
        return a,b

    def test_common_gain_cancels(self):
        rows = audit(MemorySource(self.records()))['rows']
        self.assertGreater(rows[0]['ch1']['relative_std_percent'],0)
        self.assertEqual(rows[0]['ch1_over_pdt']['relative_std_percent'],0)

    def test_different_plans_are_not_pooled(self):
        a,b = self.records()
        b['planned_indices'] = [1,2,3]
        self.assertEqual(len(audit(MemorySource([a,b]))['rows']),2)

    def test_floor_denominator_is_rejected(self):
        a,b = self.records()
        a['teacher_samples'][0]['direct_adc'][4] = 0
        self.assertIn('unavailable',audit(MemorySource([a,b]))['rows'][0]['ch1_over_pdt'])


if __name__ == '__main__': unittest.main()
