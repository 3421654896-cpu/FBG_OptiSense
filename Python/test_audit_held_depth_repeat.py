import copy
import unittest
from audit_held_depth_repeat import compare


class RepeatAuditTests(unittest.TestCase):
    def rows(self):
        a=dict(requested_depth_mm=.3, source_sha256='a', table_sha256='same',
               schedule_version=5,feedback_selectors=[2,2],profile=2,
               machine_xyz_mm=[1,2,3],teacher={'median':[0,0,0,32]})
        b=copy.deepcopy(a)
        b['source_sha256']='b'
        b['teacher']['median'][3]=53
        return a,b

    def test_differing_teacher_preserved_not_corrected(self):
        a,b=self.rows()
        result=compare([a],[b])
        self.assertEqual(result['pairs'][0]['teacher_ch1_pair'],[32,53])
        self.assertFalse(result['deployment_eligible'])

    def test_duplicate_and_gain_mismatch_rejected(self):
        a,b=self.rows()
        with self.assertRaises(ValueError): compare([a],[a])
        b['feedback_selectors']=[3,3]
        with self.assertRaises(ValueError): compare([a],[b])
