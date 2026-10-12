import unittest
from analyze_sparse_held_trajectory import audit_record
from capture_sparse_held_trajectory import decode, validate_full_map_reference
from test_capture_sparse_held_trajectory import fixture


class HeldAuditTests(unittest.TestCase):
    def test_full_map_reference_requires_entire_route_and_gap(self):
        record = {'firmware_version':0x10036,'prefix_profile':0,
                  'planned_indices':list(range(45)), 'prefix_bandwidth_discontinuity':True}
        validate_full_map_reference(record)
        for key, value in [('prefix_profile',2),('planned_indices',[16,17,18]),
                           ('prefix_bandwidth_discontinuity',False)]:
            with self.assertRaises(ValueError):
                validate_full_map_reference({**record,key:value})

    def test_monotone_and_overshooting_shapes(self):
        record = decode(fixture(),36,99,2)
        record['teacher']['median'][3] = 40
        for i, sample in enumerate(record['curve']):
            sample['direct_adc'][1] = 8+i
        self.assertEqual(audit_record(record)['transient_shape']['initial_to_teacher_interval_excursion_codes'],0)
        record['curve'][12]['direct_adc'][1] = 60
        shape = audit_record(record)['transient_shape']
        self.assertEqual(shape['initial_to_teacher_interval_excursion_codes'],20)
        self.assertTrue(shape['exceeds_descriptive_tolerance'])

    def test_undershoot_is_detected(self):
        record = decode(fixture(),36,99,2)
        record['teacher']['median'][3] = 40
        for sample in record['curve']:
            sample['direct_adc'][1] = 35
        record['curve'][5]['direct_adc'][1] = 20
        self.assertEqual(audit_record(record)['transient_shape']['initial_to_teacher_interval_excursion_codes'],15)

    def test_uses_completion_not_start_or_future_sample(self):
        record = decode(fixture(),36,99,2)
        result = audit_record(record,horizons=(1,50,750))
        self.assertIsNone(result['early_direct_errors'][0]['direct_ch1'])
        self.assertIsNone(result['early_direct_errors'][1]['direct_ch1'])
        self.assertLessEqual(result['early_direct_errors'][2]['sample_end_us'],750)

    def test_unstable_or_rewritten_teacher_is_rejected(self):
        for field,value in [('stable',False),('dac_rewritten',True),('adc_value_kind','firmware_rc_estimate')]:
            record = decode(fixture(),36,99,2)
            record['teacher'][field] = value
            with self.assertRaises(ValueError): audit_record(record)

    def test_late_disagreement_has_no_claimed_settling_bound(self):
        record = decode(fixture(),36,99,2)
        record['curve'][-1]['direct_adc'][1] = 90
        self.assertIsNone(audit_record(record)['sampled_suffix_within_tolerance_from_us'])

    def test_suffix_cannot_ignore_later_excursion(self):
        record = decode(fixture(),36,99,2)
        record['curve'][20]['direct_adc'][1] = 90
        self.assertEqual(audit_record(record)['sampled_suffix_within_tolerance_from_us'],record['curve'][21]['end_us'])


if __name__ == '__main__': unittest.main()
