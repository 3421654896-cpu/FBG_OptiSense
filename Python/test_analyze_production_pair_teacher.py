import copy
import unittest

from analyze_production_pair_teacher import analyze, rc_feasibility
from capture_production_pair_teacher import decode
from test_capture_production_pair_teacher import fixture


def document():
    r = decode(fixture(), 38, 99, 2, 5)
    r.update(grating=8, repeat=0)
    return {'schema': 'production_pair_same_hold_teacher_v1', 'trained_model': False,
            'complete': True, 'captures': [r],
            'safety_cleanup': {'exact_disarm_ack': True, 'exact_soa_shutter_ack': True}}


class PairTeacherAuditTests(unittest.TestCase):
    def test_single_pole_feasibility_cannot_hide_reversal_or_flat_bias(self):
        self.assertEqual(rc_feasibility(10,20,30)['required_alpha'], .5)
        for a,b,t in [(42,28,32),(24,19,45),(10,10,30),(30,31,30)]:
            self.assertEqual(rc_feasibility(a,b,t)['classification'], 'incompatible_with_single_pole')
        self.assertEqual(rc_feasibility(30,30,30)['classification'], 'already_at_teacher')
        self.assertEqual(rc_feasibility(10,30,30)['required_alpha'], 0)

    def test_signed_error_not_pressure_accuracy(self):
        a = analyze(document())
        s = a['aggregate']['v5_5k']
        self.assertEqual(s['methods']['firmware_rc']['bias_codes'], 8)
        self.assertEqual(s['methods']['second_direct']['mae_codes'], 2)
        self.assertEqual(s['methods']['post_pair_direct']['mae_codes'], 0)
        self.assertEqual(s['methods']['firmware_rc']['within_diagnostic_tolerance_count'], 0)
        self.assertFalse(a['deployment_qualified']); self.assertFalse(a['trained_model'])

    def test_teacher_must_match_own_raw_samples(self):
        for key,value in [('median',[400,300,0,59,2,3]), ('stable',False),
                          ('dac_rewritten',True), ('adc_value_kind','rc_estimate')]:
            d = document(); d['captures'][0]['teacher'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): analyze(d)

    def test_unsafe_duplicate_future_and_nonphysical_labels_rejected(self):
        for change in range(5):
            d = document()
            if change == 0: d['safety_cleanup']['exact_disarm_ack'] = False
            if change == 1: d['captures'].append(copy.deepcopy(d['captures'][0]))
            if change == 2: d['captures'][0]['pair_available_us'] = 1
            if change == 3: d['captures'][0]['pressure_ground_truth'] = {'area_mm2':50}
            if change == 4: d['captures'][0]['firmware_rc_estimate'] = float('nan')
            with self.subTest(change=change), self.assertRaises(ValueError): analyze(d)

    def test_partial_capture_is_not_relabelled_complete(self):
        d = document(); d['complete'] = False
        self.assertFalse(analyze(d)['source_capture_complete'])

    def test_unstable_teacher_retained_excluded_from_metrics(self):
        b = bytearray(fixture())
        for i in range(25): b[112+i*20:114+i*20] = (51+i*4).to_bytes(2,'big')
        r = decode(bytes(b),38,99,2,5); r.update(grating=8,repeat=0)
        d = document(); d['captures'] = [r]
        a = analyze(d); s = a['aggregate']['v5_5k']
        self.assertEqual(s['excluded_unstable_teacher_count'],1)
        self.assertEqual(s['methods'],{})
        self.assertEqual(len(a['records']),1)


if __name__ == '__main__': unittest.main()
