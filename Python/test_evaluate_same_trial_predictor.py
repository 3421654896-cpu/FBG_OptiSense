import copy
import unittest

import numpy as np

from capture_production_pair_teacher import decode
from evaluate_same_trial_predictor import (MODEL_NAMES, causal_features, grouped_folds,
                                          train_candidate, predict_candidate, metrics)
from test_capture_production_pair_teacher import fixture


def make_row(index=0):
    r = decode(fixture(),38,99,2,5)
    r.update(grating=index%3+1,repeat=index%2)
    return {'id':str(index),'raw':r,'gain_scale':2.5,'teacher':51.0,
            'grating':r['grating'],'repeat':r['repeat'],
            'features':{n:causal_features(r,n) for n in MODEL_NAMES}}


class CausalPredictorTests(unittest.TestCase):
    def test_features_ignore_teacher_and_point_identity(self):
        r = make_row()['raw']; changed = copy.deepcopy(r)
        for key in ['teacher','teacher_samples','grating','point_index','plan_sequence','frame_start_ms','request_tag']:
            changed[key] = 'must never become an input'
        for name in MODEL_NAMES:
            np.testing.assert_array_equal(causal_features(r,name)[0],causal_features(changed,name)[0])

    def test_features_use_real_availability_and_reject_early_extra_read(self):
        r = make_row()['raw']
        self.assertEqual(causal_features(r,'ridge_pair')[1],700)
        self.assertEqual(causal_features(r,'ridge_optical')[1],795)
        bad = copy.deepcopy(r); bad['optical_state']['start_us']=650
        with self.assertRaises(ValueError): causal_features(bad,'ridge_optical')

    def test_grouped_folds_never_mix_grating_or_repeat(self):
        rows = [make_row(i) for i in range(18)]
        for key in ('grating','repeat'):
            seen=[]
            for group,train,test in grouped_folds(rows,key):
                self.assertTrue(set(train).isdisjoint(test)); seen.extend(test)
                self.assertTrue(all(rows[i][key]!=group for i in train))
                self.assertTrue(all(rows[i][key]==group for i in test))
            self.assertEqual(sorted(seen),list(range(18)))

    def test_prediction_scales_with_input_not_stored_teacher_constant(self):
        training=[make_row(i) for i in range(18)]
        original=make_row(100); scaled=copy.deepcopy(original)
        for rd in scaled['raw']['production_reads']: rd['direct_ch1']*=2
        scaled['raw']['optical_state']['direct_adc'][1]*=2
        scaled['features']={n:causal_features(scaled['raw'],n) for n in MODEL_NAMES}
        for name in MODEL_NAMES:
            model=train_candidate(name,training)
            out=predict_candidate(name,model,[original,scaled])
            self.assertAlmostEqual(out[1],2*out[0],places=8)

    def test_zero_signal_ratio_model_abstains_and_coverage_is_explicit(self):
        r=make_row(); r['raw']['production_reads'][-1]['direct_ch1']=0
        self.assertIsNone(causal_features(r['raw'],'ratio_optical')[0])
        m=metrics([make_row(0),make_row(1)],np.array([np.nan,51]))
        self.assertEqual(m['total'],2); self.assertEqual(m['predicted'],1)
        self.assertEqual(m['coverage'],.5)


if __name__=='__main__': unittest.main()
