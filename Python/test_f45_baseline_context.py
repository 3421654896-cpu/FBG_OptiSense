import unittest
import numpy as np
from evaluate_f45_g4_ablation import candidate_features


class BaselineContextTests(unittest.TestCase):
    def test_no_future_or_label_features(self):
        frames=[dict(sequence=i,records=[dict(first_code=i,second_code=i+1) for _ in range(45)]) for i in range(100)]
        data=dict(frames=frames,commissioning={})
        x=np.zeros((100,135))
        before=candidate_features(data,x,'baseline_context')
        for f in frames[80:]:
            for r in f['records']: r['first_code']=4095; r['second_code']=4095
        data['commissioning'].update(depth_mm=.8,target={'point_index':17})
        after=candidate_features(data,x,'baseline_context')
        np.testing.assert_array_equal(before,after)
        self.assertEqual(before.shape,(100,225))
        self.assertEqual(before[0,135],47.5)
        self.assertEqual(candidate_features(data,x,'without_g4').shape,(100,120))


if __name__=='__main__': unittest.main()
