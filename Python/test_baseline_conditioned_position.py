import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from select_f45_baseline_conditioned_position import features


class BaselineFeaturesTests(unittest.TestCase):
    def test_future_release_samples_do_not_change_held_features(self):
        rows=[]
        for seq in range(140):
            rows.append({'sequence':seq,'records':[{'first_code':100+r+(3 if 90<=seq<100 else 0),
                'second_code':110+r+(4 if 90<=seq<100 else 0)} for r in range(45)]})
        events={'contact_descent_started':81,'contact_reached_commanded_depth':88,
                'release_started':100,'safe_height_reached':110}
        data={'frames':rows,'feedback_selectors':[0,2],'commissioning':{
            'frame_receipts':[{'sequence':s,'host_monotonic_ns':s*100_000_000} for s in range(140)],
            'events':[{'kind':k,'host_monotonic_ns':v*100_000_000} for k,v in events.items()]}}
        wavelengths=np.arange(45)*.02+1530
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'trace.json'
            path.write_text(json.dumps(data),'utf-8')
            original=features(path,wavelengths)
            for f in data['frames'][110:]:
                for r in f['records']: r['first_code']=3000; r['second_code']=3500
            path.write_text(json.dumps(data),'utf-8')
            modified=features(path,wavelengths)
            for key in original:
                self.assertEqual(len(original[key]),10)
                self.assertTrue(np.isfinite(original[key]).all())
                np.testing.assert_array_equal(original[key],modified[key])
            data['feedback_selectors']=[0,1]
            path.write_text(json.dumps(data),'utf-8')
            with self.assertRaisesRegex(ValueError,'gains'):
                features(path,wavelengths)


if __name__=='__main__': unittest.main()
