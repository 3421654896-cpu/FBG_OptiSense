import copy
import unittest

from study_position_feature_distance import study
from test_evaluate_fresh_sentinel_position import IndependentPositionTests


class FeatureStudyTests(unittest.TestCase):
    def captures(self):
        train = IndependentPositionTests().capture()
        for record in train['records']:
            record['baseline9'] = [20, 20]
        test = copy.deepcopy(train)
        test.update(source='test', source_sha256={'frames.jsonl': 'different'})
        return train, test

    def test_reverse_route_and_flags(self):
        train, test = self.captures()
        test['records'].reverse()
        result = study(train, test)
        for variant in result['variants'].values():
            self.assertEqual(variant['contact']['exact_grid_point_percent'], 100)
        self.assertFalse(result['deployment_eligible'])
        self.assertFalse(result['physical_15hz_verified'])

    def test_test_truth_does_not_change_predictions(self):
        train, test = self.captures()
        before = study(train, test)
        for record in test['records']:
            record['model_xy_mm'] = [99, 99]
        after = study(train, test)
        for name in before['variants']:
            self.assertEqual(
                [p['predicted_xy'] for p in before['variants'][name]['predictions']],
                [p['predicted_xy'] for p in after['variants'][name]['predictions']])

    def test_duplicate_capture_rejected(self):
        train, _ = self.captures()
        with self.assertRaises(ValueError):
            study(train, train)


if __name__ == '__main__':
    unittest.main()
