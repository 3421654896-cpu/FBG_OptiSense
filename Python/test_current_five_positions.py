import unittest
from pathlib import Path
from unittest.mock import patch
from evaluate_current_five_positions import evaluate, TRAINING


class ValidationSplitTests(unittest.TestCase):
    def test_rejects_deep_training_query_before_read(self):
        root = Path('unused')
        with patch('evaluate_current_five_positions._capture_feature') as reader:
            with self.assertRaisesRegex(ValueError, 'Training captures'):
                evaluate(root, [root / TRAINING[0] / 'capture.json'])
            reader.assert_not_called()

    def test_rejects_shallow_training_query_before_read(self):
        root = Path('unused')
        query = root / 'f45_visible_p08_depth010_v46_20260907_r1' / 'capture.json'
        with patch('evaluate_current_five_positions._capture_feature') as reader:
            with self.assertRaisesRegex(ValueError, 'Training captures'):
                evaluate(root, [query], include_shallow=True)
            reader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
