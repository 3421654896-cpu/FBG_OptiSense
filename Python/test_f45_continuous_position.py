import unittest
import numpy as np
from evaluate_f45_continuous_position import target_label, metrics


class ContinuousPositionTests(unittest.TestCase):
    def test_work_offset_is_not_position_change(self):
        def data(x, offset):
            return {'commissioning': {'depth_mm': .12, 'target': {'point_index': 13,
                'work_x': x, 'work_y': 2, 'contact_z': 3,
                'calibration_machine_minus_work': [offset, 0, 0]}}}
        item = {'point': 13, 'depth_mm': .12}
        np.testing.assert_allclose(target_label(data(1, 10), item), target_label(data(11, 0), item))
        with self.assertRaises(ValueError): target_label(data(1, 10), dict(point=14, depth_mm=.12))

    def test_xy_distance_and_depth_are_separate(self):
        result = metrics([[3, 4, .3], [0, 0, .1]], np.array([0, 0, .2]))
        self.assertAlmostEqual(result['xy_mean_mm'], 2.5)
        self.assertAlmostEqual(result['depth_mae_mm'], .1)
        self.assertAlmostEqual(result['depth_bias_mm'], 0)
        with self.assertRaises(ValueError): metrics([], np.zeros(3))


if __name__ == '__main__': unittest.main()
