import unittest
from audit_f45_dynamic_timing import sinusoid_skew


class TimingTests(unittest.TestCase):
    def test_half_period_is_opposite_phase(self):
        result = sinusoid_skew([100, 50100], 10)
        self.assertAlmostEqual(result['phase_span_degrees'], 180)
        self.assertAlmostEqual(result['first_last_max_difference_per_unit_amplitude'], 2)
        self.assertAlmostEqual(result['hypothetical_equal_average_amplitude_ratio'], 0)

    def test_absolute_offset_does_not_matter(self):
        self.assertEqual(sinusoid_skew([1, 101], 15), sinusoid_skew([1001, 1101], 15))
        with self.assertRaises(ValueError): sinusoid_skew([2, 1], 15)
        with self.assertRaises(ValueError): sinusoid_skew([1, 2], 0)


if __name__ == '__main__': unittest.main()
