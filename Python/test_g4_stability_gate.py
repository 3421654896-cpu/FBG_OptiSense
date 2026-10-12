import unittest
import numpy as np
from audit_g4_stability_gate import window_quality


class GateTests(unittest.TestCase):
    def test_constant_and_slow_drift(self):
        values=np.full((25,6),40.)
        self.assertTrue(window_quality(values)['window_pass'])
        values[:,3]=np.linspace(40,48,25)
        self.assertFalse(window_quality(values)['window_pass'])

    def test_two_flat_windows_need_separate_repeat_check(self):
        first=window_quality(np.full((25,6),40.))
        second=window_quality(np.full((25,6),50.))
        self.assertTrue(first['window_pass'] and second['window_pass'])
        self.assertGreater(abs(first['median']-second['median']),3)


if __name__=='__main__': unittest.main()
