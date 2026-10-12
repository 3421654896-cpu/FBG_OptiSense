import unittest
from copy import deepcopy
import numpy as np
from select_balanced_candidate27 import select_group


def spectrum():
    x = np.linspace(1530, 1531.6, 81)
    y = 30 + 600*np.exp(-.5*((x-1530.8)/.3)**2)*(1+.15*np.cos((x-1530.8)*95))
    return [dict(measured_wavelength_nm=float(a), ch1_adc_code=float(b), stable=True)
            for a, b in zip(x, y)]


class BalancedCandidateTests(unittest.TestCase):
    def test_ripple_does_not_clump_core_and_raw_is_preserved(self):
        rows = spectrum(); before = deepcopy(rows)
        chosen, info = select_group(rows)
        self.assertEqual(rows, before)
        self.assertEqual(len(set(chosen)), 5)
        self.assertGreater(min(info['core_spans_nm']), .25)
        self.assertLess(abs(np.diff(info['core_spans_nm'])[0]), .025)

    def test_unstable_core_is_not_selected(self):
        rows = spectrum(); rows[40]['stable'] = False
        chosen, _ = select_group(rows)
        self.assertNotIn(40, chosen)

    def test_no_stable_core_is_rejected(self):
        rows = spectrum()
        for r in rows[36:45]: r['stable'] = False
        with self.assertRaises(ValueError): select_group(rows)

    def test_nonfinite_flat_truncated_unsorted_and_saturated_rejected(self):
        cases = []
        rows = spectrum(); rows[30]['ch1_adc_code'] = float('nan'); cases.append(rows)
        rows = spectrum()
        for r in rows: r['ch1_adc_code'] = 100
        cases.append(rows)
        cases.append(spectrum()[30:])
        cases.append(spectrum()[::-1])
        rows = spectrum(); rows[40]['ch1_adc_code'] = 4095; cases.append(rows)
        for rows in cases:
            with self.subTest(first=rows[0]):
                with self.assertRaises(ValueError): select_group(rows)


if __name__ == '__main__': unittest.main()
