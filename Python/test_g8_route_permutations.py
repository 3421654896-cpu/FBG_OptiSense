import unittest
from pathlib import Path
import app_JDSU as app
from probe_g8_route_permutations import make_cases, summarize


class G8RoutePermutationTests(unittest.TestCase):
    def test_six_unique_orders_use_same_three_rows(self):
        rows, cases=make_cases(app.load_fullband_accuracy_table(Path(__file__).parent))
        self.assertEqual(len(cases),6);self.assertEqual(len(set(cases)),6)
        self.assertTrue(all(set(case)=={1523,1541,1559} for case in cases))
        self.assertEqual(set(rows),{1319,1338,1357,1523,1541,1559})

    def test_summary_requires_every_reading(self):
        base=dict(target_order=[1523,1541,1559],adc={'stable':True},
                  meter={'side_mode_suppression_db':30},wavelength_error_pm=0)
        report={'captures':[dict(base) for _ in range(3)]}
        self.assertTrue(summarize(report)[0]['all_pass'])
        report['captures'][1]['wavelength_error_pm']=51
        self.assertFalse(summarize(report)[0]['all_pass'])


if __name__=='__main__': unittest.main()
