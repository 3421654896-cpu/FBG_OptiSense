import unittest
from pathlib import Path
import app_JDSU as app
from probe_g8_guard_ladders import CASES,validate_cases,execute


class G8GuardTests(unittest.TestCase):
    def test_cases_are_bounded_ordered_and_safe(self):
        validate_cases(app.load_fullband_accuracy_table(Path(__file__).parent))
        self.assertEqual(len(CASES),8);self.assertEqual(len({n for n,_ in CASES}),8)
        self.assertEqual(max(len(g) for _,g in CASES),9)

    def test_modified_identity_is_rejected(self):
        points=list(app.load_fullband_accuracy_table(Path(__file__).parent))
        points[1400]=app.FullbandAccuracyPoint(1401,points[1400].target_nm,
                                               points[1400].measured_nm,points[1400].codes)
        with self.assertRaises(ValueError):validate_cases(points)

    def test_unknown_or_duplicate_case_rejected_before_opening_output(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as td:
            for names in (('unknown',),('one_1400','one_1400')):
                with self.assertRaises(ValueError):execute(Path(td)/('x'+str(len(names))+names[0]),1,.02,names)

    def test_sub_half_millisecond_hold_is_rejected_before_hardware(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as td:
            with self.assertRaises(ValueError):execute(Path(td)/'x',1,.0004,('one_1400',))

    def test_more_than_ten_repeats_is_rejected_before_hardware(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as td:
            with self.assertRaises(ValueError):execute(Path(td)/'x',11,.001,('one_1400',))


if __name__=='__main__':unittest.main()
