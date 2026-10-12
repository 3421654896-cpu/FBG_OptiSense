import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from probe_candidate_route_guarded import execute


class GuardedRouteTests(unittest.TestCase):
    def test_invalid_hold_and_index_fail_before_output_or_hardware(self):
        with TemporaryDirectory() as td:
            for index,hold in ((-1,.005),(1400,.0004),(1400,float('nan'))):
                output=Path(td)/f'{index}_{hold}.json'
                with self.assertRaises(ValueError):execute(Path('missing'),output,index,hold)
                self.assertFalse(output.exists())


if __name__=='__main__':unittest.main()
