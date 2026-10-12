import unittest
from unittest.mock import patch
import commission_fast_flank_center_contact as commission


class FastCommissioningPreflight(unittest.TestCase):
    def test_invalid_fast_settings_rejected_before_controller(self):
        base=['--protocol','flank18_fast','--operator-confirmed','--output','outputs/unit_never_created.json']
        cases=[['--cycles','1024','--baseline-start-sequence','256'],
               ['--cycles','2048','--baseline-start-sequence','16'],
               ['--cycles','2048','--baseline-start-sequence','256','--spacing-us','600'],
               ['--cycles','2048','--baseline-start-sequence','256','--depth-mm','0.81']]
        with patch.object(commission,'Mach3Controller') as controller:
            for args in cases:
                with self.subTest(args=args), patch('sys.stderr'), self.assertRaises(SystemExit):
                    commission.main(base+args)
            controller.assert_not_called()


if __name__=='__main__': unittest.main()
