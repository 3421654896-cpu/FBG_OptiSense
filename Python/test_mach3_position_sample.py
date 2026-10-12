import unittest
from pathlib import Path
from unittest.mock import patch
from mach3_controller import Mach3Controller


class PositionSampleTests(unittest.TestCase):
    def test_distinct_read_only_operation(self):
        with patch.object(Mach3Controller,'_request',return_value={'machine_z':14}) as request:
            self.assertEqual(Mach3Controller().position_sample(),{'machine_z':14})
        request.assert_called_once_with({'op':'position_sample'})

    def test_no_motion_or_safety_claims_in_reduced_read(self):
        source=Path(__file__).with_name('mach3_bridge.ps1').read_text('utf-8')
        body=source.split('function Get-Mach3PositionSample {',1)[1].split('\nfunction ',1)[0]
        for command in ('.Code(','.DoOEMButton(', 'Release-EmergencyStop', 'Invoke-MicroMove'):
            self.assertNotIn(command,body)
        self.assertIn('encoder_feedback_verified = $false',body)
        self.assertIn('safety_status_included = $false',body)
        self.assertIn('coordinates_simultaneous = $false',body)


if __name__=='__main__': unittest.main()
