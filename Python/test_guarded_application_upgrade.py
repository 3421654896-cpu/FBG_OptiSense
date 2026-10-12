import unittest
from guarded_application_upgrade import validate_sector,prepare_flash_clock,START,END
from unittest.mock import Mock
from pyocd.core.target import Target


class Boundaries(unittest.TestCase):
    def test_clock_must_be_halted_and_confirmed(self):
        t=Mock(); t.get_state.return_value=Target.State.RUNNING
        with self.assertRaises(RuntimeError): prepare_flash_clock(t)
        t.write32.assert_not_called()
        t.get_state.return_value=Target.State.HALTED
        t.read32.side_effect=[2,0x940a,0x9400]
        self.assertTrue(prepare_flash_clock(t)['hsi_confirmed'])
        t.write32.assert_called_once_with(0x40023808,0x9408)
    def test_protected_sectors_rejected(self):
        for base,size in [(START-0x4000,0x4000),(START-1,2),(END-1,2),(START,0)]:
            with self.assertRaises(ValueError): validate_sector(base,size)
        for base,size in [(START,0x4000),(START+0x4000,0x4000),(0x08010000,0x10000),(0x08020000,0x20000)]:
            validate_sector(base,size)


if __name__=='__main__': unittest.main()
