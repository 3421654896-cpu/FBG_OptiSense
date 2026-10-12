import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from machine_identity import MachineIdentityStore, identity_for_port, native_ports


def port(device='COM18', serial='207933893232', vid=0x0483, pid=0x5740):
    return SimpleNamespace(device=device, serial_number=serial, vid=vid, pid=pid)


class MachineIdentityTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / 'bindings.json'
        self.store = MachineIdentityStore(self.path)

    def test_verified_two_and_three_identify_without_com_number_guess(self):
        self.assertEqual(self.store.resolve('395736523033'), 'machine_2')
        self.assertEqual(self.store.resolve('207933893232'), 'machine_3')
        self.assertEqual(self.store.resolve('207A33813232'), 'machine_4')
        self.assertEqual(identity_for_port([port('COM99')], 'com99'),
                         ('COM99', '207933893232'))
        self.assertEqual(identity_for_port([port(serial='395736523033')], 'COM18')[1],
                         '395736523033')

    def test_unknown_never_defaults_to_last_machine(self):
        self.assertIsNone(self.store.resolve('AABBCCDDEEFF'))
        self.assertFalse(self.path.exists())

    def test_first_confirmed_binding_persists_without_modifying_registry(self):
        self.store.bind('AABBCCDDEEFF', 'machine_1')
        again = MachineIdentityStore(self.path)
        self.assertEqual(again.resolve('aabbccddeeff'), 'machine_1')

    def test_known_machine_and_serial_bindings_cannot_be_overwritten(self):
        for serial, machine in (('207933893232', 'machine_2'),
                                ('AABBCCDDEEFF', 'machine_3'),
                                ('AABBCCDDEEFF', 'machine_4'),
                                ('AABBCCDDEEFF', 'machine_99')):
            with self.assertRaises(ValueError):
                self.store.bind(serial, machine)
        self.assertFalse(self.path.exists())

    def test_corrupt_binding_is_not_silently_used(self):
        self.path.write_text(json.dumps(dict(schema='fbg_machine_identity_v1',
                         bindings={'207933893232': 'machine_1'})), encoding='utf-8')
        with self.assertRaises(ValueError):
            MachineIdentityStore(self.path)

    def test_debugger_ports_invalid_serials_and_duplicates_are_not_identity(self):
        self.assertEqual(native_ports([port(vid=0x1A86, pid=0x8010)]), [])
        for ports in ([port(serial=None)], [port(), port('COM19')],
                      [port(vid=0x1A86, pid=0x8010)]):
            with self.assertRaises(ValueError):
                identity_for_port(ports, 'COM18')


if __name__ == '__main__':
    unittest.main()
