"""Developer source selection checks; no packaging or hardware commands."""
import importlib.util
from pathlib import Path
import unittest


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'developer_snapshot_source_checks', REPO / 'deployment/package_developer_snapshot.py')
snapshot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(snapshot)


class DeveloperSourceSelectionTests(unittest.TestCase):
    def test_inno_installer_sources_are_included(self):
        self.assertIn('.iss', snapshot.SOURCE_EXTS)

    def test_scatter_is_source_but_keil_disassembly_is_not(self):
        self.assertTrue(snapshot.portable_file(REPO / 'JDSU/MDK-ARM/JDSU/JDSU.sct'))
        self.assertFalse(snapshot.portable_file(REPO / 'JDSU/MDK-ARM/JDSU/JDSU.disasm.txt'))
        self.assertFalse(snapshot.portable_file(REPO / 'JDSU/MDK-ARM/JDSU/JDSU.axf'))

    def test_private_backups_and_builds_are_excluded(self):
        for relative in ('Python/outputs/swd_before_PRIVATE/app.c',
                         '.release-signing/key.txt', 'build/client/module.py',
                         'Python/__pycache__/module.py'):
            self.assertFalse(snapshot.portable_file(REPO / relative), relative)


if __name__ == '__main__':
    unittest.main()
