"""Diagnostic reselection identity/authorization gates; no physical hardware."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import diagnose_optical_route_live as live
from test_temporary_symmetric_seven import selected


class DensePreflightTests(unittest.TestCase):
    def test_without_stationary_light_confirmation_no_files_or_hardware_are_read(self):
        argv = ['dense', '--source-capture', 'missing', '--output', 'unused',
                '--machine-id', 'machine_3']
        with patch('sys.argv', argv), patch.object(live, 'verify_usb') as verify, \
                patch.object(live.app, 'load_fullband_accuracy_table') as load, \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                live.main()
        verify.assert_not_called()
        load.assert_not_called()

    def test_explicit_wrong_machine_tag_cannot_be_ignored(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            plan = selected()[-1]
            source = path/'source.json'
            source.write_text(json.dumps(dict(plan=plan, machine_id='machine_2', signal_channel=0)))
            with patch('sys.argv', ['dense', '--source-capture', str(source), '--output', str(path/'out'),
                                   '--machine-id', 'machine_3', '--allow-light', '--stationary-setup']), \
                    patch.object(live, 'verify_usb') as verify:
                with self.assertRaisesRegex(ValueError, 'machine3 CH0'):
                    live.main()
                verify.assert_not_called()

    def test_legacy_tagless_capture_requires_every_source_code_to_match_current_table(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            table, _, _, plan = selected()
            plan['rows'][0]['codes'][0] += 1
            source = path/'source.json'
            source.write_text(json.dumps(dict(plan=plan, signal_channel=0)))
            with patch('sys.argv', ['dense', '--source-capture', str(source), '--output', str(path/'out'),
                                   '--machine-id', 'machine_3', '--allow-light', '--stationary-setup']), \
                    patch.object(live.app, 'load_fullband_accuracy_table', return_value=table), \
                    patch.object(live, 'verify_usb') as verify, \
                    patch.object(live, 'ThermalSerial') as port_open, \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, 'table no longer matches'):
                    live.main()
                verify.assert_not_called()
                port_open.assert_not_called()
            self.assertFalse((path/'out').exists())


if __name__ == '__main__':
    unittest.main()
