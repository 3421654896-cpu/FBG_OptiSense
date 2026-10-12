"""Render the 63-point t90 UI using a saved real capture; hardware I/O blocked."""
import argparse
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.capture.read_text(encoding='utf-8'))
    plan = json.loads(args.session.read_text(encoding='utf-8'))['plan']
    if plan['rows'] != report['route']:
        raise ValueError('Preview session route does not match the actual measurement')
    with tempfile.TemporaryDirectory(prefix='fbg-t90-preview-') as folder:
        os.environ.update(QT_QPA_PLATFORM='offscreen', QT_OPENGL='software',
                          FBG_STUDIO_BETA='1', FBG_STUDIO_DATA_DIR=folder)
        import serial
        import app_JDSU as app
        from fbg_lan_serial import FbgLanSerial
        from machine_profile import set_runtime_machine_id
        from PyQt5 import QtCore, QtGui, QtWidgets
        from hyperos_theme import install_theme
        from temporary_test_widget import TemporaryTestWindow
        set_runtime_machine_id(report['machine_id'])
        with patch.object(serial.Serial, 'open', side_effect=RuntimeError('Preview: USB blocked')), \
             patch.object(FbgLanSerial, 'open', side_effect=RuntimeError('Preview: LAN blocked')), \
             patch.object(app, 'serial_write', side_effect=RuntimeError('Preview: writes blocked')):
            qt = QtWidgets.QApplication([])
            QtGui.QFontDatabase.addApplicationFont('C:/Windows/Fonts/msyh.ttc')
            install_theme(qt)
            window = TemporaryTestWindow(None)
            window.channel_combo.setCurrentIndex(report['signal_channel'])
            window.analog_gain.setCurrentIndex(window.analog_gain.findData(report['feedback_selector']))
            window.plan = plan
            window.render_plan(ready=True)
            window.worker = SimpleNamespace(output=args.capture.resolve())
            window.switch_timing_ready(report)
            window.worker = None
            window.refresh_wait_controls()
            window.resize(1600, 1000)
            window.show()
            for _ in range(8): qt.processEvents()
            args.output_dir.mkdir(parents=True, exist_ok=True)
            window.grab().save(str(args.output_dir/'t90_controls_63.png'))
            def render_dialog(dialog):
                dialog.show()
                for _ in range(8): qt.processEvents()
                dialog.grab().save(str(args.output_dir/'t90_results_63.png'))
                dialog.reject()
                return QtWidgets.QDialog.Rejected
            with patch.object(QtWidgets.QDialog, 'exec_', render_dialog):
                window.show_switch_timing_results()
            print(json.dumps({'capture': str(args.capture), 'hardware_io': False,
                              'point_count': window.point_count(),
                              'apply_button_visible': window.beta_apply_t90_waits.isVisible(),
                              'results_enabled': window.beta_timing_results.isEnabled(),
                              'status': window.status.text()}, ensure_ascii=True))
            window.close()
            qt.processEvents(QtCore.QEventLoop.AllEvents)


if __name__ == '__main__':
    main()
