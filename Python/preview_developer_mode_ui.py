"""Offscreen UI verification with isolated test data; never opens hardware."""

import argparse
import os
from pathlib import Path
from unittest import mock

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['FBG_STUDIO_BETA'] = '1'

from PyQt5 import QtGui, QtWidgets
from fbg_unified_app import SOURCE_LOCAL
from hyperos_theme import install_theme
from test_fbg_unified_app import UnifiedStudioTests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    case = UnifiedStudioTests('test_sidebar_developer_mode_is_default_off_and_only_changes_visibility')
    case.setUpClass()
    QtGui.QFontDatabase.addApplicationFont('C:/Windows/Fonts/msyh.ttc')
    install_theme(case.app)
    case.setUp()
    try:
        window = case.window
        with mock.patch.object(window.local_controller, 'acquire_local_control', return_value=True):
            window.switch_source(SOURCE_LOCAL)
        window._mode_requested(6)
        window.resize(1640, 980)
        for _ in range(4):
            case.app.processEvents()
        page = window.local_controller.page_temporary_test
        assert page.point_diagnostics_row.isHidden()
        assert page.beta_controls_row.isHidden()
        window.grab().save(str(args.output / 'developer_off.png'))
        window.developer_mode_button.click()
        for _ in range(4):
            case.app.processEvents()
        assert not page.point_diagnostics_row.isHidden()
        assert not page.beta_controls_row.isHidden()
        window.grab().save(str(args.output / 'developer_on.png'))
        def show_budget(dialog):
            dialog.show()
            case.app.processEvents()
            dialog.grab().save(str(args.output / 'developer_budget.png'))
            dialog.close()
            return QtWidgets.QDialog.Rejected
        with mock.patch.object(QtWidgets.QDialog, 'exec_', show_budget):
            page.show_developer_budget()
        print('Verified developer OFF / ON / read-only budget; no hardware opened.')
    finally:
        case.tearDown()
        case.doCleanups()


if __name__ == '__main__':
    main()
