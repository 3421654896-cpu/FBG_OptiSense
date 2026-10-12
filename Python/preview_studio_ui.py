"""Isolated, hardware-blocked UI review. No client records or board I/O.

--render-dir captures Qt widgets offscreen; without it the same safe preview
can be inspected through Windows Computer Use. Never use this as a client.
"""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-dir', type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='fbg-ui-review-') as folder, ExitStack() as guards:
        os.environ['FBG_STUDIO_DATA_DIR'] = folder
        if args.render_dir:
            os.environ['QT_QPA_PLATFORM'] = 'offscreen'
            os.environ['QT_OPENGL'] = 'software'
        import serial
        guards.enter_context(patch.object(serial.Serial, 'open',
                                         side_effect=RuntimeError('UI preview: USB I/O disabled')))
        from PyQt5 import QtCore, QtGui, QtWidgets
        import pyqtgraph as pg
        import numpy as np
        from fbg_unified_app import UnifiedFbgStudio, SOURCE_LOCAL
        from fbg_network_transport import NETWORK_LAN
        from hyperos_theme import install_theme, polish_plots
        from ota_machine_profiles import OtaMachineProfileStore
        from fbg_lan_serial import FbgLanSerial
        import app_JDSU as app
        guards.enter_context(patch.object(app.MainWindow, 'acquire_local_control', return_value=False))
        guards.enter_context(patch.object(FbgLanSerial, 'open',
                                         side_effect=RuntimeError('UI preview: LAN I/O disabled')))
        guards.enter_context(patch.object(app, 'serial_write', return_value=False))
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
        qt = QtWidgets.QApplication([])
        # Qt's offscreen Windows plugin doesn't enumerate system fonts.
        QtGui.QFontDatabase.addApplicationFont('C:/Windows/Fonts/msyh.ttc')
        install_theme(qt)
        window = UnifiedFbgStudio(initial_source=NETWORK_LAN, demo=True,
                                 ota_settings_store=OtaMachineProfileStore(Path(folder)/'ota.json'))
        # Complete the constructor's deferred initial-source selection before
        # pinning the preview to native pages; otherwise its zero-time timer
        # returns to the remote demo page on the first paint.
        for _ in range(4):
            qt.processEvents()
        window.transport.stop()
        window.setWindowTitle('OptiSense Studio — 界面验证（无硬件通信）')
        window.remote_controller.deactivate_network()
        window.current_source = SOURCE_LOCAL
        window.source_stack.setCurrentWidget(window.local_page)
        window._set_checked_source(SOURCE_LOCAL)
        window._update_source_status('界面验证 · 无硬件通信', 'info')
        for button in window.source_buttons.values():
            button.setEnabled(False)
        window.ota_page.usb_update_button.setEnabled(False)

        def mode_only(index):
            window.local_controller.MCU_mode = index
            window.local_controller.stack.setCurrentIndex(index)
            window._sync_local_mode()
        window._mode_requested = mode_only
        for index in (2,3,6):
            window.mode_buttons[index].setEnabled(True)
        mode_only(6)
        page = window.local_controller.page_temporary_test
        page.status.setText('演示光谱 · 仅用于界面排版验证，不代表实测值或精度验收')
        page.peak_fit_status.setText('9峰布局示例 · 原始圆点与拟合线分层显示')
        # Synthetic chart content makes plot boundaries/labels testable without
        # touching the optical source, acquisition worker or saved routes.
        for i in range(9):
            x = np.linspace(1528 + 2.8*i - .4, 1528 + 2.8*i + .4, 9)
            y = -5 - i*.8 - ((x - (1528+2.8*i))/.21)**2 * 2.8
            page.plot.plot(x, y, pen=pg.mkPen(pg.intColor(i,9), width=2),
                           symbol='o', symbolSize=4, symbolBrush=pg.intColor(i,9))
            page.peak_fit_table.setItem(0, i, QtWidgets.QTableWidgetItem(f'{1528+2.8*i:.4f}'))
        page.plot.setXRange(1525,1555)
        page.plot.setYRange(-25,0)
        polish_plots(window)
        window.show()

        def drain():
            for _ in range(12):
                qt.processEvents(QtCore.QEventLoop.AllEvents, 30)

        if args.render_dir:
            args.render_dir.mkdir(parents=True, exist_ok=True)
            geometry = []
            for width, height in ((960,620), (1366,768), (1600,900)):
                window.resize(width,height)
                for name, mode, tab, advanced in (
                        ('finger',6,0,False), ('finger_advanced',6,0,True),
                        ('single',3,0,False), ('calibration',2,0,False)):
                    mode_only(mode)
                    window.workspace_tabs.setCurrentIndex(tab)
                    page.advanced_toggle.setChecked(advanced)
                    drain()
                    target = args.render_dir / f'{name}_{width}x{height}.png'
                    window.grab().save(str(target))
                    geometry.append(dict(view=name, size=[window.width(),window.height()],
                                         plot=page.plot.geometry().getRect(), screenshot=str(target),
                                         content=page.page_content.size().width() if hasattr(page,'page_content') else None,
                                         layout_min=(page.page_content.layout().minimumSize().width(),page.page_content.layout().minimumSize().height()),
                                         item_mins=[(type(page.page_content.layout().itemAt(i).widget()).__name__,page.page_content.layout().itemAt(i).minimumSize().height(),page.page_content.layout().itemAt(i).isEmpty()) for i in range(page.page_content.layout().count())],
                                         rows=[dict(rect=w.geometry().getRect(), minimum=w.minimumHeight(), hint=w.minimumSizeHint().height(),
                                                    bodies=[dict(rect=b.geometry().getRect(), minimum=b.minimumHeight(), hint=b.minimumSizeHint().height())
                                                            for b in w.findChildren(QtWidgets.QWidget) if b.layout() is not None])
                                               for w in page.findChildren(QtWidgets.QFrame,'workflowRow')],
                                         layouts=[dict(type=type(w).__name__, name=w.objectName(),
                                                       rect=w.geometry().getRect(),
                                                       minimum=(w.minimumWidth(),w.minimumHeight()),
                                                       hint=(w.minimumSizeHint().width(),w.minimumSizeHint().height()))
                                                  for w in (page.plot, page.data_tabs, page.plot_data_splitter,
                                                            page.plot_data_splitter.parentWidget())]))
            (args.render_dir/'render_manifest.json').write_text(
                json.dumps(geometry, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(dict(rendered=len(geometry), hardware_io=False)), flush=True)
            window.close()
            drain()
            return
        qt.exec_()


if __name__ == '__main__':
    main()
