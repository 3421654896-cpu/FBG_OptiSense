"""Customer offline updater: signed client package and UID-bound USB firmware."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

from PyQt5 import QtCore, QtWidgets
from serial.tools import list_ports
from fbg_ota_protocol import load_package
from fbg_usb_ota import UsbOtaClient
from hyperos_theme import install_theme
from offline_update_engine import (UpdateError, load_manifest, checked_payload,
                                  firmware_for_serial, replace_client)
from update_public_key import PUBLIC_KEY_HEX


class UpdateWorker(QtCore.QThread):
    message = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int)
    result = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, root, manifest, kind, target, parent=None):
        super().__init__(parent)
        self.root, self.manifest, self.kind, self.target = root, manifest, kind, target

    def run(self):
        try:
            # Re-check the signed manifest and hash just before use.
            manifest = load_manifest(self.root/'update_manifest.json', PUBLIC_KEY_HEX)
            if manifest != self.manifest:
                raise UpdateError('更新清单在打开后发生变化')
            if self.kind == 'client':
                item = manifest['client']
                payload = checked_payload(self.root, item)
                result = replace_client(payload, self.target, item['version'], progress=self.message.emit)
            else:
                port, serial = self.target
                item = firmware_for_serial(manifest, serial)
                package = load_package(checked_payload(self.root, item))
                if package.version_text != item['version']:
                    raise UpdateError('签名清单与固件版本不符')
                with UsbOtaClient(port, serial) as client:
                    before = client.query_info()
                    if before.firmware_version >= package.firmware_version:
                        result = dict(status='latest', version=before.version_text, serial=serial)
                    else:
                        def notify(event):
                            self.message.emit(event.message)
                            self.progress.emit(int(100*event.sent/max(1,event.total)))
                        after = client.install(package, progress=notify)
                        result = dict(status='updated', version=after.version_text,
                                      serial=serial, runtime=asdict(after),
                                      reboot_acknowledged=client.reboot_acknowledged)
            self.progress.emit(100)
            self.result.emit(result)
        except Exception as exc:
            self.failed.emit(f'{type(exc).__name__}：{exc}')


class OfflineUpdater(QtWidgets.QWidget):
    def __init__(self, root, install_dir=None):
        super().__init__()
        self.root = Path(root).resolve()
        self.manifest = None
        self.worker = None
        self.setWindowTitle('OptiSense 一键更新助手')
        self.resize(660, 480)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24,24,24,24); layout.setSpacing(14)
        title=QtWidgets.QLabel('软件与固件更新'); title.setObjectName('pageTitle')
        layout.addWidget(title)
        note=QtWidgets.QLabel('更新前请保存记录并关闭客户端；固件升级期间保持设备供电。\n'
                             '两个更新相互独立，不需要 Python、网络账号或授权密码。')
        note.setWordWrap(True); note.setObjectName('pageSubtitle'); layout.addWidget(note)
        client_box=QtWidgets.QGroupBox('客户端软件')
        client_layout=QtWidgets.QVBoxLayout(client_box)
        self.install_dir=QtWidgets.QLineEdit(str(install_dir or
            Path(os.environ.get('LOCALAPPDATA',str(Path.home())))/'Programs'/'FBG OptiSense Studio'))
        folder=QtWidgets.QPushButton('选择安装目录'); folder.clicked.connect(self.choose_folder)
        row=QtWidgets.QHBoxLayout(); row.addWidget(self.install_dir,1); row.addWidget(folder)
        client_layout.addLayout(row)
        self.client_button=QtWidgets.QPushButton('一键更新软件'); self.client_button.setProperty('role','primary')
        self.client_button.clicked.connect(lambda:self.start_update('client'))
        client_layout.addWidget(self.client_button,0,QtCore.Qt.AlignLeft); layout.addWidget(client_box)
        fw_box=QtWidgets.QGroupBox('解调仪固件 · USB')
        fw_layout=QtWidgets.QHBoxLayout(fw_box)
        self.ports=QtWidgets.QComboBox(); self.ports.setMinimumContentsLength(25)
        refresh=QtWidgets.QPushButton('刷新设备'); refresh.clicked.connect(self.refresh_ports)
        self.firmware_button=QtWidgets.QPushButton('一键更新固件'); self.firmware_button.setProperty('role','primary')
        self.firmware_button.clicked.connect(lambda:self.start_update('firmware'))
        fw_layout.addWidget(self.ports,1); fw_layout.addWidget(refresh); fw_layout.addWidget(self.firmware_button)
        layout.addWidget(fw_box)
        self.bar=QtWidgets.QProgressBar(); self.bar.setRange(0,100); self.bar.setValue(0); layout.addWidget(self.bar)
        self.status=QtWidgets.QLabel('正在验证更新包'); self.status.setWordWrap(True); layout.addWidget(self.status)
        layout.addStretch()
        try:
            self.manifest=load_manifest(self.root/'update_manifest.json',PUBLIC_KEY_HEX)
            self.status.setText(f"更新包 v{self.manifest['release']} · 签名已验证")
        except Exception as exc:
            self.status.setText(str(exc)); self.client_button.setEnabled(False)
        self.refresh_ports()

    def choose_folder(self):
        folder=QtWidgets.QFileDialog.getExistingDirectory(self,'选择客户端安装目录',self.install_dir.text())
        if folder:self.install_dir.setText(folder)

    def refresh_ports(self):
        self.ports.clear()
        for port in list_ports.comports():
            if port.vid==0x0483 and port.pid==0x5740:
                self.ports.addItem(f'{port.device} · {port.serial_number}',(port.device,port.serial_number))
        self.firmware_button.setEnabled(bool(self.manifest and self.ports.count()))

    def start_update(self,kind):
        if self.worker is not None or self.manifest is None:return
        target=self.install_dir.text().strip() if kind=='client' else self.ports.currentData()
        if kind=='firmware':
            try:firmware_for_serial(self.manifest,target[1])
            except Exception as exc:self.status.setText(str(exc));return
        self.bar.setValue(0)
        self.client_button.setEnabled(False);self.firmware_button.setEnabled(False)
        self.install_dir.setEnabled(False);self.ports.setEnabled(False)
        worker=UpdateWorker(self.root,self.manifest,kind,target,self)
        self.worker=worker
        worker.message.connect(self.status.setText);worker.progress.connect(self.bar.setValue)
        worker.result.connect(self.completed);worker.failed.connect(self.status.setText)
        worker.finished.connect(self.finished);worker.start()

    def completed(self,result):
        text='已是最新版本' if result['status']=='latest' else '更新完成'
        self.status.setText(f"{text} · v{result['version']}"+
                           (f"\n原软件已保留在：{result['backup']}" if 'backup' in result else ''))

    def finished(self):
        self.worker.deleteLater();self.worker=None
        self.client_button.setEnabled(self.manifest is not None);self.install_dir.setEnabled(True)
        self.ports.setEnabled(True);self.refresh_ports()

    def closeEvent(self,event):
        if self.worker is not None:
            self.status.setText('更新正在进行，请保持供电并等待完成');event.ignore()
        else:event.accept()


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle-dir',type=Path)
    parser.add_argument('--install-dir',type=Path)
    parser.add_argument('--smoke-seconds',type=float,default=0)
    parser.add_argument('--screenshot',type=Path)
    parser.add_argument('--apply-client-update',action='store_true',
                        help='Run the signed software update without UI (requires explicit install directory and fresh result file)')
    parser.add_argument('--result-file',type=Path)
    args=parser.parse_args(argv)
    if args.apply_client_update:
        if args.install_dir is None or args.result_file is None:
            parser.error('Noninteractive software update requires --install-dir and --result-file')
        if args.result_file.exists():
            parser.error('Result file already exists; preserving previous result')
    root=args.bundle_dir or (Path(sys.executable).parent if getattr(sys,'frozen',False) else Path(__file__).parent)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    app=QtWidgets.QApplication([]);install_theme(app,configure_plots=False)
    if args.apply_client_update:
        results, failures, messages = [], [], []
        try:
            manifest=load_manifest(root/'update_manifest.json',PUBLIC_KEY_HEX)
            worker=UpdateWorker(root,manifest,'client',str(args.install_dir))
            worker.result.connect(results.append);worker.failed.connect(failures.append)
            worker.message.connect(messages.append);worker.run()
        except Exception as exc:
            failures.append(f'{type(exc).__name__}: {exc}')
        report=dict(results=results,errors=failures,messages=messages,
                    firmware_commands_sent=False)
        args.result_file.parent.mkdir(parents=True,exist_ok=True)
        with args.result_file.open('x',encoding='utf-8') as stream:
            json.dump(report,stream,ensure_ascii=False,indent=2)
        return 0 if results and not failures else 1
    window=OfflineUpdater(root,args.install_dir);window.show()
    if args.screenshot:
        QtCore.QTimer.singleShot(1000,lambda:window.grab().save(str(args.screenshot)))
    if args.smoke_seconds:
        QtCore.QTimer.singleShot(int(args.smoke_seconds*1000),window.close)
    return app.exec_()


if __name__=='__main__':
    raise SystemExit(main())
