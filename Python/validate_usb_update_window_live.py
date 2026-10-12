"""Exercise the actual USB dialog worker offscreen against one confirmed device."""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_OPENGL"] = "software"
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(
    Path(sys.prefix) / "Lib/site-packages/PyQt5/Qt5/plugins/platforms")

from PyQt5 import QtGui, QtWidgets
from fbg_ota_protocol import load_package
from fbg_usb_ota_widget import UsbFirmwareUpdateDialog
from hyperos_theme import install_theme


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--device-serial", required=True)
    parser.add_argument("--machine-label", required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Report already exists")
    app = QtWidgets.QApplication([])
    QtGui.QFontDatabase.addApplicationFont("C:/Windows/Fonts/msyh.ttc")
    install_theme(app)
    dialog = UsbFirmwareUpdateDialog(
        machine_label=args.machine_label, package=load_package(args.package),
        before_update=lambda: True)
    index = next((i for i in range(dialog.ports.count())
                  if dialog.ports.itemData(i) == (args.port, args.device_serial)), -1)
    if index < 0:
        raise RuntimeError("Confirmed native USB device is absent; no update sent")
    dialog.ports.setCurrentIndex(index)
    result = dict(device_serial=args.device_serial, port=args.port,
                  window_test=True, verified=False)
    started = time.monotonic()

    def installed(package):
        result.update(verified=True, version=package.version_text,
                      progress=dialog.progress.value(), message=dialog.message.text())

    def failed(message):
        result["error"] = message

    def finished():
        result["elapsed_s"] = round(time.monotonic() - started, 3)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        dialog.grab().save(str(args.output.with_suffix(".png")))
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False), flush=True)
        dialog.close()
        app.quit()

    dialog.firmware_installed.connect(installed)
    dialog.show()
    with patch.object(QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Yes):
        dialog.start_update()
    if dialog.worker is None:
        raise RuntimeError("The update worker did not start")
    dialog.worker.failed.connect(failed)
    dialog.worker.finished.connect(finished)
    app.exec_()
    if not result["verified"]:
        raise RuntimeError(result.get("error", "Window test failed"))


if __name__ == "__main__":
    main()
