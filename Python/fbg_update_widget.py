"""Small customer-facing desktop update page and background Release workers."""

from __future__ import annotations

from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from app_version import APP_VERSION
from github_update_feed import (
    ReleaseAsset,
    download_release_asset,
    fetch_update_offers,
    version_parts,
)


class ReleaseCheckWorker(QtCore.QThread):
    completed = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def run(self):
        try:
            self.completed.emit(fetch_update_offers())
        except Exception as exc:
            self.failed.emit(str(exc))


class AssetDownloadWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int)
    completed = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, asset: ReleaseAsset, parent=None):
        super().__init__(parent)
        self.asset = asset

    def run(self):
        try:
            path = download_release_asset(
                self.asset,
                cancelled=self.isInterruptionRequested,
                progress=lambda sent, total: self.progress.emit(
                    int(sent * 100 / max(total, 1))
                ),
            )
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.completed.emit(str(path))


class SoftwareUpdatePage(QtWidgets.QWidget):
    """No firmware controls here: only an independently versioned app installer."""

    install_ready = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.offer: ReleaseAsset | None = None
        self._download: AssetDownloadWorker | None = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        title = QtWidgets.QLabel("客户端软件更新")
        title.setObjectName("brandTitle")
        layout.addWidget(title)
        self.version_label = QtWidgets.QLabel(f"当前客户端 v{APP_VERSION}")
        self.version_label.setObjectName("brandSubtitle")
        layout.addWidget(self.version_label)
        self.status = QtWidgets.QLabel("正在等待 GitHub Releases 检查结果…")
        self.status.setWordWrap(True)
        self.status.setObjectName("softHint")
        layout.addWidget(self.status)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.hide()
        layout.addWidget(self.progress)
        self.update_button = QtWidgets.QPushButton("一键更新客户端")
        self.update_button.setProperty("role", "primary")
        self.update_button.setEnabled(False)
        self.update_button.clicked.connect(self._start_download)
        layout.addWidget(self.update_button, 0, QtCore.Qt.AlignLeft)
        self.update_button.setMinimumWidth(150)
        layout.addStretch(1)

    def set_offer(self, offer: ReleaseAsset | None) -> None:
        self.offer = (
            offer if offer is not None
            and offer.kind == "desktop"
            and version_parts(offer.version) > version_parts(APP_VERSION)
            else None
        )
        if self.offer is None:
            self.status.setText("已是最新版本。")
        else:
            self.status.setText(
                f"发现客户端 v{self.offer.version}。点击后将下载并校验安装包，"
                "关闭本程序后原位更新；你的机号与采集记录会保留。"
            )
        self.update_button.setEnabled(
            self.offer is not None and not self.download_busy
        )

    @property
    def download_busy(self) -> bool:
        return bool(self._download is not None and self._download.isRunning())

    def _start_download(self):
        if self.offer is None or self.download_busy:
            return
        self.update_button.setEnabled(False)
        self.progress.show()
        self.progress.setValue(0)
        self.status.setText("正在从 GitHub 下载客户端安装包…")
        worker = AssetDownloadWorker(self.offer, self)
        self._download = worker
        worker.progress.connect(self.progress.setValue)
        worker.completed.connect(self._downloaded)
        worker.failed.connect(self._download_failed)
        worker.finished.connect(self._download_finished)
        worker.start()

    def _downloaded(self, path: str):
        if Path(path).suffix.lower() != ".exe":
            self._download_failed("下载文件不是安装程序")
            return
        self.progress.setValue(100)
        self.status.setText("安装包校验通过，准备安装客户端更新…")
        self.install_ready.emit(path)

    def _download_failed(self, message: str):
        self.status.setText(f"客户端更新未开始：{message}")

    def _download_finished(self):
        self.update_button.setEnabled(self.offer is not None)

    def cancel_download(self):
        if self._download is not None and self._download.isRunning():
            self._download.requestInterruption()
            self._download.wait(35000)


__all__ = ["AssetDownloadWorker", "ReleaseCheckWorker", "SoftwareUpdatePage"]
