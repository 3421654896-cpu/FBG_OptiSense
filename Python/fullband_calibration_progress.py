"""Read-only live monitor for the 145 mA / 2001-point calibration.

This process only reads the calibration state and log files.  It never opens the
laser serial port or the AQ6150B, so it can safely run beside the calibrator.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# Qt's automatic plugin discovery can lose non-ASCII path characters on this
# Windows installation.  Point it at the venv platform directory explicitly,
# as the main desktop application does, before importing any Qt modules.
qt_platform_plugins = (
    Path(sys.prefix) / "Lib" / "site-packages" / "PyQt5" / "Qt5"
    / "plugins" / "platforms"
)
if qt_platform_plugins.exists():
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(qt_platform_plugins)

from PyQt5 import QtCore, QtGui, QtWidgets


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "outputs" / "fullband_145_auto"
STATE_FILE = "auto_calibration_state.json"
LOG_FILE = "auto_calibration.log"

STAGES = [
    ("inventory_and_model", "旧数据盘点与模型训练"),
    ("mode_map", "145 mA 粗模式图测量"),
    ("select_phase_branches", "选择相位扫描分支"),
    ("phase_map_base", "基础相位映射"),
    ("phase_candidates_base", "构建基础候选点"),
    ("phase_gap_refinement", "波长空档加密"),
    ("phase_alternate_refinement", "备选分支加密"),
    ("dense_2001", "2001 点波长逐点闭环"),
    ("repair_dense", "修复波长未达标点"),
    ("maximize_power", "搜索每点最大单模功率"),
    ("soa_transfer", "测量 SOA—功率传递曲线"),
    ("equalize_common_power", "2001 点同目标功率闭环"),
    ("repair_equal_power", "修复功率超出目标范围的点"),
    ("bidirectional_certification", "所选运行路径 2001 点独立复测"),
    ("preliminary_export", "生成预发布点表"),
    ("feedback_reference", "采集 PDT/PDR 参考"),
    ("final_export", "最终验收、备份与发布"),
]
STAGE_INDEX = {name: index for index, (name, _label) in enumerate(STAGES, 1)}
STAGE_LABEL = dict(STAGES)

PROGRESS_RE = re.compile(
    r"(?:^|\s)(?:map|phase-map|dense-phase|repair|upgrade|soa-pilot|"
    r"equalize|validate|dense)\s+(\d+)/(\d+)\b",
    re.IGNORECASE,
)
REPAIR_PROGRESS_RE = re.compile(
    r"(?:^|\s)(own-seed|neighbor|redistribution|inventory-branch|alternate-branch)\b"
    r".*?\btarget=([0-9.]+)",
    re.IGNORECASE,
)
CONTINUATION_PROGRESS_RE = re.compile(
    r"(?:^|\s)continuation-step\s+(\S+).*?\btarget=([0-9.]+)",
    re.IGNORECASE,
)
METRIC_RE = re.compile(
    r"(?:wave|target)=([0-9.]+).*?power=([0-9.]+)"
    r"(?:.*?SMSR=\s*([0-9.]+))?",
    re.IGNORECASE,
)
REPAIR_METRIC_RE = re.compile(
    r"(?:own-seed|neighbor|redistribution|inventory-branch|alternate-branch)\b"
    r".*?\btarget=([0-9.]+).*?\berr=([+-]?[0-9.]+)pm"
    r".*?\bperr=([+-]?[0-9.]+)%.*?\bok=(True|False)",
    re.IGNORECASE,
)
CONTINUATION_METRIC_RE = re.compile(
    r"continuation-step\s+(\S+).*?\btarget=([0-9.]+)"
    r".*?\bGAIN=([0-9.]+).*?\bSOA=\s*([0-9.]+)"
    r".*?\berr=([+-]?[0-9.]+)pm.*?\bpower=([0-9.]+)mW"
    r".*?\bwave_ok=(True|False)",
    re.IGNORECASE | re.DOTALL,
)
VALIDATION_METRIC_RE = re.compile(
    r"validate\s+(\d+)/(\d+).*?\btarget=([0-9.]+)"
    r".*?\berr=([+-]?[0-9.]+)pm.*?\bperr=([+-]?[0-9.]+)%"
    r".*?\bok=(True|False)",
    re.IGNORECASE,
)

TARGET_START_NM = 1525.0
TARGET_STEP_NM = 0.02
TARGET_COUNT = 2001


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def read_log_tail(path: Path, maximum_bytes: int = 512_000) -> str:
    try:
        with path.open("rb") as handle:
            size = handle.seek(0, 2)
            handle.seek(max(0, size - maximum_bytes))
            payload = handle.read()
    except OSError:
        return ""
    return payload.decode("utf-8", errors="replace")


def current_stage(state: dict) -> tuple[str, int, str]:
    stages = state.get("stages")
    stages = stages if isinstance(stages, dict) else {}
    result = state.get("result")
    if isinstance(result, dict) and result.get("status") == "failed":
        known = [STAGE_INDEX[name] for name in stages if name in STAGE_INDEX]
        index = max(known, default=1)
        return STAGES[index - 1][0], index, "failed"
    for name, details in reversed(list(stages.items())):
        if isinstance(details, dict) and details.get("status") == "running":
            return name, STAGE_INDEX.get(name, 0), "running"
    for name, details in reversed(list(stages.items())):
        if isinstance(details, dict) and details.get("status") == "failed":
            return name, STAGE_INDEX.get(name, 0), "failed"

    completed = [
        STAGE_INDEX[name]
        for name, details in stages.items()
        if name in STAGE_INDEX
        and isinstance(details, dict)
        and details.get("status") == "complete"
    ]
    highest = max(completed, default=0)
    if highest >= len(STAGES):
        return STAGES[-1][0], len(STAGES), "complete"
    return STAGES[highest][0], highest + 1, "waiting"


def latest_progress(log_text: str) -> tuple[int, int, str]:
    # Progress belongs to the most recently launched subcommand.  Limiting the
    # search prevents a quick non-measuring stage from displaying the previous
    # stage's final counter.
    command_block = log_text.rsplit("\n[", 1)[-1]
    for line in reversed(command_block.splitlines()):
        match = PROGRESS_RE.search(line)
        if match:
            return int(match.group(1)), int(match.group(2)), line.strip()
        repair_match = REPAIR_PROGRESS_RE.search(line)
        if repair_match:
            repair_method = repair_match.group(1).lower()
            target_nm = float(repair_match.group(2))
            # Fallback phases revisit sparse failed rows, so their wavelength
            # is not a monotonic progress counter.  Keep the completed first
            # pass at 2001/2001 while still returning the latest repair line.
            if repair_method != "own-seed":
                return TARGET_COUNT, TARGET_COUNT, line.strip()
            current = round((target_nm - TARGET_START_NM) / TARGET_STEP_NM) + 1
            current = max(1, min(TARGET_COUNT, current))
            return current, TARGET_COUNT, line.strip()
        continuation_match = CONTINUATION_PROGRESS_RE.search(line)
        if continuation_match:
            return TARGET_COUNT, TARGET_COUNT, line.strip()
    return 0, 0, ""


def validation_file_progress(path: Path) -> tuple[int, int, str]:
    """Return per-point progress from one atomic validation checkpoint."""

    payload = read_json(path)
    rows = payload.get("rows")
    if not isinstance(rows, dict) or not rows or len(rows) > TARGET_COUNT:
        return 0, 0, ""
    _key, latest = next(reversed(rows.items()))
    if not isinstance(latest, dict):
        return 0, 0, ""
    try:
        target = float(latest["target_nm"])
        wave_error = float(latest["wavelength_error_pm"])
        power_error = 100.0 * float(latest["power_relative_error"])
    except (KeyError, TypeError, ValueError):
        return 0, 0, ""
    passed = bool(latest.get("success"))
    count = len(rows)
    line = (
        f"validate {count}/{TARGET_COUNT} target={target:.6f} "
        f"err={wave_error:+.2f}pm perr={power_error:+.2f}% ok={passed}"
    )
    return count, TARGET_COUNT, line


def parse_utc(value: object) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def duration_text(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}小时{minutes:02d}分"
    if minutes:
        return f"{minutes}分{secs:02d}秒"
    return f"{secs}秒"


class CalibrationProgressWindow(QtWidgets.QWidget):
    def __init__(self, output_dir: Path):
        super().__init__()
        self.output_dir = output_dir.resolve()
        self.state_path = self.output_dir / STATE_FILE
        self.log_path = self.output_dir / LOG_FILE
        self._validation_signature = None
        self._validation_cached = (0, 0, "", 0.0)
        self.setWindowTitle("145 mA·2001点自动校准—实时进度")
        self.resize(760, 480)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(12)

        title = QtWidgets.QLabel("145 mA · 2001点等功率自动校准")
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        root.addWidget(title)

        self.stage_label = QtWidgets.QLabel("正在读取进度…")
        self.stage_label.setStyleSheet("font-size: 17px; font-weight: 600;")
        root.addWidget(self.stage_label)

        self.stage_progress = QtWidgets.QProgressBar()
        self.stage_progress.setMinimumHeight(28)
        root.addWidget(self.stage_progress)

        self.overall_progress = QtWidgets.QProgressBar()
        self.overall_progress.setRange(0, 17000)
        self.overall_progress.setMinimumHeight(24)
        root.addWidget(self.overall_progress)

        self.metrics = QtWidgets.QLabel()
        self.metrics.setWordWrap(True)
        self.metrics.setStyleSheet(
            "QLabel { background: #eef5ff; border: 1px solid #bdd4f3; "
            "border-radius: 6px; padding: 10px; font-size: 15px; }"
        )
        root.addWidget(self.metrics)

        self.acceptance = QtWidgets.QLabel(
            "硬验收：2001点共用同一目标功率；等待读取本轮容差和扫描方向。"
        )
        self.acceptance.setWordWrap(True)
        root.addWidget(self.acceptance)

        self.latest = QtWidgets.QLabel()
        self.latest.setWordWrap(True)
        self.latest.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        root.addWidget(self.latest)

        root.addStretch(1)
        bottom = QtWidgets.QHBoxLayout()
        self.path_label = QtWidgets.QLabel(str(self.output_dir))
        self.path_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.path_label.setStyleSheet("color: #666;")
        open_button = QtWidgets.QPushButton("打开日志目录")
        open_button.clicked.connect(self.open_output_dir)
        bottom.addWidget(self.path_label, 1)
        bottom.addWidget(open_button)
        root.addLayout(bottom)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.refresh()

    def open_output_dir(self) -> None:
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.output_dir)))

    def refresh(self) -> None:
        state = read_json(self.state_path)
        log_text = read_log_tail(self.log_path)
        if not state:
            self.stage_label.setText("等待校准状态文件…")
            self.metrics.setText(f"尚未读取到：{self.state_path}")
            return

        tolerance = state.get("single_point_power_relative_tolerance", 0.005)
        try:
            tolerance_percent = 100.0 * float(tolerance)
        except (TypeError, ValueError):
            tolerance_percent = 0.5
        direction = str(state.get("certification_direction", "forward"))
        direction_text = "固定正向" if direction == "forward" else "正、反向"
        self.acceptance.setText(
            f"硬验收：2001点共用同一目标功率，任一点超出 "
            f"±{tolerance_percent:g}% 即不发布；{direction_text}独立复测必须 "
            "2001/2001 通过。"
        )

        stage_name, stage_index, status = current_stage(state)
        stage_title = STAGE_LABEL.get(stage_name, stage_name)
        current, total, line = latest_progress(log_text)
        # The strict validator intentionally writes to stdout only every forty
        # points.  Its atomic JSON checkpoint is authoritative and advances on
        # every point, so use it while it is newer than the last log event.
        try:
            candidates = list(self.output_dir.glob("validation_*_avg3_postcondition.json"))
            newest = max(candidates, key=lambda path: path.stat().st_mtime_ns)
            stat = newest.stat()
            signature = (str(newest), stat.st_mtime_ns, stat.st_size)
            if signature != self._validation_signature:
                parsed = validation_file_progress(newest)
                if parsed[1]:
                    self._validation_cached = (*parsed, stat.st_mtime)
                    self._validation_signature = signature
            vcurrent, vtotal, vline, vmtime = self._validation_cached
            log_mtime = self.log_path.stat().st_mtime
            if vtotal and vmtime >= log_mtime:
                current, total, line = vcurrent, vtotal, vline
        except (OSError, ValueError):
            pass
        if status == "complete":
            current = total = 1
        if status == "failed":
            self.stage_progress.setRange(0, 1)
            self.stage_progress.setValue(0)
            self.stage_progress.setFormat("本阶段已安全停止，等待续跑")
            fraction = 0.0
        elif total > 0:
            self.stage_progress.setRange(0, total)
            self.stage_progress.setValue(min(current, total))
            self.stage_progress.setFormat(f"本阶段 {current}/{total}  ·  %p%")
            fraction = min(1.0, current / total)
        else:
            self.stage_progress.setRange(0, 0 if status == "running" else 1)
            self.stage_progress.setValue(0)
            self.stage_progress.setFormat("本阶段进度准备中")
            fraction = 0.0

        state_suffix = "（已安全停止）" if status == "failed" else ""
        self.stage_label.setText(f"阶段 {stage_index}/17：{stage_title}{state_suffix}")
        overall = ((max(1, stage_index) - 1) + fraction) / len(STAGES)
        self.overall_progress.setValue(round(overall * 17000))
        self.overall_progress.setFormat(f"流程阶段折算 {overall * 100:.1f}%")

        created = parse_utc(state.get("created_utc"))
        elapsed = (
            duration_text((datetime.now(timezone.utc) - created).total_seconds())
            if created else "未知"
        )
        updated = "未知"
        try:
            age = datetime.now().timestamp() - self.log_path.stat().st_mtime
            updated = f"{max(0, int(age))}秒前"
        except OSError:
            pass
        self.metrics.setText(
            f"已运行：{elapsed}    ·    日志更新：{updated}    ·    "
            f"板卡：{state.get('port', '未知')}    ·    "
            f"波长计：{state.get('meter_resource', '未知')}"
        )

        if status == "failed":
            details = state.get("stages", {}).get(stage_name, {})
            self.latest.setText(
                "停止原因：" + str(details.get("error", "未知错误"))
            )
            return

        validation_match = VALIDATION_METRIC_RE.search(line)
        repair_match = REPAIR_METRIC_RE.search(line)
        continuation_match = CONTINUATION_METRIC_RE.search(line)
        metric_match = METRIC_RE.search(line)
        if validation_match:
            measured, expected, wave, wave_error, power_error, passed = (
                validation_match.groups()
            )
            result = "通过" if passed.lower() == "true" else "待本轮后返修"
            self.latest.setText(
                f"最新复测 {measured}/{expected}：目标 {float(wave):.6f} nm · "
                f"波长误差 {float(wave_error):+.2f} pm · "
                f"功率误差 {float(power_error):+.2f}% · {result}"
            )
        elif repair_match:
            wave, wave_error, power_error, passed = repair_match.groups()
            result = "通过" if passed.lower() == "true" else "待回退修复"
            self.latest.setText(
                f"最新返修：目标 {float(wave):.6f} nm · "
                f"波长误差 {float(wave_error):+.2f} pm · "
                f"功率误差 {float(power_error):+.2f}% · {result}"
            )
        elif continuation_match:
            (_method, wave, gain, soa, wave_error, power, wave_ok) = (
                continuation_match.groups()
            )
            wave_result = "波长有效" if wave_ok.lower() == "true" else "跳模/越界"
            self.latest.setText(
                f"电流延拓：目标 {float(wave):.6f} nm · "
                f"GAIN {float(gain):.3f} mA · SOA {float(soa):.3f} mA · "
                f"波长误差 {float(wave_error):+.2f} pm · "
                f"功率 {float(power):.6f} mW · {wave_result}"
            )
        elif metric_match:
            wave, power, smsr = metric_match.groups()
            detail = f"最新测量：波长 {float(wave):.6f} nm · 功率 {float(power):.4f} mW"
            if smsr is not None:
                detail += f" · SMSR {float(smsr):.1f} dB"
            self.latest.setText(detail)
        else:
            self.latest.setText(f"最新记录：{line or '等待新测量…'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("2001点自动校准进度")
    window = CalibrationProgressWindow(args.output_dir)

    def show_on_primary_screen() -> None:
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            frame = window.frameGeometry()
            frame.moveCenter(screen.availableGeometry().center())
            window.move(frame.topLeft())
        window.showNormal()
        window.raise_()
        window.activateWindow()

    show_on_primary_screen()
    QtCore.QTimer.singleShot(200, show_on_primary_screen)
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
