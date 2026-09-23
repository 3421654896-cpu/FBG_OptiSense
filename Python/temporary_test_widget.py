"""Independent temporary spectrum page; no production table installation."""
from pathlib import Path
import csv
import json
import re
import threading
import time

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from machine_profile import (
    ensure_machine_compatible,
    get_runtime_machine_id,
    runtime_machine_label,
    runtime_parameter_note,
    stamp_machine_metadata,
)
from temporary_test_mode import (
    assess_five_point_shape,
    build_manual_dense_plan,
    build_selected_plan,
    command,
    execute_device,
    load_calibrated_plan,
    load_saved_plan,
    select_points,
    validate_plan_payload,
)
from temporary_optical_units import (
    PD_RESPONSIVITY_A_PER_W,
    adc_code_to_dbm,
    adc_code_to_voltage,
    format_physical_value,
    transimpedance_ohm,
)
from runtime_paths import output_path, resource_path


LAST_SESSION_PATH = (
    output_path('temporary_test', 'last_45_session.json')
)
FINGER_CAPTURE_DIR = (
    output_path('temporary_test', 'finger_captures')
)
LAST_FINGER_CAPTURE_PATH = (
    output_path('temporary_test', 'last_finger_capture.json')
)
_LEGACY_LAST_SESSION_PATH = LAST_SESSION_PATH
_LEGACY_FINGER_CAPTURE_DIR = FINGER_CAPTURE_DIR
_LEGACY_LAST_FINGER_CAPTURE_PATH = LAST_FINGER_CAPTURE_PATH
DEFAULT_DISPLAY_MODE = 'dbm'


def temporary_transport_supported(device):
    """Return whether temporary v3/v4/v5 packets can use this transport."""

    import serial
    from fbg_lan_serial import FbgLanSerial
    return isinstance(device, (serial.Serial, FbgLanSerial))

# The board-switch experiment is diagnostic data only.  Keep it separate from
# the active route so a manually edited 45-point route never gets overwritten
# by an older capture.  The point-wait editor uses this table as a read-only
# hint for the forward jump that precedes each point.
FORWARD_SETTLE_ANALYSIS_PATH = (
    output_path('wavelength_switch_board_analysis_45x3_20260917.json')
)
FORWARD_SETTLE_CAPTURE_PATH = (
    output_path('wavelength_switch_board_capture_45x3_20260917.json')
)
_FORWARD_SETTLE_HINTS = None


def load_forward_settle_hints():
    """Load measured CH1 settle times keyed by physical forward transitions.

    The analysis file addresses transitions by their position in the captured
    route.  The companion capture file supplies the physical full-band index,
    so the UI can match an arbitrary (including keyboard-edited) current route
    without assuming that it is the route used during the experiment.
    """
    global _FORWARD_SETTLE_HINTS
    if _FORWARD_SETTLE_HINTS is not None:
        return _FORWARD_SETTLE_HINTS
    hints = {'pairs': {}, 'targets': {}, 'source_label': ''}

    def add_item(source_index, target_index, metric, *, category='',
                 source_name='', source_rank=0):
        if not isinstance(metric, dict):
            return
        item = {
            'source_index': int(source_index),
            'target_index': int(target_index),
            'category': str(category),
            'settle_us': metric.get('settle_5pct_us'),
            'settle_available': 'settle_5pct_us' in metric,
            'settle_from_switch_begin_us':
                metric.get('settle_5pct_from_switch_begin_us'),
            'edge_us': metric.get('main_edge_10_90_us'),
            'edge_complete_from_switch_begin_us':
                metric.get('main_edge_complete_from_switch_begin_us'),
            'meaningful': bool(metric.get('meaningful_step', False)),
            'classification': str(metric.get('classification', '')),
            'scope_us': 100000,
            'source_name': source_name,
            'source_rank': source_rank,
        }
        key = (item['source_index'], item['target_index'])
        previous = hints['pairs'].get(key)
        if previous is None or source_rank >= previous['source_rank']:
            hints['pairs'][key] = item
        hints['targets'].setdefault(item['target_index'], []).append(item)

    try:
        with FORWARD_SETTLE_ANALYSIS_PATH.open('r', encoding='utf-8') as fh:
            analysis = json.load(fh)
        with FORWARD_SETTLE_CAPTURE_PATH.open('r', encoding='utf-8') as fh:
            capture = json.load(fh)
        route = sorted(capture.get('dac_rows', []),
                       key=lambda row: int(row.get('order', 0)))
        physical = [int(row['fullband_index']) for row in route]
        transitions = analysis.get('transitions', [])
        base_rank = FORWARD_SETTLE_ANALYSIS_PATH.stat().st_mtime_ns
        for transition in transitions:
            source_pos = int(transition['source_index'])
            target_pos = int(transition['target_index'])
            if not (0 <= source_pos < len(physical)
                    and 0 <= target_pos < len(physical)):
                continue
            metric = (transition.get('channels', {})
                      .get('CH1', {}))
            add_item(physical[source_pos], physical[target_pos], metric,
                     category=transition.get('category', ''),
                     source_name=FORWARD_SETTLE_ANALYSIS_PATH.name,
                     source_rank=base_rank)

        # Later forward-route audits already contain physical indices and are
        # more relevant to the manually selected half-height routes.  They are
        # read-only evidence; loading them does not install their point tables.
        output_dir = output_path()
        audit_paths = [
            output_dir / 'temporary_test_half_height_2ms_audit_forward_v1075_20260917.json'
        ]
        audit_paths.extend(output_dir.glob(
            'temporary_fast_route_search*_final_attempt*.json'
        ))
        loaded_sources = 1
        for path in sorted({p for p in audit_paths if p.is_file()},
                           key=lambda p: p.stat().st_mtime_ns):
            try:
                with path.open('r', encoding='utf-8') as fh:
                    payload = json.load(fh)
                source_rank = path.stat().st_mtime_ns
                for transition in payload.get('analysis', {}).get('transitions', []):
                    source_index = transition.get(
                        'source_index', transition.get('source_fullband_index'))
                    target_index = transition.get(
                        'target_index', transition.get('target_fullband_index'))
                    metric = transition.get('aggregate')
                    if metric is None:
                        metric = transition.get('channels', {}).get('CH1')
                    if source_index is None or target_index is None:
                        continue
                    add_item(source_index, target_index, metric,
                             category=transition.get(
                                 'transition_kind', transition.get('category', '')),
                             source_name=path.name, source_rank=source_rank)
                loaded_sources += 1
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                continue
        hints['source_label'] = (
            f'已载入{loaded_sources}份板上正向跳转实测'
            '（CH1；显示目标写入后的5%稳态时间）'
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        # Missing diagnostic files must never prevent a temporary scan.
        hints = {'pairs': {}, 'targets': {}, 'source_label': '暂无板上正向跳转实测'}
    _FORWARD_SETTLE_HINTS = hints
    return hints


def forward_settle_hint_for(rows, position):
    """Return the diagnostic hint for the forward transition into *position*."""
    hints = load_forward_settle_hints()
    if not rows or not (0 <= position < len(rows)):
        return None
    source = int(rows[position - 1]['index']) if position else int(rows[-1]['index'])
    target = int(rows[position]['index'])
    exact = hints['pairs'].get((source, target))
    if exact is not None:
        result = dict(exact)
        result['match'] = 'exact'
        return result
    candidates = hints['targets'].get(target, [])
    if candidates:
        # Prefer the closest measured source point, then the newest audit.
        result = dict(min(
            candidates,
            key=lambda item: (
                abs(item['source_index'] - source),
                -item.get('source_rank', 0),
            ),
        ))
        result['match'] = 'target_only'
        return result
    return None


class KeyboardSelectionTarget(pg.TargetItem):
    """A fixed marker that can be selected but cannot be dragged."""

    selected = QtCore.pyqtSignal(object)

    def mouseClickEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            event.accept()
            self.selected.emit(self)
            return
        super().mouseClickEvent(event)

    def hoverEvent(self, event):
        if not event.isExit() and event.acceptClicks(QtCore.Qt.LeftButton):
            self.setMouseHover(True)
        else:
            self.setMouseHover(False)


class TemporaryWorker(QtCore.QThread):
    frame = QtCore.pyqtSignal(object)
    message = QtCore.pyqtSignal(str)

    def __init__(self, device, plan, output, cycles, parent=None, *, first_delay_us=350,
                 boundary_extra_us=1800, feedback_selector=2, point_delays_us=None,
                 adc_channel=1, reference_dense_indices=None,
                 reference_acquisition_method='stable_window',
                 reference_settle_s=.2):
        super().__init__(parent)
        self.device, self.plan, self.output, self.cycles = device, plan, output, cycles
        self.cancel = threading.Event()
        self.options = dict(first_delay_us=first_delay_us, boundary_extra_us=boundary_extra_us,
                            feedback_selector=feedback_selector)
        self.adc_channel = int(adc_channel)
        self.reference_dense_indices = (
            list(reference_dense_indices)
            if reference_dense_indices is not None else None
        )
        self.reference_acquisition_method = str(reference_acquisition_method)
        self.reference_settle_s = float(reference_settle_s)
        if point_delays_us is not None:
            self.options['point_delays_us'] = list(point_delays_us)

    def stop(self):
        self.cancel.set()

    def run(self):
        try:
            report = execute_device(self.device, self.plan, self.output, cycles=self.cycles,
                                    adc_channel=self.adc_channel,
                                    **self.options,
                                    on_frame=self.frame.emit, should_stop=self.cancel.is_set)
            self.message.emit(f"完成：{report['mean_hz']:.2f} Hz，P99 {report['p99_period_ms']:.2f} ms；"
                              f"允许低于15 Hz；光学精度待对照；已停光。记录：{self.output}")
        except Exception as exc:
            self.message.emit(f"{type(exc).__name__}: {exc}；记录：{self.output}")


class ReferenceWorker(TemporaryWorker):
    reference = QtCore.pyqtSignal(object)
    reference_point = QtCore.pyqtSignal(object)

    def run(self):
        from temporary_reference import acquire_reference
        try:
            report = acquire_reference(self.device, self.output, should_stop=self.cancel.is_set,
                feedback_selector=self.options['feedback_selector'],
                signal_channel=self.adc_channel,
                dense_indices=self.reference_dense_indices,
                acquisition_method=self.reference_acquisition_method,
                settle_s=self.reference_settle_s,
                continue_on_unstable=True, allow_partial_stop=True,
                on_row=lambda row, n, total: self.reference_point.emit({
                    'row': row, 'completed': n, 'total': total,
                }),
                on_progress=lambda n, total: self.message.emit(
                    f"采集密集参考 {n}/{total}；边采边画，未通过点标红后继续"))
            self.reference.emit(report)
        except Exception as exc:
            self.message.emit(f"参考采集失败：{exc}；记录：{self.output}")


class ReferenceLineWorker(ReferenceWorker):
    def run(self):
        from temporary_reference import acquire_reference
        try:
            report = acquire_reference(self.device, self.output, selected_rows=self.plan['rows'],
                feedback_selector=self.options['feedback_selector'],
                signal_channel=self.adc_channel, should_stop=self.cancel.is_set,
                on_progress=lambda n,total: self.message.emit(f'采集慢速参考线 {n}/{total}；每点等待并检查多次读数，完成后停光'))
            self.reference.emit(report)
        except Exception as exc:
            self.message.emit(f'参考线采集失败：{exc}；记录：{self.output}')


class SwitchTimingWorker(TemporaryWorker):
    """Measure the forward CH1 90% arrival time for the active route."""

    timing = QtCore.pyqtSignal(object)

    def run(self):
        import app_JDSU as app
        from benchmark_stress_realtime import _safe_disarm, _close_shutter
        from capture_ch1_teacher_spectrum import save_checkpoint
        from capture_temporary_half_height_transients import (
            capture_custom,
            transition_analysis,
        )

        route = list(self.plan.get('rows', []))
        repeats = 3
        report = {
            'schema': 'temporary_route_forward_90pct_timing_v1',
            'created_s': time.time(),
            'route': route,
            'repeats': repeats,
            'direction': 'strictly forward, small wavelength to large wavelength',
            'reverse_wrap_tested': False,
            'captures': [],
            'complete': False,
            'ch1_feedback_selector': int(self.options['feedback_selector']),
        }
        old_timeout = self.device.timeout
        try:
            if len(route) not in range(5, 46, 5):
                raise ValueError('开关测试需要当前15/45点路线')
            self.device.timeout = .1
            if not _safe_disarm(self.device) or not _close_shutter(self.device):
                raise RuntimeError('开关测试前停光未确认')
            self.device.write(app.build_work_mode_command(2))
            time.sleep(.12)
            self.device.reset_input_buffer()
            worker = app.EqualIntervalWorker(
                self.device, (), {}, settle_s=.3,
                feedback_selectors=(0, int(self.options['feedback_selector'])),
            )
            if not worker._feedback_command_and_ack():
                raise RuntimeError('CH1模拟档位设置未确认')
            tag = int(time.time_ns() & 0x7fffffff) or 1
            total = repeats * (len(route) - 1)
            for repeat in range(repeats):
                for target_order in range(1, len(route)):
                    if self.cancel.is_set():
                        raise InterruptedError('开关速度测试已取消')
                    tag = (tag + 1) & 0xffffffff or 1
                    source = route[target_order - 1]
                    target = route[target_order]
                    item = capture_custom(self.device, source, target, tag)
                    item.update({
                        'repeat': repeat,
                        'route_source_order': target_order - 1,
                        'route_target_order': target_order,
                        'source_fullband_index': int(source['index']),
                        'target_fullband_index': int(target['index']),
                        'request_tag': tag,
                    })
                    report['captures'].append(item)
                    done = len(report['captures'])
                    self.message.emit(
                        f'开关速度测试 {done}/{total}：'
                        f'第{repeat + 1}/3轮，正向进入第{target_order + 1}点'
                    )
                    if done % 10 == 0:
                        save_checkpoint(self.output, report)
            report['analysis'] = transition_analysis(
                report['captures'], route, repeats
            )
            report['complete'] = True
            self.timing.emit(report)
        except Exception as exc:
            report['error'] = str(exc)
            self.message.emit(f'开关速度测试未完成：{exc}；记录：{self.output}')
        finally:
            errors = []
            for name, action in (
                    ('exact_disarm_ack', _safe_disarm),
                    ('exact_soa_shutter_ack', _close_shutter)):
                try:
                    report[name] = bool(action(self.device))
                except Exception as exc:
                    report[name] = False
                    errors.append(str(exc))
            report['cleanup_errors'] = errors
            self.device.timeout = old_timeout
            save_checkpoint(self.output, report)
            if not report.get('exact_disarm_ack') or not report.get('exact_soa_shutter_ack'):
                self.message.emit('开关测试后停光未完整确认，请立即检查板卡')


class TemporaryTestWindow(QtWidgets.QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.persistence_enabled = controller is not None
        self._restoring_session = False
        self.worker = None
        self.plan = None
        self.plan_ready = False
        self.demo = False
        self.frame_times = []
        self.latest_values = None
        self.fit_pending = True
        self.reference_values = None
        self.reference_wavelengths = None
        self.dense_wavelengths = None
        self.dense_values = None
        self.dense_source = None
        self.dense_failed_points = []
        self._live_dense_rows = []
        self._dense_capture_active = False
        self.finger_record_name = None
        self.display_mode = DEFAULT_DISPLAY_MODE
        # Convert every data set with the feedback selector that produced it.
        # Changing the selector in the UI only affects a future acquisition.
        self.latest_feedback_selector = 2
        self.reference_feedback_selector = 2
        self.dense_feedback_selector = 2
        self.latest_channel = 1
        self.reference_channel = 1
        self.dense_channel = 1
        self.dense_expected_points = 2001
        self.dense_acquisition_method = 'stable_window'
        # A rolling session protects against accidental page changes or an
        # abnormal exit, while this flag tracks whether the operator has made
        # a deliberate named save of the currently displayed spectrum.
        self._spectrum_dirty = False
        self.selection_markers = []
        self.selected_selection_point = None
        self._reference_line_prior_ready = False
        self._updating_selection_markers = False
        self.fit_failure_streaks = [0] * 9
        self.selection_baseline_fraction = None
        self.selection_soft_baseline = False
        # A freshly selected dense route can enter a different laser branch
        # when its 45 points are replayed sparsely.  Keep the most recent
        # independently observed 9/9 sparse route as a per-peak fallback.  A
        # fallback only changes indices; DAC rows are always rebuilt from the
        # currently active 2001-point table and must pass another live slow
        # reference before realtime scanning is enabled.
        self.validated_route_indices = None
        self.validated_route_values = None
        self.validated_route_source = None
        self.runtime_machine_label = runtime_machine_label()
        self.runtime_parameter_note = runtime_parameter_note()
        self._set_machine_storage_paths(get_runtime_machine_id())
        layout = QtWidgets.QVBoxLayout(self)
        self.title_label = QtWidgets.QLabel(
            f"临时测试模式 · {self.runtime_machine_label} · CH1 · 9峰×5点 · 仅扫描选定的45点"
        )
        layout.addWidget(self.title_label)
        self.machine_profile_label = QtWidgets.QLabel(
            f"当前设备：{self.runtime_machine_label}｜{self.runtime_parameter_note}"
        )
        self.machine_profile_label.setStyleSheet(
            'color: #b45309; font-weight: 600;'
        )
        self.machine_profile_label.setWordWrap(True)
        layout.addWidget(self.machine_profile_label)
        self.note_label = QtWidgets.QLabel("已有45点表可直接扫描或采集参考线；只有重新选点才需重采密集光谱。实线为实时值，虚线为慢速参考。\n"
                               "自动选点时五点覆盖约两倍半高宽，尽量包含峰的两侧；只显示实测点，各峰之间不补线。等待自行调节，允许低于15 Hz。\n"
                               "临时表仅在RAM中，使用USB；默认连续循环，点击停止才结束；异常保护仍会停光。\n"
                               "当前实物幅值对照未通过。扫描中45点不跟随峰移动，不归一化、不平滑、不剔除幅值变化。\n"
                               "两个ADC读数接近不代表波长已稳定；等待时间需实测验证。修改模拟放大后需重新做幅值对照。")
        self.note_label.setWordWrap(True)
        self.note_label.setToolTip(self.note_label.text())
        self.note_label.setText('固定45点扫描：原始圆点/实线、参考虚线、拟合点线分开显示；等待自行调节，允许低于15 Hz。\n'
                     '拟合波长基于旧标定坐标，仅为五点估计；双ADC接近不证明波长稳定，原始幅值不作修正。')
        layout.addWidget(self.note_label)
        controls = QtWidgets.QHBoxLayout()
        self.channel_combo = QtWidgets.QComboBox()
        for channel in range(4):
            self.channel_combo.addItem(f'通道 CH{channel}', channel)
        self.channel_combo.setCurrentIndex(1)
        self.channel_combo.setToolTip(
            '选择密集光谱、自动选点、参考线、实时扫描与拟合所使用的ADC通道。'
            'CH0/CH1可调模拟跨阻；CH2/CH3固定2 kΩ。'
        )
        self.reference_button = QtWidgets.QPushButton("重采密集光谱并选点")
        self.equal_interval_reference_button = QtWidgets.QPushButton(
            '按等间隔重采光谱并选点'
        )
        self.equal_interval_nm = QtWidgets.QDoubleSpinBox()
        self.equal_interval_nm.setDecimals(2)
        self.equal_interval_nm.setRange(.02, .10)
        self.equal_interval_nm.setSingleStep(.02)
        self.equal_interval_nm.setValue(.02)
        self.equal_interval_nm.setSuffix(' nm')
        self.equal_interval_nm.setToolTip(
            '按2001点标定表每N格取一点；允许0.02～0.10 nm。'
            '间隔越大采集越快，但窄峰可用采样点更少。'
        )
        self.equal_interval_settle = QtWidgets.QDoubleSpinBox()
        self.equal_interval_settle.setDecimals(3)
        self.equal_interval_settle.setRange(0.0, 10.0)
        self.equal_interval_settle.setSingleStep(.010)
        self.equal_interval_settle.setValue(.200)
        self.equal_interval_settle.setSuffix(' s/点')
        self.equal_interval_settle.setToolTip(
            '与等间隔模式一致：DAC回读确认后等待该时间，清空旧帧，再取第一帧ADC。'
        )
        self.stop_dense_button = QtWidgets.QPushButton("停止绘制")
        self.stop_dense_button.setEnabled(False)
        self.stop_dense_button.setToolTip(
            '安全停止当前密集光谱采集并关光；如已显示足够的9个峰，'
            '已采部分仍会自动选点并可继续参考线/开关测试。'
        )
        self.save_finger_button = QtWidgets.QPushButton('保存本次手指记录')
        self.load_finger_button = QtWidgets.QPushButton('载入手指记录')
        self.save_finger_button.setEnabled(False)
        self.save_finger_button.setToolTip(
            '保存当前手指的密集谱线、15/45点路线、参考线和等待设置；'
            '下次可直接载入并开始临时测试。'
        )
        self.load_finger_button.setToolTip(
            '选择以前保存的手指记录；载入后会显示手指编号，可直接开始临时测试。'
        )
        self.spacing = QtWidgets.QDoubleSpinBox()
        self.spacing.setRange(0, 0)
        self.spacing.setSpecialValueText("半高宽内等间隔5点")
        self.spacing.setToolTip('自动点全部位于峰的半高宽内；人工用左右方向键调整后可以不等间隔。')
        self.spacing.setEnabled(False)
        self.cycles = QtWidgets.QSpinBox()
        self.cycles.setRange(32, 512)
        self.cycles.setValue(256)
        self.cycles.setSuffix(" 帧")
        self.continuous = QtWidgets.QCheckBox('连续循环')
        self.continuous.setChecked(True)
        self.cycles.setEnabled(False)
        self.continuous.toggled.connect(lambda enabled: self.cycles.setEnabled(not enabled))
        self.start_button = QtWidgets.QPushButton("开始临时测试")
        self.start_button.setEnabled(False)
        self.stop_button = QtWidgets.QPushButton("停止并关光")
        self.stop_button.setEnabled(False)
        acquisition_controls = QtWidgets.QHBoxLayout()
        for widget in (
                self.channel_combo,
                self.reference_button,
                self.equal_interval_reference_button,
                QtWidgets.QLabel('采谱间隔'), self.equal_interval_nm,
                QtWidgets.QLabel('等间隔逐点等待'), self.equal_interval_settle,
                self.stop_dense_button):
            acquisition_controls.addWidget(widget)
        acquisition_controls.addStretch(1)
        layout.addLayout(acquisition_controls)
        for widget in (
                self.save_finger_button, self.load_finger_button, self.spacing,
                self.continuous, self.cycles, self.start_button,
                self.stop_button):
            controls.addWidget(widget)
        layout.addLayout(controls)
        reference_controls = QtWidgets.QHBoxLayout()
        self.reference_line_button = QtWidgets.QPushButton('采集并绘制参考线（当前45点）')
        self.reference_line_button.setEnabled(False)
        self.show_reference = QtWidgets.QPushButton('显示参考线（虚线）')
        self.show_reference.setCheckable(True)
        self.show_reference.setChecked(True)
        self.clear_reference_button = QtWidgets.QPushButton('清除参考线')
        self.reference_gain = QtWidgets.QDoubleSpinBox()
        self.reference_gain.setRange(.1,100)
        self.reference_gain.setValue(1)
        self.reference_gain.setSuffix(' ×')
        self.reference_gain.setToolTip('仅改变参考线显示高度，原始参考ADC不变；与实时倍数不同时不能直接比较线的高度。')
        for widget in (self.reference_line_button, self.show_reference, QtWidgets.QLabel('参考线数字放大'),
                       self.reference_gain, self.clear_reference_button):
            reference_controls.addWidget(widget)
        layout.addLayout(reference_controls)
        settings = QtWidgets.QHBoxLayout()
        self.analog_gain = QtWidgets.QComboBox()
        for label, selector in (('2 kΩ', 0), ('5 kΩ（默认）', 2), ('20 kΩ', 3), ('40 kΩ', 1)):
            self.analog_gain.addItem(label, selector)
        self.analog_gain.setCurrentIndex(1)
        self.analog_gain.setToolTip('下次开始采集时生效，不移动扫描点；切换后需重新做同档位幅值对照。')
        self.digital_gain = QtWidgets.QDoubleSpinBox()
        self.digital_gain.setRange(.1, 100)
        self.digital_gain.setValue(1)
        self.digital_gain.setSuffix(' ×')
        self.display_unit_button = QtWidgets.QPushButton('显示：ADC码')
        self.display_unit_button.setToolTip(
            '点击依次切换：ADC码 → ADC输入电压(V) → 估算光功率(dBm)。\n'
            'dBm依据TPC5121的2.5 V/4096、采集时2/5/20/40 kΩ跨阻和'
            'XCPD1007在1550 nm的0.85 A/W响应度换算；不是波长计功率校准值。'
        )
        self.display_unit_button.clicked.connect(self.cycle_display_mode)
        self.first_delay = QtWidgets.QSpinBox()
        self.first_delay.setRange(50, 850)
        self.first_delay.setSingleStep(25)
        self.first_delay.setValue(350)
        self.first_delay.setSuffix(' µs')
        self.first_delay.setToolTip('以25 µs为步长；不能仅凭双ADC接近判断波长已稳定。')
        self.boundary_delay = QtWidgets.QSpinBox()
        self.boundary_delay.setRange(0, 3000)
        self.boundary_delay.setSingleStep(25)
        self.boundary_delay.setValue(1800)
        self.boundary_delay.setSuffix(' µs')
        self.boundary_delay.setToolTip('仅每峰第一个点追加；总等待不限，配套固件支持长帧，允许低于15 Hz。')
        for label, widget in (('模拟放大', self.analog_gain), ('实时数字放大', self.digital_gain),
                              ('峰内首次采样等待', self.first_delay), ('跨峰额外等待', self.boundary_delay)):
            settings.addWidget(QtWidgets.QLabel(label))
            settings.addWidget(widget)
        settings.addWidget(self.display_unit_button)
        layout.addLayout(settings)
        self.point_waits = None
        self.point_wait_route = None
        point_settings = QtWidgets.QHBoxLayout()
        self.use_point_waits = QtWidgets.QCheckBox('使用逐点等待（实验功能，替代统一/跨峰等待）')
        self.edit_point_waits = QtWidgets.QPushButton('设置45个点的等待…')
        self.test_switch_timing = QtWidgets.QPushButton('测试45点达到90%时间')
        self.test_switch_timing.setEnabled(self.plan is not None)
        self.test_switch_timing.setToolTip(
            '对当前15/45点按小波长到大波长连续测试3轮；'
            '不做反向回绕审计。CH1主变化达到90%的板上时间会写入逐点等待表。'
        )
        self.edit_point_waits.clicked.connect(self.configure_point_waits)
        self.test_switch_timing.clicked.connect(self.start_switch_timing)
        self.use_point_waits.toggled.connect(self.refresh_wait_controls)
        point_settings.addWidget(self.use_point_waits)
        point_settings.addWidget(self.edit_point_waits)
        point_settings.addWidget(self.test_switch_timing)
        layout.addLayout(point_settings)
        self.fit_button = QtWidgets.QPushButton('适合当前幅值（只调整显示范围）')
        self.fit_button.setToolTip('每次开始扫描在预热后适配一次，随后锁定纵轴；按压时不自动缩放，保留真实高度变化。')
        self.fit_button.clicked.connect(self.fit_values)
        fit_controls = QtWidgets.QHBoxLayout()
        fit_controls.addWidget(self.fit_button)
        self.peak_view = QtWidgets.QComboBox()
        self.peak_view.addItem('查看全部9峰（仍扫描固定45点）')
        for group in range(9):
            self.peak_view.addItem(f'放大查看第{group+1}峰（仅改变横轴范围）')
        self.peak_view.currentIndexChanged.connect(self.fit_peak_view)
        fit_controls.addWidget(self.peak_view)
        self.show_dense_selection = QtWidgets.QPushButton('显示密集光谱和可调整45点')
        self.show_dense_selection.setCheckable(True)
        self.show_dense_selection.setChecked(True)
        self.show_dense_selection.setEnabled(False)
        self.show_dense_selection.setToolTip(
            '单击一个采样点将其选中，再按键盘左/右键逐格移动。'
            '移动只修改45点路线；参考线仅在主动点击采集按钮时更新。'
        )
        self.show_dense_selection.toggled.connect(self.redraw_dense_selection)
        fit_controls.addWidget(self.show_dense_selection)
        self.show_dense_failures = QtWidgets.QPushButton('显示未通过点（红色×）')
        self.show_dense_failures.setCheckable(True)
        self.show_dense_failures.setChecked(True)
        self.show_dense_failures.setToolTip(
            '仅切换密集采集未通过点的红色×标记；'
            '不删除失败记录，不影响选点、参考线或开关测试。'
        )
        self.show_dense_failures.toggled.connect(self.redraw_dense_selection)
        fit_controls.addWidget(self.show_dense_failures)
        layout.addLayout(fit_controls)
        self.status = QtWidgets.QLabel("等待当前密集光谱；尚未验证帧率或光学精度")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.finger_record_label = QtWidgets.QLabel('当前手指记录：未命名（未载入保存记录）')
        self.finger_record_label.setStyleSheet('color: #2563eb;')
        layout.addWidget(self.finger_record_label)
        self.show_peak_fits = QtWidgets.QCheckBox('实时峰形拟合与中心波长（高斯＋基线，五点估计）')
        self.show_peak_fits.setChecked(True)
        self.show_peak_fits.setToolTip(
            '使用原始CH1 ADC和旧标定波长坐标；所有有限五点排布都会尝试拟合。'
            '不满足先升后降、R²偏低或中心/峰宽约束不足时仍显示结果，并用⚠标记。'
            '不修改采样点、不补偿原始幅值；显示位数不代表精度。'
        )
        layout.addWidget(self.show_peak_fits)
        self.peak_fit_status = QtWidgets.QLabel('等待实时帧；拟合波长不是波长计实测值。')
        layout.addWidget(self.peak_fit_status)
        self.peak_fit_table = QtWidgets.QTableWidget(3, 9)
        self.peak_fit_table.setHorizontalHeaderLabels([f'峰{i+1}' for i in range(9)])
        self.peak_fit_table.setVerticalHeaderLabels(['中心 nm（估计）', 'R²', '状态'])
        self.peak_fit_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.peak_fit_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.peak_fit_table.setMaximumHeight(124)
        self.peak_fits = []
        self.plot = pg.PlotWidget()
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.plot.setMinimumHeight(220)
        self.plot.setLabel('bottom', '标定实测波长', units='nm')
        self.plot.setLabel('left', 'CH1 ADC × 数字显示倍数', units='code')
        self.plot.setYRange(0, 4095, padding=0)
        self.plot.enableAutoRange(axis='y', enable=False)
        self.curves = [self.plot.plot(pen=pg.intColor(i, 9), symbol='o', symbolSize=6) for i in range(9)]
        self.dense_curve = self.plot.plot(
            pen=pg.mkPen((165, 172, 184, 190), width=1),
            name='密集CH1光谱',
        )
        self.dense_curve.setZValue(-20)
        self.dense_failed_curve = self.plot.plot(
            pen=None, symbol='x', symbolSize=10,
            symbolPen=pg.mkPen((239, 68, 68), width=2),
            name='密集采集未通过点',
        )
        self.dense_failed_curve.setZValue(25)
        self.reference_curves = [self.plot.plot(pen=pg.mkPen(pg.intColor(i,9),width=2,style=QtCore.Qt.DashLine),
                                                symbol='x',symbolSize=8) for i in range(9)]
        self.fitted_curves = [self.plot.plot(pen=pg.mkPen(pg.intColor(i,9), width=2,
                              style=QtCore.Qt.DotLine)) for i in range(9)]
        self.fit_labels = []
        for i in range(9):
            label = pg.TextItem(color=pg.intColor(i,9), anchor=(.5,1))
            self.plot.addItem(label)
            label.hide()
            self.fit_labels.append(label)
        self.show_peak_fits.toggled.connect(self.redraw_peak_fits)
        layout.addWidget(self.plot, 1)
        self.table = QtWidgets.QTableWidget(45, 6)
        self.table.setHorizontalHeaderLabels(['峰', '物理索引', '目标nm', '实测标定nm', 'CH1 ADC', '慢速参考ADC'])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setMaximumHeight(210)
        self.data_tabs = QtWidgets.QTabWidget()
        self.data_tabs.addTab(self.peak_fit_table, '9峰波长与拟合质量')
        self.data_tabs.addTab(self.table, '45点原始ADC')
        self.data_tabs.setFixedHeight(140)
        layout.addWidget(self.data_tabs)
        self.save_finger_button.clicked.connect(self.save_finger_capture)
        self.load_finger_button.clicked.connect(self.load_finger_capture)
        self.start_button.clicked.connect(self.start_scan)
        self.reference_button.clicked.connect(lambda: self.start_scan(reference=True))
        self.equal_interval_reference_button.clicked.connect(
            self.start_equal_interval_reference
        )
        self.stop_dense_button.clicked.connect(self.stop_dense_reference)
        self.stop_button.clicked.connect(self.stop)
        self.digital_gain.valueChanged.connect(self.redraw_values)
        self.reference_gain.valueChanged.connect(self.redraw_values)
        self.reference_line_button.clicked.connect(lambda: self.start_scan(reference_line=True))
        self.clear_reference_button.clicked.connect(self.clear_reference_line)
        self.show_reference.toggled.connect(self.redraw_values)
        self.analog_gain.currentIndexChanged.connect(self.clear_reference_line)
        self.analog_gain.currentIndexChanged.connect(self.invalidate_shape_verification)
        self.analog_gain.currentIndexChanged.connect(lambda _i:self.clear_peak_fits('模拟档位已变更，请重新采集'))
        self._load_machine_defaults()
        if self.persistence_enabled:
            self._load_last_route_session()
            # A named hand/finger record is stronger than the rolling route
            # session: it carries the complete 2001-point spectrum and the
            # sparse reference line, so the operator can start immediately.
            self._load_saved_finger_capture(auto=True)
        self.channel_combo.currentIndexChanged.connect(self.change_signal_channel)
        for signal in (
            self.digital_gain.valueChanged,
            self.reference_gain.valueChanged,
            self.first_delay.valueChanged,
            self.boundary_delay.valueChanged,
            self.equal_interval_nm.valueChanged,
            self.equal_interval_settle.valueChanged,
            self.use_point_waits.toggled,
            self.show_reference.toggled,
            self.show_dense_selection.toggled,
            self.show_dense_failures.toggled,
        ):
            signal.connect(lambda _value=None: self._save_last_route_session())
        self._sync_channel_ui()

    def selected_channel(self):
        return int(self.channel_combo.currentData())

    def channel_name(self, channel=None):
        channel = self.selected_channel() if channel is None else int(channel)
        return f'CH{channel}'

    def effective_feedback_selector(self, channel=None):
        channel = self.selected_channel() if channel is None else int(channel)
        return int(self.analog_gain.currentData()) if channel < 2 else 0

    def _sync_channel_ui(self):
        channel = self.selected_channel()
        name = self.channel_name(channel)
        self.analog_gain.setEnabled(self.worker is None and channel < 2)
        self.analog_gain.setToolTip(
            ('下次开始采集时生效，不移动扫描点；切换后需重新做同档位幅值对照。'
             if channel < 2 else
             f'{name}硬件跨阻固定为2 kΩ，不使用CH0/CH1可选模拟档位。')
        )
        self.show_peak_fits.setToolTip(
            f'使用原始{name} ADC和旧标定波长坐标；所有有限五点排布都会尝试拟合。'
            '不满足先升后降、R²偏低或中心/峰宽约束不足时仍显示结果，并用⚠标记。'
            '不修改采样点、不补偿原始幅值；显示位数不代表精度。'
        )
        if self.plan is not None:
            self._sync_dynamic_ui()
        else:
            self.title_label.setText(
                f'临时测试模式 · {self.runtime_machine_label} · {name} · 等待光谱选点'
            )
        self._update_display_unit_ui()

    def change_signal_channel(self, _index=None):
        """Invalidate channel-specific spectra when the operator changes ADC input."""
        if self._restoring_session:
            self._sync_channel_ui()
            return
        if self.worker is not None:
            return
        channel = self.selected_channel()
        self.plan = None
        self.plan_ready = False
        self.latest_values = None
        self.reference_values = None
        self.reference_wavelengths = None
        self.dense_wavelengths = None
        self.dense_values = None
        self.dense_failed_points = []
        self._live_dense_rows = []
        self.latest_channel = channel
        self.reference_channel = channel
        self.dense_channel = channel
        self.point_waits = None
        self.point_wait_route = None
        self.use_point_waits.setChecked(False)
        self._clear_selection_markers()
        self.dense_curve.clear()
        self.dense_failed_curve.clear()
        for curve in (*self.curves, *self.reference_curves, *self.fitted_curves):
            curve.clear()
        self.start_button.setEnabled(False)
        self.reference_line_button.setEnabled(False)
        self.save_finger_button.setEnabled(False)
        self.clear_peak_fits(f'已切换到{self.channel_name(channel)}，等待重新采集光谱')
        self._sync_channel_ui()
        self.status.setText(
            f'已切换到{self.channel_name(channel)}；通道的峰位和幅值记录相互独立，'
            '请使用任一种方案重新采集光谱并选点。'
        )

    def cycle_display_mode(self, _checked=False):
        modes = ('adc', 'voltage', 'dbm')
        self._set_display_mode(
            modes[(modes.index(self.display_mode) + 1) % len(modes)]
        )

    def _set_machine_storage_paths(self, machine_id):
        """Select private rolling records for one physical interrogator."""

        machine_id = str(machine_id)
        base = output_path('machines', machine_id, 'temporary_test')
        # Unit tests and portable callers historically patched these module
        # constants.  Honour an explicit override while normal applications
        # use the new per-machine layout.
        self.last_session_path = (
            LAST_SESSION_PATH if LAST_SESSION_PATH != _LEGACY_LAST_SESSION_PATH
            else base / 'last_45_session.json'
        )
        self.finger_capture_dir = (
            FINGER_CAPTURE_DIR if FINGER_CAPTURE_DIR != _LEGACY_FINGER_CAPTURE_DIR
            else base / 'finger_captures'
        )
        self.last_finger_capture_path = (
            LAST_FINGER_CAPTURE_PATH
            if LAST_FINGER_CAPTURE_PATH != _LEGACY_LAST_FINGER_CAPTURE_PATH
            else base / 'last_finger_capture.json'
        )
        self.storage_machine_id = machine_id

    def _machine_defaults_path(self):
        machine_default = resource_path(
            'machine_defaults', self.storage_machine_id,
            'temporary_test_defaults.json'
        )
        if machine_default.is_file():
            return machine_default
        return resource_path('temporary_test_defaults.json')

    def _load_machine_defaults(self):
        """Load only the selected machine's factory temporary-test route."""

        defaults = self._machine_defaults_path()
        if not defaults.exists():
            return False
        try:
            config = json.loads(defaults.read_text(encoding='utf-8'))
            self.selection_baseline_fraction = config.get('baseline_fraction')
            self.selection_soft_baseline = bool(config.get('soft_baseline', False))
            route = config.get('validated_route_indices')
            values = config.get('validated_reference_values')
            if (isinstance(route, list) and len(route) == 9
                    and isinstance(values, list) and len(values) == 45):
                self.validated_route_indices = [list(group) for group in route]
                self.validated_route_values = list(values)
                self.validated_route_source = config.get(
                    'validated_reference_source', 'validated live route'
                )
            self.digital_gain.setValue(config.get('digital_display_gain', 1))
            self.first_delay.setValue(config.get('first_delay_us', 350))
            self.boundary_delay.setValue(config.get('boundary_extra_us', 1800))
            if route:
                import app_JDSU as app
                self.plan = build_selected_plan(
                    app.load_fullband_accuracy_table(),
                    route,
                    values,
                    source=config.get(
                        'validated_reference_source', 'validated live route'
                    ),
                )
                self.render_plan(ready=False)
                waits = config.get('point_delays_us')
                if isinstance(waits, list) and len(waits) == 45:
                    command(
                        self.plan['rows'], 32, 1, first_delay_us=650,
                        point_delays_us=waits,
                    )
                    self.point_waits = list(waits)
                    self.point_wait_route = self.point_route_key()
                    self.use_point_waits.setChecked(True)
                self.status.setText(
                    '已载入当前机号实测通过的9×5路线及逐点等待；'
                    '为防峰位再次漂移，开始扫描前仍须采集当前45点慢速参考线。'
                )
                return True
            if config.get('selection_plan'):
                self.plan = load_saved_plan(
                    resource_path(config['selection_plan'])
                )
                self.render_plan(ready=False)
                self.status.setText(
                    '已载入旧密集光谱45点，仅供预览；峰位会漂移，开始扫描前请重采/选择当前密集光谱，'
                    '或先采集当前45点慢速参考线并通过9峰形状验证。'
                )
                return True
            if config.get('selection_reference'):
                from select_old_ch1 import select_old_reference
                self.spacing.setValue(config.get('spacing_nm', .16))
                self.plan = select_old_reference(
                    resource_path(config['selection_reference']),
                    self.spacing.value(), config.get('baseline_fraction'),
                    config.get('soft_baseline', False),
                )
                self.render_plan(ready=False)
                self.status.setText(
                    '已用保存的CH1光谱选出9×5点，仅供预览；开始扫描前须采集当前45点慢速参考线并通过9峰形状验证。'
                )
                return True
            if config.get('calibration'):
                self.plan = load_calibrated_plan(
                    resource_path(config['calibration'])
                )
                self.render_plan(ready=False)
                self.status.setText(
                    '已载入45点实测标定表，仅供预览；开始扫描前须采集当前45点慢速参考线并通过9峰形状验证。'
                )
                return True
            self.status.setText(
                f'{self.runtime_machine_label}尚无临时模式点表；请重采密集光谱并选点，或载入该机保存的手指记录。'
            )
            return False
        except Exception as exc:
            self.plan = None
            self.start_button.setEnabled(False)
            self.status.setText(f'当前机号默认临时表不可用：{exc}')
            return False

    def _reset_machine_state(self):
        self._restoring_session = True
        try:
            self.plan = None
            self.plan_ready = False
            self.latest_values = None
            self.reference_values = None
            self.reference_wavelengths = None
            self.dense_wavelengths = None
            self.dense_values = None
            self.dense_source = None
            self.dense_failed_points = []
            self._live_dense_rows = []
            self._dense_capture_active = False
            self.finger_record_name = None
            self._spectrum_dirty = False
            self.validated_route_indices = None
            self.validated_route_values = None
            self.validated_route_source = None
            self.point_waits = None
            self.point_wait_route = None
            self.use_point_waits.setChecked(False)
            self._clear_selection_markers()
            self.dense_curve.clear()
            self.dense_failed_curve.clear()
            for curve in (*self.curves, *self.reference_curves, *self.fitted_curves):
                curve.clear()
            for label in self.fit_labels:
                label.hide()
            self.table.clearContents()
            self.peak_fit_table.clearContents()
            self.start_button.setEnabled(False)
            self.reference_line_button.setEnabled(False)
            self.save_finger_button.setEnabled(False)
            self.show_dense_selection.setEnabled(False)
            self.finger_record_label.setText(
                f'当前手指记录：未命名（{self.runtime_machine_label}参数）'
            )
        finally:
            self._restoring_session = False

    def before_machine_switch(self):
        """Persist the current unit before the global runtime id changes."""

        self._save_last_route_session(ready=self.plan_ready)

    def reload_machine_profile(self, machine_label, parameter_note):
        """Switch labels, defaults and saved records as one atomic profile."""

        self.runtime_machine_label = str(machine_label)
        self.runtime_parameter_note = str(parameter_note)
        self._set_machine_storage_paths(get_runtime_machine_id())
        self.machine_profile_label.setText(
            f"当前设备：{self.runtime_machine_label}｜{self.runtime_parameter_note}"
        )
        self._reset_machine_state()
        self._load_machine_defaults()
        if self.persistence_enabled:
            self._load_last_route_session()
            self._load_saved_finger_capture(auto=True)
        self._sync_dynamic_ui()

    def _set_display_mode(self, mode, *, persist=True):
        mode = str(mode)
        if mode not in ('adc', 'voltage', 'dbm'):
            mode = DEFAULT_DISPLAY_MODE
        self.display_mode = mode
        self._update_display_unit_ui()
        if hasattr(self, 'plot'):
            self.redraw_values()
            self.fit_values()
        if persist:
            self._save_last_route_session()

    def _update_display_unit_ui(self, _value=None):
        if not hasattr(self, 'display_unit_button'):
            return
        label = {
            'adc': '显示：ADC码',
            'voltage': '显示：电压 V',
            'dbm': '显示：光功率 dBm',
        }[self.display_mode]
        self.display_unit_button.setText(label)
        physical = self.display_mode != 'adc'
        self.digital_gain.setEnabled(not physical)
        self.reference_gain.setEnabled(not physical)
        self.digital_gain.setToolTip(
            'ADC码显示时仅改变实时曲线高度，不修改原始ADC。'
            if not physical else '电压/dBm是物理换算值，数字显示倍率不参与换算。'
        )
        self.reference_gain.setToolTip(
            'ADC码显示时仅改变参考线高度，原始参考ADC不变。'
            if not physical else '电压/dBm是物理换算值，参考线显示倍率不参与换算。'
        )
        if hasattr(self, 'table'):
            channel_name = self.channel_name()
            current, reference = {
                'adc': (f'{channel_name} ADC码', '慢速参考ADC码'),
                'voltage': (f'{channel_name}电压 V', '慢速参考电压 V'),
                'dbm': (f'{channel_name}估算光功率 dBm', '参考估算光功率 dBm'),
            }[self.display_mode]
            self.table.setHorizontalHeaderLabels([
                '峰', '物理索引', '目标nm', '实测标定nm', current, reference
            ])
            self.data_tabs.setTabText(1, f'{self.point_count() or 45}点数据')
        self._refresh_table_display_values()
        if hasattr(self, 'plot'):
            self._update_plot_unit_label()

    def _plot_display_values(self, values, *, reference=False, selector=None):
        raw = np.asarray(values, dtype=float)
        if self.display_mode == 'adc':
            gain = self.reference_gain.value() if reference else self.digital_gain.value()
            return raw * gain
        if self.display_mode == 'voltage':
            return adc_code_to_voltage(raw)
        if selector is None:
            selector = int(self.analog_gain.currentData())
        return adc_code_to_dbm(raw, int(selector), floor_nonpositive=True)

    def _format_table_display_value(self, adc_code, selector=None):
        if selector is None:
            selector = int(self.analog_gain.currentData())
        return format_physical_value(adc_code, self.display_mode, int(selector))

    def _refresh_table_display_values(self):
        if not hasattr(self, 'table') or self.plan is None:
            return
        unit = {'adc': 'ADC码', 'voltage': 'V', 'dbm': 'dBm'}[self.display_mode]
        for column, values, selector in (
                (4, self.latest_values, self.latest_feedback_selector),
                (5, self.reference_values, self.reference_feedback_selector)):
            for index in range(self.point_count()):
                if not isinstance(values, list) or index >= len(values):
                    self.table.setItem(index, column, QtWidgets.QTableWidgetItem('—'))
                    continue
                raw = float(values[index])
                item = QtWidgets.QTableWidgetItem(
                    self._format_table_display_value(raw, selector)
                )
                item.setToolTip(
                    f'原始{self.channel_name(self.latest_channel if column == 4 else self.reference_channel)} '
                    f'ADC={raw:g}码；当前显示单位={unit}；'
                    f'采集时模拟跨阻={transimpedance_ohm(selector) / 1000:g} kΩ。'
                )
                self.table.setItem(index, column, item)

    def _update_plot_unit_label(self):
        if self.display_mode == 'adc':
            self.plot.setLabel(
                'left',
                f'{self.channel_name()} ADC · 实时×{self.digital_gain.value():g} / '
                f'参考×{self.reference_gain.value():g}', units='code'
            )
        elif self.display_mode == 'voltage':
            self.plot.setLabel('left', f'{self.channel_name()} ADC输入电压', units='V')
        else:
            self.plot.setLabel(
                'left', f'{self.channel_name()}估算光功率 · 按各曲线采集档位 · '
                f'{PD_RESPONSIVITY_A_PER_W:g} A/W', units='dBm'
            )

    def point_count(self):
        return len(self.plan.get('rows', [])) if isinstance(self.plan, dict) else 0

    def peak_count(self):
        return self.point_count() // 5

    @staticmethod
    def _safe_finger_filename(label):
        """Keep a user-visible hand name while making a safe Windows filename."""
        value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', str(label)).strip(' .')
        return value or '未命名手指'

    @staticmethod
    def _valid_dense_spectrum(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return bool(
            x.ndim == 1 and y.shape == x.shape and len(x) >= 5
            and np.all(np.isfinite(x)) and np.all(np.isfinite(y))
            and np.all(np.diff(x) > 0)
            and np.max(np.diff(x)) <= .10001
        )

    def has_unsaved_spectrum(self):
        """Return whether a usable displayed spectrum lacks a named save."""
        return bool(
            self._spectrum_dirty
            and self._valid_dense_spectrum(
                self.dense_wavelengths, self.dense_values
            )
        )

    def confirm_save_before_close(self, parent=None):
        """Offer a named save before the owning application is destroyed."""
        if not self.has_unsaved_spectrum():
            return True
        answer = QtWidgets.QMessageBox.question(
            parent or self,
            '保存临时模式光谱？',
            f'当前显示的{self.channel_name(self.dense_channel)}密集光谱尚未保存为命名记录。\n'
            '是否在关闭程序前保存？',
            (QtWidgets.QMessageBox.Save
             | QtWidgets.QMessageBox.Discard
             | QtWidgets.QMessageBox.Cancel),
            QtWidgets.QMessageBox.Save,
        )
        if answer == QtWidgets.QMessageBox.Cancel:
            return False
        if answer == QtWidgets.QMessageBox.Save:
            return bool(self.save_finger_capture())
        # The rolling recovery session remains available, but do not ask a
        # second time when the unified owner closes the embedded local window.
        self._spectrum_dirty = False
        return True

    def showEvent(self, event):
        """Redraw retained data after returning from another stacked page."""
        super().showEvent(event)
        if hasattr(self, 'plot'):
            self.redraw_dense_selection()
            self.redraw_values()
            self.redraw_peak_fits()

    def _capture_payload(self, finger_label):
        """Build a self-contained named record without touching the hardware."""
        dense_x = np.asarray(
            self.dense_wavelengths if self.dense_wavelengths is not None
            else self.plan.get('reference_wavelengths_nm', [])
            if isinstance(self.plan, dict) else [], dtype=float
        )
        dense_y = np.asarray(
            self.dense_values if self.dense_values is not None
            else self.plan.get('reference_values', [])
            if isinstance(self.plan, dict) else [], dtype=float
        )
        if not self._valid_dense_spectrum(dense_x, dense_y):
            raise ValueError('需要先完成一条步长不大于0.10 nm的有效密集谱线')
        point_count = self.point_count()
        if self.plan is not None and point_count not in range(5, 46, 5):
            raise ValueError('当前15/45点路线无效')
        references = self.reference_values
        has_reference = bool(
            point_count
            and isinstance(references, list) and len(references) == point_count
            and np.all(np.isfinite(np.asarray(references, dtype=float)))
        )
        reference_wavelengths = self.reference_wavelengths
        if (has_reference and (not isinstance(reference_wavelengths, list)
                or len(reference_wavelengths) != point_count
                or not np.all(np.isfinite(np.asarray(reference_wavelengths, dtype=float))))):
            reference_wavelengths = [float(row['measured_nm']) for row in self.plan['rows']]
        plan = None
        if self.plan is not None:
            plan = stamp_machine_metadata(
                validate_plan_payload(self.plan),
                ('fullband_2001', f'{self.peak_count()}_peak_route'),
            )
            plan['reference_wavelengths_nm'] = dense_x.tolist()
            plan['reference_values'] = dense_y.tolist()
            plan['finger_label'] = str(finger_label)
            plan['signal_channel'] = self.selected_channel()
        categories = (
            'fullband_2001',
            f'{self.peak_count()}_peak_route' if plan is not None
            else 'dense_spectrum',
        )
        return stamp_machine_metadata({
            'schema': 'temporary_test_finger_capture_v1',
            'saved_s': time.time(),
            'finger_label': str(finger_label),
            'ready': bool(self.plan_ready and has_reference),
            'point_count': point_count,
            'peak_count': self.peak_count(),
            'plan': plan,
            'dense_wavelengths_nm': dense_x.tolist(),
            'dense_values': dense_y.tolist(),
            'dense_source': str(
                self.dense_source
                or (self.plan.get('reference_source', '')
                    if isinstance(self.plan, dict) else '')
            ),
            'dense_point_count': int(len(dense_x)),
            'signal_channel': self.selected_channel(),
            'latest_channel': int(self.latest_channel),
            'dense_channel': int(self.dense_channel),
            'reference_channel': int(self.reference_channel),
            'dense_acquisition_method': self.dense_acquisition_method,
            'reference_values': (
                [float(value) for value in references]
                if has_reference else None
            ),
            'reference_wavelengths_nm': (
                [float(value) for value in reference_wavelengths]
                if has_reference else None
            ),
            'feedback_selector': self.effective_feedback_selector(),
            'dense_feedback_selector': int(self.dense_feedback_selector),
            'reference_feedback_selector': int(self.reference_feedback_selector),
            'digital_display_gain': float(self.digital_gain.value()),
            'reference_display_gain': float(self.reference_gain.value()),
            'display_unit': self.display_mode,
            'first_delay_us': int(self.first_delay.value()),
            'boundary_extra_us': int(self.boundary_delay.value()),
            'equal_interval_nm': float(self.equal_interval_nm.value()),
            'equal_interval_settle_s': float(self.equal_interval_settle.value()),
            'show_dense_selection': bool(self.show_dense_selection.isChecked()),
            'show_dense_failures': bool(self.show_dense_failures.isChecked()),
            'show_reference': bool(self.show_reference.isChecked()),
            'use_point_waits': bool(
                self.use_point_waits.isChecked()
                and self.point_waits is not None
                and point_count
                and self.point_wait_route == self.point_route_key()
            ),
            'point_delays_us': list(self.point_waits) if (
                self.point_waits is not None
                and point_count
                and self.point_wait_route == self.point_route_key()
            ) else None,
        }, categories)

    def save_finger_capture(self):
        """Save a named record; only a nine-peak record becomes the default."""
        if self.worker is not None:
            self.status.setText('采集进行中，停止并关光后才能保存手指记录。')
            return False
        finger_label, accepted = QtWidgets.QInputDialog.getText(
            self, '保存手指采集记录', '请输入手指编号或名称（例如：1号手指）：',
            text=(self.finger_record_name or ''),
        )
        finger_label = str(finger_label).strip()
        if not accepted or not finger_label:
            return False
        try:
            payload = self._capture_payload(finger_label)
            self.finger_capture_dir.mkdir(parents=True, exist_ok=True)
            safe_name = self._safe_finger_filename(finger_label)
            record_path = self.finger_capture_dir / f'{safe_name}.json'
            if record_path.exists():
                answer = QtWidgets.QMessageBox.question(
                    self, '覆盖已有手指记录',
                    f'已存在“{finger_label}”的记录，是否用当前采集覆盖？',
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.No,
                )
                if answer != QtWidgets.QMessageBox.Yes:
                    return False
            payload['record_file'] = str(record_path.resolve())
            encoded = json.dumps(payload, ensure_ascii=False, indent=2)
            temporary = record_path.with_suffix('.tmp')
            temporary.write_text(encoded, encoding='utf-8')
            temporary.replace(record_path)
            is_default_nine_peak = (
                int(payload.get('peak_count', 0)) == 9
                and int(payload.get('point_count', 0)) == 45
                and bool(payload.get('ready'))
            )
            if is_default_nine_peak:
                pointer = self.last_finger_capture_path.with_suffix('.tmp')
                pointer.parent.mkdir(parents=True, exist_ok=True)
                pointer.write_text(encoded, encoding='utf-8')
                pointer.replace(self.last_finger_capture_path)
            self.finger_record_name = finger_label
            self._spectrum_dirty = False
            self.finger_record_label.setText(
                f'当前手指记录：{finger_label}（{self.runtime_machine_label}参数；已保存'
                + ('，可直接使用）' if payload.get('ready') else '，尚需采集参考线）')
            )
            if is_default_nine_peak:
                suffix = '已设为下次启动的默认9峰/45点记录。'
            elif not payload.get('point_count'):
                suffix = '已保存为未选点密集谱记录，可稍后手动载入。'
            elif not payload.get('ready'):
                suffix = '已保存密集谱和选点；载入后仍需采集参考线。'
            else:
                suffix = '已保留为可手动载入的3峰/15点记录；下次仍默认载入9峰版本。'
            self.status.setText(
                f'已保存“{finger_label}”：{self.peak_count()}峰、{self.point_count()}点、'
                f'{len(payload["dense_values"])}点密集谱线'
                + ('和参考线；' if payload.get('reference_values') else '（无参考线）；')
                + suffix
            )
            self._save_last_route_session(ready=self.plan_ready)
            return True
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, '保存手指记录失败', str(exc))
            self.status.setText(f'手指记录未保存：{exc}')
            return False

    def _load_saved_finger_capture(self, path=None, *, auto=False):
        """Restore a named, self-contained record without any live I/O."""
        # Keep isolated/test or portable session roots isolated from the
        # desktop application's global "last finger" pointer.  In normal use
        # both files live in the same temporary-test directory.
        if auto and self.last_finger_capture_path.parent != self.last_session_path.parent:
            return False
        if auto:
            candidates = [self.last_finger_capture_path]
            if self.finger_capture_dir.exists():
                candidates.extend(self.finger_capture_dir.glob('*.json'))
            qualified = []
            for candidate in candidates:
                if not candidate.exists():
                    continue
                try:
                    candidate_payload = json.loads(
                        candidate.read_text(encoding='utf-8')
                    )
                    candidate_plan = candidate_payload.get('plan') or {}
                    candidate_count = int(candidate_payload.get(
                        'point_count', len(candidate_plan.get('rows', []))
                    ))
                    candidate_peaks = int(candidate_payload.get(
                        'peak_count', candidate_count // 5
                    ))
                    if (candidate_payload.get('schema')
                            == 'temporary_test_finger_capture_v1'
                            and candidate_count == 45 and candidate_peaks == 9
                            and bool(candidate_payload.get('ready'))):
                        qualified.append((
                            float(candidate_payload.get('saved_s', 0.0)),
                            candidate.stat().st_mtime,
                            candidate,
                        ))
                except (OSError, ValueError, TypeError):
                    continue
            if not qualified:
                return False
            path = max(qualified)[2]
        else:
            path = self.last_finger_capture_path if path is None else Path(path)
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
            if payload.get('schema') != 'temporary_test_finger_capture_v1':
                raise ValueError('手指记录格式不受支持')
            ensure_machine_compatible(payload)
            raw_plan = payload.get('plan')
            plan = None
            if raw_plan is not None:
                plan = stamp_machine_metadata(
                    validate_plan_payload(raw_plan),
                    ('fullband_2001', f"{payload.get('peak_count', 'unknown')}_peak_route"),
                )
            point_count = len(plan['rows']) if plan is not None else 0
            if auto and (point_count != 45 or not payload.get('ready')):
                return False
            if payload.get('point_count') not in (None, point_count):
                raise ValueError('手指记录的点数与点表不一致')
            dense_x = np.asarray(payload.get('dense_wavelengths_nm',
                                             plan.get('reference_wavelengths_nm', [])
                                             if plan is not None else []), dtype=float)
            dense_y = np.asarray(payload.get('dense_values',
                                             plan.get('reference_values', [])
                                             if plan is not None else []), dtype=float)
            if not self._valid_dense_spectrum(dense_x, dense_y):
                raise ValueError('记录中没有有效的密集谱线（步长需不大于0.10 nm）')
            references = payload.get('reference_values')
            has_reference = bool(
                point_count
                and isinstance(references, list) and len(references) == point_count
                and np.all(np.isfinite(np.asarray(references, dtype=float)))
            )
            reference_wavelengths = payload.get('reference_wavelengths_nm')
            if (has_reference and (not isinstance(reference_wavelengths, list)
                    or len(reference_wavelengths) != point_count
                    or not np.all(np.isfinite(np.asarray(reference_wavelengths, dtype=float))))):
                reference_wavelengths = [float(row['measured_nm']) for row in plan['rows']]
            channel = int(payload.get(
                'signal_channel', plan.get('signal_channel', 1)
                if plan is not None else 1
            ))
            if channel not in range(4):
                raise ValueError('记录中的ADC通道无效')
            selector = int(payload.get('feedback_selector', 2 if channel < 2 else 0))
            selector_index = self.analog_gain.findData(selector)
            if selector_index < 0:
                raise ValueError('记录中的模拟档位无效')
            dense_selector = int(payload.get('dense_feedback_selector', selector))
            reference_selector = int(payload.get('reference_feedback_selector', selector))
            transimpedance_ohm(dense_selector)
            transimpedance_ohm(reference_selector)
            label = str(payload.get('finger_label', '')).strip() or '未命名手指'
            self._restoring_session = True
            self.channel_combo.setCurrentIndex(self.channel_combo.findData(channel))
            self.analog_gain.setCurrentIndex(selector_index)
            self.latest_feedback_selector = int(selector)
            self.reference_feedback_selector = reference_selector
            self.dense_feedback_selector = dense_selector
            self.latest_channel = int(payload.get('latest_channel', channel))
            self.reference_channel = int(payload.get('reference_channel', channel))
            self.dense_channel = int(payload.get('dense_channel', channel))
            self.dense_acquisition_method = str(payload.get(
                'dense_acquisition_method', 'stable_window'
            ))
            self.dense_expected_points = int(payload.get('dense_point_count', len(dense_x)))
            self.digital_gain.setValue(float(payload.get('digital_display_gain', 1)))
            self.reference_gain.setValue(float(payload.get('reference_display_gain', 1)))
            # Every application start begins in dBm; a record's former view
            # preference must not override the operator's requested default.
            self._set_display_mode(DEFAULT_DISPLAY_MODE, persist=False)
            self.first_delay.setValue(int(payload.get('first_delay_us', 350)))
            self.boundary_delay.setValue(int(payload.get('boundary_extra_us', 1800)))
            self.equal_interval_nm.setValue(float(payload.get('equal_interval_nm', .02)))
            self.equal_interval_settle.setValue(float(
                payload.get('equal_interval_settle_s', .2)
            ))
            self.finger_record_name = label
            self.dense_wavelengths = dense_x
            self.dense_values = dense_y
            self.dense_source = str(payload.get(
                'dense_source', plan.get('reference_source', '')
                if plan is not None else ''
            ))
            if plan is not None:
                plan['reference_wavelengths_nm'] = dense_x.tolist()
                plan['reference_values'] = dense_y.tolist()
                self.plan = plan
                self.render_plan(ready=False)
            else:
                # A spectrum can be worth keeping even when automatic peak
                # selection did not finish.  Re-attempt selection on load;
                # if it still cannot identify 3/9 peaks, retain the raw curve.
                self.plan = None
                try:
                    self.prepare(
                        dense_x, dense_y, self.dense_source or str(path),
                        mark_dirty=False,
                    )
                    plan = self.plan
                    point_count = self.point_count()
                except Exception:
                    self.plan = None
                    self.plan_ready = False
                    self.start_button.setEnabled(False)
                    self.reference_line_button.setEnabled(False)
                    self.save_finger_button.setEnabled(True)
                    self.show_dense_selection.setEnabled(True)
                    self.show_dense_selection.setChecked(True)
                    self.redraw_dense_selection()
            if has_reference:
                self.reference_values = [float(value) for value in references]
                self.reference_wavelengths = [float(value) for value in reference_wavelengths]
            else:
                self.reference_values = None
                self.reference_wavelengths = None
            self.show_dense_selection.setChecked(bool(payload.get('show_dense_selection', True)))
            self.show_dense_failures.setChecked(bool(payload.get('show_dense_failures', True)))
            self.show_reference.setChecked(
                bool(payload.get('show_reference', True)) and has_reference
            )
            waits = payload.get('point_delays_us')
            if (self.plan is not None and payload.get('use_point_waits')
                    and isinstance(waits, list)
                    and len(waits) == point_count):
                command(self.plan['rows'], 32, 1, first_delay_us=650,
                        boundary_extra_us=0, point_delays_us=waits,
                        adc_channel=channel, feedback_selector=selector)
                self.point_waits = [int(value) for value in waits]
                self.point_wait_route = self.point_route_key()
                self.use_point_waits.setChecked(True)
            else:
                self.point_waits = None
                self.point_wait_route = None
                self.use_point_waits.setChecked(False)
            if has_reference:
                for index, value in enumerate(self.reference_values):
                    self.table.setItem(index, 5, QtWidgets.QTableWidgetItem(f'{value:g}'))
            self.plan_ready = bool(
                self.plan is not None and has_reference and payload.get('ready')
            )
            self.start_button.setEnabled(self.plan_ready)
            self.redraw_values()
            self.fit_values()
            self._spectrum_dirty = False
            self.finger_record_label.setText(
                f'当前手指记录：{label}（{self.runtime_machine_label}参数；已载入'
                + ('，可直接使用）' if self.plan_ready else '，尚需完成参考）')
            )
            if self.plan_ready:
                self.status.setText(
                    f'已载入“{label}”：{self.peak_count()}峰、{point_count}点及'
                    f'{len(dense_x)}点{self.channel_name(channel)}密集谱线/参考线；'
                    '可直接开始临时测试。只有主动点击参考线采集按钮才会重新采集。'
                )
            else:
                self.status.setText(
                    f'已载入“{label}”的{len(dense_x)}点'
                    f'{self.channel_name(channel)}密集谱线'
                    + (f'和{self.point_count()}个选点；' if self.plan is not None else '；')
                    + ('仍需采集参考线后才能开始临时测试。'
                       if self.plan is not None else
                       '当前仍未自动识别出可用峰，可继续查看谱线或重新采集。')
                )
            self._save_last_route_session(ready=self.plan_ready)
            return True
        except Exception as exc:
            if not auto:
                QtWidgets.QMessageBox.warning(self, '载入手指记录失败', str(exc))
                self.status.setText(f'手指记录未载入：{exc}')
            return False
        finally:
            self._restoring_session = False

    def load_finger_capture(self):
        if self.worker is not None:
            self.status.setText('采集进行中，停止并关光后才能载入手指记录。')
            return False
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, '选择手指采集记录', str(self.finger_capture_dir),
            '手指采集记录 (*.json)'
        )
        if not path:
            return False
        return self._load_saved_finger_capture(path)

    def _sync_dynamic_ui(self):
        """Resize labels/tables to the active detected peak count."""
        point_count = self.point_count()
        peak_count = self.peak_count()
        if not point_count or point_count % 5:
            return
        self.title_label.setText(
            f'临时测试模式 · {self.runtime_machine_label} · {self.channel_name()} · '
            f'{peak_count}峰×5点 · 仅扫描选定的{point_count}点'
        )
        self.note_label.setText(
            f'固定{point_count}点扫描：原始圆点/实线、参考虚线、拟合点线分开显示；'
            '等待自行调节，允许低于15 Hz。\n'
            '拟合波长基于旧标定坐标，仅为五点估计；双ADC接近不证明波长稳定，原始幅值不作修正。'
        )
        self.reference_line_button.setText(f'采集并绘制参考线（当前{point_count}点）')
        self.edit_point_waits.setText(f'设置{point_count}个点的等待…')
        self.test_switch_timing.setText(f'测试{point_count}点达到90%时间')
        self.test_switch_timing.setEnabled(
            self.worker is None and self.plan is not None
            and self.selected_channel() == 1
        )
        self.test_switch_timing.setToolTip(
            '当前高速开关诊断硬件仅测CH1；切换回CH1后可用。'
            if self.selected_channel() != 1 else
            f'对当前{point_count}点按小波长到大波长连续测试3轮；'
            '不做反向回绕审计。CH1主变化达到90%的板上时间会写入逐点等待表。'
        )
        self.show_dense_selection.setText(f'显示密集光谱和可调整{point_count}点')
        self.show_dense_selection.setToolTip(
            '单击一个采样点将其选中，再按键盘左/右键逐格移动。'
            f'移动只修改{point_count}点路线；参考线仅在主动点击采集按钮时更新。'
        )
        self.table.setRowCount(point_count)
        self.peak_fit_table.setColumnCount(peak_count)
        self.peak_fit_table.setHorizontalHeaderLabels(
            [f'峰{i + 1}' for i in range(peak_count)]
        )
        self.data_tabs.setTabText(0, f'{peak_count}峰波长与拟合质量')
        self.data_tabs.setTabText(1, f'{point_count}点数据')
        selected_view = min(self.peak_view.currentIndex(), peak_count)
        self.peak_view.blockSignals(True)
        self.peak_view.clear()
        self.peak_view.addItem(f'查看全部{peak_count}峰（仍扫描固定{point_count}点）')
        for group in range(peak_count):
            self.peak_view.addItem(f'放大查看第{group + 1}峰（仅改变横轴范围）')
        self.peak_view.setCurrentIndex(selected_view)
        self.peak_view.blockSignals(False)
        for group in range(9):
            visible = group < peak_count
            self.curves[group].setVisible(visible)
            self.reference_curves[group].setVisible(visible)
            self.fitted_curves[group].setVisible(visible)
            if not visible:
                self.fit_labels[group].hide()

    def _save_last_route_session(self, ready=None):
        if (not self.persistence_enabled or self._restoring_session
                or self.plan is None):
            return
        point_count = self.point_count()
        ready = self.plan_ready if ready is None else bool(ready)
        reference_values = list(self.reference_values) if (
            isinstance(self.reference_values, list)
            and len(self.reference_values) == point_count
            and np.all(np.isfinite(np.asarray(self.reference_values, dtype=float)))
        ) else None
        ready = bool(ready and reference_values is not None)
        plan = stamp_machine_metadata(
            self.plan,
            ('fullband_2001', f'{self.peak_count()}_peak_route'),
        )
        plan['signal_channel'] = self.selected_channel()
        plan['signal_channel_name'] = self.channel_name()
        payload = stamp_machine_metadata({
            'schema': 'temporary_test_last_session_v1',
            'saved_s': time.time(),
            'finger_record_name': self.finger_record_name,
            'unsaved_spectrum_changes': bool(self._spectrum_dirty),
            'ready': ready,
            'plan': plan,
            'reference_values': reference_values,
            'reference_wavelengths_nm': list(self.reference_wavelengths) if (
                reference_values is not None
                and isinstance(self.reference_wavelengths, list)
                and len(self.reference_wavelengths) == point_count
            ) else None,
            'signal_channel': self.selected_channel(),
            'latest_channel': int(self.latest_channel),
            'dense_channel': int(self.dense_channel),
            'reference_channel': int(self.reference_channel),
            'dense_acquisition_method': self.dense_acquisition_method,
            'dense_point_count': int(len(self.dense_wavelengths)) if (
                self.dense_wavelengths is not None
            ) else None,
            'feedback_selector': self.effective_feedback_selector(),
            'dense_feedback_selector': int(self.dense_feedback_selector),
            'reference_feedback_selector': int(self.reference_feedback_selector),
            'digital_display_gain': float(self.digital_gain.value()),
            'reference_display_gain': float(self.reference_gain.value()),
            'display_unit': self.display_mode,
            'first_delay_us': int(self.first_delay.value()),
            'boundary_extra_us': int(self.boundary_delay.value()),
            'equal_interval_nm': float(self.equal_interval_nm.value()),
            'equal_interval_settle_s': float(self.equal_interval_settle.value()),
            'show_dense_selection': bool(self.show_dense_selection.isChecked()),
            'show_dense_failures': bool(self.show_dense_failures.isChecked()),
            'show_reference': bool(self.show_reference.isChecked()),
            'use_point_waits': bool(
                self.use_point_waits.isChecked()
                and self.point_waits is not None
                and self.point_wait_route == self.point_route_key()
            ),
            'point_delays_us': list(self.point_waits) if (
                self.point_waits is not None
                and self.point_wait_route == self.point_route_key()
            ) else None,
        }, ('fullband_2001', f'{self.peak_count()}_peak_route'))
        self.last_session_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.last_session_path.with_suffix('.tmp')
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        temporary.replace(self.last_session_path)

    def _load_last_route_session(self):
        if not self.last_session_path.exists():
            return self._migrate_latest_reference_line()
        try:
            payload = json.loads(self.last_session_path.read_text(encoding='utf-8'))
            if payload.get('schema') != 'temporary_test_last_session_v1':
                raise ValueError('会话格式不匹配')
            ensure_machine_compatible(payload)
            plan = validate_plan_payload(payload.get('plan'))
            point_count = len(plan['rows'])
            peak_count = point_count // 5
            # Three-peak sessions remain named records for explicit loading;
            # application startup always returns to the nine-peak route.
            if point_count != 45 or peak_count != 9:
                return False
            plan = stamp_machine_metadata(
                plan, ('fullband_2001', f'{peak_count}_peak_route')
            )
            channel = int(payload.get('signal_channel', plan.get('signal_channel', 1)))
            if channel not in range(4):
                raise ValueError('保存的ADC通道无效')
            selector = payload.get('feedback_selector')
            selector_index = self.analog_gain.findData(selector)
            if selector_index < 0:
                raise ValueError('保存的模拟档位无效')
            dense_selector = int(payload.get('dense_feedback_selector', selector))
            reference_selector = int(payload.get('reference_feedback_selector', selector))
            transimpedance_ohm(dense_selector)
            transimpedance_ohm(reference_selector)

            references = payload.get('reference_values')
            reference_wavelengths = payload.get('reference_wavelengths_nm')
            has_reference = bool(
                isinstance(references, list) and len(references) == point_count
                and np.all(np.isfinite(np.asarray(references, dtype=float)))
            )
            restored_ready = bool(payload.get('ready') and has_reference)
            manual_reference_reuse = bool(
                plan.get('manual_adjustment_reference_reused')
            )
            reference_gate_bypassed = bool(
                plan.get('reference_shape_gate_bypassed')
            )
            checks = []
            if restored_ready:
                for group in range(peak_count):
                    selected = plan['rows'][group * 5:group * 5 + 5]
                    check = assess_five_point_shape(
                        [row['measured_nm'] for row in selected],
                        references[group * 5:group * 5 + 5],
                    )
                    check['group'] = group + 1
                    checks.append(check)
                restored_ready = bool(
                    manual_reference_reuse or reference_gate_bypassed
                    or all(check['passed'] for check in checks)
                )

            self._restoring_session = True
            self.channel_combo.setCurrentIndex(self.channel_combo.findData(channel))
            self.analog_gain.setCurrentIndex(selector_index)
            self.latest_feedback_selector = int(selector)
            self.reference_feedback_selector = reference_selector
            self.dense_feedback_selector = dense_selector
            self.latest_channel = int(payload.get('latest_channel', channel))
            self.reference_channel = int(payload.get('reference_channel', channel))
            self.dense_channel = int(payload.get('dense_channel', channel))
            self.dense_acquisition_method = str(payload.get(
                'dense_acquisition_method', plan.get(
                    'dense_acquisition_method', 'stable_window'
                )
            ))
            dense_x = np.asarray(plan.get('reference_wavelengths_nm', []), dtype=float)
            self.dense_expected_points = int(payload.get(
                'dense_point_count', len(dense_x) if dense_x.ndim == 1 else 2001
            ) or 2001)
            self._spectrum_dirty = bool(payload.get(
                'unsaved_spectrum_changes', not payload.get('finger_record_name')
            ))
            self.digital_gain.setValue(float(payload.get('digital_display_gain', 1)))
            self.reference_gain.setValue(float(payload.get('reference_display_gain', 1)))
            self._set_display_mode(DEFAULT_DISPLAY_MODE, persist=False)
            self.first_delay.setValue(int(payload.get('first_delay_us', 350)))
            self.boundary_delay.setValue(int(payload.get('boundary_extra_us', 1800)))
            self.equal_interval_nm.setValue(float(payload.get('equal_interval_nm', .02)))
            self.equal_interval_settle.setValue(float(
                payload.get('equal_interval_settle_s', .2)
            ))
            self.plan = plan
            self.finger_record_name = str(payload.get('finger_record_name', '')).strip() or None
            self.finger_record_label.setText(
                f'当前手指记录：{self.finger_record_name}（{self.runtime_machine_label}参数）'
                if self.finger_record_name else
                f'当前手指记录：未命名（{self.runtime_machine_label}参数；自动保存会话）'
            )
            self.render_plan(ready=False)
            # A restart should visibly confirm which route was restored.  The
            # visibility switches are convenient during a live run, but
            # persisting both in the hidden state made a valid 45-point route
            # look as though it had disappeared after reopening the program.
            self.show_dense_selection.setChecked(
                True if restored_ready else bool(payload.get('show_dense_selection', True))
            )
            self.show_dense_failures.setChecked(
                bool(payload.get('show_dense_failures', True))
            )
            self.show_reference.setChecked(
                True if restored_ready and has_reference
                else bool(payload.get('show_reference', True))
            )
            waits = payload.get('point_delays_us')
            if (payload.get('use_point_waits') and isinstance(waits, list)
                    and len(waits) == point_count):
                command(self.plan['rows'], 32, 1, first_delay_us=650,
                        boundary_extra_us=0, point_delays_us=waits,
                        adc_channel=channel, feedback_selector=int(selector))
                self.point_waits = [int(value) for value in waits]
                self.point_wait_route = self.point_route_key()
                self.use_point_waits.setChecked(True)
            else:
                self.point_waits = None
                self.point_wait_route = None
                self.use_point_waits.setChecked(False)
            if has_reference:
                self.reference_values = [float(value) for value in references]
                if (isinstance(reference_wavelengths, list)
                        and len(reference_wavelengths) == point_count
                        and np.all(np.isfinite(np.asarray(reference_wavelengths, dtype=float)))):
                    self.reference_wavelengths = [float(value) for value in reference_wavelengths]
                else:
                    self.reference_wavelengths = [
                        float(row['measured_nm']) for row in self.plan['rows']
                    ]
                for index, value in enumerate(self.reference_values):
                    self.table.setItem(index, 5, QtWidgets.QTableWidgetItem(f'{value:g}'))
                self.redraw_values()
                self.fit_values()
            if restored_ready:
                self.plan['live_reference_fit_checks'] = checks
                self.plan_ready = True
                self.start_button.setEnabled(True)
                if manual_reference_reuse:
                    self.status.setText(
                        f'已恢复键盘调整后的{point_count}点，可直接开始临时测试；参考虚线仍是调整前数据。'
                        '只有主动点击“采集并绘制参考线”才会重新采集。'
                    )
                elif reference_gate_bypassed:
                    self.status.setText(
                        f'已恢复{point_count}点和参考线；参考采集的稳定/峰形检查未通过，但按设置仍可直接开始临时测试。'
                        '只有主动点击“采集并绘制参考线”才会重新采集。'
                    )
                else:
                    self.status.setText(
                        f'已自动恢复上次{peak_count}/{peak_count}通过的{point_count}点、密集光谱和参考线；可直接开始临时测试。'
                        '软件启动不会自动出光；峰位明显变化时请重新采集参考。'
                    )
            else:
                self.plan_ready = False
                self.start_button.setEnabled(False)
                self.status.setText(
                    f'已自动恢复上次{point_count}点、密集光谱'
                    + ('和未通过的参考线；' if has_reference else '；')
                    + f'该路线尚无{peak_count}/{peak_count}通过记录，请调整点位或重新采集参考线。'
                )
            return True
        except Exception as exc:
            self.status.setText(f'上次临时点表会话恢复失败，已保留默认路线：{exc}')
            return False
        finally:
            self._restoring_session = False

    def _migrate_latest_reference_line(self):
        """Adopt one 9/9 record made by the pre-persistence UI version."""

        candidates = sorted(
            self.last_session_path.parent.glob('reference_line_*.json'),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            return False
        try:
            reference_path = candidates[0]
            report = json.loads(reference_path.read_text(encoding='utf-8'))
            rows = report.get('rows', [])
            if (not report.get('complete') or not report.get('disarm_ack')
                    or not report.get('shutter_ack') or len(rows) != 45):
                return False
            import app_JDSU as app
            table = tuple(app.load_fullband_accuracy_table())
            selected, values, checks, peaks = [], [], [], []
            previous = -1
            for flat_index, recorded in enumerate(rows):
                index = int(recorded['index'])
                if not previous < index < len(table):
                    raise ValueError('参考线物理索引顺序无效')
                point = table[index]
                if (list(recorded.get('dac_codes', [])) != list(point.codes)
                        or not recorded.get('stable') or recorded.get('ch1_saturated')):
                    raise ValueError('参考线与当前2001点表不一致或读数无效')
                selected.append(dict(
                    group=flat_index // 5 + 1, index=int(point.index),
                    target_nm=float(point.target_nm), measured_nm=float(point.measured_nm),
                    codes=list(point.codes),
                ))
                values.append(float(recorded['ch1_adc_code']))
                previous = index
            for group in range(9):
                group_rows = selected[group * 5:group * 5 + 5]
                check = assess_five_point_shape(
                    [row['measured_nm'] for row in group_rows],
                    values[group * 5:group * 5 + 5],
                )
                check['group'] = group + 1
                checks.append(check)
                fitted_center = check.get('fitted_center_nm')
                peaks.append(float(fitted_center) if fitted_center is not None
                             and np.isfinite(fitted_center)
                             else float(group_rows[2]['measured_nm']))
            all_passed = all(check['passed'] for check in checks)
            plan = dict(
                schema='temporary_test_45_plan_v1', rows=selected, peaks_nm=peaks,
                reference_kind='restored_selected_reference',
                reference_source=str(reference_path.resolve()),
                optical_accuracy_verified=False, discarded_weak_peaks=0,
                spacing_basis='operator_selected_calibrated_table', hidden_points=0,
                baseline_fraction=None, soft_baseline=True, endpoint_checks=[],
                selection_fit_checks=checks, live_reference_fit_checks=checks,
                selection_validation_required=not all_passed, manual_selection=True,
            )
            dense_candidates = sorted(
                (path for path in self.last_session_path.parent.glob('reference_*.json')
                 if not path.name.startswith('reference_line_')
                 and path.stat().st_mtime <= reference_path.stat().st_mtime),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if dense_candidates:
                dense = json.loads(dense_candidates[0].read_text(encoding='utf-8'))
                dense_rows = dense.get('rows', [])
                if dense.get('complete') and len(dense_rows) >= 45:
                    plan['reference_wavelengths_nm'] = [
                        float(row['measured_wavelength_nm']) for row in dense_rows
                    ]
                    plan['reference_values'] = [
                        float(row['ch1_adc_code']) for row in dense_rows
                    ]
            plan = validate_plan_payload(plan)
            selector = int(report.get('ch1_feedback_selector', 2))
            selector_index = self.analog_gain.findData(selector)
            if selector_index < 0:
                raise ValueError('参考线模拟档位无效')
            self._restoring_session = True
            self.analog_gain.setCurrentIndex(selector_index)
            self.latest_feedback_selector = selector
            self.reference_feedback_selector = selector
            self.dense_feedback_selector = selector
            self.plan = plan
            self.render_plan(ready=False)
            self.reference_values = values
            self.reference_wavelengths = [float(row['measured_nm']) for row in selected]
            self.plan_ready = all_passed
            self.start_button.setEnabled(all_passed)
            for index, value in enumerate(values):
                self.table.setItem(index, 5, QtWidgets.QTableWidgetItem(f'{value:g}'))
            self.show_reference.setChecked(True)
            self.redraw_values()
            self.fit_values()
            if all_passed:
                self.status.setText(
                    '已从上次9/9通过的参考线自动恢复45点；可直接开始临时测试。'
                    '该记录已转换为新的自动保存会话。'
                )
            else:
                failed = '、'.join(
                    f"峰{check['group']}({check['reason']})"
                    for check in checks if not check['passed']
                )
                self.status.setText(
                    '已自动恢复上次45点和未通过的参考虚线：' + failed
                    + '；点位不会丢失，请单击选中并用左右方向键调整。'
                )
            self._restoring_session = False
            self._save_last_route_session(ready=all_passed)
            return True
        except Exception as exc:
            self.status.setText(f'旧版参考线自动恢复失败，已保留默认路线：{exc}')
            return False
        finally:
            self._restoring_session = False

    def refresh_wait_controls(self, _checked=False):
        idle = self.worker is None
        if idle and self.use_point_waits.isChecked() and self.plan is not None:
            self._ensure_point_waits_for_route()
        self.first_delay.setEnabled(idle and not self.use_point_waits.isChecked())
        self.boundary_delay.setEnabled(idle and not self.use_point_waits.isChecked())
        self.use_point_waits.setEnabled(idle)
        self.edit_point_waits.setEnabled(idle)
        self.test_switch_timing.setEnabled(idle and self.plan is not None)

    def _ensure_point_waits_for_route(self):
        """Bind a valid 45-value wait vector to the current route.

        Checking the option is intentionally sufficient to start a scan.  If
        the operator has not opened the editor yet, use the visible uniform
        and cross-peak waits as the initial per-point vector.  A route change
        gets a fresh vector instead of leaving a stale route-key mismatch.
        """
        if self.plan is None:
            return False
        key = self.point_route_key()
        point_count = self.point_count()
        if (isinstance(self.point_waits, list) and len(self.point_waits) == point_count
                and self.point_wait_route == key):
            return True
        waits = [
            int(self.first_delay.value())
            + (int(self.boundary_delay.value()) if index % 5 == 0 else 0)
            for index in range(point_count)
        ]
        self.point_waits = waits
        self.point_wait_route = key
        return True

    def point_route_key(self):
        return [(r['index'], tuple(r['codes'])) for r in self.plan['rows']]

    def configure_point_waits(self):
        if self.worker is not None or self.plan is None:
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('逐点等待：固定波长，仅改变采样时刻')
        layout = QtWidgets.QVBoxLayout(dialog)
        settle_hints = load_forward_settle_hints()
        layout.addWidget(QtWidgets.QLabel(
            '每点50～15000 µs（最大15 ms），25 µs步长；总等待不限，允许低于15 Hz。\n'
            '“稳定时间”来自既有板上高速采样，只作提示且不会自动改写本次等待；'
            '同目标表示此前一点与当前路线不同。'))
        source = QtWidgets.QLabel(settle_hints['source_label'])
        source.setStyleSheet('color: #6b7280;')
        layout.addWidget(source)
        point_count = self.point_count()
        table = QtWidgets.QTableWidget(point_count, 5)
        table.setHorizontalHeaderLabels([
            '峰/点', '正向跳转波长 nm', '物理索引',
            '正向跳转主边沿/稳定时间', '本次使用等待 µs'
        ])
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        spins = []
        values = self.point_waits if self.point_wait_route == self.point_route_key() else None
        rows = self.plan['rows']
        for i, row in enumerate(rows):
            previous = rows[i - 1] if i else rows[-1]
            source_nm = float(previous.get('measured_nm', previous['target_nm']))
            target_nm = float(row.get('measured_nm', row['target_nm']))
            path_prefix = '回绕 ' if i == 0 else ''
            table.setItem(i, 0, QtWidgets.QTableWidgetItem(f'{i//5+1}/{i%5+1}'))
            path_item = QtWidgets.QTableWidgetItem(
                f'{path_prefix}{source_nm:.5f} → {target_nm:.5f}'
            )
            path_item.setToolTip(
                f"物理索引 {previous['index']} → {row['index']}；"
                + ('本行是上一帧末点到新一帧首点的回绕。' if i == 0
                   else '本行是从小波长到大波长的顺序跳转。')
            )
            table.setItem(i, 1, path_item)
            table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(row['index'])))

            hint = forward_settle_hint_for(rows, i)
            if hint is None:
                hint_text = '无对应实测'
                hint_tooltip = '现有板上高速采样中没有命中该目标物理索引。'
            elif not hint['meaningful']:
                hint_text = '无可判定边沿'
                hint_tooltip = '该次CH1变化不足20码，不能可靠计算边沿后的稳定时间。'
            elif hint['settle_available'] and hint['settle_us'] is None:
                hint_text = '>100 ms（未达5%）'
                hint_tooltip = '在100 ms观测范围内未持续进入最终值±5%的稳态带。'
            elif hint['settle_available']:
                settle_us = float(hint['settle_us'])
                settle_text = (f'{settle_us / 1000:g} ms' if settle_us >= 1000
                               else f'{settle_us:g} µs')
                hint_text = f'稳定 {settle_text}'
                hint_tooltip = 'CH1进入并保持在最终值±5%稳态带所需时间。'
            elif hint.get('edge_complete_from_switch_begin_us') is not None:
                edge_complete = float(hint['edge_complete_from_switch_begin_us'])
                edge_text = (f'{edge_complete / 1000:g} ms'
                             if edge_complete >= 1000 else f'{edge_complete:g} µs')
                hint_text = f'主边沿 {edge_text}'
                hint_tooltip = (
                    '本轮前向审计只判定主边沿是否完成，未输出5%稳态时间；'
                    '这里显示从开始切换到主边沿完成的板上时间。'
                )
            else:
                hint_text = '未输出稳定时间'
                hint_tooltip = '该次实测没有可显示的5%稳态或主边沿完成时间。'
            if hint is not None:
                if hint.get('match') == 'target_only':
                    hint_text += '（同目标）'
                    hint_tooltip += (
                        f" 实测来源索引为 {hint['source_index']}，"
                        f"当前来源索引为 {previous['index']}，因此不是同一路径。"
                    )
                elif hint.get('match') == 'exact':
                    hint_tooltip += ' 来源与目标物理索引均和当前路线完全相同。'
                edge_us = hint.get('edge_us')
                if edge_us is not None:
                    hint_tooltip += f' 实测10–90%主边沿约 {float(edge_us):g} µs。'
                hint_tooltip += f" 数据文件：{hint.get('source_name', '未知')}。"
            hint_item = QtWidgets.QTableWidgetItem(hint_text)
            hint_item.setToolTip(hint_tooltip)
            table.setItem(i, 3, hint_item)
            spin = QtWidgets.QSpinBox()
            spin.setRange(50, 15000)
            spin.setSingleStep(25)
            spin.setValue(values[i] if values is not None else
                          self.first_delay.value() + (self.boundary_delay.value() if i%5 == 0 else 0))
            table.setCellWidget(i, 4, spin)
            spins.append(spin)
        table.setColumnWidth(0, 58)
        table.setColumnWidth(1, 245)
        table.setColumnWidth(2, 78)
        table.setColumnWidth(3, 190)
        table.setColumnWidth(4, 155)
        table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(table)
        status = QtWidgets.QLabel()
        layout.addWidget(status)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        def accept():
            waits = [spin.value() for spin in spins]
            try:
                command(self.plan['rows'], 32, 1, first_delay_us=650,
                        boundary_extra_us=0, point_delays_us=waits)
            except ValueError as exc:
                status.setText(str(exc))
                return
            self.point_waits = waits
            self.point_wait_route = self.point_route_key()
            self.use_point_waits.setChecked(True)
            self._save_last_route_session()
            dialog.accept()
        buttons.accepted.connect(accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.resize(930, 720)
        dialog.exec_()

    def set_dac_label(self, _value):
        pass

    def invalidate_plan(self):
        self.plan = None
        self.plan_ready = False
        self.finger_record_name = None
        self.finger_record_label.setText('当前手指记录：未命名（未载入保存记录）')
        self.save_finger_button.setEnabled(False)
        self.clear_peak_fits('点表已失效')
        self.clear_reference_line()
        self.reference_line_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.status.setText("间隔已变更，请重新生成临时点表")

    def prepare(self, x, y, source, *, failed_points=None, selection_values=None,
                mark_dirty=True):
        import app_JDSU as app
        dense_x = np.asarray(x, dtype=float)
        dense_y = np.asarray(y, dtype=float)
        selection_y = dense_y if selection_values is None else np.asarray(
            selection_values, dtype=float
        )
        if selection_y.shape != dense_y.shape:
            raise ValueError('选点谱线与密集采集长度不一致')
        selection_feedback_selector = self.effective_feedback_selector()
        self.plan = select_points(
            app.load_fullband_accuracy_table(), x, selection_y,
            feedback_selector=selection_feedback_selector,
        )
        self.plan['reference_source'] = source
        self.plan['reference_created_s'] = time.time()
        self.plan['reference_wavelengths_nm'] = dense_x.tolist()
        self.plan['reference_values'] = dense_y.tolist()
        self.plan['signal_channel'] = self.selected_channel()
        self.plan['signal_channel_name'] = self.channel_name()
        self.plan['dense_acquisition_method'] = self.dense_acquisition_method
        self.plan['dense_point_count'] = int(len(dense_x))
        self.dense_wavelengths = dense_x
        self.dense_values = dense_y
        self.dense_feedback_selector = selection_feedback_selector
        self.dense_channel = self.selected_channel()
        self.dense_source = str(source)
        self.dense_failed_points = list(failed_points or [])
        if mark_dirty:
            self._spectrum_dirty = True
        self.plan['dense_failed_points'] = list(self.dense_failed_points)
        if selection_values is not None:
            self.plan['dense_failed_values_excluded_from_selection'] = True
        self.finger_record_name = None
        self.finger_record_label.setText('当前手指记录：未命名（本次新采集，尚未保存）')
        self.point_waits = None
        self.point_wait_route = None
        self.use_point_waits.setChecked(False)
        self.render_plan(ready=False)
        self._save_last_route_session(ready=False)

    def render_plan(self, *, ready=True):
        self._sync_dynamic_ui()
        point_count = self.point_count()
        peak_count = self.peak_count()
        dense_x = np.asarray(self.plan.get('reference_wavelengths_nm', []), dtype=float)
        dense_y = np.asarray(self.plan.get('reference_values', []), dtype=float)
        if (dense_x.ndim == 1 and dense_y.shape == dense_x.shape and len(dense_x) >= point_count
                and np.all(np.isfinite(dense_x)) and np.all(np.isfinite(dense_y))
                and np.all(np.diff(dense_x) > 0)):
            self.dense_wavelengths = dense_x
            self.dense_values = dense_y
            self.dense_source = str(self.plan.get('reference_source', 'dense spectrum'))
        self.dense_failed_points = list(self.plan.get('dense_failed_points', []))
        self.dense_channel = int(self.plan.get('signal_channel', self.selected_channel()))
        self.plan_ready = bool(ready)
        self.fit_failure_streaks = [0] * peak_count
        self.save_finger_button.setEnabled(True)
        self.load_finger_button.setEnabled(True)
        self.clear_peak_fits('等待当前点表的实时帧')
        self.clear_reference_line()
        self.latest_values = None
        self.fit_pending = True
        for curve in self.curves:
            curve.clear()
        for i, row in enumerate(self.plan['rows']):
            for col, value in enumerate((row['group'], row['index'], f"{row['target_nm']:.4f}", f"{row['measured_nm']:.6f}", '—')):
                self.table.setItem(i, col, QtWidgets.QTableWidgetItem(str(value)))
        self.start_button.setEnabled(self.plan_ready)
        self.reference_line_button.setEnabled(True)
        self.redraw_dense_selection()
        self.fit_peak_view()
        if self.dense_values is not None:
            self.fit_values()
        self.status.setText(f"已选定{peak_count}×5点：" + ', '.join(f'{v:.3f}' for v in self.plan['peaks_nm']) + " nm；精度待实测")

    def _clear_selection_markers(self):
        for marker in self.selection_markers:
            self.plot.removeItem(marker)
        self.selection_markers = []
        self.selected_selection_point = None

    def _dense_value_at(self, wavelength_nm):
        if self.dense_wavelengths is None or self.dense_values is None:
            return 0.0
        return float(np.interp(wavelength_nm, self.dense_wavelengths, self.dense_values))

    def redraw_dense_selection(self, _checked=None):
        live_rows = self._live_dense_rows if self._dense_capture_active else []
        if live_rows:
            plot_x = np.asarray([
                row['measured_wavelength_nm'] for row in live_rows
            ], dtype=float)
            plot_y = np.asarray([
                row.get('signal_adc_code', row.get('ch1_adc_code', np.nan))
                for row in live_rows
            ], dtype=float)
            failed_points = [row for row in live_rows if not row.get('point_passed')]
            plot_selector = self.effective_feedback_selector()
        else:
            plot_x = self.dense_wavelengths
            plot_y = self.dense_values
            failed_points = self.dense_failed_points
            plot_selector = self.dense_feedback_selector
        available = plot_x is not None and plot_y is not None and len(plot_x)
        self.show_dense_selection.setEnabled(available)
        visible = bool(self.show_dense_selection.isChecked() and available)
        if not visible:
            self.dense_curve.clear()
            self.dense_failed_curve.clear()
            self._clear_selection_markers()
            return
        self.dense_curve.setData(
            plot_x,
            self._plot_display_values(
                plot_y, selector=plot_selector
            ),
        )
        failed_x, failed_y = [], []
        finite_y = np.asarray(plot_y, dtype=float)
        finite_indices = np.flatnonzero(np.isfinite(finite_y))
        for row in failed_points:
            failed_x.append(float(row['measured_wavelength_nm']))
            value = float(row.get('signal_adc_code', row.get('ch1_adc_code', np.nan)))
            if not np.isfinite(value):
                position = int(np.searchsorted(plot_x, failed_x[-1]))
                if finite_indices.size:
                    nearest = int(finite_indices[np.argmin(abs(finite_indices - position))])
                    value = float(finite_y[nearest])
                else:
                    value = 0.0
            failed_y.append(value)
        if failed_x and self.show_dense_failures.isChecked():
            self.dense_failed_curve.setData(
                failed_x,
                self._plot_display_values(failed_y, selector=plot_selector),
            )
        else:
            self.dense_failed_curve.clear()
        # While a replacement dense spectrum is still arriving, the markers
        # belong to the previous completed plan and are misleading.  Show the
        # live curve alone; the newly selected markers are drawn by prepare()
        # as soon as the scan is stopped or completed.
        if self._dense_capture_active:
            self._clear_selection_markers()
            return
        if self.plan is None:
            self._clear_selection_markers()
            return
        point_count = self.point_count()
        if len(self.selection_markers) != point_count:
            self._clear_selection_markers()
            for flat_index in range(point_count):
                group, point = divmod(flat_index, 5)
                color = pg.intColor(group, 9)
                marker = KeyboardSelectionTarget(
                    size=11, symbol='o', movable=False,
                    pen=pg.mkPen(color, width=2), brush=pg.mkBrush(0, 0, 0, 30),
                    hoverPen=pg.mkPen((255, 255, 255), width=2),
                    hoverBrush=pg.mkBrush(color),
                )
                marker.selection_color = color
                marker.setZValue(30)
                marker.setToolTip(
                    f'峰{group + 1} 第{point + 1}点；单击选中后按键盘左/右键移动'
                )
                marker.selected.connect(
                    lambda item, g=group, p=point: self.select_selection_point(g, p)
                )
                self.plot.addItem(marker)
                self.selection_markers.append(marker)
        self._updating_selection_markers = True
        try:
            for marker, row in zip(self.selection_markers, self.plan['rows']):
                display_y = float(self._plot_display_values([
                    self._dense_value_at(row['measured_nm'])
                ], selector=self.dense_feedback_selector)[0])
                marker.setPos(
                    row['measured_nm'], display_y
                )
                marker.show()
        finally:
            self._updating_selection_markers = False
        self._refresh_selection_marker_styles()

    @QtCore.pyqtSlot(object)
    def update_dense_reference_point(self, payload):
        """Append and draw one dense point on the GUI thread."""
        if not self._dense_capture_active:
            return
        row = dict(payload.get('row', {}))
        self._live_dense_rows.append(row)
        self.show_dense_selection.setChecked(True)
        self.redraw_dense_selection()
        completed = int(payload.get('completed', len(self._live_dense_rows)))
        total = int(payload.get('total', 2001))
        # dBm values are negative.  Before the first live samples the plot may
        # still have an ADC/empty range such as 0..4, which clips the whole
        # trace against the lower border and makes a real spectrum look flat.
        # Rescale at a bounded cadence so live drawing stays responsive.
        range_refresh_interval = max(10, total // 100)
        if completed == 1 or completed == total or completed % range_refresh_interval == 0:
            self.fit_values()
        failed = sum(not item.get('point_passed') for item in self._live_dense_rows)
        suffix = f'；已标红{failed}个未通过点' if failed else ''
        self.status.setText(
            f'{self.channel_name()}密集光谱实时绘制 {completed}/{total}{suffix}。'
            '若已看到9个峰，可点“停止绘制”尝试提前选点。'
        )

    def stop_dense_reference(self):
        if self.worker is None or not self._dense_capture_active:
            return
        self.worker.stop()
        self.stop_dense_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.status.setText(
            '正在停止密集绘制并安全关光；已采部分会继续尝试识别峰和选点…'
        )

    def _refresh_selection_marker_styles(self):
        for flat_index, marker in enumerate(self.selection_markers):
            selected = self.selected_selection_point == divmod(flat_index, 5)
            color = marker.selection_color
            marker.setPen(pg.mkPen((255, 255, 255) if selected else color,
                                   width=3 if selected else 2))
            marker.setBrush(pg.mkBrush(color if selected else (0, 0, 0, 30)))
            marker.update()

    def select_selection_point(self, group, point):
        if (self.plan is None or not 0 <= group < self.peak_count() or not 0 <= point < 5
                or len(self.selection_markers) != self.point_count()):
            return False
        self.selected_selection_point = (group, point)
        self._refresh_selection_marker_styles()
        row = self.plan['rows'][group * 5 + point]
        self.status.setText(
            f"已选中峰{group + 1}第{point + 1}点：索引{row['index']}，"
            f"{row['measured_nm']:.6f} nm；按键盘 ←/→ 移动一个标定点。"
        )
        self.setFocus(QtCore.Qt.MouseFocusReason)
        return True

    def keyPressEvent(self, event):
        if event.key() in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Right):
            if self.selected_selection_point is None:
                self.status.setText('请先单击密集光谱上的一个采样点，再按键盘 ←/→ 调整。')
            elif self.worker is not None:
                self.status.setText('采集过程中不能调整点位；请先停止并关光。')
            else:
                self.move_selected_point(-1 if event.key() == QtCore.Qt.Key_Left else 1)
            event.accept()
            return
        super().keyPressEvent(event)

    def move_selected_point(self, step):
        if (self.selected_selection_point is None or self.plan is None
                or step not in (-1, 1)):
            return False
        group, point = self.selected_selection_point
        flat_index = group * 5 + point
        indices = [int(row['index']) for row in self.plan['rows']]
        lower = indices[flat_index - 1] + 1 if flat_index else 0
        upper = (indices[flat_index + 1] - 1
                 if flat_index < self.point_count() - 1 else 2000)
        requested = indices[flat_index] + step
        if not lower <= requested <= upper:
            self.status.setText(
                f'峰{group + 1}第{point + 1}点不能继续向'
                + ('左' if step < 0 else '右')
                + '移动：相邻采样点之间必须保留严格顺序。'
            )
            return False
        import app_JDSU as app
        table = tuple(app.load_fullband_accuracy_table())
        changed = self.apply_dragged_selection(
            group, point, float(table[requested].measured_nm)
        )
        if changed:
            self.selected_selection_point = (group, point)
            self._refresh_selection_marker_styles()
        return changed

    def apply_dragged_selection(self, group, point, requested_wavelength_nm):
        """Snap one manually adjusted point without imposing equal spacing."""

        if (self.plan is None or self.dense_wavelengths is None
                or self.dense_values is None or not 0 <= group < self.peak_count()
                or not 0 <= point < 5):
            return False
        import app_JDSU as app
        table = tuple(app.load_fullband_accuracy_table())
        measured = np.asarray([entry.measured_nm for entry in table], dtype=float)
        requested = int(np.argmin(abs(measured - requested_wavelength_nm)))
        groups = [
            [int(row['index']) for row in self.plan['rows'][start:start + 5]]
            for start in range(0, self.point_count(), 5)
        ]
        flat = [index for indices in groups for index in indices]
        flat_index = group * 5 + point
        lower = flat[flat_index - 1] + 1 if flat_index else 0
        upper = (flat[flat_index + 1] - 1
                 if flat_index < self.point_count() - 1 else len(table) - 1)
        requested = min(max(requested, lower), upper)
        if requested == groups[group][point]:
            self.redraw_dense_selection()
            return False
        was_ready = bool(self.plan_ready)
        previous_reference = (list(self.reference_values)
                              if isinstance(self.reference_values, list)
                              and len(self.reference_values) == self.point_count() else None)
        previous_reference_wavelengths = (list(self.reference_wavelengths)
                                          if isinstance(self.reference_wavelengths, list)
                                          and len(self.reference_wavelengths) == self.point_count() else None)
        groups[group][point] = requested
        try:
            plan = build_manual_dense_plan(
                table, groups, self.dense_wavelengths, self.dense_values,
                source=self.dense_source or self.plan.get('reference_source', 'dense spectrum'),
            )
        except Exception as exc:
            self.status.setText(f'手动点位未采用：{exc}')
            self.redraw_dense_selection()
            return False
        plan['signal_channel'] = self.selected_channel()
        plan['signal_channel_name'] = self.channel_name()
        plan['dense_acquisition_method'] = self.dense_acquisition_method
        plan['dense_point_count'] = int(len(self.dense_wavelengths))
        plan['manual_adjustment_reference_reused'] = was_ready
        plan['manual_adjustment_reference_note'] = (
            'operator authorized direct scanning after keyboard point adjustment; '
            'reference line remains from the pre-adjustment route until explicitly reacquired'
        )
        self.plan = plan
        self._spectrum_dirty = True
        self.point_waits = None
        self.point_wait_route = None
        self.use_point_waits.setChecked(False)
        self.render_plan(ready=was_ready)
        if previous_reference is not None and previous_reference_wavelengths is not None:
            self.reference_values = previous_reference
            self.reference_wavelengths = previous_reference_wavelengths
            for index, value in enumerate(previous_reference):
                self.table.setItem(index, 5, QtWidgets.QTableWidgetItem(f'旧:{value:g}'))
            self.redraw_values()
        check = self.plan['selection_fit_checks'][group]
        preview = '密集谱五点形状通过' if check['passed'] else f"密集谱预览警告：{check['reason']}"
        selected = self.plan['rows'][group * 5:group * 5 + 5]
        self.status.setText(
            f'已手动修改峰{group + 1}第{point + 1}点：'
            + ', '.join(f"{row['target_nm']:.2f}" for row in selected)
            + f' nm；{preview}。修改前参考虚线已保留，未自动重采；'
            + ('可直接开始临时测试。只有主动点击“采集并绘制参考线”才会更新参考线。'
               if was_ready else
               '当前路线原本尚不可开始；移动点位不会自动取得开始权限。')
        )
        self._save_last_route_session(ready=was_ready)
        return True

    def fit_peak_view(self, _index=None):
        if self.plan is None:
            return
        index = self.peak_view.currentIndex()
        rows = self.plan['rows'] if index == 0 else self.plan['rows'][(index-1)*5:index*5]
        if index == 0 and self.show_dense_selection.isChecked() and self.dense_wavelengths is not None:
            left, right = self.dense_wavelengths[0], self.dense_wavelengths[-1]
        else:
            left, right = rows[0]['measured_nm'], rows[-1]['measured_nm']
        margin = max(.04, (right-left)*.08)
        self.plot.setXRange(left-margin, right+margin, padding=0)

    def prepare_current(self):
        try:
            pages = [self.controller.page_equal_interval, self.controller.page_unlimited_accuracy]
            valid = [p for p in pages if p.completed_points == len(p.points) and len(p.points) >= 45
                     and p.worker is None and np.all(p.stable_flags)]
            if not valid:
                raise ValueError("请先完成一张全部稳定的密集光谱（等间隔或无限准确模式）")
            page = max(valid, key=lambda p: p.scan_started_at or 0)
            self.prepare(page.wavelengths, page.values[1], page.mode_display_name)
        except Exception as exc:
            self.plan = None
            self.plan_ready = False
            self.clear_reference_line()
            self.reference_line_button.setEnabled(False)
            self.start_button.setEnabled(False)
            self.status.setText(str(exc))

    def load_reference(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择密集光谱CSV或完整临时标定JSON", str(Path(__file__).parent), "参考或标定 (*.csv *.json)")
        if not path:
            return
        try:
            if Path(path).suffix.lower() == '.json':
                payload = json.loads(Path(path).read_text(encoding='utf-8'))
                self.plan = (load_saved_plan(path)
                             if payload.get('schema') in (
                                 'temporary_test_45_plan_v1',
                                 'temporary_test_variable_plan_v2')
                             else load_calibrated_plan(path))
                self.finger_record_name = None
                self.finger_record_label.setText(
                    '当前手指记录：未命名（外部导入，尚未保存）'
                )
                self.render_plan(ready=False)
                self.status.setText(
                    f'已导入{self.point_count()}点路线；峰位可能随时间漂移，请先采集当前'
                    f'{self.point_count()}点慢速参考线并通过{self.peak_count()}峰形状验证。'
                )
                return
            with open(path, encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.prepare([float(r['calibrated_wavelength_nm']) for r in rows],
                         [float(r['ch1_v']) for r in rows], str(Path(path).resolve()))
        except Exception as exc:
            self.plan = None
            self.plan_ready = False
            self.clear_reference_line()
            self.reference_line_button.setEnabled(False)
            self.start_button.setEnabled(False)
            self.status.setText(f"导入失败：{exc}")

    def reference_ready(self, report):
        self._dense_capture_active = False
        rows = list(report.get('rows', []))
        self._live_dense_rows = []
        signal_channel = int(report.get('signal_channel', self.selected_channel()))
        self.dense_channel = signal_channel
        self.dense_acquisition_method = str(report.get(
            'acquisition_method', 'stable_window'
        ))
        self.dense_expected_points = int(report.get(
            'requested_point_count', len(rows) or 2001
        ))
        failed_points = [{
            'index': int(row.get('index', -1)),
            'measured_wavelength_nm': float(row.get(
                'measured_wavelength_nm', np.nan
            )),
            'signal_adc_code': float(row.get(
                'signal_adc_code', row.get('ch1_adc_code', np.nan)
            )),
            'signal_channel': signal_channel,
            'failure_reason': str(row.get('failure_reason', '稳定性未通过')),
            'point_passed': False,
        } for row in rows
            if not row.get('point_passed', row.get('stable', False))]
        usable_rows = [
            row for row in rows
            if np.isfinite(row.get('measured_wavelength_nm', np.nan))
            and np.isfinite(row.get(
                'signal_adc_code', row.get('ch1_adc_code', np.nan)
            ))
        ]
        # Adopt the newly observed spectrum even when automatic peak selection
        # cannot yet succeed.  This lets the operator see where an early stop
        # occurred while the previous usable route remains intact.
        if usable_rows:
            self.dense_wavelengths = np.asarray([
                row['measured_wavelength_nm'] for row in usable_rows
            ], dtype=float)
            self.dense_values = np.asarray([
                row.get('signal_adc_code', row.get('ch1_adc_code'))
                for row in usable_rows
            ], dtype=float)
            self.dense_source = str(self.worker.output)
            self.dense_feedback_selector = int(
                report.get('feedback_selector', self.effective_feedback_selector())
            )
            self.dense_failed_points = failed_points
            self._spectrum_dirty = True
            self.save_finger_button.setEnabled(True)
        try:
            if len(usable_rows) < 5:
                raise ValueError('已采有效点不足5个')
            dense_x = np.asarray([
                row['measured_wavelength_nm'] for row in usable_rows
            ], dtype=float)
            dense_y = np.asarray([
                row.get('signal_adc_code', row.get('ch1_adc_code'))
                for row in usable_rows
            ], dtype=float)
            passed = np.asarray([
                bool(row.get('point_passed', row.get('stable', False)))
                for row in usable_rows
            ], dtype=bool)
            selection_y = dense_y.copy()
            # An unstable-but-finite sample is still drawn at its raw value
            # and marked red.  It must not create/delete/move a detected FBG
            # peak, so only the selector sees a local interpolation through
            # accepted neighbours.  The subsequent 15/45-point reference is
            # the independent live remeasurement of any selected wavelength.
            if np.any(~passed) and np.count_nonzero(passed) >= 2:
                selection_y[~passed] = np.interp(
                    dense_x[~passed], dense_x[passed], dense_y[passed]
                )
            self.prepare(dense_x, dense_y, str(self.worker.output),
                         failed_points=failed_points,
                         selection_values=(selection_y if np.any(~passed) else None))
            self.dense_feedback_selector = int(
                report.get('feedback_selector', self.effective_feedback_selector())
            )
            # Dense forward acquisition proves the 2001 individual points,
            # not the sparse 45-point entry path.  Require the dedicated slow
            # reference to pass all nine five-point shapes before realtime
            # scanning can be enabled.
            self.plan_ready = False
            self.start_button.setEnabled(False)
            self.reference_line_button.setEnabled(True)
            self.show_dense_selection.setChecked(True)
            self.redraw_dense_selection()
            total = self.dense_expected_points
            completion = (
                f'完成{len(rows)}/{total}点' if report.get('complete')
                else f'提前停止于{len(rows)}/{total}点'
            )
            method = (
                f'等间隔单帧方案（{float(report.get("settle_s", .2)):.3f} s/点）'
                if self.dense_acquisition_method == 'equal_interval_single'
                else '稳定窗口方案'
            )
            quality = (
                f'，{len(failed_points)}个未通过点已红色标注但未中断后续采集'
                if failed_points else '，未发现未通过点'
            )
            self.status.setText(
                f'{self.channel_name(signal_channel)}密集光谱·{method}{completion}{quality}；'
                f'已按dBm光强识别{self.peak_count()}峰并选出'
                f'{self.point_count()}点。未通过的密集点不影响继续采集当前'
                f'{self.point_count()}点参考线或执行开关速度测试。'
            )
            self._save_last_route_session(ready=False)
        except Exception as exc:
            self.redraw_dense_selection()
            previous = '；已保留上一张可用临时点表' if self.plan is not None else ''
            self.status.setText(
                f'已绘制{len(rows)}个{self.channel_name(signal_channel)}密集点'
                f'（其中{len(failed_points)}个未通过已标红），'
                f'但当前范围还不足以自动选点：{exc}{previous}。'
            )

    def clear_reference_line(self, _value=None):
        if self.plan is None:
            self.clear_peak_fits('等待有效点表及实时帧')
        self.reference_values = None
        self.reference_wavelengths = None
        for curve in self.reference_curves:
            curve.clear()
        for i in range(self.point_count()):
            self.table.setItem(i, 5, QtWidgets.QTableWidgetItem('—'))

    def invalidate_shape_verification(self, _value=None):
        if self.plan is None:
            return
        self.plan_ready = False
        self.fit_failure_streaks = [0] * self.peak_count()
        self.start_button.setEnabled(False)
        self._save_last_route_session(ready=False)

    def _replace_failed_groups_with_validated_route(self, failed_groups, live_values):
        """Rebuild only failed peaks from the last real 9/9 sparse route.

        This is deliberately a retry plan, not an acceptance shortcut.  The
        returned route remains disabled until a second slow reference proves
        all nine raw five-point shapes on the replacement DAC sequence.
        """

        if (self.plan is None
                or self.peak_count() != 9
                or self.plan.get('reference_kind') != 'dense_CH1'
                or self.selected_channel() != 1
                or self.plan.get('manual_selection')
                or not self.plan.get('reference_source')
                or self.validated_route_indices is None
                or self.validated_route_values is None):
            return False
        groups = [
            [int(row['index']) for row in self.plan['rows'][start:start + 5]]
            for start in range(0, 45, 5)
        ]
        values = list(live_values)
        replaced = []
        for group in failed_groups:
            offset = (int(group) - 1) * 5
            fallback = list(self.validated_route_indices[int(group) - 1])
            if groups[int(group) - 1] == fallback:
                continue
            groups[int(group) - 1] = fallback
            values[offset:offset + 5] = self.validated_route_values[offset:offset + 5]
            replaced.append(int(group))
        if not replaced:
            return False

        import app_JDSU as app
        replacement = build_selected_plan(
            app.load_fullband_accuracy_table(),
            groups,
            values,
            source=(
                '当前密集谱 + 稀疏失败峰回退至最近9/9实测路线：'
                + str(self.validated_route_source)
            ),
        )
        if self.dense_wavelengths is not None and self.dense_values is not None:
            replacement['reference_wavelengths_nm'] = self.dense_wavelengths.tolist()
            replacement['reference_values'] = self.dense_values.tolist()
        replacement['sparse_route_fallback'] = {
            'replaced_groups': replaced,
            'failed_reference_values': list(live_values),
            'validated_route_source': str(self.validated_route_source),
            'requires_new_live_reference': True,
            'acceptance_gate_relaxed': False,
        }
        self.plan = replacement
        self.point_waits = None
        self.point_wait_route = None
        self.use_point_waits.setChecked(False)
        self.render_plan(ready=False)
        groups_text = '、'.join(str(group) for group in replaced)
        self.status.setText(
            f'当前参考线的峰{groups_text}在稀疏路径下失配；已仅替换这些峰为最近一次真实9/9通过的位置，'
            '并用最新版2001点重新绑定DAC。请再次采集当前45点参考线；仍须9/9通过才可开始。'
        )
        self._save_last_route_session(ready=False)
        return True

    def reference_line_ready(self, report):
        try:
            rows = report.get('rows', [])
            point_count = self.point_count()
            peak_count = self.peak_count()
            signal_channel = int(report.get('signal_channel', self.selected_channel()))
            feedback_selector = int(report.get(
                'feedback_selector', report.get(
                    'ch1_feedback_selector',
                    self.effective_feedback_selector(signal_channel)
                )
            ))
            if (not report.get('complete') or not report.get('disarm_ack') or not report.get('shutter_ack')
                    or self.plan is None or len(rows) != point_count
                    or signal_channel != self.selected_channel()
                    or feedback_selector != self.effective_feedback_selector(signal_channel)):
                raise ValueError('参考未完整完成、停光未确认或模拟档位不一致')
            quality_issues = []
            for row, selected in zip(rows, self.plan['rows']):
                if (row['index'] != selected['index'] or row['dac_codes'] != selected['codes']
                        or not np.isfinite(row.get(
                            'signal_adc_code', row.get('ch1_adc_code', np.nan)
                        ))
                        or not 0 <= row.get(
                            'signal_adc_code', row.get('ch1_adc_code', np.nan)
                        ) <= 4095):
                    raise ValueError('参考点不匹配或没有有效ADC读数')
                if not row.get('stable'):
                    quality_issues.append(f"点{row['index']}稳定性未通过")
                if row.get('signal_saturated') or row.get(
                        'signal_adc_code', row.get('ch1_adc_code', 0)) >= 4080:
                    quality_issues.append(f"点{row['index']}可能饱和")
            self.reference_values = [
                row.get('signal_adc_code', row.get('ch1_adc_code')) for row in rows
            ]
            self._spectrum_dirty = True
            self.reference_feedback_selector = feedback_selector
            self.reference_channel = signal_channel
            self.reference_wavelengths = [
                float(row['measured_nm']) for row in self.plan['rows']
            ]
            # This path only runs after the operator explicitly pressed the
            # reference-line acquisition button.  The new 45-point reference
            # now supersedes any pre-adjustment line that was being reused.
            self.plan.pop('manual_adjustment_reference_reused', None)
            self.plan.pop('manual_adjustment_reference_note', None)
            for i, value in enumerate(self.reference_values):
                self.table.setItem(i, 5, QtWidgets.QTableWidgetItem(f'{value:g}'))
            self.show_reference.setChecked(True)
            self.redraw_values()
            self.fit_values()
            checks = []
            for group in range(peak_count):
                start = group * 5
                selected = self.plan['rows'][start:start + 5]
                check = assess_five_point_shape(
                    [row['measured_nm'] for row in selected],
                    self.reference_values[start:start + 5],
                )
                check['group'] = group + 1
                checks.append(check)
            self.plan['live_reference_fit_checks'] = checks
            failed = [check for check in checks if not check['passed']]
            if failed or quality_issues:
                details = '、'.join(f"峰{check['group']}({check['reason']})" for check in failed)
                issues = '、'.join(quality_issues[:6])
                reasons = '、'.join(value for value in (details, issues) if value)
                self.plan['reference_shape_gate_bypassed'] = True
                self.plan['reference_line_quality_issues'] = quality_issues
                self.plan_ready = True
                self.start_button.setEnabled(True)
                self.status.setText(
                    f'{point_count}点参考线已绘制，但稳定/峰形检查未通过：'
                    + (reasons or '未通过')
                    + '；按设置仍可直接开始临时测试。'
                    '只有主动点击“采集并绘制参考线”才会再次采集。'
                )
                self._save_last_route_session(ready=True)
                return
            self.plan.pop('reference_shape_gate_bypassed', None)
            self.plan.pop('reference_line_quality_issues', None)
            self.plan_ready = True
            self.fit_failure_streaks = [0] * peak_count
            self.start_button.setEnabled(True)
            for i,value in enumerate(self.reference_values):
                self.table.setItem(i,5,QtWidgets.QTableWidgetItem(f'{value:g}'))
            self.show_reference.setChecked(True)
            self.redraw_values()
            self.fit_values()
            self.status.setText(
                f'{point_count}点慢速参考线已绘制（虚线），{peak_count}峰形状余量通过并已停光；'
                '可开始实时扫描。'
            )
            self._save_last_route_session(ready=True)
        except Exception as exc:
            self.plan_ready = bool(
                self._reference_line_prior_ready
                or (isinstance(self.reference_values, list)
                    and len(self.reference_values) == self.point_count())
            )
            self.start_button.setEnabled(self.plan_ready)
            if self.plan_ready:
                self.status.setText(
                    f'本次{self.point_count()}点参考采集未采用：{exc}；已保留原有路线，仍可直接开始临时测试。'
                    '如需更新参考线，请再次点击采集按钮。'
                )
                self._save_last_route_session(ready=True)
            else:
                if self.reference_values is None:
                    self.clear_reference_line()
                self.status.setText(f'参考线未采用：{exc}')
                self._save_last_route_session(ready=False)

    def start_switch_timing(self, _checked=False):
        """Run three forward-only board-timestamped 90% timing passes."""
        import app_JDSU as app
        if self.selected_channel() != 1:
            self.status.setText('当前高速开关诊断固件仅测CH1；请切换到CH1后再测试90%时间。')
            return
        if self.demo:
            self.status.setText('演示模式不会启动真实开关速度测试')
            return
        if self.worker is not None or self.plan is None:
            return
        if not app.switch_mode_enable or app.ser_open:
            self.status.setText('请先停止其他采集')
            return
        if not temporary_transport_supported(app.ser):
            self.status.setText('开关速度测试需要真实USB或LAN局域网全功能通道')
            return
        try:
            command(
                self.plan['rows'], 32, 1,
                feedback_selector=int(self.analog_gain.currentData()),
            )
            app.ensure_serial_open()
        except Exception as exc:
            self.status.setText(f'开关测试启动前检查失败：{exc}')
            return
        output = output_path(
            'temporary_test', f'switch_90pct_{time.time_ns()}.json'
        )
        app.switch_mode_enable = False
        self.worker = SwitchTimingWorker(
            app.ser, json.loads(json.dumps(self.plan)), output, 0, self,
            feedback_selector=int(self.analog_gain.currentData()),
        )
        self.worker.timing.connect(self.switch_timing_ready)
        self.worker.message.connect(self.status.setText)
        self.worker.finished.connect(self.finished)
        for widget in (
                self.reference_button, self.save_finger_button,
                self.load_finger_button, self.spacing, self.cycles,
                self.start_button, self.analog_gain, self.first_delay,
                self.boundary_delay, self.reference_line_button,
                self.test_switch_timing):
            widget.setEnabled(False)
        self.continuous.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.stop_dense_button.setEnabled(False)
        self.status.setText(
            f'正在对当前{self.point_count()}点做3轮正向CH1 90%到达时间测试…'
        )
        self.worker.start()
        self.refresh_wait_controls()

    @QtCore.pyqtSlot(object)
    def switch_timing_ready(self, report):
        try:
            if not report.get('complete') or self.plan is None:
                raise ValueError('测试未完整完成')
            transitions = report.get('analysis', {}).get('transitions', [])
            if len(transitions) != self.point_count() - 1:
                raise ValueError('正向跳转记录数与当前路线不匹配')
            current = (
                list(self.point_waits)
                if isinstance(self.point_waits, list)
                and len(self.point_waits) == self.point_count()
                else [650] * self.point_count()
            )
            measurements = [None] * self.point_count()
            late, missing = [], []
            for transition in transitions:
                target = int(transition['route_target_order'])
                metric = transition['channels']['CH1']
                t90 = metric.get('t90_us')
                meaningful = bool(metric.get('meaningful_step'))
                if meaningful and t90 is None:
                    wait_us = 15000
                    missing.append(target + 1)
                elif meaningful:
                    wait_us = int(np.ceil(max(50.0, float(t90)) / 25.0) * 25)
                else:
                    # No resolvable CH1 step is already within the analyzer's
                    # 20-code deadband; the minimum legal sampling wait is
                    # sufficient and is explicitly identified in metadata.
                    wait_us = 50
                wait_us = max(50, min(15000, wait_us))
                current[target] = wait_us
                measurements[target] = {
                    'source_order': int(transition['route_source_order']),
                    'target_order': target,
                    'target_index': int(transition['target_fullband_index']),
                    'meaningful_step': meaningful,
                    't90_after_target_write_us': t90,
                    'displayed_wait_us': wait_us,
                    'classification': metric.get('classification'),
                }
                if t90 is not None and float(t90) > 3000.0:
                    late.append(target + 1)
            self.point_waits = current
            self.point_wait_route = self.point_route_key()
            self.use_point_waits.setChecked(True)
            self.plan['switch_90pct_timing'] = {
                'source': str(self.worker.output),
                'repeats': int(report.get('repeats', 3)),
                'direction': report.get('direction'),
                'reverse_wrap_tested': False,
                'first_point_unmeasured': True,
                'points': measurements,
            }
            measured = sum(item is not None for item in measurements)
            detail = []
            if late:
                detail.append('>3 ms点：' + '/'.join(map(str, late)))
            if missing:
                detail.append('未解出90%点：' + '/'.join(map(str, missing)))
            suffix = '；' + '；'.join(detail) if detail else '；所有可解出跳转均不超过3 ms'
            self.status.setText(
                f'开关速度测试完成：已将{measured}个正向跳转的CH1 90%到达时间'
                f'写入逐点等待（第1点不做反向回绕测试，保留原值）{suffix}。'
            )
            self._save_last_route_session(ready=self.plan_ready)
        except Exception as exc:
            self.status.setText(f'开关速度结果未采用：{exc}')

    def start_equal_interval_reference(self, _checked=False):
        """Acquire a selectable-stride spectrum through EqualIntervalWorker."""
        stride = max(1, min(5, int(round(self.equal_interval_nm.value() / .02))))
        actual_nm = stride * .02
        self.equal_interval_nm.setValue(actual_nm)
        indices = list(range(0, 2001, stride))
        if indices[-1] != 2000:
            indices.append(2000)
        self.start_scan(
            reference=True,
            equal_interval=True,
            dense_indices=indices,
        )

    def start_scan(self, _checked=False, *, reference=False, reference_line=False,
                   equal_interval=False, dense_indices=None):
        import app_JDSU as app
        if self.demo:
            self.status.setText("演示模式不会启动真实激光器")
            return
        if self.worker is not None or (self.plan is None and not reference):
            return
        if not reference and not reference_line and not self.plan_ready:
            self.status.setText('当前没有可直接扫描的临时路线；请先完成密集光谱选点或导入有效路线。')
            return
        if not app.switch_mode_enable or app.ser_open:
            self.status.setText("请先停止其他采集")
            return
        if not temporary_transport_supported(app.ser):
            self.status.setText("临时选点模式需要真实USB或LAN局域网全功能通道")
            return
        try:
            signal_channel = self.selected_channel()
            feedback_selector = self.effective_feedback_selector(signal_channel)
            if (self.plan is not None and not reference
                    and int(self.plan.get('signal_channel', 1)) != signal_channel):
                raise ValueError('当前点表属于其他ADC通道，请为所选通道重新采集光谱并选点')
            selected_cycles = 0 if self.continuous.isChecked() else self.cycles.value()
            options = dict(first_delay_us=self.first_delay.value(),
                           boundary_extra_us=self.boundary_delay.value(),
                           feedback_selector=feedback_selector,
                           adc_channel=signal_channel)
            if reference:
                options.update(
                    reference_dense_indices=(list(dense_indices)
                                             if dense_indices is not None else None),
                    reference_acquisition_method=(
                        'equal_interval_single' if equal_interval else 'stable_window'
                    ),
                    reference_settle_s=(
                        float(self.equal_interval_settle.value())
                        if equal_interval else .2
                    ),
                )
            if self.use_point_waits.isChecked() and not (reference or reference_line):
                if not self._ensure_point_waits_for_route():
                    raise ValueError('当前没有可用临时路线，无法生成逐点等待')
                options.update(first_delay_us=650, boundary_extra_us=0,
                               point_delays_us=list(self.point_waits))
            if not reference:
                command(self.plan['rows'], selected_cycles, 1, **options)
            app.ensure_serial_open()
        except Exception as exc:
            self.status.setText(f"启动前检查失败：{exc}")
            return
        prefix = ('reference_line' if reference_line else
                  'equal_interval_reference' if equal_interval else
                  'reference' if reference else 'capture')
        output = output_path('temporary_test', f'{prefix}_{time.time_ns()}.json')
        app.switch_mode_enable = False
        self.clear_peak_fits('等待新采集帧')
        self.frame_times = []
        self.fit_pending = True
        # Keep the last complete route until dense acquisition AND selection
        # succeed. A failed retry must not destroy the user's working route.
        self._reference_line_prior_ready = bool(self.plan_ready)
        worker_type = ReferenceLineWorker if reference_line else ReferenceWorker if reference else TemporaryWorker
        self.worker = worker_type(app.ser, json.loads(json.dumps(self.plan)), output, selected_cycles, self, **options)
        if reference:
            self._live_dense_rows = []
            self._dense_capture_active = True
            self.worker.reference.connect(self.reference_ready)
            self.worker.reference_point.connect(self.update_dense_reference_point)
        elif reference_line:
            self.worker.reference.connect(self.reference_line_ready)
        self.worker.frame.connect(self.update_frame)
        self.worker.message.connect(self.status.setText)
        self.worker.finished.connect(self.finished)
        for widget in (self.channel_combo, self.reference_button,
                       self.equal_interval_reference_button,
                       self.equal_interval_nm, self.equal_interval_settle,
                       self.save_finger_button, self.load_finger_button,
                       self.spacing, self.cycles, self.start_button,
                       self.analog_gain, self.first_delay, self.boundary_delay,
                       self.test_switch_timing):
            widget.setEnabled(False)
        self.continuous.setEnabled(False)
        self.reference_line_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.stop_dense_button.setEnabled(reference)
        self.status.setText(
            f'正在采集当前{self.point_count()}点慢速参考线…' if reference_line
            else (f'正在按等间隔模式采集{len(dense_indices)}点'
                  f'{self.channel_name()}光谱（{self.equal_interval_nm.value():.2f} nm，'
                  f'{self.equal_interval_settle.value():.3f} s/点）…')
            if reference and equal_interval
            else f"正在采集{self.channel_name()}密集稳定参考（可能需要数分钟）…" if reference
            else f"正在执行临时{self.point_count()}点扫描…"
        )
        self.worker.start()
        self.refresh_wait_controls()

    def redraw_values(self, _value=None):
        if self.plan is None:
            return
        self.redraw_dense_selection()
        rows = self.plan['rows']
        current_x = [row['measured_nm'] for row in rows]
        reference_x = self.reference_wavelengths or current_x
        for values, curves, reference, wavelengths, selector in (
                (self.latest_values, self.curves, False, current_x,
                 self.latest_feedback_selector),
                (self.reference_values if self.show_reference.isChecked() else None,
                 self.reference_curves, True, reference_x,
                 self.reference_feedback_selector)):
            for group, curve in enumerate(curves[:self.peak_count()]):
                if values is None:
                    curve.clear()
                    continue
                start = group*5
                curve.setData(wavelengths[start:start+5],
                              self._plot_display_values(
                                  values[start:start+5], reference=reference,
                                  selector=selector,
                              ))
        self._update_plot_unit_label()
        self._refresh_table_display_values()
        self.redraw_peak_fits()

    def clear_peak_fits(self, reason='等待实时帧'):
        self.peak_fits = []
        for curve, label in zip(self.fitted_curves, self.fit_labels):
            curve.clear()
            label.hide()
        for i in range(self.peak_count()):
            for row, text in enumerate(('—', '—', reason)):
                self.peak_fit_table.setItem(row, i, QtWidgets.QTableWidgetItem(text))
        self.peak_fit_status.setText(reason+'；基于旧标定坐标的估计，不代表光学精度通过。')

    def redraw_peak_fits(self, _checked=None):
        enabled = self.show_peak_fits.isChecked()
        self.data_tabs.setTabEnabled(0, enabled)
        if not enabled and self.data_tabs.currentIndex() == 0:
            self.data_tabs.setCurrentIndex(1)
        self.peak_fit_status.setVisible(enabled)
        for i, (curve, label) in enumerate(zip(self.fitted_curves, self.fit_labels)):
            result = self.peak_fits[i] if i < len(self.peak_fits) else None
            if (not enabled or not result
                    or result.get('center_nm') is None):
                curve.clear()
                label.hide()
                continue
            warning = not result.get('valid', False)
            color = QtGui.QColor('#f59e0b') if warning else pg.intColor(i, 9)
            curve.setPen(pg.mkPen(color, width=2, style=QtCore.Qt.DotLine))
            label.setColor(color)
            curve.setData(
                result['curve_x_nm'],
                self._plot_display_values(
                    result['curve_adc'], selector=self.latest_feedback_selector
                ),
            )
            marker = '⚠' if warning else ''
            label.setText(f"P{i+1}{marker}\n≈{result['center_nm']:.4f}")
            apex_display = float(self._plot_display_values([
                result['baseline_adc'] + result['amplitude_adc']
            ], selector=self.latest_feedback_selector)[0])
            label.setPos(result['center_nm'], apex_display)
            label.show()

    def fit_values(self, _checked=False):
        values = self._plot_display_values(
            self.latest_values or [], selector=self.latest_feedback_selector
        ).tolist()
        if self.show_peak_fits.isChecked():
            fit_apex = [r['baseline_adc'] + r['amplitude_adc']
                        for r in self.peak_fits
                        if r.get('center_nm') is not None]
            values.extend(self._plot_display_values(
                fit_apex, selector=self.latest_feedback_selector
            ).tolist())
        if self.show_reference.isChecked() and self.reference_values is not None:
            values.extend(self._plot_display_values(
                self.reference_values, reference=True,
                selector=self.reference_feedback_selector,
            ).tolist())
        if self.show_dense_selection.isChecked():
            if self._dense_capture_active and self._live_dense_rows:
                live_values = [
                    row.get('signal_adc_code', row.get('ch1_adc_code', np.nan))
                    for row in self._live_dense_rows
                ]
                values.extend(self._plot_display_values(
                    live_values, selector=self.effective_feedback_selector()
                ).tolist())
            elif self.dense_values is not None:
                values.extend(self._plot_display_values(
                    self.dense_values, selector=self.dense_feedback_selector
                ).tolist())
        values = [float(value) for value in values if np.isfinite(value)]
        if not values:
            return
        if self.display_mode == 'dbm':
            lower, upper = min(values) - 2.0, max(values) + 2.0
            if upper - lower < 6.0:
                middle = (upper + lower) / 2.0
                lower, upper = middle - 3.0, middle + 3.0
        else:
            lower, upper = 0.0, max(
                10.0 if self.display_mode == 'adc' else .01,
                max(values) * 1.3,
            )
        self.plot.setYRange(lower, upper, padding=0)
        self.plot.enableAutoRange(axis='y', enable=False)
        self.fit_pending = False

    def update_frame(self, frame):
        values = [r['second_code'] for r in frame['records']]
        self.latest_values = values
        self.latest_channel = int(frame.get('adc_channel', self.selected_channel()))
        self.latest_feedback_selector = int(frame.get(
            'feedback_selector', self.effective_feedback_selector(self.latest_channel)
        ))
        from temporary_peak_fit import fit_temporary_frame
        if self.plan is None:
            self.clear_peak_fits('点表无效')
            return
        self.peak_fits = fit_temporary_frame(self.plan['rows'], values, frame['sequence'] >= 16)
        if frame['sequence'] >= 16:
            for i, result in enumerate(self.peak_fits):
                self.fit_failure_streaks[i] = 0 if result['valid'] else self.fit_failure_streaks[i] + 1
            # Realtime fitting is a quality indication, not a destructive
            # verification gate.  A few poor frames can be caused by motion or
            # by stopping while a frame is in flight.  Revoking plan_ready here
            # used to overwrite the last 9/9 slow-reference result and left the
            # Start button disabled after "stop and switch off".  Keep the
            # verified route reusable; the status below still identifies every
            # peak whose live fit has failed repeatedly.
        for i, result in enumerate(self.peak_fits):
            available = result.get('center_nm') is not None
            warning = available and not result.get('valid', False)
            center_text = (
                f"{result['center_nm']:.4f}" + (' ⚠' if warning else '')
                if available else '—'
            )
            r_squared = result.get('r_squared')
            r2_text = (
                f'{float(r_squared):.3f}'
                if r_squared is not None and np.isfinite(r_squared) else '—'
            )
            texts = (center_text, r2_text, result['reason'])
            for row, text in enumerate(texts):
                item = QtWidgets.QTableWidgetItem(text)
                item.setToolTip(result['reason'])
                if warning:
                    item.setBackground(QtGui.QColor('#fff3cd'))
                    item.setForeground(QtGui.QColor('#9a6700'))
                self.peak_fit_table.setItem(row, i, item)
        available_count = sum(
            result.get('center_nm') is not None for result in self.peak_fits
        )
        valid_count = sum(result.get('valid', False) for result in self.peak_fits)
        warning_count = available_count - valid_count
        self.peak_fit_status.setText(
            f"第{frame['sequence']+1}帧：{available_count}/{self.peak_count()}峰已给出拟合，"
            f"质量通过{valid_count}峰、⚠可能有问题{warning_count}峰；"
            '点线为高斯拟合，圆点/实线仍是原始值；波长精度未验证。'
        )
        self.redraw_values()
        if self.fit_pending and frame['sequence'] >= 16:
            self.fit_values()
        self._refresh_table_display_values()
        if frame['sequence'] >= 16:
            self.frame_times.append(frame['cycle_start_us'])
            del self.frame_times[:-64]
        rate = ''
        if len(self.frame_times) > 1:
            rate = f" · {1e6/np.mean(np.diff(self.frame_times[-64:])):.2f} Hz（板端）"
        q = frame['quality']
        issues = [label for key, label in (('clipped', 'ADC饱和'), ('unsettled', '双采样差异较大（可能是信号变化）'),
                                         ('peak_at_edge', '峰在窗口边缘，扫描点仍保持不变')) if q[key]]
        drifted = [str(i + 1) for i, streak in enumerate(self.fit_failure_streaks) if streak >= 3]
        if drifted:
            issues.append(
                '峰' + '/'.join(drifted)
                + '连续出现拟合质量警告：结果已保留，请结合表格中的⚠原因判断'
            )
        if not q['warmed_up']:
            issues.append('预热中')
        self.status.setText(f"第{frame['sequence']+1}帧{rate} · " + (' / '.join(issues) or '原始数据采集中，双采样不能证明已稳定'))

    def stop(self):
        if self.worker is not None:
            self.worker.stop()
            self.stop_button.setEnabled(False)

    def finished(self):
        import app_JDSU as app
        app.switch_mode_enable = True
        self.worker.deleteLater()
        self.worker = None
        self._dense_capture_active = False
        self._live_dense_rows = []
        self.peak_fit_status.setText('已停采，保留最后一帧拟合（非实时）；波长精度未验证。')
        for widget in (self.channel_combo, self.reference_button,
                       self.equal_interval_reference_button,
                       self.equal_interval_nm, self.equal_interval_settle,
                       self.save_finger_button, self.load_finger_button,
                       self.spacing, self.cycles,
                       self.analog_gain, self.first_delay, self.boundary_delay):
            widget.setEnabled(True)
        self.spacing.setEnabled(False)
        self.start_button.setEnabled(self.plan is not None and self.plan_ready)
        self.reference_line_button.setEnabled(self.plan is not None)
        self.test_switch_timing.setEnabled(self.plan is not None)
        self.continuous.setEnabled(True)
        self.cycles.setEnabled(not self.continuous.isChecked())
        self.stop_button.setEnabled(False)
        self.stop_dense_button.setEnabled(False)
        self.redraw_dense_selection()
        self._sync_channel_ui()
        self.refresh_wait_controls()
