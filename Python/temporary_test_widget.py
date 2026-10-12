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
    _feedback_pair,
    auto_select_dense_plan,
    assess_five_point_shape,
    assess_peak_shape,
    points_per_peak_for_rows,
    build_manual_dense_plan,
    build_selected_plan,
    command,
    execute_device,
    frame_adc_values,
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
from beta_profile import beta_enabled
from responsive_layout import FlowLayout, ResponsiveScrollArea, ScrollContentWidget


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
DEFAULT_FIRST_DELAY_US = 850
DEFAULT_BOUNDARY_DELAY_US = 850
BETA_STARTUP_SCAN = dict(signal_channel=0, scan_start_nm=1525.0, scan_stop_nm=1555.0,
                         equal_interval_nm=.02, equal_interval_settle_s=.003)
MANUAL_SELECTION_POINT_COUNT = 45
DENSE_PLOT_REFRESH_MS = 33
TEMPORARY_PLOT_REFRESH_MS = 33
SCAN_PREFERENCES_PATH = output_path('preferences', 'finger_stress_scan.json')


def boundary_wait_from_settings(settings):
    """Read absolute waits; preserve the timing of explicitly loaded old records."""
    if 'boundary_wait_us' in settings:
        return int(settings['boundary_wait_us'])
    if 'boundary_extra_us' in settings:
        return (int(settings.get('first_delay_us', DEFAULT_FIRST_DELAY_US))
                + int(settings['boundary_extra_us']))
    return DEFAULT_BOUNDARY_DELAY_US


def equal_interval_indices_for_range(table, start_nm, stop_nm, spacing_nm):
    """Return calibrated-table indices inside one inclusive wavelength range."""

    table = tuple(table)
    target_axis = np.asarray([point.target_nm for point in table], dtype=float)
    if (len(table) != 2001 or target_axis.shape != (2001,)
            or not np.all(np.isfinite(target_axis))
            or np.any(np.diff(target_axis) <= 0)):
        raise ValueError('等间隔采谱需要完整且递增的2001点标定表')
    start_nm, stop_nm, spacing_nm = map(float, (start_nm, stop_nm, spacing_nm))
    if not all(map(np.isfinite, (start_nm, stop_nm, spacing_nm))):
        raise ValueError('扫描波长范围和间隔必须是有限数值')
    if start_nm >= stop_nm:
        raise ValueError('扫描起点必须小于扫描终点')
    if start_nm < target_axis[0] or stop_nm > target_axis[-1]:
        raise ValueError(
            f'扫描范围必须位于{target_axis[0]:.2f}～{target_axis[-1]:.2f} nm标定范围内'
        )
    table_step = float(np.median(np.diff(target_axis)))
    stride = max(1, min(5, int(round(spacing_nm / table_step))))
    actual_spacing_nm = stride * table_step
    eligible = np.flatnonzero(
        (target_axis >= start_nm - 1e-9) & (target_axis <= stop_nm + 1e-9)
    )
    if not len(eligible):
        raise ValueError('设定范围内没有标定波长点')
    indices = eligible[::stride].astype(int).tolist()
    if indices[-1] != int(eligible[-1]):
        indices.append(int(eligible[-1]))
    if len(indices) < MANUAL_SELECTION_POINT_COUNT:
        raise ValueError(
            f'该范围和间隔只能采{len(indices)}点，手动放置45点至少需要45个密集谱采样点'
        )
    return indices, actual_spacing_nm


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

    def __init__(self, device, plan, output, cycles, parent=None, *, first_delay_us=DEFAULT_FIRST_DELAY_US,
                 boundary_extra_us=0, feedback_selector=2, point_delays_us=None,
                 adc_channel=1, reference_dense_indices=None,
                 reference_acquisition_method='stable_window',
                 reference_settle_s=.2, single_sample=False, fast_dac_writes=False,
                 coalesce_display=False):
        super().__init__(parent)
        self.device, self.plan, self.output, self.cycles = device, plan, output, cycles
        self.cancel = threading.Event()
        from temporary_live_display import LatestFrameMailbox
        self.display_mailbox = LatestFrameMailbox() if coalesce_display else None
        self.options = dict(first_delay_us=first_delay_us, boundary_extra_us=boundary_extra_us,
                            feedback_selector=feedback_selector)
        self.adc_channel = int(adc_channel)
        if single_sample or fast_dac_writes:
            self.options.update(single_sample=single_sample, fast_dac_writes=fast_dac_writes)
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

    def take_display_frame(self):
        return self.display_mailbox.take() if self.display_mailbox is not None else None

    def run(self):
        try:
            report = execute_device(self.device, self.plan, self.output, cycles=self.cycles,
                                    adc_channel=self.adc_channel,
                                    **self.options,
                                    on_frame=(self.display_mailbox.offer
                                              if self.display_mailbox is not None else self.frame.emit),
                                    should_stop=self.cancel.is_set)
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
                    f"重采光谱 {n}/{total}；边采边画，保留原始ADC")
                    if n == 1 or n == total or n % 16 == 0 else None)
            self.reference.emit(report)
        except Exception as exc:
            self.message.emit(f"参考采集失败：{exc}；记录：{self.output}")


class ReferenceLineWorker(ReferenceWorker):
    def run(self):
        from temporary_reference import acquire_selected_reference
        try:
            report = acquire_selected_reference(self.device, self.output, selected_rows=self.plan['rows'],
                feedback_selector=self.options['feedback_selector'],
                signal_channel=self.adc_channel, should_stop=self.cancel.is_set,
                on_row=lambda row, n, total: self.reference_point.emit({
                    'row': row, 'completed': n, 'total': total,
                }),
                on_progress=lambda n,total: self.message.emit(
                    f'采集参考线 {n}/{total}；10 ms/点，单次ADC，完成后停光'))
            self.reference.emit(report)
        except Exception as exc:
            self.message.emit(f'参考线采集失败：{exc}；记录：{self.output}')


class CurrentWaitLevelWorker(TemporaryWorker):
    measured = QtCore.pyqtSignal(object)

    def run(self):
        from temporary_current_wait_levels import capture_current_wait_levels
        try:
            report = capture_current_wait_levels(
                self.device, self.plan, self.output,
                point_delays_us=self.options['point_delays_us'],
                wait_mode=self.plan['current_wait_measurement_mode'],
                channel=self.adc_channel, selector=self.options['feedback_selector'],
                fast=bool(self.options.get('fast_dac_writes', False)),
                should_stop=self.cancel.is_set, on_progress=self.message.emit)
            self.measured.emit(report)
        except Exception as exc:
            self.message.emit(f'当前等待到达比例测量未完成：{exc}；记录：{self.output}')


class SwitchTimingWorker(TemporaryWorker):
    """Measure ADC transients; Beta includes all points and the cyclic wrap."""

    timing = QtCore.pyqtSignal(object)

    def run(self):
        import app_JDSU as app
        from benchmark_stress_realtime import _safe_disarm, _close_shutter
        from capture_ch1_teacher_spectrum import save_checkpoint
        from capture_temporary_half_height_transients import (
            capture_custom,
            capture_target_level,
            transition_analysis,
        )

        route = list(self.plan.get('rows', []))
        include_wrap = self.options.get('point_delays_us') is not None
        from temporary_t90_waits import (T90_REPEATS, T90_GATE_POLICY, T90_VOLTAGE_THRESHOLD_V,
                                         T90_LOW_VOLTAGE_WAIT_US, steady_probe_level)
        repeats = T90_REPEATS if include_wrap else 3
        report = {
            'schema': ('temporary_route_cycle_voltage_gated_90pct_timing_v4' if include_wrap else
                       'temporary_route_forward_90pct_timing_v1'),
            'created_s': time.time(),
            'route': route,
            'repeats': repeats,
            'direction': ('forward scan plus last-to-first frame wrap' if include_wrap else
                          'strictly forward, small wavelength to large wavelength'),
            'reverse_wrap_tested': include_wrap,
            'machine_id': get_runtime_machine_id(),
            'captures': [],
            'complete': False,
            'ch1_feedback_selector': int(self.options['feedback_selector']) if self.adc_channel == 1 else 0,
            'signal_channel': self.adc_channel,
            'signal_channel_name': f'CH{self.adc_channel}',
            'feedback_selector': int(self.options['feedback_selector']),
            'fast_dac_writes': bool(self.options.get('fast_dac_writes', False)),
        }
        if include_wrap:
            report.update(voltage_gate_policy=T90_GATE_POLICY, voltage_threshold_v=T90_VOLTAGE_THRESHOLD_V,
                          low_voltage_wait_us=T90_LOW_VOLTAGE_WAIT_US, cycle_wrap_included=True,
                          reverse_wrap_tested=False)
        old_timeout = self.device.timeout
        try:
            if len(route) not in (set(range(5, 46, 5)) | set(range(7, 64, 7))):
                raise ValueError('开关测试需要有效的5点或7点每峰正向路线')
            command(route, 32, 1)  # Validate all DAC rows before optical commands.
            # Create and test the output location before any optical commands.
            # A missing/unwritable folder must fail before the first transition.
            save_checkpoint(self.output, report)
            self.device.timeout = .1
            if not _safe_disarm(self.device) or not _close_shutter(self.device):
                raise RuntimeError('开关测试前停光未确认')
            self.device.write(app.build_work_mode_command(2))
            time.sleep(.12)
            self.device.reset_input_buffer()
            worker = app.EqualIntervalWorker(
                self.device, (), {}, settle_s=.3,
                feedback_selectors=_feedback_pair(
                    self.adc_channel, int(self.options['feedback_selector'])),
            )
            if not worker._feedback_command_and_ack():
                raise RuntimeError('诊断通道模拟档位设置未确认')
            tag = int(time.time_ns() & 0x7fffffff) or 1
            target_orders = range(0 if include_wrap else 1, len(route))
            total = repeats * len(target_orders)
            for repeat in range(repeats):
                for target_order in target_orders:
                    if self.cancel.is_set():
                        raise InterruptedError('开关速度测试已取消')
                    tag = (tag + 1) & 0xffffffff or 1
                    source_order = (target_order - 1) % len(route)
                    source = route[source_order]
                    target = route[target_order]
                    probe = None
                    if include_wrap:
                        probe = capture_target_level(self.device, target, tag,
                                                     fast_dac_writes=report['fast_dac_writes'])
                        level = steady_probe_level(probe, self.adc_channel)
                        if self.cancel.is_set():
                            raise InterruptedError('90%稳定时间测量已取消')
                    low = probe is not None and level['target_voltage_v'] <= T90_VOLTAGE_THRESHOLD_V
                    if low:
                        item = dict(fast_dac_writes=report['fast_dac_writes'],
                                    measurement_kind='low_voltage_fixed_wait', voltage_probe=probe)
                    else:
                        if include_wrap:
                            tag = (tag + 1) & 0xffffffff or 1
                        item = capture_custom(self.device, source, target, tag,
                                              fast_dac_writes=report['fast_dac_writes'])
                        if include_wrap:
                            item.update(measurement_kind='measured_transition', voltage_probe=probe)
                            if target_order == 0:
                                report['reverse_wrap_tested'] = True
                    item.update({
                        'repeat': repeat,
                        'route_source_order': source_order,
                        'route_target_order': target_order,
                        'source_fullband_index': int(source['index']),
                        'target_fullband_index': int(target['index']),
                        'request_tag': tag,
                    })
                    report['captures'].append(item)
                    done = len(report['captures'])
                    self.message.emit(
                        f'开关速度测试 {done}/{total}：'
                        f'第{repeat + 1}/{repeats}轮，进入第{target_order + 1}点'
                        + ('（首点固定等待，未测回绕）' if target_order == 0 and low else
                           '（末点回绕到首点）' if target_order == 0 else '')
                        + ('；稳定电压≤0.2 V，跳过90%测量，固定300 µs' if low else '')
                    )
                    if done % 10 == 0:
                        save_checkpoint(self.output, report)
            if not include_wrap:
                report['analysis'] = transition_analysis(
                    report['captures'], route, repeats, signal_channel_name=f'CH{self.adc_channel}',
                    include_wrap=False,
                )
            report['complete'] = True
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
            shutdown_ok = report.get('exact_disarm_ack') and report.get('exact_soa_shutter_ack')
            if not shutdown_ok:
                report['complete'] = False
            elif report.get('complete'):
                if self.options.get('point_delays_us') is not None:
                    from temporary_t90_waits import analyze_t90_waits
                    try:
                        report['t90_waits'] = analyze_t90_waits(report)
                    except (ValueError, KeyError, TypeError) as exc:
                        report['t90_waits_error'] = str(exc)
                        self.message.emit(f'90%稳定时间分析失败：{exc}；原始记录已保留，等待未改动')
            # Saving must never bypass a shutdown warning or kill the thread
            # with an unhandled exception after a disk/permission failure.
            try:
                save_checkpoint(self.output, report)
            except (OSError, TypeError, ValueError) as exc:
                report['complete'] = False
                report['save_error'] = str(exc)
                self.message.emit(
                    f'90%测量记录保存失败：{exc}；等待未应用。'
                    + ('停采和关光均已确认。' if shutdown_ok else
                       '停采或关光未完整确认，请立即检查板卡。'))
            else:
                if not shutdown_ok:
                    self.message.emit('开关测试后停采或关光未完整确认，请立即检查板卡')
                elif not report.get('complete'):
                    self.message.emit(
                        f'开关速度测试未完成：{report.get("error", "测量未完成")}；'
                        f'停采和关光均已确认；记录：{self.output}')
            if report.get('complete') and shutdown_ok:
                self.timing.emit(report)


class TemporaryTestWindow(QtWidgets.QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.beta = beta_enabled()
        self.developer_mode = False
        self._developer_previous_frame = None
        self._developer_timing_budget = None
        self.persistence_enabled = controller is not None
        self._restoring_session = False
        self.worker = None
        self.plan = None
        self.plan_ready = False
        self.demo = False
        self.frame_times = []
        self._temporary_draw_timer = QtCore.QTimer(self)
        self._temporary_draw_timer.setInterval(TEMPORARY_PLOT_REFRESH_MS)
        self._temporary_draw_timer.timeout.connect(self._flush_temporary_display)
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
        self._auto_select_after_dense = False
        self._dense_draw_progress = None
        self._dense_last_range_count = 0
        self._dense_draw_timer = QtCore.QTimer(self)
        self._dense_draw_timer.setSingleShot(True)
        self._dense_draw_timer.setInterval(DENSE_PLOT_REFRESH_MS)
        self._dense_draw_timer.timeout.connect(self._flush_dense_reference_plot)
        self.manual_selection_indices = []
        self.manual_selection_history = []
        self.manual_selection_active = False
        self.manual_points_per_peak = 5
        self.dense_scan_range_nm = (1525.0, 1565.0)
        self.reference_failed_points = []
        self._live_reference_line_rows = []
        self._reference_line_capture_active = False
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
        outer_layout = QtWidgets.QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        self.page_scroll = ResponsiveScrollArea(self)
        self.page_scroll.setObjectName('temporaryPageScroll')
        self.page_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.page_content = ScrollContentWidget()
        self.page_content.setObjectName('temporaryPageContent')
        layout = QtWidgets.QVBoxLayout(self.page_content)
        self.page_scroll.setWidget(self.page_content)
        outer_layout.addWidget(self.page_scroll)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        def compact_field(label, editor, editor_width=108):
            holder = QtWidgets.QWidget(self)
            row = QtWidgets.QHBoxLayout(holder)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            caption = QtWidgets.QLabel(label, holder)
            caption.setObjectName('fieldCaption')
            caption.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            caption.setBuddy(editor)
            row.addWidget(caption)
            editor.setMinimumWidth(editor_width)
            editor.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
            row.addWidget(editor)
            return holder

        def flow_row(items, destination=layout):
            holder = QtWidgets.QFrame(self)
            holder.setObjectName('workflowRow')
            holder.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
            )
            outer = QtWidgets.QHBoxLayout(holder)
            outer.setContentsMargins(10, 6, 10, 6)
            outer.setSpacing(12)
            # A dedicated caption column keeps wrapped sub-functions under
            # their own category, never interleaved with another section.
            caption, *controls = items
            caption.setMinimumWidth(max(70, caption.sizeHint().width()))
            outer.addWidget(caption, 0, QtCore.Qt.AlignTop)
            body = QtWidgets.QWidget(holder)
            row = FlowLayout(body, horizontal_spacing=8, vertical_spacing=6)
            for item in controls:
                row.addWidget(item)
            outer.addWidget(body, 1)
            destination.addWidget(holder)
            return holder

        def section_label(text):
            label = QtWidgets.QLabel(text, self)
            label.setObjectName('workflowSectionTitle')
            label.setMinimumHeight(34)
            label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            return label

        self.title_label = QtWidgets.QLabel(
            f"手指应力测试模式 · {self.runtime_machine_label} · CH1 · 9峰×5点 · 仅扫描选定的45点"
        )
        layout.addWidget(self.title_label)
        self.title_label.hide()
        self.machine_profile_label = QtWidgets.QLabel(
            f"当前设备：{self.runtime_machine_label}｜{self.runtime_parameter_note}"
        )
        self.machine_profile_label.setStyleSheet(
            'color: #b45309; font-weight: 600;'
        )
        self.machine_profile_label.setWordWrap(True)
        layout.addWidget(self.machine_profile_label)
        self.machine_profile_label.hide()
        self.note_label = QtWidgets.QLabel("已有45点表可直接扫描或采集参考线；只有重新选点才需重采密集光谱。实线为实时值，虚线为10 ms参考。\n"
                               "自动选点时五点覆盖约两倍半高宽，尽量包含峰的两侧；只显示实测点，各峰之间不补线。等待自行调节，允许低于15 Hz。\n"
                               "临时表仅在RAM中，使用USB；默认连续循环，点击停止才结束；异常保护仍会停光。\n"
                               "当前实物幅值对照未通过。扫描中45点不跟随峰移动，不归一化、不平滑、不剔除幅值变化。\n"
                               "两个ADC读数接近不代表波长已稳定；等待时间需实测验证。修改模拟放大后需重新做幅值对照。")
        self.note_label.setWordWrap(True)
        self.note_label.setToolTip(self.note_label.text())
        self.note_label.setText('在设定波长范围内采集完密集光谱后，默认自动选择9峰×5点；也可撤销、微调或清空后手动选点。\n'
                     '原始圆点/实线、参考虚线、拟合点线分开显示；参考线每点等待10 ms后直接记录单次ADC。')
        layout.addWidget(self.note_label)
        self.note_label.hide()
        self.channel_combo = QtWidgets.QComboBox()
        for channel in range(4):
            self.channel_combo.addItem(f'通道 CH{channel}', channel)
        self.channel_combo.setCurrentIndex(1)
        self.channel_combo.setToolTip(
            '选择密集光谱、手动选点、参考线、实时扫描与拟合所使用的ADC通道。'
            'CH0/CH1可调模拟跨阻；CH2/CH3固定2 kΩ。'
        )
        self.channel_combo.setFixedWidth(106)
        self.equal_interval_reference_button = QtWidgets.QPushButton(
            '重采密集光谱'
        )
        self.equal_interval_reference_button.setMinimumWidth(132)
        self.scan_start_nm = QtWidgets.QDoubleSpinBox()
        self.scan_start_nm.setDecimals(2)
        self.scan_start_nm.setRange(1525.00, 1564.98)
        self.scan_start_nm.setSingleStep(.02)
        self.scan_start_nm.setValue(1525.00)
        self.scan_start_nm.setSuffix(' nm')
        self.scan_stop_nm = QtWidgets.QDoubleSpinBox()
        self.scan_stop_nm.setDecimals(2)
        self.scan_stop_nm.setRange(1525.02, 1565.00)
        self.scan_stop_nm.setSingleStep(.02)
        self.scan_stop_nm.setValue(1555.00)
        self.scan_stop_nm.setSuffix(' nm')
        for editor in (self.scan_start_nm, self.scan_stop_nm):
            editor.setToolTip(
                '只采集起点到终点（含边界）范围内的标定点；不会扫描范围外波长。'
            )
        self.equal_interval_nm = QtWidgets.QDoubleSpinBox()
        self.equal_interval_nm.setDecimals(2)
        self.equal_interval_nm.setRange(.02, .10)
        self.equal_interval_nm.setSingleStep(.02)
        self.equal_interval_nm.setValue(.04)
        self.equal_interval_nm.setSuffix(' nm')
        self.equal_interval_nm.setToolTip(
            '按2001点标定表每N格取一点；允许0.02～0.10 nm。'
            '间隔越大采集越快，但窄峰可用采样点更少。'
        )
        self.equal_interval_settle = QtWidgets.QDoubleSpinBox()
        self.equal_interval_settle.setDecimals(3)
        self.equal_interval_settle.setRange(0.0, 10.0)
        self.equal_interval_settle.setSingleStep(.010)
        self.equal_interval_settle.setValue(.001)
        self.equal_interval_settle.setSuffix(' s/点')
        self.equal_interval_settle.setToolTip(
            'DAC回读确认后的额外稳定等待，不是采集一个点的总时间。'
            '随后清空旧帧，再取第一帧ADC；回读、通信和新帧到达另有耗时。'
        )
        self.stop_dense_button = QtWidgets.QPushButton("停止绘制")
        self.stop_dense_button.setMinimumWidth(88)
        self.stop_dense_button.setEnabled(False)
        self.stop_dense_button.setToolTip(
            '安全停止当前密集光谱采集并关光；已采部分会保留，'
            '随后可在已经绘出的谱线上手动点击45个点。'
        )
        self.save_finger_button = QtWidgets.QPushButton('保存手指记录')
        self.load_finger_button = QtWidgets.QPushButton('载入手指记录')
        self.save_finger_button.setEnabled(False)
        self.save_finger_button.setToolTip(
            '保存当前手指的密集谱线、15/45点路线、参考线和等待设置；'
            '下次可直接载入并开始手指应力测试。'
        )
        self.load_finger_button.setToolTip(
            '选择以前保存的手指记录；载入后会显示手指编号，可直接开始手指应力测试。'
        )
        self.spacing = QtWidgets.QDoubleSpinBox()
        self.spacing.setRange(0, 0)
        self.spacing.setSpecialValueText("半高宽内等间隔5点")
        self.spacing.setToolTip('自动点全部位于峰的半高宽内；人工用左右方向键调整后可以不等间隔。')
        self.spacing.setEnabled(False)
        self.spacing.hide()
        # Retain the legacy control objects for internal callers, not as UI
        # choices: stress testing now always runs until Stop or a safety fault.
        self.cycles = QtWidgets.QSpinBox(self)
        self.cycles.setRange(32, 512)
        self.cycles.setValue(256)
        self.cycles.setSuffix(" 帧")
        self.continuous = QtWidgets.QCheckBox('连续循环', self)
        self.continuous.setChecked(True)
        self.cycles.setEnabled(False)
        self.cycles.hide()
        self.continuous.hide()
        self.start_button = QtWidgets.QPushButton("开始测试")
        self.start_button.setProperty('role', 'primary')
        self.start_button.setEnabled(False)
        self.stop_button = QtWidgets.QPushButton("停止并关光")
        self.stop_button.setProperty('role', 'danger')
        self.stop_button.setEnabled(False)
        flow_row([
            section_label('采谱'), self.channel_combo,
            compact_field('起点', self.scan_start_nm, 108),
            compact_field('终点', self.scan_stop_nm, 108),
            compact_field('间隔', self.equal_interval_nm, 92),
            compact_field('逐点等待', self.equal_interval_settle, 102),
            self.equal_interval_reference_button,
        ])
        self.reference_line_button = QtWidgets.QPushButton('采集参考线（45点）')
        self.reference_line_button.setEnabled(False)
        self.show_reference = QtWidgets.QPushButton('显示参考线（虚线）')
        self.show_reference.setCheckable(True)
        self.show_reference.setChecked(True)
        self.show_peak_fits = QtWidgets.QCheckBox('实时峰形拟合与中心波长（样条）')
        self.show_peak_fits.setChecked(True)
        self.show_peak_fits.setToolTip(
            'Makima 样条，五点估计。使用原始CH1 ADC和旧标定波长坐标；所有有限五点排布都会尝试拟合。'
            '不满足先升后降、R²偏低或中心/峰宽约束不足时仍显示结果，并用⚠标记。'
            '不修改采样点、不补偿原始幅值；显示位数不代表精度。'
        )
        self.clear_reference_button = QtWidgets.QPushButton('清除参考线')
        self.reference_gain = QtWidgets.QDoubleSpinBox()
        self.reference_gain.setRange(.1,100)
        self.reference_gain.setValue(1)
        self.reference_gain.setSuffix(' ×')
        self.reference_gain.setToolTip('仅改变参考线显示高度，原始参考ADC不变；与实时倍数不同时不能直接比较线的高度。')
        self.reference_line_button.setMinimumWidth(138)
        self.advanced_toggle = QtWidgets.QPushButton('更多参数与显示 ▸')
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setToolTip(
            '展开模拟增益、采样等待、参考线倍数和手动点位工具。'
        )
        self.advanced_panel = QtWidgets.QWidget(self)
        self.advanced_panel_layout = QtWidgets.QVBoxLayout(self.advanced_panel)
        self.advanced_panel_layout.setContentsMargins(0, 0, 0, 0)
        self.advanced_panel_layout.setSpacing(3)
        self.advanced_panel.hide()
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
        self.first_delay.setValue(DEFAULT_FIRST_DELAY_US)
        self.first_delay.setSuffix(' µs')
        self.first_delay.setToolTip('以25 µs为步长；不能仅凭双ADC接近判断波长已稳定。')
        self.boundary_delay = QtWidgets.QSpinBox()
        self.boundary_delay.setRange(50, 15000)
        self.boundary_delay.setSingleStep(25)
        self.boundary_delay.setValue(DEFAULT_BOUNDARY_DELAY_US)
        self.boundary_delay.setSuffix(' µs')
        self.boundary_delay.setToolTip('每峰第一个点写完DAC后的总等待，替代峰内等待，不叠加；以25 µs为步长。')
        flow_row([
            section_label('采样参数'),
            compact_field('模拟放大', self.analog_gain, 115),
            compact_field('实时放大', self.digital_gain, 88),
            compact_field('峰内等待', self.first_delay, 92),
            compact_field('跨峰等待', self.boundary_delay, 100),
            self.display_unit_button,
        ], self.advanced_panel_layout)
        flow_row([
            section_label('参考线'),
            compact_field('显示倍数', self.reference_gain, 88),
            self.clear_reference_button,
        ], self.advanced_panel_layout)
        self.point_waits = None
        self.point_wait_route = None
        self.use_point_waits = QtWidgets.QCheckBox('使用逐点等待（实验功能，替代统一/跨峰等待）')
        self.edit_point_waits = QtWidgets.QPushButton('设置45个点的等待…')
        self.test_switch_timing = QtWidgets.QPushButton(
            '测量45点90%稳定时间' if self.beta else '测试45点达到90%时间')
        self.test_switch_timing.setEnabled(self.plan is not None)
        self.test_switch_timing.setToolTip(
            '对当前选定点测量一次，包含末点回到首点；按本点ADC相对0计算90%稳定时间。'
            '先读取稳态电压，>0.2 V才测90%，≤0.2 V固定300 µs。'
            '测量后可一键应用逐点等待。' if self.beta else
            '对当前15/45点按小波长到大波长连续测试3轮；'
            '不做反向回绕审计。CH1主变化达到90%的板上时间会写入逐点等待表。'
        )
        self.edit_point_waits.clicked.connect(self.configure_point_waits)
        self.test_switch_timing.clicked.connect(self.start_switch_timing)
        self.use_point_waits.toggled.connect(self.refresh_wait_controls)
        self.point_diagnostics_row = flow_row([
            section_label('逐点诊断'), self.use_point_waits,
            self.edit_point_waits, self.test_switch_timing,
        ], layout)
        self.point_diagnostics_row.hide()
        if self.beta:
            self.beta_single_adc = QtWidgets.QCheckBox('单次ADC（每点仅等待一次）', self)
            self.beta_single_adc.setChecked(True)
            self.beta_single_adc.hide()
            self.beta_fast_dac = QtWidgets.QCheckBox('优化5路DAC（带光验证未通过，暂不可用）')
            # v115 passed dark timing but failed optical waveform comparison.
            # Neither tested burst build is optically qualified. Keep it disabled.
            self.beta_fast_dac.setChecked(False)
            self.beta_fast_dac.setEnabled(False)
            self.beta_fast_dac.setToolTip('已停用：v1.0.115和v1.0.116合并写入均未通过带光对照。保留每点单次ADC，5路DAC仍分别写入。')
            # Do not re-enable a rejected optical optimization by toggling ADC.
            self.beta_timing_results = QtWidgets.QPushButton('查看90%时间测量结果')
            self.beta_timing_results.setEnabled(False)
            self.beta_timing_results.clicked.connect(self.show_switch_timing_results)
            self.beta_apply_t90_waits = QtWidgets.QPushButton('一键应用90%逐点等待')
            self.beta_apply_t90_waits.setEnabled(False)
            self.beta_apply_t90_waits.setToolTip(
                '计算S固定为0，以本点最终ADC的90%为准；仍测相邻波长切换，不逐点从0启动。'
                '稳定电压>0.2 V的每点只测量一次，加55 µs余量并向上取整；'
                '≤0.2 V的点跳过90%测量，固定300 µs，不额外加余量。'
                '点击后直接启用逐点等待；未测可靠或超过15 ms时不截短、不改表。')
            self.measure_current_wait_levels = QtWidgets.QPushButton('测量当前等待到达比例')
            self.measure_current_wait_levels.setToolTip(
                '无需勾选使用逐点等待：不勾选时按峰内/跨峰统一等待，勾选时按当前逐点表。'
                '实际循环32帧，取预热后的最后一帧，再读取各点最终电压作对照。'
                '百分比=当前电压/最终电压×100%，不截断过冲，不自动改等待。测量时请保持光纤/传感器静止。')
            self.measure_current_wait_levels.clicked.connect(self.start_current_wait_levels)
            self.current_wait_results = QtWidgets.QPushButton('查看当前等待到达比例')
            self.current_wait_results.setEnabled(False)
            self.current_wait_results.clicked.connect(self.show_current_wait_levels)
            self.beta_controls_row = flow_row([
                section_label('稳定时间'), self.beta_fast_dac,
                self.beta_timing_results, self.beta_apply_t90_waits,
                self.measure_current_wait_levels, self.current_wait_results,
            ])
            self.beta_controls_row.hide()
            self.beta_notice = QtWidgets.QLabel(
                '90%稳定时间测量 → 一键应用逐点等待；含首点回绕、高电压点测量一次。'
                '90%按本点ADC相对0计算，采集仍是相邻切换；ADC稳定不等于波长精度认证；'
                '稳态电压>0.2 V才测90%，≤0.2 V固定300 µs；'
                '图上粉色菱形＝90%测量未通过（悬停查看原因）；低电压固定等待不是失败。'
                '默认850 µs；单次ADC需v1.0.115 Beta固件；合并DAC未通过带光对照，已停用。')
            self.beta_notice.setWordWrap(True)
            layout.addWidget(self.beta_notice)
            self.beta_notice.hide()
            self.beta_apply_t90_waits.clicked.connect(self.apply_beta_t90_waits)
        self.developer_budget_button = QtWidgets.QPushButton('帧耗时与温控预算…')
        self.developer_budget_button.clicked.connect(self.show_developer_budget)
        self.developer_budget_row = flow_row([
            section_label('性能诊断'), self.developer_budget_button,
        ])
        self.developer_budget_row.hide()
        self.fit_button = QtWidgets.QPushButton('适合当前曲线（横纵轴）')
        self.fit_button.setToolTip(
            '根据当前可见曲线调整横纵轴；扫描预热后适配一次，随后锁定纵轴，'
            '按压时不自动缩放，以保留真实高度变化。'
        )
        self.fit_button.clicked.connect(self.fit_values)
        self.peak_view = QtWidgets.QComboBox()
        self.peak_view.addItem('查看全部9峰')
        for group in range(9):
            self.peak_view.addItem(f'放大第{group+1}峰')
        self.peak_view.currentIndexChanged.connect(self.fit_peak_view)
        self.peak_view.setFixedWidth(142)
        self.peak_view.setToolTip('只改变横轴显示范围，不改变实际扫描点。')
        self.show_dense_selection = QtWidgets.QPushButton('密集光谱与45点')
        self.show_dense_selection.setCheckable(True)
        self.show_dense_selection.setChecked(True)
        self.show_dense_selection.setEnabled(False)
        self.show_dense_selection.setToolTip(
            '新采谱后直接在曲线上点击45次放置点；完成后可单击一个已放点，'
            '再按键盘左/右键逐格移动。参考线只在主动点击采集按钮时更新。'
        )
        self.show_dense_selection.toggled.connect(self.redraw_dense_selection)
        self.show_dense_selection.setMinimumWidth(160)
        self.show_dense_failures = QtWidgets.QPushButton('未通过点（红×）')
        self.show_dense_failures.setCheckable(True)
        self.show_dense_failures.setChecked(False)
        self.show_dense_failures.setToolTip(
            '仅切换密集采集未通过点的红色×标记；'
            '不删除失败记录，不影响选点、参考线或开关测试。'
        )
        self.show_dense_failures.toggled.connect(self.redraw_dense_selection)
        self.show_dense_failures.setMinimumWidth(116)
        self.show_dense_failures.hide()  # Removed from customer controls.
        self.manual_selection_label = QtWidgets.QLabel('手动选点 0/45')
        self.manual_undo_button = QtWidgets.QPushButton('撤销上一个点')
        self.manual_clear_button = QtWidgets.QPushButton('清空手动点')
        self.auto_select_button = QtWidgets.QPushButton('自动选点')
        self.auto_select_button.setToolTip(
            '识别突出度至少5 dB的峰，取最强9峰；默认Makima拟合后，在dBm视觉上半段内'
            '选中心点和左右三对高度配对实测点，波长不要求等距。'
            '峰底与峰顶的dBm中点为半高线，左右最低点固定为线以上最近的真实点；'
            '到峰顶间按高度均匀分三层，不要求波长等距。点数不足请减小密集采样间隔。'
            '仅生成点位路线，不改原始光谱；可拖动、撤销或重新手动选点。')
        self.auto_select_button.setEnabled(False)
        self.auto_select_button.clicked.connect(self.auto_select_points)
        self.manual_undo_button.setEnabled(False)
        self.manual_clear_button.setEnabled(False)
        self.manual_undo_button.clicked.connect(self.undo_manual_selection)
        self.manual_clear_button.clicked.connect(self.clear_manual_selection)
        flow_row([
            section_label('测试与记录'), self.save_finger_button,
            self.load_finger_button, self.reference_line_button,
            self.start_button, self.stop_button, self.stop_dense_button,
        ])
        self.curve_controls_row = flow_row([
            section_label('曲线与点位'), self.fit_button, self.peak_view,
            self.show_dense_selection,
            self.auto_select_button, self.show_reference, self.show_peak_fits,
            self.advanced_toggle,
        ])
        flow_row([
            section_label('手动选点'), self.manual_selection_label,
            self.manual_undo_button, self.manual_clear_button,
        ], self.advanced_panel_layout)
        layout.addWidget(self.advanced_panel)
        self.advanced_toggle.toggled.connect(self.toggle_advanced_panel)
        self.status = QtWidgets.QLabel("等待当前密集光谱；尚未验证帧率或光学精度")
        self.status.setObjectName('acquisitionStatus')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.finger_record_label = QtWidgets.QLabel('当前手指记录：未命名（未载入保存记录）')
        self.finger_record_label.setStyleSheet('color: #2563eb;')
        layout.addWidget(self.finger_record_label)
        self.finger_record_label.hide()
        self.peak_fit_method = QtWidgets.QComboBox(self)
        self.peak_fit_method.addItem('高斯＋基线', 'gaussian')
        self.peak_fit_method.addItem('Makima 样条', 'makima')
        self.peak_fit_method.setCurrentIndex(1)
        self.peak_fit_method.hide()
        self.peak_fit_method.setToolTip(
            '只切换实时五点拟合的显示方法，不修改采样、参考线、保存的点位或验收。'
            'Makima 能显示不对称峰形，但中心位置尚未通过波长计精度验证。'
        )
        self.peak_fit_method.currentIndexChanged.connect(self.change_peak_fit_method)
        self.peak_fit_status = QtWidgets.QLabel('等待实时帧；拟合波长不是波长计实测值。')
        self.peak_fit_status.setObjectName('acquisitionStatus')
        self.peak_fit_status.setWordWrap(True)
        layout.addWidget(self.peak_fit_status)
        self.peak_fit_table = QtWidgets.QTableWidget(3, 9)
        self.peak_fit_table.setHorizontalHeaderLabels([f'峰{i+1}' for i in range(9)])
        self.peak_fit_table.setVerticalHeaderLabels(['中心 nm（估计）', 'R²（形状诊断）', '状态'])
        self.peak_fit_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.peak_fit_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.peak_fit_table.setMaximumHeight(150)
        self.peak_fits = []
        self.plot = pg.PlotWidget()
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.plot.setMinimumHeight(260)
        self.plot.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.plot.setLabel('bottom', '标定实测波长', units='nm')
        self.plot.setLabel('left', 'CH1 ADC × 数字显示倍数', units='code')
        plot_item = self.plot.getPlotItem()
        plot_item.layout.setContentsMargins(10, 8, 16, 10)
        left_axis = self.plot.getAxis('left')
        bottom_axis = self.plot.getAxis('bottom')
        left_axis.enableAutoSIPrefix(False)
        bottom_axis.enableAutoSIPrefix(False)
        axis_font = QtGui.QFont('Microsoft YaHei UI', 9)
        for axis in (left_axis, bottom_axis):
            axis.setStyle(
                tickFont=axis_font, tickTextOffset=8,
                autoExpandTextSpace=True, autoReduceTextSpace=False,
            )
        left_axis.setWidth(92)
        bottom_axis.setHeight(62)
        self.plot.getViewBox().setDefaultPadding(0.035)
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
        self.manual_selection_curve = self.plot.plot(
            pen=None, symbol='o', symbolSize=11,
            symbolPen=pg.mkPen((255, 255, 255), width=2),
            symbolBrush=pg.mkBrush((59, 130, 246, 150)),
            name='待完成的手动45点',
        )
        self.manual_selection_curve.setZValue(31)
        self.reference_failed_curve = self.plot.plot(
            pen=None, symbol='x', symbolSize=13,
            symbolPen=pg.mkPen((239, 68, 68), width=3),
            name='45点参考未通过',
        )
        self.reference_failed_curve.setZValue(40)
        self.t90_failed_markers = pg.ScatterPlotItem(
            symbol='d', size=16, pen=pg.mkPen('#ff4fd8', width=2),
            brush=pg.mkBrush(255, 79, 216, 45), hoverable=True,
            hoverPen=pg.mkPen('w', width=3),
            tip=lambda x, y, data: data['tooltip'], name='90%测量未通过（粉色菱形）')
        self.t90_failed_markers.setZValue(45)
        self.t90_failed_markers.sigClicked.connect(self._select_t90_failed_point)
        self.plot.addItem(self.t90_failed_markers)
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
        self.plot.scene().sigMouseClicked.connect(self.handle_dense_plot_click)
        self.table = QtWidgets.QTableWidget(45, 6)
        self.table.setHorizontalHeaderLabels(['峰', '物理索引', '目标nm', '实测标定nm', 'CH1 ADC', '参考ADC'])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setMaximumHeight(210)
        self.data_tabs = QtWidgets.QTabWidget()
        self.data_tabs.addTab(self.peak_fit_table, '9峰波长与拟合质量')
        self.data_tabs.addTab(self.table, '45点原始ADC')
        # Keep all three result rows visible, not just the tab/header strip.
        self.data_tabs.setMinimumHeight(160)
        self.data_tabs.setMaximumHeight(190)
        self.plot_data_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.plot_data_splitter.setChildrenCollapsible(False)
        # The plot should absorb the page's spare height, not impose the
        # standalone plot/table preferred height on an embedded window.
        self.plot_data_splitter.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Ignored
        )
        self.plot_data_splitter.addWidget(self.plot)
        self.plot_data_splitter.addWidget(self.data_tabs)
        self.plot_data_splitter.setStretchFactor(0, 8)
        self.plot_data_splitter.setStretchFactor(1, 1)
        self.plot_data_splitter.setSizes([500, 160])
        layout.addWidget(self.plot_data_splitter, 1)
        self.save_finger_button.clicked.connect(self.save_finger_capture)
        self.load_finger_button.clicked.connect(self.load_finger_capture)
        self.start_button.clicked.connect(self.start_scan)
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
        if self.persistence_enabled:
            self._load_scan_preferences()
        if self.beta:
            self._apply_startup_scan_defaults()
        self._apply_startup_wait_defaults()
        for signal in (
            self.channel_combo.currentIndexChanged,
            self.scan_start_nm.valueChanged,
            self.scan_stop_nm.valueChanged,
            self.equal_interval_nm.valueChanged,
            self.equal_interval_settle.valueChanged,
        ):
            signal.connect(self._save_scan_preferences)
        for signal in (
            self.digital_gain.valueChanged,
            self.reference_gain.valueChanged,
            self.first_delay.valueChanged,
            self.boundary_delay.valueChanged,
            self.scan_start_nm.valueChanged,
            self.scan_stop_nm.valueChanged,
            self.equal_interval_nm.valueChanged,
            self.equal_interval_settle.valueChanged,
            self.use_point_waits.toggled,
            self.show_reference.toggled,
            self.show_dense_selection.toggled,
            self.show_dense_failures.toggled,
        ):
            signal.connect(lambda _value=None: self._save_last_route_session())
        self._sync_channel_ui()

    def _apply_startup_scan_defaults(self):
        """Operator-requested next-scan defaults, not relabeling saved spectra."""
        controls = (self.channel_combo, self.scan_start_nm, self.scan_stop_nm,
                    self.equal_interval_nm, self.equal_interval_settle)
        blockers = [QtCore.QSignalBlocker(control) for control in controls]
        channel_changed = self.selected_channel() != BETA_STARTUP_SCAN['signal_channel']
        self.channel_combo.setCurrentIndex(self.channel_combo.findData(BETA_STARTUP_SCAN['signal_channel']))
        self.scan_start_nm.setValue(BETA_STARTUP_SCAN['scan_start_nm'])
        self.scan_stop_nm.setValue(BETA_STARTUP_SCAN['scan_stop_nm'])
        self.equal_interval_nm.setValue(BETA_STARTUP_SCAN['equal_interval_nm'])
        self.equal_interval_settle.setValue(BETA_STARTUP_SCAN['equal_interval_settle_s'])
        del blockers
        if channel_changed:
            self.change_signal_channel()  # Never reuse another ADC channel's route.

    def _apply_startup_wait_defaults(self):
        # Keep saved diagnostic tables intact, but start with the uniform
        # delays requested by the operator instead of a hidden per-point override.
        controls = (self.first_delay, self.boundary_delay, self.use_point_waits)
        blockers = [QtCore.QSignalBlocker(control) for control in controls]
        self.first_delay.setValue(DEFAULT_FIRST_DELAY_US)
        self.boundary_delay.setValue(DEFAULT_BOUNDARY_DELAY_US)
        self.use_point_waits.setChecked(False)
        del blockers
        self.refresh_wait_controls()

    def _load_scan_preferences(self):
        """Future scan settings are independent of saved raw spectra/routes."""
        try:
            payload = json.loads(SCAN_PREFERENCES_PATH.read_text(encoding='utf-8'))
            if payload.get('schema') != 'finger_stress_scan_preferences_v1':
                return False
            channel = int(payload['signal_channel'])
            start, stop, spacing, settle = (
                float(payload[key]) for key in (
                    'scan_start_nm', 'scan_stop_nm',
                    'equal_interval_nm', 'equal_interval_settle_s',
                )
            )
            if (channel not in range(4)
                    or not all(np.isfinite([start, stop, spacing, settle]))
                    or not 1525 <= start < stop <= 1565
                    or not .02 <= spacing <= .10 or not 0 <= settle <= 10):
                return False
        except (OSError, ValueError, TypeError, KeyError):
            return False
        # Block persistence during restore, but use the normal channel handler
        # to invalidate a different channel's in-memory route, never relabel it.
        channel_changed = self.selected_channel() != channel
        restoring = self._restoring_session
        self._restoring_session = True
        try:
            self.channel_combo.setCurrentIndex(self.channel_combo.findData(channel))
            self.scan_start_nm.setValue(start)
            self.scan_stop_nm.setValue(stop)
            self.equal_interval_nm.setValue(spacing)
            self.equal_interval_settle.setValue(settle)
        finally:
            self._restoring_session = restoring
        if channel_changed:
            self.change_signal_channel()
        return True

    def _save_scan_preferences(self, _value=None):
        if not self.persistence_enabled or self._restoring_session:
            return
        payload = {
            'schema': 'finger_stress_scan_preferences_v1',
            'signal_channel': self.selected_channel(),
            'scan_start_nm': float(self.scan_start_nm.value()),
            'scan_stop_nm': float(self.scan_stop_nm.value()),
            'equal_interval_nm': float(self.equal_interval_nm.value()),
            'equal_interval_settle_s': float(self.equal_interval_settle.value()),
        }
        try:
            SCAN_PREFERENCES_PATH.parent.mkdir(parents=True, exist_ok=True)
            temporary = SCAN_PREFERENCES_PATH.with_suffix('.tmp')
            temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
            temporary.replace(SCAN_PREFERENCES_PATH)
        except OSError as exc:
            self.status.setText(f'采谱参数未能保存：{exc}')

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
        method_note = (
            'Makima 为实验性显示方法，能贴合不对称五点，但中心位置尚未验证；'
            if self.peak_fit_method.currentData() == 'makima' else
            '高斯＋基线拟合；'
        )
        self.show_peak_fits.setToolTip(
            f'使用原始{name} ADC和旧标定波长坐标；{method_note}'
            '所有有限五点排布都会尝试拟合，有质量疑问时用⚠标记。'
            '不修改采样点、不补偿原始幅值；显示位数不代表精度。'
        )
        if self.plan is not None:
            self._sync_dynamic_ui()
        else:
            self.title_label.setText(
                f'手指应力测试模式 · {self.runtime_machine_label} · {name} · 等待光谱选点'
            )
        self._update_display_unit_ui()

    def change_peak_fit_method(self, _index=None):
        """Change only the live display fitter, never the route or reference."""

        makima = self.peak_fit_method.currentData() == 'makima'
        self.peak_fit_table.setVerticalHeaderLabels([
            '中心 nm（估计）', 'R²（形状诊断）' if makima else 'R²', '状态'
        ])
        self.clear_peak_fits(
            'Makima实验性拟合：等待下一帧' if makima else '高斯拟合：等待下一帧'
        )
        self._sync_channel_ui()

    def toggle_advanced_panel(self, expanded):
        """Keep infrequently used controls available without shrinking the plot."""

        self.advanced_panel.setVisible(bool(expanded))
        self.advanced_toggle.setText(
            '收起参数与显示 ▾' if expanded else '更多参数与显示 ▸'
        )
        self.plot.setMinimumHeight(170 if expanded else 260)
        self.updateGeometry()

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
        self._reset_dense_draw_state()
        self.manual_selection_indices = []
        self.manual_selection_history = []
        self.manual_selection_active = False
        self.reference_failed_points = []
        self._live_reference_line_rows = []
        self._reference_line_capture_active = False
        self.latest_channel = channel
        self.reference_channel = channel
        self.dense_channel = channel
        self.point_waits = None
        self.point_wait_route = None
        self.use_point_waits.setChecked(False)
        self._clear_selection_markers()
        self.dense_curve.clear()
        self.dense_failed_curve.clear()
        self.manual_selection_curve.clear()
        self.reference_failed_curve.clear()
        self._sync_manual_selection_controls()
        for curve in (*self.curves, *self.reference_curves, *self.fitted_curves):
            curve.clear()
        self.start_button.setEnabled(False)
        self.reference_line_button.setEnabled(False)
        self.save_finger_button.setEnabled(False)
        self.clear_peak_fits(f'已切换到{self.channel_name(channel)}，等待重新采集光谱')
        self._sync_channel_ui()
        self.status.setText(
            f'已切换到{self.channel_name(channel)}；通道的峰位和幅值记录相互独立，'
            '请按设定范围等间隔重采光谱，再在谱线上手动放置45点。'
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
            self.first_delay.setValue(config.get('first_delay_us', DEFAULT_FIRST_DELAY_US))
            self.boundary_delay.setValue(boundary_wait_from_settings(config))
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
                    '开始扫描前可采集当前45点10 ms单次ADC参考线。'
                )
                return True
            if config.get('selection_plan'):
                self.plan = load_saved_plan(
                    resource_path(config['selection_plan'])
                )
                self.render_plan(ready=False)
                self.status.setText(
                    '已载入旧密集光谱45点，仅供预览；峰位会漂移，开始扫描前请重采/选择当前密集光谱，'
                    '或先采集当前45点10 ms单次ADC参考线。'
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
                    '已用保存的CH1光谱选出9×5点，仅供预览；开始扫描前须采集当前45点10 ms单次ADC参考线。'
                )
                return True
            if config.get('calibration'):
                self.plan = load_calibrated_plan(
                    resource_path(config['calibration'])
                )
                self.render_plan(ready=False)
                self.status.setText(
                    '已载入45点实测标定表，仅供预览；开始扫描前须采集当前45点10 ms单次ADC参考线。'
                )
                return True
            self.status.setText(
                f'{self.runtime_machine_label}尚无临时模式点表；请按范围等间隔重采光谱并手动放置45点，'
                '或载入该机保存的手指记录。'
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
            self._reset_dense_draw_state()
            self.manual_selection_indices = []
            self.manual_selection_history = []
            self.manual_selection_active = False
            self.reference_failed_points = []
            self._live_reference_line_rows = []
            self._reference_line_capture_active = False
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
            self.manual_selection_curve.clear()
            self.reference_failed_curve.clear()
            self._sync_manual_selection_controls()
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

    def set_developer_mode(self, enabled):
        """Visibility only: never reset an active wait table or enable fast DAC."""
        self.developer_mode = bool(enabled)
        for name in ('point_diagnostics_row', 'beta_controls_row', 'beta_notice',
                     'developer_budget_row'):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setVisible(self.developer_mode)

    def _developer_waits(self, count):
        options = getattr(self.worker, 'options', {})
        if not isinstance(options, dict):
            options = {}
        waits = options.get('point_delays_us')
        if waits is not None and len(waits) == count:
            return list(waits)
        if self.use_point_waits.isChecked() and self.point_waits is not None:
            if len(self.point_waits) == count:
                return list(self.point_waits)
        per_peak = self.points_per_peak() or 7
        first = options.get('first_delay_us', self.first_delay.value())
        boundary = (first + options.get('boundary_extra_us', 0)
                    if options else self.boundary_delay.value())
        return [boundary if i % per_peak == 0 else first for i in range(count)]

    def _observe_developer_frame(self, frame):
        from developer_diagnostics import measured_budget
        waits = self._developer_waits(len(frame.get('records', ())))
        previous = self._developer_previous_frame
        self._developer_timing_budget = None
        if previous is not None and previous[1] == waits:
            try:
                self._developer_timing_budget = measured_budget(
                    previous[0], frame, waits,
                    spacing_us=50)
            except (ValueError, KeyError, TypeError):
                pass  # Missing/skipped frames must not fabricate a measured budget.
        self._developer_previous_frame = (frame, waits)

    def show_developer_budget(self):
        if not self.developer_mode:
            return
        from developer_diagnostics import nominal_budget
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('开发者模式 · 帧耗时与温控预算')
        dialog.resize(700, 590)
        layout = QtWidgets.QVBoxLayout(dialog)
        note = QtWidgets.QLabel(
            '只读诊断，不改变DAC、等待或温控。400 kHz、5路分别写入；'
            '850 µs从最后一路DAC写完后开始。总线理论耗时不含起止条件和软件开销。')
        note.setWordWrap(True)
        layout.addWidget(note)
        reference = QtWidgets.QLabel(
            '850 µs参考：45点等待38.25 ms＋DAC总线20.25 ms＝58.50 ms；'
            '63点等待53.55 ms＋DAC总线28.35 ms＝81.90 ms。'
            '这只是下限，不是实测帧周期或可保证帧率。')
        reference.setWordWrap(True)
        layout.addWidget(reference)
        heading = QtWidgets.QLabel()
        heading.setWordWrap(True)
        layout.addWidget(heading)
        table = QtWidgets.QTableWidget(0, 2)
        table.setHorizontalHeaderLabels(['项目', '时间 / 帧'])
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        layout.addWidget(table)
        thermal = QtWidgets.QLabel(
            '温控参考估算（不是实时CPU负载）：正常约2%～4%；'
            '完整读温的总线忙等约8.66 ms/次，分步骤间歇执行。'
            '传感器内部12 ms转换不阻塞扫描，风扇PWM由定时器＋DMA持续输出。'
            '实际插入的温控耗时已包含在DAC及点间处理、帧间开销中，不再重复相加。')
        thermal.setWordWrap(True)
        layout.addWidget(thermal)

        def refresh():
            measured = self._developer_timing_budget
            rows = []
            if measured is not None:
                heading.setText(f"最近相邻板端帧：{measured['point_count']}点，"
                                f"周期{measured['period_ms']:.2f} ms，{measured['hz']:.2f} Hz。"
                                'DAC项含点间处理，帧间项含传输/维护，尚未独立测量温控CPU。')
                for label, key in (
                    ('DAC写入及点间处理', 'dac_and_interpoint_ms'),
                    ('设定等待', 'prescribed_wait_ms'),
                    ('ADC及等待附加开销', 'adc_and_hold_overhead_ms'),
                    ('帧内尾部处理', 'frame_tail_ms'),
                    ('帧间开销（含处理/传输）', 'frame_gap_ms'),
                    ('完整帧周期', 'period_ms'),
                ):
                    rows.append((label, f'{measured[key]:.3f} ms'))
            else:
                actual_count = self.point_count()
                count = actual_count or 63
                budget = nominal_budget(self._developer_waits(count))
                heading.setText((f'{count}点理论预算；' if actual_count else
                                 '当前没有有效点表，以下为63点参考预算；')
                                + '暂无有效相邻板端帧。'
                                '需要开始正常采集后才能显示实测拆分，主机绘图时间不冒充板端时间。')
                rows = [('当前设定等待', f"{budget['wait_ms']:.3f} ms"),
                        ('DAC总线理论耗时', f"{budget['dac_wire_ms']:.3f} ms"),
                        ('两项合计下限（不含ADC等）', f"{budget['minimum_ms']:.3f} ms")]
            table.setRowCount(len(rows))
            for i, (label, value) in enumerate(rows):
                table.setItem(i, 0, QtWidgets.QTableWidgetItem(label))
                table.setItem(i, 1, QtWidgets.QTableWidgetItem(value))
        refresh()
        timer = QtCore.QTimer(dialog)
        timer.setInterval(500)
        timer.timeout.connect(refresh)
        timer.start()
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.button(QtWidgets.QDialogButtonBox.Close).setText('关闭')
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec_()

    def before_machine_switch(self):
        """Persist the current unit before the global runtime id changes."""

        self._save_last_route_session(ready=self.plan_ready)

    def reload_machine_profile(self, machine_label, parameter_note):
        """Switch labels, defaults and saved records as one atomic profile."""

        self.runtime_machine_label = str(machine_label)
        self._developer_previous_frame = None
        self._developer_timing_budget = None
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
            self._load_scan_preferences()
        self._apply_startup_wait_defaults()
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
                'adc': (f'{channel_name} ADC码', '参考ADC码'),
                'voltage': (f'{channel_name}电压 V', '参考电压 V'),
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
        failed_reference_by_order = {
            int(item.get('order', -1)): item
            for item in self.reference_failed_points
        }
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
                if column == 5 and index in failed_reference_by_order:
                    failure = failed_reference_by_order[index].get('row', {})
                    reason = failure.get(
                        'failure_reason', '稳定性或饱和检查未通过'
                    )
                    item.setForeground(QtGui.QBrush(QtGui.QColor('#dc2626')))
                    item.setBackground(QtGui.QBrush(QtGui.QColor('#fee2e2')))
                    item.setToolTip(
                        item.toolTip() + f' 未通过但已继续采集：{reason}'
                    )
                self.table.setItem(index, column, item)

    def _update_plot_unit_label(self):
        if self.display_mode == 'adc':
            self.plot.setLabel(
                'left',
                f'{self.channel_name()} ADC', units='code'
            )
        elif self.display_mode == 'voltage':
            self.plot.setLabel('left', f'{self.channel_name()} ADC输入电压', units='V')
        else:
            self.plot.setLabel(
                'left', f'{self.channel_name()}估算光功率', units='dBm'
            )

    def point_count(self):
        return len(self.plan.get('rows', [])) if isinstance(self.plan, dict) else 0

    def peak_count(self):
        return self.point_count() // self.points_per_peak()

    def points_per_peak(self):
        return (points_per_peak_for_rows(self.plan['rows']) if self.plan is not None and not self.manual_selection_active
                else self.manual_points_per_peak)

    def manual_point_count(self):
        return 9 * self.manual_points_per_peak

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
        capture_plan = None if self.manual_selection_active else self.plan
        dense_x = np.asarray(
            self.dense_wavelengths if self.dense_wavelengths is not None
            else capture_plan.get('reference_wavelengths_nm', [])
            if isinstance(capture_plan, dict) else [], dtype=float
        )
        dense_y = np.asarray(
            self.dense_values if self.dense_values is not None
            else capture_plan.get('reference_values', [])
            if isinstance(capture_plan, dict) else [], dtype=float
        )
        if not self._valid_dense_spectrum(dense_x, dense_y):
            raise ValueError('需要先完成一条步长不大于0.10 nm的有效密集谱线')
        point_count = len(capture_plan.get('rows', [])) if capture_plan else 0
        peak_count = point_count // points_per_peak_for_rows(capture_plan['rows'])
        if capture_plan is not None and point_count not in (set(range(5, 46, 5)) | set(range(7, 64, 7))):
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
            reference_wavelengths = [float(row['measured_nm']) for row in capture_plan['rows']]
        plan = None
        if capture_plan is not None:
            plan = stamp_machine_metadata(
                validate_plan_payload(capture_plan),
                ('fullband_2001', f'{peak_count}_peak_route'),
            )
            plan['reference_wavelengths_nm'] = dense_x.tolist()
            plan['reference_values'] = dense_y.tolist()
            plan['finger_label'] = str(finger_label)
            plan['signal_channel'] = self.selected_channel()
        categories = (
            'fullband_2001',
            f'{peak_count}_peak_route' if plan is not None
            else 'dense_spectrum',
        )
        return stamp_machine_metadata({
            'schema': 'temporary_test_finger_capture_v1',
            'saved_s': time.time(),
            'finger_label': str(finger_label),
            'ready': bool(capture_plan is not None and self.plan_ready and has_reference),
            'point_count': point_count,
            'peak_count': peak_count,
            'plan': plan,
            'dense_wavelengths_nm': dense_x.tolist(),
            'dense_values': dense_y.tolist(),
            'dense_source': str(
                self.dense_source
                or (capture_plan.get('reference_source', '')
                    if isinstance(capture_plan, dict) else '')
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
            'reference_failed_points': (
                list(self.reference_failed_points) if has_reference else []
            ),
            'feedback_selector': self.effective_feedback_selector(),
            'dense_feedback_selector': int(self.dense_feedback_selector),
            'reference_feedback_selector': int(self.reference_feedback_selector),
            'digital_display_gain': float(self.digital_gain.value()),
            'reference_display_gain': float(self.reference_gain.value()),
            'display_unit': self.display_mode,
            'first_delay_us': int(self.first_delay.value()),
            'boundary_wait_us': int(self.boundary_delay.value()),
            'boundary_extra_us': max(0, int(self.boundary_delay.value()) - int(self.first_delay.value())),
            'scan_start_nm': float(self.scan_start_nm.value()),
            'scan_stop_nm': float(self.scan_stop_nm.value()),
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
                and int(payload.get('point_count', 0)) in (45, 63)
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
                + ('，可直接使用）' if payload.get('ready') else
                   '，尚需手动放置45点）' if not payload.get('point_count') else
                   '，尚需采集参考线）')
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
                f'已保存“{finger_label}”：{payload.get("peak_count", 0)}峰、'
                f'{payload.get("point_count", 0)}点、'
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
                        'peak_count', candidate_count // points_per_peak_for_rows(candidate_plan.get('rows', []))
                    ))
                    if (candidate_payload.get('schema')
                            == 'temporary_test_finger_capture_v1'
                            and candidate_count in (45, 63) and candidate_peaks == 9
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
            if auto and (point_count not in (45, 63) or not payload.get('ready')):
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
            self.first_delay.setValue(int(payload.get('first_delay_us', DEFAULT_FIRST_DELAY_US)))
            self.boundary_delay.setValue(boundary_wait_from_settings(payload))
            self.scan_start_nm.setValue(float(payload.get('scan_start_nm', dense_x[0])))
            self.scan_stop_nm.setValue(float(payload.get('scan_stop_nm', dense_x[-1])))
            self.dense_scan_range_nm = (
                float(self.scan_start_nm.value()), float(self.scan_stop_nm.value())
            )
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
                # Raw spectra never trigger automatic peak recognition.  The
                # operator resumes the same explicit 45-click workflow.
                self.plan = None
                self.begin_manual_selection()
            if has_reference:
                self.reference_values = [float(value) for value in references]
                self.reference_wavelengths = [float(value) for value in reference_wavelengths]
                self.reference_failed_points = list(
                    payload.get('reference_failed_points', [])
                )
            else:
                self.reference_values = None
                self.reference_wavelengths = None
                self.reference_failed_points = []
            self.show_dense_selection.setChecked(
                True if self.manual_selection_active
                else bool(payload.get('show_dense_selection', True))
            )
            self.show_dense_failures.setChecked(False)
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
                    '可直接开始手指应力测试。只有主动点击参考线采集按钮才会重新采集。'
                )
            else:
                self.status.setText(
                    f'已载入“{label}”的{len(dense_x)}点'
                    f'{self.channel_name(channel)}密集谱线'
                    + (f'和{self.point_count()}个选点；' if self.plan is not None else '；')
                    + ('仍需采集参考线后才能开始手指应力测试。'
                       if self.plan is not None else
                       '请直接在谱线上手动点击45个点。')
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
        if not point_count or point_count % self.points_per_peak():
            return
        self.title_label.setText(
            f'手指应力测试模式 · {self.runtime_machine_label} · {self.channel_name()} · '
            f'{peak_count}峰×{self.points_per_peak()}点 · 仅扫描选定的{point_count}点'
        )
        self.show_peak_fits.setText(
            f'实时峰形拟合与中心波长（Makima 样条，{self.points_per_peak()}点估计）')
        self.note_label.setText(
            f'固定{point_count}点扫描：原始圆点/实线、参考虚线、拟合点线分开显示；'
            '等待自行调节，允许低于15 Hz。\n'
            '拟合波长基于旧标定坐标，仅为五点估计；双ADC接近不证明波长稳定，原始幅值不作修正。'
        )
        self.reference_line_button.setText(f'采集参考线（{point_count}点）')
        self.edit_point_waits.setText(f'设置{point_count}个点的等待…')
        self.test_switch_timing.setText(f'测量{point_count}点90%稳定时间' if self.beta else
                                        f'测试{point_count}点达到90%时间')
        self.test_switch_timing.setEnabled(
            self.worker is None and self.plan is not None
            and (self.beta or self.selected_channel() == 1)
        )
        self.test_switch_timing.setToolTip(
            '当前高速开关诊断硬件仅测CH1；切换回CH1后可用。'
            if self.selected_channel() != 1 and not self.beta else
            (f'对当前{point_count}点按小波长到大波长测量一次；' if self.beta else
             f'对当前{point_count}点按小波长到大波长连续测试3轮；')
            + (f'含末点回到首点；分析{self.channel_name()}，测完可一键应用逐点等待。'
               '先读稳态电压，>0.2 V才测90%，≤0.2 V固定300 µs；'
               '计算S=0：ADC保持在本点最终值±10%内，允许偏差没有5码下限；不是波长认证。'
               if self.beta else '不做反向回绕审计。CH1主变化达到90%的板上时间会写入逐点等待表。')
        )
        self.show_dense_selection.setText(f'密集光谱与{point_count}点')
        self.show_dense_selection.setToolTip(
            '新采谱后直接在曲线上点击45次放置点；路线完成后单击一个已放点，'
            f'再按键盘左/右键移动。移动只修改{point_count}点路线。'
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
        self.peak_view.addItem(f'查看全部{peak_count}峰')
        for group in range(peak_count):
            self.peak_view.addItem(f'放大第{group + 1}峰')
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
                or self.plan is None or self.manual_selection_active):
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
            'reference_failed_points': (
                list(self.reference_failed_points)
                if reference_values is not None else []
            ),
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
            'boundary_wait_us': int(self.boundary_delay.value()),
            'boundary_extra_us': max(0, int(self.boundary_delay.value()) - int(self.first_delay.value())),
            'scan_start_nm': float(self.scan_start_nm.value()),
            'scan_stop_nm': float(self.scan_stop_nm.value()),
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
            peak_count = point_count // points_per_peak_for_rows(plan['rows'])
            # Three-peak sessions remain named records for explicit loading;
            # application startup always returns to the nine-peak route.
            if point_count not in (45, 63) or peak_count != 9:
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
            raw_reference = bool(plan.get('reference_acquisition_method') == 'equal_interval_single'
                                 and plan.get('reference_stability_evaluated') is False)
            checks = []
            if restored_ready and not raw_reference:
                for group in range(peak_count):
                    selected = plan['rows'][group * points_per_peak_for_rows(plan['rows']):group * points_per_peak_for_rows(plan['rows']) + points_per_peak_for_rows(plan['rows'])]
                    check = assess_peak_shape(
                        [row['measured_nm'] for row in selected],
                        references[group * points_per_peak_for_rows(plan['rows']):group * points_per_peak_for_rows(plan['rows']) + points_per_peak_for_rows(plan['rows'])],
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
            self.first_delay.setValue(int(payload.get('first_delay_us', DEFAULT_FIRST_DELAY_US)))
            self.boundary_delay.setValue(boundary_wait_from_settings(payload))
            dense_default_start = float(dense_x[0]) if dense_x.size else 1525.0
            dense_default_stop = float(dense_x[-1]) if dense_x.size else 1565.0
            self.scan_start_nm.setValue(float(payload.get(
                'scan_start_nm', dense_default_start
            )))
            self.scan_stop_nm.setValue(float(payload.get(
                'scan_stop_nm', dense_default_stop
            )))
            self.dense_scan_range_nm = (
                float(self.scan_start_nm.value()), float(self.scan_stop_nm.value())
            )
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
            self.show_dense_failures.setChecked(False)
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
                self.reference_failed_points = list(
                    payload.get('reference_failed_points', [])
                )
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
                if raw_reference:
                    self.status.setText(
                        f'已恢复{point_count}点及10 ms单次ADC参考线；未做稳定性/峰形验收，可直接开始手指应力测试。'
                        '软件启动不会自动出光。'
                    )
                elif manual_reference_reuse:
                    self.status.setText(
                        f'已恢复键盘调整后的{point_count}点，可直接开始手指应力测试；参考虚线仍是调整前数据。'
                        '只有主动点击“采集并绘制参考线”才会重新采集。'
                    )
                elif reference_gate_bypassed:
                    self.status.setText(
                        f'已恢复{point_count}点和参考线；参考采集的稳定/峰形检查未通过，但按设置仍可直接开始手指应力测试。'
                        '只有主动点击“采集并绘制参考线”才会重新采集。'
                    )
                else:
                    self.status.setText(
                        f'已自动恢复上次{peak_count}/{peak_count}通过的{point_count}点、密集光谱和参考线；可直接开始手指应力测试。'
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
                    group=flat_index // self.points_per_peak() + 1, index=int(point.index),
                    target_nm=float(point.target_nm), measured_nm=float(point.measured_nm),
                    codes=list(point.codes),
                ))
                values.append(float(recorded['ch1_adc_code']))
                previous = index
            for group in range(9):
                group_rows = selected[group * self.points_per_peak():group * self.points_per_peak() + self.points_per_peak()]
                check = assess_peak_shape(
                    [row['measured_nm'] for row in group_rows],
                    values[group * self.points_per_peak():group * self.points_per_peak() + self.points_per_peak()],
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
                    '已从上次9/9通过的参考线自动恢复45点；可直接开始手指应力测试。'
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
        if self.beta:
            self.beta_single_adc.setEnabled(idle)
            self.beta_fast_dac.setEnabled(False)  # Optical A/B rejected both tested burst builds.
            timing = (self.plan or {}).get('switch_90pct_timing', {})
            self.beta_timing_results.setEnabled(idle and bool(timing.get('points')))
            self.beta_apply_t90_waits.setEnabled(idle and bool(timing.get('t90_waits')))
            self.measure_current_wait_levels.setEnabled(idle and self.plan is not None)
            self.current_wait_results.setEnabled(
                idle and bool((self.plan or {}).get('current_wait_levels', {}).get('points')))

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
        waits = self._uniform_point_waits()
        self.point_waits = waits
        self.point_wait_route = key
        return True

    def _uniform_point_waits(self):
        """Cross-peak waits replace, rather than add to, within-peak waits."""
        return [int(self.boundary_delay.value()) if index % self.points_per_peak() == 0
                else int(self.first_delay.value())
                for index in range(self.point_count())]

    def point_route_key(self):
        return [(r['index'], tuple(r['codes'])) for r in self.plan['rows']]

    def configure_point_waits(self):
        if self.worker is not None or self.plan is None:
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('逐点等待：固定波长，仅改变采样时刻')
        layout = QtWidgets.QVBoxLayout(dialog)
        settle_hints = load_forward_settle_hints()
        t90 = (self.plan or {}).get('switch_90pct_timing', {}).get('t90_waits', {}) if self.beta else {}
        if t90:
            from temporary_t90_waits import T90_SCHEMA
            if t90.get('schema') != T90_SCHEMA:
                t90 = {}  # Old formulas/multi-round results are not current single measurements.
        if t90 and (t90.get('route_key') != [[r['index'], list(r['codes'])] for r in self.plan['rows']]
                or t90.get('machine_id') != get_runtime_machine_id()
                or t90.get('signal_channel') != self.selected_channel()
                or t90.get('feedback_selector') != self.effective_feedback_selector()
                or t90.get('fast_dac_writes') != (self.beta_single_adc.isChecked()
                                                    and self.beta_fast_dac.isChecked())):
            t90 = {}
        layout.addWidget(QtWidgets.QLabel(
            '每点50～15000 µs（最大15 ms），25 µs步长；总等待不限，允许低于15 Hz。\n'
            + ('当前90%稳定时间来自相同路线/通道/档位的单次实测；测量后可一键应用。'
               if self.beta else
               '“稳定时间”来自既有板上高速采样，只作提示且不会自动改写本次等待；'
               '同目标表示此前一点与当前路线不同。')))
        source = QtWidgets.QLabel(
            (f'当前{self.channel_name()} 90%稳定时间（含首点回绕）' if t90 else
             '暂无与当前路线/通道/档位匹配的90%稳定时间，请先测量。')
            if self.beta else settle_hints['source_label'])
        source.setStyleSheet('color: #6b7280;')
        layout.addWidget(source)
        point_count = self.point_count()
        table = QtWidgets.QTableWidget(point_count, 5)
        table.setHorizontalHeaderLabels([
            '峰/点', '正向跳转波长 nm', '物理索引',
            '实测90%稳定时间' if self.beta else '正向跳转主边沿/稳定时间', '本次使用等待 µs'
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
            table.setItem(i, 0, QtWidgets.QTableWidgetItem(f'{i// self.points_per_peak()+1}/{i% self.points_per_peak()+1}'))
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
            if self.beta:
                from temporary_t90_waits import REASON_LABELS
                measured = t90.get('points', [])
                point = measured[i] if len(measured) == point_count else {}
                stable = point.get('settled_90pct_after_target_write_us')
                hint_text = ('低电压固定300 µs（未测90%）'
                             if point.get('wait_source') == 'low_voltage_fixed_300us' else
                             '未可靠测得' if stable is None else f'90%稳定 {stable:g} µs')
                reasons = point.get('reasons', [])
                hint_tooltip = ('稳态电压≤0.2 V，使用指定的300 µs等待，不是实测90%时间。'
                                if point.get('wait_source') == 'low_voltage_fixed_300us' else
                                '；'.join(REASON_LABELS.get(reason, reason) for reason in reasons)
                                if reasons else '单次实测，目标5路DAC写完后计时；ADC稳定不是波长精度认证。'
                                if point else '请先测量当前路线的90%稳定时间。')
            hint_item = QtWidgets.QTableWidgetItem(hint_text)
            hint_item.setToolTip(hint_tooltip)
            table.setItem(i, 3, hint_item)
            spin = QtWidgets.QSpinBox()
            spin.setRange(50, 15000)
            spin.setSingleStep(25)
            spin.setValue(values[i] if values is not None else
                          self._uniform_point_waits()[i])
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
        self._redraw_t90_failures()
        self.fit_peak_view()
        if self.dense_values is not None:
            self.fit_values()
        self.status.setText(f"已选定{peak_count}×{self.points_per_peak()}点：" + ', '.join(f'{v:.3f}' for v in self.plan['peaks_nm']) + " nm；精度待实测")
        self._sync_manual_selection_controls()

    def _clear_selection_markers(self):
        self.t90_failed_markers.clear()
        for marker in self.selection_markers:
            self.plot.removeItem(marker)
        self.selection_markers = []
        self.selected_selection_point = None

    def _measurement_identity_matches(self, result):
        if not self.beta or self.plan is None:
            return False
        return (result.get('route_key') == [[r['index'], list(r['codes'])] for r in self.plan['rows']]
                and result.get('machine_id') == get_runtime_machine_id()
                and result.get('signal_channel') == self.selected_channel()
                and result.get('feedback_selector') == self.effective_feedback_selector()
                and result.get('fast_dac_writes') == bool(
                    self.beta_single_adc.isChecked() and self.beta_fast_dac.isChecked()))

    def _redraw_t90_failures(self):
        from temporary_t90_waits import T90_SCHEMA, REASON_LABELS, t90_point_diagnostic_text
        result = (self.plan or {}).get('switch_90pct_timing', {}).get('t90_waits', {})
        if (result.get('schema') != T90_SCHEMA or not self._measurement_identity_matches(result)
                or self._dense_capture_active or self.manual_selection_active):
            self.t90_failed_markers.clear()
            return
        spots = []
        for point in result.get('points', []):
            if not point.get('reasons'):
                continue  # Valid low-voltage fixed waits are not failed measurements.
            order = point['target_order']
            row = self.plan['rows'][order]
            if point.get('target_index') != row['index']:
                continue
            selector = self.effective_feedback_selector()
            if (self.latest_values is not None and len(self.latest_values) == self.point_count()
                    and self.latest_channel == self.selected_channel()
                    and self.latest_feedback_selector == selector):
                value = self.latest_values[order]
            elif self.dense_values is not None and self.dense_feedback_selector == selector:
                value = self._dense_value_at(row['measured_nm'])
            else:
                value = point.get('target_adc')
            if value is None or not np.isfinite(value):
                continue
            group, position = divmod(order, self.points_per_peak())
            tooltip = (f'90%测量未通过：峰{group + 1} 第{position + 1}点，{row["measured_nm"]:.6f} nm\n'
                       + '、'.join(REASON_LABELS.get(reason, reason) for reason in point['reasons'])
                       + '\n' + t90_point_diagnostic_text(point))
            spots.append(dict(pos=(row['measured_nm'], float(self._plot_display_values(
                [value], selector=selector)[0])), data=dict(target_order=order, tooltip=tooltip)))
        self.t90_failed_markers.setData(spots)

    def _select_t90_failed_point(self, _item, points, _event):
        if points:
            self.select_selection_point(*divmod(points[0].data()['target_order'], self.points_per_peak()))

    def _dense_value_at(self, wavelength_nm):
        if self.dense_wavelengths is None or self.dense_values is None:
            return 0.0
        return float(np.interp(wavelength_nm, self.dense_wavelengths, self.dense_values))

    def _sync_manual_selection_controls(self):
        count = len(self.manual_selection_indices)
        self.manual_selection_label.setText(
            f'自动选点 {len(self.plan["rows"])}/{9 * self.points_per_peak()}'
            if self.plan is not None and self.plan.get('automatic_selection')
            else f'手动选点 {count}/{self.manual_point_count()}'
        )
        enabled = bool(self.manual_selection_active and self.worker is None)
        automatic = bool(self.worker is None and self.plan is not None
                         and self.plan.get('automatic_selection'))
        self.manual_undo_button.setEnabled((enabled and count > 0) or automatic)
        self.manual_clear_button.setEnabled((enabled and count > 0) or automatic)
        self.auto_select_button.setEnabled(bool(
            self.worker is None and self.selected_channel() == self.dense_channel
            and self._valid_dense_spectrum(self.dense_wavelengths, self.dense_values)))

    def auto_select_points(self, _checked=False):
        if self.worker is not None:
            self.status.setText('采集进行中，停止后才能自动选点。')
            return False
        if (not self._valid_dense_spectrum(self.dense_wavelengths, self.dense_values)
                or self.selected_channel() != self.dense_channel):
            self.status.setText('请先重采当前通道的密集光谱，再自动选点。')
            return False
        import app_JDSU as app
        try:
            plan = auto_select_dense_plan(
                app.load_fullband_accuracy_table(), self.dense_wavelengths,
                self.dense_values, feedback_selector=int(self.dense_feedback_selector),
                source=self.dense_source or 'auto dense-spectrum selection')
        except Exception as exc:
            self.status.setText(f'自动选点失败：{exc}；原有点位保留，可继续手动选点。')
            return False
        plan.update(
            signal_channel=self.selected_channel(), signal_channel_name=self.channel_name(),
            dense_acquisition_method=self.dense_acquisition_method,
            dense_point_count=int(len(self.dense_wavelengths)),
            dense_failed_points=list(self.dense_failed_points),
            dense_scan_range_nm=list(self.dense_scan_range_nm))
        self.plan = plan
        self.finger_record_name = None
        self.finger_record_label.setText('当前手指记录：未命名（自动选点，尚未保存）')
        self.point_waits = self.point_wait_route = None
        self.use_point_waits.setChecked(False)
        self.manual_selection_active = False
        self.manual_selection_indices = []
        self.manual_selection_history = []
        self._clear_selection_markers()
        self.show_dense_selection.setChecked(True)
        self.render_plan(ready=False)
        self._sync_manual_selection_controls()
        height_difference = max(value for check in plan['symmetry_checks']
                                for value in check['fitted_pair_difference_db'])
        boundary_gap = max(value for band in plan['selection_band_checks']
                           for value in band['boundary_height_gap_db'])
        self.status.setText(
            '自动选点完成：9峰×7点，dBm视觉上半段按高度均匀分层，'
            f'左右按高度配对（不要求波长等距），最大配对高度差{height_difference:.3f} dB；'
            f'最低点锚定半高线，最大高于半高线{boundary_gap:.3f} dB（真实采样量化）；可手动微调。'
            '请采集参考线后再开始测试；未通过点仍标出并继续采集。')
        self._save_last_route_session(ready=False)
        return True

    def _resume_automatic_selection_manually(self):
        if (self.worker is not None or self.plan is None
                or not self.plan.get('automatic_selection')):
            return False
        self.manual_points_per_peak = self.points_per_peak()
        self.manual_selection_indices = [int(row['index']) for row in self.plan['rows']]
        self.manual_selection_history = list(self.manual_selection_indices)
        self.manual_selection_active = True
        self.plan = None
        self.plan_ready = False
        self.start_button.setEnabled(False)
        self.reference_line_button.setEnabled(False)
        self.clear_reference_line()
        self._clear_selection_markers()
        return True

    def begin_manual_selection(self):
        """Start a fresh operator-only 45-point placement on the retained curve."""

        self.manual_points_per_peak = 5
        self.manual_selection_indices = []
        self.manual_selection_history = []
        self.manual_selection_active = True
        self.plan_ready = False
        self.start_button.setEnabled(False)
        self.reference_line_button.setEnabled(False)
        self.save_finger_button.setEnabled(True)
        self.show_dense_selection.setChecked(True)
        self._clear_selection_markers()
        self._sync_manual_selection_controls()
        self.redraw_dense_selection()
        if self.dense_wavelengths is not None and len(self.dense_wavelengths):
            left = float(self.dense_wavelengths[0])
            right = float(self.dense_wavelengths[-1])
            margin = max(.04, (right - left) * .02)
            self.plot.setXRange(left - margin, right + margin, padding=0)
            self.fit_values()

    def add_manual_selection_at(self, wavelength_nm):
        """Snap one spectrum click to the calibrated 0.02-nm route table."""

        if (not self.manual_selection_active or self.worker is not None
                or self.dense_wavelengths is None or self.dense_values is None):
            return False
        wavelength_nm = float(wavelength_nm)
        dense_x = np.asarray(self.dense_wavelengths, dtype=float)
        if not dense_x[0] - 1e-9 <= wavelength_nm <= dense_x[-1] + 1e-9:
            self.status.setText('请在当前密集光谱覆盖的波长范围内点击。')
            return False
        import app_JDSU as app
        table = tuple(app.load_fullband_accuracy_table())
        measured_axis = np.asarray([point.measured_nm for point in table], dtype=float)
        index = int(np.argmin(abs(measured_axis - wavelength_nm)))
        if not dense_x[0] - 1e-9 <= measured_axis[index] <= dense_x[-1] + 1e-9:
            self.status.setText('点击位置无法映射到当前扫描范围内的标定点。')
            return False
        if index in self.manual_selection_indices:
            self.status.setText(
                f'索引{index}（{measured_axis[index]:.6f} nm）已经放置，请点击其他位置。'
            )
            return False
        self.manual_selection_indices.append(index)
        self.manual_selection_history.append(index)
        self.manual_selection_indices.sort()
        self._sync_manual_selection_controls()
        self.redraw_dense_selection()
        count = len(self.manual_selection_indices)
        if count == self.manual_point_count():
            return self.finish_manual_selection()
        self.status.setText(
            f'已手动放置{count}/{self.manual_point_count()}点；可按任意顺序点击，'
            f'软件按波长排序，每连续{self.points_per_peak()}点作为一峰。'
        )
        return True

    def handle_dense_plot_click(self, event):
        """Accept left clicks close to the visible dense curve."""

        if (not self.manual_selection_active or self.worker is not None
                or not self.show_dense_selection.isChecked()
                or self.dense_wavelengths is None or self.dense_values is None
                or event.button() != QtCore.Qt.LeftButton):
            return
        view_box = self.plot.getViewBox()
        scene_position = event.scenePos()
        if not view_box.sceneBoundingRect().contains(scene_position):
            return
        view_position = view_box.mapSceneToView(scene_position)
        wavelength_nm = float(view_position.x())
        raw_value = self._dense_value_at(wavelength_nm)
        display_value = float(self._plot_display_values(
            [raw_value], selector=self.dense_feedback_selector
        )[0])
        curve_position = view_box.mapViewToScene(
            QtCore.QPointF(wavelength_nm, display_value)
        )
        if abs(float(curve_position.y()) - float(scene_position.y())) > 24.0:
            self.status.setText('请靠近密集光谱曲线点击；点位会吸附到0.02 nm标定网格。')
            return
        if self.add_manual_selection_at(wavelength_nm):
            event.accept()

    def undo_manual_selection(self, _checked=False):
        if not self.manual_selection_active:
            self._resume_automatic_selection_manually()
        if not self.manual_selection_active or not self.manual_selection_indices:
            return False
        index = self.manual_selection_history.pop()
        self.manual_selection_indices.remove(index)
        self._sync_manual_selection_controls()
        self.redraw_dense_selection()
        self.status.setText(
            f'已撤销，当前手动放置{len(self.manual_selection_indices)}/{self.manual_point_count()}点。'
        )
        return True

    def clear_manual_selection(self, _checked=False):
        if not self.manual_selection_active:
            self._resume_automatic_selection_manually()
        if not self.manual_selection_active:
            return False
        self.manual_selection_indices = []
        self.manual_selection_history = []
        self._sync_manual_selection_controls()
        self.redraw_dense_selection()
        self.status.setText(f'已清空手动点；请在密集光谱线上重新点击{self.manual_point_count()}个点。')
        return True

    def finish_manual_selection(self):
        if len(self.manual_selection_indices) != self.manual_point_count():
            return False
        import app_JDSU as app
        groups = [
            self.manual_selection_indices[start:start + self.points_per_peak()]
            for start in range(0, self.manual_point_count(), self.points_per_peak())
        ]
        try:
            plan = build_manual_dense_plan(
                app.load_fullband_accuracy_table(), groups,
                self.dense_wavelengths, self.dense_values,
                source=self.dense_source or 'manual dense-spectrum selection',
            )
        except Exception as exc:
            self.status.setText(f'45点手动路线未生成：{exc}；可撤销后重新选择。')
            return False
        plan['signal_channel'] = self.selected_channel()
        plan['signal_channel_name'] = self.channel_name()
        plan['dense_acquisition_method'] = self.dense_acquisition_method
        plan['dense_point_count'] = int(len(self.dense_wavelengths))
        plan['dense_failed_points'] = list(self.dense_failed_points)
        plan['dense_scan_range_nm'] = list(self.dense_scan_range_nm)
        plan['manual_selection_method'] = f'operator_plot_click_{self.manual_point_count()}_sorted_into_9x{self.manual_points_per_peak}'
        self.plan = plan
        self.finger_record_name = None
        self.finger_record_label.setText(
            '当前手指记录：未命名（手动45点，尚未保存）'
        )
        self.point_waits = None
        self.point_wait_route = None
        self.use_point_waits.setChecked(False)
        self.manual_selection_active = False
        self.manual_selection_indices = []
        self.manual_selection_history = []
        self._sync_manual_selection_controls()
        self.render_plan(ready=False)
        warning_groups = [
            str(check['group']) for check in plan.get('selection_fit_checks', [])
            if not check.get('passed')
        ]
        warning = (
            '；密集谱预览中峰' + '/'.join(warning_groups)
            + '形状可能有问题，但点位已保留'
            if warning_groups else ''
        )
        self.status.setText(
            f'手动{self.point_count()}点已完成：已按波长排序并每{self.points_per_peak()}点分为一峰'
            + warning
            + f'。下一步可采集{self.point_count()}点参考线：每点等待10 ms后记录单次ADC，不做稳定性/峰形筛选。'
        )
        self._save_last_route_session(ready=False)
        return True

    def _redraw_reference_failures(self):
        rows = (self._live_reference_line_rows
                if self._reference_line_capture_active
                else self.reference_failed_points)
        if not rows or not self.show_reference.isChecked():
            self.reference_failed_curve.clear()
            return
        selector = (self.effective_feedback_selector()
                    if self._reference_line_capture_active
                    else self.reference_feedback_selector)
        x_values, y_values = [], []
        for item in rows:
            row = item['row'] if 'row' in item else item
            if row.get('stability_not_evaluated'):
                continue
            passed = bool(row.get('point_passed', row.get('stable', False)))
            saturated = bool(row.get('signal_saturated'))
            if passed and not saturated:
                continue
            order = int(item.get('order', len(x_values)))
            if self.plan is None or not 0 <= order < self.point_count():
                continue
            x_values.append(float(self.plan['rows'][order]['measured_nm']))
            y_values.append(float(row.get(
                'signal_adc_code', row.get('ch1_adc_code', np.nan)
            )))
        finite = np.isfinite(np.asarray(y_values, dtype=float))
        if not x_values or not np.any(finite):
            self.reference_failed_curve.clear()
            return
        x_array = np.asarray(x_values, dtype=float)[finite]
        y_array = np.asarray(y_values, dtype=float)[finite]
        self.reference_failed_curve.setData(
            x_array,
            self._plot_display_values(
                y_array, reference=True, selector=selector
            ),
        )

    def redraw_dense_selection(self, _checked=None):
        self._redraw_t90_failures()
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
            self.manual_selection_curve.clear()
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
            self.manual_selection_curve.clear()
            self._clear_selection_markers()
            return
        if self.manual_selection_active:
            self._clear_selection_markers()
            if self.manual_selection_indices:
                import app_JDSU as app
                table = tuple(app.load_fullband_accuracy_table())
                selected_x = np.asarray([
                    table[index].measured_nm
                    for index in self.manual_selection_indices
                ], dtype=float)
                selected_y = np.asarray([
                    self._dense_value_at(wavelength_nm)
                    for wavelength_nm in selected_x
                ], dtype=float)
                brushes = [
                    pg.mkBrush(pg.intColor(min(i // self.points_per_peak(), 8), 9))
                    for i in range(len(selected_x))
                ]
                self.manual_selection_curve.setData(
                    selected_x,
                    self._plot_display_values(
                        selected_y, selector=self.dense_feedback_selector
                    ),
                    symbolBrush=brushes,
                )
            else:
                self.manual_selection_curve.clear()
            return
        self.manual_selection_curve.clear()
        if self.plan is None:
            self._clear_selection_markers()
            return
        point_count = self.point_count()
        if len(self.selection_markers) != point_count:
            self._clear_selection_markers()
            for flat_index in range(point_count):
                group, point = divmod(flat_index, self.points_per_peak())
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
        """Retain every point; coalesce expensive curve redraws on the GUI thread."""
        if not self._dense_capture_active:
            return
        row = dict(payload.get('row', {}))
        self._live_dense_rows.append(row)
        self.show_dense_selection.setChecked(True)
        completed = int(payload.get('completed', len(self._live_dense_rows)))
        total = int(payload.get('total', 2001))
        self._dense_draw_progress = (completed, total)
        if len(self._live_dense_rows) == 1 or completed >= total:
            # First and final points are immediate. Intermediate points are
            # never dropped: the next redraw includes the complete raw list.
            self._flush_dense_reference_plot()
        elif not self._dense_draw_timer.isActive():
            self._dense_draw_timer.start()

    def _reset_dense_draw_state(self):
        self._dense_draw_timer.stop()
        self._dense_draw_progress = None
        self._dense_last_range_count = 0

    def _flush_dense_reference_plot(self):
        self._dense_draw_timer.stop()
        if not self._dense_capture_active or self._dense_draw_progress is None:
            return
        completed, total = self._dense_draw_progress
        self.redraw_dense_selection()
        # dBm values are negative.  Before the first live samples the plot may
        # still have an ADC/empty range such as 0..4, which clips the whole
        # trace against the lower border and makes a real spectrum look flat.
        # Rescale at a bounded cadence so live drawing stays responsive.
        range_refresh_interval = max(10, total // 100)
        if (completed == 1 or completed >= total
                or completed // range_refresh_interval
                > self._dense_last_range_count // range_refresh_interval):
            self.fit_values()
            self._dense_last_range_count = completed
        failed = sum(not item.get('point_passed') for item in self._live_dense_rows)
        suffix = f'；已标红{failed}个未通过点' if failed and self.show_dense_failures.isChecked() else ''
        self.status.setText(
            f'{self.channel_name()}密集光谱实时绘制 {completed}/{total}{suffix}。'
            '若范围已经足够，可点“停止绘制”，随后手动放置45点。'
        )

    @QtCore.pyqtSlot(object)
    def update_reference_line_point(self, payload):
        """Display fixed-dwell reference progress without a stability verdict."""

        if not self._reference_line_capture_active:
            return
        row = dict(payload.get('row', {}))
        order = max(0, int(payload.get('completed', 1)) - 1)
        self._live_reference_line_rows.append({'order': order, 'row': row})
        self._redraw_reference_failures()
        completed = int(payload.get('completed', len(self._live_reference_line_rows)))
        total = int(payload.get('total', self.point_count()))
        self.status.setText(
            f'采集参考线 {completed}/{total}；10 ms/点，单次ADC，不做稳定性筛选。'
        )

    def stop_dense_reference(self):
        if self.worker is None or not self._dense_capture_active:
            return
        self.worker.stop()
        self.stop_dense_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.status.setText(
            '正在停止密集绘制并安全关光；已采部分会保留，停光后可手动放置45点…'
        )

    def _refresh_selection_marker_styles(self):
        for flat_index, marker in enumerate(self.selection_markers):
            selected = self.selected_selection_point == divmod(flat_index, self.points_per_peak())
            color = marker.selection_color
            marker.setPen(pg.mkPen((255, 255, 255) if selected else color,
                                   width=3 if selected else 2))
            marker.setBrush(pg.mkBrush(color if selected else (0, 0, 0, 30)))
            marker.update()

    def select_selection_point(self, group, point):
        if (self.plan is None or not 0 <= group < self.peak_count() or not 0 <= point < self.points_per_peak()
                or len(self.selection_markers) != self.point_count()):
            return False
        self.selected_selection_point = (group, point)
        self._refresh_selection_marker_styles()
        row = self.plan['rows'][group * self.points_per_peak() + point]
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
        flat_index = group * self.points_per_peak() + point
        indices = [int(row['index']) for row in self.plan['rows']]
        import app_JDSU as app
        table = tuple(app.load_fullband_accuracy_table())
        measured_axis = np.asarray([entry.measured_nm for entry in table], dtype=float)
        dense_left = int(np.searchsorted(
            measured_axis, float(self.dense_wavelengths[0]), side='left'
        )) if self.dense_wavelengths is not None else 0
        dense_right = int(np.searchsorted(
            measured_axis, float(self.dense_wavelengths[-1]), side='right'
        ) - 1) if self.dense_wavelengths is not None else len(table) - 1
        lower = indices[flat_index - 1] + 1 if flat_index else dense_left
        upper = (indices[flat_index + 1] - 1
                 if flat_index < self.point_count() - 1 else dense_right)
        lower = max(lower, dense_left)
        upper = min(upper, dense_right)
        requested = indices[flat_index] + step
        if not lower <= requested <= upper:
            self.status.setText(
                f'峰{group + 1}第{point + 1}点不能继续向'
                + ('左' if step < 0 else '右')
                + '移动：相邻采样点之间必须保留严格顺序。'
            )
            return False
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
                or not 0 <= point < self.points_per_peak()):
            return False
        import app_JDSU as app
        table = tuple(app.load_fullband_accuracy_table())
        measured = np.asarray([entry.measured_nm for entry in table], dtype=float)
        requested = int(np.argmin(abs(measured - requested_wavelength_nm)))
        groups = [
            [int(row['index']) for row in self.plan['rows'][start:start + self.points_per_peak()]]
            for start in range(0, self.point_count(), self.points_per_peak())
        ]
        flat = [index for indices in groups for index in indices]
        flat_index = group * self.points_per_peak() + point
        dense_left = int(np.searchsorted(
            measured, float(self.dense_wavelengths[0]), side='left'
        ))
        dense_right = int(np.searchsorted(
            measured, float(self.dense_wavelengths[-1]), side='right'
        ) - 1)
        lower = flat[flat_index - 1] + 1 if flat_index else dense_left
        upper = (flat[flat_index + 1] - 1
                 if flat_index < self.point_count() - 1 else dense_right)
        lower = max(lower, dense_left)
        upper = min(upper, dense_right)
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
        selected = self.plan['rows'][group * self.points_per_peak():group * self.points_per_peak() + self.points_per_peak()]
        self.status.setText(
            f'已手动修改峰{group + 1}第{point + 1}点：'
            + ', '.join(f"{row['target_nm']:.2f}" for row in selected)
            + f' nm；{preview}。修改前参考虚线已保留，未自动重采；'
            + ('可直接开始手指应力测试。只有主动点击“采集并绘制参考线”才会更新参考线。'
               if was_ready else
               '当前路线原本尚不可开始；移动点位不会自动取得开始权限。')
        )
        self._save_last_route_session(ready=was_ready)
        return True

    def fit_peak_view(self, _index=None):
        if self.plan is None:
            return
        index = self.peak_view.currentIndex()
        rows = self.plan['rows'] if index == 0 else self.plan['rows'][(index-1)* self.points_per_peak():index* self.points_per_peak()]
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
                    f'{self.point_count()}点10 ms单次ADC参考线。'
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
        self._auto_select_after_dense = False
        self._reset_dense_draw_state()
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
        usable_rows = sorted([
            row for row in rows
            if np.isfinite(row.get('measured_wavelength_nm', np.nan))
            and np.isfinite(row.get(
                'signal_adc_code', row.get('ch1_adc_code', np.nan)
            ))
        ], key=lambda row: float(row['measured_wavelength_nm']))
        # A partial report can contain a duplicated retry row.  Retain the last
        # finite observation at each wavelength so the plotted axis is strict.
        usable_rows = list({
            float(row['measured_wavelength_nm']): row for row in usable_rows
        }.values())
        # Always retain the raw spectrum, even if automatic selection fails.
        # Defer selection until finished() releases the acquisition worker.
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
        try:
            if len(usable_rows) < self.manual_point_count():
                raise ValueError(
                    f'已采有效点只有{len(usable_rows)}个，手动放置45点至少需要45个谱线点'
                )
            self.dense_feedback_selector = int(
                report.get('feedback_selector', self.effective_feedback_selector())
            )
            self.dense_scan_range_nm = (
                float(self.dense_wavelengths[0]),
                float(self.dense_wavelengths[-1]),
            )
            self.begin_manual_selection()
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
            duration = (
                f'，扫描耗时{float(report["acquisition_elapsed_s"]):.3f} s'
                + ('（板端批量）' if report.get('acquisition_backend') == 'stm32_dense_batch_v1' else '')
                if 'acquisition_elapsed_s' in report else ''
            )
            self._auto_select_after_dense = bool(
                report.get('complete') and not report.get('stopped_early')
                and len(rows) == total and len(usable_rows) == total
            )
            selection_hint = (
                '采集已完成，正在准备自动选点…'
                if self._auto_select_after_dense else
                '可直接在谱线上手动点击45个点；未完整采集时不自动替换点位。'
            )
            self.status.setText(
                f'{self.channel_name(signal_channel)}密集光谱·{method}{completion}{duration}{quality}；'
                f'扫描范围{self.dense_scan_range_nm[0]:.3f}～'
                f'{self.dense_scan_range_nm[1]:.3f} nm。{selection_hint}'
            )
        except Exception as exc:
            self.redraw_dense_selection()
            previous = '；已保留上一张可用临时点表' if self.plan is not None else ''
            self.status.setText(
                f'已绘制{len(rows)}个{self.channel_name(signal_channel)}密集点'
                '，'
                f'但当前范围还不能手动放置完整45点：{exc}{previous}。'
            )

    def clear_reference_line(self, _value=None):
        if self.plan is None:
            self.clear_peak_fits('等待有效点表及实时帧')
        self.reference_values = None
        self.reference_wavelengths = None
        self.reference_failed_points = []
        self._live_reference_line_rows = []
        self._reference_line_capture_active = False
        self.reference_failed_curve.clear()
        for curve in self.reference_curves:
            curve.clear()
        for i in range(self.point_count()):
            self.table.setItem(i, 5, QtWidgets.QTableWidgetItem('—'))

    def invalidate_shape_verification(self, _value=None):
        self._redraw_t90_failures()
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
        if any(len(group) != self.points_per_peak() for group in self.validated_route_indices):
            return False  # Never replace a seven-point peak with an old five-point route.
        groups = [
            [int(row['index']) for row in self.plan['rows'][start:start + self.points_per_peak()]]
            for start in range(0, self.point_count(), self.points_per_peak())
        ]
        values = list(live_values)
        replaced = []
        for group in failed_groups:
            offset = (int(group) - 1) * self.points_per_peak()
            fallback = list(self.validated_route_indices[int(group) - 1])
            if groups[int(group) - 1] == fallback:
                continue
            groups[int(group) - 1] = fallback
            values[offset:offset + self.points_per_peak()] = self.validated_route_values[offset:offset + self.points_per_peak()]
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
        self._reference_line_capture_active = False
        self._live_reference_line_rows = []
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
            raw_reference = bool(report.get('acquisition_method') == 'equal_interval_single'
                                 and report.get('stability_filter_applied') is False)
            if (not report.get('complete') or not report.get('disarm_ack') or not report.get('shutter_ack')
                    or self.plan is None or len(rows) != point_count
                    or signal_channel != self.selected_channel()
                    or feedback_selector != self.effective_feedback_selector(signal_channel)):
                raise ValueError('参考未完整完成、停光未确认或模拟档位不一致')
            quality_issues = []
            new_failed_points = []
            for order, (row, selected) in enumerate(zip(rows, self.plan['rows'])):
                if (row['index'] != selected['index'] or row['dac_codes'] != selected['codes']
                        or not np.isfinite(row.get(
                            'signal_adc_code', row.get('ch1_adc_code', np.nan)
                        ))
                        or not 0 <= row.get(
                            'signal_adc_code', row.get('ch1_adc_code', np.nan)
                        ) <= 4095):
                    raise ValueError('参考点不匹配或没有有效ADC读数')
                if raw_reference:
                    continue
                point_passed = bool(row.get(
                    'point_passed', row.get('stable', False)
                ))
                saturated = bool(
                    row.get('signal_saturated', row.get('ch1_saturated', False))
                    or row.get('signal_adc_code', row.get('ch1_adc_code', 0)) >= 4080
                )
                if not point_passed:
                    quality_issues.append(f"点{row['index']}稳定性未通过")
                if saturated:
                    quality_issues.append(f"点{row['index']}可能饱和")
                if not point_passed or saturated:
                    new_failed_points.append({
                        'order': int(order),
                        'row': {
                            'index': int(row['index']),
                            'point_passed': bool(point_passed),
                            'stable': bool(row.get('stable', False)),
                            'signal_saturated': bool(saturated),
                            'signal_adc_code': float(row.get(
                                'signal_adc_code', row.get('ch1_adc_code')
                            )),
                            'failure_reason': str(
                                row.get('failure_reason')
                                or '稳定性或饱和检查未通过'
                            ),
                        },
                    })
            self.reference_values = [
                row.get('signal_adc_code', row.get('ch1_adc_code')) for row in rows
            ]
            self._spectrum_dirty = True
            self.reference_feedback_selector = feedback_selector
            self.reference_channel = signal_channel
            self.reference_failed_points = new_failed_points
            self.reference_wavelengths = [
                float(row['measured_nm']) for row in self.plan['rows']
            ]
            # This path only runs after the operator explicitly pressed the
            # reference-line acquisition button.  The new 45-point reference
            # now supersedes any pre-adjustment line that was being reused.
            self.plan.pop('manual_adjustment_reference_reused', None)
            self.plan.pop('manual_adjustment_reference_note', None)
            for i, value in enumerate(self.reference_values):
                item = QtWidgets.QTableWidgetItem(f'{value:g}')
                failed_item = next((
                    failed for failed in new_failed_points
                    if failed['order'] == i
                ), None)
                if failed_item is not None:
                    reason = failed_item['row'].get(
                        'failure_reason', '稳定性或饱和检查未通过'
                    )
                    item.setForeground(QtGui.QBrush(QtGui.QColor('#dc2626')))
                    item.setBackground(QtGui.QBrush(QtGui.QColor('#fee2e2')))
                    item.setToolTip(f'未通过但已继续采集：{reason}')
                self.table.setItem(i, 5, item)
            self.show_reference.setChecked(True)
            self.redraw_values()
            self.fit_values()
            if raw_reference:
                self.plan.pop('live_reference_fit_checks', None)
                self.plan.pop('reference_shape_gate_bypassed', None)
                self.plan.pop('reference_line_quality_issues', None)
                self.plan['reference_acquisition_method'] = 'equal_interval_single'
                self.plan['reference_stability_evaluated'] = False
                self.plan_ready = True
                self.start_button.setEnabled(True)
                elapsed = report.get('acquisition_elapsed_s')
                duration = f'，采集耗时{float(elapsed):.3f} s' if elapsed is not None else ''
                self.status.setText(
                    f'{point_count}点参考线已采集（10 ms/点，单次ADC{duration}）；'
                    '未做稳定性/峰形验收，已停光，可开始手指应力测试。'
                )
                self._save_last_route_session(ready=True)
                return
            self.plan.pop('reference_acquisition_method', None)
            self.plan.pop('reference_stability_evaluated', None)
            checks = []
            for group in range(peak_count):
                start = group * self.points_per_peak()
                selected = self.plan['rows'][start:start + self.points_per_peak()]
                check = assess_peak_shape(
                    [row['measured_nm'] for row in selected],
                    self.reference_values[start:start + self.points_per_peak()],
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
                    + '；按设置仍可直接开始手指应力测试。'
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
            self._redraw_reference_failures()
            self.plan_ready = bool(
                self._reference_line_prior_ready
                or (isinstance(self.reference_values, list)
                    and len(self.reference_values) == self.point_count())
            )
            self.start_button.setEnabled(self.plan_ready)
            if self.plan_ready:
                self.status.setText(
                    f'本次{self.point_count()}点参考采集未采用：{exc}；已保留原有路线，仍可直接开始手指应力测试。'
                    '如需更新参考线，请再次点击采集按钮。'
                )
                self._save_last_route_session(ready=True)
            else:
                if self.reference_values is None:
                    self.clear_reference_line()
                self.status.setText(f'参考线未采用：{exc}')
                self._save_last_route_session(ready=False)

    def _current_wait_snapshot(self):
        if self.use_point_waits.isChecked():
            if not self._ensure_point_waits_for_route():
                raise ValueError('当前没有有效逐点等待表')
            return list(self.point_waits), 'point_waits'
        return self._uniform_point_waits(), 'uniform_and_cross_peak'

    def start_current_wait_levels(self, _checked=False):
        import app_JDSU as app
        if not self.beta or self.worker is not None or self.plan is None:
            return
        if self.demo:
            self.status.setText('演示模式不会测量真实电压到达比例')
            return
        try:
            from fbg_lan_serial import FbgLanSerial
            if not app.switch_mode_enable or app.ser_open:
                raise ValueError('请先停止其他采集')
            if not temporary_transport_supported(app.ser) or isinstance(app.ser, FbgLanSerial):
                raise ValueError('当前等待到达比例测量目前需要USB本地连接')
            if int(self.plan.get('signal_channel', 1)) != self.selected_channel():
                raise ValueError('路线属于其他ADC通道，请重新选点')
            if not self.beta_single_adc.isChecked():
                raise ValueError('当前等待到达比例测量使用单次ADC')
            waits, mode = self._current_wait_snapshot()
            fast = self.beta_fast_dac.isChecked()
            command(self.plan['rows'], 32, 1, point_delays_us=waits,
                    adc_channel=self.selected_channel(), feedback_selector=self.effective_feedback_selector(),
                    single_sample=True, fast_dac_writes=fast)
            app.ensure_serial_open()
        except Exception as exc:
            self.status.setText(f'当前等待测量启动前检查失败：{exc}')
            return
        plan = json.loads(json.dumps(self.plan))
        plan['current_wait_measurement_mode'] = mode
        output = output_path('temporary_test', f'current_wait_levels_{time.time_ns()}.json')
        self.worker = CurrentWaitLevelWorker(
            app.ser, plan, output, 32, self, point_delays_us=waits,
            adc_channel=self.selected_channel(), feedback_selector=self.effective_feedback_selector(),
            single_sample=True, fast_dac_writes=fast)
        self.worker.measured.connect(self.current_wait_levels_ready)
        self.worker.message.connect(self.status.setText)
        self.worker.finished.connect(self.finished)
        app.switch_mode_enable = False
        for widget in (self.channel_combo, self.start_button, self.reference_line_button,
                       self.equal_interval_reference_button, self.save_finger_button, self.load_finger_button,
                       self.scan_start_nm, self.scan_stop_nm, self.equal_interval_nm, self.equal_interval_settle,
                       self.analog_gain, self.first_delay, self.boundary_delay, self.test_switch_timing):
            widget.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.stop_dense_button.setEnabled(False)
        self.refresh_wait_controls()
        self.status.setText('正在测量当前等待下的实际电压到达比例；请保持光纤和传感器静止…')
        self.worker.start()

    @QtCore.pyqtSlot(object)
    def current_wait_levels_ready(self, report):
        try:
            from temporary_current_wait_levels import analyze_current_wait_levels
            result = analyze_current_wait_levels(report)
            waits, mode = self._current_wait_snapshot()
            if (not self._measurement_identity_matches(result) or result['point_delays_us'] != waits
                    or result['wait_mode'] != mode or not self.beta_single_adc.isChecked()):
                raise ValueError('测量机号、路线、通道、档位、等待或写入方式与当前不同')
            result['source'] = str(self.worker.output)
            from temporary_test_mode import quality
            display_frame = dict(report['scan_capture']['frames'][-1])
            display_frame['quality'] = quality(display_frame)
            self.frame_times = []  # Do not mix timestamps from separate scans.
            self.update_frame(display_frame)
            self.plan['current_wait_levels'] = result
            usable = [p['percentage_of_final'] for p in result['points'] if not p['reasons']]
            detail = f'有效比例 {min(usable):.1f}%～{max(usable):.1f}%' if usable else '没有可靠比例'
            uncertain = sum(bool(p['reasons']) for p in result['points'])
            mode_label = '逐点等待表' if mode == 'point_waits' else '峰内/跨峰统一等待（未勾选逐点等待）'
            self.status.setText(f'当前等待到达比例测量完成：{mode_label}，{detail}；'
                                f'{uncertain}点参考不可靠或比例不可用。可查看各点电压及具体偏差；等待参数未改动。')
            self.refresh_wait_controls()
            self._save_last_route_session(ready=self.plan_ready)
        except Exception as exc:
            self.status.setText(f'当前等待到达比例结果未采用：{exc}；等待参数未改动')

    def show_current_wait_levels(self, _checked=False):
        from temporary_current_wait_levels import REASON_LABELS
        result = (self.plan or {}).get('current_wait_levels', {})
        if not self._measurement_identity_matches(result) or not result.get('points'):
            self.status.setText('当前路线、机号、通道或档位没有匹配的到达比例，请重新测量')
            return
        waits, mode = self._current_wait_snapshot()
        if waits != result['point_delays_us'] or mode != result['wait_mode']:
            self.status.setText('等待设置已改变，请重新测量当前等待到达比例；旧结果仍保留')
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('当前等待下各点到达最终电压的比例')
        layout = QtWidgets.QVBoxLayout(dialog)
        note = QtWidgets.QLabel(
            '比例=当前电压/最终电压×100%，基准为0；超过100%表示过冲，不截断。'
            '使用32帧中预热后的最后一帧，包含首点回绕；最终电压是各点保持后的末3次读数中位数。\n'
            '采样完成时间包含ADC读取开销，不冒充采样开始时间。单次比例不是90%稳定时间或波长精度认证。'
            '测量时传感器须静止，低电压点也显示实际比例，不自动应用300 µs。')
        note.setWordWrap(True)
        layout.addWidget(note)
        table = QtWidgets.QTableWidget(len(result['points']), 7)
        table.setHorizontalHeaderLabels(['波长 nm', '当前等待 µs', '采样完成 µs', '当前电压 V',
                                        '最终电压 V', '到达比例', '偏差/状态'])
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        for i, point in enumerate(result['points']):
            percentage = point['percentage_of_final']
            state = ('、'.join(REASON_LABELS.get(r, r) for r in point['reasons']) if point['reasons'] else
                     '本次读数在±10%内（不代表持续稳定）' if point['within_10pct'] else '本次读数超出±10%')
            deviation = '' if percentage is None else f'{point["deviation_percent"]:+.2f}%（相对最终值）\n'
            texts = (f'{point["wavelength_nm"]:.6f}', str(point['wait_us']),
                     str(point['sample_end_after_target_write_us']), f'{point["sample_voltage_v"]:.6f}',
                     f'{point["target_voltage_v"]:.6f}', '不可计算' if percentage is None else f'{percentage:.2f}%',
                     deviation + state)
            for col, text in enumerate(texts):
                item = QtWidgets.QTableWidgetItem(text)
                item.setToolTip(text)
                item.setTextAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
                if point['reasons'] or (percentage is not None and not point['within_10pct']):
                    item.setBackground(QtGui.QColor('#fff3cd'))
                table.setItem(i, col, item)
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(6, QtWidgets.QHeaderView.Stretch)
        def fit_result_rows(*_):
            try:
                metrics = QtGui.QFontMetrics(table.font())
            except RuntimeError:
                return  # A queued resize can outlive a closed dialog.
            width = max(40, table.columnWidth(6) - 16)
            for row in range(table.rowCount()):
                bounds = metrics.boundingRect(QtCore.QRect(0, 0, width, 100000),
                                              QtCore.Qt.TextWordWrap, table.item(row, 6).text())
                table.setRowHeight(row, max(metrics.lineSpacing() + 16, bounds.height() + 16))
        table.horizontalHeader().sectionResized.connect(
            lambda *_: QtCore.QTimer.singleShot(0, fit_result_rows))
        QtCore.QTimer.singleShot(0, fit_result_rows)
        layout.addWidget(table)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.resize(1180, 720)
        dialog.exec_()

    def start_switch_timing(self, _checked=False):
        """Run the profile's board timing; voltage-gated single passes in Beta."""
        import app_JDSU as app
        if self.selected_channel() != 1 and not self.beta:
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
        if self.beta:
            from fbg_lan_serial import FbgLanSerial
            if isinstance(app.ser, FbgLanSerial):
                self.status.setText('Beta的90%诊断目前仅支持USB，请切换到USB本地；LAN采集功能仍保留。')
                return
        try:
            if int(self.plan.get('signal_channel', 1)) != self.selected_channel():
                raise ValueError('当前路线属于其他ADC通道，请切回该路线通道或重新选点')
            command(
                self.plan['rows'], 32, 1,
                feedback_selector=self.effective_feedback_selector(),
                adc_channel=self.selected_channel(),
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
            feedback_selector=self.effective_feedback_selector(),
            adc_channel=self.selected_channel(),
            single_sample=self.beta and self.beta_single_adc.isChecked(),
            fast_dac_writes=(self.beta and self.beta_single_adc.isChecked()
                             and self.beta_fast_dac.isChecked()),
            point_delays_us=(list(self.point_waits) if self.beta and self.use_point_waits.isChecked()
                             and self._ensure_point_waits_for_route() else
                             self._uniform_point_waits() if self.beta else None),
        )
        self.worker.timing.connect(self.switch_timing_ready)
        self.worker.message.connect(self.status.setText)
        self.worker.finished.connect(self.finished)
        for widget in (
                self.equal_interval_reference_button,
                self.scan_start_nm, self.scan_stop_nm,
                self.equal_interval_nm, self.equal_interval_settle,
                self.save_finger_button,
                self.load_finger_button, self.spacing, self.cycles,
                self.start_button, self.analog_gain, self.first_delay,
                self.boundary_delay, self.reference_line_button,
                self.test_switch_timing):
            widget.setEnabled(False)
        self.continuous.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.stop_dense_button.setEnabled(False)
        self.status.setText(
            f'正在对当前{self.point_count()}点做{1 if self.beta else 3}轮{self.channel_name()} 90%时间测试'
            + ('（含末点回到首点）…' if self.beta else '（正向）…')
        )
        self.worker.start()
        self.refresh_wait_controls()

    @QtCore.pyqtSlot(object)
    def switch_timing_ready(self, report):
        try:
            if self.beta and (report.get('reverse_wrap_tested') or report.get('t90_waits')):
                self._accept_beta_t90_result(report)
                return
            if not report.get('complete') or self.plan is None:
                raise ValueError('测试未完整完成')
            if self.beta:
                if not (report.get('exact_disarm_ack') and report.get('exact_soa_shutter_ack')):
                    raise ValueError('停采和关光确认不完整')
                route_key = lambda rows: [(r['index'], tuple(r['codes'])) for r in rows]
                if route_key(report.get('route', [])) != route_key(self.plan['rows']):
                    raise ValueError('测量路线已改变；结果未应用')
                if report.get('signal_channel', 1) != self.selected_channel():
                    raise ValueError('测量通道已改变；结果未应用')
            transitions = report.get('analysis', {}).get('transitions', [])
            if len(transitions) != self.point_count() - 1:
                raise ValueError('正向跳转记录数与当前路线不匹配')
            current = (
                list(self.point_waits)
                if isinstance(self.point_waits, list)
                and len(self.point_waits) == self.point_count()
                else self._uniform_point_waits() if self.beta else [650] * self.point_count()
            )
            measurements = [None] * self.point_count()
            late, missing = [], []
            for position, transition in enumerate(transitions):
                target = int(transition['route_target_order'])
                if self.beta and (target != position + 1
                        or int(transition['route_source_order']) != position
                        or int(transition['target_fullband_index']) != self.plan['rows'][target]['index']):
                    raise ValueError('跳转顺序或目标物理索引不匹配')
                signal_name = report.get('signal_channel_name', 'CH1')
                metric = transition['channels'][signal_name]
                t90 = metric.get('t90_us')
                if t90 is not None and (not np.isfinite(float(t90)) or float(t90) < 0):
                    raise ValueError('90%时间不是有效的非负数值')
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
                if not self.beta:
                    current[target] = wait_us
                measurements[target] = {
                    'source_order': int(transition['route_source_order']),
                    'target_order': target,
                    'target_index': int(transition['target_fullband_index']),
                    'meaningful_step': meaningful,
                    't90_after_target_write_us': t90,
                    'dac_write_us': transition.get('path_update_us'),
                    't90_from_switch_begin_us': metric.get('t90_from_switch_begin_us'),
                    'displayed_wait_us': wait_us,
                    'classification': metric.get('classification'),
                }
                if t90 is not None and float(t90) > 3000.0:
                    late.append(target + 1)
            self.point_waits = current
            self.point_wait_route = self.point_route_key()
            if not self.beta:
                self.use_point_waits.setChecked(True)
            else:
                self.use_point_waits.setChecked(False)
                self.beta_timing_results.setEnabled(True)
            self.plan['switch_90pct_timing'] = {
                'source': str(self.worker.output),
                'repeats': int(report.get('repeats', 3)),
                'direction': report.get('direction'),
                'reverse_wrap_tested': False,
                'first_point_unmeasured': True,
                'fast_dac_writes': report.get('fast_dac_writes', False),
                'feedback_selector': report.get('feedback_selector', report.get(
                    'ch1_feedback_selector', self.analog_gain.currentData())),
                'signal_channel': report.get('signal_channel', 1),
                'signal_channel_name': report.get('signal_channel_name', 'CH1'),
                'route_key': [[row['index'], list(row['codes'])] for row in self.plan['rows']],
                'points': measurements,
                'wait_balance': report.get('wait_balance', {}),
            }
            measured = sum(item is not None for item in measurements)
            detail = []
            if late:
                detail.append('>3 ms点：' + '/'.join(map(str, late)))
            if missing:
                detail.append('未解出90%点：' + '/'.join(map(str, missing)))
            suffix = '；' + '；'.join(detail) if detail else '；所有可解出跳转均不超过3 ms'
            self.status.setText(
                f'开关速度测试完成：已记录{measured}个正向跳转的{signal_name} 90%到达时间'
                f'（第1点不做反向回绕测试，保留原值）{suffix}。'
                + ('90%结果未缩短等待；可查看结果及稳态候选，再做整条路线对照。'
                   if self.beta else '已写入逐点等待。')
            )
            self.refresh_wait_controls()
            self._save_last_route_session(ready=self.plan_ready)
        except Exception as exc:
            self.status.setText(f'开关速度结果未采用：{exc}')

    def _accept_beta_t90_result(self, report):
        from temporary_t90_waits import analyze_t90_waits, REASON_LABELS
        # Recompute from raw individual repeats, never trust an imported summary.
        result = analyze_t90_waits(report)
        key = [[r['index'], list(r['codes'])] for r in (self.plan or {}).get('rows', [])]
        fast = self.beta_single_adc.isChecked() and self.beta_fast_dac.isChecked()
        if (result['route_key'] != key or result['signal_channel'] != self.selected_channel()
                or result['feedback_selector'] != self.effective_feedback_selector()
                or result['fast_dac_writes'] != fast
                or result['machine_id'] != get_runtime_machine_id()):
            raise ValueError('测量机号、路线、通道、档位或DAC方式与当前不同')
        timing = dict(result)
        timing.update(source=str(self.worker.output), t90_waits=result, applied=False)
        self.plan['switch_90pct_timing'] = timing
        self._redraw_t90_failures()
        # Measuring alone does not change or enable the user's active wait table.
        bad = [p for p in result['points'] if p['reasons']]
        if bad:
            reasons = sorted({REASON_LABELS.get(r, r) for p in bad for r in p['reasons']})
            detail = ('；第' + '/'.join(str(p['target_order'] + 1) for p in bad)
                      + '点：' + '、'.join(reasons) + '。一键应用会拦截，等待未改动。')
        else:
            detail = (f"；建议等待小计{result['candidate_wait_sum_us']/1000:.2f} ms/帧。"
                      '点击“一键应用90%逐点等待”即可启用。')
        self.status.setText(f'已完成{self.point_count()}点等待测量：'
                            f"{result['measured_point_count']}点×1次90%实测，"
                            f"{result['fixed_wait_point_count']}点低电压固定300 µs（包含首点等待）"
                            + detail + ' S固定0，按本点最终ADC计算；ADC稳定不等于波长精度认证。')
        self.refresh_wait_controls()
        self._save_last_route_session(ready=self.plan_ready)

    def apply_beta_t90_waits(self, _checked=False):
        if not self.beta or self.worker is not None:
            return False
        try:
            from temporary_t90_waits import checked_t90_waits
            timing = (self.plan or {}).get('switch_90pct_timing', {})
            waits = checked_t90_waits(
                timing.get('t90_waits', {}), self.plan['rows'],
                channel=self.selected_channel(), selector=self.effective_feedback_selector(),
                fast=self.beta_single_adc.isChecked() and self.beta_fast_dac.isChecked(),
                machine_id=get_runtime_machine_id())
            # Validate everything first, then commit the entire table at once.
            self.point_waits = waits
            self.point_wait_route = self.point_route_key()
            self.use_point_waits.setChecked(True)
            timing['applied'] = True
            timing['applied_point_delays_us'] = list(waits)
            self.refresh_wait_controls()
            self.status.setText(
                f'已一键应用{len(waits)}点90%逐点等待：{min(waits)}～{max(waits)} µs，'
                f'等待小计{sum(waits)/1000:.2f} ms/帧（不含DAC、ADC和传输）。'
                '下次“开始测试”直接生效；不会自动开光。')
            self._save_last_route_session(ready=self.plan_ready)
            return True
        except (ValueError, KeyError, TypeError) as exc:
            self.status.setText(f'90%逐点等待未应用：{exc}')
            return False

    def generate_beta_balanced_waits(self, _checked=False):
        if not self.beta or self.worker is not None:
            return
        try:
            timing = (self.plan or {}).get('switch_90pct_timing', {})
            balance = timing.get('wait_balance', {})
            key = [[r['index'], list(r['codes'])] for r in self.plan['rows']]
            fast = self.beta_single_adc.isChecked() and self.beta_fast_dac.isChecked()
            if (balance.get('route_key') != key or not balance.get('candidate_available')
                    or balance.get('feedback_selector') != self.effective_feedback_selector()
                    or balance.get('signal_channel') != self.selected_channel()
                    or balance.get('fast_dac_writes') != fast):
                raise ValueError('路线、通道、档位或DAC方式不同，或有点没有可靠候选；请重新测量')
            waits = list(balance['candidate_point_delays_us'])
            # An unmeasured first/wrap point must never get shortened implicitly.
            if self._ensure_point_waits_for_route():
                waits[0] = max(waits[0], self.point_waits[0])
            command(self.plan['rows'], 32, 1, point_delays_us=waits)
            self.point_waits = waits
            self.point_wait_route = self.point_route_key()
            self.use_point_waits.setChecked(False)
            self.status.setText(
                f'已生成候选等待：等待小计{sum(waits)/1000:.2f} ms/帧，不含DAC、ADC和传输开销。'
                '未自动应用；首点/回绕未测，须整条路线与慢参考对照，绝对波长精度未验证。')
            self._save_last_route_session(ready=self.plan_ready)
        except (ValueError, KeyError, TypeError) as exc:
            self.status.setText(f'稳态候选未采用：{exc}')

    def show_switch_timing_results(self, _checked=False):
        timing = (self.plan or {}).get('switch_90pct_timing', {})
        points = timing.get('points', [])
        if not points:
            self.status.setText('当前路线尚无完整的90%测量结果')
            return
        if timing.get('route_key') != [[row['index'], list(row['codes'])] for row in self.plan['rows']]:
            self.status.setText('路线或DAC码已改变，请重新测量90%时间；旧结果未当作当前路线结果显示。')
            return
        if timing.get('t90_waits'):
            self._show_beta_t90_result_dialog(timing)
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('Beta · 逐点90%到达时间（不是波长稳定认证）')
        layout = QtWidgets.QVBoxLayout(dialog)
        style = 'GAIN/SOA合并写入' if timing.get('fast_dac_writes') else '5路各自写入'
        label = QtWidgets.QLabel(f"{timing.get('signal_channel_name', 'CH1')}，3轮正向测量，{style}；"
                                '时间起点为目标5路DAC写完。首点/反向回绕未测。')
        label.setWordWrap(True)
        layout.addWidget(label)
        table = QtWidgets.QTableWidget(len(points), 7, dialog)
        table.setHorizontalHeaderLabels(['目标点', '5路DAC写入 µs', 'ADC达到90% µs', '总到达时间 µs', '备注',
                                        '稳态候选等待 µs', '候选说明'])
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        balance = timing.get('wait_balance', {})
        balanced_points = balance.get('points', [])
        for index, item in enumerate(points):
            if item is None:
                values = [index + 1, '—', '—', '—', '首点/回绕未测']
            else:
                fmt = lambda v: '未测得' if v is None else f'{float(v):.1f}'
                values = [index + 1, fmt(item.get('dac_write_us')),
                          fmt(item.get('t90_after_target_write_us')),
                          fmt(item.get('t90_from_switch_begin_us')),
                          '无可判定边沿' if not item['meaningful_step'] else 'ADC主变化（非波长稳定）']
            candidate = balanced_points[index] if len(balanced_points) == len(points) else None
            values.extend([
                candidate.get('candidate_wait_us') if candidate and candidate.get('candidate_wait_us') else '—',
                '/'.join(candidate['reasons']) if candidate and candidate['reasons'] else
                '3轮观测长尾+55 µs余量；需整帧对照' if candidate else '未测得稳态候选',
            ])
            for column, value in enumerate(values):
                table.setItem(index, column, QtWidgets.QTableWidgetItem(str(value)))
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        layout.addWidget(table)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.resize(1000, 680)
        dialog.exec_()

    def _show_beta_t90_result_dialog(self, timing):
        from temporary_t90_waits import T90_SCHEMA, t90_point_diagnostic_text
        timing = self._t90_display_timing(timing)
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f'{len(timing["points"])}点90%稳定时间')
        layout = QtWidgets.QVBoxLayout(dialog)
        current_basis = timing['t90_waits'].get('schema') == T90_SCHEMA
        criterion_label = (
            'S固定0，按本点最终ADC的90%计算；采集仍是相邻波长切换，不从0启动。'
            '进入并保持在本点最终ADC的±10%内，允许偏差不设5码下限；' if current_basis else
            '这是旧的计算口径；当前按0.2 V分流，需重新测量后应用。')
        sampling_label = (f"{timing['signal_channel_name']}，{timing['t90_waits'].get('measured_point_count', 0)}点90%实测，"
                          f"{timing['t90_waits'].get('fixed_wait_point_count', 0)}点固定等待，包含首点等待；"
                          if current_basis else
                          f"{timing['signal_channel_name']}，历史每点{timing['t90_waits'].get('repeats', '未知')}次实测；")
        label = QtWidgets.QLabel(
            sampling_label +
            '目标5路DAC写完后开始计时。' + criterion_label +
            ('>0.2 V才测90%，测得时间加55 µs余量，按25 µs向上取整；'
             '≤0.2 V只做电压预检并固定300 µs，不代表实测稳定时间。' if current_basis else '') +
            'ADC稳定不等于波长精度认证。')
        label.setWordWrap(True)
        layout.addWidget(label)
        points = timing['points']
        table = QtWidgets.QTableWidget(len(points), 5, dialog)
        table.setHorizontalHeaderLabels(['5路DAC写入 µs', '90%稳定 µs',
                                        '总稳定时间 µs', '建议等待 µs', 'ADC偏差与判定明细'])
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setWordWrap(True)
        fmt = lambda value: '未测得' if value is None else f'{float(value):.1f}'
        for row, point in enumerate(points):
            fixed = point.get('wait_source') == 'low_voltage_fixed_300us'
            values = ['未测' if point.get('t90_measurement_skipped') else fmt(point.get('dac_write_us')),
                      '不测（≤0.2 V）' if fixed else fmt(point.get('settled_90pct_after_target_write_us')),
                      '固定等待' if fixed else fmt(point.get('settled_90pct_from_switch_begin_us')),
                      point.get('candidate_wait_us') or '不可应用',
                      t90_point_diagnostic_text(point)]
            for column, value in enumerate(values):
                cell = QtWidgets.QTableWidgetItem(str(value))
                cell.setToolTip(str(value))
                cell.setTextAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
                table.setItem(row, column, cell)
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(4, QtWidgets.QHeaderView.Stretch)
        def fit_diagnostic_rows(*_):
            # Qt's delegate can retain the huge height calculated before the
            # stretched column gets its final width. Measure that column at
            # its actual width so a few lines don't become a screen-high row.
            try:
                metrics = QtGui.QFontMetrics(table.font())
            except RuntimeError:
                return  # A queued resize may outlive an already closed dialog.
            width = max(40, table.columnWidth(4) - 16)
            for row in range(table.rowCount()):
                bounds = metrics.boundingRect(QtCore.QRect(0, 0, width, 100000),
                                              QtCore.Qt.TextWordWrap, table.item(row, 4).text())
                table.setRowHeight(row, max(metrics.lineSpacing() + 16, bounds.height() + 16))
        table.horizontalHeader().sectionResized.connect(
            lambda *_: QtCore.QTimer.singleShot(0, fit_diagnostic_rows))
        QtCore.QTimer.singleShot(0, fit_diagnostic_rows)
        layout.addWidget(table)
        notice = QtWidgets.QLabel(
            '旧计算口径不能套用，请重新测量。' if not current_basis else
            '所有点都有可用等待（含低电压固定300 µs），可一键应用。' if timing['t90_waits']['candidate_available'] else
            '存在未可靠测得或超过15 ms的点，不能整表应用；请查看表中原因。')
        notice.setWordWrap(True)
        layout.addWidget(notice)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        apply = buttons.addButton('一键应用到逐点等待', QtWidgets.QDialogButtonBox.ActionRole)
        apply.setEnabled(self.worker is None)
        def apply_waits():
            if self.apply_beta_t90_waits():
                dialog.accept()
            else:
                notice.setText(self.status.text())
        apply.clicked.connect(apply_waits)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.resize(1120, 710)
        dialog.exec_()

    def _t90_display_timing(self, timing):
        """Read raw saved capture only to fill missing display diagnostics.

        Do not alter the saved result, active waits, or device state. An old
        formula or mismatching capture must never be relabeled as current.
        """
        from temporary_t90_waits import T90_SCHEMA, analyze_t90_waits
        cached = timing.get('t90_waits', {})
        if cached.get('schema') != T90_SCHEMA:
            return timing
        if all('tail_max_deviation_from_target_adc' in metric
               for point in timing.get('points', [])
               for metric in point.get('individual_repeats', [])):
            return timing
        try:
            report = json.loads(Path(timing['source']).read_text(encoding='utf-8'))
            measured = analyze_t90_waits(report)
            bindings = ('schema', 'criterion', 'route_key', 'machine_id', 'signal_channel',
                        'feedback_selector', 'fast_dac_writes', 'repeats')
            if any(measured.get(key) != cached.get(key) for key in bindings):
                return timing
            old_points, new_points = timing['points'], measured['points']
            keys = ('target_order', 'target_index', 'source_order', 'target_adc',
                    'dac_write_us', 'settled_90pct_after_target_write_us',
                    'candidate_wait_us', 'required_wait_us', 'reasons')
            if len(old_points) != len(new_points) or any(
                    any(old.get(key) != new.get(key) for key in keys)
                    for old, new in zip(old_points, new_points)):
                return timing
            return dict(timing, points=new_points, t90_waits=measured)
        except (OSError, ValueError, KeyError, TypeError):
            return timing  # Show available values; never invent missing detail.

    def start_equal_interval_reference(self, _checked=False):
        """Acquire only the requested wavelength window at a selectable stride."""
        import app_JDSU as app
        try:
            indices, actual_nm = equal_interval_indices_for_range(
                app.load_fullband_accuracy_table(),
                self.scan_start_nm.value(), self.scan_stop_nm.value(),
                self.equal_interval_nm.value(),
            )
        except Exception as exc:
            self.status.setText(f'等间隔密集光谱未启动：{exc}')
            return
        self.equal_interval_nm.setValue(actual_nm)
        self.dense_scan_range_nm = (
            float(self.scan_start_nm.value()), float(self.scan_stop_nm.value())
        )
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
            self.status.setText("手指应力测试模式需要真实USB或LAN局域网全功能通道")
            return
        try:
            signal_channel = self.selected_channel()
            feedback_selector = self.effective_feedback_selector(signal_channel)
            if (self.plan is not None and not reference
                    and int(self.plan.get('signal_channel', 1)) != signal_channel):
                raise ValueError('当前点表属于其他ADC通道，请为所选通道重采光谱并手动放置45点')
            # Zero is the firmware's continuous-until-stopped contract. Never
            # let hidden legacy controls impose a 256-frame (or other) limit.
            selected_cycles = 0
            options = dict(first_delay_us=self.first_delay.value(),
                           boundary_extra_us=0,
                           feedback_selector=feedback_selector,
                           adc_channel=signal_channel)
            # PWT1 already specifies absolute waits on supported firmware.
            # Keep the legacy additive wire field zero, including at peak entry.
            if not (reference or reference_line):
                options['point_delays_us'] = self._uniform_point_waits()
                if self.beta:
                    single = self.beta_single_adc.isChecked()
                    options.update(single_sample=single,
                                   fast_dac_writes=single and self.beta_fast_dac.isChecked())
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
                timing = self.plan.get('switch_90pct_timing', {})
                if (self.beta and timing and
                        bool(timing.get('fast_dac_writes')) != bool(options.get('fast_dac_writes'))):
                    raise ValueError('当前DAC写入方式与90%测量不同，请重新测量或取消使用逐点等待')
                if (self.beta and timing and
                        timing.get('feedback_selector') != feedback_selector):
                    raise ValueError('当前模拟档位与时间测量不同，请重新测量或取消使用逐点等待')
                if (self.beta and timing and timing.get('signal_channel') != signal_channel):
                    raise ValueError('当前通道与时间测量不同，请重新测量或取消使用逐点等待')
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
        self.worker = worker_type(app.ser, json.loads(json.dumps(self.plan)), output, selected_cycles, self,
                                  coalesce_display=self.beta and not (reference or reference_line), **options)
        if reference:
            self._auto_select_after_dense = False
            self._live_dense_rows = []
            self._reset_dense_draw_state()
            self._dense_capture_active = True
            self.worker.reference.connect(self.reference_ready)
            self.worker.reference_point.connect(self.update_dense_reference_point)
        elif reference_line:
            self._live_reference_line_rows = []
            self._reference_line_capture_active = True
            self.worker.reference.connect(self.reference_line_ready)
            self.worker.reference_point.connect(self.update_reference_line_point)
        self.worker.frame.connect(self.update_frame)
        self.worker.message.connect(self.status.setText)
        self.worker.finished.connect(self.finished)
        for widget in (self.channel_combo, self.equal_interval_reference_button,
                       self.scan_start_nm, self.scan_stop_nm,
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
            f'正在采集当前{self.point_count()}点参考线（10 ms/点，单次ADC）…' if reference_line
            else (f'正在按等间隔模式采集{len(dense_indices)}点'
                  f'{self.channel_name()}光谱（{self.equal_interval_nm.value():.2f} nm，'
                  f'{self.scan_start_nm.value():.2f}～{self.scan_stop_nm.value():.2f} nm，'
                  f'{self.equal_interval_settle.value():.3f} s/点）…')
            if reference and equal_interval
            else f"正在采集{self.channel_name()}密集稳定参考（可能需要数分钟）…" if reference
            else f"正在执行手指应力{self.point_count()}点扫描…"
        )
        self.worker.start()
        if self.beta and not (reference or reference_line):
            self._temporary_draw_timer.start()
        self.refresh_wait_controls()

    def redraw_values(self, _value=None):
        self.redraw_dense_selection()
        if self.plan is None:
            self._update_plot_unit_label()
            return
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
                start = group* self.points_per_peak()
                curve.setData(wavelengths[start:start+ self.points_per_peak()],
                              self._plot_display_values(
                                  values[start:start+ self.points_per_peak()], reference=reference,
                                  selector=selector,
                              ))
        self._update_plot_unit_label()
        self._refresh_table_display_values()
        self._redraw_reference_failures()
        self._redraw_t90_failures()
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
        x_values = []
        peak_index = self.peak_view.currentIndex()
        if (self.manual_selection_active and self.show_dense_selection.isChecked()
                and self.dense_wavelengths is not None):
            x_values = [float(value) for value in self.dense_wavelengths
                        if np.isfinite(value)]
        elif self.plan is not None and peak_index > 0:
            rows = self.plan['rows'][(peak_index - 1) * self.points_per_peak():peak_index * self.points_per_peak()]
            x_values.extend(float(row['measured_nm']) for row in rows)
            for curve in (
                self.curves[peak_index - 1],
                self.reference_curves[peak_index - 1],
                self.fitted_curves[peak_index - 1],
            ):
                x_data, _y_data = curve.getData()
                if x_data is not None:
                    x_values.extend(
                        float(value) for value in x_data if np.isfinite(value)
                    )
        else:
            for curve in self.plot.listDataItems():
                if not curve.isVisible():
                    continue
                x_data, y_data = curve.getData()
                if x_data is None or y_data is None:
                    continue
                x_values.extend(
                    float(x_value) for x_value, y_value in zip(x_data, y_data)
                    if np.isfinite(x_value) and np.isfinite(y_value)
                )
        if not x_values and self.plan is not None:
            x_values = [float(row['measured_nm']) for row in self.plan['rows']]
        if x_values:
            left, right = min(x_values), max(x_values)
            span = right - left
            margin = max(0.04, span * 0.045) if span > 0 else 0.08
            self.plot.setXRange(left - margin, right + margin, padding=0)
            self.plot.enableAutoRange(axis='x', enable=False)

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

    def _flush_temporary_display(self):
        take = getattr(self.worker, 'take_display_frame', None)
        if callable(take):
            frame = take()
            if isinstance(frame, dict):
                self.update_frame(frame)

    def update_frame(self, frame):
        self._observe_developer_frame(frame)
        firmware_version = frame.get('firmware')
        if isinstance(firmware_version, int) and firmware_version > 0:
            self.board_firmware_version = firmware_version
            self.board_firmware_seen_monotonic = time.monotonic()
        values = frame_adc_values(frame)
        self.latest_values = values
        self.latest_channel = int(frame.get('adc_channel', self.selected_channel()))
        self.latest_feedback_selector = int(frame.get(
            'feedback_selector', self.effective_feedback_selector(self.latest_channel)
        ))
        makima = self.peak_fit_method.currentData() == 'makima' or self.points_per_peak() == 7
        if makima:
            from temporary_peak_fit_makima import fit_temporary_frame
        else:
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
            + ('点线为Makima实验性拟合；中心可能偏移，未经波长计精度验证。'
             if makima else '点线为高斯拟合；波长精度未验证。')
            + '圆点/实线仍是原始值。'
        )
        self.redraw_values()
        if self.fit_pending and frame['sequence'] >= 16:
            self.fit_values()
        self._refresh_table_display_values()
        if frame['sequence'] >= 16:
            self.frame_times.append(frame['cycle_start_us'])
            del self.frame_times[:-64]
        rate = ''
        display_stats = frame.get('display_statistics', {})
        board_hz = display_stats.get('board_hz')
        if board_hz is not None:
            rate = f" · {board_hz:.2f} Hz（板端采集）"
        elif len(self.frame_times) > 1:
            rate = f" · {1e6/np.mean(np.diff(self.frame_times[-64:])):.2f} Hz（板端）"
        if display_stats:
            rate += f"；显示最新帧，原始帧全部记录（{display_stats['received_frames']}帧）"
        q = frame['quality']
        issues = [label for key, label in (('clipped', 'ADC饱和'), ('unsettled', '双采样差异较大（可能是信号变化）'),
                                         ('peak_at_edge', '峰在窗口边缘，扫描点仍保持不变')) if q[key]]
        drifted = [str(i + 1) for i, streak in enumerate(self.fit_failure_streaks) if streak >= 3]
        if drifted:
            issues.append(
                '峰' + '/'.join(drifted)
                + ('最近显示帧出现拟合质量警告：' if display_stats else '连续出现拟合质量警告：')
                + '结果已保留，请结合表格中的⚠原因判断'
            )
        if not q['warmed_up']:
            issues.append('预热中')
        sampling_note = ('单次ADC采集中；未验证波长稳定' if frame.get('single_sample')
                         else '原始数据采集中，双采样不能证明已稳定')
        self.status.setText(f"第{frame['sequence']+1}帧{rate} · " + (' / '.join(issues) or sampling_note))

    def stop(self):
        if self.worker is not None:
            self.worker.stop()
            self.stop_button.setEnabled(False)

    def finished(self):
        import app_JDSU as app
        self._temporary_draw_timer.stop()
        # Drain the final paint before deleting its mailbox. Preserve the
        # worker's completion/error/shutdown message instead of hiding it.
        completion_message = self.status.text()
        self._flush_temporary_display()
        self.status.setText(completion_message)
        auto_select_after_dense = self._auto_select_after_dense
        self._auto_select_after_dense = False
        app.switch_mode_enable = True
        self.worker.deleteLater()
        self.worker = None
        self._dense_capture_active = False
        self._live_dense_rows = []
        self._reset_dense_draw_state()
        self._reference_line_capture_active = False
        self._live_reference_line_rows = []
        self.peak_fit_status.setText('已停采，保留最后一帧拟合（非实时）；波长精度未验证。')
        for widget in (self.channel_combo, self.equal_interval_reference_button,
                       self.scan_start_nm, self.scan_stop_nm,
                       self.equal_interval_nm, self.equal_interval_settle,
                       self.save_finger_button, self.load_finger_button,
                       self.spacing, self.cycles,
                       self.analog_gain, self.first_delay, self.boundary_delay):
            widget.setEnabled(True)
        self.spacing.setEnabled(False)
        route_available = self.plan is not None and not self.manual_selection_active
        self.start_button.setEnabled(route_available and self.plan_ready)
        self.reference_line_button.setEnabled(route_available)
        self.test_switch_timing.setEnabled(route_available)
        self.save_finger_button.setEnabled(
            self._valid_dense_spectrum(self.dense_wavelengths, self.dense_values)
        )
        self.continuous.setEnabled(True)
        self.cycles.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.stop_dense_button.setEnabled(False)
        self._sync_manual_selection_controls()
        self.redraw_dense_selection()
        self._redraw_reference_failures()
        self._sync_channel_ui()
        self.refresh_wait_controls()
        if auto_select_after_dense:
            dense_status = self.status.text()
            self.auto_select_points()
            self.status.setText(f'{dense_status}\n{self.status.text()}')
