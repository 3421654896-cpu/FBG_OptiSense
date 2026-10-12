"""Voltage-gated 90%-settling waits for a complete temporary scan cycle.

S is fixed to zero for the calculation only: 90% means a sampled suffix within
10% of THIS point's final ADC, with no minimum ADC-code tolerance. Capture still switches
between adjacent wavelengths; it never starts every point from zero output.
Raw source ADC is preserved. This is not a wavelength accuracy certificate.
A held-target tail first screens each point: <=0.2 V uses the operator's fixed
300 us without a switch-timing capture; higher points get one transition.
"""

import math
import numpy as np

from temporary_optical_units import adc_code_to_voltage
from temporary_test_mode import command, points_per_peak_for_rows
from temporary_wait_balance import _repeat_metric, ADC_GROUP_MARGIN_US, MAX_WAIT_US, WAIT_QUANTUM_US


REASON_LABELS = {
    'adc_clipped': 'ADC饱和',
    'tail_not_repeatably_flat': '末段不稳定',
    'insufficient_stable_tail': '没有足够稳定末段',
    'required_wait_exceeds_protocol_limit': '所需等待超过15 ms上限',
    'frame_wrap_unmeasured': '首点回绕未测',
}

T90_REPEATS = 1
T90_SCHEMA = 'temporary_point_voltage_gated_single_90pct_waits_v5'
T90_CRITERION = 'single_transition_within_10pct_target_adc_or_low_voltage_fixed_wait'
T90_VOLTAGE_THRESHOLD_V = .2
T90_LOW_VOLTAGE_WAIT_US = 300
T90_GATE_POLICY = 'steady_voltage_gt_0p2v_else_fixed_300us'


def steady_probe_level(record, channel):
    """Validate real board samples; the median of the last three defines F."""
    try:
        values = np.asarray(record['direct_adc'], dtype=float)
        final = float(np.median(values[-3:, channel]))
        metric = _repeat_metric(record, final, abs(final)*.10, channel, reference_baseline=0.)
    except (IndexError, TypeError, KeyError) as exc:
        raise ValueError('稳态电压预检原始ADC无效') from exc
    if 'adc_clipped' in metric['reasons']:
        raise ValueError('稳态电压预检ADC饱和，等待未应用')
    return dict(target_adc=final, target_voltage_v=float(adc_code_to_voltage(final)),
                tail_adc=metric['tail_adc'],
                tail_times_us=metric['tail_sample_times_after_target_write_us'])


def t90_point_diagnostic_text(point):
    """Show actual deviations/limits without inferring unavailable old details."""
    reasons = point.get('reasons', [])
    if 'frame_wrap_unmeasured' in reasons:
        return '首点回绕未测，无ADC记录'

    def number(value):
        return f'{float(value):.3f}'.rstrip('0').rstrip('.')

    final = point.get('target_adc')
    if point.get('wait_source') == 'low_voltage_fixed_300us' and final is not None:
        if reasons:
            return (f'最终稳定电压 {float(adc_code_to_voltage(final)):.6f} V\n'
                    '低电压固定等待未采用：' + '；'.join(REASON_LABELS.get(reason, reason) for reason in reasons))
        return (f'最终稳定电压 {float(adc_code_to_voltage(final)):.6f} V\n'
                '≤0.2 V，采用固定300 µs（不是实测90%时间）\n'
                + ('只做稳态电压预检，未采集相邻切换90%时间'
                   if point.get('t90_measurement_skipped') else
                   '预检后电压下降；按实测末段电压采用固定等待'))
    metrics = point.get('individual_repeats', [])
    tolerances = point.get('repeat_tolerances_adc', [])
    if final is None or not metrics:
        return '；'.join(REASON_LABELS.get(reason, reason) for reason in reasons) or '缺少ADC诊断明细'

    def deviation(value):
        codes = f'{number(value)}码'
        return codes + (f'（{value / abs(final) * 100:.2f}%）' if final else '（F=0，不计算百分比）')

    lines = [f'最终稳定电压 {float(adc_code_to_voltage(final)):.6f} V']
    for index, metric in enumerate(metrics):
        tolerance = metric.get('tolerance_adc', tolerances[index] if index < len(tolerances) else None)
        if len(metrics) > 1:
            lines.append(f'历史第{index + 1}轮：')
        if tolerance is None:
            lines.append('记录未保存允许偏差，不能计算超出量')
            continue
        lines.append(f'允许偏差±{deviation(tolerance)}')
        for label, key in (
                ('末段最大偏差', 'tail_max_deviation_from_target_adc'),
                ('末段波动（最大−最小）', 'tail_range_adc')):
            value = metric.get(key)
            if value is not None:
                excess = value - tolerance
                lines.append(f'{label} {deviation(value)}，'
                             + (f'超{number(excess)}码' if excess > 0 else '未超限'))
        tail = metric.get('tail_adc')
        if tail is not None:
            lines.append('末3个ADC：' + ' / '.join(number(value) for value in tail))
        if 'tail_not_repeatably_flat' in metric.get('reasons', []):
            span = metric.get('tail_span_us')
            if span is not None and span < 20000:
                lines.append(f'末3样本跨度{number(span / 1000)}ms，要求≥20ms，'
                             f'差{number((20000 - span) / 1000)}ms')
        if 'insufficient_stable_tail' in metric.get('reasons', []):
            count = metric.get('observed_stable_sample_count')
            span = metric.get('observed_stable_span_us')
            if count is not None:
                lines.append(f'稳定末段{count}个样本，要求≥3个'
                             + (f'，少{3 - count}个' if count < 3 else ''))
            if span is not None:
                lines.append(f'稳定观察跨度{number(span / 1000)}ms，要求≥20ms'
                             + (f'，差{number((20000 - span) / 1000)}ms' if span < 20000 else ''))
        if 'adc_clipped' in metric.get('reasons', []):
            peak = metric.get('signal_peak_adc')
            lines.append(f'ADC饱和：最大{number(peak)}码，判定阈值4080码' if peak is not None else 'ADC饱和')
        if any(key not in metric for key in ('tail_max_deviation_from_target_adc',
                                            'observed_stable_sample_count', 'observed_stable_span_us')):
            lines.append('旧记录缺少部分偏差/样本/时间明细，不能补算缺失值')
    if 'required_wait_exceeds_protocol_limit' in reasons:
        required = point.get('required_wait_us')
        if required is not None:
            lines.append(f'所需等待{number(required)}µs，上限15000µs，超{number(required - 15000)}µs')
    if reasons:
        lines.append('判定：' + '；'.join(REASON_LABELS.get(reason, reason) for reason in reasons))
    return '\n'.join(lines)


def analyze_t90_waits(report):
    if not (report.get('complete') and report.get('exact_disarm_ack')
            and report.get('exact_soa_shutter_ack')):
        raise ValueError('测量未完成或停采/关光未确认')
    rows = report['route']
    count = len(rows)
    group = points_per_peak_for_rows(rows)
    command(rows, 32, 1)  # Same lossless route and 145 mA ceiling as real scanning.
    repeats = report['repeats']
    if type(repeats) is not int or repeats != T90_REPEATS:
        raise ValueError('当前90%稳定时间每点只测量一次，请重新测量')
    if (report.get('voltage_gate_policy') != T90_GATE_POLICY
            or report.get('voltage_threshold_v') != T90_VOLTAGE_THRESHOLD_V
            or report.get('low_voltage_wait_us') != T90_LOW_VOLTAGE_WAIT_US):
        raise ValueError('当前按0.2 V分流测量、低电压固定300 µs，请重新测量')
    channel = report.get('signal_channel', 1)
    selector = report.get('feedback_selector', report.get('ch1_feedback_selector'))
    if (type(channel) is not int or channel not in range(4)
            or type(selector) is not int or selector not in range(4)
            or (channel >= 2 and selector != 0)):
        raise ValueError('测量通道或模拟档位无效')
    fast = bool(report.get('fast_dac_writes', False))
    wrapped = report.get('cycle_wrap_included') is True
    orders = list(range(count)) if wrapped else list(range(1, count))
    records = report['captures']
    if len(records) != repeats * len(orders):
        raise ValueError('逐点切换记录不完整')
    by_target = [[] for _ in rows]
    for position, record in enumerate(records):
        repeat, offset = divmod(position, len(orders))
        target = orders[offset]
        source = (target - 1) % count
        if (record.get('repeat') != repeat or record.get('route_source_order') != source
                or record.get('route_target_order') != target
                or record.get('source_fullband_index') != rows[source]['index']
                or record.get('target_fullband_index') != rows[target]['index']
                or bool(record.get('fast_dac_writes', False)) != fast):
            raise ValueError('切换顺序、物理索引或DAC方式不匹配')
        probe = record.get('voltage_probe', {})
        if (probe.get('source_fullband_index') != rows[target]['index']
                or probe.get('target_fullband_index') != rows[target]['index']
                or bool(probe.get('fast_dac_writes', False)) != fast):
            raise ValueError('稳态电压预检物理索引或DAC方式不匹配')
        level = steady_probe_level(probe, channel)
        low = level['target_voltage_v'] <= T90_VOLTAGE_THRESHOLD_V
        expected_kind = 'low_voltage_fixed_wait' if low else 'measured_transition'
        if record.get('measurement_kind') != expected_kind:
            raise ValueError('稳态电压与切换测量分流不一致')
        if low:
            if 'direct_adc' in record:
                raise ValueError('跳过的90%测量不能携带伪造的切换样本')
        else:
            write_us = record.get('path_update_us')
            if (type(write_us) not in (int, float) or not math.isfinite(write_us)
                    or write_us < 0):
                raise ValueError('DAC写入时间无效')
        by_target[target].append(record)
    waits = [None] * count
    points = []
    for target, selected in enumerate(by_target):
        if not selected:
            points.append({'target_order': target, 'target_index': rows[target]['index'],
                           'source_order': None, 'candidate_wait_us': None,
                           'required_wait_us': None, 'reasons': ['frame_wrap_unmeasured']})
            continue
        record = selected[0]
        probe_level = steady_probe_level(record['voltage_probe'], channel)
        skipped = record['measurement_kind'] == 'low_voltage_fixed_wait'
        if skipped:
            waits[target] = T90_LOW_VOLTAGE_WAIT_US
            points.append(dict(
                target_order=target, target_index=rows[target]['index'], source_order=(target-1)%count,
                category='frame_wrap' if target == 0 else 'cross_peak' if target % group == 0 else 'within_peak',
                target_adc=probe_level['target_adc'], target_voltage_v=probe_level['target_voltage_v'],
                calculation_start_adc=0., target_90pct_adc=None, repeat_tolerances_adc=[],
                meaningful_step=False, dac_write_us=None, settled_90pct_after_target_write_us=None,
                settled_90pct_from_switch_begin_us=None, required_wait_us=T90_LOW_VOLTAGE_WAIT_US,
                candidate_wait_us=T90_LOW_VOLTAGE_WAIT_US, individual_repeats=[], reasons=[],
                wait_source='low_voltage_fixed_300us', t90_measurement_skipped=True,
                voltage_probe_level=probe_level))
            continue
        try:
            tails = [np.asarray(r['direct_adc'], dtype=float)[-3:, channel] for r in selected]
            final = float(np.median([np.median(tail) for tail in tails]))
        except (ValueError, IndexError, TypeError) as exc:
            raise ValueError('ADC原始记录无效') from exc
        # User-defined S=0 is a calculation reference, not an optical restart.
        # Previous-point ADC must not affect the target's 90% tolerance.
        tolerances = [abs(final) * .10] * len(selected)
        metrics = [_repeat_metric(r, final, tolerance, channel, reference_baseline=0.0)
                   for r, tolerance in zip(selected, tolerances)]
        reasons = sorted({reason for metric in metrics for reason in metric['reasons']})
        final_voltage = float(adc_code_to_voltage(final))
        low_final = final_voltage <= T90_VOLTAGE_THRESHOLD_V
        if low_final:
            # A real transition was taken because the probe was high. Never
            # pretend it was skipped; the current measured tail still wins.
            reasons = [reason for reason in reasons if reason == 'adc_clipped']
        stable = required = None
        if low_final and not reasons:
            required = waits[target] = T90_LOW_VOLTAGE_WAIT_US
        elif not reasons:
            stable = max(m['settled_after_target_write_us'] for m in metrics)
            required = max(50, math.ceil((stable + ADC_GROUP_MARGIN_US) / WAIT_QUANTUM_US)
                           * WAIT_QUANTUM_US)
            if required > MAX_WAIT_US:
                reasons.append('required_wait_exceeds_protocol_limit')
            else:
                waits[target] = required
        dac = max(float(r['path_update_us']) for r in selected)
        points.append({
            'target_order': target, 'target_index': rows[target]['index'],
            'source_order': (target - 1) % count,
            'category': 'frame_wrap' if target == 0 else
                        'cross_peak' if target % group == 0 else 'within_peak',
            'target_adc': final, 'target_voltage_v': final_voltage, 'calculation_start_adc': 0.0,
            'target_90pct_adc': final * .90,
            'repeat_tolerances_adc': tolerances,
            'meaningful_step': any(m['meaningful_step'] for m in metrics),
            'dac_write_us': dac,
            'settled_90pct_after_target_write_us': stable,
            'settled_90pct_from_switch_begin_us': (None if stable is None else
                max(m['settled_after_target_write_us'] + float(r['path_update_us'])
                    for r, m in zip(selected, metrics))),
            'required_wait_us': required, 'candidate_wait_us': waits[target],
            'individual_repeats': metrics, 'reasons': reasons,
            'wait_source': 'low_voltage_fixed_300us' if low_final else 'measured_90pct',
            't90_measurement_skipped': False, 'voltage_probe_level': probe_level,
        })
    available = all(wait is not None for wait in waits)
    if available:
        command(rows, 32, 1, point_delays_us=waits)
    return {
        'schema': T90_SCHEMA,
        'criterion': T90_CRITERION,
        'tolerance_fraction': .10, 'tolerance_floor_adc': 0.0,
        'calculation_start_adc': 0.0,
        'reference_model': 'target_adc_relative_to_zero',
        'capture_mode': 'adjacent_wavelength_transitions',
        'measurement_mode': 'single_voltage_gated_transition_per_point',
        'voltage_gate_policy': T90_GATE_POLICY, 'voltage_threshold_v': T90_VOLTAGE_THRESHOLD_V,
        'low_voltage_wait_us': T90_LOW_VOLTAGE_WAIT_US,
        'route_key': [[r['index'], list(r['codes'])] for r in rows],
        'machine_id': report.get('machine_id'),
        'signal_channel': channel, 'signal_channel_name': f'CH{channel}',
        'feedback_selector': selector, 'fast_dac_writes': fast,
        'repeats': repeats, 'adc_group_margin_us': ADC_GROUP_MARGIN_US,
        'candidate_available': available, 'candidate_point_delays_us': waits,
        'candidate_wait_sum_us': sum(waits) if available else None,
        'points': points, 'cycle_wrap_included': wrapped,
        'reverse_wrap_tested': wrapped and not points[0].get('t90_measurement_skipped', True),
        'first_point_unmeasured': not wrapped or points[0].get('t90_measurement_skipped', True),
        'first_point_unhandled': not wrapped,
        'measured_point_count': sum(p.get('t90_measurement_skipped') is False for p in points),
        'fixed_wait_point_count': sum(p.get('wait_source') == 'low_voltage_fixed_300us' for p in points),
        'startup_from_shutter_tested': False,
        'absolute_wavelength_accuracy_verified': False,
        'scope': 'held-source ADC transients; sampled settling, not continuous-route certification',
    }


def checked_t90_waits(result, rows, *, channel, selector, fast, machine_id):
    """Atomic one-click application preflight; no device I/O and no capping."""
    if (result.get('schema') != T90_SCHEMA or result.get('criterion') != T90_CRITERION
            or result.get('tolerance_fraction') != .10 or result.get('tolerance_floor_adc') != 0.0
            or result.get('calculation_start_adc') != 0.0
            or result.get('reference_model') != 'target_adc_relative_to_zero'
            or result.get('capture_mode') != 'adjacent_wavelength_transitions'
            or result.get('measurement_mode') != 'single_voltage_gated_transition_per_point'
            or result.get('voltage_gate_policy') != T90_GATE_POLICY
            or result.get('voltage_threshold_v') != T90_VOLTAGE_THRESHOLD_V
            or result.get('low_voltage_wait_us') != T90_LOW_VOLTAGE_WAIT_US
            or result.get('adc_group_margin_us') != ADC_GROUP_MARGIN_US
            or type(result.get('repeats')) is not int or result['repeats'] != T90_REPEATS):
        raise ValueError('当前按0.2 V分流，本点ADC相对0单次测量或固定300 µs，请重新测量；旧结果未应用')
    if (result.get('route_key') != [[r['index'], list(r['codes'])] for r in rows]
            or result.get('signal_channel') != channel
            or result.get('feedback_selector') != selector
            or result.get('fast_dac_writes') != bool(fast)
            or result.get('machine_id') != machine_id):
        raise ValueError('机号、路线、通道、档位或DAC方式已改变，请重新测量')
    if not result.get('cycle_wrap_included') or result.get('first_point_unhandled'):
        raise ValueError('首点回绕未测，不能应用到完整循环')
    if not result.get('candidate_available'):
        bad = [str(p['target_order'] + 1) for p in result.get('points', []) if p.get('reasons')]
        raise ValueError('第' + '/'.join(bad) + '点未可靠测得或超过15 ms；等待未改动')
    waits = list(result['candidate_point_delays_us'])
    points = result.get('points', [])
    if (len(points) != len(rows) or len(waits) != len(rows)
            or any(p.get('target_order') != i or p.get('target_index') != rows[i]['index']
                   or p.get('reasons') or p.get('candidate_wait_us') != waits[i]
                   or p.get('required_wait_us') != waits[i] for i, p in enumerate(points))):
        raise ValueError('测量与逐点等待表不一致')
    for point in points:
        final = point.get('target_adc')
        if type(final) not in (int, float) or not math.isfinite(final) or not 0 <= final < 4080:
            raise ValueError('最终稳定电压无效')
        voltage = float(adc_code_to_voltage(final))
        if point.get('target_voltage_v') != voltage:
            raise ValueError('最终稳定电压与ADC记录不一致')
        if voltage <= T90_VOLTAGE_THRESHOLD_V:
            if point.get('wait_source') != 'low_voltage_fixed_300us' or point['candidate_wait_us'] != 300:
                raise ValueError('低电压点必须使用固定300 µs')
        elif point.get('wait_source') != 'measured_90pct' or point.get('t90_measurement_skipped') is not False:
            raise ValueError('高电压点缺少实际90%测量')
        else:
            stable = point.get('settled_90pct_after_target_write_us')
            if (type(stable) not in (int,float) or not math.isfinite(stable) or stable < 0
                    or point['required_wait_us'] != max(50, math.ceil((stable + ADC_GROUP_MARGIN_US)
                                                                      / WAIT_QUANTUM_US)*WAIT_QUANTUM_US)):
                raise ValueError('高电压点等待与实际稳定时间不一致')
    command(rows, 32, 1, point_delays_us=waits, adc_channel=channel,
            feedback_selector=selector)
    return waits
