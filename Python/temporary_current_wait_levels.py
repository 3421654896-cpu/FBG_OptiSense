"""Actual cyclic scan levels relative to separately held-target tail levels.

No interpolation at an unsampled wait, no clipping percentages to 100%, and
no application of waits. This is an amplitude diagnostic, not t90 certification.
"""
from pathlib import Path
import time

import numpy as np

from temporary_optical_units import adc_code_to_voltage
from temporary_test_mode import command, execute_device, frame_adc_values
from temporary_wait_balance import _repeat_metric


SCHEMA = 'temporary_current_wait_levels_v1'
SCAN_CYCLES = 32
REASON_LABELS = {
    'reference_unstable': '最终电压末段不稳定，比例仅供参考',
    'reference_clipped': '最终电压ADC饱和，比例不可用',
    'sample_clipped': '当前采样ADC饱和，比例不可用',
    'zero_reference': '最终稳定电压为0，不能计算比例',
}


def analyze_current_wait_levels(report):
    """Recompute from the actual last warmed-up frame and raw target probes."""
    if (report.get('schema') != SCHEMA or not report.get('complete')
            or not report.get('exact_disarm_ack') or not report.get('exact_soa_shutter_ack')):
        raise ValueError('当前等待测量未完整完成或停光未确认')
    route = report['route']
    waits = report['point_delays_us']
    channel, selector = report['signal_channel'], report['feedback_selector']
    fast, single = report['fast_dac_writes'], report['single_sample']
    if (type(channel) is not int or channel not in range(4) or not single
            or report.get('wait_mode') not in ('point_waits', 'uniform_and_cross_peak')):
        raise ValueError('当前等待测量需要有效通道及单次ADC')
    command(route, SCAN_CYCLES, 1, point_delays_us=waits, adc_channel=channel,
            feedback_selector=selector, single_sample=True, fast_dac_writes=fast)
    scan = report['scan_capture']
    if (not scan.get('complete') or not scan.get('disarm_ack') or not scan.get('shutter_ack')
            or scan.get('point_delays_us') != waits or scan.get('signal_channel') != channel
            or scan.get('feedback_selector') != selector or scan.get('single_sample') != single
            or scan.get('fast_dac_writes') != fast or scan.get('plan', {}).get('rows') != route
            or len(scan.get('frames', [])) != SCAN_CYCLES
            or [f['sequence'] for f in scan['frames']] != list(range(SCAN_CYCLES))):
        raise ValueError('循环采集的路线、通道、等待或完整性不匹配')
    frame = scan['frames'][-1]
    if (frame.get('kind') != 'data' or frame.get('adc_channel') != channel
            or frame.get('feedback_selector') != selector or frame.get('single_sample') != single
            or frame.get('fast_dac_writes') != fast or frame.get('i2c_errors')
            or frame.get('spi_errors') or len(frame.get('records', [])) != len(route)):
        raise ValueError('最后一帧的采样身份或记录不匹配')
    values = np.asarray(frame_adc_values(frame), dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 4095):
        raise ValueError('当前采样ADC无效')
    probes = report['voltage_probes']
    if len(probes) != len(route):
        raise ValueError('最终电压预检记录不完整')
    points = []
    previous_end = -1
    for order, (row, adc, probe, record) in enumerate(zip(route, values, probes, frame['records'])):
        if (probe.get('route_target_order') != order
                or probe.get('source_fullband_index') != row['index']
                or probe.get('target_fullband_index') != row['index']
                or probe.get('fast_dac_writes') != fast):
            raise ValueError('最终电压预检未绑定到当前真实目标行')
        write, end = record['write_end_us'], record['sample_end_us']
        if not previous_end < write < end or end - write < waits[order]:
            raise ValueError('当前采样的板上时序无效')
        previous_end = end
        target = float(np.median(np.asarray(probe['direct_adc'], dtype=float)[-3:, channel]))
        metric = _repeat_metric(probe, target, abs(target) * .1, channel, reference_baseline=0.)
        reasons = []
        if 'adc_clipped' in metric['reasons']:
            reasons.append('reference_clipped')
        if 'tail_not_repeatably_flat' in metric['reasons']:
            reasons.append('reference_unstable')
        if adc >= 4080:
            reasons.append('sample_clipped')
        if target == 0:
            reasons.append('zero_reference')
        # An unstable reference can still be inspected, but never called a
        # verified stable level. Saturation or a zero divisor has no percentage.
        percentage = (float(adc / target * 100.) if target > 0
                      and not set(reasons) & {'reference_clipped', 'sample_clipped'} else None)
        points.append(dict(
            target_order=order, target_index=row['index'], source_order=(order - 1) % len(route),
            wavelength_nm=float(row['measured_nm']), wait_us=waits[order],
            sample_end_after_target_write_us=end - write,
            sample_adc=float(adc), sample_voltage_v=float(adc_code_to_voltage(adc)),
            target_adc=target, target_voltage_v=float(adc_code_to_voltage(target)),
            percentage_of_final=percentage,
            deviation_percent=None if percentage is None else percentage - 100.,
            within_10pct=(90. <= percentage <= 110.) if percentage is not None and not reasons else None,
            reference_usable=not reasons, reasons=reasons, reference_metric=metric,
        ))
    return dict(schema=SCHEMA, machine_id=report['machine_id'],
                route_key=[[r['index'], list(r['codes'])] for r in route],
                signal_channel=channel, feedback_selector=selector,
                fast_dac_writes=fast, single_sample=single, wait_mode=report['wait_mode'],
                point_delays_us=list(waits), measured_sequence=frame['sequence'], points=points,
                stability_verified=False, absolute_wavelength_accuracy_verified=False)


def capture_current_wait_levels(device, plan, output, *, point_delays_us, wait_mode,
                                channel, selector, fast=False, should_stop=lambda: False,
                                on_progress=lambda text: None):
    """Capture one post-warmup cyclic frame, then held-target reference levels."""
    import app_JDSU as app
    from benchmark_stress_realtime import _safe_disarm, _close_shutter
    from capture_ch1_teacher_spectrum import save_checkpoint
    from capture_temporary_half_height_transients import capture_target_level
    from machine_profile import get_runtime_machine_id
    from temporary_test_mode import _feedback_pair

    output = Path(output)
    if output.exists() or output.with_suffix('.scan.json').exists():
        raise FileExistsError(output)
    waits = list(point_delays_us)
    command(plan['rows'], SCAN_CYCLES, 1, point_delays_us=waits, adc_channel=channel,
            feedback_selector=selector, single_sample=True, fast_dac_writes=fast)
    report = dict(schema=SCHEMA, route=plan['rows'], machine_id=get_runtime_machine_id(),
                  signal_channel=channel, feedback_selector=selector, fast_dac_writes=fast,
                  single_sample=True, wait_mode=wait_mode, point_delays_us=waits,
                  voltage_probes=[], complete=False, created_s=time.time())
    old_timeout = device.timeout
    try:
        # Fail on an unwritable destination before any optical command.
        save_checkpoint(output, report)
        if should_stop():
            raise InterruptedError('当前等待测量已取消')
        on_progress('按当前等待采集32帧（前16帧预热），保留最后一帧实际ADC…')
        report['scan_capture'] = execute_device(
            device, plan, output.with_suffix('.scan.json'), cycles=SCAN_CYCLES,
            point_delays_us=waits, adc_channel=channel, feedback_selector=selector,
            single_sample=True, fast_dac_writes=fast, should_stop=should_stop)
        if not all(report['scan_capture'].get(key) for key in ('complete', 'disarm_ack', 'shutter_ack')):
            raise RuntimeError('当前等待循环采集不完整或停光未确认')
        if should_stop():
            raise InterruptedError('当前等待测量已取消')
        device.timeout = .1
        device.write(app.build_work_mode_command(2))
        time.sleep(.12)
        device.reset_input_buffer()
        feedback = app.EqualIntervalWorker(device, (), {}, settle_s=0,
                                           feedback_selectors=_feedback_pair(channel, selector))
        if not feedback._feedback_command_and_ack():
            raise RuntimeError('最终电压预检模拟档位设置未确认')
        tag = int(time.time_ns() & 0x7fffffff) or 1
        for order, row in enumerate(plan['rows']):
            if should_stop():
                raise InterruptedError('当前等待测量已取消')
            tag = (tag + 1) & 0xffffffff or 1
            probe = capture_target_level(device, row, tag, fast_dac_writes=fast)
            probe['route_target_order'] = order
            report['voltage_probes'].append(probe)
            on_progress(f'当前等待测量：最终电压预检 {order + 1}/{len(plan["rows"])}')
            if (order + 1) % 10 == 0:
                save_checkpoint(output, report)
        if should_stop():
            raise InterruptedError('当前等待测量已取消')
        report['complete'] = True
    except BaseException as exc:
        report['error'] = str(exc)
        raise
    finally:
        errors = []
        for name, action in (('exact_disarm_ack', _safe_disarm), ('exact_soa_shutter_ack', _close_shutter)):
            try:
                report[name] = bool(action(device))
            except Exception as exc:
                report[name] = False
                errors.append(str(exc))
        device.timeout = old_timeout
        report['cleanup_errors'] = errors
        shutdown_ok = report.get('exact_disarm_ack') and report.get('exact_soa_shutter_ack')
        if not shutdown_ok:
            report['complete'] = False
        save_checkpoint(output, report)
        if not shutdown_ok:
            raise RuntimeError('当前等待测量后停采或关光未确认，请检查板卡')
    # Validation errors cannot be emitted as successful measurements.
    try:
        report['current_wait_levels'] = analyze_current_wait_levels(report)
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        report['complete'] = False
        report['analysis_error'] = str(exc)
        save_checkpoint(output, report)
        raise
    save_checkpoint(output, report)
    return report
