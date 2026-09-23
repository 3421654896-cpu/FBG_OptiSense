"""Finite USB dense reference acquisition for the standalone temporary mode."""
from pathlib import Path
import time


REFERENCE_RETRY_PLAN = (
    # (stable-window delay, number of preceding forward points to replay)
    (.2, 0),
    (.5, 0),
    (.2, 5),
    (.5, 10),
)
REFERENCE_RECOVERY_HOLD_S = .2
REFERENCE_MAX_SAMPLES = 150
REFERENCE_POINT_TIMEOUT_S = 3.25
SIGNAL_CHANNELS = ('CH0', 'CH1', 'CH2', 'CH3')


def feedback_selectors_for_channel(channel, selector):
    """Return the CH0/CH1 hardware selectors for one displayed ADC channel."""
    channel = int(channel)
    selector = int(selector)
    if channel not in range(4) or selector not in range(4):
        raise ValueError('Invalid signal channel or analog feedback selector')
    if channel == 0:
        return selector, 0
    if channel == 1:
        return 0, selector
    # CH2/CH3 have fixed 2 kOhm hardware feedback.  Keep the two selectable
    # front ends at their safe 2 kOhm state while those channels are acquired.
    return 0, 0


def _select_monitor_channel(row, channel):
    """Attach channel-neutral signal fields to one robust monitor-window row."""
    import numpy as np
    import app_JDSU as app

    channel = int(channel)
    if channel not in range(4):
        raise ValueError('Invalid signal channel')
    samples = row.get('monitor_samples') or []
    if samples:
        median, sigma, drift, settled = app.fullband_accuracy_statistics(samples)
        column = channel + 2
        code = float(median[column])
        noise = float(sigma[column])
        channel_drift = float(drift[column])
        all_stable = bool(settled)
    elif channel == 1 and 'ch1_adc_code' in row:
        # Compatibility for unit fixtures and legacy helper results.
        code = float(row['ch1_adc_code'])
        noise = float(row.get('ch1_sigma_codes', float('nan')))
        channel_drift = float(row.get('ch1_drift_codes', float('nan')))
        all_stable = bool(row.get('stable'))
    elif channel == 1 and 'stable' in row:
        # Lightweight unit/diagnostic fixtures may exercise retry control
        # without carrying an ADC payload. Preserve that legacy contract.
        row.update(
            signal_channel=1,
            signal_channel_name='CH1',
            signal_saturated=bool(row.get('ch1_saturated')),
        )
        return row
    else:
        raise RuntimeError(f'{SIGNAL_CHANNELS[channel]}没有可用ADC监测样本')
    saturated = bool(code >= 4080.0)
    code_to_volt = app.PD_ADC_REFERENCE_V / app.PD_ADC_CODE_COUNT
    row.update(
        signal_channel=channel,
        signal_channel_name=SIGNAL_CHANNELS[channel],
        signal_adc_code=code,
        signal_voltage_v=code * code_to_volt,
        signal_sigma_codes=noise,
        signal_drift_codes=channel_drift,
        signal_saturated=saturated,
        stable=bool(all_stable and not saturated),
        window_stable=bool(all_stable and not saturated),
        all_monitor_channels_stable=all_stable,
    )
    return row


def _equal_interval_row(worker, point, channel, scan_id):
    """Acquire one point through EqualIntervalWorker's exact single-frame path."""
    raw = worker._acquire_point(scan_id, point)
    if raw is None:
        raise InterruptedError('等间隔光谱采集已取消')
    channel = int(channel)
    code = float(raw['adc_codes'][channel])
    saturated = channel in tuple(raw.get('saturated_channels', ())) or code >= 4080.0
    code_to_volt = float(raw['voltages'][channel] / code) if code else 0.0
    return {
        'index': int(point.index),
        'target_wavelength_nm': float(point.target_nm),
        'measured_wavelength_nm': float(point.measured_nm),
        'dac_codes': [int(value) for value in point.codes],
        'signal_channel': channel,
        'signal_channel_name': SIGNAL_CHANNELS[channel],
        'signal_adc_code': code,
        'signal_voltage_v': code * code_to_volt,
        'signal_sigma_codes': float('nan'),
        'signal_drift_codes': float('nan'),
        'signal_saturated': bool(saturated),
        'stable': None,
        'window_stable': None,
        'all_monitor_channels_stable': None,
        'single_sample': True,
        'stability_not_evaluated': True,
        'sample_count': 1,
        'settle_s': float(raw.get('settle_s', worker.settle_s)),
        'elapsed_s': float(raw.get('elapsed_s', 0.0)),
        'adc_codes': list(raw['adc_codes']),
        'monitor_codes': list(raw.get('monitor_codes', ())),
        'adc_acquisition_method': 'EqualIntervalWorker.single_fresh_frame',
        'attempts': [],
    }


def _replay_forward_approach(worker, point, count, *, should_stop):
    """Re-enter a drifting point through its preceding forward path.

    A source can occasionally remain on a wandering optical branch while the
    same DAC tuple is held.  Waiting longer then keeps observing that branch.
    Replaying a short, already calibrated forward lead-in gives the laser the
    same approach history as the dense scan without weakening the stability
    limits or accepting a bad point.
    """
    from capture_ch1_teacher_spectrum import command_with_link_retry

    try:
        sequence = tuple(worker.points)
    except (AttributeError, TypeError):
        sequence = ()
    position = next((i for i, candidate in enumerate(sequence)
                     if candidate is point), None)
    if position is None:
        wanted_index = getattr(point, 'index', None)
        wanted_codes = tuple(getattr(point, 'codes', ()))
        position = next((i for i, candidate in enumerate(sequence)
                         if getattr(candidate, 'index', None) == wanted_index
                         and tuple(getattr(candidate, 'codes', ())) == wanted_codes), None)

    replayed = []
    guards = getattr(worker, 'guards', {})
    if position is not None:
        start = max(0, position - int(count))
        for predecessor in sequence[start:position]:
            if should_stop():
                raise InterruptedError('参考采集已取消')
            guard = guards.get(predecessor.index) if isinstance(guards, dict) else None
            if guard is not None and not worker._apply_guard(guard):
                raise RuntimeError(f'重建正向路径时第{predecessor.index}点过渡保护被中止')
            if command_with_link_retry(worker, predecessor.codes) is None:
                raise RuntimeError('重建正向路径时扫描已中止')
            if not worker._interruptible_wait(REFERENCE_RECOVERY_HOLD_S):
                raise InterruptedError('参考采集已取消')
            replayed.append(int(predecessor.index))

    # The normal scan already applied this guard before the first attempt.  A
    # recovery retry must apply it again before rewriting the guarded target.
    target_guard = guards.get(point.index) if isinstance(guards, dict) else None
    if target_guard is not None and not worker._apply_guard(target_guard):
        raise RuntimeError(f'重试第{point.index}点时过渡保护被中止')
    return replayed


def acquire_checked_point(worker, point, *, should_stop=lambda:False, channel=1):
    """Bounded retries with forward-path recovery; never relax the gate."""
    import numpy as np
    import app_JDSU as app
    from capture_ch1_teacher_spectrum import acquire_settled_ch1
    attempts=[]
    for number, (delay, replay_count) in enumerate(REFERENCE_RETRY_PLAN):
        if should_stop():
            raise InterruptedError('参考采集已取消')
        replayed = []
        if replay_count:
            replayed = _replay_forward_approach(
                worker, point, replay_count, should_stop=should_stop)
        apply_codes = number == 0 or bool(replay_count)
        row=acquire_settled_ch1(worker,point,delay,retain_monitor_samples=True,
                                apply_codes=apply_codes,
                                max_samples=REFERENCE_MAX_SAMPLES,
                                point_timeout_s=REFERENCE_POINT_TIMEOUT_S)
        row = _select_monitor_channel(row, channel)
        row['retry_strategy'] = ('initial_write' if number == 0 else
                                 'hold_same_codes' if not replay_count else
                                 'replay_forward_predecessors')
        row['replayed_indices'] = replayed
        samples=row.get('monitor_samples')
        if samples:
            median,sigma,drift,_=app.fullband_accuracy_statistics(samples)
            noise_limit=np.maximum(app.FULLBAND_ACCURACY_MAD_FLOOR_CODES,.003*np.maximum(64.,median))
            drift_limit=np.maximum(app.FULLBAND_ACCURACY_DRIFT_FLOOR_CODES,.004*np.maximum(64.,median))
            row['channel_checks']=[dict(channel=name,median_adc=float(median[i]),
                noise_adc=float(sigma[i]),noise_limit_adc=float(noise_limit[i]),
                drift_adc=float(drift[i]),drift_limit_adc=float(drift_limit[i]),
                passed=bool(sigma[i]<=noise_limit[i] and drift[i]<=drift_limit[i]))
                for i,name in enumerate(('PDT','PDR','CH0','CH1','CH2','CH3'))]
            row['failed_channels']=[c['channel'] for c in row['channel_checks'] if not c['passed']]
        attempts.append(row)
        if row['stable'] or row.get('signal_saturated'):
            break
    return dict(row,attempts=attempts,optical_stability_verified=False,
                retry_method='hold_then_replay_forward_predecessors_without_relaxing_gate')


def acquire_reference(device, output, *, on_progress=lambda n, total: None,
                      on_row=lambda row, n, total: None,
                      should_stop=lambda: False, indices=None, selected_rows=None, feedback_selector=2,
                      require_stable=True, continue_on_unstable=False,
                      allow_partial_stop=False, table_base_dir=None,
                      signal_channel=1, dense_indices=None,
                      acquisition_method='stable_window', settle_s=.2):
    import app_JDSU as app
    if type(feedback_selector) is not int or feedback_selector not in range(4):
        raise ValueError('Invalid analog feedback selector')
    if type(signal_channel) is not int or signal_channel not in range(4):
        raise ValueError('Invalid signal channel')
    if signal_channel >= 2:
        feedback_selector = 0
    if acquisition_method not in ('stable_window', 'equal_interval_single'):
        raise ValueError('Invalid reference acquisition method')
    if acquisition_method == 'equal_interval_single' and (
            not isinstance(settle_s, (int, float)) or not 0 <= float(settle_s) <= 10):
        raise ValueError('Equal-interval settle time must be between 0 and 10 seconds')
    from benchmark_stress_realtime import _safe_disarm, _close_shutter
    from capture_ch1_teacher_spectrum import acquire_settled_ch1, establish_forward_state, save_checkpoint
    points = app.load_fullband_accuracy_table(table_base_dir)
    guards = app.load_fullband_transition_guards(points, table_base_dir)
    if selected_rows is not None:
        if indices is not None:
            raise ValueError('Specify selected_rows or indices, not both')
        from temporary_test_mode import command
        command(selected_rows,32,1, adc_channel=signal_channel,
                feedback_selector=feedback_selector)
        indices = [r['index'] for r in selected_rows]
        points = [app.FullbandAccuracyPoint(index=r['index'],target_nm=r['target_nm'],
                  measured_nm=r['measured_nm'],codes=tuple(r['codes'])) for r in selected_rows]
        guards = {}
    if dense_indices is not None:
        if indices is not None or selected_rows is not None:
            raise ValueError('dense_indices cannot be combined with sparse selection')
        dense_indices = list(dense_indices)
        if (len(dense_indices) < 5 or len(dense_indices) > len(points)
                or any(type(i) is not int or not 0 <= i < len(points)
                       for i in dense_indices)
                or any(right <= left for left, right in zip(dense_indices, dense_indices[1:]))):
            raise ValueError('Dense indices must be 5～2001 unique increasing table indices')
        points = [points[i] for i in dense_indices]
    if indices is not None:
        if (len(indices) not in range(5, 46, 5)
                or any(type(i) is not int or not 0 <= i <= 2000 for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError('Selected reference requires 5～45 unique physical indices in five-point groups')
        if selected_rows is None:
            points = [points[i] for i in indices]
        guards = {}  # Selected-only slow reference: no hidden predecessor writes.
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    dense_scan = indices is None
    report = dict(schema='temporary_dense_reference_v1' if dense_scan else 'temporary_selected_reference_v1', rows=[], complete=False,
                  feedback_selector=feedback_selector,
                  ch1_feedback_selector=(feedback_selector if signal_channel == 1 else 0),
                  signal_channel=signal_channel,
                  signal_channel_name=SIGNAL_CHANNELS[signal_channel],
                  acquisition_method=acquisition_method,
                  requested_point_count=len(points),
                  dense_indices=(list(dense_indices) if dense_indices is not None else None),
                  settle_s=float(settle_s), started_s=time.time(),
                  stability_filter_applied=(require_stable and acquisition_method == 'stable_window'))
    feedback_pair = feedback_selectors_for_channel(signal_channel, feedback_selector)
    worker = app.EqualIntervalWorker(device, points, guards, settle_s=float(settle_s),
                                     feedback_selectors=feedback_pair)
    old_timeout = device.timeout
    try:
        device.timeout = .1
        if not _safe_disarm(device) or not _close_shutter(device):
            raise RuntimeError('参考采集前停光未确认')
        device.write(app.build_work_mode_command(2))
        time.sleep(.12)
        device.reset_input_buffer()
        if not worker._feedback_command_and_ack():
            raise RuntimeError('所选模拟反馈确认失败')
        if dense_scan and acquisition_method == 'stable_window':
            establish_forward_state(worker, points, 0)
        for scan_index, point in enumerate(points):
            if should_stop():
                if allow_partial_stop:
                    report['stopped_early'] = True
                    break
                raise InterruptedError('参考采集已取消')
            if scan_index and scan_index % app.EQUAL_INTERVAL_FEEDBACK_VERIFY_EVERY_POINTS == 0:
                if not worker._feedback_command_and_ack():
                    raise RuntimeError('反馈复核失败')
            guard = guards.get(point.index)
            if guard is not None and not worker._apply_guard(guard):
                raise RuntimeError('参考扫描过渡失败')
            if acquisition_method == 'equal_interval_single':
                try:
                    row = _equal_interval_row(worker, point, signal_channel, 1)
                except InterruptedError:
                    if allow_partial_stop:
                        report['stopped_early'] = True
                        break
                    raise
            elif require_stable:
                try:
                    row = acquire_checked_point(worker, point, should_stop=should_stop,
                                                channel=signal_channel)
                except InterruptedError:
                    if allow_partial_stop:
                        report['stopped_early'] = True
                        break
                    raise
            else:
                row = acquire_settled_ch1(worker,point,.2,retain_monitor_samples=True)
                row = _select_monitor_channel(row, signal_channel)
                row['attempts'] = []
            passed = bool(
                not row.get('signal_saturated')
                and (acquisition_method == 'equal_interval_single' or row.get('stable'))
            )
            row['point_passed'] = passed
            if not passed:
                row['failure_reason'] = (
                    f'{SIGNAL_CHANNELS[signal_channel]}饱和' if row.get('signal_saturated') else
                    '波动/漂移未通过：' + ','.join(row.get('failed_channels', []))
                )
            report['rows'].append(row)
            on_row(dict(row), len(report['rows']), len(points))
            if (not passed and require_stable
                    and acquisition_method == 'stable_window'
                    and not continue_on_unstable):
                reason = row['failure_reason']
                raise RuntimeError(f"参考点 {point.index} {reason}；已尝试{len(row['attempts'])}次，未跳过该点；原始六路读数已保存")
            on_progress(len(report['rows']), len(points))
            if len(report['rows']) % 25 == 0:
                save_checkpoint(output, report)
        report['complete'] = len(report['rows']) == len(points)
        report['partial'] = not report['complete']
        report['failed_point_count'] = sum(
            not bool(row.get('point_passed')) for row in report['rows']
        )
        report['all_points_passed'] = bool(
            report['complete'] and not report['failed_point_count']
        )
    except BaseException as exc:
        report['error'] = str(exc)
        raise
    finally:
        errors = []
        for name, action in [('disarm_ack', _safe_disarm), ('shutter_ack', _close_shutter)]:
            try:
                report[name] = bool(action(device))
            except Exception as exc:
                report[name] = False
                errors.append(str(exc))
        device.timeout = old_timeout
        report['cleanup_errors'] = errors
        save_checkpoint(output, report)
        if not report.get('disarm_ack') or not report.get('shutter_ack'):
            raise RuntimeError('参考采集后停光未确认，请检查板卡')
    return report


if __name__ == '__main__':
    import serial
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--selection-only',action='store_true')
    args=parser.parse_args()
    destination = output_path('temporary_test', f'reference_{time.time_ns()}.json')
    print(f'Output: {destination}', flush=True)
    with serial.Serial('COM6', 2000000, timeout=.1, write_timeout=3) as port:
        result = acquire_reference(port, destination,
            require_stable=not args.selection_only,
            should_stop=lambda: destination.with_suffix('.stop').exists(),
            on_progress=lambda n, total: print(f'Reference {n}/{total}', flush=True) if n % 25 == 0 else None)
    print(f"Complete: {result['complete']}; shutter: {result['shutter_ack']}", flush=True)
from runtime_paths import output_path
