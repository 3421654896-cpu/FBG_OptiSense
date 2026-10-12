"""Concurrent fixed-DAC ADC and triggered wavemeter diagnostic, without CNC.

USB monitor frames have no ADC timestamp, sequence or CRC. Host batch arrival
times are NOT ADC conversion times. Meter trigger/return times bound a query,
not a known integration aperture. This is not a physical bandwidth benchmark,
pressure/area training set, or a normal sparse-path teacher certificate.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import serial
import app_JDSU as app
from benchmark_stress_realtime import _safe_disarm, _close_shutter
from capture_ch1_rc_diagnostic import load_installed_rows
from capture_ch1_teacher_spectrum import save_checkpoint, sha256_file
from laser_power_calibration import AQ6150B
from probe_ch1_g4_candidates import checked_command, optical_review

HERE = Path(__file__).resolve().parent
GAIN_KOHM = {1: 40, 2: 5}


def timed_measure(meter, target_nm, origin_ns):
    start = time.monotonic_ns()
    reading = meter.measure(target_nm, select_main_peak=True)
    end = time.monotonic_ns()
    optical_review(reading, target_nm)  # refuse invalid/nonfinite observations
    return dict(start_s=(start-origin_ns)/1e9, end_s=(end-origin_ns)/1e9,
                reading=asdict(reading))


def collect_hold(worker, meter, executor, record, duration_s):
    """Read-only on the laser port throughout the hold; meter uses another link."""
    origin_ns = time.monotonic_ns()
    record.update(origin_host_monotonic_ns=origin_ns, adc=[], meter=[], complete=False,
                  adc_acquisition_method='checkRT/ADC_Write_Read_Stable',
                  adc_samples_are_uniform_raw_stream=False)
    future = None
    next_meter = .3
    last_adc_s = 0.
    while True:
        before_s = (time.monotonic_ns()-origin_ns)/1e9
        frames = worker._read_available_frames()
        after_s = (time.monotonic_ns()-origin_ns)/1e9
        for frame in frames:
            codes = app.decode_single_value_monitor_frame(frame)
            if codes is None:
                continue
            if any(not 0 <= x <= 4095 for x in codes):
                raise ValueError('invalid direct monitor ADC')
            record['adc'].append(dict(read_started_s=before_s, received_s=after_s,
                                      codes=list(codes), wire_hex=frame.hex()))
            last_adc_s = after_s
        if after_s-last_adc_s > 1.0:
            raise TimeoutError('fixed-hold ADC stopped for over one second')
        if len(record['adc']) > 20000:
            raise RuntimeError('bounded fixed-hold record overflow')
        if future is not None and future.done():
            record['meter'].append(future.result())
            future = None
        if after_s >= duration_s:
            if future is None:
                break
            if after_s > duration_s+6:
                raise TimeoutError('meter did not finish bounded final query')
        elif future is None and after_s >= next_meter:
            future = executor.submit(timed_measure, meter, record['target_nm'], origin_ns)
            next_meter = after_s+1.0
        time.sleep(.001)
    if len(record['adc']) < 50 or len(record['meter']) < 3:
        raise RuntimeError('insufficient continuous hold observations')
    record.update(complete=True, elapsed_s=after_s, dac_rewritten_during_hold=False)


def summarize_hold(record):
    if not record.get('complete'):
        raise ValueError('incomplete hold cannot be summarized as valid')
    adc = record['adc']
    times = np.array([r['received_s'] for r in adc])
    values = np.array([r['codes'] for r in adc], float)
    if (len(adc) < 50 or values.shape != (len(adc), 6) or np.any(np.diff(times) < 0)
            or not np.all(np.isfinite(times)) or np.any(times < 0)
            or np.any(values < 0) or np.any(values > 4095)):
        raise ValueError('invalid monitor series')
    for row in adc:
        decoded = app.decode_single_value_monitor_frame(bytes.fromhex(row['wire_hex']))
        if decoded is None or list(decoded) != row['codes']:
            raise ValueError('stored ADC contradicts raw monitor frame')
    bins = []
    for start in np.arange(0., 8., .5):
        block = values[(times >= start) & (times < start+.5)]
        if not len(block):
            continue
        bins.append(dict(start_s=float(start), sample_count=len(block),
                         median=np.median(block, axis=0).tolist(),
                         p05=np.percentile(block, 5, axis=0).tolist(),
                         p95=np.percentile(block, 95, axis=0).tolist()))
    previous_end = -1.
    for sample in record['meter']:
        start, end = sample['start_s'], sample['end_s']
        if not (0 <= start < end <= record['elapsed_s']+.05 and start >= previous_end):
            raise ValueError('invalid or overlapping meter interval')
        previous_end = end
        optical_review(SimpleNamespace(**sample['reading']), record['target_nm'])
    waves = [r['reading']['wavelength_nm'] for r in record['meter']]
    powers = [r['reading']['power_mw'] for r in record['meter']]
    if not waves:
        raise ValueError('missing optical measurements')
    early = values[(times >= .3) & (times < .6), 3]
    late = values[(times >= 5) & (times < 8), 3]
    if len(early) < 5 or len(late) < 20:
        raise ValueError('insufficient early/late coverage')
    recent = [b['median'][3] for b in bins if b['start_s'] >= 2.]
    return dict(point=record['point'], repeat=record['repeat'], selector=record['selector'],
                gain_kohm=GAIN_KOHM[record['selector']], frames=len(adc),
                meter_readings=len(waves), elapsed_s=record['elapsed_s'],
                maximum_host_batch_gap_s=float(np.max(np.diff(times))),
                early_0p3_0p6s_ch1_median=float(np.median(early)),
                late_5_8s_ch1_median=float(np.median(late)),
                early_minus_late_codes=float(np.median(early)-np.median(late)),
                ch1_after2s_halfsecond_median_span=float(np.ptp(recent)),
                wavelength_span_pm=(max(waves)-min(waves))*1000,
                power_min_mw=min(powers), power_max_mw=max(powers), bins=bins,
                physical_bandwidth_verified=False, training_eligible=False)


def execute(output, *, port='COM6', gpib='GPIB0::7::INSTR', duration_s=8., repeats=2,
            points=(16, 17, 18)):
    output = Path(output)
    if output.exists():
        raise FileExistsError('refusing to overwrite a prior fixed hold')
    if not 8 <= duration_s <= 20 or type(repeats) is not int or not 1 <= repeats <= 2:
        raise ValueError('hold must be 8..20 seconds and repeats 1..2')
    if len(points) != 3 or len(set(points)) != 3 or any(type(p) is not int or not 1 <= p <= 44 for p in points):
        raise ValueError('exactly three distinct installed points in 1..44 required')
    rows, _ = load_installed_rows()
    settings = [(p, s) for p in points for s in (2, 1)]
    report = dict(schema='ch1_fixed_hold_adc_meter_v1', scope=__doc__, captures=[],
                  created_utc=datetime.now(timezone.utc).isoformat(), complete=False,
                  input_table_sha256=sha256_file(HERE/'mode_tables_from_fullband_2001.json'),
                  duration_s=duration_s, repeats=repeats, settings=settings,
                  training_eligible=False, physical_15hz_verified=False,
                  channels=['PDT','PDR','CH0','CH1','CH2','CH3'],
                  path='installed row15 held 0.3s -> previous installed row held 0.08s -> target held continuously',
                  firmware_or_table_modified=False, machine_motion_commanded=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(output, report)
    meter = None
    executor = ThreadPoolExecutor(max_workers=1)
    with serial.Serial(port, 2_000_000, timeout=.02, write_timeout=3) as device:
        try:
            device.dtr = True
            _safe_disarm(device)
            meter = AQ6150B(gpib)
            meter.instrument.timeout = 2000
            report['meter_identity'] = meter.identity
            for repeat in range(repeats):
                for point, selector in (settings if repeat == 0 else list(reversed(settings))):
                    worker = app.EqualIntervalWorker(device, (), {}, settle_s=.3,
                                                     feedback_selectors=(0, selector))
                    if not worker._feedback_command_and_ack():
                        raise RuntimeError('feedback IO not confirmed')
                    checked_command(worker, rows[15]['codes'])
                    worker._interruptible_wait(.3)
                    checked_command(worker, rows[point-1]['codes'])
                    worker._interruptible_wait(.08)
                    checked_command(worker, rows[point]['codes'])
                    # Purge only once at the start. No more purging or DAC writes
                    # during capture: serial reading continues during GPIB calls.
                    device.reset_input_buffer()
                    worker._rx_buffer.clear()
                    record = dict(point=point, selector=selector, repeat=repeat,
                                  target_nm=rows[point]['target_wavelength_nm'],
                                  target_codes=rows[point]['codes'], source_codes=rows[point-1]['codes'])
                    report['captures'].append(record)
                    collect_hold(worker, meter, executor, record, duration_s)
                    summary = summarize_hold(record)
                    record['summary'] = summary
                    save_checkpoint(output, report)
                    print(f"p{point} {GAIN_KOHM[selector]}k rep={repeat+1} n={summary['frames']} "
                          f"early={summary['early_0p3_0p6s_ch1_median']} late={summary['late_5_8s_ch1_median']} "
                          f"post2s_span={summary['ch1_after2s_halfsecond_median_span']} "
                          f"wave_span_pm={summary['wavelength_span_pm']:.3f}", flush=True)
            report['complete'] = len(report['captures']) == len(settings)*repeats
        except BaseException as exc:
            report['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            disarmed = shuttered = False
            errors = []
            try:
                disarmed = _safe_disarm(device)
            except Exception as exc:
                errors.append(f'disarm: {exc}')
            try:
                shuttered = _close_shutter(device)
            except Exception as exc:
                errors.append(f'shutter: {exc}')
            # Shut the laser before waiting for a possible GPIB query failure.
            executor.shutdown(wait=True, cancel_futures=True)
            if meter is not None:
                try:
                    meter.close()
                except Exception as exc:
                    errors.append(f'meter: {exc}')
            report['safety_cleanup'] = dict(exact_disarm_ack=disarmed, exact_soa_shutter_ack=shuttered, errors=errors)
            report['finished_utc'] = datetime.now(timezone.utc).isoformat()
            save_checkpoint(output, report)
            print(f'disarm_confirmed={int(disarmed)} soa_shutter_confirmed={int(shuttered)}', flush=True)
            if not disarmed or not shuttered:
                raise RuntimeError('safe shutdown not confirmed')
    return report


def execute_dark(output, *, port='COM6'):
    """Separate zero-current shuttered control; no meter or positive DAC output."""
    output = Path(output)
    if output.exists():
        raise FileExistsError('refusing to overwrite a dark control')
    report = dict(schema='ch1_shuttered_dark_control_v1', captures=[], complete=False,
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  training_eligible=False, physical_15hz_verified=False,
                  channels=['PDT','PDR','CH0','CH1','CH2','CH3'],
                  control_note='All DAC codes zero and SOA shutter verified before sampling; no optical-signal response qualification')
    output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(output, report)
    with serial.Serial(port, 2_000_000, timeout=.02, write_timeout=3) as device:
        try:
            device.dtr = True
            _safe_disarm(device)
            if not _close_shutter(device):
                raise RuntimeError('dark control requires verified shutter first')
            for selector in (2, 1):
                worker = app.EqualIntervalWorker(device, (), {}, settle_s=.3,
                                                 feedback_selectors=(0, selector))
                if not worker._feedback_command_and_ack():
                    raise RuntimeError('dark feedback IO not confirmed')
                worker._interruptible_wait(.3)
                device.reset_input_buffer()
                worker._rx_buffer.clear()
                origin = time.monotonic()
                record = dict(selector=selector, gain_kohm=GAIN_KOHM[selector], adc=[])
                report['captures'].append(record)
                last = 0.
                while time.monotonic()-origin < 8.:
                    before = time.monotonic()-origin
                    frames = worker._read_available_frames()
                    after = time.monotonic()-origin
                    for frame in frames:
                        values = app.decode_single_value_monitor_frame(frame)
                        if values is not None:
                            if any(not 0 <= v <= 4095 for v in values):
                                raise ValueError('invalid dark ADC')
                            record['adc'].append(dict(read_started_s=before, received_s=after,
                                                      codes=list(values), wire_hex=frame.hex()))
                            last = after
                    if after-last > 1.0:
                        raise TimeoutError('dark ADC stream missing')
                    time.sleep(.001)
                if len(record['adc']) < 50:
                    raise RuntimeError('insufficient dark samples')
                ch1 = np.array([r['codes'][3] for r in record['adc']])
                record['summary'] = dict(frames=len(ch1), ch1_min=int(ch1.min()),
                                         ch1_max=int(ch1.max()), ch1_median=float(np.median(ch1)))
                save_checkpoint(output, report)
                print(f"dark {GAIN_KOHM[selector]}k {record['summary']}", flush=True)
            report['complete'] = True
        except BaseException as exc:
            report['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            errors = []
            disarmed = shuttered = False
            try:
                disarmed = _safe_disarm(device)
            except Exception as exc:
                errors.append(f'disarm: {exc}')
            try:
                shuttered = _close_shutter(device)
            except Exception as exc:
                errors.append(f'shutter: {exc}')
            report['safety_cleanup'] = dict(exact_disarm_ack=disarmed, exact_soa_shutter_ack=shuttered, errors=errors)
            report['finished_utc'] = datetime.now(timezone.utc).isoformat()
            save_checkpoint(output, report)
            print(f'disarm_confirmed={int(disarmed)} soa_shutter_confirmed={int(shuttered)}', flush=True)
            if not disarmed or not shuttered:
                raise RuntimeError('dark safe shutdown not confirmed')
    return report


def analyze_file(source, output, plot=None):
    source, output = Path(source), Path(output)
    if output.exists() or (plot is not None and Path(plot).exists()):
        raise FileExistsError('analysis output already exists')
    report = json.loads(source.read_text(encoding='utf-8'))
    if report.get('schema') != 'ch1_fixed_hold_adc_meter_v1' or not report.get('complete'):
        raise ValueError('analysis requires a complete fixed-hold capture')
    expected = {(p, s, r) for p, s in report['settings'] for r in range(report['repeats'])}
    actual = [(r['point'], r['selector'], r['repeat']) for r in report['captures']]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError('hold combinations missing or duplicated')
    summary = dict(schema='ch1_fixed_hold_analysis_v1', source=str(source.resolve()),
                   source_sha256=sha256_file(source), holds=[summarize_hold(r) for r in report['captures']],
                   physical_15hz_verified=False, training_eligible=False,
                   limitation=__doc__)
    save_checkpoint(output, summary)
    if plot is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        records = sorted(report['captures'], key=lambda r: (r['repeat'], r['selector'], r['point']))
        nrows = (len(records)+2)//3
        fig, axes = plt.subplots(nrows, 3, figsize=(14, 2.75*nrows), sharex=True, squeeze=False)
        for axis, record in zip(axes.flat, records):
            gain = GAIN_KOHM[record['selector']]
            t = [r['received_s'] for r in record['adc']]
            ch1 = [r['codes'][3]*5/gain for r in record['adc']]
            axis.plot(t, ch1, color='#2879C7', lw=1, label='CH1 (5k-equivalent codes)')
            axis.set_title(f"p{record['point']} / {gain}k / repeat {record['repeat']+1}")
            axis.set_ylabel('CH1 (5k-equivalent)')
            axis.grid(alpha=.2)
            twin = axis.twinx()
            optical = record['meter']
            for i, row in enumerate(optical):
                drift_pm = (row['reading']['wavelength_nm']-record['target_nm'])*1000
                twin.plot([row['start_s'], row['end_s']], [drift_pm, drift_pm],
                          color='#D47A17', lw=2, label='Wavelength; query interval' if i == 0 else None)
            twin.set_ylabel('Error vs target (pm)', color='#D47A17')
        for axis in axes[-1]:
            axis.set_xlabel('Host seconds after target ACK and purge')
        fig.suptitle('Fixed DAC, no commanded motion: CH1 drifts during 8-second holds\n'
                     'Orange bars bound meter queries, not exact integration times. No pressure ground truth.')
        fig.tight_layout(rect=(0, 0, 1, .945))
        fig.savefig(plot, dpi=135)
        plt.close(fig)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--points', type=int, nargs=3, default=[16, 17, 18])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dark-control', action='store_true')
    mode.add_argument('--analyze', type=Path)
    parser.add_argument('--plot', type=Path)
    args = parser.parse_args()
    if args.analyze:
        analyze_file(args.analyze, args.output, args.plot)
    elif args.dark_control:
        execute_dark(args.output)
    else:
        execute(args.output, repeats=args.repeats, points=args.points)
