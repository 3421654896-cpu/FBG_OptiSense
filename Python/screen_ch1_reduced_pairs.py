"""Screen a whitelisted 18-point candidate with held-source pair transitions only.

No CNC motion, firmware/table writes or model fitting. Every source/target is
an existing bounded installed DAC row. A 100 ms held source and dense six-ADC
diagnostic are NOT the continuous shortened path or the production ADC pair.
The meter reads later held light, not wavelength during the early transient.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time

import serial
import app_JDSU as app
from benchmark_stress_realtime import _safe_disarm, _close_shutter
from capture_ch1_rc_diagnostic import load_installed_rows
from capture_certified_ch1_transients import capture, decode, capture_stable_tail
from capture_ch1_teacher_spectrum import save_checkpoint, sha256_file
from laser_power_calibration import AQ6150B, Reading
from probe_ch1_g4_candidates import optical_review

HERE = Path(__file__).resolve().parent
REDUCED_INDICES = tuple(segment*5 + local for segment in range(9) for local in (1, 3))
CANDIDATE_VARIANTS = {
    'two_flanks': REDUCED_INDICES,
    'g4_g8_peak_right': tuple({16: 17, 36: 37}.get(index, index) for index in REDUCED_INDICES),
    # Reversing G8 also changes the predecessor of G9's first point (38 -> 36).
    'g8_right_left': tuple({36: 38, 38: 36}.get(index, index) for index in REDUCED_INDICES),
}
PATH_NAMES = ('adjacent_held', 'reduced18_held')


def candidate_indices(variant):
    if not isinstance(variant, str) or variant not in CANDIDATE_VARIANTS:
        raise ValueError('unknown candidate variant')
    return CANDIDATE_VARIANTS[variant]


def build_cases(rows, gratings=tuple(range(1, 10)), variant='two_flanks'):
    indices = candidate_indices(variant)
    if (not gratings or len(set(gratings)) != len(gratings)
            or any(type(g) is not int or not 1 <= g <= 9 for g in gratings)):
        raise ValueError('one to nine distinct grating IDs required')
    cases = []
    for order, target in enumerate(indices):
        grating = 1 + target//5
        if grating not in gratings:
            continue
        cases.append(dict(target=target, grating=grating, target_nm=rows[target]['target_wavelength_nm'],
                          target_codes=list(rows[target]['codes']),
                          sources=dict(adjacent_held=(target-1)%45,
                                       reduced18_held=indices[order-1])))
    return cases


def execute(output, *, port='COM6', gpib='GPIB0::7::INSTR', repeats=2,
            gratings=tuple(range(1, 10)), variant='two_flanks'):
    output = Path(output)
    if output.exists():
        raise FileExistsError('screen output already exists')
    if type(repeats) is not int or not 1 <= repeats <= 2:
        raise ValueError('one or two repeats only')
    indices = candidate_indices(variant)
    rows, hidden = load_installed_rows()
    cases = build_cases(rows, gratings, variant)
    report = dict(schema='ch1_reduced_edge_pair_screen_v1', scope=__doc__, cases=cases,
                  repeats=repeats, selector=2, channels=['CH0','CH1','CH2','CH3','PDT','PDR'],
                  candidate_variant=variant, candidate_indices=indices, installed_rows=rows,
                  hidden_codes=list(hidden), common_anchor_index=44, anchor_hold_s=.3,
                  source_hold_s=.1, captures=[], complete=False, training_eligible=False,
                  continuous_path_qualified=False, physical_15hz_verified=False,
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  table_sha256=sha256_file(HERE/'mode_tables_from_fullband_2001.json'))
    output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(output, report)
    meter = None
    with serial.Serial(port, 2_000_000, timeout=.02, write_timeout=3) as device:
        try:
            device.dtr = True
            if not _safe_disarm(device) or not _close_shutter(device):
                raise RuntimeError('initial disarm/shutter not acknowledged')
            worker = app.EqualIntervalWorker(device, (), {}, settle_s=.3, feedback_selectors=(0, 2))
            if not worker._feedback_command_and_ack():
                raise RuntimeError('feedback IO ACK not confirmed')
            meter = AQ6150B(gpib)
            meter.instrument.timeout = 2000
            report['meter_identity'] = meter.identity
            tag = int(time.time_ns() & 0x7fffffff) or 1
            for repeat in range(repeats):
                ordered = cases if repeat == 0 else list(reversed(cases))
                for case in ordered:
                    paths = PATH_NAMES if repeat == 0 else PATH_NAMES[::-1]
                    for path in paths:
                        if worker._command_and_ack(rows[44]['codes']) is None:
                            raise RuntimeError('anchor DAC ACK missing')
                        if not worker._interruptible_wait(.3):
                            raise RuntimeError('screen cancelled')
                        tag = (tag + 1) & 0xffffffff or 1
                        source = case['sources'][path]
                        record = dict(target=case['target'], source=source, path=path, repeat=repeat,
                                      target_codes=case['target_codes'], source_codes=rows[source]['codes'])
                        report['captures'].append(record)
                        record['transient'] = capture(device, source, case['target'], tag)
                        # Never rewrite the target between the transient and either tail.
                        record['before_meter'] = capture_stable_tail(worker)
                        record['meter_start_host_ns'] = time.monotonic_ns()
                        reading = meter.measure(case['target_nm'], select_main_peak=True)
                        record['meter_end_host_ns'] = time.monotonic_ns()
                        record['meter'] = asdict(reading)
                        record['review'] = optical_review(reading, case['target_nm'])
                        record['after_meter'] = capture_stable_tail(worker)
                        save_checkpoint(output, report)
                        print(f"trial={len(report['captures'])}/{len(cases)*2*repeats} "
                              f"p{case['target']} {path} rep={repeat+1} "
                              f"ADC={record['before_meter']['ch1_adc_code']}->{record['after_meter']['ch1_adc_code']} "
                              f"wave={reading.wavelength_nm:.6f} peaks={reading.peak_count}", flush=True)
            report['complete'] = len(report['captures']) == len(cases)*2*repeats
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
            if meter is not None:
                try:
                    meter.close()
                except Exception as exc:
                    errors.append(f'meter: {exc}')
            report['safety_cleanup'] = dict(exact_disarm_ack=disarmed, exact_soa_shutter_ack=shuttered,
                                           errors=errors)
            report['finished_utc'] = datetime.now(timezone.utc).isoformat()
            save_checkpoint(output, report)
            print(f'disarm_confirmed={int(disarmed)} soa_shutter_confirmed={int(shuttered)}', flush=True)
            if not disarmed or not shuttered:
                raise RuntimeError('safe shutdown not confirmed')
    return report


def verified_tail(tail):
    wires = tail['monitor_wire_hex']
    rows = [app.decode_single_value_monitor_frame(bytes.fromhex(raw)) for raw in wires]
    if not rows or any(row is None for row in rows) or [list(row) for row in rows] != tail['monitor_adc']:
        raise ValueError('tail does not match raw monitor packets')
    median, sigma, drift, stable = app.fullband_accuracy_statistics(rows)
    if not math.isfinite(tail['ch1_adc_code']) or float(median[3]) != tail['ch1_adc_code']:
        raise ValueError('tail median differs from original packets')
    if bool(stable and median[3] < 4080) != tail['stable']:
        raise ValueError('tail stability flag differs')
    return float(median[3])


def summarize(report):
    if report.get('schema') != 'ch1_reduced_edge_pair_screen_v1' or not report.get('complete'):
        raise ValueError('complete reduced pair screen required')
    repeats = report['repeats']
    if type(repeats) is not int or not 1 <= repeats <= 2:
        raise ValueError('invalid repeat count')
    rows = report['installed_rows']
    # Existing v1 captures predate named variants and used the two-flank list.
    variant = report.get('candidate_variant', 'two_flanks')
    indices = candidate_indices(variant)
    if 'candidate_indices' in report and tuple(report['candidate_indices']) != indices:
        raise ValueError('candidate indices do not match whitelisted variant')
    cases = {case['target']: case for case in report['cases']}
    expected_cases = build_cases(rows, tuple(sorted({case['grating'] for case in cases.values()})), variant)
    if len(cases) != len(report['cases']) or list(cases.values()) != expected_cases:
        raise ValueError('unexpected candidate/source map')
    expected = {(target, path, repeat) for target in cases for path in PATH_NAMES for repeat in range(repeats)}
    seen = set()
    tags = set()
    audited = []
    for row in report['captures']:
        key = row['target'], row['path'], row['repeat']
        if key not in expected or key in seen:
            raise ValueError('missing, unexpected or duplicate trial')
        seen.add(key)
        case = cases[row['target']]
        if (row['source'] != case['sources'][row['path']] or row['target_codes'] != case['target_codes']
                or row['source_codes'] != rows[row['source']]['codes']):
            raise ValueError('trial codes/source mismatch')
        transient = row['transient']
        if transient['request_tag'] in tags:
            raise ValueError('duplicate request tag')
        tags.add(transient['request_tag'])
        restored = decode(bytes.fromhex(transient['wire_hex']), row['source'], row['target'], transient['request_tag'])
        if restored != transient:
            raise ValueError('transient differs from original packet')
        before, after = (verified_tail(row[name]) for name in ('before_meter', 'after_meter'))
        reading = Reading(**row['meter'])
        review = optical_review(reading, case['target_nm'])
        if row['review'] != review or row['meter_end_host_ns'] <= row['meter_start_host_ns']:
            raise ValueError('invalid optical reading/timing')
        starts = transient['sample_start_after_target_write_us']
        values = transient['direct_adc']
        bounded = {}
        for cutoff in (650, 1000, 2000, 5000):
            # The next ADC-group start bounds completion of the previous group.
            # A group merely starting before the cutoff is NOT causal evidence.
            eligible = [index for index in range(len(starts)-1) if starts[index+1] <= cutoff]
            bounded[str(cutoff)] = None if not eligible else values[eligible[-1]][1]
        audited.append(dict(target=row['target'], path=row['path'], repeat=row['repeat'],
                            before_adc=before, after_adc=after,
                            tail_change_codes=after-before,
                            before_window_stable=row['before_meter']['stable'],
                            after_window_stable=row['after_meter']['stable'],
                            wavelength_nm=reading.wavelength_nm, power_mw=reading.power_mw,
                            target_error_pm=review['error_vs_target_pm'],
                            secondary_peak=review['detectable_secondary_peak'],
                            existing_20db_screen=review['meets_existing_20db_screen'],
                            update_us=transient['path_update_us'],
                            six_adc_group_completed_by_cutoff_codes=bounded))
    if seen != expected:
        raise ValueError('incomplete trial coverage')
    target_summary = []
    for target in cases:
        selected = [row for row in audited if row['target'] == target]
        by_path = {}
        for path in PATH_NAMES:
            subset = [row for row in selected if row['path'] == path]
            by_path[path] = dict(
                wavelength_mean_nm=statistics.fmean(row['wavelength_nm'] for row in subset),
                before_adc_mean=statistics.fmean(row['before_adc'] for row in subset),
                before_adc_span=max(row['before_adc'] for row in subset)-min(row['before_adc'] for row in subset),
                max_abs_tail_change_codes=max(abs(row['tail_change_codes']) for row in subset),
                max_target_error_pm=max(abs(row['target_error_pm']) for row in subset),
                minimum_power_mw=min(row['power_mw'] for row in subset))
        target_summary.append(dict(target=target, grating=cases[target]['grating'], by_path=by_path,
                                   reduced_minus_adjacent_wave_pm=1000*(by_path['reduced18_held']['wavelength_mean_nm']
                                       - by_path['adjacent_held']['wavelength_mean_nm']),
                                   reduced_minus_adjacent_adc=by_path['reduced18_held']['before_adc_mean']
                                       - by_path['adjacent_held']['before_adc_mean']))
    errors = {}
    for path in PATH_NAMES:
        errors[path] = {}
        for cutoff in ('650', '1000', '2000', '5000'):
            available = [row for row in audited if row['path'] == path
                         and row['six_adc_group_completed_by_cutoff_codes'][cutoff] is not None]
            differences = [row['six_adc_group_completed_by_cutoff_codes'][cutoff] - row['before_adc']
                           for row in available]
            errors[path][cutoff] = dict(samples=len(differences),
                mean_absolute_error_codes=statistics.fmean(abs(d) for d in differences) if differences else None,
                maximum_absolute_error_codes=max(map(abs, differences)) if differences else None,
                descriptive_within_3codes_or_10pct=sum(abs(d) <= max(3., .1*row['before_adc'])
                                                      for d, row in zip(differences, available)))
    return dict(schema='ch1_reduced_pair_screen_analysis_v1', candidate_variant=variant,
                candidate_indices=indices, trials=audited, targets=target_summary,
                early_vs_later_monitor_diagnostic=errors,
                diagnostic_tolerance_is_not_pressure_accuracy=True,
                all_nine_gratings_covered=len(cases)==18, continuous_path_qualified=False,
                training_eligible=False, physical_15hz_verified=False,
                scope=__doc__)


def plot_summary(result, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    segments = sorted({row['target']//5 for row in result['trials']})
    columns = min(3, len(segments))
    rows = math.ceil(len(segments)/columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.5*columns, 3.4*rows+1.4),
                             sharex=True, squeeze=False)
    for position, (segment, axis) in enumerate(zip(segments, axes.flat)):
        axis.axhline(0, color='#788493', lw=.8)
        wavelength_errors = []
        for row in result['trials']:
            if row['target']//5 != segment or row['path'] != 'reduced18_held':
                continue
            wavelength_errors.append(abs(row['target_error_pm']))
            x, y = [], []
            for cutoff, code in row['six_adc_group_completed_by_cutoff_codes'].items():
                if code is not None:
                    x.append(int(cutoff)/1000)
                    y.append(code-row['before_adc'])
            role, color = {1: ('Left flank', '#286CCA'), 2: ('Peak', '#29856A'),
                           3: ('Right flank', '#C76722')}[row['target']%5]
            axis.plot(x, y, marker='o' if row['repeat']==0 else 's', ms=4, lw=1.2,
                      color=color, alpha=1 if row['repeat']==0 else .55,
                      label=f"{role}, repeat {row['repeat']+1}")
        axis.set_title(f'G{segment+1} | max later wavelength error {max(wavelength_errors):.2f} pm')
        axis.set_xscale('log')
        axis.set_xticks([.65,1,2,5], ['0.65','1','2','5'])
        axis.minorticks_off()
        axis.grid(alpha=.18)
        if position%columns==0:
            axis.set_ylabel('Early minus later held CH1 (codes)')
        if position >= len(segments)-columns:
            axis.set_xlabel('ADC group completed by cutoff (ms)')
    for unused in list(axes.flat)[len(segments):]:
        unused.set_visible(False)
    legend = {}
    for axis in axes.flat:
        handles, labels = axis.get_legend_handles_labels()
        legend.update(zip(labels, handles))
    fig.legend(legend.values(), legend.keys(), loc='upper center', ncol=3, bbox_to_anchor=(.5,.89))
    fig.suptitle(f"18-point candidate ({result.get('candidate_variant', 'two_flanks')}): early ADC versus later held monitor\n"
                 '100 ms held-source pair tests; NOT continuous-path or 15 Hz pressure validation', y=.995)
    fig.tight_layout(rect=(0,0,1,.75 if rows == 1 else .88))
    fig.savefig(output, dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--gratings', type=int, nargs='+', default=list(range(1, 10)))
    parser.add_argument('--variant', choices=tuple(CANDIDATE_VARIANTS), default='two_flanks')
    parser.add_argument('--analyze', type=Path)
    parser.add_argument('--plot', type=Path)
    args = parser.parse_args()
    if args.analyze:
        if args.output.exists() or (args.plot is not None and args.plot.exists()):
            raise FileExistsError(args.output)
        result = summarize(json.loads(args.analyze.read_text('utf-8')))
        result['source_sha256'] = sha256_file(args.analyze)
        result['analyzer_sha256'] = sha256_file(Path(__file__))
        save_checkpoint(args.output, result)
        if args.plot is not None:
            plot_summary(result, args.plot)
        print(f"redecoded={len(result['trials'])} all_nine={result['all_nine_gratings_covered']}")
    else:
        execute(args.output, repeats=args.repeats, gratings=tuple(args.gratings), variant=args.variant)
