"""Bounded static G4 optical screening, never a production-path certificate.

Uses only existing safe table codes. No CNC motion, firmware/table installation,
PID changes or model training. The held source and host-paced target are NOT the
normal sparse prefix and cannot qualify fast transient accuracy or bandwidth.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import math
import time

import serial
import app_JDSU as app
from benchmark_stress_realtime import _safe_disarm, _close_shutter
from capture_ch1_rc_diagnostic import load_installed_rows
from capture_ch1_teacher_spectrum import acquire_settled_ch1, save_checkpoint, sha256_file
from laser_dac_safety import validate_pi11210_codes
from laser_power_calibration import AQ6150B

HERE = Path(__file__).resolve().parent
# Slots are comparison roles only: no replacement is installed by this program.
NEIGHBORS = ((749, 16), (751, 16), (753, 17), (759, 18), (763, 18), (765, 18))


def build_cases(rows, fullband):
    cases = []
    for slot in (16, 17, 18):
        row = rows[slot]
        cases.append(dict(name=f"installed_{slot}", slot=slot,
                          index=row['fullband_index'], target_nm=row['target_wavelength_nm'],
                          measured_nm=row['measured_wavelength_nm'], codes=row['codes']))
    for index, slot in NEIGHBORS:
        point = fullband[index]
        cases.append(dict(name=f"fullband_{index}", slot=slot, index=index,
                          target_nm=point.target_nm, measured_nm=point.measured_nm,
                          codes=list(point.codes)))
    for case in cases:
        case['sources'] = {
            'held_installed_previous': list(rows[case['slot']-1]['codes']),
            'held_fullband_previous': list(fullband[case['index']-1].codes),
        }
        for codes in (case['codes'], *case['sources'].values()):
            validate_pi11210_codes(codes)
    validate_pi11210_codes(rows[15]['codes'])
    return cases


def checked_command(worker, codes):
    validate_pi11210_codes(codes)
    if worker._command_and_ack(codes) is None:
        raise RuntimeError('exact DAC ACK not confirmed')


def optical_review(reading, target_nm):
    # No inferred/infinite SMSR when the meter reports only one detectable peak.
    smsr = reading.side_mode_suppression_db
    if (not math.isfinite(reading.wavelength_nm) or not math.isfinite(reading.power_mw)
            or reading.power_mw <= 0 or reading.peak_count < 1
            or (smsr is not None and not math.isfinite(smsr))):
        raise ValueError('invalid wavelength-meter reading')
    return dict(error_vs_target_pm=(reading.wavelength_nm-target_nm)*1000,
                detectable_secondary_peak=reading.peak_count > 1,
                meets_existing_20db_screen=(reading.peak_count == 1 or
                                            (smsr is not None and smsr >= 20)),
                fast_path_qualified=False)


def summarize_candidates(report):
    """Descriptive repeat/path spreads; does not invent a selection certificate."""
    if report['schema'] != 'ch1_g4_static_candidate_screen_v1':
        raise ValueError('wrong candidate probe schema')
    cases = {case['name']: case for case in report['cases']}
    seen = set()
    groups = {name: [] for name in cases}
    for row in report['captures']:
        name = row['case_name']
        key = (name, row['repeat'], row['source_name'])
        case = cases[name]
        if (key in seen or row['source_name'] not in case['sources']
                or not 0 <= row['repeat'] < report['repeats']
                or row['target_codes'] != case['codes']
                or row['source_codes'] != case['sources'][row['source_name']]):
            raise ValueError('duplicate or mismatched candidate trial')
        seen.add(key)
        groups[name].append(row)
    expected = sum(len(case['sources']) for case in cases.values()) * report['repeats']
    if report['complete'] and len(seen) != expected:
        raise ValueError('complete candidate probe is missing trials')
    results = []
    for name, records in groups.items():
        if not records:
            continue
        waves = [r['meter']['wavelength_nm'] for r in records]
        powers = [r['meter']['power_mw'] for r in records]
        adc = [r['adc']['ch1_adc_code'] for r in records]
        results.append(dict(name=name, trials=len(records),
                            stable_windows=sum(r['adc']['stable'] for r in records),
                            target_nm=cases[name]['target_nm'],
                            wavelength_min_nm=min(waves), wavelength_max_nm=max(waves),
                            wavelength_span_pm=(max(waves)-min(waves))*1000,
                            maximum_target_error_pm=max(abs(w-cases[name]['target_nm']) for w in waves)*1000,
                            power_min_mw=min(powers), power_max_mw=max(powers),
                            adc_min_codes=min(adc), adc_max_codes=max(adc),
                            adc_span_codes=max(adc)-min(adc),
                            secondary_peak_trials=sum(r['meter']['peak_count'] > 1 for r in records),
                            meter_20db_screen_trials=sum(r['review']['meets_existing_20db_screen'] for r in records),
                            fast_path_qualified=False))
    return results


def execute(output, *, port='COM6', gpib='GPIB0::7::INSTR', repeats=2):
    output = Path(output)
    if output.exists():
        raise FileExistsError('refusing to overwrite a previous probe')
    if type(repeats) is not int or not 1 <= repeats <= 2:
        raise ValueError('bounded screen allows one or two repeats')
    rows, _ = load_installed_rows()
    cases = build_cases(rows, app.load_fullband_accuracy_table())
    report = dict(schema='ch1_g4_static_candidate_screen_v1', scope=__doc__,
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  cases=cases, captures=[], complete=False, repeats=repeats,
                  training_eligible=False, firmware_or_table_modified=False,
                  fast_path_qualified=False, physical_15hz_verified=False,
                  input_hashes={name: sha256_file(HERE/name) for name in (
                      'fullband_equal_power_operational_2001.csv',
                      'mode_tables_from_fullband_2001.json')},
                  settings=dict(ch0_selector=0, ch1_selector=2, digital_gain=1,
                                anchor_hold_s=.3, source_hold_s=.08, target_min_hold_s=.3),
                  path_note='anchor installed row15 -> held source -> target; no normal scan prefix')
    output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(output, report)
    meter = None
    with serial.Serial(port, 2_000_000, timeout=.02, write_timeout=3) as device:
        try:
            device.dtr = True
            _safe_disarm(device)
            worker = app.EqualIntervalWorker(device, (), {}, settle_s=.3,
                                              feedback_selectors=(0, 2))
            if not worker._feedback_command_and_ack():
                raise RuntimeError('feedback IO ACK not confirmed')
            meter = AQ6150B(gpib)
            report['meter_identity'] = meter.identity
            for repeat in range(repeats):
                trial_cases = cases if repeat == 0 else list(reversed(cases))
                for case in trial_cases:
                    sources = list(case['sources'].items())
                    if repeat:
                        sources.reverse()
                    for source_name, source_codes in sources:
                        checked_command(worker, rows[15]['codes'])
                        if not worker._interruptible_wait(.3):
                            raise RuntimeError('probe interrupted')
                        checked_command(worker, source_codes)
                        if not worker._interruptible_wait(.08):
                            raise RuntimeError('probe interrupted')
                        point = app.FullbandAccuracyPoint(case['index'], case['target_nm'],
                                                           case['measured_nm'], tuple(case['codes']))
                        adc = acquire_settled_ch1(worker, point, .3, retain_monitor_samples=True)
                        # No DAC rewrite between the ADC window and fresh meter trigger.
                        triggered_ns = time.monotonic_ns()
                        reading = meter.measure(case['target_nm'], select_main_peak=True)
                        record = dict(case_name=case['name'], repeat=repeat, source_name=source_name,
                                      target_codes=case['codes'], source_codes=source_codes,
                                      adc=adc, meter=asdict(reading),
                                      meter_trigger_host_monotonic_ns=triggered_ns,
                                      meter_received_host_monotonic_ns=time.monotonic_ns(),
                                      review=optical_review(reading, case['target_nm']))
                        report['captures'].append(record)
                        save_checkpoint(output, report)
                        print(f"{case['name']} {source_name} rep={repeat+1} "
                              f"ADC={adc['ch1_adc_code']} stable={adc['stable']} "
                              f"wave={reading.wavelength_nm:.6f} power={reading.power_mw:.3f} "
                              f"peaks={reading.peak_count} SMSR={reading.side_mode_suppression_db}", flush=True)
            report['complete'] = len(report['captures']) == len(cases)*2*repeats
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
            if meter is not None:
                try:
                    meter.close()
                except Exception as exc:
                    errors.append(f'meter close: {exc}')
            report['safety_cleanup'] = dict(exact_disarm_ack=disarmed,
                                           exact_soa_shutter_ack=shuttered, errors=errors)
            report['finished_utc'] = datetime.now(timezone.utc).isoformat()
            save_checkpoint(output, report)
            print(f'disarm_confirmed={int(disarmed)} soa_shutter_confirmed={int(shuttered)}', flush=True)
            if not disarmed or not shuttered:
                raise RuntimeError('safe shutdown not confirmed')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--port', default='COM6')
    parser.add_argument('--gpib', default='GPIB0::7::INSTR')
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    execute(args.output, port=args.port, gpib=args.gpib, repeats=args.repeats)
