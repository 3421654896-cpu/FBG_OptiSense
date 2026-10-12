"""Explicitly authorized, stationary-optics wait sweep; no flash or defaults edits.

Runs all-15ms bracketing captures around EACH candidate, with the exact same
route, channel, feedback, ADC and DAC style. Never infer accuracy from FPS.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from serial.tools import list_ports
from thermal_serial import ThermalSerial

from fbg_usb_ota import UsbOtaClient
from machine_profile import get_runtime_machine_id, set_runtime_machine_id
from temporary_test_mode import command, execute_device, validate_plan_payload
from temporary_route_balance_validation import (compare_full_route, compare_pipeline_styles,
                                                 choose_fastest_comparison)


# Accuracy may require MORE than the current 850us, not just shorter dwells.
# These are test candidates, not settings applied to the user's session.
DEFAULT_DWELLS_US = (15000, 10000, 5000, 2000, 1000, 850, 650, 500, 350, 250, 150, 50)
SUPPORTED_BETA_VERSIONS = (0x10073, 0x10074, 0x10076)


def verify_usb(port, serial_number):
    matched = [p for p in list_ports.comports() if p.device == port and p.vid == 0x0483
               and p.pid == 0x5740 and p.serial_number == serial_number]
    if len(matched) != 1:
        raise RuntimeError('Expected native STM32 USB identity is not present; no light command sent')
    with UsbOtaClient(port, serial_number) as client:
        return asdict(client.query_info())


def save_report(path, payload):
    # Generated diagnostic artifacts only; not a production/calibration catalog.
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def run_capture(args, plan, waits, path, expected_identity, *, single_adc=None,
                fast_dac_writes=None):
    identity = verify_usb(args.port, args.serial_number)
    if identity != expected_identity:
        raise RuntimeError('Board identity/firmware/boot changed before capture')
    thermal = []
    with ThermalSerial(args.port, 115200, timeout=.03) as device:
        device.thermal_observer.callback = lambda value: thermal.append(dict(value))
        report = execute_device(
            device, plan, path, cycles=args.cycles,
            first_delay_us=850, boundary_extra_us=0,
            point_delays_us=waits, adc_channel=args.channel,
            feedback_selector=args.feedback_selector,
            single_sample=args.single_adc if single_adc is None else single_adc,
            fast_dac_writes=args.fast_dac_writes if fast_dac_writes is None else fast_dac_writes)
    # execute_device has independently confirmed disarm and shutter before
    # closing its handle. INFO is read-only and uses its own short-lived handle.
    after = verify_usb(args.port, args.serial_number)
    if after != identity:
        raise RuntimeError('Board identity/firmware/boot changed during capture')
    valid = [value for value in thermal if value['thermal_flags'] & 1]
    faults = [value for value in thermal if not value['thermal_flags'] & 1
              or value['thermal_flags'] & (4 | 8 | 32)]
    report.update(hardware_identity=identity, stationary_setup_confirmed=True,
                  thermal_snapshots=thermal,
                  thermal_summary=dict(valid_snapshots=len(valid), fault_snapshots=len(faults),
                      sensor_age_max_ms=max((value['sensor_age_ms'] for value in valid), default=None),
                      temperature_min_c=min((value['temperature_c'] for value in valid), default=None),
                      temperature_max_c=max((value['temperature_c'] for value in valid), default=None)))
    save_report(path, report)
    if not valid or faults or any(not 0 < value['temperature_c'] < 85 for value in valid):
        raise RuntimeError('Missing, faulty or unsafe USB thermal telemetry; no wait recommendation')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', default='COM18')
    parser.add_argument('--serial-number', default='207933893232')
    parser.add_argument('--expected-version', type=lambda text: int(text, 0),
                        help='Optional exact runtime version lock, e.g. v118=0x10076')
    parser.add_argument('--machine-id', default='machine_3')
    parser.add_argument('--channel', type=int, choices=range(4), default=0)
    parser.add_argument('--feedback-selector', type=int, choices=range(4), default=2)
    parser.add_argument('--cycles', type=int, default=64)
    parser.add_argument('--dwells-us', type=int, nargs='+', default=list(DEFAULT_DWELLS_US),
                        help='Test candidates, 50..15000us; default tests longer AND shorter waits')
    parser.add_argument('--candidate-waits', type=Path,
                        help='Optional saved wait_balance JSON, in addition to the uniform sweep')
    parser.add_argument('--single-adc', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--fast-dac-writes', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--allow-light', action='store_true')
    parser.add_argument('--stationary-setup', action='store_true')
    args = parser.parse_args()
    if not args.allow_light or not args.stationary_setup:
        parser.error('Needs explicit --allow-light and --stationary-setup; no hardware accessed')
    if not 32 <= args.cycles <= 512:
        parser.error('cycles must be 32..512')
    if args.fast_dac_writes and not args.single_adc:
        parser.error('Merged DAC writes need single ADC; no automatic protocol fallback')
    if args.channel >= 2 and args.feedback_selector != 0:
        parser.error('CH2/3 use fixed feedback selector 0')
    payload = json.loads(args.session.read_text(encoding='utf-8'))
    plan = validate_plan_payload(payload['plan'])
    if plan.get('signal_channel', payload.get('signal_channel', 1)) != args.channel:
        raise ValueError('Saved route belongs to another ADC channel')
    if payload.get('machine_id', plan.get('machine_id', args.machine_id)) != args.machine_id:
        raise ValueError('Saved session belongs to another machine')
    n = len(plan['rows'])
    candidates = [[dwell] * n for dwell in args.dwells_us]
    if args.candidate_waits:
        balanced = json.loads(args.candidate_waits.read_text(encoding='utf-8'))
        balanced = balanced.get('wait_balance', balanced)
        key = [[r['index'], list(r['codes'])] for r in plan['rows']]
        if (balanced.get('route_key') != key or not balanced.get('candidate_available')
                or balanced.get('signal_channel') != args.channel
                or balanced.get('feedback_selector') != args.feedback_selector
                or balanced.get('fast_dac_writes') != args.fast_dac_writes):
            raise ValueError('Candidate wait identity/style mismatch or incomplete candidate')
        candidates.append(list(balanced['candidate_point_delays_us']))
    if not candidates:
        raise ValueError('No wait candidates provided')
    for waits in candidates + [[15000]*n]:
        command(plan['rows'], args.cycles, 1, first_delay_us=850, point_delays_us=waits,
                adc_channel=args.channel, feedback_selector=args.feedback_selector,
                single_sample=args.single_adc, fast_dac_writes=args.fast_dac_writes)
    if args.single_adc:
        command(plan['rows'], args.cycles, 1, first_delay_us=850, point_delays_us=[15000]*n,
                adc_channel=args.channel, feedback_selector=args.feedback_selector,
                single_sample=False, fast_dac_writes=False)
    if args.output.exists():
        raise FileExistsError(args.output)
    identity = verify_usb(args.port, args.serial_number)
    if args.expected_version is not None and identity['firmware_version'] != args.expected_version:
        raise RuntimeError('Actual runtime firmware differs from the explicitly selected version; no light commanded')
    if args.single_adc and identity['firmware_version'] not in SUPPORTED_BETA_VERSIONS:
        raise RuntimeError('Single ADC requires verified v1.0.115/116/118 Beta; firmware not changed by this tool')
    if args.fast_dac_writes:
        raise RuntimeError('Merged DAC writes failed optical comparison and remain disabled; no light commanded')
    args.output.mkdir(parents=True)
    summary = dict(schema='temporary_wait_balance_live_sweep_v1',hardware_identity=identity,
                   comparisons=[],complete=False,absolute_wavelength_accuracy_verified=False,
                   waits_applied=False)
    previous_machine = get_runtime_machine_id()
    try:
        set_runtime_machine_id(args.machine_id)
        if args.single_adc:
            print('Qualify ADC/DAC style: legacy slow reference, optimized slow, legacy slow reference',
                  flush=True)
            legacy_before = run_capture(args, plan, [15000]*n,
                args.output/'style.legacy_before.json', identity,
                single_adc=False, fast_dac_writes=False)
            optimized = run_capture(args, plan, [15000]*n,
                args.output/'style.optimized.json', identity)
            legacy_after = run_capture(args, plan, [15000]*n,
                args.output/'style.legacy_after.json', identity,
                single_adc=False, fast_dac_writes=False)
            qualification = compare_pipeline_styles(legacy_before, optimized, legacy_after)
            summary['sampling_style_qualification'] = qualification
            save_report(args.output / 'summary.json', summary)
            if not qualification['relative_sampling_style_consistency_passed']:
                raise RuntimeError('Optimized sampling disagrees with legacy slow references; '
                                   'no short-wait recommendation: '+', '.join(qualification['reasons']))
        for number, waits in enumerate(candidates):
            print(f'Trial {number+1}/{len(candidates)}: reference before, candidate, reference after', flush=True)
            prefix = args.output / f'trial_{number+1:02d}'
            pre = run_capture(args, plan, [15000]*n, prefix.with_suffix('.before.json'), identity)
            fast = run_capture(args, plan, waits, prefix.with_suffix('.candidate.json'), identity)
            post = run_capture(args, plan, [15000]*n, prefix.with_suffix('.after.json'), identity)
            comparison = compare_full_route(pre, fast, post)
            summary['comparisons'].append(comparison)
            save_report(args.output / 'summary.json', summary)
            print(json.dumps(comparison,ensure_ascii=False), flush=True)
        summary['fastest_relative_accuracy_candidate'] = choose_fastest_comparison(summary['comparisons'])
        summary['complete'] = True
    except BaseException as exc:
        summary['error'] = str(exc)
        raise
    finally:
        set_runtime_machine_id(previous_machine)
        save_report(args.output / 'summary.json', summary)
    # Do not edit session/defaults, install firmware, or publish a calibration.
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
