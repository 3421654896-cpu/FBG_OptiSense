"""Stationary diagnostic dense scan/reselection; never overwrite UI/default routes.

Uses the existing independent five-write, single fresh ADC dense transport.
The new seven-point route is a diagnostic candidate, NOT a validated default.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import app_JDSU as app
from benchmark_temporary_wait_balance_live import verify_usb, save_report
from temporary_reference import acquire_reference
from temporary_test_mode import auto_select_dense_plan, validate_plan_payload, LIMITS
from thermal_serial import ThermalSerial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', default='COM18')
    parser.add_argument('--device-serial', default='207933893232')
    parser.add_argument('--machine-id', choices=('machine_3',), required=True)
    parser.add_argument('--allow-light', action='store_true')
    parser.add_argument('--stationary-setup', action='store_true')
    args = parser.parse_args()
    if not args.allow_light or not args.stationary_setup:
        parser.error('Explicit light authorization and stationary setup required')
    source = json.loads(args.source_capture.read_text(encoding='utf-8'))
    route = validate_plan_payload(source['plan'])
    source_machine = source.get('machine_id', route.get('machine_id'))
    if ((source_machine is not None and source_machine != args.machine_id)
            or source.get('signal_channel') != 0):
        raise ValueError('Requires machine3 CH0 source capture')
    if args.output.exists():
        raise FileExistsError(args.output)
    previous = app.get_runtime_machine_id()
    summary = dict(complete=False, default_route_changed=False,
                   absolute_wavelength_accuracy_verified=False,
                   source_machine_tag=source_machine, requested_machine_id=args.machine_id)
    try:
        app.set_runtime_machine_id('machine_3')
        table = app.load_fullband_accuracy_table()
        if (len(table) != 2001 or any(tuple(row['codes']) != table[row['index']].codes
                                    for row in route['rows'])):
            raise ValueError('Current table no longer matches the source route')
        # Older captures lack a machine tag. All sparse codes must still match
        # the explicitly selected table, and runtime USB UID is independently locked.
        summary['all_source_codes_match_machine_table'] = True
        if any(any(code < 0 or code > bound for code, bound in zip(row.codes, LIMITS))
               for row in table):
            raise ValueError('Table code exceeds 145mA/channel safety bounds')
        identity = verify_usb(args.port, args.device_serial)
        if identity['firmware_version'] != 0x10073:
            raise RuntimeError('Expected verified v1.0.115 Beta, not a different device/build')
        args.output.mkdir(parents=True)
        summary.update(hardware_identity=identity,
            table_sha256=hashlib.sha256(json.dumps([asdict(row) for row in table],
                sort_keys=True).encode()).hexdigest())
        thermal = []
        with ThermalSerial(args.port, 115200, timeout=.03, write_timeout=3) as device:
            device.thermal_observer.callback = lambda value: thermal.append(dict(value))
            report = acquire_reference(device, args.output/'dense_10ms.json',
                signal_channel=0, feedback_selector=2,
                dense_indices=list(range(0, 1501, 2)),
                acquisition_method='equal_interval_single', settle_s=.010,
                require_stable=False, continue_on_unstable=True,
                on_progress=lambda n,total: print(f'Dense {n}/{total}', flush=True) if n%100==0 else None)
        summary.update(thermal_snapshots=thermal,
                       disarm_ack=report.get('disarm_ack'), shutter_ack=report.get('shutter_ack'))
        after = verify_usb(args.port, args.device_serial)
        if after != identity:
            raise RuntimeError('USB identity/firmware/boot changed')
        if (not report.get('complete') or not report.get('disarm_ack')
                or not report.get('shutter_ack') or report.get('error')):
            raise RuntimeError('Dense scan incomplete or shutdown unconfirmed')
        if (not thermal or any(not value['thermal_flags'] & 1
                               or value['thermal_flags'] & (4|8|32)
                               or not 0 < value['temperature_c'] < 85 for value in thermal)):
            raise RuntimeError('Missing/faulty/unsafe thermal telemetry')
        x = [row['measured_wavelength_nm'] for row in report['rows']]
        y = [row['signal_adc_code'] for row in report['rows']]
        plan = auto_select_dense_plan(table, x, y, feedback_selector=2,
                                      source=str((args.output/'dense_10ms.json').resolve()))
        plan.update(signal_channel=0, machine_id='machine_3')
        plan = validate_plan_payload(plan)
        candidate = dict(plan=plan, signal_channel=0, feedback_selector=2, machine_id='machine_3',
                         hardware_identity=identity, optical_accuracy_verified=False)
        save_report(args.output/'candidate_session.json', candidate)
        summary.update(complete=True, point_count=plan['point_count'], peak_count=plan['peak_count'],
                       selection_checks=plan.get('selection_fit_checks'),
                       selection_validation_required=plan['selection_validation_required'])
    except BaseException as exc:
        summary['error'] = str(exc)
        raise
    finally:
        app.set_runtime_machine_id(previous)
        if args.output.exists():
            save_report(args.output/'summary.json', summary)
        print(json.dumps({key: summary.get(key) for key in
                         ('complete', 'error', 'point_count', 'peak_count', 'disarm_ack',
                          'shutter_ack', 'default_route_changed', 'selection_validation_required')},
                         ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
