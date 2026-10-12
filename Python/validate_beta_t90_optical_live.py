"""Identity-locked, explicitly authorized optical test of the actual Beta t90 worker.

No waits, routes, tables, firmware, or UI preferences are applied/overwritten.
Uses exactly the same worker as the restored Beta button, synchronously.
"""
import argparse
import json
from pathlib import Path

from PyQt5 import QtCore
from benchmark_temporary_wait_balance_live import verify_usb, save_report
from machine_profile import get_runtime_machine_id, set_runtime_machine_id
from temporary_test_mode import validate_plan_payload
from temporary_test_widget import SwitchTimingWorker
from thermal_serial import ThermalSerial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', default='COM18')
    parser.add_argument('--device-serial', default='207933893232')
    parser.add_argument('--allow-light', action='store_true')
    parser.add_argument('--stationary-setup', action='store_true')
    args = parser.parse_args()
    if not args.allow_light or not args.stationary_setup:
        parser.error('Requires explicit stationary optical-test authorization')
    source = json.loads(args.session.read_text(encoding='utf-8'))
    plan = validate_plan_payload(source['plan'])
    if source.get('machine_id') != 'machine_3' or source.get('signal_channel') != 0:
        raise ValueError('Requires the machine3 CH0 diagnostic route')
    if args.output.exists():
        raise FileExistsError(args.output)
    identity = verify_usb(args.port, args.device_serial)
    if identity != source['hardware_identity'] or identity['firmware_version'] != 0x10073:
        raise RuntimeError('Diagnostic route USB identity/version/boot does not match')
    qt_app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    previous = get_runtime_machine_id()
    thermal = []
    try:
        set_runtime_machine_id('machine_3')
        worker_error = None
        try:
            with ThermalSerial(args.port, 115200, timeout=.1, write_timeout=3) as device:
                device.thermal_observer.callback = lambda value: thermal.append(dict(value))
                worker = SwitchTimingWorker(device, plan, args.output, cycles=32, feedback_selector=2,
                                            adc_channel=0, fast_dac_writes=False,
                                            point_delays_us=[850]*len(plan['rows']))
                messages = [0]
                def progress(message):
                    messages[0] += 1
                    if messages[0] % 20 == 0 or '未' in message:
                        print(message, flush=True)
                worker.message.connect(progress)
                worker.run()
        except Exception as exc:
            worker_error = exc
        report = json.loads(args.output.read_text(encoding='utf-8'))
        after = verify_usb(args.port, args.device_serial)
        report.update(hardware_identity=identity, thermal_snapshots=thermal,
                      stationary_setup_confirmed=True, waits_applied=False,
                      absolute_wavelength_accuracy_verified=False)
        if worker_error:
            report['wrapper_error'] = f'{type(worker_error).__name__}: {worker_error}'
        save_report(args.output, report)
        if worker_error:
            raise RuntimeError('Worker failed; raw capture/thermal/shutdown evidence preserved') from worker_error
        if (not report.get('complete') or not report.get('exact_disarm_ack')
                or not report.get('exact_soa_shutter_ack') or report.get('error')):
            raise RuntimeError('t90 capture incomplete or shutdown unconfirmed: '+str(report.get('error')))
        if after != identity:
            raise RuntimeError('USB boot/identity/version changed')
        if (not thermal or any(not value['thermal_flags'] & 1
                               or value['thermal_flags'] & (4|8|32)
                               or not 0 < value['temperature_c'] < 85 for value in thermal)):
            raise RuntimeError('Missing/faulty/unsafe thermal telemetry')
        balance = report.get('t90_waits', report.get('wait_balance', {}))
        print(json.dumps(dict(complete=True, captures=len(report['captures']),
            candidate_available=balance.get('candidate_available'),
            candidate_wait_sum_us=balance.get('candidate_wait_sum_us'),
            disarm_ack=report['exact_disarm_ack'], shutter_ack=report['exact_soa_shutter_ack'],
            waits_applied=False), ensure_ascii=False), flush=True)
    finally:
        set_runtime_machine_id(previous)


if __name__ == '__main__':
    main()
