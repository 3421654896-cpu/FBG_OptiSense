"""Identity-locked finite dense reacquisition benchmark. --apply opens light."""
import argparse
import json
from pathlib import Path
import time

import serial
from serial.tools import list_ports
from fbg_usb_ota import UsbOtaClient
from machine_profile import set_runtime_machine_id
from temporary_reference import acquire_reference


def acquire_with_live_plot(device, args):
    """Exercise the actual worker/signals/plot offscreen, without user UI state."""
    import app_JDSU  # Same import order as the real client.
    from PyQt5 import QtCore, QtWidgets
    from temporary_test_widget import ReferenceWorker, TemporaryTestWindow
    qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = TemporaryTestWindow(None)  # No persistence or user record changes.
    window.resize(1500, 900)
    window.show()
    window._dense_capture_active = True
    worker = ReferenceWorker(device, None, args.output, 0, window,
                             adc_channel=1, feedback_selector=2,
                             reference_dense_indices=list(range(args.points)),
                             reference_acquisition_method='equal_interval_single',
                             reference_settle_s=args.settle_s)
    window.worker = worker
    reports, point_times = [], []
    loop = QtCore.QEventLoop()
    worker.reference_point.connect(window.update_dense_reference_point)
    worker.reference_point.connect(lambda payload: point_times.append(time.monotonic()))
    worker.message.connect(window.status.setText)
    worker.reference.connect(window.reference_ready)
    worker.reference.connect(reports.append)
    worker.finished.connect(window.finished)
    worker.finished.connect(loop.quit)
    timeout = QtCore.QTimer()
    timeout.setSingleShot(True)
    timeout.timeout.connect(worker.stop)
    timeout.start(30000)
    worker.start()
    loop.exec_()
    timeout.stop()
    worker.wait()
    if not reports:
        raise RuntimeError('GUI dense reference failed: ' + window.status.text())
    report = reports[0]
    qt_app.processEvents()
    gui = dict(received_points=len(point_times),
               plotted_points=len(window.dense_values) if window.dense_values is not None else 0,
               first_to_last_point_s=point_times[-1]-point_times[0] if len(point_times)>1 else 0,
               manual_selection_active=window.manual_selection_active,
               status=window.status.text())
    preview = args.output.with_suffix('.png')
    window.grab().save(str(preview))
    gui['preview'] = str(preview)
    window._spectrum_dirty = False
    window.close()
    if gui['received_points'] != args.points or gui['plotted_points'] != args.points:
        raise RuntimeError(f'GUI raw point loss: {gui}')
    return report, gui


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True)
    parser.add_argument('--device-serial', required=True)
    parser.add_argument('--machine', required=True, choices=['machine_1', 'machine_2', 'machine_3', 'machine_4'])
    parser.add_argument('--points', type=int, default=2001)
    parser.add_argument('--settle-s', type=float, default=.002)
    parser.add_argument('--legacy', action='store_true')
    parser.add_argument('--gui', action='store_true', help='Run the actual Qt worker and plot offscreen')
    parser.add_argument('--stop-after', type=int, help='Test safe partial cancellation after N received points')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if not args.apply or not 5 <= args.points <= 2001:
        parser.error('--apply and 5..2001 real points required')
    if args.stop_after is not None and (args.gui or not 1 <= args.stop_after < args.points):
        parser.error('--stop-after requires 1..points-1 and is not supported with --gui')
    if args.output.exists():
        parser.error('Output exists; preserving earlier acquisition')
    set_runtime_machine_id(args.machine)
    with UsbOtaClient(args.port, args.device_serial) as client:
        info = client.query_info()
    if not args.legacy and info.firmware_version < 0x1006a:
        raise RuntimeError('Dense batch requires firmware v1.0.106 or newer')
    connected = [p for p in list_ports.comports() if p.device.upper() == args.port.upper()
                 and p.serial_number == args.device_serial and p.vid == 0x483 and p.pid == 0x5740]
    if len(connected) != 1:
        raise RuntimeError('Exact STM32 USB identity no longer present')
    started = time.monotonic()
    gui = None
    received = [0]
    def progress(n, total):
        received[0] = n
        if n == 1 or n == total or n % 240 == 0:
            print(f'Reacquire {n}/{total}', flush=True)
    with serial.Serial(args.port, 2_000_000, timeout=.1, write_timeout=3) as device:
        device.dtr = True
        if args.gui:
            if args.legacy:
                raise ValueError('--gui exercises the real optimized client, not legacy fallback')
            report, gui = acquire_with_live_plot(device, args)
        else:
            report = acquire_reference(
                device, args.output, dense_indices=list(range(args.points)),
                acquisition_method='equal_interval_single', settle_s=args.settle_s,
                signal_channel=1, feedback_selector=2,
                prefer_board_batch=not args.legacy, continue_on_unstable=True,
                on_progress=progress,
                allow_partial_stop=args.stop_after is not None,
                should_stop=lambda: args.stop_after is not None and received[0] >= args.stop_after,
            )
    summary = dict(machine=args.machine, serial=args.device_serial,
                   backend=report['acquisition_backend'], points=len(report['rows']),
                   complete=report['complete'], failed_points=report['failed_point_count'],
                   requested_point_dwell_s=report['requested_point_dwell_s'],
                   scan_elapsed_s=report['acquisition_elapsed_s'],
                   total_elapsed_s=report['total_elapsed_s'],
                   board_elapsed_s=report.get('board_acquisition_elapsed_s'),
                   guard_dwell_s=report.get('transition_guard_dwell_s'),
                   exact_disarm_ack=report['disarm_ack'], exact_shutter_ack=report['shutter_ack'],
                   wall_s=time.monotonic()-started, output=str(args.output), gui=gui)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
