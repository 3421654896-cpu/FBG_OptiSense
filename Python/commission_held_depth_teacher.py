"""Bounded G4 teacher capture at two centre indentations with live 3D/E-stop UI.

No firmware update, spindle start, model deployment or area truth is implied.
Each existing guarded press retracts after ten seconds independently of capture.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from PyQt5 import QtCore, QtWidgets
import fbg_unified_app as studio
from commission_finger_coverage_once import CoverageWindow, validate_machine
from commission_fast_flank_center_contact import coverage_target


def validate_teacher_result(data, depth):
    if not data['complete'] or len(data['captures']) != 1 or not all(
            data['safety_cleanup'][k] for k in ('exact_disarm_ack','exact_soa_shutter_ack')):
        raise ValueError('Incomplete diagnostic or unconfirmed optical shutdown')
    record = data['captures'][0]
    if not record['teacher']['stable']:
        raise ValueError('Stable teacher unavailable')
    bracket = record['machine_position_bracket']
    for phase in ('before', 'after'):
        measured = bracket[phase]['pressure_reference']['indentation_mm']
        if abs(measured-depth) > .003:
            raise ValueError('Actual captured indentation differs from requested contact')
    return measured


class DiagnosticWindow(CoverageWindow):
    def __init__(self):
        super().__init__()
        # Only this commissioning window hides unrelated acquisition cards.
        # The shared main application layout is unchanged; E-stop stays visible.
        for button in (self.finger_controller.capture_button,
                       self.finger_controller.fast_flank_start_button,
                       self.finger_controller.coverage_preview_button):
            parent = button.parentWidget()
            while parent is not None and parent is not self.finger_page:
                if isinstance(parent, QtWidgets.QFrame) and parent.objectName() == 'controlGroup':
                    parent.hide()
                    break
                parent = parent.parentWidget()

    def switch_source(self, *args, **kwargs):
        # Exclusive USB ownership belongs to the bounded diagnostic child.
        # The original Mach3 telemetry and emergency service remain active.
        self.statusBar().showMessage('位移教师采集：光谱链路由诊断进程独占，机床实时显示保持开启')


class Run(QtCore.QObject):
    def __init__(self, window, output):
        super().__init__(window)
        self.w, self.output = window, output
        self.depths, self.index = (.1, .3), 0
        self.state = 'startup'
        self.started = time.monotonic()
        self.process = None
        self.records, self.trace = [], []
        self.error = None
        self.press_done = False
        self.child_done = False
        window.mach3_emergency.contact_reached.connect(self.contact)
        window.mach3_emergency.press_finished.connect(self.press_finished)
        window.mach3_emergency.status_received.connect(self.status)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(100)

    def status(self, status):
        self.trace.append(dict(received_ns=time.monotonic_ns(), status=dict(status)))

    def begin_press(self):
        self.target = coverage_target(9, self.depths[self.index])
        validate_machine(self.w._last_mach3_status,
                         self.target['calibration_machine_minus_work'], self.target['contact_z'])
        self.target.update(dwell_seconds=10., xy_feed_mm_min=60.,
                           z_travel_feed_mm_min=60., press_feed_mm_min=15.)
        self.press_done = self.child_done = False
        self.state = 'pressing'
        self.w.mach3_emergency.press_target(self.target)
        print(f'press_depth_mm={self.depths[self.index]}', flush=True)

    def contact(self, target, status):
        if self.state != 'pressing':
            return
        self.state = 'capturing'
        self.capture_path = self.output / f'depth_{self.index}.json'
        self.records.append(dict(requested_depth_mm=self.depths[self.index],
                                 contact_status=status, contact_received_ns=time.monotonic_ns(),
                                 target=target, capture=str(self.capture_path)))
        self.process = QtCore.QProcess(self)
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.finished.connect(self.child_finished)
        self.process.errorOccurred.connect(self.child_error)
        self.process.start(sys.executable, [str(Path(__file__).with_name('capture_sparse_held_trajectory.py')),
            '--selector', '2', '--schedule-version', '5', '--indices', '18', '--repeats', '1',
            '--record-machine-position', '--output', str(self.capture_path)])
        self.w.grab().save(str(self.output / f'contact_{self.index}.png'))

    def read_output(self):
        print(bytes(self.process.readAllStandardOutput()).decode('utf-8', 'replace'), end='', flush=True)

    def child_error(self, error):
        if error == QtCore.QProcess.FailedToStart:
            self.child_done = True
            self.error = 'Diagnostic child failed to start'

    def child_finished(self, code, status):
        self.child_done = True
        if code != 0 or status != QtCore.QProcess.NormalExit:
            self.error = f'Diagnostic child failed: {code}/{status}'
            return
        try:
            data = json.loads(self.capture_path.read_text('utf-8'))
            measured = validate_teacher_result(data, self.depths[self.index])
            self.records[-1]['actual_indentation_mm'] = measured
        except Exception as exc:
            self.error = str(exc)

    def press_finished(self, success, detail, status):
        self.press_done = True
        if self.records:
            self.records[-1].update(motion_success=success, final_status=status, motion_detail=detail)
        if not success:
            self.error = detail
        print(f'press_finished={success} {detail}', flush=True)

    def tick(self):
        try:
            if self.state == 'finished':
                return
            if time.monotonic()-self.started > 100:
                raise RuntimeError('Bounded run deadline exceeded')
            if self.error:
                if not self.w.mach3_emergency._busy and (self.process is None or self.child_done):
                    self.finish()
                return
            if self.state == 'startup':
                if not self.w._last_mach3_status:
                    return
                self.begin_press()
            elif self.press_done and self.child_done:
                self.index += 1
                if self.index == len(self.depths):
                    self.finish()
                else:
                    self.begin_press()
        except Exception as exc:
            self.error = str(exc)
            if self.w.mach3_emergency._busy:
                self.w.mach3_emergency.emergency_stop()
            elif self.process is None or self.child_done:
                self.finish()

    def finish(self):
        self.state = 'finished'
        result = dict(complete=not self.error and len(self.records)==2, error=self.error,
                      contacts=self.records, machine_trace=self.trace,
                      physical_15hz_verified=False, contact_area_verified=False,
                      trained_model=False)
        (self.output/'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), 'utf-8')
        self.w.grab().save(str(self.output/'finished.png'))
        print(json.dumps({k:v for k,v in result.items() if k not in ('contacts','machine_trace')}), flush=True)
        self.timer.stop()
        self.w.statusBar().showMessage('诊断结束，未训练或部署模型；压入量与原始曲线已保存')
        QtCore.QTimer.singleShot(1000, self.w.close)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operator-confirmed', action='store_true')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--preview', action='store_true', help='Read-only UI verification; no press or optical capture')
    args = p.parse_args()
    if not args.operator_confirmed and not args.preview:
        p.error('Confirmed calibrated fixture and emergency stop required')
    args.output.mkdir(parents=True, exist_ok=False)
    app = QtWidgets.QApplication(sys.argv[:1])
    studio.install_theme(app)
    w = DiagnosticWindow()
    w.show()
    w.workspace_tabs.setCurrentIndex(1)
    if args.preview:
        QtCore.QTimer.singleShot(2000, lambda: w.grab().save(str(args.output/'preview.png')))
        QtCore.QTimer.singleShot(3000, w.close)
    else:
        w.teacher_run = Run(w, args.output.resolve())
    return app.exec_()


if __name__ == '__main__':
    raise SystemExit(main())
