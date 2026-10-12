"""Visible 3D/E-stop shell around one guarded fast-spectrum press."""
import argparse
import json
import sys
import time
import math
from pathlib import Path
from PyQt5 import QtCore, QtWidgets
import fbg_unified_app as studio
from commission_held_depth_teacher import DiagnosticWindow


class Window(DiagnosticWindow):
    def __init__(self, output, optical_only=False, point_index=9, depth_mm=0.30,
                 high_rate_position=False, protocol='fullmap45', candidate=None,
                 candidate_probe=None, first_delay_us=400, hold_seconds=1.0, ch1_selector=2,
                 baseline_start_sequence=16):
        super().__init__()
        if point_index not in range(1,18):
            raise ValueError('Calibrated silicone coverage point required')
        self.point_index = point_index
        if not math.isfinite(depth_mm) or not 0 <= depth_mm <= 0.8:
            raise ValueError('Indentation must be within 0..0.8 mm')
        self.depth_mm = depth_mm
        if not math.isfinite(hold_seconds) or not 0.5 <= hold_seconds <= 3.0:
            raise ValueError('Hold time must be within 0.5..3.0 s')
        self.hold_seconds = hold_seconds
        if protocol not in {'flank18', 'triplet27', 'fullmap45', 'candidate27'}:
            raise ValueError('Unsupported fast-spectrum protocol')
        self.protocol = protocol
        if ch1_selector not in range(4) or (ch1_selector != 2 and protocol != 'fullmap45'):
            raise ValueError('Explicit gain requires F45')
        self.ch1_selector = ch1_selector
        self.baseline_start_sequence = baseline_start_sequence
        self.candidate, self.candidate_probe = candidate, candidate_probe
        self.first_delay_us = first_delay_us
        self.output = output
        self.optical_only = optical_only
        self.trace = []
        self.child = QtCore.QProcess(self)
        self.child.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        environment = QtCore.QProcessEnvironment.systemEnvironment()
        environment.insert('PYTHONIOENCODING', 'utf-8')
        self.child.setProcessEnvironment(environment)
        self.child.readyReadStandardOutput.connect(self.read_output)
        self.child.finished.connect(self.finished)
        self.child.errorOccurred.connect(self.child_error)
        self.position_child = QtCore.QProcess(self)
        self.position_child.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.position_child.setProcessEnvironment(environment)
        self.position_child.readyReadStandardOutput.connect(self.read_position_output)
        self.position_timer = QtCore.QTimer(self)
        self.position_timer.setSingleShot(True)
        self.position_timer.timeout.connect(self.start_position_trace)
        if high_rate_position:
            self.position_timer.start(6000)
        self.mach3_emergency.status_received.connect(self.record_status)
        self.started = time.monotonic()
        self.deadline_stopped = False
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.watch)
        self.timer.start(200)
        QtCore.QTimer.singleShot(2000, self.start_capture)

    def start_capture(self):
        candidate_arguments = []
        if self.protocol == 'candidate27':
            candidate_arguments = ['--candidate',str(self.candidate),
                '--candidate-probe',str(self.candidate_probe),'--first-delay-us',str(self.first_delay_us)]
        if self.optical_only:
            scripts = {'flank18': 'capture_fast_flank_stream.py',
                       'triplet27': 'capture_fast_triplet_stream.py',
                       'fullmap45': 'capture_fast_fullmap_stream.py',
                       'candidate27': 'capture_candidate27_stream.py'}
            arguments = ['--cycles', '256', '--output', str(self.output/'capture.json')]
            if self.protocol == 'fullmap45':
                arguments[2:2] = ['--spacing-us', '50', '--ch1-selector', str(self.ch1_selector)]
            if self.protocol == 'candidate27':
                candidate_arguments[candidate_arguments.index('--candidate-probe')]='--probe'
                arguments.extend(candidate_arguments)
            self.child.start(sys.executable, [str(Path(__file__).with_name(scripts[self.protocol])), *arguments])
            return
        self.child.start(sys.executable, [str(Path(__file__).with_name('commission_fast_flank_center_contact.py')),
            '--operator-confirmed', '--protocol', self.protocol, '--cycles', '640',
            '--spacing-us', '50', '--depth-mm', str(self.depth_mm), '--point-index', str(self.point_index),
            '--hold-seconds', str(self.hold_seconds), '--ch1-selector', str(self.ch1_selector),
            '--baseline-start-sequence', str(self.baseline_start_sequence),
            '--output', str(self.output/'capture.json'),*candidate_arguments])

    def record_status(self, status):
        self.trace.append(dict(received_ns=time.monotonic_ns(), status=dict(status)))
        if status.get('moving') and not (self.output/'moving.png').exists():
            self.grab().save(str(self.output/'moving.png'))

    def start_position_trace(self):
        if self.child.state()==QtCore.QProcess.NotRunning:
            return
        self.position_child.start(sys.executable,[str(Path(__file__).with_name('capture_mach3_position_trace.py')),
            '--seconds','10','--output',str(self.output/'high_rate_position.json')])

    def read_position_output(self):
        with (self.output/'position_trace.log').open('ab') as stream:
            stream.write(bytes(self.position_child.readAllStandardOutput()))

    def read_output(self):
        raw = bytes(self.child.readAllStandardOutput())
        with (self.output/'child_output.log').open('ab') as stream:
            stream.write(raw)
        text = raw.decode('utf-8','replace')
        encoding = sys.stdout.encoding or 'utf-8'
        print(text.encode(encoding,'backslashreplace').decode(encoding),end='',flush=True)

    def watch(self):
        if time.monotonic()-self.started > 75 and not self.deadline_stopped:
            self.deadline_stopped = True
            self.mach3_emergency.emergency_stop()
            self.statusBar().showMessage('采集超时，已请求机床急停；等待采集进程安全结束')

    def child_error(self, error):
        if error == QtCore.QProcess.FailedToStart:
            self.finished(-1, QtCore.QProcess.CrashExit)

    def finished(self, code, status):
        self.timer.stop()
        self.position_timer.stop()
        (self.output/'ui_machine_trace.json').write_text(json.dumps(dict(
            child_exit_code=code, child_exit_status=int(status), deadline_estop=self.deadline_stopped,
            machine_trace=self.trace),indent=2), 'utf-8')
        self.grab().save(str(self.output/'finished.png'))
        print(f'visible_capture_child_exit={code}',flush=True)
        self.statusBar().showMessage('采集结束；结果仍需离线验证')
        QtCore.QTimer.singleShot(1000,self.close)

    def closeEvent(self,event):
        if self.child.state()!=QtCore.QProcess.NotRunning:
            self.mach3_emergency.emergency_stop()
            event.ignore()
            return
        if self.position_child.state()!=QtCore.QProcess.NotRunning:
            event.ignore()
            QtCore.QTimer.singleShot(500,self.close)
            return
        super().closeEvent(event)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operator-confirmed',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--optical-only',action='store_true',help='No motion; test USB while live Mach3/3D window is open')
    p.add_argument('--point-index',type=int,choices=range(1,18),default=9)
    p.add_argument('--depth-mm',type=float,default=0.30)
    p.add_argument('--hold-seconds',type=float,default=1.0)
    p.add_argument('--high-rate-position',action='store_true')
    p.add_argument('--protocol',choices=('flank18','triplet27','fullmap45','candidate27'),default='fullmap45')
    p.add_argument('--candidate',type=Path)
    p.add_argument('--candidate-probe',type=Path)
    p.add_argument('--first-delay-us',type=int,default=400)
    p.add_argument('--ch1-selector',type=int,choices=range(4),default=2)
    p.add_argument('--baseline-start-sequence',type=int,choices=(16,128),default=16)
    args=p.parse_args()
    if not args.operator_confirmed and not args.optical_only: p.error('Confirmed calibrated fixture required')
    if args.protocol=='candidate27':
        if args.candidate is None or args.candidate_probe is None:
            p.error('Candidate and matching probe required')
        from capture_candidate27_stream import prepare_capture
        prepare_capture(args.candidate,args.candidate_probe,256 if args.optical_only else 640,args.first_delay_us,50)
    args.output.mkdir(parents=True,exist_ok=False)
    app=QtWidgets.QApplication(sys.argv[:1]);studio.install_theme(app)
    w=Window(args.output.resolve(),optical_only=args.optical_only,point_index=args.point_index,depth_mm=args.depth_mm,high_rate_position=args.high_rate_position,protocol=args.protocol,candidate=args.candidate,candidate_probe=args.candidate_probe,first_delay_us=args.first_delay_us,hold_seconds=args.hold_seconds,ch1_selector=args.ch1_selector,baseline_start_sequence=args.baseline_start_sequence);w.show();w.workspace_tabs.setCurrentIndex(1)
    raise SystemExit(app.exec_())
