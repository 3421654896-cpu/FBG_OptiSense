"""One explicitly confirmed 0.1 mm centre press, using the unified studio.

No spindle commands are added. Existing guarded Mach3 service performs M5,
Z-up/XY/slow contact/retract. Default invocation only explains the required flag.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtWidgets

import app_JDSU as local
import fbg_unified_app as studio
from capture_ch1_rc_diagnostic import load_installed_rows
from finger_dataset import ContactEvent, FingerSessionWriter
from mechanical_finger_3d import SILICONE_Y_CENTER, generate_silicone_coverage_plan
from stress_multirate_protocol import MultirateConfiguration, REFERENCE_MAP_PERIOD


def validate_start(status, target):
    if not status.get("connected") or not status.get("stopped") or status.get("moving"):
        raise RuntimeError("Mach3 must be connected and stationary")
    if status.get("estop") or status.get("estop_led"):
        raise RuntimeError("E-stop is latched; no automatic reset is permitted")
    if not all(status.get(f"{axis}_homed") for axis in "xyz"):
        raise RuntimeError("XYZ homing is not confirmed")
    if any(status.get(f"{axis}_{direction}_limit") for axis in "xyz" for direction in ("pos", "neg")):
        raise RuntimeError("Hardware limit is active")
    if float(target["depth_mm"]) != 0.1:
        raise RuntimeError("This commissioning run is authorised for 0.1 mm only")
    expected = target["calibration_machine_minus_work"]
    machine_xyz = [float(status[f"machine_{axis}"]) for axis in "xyz"]
    work_xyz = [float(status[f"work_{axis}"]) for axis in "xyz"]
    target_machine = [
        float(target["work_x"]) + float(expected[0]),
        float(target["work_y"]) + float(expected[1]),
        float(target["contact_z"]) + float(expected[2]),
    ]
    if not all(
        math.isfinite(value)
        for value in machine_xyz + work_xyz + target_machine
    ):
        raise RuntimeError("Machine/work coordinates are unavailable")
    if machine_xyz[2] < target_machine[2] + 3.9:
        raise RuntimeError("Initial 4 mm clear height is not confirmed")
    if max(
        abs(machine_xyz[index] - target_machine[index])
        for index in (0, 1)
    ) > 10:
        raise RuntimeError("Initial centre approach exceeds this run's 10 mm XY bound")


def complete_baseline_snapshot(snapshot, *, full_map_reference=False):
    """A 45-slot container in a sparse frame is not a 45-point baseline."""
    mask = snapshot.get("fresh_point_mask") or ()
    ages = snapshot.get("point_age_frames") or ()
    return (snapshot.get("acquisition_profile") in {"MAP", "MAP45"}
            and len(mask) == 45 and all(mask)
            and len(ages) == 45 and all(age == 0 for age in ages)
            and bool(snapshot.get("bandwidth_discontinuity")) == bool(full_map_reference))


class CommissionWindow(studio.UnifiedFbgStudio):
    def closeEvent(self, event):
        # Do not leave an unattended motion thread behind when the GUI closes.
        if self.mach3_emergency._busy:
            self.mach3_emergency.emergency_stop()
            self.statusBar().showMessage("正在停止机床；确认轴停止后才能关闭")
            event.ignore()
            return
        super().closeEvent(event)


class CenterRun(QtCore.QObject):
    def __init__(self, window, output, *, acquisition_only=False, full_map_reference=False):
        super().__init__(window)
        self.window = window
        self.output = output
        self.acquisition_only = bool(acquisition_only)
        self.full_map_reference = bool(full_map_reference)
        self.native_recorder = None
        self.started = time.monotonic()
        self.phase_started = self.started
        self.state = "startup"
        self.motion_sent = False
        self.motion_finished = False
        self.failure = None
        self.last_seen_key = None
        self.baseline_count = 0
        self.baseline_complete_maps = 0
        self.last_diagnostic_s = 0.0
        self.status_trace = []
        self.summary = {"purpose": "single centre 0.1 mm commissioning, not 15 Hz validation",
                        "operator_confirmed": True, "motion_commands_started": 0,
                        "measured_contact_area_mm2": None, "pressure_ground_truth": None}
        self.summary["acquisition_only"] = self.acquisition_only
        self.summary["full_map_reference"] = self.full_map_reference
        if self.acquisition_only:
            self.summary["purpose"] = "stationary native reception diagnostic; no motion permitted"
        window.mach3_emergency.status_received.connect(self.status_received)
        window.mach3_emergency.contact_reached.connect(lambda *_: self.mark("contact"))
        window.mach3_emergency.release_started.connect(lambda *_: self.mark("release"))
        window.mach3_emergency.release_reached.connect(lambda *_: self.mark("retracted"))
        window.mach3_emergency.press_finished.connect(self.finished_motion)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(50)

    def mark(self, phase):
        print(f"phase={phase}", flush=True)
        self.window.grab().save(str(self.output / f"ui_{phase}.png"))

    def status_received(self, status):
        self.status_trace.append({"host_received_monotonic_ns": time.monotonic_ns(), **status})

    def finished_motion(self, success, detail, status):
        self.motion_finished = True
        self.summary.update(motion_success=bool(success), motion_detail=detail,
                            final_motion_status=status)
        if not success:
            self.failure = self.failure or detail
        print(f"motion_finished={int(success)} {detail}", flush=True)

    def make_target(self):
        finger = self.window.finger_controller
        plan = generate_silicone_coverage_plan(finger.machine_touch_samples, finger.layout.sensors,
                                               spacing_mm=4.0)
        candidates = [dict(p) for p in plan if abs(p["model_x"]) < 1e-8
                      and abs(p["model_y"] - SILICONE_Y_CENTER) < 1e-8]
        if len(candidates) != 1:
            raise RuntimeError("No uniquely validated full-8-mm-face silicone centre")
        offsets = [(s.machine_x - s.work_x, s.machine_y - s.work_y, s.machine_z - s.work_z)
                   for s in finger.machine_touch_samples
                   if None not in (s.machine_x, s.machine_y, s.machine_z)]
        if len(offsets) < 2:
            raise RuntimeError("Two verified machine/work calibration offsets are required")
        target = candidates[0]
        target.update(point_index=1, coverage_repetition=1, depth_mm=0.1,
                      xy_feed_mm_min=30.0, z_travel_feed_mm_min=30.0,
                      press_feed_mm_min=5.0, approach_clearance_mm=0.3, dwell_seconds=2.0,
                      calibration_machine_minus_work=np.median(offsets, axis=0).tolist())
        return target

    def begin_recording(self):
        w = self.window
        self.target = self.make_target()
        validate_start(w._last_mach3_status, self.target)
        self.summary["target"] = self.target
        writer = FingerSessionWriter(self.output / "session", session_id=self.output.name,
                                     queue_capacity=2048,
                                     manifest_extra={"operator_confirmed": True, "depth_mm": None if self.acquisition_only else 0.1,
                                               "only_one_press": not self.acquisition_only, "physical_15hz_verified": False,
                                               "training_eligible": False,
                                               "acquisition_only": self.acquisition_only,
                                               "full_map_reference": self.full_map_reference,
                                               "sampling_purpose": "complete_spectrum_rc_reference" if self.full_map_reference else "sparse_acquisition_diagnostic",
                                               "host_receipt_clock": time.get_clock_info("monotonic").implementation,
                                               "host_receipt_resolution_s": time.get_clock_info("monotonic").resolution,
                                               "adc_event_synchronization_verified": False})
        writer.write_event(ContactEvent("session_start", metadata=self.summary))
        w._coverage_writer = writer
        w._coverage_dataset_path = writer.session_dir
        w._coverage_current = dict(self.target)
        w._coverage_queue.clear()
        w._coverage_total = 1
        w._coverage_completed = 0
        w._coverage_records = 0
        w._coverage_writer_sequence = 0
        w._coverage_phase_frame_counts = {}
        w._coverage_last_frame_key = None
        w._coverage_active = True
        # Exercise the same recording path used by the main coverage program.
        w._start_coverage_native_capture()
        self.native_recorder = w._coverage_native_recorder
        self.summary["native_capture_file"] = "session/native_frames.jsonl"
        self.summary["receive_buffer_start"] = asdict(local.frames_queue.stats())
        self.summary["recording_start_status"] = dict(w._last_mach3_status)
        w._coverage_motion_test = False
        w._coverage_cancel_after_current = False
        w.finger_controller.set_coverage_running(True)
        w._begin_coverage_phase("pre_contact", self.target, dict(w._last_mach3_status))
        w._coverage_capture_timer.start()
        self.phase_started = time.monotonic()
        self.state = "baseline"
        self.mark("baseline")

    def fail(self, reason):
        self.failure = str(reason)
        self.summary["failure_receive_buffer"] = asdict(local.frames_queue.stats())
        self.summary["failure_monotonic_ns"] = time.monotonic_ns()
        try:
            self.summary["failure_snapshot_received_ns"] = self.window._active_spectrum_snapshot().get("frame_received_monotonic_ns")
        except Exception as exc:
            self.summary["failure_snapshot_error"] = str(exc)
        print(f"commission_error={reason}", flush=True)
        if self.motion_sent and not self.motion_finished:
            self.window.mach3_emergency.emergency_stop()
            self.state = "stopping"
        else:
            self.finish()

    def finish(self):
        self.timer.stop()
        w = self.window
        if w._coverage_active:
            w._finish_coverage_run(False, self.failure or "Commissioning ended")
        optical_shutdown = False
        try:
            if local.ser_open:
                w.local_controller.page_peak.on_open_changed()
            # Recheck the final hardware SOA gate even if the page stopped itself.
            if local.ser.is_open:
                local.set_soa_shutter_and_verify(local.ser)
                optical_shutdown = True
        except Exception as exc:
            self.failure = f"{self.failure or ''}; shutdown: {exc}"
        self.summary.update(error=self.failure, optical_shutter_confirmed=optical_shutdown,
                            saved_frames=w._coverage_records,
                            phase_frame_counts=w._coverage_phase_frame_counts,
                            elapsed_s=time.monotonic() - self.started)
        self.summary["receive_buffer_end"] = asdict(local.frames_queue.stats())
        self.summary["recording_end_status"] = dict(w._last_mach3_status)
        if self.native_recorder is not None:
            try:
                self.summary["native_recording"] = self.native_recorder.close()
            except Exception as exc:
                self.failure = f"{self.failure or ''}; native capture: {exc}"
                self.summary["error"] = self.failure
                self.summary["native_recording"] = self.native_recorder.stats()
        (self.output / "summary.json").write_text(json.dumps(self.summary, ensure_ascii=False, indent=2), "utf-8")
        (self.output / "machine_status.json").write_text(json.dumps(self.status_trace, ensure_ascii=False, indent=2), "utf-8")
        self.mark("finished")
        print(json.dumps(self.summary, ensure_ascii=False), flush=True)
        self.state = "finished"
        # Leave the unified application and its emergency button visible.
        done = "静止采集诊断已结束，未发送运动指令" if self.acquisition_only else "0.1 mm单次测试已结束，光谱已保存"
        w.statusBar().showMessage(done if not self.failure else f"测试中止：{self.failure}")

    def tick(self):
        try:
            w = self.window
            if time.monotonic() - self.started > 100 and self.state not in {"stopping", "finished"}:
                raise RuntimeError("One-shot commissioning deadline exceeded")
            if self.state == "startup":
                if not w._last_mach3_status or not local.ser.is_open:
                    return
                self.target = self.make_target()
                validate_start(w._last_mach3_status, self.target)
                page = w.local_controller.page_peak
                for button, selector in zip(page.stress_feedback_buttons, (0, 2)):
                    button.setProperty("feedbackSelector", selector)
                page._update_stress_feedback_controls()
                page.multirate_configuration = MultirateConfiguration(
                    map_period_frames=REFERENCE_MAP_PERIOD if self.full_map_reference else 3,
                    activate_threshold_codes=40, release_threshold_codes=20)
                self.state = "warming"
                w.local_controller.start_stress_debug()
                w._sync_local_mode()
                self.phase_started = time.monotonic()
                return
            if self.state in {"warming", "baseline", "observing", "moving"}:
                snapshot = w._active_spectrum_snapshot()
                received = snapshot.get("frame_received_monotonic_ns")
                fresh = received is not None and 0 <= time.monotonic_ns() - received < 500_000_000
                if time.monotonic() - self.last_diagnostic_s >= 5:
                    page = w.local_controller.page_peak
                    diagnostic = {"state": self.state, "key": snapshot.get("frame_key"),
                                  "age_ms": None if received is None else (time.monotonic_ns() - received) / 1e6,
                                  "worker_running": page.worker.isRunning(),
                                  "serial_acquiring": local.ser_open,
                                  "baseline_count": self.baseline_count,
                                  "baseline_complete_maps": self.baseline_complete_maps,
                                  "receive_buffer": asdict(local.frames_queue.stats())}
                    print(json.dumps(diagnostic), flush=True)
                    self.summary["last_acquisition_diagnostic"] = diagnostic
                    self.last_diagnostic_s = time.monotonic()
                if self.state == "warming" and time.monotonic() - self.phase_started > 12:
                    raise RuntimeError("No sustained fresh CH1 stream in startup window; no press")
                if self.state == "moving" and not fresh:
                    raise RuntimeError("CH1 telemetry is stale during motion")
                if self.state == "observing":
                    if time.monotonic() - self.phase_started >= 20:
                        self.summary["stationary_observation_complete"] = True
                        self.finish()
                    return
                if not fresh or len(snapshot.get("raw_ch1_adc_codes", [])) != 45:
                    return
                codes = np.asarray(snapshot["raw_ch1_adc_codes"]).reshape(9, 5)
                if self.state == "warming" and time.monotonic() - self.phase_started >= 2:
                    if np.any(np.max(codes, axis=1) < 12) or np.any(codes >= 4090):
                        raise RuntimeError("CH1 nine-segment signal missing/too weak or saturated; no press")
                    self.begin_recording()
                elif self.state == "baseline":
                    key = snapshot["frame_key"]
                    if key != self.last_seen_key:
                        self.baseline_count += 1
                        if complete_baseline_snapshot(snapshot, full_map_reference=self.full_map_reference):
                            self.baseline_complete_maps += 1
                        self.last_seen_key = key
                    if (self.baseline_count >= 60 and self.baseline_complete_maps >= 3
                            and time.monotonic() - self.phase_started >= 3):
                        validate_start(w._last_mach3_status, self.target)
                        if self.acquisition_only:
                            self.state = "observing"
                            self.phase_started = time.monotonic()
                        else:
                            self.motion_sent = True  # Set BEFORE dispatch: never retry the press.
                            self.summary["motion_commands_started"] = 1
                            self.state = "moving"
                            self.mark("approach")
                            w.mach3_emergency.press_target(dict(self.target))
            if self.state == "moving" and self.motion_finished and not w._coverage_active:
                self.finish()
            elif self.state == "stopping" and self.motion_finished:
                self.finish()
        except Exception as exc:
            self.fail(exc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator-confirmed", action="store_true")
    parser.add_argument("--acquisition-only", action="store_true", help="Record only; never dispatch a press or other motion")
    parser.add_argument("--full-map-reference", action="store_true", help="Explicit full 45-point RC reference, not settled direct ADC or pressure bandwidth proof (firmware >=1.0.31)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.operator_confirmed:
        parser.error("Current fixture/physical emergency-stop confirmation is required")
    if args.output.exists():
        parser.error("Output must be new; an existing run is never overwritten or retried")
    load_installed_rows()  # Validate the unchanged 45-point/hidden DAC currents.
    args.output.mkdir(parents=True)
    studio.COVERAGE_POST_RELEASE_MS = 3000
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv[:1])
    studio.install_theme(app)
    window = CommissionWindow(initial_source=studio.SOURCE_LOCAL)
    window.usb_port = "COM6"
    window.usb_serial.port = "COM6"
    window.local_controller.port = "COM6"
    window.show()
    window.workspace_tabs.setCurrentIndex(1)
    window.finger_controller.press_depth_spin.setValue(0.1)
    window.commission_run = CenterRun(window, args.output.resolve(), acquisition_only=args.acquisition_only,
                                    full_map_reference=args.full_map_reference)
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
