"""Run one explicitly confirmed, guarded CH1 silicone coverage acquisition.

The runner uses the same unified application, calibrated silicone boundary and
Mach3 service as the interactive UI.  The spindle is never enabled: every
point is approached at safe Z after an M5 command, pressed at low Z feed, held,
and retracted before the next XY move.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtWidgets

import app_JDSU as local
import fbg_unified_app as studio
from capture_ch1_rc_diagnostic import load_installed_rows
from mechanical_finger_3d import MAX_PRESS_DEPTH_MM
from stress_multirate_protocol import MultirateConfiguration


def validate_machine(status: dict, calibration_offset: list[float], contact_z: float) -> None:
    if not status.get("connected") or status.get("moving") or not status.get("stopped"):
        raise RuntimeError("Mach3 must be connected and stationary before coverage")
    if status.get("estop") or status.get("estop_led"):
        raise RuntimeError("E-stop is latched; automatic reset is not permitted")
    if not all(status.get(f"{axis}_homed") for axis in "xyz"):
        raise RuntimeError("XYZ homing is not confirmed")
    if any(
        status.get(f"{axis}_{direction}_limit")
        for axis in "xyz"
        for direction in ("pos", "neg")
    ):
        raise RuntimeError("A hardware limit is active")
    machine_values = [float(status[f"machine_{axis}"]) for axis in "xyz"]
    work_values = [float(status[f"work_{axis}"]) for axis in "xyz"]
    if not all(math.isfinite(value) for value in machine_values + work_values):
        raise RuntimeError("Machine/work coordinates are unavailable")
    contact_machine_z = float(contact_z) + float(calibration_offset[2])
    if machine_values[2] < contact_machine_z + 3.9:
        raise RuntimeError("Initial 4 mm clear height is not confirmed")


class CoverageWindow(studio.UnifiedFbgStudio):
    def closeEvent(self, event):
        if self.mach3_emergency._busy:
            self.mach3_emergency.emergency_stop()
            event.ignore()
            return
        super().closeEvent(event)


class CoverageRun(QtCore.QObject):
    def __init__(
        self,
        window: CoverageWindow,
        summary_path: Path,
        *,
        depth_mm: float,
        spacing_mm: float,
        dwell_seconds: float,
        schedule_version: int = 4,
        map_period: int = 3,
    ):
        super().__init__(window)
        self.window = window
        self.summary_path = summary_path
        self.depth_mm = float(depth_mm)
        self.spacing_mm = float(spacing_mm)
        self.dwell_seconds = float(dwell_seconds)
        self.schedule_version = int(schedule_version)
        self.map_period = int(map_period)
        self.started = time.monotonic()
        self.phase_started = self.started
        self.state = "startup"
        self.dataset_path: Path | None = None
        self.failure: str | None = None
        self.last_report = 0.0
        self.summary = {
            "purpose": "one physical CH1 silicone coverage repeat",
            "operator_confirmed": True,
            "depth_mm": self.depth_mm,
            "spacing_mm": self.spacing_mm,
            "dwell_seconds": self.dwell_seconds,
            "schedule_version": self.schedule_version,
            "map_period": self.map_period,
            "physical_15hz_verified": False,
            "measured_contact_area_mm2": None,
            "pressure_ground_truth": None,
        }
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(50)

    def shutdown_optics(self) -> bool:
        confirmed = False
        try:
            if local.ser_open:
                self.window.local_controller.page_peak.on_open_changed()
            if local.ser.is_open:
                local.set_soa_shutter_and_verify(local.ser)
                confirmed = True
        except Exception as exc:  # retain evidence; do not mask the motion result
            self.failure = f"{self.failure or ''}; optical shutdown: {exc}".strip("; ")
        return confirmed

    def finish(self) -> None:
        if self.state == "finished":
            return
        self.state = "finished"
        final_status = dict(self.window._last_mach3_status or {})
        session_end = None
        if self.dataset_path is not None:
            events_path = self.dataset_path / "events.jsonl"
            if events_path.exists():
                events = [json.loads(line) for line in events_path.read_text("utf-8").splitlines()]
                session_end = next(
                    (event for event in reversed(events) if event.get("kind") == "session_end"),
                    None,
                )
        optical_shutdown = self.shutdown_optics()
        self.summary.update(
            error=self.failure,
            elapsed_s=time.monotonic() - self.started,
            dataset_path=None if self.dataset_path is None else str(self.dataset_path),
            session_end=session_end,
            final_machine_status=final_status,
            optical_shutter_confirmed=optical_shutdown,
        )
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(
            json.dumps(self.summary, ensure_ascii=False, indent=2), "utf-8"
        )
        print(json.dumps(self.summary, ensure_ascii=False), flush=True)
        self.window.close()
        QtWidgets.QApplication.instance().quit()

    def fail(self, exc: Exception) -> None:
        self.failure = str(exc)
        print(f"coverage_failure={self.failure}", flush=True)
        if self.window._coverage_active:
            self.window._coverage_cancel_after_current = True
            if not getattr(self.window, "_coverage_stop_reason", None):
                self.window._coverage_stop_reason = f"commission_failure: {exc}"
            if self.window.mach3_emergency._busy:
                self.window.mach3_emergency.emergency_stop()
                return
            self.window._finish_coverage_run(False, self.failure)
        self.finish()

    def start_coverage(self, snapshot: dict) -> None:
        w = self.window
        finger = w.finger_controller
        finger.coverage_spacing_spin.setValue(self.spacing_mm)
        finger.coverage_repeats_spin.setValue(1)
        finger.coverage_dwell_spin.setValue(self.dwell_seconds)
        finger.press_depth_spin.setValue(self.depth_mm)
        finger.xy_speed_spin.setValue(60.0)
        finger.press_speed_spin.setValue(5.0)
        payload = finger._coverage_request_payload(
            repeats=1,
            dwell_seconds=self.dwell_seconds,
            motion_test=False,
        )
        if payload is None or not payload.get("points"):
            raise RuntimeError("Calibrated silicone coverage plan is empty")
        offset = payload.get("calibration_machine_minus_work")
        if offset is None or len(offset) != 3:
            raise RuntimeError("Verified machine/work calibration offset is missing")
        contact_z = float(payload["points"][0]["contact_z"])
        validate_machine(dict(w._last_mach3_status), list(offset), contact_z)
        if len(snapshot.get("raw_ch1_adc_codes", ())) != 45:
            raise RuntimeError("CH1 45-point frame is unavailable")
        codes = np.asarray(snapshot["raw_ch1_adc_codes"], dtype=float).reshape(9, 5)
        if np.any(np.max(codes, axis=1) < 12) or np.any(codes >= 4090):
            raise RuntimeError("CH1 nine-segment signal is missing, too weak or saturated")

        original_warning = QtWidgets.QMessageBox.warning
        try:
            QtWidgets.QMessageBox.warning = lambda *_args, **_kwargs: QtWidgets.QMessageBox.Yes
            w._finger_coverage_requested(payload)
        finally:
            QtWidgets.QMessageBox.warning = original_warning
        if not w._coverage_active or w._coverage_dataset_path is None:
            raise RuntimeError(f"Coverage did not start: {finger.coverage_summary_label.text()}")
        self.dataset_path = Path(w._coverage_dataset_path)
        self.summary.update(
            requested_points=len(payload["points"]),
            dataset_path=str(self.dataset_path),
            start_machine_status=dict(w._last_mach3_status),
        )
        print(
            f"coverage_started points={len(payload['points'])} depth_mm={self.depth_mm:.2f} "
            f"dataset={self.dataset_path}",
            flush=True,
        )
        self.state = "running"

    def tick(self) -> None:
        try:
            w = self.window
            if time.monotonic() - self.started > 900 and self.state != "finished":
                raise RuntimeError("Coverage commissioning deadline exceeded")
            if self.state == "startup":
                if not w._last_mach3_status or not local.ser.is_open:
                    return
                page = w.local_controller.page_peak
                for button, selector in zip(page.stress_feedback_buttons, (0, 2)):
                    button.setProperty("feedbackSelector", selector)
                page._update_stress_feedback_controls()
                page.requested_multirate_version = self.schedule_version
                page.multirate_configuration = MultirateConfiguration(
                    map_period_frames=self.map_period,
                    activate_threshold_codes=40,
                    release_threshold_codes=20,
                )
                w.local_controller.start_stress_debug()
                w._sync_local_mode()
                self.phase_started = time.monotonic()
                self.state = "warming"
                return
            if self.state == "warming":
                snapshot = w._active_spectrum_snapshot()
                received = snapshot.get("frame_received_monotonic_ns")
                fresh = received is not None and 0 <= time.monotonic_ns() - received < 500_000_000
                if fresh and time.monotonic() - self.phase_started >= 3.0:
                    self.start_coverage(snapshot)
                    return
                if time.monotonic() - self.phase_started > 15.0:
                    raise RuntimeError("No sustained fresh CH1 stream; no coverage motion sent")
            elif self.state == "running":
                status = dict(w._last_mach3_status or {})
                if status.get("estop") or status.get("estop_led"):
                    raise RuntimeError("E-stop became active during coverage")
                if any(
                    status.get(f"{axis}_{direction}_limit")
                    for axis in "xyz"
                    for direction in ("pos", "neg")
                ):
                    raise RuntimeError("A hardware limit became active during coverage")
                snapshot = w._active_spectrum_snapshot()
                received = snapshot.get("frame_received_monotonic_ns")
                if received is None or time.monotonic_ns() - received >= 500_000_000:
                    w._coverage_cancel_after_current = True
                    if not getattr(w, "_coverage_stop_reason", None):
                        age_ms = None if received is None else (time.monotonic_ns() - received) / 1_000_000
                        native_last = local.frames_queue.stats().last_received_ns
                        native_age_ms = None if native_last is None else (time.monotonic_ns() - native_last) / 1_000_000
                        w._coverage_stop_reason = (
                            f"spectrum_timeout: snapshot_age_ms={age_ms}; "
                            f"native_age_ms={native_age_ms}; limit_ms=500"
                        )
                        print(w._coverage_stop_reason, flush=True)
                if time.monotonic() - self.last_report >= 10.0:
                    print(
                        json.dumps(
                            {
                                "completed": w._coverage_completed,
                                "total": w._coverage_total,
                                "phase": w._coverage_phase,
                                "records": w._coverage_records,
                                "work_xyz": [
                                    status.get("work_x"),
                                    status.get("work_y"),
                                    status.get("work_z"),
                                ],
                                "moving": status.get("moving"),
                            }
                        ),
                        flush=True,
                    )
                    self.last_report = time.monotonic()
                if not w._coverage_active:
                    self.finish()
        except Exception as exc:
            self.fail(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator-confirmed", action="store_true")
    parser.add_argument("--depth-mm", type=float, required=True)
    parser.add_argument("--spacing-mm", type=float, default=4.0)
    parser.add_argument("--dwell-seconds", type=float, default=1.0)
    parser.add_argument("--schedule-version", type=int, choices=(4, 5), default=4)
    parser.add_argument("--map-period", type=int, choices=(0, 3), default=3)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if not args.operator_confirmed:
        parser.error("Current fixture and physical emergency-stop confirmation is required")
    if not math.isfinite(args.depth_mm) or not 0.0 <= args.depth_mm <= MAX_PRESS_DEPTH_MM:
        parser.error(f"Depth must be within 0..{MAX_PRESS_DEPTH_MM:.2f} mm")
    if not 2.0 <= args.spacing_mm <= 8.0:
        parser.error("Spacing must be within 2..8 mm")
    if not 0.5 <= args.dwell_seconds <= 5.0:
        parser.error("Dwell must be within 0.5..5 seconds")
    if args.summary.exists():
        parser.error("Summary path must be new; physical runs are never overwritten")
    load_installed_rows()
    studio.COVERAGE_POST_RELEASE_MS = 1000
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv[:1])
    studio.install_theme(app)
    window = CoverageWindow(initial_source=studio.SOURCE_LOCAL)
    window.usb_port = "COM6"
    window.usb_serial.port = "COM6"
    window.local_controller.port = "COM6"
    window.show()
    window.workspace_tabs.setCurrentIndex(1)
    window.commission_run = CoverageRun(
        window,
        args.summary.resolve(),
        depth_mm=args.depth_mm,
        spacing_mm=args.spacing_mm,
        dwell_seconds=args.dwell_seconds,
        schedule_version=args.schedule_version,
        map_period=args.map_period,
    )
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
