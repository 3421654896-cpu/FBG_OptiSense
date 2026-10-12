"""One guarded coverage contact with a bounded fast CH1 stream.

This is a commissioning capture, not proof of 15 Hz pressure bandwidth,
position accuracy or physical contact area.  The spindle is never enabled.
The 8 mm face moves only at Z=+4 mm to one prevalidated silicone-coverage
target, then (with the optical stream off) approaches to Z=+0.5 mm.  The local
baseline is learned there before the explicitly bounded contact descent.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import threading
import time

import serial

from capture_ch1_rc_diagnostic import load_installed_rows
from capture_ch1_teacher_spectrum import save_checkpoint
from benchmark_stress_realtime import _safe_disarm, _close_shutter
from capture_fast_flank_stream import execute_device as execute_flank_device
from capture_fast_fullmap_stream import execute_device as execute_fullmap_device
from capture_fast_triplet_stream import execute_device as execute_triplet_device
from capture_candidate27_stream import execute_device as execute_candidate_device, prepare_capture
from capture_candidate27_guarded_stream import (
    execute_device as execute_guarded_candidate_device,
    prepare_capture as prepare_guarded_candidate_capture,
)
from fast_flank_realtime import FastFlankRealtimeLocalizer
from mach3_controller import Mach3CommandError, Mach3Controller
from mechanical_finger_3d import (
    SILICONE_Y_CENTER,
    generate_silicone_coverage_plan,
    load_finger_layout,
    load_machine_touch_samples,
    MAX_PRESS_DEPTH_MM,
    MODEL_X_HALF_SPAN_MM,
    MODEL_Y_HALF_SPAN_MM,
)


TRIGGER_SEQUENCE = 96
SAFE_CLEARANCE_MM = 4.0
APPROACH_CLEARANCE_MM = 0.5
XY_TOLERANCE_MM = 0.01
OFFSET_TOLERANCE_MM = 0.05
POSITION_TOLERANCE_MM = 0.003
CONTACT_DESCENT_STEP_MM = 0.5


def open_verified_optical_port(port):
    """Acquire and verify the optical channel before any positioning command."""
    device = serial.Serial(port, 2_000_000, timeout=0.03, write_timeout=3)
    try:
        if not _safe_disarm(device) or not _close_shutter(device):
            raise RuntimeError('Pre-motion disarm/shutter acknowledgment missing')
        return device
    except BaseException:
        device.close()
        raise


def move_contact_descent(controller: Mach3Controller, contact_delta: float) -> dict:
    """Move to the press plane as verified sub-millimetre Z steps."""
    steps = controller.move_relative_staged(
        "Z",
        float(contact_delta),
        step_mm=CONTACT_DESCENT_STEP_MM,
        feed_mm_min=30.0,
        safe_zone_confirmed=True,
    )
    if not steps:
        raise Mach3CommandError("Contact descent returned no verified motion steps")
    return dict(steps[-1])


def coverage_target(point_index: int, depth_mm: float = 0.0) -> dict:
    depth_mm = float(depth_mm)
    if not math.isfinite(depth_mm) or not 0.0 <= depth_mm <= MAX_PRESS_DEPTH_MM:
        raise ValueError(f"Depth must be within 0..{MAX_PRESS_DEPTH_MM:.2f} mm")
    samples = load_machine_touch_samples()
    layout = load_finger_layout()
    plan = generate_silicone_coverage_plan(samples, layout.sensors, spacing_mm=4.0)
    matches = [
        dict(point)
        for point in plan
        if int(point["point_index"]) == int(point_index)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"The 4 mm silicone plan has no unique point {point_index}")
    offsets = [
        (
            float(sample.machine_x) - float(sample.work_x),
            float(sample.machine_y) - float(sample.work_y),
            float(sample.machine_z) - float(sample.work_z),
        )
        for sample in samples
        if None
        not in (
            sample.machine_x,
            sample.machine_y,
            sample.machine_z,
            sample.work_x,
            sample.work_y,
            sample.work_z,
        )
    ]
    if len(offsets) < 2:
        raise RuntimeError("Two active machine/work calibration anchors are required")
    target = matches[0]
    target.update(
        depth_mm=depth_mm,
        safe_z=float(matches[0]["contact_z"]) + SAFE_CLEARANCE_MM,
        approach_z=float(matches[0]["contact_z"]) + APPROACH_CLEARANCE_MM,
        press_z=float(matches[0]["contact_z"]) - depth_mm,
        calibration_machine_minus_work=[
            statistics.median(axis) for axis in zip(*offsets, strict=True)
        ],
    )
    return target


def resolve_work_target(target: dict, status: dict) -> dict:
    """Preserve calibrated physical XYZ while resolving today's work zero.

    Freeze the live offset for this one run. validate_status continues to reject
    any subsequent offset change rather than following it during motion.
    """
    original = dict(target)
    expected = [float(v) for v in original['calibration_machine_minus_work']]
    live = [float(status[f'machine_{a}'])-float(status[f'work_{a}']) for a in 'xyz']
    physical = [float(original['work_x'])+expected[0],
                float(original['work_y'])+expected[1],
                float(original['contact_z'])+expected[2]]
    if not all(math.isfinite(v) for v in expected+live+physical):
        raise ValueError('Finite calibration and live coordinates required')
    target = dict(original)
    target.update(calibration_work_xy_mm=[original['work_x'],original['work_y']],
                  original_calibration_machine_minus_work=expected,
                  target_machine_xyz_mm=physical,
                  calibration_machine_minus_work=live,
                  work_x=physical[0]-live[0],work_y=physical[1]-live[1])
    for field in ('contact_z','safe_z','approach_z','press_z'):
        target[field]=float(original[field])+expected[2]-live[2]
    return target


def validate_status(
    status: dict,
    target: dict,
    *,
    require_target_xy: bool = True,
    required_z: float | None = None,
) -> None:
    if not status.get("connected") or not status.get("stopped") or status.get("moving"):
        raise RuntimeError("Mach3 must be connected and stationary")
    if status.get("estop") or status.get("estop_led"):
        raise RuntimeError("Mach3 Reset/E-stop is active")
    if not all(status.get(f"{axis}_homed") for axis in "xyz"):
        raise RuntimeError("All three axes must be homed")
    if any(
        status.get(f"{axis}_{direction}_limit")
        for axis in "xyz"
        for direction in ("pos", "neg")
    ):
        raise RuntimeError("A hardware limit is active")
    expected = target["calibration_machine_minus_work"]
    for axis, expected_value in zip("xyz", expected, strict=True):
        actual = float(status[f"machine_{axis}"]) - float(status[f"work_{axis}"])
        if not math.isfinite(actual) or abs(actual - float(expected_value)) > OFFSET_TOLERANCE_MM:
            raise RuntimeError("Machine/work offset differs from the applied finger calibration")
    if require_target_xy:
        error = max(
            abs(float(status["work_x"]) - float(target["work_x"])),
            abs(float(status["work_y"]) - float(target["work_y"])),
        )
        if error > XY_TOLERANCE_MM:
            raise RuntimeError(f"Tool is not at the calibrated coverage target ({error:.4f} mm)")
    elif max(
        abs(float(status["work_x"]) - float(target["work_x"])),
        abs(float(status["work_y"]) - float(target["work_y"])),
    ) > 10.0:
        raise RuntimeError("Coverage target is more than 10 mm from the current safe-height position")
    expected_z = float(target["safe_z"] if required_z is None else required_z)
    if abs(float(status["work_z"]) - expected_z) > POSITION_TOLERANCE_MM:
        raise RuntimeError(f"Tool is not at the required Z={expected_z:.3f} mm plane")


def append_event(events: list, lock: threading.Lock, kind: str, status: dict | None = None):
    with lock:
        events.append(
            {
                "kind": kind,
                "host_monotonic_ns": time.monotonic_ns(),
                "status": dict(status or {}),
            }
        )


def emergency_stop_separate_channel() -> dict:
    with Mach3Controller() as emergency:
        return emergency.emergency_stop()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator-confirmed", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", default="COM6")
    parser.add_argument(
        "--protocol",
        choices=("flank18", "flank18_fast", "triplet27", "fullmap45", "candidate27", "candidate27_guarded"),
        default="flank18",
        help="independent 18-, 27-, or all-fresh 45-point qualification stream",
    )
    parser.add_argument(
        "--spacing-us",
        type=int,
        default=50,
        help="F45 or explicit fast F18 pair spacing; ignored by legacy flank18/triplet27",
    )
    parser.add_argument("--cycles", type=int, default=1024)
    parser.add_argument('--ch1-selector', type=int, choices=range(4), default=2)
    parser.add_argument('--baseline-start-sequence', type=int, choices=(16,128,256), default=16)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--candidate-probe', type=Path)
    parser.add_argument('--first-delay-us', type=int, default=400)
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument(
        "--depth-mm",
        type=float,
        default=0.0,
        help=f"indentation below the calibrated contact plane (0..{MAX_PRESS_DEPTH_MM:.2f} mm)",
    )
    parser.add_argument(
        "--point-index",
        type=int,
        default=9,
        help="1..17 in the calibrated 4 mm full-face-inside-silicone plan; 9 is centre",
    )
    args = parser.parse_args(argv)
    if args.ch1_selector != 2 and args.protocol != 'fullmap45':
        parser.error('Explicit gain is only supported by F45 qualification')
    if args.protocol == 'flank18_fast':
        from f18_spacing_protocol import command as fast18_command
        fast18_command(args.cycles,1,args.spacing_us)
        if args.cycles != 2048 or args.baseline_start_sequence != 256 or args.spacing_us != 50:
            parser.error('Fast F18 initial commissioning requires 2048 cycles, baseline start 256 and 50 us spacing')
    if args.baseline_start_sequence != 16 and args.protocol not in ('fullmap45','flank18_fast'):
        parser.error('Extended startup qualification requires F45')
    trigger_sequence = args.baseline_start_sequence + 64 + 16
    if not args.operator_confirmed:
        parser.error("Current fixture, zero plane and physical emergency stop must be confirmed")
    if args.output.exists() or args.output.with_suffix(".png").exists():
        parser.error("Output must be new; commissioning data is never overwritten")
    if args.protocol != 'flank18_fast' and not 640 <= args.cycles <= 1024:
        parser.error(
            "This bounded commissioning protocol requires 640..1024 cycles so the "
            "guarded retract completes before the optical stream ends"
        )
    if not 0.5 <= args.hold_seconds <= 3.0:
        parser.error("hold-seconds must be 0.5..3.0")

    if not 1 <= args.point_index <= 17:
        parser.error("point-index must be 1..17")
    if not math.isfinite(args.depth_mm) or not 0.0 <= args.depth_mm <= MAX_PRESS_DEPTH_MM:
        parser.error(f"depth-mm must be within 0..{MAX_PRESS_DEPTH_MM:.2f}")
    prepared = None
    if args.protocol in ('candidate27', 'candidate27_guarded'):
        if args.candidate is None or args.candidate_probe is None:
            parser.error('Candidate route and completed matching optical probe are required')
        # Optical identity and timing must fail before any machine/serial access.
        prepare = (prepare_guarded_candidate_capture
                   if args.protocol == 'candidate27_guarded' else prepare_capture)
        prepared = prepare(args.candidate,args.candidate_probe,args.cycles,
                           args.first_delay_us,args.spacing_us)
    target = coverage_target(args.point_index, args.depth_mm)
    rows, _ = load_installed_rows()
    localizer = (
        FastFlankRealtimeLocalizer(rows, baseline_frames=64)
        if args.protocol == "flank18"
        else None
    )
    execute_device = {
        "flank18": execute_flank_device,
        "flank18_fast": execute_flank_device,
        "triplet27": execute_triplet_device,
        "fullmap45": execute_fullmap_device,
        "candidate27": execute_candidate_device,
        "candidate27_guarded": execute_guarded_candidate_device,
    }[args.protocol]
    controller = Mach3Controller()
    initial = controller.status()
    target = resolve_work_target(target, initial)
    validate_status(initial, target, require_target_xy=False)
    try:
        optical_device = open_verified_optical_port(args.port)
    except BaseException:
        controller.close()
        raise

    events: list[dict] = []
    realtime: list[dict] = []
    receipts: list[dict] = []
    lock = threading.Lock()
    motion_started = threading.Event()
    motion_done = threading.Event()
    motion_failure: list[str] = []
    motion_thread: list[threading.Thread] = []
    append_event(events, lock, "validated_safe_start", initial)

    # Position with the optical stream off.  Freezing a baseline before an XY
    # move makes fixture/fibre disturbance indistinguishable from a press.
    try:
        append_event(events, lock, "xy_positioning_started_stream_off", initial)
        positioned = initial
        if max(
            abs(float(initial["work_x"]) - float(target["work_x"])),
            abs(float(initial["work_y"]) - float(target["work_y"])),
        ) > POSITION_TOLERANCE_MM:
            xy_result = controller.move_absolute_xy_continuous(
                float(target["work_x"]),
                float(target["work_y"]),
                feed_mm_min=120.0,
                safe_zone_confirmed=True,
            )
            positioned = dict(xy_result["after"])
        validate_status(positioned, target, require_target_xy=True)
        append_event(events, lock, "target_xy_reached_stream_off", positioned)

        append_event(events, lock, "approach_started_stream_off", positioned)
        approach_delta = float(target["approach_z"]) - float(positioned["work_z"])
        if abs(approach_delta) > POSITION_TOLERANCE_MM:
            approach_result = controller.move_relative_continuous_axis(
                "Z", approach_delta, feed_mm_min=60.0, safe_zone_confirmed=True
            )
            positioned = dict(approach_result["after"])
        validate_status(
            positioned,
            target,
            require_target_xy=True,
            required_z=float(target["approach_z"]),
        )
        append_event(events, lock, "baseline_approach_reached_stream_off", positioned)
    except BaseException:
        try:
            emergency_stop_separate_channel()
        finally:
            optical_device.close()
            controller.close()
        raise

    def motion():
        try:
            start = controller.status()
            validate_status(
                start,
                target,
                require_target_xy=True,
                required_z=float(target["approach_z"]),
            )
            contact_delta = float(target["press_z"]) - float(start["work_z"])
            append_event(events, lock, "contact_descent_started", start)
            # The approach plane is 0.5 mm above the calibrated silicone
            # surface.  At deeper commissioning depths the complete descent
            # therefore exceeds the controller's deliberately strict 1 mm
            # single-step guard (for example 1.1 mm at a 0.6 mm indentation).
            # Keep that guard intact and make the contact approach as
            # independently verified 0.5 mm-or-smaller moves instead.
            contact_result = move_contact_descent(controller, contact_delta)
            contact = dict(contact_result["after"])
            if abs(float(contact["work_z"]) - float(target["press_z"])) > POSITION_TOLERANCE_MM:
                raise Mach3CommandError("Commanded indentation endpoint was not reached")
            append_event(events, lock, "contact_reached_commanded_depth", contact)
            time.sleep(float(args.hold_seconds))
            append_event(events, lock, "release_started", controller.status())

            retract_delta = float(target["safe_z"]) - float(contact["work_z"])
            retract_result = controller.move_relative_continuous_axis(
                "Z", retract_delta, feed_mm_min=60.0, safe_zone_confirmed=True
            )
            retracted = dict(retract_result["after"])
            validate_status(retracted, target, require_target_xy=True)
            append_event(events, lock, "safe_height_reached", retracted)
        except BaseException as exc:
            with lock:
                motion_failure.append(f"{type(exc).__name__}: {exc}")
            try:
                stopped = emergency_stop_separate_channel()
                append_event(events, lock, "automatic_estop_after_motion_error", stopped)
            except Exception as stop_exc:
                with lock:
                    motion_failure.append(f"emergency_stop_failed: {stop_exc}")
        finally:
            motion_done.set()

    def on_frame(frame: dict):
        sequence = int(frame["sequence"])
        received_ns = time.monotonic_ns()
        receipts.append({"sequence": sequence, "host_monotonic_ns": received_ns})
        if (args.protocol == 'flank18_fast' and sequence == args.baseline_start_sequence
                and frame['cycle_start_us'] < 2_000_000):
            raise RuntimeError('Fast F18 baseline started before 2 s startup qualification; no descent')
        if sequence == trigger_sequence:
            if localizer is not None:
                localizer.freeze_for_press()
            thread = threading.Thread(target=motion, name="zero-contact-motion", daemon=False)
            motion_thread.append(thread)
            motion_started.set()
            thread.start()
        if localizer is not None:
            declared_no_contact = sequence < trigger_sequence
            result = localizer.update(frame, declared_no_contact=declared_no_contact)
            realtime.append(result.to_dict())
        if motion_failure:
            raise RuntimeError(motion_failure[0])

    report = None
    capture_error = None
    try:
        with optical_device as device:
            execute_kwargs = {
                "cycles": args.cycles,
                "on_frame": on_frame,
                "should_stop": lambda: bool(motion_failure),
            }
            if args.protocol == "fullmap45":
                execute_kwargs["spacing_us"] = args.spacing_us
                execute_kwargs["ch1_selector"] = args.ch1_selector
            elif args.protocol == 'flank18_fast':
                execute_kwargs['spacing_us'] = args.spacing_us
            elif args.protocol in ('candidate27', 'candidate27_guarded'):
                execute_kwargs.pop('cycles')
                execute_kwargs['prepared'] = prepared
            report = execute_device(device, args.output, **execute_kwargs)
        if not motion_started.is_set():
            raise RuntimeError("Capture ended before the guarded motion trigger")
        if motion_thread:
            motion_thread[0].join(timeout=3.0)
        if not motion_done.is_set():
            raise RuntimeError("Motion did not complete before the commissioning deadline")
        if motion_failure:
            raise RuntimeError(motion_failure[0])
        final = controller.status()
        validate_status(final, target, require_target_xy=True)
        append_event(events, lock, "final_safe_state_confirmed", final)
    except BaseException as exc:
        capture_error = f"{type(exc).__name__}: {exc}"
        if not motion_done.is_set():
            try:
                stopped = emergency_stop_separate_channel()
                append_event(events, lock, "automatic_estop_after_capture_error", stopped)
            except Exception as stop_exc:
                capture_error += f"; emergency_stop_failed: {stop_exc}"
    finally:
        controller.close()

    if report is None and args.output.exists():
        report = json.loads(args.output.read_text("utf-8"))
    report = dict(report or {})
    with lock:
        event_copy = list(events)
        failure_copy = list(motion_failure)
    detections = [
        item for item in realtime if item.get("contact_detected") and item.get("position", {}).get("x_mm") is not None
    ]
    target_xy = (
        float(target["model_x"]) * MODEL_X_HALF_SPAN_MM,
        float(target["model_y"]) * MODEL_Y_HALF_SPAN_MM,
    )
    errors = [
        math.hypot(
            float(item["position"]["x_mm"]) - target_xy[0],
            float(item["position"]["y_mm"]) - target_xy[1],
        )
        for item in detections
    ]
    report["commissioning"] = {
        "schema": {
            "flank18": "bounded_indentation_fast_flank_commission_v3",
            "triplet27": "bounded_indentation_fast_triplet_commission_v1",
            "fullmap45": "bounded_indentation_fast_fullmap_commission_v1",
            "candidate27": "bounded_indentation_candidate27_commission_v1",
            "candidate27_guarded": "bounded_indentation_guarded_candidate27_commission_v2",
        }[args.protocol],
        "stream_protocol": args.protocol,
        "operator_confirmed": True,
        "target": target,
        "trigger_sequence": trigger_sequence,
        "baseline_start_sequence": args.baseline_start_sequence,
        "baseline_frame_count": 64,
        "events": event_copy,
        "frame_receipts": receipts,
        "realtime_results": realtime,
        "motion_failures": failure_copy,
        "capture_error": capture_error,
        "detected_onset_frames": len(detections),
        "provisional_position_error_mm": {
            "minimum": min(errors) if errors else None,
            "median": statistics.median(errors) if errors else None,
        },
        "depth_mm": float(args.depth_mm),
        "spindle_policy": "M5 before every movement; no spindle-start command",
        "physical_15hz_verified": False,
        "position_accuracy_verified": False,
        "physical_contact_area_mm2": None,
        "nominal_tool_diameter_mm": 8.0,
        "nominal_tool_area_mm2": math.pi * 4.0 ** 2,
        "area_convention": "8 mm circular tool face; not a measurement of deformed contact area",
        "contact_area_verified": False,
        "training_eligible": False,
        "note": (
            "The optical baseline was captured at the local +0.5 mm approach plane. "
            "One known coverage-target contact is an integration observation only. "
            "Independent held-out positions, synchronized pressure excitation and "
            "traceable contact-area labels are still required."
        ),
    }
    save_checkpoint(args.output, report)
    print(
        json.dumps(
            {
                "schema": report["commissioning"]["schema"],
                "point_index": int(target["point_index"]),
                "target_work_xyz_mm": [
                    float(target["work_x"]),
                    float(target["work_y"]),
                    float(target["press_z"]),
                ],
                "capture_complete": bool(report.get("complete")),
                "captured_frames": len(report.get("frames", [])),
                "motion_failures": failure_copy,
                "capture_error": capture_error,
                "detected_onset_frames": len(detections),
                "provisional_position_error_mm": report["commissioning"][
                    "provisional_position_error_mm"
                ],
                "final_event": event_copy[-1] if event_copy else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if capture_error is None and not failure_copy else 2


if __name__ == "__main__":
    raise SystemExit(main())
