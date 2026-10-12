"""Bounded machine-3 SOA transfer check against the AQ6150B.

This is a diagnostic, not calibration. It keeps the non-controlled currents
fixed, tests short holds, and requests all-zero DAC codes plus the shutter on exit.
An acknowledged DAC code is not treated as measured electrical current.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

import serial
from serial.tools import list_ports
from pyocd.core.helpers import ConnectHelper
from pyocd.core.target import Target

import app_JDSU as app
from benchmark_stress_realtime import _close_shutter, _safe_disarm
from fullband_equal_power_model import current_to_code
from guarded_application_upgrade import zero_all_dacs_and_verify
from laser_power_calibration import AQ6150B


EXPECTED_SERIAL = "207933893232"
STEPS_MA = (20.0, 70.0, 120.0, 20.0)
FIXED_MA = (120.0, 5.0, 5.0, 5.0)  # GAIN, PHASE, WAVE_A, WAVE_B


def capture_pd_medians(device, seconds=0.25):
    device.reset_input_buffer()
    rx = bytearray()
    rows = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        data = device.read(min(max(int(device.in_waiting or 0), 20), 4096))
        if not data:
            continue
        rx.extend(data)
        for frame in app.extract_single_value_frames(rx):
            values = app.decode_single_value_monitor_frame(frame)
            if values is not None:
                rows.append(values)
    if not rows:
        return {"frames": 0}
    return {"frames": len(rows),
            "pdt_code_median": statistics.median(row[0] for row in rows),
            "pdr_code_median": statistics.median(row[1] for row in rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM18")
    parser.add_argument("--meter", default="GPIB0::7::INSTR")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", choices=("soa", "gain"), default="soa")
    parser.add_argument("--fixed-gain-shutter", action="store_true")
    parser.add_argument("--probe", help="Exact debug-probe ID for read-only PB5 sampling")
    parser.add_argument("--dark-only", action="store_true",
                        help="Check the all-zero/shutter optical baseline and exit")
    parser.add_argument("--target-hold-seconds", type=float, default=0.0,
                        help="Keep the 120/120/5/5/5 mA point for a bounded manual check")
    args = parser.parse_args()
    if not 0.0 <= args.target_hold_seconds <= 300.0:
        parser.error("target hold must be 0..300 seconds")
    if args.output.exists():
        parser.error("output already exists")
    matches = [port for port in list_ports.comports()
               if port.device.upper() == args.port.upper()
               and port.vid == 0x0483 and port.pid == 0x5740
               and port.serial_number == EXPECTED_SERIAL]
    if len(matches) != 1:
        raise RuntimeError("machine-3 USB identity mismatch; no current commands sent")

    report = {
        "schema": "machine3_soa_optical_transfer_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device_serial": EXPECTED_SERIAL,
        "port": args.port,
        "meter": args.meter,
        "controlled_channel": args.control,
        "fixed_current_ma": (
            {"gain": 120.0, "phase": 5.0, "wave_a": 5.0, "wave_b": 5.0}
            if args.control == "soa" else
            {"soa": 120.0, "phase": 5.0, "wave_a": 5.0, "wave_b": 5.0}
        ),
        "steps_ma": ((0.0, *STEPS_MA, 0.0)
                     if args.control == "soa" else STEPS_MA),
        "measurements": [],
        "target_hold_seconds_requested": args.target_hold_seconds,
        "complete": False,
    }
    device = None
    meter = None
    session = None
    target = None
    cleanup = {}
    try:
        device = serial.Serial(args.port, 2_000_000, timeout=0.03, write_timeout=3)
        device.dtr = True
        _safe_disarm(device)
        zero_all_dacs_and_verify(device)
        _close_shutter(device)
        meter = AQ6150B(args.meter, average_count=1)
        report["meter_identity"] = meter.identity
        report["meter_profile"] = meter.measurement_profile
        if args.probe:
            session = ConnectHelper.session_with_chosen_probe(
                blocking=False, unique_id=args.probe,
                options={"target_override": "stm32f205re", "connect_mode": "attach",
                         "auto_unlock": False, "resume_on_disconnect": False,
                         "no_config": True, "frequency": 1000000,
                         "cache.enable_memory": False, "cache.enable_register": False})
            if session is None or session.probe.unique_id != args.probe:
                raise RuntimeError("debug probe identity mismatch")
            session.open()
            target = session.target
            if target.get_state() != Target.State.RUNNING:
                raise RuntimeError("target not running; no optical commands sent")
        try:
            off = meter.measure(1535.0, select_main_peak=True)
            report["shutter_baseline"] = asdict(off)
            print(f"shutter_baseline power_dbm={off.power_dbm:.3f} "
                  f"wavelength_nm={off.wavelength_nm:.6f}", flush=True)
        except RuntimeError as exc:
            report["shutter_baseline_error"] = str(exc)
            print(f"shutter_baseline_error={exc}", flush=True)
        if args.dark_only:
            report["complete"] = True
            return 0
        worker = app.EqualIntervalWorker(device, (), {}, settle_s=0.0,
                                         feedback_selectors=(0, 2))
        if args.control == "soa":
            report["gain_shutter_ramp"] = []
            for gain_ma in (20.0, 70.0, 120.0):
                ramp_codes = tuple(current_to_code(ma, channel)
                                   for channel, ma in enumerate(
                                       (gain_ma, 0.0, 5.0, 5.0, 5.0)))
                if worker._command_and_ack(
                        ramp_codes, app.PI11210_SOA_SHUTTER_MODE) is None:
                    raise RuntimeError("gain shutter ramp had no acknowledgement")
                report["gain_shutter_ramp"].append(
                    {"gain_ma_requested": gain_ma, "codes": ramp_codes,
                     "dac_ack": True})
                print(f"gain_shutter_ramp gain_ma={gain_ma:g} ack=True", flush=True)
                time.sleep(0.2)
        if args.fixed_gain_shutter:
            shutter_codes = tuple(current_to_code(ma, channel)
                                  for channel, ma in enumerate(
                                      (120.0, 0.0, 5.0, 5.0, 5.0)))
            if worker._command_and_ack(
                    shutter_codes, app.PI11210_SOA_SHUTTER_MODE) is None:
                raise RuntimeError("fixed-gain SOA shutter had no acknowledgement")
            if target is not None:
                report["fixed_gain_shutter_pb5"] = {
                    "gpio_odr_bit": (target.read32(0x40020414) >> 5) & 1,
                    "gpio_idr_bit": (target.read32(0x40020410) >> 5) & 1,
                }
                print(f"fixed_gain_shutter_pb5={report['fixed_gain_shutter_pb5']}",
                      flush=True)
            time.sleep(0.5)
            report["fixed_gain_shutter_pd"] = capture_pd_medians(device)
            try:
                shutter_reading = meter.measure(1535.0, select_main_peak=True)
                report["fixed_gain_shutter_reading"] = asdict(shutter_reading)
                print(f"fixed_gain_shutter power_dbm={shutter_reading.power_dbm:.3f} "
                      f"wavelength_nm={shutter_reading.wavelength_nm:.6f}", flush=True)
            except RuntimeError as exc:
                report["fixed_gain_shutter_error"] = str(exc)
                print(f"fixed_gain_shutter_error={exc}", flush=True)
        for index, controlled_ma in enumerate(report["steps_ma"], 1):
            currents = ((FIXED_MA[0], controlled_ma, *FIXED_MA[1:])
                        if args.control == "soa" else
                        (controlled_ma, 120.0, *FIXED_MA[1:]))
            codes = tuple(current_to_code(ma, channel)
                          for channel, ma in enumerate(currents))
            if worker._command_and_ack(codes) is None:
                raise RuntimeError(f"step {index}: no matching DAC acknowledgement")
            time.sleep(0.5)
            entry = {"step": index, "controlled_ma_requested": controlled_ma,
                     "codes": codes, "dac_ack": True}
            entry["pd"] = capture_pd_medians(device)
            print(f"step={index} pd={entry['pd']}", flush=True)
            if target is not None:
                entry["pb5"] = {
                    "gpio_odr_bit": (target.read32(0x40020414) >> 5) & 1,
                    "gpio_idr_bit": (target.read32(0x40020410) >> 5) & 1,
                }
                print(f"step={index} pb5={entry['pb5']}", flush=True)
            try:
                reading = meter.measure(1535.0, select_main_peak=True)
                entry["reading"] = asdict(reading)
                print(f"step={index} {args.control}_ma={controlled_ma:g} "
                      f"wavelength_nm={reading.wavelength_nm:.6f} "
                      f"power_dbm={reading.power_dbm:.3f} "
                      f"peak_count={reading.peak_count}", flush=True)
            except RuntimeError as exc:
                entry["measurement_error"] = str(exc)
                print(f"step={index} {args.control}_ma={controlled_ma:g} "
                      f"measurement_error={exc}", flush=True)
            report["measurements"].append(entry)
            if (args.control == "soa" and controlled_ma == 120.0
                    and args.target_hold_seconds > 0):
                print(json.dumps({"event": "TARGET_HOLD_READY",
                                  "currents_ma": currents,
                                  "auto_off_seconds": args.target_hold_seconds}),
                      flush=True)
                hold_start = time.monotonic()
                next_ack = hold_start + 5.0
                while time.monotonic() - hold_start < args.target_hold_seconds:
                    time.sleep(0.2)
                    if time.monotonic() >= next_ack:
                        if worker._command_and_ack(codes) is None:
                            raise RuntimeError("target hold lost matching DAC ACK")
                        next_ack = time.monotonic() + 5.0
                report["target_hold_seconds_actual"] = round(
                    time.monotonic() - hold_start, 3)
        report["complete"] = True
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if device is not None:
            for label, action in (("disarm_ack", _safe_disarm),
                                  ("all_zero_ack", zero_all_dacs_and_verify),
                                  ("shutter_ack", _close_shutter)):
                try:
                    action(device)
                    cleanup[label] = True
                except Exception as exc:
                    cleanup[label] = f"{type(exc).__name__}: {exc}"
            device.close()
            if any(value is not True for value in cleanup.values()):
                try:
                    matches = [port for port in list_ports.comports()
                               if port.device.upper() == args.port.upper()
                               and port.vid == 0x0483 and port.pid == 0x5740
                               and port.serial_number == EXPECTED_SERIAL]
                    if len(matches) != 1:
                        raise RuntimeError("machine-3 USB identity unavailable")
                    with serial.Serial(args.port, 2_000_000, timeout=0.03,
                                       write_timeout=3) as recovery:
                        recovery.dtr = True
                        _safe_disarm(recovery)
                        zero_all_dacs_and_verify(recovery)
                        _close_shutter(recovery)
                    cleanup["reconnected_safety_cleanup"] = True
                except Exception as exc:
                    cleanup["reconnected_safety_cleanup"] = (
                        f"{type(exc).__name__}: {exc}")
        if meter is not None:
            meter.close()
        if session is not None:
            session.close()
        report["safety_cleanup"] = cleanup
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        print(f"safety_cleanup={cleanup}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
