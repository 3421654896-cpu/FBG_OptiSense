"""Change one machine-3 laser current at a time and record AQ6150B peaks.

All requested currents are bounded below the established machine-3 limits.
DAC acknowledgement proves command acceptance, not measured electrical current.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import serial
from serial.tools import list_ports

import app_JDSU as app
from benchmark_stress_realtime import _close_shutter, _safe_disarm
from fullband_equal_power_model import LASER_MAX_MA, current_to_code
from guarded_application_upgrade import zero_all_dacs_and_verify
from laser_power_calibration import AQ6150B


EXPECTED_SERIAL = "207933893232"
BASELINE_MA = (120.0, 120.0, 5.0, 5.0, 5.0)
CHANNELS = ("GAIN", "SOA", "PHASE", "WAVE_A", "WAVE_B")
VARIANTS_MA = ((80.0, 130.0), (80.0, 130.0),
               (3.0, 7.0), (3.0, 7.0), (3.0, 7.0))


def checked_codes(currents):
    if len(currents) != 5:
        raise ValueError("exactly five currents are required")
    for name, value, limit in zip(CHANNELS, currents, LASER_MAX_MA):
        if not 0.0 <= value <= limit:
            raise ValueError(f"{name}={value} mA exceeds 0..{limit} mA")
    return tuple(current_to_code(value, channel)
                 for channel, value in enumerate(currents))


def exact_port_available(port_name):
    return len([port for port in list_ports.comports()
                if port.device.upper() == port_name.upper()
                and port.vid == 0x0483 and port.pid == 0x5740
                and port.serial_number == EXPECTED_SERIAL]) == 1


def safe_off(device):
    result = {}
    for label, action in (("disarm_ack", _safe_disarm),
                          ("all_zero_ack", zero_all_dacs_and_verify),
                          ("shutter_ack", _close_shutter)):
        try:
            action(device)
            result[label] = True
        except Exception as exc:
            result[label] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM18")
    parser.add_argument("--meter", default="GPIB0::7::INSTR")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    for channel, choices in enumerate(VARIANTS_MA):
        checked_codes(tuple(choices[0] if index == channel else value
                            for index, value in enumerate(BASELINE_MA)))
        checked_codes(tuple(choices[1] if index == channel else value
                            for index, value in enumerate(BASELINE_MA)))
    checked_codes(BASELINE_MA)
    if not exact_port_available(args.port):
        raise RuntimeError("machine-3 USB identity mismatch; no current sent")

    report = {
        "schema": "machine3_five_current_optical_one_at_a_time_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device_serial": EXPECTED_SERIAL,
        "port": args.port,
        "meter": args.meter,
        "current_limits_ma": dict(zip(CHANNELS, LASER_MAX_MA)),
        "baseline_currents_ma": dict(zip(CHANNELS, BASELINE_MA)),
        "variants_ma": dict(zip(CHANNELS, VARIANTS_MA)),
        "observations": [],
        "complete": False,
        "note": "DAC ACK is not an electrical-current readback",
    }
    device = None
    meter = None
    try:
        device = serial.Serial(args.port, 2_000_000, timeout=0.03,
                               write_timeout=3)
        device.dtr = True
        _safe_disarm(device)
        zero_all_dacs_and_verify(device)
        _close_shutter(device)
        meter = AQ6150B(args.meter, average_count=3)
        report["meter_identity"] = meter.identity
        report["meter_profile"] = meter.measurement_profile
        worker = app.EqualIntervalWorker(device, (), {}, settle_s=0.0,
                                         feedback_selectors=(0, 2))

        # Raise GAIN while the SOA hardware shutter is closed, then SOA.
        for gain_ma in (20.0, 70.0, 120.0):
            currents = (gain_ma, 0.0, 5.0, 5.0, 5.0)
            if worker._command_and_ack(
                    checked_codes(currents), app.PI11210_SOA_SHUTTER_MODE
            ) is None:
                raise RuntimeError("GAIN ramp lost DAC acknowledgement")
            print(f"GAIN shutter ramp {gain_ma:g} mA ACK", flush=True)
            time.sleep(0.2)
        for soa_ma in (20.0, 70.0, 120.0):
            currents = (120.0, soa_ma, 5.0, 5.0, 5.0)
            if worker._command_and_ack(checked_codes(currents)) is None:
                raise RuntimeError("SOA ramp lost DAC acknowledgement")
            print(f"SOA ramp {soa_ma:g} mA ACK", flush=True)
            time.sleep(0.2)

        def observe(channel_name, level_name, currents):
            codes = checked_codes(currents)
            if worker._command_and_ack(codes) is None:
                raise RuntimeError("one-at-a-time DAC command lost acknowledgement")
            time.sleep(0.5)
            row = {"channel": channel_name, "level": level_name,
                   "currents_ma": dict(zip(CHANNELS, currents)),
                   "codes": codes, "dac_ack": True, "readings": []}
            for sample in range(2):
                try:
                    row["readings"].append(asdict(
                        meter.measure(1545.0, select_main_peak=True)))
                except RuntimeError as exc:
                    row["readings"].append({"error": str(exc)})
            report["observations"].append(row)
            short = [(round(item["wavelength_nm"], 6),
                      round(item["power_dbm"], 3))
                     if "wavelength_nm" in item else item
                     for item in row["readings"]]
            print(json.dumps({"channel": channel_name, "level": level_name,
                              "current_ma": currents[CHANNELS.index(channel_name)]
                              if channel_name in CHANNELS else None,
                              "peak_wavelength_nm_power_dbm": short}),
                  flush=True)

        observe("BASELINE", "initial", BASELINE_MA)
        for channel, (low_ma, high_ma) in enumerate(VARIANTS_MA):
            for level, requested_ma in (("low", low_ma),
                                        ("return", BASELINE_MA[channel]),
                                        ("high", high_ma),
                                        ("return", BASELINE_MA[channel])):
                currents = list(BASELINE_MA)
                currents[channel] = requested_ma
                observe(CHANNELS[channel], level, tuple(currents))
        report["complete"] = True
        return 0
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        cleanup = {}
        if device is not None:
            cleanup = safe_off(device)
            device.close()
            if any(value is not True for value in cleanup.values()):
                try:
                    if not exact_port_available(args.port):
                        raise RuntimeError("machine-3 USB identity unavailable")
                    with serial.Serial(args.port, 2_000_000, timeout=0.03,
                                       write_timeout=3) as recovery:
                        recovery.dtr = True
                        second = safe_off(recovery)
                    cleanup["reconnected_safety_cleanup"] = second
                except Exception as exc:
                    cleanup["reconnected_safety_cleanup"] = (
                        f"{type(exc).__name__}: {exc}")
        if meter is not None:
            try:
                meter.instrument.write(":INIT:CONT ON")
                report["meter_repeat_restored"] = True
            except Exception as exc:
                report["meter_repeat_restored"] = f"{type(exc).__name__}: {exc}"
            meter.close()
        report["safety_cleanup"] = cleanup
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        print(json.dumps({"event": "FIVE_CHANNEL_TEST_ENDED",
                          "complete": report["complete"],
                          "safety_cleanup": cleanup,
                          "meter_repeat_restored": report.get(
                              "meter_repeat_restored")}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
