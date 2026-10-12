"""Recover equal-power rows that change longitudinal mode when SOA is reduced."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from laser_power_calibration import (
    AQ6150B, DAC_SOURCE, LaserSerial, code_to_current, current_to_code,
    parse_dac_rows, save_json, take_reading,
)

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "high_power_stress_72_wavelength_calibration_v3.json"
OUTPUT = HERE / "high_power_stress_72_gain_recovery.json"
TARGET_POWER_MW = 1.2311414157220184


def probe_gain(laser, meter, source_codes, target_nm, gain_ma):
    codes = list(source_codes)
    codes[0] = current_to_code(gain_ma, 0)
    # Preserve the freshly calibrated SOA branch at its safe ceiling.
    codes[1] = current_to_code(120.0, 1)
    reading = take_reading(laser, meter, codes, target_nm, 0.30, True)
    return codes, reading


def recover(laser, meter, record):
    source = record["final"]
    target_nm = float(record["target_nm"])
    source_codes = [int(x) for x in source["codes"]]
    history = []
    candidates = []
    initial_power = max(float(source["final"]["power_mw"]), 0.1)
    estimate = code_to_current(source_codes[0], 0) * TARGET_POWER_MW / initial_power
    gains = [estimate + offset for offset in (0.0, 6.0, -6.0, 12.0, -12.0)]
    gains += [120.0, 110.0, 100.0, 90.0, 80.0, 70.0, 60.0]
    seen = set()
    for gain in gains:
        gain = min(max(gain, 30.0), 120.0)
        key = round(gain, 2)
        if key in seen:
            continue
        seen.add(key)
        codes, reading = probe_gain(laser, meter, source_codes, target_nm, gain)
        sample = {"action": "gain_probe", "gain_ma": code_to_current(codes[0], 0),
                  "codes": codes, "reading": asdict(reading)}
        history.append(sample)
        if abs(reading.wavelength_error_pm) <= 35.0 and reading.power_mw >= 1.15:
            candidates.append(sample)
    if not candidates:
        return {"success": False, "target_nm": target_nm,
                "failure": "no_stable_gain_branch", "history": history}

    current = min(candidates, key=lambda x: abs(x["reading"]["power_mw"] - TARGET_POWER_MW))
    codes = list(current["codes"])
    gain_ma = float(current["gain_ma"])
    phase_ma = code_to_current(codes[2], 2)
    phase_slope = -75.0
    previous_phase = None
    best = current
    for iteration in range(14):
        reading_dict = current["reading"]
        power_error = float(reading_dict["power_mw"]) - TARGET_POWER_MW
        wave_error = float(reading_dict["wavelength_error_pm"])
        if abs(power_error) <= 0.015 and abs(wave_error) <= 5.0:
            return {"success": True, "target_nm": target_nm, "codes": codes,
                    "currents_ma": [code_to_current(c, i) for i, c in enumerate(codes)],
                    "final": reading_dict, "history": history}
        if abs(power_error) > 0.015:
            gain_ma *= TARGET_POWER_MW / max(float(reading_dict["power_mw"]), 0.1)
            gain_ma = min(max(gain_ma, 30.0), 120.0)
            codes[0] = current_to_code(gain_ma, 0)
        else:
            if previous_phase is not None:
                old_phase, old_error = previous_phase
                delta = phase_ma - old_phase
                if abs(delta) > 0.006:
                    slope = (wave_error - old_error) / delta
                    if -500 <= slope <= -8:
                        phase_slope = slope
            previous_phase = (phase_ma, wave_error)
            phase_ma -= wave_error / phase_slope
            phase_ma = min(max(phase_ma, code_to_current(source_codes[2], 2) - 0.5),
                           code_to_current(source_codes[2], 2) + 0.5)
            codes[2] = current_to_code(phase_ma, 2)
        reading = take_reading(laser, meter, codes, target_nm, 0.25, True)
        current = {"action": f"refine_{iteration}",
                   "gain_ma": code_to_current(codes[0], 0),
                   "codes": list(codes), "reading": asdict(reading)}
        history.append(current)
        if (abs(reading.wavelength_error_pm) <= 35.0
                and abs(reading.power_mw - TARGET_POWER_MW)
                < abs(best["reading"]["power_mw"] - TARGET_POWER_MW)):
            best = current
    reading = best["reading"]
    success = abs(reading["power_mw"] - TARGET_POWER_MW) <= 0.015 and abs(reading["wavelength_error_pm"]) <= 5.0
    return {"success": success, "target_nm": target_nm, "codes": best["codes"],
            "currents_ma": [code_to_current(c, i) for i, c in enumerate(best["codes"])],
            "final": reading, "history": history}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="COM6")
    p.add_argument("--gpib", default="GPIB0::7::INSTR")
    p.add_argument("--indices", default="18,42,48")
    args = p.parse_args()
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    indices = [int(x) for x in args.indices.split(",")]
    payload = {"created": datetime.now().isoformat(timespec="seconds"),
               "method": "preserve_soa_branch_adjust_gain_then_phase",
               "target_power_mw": TARGET_POWER_MW, "results": {}}
    rows = parse_dac_rows(DAC_SOURCE)
    park = list(rows[min(44, len(rows)-1)])
    park[0] = current_to_code(60.0, 0)
    park[1] = current_to_code(min(120.0, code_to_current(park[1], 1)), 1)
    laser, meter = LaserSerial(args.port), AQ6150B(args.gpib)
    try:
        for index in indices:
            result = recover(laser, meter, source["results"][str(index)])
            payload["results"][str(index)] = result
            save_json(OUTPUT, payload)
            f = result.get("final", {})
            print(f"index={index} success={result['success']} wave={f.get('wavelength_nm')} "
                  f"error_pm={f.get('wavelength_error_pm')} power_mw={f.get('power_mw')}", flush=True)
    finally:
        try:
            laser.set_codes(park); time.sleep(0.08)
        finally:
            laser.close(); meter.close()
    save_json(OUTPUT, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
