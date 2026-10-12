"""Retune only replay rows whose power drifted above the uniform band."""
from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
from laser_power_calibration import AQ6150B,LaserSerial,code_to_current,current_to_code,save_json,take_reading
HERE = Path(__file__).resolve().parent
FINAL_PATH = HERE / "stress_high_power_135_final.json"
VALIDATION_PATH = HERE / "stress_high_power_135_validation.json"


def main() -> int:
    final = json.loads(FINAL_PATH.read_text(encoding="utf-8"))
    validation = json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))
    failed = [
        index
        for index, result in enumerate(validation["results"])
        if not result["passed"]
    ]

    laser, meter = LaserSerial("COM6"), AQ6150B("GPIB0::7::INSTR")
    try:
        for index in failed:
            row = final["rows"][index]
            codes = list(row["codes"])
            soa_ma = code_to_current(codes[1], 1)
            samples = validation["results"][index]["samples"]
            old_power = sum(float(item["power_mw"]) for item in samples) / len(samples)
            soa_ma = max(70.0, min(135.0, soa_ma * 1.515 / old_power))
            codes[1] = current_to_code(soa_ma, 1)
            first = take_reading(
                laser,
                meter,
                codes,
                row["calibration"]["wavelength_nm"],
                0.30,
                True,
            )
            second = take_reading(
                laser, meter, codes, first.wavelength_nm, 0.24, True
            )
            waves = [first.wavelength_nm, second.wavelength_nm]
            powers = [first.power_mw, second.power_mw]
            smsrs = [
                99.0
                if item.peak_count == 1 and item.side_mode_suppression_db is None
                else float(item.side_mode_suppression_db or -999.0)
                for item in (first, second)
            ]
            passed = (
                max(waves) - min(waves) <= 0.006
                and min(smsrs) >= 20.0
                and min(powers) >= 1.445
                and max(powers) <= 1.555
            )
            if not passed:
                raise RuntimeError(
                    f"row {index} retune failed: {waves} {powers} {smsrs}"
                )
            row["codes"] = codes
            row["currents_ma"] = [
                code_to_current(code, channel)
                for channel, code in enumerate(codes)
            ]
            row["calibration"] = asdict(first)
            row["confirmation"] = asdict(second)
            validation["results"][index] = {
                "index": index,
                "segment": row["segment"],
                "passed": True,
                "samples": [asdict(first), asdict(second)],
                "wavelength_span_pm": (max(waves) - min(waves)) * 1000.0,
                "max_abs_from_calibration_pm": abs(
                    second.wavelength_nm - first.wavelength_nm
                )
                * 1000.0,
                "power_min_mw": min(powers),
                "power_max_mw": max(powers),
                "smsr_min_db": min(smsrs),
            }
            print(
                f"row={index} SOA={soa_ma:.3f} wave={sum(waves) / 2:.6f} "
                f"power={min(powers):.4f}..{max(powers):.4f} "
                f"SMSR={min(smsrs):.1f}"
            )
    finally:
        laser.close()
        meter.close()

    save_json(FINAL_PATH, final)
    good = sum(bool(result["passed"]) for result in validation["results"])
    validation["summary"] = {
        "tested": len(validation["results"]),
        "passed": good,
        "failed": len(validation["results"]) - good,
        "power_min_mw": min(
            result["power_min_mw"] for result in validation["results"]
        ),
        "power_max_mw": max(
            result["power_max_mw"] for result in validation["results"]
        ),
        "minimum_finite_or_single_peak_smsr_db": min(
            result["smsr_min_db"] for result in validation["results"]
        ),
        "max_wavelength_span_pm": max(
            result["wavelength_span_pm"] for result in validation["results"]
        ),
    }
    save_json(VALIDATION_PATH, validation)
    print(validation["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
