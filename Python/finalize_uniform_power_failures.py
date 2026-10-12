"""Apply live local refinements to the nine first-pass calibration failures."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from laser_power_calibration import (
    AQ6150B,
    LaserSerial,
    code_to_current,
    current_to_code,
    take_reading,
)


HERE = Path(__file__).resolve().parent
FIRST_PASS = HERE / "stress_points_shape_selected_calibrated.json"
PROPOSAL = HERE / "stress_points_shape_selected_proposal.json"
DENSE = HERE / "dense_single_reflection_135.json"
DEFAULT_OUTPUT = HERE / "stress_points_shape_selected_final.json"

# Values come from local live sweeps around each failed point.  No wavelength,
# power, or optical-mode result is inferred from the old total table.
REFINEMENTS = {
    16: {"soa_ma": 123.2, "phase_ma": 1.485},
    17: {"soa_ma": 120.5, "phase_ma": 0.095},
    40: {"soa_ma": 123.0, "phase_ma": 1.150},
    42: {"soa_ma": 114.5, "phase_ma": 2.716},
    47: {"replacement_wavelength_nm": 1551.91692, "soa_ma": 117.0, "phase_ma": 3.590},
    51: {"soa_ma": 109.0, "phase_ma": 1.385},
    52: {"soa_ma": 111.5, "phase_ma": 0.650},
    56: {"replacement_wavelength_nm": 1559.65357, "soa_ma": 128.0, "phase_ma": 0.570},
    57: {"soa_ma": 128.5, "phase_ma": 0.574},
}


def reading_smsr(reading: dict) -> float:
    if int(reading.get("peak_count", 0)) == 1 and reading.get("side_mode_suppression_db") is None:
        return 99.0
    return float(reading.get("side_mode_suppression_db") or -999.0)


def summary(rows: list[dict]) -> dict:
    powers = [float(row["final_power_mw"]) for row in rows]
    return {
        "tested": len(rows),
        "passed": sum(bool(row["passed"]) for row in rows),
        "failed": sum(not bool(row["passed"]) for row in rows),
        "power_min_mw": min(float(row["final_power_min_mw"]) for row in rows),
        "power_mean_mw": statistics.mean(powers),
        "power_max_mw": max(float(row["final_power_max_mw"]) for row in rows),
        "power_cv_percent": statistics.pstdev(powers) / statistics.mean(powers) * 100.0,
        "minimum_smsr_db": min(float(row["final_smsr_min_db"]) for row in rows),
        "maximum_wavelength_error_pm": max(
            abs(float(row["final_wavelength_error_pm"])) for row in rows
        ),
        "maximum_wavelength_span_pm": max(
            float(row["final_wavelength_span_pm"]) for row in rows
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--gpib", default="GPIB0::7::INSTR")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    first = json.loads(FIRST_PASS.read_text(encoding="utf-8"))
    proposal = json.loads(PROPOSAL.read_text(encoding="utf-8"))
    dense = json.loads(DENSE.read_text(encoding="utf-8"))
    rows = [dict(row) for row in first["rows"]]

    laser = LaserSerial(args.port)
    meter = AQ6150B(args.gpib)
    try:
        for index, refinement in REFINEMENTS.items():
            base = dict(proposal["rows"][index])
            if "replacement_wavelength_nm" in refinement:
                target = float(refinement["replacement_wavelength_nm"])
                segment = index // 7
                base = min(
                    [row for row in dense["rows"] if int(row["segment"]) == segment],
                    key=lambda row: abs(float(row["candidate_wavelength_nm"]) - target),
                )
                base = dict(base)
            target_nm = float(base["candidate_wavelength_nm"])
            codes = [int(value) for value in base["codes"]]
            codes[1] = current_to_code(float(refinement["soa_ma"]), 1)
            codes[2] = current_to_code(float(refinement["phase_ma"]), 2)
            confirmations = []
            for repeat in range(3):
                confirmations.append(asdict(take_reading(
                    laser,
                    meter,
                    codes,
                    target_nm,
                    settle_s=0.28 if repeat == 0 else 0.22,
                    select_main_peak=True,
                )))
            waves = [float(item["wavelength_nm"]) for item in confirmations]
            powers = [float(item["power_mw"]) for item in confirmations]
            smsrs = [reading_smsr(item) for item in confirmations]
            passed = bool(
                max(abs(wave - target_nm) for wave in waves) * 1000.0 <= 3.0
                and max(waves) - min(waves) <= 0.002
                and min(powers) >= 1.445
                and max(powers) <= 1.555
                and min(smsrs) >= 20.0
            )
            final_wave = statistics.median(waves)
            final_power = statistics.median(powers)
            rows[index] = {
                **base,
                "index": index,
                "segment": index // 7,
                "codes": codes,
                "currents_ma": [
                    code_to_current(code, channel) for channel, code in enumerate(codes)
                ],
                "refinement": dict(refinement),
                "confirmations": confirmations,
                "final_wavelength_nm": final_wave,
                "final_wavelength_error_pm": (final_wave - target_nm) * 1000.0,
                "final_wavelength_span_pm": (max(waves) - min(waves)) * 1000.0,
                "final_power_mw": final_power,
                "final_power_min_mw": min(powers),
                "final_power_max_mw": max(powers),
                "final_smsr_min_db": min(smsrs),
                "passed": passed,
            }
            print(
                f"refine {index:02d}: {final_wave:.6f} nm {final_power:.3f} mW "
                f"SMSR>={min(smsrs):.1f} pass={passed}",
                flush=True,
            )
    finally:
        try:
            laser.set_codes(rows[0]["codes"])
            time.sleep(0.08)
        finally:
            laser.close()
            meter.close()

    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": FIRST_PASS.name,
        "target_power_mw": 1.50,
        "limits_ma": first["limits_ma"],
        "minimum_smsr_db": 20.0,
        "rows": rows,
        "summary": summary(rows),
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    return 0 if payload["summary"]["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
