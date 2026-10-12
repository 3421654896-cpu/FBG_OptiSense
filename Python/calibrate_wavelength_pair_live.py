"""Derive and verify local PHASE corrections for two wavelength points."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from laser_power_calibration import AQ6150B, LaserSerial, enrich_pd


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
POINTS = (
    {"target_nm": 1528.440, "codes": [58982, 56182, 3216, 2458, 17203]},
    {"target_nm": 1528.461, "codes": [58982, 55782, 2979, 2458, 17203]},
)
PHASE_PROBE_CODES = 128
MAX_PHASE_OFFSET_CODES = 512


def collect(laser: LaserSerial, meter: AQ6150B, codes: list[int], target: float,
            count: int, settle_s: float) -> list[dict]:
    laser.set_codes(codes)
    time.sleep(settle_s)
    rows = []
    for repeat in range(count):
        reading = enrich_pd(meter.measure(target, select_main_peak=True), laser)
        row = asdict(reading)
        row["repeat"] = repeat + 1
        row["error_pm"] = (reading.wavelength_nm - target) * 1000.0
        rows.append(row)
        time.sleep(0.12)
    return rows


def median_wave(samples: list[dict]) -> float:
    return statistics.median(float(row["wavelength_nm"]) for row in samples)


def summarize(samples: list[dict], target: float) -> dict:
    waves = [float(row["wavelength_nm"]) for row in samples]
    powers = [float(row["power_mw"]) for row in samples]
    smsrs = [
        99.0 if int(row["peak_count"]) == 1
        else float(row["side_mode_suppression_db"] or -999.0)
        for row in samples
    ]
    center = statistics.median(waves)
    return {
        "median_wavelength_nm": center,
        "median_error_pm": (center - target) * 1000.0,
        "span_pm": (max(waves) - min(waves)) * 1000.0,
        "median_power_mw": statistics.median(powers),
        "power_span_mw": max(powers) - min(powers),
        "minimum_smsr_db": min(smsrs),
        "single_mode_pass": min(smsrs) >= 20.0,
    }


def main() -> int:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    created = datetime.now()
    output = OUTPUTS / f"wavelength_pair_calibration_{created:%Y%m%d_%H%M%S}.json"
    payload: dict = {
        "created": created.isoformat(timespec="seconds"),
        "method": (
            "Direct five-channel PI11210 output, 1 s settling; derive a local "
            "PHASE slope with a +128-code probe, then verify five readings"
        ),
        "points": [],
    }
    laser = None
    meter = None
    last_codes = [0, 0, 0, 0, 0]
    try:
        meter = AQ6150B("GPIB0::7::INSTR")
        payload["meter"] = meter.identity
        laser = LaserSerial("COM6")
        for point in POINTS:
            target = float(point["target_nm"])
            base = list(point["codes"])
            base[0] = min(base[0], 58981)
            base[1] = min(base[1], 58981)
            last_codes = base

            baseline = collect(laser, meter, base, target, 3, 1.0)
            probe = list(base)
            probe[2] += PHASE_PROBE_CODES
            probe_samples = collect(laser, meter, probe, target, 3, 1.0)
            slope_pm_per_code = (
                (median_wave(probe_samples) - median_wave(baseline)) * 1000.0
                / PHASE_PROBE_CODES
            )
            if not (-0.10 <= slope_pm_per_code <= -0.002):
                raise RuntimeError(
                    f"{target:.3f}: implausible PHASE slope {slope_pm_per_code:.6f} pm/code"
                )

            base_error_pm = (median_wave(baseline) - target) * 1000.0
            correction = round(-base_error_pm / slope_pm_per_code)
            correction = max(-MAX_PHASE_OFFSET_CODES, min(MAX_PHASE_OFFSET_CODES, correction))
            candidate = list(base)
            candidate[2] += correction
            candidate[2] = max(0, min(32767, candidate[2]))
            verified = collect(laser, meter, candidate, target, 5, 1.0)
            verified_summary = summarize(verified, target)

            # One small residual correction is allowed, using the measured
            # local slope.  This stays inside the same +/-512-code envelope.
            residual_pm = float(verified_summary["median_error_pm"])
            if abs(residual_pm) > 0.35:
                extra = round(-residual_pm / slope_pm_per_code)
                total = max(
                    -MAX_PHASE_OFFSET_CODES,
                    min(MAX_PHASE_OFFSET_CODES, candidate[2] + extra - base[2]),
                )
                candidate[2] = base[2] + total
                verified = collect(laser, meter, candidate, target, 5, 1.0)
                verified_summary = summarize(verified, target)

            last_codes = candidate
            result = {
                "target_nm": target,
                "original_codes": list(point["codes"]),
                "applied_safe_original_codes": base,
                "baseline": baseline,
                "baseline_summary": summarize(baseline, target),
                "phase_probe_codes": probe,
                "phase_probe_samples": probe_samples,
                "phase_slope_pm_per_code": slope_pm_per_code,
                "calibrated_codes": candidate,
                "verified_samples": verified,
                "verified_summary": verified_summary,
                "pass": (
                    abs(float(verified_summary["median_error_pm"])) <= 0.5
                    and float(verified_summary["span_pm"]) <= 1.0
                    and bool(verified_summary["single_mode_pass"])
                ),
            }
            payload["points"].append(result)
            output.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                f"target={target:.3f} original_phase={base[2]} "
                f"slope={slope_pm_per_code:.6f} pm/code "
                f"calibrated_phase={candidate[2]} "
                f"wave={verified_summary['median_wavelength_nm']:.8f} nm "
                f"error={verified_summary['median_error_pm']:+.3f} pm "
                f"span={verified_summary['span_pm']:.3f} pm "
                f"power={verified_summary['median_power_mw']:.4f} mW "
                f"SMSR>={verified_summary['minimum_smsr_db']:.2f} dB "
                f"pass={result['pass']}",
                flush=True,
            )
    finally:
        if laser is not None:
            try:
                laser.set_codes([0, 0, *last_codes[2:]])
                time.sleep(0.10)
            finally:
                laser.close()
        if meter is not None:
            meter.close()

    payload["all_pass"] = all(bool(row["pass"]) for row in payload["points"])
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"REPORT={output}", flush=True)
    return 0 if payload["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
