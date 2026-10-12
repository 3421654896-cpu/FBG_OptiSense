"""Live AQ6150B spot check for the two requested full-band table points."""

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

# The operational 2001-point table uses a 0.020 nm target grid.  Its
# 1528.460-nm row was historically measured as about 1528.461 nm, which is
# why both the table target and the user-visible wavelength are retained.
POINTS = (
    {
        "name": "1528.440",
        "table_target_nm": 1528.440,
        "user_reference_nm": 1528.440,
        "table_codes": [58982, 56182, 3216, 2458, 17203],
    },
    {
        "name": "1528.461",
        "table_target_nm": 1528.460,
        "user_reference_nm": 1528.461,
        "table_codes": [58982, 55782, 2979, 2458, 17203],
    },
)


def main() -> int:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    created = datetime.now()
    output = OUTPUTS / f"wavelength_pair_live_{created:%Y%m%d_%H%M%S}.json"
    payload: dict = {
        "created": created.isoformat(timespec="seconds"),
        "method": (
            "PI11210 direct five-channel DAC in EXTRA mode (fan forced full), "
            "1.0 s initial settling, then five AQ6150B triggered readings"
        ),
        "safety": {
            "gain_soa_limit_ma": 135.0,
            "gain_code_clamped_from_historical_58982_to": 58981,
            "laser_disabled_after_test": True,
        },
        "points": [],
    }

    laser = None
    meter = None
    last_safe_codes = [0, 0, 0, 0, 0]
    try:
        meter = AQ6150B("GPIB0::7::INSTR")
        payload["meter"] = meter.identity
        laser = LaserSerial("COM6")

        for point in POINTS:
            safe_codes = list(point["table_codes"])
            safe_codes[0] = min(safe_codes[0], 58981)
            safe_codes[1] = min(safe_codes[1], 58981)
            last_safe_codes = safe_codes
            laser.set_codes(safe_codes)
            time.sleep(1.0)

            samples = []
            for repeat in range(5):
                reading = enrich_pd(
                    meter.measure(
                        float(point["user_reference_nm"]), select_main_peak=True
                    ),
                    laser,
                )
                row = asdict(reading)
                row["repeat"] = repeat + 1
                row["error_vs_table_target_pm"] = (
                    reading.wavelength_nm - float(point["table_target_nm"])
                ) * 1000.0
                row["error_vs_user_reference_pm"] = (
                    reading.wavelength_nm - float(point["user_reference_nm"])
                ) * 1000.0
                samples.append(row)
                print(
                    f"{point['name']} [{repeat + 1}/5] "
                    f"{reading.wavelength_nm:.8f} nm, "
                    f"{reading.power_mw:.4f} mW, peaks={reading.peak_count}, "
                    f"SMSR={reading.side_mode_suppression_db}",
                    flush=True,
                )
                time.sleep(0.15)

            waves = [float(sample["wavelength_nm"]) for sample in samples]
            powers = [float(sample["power_mw"]) for sample in samples]
            smsrs = [
                99.0
                if int(sample["peak_count"]) == 1
                else float(sample["side_mode_suppression_db"] or -999.0)
                for sample in samples
            ]
            median_nm = statistics.median(waves)
            result = {
                **{key: value for key, value in point.items() if key != "table_codes"},
                "table_codes": list(point["table_codes"]),
                "applied_safe_codes": safe_codes,
                "samples": samples,
                "summary": {
                    "median_wavelength_nm": median_nm,
                    "span_pm": (max(waves) - min(waves)) * 1000.0,
                    "median_error_vs_table_target_pm": (
                        median_nm - float(point["table_target_nm"])
                    ) * 1000.0,
                    "median_error_vs_user_reference_pm": (
                        median_nm - float(point["user_reference_nm"])
                    ) * 1000.0,
                    "median_power_mw": statistics.median(powers),
                    "power_span_mw": max(powers) - min(powers),
                    "minimum_smsr_db": min(smsrs),
                    "single_mode_pass": min(smsrs) >= 20.0,
                },
            }
            payload["points"].append(result)
            output.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    finally:
        if laser is not None:
            try:
                # Zero GAIN and SOA turns optical output off.  Retaining the
                # three tuning-channel codes avoids an unnecessary heater jump.
                laser.set_codes([0, 0, *last_safe_codes[2:]])
                time.sleep(0.10)
            finally:
                laser.close()
        if meter is not None:
            meter.close()

    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"REPORT={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
