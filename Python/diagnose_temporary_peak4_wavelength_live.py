"""Measure the temporary-route peak-4 wavelengths on their real sparse path."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import statistics
import time

from fullband_auto_calibration import best_effort_final_shutter
from laser_dac_safety import FULLBAND_2001_CODE_LIMITS
from laser_power_calibration import AQ6150B, LaserSerial


HERE = Path(__file__).resolve().parent
DEFAULT_TABLE = HERE / "fullband_equal_power_operational_2001.json"
DEFAULT_OUTPUT = (
    HERE / "outputs" / "temporary_test" /
    "peak4_sparse_wavelength_diagnostic_latest.json"
)

# Groups 1--3 are the latest dense-spectrum selection.  Group 4 is the
# previously observed 9/9 fallback under investigation.  Replaying every
# predecessor before each meter trigger preserves the actual low-to-high
# sparse entry path without allowing earlier AQ reads to stretch that path.
ROUTE_INDICES = (
    102, 117, 132, 147, 162,
    176, 193, 210, 227, 244,
    319, 334, 349, 364, 379,
    386, 402, 418, 434, 450,
)
TARGET_INDICES = (386, 402, 418, 434, 450)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--meter", default="GPIB0::7::INSTR")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--settle-s", type=float, default=0.24)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        raise ValueError("repeats must be 1..10")
    if not 0.05 <= args.settle_s <= 5.0:
        raise ValueError("settle-s must be 0.05..5.0")

    table = json.loads(args.table.read_text(encoding="utf-8"))
    keyed = table.get("rows", {})
    ordered = sorted(keyed.values(), key=lambda row: float(row["target_nm"]))
    if len(ordered) != 2001:
        raise RuntimeError("runtime table is not a complete 2001-point table")
    rows = {index: ordered[index] for index in ROUTE_INDICES}
    payload = {
        "schema": "temporary_peak4_sparse_wavelength_diagnostic_v1",
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_table": str(args.table.resolve()),
        "route_indices": list(ROUTE_INDICES),
        "target_indices": list(TARGET_INDICES),
        "method": (
            "For each target and repeat, replay groups 1-3 plus the peak-4 "
            "prefix at the normal sparse cadence, then trigger an independent "
            "AQ6150B reading with three internal averages."
        ),
        "settle_s": args.settle_s,
        "meter_average_count": 3,
        "rows": [],
        "complete": False,
        "shutdown_confirmed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    laser = None
    meter = None
    error = None
    try:
        laser = LaserSerial(args.port, code_limits=FULLBAND_2001_CODE_LIMITS)
        meter = AQ6150B(args.meter, average_count=3)
        payload["meter"] = meter.identity
        for target_index in TARGET_INDICES:
            target_nm = float(rows[target_index]["target_nm"])
            samples = []
            prefix = ROUTE_INDICES[: ROUTE_INDICES.index(target_index) + 1]
            for repeat in range(args.repeats):
                for position, index in enumerate(prefix):
                    laser.set_codes([int(value) for value in rows[index]["codes"]])
                    time.sleep(0.50 if position == 0 else args.settle_s)
                reading = meter.measure(target_nm, select_main_peak=True)
                samples.append(asdict(reading))
                print(
                    f"index={target_index} repeat={repeat + 1}/{args.repeats} "
                    f"target={target_nm:.5f} measured={reading.wavelength_nm:.8f} "
                    f"error={reading.wavelength_error_pm:+.3f}pm "
                    f"power={reading.power_mw:.6f}mW",
                    flush=True,
                )
            waves = [float(sample["wavelength_nm"]) for sample in samples]
            powers = [float(sample["power_mw"]) for sample in samples]
            median_wave = statistics.median(waves)
            median_power = statistics.median(powers)
            payload["rows"].append({
                "index": target_index,
                "target_nm": target_nm,
                "table_measured_nm": float(rows[target_index]["reading"]["wavelength_nm"]),
                "codes": [int(value) for value in rows[target_index]["codes"]],
                "samples": samples,
                "median_wavelength_nm": median_wave,
                "median_wavelength_error_pm": (median_wave - target_nm) * 1000.0,
                "wavelength_span_pm": (max(waves) - min(waves)) * 1000.0,
                "median_power_mw": median_power,
                "power_relative_error_percent": (median_power / 2.056 - 1.0) * 100.0,
                "wavelength_pass": abs(median_wave - target_nm) <= 0.002,
                "power_pass": abs(median_power / 2.056 - 1.0) <= 0.03,
            })
            args.output.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        payload["complete"] = True
    except BaseException as exc:
        error = exc
        payload["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if meter is not None:
            meter.close()
        if laser is not None:
            laser.close()
        try:
            best_effort_final_shutter(args.port)
            payload["shutdown_confirmed"] = True
        except BaseException as shutdown_error:
            payload["shutdown_error"] = (
                f"{type(shutdown_error).__name__}: {shutdown_error}"
            )
            if error is None:
                error = shutdown_error
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if error is not None:
        raise error
    print(json.dumps(payload["rows"], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
