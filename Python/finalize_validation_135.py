"""Re-evaluate recorded 135 mA replay samples with single-peak semantics."""
from __future__ import annotations
import json
from pathlib import Path
from laser_power_calibration import save_json

HERE = Path(__file__).resolve().parent
FINAL = HERE / "stress_high_power_135_final.json"
VALIDATION = HERE / "stress_high_power_135_validation.json"


def main() -> int:
    rows = json.loads(FINAL.read_text(encoding="utf-8"))["rows"]
    payload = json.loads(VALIDATION.read_text(encoding="utf-8"))
    if len(payload.get("results", ())) != len(rows):
        raise ValueError("标定表与复测结果数量不一致")

    for result, row in zip(payload["results"], rows):
        samples = result["samples"]
        if not samples:
            raise ValueError("复测结果中存在空采样")
        waves = [float(item["wavelength_nm"]) for item in samples]
        powers = [float(item["power_mw"]) for item in samples]
        smsrs = [
            99.0
            if int(item.get("peak_count", 0)) == 1
            and item.get("side_mode_suppression_db") is None
            else float(item.get("side_mode_suppression_db") or -999.0)
            for item in samples
        ]
        reference = float(row["calibration"]["wavelength_nm"])
        result["smsr_min_db"] = min(smsrs)
        result["passed"] = bool(
            max(abs(wave - reference) for wave in waves) <= 0.010
            and max(waves) - min(waves) <= 0.006
            and min(smsrs) >= 20.0
            and min(powers) >= 1.425
            and max(powers) <= 1.575
        )
    good = sum(bool(result["passed"]) for result in payload["results"])
    payload["summary"] = {
        "tested": len(payload["results"]),
        "passed": good,
        "failed": len(payload["results"]) - good,
        "power_min_mw": min(
            result["power_min_mw"] for result in payload["results"]
        ),
        "power_max_mw": max(
            result["power_max_mw"] for result in payload["results"]
        ),
        "minimum_finite_or_single_peak_smsr_db": min(
            result["smsr_min_db"] for result in payload["results"]
        ),
        "max_wavelength_span_pm": max(
            result["wavelength_span_pm"] for result in payload["results"]
        ),
    }
    save_json(VALIDATION, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
