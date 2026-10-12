"""Stage and optionally install all live-verified 2001-point repairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import app_JDSU as app
import export_fullband_calibration as exporter
from laser_power_calibration import LaserSerial


HERE = Path(__file__).resolve().parent
FORMAL_PREFIX = HERE / "fullband_equal_power_operational_2001"
PRECISION_RESULT = HERE / "outputs" / "fullband_lan_anomaly_precision_20260904.json"
CONTINUATION_RESULT = HERE / "outputs" / "fullband_1539_soa_continuation_20260904.json"
SMSR_RESULT = HERE / "outputs" / "fullband_usb_smsr_optimization_20260904.json"
DEFAULT_STAGE_DIR = HERE / "outputs" / "fullband_operational_staged_20260904"
PRECISION_TARGETS = (1527.44, 1535.08, 1537.84, 1539.12, 1552.04, 1555.48, 1564.44)
SMSR_TARGETS = (1527.60, 1535.36, 1541.54, 1552.20)
TARGETS = PRECISION_TARGETS + SMSR_TARGETS


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def representative_confirmation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 5 or not all(bool(row.get("passed")) for row in rows):
        raise ValueError("每个修复点必须有5/5通过的独立复测")
    wavelengths = [float(row["reading"]["wavelength_nm"]) for row in rows]
    powers = [float(row["reading"]["power_mw"]) for row in rows]
    wavelength_median = statistics.median(wavelengths)
    power_median = statistics.median(powers)
    return min(
        rows,
        key=lambda row: (
            abs(float(row["reading"]["wavelength_nm"]) - wavelength_median) / 0.0002
            + abs(float(row["reading"]["power_mw"]) - power_median) / 0.001
        ),
    )


def precision_repair(payload: dict[str, Any], target_nm: float) -> dict[str, Any]:
    row = payload["rows"][f"{target_nm:.2f}"]
    if not row.get("success"):
        raise ValueError(f"{target_nm:.2f} nm尚未通过寻优")
    successful = [
        attempt for attempt in row.get("attempts", ())
        if attempt.get("confirmation", {}).get("success")
    ]
    if len(successful) != 1:
        raise ValueError(f"{target_nm:.2f} nm成功确认记录数量异常：{len(successful)}")
    confirmation = successful[0]["confirmation"]
    representative = representative_confirmation(confirmation["rows"])
    return {
        "codes": [int(value) for value in row["codes"]],
        "currents_ma": [float(value) for value in row["currents_ma"]],
        "reading": dict(representative["reading"]),
        "confirmation": confirmation,
        "search_source": row.get("seed_name"),
        "path": row.get("path"),
    }


def continuation_repair(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload.get("success") or not payload.get("soa_shutter_confirmed"):
        raise ValueError("1539.12 nm连续支路寻优或关光确认未通过")
    successful = [item for item in payload["confirmations"] if item.get("success")]
    if not successful:
        raise ValueError("1539.12 nm没有5/5成功的真路径复测")
    confirmation = successful[0]
    representative = representative_confirmation(confirmation["rows"])
    return {
        "codes": [int(value) for value in payload["final_codes"]],
        "currents_ma": [float(value) for value in payload["final_currents_ma"]],
        "reading": dict(representative["reading"]),
        "confirmation": confirmation,
        "search_source": "SOA_continuation",
        "path": "three_forward_predecessors_direct",
    }


def load_repairs() -> dict[float, dict[str, Any]]:
    precision = read_json(PRECISION_RESULT)
    if not precision.get("soa_shutter_confirmed"):
        raise ValueError("六个普通修复点完成后没有确认关光")
    repairs = {
        target: precision_repair(precision, target)
        for target in PRECISION_TARGETS if not abs(target - 1539.12) < 1e-9
    }
    repairs[1539.12] = continuation_repair(read_json(CONTINUATION_RESULT))
    smsr_payload = read_json(SMSR_RESULT)
    if not smsr_payload.get("soa_shutter_confirmed"):
        raise ValueError("四个SMSR修复点完成后没有确认关光")
    repairs.update({
        target: precision_repair(smsr_payload, target)
        for target in SMSR_TARGETS
    })
    for target, repair in repairs.items():
        LaserSerial.validate_codes(repair["codes"])
        reading = repair["reading"]
        wavelength_error_pm = (float(reading["wavelength_nm"]) - target) * 1000.0
        power_relative_error = float(reading["power_mw"]) / 1.530 - 1.0
        smsr = float(reading["side_mode_suppression_db"])
        if abs(wavelength_error_pm) > 2.0:
            raise ValueError(f"{target:.2f} nm波长复测超差")
        if abs(power_relative_error) > 0.005:
            raise ValueError(f"{target:.2f} nm功率复测超差")
        if smsr < 20.0:
            raise ValueError(f"{target:.2f} nm SMSR复测超差")
    return repairs


def feedback_by_target() -> dict[float, dict[str, Any]]:
    payload = read_json(HERE / "fullband_pdt_pdr_reference_2001.json")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 2001:
        raise ValueError("PDT/PDR参考表不是2001点")
    return {round(float(row["target_nm"]), 6): row for row in rows}


def stage(stage_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_path = FORMAL_PREFIX.with_suffix(".json")
    operational = read_json(source_path)
    rows_by_key = operational.get("rows")
    if not isinstance(rows_by_key, dict) or len(rows_by_key) != 2001:
        raise ValueError("正式JSON不是完整2001点")
    repairs = load_repairs()
    installed_at = datetime.now(timezone.utc).isoformat()
    for target, repair in repairs.items():
        key = f"{target:.6f}"
        row = rows_by_key[key]
        reading = repair["reading"]
        wavelength_error_pm = (float(reading["wavelength_nm"]) - target) * 1000.0
        power_relative_error = float(reading["power_mw"]) / 1.530 - 1.0
        row.update({
            "success": True,
            "codes": repair["codes"],
            "currents_ma": repair["currents_ma"],
            "reading": reading,
            "wavelength_error_pm": wavelength_error_pm,
            "power_error_mw": float(reading["power_mw"]) - 1.530,
            "power_relative_error": power_relative_error,
            "source": f"exact_ACK_live_anomaly_reoptimization_20260904:{repair['search_source']}",
            "lan_anomaly_reoptimization": {
                "installed_utc": installed_at,
                "path": repair["path"],
                "confirmation": repair["confirmation"],
            },
        })

    rows = exporter.ordered_rows(operational)
    if len(rows) != 2001:
        raise ValueError("暂存表点数不是2001")
    targets = [float(row["target_nm"]) for row in rows]
    measured = [float(row["reading"]["wavelength_nm"]) for row in rows]
    if any(abs((targets[index] - targets[0]) - index * 0.02) > 1e-8 for index in range(2001)):
        raise ValueError("目标波长网格不连续")
    if any(right <= left for left, right in zip(measured, measured[1:])):
        raise ValueError("实测波长不是严格递增")
    for row in rows:
        LaserSerial.validate_codes([int(value) for value in row["codes"]])

    operational["target_power_mw"] = 1.530
    operational["final_summary"] = exporter.compute_summary(rows, 1.530)
    export_meta = operational.setdefault("operational_export", {})
    export_meta.update({
        "created_utc": installed_at,
        "source_file": source_path.name,
        "scan_direction": "strictly_forward_increasing_wavelength",
        "start_nm": 1525.0,
        "stop_nm": 1565.0,
        "step_nm": 0.02,
        "point_count": 2001,
        "transition_guards": list(exporter.FORWARD_GUARDS),
        "total_guard_hold_ms_per_full_scan": sum(
            int(item["hold_ms_per_stage"]) * len(item["precondition_codes"])
            for item in exporter.FORWARD_GUARDS
        ),
        "anomaly_reoptimization": {
            "date": "2026-09-04",
            "targets_nm": list(TARGETS),
            "criterion": "wavelength<=2pm,power=1.530mW+/-0.5%,SMSR>=20dB,5/5",
        },
    })

    stage_dir.mkdir(parents=True, exist_ok=True)
    prefix = stage_dir / FORMAL_PREFIX.name
    json_path = prefix.with_suffix(".json")
    json_path.write_text(json.dumps(operational, ensure_ascii=False, indent=2), encoding="utf-8")
    feedback = feedback_by_target()
    exporter.write_csv(prefix.with_suffix(".csv"), rows, 1.530, feedback)
    exporter.write_header(prefix.with_suffix(".h"), rows, feedback)
    exporter.write_guard_file(
        prefix.with_name(prefix.name + "_transition_guards").with_suffix(".json"),
        json_path.name,
    )

    points = app.load_fullband_accuracy_table(stage_dir)
    guards = app.load_fullband_transition_guards(points, stage_dir)
    if len(points) != 2001 or set(guards) != app.FULLBAND_ACCURACY_GUARD_INDICES:
        raise ValueError("桌面端暂存表加载校验失败")
    return operational, rows


def install(stage_dir: Path) -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    backup_dir = HERE / "outputs" / f"fullband_operational_backup_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    names = (
        FORMAL_PREFIX.name + ".json",
        FORMAL_PREFIX.name + ".csv",
        FORMAL_PREFIX.name + ".h",
        FORMAL_PREFIX.name + "_transition_guards.json",
    )
    for name in names:
        source = HERE / name
        staged = stage_dir / name
        if not source.is_file() or not staged.is_file():
            raise FileNotFoundError(name)
        shutil.copy2(source, backup_dir / name)
    for name in names:
        staged = stage_dir / name
        temporary = HERE / (name + ".installing")
        shutil.copy2(staged, temporary)
        os.replace(temporary, HERE / name)
    return backup_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    operational, rows = stage(args.stage_dir)
    print(json.dumps(operational["final_summary"], ensure_ascii=False, indent=2))
    for name in (
        FORMAL_PREFIX.name + ".json",
        FORMAL_PREFIX.name + ".csv",
        FORMAL_PREFIX.name + ".h",
        FORMAL_PREFIX.name + "_transition_guards.json",
    ):
        path = args.stage_dir / name
        print(f"stage {name} sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")
    if args.install:
        backup = install(args.stage_dir)
        points = app.load_fullband_accuracy_table(HERE)
        guards = app.load_fullband_transition_guards(points, HERE)
        print(f"installed points={len(points)} guards={sorted(guards)} backup={backup}")
    else:
        print(f"staged_only={args.stage_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
