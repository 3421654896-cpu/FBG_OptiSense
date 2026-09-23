"""Export the measured 1525--1565 nm equal-power calibration for operation.

The source JSON contains the complete measurement history.  This exporter keeps
that audit trail, adds the forward-only transition guards found by live replay,
and emits compact CSV/C forms for analysis and firmware integration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


CHANNEL_NAMES = ("GAIN", "SOA", "PHASE", "WAVELENGTH_A", "WAVELENGTH_B")
START_NM = 1525.0
STEP_NM = 0.02
LIMIT_PROFILE = "fullband_2001_145mA_v1"
FULLBAND_CODE_LIMITS = (63351, 63351, 32767, 24575, 24575)
POWER_REL_TOLERANCE = 0.005

# These branch-conditioning codes were replayed on the real laser in forward
# scan order.  The former 1527.44-nm guard was removed after its replacement
# point passed five direct forward-path replays without any precondition.
FORWARD_GUARDS = (
    {
        "target_nm": 1529.82,
        "hold_ms_per_stage": 5,
        "precondition_codes": [
            [58982, 53673, 0, 19472, 9600],
            [58982, 53673, 0, 19144, 9545],
        ],
        "verification": "3/3 wavelength+mode passes",
    },
    {
        "target_nm": 1562.08,
        "hold_ms_per_stage": 20,
        "precondition_codes": [
            [58982, 44508, 14273, 2562, 6404],
            [58982, 44508, 14273, 2534, 6334],
        ],
        "verification": "5/5 wavelength+mode passes",
    },
)


def choose_power_feedback(feedback: dict) -> tuple[str, int, int]:
    """Choose the monitor with the largest symmetric 12-bit rail margin."""
    choices = []
    for channel in ("PDT", "PDR"):
        code = int(feedback[f"{channel.lower()}_code"])
        margin = min(code, 2048 - code)
        choices.append((margin, channel, code))
    margin, channel, code = max(choices)
    return channel, code, margin


def load_source(
    path: Path,
    power_relative_tolerance: float = POWER_REL_TOLERANCE,
) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    rows = data.get("rows")
    if not isinstance(rows, dict) or len(rows) != 2001:
        raise ValueError("expected exactly 2001 keyed calibration rows")
    limits = data.get("limits_ma", {})
    if (
        float(limits.get("gain_ma", -1.0)) != 145.0
        or float(limits.get("soa_ma", -1.0)) != 145.0
    ):
        raise ValueError("operational export requires the measured 145 mA profile")
    target_power_mw = float(data.get("target_power_mw", float("nan")))
    if not math.isfinite(target_power_mw) or target_power_mw <= 0.0:
        raise ValueError("operational export requires a finite positive common power target")
    ordered = sorted(rows.items(), key=lambda item: float(item[1]["target_nm"]))
    for index, (key, row) in enumerate(ordered):
        expected_target = START_NM + STEP_NM * index
        if abs(float(row["target_nm"]) - expected_target) > 1e-7:
            raise ValueError(f"row {key} is not on the complete 0.02 nm grid")
        if not row.get("success"):
            raise ValueError(f"row {key} has not passed calibration")
        measured_power_mw = float(row.get("reading", {}).get("power_mw", float("nan")))
        relative_error = abs(measured_power_mw / target_power_mw - 1.0)
        if (
            not math.isfinite(relative_error)
            or relative_error > power_relative_tolerance + 1e-12
        ):
            raise ValueError(
                f"row {key} power differs from the common target by more than "
                f"±{power_relative_tolerance * 100.0:g}%"
            )
        codes = row.get("codes", ())
        if len(codes) != len(FULLBAND_CODE_LIMITS) or any(
            isinstance(code, bool)
            or not isinstance(code, int)
            or code < 0
            or code > maximum
            for code, maximum in zip(codes, FULLBAND_CODE_LIMITS)
        ):
            raise ValueError(f"row {key} exceeds the 145 mA profile")
    return data


def ordered_rows(data: dict) -> list[dict]:
    return sorted(data["rows"].values(), key=lambda row: float(row["target_nm"]))


def validate_repeatability_pass(
    path: Path,
    source_rows: list[dict],
    target_power_mw: float,
    expected_direction: str,
    power_relative_tolerance: float = POWER_REL_TOLERANCE,
) -> dict:
    """Require one independent 2001/2001 exact-code validation pass."""

    if not path.exists():
        raise ValueError(f"missing {expected_direction} validation: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("direction") != expected_direction:
        raise ValueError(f"{path} is not the {expected_direction} validation")
    limits = payload.get("limits_ma", {})
    if (
        float(limits.get("gain_ma", -1.0)) != 145.0
        or float(limits.get("soa_ma", -1.0)) != 145.0
    ):
        raise ValueError(f"{path} was not measured with the 145 mA profile")
    if abs(float(payload.get("target_power_mw", -1.0)) - target_power_mw) > 1e-9:
        raise ValueError(f"{path} used a different common power target")

    measured = payload.get("rows")
    if not isinstance(measured, dict) or len(measured) != len(source_rows):
        raise ValueError(f"{path} does not contain all 2001 validation rows")
    source_by_target = {
        round(float(row["target_nm"]), 6): row for row in source_rows
    }
    seen_targets = set()
    for row in measured.values():
        target = round(float(row["target_nm"]), 6)
        if target in seen_targets:
            raise ValueError(f"{path} repeats target {target:.6f} nm")
        seen_targets.add(target)
        source = source_by_target.get(target)
        if source is None:
            raise ValueError(f"{path} contains unexpected target {target:.6f} nm")
        if not row.get("success"):
            raise ValueError(f"{path} failed at {target:.6f} nm")
        measured_power_mw = float(row.get("reading", {}).get("power_mw", float("nan")))
        relative_error = abs(measured_power_mw / target_power_mw - 1.0)
        if (
            not math.isfinite(relative_error)
            or relative_error > power_relative_tolerance + 1e-12
        ):
            raise ValueError(
                f"{path} power differs from the common target by more than "
                f"±{power_relative_tolerance * 100.0:g}% "
                f"at {target:.6f} nm"
            )
        if [int(value) for value in row.get("codes", ())] != [
            int(value) for value in source["codes"]
        ]:
            raise ValueError(f"{path} used different DAC codes at {target:.6f} nm")
    if seen_targets != set(source_by_target):
        raise ValueError(f"{path} does not cover the complete source grid")
    summary = payload.get("summary", {})
    if int(summary.get("tested", -1)) != 2001 or int(summary.get("passed", -1)) != 2001:
        raise ValueError(f"{path} summary is not 2001/2001")
    return {
        "file": path.name,
        "direction": expected_direction,
        "tested": 2001,
        "passed": 2001,
        "maximum_abs_wavelength_error_pm": summary.get(
            "maximum_abs_wavelength_error_pm"
        ),
        "power_min_mw": summary.get("power_min_mw"),
        "power_median_mw": summary.get("power_median_mw"),
        "power_max_mw": summary.get("power_max_mw"),
        "minimum_smsr_db": summary.get("minimum_smsr_db"),
    }


def compute_summary(
    rows: list[dict],
    target_power_mw: float,
    power_relative_tolerance: float = POWER_REL_TOLERANCE,
) -> dict:
    powers = sorted(float(row["reading"]["power_mw"]) for row in rows)
    wavelength_errors = [abs(float(row["wavelength_error_pm"])) for row in rows]
    smsrs = [
        float(row["reading"]["side_mode_suppression_db"])
        for row in rows
        if row["reading"].get("side_mode_suppression_db") is not None
    ]
    gain = [float(row["currents_ma"][0]) for row in rows]
    soa = [float(row["currents_ma"][1]) for row in rows]
    return {
        "tested": len(rows),
        "passed": sum(bool(row.get("success")) for row in rows),
        "failed": sum(not bool(row.get("success")) for row in rows),
        "target_power_mw": target_power_mw,
        "power_min_mw": powers[0],
        "power_median_mw": powers[len(powers) // 2],
        "power_max_mw": powers[-1],
        "power_peak_to_peak_percent_of_target": (
            (powers[-1] - powers[0]) / target_power_mw * 100.0
        ),
        "maximum_abs_power_error_percent": max(
            abs(value / target_power_mw - 1.0) * 100.0 for value in powers
        ),
        "power_acceptance_tolerance_percent": power_relative_tolerance * 100.0,
        "maximum_abs_wavelength_error_pm": max(wavelength_errors),
        "minimum_smsr_db": min(smsrs),
        "gain_min_ma": min(gain),
        "gain_max_ma": max(gain),
        "soa_min_ma": min(soa),
        "soa_max_ma": max(soa),
    }


def feedback_summary(feedback_rows: list[dict], target_power_mw: float) -> dict:
    spot_rows = [row for row in feedback_rows if "meter_spot_check" in row]
    powers = [float(row["meter_spot_check"]["power_mw"]) for row in spot_rows]
    wavelength_errors = [
        abs(float(row["meter_spot_check"]["wavelength_error_pm"])) for row in spot_rows
    ]
    power_abs_percent = sorted(abs(value / target_power_mw - 1.0) * 100.0 for value in powers)
    selections = [choose_power_feedback(row) for row in feedback_rows]
    return {
        "point_count": len(feedback_rows),
        "pdt_code_min": min(int(row["pdt_code"]) for row in feedback_rows),
        "pdt_code_max": max(int(row["pdt_code"]) for row in feedback_rows),
        "pdt_near_lower_rail_count_code_lt_128": sum(
            int(row["pdt_code"]) < 128 for row in feedback_rows
        ),
        "adaptive_feedback_pdt_point_count": sum(channel == "PDT" for channel, _, _ in selections),
        "adaptive_feedback_pdr_point_count": sum(channel == "PDR" for channel, _, _ in selections),
        "adaptive_feedback_hardware_eligible_count_margin_ge_128": sum(
            margin >= 128 for _, _, margin in selections
        ),
        "adaptive_feedback_minimum_rail_margin_code": min(
            margin for _, _, margin in selections
        ),
        "meter_spot_check_count": len(spot_rows),
        "spot_power_min_mw": min(powers),
        "spot_power_median_mw": sorted(powers)[len(powers) // 2],
        "spot_power_max_mw": max(powers),
        "spot_power_abs_error_median_percent": power_abs_percent[len(power_abs_percent) // 2],
        "spot_power_abs_error_max_percent": max(power_abs_percent),
        "spot_wavelength_abs_error_median_pm": sorted(wavelength_errors)[len(wavelength_errors) // 2],
        "spot_wavelength_abs_error_max_pm": max(wavelength_errors),
    }


def make_operational_json(
    source: dict,
    rows: list[dict],
    source_path: Path,
    feedback_rows: list[dict] | None = None,
    feedback_source: Path | None = None,
    power_relative_tolerance: float = POWER_REL_TOLERANCE,
) -> dict:
    target_power_mw = float(source["target_power_mw"])
    result = dict(source)
    result["operational_export"] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": source_path.name,
        "scan_direction": "strictly_forward_increasing_wavelength",
        "start_nm": START_NM,
        "stop_nm": 1565.0,
        "step_nm": STEP_NM,
        "point_count": len(rows),
        "limit_profile": LIMIT_PROFILE,
        "transition_guards": list(FORWARD_GUARDS),
        "total_guard_hold_ms_per_full_scan": sum(
            int(guard["hold_ms_per_stage"]) * len(guard["precondition_codes"])
            for guard in FORWARD_GUARDS
        ),
        "power_control_note": (
            "Static wavelength-meter equalization is the feed-forward baseline. "
            "Long-duration equal power requires a small PDT-referenced SOA trim; "
            "PDT/PDR were unavailable in this measurement set."
        ),
        "single_point_power_relative_tolerance": power_relative_tolerance,
    }
    result["final_summary"] = compute_summary(
        rows, target_power_mw, power_relative_tolerance
    )
    if feedback_rows:
        feedback_by_target = {
            round(float(row["target_nm"]), 6): row for row in feedback_rows
        }
        for row in result["rows"].values():
            feedback = feedback_by_target[round(float(row["target_nm"]), 6)]
            channel, reference_code, margin_code = choose_power_feedback(feedback)
            row["runtime_feedback_reference"] = {
                "pdt_code": int(feedback["pdt_code"]),
                "pdr_code": int(feedback["pdr_code"]),
                "pdt_signal_code": int(feedback["pdt_signal_code"]),
                "pdr_signal_code": int(feedback["pdr_signal_code"]),
                "pdr_pdt_signal_ratio": feedback["pdr_pdt_signal_ratio"],
                "power_feedback_channel": channel,
                "power_feedback_reference_code": reference_code,
                "power_feedback_rail_margin_code": margin_code,
                "power_feedback_hardware_eligible": margin_code >= 128,
            }
        result["feedback_reference"] = {
            "source_file": feedback_source.name if feedback_source else None,
            "adaptive_monitor_rail_margin_guard_code": 128,
            "control_policy": (
                "For each wavelength PDT or PDR can be selected by rail margin. "
                "These captured codes prove hardware observability only; they are "
                "not a validated common-power closed-loop target.  PDR/PDT ratio remains "
                "a wavelength/mode guard, not a global power target."
            ),
            "summary": feedback_summary(feedback_rows, target_power_mw),
        }
    return result


def write_csv(
    path: Path,
    rows: list[dict],
    target_power_mw: float,
    feedback_by_target: dict[float, dict] | None = None,
) -> None:
    fields = [
        "limit_profile", "index", "target_wavelength_nm", "measured_wavelength_nm",
        "wavelength_error_pm", "target_power_mw", "measured_power_mw",
        "power_error_percent", "smsr_db", "peak_count", "single_mode_pass",
        "gain_ma", "soa_ma", "phase_ma", "wavelength_a_ma", "wavelength_b_ma",
        "gain_code", "soa_code", "phase_code", "wavelength_a_code", "wavelength_b_code",
        "pdt_reference_code", "pdr_reference_code", "power_feedback_channel",
        "power_feedback_reference_code", "power_feedback_rail_margin_code",
        "power_feedback_hardware_eligible",
        "guard_stage_count", "guard_hold_ms_per_stage",
    ]
    guards = {round(float(g["target_nm"]), 6): g for g in FORWARD_GUARDS}
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            reading = row["reading"]
            currents = row["currents_ma"]
            codes = row["codes"]
            guard = guards.get(round(float(row["target_nm"]), 6))
            feedback = ((feedback_by_target or {}).get(round(float(row["target_nm"]), 6)))
            if feedback:
                feedback_channel, feedback_code, feedback_margin = choose_power_feedback(feedback)
            else:
                feedback_channel, feedback_code, feedback_margin = "", 0, 0
            writer.writerow({
                "limit_profile": LIMIT_PROFILE,
                "index": index,
                "target_wavelength_nm": f"{float(row['target_nm']):.6f}",
                "measured_wavelength_nm": f"{float(reading['wavelength_nm']):.8f}",
                "wavelength_error_pm": f"{float(row['wavelength_error_pm']):.6f}",
                "target_power_mw": f"{target_power_mw:.6f}",
                "measured_power_mw": f"{float(reading['power_mw']):.9f}",
                "power_error_percent": f"{100.0 * float(row['power_relative_error']):.6f}",
                "smsr_db": (f"{float(reading['side_mode_suppression_db']):.6f}"
                            if reading.get("side_mode_suppression_db") is not None else ""),
                "peak_count": int(reading["peak_count"]),
                "single_mode_pass": int(
                    reading.get("side_mode_suppression_db") is None
                    or float(reading["side_mode_suppression_db"]) >= 20.0
                ),
                "gain_ma": f"{float(currents[0]):.6f}",
                "soa_ma": f"{float(currents[1]):.6f}",
                "phase_ma": f"{float(currents[2]):.6f}",
                "wavelength_a_ma": f"{float(currents[3]):.6f}",
                "wavelength_b_ma": f"{float(currents[4]):.6f}",
                "gain_code": int(codes[0]),
                "soa_code": int(codes[1]),
                "phase_code": int(codes[2]),
                "wavelength_a_code": int(codes[3]),
                "wavelength_b_code": int(codes[4]),
                "pdt_reference_code": feedback.get("pdt_code") if feedback else "",
                "pdr_reference_code": feedback.get("pdr_code") if feedback else "",
                "power_feedback_channel": feedback_channel,
                "power_feedback_reference_code": feedback_code,
                "power_feedback_rail_margin_code": feedback_margin,
                "power_feedback_hardware_eligible": int(feedback_margin >= 128),
                "guard_stage_count": len(guard["precondition_codes"]) if guard else 0,
                "guard_hold_ms_per_stage": guard["hold_ms_per_stage"] if guard else 0,
            })


def write_header(
    path: Path,
    rows: list[dict],
    feedback_by_target: dict[float, dict] | None = None,
) -> None:
    lines = [
        "/* Auto-generated from live AQ6150B calibration. Do not hand edit. */",
        "#ifndef FULLBAND_EQUAL_POWER_2001_H",
        "#define FULLBAND_EQUAL_POWER_2001_H",
        "",
        "#include <stdint.h>",
        "",
        "#define LASER_CAL_START_PM        1525000UL",
        "#define LASER_CAL_STEP_PM         20U",
        f"#define LASER_CAL_POINT_COUNT     {len(rows)}U",
        "#define LASER_CAL_CHANNEL_COUNT   5U",
        f"#define LASER_CAL_GUARD_COUNT     {len(FORWARD_GUARDS)}U",
        "",
        "typedef struct {",
        "    uint16_t code[LASER_CAL_CHANNEL_COUNT];",
        "    uint16_t pdt_reference_code;",
        "    uint16_t pdr_reference_code;",
        "    uint16_t power_feedback_reference_code;",
        "    uint16_t power_feedback_rail_margin_code;",
        "    uint8_t power_feedback_channel; /* 0=PDT, 1=PDR */",
        "    uint8_t power_feedback_hardware_eligible;",
        "} LaserCalibrationPoint_t;",
        "typedef struct {",
        "    uint16_t target_index;",
        "    uint8_t stage_count;",
        "    uint16_t hold_us_per_stage;",
        "    uint16_t precondition_code[2][LASER_CAL_CHANNEL_COUNT];",
        "} LaserTransitionGuard_t;",
        "",
        "static const LaserCalibrationPoint_t g_laser_equal_power_2001[LASER_CAL_POINT_COUNT] = {",
    ]
    for row in rows:
        code_text = ", ".join(f"{int(value)}U" for value in row["codes"])
        feedback = ((feedback_by_target or {}).get(round(float(row["target_nm"]), 6)))
        pdt = int(feedback["pdt_code"]) if feedback else 0
        pdr = int(feedback["pdr_code"]) if feedback else 0
        if feedback:
            channel, feedback_code, feedback_margin = choose_power_feedback(feedback)
        else:
            channel, feedback_code, feedback_margin = "PDT", 0, 0
        channel_id = int(channel == "PDR")
        eligible = int(feedback_margin >= 128)
        lines.append(
            f"    {{{{{code_text}}}, {pdt}U, {pdr}U, {feedback_code}U, "
            f"{feedback_margin}U, {channel_id}U, {eligible}U}}, "
            f"/* {float(row['target_nm']):.2f} nm */"
        )
    lines.extend(["};", "", "static const LaserTransitionGuard_t g_laser_transition_guards[LASER_CAL_GUARD_COUNT] = {"])
    for guard in FORWARD_GUARDS:
        target_index = int(round((float(guard["target_nm"]) - START_NM) / STEP_NM))
        stages = list(guard["precondition_codes"])
        padded = stages + [[0, 0, 0, 0, 0]] * (2 - len(stages))
        first = ", ".join(f"{int(v)}U" for v in padded[0])
        second = ", ".join(f"{int(v)}U" for v in padded[1])
        lines.append(
            f"    {{{target_index}U, {len(stages)}U, {int(guard['hold_ms_per_stage']) * 1000}U, "
            f"{{{{{first}}}, {{{second}}}}}}}, /* {float(guard['target_nm']):.2f} nm */"
        )
    lines.extend(["};", "", "#endif /* FULLBAND_EQUAL_POWER_2001_H */", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_guard_file(path: Path, source_name: str) -> None:
    payload = {
        "source_table": source_name,
        "limit_profile": LIMIT_PROFILE,
        "scan_direction": "forward",
        "guards": list(FORWARD_GUARDS),
        "total_guard_hold_ms_per_full_scan": sum(
            int(g["hold_ms_per_stage"]) * len(g["precondition_codes"])
            for g in FORWARD_GUARDS
        ),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("fullband_equal_power_neighbor_repaired_2001.json"))
    parser.add_argument("--prefix", type=Path, default=Path("fullband_equal_power_operational_2001"))
    parser.add_argument("--feedback", type=Path, default=Path("fullband_pdt_pdr_reference_2001.json"))
    parser.add_argument(
        "--forward-validation", type=Path,
        default=Path("fullband_equal_power_forward_validation_145_2001.json"),
    )
    parser.add_argument(
        "--reverse-validation", type=Path,
        default=Path("fullband_equal_power_reverse_validation_145_2001.json"),
    )
    parser.add_argument(
        "--certification-mode",
        choices=("bidirectional", "forward-only"),
        default="bidirectional",
        help="forward-only is valid only for strictly increasing-wavelength runtime scans",
    )
    parser.add_argument(
        "--power-relative-tolerance",
        type=float,
        default=POWER_REL_TOLERANCE,
        help="single-point power acceptance tolerance as a fraction",
    )
    args = parser.parse_args()
    if not math.isfinite(args.power_relative_tolerance) or not (
        0.0 < args.power_relative_tolerance <= 0.10
    ):
        raise ValueError("power relative tolerance must be between 0 and 10%")

    source = load_source(args.source, args.power_relative_tolerance)
    rows = ordered_rows(source)
    target_power_mw = float(source["target_power_mw"])
    certification = {
        "mode": args.certification_mode,
        "runtime_scan_direction": "strictly_forward_increasing_wavelength",
        "forward": validate_repeatability_pass(
            args.forward_validation, rows, target_power_mw, "forward",
            args.power_relative_tolerance,
        ),
    }
    if args.certification_mode == "bidirectional":
        certification["reverse"] = validate_repeatability_pass(
            args.reverse_validation, rows, target_power_mw, "reverse",
            args.power_relative_tolerance,
        )
    else:
        certification["scope_note"] = (
            "Certified only for the deployed low-to-high wavelength path; "
            "reverse-order hysteresis is outside this table's release scope."
        )
    feedback_rows = None
    feedback_by_target = None
    if args.feedback.exists():
        feedback_payload = json.loads(args.feedback.read_text(encoding="utf-8"))
        feedback_rows = feedback_payload.get("rows")
        if not isinstance(feedback_rows, list) or len(feedback_rows) != len(rows):
            raise ValueError("feedback reference must contain exactly 2001 rows")
        feedback_by_target = {
            round(float(row["target_nm"]), 6): row for row in feedback_rows
        }
    operational = make_operational_json(
        source, rows, args.source, feedback_rows,
        args.feedback if feedback_rows else None,
        args.power_relative_tolerance,
    )
    operational["independent_repeatability_certification"] = certification

    json_path = args.prefix.with_suffix(".json")
    csv_path = args.prefix.with_suffix(".csv")
    header_path = args.prefix.with_suffix(".h")
    guard_path = args.prefix.with_name(args.prefix.name + "_transition_guards").with_suffix(".json")
    json_path.write_text(json.dumps(operational, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(csv_path, rows, target_power_mw, feedback_by_target)
    write_header(header_path, rows, feedback_by_target)
    write_guard_file(guard_path, json_path.name)

    for path in (json_path, csv_path, header_path, guard_path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{path.name}\t{path.stat().st_size}\tsha256={digest}")
    print(json.dumps(operational["final_summary"], indent=2))


if __name__ == "__main__":
    main()
