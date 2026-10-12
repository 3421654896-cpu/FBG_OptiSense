"""Safely re-optimize selected 2001-point laser states over the LAN RAW link.

The search follows the real forward-scan path.  It starts from measured
same-branch neighbours, closes wavelength with the existing local WAVE-A/B
Jacobian, closes power with SOA, and then independently replays the preceding
official point before every confirmation.  This script only writes a result
checkpoint; it never modifies the operational table or firmware.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import app_JDSU as app
import fullband_equal_power_model as model
from audit_fullband_2001_lan import command_with_link_retry, wait_connected
from diagnose_adaptive_stress_peaks_live import _shutter
from fbg_lan_serial import FbgLanSerial
from laser_power_calibration import AQ6150B, LaserSerial, Reading, save_json

try:
    import serial
except ImportError:  # pragma: no cover - workstation venv includes pyserial
    serial = None


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "outputs" / "fullband_lan_anomaly_optimization_20260904.json"
DEFAULT_TARGETS = (1527.44, 1535.08, 1537.84, 1539.12, 1552.04, 1555.48, 1564.44)
SOA_CURVE = HERE / "fullband_soa_transfer_pilot_85mA.json"


# Each primary seed is an interpolation/extrapolation confined to one measured
# neighbouring branch.  The fallback is another independently measured branch.
# The high-SOA state is used only as an optional mode precondition, never as an
# automatically accepted final equal-power point.
SEARCH_PLANS: dict[float, dict[str, Any]] = {
    1527.44: {
        "seeds": (
            ("right_branch_extrapolation", (58981, 49570, 3386, 5421, 1301)),
            ("historical_clean_branch", (58981, 50501, 4593, 4655, 1325)),
            ("right_neighbour_branch", (58981, 50538, 3386, 5247, 1280)),
        ),
        "mode_preset": (58981, 58981, 4593, 4710, 1434),
    },
    1535.08: {
        "seeds": (
            ("left_branch_extrapolation", (58981, 51911, 9582, 1229, 15974)),
            ("high_soa_clean_branch", (58981, 58981, 9941, 1229, 15974)),
            ("left_neighbour_branch", (58981, 51326, 10408, 1229, 15974)),
        ),
        "mode_preset": (58981, 58981, 9941, 1229, 15974),
    },
    1537.84: {
        "seeds": (
            ("left_branch_extrapolation", (58981, 50746, 7373, 7132, 4562)),
            ("high_soa_clean_branch", (58981, 58981, 5908, 7373, 4915)),
            ("left_neighbour_branch", (58981, 50746, 7373, 7227, 4699)),
        ),
        "mode_preset": (58981, 58981, 5908, 7373, 4915),
    },
    1539.12: {
        "seeds": (
            ("left_branch_extrapolation", (58981, 46407, 13866, 3638, 2461)),
            ("high_soa_clean_branch", (58981, 58981, 14737, 3651, 2458)),
            ("left_neighbour_branch", (58981, 46779, 15030, 3661, 2458)),
        ),
        "mode_preset": (58981, 58981, 14737, 3651, 2458),
    },
    1552.04: {
        "seeds": (
            ("two_sided_interpolation", (58981, 44409, 11078, 22528, 1816)),
            ("historical_clean_branch", (58981, 44289, 10702, 22432, 1843)),
            ("left_neighbour_branch", (58981, 44603, 11592, 22528, 1843)),
        ),
        "mode_preset": (58981, 58981, 11439, 22528, 1843),
    },
    1555.48: {
        "seeds": (
            ("left_branch_extrapolation", (58981, 44560, 3722, 4990, 8500)),
            ("historical_clean_branch", (58981, 46520, 4188, 4873, 8466)),
            ("left_neighbour_branch", (58981, 43991, 4298, 4952, 8530)),
        ),
        "mode_preset": (58981, 58981, 4188, 4915, 8602),
    },
    1564.44: {
        "seeds": (
            ("two_sided_interpolation", (58981, 45530, 9808, 17123, 2267)),
            ("high_soa_clean_branch", (58981, 58981, 9813, 17184, 2318)),
            ("left_neighbour_branch", (58981, 45859, 9815, 17186, 2336)),
        ),
        "mode_preset": (58981, 58981, 9813, 17184, 2318),
    },
}


class RetryingMeter:
    """AQ6150B wrapper that repeats only known transient array races."""

    _TRANSIENT = (
        "returned an empty measurement",
        "wavelength/power arrays are inconsistent",
    )

    def __init__(self, resource: str):
        self._meter = AQ6150B(resource)
        self.identity = self._meter.identity

    def close(self) -> None:
        self._meter.close()

    def measure(self, target_nm: float, select_main_peak: bool = False) -> Reading:
        for attempt in range(1, 4):
            try:
                return self._meter.measure(
                    target_nm, select_main_peak=select_main_peak
                )
            except RuntimeError as exc:
                if not any(message in str(exc) for message in self._TRANSIENT):
                    raise
                if attempt == 3:
                    raise
                time.sleep(0.25 * attempt)
        raise AssertionError("unreachable")


class LanLaserAdapter:
    """Subset of LaserSerial backed by exact PI11210 LAN acknowledgements."""

    def __init__(self, worker: app.UnlimitedAccuracyWorker):
        self.worker = worker

    def set_codes(self, codes) -> None:
        safe = LaserSerial.validate_codes([int(value) for value in codes])
        if command_with_link_retry(self.worker, safe) is None:
            raise RuntimeError("寻优已中止")

    @staticmethod
    def latest_pd_reading():
        # The single-value ACK validates all DACs and PI11210 status.  PDR/PDT
        # are not needed for wavelength-meter candidate ranking.
        return None


def smsr_db(reading: Reading | dict[str, Any]) -> float:
    if isinstance(reading, Reading):
        value = reading.side_mode_suppression_db
        count = reading.peak_count
    else:
        value = reading.get("side_mode_suppression_db")
        count = int(reading.get("peak_count", 0))
    if value is None and count == 1:
        return 99.0
    return float(value if value is not None else -999.0)


def reading_passes(
    reading: Reading,
    target_nm: float,
    target_power_mw: float,
    power_tolerance: float,
    minimum_smsr_db: float,
) -> bool:
    return (
        abs(reading.wavelength_nm - target_nm) * 1000.0 <= 2.0
        and smsr_db(reading) >= minimum_smsr_db
        and abs(reading.power_mw / target_power_mw - 1.0) <= power_tolerance
    )


def find_point_index(points, target_nm: float) -> int:
    index = int(round((target_nm - float(points[0].target_nm)) / 0.02))
    if not 0 <= index < len(points):
        raise ValueError(f"目标超出2001点范围：{target_nm:.2f} nm")
    if not math.isclose(float(points[index].target_nm), target_nm, abs_tol=1e-8):
        raise ValueError(f"目标不在0.02 nm网格上：{target_nm:.6f} nm")
    return index


def build_local_search_plan(points, index: int) -> dict[str, Any]:
    """Build bounded same-neighbour seeds for a newly discovered grid outlier."""

    if not 2 <= index < len(points) - 2:
        raise ValueError("自动邻域寻优要求目标左右各至少保留2个正式点")

    def combine(a, b, weight_a: float, weight_b: float):
        result = []
        for channel, (left, right) in enumerate(zip(a, b)):
            value = int(round(float(left) * weight_a + float(right) * weight_b))
            maximum_code = model.current_to_code(model.LASER_MAX_MA[channel], channel)
            result.append(min(max(value, 0), maximum_code))
        return tuple(LaserSerial.validate_codes(result))

    left2 = tuple(points[index - 2].codes)
    left1 = tuple(points[index - 1].codes)
    centre = tuple(points[index].codes)
    right1 = tuple(points[index + 1].codes)
    right2 = tuple(points[index + 2].codes)
    candidates = (
        ("left_branch_extrapolation", combine(left1, left2, 2.0, -1.0)),
        ("left_neighbour_branch", left1),
        ("two_sided_interpolation", combine(left1, right1, 0.5, 0.5)),
        ("right_neighbour_branch", right1),
        ("right_branch_extrapolation", combine(right1, right2, 2.0, -1.0)),
        ("formal_point", centre),
    )
    seeds = []
    seen = set()
    for name, codes in candidates:
        validated = tuple(LaserSerial.validate_codes(list(codes)))
        if validated in seen:
            continue
        seen.add(validated)
        seeds.append((name, validated))
    preset = list(seeds[0][1])
    preset[1] = model.current_to_code(135.0, 1)
    return {"seeds": tuple(seeds), "mode_preset": tuple(preset)}


def approach_target(
    laser: LanLaserAdapter,
    points,
    index: int,
    preset,
    predecessor_hold_s: float,
    preset_hold_s: float,
) -> None:
    laser.set_codes(points[max(0, index - 1)].codes)
    time.sleep(predecessor_hold_s)
    if preset is not None:
        laser.set_codes(preset)
        time.sleep(preset_hold_s)


def confirmation_replay(
    laser: LanLaserAdapter,
    meter: RetryingMeter,
    points,
    index: int,
    codes,
    preset,
    *,
    repeats: int,
    settle_s: float,
    predecessor_hold_s: float,
    preset_hold_s: float,
    target_power_mw: float,
    power_tolerance: float,
    minimum_smsr_db: float,
) -> dict[str, Any]:
    target_nm = float(points[index].target_nm)
    readings: list[dict[str, Any]] = []
    for repeat in range(repeats):
        approach_target(
            laser, points, index, preset,
            predecessor_hold_s, preset_hold_s,
        )
        laser.set_codes(codes)
        time.sleep(settle_s)
        reading = meter.measure(target_nm, select_main_peak=True)
        passed = reading_passes(
            reading, target_nm, target_power_mw,
            power_tolerance, minimum_smsr_db,
        )
        row = {
            "repeat": repeat + 1,
            "reading": asdict(reading),
            "wavelength_error_pm": (reading.wavelength_nm - target_nm) * 1000.0,
            "power_relative_error_percent":
                (reading.power_mw / target_power_mw - 1.0) * 100.0,
            "smsr_db": smsr_db(reading),
            "passed": passed,
        }
        readings.append(row)
        print(
            f"  confirm {repeat + 1}/{repeats} wave={reading.wavelength_nm:.8f} "
            f"err={row['wavelength_error_pm']:+.2f}pm "
            f"power={reading.power_mw:.4f}mW "
            f"SMSR={row['smsr_db']:.2f} ok={int(passed)}",
            flush=True,
        )
    return {
        "success": all(item["passed"] for item in readings),
        "preset_codes": list(preset) if preset is not None else None,
        "rows": readings,
    }


def close_equal_power_with_gain(
    laser: LanLaserAdapter,
    meter: RetryingMeter,
    seed_codes,
    target_nm: float,
    target_power_mw: float,
    settle_s: float,
    power_relative_tolerance: float,
    max_rounds: int,
    max_tune_measurements: int,
) -> dict[str, Any]:
    """Preserve SOA branch and reduce GAIN when SOA equalization mode-hops."""

    codes = list(seed_codes)
    history: list[dict[str, Any]] = []
    observations: list[tuple[float, float]] = []
    reading: Reading | None = None
    # GAIN also pulls wavelength on this device (about several pm/mA near the
    # 1539-nm branch).  Walk it in 1 mA increments so each WAVE-A/B correction
    # remains inside the measured local linear neighbourhood.
    rounds = max(24, max_rounds * 12)
    for round_index in range(rounds):
        tuned = model.tune_local_three_controls(
            laser,
            meter,
            codes,
            target_nm,
            settle_s,
            max_measurements=max_tune_measurements,
        )
        codes = [int(value) for value in tuned["codes"]]
        reading = Reading(**tuned["reading"])
        gain_ma = model.code_to_current(codes[0], 0)
        history.append({
            "action": f"gain_preserved_wavelength_close_{round_index}",
            "gain_ma": gain_ma,
            "codes": list(codes),
            "reading": asdict(reading),
            "tuning": tuned,
        })
        wave_ok = (
            abs(reading.wavelength_nm - target_nm) * 1000.0 <= 2.0
            and model.mode_ok(reading)
        )
        relative_error = reading.power_mw / target_power_mw - 1.0
        if wave_ok and abs(relative_error) <= power_relative_tolerance:
            break
        if not wave_ok:
            # A WAVE-A/B close that cannot stay on the selected mode is not a
            # usable derivative point; do not hide it with a power adjustment.
            break

        observations.append((gain_ma, reading.power_mw))
        slope = None
        if len(observations) >= 2:
            previous_gain, previous_power = observations[-2]
            delta_gain = gain_ma - previous_gain
            if abs(delta_gain) >= 0.5:
                measured = (reading.power_mw - previous_power) / delta_gain
                if 0.001 <= measured <= 0.05:
                    slope = measured
        if slope is None:
            # The proportional estimate is a conservative first inverse for
            # the gain medium; subsequent rounds replace it with live slope.
            requested = gain_ma * target_power_mw / max(reading.power_mw, 0.1)
        else:
            requested = gain_ma + (target_power_mw - reading.power_mw) / slope
        requested = min(max(requested, gain_ma - 1.0), gain_ma + 1.0)
        requested = min(max(requested, 30.0), 135.0)
        next_code = model.current_to_code(requested, 0)
        if next_code == codes[0]:
            break
        print(
            f"  gain-walk {round_index + 1}: {gain_ma:.3f} -> "
            f"{model.code_to_current(next_code, 0):.3f} mA "
            f"wave_err={(reading.wavelength_nm - target_nm) * 1000.0:+.2f}pm "
            f"power={reading.power_mw:.4f}mW",
            flush=True,
        )
        codes[0] = next_code

    if reading is None:
        raise RuntimeError("GAIN等功率未产生任何实测")
    wavelength_error_pm = (reading.wavelength_nm - target_nm) * 1000.0
    relative_error = reading.power_mw / target_power_mw - 1.0
    success = (
        abs(wavelength_error_pm) <= 2.0
        and model.mode_ok(reading)
        and abs(relative_error) <= power_relative_tolerance
    )
    return {
        "success": success,
        "target_nm": target_nm,
        "codes": codes,
        "currents_ma": [
            model.code_to_current(value, index) for index, value in enumerate(codes)
        ],
        "reading": asdict(reading),
        "wavelength_error_pm": wavelength_error_pm,
        "power_error_mw": reading.power_mw - target_power_mw,
        "power_relative_error": relative_error,
        "history": history,
        "power_control": "GAIN_with_SOA_preserved",
    }


def optimize_target(
    laser: LanLaserAdapter,
    meter: RetryingMeter,
    points,
    target_nm: float,
    plan: dict[str, Any],
    soa_curve,
    args,
) -> dict[str, Any]:
    index = find_point_index(points, target_nm)
    preset_options = (None, tuple(plan["mode_preset"]))
    attempts: list[dict[str, Any]] = []
    best_partial: dict[str, Any] | None = None

    for seed_name, raw_seed in plan["seeds"]:
        seed = list(LaserSerial.validate_codes(list(raw_seed)))
        for preset in preset_options:
            path_name = "direct_from_predecessor" if preset is None else "mode_preset"
            print(f"target={target_nm:.2f} seed={seed_name} path={path_name}", flush=True)
            approach_target(
                laser, points, index, preset,
                args.predecessor_hold_s, args.preset_hold_s,
            )
            laser.set_codes(seed)
            time.sleep(args.settle_s)
            screen = meter.measure(target_nm, select_main_peak=True)
            screen_row = {
                "reading": asdict(screen),
                "wavelength_error_pm": (screen.wavelength_nm - target_nm) * 1000.0,
                "smsr_db": smsr_db(screen),
            }
            print(
                f"  screen wave={screen.wavelength_nm:.8f} "
                f"err={screen_row['wavelength_error_pm']:+.2f}pm "
                f"power={screen.power_mw:.4f}mW SMSR={screen_row['smsr_db']:.2f}",
                flush=True,
            )
            attempt: dict[str, Any] = {
                "seed_name": seed_name,
                "seed_codes": seed,
                "path": path_name,
                "preset_codes": list(preset) if preset is not None else None,
                "screen": screen_row,
            }
            if (
                smsr_db(screen) < args.minimum_search_smsr_db
                or abs(screen.wavelength_nm - target_nm) > args.maximum_seed_error_nm
            ):
                attempt["rejected_at_screen"] = True
                attempts.append(attempt)
                continue

            soa_closed = model.close_equal_power_from_seed(
                laser,
                meter,
                seed,
                target_nm,
                args.target_power_mw,
                screen.power_mw,
                soa_curve,
                args.settle_s,
                args.power_relative_tolerance,
                args.max_rounds,
                args.max_tune_measurements,
            )
            closed = soa_closed
            attempt["soa_closed"] = soa_closed
            if not soa_closed["success"] and screen.power_mw > args.target_power_mw:
                # Retry from the exact same path.  Keeping SOA fixed is the
                # important fallback for branches that disappear as SOA falls.
                approach_target(
                    laser, points, index, preset,
                    args.predecessor_hold_s, args.preset_hold_s,
                )
                gain_closed = close_equal_power_with_gain(
                    laser,
                    meter,
                    seed,
                    target_nm,
                    args.target_power_mw,
                    args.settle_s,
                    args.power_relative_tolerance,
                    args.max_rounds,
                    args.max_tune_measurements,
                )
                attempt["gain_closed"] = gain_closed
                gain_reading = Reading(**gain_closed["reading"])
                soa_reading = Reading(**soa_closed["reading"])
                gain_score = (
                    (0.0 if smsr_db(gain_reading) >= args.minimum_smsr_db else 10000.0)
                    + abs(float(gain_closed["wavelength_error_pm"]))
                    + abs(float(gain_closed["power_relative_error"])) * 100.0
                )
                soa_score = (
                    (0.0 if smsr_db(soa_reading) >= args.minimum_smsr_db else 10000.0)
                    + abs(float(soa_closed["wavelength_error_pm"]))
                    + abs(float(soa_closed["power_relative_error"])) * 100.0
                )
                if gain_closed["success"] or gain_score < soa_score:
                    closed = gain_closed
            attempt["closed"] = closed
            candidate_reading = Reading(**closed["reading"])
            partial_score = (
                (0.0 if smsr_db(candidate_reading) >= args.minimum_smsr_db else 10000.0)
                + abs(float(closed["wavelength_error_pm"]))
                + abs(float(closed["power_relative_error"])) * 100.0
                - min(smsr_db(candidate_reading), 40.0) * 0.01
            )
            attempt["partial_score"] = partial_score
            if best_partial is None or partial_score < best_partial["partial_score"]:
                best_partial = attempt

            if not closed["success"]:
                attempts.append(attempt)
                continue

            confirmation = confirmation_replay(
                laser, meter, points, index, closed["codes"], preset,
                repeats=args.confirmations,
                settle_s=args.settle_s,
                predecessor_hold_s=args.predecessor_hold_s,
                preset_hold_s=args.preset_hold_s,
                target_power_mw=args.target_power_mw,
                power_tolerance=args.power_relative_tolerance,
                minimum_smsr_db=args.minimum_smsr_db,
            )
            attempt["confirmation"] = confirmation
            attempts.append(attempt)
            if confirmation["success"]:
                return {
                    "success": True,
                    "target_nm": target_nm,
                    "index": index,
                    "codes": list(closed["codes"]),
                    "currents_ma": list(closed["currents_ma"]),
                    "transition_preset_codes": confirmation["preset_codes"],
                    "seed_name": seed_name,
                    "path": path_name,
                    "attempts": attempts,
                }

    return {
        "success": False,
        "target_nm": target_nm,
        "index": index,
        "best_partial": best_partial,
        "attempts": attempts,
    }


def parse_targets(value: str) -> list[float]:
    targets = [round(float(item.strip()), 2) for item in value.split(",") if item.strip()]
    if not targets:
        raise ValueError("至少需要一个目标波长")
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.3.46")
    parser.add_argument(
        "--port",
        default="",
        help="use an exact-ACK USB CDC port (for example COM6) instead of LAN",
    )
    parser.add_argument("--meter", default="GPIB0::7::INSTR")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--targets", default=",".join(f"{v:.2f}" for v in DEFAULT_TARGETS))
    parser.add_argument("--target-power-mw", type=float, default=1.530)
    parser.add_argument("--power-relative-tolerance", type=float, default=0.005)
    parser.add_argument("--minimum-smsr-db", type=float, default=20.0)
    parser.add_argument("--minimum-search-smsr-db", type=float, default=15.0)
    parser.add_argument("--maximum-seed-error-nm", type=float, default=0.12)
    parser.add_argument("--settle-s", type=float, default=0.35)
    parser.add_argument("--predecessor-hold-s", type=float, default=0.35)
    parser.add_argument("--preset-hold-s", type=float, default=0.30)
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--max-tune-measurements", type=int, default=10)
    parser.add_argument("--confirmations", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    targets = parse_targets(args.targets)
    if not 1.45 <= args.target_power_mw <= 1.70:
        raise SystemExit("目标功率必须位于1.45–1.70 mW")
    if not 0.18 <= args.settle_s <= 2.0:
        raise SystemExit("稳定等待必须位于0.18–2.0秒")
    if args.confirmations < 3:
        raise SystemExit("独立复测不得少于3次")

    points = app.load_fullband_accuracy_table(HERE)
    for target in targets:
        index = find_point_index(points, target)
        if target not in SEARCH_PLANS:
            SEARCH_PLANS[target] = build_local_search_plan(points, index)
    # Validate every static search/preset code before opening the hardware.
    for target in targets:
        find_point_index(points, target)
        plan = SEARCH_PLANS[target]
        LaserSerial.validate_codes(list(plan["mode_preset"]))
        for _, codes in plan["seeds"]:
            LaserSerial.validate_codes(list(codes))
    if args.dry_run:
        print(f"dry_run=PASS targets={len(targets)}", flush=True)
        return 0

    soa_curve, soa_payload = model.load_soa_transfer_curve(SOA_CURVE)
    if args.resume and args.output.exists():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        payload["targets_nm"] = targets
        payload["target_power_mw"] = args.target_power_mw
        payload["power_relative_tolerance"] = args.power_relative_tolerance
        payload["minimum_smsr_db"] = args.minimum_smsr_db
    else:
        payload = {
            "created": datetime.now().astimezone().isoformat(timespec="seconds"),
            "method": (
                "USB_forward_path_same_branch_local_Jacobian_SOA_equalization"
                if args.port
                else "LAN_forward_path_same_branch_local_Jacobian_SOA_equalization"
            ),
            "source_table": "fullband_equal_power_operational_2001.json",
            "targets_nm": targets,
            "target_power_mw": args.target_power_mw,
            "wavelength_tolerance_pm": 2.0,
            "power_relative_tolerance": args.power_relative_tolerance,
            "minimum_smsr_db": args.minimum_smsr_db,
            "limits_ma": {
                "gain": 135.0,
                "soa": 135.0,
                "phase": 10.0,
                "wave_a": 30.0,
                "wave_b": 30.0,
            },
            "soa_curve_source": SOA_CURVE.name,
            "soa_curve_summary": soa_payload.get("summary", {}),
            "rows": {},
        }

    if args.port:
        if serial is None:
            raise RuntimeError("当前Python环境缺少pyserial，无法使用USB寻优")
        device = serial.Serial(
            port=args.port,
            baudrate=115200,
            timeout=0.10,
            write_timeout=3.0,
        )
    else:
        device = FbgLanSerial(board_host=args.host, timeout=0.10, write_timeout=3.0)
    worker = app.UnlimitedAccuracyWorker(device, points, {})
    laser = LanLaserAdapter(worker)
    meter: RetryingMeter | None = None
    shutter_confirmed = False
    try:
        if not args.port:
            wait_connected(device)
        device.write(app.build_work_mode_command(2))
        time.sleep(0.12)
        device.reset_input_buffer()
        worker._rx_buffer.clear()
        meter = RetryingMeter(args.meter)
        payload["meter"] = meter.identity

        for number, target_nm in enumerate(targets, start=1):
            key = f"{target_nm:.2f}"
            if args.resume and payload.get("rows", {}).get(key, {}).get("success"):
                print(f"skip accepted target={target_nm:.2f}", flush=True)
                continue
            print(f"\n[{number}/{len(targets)}] optimize {target_nm:.2f} nm", flush=True)
            result = optimize_target(
                laser,
                meter,
                points,
                target_nm,
                SEARCH_PLANS[target_nm],
                soa_curve,
                args,
            )
            payload["rows"][key] = result
            save_json(args.output, payload)
            print(
                f"result target={target_nm:.2f} accepted={int(result['success'])}",
                flush=True,
            )

        accepted = [row for row in payload["rows"].values() if row.get("success")]
        payload["summary"] = {
            "requested": len(targets),
            "accepted": len(accepted),
            "unresolved": len(targets) - len(accepted),
            "accepted_targets_nm": [row["target_nm"] for row in accepted],
            "unresolved_targets_nm": [
                target for target in targets
                if not payload["rows"].get(f"{target:.2f}", {}).get("success")
            ],
        }
        payload["finished"] = datetime.now().astimezone().isoformat(timespec="seconds")
        save_json(args.output, payload)
        print("summary=" + json.dumps(payload["summary"], ensure_ascii=False), flush=True)
        return 0 if len(accepted) == len(targets) else 2
    finally:
        try:
            if device.is_open:
                shutter_confirmed = bool(
                    command_with_link_retry(
                        worker,
                        worker._last_codes,
                        soa_mode=app.PI11210_SOA_SHUTTER_MODE,
                    )
                )
        finally:
            if meter is not None:
                meter.close()
            device.close()
            payload["soa_shutter_confirmed"] = shutter_confirmed
            save_json(args.output, payload)
            print(f"soa_shutter_confirmed={int(shutter_confirmed)}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
