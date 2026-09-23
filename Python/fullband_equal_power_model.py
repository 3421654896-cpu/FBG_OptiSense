"""Full-band, equal-power, single-mode calibration for XCWSLD-C10.

The laser is not globally smooth: WAVE-A/WAVE-B select Vernier branches and
PHASE tunes the longitudinal mode inside one branch.  This module therefore
uses a measured candidate library plus local branch models instead of one
global polynomial.  GAIN/SOA are treated as power controls; PDT (reference PD)
and PDR (etalon PD) are measured feedback features, never DAC outputs.

The live workflow is deliberately resumable:

1. ``inventory`` collects only wavelength-meter measurements made by this
   project (the old spreadsheet is explicitly excluded).
2. ``pilot`` probes sparse targets to establish reachable branches and the
   common maximum-power ceiling.
3. ``calibrate`` fills the requested 0.02 nm grid and checkpoints every point.
4. ``validate`` repeats every row in both scan directions before export.

Only this dedicated 2001-point workflow may use the authorized 145 mA
GAIN/SOA ceiling.  PHASE remains below 10 mA and WAVE-A/B below 30 mA.  Other
desktop and reduced-route modes retain their independent 135 mA limit.
Entering EXTRA mode also forces the board fan to full speed in current firmware.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from laser_dac_safety import FULLBAND_2001_CODE_LIMITS
from laser_power_calibration import (
    AQ6150B,
    LaserSerial,
    Reading,
    code_to_current,
    save_json,
    take_reading,
)


HERE = Path(__file__).resolve().parent
DEFAULT_INVENTORY = HERE / "fullband_measurement_inventory.json"
DEFAULT_PILOT = HERE / "fullband_equal_power_pilot.json"
DEFAULT_CHECKPOINT = HERE / "fullband_equal_power_2001_checkpoint.json"
DEFAULT_OUTPUT = HERE / "fullband_equal_power_2001.json"
DEFAULT_MODEL = HERE / "fullband_equal_power_surrogate.joblib"
DEFAULT_MODE_MAP = HERE / "fullband_mode_map_1p5mA.json"
DEFAULT_DENSE_RAW = HERE / "fullband_dense_raw_2001.json"
DEFAULT_PHASE_BRANCHES = HERE / "fullband_phase_sweep_branches.json"
DEFAULT_PHASE_MAP = HERE / "fullband_phase_sweep_map.json"
DEFAULT_PHASE_CANDIDATES = HERE / "fullband_phase_interpolated_candidates.json"

CONTROL_NAMES = ("gain_ma", "soa_ma", "phase_ma", "wave_a_ma", "wave_b_ma")
TARGET_START_NM = 1525.0
TARGET_STOP_NM = 1565.0
TARGET_STEP_NM = 0.02
WAVELENGTH_TOLERANCE_PM = 2.0
POWER_REL_TOLERANCE = 0.005
MIN_SMSR_DB = 20.0
# A repaired row must not merely scrape through the release boundary.  The
# subsequent 2001-point sweep takes long enough for small thermal/power drift
# to move such a row outside the release gate again.  Repair therefore aims
# for this inner guard band while the final independent certification keeps
# using WAVELENGTH_TOLERANCE_PM and the operator-selected power tolerance.
REPAIR_WAVELENGTH_TARGET_PM = 1.0
REPAIR_POWER_TARGET_RELATIVE = 0.01
# The second full forward sweep showed that every newly failed ordinary row
# had been inside the former 1.5 pm guard on the prior sweep (maximum 1.48 pm),
# while rows proactively re-centred by the repair had zero repeat failures.
# Select rows outside +/-0.5 pm for inner-band reclosure; the public release
# gate remains unchanged at +/-2 pm.
CERTIFICATION_REWORK_WAVELENGTH_GUARD_PM = 0.5
CERTIFICATION_REWORK_POWER_GUARD_RELATIVE = 0.02
# Errors this large are not ordinary picometre drift.  They mean that the
# exact DAC row is valid in isolation but enters a different longitudinal mode
# when reached from the preceding forward-scan row.  Such rows need an actual
# forward-transition confirmation during repair.
SEQUENTIAL_MODE_HOP_THRESHOLD_PM = 100.0
LASER_MAX_MA = (145.0, 145.0, 10.0, 30.0, 30.0)
MAX_GAIN_SOA_MA = 145.0
# Yokogawa recommends averaged measurement for an unstable optical input.
# Three sweeps average wavelength and peak power inside one trigger, ensuring
# calibration and certification close on the same acquisition statistic.
AQ_AVERAGE_COUNT = 3
CALIBRATION_CONFIRMATION_SAMPLES = 3
CALIBRATION_CONFIRM_SETTLE_S = 0.02


def current_to_code(current_ma: float, channel: int) -> int:
    """Convert a full-band current without crossing its exact board limit."""

    index = int(channel)
    current = min(max(float(current_ma), 0.0), LASER_MAX_MA[index])
    full_scale = (150.0, 150.0, 20.0, 80.0, 80.0)[index]
    code = round(current / full_scale * 65536.0)
    return min(max(int(code), 0), FULLBAND_2001_CODE_LIMITS[index])


def open_fullband_laser(port: str) -> LaserSerial:
    """Open the laser with the opt-in 2001-point 145 mA validation profile."""

    return LaserSerial(port, code_limits=FULLBAND_2001_CODE_LIMITS)


def validate_requested_currents(args: argparse.Namespace) -> None:
    """Reject misleading CLI currents before any serial instrument is opened."""

    limits = {
        "gain_ma": LASER_MAX_MA[0],
        "soa_ma": LASER_MAX_MA[1],
        "soa_min_ma": LASER_MAX_MA[1],
        "soa_max_ma": LASER_MAX_MA[1],
        "phase_ma": LASER_MAX_MA[2],
        "phase_start_ma": LASER_MAX_MA[2],
        "phase_stop_ma": LASER_MAX_MA[2],
        "wave_a_start_ma": LASER_MAX_MA[3],
        "wave_a_stop_ma": LASER_MAX_MA[3],
        "wave_b_start_ma": LASER_MAX_MA[4],
        "wave_b_stop_ma": LASER_MAX_MA[4],
    }
    for name, maximum in limits.items():
        value = getattr(args, name, None)
        if value is None:
            continue
        current = float(value)
        if not math.isfinite(current) or not 0.0 <= current <= maximum:
            raise ValueError(f"{name} must be within 0..{maximum:g} mA")
    soa_min = getattr(args, "soa_min_ma", None)
    soa_max = getattr(args, "soa_max_ma", None)
    if soa_min is not None and soa_max is not None and float(soa_min) >= float(soa_max):
        raise ValueError("soa_min_ma must be lower than soa_max_ma")


@dataclass(frozen=True)
class Candidate:
    codes: tuple[int, int, int, int, int]
    wavelength_nm: float
    power_mw: float
    smsr_db: float
    peak_count: int
    pdt_code: int | None
    pdr_code: int | None
    source: str

    @property
    def currents_ma(self) -> tuple[float, float, float, float, float]:
        return tuple(code_to_current(v, i) for i, v in enumerate(self.codes))


def utc_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def target_grid(start_nm: float, stop_nm: float, step_nm: float) -> list[float]:
    count = int(round((stop_nm - start_nm) / step_nm)) + 1
    return [round(start_nm + index * step_nm, 6) for index in range(count)]


def normalized_smsr(reading: dict[str, Any]) -> float:
    value = reading.get("side_mode_suppression_db")
    if value is None and int(reading.get("peak_count", 0)) == 1:
        return 99.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return -999.0


def valid_codes(codes: Iterable[Any]) -> tuple[int, int, int, int, int] | None:
    try:
        values = tuple(int(value) for value in codes)
    except (TypeError, ValueError):
        return None
    if len(values) != 5 or any(value < 0 or value > 0xFFFF for value in values):
        return None
    normalized = list(values)
    for index, (value, maximum) in enumerate(
        zip(normalized, FULLBAND_2001_CODE_LIMITS)
    ):
        if value <= maximum:
            continue
        # Historical 10/30 mA exports used the mathematical boundary one LSB
        # above the board's strict integer clamp.  Preserve those measurements
        # while rejecting any GAIN/SOA code above the new 145 mA profile.
        if index >= 2 and value == maximum + 1:
            normalized[index] = maximum
            continue
        return None
    return tuple(normalized)  # type: ignore[return-value]


def walk_measurements(node: Any, source: str) -> Iterable[Candidate]:
    """Yield every embedded live meter measurement with its exact DAC codes."""
    if isinstance(node, dict):
        codes = valid_codes(node.get("codes", ()))
        reading = node.get("reading")
        if codes is not None and isinstance(reading, dict):
            try:
                wavelength = float(reading["wavelength_nm"])
                power = float(reading["power_mw"])
                peak_count = int(reading.get("peak_count", 0))
            except (KeyError, TypeError, ValueError):
                pass
            else:
                if math.isfinite(wavelength) and math.isfinite(power) and power > 0.0:
                    yield Candidate(
                        codes=codes,
                        wavelength_nm=wavelength,
                        power_mw=power,
                        smsr_db=normalized_smsr(reading),
                        peak_count=peak_count,
                        pdt_code=(int(reading["pdt_code"])
                                  if reading.get("pdt_code") is not None else None),
                        pdr_code=(int(reading["pdr_code"])
                                  if reading.get("pdr_code") is not None else None),
                        source=source,
                    )
        for value in node.values():
            yield from walk_measurements(value, source)
    elif isinstance(node, list):
        for value in node:
            yield from walk_measurements(value, source)


def discover_json_sources() -> list[Path]:
    patterns = (
        "*high_power*.json",
        "*single_mode*.json",
        "*uniform_power*.json",
        "*calibrat*.json",
        "*validation*.json",
        "*candidate*.json",
        "*tuning_map*.json",
        "*mode_map*.json",
        "*phase_sweep_map*.json",
        "*fullband_equal_power_pilot*.json",
    )
    paths: set[Path] = set()
    # Reuse every actual full-band pilot/validation measurement.  Exclude only
    # generated inventories/checkpoints that would recursively duplicate them.
    excluded_tokens = (
        "fullband_measurement_inventory",
        "fullband_equal_power_2001",
        "总表",
    )
    for pattern in patterns:
        for path in HERE.glob(pattern):
            if path.is_file() and not any(token in path.name for token in excluded_tokens):
                paths.add(path)
    return sorted(paths)


def build_inventory(output: Path = DEFAULT_INVENTORY) -> dict[str, Any]:
    candidates: list[Candidate] = []
    parse_errors: list[dict[str, str]] = []
    for path in discover_json_sources():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # retain the filename and continue inventory
            parse_errors.append({"source": path.name, "error": str(exc)})
            continue
        candidates.extend(walk_measurements(payload, path.name))

    # Repeated measurements are useful for repeatability but exact duplicates
    # caused by nested report copies are not.  Keep one of each identical row.
    unique: dict[tuple[Any, ...], Candidate] = {}
    for item in candidates:
        key = (item.codes, round(item.wavelength_nm, 6), round(item.power_mw, 7),
               round(item.smsr_db, 4), item.pdt_code, item.pdr_code)
        unique.setdefault(key, item)
    candidates = list(unique.values())
    clean = [item for item in candidates if item.smsr_db >= MIN_SMSR_DB]
    in_requested_band = [
        item for item in clean
        if TARGET_START_NM - 0.5 <= item.wavelength_nm <= TARGET_STOP_NM + 0.5
    ]
    targets = target_grid(TARGET_START_NM, TARGET_STOP_NM, TARGET_STEP_NM)
    nearest_errors_pm = [
        min(abs(item.wavelength_nm - target) for item in in_requested_band) * 1000.0
        for target in targets
    ] if in_requested_band else []

    payload = {
        "created": utc_now(),
        "purpose": "measured_seed_library_for_fullband_equal_power_model",
        "old_total_spreadsheet_used": False,
        "limits_ma": dict(zip(CONTROL_NAMES, LASER_MAX_MA)),
        "single_mode_gate_smsr_db": MIN_SMSR_DB,
        "sources": [path.name for path in discover_json_sources()],
        "parse_errors": parse_errors,
        "summary": {
            "raw_embedded_measurements": len(unique),
            "clean_single_mode_measurements": len(clean),
            "clean_measurements_near_requested_band": len(in_requested_band),
            "measured_wavelength_min_nm": min((item.wavelength_nm for item in clean), default=None),
            "measured_wavelength_max_nm": max((item.wavelength_nm for item in clean), default=None),
            "measured_power_min_mw": min((item.power_mw for item in clean), default=None),
            "measured_power_max_mw": max((item.power_mw for item in clean), default=None),
            "grid_point_count": len(targets),
            "nearest_seed_error_median_pm": (
                statistics.median(nearest_errors_pm) if nearest_errors_pm else None
            ),
            "nearest_seed_error_max_pm": max(nearest_errors_pm, default=None),
        },
        "candidates": [
            {
                "codes": list(item.codes),
                "currents_ma": list(item.currents_ma),
                "wavelength_nm": item.wavelength_nm,
                "power_mw": item.power_mw,
                "smsr_db": item.smsr_db,
                "peak_count": item.peak_count,
                "pdt_code": item.pdt_code,
                "pdr_code": item.pdr_code,
                "source": item.source,
            }
            for item in sorted(clean, key=lambda value: value.wavelength_nm)
        ],
    }
    save_json(output, payload)
    return payload


def load_all_measured_candidates() -> list[Candidate]:
    """Load clean and rejected observations for regression/classification."""
    result: list[Candidate] = []
    seen: set[tuple[Any, ...]] = set()
    for path in discover_json_sources():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for item in walk_measurements(payload, path.name):
            key = (item.codes, round(item.wavelength_nm, 6), round(item.power_mw, 7),
                   round(item.smsr_db, 4), item.pdt_code, item.pdr_code)
            if key not in seen:
                seen.add(key)
                result.append(item)
    return result


def candidate_features(items: Iterable[Candidate]) -> np.ndarray:
    return np.asarray([item.currents_ma for item in items], dtype=np.float64)


def train_surrogate(output: Path = DEFAULT_MODEL) -> dict[str, Any]:
    """Train discontinuity-preserving control-to-optical surrogate models."""
    all_rows = load_all_measured_candidates()
    if len(all_rows) < 100:
        raise RuntimeError("Too few live measurements to train the tuning model")
    clean_rows = [
        item for item in all_rows
        if item.smsr_db >= MIN_SMSR_DB and 1518.0 <= item.wavelength_nm <= 1572.0
    ]
    x_all = candidate_features(all_rows)
    y_clean_mode = np.asarray(
        [item.smsr_db >= MIN_SMSR_DB for item in all_rows], dtype=np.int8
    )
    x_clean = candidate_features(clean_rows)
    wavelength = np.asarray([item.wavelength_nm for item in clean_rows])
    power = np.asarray([item.power_mw for item in clean_rows])
    pdt = np.asarray([
        np.nan if item.pdt_code is None else float(item.pdt_code) for item in clean_rows
    ])
    pdr = np.asarray([
        np.nan if item.pdr_code is None else float(item.pdr_code) for item in clean_rows
    ])

    regressor_options = dict(
        n_estimators=480,
        min_samples_leaf=1,
        max_features=1.0,
        n_jobs=-1,
        random_state=20260830,
    )
    wavelength_model = ExtraTreesRegressor(**regressor_options).fit(x_clean, wavelength)
    power_model = ExtraTreesRegressor(**regressor_options).fit(x_clean, power)
    classifier = ExtraTreesClassifier(
        n_estimators=480, min_samples_leaf=1, max_features=1.0,
        class_weight="balanced", n_jobs=-1, random_state=20260831,
    ).fit(x_all, y_clean_mode)
    pdt_mask, pdr_mask = np.isfinite(pdt), np.isfinite(pdr)
    pdt_model = ExtraTreesRegressor(**regressor_options).fit(x_clean[pdt_mask], pdt[pdt_mask])
    pdr_model = ExtraTreesRegressor(**regressor_options).fit(x_clean[pdr_mask], pdr[pdr_mask])

    # Split by exact DAC combination so repeated copies of one hardware point
    # cannot leak from train into validation.
    groups = np.asarray([hash(item.codes) for item in clean_rows], dtype=np.int64)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=20260830)
    train_idx, test_idx = next(splitter.split(x_clean, wavelength, groups=groups))
    cv_wave = ExtraTreesRegressor(**regressor_options).fit(x_clean[train_idx], wavelength[train_idx])
    cv_power = ExtraTreesRegressor(**regressor_options).fit(x_clean[train_idx], power[train_idx])
    mode_groups = np.asarray([hash(item.codes) for item in all_rows], dtype=np.int64)
    ctrain, ctest = next(splitter.split(x_all, y_clean_mode, groups=mode_groups))
    cv_mode = ExtraTreesClassifier(
        n_estimators=320, min_samples_leaf=1, max_features=1.0,
        class_weight="balanced", n_jobs=-1, random_state=20260831,
    ).fit(x_all[ctrain], y_clean_mode[ctrain])
    mode_probability = cv_mode.predict_proba(x_all[ctest])[:, 1]
    metrics = {
        "wavelength_mae_pm_group_holdout": float(
            mean_absolute_error(wavelength[test_idx], cv_wave.predict(x_clean[test_idx])) * 1000.0
        ),
        "power_mae_mw_group_holdout": float(
            mean_absolute_error(power[test_idx], cv_power.predict(x_clean[test_idx]))
        ),
        "single_mode_auc_group_holdout": float(
            roc_auc_score(y_clean_mode[ctest], mode_probability)
        ) if len(np.unique(y_clean_mode[ctest])) == 2 else None,
    }
    bundle = {
        "created": utc_now(),
        "control_names": CONTROL_NAMES,
        "limits_ma": LASER_MAX_MA,
        "minimum_smsr_db": MIN_SMSR_DB,
        "old_total_spreadsheet_used": False,
        "training_counts": {"all": len(all_rows), "single_mode": len(clean_rows)},
        "metrics": metrics,
        "wavelength_model": wavelength_model,
        "power_model": power_model,
        "single_mode_model": classifier,
        "pdt_model": pdt_model,
        "pdr_model": pdr_model,
    }
    joblib.dump(bundle, output, compress=3)
    return bundle


def model_seed_candidates(
    bundle: dict[str, Any],
    library: list[Candidate],
    target_nm: float,
    count: int = 12,
) -> list[list[int]]:
    """Invert the surrogate over a large software-only candidate population."""
    rng = np.random.default_rng(int(round(target_nm * 1000.0)) + 20260830)
    base = np.asarray([item.currents_ma for item in library], dtype=np.float64)
    # Build dense local clouds around measured tuning branches.  This is more
    # reliable than treating the mostly empty 3-D control cube as uniformly
    # sampled, but a global cloud is included to discover missing branches.
    repeat = 10
    local = np.repeat(base[:, 2:5], repeat, axis=0)
    local += rng.normal(
        loc=0.0, scale=np.asarray([0.75, 0.65, 0.65]), size=local.shape
    )
    global_count = max(60000, len(local) * 2)
    global_controls = rng.uniform(
        low=np.asarray([0.0, 0.0, 0.0]),
        high=np.asarray([10.0, 30.0, 30.0]),
        size=(global_count, 3),
    )
    tuning = np.vstack((base[:, 2:5], local, global_controls))
    tuning = np.clip(tuning, [0.0, 0.0, 0.0], [10.0, 30.0, 30.0])
    controls = np.column_stack((
        np.full(len(tuning), MAX_GAIN_SOA_MA),
        np.full(len(tuning), MAX_GAIN_SOA_MA),
        tuning,
    ))
    predicted_wave = bundle["wavelength_model"].predict(controls)
    predicted_power = bundle["power_model"].predict(controls)
    mode_probability = bundle["single_mode_model"].predict_proba(controls)[:, 1]
    # A 1 pm wavelength miss dominates roughly 0.01 mW of predicted power.
    score = (
        np.abs(predicted_wave - target_nm) * 1000.0
        + np.maximum(0.0, 0.75 - mode_probability) * 400.0
        - np.minimum(predicted_power, 3.0) * 0.10
    )
    ordered = np.argsort(score)
    selected: list[list[int]] = []
    selected_tuning: list[np.ndarray] = []
    for index in ordered:
        point = tuning[index]
        # Force branch diversity.  Several nearly identical model optima do
        # not provide useful fallbacks when one lies on the wrong side of a hop.
        if any(np.linalg.norm((point - old) / [1.0, 1.5, 1.5]) < 0.35
               for old in selected_tuning):
            continue
        codes = [
            current_to_code(MAX_GAIN_SOA_MA, 0),
            current_to_code(MAX_GAIN_SOA_MA, 1),
            current_to_code(float(point[0]), 2),
            current_to_code(float(point[1]), 3),
            current_to_code(float(point[2]), 4),
        ]
        selected.append(codes)
        selected_tuning.append(point)
        if len(selected) >= count:
            break
    return selected


def load_candidates(path: Path = DEFAULT_INVENTORY) -> list[Candidate]:
    if not path.exists():
        build_inventory(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: list[Candidate] = []
    for row in payload.get("candidates", []):
        codes = valid_codes(row.get("codes", ()))
        if codes is None:
            continue
        result.append(Candidate(
            codes=codes,
            wavelength_nm=float(row["wavelength_nm"]),
            power_mw=float(row["power_mw"]),
            smsr_db=float(row["smsr_db"]),
            peak_count=int(row.get("peak_count", 0)),
            pdt_code=row.get("pdt_code"),
            pdr_code=row.get("pdr_code"),
            source=str(row.get("source", "inventory")),
        ))
    return result


def candidate_score(item: Candidate, target_nm: float) -> tuple[float, float, float]:
    # Wavelength branch proximity dominates; power and mode margin break ties.
    return (abs(item.wavelength_nm - target_nm), -item.power_mw, -item.smsr_db)


def seed_candidates(library: list[Candidate], target_nm: float, count: int = 10) -> list[Candidate]:
    ordered = sorted(library, key=lambda item: candidate_score(item, target_nm))
    result: list[Candidate] = []
    seen_controls: set[tuple[int, int, int]] = set()
    for item in ordered:
        # Keep distinct tuning branches; GAIN/SOA are replaced by the safe
        # discovery ceiling, so branch identity is PHASE/WAVE-A/WAVE-B.
        identity = (item.codes[2], item.codes[3], item.codes[4])
        if identity in seen_controls:
            continue
        seen_controls.add(identity)
        result.append(item)
        if len(result) >= count:
            break
    return result


def at_discovery_power(codes: Iterable[int]) -> list[int]:
    result = list(codes)
    result[0] = current_to_code(MAX_GAIN_SOA_MA, 0)
    result[1] = current_to_code(MAX_GAIN_SOA_MA, 1)
    return result


def mode_ok(reading: Reading) -> bool:
    return (
        reading.side_mode_suppression_db is None and reading.peak_count == 1
    ) or (
        reading.side_mode_suppression_db is not None
        and reading.side_mode_suppression_db >= MIN_SMSR_DB
    )


def recommend_common_power_target(
    rows: dict[str, dict[str, Any]],
    *,
    headroom_relative: float = POWER_REL_TOLERANCE,
    lift_relative: float = 0.0,
    resolution_mw: float = 0.001,
) -> dict[str, Any]:
    """Choose the highest robust common power supported by every grid row.

    The input must be a complete, live-measured high-power table.  The weakest
    wavelength defines the physical common ceiling.  ``headroom_relative``
    reserves downward drift margin.  ``lift_relative`` instead permits the
    requested common target to sit slightly above that weakest measured point
    when the release tolerance explicitly allows it; this raises the useful
    full-band average without pretending that the weakest point reaches the
    nominal target exactly.  The two margins may be combined, then the result
    is rounded down to the requested power resolution.
    """

    if not 0.0 <= float(headroom_relative) < 0.10:
        raise ValueError("power headroom must be in [0, 0.10)")
    if not 0.0 <= float(lift_relative) < 0.10:
        raise ValueError("power lift must be in [0, 0.10)")
    if not math.isfinite(float(resolution_mw)) or float(resolution_mw) <= 0.0:
        raise ValueError("power resolution must be positive and finite")
    ordered = sorted(rows.values(), key=lambda row: float(row["target_nm"]))
    expected_targets = target_grid(TARGET_START_NM, TARGET_STOP_NM, TARGET_STEP_NM)
    if len(ordered) != len(expected_targets):
        raise ValueError("high-power source must contain exactly 2001 rows")

    powers: list[float] = []
    for expected, row in zip(expected_targets, ordered):
        target = float(row["target_nm"])
        if not math.isclose(target, expected, abs_tol=1e-7):
            raise ValueError(f"high-power source is missing target {expected:.2f} nm")
        if not row.get("success"):
            raise ValueError(f"high-power source row {target:.2f} nm is not validated")
        codes = valid_codes(row.get("codes", ()))
        if codes is None:
            raise ValueError(f"high-power source row {target:.2f} nm exceeds 145 mA limits")
        reading = row.get("reading", {})
        power = float(reading.get("power_mw", float("nan")))
        if not math.isfinite(power) or power <= 0.0:
            raise ValueError(f"high-power source row {target:.2f} nm has invalid power")
        powers.append(power)

    common_ceiling = min(powers)
    unrounded = (
        common_ceiling
        * (1.0 - float(headroom_relative))
        / (1.0 - float(lift_relative))
    )
    resolution = float(resolution_mw)
    recommended = math.floor((unrounded + 1e-12) / resolution) * resolution
    if recommended <= 0.0:
        raise ValueError("recommended common power is not positive")
    weakest_index = min(range(len(powers)), key=powers.__getitem__)
    return {
        "method": "minimum_measured_single_mode_power_with_tolerance_aware_lift",
        "tested": len(ordered),
        "common_ceiling_mw": common_ceiling,
        "weakest_target_nm": expected_targets[weakest_index],
        "headroom_relative": float(headroom_relative),
        "lift_relative": float(lift_relative),
        "resolution_mw": resolution,
        "recommended_target_power_mw": recommended,
    }


def reading_score(reading: Reading, target_nm: float) -> tuple[float, float, float]:
    mode_penalty = 0.0 if mode_ok(reading) else 1000.0
    smsr = 99.0 if reading.side_mode_suppression_db is None else reading.side_mode_suppression_db
    return (mode_penalty + abs(reading.wavelength_nm - target_nm) * 1000.0,
            -reading.power_mw, -smsr)


def probe(
    laser: LaserSerial,
    meter: AQ6150B,
    codes: list[int],
    target_nm: float,
    settle_s: float,
) -> tuple[list[int], Reading]:
    LaserSerial.validate_codes(codes, FULLBAND_2001_CODE_LIMITS)
    reading = take_reading(
        laser, meter, codes, target_nm,
        settle_s=settle_s, select_main_peak=True,
    )
    return list(codes), reading


def tune_phase_on_branch(
    laser: LaserSerial,
    meter: AQ6150B,
    seed_codes: list[int],
    target_nm: float,
    settle_s: float,
    max_measurements: int = 10,
    wavelength_tolerance_pm: float = WAVELENGTH_TOLERANCE_PM,
) -> dict[str, Any]:
    """Tune PHASE locally while rejecting mode hops and dual-mode points."""
    history: list[dict[str, Any]] = []
    codes, current = probe(laser, meter, seed_codes, target_nm, settle_s)
    history.append({"codes": codes, "reading": asdict(current), "action": "seed"})
    best_codes, best = codes, current

    # Local finite-difference probe.  Current literature and prior device data
    # put the PHASE slope around tens of pm/mA, but its sign/size changes by
    # branch, so always measure it rather than assuming one global value.
    phase_ma = code_to_current(codes[2], 2)
    direction = 1.0 if phase_ma <= 9.85 else -1.0
    probe_phase = min(max(phase_ma + direction * 0.12, 0.0), LASER_MAX_MA[2])
    if abs(probe_phase - phase_ma) >= 0.02 and len(history) < max_measurements:
        trial_codes = list(codes)
        trial_codes[2] = current_to_code(probe_phase, 2)
        _, trial = probe(laser, meter, trial_codes, target_nm, settle_s)
        history.append({"codes": trial_codes, "reading": asdict(trial), "action": "phase_slope"})
        if reading_score(trial, target_nm) < reading_score(best, target_nm):
            best_codes, best = trial_codes, trial
        delta_nm = trial.wavelength_nm - current.wavelength_nm
        slope_nm_per_ma = delta_nm / (probe_phase - phase_ma)
    else:
        slope_nm_per_ma = float("nan")

    # A local slope is valid only while both points stayed on the same cavity
    # mode.  Large jumps are mode hops, not derivatives.
    if not math.isfinite(slope_nm_per_ma) or not (0.005 <= abs(slope_nm_per_ma) <= 1.0):
        slope_nm_per_ma = -0.075

    last_phase = code_to_current(best_codes[2], 2)
    last_wave = best.wavelength_nm
    for iteration in range(max(0, max_measurements - len(history))):
        error_nm = best.wavelength_nm - target_nm
        if abs(error_nm) * 1000.0 <= wavelength_tolerance_pm and mode_ok(best):
            break
        requested = code_to_current(best_codes[2], 2) - error_nm / slope_nm_per_ma
        requested = min(max(requested, last_phase - 0.35), last_phase + 0.35)
        requested = min(max(requested, 0.0), LASER_MAX_MA[2])
        if abs(requested - code_to_current(best_codes[2], 2)) < 0.004:
            requested += 0.025 if error_nm * slope_nm_per_ma > 0.0 else -0.025
            requested = min(max(requested, 0.0), LASER_MAX_MA[2])
        trial_codes = list(best_codes)
        trial_codes[2] = current_to_code(requested, 2)
        _, trial = probe(laser, meter, trial_codes, target_nm, settle_s)
        history.append({"codes": trial_codes, "reading": asdict(trial),
                        "action": f"phase_refine_{iteration}"})

        # Update derivative only for a same-branch movement.
        delta_phase = requested - last_phase
        delta_wave = trial.wavelength_nm - last_wave
        if abs(delta_phase) >= 0.01 and abs(delta_wave) <= 0.35:
            measured = delta_wave / delta_phase
            if 0.005 <= abs(measured) <= 1.0:
                slope_nm_per_ma = 0.65 * slope_nm_per_ma + 0.35 * measured
        last_phase, last_wave = requested, trial.wavelength_nm
        if reading_score(trial, target_nm) < reading_score(best, target_nm):
            best_codes, best = trial_codes, trial
        elif abs(delta_wave) > 0.35:
            break

    return {
        "success": abs(best.wavelength_nm - target_nm) * 1000.0 <= wavelength_tolerance_pm
                   and mode_ok(best),
        "target_nm": target_nm,
        "codes": best_codes,
        "currents_ma": [code_to_current(v, i) for i, v in enumerate(best_codes)],
        "reading": asdict(best),
        "phase_slope_nm_per_ma": slope_nm_per_ma,
        "history": history,
    }


def tune_local_three_controls(
    laser: LaserSerial,
    meter: AQ6150B,
    seed_codes: list[int],
    target_nm: float,
    settle_s: float,
    max_measurements: int = 10,
    wavelength_tolerance_pm: float = WAVELENGTH_TOLERANCE_PM,
) -> dict[str, Any]:
    """Close wavelength with a measured local PHASE/WAVE-A/WAVE-B Jacobian.

    Small absolute probes estimate the local wavelength gradient.  A
    least-norm three-control step then moves the existing longitudinal mode;
    any dual-mode reading or large discontinuity is rejected as a mode hop.
    """
    # PHASE has a strongly asymmetric sawtooth near longitudinal-mode
    # boundaries: a positive finite difference can look valid while the
    # required negative move produces almost no wavelength change.  Use the
    # locally linear mirror currents for final closure and leave PHASE fixed.
    probe_steps_ma = {3: 0.05, 4: 0.05}
    max_steps_ma = {3: 0.40, 4: 0.40}
    mobility_weights = {3: 1.0, 4: 1.0}
    history: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()

    codes, reading = probe(laser, meter, list(seed_codes), target_nm, settle_s)
    history.append({"codes": codes, "reading": asdict(reading), "action": "seed"})
    seen.add(tuple(codes))
    best_codes, best = list(codes), reading
    gradients: dict[int, float] = {}

    while len(history) < max_measurements:
        error_pm = (best.wavelength_nm - target_nm) * 1000.0
        if abs(error_pm) <= wavelength_tolerance_pm and mode_ok(best):
            break
        base_codes = list(best_codes)
        base_reading = best
        improved_probe: tuple[list[int], Reading] | None = None

        # Re-estimate all three local derivatives at the current best point.
        for control_index in (3, 4):
            if len(history) >= max_measurements:
                break
            base_ma = code_to_current(base_codes[control_index], control_index)
            maximum = LASER_MAX_MA[control_index]
            direction = 1.0 if base_ma + probe_steps_ma[control_index] <= maximum else -1.0
            requested = min(max(base_ma + direction * probe_steps_ma[control_index], 0.0),
                            maximum)
            trial_codes = list(base_codes)
            trial_codes[control_index] = current_to_code(requested, control_index)
            if tuple(trial_codes) in seen:
                continue
            seen.add(tuple(trial_codes))
            _, trial = probe(laser, meter, trial_codes, target_nm, settle_s)
            history.append({
                "codes": trial_codes, "reading": asdict(trial),
                "action": f"jacobian_probe_{CONTROL_NAMES[control_index]}",
            })
            delta_ma = code_to_current(trial_codes[control_index], control_index) - base_ma
            delta_nm = trial.wavelength_nm - base_reading.wavelength_nm
            same_mode = mode_ok(trial) and abs(delta_nm) <= 0.12
            if same_mode and abs(delta_ma) >= 0.005:
                slope = delta_nm / delta_ma
                if 0.002 <= abs(slope) <= 2.0:
                    gradients[control_index] = slope
            if same_mode and reading_score(trial, target_nm) < reading_score(best, target_nm):
                if improved_probe is None or reading_score(trial, target_nm) < reading_score(
                    improved_probe[1], target_nm
                ):
                    improved_probe = (list(trial_codes), trial)
            if abs(trial.wavelength_nm - target_nm) * 1000.0 <= wavelength_tolerance_pm \
                    and mode_ok(trial):
                best_codes, best = list(trial_codes), trial
                break
        if abs(best.wavelength_nm - target_nm) * 1000.0 <= wavelength_tolerance_pm \
                and mode_ok(best):
            break

        if not gradients:
            if improved_probe is not None:
                best_codes, best = improved_probe
                continue
            break

        # Least-norm correction: delta_i = error*g_i / sum(g_i^2).
        desired_nm = target_nm - base_reading.wavelength_nm
        denominator = sum(
            value * value * mobility_weights[index]
            for index, value in gradients.items()
        )
        trial_codes = list(base_codes)
        for control_index, slope in gradients.items():
            delta_ma = desired_nm * slope * mobility_weights[control_index] / denominator
            limit = max_steps_ma[control_index]
            delta_ma = min(max(delta_ma, -limit), limit)
            current_ma = code_to_current(base_codes[control_index], control_index)
            requested = min(max(current_ma + delta_ma, 0.0), LASER_MAX_MA[control_index])
            trial_codes[control_index] = current_to_code(requested, control_index)

        if tuple(trial_codes) in seen or len(history) >= max_measurements:
            if improved_probe is not None:
                best_codes, best = improved_probe
                continue
            break
        seen.add(tuple(trial_codes))
        _, trial = probe(laser, meter, trial_codes, target_nm, settle_s)
        history.append({"codes": trial_codes, "reading": asdict(trial),
                        "action": "jacobian_correction"})
        same_mode = mode_ok(trial) and abs(
            trial.wavelength_nm - base_reading.wavelength_nm
        ) <= 0.20
        if same_mode and reading_score(trial, target_nm) < reading_score(best, target_nm):
            best_codes, best = list(trial_codes), trial
        elif improved_probe is not None:
            best_codes, best = improved_probe
        else:
            break

    return {
        "success": abs(best.wavelength_nm - target_nm) * 1000.0
                   <= wavelength_tolerance_pm and mode_ok(best),
        "target_nm": target_nm,
        "codes": best_codes,
        "currents_ma": [code_to_current(value, index)
                        for index, value in enumerate(best_codes)],
        "reading": asdict(best),
        "local_gradients_nm_per_ma": {
            CONTROL_NAMES[index]: value for index, value in gradients.items()
        },
        "history": history,
    }


def tune_wavelength_with_phase_fallback(
    laser: LaserSerial,
    meter: AQ6150B,
    seed_codes: list[int],
    target_nm: float,
    settle_s: float,
    max_measurements: int,
    wavelength_tolerance_pm: float = WAVELENGTH_TOLERANCE_PM,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Close wavelength with mirrors first, then PHASE, sharing one budget."""
    total_budget = max(6, int(max_measurements))
    mirror_budget = max(4, total_budget // 2)
    mirror = tune_local_three_controls(
        laser, meter, list(seed_codes), target_nm, settle_s,
        max_measurements=mirror_budget,
        wavelength_tolerance_pm=wavelength_tolerance_pm,
    )
    attempts: list[dict[str, Any]] = [{
        "method": "mirror_jacobian", "success": mirror["success"],
        "codes": mirror["codes"], "reading": mirror["reading"],
        "measurements": mirror["history"],
    }]
    if mirror["success"]:
        return mirror, attempts

    best = mirror
    phase_seeds: list[tuple[str, list[int]]] = [
        ("mirror_best", [int(value) for value in mirror["codes"]]),
        ("original_seed", [int(value) for value in seed_codes]),
    ]
    seen: set[tuple[int, ...]] = set()
    remaining_budget = max(0, total_budget - len(mirror["history"]))
    for seed_index, (source, phase_seed) in enumerate(phase_seeds):
        identity = tuple(phase_seed)
        if identity in seen:
            continue
        seeds_left = len(phase_seeds) - seed_index
        phase_budget = remaining_budget // seeds_left
        if phase_budget < 2:
            break
        seen.add(identity)
        phase = tune_phase_on_branch(
            laser, meter, phase_seed, target_nm, settle_s,
            max_measurements=phase_budget,
            wavelength_tolerance_pm=wavelength_tolerance_pm,
        )
        remaining_budget = max(0, remaining_budget - len(phase["history"]))
        attempts.append({
            "method": f"phase_from_{source}", "success": phase["success"],
            "codes": phase["codes"], "reading": phase["reading"],
            "measurements": phase["history"],
        })
        phase_reading = Reading(**phase["reading"])
        best_reading = Reading(**best["reading"])
        if phase["success"] or reading_score(
            phase_reading, target_nm
        ) < reading_score(best_reading, target_nm):
            best = phase
        if phase["success"]:
            break
    return best, attempts


def compact_row(result: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        "success": bool(result["success"]),
        "target_nm": float(result["target_nm"]),
        "codes": [int(value) for value in result["codes"]],
        "currents_ma": [float(value) for value in result["currents_ma"]],
        "reading": dict(result["reading"]),
        "phase_slope_nm_per_ma": float(
            result.get("phase_slope_nm_per_ma", -0.075)
        ),
        "source": source,
    }


def fast_follow_branch(
    laser: LaserSerial,
    meter: AQ6150B,
    previous: dict[str, Any],
    target_nm: float,
    settle_s: float,
    max_measurements: int = 3,
) -> dict[str, Any] | None:
    """Predict the next dense point from a same-branch measured derivative."""
    previous_codes = [int(value) for value in previous["codes"]]
    previous_reading = Reading(**previous["reading"])
    previous_phase = code_to_current(previous_codes[2], 2)
    slope = float(previous.get("phase_slope_nm_per_ma", -0.075))
    if not math.isfinite(slope) or not (0.005 <= abs(slope) <= 1.0):
        slope = -0.075
    history: list[dict[str, Any]] = []
    best_codes = list(previous_codes)
    best_reading = previous_reading

    for iteration in range(max_measurements):
        error_nm = best_reading.wavelength_nm - target_nm
        requested_phase = code_to_current(best_codes[2], 2) - error_nm / slope
        requested_phase = min(
            max(requested_phase,
                code_to_current(best_codes[2], 2) - 0.55),
            code_to_current(best_codes[2], 2) + 0.55,
        )
        if not 0.0 <= requested_phase <= LASER_MAX_MA[2]:
            return None
        trial_codes = list(best_codes)
        trial_codes[2] = current_to_code(requested_phase, 2)
        if trial_codes == best_codes:
            return None
        _, trial = probe(laser, meter, trial_codes, target_nm, settle_s)
        history.append({"codes": trial_codes, "reading": asdict(trial),
                        "action": f"dense_follow_{iteration}"})
        delta_phase = requested_phase - previous_phase
        delta_wave = trial.wavelength_nm - previous_reading.wavelength_nm
        # A large wavelength discontinuity is a cavity-mode hop, not a slope.
        if abs(delta_wave) > 0.30 or not mode_ok(trial):
            return None
        if abs(delta_phase) >= 0.01:
            measured_slope = delta_wave / delta_phase
            if 0.005 <= abs(measured_slope) <= 1.0:
                slope = 0.35 * slope + 0.65 * measured_slope
        if reading_score(trial, target_nm) < reading_score(best_reading, target_nm):
            best_codes, best_reading = trial_codes, trial
        if abs(trial.wavelength_nm - target_nm) * 1000.0 <= WAVELENGTH_TOLERANCE_PM:
            return {
                "success": True, "target_nm": target_nm,
                "codes": trial_codes,
                "currents_ma": [code_to_current(v, i) for i, v in enumerate(trial_codes)],
                "reading": asdict(trial),
                "phase_slope_nm_per_ma": slope,
                "history": history,
            }
    return None


def tune_target(
    laser: LaserSerial,
    meter: AQ6150B,
    library: list[Candidate],
    target_nm: float,
    settle_s: float,
    previous_codes: list[int] | None = None,
    model_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    seeds: list[list[int]] = []
    if previous_codes is not None:
        seeds.append(at_discovery_power(previous_codes))
    seeds.extend(at_discovery_power(item.codes) for item in seed_candidates(library, target_nm, 6))
    if model_bundle is not None:
        seeds.extend(model_seed_candidates(model_bundle, library, target_nm, count=6))

    attempts: list[dict[str, Any]] = []
    screening: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for seed in seeds:
        key = tuple(seed)
        if key in seen:
            continue
        seen.add(key)
        _, reading = probe(laser, meter, seed, target_nm, settle_s)
        screening.append({
            "codes": list(seed),
            "currents_ma": [code_to_current(v, i) for i, v in enumerate(seed)],
            "reading": asdict(reading),
        })

    if not screening:
        raise RuntimeError(f"No valid branch candidates for {target_nm:.6f} nm")
    screening.sort(key=lambda row: reading_score(Reading(**row["reading"]), target_nm))

    # Retain a direct hit, but still refine several nearby branches: equal-power
    # calibration needs the highest-power valid branch, not merely the first
    # branch that reaches the wavelength tolerance.
    direct = screening[0]
    direct_reading = Reading(**direct["reading"])
    successful: list[dict[str, Any]] = []
    if abs(direct_reading.wavelength_nm - target_nm) * 1000.0 <= WAVELENGTH_TOLERANCE_PM \
            and mode_ok(direct_reading):
        best: dict[str, Any] = {
            "success": True, "target_nm": target_nm,
            "codes": direct["codes"], "currents_ma": direct["currents_ma"],
            "reading": direct["reading"], "history": [],
        }
        successful.append(best)
    else:
        best = {
            "success": False, "target_nm": target_nm,
            "codes": direct["codes"], "currents_ma": direct["currents_ma"],
            "reading": direct["reading"], "history": [],
        }
    rows_to_tune = screening[1:3] if successful else screening[:3]
    for row in rows_to_tune:
        attempt = tune_phase_on_branch(
            laser, meter, list(row["codes"]), target_nm, settle_s,
            max_measurements=20,
        )
        attempts.append(attempt)
        if attempt["success"]:
            successful.append(attempt)
        if reading_score(Reading(**attempt["reading"]), target_nm) < reading_score(
            Reading(**best["reading"]), target_nm
        ):
            best = attempt
    if successful:
        # All rows here already meet wavelength and single-mode constraints.
        # Prefer optical power; SMSR breaks a near-power tie.
        best = max(successful, key=lambda row: (
            float(row["reading"]["power_mw"]),
            normalized_smsr(row["reading"]),
        ))
    assert best is not None
    # ``best`` is one member of ``attempts``.  Mutating it in place to contain
    # the whole list would make a self-reference that JSON cannot checkpoint.
    result = dict(best)
    result["screening"] = screening
    result["attempts"] = attempts
    return result


def pilot_targets(start_nm: float, stop_nm: float, step_nm: float) -> list[float]:
    return target_grid(start_nm, stop_nm, step_nm)


def run_mode_map(args: argparse.Namespace) -> dict[str, Any]:
    """Measure a fixed-PHASE WAVE-A/WAVE-B Vernier map."""
    values_a = np.arange(args.wave_a_start_ma,
                         args.wave_a_stop_ma + args.wave_step_ma * 0.25,
                         args.wave_step_ma, dtype=float).tolist()
    values_b = np.arange(args.wave_b_start_ma,
                         args.wave_b_stop_ma + args.wave_step_ma * 0.25,
                         args.wave_step_ma, dtype=float).tolist()
    if values_a[-1] < args.wave_a_stop_ma - 0.001:
        values_a.append(args.wave_a_stop_ma)
    if values_b[-1] < args.wave_b_stop_ma - 0.001:
        values_b.append(args.wave_b_stop_ma)
    payload: dict[str, Any]
    if args.resume and args.output.exists():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
    else:
        payload = {
            "created": utc_now(),
            "method": "boustrophedon_wave_a_wave_b_mode_map_at_fixed_phase",
            "limits_ma": dict(zip(CONTROL_NAMES, LASER_MAX_MA)),
            "gain_ma": args.gain_ma,
            "soa_ma": args.soa_ma,
            "phase_ma": args.phase_ma,
            "wave_step_ma": args.wave_step_ma,
            "wave_a_range_ma": [args.wave_a_start_ma, args.wave_a_stop_ma],
            "wave_b_range_ma": [args.wave_b_start_ma, args.wave_b_stop_ma],
            "rows": {},
        }

    points: list[tuple[float, float]] = []
    for ai, wave_a in enumerate(values_a):
        row_b = values_b if ai % 2 == 0 else list(reversed(values_b))
        points.extend((wave_a, wave_b) for wave_b in row_b)

    laser = open_fullband_laser(args.port)
    meter = AQ6150B(args.gpib, average_count=AQ_AVERAGE_COUNT)
    payload["meter"] = meter.identity
    payload["meter_profile"] = meter.measurement_profile
    last_codes: list[int] | None = None
    try:
        for index, (wave_a, wave_b) in enumerate(points):
            key = f"A{wave_a:.4f}_B{wave_b:.4f}_P{args.phase_ma:.4f}"
            if key in payload["rows"]:
                last_codes = list(payload["rows"][key]["codes"])
                continue
            codes = [
                current_to_code(args.gain_ma, 0),
                current_to_code(args.soa_ma, 1),
                current_to_code(args.phase_ma, 2),
                current_to_code(wave_a, 3),
                current_to_code(wave_b, 4),
            ]
            _, reading = probe(laser, meter, codes, 1545.0, args.settle_s)
            payload["rows"][key] = {
                "codes": codes,
                "currents_ma": [code_to_current(v, i) for i, v in enumerate(codes)],
                "reading": asdict(reading),
                "single_mode": mode_ok(reading),
            }
            last_codes = codes
            save_json(args.output, payload)
            if (index + 1) % args.print_every == 0 or index == 0:
                print(
                    f"map {index + 1}/{len(points)} A={wave_a:.3f} B={wave_b:.3f} "
                    f"wave={reading.wavelength_nm:.6f} power={reading.power_mw:.4f} "
                    f"SMSR={normalized_smsr(asdict(reading)):.1f}", flush=True,
                )
    finally:
        if last_codes is not None:
            park = list(last_codes)
            park[0] = current_to_code(60.0, 0)
            park[1] = current_to_code(90.0, 1)
            try:
                laser.set_codes(park)
            except Exception:
                pass
        meter.close()
        laser.close()

    measured = list(payload["rows"].values())
    clean = [row for row in measured if row.get("single_mode")]
    payload["summary"] = {
        "planned": len(points), "measured": len(measured),
        "single_mode": len(clean),
        "wavelength_min_nm": min(
            (float(row["reading"]["wavelength_nm"]) for row in clean), default=None),
        "wavelength_max_nm": max(
            (float(row["reading"]["wavelength_nm"]) for row in clean), default=None),
        "power_min_mw": min(
            (float(row["reading"]["power_mw"]) for row in clean), default=None),
        "power_max_mw": max(
            (float(row["reading"]["power_mw"]) for row in clean), default=None),
    }
    save_json(args.output, payload)
    return payload


def select_phase_sweep_branches(args: argparse.Namespace) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    mode_maps = getattr(args, "mode_maps", None)
    paths = list(mode_maps) if mode_maps else sorted(HERE.glob("fullband_mode_map*.json"))
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for row in payload.get("rows", {}).values():
            reading = row.get("reading", {})
            codes = valid_codes(row.get("codes", ()))
            if codes is None or not row.get("single_mode", normalized_smsr(reading) >= MIN_SMSR_DB):
                continue
            try:
                wavelength = float(reading["wavelength_nm"])
                power = float(reading["power_mw"])
            except (KeyError, TypeError, ValueError):
                continue
            if args.start_nm - args.radius_nm <= wavelength <= args.stop_nm + args.radius_nm:
                observations.append({
                    "codes": list(codes),
                    "currents_ma": [code_to_current(v, i) for i, v in enumerate(codes)],
                    "wavelength_nm": wavelength,
                    "power_mw": power,
                    "smsr_db": normalized_smsr(reading),
                    "source": path.name,
                })
    if not observations:
        raise RuntimeError("No single-mode fixed-PHASE mode-map observations found")

    selected: dict[tuple[int, int], dict[str, Any]] = {}
    targets = target_grid(args.start_nm, args.stop_nm, args.target_step_nm)
    uncovered: list[float] = []
    for target in targets:
        ranked = sorted(
            observations,
            key=lambda row: (
                abs(float(row["wavelength_nm"]) - target),
                -float(row["power_mw"]),
                -float(row["smsr_db"]),
            ),
        )
        nearby = [row for row in ranked
                  if abs(float(row["wavelength_nm"]) - target) <= args.radius_nm]
        pool = nearby if nearby else ranked[:1]
        if not nearby:
            uncovered.append(target)
        kept = 0
        for row in pool:
            key = (int(row["codes"][3]), int(row["codes"][4]))
            if key in selected:
                continue
            selected[key] = row
            kept += 1
            if kept >= args.branches_per_target:
                break

    branches = sorted(selected.values(), key=lambda row: float(row["wavelength_nm"]))
    payload = {
        "created": utc_now(),
        "method": "high_power_multi_branch_selection_from_live_mode_maps",
        "target_start_nm": args.start_nm, "target_stop_nm": args.stop_nm,
        "target_step_nm": args.target_step_nm, "selection_radius_nm": args.radius_nm,
        "branches_per_target": args.branches_per_target,
        "old_total_spreadsheet_used": False,
        "summary": {
            "fixed_phase_observations": len(observations),
            "selected_unique_wave_ab_branches": len(branches),
            "targets_without_candidate_inside_radius": len(uncovered),
            "largest_nearest_fixed_phase_distance_nm": max(
                min(abs(float(row["wavelength_nm"]) - target) for row in observations)
                for target in targets
            ),
        },
        "uncovered_targets_nm": uncovered,
        "branches": branches,
    }
    save_json(args.output, payload)
    return payload


def select_gap_phase_branches(args: argparse.Namespace) -> dict[str, Any]:
    """Select WAVE-A/B branches nearest only the still-uncovered targets.

    The resulting branch list is intended for an interleaved PHASE sweep
    (for example 0.5, 1.5, ..., 9.5 mA). It uses measured single-mode rows
    and never bridges a mode hop offline.
    """
    candidate_payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    targets = [float(value) for value in candidate_payload.get("uncovered_targets_nm", [])]
    if not targets:
        raise RuntimeError("Candidate file has no uncovered targets")

    phase_payload = json.loads(args.phase_map.read_text(encoding="utf-8"))
    observations: list[dict[str, Any]] = []
    for row in phase_payload.get("rows", {}).values():
        reading = row.get("reading", {})
        codes = valid_codes(row.get("codes", ()))
        if codes is None:
            continue
        if not bool(row.get("single_mode")) or normalized_smsr(reading) < MIN_SMSR_DB:
            continue
        try:
            wavelength = float(reading["wavelength_nm"])
            power = float(reading["power_mw"])
        except (KeyError, TypeError, ValueError):
            continue
        observations.append({
            "codes": list(codes),
            "currents_ma": [code_to_current(value, index)
                            for index, value in enumerate(codes)],
            "wavelength_nm": wavelength,
            "power_mw": power,
            "smsr_db": normalized_smsr(reading),
            "source": args.phase_map.name,
        })
    if not observations:
        raise RuntimeError("No clean measured rows found in the PHASE map")

    selected: dict[tuple[int, int], dict[str, Any]] = {}
    no_nearby: list[float] = []
    for target in targets:
        ranked = sorted(
            observations,
            key=lambda row: (
                abs(float(row["wavelength_nm"]) - target),
                -float(row["power_mw"]),
                -float(row["smsr_db"]),
            ),
        )
        nearby = [row for row in ranked
                  if abs(float(row["wavelength_nm"]) - target) <= args.radius_nm]
        if not nearby:
            no_nearby.append(target)
            nearby = ranked
        kept = 0
        seen_for_target: set[tuple[int, int]] = set()
        for row in nearby:
            identity = (int(row["codes"][3]), int(row["codes"][4]))
            if identity in seen_for_target:
                continue
            seen_for_target.add(identity)
            selected.setdefault(identity, row)
            kept += 1
            if kept >= args.branches_per_target:
                break

    branches = sorted(selected.values(), key=lambda row: float(row["wavelength_nm"]))
    payload = {
        "created": utc_now(),
        "method": "targeted_interleaved_phase_refinement_near_uncovered_targets",
        "source_candidates": args.candidates.name,
        "source_phase_map": args.phase_map.name,
        "selection_radius_nm": args.radius_nm,
        "branches_per_target": args.branches_per_target,
        "uncovered_targets_nm": targets,
        "targets_without_nearby_measured_branch": no_nearby,
        "summary": {
            "uncovered_target_count": len(targets),
            "clean_phase_observations": len(observations),
            "selected_unique_wave_ab_branches": len(branches),
            "targets_without_candidate_inside_radius": len(no_nearby),
        },
        "branches": branches,
    }
    save_json(args.output, payload)
    return payload


def select_unmeasured_mode_branches(args: argparse.Namespace) -> dict[str, Any]:
    """Select alternate WAVE-A/B branches not yet present in PHASE maps."""
    candidate_payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    targets = [float(value) for value in candidate_payload.get("uncovered_targets_nm", [])]
    if not targets:
        raise RuntimeError("Candidate file has no uncovered targets")

    measured_identities: set[tuple[int, int]] = set()
    for path in args.exclude_phase_maps:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("rows", {}).values():
            codes = valid_codes(row.get("codes", ()))
            if codes is not None:
                measured_identities.add((int(codes[3]), int(codes[4])))

    observations: dict[tuple[int, int], dict[str, Any]] = {}
    mode_maps = getattr(args, "mode_maps", None)
    paths = list(mode_maps) if mode_maps else sorted(HERE.glob("fullband_mode_map*.json"))
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for row in payload.get("rows", {}).values():
            reading = row.get("reading", {})
            codes = valid_codes(row.get("codes", ()))
            if codes is None:
                continue
            identity = (int(codes[3]), int(codes[4]))
            if identity in measured_identities:
                continue
            if not bool(row.get("single_mode", normalized_smsr(reading) >= MIN_SMSR_DB)):
                continue
            if normalized_smsr(reading) < MIN_SMSR_DB:
                continue
            try:
                wavelength = float(reading["wavelength_nm"])
                power = float(reading["power_mw"])
            except (KeyError, TypeError, ValueError):
                continue
            candidate = {
                "codes": list(codes),
                "currents_ma": [code_to_current(value, index)
                                for index, value in enumerate(codes)],
                "wavelength_nm": wavelength,
                "power_mw": power,
                "smsr_db": normalized_smsr(reading),
                "source": path.name,
            }
            old = observations.get(identity)
            if old is None or power > float(old["power_mw"]):
                observations[identity] = candidate
    if not observations:
        raise RuntimeError("No unmeasured single-mode WAVE-A/B branches found")

    selected: dict[tuple[int, int], dict[str, Any]] = {}
    nearest_distances: list[float] = []
    no_nearby: list[float] = []
    for target in targets:
        ranked = sorted(
            observations.items(),
            key=lambda item: (
                abs(float(item[1]["wavelength_nm"]) - target),
                -float(item[1]["power_mw"]),
                -float(item[1]["smsr_db"]),
            ),
        )
        nearest_distances.append(abs(float(ranked[0][1]["wavelength_nm"]) - target))
        nearby = [item for item in ranked
                  if abs(float(item[1]["wavelength_nm"]) - target) <= args.radius_nm]
        if not nearby:
            no_nearby.append(target)
            nearby = ranked
        for identity, row in nearby[:args.branches_per_target]:
            selected.setdefault(identity, row)

    branches = sorted(selected.values(), key=lambda row: float(row["wavelength_nm"]))
    payload = {
        "created": utc_now(),
        "method": "alternate_unmeasured_wave_ab_selection_from_live_fixed_phase_maps",
        "source_candidates": args.candidates.name,
        "excluded_phase_maps": [path.name for path in args.exclude_phase_maps],
        "selection_radius_nm": args.radius_nm,
        "branches_per_target": args.branches_per_target,
        "uncovered_targets_nm": targets,
        "targets_without_nearby_measured_branch": no_nearby,
        "summary": {
            "uncovered_target_count": len(targets),
            "unmeasured_clean_wave_ab_branches": len(observations),
            "selected_unique_wave_ab_branches": len(branches),
            "targets_without_candidate_inside_radius": len(no_nearby),
            "largest_nearest_fixed_phase_distance_nm": max(nearest_distances),
        },
        "branches": branches,
    }
    save_json(args.output, payload)
    return payload


def run_phase_map(args: argparse.Namespace) -> dict[str, Any]:
    selection = json.loads(args.branches.read_text(encoding="utf-8"))
    branches = selection.get("branches", [])
    phases = np.arange(args.phase_start_ma,
                       args.phase_stop_ma + args.phase_step_ma * 0.25,
                       args.phase_step_ma, dtype=float).tolist()
    if phases[-1] < args.phase_stop_ma - 0.001:
        phases.append(args.phase_stop_ma)
    if args.resume and args.output.exists():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
    else:
        payload = {
            "created": utc_now(),
            "method": "full_phase_sweep_of_selected_high_power_wave_ab_branches",
            "limits_ma": dict(zip(CONTROL_NAMES, LASER_MAX_MA)),
            "gain_ma": args.gain_ma, "soa_ma": args.soa_ma,
            "phase_start_ma": args.phase_start_ma,
            "phase_stop_ma": args.phase_stop_ma,
            "phase_step_ma": args.phase_step_ma,
            "branch_count": len(branches),
            "rows": {},
        }
    total = len(branches) * len(phases)
    laser = open_fullband_laser(args.port)
    meter = AQ6150B(args.gpib, average_count=AQ_AVERAGE_COUNT)
    payload["meter"] = meter.identity
    last_codes: list[int] | None = None
    measured_index = 0
    try:
        for branch_index, branch in enumerate(branches):
            branch_phases = phases if branch_index % 2 == 0 else list(reversed(phases))
            for phase_ma in branch_phases:
                measured_index += 1
                key = f"B{branch_index:04d}_P{phase_ma:.4f}"
                if key in payload["rows"]:
                    last_codes = list(payload["rows"][key]["codes"])
                    continue
                branch_codes = valid_codes(branch["codes"])
                if branch_codes is None:
                    continue
                codes = [
                    current_to_code(args.gain_ma, 0),
                    current_to_code(args.soa_ma, 1),
                    current_to_code(phase_ma, 2),
                    int(branch_codes[3]), int(branch_codes[4]),
                ]
                _, reading = probe(laser, meter, codes, 1545.0, args.settle_s)
                row = {
                    "branch_index": branch_index,
                    "codes": codes,
                    "currents_ma": [code_to_current(v, i) for i, v in enumerate(codes)],
                    "reading": asdict(reading),
                    "single_mode": mode_ok(reading),
                    "selection_source": branch.get("source"),
                }
                payload["rows"][key] = row
                last_codes = codes
                if measured_index % args.checkpoint_every == 0:
                    save_json(args.output, payload)
                if measured_index % args.print_every == 0 or measured_index == 1:
                    print(
                        f"phase-map {measured_index}/{total} branch={branch_index + 1}/{len(branches)} "
                        f"phase={phase_ma:.3f} wave={reading.wavelength_nm:.6f} "
                        f"power={reading.power_mw:.4f} SMSR={normalized_smsr(asdict(reading)):.1f}",
                        flush=True,
                    )
    finally:
        if last_codes is not None:
            park = list(last_codes)
            park[0] = current_to_code(60.0, 0)
            park[1] = current_to_code(90.0, 1)
            try:
                laser.set_codes(park)
            except Exception:
                pass
        meter.close()
        laser.close()
    rows = list(payload["rows"].values())
    clean = [row for row in rows if row.get("single_mode")]
    payload["summary"] = {
        "planned": total, "measured": len(rows), "single_mode": len(clean),
        "wavelength_min_nm": min(
            (float(row["reading"]["wavelength_nm"]) for row in clean), default=None),
        "wavelength_max_nm": max(
            (float(row["reading"]["wavelength_nm"]) for row in clean), default=None),
        "power_min_mw": min(
            (float(row["reading"]["power_mw"]) for row in clean), default=None),
        "power_max_mw": max(
            (float(row["reading"]["power_mw"]) for row in clean), default=None),
    }
    save_json(args.output, payload)
    return payload


def build_phase_interpolated_candidates(args: argparse.Namespace) -> dict[str, Any]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for phase_map_path in args.phase_maps:
        phase_map = json.loads(phase_map_path.read_text(encoding="utf-8"))
        for row in phase_map.get("rows", {}).values():
            codes = valid_codes(row.get("codes", ()))
            if codes is None:
                continue
            grouped.setdefault((int(codes[3]), int(codes[4])), []).append(row)

    segments: list[dict[str, Any]] = []
    for branch_identity, rows in grouped.items():
        rows.sort(key=lambda row: float(row["currents_ma"][2]))
        current: list[dict[str, Any]] = []
        for row in rows:
            clean = bool(row.get("single_mode")) and normalized_smsr(row["reading"]) >= MIN_SMSR_DB
            if not clean:
                if len(current) >= 2:
                    segments.append({"branch_identity": branch_identity, "rows": current})
                current = []
                continue
            if current:
                previous_wave = float(current[-1]["reading"]["wavelength_nm"])
                this_wave = float(row["reading"]["wavelength_nm"])
                if abs(this_wave - previous_wave) > args.jump_threshold_nm:
                    if len(current) >= 2:
                        segments.append({"branch_identity": branch_identity, "rows": current})
                    current = []
            current.append(row)
        if len(current) >= 2:
            segments.append({"branch_identity": branch_identity, "rows": current})

    targets = target_grid(args.start_nm, args.stop_nm, args.step_nm)
    output_rows: dict[str, Any] = {}
    uncovered: list[float] = []
    for target in targets:
        candidates: list[dict[str, Any]] = []
        for segment_index, segment in enumerate(segments):
            rows = segment["rows"]
            for left, right in zip(rows, rows[1:]):
                wave0 = float(left["reading"]["wavelength_nm"])
                wave1 = float(right["reading"]["wavelength_nm"])
                if abs(wave1 - wave0) < 1e-7:
                    continue
                if not (min(wave0, wave1) <= target <= max(wave0, wave1)):
                    continue
                fraction = (target - wave0) / (wave1 - wave0)
                phase0 = float(left["currents_ma"][2])
                phase1 = float(right["currents_ma"][2])
                phase = phase0 + fraction * (phase1 - phase0)
                power0 = float(left["reading"]["power_mw"])
                power1 = float(right["reading"]["power_mw"])
                predicted_power = power0 + fraction * (power1 - power0)
                codes = list(left["codes"])
                codes[0] = current_to_code(MAX_GAIN_SOA_MA, 0)
                codes[1] = current_to_code(MAX_GAIN_SOA_MA, 1)
                codes[2] = current_to_code(phase, 2)
                candidates.append({
                    "codes": codes,
                    "currents_ma": [code_to_current(v, i) for i, v in enumerate(codes)],
                    "branch_identity": list(segment["branch_identity"]),
                    "segment_index": segment_index,
                    "left_wavelength_nm": wave0,
                    "right_wavelength_nm": wave1,
                    "predicted_wavelength_nm": target,
                    "predicted_power_mw": predicted_power,
                    "endpoint_min_smsr_db": min(
                        normalized_smsr(left["reading"]), normalized_smsr(right["reading"])
                    ),
                    "phase_slope_nm_per_ma": (wave1 - wave0) / (phase1 - phase0),
                })
        # One branch can cross a target more than once after a small local
        # reversal.  Retain only its highest-power crossing.
        by_branch: dict[tuple[int, int], dict[str, Any]] = {}
        for candidate in candidates:
            key = tuple(int(value) for value in candidate["branch_identity"])
            old = by_branch.get(key)
            if old is None or candidate["predicted_power_mw"] > old["predicted_power_mw"]:
                by_branch[key] = candidate
        ranked = sorted(
            by_branch.values(),
            key=lambda row: (-float(row["predicted_power_mw"]),
                             -float(row["endpoint_min_smsr_db"])),
        )[:args.candidates_per_target]
        key = f"{target:.6f}"
        output_rows[key] = {"target_nm": target, "candidates": ranked}
        if not ranked:
            uncovered.append(target)

    covered = [row for row in output_rows.values() if row["candidates"]]
    best_powers = [float(row["candidates"][0]["predicted_power_mw"]) for row in covered]
    payload = {
        "created": utc_now(),
        "method": "piecewise_linear_interpolation_inside_measured_mode_hop_free_phase_segments",
        "phase_maps": [path.name for path in args.phase_maps],
        "jump_threshold_nm": args.jump_threshold_nm,
        "single_mode_gate_smsr_db": MIN_SMSR_DB,
        "target_start_nm": args.start_nm, "target_stop_nm": args.stop_nm,
        "target_step_nm": args.step_nm,
        "summary": {
            "measured_branches": len(grouped),
            "mode_hop_free_segments": len(segments),
            "target_count": len(targets),
            "covered_targets": len(covered),
            "uncovered_targets": len(uncovered),
            "predicted_best_power_min_mw": min(best_powers, default=None),
            "predicted_best_power_median_mw": statistics.median(best_powers) if best_powers else None,
            "predicted_best_power_max_mw": max(best_powers, default=None),
        },
        "uncovered_targets_nm": uncovered,
        "rows": output_rows,
    }
    save_json(args.output, payload)
    return payload


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    library = load_candidates(args.inventory)
    model_bundle = joblib.load(args.model) if args.model and args.model.exists() else None
    targets = pilot_targets(args.start_nm, args.stop_nm, args.step_nm)
    payload: dict[str, Any]
    if args.resume and args.output.exists():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
    else:
        payload = {
            "created": utc_now(),
            "method": "measured_branch_seed_plus_live_local_phase_model",
            "target_start_nm": args.start_nm,
            "target_stop_nm": args.stop_nm,
            "target_step_nm": args.step_nm,
            "limits_ma": dict(zip(CONTROL_NAMES, LASER_MAX_MA)),
            "single_mode_gate_smsr_db": MIN_SMSR_DB,
            "wavelength_tolerance_pm": WAVELENGTH_TOLERANCE_PM,
            "rows": {},
        }

    laser = open_fullband_laser(args.port)
    meter = AQ6150B(args.gpib, average_count=AQ_AVERAGE_COUNT)
    payload["meter"] = meter.identity
    previous_codes: list[int] | None = None
    try:
        for index, target in enumerate(targets):
            key = f"{target:.6f}"
            if key in payload["rows"] and payload["rows"][key].get("success"):
                previous_codes = list(payload["rows"][key]["codes"])
                rd = payload["rows"][key]["reading"]
                library.append(Candidate(
                    codes=tuple(previous_codes), wavelength_nm=float(rd["wavelength_nm"]),
                    power_mw=float(rd["power_mw"]), smsr_db=normalized_smsr(rd),
                    peak_count=int(rd["peak_count"]), pdt_code=rd.get("pdt_code"),
                    pdr_code=rd.get("pdr_code"), source=args.output.name,
                ))
                continue
            row = tune_target(
                laser, meter, library, target, args.settle_s,
                previous_codes=previous_codes,
                model_bundle=model_bundle,
            )
            payload["rows"][key] = row
            if row["success"]:
                previous_codes = list(row["codes"])
                rd = row["reading"]
                library.append(Candidate(
                    codes=tuple(row["codes"]),
                    wavelength_nm=float(rd["wavelength_nm"]),
                    power_mw=float(rd["power_mw"]),
                    smsr_db=normalized_smsr(rd),
                    peak_count=int(rd["peak_count"]),
                    pdt_code=rd.get("pdt_code"), pdr_code=rd.get("pdr_code"),
                    source=args.output.name,
                ))
            save_json(args.output, payload)
            rd = row["reading"]
            print(
                f"{index + 1}/{len(targets)} target={target:.6f} "
                f"measured={rd['wavelength_nm']:.6f} "
                f"error={(rd['wavelength_nm'] - target) * 1000.0:+.2f} pm "
                f"power={rd['power_mw']:.4f} mW "
                f"SMSR={normalized_smsr(rd):.1f} dB success={row['success']}",
                flush=True,
            )
    finally:
        # A moderate, valid seed is safer than leaving a failed boundary probe.
        if previous_codes is not None:
            park = list(previous_codes)
            park[0] = current_to_code(60.0, 0)
            park[1] = current_to_code(min(90.0, code_to_current(park[1], 1)), 1)
            try:
                laser.set_codes(park)
            except Exception:
                pass
        meter.close()
        laser.close()

    rows = list(payload["rows"].values())
    passed = [row for row in rows if row.get("success")]
    payload["summary"] = {
        "tested": len(rows),
        "passed": len(passed),
        "failed": len(rows) - len(passed),
        "reachable_start_nm": min((row["target_nm"] for row in passed), default=None),
        "reachable_stop_nm": max((row["target_nm"] for row in passed), default=None),
        "common_raw_power_ceiling_mw": min(
            (float(row["reading"]["power_mw"]) for row in passed), default=None
        ),
        "minimum_smsr_db": min(
            (normalized_smsr(row["reading"]) for row in passed), default=None
        ),
    }
    save_json(args.output, payload)
    return payload


def run_direct(args: argparse.Namespace) -> dict[str, Any]:
    """Close PHASE from one explicitly selected measured Vernier branch."""
    codes = [
        current_to_code(args.gain_ma, 0),
        current_to_code(args.soa_ma, 1),
        current_to_code(args.phase_ma, 2),
        current_to_code(args.wave_a_ma, 3),
        current_to_code(args.wave_b_ma, 4),
    ]
    laser = open_fullband_laser(args.port)
    meter = AQ6150B(args.gpib, average_count=AQ_AVERAGE_COUNT)
    try:
        result = tune_phase_on_branch(
            laser, meter, codes, args.target_nm, args.settle_s,
            max_measurements=args.max_measurements,
        )
        confirmations: list[dict[str, Any]] = []
        if result["success"]:
            for _ in range(args.confirmations):
                _, reading = probe(
                    laser, meter, list(result["codes"]), args.target_nm, args.settle_s
                )
                confirmations.append(asdict(reading))
            result["success"] = all(
                abs(float(row["wavelength_nm"]) - args.target_nm) * 1000.0
                    <= WAVELENGTH_TOLERANCE_PM
                and normalized_smsr(row) >= MIN_SMSR_DB
                for row in confirmations
            )
        result["confirmations"] = confirmations
        payload = {
            "created": utc_now(), "method": "explicit_branch_phase_closure",
            "meter": meter.identity,
            "limits_ma": dict(zip(CONTROL_NAMES, LASER_MAX_MA)),
            "rows": {f"{args.target_nm:.6f}": result},
        }
        save_json(args.output, payload)
        if args.merge_pilot is not None and result["success"]:
            pilot = json.loads(args.merge_pilot.read_text(encoding="utf-8"))
            pilot.setdefault("rows", {})[f"{args.target_nm:.6f}"] = result
            save_json(args.merge_pilot, pilot)
        return payload
    finally:
        park = list(codes)
        park[0] = current_to_code(60.0, 0)
        park[1] = current_to_code(90.0, 1)
        try:
            laser.set_codes(park)
        except Exception:
            pass
        meter.close()
        laser.close()


def load_clean_phase_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    loaded: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        loaded.append((path, json.loads(path.read_text(encoding="utf-8"))))

    # Raising both GAIN and SOA changes the thermal/carrier operating point and
    # therefore shifts the same PHASE/WAVE controls.  Estimate that shift from
    # exact control matches between this run's 145 mA maps and the older live
    # table.  Only small same-mode differences participate; multi-nanometre
    # matches are longitudinal/Vernier hops, not a current coefficient.
    live_by_controls: dict[tuple[int, int, int], list[float]] = {}
    legacy_rows: list[tuple[tuple[int, int, int], float]] = []
    for _path, payload in loaded:
        limits = payload.get("limits_ma", {})
        gain = float(limits.get("gain_ma", float("nan")))
        soa = float(limits.get("soa_ma", float("nan")))
        is_live_145 = math.isclose(gain, 145.0, abs_tol=0.01) and math.isclose(
            soa, 145.0, abs_tol=0.01
        )
        is_legacy = math.isfinite(gain) and math.isfinite(soa) and (
            gain < 144.0 or soa < 144.0
        )
        for row in payload.get("rows", {}).values():
            codes = valid_codes(row.get("codes", ()))
            reading = row.get("reading", {})
            if codes is None or normalized_smsr(reading) < MIN_SMSR_DB:
                continue
            try:
                wavelength = float(reading["wavelength_nm"])
            except (KeyError, TypeError, ValueError):
                continue
            controls = (int(codes[2]), int(codes[3]), int(codes[4]))
            if is_live_145:
                live_by_controls.setdefault(controls, []).append(wavelength)
            elif is_legacy:
                legacy_rows.append((controls, wavelength))
    matched_shifts = [
        live_wave - old_wave
        for controls, old_wave in legacy_rows
        for live_wave in live_by_controls.get(controls, ())
        if abs(live_wave - old_wave) <= 0.20
    ]
    legacy_shift_nm = (
        statistics.median(matched_shifts) if len(matched_shifts) >= 5 else 0.0
    )

    result: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for path, payload in loaded:
        limits = payload.get("limits_ma", {})
        gain = float(limits.get("gain_ma", float("nan")))
        soa = float(limits.get("soa_ma", float("nan")))
        apply_legacy_shift = math.isfinite(gain) and math.isfinite(soa) and (
            gain < 144.0 or soa < 144.0
        )
        for row in payload.get("rows", {}).values():
            codes = valid_codes(row.get("codes", ()))
            reading = row.get("reading", {})
            if codes is None or tuple(codes) in seen:
                continue
            if not bool(row.get("single_mode", normalized_smsr(reading) >= MIN_SMSR_DB)):
                continue
            if normalized_smsr(reading) < MIN_SMSR_DB:
                continue
            try:
                wavelength = float(reading["wavelength_nm"])
                power = float(reading["power_mw"])
            except (KeyError, TypeError, ValueError):
                continue
            seen.add(tuple(codes))
            result.append({
                "codes": list(codes),
                "wavelength_nm": wavelength + (
                    legacy_shift_nm if apply_legacy_shift else 0.0
                ),
                "source_measured_wavelength_nm": wavelength,
                "warm_start_shift_nm": (
                    legacy_shift_nm if apply_legacy_shift else 0.0
                ),
                "power_mw": power, "reading": reading, "source": path.name,
            })
    return result


def run_dense_phase_candidates(args: argparse.Namespace) -> dict[str, Any]:
    """Live-close a wavelength grid from ranked branch candidates."""
    candidate_payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    measured_library = load_clean_phase_rows(args.phase_maps)
    targets = target_grid(args.start_nm, args.stop_nm, args.step_nm)
    if args.resume and args.output.exists():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
    else:
        payload = {
            "created": utc_now(),
            "method": "ranked_measured_branch_plus_live_three_control_jacobian",
            "candidate_file": args.candidates.name,
            "phase_maps": [path.name for path in args.phase_maps],
            "target_start_nm": args.start_nm, "target_stop_nm": args.stop_nm,
            "target_step_nm": args.step_nm, "target_count": len(targets),
            "limits_ma": dict(zip(CONTROL_NAMES, LASER_MAX_MA)),
            "wavelength_tolerance_pm": WAVELENGTH_TOLERANCE_PM,
            "single_mode_gate_smsr_db": MIN_SMSR_DB,
            "rows": {},
        }

    laser = open_fullband_laser(args.port)
    meter = AQ6150B(args.gpib, average_count=AQ_AVERAGE_COUNT)
    payload["meter"] = meter.identity
    previous: dict[str, Any] | None = None
    last_codes: list[int] | None = None
    try:
        for index, target in enumerate(targets):
            key = f"{target:.6f}"
            existing = payload["rows"].get(key)
            if existing and existing.get("success"):
                previous = existing
                last_codes = list(existing["codes"])
                continue

            seeds: list[tuple[list[int], str]] = []
            candidate_row = candidate_payload.get("rows", {}).get(key, {})
            for rank, candidate in enumerate(
                candidate_row.get("candidates", [])[:args.ranked_seeds]
            ):
                seeds.append((at_discovery_power(candidate["codes"]),
                              f"phase_candidate_rank_{rank + 1}"))
            if previous is not None:
                seeds.append((at_discovery_power(previous["codes"]), "previous_dense_point"))

            nearest = sorted(
                measured_library,
                key=lambda row: (
                    abs(float(row["wavelength_nm"]) - target),
                    -float(row["power_mw"]),
                ),
            )[:args.nearest_seeds]
            seeds.extend((at_discovery_power(row["codes"]),
                          f"nearest_measured:{row['source']}") for row in nearest)

            attempts: list[dict[str, Any]] = []
            seen: set[tuple[int, ...]] = set()
            winner: dict[str, Any] | None = None
            winner_source = ""
            screened: list[tuple[list[int], str, Reading]] = []

            # First perform only one measurement per branch.  Most
            # interpolated candidates are already within tolerance; spending
            # a complete Jacobian budget on an obviously wrong branch made
            # mode-gap regions unnecessarily slow.
            for seed_codes, source in seeds:
                identity = tuple(seed_codes)
                if identity in seen:
                    continue
                seen.add(identity)
                _, reading = probe(laser, meter, seed_codes, target, args.settle_s)
                result = {
                    "success": abs(reading.wavelength_nm - target) * 1000.0
                               <= WAVELENGTH_TOLERANCE_PM and mode_ok(reading),
                    "target_nm": target, "codes": list(seed_codes),
                    "currents_ma": [code_to_current(value, control_index)
                                    for control_index, value in enumerate(seed_codes)],
                    "reading": asdict(reading), "local_gradients_nm_per_ma": {},
                    "history": [{"codes": list(seed_codes), "reading": asdict(reading),
                                 "action": "branch_screen"}],
                }
                attempts.append({
                    "source": source, "success": result["success"],
                    "codes": result["codes"], "currents_ma": result["currents_ma"],
                    "reading": result["reading"],
                    "local_gradients_nm_per_ma": result["local_gradients_nm_per_ma"],
                    "history": result["history"],
                })
                if result["success"]:
                    winner, winner_source = result, source
                    break
                screened.append((list(seed_codes), source, reading))

            if winner is None:
                screened.sort(key=lambda item: reading_score(item[2], target))
                for seed_codes, source, _ in screened[:args.refine_seeds]:
                    result = tune_local_three_controls(
                        laser, meter, seed_codes, target, args.settle_s,
                        max_measurements=args.max_measurements,
                    )
                    attempts.append({
                        "source": f"refine:{source}", "success": result["success"],
                        "codes": result["codes"], "currents_ma": result["currents_ma"],
                        "reading": result["reading"],
                        "local_gradients_nm_per_ma": result["local_gradients_nm_per_ma"],
                        "history": result["history"],
                    })
                    if result["success"]:
                        winner, winner_source = result, f"refine:{source}"
                        break

            if winner is None:
                best_attempt = min(
                    attempts,
                    key=lambda row: reading_score(Reading(**row["reading"]), target),
                )
                row = {
                    "success": False, "target_nm": target,
                    "codes": best_attempt["codes"],
                    "currents_ma": best_attempt["currents_ma"],
                    "reading": best_attempt["reading"],
                    "source": best_attempt["source"], "attempts": attempts,
                }
            else:
                row = {
                    "success": True, "target_nm": target,
                    "codes": winner["codes"], "currents_ma": winner["currents_ma"],
                    "reading": winner["reading"], "source": winner_source,
                    "local_gradients_nm_per_ma": winner["local_gradients_nm_per_ma"],
                    "attempts": attempts,
                }
                previous = row
                last_codes = list(row["codes"])
                measured_library.append({
                    "codes": list(row["codes"]),
                    "wavelength_nm": float(row["reading"]["wavelength_nm"]),
                    "power_mw": float(row["reading"]["power_mw"]),
                    "reading": row["reading"], "source": args.output.name,
                })
            payload["rows"][key] = row
            save_json(args.output, payload)
            if (index + 1) % args.print_every == 0 or index == 0 or not row["success"]:
                reading = row["reading"]
                print(
                    f"dense-phase {index + 1}/{len(targets)} target={target:.6f} "
                    f"wave={float(reading['wavelength_nm']):.6f} "
                    f"err={(float(reading['wavelength_nm']) - target) * 1000.0:+.2f}pm "
                    f"power={float(reading['power_mw']):.4f} "
                    f"SMSR={normalized_smsr(reading):.1f} ok={row['success']}",
                    flush=True,
                )
    finally:
        if last_codes is not None:
            park = list(last_codes)
            park[0] = current_to_code(60.0, 0)
            park[1] = current_to_code(90.0, 1)
            try:
                laser.set_codes(park)
            except Exception:
                pass
        meter.close()
        laser.close()

    rows = list(payload["rows"].values())
    passed = [row for row in rows if row.get("success")]
    payload["summary"] = {
        "tested": len(rows), "passed": len(passed), "failed": len(rows) - len(passed),
        "raw_power_min_mw": min(
            (float(row["reading"]["power_mw"]) for row in passed), default=None),
        "raw_power_median_mw": statistics.median(
            [float(row["reading"]["power_mw"]) for row in passed]
        ) if passed else None,
        "raw_power_max_mw": max(
            (float(row["reading"]["power_mw"]) for row in passed), default=None),
        "max_abs_wavelength_error_pm": max(
            (abs(float(row["reading"]["wavelength_nm"]) - float(row["target_nm"])) * 1000.0
             for row in passed), default=None),
        "minimum_smsr_db": min(
            (normalized_smsr(row["reading"]) for row in passed), default=None),
    }
    save_json(args.output, payload)
    return payload


def failure_clusters(rows: dict[str, Any]) -> list[list[str]]:
    keys = sorted(rows, key=float)
    clusters: list[list[str]] = []
    for key in keys:
        if rows[key].get("success"):
            continue
        if not clusters or abs(float(key) - float(clusters[-1][-1]) - TARGET_STEP_NM) > 1e-5:
            clusters.append([key])
        else:
            clusters[-1].append(key)
    return clusters


def run_repair_dense(args: argparse.Namespace) -> dict[str, Any]:
    """Repair failed wavelength clusters by branch-continuous forward/reverse walks."""
    if args.resume and args.output.exists():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
    else:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        payload["repair_created"] = utc_now()
        payload["repair_source"] = args.input.name
        payload["repair_method"] = (
            "same_local_branch_forward_then_reverse_three_control_jacobian"
        )
    rows = payload["rows"]
    ordered_keys = sorted(rows, key=float)
    key_index = {key: index for index, key in enumerate(ordered_keys)}

    laser = open_fullband_laser(args.port)
    meter = AQ6150B(args.gpib, average_count=AQ_AVERAGE_COUNT)
    payload["repair_meter"] = meter.identity
    last_codes: list[int] | None = None
    try:
        for pass_index in range(args.passes):
            clusters = failure_clusters(rows)
            if not clusters:
                break
            print(f"repair pass {pass_index + 1}: {len(clusters)} clusters", flush=True)
            for cluster_number, cluster in enumerate(clusters, 1):
                first_index = key_index[cluster[0]]
                last_index = key_index[cluster[-1]]

                # Walk upward in wavelength from the closest successful point.
                if first_index > 0 and rows[ordered_keys[first_index - 1]].get("success"):
                    previous = rows[ordered_keys[first_index - 1]]
                    for key in cluster:
                        if rows[key].get("success"):
                            previous = rows[key]
                            continue
                        target = float(key)
                        result = tune_local_three_controls(
                            laser, meter, at_discovery_power(previous["codes"]), target,
                            args.settle_s, max_measurements=args.max_measurements,
                        )
                        rows[key].setdefault("repair_attempts", []).append({
                            "direction": "forward", "pass": pass_index + 1,
                            "success": result["success"], "codes": result["codes"],
                            "currents_ma": result["currents_ma"],
                            "reading": result["reading"], "history": result["history"],
                        })
                        if not result["success"]:
                            break
                        old = rows[key]
                        rows[key] = {
                            "success": True, "target_nm": target,
                            "codes": result["codes"], "currents_ma": result["currents_ma"],
                            "reading": result["reading"],
                            "source": "repair_forward_continuous_branch",
                            "local_gradients_nm_per_ma": result["local_gradients_nm_per_ma"],
                            "original_failure": {
                                "codes": old.get("codes"), "reading": old.get("reading"),
                                "source": old.get("source"),
                            },
                            "repair_attempts": old.get("repair_attempts", []),
                        }
                        previous = rows[key]
                        last_codes = list(result["codes"])
                        save_json(args.output, payload)
                        print(
                            f"repair {cluster_number}/{len(clusters)} {target:.6f} "
                            f"forward err={(float(result['reading']['wavelength_nm']) - target) * 1000:+.2f}pm",
                            flush=True,
                        )

                # Any remaining part is attempted from the successful high side.
                if last_index + 1 < len(ordered_keys) \
                        and rows[ordered_keys[last_index + 1]].get("success"):
                    previous = rows[ordered_keys[last_index + 1]]
                    for key in reversed(cluster):
                        if rows[key].get("success"):
                            previous = rows[key]
                            continue
                        target = float(key)
                        result = tune_local_three_controls(
                            laser, meter, at_discovery_power(previous["codes"]), target,
                            args.settle_s, max_measurements=args.max_measurements,
                        )
                        rows[key].setdefault("repair_attempts", []).append({
                            "direction": "reverse", "pass": pass_index + 1,
                            "success": result["success"], "codes": result["codes"],
                            "currents_ma": result["currents_ma"],
                            "reading": result["reading"], "history": result["history"],
                        })
                        if not result["success"]:
                            break
                        old = rows[key]
                        rows[key] = {
                            "success": True, "target_nm": target,
                            "codes": result["codes"], "currents_ma": result["currents_ma"],
                            "reading": result["reading"],
                            "source": "repair_reverse_continuous_branch",
                            "local_gradients_nm_per_ma": result["local_gradients_nm_per_ma"],
                            "original_failure": {
                                "codes": old.get("codes"), "reading": old.get("reading"),
                                "source": old.get("source"),
                            },
                            "repair_attempts": old.get("repair_attempts", []),
                        }
                        previous = rows[key]
                        last_codes = list(result["codes"])
                        save_json(args.output, payload)
                        print(
                            f"repair {cluster_number}/{len(clusters)} {target:.6f} "
                            f"reverse err={(float(result['reading']['wavelength_nm']) - target) * 1000:+.2f}pm",
                            flush=True,
                        )

                # A mirror-current walk can reach the edge of a continuous
                # Vernier branch while still missing the target by a few pm.
                # At that boundary, another mirror step may mode-hop by several
                # nanometres even though a small PHASE move can close the final
                # error on the same longitudinal mode.  Try PHASE-only closure
                # from each immediately adjacent successful row before leaving
                # the point for a later pass.
                for key in cluster:
                    if rows[key].get("success"):
                        continue
                    index = key_index[key]
                    neighbor_seeds: list[tuple[str, dict[str, Any]]] = []
                    if index > 0 and rows[ordered_keys[index - 1]].get("success"):
                        neighbor_seeds.append(("phase_from_low", rows[ordered_keys[index - 1]]))
                    if index + 1 < len(ordered_keys) \
                            and rows[ordered_keys[index + 1]].get("success"):
                        neighbor_seeds.append(("phase_from_high", rows[ordered_keys[index + 1]]))
                    for direction, neighbor in neighbor_seeds:
                        target = float(key)
                        result = tune_phase_on_branch(
                            laser, meter, at_discovery_power(neighbor["codes"]), target,
                            args.settle_s, max_measurements=args.max_measurements,
                        )
                        rows[key].setdefault("repair_attempts", []).append({
                            "direction": direction, "pass": pass_index + 1,
                            "success": result["success"], "codes": result["codes"],
                            "currents_ma": result["currents_ma"],
                            "reading": result["reading"], "history": result["history"],
                        })
                        if not result["success"]:
                            continue
                        old = rows[key]
                        rows[key] = {
                            "success": True, "target_nm": target,
                            "codes": result["codes"], "currents_ma": result["currents_ma"],
                            "reading": result["reading"],
                            "source": f"repair_{direction}_continuous_branch",
                            "phase_slope_nm_per_ma": result["phase_slope_nm_per_ma"],
                            "local_gradients_nm_per_ma": {},
                            "original_failure": {
                                "codes": old.get("codes"), "reading": old.get("reading"),
                                "source": old.get("source"),
                            },
                            "repair_attempts": old.get("repair_attempts", []),
                        }
                        last_codes = list(result["codes"])
                        save_json(args.output, payload)
                        print(
                            f"repair {cluster_number}/{len(clusters)} {target:.6f} "
                            f"{direction} err="
                            f"{(float(result['reading']['wavelength_nm']) - target) * 1000:+.2f}pm",
                            flush=True,
                        )
                        break
            save_json(args.output, payload)
    finally:
        if last_codes is not None:
            park = list(last_codes)
            park[0] = current_to_code(60.0, 0)
            park[1] = current_to_code(90.0, 1)
            try:
                laser.set_codes(park)
            except Exception:
                pass
        meter.close()
        laser.close()

    all_rows = list(rows.values())
    passed = [row for row in all_rows if row.get("success")]
    payload["repair_summary"] = {
        "tested": len(all_rows), "passed": len(passed),
        "failed": len(all_rows) - len(passed),
        "remaining_failure_clusters": len(failure_clusters(rows)),
        "raw_power_min_mw": min(
            (float(row["reading"]["power_mw"]) for row in passed), default=None),
        "raw_power_max_mw": max(
            (float(row["reading"]["power_mw"]) for row in passed), default=None),
        "max_abs_wavelength_error_pm": max(
            (abs(float(row["reading"]["wavelength_nm"]) - float(row["target_nm"])) * 1000
             for row in passed), default=None),
    }
    save_json(args.output, payload)
    return payload


def run_upgrade_low_power(args: argparse.Namespace) -> dict[str, Any]:
    """Replace rows only when a live higher-power branch is valid.

    In adaptive-bottleneck mode, always search the currently weakest measured
    row.  Once that globally weakest row has already exhausted its candidate
    set, its measured maximum is the common-power bottleneck: every unsearched
    row is already above it and therefore cannot lower the common ceiling.
    """
    if args.resume and args.output.exists():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
    else:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        payload["power_upgrade_created"] = utc_now()
        payload["power_upgrade_source"] = args.input.name
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    measured = load_clean_phase_rows(args.phase_maps)
    rows = payload["rows"]
    target_keys = [
        key for key, row in sorted(rows.items(), key=lambda item: float(item[0]))
        if row.get("success") and (
            args.adaptive_bottleneck
            or float(row["reading"]["power_mw"]) < args.floor_mw
        )
    ]
    if args.adaptive_bottleneck:
        # Older checkpoints predate the explicit completion marker, but a
        # persisted attempt list is written only after the whole point search
        # finishes, so it is a safe resume marker.
        for key in target_keys:
            row = rows[key]
            if "power_upgrade_attempts" in row:
                row["power_upgrade_searched_complete"] = True

    laser = open_fullband_laser(args.port)
    meter = AQ6150B(args.gpib, average_count=AQ_AVERAGE_COUNT)
    payload["power_upgrade_meter"] = meter.identity
    last_codes: list[int] | None = None
    upgraded = 0
    try:
        for index in range(len(target_keys)):
            if args.adaptive_bottleneck:
                key = min(
                    target_keys,
                    key=lambda item: float(rows[item]["reading"]["power_mw"]),
                )
                if rows[key].get("power_upgrade_searched_complete"):
                    payload["adaptive_bottleneck_stop"] = {
                        "target_nm": float(key),
                        "power_mw": float(rows[key]["reading"]["power_mw"]),
                        "reason": "globally_weakest_row_candidate_search_complete",
                    }
                    save_json(args.output, payload)
                    break
                target = float(key)
            else:
                key = target_keys[index]
                target = float(key)
            original = rows[key]
            original_power = float(original["reading"]["power_mw"])
            if not args.adaptive_bottleneck and original_power >= args.floor_mw:
                continue

            seeds: list[tuple[list[int], str, float]] = []
            for rank, candidate in enumerate(
                candidates.get("rows", {}).get(key, {}).get("candidates", [])
            ):
                predicted = float(candidate["predicted_power_mw"])
                if predicted >= original_power + args.minimum_gain_mw:
                    seeds.append((at_discovery_power(candidate["codes"]),
                                  f"phase_candidate_rank_{rank + 1}", predicted))

            nearby = [
                row for row in measured
                if abs(float(row["wavelength_nm"]) - target) <= args.search_radius_nm
                and float(row["power_mw"]) >= original_power + args.minimum_gain_mw
            ]
            nearby.sort(key=lambda row: (
                abs(float(row["wavelength_nm"]) - target),
                -float(row["power_mw"]),
            ))
            seeds.extend((at_discovery_power(row["codes"]),
                          f"measured:{row['source']}", float(row["power_mw"]))
                         for row in nearby[:args.measured_seeds])

            unique: list[tuple[list[int], str, float]] = []
            seen: set[tuple[int, ...]] = set()
            for seed in seeds:
                identity = tuple(seed[0])
                if identity not in seen:
                    seen.add(identity)
                    unique.append(seed)
            seeds = unique[:args.screen_seeds]

            screenings: list[tuple[list[int], str, Reading]] = []
            successful: list[tuple[dict[str, Any], str]] = []
            attempts: list[dict[str, Any]] = []
            for seed_codes, source, predicted_power in seeds:
                _, reading = probe(laser, meter, seed_codes, target, args.settle_s)
                ok = abs(reading.wavelength_nm - target) * 1000.0 \
                     <= WAVELENGTH_TOLERANCE_PM and mode_ok(reading)
                attempts.append({
                    "source": source, "stage": "screen", "success": ok,
                    "predicted_power_mw": predicted_power,
                    "codes": seed_codes, "reading": asdict(reading),
                })
                screenings.append((seed_codes, source, reading))
                if ok and reading.power_mw > original_power + args.minimum_gain_mw:
                    successful.append(({
                        "success": True, "target_nm": target, "codes": seed_codes,
                        "currents_ma": [code_to_current(value, control_index)
                                        for control_index, value in enumerate(seed_codes)],
                        "reading": asdict(reading),
                        "local_gradients_nm_per_ma": {}, "history": [],
                    }, source))

            if not successful:
                screenings.sort(key=lambda item: reading_score(item[2], target))
                for seed_codes, source, screened_reading in screenings[:args.refine_seeds]:
                    if screened_reading.power_mw <= original_power + args.minimum_gain_mw:
                        continue
                    result = tune_local_three_controls(
                        laser, meter, seed_codes, target, args.settle_s,
                        max_measurements=args.max_measurements,
                    )
                    attempts.append({
                        "source": source, "stage": "refine",
                        "success": result["success"], "codes": result["codes"],
                        "reading": result["reading"], "history": result["history"],
                    })
                    if result["success"] and float(result["reading"]["power_mw"]) \
                            > original_power + args.minimum_gain_mw:
                        successful.append((result, f"refine:{source}"))

            if successful:
                winner, source = max(
                    successful, key=lambda item: float(item[0]["reading"]["power_mw"])
                )
                rows[key] = {
                    "success": True, "target_nm": target,
                    "codes": winner["codes"], "currents_ma": winner["currents_ma"],
                    "reading": winner["reading"], "source": f"power_upgrade:{source}",
                    "local_gradients_nm_per_ma": winner.get(
                        "local_gradients_nm_per_ma", {}),
                    "original_lower_power_row": {
                        "codes": original.get("codes"), "currents_ma": original.get("currents_ma"),
                        "reading": original.get("reading"), "source": original.get("source"),
                    },
                    "power_upgrade_attempts": attempts,
                    "power_upgrade_searched_complete": True,
                }
                last_codes = list(winner["codes"])
                upgraded += 1
            else:
                original.setdefault("power_upgrade_attempts", []).extend(attempts)
                original["power_upgrade_searched_complete"] = True
            save_json(args.output, payload)
            current_power = float(rows[key]["reading"]["power_mw"])
            status = (
                "searched_complete=True" if args.adaptive_bottleneck
                else f"ok={current_power >= args.floor_mw}"
            )
            print(
                f"upgrade {index + 1}/{len(target_keys)} {target:.6f} "
                f"{original_power:.4f}->{current_power:.4f}mW "
                f"{status}",
                flush=True,
            )
    finally:
        if last_codes is not None:
            park = list(last_codes)
            park[0] = current_to_code(60.0, 0)
            park[1] = current_to_code(90.0, 1)
            try:
                laser.set_codes(park)
            except Exception:
                pass
        meter.close()
        laser.close()

    all_rows = [row for row in rows.values() if row.get("success")]
    powers = [float(row["reading"]["power_mw"]) for row in all_rows]
    searched_points = sum(
        bool(row.get("power_upgrade_searched_complete")) for row in all_rows
    )
    bottleneck = min(all_rows, key=lambda row: float(row["reading"]["power_mw"]))
    payload["power_upgrade_summary"] = {
        "target_floor_mw": args.floor_mw,
        "strategy": (
            "adaptive_global_bottleneck" if args.adaptive_bottleneck
            else "all_rows_below_fixed_floor"
        ),
        "eligible_points": len(target_keys),
        "searched_points": searched_points,
        "upgraded_points_this_run": upgraded,
        "upgraded_points_total": sum(
            str(row.get("source", "")).startswith("power_upgrade:") for row in all_rows
        ),
        "points_still_below_floor": (
            None if args.adaptive_bottleneck
            else sum(value < args.floor_mw for value in powers)
        ),
        "bottleneck_target_nm": float(bottleneck["target_nm"]),
        "bottleneck_search_complete": bool(
            bottleneck.get("power_upgrade_searched_complete")
        ),
        "raw_power_min_mw": min(powers),
        "raw_power_median_mw": statistics.median(powers),
        "raw_power_max_mw": max(powers),
    }
    save_json(args.output, payload)
    return payload


def run_soa_pilot(args: argparse.Namespace) -> dict[str, Any]:
    table = json.loads(args.input.read_text(encoding="utf-8"))
    ordered = sorted(table["rows"].values(), key=lambda row: float(row["target_nm"]))
    by_power = sorted(ordered, key=lambda row: float(row["reading"]["power_mw"]))
    selected: dict[str, dict[str, Any]] = {}
    for index in np.linspace(0, len(ordered) - 1, args.wavelength_samples, dtype=int):
        row = ordered[int(index)]
        selected[f"{float(row['target_nm']):.6f}"] = row
    for row in by_power[:args.low_power_samples] + by_power[-args.high_power_samples:]:
        selected[f"{float(row['target_nm']):.6f}"] = row
    rows = list(selected.values())
    soa_levels = np.arange(args.soa_max_ma, args.soa_min_ma - 0.001,
                           -args.soa_step_ma).tolist()
    payload = {
        "created": utc_now(), "method": "measured_normalized_soa_transfer_curve",
        "source_table": args.input.name, "soa_levels_ma": soa_levels,
        "limits_ma": dict(zip(CONTROL_NAMES, LASER_MAX_MA)), "rows": {},
    }
    laser = open_fullband_laser(args.port)
    meter = AQ6150B(args.gpib, average_count=AQ_AVERAGE_COUNT)
    payload["meter"] = meter.identity
    last_codes: list[int] | None = None
    try:
        for row_index, row in enumerate(rows):
            target = float(row["target_nm"])
            levels = soa_levels if row_index % 2 == 0 else list(reversed(soa_levels))
            measurements: list[dict[str, Any]] = []
            for soa_ma in levels:
                codes = list(row["codes"])
                codes[1] = current_to_code(soa_ma, 1)
                _, reading = probe(laser, meter, codes, target, args.settle_s)
                measurements.append({
                    "soa_ma": code_to_current(codes[1], 1), "codes": codes,
                    "reading": asdict(reading), "single_mode": mode_ok(reading),
                })
                last_codes = codes
            payload["rows"][f"{target:.6f}"] = {
                "target_nm": target, "baseline_codes": row["codes"],
                "baseline_power_mw": row["reading"]["power_mw"],
                "measurements": measurements,
            }
            save_json(args.output, payload)
            print(f"soa-pilot {row_index + 1}/{len(rows)} target={target:.2f}", flush=True)
    finally:
        if last_codes is not None:
            park = list(last_codes)
            park[0] = current_to_code(60.0, 0)
            park[1] = current_to_code(90.0, 1)
            try:
                laser.set_codes(park)
            except Exception:
                pass
        meter.close()
        laser.close()

    ratios: dict[str, list[float]] = {f"{level:.6f}": [] for level in soa_levels}
    for row in payload["rows"].values():
        baseline = next(
            (m for m in row["measurements"]
             if abs(float(m["soa_ma"]) - args.soa_max_ma) < 0.02
             and bool(m.get("single_mode"))), None
        )
        if baseline is None:
            continue
        base_power = float(baseline["reading"]["power_mw"])
        if not math.isfinite(base_power) or base_power <= 0.0:
            continue
        for measurement in row["measurements"]:
            # A mode hop changes the optical branch, so its power ratio is not
            # a valid SOA transfer sample for closing the original branch.
            if not bool(measurement.get("single_mode")):
                continue
            key = min(ratios, key=lambda value: abs(float(value) - float(measurement["soa_ma"])))
            measured_power = float(measurement["reading"]["power_mw"])
            if math.isfinite(measured_power) and measured_power > 0.0:
                ratios[key].append(measured_power / base_power)
    payload["summary"] = {
        "representative_targets": len(rows),
        "normalized_power_ratio_median_by_soa": {
            key: statistics.median(values) for key, values in ratios.items() if values
        },
        "normalized_power_ratio_min_by_soa": {
            key: min(values) for key, values in ratios.items() if values
        },
        "normalized_power_ratio_max_by_soa": {
            key: max(values) for key, values in ratios.items() if values
        },
    }
    save_json(args.output, payload)
    return payload


def load_soa_transfer_curve(path: Path) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    """Load the measured SOA/power curve and retain wavelength-shift evidence."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    ratios = summary["normalized_power_ratio_median_by_soa"]
    curve = sorted((float(soa), float(ratio)) for soa, ratio in ratios.items())
    if len(curve) < 2:
        raise ValueError(f"SOA pilot {path} contains fewer than two current levels")
    if any(right[0] <= left[0] for left, right in zip(curve, curve[1:])):
        raise ValueError(f"SOA pilot {path} has duplicate current levels")
    # Small non-monotonic meter noise must not make the inverse ambiguous.
    monotonic: list[tuple[float, float]] = []
    running = -math.inf
    for soa_ma, ratio in curve:
        running = max(running, ratio)
        monotonic.append((soa_ma, running))
    return monotonic, payload


def soa_for_power_ratio(curve: list[tuple[float, float]], ratio: float) -> float:
    ordered = sorted((power_ratio, soa_ma) for soa_ma, power_ratio in curve)
    ratios = np.asarray([item[0] for item in ordered], dtype=np.float64)
    currents = np.asarray([item[1] for item in ordered], dtype=np.float64)
    return float(np.interp(ratio, ratios, currents,
                           left=currents[0], right=currents[-1]))


def soa_power_slope(
    curve: list[tuple[float, float]], baseline_power_mw: float, soa_ma: float,
) -> float:
    """Return the local measured d(power)/d(SOA), in mW/mA."""
    for (left_ma, left_ratio), (right_ma, right_ratio) in zip(curve, curve[1:]):
        if left_ma <= soa_ma <= right_ma:
            return baseline_power_mw * (right_ratio - left_ratio) / (right_ma - left_ma)
    left_ma, left_ratio = curve[0] if soa_ma < curve[0][0] else curve[-2]
    right_ma, right_ratio = curve[1] if soa_ma < curve[0][0] else curve[-1]
    return baseline_power_mw * (right_ratio - left_ratio) / (right_ma - left_ma)


def soa_wavelength_shift_curve(pilot: dict[str, Any]) -> list[tuple[float, float]]:
    """Median same-mode wavelength shift relative to the pilot's maximum SOA."""
    levels = [float(value) for value in pilot.get("soa_levels_ma", [])]
    if not levels:
        return []
    maximum = max(levels)
    shifts: dict[float, list[float]] = {level: [] for level in levels}
    for row in pilot.get("rows", {}).values():
        measurements = row.get("measurements", [])
        baseline = min(
            measurements, key=lambda item: abs(float(item["soa_ma"]) - maximum),
            default=None,
        )
        if baseline is None:
            continue
        base_wave = float(baseline["reading"]["wavelength_nm"])
        for measurement in measurements:
            if not measurement.get("single_mode"):
                continue
            shift = float(measurement["reading"]["wavelength_nm"]) - base_wave
            # Exclude an obvious mode hop; it is not a local SOA coefficient.
            if abs(shift) > 0.20:
                continue
            level = min(levels, key=lambda value: abs(value - float(measurement["soa_ma"])))
            shifts[level].append(shift)
    return sorted(
        (level, statistics.median(values))
        for level, values in shifts.items() if values
    )


def interpolate_curve(curve: list[tuple[float, float]], coordinate: float) -> float:
    if not curve:
        return 0.0
    x = np.asarray([item[0] for item in curve], dtype=np.float64)
    y = np.asarray([item[1] for item in curve], dtype=np.float64)
    return float(np.interp(coordinate, x, y, left=y[0], right=y[-1]))


def compensated_branch_codes(
    source_rows: list[dict[str, Any]], desired_high_power_nm: float,
    fallback_codes: list[int],
) -> tuple[list[int], bool]:
    """Interpolate controls only between adjacent points on one smooth branch."""
    targets = np.asarray([float(row["target_nm"]) for row in source_rows])
    right = int(np.searchsorted(targets, desired_high_power_nm, side="left"))
    if right <= 0 or right >= len(source_rows):
        return list(fallback_codes), False
    left_row, right_row = source_rows[right - 1], source_rows[right]
    left_nm, right_nm = float(left_row["target_nm"]), float(right_row["target_nm"])
    left_currents = [float(value) for value in left_row["currents_ma"]]
    right_currents = [float(value) for value in right_row["currents_ma"]]
    # These limits deliberately reject a Vernier/longitudinal-mode branch jump.
    if (abs(right_currents[2] - left_currents[2]) > 1.2
            or abs(right_currents[3] - left_currents[3]) > 0.60
            or abs(right_currents[4] - left_currents[4]) > 0.60):
        return list(fallback_codes), False
    fraction = min(max((desired_high_power_nm - left_nm) / (right_nm - left_nm), 0.0), 1.0)
    result = list(fallback_codes)
    for control_index in (2, 3, 4):
        current = left_currents[control_index] + fraction * (
            right_currents[control_index] - left_currents[control_index]
        )
        result[control_index] = current_to_code(current, control_index)
    return result, True


def run_equalize_power(args: argparse.Namespace) -> dict[str, Any]:
    """Close wavelength, single-mode state and output power at every grid point."""
    source = json.loads(args.input.read_text(encoding="utf-8"))
    power_target_selection = None
    if args.target_power_mw is None:
        power_target_selection = recommend_common_power_target(
            source["rows"],
            headroom_relative=args.power_headroom_relative,
            lift_relative=args.power_lift_relative,
            resolution_mw=args.power_resolution_mw,
        )
        args.target_power_mw = power_target_selection["recommended_target_power_mw"]
    curve, pilot = load_soa_transfer_curve(args.soa_pilot)
    source_rows = sorted(
        source["rows"].values(), key=lambda row: float(row["target_nm"])
    )
    selected = [
        row for row in source_rows
        if args.start_nm - 1e-9 <= float(row["target_nm"]) <= args.stop_nm + 1e-9
    ]
    wavelength_shift_curve = soa_wavelength_shift_curve(pilot)
    if args.resume and args.output.exists():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        if not math.isclose(
            float(payload.get("target_power_mw", -1.0)),
            float(args.target_power_mw),
            abs_tol=1e-12,
        ):
            raise ValueError("resume output belongs to a different common power target")
        for existing_row in payload.get("rows", {}).values():
            if "history" in existing_row:
                existing_row.setdefault("initial_equalization_attempted", True)
    else:
        payload = {
            "created": utc_now(),
            "method": "live_soa_power_inversion_plus_wavelength_meter_closed_loop",
            "source_table": args.input.name,
            "soa_transfer_pilot": args.soa_pilot.name,
            "target_power_mw": args.target_power_mw,
            "power_relative_tolerance": args.power_relative_tolerance,
            "wavelength_tolerance_pm": WAVELENGTH_TOLERANCE_PM,
            "single_mode_gate_smsr_db": MIN_SMSR_DB,
            "limits_ma": dict(zip(CONTROL_NAMES, LASER_MAX_MA)),
            "measured_soa_curve": curve,
            "rows": {},
        }
        if power_target_selection is not None:
            payload["power_target_selection"] = power_target_selection

    laser = open_fullband_laser(args.port)
    meter = AQ6150B(args.gpib, average_count=AQ_AVERAGE_COUNT)
    payload["meter"] = meter.identity
    last_codes: list[int] | None = None
    try:
        for index, original in enumerate(selected):
            target_nm = float(original["target_nm"])
            key = f"{target_nm:.6f}"
            existing = payload["rows"].get(key)
            if existing and (
                existing.get("success")
                or existing.get("initial_equalization_attempted")
            ):
                last_codes = list(existing["codes"])
                continue

            baseline_power = float(original["reading"]["power_mw"])
            desired_ratio = args.target_power_mw / baseline_power
            initial_soa = soa_for_power_ratio(curve, desired_ratio)
            initial_soa = min(max(initial_soa, args.soa_min_ma), args.soa_max_ma)
            predicted_shift_nm = interpolate_curve(wavelength_shift_curve, initial_soa)
            desired_high_power_nm = target_nm - predicted_shift_nm
            codes, used_branch_interpolation = compensated_branch_codes(
                source_rows, desired_high_power_nm,
                [int(value) for value in original["codes"]],
            )
            codes[1] = current_to_code(initial_soa, 1)
            history: list[dict[str, Any]] = []
            power_observations: list[tuple[float, float]] = []
            _, reading = probe(laser, meter, codes, target_nm, args.settle_s)
            history.append({
                "action": "predicted_soa_with_branch_compensation", "codes": list(codes),
                "reading": asdict(reading), "predicted_soa_shift_nm": predicted_shift_nm,
                "desired_high_power_nm": desired_high_power_nm,
                "used_branch_interpolation": used_branch_interpolation,
            })

            for round_index in range(args.max_rounds):
                wavelength_ok = (
                    abs(reading.wavelength_nm - target_nm) * 1000.0
                    <= WAVELENGTH_TOLERANCE_PM and mode_ok(reading)
                )
                if not wavelength_ok:
                    tuned, tuning_attempts = tune_wavelength_with_phase_fallback(
                        laser, meter, codes, target_nm, args.settle_s,
                        args.max_tune_measurements,
                    )
                    history.append({
                        "action": f"wavelength_close_{round_index}",
                        "success": tuned["success"],
                        "local_gradients_nm_per_ma": tuned.get(
                            "local_gradients_nm_per_ma", {}),
                        "phase_slope_nm_per_ma": tuned.get("phase_slope_nm_per_ma"),
                        "strategy_attempts": tuning_attempts,
                        "measurements": tuned["history"],
                    })
                    codes = [int(value) for value in tuned["codes"]]
                    reading = Reading(**tuned["reading"])

                wavelength_ok = (
                    abs(reading.wavelength_nm - target_nm) * 1000.0
                    <= WAVELENGTH_TOLERANCE_PM and mode_ok(reading)
                )
                if not wavelength_ok:
                    # The combined mirror/PHASE budget is already exhausted.
                    # Further SOA moves only perturb power on an invalid branch;
                    # leave this point for bidirectional neighbor repair.
                    break

                relative_error = (
                    (reading.power_mw - args.target_power_mw) / args.target_power_mw
                )
                wavelength_ok = (
                    abs(reading.wavelength_nm - target_nm) * 1000.0
                    <= WAVELENGTH_TOLERANCE_PM and mode_ok(reading)
                )
                if wavelength_ok and abs(relative_error) <= args.power_relative_tolerance:
                    break

                current_soa = code_to_current(codes[1], 1)
                power_observations.append((current_soa, reading.power_mw))
                slope = soa_power_slope(curve, baseline_power, current_soa)
                if len(power_observations) >= 2:
                    old_soa, old_power = power_observations[-2]
                    delta_soa = current_soa - old_soa
                    measured_slope = (
                        (reading.power_mw - old_power) / delta_soa
                        if abs(delta_soa) >= 0.2 else float("nan")
                    )
                    if math.isfinite(measured_slope) and 0.001 <= measured_slope <= 0.05:
                        slope = 0.65 * measured_slope + 0.35 * slope
                if not math.isfinite(slope) or slope <= 0.0005:
                    slope = max(baseline_power * 0.006, 0.005)
                requested_soa = current_soa + (
                    args.target_power_mw - reading.power_mw
                ) / slope
                requested_soa = min(
                    max(requested_soa, current_soa - args.max_soa_step_ma),
                    current_soa + args.max_soa_step_ma,
                )
                requested_soa = min(max(requested_soa, args.soa_min_ma), args.soa_max_ma)
                new_code = current_to_code(requested_soa, 1)
                if new_code == codes[1]:
                    break
                codes[1] = new_code
                _, reading = probe(laser, meter, codes, target_nm, args.settle_s)
                history.append({
                    "action": f"power_close_{round_index}", "codes": list(codes),
                    "reading": asdict(reading), "estimated_power_slope_mw_per_ma": slope,
                })

            wavelength_error_pm = (reading.wavelength_nm - target_nm) * 1000.0
            power_relative_error = (
                (reading.power_mw - args.target_power_mw) / args.target_power_mw
            )
            success = (
                abs(wavelength_error_pm) <= WAVELENGTH_TOLERANCE_PM
                and mode_ok(reading)
                and abs(power_relative_error) <= args.power_relative_tolerance
            )
            row = {
                "success": success, "target_nm": target_nm, "codes": list(codes),
                "currents_ma": [code_to_current(value, control_index)
                                for control_index, value in enumerate(codes)],
                "reading": asdict(reading), "wavelength_error_pm": wavelength_error_pm,
                "power_error_mw": reading.power_mw - args.target_power_mw,
                "power_relative_error": power_relative_error,
                "initial_equalization_attempted": True,
                "source_max_power_row": {
                    "codes": original["codes"], "currents_ma": original["currents_ma"],
                    "reading": original["reading"], "source": original.get("source"),
                },
                "history": history,
            }
            payload["rows"][key] = row
            last_codes = list(codes)
            save_json(args.output, payload)
            if ((index + 1) % args.print_every == 0 or not success or index == 0):
                print(
                    f"equalize {index + 1}/{len(selected)} target={target_nm:.6f} "
                    f"wave={reading.wavelength_nm:.6f} err={wavelength_error_pm:+.2f}pm "
                    f"power={reading.power_mw:.5f}mW "
                    f"perr={power_relative_error * 100.0:+.2f}% "
                    f"SOA={code_to_current(codes[1], 1):.3f}mA ok={success}",
                    flush=True,
                )
    finally:
        if last_codes is not None:
            park = list(last_codes)
            park[0] = current_to_code(60.0, 0)
            park[1] = current_to_code(90.0, 1)
            try:
                laser.set_codes(park)
            except Exception:
                pass
        meter.close()
        laser.close()

    rows = list(payload["rows"].values())
    passed = [row for row in rows if row.get("success")]
    powers = [float(row["reading"]["power_mw"]) for row in passed]
    payload["summary"] = {
        "tested": len(rows), "passed": len(passed), "failed": len(rows) - len(passed),
        "target_power_mw": args.target_power_mw,
        "measured_power_min_mw": min(powers, default=None),
        "measured_power_median_mw": statistics.median(powers) if powers else None,
        "measured_power_max_mw": max(powers, default=None),
        "maximum_abs_wavelength_error_pm": max(
            (abs(float(row["wavelength_error_pm"])) for row in passed), default=None),
        "minimum_smsr_db": min(
            (normalized_smsr(row["reading"]) for row in passed), default=None),
        "soa_current_min_ma": min(
            (float(row["currents_ma"][1]) for row in passed), default=None),
        "soa_current_max_ma": max(
            (float(row["currents_ma"][1]) for row in passed), default=None),
    }
    payload["soa_pilot_summary"] = pilot.get("summary", {})
    save_json(args.output, payload)
    return payload


def close_equal_power_from_seed(
    laser: LaserSerial,
    meter: AQ6150B,
    seed_codes: list[int],
    target_nm: float,
    target_power_mw: float,
    baseline_power_mw: float,
    soa_curve: list[tuple[float, float]],
    settle_s: float,
    power_relative_tolerance: float,
    max_rounds: int,
    max_tune_measurements: int,
    closure_power_relative_tolerance: float | None = None,
    closure_wavelength_tolerance_pm: float | None = None,
    closure_wavelength_offset_pm: float = 0.0,
) -> dict[str, Any]:
    """Follow an already-valid neighboring branch into one missing grid point."""
    target_power_tolerance = min(
        float(power_relative_tolerance),
        float(closure_power_relative_tolerance)
        if closure_power_relative_tolerance is not None
        else float(power_relative_tolerance),
    )
    target_wavelength_tolerance_pm = min(
        WAVELENGTH_TOLERANCE_PM,
        float(closure_wavelength_tolerance_pm)
        if closure_wavelength_tolerance_pm is not None
        else WAVELENGTH_TOLERANCE_PM,
    )
    if target_power_tolerance <= 0.0 or target_wavelength_tolerance_pm <= 0.0:
        raise ValueError("closure guard-band tolerances must be positive")
    closure_wavelength_offset_pm = max(
        min(float(closure_wavelength_offset_pm), 1.0), -1.0,
    )
    closure_target_nm = target_nm + closure_wavelength_offset_pm / 1000.0
    codes = list(seed_codes)
    history: list[dict[str, Any]] = []
    reading: Reading | None = None
    observations: list[tuple[float, float]] = []
    for round_index in range(max_rounds):
        tuned, tuning_attempts = tune_wavelength_with_phase_fallback(
            laser, meter, codes, closure_target_nm, settle_s,
            max_tune_measurements,
            wavelength_tolerance_pm=target_wavelength_tolerance_pm,
        )
        codes = [int(value) for value in tuned["codes"]]
        reading = Reading(**tuned["reading"])
        history.append({
            "action": f"neighbor_wavelength_close_{round_index}",
            "success": tuned["success"],
            "local_gradients_nm_per_ma": tuned.get("local_gradients_nm_per_ma", {}),
            "phase_slope_nm_per_ma": tuned.get("phase_slope_nm_per_ma"),
            "strategy_attempts": tuning_attempts,
            "measurements": tuned["history"],
        })
        wave_ok = (
            abs(reading.wavelength_nm - closure_target_nm) * 1000.0
            <= target_wavelength_tolerance_pm
            and mode_ok(reading)
        )
        relative_error = (reading.power_mw - target_power_mw) / target_power_mw
        if wave_ok and abs(relative_error) <= target_power_tolerance:
            confirmations = [reading]
            while len(confirmations) < CALIBRATION_CONFIRMATION_SAMPLES:
                _, sample = probe(
                    laser, meter, codes, closure_target_nm,
                    min(settle_s, CALIBRATION_CONFIRM_SETTLE_S),
                )
                confirmations.append(sample)
            reading, mode_pass_count = aggregate_repeatability_readings(
                confirmations, closure_target_nm,
            )
            wave_ok = (
                abs(reading.wavelength_nm - closure_target_nm) * 1000.0
                <= target_wavelength_tolerance_pm
                and mode_ok(reading)
                and mode_pass_count >= CALIBRATION_CONFIRMATION_SAMPLES // 2 + 1
            )
            relative_error = (reading.power_mw - target_power_mw) / target_power_mw
            history.append({
                "action": f"neighbor_candidate_confirmation_{round_index}",
                "codes": list(codes),
                "reading": asdict(reading),
                "samples": [asdict(sample) for sample in confirmations],
                "mode_pass_count": mode_pass_count,
            })
            if wave_ok and abs(relative_error) <= target_power_tolerance:
                break
        if not wave_ok:
            break

        current_soa = code_to_current(codes[1], 1)
        observations.append((current_soa, reading.power_mw))
        slope = soa_power_slope(soa_curve, baseline_power_mw, current_soa)
        if len(observations) >= 2:
            old_soa, old_power = observations[-2]
            if abs(current_soa - old_soa) >= 0.2:
                measured = (reading.power_mw - old_power) / (current_soa - old_soa)
                if 0.001 <= measured <= 0.05:
                    slope = 0.7 * measured + 0.3 * slope
        if not math.isfinite(slope) or slope <= 0.0005:
            slope = max(baseline_power_mw * 0.006, 0.005)
        requested = current_soa + (target_power_mw - reading.power_mw) / slope
        requested = min(max(requested, current_soa - 3.0), current_soa + 3.0)
        requested = min(max(requested, 85.0), MAX_GAIN_SOA_MA)
        new_code = current_to_code(requested, 1)
        if new_code == codes[1]:
            break
        codes[1] = new_code
        _, reading = probe(laser, meter, codes, closure_target_nm, settle_s)
        history.append({
            "action": f"neighbor_power_close_{round_index}",
            "codes": list(codes), "reading": asdict(reading),
            "estimated_power_slope_mw_per_ma": slope,
        })

    if reading is None:
        _, reading = probe(laser, meter, codes, closure_target_nm, settle_s)
    wavelength_error_pm = (reading.wavelength_nm - target_nm) * 1000.0
    closure_wavelength_error_pm = (
        reading.wavelength_nm - closure_target_nm
    ) * 1000.0
    relative_error = (reading.power_mw - target_power_mw) / target_power_mw
    # A certification repair is deliberately asked to land inside a tighter
    # guard band than the public release limits.  Do not silently promote a
    # candidate that missed that inner target merely because it still scrapes
    # through the +/-2 pm or operator-selected power release boundary.  Such a
    # row has no drift margin and was observed to fail at a different set of
    # points on the following 2001-point thermal sweep.
    success = (
        abs(closure_wavelength_error_pm) <= target_wavelength_tolerance_pm
        and abs(wavelength_error_pm) <= WAVELENGTH_TOLERANCE_PM
        and mode_ok(reading)
        and abs(relative_error) <= target_power_tolerance
    )
    return {
        "success": success, "target_nm": target_nm, "codes": codes,
        "currents_ma": [code_to_current(value, index)
                        for index, value in enumerate(codes)],
        "reading": asdict(reading), "wavelength_error_pm": wavelength_error_pm,
        "power_error_mw": reading.power_mw - target_power_mw,
        "power_relative_error": relative_error, "history": history,
        "closure_wavelength_offset_pm": closure_wavelength_offset_pm,
        "closure_wavelength_target_nm": closure_target_nm,
        "closure_wavelength_error_pm": closure_wavelength_error_pm,
    }


def close_equal_power_by_current_continuation(
    laser: LaserSerial,
    meter: AQ6150B,
    source_row: dict[str, Any],
    target_nm: float,
    target_power_mw: float,
    settle_s: float,
    power_relative_tolerance: float,
    max_tune_measurements: int,
) -> dict[str, Any]:
    """Reach equal power while continuously following the source cavity mode.

    Some grid points sit immediately before a PHASE/Vernier branch boundary.
    A single SOA jump can move the desired wavelength into a mode gap even
    though the point is valid at the measured 145/145 mA source operating
    point.  Walk several GAIN/SOA attenuation splits in small increments,
    closing wavelength after every increment, then bisect the first power
    bracket.  This preserves the source branch instead of copying either
    neighboring grid point across the discontinuity.
    """
    source_codes = [int(value) for value in source_row["codes"]]
    source_reading = Reading(**source_row["reading"])
    source_target_nm = float(source_row.get("target_nm", source_reading.wavelength_nm))
    needs_mirror_entry = abs(source_target_nm - target_nm) * 1000.0 > WAVELENGTH_TOLERANCE_PM
    history: list[dict[str, Any]] = []
    best_codes = list(source_codes)
    best_reading = source_reading

    def candidate_score(reading: Reading) -> tuple[float, float, float]:
        wave_penalty = 0.0 if (
            abs(reading.wavelength_nm - target_nm) * 1000.0
            <= WAVELENGTH_TOLERANCE_PM and mode_ok(reading)
        ) else 1000.0 + abs(reading.wavelength_nm - target_nm) * 1000.0
        power_error = abs(reading.power_mw - target_power_mw) / target_power_mw
        return wave_penalty, power_error, -normalized_smsr(asdict(reading))

    def tune_at_currents(
        seed_codes: list[int], gain_ma: float, soa_ma: float, label: str,
        allow_mirrors: bool = False,
    ) -> tuple[list[int], Reading, bool]:
        codes = list(seed_codes)
        codes[0] = current_to_code(gain_ma, 0)
        codes[1] = current_to_code(soa_ma, 1)

        # A small continuation step needs only PHASE.  Do not move either
        # Vernier mirror here: crossing that boundary is the failure this path
        # is specifically designed to avoid.
        phase = tune_phase_on_branch(
            laser, meter, codes, target_nm, settle_s,
            max_measurements=min(4, max_tune_measurements),
        )
        attempts: list[dict[str, Any]] = [{
            "method": "phase_continuation", "success": phase["success"],
            "codes": phase["codes"], "reading": phase["reading"],
            "measurements": phase["history"],
        }]
        tuned = phase
        if not phase["success"] and allow_mirrors:
            combined, combined_attempts = tune_wavelength_with_phase_fallback(
                laser, meter, codes, target_nm, settle_s,
                max_measurements=min(max_tune_measurements, 20),
            )
            attempts.extend(combined_attempts)
            if combined["success"] or reading_score(
                Reading(**combined["reading"]), target_nm
            ) < reading_score(Reading(**tuned["reading"]), target_nm):
                tuned = combined
        reading = Reading(**tuned["reading"])
        wave_ok = (
            abs(reading.wavelength_nm - target_nm) * 1000.0
            <= WAVELENGTH_TOLERANCE_PM and mode_ok(reading)
        )
        history.append({
            "action": label, "requested_gain_ma": gain_ma,
            "requested_soa_ma": soa_ma, "success": wave_ok,
            "codes": [int(value) for value in tuned["codes"]],
            "reading": asdict(reading), "strategy_attempts": attempts,
        })
        print(
            f"continuation-step {label} target={target_nm:.6f} "
            f"GAIN={gain_ma:.3f} SOA={soa_ma:.3f} "
            f"err={(reading.wavelength_nm - target_nm) * 1000.0:+.2f}pm "
            f"power={reading.power_mw:.6f}mW wave_ok={wave_ok}",
            flush=True,
        )
        return [int(value) for value in tuned["codes"]], reading, wave_ok

    # Start with a small-step SOA-only walk.  The ordinary equalizer can miss
    # this path because it applies the transfer-curve estimate as one large
    # jump; near a cavity boundary that jump can lose the mode.  Walking SOA
    # while re-closing PHASE at every step preserves the wavelength branch and
    # leaves GAIN (and therefore the cavity gain condition) unchanged.  Joint
    # GAIN/SOA attenuation remains as a fallback.  Each tuple is the
    # GAIN/SOA share of one total mA decrement.
    attenuation_splits = (
        (0.0, 1.0), (0.75, 0.25), (1.0, 0.0), (0.60, 0.40),
        (0.50, 0.50), (0.35, 0.65),
    )
    step_total_ma = 3.0
    minimum_gain_ma = 55.0
    minimum_soa_ma = 85.0
    source_gain_ma = code_to_current(source_codes[0], 0)
    source_soa_ma = code_to_current(source_codes[1], 1)

    for split_index, (gain_share, soa_share) in enumerate(attenuation_splits):
        # Re-establish the point's own measured source branch before starting
        # each alternative current path.
        seed_codes, seed_reading, seed_ok = tune_at_currents(
            source_codes, source_gain_ma, source_soa_ma,
            f"continuation_split_{split_index}_source",
            allow_mirrors=needs_mirror_entry,
        )
        if candidate_score(seed_reading) < candidate_score(best_reading):
            best_codes, best_reading = list(seed_codes), seed_reading
        if not seed_ok:
            if needs_mirror_entry:
                break
            continue
        seed_power_error = (seed_reading.power_mw - target_power_mw) / target_power_mw
        if abs(seed_power_error) <= power_relative_tolerance:
            best_codes, best_reading = list(seed_codes), seed_reading
            break

        previous_alpha = 0.0
        previous_codes = list(seed_codes)
        previous_reading = seed_reading
        max_alpha_gain = (
            (source_gain_ma - minimum_gain_ma) / gain_share
            if gain_share > 0.0 else float("inf")
        )
        max_alpha_soa = (
            (source_soa_ma - minimum_soa_ma) / soa_share
            if soa_share > 0.0 else float("inf")
        )
        max_alpha = min(max_alpha_gain, max_alpha_soa)
        step_ma = step_total_ma
        alpha = step_ma
        steps_taken = 0

        while alpha <= max_alpha + 1e-9 and steps_taken < 40:
            steps_taken += 1
            gain_ma = source_gain_ma - alpha * gain_share
            soa_ma = source_soa_ma - alpha * soa_share
            codes, reading, wave_ok = tune_at_currents(
                previous_codes, gain_ma, soa_ma,
                f"continuation_split_{split_index}_walk",
            )
            if wave_ok and candidate_score(reading) < candidate_score(best_reading):
                best_codes, best_reading = list(codes), reading
            if not wave_ok:
                # Reduce the forward step, with a hard floor and a per-split
                # step cap so a mode boundary cannot create an asymptotic loop.
                step_ma *= 0.5
                if step_ma >= 0.75:
                    alpha = previous_alpha + step_ma
                    continue
                break

            relative_error = (reading.power_mw - target_power_mw) / target_power_mw
            if abs(relative_error) <= power_relative_tolerance:
                best_codes, best_reading = list(codes), reading
                break
            if reading.power_mw < target_power_mw <= previous_reading.power_mw:
                low_alpha, low_codes, low_reading = alpha, list(codes), reading
                high_alpha = previous_alpha
                high_codes, high_reading = list(previous_codes), previous_reading
                for refine_index in range(8):
                    mid_alpha = 0.5 * (high_alpha + low_alpha)
                    mid_gain = source_gain_ma - mid_alpha * gain_share
                    mid_soa = source_soa_ma - mid_alpha * soa_share
                    nearer_codes = high_codes if (
                        mid_alpha - high_alpha <= low_alpha - mid_alpha
                    ) else low_codes
                    mid_codes, mid_reading, mid_ok = tune_at_currents(
                        nearer_codes, mid_gain, mid_soa,
                        f"continuation_split_{split_index}_refine_{refine_index}",
                    )
                    if not mid_ok:
                        break
                    if candidate_score(mid_reading) < candidate_score(best_reading):
                        best_codes, best_reading = list(mid_codes), mid_reading
                    mid_error = (mid_reading.power_mw - target_power_mw) / target_power_mw
                    if abs(mid_error) <= power_relative_tolerance:
                        best_codes, best_reading = list(mid_codes), mid_reading
                        break
                    if mid_reading.power_mw >= target_power_mw:
                        high_alpha, high_codes, high_reading = (
                            mid_alpha, list(mid_codes), mid_reading
                        )
                    else:
                        low_alpha, low_codes, low_reading = (
                            mid_alpha, list(mid_codes), mid_reading
                        )
                break
            previous_alpha = alpha
            previous_codes = list(codes)
            previous_reading = reading
            if step_ma < step_total_ma:
                step_ma = min(step_total_ma, step_ma * 1.5)
            alpha += step_ma

        best_wave_ok = (
            abs(best_reading.wavelength_nm - target_nm) * 1000.0
            <= WAVELENGTH_TOLERANCE_PM and mode_ok(best_reading)
        )
        best_power_error = (
            (best_reading.power_mw - target_power_mw) / target_power_mw
        )
        if best_wave_ok and abs(best_power_error) <= power_relative_tolerance:
            break

    wavelength_error_pm = (best_reading.wavelength_nm - target_nm) * 1000.0
    relative_error = (best_reading.power_mw - target_power_mw) / target_power_mw
    success = (
        abs(wavelength_error_pm) <= WAVELENGTH_TOLERANCE_PM
        and mode_ok(best_reading)
        and abs(relative_error) <= power_relative_tolerance
    )
    return {
        "success": success, "target_nm": target_nm, "codes": best_codes,
        "currents_ma": [code_to_current(value, index)
                        for index, value in enumerate(best_codes)],
        "reading": asdict(best_reading), "wavelength_error_pm": wavelength_error_pm,
        "power_error_mw": best_reading.power_mw - target_power_mw,
        "power_relative_error": relative_error, "history": history,
    }


def close_equal_power_by_gain_soa_redistribution(
    laser: LaserSerial,
    meter: AQ6150B,
    source_row: dict[str, Any],
    target_nm: float,
    target_power_mw: float,
    settle_s: float,
    power_relative_tolerance: float,
) -> dict[str, Any]:
    """Shift a neighboring branch edge with GAIN, then close power with SOA."""
    source_codes = [int(value) for value in source_row["codes"]]
    source_gain = code_to_current(source_codes[0], 0)
    source_soa = code_to_current(source_codes[1], 1)
    source_reading = Reading(**source_row["reading"])
    best_codes = list(source_codes)
    best_reading = source_reading
    history: list[dict[str, Any]] = []

    def wave_ok(reading: Reading) -> bool:
        return (
            abs(reading.wavelength_nm - target_nm) * 1000.0
            <= WAVELENGTH_TOLERANCE_PM and mode_ok(reading)
        )

    def score(reading: Reading) -> tuple[float, float]:
        penalty = 0.0 if wave_ok(reading) else (
            1000.0 + abs(reading.wavelength_nm - target_nm) * 1000.0
        )
        return penalty, abs(reading.power_mw - target_power_mw) / target_power_mw

    def tune(codes: list[int], label: str) -> tuple[list[int], Reading, bool]:
        tuned = tune_phase_on_branch(
            laser, meter, codes, target_nm, settle_s, max_measurements=8,
        )
        tuned_codes = [int(value) for value in tuned["codes"]]
        reading = Reading(**tuned["reading"])
        ok = wave_ok(reading)
        history.append({
            "action": label, "success": ok, "codes": tuned_codes,
            "reading": asdict(reading), "measurements": tuned["history"],
        })
        print(
            f"redistribution-step {label} target={target_nm:.6f} "
            f"GAIN={code_to_current(tuned_codes[0], 0):.3f} "
            f"SOA={code_to_current(tuned_codes[1], 1):.3f} "
            f"err={(reading.wavelength_nm - target_nm) * 1000.0:+.2f}pm "
            f"power={reading.power_mw:.6f}mW wave_ok={ok}",
            flush=True,
        )
        return tuned_codes, reading, ok

    # Rebalancing the two carrier-current controls moves the longitudinal-mode
    # edge without touching the Vernier mirrors.  A 2.5 mA grid is fine enough
    # to find the accessible regime and is bounded to fourteen live attempts.
    gain_offsets = (0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0,
                    17.5, 20.0, 22.5, 25.0, 27.5, 30.0, 35.0)
    for gain_offset in gain_offsets:
        gain_ma = source_gain - gain_offset
        if gain_ma < 85.0:
            break
        entry_codes = list(source_codes)
        entry_codes[0] = current_to_code(gain_ma, 0)
        entry_codes[1] = current_to_code(source_soa, 1)
        entry_codes, entry_reading, entry_ok = tune(
            entry_codes, f"gain_entry_{gain_offset:.1f}mA",
        )
        if score(entry_reading) < score(best_reading):
            best_codes, best_reading = list(entry_codes), entry_reading
        if not entry_ok:
            continue
        entry_error = (entry_reading.power_mw - target_power_mw) / target_power_mw
        if abs(entry_error) <= power_relative_tolerance:
            best_codes, best_reading = list(entry_codes), entry_reading
            break
        if entry_reading.power_mw < target_power_mw:
            continue

        high_soa = source_soa
        high_codes = list(entry_codes)
        high_reading = entry_reading
        low_soa: float | None = None
        low_codes: list[int] | None = None
        low_reading: Reading | None = None
        current_codes = list(entry_codes)
        # The measured transfer slope is roughly 5--20 mW/mA here.  One-mA
        # steps preserve the branch and normally reach the target in a few
        # measurements.
        for soa_step in range(1, 13):
            soa_ma = source_soa - float(soa_step)
            if soa_ma < 85.0:
                break
            trial_codes = list(current_codes)
            trial_codes[1] = current_to_code(soa_ma, 1)
            trial_codes, trial_reading, trial_ok = tune(
                trial_codes, f"soa_close_{soa_step}",
            )
            if score(trial_reading) < score(best_reading):
                best_codes, best_reading = list(trial_codes), trial_reading
            if not trial_ok:
                break
            trial_error = (trial_reading.power_mw - target_power_mw) / target_power_mw
            if abs(trial_error) <= power_relative_tolerance:
                best_codes, best_reading = list(trial_codes), trial_reading
                break
            if trial_reading.power_mw >= target_power_mw:
                high_soa = soa_ma
                high_codes, high_reading = list(trial_codes), trial_reading
                current_codes = list(trial_codes)
                continue
            low_soa = soa_ma
            low_codes, low_reading = list(trial_codes), trial_reading
            break

        current_error = (best_reading.power_mw - target_power_mw) / target_power_mw
        if wave_ok(best_reading) and abs(current_error) <= power_relative_tolerance:
            break
        if low_soa is not None and low_codes is not None and low_reading is not None:
            for refine_index in range(6):
                mid_soa = 0.5 * (high_soa + low_soa)
                seed = high_codes if mid_soa >= 0.5 * (high_soa + low_soa) else low_codes
                mid_codes = list(seed)
                mid_codes[1] = current_to_code(mid_soa, 1)
                mid_codes, mid_reading, mid_ok = tune(
                    mid_codes, f"soa_refine_{refine_index}",
                )
                if score(mid_reading) < score(best_reading):
                    best_codes, best_reading = list(mid_codes), mid_reading
                if not mid_ok:
                    break
                mid_error = (mid_reading.power_mw - target_power_mw) / target_power_mw
                if abs(mid_error) <= power_relative_tolerance:
                    best_codes, best_reading = list(mid_codes), mid_reading
                    break
                if mid_reading.power_mw >= target_power_mw:
                    high_soa, high_codes, high_reading = (
                        mid_soa, list(mid_codes), mid_reading
                    )
                else:
                    low_soa, low_codes, low_reading = (
                        mid_soa, list(mid_codes), mid_reading
                    )
        current_error = (best_reading.power_mw - target_power_mw) / target_power_mw
        if wave_ok(best_reading) and abs(current_error) <= power_relative_tolerance:
            break

    wavelength_error_pm = (best_reading.wavelength_nm - target_nm) * 1000.0
    relative_error = (best_reading.power_mw - target_power_mw) / target_power_mw
    success = (
        abs(wavelength_error_pm) <= WAVELENGTH_TOLERANCE_PM
        and mode_ok(best_reading)
        and abs(relative_error) <= power_relative_tolerance
    )
    return {
        "success": success, "target_nm": target_nm, "codes": best_codes,
        "currents_ma": [code_to_current(value, index)
                        for index, value in enumerate(best_codes)],
        "reading": asdict(best_reading), "wavelength_error_pm": wavelength_error_pm,
        "power_error_mw": best_reading.power_mw - target_power_mw,
        "power_relative_error": relative_error, "history": history,
    }


def reclassify_resumed_release_rows(
    rows_by_key: dict[str, Any],
    target_power: float,
    power_relative_tolerance: float,
) -> int:
    """Release rows only when their retained reading matches current codes.

    A prediction-only recenter changes SOA but deliberately retains the old
    reading for audit.  Such a row must be measured again before it can pass.
    """

    reclassified = 0
    for key, current in rows_by_key.items():
        if current.get("success"):
            continue
        if current.get("certification_power_prediction_unverified"):
            continue
        if current.get("certification_guard_band_rework_required"):
            continue
        reading = current.get("reading")
        if not isinstance(reading, dict):
            continue
        try:
            target_nm = float(key)
            wavelength_nm = float(reading["wavelength_nm"])
            power_mw = float(reading["power_mw"])
            relative_error = (power_mw - target_power) / target_power
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
        mode_pass = normalized_smsr(reading) >= MIN_SMSR_DB
        wavelength_error_pm = (wavelength_nm - target_nm) * 1000.0
        if not (
            abs(wavelength_error_pm) <= WAVELENGTH_TOLERANCE_PM
            and mode_pass
            and abs(relative_error) <= power_relative_tolerance
        ):
            continue
        current["success"] = True
        current["wavelength_error_pm"] = wavelength_error_pm
        current["power_error_mw"] = power_mw - target_power
        current["power_relative_error"] = relative_error
        current.setdefault("repair_attempts", []).append({
            "strategy": "resume_existing_live_reading_release_gate_v1",
            "power_relative_tolerance": float(power_relative_tolerance),
            "requires_full_bidirectional_recertification": True,
        })
        reclassified += 1
    return reclassified


def expand_certification_guard_band_rows(
    payload: dict[str, Any],
    validation_path: Path,
    release_power_relative_tolerance: float,
) -> dict[str, int]:
    """Invalidate near-boundary rows so a repair subprocess recenters them.

    The long-running parent process may predate the guard-band selection code.
    Performing the same expansion in the freshly spawned repair subprocess
    makes the improvement effective without interrupting an active 2001-point
    certification sweep.  Expansion is recorded once and therefore remains
    resume-safe.
    """

    existing = payload.get("certification_guard_band_expansion")
    if isinstance(existing, dict):
        return {
            "release_failed": int(existing.get("release_failed", 0)),
            "guard_band_added": int(existing.get("guard_band_added", 0)),
            "total_rework_targets": int(existing.get("total_rework_targets", 0)),
            "sequence_deferred_predictions": int(
                existing.get("sequence_deferred_predictions", 0)
            ),
        }
    rows = payload.get("rows", {})
    if not validation_path.exists():
        return {
            "release_failed": 0,
            "guard_band_added": 0,
            "total_rework_targets": 0,
            "sequence_deferred_predictions": 0,
        }
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    checked_rows = validation.get("rows", {})
    if len(rows) != 2001 or len(checked_rows) != 2001:
        return {
            "release_failed": 0,
            "guard_band_added": 0,
            "total_rework_targets": 0,
            "sequence_deferred_predictions": 0,
        }

    power_guard = min(
        float(release_power_relative_tolerance),
        CERTIFICATION_REWORK_POWER_GUARD_RELATIVE,
    )
    release_failed = 0
    guard_band_added = 0
    sequence_deferred_predictions = 0
    sequence_deferred_targets: list[str] = []
    targets: list[str] = []
    for key, checked in checked_rows.items():
        release_failure = not bool(checked.get("success"))
        try:
            signed_wavelength_error_pm = float(checked["wavelength_error_pm"])
            wavelength_error_pm = abs(signed_wavelength_error_pm)
            power_relative_error = abs(float(checked["power_relative_error"]))
        except (KeyError, TypeError, ValueError):
            signed_wavelength_error_pm = math.inf
            wavelength_error_pm = math.inf
            power_relative_error = math.inf
        near_boundary = (
            wavelength_error_pm > CERTIFICATION_REWORK_WAVELENGTH_GUARD_PM
            or power_relative_error > power_guard
        )
        if not release_failure and not near_boundary:
            continue
        current = rows.get(str(key))
        if not isinstance(current, dict):
            continue
        if (
            math.isfinite(signed_wavelength_error_pm)
            and wavelength_error_pm > CERTIFICATION_REWORK_WAVELENGTH_GUARD_PM
        ):
            current["certification_sequence_wavelength_offset_pm"] = max(
                min(-signed_wavelength_error_pm, 1.0), -1.0,
            )
        if (
            math.isfinite(signed_wavelength_error_pm)
            and wavelength_error_pm >= SEQUENTIAL_MODE_HOP_THRESHOLD_PM
        ):
            current["certification_forward_transition_confirmation_required"] = True
            current["certification_forward_transition_failure"] = {
                "observed_wavelength_error_pm": signed_wavelength_error_pm,
                "observed_wavelength_nm": checked.get("reading", {}).get(
                    "wavelength_nm"
                ),
                "validation_source": validation_path.name,
            }
        if release_failure:
            release_failed += 1
        sample_count = max(int(checked.get("sample_count", 1)), 1)
        reading = checked.get("reading", {})
        measured_smsr = normalized_smsr(reading) if isinstance(reading, dict) else -math.inf
        mode_pass_count = int(
            checked.get(
                "mode_pass_count",
                1 if measured_smsr >= MIN_SMSR_DB else 0,
            )
        )
        mode_ok = (
            mode_pass_count >= sample_count // 2 + 1
            and measured_smsr >= MIN_SMSR_DB
        )
        # A full forward sweep observes the exact thermal/history path that the
        # released table will use.  When that sweep produced a pure power
        # failure, the parent has already predicted a new SOA code from the
        # measured error.  Re-closing that prediction in this sparse repair
        # traversal can tune it back to the wrong path-dependent value.  Keep
        # the prediction untouched and require the next complete forward sweep
        # to verify it.  Wavelength and mode failures still receive live repair.
        sequence_power_prediction = (
            bool(current.get("certification_power_prediction_unverified"))
            and math.isfinite(wavelength_error_pm)
            and wavelength_error_pm <= CERTIFICATION_REWORK_WAVELENGTH_GUARD_PM
            and mode_ok
        )
        if sequence_power_prediction:
            current["success"] = True
            current["certification_sequence_context_validation_pending"] = True
            current.setdefault("certification_rework", []).append({
                "strategy": "forward_sequence_power_prediction_deferred_to_full_sweep_v1",
                "validation_source": validation_path.name,
                "wavelength_error_pm": checked.get("wavelength_error_pm"),
                "power_relative_error": checked.get("power_relative_error"),
                "reason": "preserve full-forward-sweep SOA correction",
            })
            sequence_deferred_predictions += 1
            sequence_deferred_targets.append(str(key))
            continue
        if not release_failure and current.get("success"):
            guard_band_added += 1
        current["success"] = False
        current["certification_guard_band_rework_required"] = True
        current.setdefault("certification_rework", []).append({
            "strategy": "proactive_inner_guard_band_reclosure_v1",
            "validation_source": validation_path.name,
            "wavelength_error_pm": checked.get("wavelength_error_pm"),
            "power_relative_error": checked.get("power_relative_error"),
            "wavelength_guard_pm": CERTIFICATION_REWORK_WAVELENGTH_GUARD_PM,
            "power_guard_relative": power_guard,
        })
        targets.append(str(key))

    summary = {
        "release_failed": release_failed,
        "guard_band_added": guard_band_added,
        "total_rework_targets": len(targets),
        "sequence_deferred_predictions": sequence_deferred_predictions,
    }
    payload["certification_guard_band_expansion"] = {
        **summary,
        "created": utc_now(),
        "validation_source": validation_path.name,
        "wavelength_guard_pm": CERTIFICATION_REWORK_WAVELENGTH_GUARD_PM,
        "power_guard_relative": power_guard,
        "targets": sorted(targets, key=float),
        "sequence_deferred_targets": sorted(sequence_deferred_targets, key=float),
    }
    return summary


def confirm_forward_transition_candidate(
    laser: LaserSerial,
    meter: AQ6150B,
    predecessor_rows: list[dict[str, Any]],
    result: dict[str, Any],
    target_nm: float,
    target_power_mw: float,
    settle_s: float,
    wavelength_tolerance_pm: float,
    power_relative_tolerance: float,
) -> dict[str, Any]:
    """Confirm a repaired row from the same forward history used at runtime.

    A sparse repair probe can enter the requested mode from the calibrator's
    parked state even though the identical DAC codes mode-hop when approached
    from the preceding grid row.  Replay up to two predecessor rows before
    every confirmation sample so a sequential mode hop cannot be promoted as
    a repaired point.
    """

    checked = dict(result)
    samples: list[Reading] = []
    predecessor_targets = [
        float(predecessor["target_nm"])
        for predecessor in predecessor_rows[-2:]
    ]
    for _ in range(CALIBRATION_CONFIRMATION_SAMPLES):
        for predecessor in predecessor_rows[-2:]:
            predecessor_target = float(predecessor["target_nm"])
            probe(
                laser, meter,
                [int(value) for value in predecessor["codes"]],
                predecessor_target, settle_s,
            )
        _, reading = probe(
            laser, meter, [int(value) for value in checked["codes"]],
            target_nm, settle_s,
        )
        samples.append(reading)

    reading, mode_pass_count = aggregate_repeatability_readings(samples, target_nm)
    wavelength_error_pm = (reading.wavelength_nm - target_nm) * 1000.0
    relative_error = (reading.power_mw - target_power_mw) / target_power_mw
    success = (
        abs(wavelength_error_pm) <= float(wavelength_tolerance_pm)
        and mode_ok(reading)
        and mode_pass_count >= CALIBRATION_CONFIRMATION_SAMPLES // 2 + 1
        and abs(relative_error) <= float(power_relative_tolerance)
    )
    checked.update({
        "success": success,
        "reading": asdict(reading),
        "wavelength_error_pm": wavelength_error_pm,
        "power_error_mw": reading.power_mw - target_power_mw,
        "power_relative_error": relative_error,
        "forward_transition_confirmation": {
            "predecessor_targets_nm": predecessor_targets,
            "samples": [asdict(sample) for sample in samples],
            "mode_pass_count": mode_pass_count,
            "wavelength_tolerance_pm": float(wavelength_tolerance_pm),
            "power_relative_tolerance": float(power_relative_tolerance),
            "success": success,
        },
    })
    return checked


REPAIR_PRIOR_ROUND_FIELDS = (
    "history",
    "original_failure",
    "repair_attempts",
    "own_seed_repair_attempts",
    "neighbor_repair_attempts",
    "alternate_branch_attempts",
    "redistribution_attempts",
    "phase_candidate_branch_attempts",
    "inventory_branch_attempts",
    "current_continuation_attempted",
    "current_continuation_result",
)


def compact_new_repair_input(payload: dict[str, Any], source_artifact: Path) -> int:
    """Avoid duplicating the complete prior-round audit in every checkpoint.

    The immutable source artifact remains on disk and is recorded here.  Core
    codes/readings and the historical 145 mA ceiling seed stay inline; only
    bulky histories and attempt markers from a finished earlier repair round
    are removed so this new round can retry every newly failed row.
    """

    rows = payload.get("rows")
    if not isinstance(rows, dict):
        raise ValueError("repair input rows must be a mapping")
    compacted = 0
    flattened_seeds = 0
    for row in rows.values():
        if not isinstance(row, dict):
            continue
        if not isinstance(row.get("source_max_power_row"), dict):
            ancestor = row.get("original_failure")
            for _depth in range(16):
                if not isinstance(ancestor, dict):
                    break
                source_max = ancestor.get("source_max_power_row")
                if isinstance(source_max, dict):
                    row["source_max_power_row"] = source_max
                    flattened_seeds += 1
                    break
                ancestor = ancestor.get("original_failure")
        removed = False
        for field in REPAIR_PRIOR_ROUND_FIELDS:
            if field in row:
                row.pop(field, None)
                removed = True
        compacted += int(removed)
    payload["repair_parent_artifact"] = str(source_artifact.resolve())
    payload["repair_input_compaction"] = {
        "created": utc_now(),
        "rows_with_prior_round_fields_removed": compacted,
        "flattened_source_max_power_rows": flattened_seeds,
        "fields_not_duplicated": list(REPAIR_PRIOR_ROUND_FIELDS),
        "audit_remains_in_parent_artifact": True,
    }
    return compacted


def run_repair_equal_power(args: argparse.Namespace) -> dict[str, Any]:
    resumed_output = bool(args.resume and args.output.exists())
    if resumed_output:
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        expected_limits = dict(zip(CONTROL_NAMES, LASER_MAX_MA))
        if payload.get("limits_ma") != expected_limits:
            raise ValueError("resume output belongs to a different laser-current profile")
        if not math.isclose(
            float(payload.get("target_power_mw", float("nan"))),
            float(args.target_power_mw),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("resume output belongs to a different common power target")
    else:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        payload["created"] = utc_now()
        payload["method"] = payload.get("method", "") + "+bidirectional_neighbor_branch_repair"
        # Be defensive when a seed was produced by an older long-running
        # parent: round-specific expansion metadata must never survive into a
        # newly created repair artifact.
        payload.pop("certification_guard_band_expansion", None)
        payload.pop("repair_guard_band_targets", None)
        compact_new_repair_input(payload, args.input)
    curve, pilot = load_soa_transfer_curve(args.soa_pilot)
    rows_by_key = payload["rows"]
    ordered_keys = sorted(rows_by_key, key=float)
    source = json.loads(args.source_table.read_text(encoding="utf-8"))
    source_rows = source["rows"]
    target_power = float(payload.get("target_power_mw", args.target_power_mw))
    certification_round = payload.get("certification_rework_round")
    if certification_round is not None:
        validation_path = args.input.parent / (
            f"validation_forward_round{int(certification_round)}_avg3_postcondition.json"
        )
        expansion = expand_certification_guard_band_rows(
            payload, validation_path, args.power_relative_tolerance,
        )
        if expansion["total_rework_targets"]:
            print(
                "guard-band expansion "
                f"release_failed={expansion['release_failed']} "
                f"added={expansion['guard_band_added']} "
                f"total={expansion['total_rework_targets']}",
                flush=True,
            )
    requested_repair_tolerance = float(getattr(
        args, "repair_power_target_relative", REPAIR_POWER_TARGET_RELATIVE,
    ))
    if not 0.0 < requested_repair_tolerance <= float(args.power_relative_tolerance):
        raise ValueError(
            "repair power target tolerance must be positive and no wider than "
            "the release power tolerance"
        )
    repair_power_target_relative = min(
        float(args.power_relative_tolerance), requested_repair_tolerance,
    )
    repair_wavelength_target_pm = min(
        WAVELENGTH_TOLERANCE_PM, REPAIR_WAVELENGTH_TARGET_PM,
    )
    payload["repair_guard_band_targets"] = {
        "wavelength_tolerance_pm": repair_wavelength_target_pm,
        "power_relative_tolerance": repair_power_target_relative,
        "release_wavelength_tolerance_pm": WAVELENGTH_TOLERANCE_PM,
        "release_power_relative_tolerance": float(args.power_relative_tolerance),
    }

    # A strict certification rework may have been run previously with a
    # deliberately tighter power margin (for example +/-0.2%).  When a resume
    # is intentionally launched with the release tolerance (+/-0.5%), reuse
    # only rows backed by an existing live reading that already satisfies all
    # release gates.  The following full forward/reverse certification still
    # remeasures every point, so this avoids redundant exhaustive branch walks
    # without weakening the published acceptance test.
    reclassified = 0
    if resumed_output:
        reclassified = reclassify_resumed_release_rows(
            rows_by_key, target_power, args.power_relative_tolerance,
        )
        if reclassified:
            payload.setdefault("resume_release_gate_reclassification", []).append({
                "created": utc_now(),
                "rows": reclassified,
                "power_relative_tolerance": float(args.power_relative_tolerance),
                "requires_full_bidirectional_recertification": True,
            })

    laser = open_fullband_laser(args.port)
    meter = AQ6150B(args.gpib, average_count=AQ_AVERAGE_COUNT)
    payload["meter"] = meter.identity
    payload["meter_profile"] = meter.measurement_profile
    last_codes: list[int] | None = None
    repaired = 0
    attempted = 0
    ordered_index = {key: index for index, key in enumerate(ordered_keys)}

    def confirm_sequence_if_required(
        key: str, current: dict[str, Any], result: dict[str, Any],
    ) -> dict[str, Any]:
        if not result.get("success"):
            return result
        guarded = dict(result)
        inner_wavelength_error_pm = float(
            guarded.get(
                "closure_wavelength_error_pm",
                guarded.get("wavelength_error_pm", math.inf),
            )
        )
        inner_power_error = float(
            guarded.get("power_relative_error", math.inf)
        )
        if (
            abs(inner_wavelength_error_pm) > repair_wavelength_target_pm
            or abs(inner_power_error) > repair_power_target_relative
        ):
            guarded["success"] = False
            guarded["repair_inner_guard_rejection"] = {
                "wavelength_error_pm": inner_wavelength_error_pm,
                "wavelength_tolerance_pm": repair_wavelength_target_pm,
                "power_relative_error": inner_power_error,
                "power_relative_tolerance": repair_power_target_relative,
            }
            return guarded
        if not current.get(
            "certification_forward_transition_confirmation_required"
        ):
            return guarded
        index = ordered_index[key]
        predecessor_rows = [
            rows_by_key[ordered_keys[position]]
            for position in range(max(0, index - 2), index)
        ]
        if not predecessor_rows:
            rejected = dict(guarded)
            rejected["success"] = False
            rejected["forward_transition_confirmation"] = {
                "success": False,
                "reason": "no_forward_predecessor_available",
            }
            return rejected
        return confirm_forward_transition_candidate(
            laser, meter, predecessor_rows, guarded, float(key), target_power,
            args.settle_s, repair_wavelength_target_pm,
            repair_power_target_relative,
        )

    try:
        # Certification failures already have the correct branch and a useful
        # per-point code seed.  Re-close that same row first; copying a neighbor
        # is slower and can cross a mode boundary unnecessarily.
        own_seed_strategy = "same_row_power_and_wavelength_reclosure_v1"
        for key in ordered_keys:
            current = rows_by_key[key]
            if current.get("success"):
                continue
            if any(
                attempt.get("strategy") == own_seed_strategy
                for attempt in current.get("own_seed_repair_attempts", [])
            ):
                continue
            target_nm = float(key)
            baseline = float(source_rows[key]["reading"]["power_mw"])
            result = close_equal_power_from_seed(
                laser, meter, [int(value) for value in current["codes"]],
                target_nm, target_power, baseline, curve, args.settle_s,
                args.power_relative_tolerance, args.max_rounds,
                args.max_tune_measurements,
                closure_power_relative_tolerance=repair_power_target_relative,
                closure_wavelength_tolerance_pm=repair_wavelength_target_pm,
                closure_wavelength_offset_pm=float(
                    current.get("certification_sequence_wavelength_offset_pm", 0.0)
                ),
            )
            result = confirm_sequence_if_required(key, current, result)
            attempted += 1
            attempt = {"strategy": own_seed_strategy, "result": result}
            current.setdefault("own_seed_repair_attempts", []).append(attempt)
            last_codes = list(result["codes"])
            if result["success"]:
                original_failure = dict(current)
                rows_by_key[key] = {
                    **result,
                    "source": "same_row_certification_reclosure",
                    "original_failure": original_failure,
                }
                repaired += 1
            save_json(args.output, payload)
            print(
                f"own-seed target={target_nm:.6f} "
                f"err={result['wavelength_error_pm']:+.2f}pm "
                f"perr={result['power_relative_error'] * 100.0:+.2f}% "
                f"ok={result['success']}", flush=True,
            )

        for pass_index in range(args.passes):
            for direction in (1, -1):
                indices = range(len(ordered_keys)) if direction == 1 \
                    else range(len(ordered_keys) - 1, -1, -1)
                for index in indices:
                    key = ordered_keys[index]
                    current = rows_by_key[key]
                    if current.get("success"):
                        continue
                    direction_name = "forward" if direction == 1 else "reverse"
                    if any(
                        int(attempt.get("pass", -1)) == pass_index + 1
                        and attempt.get("direction") == direction_name
                        for attempt in current.get("neighbor_repair_attempts", [])
                    ):
                        continue
                    neighbor_index = index - direction
                    if not 0 <= neighbor_index < len(ordered_keys):
                        continue
                    neighbor = rows_by_key[ordered_keys[neighbor_index]]
                    if not neighbor.get("success"):
                        continue
                    target_nm = float(key)
                    baseline = float(source_rows[key]["reading"]["power_mw"])
                    result = close_equal_power_from_seed(
                        laser, meter, [int(value) for value in neighbor["codes"]],
                        target_nm, target_power, baseline, curve, args.settle_s,
                        args.power_relative_tolerance, args.max_rounds,
                        args.max_tune_measurements,
                        closure_power_relative_tolerance=repair_power_target_relative,
                        closure_wavelength_tolerance_pm=repair_wavelength_target_pm,
                        closure_wavelength_offset_pm=float(
                            current.get("certification_sequence_wavelength_offset_pm", 0.0)
                        ),
                    )
                    result = confirm_sequence_if_required(key, current, result)
                    attempted += 1
                    attempt = {
                        "pass": pass_index + 1,
                        "direction": direction_name,
                        "neighbor_target_nm": float(neighbor["target_nm"]),
                        "result": result,
                    }
                    if result["success"]:
                        original_failure = dict(current)
                        rows_by_key[key] = {
                            **result,
                            "source": "bidirectional_neighbor_branch_repair",
                            "neighbor_repair_attempts": [attempt],
                            "original_failure": original_failure,
                        }
                        repaired += 1
                        last_codes = list(result["codes"])
                    else:
                        current.setdefault("neighbor_repair_attempts", []).append(attempt)
                        last_codes = list(result["codes"])
                    save_json(args.output, payload)
                    print(
                        f"neighbor pass={pass_index + 1} "
                        f"dir={'F' if direction == 1 else 'R'} target={target_nm:.6f} "
                        f"err={result['wavelength_error_pm']:+.2f}pm "
                        f"perr={result['power_relative_error'] * 100.0:+.2f}% "
                        f"ok={result['success']}", flush=True,
                    )

        # At a source-branch power floor, changing the GAIN/SOA split cannot
        # reduce output far enough without a longitudinal-mode hop.  Nearby
        # dense-grid rows may belong to the next Vernier/PHASE branch and can
        # have a naturally lower 145/145 mA power.  Rank those measured source
        # branches by power proximity, tune them back to this wavelength, and
        # then use the same continuous current path for fine equalization.
        for key in ordered_keys:
            current = rows_by_key[key]
            if current.get("success"):
                continue
            target_nm = float(key)
            alternate_strategy = "nearby_branch_mirror_entry_v2"
            completed_sources = {
                str(attempt.get("source_target_key"))
                for attempt in current.get("alternate_branch_attempts", [])
                if attempt.get("strategy") == alternate_strategy
            }
            alternatives: list[tuple[float, float, str, dict[str, Any]]] = []
            for source_key, candidate in source_rows.items():
                if source_key == key or source_key in completed_sources:
                    continue
                distance_nm = abs(float(source_key) - target_nm)
                if distance_nm > 0.100001 or not candidate.get("success"):
                    continue
                power_mw = float(candidate["reading"]["power_mw"])
                if power_mw < target_power * (1.0 - args.power_relative_tolerance):
                    continue
                if power_mw > target_power * 1.10:
                    continue
                alternatives.append((
                    abs(power_mw - target_power), distance_nm,
                    source_key, candidate,
                ))
            alternatives.sort(key=lambda item: (item[0], item[1]))
            for _power_gap, _distance, source_key, candidate in alternatives[:4]:
                result = close_equal_power_by_current_continuation(
                    laser, meter, candidate, target_nm, target_power,
                    args.settle_s, args.power_relative_tolerance,
                    args.max_tune_measurements,
                )
                result = confirm_sequence_if_required(key, current, result)
                attempted += 1
                attempt = {
                    "strategy": alternate_strategy,
                    "source_target_key": source_key,
                    "source_power_mw": float(candidate["reading"]["power_mw"]),
                    "result": result,
                }
                current.setdefault("alternate_branch_attempts", []).append(attempt)
                last_codes = list(result["codes"])
                save_json(args.output, payload)
                print(
                    f"alternate-branch target={target_nm:.6f} "
                    f"source={float(source_key):.6f} "
                    f"err={result['wavelength_error_pm']:+.2f}pm "
                    f"perr={result['power_relative_error'] * 100.0:+.2f}% "
                    f"ok={result['success']}", flush=True,
                )
                if not result["success"]:
                    continue
                original_failure = dict(current)
                rows_by_key[key] = {
                    **result,
                    "source": "nearby_measured_branch_current_continuation",
                    "alternate_source_target_nm": float(source_key),
                    "original_failure": original_failure,
                }
                repaired += 1
                save_json(args.output, payload)
                break

        # A low-power neighboring branch may miss the target only because its
        # PHASE edge is a few picometres too high at 145/145 mA.  Reduce GAIN
        # first to move that edge, then close the common power with SOA.  This
        # is distinct from attenuating a branch that already reaches target.
        for key in ordered_keys:
            current = rows_by_key[key]
            if current.get("success"):
                continue
            target_nm = float(key)
            redistribution_strategy = "adjacent_branch_gain_soa_redistribution_v1"
            completed_sources = {
                str(attempt.get("source_target_key"))
                for attempt in current.get("redistribution_attempts", [])
                if attempt.get("strategy") == redistribution_strategy
            }
            candidates: list[tuple[float, float, str, dict[str, Any]]] = []
            for source_key, candidate in source_rows.items():
                if source_key == key or source_key in completed_sources:
                    continue
                distance_nm = abs(float(source_key) - target_nm)
                if distance_nm > 0.100001 or not candidate.get("success"):
                    continue
                power_mw = float(candidate["reading"]["power_mw"])
                if not (
                    target_power * (1.0 - args.power_relative_tolerance)
                    <= power_mw <= target_power * 1.10
                ):
                    continue
                candidates.append((distance_nm, abs(power_mw - target_power),
                                   source_key, candidate))
            candidates.sort(key=lambda item: (item[0], item[1]))
            for _distance, _power_gap, source_key, candidate in candidates[:4]:
                result = close_equal_power_by_gain_soa_redistribution(
                    laser, meter, candidate, target_nm, target_power,
                    args.settle_s, args.power_relative_tolerance,
                )
                result = confirm_sequence_if_required(key, current, result)
                attempted += 1
                attempt = {
                    "strategy": redistribution_strategy,
                    "source_target_key": source_key,
                    "source_power_mw": float(candidate["reading"]["power_mw"]),
                    "result": result,
                }
                current.setdefault("redistribution_attempts", []).append(attempt)
                last_codes = list(result["codes"])
                save_json(args.output, payload)
                print(
                    f"redistribution target={target_nm:.6f} "
                    f"source={float(source_key):.6f} "
                    f"err={result['wavelength_error_pm']:+.2f}pm "
                    f"perr={result['power_relative_error'] * 100.0:+.2f}% "
                    f"ok={result['success']}", flush=True,
                )
                if not result["success"]:
                    continue
                original_failure = dict(current)
                rows_by_key[key] = {
                    **result,
                    "source": "adjacent_branch_gain_soa_redistribution",
                    "alternate_source_target_nm": float(source_key),
                    "original_failure": original_failure,
                }
                repaired += 1
                save_json(args.output, payload)
                break

        # The dense source table retains only one maximum-power solution per
        # wavelength, while the phase-candidate artifact retains the other
        # measured mirror branches.  A branch whose 145/145 mA power is lower
        # can remain continuous all the way to the common target even when the
        # maximum-power branch hits a mode edge first.
        phase_candidate_rows: dict[str, Any] = {}
        phase_candidate_path = getattr(args, "phase_candidates", None)
        if phase_candidate_path is not None and phase_candidate_path.exists():
            phase_candidate_rows = json.loads(
                phase_candidate_path.read_text(encoding="utf-8")
            ).get("rows", {})
        for key in ordered_keys:
            current = rows_by_key[key]
            if current.get("success"):
                continue
            target_nm = float(key)
            completed_codes = {
                tuple(int(value) for value in attempt.get("codes", []))
                for attempt in current.get("phase_candidate_branch_attempts", [])
            }
            primary_codes = tuple(int(value) for value in source_rows[key]["codes"])
            candidates = phase_candidate_rows.get(key, {}).get("candidates", [])
            if current.get(
                "certification_forward_transition_confirmation_required"
            ):
                index = ordered_index[key]
                previous_codes = (
                    rows_by_key[ordered_keys[index - 1]].get("codes", ())
                    if index > 0 else ()
                )

                def transition_distance(candidate: dict[str, Any]) -> float:
                    codes = valid_codes(candidate.get("codes", ()))
                    if codes is None or len(previous_codes) != len(CONTROL_NAMES):
                        return math.inf
                    return sum(
                        abs(
                            code_to_current(codes[channel], channel)
                            - code_to_current(int(previous_codes[channel]), channel)
                        )
                        for channel in (2, 3, 4)
                    )

                candidates = sorted(candidates, key=transition_distance)
            for candidate in candidates[:8]:
                candidate_codes = valid_codes(candidate.get("codes", ()))
                if (
                    candidate_codes is None
                    or candidate_codes == primary_codes
                    or candidate_codes in completed_codes
                ):
                    continue
                tuned, wavelength_attempts = tune_wavelength_with_phase_fallback(
                    laser, meter, list(candidate_codes), target_nm, args.settle_s,
                    max_measurements=min(args.max_tune_measurements, 20),
                )
                attempted += 1
                tuned_reading = Reading(**tuned["reading"])
                if tuned.get("success"):
                    result = close_equal_power_by_current_continuation(
                        laser, meter, {
                            "target_nm": target_nm,
                            "codes": [int(value) for value in tuned["codes"]],
                            "reading": tuned["reading"],
                        }, target_nm, target_power, args.settle_s,
                        args.power_relative_tolerance, args.max_tune_measurements,
                    )
                else:
                    wavelength_error_pm = (
                        tuned_reading.wavelength_nm - target_nm
                    ) * 1000.0
                    result = {
                        "success": False,
                        "target_nm": target_nm,
                        "codes": [int(value) for value in tuned["codes"]],
                        "currents_ma": [
                            code_to_current(int(value), index)
                            for index, value in enumerate(tuned["codes"])
                        ],
                        "reading": tuned["reading"],
                        "wavelength_error_pm": wavelength_error_pm,
                        "power_error_mw": tuned_reading.power_mw - target_power,
                        "power_relative_error": (
                            tuned_reading.power_mw - target_power
                        ) / target_power,
                        "history": wavelength_attempts,
                    }
                result = confirm_sequence_if_required(key, current, result)
                attempt = {
                    "strategy": "measured_phase_candidate_continuation_v1",
                    "codes": list(candidate_codes),
                    "predicted_power_mw": candidate.get("predicted_power_mw"),
                    "wavelength_entry_attempts": wavelength_attempts,
                    "result": result,
                }
                current.setdefault("phase_candidate_branch_attempts", []).append(attempt)
                last_codes = list(result["codes"])
                save_json(args.output, payload)
                print(
                    f"phase-candidate target={target_nm:.6f} "
                    f"predicted_power={float(candidate.get('predicted_power_mw', float('nan'))):.6f} "
                    f"err={result['wavelength_error_pm']:+.2f}pm "
                    f"perr={result['power_relative_error'] * 100.0:+.2f}% "
                    f"ok={result['success']}", flush=True,
                )
                if not result["success"]:
                    continue
                original_failure = dict(current)
                rows_by_key[key] = {
                    **result,
                    "source": "measured_phase_candidate_current_continuation",
                    "phase_candidate_codes": list(candidate_codes),
                    "original_failure": original_failure,
                }
                repaired += 1
                save_json(args.output, payload)
                break

        # The dense source table retains only one maximum-power solution per
        # wavelength.  If both adjacent dense branches have a mode gap, fall
        # back to every real AQ6150B measurement collected in the inventory.
        # This exposes different WAVE-A/WAVE-B branches that deliberately were
        # not selected for the maximum-power table.
        inventory_candidates: list[dict[str, Any]] = []
        if args.inventory is not None and args.inventory.exists():
            inventory_payload = json.loads(args.inventory.read_text(encoding="utf-8"))
            inventory_candidates = list(inventory_payload.get("candidates", []))
        for key in ordered_keys:
            current = rows_by_key[key]
            if current.get("success") or not inventory_candidates:
                continue
            target_nm = float(key)
            inventory_strategy = "inventory_measured_branch_mirror_entry_v1"
            completed_codes = {
                tuple(int(value) for value in attempt.get("codes", []))
                for attempt in current.get("inventory_branch_attempts", [])
                if attempt.get("strategy") == inventory_strategy
            }
            ranked: list[tuple[float, float, tuple[int, ...], dict[str, Any]]] = []
            seen_codes: set[tuple[int, ...]] = set()
            for candidate in inventory_candidates:
                codes = valid_codes(candidate.get("codes", ()))
                if codes is None or codes in seen_codes or codes in completed_codes:
                    continue
                seen_codes.add(codes)
                wavelength_nm = float(candidate.get("wavelength_nm", float("nan")))
                power_mw = float(candidate.get("power_mw", float("nan")))
                smsr_db = float(candidate.get("smsr_db", float("nan")))
                if not all(math.isfinite(value) for value in (
                    wavelength_nm, power_mw, smsr_db,
                )):
                    continue
                distance_nm = abs(wavelength_nm - target_nm)
                if distance_nm > 0.300001 or smsr_db < MIN_SMSR_DB:
                    continue
                if power_mw < target_power * (1.0 - args.power_relative_tolerance):
                    continue
                ranked.append((
                    abs(power_mw - target_power), distance_nm,
                    codes, candidate,
                ))
            ranked.sort(key=lambda item: (item[0], item[1]))
            for _power_gap, _distance, codes, candidate in ranked[:8]:
                measured_wave = float(candidate["wavelength_nm"])
                measured_power = float(candidate["power_mw"])
                measured_smsr = float(candidate["smsr_db"])
                source_candidate = {
                    "target_nm": measured_wave,
                    "codes": list(codes),
                    "reading": {
                        "wavelength_nm": measured_wave,
                        "wavelength_error_pm": 0.0,
                        "power_mw": measured_power,
                        "power_dbm": 10.0 * math.log10(measured_power),
                        "peak_count": int(candidate.get("peak_count", 1)),
                        "secondary_power_mw": None,
                        "side_mode_suppression_db": measured_smsr,
                        "pdt_code": candidate.get("pdt_code"),
                        "pdr_code": candidate.get("pdr_code"),
                        "pdt_ma": None, "pdr_ma": None, "pdr_pdt_ratio": None,
                    },
                }
                result = close_equal_power_by_current_continuation(
                    laser, meter, source_candidate, target_nm, target_power,
                    args.settle_s, args.power_relative_tolerance,
                    args.max_tune_measurements,
                )
                result = confirm_sequence_if_required(key, current, result)
                attempted += 1
                attempt = {
                    "strategy": inventory_strategy,
                    "codes": list(codes),
                    "measured_source_wavelength_nm": measured_wave,
                    "measured_source_power_mw": measured_power,
                    "measurement_source": candidate.get("source"),
                    "result": result,
                }
                current.setdefault("inventory_branch_attempts", []).append(attempt)
                last_codes = list(result["codes"])
                save_json(args.output, payload)
                print(
                    f"inventory-branch target={target_nm:.6f} "
                    f"source_wave={measured_wave:.6f} source_power={measured_power:.6f} "
                    f"err={result['wavelength_error_pm']:+.2f}pm "
                    f"perr={result['power_relative_error'] * 100.0:+.2f}% "
                    f"ok={result['success']}", flush=True,
                )
                if not result["success"]:
                    continue
                original_failure = dict(current)
                rows_by_key[key] = {
                    **result,
                    "source": "inventory_measured_branch_current_continuation",
                    "inventory_measurement_source": candidate.get("source"),
                    "original_failure": original_failure,
                }
                repaired += 1
                save_json(args.output, payload)
                break

        # A grid point immediately before a mode boundary can be unreachable
        # after the SOA-only power jump from either adjacent wavelength.  Its
        # own validated 145/145 mA source branch is still available, so follow
        # that branch continuously while sharing attenuation across GAIN/SOA.
        continuation_strategy = "own_branch_current_continuation_v2_soa_first"
        for key in ordered_keys:
            current = rows_by_key[key]
            if current.get("success"):
                continue
            if current.get("current_continuation_strategy") == continuation_strategy:
                continue
            target_nm = float(key)
            result = close_equal_power_by_current_continuation(
                laser, meter, source_rows[key], target_nm, target_power,
                args.settle_s, args.power_relative_tolerance,
                args.max_tune_measurements,
            )
            result = confirm_sequence_if_required(key, current, result)
            attempted += 1
            current["current_continuation_attempted"] = True
            current["current_continuation_strategy"] = continuation_strategy
            current["current_continuation_result"] = result
            if result["success"]:
                original_failure = dict(current)
                rows_by_key[key] = {
                    **result,
                    "source": "own_branch_gain_soa_current_continuation",
                    "original_failure": original_failure,
                }
                repaired += 1
            last_codes = list(result["codes"])
            save_json(args.output, payload)
            print(
                f"continuation target={target_nm:.6f} "
                f"err={result['wavelength_error_pm']:+.2f}pm "
                f"perr={result['power_relative_error'] * 100.0:+.2f}% "
                f"GAIN={code_to_current(result['codes'][0], 0):.3f}mA "
                f"SOA={code_to_current(result['codes'][1], 1):.3f}mA "
                f"ok={result['success']}", flush=True,
            )
    finally:
        if last_codes is not None:
            park = list(last_codes)
            park[0] = current_to_code(60.0, 0)
            park[1] = current_to_code(90.0, 1)
            try:
                laser.set_codes(park)
            except Exception:
                pass
        meter.close()
        laser.close()

    rows = list(rows_by_key.values())
    passed = [row for row in rows if row.get("success")]
    powers = [float(row["reading"]["power_mw"]) for row in passed]
    payload["neighbor_repair_summary"] = {
        "tested": len(rows), "passed": len(passed), "failed": len(rows) - len(passed),
        "attempted": attempted, "repaired": repaired,
        "target_power_mw": target_power,
        "measured_power_min_mw": min(powers, default=None),
        "measured_power_median_mw": statistics.median(powers) if powers else None,
        "measured_power_max_mw": max(powers, default=None),
    }
    payload["soa_pilot_summary"] = pilot.get("summary", {})
    save_json(args.output, payload)
    return payload


def aggregate_repeatability_readings(
    readings: list[Reading], target_nm: float,
) -> tuple[Reading, int]:
    """Build a robust per-point reading while retaining mode-vote evidence."""
    if not readings:
        raise ValueError("at least one repeatability reading is required")

    def median_required(name: str) -> float:
        return float(statistics.median(float(getattr(row, name)) for row in readings))

    def median_optional(name: str) -> float | None:
        values = [float(value) for row in readings
                  if (value := getattr(row, name)) is not None]
        return float(statistics.median(values)) if values else None

    wavelength_nm = median_required("wavelength_nm")
    aggregate = Reading(
        wavelength_nm=wavelength_nm,
        wavelength_error_pm=(wavelength_nm - target_nm) * 1000.0,
        power_mw=median_required("power_mw"),
        power_dbm=median_required("power_dbm"),
        peak_count=int(round(median_required("peak_count"))),
        secondary_power_mw=median_optional("secondary_power_mw"),
        side_mode_suppression_db=median_optional("side_mode_suppression_db"),
        pdt_code=(int(round(value)) if (value := median_optional("pdt_code")) is not None else None),
        pdr_code=(int(round(value)) if (value := median_optional("pdr_code")) is not None else None),
        pdt_ma=median_optional("pdt_ma"),
        pdr_ma=median_optional("pdr_ma"),
        pdr_pdt_ratio=median_optional("pdr_pdt_ratio"),
    )
    return aggregate, sum(1 for row in readings if mode_ok(row))


def run_validate_equal_power(args: argparse.Namespace) -> dict[str, Any]:
    table = json.loads(args.input.read_text(encoding="utf-8"))
    if args.target_power_mw is None:
        args.target_power_mw = float(table["target_power_mw"])
    ordered = sorted(table["rows"].values(), key=lambda row: float(row["target_nm"]))
    if args.direction == "reverse":
        ordered.reverse()
    if args.resume and args.output.exists():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        expected_limits = dict(zip(CONTROL_NAMES, LASER_MAX_MA))
        if payload.get("source_table") != args.input.name:
            raise ValueError("resume output belongs to a different source table")
        if payload.get("direction") != args.direction:
            raise ValueError("resume output belongs to a different validation direction")
        if payload.get("limits_ma") != expected_limits:
            raise ValueError("resume output belongs to a different laser-current profile")
        if not math.isclose(
            float(payload.get("target_power_mw", float("nan"))),
            float(args.target_power_mw),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("resume output belongs to a different common power target")

        previous_tolerance = float(
            payload.get("power_relative_tolerance", POWER_REL_TOLERANCE)
        )
        if not math.isclose(
            previous_tolerance,
            float(args.power_relative_tolerance),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            changed = 0
            passed = 0
            required_mode_passes = args.samples_per_point // 2 + 1
            for key, row in payload.get("rows", {}).items():
                reading_payload = row.get("reading")
                if not isinstance(reading_payload, dict):
                    continue
                reading = Reading(**reading_payload)
                target_nm = float(row.get("target_nm", key))
                wavelength_error_pm = (
                    float(reading.wavelength_nm) - target_nm
                ) * 1000.0
                power_relative_error = (
                    float(reading.power_mw) - float(args.target_power_mw)
                ) / float(args.target_power_mw)
                samples = row.get("samples")
                mode_pass_count = row.get("mode_pass_count")
                if mode_pass_count is None and isinstance(samples, list):
                    mode_pass_count = sum(
                        mode_ok(Reading(**sample))
                        for sample in samples
                        if isinstance(sample, dict)
                    )
                mode_pass_count = int(mode_pass_count or 0)
                success = (
                    abs(wavelength_error_pm) <= WAVELENGTH_TOLERANCE_PM
                    and mode_ok(reading)
                    and mode_pass_count >= required_mode_passes
                    and abs(power_relative_error)
                    <= float(args.power_relative_tolerance)
                )
                changed += int(bool(row.get("success")) != success)
                passed += int(success)
                row["success"] = success
                row["wavelength_error_pm"] = wavelength_error_pm
                row["power_error_mw"] = (
                    float(reading.power_mw) - float(args.target_power_mw)
                )
                row["power_relative_error"] = power_relative_error
                row["mode_pass_count"] = mode_pass_count
            payload.setdefault("resume_tolerance_reclassification", []).append({
                "created": utc_now(),
                "old_power_relative_tolerance": previous_tolerance,
                "new_power_relative_tolerance": float(args.power_relative_tolerance),
                "rows_reclassified": len(payload.get("rows", {})),
                "success_flags_changed": changed,
                "passed_after_reclassification": passed,
                "raw_measurements_reused": True,
            })
            payload["power_relative_tolerance"] = float(
                args.power_relative_tolerance
            )
            save_json(args.output, payload)
    else:
        payload = {
            "created": utc_now(),
            "method": "independent_exact_code_robust_median_repeatability_pass",
            "source_table": args.input.name, "direction": args.direction,
            "target_power_mw": args.target_power_mw,
            "limits_ma": dict(zip(CONTROL_NAMES, LASER_MAX_MA)),
            "power_relative_tolerance": args.power_relative_tolerance,
            "wavelength_tolerance_pm": WAVELENGTH_TOLERANCE_PM,
            "single_mode_gate_smsr_db": MIN_SMSR_DB, "rows": {},
        }
    payload["samples_per_point"] = int(args.samples_per_point)
    payload["repeat_settle_s"] = float(args.repeat_settle_s)
    laser = open_fullband_laser(args.port)
    meter = AQ6150B(args.gpib, average_count=AQ_AVERAGE_COUNT)
    payload["meter"] = meter.identity
    payload["meter_profile"] = meter.measurement_profile
    last_codes: list[int] | None = None
    try:
        for index, calibrated in enumerate(ordered):
            target_nm = float(calibrated["target_nm"])
            key = f"{target_nm:.6f}"
            existing = payload["rows"].get(key)
            samples: list[dict[str, Any]] = []
            if existing is not None:
                samples = list(existing.get("samples", []))
                if not samples and existing.get("reading"):
                    # Upgrade an earlier one-reading checkpoint in place.
                    samples = [dict(existing["reading"])]
            if len(samples) >= args.samples_per_point:
                continue
            codes = [int(value) for value in calibrated["codes"]]
            while len(samples) < args.samples_per_point:
                repeat_settle = args.settle_s if not samples else args.repeat_settle_s
                _, sample = probe(laser, meter, codes, target_nm, repeat_settle)
                samples.append(asdict(sample))
            last_codes = list(codes)
            readings = [Reading(**sample) for sample in samples]
            reading, mode_pass_count = aggregate_repeatability_readings(
                readings, target_nm,
            )
            wavelength_error_pm = (reading.wavelength_nm - target_nm) * 1000.0
            power_relative_error = (
                (reading.power_mw - args.target_power_mw) / args.target_power_mw
            )
            success = (
                abs(wavelength_error_pm) <= WAVELENGTH_TOLERANCE_PM
                and mode_ok(reading)
                and mode_pass_count >= (args.samples_per_point // 2 + 1)
                and abs(power_relative_error) <= args.power_relative_tolerance
            )
            payload["rows"][key] = {
                "success": success, "target_nm": target_nm, "codes": codes,
                "currents_ma": calibrated["currents_ma"], "reading": asdict(reading),
                "wavelength_error_pm": wavelength_error_pm,
                "power_error_mw": reading.power_mw - args.target_power_mw,
                "power_relative_error": power_relative_error,
                "samples": samples, "sample_count": len(samples),
                "mode_pass_count": mode_pass_count,
                "calibration_reading": calibrated["reading"],
            }
            save_json(args.output, payload)
            if ((index + 1) % args.print_every == 0
                    or (not success and not args.quiet_failures) or index == 0):
                print(
                    f"validate {index + 1}/{len(ordered)} target={target_nm:.6f} "
                    f"err={wavelength_error_pm:+.2f}pm "
                    f"perr={power_relative_error * 100.0:+.2f}% ok={success}",
                    flush=True,
                )
    finally:
        if last_codes is not None:
            park = list(last_codes)
            park[0] = current_to_code(60.0, 0)
            park[1] = current_to_code(90.0, 1)
            try:
                laser.set_codes(park)
            except Exception:
                pass
        meter.close()
        laser.close()
    rows = list(payload["rows"].values())
    passed = [row for row in rows if row.get("success")]
    payload["summary"] = {
        "tested": len(rows), "passed": len(passed), "failed": len(rows) - len(passed),
        "target_power_mw": args.target_power_mw,
        "power_min_mw": min((float(row["reading"]["power_mw"]) for row in rows),
                            default=None),
        "power_median_mw": statistics.median(
            [float(row["reading"]["power_mw"]) for row in rows]) if rows else None,
        "power_max_mw": max((float(row["reading"]["power_mw"]) for row in rows),
                            default=None),
        "maximum_abs_wavelength_error_pm": max(
            (abs(float(row["wavelength_error_pm"])) for row in rows), default=None),
        "minimum_smsr_db": min(
            (normalized_smsr(row["reading"]) for row in rows), default=None),
    }
    save_json(args.output, payload)
    return payload


def run_dense(args: argparse.Namespace) -> dict[str, Any]:
    """Measure every dense wavelength while following local mode branches."""
    library = load_candidates(args.inventory)
    model_bundle = joblib.load(args.model) if args.model and args.model.exists() else None
    anchors_payload = json.loads(args.anchors.read_text(encoding="utf-8"))
    anchors = anchors_payload.get("rows", {})
    targets = target_grid(args.start_nm, args.stop_nm, args.step_nm)
    if args.resume and args.output.exists():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
    else:
        payload = {
            "created": utc_now(),
            "method": "measured_anchor_plus_local_phase_branch_following",
            "target_start_nm": args.start_nm,
            "target_stop_nm": args.stop_nm,
            "target_step_nm": args.step_nm,
            "target_count": len(targets),
            "limits_ma": dict(zip(CONTROL_NAMES, LASER_MAX_MA)),
            "wavelength_tolerance_pm": WAVELENGTH_TOLERANCE_PM,
            "single_mode_gate_smsr_db": MIN_SMSR_DB,
            "gain_soa_discovery_ma": MAX_GAIN_SOA_MA,
            "rows": {},
        }

    laser = open_fullband_laser(args.port)
    meter = AQ6150B(args.gpib, average_count=AQ_AVERAGE_COUNT)
    payload["meter"] = meter.identity
    previous: dict[str, Any] | None = None
    last_codes: list[int] | None = None
    try:
        for index, target in enumerate(targets):
            key = f"{target:.6f}"
            existing = payload["rows"].get(key)
            if existing and existing.get("success"):
                previous = existing
                last_codes = list(existing["codes"])
                rd = existing["reading"]
                library.append(Candidate(
                    codes=tuple(last_codes), wavelength_nm=float(rd["wavelength_nm"]),
                    power_mw=float(rd["power_mw"]), smsr_db=normalized_smsr(rd),
                    peak_count=int(rd["peak_count"]), pdt_code=rd.get("pdt_code"),
                    pdr_code=rd.get("pdr_code"), source=args.output.name,
                ))
                continue

            successful: list[dict[str, Any]] = []
            if previous is not None and abs(
                float(previous["target_nm"]) + args.step_nm - target
            ) < 1e-6:
                followed = fast_follow_branch(
                    laser, meter, previous, target, args.settle_s,
                    max_measurements=args.fast_measurements,
                )
                if followed is not None and followed["success"]:
                    successful.append(compact_row(followed, "phase_branch_follow"))

            anchor = anchors.get(key)
            if anchor and anchor.get("success"):
                anchor_attempt = tune_phase_on_branch(
                    laser, meter, at_discovery_power(anchor["codes"]), target,
                    args.settle_s, max_measurements=8,
                )
                if anchor_attempt["success"]:
                    successful.append(compact_row(anchor_attempt, "validated_integer_anchor"))

            if not successful:
                fallback = tune_target(
                    laser, meter, library, target, args.settle_s,
                    previous_codes=(list(previous["codes"]) if previous else None),
                    model_bundle=model_bundle,
                )
                if fallback["success"]:
                    successful.append(compact_row(fallback, "full_branch_recovery"))

            if successful:
                row = max(successful, key=lambda item: (
                    float(item["reading"]["power_mw"]),
                    normalized_smsr(item["reading"]),
                ))
                previous = row
                last_codes = list(row["codes"])
                rd = row["reading"]
                library.append(Candidate(
                    codes=tuple(last_codes), wavelength_nm=float(rd["wavelength_nm"]),
                    power_mw=float(rd["power_mw"]), smsr_db=normalized_smsr(rd),
                    peak_count=int(rd["peak_count"]), pdt_code=rd.get("pdt_code"),
                    pdr_code=rd.get("pdr_code"), source=args.output.name,
                ))
            else:
                row = {
                    "success": False, "target_nm": target,
                    "failure": "no_single_mode_branch_within_wavelength_tolerance",
                }
                previous = None
            payload["rows"][key] = row
            save_json(args.output, payload)
            if (index + 1) % args.print_every == 0 or not row["success"] or index == 0:
                if row["success"]:
                    rd = row["reading"]
                    print(
                        f"dense {index + 1}/{len(targets)} target={target:.6f} "
                        f"wave={rd['wavelength_nm']:.6f} "
                        f"err={(float(rd['wavelength_nm']) - target) * 1000.0:+.2f}pm "
                        f"power={rd['power_mw']:.4f} SMSR={normalized_smsr(rd):.1f} "
                        f"source={row['source']}", flush=True,
                    )
                else:
                    print(f"dense {index + 1}/{len(targets)} target={target:.6f} FAILED",
                          flush=True)
    finally:
        if last_codes is not None:
            park = list(last_codes)
            park[0] = current_to_code(60.0, 0)
            park[1] = current_to_code(90.0, 1)
            try:
                laser.set_codes(park)
            except Exception:
                pass
        meter.close()
        laser.close()

    rows = list(payload["rows"].values())
    passed = [row for row in rows if row.get("success")]
    payload["summary"] = {
        "tested": len(rows), "passed": len(passed),
        "failed": len(rows) - len(passed),
        "raw_power_min_mw": min(
            (float(row["reading"]["power_mw"]) for row in passed), default=None),
        "raw_power_max_mw": max(
            (float(row["reading"]["power_mw"]) for row in passed), default=None),
        "maximum_abs_wavelength_error_pm": max(
            (abs(float(row["reading"]["wavelength_nm"]) - float(row["target_nm"])) * 1000.0
             for row in passed), default=None),
        "minimum_smsr_db": min(
            (normalized_smsr(row["reading"]) for row in passed), default=None),
    }
    save_json(args.output, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    inventory = sub.add_parser("inventory", help="index prior live meter data")
    inventory.add_argument("--output", type=Path, default=DEFAULT_INVENTORY)

    train = sub.add_parser("train", help="train wavelength/power/mode/PDR/PDT models")
    train.add_argument("--output", type=Path, default=DEFAULT_MODEL)

    mode_map = sub.add_parser("map", help="measure WAVE-A/WAVE-B Vernier mode map")
    mode_map.add_argument("--port", default="COM6")
    mode_map.add_argument("--gpib", default="GPIB0::7::INSTR")
    mode_map.add_argument("--output", type=Path, default=DEFAULT_MODE_MAP)
    mode_map.add_argument("--gain-ma", type=float, default=MAX_GAIN_SOA_MA)
    mode_map.add_argument("--soa-ma", type=float, default=MAX_GAIN_SOA_MA)
    mode_map.add_argument("--phase-ma", type=float, default=5.0)
    mode_map.add_argument("--wave-step-ma", type=float, default=1.5)
    mode_map.add_argument("--wave-a-start-ma", type=float, default=0.0)
    mode_map.add_argument("--wave-a-stop-ma", type=float, default=30.0)
    mode_map.add_argument("--wave-b-start-ma", type=float, default=0.0)
    mode_map.add_argument("--wave-b-stop-ma", type=float, default=30.0)
    mode_map.add_argument("--settle-s", type=float, default=0.18)
    mode_map.add_argument("--print-every", type=int, default=20)
    mode_map.add_argument("--resume", action="store_true")

    phase_select = sub.add_parser(
        "select-phase-branches", help="select high-power WAVE-A/B branches for PHASE sweeps"
    )
    phase_select.add_argument(
        "--mode-maps", type=Path, nargs="+",
        help="explicit live mode maps; default keeps legacy auto-discovery",
    )
    phase_select.add_argument("--output", type=Path, default=DEFAULT_PHASE_BRANCHES)
    phase_select.add_argument("--start-nm", type=float, default=TARGET_START_NM)
    phase_select.add_argument("--stop-nm", type=float, default=TARGET_STOP_NM)
    phase_select.add_argument("--target-step-nm", type=float, default=0.20)
    phase_select.add_argument("--radius-nm", type=float, default=0.30)
    phase_select.add_argument("--branches-per-target", type=int, default=2)

    gap_select = sub.add_parser(
        "select-gap-phase-branches",
        help="select branches near targets left uncovered by a PHASE map",
    )
    gap_select.add_argument("--candidates", type=Path, default=DEFAULT_PHASE_CANDIDATES)
    gap_select.add_argument("--phase-map", type=Path,
                            default=HERE / "fullband_phase_sweep_map_1mA.json")
    gap_select.add_argument("--output", type=Path,
                            default=HERE / "fullband_phase_gap_branches.json")
    gap_select.add_argument("--radius-nm", type=float, default=0.60)
    gap_select.add_argument("--branches-per-target", type=int, default=2)

    alternate_select = sub.add_parser(
        "select-alternate-mode-branches",
        help="select unmeasured WAVE-A/B branches near still-uncovered targets",
    )
    alternate_select.add_argument("--candidates", type=Path,
                                  default=DEFAULT_PHASE_CANDIDATES)
    alternate_select.add_argument(
        "--exclude-phase-maps", type=Path, nargs="+",
        default=[HERE / "fullband_phase_sweep_map_1mA.json"],
    )
    alternate_select.add_argument(
        "--mode-maps", type=Path, nargs="+",
        help="explicit live mode maps; default keeps legacy auto-discovery",
    )
    alternate_select.add_argument(
        "--output", type=Path,
        default=HERE / "fullband_phase_alternate_branches.json",
    )
    alternate_select.add_argument("--radius-nm", type=float, default=1.0)
    alternate_select.add_argument("--branches-per-target", type=int, default=4)

    phase_map = sub.add_parser("phase-map", help="sweep PHASE on selected WAVE-A/B branches")
    phase_map.add_argument("--port", default="COM6")
    phase_map.add_argument("--gpib", default="GPIB0::7::INSTR")
    phase_map.add_argument("--branches", type=Path, default=DEFAULT_PHASE_BRANCHES)
    phase_map.add_argument("--output", type=Path, default=DEFAULT_PHASE_MAP)
    phase_map.add_argument("--gain-ma", type=float, default=MAX_GAIN_SOA_MA)
    phase_map.add_argument("--soa-ma", type=float, default=MAX_GAIN_SOA_MA)
    phase_map.add_argument("--phase-start-ma", type=float, default=0.0)
    phase_map.add_argument("--phase-stop-ma", type=float, default=10.0)
    phase_map.add_argument("--phase-step-ma", type=float, default=0.5)
    phase_map.add_argument("--settle-s", type=float, default=0.16)
    phase_map.add_argument("--print-every", type=int, default=50)
    phase_map.add_argument("--checkpoint-every", type=int, default=10)
    phase_map.add_argument("--resume", action="store_true")

    phase_candidates = sub.add_parser(
        "build-phase-candidates", help="interpolate only inside measured mode-hop-free PHASE segments"
    )
    phase_candidates.add_argument(
        "--phase-maps", type=Path, nargs="+",
        default=[HERE / "fullband_phase_sweep_map_1mA.json"],
    )
    phase_candidates.add_argument("--output", type=Path, default=DEFAULT_PHASE_CANDIDATES)
    phase_candidates.add_argument("--start-nm", type=float, default=TARGET_START_NM)
    phase_candidates.add_argument("--stop-nm", type=float, default=TARGET_STOP_NM)
    phase_candidates.add_argument("--step-nm", type=float, default=TARGET_STEP_NM)
    phase_candidates.add_argument("--jump-threshold-nm", type=float, default=0.35)
    phase_candidates.add_argument("--candidates-per-target", type=int, default=5)

    pilot = sub.add_parser("pilot", help="live sparse branch/power pilot")
    pilot.add_argument("--port", default="COM6")
    pilot.add_argument("--gpib", default="GPIB0::7::INSTR")
    pilot.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    pilot.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    pilot.add_argument("--output", type=Path, default=DEFAULT_PILOT)
    pilot.add_argument("--start-nm", type=float, default=TARGET_START_NM)
    pilot.add_argument("--stop-nm", type=float, default=TARGET_STOP_NM)
    pilot.add_argument("--step-nm", type=float, default=1.0)
    pilot.add_argument("--settle-s", type=float, default=0.32)
    pilot.add_argument("--resume", action="store_true")

    direct = sub.add_parser("direct", help="tune one explicitly selected branch")
    direct.add_argument("--port", default="COM6")
    direct.add_argument("--gpib", default="GPIB0::7::INSTR")
    direct.add_argument("--target-nm", type=float, required=True)
    direct.add_argument("--gain-ma", type=float, default=MAX_GAIN_SOA_MA)
    direct.add_argument("--soa-ma", type=float, default=MAX_GAIN_SOA_MA)
    direct.add_argument("--phase-ma", type=float, required=True)
    direct.add_argument("--wave-a-ma", type=float, required=True)
    direct.add_argument("--wave-b-ma", type=float, required=True)
    direct.add_argument("--settle-s", type=float, default=0.26)
    direct.add_argument("--max-measurements", type=int, default=30)
    direct.add_argument("--confirmations", type=int, default=2)
    direct.add_argument("--output", type=Path, required=True)
    direct.add_argument("--merge-pilot", type=Path)

    dense = sub.add_parser("dense", help="measure the complete dense wavelength grid")
    dense.add_argument("--port", default="COM6")
    dense.add_argument("--gpib", default="GPIB0::7::INSTR")
    dense.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    dense.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    dense.add_argument("--anchors", type=Path, default=DEFAULT_PILOT)
    dense.add_argument("--output", type=Path, default=DEFAULT_DENSE_RAW)
    dense.add_argument("--start-nm", type=float, default=TARGET_START_NM)
    dense.add_argument("--stop-nm", type=float, default=TARGET_STOP_NM)
    dense.add_argument("--step-nm", type=float, default=TARGET_STEP_NM)
    dense.add_argument("--settle-s", type=float, default=0.18)
    dense.add_argument("--fast-measurements", type=int, default=3)
    dense.add_argument("--print-every", type=int, default=10)
    dense.add_argument("--resume", action="store_true")

    dense_phase = sub.add_parser(
        "dense-phase", help="live-close a dense grid from measured PHASE candidates"
    )
    dense_phase.add_argument("--port", default="COM6")
    dense_phase.add_argument("--gpib", default="GPIB0::7::INSTR")
    dense_phase.add_argument("--candidates", type=Path,
                             default=DEFAULT_PHASE_CANDIDATES)
    dense_phase.add_argument(
        "--phase-maps", type=Path, nargs="+",
        default=[HERE / "fullband_phase_sweep_map_1mA.json"],
    )
    dense_phase.add_argument("--output", type=Path,
                             default=HERE / "fullband_dense_phase_raw_2001.json")
    dense_phase.add_argument("--start-nm", type=float, default=TARGET_START_NM)
    dense_phase.add_argument("--stop-nm", type=float, default=TARGET_STOP_NM)
    dense_phase.add_argument("--step-nm", type=float, default=TARGET_STEP_NM)
    dense_phase.add_argument("--settle-s", type=float, default=0.18)
    dense_phase.add_argument("--ranked-seeds", type=int, default=2)
    dense_phase.add_argument("--nearest-seeds", type=int, default=5)
    dense_phase.add_argument("--refine-seeds", type=int, default=2)
    dense_phase.add_argument("--max-measurements", type=int, default=10)
    dense_phase.add_argument("--print-every", type=int, default=10)
    dense_phase.add_argument("--resume", action="store_true")

    repair_dense = sub.add_parser(
        "repair-dense", help="repair failed dense points by continuous branch walks"
    )
    repair_dense.add_argument("--port", default="COM6")
    repair_dense.add_argument("--gpib", default="GPIB0::7::INSTR")
    repair_dense.add_argument("--input", type=Path,
                              default=HERE / "fullband_dense_phase_raw_2001.json")
    repair_dense.add_argument("--output", type=Path,
                              default=HERE / "fullband_dense_phase_repaired_2001.json")
    repair_dense.add_argument("--settle-s", type=float, default=0.18)
    repair_dense.add_argument("--max-measurements", type=int, default=30)
    repair_dense.add_argument("--passes", type=int, default=2)
    repair_dense.add_argument("--resume", action="store_true")

    upgrade_power = sub.add_parser(
        "upgrade-power", help="replace low-power rows with valid higher-power branches"
    )
    upgrade_power.add_argument("--port", default="COM6")
    upgrade_power.add_argument("--gpib", default="GPIB0::7::INSTR")
    upgrade_power.add_argument("--input", type=Path,
                               default=HERE / "fullband_dense_phase_repaired_2001.json")
    upgrade_power.add_argument("--output", type=Path,
                               default=HERE / "fullband_dense_high_power_2001.json")
    upgrade_power.add_argument("--candidates", type=Path,
                               default=DEFAULT_PHASE_CANDIDATES)
    upgrade_power.add_argument("--phase-maps", type=Path, nargs="+", required=True)
    upgrade_power.add_argument("--floor-mw", type=float, default=1.60)
    upgrade_power.add_argument("--minimum-gain-mw", type=float, default=0.02)
    upgrade_power.add_argument("--search-radius-nm", type=float, default=0.25)
    upgrade_power.add_argument("--measured-seeds", type=int, default=4)
    upgrade_power.add_argument("--screen-seeds", type=int, default=8)
    upgrade_power.add_argument("--refine-seeds", type=int, default=2)
    upgrade_power.add_argument("--max-measurements", type=int, default=14)
    upgrade_power.add_argument("--settle-s", type=float, default=0.18)
    upgrade_power.add_argument(
        "--adaptive-bottleneck", action="store_true",
        help="search only the live global minimum until its branch set is exhausted",
    )
    upgrade_power.add_argument("--resume", action="store_true")

    soa_pilot = sub.add_parser(
        "soa-pilot", help="measure a normalized SOA-to-output-power transfer curve"
    )
    soa_pilot.add_argument("--port", default="COM6")
    soa_pilot.add_argument("--gpib", default="GPIB0::7::INSTR")
    soa_pilot.add_argument("--input", type=Path,
                           default=HERE / "fullband_dense_high_power_2001.json")
    soa_pilot.add_argument("--output", type=Path,
                           default=HERE / "fullband_soa_transfer_pilot.json")
    soa_pilot.add_argument("--soa-max-ma", type=float, default=MAX_GAIN_SOA_MA)
    soa_pilot.add_argument("--soa-min-ma", type=float, default=95.0)
    soa_pilot.add_argument("--soa-step-ma", type=float, default=10.0)
    soa_pilot.add_argument("--wavelength-samples", type=int, default=7)
    soa_pilot.add_argument("--low-power-samples", type=int, default=3)
    soa_pilot.add_argument("--high-power-samples", type=int, default=2)
    soa_pilot.add_argument("--settle-s", type=float, default=0.18)

    equalize = sub.add_parser(
        "equalize-power",
        help="close every wavelength to one measured single-mode power",
    )
    equalize.add_argument("--port", default="COM6")
    equalize.add_argument("--gpib", default="GPIB0::7::INSTR")
    equalize.add_argument("--input", type=Path,
                          default=HERE / "fullband_dense_high_power_2001.json")
    equalize.add_argument("--soa-pilot", type=Path,
                          default=HERE / "fullband_soa_transfer_pilot_85mA.json")
    equalize.add_argument("--output", type=Path,
                          default=HERE / "fullband_equal_power_2001.json")
    equalize.add_argument(
        "--target-power-mw", type=float, default=None,
        help="explicit common power; omit to choose the highest robust value from all 2001 rows",
    )
    equalize.add_argument("--power-headroom-relative", type=float,
                          default=POWER_REL_TOLERANCE)
    equalize.add_argument(
        "--power-lift-relative", type=float, default=0.0,
        help=(
            "raise the nominal target above the weakest measured maximum by "
            "this relative tolerance; use only inside a wider release band"
        ),
    )
    equalize.add_argument("--power-resolution-mw", type=float, default=0.001)
    equalize.add_argument("--power-relative-tolerance", type=float,
                          default=POWER_REL_TOLERANCE)
    equalize.add_argument("--start-nm", type=float, default=TARGET_START_NM)
    equalize.add_argument("--stop-nm", type=float, default=TARGET_STOP_NM)
    equalize.add_argument("--soa-min-ma", type=float, default=85.0)
    equalize.add_argument("--soa-max-ma", type=float, default=MAX_GAIN_SOA_MA)
    equalize.add_argument("--max-soa-step-ma", type=float, default=3.0)
    equalize.add_argument("--settle-s", type=float, default=0.18)
    equalize.add_argument("--max-rounds", type=int, default=6)
    equalize.add_argument("--max-tune-measurements", type=int, default=12)
    equalize.add_argument("--print-every", type=int, default=10)
    equalize.add_argument("--resume", action="store_true")

    repair_equal = sub.add_parser(
        "repair-equal-power",
        help="repair missing equal-power points from valid neighboring branches",
    )
    repair_equal.add_argument("--port", default="COM6")
    repair_equal.add_argument("--gpib", default="GPIB0::7::INSTR")
    repair_equal.add_argument("--input", type=Path,
                              default=HERE / "fullband_equal_power_2001.json")
    repair_equal.add_argument("--output", type=Path,
                              default=HERE / "fullband_equal_power_neighbor_repaired_2001.json")
    repair_equal.add_argument("--source-table", type=Path,
                               default=HERE / "fullband_dense_high_power_2001.json")
    repair_equal.add_argument(
        "--inventory", type=Path, default=None,
        help="live measured candidate inventory for alternate WAVE-A/WAVE-B branches",
    )
    repair_equal.add_argument(
        "--phase-candidates", type=Path, default=None,
        help="alternate measured PHASE/Vernier candidates for difficult rows",
    )
    repair_equal.add_argument("--soa-pilot", type=Path,
                              default=HERE / "fullband_soa_transfer_pilot_85mA.json")
    repair_equal.add_argument("--target-power-mw", type=float, default=1.530)
    repair_equal.add_argument("--power-relative-tolerance", type=float,
                              default=POWER_REL_TOLERANCE)
    repair_equal.add_argument(
        "--repair-power-target-relative", type=float,
        default=REPAIR_POWER_TARGET_RELATIVE,
        help="inner live tuning tolerance; must not exceed release tolerance",
    )
    repair_equal.add_argument("--passes", type=int, default=2)
    repair_equal.add_argument("--settle-s", type=float, default=0.18)
    repair_equal.add_argument("--max-rounds", type=int, default=4)
    repair_equal.add_argument("--max-tune-measurements", type=int, default=14)
    repair_equal.add_argument("--resume", action="store_true")

    validate_equal = sub.add_parser(
        "validate-equal-power",
        help="repeat every final code in one independent scan direction",
    )
    validate_equal.add_argument("--port", default="COM6")
    validate_equal.add_argument("--gpib", default="GPIB0::7::INSTR")
    validate_equal.add_argument("--input", type=Path,
                                default=HERE / "fullband_equal_power_neighbor_repaired_2001.json")
    validate_equal.add_argument("--output", type=Path,
                                default=HERE / "fullband_equal_power_reverse_validation_2001.json")
    validate_equal.add_argument("--direction", choices=("forward", "reverse"),
                                default="reverse")
    validate_equal.add_argument(
        "--target-power-mw", type=float, default=None,
        help="defaults to the calibrated table's recorded common power",
    )
    validate_equal.add_argument("--power-relative-tolerance", type=float,
                                default=POWER_REL_TOLERANCE)
    validate_equal.add_argument("--settle-s", type=float, default=0.24)
    validate_equal.add_argument("--samples-per-point", type=int, default=3)
    validate_equal.add_argument("--repeat-settle-s", type=float, default=0.02)
    validate_equal.add_argument("--print-every", type=int, default=40)
    validate_equal.add_argument("--quiet-failures", action="store_true")
    validate_equal.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_requested_currents(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.command == "inventory":
        payload = build_inventory(args.output)
        print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
        print(f"Wrote {args.output}")
        return 0
    if args.command == "train":
        bundle = train_surrogate(args.output)
        print(json.dumps({
            "training_counts": bundle["training_counts"],
            "metrics": bundle["metrics"],
            "output": str(args.output),
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "map":
        payload = run_mode_map(args)
        print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
        return 0
    if args.command == "select-phase-branches":
        payload = select_phase_sweep_branches(args)
        print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
        return 0
    if args.command == "select-gap-phase-branches":
        payload = select_gap_phase_branches(args)
        print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
        return 0
    if args.command == "select-alternate-mode-branches":
        payload = select_unmeasured_mode_branches(args)
        print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
        return 0
    if args.command == "phase-map":
        payload = run_phase_map(args)
        print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
        return 0
    if args.command == "build-phase-candidates":
        payload = build_phase_interpolated_candidates(args)
        print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
        if payload["uncovered_targets_nm"]:
            print("uncovered:", payload["uncovered_targets_nm"][:30])
        return 0 if payload["summary"]["uncovered_targets"] == 0 else 2
    if args.command == "pilot":
        payload = run_pilot(args)
        print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
        return 0 if payload["summary"]["failed"] == 0 else 2
    if args.command == "direct":
        payload = run_direct(args)
        row = payload["rows"][f"{args.target_nm:.6f}"]
        print(json.dumps({
            "success": row["success"], "target_nm": args.target_nm,
            "codes": row["codes"], "currents_ma": row["currents_ma"],
            "reading": row["reading"], "confirmations": row["confirmations"],
        }, ensure_ascii=False, indent=2))
        return 0 if row["success"] else 2
    if args.command == "dense":
        payload = run_dense(args)
        print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
        return 0 if payload["summary"]["failed"] == 0 else 2
    if args.command == "dense-phase":
        payload = run_dense_phase_candidates(args)
        print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
        return 0 if payload["summary"]["failed"] == 0 else 2
    if args.command == "repair-dense":
        payload = run_repair_dense(args)
        print(json.dumps(payload["repair_summary"], ensure_ascii=False, indent=2))
        return 0 if payload["repair_summary"]["failed"] == 0 else 2
    if args.command == "upgrade-power":
        payload = run_upgrade_low_power(args)
        print(json.dumps(payload["power_upgrade_summary"], ensure_ascii=False, indent=2))
        return 0
    if args.command == "soa-pilot":
        payload = run_soa_pilot(args)
        print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
        return 0
    if args.command == "equalize-power":
        payload = run_equalize_power(args)
        print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
        return 0 if payload["summary"]["failed"] == 0 else 2
    if args.command == "repair-equal-power":
        payload = run_repair_equal_power(args)
        print(json.dumps(payload["neighbor_repair_summary"], ensure_ascii=False, indent=2))
        return 0 if payload["neighbor_repair_summary"]["failed"] == 0 else 2
    if args.command == "validate-equal-power":
        payload = run_validate_equal_power(args)
        print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
        return 0 if payload["summary"]["failed"] == 0 else 2
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
