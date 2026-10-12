"""Safely qualify and tune a sparse CH1/9-FBG candidate table.

This is an *offline proposal* tool: it writes an auditable JSON report only.
It never installs a desktop table, changes a firmware source file, or performs
an OTA/programming operation.

The mandatory first stage, ``probe-only``, replays the audited full-band
predecessor before every candidate and records an exact-DAC-ACK AQ6150(B)
measurement.  The recommended common power is the weakest measured clean
point reduced by a configurable guard fraction; it is therefore never an
unmeasured extrapolation.

The ``calibrate`` stage requires a completed clean probe report.  It changes
only SOA and PHASE, in small steps and within both a local envelope and the
authorised absolute current limits.  Every proposed final state must pass two
independent predecessor->target replays for wavelength, power, SMSR and
repeatability.  Every hardware exit path requests EXTRA mode plus SOA CLR and
requires the exact status/DAC acknowledgement before success is reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import app_JDSU as app
from capture_ch1_teacher_spectrum import (
    DEFAULT_CONNECT_TIMEOUT_S,
    DEFAULT_HOST,
    command_with_link_retry,
    safe_shutter_and_confirm,
    save_checkpoint,
    wait_connected,
)
from fbg_lan_serial import FbgLanSerial
from laser_dac_safety import validate_pi11210_codes
from laser_power_calibration import (
    AQ6150B,
    Reading,
    code_to_current,
    current_to_code,
)

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "artifacts" / "selected_ch1_calibration_checkpoint.json"
DEFAULT_OPERATIONAL_JSON = HERE / "fullband_equal_power_operational_2001.json"
SCHEMA = "fbg_selected_ch1_calibration_v1"
CEILING_PHASE_CONTROLLER_REVISION = "measured_bracket_secant_v2"
CALIBRATION_CONTROLLER_REVISION = (
    "dual_measured_seed_staged_slope_v3_internal_1pct_1pm_smsr_latch"
)
CALIBRATION_INTERNAL_POWER_FRACTION = 0.01
CALIBRATION_INTERNAL_WAVELENGTH_TOLERANCE_PM = 1.0
SMSR_DIAGNOSTIC_REPEATS = 2
CHANNEL_NAMES = ("GAIN", "SOA", "PHASE", "WAVE_A", "WAVE_B")
ABSOLUTE_LIMITS_MA = (135.0, 135.0, 10.0, 30.0, 30.0)
COMMAND_CODE_LIMITS = (
    app.PI11210_GAIN_MAX_CODE,
    app.PI11210_SOA_SOURCE_MAX_CODE,
    app.PI11210_PHASE_MAX_CODE,
    app.PI11210_WAVELENGTH_MAX_CODE,
    app.PI11210_WAVELENGTH_MAX_CODE,
)
SINGLE_MODE_SENTINEL_DB = 99.0
DEFAULT_MIN_SMSR_DB = 20.0
DEFAULT_WAVELENGTH_TOLERANCE_PM = 2.0
DEFAULT_PROBE_BRANCH_TOLERANCE_PM = 50.0
DEFAULT_POWER_GUARD_FRACTION = 0.03
DEFAULT_CEILING_GUARD_FRACTION = 0.01
DEFAULT_POWER_TOLERANCE_FRACTION = 0.02
DEFAULT_POWER_TOLERANCE_MW = 0.010
DEFAULT_PHASE_SLOPE_PM_PER_MA = -75.0
DEFAULT_MAX_PHASE_DELTA_MA = 0.60
DEFAULT_MAX_PHASE_STEP_MA = 0.10
DEFAULT_MAX_SOA_DELTA_MA = 12.0
DEFAULT_MAX_SOA_STEP_MA = 1.50
DEFAULT_CEILING_SOA_STEP_MA = 2.0
DEFAULT_SETTLE_S = 0.30
DEFAULT_PREDECESSOR_SETTLE_S = 0.08
DEFAULT_CONFIRMATION_WAVELENGTH_SPAN_PM = 2.0
DEFAULT_CONFIRMATION_POWER_SPAN_FRACTION = 0.03
MAX_TUNING_ITERATIONS = 16
DEFAULT_SEQUENCE_ROUNDS = 2
FIRMWARE_HIDDEN_PREDECESSOR_HOLD_US = 5000
SPARSE_DIRECT_PATH = "previous_sparse->target"
SPARSE_HIDDEN_PATH = "previous_sparse->hidden_fullband_predecessor->target"


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_float(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须是有限数") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label}必须是有限数")
    return result


def _exact_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label}必须是整数")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须是整数") from exc
    if not math.isfinite(numeric) or numeric != math.trunc(numeric):
        raise ValueError(f"{label}必须是整数")
    return int(numeric)


@dataclass(frozen=True)
class CandidatePoint:
    order: int
    fullband_index: int
    target_nm: float
    codes: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class CeilingSeed:
    codes: tuple[int, int, int, int, int]
    source: str
    legacy_code_clamps: tuple[str, ...] = ()
    historical_reading: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class CalibrationSettings:
    settle_s: float = DEFAULT_SETTLE_S
    predecessor_settle_s: float = DEFAULT_PREDECESSOR_SETTLE_S
    minimum_smsr_db: float = DEFAULT_MIN_SMSR_DB
    probe_branch_tolerance_pm: float = DEFAULT_PROBE_BRANCH_TOLERANCE_PM
    wavelength_tolerance_pm: float = DEFAULT_WAVELENGTH_TOLERANCE_PM
    power_guard_fraction: float = DEFAULT_POWER_GUARD_FRACTION
    ceiling_guard_fraction: float = DEFAULT_CEILING_GUARD_FRACTION
    power_tolerance_fraction: float = DEFAULT_POWER_TOLERANCE_FRACTION
    power_tolerance_mw: float = DEFAULT_POWER_TOLERANCE_MW
    maximum_phase_delta_ma: float = DEFAULT_MAX_PHASE_DELTA_MA
    maximum_phase_step_ma: float = DEFAULT_MAX_PHASE_STEP_MA
    maximum_soa_delta_ma: float = DEFAULT_MAX_SOA_DELTA_MA
    maximum_soa_step_ma: float = DEFAULT_MAX_SOA_STEP_MA
    ceiling_soa_step_ma: float = DEFAULT_CEILING_SOA_STEP_MA
    confirmation_wavelength_span_pm: float = DEFAULT_CONFIRMATION_WAVELENGTH_SPAN_PM
    confirmation_power_span_fraction: float = DEFAULT_CONFIRMATION_POWER_SPAN_FRACTION
    maximum_iterations: int = MAX_TUNING_ITERATIONS
    sequence_rounds: int = DEFAULT_SEQUENCE_ROUNDS

    def validate(self) -> None:
        positive = {
            "settle_s": self.settle_s,
            "minimum_smsr_db": self.minimum_smsr_db,
            "probe_branch_tolerance_pm": self.probe_branch_tolerance_pm,
            "wavelength_tolerance_pm": self.wavelength_tolerance_pm,
            "power_tolerance_mw": self.power_tolerance_mw,
            "maximum_phase_delta_ma": self.maximum_phase_delta_ma,
            "maximum_phase_step_ma": self.maximum_phase_step_ma,
            "maximum_soa_delta_ma": self.maximum_soa_delta_ma,
            "maximum_soa_step_ma": self.maximum_soa_step_ma,
            "ceiling_soa_step_ma": self.ceiling_soa_step_ma,
            "confirmation_wavelength_span_pm": (self.confirmation_wavelength_span_pm),
            "confirmation_power_span_fraction": (self.confirmation_power_span_fraction),
        }
        for label, value in positive.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{label}必须是正有限数")
        if self.settle_s < app.FULLBAND_ACCURACY_MIN_SETTLE_S:
            raise ValueError(
                f"settle_s不得小于{app.FULLBAND_ACCURACY_MIN_SETTLE_S:.2f}秒"
            )
        if (
            not math.isfinite(self.predecessor_settle_s)
            or self.predecessor_settle_s < 0.0
        ):
            raise ValueError("predecessor_settle_s必须是非负有限数")
        for label, value in {
            "power_guard_fraction": self.power_guard_fraction,
            "ceiling_guard_fraction": self.ceiling_guard_fraction,
            "power_tolerance_fraction": self.power_tolerance_fraction,
        }.items():
            if not math.isfinite(float(value)) or not 0.0 <= float(value) < 1.0:
                raise ValueError(f"{label}必须位于[0, 1)")
        if self.maximum_phase_delta_ma > 1.0:
            raise ValueError("局部PHASE调整包络不得超过1.0 mA")
        if self.maximum_soa_delta_ma > 20.0:
            raise ValueError("局部SOA调整包络不得超过20.0 mA")
        if self.maximum_phase_step_ma > self.maximum_phase_delta_ma:
            raise ValueError("PHASE单步不得超过PHASE总包络")
        if self.maximum_soa_step_ma > self.maximum_soa_delta_ma:
            raise ValueError("SOA单步不得超过SOA总包络")
        if self.ceiling_soa_step_ma > 5.0:
            raise ValueError("功率上限探测SOA单步不得超过5.0 mA")
        if not 1 <= int(self.maximum_iterations) <= 40:
            raise ValueError("maximum_iterations必须位于1～40")
        if not 2 <= int(self.sequence_rounds) <= 10:
            raise ValueError("sequence_rounds必须位于2～10")


def _rows_container(payload: Mapping[str, Any]) -> object:
    stress = payload.get("stress")
    if isinstance(stress, Mapping) and "rows" in stress:
        return stress["rows"]
    if "rows" in payload:
        return payload["rows"]
    # The newer adaptive selector emits nine ``peaks[]`` groups, with each
    # selected DAC state under ``points[]`` and its full-band index named
    # ``source_index``.  Flatten that representation while retaining strict
    # peak/channel metadata for the checks below.
    owner = stress if isinstance(stress, Mapping) else payload
    peaks = owner.get("peaks") if isinstance(owner, Mapping) else None
    if isinstance(peaks, Sequence) and not isinstance(peaks, (str, bytes, bytearray)):
        if len(peaks) != 9:
            raise ValueError("CH1 peaks[]候选表必须恰好包含9个峰")
        flattened: list[dict[str, Any]] = []
        for ordinal, peak in enumerate(peaks, start=1):
            if not isinstance(peak, Mapping):
                raise ValueError(f"peaks[{ordinal - 1}]不是对象")
            peak_number = _exact_int(
                peak.get("peak_number", ordinal),
                f"peaks[{ordinal - 1}].peak_number",
            )
            points = peak.get("points")
            if (
                not isinstance(points, Sequence)
                or isinstance(points, (str, bytes, bytearray))
                or not points
            ):
                raise ValueError(f"第{peak_number}个峰没有候选points")
            for point in points:
                if not isinstance(point, Mapping):
                    raise ValueError(f"第{peak_number}个峰包含非对象候选点")
                normalized = dict(point)
                normalized.setdefault("peak_number", peak_number)
                normalized.setdefault(
                    "source_channel",
                    peak.get("source_channel", owner.get("source_channel", 1)),
                )
                flattened.append(normalized)
        return flattened
    raise ValueError("候选JSON必须包含stress.rows、rows或九组peaks[].points")


def normalize_candidate_rows(payload: Mapping[str, Any]) -> tuple[CandidatePoint, ...]:
    raw_rows = _rows_container(payload)
    if isinstance(raw_rows, Mapping):
        iterable: Iterable[object] = raw_rows.values()
    elif isinstance(raw_rows, Sequence) and not isinstance(
        raw_rows, (str, bytes, bytearray)
    ):
        iterable = raw_rows
    else:
        raise ValueError("rows必须是数组或对象")

    points: list[CandidatePoint] = []
    seen_indices: set[int] = set()
    peak_numbers: list[int | None] = []
    for order, item in enumerate(iterable):
        if not isinstance(item, Mapping):
            raise ValueError(f"第{order + 1}个候选点不是对象")
        index_value = item.get("fullband_index", item.get("source_index"))
        if index_value is None:
            raise ValueError(f"第{order + 1}个候选点缺fullband_index/source_index")
        fullband_index = _exact_int(
            index_value, f"第{order + 1}点fullband_index/source_index"
        )
        if not 0 <= fullband_index < app.FULLBAND_ACCURACY_POINT_COUNT:
            raise ValueError(f"第{order + 1}点fullband_index越界")
        if fullband_index in seen_indices:
            raise ValueError(f"fullband_index重复：{fullband_index}")
        seen_indices.add(fullband_index)
        if (
            "source_channel" in item
            and _exact_int(item["source_channel"], f"第{order + 1}点source_channel")
            != 1
        ):
            raise ValueError(f"第{order + 1}点不是CH1候选点")
        peak_number = (
            _exact_int(item["peak_number"], f"第{order + 1}点peak_number")
            if "peak_number" in item
            else None
        )
        if peak_number is not None and not 1 <= peak_number <= 9:
            raise ValueError(f"第{order + 1}点peak_number必须位于1～9")
        peak_numbers.append(peak_number)
        target_value = item.get(
            "target_wavelength_nm",
            item.get("target_nm", item.get("wavelength_nm")),
        )
        if target_value is None:
            raise ValueError(f"第{order + 1}个候选点缺target_wavelength_nm/target_nm")
        target_nm = _finite_float(target_value, f"第{order + 1}点目标波长")
        if not (
            float(app.FULLBAND_ACCURACY_START_NM) - 1e-9
            <= target_nm
            <= float(app.FULLBAND_ACCURACY_START_NM)
            + float(app.FULLBAND_ACCURACY_STEP_NM)
            * (app.FULLBAND_ACCURACY_POINT_COUNT - 1)
            + 1e-9
        ):
            raise ValueError(f"第{order + 1}点目标波长越界：{target_nm}")
        expected_target_nm = (
            float(app.FULLBAND_ACCURACY_START_NM)
            + float(app.FULLBAND_ACCURACY_STEP_NM) * fullband_index
        )
        if abs(target_nm - expected_target_nm) > (
            float(app.FULLBAND_ACCURACY_STEP_NM) / 2.0 + 1e-9
        ):
            raise ValueError(
                f"第{order + 1}点波长{target_nm:.6f} nm与"
                f"fullband_index={fullband_index}不匹配"
            )
        raw_codes = item.get("codes", item.get("dac_codes"))
        if raw_codes is None:
            raise ValueError(f"第{order + 1}个候选点缺codes/dac_codes")
        codes = validate_pi11210_codes(raw_codes)
        validate_absolute_currents(codes)
        points.append(
            CandidatePoint(
                order=order,
                fullband_index=fullband_index,
                target_nm=target_nm,
                codes=codes,
            )
        )
    if not points:
        raise ValueError("候选点表为空")
    if any(value is not None for value in peak_numbers):
        if any(value is None for value in peak_numbers):
            raise ValueError("候选表一旦提供peak_number，每行都必须提供")
        if set(peak_numbers) != set(range(1, 10)):
            raise ValueError("CH1候选表必须覆盖且仅覆盖9个光栅峰")
    return tuple(points)


def load_candidates(path: Path) -> tuple[dict[str, Any], tuple[CandidatePoint, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("候选JSON根结点必须是对象")
    return payload, normalize_candidate_rows(payload)


def _normalize_legacy_audited_codes(
    raw_codes: Sequence[object],
) -> tuple[tuple[int, int, int, int, int], tuple[str, ...]]:
    """Import old 65536-scale 135/10/30 mA maxima without sending them.

    Historical AQ6150 reports used mathematical 65536-scale maxima, one code
    above the later desktop/firmware integer clamps.  Only that exact one-code
    legacy boundary is converted, and the conversion is explicitly audited.
    Any larger excess is rejected.
    """

    if len(raw_codes) != 5:
        raise ValueError("历史高功率种子必须包含5个DAC码")
    values: list[int] = []
    clamps: list[str] = []
    for channel, (raw, maximum) in enumerate(zip(raw_codes, COMMAND_CODE_LIMITS)):
        code = _exact_int(raw, f"历史{CHANNEL_NAMES[channel]} DAC码")
        if code < 0 or code > maximum + 1:
            raise ValueError(f"历史{CHANNEL_NAMES[channel]} DAC码{code}超出可审计边界")
        if code == maximum + 1:
            clamps.append(f"{CHANNEL_NAMES[channel]}:{code}->{maximum}")
            code = maximum
        values.append(code)
    normalized = validate_absolute_currents(values)
    return tuple(normalized), tuple(clamps)


def _historical_seed_clean(
    row: Mapping[str, Any],
    point: CandidatePoint,
    minimum_smsr_db: float,
) -> bool:
    reading = row.get("reading")
    if not isinstance(reading, Mapping):
        return False
    try:
        wave = float(reading["wavelength_nm"])
        power = float(reading["power_mw"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        math.isfinite(wave)
        and math.isfinite(power)
        and power > 0.0
        and abs(wave - point.target_nm) * 1000.0 <= DEFAULT_PROBE_BRANCH_TOLERANCE_PM
        and normalized_smsr_db(reading) >= minimum_smsr_db
    )


def load_ceiling_seeds(
    path: Path,
    candidates: Sequence[CandidatePoint],
    *,
    minimum_smsr_db: float,
) -> dict[int, CeilingSeed]:
    """Extract historically measured max-power seeds; never trust blindly."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_rows = payload.get("rows", {})
    if not isinstance(raw_rows, (Mapping, Sequence)) or isinstance(
        raw_rows, (str, bytes, bytearray)
    ):
        raise ValueError("operational JSON的rows结构无效")
    result: dict[int, CeilingSeed] = {}
    for point in candidates:
        if isinstance(raw_rows, Mapping):
            operational = raw_rows.get(f"{point.target_nm:.6f}")
            if operational is None:
                # Fall back to insertion order only after target-key lookup.
                values = list(raw_rows.values())
                operational = (
                    values[point.fullband_index]
                    if point.fullband_index < len(values)
                    else None
                )
        else:
            operational = (
                raw_rows[point.fullband_index]
                if point.fullband_index < len(raw_rows)
                else None
            )
        if not isinstance(operational, Mapping):
            continue
        choices: list[tuple[str, object]] = [
            ("source_max_power_row", operational.get("source_max_power_row"))
        ]
        original_failure = operational.get("original_failure")
        if isinstance(original_failure, Mapping):
            choices.append(
                (
                    "original_failure.source_max_power_row",
                    original_failure.get("source_max_power_row"),
                )
            )
        for source, raw_seed in choices:
            if not isinstance(raw_seed, Mapping):
                continue
            if not _historical_seed_clean(raw_seed, point, minimum_smsr_db):
                continue
            try:
                seed_codes, clamps = _normalize_legacy_audited_codes(raw_seed["codes"])
            except (KeyError, ValueError):
                continue
            result[point.order] = CeilingSeed(
                codes=seed_codes,
                source=source,
                legacy_code_clamps=clamps,
                historical_reading=dict(raw_seed["reading"]),
            )
            break
    return result


def validate_absolute_currents(codes: Sequence[object]) -> tuple[int, ...]:
    normalized = validate_pi11210_codes(codes)
    for name, code, maximum in zip(CHANNEL_NAMES, normalized, COMMAND_CODE_LIMITS):
        if code > maximum:
            raise ValueError(f"{name} DAC码{code}超过实际命令上限0～{maximum}")
    currents = tuple(
        code_to_current(code, channel) for channel, code in enumerate(normalized)
    )
    for name, current, ceiling in zip(CHANNEL_NAMES, currents, ABSOLUTE_LIMITS_MA):
        if current > ceiling + 1e-9:
            raise ValueError(f"{name}={current:.6f} mA超过{ceiling:.1f} mA")
    return normalized


def safe_current_to_code(current_ma: float, channel: int) -> int:
    """Convert without ever crossing the desktop/firmware integer clamp."""

    return min(current_to_code(current_ma, channel), COMMAND_CODE_LIMITS[channel])


def validate_and_hash_fullband_points(points: Sequence[Any]) -> str:
    if len(points) != app.FULLBAND_ACCURACY_POINT_COUNT:
        raise ValueError("前驱状态必须来自完整2001点审计表")
    canonical: list[list[int]] = []
    for index, point in enumerate(points):
        try:
            point_codes = validate_absolute_currents(point.codes)
        except AttributeError as exc:
            raise ValueError(f"2001点表第{index}点缺codes") from exc
        canonical.append(list(point_codes))
    encoded = json.dumps(canonical, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def currents_ma(codes: Sequence[int]) -> list[float]:
    return [code_to_current(code, channel) for channel, code in enumerate(codes)]


def normalized_smsr_db(reading: Reading | Mapping[str, Any]) -> float:
    if isinstance(reading, Mapping):
        count = int(reading.get("peak_count", 0) or 0)
        value = reading.get("side_mode_suppression_db")
    else:
        count = int(reading.peak_count)
        value = reading.side_mode_suppression_db
    if count == 1 and value is None:
        return SINGLE_MODE_SENTINEL_DB
    if value is None:
        return float("-inf")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    return result if math.isfinite(result) else float("-inf")


def smsr_event_classification(events: Sequence[Mapping[str, Any]]) -> str | None:
    classifications = {str(event.get("classification")) for event in events}
    if "soa_phase_trajectory_unstable" in classifications:
        return "soa_phase_trajectory_unstable"
    if "transient_side_mode_event" in classifications:
        return "transient_side_mode_event"
    return None


def reading_dict(reading: Reading, target_nm: float) -> dict[str, Any]:
    result = asdict(reading)
    result["wavelength_error_pm"] = (
        float(reading.wavelength_nm) - float(target_nm)
    ) * 1000.0
    result["normalized_smsr_db"] = normalized_smsr_db(reading)
    return result


def validate_meter_reading(reading: Reading) -> None:
    for label, value in {
        "AQ6150波长": reading.wavelength_nm,
        "AQ6150功率": reading.power_mw,
    }.items():
        if not math.isfinite(float(value)):
            raise RuntimeError(f"{label}不是有限数")
    if float(reading.power_mw) <= 0.0:
        raise RuntimeError("AQ6150功率必须为正数")
    if int(reading.peak_count) < 1:
        raise RuntimeError("AQ6150未返回光学峰")


class SelectedCh1Calibrator:
    """Hardware-neutral calibration core using the production ACK worker."""

    def __init__(
        self,
        worker: app.UnlimitedAccuracyWorker,
        meter: AQ6150B,
        fullband_points: Sequence[Any],
        settings: CalibrationSettings,
    ) -> None:
        settings.validate()
        self.worker = worker
        self.meter = meter
        self.fullband_points = tuple(fullband_points)
        self.settings = settings
        if len(self.fullband_points) != app.FULLBAND_ACCURACY_POINT_COUNT:
            raise ValueError("前驱状态必须来自完整2001点审计表")

    def _wait(self, seconds: float) -> None:
        if not self.worker._interruptible_wait(seconds):
            raise RuntimeError("标定已中止")

    def _command(self, codes: Sequence[int]) -> None:
        exact = validate_absolute_currents(codes)
        if command_with_link_retry(self.worker, exact) is None:
            raise RuntimeError("未获得精确DAC回读确认")

    def _predecessor_codes(self, point: CandidatePoint) -> tuple[int, ...]:
        predecessor_index = max(0, point.fullband_index - 1)
        return validate_absolute_currents(self.fullband_points[predecessor_index].codes)

    def measure_state(
        self,
        point: CandidatePoint,
        codes: Sequence[int],
        *,
        settle_s: float | None = None,
    ) -> Reading:
        """Always reproduce the audited predecessor->target optical path."""

        self._command(self._predecessor_codes(point))
        self._wait(self.settings.predecessor_settle_s)
        self._command(codes)
        self._wait(self.settings.settle_s if settle_s is None else settle_s)
        reading = self.meter.measure(point.target_nm, select_main_peak=True)
        validate_meter_reading(reading)
        return reading

    def measure_sparse_transition(
        self,
        point: CandidatePoint,
        codes: Sequence[int],
        predecessor_codes: Sequence[int],
        *,
        hidden_predecessor_codes: Sequence[int] | None = None,
        hidden_predecessor_hold_us: int = FIRMWARE_HIDDEN_PREDECESSOR_HOLD_US,
    ) -> Reading:
        """Measure the exact runtime sparse path, including one optional hidden row."""

        self._command(predecessor_codes)
        self._wait(self.settings.predecessor_settle_s)
        if hidden_predecessor_codes is not None:
            if int(hidden_predecessor_hold_us) != FIRMWARE_HIDDEN_PREDECESSOR_HOLD_US:
                raise ValueError(
                    "隐藏前驱保持时间必须与固件严格一致为5000 us"
                )
            self._command(hidden_predecessor_codes)
            self._wait(hidden_predecessor_hold_us / 1_000_000.0)
        self._command(codes)
        self._wait(self.settings.settle_s)
        reading = self.meter.measure(point.target_nm, select_main_peak=True)
        validate_meter_reading(reading)
        return reading

    def probe(self, point: CandidatePoint) -> dict[str, Any]:
        reading = self.measure_state(point, point.codes)
        result = reading_dict(reading, point.target_nm)
        branch_ok = (
            abs(float(result["wavelength_error_pm"]))
            <= self.settings.probe_branch_tolerance_pm
        )
        smsr_ok = float(result["normalized_smsr_db"]) >= self.settings.minimum_smsr_db
        return {
            "order": point.order,
            "fullband_index": point.fullband_index,
            "target_wavelength_nm": point.target_nm,
            "codes": list(point.codes),
            "currents_ma": currents_ma(point.codes),
            "predecessor_index": max(0, point.fullband_index - 1),
            "predecessor_codes": list(self._predecessor_codes(point)),
            "reading": result,
            "branch_ok": branch_ok,
            "smsr_ok": smsr_ok,
            "qualified": bool(branch_ok and smsr_ok),
        }

    def _diagnose_low_smsr_state(
        self,
        point: CandidatePoint,
        *,
        suspect_codes: Sequence[int],
        first_reading: Reading,
        rollback_codes: Sequence[int] | None,
        history: list[dict[str, Any]],
        forbidden_codes: set[tuple[int, ...]],
        context: str,
    ) -> dict[str, Any]:
        """Latch an unsafe exact state, repeat it, then verify a clean rollback.

        A later clean sample never clears the latch.  This is deliberate: an
        intermittently bimodal state is not repeatable enough for a runtime
        table even when two immediate re-reads happen to look single-mode.
        """

        suspect = tuple(validate_absolute_currents(suspect_codes))
        forbidden_codes.add(suspect)
        repeats: list[Reading] = []
        for _repeat in range(SMSR_DIAGNOSTIC_REPEATS):
            repeats.append(self.measure_state(point, suspect))
        repeat_smsrs = [normalized_smsr_db(reading) for reading in repeats]
        classification = (
            "transient_side_mode_event"
            if all(smsr >= self.settings.minimum_smsr_db for smsr in repeat_smsrs)
            else "soa_phase_trajectory_unstable"
        )

        rollback_reading: Reading | None = None
        rollback_confirmed = False
        rollback: tuple[int, ...] | None = None
        if rollback_codes is not None:
            rollback = tuple(validate_absolute_currents(rollback_codes))
            if rollback not in forbidden_codes:
                rollback_reading = self.measure_state(point, rollback)
                rollback_confirmed = self._clean_branch(
                    rollback_reading, point.target_nm
                )
                if (
                    normalized_smsr_db(rollback_reading)
                    < self.settings.minimum_smsr_db
                ):
                    forbidden_codes.add(rollback)
                    classification = "soa_phase_trajectory_unstable"

        event = {
            "action": "low_smsr_diagnostic",
            "context": context,
            "classification": classification,
            "accept_state_for_final": False,
            "forbidden_for_current_and_resumed_run": True,
            "minimum_smsr_db": self.settings.minimum_smsr_db,
            "suspect_codes": list(suspect),
            "first_reading": reading_dict(first_reading, point.target_nm),
            "repeat_readings": [
                reading_dict(reading, point.target_nm) for reading in repeats
            ],
            "rollback_codes": list(rollback) if rollback is not None else None,
            "rollback_reading": (
                reading_dict(rollback_reading, point.target_nm)
                if rollback_reading is not None
                else None
            ),
            "rollback_confirmed_clean": rollback_confirmed,
            "rollback_latched_low_smsr": bool(
                rollback is not None and rollback in forbidden_codes
            ),
        }
        history.append(event)
        return event

    def _ceiling_phase_correct(
        self,
        point: CandidatePoint,
        seed_codes: Sequence[int],
        first_reading: Reading,
        history: list[dict[str, Any]],
        *,
        phase_limits_ma: tuple[float, float] | None = None,
        context: str = "ceiling_probe",
        acceptance_tolerance_pm: float | None = None,
        forbidden_codes: set[tuple[int, ...]] | None = None,
    ) -> tuple[tuple[int, ...], Reading] | None:
        """Correct PHASE only while SOA and branch-defining codes stay fixed."""

        base = tuple(validate_absolute_currents(seed_codes))
        codes = base
        reading = first_reading
        accepted_error_pm = (
            self.settings.wavelength_tolerance_pm
            if acceptance_tolerance_pm is None
            else _finite_float(acceptance_tolerance_pm, "PHASE内环波长容差")
        )
        if accepted_error_pm <= 0.0:
            raise ValueError("PHASE内环波长容差必须为正数")
        base_phase_ma = currents_ma(base)[2]
        phase_low_ma = max(0.0, base_phase_ma - self.settings.maximum_phase_delta_ma)
        phase_high_ma = min(
            ABSOLUTE_LIMITS_MA[2],
            base_phase_ma + self.settings.maximum_phase_delta_ma,
        )
        if phase_limits_ma is not None:
            external_low = _finite_float(phase_limits_ma[0], "PHASE外部下界")
            external_high = _finite_float(phase_limits_ma[1], "PHASE外部上界")
            if not 0.0 <= external_low <= external_high <= ABSOLUTE_LIMITS_MA[2]:
                raise ValueError("PHASE外部包络必须位于0～10mA且下界不大于上界")
            phase_low_ma = max(phase_low_ma, external_low)
            phase_high_ma = min(phase_high_ma, external_high)
            if phase_low_ma > phase_high_ma:
                return None
        # (DAC code, current in mA, wavelength error in pm).  Keeping the
        # measured points lets the controller learn the local PHASE slope.
        # The AQ6150 run showed roughly -100...-350 pm/mA, so applying the
        # old fixed -75 pm/mA estimate at the full 0.10 mA step repeatedly
        # crossed the target and oscillated.
        samples: list[tuple[int, float, float]] = []
        tested_phase_codes: set[int] = set()

        for correction in range(self.settings.maximum_iterations):
            error_pm = (reading.wavelength_nm - point.target_nm) * 1000.0
            if (
                self._clean_branch(reading, point.target_nm)
                and abs(error_pm) <= accepted_error_pm
            ):
                return codes, reading
            if not self._clean_branch(reading, point.target_nm):
                return None
            current_phase = currents_ma(codes)[2]
            current_phase_code = int(codes[2])
            samples.append((current_phase_code, current_phase, error_pm))
            tested_phase_codes.add(current_phase_code)

            # Prefer an opposite-sign bracket.  Secant interpolation inside a
            # measured bracket is both faster and safer than trusting a
            # nominal laser slope.  Choose the narrowest available bracket.
            bracket: (
                tuple[tuple[int, float, float], tuple[int, float, float]] | None
            ) = None
            ordered = sorted(samples, key=lambda sample: sample[1])
            for left, right in zip(ordered, ordered[1:]):
                if left[2] * right[2] >= 0.0:
                    continue
                if (
                    bracket is None
                    or right[1] - left[1] < bracket[1][1] - bracket[0][1]
                ):
                    bracket = (left, right)

            controller = "nominal_slope_exploration"
            estimated_slope: float | None = None
            step_limit_ma = self.settings.maximum_phase_step_ma * 0.5
            if bracket is not None:
                left, right = bracket
                denominator = right[2] - left[2]
                if abs(denominator) > 1e-12:
                    requested = left[1] - left[2] * (right[1] - left[1]) / denominator
                    controller = "measured_bracket_secant"
                else:
                    requested = (left[1] + right[1]) * 0.5
                    controller = "measured_bracket_bisection"
                requested = min(max(requested, left[1]), right[1])
                estimated_slope = denominator / (right[1] - left[1])
                step_limit_ma = self.settings.maximum_phase_step_ma
            else:
                # With no bracket, use the newest distinct measured pair when
                # it agrees with the known negative PHASE direction.  A
                # positive/noisy estimate is not allowed to reverse control.
                previous = next(
                    (
                        sample
                        for sample in reversed(samples[:-1])
                        if sample[0] != current_phase_code
                    ),
                    None,
                )
                if previous is not None:
                    delta_phase = current_phase - previous[1]
                    measured_slope = (
                        (error_pm - previous[2]) / delta_phase
                        if abs(delta_phase) > 1e-12
                        else float("nan")
                    )
                    if math.isfinite(measured_slope) and measured_slope < -1e-6:
                        estimated_slope = measured_slope
                        requested = current_phase - error_pm / measured_slope
                        controller = "measured_local_secant"
                        step_limit_ma = self.settings.maximum_phase_step_ma
                    else:
                        requested = (
                            current_phase - error_pm / DEFAULT_PHASE_SLOPE_PM_PER_MA
                        )
                else:
                    requested = current_phase - error_pm / DEFAULT_PHASE_SLOPE_PM_PER_MA

            # The first unmeasured move is intentionally half of the allowed
            # maximum.  Once a real slope or bracket exists, the configured
            # full step may be used.  All paths retain the original local and
            # absolute PHASE envelopes.
            requested = min(
                max(requested, current_phase - step_limit_ma),
                current_phase + step_limit_ma,
            )
            requested = min(max(requested, phase_low_ma), phase_high_ma)
            next_codes = list(codes)
            next_codes[2] = safe_current_to_code(requested, 2)
            next_codes = tuple(validate_absolute_currents(next_codes))

            # Quantisation or a secant landing on an already measured code
            # must not recreate an A<->B oscillation.  When bracketed, select
            # the closest still-unmeasured DAC code inside both the bracket
            # and this step's safety interval.  If none exists, the physical
            # DAC/envelope really cannot improve this state.
            if next_codes[2] in tested_phase_codes and bracket is not None:
                left_code, right_code = sorted((bracket[0][0], bracket[1][0]))
                step_low_code = safe_current_to_code(
                    max(phase_low_ma, current_phase - step_limit_ma), 2
                )
                step_high_code = safe_current_to_code(
                    min(phase_high_ma, current_phase + step_limit_ma), 2
                )
                allowed_low = max(left_code, min(step_low_code, step_high_code))
                allowed_high = min(right_code, max(step_low_code, step_high_code))
                desired_code = int(next_codes[2])
                alternatives = (
                    phase_code
                    for phase_code in range(allowed_low, allowed_high + 1)
                    if phase_code not in tested_phase_codes
                )
                replacement = min(
                    alternatives,
                    key=lambda phase_code: abs(phase_code - desired_code),
                    default=None,
                )
                if replacement is not None:
                    next_codes = list(next_codes)
                    next_codes[2] = replacement
                    next_codes = tuple(validate_absolute_currents(next_codes))

            if next_codes == codes or next_codes[2] in tested_phase_codes:
                return None
            if forbidden_codes is not None and next_codes in forbidden_codes:
                history.append(
                    {
                        "action": "forbidden_smsr_state_blocked",
                        "context": context,
                        "codes": list(next_codes),
                        "reason": (
                            "该精确DAC状态在本次或历史尝试中曾出现"
                            f"SMSR<{self.settings.minimum_smsr_db:.1f} dB"
                        ),
                    }
                )
                return None
            previous_codes = codes
            reading = self.measure_state(point, next_codes)
            codes = next_codes
            history.append(
                {
                    "action": "phase_correction",
                    "context": context,
                    "correction": correction,
                    "codes": list(codes),
                    "currents_ma": currents_ma(codes),
                    "controller": controller,
                    "estimated_phase_slope_pm_per_ma": estimated_slope,
                    "phase_envelope_ma": [phase_low_ma, phase_high_ma],
                    "reading": reading_dict(reading, point.target_nm),
                }
            )
            if (
                forbidden_codes is not None
                and normalized_smsr_db(reading) < self.settings.minimum_smsr_db
            ):
                self._diagnose_low_smsr_state(
                    point,
                    suspect_codes=codes,
                    first_reading=reading,
                    rollback_codes=(
                        previous_codes if previous_codes != codes else None
                    ),
                    history=history,
                    forbidden_codes=forbidden_codes,
                    context=context,
                )
                return None
        return None

    def ceiling_probe(
        self, point: CandidatePoint, seed: CeilingSeed | None
    ) -> dict[str, Any]:
        """Measure, never extrapolate, the highest clean local SOA state."""

        history: list[dict[str, Any]] = []
        attempts: list[tuple[str, tuple[int, ...]]] = []
        if seed is not None:
            attempts.append((seed.source, seed.codes))
        if seed is None or seed.codes != point.codes:
            attempts.append(("candidate_codes", point.codes))

        best_codes: tuple[int, ...] | None = None
        best_reading: Reading | None = None
        chosen_source = ""
        for source, start_codes in attempts:
            reading = self.measure_state(point, start_codes)
            history.append(
                {
                    "action": "seed_probe",
                    "source": source,
                    "codes": list(start_codes),
                    "currents_ma": currents_ma(start_codes),
                    "reading": reading_dict(reading, point.target_nm),
                }
            )
            corrected = self._ceiling_phase_correct(
                point, start_codes, reading, history
            )
            if corrected is not None:
                best_codes, best_reading = corrected
                chosen_source = source
                break
        if best_codes is None or best_reading is None:
            return {
                "order": point.order,
                "fullband_index": point.fullband_index,
                "target_wavelength_nm": point.target_nm,
                "phase_controller_revision": CEILING_PHASE_CONTROLLER_REVISION,
                "qualified": False,
                "failure_reason": "历史高功率种子和当前候选均未通过分支/SMSR/波长门限",
                "history": history,
            }

        # If the audited seed is not already at the command ceiling, climb in
        # small SOA steps.  No predicted point is accepted: every ceiling is
        # an actual AQ6150 measurement.
        while True:
            current_soa = currents_ma(best_codes)[1]
            next_soa = min(
                ABSOLUTE_LIMITS_MA[1],
                current_soa + self.settings.ceiling_soa_step_ma,
            )
            next_soa_code = safe_current_to_code(next_soa, 1)
            if next_soa_code <= best_codes[1]:
                break
            trial_codes = list(best_codes)
            trial_codes[1] = next_soa_code
            trial_codes = tuple(validate_absolute_currents(trial_codes))
            trial = self.measure_state(point, trial_codes)
            history.append(
                {
                    "action": "soa_ceiling_step",
                    "codes": list(trial_codes),
                    "currents_ma": currents_ma(trial_codes),
                    "reading": reading_dict(trial, point.target_nm),
                }
            )
            corrected = self._ceiling_phase_correct(point, trial_codes, trial, history)
            if corrected is None or corrected[1].power_mw < best_reading.power_mw:
                rollback = self.measure_state(point, best_codes)
                rollback_ok = bool(
                    self._clean_branch(rollback, point.target_nm)
                    and abs((rollback.wavelength_nm - point.target_nm) * 1000.0)
                    <= self.settings.wavelength_tolerance_pm
                )
                history.append(
                    {
                        "action": "rollback_after_rejected_step",
                        "codes": list(best_codes),
                        "reading": reading_dict(rollback, point.target_nm),
                        "rollback_confirmed": rollback_ok,
                    }
                )
                if not rollback_ok:
                    return {
                        "order": point.order,
                        "fullband_index": point.fullband_index,
                        "target_wavelength_nm": point.target_nm,
                        "phase_controller_revision": CEILING_PHASE_CONTROLLER_REVISION,
                        "qualified": False,
                        "failure_reason": "上限探测失败后回退状态未通过复测",
                        "history": history,
                    }
                best_reading = rollback
                break
            best_codes, best_reading = corrected

        return {
            "order": point.order,
            "fullband_index": point.fullband_index,
            "target_wavelength_nm": point.target_nm,
            "phase_controller_revision": CEILING_PHASE_CONTROLLER_REVISION,
            "seed_source": chosen_source,
            "legacy_code_clamps": (
                list(seed.legacy_code_clamps)
                if seed is not None and chosen_source == seed.source
                else []
            ),
            "historical_seed_reading": (
                dict(seed.historical_reading)
                if seed is not None
                and chosen_source == seed.source
                and seed.historical_reading is not None
                else None
            ),
            "ceiling_codes": list(best_codes),
            "ceiling_currents_ma": currents_ma(best_codes),
            "ceiling_reading": reading_dict(best_reading, point.target_nm),
            "clean_ceiling_power_mw": float(best_reading.power_mw),
            "qualified": True,
            "history": history,
        }

    def sparse_sequence_observation(
        self,
        point: CandidatePoint,
        codes: Sequence[int],
        predecessor_codes: Sequence[int],
        *,
        round_index: int,
        predecessor_order: int,
        target_power_mw: float | None = None,
        hidden_predecessor_fullband_index: int | None = None,
        hidden_predecessor_codes: Sequence[int] | None = None,
        hidden_predecessor_hold_us: int = FIRMWARE_HIDDEN_PREDECESSOR_HOLD_US,
    ) -> dict[str, Any]:
        uses_hidden = hidden_predecessor_codes is not None
        if uses_hidden:
            expected_index = max(0, point.fullband_index - 1)
            if hidden_predecessor_fullband_index != expected_index:
                raise ValueError("隐藏前驱必须是目标在2001点表中的直接前一点")
            exact_hidden = validate_absolute_currents(hidden_predecessor_codes)
            if exact_hidden != self._predecessor_codes(point):
                raise ValueError("隐藏前驱DAC码必须来自2001点审计表")
            if int(hidden_predecessor_hold_us) != FIRMWARE_HIDDEN_PREDECESSOR_HOLD_US:
                raise ValueError("隐藏前驱保持时间必须与固件严格一致为5000 us")
        else:
            if hidden_predecessor_fullband_index is not None:
                raise ValueError("未启用隐藏前驱时不得记录隐藏前驱索引")
            exact_hidden = None
        reading = self.measure_sparse_transition(
            point,
            codes,
            predecessor_codes,
            hidden_predecessor_codes=exact_hidden,
            hidden_predecessor_hold_us=hidden_predecessor_hold_us,
        )
        wavelength_error_pm = (reading.wavelength_nm - point.target_nm) * 1000.0
        power_tolerance_mw = (
            self._power_tolerance(target_power_mw)
            if target_power_mw is not None
            else None
        )
        power_ok = bool(
            target_power_mw is None
            or abs(reading.power_mw - target_power_mw) <= power_tolerance_mw
        )
        passed = bool(
            abs(wavelength_error_pm) <= self.settings.wavelength_tolerance_pm
            and normalized_smsr_db(reading) >= self.settings.minimum_smsr_db
            and power_ok
        )
        result = {
            "round": round_index,
            "order": point.order,
            "predecessor_order": predecessor_order,
            "wraparound": bool(point.order == 0),
            "fullband_index": point.fullband_index,
            "target_wavelength_nm": point.target_nm,
            "predecessor_codes": list(predecessor_codes),
            "codes": list(validate_absolute_currents(codes)),
            "reading": reading_dict(reading, point.target_nm),
            "target_power_mw": target_power_mw,
            "power_tolerance_mw": power_tolerance_mw,
            "power_ok": power_ok,
            "passed": passed,
            "requires_replacement_or_hidden_predecessor": not passed,
            "runtime_path": SPARSE_HIDDEN_PATH if uses_hidden else SPARSE_DIRECT_PATH,
            "used_hidden_predecessor": uses_hidden,
        }
        if uses_hidden:
            result["hidden_predecessor"] = {
                "fullband_index": int(hidden_predecessor_fullband_index),
                "codes": list(exact_hidden),
                "hold_us": int(hidden_predecessor_hold_us),
            }
        return result

    def _bounded_codes(
        self,
        original: CandidatePoint,
        base_codes: Sequence[int],
        requested_soa_ma: float,
        requested_phase_ma: float,
    ) -> tuple[int, ...]:
        original_ma = currents_ma(original.codes)
        # A certified common target can be substantially below a point's
        # measured 135 mA ceiling.  Reducing SOA is intrinsically current-safe,
        # so it must not be trapped by the old symmetric +/-12 mA envelope.
        # Upward movement remains locally bounded and absolutely capped.
        soa_low = 0.0
        soa_high = min(
            ABSOLUTE_LIMITS_MA[1],
            original_ma[1] + self.settings.maximum_soa_delta_ma,
        )
        phase_low = max(0.0, original_ma[2] - self.settings.maximum_phase_delta_ma)
        phase_high = min(
            ABSOLUTE_LIMITS_MA[2],
            original_ma[2] + self.settings.maximum_phase_delta_ma,
        )
        values = list(validate_absolute_currents(base_codes))
        values[1] = safe_current_to_code(
            min(max(float(requested_soa_ma), soa_low), soa_high), 1
        )
        values[2] = safe_current_to_code(
            min(max(float(requested_phase_ma), phase_low), phase_high), 2
        )
        # The calibrator is intentionally incapable of changing these three
        # branch-defining channels.
        values[0] = original.codes[0]
        values[3] = original.codes[3]
        values[4] = original.codes[4]
        return validate_absolute_currents(values)

    def _power_tolerance(self, target_power_mw: float) -> float:
        return max(
            self.settings.power_tolerance_mw,
            target_power_mw * self.settings.power_tolerance_fraction,
        )

    def _calibration_control_power_tolerance(self, target_power_mw: float) -> float:
        # Keep the persisted 2% setting as the broad runtime/certificate gate,
        # but tune and independently confirm this table to the stricter 1%
        # internal target.  If the operator configured an even tighter gate,
        # that tighter value still wins.
        return min(
            self._power_tolerance(target_power_mw),
            target_power_mw * CALIBRATION_INTERNAL_POWER_FRACTION,
        )

    def _clean_branch(self, reading: Reading, target_nm: float) -> bool:
        return bool(
            abs((reading.wavelength_nm - target_nm) * 1000.0)
            <= self.settings.probe_branch_tolerance_pm
            and normalized_smsr_db(reading) >= self.settings.minimum_smsr_db
        )

    def _adaptive_soa_request(
        self,
        samples: Sequence[tuple[int, float, float]],
        *,
        current_soa_ma: float,
        current_power_mw: float,
        power_error_mw: float,
        soa_low_ma: float,
        soa_high_ma: float,
    ) -> tuple[float, str, float | None]:
        """Choose one bounded SOA step from measured power/current points."""

        bracket: tuple[tuple[int, float, float], tuple[int, float, float]] | None = None
        ordered = sorted(samples, key=lambda sample: sample[1])
        for left, right in zip(ordered, ordered[1:]):
            if left[2] * right[2] >= 0.0:
                continue
            if bracket is None or right[1] - left[1] < bracket[1][1] - bracket[0][1]:
                bracket = (left, right)

        slope_mw_per_ma: float | None = None
        controller = "power_per_current_exploration"
        if bracket is not None:
            left, right = bracket
            denominator = right[2] - left[2]
            if abs(denominator) > 1e-12:
                requested = left[1] - left[2] * (right[1] - left[1]) / denominator
                controller = "measured_power_bracket_secant"
            else:
                requested = (left[1] + right[1]) * 0.5
                controller = "measured_power_bracket_bisection"
            requested = min(max(requested, left[1]), right[1])
            slope_mw_per_ma = denominator / (right[1] - left[1])
        else:
            current_code = int(samples[-1][0])
            previous = next(
                (
                    sample
                    for sample in reversed(samples[:-1])
                    if sample[0] != current_code
                ),
                None,
            )
            if previous is not None:
                delta_soa = current_soa_ma - previous[1]
                measured_slope = (
                    (power_error_mw - previous[2]) / delta_soa
                    if abs(delta_soa) > 1e-12
                    else float("nan")
                )
                # SOA optical power must increase with SOA current locally.
                # Reject a negative/noisy estimate rather than reversing the
                # feedback direction.
                if math.isfinite(measured_slope) and measured_slope > 1e-9:
                    slope_mw_per_ma = measured_slope
                    requested = current_soa_ma - power_error_mw / measured_slope
                    controller = "measured_local_power_secant"
                else:
                    nominal_slope = current_power_mw / max(current_soa_ma, 1e-9)
                    requested = current_soa_ma - power_error_mw / nominal_slope
            else:
                nominal_slope = current_power_mw / max(current_soa_ma, 1e-9)
                requested = current_soa_ma - power_error_mw / nominal_slope

        requested = min(
            max(
                requested,
                current_soa_ma - self.settings.maximum_soa_step_ma,
            ),
            current_soa_ma + self.settings.maximum_soa_step_ma,
        )
        requested = min(max(requested, soa_low_ma), soa_high_ma)
        return requested, controller, slope_mw_per_ma

    def calibrate(
        self,
        point: CandidatePoint,
        target_power_mw: float,
        *,
        ceiling_codes: Sequence[int] | None = None,
        inherited_forbidden_codes: Iterable[Sequence[int]] = (),
    ) -> dict[str, Any]:
        target_power_mw = _finite_float(target_power_mw, "共同目标功率")
        if target_power_mw <= 0.0:
            raise ValueError("共同目标功率必须为正数")
        tolerance_mw = self._power_tolerance(target_power_mw)
        control_tolerance_mw = self._calibration_control_power_tolerance(
            target_power_mw
        )
        control_wavelength_tolerance_pm = min(
            self.settings.wavelength_tolerance_pm,
            CALIBRATION_INTERNAL_WAVELENGTH_TOLERANCE_PM,
        )
        history: list[dict[str, Any]] = []
        inherited_forbidden = {
            tuple(validate_absolute_currents(codes))
            for codes in inherited_forbidden_codes
        }
        forbidden_codes = set(inherited_forbidden)
        seed_probes: list[dict[str, Any]] = []
        seed_candidates: list[
            tuple[tuple[float, float, float], str, tuple[int, ...], Reading]
        ] = []
        seed_options: list[tuple[str, tuple[int, ...]]] = [
            ("candidate", tuple(point.codes))
        ]
        if ceiling_codes is not None:
            normalized_ceiling = tuple(validate_absolute_currents(ceiling_codes))
            if normalized_ceiling != point.codes:
                seed_options.append(("ceiling_probe", normalized_ceiling))

        for source, seed_codes in seed_options:
            if seed_codes in forbidden_codes:
                blocked = {
                    "action": "calibration_seed_probe",
                    "source": source,
                    "codes": list(seed_codes),
                    "qualified": False,
                    "blocked_by_prior_low_smsr_latch": True,
                }
                seed_probes.append(blocked)
                history.append(blocked)
                continue
            seed_reading = self.measure_state(point, seed_codes)
            wave_error_pm = (
                float(seed_reading.wavelength_nm) - point.target_nm
            ) * 1000.0
            smsr = normalized_smsr_db(seed_reading)
            qualified = bool(
                abs(wave_error_pm) <= self.settings.probe_branch_tolerance_pm
                and smsr >= self.settings.minimum_smsr_db
            )
            seed_probe = {
                "action": "calibration_seed_probe",
                "source": source,
                "codes": list(seed_codes),
                "currents_ma": currents_ma(seed_codes),
                "reading": reading_dict(seed_reading, point.target_nm),
                "qualified": qualified,
                "power_distance_to_target_mw": abs(
                    float(seed_reading.power_mw) - target_power_mw
                ),
            }
            seed_probes.append(seed_probe)
            history.append(seed_probe)
            if smsr < self.settings.minimum_smsr_db:
                self._diagnose_low_smsr_state(
                    point,
                    suspect_codes=seed_codes,
                    first_reading=seed_reading,
                    rollback_codes=None,
                    history=history,
                    forbidden_codes=forbidden_codes,
                    context=f"calibration_seed:{source}",
                )
                seed_probe["qualified"] = False
                seed_probe["latched_low_smsr"] = True
                continue
            if qualified:
                # Power distance is intentionally the primary score: it
                # minimises the SOA trajectory.  Wavelength and SMSR only
                # break ties after both seeds passed the broad branch gate.
                score = (
                    abs(float(seed_reading.power_mw) - target_power_mw),
                    abs(wave_error_pm),
                    -smsr,
                )
                seed_candidates.append((score, source, seed_codes, seed_reading))

        if not seed_candidates:
            events = [
                item for item in history if item.get("action") == "low_smsr_diagnostic"
            ]
            return {
                "order": point.order,
                "fullband_index": point.fullband_index,
                "target_wavelength_nm": point.target_nm,
                "candidate_codes": list(point.codes),
                "candidate_currents_ma": currents_ma(point.codes),
                "target_power_mw": target_power_mw,
                "power_tolerance_mw": tolerance_mw,
                "control_power_tolerance_mw": control_tolerance_mw,
                "calibration_controller_revision": CALIBRATION_CONTROLLER_REVISION,
                "seed_probes": seed_probes,
                "inherited_forbidden_codes": [
                    list(codes) for codes in sorted(inherited_forbidden)
                ],
                "forbidden_codes": [list(codes) for codes in sorted(forbidden_codes)],
                "smsr_diagnostics": events,
                "candidate_replacement_required": True,
                "passed": False,
                "failure_reason": "candidate与ceiling均未通过实测分支/SMSR种子门限",
                "history": history,
            }

        _score, seed_source, selected_seed_codes, selected_seed_reading = min(
            seed_candidates, key=lambda item: item[0]
        )
        working_point = CandidatePoint(
            order=point.order,
            fullband_index=point.fullband_index,
            target_nm=point.target_nm,
            codes=selected_seed_codes,
        )
        codes = selected_seed_codes
        best: tuple[float, tuple[int, ...], Reading] | None = None
        failure_reason = "分阶段SOA/PHASE闭环未收敛"
        original_ma = currents_ma(working_point.codes)
        soa_low_ma = 0.0
        soa_high_ma = min(
            ABSOLUTE_LIMITS_MA[1],
            original_ma[1] + self.settings.maximum_soa_delta_ma,
        )
        phase_limits_ma = (
            max(0.0, original_ma[2] - self.settings.maximum_phase_delta_ma),
            min(
                ABSOLUTE_LIMITS_MA[2],
                original_ma[2] + self.settings.maximum_phase_delta_ma,
            ),
        )
        power_samples: list[tuple[int, float, float]] = []
        power_step_budget = self.settings.maximum_iterations * 2
        outer_budget = power_step_budget + self.settings.maximum_iterations
        power_steps = 0
        converged = False
        reading = selected_seed_reading
        history.append(
            {
                "iteration": len(history),
                "action": "selected_seed_initial_state",
                "source": seed_source,
                "codes": list(codes),
                "currents_ma": currents_ma(codes),
                "reading": reading_dict(reading, point.target_nm),
            }
        )

        for outer_iteration in range(outer_budget):
            if not self._clean_branch(reading, point.target_nm):
                failure_reason = (
                    f"候选状态跳模或SMSR低于{self.settings.minimum_smsr_db:.1f} dB"
                )
                break

            wavelength_error_pm = (
                float(reading.wavelength_nm) - point.target_nm
            ) * 1000.0
            power_error_mw = float(reading.power_mw) - target_power_mw
            score = (
                abs(wavelength_error_pm) / control_wavelength_tolerance_pm
                + abs(power_error_mw) / control_tolerance_mw
            )
            if best is None or score < best[0]:
                best = (score, tuple(codes), reading)
            if (
                abs(wavelength_error_pm) <= control_wavelength_tolerance_pm
                and abs(power_error_mw) <= control_tolerance_mw
            ):
                converged = True
                break

            current = currents_ma(codes)
            if reading.power_mw <= 1e-12 or current[1] <= 1e-9:
                failure_reason = "光功率或SOA电流过低，不允许外推"
                break

            if abs(power_error_mw) > control_tolerance_mw:
                if power_steps >= power_step_budget:
                    failure_reason = "SOA小步功率闭环达到安全迭代预算仍未收敛"
                    break
                sample = (int(codes[1]), current[1], power_error_mw)
                if power_samples and power_samples[-1][0] == sample[0]:
                    power_samples[-1] = sample
                else:
                    power_samples.append(sample)
                requested_soa, controller, measured_slope = self._adaptive_soa_request(
                    power_samples,
                    current_soa_ma=current[1],
                    current_power_mw=reading.power_mw,
                    power_error_mw=power_error_mw,
                    soa_low_ma=soa_low_ma,
                    soa_high_ma=soa_high_ma,
                )
                next_codes = self._bounded_codes(
                    working_point, codes, requested_soa, current[2]
                )
                if next_codes in forbidden_codes:
                    history.append(
                        {
                            "action": "forbidden_smsr_state_blocked",
                            "context": "calibration_soa_outer",
                            "codes": list(next_codes),
                        }
                    )
                    failure_reason = "SOA闭环要求重访已锁死的低SMSR精确状态"
                    break
                if next_codes[1] == codes[1]:
                    failure_reason = "SOA DAC分辨率或安全边界已限制功率调整"
                    break
                trial = self.measure_state(point, next_codes)
                power_steps += 1
                history.append(
                    {
                        "iteration": len(history),
                        "outer_iteration": outer_iteration,
                        "action": "soa_power_step",
                        "controller": controller,
                        "estimated_soa_power_slope_mw_per_ma": measured_slope,
                        "soa_envelope_ma": [soa_low_ma, soa_high_ma],
                        "codes": list(next_codes),
                        "currents_ma": currents_ma(next_codes),
                        "reading": reading_dict(trial, point.target_nm),
                    }
                )
                if not self._clean_branch(trial, point.target_nm):
                    if normalized_smsr_db(trial) < self.settings.minimum_smsr_db:
                        self._diagnose_low_smsr_state(
                            point,
                            suspect_codes=next_codes,
                            first_reading=trial,
                            rollback_codes=codes,
                            history=history,
                            forbidden_codes=forbidden_codes,
                            context="calibration_soa_outer",
                        )
                        failure_reason = "SOA功率轨迹出现低SMSR状态，该精确DAC已锁死"
                    else:
                        failure_reason = "SOA功率步进后超出目标波长分支门限"
                    break
                corrected = self._ceiling_phase_correct(
                    point,
                    next_codes,
                    trial,
                    history,
                    phase_limits_ma=phase_limits_ma,
                    context="calibration_phase_inner",
                    acceptance_tolerance_pm=control_wavelength_tolerance_pm,
                    forbidden_codes=forbidden_codes,
                )
                if corrected is None:
                    failure_reason = "SOA步进后的PHASE实测内环未能安全恢复目标波长"
                    break
                codes, reading = corrected
                continue

            corrected = self._ceiling_phase_correct(
                point,
                codes,
                reading,
                history,
                phase_limits_ma=phase_limits_ma,
                context="calibration_phase_inner",
                acceptance_tolerance_pm=control_wavelength_tolerance_pm,
                forbidden_codes=forbidden_codes,
            )
            if corrected is None:
                failure_reason = "PHASE实测内环达到边界或未收敛"
                break
            codes, reading = corrected

        for history_index, item in enumerate(history):
            item.setdefault("history_index", history_index)

        smsr_diagnostics = [
            item for item in history if item.get("action") == "low_smsr_diagnostic"
        ]
        trajectory_classification = smsr_event_classification(smsr_diagnostics)
        base = {
            "order": point.order,
            "fullband_index": point.fullband_index,
            "target_wavelength_nm": point.target_nm,
            "candidate_codes": list(point.codes),
            "candidate_currents_ma": currents_ma(point.codes),
            # Retain these legacy names for old report readers, but make the
            # selected source explicit so a row cannot look as if it started
            # from the same branch as previous passed rows.
            "original_codes": list(working_point.codes),
            "original_currents_ma": currents_ma(working_point.codes),
            "seed_source": seed_source,
            "selected_seed_codes": list(working_point.codes),
            "selected_seed_currents_ma": currents_ma(working_point.codes),
            "seed_probes": seed_probes,
            "inherited_forbidden_codes": [
                list(codes) for codes in sorted(inherited_forbidden)
            ],
            "forbidden_codes": [list(codes) for codes in sorted(forbidden_codes)],
            "smsr_diagnostics": smsr_diagnostics,
            "trajectory_classification": trajectory_classification,
            "target_power_mw": target_power_mw,
            "power_tolerance_mw": tolerance_mw,
            "control_power_tolerance_mw": control_tolerance_mw,
            "control_power_tolerance_fraction": (CALIBRATION_INTERNAL_POWER_FRACTION),
            "control_wavelength_tolerance_pm": control_wavelength_tolerance_pm,
            "calibration_controller_revision": CALIBRATION_CONTROLLER_REVISION,
            "soa_envelope_ma": [soa_low_ma, soa_high_ma],
            "phase_envelope_ma": list(phase_limits_ma),
            "power_step_budget": power_step_budget,
            "power_steps_used": power_steps,
            "history": history,
        }
        if best is None:
            return {
                **base,
                "candidate_replacement_required": bool(
                    trajectory_classification == "soa_phase_trajectory_unstable"
                    and len(seed_candidates) < 2
                ),
                "passed": False,
                "failure_reason": failure_reason,
            }

        best_codes = best[1]
        best_error_pm = (best[2].wavelength_nm - point.target_nm) * 1000.0
        if (
            not converged
            or abs(best_error_pm) > control_wavelength_tolerance_pm
            or abs(best[2].power_mw - target_power_mw) > control_tolerance_mw
        ):
            return {
                **base,
                "candidate_replacement_required": bool(
                    trajectory_classification == "soa_phase_trajectory_unstable"
                    and len(seed_candidates) < 2
                ),
                "passed": False,
                "failure_reason": failure_reason,
            }

        if best_codes in forbidden_codes:
            return {
                **base,
                "candidate_replacement_required": len(seed_candidates) < 2,
                "passed": False,
                "failure_reason": "最佳状态曾出现低SMSR，禁止进入最终复测",
            }

        confirmations: list[dict[str, Any]] = []
        for repeat in range(2):
            reading = self.measure_state(point, best_codes)
            confirmations.append(reading_dict(reading, point.target_nm))
            if normalized_smsr_db(reading) < self.settings.minimum_smsr_db:
                self._diagnose_low_smsr_state(
                    point,
                    suspect_codes=best_codes,
                    first_reading=reading,
                    rollback_codes=(
                        working_point.codes
                        if working_point.codes != best_codes
                        else None
                    ),
                    history=history,
                    forbidden_codes=forbidden_codes,
                    context="calibration_final_confirmation",
                )
                break

        wavelengths = [float(item["wavelength_nm"]) for item in confirmations]
        powers = [float(item["power_mw"]) for item in confirmations]
        smsrs = [float(item["normalized_smsr_db"]) for item in confirmations]
        wavelength_span_pm = (max(wavelengths) - min(wavelengths)) * 1000.0
        power_span_fraction = (max(powers) - min(powers)) / max(
            statistics.mean(powers), 1e-12
        )
        passed = bool(
            all(
                abs((wave - point.target_nm) * 1000.0)
                <= self.settings.wavelength_tolerance_pm
                for wave in wavelengths
            )
            and all(
                abs(power - target_power_mw) <= control_tolerance_mw for power in powers
            )
            and min(smsrs) >= self.settings.minimum_smsr_db
            and wavelength_span_pm <= self.settings.confirmation_wavelength_span_pm
            and power_span_fraction <= self.settings.confirmation_power_span_fraction
        )
        if not passed:
            failure_reason = "两次独立复测的波长/功率/SMSR/重复性未全部通过"
        final_smsr_diagnostics = [
            item for item in history if item.get("action") == "low_smsr_diagnostic"
        ]
        final_trajectory_classification = smsr_event_classification(
            final_smsr_diagnostics
        )
        return {
            **base,
            "forbidden_codes": [list(codes) for codes in sorted(forbidden_codes)],
            "smsr_diagnostics": final_smsr_diagnostics,
            "trajectory_classification": final_trajectory_classification,
            "codes": list(best_codes),
            "currents_ma": currents_ma(best_codes),
            "confirmations": confirmations,
            "final_wavelength_nm": statistics.mean(wavelengths),
            "final_wavelength_error_pm": (
                statistics.mean(wavelengths) - point.target_nm
            )
            * 1000.0,
            "final_wavelength_span_pm": wavelength_span_pm,
            "final_power_mw": statistics.mean(powers),
            "final_power_span_fraction": power_span_fraction,
            "final_smsr_min_db": min(smsrs),
            "passed": bool(passed and best_codes not in forbidden_codes),
            "candidate_replacement_required": bool(
                final_trajectory_classification == "soa_phase_trajectory_unstable"
                and len(seed_candidates) < 2
            ),
            **({"failure_reason": failure_reason} if not passed else {}),
        }


def summarize_probe(
    rows: Mapping[str, Mapping[str, Any]], guard: float
) -> dict[str, Any]:
    ordered = [rows[key] for key in sorted(rows, key=int)]
    clean = [row for row in ordered if bool(row.get("qualified"))]
    complete_and_clean = bool(ordered) and len(clean) == len(ordered)
    powers = [float(row["reading"]["power_mw"]) for row in clean]
    recommendation = min(powers) * (1.0 - guard) if complete_and_clean else None
    return {
        "tested": len(ordered),
        "qualified": len(clean),
        "rejected": len(ordered) - len(clean),
        "weakest_qualified_power_mw": min(powers, default=None),
        "power_guard_fraction": guard,
        "recommended_common_power_mw": recommendation,
        "recommendation_formula": (
            "min(all qualified measured power_mw) * (1 - power_guard_fraction)"
        ),
        "all_points_qualified": complete_and_clean,
    }


def summarize_calibration(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    ordered = [rows[key] for key in sorted(rows, key=int)]
    passed = [row for row in ordered if bool(row.get("passed"))]
    return {
        "tested": len(ordered),
        "passed": len(passed),
        "failed": len(ordered) - len(passed),
        "all_points_passed": bool(ordered) and len(passed) == len(ordered),
        "maximum_absolute_wavelength_error_pm": max(
            (abs(float(row["final_wavelength_error_pm"])) for row in passed),
            default=None,
        ),
        "minimum_smsr_db": min(
            (float(row["final_smsr_min_db"]) for row in passed), default=None
        ),
        "power_min_mw": min(
            (float(row["final_power_mw"]) for row in passed), default=None
        ),
        "power_max_mw": max(
            (float(row["final_power_mw"]) for row in passed), default=None
        ),
        "requires_two_round_sparse_sequence_certification": True,
        "fullband_predecessor_results_are_not_runtime_certification": True,
    }


def summarize_ceiling(
    rows: Mapping[str, Mapping[str, Any]], guard: float
) -> dict[str, Any]:
    ordered = [rows[key] for key in sorted(rows, key=int)]
    clean = [row for row in ordered if bool(row.get("qualified"))]
    all_clean = bool(ordered) and len(clean) == len(ordered)
    powers = [float(row["clean_ceiling_power_mw"]) for row in clean]
    common = min(powers) * (1.0 - guard) if all_clean else None
    gains = [int(row["ceiling_codes"][0]) for row in clean]
    return {
        "tested": len(ordered),
        "qualified": len(clean),
        "rejected": len(ordered) - len(clean),
        "all_points_qualified": all_clean,
        "minimum_measured_clean_ceiling_mw": min(powers, default=None),
        "maximum_measured_clean_ceiling_mw": max(powers, default=None),
        "ceiling_guard_fraction": guard,
        "recommended_common_power_mw": common,
        "recommendation_formula": (
            "min(all measured clean ceiling power_mw) * (1 - ceiling_guard_fraction)"
        ),
        "uses_unmeasured_power_extrapolation": False,
        "all_gain_fixed_at_runtime_135ma_code": bool(gains)
        and all(code == app.PI11210_GAIN_MAX_CODE for code in gains),
    }


def summarize_sparse_sequence(
    rounds: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    point_count: int,
    required_rounds: int,
) -> dict[str, Any]:
    observations = [
        row for round_rows in rounds.values() for row in round_rows.values()
    ]
    expected = point_count * required_rounds
    failed = [row for row in observations if not bool(row.get("passed"))]
    hidden_orders = sorted(
        {
            int(row["order"])
            for row in observations
            if bool(row.get("used_hidden_predecessor", False))
        }
    )
    completed_rounds = sum(
        len(round_rows) == point_count for round_rows in rounds.values()
    )
    return {
        "required_rounds": required_rounds,
        "completed_rounds": completed_rounds,
        "expected_observations": expected,
        "completed_observations": len(observations),
        "failed_observations": len(failed),
        "failed_orders": sorted({int(row["order"]) for row in failed}),
        "hidden_predecessor_orders": hidden_orders,
        "hidden_predecessor_hold_us": (
            FIRMWARE_HIDDEN_PREDECESSOR_HOLD_US if hidden_orders else None
        ),
        "all_points_runtime_path_certified": bool(
            len(observations) == expected and not failed
        ),
        "includes_last_to_first_wraparound_each_round": True,
        "certification_basis": (
            "final sparse order, including last->first, not fullband index-1"
        ),
    }


def select_sequence_codes(
    report: Mapping[str, Any], candidates: Sequence[CandidatePoint]
) -> tuple[str, tuple[tuple[int, ...], ...]]:
    calibration = report.get("calibration", {})
    calibration_rows = (
        calibration.get("rows", {}) if isinstance(calibration, Mapping) else {}
    )
    if (
        isinstance(calibration_rows, Mapping)
        and len(calibration_rows) == len(candidates)
        and all(
            bool(calibration_rows[str(i)].get("passed")) for i in range(len(candidates))
        )
    ):
        return "calibration", tuple(
            validate_absolute_currents(calibration_rows[str(i)]["codes"])
            for i in range(len(candidates))
        )
    ceiling = report.get("ceiling_probe", {})
    ceiling_rows = ceiling.get("rows", {}) if isinstance(ceiling, Mapping) else {}
    if (
        isinstance(ceiling_rows, Mapping)
        and len(ceiling_rows) == len(candidates)
        and all(
            bool(ceiling_rows[str(i)].get("qualified")) for i in range(len(candidates))
        )
    ):
        return "ceiling_probe", tuple(
            validate_absolute_currents(ceiling_rows[str(i)]["ceiling_codes"])
            for i in range(len(candidates))
        )
    return "candidate", tuple(point.codes for point in candidates)


def build_hidden_predecessor_request(
    candidates: Sequence[CandidatePoint],
    fullband_points: Sequence[Any],
    requested_orders: object,
    hold_us: object = FIRMWARE_HIDDEN_PREDECESSOR_HOLD_US,
) -> dict[str, Any] | None:
    """Normalize the one firmware-supported hidden full-band predecessor."""

    if requested_orders is None:
        orders: tuple[object, ...] = ()
    elif isinstance(requested_orders, Sequence) and not isinstance(
        requested_orders, (str, bytes, bytearray)
    ):
        orders = tuple(requested_orders)
    else:
        orders = (requested_orders,)
    if len(orders) > 1:
        raise ValueError("快速应力扫描最多只允许1个隐藏前驱点")
    exact_hold_us = _exact_int(hold_us, "hidden-predecessor-hold-us")
    if exact_hold_us != FIRMWARE_HIDDEN_PREDECESSOR_HOLD_US:
        raise ValueError("隐藏前驱保持时间必须与固件严格一致为5000 us")
    if not orders:
        return None
    order = _exact_int(orders[0], "hidden-predecessor-order")
    if not 0 <= order < len(candidates):
        raise ValueError("hidden-predecessor-order超出稀疏表范围")
    if len(fullband_points) != app.FULLBAND_ACCURACY_POINT_COUNT:
        raise ValueError("隐藏前驱必须来自完整2001点审计表")
    point = candidates[order]
    predecessor_index = max(0, point.fullband_index - 1)
    predecessor_codes = validate_absolute_currents(
        fullband_points[predecessor_index].codes
    )
    return {
        "mode_order": order,
        "fullband_index": point.fullband_index,
        "predecessor_fullband_index": predecessor_index,
        "predecessor_codes": list(predecessor_codes),
        "hold_us": exact_hold_us,
        "runtime_path": SPARSE_HIDDEN_PATH,
    }


def configure_sparse_sequence_path(
    report: dict[str, Any], request: Mapping[str, Any] | None
) -> None:
    """Bind a sparse replay path and archive every row before changing it.

    A hidden-path trial may never reuse or replace just the failed point from a
    direct-path replay.  The complete old 2xN evidence is retained verbatim in
    ``retry_history`` and the active rounds are cleared so the next hardware
    session must acquire every point again.
    """

    sequence = report.setdefault("sparse_sequence", {})
    if not isinstance(sequence, dict):
        raise ValueError("稀疏路径复测结构无效")
    normalized_request = dict(request) if request is not None else None
    existing_request = sequence.get("hidden_predecessor_request")
    rounds = sequence.get("rounds", {})
    if not isinstance(rounds, Mapping):
        raise ValueError("稀疏路径复测轮次结构无效")
    path_changed = existing_request != normalized_request
    if rounds and path_changed:
        snapshot = {
            "code_source": sequence.get("code_source"),
            "hidden_predecessor_request": existing_request,
            "rounds": rounds,
            "summary": sequence.get("summary", {}),
        }
        encoded = json.dumps(
            snapshot,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        sequence.setdefault("retry_history", []).append(
            {
                "archived": now_text(),
                "reason": "hidden_predecessor_configuration_changed",
                "sha256": hashlib.sha256(encoded).hexdigest(),
                **snapshot,
            }
        )
        sequence["rounds"] = {}
        sequence["summary"] = {}
        sequence.pop("code_source", None)
        report["final_runtime_certified"] = False
    sequence["hidden_predecessor_request"] = normalized_request


def _settings_dict(settings: CalibrationSettings) -> dict[str, Any]:
    return asdict(settings)


def create_or_validate_report(
    output: Path,
    *,
    candidate_path: Path,
    candidate_sha256: str,
    point_count: int,
    settings: CalibrationSettings,
    fullband_sha256: str,
) -> dict[str, Any]:
    if output.exists():
        report = json.loads(output.read_text(encoding="utf-8"))
        if report.get("schema") != SCHEMA:
            raise ValueError("输出文件不是稀疏CH1标定检查点")
        source = report.get("candidate_source", {})
        if source.get("sha256") != candidate_sha256:
            raise ValueError("候选JSON已改变，请使用新输出文件")
        if int(source.get("point_count", -1)) != point_count:
            raise ValueError("候选点数与检查点不一致")
        if report.get("settings") != _settings_dict(settings):
            raise ValueError("续跑参数与已有检查点不一致")
        if report.get("fullband_predecessor_sha256") != fullband_sha256:
            raise ValueError("2001点前驱表已改变，请使用新输出文件")
        valid_keys = {str(index) for index in range(point_count)}
        for stage in ("probe", "ceiling_probe", "calibration"):
            rows = report.get(stage, {}).get("rows", {})
            if not isinstance(rows, Mapping) or not set(rows).issubset(valid_keys):
                raise ValueError(f"{stage}检查点结构损坏")
        return report
    return {
        "schema": SCHEMA,
        "created": now_text(),
        "updated": now_text(),
        "candidate_source": {
            "path": str(candidate_path),
            "sha256": candidate_sha256,
            "point_count": point_count,
        },
        "settings": _settings_dict(settings),
        "fullband_predecessor_sha256": fullband_sha256,
        "absolute_current_limits_ma": dict(zip(CHANNEL_NAMES, ABSOLUTE_LIMITS_MA)),
        "probe": {"rows": {}, "summary": {}},
        "ceiling_probe": {"rows": {}, "summary": {}},
        "calibration": {"rows": {}, "summary": {}},
        "sparse_sequence": {"rounds": {}, "summary": {}},
        "sessions": [],
        "shutter_confirmed": None,
        "writes_runtime_table": False,
        "writes_firmware": False,
        "final_runtime_certified": False,
    }


def validate_probe_certificate(
    probe_report: Mapping[str, Any],
    *,
    candidate_sha256: str,
    point_count: int,
) -> float:
    source = probe_report.get("candidate_source", {})
    if not isinstance(source, Mapping) or source.get("sha256") != candidate_sha256:
        raise ValueError("探测报告与当前候选JSON不匹配")
    ceiling = probe_report.get("ceiling_probe", {})
    ceiling_summary = ceiling.get("summary", {}) if isinstance(ceiling, Mapping) else {}
    ceiling_rows = ceiling.get("rows", {}) if isinstance(ceiling, Mapping) else {}
    if (
        isinstance(ceiling_rows, Mapping)
        and len(ceiling_rows) == point_count
        and bool(ceiling_summary.get("all_points_qualified"))
    ):
        power = _finite_float(
            ceiling_summary.get("recommended_common_power_mw"),
            "上限探测建议共同功率",
        )
        if power <= 0.0:
            raise ValueError("上限探测建议共同功率无效")
        return power

    summary = probe_report.get("probe", {}).get("summary", {})
    rows = probe_report.get("probe", {}).get("rows", {})
    if not isinstance(rows, Mapping) or len(rows) != point_count:
        raise ValueError("探测报告未覆盖所有候选点")
    if not bool(summary.get("all_points_qualified")):
        raise ValueError("探测报告存在跳模或SMSR不合格点")
    power = _finite_float(
        summary.get("recommended_common_power_mw"), "探测建议共同功率"
    )
    if power <= 0.0:
        raise ValueError("探测建议共同功率无效")
    return power


def collect_forbidden_low_smsr_codes(
    payload: object, minimum_smsr_db: float
) -> set[tuple[int, ...]]:
    """Recover exact unsafe states from an old failed-row audit tree."""

    result: set[tuple[int, ...]] = set()

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            latched = value.get("forbidden_codes")
            if isinstance(latched, Sequence) and not isinstance(
                latched, (str, bytes, bytearray)
            ):
                for raw_codes in latched:
                    try:
                        result.add(tuple(validate_absolute_currents(raw_codes)))
                    except (TypeError, ValueError):
                        continue
            raw_codes = value.get("codes", value.get("suspect_codes"))
            reading = value.get("reading", value.get("first_reading"))
            if raw_codes is not None and isinstance(reading, Mapping):
                try:
                    exact = tuple(validate_absolute_currents(raw_codes))
                except (TypeError, ValueError):
                    exact = None
                if exact is not None and normalized_smsr_db(reading) < minimum_smsr_db:
                    result.add(exact)
            for child in value.values():
                visit(child)
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for child in value:
                visit(child)

    visit(payload)
    return result


def run_stage(
    *,
    mode: str,
    candidates: Sequence[CandidatePoint],
    calibrator: SelectedCh1Calibrator,
    report: dict[str, Any],
    output: Path,
    target_power_mw: float | None = None,
    ceiling_seeds: Mapping[int, CeilingSeed] | None = None,
    calibration_ceiling_codes: Mapping[int, Sequence[int]] | None = None,
    retry_rejected: bool = False,
    retry_failed_calibration: bool = False,
    continue_on_calibration_failure: bool = False,
    evidence_session_id: str | None = None,
    hidden_predecessor_request: Mapping[str, Any] | None = None,
) -> bool:
    def bind_session(row: dict[str, Any]) -> dict[str, Any]:
        if evidence_session_id is not None:
            row["evidence_session_id"] = evidence_session_id
        return row

    if mode == "probe-only":
        rows = report.setdefault("probe", {}).setdefault("rows", {})
        for point in candidates:
            key = str(point.order)
            if key in rows:
                continue
            rows[key] = bind_session(calibrator.probe(point))
            probe_row = rows[key]
            reading = probe_row["reading"]
            print(
                f"probe={point.order + 1}/{len(candidates)} "
                f"target={point.target_nm:.6f}nm "
                f"wave={float(reading['wavelength_nm']):.6f}nm "
                f"power={float(reading['power_mw']):.6f}mW "
                f"SMSR={float(reading['normalized_smsr_db']):.2f}dB "
                f"qualified={int(bool(probe_row['qualified']))}",
                flush=True,
            )
            report["probe"]["summary"] = summarize_probe(
                rows, calibrator.settings.power_guard_fraction
            )
            report["updated"] = now_text()
            save_checkpoint(output, report)
        report["probe"]["summary"] = summarize_probe(
            rows, calibrator.settings.power_guard_fraction
        )
        return bool(report["probe"]["summary"]["all_points_qualified"])

    if mode == "ceiling-probe":
        rows = report.setdefault("ceiling_probe", {}).setdefault("rows", {})
        retry_history = report["ceiling_probe"].setdefault("retry_history", {})
        seeds = ceiling_seeds or {}
        for point in candidates:
            key = str(point.order)
            prior_row = rows.get(key)
            if prior_row is not None and (
                not retry_rejected or bool(prior_row.get("qualified"))
            ):
                continue
            replacement = bind_session(
                calibrator.ceiling_probe(point, seeds.get(point.order))
            )
            # Only replace the on-disk row after a complete new point result
            # exists.  If measurement is interrupted, the last checkpoint
            # still contains the prior rejected row and remains resumable.
            if prior_row is not None:
                retry_history.setdefault(key, []).append(prior_row)
            rows[key] = replacement
            row = rows[key]
            print(
                f"ceiling={point.order + 1}/{len(candidates)} "
                f"target={point.target_nm:.6f}nm "
                f"power={float(row.get('clean_ceiling_power_mw', float('nan'))):.6f}mW "
                f"qualified={int(bool(row['qualified']))}",
                flush=True,
            )
            report["ceiling_probe"]["summary"] = summarize_ceiling(
                rows, calibrator.settings.ceiling_guard_fraction
            )
            report["updated"] = now_text()
            save_checkpoint(output, report)
        report["ceiling_probe"]["summary"] = summarize_ceiling(
            rows, calibrator.settings.ceiling_guard_fraction
        )
        return bool(report["ceiling_probe"]["summary"]["all_points_qualified"])

    if mode == "sparse-sequence":
        code_source, sequence_codes = select_sequence_codes(report, candidates)
        sequence_target_power = (
            float(report.get("calibration", {}).get("target_power_mw"))
            if code_source == "calibration"
            else None
        )
        sequence = report.setdefault("sparse_sequence", {})
        recorded_hidden_request = sequence.get("hidden_predecessor_request")
        expected_hidden_request = (
            dict(hidden_predecessor_request)
            if hidden_predecessor_request is not None
            else None
        )
        if recorded_hidden_request != expected_hidden_request:
            raise ValueError("稀疏复测隐藏前驱配置与当前会话不一致")
        existing_source = sequence.get("code_source")
        if existing_source not in (None, code_source):
            raise ValueError("稀疏路径复测代码源已改变，请使用新输出文件")
        sequence["code_source"] = code_source
        rounds = sequence.setdefault("rounds", {})
        for round_index in range(calibrator.settings.sequence_rounds):
            round_rows = rounds.setdefault(str(round_index), {})
            for point in candidates:
                key = str(point.order)
                if key in round_rows:
                    continue
                predecessor_order = (point.order - 1) % len(candidates)
                use_hidden = bool(
                    hidden_predecessor_request is not None
                    and point.order
                    == int(hidden_predecessor_request["mode_order"])
                )
                round_rows[key] = bind_session(
                    calibrator.sparse_sequence_observation(
                        point,
                        sequence_codes[point.order],
                        sequence_codes[predecessor_order],
                        round_index=round_index,
                        predecessor_order=predecessor_order,
                        target_power_mw=sequence_target_power,
                        hidden_predecessor_fullband_index=(
                            int(
                                hidden_predecessor_request[
                                    "predecessor_fullband_index"
                                ]
                            )
                            if use_hidden
                            else None
                        ),
                        hidden_predecessor_codes=(
                            hidden_predecessor_request["predecessor_codes"]
                            if use_hidden
                            else None
                        ),
                        hidden_predecessor_hold_us=(
                            int(hidden_predecessor_request["hold_us"])
                            if use_hidden
                            else FIRMWARE_HIDDEN_PREDECESSOR_HOLD_US
                        ),
                    )
                )
                row = round_rows[key]
                print(
                    f"sparse_round={round_index + 1}/"
                    f"{calibrator.settings.sequence_rounds} "
                    f"point={point.order + 1}/{len(candidates)} "
                    f"wrap={int(bool(row['wraparound']))} "
                    f"hidden={int(bool(row['used_hidden_predecessor']))} "
                    f"passed={int(bool(row['passed']))}",
                    flush=True,
                )
                sequence["summary"] = summarize_sparse_sequence(
                    rounds,
                    point_count=len(candidates),
                    required_rounds=calibrator.settings.sequence_rounds,
                )
                report["final_runtime_certified"] = bool(
                    code_source == "calibration"
                    and sequence["summary"]["all_points_runtime_path_certified"]
                )
                report["updated"] = now_text()
                save_checkpoint(output, report)
        sequence["summary"] = summarize_sparse_sequence(
            rounds,
            point_count=len(candidates),
            required_rounds=calibrator.settings.sequence_rounds,
        )
        sequence["summary"]["final_certification_requires_calibration_codes"] = True
        report["final_runtime_certified"] = bool(
            code_source == "calibration"
            and sequence["summary"]["all_points_runtime_path_certified"]
        )
        return bool(report["final_runtime_certified"])

    if mode != "calibrate":
        raise ValueError(f"未知模式：{mode}")
    if target_power_mw is None:
        raise ValueError("calibrate模式缺少已认证的共同功率")
    rows = report.setdefault("calibration", {}).setdefault("rows", {})
    retry_history = report["calibration"].setdefault("retry_history", {})
    ceiling_codes_by_order = calibration_ceiling_codes or {}
    for point in candidates:
        key = str(point.order)
        prior_row = rows.get(key)
        if prior_row is not None:
            if bool(prior_row.get("passed")):
                continue
            if not retry_failed_calibration:
                if not continue_on_calibration_failure:
                    return False
                continue
        inherited_forbidden: set[tuple[int, ...]] = set()
        if prior_row is not None:
            inherited_forbidden.update(
                collect_forbidden_low_smsr_codes(
                    prior_row, calibrator.settings.minimum_smsr_db
                )
            )
        for archived in retry_history.get(key, []):
            inherited_forbidden.update(
                collect_forbidden_low_smsr_codes(
                    archived, calibrator.settings.minimum_smsr_db
                )
            )
        replacement = bind_session(
            calibrator.calibrate(
                point,
                target_power_mw,
                ceiling_codes=ceiling_codes_by_order.get(point.order),
                inherited_forbidden_codes=inherited_forbidden,
            )
        )
        # As with ceiling retries, keep the last checkpoint intact until a
        # complete replacement result exists, then archive the prior failure.
        if prior_row is not None:
            retry_history.setdefault(key, []).append(prior_row)
        rows[key] = replacement
        calibration_row = rows[key]
        print(
            f"calibrate={point.order + 1}/{len(candidates)} "
            f"target={point.target_nm:.6f}nm "
            f"passed={int(bool(calibration_row['passed']))}",
            flush=True,
        )
        report["calibration"]["target_power_mw"] = target_power_mw
        report["calibration"]["summary"] = summarize_calibration(rows)
        report["updated"] = now_text()
        save_checkpoint(output, report)
        if not bool(calibration_row["passed"]) and not continue_on_calibration_failure:
            return False
    report["calibration"]["summary"] = summarize_calibration(rows)
    return bool(report["calibration"]["summary"]["all_points_passed"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="候选JSON")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="增量审计JSON"
    )
    parser.add_argument(
        "--mode",
        choices=("probe-only", "ceiling-probe", "calibrate", "sparse-sequence"),
        default="probe-only",
        help="默认仅探测；不调参、不改表",
    )
    parser.add_argument("--probe-report", type=Path)
    parser.add_argument(
        "--operational-json",
        type=Path,
        default=DEFAULT_OPERATIONAL_JSON,
        help="仅ceiling-probe读取的历史AQ6150高功率种子",
    )
    parser.add_argument(
        "--retry-rejected",
        action="store_true",
        help="仅ceiling-probe重测已有检查点中的拒绝行；保留旧失败审计记录",
    )
    parser.add_argument(
        "--retry-failed-calibration",
        action="store_true",
        help="calibrate重测已有失败行并保留旧失败审计记录；通过行不会重测",
    )
    parser.add_argument(
        "--continue-on-calibration-failure",
        action="store_true",
        help="调试专用：单点失败后继续；默认首个失败即安全退出",
    )
    parser.add_argument("--target-power-mw", type=float)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--gpib", default="GPIB0::7::INSTR")
    parser.add_argument(
        "--connect-timeout-s", type=float, default=DEFAULT_CONNECT_TIMEOUT_S
    )
    parser.add_argument("--settle-s", type=float, default=DEFAULT_SETTLE_S)
    parser.add_argument(
        "--predecessor-settle-s", type=float, default=DEFAULT_PREDECESSOR_SETTLE_S
    )
    parser.add_argument("--minimum-smsr-db", type=float, default=DEFAULT_MIN_SMSR_DB)
    parser.add_argument(
        "--probe-branch-tolerance-pm",
        type=float,
        default=DEFAULT_PROBE_BRANCH_TOLERANCE_PM,
    )
    parser.add_argument(
        "--wavelength-tolerance-pm",
        type=float,
        default=DEFAULT_WAVELENGTH_TOLERANCE_PM,
    )
    parser.add_argument(
        "--power-guard-fraction", type=float, default=DEFAULT_POWER_GUARD_FRACTION
    )
    parser.add_argument(
        "--ceiling-guard-fraction",
        type=float,
        default=DEFAULT_CEILING_GUARD_FRACTION,
    )
    parser.add_argument(
        "--power-tolerance-fraction",
        type=float,
        default=DEFAULT_POWER_TOLERANCE_FRACTION,
    )
    parser.add_argument(
        "--power-tolerance-mw", type=float, default=DEFAULT_POWER_TOLERANCE_MW
    )
    parser.add_argument(
        "--max-phase-delta-ma", type=float, default=DEFAULT_MAX_PHASE_DELTA_MA
    )
    parser.add_argument(
        "--max-phase-step-ma", type=float, default=DEFAULT_MAX_PHASE_STEP_MA
    )
    parser.add_argument(
        "--max-soa-delta-ma", type=float, default=DEFAULT_MAX_SOA_DELTA_MA
    )
    parser.add_argument(
        "--max-soa-step-ma", type=float, default=DEFAULT_MAX_SOA_STEP_MA
    )
    parser.add_argument(
        "--ceiling-soa-step-ma", type=float, default=DEFAULT_CEILING_SOA_STEP_MA
    )
    parser.add_argument(
        "--confirmation-wavelength-span-pm",
        type=float,
        default=DEFAULT_CONFIRMATION_WAVELENGTH_SPAN_PM,
    )
    parser.add_argument(
        "--confirmation-power-span-fraction",
        type=float,
        default=DEFAULT_CONFIRMATION_POWER_SPAN_FRACTION,
    )
    parser.add_argument("--maximum-iterations", type=int, default=MAX_TUNING_ITERATIONS)
    parser.add_argument("--sequence-rounds", type=int, default=DEFAULT_SEQUENCE_ROUNDS)
    parser.add_argument(
        "--hidden-predecessor-order",
        type=int,
        action="append",
        help=(
            "仅sparse-sequence：为至多1个零基mode order插入其2001点立即前驱；"
            "例如1527.820 nm候选为order 2"
        ),
    )
    parser.add_argument(
        "--hidden-predecessor-hold-us",
        type=int,
        default=FIRMWARE_HIDDEN_PREDECESSOR_HOLD_US,
        help="隐藏前驱保持时间；当前必须严格为固件值5000 us",
    )
    return parser


def settings_from_args(args: argparse.Namespace) -> CalibrationSettings:
    settings = CalibrationSettings(
        settle_s=args.settle_s,
        predecessor_settle_s=args.predecessor_settle_s,
        minimum_smsr_db=args.minimum_smsr_db,
        probe_branch_tolerance_pm=args.probe_branch_tolerance_pm,
        wavelength_tolerance_pm=args.wavelength_tolerance_pm,
        power_guard_fraction=args.power_guard_fraction,
        ceiling_guard_fraction=args.ceiling_guard_fraction,
        power_tolerance_fraction=args.power_tolerance_fraction,
        power_tolerance_mw=args.power_tolerance_mw,
        maximum_phase_delta_ma=args.max_phase_delta_ma,
        maximum_phase_step_ma=args.max_phase_step_ma,
        maximum_soa_delta_ma=args.max_soa_delta_ma,
        maximum_soa_step_ma=args.max_soa_step_ma,
        ceiling_soa_step_ma=args.ceiling_soa_step_ma,
        confirmation_wavelength_span_pm=args.confirmation_wavelength_span_pm,
        confirmation_power_span_fraction=args.confirmation_power_span_fraction,
        maximum_iterations=args.maximum_iterations,
        sequence_rounds=args.sequence_rounds,
    )
    settings.validate()
    return settings


def execute(
    args: argparse.Namespace,
    *,
    device_factory: Callable[..., Any] = FbgLanSerial,
    meter_factory: Callable[[str], Any] = AQ6150B,
    fullband_points: Sequence[Any] | None = None,
    worker_factory: Callable[..., Any] = app.UnlimitedAccuracyWorker,
    shutter_callback: Callable[[Any, Any], bool] = safe_shutter_and_confirm,
) -> int:
    settings = settings_from_args(args)
    if (
        not math.isfinite(float(args.connect_timeout_s))
        or float(args.connect_timeout_s) <= 0.0
    ):
        raise ValueError("connect-timeout-s必须是正有限数")
    candidate_path = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _, candidates = load_candidates(candidate_path)
    candidate_sha256 = sha256_file(candidate_path)
    points = (
        tuple(fullband_points)
        if fullband_points is not None
        else app.load_fullband_accuracy_table(HERE)
    )
    fullband_sha256 = validate_and_hash_fullband_points(points)
    report = create_or_validate_report(
        output,
        candidate_path=candidate_path,
        candidate_sha256=candidate_sha256,
        point_count=len(candidates),
        settings=settings,
        fullband_sha256=fullband_sha256,
    )
    raw_hidden_orders = getattr(args, "hidden_predecessor_order", None)
    if raw_hidden_orders and args.mode != "sparse-sequence":
        raise ValueError("--hidden-predecessor-order仅适用于sparse-sequence")
    hidden_predecessor_request = build_hidden_predecessor_request(
        candidates,
        points,
        raw_hidden_orders if args.mode == "sparse-sequence" else None,
        getattr(
            args,
            "hidden_predecessor_hold_us",
            FIRMWARE_HIDDEN_PREDECESSOR_HOLD_US,
        ),
    )
    target_power_mw: float | None = None
    ceiling_seeds: dict[int, CeilingSeed] = {}
    calibration_ceiling_codes: dict[int, tuple[int, ...]] = {}
    retry_rejected = bool(getattr(args, "retry_rejected", False))
    retry_failed_calibration = bool(getattr(args, "retry_failed_calibration", False))
    continue_on_calibration_failure = bool(
        getattr(args, "continue_on_calibration_failure", False)
    )
    if retry_rejected and args.mode != "ceiling-probe":
        raise ValueError("--retry-rejected仅适用于ceiling-probe")
    if retry_failed_calibration and args.mode != "calibrate":
        raise ValueError("--retry-failed-calibration仅适用于calibrate")
    if continue_on_calibration_failure and args.mode != "calibrate":
        raise ValueError("--continue-on-calibration-failure仅适用于calibrate")
    if args.mode == "ceiling-probe":
        operational_path = args.operational_json.expanduser().resolve()
        if operational_path.exists():
            operational_sha256 = sha256_file(operational_path)
            previous_seed_hash = report["ceiling_probe"].get("operational_seed_sha256")
            if previous_seed_hash not in (None, operational_sha256):
                raise ValueError("历史AQ6150高功率种子文件已改变，请使用新输出文件")
            ceiling_seeds = load_ceiling_seeds(
                operational_path,
                candidates,
                minimum_smsr_db=settings.minimum_smsr_db,
            )
            report["ceiling_probe"]["operational_seed_source"] = str(operational_path)
            report["ceiling_probe"]["operational_seed_sha256"] = operational_sha256
            report["ceiling_probe"]["historical_seed_count"] = len(ceiling_seeds)
        else:
            previous_seed_source = report["ceiling_probe"].get(
                "operational_seed_source"
            )
            if report["ceiling_probe"].get("rows") and previous_seed_source is not None:
                raise ValueError("续跑时找不到原高功率种子文件")
            report["ceiling_probe"]["operational_seed_source"] = None
            report["ceiling_probe"]["historical_seed_count"] = 0
    if args.mode == "calibrate":
        probe_path = (args.probe_report or output).expanduser().resolve()
        if not probe_path.exists():
            raise ValueError("calibrate前必须先完成probe-only")
        probe_report = json.loads(probe_path.read_text(encoding="utf-8"))
        certified_power = validate_probe_certificate(
            probe_report,
            candidate_sha256=candidate_sha256,
            point_count=len(candidates),
        )
        ceiling_rows = probe_report.get("ceiling_probe", {}).get("rows", {})
        if (
            isinstance(ceiling_rows, Mapping)
            and len(ceiling_rows) == len(candidates)
            and all(
                bool(ceiling_rows[str(i)].get("qualified"))
                for i in range(len(candidates))
            )
        ):
            calibration_ceiling_codes = {
                point.order: validate_absolute_currents(
                    ceiling_rows[str(point.order)]["ceiling_codes"]
                )
                for point in candidates
            }
            calibration_section = report["calibration"]
            calibration_section["current_seed_selection_policy"] = (
                "measure_candidate_and_ceiling_then_choose_nearest_clean_power"
            )
            if not calibration_section.get("rows"):
                calibration_section["initial_code_source"] = (
                    "per_point_measured_candidate_or_ceiling"
                )
            else:
                calibration_section["legacy_passed_rows_preserved"] = True
        else:
            report["calibration"]["initial_code_source"] = "candidate"
        target_power_mw = (
            certified_power
            if args.target_power_mw is None
            else _finite_float(args.target_power_mw, "共同目标功率")
        )
        if target_power_mw <= 0.0 or target_power_mw > certified_power + 1e-12:
            raise ValueError(
                f"共同目标功率不得超过探测认证上限{certified_power:.6f} mW"
            )
        report["calibration"]["probe_report"] = str(probe_path)
        report["calibration"]["certified_power_ceiling_mw"] = certified_power
        if isinstance(report.get("evidence_migration"), Mapping):
            from migrate_selected_ch1_candidate import reconcile_calibration_reuse

            reconcile_calibration_reuse(report, target_power_mw)

    if args.mode == "sparse-sequence" and isinstance(
        report.get("evidence_migration"), Mapping
    ):
        prior_rounds = report.get("sparse_sequence", {}).get("rounds", {})
        if isinstance(prior_rounds, Mapping) and prior_rounds:
            encoded = json.dumps(
                prior_rounds,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            report["evidence_migration"].setdefault(
                "discarded_sparse_attempts", []
            ).append(
                {
                    "discarded": now_text(),
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                    "reason": "new candidate requires one complete fresh sparse replay",
                }
            )
            report["sparse_sequence"] = {"rounds": {}, "summary": {}}
            report["final_runtime_certified"] = False

    if args.mode == "sparse-sequence":
        configure_sparse_sequence_path(report, hidden_predecessor_request)

    evidence_session_id = uuid.uuid4().hex
    session = {
        "session_id": evidence_session_id,
        "candidate_sha256": candidate_sha256,
        "mode": args.mode,
        "started": now_text(),
        "status": "starting",
        "shutter_confirmed": False,
    }
    if args.mode == "sparse-sequence":
        session["hidden_predecessor_request"] = hidden_predecessor_request
    report.setdefault("sessions", []).append(session)
    # Do not leave a stale successful shutter flag while a new hardware
    # session is in progress or after an ungraceful process termination.
    report["shutter_confirmed"] = False
    if args.mode != "sparse-sequence":
        report["final_runtime_certified"] = False
    report["updated"] = now_text()
    save_checkpoint(output, report)

    device = None
    worker = None
    meter = None
    stage_ok = False
    operation_error: BaseException | None = None
    shutter_error: BaseException | None = None
    shutter_confirmed = False
    try:
        device = device_factory(board_host=args.host, timeout=0.10, write_timeout=3.0)
        worker = worker_factory(device, (), {}, False)
        wait_connected(device, args.connect_timeout_s)
        device.write(app.build_work_mode_command(2))
        if not worker._interruptible_wait(0.12):
            raise RuntimeError("切换EXTRA模式时被中止")
        device.reset_input_buffer()
        worker._rx_buffer.clear()
        meter = meter_factory(args.gpib)
        report["meter_identity"] = str(getattr(meter, "identity", "unknown"))
        report["board_host"] = str(getattr(device, "board_ip", args.host))
        session["status"] = "running"
        calibrator = SelectedCh1Calibrator(worker, meter, points, settings)
        stage_ok = run_stage(
            mode=args.mode,
            candidates=candidates,
            calibrator=calibrator,
            report=report,
            output=output,
            target_power_mw=target_power_mw,
            ceiling_seeds=ceiling_seeds,
            calibration_ceiling_codes=calibration_ceiling_codes,
            retry_rejected=retry_rejected,
            retry_failed_calibration=retry_failed_calibration,
            continue_on_calibration_failure=continue_on_calibration_failure,
            evidence_session_id=evidence_session_id,
            hidden_predecessor_request=hidden_predecessor_request,
        )
        session["status"] = "stage_complete"
    except BaseException as exc:
        operation_error = exc
        session["status"] = (
            "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        )
        session["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        # Safety order is deliberate: shutter first, then close the meter and
        # LAN transport.  A successful stage is never reported without the
        # exact SOA CLR acknowledgement.
        if device is not None and worker is not None:
            try:
                shutter_confirmed = bool(shutter_callback(device, worker))
            except BaseException as exc:
                shutter_error = exc
                session["shutter_error"] = f"{type(exc).__name__}: {exc}"
        if meter is not None:
            try:
                meter.close()
            except Exception:
                pass
        if device is not None:
            try:
                device.close()
            except Exception:
                pass
        session["finished"] = now_text()
        session["shutter_confirmed"] = shutter_confirmed
        report["shutter_confirmed"] = shutter_confirmed
        report["updated"] = now_text()
        if operation_error is None and not shutter_confirmed:
            session["status"] = "failed"
        elif operation_error is None:
            session["status"] = "complete" if stage_ok else "completed_with_rejections"
        save_checkpoint(output, report)
        print(f"soa_shutter_confirmed={int(bool(shutter_confirmed))}", flush=True)

    if operation_error is not None:
        raise operation_error
    if not shutter_confirmed:
        detail = f"：{shutter_error}" if shutter_error is not None else ""
        raise RuntimeError(f"SOA安全关光未获精确回读确认{detail}")
    return 0 if stage_ok else 2


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return execute(args)
    except KeyboardInterrupt:
        print("calibration_interrupted=1", file=sys.stderr, flush=True)
        return 130
    except Exception as exc:
        print(f"calibration_error={exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
