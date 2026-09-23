"""Safely promote dense-scan candidates to the firmware mode tables.

The equal-interval page produces a measured candidate report.  This module is
the only place allowed to turn that report into C/YAML runtime tables: it
validates every row against the audited 2001-point source, snapshots all files
needed for rollback, writes the stress/temperature tables atomically and can
build the matching OTA package.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent
PROJECT = REPOSITORY / "JDSU"
BACKUP_ROOT = HERE / "outputs" / "mode_table_backups"
LATEST_BACKUP = BACKUP_ROOT / "latest.json"
PENDING_ROOT_NAME = "mode_table_pending"
PENDING_MARKER_NAME = "pending.json"

MODE_TABLE_FILES = (
    Path("Python/mode_tables_from_fullband_2001.json"),
    Path("Python/stress_wave_const.yaml"),
    Path("Python/wave_const.yaml"),
    Path("JDSU/Core/Inc/stress_table.h"),
    Path("JDSU/Core/Src/stress_table.c"),
    Path("JDSU/Core/Src/dac_const.c"),
    Path("JDSU/Core/Src/wave_const.c"),
)
RUNTIME_TABLE_FILES = (
    Path("Python/mode_tables_from_fullband_2001.json"),
    Path("Python/stress_wave_const.yaml"),
    Path("Python/wave_const.yaml"),
)
CODE_LIMITS = (58981, 58981, 32767, 24575, 24575)
FEEDBACK_KOHM_BY_SELECTOR = (2, 40, 5, 20)
STRESS_PATH_PRECONDITION_HOLD_US = 5000
SPARSE_DIRECT_PATH = "previous_sparse->target"
SPARSE_HIDDEN_PATH = "previous_sparse->hidden_fullband_predecessor->target"

# New reports carry their own point budgets.  Keep the legacy sizes only for
# validating already-saved v2 reports; they are not runtime defaults anymore.
SUPPORTED_CANDIDATE_SCHEMAS = frozenset(
    {
        "equal_interval_auto_mode_selection_v2",
        "equal_interval_auto_mode_selection_v3",
        "equal_interval_auto_mode_selection_v4",
    }
)
LEGACY_V2_POINT_COUNTS = {"stress": 63, "temperature": 100}
DEFAULT_MODE_POINT_COUNTS = {"stress": 45, "temperature": 99}
MINIMUM_POINTS_PER_PEAK = {"stress": 5, "temperature": 5}
# The LAN raw bridge accepts at most 2048 payload bytes.  A legacy-compatible
# scan row can occupy eight bytes (CH0..CH3), so 240 rows leave space for frame
# headers/status while still allowing substantially denser experimental tables.
MODE_POINT_COUNT_LIMITS = {"stress": (5, 240), "temperature": (5, 240)}


def sparse_stress_table_fingerprint(rows) -> str:
    """Bind a sparse-order validation result to one exact stress table.

    The fingerprint intentionally excludes presentation-only fit metadata.  A
    validation remains valid only while the ordered 2001-point indices and all
    five copied PI11210 DAC codes are unchanged.
    """

    canonical = []
    for mode_index, raw in enumerate(rows or ()):
        if not isinstance(raw, dict):
            raise ValueError(f"应力第{mode_index + 1}点格式无效")
        codes = tuple(int(value) for value in raw.get("codes", ()))
        if len(codes) != 5:
            raise ValueError(f"应力第{mode_index + 1}点必须包含5个DAC码")
        canonical.append(
            {
                "mode_index": int(mode_index),
                "peak_number": int(raw.get("peak_number", 0)),
                "fullband_index": int(raw["fullband_index"]),
                "codes": list(codes),
            }
        )
    encoded = json.dumps(
        canonical,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ModeTableInstallResult:
    backup_dir: Path
    report_path: Path
    detected_peak_count: int
    stress_segment_starts: tuple[int, ...]
    temperature_segment_starts: tuple[int, ...]
    stress_point_count: int
    temperature_point_count: int


@dataclass(frozen=True)
class ModeTableBuildResult:
    action: str
    package_path: Path
    version: str
    detected_peak_count: int
    backup_dir: Path | None
    pending_runtime_dir: Path | None
    stress_point_count: int
    temperature_point_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _same_path(first: str | Path, second: str | Path) -> bool:
    try:
        return os.path.samefile(first, second)
    except OSError:
        return os.path.normcase(os.path.abspath(first)) == os.path.normcase(
            os.path.abspath(second)
        )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _wave_pair(wavelength_nm: float) -> tuple[int, int]:
    total_pm = int(round(float(wavelength_nm) * 1000.0))
    return total_pm // 1000, total_pm % 1000


def validate_mode_point_budget(
    name: str,
    point_count: int,
    detected_peak_count: int,
    minimum_per_peak: int | None = None,
) -> tuple[int, int]:
    """Validate one dynamic sparse-table budget against fitting/transport limits."""
    if name not in MODE_POINT_COUNT_LIMITS:
        raise ValueError(f"不支持的模式表：{name}")
    point_count = int(point_count)
    detected_peak_count = int(detected_peak_count)
    required_minimum = MINIMUM_POINTS_PER_PEAK[name]
    if minimum_per_peak is None:
        minimum_per_peak = required_minimum
    minimum_per_peak = int(minimum_per_peak)
    if minimum_per_peak < required_minimum:
        raise ValueError(
            f"{name}每峰至少需要{required_minimum}点，"
            f"不能设为{minimum_per_peak}点"
        )
    lower, upper = MODE_POINT_COUNT_LIMITS[name]
    if not lower <= point_count <= upper:
        raise ValueError(f"{name}点数必须在{lower}～{upper}之间")
    required_total = detected_peak_count * minimum_per_peak
    if point_count < required_total:
        raise ValueError(
            f"{name}识别到{detected_peak_count}个峰，每峰至少"
            f"{minimum_per_peak}点，总点数不能少于{required_total}"
        )
    return point_count, minimum_per_peak


def _load_audited_rows(repository: Path) -> tuple[dict, ...]:
    import csv

    source = repository / "Python" / "fullband_equal_power_operational_2001.csv"
    rows = []
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "index",
            "target_wavelength_nm",
            "measured_wavelength_nm",
            "single_mode_pass",
            "gain_code",
            "soa_code",
            "phase_code",
            "wavelength_a_code",
            "wavelength_b_code",
        }
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("2001点标定表字段不完整")
        code_fields = (
            "gain_code",
            "soa_code",
            "phase_code",
            "wavelength_a_code",
            "wavelength_b_code",
        )
        for expected_index, raw in enumerate(reader):
            index = int(raw["index"])
            if index != expected_index:
                raise ValueError("2001点标定表索引不连续")
            if int(raw["single_mode_pass"]) != 1:
                raise ValueError(f"2001点标定表第{index + 1}点未通过单模验证")
            codes = []
            for field, limit in zip(code_fields, CODE_LIMITS):
                code = int(raw[field])
                if code == limit + 1:
                    code = limit
                if not 0 <= code <= limit:
                    raise ValueError(f"2001点标定表第{index + 1}点超过激光器安全上限")
                codes.append(code)
            rows.append(
                {
                    "index": index,
                    "target_wavelength_nm": float(raw["target_wavelength_nm"]),
                    "measured_wavelength_nm": float(raw["measured_wavelength_nm"]),
                    "codes": tuple(codes),
                }
            )
    if len(rows) != 2001:
        raise ValueError(f"2001点标定表应为2001点，实际为{len(rows)}点")
    return tuple(rows)


def _validate_mode_rows(
    name: str,
    raw_rows,
    audited_rows: tuple[dict, ...],
    detected_peak_count: int,
    expected_count: int,
    allowed_channels: tuple[int, ...],
    *,
    allow_hardware_calibrated_codes: bool = False,
) -> tuple[list[dict], tuple[int, ...]]:
    if not isinstance(raw_rows, list) or len(raw_rows) != expected_count:
        raise ValueError(f"{name}候选表必须完整包含{expected_count}点")

    rows = []
    previous_index = -1
    previous_wavelength = float("-inf")
    previous_peak = 0
    segment_starts = []
    for mode_index, raw in enumerate(raw_rows):
        if not isinstance(raw, dict):
            raise ValueError(f"{name}第{mode_index + 1}点格式无效")
        fullband_index = int(raw["fullband_index"])
        if not 0 <= fullband_index < len(audited_rows):
            raise ValueError(f"{name}第{mode_index + 1}点的2001点索引越界")
        if fullband_index <= previous_index:
            raise ValueError(f"{name}候选点必须按2001点索引严格递增")
        audited = audited_rows[fullband_index]
        codes = tuple(int(value) for value in raw["codes"])
        if len(codes) != 5 or any(
            not 0 <= value <= limit for value, limit in zip(codes, CODE_LIMITS)
        ):
            raise ValueError(f"{name}第{mode_index + 1}点DAC码超出安全上限")
        if not allow_hardware_calibrated_codes and codes != audited["codes"]:
            raise ValueError(f"{name}第{mode_index + 1}点DAC码与2001点标定表不一致")
        measured = float(raw["measured_wavelength_nm"])
        target = float(raw["target_wavelength_nm"])
        if not math.isfinite(measured) or not math.isfinite(target):
            raise ValueError(f"{name}第{mode_index + 1}点波长不是有限数")
        if (
            not allow_hardware_calibrated_codes
            and abs(measured - audited["measured_wavelength_nm"]) > 0.0000005
        ):
            raise ValueError(f"{name}第{mode_index + 1}点实测波长与标定表不一致")
        if abs(target - audited["target_wavelength_nm"]) > 0.0000005:
            raise ValueError(f"{name}第{mode_index + 1}点目标波长与标定表不一致")
        if measured <= previous_wavelength:
            raise ValueError(f"{name}候选波长必须严格递增")

        peak_number = int(raw["peak_number"])
        if not 1 <= peak_number <= detected_peak_count:
            raise ValueError(f"{name}第{mode_index + 1}点峰编号越界")
        if peak_number < previous_peak or peak_number > previous_peak + 1:
            raise ValueError(f"{name}峰编号必须连续且按波长排列")
        if peak_number != previous_peak:
            segment_starts.append(mode_index)
            previous_peak = peak_number

        source_channel = int(raw.get("source_channel", -1))
        if source_channel not in allowed_channels:
            allowed_text = "/".join(f"CH{channel}" for channel in allowed_channels)
            raise ValueError(
                f"{name}第{mode_index + 1}点的源通道不在{allowed_text}分析范围内"
            )

        rows.append(
            {
                **raw,
                "mode_index": mode_index,
                "fullband_index": fullband_index,
                "target_wavelength_nm": target,
                "measured_wavelength_nm": measured,
                "codes": list(codes),
                "source_channel": source_channel,
                # A sparse mode table can skip the optical state that made the
                # audited row repeatable.  Always derive (never trust/import)
                # the immediate full-band predecessor used by the complete
                # equal-interval scan and wavelength-auto single-value path.
                "precondition_codes": list(
                    audited_rows[max(0, fullband_index - 1)]["codes"]
                ),
            }
        )
        previous_index = fullband_index
        previous_wavelength = measured

    if tuple(range(1, detected_peak_count + 1)) != tuple(
        sorted({int(row["peak_number"]) for row in rows})
    ):
        raise ValueError(f"{name}候选表没有覆盖全部{detected_peak_count}个峰")
    if len(segment_starts) != detected_peak_count:
        raise ValueError(f"{name}波段数与识别峰数不一致")
    return rows, tuple(segment_starts)


def _validate_sparse_sequence_result(
    raw_validation,
    stress_rows: list[dict],
) -> dict:
    """Require a replay of the exact sparse order before a v4 table is applied."""

    if not isinstance(raw_validation, dict):
        raise ValueError("实测模板候选表缺少稀疏顺序验证状态")
    expected_fingerprint = sparse_stress_table_fingerprint(stress_rows)
    recorded_fingerprint = str(
        raw_validation.get("stress_table_fingerprint_sha256", "")
    ).strip().lower()
    if recorded_fingerprint != expected_fingerprint:
        raise ValueError("稀疏顺序验证与当前45点表指纹不匹配")
    status = str(raw_validation.get("status", "pending")).strip().lower()
    if status != "passed":
        raise ValueError(
            "当前45点候选表尚未通过实际稀疏顺序复现验证，"
            "已保留旧正式点表"
        )
    validation_method = str(
        raw_validation.get("validation_method", "")
    ).strip().lower()
    if validation_method != "hardware_sparse_order_replay":
        raise ValueError("45点顺序必须由真实硬件稀疏复现验证")
    validated_point_count = int(raw_validation.get("validated_point_count", 0))
    if validated_point_count != len(stress_rows):
        raise ValueError(
            f"稀疏顺序只验证了{validated_point_count}/{len(stress_rows)}点"
        )
    validated_repeats = int(raw_validation.get("validated_repeats", 0))
    if validated_repeats < 1:
        raise ValueError("稀疏顺序验证至少需要一轮完整复现")
    failed = tuple(int(value) for value in raw_validation.get("failed_mode_indices", ()))
    if failed:
        raise ValueError(
            "稀疏顺序仍有未通过点："
            + ",".join(str(value) for value in failed)
        )
    expected_fullband_indices = tuple(
        int(row["fullband_index"]) for row in stress_rows
    )
    validated_fullband_indices = tuple(
        int(value)
        for value in raw_validation.get("validated_fullband_indices", ())
    )
    if validated_fullband_indices != expected_fullband_indices:
        raise ValueError("45点稀疏复现的扫描顺序与候选表不一致")

    raw_point_results = raw_validation.get("per_point_results", ())
    if not isinstance(raw_point_results, (list, tuple)) or len(
        raw_point_results
    ) != len(stress_rows):
        raise ValueError("45点稀疏复现缺少逐点验证结果")
    required_hidden_modes = set()
    normalized_point_results = []
    for mode_index, (raw_point, row) in enumerate(
        zip(raw_point_results, stress_rows, strict=True)
    ):
        if not isinstance(raw_point, dict):
            raise ValueError("稀疏复现逐点验证结果格式无效")
        if int(raw_point.get("mode_index", -1)) != mode_index:
            raise ValueError("稀疏复现逐点验证顺序不一致")
        if int(raw_point.get("fullband_index", -1)) != int(
            row["fullband_index"]
        ):
            raise ValueError("稀疏复现逐点验证的2001点索引不一致")
        if raw_point.get("passed") is not True:
            raise ValueError(f"稀疏复现第{mode_index + 1}点未通过")
        requires_hidden = bool(
            raw_point.get("requires_hidden_predecessor", False)
        )
        if requires_hidden:
            required_hidden_modes.add(mode_index)
            if raw_point.get("runtime_path") != SPARSE_HIDDEN_PATH:
                raise ValueError("隐藏前驱点没有认证完整prev->hidden->target路径")
        elif raw_point.get("runtime_path", SPARSE_DIRECT_PATH) != SPARSE_DIRECT_PATH:
            raise ValueError("普通稀疏点的运行路径标记无效")
        normalized_point_results.append({
            **raw_point,
            "mode_index": mode_index,
            "fullband_index": int(row["fullband_index"]),
            "passed": True,
            "requires_hidden_predecessor": requires_hidden,
        })

    hidden = raw_validation.get("certified_hidden_predecessors", ())
    if not isinstance(hidden, (list, tuple)):
        raise ValueError("隐藏前驱认证列表格式无效")
    # The existing real-time firmware has one exceptional precondition slot.
    # Refuse to silently turn a 45-point frame into a predecessor-per-point
    # scan, which would destroy the 15 Hz target.
    if len(hidden) > 1:
        raise ValueError("当前快速固件最多只允许1个已认证隐藏前驱")
    normalized_hidden = []
    for raw in hidden:
        if not isinstance(raw, dict) or raw.get("certified") is not True:
            raise ValueError("隐藏前驱必须有明确的实测认证标记")
        mode_index = int(raw.get("mode_index", -1))
        if not 0 <= mode_index < len(stress_rows):
            raise ValueError("隐藏前驱的应力表索引越界")
        row = stress_rows[mode_index]
        fullband_index = int(raw.get("fullband_index", -1))
        if fullband_index != int(row["fullband_index"]):
            raise ValueError("隐藏前驱认证的2001点索引不匹配")
        predecessor_index = int(raw.get("predecessor_fullband_index", -1))
        if predecessor_index != max(0, fullband_index - 1):
            raise ValueError("隐藏前驱必须是该目标在2001点表中的直接前一点")
        predecessor_codes = tuple(
            int(value) for value in raw.get("predecessor_codes", ())
        )
        if predecessor_codes != tuple(row["precondition_codes"]):
            raise ValueError("隐藏前驱DAC码与2001点审计表不一致")
        hold_us = int(raw.get("hold_us", -1))
        if hold_us != STRESS_PATH_PRECONDITION_HOLD_US:
            raise ValueError("隐藏前驱保持时间必须与固件严格一致为5000 us")
        if raw.get("runtime_path") != SPARSE_HIDDEN_PATH:
            raise ValueError("隐藏前驱认证没有记录完整prev->hidden->target路径")
        if int(raw.get("validated_repeats", 0)) != validated_repeats:
            raise ValueError("隐藏前驱认证轮数与稀疏复测轮数不一致")
        point_hidden = normalized_point_results[mode_index].get(
            "hidden_predecessor"
        )
        if not isinstance(point_hidden, dict):
            raise ValueError("隐藏前驱点缺少逐点路径元数据")
        if (
            int(point_hidden.get("fullband_index", -1)) != predecessor_index
            or tuple(int(value) for value in point_hidden.get("codes", ()))
            != predecessor_codes
            or int(point_hidden.get("hold_us", -1)) != hold_us
        ):
            raise ValueError("隐藏前驱逐点元数据与认证列表不一致")
        normalized_hidden.append(
            {
                **raw,
                "mode_index": mode_index,
                "fullband_index": fullband_index,
                "predecessor_fullband_index": predecessor_index,
                "predecessor_codes": list(predecessor_codes),
                "hold_us": hold_us,
                "runtime_path": SPARSE_HIDDEN_PATH,
                "validated_repeats": validated_repeats,
                "certified": True,
            }
        )
    certified_hidden_modes = {
        int(item["mode_index"]) for item in normalized_hidden
    }
    if required_hidden_modes != certified_hidden_modes:
        raise ValueError("需要隐藏前驱的点必须与已认证例外逐点对应")
    return {
        **raw_validation,
        "status": "passed",
        "stress_table_fingerprint_sha256": expected_fingerprint,
        "validated_point_count": validated_point_count,
        "validated_repeats": validated_repeats,
        "validation_method": "hardware_sparse_order_replay",
        "validated_fullband_indices": list(validated_fullband_indices),
        "per_point_results": normalized_point_results,
        "failed_mode_indices": [],
        "certified_hidden_predecessors": normalized_hidden,
    }


def validate_candidate_report(report: dict, repository: Path = REPOSITORY) -> dict:
    if not isinstance(report, dict) or report.get("schema") not in SUPPORTED_CANDIDATE_SCHEMAS:
        raise ValueError("只允许应用本程序生成的完整自动选点结果")
    schema = str(report["schema"])
    if int(report.get("source_point_count", 0)) != 2001:
        raise ValueError("自动选点结果不是来自完整2001点扫描")
    detected_peak_count = int(report.get("detected_peak_count", 0))
    if not 1 <= detected_peak_count <= 12:
        raise ValueError("有效峰数必须在1～12之间")

    configured_peak_count = report.get("configured_peak_count")
    if schema in {
        "equal_interval_auto_mode_selection_v3",
        "equal_interval_auto_mode_selection_v4",
    } and configured_peak_count is None:
        raise ValueError("v3/v4自动选点结果必须声明严格期望峰数")
    if configured_peak_count is not None and int(configured_peak_count) != detected_peak_count:
        raise ValueError("实际识别峰数与严格配置峰数不一致")

    raw_channels = report.get("fbg_channels")
    if raw_channels is None:
        if schema in {
            "equal_interval_auto_mode_selection_v3",
            "equal_interval_auto_mode_selection_v4",
        }:
            raise ValueError("v3/v4自动选点结果缺少光栅分析通道")
        # v2 reports predate an explicit channel contract and may contain rows
        # selected from any of the four ADC inputs.
        allowed_channels = (0, 1, 2, 3)
    else:
        try:
            allowed_channels = tuple(int(channel) for channel in raw_channels)
        except (TypeError, ValueError):
            raise ValueError("自动选点的光栅通道配置无效") from None
        if (
            not allowed_channels
            or len(set(allowed_channels)) != len(allowed_channels)
            or any(channel < 0 or channel > 3 for channel in allowed_channels)
        ):
            raise ValueError("自动选点的光栅通道必须是不重复的CH0～CH3")
    if schema == "equal_interval_auto_mode_selection_v4":
        if detected_peak_count != 9 or int(configured_peak_count) != 9:
            raise ValueError("v4机械手指表必须严格包含CH1的9个光栅峰")
        if allowed_channels != (1,):
            raise ValueError("v4机械手指表只允许CH1光栅通道")

    analog_feedback = report.get("analog_feedback")
    if not isinstance(analog_feedback, dict):
        raise ValueError("自动选点结果缺少CH0/CH1模拟跨阻")
    feedback_selectors = []
    for channel in range(2):
        selector = int(analog_feedback.get(f"ch{channel}_selector", -1))
        if not 0 <= selector < len(FEEDBACK_KOHM_BY_SELECTOR):
            raise ValueError(f"CH{channel}模拟跨阻选择码必须为0～3")
        recorded_kohm = int(
            analog_feedback.get(
                f"ch{channel}_kohm", FEEDBACK_KOHM_BY_SELECTOR[selector]
            )
        )
        if recorded_kohm != FEEDBACK_KOHM_BY_SELECTOR[selector]:
            raise ValueError(f"CH{channel}模拟跨阻选择码与电阻值不一致")
        feedback_selectors.append(selector)
    settle_s = float(report.get("settle_s", 0.0))
    if not math.isfinite(settle_s) or not 0.0 <= settle_s <= 10.0:
        raise ValueError("自动选点的DAC后等待时间无效")

    repository = Path(repository)
    audited_rows = _load_audited_rows(repository)
    hardware_calibration_certificate = None
    if schema == "equal_interval_auto_mode_selection_v4" and report.get(
        "hardware_calibration_certificate"
    ) is not None:
        # Import lazily: the offline bridge mirrors the table fingerprint but
        # deliberately does not participate in ordinary v2/v3 selection.
        from selected_calibration_bridge import (
            validate_hardware_calibration_certificate,
        )

        hardware_calibration_certificate = (
            validate_hardware_calibration_certificate(report, repository)
        )
    normalized = dict(report)
    modes = {}
    segment_starts = {}
    for name in ("stress", "temperature"):
        raw_mode = report.get(name)
        if not isinstance(raw_mode, dict):
            raise ValueError(f"缺少{name}自动选点结果")
        declared_count = int(raw_mode.get("point_count", 0))
        if schema == "equal_interval_auto_mode_selection_v2":
            legacy_count = LEGACY_V2_POINT_COUNTS[name]
            if declared_count != legacy_count:
                raise ValueError(f"v2{name}点数必须为{legacy_count}")
        raw_minimum = raw_mode.get(
            "minimum_points_per_peak", MINIMUM_POINTS_PER_PEAK[name]
        )
        expected, minimum_per_peak = validate_mode_point_budget(
            name,
            declared_count,
            detected_peak_count,
            raw_minimum,
        )
        rows, starts = _validate_mode_rows(
            name,
            raw_mode.get("rows"),
            audited_rows,
            detected_peak_count,
            expected,
            allowed_channels,
            allow_hardware_calibrated_codes=bool(
                name == "stress" and hardware_calibration_certificate is not None
            ),
        )
        counts = tuple(int(value) for value in raw_mode.get("points_per_peak", ()))
        actual_counts = tuple(
            sum(int(row["peak_number"]) == peak for row in rows)
            for peak in range(1, detected_peak_count + 1)
        )
        if (
            len(counts) != detected_peak_count
            or any(count < minimum_per_peak for count in counts)
            or counts != actual_counts
            or sum(counts) != expected
        ):
            raise ValueError(f"{name}每峰点数与候选行不一致")
        modes[name] = {
            **raw_mode,
            "point_count": expected,
            "minimum_points_per_peak": minimum_per_peak,
            "rows": rows,
            "points_per_peak": list(counts),
        }
        segment_starts[name] = starts
    normalized["stress"] = modes["stress"]
    normalized["temperature"] = modes["temperature"]
    normalized["segment_starts"] = segment_starts
    normalized["fbg_channels"] = allowed_channels
    normalized["temperature_feedback_selectors"] = tuple(feedback_selectors)
    # Temperature acquisition is intentionally accuracy-first.  It may use a
    # longer delay than the source scan, but never less than the established
    # 12 ms optical settling interval.
    normalized["temperature_settle_us"] = max(
        12_000, min(1_000_000, int(round(settle_s * 1_000_000.0)))
    )
    if schema == "equal_interval_auto_mode_selection_v4":
        normalized["sparse_sequence_validation"] = _validate_sparse_sequence_result(
            report.get("sparse_sequence_validation"),
            modes["stress"]["rows"],
        )
        if hardware_calibration_certificate is not None:
            normalized["hardware_calibration_certificate"] = (
                hardware_calibration_certificate
            )
    return normalized


def _stress_header_text(
    stress_starts: tuple[int, ...],
    temperature_starts: tuple[int, ...],
    stress_point_count: int,
    temperature_point_count: int,
    temperature_feedback_selectors: tuple[int, int] = (1, 1),
    temperature_settle_us: int = 12000,
    stress_path_precondition_index: int | None = None,
) -> str:
    if stress_path_precondition_index is None:
        stress_path_precondition_index = int(stress_point_count)
    return f"""#ifndef __STRESS_TABLE_H
#define __STRESS_TABLE_H

#include <stdint.h>

#define STRESS_TABLE_POINT_COUNT {int(stress_point_count)}U
#define STRESS_SEGMENT_COUNT {len(stress_starts)}U
#define STRESS_LAST_SECTION_START_INDEX {stress_starts[-1]}U
#define TEMPERATURE_TABLE_POINT_COUNT {int(temperature_point_count)}U
#define TEMPERATURE_SEGMENT_COUNT {len(temperature_starts)}U
#define TEMPERATURE_FEEDBACK_SELECTOR_CH0 {temperature_feedback_selectors[0]}U
#define TEMPERATURE_FEEDBACK_SELECTOR_CH1 {temperature_feedback_selectors[1]}U
#define TEMPERATURE_ADC_SETTLE_US {int(temperature_settle_us)}U
#define TEMPERATURE_PRECONDITION_HOLD_US 5000U
#define STRESS_PATH_PRECONDITION_POINT_INDEX {int(stress_path_precondition_index)}U
#define STRESS_PATH_PRECONDITION_HOLD_US {STRESS_PATH_PRECONDITION_HOLD_US}U

extern const uint16_t Stress_Wave_DAC[STRESS_TABLE_POINT_COUNT][5];
extern const uint16_t Stress_Wave_DATA[STRESS_TABLE_POINT_COUNT][2];
extern const uint16_t Stress_Path_Precondition_DAC[5];
extern const uint16_t Temperature_Precondition_DAC[TEMPERATURE_TABLE_POINT_COUNT][5];

#endif
"""


def _stress_source_text(
    rows: list[dict],
    temperature_rows: list[dict],
    stress_path_precondition_codes: list[int],
) -> str:
    lines = [
        '#include "stress_table.h"',
        "",
        "/* Automatically selected from one complete 2001-point measured ADC",
        " * spectrum.  Every DAC row is an exact audited single-mode table row.",
        " */",
        "const uint16_t Stress_Wave_DAC[STRESS_TABLE_POINT_COUNT][5] = {",
    ]
    lines.extend(
        "{" + ",".join(str(value) for value in row["codes"]) + "},"
        for row in rows
    )
    lines.extend(["};", "", "const uint16_t Stress_Wave_DATA[STRESS_TABLE_POINT_COUNT][2] = {"])
    lines.extend("{%d,%d}," % _wave_pair(row["measured_wavelength_nm"]) for row in rows)
    lines.extend(
        [
            "};",
            "",
            "/* Immediate audited predecessor for the path-sensitive stress target.",
            " * This optical-state command is never sampled or plotted.",
            " */",
            "const uint16_t Stress_Path_Precondition_DAC[5] =",
            "{" + ",".join(str(value) for value in stress_path_precondition_codes) + "};",
            "",
            "/* Immediate audited 2001-point predecessor for every sparse",
            " * temperature target.  These rows establish the same cavity-mode",
            " * path as a complete equal-interval scan; they are never sampled.",
            " */",
            "const uint16_t Temperature_Precondition_DAC[TEMPERATURE_TABLE_POINT_COUNT][5] = {",
        ]
    )
    lines.extend(
        "{" + ",".join(str(value) for value in row["precondition_codes"]) + "},"
        for row in temperature_rows
    )
    lines.extend(["};", ""])
    return "\n".join(lines)


def _stress_path_precondition(
    rows: list[dict],
    sparse_sequence_validation: dict | None = None,
) -> tuple[int, list[int]]:
    """Return only an explicitly certified fast-mode hidden predecessor.

    ``None`` retains the v2/v3 legacy 1538.620-nm behaviour.  New v4 tables
    pass their validation object and therefore default to no hidden command;
    they can enable the single firmware exception only after sparse-order
    replay has certified the exact row and audited predecessor.
    """
    if sparse_sequence_validation is not None:
        certified = tuple(
            sparse_sequence_validation.get("certified_hidden_predecessors", ())
        )
        if not certified:
            return len(rows), [0, 0, 0, 0, 0]
        entry = certified[0]
        return int(entry["mode_index"]), list(entry["predecessor_codes"])

    # Legacy reports predate per-table sequence fingerprints.  Preserve only
    # the one historically measured exception for backward compatibility.
    target_nm = 1538.620
    index = min(
        range(len(rows)),
        key=lambda candidate: abs(
            float(rows[candidate]["measured_wavelength_nm"]) - target_nm
        ),
    )
    # A regenerated table that no longer samples this neighborhood must not
    # accidentally precondition an unrelated point.  The out-of-range index
    # disables the runtime branch while retaining a valid C initializer.
    # Only the audited 1538.620 row has evidence for this hidden predecessor.
    # A nearby sparse sample must not inherit an unnecessary 5 ms delay.
    if abs(float(rows[index]["measured_wavelength_nm"]) - target_nm) > 0.002:
        return len(rows), [0, 0, 0, 0, 0]
    return index, list(rows[index]["precondition_codes"])


def _axis_yaml_text(rows: list[dict]) -> str:
    payload = {
        "Wave_DATA": [list(_wave_pair(row["measured_wavelength_nm"])) for row in rows]
    }
    return yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)


def _replace_first_numeric_rows_text(
    text: str, width: int, replacements: list[tuple[int, ...]]
) -> str:
    row_pattern = re.compile(r"\{\s*([^{}]+?)\s*\},?")
    replacement_index = 0

    def replace(match: re.Match) -> str:
        nonlocal replacement_index
        fields = [part.strip() for part in match.group(1).split(",")]
        if len(fields) != width:
            return match.group(0)
        try:
            [int(field, 0) for field in fields]
        except ValueError:
            return match.group(0)
        if replacement_index >= len(replacements):
            return match.group(0)
        row = replacements[replacement_index]
        replacement_index += 1
        return "{" + ", ".join(str(int(value)) for value in row) + "},"

    updated = row_pattern.sub(replace, text)
    if replacement_index != len(replacements):
        raise RuntimeError(f"固件数组只替换了{replacement_index}/{len(replacements)}行")
    return updated


def _formal_report(report: dict, candidate_path: Path | None) -> dict:
    modes = {}
    for name in ("stress", "temperature"):
        mode = report[name]
        powers = [float(row.get("measured_power_mw", 0.0)) for row in mode["rows"]]
        modes[name] = {
            "summary": {
                "point_count": len(mode["rows"]),
                "fullband_candidate_count": len(
                    {int(row["fullband_index"]) for row in mode["rows"]}
                ),
                "detected_peak_count": int(report["detected_peak_count"]),
                "points_per_peak": list(mode["points_per_peak"]),
                "power_min_mw": min(powers) if powers else 0.0,
                "power_max_mw": max(powers) if powers else 0.0,
            },
            "rows": mode["rows"],
        }
    return {
        "schema": "installed_auto_mode_tables_v1",
        "source_table": "fullband_equal_power_operational_2001.csv",
        "source_selection_schema": report.get("schema"),
        "candidate_report": str(candidate_path) if candidate_path else None,
        "installed_at": datetime.now().isoformat(timespec="seconds"),
        "selection_rule": (
            "adaptive peak count from a complete measured 2001-point raw ADC scan; "
            "all DAC rows copied exactly from the audited single-mode table"
        ),
        "all_runtime_points_from_fullband_2001": True,
        "detected_peak_count": int(report["detected_peak_count"]),
        "fbg_channels": list(report.get("fbg_channels", (0, 1, 2, 3))),
        "analog_feedback": report.get("analog_feedback", {}),
        "sparse_sequence_validation": report.get(
            "sparse_sequence_validation"
        ),
        "modes": modes,
    }


def _snapshot_files(repository: Path, *, candidate_path: Path | None) -> Path:
    repository = Path(repository)
    BACKUP_ROOT_LOCAL = repository / "Python" / "outputs" / "mode_table_backups"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_dir = BACKUP_ROOT_LOCAL / stamp
    backup_dir.mkdir(parents=True, exist_ok=False)
    files = []
    for relative in MODE_TABLE_FILES:
        source = repository / relative
        if not source.is_file():
            raise FileNotFoundError(f"无法备份正式点表文件：{source}")
        destination = backup_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        files.append(
            {
                "path": relative.as_posix(),
                "sha256": _sha256(source),
                "size": source.stat().st_size,
            }
        )
    manifest = {
        "schema": "mode_table_backup_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_path": str(candidate_path) if candidate_path else None,
        "files": files,
        "state": "available",
    }
    _atomic_write_text(
        backup_dir / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    latest = BACKUP_ROOT_LOCAL / "latest.json"
    _atomic_write_text(
        latest,
        json.dumps({"backup_dir": str(backup_dir)}, ensure_ascii=False, indent=2) + "\n",
    )
    return backup_dir


def _restore_snapshot(
    backup_dir: Path,
    repository: Path,
    relative_files: tuple[Path, ...] | None = None,
) -> None:
    manifest_path = Path(backup_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "mode_table_backup_v1":
        raise ValueError("点表备份清单格式不正确")
    for entry in manifest.get("files", ()): 
        relative = Path(entry["path"])
        source = Path(backup_dir) / relative
        if not source.is_file() or _sha256(source) != entry["sha256"]:
            raise ValueError(f"点表备份损坏：{relative}")
    selected = set(relative_files) if relative_files is not None else None
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if selected is not None and relative not in selected:
            continue
        destination = Path(repository) / relative
        _atomic_write_text(destination, (Path(backup_dir) / relative).read_text(encoding="utf-8"))


def latest_mode_table_backup(repository: Path = REPOSITORY) -> Path | None:
    latest = Path(repository) / "Python" / "outputs" / "mode_table_backups" / "latest.json"
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
        path = Path(payload["backup_dir"])
        return path if (path / "manifest.json").is_file() else None
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _pending_paths(repository: Path) -> tuple[Path, Path]:
    root = Path(repository) / "Python" / "outputs" / PENDING_ROOT_NAME
    return root, root / PENDING_MARKER_NAME


def pending_mode_table_package(repository: Path = REPOSITORY) -> Path | None:
    _root, marker = _pending_paths(repository)
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        package = Path(payload["package_path"])
        return package if package.is_file() else None
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _stage_runtime_tables(
    repository: Path,
    package_path: Path,
    backup_dir: Path,
    detected_peak_count: int,
) -> Path:
    root, marker = _pending_paths(repository)
    pending_dir = root / package_path.stem
    pending_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for relative in RUNTIME_TABLE_FILES:
        source = Path(repository) / relative
        destination = pending_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        files.append(
            {
                "path": relative.as_posix(),
                "sha256": _sha256(destination),
                "size": destination.stat().st_size,
            }
        )
    _restore_snapshot(backup_dir, repository, RUNTIME_TABLE_FILES)
    payload = {
        "schema": "pending_mode_table_runtime_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "package_path": str(package_path.resolve()),
        "pending_dir": str(pending_dir.resolve()),
        "detected_peak_count": int(detected_peak_count),
        "files": files,
    }
    _atomic_write_text(marker, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return pending_dir


def activate_pending_runtime_tables(
    package_path: str | Path, repository: Path = REPOSITORY
) -> bool:
    """Commit desktop axes only after the matching package completed OTA."""
    repository = Path(repository)
    root, marker = _pending_paths(repository)
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if payload.get("schema") != "pending_mode_table_runtime_v1":
        raise ValueError("待启用桌面点表清单格式不正确")
    if not _same_path(payload["package_path"], package_path):
        return False
    pending_dir = Path(payload["pending_dir"])
    entries = payload.get("files", ())
    expected = {path.as_posix() for path in RUNTIME_TABLE_FILES}
    if {str(entry.get("path")) for entry in entries} != expected:
        raise ValueError("待启用桌面点表文件不完整")
    for entry in entries:
        relative = Path(entry["path"])
        source = pending_dir / relative
        if not source.is_file() or _sha256(source) != entry["sha256"]:
            raise ValueError(f"待启用桌面点表损坏：{relative}")
    for entry in entries:
        relative = Path(entry["path"])
        _atomic_write_text(
            repository / relative,
            (pending_dir / relative).read_text(encoding="utf-8"),
        )
    try:
        marker.unlink()
    except OSError:
        pass
    return True


def apply_candidate_mode_tables(
    report: dict,
    *,
    candidate_path: Path | None = None,
    repository: Path = REPOSITORY,
) -> ModeTableInstallResult:
    repository = Path(repository)
    report = validate_candidate_report(report, repository)
    stress_rows = report["stress"]["rows"]
    temperature_rows = report["temperature"]["rows"]
    stress_starts = tuple(report["segment_starts"]["stress"])
    temperature_starts = tuple(report["segment_starts"]["temperature"])
    stress_point_count = len(stress_rows)
    temperature_point_count = len(temperature_rows)
    feedback_selectors = tuple(report["temperature_feedback_selectors"])
    temperature_settle_us = int(report["temperature_settle_us"])
    stress_path_index, stress_path_codes = _stress_path_precondition(
        stress_rows,
        report.get("sparse_sequence_validation")
        if report.get("schema") == "equal_interval_auto_mode_selection_v4"
        else None,
    )
    dac_path = repository / "JDSU/Core/Src/dac_const.c"
    wave_path = repository / "JDSU/Core/Src/wave_const.c"
    rendered = {
        Path("JDSU/Core/Inc/stress_table.h"): _stress_header_text(
            stress_starts,
            temperature_starts,
            stress_point_count,
            temperature_point_count,
            feedback_selectors,
            temperature_settle_us,
            stress_path_index,
        ),
        Path("JDSU/Core/Src/stress_table.c"): _stress_source_text(
            stress_rows, temperature_rows, stress_path_codes
        ),
        Path("JDSU/Core/Src/dac_const.c"): _replace_first_numeric_rows_text(
            dac_path.read_text(encoding="utf-8"),
            5,
            [tuple(row["codes"]) for row in temperature_rows],
        ),
        Path("JDSU/Core/Src/wave_const.c"): _replace_first_numeric_rows_text(
            wave_path.read_text(encoding="utf-8"),
            2,
            [_wave_pair(row["measured_wavelength_nm"]) for row in temperature_rows],
        ),
        Path("Python/stress_wave_const.yaml"): _axis_yaml_text(stress_rows),
        Path("Python/wave_const.yaml"): _axis_yaml_text(temperature_rows),
        Path("Python/mode_tables_from_fullband_2001.json"): json.dumps(
            _formal_report(report, candidate_path),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    }

    backup_dir = _snapshot_files(repository, candidate_path=candidate_path)
    try:
        for relative, text in rendered.items():
            _atomic_write_text(repository / relative, text)
    except Exception:
        _restore_snapshot(backup_dir, repository)
        raise
    return ModeTableInstallResult(
        backup_dir=backup_dir,
        report_path=repository / "Python/mode_tables_from_fullband_2001.json",
        detected_peak_count=int(report["detected_peak_count"]),
        stress_segment_starts=stress_starts,
        temperature_segment_starts=temperature_starts,
        stress_point_count=stress_point_count,
        temperature_point_count=temperature_point_count,
    )


def _next_ota_version(repository: Path) -> str:
    release = Path(repository) / "outputs" / "ota"
    versions = []
    for path in release.glob("JDSU_F205RE_v*.fbgfw"):
        match = re.fullmatch(r"JDSU_F205RE_v(\d+)\.(\d+)\.(\d+)\.fbgfw", path.name)
        if match:
            value = tuple(int(part) for part in match.groups())
            if value[0] <= 0xFFFF and value[1] <= 0xFF and value[2] <= 0xFF:
                versions.append(value)
    major, minor, patch = max(versions, default=(1, 0, 0))
    patch += 1
    if patch > 0xFF:
        patch = 0
        minor += 1
    if minor > 0xFF:
        minor = 0
        major += 1
    if major > 0xFFFF:
        raise ValueError("OTA版本号空间已用完")
    return f"{major}.{minor}.{patch}"


def build_mode_table_ota(
    *, repository: Path = REPOSITORY, version: str | None = None, timeout_s: int = 180
) -> tuple[Path, str]:
    repository = Path(repository)
    version = version or _next_ota_version(repository)
    script = repository / "JDSU" / "Bootloader" / "build_ota_release.ps1"
    command = (
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
        "-Version",
        version,
    )
    completed = subprocess.run(
        command,
        cwd=str(repository),
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout_s,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode != 0:
        output = (completed.stdout + "\n" + completed.stderr).strip()
        raise RuntimeError("固件编译或OTA打包失败：\n" + output[-4000:])
    package_path = repository / "outputs" / "ota" / f"JDSU_F205RE_v{version}.fbgfw"
    if not package_path.is_file():
        raise RuntimeError("编译完成但没有找到生成的OTA固件包")
    from fbg_ota_protocol import load_package

    package = load_package(package_path)
    if package.version_text != version:
        raise RuntimeError("OTA包版本回读与本次构建版本不一致")
    return package_path, version


def apply_candidate_and_build(
    report: dict,
    *,
    candidate_path: Path | None = None,
    repository: Path = REPOSITORY,
) -> ModeTableBuildResult:
    pending = pending_mode_table_package(repository)
    if pending is not None:
        raise ValueError(
            f"已有尚未完成OTA的新点表包：{pending.name}；"
            "请先上传该包或恢复上一版点表"
        )
    install = apply_candidate_mode_tables(
        report, candidate_path=candidate_path, repository=repository
    )
    try:
        package_path, version = build_mode_table_ota(repository=repository)
        pending_runtime_dir = _stage_runtime_tables(
            Path(repository),
            package_path,
            install.backup_dir,
            install.detected_peak_count,
        )
    except Exception:
        _restore_snapshot(install.backup_dir, Path(repository))
        raise
    return ModeTableBuildResult(
        action="apply",
        package_path=package_path,
        version=version,
        detected_peak_count=install.detected_peak_count,
        backup_dir=install.backup_dir,
        pending_runtime_dir=pending_runtime_dir,
        stress_point_count=install.stress_point_count,
        temperature_point_count=install.temperature_point_count,
    )


def restore_previous_and_build(
    *, repository: Path = REPOSITORY
) -> ModeTableBuildResult:
    repository = Path(repository)
    backup_dir = latest_mode_table_backup(repository)
    if backup_dir is None:
        raise ValueError("没有可恢复的上一版正式点表")
    current_snapshot = _snapshot_files(repository, candidate_path=None)
    # The temporary snapshot above becomes the latest pointer; put the intended
    # previous backup back while this restore is in progress.
    latest = repository / "Python" / "outputs" / "mode_table_backups" / "latest.json"
    _atomic_write_text(latest, json.dumps({"backup_dir": str(backup_dir)}, indent=2) + "\n")
    try:
        _restore_snapshot(backup_dir, repository)
        package_path, version = build_mode_table_ota(repository=repository)
    except Exception:
        _restore_snapshot(current_snapshot, repository)
        raise
    try:
        latest.unlink()
    except OSError:
        pass
    _pending_root, pending_marker = _pending_paths(repository)
    try:
        pending_marker.unlink()
    except OSError:
        pass
    formal = json.loads(
        (repository / "Python/mode_tables_from_fullband_2001.json").read_text(
            encoding="utf-8"
        )
    )
    detected = int(
        formal.get("detected_peak_count")
        or len(formal.get("modes", {}).get("stress", {}).get("summary", {}).get("points_per_peak", ()))
        or 9
    )
    stress_point_count = int(
        formal.get("modes", {}).get("stress", {}).get("summary", {}).get(
            "point_count", 0
        )
    )
    temperature_point_count = int(
        formal.get("modes", {}).get("temperature", {}).get("summary", {}).get(
            "point_count", 0
        )
    )
    if stress_point_count <= 0 or temperature_point_count <= 0:
        raise ValueError("已恢复点表报告缺少动态点数")
    return ModeTableBuildResult(
        action="restore",
        package_path=package_path,
        version=version,
        detected_peak_count=detected,
        backup_dir=backup_dir,
        pending_runtime_dir=None,
        stress_point_count=stress_point_count,
        temperature_point_count=temperature_point_count,
    )


__all__ = [
    "ModeTableBuildResult",
    "ModeTableInstallResult",
    "activate_pending_runtime_tables",
    "apply_candidate_and_build",
    "apply_candidate_mode_tables",
    "build_mode_table_ota",
    "latest_mode_table_backup",
    "pending_mode_table_package",
    "restore_previous_and_build",
    "sparse_stress_table_fingerprint",
    "validate_mode_point_budget",
    "validate_candidate_report",
]
