"""Promote a certified 45-point CH1 calibration into a v4 mode candidate.

The hardware calibrator deliberately cannot install a runtime table.  This
module is the audited bridge between its JSON checkpoint and the ordinary
``equal_interval_auto_mode_selection_v4`` installer input.  It performs no
device I/O and refuses to emit a report unless all of the following bind to
the same ordered table:

* the strict G1..G9, five-points-per-peak candidate (or its audited migration)
  and its SHA256;
* all 45 final calibration rows and both independent confirmations;
* at least two complete real sparse-order replay rounds, including wraparound;
* the exact five final PI11210 DAC codes used by those replay rounds; and
* the calibrator's final exact-shutter acknowledgement.

The generated v4 report keeps the temperature table from the dense scan and
replaces only the stress table.  The source checkpoint and candidate remain
external immutable evidence, referenced by repository-relative paths and
SHA256 digests.  :mod:`mode_table_manager` reopens and revalidates both files
before accepting hardware-calibrated stress codes.
"""

from __future__ import annotations

import argparse
import bisect
import copy
import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent
BRIDGE_SCHEMA = "fbg_selected_ch1_hardware_certificate/v1"
CALIBRATION_SCHEMA = "fbg_selected_ch1_calibration_v1"
SELECTION_SCHEMA = "equal_interval_auto_mode_selection_v4"
CANDIDATE_MIGRATION_SCHEMA = "fbg_selected_ch1_candidate_migration/v1"
EVIDENCE_MIGRATION_SCHEMA = "fbg_selected_ch1_evidence_migration/v1"
EXPECTED_GROUPS = (
    (118, 122, 141, 170, 178),
    (333, 341, 357, 365, 386),
    (538, 551, 567, 584, 589),
    (724, 750, 752, 764, 784),
    (943, 957, 960, 961, 997),
    (1110, 1123, 1137, 1158, 1174),
    (1307, 1313, 1334, 1356, 1362),
    (1503, 1524, 1539, 1547, 1569),
    (1721, 1735, 1744, 1757, 1776),
)
EXPECTED_ROLES = (
    "left_baseline",
    "left_shift_slope",
    "amplitude_peak",
    "right_shift_slope",
    "right_baseline",
)
CODE_LIMITS = (58981, 58981, 32767, 24575, 24575)
FIRMWARE_HIDDEN_PREDECESSOR_HOLD_US = 5000
SPARSE_DIRECT_PATH = "previous_sparse->target"
SPARSE_HIDDEN_PATH = "previous_sparse->hidden_fullband_predecessor->target"
DUAL_MEASURED_SEED_CONTROLLER_REVISION = (
    "dual_measured_seed_staged_slope_v3_internal_1pct_1pm_smsr_latch"
)


@dataclass(frozen=True)
class CandidateRow:
    order: int
    peak_number: int
    within_peak_order: int
    fullband_index: int
    role: str
    target_nm: float
    codes: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class CertifiedCalibration:
    rows: tuple[dict[str, Any], ...]
    candidate_sha256: str
    calibration_sha256: str
    target_power_mw: float
    validated_repeats: int
    sequence_rounds: tuple[tuple[dict[str, Any], ...], ...]
    minimum_smsr_db: float
    maximum_wavelength_error_pm: float
    minimum_power_mw: float
    maximum_power_mw: float
    hidden_predecessor: Mapping[str, Any] | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_evidence_bytes(value: Any) -> bytes:
    """Return the single canonical byte representation used for evidence reuse.

    Candidate migration deliberately compares the *entire* point object, not a
    hand-picked subset of fields.  Consequently a changed numeric spelling,
    an added field, role, order, target, source index, or DAC code all change
    the digest and force a fresh hardware measurement.
    """

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_evidence_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_evidence_bytes(value)).hexdigest()


def _finite(value: object, label: str) -> float:
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
    numeric = _finite(value, label)
    if numeric != math.trunc(numeric):
        raise ValueError(f"{label}必须是整数")
    return int(numeric)


def _codes(value: object, label: str) -> tuple[int, int, int, int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label}必须包含5路DAC码")
    result = tuple(_exact_int(item, f"{label}[{i}]") for i, item in enumerate(value))
    if len(result) != 5:
        raise ValueError(f"{label}必须包含5路DAC码")
    for channel, (code, limit) in enumerate(zip(result, CODE_LIMITS, strict=True)):
        if not 0 <= code <= limit:
            raise ValueError(f"{label}第{channel + 1}路超过安全上限")
    return result  # type: ignore[return-value]


def sparse_stress_table_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    canonical = []
    for mode_index, raw in enumerate(rows):
        canonical.append(
            {
                "mode_index": mode_index,
                "peak_number": _exact_int(raw.get("peak_number", 0), "peak_number"),
                "fullband_index": _exact_int(raw["fullband_index"], "fullband_index"),
                "codes": list(_codes(raw["codes"], "codes")),
            }
        )
    encoded = json.dumps(
        canonical, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _repository_file(repository: Path, raw: object, label: str) -> Path:
    repository = repository.resolve()
    text = str(raw or "").strip()
    if not text:
        raise ValueError(f"{label}缺少文件路径")
    path = Path(text)
    path = path.resolve() if path.is_absolute() else (repository / path).resolve()
    try:
        path.relative_to(repository)
    except ValueError as exc:
        raise ValueError(f"{label}必须位于当前项目内") from exc
    if not path.is_file():
        raise ValueError(f"{label}文件不存在：{path}")
    return path


def _relative_path(path: Path, repository: Path) -> str:
    try:
        return path.resolve().relative_to(repository.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("证书仅允许引用当前项目内的文件") from exc


def _load_audited_rows(repository: Path) -> tuple[dict[str, Any], ...]:
    source = repository / "Python" / "fullband_equal_power_operational_2001.csv"
    if not source.is_file():
        raise ValueError("找不到2001点审计CSV")
    rows: list[dict[str, Any]] = []
    fields = (
        "gain_code",
        "soa_code",
        "phase_code",
        "wavelength_a_code",
        "wavelength_b_code",
    )
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        for expected, raw in enumerate(csv.DictReader(stream)):
            index = _exact_int(raw.get("index"), "2001点索引")
            if index != expected:
                raise ValueError("2001点审计表索引不连续")
            if _exact_int(raw.get("single_mode_pass"), "single_mode_pass") != 1:
                raise ValueError(f"2001点审计表第{index + 1}点未通过单模")
            values = []
            for channel, field in enumerate(fields):
                value = _exact_int(raw.get(field), field)
                if value == CODE_LIMITS[channel] + 1:
                    value = CODE_LIMITS[channel]
                values.append(value)
            rows.append(
                {
                    "index": index,
                    "target_nm": _finite(raw.get("target_wavelength_nm"), "目标波长"),
                    "measured_nm": _finite(
                        raw.get("measured_wavelength_nm"), "实测波长"
                    ),
                    "codes": _codes(values, "2001点DAC"),
                }
            )
    if len(rows) != 2001:
        raise ValueError(f"2001点审计表实际只有{len(rows)}点")
    return tuple(rows)


def _raw_candidate_points(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    peaks = payload.get("peaks")
    if not isinstance(peaks, list):
        raise ValueError("候选缺少峰列表")
    points: list[Mapping[str, Any]] = []
    for peak in peaks:
        if not isinstance(peak, Mapping) or not isinstance(peak.get("points"), list):
            raise ValueError("候选峰结构无效")
        for point in peak["points"]:
            if not isinstance(point, Mapping):
                raise ValueError("候选点结构无效")
            points.append(point)
    return tuple(points)


def _declared_orders(raw: object, label: str) -> tuple[int, ...]:
    if not isinstance(raw, list):
        raise ValueError(f"{label}必须是顺序数组")
    orders = tuple(_exact_int(value, label) for value in raw)
    if tuple(sorted(set(orders))) != orders or any(
        not 0 <= value < 45 for value in orders
    ):
        raise ValueError(f"{label}必须是0～44内严格递增且不重复的顺序")
    return orders


def _candidate_migration_context(
    payload: Mapping[str, Any],
    candidate_path: Path,
    repository: Path,
) -> dict[str, Any] | None:
    raw = payload.get("candidate_migration")
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or raw.get("schema") != CANDIDATE_MIGRATION_SCHEMA:
        raise ValueError("候选迁移声明格式无效")
    if any(
        raw.get(name) is not True
        for name in (
            "changed_points_require_ceiling_remeasurement",
            "changed_points_require_calibration_remeasurement",
            "full_sparse_sequence_requires_remeasurement",
        )
    ):
        raise ValueError("候选迁移不得放宽重测要求")

    base_path = _repository_file(
        repository, raw.get("base_candidate_path"), "迁移基准候选"
    )
    if base_path == candidate_path.resolve():
        raise ValueError("候选迁移不得自引用")
    base_sha = str(raw.get("base_candidate_sha256", "")).lower()
    if base_sha != _sha256(base_path):
        raise ValueError("迁移基准候选SHA256已失效")
    base_payload, _ = _load_candidate(base_path, repository, allow_migration=False)
    for before_peak, after_peak in zip(
        base_payload["peaks"], payload["peaks"], strict=True
    ):
        before_header = {
            key: value for key, value in before_peak.items() if key != "points"
        }
        after_header = {
            key: value for key, value in after_peak.items() if key != "points"
        }
        if canonical_evidence_bytes(before_header) != canonical_evidence_bytes(
            after_header
        ):
            raise ValueError("候选迁移不得改变G1～G9峰级身份或顺序")
    base_points = _raw_candidate_points(base_payload)
    points = _raw_candidate_points(payload)
    if len(base_points) != 45 or len(points) != 45:
        raise ValueError("候选迁移前后都必须恰好包含45点")

    actual_changed = tuple(
        order
        for order, (before, after) in enumerate(zip(base_points, points, strict=True))
        if canonical_evidence_bytes(before) != canonical_evidence_bytes(after)
    )
    changed = _declared_orders(raw.get("changed_orders"), "changed_orders")
    unchanged = _declared_orders(raw.get("unchanged_orders"), "unchanged_orders")
    if not actual_changed or changed != actual_changed:
        raise ValueError("迁移声明的变更点与候选逐点字节差异不一致")
    if unchanged != tuple(order for order in range(45) if order not in changed):
        raise ValueError("迁移声明的未变点不是变更点的严格补集")

    before_digests = raw.get("base_point_sha256_by_order")
    after_digests = raw.get("new_point_sha256_by_order")
    if not isinstance(before_digests, Mapping) or not isinstance(
        after_digests, Mapping
    ):
        raise ValueError("候选迁移缺少逐点规范字节指纹")
    expected_keys = {str(order) for order in range(45)}
    if set(before_digests) != expected_keys or set(after_digests) != expected_keys:
        raise ValueError("候选迁移逐点指纹必须完整覆盖45点")
    for order, (before, after) in enumerate(zip(base_points, points, strict=True)):
        if str(before_digests[str(order)]).lower() != canonical_evidence_sha256(before):
            raise ValueError("迁移基准点规范字节指纹不一致")
        if str(after_digests[str(order)]).lower() != canonical_evidence_sha256(after):
            raise ValueError("迁移新点规范字节指纹不一致")
        if order in changed:
            for immutable in ("plan_order", "within_peak_order", "role"):
                if before.get(immutable) != after.get(immutable):
                    raise ValueError(f"候选迁移不得改变{immutable}")

    base_report_path = _repository_file(
        repository, raw.get("base_report_path"), "迁移基准实测报告"
    )
    base_report_sha = str(raw.get("base_report_sha256", "")).lower()
    if base_report_sha != _sha256(base_report_path):
        raise ValueError("迁移基准实测报告SHA256已失效")
    return {
        "raw": dict(raw),
        "base_candidate_path": base_path,
        "base_candidate_sha256": base_sha,
        "base_candidate_payload": base_payload,
        "base_report_path": base_report_path,
        "base_report_sha256": base_report_sha,
        "changed_orders": changed,
        "unchanged_orders": unchanged,
        "base_points": base_points,
        "new_points": points,
    }


def _load_candidate(
    candidate_path: Path,
    repository: Path,
    *,
    allow_migration: bool = True,
) -> tuple[dict[str, Any], tuple[CandidateRow, ...]]:
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("候选JSON根结点必须是对象")
    if payload.get("schema") != "ch1_hardware_ceiling_constrained_candidate/v1":
        raise ValueError("候选JSON格式无效")
    for key, expected in (
        ("channel", 1),
        ("expected_fbg_count", 9),
        ("points_per_peak", 5),
        ("total_points", 45),
    ):
        if _exact_int(payload.get(key), key) != expected:
            raise ValueError(f"候选{key}必须为{expected}")
    peaks = payload.get("peaks")
    if not isinstance(peaks, list) or len(peaks) != 9:
        raise ValueError("候选必须严格包含G1～G9")
    audited = _load_audited_rows(repository)
    result: list[CandidateRow] = []
    groups: list[tuple[int, ...]] = []
    for peak_number, peak in enumerate(peaks, start=1):
        if not isinstance(peak, Mapping):
            raise ValueError(f"G{peak_number}候选结构无效")
        declared = _exact_int(peak.get("peak_number", peak_number), "peak_number")
        if declared != peak_number:
            raise ValueError("候选峰必须按G1～G9排列")
        points = peak.get("points")
        if not isinstance(points, list) or len(points) != 5:
            raise ValueError(f"G{peak_number}必须恰好包含5点")
        group: list[int] = []
        for within, point in enumerate(points):
            if not isinstance(point, Mapping):
                raise ValueError(f"G{peak_number}第{within + 1}点无效")
            order = _exact_int(point.get("plan_order", len(result)), "plan_order")
            if order != len(result):
                raise ValueError("候选45点plan_order必须连续")
            declared_within = _exact_int(
                point.get("within_peak_order", within), "within_peak_order"
            )
            if declared_within != within:
                raise ValueError(f"G{peak_number}峰内顺序不连续")
            role = str(point.get("role", ""))
            if role != EXPECTED_ROLES[within]:
                raise ValueError(f"G{peak_number}第{within + 1}点角色不正确")
            index = _exact_int(
                point.get("source_index", point.get("fullband_index")),
                "source_index",
            )
            if not 0 <= index < len(audited):
                raise ValueError("候选索引越界")
            row = audited[index]
            target = _finite(point.get("target_wavelength_nm"), "候选目标波长")
            if abs(target - float(row["target_nm"])) > 5e-7:
                raise ValueError(f"候选索引{index}的目标波长与审计表不一致")
            codes = _codes(point.get("dac_codes", point.get("codes")), "候选DAC")
            if codes != row["codes"]:
                raise ValueError(f"候选索引{index}的DAC不是2001点审计原始行")
            result.append(
                CandidateRow(order, peak_number, within, index, role, target, codes)
            )
            group.append(index)
        groups.append(tuple(group))
    flattened = tuple(index for group in groups for index in group)
    if len(set(flattened)) != 45 or tuple(sorted(flattened)) != flattened:
        raise ValueError("候选45点索引必须严格递增且不重复")
    if tuple(groups) != EXPECTED_GROUPS:
        if not allow_migration:
            raise ValueError("迁移基准候选必须是已指定的原始9×5方案")
        if _candidate_migration_context(payload, candidate_path, repository) is None:
            raise ValueError("候选索引改变但缺少可审计的候选迁移声明")
    elif payload.get("candidate_migration") is not None:
        raise ValueError("候选未发生逐点变化却声明了迁移")
    if payload.get("selected_source_indices") is not None and tuple(
        _exact_int(value, "selected_source_indices")
        for value in payload["selected_source_indices"]
    ) != tuple(row.fullband_index for row in result):
        raise ValueError("候选selected_source_indices与峰内顺序不一致")
    if (
        payload.get("writes_runtime_table") is not False
        or payload.get("writes_firmware") is not False
    ):
        raise ValueError("候选阶段不得写正式点表或固件")
    return payload, tuple(result)


def _reading_passes(
    reading: Mapping[str, Any],
    *,
    target_nm: float,
    target_power_mw: float,
    wavelength_tolerance_pm: float,
    power_tolerance_mw: float,
    minimum_smsr_db: float,
) -> bool:
    wavelength = _finite(reading.get("wavelength_nm"), "AQ6150波长")
    power = _finite(reading.get("power_mw"), "AQ6150功率")
    smsr = _finite(reading.get("normalized_smsr_db"), "AQ6150 SMSR")
    return bool(
        abs(wavelength - target_nm) * 1000.0 <= wavelength_tolerance_pm + 1e-9
        and abs(power - target_power_mw) <= power_tolerance_mw + 1e-12
        and smsr >= minimum_smsr_db
    )


def _validated_hidden_predecessor_request(
    raw: object,
    candidates: Sequence[CandidateRow],
    audited_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("隐藏前驱配置必须是单个对象")
    mode_order = _exact_int(raw.get("mode_order"), "隐藏前驱mode order")
    if not 0 <= mode_order < len(candidates):
        raise ValueError("隐藏前驱mode order越界")
    candidate = candidates[mode_order]
    fullband_index = _exact_int(raw.get("fullband_index"), "隐藏目标fullband index")
    if fullband_index != candidate.fullband_index:
        raise ValueError("隐藏前驱配置与目标候选索引不一致")
    predecessor_index = _exact_int(
        raw.get("predecessor_fullband_index"), "隐藏前驱fullband index"
    )
    expected_predecessor_index = max(0, fullband_index - 1)
    if predecessor_index != expected_predecessor_index:
        raise ValueError("隐藏前驱不是目标在2001点表中的立即前一点")
    predecessor_codes = _codes(raw.get("predecessor_codes"), "隐藏前驱DAC")
    expected_codes = _codes(
        audited_rows[expected_predecessor_index]["codes"], "2001点立即前驱DAC"
    )
    if predecessor_codes != expected_codes:
        raise ValueError("隐藏前驱DAC与2001点审计表不一致")
    hold_us = _exact_int(raw.get("hold_us"), "隐藏前驱hold_us")
    if hold_us != FIRMWARE_HIDDEN_PREDECESSOR_HOLD_US:
        raise ValueError("隐藏前驱保持时间必须与固件严格一致为5000 us")
    if raw.get("runtime_path") != SPARSE_HIDDEN_PATH:
        raise ValueError("隐藏前驱配置没有记录完整prev->hidden->target路径")
    return {
        "mode_order": mode_order,
        "fullband_index": fullband_index,
        "predecessor_fullband_index": predecessor_index,
        "predecessor_codes": list(predecessor_codes),
        "hold_us": hold_us,
        "runtime_path": SPARSE_HIDDEN_PATH,
    }


def _validated_migration_sessions(
    report: Mapping[str, Any], candidate_sha256: str
) -> dict[str, Mapping[str, Any]]:
    sessions = report.get("sessions")
    if not isinstance(sessions, list):
        raise ValueError("迁移报告缺少新候选硬件会话")
    result: dict[str, Mapping[str, Any]] = {}
    for raw in sessions:
        if not isinstance(raw, Mapping):
            raise ValueError("迁移报告硬件会话结构无效")
        session_id = str(raw.get("session_id", "")).strip()
        if not session_id:
            raise ValueError("迁移后的每个硬件会话必须带不可空的session_id")
        if session_id in result:
            raise ValueError("迁移报告硬件session_id重复")
        if str(raw.get("candidate_sha256", "")).lower() != candidate_sha256:
            raise ValueError("迁移后硬件会话没有绑定新候选SHA256")
        result[session_id] = raw
    return result


def _validate_fresh_row_session(
    row: Mapping[str, Any],
    sessions: Mapping[str, Mapping[str, Any]],
    *,
    mode: str,
    label: str,
) -> None:
    session_id = str(row.get("evidence_session_id", "")).strip()
    session = sessions.get(session_id)
    if session is None:
        raise ValueError(f"{label}不是新候选会话产生的实测证据")
    if session.get("mode") != mode:
        raise ValueError(f"{label}引用了错误模式的硬件会话")
    if (
        session.get("status") not in ("complete", "completed_with_rejections")
        or session.get("shutter_confirmed") is not True
        or not str(session.get("finished", "")).strip()
    ):
        raise ValueError(f"{label}引用的硬件会话未安全结束并确认关光")


def _validate_migrated_report_evidence(
    *,
    candidate_payload: Mapping[str, Any],
    candidate_path: Path,
    report: Mapping[str, Any],
    repository: Path,
) -> dict[str, Any] | None:
    context = _candidate_migration_context(
        candidate_payload, candidate_path, repository
    )
    raw = report.get("evidence_migration")
    if context is None:
        if raw is not None:
            raise ValueError("原始候选报告不得伪装为迁移证据")
        return None
    if not isinstance(raw, Mapping) or raw.get("schema") != EVIDENCE_MIGRATION_SCHEMA:
        raise ValueError("变更候选缺少实测证据迁移声明")

    candidate_sha = _sha256(candidate_path)
    if str(raw.get("destination_candidate_sha256", "")).lower() != candidate_sha:
        raise ValueError("迁移报告没有绑定新候选SHA256")
    for prefix, path_key, sha_key, expected_path, expected_sha in (
        (
            "source_candidate",
            "source_candidate_path",
            "source_candidate_sha256",
            context["base_candidate_path"],
            context["base_candidate_sha256"],
        ),
        (
            "source_report",
            "source_report_path",
            "source_report_sha256",
            context["base_report_path"],
            context["base_report_sha256"],
        ),
    ):
        actual_path = _repository_file(repository, raw.get(path_key), prefix)
        if (
            actual_path != expected_path
            or str(raw.get(sha_key, "")).lower() != expected_sha
        ):
            raise ValueError("迁移报告与候选迁移来源不一致")

    changed = _declared_orders(raw.get("changed_orders"), "changed_orders")
    unchanged = _declared_orders(raw.get("unchanged_orders"), "unchanged_orders")
    if changed != context["changed_orders"] or unchanged != context["unchanged_orders"]:
        raise ValueError("迁移报告的变更点集合与新候选不一致")
    if raw.get("sparse_sequence_reused") is not False:
        raise ValueError("候选变更后严禁复用任何旧sparse-sequence证据")

    source_report = json.loads(context["base_report_path"].read_text(encoding="utf-8"))
    if (
        not isinstance(source_report, Mapping)
        or source_report.get("schema") != CALIBRATION_SCHEMA
    ):
        raise ValueError("迁移来源不是CH1标定报告")
    source_candidate = source_report.get("candidate_source")
    if (
        not isinstance(source_candidate, Mapping)
        or str(source_candidate.get("sha256", "")).lower()
        != context["base_candidate_sha256"]
        or _exact_int(source_candidate.get("point_count"), "source point_count") != 45
    ):
        raise ValueError("迁移来源报告没有绑定迁移基准候选")
    if (
        source_report.get("writes_runtime_table") is not False
        or source_report.get("writes_firmware") is not False
        or source_report.get("shutter_confirmed") is not True
    ):
        raise ValueError("迁移来源报告越权或没有最终关光确认")
    source_sessions = source_report.get("sessions")
    source_last = (
        source_sessions[-1]
        if isinstance(source_sessions, list) and source_sessions
        else None
    )
    if (
        not isinstance(source_last, Mapping)
        or source_last.get("status") not in ("complete", "completed_with_rejections")
        or source_last.get("shutter_confirmed") is not True
        or not str(source_last.get("finished", "")).strip()
    ):
        raise ValueError("迁移来源报告最后一次会话没有安全结束并关光")
    if report.get("settings") != source_report.get("settings"):
        raise ValueError("迁移后标定参数改变，旧证据不得复用")
    if report.get("fullband_predecessor_sha256") != source_report.get(
        "fullband_predecessor_sha256"
    ):
        raise ValueError("迁移后2001点前驱表改变，旧证据不得复用")

    reused_ceiling = _declared_orders(
        raw.get("reused_ceiling_orders"), "reused_ceiling_orders"
    )
    reused_calibration = _declared_orders(
        raw.get("reused_calibration_orders"), "reused_calibration_orders"
    )
    unchanged_set = set(unchanged)
    if not set(reused_ceiling).issubset(unchanged_set) or not set(
        reused_calibration
    ).issubset(unchanged_set):
        raise ValueError("只有规范字节完全相同的候选点才可复用旧证据")

    source_ceiling = source_report.get("ceiling_probe", {}).get("rows", {})
    source_calibration = source_report.get("calibration", {}).get("rows", {})
    ceiling_rows = report.get("ceiling_probe", {}).get("rows", {})
    calibration_rows = report.get("calibration", {}).get("rows", {})
    if not all(
        isinstance(rows, Mapping)
        for rows in (source_ceiling, source_calibration, ceiling_rows, calibration_rows)
    ):
        raise ValueError("迁移证据行结构无效")

    for label, reused, source_rows, current_rows, digest_key in (
        (
            "ceiling",
            reused_ceiling,
            source_ceiling,
            ceiling_rows,
            "reused_ceiling_row_sha256_by_order",
        ),
        (
            "calibration",
            reused_calibration,
            source_calibration,
            calibration_rows,
            "reused_calibration_row_sha256_by_order",
        ),
    ):
        digests = raw.get(digest_key)
        if not isinstance(digests, Mapping) or set(digests) != {
            str(order) for order in reused
        }:
            raise ValueError(f"迁移{label}复用行指纹集合不完整")
        for order in reused:
            key = str(order)
            if key not in source_rows or key not in current_rows:
                raise ValueError(f"迁移{label}复用行丢失")
            if canonical_evidence_bytes(current_rows[key]) != canonical_evidence_bytes(
                source_rows[key]
            ):
                raise ValueError(f"迁移{label}复用行不是来源证据的逐字节副本")
            if str(digests[key]).lower() != canonical_evidence_sha256(source_rows[key]):
                raise ValueError(f"迁移{label}复用行指纹不一致")

    source_target_raw = raw.get("source_calibration_target_power_mw")
    if reused_calibration:
        source_target = _finite(source_target_raw, "迁移来源共同目标功率")
        current_target = _finite(
            report.get("calibration", {}).get("target_power_mw"),
            "迁移后共同目标功率",
        )
        if current_target != source_target:
            raise ValueError("共同目标功率改变后不得复用旧calibration证据")
        if raw.get("calibration_reuse_status") != "retained_exact_target":
            raise ValueError("旧calibration证据尚未完成共同目标功率复核")
    elif raw.get("calibration_reuse_status") not in (
        "no_eligible_rows",
        "discarded_target_changed",
    ):
        raise ValueError("迁移calibration复用状态无效")

    sessions = _validated_migration_sessions(report, candidate_sha)
    for mode, rows, reused in (
        ("ceiling-probe", ceiling_rows, set(reused_ceiling)),
        ("calibrate", calibration_rows, set(reused_calibration)),
    ):
        for order in range(45):
            if order in reused:
                continue
            row = rows.get(str(order))
            if not isinstance(row, Mapping):
                raise ValueError(f"迁移后第{order + 1}点缺少新实测证据")
            _validate_fresh_row_session(
                row, sessions, mode=mode, label=f"迁移后{mode}第{order + 1}点"
            )

    rounds = report.get("sparse_sequence", {}).get("rounds", {})
    if not isinstance(rounds, Mapping):
        raise ValueError("迁移后稀疏复测结构无效")
    for round_rows in rounds.values():
        if not isinstance(round_rows, Mapping):
            raise ValueError("迁移后稀疏复测轮次无效")
        for order, row in round_rows.items():
            if not isinstance(row, Mapping):
                raise ValueError("迁移后稀疏复测行无效")
            _validate_fresh_row_session(
                row,
                sessions,
                mode="sparse-sequence",
                label=f"迁移后sparse第{order}点",
            )
    return dict(raw)


def _validate_calibration(
    candidate_path: Path,
    calibration_path: Path,
    repository: Path,
) -> tuple[dict[str, Any], tuple[CandidateRow, ...], CertifiedCalibration]:
    candidate_payload, candidates = _load_candidate(candidate_path, repository)
    candidate_sha = _sha256(candidate_path)
    report = json.loads(calibration_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError("标定文件不是受支持的CH1稀疏标定报告")
    source = report.get("candidate_source")
    if not isinstance(source, Mapping):
        raise ValueError("标定报告缺少候选来源")
    if str(source.get("sha256", "")).lower() != candidate_sha:
        raise ValueError("标定报告与当前候选SHA256不匹配")
    if _exact_int(source.get("point_count"), "candidate point_count") != 45:
        raise ValueError("标定报告未覆盖45点")
    if (
        report.get("writes_runtime_table") is not False
        or report.get("writes_firmware") is not False
    ):
        raise ValueError("标定工具越权写入了运行表或固件")
    if report.get("shutter_confirmed") is not True:
        raise ValueError("标定报告没有最终精确关光确认")
    if report.get("final_runtime_certified") is not True:
        raise ValueError("标定报告尚未通过最终稀疏路径认证")

    settings = report.get("settings")
    if not isinstance(settings, Mapping):
        raise ValueError("标定报告缺少门限设置")
    wavelength_tolerance = _finite(
        settings.get("wavelength_tolerance_pm"), "波长误差门限"
    )
    minimum_smsr = _finite(settings.get("minimum_smsr_db"), "SMSR门限")
    confirmation_wave_span = _finite(
        settings.get("confirmation_wavelength_span_pm"), "复测波长跨度门限"
    )
    confirmation_power_span = _finite(
        settings.get("confirmation_power_span_fraction"), "复测功率跨度门限"
    )
    required_rounds = _exact_int(settings.get("sequence_rounds"), "sequence_rounds")
    probe_branch_tolerance = _finite(
        settings.get("probe_branch_tolerance_pm"), "探针分支波长门限"
    )
    if wavelength_tolerance > 2.0 or wavelength_tolerance <= 0.0:
        raise ValueError("硬件标定波长门限不得宽于±2 pm")
    if minimum_smsr < 20.0:
        raise ValueError("硬件标定SMSR门限不得低于20 dB")
    if confirmation_wave_span > 2.0 or confirmation_wave_span <= 0.0:
        raise ValueError("独立复测波长跨度门限不得宽于2 pm")
    if confirmation_power_span > 0.03 or confirmation_power_span <= 0.0:
        raise ValueError("独立复测功率跨度门限不得宽于3%")
    if required_rounds < 2:
        raise ValueError("最终稀疏顺序至少需要2轮实测")

    ceiling = report.get("ceiling_probe")
    if not isinstance(ceiling, Mapping) or not isinstance(ceiling.get("rows"), Mapping):
        raise ValueError("标定报告缺少完整功率上限探测")
    ceiling_rows = ceiling["rows"]
    if set(ceiling_rows) != {str(i) for i in range(45)} or not all(
        isinstance(ceiling_rows[str(i)], Mapping)
        and ceiling_rows[str(i)].get("qualified") is True
        for i in range(45)
    ):
        raise ValueError("功率上限探测未45/45通过")
    ceiling_summary = ceiling.get("summary")
    if (
        not isinstance(ceiling_summary, Mapping)
        or ceiling_summary.get("all_points_qualified") is not True
    ):
        raise ValueError("功率上限摘要未认证45/45通过")
    certified_ceiling = _finite(
        ceiling_summary.get("recommended_common_power_mw"), "认证共同功率上限"
    )

    calibration = report.get("calibration")
    if not isinstance(calibration, Mapping) or not isinstance(
        calibration.get("rows"), Mapping
    ):
        raise ValueError("标定报告缺少完整45点标定")
    calibration_rows = calibration["rows"]
    if set(calibration_rows) != {str(i) for i in range(45)}:
        raise ValueError("最终标定行必须恰好是0～44")
    summary = calibration.get("summary")
    if not isinstance(summary, Mapping) or summary.get("all_points_passed") is not True:
        raise ValueError("最终标定未45/45通过")
    target_power = _finite(calibration.get("target_power_mw"), "共同目标功率")
    if target_power <= 0.0 or target_power > certified_ceiling + 1e-12:
        raise ValueError("共同目标功率超过实测认证上限")

    normalized_rows: list[dict[str, Any]] = []
    powers: list[float] = []
    smsrs: list[float] = []
    errors: list[float] = []
    for candidate in candidates:
        raw = calibration_rows[str(candidate.order)]
        if not isinstance(raw, Mapping) or raw.get("passed") is not True:
            raise ValueError(f"最终标定第{candidate.order + 1}点未通过")
        if _exact_int(raw.get("order"), "calibration order") != candidate.order:
            raise ValueError("最终标定顺序与候选不一致")
        if (
            _exact_int(raw.get("fullband_index"), "calibration index")
            != candidate.fullband_index
        ):
            raise ValueError("最终标定索引与候选不一致")
        target = _finite(raw.get("target_wavelength_nm"), "calibration target")
        if abs(target - candidate.target_nm) > 5e-7:
            raise ValueError("最终标定目标波长与候选不一致")
        final_codes = _codes(raw.get("codes"), "最终标定DAC")
        ceiling_codes = _codes(
            ceiling_rows[str(candidate.order)].get("ceiling_codes"), "功率上限DAC"
        )
        original_codes = _codes(raw.get("original_codes"), "标定起始DAC")
        seed_source = raw.get("seed_source")
        if seed_source is None:
            # Reports produced before the dual-seed controller only had one
            # possible start state: the certified ceiling row.
            if original_codes != ceiling_codes:
                raise ValueError("旧版最终标定不是从已认证的功率上限状态开始")
        else:
            # The current controller measures both the original candidate and
            # the clean-ceiling state, then deliberately starts from whichever
            # qualified state is closest to the common-power target.  Bind that
            # choice to the candidate/ceiling evidence and to the actual probe;
            # accepting an arbitrary 'original_codes' value would break the
            # audit chain even if the final confirmations happened to pass.
            if raw.get("calibration_controller_revision") != (
                DUAL_MEASURED_SEED_CONTROLLER_REVISION
            ):
                raise ValueError("新版双实测种子证据没有绑定受支持的控制器版本")
            if seed_source not in ("candidate", "ceiling_probe"):
                raise ValueError("新版标定起始分支来源无效")
            candidate_codes = _codes(raw.get("candidate_codes"), "候选起始DAC")
            if candidate_codes != candidate.codes:
                raise ValueError("新版标定候选起始DAC与候选文件不一致")
            selected_seed_codes = _codes(
                raw.get("selected_seed_codes"), "选定起始DAC"
            )
            expected_seed_codes = (
                candidate.codes if seed_source == "candidate" else ceiling_codes
            )
            if (
                selected_seed_codes != original_codes
                or original_codes != expected_seed_codes
            ):
                raise ValueError("新版标定选定起始DAC与认证分支不一致")
            seed_probes = raw.get("seed_probes")
            if not isinstance(seed_probes, list) or not seed_probes:
                raise ValueError("新版标定缺少起始分支实测探针")
            expected_probe_codes = {"candidate": candidate.codes}
            if ceiling_codes != candidate.codes:
                expected_probe_codes["ceiling_probe"] = ceiling_codes
            probes_by_source: dict[str, Mapping[str, Any]] = {}
            qualified_scores: list[
                tuple[tuple[float, float, float], str, tuple[int, ...]]
            ] = []
            for probe in seed_probes:
                if not isinstance(probe, Mapping):
                    raise ValueError("新版标定起始分支探针格式无效")
                if probe.get("action") != "calibration_seed_probe":
                    raise ValueError("新版标定起始分支探针动作无效")
                probe_source = probe.get("source")
                if probe_source not in expected_probe_codes:
                    raise ValueError("新版标定起始分支探针来源无效")
                if probe_source in probes_by_source:
                    raise ValueError("新版标定起始分支探针来源重复")
                probes_by_source[str(probe_source)] = probe
                probe_codes = _codes(probe.get("codes"), "起始分支探针DAC")
                if probe_codes != expected_probe_codes[str(probe_source)]:
                    raise ValueError("新版标定起始分支探针DAC与认证分支不一致")
                blocked = bool(probe.get("blocked_by_prior_low_smsr_latch", False))
                latched = bool(probe.get("latched_low_smsr", False))
                reading = probe.get("reading")
                if not isinstance(reading, Mapping):
                    if not blocked or probe.get("qualified") is not False:
                        raise ValueError("新版标定起始分支探针缺少可重算读数")
                    continue
                wavelength = _finite(reading.get("wavelength_nm"), "种子探针波长")
                power = _finite(reading.get("power_mw"), "种子探针功率")
                smsr = _finite(
                    reading.get("normalized_smsr_db"), "种子探针SMSR"
                )
                wave_error_pm = (wavelength - candidate.target_nm) * 1000.0
                if "wavelength_error_pm" in reading and abs(
                    _finite(reading["wavelength_error_pm"], "种子探针波长误差")
                    - wave_error_pm
                ) > 5e-7:
                    raise ValueError("新版标定种子探针波长误差字段不可重算")
                power_distance = abs(power - target_power)
                if abs(
                    _finite(
                        probe.get("power_distance_to_target_mw"),
                        "种子探针目标功率距离",
                    )
                    - power_distance
                ) > 5e-10:
                    raise ValueError("新版标定种子探针功率距离不可重算")
                qualified = bool(
                    not blocked
                    and not latched
                    and abs(wave_error_pm) <= probe_branch_tolerance
                    and smsr >= minimum_smsr
                )
                if probe.get("qualified") is not qualified:
                    raise ValueError("新版标定种子探针合格状态不可重算")
                if qualified:
                    qualified_scores.append(
                        (
                            (power_distance, abs(wave_error_pm), -smsr),
                            str(probe_source),
                            probe_codes,
                        )
                    )
            if set(probes_by_source) != set(expected_probe_codes):
                raise ValueError("新版标定必须恰好实测全部候选起始分支")
            if not qualified_scores:
                raise ValueError("新版标定没有通过实测门限的起始分支")
            _score, winning_source, winning_codes = min(
                qualified_scores, key=lambda item: item[0]
            )
            if seed_source != winning_source or original_codes != winning_codes:
                raise ValueError("新版标定所选起始分支不是实测评分赢家")
        if any(final_codes[i] != original_codes[i] for i in (0, 3, 4)):
            raise ValueError("标定过程非法改动了GAIN/WAVE_A/WAVE_B分支码")
        tolerance = _finite(raw.get("power_tolerance_mw"), "功率误差门限")
        confirmations = raw.get("confirmations")
        if not isinstance(confirmations, list) or len(confirmations) != 2:
            raise ValueError("每个最终点必须包含两次独立AQ6150复测")
        if not all(
            isinstance(reading, Mapping)
            and _reading_passes(
                reading,
                target_nm=candidate.target_nm,
                target_power_mw=target_power,
                wavelength_tolerance_pm=wavelength_tolerance,
                power_tolerance_mw=tolerance,
                minimum_smsr_db=minimum_smsr,
            )
            for reading in confirmations
        ):
            raise ValueError(f"最终标定第{candidate.order + 1}点的独立复测未重算通过")
        waves = [
            _finite(item["wavelength_nm"], "confirmation wavelength")
            for item in confirmations
        ]
        point_powers = [
            _finite(item["power_mw"], "confirmation power") for item in confirmations
        ]
        point_smsrs = [
            _finite(item["normalized_smsr_db"], "confirmation SMSR")
            for item in confirmations
        ]
        wave_span = (max(waves) - min(waves)) * 1000.0
        power_span = (max(point_powers) - min(point_powers)) / max(
            statistics.mean(point_powers), 1e-12
        )
        if wave_span > confirmation_wave_span + 1e-9:
            raise ValueError(f"最终标定第{candidate.order + 1}点波长重复性未通过")
        if power_span > confirmation_power_span + 1e-12:
            raise ValueError(f"最终标定第{candidate.order + 1}点功率重复性未通过")
        final_wave = statistics.mean(waves)
        final_power = statistics.mean(point_powers)
        final_smsr = min(point_smsrs)
        error_pm = (final_wave - candidate.target_nm) * 1000.0
        for key, actual, allowed in (
            ("final_wavelength_nm", final_wave, 5e-10),
            ("final_wavelength_error_pm", error_pm, 5e-7),
            ("final_power_mw", final_power, 5e-10),
            ("final_smsr_min_db", final_smsr, 5e-10),
        ):
            if abs(_finite(raw.get(key), key) - actual) > allowed:
                raise ValueError(f"最终标定第{candidate.order + 1}点{key}与复测不一致")
        normalized_rows.append(
            {
                "order": candidate.order,
                "peak_number": candidate.peak_number,
                "within_peak_order": candidate.within_peak_order,
                "fullband_index": candidate.fullband_index,
                "role": candidate.role,
                "target_wavelength_nm": candidate.target_nm,
                "measured_wavelength_nm": final_wave,
                "codes": list(final_codes),
                "measured_power_mw": final_power,
                "minimum_smsr_db": final_smsr,
                "wavelength_error_pm": error_pm,
                "wavelength_span_pm": wave_span,
                "power_span_fraction": power_span,
            }
        )
        powers.append(final_power)
        smsrs.append(final_smsr)
        errors.append(abs(error_pm))

    sequence = report.get("sparse_sequence")
    if (
        not isinstance(sequence, Mapping)
        or sequence.get("code_source") != "calibration"
    ):
        raise ValueError("稀疏顺序复测必须使用最终标定DAC")
    audited_rows = _load_audited_rows(repository)
    hidden_predecessor = _validated_hidden_predecessor_request(
        sequence.get("hidden_predecessor_request"), candidates, audited_rows
    )
    raw_rounds = sequence.get("rounds")
    if not isinstance(raw_rounds, Mapping) or set(raw_rounds) != {
        str(i) for i in range(required_rounds)
    }:
        raise ValueError("稀疏顺序复测轮次不完整")
    final_codes_by_order = tuple(tuple(row["codes"]) for row in normalized_rows)
    normalized_rounds: list[tuple[dict[str, Any], ...]] = []
    active_session_ids: set[str] = set()
    hidden_round_metadata: list[dict[str, Any]] = []
    for round_index in range(required_rounds):
        raw_round = raw_rounds[str(round_index)]
        if not isinstance(raw_round, Mapping) or set(raw_round) != {
            str(i) for i in range(45)
        }:
            raise ValueError(f"稀疏顺序第{round_index + 1}轮未完成45点")
        normalized_round: list[dict[str, Any]] = []
        for candidate in candidates:
            raw = raw_round[str(candidate.order)]
            if not isinstance(raw, Mapping) or raw.get("passed") is not True:
                raise ValueError(
                    f"稀疏复测第{round_index + 1}轮第{candidate.order + 1}点未通过"
                )
            evidence_session_id = str(raw.get("evidence_session_id", "")).strip()
            if not evidence_session_id:
                raise ValueError("稀疏复测行缺少当前硬件会话ID")
            active_session_ids.add(evidence_session_id)
            predecessor = (candidate.order - 1) % 45
            checks = (
                (_exact_int(raw.get("round"), "round"), round_index),
                (_exact_int(raw.get("order"), "order"), candidate.order),
                (
                    _exact_int(raw.get("predecessor_order"), "predecessor_order"),
                    predecessor,
                ),
                (
                    _exact_int(raw.get("fullband_index"), "fullband_index"),
                    candidate.fullband_index,
                ),
            )
            if any(actual != expected for actual, expected in checks):
                raise ValueError("稀疏复测的轮次/顺序/索引与最终45点不一致")
            if bool(raw.get("wraparound")) != (candidate.order == 0):
                raise ValueError("稀疏复测没有精确记录末点到首点回绕")
            if (
                _codes(raw.get("codes"), "稀疏复测DAC")
                != final_codes_by_order[candidate.order]
            ):
                raise ValueError("稀疏复测DAC与最终标定DAC不一致")
            if (
                _codes(raw.get("predecessor_codes"), "稀疏前驱DAC")
                != final_codes_by_order[predecessor]
            ):
                raise ValueError("稀疏复测前驱DAC与最终顺序不一致")
            expects_hidden = bool(
                hidden_predecessor is not None
                and candidate.order == int(hidden_predecessor["mode_order"])
            )
            if bool(raw.get("used_hidden_predecessor", False)) != expects_hidden:
                raise ValueError("稀疏复测隐藏前驱启用标记与当前路径配置不一致")
            expected_path = SPARSE_HIDDEN_PATH if expects_hidden else SPARSE_DIRECT_PATH
            if raw.get("runtime_path") != expected_path:
                raise ValueError("稀疏复测没有记录真实prev->hidden->target路径")
            raw_hidden = raw.get("hidden_predecessor")
            if expects_hidden:
                if not isinstance(raw_hidden, Mapping):
                    raise ValueError("隐藏前驱点缺少逐轮路径元数据")
                normalized_hidden = {
                    "fullband_index": _exact_int(
                        raw_hidden.get("fullband_index"), "逐轮隐藏前驱索引"
                    ),
                    "codes": list(_codes(raw_hidden.get("codes"), "逐轮隐藏前驱DAC")),
                    "hold_us": _exact_int(
                        raw_hidden.get("hold_us"), "逐轮隐藏前驱hold_us"
                    ),
                }
                expected_hidden = {
                    "fullband_index": int(
                        hidden_predecessor["predecessor_fullband_index"]
                    ),
                    "codes": list(hidden_predecessor["predecessor_codes"]),
                    "hold_us": FIRMWARE_HIDDEN_PREDECESSOR_HOLD_US,
                }
                if normalized_hidden != expected_hidden:
                    raise ValueError("逐轮隐藏前驱元数据与2001点审计路径不一致")
                hidden_round_metadata.append(normalized_hidden)
            elif raw_hidden is not None:
                raise ValueError("普通稀疏点不得夹带隐藏前驱元数据")
            reading = raw.get("reading")
            tolerance = _finite(raw.get("power_tolerance_mw"), "稀疏功率门限")
            if not isinstance(reading, Mapping) or not _reading_passes(
                reading,
                target_nm=candidate.target_nm,
                target_power_mw=target_power,
                wavelength_tolerance_pm=wavelength_tolerance,
                power_tolerance_mw=tolerance,
                minimum_smsr_db=minimum_smsr,
            ):
                raise ValueError("稀疏复测的实测值未重算通过")
            normalized_round.append(dict(raw))
        normalized_rounds.append(tuple(normalized_round))

    sequence_summary = sequence.get("summary")
    if (
        not isinstance(sequence_summary, Mapping)
        or sequence_summary.get("all_points_runtime_path_certified") is not True
    ):
        raise ValueError("稀疏顺序摘要未通过")
    expected_hidden_orders = (
        [int(hidden_predecessor["mode_order"])]
        if hidden_predecessor is not None
        else []
    )
    if list(sequence_summary.get("hidden_predecessor_orders", ())) != expected_hidden_orders:
        raise ValueError("稀疏顺序摘要的隐藏前驱点与逐轮记录不一致")
    expected_hidden_hold_us = (
        FIRMWARE_HIDDEN_PREDECESSOR_HOLD_US
        if hidden_predecessor is not None
        else None
    )
    if sequence_summary.get("hidden_predecessor_hold_us") != expected_hidden_hold_us:
        raise ValueError("稀疏顺序摘要的隐藏前驱保持时间与逐轮记录不一致")
    expected_hidden_observations = required_rounds if hidden_predecessor else 0
    if len(hidden_round_metadata) != expected_hidden_observations:
        raise ValueError("隐藏前驱点没有完成至少两轮完整实测")
    if hidden_round_metadata and any(
        item != hidden_round_metadata[0] for item in hidden_round_metadata[1:]
    ):
        raise ValueError("两轮隐藏前驱元数据不完全一致")
    if len(active_session_ids) != 1:
        raise ValueError("稀疏顺序复测混用了旧硬件会话")
    sessions = report.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("标定报告缺少硬件会话审计记录")
    last = sessions[-1]
    if not isinstance(last, Mapping) or any(
        (
            last.get("mode") != "sparse-sequence",
            last.get("status") != "complete",
            last.get("shutter_confirmed") is not True,
        )
    ):
        raise ValueError("最后一次硬件会话必须是完成的稀疏复测且已关光")
    active_session_id = next(iter(active_session_ids))
    if str(last.get("session_id", "")).strip() != active_session_id:
        raise ValueError("当前90条稀疏复测不是最后一次完整硬件会话")
    if str(last.get("candidate_sha256", "")).lower() != candidate_sha:
        raise ValueError("最后一次稀疏复测会话与候选SHA256不一致")
    last_hidden = last.get("hidden_predecessor_request")
    if (dict(last_hidden) if isinstance(last_hidden, Mapping) else last_hidden) != (
        dict(hidden_predecessor) if hidden_predecessor is not None else None
    ):
        raise ValueError("最后一次硬件会话的隐藏前驱路径元数据不一致")

    _validate_migrated_report_evidence(
        candidate_payload=candidate_payload,
        candidate_path=candidate_path,
        report=report,
        repository=repository,
    )

    certificate = CertifiedCalibration(
        rows=tuple(normalized_rows),
        candidate_sha256=candidate_sha,
        calibration_sha256=_sha256(calibration_path),
        target_power_mw=target_power,
        validated_repeats=required_rounds,
        sequence_rounds=tuple(normalized_rounds),
        minimum_smsr_db=min(smsrs),
        maximum_wavelength_error_pm=max(errors),
        minimum_power_mw=min(powers),
        maximum_power_mw=max(powers),
        hidden_predecessor=hidden_predecessor,
    )
    return candidate_payload, candidates, certificate


def _interpolate(x: Sequence[float], y: Sequence[float], target: float) -> float:
    if len(x) != len(y) or len(x) < 3:
        raise ValueError("实测模板波长/强度数组不完整")
    axis = tuple(_finite(value, "template wavelength") for value in x)
    values = tuple(_finite(value, "template signal") for value in y)
    if any(right <= left for left, right in zip(axis, axis[1:])):
        raise ValueError("实测模板波长必须严格递增")
    if target < axis[0] or target > axis[-1]:
        raise ValueError("替换点超出所属光栅的实测模板区域")
    right = bisect.bisect_left(axis, target)
    if right == 0:
        return values[0]
    if right == len(axis):
        return values[-1]
    left = right - 1
    fraction = (target - axis[left]) / (axis[right] - axis[left])
    return values[left] + fraction * (values[right] - values[left])


def _template_values(
    peak: Mapping[str, Any], targets: Sequence[float]
) -> tuple[list[float], list[float]]:
    dense_x = peak.get("dense_wavelength_nm")
    dense_y = peak.get("dense_template_signal_v")
    if not isinstance(dense_x, list) or not isinstance(dense_y, list):
        raise ValueError("待桥接v4缺少CH1稠密实测模板")
    axis = [_finite(value, "dense wavelength") for value in dense_x]
    signal = [_finite(value, "dense signal") for value in dense_y]
    if len(axis) != len(signal) or len(axis) < 5:
        raise ValueError("待桥接v4的稠密模板不完整")
    slopes = []
    for index in range(len(axis)):
        left = max(0, index - 1)
        right = min(len(axis) - 1, index + 1)
        slopes.append((signal[right] - signal[left]) / (axis[right] - axis[left]))
    return (
        [_interpolate(axis, signal, target) for target in targets],
        [_interpolate(axis, slopes, target) for target in targets],
    )


def _selection_contract(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != SELECTION_SCHEMA:
        raise ValueError("只能桥接equal_interval_auto_mode_selection_v4")
    if _exact_int(payload.get("source_point_count"), "source_point_count") != 2001:
        raise ValueError("v4必须来自完整2001点扫描")
    if _exact_int(payload.get("detected_peak_count"), "detected_peak_count") != 9:
        raise ValueError("v4必须实测识别9个峰")
    if _exact_int(payload.get("configured_peak_count"), "configured_peak_count") != 9:
        raise ValueError("v4必须严格配置9个峰")
    if tuple(
        _exact_int(item, "fbg_channel") for item in payload.get("fbg_channels", ())
    ) != (1,):
        raise ValueError("v4机械手指只允许CH1")
    selection = payload.get("selection")
    if (
        not isinstance(selection, Mapping)
        or selection.get("method") != "measured_template_fisher_v1"
    ):
        raise ValueError("v4必须保留非高斯实测模板")
    stress = payload.get("stress")
    temperature = payload.get("temperature")
    if not isinstance(stress, Mapping) or not isinstance(stress.get("peaks"), list):
        raise ValueError("v4缺少应力实测模板")
    if len(stress["peaks"]) != 9:
        raise ValueError("v4应力实测模板必须包含9峰")
    if not isinstance(temperature, Mapping) or not isinstance(
        temperature.get("rows"), list
    ):
        raise ValueError("v4缺少完整温度候选表")


def _sparse_validation(
    rows: Sequence[Mapping[str, Any]], certificate: CertifiedCalibration
) -> dict[str, Any]:
    per_point = []
    hidden = certificate.hidden_predecessor
    hidden_mode_order = int(hidden["mode_order"]) if hidden is not None else None
    for mode_index, row in enumerate(rows):
        repetitions = []
        for round_index, round_rows in enumerate(certificate.sequence_rounds):
            observation = round_rows[mode_index]
            repetitions.append(
                {
                    "round": round_index,
                    "wavelength_nm": _finite(
                        observation["reading"]["wavelength_nm"], "wave"
                    ),
                    "power_mw": _finite(observation["reading"]["power_mw"], "power"),
                    "normalized_smsr_db": _finite(
                        observation["reading"]["normalized_smsr_db"], "smsr"
                    ),
                }
            )
        per_point.append(
            {
                "mode_index": mode_index,
                "fullband_index": int(row["fullband_index"]),
                "passed": True,
                "requires_hidden_predecessor": mode_index == hidden_mode_order,
                "runtime_path": (
                    SPARSE_HIDDEN_PATH
                    if mode_index == hidden_mode_order
                    else SPARSE_DIRECT_PATH
                ),
                "repetitions": repetitions,
            }
        )
        if mode_index == hidden_mode_order:
            per_point[-1]["hidden_predecessor"] = {
                "fullband_index": int(hidden["predecessor_fullband_index"]),
                "codes": list(hidden["predecessor_codes"]),
                "hold_us": int(hidden["hold_us"]),
            }
    certified_hidden = []
    if hidden is not None:
        certified_hidden.append(
            {
                "mode_index": int(hidden["mode_order"]),
                "fullband_index": int(hidden["fullband_index"]),
                "predecessor_fullband_index": int(
                    hidden["predecessor_fullband_index"]
                ),
                "predecessor_codes": list(hidden["predecessor_codes"]),
                "hold_us": int(hidden["hold_us"]),
                "runtime_path": SPARSE_HIDDEN_PATH,
                "validated_repeats": certificate.validated_repeats,
                "certified": True,
            }
        )
    return {
        "status": "passed",
        "validation_method": "hardware_sparse_order_replay",
        "stress_table_fingerprint_sha256": sparse_stress_table_fingerprint(rows),
        "validated_point_count": len(rows),
        "validated_repeats": certificate.validated_repeats,
        "validated_fullband_indices": [int(row["fullband_index"]) for row in rows],
        "per_point_results": per_point,
        "failed_mode_indices": [],
        "certified_hidden_predecessors": certified_hidden,
        "includes_last_to_first_wraparound_each_round": True,
        "requires_hardware_sparse_order_replay": True,
    }


def build_certified_v4_report(
    selection_payload: Mapping[str, Any],
    *,
    candidate_path: Path,
    calibration_path: Path,
    repository: Path = REPOSITORY,
) -> dict[str, Any]:
    """Return a v4 candidate whose stress rows are exact certified final rows."""

    _selection_contract(selection_payload)
    candidate_path = _repository_file(repository, candidate_path, "候选")
    calibration_path = _repository_file(repository, calibration_path, "标定报告")
    candidate_payload, candidates, certified = _validate_calibration(
        candidate_path, calibration_path, repository.resolve()
    )
    report = copy.deepcopy(dict(selection_payload))
    old_peaks = report["stress"]["peaks"]
    rows: list[dict[str, Any]] = []
    new_peaks: list[dict[str, Any]] = []
    for peak_number in range(1, 10):
        candidates_for_peak = [
            row for row in candidates if row.peak_number == peak_number
        ]
        final_for_peak = [certified.rows[row.order] for row in candidates_for_peak]
        peak = copy.deepcopy(old_peaks[peak_number - 1])
        if _exact_int(peak.get("peak_number"), "template peak_number") != peak_number:
            raise ValueError("v4实测模板必须按G1～G9排列")
        if _exact_int(peak.get("source_channel", 1), "template source_channel") != 1:
            raise ValueError("v4实测模板必须来自CH1")
        template_targets = [row.target_nm for row in candidates_for_peak]
        template_signal, template_slope = _template_values(peak, template_targets)
        center = _finite(peak.get("center_nm"), "template center")
        sigma = _finite(peak.get("sigma_nm"), "template sigma")
        if sigma <= 0.0:
            raise ValueError("实测模板峰宽无效")
        core_half_span = max(
            abs(row.target_nm - center) for row in candidates_for_peak[1:4]
        )
        half_span = max(abs(row.target_nm - center) for row in candidates_for_peak)
        for local, (candidate, final) in enumerate(
            zip(candidates_for_peak, final_for_peak, strict=True)
        ):
            selection_role = (
                "left_guard"
                if local == 0
                else "right_guard"
                if local == 4
                else "peak_body"
            )
            template_role = {
                "left_baseline": "left_baseline_anchor",
                "left_shift_slope": "left_shift_sensitive",
                "amplitude_peak": "peak_amplitude_anchor",
                "right_shift_slope": "right_shift_sensitive",
                "right_baseline": "right_baseline_anchor",
            }[candidate.role]
            rows.append(
                {
                    **final,
                    "source_channel": 1,
                    "fitted_center_nm": center,
                    "selection_half_span_nm": half_span,
                    "selection_core_half_span_nm": core_half_span,
                    "selection_role": selection_role,
                    "template_role": template_role,
                    "selection_method": "measured_template_fisher_v1",
                    "offset_sigma": (candidate.target_nm - center) / sigma,
                    "template_signal_v": template_signal[local],
                    "template_slope_v_per_nm": template_slope[local],
                    "template_reliability_weight": 1.0,
                    "hardware_calibration_code_source": (
                        f"calibration.rows[{candidate.order}].codes"
                    ),
                }
            )
        peak["selected_fullband_indices"] = [
            row.fullband_index for row in candidates_for_peak
        ]
        peak["selected_template_signal_v"] = template_signal
        peak["selected_template_slope_v_per_nm"] = template_slope
        peak["selected_reliability_weight"] = [1.0] * 5
        peak["hardware_ceiling_constrained"] = True
        new_peaks.append(peak)

    expected_indices = tuple(row.fullband_index for row in candidates)
    if (
        len(rows) != 45
        or tuple(int(row["fullband_index"]) for row in rows) != expected_indices
    ):
        raise AssertionError("桥接后45点顺序内部不一致")
    report["stress"] = {
        **report["stress"],
        "point_count": 45,
        "minimum_points_per_peak": 5,
        "points_per_peak": [5] * 9,
        "peaks": new_peaks,
        "rows": rows,
    }
    report["selection"] = {
        **report["selection"],
        "dac_row_policy": "exact_final_codes_from_certified_hardware_calibration",
        "hardware_calibration_required": True,
    }
    report["sparse_sequence_validation"] = _sparse_validation(rows, certified)
    report["hardware_calibration_certificate"] = {
        "schema": BRIDGE_SCHEMA,
        "status": "passed",
        "candidate_path": _relative_path(candidate_path, repository),
        "candidate_sha256": certified.candidate_sha256,
        "calibration_report_path": _relative_path(calibration_path, repository),
        "calibration_report_sha256": certified.calibration_sha256,
        "stress_table_fingerprint_sha256": sparse_stress_table_fingerprint(rows),
        "code_source": "calibration.rows[order].codes",
        "point_count": 45,
        "peak_count": 9,
        "validated_repeats": certified.validated_repeats,
        "target_power_mw": certified.target_power_mw,
        "measured_power_min_mw": certified.minimum_power_mw,
        "measured_power_max_mw": certified.maximum_power_mw,
        "minimum_smsr_db": certified.minimum_smsr_db,
        "maximum_absolute_wavelength_error_pm": certified.maximum_wavelength_error_pm,
        "shutter_confirmed": True,
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    if candidate_payload.get("candidate_migration") is not None:
        migration = calibration_path.read_text(encoding="utf-8")
        migration_payload = json.loads(migration).get("evidence_migration", {})
        report["hardware_calibration_certificate"]["candidate_migration"] = {
            "schema": CANDIDATE_MIGRATION_SCHEMA,
            "base_candidate_sha256": candidate_payload["candidate_migration"][
                "base_candidate_sha256"
            ],
            "source_report_sha256": candidate_payload["candidate_migration"][
                "base_report_sha256"
            ],
            "changed_orders": list(
                candidate_payload["candidate_migration"]["changed_orders"]
            ),
            "reused_ceiling_orders": list(
                migration_payload.get("reused_ceiling_orders", [])
            ),
            "reused_calibration_orders": list(
                migration_payload.get("reused_calibration_orders", [])
            ),
            "full_sparse_sequence_remeasured": True,
        }
    return report


def validate_hardware_calibration_certificate(
    selection_payload: Mapping[str, Any],
    repository: Path = REPOSITORY,
) -> dict[str, Any] | None:
    """Reopen certificate evidence and validate the exact embedded stress rows.

    Returns ``None`` for an ordinary audited-row v4 report with no hardware
    certificate.  If a certificate is present, every mismatch is fatal.
    """

    raw = selection_payload.get("hardware_calibration_certificate")
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or raw.get("schema") != BRIDGE_SCHEMA:
        raise ValueError("硬件标定证书格式无效")
    if raw.get("status") != "passed" or raw.get("shutter_confirmed") is not True:
        raise ValueError("硬件标定证书未通过或未关光")
    candidate_path = _repository_file(repository, raw.get("candidate_path"), "候选")
    calibration_path = _repository_file(
        repository, raw.get("calibration_report_path"), "标定报告"
    )
    if str(raw.get("candidate_sha256", "")).lower() != _sha256(candidate_path):
        raise ValueError("硬件标定证书的候选SHA256已失效")
    if str(raw.get("calibration_report_sha256", "")).lower() != _sha256(
        calibration_path
    ):
        raise ValueError("硬件标定证书的报告SHA256已失效")
    _, _, certified = _validate_calibration(
        candidate_path, calibration_path, repository.resolve()
    )
    stress = selection_payload.get("stress")
    rows = stress.get("rows") if isinstance(stress, Mapping) else None
    if not isinstance(rows, list) or len(rows) != 45:
        raise ValueError("硬件标定证书需要完整45点应力表")
    for order, (actual, expected) in enumerate(zip(rows, certified.rows, strict=True)):
        if not isinstance(actual, Mapping):
            raise ValueError("硬件标定应力行格式无效")
        if (
            _exact_int(actual.get("fullband_index"), "stress index")
            != expected["fullband_index"]
        ):
            raise ValueError("硬件标定应力索引与原报告不一致")
        if (
            _exact_int(actual.get("peak_number"), "stress peak")
            != expected["peak_number"]
        ):
            raise ValueError("硬件标定应力峰号与原报告不一致")
        if _codes(actual.get("codes"), "stress DAC") != tuple(expected["codes"]):
            raise ValueError(f"应力第{order + 1}点5路DAC不是最终标定结果")
        if (
            abs(
                _finite(actual.get("measured_wavelength_nm"), "stress wavelength")
                - float(expected["measured_wavelength_nm"])
            )
            > 5e-10
        ):
            raise ValueError(f"应力第{order + 1}点波长不是最终标定结果")
        if (
            abs(
                _finite(actual.get("target_wavelength_nm"), "stress target")
                - float(expected["target_wavelength_nm"])
            )
            > 5e-7
        ):
            raise ValueError("硬件标定应力目标波长与原报告不一致")
    fingerprint = sparse_stress_table_fingerprint(rows)
    if str(raw.get("stress_table_fingerprint_sha256", "")).lower() != fingerprint:
        raise ValueError("硬件标定证书与当前45点表指纹不匹配")
    sequence = selection_payload.get("sparse_sequence_validation")
    if (
        not isinstance(sequence, Mapping)
        or str(sequence.get("stress_table_fingerprint_sha256", "")).lower()
        != fingerprint
    ):
        raise ValueError("稀疏复测与硬件标定证书指纹不匹配")
    for key, expected in (
        ("point_count", 45),
        ("peak_count", 9),
        ("validated_repeats", certified.validated_repeats),
    ):
        if _exact_int(raw.get(key), key) != expected:
            raise ValueError(f"硬件标定证书{key}不一致")
    if raw.get("code_source") != "calibration.rows[order].codes":
        raise ValueError("硬件标定证书的DAC来源无效")
    return dict(raw)


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    args = parser.parse_args(argv)
    selection = json.loads(args.selection_report.read_text(encoding="utf-8"))
    result = build_certified_v4_report(
        selection,
        candidate_path=args.candidate,
        calibration_path=args.calibration_report,
        repository=args.repository.resolve(),
    )
    _atomic_write(args.output.resolve(), result)
    print(
        f"certified_v4={args.output.resolve()} "
        f"fingerprint={result['sparse_sequence_validation']['stress_table_fingerprint_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
