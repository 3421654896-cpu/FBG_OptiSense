"""Create an auditable checkpoint after changing a selected CH1 candidate.

This is an offline-only migration tool.  It never talks to the laser, meter,
firmware, or runtime table.  Evidence is copied only for candidate point JSON
objects whose complete canonical byte representation is unchanged.  Changed
points are left absent, and every sparse-sequence observation is discarded.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from selected_calibration_bridge import (
    CALIBRATION_SCHEMA,
    CANDIDATE_MIGRATION_SCHEMA,
    EVIDENCE_MIGRATION_SCHEMA,
    REPOSITORY,
    _load_audited_rows,
    _load_candidate,
    _relative_path,
    _sha256,
    canonical_evidence_bytes,
    canonical_evidence_sha256,
)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


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


def _flatten_points(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    for peak in payload["peaks"]:
        result.extend(peak["points"])
    return tuple(result)


def _row_is_bound(row: object, point: Mapping[str, Any]) -> bool:
    if not isinstance(row, Mapping):
        return False
    try:
        return bool(
            int(row.get("order", -1)) == int(point["plan_order"])
            and int(row.get("fullband_index", -1)) == int(point["source_index"])
            and float(row.get("target_wavelength_nm", math.nan))
            == float(point["target_wavelength_nm"])
        )
    except (TypeError, ValueError, KeyError):
        return False


def _closed_source_report(
    source_report: Mapping[str, Any], base_candidate_sha256: str
) -> None:
    source = source_report.get("candidate_source")
    if (
        source_report.get("schema") != CALIBRATION_SCHEMA
        or not isinstance(source, Mapping)
        or str(source.get("sha256", "")).lower() != base_candidate_sha256
        or int(source.get("point_count", -1)) != 45
    ):
        raise ValueError("来源报告没有绑定基准45点候选")
    if (
        source_report.get("writes_runtime_table") is not False
        or source_report.get("writes_firmware") is not False
        or source_report.get("shutter_confirmed") is not True
    ):
        raise ValueError("来源报告越权或没有最终关光确认")
    sessions = source_report.get("sessions")
    last = sessions[-1] if isinstance(sessions, list) and sessions else None
    if (
        not isinstance(last, Mapping)
        or last.get("status") not in ("complete", "completed_with_rejections")
        or last.get("shutter_confirmed") is not True
        or not str(last.get("finished", "")).strip()
    ):
        raise ValueError("来源报告最后一次会话没有安全结束并确认关光")


def build_migrated_candidate(
    *,
    base_candidate_path: Path,
    base_report_path: Path,
    replacements: Mapping[int, int],
    repository: Path = REPOSITORY,
    reason: str = "replace hardware-rejected candidate",
) -> dict[str, Any]:
    """Return a new candidate; no file is written by this function."""

    repository = repository.resolve()
    base_candidate_path = base_candidate_path.resolve()
    base_report_path = base_report_path.resolve()
    base_payload, _ = _load_candidate(
        base_candidate_path, repository, allow_migration=False
    )
    base_sha = _sha256(base_candidate_path)
    source_report = json.loads(base_report_path.read_text(encoding="utf-8"))
    if not isinstance(source_report, Mapping):
        raise ValueError("来源报告根结点无效")
    _closed_source_report(source_report, base_sha)
    if not replacements:
        raise ValueError("至少要指定一个真实替换点")
    normalized = dict(
        sorted((int(order), int(index)) for order, index in replacements.items())
    )
    if any(not 0 <= order < 45 for order in normalized):
        raise ValueError("替换plan_order必须位于0～44")

    audited = _load_audited_rows(repository)
    result = copy.deepcopy(base_payload)
    points = _flatten_points(result)
    old_points = _flatten_points(base_payload)
    for order, fullband_index in normalized.items():
        if not 0 <= fullband_index < len(audited):
            raise ValueError("替换fullband_index越界")
        old = points[order]
        row = audited[fullband_index]
        old["source_index"] = fullband_index
        old["target_wavelength_nm"] = float(row["target_nm"])
        old["dac_codes"] = list(row["codes"])

    indices = tuple(int(point["source_index"]) for point in points)
    if len(set(indices)) != 45 or tuple(sorted(indices)) != indices:
        raise ValueError("替换后45点必须按波长严格递增且不重复")
    changed = tuple(
        order
        for order, (before, after) in enumerate(zip(old_points, points, strict=True))
        if canonical_evidence_bytes(before) != canonical_evidence_bytes(after)
    )
    if changed != tuple(normalized):
        raise ValueError("替换没有产生声明的逐点规范字节变化")
    unchanged = tuple(order for order in range(45) if order not in changed)
    result["created"] = _now()
    result["selection_method"] = (
        "audited candidate migration; unchanged point objects require "
        "canonical-byte identity"
    )
    result["selected_source_indices"] = list(indices)
    result.setdefault("replacements", []).extend(
        {
            "plan_order": order,
            "removed_fullband_index": int(old_points[order]["source_index"]),
            "replacement_fullband_index": normalized[order],
            "reason": reason,
        }
        for order in changed
    )
    result["hardware_sparse_sequence_validation"] = "pending"
    result["writes_runtime_table"] = False
    result["writes_firmware"] = False
    result["candidate_migration"] = {
        "schema": CANDIDATE_MIGRATION_SCHEMA,
        "created": _now(),
        "base_candidate_path": _relative_path(base_candidate_path, repository),
        "base_candidate_sha256": base_sha,
        "base_report_path": _relative_path(base_report_path, repository),
        "base_report_sha256": _sha256(base_report_path),
        "changed_orders": list(changed),
        "unchanged_orders": list(unchanged),
        "point_byte_canonicalization": "json-sort-keys-utf8-no-nan-v1",
        "base_point_sha256_by_order": {
            str(order): canonical_evidence_sha256(point)
            for order, point in enumerate(old_points)
        },
        "new_point_sha256_by_order": {
            str(order): canonical_evidence_sha256(point)
            for order, point in enumerate(points)
        },
        "changed_points_require_ceiling_remeasurement": True,
        "changed_points_require_calibration_remeasurement": True,
        "full_sparse_sequence_requires_remeasurement": True,
    }
    return result


def _partial_ceiling_summary(
    rows: Mapping[str, Mapping[str, Any]], guard: float
) -> dict[str, Any]:
    clean = [row for row in rows.values() if row.get("qualified") is True]
    powers = [float(row["clean_ceiling_power_mw"]) for row in clean]
    complete = len(rows) == 45 and len(clean) == 45
    return {
        "tested": len(rows),
        "qualified": len(clean),
        "rejected": len(rows) - len(clean),
        "all_points_qualified": complete,
        "minimum_measured_clean_ceiling_mw": min(powers, default=None),
        "maximum_measured_clean_ceiling_mw": max(powers, default=None),
        "ceiling_guard_fraction": guard,
        "recommended_common_power_mw": min(powers) * (1.0 - guard)
        if complete
        else None,
        "uses_unmeasured_power_extrapolation": False,
    }


def _partial_calibration_summary(
    rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    passed = [row for row in rows.values() if row.get("passed") is True]
    return {
        "tested": len(rows),
        "passed": len(passed),
        "failed": len(rows) - len(passed),
        "all_points_passed": len(rows) == 45 and len(passed) == 45,
        "requires_two_round_sparse_sequence_certification": True,
    }


def build_migrated_checkpoint(
    *,
    migrated_candidate_path: Path,
    base_report_path: Path,
    repository: Path = REPOSITORY,
) -> dict[str, Any]:
    repository = repository.resolve()
    candidate_payload, _ = _load_candidate(migrated_candidate_path, repository)
    migration = candidate_payload["candidate_migration"]
    base_report_path = base_report_path.resolve()
    if _sha256(base_report_path) != migration["base_report_sha256"]:
        raise ValueError("来源报告在候选生成后发生改变")
    source = json.loads(base_report_path.read_text(encoding="utf-8"))
    _closed_source_report(source, migration["base_candidate_sha256"])
    base_payload = json.loads(
        (repository / migration["base_candidate_path"]).read_text(encoding="utf-8")
    )
    before = _flatten_points(base_payload)
    after = _flatten_points(candidate_payload)
    unchanged = tuple(migration["unchanged_orders"])

    source_ceiling = source.get("ceiling_probe", {}).get("rows", {})
    source_calibration = source.get("calibration", {}).get("rows", {})
    reused_ceiling: dict[str, Any] = {}
    reused_calibration: dict[str, Any] = {}
    for order in unchanged:
        key = str(order)
        if canonical_evidence_bytes(before[order]) != canonical_evidence_bytes(
            after[order]
        ):
            raise ValueError("声明未变的候选点并非规范字节完全相同")
        ceiling = (
            source_ceiling.get(key) if isinstance(source_ceiling, Mapping) else None
        )
        if (
            _row_is_bound(ceiling, before[order])
            and isinstance(ceiling, Mapping)
            and ceiling.get("qualified") is True
        ):
            reused_ceiling[key] = copy.deepcopy(ceiling)
        calibration = (
            source_calibration.get(key)
            if isinstance(source_calibration, Mapping)
            else None
        )
        if (
            key in reused_ceiling
            and _row_is_bound(calibration, before[order])
            and isinstance(calibration, Mapping)
            and calibration.get("passed") is True
            and list(calibration.get("original_codes", ()))
            == list(reused_ceiling[key].get("ceiling_codes", ()))
        ):
            reused_calibration[key] = copy.deepcopy(calibration)

    settings = copy.deepcopy(source.get("settings"))
    if not isinstance(settings, Mapping):
        raise ValueError("来源报告缺少标定参数")
    guard = float(settings.get("ceiling_guard_fraction", 0.03))
    candidate_sha = _sha256(migrated_candidate_path)
    source_target = source.get("calibration", {}).get("target_power_mw")
    report = {
        "schema": CALIBRATION_SCHEMA,
        "created": _now(),
        "updated": _now(),
        "candidate_source": {
            "path": str(migrated_candidate_path.resolve()),
            "sha256": candidate_sha,
            "point_count": 45,
        },
        "settings": settings,
        "fullband_predecessor_sha256": source.get("fullband_predecessor_sha256"),
        "absolute_current_limits_ma": copy.deepcopy(
            source.get("absolute_current_limits_ma", {})
        ),
        "probe": {"rows": {}, "summary": {}},
        "ceiling_probe": {
            **{
                key: copy.deepcopy(value)
                for key, value in source.get("ceiling_probe", {}).items()
                if key not in ("rows", "summary", "retry_history")
            },
            "rows": reused_ceiling,
            "summary": _partial_ceiling_summary(reused_ceiling, guard),
        },
        "calibration": {
            "target_power_mw": source_target,
            "rows": reused_calibration,
            "summary": _partial_calibration_summary(reused_calibration),
        },
        "sparse_sequence": {"rounds": {}, "summary": {}},
        "sessions": [],
        "shutter_confirmed": None,
        "writes_runtime_table": False,
        "writes_firmware": False,
        "final_runtime_certified": False,
    }
    report["evidence_migration"] = {
        "schema": EVIDENCE_MIGRATION_SCHEMA,
        "created": _now(),
        "destination_candidate_sha256": candidate_sha,
        "source_candidate_path": migration["base_candidate_path"],
        "source_candidate_sha256": migration["base_candidate_sha256"],
        "source_report_path": _relative_path(base_report_path, repository),
        "source_report_sha256": migration["base_report_sha256"],
        "changed_orders": list(migration["changed_orders"]),
        "unchanged_orders": list(unchanged),
        "reused_ceiling_orders": [int(key) for key in reused_ceiling],
        "reused_calibration_orders": [int(key) for key in reused_calibration],
        "reused_ceiling_row_sha256_by_order": {
            key: canonical_evidence_sha256(value)
            for key, value in reused_ceiling.items()
        },
        "reused_calibration_row_sha256_by_order": {
            key: canonical_evidence_sha256(value)
            for key, value in reused_calibration.items()
        },
        "source_calibration_target_power_mw": source_target,
        "calibration_reuse_status": "pending_exact_target_check"
        if reused_calibration
        else "no_eligible_rows",
        "sparse_sequence_reused": False,
        "full_sparse_sequence_requires_remeasurement": True,
    }
    return report


def reconcile_calibration_reuse(
    report: dict[str, Any], target_power_mw: float
) -> tuple[int, ...]:
    """Retain old calibration rows only for an exactly unchanged common target."""

    migration = report.get("evidence_migration")
    if not isinstance(migration, dict):
        return ()
    reused = tuple(int(value) for value in migration["reused_calibration_orders"])
    if not reused:
        migration["calibration_reuse_status"] = "no_eligible_rows"
        return ()
    old_target = float(migration["source_calibration_target_power_mw"])
    if float(target_power_mw) == old_target:
        migration["calibration_reuse_status"] = "retained_exact_target"
        return reused
    rows = report.setdefault("calibration", {}).setdefault("rows", {})
    discarded = {}
    for order in reused:
        key = str(order)
        if key in rows:
            discarded[key] = canonical_evidence_sha256(rows.pop(key))
    migration["discarded_calibration_row_sha256_by_order"] = discarded
    migration["reused_calibration_orders"] = []
    migration["reused_calibration_row_sha256_by_order"] = {}
    migration["calibration_reuse_status"] = "discarded_target_changed"
    migration["discard_reason"] = {
        "source_target_power_mw": old_target,
        "new_target_power_mw": float(target_power_mw),
    }
    report["calibration"]["summary"] = _partial_calibration_summary(rows)
    report["final_runtime_certified"] = False
    return ()


def _parse_replacement(value: str) -> tuple[int, int]:
    try:
        left, right = value.split("=", 1)
        return int(left), int(right)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            "替换格式必须为PLAN_ORDER=FULLBAND_INDEX"
        ) from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-candidate", type=Path, required=True)
    parser.add_argument("--base-report", type=Path, required=True)
    parser.add_argument(
        "--replace", action="append", type=_parse_replacement, required=True
    )
    parser.add_argument("--reason", default="replace hardware-rejected candidate")
    parser.add_argument("--output-candidate", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    args = parser.parse_args(argv)
    repository = args.repository.resolve()
    output_candidate = args.output_candidate.resolve()
    output_report = args.output_report.resolve()
    if output_candidate.exists() or output_report.exists():
        raise ValueError("迁移输出已存在；为保留审计链，必须使用新的输出文件名")
    replacements = dict(args.replace)
    if len(replacements) != len(args.replace):
        raise ValueError("同一plan_order不得重复声明替换")
    candidate = build_migrated_candidate(
        base_candidate_path=args.base_candidate.resolve(),
        base_report_path=args.base_report.resolve(),
        replacements=replacements,
        repository=repository,
        reason=args.reason,
    )
    _atomic_write(output_candidate, candidate)
    try:
        report = build_migrated_checkpoint(
            migrated_candidate_path=output_candidate,
            base_report_path=args.base_report.resolve(),
            repository=repository,
        )
        _atomic_write(output_report, report)
    except Exception:
        output_candidate.unlink(missing_ok=True)
        raise
    print(
        f"candidate={output_candidate} sha256={_sha256(output_candidate)} "
        f"checkpoint={output_report} "
        f"changed={candidate['candidate_migration']['changed_orders']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
