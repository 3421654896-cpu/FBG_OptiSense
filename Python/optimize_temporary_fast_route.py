"""Find and certify a forward 9x5 temporary route whose main edges finish by 3 ms.

The search never edits the installed 2001-point table or firmware.  It tests
small calibrated candidate neighborhoods around the dense-spectrum FWHM
anchors with the board-timestamped custom two-row diagnostic, then solves the
whole 45-point route with speed as the primary constraint and five-point
Gaussian+baseline fit as a group constraint.  Only a route that also passes a
fresh slow 45-point reference and a production-format 2.5 ms replay is saved
as the application's resumable last session.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
import serial

import app_JDSU as app
from benchmark_stress_realtime import _close_shutter, _safe_disarm
from capture_temporary_half_height_transients import (
    capture_custom,
    load_half_height_plan,
    main_edge_metric,
)
from laser_switching_test import SwitchTester
from temporary_peak_fit import fit_temporary_frame
from temporary_reference import acquire_reference
from temporary_test_mode import (
    assess_five_point_shape,
    build_manual_dense_plan,
    execute_device,
    validate_plan_payload,
)


HERE = Path(__file__).resolve().parent
DEFAULT_SESSION = HERE / "outputs" / "temporary_test" / "last_45_session.json"
DEFAULT_OUTPUT = HERE / "outputs" / "temporary_fast_route_search.json"
DEFAULT_PLAN = HERE / "outputs" / "temporary_test" / "temporary_fast_route_plan.json"
FAST_DEADLINE_US = 3000.0
PREFERRED_DEADLINE_US = 2000.0
FAST_POINT_DELAY_US = 2500


def _row(point, group: int) -> dict:
    return {
        "group": int(group),
        "index": int(point.index),
        "target_nm": float(point.target_nm),
        "measured_nm": float(point.measured_nm),
        "codes": [int(value) for value in point.codes],
    }


def candidate_layers(table, automatic_plan: dict, radius: int = 2) -> list[list[dict]]:
    """Make five non-overlapping candidate bands per peak around FWHM anchors."""

    if not 1 <= radius <= 8:
        raise ValueError("候选半径须为1～8个2001点表格行")
    table = tuple(table)
    rows = automatic_plan["rows"]
    checks = automatic_plan["endpoint_checks"]
    if len(table) != 2001 or len(rows) != 45 or len(checks) != 9:
        raise ValueError("需要完整2001点表和9x5半高宽初始路线")
    anchors = [int(row["index"]) for row in rows]
    layers: list[list[dict]] = []
    for flat, anchor in enumerate(anchors):
        group, role = divmod(flat, 5)
        group_anchors = anchors[group * 5:group * 5 + 5]
        left_nm = float(checks[group]["fwhm_left_nm"])
        right_nm = float(checks[group]["fwhm_right_nm"])
        lower = group_anchors[role - 1] + 1 if role else 0
        upper = group_anchors[role + 1] - 1 if role < 4 else len(table) - 1
        candidates = []
        for index in range(max(lower, anchor - radius), min(upper, anchor + radius) + 1):
            point = table[index]
            if left_nm - 1e-9 <= float(point.measured_nm) <= right_nm + 1e-9:
                candidates.append(_row(point, group + 1))
        if not candidates:
            raise ValueError(f"峰{group + 1}第{role + 1}角色没有半高宽内候选点")
        layers.append(candidates)
    return layers


def required_pairs(layers: list[list[dict]]) -> list[tuple[dict, dict]]:
    pairs = []
    # Self captures provide an independently held value for the first route
    # point without introducing the forbidden last-to-first reverse wrap.
    pairs.extend((row, row) for row in layers[0])
    for prior, current in zip(layers, layers[1:]):
        pairs.extend((source, target) for source in prior for target in current)
    unique = {}
    for source, target in pairs:
        unique[(int(source["index"]), int(target["index"]))] = (source, target)
    return list(unique.values())


def _pair_key(source: int, target: int) -> str:
    return f"{int(source)}:{int(target)}"


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def capture_with_link_retry(device, source: dict, target: dict, tag: int,
                            attempts: int = 3) -> dict:
    """Retry a missing USB response without changing the requested pair."""

    last = None
    for attempt in range(attempts):
        try:
            return capture_custom(device, source, target, tag)
        except (TimeoutError, ValueError) as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(.15)
    raise last


def capture_search(port: str, selector: int, layers: list[list[dict]], output: Path,
                   repeats: int) -> dict:
    pairs = required_pairs(layers)
    report = {
        "schema": "temporary_fast_route_search_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "port": port,
        "feedback_selector": selector,
        "candidate_radius_rows": max((len(layer) for layer in layers), default=0),
        "deadline_us_end_to_end": FAST_DEADLINE_US,
        "candidate_layers": layers,
        "requested_pair_count": len(pairs),
        "search_repeats": repeats,
        "pairs": {},
        "complete": False,
    }
    tester = SwitchTester(port)
    try:
        tester.serial.dtr = True
        worker = app.EqualIntervalWorker(
            tester.serial, (), {}, settle_s=.3, feedback_selectors=(0, selector)
        )
        if not worker._feedback_command_and_ack():
            raise RuntimeError("CH1模拟档位设置未收到确认")
        tag = int(time.time_ns() & 0x7FFFFFFF) or 1
        total = len(pairs) * repeats
        done = 0
        for repeat in range(repeats):
            for source, target in pairs:
                tag = (tag + 1) & 0xFFFFFFFF or 1
                record = capture_with_link_retry(tester.serial, source, target, tag)
                record.update({
                    "repeat": repeat,
                    "source_fullband_index": int(source["index"]),
                    "target_fullband_index": int(target["index"]),
                    "request_tag": tag,
                })
                key = _pair_key(source["index"], target["index"])
                report["pairs"].setdefault(key, {"source": source, "target": target,
                                                    "captures": []})["captures"].append(record)
                done += 1
                if done % 10 == 0 or done == total:
                    report["progress"] = {"completed": done, "total": total}
                    _atomic_json(output, report)
                    print(f"SEARCH {done}/{total}", flush=True)
        for pair in report["pairs"].values():
            pair["metric"] = main_edge_metric(pair["captures"])
        report["complete"] = True
    finally:
        errors = []
        disarmed = shuttered = False
        try:
            disarmed = _safe_disarm(tester.serial)
        except Exception as exc:
            errors.append(f"disarm: {exc}")
        try:
            shuttered = _close_shutter(tester.serial)
        except Exception as exc:
            errors.append(f"shutter: {exc}")
        report["safety_cleanup"] = {
            "exact_disarm_ack": bool(disarmed),
            "exact_soa_shutter_ack": bool(shuttered),
            "errors": errors,
        }
        tester.close()
        _atomic_json(output, report)
        if not disarmed or not shuttered:
            raise RuntimeError("候选搜索后的安全停光确认失败")
    return report


def seed_search(seed: dict, selector: int, layers: list[list[dict]], output: Path,
                repeats: int) -> dict:
    """Reuse exact pair captures when widening candidate neighborhoods."""

    if (seed.get("schema") != "temporary_fast_route_search_v1"
            or not seed.get("complete")
            or int(seed.get("feedback_selector", -1)) != selector):
        raise ValueError("候选种子文件不完整或模拟档位不匹配")
    report = {
        "schema": "temporary_fast_route_search_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "feedback_selector": selector,
        "deadline_us_end_to_end": FAST_DEADLINE_US,
        "candidate_layers": layers,
        "requested_pair_count": len(required_pairs(layers)),
        "search_repeats": repeats,
        "seed_source": seed.get("created_utc"),
        "pairs": {},
        "complete": False,
    }
    reused = 0
    for source, target in required_pairs(layers):
        key = _pair_key(source["index"], target["index"])
        old = seed.get("pairs", {}).get(key)
        captures = list(old.get("captures", []))[:repeats] if old else []
        if captures:
            reused += len(captures)
        report["pairs"][key] = {"source": source, "target": target,
                                "captures": captures}
    report["reused_capture_count"] = reused
    _atomic_json(output, report)
    print(f"SEED reused captures={reused}", flush=True)
    return report


def extend_search(port: str, selector: int, layers: list[list[dict]], output: Path,
                  report: dict, repeats: int) -> dict:
    """Append missing independent repeats to a completed candidate graph."""

    pairs = required_pairs(layers)
    missing = []
    for source, target in pairs:
        key = _pair_key(source["index"], target["index"])
        existing = report["pairs"].get(key, {}).get("captures", [])
        for repeat in range(len(existing), repeats):
            missing.append((source, target, repeat))
    if not missing:
        return report
    report["complete"] = False
    report["requested_pair_count"] = len(pairs)
    report["search_repeats"] = repeats
    tester = SwitchTester(port)
    try:
        tester.serial.dtr = True
        worker = app.EqualIntervalWorker(
            tester.serial, (), {}, settle_s=.3, feedback_selectors=(0, selector)
        )
        if not worker._feedback_command_and_ack():
            raise RuntimeError("补测候选时CH1模拟档位设置未收到确认")
        tag = int(time.time_ns() & 0x7FFFFFFF) or 1
        total = len(missing)
        for done, (source, target, repeat) in enumerate(missing, 1):
            tag = (tag + 1) & 0xFFFFFFFF or 1
            record = capture_with_link_retry(tester.serial, source, target, tag)
            record.update({"repeat": repeat,
                           "source_fullband_index": int(source["index"]),
                           "target_fullband_index": int(target["index"]),
                           "request_tag": tag})
            key = _pair_key(source["index"], target["index"])
            report["pairs"][key]["captures"].append(record)
            if done % 10 == 0 or done == total:
                report["progress"] = {"completed": done, "total": total,
                                      "phase": "repeat_extension"}
                _atomic_json(output, report)
                print(f"EXTEND {done}/{total}", flush=True)
        for pair in report["pairs"].values():
            pair["metric"] = main_edge_metric(pair["captures"])
            pair["individual_metrics"] = [
                main_edge_metric([capture]) for capture in pair["captures"]
            ]
        report["complete"] = True
    finally:
        errors = []
        disarmed = shuttered = False
        try:
            disarmed = _safe_disarm(tester.serial)
        except Exception as exc:
            errors.append(f"disarm: {exc}")
        try:
            shuttered = _close_shutter(tester.serial)
        except Exception as exc:
            errors.append(f"shutter: {exc}")
        report["safety_cleanup"] = {"exact_disarm_ack": bool(disarmed),
                                    "exact_soa_shutter_ack": bool(shuttered),
                                    "errors": errors}
        tester.close()
        _atomic_json(output, report)
        if not disarmed or not shuttered:
            raise RuntimeError("候选补测后的安全停光确认失败")
    return report


def _edge(report: dict, source: int, target: int, excluded: set[tuple[int, int]]):
    if (source, target) in excluded:
        return None
    pair = report["pairs"].get(_pair_key(source, target))
    if pair is None:
        return None
    metric = pair.get("metric") or main_edge_metric(pair["captures"])
    individual = pair.get("individual_metrics") or [
        main_edge_metric([capture]) for capture in pair["captures"]
    ]
    if not metric["passed"] or not all(item["passed"] for item in individual):
        return None
    completion = max(float(item["main_edge_complete_from_switch_begin_us"])
                     for item in individual)
    return metric, completion


def _score(path: dict) -> tuple:
    completions = path["completions"]
    return (
        sum(value > PREFERRED_DEADLINE_US for value in completions),
        max(completions, default=0.0),
        sum(completions),
        path.get("fit_rmse_sum", 0.0),
        -path.get("fit_r2_sum", 0.0),
    )


def _prune_by_endpoint(paths: list[dict], limit: int = 160) -> list[dict]:
    """Bound a layered search while preserving diverse fitted endpoints."""

    grouped = {}
    for path in paths:
        grouped.setdefault(int(path["rows"][-1]["index"]), []).append(path)
    kept = []
    for candidates in grouped.values():
        kept.extend(sorted(candidates, key=_score)[:limit])
    return kept


def _cheap_five_point_shape(values) -> bool:
    """Mirror the fit gate's inexpensive shape checks before grid fitting."""

    y = np.asarray(values, dtype=float)
    if y.shape != (5,) or not np.all(np.isfinite(y)):
        return False
    height = float(np.ptp(y))
    if height < 24.0 or np.max(y) - y[2] > .03 * height:
        return False
    tolerance = max(8.0, height * .03)
    return bool(np.all(np.diff(y[:3]) >= -tolerance)
                and np.all(np.diff(y[2:]) <= tolerance))


def solve_route(search: dict, layers: list[list[dict]], *,
                excluded_edges: set[tuple[int, int]] | None = None,
                excluded_groups: set[tuple[int, ...]] | None = None) -> dict:
    """Layered forward search; a complete Gaussian fit is required per group."""

    excluded_edges = excluded_edges or set()
    excluded_groups = excluded_groups or set()
    previous_states = []
    for group in range(9):
        first_layer = layers[group * 5]
        paths = []
        if group == 0:
            for target in first_layer:
                edge = _edge(search, target["index"], target["index"], excluded_edges)
                if edge is None:
                    continue
                metric, _completion = edge
                paths.append({
                    "rows": [target],
                    "values": [float(metric["final_tail_code"])],
                    "completions": [],
                    "edges": [],
                    "fit_checks": [],
                    "fit_rmse_sum": 0.0,
                    "fit_r2_sum": 0.0,
                })
        else:
            for prior in previous_states:
                for target in first_layer:
                    source = prior["rows"][-1]
                    edge = _edge(search, source["index"], target["index"], excluded_edges)
                    if edge is None:
                        continue
                    metric, completion = edge
                    paths.append({
                        "rows": prior["rows"] + [target],
                        "values": prior["values"] + [float(metric["final_tail_code"])],
                        "completions": prior["completions"] + [completion],
                        "edges": prior["edges"] + [(source["index"], target["index"])],
                        "fit_checks": list(prior["fit_checks"]),
                        "fit_rmse_sum": prior["fit_rmse_sum"],
                        "fit_r2_sum": prior["fit_r2_sum"],
                    })
        for role in range(1, 5):
            expanded = []
            for path in paths:
                source = path["rows"][-1]
                for target in layers[group * 5 + role]:
                    edge = _edge(search, source["index"], target["index"], excluded_edges)
                    if edge is None:
                        continue
                    metric, completion = edge
                    expanded.append({
                        "rows": path["rows"] + [target],
                        "values": path["values"] + [float(metric["final_tail_code"])],
                        "completions": path["completions"] + [completion],
                        "edges": path["edges"] + [(source["index"], target["index"])],
                        "fit_checks": list(path["fit_checks"]),
                        "fit_rmse_sum": path["fit_rmse_sum"],
                        "fit_r2_sum": path["fit_r2_sum"],
                    })
            paths = _prune_by_endpoint(expanded)
        accepted = []
        for path in paths:
            group_rows = path["rows"][-5:]
            group_values = path["values"][-5:]
            signature = tuple(int(row["index"]) for row in group_rows)
            if signature in excluded_groups or not _cheap_five_point_shape(group_values):
                continue
            check = assess_five_point_shape(
                [row["measured_nm"] for row in group_rows], group_values
            )
            check["group"] = group + 1
            if not check["passed"]:
                continue
            path["fit_checks"].append(check)
            path["fit_rmse_sum"] += float(check.get("rmse_fraction") or 0.0)
            path["fit_r2_sum"] += float(check.get("r_squared") or 0.0)
            accepted.append(path)
        # Future costs depend only on the current endpoint.  Retain the best
        # already-fit prefix for each possible endpoint to keep the search bounded.
        best = {}
        for path in accepted:
            endpoint = int(path["rows"][-1]["index"])
            if endpoint not in best or _score(path) < _score(best[endpoint]):
                best[endpoint] = path
        previous_states = list(best.values())
        if not previous_states:
            raise RuntimeError(f"峰{group + 1}没有同时满足3 ms主边沿和五点拟合的候选路线")
    return min(previous_states, key=_score)


def capture_final_route(port: str, selector: int, route: list[dict], repeats: int,
                        output: Path) -> dict:
    report = {
        "schema": "temporary_fast_route_final_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "deadline_us_end_to_end": FAST_DEADLINE_US,
        "repeats": repeats,
        "route": route,
        "captures": [],
        "complete": False,
    }
    tester = SwitchTester(port)
    try:
        tester.serial.dtr = True
        worker = app.EqualIntervalWorker(
            tester.serial, (), {}, settle_s=.3, feedback_selectors=(0, selector)
        )
        if not worker._feedback_command_and_ack():
            raise RuntimeError("最终验证的CH1模拟档位设置未收到确认")
        tag = int(time.time_ns() & 0x7FFFFFFF) or 1
        for repeat in range(repeats):
            pairs = [(route[0], route[0])] + list(zip(route, route[1:]))
            for order, (source, target) in enumerate(pairs):
                tag = (tag + 1) & 0xFFFFFFFF or 1
                item = capture_with_link_retry(tester.serial, source, target, tag)
                item.update({"repeat": repeat, "target_order": order,
                             "source_fullband_index": int(source["index"]),
                             "target_fullband_index": int(target["index"])})
                report["captures"].append(item)
            print(f"FINAL {repeat + 1}/{repeats}", flush=True)
        analyses = []
        failed_edges = set()
        values = []
        for order in range(45):
            records = [item for item in report["captures"] if item["target_order"] == order]
            aggregate = main_edge_metric(records)
            individual = [main_edge_metric([item]) for item in records]
            source = int(route[max(0, order - 1)]["index"])
            target = int(route[order]["index"])
            passed = bool(order == 0 or all(item["passed"] for item in individual))
            if not passed:
                failed_edges.add((source, target))
            analyses.append({"target_order": order, "source_index": source,
                             "target_index": target, "aggregate": aggregate,
                             "individual": individual, "passed": passed})
            values.append(float(aggregate["final_tail_code"]))
        fit_checks = []
        failed_groups = []
        for group in range(9):
            start = group * 5
            check = assess_five_point_shape(
                [row["measured_nm"] for row in route[start:start + 5]],
                values[start:start + 5],
            )
            check["group"] = group + 1
            fit_checks.append(check)
            if not check["passed"]:
                failed_groups.append(tuple(
                    int(row["index"]) for row in route[start:start + 5]
                ))
        report["analysis"] = {
            "criterion": "each repeat completes the main 10-90% edge within 3 ms end-to-end; no +/-5-code tail gate",
            "reverse_wrap_tested": False,
            "transitions": analyses,
            "failed_edges": [list(pair) for pair in sorted(failed_edges)],
            "reference_values": values,
            "fit_checks": fit_checks,
            "failed_group_routes": [list(group) for group in failed_groups],
            "speed_passed": not failed_edges,
            "fit_passed": not failed_groups,
        }
        report["complete"] = True
    finally:
        errors = []
        disarmed = shuttered = False
        try:
            disarmed = _safe_disarm(tester.serial)
        except Exception as exc:
            errors.append(f"disarm: {exc}")
        try:
            shuttered = _close_shutter(tester.serial)
        except Exception as exc:
            errors.append(f"shutter: {exc}")
        report["safety_cleanup"] = {"exact_disarm_ack": bool(disarmed),
                                    "exact_soa_shutter_ack": bool(shuttered),
                                    "errors": errors}
        tester.close()
        _atomic_json(output, report)
        if not disarmed or not shuttered:
            raise RuntimeError("最终路线验证后的安全停光确认失败")
    return report


def certified_plan(table, route: list[dict], dense_x, dense_y, final: dict,
                   source: str, *, fit_checks: list[dict] | None = None,
                   reference_values: list[float] | None = None) -> dict:
    groups = [[int(row["index"]) for row in route[start:start + 5]]
              for start in range(0, 45, 5)]
    plan = build_manual_dense_plan(table, groups, dense_x, dense_y, source=source)
    checks = list(fit_checks or final["analysis"]["fit_checks"])
    selected_values = list(reference_values or final["analysis"]["reference_values"])
    plan.update({
        "manual_selection": False,
        "selection_rule": "five_nonuniform_forward_main_edge_certified_points",
        "selection_center_method": "FWHM_role_bands_then_board_timestamped_route_search",
        "selection_fit_checks": checks,
        "selection_validation_required": not all(check["passed"] for check in checks),
        "validated_reference_values": selected_values,
        "forward_fast_stability_certificate": {
            "deadline_us_end_to_end": FAST_DEADLINE_US,
            "preferred_deadline_us_end_to_end": PREFERRED_DEADLINE_US,
            "criterion": "main_10_90_edge_complete; no fixed +/-5 ADC-code tail requirement",
            "reverse_wrap_tested": False,
            "repeats": final["repeats"],
            "all_individual_repeats_passed": final["analysis"]["speed_passed"],
            "source": source,
        },
        "point_delays_us": [FAST_POINT_DELAY_US] * 45,
    })
    plan["peaks_nm"] = [float(check["fitted_center_nm"]) for check in checks]
    return validate_plan_payload(plan)


def install_session(session_path: Path, plan_path: Path, plan: dict, selector: int,
                    reference_report: dict) -> None:
    old = {}
    if session_path.exists():
        old = json.loads(session_path.read_text(encoding="utf-8"))
        backup = session_path.with_name(
            session_path.stem + f"_before_fast_route_{time.time_ns()}.json"
        )
        backup.write_bytes(session_path.read_bytes())
    values = [float(row["ch1_adc_code"]) for row in reference_report["rows"]]
    wavelengths = [float(row["measured_nm"]) for row in plan["rows"]]
    payload = {
        "schema": "temporary_test_last_session_v1",
        "saved_s": time.time(),
        "ready": True,
        "plan": plan,
        "reference_values": values,
        "reference_wavelengths_nm": wavelengths,
        "feedback_selector": int(selector),
        "digital_display_gain": float(old.get("digital_display_gain", 1)),
        "reference_display_gain": float(old.get("reference_display_gain", 1)),
        "first_delay_us": int(old.get("first_delay_us", 350)),
        "boundary_extra_us": 0,
        "show_dense_selection": bool(old.get("show_dense_selection", True)),
        "show_reference": bool(old.get("show_reference", True)),
        "use_point_waits": True,
        "point_delays_us": [FAST_POINT_DELAY_US] * 45,
    }
    _atomic_json(plan_path, plan)
    _atomic_json(session_path, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plan-output", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--candidate-radius", type=int, default=2)
    parser.add_argument("--search-repeats", type=int, default=1)
    parser.add_argument("--final-repeats", type=int, default=3)
    parser.add_argument("--max-final-attempts", type=int, default=4)
    parser.add_argument("--seed-search", type=Path)
    args = parser.parse_args()
    if not 1 <= args.search_repeats <= 3 or not 1 <= args.final_repeats <= 10:
        raise ValueError("搜索/最终重复次数无效")

    saved = json.loads(args.session.read_text(encoding="utf-8"))
    dense = saved.get("plan", {})
    dense_x = dense.get("reference_wavelengths_nm")
    dense_y = dense.get("reference_values")
    if not isinstance(dense_x, list) or not isinstance(dense_y, list):
        raise ValueError("上次会话没有完整密集光谱")
    automatic, selector = load_half_height_plan(args.session)
    table = tuple(app.load_fullband_accuracy_table())
    layers = candidate_layers(table, automatic, args.candidate_radius)
    if args.output.exists():
        search = json.loads(args.output.read_text(encoding="utf-8"))
        saved_layers = [[int(row["index"]) for row in layer]
                        for layer in search.get("candidate_layers", [])]
        current_layers = [[int(row["index"]) for row in layer] for layer in layers]
        if (search.get("schema") != "temporary_fast_route_search_v1"
                or saved_layers != current_layers
                or int(search.get("feedback_selector", -1)) != selector):
            raise ValueError("已有候选搜索文件与当前密集谱/模拟档位不一致，拒绝续用")
        print(f"RESUME search pairs={len(search['pairs'])} complete={search.get('complete')}", flush=True)
        missing_repeats = any(
            len(search["pairs"].get(_pair_key(source["index"], target["index"]), {})
                .get("captures", [])) < args.search_repeats
            for source, target in required_pairs(layers)
        )
        if missing_repeats:
            search = extend_search(
                args.port, selector, layers, args.output, search, args.search_repeats
            )
        elif not search.get("complete"):
            raise ValueError("候选文件标记未完成但没有可补记录，请检查文件")
    else:
        if args.seed_search is not None:
            seed = json.loads(args.seed_search.read_text(encoding="utf-8"))
            search = seed_search(seed, selector, layers, args.output, args.search_repeats)
            search = extend_search(
                args.port, selector, layers, args.output, search, args.search_repeats
            )
        else:
            search = capture_search(args.port, selector, layers, args.output, args.search_repeats)

    excluded_edges: set[tuple[int, int]] = set()
    excluded_groups: set[tuple[int, ...]] = set()
    rejection_path = args.output.with_name(args.output.stem + "_rejections.json")
    if rejection_path.exists():
        rejected = json.loads(rejection_path.read_text(encoding="utf-8"))
        excluded_edges.update(tuple(pair) for pair in rejected.get("failed_edges", []))
    else:
        # Adopt a completed audit left by an interrupted earlier invocation.
        for prior in args.output.parent.glob(args.output.stem + "_final_attempt*.json"):
            try:
                audited = json.loads(prior.read_text(encoding="utf-8"))
                if (audited.get("complete")
                        and audited.get("safety_cleanup", {}).get("exact_disarm_ack")
                        and audited.get("safety_cleanup", {}).get("exact_soa_shutter_ack")):
                    excluded_edges.update(tuple(pair)
                                          for pair in audited["analysis"]["failed_edges"])
            except Exception:
                pass
    _atomic_json(rejection_path, {"failed_edges": [list(pair)
                                                    for pair in sorted(excluded_edges)]})
    final = None
    chosen = None
    for attempt in range(1, args.max_final_attempts + 1):
        chosen = solve_route(search, layers, excluded_edges=excluded_edges,
                             excluded_groups=excluded_groups)
        route = chosen["rows"]
        destination = args.output.with_name(
            args.output.stem + f"_final_attempt{attempt}.json"
        )
        final = capture_final_route(
            args.port, selector, route, args.final_repeats, destination
        )
        if final["analysis"]["speed_passed"]:
            break
        excluded_edges.update(tuple(pair) for pair in final["analysis"]["failed_edges"])
        _atomic_json(rejection_path, {"failed_edges": [list(pair)
                                                        for pair in sorted(excluded_edges)]})
        print(f"RETRY route attempt={attempt}", flush=True)
    else:
        raise RuntimeError("候选范围内没有通过三次独立主边沿与五点拟合的完整45点路线")

    plan = certified_plan(
        table, chosen["rows"], dense_x, dense_y, final,
        str(args.output.resolve()), fit_checks=chosen["fit_checks"],
        reference_values=chosen["values"],
    )
    reference_path = args.output.with_name(
        f"reference_line_fast_route_{time.time_ns()}.json"
    )
    with serial.Serial(args.port, 2_000_000, timeout=.1, write_timeout=3) as device:
        reference = acquire_reference(
            device, reference_path, selected_rows=plan["rows"],
            feedback_selector=selector,
            on_progress=lambda n, total: print(f"REFERENCE {n}/{total}", flush=True),
        )
    live_values = [float(row["ch1_adc_code"]) for row in reference["rows"]]
    live_checks = []
    for group in range(9):
        start = group * 5
        check = assess_five_point_shape(
            [row["measured_nm"] for row in plan["rows"][start:start + 5]],
            live_values[start:start + 5],
        )
        check["group"] = group + 1
        live_checks.append(check)
    if not all(check["passed"] for check in live_checks):
        failed = ", ".join(str(check["group"]) for check in live_checks if not check["passed"])
        raise RuntimeError(f"最终慢速参考的峰{failed}拟合未通过；未安装路线")
    plan["live_reference_fit_checks"] = live_checks

    fast_path = args.output.with_name(f"fast_replay_{time.time_ns()}.json")
    with serial.Serial(args.port, 2_000_000, timeout=.03, write_timeout=3) as device:
        fast = execute_device(
            device, plan, fast_path, cycles=32, first_delay_us=650,
            boundary_extra_us=0, feedback_selector=selector,
            point_delays_us=[FAST_POINT_DELAY_US] * 45,
        )
    warmed = [frame for frame in fast["frames"]
              if frame.get("kind") == "data" and frame["sequence"] >= 16]
    frame_fits = []
    for frame in warmed:
        values = [row["second_code"] for row in frame["records"]]
        fits = fit_temporary_frame(plan["rows"], values, warmed_up=True)
        frame_fits.append({"sequence": frame["sequence"],
                           "passed": all(item["valid"] for item in fits),
                           "fits": fits})
    if not warmed or not all(item["passed"] for item in frame_fits):
        raise RuntimeError("2.5 ms连续45点复放未做到每帧9/9可拟合；未安装路线")
    plan["forward_fast_stability_certificate"]["production_replay"] = {
        "point_delay_after_final_dac_write_us": FAST_POINT_DELAY_US,
        "frames_checked": len(frame_fits),
        "all_frames_nine_of_nine_fit": True,
        "source": str(fast_path.resolve()),
    }
    install_session(args.session, args.plan_output, plan, selector, reference)
    print(json.dumps({
        "installed": True,
        "route_indices": [[row["index"] for row in plan["rows"][start:start + 5]]
                          for start in range(0, 45, 5)],
        "main_edge_deadline_us": FAST_DEADLINE_US,
        "point_delay_us": FAST_POINT_DELAY_US,
        "slow_reference_9_of_9": True,
        "fast_replay_frames_9_of_9": len(frame_fits),
        "plan": str(args.plan_output.resolve()),
        "session": str(args.session.resolve()),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
