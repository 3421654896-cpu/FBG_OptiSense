"""Board-timestamped 2 ms audit of the temporary half-height 45-point route.

The input is the last saved dense temporary-mode session.  The route is
recomputed with the same selector used by the GUI, but no table or firmware is
modified: the diagnostic addresses exact rows in the installed audited 2001
point table.  Only forward route transitions are exercised.
"""

from __future__ import annotations

import argparse
import binascii
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

import numpy as np

import app_JDSU as app
from analyze_wavelength_switch_board_capture import channel_metrics
from benchmark_stress_realtime import _close_shutter, _safe_disarm
from laser_switching_test import CHANNEL_NAMES, SwitchTester
from temporary_test_mode import LIMITS, select_points


HERE = Path(__file__).resolve().parent
DEFAULT_SESSION = HERE / "outputs" / "temporary_test" / "last_45_session.json"
DEFAULT_OUTPUT = (
    HERE / "outputs" / "temporary_test_half_height_2ms_audit_20260917.json"
)
TWO_MS_US = 2000.0
THREE_MS_US = 3000.0
CUSTOM_RESPONSE_LENGTH = 18 + 12 + 33 * 16 + 2


def custom_command(source: dict, target: dict, tag: int) -> bytes:
    if not 1 <= int(tag) <= 0xFFFFFFFF:
        raise ValueError("诊断标签无效")
    packet = bytearray(808)
    packet[:4] = b"\xff\xff\x03\x01"
    packet[4:6] = int(source["index"]).to_bytes(2, "big")
    packet[6:8] = int(target["index"]).to_bytes(2, "big")
    packet[8] = 5
    packet[9:13] = int(tag).to_bytes(4, "big")
    packet[13:16] = b"C2P"
    for base, row in ((16, source), (26, target)):
        codes = tuple(row["codes"])
        if len(codes) != 5 or any(
            type(code) is not int or not 0 <= code <= limit
            for code, limit in zip(codes, LIMITS)
        ):
            raise ValueError("临时诊断DAC码超出145 mA安全上限")
        for channel, code in enumerate(codes):
            packet[base + 2 * channel:base + 2 * channel + 2] = (
                code.to_bytes(2, "big")
            )
    packet[36:40] = binascii.crc32(packet[:36]).to_bytes(4, "big")
    return bytes(packet)


def decode_custom(packet: bytes, source: dict, target: dict, tag: int) -> dict:
    if (
        len(packet) != CUSTOM_RESPONSE_LENGTH
        or packet[:4] != b"\xd5\x5d\x03\x05"
        or packet[-2:] != b"\x5d\xd5"
    ):
        raise ValueError("临时双行诊断响应格式无效")
    if packet[12:14] != bytes((33, 6)):
        raise ValueError("临时双行诊断ADC尺寸无效")
    received = (
        int.from_bytes(packet[4:6], "big"),
        int.from_bytes(packet[6:8], "big"),
        int.from_bytes(packet[14:18], "big"),
    )
    expected = (int(source["index"]), int(target["index"]), int(tag))
    if received != expected:
        raise ValueError("收到过期或不匹配的临时双行诊断响应")
    update_us = int.from_bytes(packet[8:12], "big")
    baseline = [
        int.from_bytes(packet[position:position + 2], "big")
        for position in range(18, 30, 2)
    ]
    times, samples = [], []
    for sample_index in range(33):
        position = 30 + sample_index * 16
        times.append(int.from_bytes(packet[position:position + 4], "big"))
        samples.append([
            int.from_bytes(packet[offset:offset + 2], "big")
            for offset in range(position + 4, position + 16, 2)
        ])
    if times[0] < update_us or any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("临时双行诊断板上时间戳不递增")
    if any(value > 4095 for row in [baseline, *samples] for value in row):
        raise ValueError("临时双行诊断ADC超出12位范围")
    return {
        "path_update_us": update_us,
        "baseline_adc": baseline,
        "sample_start_from_switch_begin_us": times,
        "sample_start_after_target_write_us": [
            stamp - update_us for stamp in times
        ],
        "direct_adc": samples,
    }


def capture_custom(device, source: dict, target: dict, tag: int) -> dict:
    device.reset_input_buffer()
    device.write(custom_command(source, target, tag))
    received = bytearray()
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        received.extend(device.read(4096))
        start = received.find(b"\xd5\x5d\x03\x05")
        if start >= 0 and len(received) >= start + CUSTOM_RESPONSE_LENGTH:
            return decode_custom(
                bytes(received[start:start + CUSTOM_RESPONSE_LENGTH]),
                source, target, tag,
            )
    raise TimeoutError("未收到临时双行板上时间戳响应（需要固件1.0.74或更高）")


def load_half_height_plan(path: Path) -> tuple[dict, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    saved = payload.get("plan", {})
    wavelengths = saved.get("reference_wavelengths_nm")
    values = saved.get("reference_values")
    if not isinstance(wavelengths, list) or not isinstance(values, list):
        raise ValueError("保存会话中没有完整密集光谱")
    plan = select_points(app.load_fullband_accuracy_table(), wavelengths, values)
    selector = int(payload.get("feedback_selector", 2))
    if selector not in range(4):
        raise ValueError("保存会话的CH1模拟档位无效")
    return plan, selector


def main_edge_metric(records: list[dict], channel: str = "CH1", *,
                     deadline_us: float = THREE_MS_US) -> dict:
    """Judge whether the target's main 10--90 % edge is over by a deadline.

    The long 100 ms tail is retained only as the target level used to locate
    the main edge.  It is *not* used as a post-edge +/-5-code settling gate.
    A transition smaller than 20 ADC codes has no resolvable main edge and is
    classified as already close enough to have no resolvable main edge.  Its
    early excursion is still reported for diagnosis, but is not turned into a
    disguised fixed-code settling gate.  The deadline is end-to-end from the
    first changed DAC write, so the roughly 0.5 ms five-DAC update is included.
    """

    if not records:
        raise ValueError("主边沿分析至少需要一条完整记录")
    metric = channel_metrics(records, channel)
    path_update = float(statistics.median(
        item["path_update_us"] for item in records
    ))
    metric["path_update_us"] = path_update
    metric["deadline_us"] = float(deadline_us)
    metric["criterion"] = "main_10_90_edge_complete_or_no_resolvable_edge"
    if metric["meaningful_step"]:
        t90 = metric.get("t90_us")
        completed = None if t90 is None else path_update + float(t90)
        metric["main_edge_complete_from_switch_begin_us"] = completed
        metric["passed"] = bool(completed is not None and completed <= deadline_us)
        metric["classification"] = (
            "main_edge_complete" if metric["passed"] else "main_edge_late"
        )
    else:
        # If source and target differ by less than the analyzer's resolvable
        # step, reject a hidden early detour larger than that same deadband.
        curves = np.asarray([
            [row[{"CH0": 0, "CH1": 1, "CH2": 2, "CH3": 3,
                  "PDT": 4, "PDR": 5}[channel]]
             for row in item["direct_adc"]]
            for item in records
        ], dtype=float)
        values = np.median(curves, axis=0)
        times = np.median(np.asarray([
            item["sample_start_after_target_write_us"] for item in records
        ], dtype=float), axis=0)
        baseline = float(metric["baseline_code"])
        early = values[times <= max(0.0, deadline_us - path_update)]
        excursion = float(np.max(np.abs(early - baseline))) if early.size else 0.0
        metric["early_excursion_codes"] = excursion
        metric["main_edge_complete_from_switch_begin_us"] = path_update
        metric["passed"] = bool(path_update <= deadline_us)
        metric["classification"] = "no_resolvable_edge"
    return metric


def transition_analysis(records: list[dict], route: list[dict], repeats: int) -> dict:
    transitions = []
    # The user's production route is audited only in its small-to-large
    # direction.  Deliberately exclude the last-to-first reverse wrap.
    for target_order in range(1, len(route)):
        selected = [
            item for item in records
            if int(item["route_target_order"]) == target_order
        ]
        source_order = target_order - 1
        if len(selected) != repeats or any(
            int(item["route_source_order"]) != source_order for item in selected
        ):
            raise ValueError(f"路线跳转{source_order}->{target_order}记录不完整")
        path_update = float(statistics.median(
            item["path_update_us"] for item in selected
        ))
        metrics = {
            channel: channel_metrics(selected, channel)
            for channel in ("PDT", "PDR", "CH1")
        }
        for metric in metrics.values():
            t90 = metric.get("t90_us")
            settle = metric.get("settle_5pct_us")
            metric["t90_from_switch_begin_us"] = (
                None if t90 is None else path_update + float(t90)
            )
            metric["settle_5pct_from_switch_begin_us"] = (
                None if settle is None else path_update + float(settle)
            )
            metric["reached_90pct_within_2ms_after_target_write"] = bool(
                metric["meaningful_step"] and t90 is not None and t90 <= TWO_MS_US
            )
            metric["settled_5pct_within_2ms_after_target_write"] = bool(
                metric["meaningful_step"]
                and settle is not None and settle <= TWO_MS_US
            )
            metric["reached_90pct_within_2ms_end_to_end"] = bool(
                metric["meaningful_step"]
                and metric["t90_from_switch_begin_us"] is not None
                and metric["t90_from_switch_begin_us"] <= TWO_MS_US
            )
            metric["settled_5pct_within_2ms_end_to_end"] = bool(
                metric["meaningful_step"]
                and metric["settle_5pct_from_switch_begin_us"] is not None
                and metric["settle_5pct_from_switch_begin_us"] <= TWO_MS_US
            )
        transitions.append({
            "route_source_order": source_order,
            "route_target_order": target_order,
            "source_fullband_index": int(route[source_order]["index"]),
            "target_fullband_index": int(route[target_order]["index"]),
            "source_wavelength_nm": float(route[source_order]["measured_nm"]),
            "target_wavelength_nm": float(route[target_order]["measured_nm"]),
            "peak_number": int(route[target_order]["group"]),
            "transition_kind": (
                "cross_peak" if target_order % 5 == 0 else
                "within_peak"
            ),
            "path_update_us": path_update,
            "channels": metrics,
        })

    meaningful = [
        item for item in transitions if item["channels"]["CH1"]["meaningful_step"]
    ]
    peak_entries = [
        item for item in transitions
        if item["transition_kind"] == "cross_peak"
    ]

    def counts(items: list[dict]) -> dict:
        measured = [
            item for item in items if item["channels"]["CH1"]["meaningful_step"]
        ]
        return {
            "transition_count": len(items),
            "meaningful_ch1_transition_count": len(measured),
            "ch1_t90_within_2ms_after_write_count": sum(
                item["channels"]["CH1"][
                    "reached_90pct_within_2ms_after_target_write"
                ] for item in measured
            ),
            "ch1_settled_within_2ms_after_write_count": sum(
                item["channels"]["CH1"][
                    "settled_5pct_within_2ms_after_target_write"
                ] for item in measured
            ),
            "ch1_t90_within_2ms_end_to_end_count": sum(
                item["channels"]["CH1"][
                    "reached_90pct_within_2ms_end_to_end"
                ] for item in measured
            ),
            "ch1_settled_within_2ms_end_to_end_count": sum(
                item["channels"]["CH1"][
                    "settled_5pct_within_2ms_end_to_end"
                ] for item in measured
            ),
        }

    return {
        "criterion_us": TWO_MS_US,
        "direction": "strictly forward, small wavelength to large wavelength",
        "reverse_wrap_tested": False,
        "criterion": (
            "CH1 meaningful step reaches 90% and remains within 5% of its "
            "100 ms tail; board time is reported both after the final target "
            "DAC write and end-to-end from the first changed DAC write"
        ),
        "all_forward_transitions": counts(transitions),
        "peak_entry_transitions": counts(peak_entries),
        "meaningful_transition_count": len(meaningful),
        "all_meaningful_ch1_settled_within_2ms_after_write": bool(
            meaningful and all(
                item["channels"]["CH1"][
                    "settled_5pct_within_2ms_after_target_write"
                ] for item in meaningful
            )
        ),
        "all_meaningful_ch1_settled_within_2ms_end_to_end": bool(
            meaningful and all(
                item["channels"]["CH1"][
                    "settled_5pct_within_2ms_end_to_end"
                ] for item in meaningful
            )
        ),
        "transitions": transitions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        raise ValueError("重复次数必须为1～10")
    if args.output.exists():
        raise FileExistsError(args.output)

    plan, selector = load_half_height_plan(args.session)
    route = plan["rows"]
    report = {
        "schema": "temporary_half_height_board_2ms_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "session_source": str(args.session.resolve()),
        "port": args.port,
        "ch1_feedback_selector": selector,
        "repeats": args.repeats,
        "channels": list(CHANNEL_NAMES),
        "route": route,
        "selection": {
            "rule": plan["selection_rule"],
            "height_fraction": plan["selection_height_fraction"],
            "endpoint_checks": plan["endpoint_checks"],
            "dense_preview_fit_checks": plan["selection_fit_checks"],
        },
        "captures": [],
        "complete": False,
    }
    tester = SwitchTester(args.port)
    try:
        tester.serial.dtr = True
        worker = app.EqualIntervalWorker(
            tester.serial, (), {}, settle_s=.3,
            feedback_selectors=(0, selector),
        )
        if not worker._feedback_command_and_ack():
            raise RuntimeError("CH1模拟档位设置未收到确认")
        tag = int(time.time_ns() & 0x7FFFFFFF) or 1
        for repeat in range(args.repeats):
            for target_order in range(1, 45):
                target = route[target_order]
                source_order = target_order - 1
                source = route[source_order]
                tag = (tag + 1) & 0xFFFFFFFF or 1
                item = capture_custom(tester.serial, source, target, tag)
                item.update({
                    "repeat": repeat,
                    "route_source_order": source_order,
                    "route_target_order": target_order,
                    "source_fullband_index": int(source["index"]),
                    "target_fullband_index": int(target["index"]),
                    "request_tag": tag,
                })
                report["captures"].append(item)
            print(
                f"repeat={repeat + 1}/{args.repeats} "
                f"captures={len(report['captures'])}", flush=True
            )
        report["analysis"] = transition_analysis(
            report["captures"], route, args.repeats
        )
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
            "exact_disarm_ack": disarmed,
            "exact_soa_shutter_ack": shuttered,
            "errors": errors,
        }
        tester.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"disarm_confirmed={int(disarmed)} "
            f"soa_shutter_confirmed={int(shuttered)}", flush=True
        )
        if not disarmed or not shuttered:
            raise RuntimeError("安全停光确认失败")

    print(json.dumps(report["analysis"], ensure_ascii=False, indent=2))
    print(f"report={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
