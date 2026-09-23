"""Summarize board-timestamped forward wavelength-switch captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CHANNELS = ("CH0", "CH1", "CH2", "CH3", "PDT", "PDR")
CHANNEL_INDEX = {name: index for index, name in enumerate(CHANNELS)}
MIN_STEP_CODES = 20.0
ADC_GROUP_TIME_UNCERTAINTY_US = 55.0


def category(target: int) -> str:
    if target == 0:
        return "frame_wrap"
    if target % 5 == 0:
        return "cross_peak"
    return "within_peak"


def crossing_time(times: np.ndarray, progress: np.ndarray, threshold: float) -> float | None:
    prior_time = 0.0
    prior_value = 0.0
    for current_time, current_value in zip(times, progress):
        if prior_value < threshold <= current_value:
            width = current_value - prior_value
            if abs(width) < 1e-12:
                return float(current_time)
            fraction = (threshold - prior_value) / width
            return float(prior_time + fraction * (current_time - prior_time))
        prior_time = float(current_time)
        prior_value = float(current_value)
    return None


def channel_metrics(records: list[dict], channel: str) -> dict:
    index = CHANNEL_INDEX[channel]
    times = np.median(
        np.asarray([record["sample_start_after_target_write_us"] for record in records], dtype=float),
        axis=0,
    )
    values = np.median(
        np.asarray([[row[index] for row in record["direct_adc"]] for record in records], dtype=float),
        axis=0,
    )
    baseline = float(np.median([record["baseline_adc"][index] for record in records]))
    final = float(np.median(values[-3:]))
    amplitude = final - baseline
    result = {
        "baseline_code": baseline,
        "final_tail_code": final,
        "amplitude_codes": amplitude,
        "meaningful_step": abs(amplitude) >= MIN_STEP_CODES,
        "times_us": times.tolist(),
        "median_values": values.tolist(),
    }
    if not result["meaningful_step"]:
        return result
    progress = (values - baseline) / amplitude
    t10 = crossing_time(times, progress, 0.10)
    t90 = crossing_time(times, progress, 0.90)
    tolerance = max(5.0, 0.05 * abs(amplitude))
    settle = None
    for sample_index, stamp in enumerate(times):
        if np.all(np.abs(values[sample_index:] - final) <= tolerance):
            settle = float(stamp)
            break
    result.update({
        "t10_us": t10,
        "t90_us": t90,
        "main_edge_10_90_us": None if t10 is None or t90 is None else t90 - t10,
        "settle_5pct_us": settle,
        "settle_tolerance_codes": tolerance,
        "normalized_progress": progress.tolist(),
    })
    return result


def distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "median": None, "p90": None, "max": None}
    return {
        "count": len(values),
        "median": float(statistics.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(max(values)),
    }


def analyze(payload: dict) -> dict:
    if not payload.get("complete"):
        raise ValueError("capture is incomplete")
    cleanup = payload.get("safety_cleanup", {})
    if not cleanup.get("exact_disarm_ack") or not cleanup.get("exact_soa_shutter_ack"):
        raise ValueError("capture safety cleanup was not confirmed")
    records = payload.get("captures", [])
    repeat_count = len(records) // 45 if len(records) % 45 == 0 else 0
    if not repeat_count or len(records) != 45 * repeat_count:
        raise ValueError("capture does not contain every forward transition")

    transitions = []
    for target in range(45):
        selected = [record for record in records if int(record["target_index"]) == target]
        expected_source = (target - 1) % 45
        if len(selected) != repeat_count or any(
            int(record["source_index"]) != expected_source for record in selected
        ):
            raise ValueError(f"transition {expected_source}->{target} is incomplete")
        transition = {
            "source_index": expected_source,
            "target_index": target,
            "category": category(target),
            "path_update_us": {
                "median": float(statistics.median(r["path_update_us"] for r in selected)),
                "min": int(min(r["path_update_us"] for r in selected)),
                "max": int(max(r["path_update_us"] for r in selected)),
            },
            "first_adc_group_start_after_target_write_us": float(
                statistics.median(r["sample_start_after_target_write_us"][0] for r in selected)
            ),
            "channels": {
                channel: channel_metrics(selected, channel)
                for channel in ("PDT", "PDR", "CH1")
            },
        }
        transitions.append(transition)

    summaries = {}
    for name in ("all", "within_peak", "cross_peak", "frame_wrap"):
        selected = transitions if name == "all" else [t for t in transitions if t["category"] == name]
        item = {
            "transition_count": len(selected),
            "path_update_us": distribution([t["path_update_us"]["median"] for t in selected]),
            "channels": {},
        }
        for channel in ("PDT", "PDR", "CH1"):
            meaningful = [t["channels"][channel] for t in selected if t["channels"][channel]["meaningful_step"]]
            edges = [m["main_edge_10_90_us"] for m in meaningful if m.get("main_edge_10_90_us") is not None]
            settles = [m["settle_5pct_us"] for m in meaningful if m.get("settle_5pct_us") is not None]
            item["channels"][channel] = {
                "meaningful_transition_count": len(meaningful),
                "main_edge_10_90_us": distribution(edges),
                "settle_5pct_us": distribution(settles),
                "unsettled_by_100ms_count": len(meaningful) - len(settles),
            }
        summaries[name] = item

    meaningful_ch1 = [
        t for t in transitions if t["channels"]["CH1"]["meaningful_step"]
    ]
    slowest = sorted(
        meaningful_ch1,
        key=lambda t: t["channels"]["CH1"].get("settle_5pct_us") or float("inf"),
        reverse=True,
    )[:8]
    return {
        "schema": "wavelength_switch_board_analysis_v1",
        "scope": "installed temporary 45-point table, forward predecessor-to-target transitions only",
        "repeats_per_transition": repeat_count,
        "adc_group_timestamp_scope": "six-channel group start",
        "adc_group_time_uncertainty_us": ADC_GROUP_TIME_UNCERTAINTY_US,
        "meaningful_step_threshold_codes": MIN_STEP_CODES,
        "summaries": summaries,
        "slowest_meaningful_ch1_transitions": [
            {
                "source_index": t["source_index"],
                "target_index": t["target_index"],
                "category": t["category"],
                "amplitude_codes": t["channels"]["CH1"]["amplitude_codes"],
                "main_edge_10_90_us": t["channels"]["CH1"].get("main_edge_10_90_us"),
                "settle_5pct_us": t["channels"]["CH1"].get("settle_5pct_us"),
            }
            for t in slowest
        ],
        "transitions": transitions,
    }


def plot(result: dict, destination: Path) -> None:
    colors = {"within_peak": "#3178c6", "cross_peak": "#d67820", "frame_wrap": "#a53aa5"}
    figure, axes = plt.subplots(2, 1, figsize=(12, 8.4), layout="constrained")
    for transition in result["transitions"]:
        metric = transition["channels"]["CH1"]
        if not metric["meaningful_step"]:
            continue
        axes[0].plot(
            np.asarray(metric["times_us"]) / 1000.0,
            metric["normalized_progress"],
            color=colors[transition["category"]], alpha=0.25, linewidth=0.9,
        )
    axes[0].axhline(0.9, color="#777777", linestyle="--", linewidth=0.8)
    axes[0].set_xscale("symlog", linthresh=0.1)
    axes[0].set(
        title="CH1 normalized response: 45 forward wavelength transitions × 3",
        xlabel="Board time after target write (ms)", ylabel="Normalized target progress",
    )
    axes[0].grid(alpha=0.2)

    for transition in result["transitions"]:
        metric = transition["channels"]["CH1"]
        if not metric["meaningful_step"]:
            continue
        target = transition["target_index"]
        color = colors[transition["category"]]
        edge = metric.get("main_edge_10_90_us")
        settle = metric.get("settle_5pct_us")
        if edge is not None:
            axes[1].scatter(target, edge / 1000.0, color=color, marker="o", s=28)
        if settle is not None:
            axes[1].scatter(target, settle / 1000.0, color=color, marker="x", s=32)
    axes[1].set(
        title="CH1 timing by target row (circle: 10–90%; cross: 5% settling)",
        xlabel="Target row in installed 45-point order", ylabel="Time (ms)",
    )
    axes[1].grid(alpha=0.2)
    figure.savefig(destination, dpi=170)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".png").exists():
        raise FileExistsError(args.output)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = analyze(payload)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    plot(result, args.output.with_suffix(".png"))
    print(json.dumps({"summaries": result["summaries"],
                      "slowest_ch1": result["slowest_meaningful_ch1_transitions"],
                      "output": str(args.output.resolve())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
