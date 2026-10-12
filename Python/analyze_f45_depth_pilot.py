"""Audit F45 response versus commanded indentation at one coverage point.

This is a displacement-proxy study.  It does not claim measured force or a
deformation-dependent physical contact area.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from fast_fullmap_realtime import FastFullMapRealtimeLocalizer


EXPECTED_DEPTHS = (0.10, 0.20, 0.40, 0.60, 0.80)
HOLD_SETTLE_NS = 200_000_000


def _event_times(commission: dict) -> dict[str, int]:
    return {
        str(event["kind"]): int(event["host_monotonic_ns"])
        for event in commission["events"]
    }


def _matrix(frames: list[dict], field: str) -> np.ndarray:
    values = np.asarray(
        [[record[field] for record in frame["records"]] for frame in frames],
        dtype=float,
    )
    if values.ndim != 2 or values.shape[1] != 45 or not np.isfinite(values).all():
        raise ValueError(f"Malformed F45 {field} matrix")
    return values


def load_capture(path: Path, expected_point: int) -> dict:
    data = json.loads(path.read_text("utf-8"))
    commission = data.get("commissioning") or {}
    safety = data.get("safety_cleanup") or {}
    if data.get("schema") != "ch1_fast_fullmap_stream_capture_v1":
        raise ValueError(f"{path}: not an F45 capture")
    if not data.get("complete") or len(data.get("frames") or ()) < 640:
        raise ValueError(f"{path}: incomplete capture")
    if commission.get("capture_error") or commission.get("motion_failures"):
        raise ValueError(f"{path}: capture/motion failure")
    if not safety.get("exact_disarm_ack") or not safety.get("exact_soa_shutter_ack"):
        raise ValueError(f"{path}: optical shutdown is not proven")
    if int(commission["target"]["point_index"]) != expected_point:
        raise ValueError(f"{path}: unexpected coverage point")

    frames_by_sequence = {
        int(frame["sequence"]): frame for frame in data["frames"]
    }
    receipts = {
        int(item["sequence"]): int(item["host_monotonic_ns"])
        for item in commission["frame_receipts"]
    }
    events = _event_times(commission)
    trigger = int(commission["trigger_sequence"])
    baseline_frames = [
        frames_by_sequence[sequence]
        for sequence in sorted(receipts)
        if 16 <= sequence < trigger
    ]
    held_sequences = [
        sequence
        for sequence, received in sorted(receipts.items())
        if events["contact_reached_commanded_depth"] + HOLD_SETTLE_NS
        <= received
        < events["release_started"]
    ]
    held_frames = [frames_by_sequence[sequence] for sequence in held_sequences]
    if len(baseline_frames) < 64 or len(held_frames) < 20:
        raise ValueError(f"{path}: too few baseline/held frames")

    baseline_first = _matrix(baseline_frames, "first_code")
    baseline_second = _matrix(baseline_frames, "second_code")
    held_first = _matrix(held_frames, "first_code")
    held_second = _matrix(held_frames, "second_code")
    centre_first = np.median(baseline_first, axis=0)
    centre_second = np.median(baseline_second, axis=0)
    first_delta = held_first - centre_first
    second_delta = held_second - centre_second
    transition = (held_second - held_first) - (centre_second - centre_first)
    frame_rms = np.sqrt(np.mean(first_delta**2, axis=1))
    baseline_rms = np.sqrt(np.mean((baseline_first - centre_first) ** 2, axis=1))
    median_delta = np.median(first_delta, axis=0)

    localizer = FastFullMapRealtimeLocalizer()
    results = {}
    for frame in data["frames"]:
        result = localizer.update(frame)
        results[int(frame["sequence"])] = result
    held_results = [results[sequence] for sequence in held_sequences]
    predictions = [
        int(result.predicted_point)
        for result in held_results
        if result.contact_detected and result.predicted_point is not None
    ]
    majority = Counter(predictions).most_common(1)
    confidences = [
        float(result.confidence)
        for result in held_results
        if result.contact_detected
    ]
    threshold = float(localizer.activation_threshold_codes)
    depth = float(commission["depth_mm"])
    return {
        "path": str(path.resolve()),
        "depth_mm": depth,
        "baseline_frames": len(baseline_frames),
        "held_frames": len(held_frames),
        "baseline_rms_codes": {
            "median": float(np.median(baseline_rms)),
            "p95": float(np.percentile(baseline_rms, 95)),
            "maximum": float(np.max(baseline_rms)),
        },
        "held_first_delta_rms_codes": {
            "median_frame": float(np.median(frame_rms)),
            "p05_frame": float(np.percentile(frame_rms, 5)),
            "p95_frame": float(np.percentile(frame_rms, 95)),
            "rms_of_median_vector": float(np.sqrt(np.mean(median_delta**2))),
            "maximum_absolute_median_point": float(np.max(np.abs(median_delta))),
        },
        "held_second_delta_rms_codes": float(
            np.sqrt(np.mean(np.median(second_delta, axis=0) ** 2))
        ),
        "held_transition_rms_codes": float(
            np.sqrt(np.mean(np.median(transition, axis=0) ** 2))
        ),
        "grating_first_delta_rms_codes": [
            float(value)
            for value in np.sqrt(np.mean(median_delta.reshape(9, 5) ** 2, axis=1))
        ],
        "activation": {
            "locked_0p30mm_threshold_codes": threshold,
            "detected_held_frames": len(predictions),
            "detected_held_percent": float(100.0 * len(predictions) / len(held_results)),
        },
        "position": {
            "truth_point": expected_point,
            "majority_prediction": majority[0][0] if majority else None,
            "majority_votes": majority[0][1] if majority else 0,
            "exact_detected_frame_percent": (
                float(100.0 * sum(point == expected_point for point in predictions) / len(predictions))
                if predictions
                else None
            ),
            "median_probability_margin": (
                float(np.median(confidences)) if confidences else None
            ),
            "prediction_counts": {
                str(point): int(count)
                for point, count in sorted(Counter(predictions).items())
            },
        },
        "sampling": {
            "board_rate_hz": float(data["analysis"]["board_rate_hz"]),
            "maximum_interval_us": int(data["analysis"]["board_interval_max_us"]),
            "all_45_points_fresh": bool(
                data["analysis"].get("fresh_point_count_per_cycle") == 45
            ),
        },
    }


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    for value in np.unique(values):
        indices = np.flatnonzero(values == value)
        ranks[indices] = float(np.mean(ranks[indices]))
    return ranks


def correlation(x: list[float], y: list[float]) -> dict:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if len(x_values) < 2 or np.std(y_values) <= 1e-12:
        return {"pearson": None, "spearman": None}
    return {
        "pearson": float(np.corrcoef(x_values, y_values)[0, 1]),
        "spearman": float(np.corrcoef(_rank(x_values), _rank(y_values))[0, 1]),
    }


def select_latest_valid(root: Path, point: int) -> list[dict]:
    selected = []
    for depth in EXPECTED_DEPTHS:
        token = f"{depth:.2f}".replace(".", "p")
        candidates = sorted(
            root.glob(f"f45_depthpilot_p{point:02d}_{token}_*/capture.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        failures = []
        for path in candidates:
            try:
                selected.append(load_capture(path, point))
                break
            except (KeyError, TypeError, ValueError) as exc:
                failures.append(str(exc))
        else:
            raise ValueError(
                f"No valid capture for point {point}, depth {depth:.2f}: {failures}"
            )
    return sorted(selected, key=lambda item: item["depth_mm"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--point-index", type=int, default=9)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output must be new")

    captures = select_latest_valid(args.outputs, args.point_index)
    depths = [float(item["depth_mm"]) for item in captures]
    responses = [
        float(item["held_first_delta_rms_codes"]["median_frame"])
        for item in captures
    ]
    result = {
        "schema": "f45_single_point_depth_pilot_v1",
        "point_index": int(args.point_index),
        "depths_mm": depths,
        "captures": captures,
        "response_vs_commanded_depth": {
            "quantity": "median held-frame RMS of 45 first-code deltas",
            "values_codes": responses,
            "strictly_increasing": all(
                right > left for left, right in zip(responses, responses[1:])
            ),
            "correlation": correlation(depths, responses),
        },
        "interpretation_limits": {
            "force_measured": False,
            "pressure_sensor_present": False,
            "physical_contact_area_measured": False,
            "nominal_8mm_tool_face_area_mm2": math.pi * 4.0**2,
            "physical_15hz_pressure_response_verified": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
