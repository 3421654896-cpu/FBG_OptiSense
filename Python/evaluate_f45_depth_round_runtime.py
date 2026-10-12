"""Replay one visible F45 depth round through the deployed runtime model."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from fast_fullmap_realtime import FastFullMapRealtimeLocalizer


def held_sequences(data: dict) -> list[int]:
    commission = data["commissioning"]
    events = {
        event["kind"]: int(event["host_monotonic_ns"])
        for event in commission["events"]
    }
    receipts = {
        int(item["sequence"]): int(item["host_monotonic_ns"])
        for item in commission["frame_receipts"]
    }
    return [
        sequence
        for sequence, received in sorted(receipts.items())
        if events["contact_reached_commanded_depth"] + 200_000_000
        <= received
        < events["release_started"]
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output must be new")
    manifest = json.loads(args.manifest.read_text("utf-8"))
    if not manifest.get("complete") or manifest.get("completed_points") != list(range(1, 18)):
        raise ValueError("round is not a verified 17-point capture")
    coordinates = {}
    payloads = {}
    for point in range(1, 18):
        path = Path(manifest["captures"][str(point)]["capture"])
        data = json.loads(path.read_text("utf-8"))
        payloads[point] = data
        target = data["commissioning"]["target"]
        coordinates[point] = np.asarray(
            target.get("calibration_work_xy_mm", [target["work_x"], target["work_y"]]),
            dtype=float,
        )

    per_point = []
    total = exact = within = detected = 0
    for point, data in payloads.items():
        localizer = FastFullMapRealtimeLocalizer()
        replay = {
            int(frame["sequence"]): localizer.update(frame)
            for frame in data["frames"]
        }
        held = [replay[sequence] for sequence in held_sequences(data)]
        predictions = [
            int(result.predicted_point)
            for result in held
            if result.contact_detected and result.predicted_point is not None
        ]
        counts = Counter(predictions)
        majority = counts.most_common(1)[0][0] if counts else None
        errors = [
            float(np.linalg.norm(coordinates[prediction] - coordinates[point]))
            for prediction in predictions
        ]
        total += len(held)
        detected += len(predictions)
        exact += sum(prediction == point for prediction in predictions)
        within += sum(error <= 4.0 + 1e-6 for error in errors)
        per_point.append(
            {
                "truth": point,
                "held_frames": len(held),
                "detected_frames": len(predictions),
                "majority": majority,
                "majority_exact": majority == point,
                "exact_detected_frame_percent": (
                    100.0 * sum(prediction == point for prediction in predictions) / len(predictions)
                    if predictions
                    else None
                ),
                "prediction_counts": {str(k): int(v) for k, v in sorted(counts.items())},
                "p95_error_mm": float(np.percentile(errors, 95)) if errors else None,
            }
        )
    result = {
        "schema": "f45_deployed_runtime_depth_round_replay_v1",
        "manifest": str(args.manifest.resolve()),
        "depth_mm": float(manifest["depth_mm"]),
        "contacts": 17,
        "trial_majority_exact": sum(row["majority_exact"] for row in per_point),
        "trial_majority_exact_percent": 100.0 * sum(row["majority_exact"] for row in per_point) / 17.0,
        "held_frames": total,
        "detected_frames": detected,
        "detected_percent": 100.0 * detected / total,
        "exact_detected_frame_percent": 100.0 * exact / detected if detected else None,
        "within_4mm_detected_frame_percent": 100.0 * within / detected if detected else None,
        "per_point": per_point,
        "conclusion": "diagnostic_only_model_was_trained_at_0p30mm",
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_point"}, ensure_ascii=False, indent=2))
    print(json.dumps(result["per_point"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
