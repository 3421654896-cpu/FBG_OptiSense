"""Prove that the deployed F45 runtime reproduces the locked blind replay."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from fast_fullmap_realtime import FastFullMapRealtimeLocalizer


def validate(model_path: Path, manifest_path: Path, locked_report_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text("utf-8"))
    locked = json.loads(locked_report_path.read_text("utf-8"))
    expected = {int(row["truth"]): row for row in locked["per_point"]}
    if not (locked.get("static_position_gate") or {}).get("passed"):
        raise ValueError("locked blind validation did not pass")

    per_point = []
    localizer = FastFullMapRealtimeLocalizer(model_path)
    total = exact = majority_exact = 0
    for point in range(1, 18):
        capture_path = (manifest_path.parent / manifest["round"][str(point)]).resolve()
        capture = json.loads(capture_path.read_text("utf-8"))
        receipts = {
            int(item["sequence"]): int(item["host_monotonic_ns"])
            for item in capture["commissioning"]["frame_receipts"]
        }
        events = {
            item["kind"]: int(item["host_monotonic_ns"])
            for item in capture["commissioning"]["events"]
        }
        held_sequences = {
            sequence
            for sequence, received in receipts.items()
            if events["contact_reached_commanded_depth"] + 200_000_000
            <= received
            < events["release_started"]
        }
        localizer.reset()
        predictions = []
        for frame in capture["frames"]:
            result = localizer.update(frame)
            if int(frame["sequence"]) in held_sequences:
                if not result.contact_detected or result.predicted_point is None:
                    raise ValueError(f"P{point}: held frame was not classified")
                predictions.append(int(result.predicted_point))
        counts = Counter(predictions)
        majority = min(counts, key=lambda label: (-counts[label], label))
        row = {
            "truth": point,
            "frames": len(predictions),
            "exact_frames": int(sum(value == point for value in predictions)),
            "majority": int(majority),
            "prediction_counts": {
                str(label): int(count) for label, count in sorted(counts.items())
            },
        }
        reference = expected[point]
        for field in ("frames", "exact_frames", "majority", "prediction_counts"):
            if row[field] != reference[field]:
                raise ValueError(
                    f"P{point}: runtime {field} differs from locked replay"
                )
        per_point.append(row)
        total += len(predictions)
        exact += row["exact_frames"]
        majority_exact += int(majority == point)

    frame_exact = 100.0 * exact / total
    if (
        total != int(locked["frames"])
        or majority_exact != int(locked["trial_majority_exact"])
        or abs(frame_exact - float(locked["frame_exact_percent"])) > 1e-12
    ):
        raise ValueError("aggregate runtime metrics differ from locked replay")
    return {
        "schema": "f45_runtime_equivalence_validation_v1",
        "runtime_model": str(model_path.resolve()),
        "locked_validation_report": str(locked_report_path.resolve()),
        "validation_manifest": str(manifest_path.resolve()),
        "contacts": 17,
        "frames": total,
        "trial_majority_exact": majority_exact,
        "frame_exact_percent": frame_exact,
        "per_point": per_point,
        "exact_prediction_counts_match_locked_replay": True,
        "passed": True,
        "physical_15hz_verified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("validation_manifest", type=Path)
    parser.add_argument("locked_validation_report", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    result = validate(
        args.model.resolve(),
        args.validation_manifest.resolve(),
        args.locked_validation_report.resolve(),
    )
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({key: result[key] for key in (
        "contacts", "frames", "trial_majority_exact", "frame_exact_percent",
        "exact_prediction_counts_match_locked_replay", "passed",
        "physical_15hz_verified",
    )}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
