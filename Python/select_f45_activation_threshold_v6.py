"""Select an F45 contact threshold without mixing physical acquisition rounds.

The position model and contact detector serve different purposes.  Position is
decoded from the 135-value transition feature, while contact is detected from
the RMS of the first 45 baseline-referenced ADC deltas.  This audit derives a
threshold from complete development captures and reports whole-round LOSO
behaviour for both held-contact detection and no-contact false positives.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path: Path) -> list[dict]:
    value = json.loads(path.read_text("utf-8"))
    if value.get("schema") != "f45_visible_depth_round_v1" or not value.get("complete"):
        raise ValueError(f"{path}: incomplete or unexpected round manifest")
    name = str(value["round_name"])
    result = []
    for point in value["planned_points"]:
        entry = value["captures"][str(point)]
        capture = Path(entry["capture"])
        if sha256(capture) != entry["capture_sha256"] or not all(entry["checks"].values()):
            raise ValueError(f"{capture}: guarded capture hash/check mismatch")
        result.append(
            {
                "round": name,
                "point": int(point),
                "depth_mm": float(value["depth_mm"]),
                "path": str(capture.resolve()),
                "sha256": sha256(capture),
            }
        )
    return result


def energy_samples(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text("utf-8"))
    commissioning = payload["commissioning"]
    frames = {int(frame["sequence"]): frame for frame in payload["frames"]}
    receipts = {
        int(item["sequence"]): int(item["host_monotonic_ns"])
        for item in commissioning["frame_receipts"]
    }
    events = {
        item["kind"]: int(item["host_monotonic_ns"])
        for item in commissioning["events"]
    }
    baseline_frames = [frames[sequence] for sequence in range(16, 80)]
    held_frames = [
        frame
        for sequence, frame in sorted(frames.items())
        if events["contact_reached_commanded_depth"] + 200_000_000
        <= receipts[sequence]
        < events["release_started"]
    ]
    if len(baseline_frames) != 64 or len(held_frames) < 25:
        raise ValueError(f"{path}: insufficient baseline or held frames")

    def first_matrix(selected: list[dict]) -> np.ndarray:
        values = np.asarray(
            [[record["first_code"] for record in frame["records"]] for frame in selected],
            dtype=float,
        )
        if values.shape[1:] != (45,) or not np.isfinite(values).all():
            raise ValueError(f"{path}: malformed first-code samples")
        return values

    baseline = first_matrix(baseline_frames)
    held = first_matrix(held_frames)
    reference = np.median(baseline, axis=0)
    baseline_energy = np.sqrt(np.mean((baseline - reference) ** 2, axis=1))
    held_energy = np.sqrt(np.mean((held - reference) ** 2, axis=1))
    return baseline_energy, held_energy


def select_threshold(baseline: np.ndarray, held: np.ndarray) -> dict:
    lower = float(np.max(baseline))
    upper = float(np.min(held))
    separated = lower < upper
    if separated:
        # The geometric centre gives comparable multiplicative guard on both
        # sides when a strict zero-error interval exists.
        threshold = float(np.sqrt(lower * upper)) if lower > 0 else 0.5 * upper
    else:
        # When distributions overlap, show the least-false-positive scalar
        # threshold that still detects every development contact frame.  This
        # is diagnostic only and must not be silently declared deployable.
        threshold = upper
    return {
        "threshold_codes": threshold,
        "maximum_baseline_energy_codes": lower,
        "minimum_held_energy_codes": upper,
        "separation_margin_codes": upper - lower,
        "baseline_guard_ratio": threshold / lower if lower else None,
        "held_guard_ratio": upper / threshold if threshold else None,
        "strict_zero_error_interval_exists": separated,
    }


def metrics(records: list[dict], threshold: float) -> dict:
    baseline = np.concatenate([row["baseline_energy"] for row in records])
    held = np.concatenate([row["held_energy"] for row in records])
    return {
        "captures": len(records),
        "baseline_frames": int(len(baseline)),
        "held_frames": int(len(held)),
        "baseline_false_positive_frames": int(np.sum(baseline >= threshold)),
        "baseline_false_positive_percent": float(100.0 * np.mean(baseline >= threshold)),
        "held_detected_frames": int(np.sum(held >= threshold)),
        "held_detected_percent": float(100.0 * np.mean(held >= threshold)),
        "baseline_energy_p999_codes": float(np.percentile(baseline, 99.9)),
        "held_energy_p001_codes": float(np.percentile(held, 0.1)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--development-round", action="append", default=[], type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output must be new")

    lock = json.loads(args.lock.read_text("utf-8"))
    captures = [dict(item) for item in lock["training"]["captures"]]
    for manifest in args.development_round:
        captures.extend(load_manifest(manifest))
    unique = {}
    for item in captures:
        path = Path(item["path"])
        digest = sha256(path)
        if digest != item["sha256"]:
            raise ValueError(f"{path}: capture changed")
        unique[digest] = item
    captures = list(unique.values())

    rows = []
    for item in captures:
        baseline, held = energy_samples(Path(item["path"]))
        rows.append({**item, "baseline_energy": baseline, "held_energy": held})

    by_round: dict[str, list[dict]] = {}
    for row in rows:
        by_round.setdefault(str(row["round"]), []).append(row)
    baseline_all = np.concatenate([row["baseline_energy"] for row in rows])
    held_all = np.concatenate([row["held_energy"] for row in rows])
    selected = select_threshold(baseline_all, held_all)
    threshold = float(selected["threshold_codes"])

    per_round = {
        name: {
            "depth_mm": float(np.median([float(row["depth_mm"]) for row in records])),
            **metrics(records, threshold),
        }
        for name, records in sorted(by_round.items())
    }
    loso = []
    for held_name, held_records in sorted(by_round.items()):
        train_records = [row for name, records in by_round.items() if name != held_name for row in records]
        train_baseline = np.concatenate([row["baseline_energy"] for row in train_records])
        train_held = np.concatenate([row["held_energy"] for row in train_records])
        fold = select_threshold(train_baseline, train_held)
        loso.append(
            {
                "held_round": held_name,
                "trained_threshold_codes": fold["threshold_codes"],
                **metrics(held_records, float(fold["threshold_codes"])),
            }
        )

    result = {
        "schema": "f45_activation_threshold_selection_v6",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_lock": {"path": str(args.lock.resolve()), "sha256": sha256(args.lock)},
        "quantity": "RMS of 45 first-code deltas from a causal 64-frame local baseline",
        "random_frame_split_used": False,
        "development_rounds": len(by_round),
        "development_captures": len(rows),
        "selection": selected,
        "development_metrics": metrics(rows, threshold),
        "per_round": per_round,
        "whole_round_loso": loso,
        "gates": {
            "strict_zero_error_interval_exists": bool(
                selected["strict_zero_error_interval_exists"]
            ),
            "zero_development_baseline_false_positives": all(
                row["baseline_false_positive_frames"] == 0 for row in per_round.values()
            ),
            "all_development_held_frames_detected": all(
                row["held_detected_percent"] == 100.0 for row in per_round.values()
            ),
            "loso_zero_baseline_false_positives": all(
                row["baseline_false_positive_frames"] == 0 for row in loso
            ),
            "loso_all_held_frames_detected": all(
                row["held_detected_percent"] == 100.0 for row in loso
            ),
        },
        "deployment_eligible": False,
        "physical_15hz_verified": False,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"per_round", "whole_round_loso"}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
