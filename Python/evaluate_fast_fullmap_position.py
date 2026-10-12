"""Evaluate F45 contact position repeats without commanding any hardware."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from evaluate_coverage_position import _feature_variants, _transform


EXPECTED_ROWS = tuple(range(45))
MIN_HELD_FRAMES = 20


def _capture_feature(path: Path) -> dict:
    data = json.loads(path.read_text("utf-8"))
    commission = data.get("commissioning") or {}
    if data.get("schema") != "ch1_fast_fullmap_stream_capture_v1":
        raise ValueError(f"{path.name}: not an F45 capture")
    if commission.get("schema") != "bounded_indentation_fast_fullmap_commission_v1":
        raise ValueError(f"{path.name}: not a guarded F45 contact")
    if not data.get("complete") or len(data.get("frames", [])) < 640:
        raise ValueError(f"{path.name}: incomplete capture")
    if tuple(data.get("route_rows") or ()) != EXPECTED_ROWS:
        raise ValueError(f"{path.name}: full-map route mismatch")
    if int(data.get("spacing_us", -1)) != 50:
        raise ValueError(f"{path.name}: expected the qualified 50 us spacing")
    if not (data.get("analysis") or {}).get("meets_sampling_rate_for_15hz"):
        raise ValueError(f"{path.name}: strict >30 Hz sampling gate failed")
    safety = data.get("safety_cleanup") or {}
    if not safety.get("exact_disarm_ack") or not safety.get("exact_soa_shutter_ack"):
        raise ValueError(f"{path.name}: final optical shutdown not proven")
    if commission.get("capture_error") or commission.get("motion_failures"):
        raise ValueError(f"{path.name}: capture or motion failure")

    frames = {int(frame["sequence"]): frame for frame in data["frames"]}
    receipts = {
        int(item["sequence"]): int(item["host_monotonic_ns"])
        for item in commission["frame_receipts"]
    }
    events = {
        item["kind"]: int(item["host_monotonic_ns"])
        for item in commission["events"]
    }
    trigger = int(commission["trigger_sequence"])
    # The realtime localizer freezes the first 64 post-warmup frames.  A
    # commissioning press starts a little later (normally sequence 96), so
    # using every pre-trigger frame here silently produced an 80-frame offline
    # baseline that runtime could never reproduce.
    baseline_sequences = sorted(
        sequence for sequence in receipts if 16 <= sequence < trigger
    )[:64]
    baseline_frames = [frames[sequence] for sequence in baseline_sequences]
    held_frames = [
        frames[sequence]
        for sequence, received in receipts.items()
        if events["contact_reached_commanded_depth"] + 200_000_000
        <= received
        < events["release_started"]
    ]
    if len(baseline_frames) != 64 or len(held_frames) < MIN_HELD_FRAMES:
        raise ValueError(f"{path.name}: insufficient baseline or held frames")

    def matrix(field: str, selected: list[dict]) -> np.ndarray:
        value = np.asarray(
            [[record[field] for record in frame["records"]] for frame in selected],
            dtype=float,
        )
        if value.shape[1:] != (45,) or not np.isfinite(value).all():
            raise ValueError(f"{path.name}: malformed {field} matrix")
        return value

    baseline_first = matrix("first_code", baseline_frames)
    baseline_second = matrix("second_code", baseline_frames)
    held_first = matrix("first_code", held_frames)
    held_second = matrix("second_code", held_frames)
    medians = {
        "first": (np.median(baseline_first, axis=0), np.median(held_first, axis=0)),
        "second": (np.median(baseline_second, axis=0), np.median(held_second, axis=0)),
        "mean": (
            np.median((baseline_first + baseline_second) / 2.0, axis=0),
            np.median((held_first + held_second) / 2.0, axis=0),
        ),
    }
    sigma = 1.4826 * np.median(
        np.abs(baseline_second - np.median(baseline_second, axis=0)), axis=0
    )
    features = {}
    families = {}
    for sample_name, (base, held) in medians.items():
        variants = _feature_variants(base, held, sigma)
        families[sample_name] = variants
        features.update({f"{sample_name}_{name}": value for name, value in variants.items()})
    for name in families["first"]:
        features[f"pair_{name}"] = np.concatenate(
            (families["first"][name], families["second"][name])
        )
    transition = (medians["second"][1] - medians["first"][1]) - (
        medians["second"][0] - medians["first"][0]
    )
    features["pair_transition_delta45"] = np.concatenate(
        (families["first"]["delta45"], families["second"]["delta45"], transition)
    )

    target = commission["target"]
    # Position acceptance is a physical machine-coordinate claim.  Use the
    # calibrated work-plane XY recorded in every guarded contact rather than a
    # CAD half-span approximation (which understated a 4.00 mm neighbour step
    # as about 3.65 mm).
    # New runs retain the original calibrated work plane even when the operator
    # re-zeroes Mach3. Do not compare live, re-zeroed XY against old prototypes.
    xy = np.asarray(target.get('calibration_work_xy_mm',
                               [target['work_x'],target['work_y']]), dtype=float)
    if not np.isfinite(xy).all():
        raise ValueError(f"{path.name}: invalid target coordinate")
    delta = families["second"]["delta45"]
    return {
        "path": str(path.resolve()),
        "point_index": int(target["point_index"]),
        "depth_mm": float(commission["depth_mm"]),
        "xy": xy,
        "features": features,
        "baseline_frames": len(baseline_frames),
        "held_frames": len(held_frames),
        "response_rms_codes": float(np.sqrt(np.mean(delta**2))),
        "baseline_sigma_rms_codes": float(np.sqrt(np.mean(sigma**2))),
        "maximum_board_interval_us": int(data["analysis"]["board_interval_max_us"]),
    }


def load_manifest(path: Path) -> dict[str, list[dict]]:
    raw = json.loads(path.read_text("utf-8"))
    if raw.get("schema") != "f45_position_rounds_v1":
        raise ValueError("unknown manifest schema")
    rounds = {}
    for round_name, mapping in raw["rounds"].items():
        records = [
            _capture_feature((path.parent / mapping[str(point)]).resolve())
            for point in range(1, 18)
        ]
        if [record["point_index"] for record in records] != list(range(1, 18)):
            raise ValueError(f"{round_name}: point indices are not exactly 1..17")
        rounds[round_name] = records
    depths = {record["depth_mm"] for records in rounds.values() for record in records}
    if len(depths) != 1:
        raise ValueError("all rounds must use one indentation depth")
    return rounds


def evaluate_direction(train: list[dict], test: list[dict], feature, scaling, metric):
    train_values = np.stack([record["features"][feature] for record in train])
    test_values = np.stack([record["features"][feature] for record in test])
    train_values, test_values = _transform(train_values, test_values, scaling)
    if metric == "cosine":
        train_values /= np.maximum(np.linalg.norm(train_values, axis=1, keepdims=True), 1e-12)
        test_values /= np.maximum(np.linalg.norm(test_values, axis=1, keepdims=True), 1e-12)
    predictions = []
    errors = []
    for vector, truth in zip(test_values, test, strict=True):
        if metric == "cosine":
            distances = 1.0 - train_values @ vector
        else:
            distances = np.linalg.norm(train_values - vector, axis=1)
        predicted = train[int(np.argmin(distances))]
        error = float(np.linalg.norm(predicted["xy"] - truth["xy"]))
        errors.append(error)
        predictions.append(
            {
                "truth_point": truth["point_index"],
                "predicted_point": predicted["point_index"],
                "error_mm": error,
            }
        )
    return predictions, errors


def metrics(errors, predictions):
    values = np.asarray(errors, dtype=float)
    return {
        "median_error_mm": float(np.median(values)),
        "p95_error_mm": float(np.percentile(values, 95)),
        "max_error_mm": float(np.max(values)),
        # Machine coordinates are quantised and JSON decimal round trips can
        # express a nominal 4.000 mm neighbour as 4.000000000000001.
        "within_4mm_percent": float(100.0 * np.mean(values <= 4.0 + 1e-6)),
        "exact_point_percent": float(
            100.0 * np.mean([p["truth_point"] == p["predicted_point"] for p in predictions])
        ),
    }


def select(rounds: dict[str, list[dict]]) -> dict:
    if set(rounds) != {"r1", "r2"}:
        raise ValueError("candidate selection requires exactly r1 and r2")
    first = rounds["r1"][0]
    candidates = []
    for feature in first["features"]:
        for scaling in ("none", "standard", "robust"):
            for metric in ("euclidean", "cosine"):
                predictions = []
                errors = []
                for train, test in ((rounds["r1"], rounds["r2"]), (rounds["r2"], rounds["r1"])):
                    direction_predictions, direction_errors = evaluate_direction(
                        train, test, feature, scaling, metric
                    )
                    predictions.extend(direction_predictions)
                    errors.extend(direction_errors)
                item = {"feature": feature, "scaling": scaling, "metric": metric}
                item.update(metrics(errors, predictions))
                item["predictions"] = predictions
                candidates.append(item)
    candidates.sort(
        key=lambda item: (
            item["p95_error_mm"],
            item["median_error_mm"],
            item["max_error_mm"],
            -item["exact_point_percent"],
        )
    )
    quality = {
        name: {
            "response_rms_codes_min_median_max": [
                float(function([r["response_rms_codes"] for r in records]))
                for function in (min, np.median, max)
            ],
            "baseline_sigma_rms_codes_max": float(
                max(r["baseline_sigma_rms_codes"] for r in records)
            ),
            "maximum_board_interval_us": int(
                max(r["maximum_board_interval_us"] for r in records)
            ),
        }
        for name, records in rounds.items()
    }
    return {
        "schema": "f45_position_candidate_selection_v2",
        "error_coordinate_frame": "mach3_work_xy_mm",
        "purpose": "two-repeat bidirectional algorithm selection",
        "deployment_eligible": False,
        "reason_not_deployable": "A third untouched round is required for locked validation.",
        "physical_15hz_response_verified": False,
        "physical_contact_area_verified": False,
        "round_quality": quality,
        "best_candidate": candidates[0],
        "all_candidates": candidates,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output must be new")
    result = select(load_manifest(args.manifest))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(result["best_candidate"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
