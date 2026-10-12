"""Build the compact, locked F45 realtime position model from qualified data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate_fast_fullmap_position import load_manifest, select
from validate_fast_fullmap_position_locked import validate_locked


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _baseline_and_contact_rms(path: Path) -> tuple[list[float], list[float]]:
    data = json.loads(path.read_text("utf-8"))
    commission = data["commissioning"]
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
    baseline = [
        frames[sequence]
        for sequence, _received in receipts.items()
        if 16 <= sequence < trigger
    ]
    held = [
        frames[sequence]
        for sequence, received in receipts.items()
        if events["contact_reached_commanded_depth"] + 200_000_000
        <= received
        < events["release_started"]
    ]

    def first_matrix(items: list[dict]) -> np.ndarray:
        return np.asarray(
            [[record["first_code"] for record in frame["records"]] for frame in items],
            dtype=float,
        )

    baseline_values = first_matrix(baseline)
    held_values = first_matrix(held)
    centre = np.median(baseline_values, axis=0)
    baseline_rms = np.sqrt(np.mean((baseline_values - centre) ** 2, axis=1))
    held_rms = np.sqrt(np.mean((held_values - centre) ** 2, axis=1))
    return baseline_rms.tolist(), held_rms.tolist()


def build(
    training_manifest: Path,
    selection_path: Path,
    validation_manifest: Path,
) -> dict:
    rounds = load_manifest(training_manifest)
    selection = json.loads(selection_path.read_text("utf-8"))
    recomputed = select(rounds)["best_candidate"]
    locked = {
        key: str(selection["best_candidate"][key])
        for key in ("feature", "scaling", "metric")
    }
    if locked != {key: str(recomputed[key]) for key in locked}:
        raise ValueError("candidate selection is not reproducible from r1/r2")
    validation = validate_locked(
        training_manifest, selection_path, validation_manifest
    )
    if not validation["static_position_gate"]["passed"]:
        raise ValueError("untouched r3 did not pass the locked static position gate")

    training = rounds["r1"] + rounds["r2"]
    feature = locked["feature"]
    values = np.stack([record["features"][feature] for record in training])
    if locked["scaling"] != "robust":
        raise ValueError("the deployed model requires the selected robust scaling")
    centre = np.median(values, axis=0)
    scale = 1.4826 * np.median(np.abs(values - centre), axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    transformed = (values - centre) / scale
    if locked["metric"] == "cosine":
        transformed /= np.maximum(
            np.linalg.norm(transformed, axis=1, keepdims=True), 1e-12
        )

    raw_manifest = json.loads(training_manifest.read_text("utf-8"))
    baseline_rms: list[float] = []
    contact_rms: list[float] = []
    for mapping in raw_manifest["rounds"].values():
        for filename in mapping.values():
            base, held = _baseline_and_contact_rms(
                (training_manifest.parent / filename).resolve()
            )
            baseline_rms.extend(base)
            contact_rms.extend(held)
    maximum_baseline = float(max(baseline_rms))
    minimum_contact = float(min(contact_rms))
    if not maximum_baseline < minimum_contact:
        raise ValueError("training baseline/contact RMS distributions overlap")
    activation_threshold = (maximum_baseline + minimum_contact) / 2.0

    prototypes = []
    for index, (record, vector) in enumerate(zip(training, transformed, strict=True)):
        prototypes.append(
            {
                "point_index": int(record["point_index"]),
                "training_round": "r1" if index < 17 else "r2",
                "work_xy_mm": [float(value) for value in record["xy"]],
                "vector": [float(value) for value in vector],
            }
        )
    return {
        "schema": "ch1_fast_fullmap_position_model_v1",
        "protocol": {
            "name": "F45",
            "route_rows": list(range(45)),
            "fresh_points_per_frame": 45,
            "points_per_grating": 5,
            "adc_sample": "first_code",
            "spacing_us": 50,
            "feedback_selectors": [0, 2],
        },
        "baseline": {
            "minimum_laser_warmup_frames": 16,
            "no_contact_frames": 64,
            "freeze_after_ready": True,
        },
        "activation": {
            "quantity": "RMS of 45 first-code deltas",
            "threshold_codes": activation_threshold,
            "training_maximum_no_contact_rms_codes": maximum_baseline,
            "training_minimum_0p30mm_contact_rms_codes": minimum_contact,
            "validated_indentation_depth_mm": 0.30,
            "dynamic_15hz_detection_verified": False,
        },
        "classifier": {
            "feature": feature,
            "scaling": locked["scaling"],
            "metric": locked["metric"],
            "rule": "one_nearest_training_observation",
            "centre": centre.tolist(),
            "scale": scale.tolist(),
            "prototypes": prototypes,
        },
        "static_validation": {
            "locked_before_r3": True,
            "error_coordinate_frame": "mach3_work_xy_mm",
            "metrics": validation["metrics"],
            "gate": validation["static_position_gate"],
            "validated_indentation_depth_mm": 0.30,
        },
        "physical_15hz_response_verified": False,
        "physical_contact_area_verified": False,
        "physical_contact_area_mm2": None,
        "sources": {
            "training_manifest": str(training_manifest.resolve()),
            "training_manifest_sha256": sha256(training_manifest),
            "selection": str(selection_path.resolve()),
            "selection_sha256": sha256(selection_path),
            "validation_manifest": str(validation_manifest.resolve()),
            "validation_manifest_sha256": sha256(validation_manifest),
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_manifest", type=Path)
    parser.add_argument("selection", type=Path)
    parser.add_argument("validation_manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output must be new")
    model = build(args.training_manifest, args.selection, args.validation_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({
        "schema": model["schema"],
        "feature": model["classifier"]["feature"],
        "metric": model["classifier"]["metric"],
        "activation": model["activation"],
        "static_validation": model["static_validation"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
