"""Fixed cosine nearest-prototype test across two independent contact captures.

The development session supplies one median delta9 prototype per position.
The test session is scored both per contact and per individual fresh frame.
No feature selection, smoothing, or nearest-point truth is used at inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_fresh_sentinel_position import audit


def unit(values):
    values = np.asarray(values, dtype=float)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def metrics(errors, exact):
    return {"samples": len(errors), "median_error_mm": float(np.median(errors)),
            "p95_error_mm": float(np.percentile(errors, 95)),
            "max_error_mm": float(np.max(errors)),
            "exact_grid_point_percent": 100.0 * float(np.mean(exact))}


def compare(train, test):
    if train["source"] == test["source"] or train["source_sha256"]["frames.jsonl"] == test["source_sha256"]["frames.jsonl"]:
        raise ValueError("Independent captures required")
    for key in ("protocol", "table_crc32", "wavelengths_nm", "transimpedance_ohm", "depth_mm"):
        if train[key] != test[key]:
            raise ValueError(f"Incompatible capture {key}")
    if set(train["adc_value_kind_counts"]) != set(test["adc_value_kind_counts"]):
        raise ValueError("Incompatible ADC processing")
    prototypes = unit([r["delta9"] for r in train["records"]])
    xy = np.asarray([r["model_xy_mm"] for r in train["records"]])
    indices = np.asarray([r["point_index"] for r in train["records"]])
    contact_errors, contact_exact, frame_errors, frame_exact, rows = [], [], [], [], []
    for record in test["records"]:
        truth_xy = np.asarray(record["model_xy_mm"])
        def score(vectors):
            nearest = np.argmax(unit(vectors) @ prototypes.T, axis=-1)
            errors = np.linalg.norm(xy[nearest] - truth_xy, axis=-1)
            return nearest, errors, errors < 1e-6
        nearest, error, exact = score([record["delta9"]])
        _, errors, exact_frames = score(record["hold_delta9"])
        contact_errors.extend(error.tolist())
        contact_exact.extend(exact.tolist())
        frame_errors.extend(errors.tolist())
        frame_exact.extend(exact_frames.tolist())
        rows.append({"test_route_index": record["point_index"], "predicted_train_route_index": int(indices[nearest[0]]),
                     "truth_model_xy_mm": record["model_xy_mm"], "predicted_model_xy_mm": xy[nearest[0]].tolist(),
                     "contact_error_mm": float(error[0]), "single_frame": metrics(errors, exact_frames)})
    return {"algorithm": "fresh_sentinel9_cosine_nearest_median_prototype",
            "train_source": train["source"], "test_source": test["source"],
            "train_sha256": train["source_sha256"], "test_sha256": test["source_sha256"],
            "contact_median": metrics(contact_errors, contact_exact),
            "single_fresh_frame": metrics(frame_errors, frame_exact), "positions": rows,
            "baseline_policy": "first_two_seconds_of_each_pre_contact_phase",
            "deployment_eligible": False, "physical_15hz_verified": False,
            "contact_area_evaluated": False,
            "limitations": ["Only known grid positions and captured depth are tested",
                            "Per-contact no-load baseline supplied from host phase labels",
                            "Hold frames within one contact are correlated",
                            "Physical 15 Hz response and area need independent reference measurements"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", type=Path)
    parser.add_argument("test", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare(audit(args.train), audit(args.test))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "positions"}, ensure_ascii=False, indent=2))
