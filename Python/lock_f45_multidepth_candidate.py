"""Lock the selected F45 multi-depth candidate before validation capture."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", type=Path)
    parser.add_argument("depth_0p10_manifest", type=Path)
    parser.add_argument("depth_0p80_manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output must be new")
    selection = json.loads(args.selection.read_text("utf-8"))
    if selection.get("schema") != "f45_multidepth_position_candidate_selection_v1":
        raise ValueError("unexpected candidate-selection schema")
    best = selection["best"]
    expected = {
        "feature": "pair_transition_delta45",
        "preprocessing": "unit135",
        "estimator": "extra_trees_400",
    }
    if {key: best.get(key) for key in expected} != expected:
        raise ValueError("selected candidate is not the reviewed multi-depth algorithm")
    if int(best.get("worst_selection_exact", -1)) < 16:
        raise ValueError("candidate did not meet the pre-validation development gate")
    result = {
        "schema": "f45_multidepth_position_candidate_lock_v1",
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "feature": {
            "name": expected["feature"],
            "dimensions_before_preprocessing": 135,
            "preprocessing": expected["preprocessing"],
            "block_definition": ["first_delta45", "second_delta45", "pair_transition45"],
            "normalization": "single L2 unit norm over the complete 135-D vector with 1e-12 floor",
        },
        "baseline": {
            "minimum_laser_warmup_frames": 16,
            "no_contact_frames": 64,
            "freeze_after_ready": True,
            "separate_first_and_second_medians": True,
        },
        "classifier": {
            "name": "ExtraTreesClassifier",
            "n_estimators": 400,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "random_state": 20260908,
            "n_jobs_offline": -1,
            "n_jobs_runtime": 1,
        },
        "training_depths_mm": [0.10, 0.30, 0.80],
        "training_contacts": int(selection["contacts"]),
        "development_metrics": {
            "worst_0p30mm_whole_round_exact": int(best["worst_selection_exact"]),
            "worst_0p30mm_whole_round_within_4mm_percent": float(best["worst_selection_within_4mm_percent"]),
            "training_fit_exact": int(best["training_fit"]["exact"]),
        },
        "sources": [
            {"role": "selection", "path": str(args.selection.resolve()), "sha256": sha256(args.selection)},
            {"role": "depth_0p10", "path": str(args.depth_0p10_manifest.resolve()), "sha256": sha256(args.depth_0p10_manifest)},
            {"role": "depth_0p80", "path": str(args.depth_0p80_manifest.resolve()), "sha256": sha256(args.depth_0p80_manifest)},
        ],
        "validation_protocol": {
            "captured_after_lock": True,
            "depths_mm": [0.20, 0.60],
            "points_each_depth": list(range(1, 18)),
            "random_frame_split_forbidden": True,
            "primary_gate": {
                "trial_majority_exact": "17/17 at each depth",
                "frame_within_4mm_percent": 100.0,
                "detected_held_frame_percent": 100.0,
            },
        },
        "deployment_eligible": False,
        "physical_15hz_verified": False,
        "force_ground_truth_present": False,
        "contact_area_convention": "fixed 8 mm circular tool-face footprint = 50.2654824574 mm^2",
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
