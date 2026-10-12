"""Lock the second reviewed causal-frame model pair before fresh validation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from evaluate_f45_cross_session_models import load_rounds
from select_f45_multidepth_position_candidate import load_depth_manifest


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--position-selection", required=True, type=Path)
    parser.add_argument("--depth-selection", required=True, type=Path)
    parser.add_argument("--failed-validation", required=True, type=Path)
    parser.add_argument("--depth-0p10", required=True, type=Path)
    parser.add_argument("--depth-0p20", required=True, type=Path)
    parser.add_argument("--depth-0p40", required=True, type=Path)
    parser.add_argument("--depth-0p60", required=True, type=Path)
    parser.add_argument("--depth-0p70", required=True, type=Path)
    parser.add_argument("--depth-0p80", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output must be new")

    position = json.loads(args.position_selection.read_text("utf-8"))
    depth = json.loads(args.depth_selection.read_text("utf-8"))
    failed = json.loads(args.failed_validation.read_text("utf-8"))
    if position.get("schema") != "f45_multidepth_causal_frame_position_selection_v2":
        raise ValueError("unexpected position selection")
    if depth.get("schema") != "f45_multidepth_causal_frame_depth_selection_v1":
        raise ValueError("unexpected depth selection")
    if failed.get("schema") != "f45_multidepth_causal_frame_models_validation_v1":
        raise ValueError("unexpected first locked validation report")
    if failed.get("deployment_gate_passed"):
        raise ValueError("first locked candidate was expected to be a failed development input")

    expected_position = {
        "feature": "pair_transition_delta45",
        "preprocessing": "pair_unit90",
        "estimator": "lda_shrinkage",
        "kind": "classifier",
    }
    if {key: position["best"].get(key) for key in expected_position} != expected_position:
        raise ValueError("reviewed position candidate is not the selection winner")
    if int(position["best"]["worst_trial_majority_exact"]) < 14:
        raise ValueError("position development exact-label gate failed")
    if float(position["best"]["worst_frame_within_4mm_percent"]) < 100.0 - 1e-12:
        raise ValueError("position development tool-radius gate failed")

    expected_depth = {
        "feature": "pair_transition_delta45",
        "preprocessing": "shape_amplitude_point",
        "estimator": "ridge_a100",
    }
    if {key: depth["best"].get(key) for key in expected_depth} != expected_depth:
        raise ValueError("reviewed depth candidate is not the selection winner")
    if float(depth["best"]["interpolation_trial_mae_mm_max"]) > 0.10 + 1e-12:
        raise ValueError("depth development MAE gate failed")
    if float(depth["best"]["interpolation_trial_p95_mm_max"]) > 0.20 + 1e-12:
        raise ValueError("depth development p95 gate failed")

    here = Path(__file__).resolve().parent
    old_manifest = (here.parent / "outputs" / "fast_fullmap_position_rounds_0p30_20260906.json").resolve()
    old_validation = [
        (here.parent / "outputs" / "fast_fullmap_position_validation_r3_0p30_20260906.json").resolve(),
        here / "outputs" / "f45_position_validation_after_por_r1_20260908.json",
        here / "outputs" / "f45_position_validation_after_lock_r2_20260908.json",
    ]
    old = load_rounds(old_manifest, old_validation)
    manifests = {
        "depth_0p10_r1": args.depth_0p10,
        "depth_0p20_r1": args.depth_0p20,
        "depth_0p40_r1": args.depth_0p40,
        "depth_0p60_r1": args.depth_0p60,
        "depth_0p70_r1": args.depth_0p70,
        "depth_0p80_r1": args.depth_0p80,
    }
    rounds = {
        "depth_0p10_r1": load_depth_manifest(args.depth_0p10),
        "depth_0p20_r1": load_depth_manifest(args.depth_0p20),
        **old,
        "depth_0p40_r1": load_depth_manifest(args.depth_0p40),
        "depth_0p60_r1": load_depth_manifest(args.depth_0p60),
        "depth_0p70_r1": load_depth_manifest(args.depth_0p70),
        "depth_0p80_r1": load_depth_manifest(args.depth_0p80),
    }
    captures = []
    for round_name, records in rounds.items():
        for row in records:
            path = Path(row["path"])
            captures.append({
                "round": round_name,
                "point": int(row["point_index"]),
                "depth_mm": float(row["depth_mm"]),
                "path": str(path.resolve()),
                "sha256": sha256(path),
            })
    canonical = json.dumps(
        captures, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    sources = [
        {"role": "position_selection", "path": str(args.position_selection.resolve()), "sha256": sha256(args.position_selection)},
        {"role": "depth_selection", "path": str(args.depth_selection.resolve()), "sha256": sha256(args.depth_selection)},
        {"role": "first_locked_validation_reclassified_as_development", "path": str(args.failed_validation.resolve()), "sha256": sha256(args.failed_validation)},
        {"role": "old_0p30_training_manifest", "path": str(old_manifest), "sha256": sha256(old_manifest)},
        *[
            {"role": f"old_0p30_validation_{index}", "path": str(path.resolve()), "sha256": sha256(path)}
            for index, path in enumerate(old_validation, start=1)
        ],
        *[
            {"role": name, "path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in manifests.items()
        ],
    ]
    result = {
        "schema": "f45_multidepth_causal_frame_models_lock_v2",
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": "F45",
            "route_rows": list(range(45)),
            "spacing_us": 50,
            "minimum_warmup_frames": 16,
            "frozen_baseline_frames": 64,
            "feedback_selectors": [0, 2],
        },
        "position": {
            **expected_position,
            "classifier": {
                "class": "sklearn.pipeline(StandardScaler,LinearDiscriminantAnalysis)",
                "lda_solver": "lsqr",
                "lda_shrinkage": "auto",
            },
            "development_worst_trial_majority_exact": int(position["best"]["worst_trial_majority_exact"]),
            "development_worst_frame_exact_percent": float(position["best"]["worst_frame_exact_percent"]),
            "development_worst_frame_within_4mm_percent": float(position["best"]["worst_frame_within_4mm_percent"]),
        },
        "depth_proxy": {
            **expected_depth,
            "regressor": {
                "class": "sklearn.pipeline(StandardScaler,Ridge)",
                "ridge_alpha": 100.0,
                "output_clip_mm": [0.10, 0.80],
                "point_feature_at_runtime": "position classifier prediction as 17-D one-hot",
            },
            "development_interpolation_mae_mm_max": float(depth["best"]["interpolation_trial_mae_mm_max"]),
            "development_interpolation_p95_mm_max": float(depth["best"]["interpolation_trial_p95_mm_max"]),
            "target": "commanded CNC indentation depth, not force",
        },
        "training": {
            "depths_mm": [0.10, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80],
            "rounds": list(rounds),
            "contacts": len(captures),
            "capture_manifest_sha256": hashlib.sha256(canonical).hexdigest(),
            "captures": captures,
        },
        "sources": sources,
        "validation_protocol": {
            "captured_after_lock": True,
            "depths_mm": [0.25, 0.55],
            "points_each_depth": list(range(1, 18)),
            "random_frame_split_forbidden": True,
            "position_gate": {
                "physical_basis": "4 mm tolerance equals the radius of the fixed 8 mm circular press face",
                "trial_majority_exact_min_0p25": 14,
                "trial_majority_exact_min_0p55": 17,
                "trial_majority_exact_min_total": 31,
                "frame_within_4mm_percent_min_each_depth": 100.0,
                "detected_held_frame_percent_min_each_depth": 100.0,
            },
            "depth_proxy_gate": {
                "trial_median_prediction_mae_mm_max_each_depth": 0.10,
                "trial_median_prediction_p95_absolute_error_mm_max_each_depth": 0.20,
                "per_point_depth_order_0p55_greater_than_0p25": "17/17",
            },
        },
        "deployment_eligible": False,
        "force_ground_truth_present": False,
        "physical_15hz_verified": False,
        "contact_area_convention": "fixed 8 mm circular tool-face footprint = 50.2654824574 mm^2",
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({
        "schema": result["schema"],
        "locked_utc": result["locked_utc"],
        "position": result["position"],
        "depth_proxy": result["depth_proxy"],
        "training_contacts": len(captures),
        "capture_manifest_sha256": result["training"]["capture_manifest_sha256"],
        "validation_protocol": result["validation_protocol"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
