"""Freeze the reviewed F45 position/8 mm footprint candidate.

This command is deliberately offline.  It records every development capture
and its digest before any new physical validation is acquired.  The lock is
not a deployment approval; a post-lock untouched validation must pass the
precommitted gates below.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from evaluate_f45_cross_session_models import load_rounds
from select_f45_continuous_position_v3 import _load_visible_round, _parse_round


FEATURE = "pair_transition_delta45"
TOOL_RADIUS_MM = 4.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument(
        "--candidate-v4",
        action="store_true",
        help="lock the reviewed block-unit/LDA candidate with new 0.15/0.75 mm validation",
    )
    parser.add_argument(
        "--candidate-v5",
        action="store_true",
        help="lock the reviewed unit/LDA causal-EMA position candidate",
    )
    parser.add_argument(
        "--candidate-v6",
        action="store_true",
        help="lock the pair-unit/LDA causal-mean position and learned contact detector",
    )
    parser.add_argument(
        "--candidate-v7",
        action="store_true",
        help="lock the pair-unit/LDA posterior-temperature/EMA position candidate",
    )
    parser.add_argument(
        "--activation-selection",
        type=Path,
        help="required whole-round contact-detector selection report for v6/v7",
    )
    parser.add_argument(
        "--round",
        action="append",
        default=[],
        required=True,
        type=_parse_round,
        metavar="NAME=MANIFEST",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output must be new")

    if sum((args.candidate_v4, args.candidate_v5, args.candidate_v6, args.candidate_v7)) > 1:
        parser.error("choose only one candidate generation")
    if (args.candidate_v6 or args.candidate_v7) and args.activation_selection is None:
        parser.error("--candidate-v6/--candidate-v7 requires --activation-selection")
    selection = json.loads(args.selection.read_text("utf-8"))
    expected_selection_schema = (
        "f45_spatial_temporal_decoder_selection_v5"
        if (args.candidate_v5 or args.candidate_v6 or args.candidate_v7)
        else "f45_continuous_position_selection_v3"
    )
    if selection.get("schema") != expected_selection_schema:
        raise ValueError("unexpected selection report")
    expected = (
        {
            "representation": "pair_unit90",
            "estimator": "lda_shrinkage",
            "decoder": "posterior_temperature_3.0_ema_0.25",
        }
        if args.candidate_v7
        else
        {
            "representation": "pair_unit90",
            "estimator": "lda_shrinkage",
            "decoder": "hard_causal_mean_5",
        }
        if args.candidate_v6
        else
        {
            "representation": "unit135",
            "estimator": "lda_shrinkage",
            "decoder": "hard_causal_ema_0.25",
        }
        if args.candidate_v5
        else
        {
            "representation": "block_unit135",
            "estimator": "lda_shrinkage",
            "kind": "classifier",
            "decoder": "hard",
        }
        if args.candidate_v4
        else {
            "representation": "unit135",
            "estimator": "extra_trees_leaf1_sqrt",
            "kind": "classifier",
            "decoder": "hard",
        }
    )
    best = selection.get("best") or {}
    if {key: best.get(key) for key in expected} != expected:
        raise ValueError("reviewed candidate is no longer the selection winner")
    if float(best["worst_trial_within_tool_radius_percent"]) < 100.0 - 1e-12:
        raise ValueError("development trial footprint gate failed")
    development_frame_gate = 99.0 if (args.candidate_v6 or args.candidate_v7) else (98.5 if (args.candidate_v4 or args.candidate_v5) else 99.0)
    if float(best["worst_frame_within_tool_radius_percent"]) < development_frame_gate:
        raise ValueError("development frame footprint gate failed")
    if args.candidate_v5:
        if not bool(best.get("development_footprint_gate_passed")):
            raise ValueError("v5 development footprint gate was not recorded as passed")
        if int(best.get("worst_full_round_trial_nearest_label_exact", -1)) < 14:
            raise ValueError("v5 development exact-position gate failed")
    if args.candidate_v6 or args.candidate_v7:
        if not bool(best.get("development_footprint_gate_passed")):
            raise ValueError("v6/v7 development footprint gate was not recorded as passed")
        exact_floor = 13 if args.candidate_v7 else 12
        if int(best.get("worst_full_round_trial_nearest_label_exact", -1)) < exact_floor:
            raise ValueError("v6/v7 development exact-position floor failed")
        activation_selection = json.loads(args.activation_selection.read_text("utf-8"))
        if activation_selection.get("schema") != "f45_activation_detector_selection_v6":
            raise ValueError("unexpected v6 activation selection report")
        reviewed_activation = next(
            (
                row for row in activation_selection.get("ranked", ())
                if row.get("feature") == FEATURE
                and row.get("representation") == "raw135"
                and row.get("estimator") == "extra_trees_300_leaf10"
            ),
            None,
        )
        if reviewed_activation is None:
            raise ValueError("reviewed v6 activation candidate is absent")
        if (
            float(reviewed_activation["worst_capture_held_detected_percent"]) < 100.0
            or int(reviewed_activation["total_baseline_false_positive_frames"]) != 0
            or int(reviewed_activation["total_held_missed_frames"]) != 0
        ):
            raise ValueError("reviewed v6 activation candidate failed development gates")
    else:
        reviewed_activation = None

    here = Path(__file__).resolve().parent
    old_manifest = (here.parent / "outputs" / "fast_fullmap_position_rounds_0p30_20260906.json").resolve()
    old_validations = [
        (here.parent / "outputs" / "fast_fullmap_position_validation_r3_0p30_20260906.json").resolve(),
        here / "outputs" / "f45_position_validation_after_por_r1_20260908.json",
        here / "outputs" / "f45_position_validation_after_lock_r2_20260908.json",
    ]
    rounds = load_rounds(old_manifest, old_validations)
    sources = [
        {
            "role": "selection",
            "path": str(args.selection.resolve()),
            "sha256": sha256(args.selection),
        },
        *(
            [
                {
                    "role": "activation_selection",
                    "path": str(args.activation_selection.resolve()),
                    "sha256": sha256(args.activation_selection),
                }
            ]
            if (args.candidate_v6 or args.candidate_v7)
            else []
        ),
        {
            "role": "old_0p30_training_manifest",
            "path": str(old_manifest),
            "sha256": sha256(old_manifest),
        },
        *[
            {
                "role": f"old_0p30_validation_{index}",
                "path": str(path.resolve()),
                "sha256": sha256(path),
            }
            for index, path in enumerate(old_validations, start=1)
        ],
    ]
    seen = set(rounds)
    for name, path in args.round:
        if name in seen:
            parser.error(f"duplicate round name: {name}")
        seen.add(name)
        records = _load_visible_round(path)
        selected_round = (selection.get("rounds") or {}).get(name)
        actual_points = [int(row["point_index"]) for row in records]
        actual_depth = float(records[0]["depth_mm"])
        if selected_round is None:
            raise ValueError(f"round {name!r} was not used by the selection")
        if (
            actual_points != [int(value) for value in selected_round["points"]]
            or abs(actual_depth - float(selected_round["depth_mm"])) > 1e-9
        ):
            raise ValueError(f"round {name!r} differs from selection input")
        rounds[name] = records
        sources.append(
            {
                "role": name,
                "path": str(path.resolve()),
                "sha256": sha256(path),
            }
        )
    if set(rounds) != set(selection["rounds"]):
        missing = sorted(set(selection["rounds"]) - set(rounds))
        extra = sorted(set(rounds) - set(selection["rounds"]))
        raise ValueError(f"selection/lock round mismatch; missing={missing}, extra={extra}")

    captures = []
    coordinates = {}
    for round_name, records in rounds.items():
        for row in records:
            path = Path(row["path"]).resolve()
            point = int(row["point_index"])
            coordinates.setdefault(point, [float(value) for value in row["xy"]])
            captures.append(
                {
                    "round": round_name,
                    "point": point,
                    "depth_mm": float(row["depth_mm"]),
                    "path": str(path),
                    "sha256": sha256(path),
                }
            )
    if set(coordinates) != set(range(1, 18)):
        raise ValueError("calibrated 17-position coordinate map is incomplete")
    canonical = json.dumps(
        captures, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    estimator = (
        {
            "class": "sklearn.discriminant_analysis.LinearDiscriminantAnalysis",
            "pipeline": ["StandardScaler", "LinearDiscriminantAnalysis"],
            "solver": "lsqr",
            "shrinkage": "auto",
            "decoder": "posterior_probability_xy_causal_ema",
            "position_probability_temperature": 3.0,
            "position_ema_alpha": 0.25,
            "reset_position_state_on_no_contact": True,
            "pressure_path": "unsmoothed_full_rate_feature_energy",
        }
        if args.candidate_v7
        else
        {
            "class": "sklearn.discriminant_analysis.LinearDiscriminantAnalysis",
            "pipeline": ["StandardScaler", "LinearDiscriminantAnalysis"],
            "solver": "lsqr",
            "shrinkage": "auto",
            "decoder": "hard_class_xy_causal_mean",
            "position_mean_window_frames": 5,
            "reset_position_state_on_no_contact": True,
            "pressure_path": "unsmoothed_full_rate_feature_energy",
        }
        if args.candidate_v6
        else
        {
            "class": "sklearn.discriminant_analysis.LinearDiscriminantAnalysis",
            "pipeline": ["StandardScaler", "LinearDiscriminantAnalysis"],
            "solver": "lsqr",
            "shrinkage": "auto",
            "decoder": "hard_class_xy_causal_ema",
            "position_ema_alpha": 0.25,
            "reset_position_state_on_no_contact": True,
            "pressure_path": "unsmoothed_full_rate_feature_energy",
        }
        if args.candidate_v5
        else
        {
            "class": "sklearn.discriminant_analysis.LinearDiscriminantAnalysis",
            "pipeline": ["StandardScaler", "LinearDiscriminantAnalysis"],
            "solver": "lsqr",
            "shrinkage": "auto",
            "decoder": "hard_class_to_calibrated_xy",
        }
        if args.candidate_v4
        else {
            "class": "sklearn.ensemble.ExtraTreesClassifier",
            "n_estimators": 300,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "random_state": 20260908,
            "n_jobs_fit": -1,
            "decoder": "hard_class_to_calibrated_xy",
        }
    )
    validation_depths = (
        [0.12, 0.72]
        if args.candidate_v7
        else [0.15, 0.75]
        if (args.candidate_v4 or args.candidate_v5 or args.candidate_v6)
        else [0.35, 0.65]
    )
    result = {
        "schema": (
            "f45_position_footprint_lock_v7"
            if args.candidate_v7
            else "f45_position_footprint_lock_v6"
            if args.candidate_v6
            else "f45_position_footprint_lock_v5"
            if args.candidate_v5
            else ("f45_position_footprint_lock_v4" if args.candidate_v4 else "f45_position_footprint_lock_v3")
        ),
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
            "feature": FEATURE,
            "preprocessing": expected["representation"],
            "estimator": estimator,
            "coordinates_by_point": {str(key): value for key, value in sorted(coordinates.items())},
            "development": {
                "worst_trial_within_4mm_percent": float(best["worst_trial_within_tool_radius_percent"]),
                "worst_frame_within_4mm_percent": float(best["worst_frame_within_tool_radius_percent"]),
                "worst_trial_nearest_label_exact_percent": float(best["worst_trial_nearest_label_exact_percent"]),
                "mean_frame_error_mm": float(best["mean_frame_error_mm"]),
            },
        },
        **(
            {
                "activation": {
                    "feature": FEATURE,
                    "preprocessing": "raw135",
                    "estimator": {
                        "class": "sklearn.ensemble.ExtraTreesClassifier",
                        "n_estimators": 300,
                        "min_samples_leaf": 10,
                        "max_features": "sqrt",
                        "class_weight": "balanced",
                        "random_state": 20260908,
                        "n_jobs_fit": -1,
                    },
                    "temporal_smoothing_used": False,
                    "development": {
                        "worst_capture_held_detected_percent": float(
                            reviewed_activation["worst_capture_held_detected_percent"]
                        ),
                        "total_baseline_false_positive_frames": int(
                            reviewed_activation["total_baseline_false_positive_frames"]
                        ),
                        "total_baseline_frames": int(reviewed_activation["total_baseline_frames"]),
                        "total_held_missed_frames": int(
                            reviewed_activation["total_held_missed_frames"]
                        ),
                        "total_held_frames": int(reviewed_activation["total_held_frames"]),
                    },
                }
            }
            if (args.candidate_v6 or args.candidate_v7)
            else {}
        ),
        "training": {
            "rounds": list(rounds),
            "contacts": len(captures),
            "capture_manifest_sha256": hashlib.sha256(canonical).hexdigest(),
            "captures": captures,
        },
        "sources": sources,
        "validation_protocol": {
            "captured_after_lock": True,
            "untouched_depths_mm": validation_depths,
            "points_each_depth": list(range(1, 18)),
            "random_frame_split_forbidden": True,
            "gates": {
                "trial_nearest_label_exact_min_each_depth": 14,
                "trial_nearest_label_exact_min_total": 30,
                "trial_within_4mm": "17/17 each depth",
                "frame_within_4mm_percent_min_each_depth": 99.0 if (args.candidate_v6 or args.candidate_v7) else (98.5 if (args.candidate_v4 or args.candidate_v5) else 99.0),
                "detected_held_frame_percent_min_each_depth": 100.0,
                **(
                    {"baseline_false_positive_percent_max_each_depth": 0.0}
                    if (args.candidate_v6 or args.candidate_v7)
                    else {}
                ),
                "board_sampling_rate_gate_for_15hz_required": True,
            },
        },
        "deployment_eligible": False,
        "physical_15hz_verified": False,
        "force_ground_truth_present": False,
        "pressure_proxy": "commanded CNC indentation depth only",
        "contact_area": {
            "basis": "fixed 8 mm circular tool-face footprint; not deformed contact-area truth",
            "diameter_mm": 8.0,
            "nominal_area_mm2": 50.26548245743669,
            "tool_radius_position_tolerance_mm": TOOL_RADIUS_MM,
        },
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "locked_utc": result["locked_utc"],
                "training_rounds": len(rounds),
                "training_contacts": len(captures),
                "capture_manifest_sha256": result["training"]["capture_manifest_sha256"],
                "validation_protocol": result["validation_protocol"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
