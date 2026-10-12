"""Freeze the selected F18 multi-depth position candidate before validation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from capture_fast_flank_stream import ROWS
from evaluate_f18_cross_session_models import ROUND_PATTERNS
from evaluate_f18_multidepth_sparse_holdout import PROBE_POINTS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(path: Path) -> dict:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--complete-round", type=Path, required=True)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    selection = json.loads(args.selection.read_text("utf-8"))
    best = selection.get("best") or {}
    expected = {
        "training": "complete_0p30_plus_four_0p10_rounds",
        "feature": "pair_flank_shape36",
        "estimator": "lda_shrinkage",
    }
    if any(best.get(key) != value for key, value in expected.items()):
        raise ValueError("selected F18 candidate changed")
    if (
        float(best.get("worst_fold_trial_majority_exact_percent", -1.0)) != 100.0
        or float(best.get("worst_fold_frame_exact_percent", -1.0)) < 99.0
        or float(best.get("worst_fold_frame_within_4mm_percent", -1.0)) != 100.0
    ):
        raise ValueError("sparse independent selection gate did not pass")
    complete = json.loads(args.complete_round.read_text("utf-8"))
    if complete.get("schema") != "f18_guarded_position_round_manifest_v1":
        raise ValueError("unsupported complete-round manifest")
    if len(complete.get("captures") or ()) != 17:
        raise ValueError("complete training round is not P1..P17")

    shallow = []
    for round_name, pattern in ROUND_PATTERNS.items():
        for point in range(1, 18):
            shallow.append(
                {
                    "round": round_name,
                    "point_index": point,
                    **source(args.capture_directory / pattern.format(point=point)),
                }
            )
    sparse_holdout = []
    for repeat in (1, 2):
        for point in PROBE_POINTS:
            sparse_holdout.append(
                {
                    "round": f"historical_probe_r{repeat}",
                    "point_index": point,
                    **source(
                        args.capture_directory
                        / f"ch1_fast_flank_contact_p{point:02d}_0p30mm_probe_r{repeat}_20260906.json"
                    ),
                }
            )

    lock = {
        "schema": "f18_multidepth_position_candidate_lock_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "deployment_status": "candidate_only",
        "protocol": {
            "name": "F18",
            "route_rows": list(ROWS),
            "fresh_points_per_frame": 18,
            "adc_samples": ["first_code", "second_code"],
            "feedback_selectors": [0, 2],
            "minimum_observed_rate_hz": float(complete["minimum_rate_hz"]),
        },
        "baseline": {
            "minimum_laser_warmup_frames": 16,
            "no_contact_frames": 64,
            "separate_first_and_second_medians": True,
            "freeze_after_ready": True,
        },
        "feature": {
            "name": "pair_flank_shape36",
            "dimensions": 36,
            "definition": (
                "For each of nine left/right flank pairs and for both physical ADC reads: "
                "baseline-relative normalized differential followed by baseline-relative pair sum."
            ),
        },
        "classifier": {
            "name": "Pipeline(StandardScaler, LinearDiscriminantAnalysis)",
            "standard_scaler": {"with_mean": True, "with_std": True},
            "lda": {"solver": "lsqr", "shrinkage": "auto"},
        },
        "training": {
            "contacts": 85,
            "depths_mm": [0.10, 0.30],
            "complete_0p30_round_manifest": source(args.complete_round),
            "complete_0p30_capture_sources": complete["captures"],
            "complete_0p10_capture_sources": shallow,
        },
        "selection_holdout": {
            "selection_report": source(args.selection),
            "sources": sparse_holdout,
            "trial_majority_exact_percent": 100.0,
            "worst_frame_exact_percent": float(best["worst_fold_frame_exact_percent"]),
            "worst_frame_within_4mm_percent": 100.0,
        },
        "untouched_full_round_gate": {
            "positions": 17,
            "depth_mm": 0.30,
            "trial_majority_exact_required": 17,
            "minimum_frame_exact_percent": 95.0,
            "minimum_frame_within_4mm_percent": 100.0,
            "maximum_p95_error_mm": 4.0,
            "minimum_sampling_rate_hz": 45.0,
        },
        "nominal_tool_diameter_mm": 8.0,
        "nominal_tool_area_mm2": 50.26548245743669,
        "area_convention": "fixed circular tool-face footprint, not deformed contact-area truth",
        "force_proxy": "commanded indentation depth; no N/Pa calibration in this version",
        "maximum_allowed_indentation_mm": 0.8,
        "physical_15hz_verified": False,
    }
    args.output.write_text(json.dumps(lock, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({"path": str(args.output.resolve()), "sha256": sha256(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
