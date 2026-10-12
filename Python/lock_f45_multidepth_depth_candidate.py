"""Lock the training-only F45 indentation proxy before validation capture."""

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
    parser.add_argument("position_lock", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output must be new")
    selection = json.loads(args.selection.read_text("utf-8"))
    position_lock = json.loads(args.position_lock.read_text("utf-8"))
    if selection.get("schema") != "f45_monotone_depth_proxy_candidate_selection_v1":
        raise ValueError("unexpected depth candidate-selection schema")
    if position_lock.get("schema") != "f45_multidepth_position_candidate_lock_v1":
        raise ValueError("unexpected position lock schema")
    best = selection["best"]
    if best.get("response") != "second_rms":
        raise ValueError("reviewed depth response was not selected")
    if float(best.get("worst_loro_p95_mm", 1.0)) > 0.20 + 1e-12:
        raise ValueError("depth proxy failed the development gate")
    result = {
        "schema": "f45_multidepth_depth_proxy_lock_v1",
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "target": "commanded_cnc_indentation_depth_mm_not_force",
        "response": {
            "name": "second_rms",
            "definition": "RMS of the 45 second-code deltas from the frozen 64-frame no-contact median baseline",
            "contact_summary": "median response across held frames beginning 0.200 s after commanded depth is reached",
        },
        "calibration": {
            "scope": "one monotone curve per predicted coverage point P1..P17",
            "training_depths_mm": [0.10, 0.30, 0.80],
            "knot_response": "median across complete acquisition rounds at each depth",
            "monotonic_fit": "sklearn IsotonicRegression increasing=True out_of_bounds=clip weighted by replicate count",
            "inverse": "collapse equal response plateaus to mean depth then linear np.interp clipped to 0.10..0.80 mm",
            "curves": best["curves"],
        },
        "development_metrics": {
            "selection_protocol": selection["selection_protocol"],
            "mean_loro_mae_mm": float(best["mean_loro_mae_mm"]),
            "worst_loro_mae_mm": float(best["worst_loro_mae_mm"]),
            "worst_loro_p95_mm": float(best["worst_loro_p95_mm"]),
            "strict_raw_monotone_points": int(best["strict_raw_monotone_points"]),
        },
        "sources": [
            {"role": "depth_selection", "path": str(args.selection.resolve()), "sha256": sha256(args.selection)},
            {"role": "position_lock", "path": str(args.position_lock.resolve()), "sha256": sha256(args.position_lock)},
        ],
        "validation_protocol": {
            "captured_after_lock": True,
            "depths_mm": [0.20, 0.60],
            "points_each_depth": list(range(1, 18)),
            "random_frame_split_forbidden": True,
            "primary_gate": {
                "trial_depth_mae_mm_each_depth_max": 0.10,
                "trial_depth_p95_absolute_error_mm_each_depth_max": 0.20,
                "per_point_depth_order_0p60_greater_than_0p20": "17/17",
            },
        },
        "deployment_eligible": False,
        "force_ground_truth_present": False,
        "physical_15hz_verified": False,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
