"""Select an F45 position candidate using real 0.10/0.30/0.80 mm rounds.

All evaluation folds exclude complete physical acquisition rounds.  The five
0.30 mm folds are the model-selection score because both endpoint-depth rounds
remain in training; endpoint leave-one-depth-out results are reported only as
an extrapolation diagnostic.  A later 0.20/0.60 mm round must remain untouched
for the final depth-generalisation decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_f45_cross_session_models import load_rounds
from evaluate_fast_fullmap_position import _capture_feature


FEATURE = "pair_transition_delta45"


def load_depth_manifest(path: Path) -> list[dict]:
    payload = json.loads(path.read_text("utf-8"))
    if (
        payload.get("schema") != "f45_visible_depth_round_v1"
        or not payload.get("complete")
        or set(int(value) for value in payload.get("completed_points", ())) != set(range(1, 18))
    ):
        raise ValueError(f"{path}: not a complete F45 17-point depth round")
    records = [
        _capture_feature(Path(payload["captures"][str(point)]["capture"]))
        for point in range(1, 18)
    ]
    if any(abs(float(record["depth_mm"]) - float(payload["depth_mm"])) > 1e-9 for record in records):
        raise ValueError(f"{path}: capture depth mismatch")
    return records


def unit_rows(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def transform(values: np.ndarray, variant: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 135:
        raise ValueError("expected 135-D pair-transition feature")
    if variant == "raw135":
        return values
    if variant == "unit135":
        return unit_rows(values)
    if variant == "signed_sqrt_unit135":
        return unit_rows(np.sign(values) * np.sqrt(np.abs(values)))
    blocks = values.reshape(-1, 3, 45)
    if variant == "block_unit135":
        return (
            blocks
            / np.maximum(np.linalg.norm(blocks, axis=2, keepdims=True), 1e-12)
        ).reshape(-1, 135)
    if variant == "first_unit45":
        return unit_rows(blocks[:, 0, :])
    if variant == "pair_unit90":
        return unit_rows(blocks[:, :2, :].reshape(-1, 90))
    if variant == "segment_energy9":
        first = blocks[:, 0, :].reshape(-1, 9, 5)
        return unit_rows(np.sqrt(np.mean(first**2, axis=2)))
    raise ValueError(variant)


def candidates():
    for variant in (
        "raw135",
        "unit135",
        "signed_sqrt_unit135",
        "block_unit135",
        "first_unit45",
        "pair_unit90",
        "segment_energy9",
    ):
        yield variant, "extra_trees_400", ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=20260908,
            n_jobs=-1,
        )
        yield variant, "lda_shrinkage", make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        )
        yield variant, "logistic", make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)
        )


def score(train: list[dict], test: list[dict], variant: str, template, coordinates) -> dict:
    x_train = transform(np.stack([row["features"][FEATURE] for row in train]), variant)
    y_train = np.asarray([row["point_index"] for row in train], dtype=int)
    x_test = transform(np.stack([row["features"][FEATURE] for row in test]), variant)
    truth = np.asarray([row["point_index"] for row in test], dtype=int)
    model = clone(template).fit(x_train, y_train)
    predicted = np.asarray(model.predict(x_test), dtype=int)
    errors = np.asarray(
        [np.linalg.norm(coordinates[int(a)] - coordinates[int(b)]) for a, b in zip(predicted, truth, strict=True)],
        dtype=float,
    )
    return {
        "exact": int(np.sum(predicted == truth)),
        "exact_percent": float(100.0 * np.mean(predicted == truth)),
        "within_4mm_percent": float(100.0 * np.mean(errors <= 4.0 + 1e-6)),
        "p95_error_mm": float(np.percentile(errors, 95)),
        "maximum_error_mm": float(np.max(errors)),
        "predictions": [int(value) for value in predicted],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("depth_0p10_manifest", type=Path)
    parser.add_argument("depth_0p80_manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output must be new")
    here = Path(__file__).resolve().parent
    old = load_rounds(
        (here.parent / "outputs" / "fast_fullmap_position_rounds_0p30_20260906.json").resolve(),
        [
            (here.parent / "outputs" / "fast_fullmap_position_validation_r3_0p30_20260906.json").resolve(),
            here / "outputs" / "f45_position_validation_after_por_r1_20260908.json",
            here / "outputs" / "f45_position_validation_after_lock_r2_20260908.json",
        ],
    )
    rounds = {
        "depth_0p10_r1": load_depth_manifest(args.depth_0p10_manifest),
        **old,
        "depth_0p80_r1": load_depth_manifest(args.depth_0p80_manifest),
    }
    coordinates = {
        int(row["point_index"]): np.asarray(row["xy"], dtype=float)
        for row in old["old_r1"]
    }
    rows = []
    selection_names = tuple(old)
    for variant, estimator_name, template in candidates():
        selection_folds = []
        for held_name in selection_names:
            train = [row for name, values in rounds.items() if name != held_name for row in values]
            selection_folds.append(
                {"held_round": held_name, **score(train, rounds[held_name], variant, template, coordinates)}
            )
        endpoint_folds = []
        for held_name in ("depth_0p10_r1", "depth_0p80_r1"):
            train = [row for name, values in rounds.items() if name != held_name for row in values]
            endpoint_folds.append(
                {"held_round": held_name, **score(train, rounds[held_name], variant, template, coordinates)}
            )
        all_rows = [row for values in rounds.values() for row in values]
        training_fit = score(all_rows, all_rows, variant, template, coordinates)
        rows.append(
            {
                "feature": FEATURE,
                "preprocessing": variant,
                "estimator": estimator_name,
                "selection_folds": selection_folds,
                "worst_selection_exact": min(item["exact"] for item in selection_folds),
                "worst_selection_within_4mm_percent": min(item["within_4mm_percent"] for item in selection_folds),
                "mean_selection_exact_percent": float(np.mean([item["exact_percent"] for item in selection_folds])),
                "endpoint_leave_depth_out_diagnostic": endpoint_folds,
                "training_fit": training_fit,
            }
        )
    rows.sort(
        key=lambda item: (
            -item["worst_selection_exact"],
            -item["worst_selection_within_4mm_percent"],
            -item["mean_selection_exact_percent"],
            -min(fold["exact"] for fold in item["endpoint_leave_depth_out_diagnostic"]),
            item["training_fit"]["p95_error_mm"],
        )
    )
    result = {
        "schema": "f45_multidepth_position_candidate_selection_v1",
        "training_depths_mm": [0.10, 0.30, 0.80],
        "rounds": list(rounds),
        "contacts": sum(len(values) for values in rounds.values()),
        "model_selection_folds": list(selection_names),
        "random_frame_split_used": False,
        "untouched_validation_required": "complete 17-point rounds at 0.20 and 0.60 mm",
        "deployment_eligible": False,
        "best": rows[0],
        "ranked": rows,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(result["best"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
