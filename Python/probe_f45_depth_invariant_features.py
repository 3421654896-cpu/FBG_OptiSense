"""Probe scale-invariant F45 position features before more machine contacts.

The existing five independent 0.30 mm rounds remain whole-round holdouts.  The
new P9 depth pilot is used only as an out-of-depth diagnostic, not as a formal
all-position validation set.
"""

from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze_f45_depth_pilot import EXPECTED_DEPTHS
from evaluate_f45_cross_session_frame_models import causal_held_features
from evaluate_f45_cross_session_models import load_rounds


FEATURE = "pair_transition_delta45"


def _unit_rows(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def transform(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 135:
        raise ValueError("expected pair-transition 135-D features")
    if name == "raw135":
        return values
    if name == "unit135":
        return _unit_rows(values)
    if name == "signed_sqrt_unit135":
        return _unit_rows(np.sign(values) * np.sqrt(np.abs(values)))
    blocks = values.reshape(-1, 3, 45)
    if name == "block_unit135":
        return (
            blocks
            / np.maximum(np.linalg.norm(blocks, axis=2, keepdims=True), 1e-12)
        ).reshape(-1, 135)
    if name == "first_unit45":
        return _unit_rows(blocks[:, 0, :])
    if name == "second_unit45":
        return _unit_rows(blocks[:, 1, :])
    if name == "pair_unit90":
        return _unit_rows(blocks[:, :2, :].reshape(-1, 90))
    if name == "segment_energy9":
        first = blocks[:, 0, :].reshape(-1, 9, 5)
        energy = np.sqrt(np.mean(first**2, axis=2))
        return _unit_rows(energy)
    raise ValueError(name)


def estimators() -> list[tuple[str, object]]:
    return [
        (
            "extra_trees_400",
            ExtraTreesClassifier(
                n_estimators=400,
                max_features="sqrt",
                min_samples_leaf=1,
                random_state=20260908,
                n_jobs=-1,
            ),
        ),
        (
            "lda_shrinkage",
            make_pipeline(
                StandardScaler(),
                LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            ),
        ),
        (
            "logistic",
            make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
        ),
        ("knn_cosine", KNeighborsClassifier(5, weights="distance", metric="cosine")),
    ]


def contact_matrix(path: Path) -> np.ndarray:
    return causal_held_features(path)[FEATURE]


def latest_depth_paths(root: Path, point: int) -> dict[float, Path]:
    result = {}
    for depth in EXPECTED_DEPTHS:
        token = f"{depth:.2f}".replace(".", "p")
        candidates = sorted(
            root.glob(f"f45_depthpilot_p{point:02d}_{token}_*/capture.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in candidates:
            try:
                contact_matrix(path)
            except Exception:
                continue
            result[depth] = path
            break
        if depth not in result:
            raise ValueError(f"no valid depth pilot at {depth:.2f} mm")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output must be new")

    python_root = Path(__file__).resolve().parent
    rounds = load_rounds(
        (python_root.parent / "outputs" / "fast_fullmap_position_rounds_0p30_20260906.json").resolve(),
        [
            (python_root.parent / "outputs" / "fast_fullmap_position_validation_r3_0p30_20260906.json").resolve(),
            (python_root / "outputs" / "f45_position_validation_after_por_r1_20260908.json").resolve(),
            (python_root / "outputs" / "f45_position_validation_after_lock_r2_20260908.json").resolve(),
        ],
    )
    matrices = {
        (round_name, int(row["point_index"])): contact_matrix(Path(row["path"]))
        for round_name, records in rounds.items()
        for row in records
    }
    depth_paths = latest_depth_paths(python_root / "outputs", 9)
    depth_matrices = {depth: contact_matrix(path) for depth, path in depth_paths.items()}
    variants = (
        "raw135",
        "unit135",
        "signed_sqrt_unit135",
        "block_unit135",
        "first_unit45",
        "second_unit45",
        "pair_unit90",
        "segment_energy9",
    )
    rows = []
    for variant in variants:
        for estimator_name, template in estimators():
            fold_rows = []
            for held_name in rounds:
                x_train = np.concatenate(
                    [
                        transform(matrices[(name, int(row["point_index"]))], variant)
                        for name, records in rounds.items()
                        if name != held_name
                        for row in records
                    ]
                )
                y_train = np.concatenate(
                    [
                        np.full(len(matrices[(name, int(row["point_index"]))]), int(row["point_index"]), dtype=int)
                        for name, records in rounds.items()
                        if name != held_name
                        for row in records
                    ]
                )
                x_test = np.concatenate(
                    [
                        transform(matrices[(held_name, int(row["point_index"]))], variant)
                        for row in rounds[held_name]
                    ]
                )
                truth_by_contact = [
                    (int(row["point_index"]), len(matrices[(held_name, int(row["point_index"]))]))
                    for row in rounds[held_name]
                ]
                model = clone(template).fit(x_train, y_train)
                prediction = np.asarray(model.predict(x_test), dtype=int)
                offset = 0
                majority_exact = 0
                frame_exact = 0
                total = 0
                for truth, count in truth_by_contact:
                    block = prediction[offset : offset + count]
                    majority = Counter(block.tolist()).most_common(1)[0][0]
                    majority_exact += int(majority == truth)
                    frame_exact += int(np.sum(block == truth))
                    total += count
                    offset += count
                fold_rows.append(
                    {
                        "held_round": held_name,
                        "trial_majority_exact": majority_exact,
                        "frame_exact_percent": 100.0 * frame_exact / total,
                    }
                )

            x_all = np.concatenate(
                [
                    transform(matrices[(name, int(row["point_index"]))], variant)
                    for name, records in rounds.items()
                    for row in records
                ]
            )
            y_all = np.concatenate(
                [
                    np.full(len(matrices[(name, int(row["point_index"]))]), int(row["point_index"]), dtype=int)
                    for name, records in rounds.items()
                    for row in records
                ]
            )
            fitted = clone(template).fit(x_all, y_all)
            depth_results = {}
            for depth, matrix in depth_matrices.items():
                prediction = np.asarray(fitted.predict(transform(matrix, variant)), dtype=int)
                counts = Counter(prediction.tolist())
                majority = counts.most_common(1)[0][0]
                depth_results[f"{depth:.2f}"] = {
                    "majority": int(majority),
                    "exact_frame_percent": float(100.0 * np.mean(prediction == 9)),
                    "counts": {str(k): int(v) for k, v in sorted(counts.items())},
                }
            rows.append(
                {
                    "variant": variant,
                    "estimator": estimator_name,
                    "worst_0p30_round_trial_majority": min(
                        row["trial_majority_exact"] for row in fold_rows
                    ),
                    "worst_0p30_round_frame_exact_percent": min(
                        row["frame_exact_percent"] for row in fold_rows
                    ),
                    "p9_depth_majority_exact_count": sum(
                        row["majority"] == 9 for row in depth_results.values()
                    ),
                    "p9_depth_minimum_exact_frame_percent": min(
                        row["exact_frame_percent"] for row in depth_results.values()
                    ),
                    "folds": fold_rows,
                    "p9_depth_probe": depth_results,
                }
            )
    rows.sort(
        key=lambda row: (
            -row["p9_depth_majority_exact_count"],
            -row["worst_0p30_round_trial_majority"],
            -row["worst_0p30_round_frame_exact_percent"],
            -row["p9_depth_minimum_exact_frame_percent"],
        )
    )
    result = {
        "schema": "f45_depth_invariant_feature_probe_v1",
        "selection_status": "exploratory_only_p9_depths_are_not_all_position_validation",
        "training_rounds_0p30mm": list(rounds),
        "best": rows[0],
        "ranked": rows,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(result["best"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
