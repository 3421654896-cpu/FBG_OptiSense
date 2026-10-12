"""Select a causal continuous-position decoder for the 8 mm F45 press tool.

Every acquisition round is held out as a whole.  Frames from one physical
contact are therefore never split between training and validation.  The
primary metric is continuous XY error relative to the calibrated press-centre
coordinate; the legacy 1..17 grid label is retained only as a diagnostic.

This is an offline model-selection tool.  It never opens the optical board or
commands Mach3, and its output is not deployment-authorising evidence.  After
selection the chosen algorithm must be locked and tested on a new untouched
physical round.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_f45_cross_session_frame_models import causal_held_features
from evaluate_f45_cross_session_models import load_rounds
from evaluate_fast_fullmap_position import _capture_feature
from select_f45_multidepth_position_candidate import transform


FEATURE = "pair_transition_delta45"
TOOL_RADIUS_MM = 4.0


@dataclass(frozen=True)
class Candidate:
    representation: str
    estimator_name: str
    kind: str
    decoder: str
    estimator: object


def _parse_round(value: str) -> tuple[str, Path]:
    try:
        name, raw_path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("round must be NAME=MANIFEST") from exc
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("round name must not be empty")
    return name, Path(raw_path).expanduser().resolve()


def _load_visible_round(path: Path) -> list[dict]:
    payload = json.loads(path.read_text("utf-8"))
    completed = sorted(int(value) for value in payload.get("completed_points") or ())
    if (
        payload.get("schema") != "f45_visible_depth_round_v1"
        or not payload.get("complete")
        or not completed
    ):
        raise ValueError(f"{path}: not a completed visible F45 round")
    records = []
    for point in completed:
        entry = payload["captures"][str(point)]
        if not all((entry.get("checks") or {}).values()):
            raise ValueError(f"{path}: point {point} failed guarded capture checks")
        record = _capture_feature(Path(entry["capture"]))
        if int(record["point_index"]) != point:
            raise ValueError(f"{path}: point label mismatch")
        records.append(record)
    return records


def _candidate_models():
    for representation in ("pair_unit90", "unit135", "block_unit135"):
        classifiers = (
            (
                "lda_shrinkage",
                make_pipeline(
                    StandardScaler(),
                    LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
                ),
            ),
            (
                "logistic_c1",
                make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced"),
                ),
            ),
            (
                "knn3_distance",
                KNeighborsClassifier(n_neighbors=3, weights="distance", n_jobs=-1),
            ),
            (
                "extra_trees_leaf1_sqrt",
                ExtraTreesClassifier(
                    n_estimators=300,
                    min_samples_leaf=1,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=20260908,
                    n_jobs=-1,
                ),
            ),
            (
                "extra_trees_leaf2_half",
                ExtraTreesClassifier(
                    n_estimators=300,
                    min_samples_leaf=2,
                    max_features=0.5,
                    class_weight="balanced",
                    random_state=20260908,
                    n_jobs=-1,
                ),
            ),
            (
                "random_forest_leaf1_sqrt",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=1,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=20260908,
                    n_jobs=-1,
                ),
            ),
        )
        for estimator_name, estimator in classifiers:
            yield Candidate(representation, estimator_name, "classifier", "hard", estimator)
            yield Candidate(representation, estimator_name, "classifier", "posterior_xy", estimator)

        regressors = (
            (
                "ridge_xy_a10",
                make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
            ),
            (
                "ridge_xy_a100",
                make_pipeline(StandardScaler(), Ridge(alpha=100.0)),
            ),
            (
                "knn_xy_3_distance",
                KNeighborsRegressor(n_neighbors=3, weights="distance", n_jobs=-1),
            ),
            (
                "extra_trees_xy_leaf2_half",
                ExtraTreesRegressor(
                    n_estimators=300,
                    min_samples_leaf=2,
                    max_features=0.5,
                    random_state=20260908,
                    n_jobs=-1,
                ),
            ),
            (
                "random_forest_xy_leaf1_sqrt",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=1,
                    max_features="sqrt",
                    random_state=20260908,
                    n_jobs=-1,
                ),
            ),
        )
        for estimator_name, estimator in regressors:
            yield Candidate(representation, estimator_name, "xy_regressor", "direct_xy", estimator)


def _nearest_labels(
    predicted_xy: np.ndarray,
    labels: np.ndarray,
    label_xy: np.ndarray,
) -> np.ndarray:
    distances = np.linalg.norm(predicted_xy[:, None, :] - label_xy[None, :, :], axis=2)
    return labels[np.argmin(distances, axis=1)]


def _decode(model, candidate: Candidate, values: np.ndarray, coordinates: dict[int, np.ndarray]):
    labels = np.asarray(sorted(coordinates), dtype=int)
    label_xy = np.stack([coordinates[int(label)] for label in labels])
    if candidate.kind == "xy_regressor":
        predicted_xy = np.asarray(model.predict(values), dtype=float)
        return predicted_xy, _nearest_labels(predicted_xy, labels, label_xy)

    hard = np.asarray(model.predict(values), dtype=int)
    if candidate.decoder == "hard":
        return np.stack([coordinates[int(label)] for label in hard]), hard
    model_classes = np.asarray(model.classes_, dtype=int)
    probability = np.asarray(model.predict_proba(values), dtype=float)
    class_xy = np.stack([coordinates[int(label)] for label in model_classes])
    predicted_xy = probability @ class_xy
    return predicted_xy, _nearest_labels(predicted_xy, labels, label_xy)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--round",
        action="append",
        default=[],
        type=_parse_round,
        metavar="NAME=MANIFEST",
        help="completed full or partial f45_visible_depth_round_v1 manifest",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output must be new")

    here = Path(__file__).resolve().parent
    rounds = load_rounds(
        (here.parent / "outputs" / "fast_fullmap_position_rounds_0p30_20260906.json").resolve(),
        [
            (here.parent / "outputs" / "fast_fullmap_position_validation_r3_0p30_20260906.json").resolve(),
            here / "outputs" / "f45_position_validation_after_por_r1_20260908.json",
            here / "outputs" / "f45_position_validation_after_lock_r2_20260908.json",
        ],
    )
    for name, path in args.round:
        if name in rounds:
            parser.error(f"duplicate round name: {name}")
        rounds[name] = _load_visible_round(path)

    coordinates = {
        int(row["point_index"]): np.asarray(row["xy"], dtype=float)
        for row in rounds["old_r1"]
    }
    raw = {
        (round_name, int(row["point_index"])): causal_held_features(Path(row["path"]))[FEATURE]
        for round_name, records in rounds.items()
        for row in records
    }
    labels = np.asarray(sorted(coordinates), dtype=int)
    label_xy = np.stack([coordinates[int(label)] for label in labels])

    ranked = []
    for candidate in _candidate_models():
        folds = []
        failed = None
        for held_name, held_records in rounds.items():
            train_parts = []
            train_labels = []
            for round_name, records in rounds.items():
                if round_name == held_name:
                    continue
                for row in records:
                    point = int(row["point_index"])
                    values = raw[(round_name, point)]
                    train_parts.append(values)
                    train_labels.append(np.full(len(values), point, dtype=int))
            raw_train = np.concatenate(train_parts)
            point_train = np.concatenate(train_labels)
            x_train = transform(raw_train, candidate.representation)
            target_train = (
                point_train
                if candidate.kind == "classifier"
                else np.stack([coordinates[int(point)] for point in point_train])
            )
            try:
                model = clone(candidate.estimator).fit(x_train, target_train)
                frame_errors = []
                frame_exact = []
                trials = []
                for row in held_records:
                    point = int(row["point_index"])
                    x_test = transform(raw[(held_name, point)], candidate.representation)
                    predicted_xy, predicted_label = _decode(
                        model, candidate, x_test, coordinates
                    )
                    errors = np.linalg.norm(predicted_xy - coordinates[point], axis=1)
                    frame_errors.extend(errors.tolist())
                    frame_exact.extend((predicted_label == point).tolist())
                    trial_xy = np.median(predicted_xy, axis=0)
                    trial_error = float(np.linalg.norm(trial_xy - coordinates[point]))
                    trial_label = int(_nearest_labels(trial_xy[None, :], labels, label_xy)[0])
                    counts = Counter(predicted_label.tolist())
                    trials.append(
                        {
                            "point": point,
                            "frames": int(len(x_test)),
                            "trial_xy_mm": [float(value) for value in trial_xy],
                            "trial_error_mm": trial_error,
                            "trial_nearest_label": trial_label,
                            "trial_nearest_label_exact": trial_label == point,
                            "frame_error_median_mm": float(np.median(errors)),
                            "frame_error_p95_mm": float(np.percentile(errors, 95)),
                            "frame_within_tool_radius_percent": float(
                                100.0 * np.mean(errors <= TOOL_RADIUS_MM + 1e-6)
                            ),
                            "nearest_label_counts": {
                                str(key): int(value) for key, value in sorted(counts.items())
                            },
                        }
                    )
            except Exception as exc:
                failed = f"{type(exc).__name__}: {exc}"
                break
            frame_errors = np.asarray(frame_errors, dtype=float)
            trial_errors = np.asarray([item["trial_error_mm"] for item in trials])
            folds.append(
                {
                    "held_round": held_name,
                    "held_points": len(held_records),
                    "depth_mm": float(np.median([row["depth_mm"] for row in held_records])),
                    "trial_nearest_label_exact": int(
                        sum(item["trial_nearest_label_exact"] for item in trials)
                    ),
                    "trial_within_tool_radius": int(
                        np.sum(trial_errors <= TOOL_RADIUS_MM + 1e-6)
                    ),
                    "trial_error_mean_mm": float(np.mean(trial_errors)),
                    "trial_error_p95_mm": float(np.percentile(trial_errors, 95)),
                    "trial_error_maximum_mm": float(np.max(trial_errors)),
                    "frame_nearest_label_exact_percent": float(100.0 * np.mean(frame_exact)),
                    "frame_within_tool_radius_percent": float(
                        100.0 * np.mean(frame_errors <= TOOL_RADIUS_MM + 1e-6)
                    ),
                    "frame_error_mean_mm": float(np.mean(frame_errors)),
                    "frame_error_p95_mm": float(np.percentile(frame_errors, 95)),
                    "frame_error_maximum_mm": float(np.max(frame_errors)),
                    "trials": trials,
                }
            )
        if failed:
            ranked.append(
                {
                    "representation": candidate.representation,
                    "estimator": candidate.estimator_name,
                    "kind": candidate.kind,
                    "decoder": candidate.decoder,
                    "failed": failed,
                }
            )
            continue
        ranked.append(
            {
                "representation": candidate.representation,
                "estimator": candidate.estimator_name,
                "kind": candidate.kind,
                "decoder": candidate.decoder,
                "worst_trial_within_tool_radius_percent": float(
                    min(100.0 * row["trial_within_tool_radius"] / row["held_points"] for row in folds)
                ),
                "worst_frame_within_tool_radius_percent": float(
                    min(row["frame_within_tool_radius_percent"] for row in folds)
                ),
                "worst_trial_error_maximum_mm": float(
                    max(row["trial_error_maximum_mm"] for row in folds)
                ),
                "worst_frame_error_p95_mm": float(
                    max(row["frame_error_p95_mm"] for row in folds)
                ),
                "worst_trial_nearest_label_exact_percent": float(
                    min(100.0 * row["trial_nearest_label_exact"] / row["held_points"] for row in folds)
                ),
                "mean_frame_error_mm": float(
                    np.mean([row["frame_error_mean_mm"] for row in folds])
                ),
                "folds": folds,
            }
        )

    viable = [row for row in ranked if not row.get("failed")]
    viable.sort(
        key=lambda row: (
            -row["worst_trial_within_tool_radius_percent"],
            -row["worst_frame_within_tool_radius_percent"],
            row["worst_trial_error_maximum_mm"],
            row["worst_frame_error_p95_mm"],
            -row["worst_trial_nearest_label_exact_percent"],
            row["mean_frame_error_mm"],
        )
    )
    result = {
        "schema": "f45_continuous_position_selection_v3",
        "feature": FEATURE,
        "tool_radius_mm": TOOL_RADIUS_MM,
        "target": "calibrated_press_centre_xy_mm",
        "rounds": {
            name: {
                "points": [int(row["point_index"]) for row in records],
                "depth_mm": float(np.median([row["depth_mm"] for row in records])),
            }
            for name, records in rounds.items()
        },
        "random_frame_split_used": False,
        "deployment_eligible": False,
        "best": viable[0],
        "ranked": viable + [row for row in ranked if row.get("failed")],
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(
        json.dumps(
            [
                {key: value for key, value in row.items() if key != "folds"}
                for row in viable[:15]
            ],
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
