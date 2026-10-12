"""Select a depth-robust continuous-position decoder for guarded C27 data.

The five existing 17-position acquisition passes are kept as indivisible
folds.  This prevents adjacent frames from one physical press from appearing
in both training and validation.  The script is offline-only and never opens
the optical board or commands Mach3.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from evaluate_candidate27_feature_variants import _capture


TOOL_RADIUS_MM = 4.0
ARCHIVED_FIRMWARE = 0x00010031
BASE_FEATURES = (
    "legacy_shape_delta",
    "delta_unit_global",
    "delta_unit_per_fbg",
    "group_energy_share",
    "flank_shift",
    "delta_plus_energy",
    "per_fbg_plus_energy",
    "per_fbg_plus_flank",
    "per_fbg_energy_flank",
)
READ_SOURCES = ("first_code", "second_code", "mean_reads", "concat_reads")


def _local_source(value: str, output_root: Path) -> Path:
    """Use report provenance while keeping archived reports relocatable."""
    candidate = Path(value)
    if candidate.exists():
        return candidate.resolve()
    fallback = output_root / candidate.name
    if not fallback.exists():
        raise FileNotFoundError(fallback)
    return fallback.resolve()


def load_default_rounds(repo_root: Path) -> dict[str, list[Path]]:
    output_root = repo_root / "outputs"
    d020 = json.loads(
        (output_root / "guarded_c27_full17_feature_variants_pass1_to_pass2_v1049_20260907.json")
        .read_text("utf-8")
    )
    d040_model = json.loads(
        (output_root / "guarded_c27_full17_grouped_model_0p40_v1049_20260907.json")
        .read_text("utf-8")
    )
    d040_holdout = json.loads(
        (output_root / "guarded_c27_full17_grouped_model_holdout3_0p40_v1049_20260907.json")
        .read_text("utf-8")
    )
    grouped_training = [
        _local_source(row["source"], output_root) for row in d040_model["training"]
    ]
    rounds = {
        "d020_r1": [_local_source(value, output_root) for value in d020["training"]],
        "d020_r2": [_local_source(value, output_root) for value in d020["validation"]],
        "d040_train": [path for path in grouped_training if path.stem.endswith("_train")],
        "d040_val": [path for path in grouped_training if path.stem.endswith("_val")],
        "d040_holdout3": [
            _local_source(row["source"], output_root) for row in d040_holdout["validation"]
        ],
    }
    for name, paths in rounds.items():
        if len(paths) != 17:
            raise ValueError(f"{name}: expected 17 contacts, found {len(paths)}")
    return rounds


def _physical_xy(row: dict) -> np.ndarray:
    target = row["target"]
    if "calibration_work_xy_mm" in target:
        return np.asarray(target["calibration_work_xy_mm"], dtype=float)
    return np.asarray([target["work_x"], target["work_y"]], dtype=float)


def _load_capture_with_context(path: Path) -> dict:
    try:
        return _capture(path, expected_firmware=ARCHIVED_FIRMWARE)
    except Exception as exc:
        raise ValueError(f"failed to load guarded capture {path}: {exc}") from exc


def _field_feature(row: dict, field: str, feature: str) -> np.ndarray:
    values = row["fields"][field]
    if feature in values:
        return np.asarray(values[feature], dtype=float)
    per_fbg = np.asarray(values["delta_unit_per_fbg"], dtype=float)
    energy = np.asarray(values["group_energy_share"], dtype=float)
    flank = np.asarray(values["flank_shift"], dtype=float)
    if feature == "per_fbg_plus_energy":
        return np.concatenate([per_fbg, energy], axis=1)
    if feature == "per_fbg_plus_flank":
        return np.concatenate([per_fbg, flank], axis=1)
    if feature == "per_fbg_energy_flank":
        return np.concatenate([per_fbg, energy, flank], axis=1)
    raise ValueError(feature)


def frame_features(row: dict, read_source: str, feature: str) -> np.ndarray:
    first = _field_feature(row, "first_code", feature)
    second = _field_feature(row, "second_code", feature)
    if read_source == "first_code":
        return first
    if read_source == "second_code":
        return second
    if read_source == "mean_reads":
        return (first + second) / 2.0
    if read_source == "concat_reads":
        return np.concatenate([first, second], axis=1)
    raise ValueError(read_source)


def candidates():
    estimators = (
        (
            "lda_shrinkage",
            make_pipeline(
                StandardScaler(),
                LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            ),
        ),
        (
            "logistic_c0p1",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced"),
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
            "rbf_c10",
            make_pipeline(StandardScaler(), SVC(C=10.0, gamma="scale")),
        ),
        (
            "extra_trees_leaf2_half",
            ExtraTreesClassifier(
                n_estimators=100,
                min_samples_leaf=2,
                max_features=0.5,
                class_weight="balanced",
                random_state=20260908,
                n_jobs=-1,
            ),
        ),
    )
    for read_source in READ_SOURCES:
        for feature in BASE_FEATURES:
            for estimator_name, estimator in estimators:
                yield read_source, feature, estimator_name, "hard", estimator
                if estimator_name.startswith(("lda_", "logistic_")):
                    yield read_source, feature, estimator_name, "posterior_xy", estimator


def _nearest_labels(
    predicted_xy: np.ndarray,
    labels: np.ndarray,
    label_xy: np.ndarray,
) -> np.ndarray:
    distances = np.linalg.norm(predicted_xy[:, None, :] - label_xy[None, :, :], axis=2)
    return labels[np.argmin(distances, axis=1)]


def _decode(model, decoder: str, values: np.ndarray, coordinates: dict[int, np.ndarray]):
    labels = np.asarray(sorted(coordinates), dtype=int)
    label_xy = np.stack([coordinates[int(label)] for label in labels])
    hard = np.asarray(model.predict(values), dtype=int)
    if decoder == "hard":
        return np.stack([coordinates[int(label)] for label in hard]), hard
    model_classes = np.asarray(model.classes_, dtype=int)
    probability = np.asarray(model.predict_proba(values), dtype=float)
    class_xy = np.stack([coordinates[int(label)] for label in model_classes])
    predicted_xy = probability @ class_xy
    return predicted_xy, _nearest_labels(predicted_xy, labels, label_xy)


def _majority(values: np.ndarray) -> int:
    counts = Counter(map(int, values))
    return min(counts, key=lambda label: (-counts[label], label))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output must be new")

    python_root = Path(__file__).resolve().parent
    repo_root = python_root.parent
    round_paths = load_default_rounds(repo_root)
    rounds = {
        name: [_load_capture_with_context(path) for path in paths]
        for name, paths in round_paths.items()
    }
    for name, rows in rounds.items():
        labels = sorted(int(row["point"]) for row in rows)
        if labels != list(range(1, 18)):
            raise ValueError(f"{name}: invalid point labels {labels}")

    coordinate_samples: dict[int, list[np.ndarray]] = {point: [] for point in range(1, 18)}
    for rows in rounds.values():
        for row in rows:
            coordinate_samples[int(row["point"])].append(_physical_xy(row))
    coordinates = {
        point: np.median(np.stack(samples), axis=0)
        for point, samples in coordinate_samples.items()
    }
    coordinate_spread = {
        point: float(np.max(np.linalg.norm(np.stack(samples) - coordinates[point], axis=1)))
        for point, samples in coordinate_samples.items()
    }
    if max(coordinate_spread.values()) > 0.05:
        raise ValueError(f"physical coordinate mismatch between rounds: {coordinate_spread}")

    cache = {
        (round_name, int(row["point"]), read_source, feature): frame_features(
            row, read_source, feature
        )
        for round_name, rows in rounds.items()
        for row in rows
        for read_source in READ_SOURCES
        for feature in BASE_FEATURES
    }
    labels = np.asarray(sorted(coordinates), dtype=int)
    label_xy = np.stack([coordinates[int(label)] for label in labels])

    ranked = []
    for read_source, feature, estimator_name, decoder, template in candidates():
        folds = []
        for held_name, held_rows in rounds.items():
            train_x = []
            train_y = []
            for round_name, rows in rounds.items():
                if round_name == held_name:
                    continue
                for row in rows:
                    point = int(row["point"])
                    values = cache[(round_name, point, read_source, feature)]
                    train_x.append(values)
                    train_y.append(np.full(len(values), point, dtype=int))
            model = clone(template).fit(np.concatenate(train_x), np.concatenate(train_y))

            frame_errors = []
            frame_exact = []
            trials = []
            for row in held_rows:
                point = int(row["point"])
                values = cache[(held_name, point, read_source, feature)]
                predicted_xy, predicted_label = _decode(model, decoder, values, coordinates)
                errors = np.linalg.norm(predicted_xy - coordinates[point], axis=1)
                trial_xy = np.median(predicted_xy, axis=0)
                trial_error = float(np.linalg.norm(trial_xy - coordinates[point]))
                trial_label = int(_nearest_labels(trial_xy[None, :], labels, label_xy)[0])
                counts = Counter(map(int, predicted_label))
                frame_errors.extend(errors.tolist())
                frame_exact.extend((predicted_label == point).tolist())
                trials.append(
                    {
                        "point": point,
                        "frames": int(len(values)),
                        "trial_xy_mm": [float(value) for value in trial_xy],
                        "trial_error_mm": trial_error,
                        "trial_nearest_label": trial_label,
                        "trial_nearest_label_exact": trial_label == point,
                        "frame_error_p95_mm": float(np.percentile(errors, 95)),
                        "frame_within_tool_radius_percent": float(
                            100.0 * np.mean(errors <= TOOL_RADIUS_MM + 1e-6)
                        ),
                        "nearest_label_counts": {
                            str(key): int(value) for key, value in sorted(counts.items())
                        },
                    }
                )
            frame_errors_array = np.asarray(frame_errors, dtype=float)
            trial_errors = np.asarray([row["trial_error_mm"] for row in trials], dtype=float)
            folds.append(
                {
                    "held_round": held_name,
                    "depth_mm": float(np.median([row["metadata"]["depth_mm"] for row in held_rows])),
                    "held_points": len(held_rows),
                    "trial_nearest_label_exact": int(
                        sum(row["trial_nearest_label_exact"] for row in trials)
                    ),
                    "trial_within_tool_radius": int(
                        np.sum(trial_errors <= TOOL_RADIUS_MM + 1e-6)
                    ),
                    "trial_error_mean_mm": float(np.mean(trial_errors)),
                    "trial_error_p95_mm": float(np.percentile(trial_errors, 95)),
                    "trial_error_maximum_mm": float(np.max(trial_errors)),
                    "frame_nearest_label_exact_percent": float(100.0 * np.mean(frame_exact)),
                    "frame_within_tool_radius_percent": float(
                        100.0 * np.mean(frame_errors_array <= TOOL_RADIUS_MM + 1e-6)
                    ),
                    "frame_error_mean_mm": float(np.mean(frame_errors_array)),
                    "frame_error_p95_mm": float(np.percentile(frame_errors_array, 95)),
                    "frame_error_maximum_mm": float(np.max(frame_errors_array)),
                    "trials": trials,
                }
            )
        ranked.append(
            {
                "read_source": read_source,
                "feature": feature,
                "estimator": estimator_name,
                "decoder": decoder,
                "worst_trial_within_tool_radius_percent": float(
                    min(100.0 * row["trial_within_tool_radius"] / row["held_points"] for row in folds)
                ),
                "worst_frame_within_tool_radius_percent": float(
                    min(row["frame_within_tool_radius_percent"] for row in folds)
                ),
                "worst_trial_error_maximum_mm": float(
                    max(row["trial_error_maximum_mm"] for row in folds)
                ),
                "worst_frame_error_p95_mm": float(max(row["frame_error_p95_mm"] for row in folds)),
                "worst_trial_nearest_label_exact_percent": float(
                    min(100.0 * row["trial_nearest_label_exact"] / row["held_points"] for row in folds)
                ),
                "mean_frame_error_mm": float(np.mean([row["frame_error_mean_mm"] for row in folds])),
                "folds": folds,
            }
        )

    ranked.sort(
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
        "schema": "c27_multidepth_continuous_position_selection_v2",
        "tool_radius_mm": TOOL_RADIUS_MM,
        "target": "calibrated_press_centre_xy_mm",
        "rounds": {
            name: {
                "depth_mm": float(np.median([row["metadata"]["depth_mm"] for row in rows])),
                "points": [int(row["point"]) for row in rows],
                "captures": [row["path"] for row in rows],
            }
            for name, rows in rounds.items()
        },
        "coordinate_maximum_round_spread_mm": float(max(coordinate_spread.values())),
        "random_frame_split_used": False,
        "deployment_eligible": False,
        "best": ranked[0],
        "ranked": ranked,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(
        json.dumps(
            [{key: value for key, value in row.items() if key != "folds"} for row in ranked[:20]],
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
