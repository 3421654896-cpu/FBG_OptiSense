"""Evaluate the centre-three rows of each installed five-row FBG group.

The input remains the strictly validated F45 physical captures, but only rows
1, 2 and 3 of every five-row group enter the model.  This answers whether a
27-point route can retain spatial information across depth before spending
machine time qualifying that route.  It does not prove C27 optical-path or
timing equivalence and is never deployment evidence.
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

from evaluate_candidate27_feature_variants import _features
from evaluate_f45_cross_session_models import load_rounds
from evaluate_fast_fullmap_position import _capture_feature
from select_c27_multidepth_continuous_position_v2 import (
    TOOL_RADIUS_MM,
    _decode,
    _nearest_labels,
    frame_features,
)
from select_f45_continuous_position_v3 import _load_visible_round


CORE_ROWS = np.asarray([group * 5 + offset for group in range(9) for offset in (1, 2, 3)])
FEATURES = (
    "delta_plus_energy",
    "delta_unit_global",
    "per_fbg_plus_energy",
    "per_fbg_energy_flank",
)
READ_SOURCES = ("second_code", "mean_reads", "first_code")


def _matrix(frames: list[dict], field: str, path: Path) -> np.ndarray:
    values = np.asarray(
        [[record[field] for record in frame["records"]] for frame in frames],
        dtype=float,
    )
    if values.ndim != 2 or values.shape[1] != 45 or not np.isfinite(values).all():
        raise ValueError(f"{path.name}: malformed {field} matrix")
    return values[:, CORE_ROWS]


def load_f27_capture(path: Path) -> dict:
    validated = _capture_feature(path)
    payload = json.loads(path.read_text("utf-8"))
    commission = payload["commissioning"]
    by_sequence = {int(frame["sequence"]): frame for frame in payload["frames"]}
    receipts = {
        int(row["sequence"]): int(row["host_monotonic_ns"])
        for row in commission["frame_receipts"]
    }
    events = {
        row["kind"]: int(row["host_monotonic_ns"])
        for row in commission["events"]
    }
    baseline = [by_sequence[sequence] for sequence in range(16, 80)]
    held = [
        frame
        for sequence, frame in sorted(by_sequence.items())
        if events["contact_reached_commanded_depth"] + 200_000_000
        <= receipts[sequence]
        < events["release_started"]
    ]
    if len(baseline) != 64 or len(held) < 25:
        raise ValueError(f"{path.name}: insufficient causal baseline/held frames")
    fields = {}
    matrices = {}
    for field in ("first_code", "second_code"):
        baseline_matrix = _matrix(baseline, field, path)
        held_matrix = _matrix(held, field, path)
        matrices[field] = (baseline_matrix, held_matrix)
        fields[field] = _features(
            baseline_matrix,
            held_matrix,
        )
    base_first = np.median(matrices["first_code"][0], axis=0)
    base_second = np.median(matrices["second_code"][0], axis=0)
    held_first = matrices["first_code"][1]
    held_second = matrices["second_code"][1]
    pair_transition = np.concatenate(
        (
            held_first - base_first,
            held_second - base_second,
            (held_second - held_first) - (base_second - base_first),
        ),
        axis=1,
    )
    return {
        "path": str(path.resolve()),
        "point": int(validated["point_index"]),
        "metadata": {"depth_mm": float(validated["depth_mm"])},
        "xy": np.asarray(validated["xy"], dtype=float),
        "fields": fields,
        "pair_transition_delta27": pair_transition,
    }


def pair_transition_features(row: dict, variant: str) -> np.ndarray:
    values = np.asarray(row["pair_transition_delta27"], dtype=float)
    if values.ndim != 2 or values.shape[1] != 81:
        raise ValueError("expected 81-D F27 pair-transition feature")
    unit = lambda matrix: matrix / np.maximum(
        np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12
    )
    if variant == "pair_transition_unit81":
        return unit(values)
    blocks = values.reshape(-1, 3, 27)
    if variant == "pair_transition_block_unit81":
        return (
            blocks / np.maximum(np.linalg.norm(blocks, axis=2, keepdims=True), 1e-12)
        ).reshape(-1, 81)
    if variant == "pair_transition_pair_unit54":
        return unit(blocks[:, :2, :].reshape(-1, 54))
    if variant == "pair_transition_per_fbg_unit81":
        grouped = blocks.reshape(-1, 3, 9, 3)
        return (
            grouped / np.maximum(np.linalg.norm(grouped, axis=3, keepdims=True), 1e-12)
        ).reshape(-1, 81)
    if variant == "pair_transition_signed_sqrt_unit81":
        return unit(np.sign(values) * np.sqrt(np.abs(values)))
    raise ValueError(variant)


def _candidate_models(sources, features):
    templates = (
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
            "extra_trees_leaf2_half",
            ExtraTreesClassifier(
                n_estimators=150,
                min_samples_leaf=2,
                max_features=0.5,
                class_weight="balanced",
                random_state=20260908,
                n_jobs=-1,
            ),
        ),
    )
    for source in sources:
        for feature in features:
            for estimator_name, template in templates:
                yield source, feature, estimator_name, "hard", template
                if estimator_name != "extra_trees_leaf2_half":
                    yield source, feature, estimator_name, "posterior_xy", template


def load_all_rounds(python_root: Path) -> dict[str, list[dict]]:
    repo_root = python_root.parent
    rounds = load_rounds(
        (repo_root / "outputs" / "fast_fullmap_position_rounds_0p30_20260906.json").resolve(),
        [
            (repo_root / "outputs" / "fast_fullmap_position_validation_r3_0p30_20260906.json").resolve(),
            python_root / "outputs" / "f45_position_validation_after_por_r1_20260908.json",
            python_root / "outputs" / "f45_position_validation_after_lock_r2_20260908.json",
        ],
    )
    visible = {
        "d010": "f45_multidepth_train_0p10_r1_20260908_060423/manifest.json",
        "d020": "f45_multidepth_validation_0p20_r1_20260908_0657/manifest.json",
        "d025": "f45_multidepth_final2_validation_0p25_r1_20260908_085951/manifest.json",
        "d035": "f45_position_v3_blind_0p35_r1_20260908/manifest.json",
        "d040": "f45_multidepth_final_validation_0p40_r1_20260908_080053/manifest.json",
        "d055": "f45_multidepth_final2_validation_0p55_r1_20260908_092016/manifest.json",
        "d060": "f45_multidepth_validation_0p60_r1_20260908_0708/manifest.json",
        "d065": "f45_position_v3_blind_0p65_r1_20260908/manifest.json",
        "d070": "f45_multidepth_final_validation_0p70_r1_20260908_081404/manifest.json",
        "d080": "f45_multidepth_train_0p80_r1_merged_20260908.json",
        "d055_repeat_edge": "f45_multidepth_diagnostic_0p55_p11_p17_postpor_20260908_095043/manifest.json",
    }
    for name, relative in visible.items():
        rounds[name] = _load_visible_round(python_root / "outputs" / relative)
    return {
        name: [load_f27_capture(Path(row["path"])) for row in records]
        for name, records in rounds.items()
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--pair-transition",
        action="store_true",
        help="evaluate the F45-style two-read plus transition feature on only 27 rows",
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output must be new")

    rounds = load_all_rounds(Path(__file__).resolve().parent)
    coordinates = {
        int(row["point"]): np.asarray(row["xy"], dtype=float)
        for row in rounds["old_r1"]
    }
    for name, rows in rounds.items():
        for row in rows:
            if np.linalg.norm(row["xy"] - coordinates[int(row["point"])]) > 0.05:
                raise ValueError(f"{name}/P{row['point']}: coordinate mismatch")

    sources = ("pair",) if args.pair_transition else READ_SOURCES
    features = (
        (
            "pair_transition_unit81",
            "pair_transition_block_unit81",
            "pair_transition_pair_unit54",
            "pair_transition_per_fbg_unit81",
            "pair_transition_signed_sqrt_unit81",
        )
        if args.pair_transition
        else FEATURES
    )
    cache = {
        (round_name, int(row["point"]), source, feature): (
            pair_transition_features(row, feature)
            if args.pair_transition
            else frame_features(row, source, feature)
        )
        for round_name, rows in rounds.items()
        for row in rows
        for source in sources
        for feature in features
    }
    labels = np.asarray(sorted(coordinates), dtype=int)
    label_xy = np.stack([coordinates[int(label)] for label in labels])
    ranked = []
    for source, feature, estimator_name, decoder, template in _candidate_models(sources, features):
        folds = []
        for held_name, held_rows in rounds.items():
            train_x = []
            train_y = []
            for round_name, rows in rounds.items():
                if round_name == held_name:
                    continue
                for row in rows:
                    point = int(row["point"])
                    values = cache[(round_name, point, source, feature)]
                    train_x.append(values)
                    train_y.append(np.full(len(values), point, dtype=int))
            model = clone(template).fit(np.concatenate(train_x), np.concatenate(train_y))
            frame_errors = []
            frame_exact = []
            trials = []
            for row in held_rows:
                point = int(row["point"])
                values = cache[(held_name, point, source, feature)]
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
                    "trial_error_maximum_mm": float(np.max(trial_errors)),
                    "frame_nearest_label_exact_percent": float(100.0 * np.mean(frame_exact)),
                    "frame_within_tool_radius_percent": float(
                        100.0 * np.mean(frame_errors_array <= TOOL_RADIUS_MM + 1e-6)
                    ),
                    "frame_error_mean_mm": float(np.mean(frame_errors_array)),
                    "frame_error_p95_mm": float(np.percentile(frame_errors_array, 95)),
                    "trials": trials,
                }
            )
        ranked.append(
            {
                "read_source": source,
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
        "schema": (
            "f27_subset_pair_transition_multidepth_position_evaluation_v1"
            if args.pair_transition
            else "f27_subset_multidepth_position_evaluation_v1"
        ),
        "selected_installed_f45_rows": CORE_ROWS.tolist(),
        "selected_rows_per_grating": 3,
        "source_capture_protocol": "F45",
        "candidate_runtime_protocol": "C27 not yet optically/timing qualified",
        "random_frame_split_used": False,
        "deployment_eligible": False,
        "scope": "Information-retention simulation only; fresh C27 route validation is mandatory.",
        "rounds": {
            name: {
                "depth_mm": float(np.median([row["metadata"]["depth_mm"] for row in rows])),
                "points": [int(row["point"]) for row in rows],
            }
            for name, rows in rounds.items()
        },
        "best": ranked[0],
        "ranked": ranked,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(
        json.dumps(
            [{key: value for key, value in row.items() if key != "folds"} for row in ranked[:15]],
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
