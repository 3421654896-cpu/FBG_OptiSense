"""Select a C27 position classifier with capture-grouped cross-validation.

Adjacent frames from one physical press are correlated, so a complete capture
is always kept in one fold.  With two captures per position, fold A trains on
the first capture of every position and validates the second; fold B reverses
that assignment.  A third, disjoint pass can then be used exactly once for the
reported hold-out result.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from capture_ch1_teacher_spectrum import now_text, save_checkpoint, sha256_file
from evaluate_candidate27_feature_variants import _capture


READ_SOURCES = ("first_code", "second_code", "mean_reads", "concat_reads")
FEATURES = ("legacy_shape_delta", "flank_shift", "delta_plus_energy", "delta_unit_global")


def frame_features(row: dict, read_source: str, feature: str) -> np.ndarray:
    first = row["fields"]["first_code"][feature]
    second = row["fields"]["second_code"][feature]
    if read_source == "first_code":
        return first
    if read_source == "second_code":
        return second
    if read_source == "mean_reads":
        return (first + second) / 2.0
    if read_source == "concat_reads":
        return np.concatenate([first, second], axis=1)
    raise ValueError(read_source)


def build_estimator(kind: str) -> Pipeline:
    if kind == "lda_shrinkage":
        estimator = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    elif kind.startswith("logistic_c"):
        c_value = float(kind.removeprefix("logistic_c"))
        estimator = LogisticRegression(C=c_value, max_iter=4000, solver="lbfgs")
    elif kind.startswith("rbf_c"):
        c_value = float(kind.removeprefix("rbf_c"))
        estimator = SVC(C=c_value, gamma="scale", kernel="rbf")
    else:
        raise ValueError(kind)
    return Pipeline([("scale", StandardScaler()), ("classifier", estimator)])


def _majority(predictions: np.ndarray) -> int:
    counts = Counter(map(int, predictions))
    return min(counts, key=lambda label: (-counts[label], label))


def _physical_xy(row: dict) -> np.ndarray:
    target = row["target"]
    if "calibration_work_xy_mm" in target:
        return np.asarray(target["calibration_work_xy_mm"], dtype=float)
    return np.asarray([target["work_x"], target["work_y"]], dtype=float)


def _validate_training(rows: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["point"])].append(row)
    if len(grouped) < 2 or any(len(captures) != 2 for captures in grouped.values()):
        raise ValueError("Exactly two independent captures per position are required")
    contracts = {
        (
            row["metadata"]["route_crc32"], row["metadata"]["first_delay_us"],
            row["metadata"]["spacing_us"], row["metadata"]["depth_mm"],
            row["metadata"]["stream_protocol"],
        )
        for row in rows
    }
    if len(contracts) != 1 or next(iter(contracts))[-1] != "candidate27_guarded":
        raise ValueError("Training route/timing/depth/protocol mismatch")
    return dict(sorted(grouped.items()))


def select_and_train(paths: list[Path], model_path: Path, manifest_path: Path) -> dict:
    if model_path.exists() or manifest_path.exists():
        raise FileExistsError("Model and manifest outputs must be new")
    rows = [_capture(path) for path in paths]
    grouped = _validate_training(rows)
    estimators = (
        "lda_shrinkage", "logistic_c0.1", "logistic_c1", "logistic_c10",
        "rbf_c1", "rbf_c10", "rbf_c100",
    )
    candidates = []
    for read_source in READ_SOURCES:
        for feature in FEATURES:
            for estimator_kind in estimators:
                fold_results = []
                for train_slot, validation_slot in ((0, 1), (1, 0)):
                    training_rows = [captures[train_slot] for captures in grouped.values()]
                    validation_rows = [captures[validation_slot] for captures in grouped.values()]
                    x_train = np.concatenate([
                        frame_features(row, read_source, feature) for row in training_rows
                    ])
                    y_train = np.concatenate([
                        np.full(len(frame_features(row, read_source, feature)), int(row["point"]))
                        for row in training_rows
                    ])
                    model = build_estimator(estimator_kind)
                    model.fit(x_train, y_train)
                    trial_correct = frame_correct = frame_count = 0
                    for row in validation_rows:
                        values = frame_features(row, read_source, feature)
                        predictions = model.predict(values)
                        truth = int(row["point"])
                        frame_correct += int(np.count_nonzero(predictions == truth))
                        frame_count += len(predictions)
                        trial_correct += int(_majority(predictions) == truth)
                    fold_results.append({
                        "train_slot": train_slot,
                        "trial_accuracy": trial_correct / len(validation_rows),
                        "frame_accuracy": frame_correct / frame_count,
                    })
                candidates.append({
                    "read_source": read_source,
                    "feature": feature,
                    "estimator": estimator_kind,
                    "mean_trial_accuracy": float(np.mean([row["trial_accuracy"] for row in fold_results])),
                    "worst_trial_accuracy": float(np.min([row["trial_accuracy"] for row in fold_results])),
                    "mean_frame_accuracy": float(np.mean([row["frame_accuracy"] for row in fold_results])),
                    "folds": fold_results,
                })
    candidates.sort(key=lambda row: (
        row["worst_trial_accuracy"], row["mean_trial_accuracy"], row["mean_frame_accuracy"]
    ), reverse=True)
    selected = candidates[0]
    x_train = np.concatenate([
        frame_features(row, selected["read_source"], selected["feature"]) for row in rows
    ])
    y_train = np.concatenate([
        np.full(len(frame_features(row, selected["read_source"], selected["feature"])), int(row["point"]))
        for row in rows
    ])
    model = build_estimator(selected["estimator"])
    model.fit(x_train, y_train)
    joblib.dump(model, model_path)
    first = rows[0]["metadata"]
    manifest = {
        "schema": "candidate27_grouped_position_model_v1",
        "created": now_text(),
        "model_file": str(model_path.resolve()),
        "model_sha256": sha256_file(model_path),
        "deployed": False,
        "stream_protocol": first["stream_protocol"],
        "route_crc32": first["route_crc32"],
        "first_delay_us": first["first_delay_us"],
        "spacing_us": first["spacing_us"],
        "depth_mm": first["depth_mm"],
        "read_source": selected["read_source"],
        "feature": selected["feature"],
        "estimator": selected["estimator"],
        "selection": selected,
        "top_cross_validation_candidates": candidates[:12],
        "training": [
            {"point": int(row["point"]), "source": row["path"],
             "sha256": row["metadata"]["sha256"], "xy_mm": _physical_xy(row).tolist()}
            for row in rows
        ],
        "physical_15hz_verified": False,
        "full_spatial_accuracy_verified": False,
        "scope": "Static fixed-depth 17-position model; third-pass hold-out required before deployment.",
    }
    save_checkpoint(manifest_path, manifest)
    return manifest


def validate(manifest_path: Path, paths: list[Path], output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if manifest.get("schema") != "candidate27_grouped_position_model_v1":
        raise ValueError("Unsupported manifest")
    model_path = Path(manifest["model_file"])
    if sha256_file(model_path) != manifest["model_sha256"]:
        raise ValueError("Model file hash mismatch")
    model = joblib.load(model_path)
    training_hashes = {row["sha256"] for row in manifest["training"]}
    xy_by_point = {
        int(row["point"]): np.asarray(row["xy_mm"], dtype=float)
        for row in manifest["training"]
    }
    rows = [_capture(path) for path in paths]
    results = []
    for row in rows:
        if row["metadata"]["sha256"] in training_hashes:
            raise ValueError("Training capture cannot be used as hold-out")
        for key in ("stream_protocol", "route_crc32", "first_delay_us", "spacing_us", "depth_mm"):
            if row["metadata"][key] != manifest[key]:
                raise ValueError(f"Hold-out {key} mismatch")
        values = frame_features(row, manifest["read_source"], manifest["feature"])
        predictions = model.predict(values)
        truth = int(row["point"])
        predicted = _majority(predictions)
        frame_errors = [float(np.linalg.norm(xy_by_point[int(value)] - _physical_xy(row)))
                        for value in predictions]
        results.append({
            "source": row["path"], "sha256": row["metadata"]["sha256"],
            "truth": truth, "trial_prediction": predicted,
            "trial_position_error_mm": float(np.linalg.norm(xy_by_point[predicted] - _physical_xy(row))),
            "held_frames": len(predictions),
            "correct_frames": int(np.count_nonzero(predictions == truth)),
            "frame_confusion": dict(Counter(map(str, map(int, predictions)))),
            "frame_median_position_error_mm": float(np.median(frame_errors)),
            "frame_p95_position_error_mm": float(np.percentile(frame_errors, 95)),
        })
    frame_count = sum(row["held_frames"] for row in results)
    correct_frames = sum(row["correct_frames"] for row in results)
    report = {
        "schema": "candidate27_grouped_position_holdout_v1",
        "created": now_text(), "manifest_sha256": sha256_file(manifest_path),
        "validation": results,
        "summary": {
            "trial_count": len(results),
            "trial_correct": sum(row["trial_prediction"] == row["truth"] for row in results),
            "trial_accuracy": sum(row["trial_prediction"] == row["truth"] for row in results) / len(results),
            "trial_median_position_error_mm": float(np.median([row["trial_position_error_mm"] for row in results])),
            "trial_p95_position_error_mm": float(np.percentile([row["trial_position_error_mm"] for row in results], 95)),
            "frame_count": frame_count, "correct_frames": correct_frames,
            "frame_accuracy": correct_frames / frame_count,
            "frame_p95_position_error_mm": float(np.percentile(
                [row["frame_p95_position_error_mm"] for row in results], 95)),
        },
        "deployed": False, "physical_15hz_verified": False,
        "full_spatial_accuracy_verified": False,
        "scope": "One independent static third pass; adjacent held frames are correlated.",
    }
    save_checkpoint(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "validate"))
    parser.add_argument("captures", type=Path, nargs="+")
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "train":
        if args.model_output is None or args.manifest_output is None:
            parser.error("train requires --model-output and --manifest-output")
        report = select_and_train(args.captures, args.model_output, args.manifest_output)
        print(json.dumps(report["selection"], indent=2))
    else:
        if args.manifest is None or args.output is None:
            parser.error("validate requires --manifest and --output")
        report = validate(args.manifest, args.captures, args.output)
        print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
