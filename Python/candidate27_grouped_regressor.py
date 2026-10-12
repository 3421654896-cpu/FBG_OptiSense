"""Select a physical-XY C27 regressor using capture-grouped cross-validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from candidate27_grouped_classifier import (
    FEATURES, READ_SOURCES, _physical_xy, _validate_training, frame_features,
)
from capture_ch1_teacher_spectrum import now_text, save_checkpoint, sha256_file
from evaluate_candidate27_feature_variants import _capture


def build_regressor(kind: str):
    if kind.startswith("ridge_a"):
        model = Ridge(alpha=float(kind.removeprefix("ridge_a")))
        return Pipeline([("scale", StandardScaler()), ("regressor", model)])
    if kind.startswith("knn_k"):
        model = KNeighborsRegressor(
            n_neighbors=int(kind.removeprefix("knn_k")), weights="distance", p=2)
        return Pipeline([("scale", StandardScaler()), ("regressor", model)])
    if kind.startswith("extra_leaf"):
        model = ExtraTreesRegressor(
            n_estimators=240, min_samples_leaf=int(kind.removeprefix("extra_leaf")),
            max_features=1.0, random_state=20260907, n_jobs=-1)
        return Pipeline([("regressor", model)])
    if kind.startswith("svr_c"):
        model = MultiOutputRegressor(SVR(
            C=float(kind.removeprefix("svr_c")), gamma="scale", epsilon=0.05,
            kernel="rbf"))
        return Pipeline([("scale", StandardScaler()), ("regressor", model)])
    raise ValueError(kind)


def _fit_arrays(rows: list[dict], source: str, feature: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([frame_features(row, source, feature) for row in rows])
    y = np.concatenate([
        np.repeat(_physical_xy(row)[None, :], len(frame_features(row, source, feature)), axis=0)
        for row in rows
    ])
    return x, y


def _score(model, rows: list[dict], source: str, feature: str) -> dict:
    trial_errors = []
    frame_errors = []
    for row in rows:
        values = frame_features(row, source, feature)
        predicted = np.asarray(model.predict(values), dtype=float)
        truth = _physical_xy(row)
        frame_errors.extend(np.linalg.norm(predicted - truth, axis=1).tolist())
        trial_errors.append(float(np.linalg.norm(np.median(predicted, axis=0) - truth)))
    return {
        "trial_median_error_mm": float(np.median(trial_errors)),
        "trial_p95_error_mm": float(np.percentile(trial_errors, 95)),
        "trial_max_error_mm": float(np.max(trial_errors)),
        "frame_median_error_mm": float(np.median(frame_errors)),
        "frame_p95_error_mm": float(np.percentile(frame_errors, 95)),
    }


def select_and_train(paths: list[Path], model_path: Path, manifest_path: Path) -> dict:
    if model_path.exists() or manifest_path.exists():
        raise FileExistsError("Model and manifest outputs must be new")
    rows = [_capture(path) for path in paths]
    grouped = _validate_training(rows)
    regressors = (
        "ridge_a0.1", "ridge_a1", "ridge_a10", "ridge_a100",
        "knn_k1", "knn_k3", "knn_k5",
        "extra_leaf1", "extra_leaf4", "extra_leaf16",
        "svr_c1", "svr_c10", "svr_c100",
    )
    candidates = []
    for source in READ_SOURCES:
        for feature in FEATURES:
            for kind in regressors:
                folds = []
                for train_slot, validation_slot in ((0, 1), (1, 0)):
                    training_rows = [captures[train_slot] for captures in grouped.values()]
                    validation_rows = [captures[validation_slot] for captures in grouped.values()]
                    x_train, y_train = _fit_arrays(training_rows, source, feature)
                    model = build_regressor(kind)
                    model.fit(x_train, y_train)
                    folds.append({"train_slot": train_slot, **_score(model, validation_rows, source, feature)})
                candidates.append({
                    "read_source": source, "feature": feature, "regressor": kind,
                    "worst_trial_p95_error_mm": max(row["trial_p95_error_mm"] for row in folds),
                    "mean_trial_p95_error_mm": float(np.mean([row["trial_p95_error_mm"] for row in folds])),
                    "mean_trial_median_error_mm": float(np.mean([row["trial_median_error_mm"] for row in folds])),
                    "mean_frame_p95_error_mm": float(np.mean([row["frame_p95_error_mm"] for row in folds])),
                    "folds": folds,
                })
    candidates.sort(key=lambda row: (
        row["worst_trial_p95_error_mm"], row["mean_trial_p95_error_mm"],
        row["mean_trial_median_error_mm"], row["mean_frame_p95_error_mm"]
    ))
    selected = candidates[0]
    x_train, y_train = _fit_arrays(rows, selected["read_source"], selected["feature"])
    model = build_regressor(selected["regressor"])
    model.fit(x_train, y_train)
    joblib.dump(model, model_path)
    first = rows[0]["metadata"]
    manifest = {
        "schema": "candidate27_grouped_xy_regressor_v1", "created": now_text(),
        "model_file": str(model_path.resolve()), "model_sha256": sha256_file(model_path),
        "deployed": False, "stream_protocol": first["stream_protocol"],
        "route_crc32": first["route_crc32"], "first_delay_us": first["first_delay_us"],
        "spacing_us": first["spacing_us"], "depth_mm": first["depth_mm"],
        "read_source": selected["read_source"], "feature": selected["feature"],
        "regressor": selected["regressor"], "selection": selected,
        "top_cross_validation_candidates": candidates[:12],
        "training": [{"point": int(row["point"]), "source": row["path"],
                      "sha256": row["metadata"]["sha256"], "xy_mm": _physical_xy(row).tolist()}
                     for row in rows],
        "physical_15hz_verified": False, "full_spatial_accuracy_verified": False,
        "scope": "Static fixed-depth XY regressor; third-pass hold-out required before deployment.",
    }
    save_checkpoint(manifest_path, manifest)
    return manifest


def validate(manifest_path: Path, paths: list[Path], output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if manifest.get("schema") != "candidate27_grouped_xy_regressor_v1":
        raise ValueError("Unsupported manifest")
    model_path = Path(manifest["model_file"])
    if sha256_file(model_path) != manifest["model_sha256"]:
        raise ValueError("Model hash mismatch")
    model = joblib.load(model_path)
    training_hashes = {row["sha256"] for row in manifest["training"]}
    rows = [_capture(path) for path in paths]
    results = []
    for row in rows:
        if row["metadata"]["sha256"] in training_hashes:
            raise ValueError("Training capture cannot be used as hold-out")
        for key in ("stream_protocol", "route_crc32", "first_delay_us", "spacing_us", "depth_mm"):
            if row["metadata"][key] != manifest[key]:
                raise ValueError(f"Hold-out {key} mismatch")
        values = frame_features(row, manifest["read_source"], manifest["feature"])
        predicted = np.asarray(model.predict(values), dtype=float)
        truth = _physical_xy(row)
        frame_errors = np.linalg.norm(predicted - truth, axis=1)
        trial_xy = np.median(predicted, axis=0)
        results.append({
            "source": row["path"], "sha256": row["metadata"]["sha256"],
            "truth": int(row["point"]), "truth_xy_mm": truth.tolist(),
            "predicted_xy_mm": trial_xy.tolist(),
            "trial_position_error_mm": float(np.linalg.norm(trial_xy - truth)),
            "held_frames": len(predicted),
            "frame_median_position_error_mm": float(np.median(frame_errors)),
            "frame_p95_position_error_mm": float(np.percentile(frame_errors, 95)),
        })
    trial_errors = [row["trial_position_error_mm"] for row in results]
    frame_p95 = [row["frame_p95_position_error_mm"] for row in results]
    report = {
        "schema": "candidate27_grouped_xy_holdout_v1", "created": now_text(),
        "manifest_sha256": sha256_file(manifest_path), "validation": results,
        "summary": {
            "trial_count": len(results),
            "trial_median_error_mm": float(np.median(trial_errors)),
            "trial_p95_error_mm": float(np.percentile(trial_errors, 95)),
            "trial_max_error_mm": float(np.max(trial_errors)),
            "captures_within_2mm": int(np.count_nonzero(np.asarray(trial_errors) <= 2.0)),
            "captures_within_4mm": int(np.count_nonzero(np.asarray(trial_errors) <= 4.0)),
            "worst_capture_frame_p95_error_mm": float(np.max(frame_p95)),
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
