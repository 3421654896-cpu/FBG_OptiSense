"""Select a causal 1--3 frame CH1 position model with contact-level splits.

The source captures are slow static CNC contacts.  This utility measures
repeat-to-repeat *per-frame* localization only.  It does not certify a driven
15 Hz pressure transfer function, because no synchronized 15 Hz actuator or
pressure reference is present in these captures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from calibrate_fast_flank_position_model import (
    ContactSample,
    _adc_vector,
    _metrics,
    _resolve_default_inputs,
    _resolve_template_inputs,
    _resolve_validation_inputs,
    _sha256,
    load_contact_sample,
)


@dataclass(frozen=True)
class OnlineContact:
    sample: ContactSample
    features: dict[str, np.ndarray]


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    if window < 1 or window > 3:
        raise ValueError("online window must be 1..3 frames")
    result = []
    for index in range(values.shape[0]):
        start = max(0, index - window + 1)
        result.append(np.median(values[start : index + 1], axis=0))
    return np.asarray(result)


def _centred_family(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centred = values - np.median(values, axis=1, keepdims=True)
    norms = np.linalg.norm(centred, axis=1, keepdims=True)
    unit = centred / np.maximum(norms, 1e-9)
    return unit, np.log1p(norms)


def _online_features(
    log_ratio: np.ndarray,
    log_change: np.ndarray,
    symmetric_ratio_change: np.ndarray,
) -> dict[str, np.ndarray]:
    result = {}
    for window in (1, 2, 3):
        filtered = _rolling_median(log_ratio, window)
        unit, amplitude = _centred_family(filtered)
        result[f"log_ratio_unit_amp_w{window}"] = np.concatenate([unit, amplitude], axis=1)

        symmetric = _rolling_median(symmetric_ratio_change, window)
        symmetric_unit, symmetric_amplitude = _centred_family(symmetric)
        result[f"symmetric_ratio_unit_amp_w{window}"] = np.concatenate(
            [symmetric_unit, symmetric_amplitude], axis=1
        )

        raw_change = _rolling_median(log_change, window)
        raw_unit, raw_amplitude = _centred_family(raw_change)
        result[f"raw_log_unit_amp_w{window}"] = np.concatenate(
            [raw_unit, raw_amplitude], axis=1
        )

        pair_common = 0.5 * (raw_change[:, 0::2] + raw_change[:, 1::2])
        common_unit, common_amplitude = _centred_family(pair_common)
        result[f"ratio_common_unit_amp_w{window}"] = np.concatenate(
            [unit, common_unit, amplitude, common_amplitude], axis=1
        )
        result[f"symmetric_common_unit_amp_w{window}"] = np.concatenate(
            [symmetric_unit, common_unit, symmetric_amplitude, common_amplitude],
            axis=1,
        )
    return result


def load_online_contact(
    path: Path,
    *,
    repeat_override: int,
    expected_depth_mm: float = 0.0,
    required_commissioning_schema: str | None = None,
) -> OnlineContact:
    sample = load_contact_sample(
        path,
        repeat_override=repeat_override,
        expected_depth_mm=expected_depth_mm,
        required_commissioning_schema=required_commissioning_schema,
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    commissioning = data["commissioning"]
    trigger = int(commissioning.get("trigger_sequence", 96))
    baseline_adc = np.stack(
        [
            _adc_vector(frame)
            for frame in data["frames"]
            if trigger - 64 <= int(frame["sequence"]) < trigger
        ]
    )
    hold_adc = np.stack(
        [
            _adc_vector(frame)
            for frame in data["frames"]
            if sample.contact_sequence + 10
            <= int(frame["sequence"])
            <= sample.release_sequence - 10
        ]
    )
    baseline = np.median(baseline_adc, axis=0)
    left_baseline = np.maximum(baseline[0::2], 1.0)
    right_baseline = np.maximum(baseline[1::2], 1.0)
    baseline_ratio = np.log(right_baseline / left_baseline)
    left = np.maximum(hold_adc[:, 0::2], 1.0)
    right = np.maximum(hold_adc[:, 1::2], 1.0)
    log_ratio = np.log(right / left) - baseline_ratio
    log_change = np.log(np.maximum(hold_adc, 1.0) / np.maximum(baseline, 1.0))
    baseline_symmetric = (right_baseline - left_baseline) / np.maximum(
        right_baseline + left_baseline, 1.0
    )
    symmetric_ratio_change = (right - left) / np.maximum(right + left, 1.0)
    symmetric_ratio_change -= baseline_symmetric
    return OnlineContact(
        sample=sample,
        features=_online_features(log_ratio, log_change, symmetric_ratio_change),
    )


def _models(train_frames: int) -> dict[str, Any]:
    models = {}
    for alpha in (0.1, 1.0, 10.0, 100.0):
        models[f"ridge_a{alpha:g}"] = make_pipeline(
            StandardScaler(), Ridge(alpha=alpha)
        )
    for neighbors in (3, 7, 15, 31):
        if neighbors <= train_frames:
            models[f"knn_k{neighbors}"] = make_pipeline(
                StandardScaler(),
                KNeighborsRegressor(n_neighbors=neighbors, weights="distance", n_jobs=-1),
            )
    for leaf in (1, 3, 7, 15):
        models[f"extra_trees_leaf{leaf}"] = ExtraTreesRegressor(
            n_estimators=192,
            min_samples_leaf=leaf,
            max_features=1.0,
            random_state=1127,
            n_jobs=-1,
        )
        models[f"random_forest_leaf{leaf}"] = RandomForestRegressor(
            n_estimators=192,
            min_samples_leaf=leaf,
            max_features=1.0,
            random_state=1127,
            n_jobs=-1,
        )
    for cost in (1.0, 10.0, 100.0):
        for gamma_name, gamma in (("scale", "scale"), ("g0.03", 0.03), ("g0.1", 0.1)):
            models[f"svr_c{cost:g}_{gamma_name}"] = make_pipeline(
                StandardScaler(),
                MultiOutputRegressor(
                    SVR(C=cost, gamma=gamma, epsilon=0.1), n_jobs=-1
                ),
            )
    return models


def _training_rows(
    contacts: list[OnlineContact], feature: str, aggregation: str
) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    labels = []
    for contact in contacts:
        values = contact.features[feature]
        if aggregation == "frame":
            selected = values
        elif aggregation == "event_median":
            selected = np.median(values, axis=0, keepdims=True)
        elif aggregation == "event_quantiles":
            selected = np.quantile(values, (0.1, 0.3, 0.5, 0.7, 0.9), axis=0)
        else:
            raise ValueError(f"unsupported training aggregation: {aggregation}")
        rows.append(selected)
        labels.append(
            np.tile(
                [contact.sample.x_mm, contact.sample.y_mm],
                (selected.shape[0], 1),
            )
        )
    return np.concatenate(rows), np.concatenate(labels)


def _score_model(
    model,
    feature: str,
    train: list[OnlineContact],
    test: list[OnlineContact],
    *,
    training_aggregation: str = "frame",
):
    train_x, train_y = _training_rows(train, feature, training_aggregation)
    model.fit(train_x, train_y)
    all_predictions = []
    all_truth = []
    event_predictions = []
    event_truth = []
    event_points = []
    for contact in test:
        prediction = np.asarray(model.predict(contact.features[feature]), dtype=float)
        truth = np.tile(
            [contact.sample.x_mm, contact.sample.y_mm],
            (prediction.shape[0], 1),
        )
        all_predictions.append(prediction)
        all_truth.append(truth)
        event_predictions.append(np.median(prediction, axis=0))
        event_truth.append([contact.sample.x_mm, contact.sample.y_mm])
        event_points.append(contact.sample.point_index)
    frame_metrics = _metrics(np.concatenate(all_predictions), np.concatenate(all_truth))
    # Frame metrics can contain thousands of rows; keep aggregates and omit its
    # synthetic sequential point labels from the report.
    frame_metrics.pop("per_point", None)
    event_metrics = _metrics(np.asarray(event_predictions), np.asarray(event_truth))
    for row, point in zip(event_metrics["per_point"], event_points):
        row["point_index"] = int(point)
    return frame_metrics, event_metrics, model


def select_candidate(
    contacts: list[OnlineContact], *, training_aggregation: str = "frame",
    model_prefix: str = "",
):
    repeats = sorted({contact.sample.repeat for contact in contacts})
    if len(repeats) < 2:
        raise ValueError(f"at least two complete development repeats are required, got {repeats}")
    by_repeat = {
        repeat: sorted(
            [contact for contact in contacts if contact.sample.repeat == repeat],
            key=lambda contact: contact.sample.point_index,
        )
        for repeat in repeats
    }
    expected = list(range(1, 18))
    for repeat, values in by_repeat.items():
        if [value.sample.point_index for value in values] != expected:
            raise ValueError(f"repeat {repeat} is incomplete")

    rows = []
    for feature in sorted(contacts[0].features):
        for test_repeat in repeats:
            train_repeats = [repeat for repeat in repeats if repeat != test_repeat]
            train = [contact for repeat in train_repeats for contact in by_repeat[repeat]]
            test = by_repeat[test_repeat]
            train_x, _ = _training_rows(train, feature, training_aggregation)
            models = _models(train_x.shape[0])
            if model_prefix:
                models = {
                    name: model
                    for name, model in models.items()
                    if name.startswith(model_prefix)
                }
            if not models:
                raise ValueError(f"no models match prefix {model_prefix!r}")
            for model_name, model in models.items():
                frame_metrics, event_metrics, _ = _score_model(
                    model,
                    feature,
                    train,
                    test,
                    training_aggregation=training_aggregation,
                )
                event_metrics.pop("per_point", None)
                rows.append(
                    {
                        "feature": feature,
                        "model": model_name,
                        "training_aggregation": training_aggregation,
                        "train_repeats": train_repeats,
                        "test_repeat": test_repeat,
                        "frame_metrics": frame_metrics,
                        "event_metrics": event_metrics,
                    }
                )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["feature"], row["model"]), []).append(row)
    summaries = []
    for (feature, model), splits in grouped.items():
        if len(splits) != len(repeats):
            continue
        # Exact pooled percentiles cannot be reconstructed from aggregates;
        # use the worst held-out repeat to select conservatively.
        summaries.append(
            {
                "feature": feature,
                "model": model,
                "training_aggregation": training_aggregation,
                "worst_heldout_repeat_frame_p95_error_mm": max(
                    split["frame_metrics"]["p95_error_mm"] for split in splits
                ),
                "worst_heldout_repeat_frame_median_error_mm": max(
                    split["frame_metrics"]["median_error_mm"] for split in splits
                ),
                "worst_heldout_repeat_event_p95_error_mm": max(
                    split["event_metrics"]["p95_error_mm"] for split in splits
                ),
                "minimum_heldout_repeat_frame_within_4mm_percent": min(
                    split["frame_metrics"]["within_4mm_percent"] for split in splits
                ),
            }
        )
    summaries.sort(
        key=lambda row: (
            row["worst_heldout_repeat_frame_p95_error_mm"],
            row["worst_heldout_repeat_event_p95_error_mm"],
        )
    )
    return rows, summaries[0]


def _script_sha256() -> str:
    return _sha256(Path(__file__))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", type=Path, default=Path(__file__).with_name("outputs"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--freeze-candidate-output", type=Path)
    parser.add_argument("--locked-candidate", type=Path)
    parser.add_argument("--validation-repeat", type=int, default=5)
    parser.add_argument("--model-output", type=Path)
    parser.add_argument(
        "--expected-depth-mm",
        type=float,
        default=0.0,
        help="Required indentation for all development and validation contacts (0..0.8 mm).",
    )
    parser.add_argument(
        "--development-template",
        help="Explicit filename template containing {point:02d} and {repeat}.",
    )
    parser.add_argument(
        "--development-repeats",
        type=int,
        nargs="+",
        help="Development repeat IDs; requires --development-template.",
    )
    parser.add_argument(
        "--validation-template",
        help="Explicit validation filename template containing {point:02d} and {repeat}.",
    )
    parser.add_argument(
        "--required-commissioning-schema",
        help="Reject captures from a different physical acquisition protocol.",
    )
    parser.add_argument(
        "--training-aggregation",
        choices=("frame", "event_median", "event_quantiles"),
        default="frame",
        help="balance each physical press instead of weighting every hold frame equally",
    )
    parser.add_argument(
        "--model-prefix",
        default="",
        help="optional development-search filter; for example grid_",
    )
    args = parser.parse_args()

    if not np.isfinite(args.expected_depth_mm) or not 0.0 <= args.expected_depth_mm <= 0.8:
        raise SystemExit("--expected-depth-mm must be within 0..0.8 mm")
    if args.development_template:
        repeats = tuple(args.development_repeats or ())
        if len(repeats) < 2:
            raise SystemExit(
                "--development-template requires at least two --development-repeats"
            )
        development_inputs = _resolve_template_inputs(
            args.capture_directory,
            args.development_template,
            repeats,
        )
    else:
        if args.development_repeats:
            raise SystemExit("--development-repeats requires --development-template")
        development_inputs = (
            _resolve_default_inputs(args.capture_directory)
            + _resolve_validation_inputs(args.capture_directory, 3)
            + _resolve_validation_inputs(args.capture_directory, 4)
        )
    development = [
        load_online_contact(
            path,
            repeat_override=repeat,
            expected_depth_mm=args.expected_depth_mm,
            required_commissioning_schema=args.required_commissioning_schema,
        )
        for path, repeat in development_inputs
    ]
    lock = None
    if args.locked_candidate:
        lock = json.loads(args.locked_candidate.read_text(encoding="utf-8"))
        if lock.get("schema") != "fast_flank_online_position_candidate_lock_v1":
            raise SystemExit("wrong online candidate lock schema")
        if abs(float(lock.get("expected_depth_mm", 0.0)) - args.expected_depth_mm) > 1e-9:
            raise SystemExit("online candidate lock depth does not match --expected-depth-mm")
        if lock.get("required_commissioning_schema") != args.required_commissioning_schema:
            raise SystemExit("online candidate lock commissioning schema does not match")
        hashes = {contact.sample.path.name: contact.sample.sha256 for contact in development}
        if hashes != lock.get("development_capture_sha256"):
            raise SystemExit("development capture hashes changed after online lock")
        if _script_sha256() != lock.get("script_sha256"):
            raise SystemExit("online calibration code changed after lock")
        candidate = dict(lock["candidate"])
        development_results = []
    else:
        development_results, candidate = select_candidate(
            development,
            training_aggregation=args.training_aggregation,
            model_prefix=args.model_prefix,
        )

    if args.development_template and not args.validation_template:
        raise SystemExit("--validation-template is required with --development-template")
    validation_inputs = (
        _resolve_template_inputs(
            args.capture_directory,
            args.validation_template,
            (args.validation_repeat,),
        )
        if args.validation_template
        else _resolve_validation_inputs(args.capture_directory, args.validation_repeat)
    )
    if args.freeze_candidate_output:
        existing = [str(path) for path, _ in validation_inputs if path.exists()]
        if existing:
            raise SystemExit("reserved online validation files already exist:\n" + "\n".join(existing))
        lock_payload = {
            "schema": "fast_flank_online_position_candidate_lock_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script_sha256": _script_sha256(),
            "candidate": {
                "feature": candidate["feature"],
                "model": candidate["model"],
                "training_aggregation": candidate.get(
                    "training_aggregation", args.training_aggregation
                ),
            },
            "development_capture_sha256": {
                contact.sample.path.name: contact.sample.sha256 for contact in development
            },
            "validation_repeat_reserved": int(args.validation_repeat),
            "expected_depth_mm": float(args.expected_depth_mm),
            "development_template": args.development_template,
            "validation_template": args.validation_template,
            "required_commissioning_schema": args.required_commissioning_schema,
            "physical_15hz_verified": False,
        }
        args.freeze_candidate_output.parent.mkdir(parents=True, exist_ok=True)
        args.freeze_candidate_output.write_text(
            json.dumps(lock_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    validation = []
    validation_result = None
    if args.locked_candidate:
        missing = [str(path) for path, _ in validation_inputs if not path.exists()]
        if missing:
            raise SystemExit("missing reserved online validation files:\n" + "\n".join(missing))
        validation = [
            load_online_contact(
                path,
                repeat_override=repeat,
                expected_depth_mm=args.expected_depth_mm,
                required_commissioning_schema=args.required_commissioning_schema,
            )
            for path, repeat in validation_inputs
        ]
        feature = candidate["feature"]
        aggregation = candidate.get("training_aggregation", "frame")
        train_x, _ = _training_rows(development, feature, aggregation)
        model = _models(train_x.shape[0])[candidate["model"]]
        frame_metrics, event_metrics, fitted = _score_model(
            model,
            feature,
            development,
            validation,
            training_aggregation=aggregation,
        )
        validation_result = {
            "protocol": "Code/data/candidate locked before reserved repeat files existed.",
            "candidate_lock": str(args.locked_candidate.resolve()),
            "frame_metrics": frame_metrics,
            "event_median_metrics": event_metrics,
            "static_hold_only": True,
            "physical_15hz_verified": False,
        }
        if args.model_output:
            all_contacts = development + validation
            all_x, all_y = _training_rows(all_contacts, feature, aggregation)
            fitted = _models(all_x.shape[0])[candidate["model"]]
            fitted.fit(all_x, all_y)
            artifact = {
                "schema": "fast_flank_online_position_model_provisional_v1",
                "feature": feature,
                "model_name": candidate["model"],
                "training_aggregation": aggregation,
                "model": fitted,
                "baseline_frames": 64,
                "frame_window": int(feature.rsplit("w", 1)[1]),
                "expected_depth_mm": float(args.expected_depth_mm),
                "required_commissioning_schema": args.required_commissioning_schema,
                "capture_sha256": [contact.sample.sha256 for contact in all_contacts],
                "static_hold_repeat_accuracy": validation_result,
                "physical_15hz_verified": False,
                "contact_area_verified": False,
            }
            args.model_output.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(artifact, args.model_output)

    report = {
        "schema": "fast_flank_online_position_validation_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": candidate,
        "depth_mm": float(args.expected_depth_mm),
        "required_commissioning_schema": args.required_commissioning_schema,
        "development_contact_count": len(development),
        "development_results": development_results,
        "locked_validation": validation_result,
        "physical_15hz_verified": False,
        "contact_area_verified": False,
        "note": (
            "Per-frame accuracy on static holds is not a driven 15 Hz test. "
            "A synchronized pressure waveform and force/area ground truth remain required."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report.resolve()),
                "candidate": candidate,
                "locked_validation": validation_result,
                "physical_15hz_verified": False,
                "contact_area_verified": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
