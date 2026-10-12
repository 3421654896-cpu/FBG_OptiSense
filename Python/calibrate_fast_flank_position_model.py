"""Evaluate repeat-to-repeat CH1 fast-flank position calibration.

Each capture is one physical contact and therefore one statistical group.  Frames
from the same contact are never split between train and test.  The report is a
commissioning result, not an independent accuracy certificate: the same two
repeats are used to compare feature/model candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


POINT_RE = re.compile(r"_p(?P<point>\d{2})_.*_r(?P<repeat>\d+)_")
# Both the legacy 0 mm writer (``contact_reached_depth_*``) and the bounded
# indentation writer (``contact_reached_commanded_depth``) use this stem.
CONTACT_EVENT_PREFIX = "contact_reached"
RELEASE_EVENT = "release_started"
SAFE_EVENT = "final_safe_state_confirmed"
DESCENT_EVENT = "contact_descent_started"
EXPECTED_GRATINGS = 9
EXPECTED_FLANKS = EXPECTED_GRATINGS * 2


class CaptureRejected(ValueError):
    """Raised when a capture cannot be used as a labelled contact."""


@dataclass(frozen=True)
class ContactSample:
    path: Path
    point_index: int
    repeat: int
    x_mm: float
    y_mm: float
    contact_sequence: int
    release_sequence: int
    hold_frames: int
    features: dict[str, np.ndarray]
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nearest_sequence(receipts: list[dict[str, Any]], host_ns: int) -> int:
    if not receipts:
        raise CaptureRejected("missing frame receipts")
    nearest = min(receipts, key=lambda item: abs(int(item["host_monotonic_ns"]) - host_ns))
    return int(nearest["sequence"])


def _event(events: Iterable[dict[str, Any]], *, exact: str | None = None, prefix: str | None = None):
    matches = []
    for event in events:
        kind = str(event.get("kind", ""))
        if (exact is not None and kind == exact) or (prefix is not None and kind.startswith(prefix)):
            matches.append(event)
    if len(matches) != 1:
        wanted = exact if exact is not None else f"{prefix}*"
        raise CaptureRejected(f"expected exactly one {wanted} event, got {len(matches)}")
    return matches[0]


def _robust_vector(values: np.ndarray) -> np.ndarray:
    """Return a stable signed response signature without assuming sensor order."""
    median = np.median(values, axis=0)
    return median - np.median(median)


def _unit(values: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    return values / max(norm, 1e-9)


def _feature_families(hold: np.ndarray) -> dict[str, np.ndarray]:
    if hold.ndim != 2 or hold.shape[1] != EXPECTED_GRATINGS:
        raise CaptureRejected(f"expected Nx{EXPECTED_GRATINGS} hold values, got {hold.shape}")
    if hold.shape[0] < 12:
        raise CaptureRejected(f"hold window too short: {hold.shape[0]} frames")

    settled = _robust_vector(hold)
    thirds = np.array_split(hold, 3)
    temporal = np.concatenate([_robust_vector(part) for part in thirds])
    q25 = np.quantile(hold, 0.25, axis=0)
    q75 = np.quantile(hold, 0.75, axis=0)
    spread = q75 - q25
    signed_peak = np.take_along_axis(
        hold,
        np.argmax(np.abs(hold), axis=0, keepdims=True),
        axis=0,
    )[0]
    signed_peak = signed_peak - np.median(signed_peak)
    robust = np.concatenate([settled, spread, signed_peak])
    amplitude = np.array([math.log1p(float(np.linalg.norm(settled)))])
    return {
        "settled_centered": settled,
        "settled_unit": _unit(settled),
        "temporal_centered": temporal,
        "temporal_unit": _unit(temporal),
        "robust_shape": robust,
        "robust_shape_unit": _unit(robust),
        "shape_plus_amplitude": np.concatenate([_unit(robust), amplitude]),
    }


def _raw_adc_feature_families(
    baseline_adc: np.ndarray,
    hold_adc: np.ndarray,
    apparent_features: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if (
        baseline_adc.ndim != 2
        or hold_adc.ndim != 2
        or baseline_adc.shape[1] != EXPECTED_FLANKS
        or hold_adc.shape[1] != EXPECTED_FLANKS
    ):
        raise CaptureRejected("raw ADC features require two Nx18 matrices")
    baseline = np.median(baseline_adc, axis=0)
    hold = np.median(hold_adc, axis=0)
    delta = hold - baseline
    delta_centered = delta - np.median(delta)
    fractional = delta / np.maximum(np.abs(baseline), 8.0)
    fractional -= np.median(fractional)

    left_baseline = np.maximum(baseline[0::2], 1.0)
    right_baseline = np.maximum(baseline[1::2], 1.0)
    left_hold = np.maximum(hold[0::2], 1.0)
    right_hold = np.maximum(hold[1::2], 1.0)
    log_ratio = np.log(right_hold / left_hold) - np.log(right_baseline / left_baseline)
    log_ratio -= np.median(log_ratio)
    baseline_contrast = (right_baseline - left_baseline) / (right_baseline + left_baseline)
    hold_contrast = (right_hold - left_hold) / (right_hold + left_hold)
    contrast = hold_contrast - baseline_contrast
    contrast -= np.median(contrast)

    temporal_parts = []
    for part in np.array_split(hold_adc, 3):
        part_median = np.median(part, axis=0)
        part_fractional = (part_median - baseline) / np.maximum(np.abs(baseline), 8.0)
        temporal_parts.append(part_fractional - np.median(part_fractional))
    temporal_fractional = np.concatenate(temporal_parts)

    apparent_unit = apparent_features["settled_unit"]
    return {
        "adc_delta_centered": delta_centered,
        "adc_delta_unit": _unit(delta_centered),
        "adc_fractional": fractional,
        "adc_fractional_unit": _unit(fractional),
        "adc_log_ratio": log_ratio,
        "adc_log_ratio_unit": _unit(log_ratio),
        "adc_pair_contrast": contrast,
        "adc_pair_contrast_unit": _unit(contrast),
        "adc_temporal_fractional": temporal_fractional,
        "adc_temporal_fractional_unit": _unit(temporal_fractional),
        "adc_apparent_combined_unit": np.concatenate(
            [_unit(fractional), _unit(log_ratio), apparent_unit]
        ),
    }


def _signed_peak(matrix: np.ndarray) -> np.ndarray:
    return np.take_along_axis(
        matrix,
        np.argmax(np.abs(matrix), axis=0, keepdims=True),
        axis=0,
    )[0]


def _ramp_feature_families(
    baseline_adc: np.ndarray,
    ramp_adc: np.ndarray,
    ramp_apparent: np.ndarray,
) -> dict[str, np.ndarray]:
    """Describe the repeatable final 0.5 mm approach, not just the held state."""
    if ramp_adc.shape[0] < 12 or ramp_apparent.shape[0] != ramp_adc.shape[0]:
        raise CaptureRejected("contact ramp requires at least 12 aligned frames")
    baseline = np.median(baseline_adc, axis=0)
    left_baseline = np.maximum(baseline[0::2], 1.0)
    right_baseline = np.maximum(baseline[1::2], 1.0)
    baseline_log_ratio = np.log(right_baseline / left_baseline)

    left = np.maximum(ramp_adc[:, 0::2], 1.0)
    right = np.maximum(ramp_adc[:, 1::2], 1.0)
    log_ratio_series = np.log(right / left) - baseline_log_ratio
    log_ratio_peak = _signed_peak(log_ratio_series)
    log_ratio_peak -= np.median(log_ratio_peak)
    log_ratio_end = np.median(log_ratio_series[-5:], axis=0)
    log_ratio_end -= np.median(log_ratio_end)
    log_ratio_area = np.median(log_ratio_series, axis=0)
    log_ratio_area -= np.median(log_ratio_area)

    apparent_peak = _signed_peak(ramp_apparent)
    apparent_peak -= np.median(apparent_peak)
    apparent_end = np.median(ramp_apparent[-5:], axis=0)
    apparent_end -= np.median(apparent_end)
    combined = np.concatenate(
        [
            _unit(log_ratio_peak),
            _unit(log_ratio_end),
            _unit(log_ratio_area),
            _unit(apparent_peak),
            _unit(apparent_end),
        ]
    )
    return {
        "ramp_adc_log_ratio_peak": log_ratio_peak,
        "ramp_adc_log_ratio_peak_unit": _unit(log_ratio_peak),
        "ramp_adc_log_ratio_end": log_ratio_end,
        "ramp_adc_log_ratio_end_unit": _unit(log_ratio_end),
        "ramp_apparent_peak": apparent_peak,
        "ramp_apparent_peak_unit": _unit(apparent_peak),
        "ramp_combined_unit": combined,
    }


def _adc_vector(frame: dict[str, Any]) -> np.ndarray:
    records = frame.get("records") or []
    if len(records) != EXPECTED_FLANKS:
        raise CaptureRejected(
            f"expected {EXPECTED_FLANKS} flank records, got {len(records)}"
        )
    return np.asarray([int(record["second_code"]) for record in records], dtype=float)


def load_contact_sample(
    path: Path,
    *,
    repeat_override: int | None = None,
    expected_depth_mm: float = 0.0,
    required_commissioning_schema: str | None = None,
) -> ContactSample:
    data = json.loads(path.read_text(encoding="utf-8"))
    commissioning = data.get("commissioning") or {}
    if (
        required_commissioning_schema is not None
        and commissioning.get("schema") != required_commissioning_schema
    ):
        raise CaptureRejected(
            f"commissioning schema {commissioning.get('schema')!r} does not match "
            f"{required_commissioning_schema!r}"
        )
    target = commissioning.get("target") or {}
    match = POINT_RE.search(path.name)
    point_index = int(target.get("point_index", match.group("point") if match else 0))
    repeat = int(repeat_override if repeat_override is not None else (match.group("repeat") if match else 0))
    if not 1 <= point_index <= 17:
        raise CaptureRejected(f"invalid point index {point_index}")
    if repeat < 1:
        raise CaptureRejected(f"invalid repeat {repeat}")
    if not bool(data.get("complete")) or not bool(commissioning.get("training_eligible", False)):
        # The guarded commissioning writer deliberately marks single captures as
        # ineligible for final training.  For this grouped evaluator, accept only
        # complete captures whose remaining hard safety evidence is intact.
        if not bool(data.get("complete")):
            raise CaptureRejected("capture is incomplete")
    if commissioning.get("motion_failures"):
        raise CaptureRejected("capture contains motion failures")
    if commissioning.get("capture_error"):
        raise CaptureRejected(f"capture error: {commissioning['capture_error']}")
    expected_depth_mm = float(expected_depth_mm)
    if not math.isfinite(expected_depth_mm) or not 0.0 <= expected_depth_mm <= 0.8:
        raise CaptureRejected(f"invalid expected depth {expected_depth_mm!r} mm")
    actual_depth_mm = float(
        commissioning.get("depth_mm", target.get("depth_mm", math.nan))
    )
    target_depth_mm = float(target.get("depth_mm", actual_depth_mm))
    if (
        not math.isfinite(actual_depth_mm)
        or not math.isfinite(target_depth_mm)
        or abs(actual_depth_mm - expected_depth_mm) > 1e-9
        or abs(target_depth_mm - expected_depth_mm) > 1e-9
    ):
        raise CaptureRejected(
            f"capture depth {actual_depth_mm!r}/{target_depth_mm!r} mm does not match "
            f"the requested {expected_depth_mm:.6f} mm"
        )
    cleanup = data.get("safety_cleanup") or {}
    if cleanup.get("exact_disarm_ack") is not True or cleanup.get("exact_soa_shutter_ack") is not True:
        raise CaptureRejected("missing exact laser disarm/shutter acknowledgement")

    frames = data.get("frames") or []
    receipts = commissioning.get("frame_receipts") or []
    results = commissioning.get("realtime_results") or []
    expected_cycles = int(data.get("cycles_requested", -1))
    if not 640 <= expected_cycles <= 1024:
        raise CaptureRejected(f"invalid bounded capture length {expected_cycles}")
    expected_sequences = list(range(expected_cycles))
    for label, values in (
        ("raw frame", frames),
        ("frame receipt", receipts),
        ("realtime result", results),
    ):
        sequences = [int(value.get("sequence", -1)) for value in values]
        if sequences != expected_sequences:
            raise CaptureRejected(
                f"{label} sequence is incomplete, duplicated, reset or reordered"
            )
    analysis = data.get("analysis") or {}
    if (
        analysis.get("meets_sampling_rate_for_15hz") is not True
        or float(analysis.get("board_interval_max_us", math.inf))
        > 1_000_000.0 / 30.0
    ):
        raise CaptureRejected("capture does not pass the 15 Hz sampling-rate gate")

    events = commissioning.get("events") or []
    contact_event = _event(events, prefix=CONTACT_EVENT_PREFIX)
    descent_event = _event(events, exact=DESCENT_EVENT)
    release_event = _event(events, exact=RELEASE_EVENT)
    final_event = _event(events, exact=SAFE_EVENT)
    contact_status = contact_event.get("status") or {}
    press_z = float(target.get("press_z", float(target.get("contact_z", 0.0)) - expected_depth_mm))
    if (
        abs(float(contact_status.get("work_x", math.nan)) - float(target["work_x"])) > 0.02
        or abs(float(contact_status.get("work_y", math.nan)) - float(target["work_y"])) > 0.02
        or abs(float(contact_status.get("work_z", math.nan)) - press_z) > 0.02
        or contact_status.get("moving") is not False
        or contact_status.get("stopped") is not True
        or contact_status.get("estop") is not False
    ):
        raise CaptureRejected("contact event did not reach the labelled commanded position")
    final_status = final_event.get("status") or {}
    if (
        final_status.get("connected") is not True
        or final_status.get("moving") is not False
        or final_status.get("stopped") is not True
        or final_status.get("estop") is not False
        or abs(float(final_status.get("work_z", math.nan)) - float(target.get("safe_z", 4.0))) > 0.02
    ):
        raise CaptureRejected("final machine state is not the confirmed safe state")

    contact_sequence = _nearest_sequence(receipts, int(contact_event["host_monotonic_ns"]))
    descent_sequence = _nearest_sequence(receipts, int(descent_event["host_monotonic_ns"]))
    release_sequence = _nearest_sequence(receipts, int(release_event["host_monotonic_ns"]))
    start = contact_sequence + 10
    stop = release_sequence - 10
    hold_rows = [
        result.get("relative_shift_pm")
        for result in results
        if start <= int(result.get("sequence", -1)) <= stop
    ]
    if not hold_rows:
        raise CaptureRejected(f"no realtime frames in settled hold window {start}:{stop}")
    hold = np.asarray(hold_rows, dtype=float)
    if not np.all(np.isfinite(hold)):
        raise CaptureRejected("non-finite relative-shift value")

    trigger_sequence = int(commissioning.get("trigger_sequence", 96))
    raw_baseline = np.stack(
        [
            _adc_vector(frame)
            for frame in frames
            if trigger_sequence - 64 <= int(frame.get("sequence", -1)) < trigger_sequence
        ]
    )
    raw_hold = np.stack(
        [
            _adc_vector(frame)
            for frame in frames
            if start <= int(frame.get("sequence", -1)) <= stop
        ]
    )
    raw_ramp = np.stack(
        [
            _adc_vector(frame)
            for frame in frames
            if descent_sequence <= int(frame.get("sequence", -1)) <= contact_sequence + 10
        ]
    )
    ramp_apparent = np.asarray(
        [
            result["relative_shift_pm"]
            for result in results
            if descent_sequence <= int(result.get("sequence", -1)) <= contact_sequence + 10
        ],
        dtype=float,
    )
    if raw_baseline.shape[0] != 64 or raw_hold.shape[0] != hold.shape[0]:
        raise CaptureRejected(
            f"raw ADC window mismatch: baseline={raw_baseline.shape}, hold={raw_hold.shape}"
        )
    apparent_features = _feature_families(hold)
    features = {
        **apparent_features,
        **_raw_adc_feature_families(raw_baseline, raw_hold, apparent_features),
        **_ramp_feature_families(raw_baseline, raw_ramp, ramp_apparent),
    }

    return ContactSample(
        path=path,
        point_index=point_index,
        repeat=repeat,
        x_mm=float(target["work_x"]),
        y_mm=float(target["work_y"]),
        contact_sequence=contact_sequence,
        release_sequence=release_sequence,
        hold_frames=int(hold.shape[0]),
        features=features,
        sha256=_sha256(path),
    )


def _model_candidates(train_size: int):
    models: dict[str, Any] = {}
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
        models[f"ridge_a{alpha:g}"] = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    for components in range(1, min(6, train_size - 1) + 1):
        models[f"pls_c{components}"] = make_pipeline(
            StandardScaler(),
            PLSRegression(n_components=components, scale=False, max_iter=1000),
        )
    for neighbors in range(1, min(5, train_size) + 1):
        models[f"knn_k{neighbors}"] = make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=neighbors, weights="distance"),
        )
    models["extra_trees_leaf1"] = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=1,
        max_features=1.0,
        random_state=619,
        n_jobs=-1,
    )
    models["extra_trees_leaf2"] = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        max_features=1.0,
        random_state=619,
        n_jobs=-1,
    )
    models["random_forest_leaf1"] = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=1,
        max_features=1.0,
        random_state=619,
        n_jobs=-1,
    )
    return models


def _metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    errors = np.linalg.norm(predicted - truth, axis=1)
    return {
        "count": int(errors.size),
        "median_error_mm": float(np.median(errors)),
        "mean_error_mm": float(np.mean(errors)),
        "p95_error_mm": float(np.quantile(errors, 0.95)),
        "maximum_error_mm": float(np.max(errors)),
        "within_4mm_percent": float(100.0 * np.mean(errors <= 4.0)),
        "per_point": [
            {
                "point_index": int(index),
                "predicted_xy_mm": [float(value) for value in prediction],
                "true_xy_mm": [float(value) for value in actual],
                "error_mm": float(error),
            }
            for index, prediction, actual, error in zip(
                range(1, errors.size + 1), predicted, truth, errors
            )
        ],
    }


def evaluate_repeat_splits(samples: list[ContactSample]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repeats = sorted({sample.repeat for sample in samples})
    if len(repeats) != 2:
        raise ValueError(f"exactly two balanced repeats are required, got {repeats}")
    by_repeat = {repeat: sorted((s for s in samples if s.repeat == repeat), key=lambda s: s.point_index) for repeat in repeats}
    expected = list(range(1, 18))
    for repeat, values in by_repeat.items():
        points = [sample.point_index for sample in values]
        if points != expected:
            raise ValueError(f"repeat {repeat} is not a complete 17-point set: {points}")

    results = []
    feature_names = sorted(samples[0].features)
    for feature_name in feature_names:
        for train_repeat, test_repeat in ((repeats[0], repeats[1]), (repeats[1], repeats[0])):
            train = by_repeat[train_repeat]
            test = by_repeat[test_repeat]
            train_x = np.stack([sample.features[feature_name] for sample in train])
            test_x = np.stack([sample.features[feature_name] for sample in test])
            train_y = np.asarray([[sample.x_mm, sample.y_mm] for sample in train], dtype=float)
            test_y = np.asarray([[sample.x_mm, sample.y_mm] for sample in test], dtype=float)
            for model_name, model in _model_candidates(len(train)).items():
                try:
                    model.fit(train_x, train_y)
                    prediction = np.asarray(model.predict(test_x), dtype=float)
                    metrics = _metrics(prediction, test_y)
                except Exception as exc:  # keep the audit complete when one candidate is singular
                    results.append({
                        "feature": feature_name,
                        "model": model_name,
                        "train_repeat": train_repeat,
                        "test_repeat": test_repeat,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    continue
                results.append({
                    "feature": feature_name,
                    "model": model_name,
                    "train_repeat": train_repeat,
                    "test_repeat": test_repeat,
                    "metrics": metrics,
                })

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for result in results:
        if "metrics" in result:
            grouped.setdefault((result["feature"], result["model"]), []).append(result["metrics"])
    summaries = []
    for (feature, model), split_metrics in grouped.items():
        if len(split_metrics) != 2:
            continue
        all_errors = np.asarray([
            point["error_mm"]
            for metrics in split_metrics
            for point in metrics["per_point"]
        ])
        summaries.append({
            "feature": feature,
            "model": model,
            "two_way_count": int(all_errors.size),
            "two_way_median_error_mm": float(np.median(all_errors)),
            "two_way_mean_error_mm": float(np.mean(all_errors)),
            "two_way_p95_error_mm": float(np.quantile(all_errors, 0.95)),
            "two_way_maximum_error_mm": float(np.max(all_errors)),
            "two_way_within_4mm_percent": float(100.0 * np.mean(all_errors <= 4.0)),
        })
    if not summaries:
        raise RuntimeError("no candidate model completed both repeat directions")
    summaries.sort(key=lambda item: (item["two_way_p95_error_mm"], item["two_way_median_error_mm"]))
    return results, summaries[0]


def train_provisional_model(samples: list[ContactSample], best: dict[str, Any]):
    feature_name = best["feature"]
    models = _model_candidates(len(samples))
    model = models[best["model"]]
    train_x = np.stack([sample.features[feature_name] for sample in samples])
    train_y = np.asarray([[sample.x_mm, sample.y_mm] for sample in samples], dtype=float)
    model.fit(train_x, train_y)
    return {
        "schema": "fast_flank_position_model_provisional_v1",
        "feature": feature_name,
        "model_name": best["model"],
        "model": model,
        "point_indices": [sample.point_index for sample in samples],
        "repeat_ids": [sample.repeat for sample in samples],
        "capture_sha256": [sample.sha256 for sample in samples],
        "independent_accuracy_verified": False,
        "physical_15hz_verified": False,
        "contact_area_verified": False,
    }


def evaluate_fixed_candidate(
    train_samples: list[ContactSample],
    validation_samples: list[ContactSample],
    best: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Fit the already selected candidate and score a later untouched repeat."""
    feature_name = best["feature"]
    model = _model_candidates(len(train_samples))[best["model"]]
    train_x = np.stack([sample.features[feature_name] for sample in train_samples])
    train_y = np.asarray([[sample.x_mm, sample.y_mm] for sample in train_samples], dtype=float)
    validation_x = np.stack([sample.features[feature_name] for sample in validation_samples])
    validation_y = np.asarray([[sample.x_mm, sample.y_mm] for sample in validation_samples], dtype=float)
    model.fit(train_x, train_y)
    prediction = np.asarray(model.predict(validation_x), dtype=float)
    return model, _metrics(prediction, validation_y)


def _resolve_default_inputs(directory: Path) -> list[tuple[Path, int]]:
    resolved = []
    for point in range(1, 18):
        first_name = (
            "ch1_fast_flank_contact_p09_0mm_clean_r2_20260906.json"
            if point == 9
            else f"ch1_fast_flank_contact_p{point:02d}_0mm_clean_r1_20260906.json"
        )
        second_name = (
            "ch1_fast_flank_contact_p09_0mm_clean_r3_20260906.json"
            if point == 9
            else f"ch1_fast_flank_contact_p{point:02d}_0mm_clean_r2_20260906.json"
        )
        resolved.append((directory / first_name, 1))
        resolved.append((directory / second_name, 2))
    return resolved


def _resolve_validation_inputs(directory: Path, repeat: int = 3) -> list[tuple[Path, int]]:
    return [
        (
            directory
            / f"ch1_fast_flank_contact_p{point:02d}_0mm_clean_validation_r{repeat}_20260906.json",
            repeat,
        )
        for point in range(1, 18)
    ]


def _resolve_template_inputs(
    directory: Path,
    template: str,
    repeats: Iterable[int],
) -> list[tuple[Path, int]]:
    """Resolve an explicit point/repeat filename template without guessing labels."""
    resolved = []
    for repeat in repeats:
        if int(repeat) < 1:
            raise ValueError(f"repeat must be positive, got {repeat!r}")
        for point in range(1, 18):
            try:
                name = template.format(point=point, repeat=int(repeat))
            except (KeyError, ValueError) as exc:
                raise ValueError(f"invalid capture filename template {template!r}: {exc}") from exc
            resolved.append((directory / name, int(repeat)))
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", type=Path, default=Path(__file__).with_name("outputs"))
    parser.add_argument(
        "--validation-capture-directory",
        type=Path,
        default=Path(__file__).with_name("outputs"),
    )
    parser.add_argument("--skip-independent-validation", action="store_true")
    parser.add_argument("--validation-repeat", type=int, default=3)
    parser.add_argument(
        "--expected-depth-mm",
        type=float,
        default=0.0,
        help="Required physical indentation for every capture (0..0.8 mm).",
    )
    parser.add_argument(
        "--development-template",
        help=(
            "Explicit filename template containing {point:02d} and {repeat}; "
            "when omitted, use the legacy 0 mm capture names."
        ),
    )
    parser.add_argument(
        "--development-repeats",
        type=int,
        nargs=2,
        default=(1, 2),
        metavar=("FIRST", "SECOND"),
    )
    parser.add_argument(
        "--validation-template",
        help=(
            "Explicit validation filename template containing {point:02d} and {repeat}; "
            "required with --development-template unless validation is skipped."
        ),
    )
    parser.add_argument(
        "--required-commissioning-schema",
        help="Reject captures from a different physical acquisition protocol.",
    )
    parser.add_argument(
        "--freeze-candidate-output",
        type=Path,
        help="Write a code/data/model lock before acquiring a later validation repeat.",
    )
    parser.add_argument(
        "--locked-candidate",
        type=Path,
        help="Evaluate the exact feature/model and development data recorded in a prior lock.",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model-output", type=Path)
    args = parser.parse_args()

    if not math.isfinite(args.expected_depth_mm) or not 0.0 <= args.expected_depth_mm <= 0.8:
        raise SystemExit("--expected-depth-mm must be within 0..0.8 mm")
    if args.development_template:
        inputs = _resolve_template_inputs(
            args.capture_directory,
            args.development_template,
            args.development_repeats,
        )
    else:
        if tuple(args.development_repeats) != (1, 2):
            raise SystemExit("--development-repeats requires --development-template")
        inputs = _resolve_default_inputs(args.capture_directory)
    missing = [str(path) for path, _ in inputs if not path.is_file()]
    if missing:
        raise SystemExit("missing captures:\n" + "\n".join(missing))
    samples = [
        load_contact_sample(
            path,
            repeat_override=repeat,
            expected_depth_mm=args.expected_depth_mm,
            required_commissioning_schema=args.required_commissioning_schema,
        )
        for path, repeat in inputs
    ]
    lock_data: dict[str, Any] | None = None
    if args.locked_candidate:
        lock_data = json.loads(args.locked_candidate.read_text(encoding="utf-8"))
        if lock_data.get("schema") != "fast_flank_position_candidate_lock_v1":
            raise SystemExit("locked candidate has the wrong schema")
        if abs(float(lock_data.get("expected_depth_mm", 0.0)) - args.expected_depth_mm) > 1e-9:
            raise SystemExit("locked candidate depth does not match --expected-depth-mm")
        if lock_data.get("required_commissioning_schema") != args.required_commissioning_schema:
            raise SystemExit("locked candidate commissioning schema does not match")
        actual_hashes = {sample.path.name: sample.sha256 for sample in samples}
        if actual_hashes != lock_data.get("development_capture_sha256"):
            raise SystemExit("development capture hashes do not match the locked candidate")
        if _sha256(Path(__file__)) != lock_data.get("script_sha256"):
            raise SystemExit("calibration code changed after candidate lock")
        best = dict(lock_data["candidate"])
        if best.get("feature") not in samples[0].features:
            raise SystemExit("locked feature is unavailable")
        if best.get("model") not in _model_candidates(len(samples)):
            raise SystemExit("locked model is unavailable")
        split_results = []
    else:
        split_results, best = evaluate_repeat_splits(samples)

    if args.freeze_candidate_output:
        lock = {
            "schema": "fast_flank_position_candidate_lock_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script_sha256": _sha256(Path(__file__)),
            "candidate": {"feature": best["feature"], "model": best["model"]},
            "development_capture_sha256": {
                sample.path.name: sample.sha256 for sample in samples
            },
            "development_contact_count": len(samples),
            "validation_repeat_reserved": int(args.validation_repeat),
            "expected_depth_mm": float(args.expected_depth_mm),
            "development_template": args.development_template,
            "validation_template": args.validation_template,
            "required_commissioning_schema": args.required_commissioning_schema,
            "validation_files_must_not_exist_when_locked": True,
        }
        validation_paths = (
            _resolve_template_inputs(
                args.validation_capture_directory,
                args.validation_template,
                (args.validation_repeat,),
            )
            if args.validation_template
            else _resolve_validation_inputs(
                args.validation_capture_directory, args.validation_repeat
            )
        )
        existing = [str(path) for path, _ in validation_paths if path.exists()]
        if existing:
            raise SystemExit(
                "refusing to preregister after reserved validation files exist:\n"
                + "\n".join(existing)
            )
        args.freeze_candidate_output.parent.mkdir(parents=True, exist_ok=True)
        args.freeze_candidate_output.write_text(
            json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    validation_samples: list[ContactSample] = []
    independent_validation: dict[str, Any] | None = None
    if not args.skip_independent_validation:
        if args.development_template and not args.validation_template:
            raise SystemExit(
                "--validation-template is required with --development-template unless "
                "--skip-independent-validation is used"
            )
        validation_inputs = (
            _resolve_template_inputs(
                args.validation_capture_directory,
                args.validation_template,
                (args.validation_repeat,),
            )
            if args.validation_template
            else _resolve_validation_inputs(
                args.validation_capture_directory, args.validation_repeat
            )
        )
        validation_missing = [str(path) for path, _ in validation_inputs if not path.is_file()]
        if validation_missing:
            raise SystemExit("missing independent validation captures:\n" + "\n".join(validation_missing))
        validation_samples = [
            load_contact_sample(
                path,
                repeat_override=repeat,
                expected_depth_mm=args.expected_depth_mm,
                required_commissioning_schema=args.required_commissioning_schema,
            )
            for path, repeat in validation_inputs
        ]
        _, metrics = evaluate_fixed_candidate(samples, validation_samples, best)
        independent_validation = {
            "protocol": (
                "Candidate feature/model, calibration code and development capture hashes were "
                "locked before the reserved validation files existed; fit on all 34 development "
                "contacts and evaluated once on 17 later contacts."
                if lock_data is not None
                else "Validation contacts were held out from numerical fitting/model selection, but "
                "the feature pipeline was finalized after their acquisition; treat this as a post-hoc "
                "holdout, not an independent accuracy certificate."
            ),
            "pipeline_preregistered_before_validation_acquisition": lock_data is not None,
            "candidate_lock": str(args.locked_candidate.resolve()) if args.locked_candidate else None,
            "feature": best["feature"],
            "model": best["model"],
            "train_contact_count": len(samples),
            "validation_contact_count": len(validation_samples),
            "metrics": metrics,
        }

    report = {
        "schema": "fast_flank_position_repeat_validation_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(samples),
        "unique_positions": len({sample.point_index for sample in samples}),
        "repeat_count": len({sample.repeat for sample in samples}),
        "grouping": "one physical contact equals one indivisible sample",
        "depth_mm": float(args.expected_depth_mm),
        "required_commissioning_schema": args.required_commissioning_schema,
        "hold_frame_minimum": min(sample.hold_frames for sample in samples),
        "captures": [
            {
                "path": str(sample.path.resolve()),
                "sha256": sample.sha256,
                "point_index": sample.point_index,
                "repeat": sample.repeat,
                "xy_mm": [sample.x_mm, sample.y_mm],
                "contact_sequence": sample.contact_sequence,
                "release_sequence": sample.release_sequence,
                "hold_frames": sample.hold_frames,
            }
            for sample in samples
        ],
        "best_candidate": best,
        "candidate_split_results": split_results,
        "candidate_lock": str(args.locked_candidate.resolve()) if args.locked_candidate else None,
        "independent_validation": independent_validation,
        "independent_accuracy_verified": False,
        "physical_15hz_verified": False,
        "contact_area_verified": False,
        "deployment_status": "provisional_not_for_accuracy_claims",
        "limitations": [
            "Only two repeats per position were used and they also selected the candidate model.",
            "The CNC contact is static and does not supply synchronized 15 Hz pressure ground truth.",
            "No measured contact patch or force label was recorded, so physical area is unknown.",
            "The current grid samples 17 positions and does not independently validate arbitrary continuous positions.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.model_output:
        final_samples = samples + validation_samples
        artifact = train_provisional_model(final_samples, best)
        artifact["refit_after_independent_validation"] = bool(validation_samples)
        artifact["independent_validation_metrics_before_refit"] = (
            independent_validation["metrics"] if independent_validation else None
        )
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, args.model_output)

    print(json.dumps({
        "report": str(args.report.resolve()),
        "model_output": str(args.model_output.resolve()) if args.model_output else None,
        "sample_count": len(samples),
        "best_candidate": best,
        "independent_validation": independent_validation,
        "independent_accuracy_verified": False,
        "physical_15hz_verified": False,
        "contact_area_verified": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
