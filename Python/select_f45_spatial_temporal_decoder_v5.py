"""Select a causal, spatially aware decoder for F45 contact localisation.

The optical board produces pressure/amplitude samples at the full F45 frame
rate.  Contact position changes much more slowly, so this audit deliberately
decouples the two outputs: the classifier still runs on every frame while only
its XY position is causally smoothed.  Every fold excludes a complete physical
acquisition round; frames are never randomly split across train and test.

This command is offline and never opens the optical board or Mach3.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_f45_cross_session_frame_models import causal_held_features
from evaluate_f45_cross_session_models import load_rounds
from evaluate_fast_fullmap_position import _capture_feature
from select_f45_continuous_position_v3 import _load_visible_round, _parse_round
from select_f45_multidepth_position_candidate import transform


FEATURE = "pair_transition_delta45"
TOOL_RADIUS_MM = 4.0


@dataclass(frozen=True)
class Decoder:
    name: str
    kind: str
    parameter: float
    secondary: float | None = None


def decoders():
    yield Decoder("hard_xy", "hard", 0.0)
    for window in (2, 3, 4, 5):
        yield Decoder(f"hard_causal_mean_{window}", "hard_mean", float(window))
    for alpha in (0.20, 0.25, 0.33, 0.50, 0.67):
        yield Decoder(f"hard_causal_ema_{alpha:.2f}", "hard_ema", alpha)
    for temperature in (1.0, 1.5, 2.0, 3.0, 5.0):
        yield Decoder(f"posterior_temperature_{temperature:.1f}", "posterior", temperature)
    for temperature in (1.5, 2.0, 3.0):
        for alpha in (0.25, 0.50):
            yield Decoder(
                f"posterior_temperature_{temperature:.1f}_ema_{alpha:.2f}",
                "posterior_ema",
                temperature,
                alpha,
            )


def nearest_labels(xy: np.ndarray, labels: np.ndarray, label_xy: np.ndarray) -> np.ndarray:
    distance = np.linalg.norm(xy[:, None, :] - label_xy[None, :, :], axis=2)
    return labels[np.argmin(distance, axis=1)]


def causal_mean(values: np.ndarray, window: int) -> np.ndarray:
    result = np.empty_like(values, dtype=float)
    total = np.zeros(values.shape[1], dtype=float)
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        result[index] = total / min(index + 1, window)
    return result


def causal_ema(values: np.ndarray, alpha: float) -> np.ndarray:
    result = np.empty_like(values, dtype=float)
    state = np.asarray(values[0], dtype=float)
    result[0] = state
    for index in range(1, len(values)):
        state = alpha * values[index] + (1.0 - alpha) * state
        result[index] = state
    return result


def temperature_probability(probability: np.ndarray, temperature: float) -> np.ndarray:
    # Work from probabilities because sklearn's LDA decision_function has a
    # binary special case.  Clipping only protects log(0); it does not invent
    # evidence for an absent class.
    logits = np.log(np.clip(probability, 1e-300, 1.0)) / temperature
    logits -= np.max(logits, axis=1, keepdims=True)
    value = np.exp(logits)
    return value / np.sum(value, axis=1, keepdims=True)


def decode(model, values: np.ndarray, decoder: Decoder, coordinates: dict[int, np.ndarray]):
    classes = np.asarray(model.classes_, dtype=int)
    class_xy = np.stack([coordinates[int(label)] for label in classes])
    if decoder.kind.startswith("hard"):
        hard = np.asarray(model.predict(values), dtype=int)
        xy = np.stack([coordinates[int(label)] for label in hard])
        if decoder.kind == "hard_mean":
            xy = causal_mean(xy, int(decoder.parameter))
        elif decoder.kind == "hard_ema":
            xy = causal_ema(xy, decoder.parameter)
    else:
        if decoder.kind == "posterior":
            temperature = decoder.parameter
            alpha = None
        else:
            temperature = decoder.parameter
            alpha = decoder.secondary
        probability = temperature_probability(model.predict_proba(values), temperature)
        xy = probability @ class_xy
        if alpha is not None:
            xy = causal_ema(xy, alpha)
    labels = np.asarray(sorted(coordinates), dtype=int)
    label_xy = np.stack([coordinates[int(label)] for label in labels])
    return xy, nearest_labels(xy, labels, label_xy)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", action="append", default=[], type=_parse_round)
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
    labels = np.asarray(sorted(coordinates), dtype=int)
    label_xy = np.stack([coordinates[int(label)] for label in labels])
    raw = {
        (round_name, int(row["point_index"])): causal_held_features(Path(row["path"]))[FEATURE]
        for round_name, records in rounds.items()
        for row in records
    }

    ranked = []
    for representation in ("block_unit135", "pair_unit90", "unit135"):
        folds_by_decoder = {decoder.name: [] for decoder in decoders()}
        decoder_specs = {decoder.name: decoder for decoder in decoders()}
        for held_name, held_records in rounds.items():
            train_x = []
            train_y = []
            for round_name, records in rounds.items():
                if round_name == held_name:
                    continue
                for row in records:
                    point = int(row["point_index"])
                    values = raw[(round_name, point)]
                    train_x.append(values)
                    train_y.append(np.full(len(values), point, dtype=int))
            model = make_pipeline(
                StandardScaler(),
                LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            ).fit(transform(np.concatenate(train_x), representation), np.concatenate(train_y))

            for name, decoder in decoder_specs.items():
                trial_rows = []
                frame_errors = []
                frame_exact = []
                for row in held_records:
                    point = int(row["point_index"])
                    x_test = transform(raw[(held_name, point)], representation)
                    predicted_xy, predicted_label = decode(model, x_test, decoder, coordinates)
                    errors = np.linalg.norm(predicted_xy - coordinates[point], axis=1)
                    trial_xy = np.mean(predicted_xy, axis=0)
                    trial_error = float(np.linalg.norm(trial_xy - coordinates[point]))
                    trial_label = int(nearest_labels(trial_xy[None, :], labels, label_xy)[0])
                    frame_errors.extend(errors.tolist())
                    frame_exact.extend((predicted_label == point).tolist())
                    trial_rows.append(
                        {
                            "point": point,
                            "frames": int(len(x_test)),
                            "trial_xy_mm": [float(value) for value in trial_xy],
                            "trial_error_mm": trial_error,
                            "trial_nearest_label": trial_label,
                            "trial_nearest_label_exact": trial_label == point,
                            "frame_within_tool_radius_percent": float(
                                100.0 * np.mean(errors <= TOOL_RADIUS_MM + 1e-6)
                            ),
                        }
                    )
                frame_errors = np.asarray(frame_errors, dtype=float)
                trial_errors = np.asarray([row["trial_error_mm"] for row in trial_rows])
                folds_by_decoder[name].append(
                    {
                        "held_round": held_name,
                        "depth_mm": float(np.median([row["depth_mm"] for row in held_records])),
                        "held_points": len(held_records),
                        "trial_nearest_label_exact": int(sum(row["trial_nearest_label_exact"] for row in trial_rows)),
                        "trial_within_tool_radius": int(np.sum(trial_errors <= TOOL_RADIUS_MM + 1e-6)),
                        "trial_error_maximum_mm": float(np.max(trial_errors)),
                        "frame_nearest_label_exact_percent": float(100.0 * np.mean(frame_exact)),
                        "frame_within_tool_radius_percent": float(
                            100.0 * np.mean(frame_errors <= TOOL_RADIUS_MM + 1e-6)
                        ),
                        "frame_error_mean_mm": float(np.mean(frame_errors)),
                        "frame_error_p95_mm": float(np.percentile(frame_errors, 95)),
                        "trials": trial_rows,
                    }
                )

        for name, folds in folds_by_decoder.items():
            full_round_folds = [row for row in folds if row["held_points"] == 17]
            if not full_round_folds:
                raise ValueError("at least one complete 17-position round is required")
            ranked.append(
                {
                    "representation": representation,
                    "estimator": "lda_shrinkage",
                    "decoder": name,
                    "pressure_path": "unsmoothed_full_rate_feature_energy",
                    "position_path": "causal_spatial_decoder",
                    "worst_trial_within_tool_radius_percent": float(
                        min(100.0 * row["trial_within_tool_radius"] / row["held_points"] for row in folds)
                    ),
                    "worst_frame_within_tool_radius_percent": float(
                        min(row["frame_within_tool_radius_percent"] for row in folds)
                    ),
                    "worst_trial_error_maximum_mm": float(max(row["trial_error_maximum_mm"] for row in folds)),
                    "worst_frame_error_p95_mm": float(max(row["frame_error_p95_mm"] for row in folds)),
                    "worst_trial_nearest_label_exact_percent": float(
                        min(100.0 * row["trial_nearest_label_exact"] / row["held_points"] for row in folds)
                    ),
                    "worst_full_round_trial_nearest_label_exact": int(
                        min(row["trial_nearest_label_exact"] for row in full_round_folds)
                    ),
                    "development_footprint_gate_passed": bool(
                        all(row["trial_within_tool_radius"] == row["held_points"] for row in folds)
                        and min(row["frame_within_tool_radius_percent"] for row in folds) >= 98.5
                    ),
                    "mean_frame_error_mm": float(np.mean([row["frame_error_mean_mm"] for row in folds])),
                    "folds": folds,
                }
            )

    ranked.sort(
        key=lambda row: (
            -row["worst_trial_within_tool_radius_percent"],
            -int(row["development_footprint_gate_passed"]),
            -row["worst_full_round_trial_nearest_label_exact"],
            -row["worst_frame_within_tool_radius_percent"],
            row["worst_trial_error_maximum_mm"],
            row["worst_frame_error_p95_mm"],
            -row["worst_trial_nearest_label_exact_percent"],
            row["mean_frame_error_mm"],
        )
    )
    result = {
        "schema": "f45_spatial_temporal_decoder_selection_v5",
        "feature": FEATURE,
        "tool_radius_mm": TOOL_RADIUS_MM,
        "nominal_tool_face_area_mm2": float(np.pi * TOOL_RADIUS_MM**2),
        "random_frame_split_used": False,
        "physical_15hz_verified": False,
        "deployment_eligible": False,
        "rounds": {
            name: {
                "points": [int(row["point_index"]) for row in records],
                "depth_mm": float(np.median([row["depth_mm"] for row in records])),
            }
            for name, records in rounds.items()
        },
        "best": ranked[0],
        "ranked": ranked,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps([{key: value for key, value in row.items() if key != "folds"} for row in ranked[:15]], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
