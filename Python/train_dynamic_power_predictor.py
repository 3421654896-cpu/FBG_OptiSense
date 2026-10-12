"""Train and cross-validate early-PD predictors for settled laser output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error


FEATURE_NAMES = [
    "target_nm", "gain_ma", "soa_ma", "phase_ma", "wave_a_ma", "wave_b_ma",
    "previous_target_nm", "previous_gain_ma", "previous_soa_ma", "previous_phase_ma",
    "previous_wave_a_ma", "previous_wave_b_ma", "delta_gain_ma", "delta_soa_ma",
    "delta_phase_ma", "delta_wave_a_ma", "delta_wave_b_ma", "early_pdt_code",
    "early_pdr_code", "early_pdt_signal", "early_pdr_signal", "early_signal_ratio",
]


def features(sample: dict) -> list[float]:
    current = [float(value) for value in sample["currents_ma"]]
    previous = [float(value) for value in sample["previous_currents_ma"]]
    target = float(sample["target_nm"])
    previous_target = (
        float(sample["previous_target_nm"])
        if sample.get("previous_target_nm") is not None else target
    )
    pdt = float(sample["early_pdt_code"])
    pdr = float(sample["early_pdr_code"])
    pdt_signal = 2048.0 - pdt
    pdr_signal = 2048.0 - pdr
    return [
        target, *current, previous_target, *previous,
        *[cur - prev for cur, prev in zip(current, previous)],
        pdt, pdr, pdt_signal, pdr_signal,
        pdr_signal / max(1.0, pdt_signal),
    ]


def metrics(y_true: np.ndarray, y_pred: np.ndarray, target_power_mw: float = 1.53) -> dict:
    error = np.abs(y_pred - y_true)
    return {
        "mae_mw": float(mean_absolute_error(y_true, y_pred)),
        "median_abs_error_mw": float(np.median(error)),
        "p95_abs_error_mw": float(np.percentile(error, 95)),
        "max_abs_error_mw": float(np.max(error)),
        "within_0p5_percent_count": int(np.sum(error <= target_power_mw * 0.005)),
        "sample_count": int(len(error)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("fullband_dynamic_power_training_0p2nm.json"))
    parser.add_argument("--output", type=Path, default=Path("fullband_dynamic_power_predictor.joblib"))
    parser.add_argument("--report", type=Path, default=Path("fullband_dynamic_power_predictor_report.json"))
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    clean = [
        sample for sample in payload["rows"]
        if (
            sample["settled"].get("smsr_db") is None
            or float(sample["settled"]["smsr_db"]) >= 20.0
        )
        and abs(float(sample["settled"]["wavelength_error_pm"])) <= 10.0
    ]
    x = np.asarray([features(sample) for sample in clean], dtype=float)
    y_power = np.asarray([float(sample["settled"]["power_mw"]) for sample in clean], dtype=float)
    y_wave = np.asarray([float(sample["settled"]["wavelength_error_pm"]) for sample in clean], dtype=float)

    unique_targets = sorted({round(float(sample["target_nm"]), 6) for sample in clean})
    held_targets = set(unique_targets[::5])
    test_mask = np.asarray([
        round(float(sample["target_nm"]), 6) in held_targets for sample in clean
    ])
    train_mask = ~test_mask
    candidates = {
        "extra_trees": ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=1, max_features=0.9,
            random_state=20260830, n_jobs=-1,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=500, min_samples_leaf=1, max_features=0.9,
            random_state=20260830, n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=0.01, random_state=20260830,
        ),
    }
    reports = {}
    fitted = {}
    for name, model in candidates.items():
        model.fit(x[train_mask], y_power[train_mask])
        prediction = model.predict(x[test_mask])
        reports[name] = metrics(y_power[test_mask], prediction)
        fitted[name] = model
    best_name = min(reports, key=lambda name: reports[name]["mae_mw"])

    power_model = candidates[best_name]
    power_model.fit(x, y_power)
    wavelength_model = ExtraTreesRegressor(
        n_estimators=500, min_samples_leaf=1, max_features=0.9,
        random_state=20260831, n_jobs=-1,
    ).fit(x, y_wave)
    artifact = {
        "feature_names": FEATURE_NAMES,
        "power_model_name": best_name,
        "power_model": power_model,
        "wavelength_error_model": wavelength_model,
        "source": args.input.name,
        "training_sample_count": len(clean),
        "target_power_mw": 1.53,
    }
    joblib.dump(artifact, args.output, compress=3)
    report = {
        "source": args.input.name,
        "total_samples": len(payload["rows"]),
        "clean_samples": len(clean),
        "rejected_samples": len(payload["rows"]) - len(clean),
        "training_samples": int(np.sum(train_mask)),
        "held_out_samples": int(np.sum(test_mask)),
        "held_out_target_count": len(held_targets),
        "split": "every fifth 0.20 nm target held out with all SOA variants",
        "feature_names": FEATURE_NAMES,
        "candidate_metrics": reports,
        "selected_power_model": best_name,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
