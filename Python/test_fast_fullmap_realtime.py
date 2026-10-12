import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import joblib

from evaluate_coverage_position import _feature_variants
from fast_fullmap_realtime import (
    NOMINAL_TOOL_FACE_AREA_MM2,
    FastFullMapRealtimeLocalizer,
)


def make_model(path: Path):
    baseline = np.full(45, 100.0)
    centre = np.zeros(81)
    scale = np.ones(81)
    prototypes = []
    for round_name in ("r1", "r2"):
        for point in range(1, 18):
            delta = np.zeros(45)
            delta[point - 1] = 10.0
            vector = _feature_variants(
                baseline, baseline + delta, np.ones(45)
            )["delta45_plus_summary36"]
            prototypes.append(
                {
                    "point_index": point,
                    "training_round": round_name,
                    "work_xy_mm": [float(point), float(point * 2)],
                    "vector": vector.tolist(),
                }
            )
    payload = {
        "schema": "ch1_fast_fullmap_position_model_v1",
        "protocol": {
            "name": "F45",
            "route_rows": list(range(45)),
            "spacing_us": 50,
            "feedback_selectors": [0, 2],
        },
        "baseline": {
            "minimum_laser_warmup_frames": 16,
            "no_contact_frames": 64,
        },
        "activation": {"threshold_codes": 0.2},
        "classifier": {
            "feature": "first_delta45_plus_summary36",
            "scaling": "robust",
            "metric": "euclidean",
            "centre": centre.tolist(),
            "scale": scale.tolist(),
            "prototypes": prototypes,
        },
        "static_validation": {"gate": {"passed": True}},
    }
    path.write_text(json.dumps(payload), "utf-8")


def frame(sequence: int, values: np.ndarray | None = None):
    values = np.full(45, 100.0) if values is None else np.asarray(values)
    return {
        "sequence": sequence,
        "cycle_start_us": sequence * 32_000,
        "spacing_us": 50,
        "minimum_warmup_complete": sequence >= 16,
        "records": [
            {
                "row": index,
                "first_code": int(values[index]),
                "second_code": int(values[index]),
            }
            for index in range(45)
        ],
    }


def make_v2_model(path: Path):
    train_x = []
    train_y = []
    for point in range(1, 18):
        vector = np.zeros(135)
        vector[point - 1] = 10.0
        vector[45 + point - 1] = 10.0
        train_x.append(vector)
        train_y.append(point)
    classifier = ExtraTreesClassifier(
        n_estimators=400, random_state=7, max_features="sqrt"
    ).fit(np.asarray(train_x), np.asarray(train_y))
    payload = {
        "schema": "ch1_fast_fullmap_position_model_v2",
        "protocol": {
            "name": "F45",
            "route_rows": list(range(45)),
            "spacing_us": 50,
            "feedback_selectors": [0, 2],
        },
        "baseline": {
            "minimum_laser_warmup_frames": 16,
            "no_contact_frames": 64,
        },
        "activation": {"threshold_codes": 0.2},
        "feature": {
            "name": "pair_transition_delta45",
            "dimensions": 135,
            "preprocessing": "none",
        },
        "classifier": classifier,
        "coordinates_by_point": {
            point: [float(point), float(point * 2)] for point in range(1, 18)
        },
        "static_validation": {
            "trial_majority_exact_percent": 100.0,
            "frame_exact_percent": 96.15,
            "frame_within_4mm_percent": 100.0,
            "gate": {"passed": True},
        },
    }
    joblib.dump(payload, path)


class FastFullMapRealtimeTests(unittest.TestCase):
    def test_high_gain_data_cannot_use_legacy_low_gain_model(self):
        localizer = FastFullMapRealtimeLocalizer(self.model)
        item = frame(0)
        item['feedback_selectors'] = [0,1]
        with self.assertRaisesRegex(ValueError, 'F45'):
            localizer.update(item)
        self.assertFalse(localizer.baseline_ready)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.model = Path(self.temporary.name) / "model.json"
        make_model(self.model)

    def tearDown(self):
        self.temporary.cleanup()

    def test_baseline_freezes_then_classifies_one_all_fresh_frame(self):
        localizer = FastFullMapRealtimeLocalizer(self.model)
        for sequence in range(80):
            result = localizer.update(frame(sequence))
        self.assertTrue(result.baseline_ready)
        self.assertEqual(result.press_footprint_area_mm2, 0.0)
        values = np.full(45, 100)
        values[6] += 10
        result = localizer.update(frame(80, values))
        self.assertTrue(result.contact_detected)
        self.assertEqual(result.predicted_point, 7)
        self.assertEqual(result.provisional_x_mm, 7.0)
        self.assertEqual(result.provisional_y_mm, 14.0)
        self.assertTrue(result.sampling_gate_for_15hz)
        self.assertEqual(len(result.first_delta_codes), 45)
        self.assertEqual(len(result.response_strength), 9)
        self.assertFalse(result.physical_15hz_verified)
        self.assertIsNone(result.physical_contact_area_mm2)
        self.assertAlmostEqual(result.nominal_tool_face_area_mm2, math.pi * 4.0**2)
        self.assertAlmostEqual(result.press_footprint_area_mm2, math.pi * 4.0**2)
        payload = result.to_dict()
        self.assertAlmostEqual(
            payload["nominal_tool_face_area_mm2"], NOMINAL_TOOL_FACE_AREA_MM2
        )
        self.assertEqual(payload["press_face_diameter_mm"], 8.0)
        self.assertAlmostEqual(payload["press_footprint_area_mm2"], math.pi * 4.0**2)
        self.assertEqual(
            payload["press_footprint_basis"],
            "detected_contact_x_fixed_8mm_tool_face",
        )
        self.assertIsNone(payload["physical_contact_area_mm2"])

    def test_sequence_gap_invalidates_baseline(self):
        localizer = FastFullMapRealtimeLocalizer(self.model)
        localizer.update(frame(0))
        with self.assertRaisesRegex(ValueError, "missing"):
            localizer.update(frame(2))
        self.assertFalse(localizer.baseline_ready)

    def test_v2_uses_both_physical_adc_samples_and_reports_validation(self):
        model = Path(self.temporary.name) / "model_v2.joblib"
        make_v2_model(model)
        localizer = FastFullMapRealtimeLocalizer(model)
        for sequence in range(80):
            result = localizer.update(frame(sequence))
        values = np.full(45, 100)
        values[11] += 10
        result = localizer.update(frame(80, values))
        self.assertTrue(result.contact_detected)
        self.assertEqual(result.predicted_point, 12)
        self.assertEqual(result.provisional_x_mm, 12.0)
        self.assertEqual(result.confidence_kind, "extratrees_probability_margin")
        self.assertEqual(
            result.model_schema, "ch1_fast_fullmap_position_model_v2"
        )
        self.assertAlmostEqual(result.static_validation_frame_exact_percent, 96.15)
        self.assertAlmostEqual(
            result.to_dict()["static_validation"]["frame_within_4mm_percent"],
            100.0,
        )

    def test_v2_rejects_missing_second_sample(self):
        model = Path(self.temporary.name) / "model_v2.joblib"
        make_v2_model(model)
        localizer = FastFullMapRealtimeLocalizer(model)
        bad = frame(0)
        del bad["records"][0]["second_code"]
        with self.assertRaisesRegex(ValueError, "second-code"):
            localizer.update(bad)

    def test_v2_fast_single_frame_vote_matches_sklearn_probability(self):
        model = Path(self.temporary.name) / "model_v2.joblib"
        make_v2_model(model)
        localizer = FastFullMapRealtimeLocalizer(model)
        vector = np.linspace(-3.0, 4.0, 135)
        probabilities = localizer._tree_classifier.predict_proba(
            vector.reshape(1, -1)
        )[0]
        predicted, margin = localizer._predict_extra_trees(vector)
        ordered = np.sort(probabilities)
        self.assertEqual(
            predicted,
            int(localizer._tree_classifier.classes_[np.argmax(probabilities)]),
        )
        self.assertAlmostEqual(margin, float(ordered[-1] - ordered[-2]), places=12)


if __name__ == "__main__":
    unittest.main()
