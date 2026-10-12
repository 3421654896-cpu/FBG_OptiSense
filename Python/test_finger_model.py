import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from finger_dataset import ContactEvent, FingerSessionWriter, SpectrumFrame
from finger_features import CausalFeatureWindow, PeakQuality
from finger_inference import FingerInferenceEngine
from finger_model import (
    AreaNotIdentifiableError,
    FingerModelBundle,
    FingerRuntimeSignature,
    ModelCompatibilityError,
    ModelTrainingConfig,
    train_finger_model,
)
from train_finger_localizer import (
    _session_adc_value_kind,
    generate_synthetic_samples,
    load_session_training_samples,
)

TEST_CRC = 0x1234ABCD


def _training_config():
    return ModelTrainingConfig(
        n_estimators=40,
        min_samples_leaf=1,
        inference_benchmark_repeats=20,
        random_state=17,
    )


def _spectrum(shift_nm=0.0):
    wavelengths = []
    values = []
    for peak in range(9):
        center = 1526.0 + 4.0 * peak
        local = center + np.asarray((-0.020, -0.010, 0.0, 0.010, 0.020))
        voltage = 0.04 + 0.70 * np.exp(
            -0.5 * ((local - center - shift_nm) / 0.010) ** 2
        )
        wavelengths.extend(local.tolist())
        values.extend(voltage.tolist())
    codes = np.rint(np.asarray(values) * 4096.0 / 2.5).astype(int)
    return tuple(wavelengths), tuple(codes.tolist())


class FingerModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.signature = FingerRuntimeSignature(table_crc32=TEST_CRC, adc_value_kind="direct_adc")
        cls.variable_samples = generate_synthetic_samples(
            fixed_eight_mm_area=False,
            seed=123,
            session_count=6,
            press_events_per_session=12,
        )
        cls.variable_bundle, cls.variable_report = train_finger_model(
            cls.variable_samples, cls.signature, config=_training_config()
        )
        cls.fixed_samples = generate_synthetic_samples(
            fixed_eight_mm_area=True,
            seed=456,
            session_count=6,
            press_events_per_session=12,
        )
        cls.fixed_bundle, cls.fixed_report = train_finger_model(
            cls.fixed_samples, cls.signature, config=_training_config()
        )

    def test_grouped_holdout_never_splits_frames_randomly(self):
        split = self.variable_report["metrics"]["split"]
        self.assertIn(split["level"], {"session", "event"})
        self.assertFalse(split["random_frame_split_used"])
        self.assertEqual(split["group_overlap"], [])
        self.assertTrue(set(split["train_groups"]).isdisjoint(split["test_groups"]))

    def test_adc_provenance_is_pinned_in_model_and_unknown_is_rejected(self):
        self.assertEqual(self.variable_bundle.signature.to_dict()['adc_value_kind'],'direct_adc')
        for kind in ('unknown','firmware_rc_estimate'):
            with self.subTest(kind=kind),self.assertRaisesRegex(ModelCompatibilityError,'adc_value_kind'):
                self.variable_bundle.assert_compatible(replace(self.signature,adc_value_kind=kind))
        for version in ('fbg-finger-localizer/v1', 'fbg-finger-localizer/v2'):
            with self.subTest(version=version):
                legacy = replace(self.variable_bundle, bundle_version=version)
                with self.assertRaisesRegex(ModelCompatibilityError, 'unsupported'):
                    legacy.assert_compatible(self.signature)

    def test_adc_kind_change_resets_causal_inference_history(self):
        wavelength,codes = _spectrum()
        base=SpectrumFrame(0,wavelength,codes,monotonic_ns=1,
                           table_crc32=TEST_CRC,packet_crc_ok=True,adc_value_kind='direct_adc')
        engine=FingerInferenceEngine(self.variable_bundle,base)
        engine.update(replace(base,sequence=1,monotonic_ns=2))
        self.assertEqual(engine.history_size,1)
        with self.assertRaisesRegex(ModelCompatibilityError,'adc_value_kind'):
            engine.update(replace(base,sequence=2,monotonic_ns=3,adc_value_kind='firmware_rc_estimate'))
        self.assertEqual(engine.history_size,0)
        with self.assertRaisesRegex(ModelCompatibilityError,'unknown'):
            FingerInferenceEngine(self.variable_bundle,replace(base,adc_value_kind='unknown'))

    def test_training_session_mixed_or_unknown_adc_kinds_are_refused(self):
        wavelength,codes = _spectrum()
        raw=SpectrumFrame(0,wavelength,codes,adc_value_kind='direct_adc')
        estimated=replace(raw,adc_value_kind='firmware_rc_estimate')
        self.assertEqual(_session_adc_value_kind([raw,raw]),'direct_adc')
        self.assertEqual(_session_adc_value_kind([estimated,estimated]),'firmware_rc_estimate')
        for frames in ([raw,estimated],[replace(raw,adc_value_kind='unknown')],[]):
            with self.assertRaisesRegex(ValueError,'adc_value_kind'):
                _session_adc_value_kind(frames)

    def test_metrics_and_inference_budget_are_reported(self):
        metrics = self.variable_report["metrics"]
        self.assertGreaterEqual(metrics["contact"]["f1"], 0.90)
        self.assertIsNotNone(metrics["position"]["median_error_mm"])
        self.assertIsNotNone(metrics["position"]["p95_error_mm"])
        self.assertIsNotNone(metrics["area"]["mae_mm2"])
        self.assertIsNotNone(metrics["area"]["p95_abs_error_mm2"])
        self.assertIn("p99_ms", metrics["inference_timing"])
        self.assertEqual(metrics["inference_timing"]["budget_ms_at_15_hz"], 1000 / 15)

    def test_single_eight_mm_area_is_explicitly_refused(self):
        self.assertFalse(self.fixed_bundle.area_model_available)
        reason = self.fixed_bundle.area_unavailable_reason
        self.assertIn("single 8 mm indenter", reason)
        prediction = self.fixed_bundle.predict_vector(
            self.fixed_samples[-1].features, self.signature
        )
        self.assertIsNone(prediction.contact_area_mm2)
        self.assertFalse(prediction.area_model_available)
        self.assertEqual(prediction.area_unavailable_reason, reason)
        with self.assertRaises(AreaNotIdentifiableError):
            train_finger_model(
                self.fixed_samples,
                self.signature,
                config=_training_config(),
                require_area_model=True,
            )

    def test_nominal_tool_face_variation_is_not_called_true_contact_area(self):
        nominal_samples = [
            replace(sample, area_label_source="nominal_indenter")
            if sample.contact
            else sample
            for sample in self.variable_samples
        ]
        bundle, report = train_finger_model(
            nominal_samples, self.signature, config=_training_config()
        )
        self.assertFalse(bundle.area_model_available)
        self.assertIn(
            "nominal tool-face area",
            report["area_policy"]["reason_if_not_trained"],
        )

    def test_measured_name_without_traceable_record_cannot_train_area(self):
        untraceable = [
            replace(
                sample,
                area_label_source="measured",
                area_label_traceable=False,
                area_label_independent=False,
                area_measurement_id=None,
                area_calibration_id=None,
            )
            if sample.contact
            else sample
            for sample in self.variable_samples
        ]
        bundle, report = train_finger_model(
            untraceable, self.signature, config=_training_config()
        )
        self.assertFalse(bundle.area_model_available)
        self.assertIn(
            "traceable measurement and calibration record",
            report["area_policy"]["reason_if_not_trained"],
        )

    def test_bundle_roundtrip_and_strict_runtime_compatibility(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "localizer.joblib"
            self.variable_bundle.save(path)
            loaded = FingerModelBundle.load(path)
            before = self.variable_bundle.predict_vector(
                self.variable_samples[-1].features, self.signature
            )
            after = loaded.predict_vector(
                self.variable_samples[-1].features, self.signature
            )
            self.assertEqual(before, after)

        with self.assertRaisesRegex(ModelCompatibilityError, "table CRC mismatch"):
            self.variable_bundle.predict_vector(
                self.variable_samples[-1].features,
                FingerRuntimeSignature(table_crc32=TEST_CRC + 1, adc_value_kind="direct_adc"),
            )
        with self.assertRaisesRegex(ModelCompatibilityError, "expected CH1"):
            self.variable_bundle.predict_vector(
                self.variable_samples[-1].features,
                FingerRuntimeSignature(table_crc32=TEST_CRC, channel=0, adc_value_kind="direct_adc"),
            )
        with self.assertRaisesRegex(ModelCompatibilityError, "expected 5"):
            self.variable_bundle.predict_vector(
                self.variable_samples[-1].features,
                FingerRuntimeSignature(table_crc32=TEST_CRC, points_per_peak=4, adc_value_kind="direct_adc"),
            )

    def test_prediction_accepts_only_full_three_frame_window(self):
        vector_shape = (3, 9, 6)
        ready = CausalFeatureWindow(
            values=np.zeros(vector_shape),
            quality_masks=np.zeros((3, 9), dtype=np.uint16),
            monotonic_ns=(1, 2, 3),
            sequences=(1, 2, 3),
            history_size=3,
        )
        prediction = self.variable_bundle.predict_window(ready, self.signature)
        self.assertGreaterEqual(prediction.contact_probability, 0.0)
        warmup = CausalFeatureWindow(
            values=np.zeros(vector_shape),
            quality_masks=np.zeros((3, 9), dtype=np.uint16),
            monotonic_ns=(1, 1, 2),
            sequences=(1, 1, 2),
            history_size=2,
        )
        with self.assertRaisesRegex(ValueError, "full three-frame"):
            self.variable_bundle.predict_window(warmup, self.signature)
        invalid_history = np.zeros((3, 9), dtype=np.uint16)
        invalid_history[0, 0] = int(PeakQuality.FRAME_CRC_FAILED)
        corrupt = CausalFeatureWindow(
            values=np.zeros(vector_shape),
            quality_masks=invalid_history,
            monotonic_ns=(1, 2, 3),
            sequences=(1, 2, 3),
            history_size=3,
        )
        with self.assertRaisesRegex(RuntimeError, "26/27"):
            self.variable_bundle.predict_window(corrupt, self.signature)

    def test_raw_session_loader_creates_causal_grouped_samples(self):
        wavelength, baseline = _spectrum()
        _, pressed = _spectrum(shift_nm=0.003)
        with tempfile.TemporaryDirectory() as temporary:
            session_dir = Path(temporary) / "capture"
            with FingerSessionWriter(session_dir, session_id="session-A") as writer:
                writer.write_event(
                    ContactEvent(
                        kind="baseline", event_id="idle-1", monotonic_ns=1
                    )
                )
                for sequence, timestamp in enumerate((2, 3, 4), start=1):
                    writer.write_frame(
                        SpectrumFrame(
                            sequence,
                            wavelength,
                            baseline,
                            monotonic_ns=timestamp,
                            transimpedance_ohm={1: 2000},
                            table_crc32=TEST_CRC,
                            packet_crc_ok=True,
                            adc_value_kind="direct_adc",
                        )
                    )
                writer.write_event(
                    ContactEvent(
                        kind="contact",
                        event_id="press-1",
                        monotonic_ns=5,
                        indenter_diameter_mm=8.0,
                        nominal_contact_area_mm2=math.pi * 16.0,
                        metadata={"model_x_mm": 1.5, "model_y_mm": -2.0},
                    )
                )
                for sequence, timestamp in enumerate((6, 7, 8, 9, 10), start=4):
                    writer.write_frame(
                        SpectrumFrame(
                            sequence,
                            wavelength,
                            pressed,
                            monotonic_ns=timestamp,
                            transimpedance_ohm={1: 2000},
                            table_crc32=TEST_CRC,
                            packet_crc_ok=True,
                            adc_value_kind="direct_adc",
                        )
                    )

            samples, signature, report = load_session_training_samples(session_dir)
            self.assertEqual(signature.table_crc32, TEST_CRC)
            self.assertEqual(signature.point_count, 45)
            self.assertEqual(report.baseline_frames, 3)
            # The three calibration frames cannot also become model history.
            self.assertEqual(report.samples_created, 3)
            self.assertEqual(report.skipped_baseline_calibration, 3)
            contact = [sample for sample in samples if sample.contact]
            self.assertEqual(len(contact), 3)
            self.assertEqual(contact[0].model_x_mm, 1.5)
            self.assertEqual(contact[0].model_y_mm, -2.0)

    def test_deployment_engine_requires_consecutive_crc_pinned_frames(self):
        wavelength, baseline = _spectrum()
        base = SpectrumFrame(
            0,
            wavelength,
            baseline,
            monotonic_ns=1,
            table_crc32=TEST_CRC,
            packet_crc_ok=True,
            device_sequence=0,
            device_boot_id=7,
            adc_value_kind="direct_adc",
        )
        engine = FingerInferenceEngine(self.variable_bundle, base)
        for sequence in (1, 2):
            update = engine.update(
                SpectrumFrame(
                    sequence,
                    wavelength,
                    baseline,
                    monotonic_ns=sequence + 1,
                    table_crc32=TEST_CRC,
                    packet_crc_ok=True,
                    device_sequence=sequence,
                    device_boot_id=7,
                    adc_value_kind="direct_adc",
                )
            )
            self.assertIsNone(update.prediction)
        ready = engine.update(
            SpectrumFrame(
                3,
                wavelength,
                baseline,
                monotonic_ns=4,
                table_crc32=TEST_CRC,
                packet_crc_ok=True,
                device_sequence=3,
                device_boot_id=7,
                adc_value_kind="direct_adc",
            )
        )
        self.assertIsNotNone(ready.prediction)
        gap = engine.update(
            SpectrumFrame(
                5,
                wavelength,
                baseline,
                monotonic_ns=5,
                table_crc32=TEST_CRC,
                packet_crc_ok=True,
                device_sequence=5,
                device_boot_id=7,
                adc_value_kind="direct_adc",
            )
        )
        self.assertEqual(gap.reset_reason, "frame sequence gap")
        self.assertIsNone(gap.prediction)
        with self.assertRaisesRegex(ModelCompatibilityError, "table CRC mismatch"):
            engine.update(
                SpectrumFrame(
                    6,
                    wavelength,
                    baseline,
                    monotonic_ns=6,
                    table_crc32=TEST_CRC + 1,
                    packet_crc_ok=True,
                    device_sequence=6,
                    device_boot_id=7,
                    adc_value_kind="direct_adc",
                )
            )

    def test_full_map_engine_rejects_cached_points_and_discontinuities(self):
        wavelength, codes = _spectrum()
        baseline = SpectrumFrame(0, wavelength, codes, monotonic_ns=1,
                                 table_crc32=TEST_CRC, adc_value_kind="direct_adc")
        for overrides in (
            {"acquisition_profile": "SURVEY"},
            {"fresh_point_mask": [False] * 45},
            {"point_age_frames": [1] * 45},
            {"bandwidth_discontinuity": True},
        ):
            with self.subTest(overrides=overrides):
                engine = FingerInferenceEngine(self.variable_bundle, baseline)
                engine.update(replace(baseline, sequence=1, monotonic_ns=2))
                engine.update(replace(baseline, sequence=2, monotonic_ns=3))
                with self.assertRaises(ModelCompatibilityError):
                    engine.update(replace(baseline, sequence=3, monotonic_ns=4, **overrides))
                self.assertEqual(engine.history_size, 0)
                with self.assertRaises(ModelCompatibilityError):
                    FingerInferenceEngine(self.variable_bundle, replace(baseline, **overrides))


if __name__ == "__main__":
    unittest.main()
