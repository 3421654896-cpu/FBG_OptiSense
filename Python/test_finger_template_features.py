import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from adaptive_sampling_optimizer import optimize_ch1_sampling
from finger_dataset import SpectrumFrame
from finger_features import PeakQuality, extract_ch1_peak_features
from finger_inference import FingerInferenceEngine
from finger_model import FingerModelError, ModelCompatibilityError
from finger_sampling_template import (
    FingerSamplingTemplate,
    load_finger_sampling_template,
    wavelength_table_crc32,
)


def _teacher_and_template():
    x = np.linspace(1525.0, 1561.0, 3601)
    y = np.full_like(x, 0.035)
    regions = []
    for number in range(9):
        center = 1527.0 + 4.0 * number
        relative = x - center
        # Intentionally non-Gaussian: unequal flanks, a shoulder and ripple.
        y += 0.46 * np.exp(-0.5 * (relative / 0.145) ** 2)
        y += 0.15 * np.exp(-0.5 * ((relative - 0.14) / 0.075) ** 2)
        y += 0.012 * np.sin(52.0 * relative) * np.exp(-0.5 * (relative / 0.31) ** 2)
        regions.append((center - 0.62, center + 0.62))
    plan = optimize_ch1_sampling(x, y, regions)
    return x, y, FingerSamplingTemplate(plan)


def _codes(voltage):
    return tuple(
        np.clip(np.rint(np.asarray(voltage) * 4096.0 / 2.5), 0, 4095)
        .astype(int)
        .tolist()
    )


def _signals(template, shift_nm):
    values = []
    for peak in template.plan.peaks:
        values.extend(
            np.interp(
                peak.selected_wavelength_nm - shift_nm,
                peak.dense_wavelength_nm,
                peak.dense_signal,
            )
        )
    return values


def _v4_report(template):
    peaks = []
    rows = []
    for peak in template.plan.peaks:
        peak_number = int(peak.peak_number)
        peaks.append(
            {
                "peak_number": peak_number,
                "center_nm": float(
                    peak.dense_wavelength_nm[int(np.argmax(peak.dense_signal))]
                ),
                "region_start_nm": float(peak.dense_wavelength_nm[0]),
                "region_stop_nm": float(peak.dense_wavelength_nm[-1]),
                "teacher_peak_fullband_index": int(peak.peak_index),
                "selected_fullband_indices": [
                    int(value) for value in peak.selected_indices
                ],
                "projected_shift_information": float(peak.projected_shift_information),
                "dense_wavelength_nm": [
                    float(value) for value in peak.dense_wavelength_nm
                ],
                "dense_template_signal_v": [
                    float(value) for value in peak.dense_signal
                ],
                "selected_template_signal_v": [
                    float(value) for value in peak.selected_template_signal
                ],
                "selected_template_slope_v_per_nm": [
                    float(value) for value in peak.selected_template_slope
                ],
                "selected_reliability_weight": [
                    float(value) for value in peak.selected_reliability_weight
                ],
            }
        )
        for index, wavelength, signal, slope, weight in zip(
            peak.selected_indices,
            peak.selected_wavelength_nm,
            peak.selected_template_signal,
            peak.selected_template_slope,
            peak.selected_reliability_weight,
            strict=True,
        ):
            rows.append(
                {
                    "peak_number": peak_number,
                    "fullband_index": int(index),
                    "target_wavelength_nm": float(wavelength),
                    "measured_wavelength_nm": float(wavelength),
                    "codes": [1, 2, 3, 4, 5],
                    "template_signal_v": float(signal),
                    "template_slope_v_per_nm": float(slope),
                    "template_reliability_weight": float(weight),
                }
            )
    canonical = [
        {
            "mode_index": mode_index,
            "peak_number": row["peak_number"],
            "fullband_index": row["fullband_index"],
            "codes": row["codes"],
        }
        for mode_index, row in enumerate(rows)
    ]
    fingerprint = hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    return {
        "schema": "equal_interval_auto_mode_selection_v4",
        "detected_peak_count": 9,
        "configured_peak_count": 9,
        "fbg_channels": [1],
        "selection": {"method": "measured_template_fisher_v1"},
        "stress": {
            "point_count": 45,
            "points_per_peak": [5] * 9,
            "peaks": peaks,
            "rows": rows,
        },
        "sparse_sequence_validation": {
            "status": "passed",
            "stress_table_fingerprint_sha256": fingerprint,
        },
    }


class _NoOpBundle:
    def assert_compatible(self, runtime):
        runtime.validate_finger_layout()

    def predict_window(self, window, runtime):
        return "prediction"


class FingerTemplateFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, _, cls.template = _teacher_and_template()
        cls.baseline_codes = _codes(_signals(cls.template, 0.0))

    def test_non_gaussian_template_recovers_three_pm_shift(self):
        shifted_codes = _codes(_signals(self.template, 0.003))
        result = extract_ch1_peak_features(
            self.template.wavelengths_nm,
            shifted_codes,
            baseline_adc_codes=self.baseline_codes,
            sampling_template=self.template,
        )
        self.assertTrue(np.all(result.valid_mask))
        for peak in result.peaks:
            self.assertAlmostEqual(peak.center_shift_pm, 3.0, delta=0.20)
            self.assertTrue(np.isfinite(peak.center_uncertainty_pm))
            self.assertLess(peak.center_uncertainty_pm, 5.0)
            self.assertEqual(peak.template_fit_reason, "current:ok;baseline:ok")
            self.assertEqual(len(peak.normalized_profile), 5)

    def test_v4_report_load_and_serialized_roundtrip_are_crc_bound(self):
        loaded = load_finger_sampling_template(_v4_report(self.template))
        self.assertEqual(loaded.point_count, 45)
        self.assertEqual(loaded.channel, 1)
        self.assertEqual(loaded.grating_count, 9)
        self.assertEqual(
            loaded.table_crc32,
            wavelength_table_crc32(loaded.wavelengths_nm),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = loaded.save(Path(temporary) / "template.json")
            roundtrip = load_finger_sampling_template(path)
        self.assertEqual(roundtrip.table_crc32, loaded.table_crc32)
        np.testing.assert_allclose(
            roundtrip.plan.peaks[1].dense_signal,
            loaded.plan.peaks[1].dense_signal,
        )

    def test_v4_report_rejects_tampered_dac_fingerprint(self):
        report = _v4_report(self.template)
        report["stress"]["rows"][8]["codes"][2] += 1
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            load_finger_sampling_template(report)

    def test_flat_measurement_has_explicit_template_reject_quality(self):
        flat_codes = _codes(np.full(45, 0.20))
        result = extract_ch1_peak_features(
            self.template.wavelengths_nm,
            flat_codes,
            baseline_adc_codes=self.baseline_codes,
            sampling_template=self.template,
        )
        self.assertFalse(np.any(result.valid_mask))
        for peak in result.peaks:
            quality = PeakQuality(peak.quality_mask)
            self.assertTrue(quality & PeakQuality.TEMPLATE_FIT_REJECTED)
            self.assertNotEqual(peak.template_fit_reason, "current:ok;baseline:ok")

    def test_inference_rejects_bad_fit_and_clears_causal_history(self):
        crc = int(self.template.table_crc32)
        baseline = SpectrumFrame(
            0,
            self.template.wavelengths_nm,
            self.baseline_codes,
            monotonic_ns=1,
            table_crc32=crc,
            packet_crc_ok=True,
            device_sequence=0,
            device_boot_id=7,
            adc_value_kind="direct_adc",
        )
        engine = FingerInferenceEngine(
            _NoOpBundle(), baseline, sampling_template=self.template
        )
        for sequence in (1, 2):
            engine.update(
                SpectrumFrame(
                    sequence,
                    self.template.wavelengths_nm,
                    self.baseline_codes,
                    monotonic_ns=sequence + 1,
                    table_crc32=crc,
                    packet_crc_ok=True,
                    device_sequence=sequence,
                    device_boot_id=7,
                    adc_value_kind="direct_adc",
                )
            )
        self.assertEqual(engine.history_size, 2)
        with self.assertRaisesRegex(FingerModelError, "causal history was reset"):
            engine.update(
                SpectrumFrame(
                    3,
                    self.template.wavelengths_nm,
                    _codes(np.full(45, 0.20)),
                    monotonic_ns=4,
                    table_crc32=crc,
                    packet_crc_ok=True,
                    device_sequence=3,
                    device_boot_id=7,
                    adc_value_kind="direct_adc",
                )
            )
        self.assertEqual(engine.history_size, 0)

    def test_inference_rejects_template_table_crc_mismatch(self):
        crc = (int(self.template.table_crc32) + 1) & 0xFFFFFFFF
        baseline = SpectrumFrame(
            0,
            self.template.wavelengths_nm,
            self.baseline_codes,
            monotonic_ns=1,
            table_crc32=crc,
            packet_crc_ok=True,
            adc_value_kind="direct_adc",
        )
        with self.assertRaisesRegex(ModelCompatibilityError, "CRC mismatch"):
            FingerInferenceEngine(
                _NoOpBundle(), baseline, sampling_template=self.template
            )


if __name__ == "__main__":
    unittest.main()
