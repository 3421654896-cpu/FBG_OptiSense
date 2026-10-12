import unittest

import numpy as np

from finger_dataset import SpectrumFrame
from finger_features import (
    CausalFeatureExtractor,
    PeakQuality,
    extract_ch1_peak_features,
)


def synthetic_spectrum(shift_nm=0.0, amplitude_scale=1.0):
    wavelengths = []
    volts = []
    for peak in range(9):
        center = 1526.0 + peak * 4.0
        local = center + np.asarray((-0.020, -0.010, 0.0, 0.010, 0.020))
        shape = 0.05 + 0.60 * amplitude_scale * np.exp(
            -0.5 * ((local - center - shift_nm) / 0.010) ** 2
        )
        wavelengths.extend(local.tolist())
        volts.extend(shape.tolist())
    codes = np.rint(np.asarray(volts) * 4096.0 / 2.5).astype(int)
    return tuple(wavelengths), tuple(codes.tolist())


class FingerFeatureTests(unittest.TestCase):
    def test_extracts_exactly_nine_ch1_peaks_and_shift(self):
        wavelength, baseline = synthetic_spectrum()
        _, shifted = synthetic_spectrum(shift_nm=0.003)
        result = extract_ch1_peak_features(
            wavelength, shifted, baseline_adc_codes=baseline
        )
        self.assertEqual(result.channel, 1)
        self.assertEqual(len(result.peaks), 9)
        for peak in result.peaks:
            self.assertTrue(peak.valid)
            self.assertAlmostEqual(peak.center_shift_pm, 3.0, delta=1.0)
            self.assertEqual(len(peak.normalized_profile), 5)

    def test_digital_gain_is_audit_only(self):
        wavelength, codes = synthetic_spectrum(shift_nm=0.002)
        _, baseline = synthetic_spectrum()
        frame_a = SpectrumFrame(
            1, wavelength, codes, monotonic_ns=1, digital_gain={1: 1.0}
        )
        frame_b = SpectrumFrame(
            2, wavelength, codes, monotonic_ns=2, digital_gain={1: 99.0}
        )
        result_a = extract_ch1_peak_features(frame_a, baseline_adc_codes=baseline)
        result_b = extract_ch1_peak_features(frame_b, baseline_adc_codes=baseline)
        np.testing.assert_allclose(result_a.values, result_b.values)
        self.assertEqual(result_a.digital_gain_audit, 1.0)
        self.assertEqual(result_b.digital_gain_audit, 99.0)

    def test_transimpedance_is_used_for_physical_amplitude_ratio(self):
        wavelength, baseline = synthetic_spectrum(amplitude_scale=0.5)
        _, doubled_voltage = synthetic_spectrum(amplitude_scale=1.0)
        result = extract_ch1_peak_features(
            wavelength,
            doubled_voltage,
            baseline_adc_codes=baseline,
            transimpedance_ohm=40000,
            baseline_transimpedance_ohm=20000,
        )
        for peak in result.peaks:
            self.assertAlmostEqual(peak.amplitude_relative, 0.0, delta=0.02)

    def test_three_frame_window_is_strictly_causal(self):
        wavelength, baseline = synthetic_spectrum()
        extractor = CausalFeatureExtractor()
        windows = []
        for sequence, shift in enumerate((0.0, 0.001, 0.003), start=1):
            _, codes = synthetic_spectrum(shift_nm=shift)
            frame = SpectrumFrame(
                sequence, wavelength, codes, monotonic_ns=sequence * 10
            )
            features = extract_ch1_peak_features(frame, baseline_adc_codes=baseline)
            windows.append(extractor.update(features))
        self.assertFalse(windows[0].ready)
        self.assertFalse(windows[1].ready)
        self.assertTrue(windows[2].ready)
        self.assertEqual(windows[2].values.shape, (3, 9, 6))
        self.assertEqual(windows[2].model_vector().shape, (9 * 18,))
        self.assertEqual(windows[2].sequences, (1, 2, 3))
        self.assertGreater(windows[2].first_difference[0, 0], 0.0)

    def test_crc_failure_is_present_in_each_peak_quality_mask(self):
        wavelength, codes = synthetic_spectrum()
        frame = SpectrumFrame(
            1,
            wavelength,
            codes,
            monotonic_ns=1,
            packet_crc_ok=False,
        )
        result = extract_ch1_peak_features(frame, baseline_adc_codes=codes)
        for peak in result.peaks:
            self.assertTrue(
                PeakQuality(peak.quality_mask) & PeakQuality.FRAME_CRC_FAILED
            )
            self.assertFalse(peak.valid)

    def test_rejects_eight_peak_layout(self):
        wavelength, codes = synthetic_spectrum()
        with self.assertRaisesRegex(ValueError, "exactly 9"):
            extract_ch1_peak_features(
                wavelength, codes, baseline_adc_codes=codes, grating_count=8
            )


if __name__ == "__main__":
    unittest.main()
