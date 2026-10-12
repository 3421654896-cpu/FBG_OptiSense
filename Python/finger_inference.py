"""Small stateful deployment wrapper for the offline finger model.

This module connects captured :class:`SpectrumFrame` objects to feature
extraction and model inference.  It has no acquisition or machine-control code.
Sequence gaps, boot changes, invalid peaks, and table changes reset the causal
history so non-consecutive frames can never be presented as a three-frame
window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from finger_dataset import (
    DEFAULT_GRATING_COUNT,
    DEFAULT_POINTS_PER_PEAK,
    DEFAULT_SENSOR_CHANNEL,
    SpectrumFrame,
    is_complete_fresh_map,
)
from finger_features import (
    CausalFeatureExtractor,
    PeakQuality,
    extract_ch1_peak_features,
)
from finger_model import (
    DEFAULT_POINT_COUNT,
    FingerModelBundle,
    FingerModelError,
    FingerPrediction,
    FingerRuntimeSignature,
    ModelCompatibilityError,
)
from finger_sampling_template import (
    FingerSamplingTemplate,
    load_finger_sampling_template,
)


@dataclass(frozen=True)
class InferenceUpdate:
    prediction: FingerPrediction | None
    causal_frames_ready: int
    reset_reason: str | None = None


class FingerInferenceEngine:
    """Validate and transform consecutive CH1 frames for one model bundle."""

    def __init__(
        self,
        bundle: FingerModelBundle,
        baseline_frame: SpectrumFrame,
        *,
        sampling_template=None,
    ):
        self.bundle = bundle
        self.baseline_frame = baseline_frame
        self._validate_frame_layout(baseline_frame, require_crc=True)
        if baseline_frame.packet_crc_ok is False:
            raise FingerModelError("baseline frame packet CRC failed")
        self.signature = FingerRuntimeSignature(
            table_crc32=baseline_frame.table_crc32,
            channel=DEFAULT_SENSOR_CHANNEL,
            grating_count=DEFAULT_GRATING_COUNT,
            points_per_peak=DEFAULT_POINTS_PER_PEAK,
            adc_value_kind=baseline_frame.adc_value_kind,
        )
        bundle.assert_compatible(self.signature)
        self.sampling_template: FingerSamplingTemplate | None = None
        if sampling_template is not None:
            try:
                self.sampling_template = load_finger_sampling_template(
                    sampling_template,
                    expected_table_crc32=baseline_frame.table_crc32,
                )
            except (OSError, TypeError, ValueError) as exc:
                raise ModelCompatibilityError(
                    f"invalid measured sampling template: {exc}"
                ) from exc
            if not self.sampling_template.wavelength_axis_matches(
                baseline_frame.wavelengths_nm
            ):
                raise ModelCompatibilityError(
                    "sampling-template wavelengths/order differ from the baseline table"
                )
        baseline_features = extract_ch1_peak_features(
            baseline_frame,
            baseline_frame=baseline_frame,
            sampling_template=self.sampling_template,
        )
        if not bool(np.all(baseline_features.valid_mask)):
            raise FingerModelError(
                "baseline frame does not contain nine valid CH1 peak segments"
            )
        self._wavelengths = tuple(baseline_frame.wavelengths_nm)
        self._extractor = CausalFeatureExtractor()
        self._history_size = 0
        self._last_sequence: int | None = None
        self._last_boot_id: int | None = baseline_frame.device_boot_id

    @property
    def history_size(self) -> int:
        return self._history_size

    def reset(self) -> None:
        self._extractor.reset()
        self._history_size = 0
        self._last_sequence = None

    @staticmethod
    def _validate_frame_layout(
        frame: SpectrumFrame, *, require_crc: bool
    ) -> None:
        if not is_complete_fresh_map(frame):
            raise ModelCompatibilityError("full-MAP model cannot consume sparse/cached points")
        if frame.bandwidth_discontinuity:
            raise ModelCompatibilityError("frame acquisition discontinuity; fresh history required")
        if frame.adc_value_kind not in {"direct_adc", "firmware_rc_estimate"}:
            raise ModelCompatibilityError("runtime adc_value_kind is unknown; inference refused")
        if DEFAULT_SENSOR_CHANNEL not in frame.raw_adc_codes:
            raise ModelCompatibilityError("runtime frame does not contain CH1")
        if len(frame.wavelengths_nm) != DEFAULT_POINT_COUNT:
            raise ModelCompatibilityError(
                f"runtime frame has {len(frame.wavelengths_nm)} points; expected 45"
            )
        if len(frame.raw_adc_codes[DEFAULT_SENSOR_CHANNEL]) != DEFAULT_POINT_COUNT:
            raise ModelCompatibilityError("runtime CH1 ADC array does not contain 45 points")
        if require_crc and frame.table_crc32 is None:
            raise ModelCompatibilityError("runtime frame does not provide table CRC")

    @staticmethod
    def _stream_sequence(frame: SpectrumFrame) -> int:
        return int(
            frame.device_sequence
            if frame.device_sequence is not None
            else frame.sequence
        )

    def update(self, frame: SpectrumFrame) -> InferenceUpdate:
        try:
            self._validate_frame_layout(frame, require_crc=True)
        except (ModelCompatibilityError, ValueError):
            self.reset()
            raise
        if frame.adc_value_kind != self.baseline_frame.adc_value_kind:
            self.reset()
            raise ModelCompatibilityError("runtime adc_value_kind differs from baseline; causal history reset")
        runtime = FingerRuntimeSignature(table_crc32=frame.table_crc32,
                                         adc_value_kind=frame.adc_value_kind)
        try:
            self.bundle.assert_compatible(runtime)
        except ModelCompatibilityError:
            self.reset()
            raise
        if tuple(frame.wavelengths_nm) != self._wavelengths:
            self.reset()
            raise ModelCompatibilityError(
                "runtime wavelength values/order differ from the baseline table"
            )
        if frame.packet_crc_ok is False:
            self.reset()
            raise FingerModelError("packet CRC failed; causal history was reset")

        sequence = self._stream_sequence(frame)
        reset_reason = None
        if (
            self._last_boot_id is not None
            and frame.device_boot_id is not None
            and frame.device_boot_id != self._last_boot_id
        ):
            self.reset()
            reset_reason = "device boot changed"
        elif self._last_sequence is not None:
            expected = (self._last_sequence + 1) & 0xFFFFFFFF
            if sequence != expected:
                self.reset()
                reset_reason = "frame sequence gap"
        self._last_sequence = sequence
        if frame.device_boot_id is not None:
            self._last_boot_id = frame.device_boot_id

        feature_frame = extract_ch1_peak_features(
            frame,
            baseline_frame=self.baseline_frame,
            sampling_template=self.sampling_template,
        )
        if not bool(np.all(feature_frame.valid_mask)):
            masks = [
                f"G{index + 1}=0x{int(mask):04X}"
                for index, mask in enumerate(feature_frame.quality_masks)
                if PeakQuality(int(mask)) != PeakQuality.GOOD
            ]
            self.reset()
            raise FingerModelError(
                "invalid CH1 peak features; causal history was reset: "
                + ", ".join(masks)
            )
        window = self._extractor.update(feature_frame)
        self._history_size = window.history_size
        if not window.ready:
            return InferenceUpdate(None, self._history_size, reset_reason)
        prediction = self.bundle.predict_window(window, runtime)
        return InferenceUpdate(prediction, self._history_size, reset_reason)


__all__ = ["FingerInferenceEngine", "InferenceUpdate"]
