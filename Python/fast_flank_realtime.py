"""Causal nine-FBG localization bridge for the fixed 18-flank stream.

This module intentionally does not reuse the trained 45-point model: two
direct ADC flank samples per grating are a different acquisition contract.
The output is a provisional, geometry-weighted position until labelled press
events prove its accuracy.  True physical contact area is never synthesized
from sensor spread or the nominal 8 mm tool face.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from ch1_contact_localizer import (
    CH1NinePeakLocalizer,
    ContactEstimate,
    ContactLocalizerConfig,
    load_contact_geometry,
)
from fast_flank_features import BASELINE_FRAMES, FastFlankBaseline


SENSOR_COUNT = 9
MINIMUM_ACTIVATION_PM = 5.0
NOISE_SIGMA_MULTIPLIER = 6.0
MINIMUM_CHANGE_ACTIVATION_PM = 3.0
# One isolated FBG must exceed its adaptive change threshold by 2.5x.  The
# lower 1.5x path remains available only when two sensors change together.
# This rejects observed single-code G9 spikes without imposing a two-frame
# temporal delay that would consume most of a 15 Hz half-cycle.
CHANGE_SINGLE_STRONG_SCORE = 2.5
CHANGE_SPATIAL_SCORE = 1.5
FIFTEEN_HZ_NYQUIST_INTERVAL_US = 1_000_000.0 / 30.0
CADENCE_WINDOW = 32
# A single-frame branch preserves the fastest possible 15 Hz onset response.
# The eight-frame branch catches a mechanically slow contact ramp without
# converting the frozen absolute baseline into a permanent contact flag.
CHANGE_WINDOWS_FRAMES = (1, 8)


@dataclass(frozen=True)
class FastFlankRealtimeResult:
    sequence: int
    board_time_us: int
    state: str
    baseline_ready: bool
    baseline_frozen: bool
    contact_detected: bool
    detection_kind: str
    provisional_x_mm: float | None
    provisional_y_mm: float | None
    position_coordinate_frame: str
    geometry_calibrated: bool
    position_accuracy_verified: bool
    confidence: float
    dominant_sensor_id: int | None
    relative_shift_pm: tuple[float | None, ...]
    one_frame_delta_pm: tuple[float | None, ...]
    activation_threshold_pm: tuple[float | None, ...]
    cadence_hz: float | None
    sampling_gate_for_15hz: bool
    physical_15hz_verified: bool
    physical_contact_area_mm2: None = None
    contact_area_verified: bool = False
    quality_flags: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "sequence": self.sequence,
            "board_time_us": self.board_time_us,
            "state": self.state,
            "baseline_ready": self.baseline_ready,
            "baseline_frozen": self.baseline_frozen,
            "contact_detected": self.contact_detected,
            "detection_kind": self.detection_kind,
            "position": {
                "x_mm": self.provisional_x_mm,
                "y_mm": self.provisional_y_mm,
                "coordinate_frame": self.position_coordinate_frame,
                "geometry_calibrated": self.geometry_calibrated,
                "accuracy_verified": self.position_accuracy_verified,
                "provisional": True,
            },
            "confidence": self.confidence,
            "dominant_sensor_id": self.dominant_sensor_id,
            "relative_shift_pm": list(self.relative_shift_pm),
            "one_frame_delta_pm": list(self.one_frame_delta_pm),
            "activation_threshold_pm": list(self.activation_threshold_pm),
            "cadence_hz": self.cadence_hz,
            "sampling_gate_for_15hz": self.sampling_gate_for_15hz,
            "physical_15hz_verified": self.physical_15hz_verified,
            "physical_contact_area_mm2": None,
            "contact_area_verified": False,
            "quality_flags": list(self.quality_flags),
        }


class FastFlankRealtimeLocalizer:
    """Fail-closed state machine for baseline and provisional XY inference."""

    def __init__(
        self,
        rows: Sequence[Mapping],
        *,
        baseline_frames: int = BASELINE_FRAMES,
        minimum_activation_pm: float = MINIMUM_ACTIVATION_PM,
        noise_sigma_multiplier: float = NOISE_SIGMA_MULTIPLIER,
    ) -> None:
        if not math.isfinite(minimum_activation_pm) or minimum_activation_pm <= 0.0:
            raise ValueError("minimum_activation_pm must be finite and positive")
        if not math.isfinite(noise_sigma_multiplier) or noise_sigma_multiplier < 3.0:
            raise ValueError("noise_sigma_multiplier must be at least 3")
        self.baseline = FastFlankBaseline(rows, baseline_frames=baseline_frames)
        self.minimum_activation_pm = float(minimum_activation_pm)
        self.noise_sigma_multiplier = float(noise_sigma_multiplier)
        geometry = load_contact_geometry()
        self.localizer = CH1NinePeakLocalizer(
            geometry,
            config=ContactLocalizerConfig(
                activation_pm=geometry.activation_pm,
                full_scale_pm=geometry.full_scale_pm,
                remove_common_mode=True,
            ),
        )
        self._intervals_us: deque[int] = deque(maxlen=CADENCE_WINDOW)
        self._last_board_time_us: int | None = None
        self._previous_relative_pm: np.ndarray | None = None
        self._relative_history: deque[np.ndarray] = deque(
            maxlen=max(CHANGE_WINDOWS_FRAMES)
        )

    def reset(self) -> None:
        self.baseline.reset()
        self._intervals_us.clear()
        self._last_board_time_us = None
        self._previous_relative_pm = None
        self._relative_history.clear()

    def freeze_for_press(self) -> None:
        self.baseline.freeze_for_contact()

    def resume_no_contact_learning(self) -> None:
        self.baseline.resume_no_contact_learning()

    def _observe_cadence(self, board_time_us: int) -> None:
        if self._last_board_time_us is not None:
            interval = int(board_time_us) - self._last_board_time_us
            if interval <= 0:
                self.reset()
                raise ValueError("nonmonotonic board time invalidated the realtime state")
            self._intervals_us.append(interval)
        self._last_board_time_us = int(board_time_us)

    def _cadence(self) -> tuple[float | None, bool]:
        if len(self._intervals_us) < CADENCE_WINDOW:
            return None, False
        worst = max(self._intervals_us)
        return 1_000_000.0 / float(np.median(self._intervals_us)), bool(
            worst <= FIFTEEN_HZ_NYQUIST_INTERVAL_US
        )

    def update(
        self, frame: Mapping, *, declared_no_contact: bool
    ) -> FastFlankRealtimeResult:
        # The baseline object checks sequence continuity.  Cadence is observed
        # only after the frame itself has passed that contract.
        try:
            feature = self.baseline.update(
                frame, declared_no_contact=declared_no_contact
            )
        except ValueError:
            # Bad phase declarations and sequence breaks are both unsafe for
            # causal inference.  Drop cadence and baseline together so an old
            # timing window cannot make the next event look certified.
            self.reset()
            raise
        self._observe_cadence(feature.board_time_us)
        cadence_hz, sampling_gate = self._cadence()
        flags = list(feature.quality_flags)
        state = "learning_no_contact_baseline"
        estimate: ContactEstimate | None = None
        thresholds: tuple[float | None, ...] = (None,) * SENSOR_COUNT
        delta_tuple: tuple[float | None, ...] = (None,) * SENSOR_COUNT

        if feature.baseline_ready and not feature.baseline_frozen:
            state = "baseline_ready_waiting_for_freeze"
        elif feature.baseline_frozen:
            state = "pressure_change_monitoring"
            relative = np.asarray(
                [math.nan if value is None else value for value in feature.relative_shift_pm],
                dtype=float,
            )
            noise = np.asarray(
                [
                    math.nan if value is None else value
                    for value in feature.baseline_noise_sigma_pm
                ],
                dtype=float,
            )
            valid = np.isfinite(relative) & np.isfinite(noise)
            threshold_values = np.maximum(
                MINIMUM_CHANGE_ACTIVATION_PM,
                self.noise_sigma_multiplier * noise,
            )
            thresholds = tuple(
                float(value) if finite else None
                for value, finite in zip(threshold_values, valid)
            )
            if np.count_nonzero(valid) < 3:
                flags.append("insufficient_valid_fast_flanks")
            elif not self._relative_history:
                flags.append("pressure_change_history_not_ready")
            elif "contact_phase_without_frozen_baseline" not in flags:
                one_frame_delta = relative - self._relative_history[-1]
                one_frame_valid = valid & np.isfinite(one_frame_delta)
                one_frame_delta[one_frame_valid] -= float(
                    np.median(one_frame_delta[one_frame_valid])
                )
                delta_tuple = tuple(
                    float(value) if finite else None
                    for value, finite in zip(one_frame_delta, one_frame_valid)
                )

                qualifying = []
                for window_frames in CHANGE_WINDOWS_FRAMES:
                    if len(self._relative_history) < window_frames:
                        continue
                    delta = relative - self._relative_history[-window_frames]
                    delta_valid = valid & np.isfinite(delta)
                    if np.count_nonzero(delta_valid) < 3:
                        continue
                    delta[delta_valid] -= float(np.median(delta[delta_valid]))
                    scores = np.zeros(SENSOR_COUNT, dtype=float)
                    scores[delta_valid] = (
                        np.abs(delta[delta_valid]) / threshold_values[delta_valid]
                    )
                    ordered = np.sort(scores[delta_valid])
                    strong_single = bool(
                        ordered.size and ordered[-1] >= CHANGE_SINGLE_STRONG_SCORE
                    )
                    spatial_candidates = np.flatnonzero(
                        scores >= CHANGE_SPATIAL_SCORE
                    )
                    spatial_pair = any(
                        max(
                            abs(first // 3 - second // 3),
                            abs(first % 3 - second % 3),
                        )
                        <= 1
                        for offset, first in enumerate(spatial_candidates)
                        for second in spatial_candidates[offset + 1 :]
                    )
                    if strong_single or spatial_pair:
                        qualifying.append(
                            (
                                float(ordered[-1]),
                                window_frames,
                                delta_valid,
                                scores,
                            )
                        )
                if qualifying:
                    _, selected_window, delta_valid, scores = max(
                        qualifying, key=lambda candidate: candidate[0]
                    )
                    if selected_window > 1:
                        flags.append(
                            f"pressure_change_accumulated_{selected_window}_frames"
                        )
                    # Localization weights are the newly observed causal
                    # change. Absolute frozen-baseline shift is retained as
                    # audit data but is not allowed to turn slow drift into a
                    # persistent contact declaration.
                    estimate = self.localizer.estimate_peak_shifts(
                        relative,
                        timestamp_ns=feature.board_time_us * 1000,
                        valid_mask=delta_valid,
                        fresh_peak_mask=delta_valid,
                        peak_age_frames=np.zeros(SENSOR_COUNT),
                        peak_uncertainty_pm=noise,
                        peak_response_scores=scores,
                    )
                else:
                    flags.append("below_pressure_change_onset")

        current_relative = np.asarray(
            [math.nan if value is None else value for value in feature.relative_shift_pm],
            dtype=float,
        )
        self._previous_relative_pm = current_relative
        self._relative_history.append(current_relative.copy())

        if not sampling_gate:
            flags.append("15hz_sampling_gate_not_ready_or_failed")
        if estimate is None:
            contact = False
            x_value = y_value = None
            confidence = 0.0
            dominant = None
        else:
            contact = bool(estimate.contact_detected)
            x_value = estimate.x_mm
            y_value = estimate.y_mm
            confidence = float(estimate.confidence)
            dominant = estimate.dominant_sensor_id
            flags.extend(estimate.quality_flags)

        return FastFlankRealtimeResult(
            sequence=feature.sequence,
            board_time_us=feature.board_time_us,
            state=state,
            baseline_ready=feature.baseline_ready,
            baseline_frozen=feature.baseline_frozen,
            contact_detected=contact,
            detection_kind="pressure_change_onset_not_static_contact_state",
            provisional_x_mm=x_value,
            provisional_y_mm=y_value,
            position_coordinate_frame=self.localizer.geometry.coordinate_frame,
            geometry_calibrated=self.localizer.geometry.position_calibrated,
            position_accuracy_verified=False,
            confidence=confidence,
            dominant_sensor_id=dominant,
            relative_shift_pm=feature.relative_shift_pm,
            one_frame_delta_pm=delta_tuple,
            activation_threshold_pm=thresholds,
            cadence_hz=cadence_hz,
            sampling_gate_for_15hz=sampling_gate,
            physical_15hz_verified=False,
            quality_flags=tuple(dict.fromkeys(flags)),
        )


__all__ = [
    "CADENCE_WINDOW",
    "FastFlankRealtimeLocalizer",
    "FastFlankRealtimeResult",
]
