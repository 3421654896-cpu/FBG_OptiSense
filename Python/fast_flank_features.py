"""Causal features for the direct 18-flank CH1 stream.

Two flank samples per FBG are converted to a template-linearized *apparent*
wavelength displacement.  The conversion is useful for repeatability and
localization features but is not synchronized wavelength-meter truth.

An explicit no-contact baseline phase is mandatory.  The baseline is frozen
before a controlled press so slow laser drift cannot be learned from contact
data or make a held press disappear.  No contact-area value is inferred here.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import statistics
from typing import Mapping, Sequence

import numpy as np

from capture_fast_flank_stream import ROWS, WARMUP_CYCLES


SENSOR_COUNT = 9
BASELINE_FRAMES = 64
MINIMUM_FLANK_CODE = 8
MINIMUM_NOISE_SIGMA_PM = 0.10


def _template_pair(rows: Sequence[Mapping], grating: int):
    left_row = grating * 5 + 1
    right_row = grating * 5 + 3
    left = rows[left_row]
    right = rows[right_row]
    return (
        float(left["template_signal_v"]),
        float(left["template_slope_v_per_nm"]),
        float(right["template_signal_v"]),
        float(right["template_slope_v_per_nm"]),
    )


def apparent_shift_pm(frame: Mapping, rows: Sequence[Mapping]) -> np.ndarray:
    """Return nine two-flank template shifts from one validated stream frame."""
    records = tuple(frame["records"])
    if len(records) != len(ROWS) or tuple(int(item["row"]) for item in records) != ROWS:
        raise ValueError("frame does not use the fixed 18-flank route")
    result = np.full(SENSOR_COUNT, np.nan, dtype=float)
    by_row = {int(item["row"]): int(item["second_code"]) for item in records}
    for grating in range(SENSOR_COUNT):
        left_row = grating * 5 + 1
        right_row = grating * 5 + 3
        left_code = by_row[left_row]
        right_code = by_row[right_row]
        if left_code < MINIMUM_FLANK_CODE or right_code < MINIMUM_FLANK_CODE:
            continue
        signal_left, slope_left, signal_right, slope_right = _template_pair(rows, grating)
        ratio = right_code / left_code
        denominator = ratio * slope_left - slope_right
        if not math.isfinite(denominator) or abs(denominator) <= 1e-12:
            continue
        result[grating] = 1000.0 * (signal_right - ratio * signal_left) / denominator
    return result


@dataclass(frozen=True)
class FastFlankFeatureFrame:
    sequence: int
    board_time_us: int
    minimum_warmup_complete: bool
    baseline_ready: bool
    baseline_frozen: bool
    apparent_shift_pm: tuple[float | None, ...]
    relative_shift_pm: tuple[float | None, ...]
    baseline_noise_sigma_pm: tuple[float | None, ...]
    quality_flags: tuple[str, ...]
    physical_contact_area_mm2: None = None


class FastFlankBaseline:
    """Learn only declared no-contact frames, then freeze for one press event."""

    def __init__(self, rows: Sequence[Mapping], baseline_frames: int = BASELINE_FRAMES):
        if int(baseline_frames) < 16:
            raise ValueError("baseline_frames must be at least 16")
        self.rows = tuple(rows)
        self.baseline_frames = int(baseline_frames)
        self._history = deque(maxlen=self.baseline_frames)
        self._reference: np.ndarray | None = None
        self._noise_sigma_pm: np.ndarray | None = None
        self._frozen = False
        self._last_sequence: int | None = None

    @property
    def ready(self) -> bool:
        return self._reference is not None

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def noise_sigma_pm(self) -> tuple[float, ...] | None:
        if self._noise_sigma_pm is None:
            return None
        return tuple(float(value) for value in self._noise_sigma_pm)

    def reset(self):
        self._history.clear()
        self._reference = None
        self._noise_sigma_pm = None
        self._frozen = False
        self._last_sequence = None

    def freeze_for_contact(self):
        if not self.ready:
            raise RuntimeError("cannot freeze before a complete no-contact baseline")
        self._frozen = True

    def resume_no_contact_learning(self):
        self._frozen = False

    def update(self, frame: Mapping, *, declared_no_contact: bool) -> FastFlankFeatureFrame:
        sequence = int(frame["sequence"])
        if self._last_sequence is not None and sequence != self._last_sequence + 1:
            self.reset()
            raise ValueError("stream sequence discontinuity invalidated the baseline")
        self._last_sequence = sequence
        minimum_complete = bool(frame["minimum_warmup_complete"])
        values = apparent_shift_pm(frame, self.rows)
        flags = []
        if not minimum_complete:
            flags.append("below_minimum_warmup")
        if np.any(~np.isfinite(values)):
            flags.append("invalid_or_dark_flank")
        if declared_no_contact and self._frozen:
            raise ValueError("cannot update a frozen contact baseline as no-contact")
        if declared_no_contact and minimum_complete and np.all(np.isfinite(values)):
            self._history.append(values.copy())
            if len(self._history) == self.baseline_frames:
                history = np.stack(tuple(self._history))
                self._reference = np.median(history, axis=0)
                # Consecutive differences suppress slow offset drift.  Divide
                # by sqrt(2) to express the result as one-frame noise.  A
                # small floor retains a non-zero threshold for quantized ADC
                # plateaus; it is not a claimed physical accuracy.
                differences = np.diff(history, axis=0)
                difference_median = np.median(differences, axis=0)
                difference_mad = np.median(
                    np.abs(differences - difference_median), axis=0
                )
                self._noise_sigma_pm = np.maximum(
                    MINIMUM_NOISE_SIGMA_PM,
                    1.4826 * difference_mad / math.sqrt(2.0),
                )
        if not declared_no_contact and not self._frozen:
            flags.append("contact_phase_without_frozen_baseline")
        if self._reference is None:
            relative = np.full(SENSOR_COUNT, np.nan)
            flags.append("baseline_not_ready")
        else:
            relative = values - self._reference
        return FastFlankFeatureFrame(
            sequence=sequence,
            board_time_us=int(frame["cycle_start_us"]),
            minimum_warmup_complete=minimum_complete,
            baseline_ready=self.ready,
            baseline_frozen=self.frozen,
            apparent_shift_pm=tuple(float(v) if math.isfinite(v) else None for v in values),
            relative_shift_pm=tuple(float(v) if math.isfinite(v) else None for v in relative),
            baseline_noise_sigma_pm=(
                (None,) * SENSOR_COUNT
                if self._noise_sigma_pm is None
                else tuple(float(value) for value in self._noise_sigma_pm)
            ),
            quality_flags=tuple(flags),
        )


def _tone_amplitude(times_s: np.ndarray, values: np.ndarray, frequency_hz: float) -> float:
    """Least-squares tone after removing constant, linear and quadratic drift."""
    centered = times_s - float(np.mean(times_s))
    omega = 2.0 * math.pi * float(frequency_hz)
    design = np.column_stack(
        (
            np.ones_like(centered),
            centered,
            centered * centered,
            np.sin(omega * centered),
            np.cos(omega * centered),
        )
    )
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return float(math.hypot(coefficients[-2], coefficients[-1]))


def audit_static_stream(
    frames: Sequence[Mapping], rows: Sequence[Mapping], *, tone_hz: float = 15.0
) -> dict:
    """Quantify static false response; never claim a driven transfer function."""
    usable = [frame for frame in frames if frame["minimum_warmup_complete"]]
    if len(usable) < 96:
        raise ValueError("at least 96 post-minimum-warm-up frames are required")
    times = np.asarray([int(frame["cycle_start_us"]) for frame in usable], dtype=float) / 1e6
    shifts = np.stack([apparent_shift_pm(frame, rows) for frame in usable])
    if np.any(~np.isfinite(shifts)):
        raise ValueError("static audit requires nine finite flank pairs")
    early = shifts[:32]
    late = shifts[-32:]
    result = {}
    for grating in range(SENSOR_COUNT):
        differences = np.diff(shifts[:, grating])
        median_difference = float(np.median(differences))
        mad = float(np.median(np.abs(differences - median_difference)))
        result[f"G{grating + 1}"] = {
            "early_median_pm": float(np.median(early[:, grating])),
            "late_median_pm": float(np.median(late[:, grating])),
            "early_to_late_drift_pm": float(
                np.median(late[:, grating]) - np.median(early[:, grating])
            ),
            "one_frame_difference_robust_sigma_pm": 1.4826 * mad,
            f"static_fitted_{tone_hz:g}hz_amplitude_pm": _tone_amplitude(
                times, shifts[:, grating], tone_hz
            ),
        }
    return {
        "schema": "fast_flank_static_false_response_v1",
        "tone_hz": float(tone_hz),
        "post_minimum_warmup_frames": len(usable),
        "per_grating": result,
        "physical_15hz_verified": False,
        "contact_area_verified": False,
        "note": (
            "Static fitted tone is a false-response floor, not sensitivity. "
            "A synchronized driven pressure reference is still required."
        ),
    }


__all__ = [
    "BASELINE_FRAMES",
    "FastFlankBaseline",
    "FastFlankFeatureFrame",
    "MINIMUM_NOISE_SIGMA_PM",
    "apparent_shift_pm",
    "audit_static_stream",
]
