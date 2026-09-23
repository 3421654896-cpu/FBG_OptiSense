"""Host-side contract for opt-in CH1 multirate stress acquisition.

The board continues to emit a 45-row spectrum so existing transports retain a
stable table CRC and wavelength axis.  A reduced SURVEY/TRACK frame contains
cached values at the rows that were not converted.  Consumers must call
``select_current_samples`` and fit only the explicitly fresh rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeVar

from frame_telemetry import (
    BoardFrameSchedule,
    MULTIRATE_EXTENSION_VERSION,
    MULTIRATE_EXTENSION_VERSION_V1,
    MULTIRATE_EXTENSION_VERSION_V2,
    MULTIRATE_EXTENSION_VERSION_V3,
    MULTIRATE_EXTENSION_VERSION_V4,
    MULTIRATE_EXTENSION_VERSION_V5,
    MULTIRATE_POINT_COUNT,
    MULTIRATE_PROFILE_SURVEY,
    current_frame_fresh_indices,
    decode_board_frame_schedule,
)


COMMAND_SIZE = 808
ACK_SIZE = 20
ACK_OK = 0x21
ACK_ERROR = 0xE1
DEFAULT_MAP_PERIOD = 0
REFERENCE_MAP_PERIOD = 1  # Firmware >=1.0.31; explicit full MAP, not dynamic.
DEFAULT_PERIODIC_MAP_PERIOD = 30
MIN_MAP_PERIOD = 3
MAX_MAP_PERIOD = 100
DEFAULT_ACTIVATE_CODES = 40
DEFAULT_RELEASE_CODES = 20
MODE_STRESS = 0
MODE_EXTRA = 2
CURRENT_SCHEDULE_VERSION = MULTIRATE_EXTENSION_VERSION
ARMABLE_SCHEDULE_VERSIONS = (
    MULTIRATE_EXTENSION_VERSION_V3,
    MULTIRATE_EXTENSION_VERSION_V4,
    MULTIRATE_EXTENSION_VERSION_V5,
)


@dataclass(frozen=True)
class MultirateConfiguration:
    enabled: bool = True
    map_period_frames: int = DEFAULT_MAP_PERIOD
    activate_threshold_codes: int = DEFAULT_ACTIVATE_CODES
    release_threshold_codes: int = DEFAULT_RELEASE_CODES

    def validate(self) -> "MultirateConfiguration":
        period = int(self.map_period_frames)
        if period not in (0, REFERENCE_MAP_PERIOD) and not MIN_MAP_PERIOD <= period <= MAX_MAP_PERIOD:
            raise ValueError(
                "map_period_frames must be 0 (initial MAP only; no automatic MAP) "
                "or 1 (full reference only), or 3..100"
            )
        activate = int(self.activate_threshold_codes)
        release = int(self.release_threshold_codes)
        if not 1 <= activate <= 0xFFFF:
            raise ValueError("activate_threshold_codes must be in 1..65535")
        if not 0 <= release <= activate:
            raise ValueError("release threshold must be in 0..activation threshold")
        return self


@dataclass(frozen=True)
class MultirateAck:
    configuration: MultirateConfiguration
    accepted: bool
    schedule_version: int | None


def build_multirate_arm_command(
    configuration: MultirateConfiguration = MultirateConfiguration(),
    *,
    schedule_version: int = CURRENT_SCHEDULE_VERSION,
) -> bytes:
    """Build the fixed-size EXTRA-mode arm command.

    Arming is volatile and applies to the next stress session only.  Omitting
    this command is the backwards-compatible MAP45 default.
    """

    configuration.validate()
    requested_version = int(schedule_version)
    if requested_version not in ARMABLE_SCHEDULE_VERSIONS:
        raise ValueError(
            "schedule_version must be 3 (TRACK13 compatibility) or 4 "
            "(adaptive TRACK11/TRACK13), or 5 (experimental G8 right sentinel)"
        )
    command = bytearray(COMMAND_SIZE)
    command[:4] = b"\xFF\xFF\x01\x08"
    command[4] = int(bool(configuration.enabled))
    command[5] = int(configuration.map_period_frames)
    command[6:8] = int(configuration.activate_threshold_codes).to_bytes(2, "big")
    command[8:10] = int(configuration.release_threshold_codes).to_bytes(2, "big")
    command[10] = requested_version
    return bytes(command)


def build_multirate_disarm_command(
    *, schedule_version: int = CURRENT_SCHEDULE_VERSION
) -> bytes:
    """Build an explicit volatile disarm request for use after entering EXTRA."""

    return build_multirate_arm_command(
        MultirateConfiguration(enabled=False),
        schedule_version=schedule_version,
    )


def build_mode_command(mode: int) -> bytes:
    """Build the established fixed-size board work-mode command."""

    if int(mode) not in (MODE_STRESS, MODE_EXTRA):
        raise ValueError("multirate control only permits STRESS or safe EXTRA")
    command = bytearray(COMMAND_SIZE)
    command[:4] = b"\xFF\xFF\x01\x02"
    command[8] = int(mode)
    return bytes(command)


def build_safe_arm_command_sequence(
    configuration: MultirateConfiguration = MultirateConfiguration(),
    *,
    schedule_version: int = CURRENT_SCHEDULE_VERSION,
) -> tuple[bytes, bytes, bytes]:
    """Return commands in the only safe arm order.

    The caller must observe EXTRA before sending item 2, wait for and validate
    the 02/05 arm ACK before sending item 3, then reject dynamic data until the
    first MAP45 frame for the requested v3/v4 contract arrives.  The tuple
    does not remove those
    acknowledgement barriers; it only prevents command construction drift.
    """

    if not configuration.enabled:
        raise ValueError("safe arm sequence requires enabled=True")
    return (
        build_mode_command(MODE_EXTRA),
        build_multirate_arm_command(
            configuration, schedule_version=schedule_version
        ),
        build_mode_command(MODE_STRESS),
    )


def build_safe_disarm_command_sequence(
    *, schedule_version: int = CURRENT_SCHEDULE_VERSION
) -> tuple[bytes, bytes]:
    """Return EXTRA-first disarm commands.

    The caller must observe EXTRA before item 2 and validate its 02/05 ACK.
    Disconnect, emergency stop, OTA entry, and application exit should invoke
    this sequence when possible; firmware also clears volatile state locally.
    """

    return (
        build_mode_command(MODE_EXTRA),
        build_multirate_disarm_command(schedule_version=schedule_version),
    )


def decode_multirate_ack(frame: bytes) -> MultirateAck | None:
    if len(frame) != ACK_SIZE or frame[:4] != b"\xFF\xFF\x02\x05":
        return None
    if frame[4] not in (0, 1) or frame[10] not in (ACK_OK, ACK_ERROR):
        return None
    configuration = MultirateConfiguration(
        enabled=bool(frame[4]),
        map_period_frames=int(frame[5]),
        activate_threshold_codes=int.from_bytes(frame[6:8], "big"),
        release_threshold_codes=int.from_bytes(frame[8:10], "big"),
    )
    try:
        configuration.validate()
    except ValueError:
        return None
    version_raw = int(frame[11])
    if version_raw not in (
        0,
        MULTIRATE_EXTENSION_VERSION_V1,
        MULTIRATE_EXTENSION_VERSION_V2,
        MULTIRATE_EXTENSION_VERSION_V3,
        MULTIRATE_EXTENSION_VERSION_V4,
        MULTIRATE_EXTENSION_VERSION_V5,
    ):
        return None
    return MultirateAck(
        configuration,
        frame[10] == ACK_OK,
        version_raw or None,
    )


def validate_current_arm_ack(
    ack: MultirateAck | None,
    configuration: MultirateConfiguration = MultirateConfiguration(),
    *,
    requested_version: int = CURRENT_SCHEDULE_VERSION,
) -> MultirateAck:
    """Require an accepted ACK for exactly the requested v3/v4 contract.

    An old board may leave ACK byte 11 at zero.  Such an ACK remains decodable
    for diagnostics, but must not be followed by reduced-profile acquisition.
    """

    configuration.validate()
    requested_version = int(requested_version)
    if requested_version not in ARMABLE_SCHEDULE_VERSIONS:
        raise ValueError("requested schedule version is not armable")
    if ack is None or not ack.accepted:
        raise ValueError("multirate arm was not accepted")
    if ack.schedule_version != requested_version:
        raise ValueError(
            "board did not acknowledge requested schedule version "
            f"v{requested_version}"
        )
    if ack.configuration != configuration:
        raise ValueError("board ACK does not match the requested configuration")
    return ack


def validate_current_disarm_ack(
    ack: MultirateAck | None,
    *,
    requested_version: int = CURRENT_SCHEDULE_VERSION,
) -> MultirateAck:
    """Require proof that the requested v3/v4 multirate state is disabled.

    Firmware keeps the previous thresholds/period for a later arm, so disarm
    validation intentionally checks only status, version, and ``enabled``.
    """

    requested_version = int(requested_version)
    if requested_version not in ARMABLE_SCHEDULE_VERSIONS:
        raise ValueError("requested schedule version is not armable")
    if ack is None or not ack.accepted:
        raise ValueError("multirate disarm was not accepted")
    if ack.schedule_version != requested_version:
        raise ValueError(
            "board did not acknowledge requested schedule version "
            f"v{requested_version}"
        )
    if ack.configuration.enabled:
        raise ValueError("board ACK still reports multirate enabled")
    return ack


T = TypeVar("T")


def select_current_samples(
    samples: Sequence[T],
    *,
    extension: bytes,
) -> tuple[tuple[int, T], ...]:
    """Discard cached rows before a fit, detector, or training update."""

    if len(samples) != MULTIRATE_POINT_COUNT:
        raise ValueError("stress samples must use the certified 45-point axis")
    fresh = current_frame_fresh_indices(extension, point_count=len(samples))
    return tuple((index, samples[index]) for index in fresh)


def select_current_timed_samples(
    samples: Sequence[T],
    *,
    extension: bytes,
) -> tuple[tuple[int, T, int], ...]:
    """Return fresh values with their real within-frame acquisition times.

    Version-1 (52-byte) schedules remain freshness-compatible but deliberately
    fail here: they do not contain enough information for a 15 Hz dynamic fit.
    """

    selected = select_current_samples(samples, extension=extension)
    schedule = decode_board_frame_schedule(extension)
    if schedule is None or not schedule.has_point_timing:
        raise ValueError("per-point sample timing requires multirate extension v2")
    if schedule.bandwidth_discontinuity:
        raise ValueError("forced MAP marks a pressure-bandwidth discontinuity")
    timed: list[tuple[int, T, int]] = []
    for index, value in selected:
        offset_us = schedule.sample_offset_us[index]
        if offset_us is None:  # Defensive: decoder already enforces this.
            raise ValueError(f"fresh point {index} has no sample time")
        timed.append((index, value, offset_us))
    return tuple(timed)


@dataclass(frozen=True)
class SurveyShiftEstimate:
    segment: int
    shift_nm: float
    common_mode_codes: float
    estimate_offset_us: float


def estimate_survey_peak_shifts(
    wavelengths_nm: Sequence[float],
    baseline_codes: Sequence[float],
    current_codes: Sequence[float],
    *,
    extension: bytes,
    minimum_slope_codes_per_nm: float = 1.0,
) -> tuple[SurveyShiftEstimate, ...]:
    """Estimate all nine small peak shifts from legacy v2 SURVEY18 dual flanks.

    For each five-point grating, SURVEY samples offsets 1 and 3.  Baseline
    slopes at those locations come from (0, 2) and (2, 4).  Differencing the
    two flank changes rejects a first-order common optical-power change:
    ``delta_lambda = -(delta_right-delta_left)/(slope_right-slope_left)``.
    The estimate is local/linear; a large shift leaving the two slopes must be
    corrected by the periodic MAP45 rather than extrapolated.
    """

    if not (
        len(wavelengths_nm)
        == len(baseline_codes)
        == len(current_codes)
        == MULTIRATE_POINT_COUNT
    ):
        raise ValueError("survey shift estimation requires three 45-point arrays")
    schedule = decode_board_frame_schedule(extension)
    if schedule is None or schedule.profile != MULTIRATE_PROFILE_SURVEY:
        raise ValueError("survey shift estimation requires a SURVEY18 frame")
    if schedule.version != MULTIRATE_EXTENSION_VERSION_V2:
        raise ValueError("dual-flank shift estimation requires v2 SURVEY18")
    if not schedule.has_point_timing:
        raise ValueError("SURVEY18 shift estimation requires per-point timing")

    results: list[SurveyShiftEstimate] = []
    for segment in range(9):
        base = segment * 5
        left = base + 1
        centre = base + 2
        right = base + 3
        left_span = float(wavelengths_nm[centre]) - float(wavelengths_nm[base])
        right_span = float(wavelengths_nm[base + 4]) - float(
            wavelengths_nm[centre]
        )
        if left_span <= 0.0 or right_span <= 0.0:
            raise ValueError("wavelengths must increase within every grating")
        slope_left = (
            float(baseline_codes[centre]) - float(baseline_codes[base])
        ) / left_span
        slope_right = (
            float(baseline_codes[base + 4]) - float(baseline_codes[centre])
        ) / right_span
        slope_difference = slope_right - slope_left
        if abs(slope_left) < minimum_slope_codes_per_nm or abs(
            slope_right
        ) < minimum_slope_codes_per_nm:
            raise ValueError(f"segment {segment} has an unobservable flank")
        if abs(slope_difference) < 2.0 * minimum_slope_codes_per_nm:
            raise ValueError(f"segment {segment} flanks are not opposite enough")

        delta_left = float(current_codes[left]) - float(baseline_codes[left])
        delta_right = float(current_codes[right]) - float(baseline_codes[right])
        shift = -(delta_right - delta_left) / slope_difference
        common_left = delta_left + slope_left * shift
        common_right = delta_right + slope_right * shift
        left_time = schedule.sample_offset_us[left]
        right_time = schedule.sample_offset_us[right]
        if left_time is None or right_time is None:
            raise ValueError(f"segment {segment} is missing a fresh flank")
        results.append(
            SurveyShiftEstimate(
                segment=segment,
                shift_nm=shift,
                common_mode_codes=0.5 * (common_left + common_right),
                estimate_offset_us=0.5 * (left_time + right_time),
            )
        )
    return tuple(results)


@dataclass(frozen=True)
class TimingBudget:
    profile: str
    fresh_points: int
    dac_writes: int
    estimated_period_us: int

    @property
    def estimated_rate_hz(self) -> float:
        return 1_000_000.0 / self.estimated_period_us


def estimate_timing_budget(
    *,
    profile: str,
    fresh_points: int,
    dac_writes: int = 177,
    i2c_write_us: int = 90,
    adc_pair_us: int = 675,
    hidden_predecessor_hold_us: int = 5_000,
    fixed_overhead_us: int = 1_000,
) -> TimingBudget:
    """Conservative first-order budget, not a substitute for 300-frame proof.

    ``177`` accounts for the full steady 45-row changed-channel path and the
    audited hidden predecessor/target full rewrites.  At 400 kHz I2C, 90 us is
    the wire-time floor for one three-byte register write.  Hardware acceptance
    still requires mean and p99 telemetry, with pressure tracking strictly
    above 30 Hz for a 15 Hz physical input.
    """

    if fresh_points <= 0 or dac_writes <= 0:
        raise ValueError("fresh_points and dac_writes must be positive")
    period = (
        int(dac_writes) * int(i2c_write_us)
        + int(fresh_points) * int(adc_pair_us)
        + int(hidden_predecessor_hold_us)
        + int(fixed_overhead_us)
    )
    return TimingBudget(profile, int(fresh_points), int(dac_writes), period)


def default_profile_budgets() -> tuple[TimingBudget, ...]:
    return (
        estimate_timing_budget(profile="MAP45", fresh_points=45),
        estimate_timing_budget(profile="SURVEY9", fresh_points=9),
        estimate_timing_budget(profile="TRACK11_SINGLE", fresh_points=11),
        # Retain the historic key for saved analyses while exposing the
        # unambiguous v4 name alongside it.
        estimate_timing_budget(profile="TRACK13", fresh_points=13),
        estimate_timing_budget(profile="TRACK13_WIDE", fresh_points=13),
    )
