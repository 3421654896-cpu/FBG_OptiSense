"""Deterministic host simulator for the firmware CH1 multirate scheduler.

This is intentionally hardware-free.  It lets tests and dataset tooling prove
profile coverage/freshness before the corresponding firmware is ever enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable


POINT_COUNT = 45
SEGMENT_COUNT = 9
POINTS_PER_SEGMENT = 5
DEFAULT_MAP_PERIOD = 0
REFERENCE_MAP_PERIOD = 1
BASELINE_QUIET_FRAMES = 4
BASELINE_EMA_SHIFT = 6
STARTUP_STABLE_FRAMES = 8
STARTUP_MAX_FRAMES = 32
MAX_MAP_DEFERRAL_FRAMES = 120
MAP_AGE_MASK = 0x7F
SINGLE_CONFIRM_FRAMES = 8
ADC_MAX_CODE = 4095

# Certified high-slope sentinels for the nine 5-point gratings.  Each value is
# a local index within its grating.  Source and SHA-256 are documented in the
# firmware table in stress_multirate.c.
SENTINEL_LOCAL_OFFSETS = (3, 3, 3, 3, 1, 3, 1, 1, 1)
SENTINEL_INVERSE_SLOPE_Q8 = (136, 79, 82, 205, 128, 134, 182, 71, 177)


class Profile(IntEnum):
    MAP = 0
    SURVEY = 1
    TRACK_WIDE = 2
    TRACK = TRACK_WIDE  # v3 compatibility name
    TRACK_SINGLE = 3


class BaselineStatus(IntEnum):
    READY = 0
    LEARNING = 1
    FORCED_READY = 2


@dataclass(frozen=True)
class Plan:
    sequence: int
    profile: Profile
    fresh_indices: tuple[int, ...]
    primary_segment: int | None = None
    secondary_segment: int | None = None
    map_age_frames: int = 0
    bandwidth_discontinuity: bool = False


def flank_indices(segment: int) -> tuple[int, int]:
    if not 0 <= int(segment) < SEGMENT_COUNT:
        raise ValueError("segment must be in 0..8")
    base = int(segment) * POINTS_PER_SEGMENT
    return base + 1, base + 3


def triplet_indices(segment: int) -> tuple[int, int, int]:
    left, right = flank_indices(segment)
    return left, int(segment) * POINTS_PER_SEGMENT + 2, right


def sentinel_index(segment: int, *, g8_right=False) -> int:
    if not 0 <= int(segment) < SEGMENT_COUNT:
        raise ValueError("segment must be in 0..8")
    offset = 3 if g8_right and int(segment) == 7 else SENTINEL_LOCAL_OFFSETS[int(segment)]
    return int(segment) * POINTS_PER_SEGMENT + offset


def track_indices(primary: int, secondary: int, *, g8_right=False) -> tuple[int, ...]:
    """Return TRACK13 rows in their real ascending DAC/sample order."""

    if primary == secondary:
        raise ValueError("TRACK requires two different ROI segments")
    roi = {int(primary), int(secondary)}
    if any(not 0 <= segment < SEGMENT_COUNT for segment in roi):
        raise ValueError("ROI segments must be in 0..8")
    points = set(triplet_indices(int(primary)) + triplet_indices(int(secondary)))
    points.update(
        sentinel_index(segment, g8_right=g8_right)
        for segment in range(SEGMENT_COUNT)
        if segment not in roi
    )
    return tuple(sorted(points))


def track_single_indices(primary: int, *, g8_right=False) -> tuple[int, ...]:
    """Return TRACK11 rows in their real ascending DAC/sample order."""

    primary = int(primary)
    if not 0 <= primary < SEGMENT_COUNT:
        raise ValueError("primary ROI segment must be in 0..8")
    points = set(triplet_indices(primary))
    points.update(
        sentinel_index(segment, g8_right=g8_right)
        for segment in range(SEGMENT_COUNT)
        if segment != primary
    )
    return tuple(sorted(points))


class Scheduler:
    """Reference implementation matching ``stress_multirate.c`` policy."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        map_period_frames: int = DEFAULT_MAP_PERIOD,
        activate_threshold_codes: int = 40,
        release_threshold_codes: int = 20,
        release_frames: int = 4,
        single_track_enabled: bool = False,
        g8_right: bool = False,
    ) -> None:
        if int(map_period_frames) not in (0, REFERENCE_MAP_PERIOD) and not 3 <= int(map_period_frames) <= 100:
            raise ValueError(
                "map period must be 0 (initial MAP only; no automatic MAP) "
                "or 1 (full reference), or in 3..100"
            )
        if not 0 <= int(release_threshold_codes) <= int(
            activate_threshold_codes
        ):
            raise ValueError("release threshold must not exceed activation")
        self.enabled = bool(enabled)
        self.map_period = int(map_period_frames)
        self.activate = int(activate_threshold_codes)
        self.release = int(release_threshold_codes)
        self.release_frames = int(release_frames)
        self.single_track_enabled = bool(single_track_enabled)
        self.g8_right = bool(g8_right)
        self.start_session()

    def start_session(self) -> None:
        self.cache: list[int | None] = [None] * POINT_COUNT
        self.baseline: list[int] | None = None
        self.baseline_q6: list[int] | None = None
        self.baseline_quiet_frames = 0
        self.startup_baseline_frames = 0
        self.startup_baseline_stable_frames = 0
        self.baseline_status = BaselineStatus.LEARNING
        self.last_scores = [0] * SEGMENT_COUNT
        self.frame_sentinel_seen: set[int] = set()
        self.frame_sentinel_out_of_domain = False
        self.activity_latched = False
        self.active_pair: tuple[int, int] | None = None
        self.single_track_active = False
        self.compact_unique_frames = 0
        self.compact_candidate_segment: int | None = None
        self.low_activity_frames = 0
        self.frames_since_map = self.map_period
        self.force_map_requested = False
        self.next_sequence = 0
        self.current: Plan | None = None
        self.frame_scores = [0] * SEGMENT_COUNT

    def force_map(self) -> None:
        self.force_map_requested = True

    def set_single_track_enabled(self, enabled: bool) -> None:
        """Apply the negotiated v4 capability without resetting the session."""

        self.single_track_enabled = bool(enabled)
        if not self.single_track_enabled:
            self.single_track_active = False
            self.compact_unique_frames = 0
            self.compact_candidate_segment = None

    def _rank_pair(self) -> tuple[int, int]:
        ranked = sorted(
            range(SEGMENT_COUNT), key=lambda segment: (-self._rank_score(segment), segment)
        )
        return ranked[0], ranked[1]

    def _rank_score(self, segment: int) -> int:
        weight = 165 if self.g8_right and segment == 7 else SENTINEL_INVERSE_SLOPE_Q8[segment]
        return self.last_scores[segment] * (
            weight if self.single_track_enabled else 1
        )

    def _sentinels_complete_and_in_domain(self) -> bool:
        return (
            self.frame_sentinel_seen == set(range(SEGMENT_COUNT))
            and not self.frame_sentinel_out_of_domain
        )

    def _compact_unique_response(self, primary: int, secondary: int) -> bool:
        if not self._sentinels_complete_and_in_domain():
            return False
        primary_score = self._rank_score(primary)
        secondary_score = self._rank_score(secondary)
        if self.last_scores[primary] < self.activate or any(
            self.last_scores[segment] >= self.activate
            for segment in range(SEGMENT_COUNT) if segment != primary
        ):
            return False
        scores = [self._rank_score(segment) for segment in range(SEGMENT_COUNT)]
        total = sum(scores)
        sum_squares = sum(score * score for score in scores)
        if total == 0:
            return False
        left, centre, right = triplet_indices(primary)
        if not set((left, centre, right)).issubset(self.current.fresh_indices):
            return False
        values = [int(self.cache[point]) for point in (left, centre, right)]
        if any(value >= ADC_MAX_CODE for value in values):
            return False
        if values[1] <= min(values[0], values[2]) + self.release:
            return False
        # Integer-exact mirrors of primary share >= .60, second/primary < .35,
        # and N_eff=(sum^2/sumSquares) <= 1.8.
        return (
            primary_score * 5 >= total * 3
            and secondary_score * 20 < primary_score * 7
            and 5 * total * total <= 9 * sum_squares
        )

    def begin_frame(self) -> Plan:
        if self.current is not None:
            raise RuntimeError("previous frame has not ended")
        sequence = self.next_sequence
        self.next_sequence += 1
        map_age = min(self.frames_since_map, MAP_AGE_MASK)
        if self.enabled and self.map_period == REFERENCE_MAP_PERIOD:
            plan = Plan(sequence, Profile.MAP, tuple(range(POINT_COUNT)),
                        map_age_frames=map_age, bandwidth_discontinuity=True)
        elif not self.enabled or any(value is None for value in self.cache):
            plan = Plan(
                sequence,
                Profile.MAP,
                tuple(range(POINT_COUNT)),
                map_age_frames=map_age,
            )
        elif self.force_map_requested:
            plan = Plan(
                sequence,
                Profile.MAP,
                tuple(range(POINT_COUNT)),
                map_age_frames=map_age,
                bandwidth_discontinuity=self.activity_latched,
            )
        elif self.activity_latched:
            if (
                self.map_period != 0
                and self.frames_since_map >= MAX_MAP_DEFERRAL_FRAMES - 1
            ):
                plan = Plan(
                    sequence,
                    Profile.MAP,
                    tuple(range(POINT_COUNT)),
                    map_age_frames=map_age,
                    bandwidth_discontinuity=True,
                )
            else:
                assert self.active_pair is not None
                primary, secondary = self.active_pair
                if self.single_track_enabled and self.single_track_active:
                    plan = Plan(
                        sequence,
                        Profile.TRACK_SINGLE,
                        track_single_indices(primary, g8_right=self.g8_right),
                        primary,
                        None,
                        map_age,
                    )
                else:
                    plan = Plan(
                        sequence,
                        Profile.TRACK_WIDE,
                        track_indices(primary, secondary, g8_right=self.g8_right),
                        primary,
                        secondary,
                        map_age,
                    )
        elif (
            self.map_period != 0
            and self.frames_since_map >= self.map_period - 1
            and self.baseline_quiet_frames >= BASELINE_QUIET_FRAMES
        ):
            plan = Plan(
                sequence,
                Profile.MAP,
                tuple(range(POINT_COUNT)),
                map_age_frames=map_age,
            )
        else:
            plan = Plan(
                sequence,
                Profile.SURVEY,
                tuple(sentinel_index(segment, g8_right=self.g8_right) for segment in range(SEGMENT_COUNT)),
                map_age_frames=map_age,
            )
        self.force_map_requested = False
        self.frame_scores = [0] * SEGMENT_COUNT
        self.frame_sentinel_seen = set()
        self.frame_sentinel_out_of_domain = False
        self.current = plan
        return plan

    def record(self, point_index: int, adc_code: int) -> None:
        if self.current is None:
            raise RuntimeError("begin_frame must be called first")
        point = int(point_index)
        if point not in self.current.fresh_indices:
            raise ValueError("cannot record a non-fresh point")
        code = int(adc_code)
        if not 0 <= code <= 0xFFFF:
            raise ValueError("ADC code must fit uint16")
        self.cache[point] = code
        if self.baseline is not None:
            segment = point // POINTS_PER_SEGMENT
            if point == sentinel_index(segment, g8_right=self.g8_right):
                self.frame_sentinel_seen.add(segment)
                if code > ADC_MAX_CODE:
                    self.frame_sentinel_out_of_domain = True
                # Only the nine common sentinels participate in spatial rank;
                # extra ROI points exist exclusively for peak-shape fitting.
                self.frame_scores[segment] = abs(code - self.baseline[point])

    def record_plan(self, values: Iterable[int]) -> None:
        if self.current is None:
            raise RuntimeError("begin_frame must be called first")
        exact = tuple(int(value) for value in values)
        if len(exact) != len(self.current.fresh_indices):
            raise ValueError("one ADC code is required for every fresh point")
        for index, code in zip(self.current.fresh_indices, exact, strict=True):
            self.record(index, code)

    def end_frame(self) -> None:
        if self.current is None:
            raise RuntimeError("begin_frame must be called first")
        profile = self.current.profile
        if profile is Profile.MAP:
            self.frames_since_map = 0
            if self.baseline is None:
                if any(value is None for value in self.cache):
                    raise RuntimeError("initial MAP did not initialize the full cache")
                self.baseline = [int(value) for value in self.cache]
                self.baseline_q6 = [value << BASELINE_EMA_SHIFT for value in self.baseline]
                self.baseline_quiet_frames = 0
                self.startup_baseline_frames = 0
                self.startup_baseline_stable_frames = 0
                self.baseline_status = BaselineStatus.LEARNING
                self.last_scores = [0] * SEGMENT_COUNT
                self.current = None
                return
        else:
            self.frames_since_map += 1

        if self.baseline is not None and self.baseline_status is BaselineStatus.LEARNING:
            if profile is Profile.MAP:
                self.baseline = [int(value) for value in self.cache]
                self.baseline_q6 = [
                    value << BASELINE_EMA_SHIFT for value in self.baseline
                ]
                self.startup_baseline_frames = 0
                self.startup_baseline_stable_frames = 0
            else:
                maximum = max(self.frame_scores)
                self.startup_baseline_frames += 1
                if self._sentinels_complete_and_in_domain() and maximum <= self.release:
                    self.startup_baseline_stable_frames += 1
                else:
                    self.startup_baseline_stable_frames = 0
                assert self.baseline_q6 is not None
                for point in self.current.fresh_indices:
                    value = int(self.cache[point])
                    self.baseline[point] = value
                    self.baseline_q6[point] = value << BASELINE_EMA_SHIFT
                if self.startup_baseline_stable_frames >= STARTUP_STABLE_FRAMES:
                    self.baseline_status = BaselineStatus.READY
                elif self.startup_baseline_frames >= STARTUP_MAX_FRAMES:
                    self.baseline_status = BaselineStatus.FORCED_READY
            self.baseline_quiet_frames = 0
            self.activity_latched = False
            self.active_pair = None
            self.single_track_active = False
            self.compact_unique_frames = 0
            self.compact_candidate_segment = None
            self.low_activity_frames = 0
            self.last_scores = [0] * SEGMENT_COUNT
            self.current = None
            return

        valid = self._sentinels_complete_and_in_domain()
        if valid:
            was_latched = self.activity_latched
            self.last_scores = list(self.frame_scores)
            maximum = max(self.last_scores)
            primary, secondary = self._rank_pair()
            compact = self._compact_unique_response(primary, secondary)
            if maximum >= self.activate:
                self.active_pair = (primary, secondary)
                self.activity_latched = True
                self.low_activity_frames = 0
                if not was_latched:
                    self.single_track_active = False
                    self.compact_unique_frames = 0
                    self.compact_candidate_segment = None
                elif profile is Profile.TRACK_SINGLE:
                    if (not self.single_track_enabled or not compact
                            or primary != self.current.primary_segment):
                        self.single_track_active = False
                        self.compact_unique_frames = 0
                        self.compact_candidate_segment = None
                elif (profile is Profile.TRACK_WIDE
                      and self.single_track_enabled and compact):
                    if self.compact_candidate_segment == primary:
                        self.compact_unique_frames = min(255, self.compact_unique_frames + 1)
                    else:
                        self.compact_candidate_segment = primary
                        self.compact_unique_frames = 1
                    if self.compact_unique_frames >= SINGLE_CONFIRM_FRAMES:
                        self.single_track_active = True
                else:
                    self.single_track_active = False
                    self.compact_unique_frames = 0
                    self.compact_candidate_segment = None
            elif self.activity_latched and maximum <= self.release:
                self.low_activity_frames = min(255, self.low_activity_frames + 1)
                self.compact_unique_frames = 0
                self.compact_candidate_segment = None
                if self.low_activity_frames >= self.release_frames:
                    self.activity_latched = False
                    self.active_pair = None
                    self.last_scores = [0] * SEGMENT_COUNT
                    self.single_track_active = False
            elif self.activity_latched:
                self.low_activity_frames = 0
                self.compact_unique_frames = 0
                self.compact_candidate_segment = None
                if (profile is Profile.TRACK_SINGLE
                        and (not self.single_track_enabled or not compact
                             or primary != self.current.primary_segment)):
                    self.single_track_active = False
        else:
            maximum = self.release + 1
            self.baseline_quiet_frames = 0
            self.low_activity_frames = 0
            self.single_track_active = False
            self.compact_unique_frames = 0
            self.compact_candidate_segment = None

        if valid and not self.activity_latched and maximum <= self.release:
            self.baseline_quiet_frames += 1
            if self.baseline_quiet_frames >= BASELINE_QUIET_FRAMES:
                assert self.baseline is not None and self.baseline_q6 is not None
                if self.baseline_status is BaselineStatus.FORCED_READY:
                    self.baseline_status = BaselineStatus.READY
                divisor = 1 << BASELINE_EMA_SHIFT
                for point in self.current.fresh_indices:
                    delta = (int(self.cache[point]) << BASELINE_EMA_SHIFT) - self.baseline_q6[point]
                    # C99 signed integer division truncates toward zero.
                    step = delta // divisor if delta >= 0 else -((-delta) // divisor)
                    self.baseline_q6[point] += step
                    self.baseline[point] = (
                        self.baseline_q6[point] + (divisor >> 1)
                    ) >> BASELINE_EMA_SHIFT
        else:
            self.baseline_quiet_frames = 0
        self.current = None

    def cached_frame(self) -> tuple[int, ...]:
        if any(value is None for value in self.cache):
            raise RuntimeError("cache is incomplete; a MAP frame is required")
        return tuple(int(value) for value in self.cache)
