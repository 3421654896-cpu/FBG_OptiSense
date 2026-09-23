"""Backward-compatible timing telemetry for raw STM32 scan frames.

The first 28 bytes of the B5 4D v1 board-status extension are deliberately
unchanged.  Firmware that supports acquisition timing appends sequence and
duration at offsets 28 and 32; full-map frames may append their start tick at
offset 36.  New firmware appends a tagged runtime-identity block after the
complete base/multirate body.  Keeping this decoder independent of Qt lets
command-line benchmarks and GUI code use exactly the same semantics.
"""

from __future__ import annotations

from dataclasses import dataclass


BOARD_STATUS_V1_LEGACY_LENGTH = 28
BOARD_STATUS_V1_TIMING_LENGTH = 36
BOARD_STATUS_V1_FRAME_START_LENGTH = 40
BOARD_STATUS_V1_MULTIRATE_V1_LENGTH = 52
BOARD_STATUS_V1_MULTIRATE_LENGTH = 142
BOARD_STATUS_RUNTIME_IDENTITY_MAGIC = b"\xC7\x49"
BOARD_STATUS_RUNTIME_IDENTITY_VERSION = 1
BOARD_STATUS_RUNTIME_IDENTITY_PAYLOAD_LENGTH = 12
BOARD_STATUS_RUNTIME_IDENTITY_BLOCK_LENGTH = 16
MULTIRATE_EXTENSION_VERSION_V1 = 1
MULTIRATE_EXTENSION_VERSION_V2 = 2
MULTIRATE_EXTENSION_VERSION_V3 = 3
MULTIRATE_EXTENSION_VERSION_V4 = 4
MULTIRATE_EXTENSION_VERSION_V5 = 5
# Production/default negotiation remains v4 until the changed optical path
# has been qualified. v5 is the explicit G8-right-flank experiment.
MULTIRATE_EXTENSION_VERSION = MULTIRATE_EXTENSION_VERSION_V4
MULTIRATE_PROFILE_MAP = 0
MULTIRATE_PROFILE_SURVEY = 1
MULTIRATE_PROFILE_TRACK_WIDE = 2
MULTIRATE_PROFILE_TRACK_SINGLE = 3
# Backwards-compatible public name: every historic ``TRACK`` frame is the
# two-ROI WIDE contract.  It must never be repurposed for TRACK11.
MULTIRATE_PROFILE_TRACK = MULTIRATE_PROFILE_TRACK_WIDE
MULTIRATE_POINT_COUNT = 45
MULTIRATE_BITMAP_BYTES = 6
MULTIRATE_SAMPLE_TIME_UNIT_US = 2
MULTIRATE_SAMPLE_TIME_UNAVAILABLE = 0xFFFF
MULTIRATE_TRACK_WIDE_POINT_COUNT = 13
MULTIRATE_TRACK_SINGLE_POINT_COUNT = 11
MULTIRATE_TRACK_POINT_COUNT = MULTIRATE_TRACK_WIDE_POINT_COUNT
MULTIRATE_V1_TRACK_POINT_COUNT = 6
MULTIRATE_SENTINEL_LOCAL_OFFSETS = (3, 3, 3, 3, 1, 3, 1, 1, 1)
MULTIRATE_V5_SENTINEL_LOCAL_OFFSETS = (3, 3, 3, 3, 1, 3, 1, 3, 1)
MULTIRATE_MAP_AGE_MASK = 0x7F
MULTIRATE_BANDWIDTH_GAP_FLAG = 0x80
UINT32_MASK = 0xFFFFFFFF
UINT32_FORWARD_LIMIT = 0x80000000
RAW_SCAN_MAX_POINTS = 2500
RAW_SCAN_MAX_PEAKS_PER_CHANNEL = 63


def sentinel_offsets_for_version(version: int) -> tuple[int, ...]:
    if version == MULTIRATE_EXTENSION_VERSION_V5:
        return MULTIRATE_V5_SENTINEL_LOCAL_OFFSETS
    if version in (1, 2, 3, 4):
        return MULTIRATE_SENTINEL_LOCAL_OFFSETS
    raise ValueError(f"unknown sentinel schedule version {version}")


@dataclass(frozen=True)
class BoardFrameTiming:
    """Timing fields appended to a B5 4D v1 board-status extension."""

    sequence: int
    acquisition_duration_us: int

    @property
    def acquisition_duration_ms(self) -> float:
        return self.acquisition_duration_us / 1000.0


@dataclass(frozen=True)
class BoardRuntimeIdentity:
    """Identity binding a raw USB frame to one boot and wavelength table."""

    table_crc32: int
    boot_id: int
    uptime_ms: int


def _runtime_identity_body(extension: bytes) -> bytes:
    """Return the legacy/schedule body before an optional tagged suffix."""

    if len(extension) < BOARD_STATUS_RUNTIME_IDENTITY_BLOCK_LENGTH:
        return extension
    block = extension[-BOARD_STATUS_RUNTIME_IDENTITY_BLOCK_LENGTH:]
    if block[:2] != BOARD_STATUS_RUNTIME_IDENTITY_MAGIC:
        return extension
    if block[2] != BOARD_STATUS_RUNTIME_IDENTITY_VERSION:
        raise ValueError(
            f"unsupported board runtime identity version {int(block[2])}"
        )
    if block[3] != BOARD_STATUS_RUNTIME_IDENTITY_PAYLOAD_LENGTH:
        raise ValueError(
            f"invalid board runtime identity length {int(block[3])}"
        )
    return extension[:-BOARD_STATUS_RUNTIME_IDENTITY_BLOCK_LENGTH]


def decode_board_runtime_identity(extension: bytes) -> BoardRuntimeIdentity | None:
    """Decode the tagged final block; return ``None`` for older firmware."""

    body = _runtime_identity_body(extension)
    if len(body) == len(extension):
        return None
    block = extension[-BOARD_STATUS_RUNTIME_IDENTITY_BLOCK_LENGTH:]
    return BoardRuntimeIdentity(
        table_crc32=int.from_bytes(block[4:8], "big"),
        boot_id=int.from_bytes(block[8:12], "big"),
        uptime_ms=int.from_bytes(block[12:16], "big"),
    )


@dataclass(frozen=True)
class BoardFrameSchedule:
    """Freshness contract appended to a multirate stress frame.

    Values outside ``fresh_indices`` are historical cache and must never enter
    a current-frame fit, trigger detector, or training label.
    """

    version: int
    profile: int
    primary_segment: int | None
    secondary_segment: int | None
    fresh_indices: tuple[int, ...]
    map_age_frames: int
    frame_start_ms: int
    sample_offset_us: tuple[int | None, ...]
    bandwidth_discontinuity: bool = False

    @property
    def is_complete_map(self) -> bool:
        return (
            self.profile == MULTIRATE_PROFILE_MAP
            and len(self.fresh_indices) == MULTIRATE_POINT_COUNT
        )

    @property
    def has_point_timing(self) -> bool:
        return self.version >= MULTIRATE_EXTENSION_VERSION_V2

    @property
    def is_track_wide(self) -> bool:
        return self.profile == MULTIRATE_PROFILE_TRACK_WIDE

    @property
    def is_track_single(self) -> bool:
        return self.profile == MULTIRATE_PROFILE_TRACK_SINGLE

    @property
    def is_track(self) -> bool:
        return self.is_track_wide or self.is_track_single


def decode_board_frame_timing(extension: bytes) -> BoardFrameTiming | None:
    """Decode optional timing fields, returning ``None`` for legacy v1 data."""

    if len(extension) < BOARD_STATUS_V1_TIMING_LENGTH:
        return None
    return BoardFrameTiming(
        sequence=int.from_bytes(extension[28:32], "big"),
        acquisition_duration_us=int.from_bytes(extension[32:36], "big"),
    )


def decode_board_frame_start_ms(extension: bytes) -> int | None:
    """Decode the MCU frame-start tick from full or multirate frames.

    A non-multirate v1 extension appends the tick at offset 36.  Multirate
    v1/v2/v3/v4 use offset 36 for their schedule version and retain the
    original tick at offset 48.  Older 28/36-byte frames have no such field.
    """

    extension = _runtime_identity_body(extension)
    if (
        len(extension) >= BOARD_STATUS_V1_MULTIRATE_V1_LENGTH
        and extension[36]
        in (
            MULTIRATE_EXTENSION_VERSION_V1,
            MULTIRATE_EXTENSION_VERSION_V2,
            MULTIRATE_EXTENSION_VERSION_V3,
            MULTIRATE_EXTENSION_VERSION_V4,
            MULTIRATE_EXTENSION_VERSION_V5,
        )
    ):
        return int.from_bytes(extension[48:52], "big")
    if len(extension) >= BOARD_STATUS_V1_FRAME_START_LENGTH:
        return int.from_bytes(extension[36:40], "big")
    return None


def decode_board_frame_schedule(extension: bytes) -> BoardFrameSchedule | None:
    """Decode the opt-in CH1 multirate suffix.

    A legacy 28/36-byte extension returns ``None``.  That is safe because the
    firmware only permits reduced sampling after a new host explicitly arms
    it; legacy sessions therefore contain genuine MAP frames.  A present but
    malformed suffix raises ``ValueError`` rather than silently treating
    cached values as fresh.
    """

    extension = _runtime_identity_body(extension)
    if len(extension) < BOARD_STATUS_V1_MULTIRATE_V1_LENGTH:
        return None
    version = int(extension[36])
    if version not in (
        MULTIRATE_EXTENSION_VERSION_V1,
        MULTIRATE_EXTENSION_VERSION_V2,
        MULTIRATE_EXTENSION_VERSION_V3,
        MULTIRATE_EXTENSION_VERSION_V4,
        MULTIRATE_EXTENSION_VERSION_V5,
    ):
        raise ValueError(
            f"unsupported stress multirate extension {extension[36]}"
        )
    sentinel_offsets = sentinel_offsets_for_version(version)
    required_length = (
        BOARD_STATUS_V1_MULTIRATE_LENGTH
        if version >= MULTIRATE_EXTENSION_VERSION_V2
        else BOARD_STATUS_V1_MULTIRATE_V1_LENGTH
    )
    if len(extension) < required_length:
        raise ValueError(
            f"truncated stress multirate v{version} extension: "
            f"need {required_length}, got {len(extension)}"
        )
    profile = int(extension[37])
    if profile not in (
        MULTIRATE_PROFILE_MAP,
        MULTIRATE_PROFILE_SURVEY,
        MULTIRATE_PROFILE_TRACK_WIDE,
        MULTIRATE_PROFILE_TRACK_SINGLE,
    ):
        raise ValueError(f"invalid stress multirate profile {profile}")
    if (
        profile == MULTIRATE_PROFILE_TRACK_SINGLE
        and version < MULTIRATE_EXTENSION_VERSION_V4
    ):
        raise ValueError("TRACK11-SINGLE requires stress multirate v4")

    primary_raw = int(extension[38])
    secondary_raw = int(extension[39])
    fresh_count = int(extension[40])
    map_age_and_flags = int(extension[41])
    if version >= MULTIRATE_EXTENSION_VERSION_V3:
        map_age = map_age_and_flags & MULTIRATE_MAP_AGE_MASK
        bandwidth_discontinuity = bool(
            map_age_and_flags & MULTIRATE_BANDWIDTH_GAP_FLAG
        )
    else:
        map_age = map_age_and_flags
        bandwidth_discontinuity = False
    bitmap = bytes(extension[42 : 42 + MULTIRATE_BITMAP_BYTES])
    if bitmap[-1] & 0xE0:
        raise ValueError("fresh bitmap sets points beyond the 45-point table")
    fresh_indices = tuple(
        index
        for index in range(MULTIRATE_POINT_COUNT)
        if bitmap[index >> 3] & (1 << (index & 7))
    )
    if fresh_count != len(fresh_indices):
        raise ValueError(
            f"fresh-count mismatch: header {fresh_count}, bitmap {len(fresh_indices)}"
        )

    expected_count = {
        MULTIRATE_PROFILE_MAP: 45,
        MULTIRATE_PROFILE_SURVEY: 18,
        MULTIRATE_PROFILE_TRACK_WIDE: (
            MULTIRATE_TRACK_WIDE_POINT_COUNT
            if version >= MULTIRATE_EXTENSION_VERSION_V2
            else MULTIRATE_V1_TRACK_POINT_COUNT
        ),
        MULTIRATE_PROFILE_TRACK_SINGLE: MULTIRATE_TRACK_SINGLE_POINT_COUNT,
    }[profile]
    if (
        version >= MULTIRATE_EXTENSION_VERSION_V3
        and profile == MULTIRATE_PROFILE_SURVEY
    ):
        expected_count = 9
    if fresh_count != expected_count:
        raise ValueError(
            f"profile {profile} requires {expected_count} fresh points, got {fresh_count}"
        )

    if profile == MULTIRATE_PROFILE_TRACK_WIDE:
        if not (0 <= primary_raw < 9 and 0 <= secondary_raw < 9):
            raise ValueError("TRACK13-WIDE requires two valid FBG segment indices")
        if primary_raw == secondary_raw:
            raise ValueError("TRACK13-WIDE primary and secondary segments must differ")
        primary = primary_raw
        secondary = secondary_raw
        roi = {primary, secondary}
        expected_track_points = set()
        for segment in range(9):
            base = segment * 5
            if segment in roi:
                expected_track_points.update((base + 1, base + 2, base + 3))
            elif version >= MULTIRATE_EXTENSION_VERSION_V2:
                expected_track_points.add(
                    base + sentinel_offsets[segment]
                )
        if set(fresh_indices) != expected_track_points:
            raise ValueError(
                "TRACK13-WIDE fresh bitmap does not match its ROI/sentinel contract"
            )
    elif profile == MULTIRATE_PROFILE_TRACK_SINGLE:
        if not 0 <= primary_raw < 9:
            raise ValueError("TRACK11-SINGLE requires one valid primary segment")
        if secondary_raw != 0xFF:
            raise ValueError("TRACK11-SINGLE secondary segment must be 0xFF")
        primary = primary_raw
        secondary = None
        expected_track_points = set()
        for segment in range(9):
            base = segment * 5
            if segment == primary:
                expected_track_points.update((base + 1, base + 2, base + 3))
            else:
                expected_track_points.add(
                    base + sentinel_offsets[segment]
                )
        if set(fresh_indices) != expected_track_points:
            raise ValueError(
                "TRACK11-SINGLE fresh bitmap does not match its ROI/sentinel contract"
            )
    else:
        if primary_raw != 0xFF or secondary_raw != 0xFF:
            raise ValueError("MAP/SURVEY must not claim an active ROI")
        primary = None
        secondary = None

    if profile == MULTIRATE_PROFILE_MAP:
        if fresh_indices != tuple(range(MULTIRATE_POINT_COUNT)):
            raise ValueError("MAP fresh bitmap must contain all 45 points")
    elif profile == MULTIRATE_PROFILE_SURVEY:
        if version >= MULTIRATE_EXTENSION_VERSION_V3:
            expected_survey_points = tuple(
                segment * 5 + sentinel_offsets[segment]
                for segment in range(9)
            )
        else:
            expected_survey_points = tuple(
                point
                for segment in range(9)
                for point in (segment * 5 + 1, segment * 5 + 3)
            )
        if fresh_indices != expected_survey_points:
            raise ValueError("SURVEY fresh bitmap does not match its version")

    if bandwidth_discontinuity and profile != MULTIRATE_PROFILE_MAP:
        raise ValueError("bandwidth-gap flag is valid only on a forced MAP")

    if version >= MULTIRATE_EXTENSION_VERSION_V2:
        sample_offsets: list[int | None] = []
        last_fresh_offset = -1
        for index in range(MULTIRATE_POINT_COUNT):
            position = 52 + 2 * index
            ticks = int.from_bytes(extension[position : position + 2], "big")
            is_fresh = index in fresh_indices
            if is_fresh:
                if ticks == MULTIRATE_SAMPLE_TIME_UNAVAILABLE:
                    raise ValueError(f"fresh point {index} has no sample time")
                offset_us = ticks * MULTIRATE_SAMPLE_TIME_UNIT_US
                if offset_us <= last_fresh_offset:
                    raise ValueError("fresh sample times must be strictly increasing")
                last_fresh_offset = offset_us
                sample_offsets.append(offset_us)
            else:
                if ticks != MULTIRATE_SAMPLE_TIME_UNAVAILABLE:
                    raise ValueError(f"cached point {index} claims a sample time")
                sample_offsets.append(None)

        acquisition_us = int.from_bytes(extension[32:36], "big")
        if acquisition_us and last_fresh_offset > acquisition_us + 1:
            raise ValueError("sample time exceeds frame acquisition duration")
        point_times = tuple(sample_offsets)
    else:
        # Version 1 (52 bytes) predates per-point timing.  Freshness remains
        # safe and usable, but dynamic consumers must not assume equal spacing.
        point_times = (None,) * MULTIRATE_POINT_COUNT

    return BoardFrameSchedule(
        version=version,
        profile=profile,
        primary_segment=primary,
        secondary_segment=secondary,
        fresh_indices=fresh_indices,
        map_age_frames=map_age,
        frame_start_ms=int.from_bytes(extension[48:52], "big"),
        sample_offset_us=point_times,
        bandwidth_discontinuity=bandwidth_discontinuity,
    )


def current_frame_fresh_indices(
    extension: bytes, *, point_count: int
) -> tuple[int, ...]:
    """Return only samples legal for a current-frame fit.

    Legacy frames are complete maps.  Reduced profiles are valid only for the
    certified 45-point table and carry an explicit bitmap.
    """

    schedule = decode_board_frame_schedule(extension)
    if schedule is None:
        return tuple(range(int(point_count)))
    if int(point_count) != MULTIRATE_POINT_COUNT:
        raise ValueError("multirate freshness requires the 45-point stress table")
    return schedule.fresh_indices


def extract_raw_scan_frames(
    buffer: bytearray, *, max_points: int = RAW_SCAN_MAX_POINTS
) -> list[bytes]:
    """Extract length-derived EE/EE scan frames from a mutable byte buffer.

    ADC samples and the new uint32 telemetry fields may legally contain the
    byte pair ``FF EF``.  Treating that pair as a delimiter can truncate an
    otherwise valid frame, so the end is derived from point/peak/extension
    lengths and only then is the two-byte footer checked.
    """

    frames: list[bytes] = []
    while True:
        start = buffer.find(b"\xEE\xEE")
        if start < 0:
            if len(buffer) > 1:
                del buffer[:-1]
            break
        if start:
            del buffer[:start]
        if len(buffer) < 4:
            break

        point_count = int.from_bytes(buffer[2:4], "big")
        if point_count <= 0 or point_count > int(max_points):
            del buffer[:2]
            continue
        position = 4 + point_count * 8
        if len(buffer) <= position:
            break
        if buffer[position] != 0xAB:
            del buffer[:2]
            continue
        position += 1

        incomplete = False
        malformed = False
        for _channel in range(4):
            if len(buffer) <= position:
                incomplete = True
                break
            peak_count = int(buffer[position])
            if peak_count > RAW_SCAN_MAX_PEAKS_PER_CHANNEL:
                malformed = True
                break
            position += 1 + peak_count * 4
            if len(buffer) < position:
                incomplete = True
                break
        if incomplete:
            break
        if malformed:
            del buffer[:2]
            continue

        position += 4  # temperature
        if len(buffer) < position + 2:
            break

        frame_end = None
        # Current layout: four legacy gain-mask bytes, then optional extension.
        if len(buffer) >= position + 8 and buffer[position + 4 : position + 6] == b"\xB5\x4D":
            extension_length = int(buffer[position + 7])
            candidate_end = position + 8 + extension_length
            if len(buffer) < candidate_end + 2:
                break
            if buffer[candidate_end : candidate_end + 2] == b"\xFF\xEF":
                frame_end = candidate_end + 2
            else:
                malformed = True
        elif len(buffer) >= position + 6 and buffer[position + 4 : position + 6] == b"\xFF\xEF":
            frame_end = position + 6
        elif len(buffer) >= position + 4 and buffer[position + 2 : position + 4] == b"\xFF\xEF":
            # Former firmware carried one byte per CH0/CH1 gain mask.
            frame_end = position + 4
        elif buffer[position : position + 2] == b"\xFF\xEF":
            frame_end = position + 2
        elif len(buffer) < position + 8:
            break
        else:
            malformed = True

        if malformed or frame_end is None:
            del buffer[:2]
            continue
        frames.append(bytes(buffer[:frame_end]))
        del buffer[:frame_end]
    return frames


@dataclass(frozen=True)
class BoardSequenceStats:
    received: int
    missing: int
    duplicates: int
    resets: int
    first_sequence: int | None
    last_sequence: int | None


class BoardSequenceTracker:
    """Track loss while respecting uint32 wrap and probable board resets."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._received = 0
        self._missing = 0
        self._duplicates = 0
        self._resets = 0
        self._first_sequence: int | None = None
        self._last_sequence: int | None = None

    def observe(self, sequence: int) -> None:
        sequence = int(sequence) & UINT32_MASK
        self._received += 1
        if self._first_sequence is None:
            self._first_sequence = sequence
            self._last_sequence = sequence
            return

        delta = (sequence - int(self._last_sequence)) & UINT32_MASK
        if delta == 0:
            self._duplicates += 1
        elif delta < UINT32_FORWARD_LIMIT:
            self._missing += delta - 1
            self._last_sequence = sequence
        else:
            # A large backwards jump is a restart/new firmware session, not
            # billions of missing frames.  Start a new baseline at this value.
            self._resets += 1
            self._last_sequence = sequence

    def stats(self) -> BoardSequenceStats:
        return BoardSequenceStats(
            received=self._received,
            missing=self._missing,
            duplicates=self._duplicates,
            resets=self._resets,
            first_sequence=self._first_sequence,
            last_sequence=self._last_sequence,
        )
