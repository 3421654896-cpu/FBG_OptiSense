"""Prove (or reject) the timing prerequisite for CH1 stress bandwidth.

``15 Hz`` here means a 15 Hz *physical pressure input*, not a 15 fps display.
Acceptance therefore requires at least 30 fresh samples/s for every one of the
nine gratings, plus <=33.333 ms p99 board and host update gaps.  Cached rows in
a multirate frame are never counted as a new measurement.  A 60 Hz sampling
rate is reported as desirable engineering margin, but is not a hard gate.
Passing is necessary, not sufficient, for end-to-end 15 Hz metrology: a later
calibrated 15 Hz mechanical excitation must still measure amplitude and phase.

The benchmark never moves the CNC machine.  It opts into multirate acquisition
with the safe EXTRA -> exact 02/05 ACK -> STRESS sequence, refuses to time data
until the first current-version MAP45 frame, and always attempts an exact disarm
before requesting the PI11210 SOA shutter.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

import serial

import app_JDSU as app
from audit_stress_timing_budget import timing_trace
from fbg_lan_serial import FbgLanSerial
from frame_telemetry import (
    MULTIRATE_EXTENSION_VERSION,
    MULTIRATE_POINT_COUNT,
    MULTIRATE_PROFILE_MAP,
    MULTIRATE_PROFILE_SURVEY,
    MULTIRATE_PROFILE_TRACK_SINGLE,
    MULTIRATE_PROFILE_TRACK_WIDE,
    MULTIRATE_SENTINEL_LOCAL_OFFSETS,
    sentinel_offsets_for_version,
    BoardFrameSchedule,
    BoardFrameTiming,
    BoardSequenceTracker,
    decode_board_frame_schedule,
    decode_board_frame_start_ms,
    decode_board_frame_timing,
    extract_raw_scan_frames,
)
from stress_multirate_protocol import (
    DEFAULT_ACTIVATE_CODES,
    DEFAULT_RELEASE_CODES,
    MultirateConfiguration,
    build_safe_arm_command_sequence,
    build_safe_disarm_command_sequence,
    decode_multirate_ack,
    validate_current_arm_ack,
    validate_current_disarm_ack,
)
from verify_lan_full_function_live import wait_single
from verify_runtime_firmware import board_extension

GRATING_COUNT = 9
POINTS_PER_GRATING = 5
RECOMMENDED_OVERSAMPLE_MULTIPLE = 4.0
UINT32_MASK = 0xFFFFFFFF
UINT32_FORWARD_LIMIT = 0x80000000
PROFILE_NAMES = {
    MULTIRATE_PROFILE_MAP: "MAP45",
    MULTIRATE_PROFILE_SURVEY: "SURVEY9",
    MULTIRATE_PROFILE_TRACK_WIDE: "TRACK13_WIDE",
    MULTIRATE_PROFILE_TRACK_SINGLE: "TRACK11_SINGLE",
}


def summarize_sentinel_signal_health(trace: Sequence[dict], *, requested_version: int = MULTIRATE_EXTENSION_VERSION) -> dict:
    """Flag mostly clipped/near-zero sentinels without inventing pressure truth.

    Only entries explicitly present in the fresh-value trace are inspected.
    A zero may be real reflection loss or an estimator/path fault; timing
    alone cannot distinguish them, so this is a review flag, not a diagnosis.
    """
    result = {'available': bool(trace), 'per_grating': {}, 'review_gratings': [],
              'pressure_response_verified': False,
              'note': 'RC output signal-health audit only; no known mechanical input or stable-reference truth'}
    for grating, offset in enumerate(sentinel_offsets_for_version(requested_version)):
        point = grating * POINTS_PER_GRATING + offset
        values = [int(row['fresh_ch1_codes'][str(point)]) for row in trace
                  if str(point) in row.get('fresh_ch1_codes', {})]
        name = f'G{grating+1}'
        if not values:
            result['per_grating'][name] = {'fresh_samples': 0, 'needs_review': True}
            if trace:
                result['review_gratings'].append(name)
            continue
        near_zero = sum(value <= 2 for value in values) / len(values)
        high_clip = sum(value >= 4080 for value in values) / len(values)
        review = len(values) < 3 or near_zero >= .9 or high_clip >= .9
        result['per_grating'][name] = {
            'point_index': point, 'fresh_samples': len(values),
            'minimum_code': min(values), 'maximum_code': max(values),
            'mean_code': statistics.mean(values),
            'near_zero_fraction': near_zero, 'high_clip_fraction': high_clip,
            'needs_review': review,
        }
        if review:
            result['review_gratings'].append(name)
    return result


def _extract_scan_frames(buffer: bytearray, expected_points: int) -> list[bytes]:
    """Extract complete status-bearing frames of the requested point count."""

    frames: list[bytes] = []
    for candidate in extract_raw_scan_frames(buffer):
        point_count = int.from_bytes(candidate[2:4], "big")
        if point_count != expected_points:
            continue
        extension = board_extension(candidate)
        if extension is None or len(extension) < 28:
            continue
        frames.append(candidate)
    return frames


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summarize_periods(intervals_ms: list[float]) -> dict | None:
    """Return one common period summary without inventing missing samples."""

    if not intervals_ms:
        return None
    mean = statistics.fmean(intervals_ms)
    return {
        "intervals": len(intervals_ms),
        "mean": mean,
        "median": _percentile(intervals_ms, 50.0),
        "p95": _percentile(intervals_ms, 95.0),
        "p99": _percentile(intervals_ms, 99.0),
        "min": min(intervals_ms),
        "max": max(intervals_ms),
        "mean_hz": 1000.0 / mean if mean > 0.0 else 0.0,
    }


def _summarize_host_delivery(
    arrivals: list[float], batch_ids: list[int]
) -> dict:
    """Separate socket-receive batches from exact per-frame arrival timing.

    A single ``recv`` can contain several complete FBR frames. Those frames
    have one observable socket timestamp, so reporting zero-millisecond frame
    periods would be false precision. We report the real inter-batch gaps, an
    explicitly labelled amortized diagnostic, and expose exact per-frame
    periods only when every batch contains exactly one unique frame.
    """

    if len(arrivals) != len(batch_ids):
        raise ValueError("arrival and batch-id counts differ")

    groups: list[dict[str, int | float]] = []
    seen_batches: set[int] = set()
    repeated_noncontiguous_batches = 0
    inconsistent_batch_timestamps = 0
    for arrived_at, batch_id in zip(arrivals, batch_ids, strict=True):
        if groups and int(groups[-1]["batch_id"]) == int(batch_id):
            groups[-1]["frames"] = int(groups[-1]["frames"]) + 1
            if abs(float(groups[-1]["arrived_at"]) - float(arrived_at)) > 1e-9:
                inconsistent_batch_timestamps += 1
            continue
        if int(batch_id) in seen_batches:
            repeated_noncontiguous_batches += 1
        seen_batches.add(int(batch_id))
        groups.append(
            {
                "batch_id": int(batch_id),
                "arrived_at": float(arrived_at),
                "frames": 1,
            }
        )

    inter_batch_ms: list[float] = []
    amortized_frame_ms: list[float] = []
    nonpositive_inter_batch_intervals = 0
    for previous, current in pairwise(groups):
        delta_ms = (
            float(current["arrived_at"]) - float(previous["arrived_at"])
        ) * 1000.0
        if delta_ms <= 0.0:
            nonpositive_inter_batch_intervals += 1
            continue
        inter_batch_ms.append(delta_ms)
        # Throughput accounting only: this does not claim that individual
        # arrivals were observed inside the receive batch.
        current_frames = int(current["frames"])
        amortized = delta_ms / current_frames
        amortized_frame_ms.extend([amortized] * current_frames)

    batch_sizes = [int(group["frames"]) for group in groups]
    all_batches_single_frame = bool(groups) and all(
        size == 1 for size in batch_sizes
    )
    batch_period = _summarize_periods(inter_batch_ms)
    exact_unique_period = (
        batch_period
        if all_batches_single_frame
        and repeated_noncontiguous_batches == 0
        and inconsistent_batch_timestamps == 0
        and nonpositive_inter_batch_intervals == 0
        else None
    )
    effective_hz = 0.0
    effective_timed_frames = 0
    if len(groups) >= 2:
        elapsed_s = float(groups[-1]["arrived_at"]) - float(
            groups[0]["arrived_at"]
        )
        effective_timed_frames = sum(batch_sizes[1:])
        if elapsed_s > 0.0:
            effective_hz = effective_timed_frames / elapsed_s

    return {
        "unique_frames": len(arrivals),
        "socket_batches": len(groups),
        "multi_frame_batches": sum(size > 1 for size in batch_sizes),
        "max_frames_per_batch": max(batch_sizes, default=0),
        "mean_frames_per_batch": (
            statistics.fmean(batch_sizes) if batch_sizes else 0.0
        ),
        "nonpositive_inter_batch_intervals": (
            nonpositive_inter_batch_intervals
        ),
        "repeated_noncontiguous_batch_ids": repeated_noncontiguous_batches,
        "inconsistent_timestamps_within_batch": inconsistent_batch_timestamps,
        "socket_batch_arrival_period_ms": batch_period,
        "exact_unique_frame_arrival_period_ms": exact_unique_period,
        "amortized_unique_frame_delivery_period_ms": _summarize_periods(
            amortized_frame_ms
        ),
        "effective_timed_unique_frames": effective_timed_frames,
        "effective_unique_frame_hz": effective_hz,
        "note": (
            "同一socket接收批次的多帧只记一个实测到达时刻，"
            "不生成0 ms伪周期；只有每批一帧时才给出精确单帧到达周期。"
        ),
    }


def _summarize_board_telemetry(
    samples: list[BoardFrameTiming | None],
) -> dict:
    """Summarize MCU acquisition duration and raw-frame sequence continuity."""

    tracker = BoardSequenceTracker()
    acquisition_ms: list[float] = []
    missing_telemetry_frames = 0
    for sample in samples:
        if sample is None:
            missing_telemetry_frames += 1
            continue
        tracker.observe(sample.sequence)
        acquisition_ms.append(sample.acquisition_duration_ms)

    sequence = tracker.stats()
    acquisition_summary = None
    if acquisition_ms:
        acquisition_summary = {
            "mean": statistics.fmean(acquisition_ms),
            "median": _percentile(acquisition_ms, 50.0),
            "p95": _percentile(acquisition_ms, 95.0),
            "p99": _percentile(acquisition_ms, 99.0),
            "min": min(acquisition_ms),
            "max": max(acquisition_ms),
        }
    return {
        "available_frames": len(acquisition_ms),
        "missing_telemetry_frames": missing_telemetry_frames,
        "acquisition_ms": acquisition_summary,
        "sequence": {
            "first": sequence.first_sequence,
            "last": sequence.last_sequence,
            "missing": sequence.missing,
            "duplicates": sequence.duplicates,
            "resets": sequence.resets,
        },
    }


def _summarize_board_start_ticks(ticks: list[int | None]) -> dict | None:
    """Summarize true MCU frame-start cadence from the MAP45 extension."""

    intervals: list[float] = []
    for left, right in pairwise(ticks):
        if left is None or right is None:
            continue
        delta = (int(right) - int(left)) & 0xFFFFFFFF
        if 0 < delta < 0x80000000:
            intervals.append(float(delta))  # HAL tick is one millisecond.
    if not intervals:
        return None
    return _summarize_periods(intervals)


def _unwrap_frame_start_us(
    schedules: Sequence[BoardFrameSchedule | None],
) -> tuple[int | None, ...]:
    """Unwrap the board's uint32 millisecond tick without host-time guesses."""

    unwrapped: list[int | None] = []
    previous_raw: int | None = None
    elapsed_ms = 0
    for schedule in schedules:
        if schedule is None:
            unwrapped.append(None)
            continue
        current_raw = int(schedule.frame_start_ms) & UINT32_MASK
        if previous_raw is None:
            elapsed_ms = 0
        else:
            delta_ms = (current_raw - previous_raw) & UINT32_MASK
            if delta_ms == 0 or delta_ms >= UINT32_FORWARD_LIMIT:
                raise ValueError("board frame-start ticks are not strictly forward")
            elapsed_ms += delta_ms
        previous_raw = current_raw
        unwrapped.append(elapsed_ms * 1000)
    return tuple(unwrapped)


def _summarize_schedule_freshness(
    schedules: Sequence[BoardFrameSchedule | None],
    *, requested_version: int = MULTIRATE_EXTENSION_VERSION,
) -> dict:
    """Report true per-grating sample cadence from versioned freshness metadata.

    One grating update becomes available at the latest fresh sample time for
    that grating in a frame.  Historical cache rows have no sample time and do
    not create an event.  This definition is deliberately conservative for a
    multi-point ROI because a fit cannot use a point that has not yet arrived.
    """

    profile_counts: Counter[str] = Counter()
    fresh_count_distribution: Counter[str] = Counter()
    invalid_schedule_frames = 0
    bandwidth_discontinuity_frames = 0
    current_version_frames = 0
    complete_map_frames = 0
    frame_start_us = _unwrap_frame_start_us(schedules)
    event_times_us: list[list[int]] = [[] for _ in range(GRATING_COUNT)]

    for schedule, start_us in zip(schedules, frame_start_us, strict=True):
        if schedule is None:
            invalid_schedule_frames += 1
            profile_counts["MISSING"] += 1
            fresh_count_distribution["0"] += 1
            continue
        profile_name = PROFILE_NAMES.get(
            schedule.profile,
            f"UNKNOWN_{schedule.profile}_{len(schedule.fresh_indices)}",
        )
        profile_counts[profile_name] += 1
        fresh_count_distribution[str(len(schedule.fresh_indices))] += 1
        if schedule.version != requested_version or not schedule.has_point_timing:
            invalid_schedule_frames += 1
            continue
        current_version_frames += 1
        complete_map_frames += int(schedule.is_complete_map)
        bandwidth_discontinuity_frames += int(
            bool(getattr(schedule, "bandwidth_discontinuity", False))
        )
        if start_us is None:  # Defensive; a concrete schedule always has a tick.
            invalid_schedule_frames += 1
            continue

        fresh_set = set(schedule.fresh_indices)
        for grating in range(GRATING_COUNT):
            first = grating * POINTS_PER_GRATING
            offsets = [
                schedule.sample_offset_us[index]
                for index in range(first, first + POINTS_PER_GRATING)
                if index in fresh_set
            ]
            concrete_offsets = [
                int(offset) for offset in offsets if offset is not None
            ]
            if concrete_offsets:
                event_times_us[grating].append(start_us + max(concrete_offsets))

    per_grating: dict[str, dict] = {}
    for grating, times_us in enumerate(event_times_us, 1):
        intervals_ms = [
            (right - left) / 1000.0
            for left, right in pairwise(times_us)
            if right > left
        ]
        per_grating[f"G{grating}"] = {
            "fresh_events": len(times_us),
            "fresh_inter_sample_period_ms": _summarize_periods(intervals_ms),
        }

    fresh_counts = [
        len(schedule.fresh_indices) if schedule is not None else 0
        for schedule in schedules
    ]
    return {
        "frames": len(schedules),
        "schedule_version": requested_version,
        "current_version_frames": current_version_frames,
        "invalid_or_legacy_schedule_frames": invalid_schedule_frames,
        "bandwidth_discontinuity_frames": bandwidth_discontinuity_frames,
        "complete_map_frames": complete_map_frames,
        "profile_distribution": dict(sorted(profile_counts.items())),
        "fresh_point_count_distribution": dict(
            sorted(fresh_count_distribution.items(), key=lambda item: int(item[0]))
        ),
        "fresh_points_per_frame": {
            "mean": statistics.fmean(fresh_counts) if fresh_counts else 0.0,
            "min": min(fresh_counts, default=0),
            "max": max(fresh_counts, default=0),
        },
        "per_grating": per_grating,
        "freshness_note": (
            "G1..G9只统计fresh bitmap内当帧新采样；缓存行不是新测量。"
        ),
    }


def _strict_bandwidth_acceptance(
    *,
    signal_bandwidth_hz: float,
    measured_frame_hz: float,
    host_exact_period: dict | None,
    board_frame_period: dict | None,
    schedule_freshness: dict,
) -> tuple[dict, dict]:
    """Return hard Nyquist gates and a non-gating 4x margin assessment."""

    if not math.isfinite(signal_bandwidth_hz) or signal_bandwidth_hz <= 0.0:
        raise ValueError("signal bandwidth must be a positive finite number")
    nyquist_rate_hz = 2.0 * signal_bandwidth_hz
    max_period_ms = 1000.0 / nyquist_rate_hz
    recommended_rate_hz = RECOMMENDED_OVERSAMPLE_MULTIPLE * signal_bandwidth_hz
    recommended_period_ms = 1000.0 / recommended_rate_hz

    grating_periods = [
        item["fresh_inter_sample_period_ms"]
        for item in schedule_freshness["per_grating"].values()
    ]
    all_gratings_have_periods = all(period is not None for period in grating_periods)
    all_gratings_mean_rate_ok = all_gratings_have_periods and all(
        period["mean_hz"] >= nyquist_rate_hz for period in grating_periods
    )
    all_gratings_p99_ok = all_gratings_have_periods and all(
        period["p99"] <= max_period_ms for period in grating_periods
    )
    all_gratings_max_gap_ok = all_gratings_have_periods and all(
        period["max"] <= max_period_ms for period in grating_periods
    )

    hard = {
        "mean_frame_rate_meets_nyquist": measured_frame_hz >= nyquist_rate_hz,
        "host_p99_meets_nyquist_interval": (
            host_exact_period is not None
            and host_exact_period["p99"] <= max_period_ms
        ),
        "board_p99_meets_nyquist_interval": (
            board_frame_period is not None
            and board_frame_period["p99"] <= max_period_ms
        ),
        "all_gratings_have_fresh_intervals": all_gratings_have_periods,
        "all_gratings_mean_fresh_rate_meets_nyquist": all_gratings_mean_rate_ok,
        "all_gratings_fresh_p99_meets_nyquist_interval": all_gratings_p99_ok,
        "all_gratings_fresh_max_gap_meets_nyquist_interval": (
            all_gratings_max_gap_ok
        ),
        "no_reported_bandwidth_discontinuity": (
            schedule_freshness["bandwidth_discontinuity_frames"] == 0
        ),
        "all_frames_have_current_multirate_schedule": (
            schedule_freshness["frames"] > 0
            and schedule_freshness["current_version_frames"]
            == schedule_freshness["frames"]
            and schedule_freshness["invalid_or_legacy_schedule_frames"] == 0
        ),
    }
    recommended = {
        "target_fresh_rate_hz": recommended_rate_hz,
        "target_max_period_ms": recommended_period_ms,
        "mean_frame_rate_reaches_4x_margin": measured_frame_hz >= recommended_rate_hz,
        "host_p99_reaches_4x_margin": (
            host_exact_period is not None
            and host_exact_period["p99"] <= recommended_period_ms
        ),
        "board_p99_reaches_4x_margin": (
            board_frame_period is not None
            and board_frame_period["p99"] <= recommended_period_ms
        ),
        "all_gratings_mean_fresh_rate_reaches_4x_margin": (
            all_gratings_have_periods
            and all(
                period["mean_hz"] >= recommended_rate_hz
                for period in grating_periods
            )
        ),
        "all_gratings_fresh_p99_reaches_4x_margin": (
            all_gratings_have_periods
            and all(
                period["p99"] <= recommended_period_ms
                for period in grating_periods
            )
        ),
        "all_gratings_fresh_max_gap_reaches_4x_margin": (
            all_gratings_have_periods
            and all(
                period["max"] <= recommended_period_ms
                for period in grating_periods
            )
        ),
        "note": "60 Hz是15 Hz输入的4倍采样裕量，只报告，不放宽也不替代30 Hz硬验收。",
    }
    return hard, recommended


def _extract_multirate_ack_frames(buffer: bytearray) -> list[bytes]:
    """Extract only exact 20-byte FF FF 02 05 multirate ACK frames."""

    marker = b"\xFF\xFF\x02\x05"
    frames: list[bytes] = []
    while True:
        start = buffer.find(marker)
        if start < 0:
            if len(buffer) > len(marker) - 1:
                del buffer[: -(len(marker) - 1)]
            break
        if start:
            del buffer[:start]
        if len(buffer) < 20:
            break
        frames.append(bytes(buffer[:20]))
        del buffer[:20]
    return frames


def _wait_raw_multirate_ack(device, timeout_s: float = 3.0):
    """Return one structurally valid exact-header 02/05 ACK."""

    buffer = bytearray()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        waiting = int(device.in_waiting or 0)
        chunk = device.read(min(max(waiting, 64), 4096))
        if not chunk:
            time.sleep(0.002)
            continue
        buffer.extend(chunk)
        for frame in _extract_multirate_ack_frames(buffer):
            ack = decode_multirate_ack(frame)
            if ack is None:
                raise RuntimeError("收到畸形的02/05多速率ACK")
            return ack
    raise RuntimeError("等待板卡精确02/05多速率ACK超时")


def _wait_multirate_ack(
    device,
    expected: MultirateConfiguration,
    timeout_s: float = 3.0,
    *, requested_version: int = MULTIRATE_EXTENSION_VERSION,
):
    """Wait for an accepted arm ACK whose echoed configuration is exact."""

    expected.validate()
    ack = _wait_raw_multirate_ack(device, timeout_s)
    try:
        return validate_current_arm_ack(ack, expected, requested_version=requested_version)
    except ValueError as exc:
        raise RuntimeError(f"02/05 arm ACK未通过当前协议校验：{exc}") from exc


def _wait_multirate_disarm_ack(device, timeout_s: float = 3.0):
    """Require accepted v3 disabled state; preserved thresholds are legal."""

    ack = _wait_raw_multirate_ack(device, timeout_s)
    try:
        return validate_current_disarm_ack(ack)
    except ValueError as exc:
        raise RuntimeError(
            f"02/05 disarm ACK未通过当前协议校验：{exc}"
        ) from exc


def _safe_disarm(device) -> bool:
    """Enter EXTRA and require an exact accepted disabled 02/05 ACK."""

    enter_extra, disarm = build_safe_disarm_command_sequence()
    device.write(enter_extra)
    time.sleep(0.12)
    device.reset_input_buffer()
    device.write(disarm)
    _wait_multirate_disarm_ack(device)
    return True


def _require_initial_current_map(
    schedule: BoardFrameSchedule | None,
    timing: BoardFrameTiming | None,
    *, requested_version: int = MULTIRATE_EXTENSION_VERSION,
) -> None:
    """Reject timing startup until the opt-in contract is unambiguous."""

    if (
        schedule is None
        or schedule.version != requested_version
        or not schedule.is_complete_map
    ):
        profile = None if schedule is None else schedule.profile
        version = None if schedule is None else schedule.version
        raise RuntimeError(
            "进入STRESS后首帧不是合法的当前版本multirate MAP45："
            f"version={version}, profile={profile}"
        )
    if timing is None:
        raise RuntimeError("首个当前版本MAP45缺少板端时序字段")


def _close_shutter(device) -> bool:
    device.write(app.build_work_mode_command(2))
    time.sleep(0.12)
    device.reset_input_buffer()
    command = app.build_single_value_dac_command(
        (0, 0, 0, 0, 0), soa_mode=app.PI11210_SOA_SHUTTER_MODE
    )
    expected = tuple(
        int.from_bytes(command[4 + index * 2 : 6 + index * 2], "big")
        for index in range(5)
    )
    device.write(command)

    def decode(frame: bytes):
        if frame[:4] != b"\xff\xff\x02\x00":
            return None
        returned = tuple(
            int.from_bytes(frame[4 + index * 2 : 6 + index * 2], "big")
            for index in range(5)
        )
        if returned != expected or frame[14] != app.PI11210_SOA_SHUTTER_MODE:
            return None
        app.UnlimitedAccuracyWorker._validate_ack_status(
            frame, app.PI11210_SOA_SHUTTER_MODE
        )
        return True

    return bool(wait_single(device, decode, 3.0))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.3.46")
    parser.add_argument("--transport", choices=("usb", "lan"), default="usb")
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--baud", type=int, default=2_000_000)
    parser.add_argument("--schedule-version", type=int, choices=(4, 5), default=4,
                        help="4=existing path; 5=experimental G8 right sentinel")
    parser.add_argument(
        "--selector",
        type=int,
        choices=(0, 1, 2, 3),
        default=1,
        help="CH0/CH1 selector: 0=2k, 1=40k, 2=5k, 3=20k",
    )
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--expected-points", type=int, default=45)
    parser.add_argument(
        "--signal-bandwidth-hz",
        type=float,
        default=15.0,
        help=(
            "physical input bandwidth; hard sampling gate is twice this value "
            "for every grating (default: 15 Hz input -> >=30 fresh samples/s)"
        ),
    )
    parser.add_argument(
        "--map-period",
        type=int,
        default=0,
        help=(
            "0 keeps the initial MAP45 but suppresses periodic MAPs during the "
            "timed bandwidth proof; 3..100 enables an audit cadence"
        ),
    )
    parser.add_argument(
        "--activate-threshold-codes", type=int, default=DEFAULT_ACTIVATE_CODES
    )
    parser.add_argument(
        "--release-threshold-codes", type=int, default=DEFAULT_RELEASE_CODES
    )
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument(
        "--trace-fresh-values",
        action="store_true",
        help=(
            "append each frame's fresh CH1 RC codes, MCU completion offsets "
            "and original wire bytes for trigger/timing audit (not ADC aperture times)"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.frames < 3:
        raise ValueError("--frames must be at least 3")
    if args.expected_points != MULTIRATE_POINT_COUNT:
        raise ValueError("multirate bandwidth proof requires the certified 45-point table")
    if not math.isfinite(args.signal_bandwidth_hz) or args.signal_bandwidth_hz <= 0:
        raise ValueError("--signal-bandwidth-hz must be a positive finite value")

    multirate_configuration = MultirateConfiguration(
        enabled=True,
        map_period_frames=args.map_period,
        activate_threshold_codes=args.activate_threshold_codes,
        release_threshold_codes=args.release_threshold_codes,
    ).validate()

    if args.transport == "usb":
        device = serial.Serial(
            args.port, args.baud, timeout=0.04, write_timeout=3.0
        )
    else:
        device = FbgLanSerial(board_host=args.host, timeout=0.04, write_timeout=3.0)
    buffer = bytearray()
    arrivals: list[float] = []
    arrival_batch_ids: list[int] = []
    active_masks: list[int] = []
    firmware_versions: list[str] = []
    board_timing_samples: list[BoardFrameTiming | None] = []
    board_start_ticks: list[int | None] = []
    board_schedules: list[BoardFrameSchedule | None] = []
    fresh_value_trace: list[dict] = []
    seen_board_sequences: set[int] = set()
    duplicate_board_sequences = 0
    arrival_timestamp_fallbacks = 0
    read_batch_sequence = 0
    initial_map_confirmed = False
    arm_ack_confirmed = False
    disarm_ack_confirmed = False
    shutter_confirmed = False
    cleanup_errors: list[str] = []
    report: dict | None = None
    output: Path | None = None
    try:
        if args.transport == "lan":
            device.open()
            deadline = time.monotonic() + min(args.timeout, 20.0)
            while time.monotonic() < deadline and not device.connected:
                time.sleep(0.02)
            if not device.connected:
                raise RuntimeError("未发现局域网板卡的全功能通道")
        else:
            device.dtr = True
            time.sleep(0.20)

        enter_extra, arm_multirate, enter_stress = build_safe_arm_command_sequence(
            multirate_configuration, schedule_version=args.schedule_version
        )
        device.write(enter_extra)
        time.sleep(0.12)
        device.reset_input_buffer()
        device.write(
            app.build_stress_feedback_command(args.selector, args.selector)
        )
        feedback = wait_single(device, app.decode_stress_feedback_status)
        if feedback != ((args.selector, args.selector), app.ACK_VALUE):
            raise RuntimeError(f"应力跨阻IO回读异常：{feedback}")

        device.reset_input_buffer()
        device.write(arm_multirate)
        _wait_multirate_ack(device, multirate_configuration, requested_version=args.schedule_version)
        arm_ack_confirmed = True
        device.reset_input_buffer()
        device.write(enter_stress)

        collection_deadline = time.monotonic() + args.timeout
        while len(arrivals) < args.frames and time.monotonic() < collection_deadline:
            waiting = int(device.in_waiting or 0)
            chunk = device.read(min(max(waiting, 256), 8192))
            if not chunk:
                continue
            read_batch_sequence += 1
            buffer.extend(chunk)
            observed_at = time.perf_counter()
            for frame in _extract_scan_frames(buffer, args.expected_points):
                extension = board_extension(frame)
                if extension is None:
                    continue
                timing = decode_board_frame_timing(extension)
                schedule = decode_board_frame_schedule(extension)
                if args.transport == "lan":
                    arrival_info = device.take_frame_arrival_info(frame)
                    if arrival_info is None:
                        arrived_at = observed_at
                        arrival_batch_id = -read_batch_sequence
                        arrival_timestamp_fallbacks += 1
                    else:
                        arrived_at, arrival_batch_id = arrival_info
                else:
                    arrived_at = observed_at
                    arrival_batch_id = read_batch_sequence

                if not initial_map_confirmed:
                    _require_initial_current_map(schedule, timing, requested_version=args.schedule_version)
                    initial_map_confirmed = True
                    assert timing is not None
                    seen_board_sequences.add(timing.sequence)
                    # Synchronisation frame is deliberately outside the timed
                    # benchmark.  LAN arrival metadata must still be consumed
                    # above so subsequent frame/batch attribution stays exact.
                    continue

                if schedule is None or schedule.version != args.schedule_version:
                    raise RuntimeError("计时阶段收到非当前版本multirate应力帧")
                if timing is not None:
                    if timing.sequence in seen_board_sequences:
                        duplicate_board_sequences += 1
                        # A reused frame is never a fresh pressure sample.
                        continue
                    seen_board_sequences.add(timing.sequence)
                board_timing_samples.append(timing)
                board_start_ticks.append(decode_board_frame_start_ms(extension))
                board_schedules.append(schedule)
                if args.trace_fresh_values:
                    fresh_value_trace.append(timing_trace(frame))
                arrivals.append(arrived_at)
                arrival_batch_ids.append(arrival_batch_id)
                active_masks.append(int(extension[16]))
                version = int.from_bytes(extension[24:28], "big")
                firmware_versions.append(
                    f"{(version >> 16) & 0xFF}.{(version >> 8) & 0xFF}.{version & 0xFF}"
                )
                if len(arrivals) >= args.frames:
                    break

        if not initial_map_confirmed:
            raise RuntimeError("未收到进入STRESS后的首个当前版本MAP45")
        if len(arrivals) < args.frames:
            raise RuntimeError(
                f"只收到{len(arrivals)}/{args.frames}个唯一计时帧"
            )

        host_delivery = _summarize_host_delivery(
            arrivals, arrival_batch_ids
        )
        measured_hz = host_delivery["effective_unique_frame_hz"]
        host_exact_period = host_delivery[
            "exact_unique_frame_arrival_period_ms"
        ]
        host_batch_period = host_delivery["socket_batch_arrival_period_ms"]
        # Compatibility alias for timing-reference readers. If batching
        # occurred this is deliberately the conservative directly observed
        # batch period; exact unique-frame timing remains unavailable.
        host_period = host_exact_period or host_batch_period
        board_telemetry = _summarize_board_telemetry(board_timing_samples)
        board_frame_period = _summarize_board_start_ticks(board_start_ticks)
        schedule_freshness = _summarize_schedule_freshness(board_schedules, requested_version=args.schedule_version)
        board_sequence = board_telemetry["sequence"]
        acquisition_summary = board_telemetry["acquisition_ms"]
        host_mean_ms = (
            host_exact_period["mean"] if host_exact_period is not None else None
        )
        bandwidth_acceptance, recommended_margin = _strict_bandwidth_acceptance(
            signal_bandwidth_hz=args.signal_bandwidth_hz,
            measured_frame_hz=measured_hz,
            host_exact_period=host_exact_period,
            board_frame_period=board_frame_period,
            schedule_freshness=schedule_freshness,
        )
        nyquist_rate_hz = 2.0 * args.signal_bandwidth_hz
        hard_max_period_ms = 1000.0 / nyquist_rate_hz
        report = {
            "schema": "stress_physical_bandwidth_benchmark_v4",
            "pass_scope": "sampling_timing_prerequisite_only_not_end_to_end_pressure",
            "sentinel_signal_health": summarize_sentinel_signal_health(fresh_value_trace, requested_version=args.schedule_version),
            "adc_value_kind": "firmware_rc_estimate",
            "adc_value_note": (
                "fresh_ch1_codes是本帧固件两次ADC读取经RC预测后的码值，"
                "不是预测前的直接ADC读数；时序证据不代表稳定值预测准确率。"
            ),
            "measurement_scope": (
                "验证ADC/帧链路满足15 Hz物理输入的Nyquist时序必要条件；"
                "通过不等于已验证激光器、光栅、模拟前端和机械结构的"
                "15 Hz幅相响应，后者仍需标定正弦按压试验。"
            ),
            "transport": args.transport,
            "endpoint": (
                device.board_ip if args.transport == "lan" else args.port
            ),
            "feedback_selector": args.selector,
            "point_count": args.expected_points,
            "physical_signal_bandwidth_hz": args.signal_bandwidth_hz,
            "nyquist_minimum_fresh_rate_hz": nyquist_rate_hz,
            "nyquist_maximum_update_period_ms": hard_max_period_ms,
            "recommended_sampling_margin": recommended_margin,
            "multirate_configuration": {
                "schedule_version": args.schedule_version,
                "map_period_frames": multirate_configuration.map_period_frames,
                "activate_threshold_codes": (
                    multirate_configuration.activate_threshold_codes
                ),
                "release_threshold_codes": (
                    multirate_configuration.release_threshold_codes
                ),
                "exact_arm_ack_confirmed": arm_ack_confirmed,
                "initial_current_version_map45_confirmed": initial_map_confirmed,
            },
            "timed_intervals": host_delivery["effective_timed_unique_frames"],
            "measured_frame_hz": measured_hz,
            "measured_hz": measured_hz,
            "measured_hz_legacy_field_note": (
                "仅为帧吞吐率兼容字段，不代表可测物理信号带宽；"
                "15 Hz输入必须看G1..G9 fresh采样与30 Hz硬门槛。"
            ),
            "period_ms": host_period,
            "host_unique_frame_arrival_period_ms": host_exact_period,
            "host_socket_batch_arrival_period_ms": host_batch_period,
            "host_amortized_unique_frame_delivery_period_ms": host_delivery[
                "amortized_unique_frame_delivery_period_ms"
            ],
            "active_channel_masks": sorted(set(active_masks)),
            "firmware_versions": sorted(set(firmware_versions)),
            "board_telemetry": board_telemetry,
            "board_frame_period_ms": board_frame_period,
            "multirate_freshness": schedule_freshness,
            "fresh_value_trace": (
                fresh_value_trace if args.trace_fresh_values else None
            ),
            "host_period_minus_board_acquisition_ms": (
                host_mean_ms - acquisition_summary["mean"]
                if acquisition_summary is not None and host_mean_ms is not None
                else None
            ),
            "host_receive_batching": host_delivery,
            "arrival_timestamp_fallbacks": arrival_timestamp_fallbacks,
            "lan_raw_transport": (
                device.transport_stats if args.transport == "lan" else None
            ),
            "acceptance": {
                **bandwidth_acceptance,
                "exact_multirate_arm_ack": arm_ack_confirmed,
                "initial_frame_is_current_version_map45": initial_map_confirmed,
                "host_arrivals_are_unbatched": (
                    host_delivery["multi_frame_batches"] == 0
                    and host_delivery["nonpositive_inter_batch_intervals"] == 0
                    and host_delivery["repeated_noncontiguous_batch_ids"] == 0
                    and host_delivery["inconsistent_timestamps_within_batch"] == 0
                ),
                "ch1_only_after_initial_map": set(active_masks) == {0x02},
                "board_timing_available_for_all_timed_frames": (
                    board_telemetry["available_frames"] == args.frames
                    and board_telemetry["missing_telemetry_frames"] == 0
                ),
                "board_sequence_contiguous": (
                    board_sequence["missing"] == 0
                    and board_sequence["duplicates"] == 0
                    and board_sequence["resets"] == 0
                    and duplicate_board_sequences == 0
                ),
                "board_frame_start_available_for_all_timed_frames": (
                    board_frame_period is not None
                    and board_frame_period["intervals"] == args.frames - 1
                ),
            },
        }
        output = args.output or Path(__file__).with_name(
            "stress_realtime_benchmark_latest.json"
        )
    finally:
        connected = False
        try:
            connected = bool(
                device.is_open
                and (args.transport == "usb" or device.connected)
            )
            if connected:
                try:
                    disarm_ack_confirmed = _safe_disarm(device)
                except Exception as exc:  # noqa: BLE001 - shutter must still run
                    cleanup_errors.append(f"multirate_disarm: {exc}")
                try:
                    shutter_confirmed = _close_shutter(device)
                    if not shutter_confirmed:
                        cleanup_errors.append("soa_shutter: ACK未确认")
                except Exception as exc:  # noqa: BLE001 - report after closing port
                    cleanup_errors.append(f"soa_shutter: {exc}")
            elif report is not None:
                cleanup_errors.append("计时完成后设备已断开，无法确认disarm/关光")
        finally:
            device.close()
            print(f"multirate_disarm_confirmed={int(disarm_ack_confirmed)}")
            print(f"soa_shutter_confirmed={int(shutter_confirmed)}")
        if report is not None:
            report["safety_cleanup"] = {
                "connected_before_cleanup": connected,
                "exact_multirate_disarm_ack": disarm_ack_confirmed,
                "exact_soa_shutter_ack": shutter_confirmed,
                "errors": list(cleanup_errors),
            }
            report["acceptance"]["exact_multirate_disarm_ack"] = (
                disarm_ack_confirmed
            )
            report["acceptance"]["exact_soa_shutter_ack"] = shutter_confirmed
            report["pass"] = all(report["acceptance"].values())
            assert output is not None
            output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))
            print(f"report={output.resolve()}")
        if cleanup_errors:
            raise RuntimeError("安全收尾失败：" + " | ".join(cleanup_errors))

    assert report is not None
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
