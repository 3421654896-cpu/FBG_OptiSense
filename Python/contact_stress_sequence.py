"""Multi-depth contact-stress sequence acquisition for time-series training.

One press location is visited once; at that location the tool presses through
an ordered depth ladder (different contact stress levels).  Every depth change
is a separate guarded press with the full phase recording of the coverage
acquisition (pre_contact / contact / hold / release / release_complete), so a
time-series model can learn the 45-point signature of each stress level at a
known location.
"""

from __future__ import annotations

import math

#: Ordered contact depths (mm) sampled at one press location.  Ascending order
#: keeps material relaxation monotonic between levels; the last step stays at
#: the firmware/press hardware limit (MAX_PRESS_DEPTH_MM = 0.80).
DEPTH_STEPS_MM = (0.10, 0.20, 0.35, 0.50, 0.65)

MAX_DEPTH_MM = 0.80

#: Recording phases reused from the coverage acquisition.
PHASES = ("pre_contact", "contact", "hold", "release", "release_complete")


def _positive_depths(depth_steps_mm) -> tuple[float, ...]:
    depths = tuple(float(value) for value in depth_steps_mm)
    if not depths:
        raise ValueError("至少需要一个应力深度档")
    for depth in depths:
        if not math.isfinite(depth) or not 0.0 < depth <= MAX_DEPTH_MM:
            raise ValueError(
                f"应力深度必须位于 0 < d <= {MAX_DEPTH_MM:.2f} mm：{depth}"
            )
    if any(right <= left for left, right in zip(depths, depths[1:])):
        raise ValueError("应力深度档必须严格递增")
    return depths


def build_depth_targets(point: dict, depth_steps_mm=DEPTH_STEPS_MM) -> list[dict]:
    """Expand one press location into one guarded press per stress depth.

    The XY target, contact plane and calibration offsets are copied unchanged:
    only the requested indentation depth differs between steps.  Each target
    carries ``stress_depth_mm`` as the time-series label for every frame
    recorded during that press.
    """

    depths = _positive_depths(depth_steps_mm)
    base = dict(point)
    base.pop("depth_mm", None)
    base.pop("stress_depth_mm", None)
    targets = []
    for order, depth in enumerate(depths, start=1):
        target = dict(base)
        target["stress_depth_mm"] = depth
        target["depth_mm"] = depth
        target["stress_depth_order"] = order
        target["stress_depth_count"] = len(depths)
        targets.append(target)
    return targets


def stress_sequence_request_payload(
    points,
    *,
    depth_steps_mm=DEPTH_STEPS_MM,
    dwell_seconds: float = 0.4,
    repeats: int = 1,
    xy_feed_mm_min: float = 60.0,
    z_travel_feed_mm_min: float = 60.0,
    press_feed_mm_min: float = 15.0,
    calibration_machine_minus_work=None,
) -> dict:
    """Build the run request consumed by the unified-app sequence runner."""

    depths = _positive_depths(depth_steps_mm)
    locations = [dict(point) for point in points]
    if not locations:
        raise ValueError("至少需要一个按压位置")
    if int(repeats) < 1:
        raise ValueError("重复次数必须至少为1")
    queue: list[dict] = []
    for repetition in range(1, int(repeats) + 1):
        for location in locations:
            for target in build_depth_targets(location, depths):
                target = dict(target)
                target["coverage_repetition"] = repetition
                target["dwell_seconds"] = float(dwell_seconds)
                target["xy_feed_mm_min"] = float(xy_feed_mm_min)
                target["z_travel_feed_mm_min"] = float(z_travel_feed_mm_min)
                target["press_feed_mm_min"] = float(press_feed_mm_min)
                target["approach_clearance_mm"] = 0.5
                if calibration_machine_minus_work is not None:
                    target["calibration_machine_minus_work"] = list(
                        calibration_machine_minus_work
                    )
                queue.append(target)
    return {
        "kind": "contact_stress_sequence",
        "points": locations,
        "depth_steps_mm": list(depths),
        "dwell_seconds": float(dwell_seconds),
        "repeats": int(repeats),
        "xy_feed_mm_min": float(xy_feed_mm_min),
        "z_travel_feed_mm_min": float(z_travel_feed_mm_min),
        "press_feed_mm_min": float(press_feed_mm_min),
        "motion_test": False,
        "calibration_machine_minus_work": calibration_machine_minus_work,
        "queue": queue,
    }
