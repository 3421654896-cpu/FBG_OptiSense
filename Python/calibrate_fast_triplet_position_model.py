"""Select a position model from two complete F27 contact repeats.

One physical press is one indivisible sample.  The two opposite-order repeats
are used only for candidate development; a selected candidate is locked before
any third repeat is acquired.  Static CNC contacts do not prove physical 15 Hz
response or true contact area.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from calibrate_fast_flank_position_model import evaluate_repeat_splits


POINT_RE = re.compile(r"_p(?P<point>\d{2})_.*_r(?P<repeat>\d+)_")
SCHEMA = "bounded_indentation_fast_triplet_commission_v1"
EXPECTED_POINTS = 27


class CaptureRejected(ValueError):
    pass


@dataclass(frozen=True)
class ContactSample:
    path: Path
    point_index: int
    repeat: int
    x_mm: float
    y_mm: float
    contact_sequence: int
    release_sequence: int
    hold_frames: int
    features: dict[str, np.ndarray]
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    return values / max(float(np.linalg.norm(values)), 1e-9)


def _group_center(values: np.ndarray) -> np.ndarray:
    shaped = np.asarray(values, dtype=float).reshape(9, 3)
    return (shaped - np.mean(shaped, axis=1, keepdims=True)).ravel()


def _signed_peak(matrix: np.ndarray) -> np.ndarray:
    return np.take_along_axis(
        matrix,
        np.argmax(np.abs(matrix), axis=0, keepdims=True),
        axis=0,
    )[0]


def _feature_families(
    baseline_first: np.ndarray,
    baseline_second: np.ndarray,
    hold_first: np.ndarray,
    hold_second: np.ndarray,
    ramp_second: np.ndarray,
) -> dict[str, np.ndarray]:
    for label, values in (
        ("baseline first", baseline_first),
        ("baseline second", baseline_second),
        ("hold first", hold_first),
        ("hold second", hold_second),
        ("ramp second", ramp_second),
    ):
        if values.ndim != 2 or values.shape[1] != EXPECTED_POINTS:
            raise CaptureRejected(f"{label} must be Nx{EXPECTED_POINTS}, got {values.shape}")
    if baseline_second.shape[0] != 64 or hold_second.shape[0] < 12 or ramp_second.shape[0] < 12:
        raise CaptureRejected("baseline/hold/ramp windows are too short")

    eps = 1.0
    b1 = np.median(baseline_first, axis=0)
    b2 = np.median(baseline_second, axis=0)
    h1 = np.median(hold_first, axis=0)
    h2 = np.median(hold_second, axis=0)
    response_first = np.log((h1 + eps) / (b1 + eps))
    response_second = np.log((h2 + eps) / (b2 + eps))
    response_settle = np.log((h2 + eps) / (h1 + eps)) - np.log((b2 + eps) / (b1 + eps))
    response_first_centered = response_first - np.median(response_first)
    response_second_centered = response_second - np.median(response_second)
    response_settle_centered = response_settle - np.median(response_settle)
    response_first_group = _group_center(response_first)
    response_second_group = _group_center(response_second)
    response_settle_group = _group_center(response_settle)

    baseline_triplet = (b2.reshape(9, 3) + eps)
    hold_triplet = (h2.reshape(9, 3) + eps)
    triplet_shape = (
        hold_triplet / np.sum(hold_triplet, axis=1, keepdims=True)
        - baseline_triplet / np.sum(baseline_triplet, axis=1, keepdims=True)
    ).ravel()

    temporal = []
    for part in np.array_split(hold_second, 3):
        part_response = np.log((np.median(part, axis=0) + eps) / (b2 + eps))
        temporal.append(_group_center(part_response))
    temporal_group = np.concatenate(temporal)

    ramp_response = np.log((ramp_second + eps) / (b2 + eps))
    ramp_group = np.stack([_group_center(row) for row in ramp_response])
    ramp_peak = _signed_peak(ramp_group)
    ramp_end = np.median(ramp_group[-5:], axis=0)
    ramp_area = np.median(ramp_group, axis=0)

    return {
        "second_log_centered": response_second_centered,
        "second_log_centered_unit": _unit(response_second_centered),
        "second_log_group": response_second_group,
        "second_log_group_unit": _unit(response_second_group),
        "first_log_group_unit": _unit(response_first_group),
        "settle_log_group_unit": _unit(response_settle_group),
        "triplet_shape": triplet_shape,
        "triplet_shape_unit": _unit(triplet_shape),
        "temporal_group_unit": _unit(temporal_group),
        "first_second_group_unit": np.concatenate(
            [_unit(response_first_group), _unit(response_second_group), _unit(response_settle_group)]
        ),
        "ramp_peak_group_unit": _unit(ramp_peak),
        "ramp_end_group_unit": _unit(ramp_end),
        "ramp_area_group_unit": _unit(ramp_area),
        "ramp_combined_group_unit": np.concatenate(
            [_unit(ramp_peak), _unit(ramp_end), _unit(ramp_area)]
        ),
        "held_combined_unit": np.concatenate(
            [_unit(response_second_group), _unit(triplet_shape), _unit(temporal_group)]
        ),
    }


def _nearest_sequence(receipts: list[dict[str, Any]], host_ns: int) -> int:
    if not receipts:
        raise CaptureRejected("missing frame receipts")
    return int(min(receipts, key=lambda row: abs(int(row["host_monotonic_ns"]) - host_ns))["sequence"])


def _one_event(events: list[dict[str, Any]], kind: str, *, prefix: bool = False) -> dict[str, Any]:
    matches = [
        event
        for event in events
        if str(event.get("kind", "")).startswith(kind)
        if prefix or str(event.get("kind", "")) == kind
    ]
    if len(matches) != 1:
        raise CaptureRejected(f"expected exactly one {kind} event, got {len(matches)}")
    return matches[0]


def _adc_matrix(frames: list[dict[str, Any]], start: int, stop: int, field: str) -> np.ndarray:
    selected = [
        [int(record[field]) for record in frame.get("records", ())]
        for frame in frames
        if start <= int(frame.get("sequence", -1)) < stop
    ]
    if not selected or any(len(row) != EXPECTED_POINTS for row in selected):
        raise CaptureRejected(f"invalid {field} matrix for frames {start}:{stop}")
    return np.asarray(selected, dtype=float)


def load_contact_sample(path: Path, *, repeat: int, expected_depth_mm: float) -> ContactSample:
    data = json.loads(path.read_text("utf-8"))
    commissioning = data.get("commissioning") or {}
    target = commissioning.get("target") or {}
    match = POINT_RE.search(path.name)
    point_index = int(target.get("point_index", match.group("point") if match else 0))
    if commissioning.get("schema") != SCHEMA or commissioning.get("stream_protocol") != "triplet27":
        raise CaptureRejected("capture does not use the F27 local-baseline protocol")
    if not data.get("complete") or commissioning.get("capture_error") or commissioning.get("motion_failures"):
        raise CaptureRejected("capture is incomplete or has a motion failure")
    if not 1 <= point_index <= 17 or repeat not in (1, 2):
        raise CaptureRejected("invalid point/repeat label")
    if abs(float(commissioning.get("depth_mm", math.nan)) - expected_depth_mm) > 1e-9:
        raise CaptureRejected("indentation depth mismatch")
    cleanup = data.get("safety_cleanup") or {}
    if cleanup.get("exact_disarm_ack") is not True or cleanup.get("exact_soa_shutter_ack") is not True:
        raise CaptureRejected("laser shutdown was not acknowledged")
    analysis = data.get("analysis") or {}
    if analysis.get("meets_sampling_rate_for_15hz") is not True or float(
        analysis.get("board_interval_max_us", math.inf)
    ) > 1_000_000.0 / 30.0:
        raise CaptureRejected("15 Hz sampling gate failed")

    frames = data.get("frames") or []
    receipts = commissioning.get("frame_receipts") or []
    cycles = int(data.get("cycles_requested", -1))
    expected_sequences = list(range(cycles))
    if not 640 <= cycles <= 1024:
        raise CaptureRejected("capture length is outside the bounded protocol")
    for label, values in (("frame", frames), ("receipt", receipts)):
        if [int(item.get("sequence", -1)) for item in values] != expected_sequences:
            raise CaptureRejected(f"{label} sequence is incomplete or reordered")
    if any(len(frame.get("records") or ()) != EXPECTED_POINTS for frame in frames):
        raise CaptureRejected("not every frame contains 27 records")

    events = commissioning.get("events") or []
    descent = _one_event(events, "contact_descent_started")
    contact = _one_event(events, "contact_reached", prefix=True)
    release = _one_event(events, "release_started")
    final = _one_event(events, "final_safe_state_confirmed")
    contact_status = contact.get("status") or {}
    final_status = final.get("status") or {}
    if (
        abs(float(contact_status.get("work_x", math.nan)) - float(target["work_x"])) > 0.02
        or abs(float(contact_status.get("work_y", math.nan)) - float(target["work_y"])) > 0.02
        or abs(float(contact_status.get("work_z", math.nan)) - float(target["press_z"])) > 0.02
        or contact_status.get("stopped") is not True
        or contact_status.get("estop") is not False
    ):
        raise CaptureRejected("contact coordinate evidence failed")
    if (
        final_status.get("connected") is not True
        or final_status.get("stopped") is not True
        or final_status.get("moving") is not False
        or final_status.get("estop") is not False
        or abs(float(final_status.get("work_z", math.nan)) - float(target["safe_z"])) > 0.02
    ):
        raise CaptureRejected("final safe-state evidence failed")

    descent_sequence = _nearest_sequence(receipts, int(descent["host_monotonic_ns"]))
    contact_sequence = _nearest_sequence(receipts, int(contact["host_monotonic_ns"]))
    release_sequence = _nearest_sequence(receipts, int(release["host_monotonic_ns"]))
    trigger = int(commissioning.get("trigger_sequence", 96))
    hold_start, hold_stop = contact_sequence + 10, release_sequence - 10
    baseline_first = _adc_matrix(frames, trigger - 64, trigger, "first_code")
    baseline_second = _adc_matrix(frames, trigger - 64, trigger, "second_code")
    hold_first = _adc_matrix(frames, hold_start, hold_stop, "first_code")
    hold_second = _adc_matrix(frames, hold_start, hold_stop, "second_code")
    ramp_second = _adc_matrix(frames, descent_sequence, contact_sequence + 11, "second_code")
    features = _feature_families(
        baseline_first,
        baseline_second,
        hold_first,
        hold_second,
        ramp_second,
    )
    return ContactSample(
        path=path,
        point_index=point_index,
        repeat=repeat,
        x_mm=float(target["work_x"]),
        y_mm=float(target["work_y"]),
        contact_sequence=contact_sequence,
        release_sequence=release_sequence,
        hold_frames=int(hold_second.shape[0]),
        features=features,
        sha256=_sha256(path),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--expected-depth-mm", type=float, default=0.10)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--freeze-candidate-output", type=Path)
    parser.add_argument("--validation-template")
    parser.add_argument("--validation-repeat", type=int, default=3)
    args = parser.parse_args()
    if not math.isfinite(args.expected_depth_mm) or not 0.0 <= args.expected_depth_mm <= 0.8:
        raise SystemExit("--expected-depth-mm must be 0..0.8")
    inputs = [
        (
            args.capture_directory / args.template.format(point=point, repeat=repeat),
            repeat,
        )
        for repeat in (1, 2)
        for point in range(1, 18)
    ]
    missing = [str(path) for path, _ in inputs if not path.is_file()]
    if missing:
        raise SystemExit("missing captures:\n" + "\n".join(missing))
    samples = [
        load_contact_sample(path, repeat=repeat, expected_depth_mm=args.expected_depth_mm)
        for path, repeat in inputs
    ]
    split_results, best = evaluate_repeat_splits(samples)
    report = {
        "schema": "fast_triplet_position_repeat_validation_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "development_contact_count": len(samples),
        "unique_positions": 17,
        "opposite_order_repeats": True,
        "depth_mm": float(args.expected_depth_mm),
        "best_candidate": best,
        "candidate_split_results": split_results,
        "independent_accuracy_verified": False,
        "physical_15hz_verified": False,
        "contact_area_verified": False,
        "deployment_status": "development_only",
        "captures": [
            {
                "path": str(sample.path.resolve()),
                "sha256": sample.sha256,
                "point_index": sample.point_index,
                "repeat": sample.repeat,
                "xy_mm": [sample.x_mm, sample.y_mm],
                "contact_sequence": sample.contact_sequence,
                "release_sequence": sample.release_sequence,
                "hold_frames": sample.hold_frames,
            }
            for sample in samples
        ],
        "limitations": [
            "The same two repeats selected the feature/model and therefore are development data.",
            "A third untouched repeat is required before any position accuracy claim.",
            "Static CNC contact does not verify physical 15 Hz pressure response.",
            "No independent force or contact-patch label was recorded.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    if args.freeze_candidate_output:
        validation_template = args.validation_template or args.template
        reserved = [
            args.capture_directory
            / validation_template.format(point=point, repeat=args.validation_repeat)
            for point in range(1, 18)
        ]
        existing = [str(path) for path in reserved if path.exists()]
        if existing:
            raise SystemExit("reserved validation files already exist:\n" + "\n".join(existing))
        lock = {
            "schema": "fast_triplet_position_candidate_lock_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script_sha256": _sha256(Path(__file__)),
            "candidate": {"feature": best["feature"], "model": best["model"]},
            "development_capture_sha256": {sample.path.name: sample.sha256 for sample in samples},
            "expected_depth_mm": float(args.expected_depth_mm),
            "development_template": args.template,
            "validation_template": validation_template,
            "validation_repeat_reserved": int(args.validation_repeat),
            "validation_files_must_not_exist_when_locked": True,
        }
        args.freeze_candidate_output.parent.mkdir(parents=True, exist_ok=True)
        args.freeze_candidate_output.write_text(json.dumps(lock, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
