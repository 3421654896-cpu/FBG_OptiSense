"""Measure provisional-localizer false contacts on a closed static capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from capture_ch1_rc_diagnostic import load_installed_rows
from capture_ch1_teacher_spectrum import save_checkpoint, sha256_file
from fast_flank_realtime import FastFlankRealtimeLocalizer


def _normalize_frame(raw: dict) -> dict:
    frame = dict(raw)
    if "minimum_warmup_complete" not in frame:
        frame["minimum_warmup_complete"] = bool(frame.get("warmup_ready"))
    return frame


def audit(source: Path, output: Path, *, baseline_frames: int = 64) -> dict:
    source = Path(source)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    before = sha256_file(source)
    capture = json.loads(source.read_text(encoding="utf-8"))
    if not capture.get("complete"):
        raise ValueError("source capture is not complete")
    cleanup = capture.get("safety_cleanup") or {}
    if not cleanup.get("exact_disarm_ack") or not cleanup.get("exact_soa_shutter_ack"):
        raise ValueError("source capture lacks exact optical shutdown proof")
    if capture.get("cnc_motion_commands") != 0:
        raise ValueError("static false-contact audit requires no CNC motion")

    rows, _ = load_installed_rows()
    pipeline = FastFlankRealtimeLocalizer(rows, baseline_frames=baseline_frames)
    monitored = []
    processing_ms = []
    frozen_at_sequence = None
    for raw in capture.get("frames", []):
        frame = _normalize_frame(raw)
        started_ns = time.perf_counter_ns()
        if frozen_at_sequence is None:
            result = pipeline.update(frame, declared_no_contact=True)
            if result.baseline_ready:
                pipeline.freeze_for_press()
                frozen_at_sequence = int(frame["sequence"])
        else:
            # The source is physically static/no-contact.  Feed the latter
            # part through contact-monitoring state to measure false alarms.
            result = pipeline.update(frame, declared_no_contact=False)
            monitored.append(result)
        processing_ms.append((time.perf_counter_ns() - started_ns) / 1e6)

    if frozen_at_sequence is None or not monitored:
        raise ValueError("capture is too short to build then freeze a baseline")
    detections = np.asarray([item.contact_detected for item in monitored], dtype=bool)
    longest = current = 0
    for detected in detections:
        current = current + 1 if detected else 0
        longest = max(longest, current)
    result = {
        "schema": "fast_flank_static_pressure_change_false_alarm_v1",
        "source": str(source.resolve()),
        "source_sha256": before,
        "baseline_frames": int(baseline_frames),
        "baseline_frozen_at_sequence": frozen_at_sequence,
        "monitored_frames": int(len(monitored)),
        "false_pressure_change_alarm_frames": int(np.count_nonzero(detections)),
        "false_pressure_change_alarm_rate": float(np.mean(detections)),
        "longest_false_alarm_run_frames": int(longest),
        "sampling_gate_pass_frames": int(
            sum(item.sampling_gate_for_15hz for item in monitored)
        ),
        "sampling_gate_pass_rate": float(
            np.mean([item.sampling_gate_for_15hz for item in monitored])
        ),
        "host_processing_timing_ms": {
            "median": float(np.median(processing_ms)),
            "p95": float(np.percentile(processing_ms, 95)),
            "p99": float(np.percentile(processing_ms, 99)),
            "maximum": float(np.max(processing_ms)),
            "p99_under_5ms": bool(np.percentile(processing_ms, 99) < 5.0),
            "under_15hz_output_budget": bool(np.max(processing_ms) < 1000.0 / 15.0),
            "scope": "offline replay compute time; excludes USB, Qt drawing and OS scheduling",
        },
        "false_alarm_acceptance_limit": 0.01,
        "false_alarm_acceptance_passed": bool(np.mean(detections) <= 0.01),
        "physical_15hz_verified": False,
        "position_accuracy_verified": False,
        "contact_area_verified": False,
        "training_eligible": False,
        "note": (
            "This is a static pressure-change false-alarm audit after a frozen "
            "no-contact baseline. It does not measure pressure sensitivity, "
            "static contact state, or localization accuracy."
        ),
    }
    if sha256_file(source) != before:
        raise RuntimeError("source capture changed during audit")
    save_checkpoint(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-frames", type=int, default=64)
    args = parser.parse_args()
    print(
        json.dumps(
            audit(args.source, args.output, baseline_frames=args.baseline_frames),
            ensure_ascii=False,
            indent=2,
        )
    )
