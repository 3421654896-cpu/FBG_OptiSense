"""Short hardware smoke test for the deployed F45 realtime localizer."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

import serial

from fast_fullmap_live import execute_realtime_device
from fast_fullmap_realtime import FastFullMapRealtimeLocalizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--cycles", type=int, default=160)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.capture.exists() or args.output.exists():
        parser.error("capture and output must be new")

    localizer = FastFullMapRealtimeLocalizer()
    inference_ms = []
    states = {}
    contacts = []
    last_result = None

    def consume(frame):
        nonlocal last_result
        started = time.perf_counter_ns()
        result = localizer.update(frame)
        inference_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
        states[result.state] = states.get(result.state, 0) + 1
        if result.baseline_ready and result.contact_detected:
            contacts.append(
                {
                    "sequence": result.sequence,
                    "point": result.predicted_point,
                    "response_rms_codes": result.response_rms_codes,
                    "confidence": result.confidence,
                }
            )
        last_result = result

    with serial.Serial(args.port, 2_000_000, timeout=0.05, write_timeout=1.0) as device:
        hardware = execute_realtime_device(
            device,
            args.capture,
            cycles=args.cycles,
            spacing_us=50,
            on_frame=consume,
        )
    ordered = sorted(inference_ms)
    report = {
        "schema": "f45_runtime_live_smoke_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "port": args.port,
        "cycles": args.cycles,
        "model": str(localizer.model_path),
        "model_schema": localizer.model_schema,
        "state_counts": states,
        "baseline_ready": bool(last_result and last_result.baseline_ready),
        "idle_contact_detections_after_baseline": contacts,
        "idle_contact_detection_count": len(contacts),
        "inference_ms": {
            "mean": statistics.fmean(inference_ms),
            "p95": ordered[round(0.95 * (len(ordered) - 1))],
            "maximum": max(inference_ms),
        },
        "hardware_analysis": hardware.get("analysis"),
        "safety_cleanup": hardware.get("safety_cleanup"),
        "complete": bool(hardware.get("complete")),
        "physical_15hz_verified": False,
    }
    report["passed"] = not (
        not report["complete"]
        or not report["baseline_ready"]
        or not report["safety_cleanup"]["exact_disarm_ack"]
        or not report["safety_cleanup"]["exact_soa_shutter_ack"]
        or report["idle_contact_detection_count"] != 0
        or not (report["hardware_analysis"] or {}).get("meets_sampling_rate_for_15hz")
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
