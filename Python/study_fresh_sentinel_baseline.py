"""Offline development study of recent, geometrically clear no-load baselines.

This uses CNC clearance solely to identify a no-contact calibration window.
Classifier inputs remain spectral deltas. Neither CNC XY nor route index is
fed to inference. Results are development evidence, not held-out validation.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from audit_fresh_sentinel_position import audit, fresh_vector
from evaluate_fresh_sentinel_position import compare
from qualify_position_coverage_session import read_jsonl


def recent_baseline(source):
    result = audit(source)
    events = list(read_jsonl(source / "events.jsonl"))
    contacts = {e["metadata"]["press_event_id"]: e for e in events if e["kind"] == "contact"}
    safe = defaultdict(list)
    for frame in read_jsonl(source / "frames.jsonl"):
        q = frame.get("quality") or {}
        group = q.get("press_event_id")
        if q.get("coverage_phase") != "pre_contact" or group not in contacts:
            continue
        status = q.get("machine_status") or {}
        contact = contacts[group]
        # Derive a per-event contact plane from recorded absolute position and
        # commanded indentation. Never substitute today's work-coordinate zero.
        contact_z = float(contact["metadata"]["machine_status"]["machine_z"]) + float(contact["indentation_mm"])
        received = int(frame["frame_received_monotonic_ns"])
        status_received = q.get("machine_status_received_monotonic_ns")
        if status_received is None or abs(received-int(status_received)) > 150_000_000:
            continue
        if not status.get("connected") or float(status["machine_z"]) < contact_z + 0.1:
            continue
        try:
            values = fresh_vector(frame)
        except ValueError:
            continue
        safe[group].append((received, values))
    for record in result["records"]:
        samples = safe[record["press_event_id"]]
        if not samples:
            raise ValueError("No timestamped clear baseline")
        latest = max(t for t, _ in samples)
        values = np.asarray([v for t, v in samples if t >= latest - 500_000_000])
        if len(values) < 10:
            raise ValueError("Insufficient recent clear baseline")
        old = np.asarray(record["baseline9"])
        new = np.median(values, axis=0)
        record["early_to_recent_baseline_rms_codes"] = float(np.sqrt(np.mean((new-old)**2)))
        record["delta9"] = (np.asarray(record["delta9"]) + old - new).tolist()
        record["hold_delta9"] = (np.asarray(record["hold_delta9"]) + old - new).tolist()
        record["baseline9"] = new.tolist()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", type=Path)
    parser.add_argument("test", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train, test = recent_baseline(args.train), recent_baseline(args.test)
    result = compare(train, test)
    result.update(purpose="development baseline study; not locked validation",
                  baseline_policy="last_500ms_at_least_0p1mm_clearance_with_recent_CNC_telemetry",
                  baseline_drift_rms_codes={capture["source"]: [r["early_to_recent_baseline_rms_codes"] for r in capture["records"]]
                                           for capture in (train, test)})
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: result[k] for k in ("contact_median", "single_fresh_frame", "purpose")}, indent=2))
