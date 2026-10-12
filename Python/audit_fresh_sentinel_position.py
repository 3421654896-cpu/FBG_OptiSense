"""Extract fresh schedule-v4 CH1 features for offline position development.

This does not qualify a capture for deployment or establish contact-area truth.
Values retain their recorded ADC processing kind, including firmware RC estimates.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from qualify_position_coverage_session import read_jsonl, sha256
from frame_telemetry import sentinel_offsets_for_version


SENTINELS = (3, 8, 13, 18, 21, 28, 31, 36, 41)
PROTOCOL = "fresh_sentinel9_v1"


def fresh_vector(frame: dict) -> np.ndarray:
    if frame.get("schedule_version") not in (4, 5) or frame.get("bandwidth_discontinuity") is not False:
        raise ValueError("schedule_or_discontinuity")
    indices = tuple(5*g+offset for g, offset in enumerate(sentinel_offsets_for_version(frame["schedule_version"])))
    mask = frame.get("fresh_point_mask", [])
    ages = frame.get("point_age_frames", [])
    offsets = frame.get("sample_offset_us", [])
    codes = frame.get("raw_adc_codes", {}).get("1", [])
    if any(len(values) != 45 for values in (mask, ages, offsets, codes)):
        raise ValueError("incomplete_frame")
    for index in indices:
        if mask[index] is not True or ages[index] != 0:
            raise ValueError("stale_sentinel")
        if not isinstance(offsets[index], (int, float)) or not np.isfinite(offsets[index]) or offsets[index] < 0:
            raise ValueError("missing_sample_time")
        if type(codes[index]) is not int or not 0 <= codes[index] <= 4095:
            raise ValueError("invalid_adc")
    return np.asarray([codes[index] for index in indices], dtype=float)


def audit(source: Path) -> dict:
    source = source.resolve()
    paths = {name: source / name for name in ("manifest.json", "frames.jsonl", "events.jsonl")}
    hashes = {name: sha256(path) for name, path in paths.items()}
    manifest = json.loads(paths["manifest.json"].read_text("utf-8"))
    if not manifest.get("closed_utc"):
        raise ValueError("Capture must be closed")
    events = list(read_jsonl(paths["events.jsonl"]))
    ends = [e for e in events if e.get("kind") == "session_end"]
    if len(ends) != 1 or not (ends[0].get("metadata") or {}).get("success"):
        raise ValueError("Capture did not finish successfully")
    phase_events = {e.get("event_id"): e for e in events}
    contacts = {e["metadata"]["press_event_id"]: e for e in events if e.get("kind") == "contact"}
    pres = {e["metadata"]["press_event_id"]: e for e in events if e.get("kind") == "pre_contact"}
    grouped = defaultdict(lambda: defaultdict(list))
    profiles = defaultdict(lambda: defaultdict(list))
    rejected = Counter()
    identity = None
    kinds = Counter()
    for frame in read_jsonl(paths["frames.jsonl"]):
        quality = frame.get("quality") or {}
        group, phase = quality.get("press_event_id"), quality.get("coverage_phase")
        if group not in contacts or phase not in ("pre_contact", "hold"):
            continue
        phase_event = phase_events.get(quality.get("phase_event_id")) or {}
        if phase_event.get("kind") != phase or (phase_event.get("metadata") or {}).get("press_event_id") != group:
            raise ValueError("Frame phase provenance differs from event log")
        received = int(frame.get("frame_received_monotonic_ns") or frame["monotonic_ns"])
        if phase == "pre_contact" and not int(pres[group]["monotonic_ns"]) <= received <= int(pres[group]["monotonic_ns"]) + 2_000_000_000:
            continue
        current_identity = (
            frame.get("table_crc32"), tuple(frame.get("wavelengths_nm", [])),
            frame.get("transimpedance_ohm", {}).get("1"), frame.get("adc_value_kind"),
            frame.get("schedule_version"),
        )
        if current_identity[0] is None or len(current_identity[1]) != 45 or current_identity[2] is None:
            raise ValueError("Missing table or gain identity")
        if identity is not None and current_identity != identity:
            raise ValueError("Table, gain or ADC processing changed")
        identity = current_identity
        try:
            vector = fresh_vector(frame)
        except ValueError as error:
            rejected[str(error)] += 1
            continue
        grouped[group][phase].append(vector)
        profiles[group][(phase, str(frame.get("acquisition_profile")))].append(vector)
        kinds[frame.get("adc_value_kind", "unknown")] += 1
    records = []
    for group, contact in contacts.items():
        phases = grouped[group]
        counts = {phase: len(phases[phase]) for phase in ("pre_contact", "hold")}
        if min(counts.values()) < 10:
            raise ValueError(f"{group}: insufficient fresh baseline/hold frames: {counts}")
        baseline = np.asarray(phases["pre_contact"])
        held = np.asarray(phases["hold"])
        base = np.median(baseline, axis=0)
        noise = 1.4826 * np.median(np.abs(baseline - base), axis=0)
        delta = np.median(held, axis=0) - base
        # Nearest other endpoint is development evidence, never an accuracy test.
        records.append({
            "press_event_id": group, "point_index": contact["metadata"]["point_index"],
            "model_xy_mm": contact["metadata"]["model_xy_mm"], "counts": counts,
            "baseline9": base.tolist(), "baseline_sigma9": noise.tolist(),
            "delta9": delta.tolist(), "hold_delta9": (held - base).tolist(),
            "profile_medians": [
                {"phase": phase, "profile": profile, "frames": len(values),
                 "median9": np.median(values, axis=0).tolist()}
                for (phase, profile), values in sorted(profiles[group].items())
            ],
            "response_rms_codes": float(np.sqrt(np.mean(delta ** 2))),
        })
    if not records:
        raise ValueError("No physical contacts")
    records.sort(key=lambda item: item["point_index"])
    if hashes != {name: sha256(path) for name, path in paths.items()}:
        raise ValueError("Capture changed during audit")
    return {
        "protocol": PROTOCOL if identity[4] == 4 else "fresh_sentinel9_v5_v1", "source": str(source), "source_sha256": hashes,
        "depth_mm": manifest["metadata"]["depth_mm"],
        "sentinel_indices": [5*g+offset for g, offset in enumerate(sentinel_offsets_for_version(identity[4]))], "table_crc32": identity[0],
        "wavelengths_nm": identity[1], "transimpedance_ohm": identity[2],
        "adc_value_kind_counts": kinds, "rejected_frames": rejected,
        "position_count": len(records), "records": records,
        "deployment_eligible": False, "accuracy_evaluated": False,
        "physical_15hz_verified": False, "contact_area_evaluated": False,
        "limitations": ["Single-session development features; independent repeat required",
                        "Host phase labels do not verify dynamic contact timing",
                        "Recorded values preserve firmware ADC processing; not certified direct ADC"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k not in ("records", "wavelengths_nm")}, ensure_ascii=False, indent=2))
