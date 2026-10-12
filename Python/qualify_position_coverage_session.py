"""Qualify a closed physical coverage session for position-only model audit.

This never upgrades the source capture in place.  It creates a derived session
whose large immutable JSONL files are hard-linked, pins their SHA-256 hashes,
and marks only geometric position training as eligible.  Contact-area and
physical 15 Hz claims remain explicitly ineligible.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path

from frame_telemetry import decode_board_frame_timing, extract_raw_scan_frames
from verify_runtime_firmware import board_extension


PHASES = ("pre_contact", "contact", "hold", "release", "release_complete")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            yield value


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_machine_status(status: dict) -> None:
    if status.get("estop") or status.get("estop_led"):
        raise ValueError("E-stop was active in a recorded phase")
    if any(
        status.get(f"{axis}_{direction}_limit")
        for axis in "xyz"
        for direction in ("pos", "neg")
    ):
        raise ValueError("Hardware limit was active in a recorded phase")


def qualify(source: Path, destination: Path) -> dict:
    source = source.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError("Refusing to overwrite a reviewed session")
    paths = {
        name: source / name
        for name in ("manifest.json", "frames.jsonl", "events.jsonl", "native_frames.jsonl")
    }
    if not all(path.is_file() for path in paths.values()):
        raise FileNotFoundError("Coverage manifest, UI frames, events and native frames are required")
    hashes_before = {name: sha256(path) for name, path in paths.items()}
    manifest = json.loads(paths["manifest.json"].read_text("utf-8"))
    metadata = manifest.get("metadata") or {}
    if not manifest.get("closed_utc"):
        raise ValueError("Source coverage session is not closed")
    if metadata.get("source_at_start") != "local":
        raise ValueError("Only the reviewed local USB path is accepted")
    if int(metadata.get("unique_positions", 0)) != 17 or int(metadata.get("repeats", 0)) != 1:
        raise ValueError("Exactly one 17-position physical repeat is required")
    depth = float(metadata.get("depth_mm", math.nan))
    if not math.isfinite(depth) or not 0.0 < depth <= 0.8:
        raise ValueError("A finite nonzero indentation within the 0.8 mm limit is required")

    events = list(read_jsonl(paths["events.jsonl"]))
    session_end = [event for event in events if event.get("kind") == "session_end"]
    if len(session_end) != 1 or not session_end[0].get("metadata", {}).get("success"):
        raise ValueError("Coverage session did not end successfully")
    end_meta = session_end[0]["metadata"]
    if int(end_meta.get("completed_contacts", -1)) != 17 or int(end_meta.get("total_contacts", -1)) != 17:
        raise ValueError("Coverage did not complete all 17 contacts")
    native_stats = end_meta.get("native_recording") or {}
    if native_stats.get("error") is not None or int(native_stats.get("dropped", -1)) != 0:
        raise ValueError("Native receiver reported loss")

    groups: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        event_meta = event.get("metadata") or {}
        event_group = event_meta.get("press_event_id")
        if event_group is None:
            continue
        groups[str(event_group)].append(event)
        check_machine_status(event_meta.get("machine_status") or {})
    if len(groups) != 17:
        raise ValueError(f"Expected 17 press groups, found {len(groups)}")

    endpoint_errors = []
    point_indices = set()
    phase_event_ids = {}
    for group, group_events in groups.items():
        kinds = tuple(event.get("kind") for event in group_events)
        if kinds != PHASES:
            raise ValueError(f"{group}: phase order {kinds!r} does not match {PHASES!r}")
        contact = group_events[1]
        target = contact.get("target_xyz_mm")
        actual = contact.get("actual_xyz_mm")
        if not isinstance(target, list) or not isinstance(actual, list) or len(target) != 3 or len(actual) != 3:
            raise ValueError(f"{group}: contact endpoint is incomplete")
        error = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(actual, target)))
        if error > 0.01:
            raise ValueError(f"{group}: endpoint error {error:.4f} mm exceeds 0.01 mm")
        if abs(float(contact.get("indentation_mm")) - depth) > 1e-9:
            raise ValueError(f"{group}: indentation label differs from manifest")
        endpoint_errors.append(error)
        point_indices.add(int(contact["metadata"]["point_index"]))
        for event in group_events:
            phase_event_ids[event["event_id"]] = (group, event["kind"])
    if point_indices != set(range(1, 18)):
        raise ValueError("Coverage point indices are not exactly 1..17")

    phase_counts = Counter()
    complete_map_counts = Counter()
    frame_count = 0
    wavelengths = None
    tia = None
    gain = None
    previous_monotonic = None
    for frame in read_jsonl(paths["frames.jsonl"]):
        frame_count += 1
        if frame.get("record_type") != "spectrum_frame":
            raise ValueError("Unexpected frame record type")
        current_time = int(frame["monotonic_ns"])
        if previous_monotonic is not None and current_time < previous_monotonic:
            raise ValueError("Frame time moved backwards")
        previous_monotonic = current_time
        codes = frame.get("raw_adc_codes", {}).get("1")
        if not isinstance(codes, list) or len(codes) != 45:
            raise ValueError("Every frame must retain 45 CH1 ADC codes")
        if any(not isinstance(value, int) or not 0 <= value <= 4095 for value in codes):
            raise ValueError("CH1 ADC code is outside the 12-bit range")
        frame_waves = tuple(float(value) for value in frame.get("wavelengths_nm", ()))
        frame_tia = float(frame.get("transimpedance_ohm", {}).get("1", math.nan))
        frame_gain = float(frame.get("digital_gain_audit", {}).get("1", math.nan))
        wavelengths = frame_waves if wavelengths is None else wavelengths
        tia = frame_tia if tia is None else tia
        gain = frame_gain if gain is None else gain
        if len(frame_waves) != 45 or frame_waves != wavelengths:
            raise ValueError("Wavelength table changed within the session")
        if frame_tia != tia or frame_gain != gain:
            raise ValueError("CH1 analogue or digital gain changed within the session")
        quality = frame.get("quality") or {}
        phase_id = quality.get("phase_event_id")
        group = str(quality.get("press_event_id") or "")
        if phase_id not in phase_event_ids or phase_event_ids[phase_id][0] != group:
            raise ValueError("Frame phase/event provenance does not match the event log")
        phase = str(quality.get("coverage_phase"))
        if phase_event_ids[phase_id][1] != phase:
            raise ValueError("Frame coverage phase differs from its event")
        phase_counts[(group, phase)] += 1
        if (
            frame.get("acquisition_profile") in {"MAP", "MAP45"}
            and frame.get("fresh_point_mask") == [True] * 45
            and frame.get("point_age_frames") == [0] * 45
            and not frame.get("bandwidth_discontinuity")
        ):
            complete_map_counts[(group, phase)] += 1
    if frame_count < 1000:
        raise ValueError("Too few saved UI frames")
    for group in groups:
        if phase_counts[(group, "hold")] < 10:
            raise ValueError(f"{group}: fewer than 10 settled hold frames")
        if complete_map_counts[(group, "hold")] < 3:
            raise ValueError(f"{group}: fewer than 3 complete settled MAP frames")

    native_count = 0
    previous_sequence = None
    for record in read_jsonl(paths["native_frames.jsonl"]):
        native_count += 1
        if record.get("schema") != "fbg-native-receive/v1" or int(record.get("record_number", -1)) != native_count:
            raise ValueError("Native receive records are not complete and ordered")
        payload = bytes.fromhex(record["payload_hex"])
        buffer = bytearray(payload)
        if extract_raw_scan_frames(buffer) != [payload] or buffer:
            raise ValueError("Malformed native transport frame")
        timing = decode_board_frame_timing(board_extension(payload))
        if timing is None:
            raise ValueError("Native frame lacks board timing")
        if previous_sequence is not None and timing.sequence != (previous_sequence + 1) & 0xFFFFFFFF:
            raise ValueError("Native device sequence is discontinuous")
        previous_sequence = timing.sequence
    if native_count != int(native_stats.get("written", -1)):
        raise ValueError("Native file count differs from the close acknowledgement")
    if hashes_before != {name: sha256(path) for name, path in paths.items()}:
        raise ValueError("Source capture changed while it was being reviewed")

    destination.mkdir(parents=True)
    for name in ("frames.jsonl", "events.jsonl", "native_frames.jsonl"):
        os.link(paths[name], destination / name)
    reviewed_manifest = dict(manifest)
    reviewed_manifest["session_id"] = destination.name
    reviewed_manifest["metadata"] = {
        **metadata,
        "training_eligible": True,
        "training_scope": "position_and_contact_only",
        "position_training_eligible": True,
        "area_training_eligible": False,
        "physical_15hz_verified": False,
        "adc_event_synchronization_verified": False,
        "review_basis": "long_settled_host_phase_with_exact_phase_event_provenance",
        "source_session": str(source),
        "source_sha256": hashes_before,
    }
    (destination / "manifest.json").write_text(
        json.dumps(reviewed_manifest, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    report = {
        "source": str(source),
        "destination": str(destination),
        "training_scope": "position_and_contact_only",
        "contact_area_training_allowed": False,
        "physical_15hz_verified": False,
        "depth_mm": depth,
        "press_groups": len(groups),
        "frames": frame_count,
        "native_frames": native_count,
        "native_sequence_continuous": True,
        "endpoint_error_mm_max": max(endpoint_errors),
        "endpoint_error_mm_median": sorted(endpoint_errors)[len(endpoint_errors) // 2],
        "hold_complete_map_frames_min": min(
            complete_map_counts[(group, "hold")] for group in groups
        ),
        "fixed_transimpedance_ohm": tia,
        "fixed_digital_gain_audit": gain,
        "source_sha256": hashes_before,
    }
    (destination / "qualification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(qualify(arguments.source, arguments.destination), ensure_ascii=False, indent=2))
