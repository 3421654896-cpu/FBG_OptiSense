"""Create an area-labelled training session without modifying raw capture data.

The source must already be a position-qualified coverage session.  A separate
``fbg-contact-area-truth/v1`` document is matched to every physical press by
``press_event_id``.  Only contact/hold event annotations are rewritten; frame
and native transport files are hard-linked and their hashes are pinned.

This tool is deliberately offline.  It never opens the laser, ADC or Mach3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from physical_pressure_validation import (
    ContactAreaTruthRecord,
    ContactAreaTruthSet,
    load_contact_area_truth,
)


CONTACT_PHASES = {"contact", "hold"}
REQUIRED_FILES = ("manifest.json", "frames.jsonl", "events.jsonl")
OPTIONAL_IMMUTABLE_FILES = ("native_frames.jsonl",)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")


def _source_press_groups(events: list[dict[str, Any]]) -> set[str]:
    groups: set[str] = set()
    contact_phases: dict[str, set[str]] = {}
    for event in events:
        metadata = event.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("event metadata must be an object")
        value = metadata.get("press_event_id")
        if value is None:
            continue
        group = str(value).strip()
        if not group:
            raise ValueError("press_event_id must not be empty")
        groups.add(group)
        kind = str(event.get("kind", "")).strip().lower()
        if kind in CONTACT_PHASES:
            contact_phases.setdefault(group, set()).add(kind)
    incomplete = sorted(
        group for group in groups if contact_phases.get(group) != CONTACT_PHASES
    )
    if incomplete:
        raise ValueError(
            "every press needs both contact and hold events; incomplete="
            f"{incomplete[:3]}"
        )
    return groups


def _label_event(
    event: dict[str, Any], truth_by_event: dict[str, ContactAreaTruthRecord]
) -> dict[str, Any]:
    result = dict(event)
    metadata = dict(result.get("metadata") or {})
    group = str(metadata.get("press_event_id") or "").strip()
    kind = str(result.get("kind", "")).strip().lower()
    if group in truth_by_event and kind in CONTACT_PHASES:
        truth = truth_by_event[group]
        metadata["measured_contact_area_mm2"] = truth.area_mm2
        metadata["area_truth_provenance"] = {
            "source": truth.source,
            "measurement_id": truth.measurement_id,
            "calibration_id": truth.calibration_id,
            "linked_event_id": group,
            "independent": True,
        }
    result["metadata"] = metadata
    return result


def attach_contact_area_truth(
    source: str | Path,
    truth_path: str | Path,
    destination: str | Path,
) -> dict[str, Any]:
    """Create one auditable, area-labelled derivative session."""

    source_path = Path(source).resolve()
    truth_file = Path(truth_path).resolve()
    destination_path = Path(destination).resolve()
    if destination_path.exists():
        raise FileExistsError("refusing to overwrite an existing destination")
    paths = {name: source_path / name for name in REQUIRED_FILES}
    if not all(path.is_file() for path in paths.values()):
        raise FileNotFoundError("source manifest, frames and events are required")
    for name in OPTIONAL_IMMUTABLE_FILES:
        candidate = source_path / name
        if candidate.is_file():
            paths[name] = candidate

    manifest = json.loads(paths["manifest.json"].read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("source manifest must be an object")
    metadata = manifest.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("source manifest metadata must be an object")
    if metadata.get("training_eligible") is not True:
        raise ValueError("source session is not qualified for training")
    if metadata.get("position_training_eligible") is not True:
        raise ValueError("source session lacks qualified position labels")
    if metadata.get("area_training_eligible") is not False:
        raise ValueError("source must be a position-only session awaiting area truth")
    if not manifest.get("closed_utc"):
        raise ValueError("source session is not closed")

    source_hashes = {name: _sha256(path) for name, path in paths.items()}
    truth: ContactAreaTruthSet = load_contact_area_truth(truth_file)
    sources = {record.source for record in truth.records}
    if len(sources) != 1:
        raise ValueError("one training session must use one contact-area truth source")
    events = _read_jsonl(paths["events.jsonl"])
    press_groups = _source_press_groups(events)
    truth_by_event = {record.event_id: record for record in truth.records}
    if press_groups != set(truth_by_event):
        missing = sorted(press_groups - set(truth_by_event))
        extra = sorted(set(truth_by_event) - press_groups)
        raise ValueError(
            "area truth must match press_event_id values exactly; "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )

    labelled_events = [_label_event(event, truth_by_event) for event in events]
    # Check source immutability before creating any output.
    if source_hashes != {name: _sha256(path) for name, path in paths.items()}:
        raise ValueError("source session changed during area-truth attachment")
    truth_hash = _sha256(truth_file)

    destination_path.mkdir(parents=True)
    try:
        for name in ("frames.jsonl", "native_frames.jsonl"):
            if name in paths:
                os.link(paths[name], destination_path / name)
        _write_jsonl(destination_path / "events.jsonl", labelled_events)
        derived_manifest = dict(manifest)
        derived_manifest["session_id"] = destination_path.name
        derived_manifest["metadata"] = {
            **metadata,
            "training_eligible": True,
            "training_scope": "position_contact_and_physical_area",
            "position_training_eligible": True,
            "area_training_eligible": True,
            "physical_15hz_verified": False,
            "area_truth_source": next(iter(sources)),
            "area_truth_record_count": len(truth.records),
            "area_truth_distinct_levels": truth.distinct_area_count,
            "area_truth_calibration_ids": list(truth.calibration_ids),
            "area_truth_file": str(truth_file),
            "area_truth_sha256": truth_hash,
            "source_position_session": str(source_path),
            "source_position_session_sha256": source_hashes,
        }
        (destination_path / "manifest.json").write_text(
            json.dumps(derived_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report = {
            "schema": "fbg-area-labelled-training-session/v1",
            "source_session": str(source_path),
            "destination_session": str(destination_path),
            "press_event_count": len(press_groups),
            "contact_and_hold_events_labelled": 2 * len(press_groups),
            "area_truth_source": next(iter(sources)),
            "area_truth_record_count": len(truth.records),
            "distinct_area_count": truth.distinct_area_count,
            "area_calibration_ids": list(truth.calibration_ids),
            "area_truth_sha256": truth_hash,
            "source_sha256": source_hashes,
            "physical_15hz_verified": False,
            "ready_for_area_model_training": True,
        }
        (destination_path / "area_attachment_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:
        # The destination did not exist on entry and contains only files made
        # by this function.  Remove them individually so a partial derivative
        # can never be mistaken for a qualified session.
        for child in destination_path.iterdir():
            child.unlink()
        destination_path.rmdir()
        raise
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="position-qualified session")
    parser.add_argument("area_truth", type=Path, help="contact-area truth JSON")
    parser.add_argument("destination", type=Path, help="new labelled session")
    args = parser.parse_args()
    report = attach_contact_area_truth(args.source, args.area_truth, args.destination)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
