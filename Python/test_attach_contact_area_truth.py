"""Offline tests for traceable contact-area attachment."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from attach_contact_area_truth import attach_contact_area_truth


class AttachContactAreaTruthTests(unittest.TestCase):
    def _source(self, root: Path, *, groups: int = 12) -> Path:
        source = root / "position-qualified"
        source.mkdir()
        manifest = {
            "session_id": "position-qualified",
            "closed_utc": "2026-09-06T00:00:00Z",
            "metadata": {
                "training_eligible": True,
                "training_scope": "position_and_contact_only",
                "position_training_eligible": True,
                "area_training_eligible": False,
            },
        }
        (source / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        (source / "frames.jsonl").write_text("{}\n", encoding="utf-8")
        events = []
        for index in range(groups):
            group = f"press-{index:03d}"
            for phase in (
                "pre_contact",
                "contact",
                "hold",
                "release",
                "release_complete",
            ):
                events.append(
                    {
                        "event_id": f"{group}-{phase}",
                        "kind": phase,
                        "metadata": {"press_event_id": group},
                    }
                )
        (source / "events.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in events),
            encoding="utf-8",
        )
        return source

    def _truth(self, root: Path, *, groups: int = 12, omit: int | None = None) -> Path:
        truth = root / "areas.json"
        records = []
        for index in range(groups):
            if index == omit:
                continue
            records.append(
                {
                    "event_id": f"press-{index:03d}",
                    "area_mm2": (8.0, 16.0, 24.0)[index % 3],
                    "source": "pressure_film",
                    "measurement_id": f"film-{index:03d}",
                    "calibration_id": "film-cal-2026",
                    "independent": True,
                }
            )
        truth.write_text(
            json.dumps({"schema": "fbg-contact-area-truth/v1", "records": records}),
            encoding="utf-8",
        )
        return truth

    def test_labels_only_contact_and_hold_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            truth = self._truth(root)
            destination = root / "area-qualified"
            report = attach_contact_area_truth(source, truth, destination)

            self.assertTrue(report["ready_for_area_model_training"])
            self.assertEqual(report["contact_and_hold_events_labelled"], 24)
            manifest = json.loads(
                (destination / "manifest.json").read_text(encoding="utf-8")
            )
            metadata = manifest["metadata"]
            self.assertTrue(metadata["area_training_eligible"])
            self.assertEqual(
                metadata["training_scope"], "position_contact_and_physical_area"
            )
            events = [
                json.loads(line)
                for line in (destination / "events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            labelled = [
                event
                for event in events
                if "area_truth_provenance" in event["metadata"]
            ]
            self.assertEqual({event["kind"] for event in labelled}, {"contact", "hold"})
            self.assertEqual(len(labelled), 24)
            self.assertTrue(
                all(
                    event["metadata"]["area_truth_provenance"]["linked_event_id"]
                    == event["metadata"]["press_event_id"]
                    for event in labelled
                )
            )
            self.assertEqual((source / "frames.jsonl").read_text(), "{}\n")

    def test_event_mismatch_fails_before_creating_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root, groups=13)
            truth = self._truth(root, groups=13, omit=12)
            destination = root / "area-qualified"
            with self.assertRaisesRegex(ValueError, "match press_event_id"):
                attach_contact_area_truth(source, truth, destination)
            self.assertFalse(destination.exists())

    def test_unqualified_source_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["metadata"]["training_eligible"] = False
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not qualified"):
                attach_contact_area_truth(
                    source, self._truth(root), root / "area-qualified"
                )

    def test_mixed_truth_sources_are_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._source(root)
            truth = self._truth(root)
            payload = json.loads(truth.read_text(encoding="utf-8"))
            payload["records"][0]["source"] = "measured"
            truth.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "one contact-area truth source"):
                attach_contact_area_truth(source, truth, root / "area-qualified")


if __name__ == "__main__":
    unittest.main()
