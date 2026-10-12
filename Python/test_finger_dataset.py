import json
import tempfile
import unittest
from pathlib import Path

from finger_dataset import ContactEvent, FingerSessionWriter, SpectrumFrame


class FingerDatasetTests(unittest.TestCase):
    def test_raw_voltage_ignores_digital_gain(self):
        low = SpectrumFrame(
            sequence=1,
            wavelengths_nm=(1530.0, 1530.02),
            raw_adc_codes=(0, 2048),
            digital_gain=1.0,
        )
        high = SpectrumFrame(
            sequence=2,
            wavelengths_nm=(1530.0, 1530.02),
            raw_adc_codes=(0, 2048),
            digital_gain=100.0,
        )
        self.assertEqual(low.adc_voltage_v(), high.adc_voltage_v())
        self.assertEqual(low.adc_voltage_v(), (0.0, 1.25))
        self.assertEqual(high.to_record()["digital_gain_audit"]["1"], 100.0)

    def test_writer_creates_manifest_and_lossless_jsonl(self):
        with tempfile.TemporaryDirectory() as temporary:
            session = Path(temporary) / "session-001"
            with FingerSessionWriter(
                session, queue_capacity=2, manifest_extra={"operator": "test"}
            ) as writer:
                writer.write_frame(
                    SpectrumFrame(
                        sequence=7,
                        monotonic_ns=100,
                        wavelengths_nm=(1528.40, 1528.42),
                        raw_adc_codes={1: (123, 4095)},
                        transimpedance_ohm={1: 2000},
                        digital_gain={1: 8},
                        table_crc32=0x12345678,
                        packet_crc_ok=True,
                        quality={"active_peaks": 9},
                    )
                )
                writer.write_event(
                    ContactEvent(
                        kind="contact",
                        event_id="contact-1",
                        monotonic_ns=101,
                        target_xyz_mm=(1.0, 2.0, 3.0),
                        indentation_mm=0.0,
                    )
                )

            manifest = json.loads((session / "manifest.json").read_text("utf-8"))
            frame = json.loads((session / "frames.jsonl").read_text("utf-8"))
            event = json.loads((session / "events.jsonl").read_text("utf-8"))
            self.assertEqual(manifest["sensor_channel"], 1)
            self.assertEqual(manifest["grating_count"], 9)
            self.assertEqual(manifest["points_per_peak"], 5)
            self.assertEqual(manifest["counts"]["frames"], 1)
            self.assertEqual(manifest["counts"]["events"], 1)
            self.assertEqual(
                manifest["digital_gain_policy"],
                "audit_only_not_applied_to_physical_values",
            )
            self.assertEqual(frame["raw_adc_codes"]["1"], [123, 4095])
            self.assertEqual(frame["table_crc32"], 0x12345678)
            self.assertAlmostEqual(frame["adc_voltage_v"]["1"][0], 123 * 2.5 / 4096)
            self.assertEqual(event["target_xyz_mm"], [1.0, 2.0, 3.0])

    def test_writer_rejects_out_of_order_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = FingerSessionWriter(Path(temporary) / "ordered")
            try:
                writer.write_event(ContactEvent(kind="start", monotonic_ns=200))
                with self.assertRaisesRegex(ValueError, "non-decreasing"):
                    writer.write_event(ContactEvent(kind="old", monotonic_ns=199))
            finally:
                writer.close()

    def test_frame_preserves_device_identity_and_explicit_unknown_settings(self):
        frame = SpectrumFrame(
            sequence=3,
            monotonic_ns=123_000,
            wavelengths_nm=(1530.0, 1530.02),
            raw_adc_codes={1: (101, 202)},
            transimpedance_ohm={1: None},
            digital_gain={1: None},
            analogue_gain_mask={1: 0x24},
            feedback_selector={1: None},
            table_crc32=0xA1B2C3D4,
            packet_crc_ok=True,
            device_sequence=999,
            device_boot_id=0x10203040,
            device_uptime_ms=8765,
            frame_received_monotonic_ns=122_900,
        )

        record = frame.to_record()
        self.assertEqual(record["raw_adc_codes"]["1"], (101, 202))
        self.assertIsNone(record["transimpedance_ohm"]["1"])
        self.assertIsNone(record["digital_gain_audit"]["1"])
        self.assertEqual(record["analogue_gain_mask"]["1"], 0x24)
        self.assertIsNone(record["feedback_selector"]["1"])
        self.assertEqual(record["device_sequence"], 999)
        self.assertEqual(record["device_boot_id"], 0x10203040)
        self.assertEqual(record["device_uptime_ms"], 8765)
        self.assertEqual(record["frame_received_monotonic_ns"], 122_900)

        # JSON output must retain explicit nulls, so downstream training can
        # distinguish unavailable hardware metadata from a fabricated default.
        decoded = json.loads(json.dumps(record))
        self.assertIsNone(decoded["transimpedance_ohm"]["1"])
        self.assertIsNone(decoded["feedback_selector"]["1"])

    def test_contact_annotation_enforces_physical_depth_cap(self):
        ContactEvent(kind="contact", indentation_mm=0.8)
        with self.assertRaisesRegex(ValueError, "0.00..0.80"):
            ContactEvent(kind="contact", indentation_mm=0.801)

    def test_multirate_freshness_and_real_sample_times_are_lossless(self):
        frame = SpectrumFrame(
            sequence=4,
            wavelengths_nm=(1530.0, 1530.02, 1530.04),
            raw_adc_codes={1: (100, 200, 300)},
            acquisition_profile="track",
            schedule_version=4,
            acquisition_profile_code=3,
            fresh_point_mask=(True, False, True),
            point_age_frames=(0, 2, 0),
            sample_offset_us=(125, None, 980),
            map_age_frames=7,
            bandwidth_discontinuity=True,
            frame_start_device_ms=123456,
        )

        record = frame.to_record()
        self.assertEqual(record["acquisition_profile"], "TRACK")
        self.assertEqual(record["schedule_version"], 4)
        self.assertEqual(record["acquisition_profile_code"], 3)
        self.assertEqual(record["fresh_point_mask"], (True, False, True))
        self.assertEqual(record["point_age_frames"], (0, 2, 0))
        self.assertEqual(record["sample_offset_us"], (125, None, 980))
        self.assertEqual(record["map_age_frames"], 7)
        self.assertTrue(record["bandwidth_discontinuity"])
        self.assertEqual(record["frame_start_device_ms"], 123456)

    def test_multirate_numeric_metadata_rejects_negative_values(self):
        for field_name in ("schedule_version", "acquisition_profile_code"):
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(ValueError, field_name):
                    SpectrumFrame(
                        sequence=8,
                        wavelengths_nm=(1530.0,),
                        raw_adc_codes={1: (100,)},
                        **{field_name: -1},
                    )

    def test_multirate_metadata_rejects_cached_points_with_fake_times(self):
        with self.assertRaisesRegex(ValueError, "cached points"):
            SpectrumFrame(
                sequence=5,
                wavelengths_nm=(1530.0, 1530.02),
                raw_adc_codes={1: (100, 200)},
                fresh_point_mask=(True, False),
                sample_offset_us=(50, 100),
            )

    def test_legacy_full_map_can_mark_point_timing_unavailable(self):
        frame = SpectrumFrame(
            sequence=6,
            wavelengths_nm=(1530.0, 1530.02),
            raw_adc_codes={1: (100, 200)},
            acquisition_profile="MAP",
            fresh_point_mask=(True, True),
            point_age_frames=(0, 0),
            sample_offset_us=(None, None),
        )
        self.assertEqual(frame.to_record()["sample_offset_us"], (None, None))


if __name__ == "__main__":
    unittest.main()
