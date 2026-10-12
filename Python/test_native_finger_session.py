import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from finger_dataset import ContactEvent, FingerSessionWriter, SpectrumFrame
from native_finger_session import NativeFingerDecoder, export_diagnostic_session
from test_stress_multirate import _extension, _native_frame
from frame_telemetry import MULTIRATE_PROFILE_MAP, MULTIRATE_PROFILE_SURVEY, MULTIRATE_SENTINEL_LOCAL_OFFSETS
from train_finger_localizer import load_session_training_samples

WAVES = [1528.0 + i * .02 for i in range(45)]
SENTINELS = tuple(i*5+v for i, v in enumerate(MULTIRATE_SENTINEL_LOCAL_OFFSETS))


def native(sequence=1, *, survey=False, selector=2, active=2):
    ext = bytearray(_extension(MULTIRATE_PROFILE_SURVEY if survey else MULTIRATE_PROFILE_MAP,
                              SENTINELS if survey else tuple(range(45))))
    ext[16] = active
    ext[23] = selector
    ext[28:32] = sequence.to_bytes(4, "big")
    frame = bytearray(_native_frame(ext))
    for index in range(45):
        for channel in range(4):
            pos = 4 + index*8 + channel*2
            frame[pos:pos+2] = (channel*1000+index).to_bytes(2, "big")
    # CH1 mask after AB, four empty peak-count fields, and temperature.
    frame[4+45*8+5+6:4+45*8+5+8] = (0x0123).to_bytes(2, "big")
    return bytes(frame)


class NativeFingerDecoderTests(unittest.TestCase):
    def test_exact_ch1_codes_and_hardware_gain_not_digital_scaling(self):
        frame = NativeFingerDecoder(WAVES, digital_gain_audit=12).decode(native(), 100)
        self.assertEqual(frame.raw_adc_codes[1], tuple(range(1000, 1045)))
        self.assertEqual(frame.transimpedance_ohm[1], 5000)
        self.assertEqual(frame.analogue_gain_mask[1], 0x0123)
        self.assertEqual(frame.adc_voltage_v(1)[0], 1000 * 2.5/4096)
        self.assertEqual(frame.monotonic_ns, frame.frame_received_monotonic_ns)
        self.assertIsNone(frame.packet_crc_ok)
        self.assertIsNone(frame.table_crc32)
        self.assertEqual(frame.adc_value_kind, "firmware_rc_estimate")

    def test_ages_are_based_on_native_sequence_and_reset_on_gap(self):
        decoder = NativeFingerDecoder(WAVES)
        decoder.decode(native(0xFFFFFFFF), 100)
        frame = decoder.decode(native(0, survey=True), 200)
        self.assertFalse(frame.bandwidth_discontinuity)
        self.assertEqual(frame.point_age_frames, tuple(0 if i in SENTINELS else 1 for i in range(45)))
        gap = decoder.decode(native(2, survey=True), 300)
        self.assertTrue(gap.bandwidth_discontinuity)
        self.assertEqual(gap.point_age_frames, tuple(0 if i in SENTINELS else None for i in range(45)))

    def test_initial_sparse_cache_is_not_claimed_known(self):
        frame = NativeFingerDecoder(WAVES).decode(native(survey=True), 100)
        self.assertEqual(sum(frame.fresh_point_mask), 9)
        self.assertEqual(sum(age is None for age in frame.point_age_frames), 36)

    def test_invalid_frames_and_time_fail(self):
        for payload in (native()[:-2], native()+b"extra", native(active=1), native(selector=4),
                        b"prefix"+native()):
            with self.subTest(payload=payload[:8]), self.assertRaises(ValueError):
                NativeFingerDecoder(WAVES).decode(payload, 100)
        decoder = NativeFingerDecoder(WAVES)
        decoder.decode(native(), 100)
        with self.assertRaisesRegex(ValueError, "backwards"):
            decoder.decode(native(2), 99)

    def test_equal_clock_ticks_are_retained_not_fabricated(self):
        decoder = NativeFingerDecoder(WAVES)
        decoder.decode(native(), 100)
        frame = decoder.decode(native(2), 100)
        self.assertEqual(frame.frame_received_monotonic_ns, 100)
        self.assertFalse(frame.bandwidth_discontinuity)

    def test_export_merges_receive_times_and_preserves_quarantine(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/"source"
            with FingerSessionWriter(source/"session", manifest_extra={"training_eligible": True}) as writer:
                writer.write_event(ContactEvent("pre_contact", monotonic_ns=50, event_id="before"))
                writer.write_event(ContactEvent("contact", monotonic_ns=150, event_id="touch"))
                # GUI emits the earlier ADC frame after the new contact event.
                writer.write_frame(SpectrumFrame(1, WAVES, [1000]*45, monotonic_ns=200,
                                                frame_received_monotonic_ns=100,
                                                digital_gain={1: 1}))
            rows = [{"schema": "fbg-native-receive/v1", "record_number": i,
                     "host_received_monotonic_ns": 100*i, "payload_hex": native(i).hex()}
                    for i in (1, 2)]
            (source/"native_frames.jsonl").write_text("\n".join(json.dumps(r) for r in rows), "utf-8")
            destination = Path(directory)/"export"
            result = export_diagnostic_session(source, destination)
            self.assertEqual(result["native_frames_exported"], 2)
            frames = [json.loads(line) for line in (destination/"frames.jsonl").read_text("utf-8").splitlines()]
            events = [json.loads(line) for line in (destination/"events.jsonl").read_text("utf-8").splitlines()]
            self.assertEqual([r["monotonic_ns"] for r in frames], [100, 200])
            self.assertEqual([r["event_id"] for r in events], ["before", "touch"])
            self.assertNotIn("phase_event_id", frames[0]["quality"])
            with self.assertRaisesRegex(ValueError, "ineligible"):
                load_session_training_samples(destination)
            with self.assertRaises(FileExistsError):
                export_diagnostic_session(source, destination)

    def test_main_coverage_layout_and_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/"source"
            with FingerSessionWriter(source, manifest_extra={"source_at_start": "lan"}) as writer:
                writer.write_frame(SpectrumFrame(1, WAVES, [1000]*45, monotonic_ns=100))
            (source/"native_frames.jsonl").write_text(json.dumps({
                "schema": "fbg-native-receive/v1", "record_number": 1,
                "host_received_monotonic_ns": 100, "payload_hex": native().hex()}), "utf-8")
            destination = Path(directory)/"export"
            export_diagnostic_session(source, destination)
            row = json.loads((destination/"frames.jsonl").read_text("utf-8"))
            self.assertEqual(row["source"], "LAN_NATIVE")
            manifest = json.loads((source/"manifest.json").read_text("utf-8"))
            manifest.pop("closed_utc")
            (source/"manifest.json").write_text(json.dumps(manifest), "utf-8")
            with self.assertRaisesRegex(ValueError, "still open"):
                export_diagnostic_session(source, Path(directory)/"open_export")
            manifest["closed_utc"] = "2026-09-05T00:00:00Z"
            manifest["metadata"]["source_at_start"] = "mqtt"
            (source/"manifest.json").write_text(json.dumps(manifest), "utf-8")
            with self.assertRaisesRegex(ValueError, "transport"):
                export_diagnostic_session(source, Path(directory)/"wrong_transport")
            manifest["metadata"]["source_at_start"] = "local"
            (source/"manifest.json").write_text(json.dumps(manifest), "utf-8")
            with patch("native_finger_session._sha256", side_effect=["initial"]*4+["changed"]):
                with self.assertRaisesRegex(ValueError, "changed during export"):
                    export_diagnostic_session(source, Path(directory)/"changed_export")


if __name__ == "__main__":
    unittest.main()
