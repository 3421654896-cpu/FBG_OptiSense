"""Decode loss-visible native logs into timestamped CH1 diagnostic sessions.

No devices are opened. Host receive time is retained, not replaced with UI time.
An exported session is quarantined: receive/event clocks are not synchronized
ADC/force truth, and native USB frames do not carry a table or packet CRC.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path

from finger_dataset import ContactEvent, FingerSessionWriter, SpectrumFrame
from frame_telemetry import decode_board_frame_schedule, decode_board_frame_timing, extract_raw_scan_frames
from verify_runtime_firmware import board_extension

TIA_OHM = (2000, 40000, 5000, 20000)
PROFILE_NAMES = {0: "MAP45", 1: "SURVEY9", 2: "TRACK13_WIDE", 3: "TRACK11_SINGLE"}


class NativeFingerDecoder:
    def __init__(self, wavelengths_nm, *, digital_gain_audit=1.0, source="USB_NATIVE"):
        self.wavelengths = tuple(float(v) for v in wavelengths_nm)
        if len(self.wavelengths) != 45 or not all(math.isfinite(v) for v in self.wavelengths):
            raise ValueError("Exactly 45 finite captured wavelengths are required")
        if not math.isfinite(digital_gain_audit) or digital_gain_audit <= 0:
            raise ValueError("Invalid digital gain audit value")
        self.digital_gain_audit = digital_gain_audit
        self.source = str(source)
        self.previous_sequence = None
        self.previous_receive_ns = None
        self.ages = [None] * 45
        self.records = 0

    def decode(self, payload, received_ns):
        if type(received_ns) is not int or received_ns < 0:
            raise ValueError("Invalid receive timestamp")
        if self.previous_receive_ns is not None and received_ns < self.previous_receive_ns:
            raise ValueError("Receive clock moved backwards")
        frame = bytes(payload)
        buffer = bytearray(frame)
        if extract_raw_scan_frames(buffer) != [frame] or buffer:
            raise ValueError("Malformed native scan frame")
        if int.from_bytes(frame[2:4], "big") != 45:
            raise ValueError("Native CH1 capture requires 45 table slots")
        extension = board_extension(frame)
        if extension is None:
            raise ValueError("Missing board status extension")
        timing = decode_board_frame_timing(extension)
        schedule = decode_board_frame_schedule(extension)
        if timing is None or schedule is None or not schedule.has_point_timing:
            raise ValueError("Explicit fresh-point schedule and sample timing are required")
        if not extension[16] & 2:
            raise ValueError("CH1 is not active in this frame")
        selector = int(extension[23])
        if selector >= len(TIA_OHM):
            raise ValueError("Unsupported CH1 transimpedance selector")
        # Four interleaved uint16 channel values per table row; preserve CH1.
        codes = tuple(int.from_bytes(frame[6+i*8:8+i*8], "big") for i in range(45))
        position = 4 + 45 * 8 + 1
        for _ in range(4):
            position += 1 + 4 * frame[position]
        gain_mask = int.from_bytes(frame[position+6:position+8], "big")
        delta = None if self.previous_sequence is None else (timing.sequence-self.previous_sequence) & 0xFFFFFFFF
        broken = delta is not None and delta != 1
        # Never infer ages across a missing frame, duplicate, or reset.
        ages = [None] * 45 if broken else list(self.ages)
        fresh = set(schedule.fresh_indices)
        ages = [0 if i in fresh else None if ages[i] is None else ages[i]+1 for i in range(45)]
        flags = ["native_usb_table_crc_unavailable", "adc_event_synchronization_unverified"]
        if broken:
            flags.append("native_device_sequence_discontinuity")
        if any(code > 4095 for code in codes):
            flags.append("firmware_estimate_outside_adc_range")
        result = SpectrumFrame(
            sequence=self.records+1, monotonic_ns=received_ns,
            frame_received_monotonic_ns=received_ns, wavelengths_nm=self.wavelengths,
            raw_adc_codes={1: codes}, transimpedance_ohm={1: TIA_OHM[selector]},
            feedback_selector={1: selector}, analogue_gain_mask={1: gain_mask},
            digital_gain={1: self.digital_gain_audit}, device_sequence=timing.sequence,
            # These are absent in the USB payload, not implicitly verified.
            table_crc32=None, packet_crc_ok=None, device_boot_id=None,
            acquisition_profile=PROFILE_NAMES[schedule.profile], schedule_version=schedule.version,
            acquisition_profile_code=schedule.profile, adc_value_kind="firmware_rc_estimate",
            fresh_point_mask=tuple(i in fresh for i in range(45)), point_age_frames=ages,
            sample_offset_us=schedule.sample_offset_us, map_age_frames=schedule.map_age_frames,
            bandwidth_discontinuity=broken or schedule.bandwidth_discontinuity,
            frame_start_device_ms=schedule.frame_start_ms, quality_flags=flags,
            quality={"capture_path": "native_receive_without_gui_decimation",
                     "label_timebase": "host_frame_receive_not_ui_snapshot",
                     "adc_event_synchronization_verified": False,
                     "board_acquisition_duration_us": timing.acquisition_duration_us,
                     "board_sequence_delta": delta},
            mode="stress", source=self.source,
        )
        self.records += 1
        self.previous_sequence = timing.sequence
        self.previous_receive_ns = received_ns
        self.ages = ages
        return result


def _read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export_diagnostic_session(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError("Refusing to overwrite an exported session")
    session = source / "session" if (source / "session/manifest.json").exists() else source
    inputs = {"native": source / "native_frames.jsonl",
              "ui_frames": session / "frames.jsonl",
              "events": session / "events.jsonl",
              "manifest": session / "manifest.json"}
    if not inputs["native"].exists():
        inputs["native"] = session / "native_frames.jsonl"
    hashes = {key: _sha256(path) for key, path in inputs.items()}
    manifest = json.loads(inputs["manifest"].read_text("utf-8"))
    if not manifest.get("closed_utc"):
        raise ValueError("Source session is still open; finish recording before export")
    # Wavelengths are not in the raw frame. Require one unchanged recorded
    # table; never silently substitute the current editable table on disk.
    ui = list(_read_jsonl(inputs["ui_frames"]))
    if not ui:
        raise ValueError("Captured wavelength context is unavailable")
    waves = ui[0]["wavelengths_nm"]
    if any(row["wavelengths_nm"] != waves for row in ui):
        raise ValueError("Wavelength context changed within the capture")
    gain = ui[0].get("digital_gain_audit", {}).get("1", 1.0)
    if any(row.get("digital_gain_audit", {}).get("1", 1.0) != gain for row in ui):
        raise ValueError("Digital audit gain changed without native timing context")
    transport = manifest.get("metadata", {}).get("source_at_start", "local")
    if transport not in {"local", "lan"}:
        raise ValueError("Unsupported native source transport")
    decoder = NativeFingerDecoder(waves, digital_gain_audit=gain,
                                  source="LAN_NATIVE" if transport == "lan" else "USB_NATIVE")
    event_fields = {f.name for f in fields(ContactEvent)}
    events = [ContactEvent(**{k: v for k, v in r.items() if k in event_fields})
              for r in _read_jsonl(inputs["events"])]
    if any(b.monotonic_ns < a.monotonic_ns for a, b in zip(events, events[1:])):
        raise ValueError("Source event times moved backwards")
    metadata = {**manifest.get("metadata", {}),
                "training_eligible": False,
                "training_exclusion_reason": "Diagnostic export; no verified table CRC or ADC/contact synchronization. Export does not approve training.",
                "parent_session_id": manifest["session_id"], "source_sha256": hashes,
                "capture_path": "native_receive_without_gui_decimation",
                "wavelength_context": "unchanged_table_in_original_ui_log_not_wire_verified",
                "physical_15hz_verified": False, "adc_event_synchronization_verified": False}
    event_index = 0
    # Offline merge prevents the UI callback order from moving an older native
    # frame into a newer contact phase. Original event times/IDs remain intact.
    with FingerSessionWriter(destination, manifest_extra=metadata) as writer:
        for ordinal, record in enumerate(_read_jsonl(inputs["native"]), 1):
            if record.get("schema") != "fbg-native-receive/v1" or record.get("record_number") != ordinal:
                raise ValueError("Invalid native record schema/order")
            frame = decoder.decode(bytes.fromhex(record["payload_hex"]), record["host_received_monotonic_ns"])
            while event_index < len(events) and events[event_index].monotonic_ns <= frame.monotonic_ns:
                writer.write_event(events[event_index])
                event_index += 1
            writer.write_frame(frame)
        for event in events[event_index:]:
            writer.write_event(event)
    if decoder.records == 0:
        raise ValueError("No native frames to export")
    if any(_sha256(path) != hashes[key] for key, path in inputs.items()):
        raise ValueError("Source files changed during export; output remains quarantined")
    return {"native_frames_exported": decoder.records, "events_preserved": len(events),
            "training_eligible": False, "destination": str(destination.resolve()), "source_sha256": hashes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(export_diagnostic_session(args.source, args.destination), ensure_ascii=False, indent=2))
