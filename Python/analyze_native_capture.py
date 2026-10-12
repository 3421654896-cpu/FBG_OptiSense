"""Offline native-receive/UI comparison. Never opens a device or labels pressure."""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_stress_realtime import (
    _summarize_board_telemetry, _summarize_schedule_freshness,
)
from frame_telemetry import (
    decode_board_frame_timing, decode_board_frame_schedule, extract_raw_scan_frames,
)
from verify_runtime_firmware import board_extension


def analyze(path):
    native_path = path / "native_frames.jsonl"
    if not native_path.exists():
        native_path = path / "session/native_frames.jsonl"
    records = [json.loads(line) for line in native_path.read_text("utf-8").splitlines()]
    if len(records) < 2:
        raise ValueError("At least two native frames are required")
    timings, schedules, receive_ns = [], [], []
    for number, record in enumerate(records, 1):
        if record["schema"] != "fbg-native-receive/v1" or record["record_number"] != number:
            raise ValueError("Invalid native record order/schema")
        frame = bytes.fromhex(record["payload_hex"])
        buffer = bytearray(frame)
        if extract_raw_scan_frames(buffer) != [frame] or buffer:
            raise ValueError(f"Invalid transport frame at record {number}")
        extension = board_extension(frame)
        if extension is None:
            raise ValueError("Native frame has no board extension")
        timings.append(decode_board_frame_timing(extension))
        schedules.append(decode_board_frame_schedule(extension))
        receive_ns.append(record["host_received_monotonic_ns"])
    receive_ns = np.asarray(receive_ns, dtype=np.int64)
    delta_ms = np.diff(receive_ns) / 1e6
    if np.any(delta_ms < 0) or receive_ns[-1] <= receive_ns[0]:
        raise ValueError("Host receive clock moved backwards or has no duration")
    gui_path = path / "session/frames.jsonl"
    if not gui_path.exists():
        gui_path = path / "frames.jsonl"
    gui = [json.loads(line) for line in gui_path.read_text("utf-8").splitlines()]
    # Count inside the common time span; coverage starts before the subscription
    # and ends before stop/shutter acknowledgements, so total counts alone differ.
    gui_window = [r for r in gui if receive_ns[0] <= r["frame_received_monotonic_ns"] <= receive_ns[-1]]
    gui_sequences = {r["device_sequence"] for r in gui_window}
    native_sequences = {t.sequence for t in timings if t is not None}
    result = {
        "purpose": "receive-path diagnostic, not 15 Hz physical pressure validation",
        "native_frames": len(records), "gui_records_total": len(gui),
        "gui_records_in_native_window": len(gui_window),
        "native_sequences_missing_from_gui_records": len(native_sequences - gui_sequences),
        "gui_sequences_not_in_native_file": len(gui_sequences - native_sequences),
        "native_receive_span_s": float((receive_ns[-1] - receive_ns[0]) / 1e9),
        "native_receive_rate_hz": float((len(records)-1) * 1e9 / (receive_ns[-1] - receive_ns[0])),
        "native_receive_gap_ms_p50_p95_max": np.percentile(delta_ms, [50, 95, 100]).tolist(),
        "board_timing": _summarize_board_telemetry(timings),
        "schedule": _summarize_schedule_freshness(schedules),
        "value_note": "Verbatim firmware frames; CH1 values remain RC estimates. Cached points are not fresh spectra. No contact-area or pressure labels are inferred.",
    }
    (path / "native_diagnostics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), "utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(analyze(parser.parse_args().directory), ensure_ascii=False, indent=2))
