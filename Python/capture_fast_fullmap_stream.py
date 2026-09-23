"""Capture 45 fresh direct CH1 points per board cycle; never move CNC.

F45 is an isolated qualification protocol.  It preserves the installed table
order, the nine section-boundary waits, and the qualified hidden predecessor
for the path-sensitive third row.  The requested two-read spacing is carried
in every frame.  No cache, RC reconstruction, digital scaling, wavelength
correction, position inference, or contact-area inference is applied here.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import serial

import app_JDSU as app
import capture_fast_flank_stream as flank
from benchmark_stress_realtime import _close_shutter, _safe_disarm
from capture_ch1_rc_diagnostic import load_installed_rows
from capture_ch1_teacher_spectrum import save_checkpoint, sha256_file
from capture_short_route_teacher import codes_crc


HERE = Path(__file__).resolve().parent
COMMAND_SIZE = 808
FIRMWARE = flank.FIRMWARE
MIN_CYCLES = flank.MIN_CYCLES
MAX_CYCLES = 65535
WARMUP_CYCLES = flank.WARMUP_CYCLES
DATA_LENGTH = 453
END_LENGTH = flank.END_LENGTH
REJECT_LENGTH = flank.REJECT_LENGTH
ROWS = tuple(range(45))
MIN_SPACING_US = 50
MAX_SPACING_US = 600
SPACING_QUANTUM_US = 25
PRECONDITION_HOLD_US = 5000


StreamRejected = flank.StreamRejected


def command(cycles: int, tag: int, spacing_us: int, *, ch1_selector: int = 2) -> bytes:
    if type(cycles) is not int or not MIN_CYCLES <= cycles <= MAX_CYCLES:
        raise ValueError(f"cycles must be {MIN_CYCLES}..{MAX_CYCLES}")
    if type(tag) is not int or not 1 <= tag <= 0xFFFFFFFF:
        raise ValueError("tag must be a non-zero uint32")
    if (
        type(spacing_us) is not int
        or not MIN_SPACING_US <= spacing_us <= MAX_SPACING_US
        or spacing_us % SPACING_QUANTUM_US
    ):
        raise ValueError("spacing_us must be a 25 us multiple from 50 through 600")
    data = bytearray(COMMAND_SIZE)
    data[:4] = b"\xff\xff\x03\x09"
    data[4] = 1
    data[6:8] = PRECONDITION_HOLD_US.to_bytes(2, "big")
    data[8:12] = tag.to_bytes(4, "big")
    data[12:16] = b"F45!"
    data[16:18] = cycles.to_bytes(2, "big")
    data[18] = WARMUP_CYCLES
    data[19:21] = spacing_us.to_bytes(2, "big")
    if type(ch1_selector) is not int or ch1_selector not in range(4):
        raise ValueError('invalid CH1 selector')
    if ch1_selector != 2:
        data[21:23] = bytes((0x47, ch1_selector))
    return bytes(data)


def extract_frames(buffer: bytearray, *, data_kind: int = 3, end_kind: int = 4):
    while True:
        start = buffer.find(b"\xd9\x9d")
        if start < 0:
            if len(buffer) > 1:
                del buffer[:-1]
            return
        if start:
            del buffer[:start]
        if len(buffer) < 4:
            return
        kind, status = buffer[2], buffer[3]
        if kind == data_kind and status == 0x7F:
            length = REJECT_LENGTH
        elif kind == data_kind:
            length = DATA_LENGTH
        elif kind == end_kind:
            length = END_LENGTH
        else:
            del buffer[0]
            continue
        if len(buffer) < length:
            return
        frame = bytes(buffer[:length])
        del buffer[:length]
        yield frame


def decode_data(
    frame: bytes,
    *,
    tag: int,
    table_crc32: int,
    spacing_us: int,
    frame_kind: int = 3,
    ch1_selector: int = 2,
    expected_firmware: int = FIRMWARE,
) -> dict:
    if (
        len(frame) != DATA_LENGTH
        or frame[:3] != bytes((0xD9, 0x9D, frame_kind))
        or not flank._crc_ok(frame)
    ):
        raise ValueError("data frame header/length/CRC/trailer mismatch")
    flags = frame[3]
    if flags & ~3:
        raise ValueError("unknown data flags")
    if int.from_bytes(frame[4:8], "big") != tag:
        raise ValueError("data tag mismatch")
    firmware = int.from_bytes(frame[8:12], "big")
    sequence = int.from_bytes(frame[12:16], "big")
    cycle_start_us = int.from_bytes(frame[16:20], "big")
    elapsed_us = int.from_bytes(frame[20:24], "big")
    actual_table_crc = int.from_bytes(frame[24:28], "big")
    if firmware != expected_firmware or actual_table_crc != table_crc32:
        raise ValueError(f"firmware or installed table CRC mismatch: firmware received={firmware:#010x} expected={expected_firmware:#010x}; table received={actual_table_crc:#010x} expected={table_crc32:#010x}")
    spacing_code = spacing_us // SPACING_QUANTUM_US
    if frame[28:34] != bytes((1, len(ROWS), WARMUP_CYCLES, spacing_code, 0, ch1_selector)):
        raise ValueError("route/count/warm-up/spacing/feedback mismatch")
    i2c_errors = int.from_bytes(frame[34:36], "big")
    spi_errors = int.from_bytes(frame[36:40], "big")
    rows_end = 40 + len(ROWS)
    if tuple(frame[40:rows_end]) != ROWS:
        raise ValueError("full-map row order mismatch")
    minimum_warmup_complete = bool(flags & 1)
    if minimum_warmup_complete != (sequence >= WARMUP_CYCLES):
        raise ValueError("warm-up validity flag is non-causal")
    records = []
    previous_second_end = -1
    for local, row in enumerate(ROWS):
        at = rows_end + local * 8
        first = int.from_bytes(frame[at : at + 2], "big")
        second = int.from_bytes(frame[at + 2 : at + 4], "big")
        write_end = int.from_bytes(frame[at + 4 : at + 6], "big")
        second_end = int.from_bytes(frame[at + 6 : at + 8], "big")
        if not (0 <= first <= 4095 and 0 <= second <= 4095):
            raise ValueError("ADC code outside 12-bit range")
        if not previous_second_end < write_end < second_end <= elapsed_us + 1:
            raise ValueError("non-causal point timestamps")
        records.append(
            {
                "row": row,
                "first_code": first,
                "second_code": second,
                "write_end_us": write_end,
                "second_end_us": second_end,
            }
        )
        previous_second_end = second_end
    return {
        "sequence": sequence,
        "cycle_start_us": cycle_start_us,
        "elapsed_us": elapsed_us,
        "spacing_us": spacing_us,
        "feedback_selectors": [int(frame[32]), int(frame[33])],
        "firmware_version": firmware,
        "minimum_warmup_complete": minimum_warmup_complete,
        "lan_compact_first_only": bool(flags & 2),
        "second_code_is_physical_sample": not bool(flags & 2),
        "point_timestamps_are_physical": not bool(flags & 2),
        "i2c_error_delta": i2c_errors,
        "spi_error_delta": spi_errors,
        "records": records,
        "wire_hex": frame.hex(),
    }


def receive(
    device,
    *,
    cycles,
    tag,
    table_crc32,
    spacing_us,
    checkpoint=None,
    on_frame=None,
    should_stop=None,
    max_duration_s=55.0,
    data_kind=3,
    end_kind=4,
    ch1_selector=2,
    expected_firmware=FIRMWARE,
):
    buffer = bytearray()
    frames = []
    terminal = None
    if not math.isfinite(max_duration_s) or not 5 <= max_duration_s <= 240:
        raise ValueError('Capture duration cap must be 5..240 seconds')
    deadline = time.monotonic() + min(max_duration_s, 5.0 + cycles * 0.055)
    checkpoint_every = max(32, cycles // 16)
    while time.monotonic() < deadline and terminal is None:
        if should_stop is not None and should_stop():
            raise InterruptedError("fast full-map stream stop requested")
        buffer.extend(device.read(8192))
        for wire in extract_frames(buffer, data_kind=data_kind, end_kind=end_kind):
            if wire[:4] == bytes((0xD9, 0x9D, data_kind, 0x7F)):
                if int.from_bytes(wire[4:8], "big") == tag:
                    raise StreamRejected(wire)
                continue
            if wire[2] == data_kind:
                item = decode_data(
                    wire,
                    tag=tag,
                    table_crc32=table_crc32,
                    spacing_us=spacing_us,
                    frame_kind=data_kind,
                    ch1_selector=ch1_selector,
                    expected_firmware=expected_firmware,
                )
                if item["sequence"] != len(frames):
                    raise ValueError("missing, duplicate or reordered board cycle")
                if frames and item["cycle_start_us"] <= frames[-1]["cycle_start_us"]:
                    raise ValueError("cycle start time is not strictly increasing")
                if item["i2c_error_delta"] or item["spi_error_delta"]:
                    raise RuntimeError("board I2C/SPI error during fast full-map stream")
                frames.append(item)
                if on_frame is not None:
                    on_frame(item)
                if checkpoint and len(frames) % checkpoint_every == 0:
                    checkpoint(frames)
            else:
                terminal = flank.decode_end(
                    wire,
                    tag=tag,
                    table_crc32=table_crc32,
                    frame_kind=end_kind,
                    feedback_selectors=(0, ch1_selector),
                    expected_firmware=expected_firmware,
                )
    if terminal is None:
        raise TimeoutError(f"terminal frame missing after {len(frames)} cycles")
    if (
        terminal["status"] != 0
        or terminal["stop_reason"] != 0
        or not terminal["firmware_shutter_asserted"]
        or terminal["requested_cycles"] != cycles
        or terminal["completed_cycles"] != cycles
        or len(frames) != cycles
        or terminal["i2c_error_delta"]
        or terminal["spi_error_delta"]
    ):
        raise RuntimeError(f"fast full-map stream did not finish cleanly: {terminal}")
    return frames, terminal


def analyze(frames: list[dict]) -> dict:
    result = flank.analyze(frames)
    result["point_count_per_cycle"] = len(ROWS)
    result["fresh_point_count_per_cycle"] = len(ROWS)
    result["samples_per_grating"] = 5
    result["protocol"] = "F45"
    result["spacing_us"] = frames[0]["spacing_us"] if frames else None
    return result


def plot(frames: list[dict], output: Path):
    figure, axes = plt.subplots(3, 3, figsize=(13, 8), sharex=True, layout="constrained")
    sequence = [frame["sequence"] for frame in frames]
    for grating, axis in enumerate(axes.flat):
        for offset in range(5):
            local = grating * 5 + offset
            axis.plot(sequence, [f["records"][local]["second_code"] for f in frames])
        axis.axvspan(-0.5, WARMUP_CYCLES - 0.5, color="#f3a34a", alpha=0.15)
        axis.set_title(f"G{grating + 1}")
        axis.grid(alpha=0.2)
    figure.supxlabel("board cycle")
    figure.supylabel("direct CH1 ADC code")
    figure.savefig(output, dpi=145)
    plt.close(figure)


def execute_device(
    device,
    output: Path,
    *,
    cycles: int = 256,
    spacing_us: int = 75,
    on_frame=None,
    should_stop=None,
    max_duration_s=55.0,
    ch1_selector=2,
):
    output = Path(output)
    png = output.with_suffix(".png")
    if output.exists() or png.exists():
        raise FileExistsError(output)
    if not math.isfinite(max_duration_s) or not 5 <= max_duration_s <= 240:
        raise ValueError('Capture duration cap must be 5..240 seconds')
    rows, _ = load_installed_rows()
    table_crc32 = codes_crc(rows)
    report = {
        "schema": "ch1_fast_fullmap_stream_capture_v1",
        "scope": __doc__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": False,
        "cycles_requested": cycles,
        "spacing_us": spacing_us,
        "feedback_selectors": [0, ch1_selector],
        "ch1_transimpedance_kohm": app.PD_FEEDBACK_KOHM_BY_SELECTOR[ch1_selector],
        "host_duration_cap_s": max_duration_s,
        "route_rows": list(ROWS),
        "table_codes_crc32": table_crc32,
        "table_sha256": sha256_file(HERE / "mode_tables_from_fullband_2001.json"),
        "script_sha256": sha256_file(Path(__file__)),
        "training_eligible": False,
        "physical_15hz_verified": False,
        "measured_contact_area_mm2": None,
        "pressure_ground_truth": None,
        "cnc_motion_commands": 0,
        "frames": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(output, report)
    try:
        device.dtr = True
        if not _safe_disarm(device) or not _close_shutter(device):
            raise RuntimeError("initial exact disarm/shutter ACK missing")
        worker = app.EqualIntervalWorker(device, (), {}, settle_s=0.0, feedback_selectors=(0, ch1_selector))
        if not worker._feedback_command_and_ack() or not worker._interruptible_wait(0.01):
            raise RuntimeError("fixed feedback ACK/cancellation")
        tag = int(time.time_ns() & 0x7FFFFFFF) or 1
        device.reset_input_buffer()
        device.write(command(cycles, tag, spacing_us, ch1_selector=ch1_selector))
        device.flush()

        def checkpoint(items):
            report["frames"] = list(items)
            save_checkpoint(output, report)

        frames, terminal = receive(
            device,
            cycles=cycles,
            tag=tag,
            table_crc32=table_crc32,
            spacing_us=spacing_us,
            checkpoint=checkpoint,
            on_frame=on_frame,
            should_stop=should_stop,
            max_duration_s=max_duration_s,
            ch1_selector=ch1_selector,
        )
        report["frames"] = frames
        report["terminal"] = terminal
        report["analysis"] = analyze(frames)
        report["complete"] = True
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        errors = []
        try:
            disarmed = _safe_disarm(device)
        except Exception as exc:
            disarmed = False
            errors.append(f"disarm:{exc}")
        try:
            shuttered = _close_shutter(device)
        except Exception as exc:
            shuttered = False
            errors.append(f"shutter:{exc}")
        report["safety_cleanup"] = {
            "exact_disarm_ack": disarmed,
            "exact_soa_shutter_ack": shuttered,
            "errors": errors,
        }
        save_checkpoint(output, report)
        print(f"disarm_confirmed={int(disarmed)} soa_shutter_confirmed={int(shuttered)}", flush=True)
        if not disarmed or not shuttered:
            raise RuntimeError("exact final optical shutdown not confirmed")
    plot(report["frames"], png)
    print(json.dumps(report["analysis"], ensure_ascii=False, indent=2))
    return report


def execute(output: Path, *, cycles: int = 256, spacing_us: int = 75, port: str = "COM6", max_duration_s: float = 55.0, ch1_selector: int = 2):
    with serial.Serial(port, 2_000_000, timeout=0.03, write_timeout=3) as device:
        return execute_device(device, output, cycles=cycles, spacing_us=spacing_us, max_duration_s=max_duration_s, ch1_selector=ch1_selector)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=256)
    parser.add_argument("--spacing-us", type=int, default=75)
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--max-duration-s", type=float, default=55.0)
    parser.add_argument("--ch1-selector", type=int, choices=range(4), default=2)
    args = parser.parse_args()
    execute(args.output, cycles=args.cycles, spacing_us=args.spacing_us, port=args.port, max_duration_s=args.max_duration_s, ch1_selector=args.ch1_selector)
