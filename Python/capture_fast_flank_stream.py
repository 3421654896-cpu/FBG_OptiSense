"""Capture the bounded real-time 18-flank CH1 stream; never move the CNC.

The first 16 cycles are below the empirical minimum warm-up by protocol.  The
flag does not certify absolute optical stability: a causal per-session baseline
is still required. Values are direct ADC codes with board timestamps, never RC
estimates, cached samples, scaled plot values, wavelength truth, force truth or
contact-area truth.
"""

from __future__ import annotations

import argparse
import binascii
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import serial

import app_JDSU as app
from benchmark_stress_realtime import _close_shutter, _safe_disarm
from capture_ch1_rc_diagnostic import load_installed_rows
from capture_ch1_teacher_spectrum import save_checkpoint, sha256_file
from capture_short_route_teacher import codes_crc


HERE = Path(__file__).resolve().parent
COMMAND_SIZE = 808
FIRMWARE = 0x00010048  # 1.0.72 adds operator-edited nonuniform T45 routes
MIN_CYCLES = 32
MAX_CYCLES = 1024
WARMUP_CYCLES = 16
DATA_LENGTH = 210
END_LENGTH = 42
REJECT_LENGTH = 20
ROWS = (1, 3, 6, 8, 11, 13, 16, 18, 21, 23, 26, 28, 31, 33, 38, 36, 41, 43)


class StreamRejected(RuntimeError):
    def __init__(self, frame: bytes):
        self.frame = frame
        self.reason = int.from_bytes(frame[8:10], "big") if len(frame) >= 10 else None
        super().__init__(f"fast flank stream rejected: reason=0x{self.reason or 0:04x}")


def command(cycles: int, tag: int) -> bytes:
    if type(cycles) is not int or not MIN_CYCLES <= cycles <= MAX_CYCLES:
        raise ValueError(f"cycles must be {MIN_CYCLES}..{MAX_CYCLES}")
    if type(tag) is not int or not 1 <= tag <= 0xFFFFFFFF:
        raise ValueError("tag must be a non-zero uint32")
    data = bytearray(COMMAND_SIZE)
    data[:4] = b"\xff\xff\x03\x07"
    data[4] = 1  # only the measured G8 right->left route
    data[6:8] = (1500).to_bytes(2, "big")
    data[8:12] = tag.to_bytes(4, "big")
    data[12:16] = b"F15!"
    data[16:18] = cycles.to_bytes(2, "big")
    data[18] = WARMUP_CYCLES
    return bytes(data)


def _crc_ok(frame: bytes) -> bool:
    return (
        len(frame) >= 6
        and frame[-2:] == b"\x9d\xd9"
        and int.from_bytes(frame[-6:-2], "big") == binascii.crc32(frame[:-6])
    )


def extract_frames(buffer: bytearray):
    """Yield complete known frames while retaining a partial suffix."""
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
        if kind == 3 and status == 0x7F:
            length = REJECT_LENGTH
        elif kind == 3:
            length = DATA_LENGTH
        elif kind == 4:
            length = END_LENGTH
        else:
            del buffer[0]
            continue
        if len(buffer) < length:
            return
        frame = bytes(buffer[:length])
        del buffer[:length]
        yield frame


def decode_data(frame: bytes, *, tag: int, table_crc32: int,
                expected_firmware: int = FIRMWARE, spacing_us=None) -> dict:
    if len(frame) != DATA_LENGTH or frame[:3] != b"\xd9\x9d\x03" or not _crc_ok(frame):
        raise ValueError("data frame header/length/CRC/trailer mismatch")
    flags = frame[3]
    if flags & ~1:
        raise ValueError("unknown data flags")
    if int.from_bytes(frame[4:8], "big") != tag:
        raise ValueError("data tag mismatch")
    firmware = int.from_bytes(frame[8:12], "big")
    sequence = int.from_bytes(frame[12:16], "big")
    cycle_start_us = int.from_bytes(frame[16:20], "big")
    elapsed_us = int.from_bytes(frame[20:24], "big")
    if firmware != expected_firmware or int.from_bytes(frame[24:28], "big") != table_crc32:
        raise ValueError("firmware or installed table CRC mismatch")
    if spacing_us is not None and (type(spacing_us) is not int or not 50 <= spacing_us <= 600 or spacing_us % 25):
        raise ValueError('Invalid explicit F18 spacing')
    spacing_code = 0 if spacing_us is None else spacing_us // 25
    if frame[28:34] != bytes((1, len(ROWS), WARMUP_CYCLES, spacing_code, 0, 2)):
        raise ValueError("route/count/warm-up/feedback mismatch")
    i2c_errors = int.from_bytes(frame[34:36], "big")
    spi_errors = int.from_bytes(frame[36:40], "big")
    if tuple(frame[40:58]) != ROWS:
        raise ValueError("fast flank row order mismatch")
    minimum_warmup_complete = bool(flags & 1)
    if minimum_warmup_complete != (sequence >= WARMUP_CYCLES):
        raise ValueError("warm-up validity flag is non-causal")
    records = []
    previous_second_end = -1
    for local, row in enumerate(ROWS):
        at = 58 + local * 8
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
        "minimum_warmup_complete": minimum_warmup_complete,
        "i2c_error_delta": i2c_errors,
        "spi_error_delta": spi_errors,
        "records": records,
        "wire_hex": frame.hex(),
    }


def decode_end(frame: bytes, *, tag: int, table_crc32: int, frame_kind: int = 4,
               feedback_selectors=(0, 2), expected_firmware: int = FIRMWARE) -> dict:
    if (
        len(frame) != END_LENGTH
        or frame[:3] != bytes((0xD9, 0x9D, frame_kind))
        or not _crc_ok(frame)
    ):
        raise ValueError("terminal frame header/length/CRC/trailer mismatch")
    if int.from_bytes(frame[4:8], "big") != tag:
        raise ValueError("terminal tag mismatch")
    if int.from_bytes(frame[8:12], "big") != expected_firmware:
        raise ValueError("terminal firmware mismatch")
    if frame[18] != 1 or frame[20:22] != bytes(feedback_selectors):
        raise ValueError("terminal route/feedback mismatch")
    if int.from_bytes(frame[32:36], "big") != table_crc32:
        raise ValueError("terminal table CRC mismatch")
    return {
        "status": frame[3],
        "completed_cycles": int.from_bytes(frame[12:14], "big"),
        "requested_cycles": int.from_bytes(frame[14:16], "big"),
        "stop_reason": int.from_bytes(frame[16:18], "big"),
        "firmware_shutter_asserted": bool(frame[19]),
        "i2c_error_delta": int.from_bytes(frame[22:24], "big"),
        "spi_error_delta": int.from_bytes(frame[24:28], "big"),
        "elapsed_us": int.from_bytes(frame[28:32], "big"),
        "wire_hex": frame.hex(),
    }


def receive(
    device,
    *,
    cycles: int,
    tag: int,
    table_crc32: int,
    checkpoint=None,
    on_frame=None,
    should_stop=None,
    spacing_us=None,
):
    expected_firmware = FIRMWARE
    if spacing_us is not None:
        from f18_spacing_protocol import command as spaced_command, FIRMWARE as spaced_firmware
        spaced_command(cycles, tag, spacing_us)  # validate before reading
        expected_firmware = spaced_firmware
    buffer = bytearray()
    frames = []
    terminal = None
    deadline = time.monotonic() + (65.0 if spacing_us is not None else min(40.0, 5.0 + cycles * 0.04))
    while time.monotonic() < deadline and terminal is None:
        if should_stop is not None and should_stop():
            raise InterruptedError("fast flank stream stop requested")
        buffer.extend(device.read(8192))
        for wire in extract_frames(buffer):
            if wire[:4] == b"\xd9\x9d\x03\x7f":
                if int.from_bytes(wire[4:8], "big") == tag:
                    raise StreamRejected(wire)
                continue
            if wire[2] == 3:
                item = decode_data(wire, tag=tag, table_crc32=table_crc32,
                                   expected_firmware=expected_firmware, spacing_us=spacing_us)
                if item["sequence"] != len(frames):
                    raise ValueError("missing, duplicate or reordered board cycle")
                if frames and item["cycle_start_us"] <= frames[-1]["cycle_start_us"]:
                    raise ValueError("cycle start time is not strictly increasing")
                if item["i2c_error_delta"] or item["spi_error_delta"]:
                    raise RuntimeError("board I2C/SPI error during fast flank stream")
                frames.append(item)
                if on_frame is not None:
                    on_frame(item)
                if checkpoint and len(frames) % 32 == 0:
                    checkpoint(frames)
            else:
                terminal = decode_end(wire, tag=tag, table_crc32=table_crc32,
                                      expected_firmware=expected_firmware)
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
        raise RuntimeError(f"fast flank stream did not finish cleanly: {terminal}")
    return frames, terminal


def analyze(frames: list[dict]) -> dict:
    post_warmup = [frame for frame in frames if frame["minimum_warmup_complete"]]
    starts = [frame["cycle_start_us"] for frame in post_warmup]
    intervals = [b - a for a, b in zip(starts, starts[1:])]
    if len(intervals) < 2:
        raise ValueError("too few post-warm-up cycles")
    return {
        "total_cycles": len(frames),
        "minimum_warmup_discarded_cycles": len(frames) - len(post_warmup),
        "post_minimum_warmup_cycles": len(post_warmup),
        "board_interval_mean_us": statistics.fmean(intervals),
        "board_interval_median_us": statistics.median(intervals),
        "board_interval_p95_us": sorted(intervals)[round(0.95 * (len(intervals) - 1))],
        "board_interval_max_us": max(intervals),
        "board_rate_hz": 1_000_000.0 / statistics.fmean(intervals),
        "meets_sampling_rate_for_15hz": 1_000_000.0 / max(intervals) >= 30.0,
        "absolute_optical_stability_verified": False,
        "physical_15hz_verified": False,
        "location_accuracy_verified": False,
        "contact_area_verified": False,
        "note": (
            "Sampling-rate gate only. The 16-cycle flag is a minimum warm-up, "
            "not an absolute-stability certificate; no pressure excitation or "
            "area truth was present."
        ),
    }


def plot(frames: list[dict], output: Path):
    figure, axes = plt.subplots(3, 3, figsize=(13, 8), sharex=True, layout="constrained")
    for grating, axis in enumerate(axes.flat):
        left, right = grating * 2, grating * 2 + 1
        sequence = [frame["sequence"] for frame in frames]
        axis.plot(sequence, [frame["records"][left]["second_code"] for frame in frames], label="left")
        axis.plot(sequence, [frame["records"][right]["second_code"] for frame in frames], label="right")
        axis.axvspan(-0.5, WARMUP_CYCLES - 0.5, color="#f3a34a", alpha=0.15)
        axis.set_title(f"G{grating + 1}")
        axis.grid(alpha=0.2)
        if grating == 0:
            axis.legend(fontsize=8)
    figure.supxlabel("board cycle")
    figure.supylabel("direct CH1 ADC code")
    figure.savefig(output, dpi=145)
    plt.close(figure)


def execute_device(
    device,
    output: Path,
    *,
    cycles: int = 256,
    on_frame=None,
    should_stop=None,
    spacing_us=None,
):
    packet_builder = command
    expected_firmware = FIRMWARE
    if spacing_us is not None:
        from f18_spacing_protocol import command as spaced_command, FIRMWARE as spaced_firmware
        packet_builder = lambda count, tag: spaced_command(count, tag, spacing_us)
        expected_firmware = spaced_firmware
    packet_builder(cycles, 1)  # Reject invalid requests before files or hardware.
    output = Path(output)
    png = output.with_suffix(".png")
    if output.exists() or png.exists():
        raise FileExistsError(output)
    rows, _ = load_installed_rows()
    table_crc32 = codes_crc(rows)
    report = {
        "schema": "ch1_fast_flank_stream_capture_v1",
        "scope": __doc__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": False,
        "cycles_requested": cycles,
        "requested_spacing_us": 600 if spacing_us is None else spacing_us,
        "explicit_spacing_protocol": spacing_us is not None,
        "expected_firmware_version": expected_firmware,
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
        worker = app.EqualIntervalWorker(
            device, (), {}, settle_s=0.0, feedback_selectors=(0, 2)
        )
        if not worker._feedback_command_and_ack() or not worker._interruptible_wait(0.01):
            raise RuntimeError("fixed feedback ACK/cancellation")
        tag = int(time.time_ns() & 0x7FFFFFFF) or 1
        device.reset_input_buffer()
        device.write(packet_builder(cycles, tag))
        device.flush()

        def checkpoint(frames):
            report["frames"] = list(frames)
            save_checkpoint(output, report)

        frames, terminal = receive(
            device,
            cycles=cycles,
            tag=tag,
            table_crc32=table_crc32,
            checkpoint=checkpoint,
            on_frame=on_frame,
            should_stop=should_stop,
            spacing_us=spacing_us,
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
        print(
            f"disarm_confirmed={int(disarmed)} soa_shutter_confirmed={int(shuttered)}",
            flush=True,
        )
        if not disarmed or not shuttered:
            raise RuntimeError("exact final optical shutdown not confirmed")
    plot(report["frames"], png)
    print(json.dumps(report["analysis"], ensure_ascii=False, indent=2))
    return report


def execute(
    output: Path,
    *,
    cycles: int = 256,
    port: str = "COM6",
    on_frame=None,
    should_stop=None,
    spacing_us=None,
):
    if spacing_us is not None:
        from f18_spacing_protocol import command as spaced_command
        spaced_command(cycles, 1, spacing_us)
    else:
        command(cycles, 1)
    with serial.Serial(port, 2_000_000, timeout=0.03, write_timeout=3) as device:
        return execute_device(
            device,
            output,
            cycles=cycles,
            on_frame=on_frame,
            should_stop=should_stop,
            spacing_us=spacing_us,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=256)
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--spacing-us", type=int, default=None,
                        help="Opt-in F18 spacing in merged 1.0.59; omitted keeps the legacy wire layout")
    args = parser.parse_args()
    execute(args.output, cycles=args.cycles, port=args.port, spacing_us=args.spacing_us)
