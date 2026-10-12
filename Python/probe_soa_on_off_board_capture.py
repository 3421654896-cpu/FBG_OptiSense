"""Board-timestamped SOA optical on/off response capture.

Firmware command ``FF FF 03 0B`` switches only the PI11210 SOA hardware gate,
captures PDT/PDR/CH0..CH3 at fixed board-clock offsets from 0 to 500 ms, then
uploads one packet.  USB arrival time is deliberately excluded from all
response metrics.  Every command and every program exit leaves SOA shuttered.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import struct
import time
import zlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import serial

import app_JDSU as app
from benchmark_stress_realtime import _close_shutter, _safe_disarm
from probe_pdr_pdt_ch1_stability_live import CHANNELS, DEFAULT_SESSION, choose_peak


COMMAND_SIZE = app.tx_size
RESPONSE_MAGIC = b"\xD4\x4D"
RESPONSE_TRAILER = b"\x4D\xD4"
RESPONSE_HEADER = 48
RESPONSE_RECORD = 20
REJECT_SIZE = 20
DIRECTION_OFF = 0
DIRECTION_ON = 1
WIRE_CHANNELS = ("CH0", "CH1", "CH2", "CH3", "PDT", "PDR")


def make_command(direction: int, tag: int) -> bytes:
    if direction not in (DIRECTION_OFF, DIRECTION_ON):
        raise ValueError("invalid SOA response direction")
    if not 0 < tag <= 0xFFFFFFFF:
        raise ValueError("tag must be a non-zero uint32")
    command = bytearray(COMMAND_SIZE)
    command[:4] = b"\xFF\xFF\x03\x0B"
    command[4] = direction
    command[6:10] = tag.to_bytes(4, "big")
    command[10:14] = b"SOA1"
    return bytes(command)


def _extract_response(buffer: bytearray) -> bytes | None:
    while True:
        start = buffer.find(RESPONSE_MAGIC)
        if start < 0:
            buffer[:] = buffer[-1:] if buffer[-1:] == RESPONSE_MAGIC[:1] else b""
            return None
        if start:
            del buffer[:start]
        if len(buffer) < REJECT_SIZE:
            return None
        if buffer[3] & 0x80:
            if bytes(buffer[18:20]) != RESPONSE_TRAILER:
                del buffer[0]
                continue
            packet = bytes(buffer[:REJECT_SIZE])
            del buffer[:REJECT_SIZE]
            return packet
        if len(buffer) < 21:
            return None
        sample_count = int(buffer[20])
        length = RESPONSE_HEADER + sample_count * RESPONSE_RECORD + 6
        if len(buffer) < length:
            return None
        if bytes(buffer[length - 2:length]) != RESPONSE_TRAILER:
            del buffer[0]
            continue
        packet = bytes(buffer[:length])
        del buffer[:length]
        return packet


def parse_response(packet: bytes, expected_tag: int, expected_direction: int) -> dict:
    if packet[:2] != RESPONSE_MAGIC:
        raise ValueError("SOA响应帧头错误")
    if packet[3] & 0x80:
        tag = int.from_bytes(packet[6:10], "big")
        reason = int.from_bytes(packet[10:12], "big")
        raise RuntimeError(
            f"板上SOA采样请求被拒绝：tag=0x{tag:08X}, reason=0x{reason:04X}"
        )
    sample_count = int(packet[20])
    expected_length = RESPONSE_HEADER + sample_count * RESPONSE_RECORD + 6
    if len(packet) != expected_length or packet[-2:] != RESPONSE_TRAILER:
        raise ValueError("板上SOA响应长度或帧尾错误")
    stored_crc = int.from_bytes(packet[-6:-2], "big")
    computed_crc = zlib.crc32(packet[:-6]) & 0xFFFFFFFF
    if stored_crc != computed_crc:
        raise ValueError(
            f"板上SOA响应CRC错误：0x{stored_crc:08X} != 0x{computed_crc:08X}"
        )
    tag = int.from_bytes(packet[8:12], "big")
    direction = int(packet[4])
    if tag != expected_tag or direction != expected_direction:
        raise ValueError("板上SOA响应的tag或方向不匹配")
    records = []
    offset = RESPONSE_HEADER
    for _ in range(sample_count):
        start_us, end_us, *values = struct.unpack_from(">II6H", packet, offset)
        records.append({
            "start_us": int(start_us),
            "end_us": int(end_us),
            "mid_us": 0.5 * (int(start_us) + int(end_us)),
            **{name.lower(): int(value) for name, value in zip(WIRE_CHANNELS, values)},
        })
        offset += RESPONSE_RECORD
    baseline = struct.unpack_from(">6H", packet, 30)
    result = {
        "protocol_version": int(packet[2]),
        "status": int(packet[3]),
        "direction": direction,
        "initial_mode": int(packet[5]),
        "transition_mode": int(packet[6]),
        "final_mode": int(packet[7]),
        "tag": tag,
        "firmware_version": int.from_bytes(packet[12:16], "big"),
        "switch_call_us": int.from_bytes(packet[16:20], "big"),
        "sample_count": sample_count,
        "channel_count": int(packet[21]),
        "feedback_selectors": [int(packet[22]), int(packet[23])],
        "adc_spi_error_delta": int.from_bytes(packet[24:28], "big"),
        "i2c_error_delta": int.from_bytes(packet[28:30], "big"),
        "baseline": {
            name.lower(): int(value) for name, value in zip(WIRE_CHANNELS, baseline)
        },
        "samples": records,
    }
    if result["status"]:
        raise RuntimeError(f"板上SOA采样失败：status=0x{result['status']:02X}")
    if sample_count != 43 or result["channel_count"] != 6:
        raise RuntimeError("板上SOA采样数量不完整")
    if result["final_mode"] != app.PI11210_SOA_SHUTTER_MODE:
        raise RuntimeError("板上SOA采样结束后未保持关光")
    return result


def capture(device, direction: int, tag: int, timeout_s: float = 2.0) -> dict:
    device.reset_input_buffer()
    command = make_command(direction, tag)
    device.write(command)
    device.flush()
    buffer = bytearray()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        waiting = int(getattr(device, "in_waiting", 0) or 0)
        data = device.read(min(max(waiting, REJECT_SIZE), 4096))
        if data:
            buffer.extend(data)
            packet = _extract_response(buffer)
            if packet is not None:
                return parse_response(packet, tag, direction)
        else:
            time.sleep(0.0005)
    raise TimeoutError("等待板上SOA高速采样结果超时")


def transition_metrics(record: dict) -> dict:
    result = {}
    times_us = np.asarray([sample["mid_us"] for sample in record["samples"]], dtype=float)
    for name in ("pdt", "pdr", "ch1"):
        initial = float(record["baseline"][name])
        values = np.asarray([sample[name] for sample in record["samples"]], dtype=float)
        final = float(np.median(values[-5:]))
        step = final - initial
        span = abs(step)
        if span < 8.0:
            t10_us = t90_us = settle_us = None
        else:
            progress = (values - initial) / step
            t10_us = next((float(t) for t, p in zip(times_us, progress) if p >= 0.10), None)
            t90_us = next((float(t) for t, p in zip(times_us, progress) if p >= 0.90), None)
            tolerance = max(4.0, 0.02 * span)
            settled = np.abs(values - final) <= tolerance
            settle_us = None
            for index in range(len(settled) - 2):
                if bool(np.all(settled[index:index + 3])):
                    settle_us = float(times_us[index])
                    break
        result[name] = {
            "initial_code": initial,
            "final_tail_median_code": final,
            "step_codes": float(step),
            "t10_us": t10_us,
            "t90_us": t90_us,
            "t10_to_t90_us": None if t10_us is None or t90_us is None else t90_us - t10_us,
            "settle_us": settle_us,
        }
    durations = [sample["end_us"] - sample["start_us"] for sample in record["samples"]]
    result["adc_read_duration_median_us"] = float(statistics.median(durations))
    result["first_sample_mid_us"] = float(times_us[0])
    return result


def summarize(records: list[dict]) -> dict:
    summary = {}
    for direction, label in ((DIRECTION_OFF, "off"), (DIRECTION_ON, "on")):
        selected = [record for record in records if record["direction"] == direction]
        summary[label] = {
            "repeats": len(selected),
            "switch_call_median_us": statistics.median(r["switch_call_us"] for r in selected),
            "adc_read_duration_median_us": statistics.median(
                r["metrics"]["adc_read_duration_median_us"] for r in selected
            ),
            "channels": {},
        }
        for channel in ("pdt", "pdr", "ch1"):
            durations = [r["metrics"][channel]["t10_to_t90_us"] for r in selected]
            settles = [r["metrics"][channel]["settle_us"] for r in selected]
            durations = [value for value in durations if value is not None]
            settles = [value for value in settles if value is not None]
            summary[label]["channels"][channel] = {
                "t10_to_t90_median_us": None if not durations else statistics.median(durations),
                "t10_to_t90_p90_us": None if not durations else float(np.percentile(durations, 90)),
                "settle_median_us": None if not settles else statistics.median(settles),
                "settled_repeats": len(settles),
            }
    return summary


def plot_report(report: dict, output: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(11.5, 8.0), layout="constrained")
    for axis, direction, title in zip(
        axes,
        (DIRECTION_OFF, DIRECTION_ON),
        ("SOA source → shutter", "SOA shutter → source"),
    ):
        selected = [record for record in report["records"] if record["direction"] == direction]
        for record_index, record in enumerate(selected):
            time_ms = np.asarray([sample["mid_us"] for sample in record["samples"]]) / 1000.0
            for name, color in zip(("pdt", "pdr", "ch1"), ("#3178c6", "#2b9a66", "#d74b57")):
                values = [sample[name] for sample in record["samples"]]
                axis.plot(time_ms, values, color=color, alpha=0.16, linewidth=0.8)
        for name, color in zip(("pdt", "pdr", "ch1"), ("#3178c6", "#2b9a66", "#d74b57")):
            arrays = np.asarray([[sample[name] for sample in record["samples"]] for record in selected])
            time_ms = np.asarray([sample["mid_us"] for sample in selected[0]["samples"]]) / 1000.0
            axis.plot(time_ms, np.median(arrays, axis=0), color=color, linewidth=2.0, label=name.upper())
        axis.set_xscale("symlog", linthresh=0.1)
        axis.set(title=f"{title}: board-timestamped ADC", xlabel="Time after gate command (ms)", ylabel="ADC code")
        axis.grid(alpha=0.2)
        axis.legend()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--group", type=int, choices=range(1, 10))
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--source-settle-s", type=float, default=0.30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 3 <= args.repeats <= 30:
        raise ValueError("repeats must be 3..30")
    if not 0.1 <= args.source_settle_s <= 2.0:
        raise ValueError("source settle must be 0.1..2.0 s")
    if args.output.exists() or args.output.with_suffix(".png").exists():
        raise FileExistsError(args.output)

    choice = choose_peak(args.session, args.group)
    report = {
        "schema": "soa_on_off_board_capture_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": False,
        "scope": "same DAC codes; only PI11210 SOA hardware gate changes",
        "timing_scope": "DWT board timestamps; USB transfer occurs after capture",
        "selection": choice,
        "requested_repeats": args.repeats,
        "records": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = None
    disarmed = shuttered = False
    try:
        device = serial.Serial(args.port, 2_000_000, timeout=0.005, write_timeout=3)
        device.dtr = True
        if not _safe_disarm(device) or not _close_shutter(device):
            raise RuntimeError("测试前停扫/关光回读未确认")
        worker = app.EqualIntervalWorker(
            device, (), {}, settle_s=0.0, feedback_selectors=(0, 2)
        )
        if not worker._feedback_command_and_ack():
            raise RuntimeError("CH1 5 kΩ跨阻状态回读未确认")
        for repeat in range(1, args.repeats + 1):
            if worker._command_and_ack(
                choice["target"]["codes"], soa_mode=app.PI11210_SOA_SOURCE_MODE
            ) is None:
                raise RuntimeError("峰值点出光准备失败")
            if not worker._interruptible_wait(args.source_settle_s):
                raise RuntimeError("出光稳定等待被中断")
            base_tag = int(time.time_ns() & 0x7FFFFFFF) | 1
            off = capture(device, DIRECTION_OFF, base_tag)
            off["repeat"] = repeat
            off["metrics"] = transition_metrics(off)
            report["records"].append(off)
            time.sleep(0.05)
            on = capture(device, DIRECTION_ON, base_tag + 1)
            on["repeat"] = repeat
            on["metrics"] = transition_metrics(on)
            report["records"].append(on)
            print(
                f"repeat={repeat}/{args.repeats} "
                f"off_adc={off['metrics']['adc_read_duration_median_us']:.1f}us "
                f"on_adc={on['metrics']['adc_read_duration_median_us']:.1f}us",
                flush=True,
            )
        report["analysis"] = summarize(report["records"])
        report["complete"] = True
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        errors = []
        if device is not None:
            try:
                disarmed = bool(_safe_disarm(device))
            except Exception as exc:
                errors.append(f"disarm: {exc}")
            try:
                shuttered = bool(_close_shutter(device))
            except Exception as exc:
                errors.append(f"shutter: {exc}")
            device.close()
        report["safety_cleanup"] = {
            "exact_disarm_ack": disarmed,
            "exact_soa_shutter_ack": shuttered,
            "errors": errors,
        }
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not disarmed or not shuttered:
        raise RuntimeError("测试结束后的停扫/关光未确认")
    plot_report(report, args.output.with_suffix(".png"))
    print(json.dumps({
        "selection": {key: choice[key] for key in ("group", "target", "target_reference_adc")},
        "analysis": report["analysis"],
        "output": str(args.output.resolve()),
        "plot": str(args.output.with_suffix(".png").resolve()),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
