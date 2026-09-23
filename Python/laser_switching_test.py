"""Measure raw laser wavelength-switching transients with the STM32 ADC.

The matching firmware diagnostic command returns timestamped CH0..CH3, PDT and
PDR codes.  Normal peak-mode steady-state reconstruction is deliberately not
used, so this file is suitable for identifying the real optical/electrical step
response.  The calibrated DAC table is read but never modified.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import serial
import yaml

from laser_dac_safety import validate_pi11210_codes


FRAME_SIZE = 808
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent / "JDSU"
DAC_SOURCE = PROJECT / "Core" / "Src" / "dac_const.c"
WAVE_YAML = HERE / "wave_const.yaml"
DEFAULT_REPORT = HERE / "laser_switching_test_report.json"
CHANNEL_NAMES = ("CH0", "CH1", "CH2", "CH3", "PDT", "PDR")


def parse_dac_rows(source: Path) -> list[list[int]]:
    import re

    text = source.read_text(encoding="utf-8")
    rows: list[list[int]] = []
    for match in re.finditer(r"\{\s*([^{}]+?)\s*\},?", text):
        fields = [part.strip() for part in match.group(1).split(",")]
        try:
            values = [int(field, 0) for field in fields]
        except ValueError:
            continue
        if len(values) != 5 or values[:3] == [0xFFFF, 0xFFFF, 0xFFFF]:
            continue
        rows.append(values)
    return rows


def load_wavelengths(source: Path) -> list[float]:
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    return [float(integer) + float(decimal) / 1000.0 for integer, decimal in payload["Wave_DATA"]]


def read_u16(data: bytes, position: int) -> int:
    return (data[position] << 8) | data[position + 1]


def read_u32(data: bytes, position: int) -> int:
    return (
        (data[position] << 24)
        | (data[position + 1] << 16)
        | (data[position + 2] << 8)
        | data[position + 3]
    )


@dataclass
class Capture:
    source_index: int
    target_index: int
    full_table: bool
    dac_write_us: int
    baseline: list[int]
    times_us: list[int]
    values: list[list[int]]


class SwitchTester:
    def __init__(self, port: str):
        self.serial = serial.Serial(port, 115200, timeout=0.03, write_timeout=1.0)
        self._rx = bytearray()
        self.set_mode(2)

    def close(self) -> None:
        self.serial.close()

    def _write(self, frame: bytearray) -> None:
        self.serial.write(frame)
        self.serial.flush()

    def set_mode(self, mode: int) -> None:
        frame = bytearray(FRAME_SIZE)
        frame[0:4] = bytes((0xFF, 0xFF, 0x01, 0x02))
        frame[8] = mode
        self._write(frame)
        time.sleep(0.08)
        self.serial.reset_input_buffer()
        self._rx.clear()

    def park(self, codes: list[int]) -> None:
        safe_codes = validate_pi11210_codes(codes)
        frame = bytearray(FRAME_SIZE)
        frame[0:4] = bytes((0xFF, 0xFF, 0x00, 0x01))
        for channel, value in enumerate(safe_codes):
            frame[4 + 2 * channel] = (value >> 8) & 0xFF
            frame[5 + 2 * channel] = value & 0xFF
        self._write(frame)
        time.sleep(0.10)

    def capture(self, source_index: int, target_index: int, full_table: bool) -> Capture:
        frame = bytearray(FRAME_SIZE)
        frame[0:4] = bytes((0xFF, 0xFF, 0x03, 0x01))
        frame[4] = (source_index >> 8) & 0xFF
        frame[5] = source_index & 0xFF
        frame[6] = (target_index >> 8) & 0xFF
        frame[7] = target_index & 0xFF
        frame[8] = int(full_table)
        self.serial.reset_input_buffer()
        self._rx.clear()
        self._write(frame)

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            waiting = self.serial.in_waiting
            self._rx.extend(self.serial.read(waiting or 1))
            start = self._rx.find(b"\xD5\x5D\x01")
            if start < 0 or len(self._rx) < start + 14:
                continue
            count = self._rx[start + 12]
            channels = self._rx[start + 13]
            expected = 14 + channels * 2 + count * (4 + channels * 2) + 2
            if len(self._rx) < start + expected:
                continue
            packet = bytes(self._rx[start : start + expected])
            if packet[-2:] != b"\x5D\xD5":
                del self._rx[: start + 2]
                continue
            return self._parse(packet)
        raise TimeoutError("STM32 switching-test response timed out")

    @staticmethod
    def _parse(packet: bytes) -> Capture:
        full_table = bool(packet[3] & 1)
        source_index = read_u16(packet, 4)
        target_index = read_u16(packet, 6)
        dac_write_us = read_u32(packet, 8)
        count = packet[12]
        channels = packet[13]
        position = 14
        baseline = [read_u16(packet, position + 2 * channel) for channel in range(channels)]
        position += 2 * channels
        times: list[int] = []
        values: list[list[int]] = []
        for _ in range(count):
            times.append(read_u32(packet, position))
            position += 4
            values.append([read_u16(packet, position + 2 * channel) for channel in range(channels)])
            position += 2 * channels
        return Capture(
            source_index=source_index,
            target_index=target_index,
            full_table=full_table,
            dac_write_us=dac_write_us,
            baseline=baseline,
            times_us=times,
            values=values,
        )


def median_trace(captures: list[Capture], channel: int) -> tuple[list[float], list[float], float]:
    times = [statistics.median(capture.times_us[index] for capture in captures) for index in range(len(captures[0].times_us))]
    values = [statistics.median(capture.values[index][channel] for capture in captures) for index in range(len(captures[0].values))]
    baseline = statistics.median(capture.baseline[channel] for capture in captures)
    return times, values, baseline


def pd_ratio(capture: Capture, sample_index: int | None) -> float | None:
    values = capture.baseline if sample_index is None else capture.values[sample_index]
    pdt = 2048.0 - values[4]
    pdr = 2048.0 - values[5]
    if pdt <= 1.0:
        return None
    return pdr / pdt


def ratio_trace(captures: list[Capture]) -> tuple[list[float], list[float], float]:
    times = [statistics.median(capture.times_us[index] for capture in captures) for index in range(len(captures[0].times_us))]
    baseline_values = [value for capture in captures if (value := pd_ratio(capture, None)) is not None]
    baseline = statistics.median(baseline_values)
    ratios: list[float] = []
    for index in range(len(captures[0].values)):
        values = [value for capture in captures if (value := pd_ratio(capture, index)) is not None]
        ratios.append(statistics.median(values))
    return times, ratios, baseline


def crossing_time(times: list[float], progress: list[float], threshold: float) -> float | None:
    for index in range(1, len(times)):
        before = progress[index - 1]
        after = progress[index]
        if (before <= threshold <= after) or (before >= threshold >= after):
            delta = after - before
            if abs(delta) < 1e-12:
                return times[index]
            fraction = (threshold - before) / delta
            return times[index - 1] + fraction * (times[index] - times[index - 1])
    return None


def summarize_trace(times: list[float], values: list[float], baseline: float) -> dict:
    tail = statistics.median(values[-3:])
    amplitude = tail - baseline
    if abs(amplitude) < 1e-9:
        return {"baseline": baseline, "final": tail, "amplitude": amplitude}
    progress = [(value - baseline) / amplitude for value in values]
    # The separately captured baseline is the optical state immediately before
    # the first changed DAC write.  Include it at t=0 so sub-I2C-update
    # crossings are bounded instead of being reported as missing.
    metric_times = [0.0, *times]
    metric_progress = [0.0, *progress]
    t10 = crossing_time(metric_times, metric_progress, 0.10)
    t50 = crossing_time(metric_times, metric_progress, 0.50)
    t90 = crossing_time(metric_times, metric_progress, 0.90)
    settle_5 = None
    for index, stamp in enumerate(metric_times):
        if all(abs(value - 1.0) <= 0.05 for value in metric_progress[index:]):
            settle_5 = stamp
            break
    return {
        "baseline": baseline,
        "final": tail,
        "amplitude": amplitude,
        "t10_us": t10,
        "t50_us": t50,
        "t90_us": t90,
        "rise_10_90_us": None if t10 is None or t90 is None else t90 - t10,
        "settle_5pct_us": settle_5,
        "times_us": times,
        "median_values": values,
        "normalized_progress": progress,
    }


def capture_to_dict(capture: Capture) -> dict:
    return {
        "source_index": capture.source_index,
        "target_index": capture.target_index,
        "full_table": capture.full_table,
        "dac_write_us": capture.dac_write_us,
        "baseline": dict(zip(CHANNEL_NAMES, capture.baseline)),
        "times_us": capture.times_us,
        "values": [dict(zip(CHANNEL_NAMES, values)) for values in capture.values],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--source", type=int, default=89)
    parser.add_argument("--target", type=int, default=90)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    rows = parse_dac_rows(DAC_SOURCE)
    wavelengths = load_wavelengths(WAVE_YAML)
    if not (0 <= args.source < len(rows) and 0 <= args.target < len(rows)):
        raise SystemExit("Source or target table index is out of range")
    if not 1 <= args.repeats <= 200:
        raise SystemExit("Repeats must be between 1 and 200")

    groups: dict[str, list[Capture]] = {}
    tester = SwitchTester(args.port)
    try:
        for full_table in (False, True):
            mode_name = "full_table" if full_table else "wavelength_only"
            for source, target, direction in (
                (args.source, args.target, "forward"),
                (args.target, args.source, "reverse"),
            ):
                name = f"{mode_name}_{direction}"
                groups[name] = []
                for repeat in range(args.repeats):
                    capture = tester.capture(source, target, full_table)
                    groups[name].append(capture)
                    print(
                        f"{name} {repeat + 1:03d}/{args.repeats:03d} "
                        f"DAC={capture.dac_write_us} us",
                        flush=True,
                    )
    finally:
        tester.park(rows[44] if len(rows) > 44 else rows[0])
        tester.close()

    summary: dict[str, dict] = {}
    for name, captures in groups.items():
        metrics: dict[str, dict] = {
            "dac_write_us": {
                "median": statistics.median(capture.dac_write_us for capture in captures),
                "min": min(capture.dac_write_us for capture in captures),
                "max": max(capture.dac_write_us for capture in captures),
            }
        }
        for channel, channel_name in enumerate(CHANNEL_NAMES):
            times, values, baseline = median_trace(captures, channel)
            metrics[channel_name] = summarize_trace(times, values, baseline)
        times, values, baseline = ratio_trace(captures)
        metrics["PDR_PDT_ratio"] = summarize_trace(times, values, baseline)
        summary[name] = metrics

    report = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "source_index": args.source,
        "source_wavelength_nm": wavelengths[args.source],
        "target_index": args.target,
        "target_wavelength_nm": wavelengths[args.target],
        "repeats": args.repeats,
        "channels": list(CHANNEL_NAMES),
        "summary": summary,
        "captures": {
            name: [capture_to_dict(capture) for capture in captures]
            for name, captures in groups.items()
        },
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {args.report}", flush=True)
    for name, metrics in summary.items():
        print(f"\n{name}: DAC median {metrics['dac_write_us']['median']} us")
        for channel_name in ("CH1", "CH2", "CH3", "PDR_PDT_ratio"):
            item = metrics[channel_name]
            print(
                f"  {channel_name}: amplitude={item.get('amplitude', 0):+.6g} "
                f"t10-90={item.get('rise_10_90_us')} us "
                f"settle5={item.get('settle_5pct_us')} us"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
