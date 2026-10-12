"""Compare sparse temperature jumps with full-band predecessor conditioning.

The laser is always shuttered in ``finally``.  This tool is intentionally
limited to the two user-reported wavelengths and never changes a saved table.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import statistics
import time

import serial

from app_JDSU import (
    PI11210_SOA_SHUTTER_MODE,
    build_extra_feedback_command,
    build_single_value_dac_command,
    build_work_mode_command,
    decode_single_value_monitor_frame,
    extract_single_value_frames,
    load_fullband_accuracy_table,
)


HERE = Path(__file__).resolve().parent
ADC_TO_V = 2.5 / 4095.0


def collect_ch0(device: serial.Serial, seconds: float = 0.24) -> list[int]:
    buffer = bytearray()
    values: list[int] = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        waiting = int(device.in_waiting)
        if waiting:
            buffer.extend(device.read(waiting))
            for frame in extract_single_value_frames(buffer):
                decoded = decode_single_value_monitor_frame(frame)
                if decoded is not None:
                    values.append(int(decoded[2]))
        else:
            time.sleep(0.001)
    return values


def write_point(device: serial.Serial, point, hold_s: float) -> None:
    device.write(build_single_value_dac_command(point.codes))
    device.flush()
    time.sleep(hold_s)


def measure_case(device: serial.Serial, name: str, path, holds) -> dict:
    device.reset_input_buffer()
    for point, hold_s in zip(path, holds):
        write_point(device, point, hold_s)
    device.reset_input_buffer()
    samples = collect_ch0(device)
    if not samples:
        raise RuntimeError(f"{name}: 未收到ADC监测帧")
    median_code = float(statistics.median(samples))
    result = {
        "name": name,
        "path_indices": [int(point.index) for point in path],
        "path_measured_nm": [float(point.measured_nm) for point in path],
        "hold_s": [float(value) for value in holds],
        "sample_count": len(samples),
        "ch0_codes": samples,
        "ch0_median_code": median_code,
        "ch0_median_v": median_code * ADC_TO_V,
    }
    print(
        f"{name}: path={result['path_indices']} "
        f"CH0={median_code:.1f} code / {result['ch0_median_v']:.4f} V",
        flush=True,
    )
    return result


def main() -> int:
    points = load_fullband_accuracy_table(HERE)
    cases = (
        # Temperature selection skips index 171 (1528.420 nm).
        ("sparse_170_to_172", (points[170], points[172]), (0.30, 0.20)),
        # Full equal-interval and wavelength-auto single-value paths include it.
        ("adjacent_171_to_172", (points[171], points[172]), (0.30, 0.20)),
        # Re-enter from the same sparse source, but hold the missing adjacent
        # row for only 5 ms and the target for the firmware's normal 12 ms.
        (
            "adjacent_5ms_to_172",
            (points[170], points[171], points[172]),
            (0.30, 0.005, 0.012),
        ),
        # Reproduce the two paths through the following 1528.461-nm row.
        ("sparse_chain_to_173", (points[170], points[172], points[173]), (0.30, 0.20, 0.20)),
        ("adjacent_chain_to_173", (points[171], points[172], points[173]), (0.30, 0.20, 0.20)),
        (
            "adjacent_5ms_chain_to_173",
            (points[170], points[171], points[172], points[173]),
            (0.30, 0.005, 0.005, 0.012),
        ),
    )
    report = {
        "schema": "temperature_predecessor_path_diagnostic_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "feedback_selectors": [0, 0],
        "cases": [],
    }
    last = points[173]
    with serial.Serial("COM6", 115200, timeout=0.02, write_timeout=1.0) as device:
        device.dtr = True
        device.write(build_work_mode_command(2))
        device.flush()
        time.sleep(0.15)
        device.reset_input_buffer()
        device.write(build_extra_feedback_command(0, 0))
        device.flush()
        time.sleep(0.12)
        try:
            for name, path, holds in cases:
                report["cases"].append(measure_case(device, name, path, holds))
                last = path[-1]
        finally:
            device.write(
                build_single_value_dac_command(
                    last.codes, soa_mode=PI11210_SOA_SHUTTER_MODE
                )
            )
            device.flush()
            time.sleep(0.12)

    output = HERE / "outputs" / (
        "temperature_predecessor_path_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".json"
    )
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"REPORT={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
