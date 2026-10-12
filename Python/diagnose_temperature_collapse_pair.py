"""Measure the transient ADC response of the two collapsed temperature points."""

from __future__ import annotations

import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import serial

from app_JDSU import (
    PI11210_SOA_SHUTTER_MODE,
    build_extra_feedback_command,
    build_single_value_dac_command,
    build_work_mode_command,
    decode_single_value_monitor_frame,
    extract_single_value_frames,
)


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
POINTS = (
    {
        "name": "1528.399_to_1528.440",
        "from_nm": 1528.399,
        "from_codes": [58981, 56712, 2613, 3072, 16220],
        "to_nm": 1528.440,
        "to_codes": [58981, 56182, 3216, 2458, 17203],
    },
    {
        "name": "1528.440_to_1528.461",
        "from_nm": 1528.440,
        "from_codes": [58981, 56182, 3216, 2458, 17203],
        "to_nm": 1528.461,
        "to_codes": [58981, 55782, 2979, 2458, 17203],
    },
    {
        "name": "1528.461_to_1528.498",
        "from_nm": 1528.461,
        "from_codes": [58981, 55782, 2979, 2458, 17203],
        "to_nm": 1528.498,
        "to_codes": [58981, 53968, 2577, 2381, 17173],
    },
)


def collect_frames(device: serial.Serial, seconds: float) -> list[dict]:
    buffer = bytearray()
    rows = []
    start = time.monotonic()
    deadline = start + seconds
    while time.monotonic() < deadline:
        waiting = device.in_waiting
        if waiting:
            buffer.extend(device.read(waiting))
            for frame in extract_single_value_frames(buffer):
                values = decode_single_value_monitor_frame(frame)
                if values is not None:
                    rows.append({
                        "elapsed_s": time.monotonic() - start,
                        "pdt_code": int(values[0]),
                        "pdr_code": int(values[1]),
                        "adc_codes": [int(value) for value in values[2:]],
                    })
        else:
            time.sleep(0.001)
    return rows


def value_at(rows: list[dict], threshold: float) -> dict | None:
    return next((row for row in rows if float(row["elapsed_s"]) >= threshold), None)


def main() -> int:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    created = datetime.now()
    output = OUTPUTS / f"temperature_collapse_dynamic_{created:%Y%m%d_%H%M%S}.json"
    payload: dict = {
        "created": created.isoformat(timespec="seconds"),
        "feedback_selectors": [0, 0],
        "feedback_kohm": [2, 2],
        "sample_rate_note": "EXTRA monitor is produced at approximately 50 Hz",
        "transitions": [],
    }
    last_codes = list(POINTS[-1]["to_codes"])

    with serial.Serial("COM6", 115200, timeout=0.02, write_timeout=1.0) as device:
        device.dtr = True
        device.write(build_work_mode_command(2))
        device.flush()
        time.sleep(0.15)
        device.reset_input_buffer()
        device.write(build_extra_feedback_command(0, 0))
        device.flush()
        time.sleep(0.12)
        device.reset_input_buffer()
        try:
            for transition in POINTS:
                source_codes = list(transition["from_codes"])
                target_codes = list(transition["to_codes"])
                device.write(build_single_value_dac_command(source_codes))
                device.flush()
                time.sleep(1.0)
                device.reset_input_buffer()
                source_rows = collect_frames(device, 0.16)
                source_ch0 = [row["adc_codes"][0] for row in source_rows]
                source_pdt = [row["pdt_code"] for row in source_rows]
                source_pdr = [row["pdr_code"] for row in source_rows]

                device.reset_input_buffer()
                switched_at = time.monotonic()
                device.write(build_single_value_dac_command(target_codes))
                device.flush()
                rows = collect_frames(device, 1.25)
                command_overhead_s = time.monotonic() - switched_at - 1.25
                tail = [row["adc_codes"][0] for row in rows if row["elapsed_s"] >= 0.9]
                result = {
                    **transition,
                    "source_ch0_median_code": (
                        statistics.median(source_ch0) if source_ch0 else None
                    ),
                    "source_pdt_median_code": (
                        statistics.median(source_pdt) if source_pdt else None
                    ),
                    "source_pdr_median_code": (
                        statistics.median(source_pdr) if source_pdr else None
                    ),
                    "command_overhead_s": command_overhead_s,
                    "samples": rows,
                    "checkpoints": {
                        str(threshold): value_at(rows, threshold)
                        for threshold in (0.012, 0.05, 0.10, 0.20, 0.50, 1.00)
                    },
                    "target_ch0_tail_median_code": statistics.median(tail) if tail else None,
                    "target_ch0_tail_span_code": (max(tail) - min(tail)) if tail else None,
                }
                payload["transitions"].append(result)
                print(
                    f"{transition['name']}: source={result['source_ch0_median_code']} "
                    + " ".join(
                        f"t={threshold:.3f}s:{(result['checkpoints'][str(threshold)] or {'adc_codes':[None]})['adc_codes'][0]}"
                        for threshold in (0.012, 0.05, 0.10, 0.20, 0.50, 1.00)
                    )
                    + f" tail={result['target_ch0_tail_median_code']}",
                    flush=True,
                )
                output.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                last_codes = target_codes
        finally:
            device.write(build_single_value_dac_command(
                last_codes, soa_mode=PI11210_SOA_SHUTTER_MODE
            ))
            device.flush()
            time.sleep(0.12)

    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"REPORT={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
