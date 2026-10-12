"""Diagnose a suspicious stress-table wavelength against equal/stable paths.

The script never changes a saved wavelength table.  It uses the installed
CH0/CH1 selector protocol, limits itself to a narrow wavelength neighbourhood,
and confirms the PI11210 SOA shutter before releasing the serial port.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics
import time

import numpy as np
import serial

from app_JDSU import (
    ACK_VALUE,
    PD_ADC_CODE_COUNT,
    PD_ADC_REFERENCE_V,
    PD_FEEDBACK_KOHM_BY_SELECTOR,
    PI11210_SOA_SHUTTER_MODE,
    SINGLE_VALUE_FRAME_SIZE,
    build_extra_feedback_command,
    build_single_value_dac_command,
    build_stress_feedback_command,
    build_work_mode_command,
    decode_extra_feedback_status,
    decode_single_value_monitor_frame,
    decode_stress_feedback_status,
    extract_single_value_frames,
    load_fullband_accuracy_table,
)


HERE = Path(__file__).resolve().parent


def _read_frames(device: serial.Serial, buffer: bytearray, deadline: float):
    while time.monotonic() < deadline:
        waiting = int(device.in_waiting)
        data = device.read(min(max(waiting, SINGLE_VALUE_FRAME_SIZE), 4096))
        if not data:
            time.sleep(0.001)
            continue
        buffer.extend(data)
        yield from extract_single_value_frames(buffer)


def _set_extra_feedback(device: serial.Serial, selectors: tuple[int, int]) -> None:
    device.reset_input_buffer()
    device.write(build_extra_feedback_command(*selectors))
    device.flush()
    buffer = bytearray()
    for frame in _read_frames(device, buffer, time.monotonic() + 1.5):
        result = decode_extra_feedback_status(frame)
        if result is not None and result == (selectors, ACK_VALUE):
            return
    raise RuntimeError("等间隔/单值跨阻IO回读失败")


def _set_stress_feedback(device: serial.Serial, selectors: tuple[int, int]) -> None:
    device.reset_input_buffer()
    device.write(build_stress_feedback_command(*selectors))
    device.flush()
    buffer = bytearray()
    for frame in _read_frames(device, buffer, time.monotonic() + 1.5):
        result = decode_stress_feedback_status(frame)
        if result is not None and result == (selectors, ACK_VALUE):
            return
    raise RuntimeError("应力跨阻IO回读失败")


def _write_codes_and_wait_ack(
    device: serial.Serial,
    codes,
    soa_mode: int = 0,
) -> bytes:
    command = build_single_value_dac_command(codes, soa_mode=soa_mode)
    expected = tuple(
        (command[4 + 2 * channel] << 8) | command[5 + 2 * channel]
        for channel in range(5)
    )
    device.reset_input_buffer()
    device.write(command)
    device.flush()
    buffer = bytearray()
    for frame in _read_frames(device, buffer, time.monotonic() + 2.0):
        if len(frame) != SINGLE_VALUE_FRAME_SIZE or frame[:4] != b"\xFF\xFF\x02\x00":
            continue
        returned = tuple(
            (frame[4 + 2 * channel] << 8) | frame[5 + 2 * channel]
            for channel in range(5)
        )
        if returned == expected:
            return frame
    raise RuntimeError("DAC写入后未收到匹配回读")


def _first_monitor_sample(device: serial.Serial, timeout_s: float = 1.0):
    device.reset_input_buffer()
    buffer = bytearray()
    for frame in _read_frames(device, buffer, time.monotonic() + timeout_s):
        values = decode_single_value_monitor_frame(frame)
        if values is not None:
            return tuple(int(value) for value in values[2:])
    raise RuntimeError("未收到ADC监测帧")


def _monitor_samples(device: serial.Serial, duration_s: float):
    device.reset_input_buffer()
    buffer = bytearray()
    readings = []
    for frame in _read_frames(device, buffer, time.monotonic() + duration_s):
        values = decode_single_value_monitor_frame(frame)
        if values is not None:
            readings.append(tuple(int(value) for value in values[2:]))
    if not readings:
        raise RuntimeError("稳定单值阶段未收到ADC监测帧")
    return readings


def _extract_scan_frames(buffer: bytearray):
    frames = []
    while True:
        start = buffer.find(b"\xEE\xEE")
        if start < 0:
            if len(buffer) > 1:
                del buffer[:-1]
            break
        if start:
            del buffer[:start]
        end = buffer.find(b"\xFF\xEF", 2)
        if end < 0:
            break
        frames.append(bytes(buffer[: end + 2]))
        del buffer[: end + 2]
    return frames


def _collect_stress(device: serial.Serial, point_count: int, duration_s: float):
    device.reset_input_buffer()
    buffer = bytearray()
    spectra = []
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        data = device.read(4096)
        if not data:
            continue
        buffer.extend(data)
        for frame in _extract_scan_frames(buffer):
            if int.from_bytes(frame[2:4], "big") != point_count:
                continue
            data_end = 4 + point_count * 8
            if len(frame) <= data_end or frame[data_end] != 0xAB:
                continue
            values = np.frombuffer(
                frame[4:data_end], dtype=">u2"
            ).reshape(point_count, 4)
            spectra.append(values.astype(float).T)
    return spectra


def _volts(code: float) -> float:
    return float(code) * PD_ADC_REFERENCE_V / PD_ADC_CODE_COUNT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--baud", type=int, default=2_000_000)
    parser.add_argument("--target", type=float, default=1538.620)
    parser.add_argument("--equal-settle", type=float, default=0.005)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--stress-seconds", type=float, default=6.0)
    parser.add_argument("--selector", type=int, choices=range(4), default=1)
    args = parser.parse_args()

    fullband = load_fullband_accuracy_table(HERE)
    target = min(fullband, key=lambda point: abs(point.measured_nm - args.target))
    if abs(target.measured_nm - args.target) > 0.011:
        raise RuntimeError("2001点表中没有足够接近的目标点")

    installed = json.loads(
        (HERE / "mode_tables_from_fullband_2001.json").read_text(encoding="utf-8")
    )
    stress_rows = installed["modes"]["stress"]["rows"]
    stress_index = min(
        range(len(stress_rows)),
        key=lambda index: abs(
            float(stress_rows[index]["measured_wavelength_nm"]) - args.target
        ),
    )
    if abs(float(stress_rows[stress_index]["measured_wavelength_nm"]) - args.target) > 0.011:
        raise RuntimeError("应力表没有目标点")
    stress_predecessor = tuple(stress_rows[(stress_index - 1) % len(stress_rows)]["codes"])

    neighbourhood = tuple(
        point for point in fullband
        if args.target - 0.14 <= point.measured_nm <= args.target + 0.14
    )
    target_position = next(
        index for index, point in enumerate(neighbourhood) if point.index == target.index
    )
    selectors = (args.selector, args.selector)
    equal_runs = []
    stable_cases = {
        "fullband_predecessor": [],
        "stress_sparse_predecessor": [],
        "reverse_neighbour": [],
    }
    stress_spectra = []
    stress_elapsed_s = 0.0
    shutter_confirmed = False

    with serial.Serial(
        args.port, args.baud, timeout=0.02, write_timeout=1.0
    ) as device:
        device.dtr = True
        device.write(build_work_mode_command(2))
        device.flush()
        time.sleep(0.15)
        _set_extra_feedback(device, selectors)
        try:
            # Reproduce the equal-interval page: exact DAC ACK, user delay,
            # discard transient USB data, then use the first fresh ADC frame.
            for _repeat in range(args.repeats):
                run = []
                for point in neighbourhood:
                    _write_codes_and_wait_ack(device, point.codes)
                    time.sleep(args.equal_settle)
                    run.append(_first_monitor_sample(device))
                equal_runs.append(run)

            stable_paths = (
                ("fullband_predecessor", fullband[target.index - 1].codes),
                ("stress_sparse_predecessor", stress_predecessor),
                ("reverse_neighbour", fullband[target.index + 1].codes),
            )
            for name, predecessor_codes in stable_paths:
                for _repeat in range(args.repeats):
                    _write_codes_and_wait_ack(device, predecessor_codes)
                    time.sleep(0.20)
                    _write_codes_and_wait_ack(device, target.codes)
                    time.sleep(0.30)
                    readings = np.asarray(_monitor_samples(device, 0.12), dtype=float)
                    stable_cases[name].append(np.median(readings, axis=0).tolist())

            # Re-enter EXTRA to arm the fixed stress selector, then collect the
            # real TABLE-mode RC estimator and sparse predecessor sequence.
            device.write(build_work_mode_command(2))
            device.flush()
            time.sleep(0.15)
            _set_stress_feedback(device, selectors)
            device.write(build_work_mode_command(0))
            device.flush()
            stress_started = time.monotonic()
            stress_spectra = _collect_stress(
                device, len(stress_rows), args.stress_seconds
            )
            stress_elapsed_s = time.monotonic() - stress_started
        finally:
            device.write(build_work_mode_command(2))
            device.flush()
            time.sleep(0.12)
            frame = _write_codes_and_wait_ack(
                device,
                target.codes,
                soa_mode=PI11210_SOA_SHUTTER_MODE,
            )
            shutter_confirmed = (
                frame[14] == PI11210_SOA_SHUTTER_MODE
                and bool(frame[15] & 0x04)
            )

    if len(stress_spectra) < 5:
        raise RuntimeError(f"只收到{len(stress_spectra)}帧应力光谱")

    equal = np.asarray(equal_runs, dtype=float)
    # Exclude discovery/start-up frames and take the frame-wise median.
    stress_stack = np.stack(stress_spectra[3:], axis=0)
    stress_codes = np.median(stress_stack, axis=0)
    equal_target = equal[:, target_position, 0]
    stress_target = stress_stack[:, 0, stress_index]
    stable_summary = {}
    for name, readings in stable_cases.items():
        values = np.asarray(readings, dtype=float)[:, 0]
        stable_summary[name] = {
            "runs_ch0_v": [_volts(value) for value in values],
            "median_ch0_v": _volts(np.median(values)),
            "span_ch0_v": _volts(np.ptp(values)),
        }

    local_indices = range(max(0, stress_index - 3), min(len(stress_rows), stress_index + 4))
    stress_local = [
        {
            "mode_index": index,
            "wavelength_nm": float(stress_rows[index]["measured_wavelength_nm"]),
            "median_ch0_v": _volts(stress_codes[0, index]),
        }
        for index in local_indices
    ]
    equal_local = [
        {
            "fullband_index": point.index,
            "wavelength_nm": point.measured_nm,
            "median_ch0_v": _volts(np.median(equal[:, index, 0])),
        }
        for index, point in enumerate(neighbourhood)
    ]

    settled_reference = stable_summary["fullband_predecessor"]["median_ch0_v"]
    stress_median_v = _volts(np.median(stress_target))
    equal_median_v = _volts(np.median(equal_target))
    report = {
        "schema": "stress_point_path_diagnostic_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "target": {
            "requested_nm": args.target,
            "fullband_index": target.index,
            "measured_nm": target.measured_nm,
            "codes": list(target.codes),
            "stress_mode_index": stress_index,
        },
        "settings": {
            "feedback_selectors": list(selectors),
            "feedback_kohm": [
                PD_FEEDBACK_KOHM_BY_SELECTOR[args.selector],
                PD_FEEDBACK_KOHM_BY_SELECTOR[args.selector],
            ],
            "equal_settle_s": args.equal_settle,
            "repeats": args.repeats,
        },
        "equal_interval": {
            "target_runs_ch0_v": [_volts(value) for value in equal_target],
            "target_median_ch0_v": equal_median_v,
            "target_span_ch0_v": _volts(np.ptp(equal_target)),
            "local_spectrum": equal_local,
        },
        "settled_single": stable_summary,
        "stress": {
            "frame_count": len(stress_spectra),
            "used_frame_count": len(stress_spectra[3:]),
            "collection_elapsed_s": stress_elapsed_s,
            "measured_frame_rate_hz": (
                len(stress_spectra) / stress_elapsed_s
                if stress_elapsed_s > 0.0
                else 0.0
            ),
            "target_runs_ch0_v": [_volts(value) for value in stress_target],
            "target_median_ch0_v": stress_median_v,
            "target_span_ch0_v": _volts(np.ptp(stress_target)),
            "local_spectrum": stress_local,
        },
        "comparison": {
            "equal_minus_settled_v": equal_median_v - settled_reference,
            "stress_minus_settled_v": stress_median_v - settled_reference,
            "stress_minus_equal_v": stress_median_v - equal_median_v,
            "stable_path_spread_v": max(
                item["median_ch0_v"] for item in stable_summary.values()
            ) - min(item["median_ch0_v"] for item in stable_summary.values()),
        },
        "safety": {"soa_shutter_confirmed": shutter_confirmed},
    }
    output = HERE / "outputs" / (
        f"stress_1538_620_path_{PD_FEEDBACK_KOHM_BY_SELECTOR[args.selector]}k_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"REPORT={output}", flush=True)
    if not shutter_confirmed:
        raise RuntimeError("SOA关光状态未确认")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
