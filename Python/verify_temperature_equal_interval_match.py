"""Compare temperature-table raw ADC against equal-interval reads of the same rows.

This is a hardware acceptance test, not a simulator.  It first reads every
installed temperature wavelength through the normal equal-interval/single-value
path, then requests several temperature frames over LAN and compares CH0/CH1
point for point.  The board is returned to local EXTRA/idle mode on exit.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
from PyQt5 import QtCore
import serial

from app_JDSU import (
    EqualIntervalWorker,
    PI11210_SOA_SHUTTER_MODE,
    build_single_value_dac_command,
    load_fullband_accuracy_table,
)
from fbg_peak_fitting import fit_fbg_segment
from fbg_lan_transport import FbgLanTransport
from fbg_remote_protocol import MODE_TEMPERATURE


ADC_TO_V = 2.5 / 4095.0


def _return_to_local_idle(port_name: str) -> None:
    command = bytearray(808)
    command[0:4] = bytes((0xFF, 0xFF, 0x01, 0x02))
    command[8] = 2
    with serial.Serial(port_name, 115200, timeout=0.2) as device:
        device.dtr = True
        time.sleep(0.25)
        device.reset_input_buffer()
        device.write(command)
        device.flush()
        time.sleep(0.25)
        device.reset_input_buffer()
        device.write(
            build_single_value_dac_command(
                (0, 0, 0, 0, 0), soa_mode=PI11210_SOA_SHUTTER_MODE
            )
        )
        device.flush()
        time.sleep(0.25)


def _equal_interval_reference(
    port_name: str,
    points,
    predecessor_points,
    settle_s: float,
    feedback_selectors: tuple[int, int],
) -> np.ndarray:
    # A full 2001-point equal-interval scan reaches every selected target from
    # its immediate audited predecessor.  Recreate that optical path explicitly
    # instead of jumping through the sparse 100-point temperature selection.
    scan_points = tuple(
        point
        for predecessor, target in zip(predecessor_points, points)
        for point in (predecessor, target)
    )
    captured = []
    errors = []
    with serial.Serial(port_name, 115200, timeout=0.2) as device:
        device.dtr = True
        time.sleep(0.35)
        worker = EqualIntervalWorker(
            device,
            scan_points,
            {},
            settle_s=settle_s,
            feedback_selectors=feedback_selectors,
        )
        worker.point_ready.connect(captured.append)
        worker.error_signal.connect(errors.append)
        worker.run()
        if errors or worker.error_message:
            raise RuntimeError(errors[-1] if errors else worker.error_message)
        if worker.completed_scans != 1 or len(captured) != len(scan_points):
            raise RuntimeError(
                f"等间隔参考不完整：{len(captured)}/{len(scan_points)}点"
            )
    target_rows = captured[1::2]
    return np.asarray(
        [[row["adc_codes"][0], row["adc_codes"][1]] for row in target_rows],
        dtype=float,
    )


def _temperature_frames(board: str, count: int, timeout_s: float):
    metadata = []
    acknowledgements = []
    transport = FbgLanTransport(
        board_host=board,
        on_metadata=metadata.append,
        on_ack=acknowledgements.append,
    )
    frames = []
    transport.start()
    try:
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline and not any(
            item.mode == MODE_TEMPERATURE for item in metadata
        ):
            time.sleep(0.05)
        temperature_metadata = next(
            (item for item in metadata if item.mode == MODE_TEMPERATURE), None
        )
        if temperature_metadata is None:
            raise RuntimeError("未收到温度表波长元数据")

        command_id = int(time.time() * 1000.0) & 0xFFFFFFFF
        transport.publish_mode(MODE_TEMPERATURE, command_id)
        deadline = time.monotonic() + timeout_s
        seen_sequences = set()
        while time.monotonic() < deadline and len(frames) < count:
            frame = transport.take_latest_telemetry()
            if (
                frame is not None
                and frame.mode == MODE_TEMPERATURE
                and frame.seq not in seen_sequences
            ):
                seen_sequences.add(frame.seq)
                frames.append(frame)
            time.sleep(0.01)
        if len(frames) < count:
            raise RuntimeError(f"温度帧不完整：{len(frames)}/{count}帧")
        if not any(
            ack.command_id == command_id and ack.status == "APPLIED"
            for ack in acknowledgements
        ):
            raise RuntimeError("未收到温度模式已应用回执")
        return temperature_metadata, frames
    finally:
        transport.stop()


def _metrics(reference_codes, temperature_codes, wavelengths_nm, peak_numbers):
    reference_v = np.asarray(reference_codes, dtype=float) * ADC_TO_V
    temperature_v = np.asarray(temperature_codes, dtype=float) * ADC_TO_V
    channel_metrics = []
    peak_metrics = []
    dropout_count = 0
    for channel in range(2):
        delta = temperature_v[:, channel] - reference_v[:, channel]
        if (
            float(np.std(reference_v[:, channel])) > 1e-9
            and float(np.std(temperature_v[:, channel])) > 1e-9
        ):
            correlation = float(np.corrcoef(
                reference_v[:, channel], temperature_v[:, channel]
            )[0, 1])
        else:
            correlation = None
        channel_metrics.append({
            "channel": channel,
            "correlation": correlation,
            "mae_v": float(np.mean(np.abs(delta))),
            "p95_abs_v": float(np.percentile(np.abs(delta), 95)),
            "max_abs_v": float(np.max(np.abs(delta))),
            "rmse_v": float(np.sqrt(np.mean(delta * delta))),
        })
        dropout_count += int(np.sum(
            (reference_v[:, channel] >= 0.20)
            & (temperature_v[:, channel] <= 0.05)
        ))

        for peak_number in sorted(set(int(value) for value in peak_numbers)):
            indices = np.flatnonzero(np.asarray(peak_numbers) == peak_number)
            ref_local = reference_v[indices, channel]
            temp_local = temperature_v[indices, channel]
            ref_index = int(indices[int(np.argmax(ref_local))])
            temp_index = int(indices[int(np.argmax(temp_local))])
            ref_fit = fit_fbg_segment(
                wavelengths_nm[indices],
                ref_local,
                min_prominence_v=0.02,
                allow_edge_peak=True,
            )
            temp_fit = fit_fbg_segment(
                wavelengths_nm[indices],
                temp_local,
                min_prominence_v=0.02,
                allow_edge_peak=True,
            )
            fitted = bool(ref_fit.valid and temp_fit.valid)
            reference_peak_nm = (
                float(ref_fit.center_nm)
                if fitted else float(wavelengths_nm[ref_index])
            )
            temperature_peak_nm = (
                float(temp_fit.center_nm)
                if fitted else float(wavelengths_nm[temp_index])
            )
            ref_range = (
                float(ref_fit.amplitude_v) if fitted else float(np.ptp(ref_local))
            )
            temp_range = (
                float(temp_fit.amplitude_v) if fitted else float(np.ptp(temp_local))
            )
            peak_metrics.append({
                "channel": channel,
                "peak_number": peak_number,
                "center_method": "gaussian_fit" if fitted else "sample_maximum",
                "reference_peak_nm": reference_peak_nm,
                "temperature_peak_nm": temperature_peak_nm,
                "peak_shift_nm": temperature_peak_nm - reference_peak_nm,
                "amplitude_ratio": (
                    temp_range / ref_range if ref_range > 0.02 else None
                ),
            })

    active_channels = [
        item for item in channel_metrics
        if np.ptp(reference_v[:, item["channel"]]) >= 0.05
    ]
    active_peaks = [
        item for item in peak_metrics
        if item["amplitude_ratio"] is not None
        and np.ptp(reference_v[
            np.flatnonzero(np.asarray(peak_numbers) == item["peak_number"]),
            item["channel"],
        ]) >= 0.05
    ]
    passed = bool(active_channels) and all(
        item["correlation"] is not None
        and item["correlation"] >= 0.95
        and item["p95_abs_v"] <= 0.18
        for item in active_channels
    ) and all(
        abs(item["peak_shift_nm"]) <= 0.04
        and 0.75 <= item["amplitude_ratio"] <= 1.25
        for item in active_peaks
    ) and dropout_count == 0
    return {
        "passed": passed,
        "dropout_count": dropout_count,
        "channels": channel_metrics,
        "peaks": peak_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", default="192.168.3.46")
    parser.add_argument("--serial", default="COM6")
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--settle", type=float, default=0.012)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    formal = json.loads(
        (root / "Python/mode_tables_from_fullband_2001.json").read_text(
            encoding="utf-8"
        )
    )
    rows = formal["modes"]["temperature"]["rows"]
    all_points = load_fullband_accuracy_table(root / "Python")
    by_index = {point.index: point for point in all_points}
    points = tuple(by_index[int(row["fullband_index"])] for row in rows)
    predecessors = tuple(
        by_index[max(0, int(row["fullband_index"]) - 1)] for row in rows
    )
    selectors = tuple(
        int(formal["analog_feedback"][f"ch{channel}_selector"])
        for channel in range(2)
    )
    print(
        f"equal_interval_reference points={len(points)} settle={args.settle:.3f}s "
        f"selectors={selectors}",
        flush=True,
    )
    reference = _equal_interval_reference(
        args.serial, points, predecessors, args.settle, selectors
    )
    print("equal_interval_reference complete", flush=True)

    # Closing COM returns ownership to LAN automatically.
    time.sleep(1.0)
    try:
        metadata, frames = _temperature_frames(args.board, args.frames, 45.0)
    finally:
        _return_to_local_idle(args.serial)
    wavelength_pm = tuple(int(value) for value in metadata.wavelength_pm)
    expected_pm = tuple(round(float(row["measured_wavelength_nm"]) * 1000) for row in rows)
    if wavelength_pm != expected_pm:
        raise RuntimeError("板卡温度波长轴与桌面正式点表不一致")

    wavelengths_nm = np.asarray(wavelength_pm, dtype=float) / 1000.0
    peak_numbers = np.asarray([int(row["peak_number"]) for row in rows])
    frame_reports = []
    for frame in frames:
        samples = np.asarray(frame.samples, dtype=float)
        if samples.shape != (len(points), 2):
            raise RuntimeError(f"温度ADC矩阵尺寸错误：{samples.shape}")
        result = _metrics(reference, samples, wavelengths_nm, peak_numbers)
        result["sequence"] = int(frame.seq)
        result["temperature_codes"] = samples.astype(int).tolist()
        frame_reports.append(result)
        active = [
            item for item in result["channels"]
            if item["correlation"] is not None
        ]
        failed_peaks = [
            item for item in result["peaks"]
            if item["amplitude_ratio"] is not None
            and (
                abs(item["peak_shift_nm"]) > 0.04
                or not 0.75 <= item["amplitude_ratio"] <= 1.25
            )
        ]
        print(
            "temperature "
            f"seq={frame.seq} pass={result['passed']} "
            + " ".join(
                f"CH{item['channel']}:r={item['correlation']:.4f},"
                f"p95={item['p95_abs_v']:.4f}V"
                for item in active
            )
            + f" dropouts={result['dropout_count']}",
            flush=True,
        )
        if failed_peaks:
            print(
                "failed_peaks="
                + ",".join(
                    f"CH{item['channel']}/P{item['peak_number']}:"
                    f"shift={item['peak_shift_nm'] * 1000:.0f}pm,"
                    f"ratio={item['amplitude_ratio']:.3f}"
                    for item in failed_peaks
                ),
                flush=True,
            )

    payload = {
        "schema": "temperature_equal_interval_hardware_match_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "board": args.board,
        "serial": args.serial,
        "settle_s": args.settle,
        "feedback_selectors": list(selectors),
        "point_count": len(points),
        "wavelength_nm": wavelengths_nm.tolist(),
        "peak_numbers": peak_numbers.tolist(),
        "equal_interval_reference_codes": reference.astype(int).tolist(),
        "frames": frame_reports,
        "passed": all(item["passed"] for item in frame_reports),
    }
    output_dir = root / "Python/outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / (
        "temperature_equal_interval_match_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".json"
    )
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"report={output}", flush=True)
    print(f"ACCEPTANCE_PASS={int(payload['passed'])}", flush=True)
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    # A core application is enough for direct signal delivery; no GUI is shown.
    _application = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    raise SystemExit(main())
