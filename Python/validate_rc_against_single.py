"""Compare fast stress-mode RC reconstruction with settled single-value truth."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import serial

from fbg_peak_fitting import build_segments, detect_wavelength_gaps, fit_channel_segments
from laser_dac_safety import validate_pi11210_codes


HERE = Path(__file__).resolve().parent
TABLE_SOURCE = HERE / "stress_high_power_135_final.json"
DEFAULT_OUTPUT = HERE / "rc_model_accuracy_63.json"
DEFAULT_PLOT = HERE / "rc_model_accuracy_63.png"
COMMAND_SIZE = 808
ADC_VREF = 2.5
ADC_CODES = 4096.0


def mode_command(mode: int) -> bytes:
    packet = bytearray(COMMAND_SIZE)
    packet[:4] = bytes((0xFF, 0xFF, 0x01, 0x02))
    packet[8] = int(mode)
    return bytes(packet)


def dac_command(codes: list[int]) -> bytes:
    safe_codes = validate_pi11210_codes(codes)
    packet = bytearray(COMMAND_SIZE)
    packet[:4] = bytes((0xFF, 0xFF, 0x00, 0x01))
    for channel, code in enumerate(safe_codes):
        packet[4 + 2 * channel : 6 + 2 * channel] = code.to_bytes(2, "big")
    return bytes(packet)


def extract_single_frames(buffer: bytearray) -> list[bytes]:
    frames: list[bytes] = []
    while True:
        start = buffer.find(b"\xFF\xFF")
        if start < 0:
            if len(buffer) > 1:
                del buffer[:-1]
            break
        if start:
            del buffer[:start]
        if len(buffer) < 20:
            break
        nested = buffer.find(b"\xFF\xFF", 2, 20)
        if nested > 0:
            del buffer[:nested]
            continue
        candidate = bytes(buffer[:20])
        if candidate[2] != 2 or candidate[3] not in (0, 1, 2):
            del buffer[0]
            continue
        del buffer[:20]
        frames.append(candidate)
    return frames


def collect_single_rt(port: serial.Serial, duration_s: float) -> list[list[int]]:
    buffer = bytearray()
    readings: list[list[int]] = []
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        buffer.extend(port.read(4096))
        for frame in extract_single_frames(buffer):
            if frame[3] == 2:
                readings.append([
                    int.from_bytes(frame[8 + 2 * channel : 10 + 2 * channel], "big")
                    for channel in range(4)
                ])
    return readings


def extract_scan_frames(buffer: bytearray) -> list[bytes]:
    frames: list[bytes] = []
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


def collect_stress_frames(port: serial.Serial, duration_s: float, points: int) -> list[np.ndarray]:
    buffer = bytearray()
    spectra: list[np.ndarray] = []
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        buffer.extend(port.read(4096))
        for frame in extract_scan_frames(buffer):
            if len(frame) < 4 or int.from_bytes(frame[2:4], "big") != points:
                continue
            data_end = 4 + points * 8
            if len(frame) <= data_end or frame[data_end] != 0xAB:
                continue
            values = np.frombuffer(frame[4:data_end], dtype=">u2").reshape(points, 4)
            spectra.append(values.astype(float).T)
    return spectra


def regression_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict:
    error = predicted - truth
    truth_centered = truth - np.mean(truth)
    residual_sum = float(np.sum(error**2))
    total_sum = float(np.sum(truth_centered**2))
    correlation = float(np.corrcoef(truth, predicted)[0, 1]) if truth.size > 1 else math.nan
    return {
        "bias_v": float(np.mean(error)),
        "mae_v": float(np.mean(np.abs(error))),
        "rmse_v": float(np.sqrt(np.mean(error**2))),
        "maximum_absolute_error_v": float(np.max(np.abs(error))),
        "r_squared": 1.0 - residual_sum / max(total_sum, 1e-12),
        "pearson_r": correlation,
        "within_0p02_v_percent": float(np.mean(np.abs(error) <= 0.02) * 100.0),
        "within_0p05_v_percent": float(np.mean(np.abs(error) <= 0.05) * 100.0),
        "within_0p10_v_percent": float(np.mean(np.abs(error) <= 0.10) * 100.0),
    }


def fit_summary(wavelengths: np.ndarray, values: np.ndarray, segments: list[slice]) -> list[dict]:
    fits = fit_channel_segments(
        wavelengths,
        values,
        segments,
        min_prominence_v=0.02,
        allow_edge_peak=True,
    )
    return [
        {
            "segment": index,
            "start_index": segment.start,
            "stop_index": segment.stop,
            "center_nm": float(fit.center_nm),
            "amplitude_v": float(fit.amplitude_v),
            "r_squared": float(fit.r_squared),
            "rmse_v": float(fit.rmse_v),
            "center_std_pm": float(fit.center_std_pm),
            "valid": bool(fit.valid),
        }
        for index, (segment, fit) in enumerate(zip(segments, fits))
    ]


def render_plot(
    path: Path,
    wavelengths: np.ndarray,
    single_v: np.ndarray,
    stress_v: np.ndarray,
    segments: list[slice],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        # Measurement JSON is the primary artifact; plotting is optional on
        # lightweight acquisition environments.
        return

    figure, axes = plt.subplots(3, 3, figsize=(15, 10), constrained_layout=True)
    for segment_index, (axis, segment) in enumerate(zip(axes.flat, segments)):
        x = wavelengths[segment]
        axis.plot(x, single_v[segment], "o-", label="settled single", linewidth=1.6)
        axis.plot(x, stress_v[segment], "s--", label="fast RC", linewidth=1.2)
        axis.set_title(f"Segment {segment_index + 1}")
        axis.set_xlabel("Wavelength (nm)")
        axis.set_ylabel("CH1 (V)")
        axis.grid(True, alpha=0.3)
        if segment_index == 0:
            axis.legend()
    figure.suptitle("CH1 settled single-value truth vs stress-mode RC reconstruction")
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--baud", type=int, default=2_000_000)
    parser.add_argument("--precondition", type=float, default=0.30)
    parser.add_argument("--target-settle", type=float, default=0.32)
    parser.add_argument("--stress-seconds", type=float, default=8.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    args = parser.parse_args()

    source = json.loads(TABLE_SOURCE.read_text(encoding="utf-8"))
    rows = source["rows"]
    point_count = len(rows)
    wavelengths = np.asarray([float(row["final_wavelength_nm"]) for row in rows])
    powers_mw = np.asarray([float(row["final_power_mw"]) for row in rows])
    segments = build_segments(point_count, detect_wavelength_gaps(wavelengths))
    single_codes = np.zeros((4, point_count), dtype=float)
    single_spans = np.zeros((4, point_count), dtype=float)
    single_frame_counts: list[int] = []

    port = serial.Serial(args.port, args.baud, timeout=0.02, write_timeout=1.0)
    try:
        port.dtr = True
        port.reset_input_buffer()
        port.write(mode_command(2))
        port.flush()
        time.sleep(0.12)
        port.reset_input_buffer()

        for index, row in enumerate(rows):
            predecessor = rows[(index - 1) % point_count]
            port.write(dac_command(predecessor["codes"]))
            port.flush()
            collect_single_rt(port, args.precondition)
            port.write(dac_command(row["codes"]))
            port.flush()
            readings = collect_single_rt(port, args.target_settle)
            if len(readings) < 5:
                raise RuntimeError(f"point {index}: only {len(readings)} RT frames")
            stable = np.asarray(readings[-min(80, len(readings)) :], dtype=float)
            single_codes[:, index] = np.median(stable, axis=0)
            single_spans[:, index] = np.ptp(stable, axis=0)
            single_frame_counts.append(len(readings))
            print(
                f"single {index + 1:02d}/{point_count}: "
                f"{wavelengths[index]:.6f} nm CH1="
                f"{single_codes[1, index] * ADC_VREF / ADC_CODES:.4f} V",
                flush=True,
            )

        port.reset_input_buffer()
        port.write(mode_command(0))
        port.flush()
        stress_start = time.monotonic()
        stress_frames = collect_stress_frames(port, args.stress_seconds, point_count)
        stress_elapsed = time.monotonic() - stress_start
    finally:
        port.close()

    if len(stress_frames) < 8:
        raise RuntimeError(f"only {len(stress_frames)} stress frames")
    # Exclude channel-discovery and initial gain/path settling frames.
    stress_stack = np.stack(stress_frames[3:], axis=0)
    stress_codes = np.median(stress_stack, axis=0)
    single_v = single_codes * ADC_VREF / ADC_CODES
    stress_v = stress_codes * ADC_VREF / ADC_CODES
    single_fits = fit_summary(wavelengths, single_v[1], segments)
    stress_fits = fit_summary(wavelengths, stress_v[1], segments)
    peak_comparison: list[dict] = []
    for truth, predicted in zip(single_fits, stress_fits):
        center_error_pm = (
            (predicted["center_nm"] - truth["center_nm"]) * 1000.0
            if math.isfinite(truth["center_nm"]) and math.isfinite(predicted["center_nm"])
            else math.nan
        )
        peak_comparison.append({
            "segment": truth["segment"],
            "single": truth,
            "stress": predicted,
            "center_error_pm": center_error_pm,
            "amplitude_error_v": predicted["amplitude_v"] - truth["amplitude_v"],
        })

    rows_out = []
    for index, row in enumerate(rows):
        rows_out.append({
            "index": index,
            "segment": int(row["segment"]),
            "wavelength_nm": float(wavelengths[index]),
            "laser_power_mw": float(powers_mw[index]),
            "codes": [int(value) for value in row["codes"]],
            "single_ch1_v": float(single_v[1, index]),
            "stress_ch1_v": float(stress_v[1, index]),
            "error_v": float(stress_v[1, index] - single_v[1, index]),
            "absolute_error_v": float(abs(stress_v[1, index] - single_v[1, index])),
        })

    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "method": {
            "truth": "predecessor row -> target row -> settled EXTRA/single-value ADC median",
            "precondition_s": args.precondition,
            "target_settle_s": args.target_settle,
            "stress": "TABLE mode fast RC reconstruction, median after 3 startup frames",
            "stress_seconds": args.stress_seconds,
        },
        "point_count": point_count,
        "segment_count": len(segments),
        "single_rt_frames_per_point": {
            "minimum": min(single_frame_counts),
            "median": statistics.median(single_frame_counts),
            "maximum": max(single_frame_counts),
        },
        "stress_frame_count": len(stress_frames),
        "stress_fps": len(stress_frames) / stress_elapsed,
        "ch1_voltage_metrics": regression_metrics(single_v[1], stress_v[1]),
        "laser_power_mw": {
            "minimum": float(np.min(powers_mw)),
            "mean": float(np.mean(powers_mw)),
            "maximum": float(np.max(powers_mw)),
            "coefficient_of_variation_percent": float(np.std(powers_mw) / np.mean(powers_mw) * 100.0),
        },
        "peak_comparison": peak_comparison,
        "rows": rows_out,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    render_plot(args.plot, wavelengths, single_v[1], stress_v[1], segments)
    print(json.dumps({
        "stress_fps": payload["stress_fps"],
        "ch1_voltage_metrics": payload["ch1_voltage_metrics"],
        "peak_center_errors_pm": [item["center_error_pm"] for item in peak_comparison],
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
