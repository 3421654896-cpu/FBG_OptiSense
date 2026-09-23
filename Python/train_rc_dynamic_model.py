"""Capture continuous stress scans and validate translation-invariant ADC models.

The candidate models deliberately receive no wavelength, table-row, segment, or
laser-current identity.  They can learn acquisition-chain dynamics, but cannot
memorise the static FBG reflection spectrum that later carries the strain signal.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import serial

from fbg_peak_fitting import build_segments, detect_wavelength_gaps, fit_channel_segments


HERE = Path(__file__).resolve().parent
TRUTH_SOURCE = HERE / "rc_model_accuracy_63.json"
DEFAULT_OUTPUT = HERE / "rc_dynamic_model_validation.json"
COMMAND_SIZE = 808
ADC_VREF = 2.5
ADC_CODES = 4096.0


def mode_command(mode: int) -> bytes:
    packet = bytearray(COMMAND_SIZE)
    packet[:4] = bytes((0xFF, 0xFF, 0x01, 0x02))
    packet[8] = int(mode)
    return bytes(packet)


def raw_capture_command() -> bytes:
    packet = bytearray(COMMAND_SIZE)
    packet[:4] = bytes((0xFF, 0xFF, 0x03, 0x02))
    return bytes(packet)


def extract_raw_frames(buffer: bytearray) -> list[dict]:
    frames: list[dict] = []
    header = b"\xD6\x6D"
    while True:
        start = buffer.find(header)
        if start < 0:
            if len(buffer) > 1:
                del buffer[:-1]
            break
        if start:
            del buffer[:start]
        if len(buffer) < 8:
            break
        if buffer[2] != 1:
            del buffer[0]
            continue
        point_count = int.from_bytes(buffer[3:5], "big")
        if not 1 <= point_count <= 2500:
            del buffer[0]
            continue
        frame_length = 8 + 9 * point_count
        if len(buffer) < frame_length:
            break
        if buffer[frame_length - 2 : frame_length] != b"\x6D\xD6":
            del buffer[0]
            continue
        frame = bytes(buffer[:frame_length])
        del buffer[:frame_length]
        records = np.zeros((point_count, 5), dtype=np.uint16)
        offset = 6
        for point in range(point_count):
            records[point, 0] = int.from_bytes(frame[offset : offset + 2], "big")
            records[point, 1] = int.from_bytes(frame[offset + 2 : offset + 4], "big")
            records[point, 2] = int.from_bytes(frame[offset + 4 : offset + 6], "big")
            records[point, 3] = int.from_bytes(frame[offset + 6 : offset + 8], "big")
            records[point, 4] = frame[offset + 8]
            offset += 9
        frames.append({"active_channel_mask": frame[5], "records": records})
    return frames


def capture_frames(port: serial.Serial, count: int, points: int) -> list[np.ndarray]:
    buffer = bytearray()
    captures: list[np.ndarray] = []
    for capture_index in range(count):
        port.reset_input_buffer()
        port.write(raw_capture_command())
        port.flush()
        deadline = time.monotonic() + 3.0
        received = None
        mask = 0
        while time.monotonic() < deadline and received is None:
            chunk = port.read(4096)
            if chunk:
                buffer.extend(chunk)
            for frame in extract_raw_frames(buffer):
                if frame["records"].shape[0] == points:
                    received = frame["records"]
                    mask = int(frame["active_channel_mask"])
                    break
        if received is None:
            raise RuntimeError(f"capture {capture_index + 1}: diagnostic frame timeout")
        if not (mask & 0x02):
            raise RuntimeError(f"capture {capture_index + 1}: CH1 is disabled (mask=0x{mask:02X})")
        captures.append(received.astype(float))
        print(
            f"capture {capture_index + 1:02d}/{count}: "
            f"rechecks={int(np.sum(received[:, 4]))}",
            flush=True,
        )
    return captures


def metrics(truth: np.ndarray, predicted: np.ndarray) -> dict:
    error_v = (predicted - truth) * ADC_VREF / ADC_CODES
    centered = truth - np.mean(truth)
    ss_res = float(np.sum((predicted - truth) ** 2))
    ss_tot = float(np.sum(centered**2))
    return {
        "bias_v": float(np.mean(error_v)),
        "mae_v": float(np.mean(np.abs(error_v))),
        "rmse_v": float(np.sqrt(np.mean(error_v**2))),
        "maximum_absolute_error_v": float(np.max(np.abs(error_v))),
        "r_squared": 1.0 - ss_res / max(ss_tot, 1e-12),
        "within_0p02_v_percent": float(np.mean(np.abs(error_v) <= 0.02) * 100.0),
        "within_0p05_v_percent": float(np.mean(np.abs(error_v) <= 0.05) * 100.0),
    }


def fit_scalar_gain(x: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> float:
    denominator = float(np.dot(x, x) + ridge)
    return float(np.dot(x, y) / denominator) if denominator > 0.0 else 0.0


def predict_two_sample(records: np.ndarray, fast_gain: float, slow_gain: float) -> np.ndarray:
    first = records[..., 0]
    second = records[..., 1]
    slow = records[..., 2]
    recheck = records[..., 4] > 0.5
    predicted = first + fast_gain * (second - first)
    predicted[recheck] = second[recheck] + slow_gain * (slow[recheck] - second[recheck])
    return np.clip(predicted, 0.0, 4095.0)


def cross_validate_by_segment(
    captures: np.ndarray,
    truth: np.ndarray,
    segment_ids: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    predicted = np.zeros(captures.shape[:2], dtype=float)
    folds: list[dict] = []
    recheck = captures[..., 4] > 0.5
    for segment in sorted(set(int(value) for value in segment_ids)):
        test_points = segment_ids == segment
        train_points = ~test_points
        train_mask = np.broadcast_to(train_points, captures.shape[:2])
        truth_grid = np.broadcast_to(truth, captures.shape[:2])

        fast_train = train_mask & ~recheck
        fast_delta = captures[..., 1] - captures[..., 0]
        fast_target = truth_grid - captures[..., 0]
        fast_gain = fit_scalar_gain(fast_delta[fast_train], fast_target[fast_train])

        slow_train = train_mask & recheck
        slow_delta = captures[..., 2] - captures[..., 1]
        slow_target = truth_grid - captures[..., 1]
        slow_gain = fit_scalar_gain(slow_delta[slow_train], slow_target[slow_train])
        if np.count_nonzero(slow_train) < 3:
            slow_gain = fast_gain

        fold_prediction = predict_two_sample(captures[:, test_points, :], fast_gain, slow_gain)
        predicted[:, test_points] = fold_prediction
        folds.append({
            "held_out_segment": segment,
            "fast_gain": fast_gain,
            "slow_gain": slow_gain,
            "fast_training_samples": int(np.count_nonzero(fast_train)),
            "slow_training_samples": int(np.count_nonzero(slow_train)),
        })
    return predicted, folds


def fit_peak_centers(
    wavelengths: np.ndarray,
    values_codes: np.ndarray,
    segments: list[slice],
) -> list[dict]:
    values_v = values_codes * ADC_VREF / ADC_CODES
    fits = fit_channel_segments(
        wavelengths,
        values_v,
        segments,
        min_prominence_v=0.02,
        allow_edge_peak=True,
    )
    return [{
        "center_nm": float(fit.center_nm),
        "r_squared": float(fit.r_squared),
        "valid": bool(fit.valid),
    } for fit in fits]


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--baud", type=int, default=2_000_000)
    parser.add_argument("--captures", type=int, default=20)
    parser.add_argument("--truth-source", type=Path, default=TRUTH_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    truth_payload = json.loads(args.truth_source.read_text(encoding="utf-8"))
    rows = truth_payload["rows"]
    truth = np.asarray([
        float(row.get("single_ch1_v", row.get("ch1_v"))) * ADC_CODES / ADC_VREF
        for row in rows
    ])
    wavelengths = np.asarray([float(row["wavelength_nm"]) for row in rows])
    segment_ids = np.asarray([int(row["segment"]) for row in rows])
    segments = build_segments(len(rows), detect_wavelength_gaps(wavelengths))

    port = serial.Serial(args.port, args.baud, timeout=0.02, write_timeout=1.0)
    try:
        port.dtr = True
        port.reset_input_buffer()
        port.write(mode_command(0))
        port.flush()
        # Allow the first discovery frame and gain state to become final.
        time.sleep(1.2)
        captures = np.stack(capture_frames(port, args.captures, len(rows)), axis=0)
    finally:
        port.close()

    truth_grid = np.broadcast_to(truth, captures.shape[:2])
    current = captures[..., 3]
    candidate_cv, folds = cross_validate_by_segment(captures, truth, segment_ids)

    recheck = captures[..., 4] > 0.5
    fast_mask = ~recheck
    fast_gain = fit_scalar_gain(
        (captures[..., 1] - captures[..., 0])[fast_mask],
        (truth_grid - captures[..., 0])[fast_mask],
    )
    slow_gain = fit_scalar_gain(
        (captures[..., 2] - captures[..., 1])[recheck],
        (truth_grid - captures[..., 1])[recheck],
    )
    candidate_all = predict_two_sample(captures, fast_gain, slow_gain)

    truth_fits = fit_peak_centers(wavelengths, truth, segments)
    current_median = np.median(current, axis=0)
    candidate_median = np.median(candidate_cv, axis=0)
    current_fits = fit_peak_centers(wavelengths, current_median, segments)
    candidate_fits = fit_peak_centers(wavelengths, candidate_median, segments)
    peak_comparison = []
    for index, (truth_fit, current_fit, candidate_fit) in enumerate(
        zip(truth_fits, current_fits, candidate_fits)
    ):
        truth_center = truth_fit["center_nm"]
        peak_comparison.append({
            "segment": index,
            "truth": truth_fit,
            "current": current_fit,
            "candidate_cross_validated": candidate_fit,
            "current_center_error_pm": finite_or_none(
                (current_fit["center_nm"] - truth_center) * 1000.0
            ),
            "candidate_center_error_pm": finite_or_none(
                (candidate_fit["center_nm"] - truth_center) * 1000.0
            ),
        })

    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "method": {
            "capture": "normal continuous scan; no extra ADC conversions or delays",
            "truth": str(args.truth_source.name) + " settled single-value CH1",
            "validation": "leave-one-wavelength-segment-out",
            "candidate": (
                "translation-invariant first+k*(second-first); recheck uses "
                "second+k_slow*(slow-second); no wavelength/DAC/segment feature"
            ),
        },
        "capture_count": int(captures.shape[0]),
        "point_count": int(captures.shape[1]),
        "recheck_percent": float(np.mean(recheck) * 100.0),
        "full_data_coefficients": {
            "fast_gain": fast_gain,
            "slow_gain": slow_gain,
        },
        "folds": folds,
        "current_firmware": metrics(truth_grid, current),
        "candidate_cross_validated": metrics(truth_grid, candidate_cv),
        "candidate_full_data_reference_only": metrics(truth_grid, candidate_all),
        "peak_comparison": peak_comparison,
        "captures": [
            {
                "first": frame[:, 0].astype(int).tolist(),
                "second": frame[:, 1].astype(int).tolist(),
                "slow": frame[:, 2].astype(int).tolist(),
                "current_estimate": frame[:, 3].astype(int).tolist(),
                "recheck": frame[:, 4].astype(int).tolist(),
            }
            for frame in captures
        ],
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "recheck_percent": payload["recheck_percent"],
        "coefficients": payload["full_data_coefficients"],
        "current": payload["current_firmware"],
        "candidate_cv": payload["candidate_cross_validated"],
        "current_peak_error_pm": [row["current_center_error_pm"] for row in peak_comparison],
        "candidate_peak_error_pm": [row["candidate_center_error_pm"] for row in peak_comparison],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
