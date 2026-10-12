"""Audit every stress-table segment with the same fitting gates as the GUI."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

import app_JDSU as app
from fbg_lan_serial import FbgLanSerial
from fbg_peak_fitting import (
    AdaptivePeakTracker,
    build_segments,
    detect_wavelength_gaps,
    fit_channel_segments,
    mask_unconnected_fbg_segments,
)
from fbg_processing_common import scan_min_prominence
from verify_lan_full_function_live import wait_scan_frame, wait_single


def _decode_adc(frame: bytes, point_count: int) -> np.ndarray:
    payload = np.frombuffer(frame[4 : 4 + point_count * 8], dtype=">u2")
    return payload.reshape(point_count, 4).T.astype(float) * 2.5 / 4096.0


def _wait_feedback(device, selectors):
    device.write(app.build_stress_feedback_command(*selectors))
    result = wait_single(device, app.decode_stress_feedback_status)
    if result != (tuple(selectors), app.ACK_VALUE):
        raise RuntimeError(f"应力跨阻IO回读异常：{result}")


def _shutter(device):
    device.write(app.build_work_mode_command(2))
    time.sleep(0.12)
    device.reset_input_buffer()
    command = app.build_single_value_dac_command(
        (0, 0, 0, 0, 0), soa_mode=app.PI11210_SOA_SHUTTER_MODE
    )
    expected = tuple(
        int.from_bytes(command[4 + index * 2 : 6 + index * 2], "big")
        for index in range(5)
    )
    device.write(command)

    def decode(frame):
        if frame[:4] != b"\xff\xff\x02\x00":
            return None
        returned = tuple(
            int.from_bytes(frame[4 + index * 2 : 6 + index * 2], "big")
            for index in range(5)
        )
        if returned != expected or frame[14] != app.PI11210_SOA_SHUTTER_MODE:
            return None
        app.UnlimitedAccuracyWorker._validate_ack_status(
            frame, app.PI11210_SOA_SHUTTER_MODE
        )
        return True

    return bool(wait_single(device, decode, 3.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.3.46")
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--selector", type=int, default=1)
    args = parser.parse_args()

    payload = yaml.safe_load(
        Path(__file__).with_name("stress_wave_const.yaml").read_text(encoding="utf-8")
    )
    wavelength_nm = np.asarray(
        [float(row[0]) + float(row[1]) * 0.001 for row in payload["Wave_DATA"]],
        dtype=float,
    )
    segments = build_segments(
        len(wavelength_nm), detect_wavelength_gaps(wavelength_nm)
    )
    threshold = scan_min_prominence(0.03, args.channel, False)
    device = FbgLanSerial(board_host=args.host, timeout=0.12, write_timeout=3.0)
    reports = [[] for _ in segments]
    tracker = AdaptivePeakTracker(len(segments), immediate_response=True)
    shutter_confirmed = False
    try:
        device.open()
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and not device.connected:
            time.sleep(0.05)
        if not device.connected:
            raise RuntimeError("未发现局域网板卡的全功能通道")

        device.write(app.build_work_mode_command(2))
        time.sleep(0.12)
        device.reset_input_buffer()
        selectors = [args.selector, args.selector]
        _wait_feedback(device, selectors)
        device.write(app.build_work_mode_command(0))
        time.sleep(0.05)
        device.reset_input_buffer()

        for frame_number in range(args.frames):
            frame = wait_scan_frame(device, len(wavelength_nm), 8.0)
            values = _decode_adc(frame, len(wavelength_nm))[args.channel]
            fits = fit_channel_segments(
                wavelength_nm,
                values,
                segments,
                threshold,
                allow_edge_peak=True,
            )
            mask_unconnected_fbg_segments(
                fits, args.channel, False, stress_all_segments=True
            )
            valid = [index + 1 for index, fit in enumerate(fits) if fit.valid]
            tracked = tracker.update(fits)
            displayed = [
                index + 1 for index, center in enumerate(tracked) if np.isfinite(center)
            ]
            print(
                f"frame={frame_number + 1} raw_valid={valid} "
                f"displayed={displayed} count={len(displayed)}"
            )
            for index, (segment, fit) in enumerate(zip(segments, fits, strict=True)):
                samples = values[segment]
                reports[index].append(
                    (
                        bool(fit.valid),
                        float(np.ptp(samples)),
                        float(fit.amplitude_v),
                        float(fit.r_squared),
                        float(fit.rmse_v),
                        float(fit.center_std_pm),
                        float(fit.center_nm),
                    )
                )

        print(f"channel=CH{args.channel} threshold_v={threshold:.5f} segments={len(segments)}")
        for index, (segment, rows) in enumerate(zip(segments, reports, strict=True)):
            array = np.asarray(rows, dtype=float)
            valid_frames = int(np.sum(array[:, 0]))
            center = float(np.nanmedian(array[:, 6]))
            print(
                f"P{index + 1} range={wavelength_nm[segment.start]:.5f}-"
                f"{wavelength_nm[segment.stop - 1]:.5f}nm "
                f"valid={valid_frames}/{len(rows)} center={center:.6f}nm "
                f"span={np.median(array[:, 1]) * 1000.0:.2f}mV "
                f"amp={np.median(array[:, 2]) * 1000.0:.2f}mV "
                f"r2={np.median(array[:, 3]):.3f} "
                f"rmse={np.median(array[:, 4]) * 1000.0:.2f}mV "
                f"center_std={np.median(array[:, 5]):.2f}pm"
            )
        return 0
    finally:
        try:
            if device.is_open and device.connected:
                shutter_confirmed = _shutter(device)
        finally:
            device.close()
            print(f"soa_shutter_confirmed={int(shutter_confirmed)}")


if __name__ == "__main__":
    raise SystemExit(main())
