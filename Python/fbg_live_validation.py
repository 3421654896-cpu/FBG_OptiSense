"""Capture live STM32 scan frames and validate the FBG fitting pipeline."""

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import serial
import yaml

from fbg_peak_fitting import (
    AdaptivePeakTracker,
    PeakDisplayNormalizer,
    TemperaturePeakTracker,
    build_segments,
    detect_wavelength_gaps,
    fit_channel_segments,
    mask_unconnected_fbg_segments,
)


ADC_REFERENCE_V = 2.5
ADC_CODE_COUNT = 4096.0
COMMAND_SIZE = 808


def detect_wave_gaps(waves, gap_ratio=1.75, local_radius=4):
    return detect_wavelength_gaps(waves)


def mode_command(mode):
    command = bytearray(COMMAND_SIZE)
    command[0:4] = bytes((0xFF, 0xFF, 0x01, 0x02))
    command[8] = int(mode)
    return bytes(command)


def peak_focus_command(waves, segments, channel_two_fits):
    """Build the same CH2 peak-neighborhood command used by the GUI."""
    focus = set()
    for section, fit in zip(segments, channel_two_fits):
        if not fit.valid or not np.isfinite(fit.center_nm):
            continue
        indices = np.arange(section.start, section.stop)
        nearest = int(indices[np.argmin(np.abs(waves[indices] - fit.center_nm))])
        for index in range(nearest - 2, nearest + 3):
            if section.start <= index < section.stop:
                focus.add(index)

    channel_points = [[], [], sorted(focus), []]
    command = bytearray(COMMAND_SIZE)
    command[0:4] = bytes((0xFF, 0xFF, 0x01, 0x04))
    position = 4
    for points in channel_points:
        command[position] = len(points)
        position += 1
        for index in points:
            command[position] = (index >> 8) & 0xFF
            command[position + 1] = index & 0xFF
            position += 2
    return bytes(command), tuple(tuple(points) for points in channel_points)


def extract_frames(buffer):
    frames = []
    position = 0
    while True:
        start = buffer.find(b"\xEE\xEE", position)
        if start < 0:
            break
        end = buffer.find(b"\xFF\xEF", start + 2)
        if end < 0:
            break
        frames.append(bytes(buffer[start:end + 2]))
        position = end + 2
    if position:
        del buffer[:position]
    elif len(buffer) > 4096:
        del buffer[:-2]
    return frames


def decode_adc_frame(frame, expected_points):
    if len(frame) < 4 or frame[:2] != b"\xEE\xEE":
        return None
    point_count = (frame[2] << 8) | frame[3]
    if point_count != expected_points:
        return None
    data_end = 4 + point_count * 8
    if len(frame) <= data_end or frame[data_end] != 0xAB:
        return None
    payload = np.frombuffer(frame[4:data_end], dtype=">u2").reshape(point_count, 4)
    return payload.astype(float).T * ADC_REFERENCE_V / ADC_CODE_COUNT


def finite_stats(values, discard=0):
    data = np.asarray(values[discard:], dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return {"count": 0}
    return {
        "count": int(data.size),
        "median_nm": float(np.median(data)),
        "std_pm": float(np.std(data) * 1000.0),
        "peak_to_peak_pm": float(np.ptp(data) * 1000.0),
        "p95_abs_from_median_pm": float(
            np.percentile(np.abs(data - np.median(data)), 95) * 1000.0
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--baud", type=int, default=2_000_000)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument(
        "--focus",
        action="store_true",
        help="enable experimental selective CH2 oversampling for an A/B comparison",
    )
    parser.add_argument("--include-series", action="store_true")
    parser.add_argument(
        "--temperature",
        action="store_true",
        help="validate MCU precision-temperature mode (mode 3) with GUI-equivalent fusion",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).with_name("fbg_static_validation.json")),
    )
    args = parser.parse_args()

    yaml_path = Path(__file__).with_name(
        "wave_const.yaml" if args.temperature else "stress_wave_const.yaml"
    )
    wave_data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))["Wave_DATA"]
    # Temperature uses the complete 100-point precision table; stress uses
    # the independently validated 68-point high-power table.  Temperature saves time by converting
    # only CH0/CH1; retaining the obsolete [34:] slice made every valid
    # 100-point firmware frame look malformed to this verifier.
    waves = np.asarray([major + minor * 0.001 for major, minor in wave_data])
    segments = build_segments(len(waves), detect_wave_gaps(waves))
    tracker_type = TemperaturePeakTracker if args.temperature else AdaptivePeakTracker
    trackers = [
        (
            tracker_type(len(segments))
            if args.temperature
            else tracker_type(len(segments), immediate_response=True)
        )
        for _ in range(4)
    ]
    active_channels = {0, 1} if args.temperature else {0, 1, 2, 3}
    normalizer = PeakDisplayNormalizer(channel_count=4)
    precision_frame_history = deque(maxlen=5)

    raw_centers = [[[] for _ in segments] for _ in range(4)]
    tracked_centers = [[[] for _ in segments] for _ in range(4)]
    amplitudes = [[[] for _ in segments] for _ in range(4)]
    qualities = [[[] for _ in segments] for _ in range(4)]
    uncertainties = [[[] for _ in segments] for _ in range(4)]
    frame_times = []
    display_scales = []
    captured_channels = []
    raw_captured_channels = []

    receive_buffer = bytearray()
    last_focus_signature = None
    with serial.Serial(args.port, args.baud, timeout=0.05) as device:
        device.reset_input_buffer()
        if not args.focus:
            clear_focus = bytearray(COMMAND_SIZE)
            clear_focus[0:4] = bytes((0xFF, 0xFF, 0x01, 0x04))
            device.write(bytes(clear_focus))
            time.sleep(0.1)
        device.write(mode_command(3 if args.temperature else 0))
        deadline = time.perf_counter() + args.seconds
        while time.perf_counter() < deadline:
            receive_buffer.extend(device.read(4096))
            for frame in extract_frames(receive_buffer):
                channels = decode_adc_frame(frame, len(waves))
                if channels is None:
                    continue
                if args.include_series:
                    raw_captured_channels.append(channels.tolist())
                if args.temperature:
                    precision_frame_history.append(channels.copy())
                    channels = np.median(
                        np.stack(tuple(precision_frame_history), axis=0), axis=0
                    )
                frame_times.append(time.perf_counter())
                captured_channels.append(channels.tolist())
                frame_fits = []
                for channel in range(4):
                    if channel not in active_channels:
                        fits = []
                        frame_fits.append(fits)
                        trackers[channel].update(fits)
                        for segment_index in range(len(segments)):
                            raw_centers[channel][segment_index].append(np.nan)
                            tracked_centers[channel][segment_index].append(np.nan)
                            amplitudes[channel][segment_index].append(np.nan)
                            qualities[channel][segment_index].append(np.nan)
                            uncertainties[channel][segment_index].append(np.nan)
                        continue
                    threshold = 0.03 if channel < 2 else 0.00375
                    fits = fit_channel_segments(
                        waves,
                        channels[channel],
                        segments,
                        threshold,
                        allow_edge_peak=not args.temperature,
                    )
                    mask_unconnected_fbg_segments(
                        fits,
                        channel,
                        args.temperature,
                        stress_ch1_all_segments=not args.temperature,
                    )
                    if channel == 3:
                        for fit in fits:
                            fit.valid = False
                    frame_fits.append(fits)
                    tracked = trackers[channel].update(fits)
                    for segment_index, fit in enumerate(fits):
                        raw_centers[channel][segment_index].append(
                            fit.center_nm if fit.valid else np.nan
                        )
                        tracked_centers[channel][segment_index].append(
                            tracked[segment_index]
                        )
                        amplitudes[channel][segment_index].append(
                            fit.amplitude_v if fit.valid else np.nan
                        )
                        qualities[channel][segment_index].append(
                            fit.r_squared if fit.valid else np.nan
                        )
                        uncertainties[channel][segment_index].append(
                            fit.center_std_pm if fit.valid else np.nan
                        )
                focus_command, focus_signature = peak_focus_command(
                    waves, segments, frame_fits[2]
                )
                if args.focus and focus_signature != last_focus_signature:
                    device.write(focus_command)
                    last_focus_signature = focus_signature
                display_scales.append(normalizer.update(frame_fits).tolist())
        device.write(mode_command(2))

    elapsed = frame_times[-1] - frame_times[0] if len(frame_times) >= 2 else 0.0
    report = {
        "port": args.port,
        "baud": args.baud,
        "requested_seconds": args.seconds,
        "mode": "temperature" if args.temperature else "stress",
        "selective_ch2_focus": args.focus,
        "frame_count": len(frame_times),
        "fps": (len(frame_times) - 1) / elapsed if elapsed > 0 else 0.0,
        "segments": [
            {
                "index": index,
                "start_index": section.start,
                "stop_index": section.stop,
                "start_nm": float(waves[section.start]),
                "stop_nm": float(waves[section.stop - 1]),
            }
            for index, section in enumerate(segments)
        ],
        "channels": [],
        "final_display_scales": display_scales[-1] if display_scales else [1] * 4,
    }
    if captured_channels:
        captured = np.asarray(captured_channels, dtype=float)
        median_waveforms = np.median(captured, axis=0)
        report["median_voltage_waveforms_v"] = median_waveforms.tolist()
        report["voltage_summary_v"] = [
            {
                "channel": channel,
                "minimum": float(np.min(median_waveforms[channel])),
                "maximum": float(np.max(median_waveforms[channel])),
                "span": float(np.ptp(median_waveforms[channel])),
                "median": float(np.median(median_waveforms[channel])),
            }
            for channel in range(4)
        ]
    if args.include_series:
        report["raw_voltage_waveform_series_v"] = raw_captured_channels
    discard = min(10, max(len(frame_times) // 5, 0))
    for channel in range(4):
        channel_report = {"channel": channel, "peaks": []}
        for segment_index in range(len(segments)):
            raw_stat = finite_stats(raw_centers[channel][segment_index])
            tracked_stat = finite_stats(
                tracked_centers[channel][segment_index], discard=discard
            )
            amp = np.asarray(amplitudes[channel][segment_index], dtype=float)
            quality = np.asarray(qualities[channel][segment_index], dtype=float)
            channel_report["peaks"].append(
                {
                    "segment": segment_index,
                    "valid_frames": int(np.count_nonzero(np.isfinite(amp))),
                    "raw": raw_stat,
                    "stabilized": tracked_stat,
                    "median_amplitude_v": (
                        float(np.nanmedian(amp)) if np.any(np.isfinite(amp)) else None
                    ),
                    "median_r_squared": (
                        float(np.nanmedian(quality))
                        if np.any(np.isfinite(quality)) else None
                    ),
                    **(
                        {
                            "raw_center_series_nm": raw_centers[channel][segment_index],
                            "tracked_center_series_nm": tracked_centers[channel][segment_index],
                            "r_squared_series": qualities[channel][segment_index],
                            "center_std_series_pm": uncertainties[channel][segment_index],
                        }
                        if args.include_series
                        else {}
                    ),
                }
            )
        report["channels"].append(channel_report)

    output = Path(args.output)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
