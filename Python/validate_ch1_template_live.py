"""Validate the certified nine-FBG measured-template fit on live CH1 frames.

The command is diagnostic only: it does not move the machine and it never
changes the installed wavelength table.  A median no-load baseline is built
from the first frames, then every later 45-point frame is evaluated through
the exact measured template bound to the active table CRC.  The laser is
returned to the PI11210 SOA shutter state on every exit path.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

import app_JDSU as app
from diagnose_adaptive_stress_peaks_live import _shutter, _wait_feedback
from fbg_lan_serial import FbgLanSerial
from finger_dataset import SpectrumFrame
from finger_features import PeakQuality, extract_ch1_peak_features
from finger_sampling_template import load_finger_sampling_template
from verify_lan_full_function_live import wait_scan_frame


HERE = Path(__file__).resolve().parent
DEFAULT_INSTALLED = HERE / "mode_tables_from_fullband_2001.json"


def _active_template_path(installed_path: Path) -> Path:
    payload = json.loads(installed_path.read_text(encoding="utf-8"))
    candidate = Path(str(payload.get("candidate_report", "")))
    if not candidate.is_absolute():
        candidate = (installed_path.parent / candidate).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(
            f"installed table does not reference a readable certified report: {candidate}"
        )
    return candidate


def _stress_axis() -> tuple[float, ...]:
    payload = yaml.safe_load((HERE / "stress_wave_const.yaml").read_text(encoding="utf-8"))
    return tuple(float(row[0]) + float(row[1]) * 0.001 for row in payload["Wave_DATA"])


def _decode_ch1_codes(frame: bytes, point_count: int) -> tuple[int, ...]:
    payload = np.frombuffer(frame[4 : 4 + point_count * 8], dtype=">u2")
    return tuple(int(value) for value in payload.reshape(point_count, 4)[:, 1])


def _frame(
    sequence: int,
    wavelengths: tuple[float, ...],
    codes: tuple[int, ...],
    *,
    table_crc32: int,
) -> SpectrumFrame:
    return SpectrumFrame(
        sequence=sequence,
        wavelengths_nm=wavelengths,
        raw_adc_codes={1: codes},
        transimpedance_ohm={1: 40_000.0},
        digital_gain={1: 1.0},
        feedback_selector={1: 1},
        table_crc32=table_crc32,
        packet_crc_ok=True,
        device_sequence=sequence,
        mode="stress",
        source="lan",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.3.46")
    parser.add_argument("--warmup-frames", type=int, default=60)
    parser.add_argument("--baseline-frames", type=int, default=15)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--selector", type=int, default=1)
    parser.add_argument("--max-uncertainty-pm", type=float, default=20.0)
    parser.add_argument("--max-normalized-rmse", type=float, default=0.40)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--installed", type=Path, default=DEFAULT_INSTALLED)
    args = parser.parse_args()

    if args.warmup_frames < 0 or args.baseline_frames < 3 or args.frames < 1:
        raise ValueError(
            "warmup-frames must be >=0, baseline-frames >=3 and frames >=1"
        )
    template_path = _active_template_path(args.installed.resolve())
    template = load_finger_sampling_template(template_path)
    wavelengths = _stress_axis()
    if len(wavelengths) != 45 or not template.wavelength_axis_matches(wavelengths):
        raise RuntimeError("active 45-point axis does not match the certified template")

    device = FbgLanSerial(board_host=args.host, timeout=0.12, write_timeout=3.0)
    shutter_confirmed = False
    records: list[dict[str, object]] = []
    try:
        device.open()
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and not device.connected:
            time.sleep(0.05)
        if not device.connected:
            raise RuntimeError("LAN full-function channel is unavailable")

        device.write(app.build_work_mode_command(2))
        time.sleep(0.12)
        device.reset_input_buffer()
        _wait_feedback(device, [args.selector, args.selector])
        device.write(app.build_work_mode_command(0))
        time.sleep(0.05)
        device.reset_input_buffer()

        for _ in range(args.warmup_frames):
            wait_scan_frame(device, 45, 8.0)

        baseline_rows = []
        for _ in range(args.baseline_frames):
            baseline_rows.append(
                _decode_ch1_codes(wait_scan_frame(device, 45, 8.0), 45)
            )
        baseline_codes = tuple(
            int(round(value))
            for value in np.median(np.asarray(baseline_rows, dtype=float), axis=0)
        )
        baseline = _frame(
            0,
            wavelengths,
            baseline_codes,
            table_crc32=int(template.table_crc32),
        )

        accepted_counts = np.zeros(9, dtype=int)
        shifts: list[list[float]] = [[] for _ in range(9)]
        reasons = [Counter() for _ in range(9)]
        for sequence in range(1, args.frames + 1):
            codes = _decode_ch1_codes(wait_scan_frame(device, 45, 8.0), 45)
            current = _frame(
                sequence,
                wavelengths,
                codes,
                table_crc32=int(template.table_crc32),
            )
            features = extract_ch1_peak_features(
                current,
                baseline_frame=baseline,
                sampling_template=template,
                template_max_uncertainty_pm=args.max_uncertainty_pm,
                template_max_normalized_rmse=args.max_normalized_rmse,
            )
            frame_record = {
                "sequence": sequence,
                "raw_ch1_adc_codes": list(codes),
                "peaks": [],
            }
            for peak_index, peak in enumerate(features.peaks):
                accepted = bool(peak.valid and math.isfinite(peak.center_shift_pm))
                accepted_counts[peak_index] += int(accepted)
                if accepted:
                    shifts[peak_index].append(float(peak.center_shift_pm))
                reasons[peak_index][peak.template_fit_reason] += 1
                frame_record["peaks"].append(
                    {
                        "accepted": accepted,
                        "center_nm": (
                            float(peak.center_nm) if math.isfinite(peak.center_nm) else None
                        ),
                        "shift_pm": (
                            float(peak.center_shift_pm)
                            if math.isfinite(peak.center_shift_pm)
                            else None
                        ),
                        "uncertainty_pm": (
                            float(peak.center_uncertainty_pm)
                            if math.isfinite(peak.center_uncertainty_pm)
                            else None
                        ),
                        "normalized_rmse": (
                            float(peak.template_fit_normalized_rmse)
                            if math.isfinite(peak.template_fit_normalized_rmse)
                            else None
                        ),
                        "quality_mask": int(peak.quality_mask),
                        "quality_names": [
                            flag.name
                            for flag in PeakQuality
                            if flag and PeakQuality(peak.quality_mask) & flag
                        ],
                        "reason": peak.template_fit_reason,
                    }
                )
            records.append(frame_record)

        peak_summary = []
        for index in range(9):
            values = np.asarray(shifts[index], dtype=float)
            summary = {
                "peak_number": index + 1,
                "accepted_frames": int(accepted_counts[index]),
                "total_frames": int(args.frames),
                "acceptance_rate": float(accepted_counts[index] / args.frames),
                "median_shift_pm": float(np.median(values)) if values.size else None,
                "shift_std_pm": float(np.std(values)) if values.size else None,
                "shift_p2p_pm": float(np.ptp(values)) if values.size else None,
                "reasons": dict(reasons[index]),
            }
            peak_summary.append(summary)
            print(
                f"G{index + 1} accepted={accepted_counts[index]}/{args.frames} "
                f"median={summary['median_shift_pm']}pm "
                f"std={summary['shift_std_pm']}pm p2p={summary['shift_p2p_pm']}pm"
            )

        report = {
            "schema": "ch1_measured_template_live_validation_v1",
            "host": args.host,
            "channel": 1,
            "point_count": 45,
            "grating_count": 9,
            "template_path": str(template_path),
            "table_crc32": int(template.table_crc32),
            "warmup_frames": int(args.warmup_frames),
            "baseline_frames": int(args.baseline_frames),
            "validation_frames": int(args.frames),
            "maximum_uncertainty_pm": float(args.max_uncertainty_pm),
            "maximum_normalized_rmse": float(args.max_normalized_rmse),
            "baseline_raw_ch1_adc_codes": list(baseline_codes),
            "peaks": peak_summary,
            "frames": records,
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
            print(f"report={args.output.resolve()}")
        return 0 if np.all(accepted_counts == args.frames) else 1
    finally:
        try:
            if device.is_open and device.connected:
                shutter_confirmed = _shutter(device)
        finally:
            device.close()
            print(f"soa_shutter_confirmed={int(shutter_confirmed)}")


if __name__ == "__main__":
    raise SystemExit(main())
