"""Build the strict nine-FBG v4 selection input from audited CH1 teachers.

The complete 2 kOhm capture supplies a measured value at every one of the 2001
wavelengths.  The 40 kOhm capture supplies higher-SNR measurements in nine
81-point peak regions.  A short edge blend maps the low-gain trace into the
40 kOhm scale without creating artificial window-edge peaks.

This is an offline operation.  It neither opens the laser nor talks to Mach3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import app_JDSU as app


HERE = Path(__file__).resolve().parent
SCHEMA = "fbg_ch1_teacher_spectrum_v1"
ADC_LSB_V = 10.0 / 16384.0
EXPECTED_PEAK_WINDOWS = 9
DEFAULT_BLEND_POINTS = 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_teacher(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or int(payload.get("channel", -1)) != 1:
        raise ValueError(f"{path.name}不是CH1教师光谱")
    if payload.get("soa_shutter_confirmed") is not True:
        raise ValueError(f"{path.name}没有精确关光确认")
    rows = payload.get("rows")
    if not isinstance(rows, Mapping):
        raise ValueError(f"{path.name}的rows字段损坏")
    return payload


def _contiguous_ranges(indices: Sequence[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    ranges: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for raw in indices[1:]:
        index = int(raw)
        if index == previous + 1:
            previous = index
            continue
        ranges.append((start, previous))
        start = previous = index
    ranges.append((start, previous))
    return ranges


def _validate_row(row: Mapping[str, Any], point: app.FullbandAccuracyPoint) -> None:
    if int(row.get("index", -1)) != point.index:
        raise ValueError(f"教师光谱第{point.index}点索引错误")
    if not math.isclose(
        float(row.get("measured_wavelength_nm", float("nan"))),
        point.measured_nm,
        rel_tol=0.0,
        abs_tol=5e-7,
    ):
        raise ValueError(f"教师光谱第{point.index}点波长轴与审计表不一致")
    if tuple(int(value) for value in row.get("dac_codes", ())) != point.codes:
        raise ValueError(f"教师光谱第{point.index}点DAC与审计表不一致")


def build_selection(
    full_teacher_path: Path,
    peak_teacher_path: Path,
    *,
    blend_points: int = DEFAULT_BLEND_POINTS,
) -> dict[str, Any]:
    if blend_points < 1:
        raise ValueError("边缘融合点数必须大于0")

    full_path = full_teacher_path.resolve()
    peak_path = peak_teacher_path.resolve()
    full = _load_teacher(full_path)
    peak = _load_teacher(peak_path)
    points = app.load_fullband_accuracy_table(HERE)
    table_path = HERE / "fullband_equal_power_operational_2001.csv"
    table_sha256 = _sha256(table_path)
    for name, payload in (("全带宽", full), ("峰区", peak)):
        recorded = str(payload.get("source", {}).get("table_sha256", "")).lower()
        if recorded != table_sha256:
            raise ValueError(f"{name}教师光谱对应的2001点表已改变")

    full_rows = full["rows"]
    if set(full_rows) != {str(index) for index in range(len(points))}:
        raise ValueError("全带宽教师光谱必须完整包含2001点")
    for point in points:
        _validate_row(full_rows[str(point.index)], point)

    peak_rows = peak["rows"]
    peak_indices = sorted(int(key) for key in peak_rows)
    if any(str(index) not in peak_rows for index in peak_indices):
        raise ValueError("峰区教师光谱索引格式不规范")
    for index in peak_indices:
        _validate_row(peak_rows[str(index)], points[index])
    ranges = _contiguous_ranges(peak_indices)
    if len(ranges) != EXPECTED_PEAK_WINDOWS:
        raise ValueError(f"峰区教师光谱应包含9个连续窗口，实际为{len(ranges)}个")
    if any(stop - start + 1 <= 2 * blend_points for start, stop in ranges):
        raise ValueError("峰区窗口太短，无法完成无边缘伪峰的融合")

    full_signal = np.asarray(
        [float(full_rows[str(i)]["ch1_voltage_v"]) for i in range(len(points))],
        dtype=float,
    )
    overlap_low = full_signal[np.asarray(peak_indices, dtype=int)]
    overlap_high = np.asarray(
        [float(peak_rows[str(i)]["ch1_voltage_v"]) for i in peak_indices],
        dtype=float,
    )
    denominator = float(np.dot(overlap_low, overlap_low))
    if denominator <= 0.0:
        raise ValueError("全带宽教师光谱没有可用于跨阻换算的CH1信号")
    scale = float(np.dot(overlap_low, overlap_high) / denominator)
    if not math.isfinite(scale) or not 10.0 <= scale <= 30.0:
        raise ValueError(f"2 kOhm到40 kOhm的实测比例异常：{scale}")

    signal = full_signal * scale
    noise = np.asarray(
        [
            max(float(full_rows[str(i)].get("ch1_sigma_v", 0.0)) * scale, ADC_LSB_V)
            for i in range(len(points))
        ],
        dtype=float,
    )
    stable = np.asarray(
        [
            bool(full_rows[str(i)].get("stable"))
            and not bool(full_rows[str(i)].get("ch1_saturated"))
            for i in range(len(points))
        ],
        dtype=bool,
    )
    for start, stop in ranges:
        for index in range(start, stop + 1):
            edge_weight = min(
                1.0,
                (index - start) / float(blend_points),
                (stop - index) / float(blend_points),
            )
            row = peak_rows[str(index)]
            high_signal = float(row["ch1_voltage_v"])
            high_noise = max(float(row.get("ch1_sigma_v", 0.0)), ADC_LSB_V)
            signal[index] = (
                (1.0 - edge_weight) * signal[index] + edge_weight * high_signal
            )
            noise[index] = (
                (1.0 - edge_weight) * noise[index] + edge_weight * high_noise
            )
            stable[index] = stable[index] and bool(row.get("stable")) and not bool(
                row.get("ch1_saturated")
            )

    values = np.zeros((4, len(points)), dtype=float)
    values[1] = signal
    wavelength = np.asarray([point.measured_nm for point in points], dtype=float)
    report = app.select_mode_points_from_dense_spectrum(
        wavelength,
        values,
        points,
        expected_peaks=9,
        fbg_channels=(1,),
        stress_point_count=45,
        temperature_point_count=54,
        stress_minimum_per_peak=5,
        temperature_minimum_per_peak=6,
        # The user explicitly excludes tiny reflections.  Ten percent rejects
        # the measured 1535.10 nm parasitic feature while preserving all 9 FBGs.
        minimum_relative_amplitude=0.10,
        feedback_selectors=(0, 1),
        sample_noise_std=noise,
        stable_mask=stable,
    )
    if report.get("schema") != "equal_interval_auto_mode_selection_v4":
        raise AssertionError("没有生成严格九峰v4报告")
    report["teacher_spectrum"] = {
        "method": "audited_full_2kohm_plus_blended_peak_40kohm_v1",
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "full_teacher_path": str(full_path),
        "full_teacher_sha256": _sha256(full_path),
        "peak_teacher_path": str(peak_path),
        "peak_teacher_sha256": _sha256(peak_path),
        "operational_table_path": str(table_path.resolve()),
        "operational_table_sha256": table_sha256,
        "measured_2k_to_40k_scale": scale,
        "edge_blend_points": int(blend_points),
        "peak_windows": [[start, stop] for start, stop in ranges],
        "full_point_count": len(points),
        "high_snr_peak_point_count": len(peak_indices),
        "stable_composite_point_count": int(np.count_nonzero(stable)),
        "minimum_relative_amplitude": 0.10,
        "parasitic_tiny_reflections_excluded": True,
        "hardware_io_performed": False,
    }
    return report


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-teacher",
        type=Path,
        default=HERE / "artifacts" / "ch1_teacher_full_20260905_2k.json",
    )
    parser.add_argument(
        "--peak-teacher",
        type=Path,
        default=HERE / "artifacts" / "ch1_teacher_peaks_20260905_40k.json",
    )
    parser.add_argument("--blend-points", type=int, default=DEFAULT_BLEND_POINTS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    report = build_selection(
        args.full_teacher,
        args.peak_teacher,
        blend_points=args.blend_points,
    )
    output = args.output.resolve()
    _atomic_write(output, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "schema": report["schema"],
                "peaks": report["detected_peak_count"],
                "stress_points": report["stress"]["point_count"],
                "temperature_points": report["temperature"]["point_count"],
                "teacher_spectrum": report["teacher_spectrum"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
