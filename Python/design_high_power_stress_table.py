"""Design a 72-point stress scan from fresh high-power tuning maps."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from fbg_peak_fitting import build_segments, detect_wavelength_gaps, fit_channel_segments


HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "high_power_design_reference.json"
WAVE_YAML = HERE / "wave_const.yaml"
MAPS = (
    HERE / "high_power_tuning_map_mainpeak_v2.json",
    HERE / "high_power_tuning_map_gain105.json",
    HERE / "high_power_tuning_map_gain90.json",
)
OUTPUT = HERE / "high_power_stress_72_design_v3.json"

# Fresh high-current mapping showed that these two legacy extreme-left points
# sit beyond the stable longitudinal-mode branch.  Pulling them inward keeps
# the FBG slope/baseline observable without accepting a mode hop.
STABLE_START_OVERRIDES_NM = {
    1: 1531.741,
    2: 1536.081,
}
STABLE_STOP_OVERRIDES_NM = {
    0: 1528.347,
}


def target_points(start: float, center: float, stop: float) -> list[float]:
    left = max(center - start, 0.05)
    right = max(stop - center, 0.05)
    return [
        start,
        center - 0.65 * left,
        center - 0.35 * left,
        center - 0.10 * left,
        center + 0.10 * right,
        center + 0.35 * right,
        center + 0.65 * right,
        stop,
    ]


def main() -> int:
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    wave_rows = yaml.safe_load(WAVE_YAML.read_text(encoding="utf-8"))["Wave_DATA"]
    wavelengths = np.asarray([a + b / 1000.0 for a, b in wave_rows], dtype=float)
    ch1 = np.asarray(reference["median_voltage_waveforms_v"][1], dtype=float)
    segments = build_segments(len(wavelengths), detect_wavelength_gaps(wavelengths))
    fits = fit_channel_segments(
        wavelengths, ch1, segments, min_prominence_v=0.008, allow_edge_peak=True
    )

    candidates = []
    for source in MAPS:
        payload = json.loads(source.read_text(encoding="utf-8"))
        for value in payload["results"].values():
            reading = value["reading"]
            smsr = reading.get("side_mode_suppression_db")
            if smsr is None:
                smsr = 99.0
            candidates.append(
                {
                    "map": source.name,
                    "seed_index": int(value["seed_index"]),
                    "measured_nm": float(reading["wavelength_nm"]),
                    "power_mw": float(reading["power_mw"]),
                    "smsr_db": float(smsr),
                    "currents_ma": [float(item) for item in value["currents_ma"]],
                    "codes": [int(item) for item in value["codes"]],
                }
            )

    output_segments = []
    all_points = []
    for segment_index, (segment, fit) in enumerate(zip(segments, fits)):
        start = float(
            STABLE_START_OVERRIDES_NM.get(
                segment_index, float(wavelengths[segment.start])
            )
        )
        stop = float(
            STABLE_STOP_OVERRIDES_NM.get(
                segment_index, float(wavelengths[segment.stop - 1])
            )
        )
        center = float(fit.center_nm)
        targets = target_points(start, center, stop)
        points = []
        for target in targets:
            eligible = [
                item
                for item in candidates
                if start - 0.20 <= item["measured_nm"] <= stop + 0.20
                and item["power_mw"] >= 0.80
                and item["smsr_db"] >= 15.0
            ]
            eligible.sort(
                key=lambda item: (
                    abs(item["measured_nm"] - target),
                    -item["currents_ma"][0],
                    -item["power_mw"],
                )
            )
            if not eligible:
                raise RuntimeError(f"No safe high-power seed for {target:.6f} nm")
            seed = eligible[0]
            point = {
                "point_index": len(all_points),
                "segment": segment_index,
                "target_nm": float(target),
                "seed_error_pm": (seed["measured_nm"] - target) * 1000.0,
                **seed,
            }
            points.append(point)
            all_points.append(point)
        output_segments.append(
            {
                "segment": segment_index,
                "start_nm": start,
                "center_nm": center,
                "stop_nm": stop,
                "fit_sigma_nm": float(fit.sigma_nm),
                "fit_r_squared": float(fit.r_squared),
                "points": points,
            }
        )

    output = {
        "method": "nine_segments_eight_nonuniform_points_current_path_only",
        "point_count": len(all_points),
        "segments": output_segments,
        "points": all_points,
        "summary": {
            "max_abs_seed_error_pm": max(abs(p["seed_error_pm"]) for p in all_points),
            "p95_abs_seed_error_pm": float(
                np.percentile([abs(p["seed_error_pm"]) for p in all_points], 95)
            ),
            "seed_power_min_mw": min(p["power_mw"] for p in all_points),
            "seed_power_median_mw": float(
                np.median([p["power_mw"] for p in all_points])
            ),
        },
    }
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False), flush=True)
    for section in output_segments:
        print(
            f"segment={section['segment']} center={section['center_nm']:.6f} "
            f"targets={section['start_nm']:.3f}..{section['stop_nm']:.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
