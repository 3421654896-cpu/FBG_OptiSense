"""Build untested backup candidates for failed hybrid-table targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from optimize_fbg_sampling_table import load_candidates, write_candidate_c, ACTIVE_DAC_C


HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--alternatives", type=int, default=4)
    parser.add_argument("--only-index", type=int, default=None)
    parser.add_argument("--exclude-json", type=Path, default=None)
    parser.add_argument("--output-prefix", default="fbg_replacement")
    parser.add_argument("--window-nm", type=float, default=0.060)
    args = parser.parse_args()

    source_candidates = load_candidates(args.source_json)
    primary = json.loads((HERE / "fbg_sampling_selection.json").read_text(encoding="utf-8"))
    used_excel_rows = {
        int(point["source_excel_row"])
        for segment in primary["segments"]
        for point in segment["points"]
    }
    if args.exclude_json is not None and args.exclude_json.exists():
        excluded = json.loads(args.exclude_json.read_text(encoding="utf-8"))
        used_excel_rows.update(
            int(item["source_excel_row"])
            for group in excluded.get("groups", [])
            for item in group.get("candidates", [])
        )
    hybrid_report = json.loads(
        (HERE / "fbg_hybrid_calibration_report.json").read_text(encoding="utf-8")
    )
    failed = [
        result
        for result in hybrid_report["results"].values()
        if not result.get("success")
    ]
    failed.sort(key=lambda result: int(result["index"]))
    if args.only_index is not None:
        failed = [result for result in failed if int(result["index"]) == args.only_index]

    chosen = []
    groups = []
    for result in failed:
        target = float(result["target_nm"])
        available = [
            item
            for item in source_candidates
            if item.source_excel_row not in used_excel_rows
            and abs(item.wavelength_nm - target) <= args.window_nm
        ]
        available.sort(
            key=lambda item: (
                abs(item.wavelength_nm - target),
                item.repeat_error_pm,
                item.secondary_power_mw / max(item.historical_power_mw, 1e-6),
            )
        )
        alternatives = available[: args.alternatives]
        if len(alternatives) < args.alternatives:
            raise RuntimeError(f"Not enough backup candidates around {target:.3f} nm")
        start = len(chosen)
        chosen.extend(alternatives)
        used_excel_rows.update(item.source_excel_row for item in alternatives)
        groups.append(
            {
                "hybrid_index": int(result["index"]),
                "failed_target_nm": target,
                "replacement_indices": list(range(start, len(chosen))),
                "candidates": [
                    {
                        "replacement_index": start + offset,
                        "source_excel_row": item.source_excel_row,
                        "wavelength_nm": item.wavelength_nm,
                        "repeat_error_pm": item.repeat_error_pm,
                        "historical_power_mw": item.historical_power_mw,
                        "codes": list(item.codes),
                    }
                    for offset, item in enumerate(alternatives)
                ],
            }
        )

    (HERE / f"{args.output_prefix}_candidates.json").write_text(
        json.dumps({"groups": groups}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    wave_rows = []
    for item in chosen:
        rounded = round(item.wavelength_nm, 3)
        major = int(rounded)
        minor = int(round((rounded - major) * 1000.0))
        wave_rows.append([major, minor])
    (HERE / f"wave_const_{args.output_prefix}.yaml").write_text(
        yaml.safe_dump({"Wave_DATA": wave_rows}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    write_candidate_c(
        ACTIVE_DAC_C,
        [list(item.codes) for item in chosen],
        HERE / f"dac_const_{args.output_prefix}_uncalibrated.c",
    )
    print(f"failed_targets={len(failed)} replacement_candidates={len(chosen)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
