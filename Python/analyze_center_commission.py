"""Offline commissioning diagnostics; never trains or commands hardware."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def analyze(path: Path):
    records = [json.loads(line) for line in (path / "session/frames.jsonl").read_text("utf-8").splitlines()]
    events = [json.loads(line) for line in (path / "session/events.jsonl").read_text("utf-8").splitlines()]
    start = next(e["monotonic_ns"] for e in events if e["kind"] == "pre_contact")
    contact = next(e for e in events if e["kind"] == "contact")
    release = next(e for e in events if e["kind"] == "release")
    time_ns = np.array([r["frame_received_monotonic_ns"] for r in records], dtype=np.int64)
    captured = np.array([r["monotonic_ns"] for r in records], dtype=np.int64)
    codes = np.array([r["raw_adc_codes"]["1"] for r in records], dtype=float)
    fresh = np.array([r["fresh_point_mask"] for r in records], dtype=bool)
    ages = np.array([r["point_age_frames"] for r in records], dtype=int)
    if codes.shape[1] != 45 or any(r["transimpedance_ohm"]["1"] != 5000 for r in records):
        raise ValueError("Expected this commission's fixed 45-point CH1 / 5 kohm acquisition")
    time_ok = (captured >= time_ns) & (captured - time_ns < 500_000_000)
    # Only initial Z-up baseline, never the later downward approach.
    baseline = (time_ns >= start) & (time_ns < start + 3_000_000_000) & time_ok
    full = fresh.all(axis=1) & (ages == 0).all(axis=1)
    baseline &= full
    held = (time_ns >= contact["monotonic_ns"] + 200_000_000) & (time_ns < release["monotonic_ns"]) & time_ok
    if baseline.sum() < 3 or held.sum() < 3:
        raise ValueError("Insufficient initial baseline or settled contact frames")
    base = np.median(codes[baseline], axis=0)
    mad = np.median(np.abs(codes[baseline] - base), axis=0) * 1.4826
    held_median, held_counts = [], []
    for index in range(45):
        selected = held & fresh[:, index] & (ages[:, index] == 0)
        held_counts.append(int(selected.sum()))
        held_median.append(float(np.median(codes[selected, index])) if selected.any() else None)
    point_change = [None if b is None else float(b - a) for a, b in zip(base, held_median)]
    seq = np.array([r["device_sequence"] for r in records], dtype=np.int64)
    seq_delta = (np.diff(seq) % (2**32))
    rx_delta = np.diff(time_ns) / 1e9
    span = (time_ns[-1] - time_ns[0]) / 1e9
    result = {
        "interpretation": "one static contact, not 15 Hz pressure/position/area validation",
        "adc_value_kinds": sorted({r["adc_value_kind"] for r in records}),
        "saved_frames": len(records), "span_s": float(span),
        "saved_frame_rate_hz": float((len(records) - 1) / span),
        "device_sequence_span_frames": int((seq[-1] - seq[0]) % (2**32)) + 1,
        "missing_device_sequences_in_saved_file": int(np.maximum(seq_delta - 1, 0).sum()),
        "duplicate_device_sequences": int((seq_delta == 0).sum()),
        "profile_counts": dict(Counter(r["acquisition_profile"] for r in records)),
        "saved_receive_gap_ms_p50_p95_max": (np.percentile(rx_delta, [50, 95, 100]) * 1000).tolist(),
        "host_receive_to_capture_ms_p50_p95_max": np.percentile((captured-time_ns)/1e6, [50, 95, 100]).tolist(),
        "initial_full_map_baseline_frames": int(baseline.sum()),
        "settled_contact_frames": int(held.sum()), "settled_contact_full_maps": int((held & full).sum()),
        "contact_endpoint_xyz_mm": contact["actual_xyz_mm"],
        "endpoint_hold_s": (release["monotonic_ns"] - contact["monotonic_ns"])/1e9,
        "baseline_codes": base.tolist(), "baseline_mad_sigma_codes": mad.tolist(),
        "held_fresh_point_median_codes": held_median, "held_fresh_point_counts": held_counts,
        "held_minus_baseline_codes": point_change,
        "note": "Held point medians pool different acquisition times; missing positions remain null. Sequence gaps do not identify whether loss occurred in UI, USB or firmware. No claim of physical response bandwidth or measured contact area.",
    }
    (path / "signal_diagnostics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), "utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(12, 8), layout="constrained")
    wavelengths = np.array(records[0]["wavelengths_nm"])
    for i, ax in enumerate(axes.flat):
        s = slice(i*5, i*5+5)
        ax.plot(wavelengths[s], base[s], "o-", label="Initial baseline (full MAP)", color="#2574bf")
        ax.plot(wavelengths[s], np.array(held_median, dtype=float)[s], "s-", label="Contact (fresh-point medians)", color="#de6b35")
        ax.set_title(f"FBG {i+1}")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Firmware RC estimate (ADC code)")
        ax.ticklabel_format(useOffset=False, axis="x")
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Centre contact: actual endpoint Z=-0.0984 mm; fixed CH1 5 kohm\nStatic diagnostic only; contact medians are not simultaneous spectra")
    fig.savefig(path / "nine_fbg_contact_diagnostic.png", dpi=140)
    plt.close(fig)
    print(json.dumps({k: v for k,v in result.items() if k not in {"baseline_codes", "baseline_mad_sigma_codes", "held_fresh_point_median_codes", "held_minus_baseline_codes"}}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    analyze(parser.parse_args().capture)
