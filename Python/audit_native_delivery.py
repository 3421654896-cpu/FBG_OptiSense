"""Compare board sampling continuity with desktop native-frame arrival timing."""
import argparse
import json
from pathlib import Path

import numpy as np

from frame_telemetry import decode_board_frame_schedule, decode_board_frame_timing, sentinel_offsets_for_version
from qualify_position_coverage_session import read_jsonl, sha256
from verify_runtime_firmware import board_extension


def audit(path):
    previous = None
    gaps, starts, sequence_breaks, outliers = [], [], [], []
    fresh_times = [[] for _ in range(9)]
    unwrapped_start_ms = None
    profiles = {}
    for record in read_jsonl(path):
        extension = board_extension(bytes.fromhex(record["payload_hex"]))
        timing = decode_board_frame_timing(extension)
        schedule = decode_board_frame_schedule(extension)
        if timing is None or schedule is None:
            raise ValueError("Missing board timing")
        item = (record["host_received_monotonic_ns"], timing.sequence,
                schedule.frame_start_ms, record["record_number"])
        if item[2] is None:
            raise ValueError("Missing board frame start")
        if unwrapped_start_ms is None:
            unwrapped_start_ms = item[2]
        else:
            unwrapped_start_ms += (item[2] - previous[2]) & 0xFFFFFFFF
        profiles[str(schedule.profile)] = profiles.get(str(schedule.profile), 0) + 1
        for g, offset in enumerate(sentinel_offsets_for_version(schedule.version)):
            index = 5*g+offset
            if index in schedule.fresh_indices and schedule.sample_offset_us[index] is not None:
                fresh_times[g].append(unwrapped_start_ms + schedule.sample_offset_us[index]/1000)
        if previous is not None:
            gap = (item[0] - previous[0]) / 1e6
            board_gap = (item[2] - previous[2]) & 0xFFFFFFFF
            delta = (item[1] - previous[1]) & 0xFFFFFFFF
            if gap < 0:
                raise ValueError("Host time moved backwards")
            gaps.append(gap)
            starts.append(board_gap)
            if delta != 1:
                sequence_breaks.append({"record": item[3], "sequence_delta": delta})
            if gap >= 500:
                outliers.append({"record": item[3], "host_gap_ms": gap,
                                 "board_start_gap_ms": board_gap, "sequence_delta": delta})
        previous = item
    if not gaps:
        raise ValueError("Insufficient frames")
    per_grating = {}
    for g, times in enumerate(fresh_times, 1):
        periods = np.diff(times)
        per_grating[f"G{g}"] = {"samples": len(times),
            "mean_fresh_hz": float(1000/np.mean(periods)) if len(periods) and np.mean(periods)>0 else None,
            "max_gap_ms": float(max(periods)) if len(periods) else None,
            "p99_gap_ms": float(np.percentile(periods, 99)) if len(periods) else None,
            "gaps_over_15hz_nyquist_interval": int(np.sum(periods > 1000/30))}
    return {"source": str(path.resolve()), "sha256": sha256(path), "frames": len(gaps)+1,
            "profiles": profiles, "per_grating_fresh_timing": per_grating,
            "host_gap_ms_max": max(gaps), "host_gap_ms_p99": float(np.percentile(gaps, 99)),
            "equal_host_timestamps": int(sum(gap == 0 for gap in gaps)),
            "board_start_gap_ms_max": max(starts), "board_start_gap_ms_p99": float(np.percentile(starts, 99)),
            "sequence_breaks": sequence_breaks, "host_gaps_over_500ms": outliers,
            "physical_15hz_verified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.native_file)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
