"""Offline timing accounting, not a controller or a pressure-bandwidth proof.

The replay model describes the audited v1.0.31 C control flow: every DAC row
is visited, unchanged channels are skipped, except the hidden predecessor and
its target which each force five writes. No timing guard is changed here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics

from frame_telemetry import (
    decode_board_frame_schedule, decode_board_frame_timing,
    extract_raw_scan_frames,
)
from laser_dac_safety import validate_pi11210_codes
from verify_runtime_firmware import board_extension

PROFILES = {0: "MAP45", 1: "SURVEY9", 2: "TRACK13_WIDE", 3: "TRACK11_SINGLE"}


def timing_trace(frame: bytes) -> dict:
    """Retain existing trace keys plus original wire bytes and MCU point times.

    These point times bracket completed ADC/RC work, not precise conversion
    apertures. Native frames have no CRC; retaining bytes does not add one.
    """
    pending = bytearray(frame)
    if extract_raw_scan_frames(pending, max_points=45) != [frame] or pending:
        raise ValueError("invalid complete scan frame")
    if int.from_bytes(frame[2:4], "big") != 45:
        raise ValueError("timing audit requires 45 rows")
    extension = board_extension(frame)
    if extension is None:
        raise ValueError("missing board extension")
    schedule = decode_board_frame_schedule(extension)
    timing = decode_board_frame_timing(extension)
    if schedule is None or timing is None or schedule.version not in (4, 5):
        raise ValueError("missing current schedule/timing")
    indices = schedule.fresh_indices
    offsets = [schedule.sample_offset_us[index] for index in indices]
    if (not offsets or any(value is None for value in offsets)
            or any(value <= 0 or value > timing.acquisition_duration_us for value in offsets)
            or any(a >= b for a, b in zip(offsets, offsets[1:]))):
        raise ValueError("invalid fresh sample completion offsets")
    values = {str(index): int.from_bytes(frame[6 + index*8:8 + index*8], "big")
              for index in indices}
    if any(value > 4095 for value in values.values()):
        raise ValueError("CH1 code outside 12-bit range")
    return {
        "sequence": timing.sequence,
        "frame_start_ms": schedule.frame_start_ms,
        "schedule_version": schedule.version,
        "profile": PROFILES[schedule.profile],
        "primary_segment": schedule.primary_segment,
        "secondary_segment": schedule.secondary_segment,
        "fresh_ch1_codes": values,
        "firmware_version": int.from_bytes(extension[24:28], "big"),
        "active_channel_mask": extension[16],
        "feedback_selectors": list(extension[22:24]),
        "acquisition_duration_us": timing.acquisition_duration_us,
        "fresh_sample_completion_offset_us": dict(zip(map(str, indices), offsets)),
        "bandwidth_discontinuity": schedule.bandwidth_discontinuity,
        "wire_hex": frame.hex(),
    }


def replay_write_counts(rows, hidden, hidden_index, *, cold=False):
    rows = [validate_pi11210_codes(row) for row in rows]
    validate_pi11210_codes(hidden)
    if not rows or not 0 <= hidden_index < len(rows):
        raise ValueError("invalid hidden predecessor index")
    previous = None if cold else rows[-1]
    counts = []
    for index, row in enumerate(rows):
        counts.append(10 if index == hidden_index else
                      sum(previous is None or value != previous[channel]
                          for channel, value in enumerate(row)))
        previous = row
    return counts


def read_source_model(repo: Path) -> dict:
    paths = [repo / name for name in (
        "JDSU/Core/Src/ms5614t.c", "JDSU/Core/Src/main.c",
        "JDSU/Core/Src/stress_table.c", "JDSU/Core/Inc/stress_table.h",
        "Python/mode_tables_from_fullband_2001.json")]
    ms, main, table, header, desktop = [path.read_text("utf-8") for path in paths]

    def constant(text, name):
        match = re.search(r"#define\s+" + name + r"\s+(\d+)U?\b", text)
        if not match:
            raise ValueError(f"unrecognised constant: {name}")
        return int(match[1])

    def array(name, width):
        body = table.split("const uint16_t " + name, 1)[1].split("=", 1)[1].split("};", 1)[0]
        result = [tuple(map(int, re.findall(r"\d+", group)))
                  for group in re.findall(r"\{([^{}]+)\}", body)]
        if not result or any(len(row) != width for row in result):
            raise ValueError(f"invalid table: {name}")
        return result

    rows = array("Stress_Wave_DAC", 5)
    wavelengths = array("Stress_Wave_DATA", 2)
    hidden_body = table.split("const uint16_t Stress_Path_Precondition_DAC", 1)[1].split("=", 1)[1].split("}", 1)[0]
    hidden = tuple(map(int, re.findall(r"\d+", hidden_body)))
    desktop_rows = json.loads(desktop)["modes"]["stress"]["rows"]
    if (len(rows) != 45 or len(wavelengths) != 45
            or rows != [tuple(row["codes"]) for row in desktop_rows]):
        raise ValueError("desktop/firmware 45-row table mismatch")
    hidden_index = constant(header, "STRESS_PATH_PRECONDITION_POINT_INDEX")
    clock = int(re.search(r"hi2c1.Init.ClockSpeed\s*=\s*(\d+)", main)[1])
    if clock <= 0:
        raise ValueError("invalid I2C clock")
    # Nominal payload-only wire floor, NOT measured bus duration. The address,
    # 8-bit register pointer, and two data bytes each need eight bits + ACK.
    writes = replay_write_counts(rows, hidden, hidden_index)
    pm = [nm*1000 + fraction for nm, fraction in wavelengths]
    boundaries = [index for index in range(45)
                  if index == 0 or pm[index] - pm[index-1] < 0
                  or pm[index] - pm[index-1] > 500]
    return {
        "source_sha256": {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in paths},
        "model_scope": "audited_v1031_control_flow_nominal_floor_not_hardware_profile",
        "i2c_hz_configured": clock,
        "register_write_payload_clocks": 36,
        "nominal_register_write_wire_us": 36*1e6/clock,
        "steady_writes_per_row": writes,
        "steady_register_write_count": sum(writes),
        "cold_register_write_count": sum(replay_write_counts(rows, hidden, hidden_index, cold=True)),
        "nominal_all_dac_wire_floor_us": sum(writes)*36*1e6/clock,
        "hidden_index": hidden_index,
        "hidden_hold_us": constant(header, "STRESS_PATH_PRECONDITION_HOLD_US"),
        "first_delay_us": constant(ms, "FAST_ADC_FIRST_DELAY_US"),
        "pair_spacing_us": constant(ms, "FAST_ADC_SPACING_US"),
        "boundary_extra_us": constant(ms, "FAST_BOUNDARY_EXTRA_DELAY_US"),
        "boundary_indices": boundaries,
        "wave_time_us": None,
        "unaccounted": ["START/STOP/bus-free and software overhead", "ADC SPI time",
                        "conditional slow rechecks", "protection and radio maintenance",
                        "interrupts", "runtime wave_time (not in normal packet)",
                        "I2C retry/fault paths"],
    }


def audit_report(report: dict, model: dict) -> dict:
    trace = report.get("fresh_value_trace")
    if not trace:
        raise ValueError("trace unavailable")
    if report.get("point_count") != 45 or report.get("firmware_versions") != ["1.0.31"]:
        raise ValueError("source replay audit currently qualified only for v1.0.31 / 45 rows")
    if len(trace) != report["board_telemetry"]["available_frames"]:
        raise ValueError("partial frame coverage")
    decoded = []
    for row in trace:
        if "wire_hex" not in row:
            raise ValueError("original bytes absent; old summaries are not timing traces")
        restored = timing_trace(bytes.fromhex(row["wire_hex"]))
        if any(row.get(key) != value for key, value in restored.items()):
            raise ValueError("trace differs from original packet")
        if restored["firmware_version"] != 0x01001F or restored["active_channel_mask"] != 2:
            raise ValueError("wire firmware/channel differs from audited CH1 v1.0.31")
        if restored["feedback_selectors"][1] != report.get("feedback_selector"):
            raise ValueError("wire CH1 gain differs from report")
        if decoded and ((restored["sequence"] - decoded[-1]["sequence"]) & 0xFFFFFFFF) != 1:
            raise ValueError("non-contiguous frame sequence")
        decoded.append(restored)
    groups = defaultdict(list)
    intervals = defaultdict(list)
    for row in decoded:
        offsets = row["fresh_sample_completion_offset_us"]
        indices = [int(index) for index in offsets]
        delay = (len(indices)*(model["first_delay_us"] + model["pair_spacing_us"])
                 + len(set(indices) & set(model["boundary_indices"]))*model["boundary_extra_us"])
        floor = model["nominal_all_dac_wire_floor_us"] + model["hidden_hold_us"] + delay
        groups[row["profile"]].append((row["acquisition_duration_us"], delay, floor))
        previous_index, previous_time = -1, 0
        for index, elapsed in offsets.items():
            point = int(index)
            key = (row["profile"], previous_index, point)
            intervals[key].append(elapsed - previous_time)
            previous_index, previous_time = point, elapsed
    per_profile = {}
    for profile, values in groups.items():
        acquired, delays, floors = zip(*values)
        per_profile[profile] = {
            "frames": len(values), "acquisition_mean_ms": statistics.fmean(acquired)/1000,
            "acquisition_max_ms": max(acquired)/1000,
            "nominal_pair_wait_floor_mean_ms": statistics.fmean(delays)/1000,
            "nominal_accounted_floor_mean_ms": statistics.fmean(floors)/1000,
            "unattributed_mean_ms": statistics.fmean(a-f for a, f in zip(acquired, floors))/1000,
        }
    interval_audit = []
    for (profile, previous, point), values in sorted(intervals.items()):
        write_count = sum(model["steady_writes_per_row"][previous+1:point+1])
        hold = model["hidden_hold_us"] if previous < model["hidden_index"] <= point else 0
        wait = (model["first_delay_us"] + model["pair_spacing_us"]
                + (model["boundary_extra_us"] if point in model["boundary_indices"] else 0))
        floor = write_count*model["nominal_register_write_wire_us"] + hold + wait
        interval_audit.append({
            "profile": profile, "previous_fresh_index": previous, "fresh_index": point,
            "count": len(values), "mean_us": statistics.fmean(values), "max_us": max(values),
            "replayed_dac_rows": point-previous, "register_writes": write_count,
            "nominal_accounted_floor_us": floor,
            "unattributed_mean_us": statistics.fmean(values)-floor,
        })
    return {
        "schema": "stress_timing_budget_audit_v1", "source_model": model,
        "wire_redecoded_frames": len(decoded), "profiles": per_profile,
        "sample_completion_intervals": interval_audit,
        "reported_host_mean_minus_board_mean_ms": report.get("host_period_minus_board_acquisition_ms"),
        "host_difference_is_not_transport_latency": True,
        "physical_bandwidth_verified": False, "training_eligible": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    model = read_source_model(Path(__file__).resolve().parent.parent)
    result = audit_report(json.loads(args.source.read_text("utf-8")), model)
    result["source_file"] = str(args.source.resolve())
    result["source_sha256"] = hashlib.sha256(args.source.read_bytes()).hexdigest()
    result["analyzer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("source_model", "sample_completion_intervals")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
