"""Memory-bounded live F45 transport for the integrated 3-D application.

Unlike the qualification capture, this path intentionally does not retain all
45x2 ADC records for a long session. It validates every frame, publishes it to
the causal localizer, keeps only cadence statistics, and always performs exact
DAC disarm plus SOA shutter cleanup.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

import app_JDSU as app
from benchmark_stress_realtime import _close_shutter, _safe_disarm
import capture_fast_fullmap_stream as fullmap
from capture_ch1_rc_diagnostic import load_installed_rows
from capture_ch1_teacher_spectrum import save_checkpoint, sha256_file
from capture_short_route_teacher import codes_crc


def execute_realtime_device(
    device,
    output: Path | str,
    *,
    cycles: int = 60_000,
    spacing_us: int = 50,
    on_frame=None,
    should_stop=None,
):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows, _metadata = load_installed_rows()
    table_crc32 = codes_crc(rows)
    report = {
        "schema": "ch1_fast_fullmap_live_summary_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": False,
        "cycles_requested": int(cycles),
        "spacing_us": int(spacing_us),
        "route_rows": list(fullmap.ROWS),
        "table_codes_crc32": table_crc32,
        "table_sha256": sha256_file(fullmap.HERE / "mode_tables_from_fullband_2001.json"),
        "frames_retained": 0,
        "frame_count": 0,
        "training_eligible": False,
        "physical_15hz_verified": False,
        "physical_contact_area_mm2": None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(output, report)
    terminal = None
    intervals: list[int] = []
    previous_start = None
    frame_count = 0
    try:
        device.dtr = True
        if not _safe_disarm(device) or not _close_shutter(device):
            raise RuntimeError("initial exact disarm/shutter ACK missing")
        worker = app.EqualIntervalWorker(
            device, (), {}, settle_s=0.0, feedback_selectors=(0, 2)
        )
        if not worker._feedback_command_and_ack() or not worker._interruptible_wait(0.01):
            raise RuntimeError("fixed feedback ACK/cancellation")
        tag = int(time.time_ns() & 0x7FFFFFFF) or 1
        device.reset_input_buffer()
        device.write(fullmap.command(int(cycles), tag, int(spacing_us)))
        device.flush()

        buffer = bytearray()
        deadline = time.monotonic() + max(15.0, float(cycles) * 0.04 + 60.0)
        while time.monotonic() < deadline and terminal is None:
            if should_stop is not None and should_stop():
                raise InterruptedError("live F45 stop requested")
            buffer.extend(device.read(8192))
            for wire in fullmap.extract_frames(buffer):
                if wire[:4] == b"\xd9\x9d\x03\x7f":
                    if int.from_bytes(wire[4:8], "big") == tag:
                        raise fullmap.StreamRejected(wire)
                    continue
                if wire[2] == 3:
                    item = fullmap.decode_data(
                        wire,
                        tag=tag,
                        table_crc32=table_crc32,
                        spacing_us=int(spacing_us),
                    )
                    if int(item["sequence"]) != frame_count:
                        raise ValueError("missing, duplicate or reordered live F45 cycle")
                    start = int(item["cycle_start_us"])
                    if previous_start is not None:
                        interval = start - previous_start
                        if interval <= 0:
                            raise ValueError("live F45 board time is not monotonic")
                        if bool(item["minimum_warmup_complete"]):
                            intervals.append(interval)
                    previous_start = start
                    if item["i2c_error_delta"] or item["spi_error_delta"]:
                        raise RuntimeError("board I2C/SPI error during live F45 stream")
                    frame_count += 1
                    report["frame_count"] = frame_count
                    if on_frame is not None:
                        on_frame(item)
                    if frame_count % 1024 == 0:
                        save_checkpoint(output, report)
                else:
                    terminal = fullmap.flank.decode_end(
                        wire, tag=tag, table_crc32=table_crc32
                    )
        if terminal is None:
            raise TimeoutError(f"live F45 terminal missing after {frame_count} cycles")
        if (
            terminal["status"] != 0
            or terminal["stop_reason"] != 0
            or not terminal["firmware_shutter_asserted"]
            or terminal["requested_cycles"] != int(cycles)
            or terminal["completed_cycles"] != int(cycles)
            or frame_count != int(cycles)
            or terminal["i2c_error_delta"]
            or terminal["spi_error_delta"]
        ):
            raise RuntimeError(f"live F45 did not finish cleanly: {terminal}")
        if len(intervals) < 2:
            raise RuntimeError("too few post-warm-up intervals")
        report["terminal"] = terminal
        report["analysis"] = {
            "total_cycles": frame_count,
            "board_interval_mean_us": statistics.fmean(intervals),
            "board_interval_median_us": statistics.median(intervals),
            "board_interval_p95_us": sorted(intervals)[
                round(0.95 * (len(intervals) - 1))
            ],
            "board_interval_max_us": max(intervals),
            "board_rate_hz": 1_000_000.0 / statistics.fmean(intervals),
            "meets_sampling_rate_for_15hz": max(intervals) <= 1_000_000.0 / 30.0,
            "physical_15hz_verified": False,
            "contact_area_verified": False,
        }
        report["complete"] = True
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        errors = []
        try:
            disarmed = _safe_disarm(device)
        except Exception as exc:
            disarmed = False
            errors.append(f"disarm:{exc}")
        try:
            shuttered = _close_shutter(device)
        except Exception as exc:
            shuttered = False
            errors.append(f"shutter:{exc}")
        report["frame_count"] = frame_count
        report["safety_cleanup"] = {
            "exact_disarm_ack": disarmed,
            "exact_soa_shutter_ack": shuttered,
            "errors": errors,
        }
        save_checkpoint(output, report)
        if not disarmed or not shuttered:
            raise RuntimeError("exact final optical shutdown not confirmed")
    return report


__all__ = ["execute_realtime_device"]
