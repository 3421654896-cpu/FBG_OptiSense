"""Bounded SOA optical on/off response test at one calibrated FBG peak.

Only the SOA hardware gate/current is switched.  Gain, phase and both sampled
grating currents stay at the same calibrated wavelength codes.  This separates
optical blanking from wavelength-table transitions as far as the existing
hardware permits.  The monitor stream is about 16 ms per frame, so results are
system-level bounds rather than nanosecond laser-chip timing measurements.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import serial

import app_JDSU as app
from benchmark_stress_realtime import _close_shutter, _safe_disarm
from probe_pdr_pdt_ch1_stability_live import CHANNELS, DEFAULT_SESSION, choose_peak


def capture_transition(device, worker, codes, soa_mode: int, hold_s: float) -> dict:
    device.reset_input_buffer()
    worker._rx_buffer.clear()
    command = app.build_single_value_dac_command(
        codes, soa_mode=soa_mode, fullband_2001=True
    )
    expected = tuple(
        (command[4 + channel * 2] << 8) | command[5 + channel * 2]
        for channel in range(5)
    )
    rx = bytearray()
    samples = []
    ack_s = None
    write_started = time.monotonic()
    device.write(command)
    write_finished = time.monotonic()
    deadline = write_finished + hold_s
    while time.monotonic() < deadline:
        waiting = int(getattr(device, "in_waiting", 0) or 0)
        data = device.read(min(max(waiting, app.SINGLE_VALUE_FRAME_SIZE), 4096))
        if not data:
            time.sleep(0.0005)
            continue
        rx.extend(data)
        arrived_s = time.monotonic() - write_finished
        for frame in app.extract_single_value_frames(rx):
            returned = worker._ack_codes(frame)
            if returned == expected and int(frame[14]) == int(soa_mode):
                worker._validate_ack_status(frame, soa_mode)
                if ack_s is None:
                    ack_s = arrived_s
            values = app.decode_single_value_monitor_frame(frame)
            if values is not None and all(0 <= value <= 4095 for value in values):
                samples.append({
                    "arrival_after_write_s": float(arrived_s),
                    **{name.lower(): int(value) for name, value in zip(CHANNELS, values)},
                })
    if ack_s is None:
        raise RuntimeError("SOA开关命令没有收到匹配状态回读")
    if len(samples) < 15:
        raise RuntimeError(f"SOA开关后的同步帧不足：{len(samples)}")
    return {
        "soa_mode": int(soa_mode),
        "write_call_duration_s": float(write_finished - write_started),
        "ack_after_write_s": float(ack_s),
        "samples": samples,
    }


def capture_held_monitor(device, duration_s: float) -> np.ndarray:
    """Return a settled-state monitor window without changing any DAC code."""
    device.reset_input_buffer()
    rx = bytearray()
    rows = []
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        waiting = int(getattr(device, "in_waiting", 0) or 0)
        data = device.read(min(max(waiting, app.SINGLE_VALUE_FRAME_SIZE), 4096))
        if not data:
            time.sleep(0.0005)
            continue
        rx.extend(data)
        for frame in app.extract_single_value_frames(rx):
            values = app.decode_single_value_monitor_frame(frame)
            if values is not None and all(0 <= value <= 4095 for value in values):
                rows.append([values[index] for index in (0, 1, 3)])
    if len(rows) < 5:
        raise RuntimeError("保持状态的同步监测帧不足")
    return np.asarray(rows, dtype=float)


def _first_true_run(flags: np.ndarray, times: np.ndarray, length: int = 3):
    run = 0
    for flag, at in zip(flags, times):
        run = run + 1 if bool(flag) else 0
        if run >= length:
            return float(at)
    return None


def transition_metrics(record: dict, initial: np.ndarray) -> dict:
    keys = ("pdt", "pdr", "ch1")
    values = np.asarray([[sample[key] for key in keys] for sample in record["samples"]], dtype=float)
    times = np.asarray([sample["arrival_after_write_s"] for sample in record["samples"]], dtype=float)
    tail_count = min(12, max(8, len(values) // 3))
    final = np.median(values[-tail_count:], axis=0)
    step = final - initial
    smoothed = np.asarray([
        np.median(values[max(0, index - 2):index + 1], axis=0)
        for index in range(len(values))
    ])
    result = {}
    for channel, key in enumerate(keys):
        span = abs(float(step[channel]))
        tolerance = max(8.0, 0.02 * span)
        settled = np.abs(smoothed[:, channel] - final[channel]) <= tolerance
        settle_s = _first_true_run(settled, times)
        if span >= 16.0:
            progress = (smoothed[:, channel] - initial[channel]) / step[channel]
            at10 = next((float(times[index]) for index, value in enumerate(progress) if value >= 0.10), None)
            at90 = next((float(times[index]) for index, value in enumerate(progress) if value >= 0.90), None)
        else:
            at10 = at90 = None
        result[key] = {
            "initial_median_code": float(initial[channel]),
            "final_tail_median_code": float(final[channel]),
            "step_codes": float(step[channel]),
            "settle_tolerance_codes": float(tolerance),
            "settle_s": settle_s,
            "t10_s": at10,
            "t90_s": at90,
            "t10_to_t90_s": None if at10 is None or at90 is None else float(at90 - at10),
        }
    result["sample_count"] = len(values)
    result["first_sample_s"] = float(times[0])
    result["tail_median"] = {key: float(value) for key, value in zip(keys, final)}
    return result


def summarize(records: list[dict]) -> dict:
    result = {}
    for kind in ("off", "on"):
        metrics = [record["metrics"] for record in records]
        result[kind] = {
            "repeats": len(records),
            "ack_median_ms": 1000.0 * statistics.median(record[kind]["ack_after_write_s"] for record in records),
            "first_sample_median_ms": 1000.0 * statistics.median(item[kind]["first_sample_s"] for item in metrics),
            "channels": {},
        }
        for channel in ("pdt", "pdr", "ch1"):
            settled = [1000.0 * item[kind][channel]["settle_s"] for item in metrics
                       if item[kind][channel]["settle_s"] is not None]
            durations = [1000.0 * item[kind][channel]["t10_to_t90_s"] for item in metrics
                         if item[kind][channel]["t10_to_t90_s"] is not None]
            steps = [item[kind][channel]["step_codes"] for item in metrics]
            result[kind]["channels"][channel] = {
                "settled_repeats": len(settled),
                "settle_median_ms": None if not settled else statistics.median(settled),
                "settle_p90_ms": None if not settled else float(np.percentile(settled, 90)),
                "settle_max_ms": None if not settled else max(settled),
                "t10_to_t90_median_ms": None if not durations else statistics.median(durations),
                "step_median_codes": statistics.median(steps),
            }
    periods = []
    for record in records:
        for kind in ("off", "on"):
            times = np.asarray([sample["arrival_after_write_s"] for sample in record[kind]["samples"]])
            periods.extend(np.diff(times)[np.diff(times) > 0].tolist())
    result["monitor_period_median_ms"] = 1000.0 * statistics.median(periods)
    result["resolution_note"] = "response faster than one monitor period is reported only as an upper bound"
    return result


def plot_report(report: dict, output: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(11.5, 7.8), layout="constrained")
    for axis, kind, title in zip(axes, ("off", "on"), ("SOA source to shutter", "SOA shutter to source")):
        records = report["records"]
        minimum = min(len(record[kind]["samples"]) for record in records)
        times = np.median([[sample["arrival_after_write_s"] for sample in record[kind]["samples"][:minimum]]
                           for record in records], axis=0) * 1000.0
        arrays = np.asarray([[[sample[key] for key in ("pdt", "pdr", "ch1")]
                              for sample in record[kind]["samples"][:minimum]]
                             for record in records], dtype=float)
        median = np.median(arrays, axis=0)
        low = np.percentile(arrays, 10, axis=0)
        high = np.percentile(arrays, 90, axis=0)
        for index, (name, color) in enumerate(zip(("PDT", "PDR", "CH1"),
                                                   ("#3178c6", "#2b9a66", "#d74b57"))):
            axis.plot(times, median[:, index], color=color, label=name)
            axis.fill_between(times, low[:, index], high[:, index], color=color, alpha=0.12)
        axis.set(xlabel="Host arrival after SOA command (ms)", ylabel="ADC code",
                 title=f"{title}: median and 10–90% across {len(records)} trials")
        axis.grid(alpha=0.2)
        axis.legend()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--group", type=int, choices=range(1, 10))
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--off-hold-s", type=float, default=0.35)
    parser.add_argument("--on-hold-s", type=float, default=0.60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 5 <= args.repeats <= 50:
        raise ValueError("repeats must be 5..50")
    if not 0.25 <= args.off_hold_s <= 2.0 or not 0.30 <= args.on_hold_s <= 2.0:
        raise ValueError("hold time outside bounded range")
    if args.output.exists() or args.output.with_suffix(".png").exists():
        raise FileExistsError(args.output)

    choice = choose_peak(args.session, args.group)
    report = {
        "schema": "soa_on_off_fbg_response_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": False,
        "scope": "same wavelength codes; only SOA hardware shutter/source state changes",
        "timing_scope": "host frame arrival; monitor period limits time resolution",
        "selection": choice,
        "requested_repeats": args.repeats,
        "off_hold_s": args.off_hold_s,
        "on_hold_s": args.on_hold_s,
        "records": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = None
    try:
        device = serial.Serial(args.port, 2_000_000, timeout=0.005, write_timeout=3)
        device.dtr = True
        if not _safe_disarm(device) or not _close_shutter(device):
            raise RuntimeError("测试前停扫/关光回读未确认")
        worker = app.EqualIntervalWorker(device, (), {}, settle_s=0.0,
                                         feedback_selectors=(0, 2))
        if not worker._feedback_command_and_ack():
            raise RuntimeError("CH1 5 kΩ跨阻状态回读未确认")
        if worker._command_and_ack(choice["target"]["codes"]) is None:
            raise RuntimeError("初始峰值点写入失败")
        if not worker._interruptible_wait(0.50):
            raise RuntimeError("初始峰值稳定等待被中断")
        held_on = capture_held_monitor(device, 0.20)
        previous_on = np.median(held_on[-min(12, len(held_on)):], axis=0)
        for repeat in range(1, args.repeats + 1):
            off = capture_transition(
                device, worker, choice["target"]["codes"],
                app.PI11210_SOA_SHUTTER_MODE, args.off_hold_s,
            )
            off_values = np.asarray([[sample[key] for key in ("pdt", "pdr", "ch1")]
                                     for sample in off["samples"]], dtype=float)
            off_tail = np.median(off_values[-min(12, len(off_values)):], axis=0)
            off_metrics = transition_metrics(off, previous_on)

            on = capture_transition(
                device, worker, choice["target"]["codes"],
                app.PI11210_SOA_SOURCE_MODE, args.on_hold_s,
            )
            on_metrics = transition_metrics(on, off_tail)
            previous_on = np.asarray([on_metrics[key]["final_tail_median_code"]
                                      for key in ("pdt", "pdr", "ch1")], dtype=float)
            report["records"].append({
                "repeat": repeat,
                "off": off,
                "on": on,
                "metrics": {"off": off_metrics, "on": on_metrics},
            })
            print(f"repeat={repeat}/{args.repeats} off_frames={len(off['samples'])} "
                  f"on_frames={len(on['samples'])}", flush=True)
        report["analysis"] = summarize(report["records"])
        report["complete"] = True
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        disarmed = shuttered = False
        errors = []
        if device is not None:
            try:
                disarmed = bool(_safe_disarm(device))
            except Exception as exc:
                errors.append(f"disarm: {exc}")
            try:
                shuttered = bool(_close_shutter(device))
            except Exception as exc:
                errors.append(f"shutter: {exc}")
            device.close()
        report["safety_cleanup"] = {
            "exact_disarm_ack": disarmed,
            "exact_soa_shutter_ack": shuttered,
            "errors": errors,
        }
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"disarm_confirmed={int(disarmed)} soa_shutter_confirmed={int(shuttered)}", flush=True)
    if not disarmed or not shuttered:
        raise RuntimeError("测试结束后的停扫/关光未确认")
    plot_report(report, args.output.with_suffix(".png"))
    print(json.dumps({
        "selection": {key: choice[key] for key in ("group", "target", "target_reference_adc")},
        "analysis": report["analysis"],
        "output": str(args.output.resolve()),
        "plot": str(args.output.with_suffix('.png').resolve()),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
