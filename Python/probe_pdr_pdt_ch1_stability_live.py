"""Measure whether PDT/PDR settling predicts FBG CH1 intensity settling.

The probe uses one peak from the last temporary-mode 45-point session.  Each
trial holds a lower-wavelength point in the same grating and then makes one
forward transition to the measured peak point.  PDT, PDR and CH0..CH3 are read
from the same firmware monitor frame.  Host arrival time is useful for settling
order, but is not claimed to be the ADC conversion timestamp.

The laser is disarmed and its SOA shutter is verified both before and after the
bounded experiment.  No table, firmware or calibration file is modified.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
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


HERE = Path(__file__).resolve().parent
DEFAULT_SESSION = HERE / "outputs" / "temporary_test" / "last_45_session.json"
CHANNELS = ("PDT", "PDR", "CH0", "CH1", "CH2", "CH3")


def _robust_metrics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    median = np.median(values, axis=0)
    sigma = 1.4826 * np.median(np.abs(values - median), axis=0)
    span = max(3, len(values) // 3)
    drift = np.abs(np.median(values[-span:], axis=0) - np.median(values[:span], axis=0))
    return median, sigma, drift


def _quiet(median: np.ndarray, sigma: np.ndarray, drift: np.ndarray) -> np.ndarray:
    mad_limit = np.maximum(app.FULLBAND_ACCURACY_MAD_FLOOR_CODES,
                           0.003 * np.maximum(64.0, median))
    drift_limit = np.maximum(app.FULLBAND_ACCURACY_DRIFT_FLOOR_CODES,
                             0.004 * np.maximum(64.0, median))
    return (sigma <= mad_limit) & (drift <= drift_limit)


def _first_true_run(flags: list[bool], times: list[float], length: int = 3) -> float | None:
    run = 0
    for flag, at in zip(flags, times):
        run = run + 1 if flag else 0
        if run >= length:
            return float(at)
    return None


def choose_peak(session_path: Path, requested_group: int | None = None) -> dict:
    payload = json.loads(session_path.read_text(encoding="utf-8"))
    rows = payload.get("plan", {}).get("rows", [])
    references = payload.get("reference_values")
    if len(rows) != 45 or not isinstance(references, list) or len(references) != 45:
        raise RuntimeError("上次临时模式会话没有完整的45点参考线")

    checks = {
        int(item.get("group", 0)): bool(item.get("passed"))
        for item in payload.get("plan", {}).get("live_reference_fit_checks", [])
    }
    table = tuple(app.load_fullband_accuracy_table())
    candidates = []
    for group in range(1, 10):
        start = (group - 1) * 5
        group_values = [float(value) for value in references[start:start + 5]]
        peak_local = int(np.argmax(group_values))
        if peak_local == 0:
            continue
        if requested_group is not None and group != requested_group:
            continue
        if requested_group is None and checks and not checks.get(group, False):
            continue
        target = dict(rows[start + peak_local])
        source = dict(rows[start])
        for selected in (source, target):
            index = int(selected["index"])
            point = table[index]
            if (tuple(int(value) for value in selected["codes"]) != tuple(point.codes)
                    or abs(float(selected["measured_nm"]) - float(point.measured_nm)) > 1e-9):
                raise RuntimeError("45点会话与当前2001点表不一致")
        candidates.append({
            "group": group,
            "source": source,
            "target": target,
            "reference_values": group_values,
            "target_reference_adc": group_values[peak_local],
            "peak_local_point": peak_local + 1,
        })
    if not candidates:
        raise RuntimeError("没有可用于试验的已通过峰")
    return max(candidates, key=lambda item: item["target_reference_adc"])


def capture_forward_transition(device, worker, target_codes, hold_s: float) -> dict:
    device.reset_input_buffer()
    worker._rx_buffer.clear()
    command = app.build_single_value_dac_command(
        target_codes, soa_mode=app.PI11210_SOA_SOURCE_MODE, fullband_2001=True
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
            if returned == expected:
                worker._validate_ack_status(frame, app.PI11210_SOA_SOURCE_MODE)
                if ack_s is None:
                    ack_s = arrived_s
            values = app.decode_single_value_monitor_frame(frame)
            if values is not None and all(0 <= value <= 4095 for value in values):
                samples.append({
                    "arrival_after_write_s": float(arrived_s),
                    **{name.lower(): int(value) for name, value in zip(CHANNELS, values)},
                })
    if ack_s is None:
        raise RuntimeError("目标峰DAC写入没有收到匹配状态回读")
    if len(samples) < 25:
        raise RuntimeError(f"目标峰同步监测帧不足：{len(samples)}")
    return {
        "write_call_duration_s": float(write_finished - write_started),
        "ack_after_write_s": float(ack_s),
        "samples": samples,
    }


def analyze_trials(trials: list[dict], window: int = 15) -> dict:
    windows = []
    trial_results = []
    sample_periods = []
    for trial_index, trial in enumerate(trials, start=1):
        samples = trial["samples"]
        values = np.asarray([[sample[name.lower()] for name in CHANNELS] for sample in samples], dtype=float)
        times = np.asarray([sample["arrival_after_write_s"] for sample in samples], dtype=float)
        positive_periods = np.diff(times)
        positive_periods = positive_periods[positive_periods > 0]
        sample_periods.extend(positive_periods.tolist())
        tail_count = max(window, min(30, len(values) // 3))
        tail = values[-tail_count:]
        tail_median, tail_sigma, tail_drift = _robust_metrics(tail)
        proximity = np.maximum(
            np.asarray([8.0, 8.0, 8.0, 8.0, 8.0, 8.0]),
            np.maximum(3.0 * tail_sigma, 0.01 * np.maximum(64.0, np.abs(tail_median))),
        )
        pd_flags, ch1_flags, window_times = [], [], []
        for end in range(window - 1, len(values)):
            subset = values[end - window + 1:end + 1]
            median, sigma, drift = _robust_metrics(subset)
            quiet = _quiet(median, sigma, drift)
            close = np.abs(median - tail_median) <= proximity
            pd_stable = bool(np.all(quiet[:2] & close[:2]))
            ch1_stable = bool(quiet[3] and close[3])
            record = {
                "trial": trial_index,
                "window_end_sample": end,
                "window_end_s": float(times[end]),
                "pd_stable": pd_stable,
                "ch1_stable": ch1_stable,
                "pdt_median": float(median[0]),
                "pdr_median": float(median[1]),
                "ch1_median": float(median[3]),
                "pdt_sigma": float(sigma[0]),
                "pdr_sigma": float(sigma[1]),
                "ch1_sigma": float(sigma[3]),
                "ch1_error_from_tail_codes": float(abs(median[3] - tail_median[3])),
            }
            windows.append(record)
            pd_flags.append(pd_stable)
            ch1_flags.append(ch1_stable)
            window_times.append(float(times[end]))
        trial_results.append({
            "trial": trial_index,
            "sample_count": len(samples),
            "tail_sample_count": tail_count,
            "tail_median": {name.lower(): float(value) for name, value in zip(CHANNELS, tail_median)},
            "tail_sigma": {name.lower(): float(value) for name, value in zip(CHANNELS, tail_sigma)},
            "tail_drift": {name.lower(): float(value) for name, value in zip(CHANNELS, tail_drift)},
            "pd_stable_onset_s": _first_true_run(pd_flags, window_times),
            "ch1_stable_onset_s": _first_true_run(ch1_flags, window_times),
        })

    pd_stable_windows = [item for item in windows if item["pd_stable"]]
    pd_unstable_windows = [item for item in windows if not item["pd_stable"]]

    def fraction(items, predicate):
        return None if not items else sum(bool(predicate(item)) for item in items) / len(items)

    def median_or_none(items, key):
        return None if not items else float(statistics.median(item[key] for item in items))

    pd_onsets = [item["pd_stable_onset_s"] for item in trial_results
                 if item["pd_stable_onset_s"] is not None]
    ch1_onsets = [item["ch1_stable_onset_s"] for item in trial_results
                  if item["ch1_stable_onset_s"] is not None]
    paired_onset_deltas = [
        item["ch1_stable_onset_s"] - item["pd_stable_onset_s"]
        for item in trial_results
        if item["pd_stable_onset_s"] is not None and item["ch1_stable_onset_s"] is not None
    ]
    return {
        "definition": {
            "window_samples": window,
            "quiet_gate": "same robust MAD/drift limits as 2001-point acquisition",
            "convergence_gate": "rolling median within max(8 codes, 3*tail sigma, 1% tail code)",
            "stable_onset": "first three consecutive stable windows",
        },
        "sample_period_median_ms": (
            None if not sample_periods else 1000.0 * statistics.median(sample_periods)
        ),
        "window_count": len(windows),
        "pd_stable_window_count": len(pd_stable_windows),
        "pd_unstable_window_count": len(pd_unstable_windows),
        "when_pd_stable_ch1_stable_fraction": fraction(
            pd_stable_windows, lambda item: item["ch1_stable"]
        ),
        "when_pd_unstable_ch1_unstable_fraction": fraction(
            pd_unstable_windows, lambda item: not item["ch1_stable"]
        ),
        "ch1_error_codes_when_pd_stable_median": median_or_none(
            pd_stable_windows, "ch1_error_from_tail_codes"
        ),
        "ch1_error_codes_when_pd_unstable_median": median_or_none(
            pd_unstable_windows, "ch1_error_from_tail_codes"
        ),
        "pd_stable_onset_median_ms": None if not pd_onsets else 1000.0 * statistics.median(pd_onsets),
        "ch1_stable_onset_median_ms": None if not ch1_onsets else 1000.0 * statistics.median(ch1_onsets),
        "ch1_minus_pd_onset_median_ms": (
            None if not paired_onset_deltas else 1000.0 * statistics.median(paired_onset_deltas)
        ),
        "trials": trial_results,
        "windows": windows,
    }


def plot_report(report: dict, output: Path) -> None:
    trials = report["trials"]
    minimum = min(len(trial["samples"]) for trial in trials)
    times = np.median(np.asarray([
        [sample["arrival_after_write_s"] for sample in trial["samples"][:minimum]]
        for trial in trials
    ]), axis=0) * 1000.0
    arrays = np.asarray([
        [[sample[name] for name in ("pdt", "pdr", "ch1")]
         for sample in trial["samples"][:minimum]]
        for trial in trials
    ], dtype=float)
    median = np.median(arrays, axis=0)
    low = np.percentile(arrays, 10, axis=0)
    high = np.percentile(arrays, 90, axis=0)

    figure, axes = plt.subplots(2, 1, figsize=(11.5, 8.0), layout="constrained")
    colors = ("#3178c6", "#2b9a66", "#d74b57")
    for index, (name, color) in enumerate(zip(("PDT", "PDR", "CH1"), colors)):
        normalized = median[:, index] - median[-1, index]
        axes[0].plot(times, normalized, color=color, label=f"{name} median")
        axes[0].fill_between(times, low[:, index] - median[-1, index],
                             high[:, index] - median[-1, index], color=color, alpha=0.12)
    axes[0].axhline(0, color="#555", linewidth=0.8)
    axes[0].set(xlabel="Host arrival after target write (ms)", ylabel="ADC code minus final sample",
                title="Forward switch into one FBG peak; median and 10–90% across trials")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    windows = report["analysis"]["windows"]
    stable = [item for item in windows if item["pd_stable"]]
    unstable = [item for item in windows if not item["pd_stable"]]
    if unstable:
        axes[1].scatter([max(item["pdt_sigma"], item["pdr_sigma"]) for item in unstable],
                        [item["ch1_error_from_tail_codes"] for item in unstable],
                        s=16, alpha=0.45, color="#d9822b", label="PDT/PDR not settled")
    if stable:
        axes[1].scatter([max(item["pdt_sigma"], item["pdr_sigma"]) for item in stable],
                        [item["ch1_error_from_tail_codes"] for item in stable],
                        s=16, alpha=0.45, color="#2b9a66", label="PDT/PDR settled")
    axes[1].set(xlabel="max(PDT, PDR) robust sigma (ADC codes)",
                ylabel="CH1 rolling-median error from settled tail (codes)",
                title="Does a quiet PDT/PDR window imply settled FBG intensity?")
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--group", type=int, choices=range(1, 10))
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--source-hold-s", type=float, default=0.25)
    parser.add_argument("--target-hold-s", type=float, default=0.60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 5 <= args.repeats <= 50:
        raise ValueError("repeats must be 5..50")
    if not 0.2 <= args.source_hold_s <= 2.0 or not 0.3 <= args.target_hold_s <= 2.0:
        raise ValueError("hold times outside bounded diagnostic range")
    if args.output.exists() or args.output.with_suffix(".png").exists():
        raise FileExistsError(args.output)

    choice = choose_peak(args.session, args.group)
    report = {
        "schema": "pdr_pdt_ch1_peak_stability_probe_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "complete": False,
        "scope": "one FBG peak, repeated lower-wavelength to peak forward transitions",
        "timing_scope": "host frame arrival; PDT/PDR/CH1 values share one firmware monitor frame",
        "session": str(args.session.resolve()),
        "selection": choice,
        "feedback_selectors": {"ch0": 0, "ch1": 2, "ch1_kohm": app.PD_FEEDBACK_KOHM_BY_SELECTOR[2]},
        "source_hold_s": args.source_hold_s,
        "target_hold_s": args.target_hold_s,
        "requested_repeats": args.repeats,
        "trials": [],
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
        for repeat in range(1, args.repeats + 1):
            device.reset_input_buffer()
            worker._rx_buffer.clear()
            if worker._command_and_ack(choice["source"]["codes"]) is None:
                raise RuntimeError("源点DAC写入失败")
            if not worker._interruptible_wait(args.source_hold_s):
                raise RuntimeError("源点等待被中断")
            trial = capture_forward_transition(
                device, worker, choice["target"]["codes"], args.target_hold_s
            )
            trial["repeat"] = repeat
            report["trials"].append(trial)
            print(f"repeat={repeat}/{args.repeats} frames={len(trial['samples'])} "
                  f"ack_ms={trial['ack_after_write_s'] * 1000.0:.3f}", flush=True)
        report["analysis"] = analyze_trials(report["trials"])
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
    if not report["safety_cleanup"]["exact_disarm_ack"] or not report["safety_cleanup"]["exact_soa_shutter_ack"]:
        raise RuntimeError("测试结束后的停扫/关光未确认")
    plot_report(report, args.output.with_suffix(".png"))
    print(json.dumps({
        "selection": report["selection"],
        "analysis": {key: value for key, value in report["analysis"].items()
                     if key not in ("trials", "windows")},
        "output": str(args.output.resolve()),
        "plot": str(args.output.with_suffix('.png').resolve()),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
