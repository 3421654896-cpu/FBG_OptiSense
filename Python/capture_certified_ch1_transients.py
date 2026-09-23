"""Timestamped direct ADC curves for the certified 45-row stress table.

Only EXTRA-mode USB diagnostics are used. No CNC import, motion, table edit,
current optimization or model deployment. The source is held for 100 ms;
source stability is not checked and must not be inferred from that duration.
these pair transitions do not certify the continuous sparse optical trajectory.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import serial
import app_JDSU as app
from benchmark_stress_realtime import _safe_disarm, _close_shutter
from capture_ch1_rc_diagnostic import load_installed_rows

OFFSETS_US = (0,25,50,75,100,150,200,300,400,500,650,800,1000,1250,
              1500,2000,2500,3000,4000,5000,6000,8000,10000,12000,
              16000,20000,25000,30000,40000,50000,60000,80000,100000)
CHANNELS = ("CH0", "CH1", "CH2", "CH3", "PDT", "PDR")


def command(source, target, tag):
    if not 0 <= source < 45 or not 0 <= target < 45 or not 1 <= tag <= 0xffffffff:
        raise ValueError("invalid certified row or request tag")
    packet = bytearray(808)
    packet[:4] = b"\xff\xff\x03\x01"
    packet[4:6] = source.to_bytes(2, "big")
    packet[6:8] = target.to_bytes(2, "big")
    packet[8] = 3
    packet[9:13] = tag.to_bytes(4, "big")
    return bytes(packet)


def decode(packet, source, target, tag):
    if len(packet) < 18 or packet[:4] != b"\xd5\x5d\x02\x03":
        raise ValueError("expected v2 certified-table diagnostic")
    if packet[12:14] != bytes((33, 6)):
        raise ValueError("unexpected ADC shape")
    if len(packet) != 18 + 12 + 33 * 16 + 2 or packet[-2:] != b"\x5d\xd5":
        raise ValueError("invalid diagnostic length or trailer")
    if (int.from_bytes(packet[4:6], "big"), int.from_bytes(packet[6:8], "big"),
        int.from_bytes(packet[14:18], "big")) != (source, target, tag):
        raise ValueError("stale or mismatched diagnostic response")
    update_us = int.from_bytes(packet[8:12], "big")
    baseline = [int.from_bytes(packet[p:p+2], "big") for p in range(18,30,2)]
    times, samples = [], []
    for i in range(33):
        p = 30 + i * 16
        times.append(int.from_bytes(packet[p:p+4], "big"))
        samples.append([int.from_bytes(packet[c:c+2], "big") for c in range(p+4,p+16,2)])
    if times[0] < update_us or any(b <= a for a,b in zip(times,times[1:])):
        raise ValueError("nonmonotonic sample clock")
    if any(value > 4095 for row in [baseline, *samples] for value in row):
        raise ValueError("out-of-range direct ADC")
    return {"source_index":source, "target_index":target, "request_tag":tag,
            "path_update_us":update_us, "baseline_adc":baseline,
            "sample_start_from_update_begin_us":times,
            "sample_start_after_target_write_us":[value-update_us for value in times],
            "direct_adc":samples, "wire_hex":packet.hex()}


def capture(device, source, target, tag):
    device.reset_input_buffer()
    device.write(command(source,target,tag))
    rx = bytearray()
    deadline = time.monotonic() + 3
    length = 18 + 12 + 33 * 16 + 2
    while time.monotonic() < deadline:
        rx.extend(device.read(4096))
        start = rx.find(b"\xd5\x5d\x02\x03")
        if start >= 0 and len(rx) >= start + length:
            return decode(bytes(rx[start:start+length]),source,target,tag)
    raise TimeoutError("v2 certified transient not received (requires firmware >=1.0.26)")


def capture_stable_tail(worker, minimum_additional_wait_s=.2):
    """Read the same held target, without another DAC command changing its path.

    The timestamped curve has already run for 100 ms. Wait at least another
    200 ms, discard older monitor packets, then retain the actual six-channel
    monitor window and its stability gate. Host timestamps are not ADC clocks.
    """
    if minimum_additional_wait_s < .2:
        raise ValueError("teacher must wait at least 200 ms after the 100 ms curve")
    if not worker._interruptible_wait(minimum_additional_wait_s):
        raise RuntimeError("teacher cancelled")
    worker.port.reset_input_buffer()
    worker._rx_buffer.clear()
    samples = []
    wire_frames = []
    arrived_s = []
    started = time.monotonic()
    deadline = started + app.FULLBAND_ACCURACY_POINT_TIMEOUT_S
    while worker.running and time.monotonic() < deadline:
        for frame in worker._read_available_frames():
            values = app.decode_single_value_monitor_frame(frame)
            if values is None or any(value < 0 or value > 4095 for value in values):
                continue
            samples.append(values)
            wire_frames.append(frame.hex())
            arrived_s.append(time.monotonic() - started)
            median, sigma, drift, settled = app.fullband_accuracy_statistics(samples)
            if settled or len(samples) >= app.FULLBAND_ACCURACY_MAX_SAMPLES:
                break
        if samples and (settled or len(samples) >= app.FULLBAND_ACCURACY_MAX_SAMPLES):
            break
        time.sleep(.001)
    if not samples:
        raise RuntimeError("no held-target stable teacher monitor data")
    median, sigma, drift, settled = app.fullband_accuracy_statistics(samples)
    return {
        "adc_value_kind": "direct_adc",
        "dac_rewritten": False,
        "minimum_target_hold_before_window_s": .1 + minimum_additional_wait_s,
        "channels": ["PDT", "PDR", "CH0", "CH1", "CH2", "CH3"],
        "monitor_adc": [list(row) for row in samples],
        "monitor_wire_hex": wire_frames,
        "host_arrival_since_window_s": arrived_s,
        "median_adc": median.tolist(), "sigma_codes": sigma.tolist(),
        "drift_codes": drift.tolist(),
        "all_channels_stable": bool(settled),
        "stable": bool(settled and median[3] < 4080),
        "ch1_adc_code": float(median[3]),
        "ch1_saturated": bool(np.any(np.asarray(samples)[:, 3] >= 4080)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port",default="COM6")
    parser.add_argument("--selector",type=int,choices=range(4),default=1)
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--indices",type=int,nargs="*",default=list(range(45)))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--stable-teacher", action="store_true",
                        help="pair every curve with a >=300 ms held-target monitor window")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10 or not args.indices or any(not 0 <= i < 45 for i in args.indices):
        raise ValueError("invalid bounded capture request")
    rows, hidden = load_installed_rows()
    report = {"schema":"certified_ch1_transients_v3", "selector":args.selector,
              "created_utc":datetime.now(timezone.utc).isoformat(), "channels":CHANNELS,
              "adc_value_kind":"direct_adc", "paired_stable_teacher":args.stable_teacher,
              "diagnostic_protocol_version":2,
              "timing_scope":"six-channel conversion group start, not individual CH1 sample timestamp",
              "requested_offsets_us":OFFSETS_US, "source_settle_us":100000,
              "scope":"source held 100 ms (stability unverified) to target; not continuous sparse waveform",
              "dac_rows":rows,"hidden_order2_codes":hidden,"captures":[],"complete":False}
    with serial.Serial(args.port,2_000_000,timeout=.02,write_timeout=3) as device:
        try:
            device.dtr = True
            _safe_disarm(device)
            worker = app.EqualIntervalWorker(device,(),{},settle_s=.3,
                                            feedback_selectors=(0,args.selector))
            if not worker._feedback_command_and_ack():
                raise RuntimeError("feedback IO ACK missing")
            tag = int(time.time_ns() & 0x7fffffff) or 1
            for repeat in range(args.repeats):
                for target in args.indices:
                    tag = (tag + 1) & 0xffffffff or 1
                    record = capture(device,(target-1)%45,target,tag)
                    record["repeat"] = repeat
                    report["captures"].append(record)
                    if args.stable_teacher:
                        record["stable_teacher"] = capture_stable_tail(worker)
                    if len(report["captures"]) % 15 == 0:
                        print(f"captures={len(report['captures'])}/{len(args.indices)*args.repeats}", flush=True)
                print(f"repeat={repeat+1}/{args.repeats} captures={len(report['captures'])}",flush=True)
            report["complete"] = True
        finally:
            errors = []
            disarmed = shuttered = False
            try:
                disarmed = _safe_disarm(device)
            except Exception as exc:
                errors.append(f"disarm: {exc}")
            try:
                shuttered = _close_shutter(device)
            except Exception as exc:
                errors.append(f"shutter: {exc}")
            report["safety_cleanup"] = {"exact_disarm_ack":disarmed,
                                       "exact_soa_shutter_ack":shuttered,"errors":errors}
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
            print(f"disarm_confirmed={int(disarmed)} soa_shutter_confirmed={int(shuttered)}",flush=True)
            if not disarmed or not shuttered:
                raise RuntimeError("safe shutdown not confirmed")
    print(f"report={args.output.resolve()}")


if __name__ == "__main__":
    main()
