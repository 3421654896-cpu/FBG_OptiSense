"""Destructive-free live acceptance test for the full-function LAN path.

The optical source is enabled only at one audited table point.  EXTRA mode
keeps the cooling fan at full speed, and the ``finally`` block always requests
the PI11210 zero-current SOA shutter before the socket is closed.
"""

from __future__ import annotations

import argparse
import time

import app_JDSU as app
from fbg_lan_serial import FbgLanSerial
from verify_runtime_firmware import board_extension


def read_single_frames(device, rx, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        waiting = int(device.in_waiting or 0)
        data = device.read(min(max(waiting, 64), 4096))
        if data:
            rx.extend(data)
            yield from app.extract_single_value_frames(rx)
        else:
            time.sleep(0.002)


def wait_single(device, predicate, timeout_s=3.0):
    rx = bytearray()
    for frame in read_single_frames(device, rx, timeout_s):
        value = predicate(frame)
        if value is not None:
            return value
    raise RuntimeError("等待板卡原始20字节回读超时")


def wait_dac_ack(device, point):
    expected = tuple(point.codes)

    def decode(frame):
        if frame[:4] != b"\xff\xff\x02\x00":
            return None
        returned = tuple(
            int.from_bytes(frame[4 + index * 2 : 6 + index * 2], "big")
            for index in range(5)
        )
        if returned != expected:
            return None
        app.UnlimitedAccuracyWorker._validate_ack_status(frame, 0)
        return returned

    device.write(app.build_single_value_dac_command(expected))
    return wait_single(device, decode)


def wait_rt(device):
    return wait_single(device, app.decode_single_value_rt_frame)


def wait_scan_frame(device, expected_points, timeout_s=8.0):
    rx = bytearray()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        waiting = int(device.in_waiting or 0)
        data = device.read(min(max(waiting, 256), 4096))
        if not data:
            time.sleep(0.003)
            continue
        rx.extend(data)
        start = rx.find(b"\xee\xee")
        if start < 0:
            if len(rx) > 1:
                del rx[:-1]
            continue
        if start:
            del rx[:start]
        end = rx.find(b"\xff\xef", 4)
        if end >= 0:
            frame = bytes(rx[: end + 2])
            points = int.from_bytes(frame[2:4], "big")
            if points == expected_points and len(frame) >= 4 + points * 8 + 2:
                return frame
            del rx[:2]
    raise RuntimeError(f"局域网未收到完整{expected_points}点原始帧")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.3.46")
    parser.add_argument("--timeout", type=float, default=25.0)
    parser.add_argument("--expected-version", default="1.0.15")
    args = parser.parse_args()
    version_parts = tuple(int(part) for part in args.expected_version.split("."))
    if len(version_parts) != 3 or any(part < 0 or part > 255 for part in version_parts):
        raise ValueError("--expected-version 必须是 major.minor.patch")
    expected_runtime_version = (
        (version_parts[0] << 16) | (version_parts[1] << 8) | version_parts[2]
    )

    events = []
    device = FbgLanSerial(
        board_host=args.host,
        timeout=0.12,
        write_timeout=3.0,
        on_connection=lambda connected, text: events.append((connected, text)),
    )
    last_codes = (0, 0, 0, 0, 0)
    shutter_confirmed = False
    try:
        device.open()
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline and not device.connected:
            time.sleep(0.05)
        if not device.connected:
            raise RuntimeError("未发现局域网板卡的全功能通道")
        print(f"lan_ready=1 board_ip={device.board_ip}")

        device.write(app.build_work_mode_command(2))
        time.sleep(0.15)
        device.reset_input_buffer()

        device.write(app.build_extra_feedback_command(0, 0))
        feedback = wait_single(device, app.decode_extra_feedback_status)
        if feedback != ((0, 0), app.ACK_VALUE):
            raise RuntimeError(f"等间隔/单值跨阻IO回读异常：{feedback}")
        print("manual_feedback_ack=1 selectors=0,0")

        points = app.load_fullband_accuracy_table()
        target_index = 681  # 1538.620 nm, already wavelength-meter audited.
        for point in points[target_index - 1 : target_index + 2]:
            last_codes = tuple(point.codes)
            wait_dac_ack(device, point)
            time.sleep(0.020)
            device.reset_input_buffer()
            rt = wait_rt(device)
            volts = tuple(value * app.PD_ADC_REFERENCE_V / app.PD_ADC_CODE_COUNT for value in rt)
            print(
                f"equal_point={point.target_nm:.3f} "
                + "adc_v="
                + ",".join(f"{value:.5f}" for value in volts)
            )

        device.write(app.build_stress_feedback_command(0, 0))
        stress_feedback = wait_single(device, app.decode_stress_feedback_status)
        if stress_feedback != ((0, 0), app.ACK_VALUE):
            raise RuntimeError(f"应力跨阻IO回读异常：{stress_feedback}")
        device.write(app.build_work_mode_command(0))
        time.sleep(0.05)
        device.reset_input_buffer()
        stress_frame = wait_scan_frame(device, 63)
        extension = board_extension(stress_frame)
        if extension is None or len(extension) < 28:
            raise RuntimeError("应力帧缺少板卡状态扩展")
        runtime_version = int.from_bytes(extension[24:28], "big")
        if runtime_version != expected_runtime_version:
            raise RuntimeError(
                f"板卡固件版本不符：0x{runtime_version:08X}，"
                f"期望{args.expected_version}"
            )
        print(
            f"stress_raw_frame=1 points=63 bytes={len(stress_frame)} "
            f"head={stress_frame[:4].hex()} tail={stress_frame[-2:].hex()} "
            f"firmware={args.expected_version}"
        )

        device.write(app.build_work_mode_command(3))
        time.sleep(0.05)
        device.reset_input_buffer()
        temperature_frame = wait_scan_frame(device, 100)
        print(
            f"temperature_raw_frame=1 points=100 bytes={len(temperature_frame)} "
            f"head={temperature_frame[:4].hex()} "
            f"tail={temperature_frame[-2:].hex()}"
        )
        print("validation=PASS")
        return 0
    finally:
        try:
            if device.is_open:
                device.write(app.build_work_mode_command(2))
                time.sleep(0.12)
                device.reset_input_buffer()
                command = app.build_single_value_dac_command(
                    last_codes, soa_mode=app.PI11210_SOA_SHUTTER_MODE
                )
                expected = tuple(
                    int.from_bytes(command[4 + index * 2 : 6 + index * 2], "big")
                    for index in range(5)
                )
                device.write(command)

                def shutter_ack(frame):
                    if frame[:4] != b"\xff\xff\x02\x00":
                        return None
                    returned = tuple(
                        int.from_bytes(frame[4 + index * 2 : 6 + index * 2], "big")
                        for index in range(5)
                    )
                    if returned != expected or frame[14] != app.PI11210_SOA_SHUTTER_MODE:
                        return None
                    app.UnlimitedAccuracyWorker._validate_ack_status(
                        frame, app.PI11210_SOA_SHUTTER_MODE
                    )
                    return True

                shutter_confirmed = bool(wait_single(device, shutter_ack, 3.0))
        finally:
            device.close()
            print(f"soa_shutter_confirmed={int(shutter_confirmed)}")


if __name__ == "__main__":
    raise SystemExit(main())
