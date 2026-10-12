"""Verify running firmware version and continuous fan PID telemetry over LAN."""

from __future__ import annotations

import argparse
import time

import app_JDSU as app
from diagnose_adaptive_stress_peaks_live import _shutter
from fbg_lan_serial import FbgLanSerial
from verify_lan_full_function_live import wait_scan_frame
from verify_runtime_firmware import board_extension


def _temperature_c(frame: bytes) -> float:
    point_count = int.from_bytes(frame[2:4], "big")
    position = 4 + point_count * 8
    if position >= len(frame) or frame[position] != 0xAB:
        raise RuntimeError("温度帧缺少峰值状态标记")
    position += 1
    for _channel in range(4):
        peak_count = frame[position]
        position += 1 + peak_count * 4
    integer = int.from_bytes(frame[position : position + 2], "big")
    fraction = int.from_bytes(frame[position + 2 : position + 4], "big")
    return integer + fraction * 0.0001


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.3.46")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--expected", default="1.0.11")
    args = parser.parse_args()

    expected_parts = tuple(int(part) for part in args.expected.split("."))
    expected_version = (
        (expected_parts[0] << 16) | (expected_parts[1] << 8) | expected_parts[2]
    )
    device = FbgLanSerial(
        board_host=args.host, timeout=0.12, write_timeout=3.0
    )
    shutter_confirmed = False
    try:
        device.open()
        deadline = time.monotonic() + 25.0
        while time.monotonic() < deadline and not device.connected:
            time.sleep(0.05)
        if not device.connected:
            raise RuntimeError("OTA 后板卡未重新上线")

        device.write(app.build_work_mode_command(3))
        time.sleep(0.05)
        device.reset_input_buffer()
        for index in range(args.frames):
            frame = wait_scan_frame(device, 100, 8.0)
            extension = board_extension(frame)
            if extension is None or len(extension) < 28:
                raise RuntimeError("温度帧缺少完整板卡状态扩展")
            version = int.from_bytes(extension[24:28], "big")
            rpm = int.from_bytes(extension[10:12], "big")
            duty_permille = int.from_bytes(extension[12:14], "big")
            flags = int.from_bytes(extension[14:16], "big")
            print(
                f"frame={index + 1} temperature={_temperature_c(frame):.4f}C "
                f"duty={duty_permille / 10.0:.1f}% rpm={rpm} "
                f"flags=0x{flags:04X} version="
                f"{(version >> 16) & 0xFF}.{(version >> 8) & 0xFF}.{version & 0xFF}"
            )
            if version != expected_version:
                raise RuntimeError(
                    f"运行版本不匹配：0x{version:08X}，期望0x{expected_version:08X}"
                )
        print("thermal_pid_validation=PASS")
        return 0
    finally:
        try:
            if device.is_open and device.connected:
                shutter_confirmed = _shutter(device)
        finally:
            device.close()
            print(f"soa_shutter_confirmed={int(shutter_confirmed)}")


if __name__ == "__main__":
    raise SystemExit(main())
