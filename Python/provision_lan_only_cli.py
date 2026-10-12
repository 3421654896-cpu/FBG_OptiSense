"""Safely provision native-USB LAN-only Wi-Fi without printing credentials."""

from __future__ import annotations

import argparse
import os
import time

import serial
from serial.tools import list_ports

from benchmark_stress_realtime import _close_shutter, _safe_disarm
from fbg_wifi_setup import (
    ProvisioningSettings,
    build_provisioning_frame,
    find_provisioning_ack,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--device-serial", required=True)
    parser.add_argument("--ssid", required=True)
    parser.add_argument("--password-env", required=True)
    parser.add_argument("--device-id", required=True)
    args = parser.parse_args()

    matches = [
        item for item in list_ports.comports()
        if item.device.upper() == args.port.upper()
        and item.vid == 0x0483 and item.pid == 0x5740
        and item.serial_number == args.device_serial
    ]
    if len(matches) != 1:
        raise RuntimeError("STM32 native USB identity mismatch; refusing provisioning")
    wifi_password = os.environ.get(args.password_env)
    if not wifi_password:
        raise RuntimeError("Wi-Fi password environment variable is missing")

    settings = ProvisioningSettings(
        ssid=args.ssid,
        wifi_password=wifi_password,
        broker_host="",
        broker_port=8883,
        client_id=args.device_id,
        mqtt_username="",
        mqtt_password="",
        topic_prefix="fbg/v1",
        device_id=args.device_id,
        tls=True,
        lan_only=True,
    )
    frame = build_provisioning_frame(settings)
    expected_crc = int.from_bytes(frame[-4:], "big")
    with serial.Serial(args.port, 2_000_000, timeout=0.05, write_timeout=3.0) as device:
        device.dtr = True
        if not _safe_disarm(device) or not _close_shutter(device):
            raise RuntimeError("Could not confirm disarm and closed SOA shutter")
        try:
            device.reset_input_buffer()
            device.write(frame)
            device.flush()
            received = bytearray()
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                chunk = device.read(1024)
                if not chunk:
                    continue
                received.extend(chunk)
                if len(received) > 8192:
                    del received[:-4096]
                ack = find_provisioning_ack(received, expected_crc)
                if ack is None:
                    continue
                status, state, valid, checksum = ack
                if status != 0 or valid != 1:
                    raise RuntimeError(
                        f"Provisioning rejected: status={status}, state={state}"
                    )
                print(f"LAN-only configuration accepted; state={state}; crc={checksum:08X}")
                return 0
            raise TimeoutError("No matching LAN-only provisioning ACK in 10 seconds")
        finally:
            if not _safe_disarm(device) or not _close_shutter(device):
                raise RuntimeError("Final disarm/shutter confirmation missing")


if __name__ == "__main__":
    raise SystemExit(main())
