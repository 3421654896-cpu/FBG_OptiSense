#!/usr/bin/env python3
"""Read-only health check for the WCH-Link CMSIS-DAP USB endpoint."""

from __future__ import annotations

import argparse
import sys

import libusb_package
import usb.backend.libusb1
import usb.core
import usb.util


INFO_IDS = {
    0x01: "vendor",
    0x02: "product",
    0x03: "serial",
    0x04: "firmware",
    0xF0: "capabilities",
    0xFE: "packet_count",
    0xFF: "packet_size",
}


def _backend():
    return usb.backend.libusb1.get_backend(
        find_library=lambda name: libusb_package.find_library(name)
    )


def _decode_info(info_id: int, response: bytes):
    if len(response) < 2 or response[0] != 0x00:
        raise RuntimeError(f"invalid DAP_Info response: {response.hex()}")
    length = response[1]
    payload = response[2 : 2 + length]
    if len(payload) != length:
        raise RuntimeError(f"truncated DAP_Info response: {response.hex()}")
    if info_id in (0x01, 0x02, 0x03, 0x04):
        return payload.rstrip(b"\x00").decode("utf-8", errors="replace")
    if info_id == 0xFF:
        return int.from_bytes(payload, "little")
    if info_id in (0xF0, 0xFE):
        return int.from_bytes(payload, "little")
    return payload.hex()


def query(timeout_ms: int) -> dict[str, object]:
    backend = _backend()
    if backend is None:
        raise RuntimeError("libusb backend unavailable")
    device = usb.core.find(
        idVendor=0x1A86,
        idProduct=0x8012,
        backend=backend,
    )
    if device is None:
        raise RuntimeError("WCH-Link USB device not found")

    values: dict[str, object] = {
        "usb_bus": device.bus,
        "usb_address": device.address,
        "usb_serial": usb.util.get_string(device, device.iSerialNumber),
    }
    try:
        try:
            device.set_configuration()
        except usb.core.USBError:
            # The active WinUSB configuration may already be selected.
            pass
        usb.util.claim_interface(device, 0)
        for info_id, name in INFO_IDS.items():
            packet = bytearray(64)
            packet[0] = 0x00  # DAP_Info
            packet[1] = info_id
            device.write(0x02, packet, timeout=timeout_ms)
            response = bytes(device.read(0x83, 64, timeout=timeout_ms))
            values[name] = _decode_info(info_id, response)
        return values
    finally:
        try:
            usb.util.release_interface(device, 0)
        except usb.core.USBError:
            pass
        usb.util.dispose_resources(device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-ms", type=int, default=1500)
    args = parser.parse_args()
    try:
        values = query(args.timeout_ms)
    except Exception as exc:
        print(f"cmsis_dap_ready=0 error={type(exc).__name__}: {exc}")
        return 1
    print("cmsis_dap_ready=1")
    for name, value in values.items():
        print(f"{name}={value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
