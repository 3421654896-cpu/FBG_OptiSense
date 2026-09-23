"""Read the running application version from a fresh USB spectrum frame."""

from __future__ import annotations

import argparse
import time

import serial


COMMAND_SIZE = 808


def mode_command(mode: int) -> bytes:
    command = bytearray(COMMAND_SIZE)
    command[0:4] = bytes((0xFF, 0xFF, 0x01, 0x02))
    command[8] = int(mode)
    return bytes(command)


def extract_frames(buffer: bytearray) -> list[bytes]:
    frames: list[bytes] = []
    while True:
        start = buffer.find(b"\xEE\xEE")
        if start < 0:
            if len(buffer) > 1:
                del buffer[:-1]
            break
        if start:
            del buffer[:start]
        end = buffer.find(b"\xFF\xEF", 2)
        if end < 0:
            break
        frames.append(bytes(buffer[: end + 2]))
        del buffer[: end + 2]
    return frames


def board_extension(frame: bytes) -> bytes | None:
    if len(frame) < 8 or frame[:2] != b"\xEE\xEE":
        return None
    point_count = int.from_bytes(frame[2:4], "big")
    position = 4 + point_count * 8
    if position >= len(frame) or frame[position] != 0xAB:
        return None
    position += 1
    for _channel in range(4):
        if position >= len(frame):
            return None
        peak_count = frame[position]
        position += 1 + peak_count * 4
    position += 4  # temperature
    extension_start = position + 4  # legacy CH0/CH1 gain masks
    if extension_start + 4 > len(frame) - 2:
        return None
    if frame[extension_start : extension_start + 3] != b"\xB5\x4D\x01":
        return None
    extension_length = frame[extension_start + 3]
    start = extension_start + 4
    end = start + extension_length
    if end > len(frame) - 2:
        return None
    return frame[start:end]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--baud", type=int, default=2_000_000)
    parser.add_argument("--expected", default="")
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    expected = None
    if args.expected:
        parts = [int(part) for part in args.expected.split(".")]
        if len(parts) != 3 or any(not 0 <= part <= 255 for part in parts):
            raise ValueError("--expected must use major.minor.patch")
        expected = (parts[0] << 16) | (parts[1] << 8) | parts[2]

    buffer = bytearray()
    extension = None
    diagnostics: list[str] = []
    with serial.Serial(args.port, args.baud, timeout=0.05) as device:
        device.dtr = True
        time.sleep(0.2)
        device.reset_input_buffer()
        device.write(mode_command(3))
        device.flush()
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline and extension is None:
            buffer.extend(device.read(4096))
            for frame in extract_frames(buffer):
                diagnostics.append(
                    f"len={len(frame)} points={int.from_bytes(frame[2:4], 'big')} "
                    f"tail={frame[-40:].hex()}"
                )
                extension = board_extension(frame)
                if extension is not None:
                    break
        device.write(mode_command(2))  # EXTRA/idle: close the optical source.
        device.flush()
        time.sleep(0.15)

    if extension is None or len(extension) < 28:
        detail = diagnostics[-1] if diagnostics else "no complete EE/FFEF frame"
        raise RuntimeError(
            "running firmware did not return the v1.0.4 status extension; " + detail
        )
    selectors = (int(extension[22]), int(extension[23]))
    version = int.from_bytes(extension[24:28], "big")
    version_text = f"{(version >> 16) & 0xFF}.{(version >> 8) & 0xFF}.{version & 0xFF}"
    print(f"runtime_version={version_text}")
    print(f"feedback_selectors={selectors[0]},{selectors[1]}")
    print("board_returned_to_idle=1")
    if expected is not None and version != expected:
        raise RuntimeError(
            f"runtime version mismatch: expected 0x{expected:08X}, got 0x{version:08X}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
