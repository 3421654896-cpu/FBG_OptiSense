"""Identity-locked USB update acceptance test; no Wi-Fi or SWD commands."""
import argparse
import hashlib
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from fbg_ota_protocol import load_package
from fbg_usb_ota import UsbOtaClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--device-serial", required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; preserving the previous report")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    package = load_package(args.package)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  requested_port=args.port, device_serial=args.device_serial,
                  package_version=package.version_text,
                  package_sha256=hashlib.sha256(args.package.read_bytes()).hexdigest(),
                  image_bytes=package.image_size, apply_requested=args.apply,
                  usb_only=True, swd_commands=False, verified=False)
    started = time.monotonic()
    payload_started = None
    payload_finished = None
    last_bucket = -1

    def progress(event):
        nonlocal payload_started, payload_finished, last_bucket
        now = time.monotonic()
        if payload_started is None and "写入暂存区" in event.message:
            payload_started = now
        if event.sent == event.total and payload_finished is None:
            payload_finished = now
        bucket = int(event.sent * 10 / max(1, event.total))
        if bucket != last_bucket or "重启" in event.message or "完成" in event.message:
            print(event.message, flush=True)
            last_bucket = bucket

    try:
        with UsbOtaClient(args.port, args.device_serial) as client:
            before = client.query_info()
            report["before"] = asdict(before)
            print(f"USB device {before.serial_number}: v{before.version_text}; "
                  f"bootloader_ready={before.bootloader_ready}", flush=True)
            if args.apply:
                after = client.install(package, progress=progress)
                report.update(after=asdict(after), final_port=client.port,
                              reboot_acknowledged=client.reboot_acknowledged,
                              verified=True)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_s"] = round(time.monotonic() - started, 3)
        if payload_started is not None and payload_finished is not None:
            elapsed = payload_finished - payload_started
            report["payload_transfer_s"] = round(elapsed, 3)
            report["payload_kib_per_s"] = round(package.image_size / 1024 / max(.001, elapsed), 2)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
