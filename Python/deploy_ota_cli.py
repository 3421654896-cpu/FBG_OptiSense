"""Non-interactive OTA uploader with concise progress and no secret logging."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from fbg_ota_protocol import OtaClient, OtaProgress, load_package


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("--host", default="192.168.3.46")
    args = parser.parse_args()
    password = os.environ.get("FBG_OTA_PASSWORD", "")
    if not password:
        raise RuntimeError("FBG_OTA_PASSWORD is not set")

    package = load_package(args.package)
    last_bucket = [-1]
    last_message = [""]

    def progress(item: OtaProgress) -> None:
        percent = int(item.sent * 100 / max(item.total, 1))
        bucket = min(10, percent // 10)
        transferring = 0 < item.sent < item.total
        if transferring and bucket != last_bucket[0]:
            print(
                f"progress={bucket * 10}% sent={item.sent}/{item.total}",
                flush=True,
            )
        elif not transferring and item.message != last_message[0]:
            print(
                f"progress={percent}% sent={item.sent}/{item.total} "
                f"stage={item.message}",
                flush=True,
            )
        last_bucket[0] = bucket
        last_message[0] = item.message

    OtaClient(args.host, password, response_timeout_s=30.0).upload(
        package, progress=progress
    )
    print(
        f"ota_upload_complete=1 reboot_verified=1 raw_ready=1 "
        f"version={package.version_text}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
