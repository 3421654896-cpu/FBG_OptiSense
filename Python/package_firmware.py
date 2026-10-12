"""Create the CRC-protected .fbgfw file consumed by the unified OTA page."""

from __future__ import annotations

import argparse
from pathlib import Path

from fbg_ota_protocol import build_package, parse_version


def main(argv=None):
    parser = argparse.ArgumentParser(description="打包 STM32F205RE OTA 固件")
    parser.add_argument("image", type=Path, help="链接到 0x08008000 的应用 bin")
    parser.add_argument("--version", required=True, help="例如 1.0.0")
    parser.add_argument("--output", type=Path, help="输出 .fbgfw 路径")
    args = parser.parse_args(argv)
    output = args.output or args.image.with_suffix(".fbgfw")
    package = build_package(args.image, output, parse_version(args.version))
    print(
        f"{package.path}\n"
        f"version={package.version_text} size={package.image_size} "
        f"crc32={package.image_crc32:08X}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
