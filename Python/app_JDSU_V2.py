"""Minimal AQ6150B communication smoke test.

Hardware access is intentionally kept behind ``main`` so importing this
module for discovery or static analysis never operates the wavelength meter.
"""

from __future__ import annotations

import argparse
import time

import pyvisa


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="AQ6150B GPIB 通信检查")
    parser.add_argument("--resource", default="GPIB0::7::INSTR")
    args = parser.parse_args(argv)

    manager = pyvisa.ResourceManager()
    instrument = None
    try:
        resources = manager.list_resources()
        print(resources)
        if args.resource not in resources:
            raise RuntimeError(f"未发现波长计：{args.resource}")

        instrument = manager.open_resource(args.resource)
        instrument.timeout = 10_000
        instrument.write("*CLS")
        instrument.write(":CONF:ARR:POW")
        instrument.write(":INIT")
        instrument.write("*WAI")
        print(instrument.query(":FETCh:ARRay:POWer?"))
        time.sleep(1)
        print(instrument.query(":FETCh:ARRay:POWer:WAVelength?"))
        print(instrument.query("*IDN?"))
    finally:
        if instrument is not None:
            instrument.close()
        manager.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
