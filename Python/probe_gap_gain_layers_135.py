"""Probe GAIN layers for the remaining sparse 1548/1560 nm branches."""
from __future__ import annotations
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from laser_power_calibration import (AQ6150B,DAC_SOURCE,LaserSerial,code_to_current,
 current_to_code,parse_dac_rows,save_json,take_reading)
HERE = Path(__file__).resolve().parent
OUT = HERE / "gap_gain_layers_soa135.json"


def main() -> int:
    rows = parse_dac_rows(DAC_SOURCE)
    data = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "results": {},
    }
    laser, meter = LaserSerial("COM6"), AQ6150B("GPIB0::7::INSTR")
    try:
        for segment, start, stop, target in (
            (5, 56, 67, 1547.9),
            (8, 89, 100, 1559.9),
        ):
            for seed in range(start, stop):
                for gain in (105.0, 120.0):
                    codes = list(rows[seed])
                    codes[0] = current_to_code(gain, 0)
                    codes[1] = current_to_code(135, 1)
                    reading = take_reading(laser, meter, codes, target, 0.28, True)
                    key = f"{segment}:{seed}:{gain:.0f}"
                    data["results"][key] = {
                        "segment": segment,
                        "seed_index": seed,
                        "codes": codes,
                        "currents_ma": [
                            code_to_current(code, channel)
                            for channel, code in enumerate(codes)
                        ],
                        "reading": asdict(reading),
                    }
                    save_json(OUT, data)
                    print(
                        f"seg={segment} seed={seed} gain={gain:.0f} "
                        f"wave={reading.wavelength_nm:.6f} "
                        f"power={reading.power_mw:.3f} "
                        f"SMSR={reading.side_mode_suppression_db}",
                        flush=True,
                    )
    finally:
        try:
            park = list(rows[44])
            park[0] = current_to_code(60, 0)
            park[1] = current_to_code(80, 1)
            laser.set_codes(park)
            time.sleep(0.08)
        finally:
            laser.close()
            meter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
