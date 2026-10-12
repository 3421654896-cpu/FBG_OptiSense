"""Find clean high-power left-slope candidates around the 1536.326 nm FBG."""
from __future__ import annotations
import json, time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from laser_power_calibration import (AQ6150B, DAC_SOURCE, LaserSerial,
    code_to_current, current_to_code, parse_dac_rows, save_json, take_reading)

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "segment2_soa_layers_135.json"


def main() -> int:
    rows = parse_dac_rows(DAC_SOURCE)
    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "results": {},
    }
    if OUTPUT.exists():
        saved = json.loads(OUTPUT.read_text(encoding="utf-8"))
        payload["results"].update(saved.get("results", {}))

    laser, meter = LaserSerial("COM6"), AQ6150B("GPIB0::7::INSTR")
    try:
        for seed in range(23, 34):
            for soa in (100.0, 110.0, 120.0, 125.0, 130.0, 135.0):
                key = f"{seed}:{soa:.1f}"
                if key in payload["results"]:
                    continue
                codes = list(rows[seed])
                codes[0] = current_to_code(135.0, 0)
                codes[1] = current_to_code(soa, 1)
                reading = take_reading(
                    laser, meter, codes, 1536.326, 0.28, True
                )
                payload["results"][key] = {
                    "seed_index": seed,
                    "codes": codes,
                    "currents_ma": [
                        code_to_current(code, channel)
                        for channel, code in enumerate(codes)
                    ],
                    "reading": asdict(reading),
                }
                save_json(OUTPUT, payload)
                print(
                    f"seed={seed} soa={soa:.0f} "
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
