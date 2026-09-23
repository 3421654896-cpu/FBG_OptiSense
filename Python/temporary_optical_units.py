"""Physical CH1 display units derived from the populated PD front end.

Sources inspected for these constants:

* JDSU_BIG_V1.2 schematic, sheet 3: TPC5121 Range 1, 2.5 V reference.
* JDSU_BIG_V1.2 schematic, sheet 10: selectable 2/5/20/40 kOhm
  transimpedance feedback paths (the 4 kOhm drawing note says 5 kOhm fitted).
* XCPD1007-P1-1131 datasheet, page 1: InGaAs PIN responsivity is at
  least 0.85 A/W at 1550 nm and -3 dBm.

The dBm result is an engineering estimate, not a traceable power-meter
calibration. It assumes the transimpedance output has no material zero-light
offset and uses the datasheet's minimum responsivity across the FBG band.
"""

from __future__ import annotations

import math

import numpy as np


ADC_REFERENCE_V = 2.5
ADC_CODE_COUNT = 4096.0
PD_RESPONSIVITY_A_PER_W = 0.85
TRANSIMPEDANCE_KOHM_BY_SELECTOR = (2.0, 40.0, 5.0, 20.0)


def transimpedance_ohm(selector: int) -> float:
    selector = int(selector)
    if selector not in range(len(TRANSIMPEDANCE_KOHM_BY_SELECTOR)):
        raise ValueError("CH1模拟跨阻档位无效")
    return TRANSIMPEDANCE_KOHM_BY_SELECTOR[selector] * 1000.0


def adc_code_to_voltage(adc_code):
    values = np.asarray(adc_code, dtype=float)
    return values * ADC_REFERENCE_V / ADC_CODE_COUNT


def adc_code_to_optical_power_w(adc_code, selector: int):
    return (
        adc_code_to_voltage(adc_code)
        / transimpedance_ohm(selector)
        / PD_RESPONSIVITY_A_PER_W
    )


def adc_code_to_dbm(adc_code, selector: int, *, floor_nonpositive=False):
    power_w = np.asarray(
        adc_code_to_optical_power_w(adc_code, selector), dtype=float
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        result = 10.0 * np.log10(power_w / 1e-3)
    if floor_nonpositive:
        result = np.where(
            np.isfinite(result), result, optical_detection_floor_dbm(selector)
        )
    return result


def optical_detection_floor_dbm(selector: int) -> float:
    """Half-LSB optical-power floor used only to place zero-code plot points."""
    half_lsb_power = float(adc_code_to_optical_power_w(0.5, selector))
    return 10.0 * math.log10(half_lsb_power / 1e-3)


def format_physical_value(adc_code, mode: str, selector: int) -> str:
    value = float(adc_code)
    if not math.isfinite(value):
        return "—"
    if mode == "adc":
        return f"{value:g}"
    if mode == "voltage":
        return f"{float(adc_code_to_voltage(value)):.6f}"
    if mode == "dbm":
        if value <= 0.0:
            return f"≤{optical_detection_floor_dbm(selector):.2f}"
        return f"{float(adc_code_to_dbm(value, selector)):.2f}"
    raise ValueError(f"不支持的显示单位：{mode}")
