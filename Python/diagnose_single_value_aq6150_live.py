"""Reproduce one desktop single-value output and verify it with the AQ6150B.

The diagnostic uses the same table lookup, predecessor routing and exact DAC ACK
path as the desktop single-value page.  It always requests the hardware SOA
shutter before closing the serial port.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets

import app_JDSU as local
from laser_power_calibration import AQ6150B


def measure_with_retry(meter: AQ6150B, target_nm: float, attempts: int = 3):
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return meter.measure(target_nm, select_main_peak=True), attempt
        except RuntimeError as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(0.25 * attempt)
    assert last_error is not None
    raise last_error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--meter", default="GPIB0::7::INSTR")
    parser.add_argument("--target-nm", type=float, default=1535.0)
    parser.add_argument("--settle-s", type=float, default=0.5)
    parser.add_argument("--samples", type=int, default=1)
    args = parser.parse_args()

    if not 0.2 <= args.settle_s <= 5.0:
        raise SystemExit("settle-s must be between 0.2 and 5.0 seconds")
    if not 1 <= args.samples <= 50:
        raise SystemExit("samples must be between 1 and 50")

    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    local.ser.port = args.port
    local.ser.baudrate = 115200
    local.local_control_session = False
    page = local.extraWindow()
    meter: AQ6150B | None = None
    point = None
    shutter_confirmed = False
    try:
        meter = AQ6150B(args.meter, average_count=1)
        page.auto_wavelength_spin.setValue(args.target_nm)
        point = page.preview_wavelength_match()
        if point is None:
            raise RuntimeError("2001-point wavelength lookup failed")

        page.output_by_wavelength()
        states = (
            page.gain_state,
            page.soa_state,
            page.phase_state,
            page.wavelena_state,
            page.wavelenb_state,
        )
        if not all("已写入" in state.text() for state in states):
            raise RuntimeError("not all five DAC channels returned a matching ACK")

        time.sleep(args.settle_s)
        readings = []
        for sample in range(1, args.samples + 1):
            reading, attempts = measure_with_retry(meter, point.measured_nm)
            readings.append(reading)
            print(
                "AQ_SAMPLE "
                f"sample={sample}/{args.samples} "
                f"measured={reading.wavelength_nm:.8f} "
                f"power_mw={reading.power_mw:.6f} "
                f"peaks={reading.peak_count} "
                f"smsr_db={reading.side_mode_suppression_db} "
                f"attempts={attempts}",
                flush=True,
            )

        wavelengths = [reading.wavelength_nm for reading in readings]
        powers = [reading.power_mw for reading in readings]
        reading = readings[-1]
        print(
            "SINGLE_VALUE_AQ_OK "
            f"requested={args.target_nm:.6f} "
            f"table={point.target_nm:.6f} "
            f"calibrated={point.measured_nm:.8f} "
            f"measured={reading.wavelength_nm:.8f} "
            f"power_mw={reading.power_mw:.6f} "
            f"power_dbm={reading.power_dbm:.6f} "
            f"peaks={reading.peak_count} "
            f"smsr_db={reading.side_mode_suppression_db} "
            f"samples={len(readings)} "
            f"wavelength_span_pm={(max(wavelengths) - min(wavelengths)) * 1000.0:.6f} "
            f"power_mean_mw={statistics.fmean(powers):.6f} "
            f"power_span_percent={(max(powers) - min(powers)) / statistics.fmean(powers) * 100.0:.6f} "
            f"codes={','.join(str(value) for value in point.codes)}",
            flush=True,
        )
        return 0
    finally:
        try:
            if local.ser.is_open:
                page._write_direct_dac_codes(
                    (0, 0, 0, 0, 0),
                    soa_mode=local.PI11210_SOA_SHUTTER_MODE,
                )
                shutter_confirmed = True
        finally:
            if page._single_value_monitor_is_active():
                with local.ap_cond:
                    local.ap_open = False
                page.stop_recv_thread()
                local.switch_mode_enable = True
            if local.ser.is_open:
                local.ser.close()
            if meter is not None:
                meter.close()
            page.deleteLater()
            application.processEvents()
            print(f"SAFE_SHUTTER_ACK_OK={int(shutter_confirmed)}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
