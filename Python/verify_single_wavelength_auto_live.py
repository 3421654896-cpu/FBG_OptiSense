"""Hardware smoke test for the single-value wavelength auto-output UI path."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtWidgets

import app_JDSU as local


def main() -> int:
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    local.ser.port = "COM6"
    local.ser.baudrate = 115200
    local.local_control_session = False
    page = local.extraWindow()
    passed = False
    try:
        page.auto_wavelength_spin.setValue(1528.461)
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
        print(
            f"AUTO_OUTPUT_OK requested=1528.461000 "
            f"matched={point.measured_nm:.8f} index={point.index} "
            f"codes={','.join(str(value) for value in point.codes)}",
            flush=True,
        )
        passed = True
    finally:
        if page._single_value_monitor_is_active():
            try:
                codes = (
                    page.gain,
                    page.soa,
                    page.phase,
                    page.wavelena,
                    page.wavelenb,
                )
                page._write_direct_dac_codes(
                    codes, soa_mode=local.PI11210_SOA_SHUTTER_MODE
                )
                print("SAFE_SHUTTER_ACK_OK", flush=True)
            finally:
                with local.ap_cond:
                    local.ap_open = False
                page.stop_recv_thread()
                local.switch_mode_enable = True
        if local.ser.is_open:
            local.ser.close()
        page.deleteLater()
        application.processEvents()
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
