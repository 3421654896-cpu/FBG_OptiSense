"""Live CH0/CH1 single-value TIA selector and IO-readback smoke test."""

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
        if not page._start_single_value_monitor():
            raise RuntimeError("single-value monitor did not start")

        # Keep the laser dark while exercising only the analogue front end.
        page._write_direct_dac_codes(
            (0, 0, 0, 0, 0), soa_mode=local.PI11210_SOA_SHUTTER_MODE
        )
        selector_pairs = (
            (1, 1),
            (3, 1),
            (2, 1),
            (0, 1),
            (1, 3),
            (1, 2),
            (1, 0),
            (1, 1),
        )
        for expected in selector_pairs:
            page.ch0_feedback_btn.setProperty("feedbackSelector", expected[0])
            page.ch1_feedback_btn.setProperty("feedbackSelector", expected[1])
            page.feedback_selection_dirty = True
            if not page._apply_single_value_feedback(
                ensure_monitor=False, show_error=False
            ):
                raise RuntimeError(
                    f"feedback IO verification failed for selectors {expected}"
                )
            if page.confirmed_feedback_selectors != expected:
                raise RuntimeError(
                    f"feedback mismatch: requested={expected}, "
                    f"confirmed={page.confirmed_feedback_selectors}"
                )
            resistances = page._feedback_resistances(expected)
            print(
                f"TIA_IO_ACK CH0={resistances[0]}k CH1={resistances[1]}k",
                flush=True,
            )
        passed = True
    finally:
        if page._single_value_monitor_is_active():
            try:
                page._write_direct_dac_codes(
                    (0, 0, 0, 0, 0),
                    soa_mode=local.PI11210_SOA_SHUTTER_MODE,
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
