"""Backward-compatibility stub for the removed 3-D/Mach3 finger window.

The visible stress workspace is now the fixed 25-point manual grid
(:mod:`manual_stress_capture`).  The app still has live code paths (stress
GraphWindow callbacks, remote viewer, fast-flank worker, Mach3 telemetry) that
call into the old window's API.  This sink accepts those calls cheaply so the
rest of the application keeps working without the heavy OpenGL scene.
"""

from __future__ import annotations

from PyQt5 import QtCore


class LegacyFingerControllerStub(QtCore.QObject):
    """No-op sink replacing MechanicalFinger3DWindow."""

    press_target_requested = QtCore.pyqtSignal(dict)
    coverage_run_requested = QtCore.pyqtSignal(dict)
    coverage_stop_requested = QtCore.pyqtSignal()
    stress_sequence_requested = QtCore.pyqtSignal(dict)
    fast_flank_start_requested = QtCore.pyqtSignal()
    fast_flank_stop_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.press_depth_spin = self._Spin(0.10)
        self.xy_speed_spin = self._Spin(60.0)
        self.press_speed_spin = self._Spin(15.0)
        self.coverage_dwell_spin = self._Spin(1.0)
        self.click_press_arm = self._Check(False)
        self.click_press_status_label = self._Label()
        self.coverage_summary_label = self._Label()
        self.source_label = self._Label()
        self.current_peaks_by_channel = None
        self.machine_touch_samples = ()

    class _Spin:
        def __init__(self, value):
            self._value = float(value)

        def value(self):
            return self._value

    class _Check:
        def __init__(self, value):
            self._value = bool(value)

        def isChecked(self):
            return self._value

        def setChecked(self, value):
            self._value = bool(value)

        def setEnabled(self, value):
            pass

    class _Label:
        def __init__(self):
            self._text = ""

        def setText(self, text):
            self._text = str(text)

        def text(self):
            return self._text

    # Attribute writes used by the stress page (e.g. finger_3d_open_handler).
    def __setattr__(self, name, value):
        super().__setattr__(name, value)

    # ---- API surface historically consumed by the app ----
    def update_machine_status(self, status):
        pass

    def set_machine_status_error(self, detail):
        pass

    def update_from_spectrum(self, *args, **kwargs):
        pass

    def update_from_contact_estimate(self, *args, **kwargs):
        pass

    def update_from_fast_flank_result(self, result):
        pass

    def set_fast_flank_running(self, running):
        pass

    def set_fast_flank_progress(self, completed, total):
        pass

    def finish_fast_flank(self, message, *, success=False):
        pass

    def set_coverage_running(self, running):
        pass

    def set_coverage_progress(self, *args, **kwargs):
        pass

    def remember_click_press_location(self, request):
        pass

    def open_finger_tab(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def raise_(self):
        pass

    def activateWindow(self):
        pass
