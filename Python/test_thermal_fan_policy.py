"""Contracts for one mode-independent, fail-safe firmware fan controller."""
import unittest
import re
from pathlib import Path

from test_firmware_scan_scheduler import _function


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "JDSU/Core/Src/temperature.c"
HEADER = ROOT / "JDSU/Core/Inc/temperature.h"


class ThermalFanPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text(encoding="utf-8")
        cls.header = HEADER.read_text(encoding="utf-8")

    def test_all_work_modes_share_the_same_temperature_policy(self):
        # Neither normal control nor safety housekeeping may special-case a
        # scan/calibration/idle mode, irrespective of USB or LAN ownership.
        self.assertNotIn("workState", self.source)
        self.assertNotIn("EXTRA_STATE", self.source)
        self.assertNotIn("usbCdcHostOpen", self.source)
        self.assertIn("ApplyFanCommand(output, dt, 0U);",
                      _function(self.source, "ApplyPid"))

    def test_requested_target_is_40_c_with_other_pid_parameters_unchanged(self):
        expected = {
            "FAN_TARGET_C": 40.0,
            "FAN_PID_BASE_PERCENT": 52.0,
            "FAN_PID_KP": 24.0,
            "FAN_PID_KI": 0.55,
            "FAN_PID_KD": 2.5,
            "FAN_MIN_EFFECTIVE_PERCENT": 25.0,
        }
        for name, value in expected.items():
            with self.subTest(name=name):
                match = re.search(r"^#define\s+" + name + r"\s+([\d.]+)",
                                  self.source, re.MULTILINE)
                self.assertIsNotNone(match)
                self.assertEqual(float(match[1]), value)
        self.assertIn("filteredTemperatureC - FAN_TARGET_C",
                      _function(self.source, "ApplyPid"))
        self.assertIn("filteredTemperatureC >= FAN_TARGET_C",
                      _function(self.source, "ThermalControl_GetStatusFlags"))

    def test_raw_pwm_is_private_to_the_shared_controller(self):
        self.assertIn("static void Set_Soft_PWM_Duty(", self.source)
        self.assertNotIn("Set_Soft_PWM_Duty", self.header)
        for path in (ROOT / "JDSU/Core/Src").glob("*.c"):
            if path != SOURCE:
                self.assertNotIn("Set_Soft_PWM_Duty(",
                                 path.read_text(encoding="utf-8"), str(path))

    def test_startup_is_full_speed_until_temperature_is_known(self):
        self.assertIn("Set_Soft_PWM_Duty(100U);",
                      _function(self.source, "ThermalControl_Init"))

    def test_sensor_failure_and_stale_reading_still_force_full_speed(self):
        failure = _function(self.source, "M1820RecordError")
        safety = _function(self.source, "ThermalControl_UpdateSafety")
        self.assertIn("if(!temperatureValid) ApplyFanCommand(100.0f, 0.25f, 1U);",
                      failure)
        self.assertIn("M1820_STALE_TIMEOUT_MS", safety)
        self.assertIn("temperatureValid = 0U;", safety)
        self.assertIn("ApplyFanCommand(100.0f, 0.25f, 1U);", safety)

    def test_pwm_fault_remains_latched_despite_later_pid_updates(self):
        apply = _function(self.source, "ApplyFanCommand")
        pwm = _function(self.source, "Set_Soft_PWM_Duty")
        fault = _function(self.source, "ThermalControl_OnFanPwmFault")
        self.assertIn("if(fanPwmFault) forceFull = 1U;", apply)
        self.assertIn("if(fanPwmFault) cooling_percent = 100U;", pwm)
        self.assertIn("fanPwmFault = 1U;", fault)
        self.assertIn("fanCoolingPercent = 100U;", fault)
        self.assertIn("HAL_GPIO_WritePin(FAN_PORT, FAN_PIN, GPIO_PIN_RESET);", fault)

    def test_pid_ramp_limits_and_physical_pwm_polarity_are_retained(self):
        apply = _function(self.source, "ApplyFanCommand")
        pwm = _function(self.source, "Set_Soft_PWM_Duty")
        self.assertIn("FAN_NORMAL_RISE_PERCENT_PER_S * dt", apply)
        self.assertIn("FAN_NORMAL_FALL_PERCENT_PER_S * dt", apply)
        self.assertIn("FAN_MIN_EFFECTIVE_PERCENT", apply)
        self.assertIn("if(index < cooling_percent)", pwm)
        self.assertIn("((uint32_t)FAN_PIN << 16U)", pwm)


if __name__ == "__main__":
    unittest.main()
