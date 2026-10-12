"""Static regression tests for the latency-sensitive firmware scan loop.

These tests deliberately do not emulate STM32 peripherals.  They protect the
important scheduling invariant: optical point boundaries service the fast
fail-safe, PI11210 protection and Wi-Fi/OTA; frame-boundary temperature work
advances at most one sub-millisecond 1-Wire primitive.  This PCB's CH224Q pins
are hardware configuration straps and must never be configured as I2C.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MS5614T = ROOT / "JDSU" / "Core" / "Src" / "ms5614t.c"
TEMPERATURE = ROOT / "JDSU" / "Core" / "Src" / "temperature.c"
TEMPERATURE_H = ROOT / "JDSU" / "Core" / "Inc" / "temperature.h"
CH224Q = ROOT / "JDSU" / "Core" / "Src" / "ch224q.c"
MAIN = ROOT / "JDSU" / "Core" / "Src" / "main.c"
MAIN_H = ROOT / "JDSU" / "Core" / "Inc" / "main.h"
MSP = ROOT / "JDSU" / "Core" / "Src" / "stm32f2xx_hal_msp.c"


def _function(source: str, name: str) -> str:
    # Match a declaration with a return type, not an earlier call inside if().
    match = re.search(rf"^[ \t]*(?:static\s+)?[A-Za-z_]\w*\s+{name}\s*\([^;{{}}]*?\)\s*\{{", source, re.M)
    if match is None:
        raise AssertionError(f"function not found: {name}")
    depth = 1
    cursor = match.end()
    while cursor < len(source) and depth:
        if source[cursor] == "{":
            depth += 1
        elif source[cursor] == "}":
            depth -= 1
        cursor += 1
    if depth:
        raise AssertionError(f"unterminated function: {name}")
    return source[match.start() : cursor]


class FirmwareScanSchedulerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scan_source = MS5614T.read_text(encoding="utf-8")
        cls.thermal_source = TEMPERATURE.read_text(encoding="utf-8")
        cls.thermal_header = TEMPERATURE_H.read_text(encoding="utf-8")
        cls.ch224q_source = CH224Q.read_text(encoding="utf-8")
        cls.main_source = MAIN.read_text(encoding="utf-8")
        cls.main_header = MAIN_H.read_text(encoding="utf-8")
        cls.msp_source = MSP.read_text(encoding="utf-8")

    def test_point_loop_keeps_safety_and_ota_responsive(self) -> None:
        scan = _function(self.scan_source, "write_ms5614t_table")
        self.assertIn("ThermalControl_SafetyTick();", scan)
        self.assertIn("PI11210_Process();", scan)
        self.assertIn("WifiTransport_Process();", scan)
        self.assertIn("WifiTransport_IsOtaActive()", scan)

        ota_check = scan.index("WifiTransport_IsOtaActive()")
        first_dac_apply = min(
            index
            for token in (
                "PI11210_ApplyCalibrationCodes(",
                "PI11210_ApplyStressRowCalibrationOrder()",
                "PI11210_ApplyChangedTableCodes()",
            )
            if (index := scan.find(token)) >= 0
        )
        self.assertLess(ota_check, first_dac_apply)

    def test_cooperative_board_maintenance_is_not_in_optical_frame(self) -> None:
        scan = _function(self.scan_source, "write_ms5614t_table")
        maintenance = re.search(
            r"if\(workState == TABLE_STATE\)\s*\{(?P<stress>.*?)\}"
            r"\s*else\s*\{(?P<precision>.*?)\}\s*PI11210_Process\(\);",
            scan,
            re.S,
        )
        self.assertIsNotNone(maintenance)
        assert maintenance is not None
        stress = maintenance.group("stress")
        precision = maintenance.group("precision")
        self.assertIn("ThermalControl_SafetyTick();", stress)
        self.assertNotIn("ThermalControl_Process();", stress)
        self.assertNotIn("CH224Q_Process();", stress)

        # Slow precision scans retain per-point cooperative maintenance;
        # otherwise a >1.5 s frame could stale the temperature controller.
        self.assertIn("ThermalControl_Process();", precision)
        self.assertIn("CH224Q_Process();", precision)

        # main advances temperature immediately around calls to the complete-
        # frame scanner, including while USB DMA prevents a new frame start.
        self.assertGreaterEqual(self.main_source.count("ThermalControl_Process();"), 3)

    def test_seven_point_thermal_service_precedes_dac_settling(self) -> None:
        stream = _function(self.scan_source, 'runFastFlankStream')
        loop = stream[stream.index('for(uint8_t local = 0U; local < pointCount; local++)'):]
        service = re.search(r'if\(temporary45Stream && \(route == 6U \|\| route == 7U\) &&\s*'
                            r'local % candidateRoute.pointsPerPeak == 0U\)\s*'
                            r'ThermalControl_Process\(\);', loop)
        self.assertIsNotNone(service)
        self.assertLess(service.start(), loop.index('PI11210_ApplyCalibrationCodes('))
        self.assertLess(service.start(), loop.index('uint32_t writeEnd = DWT->CYCCNT;'))
        # Do not solve starvation by relaxing the genuine sensor fail-safe.
        self.assertIn('#define M1820_STALE_TIMEOUT_MS            1500U', self.thermal_source)
        self.assertIn('#define FAN_TARGET_C                      40.0f', self.thermal_source)

    def test_seven_point_short_hold_has_enough_thermal_service_slots(self) -> None:
        # Worst case: each of the 17 cooperative state advances waits until
        # the next eligible slot. Only one boundary slot at 92 ms/frame was
        # >1.5 s including the next-conversion interval. Group service is
        # nine slots/frame without borrowing any DAC-to-ADC hold time.
        states = 17
        frame_ms = 92
        next_conversion_ms = 250
        self.assertGreater(states * frame_ms + next_conversion_ms, 1500)
        group_slots = [i * frame_ms / 9 for i in range(1, 19)]
        self.assertLess(group_slots[states - 1] + next_conversion_ms, 1500)

    def test_frame_temperature_trailer_is_cached_only(self) -> None:
        sample = _function(self.scan_source, "sampleTemperature")
        self.assertIn("ThermalControl_SafetyTick();", sample)
        self.assertNotIn("ThermalControl_Process();", sample)

    def test_fast_thermal_path_cannot_start_one_wire_transaction(self) -> None:
        safety = _function(self.thermal_source, "ThermalControl_SafetyTick")
        helper = _function(self.thermal_source, "ThermalControl_UpdateSafety")
        combined = safety + helper
        self.assertNotIn("M1820StartConversion", combined)
        self.assertNotIn("M1820ReadTemperature", combined)
        self.assertNotIn("delay_us", combined)
        self.assertIn("ThermalControl_SafetyTick", self.thermal_header)

    def test_full_thermal_service_is_one_cooperative_primitive(self) -> None:
        full = _function(self.thermal_source, "ThermalControl_Process")
        self.assertIn("ThermalControl_UpdateSafety(now);", full)
        self.assertIn("switch(sensorState)", full)
        self.assertNotIn("M1820StartConversion", self.thermal_source)
        self.assertNotIn("M1820ReadTemperature", self.thermal_source)

        # Every timed state returns immediately after its one bus primitive;
        # scratchpad reception reads one byte rather than all nine in a loop.
        for state, primitive in (
            ("M1820_CONVERT_RESET", "M1820Z_Reset()"),
            ("M1820_CONVERT_SKIP_ROM", "M1820Z_WriteByte(SKIP_ROM)"),
            ("M1820_CONVERT_COMMAND", "M1820Z_WriteByte(CONVERT_T)"),
            ("M1820_READ_RESET", "M1820Z_Reset()"),
            ("M1820_READ_SKIP_ROM", "M1820Z_WriteByte(SKIP_ROM)"),
            ("M1820_READ_COMMAND", "M1820Z_WriteByte(READ_SCRATCHPAD)"),
            ("M1820_READ_SCRATCHPAD", "M1820Z_ReadByte()"),
        ):
            arm = full[full.index(f"case {state}:") :]
            end = arm.find("case ", 5)
            if end >= 0:
                arm = arm[:end]
            self.assertIn(primitive, arm)
            self.assertEqual(
                sum(
                    arm.count(token)
                    for token in ("M1820Z_Reset()", "M1820Z_WriteByte(", "M1820Z_ReadByte()")
                ),
                1,
                state,
            )
            self.assertIn("return;", arm)

        read_arm = full[full.index("case M1820_READ_SCRATCHPAD:") :]
        self.assertNotRegex(read_arm, r"for\s*\([^)]*M1820")
        self.assertIn("M1820_COOPERATIVE_GUARD_MS", self.thermal_source)

    def test_one_wire_primitive_nominal_budget_is_below_one_millisecond(self) -> None:
        reset = _function(self.thermal_source, "M1820Z_Reset")
        write_bit = _function(self.thermal_source, "M1820Z_WriteBit")
        read_bit = _function(self.thermal_source, "M1820Z_ReadBit")
        self.assertEqual(
            sum(int(value) for value in re.findall(r"delay_us\((\d+)U\)", reset)),
            980,
        )
        # The longest write-bit arm is 70 us; a complete byte is 560 us.
        write_delays = [int(value) for value in re.findall(r"delay_us\((\d+)U\)", write_bit)]
        self.assertEqual(max(write_delays[0] + write_delays[1], write_delays[2] + write_delays[3]), 70)
        self.assertEqual(
            sum(int(value) for value in re.findall(r"delay_us\((\d+)U\)", read_bit)),
            62,
        )

    def test_ch224q_hardware_straps_are_never_driven_as_i2c(self) -> None:
        combined = self.ch224q_source + self.main_source + self.main_header + self.msp_source
        self.assertNotIn("HAL_I2C_IsDeviceReady", self.ch224q_source)
        self.assertNotIn("HAL_I2C_Mem_Read", self.ch224q_source)
        self.assertNotIn("hi2c2", combined)
        self.assertNotIn("MX_I2C2_Init", combined)
        self.assertNotIn("GPIO_AF4_I2C2", self.msp_source)
        self.assertNotRegex(
            self.msp_source,
            r"GPIO_PIN_10\s*\|\s*GPIO_PIN_11",
        )

        init = _function(self.ch224q_source, "CH224Q_Init")
        process = _function(self.ch224q_source, "CH224Q_Process")
        self.assertIn("CH224Q_BOARD_REQUEST_MV", init)
        self.assertNotIn("status.online = 1", self.ch224q_source)
        self.assertNotIn("status.current_valid = 1", self.ch224q_source)
        self.assertNotRegex(process, r"HAL_|GPIO|I2C")

    def test_multirate_accepts_only_v3_or_v4_and_rejects_fail_closed(self) -> None:
        command = _function(self.scan_source, "modify_table_loop")
        self.assertIn("#define STRESS_MULTIRATE_EXTENSION_VERSION_V3 3U", self.scan_source)
        self.assertIn("#define STRESS_MULTIRATE_EXTENSION_VERSION_V4 4U", self.scan_source)
        self.assertRegex(
            self.scan_source,
            r"static\s+volatile\s+uint8_t\s+"
            r"stressMultirateNegotiatedVersion\s*=\s*0U;",
        )
        self.assertRegex(
            command,
            r"requestedVersion\s*==\s*STRESS_MULTIRATE_EXTENSION_VERSION_V3\s*\|\|\s*"
            r"requestedVersion\s*==\s*STRESS_MULTIRATE_EXTENSION_VERSION_V4",
        )
        self.assertRegex(
            command,
            r"accepted\s*=\s*\(uint8_t\)\(workState\s*==\s*EXTRA_STATE\s*&&\s*"
            r"enable\s*<=\s*1U\s*&&\s*versionSupported\s*&&\s*"
            r"\(!enable\s*\|\|\s*FBG_FEATURE_LEGACY_PEAK_MODES\)\)",
        )
        # Invalid enable/configuration/version requests all take the same
        # fail-closed path rather than retaining an earlier arm.
        self.assertGreaterEqual(
            command.count("else stressMultirateDisarmAndClear();"), 2
        )

    def test_multirate_session_version_controls_profile_three(self) -> None:
        command = _function(self.scan_source, "modify_table_loop")
        apply_state = _function(self.scan_source, "ApplyWorkState")
        self.assertIn(
            "stressMultirateNegotiatedVersion = requestedVersion;", command
        )
        self.assertRegex(
            command,
            r"StressMultirate_SetSingleTrackEnabled\(\s*"
            r"&stressMultirateState,\s*\(uint8_t\)\(requestedVersion\s*>=\s*"
            r"STRESS_MULTIRATE_EXTENSION_VERSION_V4\)\)",
        )
        configure = apply_state.index("StressMultirate_Configure(")
        capability = apply_state.index("StressMultirate_SetSingleTrackEnabled(")
        start = apply_state.index("StressMultirate_StartSession(")
        self.assertLess(configure, capability)
        self.assertLess(capability, start)
        self.assertRegex(
            apply_state,
            r"stressMultirateArmed\s*&&\s*stressMultirateNegotiatedVersion\s*>=\s*"
            r"STRESS_MULTIRATE_EXTENSION_VERSION_V4",
        )
        # >=4 is safe only behind the exact supported-version whitelist.
        supported = re.search(r"versionSupported\s*=\s*\(uint8_t\)\((.*?)\);",command,re.S).group(1)
        self.assertEqual(re.findall(r"requestedVersion\s*==\s*STRESS_MULTIRATE_EXTENSION_VERSION_V(\d)",supported),['3','4','5'])
        g8 = apply_state.index("StressMultirate_SetG8RightSentinel(")
        self.assertLess(configure,g8)
        self.assertLess(g8,start)
        self.assertRegex(apply_state[g8:start],r"stressMultirateArmed\s*&&\s*stressMultirateNegotiatedVersion\s*==\s*STRESS_MULTIRATE_EXTENSION_VERSION_V5")

    def test_multirate_frame_and_ack_echo_session_version(self) -> None:
        scan = _function(self.scan_source, "write_ms5614t_table")
        command = _function(self.scan_source, "modify_table_loop")
        self.assertIn(
            "txBuffer[txCount++] = stressMultirateNegotiatedVersion;", scan
        )
        self.assertIn(
            "aTxBuffer[11] = (accepted && enable)", command
        )
        self.assertIn(
            "? stressMultirateNegotiatedVersion : requestedVersion;", command
        )

    def test_multirate_disarm_clears_capability_and_scheduler_state(self) -> None:
        disarm = _function(self.scan_source, "stressMultirateDisarmAndClear")
        self.assertIn("stressMultirateArmed = 0U;", disarm)
        self.assertIn("stressMultirateNegotiatedVersion = 0U;", disarm)
        self.assertIn("StressMultirate_Configure(&stressMultirateState,", disarm)
        self.assertIn(
            "StressMultirate_SetSingleTrackEnabled(&stressMultirateState, 0U);",
            disarm,
        )
        self.assertIn("StressMultirate_StartSession(&stressMultirateState);", disarm)


if __name__ == "__main__":
    unittest.main()
