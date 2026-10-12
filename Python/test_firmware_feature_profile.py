"""Check the real C preprocessor's customer and recoverable developer profiles.

These checks never open a serial port or debug probe. Compiler-dependent tests
are skipped on hosts without armclang, rather than treating raw #if text as proof
that a feature has actually been excluded.
"""
from functools import lru_cache
from pathlib import Path
import os
import re
import shutil
import subprocess
import unittest

from test_firmware_scan_scheduler import _function

ROOT = Path(__file__).resolve().parents[1]
FIRMWARE = ROOT / "JDSU"
HEADER = FIRMWARE / "Core/Inc/firmware_features.h"
COMPILER = (os.environ.get("FBG_ARMCLANG") or shutil.which("armclang")
            or r"D:\Keil_v5\ARM\ARMCLANG\Bin\armclang.exe")
FEATURES = ("LEGACY_PEAK_MODES", "MQTT")
INCLUDES = (
    "Core/Inc", "Drivers/STM32F2xx_HAL_Driver/Inc",
    "Drivers/STM32F2xx_HAL_Driver/Inc/Legacy",
    "Drivers/CMSIS/Device/ST/STM32F2xx/Include", "Drivers/CMSIS/Include",
    "USB_DEVICE/App", "USB_DEVICE/Target",
    "Middlewares/ST/STM32_USB_Device_Library/Core/Inc",
    "Middlewares/ST/STM32_USB_Device_Library/Class/CDC/Inc",
)


@lru_cache(maxsize=None)
def preprocess(source, profile=()):
    command = [COMPILER, "--target=arm-arm-none-eabi", "-mcpu=cortex-m3",
               "-mthumb", "-E", "-P", "-DUSE_HAL_DRIVER", "-DSTM32F205xx"]
    for include in INCLUDES:
        command += ["-I", str(FIRMWARE / include)]
    command += [f"-DFBG_FEATURE_{name}={value}" for name, value in profile]
    command += [str(FIRMWARE / "Core/Src" / source)]
    result = subprocess.run(command, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=30)
    if result.returncode:
        raise AssertionError(result.stderr)
    return result.stdout


class FeatureDefaultsTests(unittest.TestCase):
    def test_hidden_independent_features_default_off_and_allow_override(self):
        source = HEADER.read_text(encoding="utf-8")
        for name in FEATURES:
            self.assertRegex(source, rf"#ifndef FBG_FEATURE_{name}\s+"
                             rf"#define FBG_FEATURE_{name} 0\s+#endif")
        self.assertIn('#error Network firmware updates have been removed', source)
        self.assertIn('#include "firmware_features.h"',
                      (FIRMWARE / "Core/Inc/main.h").read_text(encoding="utf-8"))


@unittest.skipUnless(Path(COMPILER).is_file() or shutil.which(COMPILER),
                     "armclang is needed to verify compiled feature profiles")
class PreprocessedProfileTests(unittest.TestCase):
    def test_customer_main_only_dispatches_visible_mcu_modes(self):
        body = _function(preprocess("main.c"), "main")
        self.assertNotIn("write_ms5614t_table();", body)
        self.assertNotIn("case TABLE_STATE:", body)
        self.assertNotIn("case PRECISION_TABLE_STATE:", body)
        for call in ("write_ms5614t_manual();", "write_ms5614t_extra();",
                     "OtaUpdate_ProcessUsb();", "ThermalControl_Process();"):
            self.assertIn(call, body)

    def test_old_scan_request_fails_dark_before_switching_to_extra(self):
        body = _function(preprocess("ms5614t.c"), "ApplyWorkState")
        self.assertIn("newState == TABLE_STATE || newState == PRECISION_TABLE_STATE", body)
        self.assertLess(body.index("PI11210_ForceShutter();"),
                        body.index("newState = EXTRA_STATE;"))
        self.assertLess(body.index("newState = EXTRA_STATE;"),
                        body.index("workState = newState;"))
        self.assertNotIn("case TABLE_STATE:", body)
        self.assertNotIn("case PRECISION_TABLE_STATE:", body)
        self.assertNotIn("StressMultirate_Configure", body)
        self.assertNotIn("unstableFlags", body)

    def test_dense_finger_single_and_calibration_paths_remain(self):
        source = preprocess("ms5614t.c")
        with self.assertRaises(AssertionError):
            _function(source, "write_ms5614t_table")
        extra = _function(source, "write_ms5614t_extra")
        for call in ("runFastFlankStream();", "runDenseScanBlock();", "singleValue();"):
            self.assertIn(call, extra)
        _function(source, "write_ms5614t_manual")
        _function(source, "runFastFlankStream")
        _function(source, "runDenseScanBlock")
        command = _function(source, "modify_table_loop")
        self.assertRegex(command, r"accepted\s*=\s*\(uint8_t\)\(workState\s*==\s*EXTRA_STATE\s*&&")
        self.assertRegex(command, r"\(!enable\s*\|\|\s*0\)")
        self.assertNotIn("getFilterDiff();", command)
        self.assertNotIn("getWaveRange();", command)
        self.assertNotRegex(source, r"int8_t\s+unstableFlags\[")

    def test_safe_disarm_survives_trim_without_reenabling_legacy_arm(self):
        body = _function(preprocess("ms5614t.c"), "modify_table_loop")
        self.assertIn("enable <= 1U && versionSupported && (!enable || 0)",
                      " ".join(body.split()))
        self.assertIn("else stressMultirateDisarmAndClear();", body)
        self.assertIn("aTxBuffer[10] = accepted ? 0x21U : 0xE1U;", body)
        self.assertIn("aTxBuffer[11] = (accepted && enable)", body)
        # The recoverable developer profile must still allow deliberate ARM.
        developer = _function(preprocess("ms5614t.c", (("LEGACY_PEAK_MODES", 1),)),
                              "modify_table_loop")
        self.assertIn("(!enable || 1)", developer)

    def test_network_ota_and_mqtt_entries_are_removed_but_lan_is_retained(self):
        source = preprocess("wifi_transport.c")
        for name in ("ParseRemoteCommand", "StartNextPublish", "SignWanOtaResponse"):
            with self.subTest(name=name), self.assertRaises(AssertionError):
                _function(source, name)
        self.assertNotIn("OtaUpdate_HandleCommand(",
                         _function(source, "ParseLanPayload"))
        self.assertIn("return 1U;", _function(source, "WifiLanOnly"))
        self.assertIn("ParseLanSocketLine(line);", _function(source, "ParseAtLine"))
        _function(source, "ParseLanRawCommand")
        _function(source, "StartNextLanTransfer")
        self.assertNotIn("StartNextPublish();",
                         _function(source, "WifiTransport_Process"))

    def test_usb_update_is_not_removed_with_network_ota(self):
        source = preprocess("ota_update.c")
        body = _function(source, "OtaUpdate_ProcessUsb")
        self.assertIn("OtaUpdate_HandleCommand(", body)
        self.assertRegex(body, r"OtaUpdate_HandleCommand\(command,\s*length,\s*key,\s*3U,")
        self.assertIn("PI11210_SetSOAShutter(1U)", body)
        self.assertIn("NVIC_SystemReset();", body)

    def test_update_engine_only_accepts_physical_usb_and_is_private(self):
        source = preprocess("ota_update.c")
        body = _function(source, "OtaUpdate_HandleCommand")
        self.assertIn("transportKind != 3U", body)
        self.assertIn("static uint8_t OtaUpdate_HandleCommand", source)
        self.assertNotIn("OtaUpdate_HandleCommand", (FIRMWARE / "Core/Inc/ota_update.h").read_text(encoding="utf-8"))
        for name in ("HandlePing", "HandleAbort", "OtaUpdate_ReleaseCommittedSession"):
            with self.assertRaises(AssertionError):
                _function(source, name)
        self.assertNotIn("suppliedMac", _function(source, "HandleEnd"))
        self.assertIn("PI11210_SetSOAShutter(1U) != HAL_OK", _function(source, "HandleBegin"))
        self.assertNotIn("PI11210_SetSOAShutter(0U)", _function(source, "HandleBegin"))

    def test_network_update_cannot_be_restored_by_a_build_flag(self):
        with self.assertRaisesRegex(AssertionError, "Network firmware updates have been removed"):
            preprocess("wifi_transport.c", (("NETWORK_OTA", 1),))
        source = preprocess("wifi_transport.c", (("MQTT", 1),))
        for token in ("otaResponse", "wanOtaLastCommandMs", "otaResetAfterLanDelete",
                      "WIFI_PUBLISH_OTA", "OtaUpdate_HandleCommand"):
            self.assertNotIn(token, source)
        for name in ("ParseLanRawCommand", "StartNextLanTransfer", "ParseLanSocketLine"):
            _function(source, name)

    def test_developer_override_restores_backed_up_features(self):
        developer = tuple((name, 1) for name in FEATURES)
        _function(preprocess("ms5614t.c", developer), "write_ms5614t_table")
        body = _function(preprocess("main.c", developer), "main")
        self.assertIn("write_ms5614t_table();", body)
        source = preprocess("wifi_transport.c", developer)
        for name in ("ParseRemoteCommand", "StartNextPublish"):
            _function(source, name)
        self.assertNotIn("OtaUpdate_HandleCommand(", _function(source, "ParseLanPayload"))
        self.assertNotIn("OtaUpdate_HandleCommand(", _function(source, "ParseRemoteCommand"))

    def test_mqtt_override_alone_cannot_reenable_network_ota(self):
        source = preprocess("wifi_transport.c", (("MQTT", 1),))
        self.assertNotIn("OtaUpdate_HandleCommand(", _function(source, "ParseRemoteCommand"))
        self.assertNotIn("OtaUpdate_HandleCommand(", _function(source, "ParseLanPayload"))


if __name__ == "__main__":
    unittest.main()
