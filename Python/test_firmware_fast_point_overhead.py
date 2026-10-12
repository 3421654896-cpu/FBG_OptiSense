"""Timing optimizations must preserve the calibrated path and fail-safe logic."""
import binascii
import random
import re
import unittest
from pathlib import Path

from test_firmware_scan_scheduler import _function


ROOT = Path(__file__).resolve().parents[1]


class FastPointOverheadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (ROOT / 'JDSU/Core/Src/ms5614t.c').read_text(encoding='utf-8')

    def test_register_transmit_keeps_address_msb_lsb_and_failure_shutter(self):
        write = _function(self.source, 'PI11210_WriteRegisterRaw')
        self.assertIn('(uint16_t)(registerAddress << 1)', write)
        self.assertIn('data[0] = (uint8_t)(value >> 8);', write)
        self.assertIn('data[1] = (uint8_t)value;', write)
        self.assertIn('HAL_I2C_Mem_Write', write)
        self.assertIn('I2C_MEMADD_SIZE_8BIT, data, 2U', write)
        self.assertNotIn('result = HAL_I2C_Master_Transmit', write)
        self.assertIn('attempt < 2U', write)
        self.assertIn('PI11210_RecoverI2CBus()', write)
        self.assertIn('PI11210_CountI2CError()', write)
        self.assertIn('PI11210_ForceShutter()', write)
        main = (ROOT / 'JDSU/Core/Src/main.c').read_text(encoding='utf-8')
        self.assertIn('hi2c1.Init.ClockSpeed = 400000;', main)

    def test_batch_checks_readiness_once_but_keeps_all_five_ordered_writes(self):
        batch = _function(self.source, 'PI11210_ApplyCalibrationCodes')
        self.assertEqual(batch.count('PI11210_EnsureReady()'), 1)
        self.assertIn('channelIndex < 5U', batch)
        self.assertIn('code = codes[channelIndex]', batch)
        self.assertIn('PI11210_WriteReadyCode(tableDACChannels[channelIndex], code)', batch)
        self.assertNotIn('prevDAC[channelIndex] !=', batch)
        self.assertNotIn('PI11210_Init()', batch)
        self.assertRegex(batch, r'else\s*\{[\s\S]*?prevDACValid = 0U;[\s\S]*?'
                         r'PI11210_ForceShutter\(\);\s*return HAL_ERROR;')
        ready = _function(self.source, 'PI11210_WriteReadyCode')
        self.assertIn('PI11210_LimitCode(channel, code)', ready)
        self.assertIn('PI11210_STATUS_PRO_TEMP | PI11210_STATUS_OVR_TEMP', ready)
        self.assertIn('HAL_GPIO_ReadPin(PI11210_CLR_PORT, PI11210_CLR_PIN)', ready)
        self.assertIn('PI11210_ForceShutter()', ready)
        public = _function(self.source, 'PI11210_SetCode')
        self.assertIn('default: return HAL_ERROR;', public)
        self.assertIn('PI11210_EnsureReady()', public)

    def test_cycle_wait_preserves_deadline_and_protection_service(self):
        wait = _function(self.source, 'stressHoldWait')
        self.assertIn('offsetUs > 0xFFFFFFFFUL / cyclesPerUs', wait)
        loop = wait[wait.index('while('):]
        self.assertNotIn('/ cyclesPerUs', loop)
        self.assertIn('targetCycles = offsetUs * cyclesPerUs', wait)
        self.assertIn('serviceCycles = 1000U * cyclesPerUs', wait)
        for token in ('ReceEndFlag', 'fastStreamControlActive()',
                      'ThermalControl_Process()', 'ThermalControl_SafetyTick()',
                      'PI11210_Process()', 'PI11210_SOA_SOURCE'):
            self.assertIn(token, loop)
        # Integer deadline equivalence, including DWT wrap and zero-length hold.
        for per_us in (1, 72, 120, 168):
            for offset in (0, 50, 850, 1700, 15000, 800000):
                for elapsed in (0, max(0, offset * per_us - 1),
                                offset * per_us, offset * per_us + 1):
                    start = 0xFFFF0000
                    delta = ((start + elapsed) - start) & 0xFFFFFFFF
                    self.assertEqual(delta // per_us < offset,
                                     delta < offset * per_us)

    def test_nibble_crc_matches_wire_crc_for_packets_and_partial_states(self):
        body = _function(self.source, 'routeTestCrcByte')
        table_text = re.search(r'nibble\[16\]\s*=\s*\{([^}]+)\}', body).group(1)
        table = [int(value, 16) for value in re.findall(r'0x([0-9A-F]+)UL', table_text)]
        self.assertEqual(len(table), 16)
        self.assertEqual(body.count('(crc >> 4) ^ nibble[crc & 0x0FU]'), 2)
        rng = random.Random(111)
        def optimized(crc, value):
            crc ^= value
            crc = (crc >> 4) ^ table[crc & 15]
            return (crc >> 4) ^ table[crc & 15]
        for state in (0, 0xFFFFFFFF, 0x12345678, *[rng.getrandbits(32) for _ in range(20)]):
            for value in range(256):
                expected = state ^ value
                for _ in range(8):
                    expected = (expected >> 1) ^ (0xEDB88320 if expected & 1 else 0)
                self.assertEqual(optimized(state, value), expected)
        for size in (0, 1, 42, 631, 808, 2048):
            data = bytes(rng.randrange(256) for _ in range(size))
            crc = 0xFFFFFFFF
            for value in data:
                crc = optimized(crc, value)
            self.assertEqual(crc ^ 0xFFFFFFFF, binascii.crc32(data))


if __name__ == '__main__':
    unittest.main()
