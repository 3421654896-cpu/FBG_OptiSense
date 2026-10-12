"""Compile the real polling driver and fault-inject STM32 MMIO under Unicorn.

Optional validation dependency only; never added to customer requirements.
No test connects a serial port, probe, or optical device.
"""
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

from test_firmware_feature_profile import COMPILER, INCLUDES, FIRMWARE, preprocess
from test_firmware_scan_scheduler import _function

try:
    import unicorn as u
    from unicorn.arm_const import UC_ARM_REG_R0, UC_ARM_REG_SP, UC_ARM_REG_LR
    from elftools.elf.elffile import ELFFile
except ImportError:
    u = None

I2C = 0x40005400
DWT = 0xE0001000
DEMCR = 0xE000EDFC


class PollingContractTests(unittest.TestCase):
    def test_beta_only_opt_in_and_calibrated_order_unchanged(self):
        source = preprocess('ms5614t.c')
        self.assertNotIn('PI11210_WriteRegisterPolling', source)
        beta = preprocess('ms5614t.c', (('BETA_SINGLE_ADC', 1), ('BETA_DAC_POLLING', 1)))
        body = _function(beta, 'PI11210_WriteRegisterRaw')
        for token in ('pi11210Status.initialized', 'pi11210Status.online',
                      'pi11210Status.partValid', 'HAL_I2C_Mem_Write',
                      'PI11210_WriteRegisterPolling', 'PI11210_ForceShutter();'):
            self.assertIn(token, body)
        error = body.index('PI11210_CountI2CError();')
        self.assertLess(body.index('PI11210_ForceShutter();', error),
                        body.index('PI11210_RecoverI2CBus()', error))
        self.assertIn('if(attempt != 0U)', body)
        batch = _function(beta, 'PI11210_ApplyCalibrationCodes')
        self.assertIn('channelIndex < 5U', batch)
        self.assertIn('tableDACChannels[channelIndex]', batch)
        self.assertNotIn('PI11210_ApplyTemporaryCodesFast', batch)

    def test_polling_does_not_change_clock_or_reset_cycle_counter(self):
        source = (FIRMWARE / 'Core/Inc/pi11210_register_polling.h').read_text()
        self.assertNotIn('CYCCNT =', source)
        self.assertNotIn('CCR =', source)
        self.assertNotIn('TRISE =', source)
        self.assertNotIn('__disable_irq', source)
        self.assertIn('bus->Init.ClockSpeed != 400000U', source)
        self.assertLess(source.index('I2C_SR1_BTF, started'),
                        source.index('SET_BIT(bus->Instance->CR1, I2C_CR1_STOP);'))


class Mmio:
    """Event-driven ACK/flag device; timeout faults do not manufacture ACKs."""
    def __init__(self, cpu, fault=None, cycle_start=0):
        self.cpu, self.fault = cpu, fault
        self.sr1 = 0
        self.sr2 = 2 if fault == 'busy' else 0
        self.cr1 = 1
        self.cr2 = 0
        self.cycles = cycle_start
        self.stage = 'idle'
        self.bytes = []
        self.events = []
        self.addr_sr1_read = False
        self.stopped = False
        self.cpu.mem_write(DWT, struct.pack('<I', 1))
        self.cpu.mem_write(DEMCR, struct.pack('<I', 1 << 24))
        cpu.hook_add(u.UC_HOOK_MEM_READ, self.read)
        cpu.hook_add(u.UC_HOOK_MEM_WRITE, self.write)

    def flag(self, stage, flag):
        self.stage = stage
        self.sr1 = 0 if self.fault == stage + '_timeout' else flag
        if self.fault == stage + '_nack': self.sr1 |= 0x400
        if stage == 'sb' and self.fault in ('berr', 'arlo', 'ovr'):
            self.sr1 |= {'berr': 0x100, 'arlo': 0x200, 'ovr': 0x800}[self.fault]
            if self.fault == 'arlo': self.sr2 = 2  # another master owns BUSY

    def read(self, cpu, access, address, size, value, data):
        if address == DWT + 4:
            self.cycles = (self.cycles + 10000) & 0xFFFFFFFF
            cpu.mem_write(address, struct.pack('<I', self.cycles))
        elif address == I2C:
            cpu.mem_write(address, struct.pack('<I', self.cr1))
        elif address == I2C + 4:
            cpu.mem_write(address, struct.pack('<I', self.cr2))
        elif address == I2C + 0x14:
            cpu.mem_write(address, struct.pack('<I', self.sr1))
            if self.stage == 'addr': self.addr_sr1_read = True
        elif address == I2C + 0x18:
            cpu.mem_write(address, struct.pack('<I', self.sr2))
            if self.stage == 'addr' and self.addr_sr1_read and self.sr1 & 2:
                self.events.append('clear_addr')
                self.flag('reg_ready', 0x80)

    def write(self, cpu, access, address, size, value, data):
        if address == I2C:
            self.cr1 = value & ~0x100
            if value & 0x100:
                self.events.append('start')
                self.sr2 = 3
                self.flag('sb', 1)
            if value & 0x200:
                self.events.append('stop')
                self.stopped = True
                if self.fault != 'stop_timeout':
                    self.cr1 &= ~0x200
                    self.sr2 = 2 if self.fault == 'release_timeout' else 0
                    self.sr1 = 0
        elif address == I2C + 0x10:
            self.bytes.append(value & 255)
            self.events.append('byte')
            count = (len(self.bytes) - 1) % 4 + 1
            stages = {1: ('addr', 2), 2: ('msb_ready', 0x80),
                      3: ('lsb_ready', 0x80), 4: ('btf', 0x84)}
            self.flag(*stages[count])
        elif address == I2C + 0x14:
            self.sr1 = value


@unittest.skipUnless(u and Path(COMPILER).is_file(), 'ARM compiler and optional unicorn required')
class CompiledPollingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory(prefix='pi11210_compiled_')
        cls.addClassCleanup(cls.folder.cleanup)
        folder = Path(cls.folder.name)
        obj, elf = folder / 'fixture.o', folder / 'fixture.elf'
        cmd = [COMPILER, '--target=arm-arm-none-eabi', '-mcpu=cortex-m3', '-mthumb',
               '-O1', '-g', '-ffreestanding', '-DUSE_HAL_DRIVER', '-DSTM32F205xx']
        for path in INCLUDES: cmd += ['-I', str(FIRMWARE / path)]
        cmd += ['-c', str(Path(__file__).parent / 'firmware_tests/pi11210_polling_harness.c'), '-o', str(obj)]
        subprocess.run(cmd, check=True, capture_output=True)
        link = str(Path(COMPILER).with_name('armlink.exe'))
        subprocess.run([link, str(obj), '--entry=fixture_run', '--ro-base=0x10000',
                        '--rw-base=0x20000000', '--no_scanlib', '--output', str(elf)],
                       check=True, capture_output=True)
        with elf.open('rb') as stream:
            binary = ELFFile(stream)
            # Keil scatter loading stores RW initializers in the RO load
            # segment. This fixture has no startup scatter-copy routine, so
            # load allocated sections at their actual execution addresses.
            cls.segments = [(s['sh_addr'], s.data()) for s in binary.iter_sections()
                            if s['sh_flags'] & 2 and s['sh_type'] != 'SHT_NOBITS']
            symbols = binary.get_section_by_name('.symtab')
            cls.symbols = {s.name: s['st_value'] for s in symbols.iter_symbols()}

    def run_driver(self, fault=None, guard=0, cycle_start=0):
        cpu = u.Uc(u.UC_ARCH_ARM, u.UC_MODE_THUMB | u.UC_MODE_MCLASS)
        for address, length in ((0x10000, 0x30000), (0x20000000, 0x20000),
                                (0x40005000, 0x1000), (DWT, 0x1000), (0xE000E000, 0x1000)):
            cpu.mem_map(address, length)
        for address, payload in self.segments: cpu.mem_write(address, payload)
        mmio = Mmio(cpu, fault, cycle_start)
        if fault == 'dma': mmio.cr2 = 0x800
        if fault == 'irq': mmio.cr2 = 0x700
        if fault == 'dwt_off': cpu.mem_write(DWT, b'\0' * 4)
        if fault == 'trace_off': cpu.mem_write(DEMCR, b'\0' * 4)
        cpu.reg_write(UC_ARM_REG_SP, 0x2001F000)
        cpu.reg_write(UC_ARM_REG_R0, guard)
        cpu.reg_write(UC_ARM_REG_LR, 0x30001)
        cpu.emu_start(self.symbols['fixture_run'] | 1, 0x30000, count=200000)
        result = struct.unpack('<6I', cpu.mem_read(self.symbols['fixture_result'], 24))
        return result, mmio

    def test_success_exact_address_register_msb_lsb_and_stop(self):
        for start in (0, 0xFFFF0000):
            with self.subTest(cycle_start=start):
                result, bus = self.run_driver(cycle_start=start)
                self.assertEqual(result, (0, 0, 0x20, 0, 0, 0))
                self.assertEqual(bus.bytes, [0xB0, 0x2A, 0x12, 0x34])
                self.assertEqual(bus.events, ['start', 'byte', 'clear_addr', 'byte', 'byte', 'byte', 'stop'])

    def test_nack_and_bus_errors_are_never_success(self):
        for fault, code in (('addr_nack', 4), ('msb_ready_nack', 4),
                            ('lsb_ready_nack', 4), ('btf_nack', 4),
                            ('berr', 1), ('arlo', 2), ('ovr', 8)):
            with self.subTest(fault=fault):
                result, bus = self.run_driver(fault)
                self.assertEqual(result[:2], (1, code))
                self.assertEqual(result[2:5], (0x20, 0, 0))
                self.assertEqual('stop' in bus.events, fault != 'arlo')

    def test_bounded_timeout_in_every_phase_and_wrap(self):
        for fault in ('busy', 'sb_timeout', 'addr_timeout', 'reg_ready_timeout',
                      'msb_ready_timeout', 'lsb_ready_timeout', 'btf_timeout',
                      'stop_timeout', 'release_timeout'):
            for start in (0, 0xFFFF0000):
                with self.subTest(fault=fault, start=start):
                    result, bus = self.run_driver(fault, cycle_start=start)
                    self.assertEqual(result[:2], (3, 0x20))
                    self.assertEqual(result[2:5], (0x20, 0, 0))
                    if fault == 'busy': self.assertEqual(bus.events, [])

    def test_non_owned_or_unsupported_bus_is_not_touched(self):
        for fault, guard in ((None, 1), (None, 2), (None, 3), (None, 4),
                             ('dma', 0), ('irq', 0), ('dwt_off', 0), ('trace_off', 0)):
            with self.subTest(fault=fault, guard=guard):
                result, bus = self.run_driver(fault, guard)
                self.assertEqual(result[0], 2)
                self.assertEqual(bus.events, [])

    def test_deferred_stop_never_overlaps_the_next_start(self):
        result, bus = self.run_driver('stop_timeout', guard=5)
        self.assertEqual(result[:2], (0, 0))  # All bytes were ACKed before STOP.
        self.assertTrue(bus.cr1 & 0x200)
        result, bus = self.run_driver('stop_timeout', guard=6)
        self.assertEqual(result[:2], (3, 0x20))
        self.assertEqual(bus.events.count('start'), 1)
        self.assertEqual(bus.bytes, [0xB0, 0x2A, 0x12, 0x34])
        result, bus = self.run_driver(guard=7)
        self.assertEqual(result, (0, 0, 0x20, 0, 0, 0))
        self.assertEqual(bus.bytes, [0xB0, 0x2A, 0x12, 0x34, 0xB0, 0x2E, 0x12, 0x34])
        self.assertEqual(bus.events.count('start'), 2)
        self.assertEqual(bus.events.count('stop'), 2)


if __name__ == '__main__':
    unittest.main()
