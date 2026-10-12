"""Read whitelisted Wi-Fi state via SWD; never halt, reset, unlock or write RAM.

No credential/configuration buffers are read. The running flash image must match
the supplied ELF before symbols are used. Requires explicit probe serial.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from elftools.elf.elffile import ELFFile
from pyocd.core.helpers import ConnectHelper
from pyocd.core.target import Target

FIELDS = {
    'lanSocketId': (4, True), 'wifiConfigValid': (1, False),
    'wifiState': (1, False), 'rxOverflow': (1, False),
    'uartTxBusy': (1, False), 'wifiConnected': (1, False),
    'retryExponent': (1, False), 'warmWifiStatusSeen': (1, False),
    'warmWifiMatchesConfig': (1, False), 'rxHead': (2, False),
    'rxTail': (2, False), 'wifiLastError': (2, False),
    'lineLength': (2, False), 'bootId': (4, False),
    'stateDeadlineMs': (4, False), 'clockMilliseconds': (4, False),
    'uartErrorPending': (1, False), 'uartLastError': (4, False),
    'lanClientActive': (1, False), 'lanSocketResetPending': (1, False),
    'lanCleanupOverflow': (1, False), 'lanCleanupSocketId': (4, True),
    'lanDeleteKind': (1, False), 'lanDeleteTargetId': (4, True),
    'lanServerDeleteVerifyAttempts': (1, False),
    'lanStartupServerStatus': (1, True),
    'lanListenerCreateVerifyAttempts': (1, False),
    'lanListenerCreateStartup': (1, False), 'lanReadPending': (1, False),
    'lanCreateAttempts': (2, False), 'lanCreateErrors': (2, False),
    'lanCreateLastResult': (1, False), 'lanCreateLastErrorCode': (2, True),
    'lanQueryRows': (1, False), 'lanQuerySeeds': (1, False),
    'lanQueryParents': (1, False),
    'nextLanRetryMs': (4, False), 'lanResumeState': (1, False),
    'responseError': (1, False), 'responseOk': (1, False),
    'lanStartupCleanupAttempts': (1, False),
    'lanStartupStaleChildId': (4, True),
    'lanServerDeleteVerifyStartup': (1, False),
    'connectDeadlineMs': (4, False), 'nextPollMs': (4, False),
    'nextPublishMs': (4, False),
}
# STM32F205 CMSIS register map. No data register (DR), read-clear or write-only
# register is accessed. Status/control reads do not consume received UART data.
REGISTERS = {
    'USART6_SR': 0x40011400, 'USART6_BRR': 0x40011408,
    'USART6_CR1': 0x4001140c, 'USART6_CR2': 0x40011410,
    'USART6_CR3': 0x40011414, 'GPIOC_MODER': 0x40020800,
    'GPIOC_IDR': 0x40020810, 'GPIOC_AFRL': 0x40020820,
    'UART4_SR': 0x40004c00, 'UART4_BRR': 0x40004c08,
    'UART4_CR1': 0x40004c0c, 'UART4_CR2': 0x40004c10,
    'UART4_CR3': 0x40004c14, 'GPIOA_MODER': 0x40020000,
    'GPIOA_IDR': 0x40020010, 'GPIOA_AFRL': 0x40020020,
    'RCC_APB2ENR': 0x40023844, 'DMA2_S1CR': 0x40026428,
    'DMA2_S1NDTR': 0x4002642c, 'NVIC_ISER2': 0xe000e108,
    'NVIC_ISPR2': 0xe000e208,
}

ROUTE_FIELDS = {name: (4, False) for name in (
    'routeDiagnosticRequests', 'routeDiagnosticAccepted', 'routeDiagnosticCompleted',
    'routeDiagnosticLastTag', 'routeDiagnosticRejectMask')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', required=True)
    parser.add_argument('--elf', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--scope', choices=('wifi', 'short-route'), default='wifi')
    args = parser.parse_args()
    fields = FIELDS if args.scope == 'wifi' else ROUTE_FIELDS
    registers = REGISTERS if args.scope == 'wifi' else {}
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 1 <= args.samples <= 180:
        raise ValueError('samples must be 1..180')
    symbols, flash = {}, []
    with args.elf.open('rb') as handle:
        elf = ELFFile(handle)
        for symbol in elf.get_section_by_name('.symtab').iter_symbols():
            if symbol.name in fields:
                size, signed = fields[symbol.name]
                address = symbol['st_value']
                if symbol.name in symbols or symbol['st_size'] != size:
                    raise ValueError('ambiguous or incorrect symbol size: ' + symbol.name)
                if not 0x20000000 <= address < address + size <= 0x20020000:
                    raise ValueError('symbol outside STM32F205 SRAM')
                symbols[symbol.name] = (address, size, signed)
        for segment in elf.iter_segments():
            address, size = segment['p_paddr'], segment['p_filesz']
            if segment['p_type'] == 'PT_LOAD' and size:
                if not 0x08000000 <= address < address + size <= 0x08080000:
                    raise ValueError('file-backed segment outside device flash')
                flash.append((address, segment.data()))
    if set(symbols) != set(fields) or not flash:
        raise ValueError('missing whitelisted ELF symbols or flash segments')
    options = {'target_override': 'cortex_m', 'connect_mode': 'attach',
               'auto_unlock': False, 'resume_on_disconnect': False,
               'no_config': True, 'frequency': 1000000,
               'cache.enable_memory': False, 'cache.enable_register': False,
               'cache.read_code_from_elf': False}
    session = ConnectHelper.session_with_chosen_probe(
        blocking=False, unique_id=args.probe, options=options)
    if session is None or session.probe.unique_id != args.probe:
        raise RuntimeError('exact requested probe unavailable')
    report = {'schema': 'wifi_ram_readonly_v1' if args.scope == 'wifi' else 'short_route_ram_readonly_v1',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'elf_sha256': hashlib.sha256(args.elf.read_bytes()).hexdigest(),
              'snapshots': [], 'flash_verified': False,
              'scope': 'nonhalting RAM scalar reads; no credentials; not atomic snapshots'}
    try:
        with session:
            target = session.target
            if target.get_state() != Target.State.RUNNING:
                raise RuntimeError('CPU not running; will not resume or reset it')
            for address, expected in flash:
                for offset in range(0, len(expected), 1024):
                    want = expected[offset:offset+1024]
                    actual = bytes(target.read_memory_block8(address+offset, len(want)))
                    if actual != want:
                        raise RuntimeError('running flash differs from ELF; refusing symbol reads')
            report['flash_verified'] = True
            # Hash only. Never display or persist credential-bearing bytes.
            protected_hash = hashlib.sha256()
            for offset in range(0, 0x8000, 1024):
                protected_hash.update(bytes(target.read_memory_block8(0x08000000+offset,1024)))
            report['boot_config_sha256'] = protected_hash.hexdigest()
            for _ in range(args.samples):
                if target.get_state() != Target.State.RUNNING:
                    raise RuntimeError('CPU no longer running; no control action taken')
                snapshot = {name: int.from_bytes(
                    bytes(target.read_memory_block8(address, size)), 'little', signed=signed)
                    for name, (address, size, signed) in symbols.items()}
                snapshot['cpu_state'] = str(target.get_state())
                snapshot['registers'] = {name: hex(target.read32(address))
                                         for name, address in registers.items()}
                report['snapshots'].append(snapshot)
                print(json.dumps(snapshot), flush=True)
                time.sleep(1)
    finally:
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
