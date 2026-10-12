"""Guarded AT-only baud sweep on an exact, laser-disabled diagnostic image."""
import argparse
import json
import time
from pathlib import Path

import serial
from elftools.elf.elffile import ELFFile
from pyocd.core.helpers import ConnectHelper
from pyocd.core.target import Target

from fbg_ota_protocol import load_package
from guarded_application_upgrade import read
from validate_wan_ota_live import require_usb_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', required=True)
    parser.add_argument('--port', required=True)
    parser.add_argument('--device-serial', required=True)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--elf', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    require_usb_identity(args.port, args.device_serial)
    symbols = {}
    with args.elf.open('rb') as source:
        elf = ELFFile(source)
        for symbol in elf.get_section_by_name('.symtab').iter_symbols():
            if symbol.name in ('bw20BridgeBaud', 'bw20BridgeActiveBaud'):
                address = symbol['st_value']
                if symbol['st_size'] != 4 or not 0x20000000 <= address < 0x20020000:
                    raise ValueError('Unsafe diagnostic symbol')
                symbols[symbol.name] = address
    if len(symbols) != 2:
        raise ValueError('Missing bridge diagnostic symbols')
    package = load_package(args.package)
    session = ConnectHelper.session_with_chosen_probe(blocking=False, unique_id=args.probe,
        options={'target_override': 'cortex_m', 'connect_mode': 'attach',
                 'auto_unlock': False, 'resume_on_disconnect': False, 'no_config': True,
                 'frequency': 1000000, 'cache.enable_memory': False,
                 'cache.enable_register': False})
    if session is None or session.probe.unique_id != args.probe:
        raise RuntimeError('Requested probe missing')
    report = {'samples': [], 'flash_written': False, 'at_only': True}
    try:
        with session:
            target = session.target
            if target.get_state() != Target.State.RUNNING:
                raise RuntimeError('Bridge CPU is not running')
            uid = [target.read32(0x1FFF7A10+4*i) for i in range(3)]
            actual_serial = f'{(uid[0]+uid[2]) & 0xFFFFFFFF:08X}{uid[1]:08X}'[:12]
            if actual_serial != args.device_serial:
                raise RuntimeError('SWD UID does not match expected native USB serial')
            if read(target, package.link_address, len(package.image)) != package.image:
                raise RuntimeError('Installed bridge differs from approved package')
            report['exact_bridge_verified'] = True
            with serial.Serial(args.port, 115200, timeout=.05, write_timeout=2) as port:
                port.dtr = True
                def set_baud(baud):
                    target.write32(symbols['bw20BridgeBaud'], baud)
                    deadline = time.monotonic()+2
                    while time.monotonic() < deadline:
                        if target.read32(symbols['bw20BridgeActiveBaud']) == baud:
                            return
                        time.sleep(.02)
                    raise TimeoutError('Bridge baud change not confirmed')
                try:
                    for baud in (115200, 9600, 19200, 38400, 57600, 230400,
                                 460800, 921600, 1000000, 2000000):
                        set_baud(baud)
                        time.sleep(.15)
                        port.reset_input_buffer()
                        port.write(b'\r\nAT\r\n')
                        port.flush()
                        buffer = bytearray()
                        deadline = time.monotonic()+1.2
                        while time.monotonic() < deadline:
                            buffer.extend(port.read(4096))
                        sample = {'baud': baud, 'rx_bytes': len(buffer),
                                  'at_ok': b'OK' in buffer,
                                  'usart6_sr': f'{target.read32(0x40011400):08X}',
                                  'pc7_rx_high': bool(target.read32(0x40020810)&128)}
                        report['samples'].append(sample)
                        print(json.dumps(sample), flush=True)
                        if sample['at_ok']:
                            port.reset_input_buffer()
                            port.write(b'AT+GMR\r\n')
                            port.flush()
                            time.sleep(1)
                            report['module_version_reply'] = port.read_all().decode('ascii', 'replace')
                            break
                    if not any(s['at_ok'] for s in report['samples']):
                        # Weak RX pull-up only; never drives the BW20 line high.
                        set_baud(115200)
                        old_pull = target.read32(0x4002080C)
                        try:
                            target.write32(0x4002080C, (old_pull & ~(3 << 14)) | (1 << 14))
                            time.sleep(.1)
                            report['pc7_high_with_weak_pullup'] = bool(target.read32(0x40020810)&128)
                        finally:
                            target.write32(0x4002080C, old_pull)
                            report['pc7_pull_restored'] = target.read32(0x4002080C) == old_pull
                finally:
                    set_baud(115200)
                    report['baud_restored'] = True
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
