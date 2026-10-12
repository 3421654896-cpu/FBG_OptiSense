"""Non-halting STM32 identity/flash backup before an explicitly requested flash.

Protected Flash may contain credentials. Keep the output in a *_PRIVATE
directory; print only identity and hashes, never Flash/configuration bytes.
"""

import argparse
import hashlib
import json
from pathlib import Path

from pyocd.core.helpers import ConnectHelper
from pyocd.core.target import Target
from serial.tools import list_ports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', required=True)
    parser.add_argument('--frequency', type=int, default=1_000_000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.name.endswith('_PRIVATE'):
        raise ValueError('Require a new private backup directory')
    args.output.mkdir(parents=True)
    report = dict(probe=args.probe, frequency=args.frequency, backup_complete=False,
                  no_halt_reset_or_flash_writes=True)
    try:
        session = ConnectHelper.session_with_chosen_probe(
            blocking=False, unique_id=args.probe, options={
                'target_override': 'cortex_m', 'connect_mode': 'attach',
                'auto_unlock': False, 'resume_on_disconnect': False,
                'no_config': True, 'frequency': args.frequency,
                'cache.enable_memory': False, 'cache.enable_register': False,
            })
        if session is None or session.probe.unique_id != args.probe:
            raise RuntimeError('Requested CMSIS probe unavailable')
        with session:
            target = session.target
            report['state'] = str(target.get_state())
            if target.get_state() != Target.State.RUNNING:
                raise RuntimeError('CPU is not running; no automatic reset/resume')
            report['device_id'] = hex(target.read32(0xE0042000))
            words = [target.read32(0x1FFF7A10 + 4*i) for i in range(3)]
            serial = f'{(words[0]+words[2]) & 0xFFFFFFFF:08X}{words[1]:08X}'[:12]
            report['usb_serial'] = serial
            report['flash_kib'] = target.read16(0x1FFF7A22)
            if report['flash_kib'] not in (128, 256, 512, 1024):
                raise RuntimeError('Unexpected Flash geometry; do not guess backup bounds')
            report['native_usb_ports'] = [p.device for p in list_ports.comports()
                if p.vid == 0x0483 and p.pid == 0x5740 and p.serial_number == serial]
            length = report['flash_kib'] * 1024
            flash = bytearray()
            for offset in range(0, length, 1024):
                flash.extend(target.read_memory_block8(0x08000000 + offset,
                                                      min(1024, length-offset)))
            (args.output / 'full_flash.bin').write_bytes(flash)
            report['flash_sha256'] = hashlib.sha256(flash).hexdigest()
            report['protected_sha256'] = hashlib.sha256(flash[:0x8000]).hexdigest()
            report['application_sha256'] = hashlib.sha256(flash[0x8000:0x40000]).hexdigest()
            report['application_vectors_valid'] = (
                0x20000000 <= int.from_bytes(flash[0x8000:0x8004], 'little') <= 0x20020000
                and 0x08008000 <= (int.from_bytes(flash[0x8004:0x8008], 'little') & ~1) < 0x08040000)
            report['backup_complete'] = True
            report['state_after'] = str(target.get_state())
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (args.output / 'report.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
