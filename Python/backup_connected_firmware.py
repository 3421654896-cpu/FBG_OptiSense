"""Read-only, identity-locked STM32 flash backup before a firmware test.

Never halts/resets/unlocks the MCU or writes application registers. The SWD UID
must match the observed physical USB descriptor before any flash is read. Wi-Fi
credentials may be present in the binary backup; only their digest is printed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from pyocd.core.helpers import ConnectHelper
from pyocd.core.target import Target
from serial.tools import list_ports

from fbg_ota_protocol import load_package


def read_flash(target, start, size):
    result = bytearray()
    for offset in range(0, size, 4096):
        result.extend(target.read_memory_block8(start + offset,
                      min(4096, size - offset)))
    return bytes(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', required=True)
    parser.add_argument('--device-serial', required=True)
    parser.add_argument('--port', required=True)
    parser.add_argument('--expected-current-package', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Backup directory exists; refusing to overwrite it')
    ports = [p for p in list_ports.comports()
             if p.device.upper() == args.port.upper()
             and p.vid == 0x0483 and p.pid == 0x5740
             and p.serial_number == args.device_serial]
    if len(ports) != 1:
        raise RuntimeError('USB descriptor identity mismatch; no SWD operation')
    package = load_package(args.expected_current_package)
    options = {'target_override': 'cortex_m', 'connect_mode': 'attach',
               'auto_unlock': False, 'resume_on_disconnect': False,
               'no_config': True, 'frequency': 1000000,
               'cache.enable_memory': False, 'cache.enable_register': False}
    session = ConnectHelper.session_with_chosen_probe(blocking=False,
              unique_id=args.probe, options=options)
    if session is None or session.probe.unique_id != args.probe:
        raise RuntimeError('Exact debug probe unavailable')
    with session:
        target = session.target
        if target.get_state() != Target.State.RUNNING:
            raise RuntimeError('CPU is not running; no automatic resume/reset')
        uid = [target.read32(0x1FFF7A10 + 4 * i) for i in range(3)]
        actual_serial = f'{(uid[0] + uid[2]) & 0xFFFFFFFF:08X}{uid[1] >> 16:04X}'
        if actual_serial != args.device_serial:
            raise RuntimeError('SWD target is not the requested USB board; no flash read/write')
        device_id = target.read32(0xE0042000) & 0xFFF
        flash_kib = target.read16(0x1FFF7A22)
        if device_id != 0x411 or flash_kib != 512:
            raise RuntimeError('Expected STM32F205 512 KiB identity was not confirmed')
        vtor = target.read32(0xE000ED08)
        raw = read_flash(target, 0x08000000, 512 * 1024)
        application = raw[0x8000:0x40000]
        image_match = application[:len(package.image)] == package.image
        report = {
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'read_only': True, 'no_reset_halt_or_unlock': True,
            'device_serial': actual_serial, 'port': args.port,
            'probe': args.probe, 'device_id': hex(device_id),
            'flash_kib': flash_kib, 'vtor': hex(vtor),
            'expected_current_package': str(args.expected_current_package.resolve()),
            'expected_current_version': package.version_text,
            'current_image_matches_package': image_match,
            'full_flash_bytes': len(raw),
            'full_flash_sha256': hashlib.sha256(raw).hexdigest(),
            'boot_config_sha256': hashlib.sha256(raw[:0x8000]).hexdigest(),
            'active_application_sha256': hashlib.sha256(application).hexdigest(),
            'cpu_state_after_read': str(target.get_state()),
            'hardware_written': False,
            'backup_contains_private_configuration': True,
        }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'full_flash_512k.bin').write_bytes(raw)
    (args.output_dir / 'active_application_224k.bin').write_bytes(application)
    (args.output_dir / 'report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if hashlib.sha256((args.output_dir / 'full_flash_512k.bin').read_bytes()).hexdigest() \
            != report['full_flash_sha256']:
        raise RuntimeError('Saved backup digest mismatch')
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)


if __name__ == '__main__':
    main()
