"""Check USB/SWD status without reset, halt, flash writes, DAC writes or motion."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from pyocd.core.helpers import ConnectHelper
from pyocd.core.target import Target
import serial

from fbg_ota_protocol import load_package
from wifi_status_query import build_status_query, find_status_reply


def check(probe, package_path):
    package = load_package(package_path)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  probe=probe, read_only=True, no_motion=True,
                  no_reset_or_halt=True, expected_package=package.version_text)
    options = dict(target_override='cortex_m', connect_mode='attach',
                   auto_unlock=False, resume_on_disconnect=False, no_config=True,
                   frequency=1000000)
    options.update({'cache.enable_memory': False, 'cache.enable_register': False})
    session = ConnectHelper.session_with_chosen_probe(
        blocking=False, unique_id=probe, options=options)
    if session is None or session.probe.unique_id != probe:
        raise RuntimeError('Exact WCH probe unavailable')
    with session:
        target = session.target
        if target.get_state() != Target.State.RUNNING:
            raise RuntimeError('CPU not running; will not resume it')
        report['cpuid'] = hex(target.read32(0xe000ed00))
        report['dbgmcu_idcode'] = hex(target.read32(0xe0042000))
        actual_hash = hashlib.sha256()
        matched = True
        for offset in range(0, len(package.image), 1024):
            expected = package.image[offset:offset+1024]
            actual = bytes(target.read_memory_block8(
                package.link_address+offset, len(expected)))
            matched = matched and actual == expected
            actual_hash.update(actual)
        report['application_bytes_verified'] = len(package.image)
        report['application_matches_package'] = matched
        report['application_sha256'] = actual_hash.hexdigest()
        # Preserve only a digest of boot/config; never record credentials.
        protected_hash = hashlib.sha256()
        for offset in range(0, 0x8000, 1024):
            protected_hash.update(bytes(target.read_memory_block8(0x08000000+offset, 1024)))
        report['boot_config_sha256'] = protected_hash.hexdigest()
        report['cpu_state_after_read'] = str(target.get_state())
    print(json.dumps(report), flush=True)
    # This status request does not change mode, gain, DAC or shutter state.
    with serial.Serial('COM6', 2_000_000, timeout=.05, write_timeout=1) as device:
        device.dtr = True
        time.sleep(.15)
        report['usb_status_replies'] = []
        for _ in range(3):
            device.reset_input_buffer()
            device.write(build_status_query())
            device.flush()
            deadline = time.monotonic()+2
            buffer = bytearray()
            reply = None
            while time.monotonic() < deadline:
                buffer.extend(device.read(4096))
                reply = find_status_reply(buffer)
                if reply is not None:
                    break
            if reply is None:
                raise TimeoutError('No fresh COM6 status reply')
            report['usb_status_replies'].append(reply)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', required=True)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = check(args.probe, args.package)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
