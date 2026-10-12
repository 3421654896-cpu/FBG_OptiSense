"""Operator-only WAN OTA acceptance. Never print broker or Wi-Fi secrets."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import serial
from serial.tools import list_ports

from benchmark_stress_realtime import _close_shutter, _safe_disarm
from fbg_mqtt_ota import MqttOtaClient
from fbg_mqtt_transport import load_remote_config
from fbg_ota_protocol import load_package
from fbg_wifi_setup import ProvisioningSettings, build_provisioning_frame, find_provisioning_ack
from validate_ota_hardware_stability import load_private_wifi_password


def require_usb_identity(port, expected):
    matches = [p for p in list_ports.comports() if p.device.upper() == port.upper()
               and p.vid == 0x0483 and p.pid == 0x5740 and p.serial_number == expected]
    if len(matches) != 1:
        raise RuntimeError("Exact native USB identity is not present")


def backup(args):
    from pyocd.core.helpers import ConnectHelper
    from pyocd.core.target import Target
    from guarded_application_upgrade import read
    require_usb_identity(args.port, args.device_serial)
    if args.backup_dir.exists():
        raise FileExistsError(args.backup_dir)
    package = load_package(args.package)
    session = ConnectHelper.session_with_chosen_probe(blocking=False, unique_id=args.probe,
        options={'target_override': 'stm32f205re', 'connect_mode': 'attach',
                 'auto_unlock': False, 'resume_on_disconnect': False, 'no_config': True,
                 'frequency': 1000000, 'cache.enable_memory': False,
                 'cache.enable_register': False})
    if session is None or session.probe.unique_id != args.probe:
        raise RuntimeError("Requested probe missing")
    with session:
        target = session.target
        if target.get_state() != Target.State.RUNNING:
            raise RuntimeError("CPU is not running; no reset/resume performed")
        words = [target.read32(0x1FFF7A10 + 4*i) for i in range(3)]
        usb_serial = f'{(words[0]+words[2]) & 0xFFFFFFFF:08X}{words[1]:08X}'[:12]
        if usb_serial != args.device_serial:
            raise RuntimeError("SWD UID does not match native USB serial")
        image = read(target, 0x08000000, 0x40000)
        if image[0x8000:0x8000+len(package.image)] != package.image:
            raise RuntimeError("Running image differs from approved package")
    args.backup_dir.mkdir(parents=True)
    (args.backup_dir / 'flash_08000000_PRIVATE.bin').write_bytes(image)
    (args.backup_dir / 'protected_flash.bin').write_bytes(image[:0x8000])
    (args.backup_dir / 'application_flash.bin').write_bytes(image[0x8000:])
    return dict(usb_serial=usb_serial, version=package.version_text,
                sha256=hashlib.sha256(image).hexdigest(), bytes=len(image), readonly=True)


def provision(args):
    require_usb_identity(args.port, args.device_serial)
    config = load_remote_config(args.config)
    if config.device_id != args.device_id:
        raise RuntimeError("MQTT device ID mismatch")
    key, saved = load_private_wifi_password(args.private_wifi_config)
    if not config.username or not config.password:
        raise RuntimeError("Broker credentials missing")
    settings = ProvisioningSettings(
        ssid=saved[16:49].split(b'\0', 1)[0].decode('utf-8'), wifi_password=key,
        broker_host=config.host, broker_port=config.port, client_id=config.device_id,
        mqtt_username=config.username, mqtt_password=config.password,
        topic_prefix=config.topic_root, device_id=config.device_id, tls=True)
    frame = build_provisioning_frame(settings)
    expected_crc = int.from_bytes(frame[-4:], 'big')
    with serial.Serial(args.port, 2000000, timeout=.05, write_timeout=3) as device:
        device.dtr = True
        if not _safe_disarm(device) or not _close_shutter(device):
            raise RuntimeError("Exact disarm/shutter ACK missing")
        try:
            device.reset_input_buffer()
            device.write(frame)
            device.flush()
            buffer = bytearray()
            deadline = time.monotonic()+10
            while time.monotonic() < deadline:
                buffer.extend(device.read(4096))
                ack = find_provisioning_ack(buffer, expected_crc)
                if ack is not None:
                    status, state, valid, checksum = ack
                    if status != 0 or valid != 1:
                        raise RuntimeError(f"Provisioning rejected: status={status}, state={state}")
                    return dict(device_id=config.device_id, config_crc=f'{checksum:08X}', accepted=True)
            raise TimeoutError("Exact provisioning ACK missing")
        finally:
            if not _safe_disarm(device) or not _close_shutter(device):
                raise RuntimeError("Final disarm/shutter ACK missing")


class MeteredClient(MqttOtaClient):
    def __init__(self, *args):
        super().__init__(*args)
        self.publishes = Counter()
        self.exchanges = Counter()

    def __enter__(self):
        super().__enter__()
        original = self._client.publish
        def publish(topic, payload, *args, **kwargs):
            if topic == self.config.command_topic:
                self.publishes[bytes(payload).split(b'|')[1].decode('ascii')] += 1
            return original(topic, payload, *args, **kwargs)
        self._client.publish = publish
        return self

    def _exchange(self, command, *args, **kwargs):
        self.exchanges[command.split('|')[1]] += 1
        return super()._exchange(command, *args, **kwargs)


def mqtt_test(args, report, save):
    config = load_remote_config(args.config)
    if config.device_id != args.device_id:
        raise RuntimeError("MQTT device ID mismatch")
    key, _ = load_private_wifi_password(args.private_wifi_config)
    started = time.monotonic()
    with MeteredClient(config, key) as client:
        boot, version = client.ping()
        report.update(device_id=config.device_id, old_boot=f'{boot:08X}', old_version=f'{version:08X}',
                      transport='MQTT over TLS; native USB port is not opened during transfer')
        save()
        print(json.dumps(report), flush=True)
        if args.action == 'install':
            package = load_package(args.package)
            report['target_version'] = package.version_text
            report['image_crc32'] = f'{package.image_crc32:08X}'
            last_print = 0
            def progress(sent, total):
                nonlocal last_print
                report.update(sent_bytes=sent, total_bytes=total, elapsed_s=round(time.monotonic()-started, 2),
                              publishes=dict(client.publishes), exchanges=dict(client.exchanges))
                if time.monotonic()-last_print >= 15 or sent == total:
                    save()
                    print(json.dumps({k: report[k] for k in ('sent_bytes','total_bytes','elapsed_s','publishes')}), flush=True)
                    last_print = time.monotonic()
            def phase(value):
                report['phase'] = value
                save()
                print('phase='+value, flush=True)
            client.install(package, progress=progress, phase=phase)
            boot_after, version_after = client.ping()
            report.update(new_boot=f'{boot_after:08X}', new_version=f'{version_after:08X}',
                          verified=boot_after != boot and version_after == package.firmware_version)
            if not report['verified']:
                raise RuntimeError("Final signed boot/version check failed")
        else:
            report['verified'] = True
        report.update(elapsed_s=round(time.monotonic()-started, 2), publishes=dict(client.publishes),
                      exchanges=dict(client.exchanges))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('backup', 'provision', 'ping', 'install'))
    parser.add_argument('--port', default='COM20')
    parser.add_argument('--device-serial', required=True)
    parser.add_argument('--probe')
    parser.add_argument('--device-id', required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--private-wifi-config', type=Path)
    parser.add_argument('--package', type=Path)
    parser.add_argument('--backup-dir', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error('Report already exists')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = dict(action=args.action, created_utc=datetime.now(timezone.utc).isoformat(), verified=False)
    def save():
        args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    try:
        require_usb_identity(args.port, args.device_serial)
        if args.action == 'backup':
            report.update(backup(args))
        elif args.action == 'provision':
            report.update(provision(args))
        else:
            mqtt_test(args, report, save)
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        save()
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
