"""Approved local USB packages; never pick arbitrary candidate build files."""
import hashlib
import json

from fbg_ota_protocol import OtaError, load_package
from runtime_paths import resource_path


def resolve_usb_package(machine_id, serial_number, fallback=None):
    path = resource_path('usb_firmware_catalog.json')
    try:
        catalog = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        catalog = {'schema': 'fbg_usb_firmware_catalog_v1', 'machines': {}}
    if catalog.get('schema') != 'fbg_usb_firmware_catalog_v1':
        raise OtaError('USB 固件清单格式无效')
    item = catalog.get('machines', {}).get(machine_id)
    approved = None
    if item:
        if serial_number != item['usb_serial']:
            raise OtaError('实际 USB 设备与所选机号的已验证设备不匹配，拒绝更新')
        candidates = [resource_path('firmware', 'usb', item['filename']),
                      resource_path('..', 'outputs', 'ota', item['filename'])]
        package_path = next((p for p in candidates if p.is_file()), None)
        if package_path is None:
            raise OtaError('已批准的 USB 固件包尚未随客户端提供')
        digest = hashlib.sha256(package_path.read_bytes()).hexdigest()
        if digest != item['sha256']:
            raise OtaError('USB 固件包 SHA-256 校验失败，拒绝更新')
        approved = load_package(package_path)
    if fallback is not None and fallback.firmware_version >= 0x00010069:
        if approved is None or fallback.firmware_version > approved.firmware_version:
            approved = fallback
    if approved is None:
        raise OtaError('暂无适用的 USB 更新包；请联系开发者提供 v1.0.105 或更新的固件')
    return approved
