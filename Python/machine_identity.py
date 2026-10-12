"""Local USB serial-to-machine bindings, independent of firmware approval."""

import json
from pathlib import Path

from runtime_paths import output_path, resource_path

SCHEMA = 'fbg_machine_identity_v1'
NATIVE_USB = (0x0483, 0x5740)


def normalize_serial(value):
    value = str(value or '').strip().upper()
    if len(value) != 12 or any(c not in '0123456789ABCDEF' for c in value):
        raise ValueError('缺少有效的STM32 USB唯一编号，不能自动识别机号')
    return value


def native_ports(ports):
    return [p for p in ports if (getattr(p, 'vid', None), getattr(p, 'pid', None)) == NATIVE_USB]


def identity_for_port(ports, port):
    matches = [p for p in native_ports(ports)
               if str(p.device).upper() == str(port).upper()]
    if len(matches) != 1:
        raise ValueError('所选端口不是唯一的STM32原生USB设备')
    serial = normalize_serial(matches[0].serial_number)
    if sum(str(getattr(p, 'serial_number', '') or '').strip().upper() == serial
           for p in native_ports(ports)) != 1:
        raise ValueError('USB唯一编号重复，拒绝自动选择校准数据')
    return str(matches[0].device), serial


class MachineIdentityStore:
    def __init__(self, path=None, *, registry_path=None):
        self.path = Path(path) if path is not None else output_path('device_profiles', 'machine_identity.json')
        registry = json.loads((Path(registry_path) if registry_path is not None
                               else resource_path('machine_registry.json')).read_text(encoding='utf-8'))
        if registry.get('schema') != 'fbg_machine_registry_v1':
            raise ValueError('机号注册表格式无效')
        self.labels = {str(k): str(v['machine_label']) for k, v in registry['machines'].items()}
        self.known = {}
        for machine_id, config in registry['machines'].items():
            for serial in config.get('usb_serials', []):
                serial = normalize_serial(serial)
                if serial in self.known and self.known[serial] != machine_id:
                    raise ValueError('注册表包含冲突的USB机号绑定')
                self.known[serial] = machine_id
        self.bindings = {}
        if self.path.is_file():
            payload = json.loads(self.path.read_text(encoding='utf-8'))
            if payload.get('schema') != SCHEMA or not isinstance(payload.get('bindings'), dict):
                raise ValueError('USB机号绑定文件格式无效，不能猜测机号')
            for serial, machine_id in payload['bindings'].items():
                serial = normalize_serial(serial)
                self._check_binding(serial, machine_id)
                self.bindings[serial] = machine_id

    def _check_binding(self, serial, machine_id):
        if machine_id not in self.labels:
            raise ValueError('机号尚未登记，不能选择其他机器的数据')
        existing = {**self.known, **self.bindings}
        if serial in existing and existing[serial] != machine_id:
            raise ValueError('该USB设备已属于另一机号，拒绝覆盖绑定')
        if any(s != serial and m == machine_id for s, m in existing.items()):
            raise ValueError('该机号已绑定另一USB设备，请先核对机器身份')

    def resolve(self, serial):
        serial = normalize_serial(serial)
        return self.known.get(serial) or self.bindings.get(serial)

    def bind(self, serial, machine_id):
        serial = normalize_serial(serial)
        self._check_binding(serial, machine_id)
        bindings = {**self.bindings, serial: machine_id}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + '.tmp')
        temporary.write_text(json.dumps(dict(schema=SCHEMA, bindings=bindings),
                                        ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(self.path)
        self.bindings = bindings
