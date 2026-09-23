"""Per-machine OTA connection settings with Windows-user credential protection."""

from __future__ import annotations

import base64
import ctypes
import json
import os
from ctypes import wintypes
from pathlib import Path
from typing import Callable

from runtime_paths import output_path, resource_path


STORE_SCHEMA = "fbg_ota_machine_profiles_v1"
STORE_PATH = (
    output_path("device_profiles", "ota_machine_profiles.json")
)

def _load_machine_registry() -> tuple[tuple[tuple[str, str], ...], str]:
    path = resource_path("machine_registry.json")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "fbg_machine_registry_v1":
            raise ValueError("registry schema")
        machines = payload.get("machines", {})
        options = tuple(
            (str(machine_id), str(config["machine_label"]))
            for machine_id, config in machines.items()
            if str(machine_id).startswith("machine_")
            and str(config.get("machine_label", "")).strip()
        )
        default_id = str(payload.get("default_machine_id", ""))
        if not options or default_id not in dict(options):
            raise ValueError("registry default")
        return options, default_id
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return (("machine_1", "一号机"), ("machine_2", "二号机")), "machine_1"


MACHINE_OPTIONS, DEFAULT_MACHINE_ID = _load_machine_registry()
MACHINE_LABELS = dict(MACHINE_OPTIONS)


def _bundled_firmware_package() -> Path | None:
    """Return the newest versioned firmware bundled with this app build."""

    firmware_dir = resource_path("firmware")
    ranked: list[tuple[tuple[int, ...], str, Path]] = []
    if firmware_dir.is_dir():
        for candidate in firmware_dir.glob("JDSU_F205RE_v*.fbgfw"):
            version_text = candidate.stem.rsplit("_v", 1)[-1]
            try:
                version = tuple(int(part) for part in version_text.split("."))
            except ValueError:
                continue
            ranked.append((version, candidate.name.casefold(), candidate))
    return max(ranked)[2] if ranked else None


class OtaCredentialError(RuntimeError):
    """Raised when the local Windows credential cannot be protected or restored."""


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_byte)),
    ]


def _blob_from_bytes(value: bytes) -> tuple[_DataBlob, ctypes.Array]:
    buffer = ctypes.create_string_buffer(value)
    blob = _DataBlob(
        len(value),
        ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte)),
    )
    return blob, buffer


def protect_secret(value: str) -> str:
    """Encrypt a secret for the current Windows user and return base64 text."""

    if not value:
        return ""
    if os.name != "nt":
        raise OtaCredentialError("只有 Windows 才能安全保存 OTA 密码")
    raw = value.encode("utf-8")
    input_blob, input_buffer = _blob_from_bytes(raw)
    output_blob = _DataBlob()
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    ok = crypt32.CryptProtectData(
        ctypes.byref(input_blob),
        "FBG OTA machine credential",
        None,
        None,
        None,
        0x01,  # CRYPTPROTECT_UI_FORBIDDEN
        ctypes.byref(output_blob),
    )
    # Keep the source buffer alive through CryptProtectData.
    del input_buffer
    if not ok:
        raise OtaCredentialError("无法使用 Windows 当前用户保护 OTA 密码")
    try:
        encrypted = ctypes.string_at(output_blob.pbData, output_blob.cbData)
    finally:
        kernel32.LocalFree(output_blob.pbData)
    return base64.b64encode(encrypted).decode("ascii")


def unprotect_secret(value: str) -> str:
    """Decrypt base64 DPAPI text for the current Windows user."""

    if not value:
        return ""
    if os.name != "nt":
        raise OtaCredentialError("只有保存密码的 Windows 用户才能读取 OTA 密码")
    try:
        encrypted = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise OtaCredentialError("OTA 密码记录已损坏") from exc
    input_blob, input_buffer = _blob_from_bytes(encrypted)
    output_blob = _DataBlob()
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    ok = crypt32.CryptUnprotectData(
        ctypes.byref(input_blob),
        None,
        None,
        None,
        None,
        0x01,  # CRYPTPROTECT_UI_FORBIDDEN
        ctypes.byref(output_blob),
    )
    del input_buffer
    if not ok:
        raise OtaCredentialError(
            "OTA 密码无法解密，可能由其他 Windows 用户保存"
        )
    try:
        raw = ctypes.string_at(output_blob.pbData, output_blob.cbData)
    finally:
        kernel32.LocalFree(output_blob.pbData)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OtaCredentialError("OTA 密码记录已损坏") from exc


class OtaMachineProfileStore:
    """Persist the last OTA connection fields independently for each machine."""

    def __init__(
        self,
        path: str | Path = STORE_PATH,
        *,
        protector: Callable[[str], str] = protect_secret,
        unprotector: Callable[[str], str] = unprotect_secret,
    ):
        self.path = Path(path)
        self._protector = protector
        self._unprotector = unprotector
        self._data = self._read()

    def _defaults(self) -> dict:
        bundled_package = _bundled_firmware_package()
        machine_one = {
            "machine_label": MACHINE_LABELS[DEFAULT_MACHINE_ID],
            "board_ip": "192.168.3.46",
            "package_path": str(bundled_package) if bundled_package else "",
            "password_dpapi": "",
        }
        return {
            "schema": STORE_SCHEMA,
            "selected_machine_id": DEFAULT_MACHINE_ID,
            "machines": {DEFAULT_MACHINE_ID: machine_one},
        }

    def _read(self) -> dict:
        if not self.path.exists():
            return self._defaults()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return self._defaults()
        if payload.get("schema") != STORE_SCHEMA:
            return self._defaults()
        if not isinstance(payload.get("machines"), dict):
            payload["machines"] = {}
        selected = str(payload.get("selected_machine_id", ""))
        if selected not in MACHINE_LABELS:
            payload["selected_machine_id"] = DEFAULT_MACHINE_ID
        return payload

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)

    @property
    def selected_machine_id(self) -> str:
        value = str(self._data.get("selected_machine_id", DEFAULT_MACHINE_ID))
        return value if value in MACHINE_LABELS else DEFAULT_MACHINE_ID

    def set_selected_machine(self, machine_id: str) -> None:
        if machine_id not in MACHINE_LABELS:
            raise ValueError(f"不支持的机号：{machine_id}")
        self._data["selected_machine_id"] = machine_id
        self._write()

    def load_profile(self, machine_id: str) -> dict:
        if machine_id not in MACHINE_LABELS:
            raise ValueError(f"不支持的机号：{machine_id}")
        stored = self._data.get("machines", {}).get(machine_id, {})
        encrypted = str(stored.get("password_dpapi", ""))
        password = self._unprotector(encrypted) if encrypted else ""
        package_path = str(stored.get("package_path", ""))
        if not Path(package_path).is_file():
            bundled = _bundled_firmware_package()
            if bundled is not None:
                package_path = str(bundled)
        return {
            "machine_id": machine_id,
            "machine_label": MACHINE_LABELS[machine_id],
            "board_ip": str(stored.get("board_ip", "")),
            "package_path": package_path,
            "password": password,
        }

    def save_profile(
        self,
        machine_id: str,
        *,
        board_ip: str,
        password: str,
        package_path: str,
    ) -> None:
        if machine_id not in MACHINE_LABELS:
            raise ValueError(f"不支持的机号：{machine_id}")
        record = {
            "machine_label": MACHINE_LABELS[machine_id],
            "board_ip": str(board_ip).strip(),
            "package_path": str(package_path).strip(),
            "password_dpapi": self._protector(password) if password else "",
        }
        self._data.setdefault("machines", {})[machine_id] = record
        self._data["selected_machine_id"] = machine_id
        self._write()


__all__ = [
    "DEFAULT_MACHINE_ID",
    "MACHINE_LABELS",
    "MACHINE_OPTIONS",
    "OtaCredentialError",
    "OtaMachineProfileStore",
    "_bundled_firmware_package",
    "protect_secret",
    "unprotect_secret",
]
