"""Repeated real-board OTA acceptance with safe disconnect injection.

The Wi-Fi password is read from an operator-owned private flash backup and is
never printed or written to the report.  Every cycle begins and ends with an
exact USB disarm/shutter acknowledgement, then verifies the bootloader,
application and preserved Wi-Fi record through read-only SWD.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import serial
from pyocd.core.helpers import ConnectHelper
from pyocd.core.target import Target

from app_JDSU import set_soa_shutter_and_verify
from fbg_lan_serial import FbgLanSerial
from fbg_ota_protocol import OtaClient, OtaProgress, load_package
from guarded_application_upgrade import read
from wifi_status_query import query as query_wifi_status


BOOT_START = 0x08000000
CONFIG_START = 0x08004000
METADATA_START = 0x08006000
APPLICATION_START = 0x08008000
CONFIG_PRESERVE_SIZE = 1024


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_private_wifi_password(path: Path) -> tuple[str, bytes]:
    raw = path.read_bytes()
    if len(raw) < CONFIG_PRESERVE_SIZE:
        raise ValueError("Private configuration backup is truncated")
    if int.from_bytes(raw[0:4], "little") != 0x57464331:
        raise ValueError("Stored Wi-Fi configuration magic is invalid")
    size = int.from_bytes(raw[6:8], "little")
    if not 32 <= size <= CONFIG_PRESERVE_SIZE:
        raise ValueError("Stored Wi-Fi configuration size is invalid")
    expected_crc = int.from_bytes(raw[size - 4 : size], "little")
    if zlib.crc32(raw[: size - 4]) & 0xFFFFFFFF != expected_crc:
        raise ValueError("Stored Wi-Fi configuration CRC is invalid")
    # WifiStoredConfig header (12), flags/scheme/port (4), SSID (33), then
    # the fixed 65-byte NUL-terminated Wi-Fi password field.
    password = raw[49:114].split(b"\0", 1)[0].decode("utf-8")
    if not password:
        raise ValueError("Stored Wi-Fi password is empty")
    return password, raw[:CONFIG_PRESERVE_SIZE]


def ensure_shutter(
    port: str = "COM6", attempts: int = 5, *, host: str | None = None
) -> str:
    last_error: Exception | None = None
    for _ in range(max(1, int(attempts))):
        try:
            with serial.Serial(port, 115200, timeout=0.05, write_timeout=1) as device:
                if set_soa_shutter_and_verify(device, timeout_s=5):
                    return "usb"
        except (OSError, serial.SerialException, RuntimeError) as exc:
            last_error = exc
        if host:
            device = FbgLanSerial(
                board_host=host, timeout=0.05, write_timeout=3.0
            )
            try:
                device.open()
                if set_soa_shutter_and_verify(device, timeout_s=5):
                    return "lan"
            except (OSError, RuntimeError) as exc:
                last_error = exc
            finally:
                device.close()
        time.sleep(1.0)
    raise RuntimeError(f"Exact disarm/shutter confirmation failed: {last_error}")


def wait_for_wifi_online(
    port: str = "COM6", timeout_s: float = 45.0, *, host: str | None = None
) -> tuple[dict, list[dict]]:
    """Wait through the valid post-reboot MQTT transition to stable ONLINE."""

    started = time.monotonic()
    observations: list[dict] = []
    last_status: dict | None = None
    while time.monotonic() - started < float(timeout_s):
        if host:
            device = FbgLanSerial(
                board_host=host, timeout=0.05, write_timeout=3.0
            )
            try:
                device.open()
                last_status = query_wifi_status(device, timeout_s=3.0)
            finally:
                device.close()
        else:
            with serial.Serial(
                port, 2_000_000, timeout=0.05, write_timeout=1
            ) as device:
                device.dtr = True
                last_status = query_wifi_status(device, timeout_s=3.0)
        observations.append(
            {
                "elapsed_s": round(time.monotonic() - started, 3),
                "external_state": last_status["external_state_name"],
                "config_valid": last_status["config_valid"],
                "wifi_connected": last_status["wifi_connected"],
                "mqtt_connected": last_status["mqtt_connected"],
                "ota_active": last_status["ota_active"],
                "last_error": last_status["last_error"],
            }
        )
        if (
            last_status["config_valid"]
            and last_status["enabled"]
            and last_status["wifi_connected"]
            and last_status["mqtt_connected"]
            and last_status["external_state_name"] == "online"
            and not last_status["ota_active"]
        ):
            return last_status, observations
        time.sleep(1.0)
    raise RuntimeError(
        f"Wi-Fi/MQTT did not reach stable online state: {last_status}; "
        f"observations={observations}"
    )


class FaultInjectingOtaClient(OtaClient):
    def __init__(
        self,
        *args,
        drop_after_data_command: int | None = None,
        drop_after_end: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.drop_after_data_command = drop_after_data_command
        self.drop_after_end = bool(drop_after_end)
        self.data_commands = 0
        self.fault_injected = False

    def _send(self, sock: socket.socket, line: str) -> None:
        super()._send(sock, line)
        should_drop = False
        if line.startswith("FOTA1|DATA|"):
            self.data_commands += 1
            should_drop = (
                not self.fault_injected
                and self.drop_after_data_command is not None
                and self.data_commands == self.drop_after_data_command
            )
        elif line.startswith("FOTA1|END|"):
            should_drop = not self.fault_injected and self.drop_after_end
        if should_drop:
            self.fault_injected = True
            self._close(sock)

    def _response(self, sock: socket.socket) -> list[str]:
        response = super()._response(sock)
        if len(response) == 5 and response[1] == "READY":
            self.source_boot_id = self._hex_u32(response[4], "升级前启动标识")
        return response

    def wait_for_raw_ready(self, **kwargs) -> int:
        result = super().wait_for_raw_ready(**kwargs)
        self.final_boot_id = result
        return result


def flash_snapshot(
    probe: str,
    package_image: bytes,
    bootloader: bytes,
    expected_config: bytes,
) -> dict[str, object]:
    session = ConnectHelper.session_with_chosen_probe(
        blocking=False,
        unique_id=probe,
        options={
            "target_override": "cortex_m",
            "connect_mode": "attach",
            "auto_unlock": False,
            "resume_on_disconnect": False,
            "no_config": True,
            "frequency": 1_000_000,
            "cache.enable_memory": False,
            "cache.enable_register": False,
        },
    )
    if session is None or session.probe.unique_id != probe:
        raise RuntimeError("Exact debug probe is unavailable")
    with session:
        target = session.target
        if target.get_state() != Target.State.RUNNING:
            raise RuntimeError("CPU is not running after OTA")
        actual_boot = read(target, BOOT_START, len(bootloader))
        actual_config = read(target, CONFIG_START, len(expected_config))
        actual_image = read(target, APPLICATION_START, len(package_image))
        metadata = read(target, METADATA_START, 64)
        result = {
            "cpu_state": str(target.get_state()),
            "bootloader_sha256": sha256(actual_boot),
            "bootloader_matches": actual_boot == bootloader,
            "config_preserve_sha256": sha256(actual_config),
            "config_preserve_matches": actual_config == expected_config,
            "application_sha256": sha256(actual_image),
            "application_matches": actual_image == package_image,
            "metadata_magic": f"0x{int.from_bytes(metadata[0:4], 'little'):08X}",
            "metadata_image_version": f"0x{int.from_bytes(metadata[12:16], 'little'):08X}",
            "metadata_image_size": int.from_bytes(metadata[16:20], "little"),
            "metadata_image_crc32": f"0x{int.from_bytes(metadata[20:24], 'little'):08X}",
            "metadata_boot_token": f"0x{int.from_bytes(metadata[32:36], 'little'):08X}",
        }
    if not (
        result["bootloader_matches"]
        and result["config_preserve_matches"]
        and result["application_matches"]
        and result["metadata_magic"] == "0x00000000"
    ):
        raise RuntimeError(f"Post-OTA flash verification failed: {result}")
    return result


FAULT_PLANS = {
    1: {"kind": "normal"},
    2: {"kind": "normal"},
    3: {"kind": "drop-data", "command": 1},
    4: {"kind": "drop-data", "command": 250},
    5: {"kind": "drop-data", "command": 500},
    6: {"kind": "drop-data", "command": 987},
    7: {"kind": "drop-end"},
    8: {"kind": "normal"},
    9: {"kind": "drop-data", "command": 100},
    10: {"kind": "normal"},
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.3.46")
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--probe", required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--bootloader", type=Path, required=True)
    parser.add_argument("--private-config-backup", type=Path, required=True)
    parser.add_argument("--first-cycle", type=int, default=1)
    parser.add_argument("--last-cycle", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output report already exists")
    if not 1 <= args.first_cycle <= args.last_cycle <= 10:
        parser.error("Cycle range must be within 1..10")

    package = load_package(args.package)
    bootloader = args.bootloader.read_bytes()
    password, expected_config = load_private_wifi_password(
        args.private_config_backup
    )
    report: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "real_BW20_STM32_OTA_with_readonly_SWD_postverification",
        "host": args.host,
        "package_version": package.version_text,
        "package_sha256": sha256(args.package.read_bytes()),
        "image_size": package.image_size,
        "image_crc32": f"0x{package.image_crc32:08X}",
        "bootloader_sha256": sha256(bootloader),
        "config_preserve_sha256_expected": sha256(expected_config),
        "password_logged_or_saved": False,
        "no_cnc_commands": True,
        "cycles": [],
        "complete": False,
    }

    def save() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    save()
    active_cycle: int | None = None
    active_plan: dict[str, object] | None = None
    active_client: FaultInjectingOtaClient | None = None
    try:
        for cycle in range(args.first_cycle, args.last_cycle + 1):
            plan = FAULT_PLANS[cycle]
            active_cycle = cycle
            active_plan = plan
            report["in_progress_cycle"] = {
                "cycle": cycle,
                "fault_plan": plan,
                "stage": "preflight",
            }
            save()
            report["in_progress_cycle"]["preflight_shutter_path"] = (
                ensure_shutter(args.port, host=args.host)
            )
            progress_messages: list[str] = []
            progress_bucket = [-1]
            client = FaultInjectingOtaClient(
                args.host,
                password,
                response_timeout_s=30.0,
                drop_after_data_command=plan.get("command"),
                drop_after_end=plan["kind"] == "drop-end",
            )
            active_client = client

            def progress(item: OtaProgress) -> None:
                percent = int(item.sent * 100 / max(item.total, 1))
                bucket = percent // 25
                if bucket != progress_bucket[0]:
                    print(
                        f"cycle={cycle} progress={percent}% "
                        f"sent={item.sent}/{item.total}",
                        flush=True,
                    )
                    progress_bucket[0] = bucket
                if "连接中断" in item.message:
                    print(
                        f"cycle={cycle} recovery={item.message} "
                        f"data_commands={client.data_commands} "
                        f"fault_injected={client.fault_injected} "
                        f"source_boot_id="
                        f"{f'0x{client.source_boot_id:08X}' if client.source_boot_id is not None else None}",
                        flush=True,
                    )
                if not progress_messages or progress_messages[-1] != item.message:
                    progress_messages.append(item.message)

            started = time.monotonic()
            report["in_progress_cycle"]["stage"] = "upload"
            save()
            client.upload(package, progress=progress)
            elapsed = time.monotonic() - started
            report["in_progress_cycle"] = {
                "cycle": cycle,
                "fault_plan": plan,
                "fault_injected": client.fault_injected,
                "data_commands": client.data_commands,
                "source_boot_id": (
                    f"0x{client.source_boot_id:08X}"
                    if client.source_boot_id is not None
                    else None
                ),
                "final_boot_id": (
                    f"0x{client.final_boot_id:08X}"
                    if client.final_boot_id is not None
                    else None
                ),
                "upload_returned_after_raw_ready": True,
                "elapsed_s": round(elapsed, 3),
                "stage": "post_verification",
            }
            save()
            report["in_progress_cycle"]["post_shutter_path"] = ensure_shutter(
                args.port, host=args.host
            )
            # RAW READY has already proved the LAN data plane.  Use the
            # independent USB status path here so post-probe verification does
            # not create a second BW20 seed inside the release cooldown.
            status, status_observations = wait_for_wifi_online(args.port)
            flash = flash_snapshot(
                args.probe, package.image, bootloader, expected_config
            )
            if (
                client.source_boot_id is None
                or client.final_boot_id is None
                or client.source_boot_id == client.final_boot_id
            ):
                raise RuntimeError(f"Cycle {cycle} boot identity did not change")
            item = {
                "cycle": cycle,
                "fault_plan": plan,
                "fault_injected": client.fault_injected,
                "data_commands": client.data_commands,
                "recovery_messages": sum(
                    "连接中断" in message for message in progress_messages
                ),
                "source_boot_id": f"0x{client.source_boot_id:08X}",
                "final_boot_id": f"0x{client.final_boot_id:08X}",
                "elapsed_s": round(elapsed, 3),
                "post_shutter_verified": True,
                "wifi_status": status,
                "wifi_transition_observations": status_observations,
                "flash": flash,
                "passed": True,
            }
            report["cycles"].append(item)
            report.pop("in_progress_cycle", None)
            save()
            print(
                f"cycle={cycle} PASS fault={plan['kind']} "
                f"boot={client.source_boot_id:08X}->{client.final_boot_id:08X} "
                f"elapsed_s={elapsed:.1f}",
                flush=True,
            )
            active_cycle = None
            active_plan = None
            active_client = None
        report["complete"] = True
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    except BaseException as exc:
        report["failure"] = {
            "cycle": active_cycle,
            "fault_plan": active_plan,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "data_commands": (
                active_client.data_commands if active_client is not None else None
            ),
            "fault_injected": (
                active_client.fault_injected if active_client is not None else None
            ),
            "source_boot_id": (
                f"0x{active_client.source_boot_id:08X}"
                if active_client is not None
                and active_client.source_boot_id is not None
                else None
            ),
            "final_boot_id": (
                f"0x{active_client.final_boot_id:08X}"
                if active_client is not None
                and active_client.final_boot_id is not None
                else None
            ),
        }
        save()
        raise
    finally:
        try:
            report["final_shutter_path"] = ensure_shutter(
                args.port, host=args.host
            )
            report["final_shutter_verified"] = True
        except Exception as exc:
            report["final_shutter_verified"] = False
            report["final_shutter_error"] = f"{type(exc).__name__}: {exc}"
        save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
