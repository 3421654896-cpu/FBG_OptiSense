"""Read-only proof of feature exclusion and STM32 capacity after a release build.

Prints JSON for the build record; never discovers or contacts any hardware.
The package's CRC, vectors and target are checked by the normal USB updater's
package reader, rather than relying on the AXF's debug-inclusive file size.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from fbg_ota_protocol import load_package


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def memory_report(image, map_path, callgraph_path):
    mapping = map_path.read_text(encoding="utf-8")
    callgraph = callgraph_path.read_text(encoding="utf-8")
    ram = int(re.search(r"Total RW\s+Size.*?([0-9]+)\s+\(", mapping)[1])
    image_bytes = image.stat().st_size
    stack = int(re.search(r"Maximum Stack Usage\s*=\s*([0-9]+)", callgraph)[1])
    return {
        "image_bytes": image_bytes,
        "app_capacity_bytes": 224 * 1024,
        "app_percent": image_bytes / (224 * 1024) * 100,
        "app_free_bytes": 224 * 1024 - image_bytes,
        "static_ram_bytes": ram,
        "ram_capacity_bytes": 128 * 1024,
        "static_ram_percent": ram / (128 * 1024) * 100,
        "ram_unallocated_bytes": 128 * 1024 - ram,
        "reserved_stack_bytes": 4096,
        "reserved_heap_bytes": 1024,
        "known_callgraph_stack_bytes": stack,
        "runtime_stack_peak_measured": False,
        "callgraph_has_unknown_stack_paths": "Unknown(" in callgraph,
    }, mapping, callgraph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--before-image", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    backup = args.backup.resolve()
    before, _, _ = memory_report(args.before_image,
        backup / "JDSU/MDK-ARM/JDSU/JDSU.map",
        backup / "JDSU/MDK-ARM/JDSU/JDSU.htm")
    after, mapping, callgraph = memory_report(args.image,
        root / "JDSU/MDK-ARM/JDSU/JDSU.map",
        root / "JDSU/MDK-ARM/JDSU/JDSU.htm")
    package = load_package(args.package)
    image_data = args.image.read_bytes()
    symbols = set(re.findall(r"<P><STRONG><a name=.*?</a>(.*?)</STRONG>", callgraph))
    excluded = ("write_ms5614t_table", "StartNextPublish", "ParseRemoteCommand",
                "SignWanOtaResponse", "getFilterDiff", "getWaveRange",
                "sampleVoltageStable", "sampleVoltagePrecision")
    retained = ("OtaUpdate_ProcessUsb", "OtaUpdate_HandleCommand",
                "write_ms5614t_extra", "write_ms5614t_manual",
                "ThermalControl_Process", "PI11210_EmergencyShutter")
    checks = {"excluded_" + name: name not in symbols for name in excluded}
    checks.update({"retained_" + name: name in symbols for name in retained})
    for marker in (b"AT+MQTT=", b"AT+MQTTPUBRAW", b"AT+MQTTSUB", b"+EVENT:MQTT_SUB"):
        checks["excluded_string_" + marker.decode("ascii")] = marker not in image_data
    checks.update({
        "retained_dense_state": ".bss.runDenseScanBlock.request" in mapping,
        "retained_finger_state": ".bss.runFastFlankStream.candidateRoute" in mapping,
        "removed_legacy_flags_storage": ".bss.unstableFlags" not in mapping,
        "thermal_source_unchanged": sha256(root / "JDSU/Core/Src/temperature.c")
            == sha256(backup / "JDSU/Core/Src/temperature.c"),
        "package_payload_matches_image": package.image == image_data,
        "app_space_available": after["app_free_bytes"] > 0,
    })
    # A reproducible directory digest allows later backup-integrity checks.
    # Algorithm: sorted POSIX relative path UTF-8, NUL, file SHA-256 ASCII, LF.
    digest = hashlib.sha256()
    files = sorted((p.relative_to(backup).as_posix(), p)
                   for p in backup.rglob("*") if p.is_file())
    for relative, path in files:
        digest.update(relative.encode("utf-8") + b"\0"
                      + sha256(path).encode("ascii") + b"\n")
    report = {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "compile_and_package_verified_not_hardware_accepted",
        "backup": {"path": str(backup), "files": len(files),
                   "bytes": sum(p.stat().st_size for _, p in files),
                   "directory_digest_sha256": digest.hexdigest()},
        "before": before, "after": after,
        "flash_saved_bytes": before["image_bytes"] - after["image_bytes"],
        "static_ram_saved_bytes": before["static_ram_bytes"] - after["static_ram_bytes"],
        "package": {"path": str(args.package.resolve()),
                    "version": package.version_text,
                    "bytes": args.package.stat().st_size,
                    "crc32": f"{package.image_crc32:08X}",
                    "sha256": sha256(args.package),
                    "image_sha256": sha256(args.image)},
        "checks": checks, "all_checks_pass": all(checks.values()),
        "hardware_flashed": False, "customer_catalog_changed": False,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
