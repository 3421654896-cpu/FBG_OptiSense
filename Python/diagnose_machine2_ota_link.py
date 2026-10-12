"""Run a machine-profile OTA while comparing target/control ICMP health.

The report intentionally excludes the protected OTA credential.  One ICMP
echo per second is used so the RF/link observation does not create additional
TCP seeds on the BW20 server during recovery.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import secrets
import subprocess
import time

from fbg_ota_protocol import OtaClient, OtaProgress, load_package
from ota_machine_profiles import (
    OtaMachineProfileStore,
    protect_secret,
    unprotect_secret,
)
from ota_debug_recovery import resume_halted_verified_ota
from ota_swd_resume import read_verified_resume


TIME_RE = re.compile(r"(?:time|时间)[=<](\d+)ms", re.IGNORECASE)
NONCE_RE = re.compile(r"[0-9A-F]{16}\Z")
JOURNAL_SCHEMA = "fbg_ota_resume_journal_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resume_journal_nonce(
    path: Path, *, machine_id: str, host: str, package,
) -> tuple[str, bool]:
    """Persist an authenticated OTA nonce across diagnostic process exits.

    BEGIN with the same nonce, image identity and password returns the board's
    authoritative staged offset while its session is alive.  If the board has
    rebooted and lost the session, BEGIN safely starts a fresh staging erase.
    The journal never contains the authorization password or plaintext nonce.
    """

    identity = {
        "schema": JOURNAL_SCHEMA,
        "machine_id": machine_id,
        "host": host,
        "firmware_version": f"{package.firmware_version:08X}",
        "image_size": package.image_size,
        "image_crc32": f"{package.image_crc32:08X}",
    }
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if any(payload.get(key) != value for key, value in identity.items()):
            raise RuntimeError("OTA resume journal does not match target/package")
        if payload.get("status") != "active":
            raise RuntimeError("OTA resume journal is not active")
        nonce = unprotect_secret(str(payload.get("nonce_dpapi", ""))).upper()
        if not NONCE_RE.fullmatch(nonce):
            raise RuntimeError("OTA resume journal nonce is invalid")
        return nonce, True

    nonce = secrets.token_hex(8).upper()
    payload = {
        **identity,
        "status": "active",
        "created_utc": utc_now(),
        "nonce_dpapi": protect_secret(nonce),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return nonce, False


def complete_resume_journal(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != JOURNAL_SCHEMA or payload.get("status") != "active":
        raise RuntimeError("OTA resume journal changed unexpectedly")
    payload["status"] = "complete"
    payload["completed_utc"] = utc_now()
    payload.pop("nonce_dpapi", None)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def start_ping(host: str, path: Path) -> tuple[subprocess.Popen, object]:
    handle = path.open("wb")
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        ["ping.exe", "-t", "-w", "1000", host],
        stdout=handle,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
    )
    return process, handle


def stop_ping(process: subprocess.Popen, handle: object) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3.0)
    handle.close()


def summarize_ping(path: Path) -> dict:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        # ping.exe writes in the active Windows console code page.  Chinese
        # Windows normally uses CP936/GBK rather than UTF-8.
        text = raw.decode("cp936", errors="replace")
    times = [int(value) for value in TIME_RE.findall(text)]
    lowered = text.lower()
    timeouts = lowered.count("request timed out") + text.count("请求超时")
    total = len(times) + timeouts
    return {
        "samples": total,
        "replies": len(times),
        "timeouts": timeouts,
        "loss_percent": round(100.0 * timeouts / total, 3) if total else None,
        "latency_ms_min": min(times) if times else None,
        "latency_ms_max": max(times) if times else None,
        "latency_ms_mean": round(sum(times) / len(times), 3) if times else None,
        "raw_log": str(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine-id", default="machine_2")
    parser.add_argument("--profile-path", type=Path,
                        help="Optional existing encrypted machine-profile file")
    parser.add_argument("--host", help="Verified target IP for this run")
    parser.add_argument("--package", type=Path,
                        help="Verified firmware package for this run")
    parser.add_argument("--control-host", default="192.168.3.46")
    parser.add_argument("--chunk-delay-ms", type=float, default=0.0)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--chunk-delay-max-ms", type=float, default=450.0)
    parser.add_argument("--debugger-recovery", action="store_true")
    parser.add_argument("--probe-id", help="Exact debug-probe ID for guarded reboot recovery")
    parser.add_argument("--resume-installed", type=Path,
                        help="Exact installed package for SWD-verified staged resume")
    parser.add_argument("--resume-map", type=Path,
                        help="Map for SWD-verified staged resume")
    parser.add_argument("--resume-journal", type=Path,
                        help="Windows-protected LAN-only retry journal for this exact machine/package")
    parser.add_argument("--retries", type=int, default=48)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    store = (OtaMachineProfileStore(path=args.profile_path)
             if args.profile_path else OtaMachineProfileStore())
    profile = store.load_profile(args.machine_id)
    if not profile["password"]:
        raise RuntimeError("selected machine has no protected OTA password")
    package_path = args.package or Path(profile["package_path"])
    package = load_package(package_path)
    host = args.host or profile["board_ip"]
    if bool(args.resume_installed) != bool(args.resume_map):
        raise ValueError("--resume-installed and --resume-map must be provided together")
    if args.resume_installed and args.resume_journal:
        raise ValueError("choose SWD resume or LAN-only journal, not both")
    resume = None
    if args.resume_installed:
        if not args.probe_id:
            raise ValueError("--probe-id is required for staged resume")
        resume = read_verified_resume(
            probe_id=args.probe_id, package=package,
            installed_package=args.resume_installed, map_path=args.resume_map,
        )
        print(f"verified_staged_resume=1 offset={resume.offset}/{package.image_size}",
              flush=True)
    journal_nonce = None
    journal_reused = False
    if args.resume_journal:
        journal_nonce, journal_reused = resume_journal_nonce(
            args.resume_journal, machine_id=args.machine_id,
            host=host, package=package,
        )
        print(f"protected_resume_journal=1 reused={int(journal_reused)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    target_ping_path = args.output.with_suffix(".target_ping.txt")
    control_ping_path = args.output.with_suffix(".control_ping.txt")
    target_ping, target_handle = start_ping(host, target_ping_path)
    control_ping, control_handle = start_ping(args.control_host, control_ping_path)

    started = time.monotonic()
    report = {
        "schema": "machine_ota_link_diagnostic_v1",
        "started_utc": utc_now(),
        "machine_id": args.machine_id,
        "target_host": host,
        "control_host": args.control_host,
        "package_path": str(package_path),
        "package_version": package.version_text,
        "image_size": package.image_size,
        "image_crc32": f"{package.image_crc32:08X}",
        "password_logged_plaintext": False,
        "verified_staged_resume_bytes": resume.offset if resume else None,
        "protected_resume_journal": str(args.resume_journal) if args.resume_journal else None,
        "resume_journal_reused": journal_reused,
        "ota_settings": {
            "retries": args.retries,
            "recovery_quiet_s": 30.0,
            "retry_owner_timeout_s": 20.0,
            "chunk_delay_ms": args.chunk_delay_ms,
            "adaptive_chunk_pacing": args.adaptive,
            "chunk_delay_max_ms": args.chunk_delay_max_ms,
        "debugger_recovery": args.debugger_recovery,
        "probe_id": args.probe_id,
        },
        "events": [],
        "status": "running",
    }

    client = OtaClient(
        host,
        profile["password"],
        retries=max(1, args.retries),
        recovery_quiet_s=30.0,
        retry_owner_timeout_s=20.0,
        chunk_delay_s=max(0.0, args.chunk_delay_ms) / 1000.0,
        response_timeout_s=8.0 if args.adaptive else 20.0,
        adaptive_chunk_pacing=args.adaptive,
        chunk_delay_max_s=max(
            args.chunk_delay_ms, args.chunk_delay_max_ms
        ) / 1000.0,
        chunk_delay_backoff=1.5,
        chunk_delay_stable_blocks=384,
        post_reboot_recovery=(
            (lambda: resume_halted_verified_ota(package, probe_id=args.probe_id))
            if args.debugger_recovery
            else None
        ),
        post_reboot_recovery_interval_s=10.0,
    )

    def progress(item: OtaProgress) -> None:
        event = {
            "utc": utc_now(),
            "elapsed_s": round(time.monotonic() - started, 3),
            "sent": item.sent,
            "total": item.total,
            "percent": round(100.0 * item.sent / max(item.total, 1), 3),
            "message": item.message,
        }
        report["events"].append(event)
        print(
            f"elapsed={event['elapsed_s']:.3f}s "
            f"progress={event['percent']:.3f}% {item.message}",
            flush=True,
        )

    exit_code = 0
    try:
        client.upload(
            package, progress=progress,
            verified_resume_nonce=resume.nonce if resume else journal_nonce,
        )
        report["status"] = "passed"
        if args.resume_journal:
            complete_resume_journal(args.resume_journal)
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        exit_code = 1
    finally:
        stop_ping(target_ping, target_handle)
        stop_ping(control_ping, control_handle)
        report["completed_utc"] = utc_now()
        report["elapsed_s"] = round(time.monotonic() - started, 3)
        report["source_boot_id"] = (
            f"{client.source_boot_id:08X}"
            if client.source_boot_id is not None
            else None
        )
        report["final_boot_id"] = (
            f"{client.final_boot_id:08X}"
            if client.final_boot_id is not None
            else None
        )
        report["target_ping"] = summarize_ping(target_ping_path)
        report["control_ping"] = summarize_ping(control_ping_path)
        report["recovery_events"] = sum(
            "恢复续传" in event["message"] for event in report["events"]
        )
        report["stale_ack_events"] = sum(
            "迟到 ACK" in event["message"] for event in report["events"]
        )
        report["client_recovery_count"] = client.recovery_count
        report["final_chunk_delay_ms"] = round(
            client.final_chunk_delay_s * 1000.0, 3
        )
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(json.dumps({
        "status": report["status"],
        "elapsed_s": report["elapsed_s"],
        "recovery_events": report["recovery_events"],
        "target_ping": report["target_ping"],
        "control_ping": report["control_ping"],
        "report": str(args.output),
    }, ensure_ascii=False, indent=2), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
