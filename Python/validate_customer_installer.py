"""Isolated acceptance of the shipped Windows installer and offline updater.

Never touches hardware or an existing OptiSense installation. The older version
used for replacement is a metadata fixture, not an actual historical binary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import winreg


APP_KEY = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\{7E127A4E-06B6-4B82-A632-CF8C8443CDA8}_is1"
EXPECTED_SETUP_SHA = "7b6dd17442d052800f3c17f79f1da9dca1d62df9f1b62546dfe6bf43b09cf4b6"
EXPECTED_CLIENT_SHA = "b063f2ee24b9c622fdcdc9990dd783130b0f19a7defaa3f8db1bf24013619b26"
EXPECTED_UPDATER_SHA = "c26f1fc52435dbd76cf5bf9142bff385587b8ef916fb9cc6059427fc3c5c508b"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def registrations() -> list[dict]:
    result = []
    for hive_name, hive in (("HKCU", winreg.HKEY_CURRENT_USER), ("HKLM", winreg.HKEY_LOCAL_MACHINE)):
        for view_name, view in (("64", winreg.KEY_WOW64_64KEY), ("32", winreg.KEY_WOW64_32KEY)):
            try:
                key = winreg.OpenKey(hive, APP_KEY, 0, winreg.KEY_READ | view)
            except FileNotFoundError:
                continue
            with key:
                item = {"hive": hive_name, "view": view_name}
                for name in ("DisplayName", "DisplayVersion", "InstallLocation", "UninstallString"):
                    try:
                        item[name] = winreg.QueryValueEx(key, name)[0]
                    except FileNotFoundError:
                        item[name] = None
                result.append(item)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--version", default="1.0.6")
    parser.add_argument("--expected-setup-sha", default=EXPECTED_SETUP_SHA)
    parser.add_argument("--expected-client-sha", default=EXPECTED_CLIENT_SHA)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = (args.report_dir or repo / "build" / "customer_installer_smoke_20261010").resolve()
    if root.parent != (repo / "build").resolve() or not root.name.startswith("customer_installer_smoke_"):
        parser.error("The fresh QA directory must be directly under this repository's build directory")
    target = root / "OptiSense"
    bundle = args.bundle.resolve(strict=True)
    setup = bundle / f"FBG_OptiSense_Studio_Universal_v{args.version}_Setup.exe"
    updater = bundle / "OptiSense_Update.exe"
    if root.exists():
        parser.error("QA directory already exists; preserving previous results")
    if registrations():
        parser.error("An existing installation is registered; refusing to change it")
    default = Path(os.environ["LOCALAPPDATA"]) / "Programs" / "FBG OptiSense Studio"
    if default.exists():
        parser.error("Default installation directory exists; refusing to change it")
    menu = Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "光栅解调工作站"
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders") as key:
        desktop = Path(os.path.expandvars(winreg.QueryValueEx(key, "Desktop")[0]))
    desktop_shortcut = desktop / "光栅解调工作站.lnk"
    if menu.exists() or desktop_shortcut.exists():
        parser.error("An existing OptiSense shortcut exists; refusing to replace it")
    for file, expected in ((setup, args.expected_setup_sha), (updater, EXPECTED_UPDATER_SHA)):
        if digest(file) != expected:
            parser.error(f"Unexpected shipped binary: {file.name}")
    root.mkdir()
    environment = dict(os.environ)
    system = os.environ["SystemRoot"]
    environment["PATH"] = system + "/System32;" + system
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_PLATFORM"):
        environment.pop(name, None)
    environment["FBG_STUDIO_DATA_DIR"] = str(root / "user_data")
    report = {
        "verified": False,
        "hardware_commands": False,
        "external_python_paths_removed": True,
        "test_install_dir": str(target),
        "uninstall_scope": "Only the freshly created isolated QA installation",
        "desktop_shortcut_task_disabled": True,
        "default_installation_absent_before_test": True,
        "older_version_is_simulated_metadata_not_a_historical_binary": True,
    }
    started = time.monotonic()

    def execute(command, label, timeout=120):
        print(label, flush=True)
        step = time.monotonic()
        completed = subprocess.run(command, env=environment, timeout=timeout, capture_output=True)
        (root / f"{label}.stdout.log").write_bytes(completed.stdout)
        (root / f"{label}.stderr.log").write_bytes(completed.stderr)
        report[label] = {"exit_code": completed.returncode, "elapsed_s": round(time.monotonic() - step, 3)}
        assert completed.returncode == 0, f"{label} failed with exit code {completed.returncode}; see QA logs"
        return completed

    def run_client(label):
        execute([str(target / "FBG_OptiSense_Studio.exe"), "--demo", "--demo-seconds", "5"], label, 60)

    def update(label, expected):
        output = root / f"{label}.json"
        execute([str(updater), "--apply-client-update", "--install-dir", str(target), "--result-file", str(output)], label)
        payload = json.loads(output.read_text(encoding="utf-8"))
        report[label]["result"] = payload
        assert not payload["errors"] and payload["results"][0]["status"] == expected, payload
        assert payload["firmware_commands_sent"] is False
        return payload["results"][0]

    try:
        execute([str(setup), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/NOICONS",
                 "/NOCLOSEAPPLICATIONS", '/MERGETASKS=!desktopicon', f"/DIR={target}", f"/LOG={root / 'install.log'}"], "install")
        assert target.resolve(strict=True).parent == root.resolve(strict=True)
        assert not target.is_symlink() and not os.path.isjunction(target)
        assert digest(target / "FBG_OptiSense_Studio.exe") == args.expected_client_sha
        marker = target / "client_release.json"
        assert json.loads(marker.read_text(encoding="utf-8"))["version"] == args.version
        registered = registrations()
        assert registered, "Installer did not register an uninstaller"
        uninstall = target / "unins000.exe"
        assert uninstall.is_file()
        for entry in registered:
            assert Path(entry["InstallLocation"]).resolve() == target.resolve(), entry
            assert entry["UninstallString"].strip('"').casefold() == str(uninstall).casefold(), entry
        report["registration_after_install"] = registered
        report["start_menu_shortcuts_created"] = sorted(p.name for p in menu.glob("*.lnk"))
        assert len(report["start_menu_shortcuts_created"]) == 2
        assert not desktop_shortcut.exists()
        uninstaller_hashes = {p.name: digest(p) for p in target.glob("unins000.*") if p.is_file()}
        assert "unins000.exe" in uninstaller_hashes and "unins000.dat" in uninstaller_hashes
        run_client("installed_client")
        record = root / "user_data" / "outputs" / "customer_record.json"
        record.write_text(json.dumps({"must_preserve": True, "fixture": "installer_acceptance"}), encoding="utf-8")
        record_hash = digest(record)
        update("same_version", "latest")
        # Trigger a normal replacement with explicit, isolated simulated metadata.
        marker.write_text('{"version":"1.0.4"}\n', encoding="utf-8")
        result = update("replacement", "updated")
        backup = Path(result["backup"]).resolve(strict=True)
        assert backup.parent == root.resolve() and backup.name.startswith("OptiSense.backup-")
        assert digest(target / "FBG_OptiSense_Studio.exe") == args.expected_client_sha
        assert all(digest(target / name) == value and digest(backup / name) == value
                   for name, value in uninstaller_hashes.items())
        assert digest(record) == record_hash
        report["real_inno_uninstaller_preserved_after_update"] = True
        report["backup"] = str(backup)
        run_client("updated_client")
        # Recheck exact registered target immediately before uninstalling only QA.
        assert target.resolve().parent == root.resolve() and not os.path.isjunction(target)
        for entry in registrations():
            assert Path(entry["InstallLocation"]).resolve() == target.resolve()
            assert entry["UninstallString"].strip('"').casefold() == str(uninstall).casefold()
        execute([str(uninstall), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", f"/LOG={root / 'uninstall.log'}"], "uninstall")
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and (registrations() or uninstall.exists()):
            time.sleep(0.25)
        assert not registrations(), "Test uninstaller registration remains"
        assert not (target / "FBG_OptiSense_Studio.exe").exists()
        assert not (target / "_internal").exists()
        assert not uninstall.exists()
        assert not menu.exists() and not desktop_shortcut.exists()
        assert digest(record) == record_hash and backup.is_dir()
        assert not default.exists()
        report["registered_installation_removed"] = True
        report["test_start_menu_shortcuts_removed_by_uninstaller"] = True
        report["user_record_and_update_backup_preserved_after_uninstall"] = True
        report["verified"] = True
        return 0
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(report["error"], flush=True)
        return 1
    finally:
        report["elapsed_s"] = round(time.monotonic() - started, 3)
        (root / "acceptance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
