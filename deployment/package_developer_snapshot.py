"""Create a private-data-free snapshot of the current, including dirty, worktree.

No hardware, Git writes, deletion, or signing keys are used. Existing bundles
are never replaced. The manifest records every byte and any sanitized config.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import subprocess
import sys
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", ".venv", "__pycache__", ".ruff_cache", ".pytest_cache",
              ".release-signing", "build", "dist", "tmp", ".codex_tmp"}
SOURCE_EXTS = {".py", ".pyw", ".c", ".h", ".s", ".sct", ".ioc", ".uvprojx",
               ".ps1", ".bat", ".vbs", ".spec", ".iss", ".md", ".txt", ".xml", ".pdsc"}
SECRET_KEY = re.compile(r"password|passwd|passphrase|credential|secret|private.?key|"
                        r"access.?token|refresh.?token|auth.?token|dpapi|protected.*password", re.I)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sanitize_config(obj, removed, location=""):
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            field = f"{location}{key}"
            if SECRET_KEY.search(key) and value:
                removed.append(field)
                result[key] = "" if isinstance(value, str) else None
            else:
                result[key] = sanitize_config(value, removed, field + ".")
        return result
    if isinstance(obj, list):
        return [sanitize_config(v, removed, location) for v in obj]
    return obj


def portable_file(path: Path) -> bool:
    parts = path.relative_to(REPO).parts
    # Keil keeps generated dumps beside the manually maintained scatter file.
    if parts[:3] == ("JDSU", "MDK-ARM", "JDSU") and path.suffix.lower() != ".sct":
        return False
    return not any(p in SKIP_PARTS or p.endswith("_PRIVATE") for p in parts)


def payload_map():
    files: dict[str, Path] = {}

    def add(source, dest=None, required=True):
        path = REPO / source
        if not path.is_file():
            if required:
                raise FileNotFoundError(source)
            return
        if not portable_file(path):
            raise ValueError(f"Forbidden source path: {source}")
        files[dest or Path(source).as_posix()] = path

    # Include ALL current local Python tools/tests, not an obsolete module list.
    # JSON/CSV/YAML tables are included for analysis tests and calibration tools;
    #现场 remote credentials and model binaries are deliberately not included.
    for path in (REPO / "Python").iterdir():
        if not path.is_file():
            continue
        if path.name.startswith("fbg_remote_config") and not path.name.endswith(".example"):
            continue
        if path.suffix.lower() == ".txt" and not path.name.startswith("requirements"):
            continue  # calibration consoles/diagnostics are not source
        if path.suffix.lower() in SOURCE_EXTS | {".json", ".csv", ".yaml", ".yml"} or path.name.endswith(".example"):
            if path.name.endswith("_console.txt") or "console_" in path.name:
                continue
            add(path.relative_to(REPO).as_posix())
    for base in ["JDSU", "deployment", "docs", "Python/assets", "Python/tests", "Python/machine_defaults"]:
        for path in (REPO / base).rglob("*"):
            if not path.is_file() or not portable_file(path):
                continue
            ext = path.suffix.lower()
            permitted = ext in SOURCE_EXTS | {".json", ".yaml", ".yml", ".csv", ".stl", ".ico"}
            if path.name in {".gitignore", ".mxproject", "LICENSE", "LICENSE.txt"}:
                permitted = True
            if permitted:
                add(path.relative_to(REPO).as_posix())
    for path in REPO.iterdir():
        if path.is_file() and path.suffix.lower() in {".md", ".txt", ".bat", ".vbs"}:
            add(path.relative_to(REPO).as_posix())
    add(".gitignore")
    add("Python/artifacts/ch1_measured_template_selection_v4_certified_20260905.json")
    add("Python/calibration_backups/2号解调仪第一次校准/2号解调仪第一次校准_2001点.json")
    for name in ["SolidWorks.Interop.sldworks.dll", "SolidWorks.Interop.swconst.dll"]:
        add("Python/" + name, required=False)
    # These are public compiled compatibility resources required by checked-in
    # PyInstaller specs, not dumps of connected devices' Flash/configuration.
    for name in ["JDSU_F205RE_v1.0.87.fbgfw", "JDSU_OTA_FirstInstall_v1.0.87.hex",
                 "JDSU_F205RE_v1.0.106.fbgfw", "JDSU_F205RE_v1.0.109_thermal_candidate.fbgfw"]:
        add("outputs/ota/" + name)
    # Retain the backed-up original 45-point logic and legacy feature source,
    # but never include backup binaries, runtime records or private board dumps.
    for base in ["backups/reference45_before_10ms_20261010",
                 "backups/hidden_features_20261009_d8e7f706"]:
        for path in (REPO / base).rglob("*"):
            if (path.is_file() and portable_file(path) and path.suffix.lower() in SOURCE_EXTS
                    and (path.suffix.lower() != ".txt" or path.name.startswith(("README", "LICENSE")))):
                add(path.relative_to(REPO).as_posix())
    for name in ["README_开发调试.md", "requirements-lock.txt", "requirements-build.txt",
                 "setup_dev.ps1", "start_demo.ps1", "start_dev.ps1"]:
        add("deployment/developer_bundle/" + name, name)
    # Empty portable output seed replaces现场 runtime profiles/DPAPI storage.
    seed = REPO / "deployment/universal/client_seed/outputs"
    for path in seed.rglob("*.json"):
        add(path.relative_to(REPO).as_posix(), "Python/outputs/" + path.relative_to(seed).as_posix())
    return files


def verify_archive(archive: Path):
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None, "ZIP CRC failure"
        manifests = [name for name in z.namelist() if name.endswith("/MANIFEST.sha256.json")]
        assert len(manifests) == 1
        manifest = json.loads(z.read(manifests[0]))
        prefix = manifests[0].removesuffix("MANIFEST.sha256.json")
        expected = {prefix + e["path"] for e in manifest["files"]} | {manifests[0]}
        assert set(z.namelist()) == expected, "Unlisted or missing ZIP members"
        for entry in manifest["files"]:
            rel = Path(entry["path"])
            assert not rel.is_absolute() and ".." not in rel.parts
            assert not any(p in SKIP_PARTS or p.endswith("_PRIVATE") for p in rel.parts)
            assert rel.suffix.lower() not in {".exe", ".pyc", ".key", ".pem", ".o", ".axf"}
            content = z.read(prefix + entry["path"])
            assert len(content) == entry["bytes"] and digest(content) == entry["sha256"], entry["path"]
            if rel.suffix.lower() in {".py", ".pyw"}:
                # Check without importing scripts that might control hardware.
                ast.parse(content, filename=entry["path"])
            if rel.suffix.lower() == ".json":
                config = json.loads(content.decode("utf-8-sig"))
                removed = []
                sanitize_config(config, removed)
                assert not removed, f"Nonempty sensitive JSON fields: {entry['path']}"
        for rel in ["Python/fbg_unified_app.py", "Python/dense_scan_transport.py",
                    "Python/offline_update_helper.py", "Python/offline_update_engine.py",
                    "Python/board_thermal_protocol.py", "Python/test_temporary_reference.py",
                    "JDSU/AGENTS.md", "JDSU/Core/Inc/firmware_features.h",
                    "JDSU/MDK-ARM/JDSU/JDSU.sct", "JDSU/MDK-ARM/RTE/_JDSU/RTE_Components.h",
                    "JDSU/Bootloader/ota_bootloader.c", "deployment/pyinstaller/offline_update_helper.spec"]:
            assert prefix + rel in expected, f"Missing required source: {rel}"
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="")
    parser.add_argument("--output-directory", type=Path, default=REPO / "release/developer")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        manifest = verify_archive(args.verify)
        print(json.dumps({"verified": True, "files": len(manifest["files"]),
                          "client": manifest["application_version"], "firmware": manifest["firmware_version"]}))
        return
    version = re.search(r'APP_VERSION\s*=\s*"([^"]+)"', (REPO / "Python/app_version.py").read_text())[1]
    if args.version and args.version != version:
        raise ValueError("Requested version differs from actual current source")
    header = (REPO / "JDSU/Core/Inc/firmware_version.h").read_text()
    firmware = ".".join(re.search(rf"APP_FIRMWARE_VERSION_{part}\s+(\d+)U", header)[1]
                        for part in ["MAJOR", "MINOR", "PATCH"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"OptiSense_最新源码_v{version}_固件v{firmware}_{stamp}"
    output = args.output_directory.resolve()
    output.relative_to(REPO)  # no broad external filesystem operations
    output.mkdir(parents=True, exist_ok=True)
    archive = output / (name + ".zip")
    if archive.exists():
        raise FileExistsError(archive)
    stage = REPO / "build" / ("developer-snapshot-" + uuid.uuid4().hex) / name
    stage.mkdir(parents=True)
    entries = []
    for relative, source in sorted(payload_map().items()):
        raw = source.read_bytes()
        data = raw
        removed = []
        if source.suffix.lower() == ".json":
            obj = json.loads(raw.decode("utf-8-sig"))
            safe = sanitize_config(obj, removed)
            if removed:
                data = (json.dumps(safe, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        entry = {"path": relative, "bytes": len(data), "sha256": digest(data),
                 "source_path": source.relative_to(REPO).as_posix(), "source_sha256": digest(raw)}
        if removed:
            entry["sanitized_fields"] = removed
        entries.append(entry)
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    manifest = {"schema": "fbg_current_worktree_developer_snapshot_v3", "application_version": version,
                "firmware_version": firmware, "base_git_commit": base,
                "snapshot": "current working files including uncommitted/untracked source; not git archive",
                "generated_utc": datetime.now(timezone.utc).isoformat(), "python": "CPython 3.12 x64",
                "payload_file_count": len(entries), "files": entries,
                "exclusions": ["Git history", "signing private keys", "现场 passwords/DPAPI",
                               "private full-Flash backups", "venv/build caches/EXE", "historical output logs",
                               "large trained joblib models (regenerate separately)"]}
    (stage / "MANIFEST.sha256.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    partial = archive.with_suffix(".zip.partial")
    with zipfile.ZipFile(partial, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                z.write(path, name + "/" + path.relative_to(stage).as_posix())
    verify_archive(partial)
    # Confirm every packaged current source still matches the local worktree.
    for entry in entries:
        assert digest((REPO / entry["source_path"]).read_bytes()) == entry["source_sha256"], "Worktree changed while packaging"
    partial.rename(archive)
    sha = digest(archive.read_bytes())
    archive.with_suffix(".zip.sha256").write_text(sha + "  " + archive.name + "\n", encoding="utf-8")
    print(json.dumps({"zip": str(archive), "stage": str(stage), "bytes": archive.stat().st_size,
                      "sha256": sha, "files": len(entries), "verified": True,
                      "sanitized_files": sum("sanitized_fields" in e for e in entries)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
