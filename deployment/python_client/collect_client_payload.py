"""Collect the local import graph for the customer Python package."""

from __future__ import annotations

import argparse
import ast
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ENTRY_MODULES = (
    "fbg_studio_launcher",
    "fbg_unified_app",
    "installed_launcher",
)
EXTRA_MODULES = (
    # These are launched or imported only after operator actions.
    "fullband_auto_calibration",
    "fullband_calibration_progress",
    "temporary_reference",
)
RESOURCE_EXTENSIONS = {
    ".csv",
    ".json",
    ".joblib",
    ".onnx",
    ".ps1",
    ".stl",
    ".txt",
    ".yaml",
    ".yml",
}


def local_module_path(source: Path, module: str) -> Path | None:
    top = module.split(".", 1)[0]
    module_file = source / f"{top}.py"
    if module_file.is_file():
        return module_file
    package = source / top / "__init__.py"
    return package if package.is_file() else None


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


def collect_modules(source: Path) -> set[Path]:
    pending = list(ENTRY_MODULES + EXTRA_MODULES)
    collected: set[Path] = set()
    while pending:
        module = pending.pop()
        path = local_module_path(source, module)
        if path is None or path in collected:
            continue
        collected.add(path)
        for imported in imported_modules(path):
            if local_module_path(source, imported) is not None:
                pending.append(imported)
    return collected


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_literal_resources(source: Path, destination: Path, modules: set[Path]) -> None:
    for module in modules:
        tree = ast.parse(module.read_text(encoding="utf-8-sig"), filename=str(module))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            raw = node.value.strip().replace("\\", "/")
            if not raw or Path(raw).suffix.lower() not in RESOURCE_EXTENSIONS:
                continue
            candidate = source / raw
            if candidate.is_file() and source in candidate.resolve().parents:
                copy_file(candidate, destination / candidate.relative_to(source))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--installed-launcher", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--firmware", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    copy_file(args.installed_launcher, destination / "installed_launcher.pyw")
    copy_file(args.requirements, destination / "client_requirements.txt")
    modules = collect_modules(source)
    for module in modules:
        copy_file(module, destination / module.relative_to(source))
    copy_literal_resources(source, destination, modules)

    required_resources = (
        "machine_profile.json",
        "fullband_equal_power_operational_2001.json",
        "fullband_equal_power_operational_2001.csv",
        "mode_tables_from_fullband_2001.json",
        "wave_const.yaml",
        "stress_wave_const.yaml",
        "temporary_test_default_plan.json",
        "temporary_test_defaults.json",
        "mach3_bridge.ps1",
    )
    for name in required_resources:
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(path)
        copy_file(path, destination / name)
    if (source / "assets").is_dir():
        shutil.copytree(source / "assets", destination / "assets", dirs_exist_ok=True)
    for dll_name in ("SolidWorks.Interop.sldworks.dll", "SolidWorks.Interop.swconst.dll"):
        dll = source / dll_name
        if dll.is_file():
            copy_file(dll, destination / dll.name)

    copy_file(args.firmware, destination / "firmware" / args.firmware.name)

    captures = source / "outputs" / "temporary_test" / "finger_captures"
    nine_peak = captures / "9个光栅最新通过.json"
    three_peak = captures / "0号手指.json"
    last_session = source / "outputs" / "temporary_test" / "last_45_session.json"
    for path in (nine_peak, three_peak, last_session):
        if not path.is_file():
            raise FileNotFoundError(path)

    seed_temp = destination / "client_seed" / "outputs" / "temporary_test"
    copy_file(nine_peak, seed_temp / "last_finger_capture.json")
    copy_file(last_session, seed_temp / "last_45_session.json")
    copy_file(nine_peak, seed_temp / "finger_captures" / nine_peak.name)
    copy_file(three_peak, seed_temp / "finger_captures" / three_peak.name)

    # Keep read-only copies at their profile-pinned paths for integrity checks.
    pinned = destination / "outputs" / "temporary_test" / "finger_captures"
    copy_file(nine_peak, pinned / nine_peak.name)
    copy_file(three_peak, pinned / three_peak.name)
    copy_file(last_session, destination / "outputs" / "temporary_test" / last_session.name)

    for name in (
        "wavelength_switch_board_analysis_45x3_20260917.json",
        "wavelength_switch_board_capture_45x3_20260917.json",
        "temporary_test_half_height_2ms_audit_forward_v1075_20260917.json",
    ):
        path = source / "outputs" / name
        if path.is_file():
            copy_file(path, destination / "client_seed" / "outputs" / name)

    version_namespace: dict[str, str] = {}
    exec((source / "app_version.py").read_text(encoding="utf-8"), version_namespace)
    release = {
        "schema": "fbg_python_customer_release_v1",
        "app_version": version_namespace["APP_VERSION"],
        "release_channel": version_namespace["APP_RELEASE_CHANNEL"],
        "machine_id": "machine_1",
        "machine_label": "一号机",
        "firmware_version": "1.0.86",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "module_count": len(modules),
    }
    (destination / "release.json").write_text(
        json.dumps(release, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(release, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
