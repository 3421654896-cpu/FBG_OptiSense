"""Runtime resource and per-user data locations for source and packaged builds.

Source checkouts keep their historical layout so engineering tools and tests
continue to use ``Python/outputs``.  A frozen customer build stores mutable
files below LocalAppData instead of the installation directory.  This keeps
manual routes, reference spectra and encrypted OTA settings intact when the
desktop application is replaced during a remote update.
"""

from __future__ import annotations

import os
import json
import shutil
import sys
from pathlib import Path


PRODUCT_DATA_DIRECTORY = "FBGOptiSenseStudio"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def resource_root() -> Path:
    """Return the read-only application resource root."""

    bundled = getattr(sys, "_MEIPASS", None)
    if bundled:
        return Path(bundled).resolve()
    return Path(__file__).resolve().parent


def data_root() -> Path:
    """Return the writable application-data root.

    ``FBG_STUDIO_DATA_DIR`` is intentionally supported for release smoke tests
    and managed deployments.  In a source checkout we retain the old paths.
    """

    override = os.environ.get("FBG_STUDIO_DATA_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if not is_frozen():
        return resource_root()
    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
    return base / PRODUCT_DATA_DIRECTORY


def resource_path(*parts: str | os.PathLike[str]) -> Path:
    return resource_root().joinpath(*map(Path, parts))


def data_path(*parts: str | os.PathLike[str]) -> Path:
    return data_root().joinpath(*map(Path, parts))


def output_path(*parts: str | os.PathLike[str]) -> Path:
    return data_path("outputs", *parts)


def _migrate_legacy_temporary_data(root: Path) -> None:
    """Copy old flat temporary records into their per-machine directories.

    Releases before the universal multi-machine build stored every unit below
    ``outputs/temporary_test``.  Migrate before factory seeds are installed so
    an operator's own newer route wins.  Legacy records without ownership
    metadata predate machine two and therefore belong to machine one.
    """

    legacy_root = root / "outputs" / "temporary_test"
    if not legacy_root.is_dir():
        return
    for source in legacy_root.rglob("*.json"):
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        machine_id = str(payload.get("machine_id", "")).strip() or "machine_1"
        if not machine_id.startswith("machine_"):
            machine_id = "machine_1"
        relative = source.relative_to(legacy_root)
        destination = (
            root / "outputs" / "machines" / machine_id / "temporary_test" / relative
        )
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def ensure_client_data() -> Path:
    """Create writable folders and install only missing factory defaults."""

    root = data_root()
    output_path("logs").mkdir(parents=True, exist_ok=True)
    if not is_frozen() and root == resource_root():
        return root

    _migrate_legacy_temporary_data(root)

    seed_root = resource_path("client_seed")
    if not seed_root.is_dir():
        return root
    for source in seed_root.rglob("*"):
        relative = source.relative_to(seed_root)
        destination = root / relative
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    default_nine_peak = resource_path("factory_defaults", "9个光栅最新通过.json")
    rolling_capture = output_path("temporary_test", "last_finger_capture.json")
    if default_nine_peak.is_file() and not rolling_capture.exists():
        rolling_capture.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(default_nine_peak, rolling_capture)
    return root


__all__ = [
    "PRODUCT_DATA_DIRECTORY",
    "data_path",
    "data_root",
    "ensure_client_data",
    "is_frozen",
    "output_path",
    "resource_path",
    "resource_root",
]
