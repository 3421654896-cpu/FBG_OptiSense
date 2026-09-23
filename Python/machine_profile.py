"""Active hardware-unit identity and calibration ownership metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROFILE_PATH = Path(__file__).with_name("machine_profile.json")
PROFILE_SCHEMA = "fbg_machine_profile_v1"

def _load_runtime_fullband_profiles() -> dict[str, dict]:
    """Load registered hardware units without requiring Python code changes."""

    path = Path(__file__).with_name("machine_registry.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "fbg_machine_registry_v1":
        raise ValueError("机号注册表格式不受支持")
    profiles = {}
    for machine_id, config in payload.get("machines", {}).items():
        fullband = dict(config.get("fullband", {}))
        machine_label = str(config.get("machine_label", "")).strip()
        if not machine_label or not fullband.get("source_path"):
            continue
        fullband["machine_label"] = machine_label
        fullband["parameter_owner"] = machine_label
        profiles[str(machine_id)] = fullband
    if not profiles:
        raise ValueError("机号注册表没有可用的2001点配置")
    return profiles


RUNTIME_FULLBAND_PROFILES = _load_runtime_fullband_profiles()


def load_machine_profile(path: Path = PROFILE_PATH) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != PROFILE_SCHEMA:
        raise ValueError("机号参数档案格式不受支持")
    for key in ("machine_id", "machine_label", "parameter_owner", "parameter_note"):
        if not str(payload.get(key, "")).strip():
            raise ValueError(f"机号参数档案缺少 {key}")
    if not isinstance(payload.get("parameter_groups"), dict):
        raise ValueError("机号参数档案缺少参数分组")
    if not isinstance(payload.get("artifacts"), list):
        raise ValueError("机号参数档案缺少文件指纹")
    return payload


ACTIVE_MACHINE_PROFILE = load_machine_profile()
MACHINE_ID = ACTIVE_MACHINE_PROFILE["machine_id"]
MACHINE_LABEL = ACTIVE_MACHINE_PROFILE["machine_label"]
PARAMETER_OWNER = ACTIVE_MACHINE_PROFILE["parameter_owner"]
PARAMETER_NOTE = ACTIVE_MACHINE_PROFILE["parameter_note"]
_runtime_machine_id = MACHINE_ID


def set_runtime_machine_id(machine_id: str) -> str:
    """Select the physical machine used by runtime calibration lookups."""

    from ota_machine_profiles import MACHINE_LABELS

    machine_id = str(machine_id)
    if machine_id not in MACHINE_LABELS:
        raise ValueError(f"不支持的机号：{machine_id}")
    global _runtime_machine_id
    _runtime_machine_id = machine_id
    return machine_id


def get_runtime_machine_id() -> str:
    return _runtime_machine_id


def runtime_machine_label(machine_id: str | None = None) -> str:
    from ota_machine_profiles import MACHINE_LABELS

    selected = str(machine_id or get_runtime_machine_id())
    return MACHINE_LABELS.get(selected, selected)


def runtime_fullband_profile(machine_id: str | None = None) -> dict:
    """Return a copy of the selected machine's 2001-point source metadata."""

    selected = str(machine_id or get_runtime_machine_id())
    profile = RUNTIME_FULLBAND_PROFILES.get(selected)
    if profile is None:
        raise ValueError(f"{runtime_machine_label(selected)}尚未配置2001点校准表")
    result = dict(profile)
    result["machine_id"] = selected
    return result


def runtime_parameter_note(machine_id: str | None = None) -> str:
    selected = str(machine_id or get_runtime_machine_id())
    try:
        profile = runtime_fullband_profile(selected)
    except ValueError:
        return f"{runtime_machine_label(selected)}尚未配置2001点校准表"
    if profile["certified"]:
        return (
            "当前9峰、2001点和3峰/15点参数均为一号机专用参数；"
            f"2001点：{profile['label']}（已认证）"
        )
    return (
        f"2001点：{profile['label']}（独立复测"
        f"{profile['validation_passed']}/2001通过，"
        f"{profile['validation_failed']}点待修复）"
    )


def machine_metadata(parameter_sets=()) -> dict:
    """Return stable metadata to embed in newly saved measurements."""

    machine_id = get_runtime_machine_id()
    machine_label = runtime_machine_label(machine_id)
    return {
        "machine_profile_schema": PROFILE_SCHEMA,
        "machine_id": machine_id,
        "machine_label": machine_label,
        "parameter_owner": machine_label,
        "parameter_sets": [str(item) for item in parameter_sets],
    }


def stamp_machine_metadata(payload: dict, parameter_sets=()) -> dict:
    """Copy a payload and mark it as belonging to the active hardware unit."""

    stamped = dict(payload)
    stamped.update(machine_metadata(parameter_sets))
    return stamped


def ensure_machine_compatible(payload: dict, *, allow_legacy=True) -> bool:
    """Reject records explicitly marked for a different physical machine.

    Existing records predate the machine-profile field.  They are allowed
    because their hashes are pinned in ``machine_profile.json`` and the user
    has explicitly identified them as 一号机 data.
    """

    record_id = str(payload.get("machine_id", "")).strip()
    record_label = str(payload.get("machine_label", "")).strip()
    if not record_id and not record_label:
        if allow_legacy:
            return False
        raise ValueError("记录未注明所属机号")
    machine_id = get_runtime_machine_id()
    machine_label = runtime_machine_label(machine_id)
    if record_id and record_id != machine_id:
        raise ValueError(
            f"记录属于 {record_label or record_id}，当前软件配置为 {machine_label}"
        )
    if record_label and record_label != machine_label:
        raise ValueError(
            f"记录属于 {record_label}，当前软件配置为 {machine_label}"
        )
    return True


def verify_profile_artifacts(root: Path | None = None) -> list[dict]:
    """Verify that the pinned 一号机 parameter artifacts are unchanged."""

    root = Path(root) if root is not None else Path(__file__).resolve().parent
    results = []
    for item in ACTIVE_MACHINE_PROFILE["artifacts"]:
        path = root / item["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        size = path.stat().st_size if path.exists() else None
        results.append({
            "path": item["path"],
            "exists": path.exists(),
            "sha256_ok": digest == item["sha256"],
            "size_ok": size == item["bytes"],
        })
    return results
