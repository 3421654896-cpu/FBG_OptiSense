"""Read public GitHub Release assets for separate desktop and board updates.

Only versioned, machine-specific firmware names with a GitHub SHA-256 digest
are eligible.  This prevents a generic STM32 package from being silently
offered to the wrong calibrated instrument.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

from ota_machine_profiles import MACHINE_LABELS
from runtime_paths import data_path


UPDATE_REPOSITORY = "3421654896-cpu/JDSU_Laser"
_API_URL = f"https://api.github.com/repos/{UPDATE_REPOSITORY}/releases?per_page=30"
_MANIFEST_URL = (
    f"https://github.com/{UPDATE_REPOSITORY}/releases/latest/download/"
    "FBG_OptiSense_updates.json"
)
_RELEASES_ATOM_URL = f"https://github.com/{UPDATE_REPOSITORY}/releases.atom"
_DESKTOP_NAME = re.compile(
    r"FBG_OptiSense_Studio_Universal_v(\d+\.\d+\.\d+)_Setup\.exe\Z"
)
_FIRMWARE_NAME = re.compile(
    r"JDSU_F205RE_(machine_\d+)_v(\d+\.\d+\.\d+)\.fbgfw\Z"
)
_SHA256 = re.compile(r"sha256:([0-9a-fA-F]{64})\Z")
_MAX_API_BYTES = 2 * 1024 * 1024
_MAX_DESKTOP_BYTES = 500 * 1024 * 1024
_MAX_FIRMWARE_BYTES = 1024 * 1024


class UpdateFeedError(RuntimeError):
    """An update offer could not be fetched or validated."""


def version_parts(value: str) -> tuple[int, int, int]:
    parts = str(value).split(".")
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise ValueError("更新版本号必须是 major.minor.patch")
    result = tuple(int(part) for part in parts)
    if result[0] > 65535 or any(part > 255 for part in result[1:]):
        raise ValueError("更新版本号超出固件版本格式")
    return result  # type: ignore[return-value]


@dataclass(frozen=True)
class ReleaseAsset:
    kind: str
    version: str
    name: str
    url: str
    sha256: str
    size: int
    machine_id: str | None = None


@dataclass(frozen=True)
class UpdateOffers:
    desktop: ReleaseAsset | None
    firmware: dict[str, ReleaseAsset]


def _github_asset_url(url: str) -> bool:
    parsed = urlsplit(url)
    return (
        parsed.scheme == "https"
        and parsed.hostname == "github.com"
        and parsed.path.startswith(
            f"/{UPDATE_REPOSITORY}/releases/download/"
        )
    )


def parse_release_offers(payload: object) -> UpdateOffers:
    if not isinstance(payload, list):
        raise UpdateFeedError("GitHub Releases 返回格式不正确")
    desktop = None
    firmware: dict[str, ReleaseAsset] = {}
    for release in payload:
        if not isinstance(release, dict) or release.get("draft") or release.get("prerelease"):
            continue
        for raw in release.get("assets", []):
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name", ""))
            desktop_match = _DESKTOP_NAME.fullmatch(name)
            firmware_match = _FIRMWARE_NAME.fullmatch(name)
            if desktop_match:
                kind, machine_id, version, maximum = (
                    "desktop", None, desktop_match.group(1), _MAX_DESKTOP_BYTES
                )
            elif firmware_match and firmware_match.group(1) in MACHINE_LABELS:
                kind, machine_id, version, maximum = (
                    "firmware", firmware_match.group(1),
                    firmware_match.group(2), _MAX_FIRMWARE_BYTES
                )
            else:
                continue
            try:
                version_parts(version)
                size = int(raw.get("size", -1))
                digest_match = _SHA256.fullmatch(str(raw.get("digest", "")))
                url = str(raw.get("browser_download_url", ""))
                if not 0 < size <= maximum or not digest_match or not _github_asset_url(url):
                    continue
            except (ValueError, TypeError):
                continue
            asset = ReleaseAsset(
                kind, version, name, url, digest_match.group(1).lower(),
                size, machine_id
            )
            if kind == "desktop":
                if desktop is None or version_parts(version) > version_parts(desktop.version):
                    desktop = asset
            else:
                previous = firmware.get(machine_id)
                if previous is None or version_parts(version) > version_parts(previous.version):
                    firmware[machine_id] = asset
    return UpdateOffers(desktop, firmware)


def parse_update_manifest(payload: object) -> UpdateOffers:
    """Parse the public latest-Release index without consuming GitHub API quota."""

    if not isinstance(payload, dict) or payload.get("schema") != 1:
        raise UpdateFeedError("GitHub 更新清单格式不正确")
    assets = payload.get("assets")
    if not isinstance(assets, list):
        raise UpdateFeedError("GitHub 更新清单缺少资源列表")
    return parse_release_offers([{"assets": assets}])


def _fetch_json(url: str, opener, *, api: bool) -> object:
    headers = {
        "Accept": "application/vnd.github+json" if api else "application/json",
        "User-Agent": "FBG-OptiSense-Studio-Updater",
    }
    if api:
        headers["X-GitHub-Api-Version"] = "2022-11-28"
    request = urllib.request.Request(url, headers=headers)
    try:
        with opener(request, timeout=8) as response:
            data = response.read(_MAX_API_BYTES + 1)
            if len(data) > _MAX_API_BYTES:
                raise UpdateFeedError("GitHub 更新列表过大")
            final = urlsplit(response.geturl())
            trusted = (
                final.hostname == "api.github.com" if api else
                final.hostname == "github.com" or
                (final.hostname or "").endswith(".githubusercontent.com")
            )
            if final.scheme != "https" or not trusted:
                raise UpdateFeedError("更新列表不是从 GitHub 安全连接取得")
    except (OSError, ValueError) as exc:
        raise UpdateFeedError(f"无法检查 GitHub 更新：{exc}") from exc
    try:
        return json.loads(data)
    except (ValueError, TypeError) as exc:
        raise UpdateFeedError("GitHub 更新列表不是有效 JSON") from exc


def _has_public_releases(opener) -> bool:
    """Check the public Atom feed when the manifest is not yet published.

    An empty, valid feed means there are no Releases. A failed request must
    never be interpreted as an empty feed or as an up-to-date installation.
    """

    request = urllib.request.Request(
        _RELEASES_ATOM_URL,
        headers={"User-Agent": "FBG-OptiSense-Studio-Updater"},
    )
    try:
        with opener(request, timeout=8) as response:
            data = response.read(_MAX_API_BYTES + 1)
            if len(data) > _MAX_API_BYTES:
                raise UpdateFeedError("GitHub Release 订阅列表过大")
            final = urlsplit(response.geturl())
            if final.scheme != "https" or final.hostname != "github.com":
                raise UpdateFeedError("Release 订阅列表不是从 GitHub 安全连接取得")
        root = ET.fromstring(data)
    except (OSError, ValueError, ET.ParseError) as exc:
        raise UpdateFeedError(f"无法读取公开 Release 列表：{exc}") from exc
    atom = "{http://www.w3.org/2005/Atom}"
    if root.tag != atom + "feed":
        raise UpdateFeedError("GitHub Release 订阅列表格式不正确")
    return bool(root.findall(atom + "entry"))


def fetch_update_offers(*, opener=None) -> UpdateOffers:
    opener = opener or urllib.request.urlopen
    try:
        return parse_update_manifest(_fetch_json(_MANIFEST_URL, opener, api=False))
    except UpdateFeedError as manifest_error:
        try:
            if not _has_public_releases(opener):
                return UpdateOffers(None, {})
        except UpdateFeedError:
            # If the feed itself is unavailable, the API may still work.
            pass
        # Older Releases may not have a manifest. The REST API remains a
        # compatibility fallback, but the index avoids its anonymous limit.
        try:
            return parse_release_offers(_fetch_json(_API_URL, opener, api=True))
        except UpdateFeedError as api_error:
            raise UpdateFeedError(
                f"公开清单和 GitHub API 均不可用：{manifest_error}；{api_error}"
            ) from api_error


def download_release_asset(
    asset: ReleaseAsset,
    *,
    cache_dir: Path | None = None,
    opener=None,
    cancelled: Callable[[], bool] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Download to a private cache and atomically publish only verified bytes."""

    if not _github_asset_url(asset.url) or not re.fullmatch(r"[0-9a-f]{64}", asset.sha256):
        raise UpdateFeedError("更新包来源或摘要不可信")
    if asset.kind == "desktop":
        match = _DESKTOP_NAME.fullmatch(asset.name)
        valid_name = bool(match and match.group(1) == asset.version)
    elif asset.kind == "firmware":
        match = _FIRMWARE_NAME.fullmatch(asset.name)
        valid_name = bool(
            match and match.group(1) == asset.machine_id
            and match.group(2) == asset.version
        )
    else:
        valid_name = False
    if not valid_name:
        raise UpdateFeedError("更新包文件名与类型不符")
    maximum = _MAX_DESKTOP_BYTES if asset.kind == "desktop" else _MAX_FIRMWARE_BYTES
    if not 0 < asset.size <= maximum:
        raise UpdateFeedError("更新包大小超出限制")
    if asset.kind == "firmware" and asset.machine_id not in MACHINE_LABELS:
        raise UpdateFeedError("更新包没有有效机号")
    destination_dir = Path(cache_dir or data_path("updates"))
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / asset.name
    if destination.is_file() and _sha256_file(destination) == asset.sha256:
        if asset.kind != "desktop" or _is_windows_executable(destination):
            return destination
    temporary = destination_dir / f"{asset.name}.{asset.sha256[:12]}.part"
    opener = opener or urllib.request.urlopen
    request = urllib.request.Request(
        asset.url, headers={"User-Agent": "FBG-OptiSense-Studio-Updater"}
    )
    digest = hashlib.sha256()
    transferred = 0
    try:
        with opener(request, timeout=30) as response:
            final = urlsplit(response.geturl())
            if final.scheme != "https" or not (
                final.hostname == "github.com"
                or (final.hostname or "").endswith(".githubusercontent.com")
            ):
                raise UpdateFeedError("下载被重定向到非 GitHub 地址")
            with temporary.open("wb") as stream:
                while True:
                    if cancelled is not None and cancelled():
                        raise UpdateFeedError("已取消下载")
                    chunk = response.read(256 * 1024)
                    if not chunk:
                        break
                    transferred += len(chunk)
                    if transferred > asset.size or transferred > maximum:
                        raise UpdateFeedError("下载数据超过更新包声明大小")
                    digest.update(chunk)
                    stream.write(chunk)
                    if progress is not None:
                        progress(transferred, asset.size)
        if transferred != asset.size or digest.hexdigest() != asset.sha256:
            raise UpdateFeedError("更新包 SHA-256 或长度校验失败")
        if asset.kind == "desktop":
            if not _is_windows_executable(temporary):
                raise UpdateFeedError("客户端安装包不是 Windows 可执行文件")
        temporary.replace(destination)
        return destination
    except (OSError, ValueError) as exc:
        raise UpdateFeedError(f"下载更新包失败：{exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(256 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_windows_executable(path: Path) -> bool:
    with path.open("rb") as stream:
        return stream.read(2) == b"MZ"


__all__ = [
    "ReleaseAsset", "UpdateFeedError", "UpdateOffers", "UPDATE_REPOSITORY",
    "download_release_asset", "fetch_update_offers", "parse_release_offers",
    "parse_update_manifest", "version_parts",
]
