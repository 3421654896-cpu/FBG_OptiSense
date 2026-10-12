"""Signed offline delivery. No passwords, GitHub access or remote commands.

Client replacement is staged next to the installation, with a recoverable
old directory and rollback on rename failure. User data lives elsewhere and
is never deleted. Device firmware is selected by the physical USB UID.
"""
import base64
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import uuid
import zipfile
import psutil

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

CLIENT_EXE = 'FBG_OptiSense_Studio.exe'


def _is_link(path):
    return path.is_symlink() or getattr(path, 'is_junction', lambda: False)()


class UpdateError(RuntimeError):
    pass


def canonical_manifest(manifest):
    unsigned = {k: v for k, v in manifest.items() if k != 'signature'}
    return json.dumps(unsigned, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('utf-8')


def load_manifest(path, public_key):
    path = Path(path)
    if path.stat().st_size > 256*1024:
        raise UpdateError('更新清单过大')
    try:
        manifest = json.loads(path.read_text(encoding='utf-8'))
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key)).verify(
            base64.b64decode(manifest['signature'], validate=True), canonical_manifest(manifest))
        if manifest['schema'] != 'fbg_offline_update_v1':
            raise ValueError('schema')
    except Exception as exc:
        raise UpdateError('更新包签名无效；未执行更新') from exc
    return manifest


def checked_payload(root, item):
    root = Path(root).resolve()
    name = item['filename']
    if PurePosixPath(name).name != name or '\\' in name or ':' in name:
        raise UpdateError('更新文件名无效')
    path = root / name
    if path.is_symlink() or not path.is_file() or path.resolve().parent != root:
        raise UpdateError('更新文件缺失或不是普通文件')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    if path.stat().st_size != item['bytes'] or digest.hexdigest() != item['sha256']:
        raise UpdateError('更新文件 SHA-256 校验失败')
    return path


def firmware_for_serial(manifest, serial_number):
    items = [item for item in manifest.get('firmware', []) if serial_number in item['usb_serials']]
    if len(items) != 1:
        raise UpdateError('本更新包未授权给这台设备，未向它发送更新命令')
    return items[0]


def version_tuple(text):
    try:
        parts = tuple(int(p) for p in text.split('.'))
        if len(parts) != 3 or any(p < 0 for p in parts):
            raise ValueError()
        return parts
    except (TypeError, AttributeError, ValueError) as exc:
        raise UpdateError('版本号格式无效') from exc


def validate_install_target(target):
    target = Path(target).absolute()
    if any(_is_link(p) for p in (target, *target.parents)):
        raise UpdateError('安装目录不能是目录联接或符号链接')
    target = target.resolve()  # Windows 8.3 aliases are not directory links.
    blocked = [Path.home(), Path(os.environ.get('WINDIR', 'C:/Windows')),
               Path(os.environ.get('ProgramFiles', 'C:/Program Files'))]
    if target == Path(target.anchor) or target in blocked or len(target.parts) < 3:
        raise UpdateError('拒绝更新系统目录或宽泛目录')
    if not (target / CLIENT_EXE).is_file() or (target / CLIENT_EXE).is_symlink():
        raise UpdateError('请选择含有 OptiSense 客户端 EXE 的安装目录')
    for process in psutil.process_iter(['exe']):
        try:
            exe = process.info.get('exe')
            if exe and Path(exe).resolve() == (target / CLIENT_EXE).resolve():
                raise UpdateError('请先保存记录并关闭客户端，再更新软件')
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return target


def unpack_client(archive_path, stage):
    seen = set()
    with zipfile.ZipFile(archive_path) as archive:
        total = 0
        for info in archive.infolist():
            name = info.filename
            path = PurePosixPath(name)
            key = name.rstrip('/').casefold()
            if (not name or path.is_absolute() or '..' in path.parts or '\\' in name
                    or ':' in name or key in seen or any(p.rstrip(' .') != p for p in path.parts)
                    or any(p.split('.')[0].upper() in
                           {'CON','PRN','AUX','NUL',*[f'COM{i}' for i in range(1,10)],
                            *[f'LPT{i}' for i in range(1,10)]} for p in path.parts)
                    or stat.S_ISLNK(info.external_attr >> 16)):
                raise UpdateError('压缩包含不安全或重复路径')
            seen.add(key)
            total += info.file_size
            if total > 4*1024**3:
                raise UpdateError('软件解压大小超过上限')
            destination = stage.joinpath(*path.parts)
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, destination.open('xb') as output:
                    shutil.copyfileobj(source, output)
    if not (stage / CLIENT_EXE).is_file() or not (stage / '_internal').is_dir():
        raise UpdateError('不是完整的 OptiSense 软件包')


def replace_client(archive_path, target, expected_version, *, progress=lambda _: None):
    target = validate_install_target(target)
    identity = target / 'client_release.json'
    if identity.exists():
        current = json.loads(identity.read_text(encoding='utf-8'))['version']
        if version_tuple(current) >= version_tuple(expected_version):
            return dict(status='latest', version=current)
    stage = Path(tempfile.mkdtemp(prefix=target.name+'.staging-', dir=target.parent))
    backup = target.with_name(target.name+'.backup-'+uuid.uuid4().hex[:10])
    try:
        progress('正在校验并准备新软件，原软件保持不变')
        unpack_client(archive_path, stage)
        (stage/'client_release.json').write_text(json.dumps({'version': expected_version}), encoding='utf-8')
        # Legacy portable customer data is retained in both the backup and
        # the replacement. Frozen modern clients use LocalAppData instead.
        for name in ('outputs', 'user_data'):
            source = target/name
            if source.exists():
                if _is_link(source) or any(_is_link(p) for p in source.rglob('*')):
                    raise UpdateError('用户数据包含链接，需人工处理')
                shutil.copytree(source, stage/name, dirs_exist_ok=True)
        # Inno Setup's registered uninstaller still points to this directory.
        # Preserve its executable/data instead of orphaning the uninstall entry.
        for source in target.glob('unins*'):
            if re.fullmatch(r'unins\d+\.(exe|dat|msg)', source.name, flags=re.IGNORECASE):
                if _is_link(source) or not source.is_file():
                    raise UpdateError('卸载程序包含链接或不是普通文件，需人工处理')
                shutil.copy2(source, stage/source.name)
        validate_install_target(target)  # Reject a client opened during staging.
        target.rename(backup)
        try:
            stage.rename(target)
        except Exception:
            backup.rename(target)
            raise
        return dict(status='updated', version=expected_version, backup=str(backup), target=str(target))
    finally:
        # Only our exact newly-created staging directory may be removed.
        if stage.exists():
            if stage.resolve().parent != target.parent.resolve() or not stage.name.startswith(target.name+'.staging-'):
                raise UpdateError('暂存路径异常，拒绝清理')
            shutil.rmtree(stage)
