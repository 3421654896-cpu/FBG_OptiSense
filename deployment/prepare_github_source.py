"""Prepare a clean source tree atop origin/main without changing the checkout.

No push, checkout, reset, hardware I/O or installer/source ZIP build. A separate
Git index preserves the user's index and dirty files. Remote history is retained.
"""

import argparse
import ast
import hashlib
import json
import os
import subprocess
import uuid
from pathlib import Path

from package_developer_snapshot import REPO, SOURCE_EXTS, payload_map, sanitize_config

REMOTE = 'https://github.com/3421654896-cpu/FBG_OptiSense.git'


def git(*args, data=None, env=None):
    return subprocess.run(['git', '-c', 'core.quotepath=false', *args], cwd=REPO,
                          input=data, stdout=subprocess.PIPE, check=True, env=env).stdout


def paths(ref):
    return git('ls-tree', '-r', '-z', '--name-only', ref).decode('utf-8').split('\0')[:-1]


def forbidden(path):
    parts = Path(path).parts
    return (any(p in {'.git', '.venv', '__pycache__', 'beta_data', 'build', 'dist',
                      '.release-signing', '.codex_tmp'} or p.endswith('_PRIVATE') for p in parts)
            or path.startswith('Python/fbg_remote_config.') and not path.endswith('.example')
            or Path(path).suffix.lower() in {'.pyc', '.pyo', '.exe', '.key', '.pem', '.axf', '.o', '.obj', '.log'})


def bulk_index(entries, env):
    """Use one Git index transaction; paths may contain spaces or Chinese."""
    if entries:
        data = ''.join(f'100644 {blob}\t{relative}\0' for relative, blob in entries)
        git('update-index', '-z', '--index-info', data=data.encode('utf-8'), env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='origin/main')
    args = parser.parse_args()
    if git('remote', 'get-url', 'origin').decode().strip() != REMOTE:
        raise ValueError('Unexpected publication remote')
    base = git('rev-parse', args.base).decode().strip()
    remote_paths = set(paths(base))
    raw_map = payload_map()
    # Current runtime data already present in the developer repository are
    # retained; new machine data comes only from the registry's pinned source.
    resources = {p for p in remote_paths if not forbidden(p) and '/outputs/' not in p}
    registry = json.loads((REPO / 'Python/machine_registry.json').read_text(encoding='utf-8'))
    for config in registry['machines'].values():
        fullband = config.get('fullband', {})
        for key in ('source_path', 'guard_path'):
            if fullband.get(key):
                resources.add('Python/' + fullband[key])
    resources.update({'Python/usb_firmware_catalog.json', 'Python/machine_registry.json',
                      'Python/machine_profile.json', 'Python/temporary_test_default_plan.json',
                      'Python/temporary_test_defaults.json',
                      'Python/tests/fixtures/forward_settle_analysis.json',
                      'Python/tests/fixtures/forward_settle_capture.json'})
    selected = {}
    for relative, source in raw_map.items():
        if forbidden(relative) or relative.startswith('backups/'):
            continue
        suffix = source.suffix.lower()
        # Factory seeds are approved deployment files, not the operator's
        # similarly named outputs. Raw captures/analysis JSON stay local.
        factory = source.is_relative_to(REPO / 'deployment')
        config = relative.startswith('Python/machine_defaults/') and source.name == 'temporary_test_defaults.json'
        code = suffix in SOURCE_EXTS or source.name.endswith('.example')
        approved_binary = relative.startswith('outputs/ota/') and suffix in {'.fbgfw', '.hex'}
        if code or factory or config or relative in resources or suffix in {'.stl', '.ico', '.dll'} or approved_binary:
            selected[relative] = source
    selected['README.md'] = REPO / 'README.md'
    selected['AGENTS.md'] = REPO / 'AGENTS.md'
    selected['.gitattributes'] = REPO / '.gitattributes'
    # The same reviewed factory defaults support both source/Beta layouts
    # and data-root overrides. Never alias the live operator outputs here.
    seed = REPO / 'deployment/universal/client_seed'
    for path in seed.rglob('*.json'):
        selected['Python/client_seed/' + path.relative_to(seed).as_posix()] = path
    legacy_seed = seed / 'outputs/machines/machine_1/temporary_test'
    profile = json.loads((REPO / 'Python/machine_profile.json').read_text(encoding='utf-8'))
    for item in profile['artifacts']:
        relative = 'Python/' + item['path']
        if relative.startswith('Python/outputs/temporary_test/'):
            selected[relative] = legacy_seed / Path(item['path']).relative_to('outputs/temporary_test')
        if relative not in selected or hashlib.sha256(selected[relative].read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Pinned calibration resource mismatch: ' + relative)
    # Do not replace the current README with the historical ZIP README alias.
    selected.pop('README_开发调试.md', None)
    for path in (REPO / 'JDSU/Drivers').rglob('*'):
        if path.is_file() and path.suffix.lower() in {'.lib', '.a'}:
            selected[path.relative_to(REPO).as_posix()] = path
    catalog = json.loads((REPO / 'Python/usb_firmware_catalog.json').read_text(encoding='utf-8'))
    for item in catalog['machines'].values():
        source = REPO / 'outputs/ota' / item['filename']
        if hashlib.sha256(source.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Approved compatibility firmware hash mismatch')
        selected['Python/firmware/usb/' + item['filename']] = source
    stage = REPO / '.codex_tmp' / ('github-publish-' + uuid.uuid4().hex)
    stage.mkdir(parents=True)
    default_index = Path(git('rev-parse', '--git-path', 'index').decode().strip())
    if not default_index.is_absolute():
        default_index = REPO / default_index
    index_before = default_index.read_bytes() if default_index.exists() else None
    env = {**os.environ, 'GIT_INDEX_FILE': str(stage / 'index')}
    git('read-tree', base, env=env)
    # Drop generated caches and live configurations/records from the new tree
    # only. Historical commits and all local files remain recoverable/untouched.
    removed = sorted(p for p in remote_paths if forbidden(p) or '/outputs/' in p)
    if removed:
        git('update-index', '--force-remove', '-z', '--stdin',
            data=('\0'.join(removed) + '\0').encode('utf-8'), env=env)
    entries = []
    sanitized = []
    index_entries = []
    unchanged_sources = []
    source_digests = {}
    for relative, source in sorted(selected.items()):
        data = source.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if source_digests.setdefault(source, digest) != digest:
            raise ValueError('Aliased source changed while preparing publication: ' + str(source))
        fields = []
        if source.suffix.lower() == '.json':
            safe = sanitize_config(json.loads(data.decode('utf-8-sig')), fields)
            if fields:
                data = (json.dumps(safe, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
                sanitized.append({'path': relative, 'fields': fields})
        if source.suffix.lower() in {'.py', '.pyw'}:
            ast.parse(data, filename=relative)
        if len(data) >= 100 * 1024 * 1024:
            raise ValueError('Oversized payload: ' + relative)
        if fields:
            blob = git('hash-object', '-w', '--stdin', data=data).decode().strip()
            index_entries.append((relative, blob))
        else:
            unchanged_sources.append((relative, source))
        entries.append(dict(path=relative, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    # Raw bytes are intentional: the manifest describes the exact tree, not
    # an implicit CRLF/filter transformation of a developer's current files.
    if unchanged_sources:
        names = ''.join(json.dumps(str(source), ensure_ascii=False) + '\n'
                        for _, source in unchanged_sources).encode('utf-8')
        blobs = git('hash-object', '-w', '--no-filters', '--stdin-paths', data=names).decode().splitlines()
        if len(blobs) != len(unchanged_sources):
            raise ValueError('Git did not hash every selected source')
        index_entries.extend((relative, blob) for (relative, _), blob in zip(unchanged_sources, blobs))
    bulk_index(index_entries, env)
    retained = sorted(set(paths(git('write-tree', env=env).decode().strip()))
                      - set(selected) - {'MANIFEST.sha256.json'})
    manifest = dict(schema='fbg_github_source_manifest_v1', base_remote_commit=base,
                    client_version='1.0.7', beta_firmware_version='1.0.118',
                    note='Source publication, not a customer installer or Release. No private runtime records.',
                    hash_scope='files lists current selected payload only; retained remote-only files listed separately',
                    retained_remote_paths=retained,
                    files=entries)
    raw = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    blob = git('hash-object', '-w', '--stdin', data=raw).decode().strip()
    git('update-index', '--add', '--cacheinfo', f'100644,{blob},MANIFEST.sha256.json', env=env)
    tree = git('write-tree', env=env).decode().strip()
    for path in paths(tree):
        if forbidden(path):
            raise ValueError('Forbidden content remains in publication tree: ' + path)
    for source, digest in source_digests.items():
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError('Source changed while preparing publication: ' + str(source))
    index_after = default_index.read_bytes() if default_index.exists() else None
    if index_before != index_after:
        raise ValueError('The default Git index changed while preparing publication')
    report = dict(base=base, tree=tree, files=len(entries),
                  selected_bytes=sum(e['bytes'] for e in entries),
                  removed_runtime_paths=removed, sanitized_fields=sanitized,
                  worktree_and_default_index_unchanged=True)
    (stage / 'receipt.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({**report, 'receipt': str(stage / 'receipt.json')}, ensure_ascii=True))


if __name__ == '__main__':
    main()
