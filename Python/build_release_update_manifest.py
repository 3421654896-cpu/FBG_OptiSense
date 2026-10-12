"""Build the public latest-Release update index from finished release assets.

Example:
    python build_release_update_manifest.py --tag v1.0.5 \
      --asset FBG_OptiSense_Studio_Universal_v1.0.5_Setup.exe \
      --asset JDSU_F205RE_machine_3_v1.0.97.fbgfw \
      --out FBG_OptiSense_updates.json

Upload the resulting JSON alongside these binaries in the same GitHub Release.
For binaries retained from older releases, provide their original tag as
``--asset-tag OLD_TAG PATH`` instead of ``--asset PATH``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

from github_update_feed import UPDATE_REPOSITORY, parse_update_manifest


def build_manifest(entries: list[tuple[str, Path]]) -> dict:
    assets = []
    for tag, path in entries:
        if not tag or "/" in tag or "\\" in tag or tag in (".", ".."):
            raise ValueError(f"无效 Release 标签：{tag!r}")
        if not path.is_file():
            raise FileNotFoundError(path)
        name = path.name
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(256 * 1024), b""):
                digest.update(block)
        assets.append({
            "name": name,
            "size": path.stat().st_size,
            "digest": "sha256:" + digest.hexdigest(),
            "browser_download_url": (
                f"https://github.com/{UPDATE_REPOSITORY}/releases/download/"
                f"{quote(tag, safe='')}/{quote(name, safe='')}"
            ),
        })
    manifest = {"schema": 1, "assets": assets}
    offers = parse_update_manifest(manifest)
    if offers.desktop is None and not offers.firmware:
        raise ValueError("没有符合更新命名规范的客户端或固件资源")
    if len(assets) != int(offers.desktop is not None) + len(offers.firmware):
        raise ValueError("存在重复、无效或缺少机号的资源；拒绝生成更新清单")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="当前 Release 标签")
    parser.add_argument("--asset", action="append", type=Path, default=[],
                        help="当前 Release 上传的安装包或固件包，可重复")
    parser.add_argument("--asset-tag", action="append", nargs=2,
                        metavar=("TAG", "PATH"), default=[],
                        help="来自旧 Release 的资源标签与原文件路径，可重复")
    parser.add_argument("--out", required=True, type=Path,
                        help="待上传的 FBG_OptiSense_updates.json")
    args = parser.parse_args()
    entries = [(args.tag, path) for path in args.asset]
    entries.extend((tag, Path(path)) for tag, path in args.asset_tag)
    if not entries:
        parser.error("至少指定一个 --asset 或 --asset-tag")
    if args.out.name != "FBG_OptiSense_updates.json":
        parser.error("输出文件必须命名为 FBG_OptiSense_updates.json")
    args.out.write_text(
        json.dumps(build_manifest(entries), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"已生成 {args.out}，请与更新包一起上传到公开 Release")


if __name__ == "__main__":
    main()
