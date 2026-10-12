"""No-network checks for the public GitHub Releases update boundary."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path

from build_release_update_manifest import build_manifest
from github_update_feed import (
    ReleaseAsset,
    UpdateFeedError,
    download_release_asset,
    fetch_update_offers,
    parse_release_offers,
    parse_update_manifest,
)


REPO = "3421654896-cpu/JDSU_Laser"
ATOM_EMPTY = b'<feed xmlns="http://www.w3.org/2005/Atom"><title>Releases</title></feed>'
ATOM_WITH_RELEASE = (
    b'<feed xmlns="http://www.w3.org/2005/Atom">'
    b'<entry><title>v1.0.6</title></entry></feed>'
)


def item(name: str, content: bytes, *, digest: str | None = None) -> dict:
    return {
        "name": name,
        "size": len(content),
        "digest": digest or "sha256:" + hashlib.sha256(content).hexdigest(),
        "browser_download_url": (
            f"https://github.com/{REPO}/releases/download/v1/{name}"
        ),
    }


class FakeResponse:
    def __init__(self, content: bytes, url: str):
        self.content = content
        self.url = url
        self.position = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def geturl(self):
        return self.url

    def read(self, size=-1):
        if size < 0:
            size = len(self.content)
        start = self.position
        self.position += size
        return self.content[start:self.position]


class GitHubUpdateFeedTests(unittest.TestCase):
    def test_release_manifest_builder_hashes_local_assets(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "FBG_OptiSense_Studio_Universal_v1.0.6_Setup.exe"
            path.write_bytes(b"MZinstaller")
            manifest = build_manifest([("v1.0.6", path)])
        self.assertEqual(manifest["schema"], 1)
        self.assertEqual(
            manifest["assets"][0]["digest"],
            "sha256:" + hashlib.sha256(b"MZinstaller").hexdigest(),
        )
        self.assertEqual(parse_update_manifest(manifest).desktop.version, "1.0.6")

    def test_public_manifest_works_without_using_rate_limited_api(self):
        name = "FBG_OptiSense_Studio_Universal_v1.0.6_Setup.exe"
        manifest = {"schema": 1, "assets": [item(name, b"MZinstaller")]}
        self.assertEqual(parse_update_manifest(manifest).desktop.version, "1.0.6")
        calls = []

        def opener(request, **_kwargs):
            calls.append(request.full_url)
            return FakeResponse(
                json.dumps(manifest).encode(),
                "https://release-assets.githubusercontent.com/manifest",
            )

        offers = fetch_update_offers(opener=opener)
        self.assertEqual(offers.desktop.version, "1.0.6")
        self.assertEqual(len(calls), 1)
        self.assertIn("/releases/latest/download/", calls[0])

    def test_missing_manifest_uses_api_compatibility_fallback(self):
        name = "JDSU_F205RE_machine_3_v1.0.97.fbgfw"
        calls = []

        def opener(request, **_kwargs):
            calls.append(request.full_url)
            if "releases/latest/download" in request.full_url:
                raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)
            if request.full_url.endswith("/releases.atom"):
                return FakeResponse(
                    ATOM_WITH_RELEASE,
                    f"https://github.com/{REPO}/releases.atom",
                )
            return FakeResponse(
                json.dumps([{"assets": [item(name, b"firmware")]}]).encode(),
                "https://api.github.com/repos/3421654896-cpu/JDSU_Laser/releases",
            )

        offers = fetch_update_offers(opener=opener)
        self.assertEqual(offers.firmware["machine_3"].version, "1.0.97")
        self.assertEqual(len(calls), 3)

    def test_no_public_releases_is_empty_without_rate_limited_api(self):
        calls = []

        def opener(request, **_kwargs):
            calls.append(request.full_url)
            if "releases/latest/download" in request.full_url:
                raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)
            if request.full_url.endswith("/releases.atom"):
                return FakeResponse(
                    ATOM_EMPTY, f"https://github.com/{REPO}/releases.atom"
                )
            raise AssertionError("empty public feed must not call the API")

        self.assertEqual(fetch_update_offers(opener=opener).firmware, {})
        self.assertEqual(len(calls), 2)

    def test_failed_checks_never_claim_no_update(self):
        def opener(request, **_kwargs):
            raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, None)

        with self.assertRaisesRegex(UpdateFeedError, "均不可用"):
            fetch_update_offers(opener=opener)

    def test_desktop_and_per_machine_firmware_are_separate_and_newest(self):
        installer = b"MZinstaller"
        firmware = b"firmware"
        offers = parse_release_offers([
            {"draft": False, "prerelease": False, "assets": [
                item("FBG_OptiSense_Studio_Universal_v1.0.5_Setup.exe", installer),
                item("JDSU_F205RE_machine_3_v1.0.96.fbgfw", firmware),
            ]},
            {"draft": False, "prerelease": False, "assets": [
                item("FBG_OptiSense_Studio_Universal_v1.0.6_Setup.exe", installer),
                item("JDSU_F205RE_machine_3_v1.0.97.fbgfw", firmware),
                item("JDSU_F205RE_v9.9.9.fbgfw", firmware),
            ]},
            {"draft": True, "prerelease": False, "assets": [
                item("FBG_OptiSense_Studio_Universal_v9.0.0_Setup.exe", installer),
            ]},
        ])
        self.assertEqual(offers.desktop.version, "1.0.6")
        self.assertEqual(offers.firmware["machine_3"].version, "1.0.97")
        self.assertNotIn("machine_1", offers.firmware)

    def test_missing_digest_or_foreign_url_is_not_offered(self):
        unknown = item("JDSU_F205RE_machine_2_v1.0.98.fbgfw", b"firmware")
        unknown["digest"] = None
        foreign = item("FBG_OptiSense_Studio_Universal_v1.0.6_Setup.exe", b"MZ")
        foreign["browser_download_url"] = "https://example.com/setup.exe"
        offers = parse_release_offers([{"assets": [unknown, foreign]}])
        self.assertIsNone(offers.desktop)
        self.assertEqual(offers.firmware, {})

    def test_download_checks_sha256_and_does_not_leave_partial_file(self):
        content = b"MZ" + b"safe-test-installer"
        name = "FBG_OptiSense_Studio_Universal_v1.0.6_Setup.exe"
        raw = item(name, content)
        asset = ReleaseAsset(
            "desktop", "1.0.6", name, raw["browser_download_url"],
            raw["digest"].removeprefix("sha256:"), len(content), None,
        )
        with tempfile.TemporaryDirectory() as folder:
            target = download_release_asset(
                asset, cache_dir=Path(folder),
                opener=lambda *_args, **_kwargs: FakeResponse(
                    content, "https://release-assets.githubusercontent.com/test"
                ),
            )
            self.assertEqual(target.read_bytes(), content)
            bad_asset = ReleaseAsset(
                "desktop", asset.version, name, asset.url,
                "0" * 64, len(content), None,
            )
            with self.assertRaisesRegex(UpdateFeedError, "SHA-256"):
                download_release_asset(
                    bad_asset, cache_dir=Path(folder),
                    opener=lambda *_args, **_kwargs: FakeResponse(
                        content, "https://release-assets.githubusercontent.com/test"
                    ),
                )
            self.assertEqual(list(Path(folder).glob("*.part")), [])


if __name__ == "__main__":
    unittest.main()
