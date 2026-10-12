"""Explicit, local-only engineering profile; never enables customer updates."""

import os
from pathlib import Path
import shutil
import sys


def beta_enabled():
    return os.environ.get('FBG_STUDIO_BETA', '') == '1'


def enable_beta():
    """Call before importing launcher/acquisition modules and their paths."""
    os.environ['FBG_STUDIO_BETA'] = '1'
    if not os.environ.get('FBG_STUDIO_DATA_DIR', '').strip():
        if getattr(sys, 'frozen', False):
            base = Path(os.environ.get('LOCALAPPDATA') or Path.home() / 'AppData' / 'Local')
            root = base / 'FBGOptiSenseStudioBeta'
        else:
            root = Path(__file__).resolve().parent / 'beta_data'
        os.environ['FBG_STUDIO_DATA_DIR'] = str(root)


def seed_beta_data():
    """One-time copies of local settings/latest routes; never write back."""
    if not beta_enabled() or getattr(sys, 'frozen', False):
        return
    source = Path(__file__).resolve().parent
    destination = Path(os.environ['FBG_STUDIO_DATA_DIR']).resolve()
    if destination == source:
        raise ValueError('Beta data must not use the normal source data directory')
    relative_paths = [
        Path('outputs/preferences/finger_stress_scan.json'),
        Path('outputs/device_profiles/ota_machine_profiles.json'),
        Path('outputs/device_profiles/machine_identity.json'),
    ]
    for number in range(1, 5):
        for name in ('last_45_session.json', 'last_finger_capture.json'):
            relative_paths.append(Path(f'outputs/machines/machine_{number}/temporary_test/{name}'))
    for relative in relative_paths:
        existing, copied = source / relative, destination / relative
        if existing.is_file() and not copied.exists():
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(existing, copied)

