"""Installed, console-free entry point for the Python customer package."""

from __future__ import annotations

import os
from pathlib import Path


local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
if local_app_data:
    os.environ.setdefault(
        "FBG_STUDIO_DATA_DIR",
        str(Path(local_app_data) / "FBGOptiSenseStudio"),
    )

from fbg_studio_launcher import main


raise SystemExit(main())
