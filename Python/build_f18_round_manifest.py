"""Build an immutable manifest for one guarded 17-position F18 round.

The command validates acquisition, motion cleanup and optical shutdown before
recording hashes.  It never opens a device or commands the CNC.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from capture_fast_flank_stream import ROWS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_capture(path: Path, point: int, depth_mm: float) -> dict:
    payload = json.loads(path.read_text("utf-8"))
    commission = payload.get("commissioning") or {}
    safety = payload.get("safety_cleanup") or {}
    analysis = payload.get("analysis") or {}
    target = commission.get("target") or {}
    if payload.get("schema") != "ch1_fast_flank_stream_capture_v1":
        raise ValueError(f"{path}: not an F18 capture")
    if commission.get("schema") != "bounded_indentation_fast_flank_commission_v3":
        raise ValueError(f"{path}: not a guarded F18 commissioning capture")
    if not payload.get("complete") or len(payload.get("frames") or ()) != 640:
        raise ValueError(f"{path}: incomplete 640-frame capture")
    if tuple(payload.get("route_rows") or ()) != tuple(ROWS):
        raise ValueError(f"{path}: F18 route mismatch")
    if int(target.get("point_index", -1)) != point:
        raise ValueError(f"{path}: point label mismatch")
    if abs(float(target.get("depth_mm", -1.0)) - depth_mm) > 1e-9:
        raise ValueError(f"{path}: indentation mismatch")
    if commission.get("capture_error") or commission.get("motion_failures"):
        raise ValueError(f"{path}: capture or motion failure")
    if not safety.get("exact_disarm_ack") or not safety.get("exact_soa_shutter_ack"):
        raise ValueError(f"{path}: final optical shutdown not proven")
    if not analysis.get("meets_sampling_rate_for_15hz"):
        raise ValueError(f"{path}: >30 Hz sampling-rate gate failed")
    events = commission.get("events") or []
    if not events or events[-1].get("kind") != "final_safe_state_confirmed":
        raise ValueError(f"{path}: final safe CNC state missing")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "point_index": point,
        "depth_mm": depth_mm,
        "frames": len(payload["frames"]),
        "board_rate_hz": float(analysis["board_rate_hz"]),
        "board_interval_max_us": int(analysis["board_interval_max_us"]),
        "final_machine_status": events[-1].get("status"),
    }


def build(root: Path, pattern: str, depth_mm: float) -> dict:
    records = []
    for point in range(1, 18):
        matches = sorted(root.glob(pattern.format(point=point)))
        if len(matches) != 1:
            raise ValueError(
                f"P{point}: expected one capture for {pattern!r}, found {len(matches)}"
            )
        records.append(validate_capture(matches[0], point, depth_mm))
    return {
        "schema": "f18_guarded_position_round_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "depth_mm": depth_mm,
        "positions": 17,
        "captures": records,
        "minimum_rate_hz": min(row["board_rate_hz"] for row in records),
        "maximum_interval_us": max(row["board_interval_max_us"] for row in records),
        "all_motion_cleanup_confirmed": True,
        "all_optical_shutdown_confirmed": True,
        "physical_15hz_verified": False,
        "note": "Sampling cadence is qualified; no 15 Hz physical pressure waveform was applied.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--depth-mm", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    manifest = build(args.root, args.pattern, args.depth_mm)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "captures"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
