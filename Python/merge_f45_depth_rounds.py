"""Merge verified partial F45 depth rounds without copying capture payloads."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from run_f45_depth_round_visible import verify_capture


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output must be new")

    sources = []
    captures = {}
    depth = None
    for path in args.manifests:
        payload = json.loads(path.read_text("utf-8"))
        if payload.get("schema") != "f45_visible_depth_round_v1":
            raise ValueError(f"{path}: unsupported round manifest")
        source_depth = float(payload["depth_mm"])
        if depth is None:
            depth = source_depth
        elif abs(source_depth - depth) > 1e-9:
            raise ValueError("source depths do not match")
        for key, stored in payload.get("captures", {}).items():
            point = int(key)
            if point in captures:
                raise ValueError(f"duplicate verified point {point}")
            verified = verify_capture(Path(stored["capture"]), point, source_depth)
            verified["source_manifest"] = str(path.resolve())
            captures[point] = verified
        sources.append({"path": str(path.resolve()), "sha256": sha256(path)})
    if set(captures) != set(range(1, 18)):
        raise ValueError(f"merged points are not exactly 1..17: {sorted(captures)}")

    result = {
        "schema": "f45_visible_depth_round_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "round_name": args.output.stem,
        "depth_mm": depth,
        "planned_points": list(range(1, 18)),
        "completed_points": list(range(1, 18)),
        "captures": {str(point): captures[point] for point in range(1, 18)},
        "complete": True,
        "merged_from": sources,
        "invalid_or_unlisted_attempts_are_not_referenced": True,
        "safety": {
            "maximum_indentation_mm": 0.8,
            "spindle_policy": "M5 before every movement; spindle never started",
            "all_captures_reverified_during_merge": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "depth_mm": depth,
        "points": len(captures),
        "sources": sources,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
