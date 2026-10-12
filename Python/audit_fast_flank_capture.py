"""Re-audit one closed fast-flank capture without touching hardware."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from capture_ch1_rc_diagnostic import load_installed_rows
from capture_ch1_teacher_spectrum import save_checkpoint, sha256_file
from fast_flank_features import audit_static_stream


def audit(source: Path, output: Path) -> dict:
    source = Path(source)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    before = sha256_file(source)
    report = json.loads(source.read_text(encoding="utf-8"))
    if not report.get("complete"):
        raise ValueError("source capture is not complete")
    cleanup = report.get("safety_cleanup") or {}
    if not cleanup.get("exact_disarm_ack") or not cleanup.get("exact_soa_shutter_ack"):
        raise ValueError("source capture lacks exact optical shutdown proof")
    if report.get("cnc_motion_commands") != 0:
        raise ValueError("static audit requires a no-motion capture")
    frames = []
    for raw in report.get("frames", []):
        frame = dict(raw)
        if "minimum_warmup_complete" not in frame:
            # 1.0.38 wire bit was already the minimum-warm-up bit.  Only the
            # first host field name was too strong; preserve source bytes.
            frame["minimum_warmup_complete"] = bool(frame.get("warmup_ready"))
        frames.append(frame)
    rows, _ = load_installed_rows()
    result = {
        "schema": "fast_flank_capture_audit_v1",
        "source": str(source.resolve()),
        "source_sha256": before,
        "source_frames": len(frames),
        "analysis": audit_static_stream(frames, rows),
        "training_eligible": False,
        "physical_15hz_verified": False,
        "location_accuracy_verified": False,
        "contact_area_verified": False,
    }
    if sha256_file(source) != before:
        raise RuntimeError("source capture changed during audit")
    save_checkpoint(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.source, args.output), ensure_ascii=False, indent=2))
