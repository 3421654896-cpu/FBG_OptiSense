"""Run one visible, resumable F45 depth round over calibrated coverage points."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_capture(path: Path, point: int, depth: float, ch1_selector: int = 2) -> dict:
    data = json.loads(path.read_text("utf-8"))
    commissioning = data.get("commissioning") or {}
    safety = data.get("safety_cleanup") or {}
    events = commissioning.get("events") or []
    final = events[-1] if events else {}
    status = final.get("status") or {}
    checks = {
        "complete": bool(data.get("complete")),
        "gain_matches": data.get('feedback_selectors', [0,2]) == [0,ch1_selector]
            and all(f.get('feedback_selectors', [0,2]) == [0,ch1_selector]
                    for f in data.get('frames', [])),
        "frame_count_at_least_640": len(data.get("frames") or ()) >= 640,
        "point_matches": int((commissioning.get("target") or {}).get("point_index", -1)) == point,
        "depth_matches": abs(float(commissioning.get("depth_mm", -1.0)) - depth) <= 1e-9,
        "no_capture_error": commissioning.get("capture_error") is None,
        "no_motion_failure": not commissioning.get("motion_failures"),
        "final_safe_event": final.get("kind") == "final_safe_state_confirmed",
        "machine_stopped": bool(status.get("stopped")) and not bool(status.get("moving")),
        "not_estopped": not bool(status.get("estop")) and not bool(status.get("estop_led")),
        "axes_homed": all(bool(status.get(f"{axis}_homed")) for axis in "xyz"),
        "no_axis_limit": not any(bool(status.get(key)) for key in (
            "x_pos_limit", "x_neg_limit", "y_pos_limit", "y_neg_limit", "z_pos_limit", "z_neg_limit"
        )),
        "exact_disarm_ack": bool(safety.get("exact_disarm_ack")),
        "exact_soa_shutter_ack": bool(safety.get("exact_soa_shutter_ack")),
        "all_45_points_fresh": int((data.get("analysis") or {}).get("fresh_point_count_per_cycle", -1)) == 45,
        "sampling_rate_gate": bool((data.get("analysis") or {}).get("meets_sampling_rate_for_15hz")),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"capture verification failed: {failed}")
    return {
        "capture": str(path.resolve()),
        "capture_sha256": sha256(path),
        "checks": checks,
        "board_rate_hz": float(data["analysis"]["board_rate_hz"]),
        "maximum_interval_us": int(data["analysis"]["board_interval_max_us"]),
        "final_status": status,
    }


def save_manifest(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), "utf-8")
    temporary.replace(path)


def unused_retry_output(root: Path, base_name: str) -> Path:
    """Return a new sibling path without reusing a failed child directory."""
    retry = 1
    while (root / f"{base_name}_retry{retry}").exists():
        retry += 1
    return root / f"{base_name}_retry{retry}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator-confirmed", action="store_true")
    parser.add_argument("--round-name", required=True)
    parser.add_argument("--depth-mm", required=True, type=float)
    parser.add_argument("--points", nargs="+", type=int, default=list(range(1, 18)))
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument('--ch1-selector', type=int, choices=range(4), default=2)
    args = parser.parse_args(argv)
    if not args.operator_confirmed:
        parser.error("calibrated fixture and emergency-stop access must be confirmed")
    if not 0.0 <= args.depth_mm <= 0.8:
        parser.error("depth must be within 0..0.8 mm")
    if len(set(args.points)) != len(args.points) or any(point not in range(1, 18) for point in args.points):
        parser.error("points must be unique calibrated indices in 1..17")

    root = (args.outputs / args.round_name).resolve()
    manifest_path = root / "manifest.json"
    if args.resume:
        if not manifest_path.exists():
            parser.error("resume requested but the round manifest does not exist")
        manifest = json.loads(manifest_path.read_text("utf-8"))
        if (
            manifest.get("schema") != "f45_visible_depth_round_v1"
            or manifest.get("round_name") != args.round_name
            or abs(float(manifest.get("depth_mm", -1.0)) - args.depth_mm) > 1e-9
            or manifest.get("planned_points") != list(args.points)
            or manifest.get('ch1_selector', 2) != args.ch1_selector
        ):
            parser.error("resume arguments do not exactly match the saved round")
        manifest.pop("failure", None)
        manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
        manifest["complete"] = False
    else:
        if root.exists():
            parser.error("round output directory must be new")
        root.mkdir(parents=True)
        manifest = {
            "schema": "f45_visible_depth_round_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "round_name": args.round_name,
            "depth_mm": float(args.depth_mm),
            "ch1_selector": args.ch1_selector,
            "planned_points": list(args.points),
            "completed_points": [],
            "captures": {},
            "complete": False,
            "safety": {
                "maximum_indentation_mm": 0.8,
                "spindle_policy": "M5 before every movement; spindle never started",
                "failure_policy": "stop batch immediately; never auto-release E-stop",
            },
        }
    save_manifest(manifest_path, manifest)
    depth_token = f"{args.depth_mm:.2f}".replace(".", "p")
    script = Path(__file__).with_name("commission_f45_visible.py")
    for ordinal, point in enumerate(args.points, start=1):
        if point in manifest["completed_points"]:
            continue
        base_name = f"d{depth_token}_p{point:02d}"
        output = root / base_name
        if (output / "capture.json").exists():
            try:
                verification = verify_capture(output / "capture.json", point, args.depth_mm, args.ch1_selector)
            except Exception:
                output = unused_retry_output(root, base_name)
            else:
                verification["recovered_after_ui_process_exit"] = True
                manifest["completed_points"].append(point)
                manifest["captures"][str(point)] = verification
                save_manifest(manifest_path, manifest)
                print(
                    json.dumps(
                        {"state": "recovered_verified_capture", "point": point},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
        elif output.exists():
            # The child creates its output directory before opening the optical
            # port.  A transient serial failure can therefore leave an empty or
            # partial directory that must never be reused on resume.
            output = unused_retry_output(root, base_name)
        print(
            json.dumps(
                {
                    "state": "starting_contact",
                    "ordinal": ordinal,
                    "total": len(args.points),
                    "point": point,
                    "depth_mm": args.depth_mm,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--operator-confirmed",
                "--protocol",
                "fullmap45",
                "--point-index",
                str(point),
                "--depth-mm",
                str(args.depth_mm),
                "--hold-seconds",
                "1.0",
                '--ch1-selector',
                str(args.ch1_selector),
                "--output",
                str(output),
            ],
            check=False,
        )
        try:
            verification = verify_capture(output / "capture.json", point, args.depth_mm, args.ch1_selector)
        except Exception as exc:
            manifest["failure"] = {
                "point": point,
                "child_exit_code": completed.returncode,
                "verification_error": f"{type(exc).__name__}: {exc}",
            }
            save_manifest(manifest_path, manifest)
            return int(completed.returncode or 2)
        verification["ui_child_exit_code"] = int(completed.returncode)
        verification["ui_process_clean_exit"] = completed.returncode == 0
        manifest["completed_points"].append(point)
        manifest["captures"][str(point)] = verification
        save_manifest(manifest_path, manifest)
        print(
            json.dumps(
                {
                    "state": "verified_contact",
                    "ordinal": ordinal,
                    "total": len(args.points),
                    "point": point,
                    "board_rate_hz": verification["board_rate_hz"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    manifest["complete"] = True
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    save_manifest(manifest_path, manifest)
    print(json.dumps({"state": "round_complete", "manifest": str(manifest_path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
