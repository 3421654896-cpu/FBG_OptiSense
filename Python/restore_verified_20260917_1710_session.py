"""Restore the independently backed-up 2026-09-17 17:10 temporary session."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import time

from temporary_test_mode import assess_five_point_shape, validate_plan_payload


HERE = Path(__file__).resolve().parent
SESSION = HERE / "outputs" / "temporary_test" / "last_45_session.json"
BACKUP = Path(
    r"C:\Users\xiechengxin\Desktop\chat光栅解调工作室\光栅触觉_20260917_1710_九峰全通过版_备用\temporary_test_outputs"
)


def main() -> int:
    capture_path = BACKUP / "capture_1789636184742347300.json"
    reference_path = BACKUP / "reference_line_1789636121834129300.json"
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if not capture.get("complete") or not reference.get("complete"):
        raise ValueError("17:10备用采集或17:09参考线不完整")
    plan = validate_plan_payload(capture["plan"])
    recorded = reference.get("rows", [])
    if len(plan["rows"]) != 45 or len(recorded) != 45:
        raise ValueError("备用路线不是完整45点")
    for selected, row in zip(plan["rows"], recorded):
        if (int(row["index"]) != int(selected["index"])
                or list(row["dac_codes"]) != list(selected["codes"])
                or not row.get("stable") or row.get("ch1_saturated")):
            raise ValueError("备用参考线与17:10路线不一致或存在不稳定点")
    values = [float(row["ch1_adc_code"]) for row in recorded]
    wavelengths = [float(row["measured_nm"]) for row in plan["rows"]]
    checks = []
    for group in range(9):
        start = group * 5
        check = assess_five_point_shape(
            wavelengths[start:start + 5], values[start:start + 5]
        )
        check["group"] = group + 1
        checks.append(check)
    if not all(check["passed"] for check in checks):
        raise ValueError("备用参考线未通过当前九峰严格拟合复核")
    plan["live_reference_fit_checks"] = checks
    plan["selection_fit_checks"] = checks
    plan["selection_validation_required"] = False
    plan["validated_reference_values"] = values
    plan["restored_verified_session"] = {
        "capture": str(capture_path),
        "reference": str(reference_path),
        "verified_at": "2026-09-17 17:10:18+08:00",
        "post_warmup_frames_nine_of_nine": "512/512",
    }
    payload = {
        "schema": "temporary_test_last_session_v1",
        "saved_s": time.time(),
        "ready": True,
        "plan": plan,
        "reference_values": values,
        "reference_wavelengths_nm": wavelengths,
        "feedback_selector": 2,
        "digital_display_gain": 1.0,
        "reference_display_gain": 1.0,
        "first_delay_us": 350,
        "boundary_extra_us": 1800,
        "show_dense_selection": True,
        "show_reference": True,
        "use_point_waits": False,
        "point_delays_us": None,
    }
    SESSION.parent.mkdir(parents=True, exist_ok=True)
    if SESSION.exists():
        destination = SESSION.with_name(
            f"last_45_session_before_1710_restore_{time.time_ns()}.json"
        )
        shutil.copy2(SESSION, destination)
    temporary = SESSION.with_suffix(".restore.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(SESSION)
    # Re-read through the same structural gate used by the GUI.
    restored = json.loads(SESSION.read_text(encoding="utf-8"))
    validate_plan_payload(restored["plan"])
    print(json.dumps({
        "restored": True,
        "session": str(SESSION.resolve()),
        "route_indices": [[row["index"] for row in plan["rows"][i:i + 5]]
                          for i in range(0, 45, 5)],
        "strict_fit_groups": sum(check["passed"] for check in checks),
        "feedback_selector": 2,
        "first_delay_us": 350,
        "boundary_extra_us": 1800,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
