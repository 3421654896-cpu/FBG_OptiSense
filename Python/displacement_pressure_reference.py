"""Current-version pressure proxy from calibrated CNC displacement, not force."""
import math


def from_clearance(clearance_mm, *, full_scale_mm=0.8):
    clearance, scale = float(clearance_mm), float(full_scale_mm)
    if not math.isfinite(clearance) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Finite clearance and positive displacement scale required")
    depth = max(0.0, -clearance)
    return {"source": "cnc_displacement_proxy", "indentation_mm": depth,
            "clearance_mm": clearance, "relative_pressure_percent": 100 * depth / scale,
            "full_scale_indentation_mm": scale, "over_depth_limit": depth > scale,
            "force_n": None, "pressure_pa": None,
            "independent_pressure_measurement": False,
            "relation": "linear_displacement_index_not_calibrated_material_stiffness"}


def from_machine(target, status):
    if not status.get("connected"):
        return None
    contact_z = target.get("contact_machine_z")
    if contact_z is None:
        offset = target.get("calibration_machine_minus_work")
        if offset is None or len(offset) != 3 or target.get("contact_z") is None:
            return None
        contact_z = float(target["contact_z"]) + float(offset[2])
    if status.get("machine_z") is None:
        return None
    result = from_clearance(float(status["machine_z"]) - float(contact_z))
    result.update(contact_machine_z=float(contact_z), machine_z=float(status["machine_z"]),
                  timing_basis="latest_machine_telemetry_not_synchronized_force",
                  moving=bool(status.get("moving")))
    return result
