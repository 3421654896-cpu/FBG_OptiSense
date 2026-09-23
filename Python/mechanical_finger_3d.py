"""Interactive 3-D mechanical fingertip stress and CNC-position view.

The widget consumes peak centres fitted from the *current* ADC spectrum and
accepts read-only Mach3 status snapshots.  It never issues machine movement
commands and it never averages live stress frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

# Locate the Qt Windows platform plugin before pyqtgraph imports Qt.  The local
# project virtual environment is stored below a non-ASCII workspace path, so
# relying on Qt's implicit plugin search is not reliable on this workstation.
qt_platform_plugins = (
    Path(sys.prefix)
    / "Lib"
    / "site-packages"
    / "PyQt5"
    / "Qt5"
    / "plugins"
    / "platforms"
)
if qt_platform_plugins.exists():
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(qt_platform_plugins))

import numpy as np
import pyqtgraph.opengl as gl
import yaml
from PyQt5 import QtCore, QtGui, QtWidgets
from attach_contact_area_truth import attach_contact_area_truth
from physical_pressure_validation import (
    ContactAreaTruthSet,
    PhysicalAcceptanceSession,
    PhysicalReferenceTrace,
    evaluate_physical_acceptance_session,
    load_contact_area_truth,
    load_physical_acceptance_session,
    load_physical_reference,
)
from responsive_layout import FlowLayout, compact_field

HERE = Path(__file__).resolve().parent
DEFAULT_LAYOUT_PATH = HERE / "finger_sensor_layout.yaml"
DEFAULT_MESH_PATH = HERE / "assets" / "thumb_real.stl"
DEFAULT_PRESS_TOOL_PATH = HERE / "assets" / "press_tool_v2.stl"
MAX_PRESS_DEPTH_MM = 0.80
PRESS_FACE_DIAMETER_MM = 8.0
PRESS_FACE_RADIUS_MM = PRESS_FACE_DIAMETER_MM / 2.0
PRESS_NOMINAL_AREA_MM2 = math.pi * PRESS_FACE_RADIUS_MM ** 2
PRESS_NOMINAL_AREA_LABEL = (
    f"压头名义接触面积：{PRESS_NOMINAL_AREA_MM2:.2f} mm²（直径8 mm圆面；非实际接触面积实测）"
)
DEFAULT_TOUCH_CALIBRATION_PATH = HERE / "t31_touch_calibration.json"


def relative_effort_label(clearance_mm):
    """Display a displacement proxy, never calibrated force or optical inference."""
    if not math.isfinite(clearance_mm):
        return "相对力度：未知（位移无效）"
    depth = max(0.0, -clearance_mm)
    if depth > MAX_PRESS_DEPTH_MM:
        return f"相对力度：超出标定范围（压入{depth:.3f} mm，超过0.80 mm上限）"
    effort = depth / MAX_PRESS_DEPTH_MM * 100.0
    return (
        f"相对力度 {effort:.1f}%（机床位移参考：压入{depth:.3f} mm；"
        "0.80 mm=100%，非实测力、非光谱预测）"
    )


SENSOR_COUNT = 9
# Reconstructed from the yellow recess outline in the physical/top-view image.
# The small inset is deliberate: the silicone must remain inside the recessed
# pocket and leave the surrounding printed rim visible at every viewing angle.
SILICONE_X_RADIUS = 0.53
SILICONE_Y_CENTER = 0.32
SILICONE_Y_RADIUS = 0.50
SILICONE_TIP_FILL = 0.055
# ``thumb_real.stl`` is a true-size SolidWorks export.  Normalized X and Y do
# not represent the same physical distance, so every two-point calibration is
# first expressed in this CAD millimetre plane.  This preserves the model's
# real aspect ratio when only the two silicone-root boundary points are known.
MODEL_X_HALF_SPAN_MM = 17.832737
MODEL_Y_HALF_SPAN_MM = 24.3053895

# Named, repeatable landmarks on the rendered silicone's lower boundary.  The
# horizontal position follows the previously operator-confirmed right-root
# landmark; Y is projected onto the current superellipse instead of leaving
# the marker suspended on the shell below the silicone.
SILICONE_LOWER_LANDMARK_X = 0.34
SILICONE_LOWER_LANDMARK_Y = SILICONE_Y_CENTER - SILICONE_Y_RADIUS * (
    1.0 - (SILICONE_LOWER_LANDMARK_X / SILICONE_X_RADIUS) ** 4
) ** 0.25

# Light 3-D workstation palette.  The viewport is deliberately a cool
# off-white rather than pure white so the pale physical parts retain a clear
# silhouette without looking disconnected from the app's light UI.
VIEW_BACKGROUND_RGBA = (241, 244, 248, 255)
BODY_BASE_RGBA = (0.56, 0.68, 0.81, 1.0)
SILICONE_BASE_RGBA = (0.43, 0.72, 0.87, 0.82)
SENSOR_IDLE_RGBA = (0.84, 0.20, 0.25, 1.0)
SENSOR_HALO_RGBA = (0.92, 0.31, 0.35, 0.28)
# The FBG coordinates remain part of the signal-processing and stress heat-map
# model, but the physical grating marks are embedded beneath the silicone and
# should not be drawn as exposed red bars/numbers in the 3-D scene.
RENDER_SENSOR_MARKERS = False
TOOL_READY_RGBA = (0.43, 0.55, 0.82, 0.96)
TOOL_PREVIEW_RGBA = (0.49, 0.61, 0.83, 0.90)
TOOL_MOVING_RGBA = (0.96, 0.62, 0.27, 0.96)
TOOL_ESTOP_RGBA = (0.90, 0.31, 0.35, 0.95)


@dataclass(frozen=True)
class FingerSensor:
    sensor_id: int
    peak_index: int
    name: str
    x: float
    y: float
    orientation_deg: float = 0.0


@dataclass(frozen=True)
class FingerLayout:
    channel: int
    baseline_frames: int
    activation_pm: float
    full_scale_pm: float
    sensors: tuple[FingerSensor, ...]


@dataclass(frozen=True)
class MachineTouchSample:
    """One verified tool-contact coordinate associated with an FBG marker."""

    sensor_id: int | None
    work_x: float
    work_y: float
    work_z: float
    machine_x: float | None = None
    machine_y: float | None = None
    machine_z: float | None = None
    x_norm: float | None = None
    y_norm: float | None = None
    label: str = ""
    surface_kind: str = "silicone"


@dataclass(frozen=True)
class MachineToolPose:
    """Tool-tip pose in the finger's normalized surface coordinate system."""

    x_norm: float
    y_norm: float
    clearance_mm: float
    nearest_sensor_id: int
    mapping_kind: str = "line"
    calibration_label: str = ""
    surface_kind: str = "silicone"


def load_finger_layout(path: Path | str = DEFAULT_LAYOUT_PATH) -> FingerLayout:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    sensors = []
    for item in payload.get("sensors", []):
        sensors.append(
            FingerSensor(
                sensor_id=int(item["id"]),
                peak_index=int(item["peak_index"]),
                name=str(item.get("name", f"光栅{item['id']}")),
                x=float(item["x"]),
                y=float(item["y"]),
                orientation_deg=float(item.get("orientation_deg", 0.0)),
            )
        )
    if len(sensors) != SENSOR_COUNT:
        raise ValueError(f"机械手指必须配置{SENSOR_COUNT}个光栅位置")
    if sorted(sensor.sensor_id for sensor in sensors) != list(range(1, 10)):
        raise ValueError("光栅编号必须为1～9且不能重复")
    if sorted(sensor.peak_index for sensor in sensors) != list(range(9)):
        raise ValueError("peak_index必须完整覆盖0～8且不能重复")
    if any(abs(sensor.x) > 1.0 or abs(sensor.y) > 1.0 for sensor in sensors):
        raise ValueError("光栅归一化位置x/y必须位于-1～1")
    channel = int(payload.get("channel", 1))
    if channel not in (0, 1, 2):
        raise ValueError("机械手指光谱通道只能是CH0、CH1或CH2")
    return FingerLayout(
        channel=channel,
        baseline_frames=max(3, int(payload.get("baseline_frames", 20))),
        activation_pm=max(0.0, float(payload.get("activation_pm", 5.0))),
        full_scale_pm=max(1.0, float(payload.get("full_scale_pm", 100.0))),
        sensors=tuple(sorted(sensors, key=lambda sensor: sensor.sensor_id)),
    )


def load_machine_touch_samples(
    path: Path | str = DEFAULT_TOUCH_CALIBRATION_PATH,
) -> tuple[MachineTouchSample, ...]:
    """Load contact points measured with the T31-assisted CNC procedure."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = []
    for item in payload.get("samples", []):
        if not bool(item.get("contact")) or not bool(item.get("active", True)):
            continue
        sensor_value = item.get("target_index")
        x_norm_value = item.get("model_x_norm")
        y_norm_value = item.get("model_y_norm")
        if sensor_value is None and (x_norm_value is None or y_norm_value is None):
            raise ValueError("机床接触标定必须关联光栅编号或明确的模型坐标")
        surface_kind = str(item.get("surface_kind", "silicone")).strip().lower()
        if surface_kind not in {"silicone", "shell", "shell_top"}:
            raise ValueError(
                "接触标定surface_kind只能是silicone、shell或shell_top"
            )
        sample = MachineTouchSample(
            sensor_id=int(sensor_value) if sensor_value is not None else None,
            work_x=float(item["work_x"]),
            work_y=float(item["work_y"]),
            work_z=float(item["work_z"]),
            machine_x=(
                float(item["machine_x"]) if item.get("machine_x") is not None else None
            ),
            machine_y=(
                float(item["machine_y"]) if item.get("machine_y") is not None else None
            ),
            machine_z=(
                float(item["machine_z"]) if item.get("machine_z") is not None else None
            ),
            x_norm=float(x_norm_value) if x_norm_value is not None else None,
            y_norm=float(y_norm_value) if y_norm_value is not None else None,
            label=str(item.get("label", "")),
            surface_kind=surface_kind,
        )
        if not all(
            math.isfinite(value)
            for value in (sample.work_x, sample.work_y, sample.work_z)
        ):
            raise ValueError("机床接触标定含有无效坐标")
        machine_values = (sample.machine_x, sample.machine_y, sample.machine_z)
        if any(value is not None for value in machine_values) and not all(
            value is not None and math.isfinite(value) for value in machine_values
        ):
            raise ValueError("机床接触标定的机械绝对坐标必须完整且有效")
        if sample.x_norm is not None and (
            not math.isfinite(sample.x_norm)
            or not math.isfinite(sample.y_norm)
            or abs(sample.x_norm) > 1.0
            or abs(sample.y_norm) > 1.0
        ):
            raise ValueError("机床接触标定的模型坐标必须位于-1～1")
        samples.append(sample)
    return tuple(samples)


def machine_coordinates(
    status: Mapping[str, object], prefix: str = "work"
) -> np.ndarray | None:
    """Extract one XYZ coordinate triplet from a Mach3 status response."""

    try:
        values = np.asarray(
            [status[f"{prefix}_{axis}"] for axis in ("x", "y", "z")],
            dtype=float,
        )
    except (KeyError, TypeError, ValueError):
        return None
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        return None
    return values


def machine_pose_from_touch_samples(
    status: Mapping[str, object],
    samples: Sequence[MachineTouchSample],
    sensors: Sequence[FingerSensor],
    *,
    lateral_tolerance_mm: float = 0.75,
    longitudinal_margin_mm: float = 1.0,
) -> MachineToolPose | None:
    """Map the live Mach3 position into the calibrated finger model."""

    work = machine_coordinates(status, "work")
    machine = machine_coordinates(status, "machine")
    sensor_by_id = {sensor.sensor_id: sensor for sensor in sensors}
    usable = []
    for sample in samples:
        if sample.x_norm is not None and sample.y_norm is not None:
            x_norm, y_norm = float(sample.x_norm), float(sample.y_norm)
        elif sample.sensor_id in sensor_by_id:
            sensor = sensor_by_id[sample.sensor_id]
            x_norm, y_norm = float(sensor.x), float(sensor.y)
        else:
            continue
        usable.append((sample, x_norm, y_norm))
    if work is None or not usable:
        return None

    # Prefer machine-absolute coordinates whenever the calibration and live
    # frame both provide them.  Work offsets can be re-zeroed in Mach3 and are
    # therefore unsuitable as the long-lived source of truth for rendering.
    use_machine = machine is not None and all(
        sample.machine_x is not None
        and sample.machine_y is not None
        and sample.machine_z is not None
        for sample, _x, _y in usable
    )
    current = machine if use_machine else work

    def sample_xyz(sample: MachineTouchSample) -> np.ndarray:
        if use_machine:
            return np.asarray(
                (sample.machine_x, sample.machine_y, sample.machine_z), dtype=float
            )
        return np.asarray((sample.work_x, sample.work_y, sample.work_z), dtype=float)

    def nearest_sensor_id(x_norm, y_norm):
        return min(
            sensors,
            key=lambda sensor: (sensor.x - x_norm) ** 2 + (sensor.y - y_norm) ** 2,
        ).sensor_id

    # At a measured calibration coordinate, preserve the operator-confirmed
    # model point exactly.  A least-squares surface fit is intentionally used
    # only between samples; otherwise its small residual could make the tool
    # look visibly offset from a red FBG marker at the very point used to
    # calibrate it.
    sample_distances = np.asarray(
        [
            float(np.linalg.norm(current[:2] - sample_xyz(sample)[:2]))
            for sample, _x, _y in usable
        ],
        dtype=float,
    )
    exact_index = int(np.argmin(sample_distances))
    if sample_distances[exact_index] <= 0.05:
        sample, x_norm, y_norm = usable[exact_index]
        return MachineToolPose(
            x_norm=x_norm,
            y_norm=y_norm,
            clearance_mm=float(current[2] - sample_xyz(sample)[2]),
            nearest_sensor_id=nearest_sensor_id(x_norm, y_norm),
            mapping_kind="anchor",
            calibration_label=sample.label,
            surface_kind=sample.surface_kind,
        )

    if len(usable) == 1:
        sample, x_norm, y_norm = usable[0]
        if float(np.linalg.norm(current[:2] - sample_xyz(sample)[:2])) > float(lateral_tolerance_mm):
            return None
        return MachineToolPose(
            x_norm=x_norm,
            y_norm=y_norm,
            clearance_mm=float(current[2] - sample_xyz(sample)[2]),
            nearest_sensor_id=nearest_sensor_id(x_norm, y_norm),
            mapping_kind="anchor",
            calibration_label=sample.label,
            surface_kind=sample.surface_kind,
        )

    # Two explicit silicone-boundary landmarks determine translation,
    # in-plane rotation and uniform physical scale.  Work in the CAD-mm plane
    # rather than directly in normalized coordinates: the real thumb model's
    # Y half-span is 1.36 times its X half-span.
    explicit_usable = [
        entry
        for entry in usable
        if entry[0].sensor_id is None
        and entry[0].x_norm is not None
        and entry[0].y_norm is not None
    ]
    similarity = _two_point_model_similarity(
        explicit_usable,
        lambda entry: sample_xyz(entry[0])[:2],
    )
    if similarity is not None:
        model_to_current, intercept = similarity
        inverse = np.linalg.inv(model_to_current)
        predicted = (current[:2] - intercept) @ inverse
        if not (-0.80 <= predicted[0] <= 0.80 and -0.32 <= predicted[1] <= 0.95):
            return None
        positions = np.asarray(
            [sample_xyz(entry[0])[:2] for entry in explicit_usable], dtype=float
        )
        nearest_index = int(
            np.argmin(np.sum((positions - current[:2]) ** 2, axis=1))
        )
        nearest_sample = explicit_usable[nearest_index][0]
        contact_z = float(
            np.mean([sample_xyz(entry[0])[2] for entry in explicit_usable])
        )
        return MachineToolPose(
            x_norm=float(predicted[0]),
            y_norm=float(predicted[1]),
            clearance_mm=float(current[2] - contact_z),
            nearest_sensor_id=nearest_sensor_id(*predicted),
            mapping_kind="similarity",
            calibration_label="硅胶下边界双点标定",
            surface_kind=nearest_sample.surface_kind,
        )

    # Once at least three non-collinear FBG locations are available, fit the
    # sensing area from those red markers alone.  A shell reference such as the
    # yellow XY origin remains an exact local anchor, but its hand-annotated
    # model coordinate must not distort interpolation across the FBG array.
    all_usable = list(usable)
    sensor_usable = [entry for entry in usable if entry[0].sensor_id is not None]
    if len(sensor_usable) >= 3:
        sensor_machine_xy = np.asarray(
            [sample_xyz(entry[0])[:2] for entry in sensor_usable],
            dtype=float,
        )
        sensor_design = np.column_stack(
            (sensor_machine_xy, np.ones(len(sensor_machine_xy)))
        )
        if np.linalg.matrix_rank(sensor_design) >= 3:
            usable = sensor_usable

    machine_xy = np.asarray(
        [sample_xyz(sample)[:2] for sample, _x, _y in usable],
        dtype=float,
    )
    model_xy = np.asarray([[_x, _y] for _sample, _x, _y in usable], dtype=float)
    design = np.column_stack((machine_xy, np.ones(len(machine_xy))))
    if len(usable) >= 3 and np.linalg.matrix_rank(design) >= 3:
        coefficients = np.linalg.lstsq(design, model_xy, rcond=None)[0]
        current_design = np.asarray((current[0], current[1], 1.0))
        predicted = current_design @ coefficients

        # The yellow shell reference lies just below the silicone grid.  Blend
        # its measured model coordinate into the sensor-plane solution so the
        # rendered tool moves continuously from that reference into the FBG
        # area instead of jumping to a fixed preview pose.
        for anchor, anchor_x, anchor_y in all_usable:
            if anchor.sensor_id is not None:
                continue
            anchor_position = sample_xyz(anchor)
            distance = float(np.linalg.norm(current[:2] - anchor_position[:2]))
            blend_radius_mm = 4.5
            if distance >= blend_radius_mm:
                continue
            anchor_design = np.asarray(
                (anchor_position[0], anchor_position[1], 1.0), dtype=float
            )
            correction = np.asarray((anchor_x, anchor_y)) - anchor_design @ coefficients
            normalized_distance = distance / blend_radius_mm
            blend = (1.0 - normalized_distance * normalized_distance) ** 2
            predicted = predicted + blend * correction

        # Reject only genuinely remote extrapolation.  The previous machine-
        # rectangle guard excluded the valid route between the yellow anchor
        # and the silicone pad, which made the 3-D tool appear to lose track.
        if not (-0.80 <= predicted[0] <= 0.80 and -0.32 <= predicted[1] <= 0.95):
            return None
        contact_plane = np.linalg.lstsq(
            design,
            np.asarray(
                [sample_xyz(sample)[2] for sample, _x, _y in usable], dtype=float
            ),
            rcond=None,
        )[0]
        contact_z = float(current_design @ contact_plane)
        nearest_index = int(
            np.argmin(np.sum((machine_xy - current[:2]) ** 2, axis=1))
        )
        nearest_sample = usable[nearest_index][0]
        all_positions = np.asarray(
            [sample_xyz(sample)[:2] for sample, _x, _y in all_usable], dtype=float
        )
        surface_index = int(
            np.argmin(np.sum((all_positions - current[:2]) ** 2, axis=1))
        )
        surface_sample = all_usable[surface_index][0]
        surface_distance = float(
            np.linalg.norm(all_positions[surface_index] - current[:2])
        )
        return MachineToolPose(
            x_norm=float(predicted[0]),
            y_norm=float(predicted[1]),
            clearance_mm=float(current[2] - contact_z),
            nearest_sensor_id=nearest_sensor_id(*predicted),
            mapping_kind="surface",
            calibration_label=nearest_sample.label,
            surface_kind=(
                surface_sample.surface_kind
                if surface_distance <= 0.75
                else "silicone"
            ),
        )

    usable.sort(key=lambda entry: entry[0].work_y)
    reference_x = float(np.median([sample_xyz(entry[0])[0] for entry in usable]))
    if abs(float(current[0]) - reference_x) > float(lateral_tolerance_mm):
        return None
    machine_y = np.asarray([sample_xyz(entry[0])[1] for entry in usable], dtype=float)
    if not (
        machine_y[0] - longitudinal_margin_mm
        <= current[1]
        <= machine_y[-1] + longitudinal_margin_mm
    ):
        return None
    sensor_x = np.asarray([entry[1] for entry in usable], dtype=float)
    sensor_y = np.asarray([entry[2] for entry in usable], dtype=float)
    contact_z = np.asarray([sample_xyz(entry[0])[2] for entry in usable], dtype=float)
    nearest = usable[int(np.argmin(np.abs(machine_y - current[1])))]
    predicted_x = float(np.interp(current[1], machine_y, sensor_x))
    predicted_y = float(np.interp(current[1], machine_y, sensor_y))
    return MachineToolPose(
        x_norm=predicted_x,
        y_norm=predicted_y,
        clearance_mm=float(current[2] - np.interp(current[1], machine_y, contact_z)),
        nearest_sensor_id=nearest_sensor_id(predicted_x, predicted_y),
        mapping_kind="line",
        calibration_label=nearest[0].label,
        surface_kind=nearest[0].surface_kind,
    )


def silicone_contains_point(x_norm: float, y_norm: float) -> bool:
    """Return whether a target centre lies inside the rendered silicone pad."""

    x = float(x_norm)
    y = float(y_norm)
    if not math.isfinite(x) or not math.isfinite(y):
        return False
    rho = (abs(x) / SILICONE_X_RADIUS) ** 4 + (
        abs(y - SILICONE_Y_CENTER) / SILICONE_Y_RADIUS
    ) ** 4
    return rho <= 1.0 + 1e-9


def _two_point_model_similarity(usable, coordinate_getter):
    """Return ``normalized_model_xy @ A + b -> measured_xy`` for two anchors.

    A two-point similarity is unique once the CAD plane supplies the true X/Y
    aspect ratio.  Returning the composed normalized-to-measured matrix keeps
    the rest of the viewer independent of STL units.
    """

    if len(usable) != 2:
        return None
    normalized = np.asarray(
        [[float(entry[1]), float(entry[2])] for entry in usable], dtype=float
    )
    measured = np.asarray([coordinate_getter(entry) for entry in usable], dtype=float)
    if normalized.shape != (2, 2) or measured.shape != (2, 2):
        return None
    half_spans = np.asarray(
        (MODEL_X_HALF_SPAN_MM, MODEL_Y_HALF_SPAN_MM), dtype=float
    )
    model_mm = normalized * half_spans
    model_delta = model_mm[1] - model_mm[0]
    measured_delta = measured[1] - measured[0]
    model_length = float(np.linalg.norm(model_delta))
    measured_length = float(np.linalg.norm(measured_delta))
    if model_length <= 1e-9 or measured_length <= 1e-9:
        return None
    scale = measured_length / model_length
    angle = math.atan2(measured_delta[1], measured_delta[0]) - math.atan2(
        model_delta[1], model_delta[0]
    )
    cosine = math.cos(angle)
    sine = math.sin(angle)
    metric_to_measured = scale * np.asarray(
        ((cosine, sine), (-sine, cosine)), dtype=float
    )
    normalized_to_measured = np.diag(half_spans) @ metric_to_measured
    intercept = measured[0] - normalized[0] @ normalized_to_measured
    if abs(float(np.linalg.det(normalized_to_measured))) <= 1e-9:
        return None
    return normalized_to_measured, intercept


def machine_target_from_model_point(
    x_norm: float,
    y_norm: float,
    samples: Sequence[MachineTouchSample],
    sensors: Sequence[FingerSensor],
) -> tuple[float, float, float] | None:
    """Predict a work-coordinate contact point for one silicone target."""

    x = float(x_norm)
    y = float(y_norm)
    if not silicone_contains_point(x, y):
        return None
    sensor_by_id = {sensor.sensor_id: sensor for sensor in sensors}
    usable = []
    for sample in samples:
        if sample.x_norm is not None and sample.y_norm is not None:
            sample_x, sample_y = float(sample.x_norm), float(sample.y_norm)
        else:
            sensor = sensor_by_id.get(sample.sensor_id)
            if sensor is None:
                continue
            sample_x, sample_y = float(sensor.x), float(sensor.y)
        usable.append((sample, sample_x, sample_y))
    if not usable:
        return None

    # Preserve every operator-confirmed anchor exactly.
    distances = np.asarray(
        [math.hypot(x - sx, y - sy) for _sample, sx, sy in usable], dtype=float
    )
    nearest_index = int(np.argmin(distances))
    if distances[nearest_index] <= 0.015:
        sample = usable[nearest_index][0]
        return float(sample.work_x), float(sample.work_y), float(sample.work_z)

    sensor_usable = [entry for entry in usable if entry[0].sensor_id in sensor_by_id]
    if len(sensor_usable) >= 3:
        sensor_design = np.asarray(
            [[sx, sy, 1.0] for _sample, sx, sy in sensor_usable], dtype=float
        )
        if np.linalg.matrix_rank(sensor_design) >= 3:
            usable = sensor_usable

    explicit_usable = [
        entry
        for entry in usable
        if entry[0].sensor_id is None
        and entry[0].x_norm is not None
        and entry[0].y_norm is not None
    ]
    similarity = _two_point_model_similarity(
        explicit_usable,
        lambda entry: (entry[0].work_x, entry[0].work_y),
    )
    if similarity is not None:
        model_to_work, intercept = similarity
        target_xy = np.asarray((x, y), dtype=float) @ model_to_work + intercept
        target_z = float(np.mean([entry[0].work_z for entry in explicit_usable]))
        if not np.all(np.isfinite(target_xy)) or not math.isfinite(target_z):
            return None
        return float(target_xy[0]), float(target_xy[1]), target_z

    if len(usable) < 3:
        return None

    # Preserve measured red-marker coordinates when the operator clicks one of
    # them; interpolate or conservatively extrapolate only between/around them.
    model_design = np.asarray(
        [[sx, sy, 1.0] for _sample, sx, sy in usable], dtype=float
    )
    if np.linalg.matrix_rank(model_design) < 3:
        return None
    machine_xy = np.asarray(
        [[sample.work_x, sample.work_y] for sample, _sx, _sy in usable],
        dtype=float,
    )
    contact_z = np.asarray(
        [sample.work_z for sample, _sx, _sy in usable], dtype=float
    )
    coefficients = np.linalg.lstsq(model_design, machine_xy, rcond=None)[0]
    z_coefficients = np.linalg.lstsq(model_design, contact_z, rcond=None)[0]
    model_point = np.asarray((x, y, 1.0), dtype=float)
    target_xy = model_point @ coefficients
    target_z = float(model_point @ z_coefficients)
    if not np.all(np.isfinite(target_xy)) or not math.isfinite(target_z):
        return None
    return float(target_xy[0]), float(target_xy[1]), target_z


def generate_silicone_coverage_plan(
    samples: Sequence[MachineTouchSample],
    sensors: Sequence[FingerSensor],
    *,
    spacing_mm: float = 2.0,
) -> tuple[dict[str, float | int], ...]:
    """Generate a serpentine hex-grid whose 8 mm face covers the silicone.

    The default two-millimetre pitch gives dense, repeatable training coverage
    with seventy-five-percent overlap between neighbouring press faces.  Points
    are generated in calibrated machine
    millimetres and mapped back to the silicone model.  A point is retained
    only when the *entire* 8 mm circular face lies inside the silicone boundary;
    checking the centre alone can put half of the tool on the printed shell at
    the narrow root and tip.
    """

    spacing = float(spacing_mm)
    if not math.isfinite(spacing) or not 2.0 <= spacing <= 8.0:
        raise ValueError("覆盖采集点距必须位于2～8 mm")
    sensor_by_id = {sensor.sensor_id: sensor for sensor in sensors}
    usable = []
    for sample in samples:
        if sample.x_norm is not None and sample.y_norm is not None:
            usable.append((sample, float(sample.x_norm), float(sample.y_norm)))
        elif sample.sensor_id in sensor_by_id:
            sensor = sensor_by_id[sample.sensor_id]
            usable.append((sample, float(sensor.x), float(sensor.y)))

    sensor_usable = [entry for entry in usable if entry[0].sensor_id in sensor_by_id]
    model_design = np.asarray(
        [[x_norm, y_norm, 1.0] for _sample, x_norm, y_norm in sensor_usable],
        dtype=float,
    )
    if len(sensor_usable) >= 3 and np.linalg.matrix_rank(model_design) >= 3:
        machine_xy = np.asarray(
            [[sample.work_x, sample.work_y] for sample, _x, _y in sensor_usable],
            dtype=float,
        )
        coefficients = np.linalg.lstsq(model_design, machine_xy, rcond=None)[0]
        linear = coefficients[:2, :]
        intercept = coefficients[2, :]
    else:
        explicit_usable = [
            entry
            for entry in usable
            if entry[0].sensor_id is None
            and entry[0].x_norm is not None
            and entry[0].y_norm is not None
        ]
        similarity = _two_point_model_similarity(
            explicit_usable,
            lambda entry: (entry[0].work_x, entry[0].work_y),
        )
        if similarity is None:
            raise ValueError("需要两个硅胶边界点或三个非共线光栅点才能生成覆盖方案")
        linear, intercept = similarity

    if abs(float(np.linalg.det(linear))) <= 1e-9:
        raise ValueError("机床与手指模型的二维标定矩阵不可逆")
    inverse = np.linalg.inv(linear)

    # The physical face is circular in machine millimetres.  Under the affine
    # model calibration it becomes an ellipse, so test the footprint in the
    # machine plane before accepting a centre.  The silicone is convex and a
    # dense perimeter test is conservative enough at CNC positioning scale.
    footprint_angles = np.linspace(0.0, 2.0 * math.pi, 721, endpoint=False)
    footprint_offsets_machine = PRESS_FACE_RADIUS_MM * np.column_stack(
        (np.cos(footprint_angles), np.sin(footprint_angles))
    )

    def full_press_face_inside(candidate_machine: np.ndarray) -> bool:
        perimeter_model = (
            candidate_machine[None, :] + footprint_offsets_machine - intercept
        ) @ inverse
        return all(
            silicone_contains_point(float(x_norm), float(y_norm))
            for x_norm, y_norm in perimeter_model
        )

    angles = np.linspace(0.0, 2.0 * math.pi, 721)
    cosine = np.cos(angles)
    sine = np.sin(angles)
    boundary_model = np.column_stack(
        (
            SILICONE_X_RADIUS * np.sign(cosine) * np.sqrt(np.abs(cosine)),
            SILICONE_Y_CENTER
            + SILICONE_Y_RADIUS * np.sign(sine) * np.sqrt(np.abs(sine)),
        )
    )
    boundary_machine = boundary_model @ linear + intercept
    center_machine = (
        np.asarray((0.0, SILICONE_Y_CENTER), dtype=float) @ linear + intercept
    )
    row_spacing = spacing * math.sqrt(3.0) / 2.0
    row_min = math.floor(
        (float(np.min(boundary_machine[:, 1])) - center_machine[1]) / row_spacing
    ) - 1
    row_max = math.ceil(
        (float(np.max(boundary_machine[:, 1])) - center_machine[1]) / row_spacing
    ) + 1

    rows = []
    machine_x_min = float(np.min(boundary_machine[:, 0]))
    machine_x_max = float(np.max(boundary_machine[:, 0]))
    for row_index in range(row_min, row_max + 1):
        machine_y = float(center_machine[1] + row_index * row_spacing)
        row_offset = 0.5 * spacing if row_index % 2 else 0.0
        column_min = math.floor(
            (machine_x_min - center_machine[0] - row_offset) / spacing
        ) - 1
        column_max = math.ceil(
            (machine_x_max - center_machine[0] - row_offset) / spacing
        ) + 1
        row = []
        for column_index in range(column_min, column_max + 1):
            candidate_machine = np.asarray(
                (
                    center_machine[0] + row_offset + column_index * spacing,
                    machine_y,
                ),
                dtype=float,
            )
            model_xy = (candidate_machine - intercept) @ inverse
            if not silicone_contains_point(
                float(model_xy[0]), float(model_xy[1])
            ) or not full_press_face_inside(candidate_machine):
                continue
            target = machine_target_from_model_point(
                float(model_xy[0]), float(model_xy[1]), samples, sensors
            )
            if target is None:
                continue
            row.append(
                {
                    "model_x": float(model_xy[0]),
                    "model_y": float(model_xy[1]),
                    "work_x": float(target[0]),
                    "work_y": float(target[1]),
                    "contact_z": float(target[2]),
                    "row": int(row_index),
                }
            )
        row.sort(key=lambda point: float(point["work_x"]))
        if row_index % 2:
            row.reverse()
        if row:
            rows.append(row)

    plan = []
    for row in rows:
        plan.extend(row)
    for index, point in enumerate(plan, start=1):
        point["point_index"] = index
    return tuple(plan)


def ordered_sensor_peaks(
    channel_peaks: Sequence[float], layout: FingerLayout
) -> np.ndarray:
    values = np.asarray(channel_peaks, dtype=float).reshape(-1)
    output = np.full(SENSOR_COUNT, np.nan, dtype=float)
    for index, sensor in enumerate(layout.sensors):
        if sensor.peak_index >= values.size:
            continue
        value = float(values[sensor.peak_index])
        if math.isfinite(value) and value > 0.0:
            output[index] = value
    return output


def wavelength_shifts_pm(
    current_nm: Sequence[float],
    baseline_nm: Sequence[float] | None,
    *,
    remove_common_mode: bool = False,
) -> np.ndarray:
    current = np.asarray(current_nm, dtype=float)
    if baseline_nm is None:
        return np.full_like(current, np.nan, dtype=float)
    baseline = np.asarray(baseline_nm, dtype=float)
    if current.shape != baseline.shape:
        raise ValueError("当前峰值与无应力基准点数不一致")
    shifts = (current - baseline) * 1000.0
    valid = np.isfinite(current) & np.isfinite(baseline)
    shifts[~valid] = np.nan
    if remove_common_mode and np.any(valid):
        shifts[valid] -= float(np.median(shifts[valid]))
    return shifts


def stress_strength(
    shifts_pm: Sequence[float], activation_pm: float, full_scale_pm: float
) -> np.ndarray:
    shifts = np.abs(np.asarray(shifts_pm, dtype=float))
    span = max(float(full_scale_pm) - float(activation_pm), 1e-9)
    values = np.clip((shifts - float(activation_pm)) / span, 0.0, 1.0)
    values[~np.isfinite(shifts)] = 0.0
    return values


def load_stl_mesh(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Read a binary or ASCII STL and return shared vertices and triangle faces."""

    source = Path(path)
    data = source.read_bytes()
    triangles = None
    if len(data) >= 84:
        triangle_count = struct.unpack_from("<I", data, 80)[0]
        expected_size = 84 + triangle_count * 50
        if triangle_count > 0 and expected_size == len(data):
            record_type = np.dtype(
                [
                    ("normal", "<f4", (3,)),
                    ("vertices", "<f4", (3, 3)),
                    ("attribute", "<u2"),
                ]
            )
            records = np.frombuffer(
                data, dtype=record_type, count=triangle_count, offset=84
            )
            triangles = records["vertices"].astype(np.float64, copy=True)

    if triangles is None:
        text = data.decode("ascii", errors="ignore")
        number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
        matches = re.findall(rf"\bvertex\s+({number})\s+({number})\s+({number})", text)
        if not matches or len(matches) % 3:
            raise ValueError(f"无法解析STL网格：{source}")
        triangles = np.asarray(matches, dtype=float).reshape(-1, 3, 3)

    if not np.all(np.isfinite(triangles)):
        raise ValueError(f"STL含有无效坐标：{source}")
    flat = triangles.reshape(-1, 3)
    vertices, inverse = np.unique(
        np.round(flat, decimals=6), axis=0, return_inverse=True
    )
    faces = inverse.reshape(-1, 3).astype(np.uint32)
    keep = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 2] != faces[:, 0])
    )
    faces = faces[keep]
    if len(vertices) < 4 or len(faces) < 2:
        raise ValueError(f"STL网格为空或已退化：{source}")
    return vertices.astype(float), faces


def prepare_thumb_mesh(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Place the SolidWorks thumb in display coordinates and normalize its pad XY."""

    prepared = np.asarray(vertices, dtype=float).copy()
    if prepared.ndim != 2 or prepared.shape[1] != 3:
        raise ValueError("拇指网格顶点必须为N×3")
    lower = np.min(prepared, axis=0)
    upper = np.max(prepared, axis=0)
    span = upper - lower
    if np.any(span <= 1e-6):
        raise ValueError("拇指网格尺寸无效")

    # The supplied part was modelled with Y along the thumb and X normal to the
    # broad sensing face.  Convert it to viewer coordinates: X=left/right,
    # Y=root/tip, Z=outward.  Keep the original millimetre proportions.
    prepared = prepared[:, (2, 1, 0)]
    prepared[:, 0] *= -1.0
    lower = np.min(prepared, axis=0)
    upper = np.max(prepared, axis=0)
    span = upper - lower
    prepared[:, 0] -= (lower[0] + upper[0]) * 0.5
    prepared[:, 1] -= (lower[1] + upper[1]) * 0.5
    prepared[:, 2] -= lower[2]
    normalized = np.column_stack(
        (
            prepared[:, 0] / (span[0] * 0.5),
            prepared[:, 1] / (span[1] * 0.5),
        )
    )
    return prepared, normalized


def prepare_press_tool_mesh(vertices: np.ndarray) -> np.ndarray:
    """Put the V2 press head in a tip-at-origin display coordinate system.

    The supplied STL uses Z along the tool shaft.  Its 8 mm circular contact
    face is at Z=0 and the narrow shank extends toward positive Z, which already
    matches the finger viewer's outward surface normal.
    """

    prepared = np.asarray(vertices, dtype=float).copy()
    if prepared.ndim != 2 or prepared.shape[1] != 3:
        raise ValueError("压头网格顶点必须为N×3")
    lower = np.min(prepared, axis=0)
    upper = np.max(prepared, axis=0)
    span = upper - lower
    if np.any(span <= 1e-6):
        raise ValueError("压头网格尺寸无效")
    prepared[:, 0] -= (lower[0] + upper[0]) * 0.5
    prepared[:, 1] -= (lower[1] + upper[1]) * 0.5
    prepared[:, 2] -= lower[2]
    return prepared


def rotation_from_positive_z(
    target_normal: Sequence[float],
) -> tuple[float, np.ndarray]:
    """Return the axis/angle that aligns the tool shaft with a surface normal."""

    normal = np.asarray(target_normal, dtype=float).reshape(3)
    length = float(np.linalg.norm(normal))
    if not math.isfinite(length) or length <= 1e-12:
        raise ValueError("手指顶面法向量无效")
    normal /= length
    positive_z = np.asarray((0.0, 0.0, 1.0), dtype=float)
    cosine = float(np.clip(np.dot(positive_z, normal), -1.0, 1.0))
    angle = math.degrees(math.acos(cosine))
    axis = np.cross(positive_z, normal)
    axis_length = float(np.linalg.norm(axis))
    if axis_length <= 1e-12:
        axis = np.asarray((1.0, 0.0, 0.0), dtype=float)
    else:
        axis /= axis_length
    return angle, axis


def silicone_surface_height(
    x_norm: np.ndarray | float,
    y_norm: np.ndarray | float,
    model_height: float,
) -> np.ndarray | float:
    """Approximate the photographed silicone skin laid over the open CAD shell."""

    local_x = np.asarray(x_norm, dtype=float) / SILICONE_X_RADIUS
    local_y = (np.asarray(y_norm, dtype=float) - SILICONE_Y_CENTER) / SILICONE_Y_RADIUS
    radius = np.abs(local_x) ** 4 + np.abs(local_y) ** 4
    dome = np.maximum(0.0, 1.0 - radius) ** 0.55
    # The real pad is inclined: it sits farther outward near the cable/root end
    # and curves gently across the middle where the FBGs are embedded.
    return float(model_height) * (0.925 - 0.205 * np.asarray(y_norm) + 0.035 * dome)


def _silicone_pad_mesh(
    xy_bounds: tuple[np.ndarray, np.ndarray],
    model_height: float,
    radial_rings: int = 24,
    angular_segments: int = 112,
):
    """Build the continuous translucent silicone layer visible in the photos."""

    lower, upper = xy_bounds
    vertices = []
    normalized = []
    faces = []
    for ring in range(radial_rings + 1):
        radius = ring / radial_rings
        segments = 1 if ring == 0 else angular_segments
        for segment in range(segments):
            angle = 0.0 if ring == 0 else 2.0 * math.pi * segment / segments
            cosine = math.cos(angle)
            sine = math.sin(angle)
            local_x = radius * math.copysign(abs(cosine) ** 0.5, cosine)
            local_y = radius * math.copysign(abs(sine) ** 0.5, sine)
            x_norm = SILICONE_X_RADIUS * local_x
            y_norm = SILICONE_Y_CENTER + SILICONE_Y_RADIUS * local_y
            # The recess has a shallow crescent at its rounded tip.  Extend only
            # the middle of that end; fading by X keeps both side walls inset.
            tip_fill = (
                SILICONE_TIP_FILL
                * max(local_y, 0.0) ** 4
                * max(0.0, 1.0 - abs(local_x) ** 4)
            )
            y_norm += tip_fill
            xy = lower + (np.asarray((x_norm, y_norm)) + 1.0) * 0.5 * (upper - lower)
            z = silicone_surface_height(x_norm, y_norm, model_height)
            vertices.append((xy[0], xy[1], z))
            normalized.append((x_norm, y_norm))

    for segment in range(angular_segments):
        faces.append((0, 1 + segment, 1 + (segment + 1) % angular_segments))
    for ring in range(1, radial_rings):
        inner_start = 1 + (ring - 1) * angular_segments
        outer_start = 1 + ring * angular_segments
        for segment in range(angular_segments):
            following = (segment + 1) % angular_segments
            inner = inner_start + segment
            inner_next = inner_start + following
            outer = outer_start + segment
            outer_next = outer_start + following
            faces.extend(((inner, outer, outer_next), (inner, outer_next, inner_next)))
    return (
        np.asarray(vertices, dtype=float),
        np.asarray(faces, dtype=np.uint32),
        np.asarray(normalized, dtype=float),
    )


def _superellipse_height(x_norm: np.ndarray | float, y_norm: np.ndarray | float):
    rho = np.abs(x_norm) ** 4 + np.abs(y_norm) ** 4
    return 3.0 + 5.5 * np.maximum(0.0, 1.0 - rho) ** 0.48


def _top_mesh(radial_rings: int = 26, angular_segments: int = 120):
    """Build a smooth superellipse dome without a clipped grid boundary."""
    vertices = [(0.0, 0.0, float(_superellipse_height(0.0, 0.0)))]
    normalized = [(0.0, 0.0)]
    exponent = 4.0
    for ring in range(1, radial_rings + 1):
        radius = ring / radial_rings
        for segment in range(angular_segments):
            angle = 2.0 * math.pi * segment / angular_segments
            cosine = math.cos(angle)
            sine = math.sin(angle)
            boundary_x = math.copysign(abs(cosine) ** (2.0 / exponent), cosine)
            boundary_y = math.copysign(abs(sine) ** (2.0 / exponent), sine)
            x = radius * boundary_x
            y = radius * boundary_y
            vertices.append((24.0 * x, 36.0 * y, float(_superellipse_height(x, y))))
            normalized.append((x, y))
    faces = []
    for segment in range(angular_segments):
        current = 1 + segment
        following = 1 + (segment + 1) % angular_segments
        faces.append((0, current, following))
    for ring in range(1, radial_rings):
        inner_start = 1 + (ring - 1) * angular_segments
        outer_start = 1 + ring * angular_segments
        for segment in range(angular_segments):
            following = (segment + 1) % angular_segments
            inner = inner_start + segment
            inner_next = inner_start + following
            outer = outer_start + segment
            outer_next = outer_start + following
            faces.extend(((inner, outer, outer_next), (inner, outer_next, inner_next)))
    return (
        np.asarray(vertices, dtype=float),
        np.asarray(faces, dtype=np.uint32),
        np.asarray(normalized, dtype=float),
    )


def _side_mesh(segments: int = 120):
    vertices = []
    faces = []
    exponent = 4.0
    for index in range(segments):
        angle = 2.0 * math.pi * index / segments
        cosine = math.cos(angle)
        sine = math.sin(angle)
        x = math.copysign(abs(cosine) ** (2.0 / exponent), cosine)
        y = math.copysign(abs(sine) ** (2.0 / exponent), sine)
        vertices.extend(((24.0 * x, 36.0 * y, 0.0), (24.0 * x, 36.0 * y, 3.0)))
    for index in range(segments):
        next_index = (index + 1) % segments
        a, b = 2 * index, 2 * index + 1
        c, d = 2 * next_index, 2 * next_index + 1
        faces.extend(((a, c, d), (a, d, b)))
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.uint32)


class MechanicalFinger3DWindow(QtWidgets.QMainWindow):
    """A live 3-D heat map with the synchronized read-only CNC position."""

    press_target_requested = QtCore.pyqtSignal(dict)
    coverage_run_requested = QtCore.pyqtSignal(dict)
    coverage_stop_requested = QtCore.pyqtSignal()
    fast_flank_start_requested = QtCore.pyqtSignal()
    fast_flank_stop_requested = QtCore.pyqtSignal()

    def __init__(
        self,
        layout_path: Path | str = DEFAULT_LAYOUT_PATH,
        parent=None,
        mesh_path: Path | str = DEFAULT_MESH_PATH,
        press_tool_path: Path | str = DEFAULT_PRESS_TOOL_PATH,
        touch_calibration_path: Path | str = DEFAULT_TOUCH_CALIBRATION_PATH,
    ):
        super().__init__(parent)
        self.layout_path = Path(layout_path)
        self.mesh_path = Path(mesh_path)
        self.press_tool_path = Path(press_tool_path)
        self.touch_calibration_path = Path(touch_calibration_path)
        self.layout = load_finger_layout(self.layout_path)
        try:
            self.machine_touch_samples = load_machine_touch_samples(
                self.touch_calibration_path
            )
            self.machine_calibration_error = ""
        except (OSError, ValueError, json.JSONDecodeError) as exception:
            self.machine_touch_samples = ()
            self.machine_calibration_error = str(exception)
        self.current_peaks_by_channel = [np.full(9, np.nan) for _ in range(4)]
        self.current_sensor_peaks = np.full(9, np.nan)
        self.baseline_nm = None
        self.baseline_samples = []
        self.baseline_remaining = 0
        # The stress GraphWindow may replace the historical peak-centre
        # baseline with a stable, full-MAP raw CH1 45-point baseline.  Plain
        # callbacks keep this viewer usable as a standalone demo and avoid a
        # dependency from this Qt/OpenGL module back into app_JDSU.
        self.raw_baseline_capture_handler = None
        self.raw_baseline_clear_handler = None
        self.current_contact_estimate = None
        self.contact_response_strength = None
        self.frame_number = 0
        self.latest_machine_status: dict[str, object] = {}
        self.frame_machine_status: dict[str, object] = {}
        self.coverage_plan: tuple[dict[str, float | int], ...] = ()
        self.coverage_running = False
        self.fast_flank_running = False
        self.physical_reference_trace: PhysicalReferenceTrace | None = None
        self.physical_reference_path: Path | None = None
        self.physical_reference_sha256: str | None = None
        self.physical_reference_error = ""
        self.contact_area_truth: ContactAreaTruthSet | None = None
        self.contact_area_truth_path: Path | None = None
        self.contact_area_truth_sha256: str | None = None
        self.contact_area_truth_error = ""
        self.area_training_session_path: Path | None = None
        self.area_training_session_report: dict | None = None
        self.area_training_session_error = ""
        self.physical_acceptance_session: PhysicalAcceptanceSession | None = None
        self.physical_acceptance_report: dict | None = None
        self.physical_acceptance_path: Path | None = None
        self.physical_acceptance_report_path: Path | None = None
        self.physical_acceptance_error = ""

        self.setWindowTitle("机械手指3D应力定位（ADC光谱）")
        self.resize(1280, 820)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        self._build_ui()
        self._build_scene()
        self._refresh_coverage_plan()
        self._refresh_display()

    def _build_ui(self):
        central = QtWidgets.QWidget(self)
        central.setObjectName("fingerPage")
        central.installEventFilter(self)
        self._responsive_central = central
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(10)

        controls_card = QtWidgets.QFrame()
        controls_card.setObjectName("controlGroup")
        controls = FlowLayout(
            controls_card,
            margin=10,
            horizontal_spacing=9,
            vertical_spacing=8,
        )
        self.channel_combo = QtWidgets.QComboBox()
        self.channel_combo.addItems(("CH0", "CH1", "CH2"))
        self.channel_combo.setCurrentIndex(self.layout.channel)
        self.channel_combo.currentIndexChanged.connect(self._channel_changed)
        controls.addWidget(compact_field("光谱通道：", self.channel_combo))

        self.capture_button = QtWidgets.QPushButton(
            f"采集无应力基准（{self.layout.baseline_frames}帧）"
        )
        self.capture_button.setProperty("role", "primary")
        self.capture_button.clicked.connect(self.begin_baseline_capture)
        controls.addWidget(self.capture_button)

        self.clear_button = QtWidgets.QPushButton("清除基准")
        self.clear_button.setProperty("role", "danger")
        self.clear_button.clicked.connect(self.clear_baseline)
        controls.addWidget(self.clear_button)

        self.common_mode_check = QtWidgets.QCheckBox("去除九点公共波长漂移")
        self.common_mode_check.setChecked(False)
        self.common_mode_check.toggled.connect(self._refresh_display)
        controls.addWidget(self.common_mode_check)

        self.activation_spin = QtWidgets.QDoubleSpinBox()
        self.activation_spin.setRange(0.0, 10000.0)
        self.activation_spin.setDecimals(1)
        self.activation_spin.setSuffix(" pm")
        self.activation_spin.setValue(self.layout.activation_pm)
        self.activation_spin.valueChanged.connect(self._refresh_display)
        controls.addWidget(compact_field("起始阈值：", self.activation_spin))

        self.full_scale_spin = QtWidgets.QDoubleSpinBox()
        self.full_scale_spin.setRange(1.0, 100000.0)
        self.full_scale_spin.setDecimals(1)
        self.full_scale_spin.setSuffix(" pm")
        self.full_scale_spin.setValue(self.layout.full_scale_pm)
        self.full_scale_spin.valueChanged.connect(self._refresh_display)
        controls.addWidget(compact_field("满量程：", self.full_scale_spin))

        self.top_button = QtWidgets.QPushButton("顶视图")
        self.top_button.clicked.connect(
            lambda: self._set_camera(90, -90, getattr(self, "camera_distance", 100.0))
        )
        controls.addWidget(self.top_button)
        self.angle_button = QtWidgets.QPushButton("斜视图")
        self.angle_button.clicked.connect(
            lambda: self._set_camera(55, -90, getattr(self, "camera_distance", 105.0))
        )
        controls.addWidget(self.angle_button)

        self.press_depth_spin = QtWidgets.QDoubleSpinBox()
        self.press_depth_spin.setRange(0.0, MAX_PRESS_DEPTH_MM)
        self.press_depth_spin.setDecimals(2)
        self.press_depth_spin.setSingleStep(0.05)
        self.press_depth_spin.setSuffix(" mm")
        self.press_depth_spin.setValue(0.10)
        self.press_depth_spin.setToolTip(
            f"相对统一接触平面的压入量；硬安全上限{MAX_PRESS_DEPTH_MM:.2f} mm"
        )
        controls.addWidget(compact_field("压入量：", self.press_depth_spin))

        self.xy_speed_spin = QtWidgets.QDoubleSpinBox()
        self.xy_speed_spin.setRange(10.0, 120.0)
        self.xy_speed_spin.setDecimals(0)
        self.xy_speed_spin.setSingleStep(10.0)
        self.xy_speed_spin.setSuffix(" mm/min")
        self.xy_speed_spin.setValue(60.0)
        self.xy_speed_spin.setToolTip("安全高度下的XY协调定位速度；上限120 mm/min")
        controls.addWidget(compact_field("XY速度：", self.xy_speed_spin))

        self.press_speed_spin = QtWidgets.QDoubleSpinBox()
        self.press_speed_spin.setRange(2.0, 30.0)
        self.press_speed_spin.setDecimals(0)
        self.press_speed_spin.setSingleStep(2.0)
        self.press_speed_spin.setSuffix(" mm/min")
        self.press_speed_spin.setValue(15.0)
        self.press_speed_spin.setToolTip("接触硅胶时的Z轴下降速度；上限30 mm/min")
        controls.addWidget(compact_field("按压速度：", self.press_speed_spin))

        self.click_press_arm = QtWidgets.QPushButton("解锁3D点选按压")
        self.click_press_arm.setCheckable(True)
        self.click_press_arm.setProperty("role", "danger")
        self.click_press_arm.setToolTip(
            "解锁后仅下一次左键点击有效；只接受硅胶区域，完成后自动重新锁定"
        )
        self.click_press_arm.toggled.connect(self._click_press_arm_changed)
        controls.addWidget(self.click_press_arm)
        outer.addWidget(controls_card)

        fast_card = QtWidgets.QFrame()
        fast_card.setObjectName("controlGroup")
        fast = FlowLayout(
            fast_card,
            margin=10,
            horizontal_spacing=9,
            vertical_spacing=8,
        )
        fast_title = QtWidgets.QLabel("CH1 九光栅快速压力变化测试")
        fast_title.setObjectName("sectionCaption")
        fast.addWidget(fast_title)
        fast_contract = QtWidgets.QLabel("45点全新直采 · 持续约32分钟 · 不移动机床")
        fast_contract.setObjectName("statusPill")
        fast_contract.setProperty("statusKind", "info")
        fast.addWidget(fast_contract)
        self.fast_flank_start_button = QtWidgets.QPushButton(
            "启动实时定位（可随时停止）"
        )
        self.fast_flank_start_button.setProperty("role", "primary")
        self.fast_flank_start_button.setToolTip(
            "启动后先保持约1.7秒完全无按压，完成64帧基线后才可施压"
        )
        self.fast_flank_start_button.clicked.connect(
            self.fast_flank_start_requested.emit
        )
        fast.addWidget(self.fast_flank_start_button)
        self.fast_flank_stop_button = QtWidgets.QPushButton("安全停止并关光")
        self.fast_flank_stop_button.setProperty("role", "danger")
        self.fast_flank_stop_button.setEnabled(False)
        self.fast_flank_stop_button.clicked.connect(
            self.fast_flank_stop_requested.emit
        )
        fast.addWidget(self.fast_flank_stop_button)
        self.fast_flank_progress = QtWidgets.QProgressBar()
        self.fast_flank_progress.setRange(0, 1024)
        self.fast_flank_progress.setValue(0)
        self.fast_flank_progress.setTextVisible(True)
        fast.addWidget(self.fast_flank_progress)
        self.fast_flank_status_label = QtWidgets.QLabel(
            "待机；启动后的前64个有效帧必须保持无按压"
        )
        self.fast_flank_status_label.setObjectName("softHint")
        fast.addWidget(self.fast_flank_status_label)
        outer.addWidget(fast_card)

        coverage_card = QtWidgets.QFrame()
        coverage_card.setObjectName("controlGroup")
        coverage = FlowLayout(
            coverage_card,
            margin=10,
            horizontal_spacing=9,
            vertical_spacing=8,
        )
        coverage_title = QtWidgets.QLabel("硅胶全覆盖采集")
        coverage_title.setObjectName("sectionCaption")
        coverage.addWidget(coverage_title)
        contact_only = QtWidgets.QLabel(
            "正式采集使用上方压入量 · 动作测试固定 0.00 mm"
        )
        contact_only.setObjectName("statusPill")
        contact_only.setProperty("statusKind", "info")
        coverage.addWidget(contact_only)

        self.coverage_spacing_spin = QtWidgets.QDoubleSpinBox()
        self.coverage_spacing_spin.setRange(2.0, 8.0)
        self.coverage_spacing_spin.setDecimals(1)
        self.coverage_spacing_spin.setSingleStep(0.5)
        self.coverage_spacing_spin.setSuffix(" mm")
        self.coverage_spacing_spin.setValue(2.0)
        self.coverage_spacing_spin.setToolTip(
            "8 mm圆形压面默认采用2 mm点距，约75%重叠；整个圆面必须留在硅胶内"
        )
        self.coverage_spacing_spin.valueChanged.connect(
            self._refresh_coverage_plan
        )
        coverage.addWidget(compact_field("点距：", self.coverage_spacing_spin))

        self.coverage_repeats_spin = QtWidgets.QSpinBox()
        self.coverage_repeats_spin.setRange(1, 5)
        self.coverage_repeats_spin.setValue(3)
        self.coverage_repeats_spin.setSuffix(" 次")
        self.coverage_repeats_spin.valueChanged.connect(
            self._refresh_coverage_summary
        )
        coverage.addWidget(compact_field("重复：", self.coverage_repeats_spin))

        self.coverage_dwell_spin = QtWidgets.QDoubleSpinBox()
        self.coverage_dwell_spin.setRange(0.5, 5.0)
        self.coverage_dwell_spin.setDecimals(1)
        self.coverage_dwell_spin.setSingleStep(0.5)
        self.coverage_dwell_spin.setSuffix(" s")
        self.coverage_dwell_spin.setValue(1.0)
        self.coverage_dwell_spin.setToolTip("到达接触面后保持，用于采集稳定光谱帧")
        self.coverage_dwell_spin.valueChanged.connect(
            self._refresh_coverage_summary
        )
        coverage.addWidget(compact_field("接触保持：", self.coverage_dwell_spin))

        self.coverage_preview_button = QtWidgets.QPushButton("刷新覆盖点")
        self.coverage_preview_button.clicked.connect(self._refresh_coverage_plan)
        coverage.addWidget(self.coverage_preview_button)
        self.coverage_start_button = QtWidgets.QPushButton("开始覆盖采集")
        self.coverage_start_button.setProperty("role", "primary")
        self.coverage_start_button.clicked.connect(self._request_coverage_run)
        coverage.addWidget(self.coverage_start_button)
        self.coverage_motion_test_button = QtWidgets.QPushButton(
            "动作测试1轮（不采光谱）"
        )
        self.coverage_motion_test_button.setToolTip(
            "依次接触31个覆盖位置；压入量0.00 mm，不要求九峰在线，也不保存光谱"
        )
        self.coverage_motion_test_button.clicked.connect(
            self._request_coverage_motion_test
        )
        coverage.addWidget(self.coverage_motion_test_button)
        self.coverage_stop_button = QtWidgets.QPushButton("完成当前点后停止")
        self.coverage_stop_button.setProperty("role", "danger")
        self.coverage_stop_button.setEnabled(False)
        self.coverage_stop_button.clicked.connect(
            self.coverage_stop_requested.emit
        )
        coverage.addWidget(self.coverage_stop_button)
        self.coverage_summary_label = QtWidgets.QLabel("正在计算覆盖方案…")
        self.coverage_summary_label.setObjectName("softHint")
        coverage.addWidget(self.coverage_summary_label)
        self.xy_speed_spin.valueChanged.connect(self._refresh_coverage_summary)
        self.press_speed_spin.valueChanged.connect(self._refresh_coverage_summary)
        self.press_depth_spin.valueChanged.connect(self._refresh_coverage_summary)
        outer.addWidget(coverage_card)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.view = gl.GLViewWidget()
        self.view.setObjectName("fingerViewport")
        self.view.setBackgroundColor(VIEW_BACKGROUND_RGBA)
        self.view.opts["center"] = QtGui.QVector3D(0, 0, 4)
        self.view.setMinimumSize(260, 210)
        self.view.installEventFilter(self)
        self._set_camera(55, -90, 105)
        self.splitter.addWidget(self.view)

        right = QtWidgets.QFrame()
        right.setObjectName("contentCard")
        right.setMinimumWidth(300)
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(14, 14, 14, 14)
        right_layout.setSpacing(9)
        self.source_label = QtWidgets.QLabel(
            "数据源：本帧ADC反射光谱 → 分波段拟合中心波长"
        )
        self.source_label.setWordWrap(True)
        self.source_label.setObjectName("softHint")
        right_layout.addWidget(self.source_label)

        self.status_label = QtWidgets.QLabel("等待应力寻峰光谱；请在无应力时采集基准")
        self.status_label.setWordWrap(True)
        self.status_label.setObjectName("statusPill")
        self.status_label.setProperty("statusKind", "warning")
        right_layout.addWidget(self.status_label)

        self.contact_metrics_label = QtWidgets.QLabel(
            "CH1九峰定位：等待稳定MAP原始基准\n"
            + PRESS_NOMINAL_AREA_LABEL
        )
        self.contact_metrics_label.setWordWrap(True)
        self.contact_metrics_label.setObjectName("softHint")
        right_layout.addWidget(self.contact_metrics_label)

        truth_card = QtWidgets.QFrame()
        truth_card.setObjectName("controlGroup")
        truth_layout = QtWidgets.QVBoxLayout(truth_card)
        truth_layout.setContentsMargins(12, 10, 12, 10)
        truth_layout.setSpacing(7)
        truth_title = QtWidgets.QLabel("物理验收参考")
        truth_title.setObjectName("sectionCaption")
        truth_layout.addWidget(truth_title)
        truth_actions = QtWidgets.QHBoxLayout()
        self.import_pressure_reference_button = QtWidgets.QPushButton(
            "导入15 Hz压力参考"
        )
        self.import_pressure_reference_button.clicked.connect(
            self._choose_physical_reference
        )
        truth_actions.addWidget(self.import_pressure_reference_button)
        self.import_area_truth_button = QtWidgets.QPushButton("导入接触面积真值")
        self.import_area_truth_button.clicked.connect(self._choose_contact_area_truth)
        truth_actions.addWidget(self.import_area_truth_button)
        self.build_area_training_session_button = QtWidgets.QPushButton(
            "生成面积训练会话"
        )
        self.build_area_training_session_button.setToolTip(
            "把逐次独立测得的面积真值与位置合格的覆盖采集事件一一绑定"
        )
        self.build_area_training_session_button.clicked.connect(
            self._choose_area_training_session
        )
        truth_actions.addWidget(self.build_area_training_session_button)
        self.run_physical_acceptance_button = QtWidgets.QPushButton(
            "运行CH1物理验收"
        )
        self.run_physical_acceptance_button.clicked.connect(
            self._choose_physical_acceptance_session
        )
        truth_actions.addWidget(self.run_physical_acceptance_button)
        truth_actions.addStretch(1)
        truth_layout.addLayout(truth_actions)
        self.truth_readiness_label = QtWidgets.QLabel()
        self.truth_readiness_label.setWordWrap(True)
        self.truth_readiness_label.setObjectName("statusPill")
        self.truth_readiness_label.setProperty("statusKind", "warning")
        truth_layout.addWidget(self.truth_readiness_label)
        truth_note = QtWidgets.QLabel(
            "压力参考必须与CH1帧共用主机单调时钟；面积真值必须逐事件独立测量并带测量、标定编号。"
        )
        truth_note.setWordWrap(True)
        truth_note.setObjectName("softHint")
        truth_layout.addWidget(truth_note)
        right_layout.addWidget(truth_card)
        self._refresh_truth_readiness()

        position_card = QtWidgets.QFrame()
        position_card.setObjectName("controlGroup")
        position_layout = QtWidgets.QVBoxLayout(position_card)
        position_layout.setContentsMargins(12, 10, 12, 10)
        position_layout.setSpacing(4)
        position_title = QtWidgets.QLabel("机床与本帧采样位置")
        position_title.setObjectName("sectionCaption")
        position_layout.addWidget(position_title)
        coordinate_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.machine_live_label = QtWidgets.QLabel(
            "实时工件坐标   X —        Y —        Z — mm"
        )
        self.machine_live_label.setFont(coordinate_font)
        self.machine_live_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        position_layout.addWidget(self.machine_live_label)
        self.machine_absolute_label = QtWidgets.QLabel(
            "机械绝对坐标   X —        Y —        Z — mm"
        )
        self.machine_absolute_label.setFont(coordinate_font)
        self.machine_absolute_label.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse
        )
        position_layout.addWidget(self.machine_absolute_label)
        self.machine_frame_label = QtWidgets.QLabel(
            "本帧采样位置   尚未收到ADC帧或机床坐标"
        )
        self.machine_frame_label.setFont(coordinate_font)
        self.machine_frame_label.setWordWrap(True)
        self.machine_frame_label.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse
        )
        position_layout.addWidget(self.machine_frame_label)
        self.machine_mapping_label = QtWidgets.QLabel("3D压头定位：等待Mach3坐标")
        self.machine_mapping_label.setObjectName("softHint")
        self.machine_mapping_label.setWordWrap(True)
        position_layout.addWidget(self.machine_mapping_label)
        self.click_press_status_label = QtWidgets.QLabel(
            "3D点选按压：已锁定；解锁后可点击硅胶区域"
        )
        self.click_press_status_label.setObjectName("softHint")
        self.click_press_status_label.setWordWrap(True)
        position_layout.addWidget(self.click_press_status_label)
        right_layout.addWidget(position_card)

        self.sensor_table = QtWidgets.QTableWidget(9, 6)
        self.sensor_table.setHorizontalHeaderLabels(
            ("光栅", "位置", "当前λ/nm", "基准λ/nm", "Δλ/pm", "响应")
        )
        self.sensor_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.sensor_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.sensor_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.sensor_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents
        )
        self.sensor_table.horizontalHeader().setStretchLastSection(True)
        self.sensor_table.setHorizontalScrollMode(
            QtWidgets.QAbstractItemView.ScrollPerPixel
        )
        right_layout.addWidget(self.sensor_table, 1)

        mapping_note = QtWidgets.QLabel(
            "九个光栅位置读取 finger_sensor_layout.yaml；"
            "三维外形使用真实SolidWorks拇指模型与V2压头。"
        )
        mapping_note.setWordWrap(True)
        mapping_note.setObjectName("softHint")
        right_layout.addWidget(mapping_note)
        self.splitter.addWidget(right)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setSizes((800, 480))
        outer.addWidget(self.splitter, 1)
        self._update_splitter_orientation(central.width())

    def _choose_physical_reference(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "导入同步压力参考",
            str(HERE),
            "JSON 数据 (*.json);;所有文件 (*)",
        )
        if path:
            self.load_physical_reference_file(path)

    def _choose_contact_area_truth(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "导入接触面积真值",
            str(HERE),
            "JSON 数据 (*.json);;所有文件 (*)",
        )
        if path:
            self.load_contact_area_truth_file(path)

    def _choose_area_training_session(self):
        if self.contact_area_truth_path is None:
            self.area_training_session_error = "请先导入接触面积真值"
            self._refresh_truth_readiness()
            return
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "选择已完成位置审核的覆盖采集会话",
            str(HERE / "outputs"),
        )
        if not path:
            return
        source = Path(path)
        base = source.with_name(f"{source.name}_area")
        destination = base
        suffix = 1
        while destination.exists():
            destination = base.with_name(f"{base.name}_{suffix:02d}")
            suffix += 1
        self.create_area_training_session(source, destination)

    def _choose_physical_acceptance_session(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择CH1物理验收会话",
            str(HERE),
            "JSON 数据 (*.json);;所有文件 (*)",
        )
        if path:
            self.load_physical_acceptance_session_file(path)

    @staticmethod
    def _qualified_pressure_reference(
        trace: PhysicalReferenceTrace,
    ) -> tuple[bool, str]:
        if not trace.synchronized:
            return False, "文件未声明已同步"
        if trace.timebase != "host_monotonic_ns":
            return False, "时间基准不是 host_monotonic_ns"
        if trace.maximum_clock_error_ms > 2.0:
            return False, "最大时钟误差超过2.0 ms"
        return True, ""

    def load_physical_reference_file(self, path: Path | str) -> bool:
        """Validate and stage a traceable pressure reference; never starts motion."""

        try:
            source_path = Path(path)
            trace = load_physical_reference(source_path)
            qualified, reason = self._qualified_pressure_reference(trace)
            if not qualified:
                raise ValueError(reason)
            source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exception:
            self.physical_reference_trace = None
            self.physical_reference_path = None
            self.physical_reference_sha256 = None
            self.physical_reference_error = str(exception)
            self._clear_physical_acceptance_result()
            self._refresh_truth_readiness()
            return False
        self.physical_reference_trace = trace
        self.physical_reference_path = source_path
        self.physical_reference_sha256 = source_sha256
        self.physical_reference_error = ""
        self._clear_physical_acceptance_result()
        self._refresh_truth_readiness()
        return True

    def load_contact_area_truth_file(self, path: Path | str) -> bool:
        """Validate independently measured area labels; never infers tool area."""

        try:
            source_path = Path(path)
            truth = load_contact_area_truth(source_path)
            source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exception:
            self.contact_area_truth = None
            self.contact_area_truth_path = None
            self.contact_area_truth_sha256 = None
            self.contact_area_truth_error = str(exception)
            self.area_training_session_path = None
            self.area_training_session_report = None
            self.area_training_session_error = ""
            self._clear_physical_acceptance_result()
            self._refresh_truth_readiness()
            return False
        self.contact_area_truth = truth
        self.contact_area_truth_path = source_path
        self.contact_area_truth_sha256 = source_sha256
        self.contact_area_truth_error = ""
        self.area_training_session_path = None
        self.area_training_session_report = None
        self.area_training_session_error = ""
        self._clear_physical_acceptance_result()
        self._refresh_truth_readiness()
        return True

    def create_area_training_session(
        self, source: Path | str, destination: Path | str
    ) -> bool:
        """Bind imported physical area truth to one reviewed coverage session."""

        if (
            self.contact_area_truth_path is None
            or self.contact_area_truth_sha256 is None
        ):
            self.area_training_session_path = None
            self.area_training_session_report = None
            self.area_training_session_error = "请先导入接触面积真值"
            self._refresh_truth_readiness()
            return False
        try:
            if hashlib.sha256(
                self.contact_area_truth_path.read_bytes()
            ).hexdigest() != self.contact_area_truth_sha256:
                raise ValueError("接触面积真值文件在导入后发生变化，请重新导入")
            report = attach_contact_area_truth(
                source,
                self.contact_area_truth_path,
                destination,
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exception:
            self.area_training_session_path = None
            self.area_training_session_report = None
            self.area_training_session_error = str(exception)
            self._refresh_truth_readiness()
            return False
        self.area_training_session_path = Path(destination)
        self.area_training_session_report = report
        self.area_training_session_error = ""
        self._refresh_truth_readiness()
        return True

    def _clear_physical_acceptance_result(self):
        self.physical_acceptance_session = None
        self.physical_acceptance_report = None
        self.physical_acceptance_path = None
        self.physical_acceptance_report_path = None
        self.physical_acceptance_error = ""

    def load_physical_acceptance_session_file(self, path: Path | str) -> bool:
        """Run frequency, position and true-area gates on one held-out session."""

        if self.physical_reference_trace is None or self.contact_area_truth is None:
            self._clear_physical_acceptance_result()
            self.physical_acceptance_error = (
                "请先导入合格的压力参考和接触面积真值"
            )
            self._refresh_truth_readiness()
            return False
        try:
            if (
                self.physical_reference_path is None
                or self.physical_reference_sha256 is None
                or self.contact_area_truth_path is None
                or self.contact_area_truth_sha256 is None
            ):
                raise ValueError("物理参考文件来源不完整，请重新导入")
            if hashlib.sha256(
                self.physical_reference_path.read_bytes()
            ).hexdigest() != self.physical_reference_sha256:
                raise ValueError("压力参考文件在导入后发生变化，请重新导入")
            if hashlib.sha256(
                self.contact_area_truth_path.read_bytes()
            ).hexdigest() != self.contact_area_truth_sha256:
                raise ValueError("接触面积真值文件在导入后发生变化，请重新导入")
            session = load_physical_acceptance_session(path)
            report = evaluate_physical_acceptance_session(
                session,
                self.physical_reference_trace,
                self.contact_area_truth,
            )
            source_path = Path(path)
            session_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
            report["input_files"] = {
                "pressure_reference": {
                    "path": str(self.physical_reference_path.resolve()),
                    "sha256": self.physical_reference_sha256,
                },
                "contact_area_truth": {
                    "path": str(self.contact_area_truth_path.resolve()),
                    "sha256": self.contact_area_truth_sha256,
                },
                "ch1_acceptance_session": {
                    "path": str(source_path.resolve()),
                    "sha256": session_sha256,
                },
            }
            report_path = source_path.with_name(
                f"{source_path.stem}_physical_acceptance_report.json"
            )
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exception:
            self.physical_acceptance_session = None
            self.physical_acceptance_report = None
            self.physical_acceptance_path = None
            self.physical_acceptance_report_path = None
            self.physical_acceptance_error = str(exception)
            self._refresh_truth_readiness()
            return False
        self.physical_acceptance_session = session
        self.physical_acceptance_report = report
        self.physical_acceptance_path = source_path
        self.physical_acceptance_report_path = report_path
        self.physical_acceptance_error = ""
        self._refresh_truth_readiness()
        return True

    def _refresh_truth_readiness(self):
        if not hasattr(self, "truth_readiness_label"):
            return
        parts = []
        if self.physical_reference_trace is None:
            detail = (
                f"（{self.physical_reference_error}）"
                if self.physical_reference_error
                else ""
            )
            parts.append(f"15 Hz压力参考：未就绪{detail}")
        else:
            trace = self.physical_reference_trace
            parts.append(
                "15 Hz压力参考：已导入 "
                f"{len(trace.values)}点 · {trace.source_device} · {trace.calibration_id} · "
                f"时钟误差≤{trace.maximum_clock_error_ms:.3f} ms"
            )
        if self.contact_area_truth is None:
            detail = (
                f"（{self.contact_area_truth_error}）"
                if self.contact_area_truth_error
                else ""
            )
            parts.append(f"接触面积真值：未就绪{detail}")
        else:
            truth = self.contact_area_truth
            parts.append(
                "接触面积真值：已导入 "
                f"{len(truth.records)}次 · {truth.distinct_area_count}个面积等级 · "
                f"标定 {', '.join(truth.calibration_ids)}"
            )
        if self.area_training_session_path is not None:
            parts.append(
                "面积训练会话：已生成 "
                f"{self.area_training_session_path.name} · "
                f"{self.area_training_session_report['press_event_count']}次按压已一一绑定"
            )
        elif self.area_training_session_error:
            parts.append(f"面积训练会话：未生成（{self.area_training_session_error}）")
        both_ready = (
            self.physical_reference_trace is not None
            and self.contact_area_truth is not None
        )
        self.run_physical_acceptance_button.setEnabled(both_ready)
        self.build_area_training_session_button.setEnabled(
            self.contact_area_truth is not None
        )
        report = self.physical_acceptance_report
        if report is not None:
            frequency = report["frequency_evidence"]
            spatial = report["spatial_evidence"]
            amplitude = frequency["amplitude_error_db"]
            amplitude_text = (
                "—" if amplitude is None else f"{float(amplitude):+.2f} dB"
            )
            suffix = (
                f"物理验收：{'全部通过' if report['end_to_end_goal_verified'] else '未通过'} · "
                f"15 Hz {'通过' if report['physical_15hz_verified'] else '未通过'}"
                f"（幅差 {amplitude_text}，相差 "
                f"{float(frequency['phase_difference_deg']):+.2f}°）\n"
                f"位置P95 {float(spatial['position_p95_error_mm']):.3f} mm · "
                f"面积P95 {float(spatial['area_p95_abs_error_mm2']):.3f} mm²\n"
                f"报告已保存：{self.physical_acceptance_report_path.name}"
            )
        elif self.physical_acceptance_error:
            suffix = f"CH1物理验收失败：{self.physical_acceptance_error}"
        elif both_ready:
            suffix = (
                "参考数据就绪；仍需选择CH1会话并执行15 Hz幅相、位置和面积验收。"
            )
        elif self.physical_reference_trace is not None:
            suffix = (
                "压力参考仅完成导入；仍需与CH1响应配对，且面积真值尚未就绪。"
            )
        else:
            suffix = "未满足物理验收输入，软件不会宣称15 Hz或真实面积已通过。"
        self.truth_readiness_label.setText("\n".join((*parts, suffix)))
        self.truth_readiness_label.setProperty(
            "statusKind",
            "info"
            if report is not None and report["end_to_end_goal_verified"]
            else "warning",
        )
        self.truth_readiness_label.style().unpolish(self.truth_readiness_label)
        self.truth_readiness_label.style().polish(self.truth_readiness_label)

    def _update_splitter_orientation(self, width: int):
        """Stack the table below the 3-D scene when side-by-side is too tight."""

        if not hasattr(self, "splitter"):
            return
        orientation = QtCore.Qt.Horizontal if int(width) >= 900 else QtCore.Qt.Vertical
        if self.splitter.orientation() == orientation:
            return
        self.splitter.setOrientation(orientation)
        self.splitter.setSizes((3, 2))

    def eventFilter(self, watched, event):
        if (
            watched is getattr(self, "view", None)
            and event.type() == QtCore.QEvent.MouseButtonPress
            and event.button() == QtCore.Qt.LeftButton
            and getattr(self, "click_press_arm", None) is not None
            and self.click_press_arm.isChecked()
        ):
            self._request_press_from_view_position(event.pos())
            return True
        if (
            watched is getattr(self, "_responsive_central", None)
            and event.type() == QtCore.QEvent.Resize
        ):
            self._update_splitter_orientation(event.size().width())
        return super().eventFilter(watched, event)

    def _build_scene(self):
        self.real_mesh_loaded = False
        mesh_error = ""
        try:
            mesh_vertices, mesh_faces = load_stl_mesh(self.mesh_path)
            self.top_vertices, self.top_norm = prepare_thumb_mesh(mesh_vertices)
            self.top_faces = mesh_faces
            self.real_mesh_loaded = True
        except (OSError, ValueError, struct.error) as exception:
            mesh_error = str(exception)
            self.top_vertices, self.top_faces, self.top_norm = _top_mesh()

        neutral = np.tile(np.array(BODY_BASE_RGBA), (len(self.top_vertices), 1))
        self.body_item = gl.GLMeshItem(
            vertexes=self.top_vertices,
            faces=self.top_faces,
            vertexColors=neutral,
            smooth=True,
            drawEdges=False,
            shader="edgeHilight",
            glOptions="opaque",
        )
        self.view.addItem(self.body_item)

        if not self.real_mesh_loaded:
            side_vertices, side_faces = _side_mesh()
            self.side_item = gl.GLMeshItem(
                vertexes=side_vertices,
                faces=side_faces,
                color=(0.66, 0.74, 0.83, 1.0),
                smooth=True,
                drawEdges=False,
                shader=None,
                glOptions="opaque",
            )
            self.view.addItem(self.side_item)

            base = gl.GLBoxItem(
                size=QtGui.QVector3D(34, 13, 5),
                color=(0.58, 0.67, 0.77, 1.0),
            )
            base.translate(-17, -43, -2)
            self.view.addItem(base)
            for x in (-7.0, 0.0, 7.0):
                cable = gl.GLLinePlotItem(
                    pos=np.asarray(((x, -40, 0), (0.65 * x, -63, -8)), dtype=float),
                    color=(0.50, 0.59, 0.70, 1.0),
                    width=3,
                    antialias=True,
                    mode="line_strip",
                )
                self.view.addItem(cable)

        lower = np.min(self.top_vertices, axis=0)
        upper = np.max(self.top_vertices, axis=0)
        self.model_span = upper - lower
        self.mesh_xy_bounds = (lower[:2].copy(), upper[:2].copy())
        self.surface_offset = max(float(np.max(self.model_span)) * 0.008, 0.28)
        self.camera_distance = max(float(np.max(self.model_span)) * 1.58, 72.0)
        model_center = (lower + upper) * 0.5
        self.view.opts["center"] = QtGui.QVector3D(
            float(model_center[0]), float(model_center[1]), float(model_center[2])
        )
        self._set_camera(55, -90, self.camera_distance)

        if self.real_mesh_loaded:
            (
                self.silicone_vertices,
                self.silicone_faces,
                self.silicone_norm,
            ) = _silicone_pad_mesh(self.mesh_xy_bounds, float(self.model_span[2]))
            self.heat_base_color = np.asarray(SILICONE_BASE_RGBA)
            silicone_colors = np.tile(
                self.heat_base_color, (len(self.silicone_vertices), 1)
            )
            self.silicone_item = gl.GLMeshItem(
                vertexes=self.silicone_vertices,
                faces=self.silicone_faces,
                vertexColors=silicone_colors,
                smooth=True,
                drawEdges=False,
                shader="edgeHilight",
                glOptions="translucent",
            )
            self.view.addItem(self.silicone_item)
            self.heat_item = self.silicone_item
            self.heat_vertices = self.silicone_vertices
            self.heat_faces = self.silicone_faces
            self.heat_norm = self.silicone_norm
            self.source_label.setText(
                "数据源：本帧ADC反射光谱 → 分波段拟合中心波长\n"
                f"3D模型：{self.mesh_path.name}（真实壳体 + 凹槽内硅胶层）"
            )
        else:
            self.heat_base_color = np.asarray(BODY_BASE_RGBA)
            self.heat_item = self.body_item
            self.heat_vertices = self.top_vertices
            self.heat_faces = self.top_faces
            self.heat_norm = self.top_norm
            self.source_label.setText(
                "数据源：本帧ADC反射光谱 → 分波段拟合中心波长\n"
                f"3D模型：真实网格载入失败，已使用备用外形（{mesh_error}）"
            )

        marker_offset = 0.025 if self.real_mesh_loaded else self.surface_offset
        self.sensor_positions = []
        self.sensor_lines = []
        self.sensor_labels = []
        for sensor in self.layout.sensors:
            position = self._sensor_position(sensor.x, sensor.y, marker_offset)
            self.sensor_positions.append(position)
            angle = math.radians(sensor.orientation_deg)
            dx = math.sin(angle) * 0.045
            dy = math.cos(angle) * 0.045
            endpoints = np.asarray(
                (
                    self._sensor_position(
                        sensor.x - dx,
                        sensor.y - dy,
                        marker_offset,
                    ),
                    self._sensor_position(
                        sensor.x + dx,
                        sensor.y + dy,
                        marker_offset,
                    ),
                ),
                dtype=float,
            )
            line = gl.GLLinePlotItem(
                pos=endpoints,
                color=SENSOR_IDLE_RGBA,
                width=5,
                antialias=True,
                mode="line_strip",
            )
            if RENDER_SENSOR_MARKERS:
                self.view.addItem(line)
            self.sensor_lines.append(line)
            label_offset = max(float(np.max(self.model_span)) * 0.026, 1.15)
            label = gl.GLTextItem(
                pos=np.asarray(position)
                + np.asarray((0.65 * label_offset, 0.35 * label_offset, label_offset)),
                text=str(sensor.sensor_id),
                color=QtGui.QColor(45, 57, 76),
                font=QtGui.QFont("Microsoft YaHei", 10),
            )
            if RENDER_SENSOR_MARKERS:
                self.view.addItem(label)
            self.sensor_labels.append(label)
        self.sensor_positions = np.asarray(self.sensor_positions, dtype=float)
        self.halo_item = gl.GLScatterPlotItem(
            pos=self.sensor_positions,
            size=np.full(9, 14.0),
            color=np.tile(np.asarray(SENSOR_HALO_RGBA), (9, 1)),
            pxMode=True,
        )
        if RENDER_SENSOR_MARKERS:
            self.view.addItem(self.halo_item)

        self.machine_anchor_items = []
        for sample in self.machine_touch_samples:
            if sample.x_norm is None or sample.y_norm is None:
                continue
            anchor_position = self._surface_position(
                sample.x_norm,
                sample.y_norm,
                sample.surface_kind,
                0.12,
            )
            anchor = gl.GLScatterPlotItem(
                pos=np.asarray((anchor_position,), dtype=float),
                size=18.0,
                color=np.asarray(((0.96, 0.61, 0.10, 0.98),)),
                pxMode=True,
            )
            self.view.addItem(anchor)
            self.machine_anchor_items.append(anchor)
            anchor_label = gl.GLTextItem(
                pos=anchor_position + np.asarray((0.8, 0.8, 1.0)),
                text=sample.label or "XY原点",
                color=QtGui.QColor(183, 104, 7),
                font=QtGui.QFont("Microsoft YaHei", 10),
            )
            self.view.addItem(anchor_label)
            self.machine_anchor_items.append(anchor_label)

        self.press_tool_loaded = False
        self.press_tool_error = ""
        self.press_tool_pose_kind = "unavailable"
        try:
            tool_vertices, tool_faces = load_stl_mesh(self.press_tool_path)
            self.press_tool_vertices = prepare_press_tool_mesh(tool_vertices)
            self.press_tool_faces = tool_faces
            self.press_tool_item = gl.GLMeshItem(
                vertexes=self.press_tool_vertices,
                faces=self.press_tool_faces,
                color=TOOL_READY_RGBA,
                smooth=True,
                drawEdges=False,
                shader="edgeHilight",
                glOptions="translucent",
            )
            self.view.addItem(self.press_tool_item)
            self.press_tool_loaded = True
        except (OSError, ValueError, struct.error) as exception:
            self.press_tool_error = str(exception)
            self.press_tool_item = None

        self.tool_target_item = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=float),
            size=19.0,
            color=np.asarray(((0.22, 0.49, 0.88, 0.96),)),
            pxMode=True,
        )
        self.view.addItem(self.tool_target_item)
        self.tool_target_item.hide()
        self.click_target_item = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=float),
            size=25.0,
            color=np.asarray(((0.96, 0.55, 0.08, 0.98),)),
            pxMode=True,
        )
        self.view.addItem(self.click_target_item)
        self.click_target_item.hide()
        self.coverage_path_item = gl.GLLinePlotItem(
            pos=np.zeros((1, 3), dtype=float),
            color=(0.28, 0.53, 0.84, 0.35),
            width=1.5,
            antialias=True,
            mode="line_strip",
        )
        self.view.addItem(self.coverage_path_item)
        self.coverage_point_item = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=float),
            size=np.asarray((10.0,)),
            color=np.asarray(((0.22, 0.48, 0.84, 0.78),)),
            pxMode=True,
        )
        self.view.addItem(self.coverage_point_item)
        # Keep the real V2 press-head geometry in the scene even before Mach3
        # supplies a trustworthy, calibrated position.  Previously the item was
        # hidden here and also hidden on every Mach3 read error, which made the
        # 3-D page look as though the press apparatus had not been loaded at all.
        self._show_calibrated_anchor_preview()
        tool_text = (
            f"压头：{self.press_tool_path.name}（真实V2结构，始终显示）"
            if self.press_tool_loaded
            else f"压头：载入失败（{self.press_tool_error}）"
        )
        self.source_label.setText(f"{self.source_label.text()}\n{tool_text}")

    def _refresh_coverage_plan(self, *_args):
        try:
            self.coverage_plan = generate_silicone_coverage_plan(
                self.machine_touch_samples,
                self.layout.sensors,
                spacing_mm=float(self.coverage_spacing_spin.value()),
            )
        except (TypeError, ValueError) as exc:
            self.coverage_plan = ()
            self.coverage_summary_label.setText(f"覆盖方案不可用：{exc}")
            if hasattr(self, "coverage_point_item"):
                self.coverage_point_item.hide()
                self.coverage_path_item.hide()
            return
        positions = np.asarray(
            [
                self._sensor_position(
                    float(point["model_x"]), float(point["model_y"]), 0.20
                )
                for point in self.coverage_plan
            ],
            dtype=float,
        )
        if positions.size:
            self.coverage_path_item.setData(pos=positions)
            self.coverage_point_item.setData(
                pos=positions,
                size=np.full(len(positions), 10.0),
                color=np.tile(
                    np.asarray((0.22, 0.48, 0.84, 0.78)), (len(positions), 1)
                ),
                pxMode=True,
            )
            self.coverage_path_item.show()
            self.coverage_point_item.show()
        self._refresh_coverage_summary()

    def _refresh_coverage_summary(self, *_args):
        unique = len(self.coverage_plan)
        repeats = int(self.coverage_repeats_spin.value())
        depth_mm = float(self.press_depth_spin.value())
        total = unique * repeats
        path_length = sum(
            math.hypot(
                float(current["work_x"]) - float(previous["work_x"]),
                float(current["work_y"]) - float(previous["work_y"]),
            )
            for previous, current in zip(
                self.coverage_plan[:-1], self.coverage_plan[1:], strict=True
            )
        )
        xy_seconds = (
            path_length
            * repeats
            / max(float(self.xy_speed_spin.value()), 1.0)
            * 60.0
        )
        per_contact_seconds = (
            3.5 / 60.0 * 60.0
            + 0.5 / max(float(self.press_speed_spin.value()), 1.0) * 60.0
            + 4.0 / 60.0 * 60.0
            + float(self.coverage_dwell_spin.value())
        )
        estimate_minutes = (xy_seconds + total * per_contact_seconds) / 60.0
        self.coverage_summary_label.setText(
            f"覆盖方案：{unique}个位置 × {repeats}轮 = {total}次接触；"
            f"每点保持 {self.coverage_dwell_spin.value():.1f}s，压入量{depth_mm:.2f} mm；"
            f"预计约{estimate_minutes:.0f}分钟（不含首次定位）"
        )

    def _calibrated_machine_work_offset(self) -> list[float] | None:
        offsets = [
            (
                sample.machine_x - sample.work_x,
                sample.machine_y - sample.work_y,
                sample.machine_z - sample.work_z,
            )
            for sample in self.machine_touch_samples
            if sample.machine_x is not None
            and sample.machine_y is not None
            and sample.machine_z is not None
        ]
        if not offsets:
            return None
        return np.median(np.asarray(offsets, dtype=float), axis=0).tolist()

    def _coverage_request_payload(
        self, *, repeats: int, dwell_seconds: float, motion_test: bool
    ) -> dict | None:
        if not self.coverage_plan:
            self._refresh_coverage_plan()
        if not self.coverage_plan:
            return None
        calibration_offset = self._calibrated_machine_work_offset()
        points = [dict(point) for point in self.coverage_plan]
        if calibration_offset is not None:
            # Work offsets are routinely re-zeroed in Mach3. Persist the
            # physical target in machine-absolute coordinates so a harmless
            # work-zero change cannot move the press to a different place.
            for point in points:
                point["target_machine_x"] = (
                    float(point["work_x"]) + calibration_offset[0]
                )
                point["target_machine_y"] = (
                    float(point["work_y"]) + calibration_offset[1]
                )
                point["contact_machine_z"] = (
                    float(point["contact_z"]) + calibration_offset[2]
                )
        return {
            "points": points,
            "repeats": int(repeats),
            "spacing_mm": float(self.coverage_spacing_spin.value()),
            "dwell_seconds": float(dwell_seconds),
            # Motion-only coverage deliberately remains contact-plane-only.
            # A real acquisition uses the same guarded depth control as a
            # single 3-D point press (0..MAX_PRESS_DEPTH_MM).
            "depth_mm": (
                0.0 if motion_test else float(self.press_depth_spin.value())
            ),
            "xy_feed_mm_min": float(self.xy_speed_spin.value()),
            "z_travel_feed_mm_min": 60.0,
            "press_feed_mm_min": float(self.press_speed_spin.value()),
            "motion_test": bool(motion_test),
            "calibration_machine_minus_work": calibration_offset,
        }

    def _request_coverage_run(self):
        payload = self._coverage_request_payload(
            repeats=int(self.coverage_repeats_spin.value()),
            dwell_seconds=float(self.coverage_dwell_spin.value()),
            motion_test=False,
        )
        if payload is None:
            return
        self.coverage_run_requested.emit(payload)

    def _request_coverage_motion_test(self):
        payload = self._coverage_request_payload(
            repeats=1,
            dwell_seconds=0.3,
            motion_test=True,
        )
        if payload is None:
            return
        self.coverage_run_requested.emit(payload)

    def set_coverage_running(self, running: bool):
        self.coverage_running = bool(running)
        controls_enabled = not self.coverage_running
        for widget in (
            self.coverage_spacing_spin,
            self.coverage_repeats_spin,
            self.coverage_dwell_spin,
            self.coverage_preview_button,
            self.click_press_arm,
        ):
            widget.setEnabled(controls_enabled)
        self.coverage_start_button.setEnabled(
            controls_enabled and not self.fast_flank_running
        )
        self.coverage_motion_test_button.setEnabled(
            controls_enabled and not self.fast_flank_running
        )
        self.coverage_stop_button.setEnabled(self.coverage_running)

    def set_fast_flank_running(self, running: bool):
        """Reflect ownership of the shared optical byte stream in the UI."""

        self.fast_flank_running = bool(running)
        self.fast_flank_start_button.setEnabled(not self.fast_flank_running)
        self.fast_flank_stop_button.setEnabled(self.fast_flank_running)
        self.coverage_start_button.setEnabled(
            not self.fast_flank_running and not self.coverage_running
        )
        self.coverage_motion_test_button.setEnabled(
            not self.fast_flank_running and not self.coverage_running
        )
        if self.fast_flank_running:
            self.fast_flank_progress.setRange(0, 60_000)
            self.fast_flank_progress.setValue(0)
            self.fast_flank_status_label.setText(
                "正在建立无按压基线：请先不要触碰硅胶（64个有效帧）"
            )

    def set_fast_flank_progress(self, completed: int, total: int):
        total = max(1, int(total))
        completed = min(max(0, int(completed)), total)
        self.fast_flank_progress.setRange(0, total)
        self.fast_flank_progress.setValue(completed)
        # The firmware marks cycles 0..15 as minimum warm-up.  The baseline
        # then consumes 64 valid frames, so monitoring begins at frame 81.
        baseline_monitor_start = 16 + 64
        if completed <= 16:
            detail = "激光最短预热阶段；尚未用于基线"
        elif completed <= baseline_monitor_start:
            detail = (
                f"无按压基线 {completed - 16}/64：请继续保持不触碰硅胶"
            )
        else:
            detail = "基线已冻结，可以施压；正在显示因果压力变化位置"
        self.fast_flank_status_label.setText(
            f"{detail} · {completed}/{total}帧"
        )

    def finish_fast_flank(self, message: str, *, success: bool):
        self.set_fast_flank_running(False)
        self.fast_flank_status_label.setText(str(message))
        self.status_label.setProperty(
            "statusKind", "info" if success else "warning"
        )
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def set_coverage_progress(
        self,
        completed: int,
        total: int,
        *,
        active_point_index: int | None = None,
        detail: str = "",
    ):
        unique = len(self.coverage_plan)
        colors = np.tile(
            np.asarray((0.22, 0.48, 0.84, 0.70)), (max(unique, 1), 1)
        )
        if unique:
            completed_unique = min(unique, max(0, int(completed)))
            colors[:completed_unique] = np.asarray((0.18, 0.65, 0.43, 0.82))
            if active_point_index is not None and 0 <= active_point_index < unique:
                colors[active_point_index] = np.asarray((0.98, 0.52, 0.08, 0.98))
            positions = np.asarray(
                [
                    self._sensor_position(
                        float(point["model_x"]), float(point["model_y"]), 0.20
                    )
                    for point in self.coverage_plan
                ],
                dtype=float,
            )
            self.coverage_point_item.setData(
                pos=positions,
                size=np.full(unique, 11.0),
                color=colors,
                pxMode=True,
            )
        suffix = f"；{detail}" if detail else ""
        self.coverage_summary_label.setText(
            f"覆盖采集进度：{int(completed)}/{int(total)}{suffix}"
        )

    def _sensor_position(self, x_norm: float, y_norm: float, z_offset: float = 0.0):
        """Project a normalized sensor coordinate into the photographed silicone."""

        x_norm = float(np.clip(x_norm, -0.94, 0.94))
        y_norm = float(np.clip(y_norm, -0.94, 0.94))
        lower, upper = self.mesh_xy_bounds
        target = lower + (np.asarray((x_norm, y_norm)) + 1.0) * 0.5 * (upper - lower)

        # This CAD variant omits the silicone.  Its inclined, gently domed skin
        # is reconstructed from the two physical photographs; the tiny offset
        # keeps the embedded red segment visible without a suspended gap.
        if self.real_mesh_loaded:
            z = silicone_surface_height(x_norm, y_norm, float(self.model_span[2]))
            return np.asarray((target[0], target[1], z + z_offset), dtype=float)

        return self._mesh_surface_position(x_norm, y_norm, z_offset)

    def _mesh_surface_position(
        self, x_norm: float, y_norm: float, z_offset: float = 0.0
    ) -> np.ndarray:
        """Project a normalized coordinate onto the actual SolidWorks shell."""

        x_norm = float(np.clip(x_norm, -0.98, 0.98))
        y_norm = float(np.clip(y_norm, -0.98, 0.98))
        lower, upper = self.mesh_xy_bounds
        target = lower + (np.asarray((x_norm, y_norm)) + 1.0) * 0.5 * (upper - lower)

        triangles = self.top_vertices[self.top_faces]
        a = triangles[:, 0, :2]
        edge_ab = triangles[:, 1, :2] - a
        edge_ac = triangles[:, 2, :2] - a
        point = target[None, :] - a
        denominator = edge_ab[:, 0] * edge_ac[:, 1] - edge_ab[:, 1] * edge_ac[:, 0]
        valid = np.abs(denominator) > 1e-10
        first = np.zeros_like(denominator)
        second = np.zeros_like(denominator)
        first[valid] = (
            point[valid, 0] * edge_ac[valid, 1] - point[valid, 1] * edge_ac[valid, 0]
        ) / denominator[valid]
        second[valid] = (
            edge_ab[valid, 0] * point[valid, 1] - edge_ab[valid, 1] * point[valid, 0]
        ) / denominator[valid]
        inside = (
            valid
            & (first >= -1e-7)
            & (second >= -1e-7)
            & (first + second <= 1.0 + 1e-7)
        )
        if np.any(inside):
            z_values = (
                triangles[:, 0, 2]
                + first * (triangles[:, 1, 2] - triangles[:, 0, 2])
                + second * (triangles[:, 2, 2] - triangles[:, 0, 2])
            )
            z = float(np.max(z_values[inside]))
            return np.asarray((target[0], target[1], z + z_offset), dtype=float)

        distance2 = np.sum((self.top_norm - np.asarray((x_norm, y_norm))) ** 2, axis=1)
        nearest = np.argsort(distance2)[: min(48, len(distance2))]
        index = int(nearest[np.argmax(self.top_vertices[nearest, 2])])
        position = self.top_vertices[index].copy()
        position[2] += z_offset
        return position

    def _surface_position(
        self,
        x_norm: float,
        y_norm: float,
        surface_kind: str,
        z_offset: float = 0.0,
    ) -> np.ndarray:
        if str(surface_kind).lower() in {"shell", "shell_top"}:
            return self._mesh_surface_position(x_norm, y_norm, z_offset)
        return self._sensor_position(x_norm, y_norm, z_offset)

    def _surface_normal(
        self, x_norm: float, y_norm: float, surface_kind: str = "silicone"
    ) -> np.ndarray:
        """Calculate the local outward normal of the rendered finger top."""

        normal_surface_kind = (
            "silicone" if str(surface_kind).lower() == "shell_top" else surface_kind
        )
        epsilon = 0.002
        tangent_x = self._surface_position(
            x_norm + epsilon, y_norm, normal_surface_kind, 0.0
        ) - self._surface_position(
            x_norm - epsilon, y_norm, normal_surface_kind, 0.0
        )
        tangent_y = self._surface_position(
            x_norm, y_norm + epsilon, normal_surface_kind, 0.0
        ) - self._surface_position(
            x_norm, y_norm - epsilon, normal_surface_kind, 0.0
        )
        normal = np.cross(tangent_x, tangent_y)
        length = float(np.linalg.norm(normal))
        if not math.isfinite(length) or length <= 1e-12:
            return np.asarray((0.0, 0.0, 1.0), dtype=float)
        normal /= length
        if normal[2] < 0.0:
            normal *= -1.0
        return normal

    def _set_camera(self, elevation: float, azimuth: float, distance: float):
        self.view.setCameraPosition(
            elevation=float(elevation), azimuth=float(azimuth), distance=float(distance)
        )

    def _click_press_arm_changed(self, armed: bool):
        if armed:
            self._set_camera(90, -90, getattr(self, "camera_distance", 100.0))
            self.view.setCursor(QtCore.Qt.CrossCursor)
            self.click_press_arm.setText("点选已解锁：点击硅胶")
            self.click_press_status_label.setText(
                "3D点选按压：等待一次点击；硅胶外目标会被拒绝"
            )
        else:
            self.view.unsetCursor()
            self.click_press_arm.setText("解锁3D点选按压")

    def _normalized_point_from_top_view(self, position: QtCore.QPoint) -> tuple[float, float] | None:
        """Unproject a top-view click into the model's normalized XY plane."""

        width = max(1, int(self.view.width()))
        height = max(1, int(self.view.height()))
        ndc_x = 2.0 * float(position.x()) / float(width) - 1.0
        ndc_y = 1.0 - 2.0 * float(position.y()) / float(height)
        viewport = (0, 0, width, height)
        combined = (
            self.view.projectionMatrix(viewport, viewport) * self.view.viewMatrix()
        )
        inverse, invertible = combined.inverted()
        if not invertible:
            return None
        near = inverse * QtGui.QVector4D(ndc_x, ndc_y, -1.0, 1.0)
        far = inverse * QtGui.QVector4D(ndc_x, ndc_y, 1.0, 1.0)
        if abs(float(near.w())) <= 1e-12 or abs(float(far.w())) <= 1e-12:
            return None
        near3 = np.asarray(
            (near.x() / near.w(), near.y() / near.w(), near.z() / near.w()),
            dtype=float,
        )
        far3 = np.asarray(
            (far.x() / far.w(), far.y() / far.w(), far.z() / far.w()),
            dtype=float,
        )
        direction = far3 - near3
        if abs(float(direction[2])) <= 1e-12:
            return None
        reference_z = float(
            silicone_surface_height(
                0.0, SILICONE_Y_CENTER, float(self.model_span[2])
            )
        )
        distance = (reference_z - near3[2]) / direction[2]
        point = near3 + direction * distance
        lower, upper = self.mesh_xy_bounds
        span = upper - lower
        if np.any(span <= 1e-12):
            return None
        normalized = 2.0 * (point[:2] - lower) / span - 1.0
        if not np.all(np.isfinite(normalized)):
            return None
        return float(normalized[0]), float(normalized[1])

    def _request_press_from_view_position(self, position: QtCore.QPoint):
        normalized = self._normalized_point_from_top_view(position)
        if normalized is None:
            self.click_press_status_label.setText("3D点选按压：无法解析点击位置")
            self.click_press_arm.setChecked(False)
            return
        x_norm, y_norm = normalized
        target = machine_target_from_model_point(
            x_norm, y_norm, self.machine_touch_samples, self.layout.sensors
        )
        if target is None:
            self.click_press_status_label.setText(
                "3D点选按压：目标不在硅胶区域内，已拒绝"
            )
            self.click_press_arm.setChecked(False)
            return
        work_x, work_y, contact_z = target
        surface = self._sensor_position(x_norm, y_norm, 0.16)
        self.click_target_item.setData(
            pos=np.asarray((surface,), dtype=float),
            size=25.0,
            color=np.asarray(((0.96, 0.55, 0.08, 0.98),)),
            pxMode=True,
        )
        self.click_target_item.show()
        depth_mm = float(self.press_depth_spin.value())
        if not 0.0 <= depth_mm <= MAX_PRESS_DEPTH_MM:
            self.click_press_status_label.setText(
                f"3D点选按压：压入量必须位于0～{MAX_PRESS_DEPTH_MM:.2f} mm"
            )
            self.click_press_arm.setChecked(False)
            return
        xy_feed = float(self.xy_speed_spin.value())
        press_feed = float(self.press_speed_spin.value())
        self.click_press_status_label.setText(
            f"3D点选按压：目标 X {work_x:+.3f} Y {work_y:+.3f}，"
            f"接触Z {contact_z:+.3f}，压入 {depth_mm:.2f} mm，"
            f"XY {xy_feed:.0f} / 按压 {press_feed:.0f} mm/min"
        )
        self.click_press_arm.setChecked(False)
        request = {
            "model_x": x_norm,
            "model_y": y_norm,
            "work_x": work_x,
            "work_y": work_y,
            "contact_z": contact_z,
            "depth_mm": depth_mm,
            "xy_feed_mm_min": xy_feed,
            "z_travel_feed_mm_min": 60.0,
            "press_feed_mm_min": press_feed,
        }
        calibrated_offsets = [
            (
                sample.machine_x - sample.work_x,
                sample.machine_y - sample.work_y,
                sample.machine_z - sample.work_z,
            )
            for sample in self.machine_touch_samples
            if sample.machine_x is not None
            and sample.machine_y is not None
            and sample.machine_z is not None
        ]
        if calibrated_offsets:
            calibration_offset = np.median(
                np.asarray(calibrated_offsets, dtype=float), axis=0
            ).tolist()
            request["calibration_machine_minus_work"] = calibration_offset
            request["target_machine_x"] = work_x + calibration_offset[0]
            request["target_machine_y"] = work_y + calibration_offset[1]
            request["contact_machine_z"] = contact_z + calibration_offset[2]
        self.press_target_requested.emit(request)

    @staticmethod
    def _coordinate_text(status: Mapping[str, object], prefix: str) -> str | None:
        coordinates = machine_coordinates(status, prefix)
        if coordinates is None:
            return None
        return "   ".join(
            f"{axis} {value:+09.3f}"
            for axis, value in zip(("X", "Y", "Z"), coordinates, strict=True)
        )

    def _hide_press_tool(self):
        if self.press_tool_item is not None:
            self.press_tool_item.hide()
        self.tool_target_item.hide()
        self.press_tool_pose_kind = "unavailable"

    def _place_press_tool(
        self,
        tip: Sequence[float],
        color: Sequence[float],
        *,
        surface: Sequence[float] | None = None,
        surface_normal: Sequence[float] = (0.0, 0.0, 1.0),
        pose_kind: str,
    ):
        """Place the V2 mesh with its 8 mm face parallel to the finger top."""

        if not self.press_tool_loaded or self.press_tool_item is None:
            self._hide_press_tool()
            return
        tip_array = np.asarray(tip, dtype=float)
        self.press_tool_item.setMeshData(
            vertexes=self.press_tool_vertices,
            faces=self.press_tool_faces,
            color=tuple(float(value) for value in color),
            smooth=True,
            drawEdges=False,
            shader="edgeHilight",
        )
        self.press_tool_item.resetTransform()
        angle, axis = rotation_from_positive_z(surface_normal)
        if angle > 1e-9:
            self.press_tool_item.rotate(
                float(angle), float(axis[0]), float(axis[1]), float(axis[2])
            )
        self.press_tool_item.translate(
            float(tip_array[0]), float(tip_array[1]), float(tip_array[2])
        )
        self.press_tool_item.show()
        self.press_tool_pose_kind = str(pose_kind)
        if surface is None:
            self.tool_target_item.hide()
            return
        surface_array = np.asarray(surface, dtype=float)
        self.tool_target_item.setData(
            pos=np.asarray((surface_array,), dtype=float),
            size=19.0,
            color=np.asarray((tuple(float(value) for value in color),), dtype=float),
            pxMode=True,
        )
        self.tool_target_item.show()

    def _show_press_tool_preview(self):
        """Show a clearly marked safe hover pose when live XYZ is unavailable."""

        if not self.press_tool_loaded:
            self._hide_press_tool()
            return
        # Park above the tip-side edge of the pad so the nine FBGs remain
        # readable.  This is a display-only pose and never drives Mach3.
        x_norm, y_norm = 0.70, 0.72
        surface = self._sensor_position(x_norm, y_norm, 0.02)
        surface_normal = self._surface_normal(x_norm, y_norm)
        preview_clearance = max(7.0, float(np.max(self.model_span)) * 0.12)
        tip = surface + surface_normal * preview_clearance
        self._place_press_tool(
            tip,
            TOOL_PREVIEW_RGBA,
            surface=None,
            surface_normal=surface_normal,
            pose_kind="preview",
        )

    def _show_calibrated_anchor_preview(self) -> bool:
        """Place a non-live tool at the operator-confirmed reference contact."""

        anchor = next(
            (
                sample
                for sample in self.machine_touch_samples
                if sample.x_norm is not None and sample.y_norm is not None
            ),
            None,
        )
        if anchor is None or not self.press_tool_loaded:
            self._show_press_tool_preview()
            return False
        surface = self._surface_position(
            anchor.x_norm, anchor.y_norm, anchor.surface_kind, 0.02
        )
        surface_normal = self._surface_normal(
            anchor.x_norm, anchor.y_norm, anchor.surface_kind
        )
        self._place_press_tool(
            surface,
            TOOL_PREVIEW_RGBA,
            surface=surface,
            surface_normal=surface_normal,
            pose_kind="calibration_preview",
        )
        return True

    def _update_press_tool_pose(
        self,
        pose: MachineToolPose | None,
        status: Mapping[str, object],
    ):
        if not self.press_tool_loaded:
            self._hide_press_tool()
            return
        if pose is None:
            self._show_press_tool_preview()
            return
        # Very distant coordinates would move the mesh outside the useful
        # camera volume.  Keep the numeric position visible, but do not imply
        # that an extrapolated pose is trustworthy.
        if not -3.0 <= pose.clearance_mm <= 30.0:
            self._show_press_tool_preview()
            return
        surface = self._surface_position(
            pose.x_norm, pose.y_norm, pose.surface_kind, 0.02
        )
        surface_normal = self._surface_normal(
            pose.x_norm, pose.y_norm, pose.surface_kind
        )
        tip = surface + surface_normal * pose.clearance_mm
        if bool(status.get("estop") or status.get("estop_led")):
            color = TOOL_ESTOP_RGBA
        elif bool(status.get("moving")):
            color = TOOL_MOVING_RGBA
        else:
            color = TOOL_READY_RGBA
        self._place_press_tool(
            tip,
            color,
            surface=surface,
            surface_normal=surface_normal,
            pose_kind="live",
        )

    @QtCore.pyqtSlot(dict)
    def update_machine_status(self, status: Mapping[str, object]):
        """Display read-only Mach3 coordinates and update the calibrated tool."""

        work_text = self._coordinate_text(status, "work")
        machine_text = self._coordinate_text(status, "machine")
        if work_text is None or machine_text is None:
            self.set_machine_status_error("Mach3返回的坐标不完整")
            return
        self.latest_machine_status = dict(status)
        if status.get("estop") or status.get("estop_led"):
            motion_text = "急停锁定"
        elif status.get("moving"):
            motion_text = "移动中"
        else:
            motion_text = "已停止"
        source_text = "演示" if status.get("simulated") else "实时"
        self.machine_live_label.setText(
            f"{source_text}工件坐标   {work_text} mm   · {motion_text}"
        )
        self.machine_absolute_label.setText(f"机械绝对坐标   {machine_text} mm")

        pose = machine_pose_from_touch_samples(
            status, self.machine_touch_samples, self.layout.sensors
        )
        self._update_press_tool_pose(pose, status)
        if not self.press_tool_loaded:
            self.machine_mapping_label.setText(
                f"3D压头载入失败：{self.press_tool_error}"
            )
        elif not self.machine_touch_samples:
            detail = self.machine_calibration_error or "没有有效接触点"
            self.machine_mapping_label.setText(
                f"3D压头定位未标定：{detail}；当前显示半透明安全悬停预览，"
                "上方XYZ数值仍为Mach3原始坐标"
            )
        elif pose is None:
            self.machine_mapping_label.setText(
                "3D压头定位：当前位置在已标定区域之外；压头切换为半透明安全悬停预览，"
                "仅显示原始XYZ，不对尚未校准的位置做空间猜测"
            )
        else:
            sensor = next(
                item
                for item in self.layout.sensors
                if item.sensor_id == pose.nearest_sensor_id
            )
            surface_name = (
                "壳体顶面"
                if pose.surface_kind in {"shell", "shell_top"}
                else "硅胶表面"
            )
            if pose.clearance_mm >= 0.0:
                distance_text = f"距{surface_name}约 {pose.clearance_mm:.3f} mm"
            else:
                distance_text = f"接触压入量估计 {-pose.clearance_mm:.3f} mm"
            if pose.mapping_kind == "anchor":
                mapping_text = pose.calibration_label or "单点基准"
                location_text = "机床—模型基准（非光栅点）"
            elif pose.mapping_kind == "similarity":
                mapping_text = "硅胶下边界双点模型标定区"
                location_text = f"最近G{sensor.sensor_id} {sensor.name}"
            elif pose.mapping_kind == "surface":
                mapping_text = "完整平面标定区"
                location_text = f"最近G{sensor.sensor_id} {sensor.name}"
            else:
                mapping_text = "线性标定区"
                location_text = f"最近G{sensor.sensor_id} {sensor.name}"
            self.machine_mapping_label.setText(
                f"3D压头定位：{mapping_text} · "
                f"{location_text} · {distance_text}"
                + (" · " + relative_effort_label(pose.clearance_mm)
                   if surface_name == "硅胶表面" else "")
            )

    def set_machine_status_error(self, detail: str):
        self.latest_machine_status = {}
        self.machine_live_label.setText(f"实时工件坐标   读取中断：{detail}")
        self.machine_absolute_label.setText("机械绝对坐标   —")
        if self._show_calibrated_anchor_preview():
            self.machine_mapping_label.setText(
                "3D压头定位：Mach3坐标不可用，当前显示硅胶下边界基准的半透明标定预览"
                "（非实时位置）；没有发送任何移动指令"
            )
        else:
            self.machine_mapping_label.setText(
                "3D压头定位：Mach3坐标不可用，当前显示半透明安全悬停预览；"
                "没有发送任何移动指令"
            )

    def _snapshot_machine_position_for_frame(self):
        if not self.latest_machine_status:
            self.frame_machine_status = {}
            self.machine_frame_label.setText(
                f"第{self.frame_number}帧位置   未取得Mach3坐标"
            )
            return
        self.frame_machine_status = dict(self.latest_machine_status)
        frame_text = self._coordinate_text(self.frame_machine_status, "work")
        self.machine_frame_label.setText(
            f"第{self.frame_number}帧最近位置   {frame_text} mm"
        )

    @property
    def selected_channel(self) -> int:
        return int(self.channel_combo.currentIndex())

    def _channel_changed(self, _index):
        self.clear_baseline()
        self.current_sensor_peaks = self.current_peaks_by_channel[
            self.selected_channel
        ].copy()
        self._refresh_display()

    def update_from_spectrum(
        self,
        peaks_by_channel: Sequence[Sequence[float]],
        frame_number: int = 0,
    ):
        for channel in range(min(4, len(peaks_by_channel))):
            self.current_peaks_by_channel[channel] = ordered_sensor_peaks(
                peaks_by_channel[channel], self.layout
            )
        self.current_sensor_peaks = self.current_peaks_by_channel[
            self.selected_channel
        ].copy()
        self.frame_number = int(frame_number)
        self._snapshot_machine_position_for_frame()
        if self.baseline_remaining > 0:
            missing = np.flatnonzero(~np.isfinite(self.current_sensor_peaks))
            if missing.size:
                missing_text = "、".join(f"G{index + 1}" for index in missing)
                self.status_label.setText(
                    f"基准暂停：当前光谱缺少{missing_text}的有效拟合峰"
                )
            else:
                self.baseline_samples.append(self.current_sensor_peaks.copy())
                self.baseline_remaining -= 1
                completed = self.layout.baseline_frames - self.baseline_remaining
                self.capture_button.setText(
                    f"正在采集无应力基准 {completed}/{self.layout.baseline_frames}"
                )
                if self.baseline_remaining == 0:
                    self.baseline_nm = np.median(
                        np.stack(self.baseline_samples), axis=0
                    )
                    self.baseline_samples.clear()
                    self.capture_button.setEnabled(True)
                    self.capture_button.setText(
                        f"重新采集无应力基准（{self.layout.baseline_frames}帧）"
                    )
        self._refresh_display()

    def begin_baseline_capture(self):
        if callable(self.raw_baseline_capture_handler):
            self.raw_baseline_capture_handler()
            return
        self.baseline_samples.clear()
        self.baseline_remaining = self.layout.baseline_frames
        self.capture_button.setEnabled(False)
        self.capture_button.setText(
            f"正在采集无应力基准 0/{self.layout.baseline_frames}"
        )
        self.status_label.setText("请保持机械手指无应力且静止，正在采集基准")

    def clear_baseline(self):
        if callable(self.raw_baseline_clear_handler):
            self.raw_baseline_clear_handler()
            return
        self.baseline_nm = None
        self.baseline_samples.clear()
        self.baseline_remaining = 0
        self.capture_button.setEnabled(True)
        self.capture_button.setText(
            f"采集无应力基准（{self.layout.baseline_frames}帧）"
        )
        self._refresh_display()

    def set_raw_baseline_progress(
        self,
        completed: int,
        total: int,
        *,
        message: str = "",
    ):
        """Reflect GraphWindow's stable full-MAP raw-baseline progress."""

        completed = max(0, int(completed))
        total = max(1, int(total))
        self.capture_button.setEnabled(completed >= total)
        if completed >= total:
            self.capture_button.setText(f"重新采集无应力基准（{total}帧）")
        else:
            self.capture_button.setText(f"正在采集无应力基准 {completed}/{total}")
        if message:
            self.status_label.setText(str(message))

    def set_raw_baseline_cleared(self, *, message: str = ""):
        """Return the shared capture action to its idle state."""

        self.capture_button.setEnabled(True)
        self.capture_button.setText(
            f"采集无应力基准（{self.layout.baseline_frames}帧）"
        )
        self.current_contact_estimate = None
        self.contact_response_strength = None
        if message:
            self.status_label.setText(str(message))
        self._refresh_display()

    @staticmethod
    def _contact_payload(estimate):
        """Return measurement/track fields without importing the localizer."""

        measurement = getattr(estimate, "measurement", estimate)
        return measurement, estimate

    def update_from_contact_estimate(self, estimate, frame_number: int | None = None):
        """Render a CH1 nine-peak contact estimate and its response footprint.

        ``effective_response_area_mm2`` is intentionally presented as an
        estimated influence range.  It is never relabelled as physical contact
        area because this fixture has no independent area calibration.
        """

        if estimate is None:
            self.current_contact_estimate = None
            self.contact_response_strength = None
            self.contact_metrics_label.setText(
                "CH1九峰定位：等待稳定MAP原始基准\n"
                + PRESS_NOMINAL_AREA_LABEL
            )
            self._refresh_display()
            return

        measurement, track = self._contact_payload(estimate)
        raw_weights = np.asarray(
            getattr(measurement, "response_weights", ()), dtype=float
        ).reshape(-1)
        if raw_weights.size != SENSOR_COUNT:
            raise ValueError("contact estimate must carry nine response weights")
        raw_weights = np.where(np.isfinite(raw_weights), np.maximum(raw_weights, 0), 0)
        peak_strength = (
            np.sqrt(raw_weights / float(np.max(raw_weights)))
            if np.max(raw_weights) > 0.0
            else np.zeros(SENSOR_COUNT, dtype=float)
        )
        self.contact_response_strength = np.asarray(
            [peak_strength[sensor.peak_index] for sensor in self.layout.sensors],
            dtype=float,
        )
        self.current_contact_estimate = estimate
        if frame_number is not None:
            self.frame_number = int(frame_number)
        self._snapshot_machine_position_for_frame()

        detected = bool(getattr(measurement, "contact_detected", False))
        x_mm = getattr(track, "x_mm", getattr(measurement, "x_mm", None))
        y_mm = getattr(track, "y_mm", getattr(measurement, "y_mm", None))
        confidence = float(
            getattr(track, "confidence", getattr(measurement, "confidence", 0.0))
        )
        area = float(
            getattr(
                track,
                "effective_response_area_mm2",
                getattr(measurement, "effective_response_area_mm2", 0.0),
            )
        )
        profile = getattr(
            track,
            "profile",
            getattr(measurement, "acquisition_profile", None),
        )
        state = getattr(track, "track_state", "MEASURED" if detected else "NO_CONTACT")
        if detected and x_mm is not None and y_mm is not None:
            position = f"X {float(x_mm):+.2f} / Y {float(y_mm):+.2f} mm"
        else:
            position = "未检出有效接触"
        self.contact_metrics_label.setText(
            f"CH1九峰定位：{position} · 置信度 {confidence:.0%} · "
            f"有效响应面积（估计影响范围） {area:.2f} mm²\n"
            f"采集 {profile or '旧版MAP'} / {state}；"
            + PRESS_NOMINAL_AREA_LABEL
        )
        self._refresh_display()

    def update_from_fast_flank_result(self, result):
        """Render one causal fast-stream result without inventing area truth.

        This display deliberately says ``pressure change`` rather than static
        contact and never shows sensor spread as physical contact area.  The
        independent 45-point path remains unchanged.
        """

        if result is None:
            self.current_contact_estimate = None
            self.contact_response_strength = None
            self.contact_metrics_label.setText(
                "CH1 18点快速定位：等待64帧无接触基线\n"
                + PRESS_NOMINAL_AREA_LABEL
            )
            self._refresh_display()
            return
        supplied_strength = np.asarray(
            getattr(result, "response_strength", ()), dtype=float
        ).reshape(-1)
        fullmap = supplied_strength.size == SENSOR_COUNT
        if fullmap:
            peak_strength = np.where(
                np.isfinite(supplied_strength),
                np.clip(supplied_strength, 0.0, 1.0),
                0.0,
            )
        else:
            delta = np.asarray(
                [
                    np.nan if value is None else float(value)
                    for value in getattr(result, "one_frame_delta_pm", ())
                ],
                dtype=float,
            )
            thresholds = np.asarray(
                [
                    np.nan if value is None else float(value)
                    for value in getattr(result, "activation_threshold_pm", ())
                ],
                dtype=float,
            )
            if delta.size != SENSOR_COUNT or thresholds.size != SENSOR_COUNT:
                raise ValueError("fast result must carry nine response values")
            valid = np.isfinite(delta) & np.isfinite(thresholds) & (thresholds > 0.0)
            peak_strength = np.zeros(SENSOR_COUNT, dtype=float)
            peak_strength[valid] = np.clip(
                np.abs(delta[valid]) / (2.5 * thresholds[valid]), 0.0, 1.0
            )
        self.contact_response_strength = np.asarray(
            [peak_strength[sensor.peak_index] for sensor in self.layout.sensors],
            dtype=float,
        )
        self.current_contact_estimate = result
        self.frame_number = int(getattr(result, "sequence", self.frame_number))
        self._snapshot_machine_position_for_frame()
        self._refresh_display()

        detected = bool(getattr(result, "contact_detected", False))
        x_mm = getattr(result, "provisional_x_mm", None)
        y_mm = getattr(result, "provisional_y_mm", None)
        confidence = float(getattr(result, "confidence", 0.0))
        sampling = bool(getattr(result, "sampling_gate_for_15hz", False))
        footprint_area = float(getattr(result, "press_footprint_area_mm2", 0.0))
        point = getattr(result, "predicted_point", None)
        if detected and x_mm is not None and y_mm is not None:
            prefix = f"P{int(point)} · " if point is not None else ""
            position = f"{prefix}X {float(x_mm):+.2f} / Y {float(y_mm):+.2f} mm"
            state_text = "检测到压力变化"
        else:
            position = "未检测到超过门槛的局部变化"
            state_text = "监测中"
        if fullmap:
            response_rms = getattr(result, "response_rms_codes", None)
            response_text = (
                "—" if response_rms is None else f"{float(response_rms):.2f}码"
            )
            static_trial = float(
                getattr(
                    result,
                    "static_validation_trial_majority_exact_percent",
                    100.0,
                )
            )
            static_frame = float(
                getattr(result, "static_validation_frame_exact_percent", 100.0)
            )
            self.contact_metrics_label.setText(
                f"CH1 45点锁定定位：{position} · 概率间隔 {confidence:.0%} · "
                f"响应RMS {response_text}\n"
                f"{state_text}；独立静态验证轮次 {static_trial:.1f}%、单帧 {static_frame:.1f}%；"
                f"15 Hz采样率门槛{'通过' if sampling else '未就绪'}；"
                f"按压覆盖面积 {footprint_area:.2f} mm²（按直径8 mm圆面标定）"
            )
            self.status_label.setText(
                "45点每帧全部新采；动态15 Hz响应仍需高频位移参考验收；"
                "面积采用8 mm压头名义圆面，不作实际面积反演"
            )
        else:
            self.contact_metrics_label.setText(
                f"CH1 18点快速定位：{position} · 置信度 {confidence:.0%}\n"
                f"{state_text}；15 Hz采样率门槛{'通过' if sampling else '未就绪'}；"
                "位置准确率待复核；" + PRESS_NOMINAL_AREA_LABEL
            )
            self.status_label.setText(
                "快速路径显示的是因果压力变化，不代表持续静态接触；"
                "面积采用8 mm压头名义圆面，不作实际面积反演"
            )

    def _current_shifts(self):
        return wavelength_shifts_pm(
            self.current_sensor_peaks,
            self.baseline_nm,
            remove_common_mode=self.common_mode_check.isChecked(),
        )

    def _refresh_display(self, *_args):
        if not hasattr(self, "body_item"):
            return
        shifts = self._current_shifts()
        activation = float(self.activation_spin.value())
        full_scale = max(float(self.full_scale_spin.value()), activation + 1.0)
        strength = stress_strength(shifts, activation, full_scale)
        if self.contact_response_strength is not None:
            strength = self.contact_response_strength.copy()

        colors = np.tile(self.heat_base_color, (len(self.heat_vertices), 1))
        halo_colors = np.tile(np.asarray(SENSOR_HALO_RGBA), (9, 1))
        halo_sizes = 13.0 + 24.0 * strength
        for index, sensor in enumerate(self.layout.sensors):
            if strength[index] <= 0.0:
                self.sensor_lines[index].setData(color=SENSOR_IDLE_RGBA)
                continue
            positive = not math.isfinite(shifts[index]) or shifts[index] >= 0.0
            target = np.asarray(
                (0.98, 0.22, 0.10, 1.0)
                if positive
                else (0.12, 0.43, 0.92, 1.0)
            )
            distance2 = ((self.heat_norm[:, 0] - sensor.x) / 0.32) ** 2 + (
                (self.heat_norm[:, 1] - sensor.y) / 0.24
            ) ** 2
            blend = np.exp(-2.4 * distance2) * strength[index]
            colors[:, :3] = (
                colors[:, :3] * (1.0 - blend[:, None]) + target[:3] * blend[:, None]
            )
            line_color = tuple(target)
            self.sensor_lines[index].setData(color=line_color)
            halo_colors[index] = target * np.asarray((1.0, 1.0, 1.0, 0.78))

        self.heat_item.setMeshData(
            vertexes=self.heat_vertices,
            faces=self.heat_faces,
            vertexColors=colors,
            smooth=True,
            drawEdges=False,
            shader="edgeHilight",
        )
        self.halo_item.setData(
            pos=self.sensor_positions,
            size=halo_sizes,
            color=halo_colors,
            pxMode=True,
        )
        self._refresh_table(shifts, strength)

    def _refresh_table(self, shifts: np.ndarray, strength: np.ndarray):
        for row, sensor in enumerate(self.layout.sensors):
            current = self.current_sensor_peaks[row]
            baseline = np.nan if self.baseline_nm is None else self.baseline_nm[row]
            shift = shifts[row]
            response = "无数据"
            if math.isfinite(current):
                response = (
                    "等待基准"
                    if self.baseline_nm is None
                    else ("受力" if strength[row] > 0.0 else "静止")
                )
            values = (
                f"G{sensor.sensor_id}",
                sensor.name,
                f"{current:.6f}" if math.isfinite(current) else "—",
                f"{baseline:.6f}" if math.isfinite(baseline) else "—",
                f"{shift:+.2f}" if math.isfinite(shift) else "—",
                response,
            )
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                if response == "受力" and column in (0, 4, 5):
                    item.setBackground(QtGui.QColor(255, 105, 70, 110))
                self.sensor_table.setItem(row, column, item)

        valid = np.isfinite(shifts)
        active = valid & (strength > 0.0)
        if self.baseline_remaining > 0:
            return
        if not np.any(np.isfinite(self.current_sensor_peaks)):
            self.status_label.setText("等待应力寻峰ADC光谱")
        elif self.baseline_nm is None:
            self.status_label.setText("已收到光谱峰值；请在机械手指无应力时采集基准")
        elif not np.any(active):
            self.status_label.setText(
                f"第{self.frame_number}帧：九个位置均低于应力阈值"
            )
        else:
            index = int(np.nanargmax(np.where(valid, np.abs(shifts), np.nan)))
            sensor = self.layout.sensors[index]
            sign = "拉伸/正移" if shifts[index] >= 0.0 else "压缩/负移"
            self.status_label.setText(
                f"第{self.frame_number}帧：最大响应位于G{sensor.sensor_id} {sensor.name}，"
                f"Δλ={shifts[index]:+.2f} pm（{sign}）"
            )


class _DemoSource(QtCore.QObject):
    def __init__(self, window: MechanicalFinger3DWindow):
        super().__init__(window)
        self.window = window
        self.frame = 0
        self.base = np.asarray(
            (
                1527.91,
                1532.14,
                1536.39,
                1540.17,
                1544.46,
                1547.89,
                1551.78,
                1555.81,
                1559.92,
            )
        )
        self.window.baseline_nm = self.base.copy()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(100)

    def tick(self):
        self.frame += 1
        phase = self.frame / 12.0
        shift = np.zeros(9)
        shift[4] = 0.12 * max(0.0, math.sin(phase))
        shift[1] = -0.055 * max(0.0, math.sin(phase - 1.1))
        clearance = 1.8 + 1.1 * math.sin(phase * 0.45)
        sample = self.window.machine_touch_samples[0]
        machine_x = (
            sample.machine_x if sample.machine_x is not None else sample.work_x
        )
        machine_y = (
            sample.machine_y if sample.machine_y is not None else sample.work_y
        )
        machine_z = (
            sample.machine_z if sample.machine_z is not None else sample.work_z
        )
        self.window.update_machine_status(
            {
                "work_x": sample.work_x,
                "work_y": sample.work_y,
                "work_z": sample.work_z + clearance,
                "machine_x": machine_x,
                "machine_y": machine_y,
                "machine_z": machine_z + clearance,
                "moving": True,
                "stopped": False,
                "estop": False,
                "estop_led": False,
                "simulated": True,
            }
        )
        peaks = [self.base.copy() for _ in range(4)]
        peaks[1] = self.base + shift
        self.window.update_from_spectrum(peaks, self.frame)


class _Mach3LiveSource(QtCore.QObject):
    """Read-only 10 Hz Mach3 source for the standalone 3-D window."""

    def __init__(self, window: MechanicalFinger3DWindow, interval_ms: int = 100):
        super().__init__(window)
        from mach3_controller import Mach3Controller

        self.window = window
        self.controller = Mach3Controller()
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(interval_ms)
        self.timer.timeout.connect(self._poll)
        self.timer.start()
        QtCore.QTimer.singleShot(0, self._poll)

    @QtCore.pyqtSlot()
    def _poll(self):
        try:
            self.window.update_machine_status(self.controller.status())
        except Exception as exception:
            self.window.set_machine_status_error(str(exception))

    def close(self):
        self.timer.stop()
        self.controller.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="机械手指3D应力定位界面")
    parser.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT_PATH)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH_PATH)
    parser.add_argument("--press-tool", type=Path, default=DEFAULT_PRESS_TOOL_PATH)
    parser.add_argument(
        "--touch-calibration",
        type=Path,
        default=DEFAULT_TOUCH_CALIBRATION_PATH,
    )
    parser.add_argument("--demo", action="store_true")
    parser.add_argument(
        "--mach3-live",
        action="store_true",
        help="read-only live Mach3 press-tool animation; never sends movement commands",
    )
    parser.add_argument("--demo-seconds", type=float, default=0.0)
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args(argv)
    if args.demo and args.mach3_live:
        parser.error("--demo and --mach3-live are mutually exclusive")
    application = QtWidgets.QApplication(sys.argv[:1])
    window = MechanicalFinger3DWindow(
        args.layout,
        mesh_path=args.mesh,
        press_tool_path=args.press_tool,
        touch_calibration_path=args.touch_calibration,
    )
    window.show()
    demo = _DemoSource(window) if args.demo else None
    window._demo_source = demo
    mach3_live = _Mach3LiveSource(window) if args.mach3_live else None
    window._mach3_live_source = mach3_live
    if mach3_live is not None:
        application.aboutToQuit.connect(mach3_live.close)
    if args.screenshot is not None:

        def save_screenshot():
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            screen = application.primaryScreen()
            if screen is not None:
                screen.grabWindow(int(window.winId())).save(str(args.screenshot))

        QtCore.QTimer.singleShot(1800, save_screenshot)
    if args.demo_seconds > 0.0:

        def finish_demo():
            window.close()
            application.quit()

        QtCore.QTimer.singleShot(int(args.demo_seconds * 1000.0), finish_demo)
    return application.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
