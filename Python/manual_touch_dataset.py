"""Manual touch labels and native 45-point frame decoding.

This module is intentionally Qt-free so saved datasets and future time-series
training tools can validate labels/frames without starting the desktop UI.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


STRESS_POINT_COUNT = 45


@dataclass(frozen=True)
class ManualTouchLabel:
    location_id: str
    location_name: str
    model_x: float
    model_y: float
    stress_label: str = ""

    def __post_init__(self):
        location_id = str(self.location_id).strip()
        location_name = str(self.location_name).strip()
        model_x = float(self.model_x)
        model_y = float(self.model_y)
        if not location_id:
            raise ValueError("location_id must not be empty")
        if not location_name:
            raise ValueError("location_name must not be empty")
        if (
            not math.isfinite(model_x)
            or not math.isfinite(model_y)
            or not -1.0 <= model_x <= 1.0
            or not -1.0 <= model_y <= 1.0
        ):
            raise ValueError("manual model position must lie within -1..1")
        object.__setattr__(self, "location_id", location_id)
        object.__setattr__(self, "location_name", location_name)
        object.__setattr__(self, "model_x", model_x)
        object.__setattr__(self, "model_y", model_y)
        object.__setattr__(self, "stress_label", str(self.stress_label).strip())


def decode_native_ch1_stress_frame(payload: bytes) -> dict:
    """Decode the first 45×4 uint16 samples of one native EE EE scan frame."""

    raw = bytes(payload)
    if len(raw) < 4 or raw[:2] != b"\xEE\xEE":
        raise ValueError("不是原生光谱帧")
    point_count = int.from_bytes(raw[2:4], "big")
    if point_count != STRESS_POINT_COUNT:
        raise ValueError(f"时序采集只接受45点应力帧，收到{point_count}点")
    sample_end = 4 + point_count * 8
    if len(raw) < sample_end:
        raise ValueError("45点应力帧ADC数据不完整")
    channels: list[list[int]] = []
    for point in range(point_count):
        offset = 4 + point * 8
        channels.append([
            int.from_bytes(raw[offset + channel * 2:offset + channel * 2 + 2], "big")
            for channel in range(4)
        ])
    return {
        "point_count": point_count,
        "ch1_adc_codes": [row[1] for row in channels],
        "all_channels": channels,
    }


def trial_frame_quality(
    label: ManualTouchLabel,
    *,
    trial_id: str,
    phase: str,
    frame_index: int,
) -> dict:
    phase = str(phase).strip()
    if phase not in {"baseline", "pressed", "released"}:
        raise ValueError("manual phase must be baseline/pressed/released")
    if int(frame_index) < 0:
        raise ValueError("frame_index must be non-negative")
    return {
        "dataset_kind": "manual_touch_sequence",
        "trial_id": str(trial_id),
        "trial_frame_index": int(frame_index),
        "manual_phase": phase,
        "manual_location_id": label.location_id,
        "manual_location_name": label.location_name,
        "manual_model_xy": [label.model_x, label.model_y],
        "manual_stress_label": label.stress_label,
        "annotation_source": "operator_3d_model_click",
    }
