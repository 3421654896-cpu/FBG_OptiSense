#!/usr/bin/env python3
"""Train and validate the CH1 nine-FBG finger localizer offline.

Examples::

    python train_finger_localizer.py session_001 session_002 \
        --output artifacts/finger_localizer.joblib --report artifacts/report.json

    python train_finger_localizer.py --synthetic-self-test

The script reads only captured session files.  It never opens a device, changes
laser current, sends a network command, or moves the CNC machine.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from finger_dataset import (
    DEFAULT_GRATING_COUNT,
    DEFAULT_POINTS_PER_PEAK,
    DEFAULT_SENSOR_CHANNEL,
    SpectrumFrame,
    is_complete_fresh_map,
)
from finger_features import CausalFeatureExtractor, extract_ch1_peak_features
from finger_model import (
    DEFAULT_POINT_COUNT,
    FingerModelBundle,
    FingerRuntimeSignature,
    FingerTrainingSample,
    ModelTrainingConfig,
    train_finger_model,
)

CONTACT_KINDS = {
    "contact",
    "contact_start",
    "indent",
    "loading",
    "hold",
    "press",
    "pressed",
}
NO_CONTACT_KINDS = {
    "baseline",
    "idle",
    "no_contact",
    "pre_contact",
    "release_complete",
    "released",
    "retracted",
}


@dataclass(frozen=True)
class AreaLabel:
    value_mm2: float | None
    source: str | None
    traceable: bool = False
    independent: bool = False
    measurement_id: str | None = None
    calibration_id: str | None = None


@dataclass(frozen=True)
class SessionLoadReport:
    session_id: str
    source: str
    frame_records: int
    event_records: int
    baseline_frames: int
    samples_created: int
    skipped_before_event: int
    skipped_warmup: int
    skipped_reduced_profile: int
    skipped_bad_peak_quality: int
    table_crc32: int | None
    baseline_frozen_at_monotonic_ns: int = 0
    skipped_baseline_calibration: int = 0
    skipped_unlabelled: int = 0
    skipped_timing_mismatch: int = 0
    causal_resets: int = 0
    label_timing_scope: str = "host annotation/receive clock; not ADC-synchronized contact truth"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def _parse_crc(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        parsed = int(value, 0)
    else:
        parsed = int(value)
    if not 0 <= parsed <= 0xFFFFFFFF:
        raise ValueError("table CRC must fit uint32")
    return parsed


def _event_contact(event: dict[str, Any]) -> bool | None:
    metadata = event.get("metadata") or {}
    if "contact" in metadata:
        value = metadata["contact"]
        if value is not None and not isinstance(value, bool):
            raise ValueError("metadata.contact must be a boolean or null (unknown)")
        return value
    kind = str(event.get("kind", "")).strip().lower().replace("-", "_")
    if kind in CONTACT_KINDS:
        return True
    if kind in NO_CONTACT_KINDS:
        return False
    return None


def _event_group(event: dict[str, Any]) -> str:
    metadata = event.get("metadata") or {}
    value = (
        metadata.get("press_event_id")
        or metadata.get("event_group")
    )
    # Old coverage captures used a new event_id for each phase of ONE press.
    # Keep all its frames in one validation fold, including release/baseline.
    if value is None and "point_index" in metadata and "repetition" in metadata:
        value = f"coverage-r{int(metadata['repetition'])}-p{int(metadata['point_index'])}"
    if value is None:
        value = event.get("event_id")
    if value is None or not str(value).strip():
        raise ValueError("every labelled event needs event_id or press_event_id")
    return str(value)


def _model_xy(event: dict[str, Any]) -> tuple[float | None, float | None]:
    metadata = event.get("metadata") or {}
    if "model_xy_mm" in metadata:
        value = metadata["model_xy_mm"]
        if not isinstance(value, Sequence) or len(value) < 2:
            raise ValueError("metadata.model_xy_mm must contain x and y")
        return float(value[0]), float(value[1])
    x_value = metadata.get("model_x_mm")
    y_value = metadata.get("model_y_mm")
    if x_value is None or y_value is None:
        return None, None
    return float(x_value), float(y_value)


def _area_label(
    event: dict[str, Any], contact: bool, event_group: str
) -> AreaLabel:
    if not contact:
        return AreaLabel(0.0, "no_contact")
    metadata = event.get("metadata") or {}
    candidates = (
        ("measured_contact_area_mm2", "measured"),
        ("calibrated_contact_area_mm2", "calibrated"),
        ("known_indenter_contact_area_mm2", "known_indenter_contact"),
    )
    area = None
    default_source = None
    for key, source in candidates:
        if metadata.get(key) is not None:
            area = float(metadata[key])
            default_source = source
            break
    if area is None and event.get("nominal_contact_area_mm2") is not None:
        return AreaLabel(
            float(event["nominal_contact_area_mm2"]), "nominal_indenter"
        )
    if area is None:
        return AreaLabel(None, None)

    provenance = metadata.get("area_truth_provenance")
    if provenance is None:
        return AreaLabel(area, default_source)
    if not isinstance(provenance, dict):
        raise ValueError("metadata.area_truth_provenance must be an object")
    independent = provenance.get("independent", False)
    if type(independent) is not bool:
        raise ValueError("area truth independent flag must be a boolean")
    source = str(provenance.get("source") or default_source or "").strip().lower()
    measurement_id = str(provenance.get("measurement_id") or "").strip()
    calibration_id = str(provenance.get("calibration_id") or "").strip()
    linked_event_id = str(provenance.get("linked_event_id") or "").strip()
    traceable = bool(
        source
        and measurement_id
        and calibration_id
        and linked_event_id == str(event_group)
    )
    return AreaLabel(
        area,
        source or default_source,
        traceable=traceable,
        independent=independent,
        measurement_id=measurement_id or None,
        calibration_id=calibration_id or None,
    )


def _channel_value(mapping: Any, channel: int, default: Any = None) -> Any:
    if not isinstance(mapping, dict):
        return default
    if str(channel) in mapping:
        return mapping[str(channel)]
    return mapping.get(channel, default)


def _record_to_frame(record: dict[str, Any]) -> SpectrumFrame:
    raw = record.get("raw_adc_codes")
    if not isinstance(raw, dict) or (
        str(DEFAULT_SENSOR_CHANNEL) not in raw
        and DEFAULT_SENSOR_CHANNEL not in raw
    ):
        raise ValueError("frame does not contain raw CH1 ADC codes")
    wavelengths = record.get("wavelengths_nm")
    if wavelengths is None or len(wavelengths) != DEFAULT_POINT_COUNT:
        actual = 0 if wavelengths is None else len(wavelengths)
        raise ValueError(
            f"session frame has {actual} wavelengths; CH1 model requires 45"
        )
    return SpectrumFrame(
        sequence=int(record.get("sequence", 0)),
        monotonic_ns=int(record["monotonic_ns"]),
        wavelengths_nm=wavelengths,
        raw_adc_codes=raw,
        transimpedance_ohm=record.get("transimpedance_ohm") or {},
        digital_gain=record.get("digital_gain_audit") or {},
        analogue_gain_mask=record.get("analogue_gain_mask") or {},
        feedback_selector=record.get("feedback_selector") or {},
        table_crc32=_parse_crc(record.get("table_crc32")),
        packet_crc_ok=record.get("packet_crc_ok"),
        device_sequence=record.get("device_sequence"),
        device_boot_id=record.get("device_boot_id"),
        device_uptime_ms=record.get("device_uptime_ms"),
        frame_received_monotonic_ns=record.get("frame_received_monotonic_ns"),
        acquisition_profile=record.get("acquisition_profile"),
        schedule_version=record.get("schedule_version"),
        acquisition_profile_code=record.get("acquisition_profile_code"),
        adc_value_kind=record.get("adc_value_kind", "unknown"),
        fresh_point_mask=record.get("fresh_point_mask"),
        point_age_frames=record.get("point_age_frames"),
        sample_offset_us=record.get("sample_offset_us"),
        map_age_frames=record.get("map_age_frames"),
        bandwidth_discontinuity=bool(record.get("bandwidth_discontinuity", False)),
        frame_start_device_ms=record.get("frame_start_device_ms"),
        quality_flags=record.get("quality_flags") or (),
        quality=record.get("quality") or {},
        mode=str(record.get("mode", "stress")),
        source=str(record.get("source", "session")),
    )


def _session_adc_value_kind(frames: Sequence[SpectrumFrame]) -> str:
    """Do not silently train on a mixture of direct values and RC estimates."""
    kinds = {frame.adc_value_kind for frame in frames}
    if len(kinds) != 1 or not kinds <= {"direct_adc", "firmware_rc_estimate"}:
        raise ValueError("session adc_value_kind must be one declared kind; mixed/unknown ADC provenance refused")
    return next(iter(kinds))


def _frame_label_time(frame: SpectrumFrame) -> int | None:
    received = frame.frame_received_monotonic_ns
    if received is not None:
        # UI snapshot time is not the time the stored ADC frame was received.
        return received if received <= frame.monotonic_ns else None
    if "coverage_phase" in frame.quality:
        return None  # A UI snapshot without receive provenance cannot be labelled.
    return frame.monotonic_ns  # Offline/direct producers declare this timebase.


def _stream_sequence(frame: SpectrumFrame) -> int:
    return frame.sequence if frame.device_sequence is None else frame.device_sequence


def _stream_break(previous: SpectrumFrame, current: SpectrumFrame) -> bool:
    return (
        current.device_boot_id != previous.device_boot_id
        or (current.device_sequence is None) != (previous.device_sequence is None)
        or _stream_sequence(current) != ((_stream_sequence(previous) + 1) & 0xFFFFFFFF)
        or _frame_label_time(current) <= _frame_label_time(previous)
        or current.bandwidth_discontinuity
    )


def load_session_training_samples(
    session_dir: str | Path,
    *,
    table_crc_override: int | None = None,
) -> tuple[list[FingerTrainingSample], FingerRuntimeSignature, SessionLoadReport]:
    """Convert one raw capture session into leakage-safe causal samples."""

    directory = Path(session_dir)
    manifest = json.loads((directory / "manifest.json").read_text("utf-8"))
    metadata = manifest.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"{directory}: session metadata must be an object")
    if "training_eligible" in metadata:
        eligible = metadata["training_eligible"]
        # bool('false') is True; malformed flags must never bypass quarantine.
        if type(eligible) is not bool:
            raise ValueError(f"{directory}: training_eligible must be a boolean")
        if not eligible:
            raise ValueError(f"{directory}: session marked ineligible for training (diagnostic/quarantined capture)")
    if int(manifest.get("sensor_channel", -1)) != DEFAULT_SENSOR_CHANNEL:
        raise ValueError(f"{directory}: expected sensor_channel CH1")
    if int(manifest.get("grating_count", -1)) != DEFAULT_GRATING_COUNT:
        raise ValueError(f"{directory}: expected exactly 9 gratings")
    if int(manifest.get("points_per_peak", -1)) != DEFAULT_POINTS_PER_PEAK:
        raise ValueError(f"{directory}: expected exactly 5 points per peak")

    session_id = str(manifest.get("session_id") or directory.name)
    frames = [_record_to_frame(value) for value in _read_jsonl(directory / "frames.jsonl")]
    frames.sort(key=lambda item: item.monotonic_ns)
    events = _read_jsonl(directory / "events.jsonl")
    events.sort(key=lambda item: int(item["monotonic_ns"]))
    labelled_events = [event for event in events if _event_contact(event) is not None]
    if not frames or not labelled_events:
        raise ValueError(f"{directory}: frames and labelled contact events are required")
    adc_value_kind = _session_adc_value_kind(frames)
    # Unknown/release/abort events are barriers, not permission to carry forward
    # the last known contact label indefinitely.
    event_times = [int(event["monotonic_ns"]) for event in events]

    event_for_frame: list[dict[str, Any] | None] = []
    for frame in frames:
        timestamp = _frame_label_time(frame)
        index = -1 if timestamp is None else bisect.bisect_right(event_times, timestamp) - 1
        event_for_frame.append(None if index < 0 else events[index])

    def timing_matches(frame, event):
        expected_event = frame.quality.get("phase_event_id")
        return _frame_label_time(frame) is not None and (
            expected_event is None or (event is not None and expected_event == event.get("event_id"))
        )

    first_contact = min(
        (int(event["monotonic_ns"]) for event in events
         if _event_contact(event) is True
         or str(event.get("kind", "")).lower() in CONTACT_KINDS | {"release", "release_started"}),
        default=float("inf"),
    )

    baseline_frames = [
        frame
        for frame, event in zip(frames, event_for_frame, strict=True)
        if event is not None
        and _event_contact(event) is False
        and timing_matches(frame, event)
        and _frame_label_time(frame) < first_contact
        and is_complete_fresh_map(frame)
        and not frame.bandwidth_discontinuity
        and bool(np.all(extract_ch1_peak_features(frame, baseline_frame=frame).valid_mask))
    ]
    unique_baseline = {}
    for frame in baseline_frames:
        key = (frame.source, frame.device_boot_id,
               frame.device_sequence is not None, _stream_sequence(frame))
        unique_baseline.setdefault(key, frame)
    baseline_frames = list(unique_baseline.values())
    if len(baseline_frames) < 3:
        raise ValueError(
            f"{directory}: at least three valid no-contact baseline frames before first contact are required"
        )
    baseline_frozen_at = max(_frame_label_time(frame) for frame in baseline_frames)
    reference_wavelengths = tuple(baseline_frames[0].wavelengths_nm)
    for frame in frames:
        if tuple(frame.wavelengths_nm) != reference_wavelengths:
            raise ValueError(f"{directory}: wavelength table changed within a session")
    baseline_codes = tuple(
        np.rint(
            np.median(
                np.asarray(
                    [
                        frame.raw_adc_codes[DEFAULT_SENSOR_CHANNEL]
                        for frame in baseline_frames
                    ],
                    dtype=float,
                ),
                axis=0,
            )
        )
        .astype(int)
        .tolist()
    )
    baseline_tias = {
        frame.transimpedance_ohm.get(DEFAULT_SENSOR_CHANNEL)
        for frame in baseline_frames
    }
    if len(baseline_tias) != 1:
        raise ValueError(f"{directory}: baseline transimpedance changed; capture a fixed-gain baseline")
    baseline_tia = next(iter(baseline_tias))

    observed_crcs = {frame.table_crc32 for frame in frames if frame.table_crc32 is not None}
    if len(observed_crcs) > 1:
        raise ValueError(f"{directory}: mixed wavelength-table CRCs in one session")
    observed_crc = next(iter(observed_crcs), None)
    if table_crc_override is not None:
        table_crc_override = _parse_crc(table_crc_override)
        if observed_crc is not None and table_crc_override != observed_crc:
            raise ValueError(
                f"{directory}: supplied table CRC does not match captured frames"
            )
        table_crc = table_crc_override
    else:
        table_crc = observed_crc

    extractor = CausalFeatureExtractor()
    previous_group = None
    previous_frame = None
    samples: list[FingerTrainingSample] = []
    skipped_before = skipped_warmup = skipped_reduced = skipped_quality = 0
    skipped_baseline = skipped_unlabelled = skipped_timing = causal_resets = 0
    for frame, event in zip(frames, event_for_frame, strict=True):
        if not timing_matches(frame, event):
            skipped_timing += 1
            extractor.reset()
            causal_resets += 1
            continue
        if event is None:
            skipped_before += 1
            extractor.reset()
            continue
        if _event_contact(event) is None:
            skipped_unlabelled += 1
            extractor.reset()
            causal_resets += 1
            continue
        if _frame_label_time(frame) <= baseline_frozen_at:
            # Calibration samples are not classifier negatives or causal history.
            skipped_baseline += 1
            continue
        # The classic 45-point feature trainer must never learn from cached
        # values carried by SURVEY/TRACK frames.  Sparse dynamic training uses
        # the dedicated causal multirate localizer instead.
        if not is_complete_fresh_map(frame):
            skipped_reduced += 1
            extractor.reset()
            causal_resets += 1
            continue
        if frame.bandwidth_discontinuity:
            skipped_quality += 1
            extractor.reset()
            causal_resets += 1
            continue
        contact = bool(_event_contact(event))
        group = _event_group(event)
        # Contact/hold can share a history, but contact/no-contact cannot.
        label_group = (group, contact)
        if label_group != previous_group:
            extractor.reset()
            previous_group = label_group
        if previous_frame is not None and _stream_break(previous_frame, frame):
            extractor.reset()
            causal_resets += 1
            if (frame.device_boot_id == previous_frame.device_boot_id
                    and _stream_sequence(frame) == _stream_sequence(previous_frame)):
                continue  # A repeated UI/device frame is not a fresh observation.
        previous_frame = frame
        feature_frame = extract_ch1_peak_features(
            frame,
            baseline_adc_codes=baseline_codes,
            baseline_transimpedance_ohm=baseline_tia,
        )
        # Training is strict: all nine local peaks must be valid.  Deployment
        # may tolerate two invalid peaks, but it must not learn from corrupt or
        # saturated examples.
        if not bool(np.all(feature_frame.valid_mask)):
            skipped_quality += 1
            extractor.reset()
            continue
        window = extractor.update(replace(feature_frame, monotonic_ns=_frame_label_time(frame)))
        if not window.ready:
            skipped_warmup += 1
            continue
        x_value, y_value = _model_xy(event)
        area = _area_label(event, contact, group)
        samples.append(
            FingerTrainingSample.from_window(
                window,
                session_id=session_id,
                event_id=group,
                contact=contact,
                model_x_mm=x_value,
                model_y_mm=y_value,
                contact_area_mm2=area.value_mm2,
                area_label_source=area.source,
                area_label_traceable=area.traceable,
                area_label_independent=area.independent,
                area_measurement_id=area.measurement_id,
                area_calibration_id=area.calibration_id,
                indenter_diameter_mm=event.get("indenter_diameter_mm"),
            )
        )

    signature = FingerRuntimeSignature(table_crc32=table_crc, adc_value_kind=adc_value_kind)
    report = SessionLoadReport(
        session_id=session_id,
        source=str(directory.resolve()),
        frame_records=len(frames),
        event_records=len(events),
        baseline_frames=len(baseline_frames),
        samples_created=len(samples),
        skipped_before_event=skipped_before,
        skipped_warmup=skipped_warmup,
        skipped_reduced_profile=skipped_reduced,
        skipped_bad_peak_quality=skipped_quality,
        table_crc32=table_crc,
        baseline_frozen_at_monotonic_ns=baseline_frozen_at,
        skipped_baseline_calibration=skipped_baseline,
        skipped_unlabelled=skipped_unlabelled,
        skipped_timing_mismatch=skipped_timing,
        causal_resets=causal_resets,
    )
    return samples, signature, report


def generate_synthetic_samples(
    *,
    fixed_eight_mm_area: bool,
    seed: int = 20260905,
    session_count: int = 8,
    press_events_per_session: int = 18,
) -> list[FingerTrainingSample]:
    """Create deterministic offline-only data with complete grouped events."""

    rng = np.random.default_rng(seed)
    sensor_xy = np.asarray(
        [
            (-10.0, 12.0),
            (0.0, 12.0),
            (10.0, 12.0),
            (-10.0, 4.0),
            (0.0, 4.0),
            (10.0, 4.0),
            (-10.0, -4.0),
            (0.0, -4.0),
            (10.0, -4.0),
        ],
        dtype=float,
    )
    records: list[FingerTrainingSample] = []

    def vector_for(
        x_value: float,
        y_value: float,
        diameter: float,
        contact: bool,
        session_bias: np.ndarray,
    ) -> np.ndarray:
        sigma = 4.0 + 0.45 * diameter
        distance_sq = np.sum(
            (sensor_xy - np.asarray((x_value, y_value))) ** 2, axis=1
        )
        spatial = np.exp(-0.5 * distance_sq / sigma**2) if contact else np.zeros(9)
        factors = (0.45, 0.78, 1.0) if contact else (0.0, 0.0, 0.0)
        frames = []
        for factor in factors:
            response = factor * spatial
            values = np.stack(
                (
                    12.0 * response,
                    -0.25 * response,
                    0.16 * response * diameter / 8.0,
                    0.30 * response + 0.04 * (x_value / 12.0),
                    -0.28 * response + 0.04 * (y_value / 14.0),
                    0.10 * response * (x_value / 12.0),
                ),
                axis=1,
            )
            noise = rng.normal(0.0, (0.035 if contact else 0.018), values.shape)
            frames.append(values + noise + session_bias)
        current = frames[-1]
        first_difference = frames[-1] - frames[-2]
        second_difference = frames[-1] - 2.0 * frames[-2] + frames[-3]
        return np.concatenate(
            (current, first_difference, second_difference), axis=1
        ).reshape(-1)

    diameter_choices = (4.0, 6.0, 8.0)
    for session_index in range(session_count):
        session = f"synthetic-session-{session_index:02d}"
        session_bias = rng.normal(0.0, 0.008, (9, 6))
        for idle_index in range(max(4, press_events_per_session // 4)):
            event = f"idle-{idle_index:03d}"
            for _ in range(3):
                records.append(
                    FingerTrainingSample(
                        vector_for(0.0, 0.0, 8.0, False, session_bias),
                        session_id=session,
                        event_id=event,
                        contact=False,
                    )
                )
        for press_index in range(press_events_per_session):
            x_value = float(rng.uniform(-10.5, 10.5))
            y_value = float(rng.uniform(-4.5, 12.5))
            diameter = (
                8.0
                if fixed_eight_mm_area
                else diameter_choices[press_index % len(diameter_choices)]
            )
            area = math.pi * (diameter / 2.0) ** 2
            event = f"press-{press_index:03d}"
            for _ in range(3):
                records.append(
                    FingerTrainingSample(
                        vector_for(
                            x_value, y_value, diameter, True, session_bias
                        ),
                        session_id=session,
                        event_id=event,
                        contact=True,
                        model_x_mm=x_value,
                        model_y_mm=y_value,
                        contact_area_mm2=area,
                        area_label_source="measured",
                        area_label_traceable=True,
                        area_label_independent=True,
                        area_measurement_id=(
                            f"synthetic-area-{session_index:02d}-{press_index:03d}"
                        ),
                        area_calibration_id="synthetic-calibration-v1",
                        indenter_diameter_mm=diameter,
                    )
                )
    return records


def run_synthetic_self_test() -> dict[str, Any]:
    """Exercise training, area refusal, save/load, and CRC checks offline."""

    signature = FingerRuntimeSignature(table_crc32=0x91A2B3C4, adc_value_kind="direct_adc")
    config = ModelTrainingConfig(
        n_estimators=72,
        min_samples_leaf=1,
        inference_benchmark_repeats=60,
    )
    variable = generate_synthetic_samples(fixed_eight_mm_area=False)
    bundle, report = train_finger_model(variable, signature, config=config)
    if not bundle.area_model_available:
        raise AssertionError("variable synthetic areas should be identifiable")
    fixed = generate_synthetic_samples(fixed_eight_mm_area=True, seed=20260906)
    fixed_bundle, fixed_report = train_finger_model(fixed, signature, config=config)
    if fixed_bundle.area_model_available:
        raise AssertionError("single 8 mm synthetic area must be rejected")
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "finger_model.joblib"
        bundle.save(path)
        loaded = FingerModelBundle.load(path)
        prediction = loaded.predict_vector(variable[-1].features, signature)
    return {
        "status": "ok",
        "variable_area": {
            "metrics": report["metrics"],
            "roundtrip_prediction": asdict(prediction),
        },
        "fixed_8mm_area": {
            "area_policy": fixed_report["area_policy"],
            "area_model_available": fixed_bundle.area_model_available,
        },
    }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="*", help="captured session directories")
    parser.add_argument("--output", type=Path, help="destination .joblib bundle")
    parser.add_argument("--report", type=Path, help="destination JSON report")
    parser.add_argument(
        "--table-crc32",
        type=lambda value: int(value, 0),
        help="expected uint32 table CRC, e.g. 0x1234ABCD",
    )
    parser.add_argument("--n-estimators", type=int, default=64)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument(
        "--split-level", choices=("auto", "session", "event"), default="auto"
    )
    parser.add_argument(
        "--require-area-model",
        action="store_true",
        help="fail instead of training only contact/position if area is unidentifiable",
    )
    parser.add_argument(
        "--allow-unpinned-table",
        action="store_true",
        help="allow an audit-only bundle without a captured/supplied table CRC",
    )
    parser.add_argument(
        "--synthetic-self-test",
        action="store_true",
        help="run deterministic offline tests without reading devices or sessions",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    if args.synthetic_self_test:
        print(json.dumps(run_synthetic_self_test(), ensure_ascii=False, indent=2))
        return 0
    if not args.sessions:
        raise SystemExit("at least one session directory is required")
    if args.output is None:
        raise SystemExit("--output is required")

    all_samples: list[FingerTrainingSample] = []
    load_reports = []
    signatures = []
    for session in args.sessions:
        samples, signature, report = load_session_training_samples(
            session, table_crc_override=args.table_crc32
        )
        all_samples.extend(samples)
        signatures.append(signature)
        load_reports.append(asdict(report))
    observed_crcs = {signature.table_crc32 for signature in signatures}
    if len(observed_crcs) != 1:
        raise SystemExit("all training sessions must use the same wavelength-table CRC")
    if len({s.adc_value_kind for s in signatures}) != 1:
        raise SystemExit("all training sessions must use the same adc_value_kind")
    signature = signatures[0]
    if signature.table_crc32 is None and not args.allow_unpinned_table:
        raise SystemExit(
            "captured frames have no table CRC; supply --table-crc32 or explicitly "
            "use --allow-unpinned-table for a non-deployable audit bundle"
        )

    config = ModelTrainingConfig(
        n_estimators=args.n_estimators,
        test_size=args.test_size,
        random_state=args.seed,
        split_level=args.split_level,
    )
    bundle, report = train_finger_model(
        all_samples,
        signature,
        config=config,
        require_area_model=args.require_area_model,
    )
    report["session_loading"] = load_reports
    bundle.training_report = report
    bundle.save(args.output)
    report_path = args.report or args.output.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved model: {args.output.resolve()}", file=sys.stderr)
    print(f"saved report: {report_path.resolve()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
