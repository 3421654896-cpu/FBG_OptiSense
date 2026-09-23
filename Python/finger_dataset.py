"""Loss-aware session storage for mechanical-finger FBG experiments.

This module deliberately contains no device-control code.  Acquisition code may
hand it immutable raw frames and CNC/contact annotations while a dedicated
thread performs the disk I/O.  ADC voltage is always derived from the original
ADC code.  ``digital_gain`` is retained for audit only and is never applied to
the stored or returned physical values.
"""

from __future__ import annotations

import json
import math
import queue
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any, Self

SCHEMA_VERSION = "fbg-finger-session/v1"
DEFAULT_SENSOR_CHANNEL = 1
DEFAULT_GRATING_COUNT = 9
DEFAULT_POINTS_PER_PEAK = 5
ADC_REFERENCE_V = 2.5
ADC_CODE_COUNT = 4096.0
ADC_MAX_CODE = 4095
MAX_INDENTATION_MM = 0.80


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    """Convert common numerical/container values to strict JSON values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return _jsonable(value.item())
    raise TypeError(f"value is not JSON serializable: {type(value)!r}")


def _channel_map(
    values: Any, *, name: str, item_converter
) -> dict[int, tuple[Any, ...]]:
    if not isinstance(values, Mapping):
        values = {DEFAULT_SENSOR_CHANNEL: values}
    result: dict[int, tuple[Any, ...]] = {}
    for channel, samples in values.items():
        channel_number = int(channel)
        if channel_number < 0 or channel_number > 3:
            raise ValueError(f"{name} channel must be in CH0..CH3")
        result[channel_number] = tuple(item_converter(sample) for sample in samples)
    if not result:
        raise ValueError(f"{name} must contain at least one channel")
    return result


def _adc_code(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("raw ADC codes must be integers, not booleans")
    converted = int(value)
    if isinstance(value, float) and (not math.isfinite(value) or value != converted):
        raise ValueError("raw ADC codes must be integers")
    return converted


def _numeric_channel_map(values: Any, *, name: str) -> dict[int, float | None]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        values = {DEFAULT_SENSOR_CHANNEL: values}
    result = {
        int(channel): None if value is None else float(value)
        for channel, value in values.items()
    }
    for channel, value in result.items():
        if channel < 0 or channel > 3:
            raise ValueError(f"{name} channel must be in CH0..CH3")
        if value is None:
            continue
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} values must be finite and positive")
    return result


def _optional_channel_int_map(values: Any, *, name: str) -> dict[int, int | None]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        values = {DEFAULT_SENSOR_CHANNEL: values}
    result = {
        int(channel): None if value is None else int(value)
        for channel, value in values.items()
    }
    for channel, value in result.items():
        if channel < 0 or channel > 3:
            raise ValueError(f"{name} channel must be in CH0..CH3")
        if value is not None and value < 0:
            raise ValueError(f"{name} values must be non-negative")
    return result


@dataclass(frozen=True)
class SpectrumFrame:
    """One unmodified received spectrum and the settings needed to audit it.

    ``raw_adc_codes`` accepts either ``{channel: samples}`` or a flat sequence;
    a flat sequence means CH1.  Codes outside the STM32 12-bit range are kept in
    JSON exactly as received, but :meth:`adc_voltage_v` returns ``None`` for
    those invalid samples. The legacy field name does NOT imply direct ADC:
    ``adc_value_kind`` distinguishes firmware RC estimates from direct samples.
    """

    sequence: int
    wavelengths_nm: Sequence[float]
    raw_adc_codes: Any
    monotonic_ns: int = field(default_factory=time.monotonic_ns)
    transimpedance_ohm: Any = field(default_factory=dict)
    digital_gain: Any = field(default_factory=dict)
    analogue_gain_mask: Any = field(default_factory=dict)
    feedback_selector: Any = field(default_factory=dict)
    table_crc32: int | None = None
    packet_crc_ok: bool | None = None
    device_sequence: int | None = None
    device_boot_id: int | None = None
    device_uptime_ms: int | None = None
    frame_received_monotonic_ns: int | None = None
    acquisition_profile: str | None = None
    schedule_version: int | None = None
    acquisition_profile_code: int | None = None
    adc_value_kind: str = "unknown"
    fresh_point_mask: Sequence[bool] | None = None
    point_age_frames: Sequence[int | None] | None = None
    sample_offset_us: Sequence[int | None] | None = None
    map_age_frames: int | None = None
    bandwidth_discontinuity: bool = False
    frame_start_device_ms: int | None = None
    quality_flags: Sequence[str] = field(default_factory=tuple)
    quality: Mapping[str, Any] = field(default_factory=dict)
    mode: str = "stress"
    source: str = "unknown"

    def __post_init__(self) -> None:
        if self.adc_value_kind not in {"unknown", "direct_adc", "firmware_rc_estimate"}:
            raise ValueError("unsupported adc_value_kind")
        if isinstance(self.sequence, bool) or int(self.sequence) < 0:
            raise ValueError("sequence must be a non-negative integer")
        if isinstance(self.monotonic_ns, bool) or int(self.monotonic_ns) < 0:
            raise ValueError("monotonic_ns must be a non-negative integer")
        wavelengths = tuple(float(value) for value in self.wavelengths_nm)
        if not wavelengths or not all(math.isfinite(value) for value in wavelengths):
            raise ValueError("wavelengths_nm must be non-empty and finite")
        channels = _channel_map(
            self.raw_adc_codes, name="raw_adc_codes", item_converter=_adc_code
        )
        for channel, samples in channels.items():
            if len(samples) != len(wavelengths):
                raise ValueError(
                    f"CH{channel} has {len(samples)} ADC samples for {len(wavelengths)} wavelengths"
                )
        transimpedance = _numeric_channel_map(
            self.transimpedance_ohm, name="transimpedance_ohm"
        )
        digital_gain = _numeric_channel_map(self.digital_gain, name="digital_gain")
        gain_mask = _optional_channel_int_map(
            self.analogue_gain_mask, name="analogue_gain_mask"
        )
        feedback_selector = _optional_channel_int_map(
            self.feedback_selector, name="feedback_selector"
        )
        if (
            self.table_crc32 is not None
            and not 0 <= int(self.table_crc32) <= 0xFFFFFFFF
        ):
            raise ValueError("table_crc32 must fit uint32")
        for name in (
            "device_sequence",
            "device_boot_id",
            "device_uptime_ms",
            "frame_received_monotonic_ns",
            "schedule_version",
            "acquisition_profile_code",
            "map_age_frames",
            "frame_start_device_ms",
        ):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or int(value) < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")

        point_count = len(wavelengths)
        fresh_mask = (
            None
            if self.fresh_point_mask is None
            else tuple(bool(value) for value in self.fresh_point_mask)
        )
        point_ages = (
            None
            if self.point_age_frames is None
            else tuple(None if value is None else int(value) for value in self.point_age_frames)
        )
        sample_offsets = (
            None
            if self.sample_offset_us is None
            else tuple(None if value is None else int(value) for value in self.sample_offset_us)
        )
        for name, values in (
            ("fresh_point_mask", fresh_mask),
            ("point_age_frames", point_ages),
            ("sample_offset_us", sample_offsets),
        ):
            if values is not None and len(values) != point_count:
                raise ValueError(f"{name} must contain one value per wavelength")
        if point_ages is not None and any(
            value is not None and value < 0 for value in point_ages
        ):
            raise ValueError("point_age_frames values must be non-negative or None")
        if sample_offsets is not None and any(
            value is not None and value < 0 for value in sample_offsets
        ):
            raise ValueError("sample_offset_us values must be non-negative or None")
        if fresh_mask is not None:
            if point_ages is not None:
                for index, fresh in enumerate(fresh_mask):
                    age = point_ages[index]
                    if fresh and age not in (None, 0):
                        raise ValueError("fresh points must have zero point_age_frames")
            # An all-None vector explicitly means that this otherwise valid
            # frame predates per-point timing telemetry.  Once any timestamp
            # is present, require a complete fresh/cached correlation.
            if sample_offsets is not None and any(
                value is not None for value in sample_offsets
            ):
                timed_fresh_offsets = []
                for index, fresh in enumerate(fresh_mask):
                    offset = sample_offsets[index]
                    if fresh and offset is None:
                        raise ValueError("fresh points must have sample_offset_us")
                    if not fresh and offset is not None:
                        raise ValueError("cached points must not claim sample_offset_us")
                    if fresh:
                        timed_fresh_offsets.append(int(offset))
                if any(
                    right <= left
                    for left, right in pairwise(timed_fresh_offsets)
                ):
                    raise ValueError("fresh sample_offset_us values must increase")

        object.__setattr__(self, "sequence", int(self.sequence))
        object.__setattr__(self, "monotonic_ns", int(self.monotonic_ns))
        object.__setattr__(self, "wavelengths_nm", wavelengths)
        object.__setattr__(self, "raw_adc_codes", channels)
        object.__setattr__(self, "transimpedance_ohm", transimpedance)
        object.__setattr__(self, "digital_gain", digital_gain)
        object.__setattr__(self, "analogue_gain_mask", gain_mask)
        object.__setattr__(self, "feedback_selector", feedback_selector)
        for name in (
            "device_sequence",
            "device_boot_id",
            "device_uptime_ms",
            "frame_received_monotonic_ns",
            "schedule_version",
            "acquisition_profile_code",
            "map_age_frames",
            "frame_start_device_ms",
        ):
            value = getattr(self, name)
            object.__setattr__(self, name, None if value is None else int(value))
        profile = (
            None
            if self.acquisition_profile is None
            else str(self.acquisition_profile).strip().upper()
        )
        object.__setattr__(self, "acquisition_profile", profile or None)
        object.__setattr__(self, "fresh_point_mask", fresh_mask)
        object.__setattr__(self, "point_age_frames", point_ages)
        object.__setattr__(self, "sample_offset_us", sample_offsets)
        object.__setattr__(
            self, "bandwidth_discontinuity", bool(self.bandwidth_discontinuity)
        )
        object.__setattr__(
            self, "quality_flags", tuple(str(v) for v in self.quality_flags)
        )
        object.__setattr__(self, "quality", dict(self.quality))
        object.__setattr__(self, "mode", str(self.mode))
        object.__setattr__(self, "source", str(self.source))

    def adc_voltage_v(
        self, channel: int = DEFAULT_SENSOR_CHANNEL
    ) -> tuple[float | None, ...]:
        """Return raw-code voltage; digital gain is intentionally ignored."""

        samples = self.raw_adc_codes[int(channel)]
        return tuple(
            code * ADC_REFERENCE_V / ADC_CODE_COUNT
            if 0 <= code <= ADC_MAX_CODE
            else None
            for code in samples
        )

    def to_record(self) -> dict[str, Any]:
        voltages = {
            str(channel): self.adc_voltage_v(channel)
            for channel in sorted(self.raw_adc_codes)
        }
        return {
            "record_type": "spectrum_frame",
            "sequence": self.sequence,
            "monotonic_ns": self.monotonic_ns,
            "mode": self.mode,
            "source": self.source,
            "wavelengths_nm": self.wavelengths_nm,
            "adc_value_kind": self.adc_value_kind,
            "raw_adc_codes": {
                str(channel): self.raw_adc_codes[channel]
                for channel in sorted(self.raw_adc_codes)
            },
            "adc_voltage_v": voltages,
            "transimpedance_ohm": {
                str(channel): value
                for channel, value in sorted(self.transimpedance_ohm.items())
            },
            # Metadata only.  It is never multiplied into adc_voltage_v.
            "digital_gain_audit": {
                str(channel): value
                for channel, value in sorted(self.digital_gain.items())
            },
            "analogue_gain_mask": {
                str(channel): value
                for channel, value in sorted(self.analogue_gain_mask.items())
            },
            "feedback_selector": {
                str(channel): value
                for channel, value in sorted(self.feedback_selector.items())
            },
            "table_crc32": self.table_crc32,
            "packet_crc_ok": self.packet_crc_ok,
            "device_sequence": self.device_sequence,
            "device_boot_id": self.device_boot_id,
            "device_uptime_ms": self.device_uptime_ms,
            "frame_received_monotonic_ns": self.frame_received_monotonic_ns,
            "acquisition_profile": self.acquisition_profile,
            "schedule_version": self.schedule_version,
            "acquisition_profile_code": self.acquisition_profile_code,
            "fresh_point_mask": self.fresh_point_mask,
            "point_age_frames": self.point_age_frames,
            "sample_offset_us": self.sample_offset_us,
            "map_age_frames": self.map_age_frames,
            "bandwidth_discontinuity": self.bandwidth_discontinuity,
            "frame_start_device_ms": self.frame_start_device_ms,
            "quality_flags": self.quality_flags,
            "quality": self.quality,
        }


def _xyz(value: Sequence[float] | None, name: str) -> tuple[float, float, float] | None:
    if value is None:
        return None
    result = tuple(float(component) for component in value)
    if len(result) != 3 or not all(math.isfinite(component) for component in result):
        raise ValueError(f"{name} must contain three finite coordinates")
    return result  # type: ignore[return-value]


def is_complete_fresh_map(frame: SpectrumFrame) -> bool:
    """Whether a legacy/full-MAP feature model may consume every stored point.

    Sparse profiles keep old values in the same 45-value container; its length
    alone is not evidence of 45 new measurements. Sparse inference has its own
    timestamp-aware path and must not be passed through a full-MAP model.
    """
    return (
        (frame.acquisition_profile or "MAP").strip().upper() in {"MAP", "MAP45", "LEGACY"}
        and (frame.fresh_point_mask is None or all(frame.fresh_point_mask))
        and (frame.point_age_frames is None or all(age in (0, None) for age in frame.point_age_frames))
    )


@dataclass(frozen=True)
class ContactEvent:
    """Ground-truth annotation for one motion/contact phase."""

    kind: str
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    monotonic_ns: int = field(default_factory=time.monotonic_ns)
    target_xyz_mm: Sequence[float] | None = None
    actual_xyz_mm: Sequence[float] | None = None
    indentation_mm: float | None = None
    indenter_diameter_mm: float | None = 8.0
    nominal_contact_area_mm2: float | None = None
    quality_flags: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.kind).strip():
            raise ValueError("kind must not be empty")
        if isinstance(self.monotonic_ns, bool) or int(self.monotonic_ns) < 0:
            raise ValueError("monotonic_ns must be a non-negative integer")
        indentation = (
            None if self.indentation_mm is None else float(self.indentation_mm)
        )
        if indentation is not None and (
            not math.isfinite(indentation)
            or indentation < 0.0
            or indentation > MAX_INDENTATION_MM
        ):
            raise ValueError("indentation_mm must be within 0.00..0.80 mm")
        diameter = (
            None
            if self.indenter_diameter_mm is None
            else float(self.indenter_diameter_mm)
        )
        if diameter is not None and (not math.isfinite(diameter) or diameter <= 0.0):
            raise ValueError("indenter_diameter_mm must be finite and positive")
        area = (
            None
            if self.nominal_contact_area_mm2 is None
            else float(self.nominal_contact_area_mm2)
        )
        if area is not None and (not math.isfinite(area) or area < 0.0):
            raise ValueError("nominal_contact_area_mm2 must be finite and non-negative")

        object.__setattr__(self, "kind", str(self.kind))
        object.__setattr__(self, "event_id", str(self.event_id))
        object.__setattr__(self, "monotonic_ns", int(self.monotonic_ns))
        object.__setattr__(
            self, "target_xyz_mm", _xyz(self.target_xyz_mm, "target_xyz_mm")
        )
        object.__setattr__(
            self, "actual_xyz_mm", _xyz(self.actual_xyz_mm, "actual_xyz_mm")
        )
        object.__setattr__(self, "indentation_mm", indentation)
        object.__setattr__(self, "indenter_diameter_mm", diameter)
        object.__setattr__(self, "nominal_contact_area_mm2", area)
        object.__setattr__(
            self, "quality_flags", tuple(str(v) for v in self.quality_flags)
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_record(self) -> dict[str, Any]:
        return {
            "record_type": "contact_event",
            "event_id": self.event_id,
            "kind": self.kind,
            "monotonic_ns": self.monotonic_ns,
            "target_xyz_mm": self.target_xyz_mm,
            "actual_xyz_mm": self.actual_xyz_mm,
            "indentation_mm": self.indentation_mm,
            "indenter_diameter_mm": self.indenter_diameter_mm,
            "nominal_contact_area_mm2": self.nominal_contact_area_mm2,
            "quality_flags": self.quality_flags,
            "metadata": self.metadata,
        }


class FingerSessionWriter:
    """Write a session through a bounded, loss-visible background queue."""

    def __init__(
        self,
        session_dir: Any,
        *,
        session_id: str | None = None,
        queue_capacity: int = 256,
        enqueue_timeout_s: float = 1.0,
        manifest_extra: Mapping[str, Any] | None = None,
    ) -> None:
        if int(queue_capacity) <= 0:
            raise ValueError("queue_capacity must be positive")
        if float(enqueue_timeout_s) < 0.0:
            raise ValueError("enqueue_timeout_s must be non-negative")
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.session_dir / "manifest.json"
        self.events_path = self.session_dir / "events.jsonl"
        self.frames_path = self.session_dir / "frames.jsonl"
        for path in (self.manifest_path, self.events_path, self.frames_path):
            if path.exists():
                raise FileExistsError(
                    f"refusing to overwrite existing session file: {path}"
                )
        self.events_path.touch()
        self.frames_path.touch()

        self.session_id = session_id or self.session_dir.name or uuid.uuid4().hex
        self.queue_capacity = int(queue_capacity)
        self.enqueue_timeout_s = float(enqueue_timeout_s)
        self._queue: queue.Queue = queue.Queue(maxsize=self.queue_capacity)
        self._sentinel = object()
        self._closed = False
        self._error: BaseException | None = None
        self._counts = {"frames": 0, "events": 0, "rejected_queue_full": 0}
        self._state_lock = threading.Lock()
        self._time_lock = threading.Lock()
        self._last_monotonic_ns = -1
        self._manifest: dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "session_id": self.session_id,
            "created_utc": _utc_now(),
            "timebase": "time.monotonic_ns",
            "sensor_channel": DEFAULT_SENSOR_CHANNEL,
            "grating_count": DEFAULT_GRATING_COUNT,
            "points_per_peak": DEFAULT_POINTS_PER_PEAK,
            "adc": {
                "reference_v": ADC_REFERENCE_V,
                "code_count": int(ADC_CODE_COUNT),
                "raw_codes_preserved": True,
            },
            "digital_gain_policy": "audit_only_not_applied_to_physical_values",
            "files": {"frames": "frames.jsonl", "events": "events.jsonl"},
            "writer": {"queue_capacity": self.queue_capacity},
            "counts": dict(self._counts),
        }
        if manifest_extra:
            self._manifest["metadata"] = dict(manifest_extra)
        self._write_manifest()
        self._thread = threading.Thread(
            target=self._worker, name="FingerSessionWriter", daemon=True
        )
        self._thread.start()

    @property
    def counts(self) -> dict[str, int]:
        with self._state_lock:
            return dict(self._counts)

    def _write_manifest(self) -> None:
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                _jsonable(self._manifest), ensure_ascii=False, indent=2, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.manifest_path)

    def _check_error(self) -> None:
        if self._error is not None:
            raise RuntimeError(
                "finger session background writer failed"
            ) from self._error

    def _enqueue(self, kind: str, record: Any, monotonic_ns: int) -> None:
        if self._closed:
            raise RuntimeError("finger session writer is closed")
        self._check_error()
        with self._time_lock:
            if monotonic_ns < self._last_monotonic_ns:
                raise ValueError(
                    "records must be submitted in non-decreasing monotonic_ns order"
                )
            self._last_monotonic_ns = monotonic_ns
            try:
                self._queue.put(
                    (kind, record), block=True, timeout=self.enqueue_timeout_s
                )
            except queue.Full as exc:
                with self._state_lock:
                    self._counts["rejected_queue_full"] += 1
                raise BufferError(
                    "finger session queue is full; frame was not written"
                ) from exc

    def write_frame(self, frame: SpectrumFrame) -> None:
        if not isinstance(frame, SpectrumFrame):
            raise TypeError("frame must be SpectrumFrame")
        self._enqueue("frame", frame.to_record(), frame.monotonic_ns)

    append_frame = write_frame
    submit_frame = write_frame

    def write_event(self, event: ContactEvent) -> None:
        if not isinstance(event, ContactEvent):
            raise TypeError("event must be ContactEvent")
        self._enqueue("event", event.to_record(), event.monotonic_ns)

    append_event = write_event
    submit_event = write_event

    def _worker(self) -> None:
        try:
            with (
                self.frames_path.open("a", encoding="utf-8", newline="\n") as frames,
                self.events_path.open("a", encoding="utf-8", newline="\n") as events,
            ):
                while True:
                    item = self._queue.get()
                    try:
                        if item is self._sentinel:
                            return
                        kind, record = item
                        if self._error is not None:
                            continue
                        destination = frames if kind == "frame" else events
                        destination.write(
                            json.dumps(
                                _jsonable(record),
                                ensure_ascii=False,
                                separators=(",", ":"),
                                allow_nan=False,
                            )
                            + "\n"
                        )
                        destination.flush()
                        with self._state_lock:
                            self._counts["frames" if kind == "frame" else "events"] += 1
                    except Exception as exc:  # noqa: BLE001 - cross-thread propagation
                        self._error = exc
                    finally:
                        self._queue.task_done()
        except Exception as exc:  # noqa: BLE001 - cross-thread propagation
            self._error = exc
            # Ensure callers blocked in flush do not deadlock after open failure.
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    self._queue.task_done()

    def flush(self) -> None:
        self._queue.join()
        self._check_error()

    def close(self) -> None:
        if self._closed:
            self._check_error()
            return
        # Always stop the worker, including after a disk/serialization failure.
        # Calling flush() here could raise before the sentinel is delivered.
        self._queue.join()
        self._queue.put(self._sentinel)
        self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            raise RuntimeError("finger session background writer did not stop")
        self._closed = True
        self._manifest["closed_utc"] = _utc_now()
        self._manifest["counts"] = self.counts
        self._write_manifest()
        self._check_error()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


__all__ = [
    "ADC_CODE_COUNT",
    "ADC_MAX_CODE",
    "ADC_REFERENCE_V",
    "DEFAULT_GRATING_COUNT",
    "DEFAULT_POINTS_PER_PEAK",
    "DEFAULT_SENSOR_CHANNEL",
    "MAX_INDENTATION_MM",
    "SCHEMA_VERSION",
    "ContactEvent",
    "FingerSessionWriter",
    "SpectrumFrame",
    "is_complete_fresh_map",
]
