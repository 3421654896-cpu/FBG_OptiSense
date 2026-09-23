"""Wire protocol for remote FBG scan telemetry and mode control.

Telemetry is deliberately independent from MQTT so the same encoder/decoder can
be used on the STM32-to-WiFi UART, in tests, and by a remote viewer.  All integer
fields are network byte order (big endian).  ADC samples are point-major: for
each wavelength point, values for the channels selected by ``channel_mask`` are
stored in ascending channel-number order.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Sequence, Union
import zlib


MAGIC = b"FBG1"
METADATA_MAGIC = b"FBGM"
PROTOCOL_VERSION = 1
TELEMETRY_VERSION = 2
MESSAGE_TYPE_TELEMETRY = 1
LEGACY_HEADER_SIZE = 40
HEADER_SIZE = 56
CRC_SIZE = 4
SAMPLE_FORMAT_UINT16_BE = 1
MAX_POINT_COUNT = 2500
VALID_CHANNEL_MASK = 0x0F

MODE_STRESS = 0
MODE_TEMPERATURE = 3
VALID_MODES = (MODE_STRESS, MODE_TEMPERATURE)

ACK_ACCEPTED = "ACCEPTED"
ACK_APPLIED = "APPLIED"
ACK_REJECTED = "REJECTED"
VALID_ACK_STATUSES = (ACK_ACCEPTED, ACK_APPLIED, ACK_REJECTED)

_MODE_TO_NAME = {
    MODE_STRESS: "STRESS",
    MODE_TEMPERATURE: "TEMPERATURE",
}
_NAME_TO_MODE = {name: mode for mode, name in _MODE_TO_NAME.items()}

_HEADER_V1 = struct.Struct(">4sBBBBHHHBBIIIiIHH")
_HEADER_V2 = struct.Struct(">4sBBBBHHHBBIIIiIHHHHIHHHBB")
_METADATA_HEADER = struct.Struct(">4sBBHI")
_UINT16 = struct.Struct(">H")
_UINT32 = struct.Struct(">I")


class ProtocolError(ValueError):
    """Raised when a remote-protocol message is malformed or inconsistent."""


@dataclass(frozen=True)
class TelemetryFrame:
    """One raw wavelength scan.

    ``samples`` is point-major.  With ``channel_mask == 0x03``, each point is
    represented as ``(ch0_code, ch1_code)``.
    """

    mode: int
    flags: int
    boot_id: int
    seq: int
    uptime_ms: int
    temperature_mC: int
    table_crc32: int
    gainmask0: int
    gainmask1: int
    channel_mask: int
    samples: Sequence[Sequence[int]]
    pd_voltage_mV: int = 0
    pd_current_mA: int = 0
    pd_power_limit_mW: int = 0
    fan_rpm: int = 0
    fan_duty_permille: int = 0
    thermal_flags: int = 0
    pd_flags: int = 0
    pd_protocol: int = 0

    @property
    def point_count(self) -> int:
        return len(self.samples)

    @property
    def selected_channels(self) -> tuple[int, ...]:
        return selected_channels(self.channel_mask)


@dataclass(frozen=True)
class ModeCommand:
    command_id: int
    mode: int


@dataclass(frozen=True)
class ModeAck:
    command_id: int
    status: str
    mode: int
    applied_seq: int


@dataclass(frozen=True)
class MetadataFrame:
    """Wavelength coordinates associated with a telemetry table revision.

    ``table_crc32`` is the table revision identifier supplied by the firmware.
    It is not recomputed from the wavelength array: a calibration revision may
    include DAC/current data that is intentionally absent from this compact
    metadata message.  The trailing packet CRC protects the complete metadata
    message itself.
    """

    mode: int
    table_crc32: int
    wavelength_pm: Sequence[int]

    @property
    def point_count(self) -> int:
        return len(self.wavelength_pm)


def selected_channels(channel_mask: int) -> tuple[int, ...]:
    _require_uint("channel_mask", channel_mask, 8)
    if channel_mask == 0 or channel_mask & ~VALID_CHANNEL_MASK:
        raise ProtocolError(
            f"channel_mask must select one or more of CH0-CH3, got 0x{channel_mask:02X}"
        )
    return tuple(index for index in range(4) if channel_mask & (1 << index))


def _require_uint(name: str, value: int, bits: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{name} must be an integer")
    if value < 0 or value >= (1 << bits):
        raise ProtocolError(f"{name} must fit uint{bits}")


def _require_int32(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{name} must be an integer")
    if value < -(1 << 31) or value >= (1 << 31):
        raise ProtocolError(f"{name} must fit int32")


def _validate_mode(mode: int) -> None:
    if isinstance(mode, bool) or not isinstance(mode, int):
        raise ProtocolError("mode must be an integer")
    if mode not in VALID_MODES:
        raise ProtocolError(f"unsupported mode {mode}; expected 0 or 3")


def _validate_frame(frame: TelemetryFrame) -> tuple[int, tuple[int, ...]]:
    _validate_mode(frame.mode)
    _require_uint("flags", frame.flags, 8)
    _require_uint("boot_id", frame.boot_id, 32)
    _require_uint("seq", frame.seq, 32)
    _require_uint("uptime_ms", frame.uptime_ms, 32)
    _require_int32("temperature_mC", frame.temperature_mC)
    _require_uint("table_crc32", frame.table_crc32, 32)
    _require_uint("gainmask0", frame.gainmask0, 16)
    _require_uint("gainmask1", frame.gainmask1, 16)
    _require_uint("pd_voltage_mV", frame.pd_voltage_mV, 16)
    _require_uint("pd_current_mA", frame.pd_current_mA, 16)
    _require_uint("pd_power_limit_mW", frame.pd_power_limit_mW, 32)
    _require_uint("fan_rpm", frame.fan_rpm, 16)
    _require_uint("fan_duty_permille", frame.fan_duty_permille, 16)
    _require_uint("thermal_flags", frame.thermal_flags, 16)
    _require_uint("pd_flags", frame.pd_flags, 8)
    _require_uint("pd_protocol", frame.pd_protocol, 8)
    channels = selected_channels(frame.channel_mask)

    point_count = len(frame.samples)
    if point_count <= 0 or point_count > MAX_POINT_COUNT:
        raise ProtocolError(
            f"point_count must be in 1..{MAX_POINT_COUNT}, got {point_count}"
        )
    for point_index, point in enumerate(frame.samples):
        if len(point) != len(channels):
            raise ProtocolError(
                f"point {point_index} has {len(point)} samples, expected {len(channels)}"
            )
        for channel_offset, code in enumerate(point):
            _require_uint(
                f"samples[{point_index}][{channel_offset}]", code, 16
            )
    return point_count, channels


def encode_telemetry(frame: TelemetryFrame) -> bytes:
    """Encode a telemetry frame and append a zlib/IEEE CRC-32."""

    point_count, channels = _validate_frame(frame)
    payload_len = point_count * len(channels) * _UINT16.size
    header = _HEADER_V2.pack(
        MAGIC,
        TELEMETRY_VERSION,
        MESSAGE_TYPE_TELEMETRY,
        frame.mode,
        frame.flags,
        HEADER_SIZE,
        payload_len,
        point_count,
        frame.channel_mask,
        SAMPLE_FORMAT_UINT16_BE,
        frame.boot_id,
        frame.seq,
        frame.uptime_ms,
        frame.temperature_mC,
        frame.table_crc32,
        frame.gainmask0,
        frame.gainmask1,
        frame.pd_voltage_mV,
        frame.pd_current_mA,
        frame.pd_power_limit_mW,
        frame.fan_rpm,
        frame.fan_duty_permille,
        frame.thermal_flags,
        frame.pd_flags,
        frame.pd_protocol,
    )
    payload = bytearray(payload_len)
    offset = 0
    for point in frame.samples:
        for code in point:
            _UINT16.pack_into(payload, offset, code)
            offset += _UINT16.size

    message_without_crc = header + payload
    crc = zlib.crc32(message_without_crc) & 0xFFFFFFFF
    return message_without_crc + _UINT32.pack(crc)


def decode_telemetry(data: Union[bytes, bytearray, memoryview]) -> TelemetryFrame:
    """Strictly validate and decode exactly one telemetry frame."""

    raw = bytes(data)
    if len(raw) < LEGACY_HEADER_SIZE + CRC_SIZE:
        raise ProtocolError("telemetry frame is truncated before header/CRC")

    (
        magic,
        version,
        message_type,
        mode,
        flags,
        header_len,
        payload_len,
        point_count,
        channel_mask,
        sample_format,
        boot_id,
        seq,
        uptime_ms,
        temperature_mC,
        table_crc32,
        gainmask0,
        gainmask1,
    ) = _HEADER_V1.unpack_from(raw)

    if magic != MAGIC:
        raise ProtocolError("invalid telemetry magic")
    if version not in (PROTOCOL_VERSION, TELEMETRY_VERSION):
        raise ProtocolError(f"unsupported protocol version {version}")
    if message_type != MESSAGE_TYPE_TELEMETRY:
        raise ProtocolError(f"unsupported message type {message_type}")
    _validate_mode(mode)
    expected_header_len = (
        LEGACY_HEADER_SIZE if version == PROTOCOL_VERSION else HEADER_SIZE
    )
    if header_len != expected_header_len:
        raise ProtocolError(
            f"header_len must be {expected_header_len} for version {version}, got {header_len}"
        )

    pd_voltage_mV = 0
    pd_current_mA = 0
    pd_power_limit_mW = 0
    fan_rpm = 0
    fan_duty_permille = 0
    thermal_flags = 0
    pd_flags = 0
    pd_protocol = 0
    if version == TELEMETRY_VERSION:
        if len(raw) < HEADER_SIZE + CRC_SIZE:
            raise ProtocolError("telemetry v2 frame is truncated before header/CRC")
        extended = _HEADER_V2.unpack_from(raw)
        (
            pd_voltage_mV,
            pd_current_mA,
            pd_power_limit_mW,
            fan_rpm,
            fan_duty_permille,
            thermal_flags,
            pd_flags,
            pd_protocol,
        ) = extended[-8:]
    if sample_format != SAMPLE_FORMAT_UINT16_BE:
        raise ProtocolError(f"unsupported sample format {sample_format}")
    if point_count <= 0 or point_count > MAX_POINT_COUNT:
        raise ProtocolError(f"invalid point_count {point_count}")

    channels = selected_channels(channel_mask)
    expected_payload_len = point_count * len(channels) * _UINT16.size
    if payload_len != expected_payload_len:
        raise ProtocolError(
            f"payload_len {payload_len} does not match point/channel counts "
            f"({expected_payload_len})"
        )
    expected_total_len = header_len + payload_len + CRC_SIZE
    if len(raw) != expected_total_len:
        raise ProtocolError(
            f"frame length {len(raw)} does not match declared length {expected_total_len}"
        )

    expected_crc = _UINT32.unpack_from(raw, len(raw) - CRC_SIZE)[0]
    actual_crc = zlib.crc32(raw[:-CRC_SIZE]) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise ProtocolError(
            f"CRC32 mismatch: received {expected_crc:08X}, calculated {actual_crc:08X}"
        )

    samples = []
    offset = header_len
    for _ in range(point_count):
        point = []
        for _channel in channels:
            point.append(_UINT16.unpack_from(raw, offset)[0])
            offset += _UINT16.size
        samples.append(tuple(point))

    return TelemetryFrame(
        mode=mode,
        flags=flags,
        boot_id=boot_id,
        seq=seq,
        uptime_ms=uptime_ms,
        temperature_mC=temperature_mC,
        table_crc32=table_crc32,
        gainmask0=gainmask0,
        gainmask1=gainmask1,
        channel_mask=channel_mask,
        samples=tuple(samples),
        pd_voltage_mV=pd_voltage_mV,
        pd_current_mA=pd_current_mA,
        pd_power_limit_mW=pd_power_limit_mW,
        fan_rpm=fan_rpm,
        fan_duty_permille=fan_duty_permille,
        thermal_flags=thermal_flags,
        pd_flags=pd_flags,
        pd_protocol=pd_protocol,
    )


def encode_metadata(frame: MetadataFrame) -> bytes:
    """Encode an ``FBGM`` wavelength metadata packet with IEEE CRC-32."""

    _validate_mode(frame.mode)
    _require_uint("table_crc32", frame.table_crc32, 32)
    point_count = len(frame.wavelength_pm)
    if point_count <= 0 or point_count > MAX_POINT_COUNT:
        raise ProtocolError(
            f"point_count must be in 1..{MAX_POINT_COUNT}, got {point_count}"
        )

    payload = bytearray(point_count * _UINT32.size)
    for index, wavelength_pm in enumerate(frame.wavelength_pm):
        _require_uint(f"wavelength_pm[{index}]", wavelength_pm, 32)
        _UINT32.pack_into(payload, index * _UINT32.size, wavelength_pm)

    message_without_crc = _METADATA_HEADER.pack(
        METADATA_MAGIC,
        PROTOCOL_VERSION,
        frame.mode,
        point_count,
        frame.table_crc32,
    ) + payload
    crc = zlib.crc32(message_without_crc) & 0xFFFFFFFF
    return message_without_crc + _UINT32.pack(crc)


def decode_metadata(
    data: Union[bytes, bytearray, memoryview]
) -> MetadataFrame:
    """Strictly validate and decode exactly one ``FBGM`` metadata packet."""

    raw = bytes(data)
    minimum_size = _METADATA_HEADER.size + CRC_SIZE
    if len(raw) < minimum_size:
        raise ProtocolError("metadata packet is truncated before header/CRC")

    magic, version, mode, point_count, table_crc32 = _METADATA_HEADER.unpack_from(raw)
    if magic != METADATA_MAGIC:
        raise ProtocolError("invalid metadata magic")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported metadata version {version}")
    _validate_mode(mode)
    if point_count <= 0 or point_count > MAX_POINT_COUNT:
        raise ProtocolError(f"invalid metadata point_count {point_count}")

    expected_length = _METADATA_HEADER.size + point_count * _UINT32.size + CRC_SIZE
    if len(raw) != expected_length:
        raise ProtocolError(
            f"metadata length {len(raw)} does not match declared length {expected_length}"
        )

    expected_crc = _UINT32.unpack_from(raw, len(raw) - CRC_SIZE)[0]
    actual_crc = zlib.crc32(raw[:-CRC_SIZE]) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise ProtocolError(
            f"metadata CRC32 mismatch: received {expected_crc:08X}, "
            f"calculated {actual_crc:08X}"
        )

    wavelengths = tuple(
        _UINT32.unpack_from(raw, _METADATA_HEADER.size + index * _UINT32.size)[0]
        for index in range(point_count)
    )
    return MetadataFrame(
        mode=mode,
        table_crc32=table_crc32,
        wavelength_pm=wavelengths,
    )


def _decode_ascii(data: Union[str, bytes, bytearray, memoryview], kind: str) -> str:
    if isinstance(data, str):
        text = data
    else:
        try:
            text = bytes(data).decode("ascii")
        except UnicodeDecodeError as exc:
            raise ProtocolError(f"{kind} must contain ASCII only") from exc
    if not text or text != text.strip():
        raise ProtocolError(f"{kind} must not contain leading/trailing whitespace")
    return text


def _parse_hex_u32(value: str, field_name: str) -> int:
    if len(value) != 8 or any(ch not in "0123456789ABCDEF" for ch in value):
        raise ProtocolError(f"{field_name} must be exactly 8 uppercase hex digits")
    return int(value, 16)


def encode_mode_command(
    command: Union[ModeCommand, int], mode: int | None = None
) -> bytes:
    """Encode ``FCMD1|XXXXXXXX|MODE|STRESS`` (or ``TEMPERATURE``)."""

    if isinstance(command, ModeCommand):
        if mode is not None:
            raise ProtocolError("mode must be omitted when passing ModeCommand")
        command_id = command.command_id
        selected_mode = command.mode
    else:
        command_id = command
        if mode is None:
            raise ProtocolError("mode is required")
        selected_mode = mode
    _require_uint("command_id", command_id, 32)
    _validate_mode(selected_mode)
    return (
        f"FCMD1|{command_id:08X}|MODE|{_MODE_TO_NAME[selected_mode]}".encode("ascii")
    )


def decode_mode_command(
    data: Union[str, bytes, bytearray, memoryview]
) -> ModeCommand:
    text = _decode_ascii(data, "mode command")
    parts = text.split("|")
    if len(parts) != 4 or parts[0] != "FCMD1" or parts[2] != "MODE":
        raise ProtocolError("invalid mode command format")
    command_id = _parse_hex_u32(parts[1], "command_id")
    try:
        mode = _NAME_TO_MODE[parts[3]]
    except KeyError as exc:
        raise ProtocolError(f"unsupported mode name {parts[3]!r}") from exc
    return ModeCommand(command_id=command_id, mode=mode)


def encode_mode_ack(
    ack: Union[ModeAck, int],
    status: str | None = None,
    mode: int | None = None,
    applied_seq: int | None = None,
) -> bytes:
    """Encode a mode acknowledgement in the ``FACK1`` ASCII format."""

    if isinstance(ack, ModeAck):
        if status is not None or mode is not None or applied_seq is not None:
            raise ProtocolError("extra fields must be omitted when passing ModeAck")
        command_id = ack.command_id
        selected_status = ack.status
        selected_mode = ack.mode
        selected_seq = ack.applied_seq
    else:
        command_id = ack
        if status is None or mode is None or applied_seq is None:
            raise ProtocolError("status, mode and applied_seq are required")
        selected_status = status
        selected_mode = mode
        selected_seq = applied_seq

    _require_uint("command_id", command_id, 32)
    if selected_status not in VALID_ACK_STATUSES:
        raise ProtocolError(f"unsupported ACK status {selected_status!r}")
    _validate_mode(selected_mode)
    _require_uint("applied_seq", selected_seq, 32)
    return (
        f"FACK1|{command_id:08X}|{selected_status}|"
        f"{_MODE_TO_NAME[selected_mode]}|{selected_seq:08X}"
    ).encode("ascii")


def decode_mode_ack(data: Union[str, bytes, bytearray, memoryview]) -> ModeAck:
    text = _decode_ascii(data, "mode acknowledgement")
    parts = text.split("|")
    if len(parts) != 5 or parts[0] != "FACK1":
        raise ProtocolError("invalid mode acknowledgement format")
    command_id = _parse_hex_u32(parts[1], "command_id")
    status = parts[2]
    if status not in VALID_ACK_STATUSES:
        raise ProtocolError(f"unsupported ACK status {status!r}")
    try:
        mode = _NAME_TO_MODE[parts[3]]
    except KeyError as exc:
        raise ProtocolError(f"unsupported mode name {parts[3]!r}") from exc
    applied_seq = _parse_hex_u32(parts[4], "applied_seq")
    return ModeAck(
        command_id=command_id,
        status=status,
        mode=mode,
        applied_seq=applied_seq,
    )


# Concise aliases for callers that treat telemetry as the only binary frame.
encode_frame = encode_telemetry
decode_frame = decode_telemetry
encode_ack = encode_mode_ack
decode_ack = decode_mode_ack


__all__ = [
    "ACK_ACCEPTED",
    "ACK_APPLIED",
    "ACK_REJECTED",
    "CRC_SIZE",
    "HEADER_SIZE",
    "MAGIC",
    "MAX_POINT_COUNT",
    "METADATA_MAGIC",
    "MODE_STRESS",
    "MODE_TEMPERATURE",
    "ModeAck",
    "ModeCommand",
    "MetadataFrame",
    "ProtocolError",
    "TelemetryFrame",
    "decode_ack",
    "decode_frame",
    "decode_mode_ack",
    "decode_mode_command",
    "decode_metadata",
    "decode_telemetry",
    "encode_ack",
    "encode_frame",
    "encode_mode_ack",
    "encode_mode_command",
    "encode_metadata",
    "encode_telemetry",
    "selected_channels",
]
