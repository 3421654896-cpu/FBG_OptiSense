"""Shared PI11210 laser-current code limits for host-side utilities."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Iterable


# Default operating ceilings: GAIN/SOA 135 mA, PHASE 10 mA and both wavelength
# heaters 30 mA.  The 2001-point full-band recalibration has a separately
# authorized 145 mA GAIN/SOA ceiling; callers must opt into it explicitly so
# the older stress and candidate-v1/v2 routes retain their 135 mA limit.  The
# temporary T45 route opts in because it consumes the new 2001-point table.
PI11210_SAFE_CODE_LIMITS = (58982, 58982, 32768, 24576, 24576)
FULLBAND_2001_CODE_LIMITS = (63351, 63351, 32767, 24575, 24575)


def parse_pi11210_code(value: object, maximum: int, channel: object) -> int:
    """Parse one exact integer code without silently truncating Excel/JSON data."""

    if isinstance(value, bool):
        raise ValueError(f"PI11210 第{channel}通道DAC码必须是整数：{value!r}")
    try:
        decimal_value = Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"PI11210 第{channel}通道DAC码不是数字：{value!r}") from exc
    if not decimal_value.is_finite() or decimal_value != decimal_value.to_integral_value():
        raise ValueError(f"PI11210 第{channel}通道DAC码必须是整数：{value!r}")
    code = int(decimal_value)
    if code < 0 or code > maximum:
        raise ValueError(
            f"PI11210 第{channel}通道DAC码{code}超过安全范围0～{maximum}"
        )
    return code


def validate_pi11210_codes(
    codes: Iterable[object],
    limits: Iterable[int] = PI11210_SAFE_CODE_LIMITS,
) -> tuple[int, int, int, int, int]:
    """Return five exact codes or reject malformed/out-of-limit input."""

    values = tuple(codes)
    configured_limits = tuple(int(value) for value in limits)
    if len(configured_limits) != len(PI11210_SAFE_CODE_LIMITS):
        raise ValueError("PI11210 safety limit set must contain five channels")
    if len(values) != len(configured_limits):
        raise ValueError(f"PI11210 命令必须包含5个DAC码，实际为{len(values)}个")
    normalized = []
    for channel, (value, maximum) in enumerate(
        zip(values, configured_limits)
    ):
        normalized.append(parse_pi11210_code(value, maximum, channel))
    return tuple(normalized)  # type: ignore[return-value]


__all__ = [
    "PI11210_SAFE_CODE_LIMITS",
    "FULLBAND_2001_CODE_LIMITS",
    "parse_pi11210_code",
    "validate_pi11210_codes",
]
