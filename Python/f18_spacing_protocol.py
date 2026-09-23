"""Opt-in F18 spacing packet for merged firmware 1.0.72; no hardware access.

Legacy 18-point capture remains unchanged. Historical 1.0.55/56 responses may
be decoded only with an explicit expected version, never silently accepted live.
"""
from capture_fast_flank_stream import command as legacy_command
from capture_fast_flank_stream import decode_data as legacy_decode_data, decode_end as legacy_decode_end

FIRMWARE = 0x00010048


def command(cycles, tag, spacing_us):
    if type(cycles) is not int or not 32 <= cycles <= 2048:
        raise ValueError('Explicit F18 cycles must be 32..2048')
    if type(spacing_us) is not int or not 50 <= spacing_us <= 600 or spacing_us % 25:
        raise ValueError('F18 spacing must be 50..600 us in multiples of 25')
    packet = bytearray(legacy_command(min(cycles,1024), tag))
    packet[16:18] = cycles.to_bytes(2,'big')
    packet[19] = 0x53
    packet[20:22] = spacing_us.to_bytes(2, 'big')
    return bytes(packet)


def decode_data(wire, *, tag, table_crc32, spacing_us, expected_firmware=FIRMWARE):
    if expected_firmware not in (0x10037, 0x10038, 0x10039, 0x1003A, 0x1003B, 0x1003C, 0x1003D, 0x1003E, 0x1003F, 0x10040, 0x10041, 0x10042, 0x10043, 0x10044, 0x10045, 0x10046, 0x10047, 0x10048):
        raise ValueError('Unsupported explicit F18 firmware')
    return legacy_decode_data(wire, tag=tag, table_crc32=table_crc32,
                              expected_firmware=expected_firmware, spacing_us=spacing_us)


def decode_end(wire, *, tag, table_crc32, expected_firmware=FIRMWARE):
    if expected_firmware not in (0x10037, 0x10038, 0x10039, 0x1003A, 0x1003B, 0x1003C, 0x1003D, 0x1003E, 0x1003F, 0x10040, 0x10041, 0x10042, 0x10043, 0x10044, 0x10045, 0x10046, 0x10047, 0x10048):
        raise ValueError('Unsupported explicit F18 firmware')
    return legacy_decode_end(wire, tag=tag, table_crc32=table_crc32, expected_firmware=expected_firmware)
