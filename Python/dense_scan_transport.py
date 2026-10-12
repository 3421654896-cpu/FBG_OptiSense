"""Finite board-side dense scan. Never opens a port or changes flash itself.

Old firmware is probed while shuttered and falls back to the existing path.
After a successful capability response, faults never silently retry or fall
back: the caller must perform the existing exact disarm/shutter cleanup.
"""
from __future__ import annotations
import binascii
import math
import struct
import time

from laser_dac_safety import FULLBAND_2001_CODE_LIMITS

MAX_OPS = 48
MAX_DWELL_US = 100_000
MAX_BLOCK_WAIT_US = 10_000_000
HEADER_SIZE = 48
RECORD_SIZE = 22
GUARD_FLAG = 0x8000


def _crc(data):
    return binascii.crc32(data) & 0xffffffff


def _wait_us(seconds):
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Invalid dense dwell')
    # Round UP, never shorten the requested optical settling time.
    return math.ceil(seconds * 1_000_000)


def _codes(codes):
    values = tuple(codes)
    if len(values) != 5 or any(type(v) is not int or not 0 <= v <= limit
                             for v, limit in zip(values, FULLBAND_2001_CODE_LIMITS)):
        raise ValueError('Dense DAC code exceeds current ceiling or is not integral')
    return values


def operations(points, guards, settle_us):
    result = []
    previous = -1
    for point in points:
        if type(point.index) is not int or not previous < point.index <= 2000:
            raise ValueError('Dense physical indices must increase within 0..2000')
        previous = point.index
        guard = guards.get(point.index)
        if guard is not None:
            wait = _wait_us(guard.hold_s_per_stage)
            if not 0 < wait <= 2_000_000:
                raise ValueError('Dense transition guard dwell is out of bounds')
            result.extend((point.index | GUARD_FLAG, _codes(codes), wait)
                          for codes in guard.precondition_codes)
        result.append((point.index, _codes(point.codes), settle_us))
    return result


def blocks(ops):
    block, wait_sum = [], 0
    for op in ops:
        if block and (len(block) == MAX_OPS or wait_sum + op[2] > MAX_BLOCK_WAIT_US):
            yield block
            block, wait_sum = [], 0
        block.append(op)
        wait_sum += op[2]
    if block:
        yield block


def command(tag, sequence, ops, *, flags, settle_us, selectors, total):
    if (type(tag) is not int or not 0 < tag <= 0xffffffff
            or type(sequence) is not int or not 0 <= sequence < 65536
            or not 1 <= len(ops) <= MAX_OPS or not 0 <= flags <= 3
            or not 0 <= settle_us <= MAX_DWELL_US or not 1 <= total <= 2001
            or len(selectors) != 2 or any(type(v) is not int or v not in range(4) for v in selectors)):
        raise ValueError('Invalid dense block header')
    packet = bytearray(808)
    packet[:8] = b'\xff\xff\x03\x0dDNS1'
    struct.pack_into('>IHBBIBBH', packet, 8, tag, sequence, len(ops), flags,
                     settle_us, *selectors, total)
    previous, wait_sum = -1, 0
    for i, (index, codes, wait) in enumerate(ops):
        guard, physical = bool(index & GUARD_FLAG), index & 0x7fff
        if (type(index) is not int or not 0 <= index <= 0xffff or not previous < physical <= 2000
                or type(wait) is not int or (guard and not 0 < wait <= 2_000_000)
                or (not guard and wait != settle_us)):
            raise ValueError('Invalid dense operation')
        if not guard:
            previous = physical
        wait_sum += wait
        if wait_sum > MAX_BLOCK_WAIT_US:
            raise ValueError('Dense block time exceeds hard bound')
        struct.pack_into('>6HI', packet, 24 + i * 16, index, *_codes(codes), wait)
    struct.pack_into('>I', packet, 804, _crc(packet[:804]))
    return bytes(packet)


def extract_frames(buffer):
    while True:
        start = buffer.find(b'\xd6\x6d\x01')
        if start < 0:
            if len(buffer) > 2:
                del buffer[:-2]
            return
        if start:
            del buffer[:start]
        if len(buffer) < 6:
            return
        length = int.from_bytes(buffer[4:6], 'big')
        if not 54 <= length <= 54 + RECORD_SIZE * MAX_OPS:
            del buffer[0]
            continue
        if len(buffer) < length:
            return
        wire = bytes(buffer[:length])
        del buffer[:length]
        if wire[-2:] != b'\x6d\xd6' or _crc(wire[:-6]) != int.from_bytes(wire[-6:-2], 'big'):
            raise ValueError('Dense response CRC/trailer failed')
        yield wire


def decode(wire, tag, sequence, packet=None):
    if int.from_bytes(wire[6:10], 'big') != tag or int.from_bytes(wire[10:12], 'big') != sequence:
        raise ValueError('Dense response tag/sequence mismatch')
    kind, requested, completed = wire[3], wire[12], wire[13]
    status = int.from_bytes(wire[14:16], 'big')
    if len(wire) != 54 + completed * RECORD_SIZE or kind not in (0, 1):
        raise ValueError('Dense response shape mismatch')
    result = dict(kind=kind, requested=requested, completed=completed, status=status,
                  firmware_version=int.from_bytes(wire[16:20], 'big'),
                  request_crc=int.from_bytes(wire[20:24], 'big'),
                  elapsed_us=int.from_bytes(wire[24:28], 'big'),
                  settle_us=int.from_bytes(wire[28:32], 'big'),
                  selectors=tuple(wire[32:34]), soa_mode=wire[34], pi_flags=wire[35],
                  i2c_error_delta=int.from_bytes(wire[36:38], 'big'),
                  spi_error_delta=int.from_bytes(wire[38:42], 'big'), records=[])
    if status or result['i2c_error_delta'] or result['spi_error_delta']:
        raise RuntimeError(f'Dense board acquisition failed: status={status}, '
                           f"I2C={result['i2c_error_delta']}, SPI={result['spi_error_delta']}")
    if result['pi_flags'] & 0x87 != 0x87 or result['pi_flags'] & 0x30:
        raise RuntimeError('Dense PI11210 protection/identity/write status failed')
    if packet is not None:
        if (kind != 1 or requested != packet[14] or completed != requested
                or result['request_crc'] != int.from_bytes(packet[804:], 'big')
                or result['selectors'] != tuple(packet[20:22])
                or result['settle_us'] != int.from_bytes(packet[16:20], 'big')
                or int.from_bytes(wire[42:44], 'big') != int.from_bytes(packet[22:24], 'big')
                or result['soa_mode'] != (1 if packet[15] & 2 else 0)):
            raise ValueError('Dense accepted request/feedback/shutter provenance mismatch')
        wait_sum = sum(int.from_bytes(packet[36 + i*16:40 + i*16], 'big')
                       for i in range(requested))
        if result['elapsed_us'] > wait_sum + 3_000_000:
            raise ValueError('Dense board timestamp exceeds the bounded block budget')
    previous_end = 0
    for i in range(completed):
        index, *values = struct.unpack_from('>7HII', wire, HEADER_SIZE + i * RECORD_SIZE)
        adc, write_end, sample_end = values[:6], values[6], values[7]
        if packet is not None:
            request_index = int.from_bytes(packet[24 + 16*i:26 + 16*i], 'big')
            wait = int.from_bytes(packet[36 + 16*i:40 + 16*i], 'big')
            if index != request_index or write_end < previous_end or sample_end < write_end + wait:
                raise ValueError('Dense physical point/order/dwell mismatch')
        if any(v > 4095 for v in adc) or sample_end > result['elapsed_us']:
            raise ValueError('Dense raw ADC/timestamp out of bounds')
        previous_end = sample_end
        result['records'].append(dict(index=index, adc=adc,
                                      write_end_us=write_end, sample_end_us=sample_end))
    return result


def _receive(device, tag, sequence, deadline, *, should_stop):
    buffer = bytearray()
    while time.monotonic() < deadline:
        if should_stop():
            raise InterruptedError('重采光谱已取消')
        waiting = int(device.in_waiting)
        if waiting:
            buffer.extend(device.read(min(waiting, 4096)))
            for wire in extract_frames(buffer):
                if int.from_bytes(wire[6:10], 'big') == tag:
                    return wire
        else:
            time.sleep(.001)
    return None


def probe(device, *, should_stop=lambda: False):
    if type(getattr(device, 'in_waiting', None)) is not int:
        return None
    tag = time.time_ns() & 0xffffffff or 1
    packet = bytearray(808)
    packet[:8] = b'\xff\xff\x03\x0cDNS1'
    struct.pack_into('>I', packet, 8, tag)
    device.reset_input_buffer()
    device.write(packet)
    wire = _receive(device, tag, 0, time.monotonic() + .6, should_stop=should_stop)
    if wire is None:
        return None  # Unsupported old firmware; no DAC write was requested.
    result = decode(wire, tag, 0)
    if result['kind'] != 0 or result['requested'] != MAX_OPS or result['soa_mode'] != 1:
        raise ValueError('Invalid dense capability or initial shutter state')
    return result


def acquire(device, points, guards, *, settle_s, selectors, should_stop=lambda: False):
    """Yield every real point; guards preserve their full original dwell/order."""
    settle_us = _wait_us(settle_s)
    ops = operations(points, guards, settle_us)
    chunks = list(blocks(ops))
    tag = time.time_ns() & 0xffffffff or 1
    by_index = {p.index: p for p in points}
    for sequence, chunk in enumerate(chunks):
        if should_stop():
            raise InterruptedError('重采光谱已取消')
        flags = (1 if sequence == 0 else 0) | (2 if sequence == len(chunks)-1 else 0)
        packet = command(tag, sequence, chunk, flags=flags, settle_us=settle_us,
                         selectors=selectors, total=len(points))
        device.reset_input_buffer()
        device.write(packet)
        deadline = time.monotonic() + sum(op[2] for op in chunk) / 1_000_000 + 3
        wire = _receive(device, tag, sequence, deadline, should_stop=should_stop)
        if wire is None:
            raise TimeoutError('板卡批量重采超时；不会重放或跳过这一批')
        result = decode(wire, tag, sequence, packet)
        previous_end = 0
        for record in result['records']:
            elapsed = (record['sample_end_us'] - previous_end) / 1_000_000
            previous_end = record['sample_end_us']
            if record['index'] & GUARD_FLAG:
                continue
            point = by_index[record['index']]
            adc = record['adc']
            yield dict(index=point.index, target_wavelength_nm=point.target_nm,
                       measured_wavelength_nm=point.measured_nm, dac_codes=list(point.codes),
                       adc_codes=adc[:4], monitor_codes=adc[4:],
                       settle_s=settle_us / 1_000_000, elapsed_s=elapsed,
                       board_write_end_us=record['write_end_us'],
                       board_sample_end_us=record['sample_end_us'],
                       board_block_sequence=sequence,
                       board_block_elapsed_us=result['elapsed_us'],
                       adc_acquisition_method='STM32.dense_batch.single_fresh_adc',
                       attempts=[], single_sample=True, stability_not_evaluated=True,
                       sample_count=1, stable=None, window_stable=None,
                       all_monitor_channels_stable=None)
