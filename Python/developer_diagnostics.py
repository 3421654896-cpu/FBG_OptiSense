"""Read-only scan timing budgets. Never changes waits or drives hardware."""

import math


def nominal_budget(point_waits_us, *, i2c_hz=400_000):
    waits = [float(value) for value in point_waits_us]
    if not waits or any(not math.isfinite(v) or v < 0 for v in waits):
        raise ValueError('等待表无效')
    if not math.isfinite(i2c_hz) or i2c_hz <= 0:
        raise ValueError('I²C频率无效')
    # Five independent writes: address, register, MSB, LSB; nine clocks/byte.
    wire_ms = len(waits) * 5 * 4 * 9 / i2c_hz * 1000
    wait_ms = sum(waits) / 1000
    minimum_ms = wire_ms + wait_ms
    return dict(point_count=len(waits), wait_ms=wait_ms, dac_wire_ms=wire_ms,
                minimum_ms=minimum_ms, ceiling_hz=1000 / minimum_ms)


def measured_budget(frame, following, point_waits_us, *, spacing_us=0):
    """Use consecutive board frames only; do not time host rendering/arrival."""
    records = frame['records']
    if (following['sequence'] != frame['sequence'] + 1
            or len(records) != len(point_waits_us)
            or len(following['records']) != len(records)):
        raise ValueError('需要相同点数的相邻板端帧')
    for key in ('firmware', 'adc_channel', 'feedback_selector', 'single_sample',
                'points_per_peak', 'fast_dac_writes'):
        if frame.get(key) != following.get(key):
            raise ValueError('帧的采集配置已改变')
    if any(f.get('i2c_errors', 0) or f.get('spi_errors', 0)
           for f in (frame, following)):
        raise ValueError('帧包含总线错误，不作为正常耗时预算')
    single = bool(frame.get('single_sample'))
    end_key = 'sample_end_us' if single else 'second_end_us'
    point_work = held = 0
    previous_end = 0
    for record in records:
        write, end = record['write_end_us'], record[end_key]
        if not (0 <= previous_end <= write <= end):
            raise ValueError('点时间戳顺序异常')
        point_work += write - previous_end
        held += end - write
        previous_end = end
    waits = nominal_budget(point_waits_us)['wait_ms'] * 1000
    if not math.isfinite(spacing_us) or spacing_us < 0:
        raise ValueError('ADC间隔无效')
    if not single:
        waits += len(records) * spacing_us
    elapsed = frame['elapsed_us']
    period = following['cycle_start_us'] - frame['cycle_start_us']
    if not (0 < elapsed <= period) or elapsed < previous_end or held < waits:
        raise ValueError('帧时间戳或等待预算不一致')
    return dict(point_count=len(records), sequence=frame['sequence'],
                dac_and_interpoint_ms=point_work / 1000,
                prescribed_wait_ms=waits / 1000,
                adc_and_hold_overhead_ms=(held - waits) / 1000,
                frame_tail_ms=(elapsed - previous_end) / 1000,
                frame_gap_ms=(period - elapsed) / 1000,
                period_ms=period / 1000, hz=1_000_000 / period)
