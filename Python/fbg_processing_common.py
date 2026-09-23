"""Shared numerical front end for local USB and remote MQTT scans."""

from __future__ import annotations

import numpy as np


ADC_REFERENCE_V = 2.5
ADC_CODE_COUNT = 4096.0


def adc_codes_to_voltage(codes):
    """Convert STM32 12-bit ADC codes with identical local/remote scaling."""
    values = np.asarray(codes, dtype=float)
    values = values.copy()
    values[(values < 0.0) | (values > 4095.0)] = np.nan
    return values * ADC_REFERENCE_V / ADC_CODE_COUNT


def scan_min_prominence(base_threshold_v, channel, temperature_mode):
    """Return the one canonical per-channel peak threshold."""
    base = float(base_threshold_v)
    channel = int(channel)
    if temperature_mode and channel < 2:
        return min(base, 0.005)
    if not temperature_mode and channel == 1:
        return min(base, 0.008)
    if channel >= 2:
        return max(0.003, base / 8.0)
    return base


def precision_median_fuse(history, matrix, present_channels):
    """Append one precision frame and return the common five-frame median."""
    values = np.asarray(matrix, dtype=float)
    history.append(values.copy())
    fused = values.copy()
    stack = np.stack(tuple(history), axis=0)
    for channel in present_channels:
        fused[int(channel)] = np.nanmedian(stack[:, int(channel), :], axis=0)
    return fused
