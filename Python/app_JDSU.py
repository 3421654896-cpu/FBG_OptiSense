import sys
import os
import csv
import json
import unicodedata
from pathlib import Path

# Ensure the PyQt5 Windows platform plugin can be found in a local virtual
# environment before importing any Qt modules.
qt_platform_plugins = (
    Path(sys.prefix) / "Lib" / "site-packages" / "PyQt5" / "Qt5"
    / "plugins" / "platforms"
)
if qt_platform_plugins.exists():
    # A stale system/user value takes precedence over PyQt's own discovery and
    # can make QApplication abort before a window is created.  This only sets
    # the plugin *directory*; QT_QPA_PLATFORM=offscreen remains untouched.
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(qt_platform_plugins)

import serial
import math
import time
import yaml
import pyvisa
import openpyxl
import threading
import statistics
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass, replace

import pyqtgraph as pg
import numpy as np

from openpyxl import load_workbook
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,QLineEdit,QPushButton,
    QVBoxLayout, QLabel, QStackedWidget, QAction, QMessageBox,
    QFileDialog, QPlainTextEdit, QRadioButton, QButtonGroup
)
from PyQt5.QtCore import (
    QThread, pyqtSignal
)
from serial.tools import list_ports
from datetime import datetime
from collections import deque
from queue import Queue
from queue import Empty

from scipy.signal import filtfilt, medfilt, savgol_filter, butter, find_peaks, peak_widths
from pyvisa.constants import InterfaceType

from fbg_peak_fitting import (
    AdaptivePeakTracker,
    PeakDisplayNormalizer,
    TemperaturePeakTracker,
    build_segments,
    detect_wavelength_gaps,
    fit_channel_segments,
    fit_dense_reflection_spectrum,
    mask_unconnected_fbg_segments,
    required_display_gain,
    rounded_display_curve,
    scale_segment_samples,
)
from fbg_processing_common import (
    ADC_CODE_COUNT,
    ADC_REFERENCE_V,
    precision_median_fuse,
    scan_min_prominence,
)
from responsive_layout import (
    FlowLayout,
    ResponsiveScrollArea,
    ScrollContentWidget,
    compact_field,
    configure_form_layout,
)
from mode_table_manager import (
    DEFAULT_MODE_POINT_COUNTS,
    MINIMUM_POINTS_PER_PEAK,
    MODE_POINT_COUNT_LIMITS,
    apply_candidate_and_build,
    latest_mode_table_backup,
    restore_previous_and_build,
    sparse_stress_table_fingerprint,
    validate_mode_point_budget,
)
from adaptive_sampling_optimizer import optimize_ch1_sampling
from fullband_auto_calibration_widget import FullbandAutoCalibrationWindow
from frame_telemetry import (
    BoardSequenceTracker,
    MULTIRATE_POINT_COUNT,
    MULTIRATE_PROFILE_MAP,
    MULTIRATE_PROFILE_SURVEY,
    MULTIRATE_PROFILE_TRACK,
    MULTIRATE_PROFILE_TRACK_SINGLE,
    decode_board_runtime_identity,
    decode_board_frame_schedule,
    decode_board_frame_timing,
    extract_raw_scan_frames,
)
from ch1_contact_localizer import CH1NinePeakLocalizer, MultiRateContactTracker
from realtime_frame_queue import LatestFrameBuffer
from stress_multirate_protocol import (
    CURRENT_SCHEDULE_VERSION,
    MultirateConfiguration,
    build_multirate_arm_command,
    REFERENCE_MAP_PERIOD,
    build_multirate_disarm_command,
    decode_multirate_ack,
    validate_current_arm_ack,
    validate_current_disarm_ack,
)
from runtime_paths import output_path
from machine_profile import (
    get_runtime_machine_id,
    runtime_fullband_profile,
    runtime_machine_label,
    runtime_parameter_note,
    set_runtime_machine_id,
)

#测试
# ====== 参数（按需改）======
ADDR = "GPIB0::7::INSTR"

# 当前与备用波长计。保持精确到序列号的白名单，既允许更换后的
# AQ6150，也避免误连到同一 GPIB 总线上的其他横河仪器。
TARGET_VENDOR = "YOKOGAWA"
TARGET_MODEL = "AQ6150"
TARGET_SN = "91P102177"
SUPPORTED_AQ6150_IDENTITIES = frozenset(
    {
        (TARGET_VENDOR, TARGET_MODEL, TARGET_SN),
        ("YOKOGAWA", "AQ6150B", "9027C2596"),
    }
)


def is_supported_aq6150_identity(identity: str) -> bool:
    """Return whether an ``*IDN?`` reply is one of the approved meters."""
    parts = tuple(part.strip().upper() for part in str(identity).split(","))
    if len(parts) < 3:
        return False
    return parts[:3] in SUPPORTED_AQ6150_IDENTITIES

PRINT_EXCEL_EVERY = 1      # 每隔N行打印一次Excel数据（1=每行都打印）
PRINT_ACK = True           # 打印接收到的ACK

VISA_TIMEOUT_MS = 3000
VISA_RETRY = 10

FLUSH_EVERY_N = 1000

ACK_VALUE = 0x21
ACK_ERROR_VALUE = 0xE1
ACK_WAIT_SLICE_S = 0.05
ACK_ATTEMPT_TIMEOUT_S = 2.0
ACK_MAX_ATTEMPTS = 2
AP_MODE_SWITCH_SETTLE_S = 0.05
AP_SAFE_SHUTDOWN_TIMEOUT_S = 1.0

array_size = 4000

CONTACT_BASELINE_FRAMES = 20
CONTACT_BASELINE_MEDIAN_DELTA_CODES = 12.0
CONTACT_BASELINE_P95_DELTA_CODES = 40.0
MULTIRATE_PROFILE_NAMES = {
    MULTIRATE_PROFILE_MAP: "MAP",
    MULTIRATE_PROFILE_SURVEY: "SURVEY",
    MULTIRATE_PROFILE_TRACK: "TRACK13_WIDE",
    MULTIRATE_PROFILE_TRACK_SINGLE: "TRACK11_SINGLE",
}

tx_size = 808  # 固定控制命令长度（所有控制/差异点帧统一为 808 字节）
# 包格式（差异点/控制统一长度）：0xFF 0xFF 0x01 0x04 | len0 | len1 | len2 | len3 | idx0_hi idx0_lo ...
# 每通道最多 100 个差异点，索引用 2 字节编码。
# ==========================

ser = serial.Serial(timeout=0.2)
local_control_session = False


def ensure_serial_open():
    """Open the selected CDC port and keep DTR asserted for local ownership."""
    if not ser.is_open:
        ser.open()
    ser.dtr = True


def release_serial_if_allowed():
    """Pages may stop independently; only application exit releases CDC."""
    if not local_control_session and ser.is_open:
        ser.close()

ser_open = False
ser_cond = threading.Condition()

ap_open = False
ap_cond = threading.Condition()

switch_mode_enable = True

# PI11210 is the only enabled laser-tuning DAC.  The value is retained in
# transmitted frames for compatibility with the existing MCU protocol.
PI11210_PROTOCOL_ID = 0x01
dac_type = PI11210_PROTOCOL_ID

STM32_CDC_VID_PID = (0x0483, 0x5740)


def find_native_stm32_cdc_port(ports):
    """Return only the board's native USB CDC port for automatic selection.

    Other USB serial ports remain available in the menu for an explicit
    operator choice, but must never be guessed as the acquisition endpoint.
    """

    for port_info in ports:
        if (
            getattr(port_info, "vid", None),
            getattr(port_info, "pid", None),
        ) == STM32_CDC_VID_PID:
            return getattr(port_info, "device", None)
    return None

PI11210_GAIN_FS_MA = 150.0
PI11210_SOA_SOURCE_FS_MA = 150.0
PI11210_SOA_SINK_FS_MA = 50.0
PI11210_PHASE_FS_MA = 20.0
PI11210_WAVELENGTH_FS_MA = 80.0
PI11210_SOA_SOURCE_MODE = 0
PI11210_SOA_SHUTTER_MODE = 1

# Laser safety limits.  These are intentionally stricter than the DAC ranges.
LASER_GAIN_MAX_MA = 135.0
LASER_SOA_SOURCE_MAX_MA = 135.0
LASER_PHASE_MAX_MA = 10.0
LASER_WAVELENGTH_MAX_MA = 30.0

# Direct-code limits used by the ordinary single-value page.  They remain at
# 135 mA even though firmware admits the separately authorized 145 mA
# 2001-point table profile.
PI11210_GAIN_MAX_CODE = 58981
PI11210_SOA_SOURCE_MAX_CODE = 58981
# Dedicated 2001-point calibration/runtime ceiling.  These codes are the
# largest integer values strictly not exceeding 145 mA on a 150 mA full scale.
# Other desktop modes continue to use the 135 mA constants above.
FULLBAND_ACCURACY_GAIN_MAX_CODE = 63351
FULLBAND_ACCURACY_SOA_MAX_CODE = 63351
# A negative SOA request means "laser off".  OP6 is wired directly to the SOA
# with no reverse-voltage clamp or monitor, so firmware safely implements this
# as the PI11210 zero-current Gate state.  It must not generate an unverified
# negative voltage beyond the laser's 3 V reverse-voltage limit.
PI11210_SOA_SHUTTER_CODE = 0
PI11210_PHASE_MAX_CODE = 32767
PI11210_WAVELENGTH_MAX_CODE = 24575
SINGLE_VALUE_TABLE_PRECONDITION_S = 0.30
SINGLE_VALUE_FRAME_SIZE = 20
STABLE_REFERENCE_PRECONDITION_S = 0.055
STABLE_REFERENCE_SETTLE_S = 0.20
STABLE_REFERENCE_MIN_FRAMES = 5
STABLE_REFERENCE_MAX_MEDIAN_FRAMES = 32
STRESS_REFERENCE_FAST_CALIBRATION_FRAMES = 8
STRESS_REFERENCE_MAX_OFFSET_V = 0.15

# “无限准确模式” deliberately has no frame-rate target.  It replays the
# final, wavelength-meter-verified 2001-point table through the exact same
# five-channel single-value command used during calibration, then waits for a
# settled run of monitor samples before publishing each point.
FULLBAND_ACCURACY_POINT_COUNT = 2001
FULLBAND_ACCURACY_START_NM = Decimal("1525.00")
FULLBAND_ACCURACY_STEP_NM = Decimal("0.02")
FULLBAND_ACCURACY_MIN_SETTLE_S = 0.20
FULLBAND_ACCURACY_FIRST_POINT_SETTLE_S = 0.50
FULLBAND_ACCURACY_MIN_SAMPLES = 15
FULLBAND_ACCURACY_MAX_SAMPLES = 35
FULLBAND_ACCURACY_POINT_TIMEOUT_S = 1.50
FULLBAND_ACCURACY_STAT_WINDOW = 25
FULLBAND_ACCURACY_MAD_FLOOR_CODES = 6.0
FULLBAND_ACCURACY_DRIFT_FLOOR_CODES = 8.0
FULLBAND_ACCURACY_GUARD_INDICES = frozenset((241, 1854))
EQUAL_INTERVAL_FEEDBACK_VERIFY_EVERY_POINTS = 32

# This thumb routes all nine FBG reflections to CH1.  These are defaults, not
# hidden assumptions: callers can explicitly request a different channel set,
# peak count or safe point budget for another sensor/report.
FINGER_FBG_CHANNELS = (1,)
FINGER_EXPECTED_PEAK_COUNT = 9
FINGER_STRESS_POINT_COUNT = DEFAULT_MODE_POINT_COUNTS["stress"]
FINGER_TEMPERATURE_POINT_COUNT = DEFAULT_MODE_POINT_COUNTS["temperature"]

# RS2255 selector is encoded directly as (B << 1) | A.  R118/R119 are
# annotated as 4 kOhm in the drawing but explicitly marked "实贴5K".
PD_FEEDBACK_KOHM_BY_SELECTOR = (2, 40, 5, 20)
PD_FEEDBACK_DEFAULT_SELECTOR = 1


@dataclass(frozen=True)
class FullbandAccuracyPoint:
    index: int
    target_nm: float
    measured_nm: float
    codes: tuple


@dataclass(frozen=True)
class FullbandTransitionGuard:
    target_index: int
    target_nm: float
    hold_s_per_stage: float
    precondition_codes: tuple


@dataclass(frozen=True)
class MeasuredTemplatePeak:
    """Non-Gaussian CH1 peak descriptor used by sparse template selection."""

    center_nm: float
    amplitude_v: float
    sigma_nm: float
    rmse_v: float
    center_std_pm: float
    peak_index: int
    region_start_nm: float
    region_stop_nm: float

# PDR/PDT front end: 1.25 V virtual ground, 2 kOhm transimpedance and
# a 2.5 V, 12-bit ADC.  Keep these as calibration constants so measured
# board values can replace the nominal values later without changing math.
PD_ADC_REFERENCE_V = ADC_REFERENCE_V
PD_ADC_CODE_COUNT = ADC_CODE_COUNT
PD_TIA_BIAS_V = 1.25
PD_TIA_FEEDBACK_OHM = 2000.0
PD_RATIO_MIN_CURRENT_MA = 1e-6


def compute_fixed_stress_reference_offset(
    fast_frames,
    stable_reference,
    channels,
    maximum_absolute_offset_v=STRESS_REFERENCE_MAX_OFFSET_V,
):
    """Return one fixed fast-scan correction without temporal filtering.

    ``fast_frames`` has shape (frame, channel, wavelength).  The correction is
    learned once immediately after a settled single-value reference capture.
    Later stress frames remain independent and therefore retain same-frame
    transient response.
    """
    frames = np.asarray(fast_frames, dtype=float)
    reference = np.asarray(stable_reference, dtype=float)
    if frames.ndim != 3 or reference.ndim != 2:
        raise ValueError("invalid stress-reference calibration shape")
    if frames.shape[1:] != reference.shape or frames.shape[0] < 1:
        raise ValueError("fast/reference dimensions do not match")
    correction = np.zeros_like(reference, dtype=float)
    fast_baseline = np.median(frames, axis=0)
    limit = abs(float(maximum_absolute_offset_v))
    for channel in channels:
        channel = int(channel)
        if not 0 <= channel < reference.shape[0]:
            continue
        finite = np.isfinite(reference[channel]) & np.isfinite(fast_baseline[channel])
        correction[channel, finite] = np.clip(
            fast_baseline[channel, finite] - reference[channel, finite],
            -limit,
            limit,
        )
    return correction

def pi11210_current_to_code(current_ma, full_scale_ma):
    """Convert current in mA to a saturated PI11210 16-bit DAC code."""
    current = float(current_ma)
    full_scale = float(full_scale_ma)
    if not math.isfinite(current):
        raise ValueError("PI11210电流必须是有限数")
    if not math.isfinite(full_scale) or full_scale <= 0.0:
        raise ValueError("PI11210满量程必须是正的有限数")
    code = int(current / full_scale * 65536)
    return min(max(code, 0), 0xFFFF)

def limited_pi11210_code(current_ma, full_scale_ma, safe_max_ma):
    """Convert a source current after applying the laser's safe limit."""
    safe_max = float(safe_max_ma)
    if not math.isfinite(safe_max) or safe_max < 0.0:
        raise ValueError("激光电流安全上限必须是非负有限数")
    safe_current_ma = min(max(float(current_ma), 0.0), safe_max)
    return pi11210_current_to_code(safe_current_ma, full_scale_ma)

def parse_pi11210_dac_code(text, safe_max_code):
    """Parse an exact decimal DAC code without silently rounding or clipping."""
    try:
        value = Decimal(str(text).strip())
    except (InvalidOperation, ValueError):
        raise ValueError("DAC码必须是十进制整数") from None
    if not value.is_finite() or value != value.to_integral_value():
        raise ValueError("DAC码必须是十进制整数")
    code = int(value)
    if code < 0 or code > int(safe_max_code):
        raise ValueError(f"DAC码必须在0～{safe_max_code}之间")
    return code


def normalize_pi11210_calibration_code(value, safe_max_code):
    """Normalize the historical one-LSB table boundary to the MCU clamp.

    Early calibration exports used the mathematically rounded 135/10/30 mA
    boundary, while the deployed firmware deliberately clamps those channels
    one DAC count lower.  Those measurements were therefore made at the lower
    (echoed) code already.  Accept only this exact legacy boundary and retain
    strict rejection for every other out-of-range value.
    """
    maximum = int(safe_max_code)
    try:
        code = parse_pi11210_dac_code(value, maximum)
    except ValueError:
        try:
            parsed = Decimal(str(value).strip())
        except (InvalidOperation, ValueError):
            raise ValueError("DAC码必须是十进制整数") from None
        if (
            parsed.is_finite()
            and parsed == parsed.to_integral_value()
            and int(parsed) == maximum + 1
        ):
            return maximum
        raise
    return code

def load_stress_table_dac_rows():
    """Load the exact, replay-validated stress rows for single-point preconditioning."""
    base = Path(__file__).resolve().parent
    limits = _pi11210_table_code_limits()
    for source in (
        base / "mode_tables_from_fullband_2001.json",
        base / "stress_shape_power_dynamic_v3.json",
        base / "stress_shape_power_final_v2.json",
        base / "stress_high_power_135_final.json",
    ):
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
            source_rows = (
                payload.get("modes", {}).get("stress", {}).get("rows")
                if source.name == "mode_tables_from_fullband_2001.json"
                else payload.get("rows")
            )
            rows = []
            for row in source_rows or ():
                raw_codes = tuple(row["codes"])
                if len(raw_codes) != len(limits):
                    raise ValueError("应力扫描DAC行必须包含5个码")
                rows.append(
                    tuple(
                        normalize_pi11210_calibration_code(code, maximum)
                        for code, maximum in zip(raw_codes, limits)
                    )
                )
            if rows:
                return tuple(rows)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return ()


def _pi11210_table_code_limits():
    return (
        PI11210_GAIN_MAX_CODE,
        PI11210_SOA_SOURCE_MAX_CODE,
        PI11210_PHASE_MAX_CODE,
        PI11210_WAVELENGTH_MAX_CODE,
        PI11210_WAVELENGTH_MAX_CODE,
    )


def _fullband_accuracy_code_limits():
    return (
        FULLBAND_ACCURACY_GAIN_MAX_CODE,
        FULLBAND_ACCURACY_SOA_MAX_CODE,
        PI11210_PHASE_MAX_CODE,
        PI11210_WAVELENGTH_MAX_CODE,
        PI11210_WAVELENGTH_MAX_CODE,
    )


FULLBAND_ACCURACY_LIMIT_PROFILE = "fullband_2001_145mA_v1"


PI11210_EXCEL_CODE_CHANNELS = (
    "GAIN",
    "SOA",
    "PHASE",
    "WAVE_A",
    "WAVE_B",
)

_PI11210_EXCEL_HEADER_ALIASES = {
    "GAIN": ("GAIN",),
    "SOA": ("SOA",),
    "PHASE": ("PHASE",),
    "WAVE_A": ("WAVELENGTHA", "WAVEA"),
    "WAVE_B": ("WAVELENGTHB", "WAVEB"),
}

_PI11210_EXCEL_NON_CODE_HINTS = (
    "CURRENT",
    "POWER",
    "OPTICALPOWER",
    "INTENSITY",
    "VOLTAGE",
    "ERROR",
    "SMSR",
    "电流",
    "功率",
    "光强",
    "电压",
    "误差",
)


def _compact_pi11210_excel_header(value):
    """Return a comparison form while retaining Chinese/code-garble markers."""

    if value is None:
        return ""
    normalized = unicodedata.normalize("NFKC", str(value)).strip().upper()
    return "".join(
        character
        for character in normalized
        if character.isalnum() or character in ("码", "?", "\ufffd")
    )


def _pi11210_header_has_non_code_meaning(value):
    normalized = unicodedata.normalize("NFKC", str(value or "")).strip().upper()
    compact = _compact_pi11210_excel_header(normalized)
    if any(hint in compact for hint in _PI11210_EXCEL_NON_CODE_HINTS):
        return True
    # The current columns in both calibration-table generations are labelled
    # "(mA)".  Treat a trailing/current-unit MA as unsafe even if all Chinese
    # text around it was damaged by an encoding conversion.
    return "(MA)" in normalized or "[MA]" in normalized or compact.endswith("MA")


def _pi11210_header_channel_and_residue(value):
    """Return the channel and non-channel portion of a compact table header."""

    compact = _compact_pi11210_excel_header(value)
    matches = []
    for channel, aliases in _PI11210_EXCEL_HEADER_ALIASES.items():
        for alias in aliases:
            position = compact.find(alias)
            if position < 0:
                continue
            residue = compact[:position] + compact[position + len(alias):]
            matches.append((channel, residue))
            break
    if len(matches) != 1:
        return None, ""
    return matches[0]


def _explicit_pi11210_code_header_channel(value):
    """Recognise only headers that explicitly say code/DAC/码."""

    if _pi11210_header_has_non_code_meaning(value):
        return None
    channel, residue = _pi11210_header_channel_and_residue(value)
    if channel is None:
        return None
    if not any(marker in residue for marker in ("DACCODE", "CODE", "DAC", "码")):
        return None
    for marker in ("PI11210", "DACCODE", "CODE", "DAC", "码"):
        residue = residue.replace(marker, "")
    return channel if not residue else None


def _current_pi11210_header_channel(value):
    """Recognise the mA/current block used immediately before old code blocks."""

    channel, residue = _pi11210_header_channel_and_residue(value)
    if channel is None:
        return None
    for marker in ("MILLIAMPERE", "MILLIAMP", "CURRENT", "电流", "MA"):
        residue = residue.replace(marker, "")
    return channel if not residue and _pi11210_header_has_non_code_meaning(value) else None


def _looks_like_corrupted_code_marker(residue):
    """Accept common mojibake forms of 码, but never a unit-less bare header."""

    if not residue:
        return False
    if "\ufffd" in residue or "?" in residue:
        return True
    # 码 after common UTF-8/GBK -> Latin-1/Windows decoding mistakes.  The
    # compact form intentionally removes the non-breaking/control byte.
    known = {
        _compact_pi11210_excel_header(text)
        for text in ("ç\xa0\x81", "ç\xa0�", "Âë", "鐮�")
    }
    return residue in known


def resolve_pi11210_excel_code_columns(header_row):
    """Resolve GAIN/SOA/PHASE/WAVE_A/WAVE_B code columns from one header row.

    There is deliberately no numeric-position fallback.  A calibration sheet
    often places fractional power/error/current columns before the integer DAC
    codes, so guessing ``row[6:11]`` can both fail and command the wrong laser
    state.  Legacy ``code_*`` names and the current ``*码`` names are accepted.
    A structurally paired, visibly corrupted code block is accepted only when
    it follows a complete five-column mA/current block.
    """

    headers = tuple(header_row or ())
    candidates = {channel: [] for channel in PI11210_EXCEL_CODE_CHANNELS}
    for index, header in enumerate(headers):
        channel = _explicit_pi11210_code_header_channel(header)
        if channel is not None:
            candidates[channel].append(index)

    duplicated = {
        channel: indices
        for channel, indices in candidates.items()
        if len(indices) > 1
    }
    if duplicated:
        details = ", ".join(
            f"{channel}={','.join(str(index + 1) for index in indices)}列"
            for channel, indices in duplicated.items()
        )
        raise ValueError(f"五路DAC码表头存在歧义：{details}")

    if all(len(candidates[channel]) == 1 for channel in PI11210_EXCEL_CODE_CHANNELS):
        return tuple(candidates[channel][0] for channel in PI11210_EXCEL_CODE_CHANNELS)

    # Encoding-damaged workbooks can leave GAIN/SOA/... readable while the
    # Chinese 码 suffix becomes replacement/mojibake characters.  Recover only
    # a complete ordered block, and only when each uncertain member has a known
    # corrupt marker or the block directly follows the known five-column mA
    # block.  This prevents current/power columns from becoming a fallback.
    structured = []
    width = len(PI11210_EXCEL_CODE_CHANNELS)
    for start in range(max(0, len(headers) - width + 1)):
        preceding_current_block = start >= width and all(
            _current_pi11210_header_channel(headers[start - width + offset])
            == expected
            for offset, expected in enumerate(PI11210_EXCEL_CODE_CHANNELS)
        )
        valid = True
        for offset, expected in enumerate(PI11210_EXCEL_CODE_CHANNELS):
            index = start + offset
            explicit = _explicit_pi11210_code_header_channel(headers[index])
            if explicit is not None:
                if explicit != expected:
                    valid = False
                    break
            else:
                if _pi11210_header_has_non_code_meaning(headers[index]):
                    valid = False
                    break
                channel, residue = _pi11210_header_channel_and_residue(headers[index])
                if channel != expected or not residue or not (
                    preceding_current_block or _looks_like_corrupted_code_marker(residue)
                ):
                    valid = False
                    break
            known_indices = candidates[expected]
            if known_indices and known_indices[0] != index:
                valid = False
                break
        if valid:
            structured.append(tuple(range(start, start + width)))

    if len(structured) == 1:
        return structured[0]
    if len(structured) > 1:
        raise ValueError("五路DAC码表头存在多个可用列组，无法安全选择")

    missing = [
        channel
        for channel in PI11210_EXCEL_CODE_CHANNELS
        if not candidates[channel]
    ]
    raise ValueError(
        "未找到完整且唯一的五路DAC码表头（缺少："
        + ", ".join(missing)
        + "）；请使用 GAIN码/SOA码/PHASE码/WAVE_A码/WAVE_B码，"
        "或 code_GAIN/code_SOA/code_PHASE/code_wavelengthA/code_wavelengthB"
    )


def _load_fullband_calibration_json(source):
    """Load a complete first-calibration archive as runtime DAC rows.

    This validates the calibration pass itself, not the later independent
    repeatability audit.  Machine two has 2001 successful calibration rows;
    its 285 independent recheck failures remain visible as an operator warning.
    """

    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, dict) or len(rows) != FULLBAND_ACCURACY_POINT_COUNT:
        raise ValueError("二号机第一次校准表不是完整2001点")
    limits_ma = payload.get("limits_ma", {})
    if (
        float(limits_ma.get("gain_ma", -1.0)) != 145.0
        or float(limits_ma.get("soa_ma", -1.0)) != 145.0
    ):
        raise ValueError("二号机第一次校准表不是145 mA配置")
    limits = _fullband_accuracy_code_limits()
    ordered = sorted(rows.values(), key=lambda row: float(row["target_nm"]))
    points = []
    for expected_index, row in enumerate(ordered):
        expected_target = (
            FULLBAND_ACCURACY_START_NM
            + FULLBAND_ACCURACY_STEP_NM * expected_index
        )
        target = Decimal(str(row["target_nm"]))
        if target != expected_target:
            raise ValueError(
                f"二号机第{expected_index + 1}点目标波长应为{expected_target} nm"
            )
        if not bool(row.get("success")):
            raise ValueError(f"二号机第{expected_index + 1}点首次校准未通过")
        raw_codes = tuple(row.get("codes", ()))
        if len(raw_codes) != len(limits):
            raise ValueError(f"二号机第{expected_index + 1}点DAC码不完整")
        codes = tuple(
            parse_pi11210_dac_code(code, maximum)
            for code, maximum in zip(raw_codes, limits)
        )
        reading = row.get("reading") or {}
        measured = Decimal(str(reading.get("wavelength_nm", "NaN")))
        if not measured.is_finite():
            raise ValueError(f"二号机第{expected_index + 1}点实测波长无效")
        smsr = reading.get("side_mode_suppression_db")
        if smsr is not None and float(smsr) < 20.0:
            raise ValueError(f"二号机第{expected_index + 1}点首次校准不是单模")
        points.append(
            FullbandAccuracyPoint(
                index=expected_index,
                target_nm=float(target),
                measured_nm=float(measured),
                codes=codes,
            )
        )
    return points


def _validate_fullband_accuracy_points(points):
    if len(points) != FULLBAND_ACCURACY_POINT_COUNT:
        raise ValueError(f"2001点标定表实际只有{len(points)}点")
    measured = np.asarray([point.measured_nm for point in points], dtype=float)
    if not np.all(np.diff(measured) > 0.0):
        raise ValueError("2001点标定表的实测波长不是严格递增")
    return tuple(points)


def load_fullband_accuracy_table(base_dir=None, machine_id=None):
    """Load and strictly validate the final 2001-point operational table.

    The compact CSV is the runtime copy of the audited JSON export.  Refusing
    malformed, incomplete or out-of-limit data is intentional: silently
    scanning a partial table would make the wavelength axis look plausible
    while no longer representing the calibrated laser state.
    """
    base = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
    # Explicit fixture directories retain the historical machine-one CSV
    # behavior unless a machine id is explicitly requested.
    selected_machine = (
        "machine_1" if base_dir is not None and machine_id is None
        else str(machine_id or get_runtime_machine_id())
    )
    profile = runtime_fullband_profile(selected_machine)
    source = base / profile["source_path"]
    if profile["source_format"] == "calibration_json":
        return _validate_fullband_accuracy_points(
            _load_fullband_calibration_json(source)
        )
    if profile["source_format"] != "operational_csv":
        raise ValueError("不支持的2001点标定表格式")
    points = []
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        profiled = "limit_profile" in set(reader.fieldnames or ())
        limits = (
            _fullband_accuracy_code_limits()
            if profiled
            else _pi11210_table_code_limits()
        )
        required = {
            "index",
            "target_wavelength_nm",
            "measured_wavelength_nm",
            "single_mode_pass",
            "gain_code",
            "soa_code",
            "phase_code",
            "wavelength_a_code",
            "wavelength_b_code",
        }
        if not required.issubset(set(reader.fieldnames or ())):
            raise ValueError("2001点标定表字段不完整")
        for expected_index, row in enumerate(reader):
            if profiled and row["limit_profile"] != FULLBAND_ACCURACY_LIMIT_PROFILE:
                raise ValueError("不识别的2001点电流上限配置")
            index = int(row["index"])
            if index != expected_index:
                raise ValueError(f"2001点标定表索引不连续：{index}")
            expected_target = (
                FULLBAND_ACCURACY_START_NM
                + FULLBAND_ACCURACY_STEP_NM * expected_index
            )
            target = Decimal(row["target_wavelength_nm"])
            if target != expected_target:
                raise ValueError(
                    f"第{expected_index + 1}点目标波长应为{expected_target} nm"
                )
            if int(row["single_mode_pass"]) != 1:
                raise ValueError(f"第{expected_index + 1}点未通过单模验证")
            code_names = (
                    "gain_code",
                    "soa_code",
                    "phase_code",
                    "wavelength_a_code",
                    "wavelength_b_code",
                )
            normalizer = (
                parse_pi11210_dac_code
                if profiled
                else normalize_pi11210_calibration_code
            )
            codes = tuple(
                normalizer(row[name], maximum)
                for name, maximum in zip(code_names, limits)
            )
            if any(code < 0 or code > maximum
                   for code, maximum in zip(codes, limits)):
                raise ValueError(f"第{expected_index + 1}点DAC码超过安全上限")
            measured = Decimal(row["measured_wavelength_nm"])
            if not measured.is_finite():
                raise ValueError(f"第{expected_index + 1}点实测波长不是有限数")
            points.append(
                FullbandAccuracyPoint(
                    index=index,
                    target_nm=float(target),
                    measured_nm=float(measured),
                    codes=codes,
                )
            )
    return _validate_fullband_accuracy_points(points)


def nearest_fullband_accuracy_point(points, requested_nm):
    """Return the closest wavelength-meter-verified row without DAC interpolation.

    PI11210 tuning rows can cross cavity branches, so interpolating the five
    channel codes may create a mode hop or a double peak.  The measured
    wavelength, rather than the nominal 0.020-nm grid value, is therefore used
    for nearest-neighbour matching.
    """

    rows = tuple(points)
    if not rows:
        raise ValueError("2001点标定表为空")
    requested = float(requested_nm)
    if not math.isfinite(requested):
        raise ValueError("目标波长必须是有限数")
    minimum = float(rows[0].target_nm)
    maximum = float(rows[-1].target_nm)
    if requested < minimum or requested > maximum:
        raise ValueError(
            f"目标波长必须位于{minimum:.2f}–{maximum:.2f} nm之内"
        )
    return min(
        rows,
        key=lambda point: (
            abs(float(point.measured_nm) - requested),
            abs(float(point.target_nm) - requested),
            int(point.index),
        ),
    )


def select_fullband_target_range(points, start_nm, end_nm):
    """Return calibrated points whose target wavelengths lie in a closed range."""

    points = tuple(points)
    if not points:
        raise ValueError("没有可用的标定波长点")
    start_nm = float(start_nm)
    end_nm = float(end_nm)
    if not math.isfinite(start_nm) or not math.isfinite(end_nm):
        raise ValueError("开始和结束波长必须是有限数")
    if start_nm >= end_nm:
        raise ValueError("开始波长必须小于结束波长")

    minimum = float(points[0].target_nm)
    maximum = float(points[-1].target_nm)
    tolerance = 1e-9
    if start_nm < minimum - tolerance or end_nm > maximum + tolerance:
        raise ValueError(
            f"波长范围必须位于{minimum:.2f}–{maximum:.2f} nm之内"
        )
    selected = tuple(
        point
        for point in points
        if start_nm - tolerance <= float(point.target_nm) <= end_nm + tolerance
    )
    if not selected:
        raise ValueError("该范围内没有0.02 nm标定点，请适当扩大范围")
    return selected


def load_fullband_transition_guards(points, base_dir=None, machine_id=None):
    """Load the wavelength-branch guards paired with the 2001-point table."""
    base = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent
    selected_machine = (
        "machine_1" if base_dir is not None and machine_id is None
        else str(machine_id or get_runtime_machine_id())
    )
    profile = runtime_fullband_profile(selected_machine)
    guard_path = profile.get("guard_path")
    if not guard_path:
        return {}
    source = base / guard_path
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("scan_direction") != "forward":
        raise ValueError("无限准确标定表只允许严格正向扫描")
    profile = payload.get("limit_profile")
    if profile not in (None, FULLBAND_ACCURACY_LIMIT_PROFILE):
        raise ValueError("不识别的2001点过渡保护电流上限配置")
    limits = (
        _fullband_accuracy_code_limits()
        if profile == FULLBAND_ACCURACY_LIMIT_PROFILE
        else _pi11210_table_code_limits()
    )
    guards = {}
    for item in payload.get("guards", ()):
        target = Decimal(str(item["target_nm"]))
        offset = (target - FULLBAND_ACCURACY_START_NM) / FULLBAND_ACCURACY_STEP_NM
        if offset != offset.to_integral_value():
            raise ValueError(f"过渡保护波长不在0.02 nm网格上：{target}")
        index = int(offset)
        if index < 0 or index >= len(points):
            raise ValueError(f"过渡保护索引越界：{index}")
        if Decimal(str(points[index].target_nm)) != target:
            raise ValueError(f"过渡保护与第{index + 1}点不匹配")
        raw_stages = tuple(item.get("precondition_codes", ()))
        if not raw_stages or any(len(codes) != 5 for codes in raw_stages):
            raise ValueError(f"{target} nm过渡保护码不完整")
        stages = tuple(
            tuple(
                normalize_pi11210_calibration_code(code, maximum)
                for code, maximum in zip(codes, limits)
            )
            for codes in raw_stages
        )
        hold_s = float(item["hold_ms_per_stage"]) / 1000.0
        if not math.isfinite(hold_s) or hold_s <= 0.0:
            raise ValueError(f"{target} nm过渡保护等待时间无效")
        if index in guards:
            raise ValueError(f"过渡保护索引重复：{index}")
        guards[index] = FullbandTransitionGuard(
            target_index=index,
            target_nm=float(target),
            hold_s_per_stage=hold_s,
            precondition_codes=stages,
        )
    if set(guards) != FULLBAND_ACCURACY_GUARD_INDICES:
        raise ValueError(
            "过渡保护索引与最终2001点标定表不匹配："
            + ",".join(str(index) for index in sorted(guards))
        )
    return guards


def _merge_dense_peak_candidates(
    channel_results,
    allowed_channels,
    expected_peaks=None,
    *,
    minimum_amplitude_v=0.0,
    minimum_relative_amplitude=0.08,
    minimum_noise_multiple=10.0,
):
    """Reject insignificant bumps and merge one optical peak across channels.

    Dense scans can contain small baseline ripples which still satisfy the
    Gaussian fit checks.  A fixed peak count used to promote those ripples to
    real gratings.  The adaptive gate below combines the measured MAD-derived
    channel noise and a fraction of that channel's strongest peak.  There is no
    fixed 40 mV floor because a valid CH1 spectrum at 2 kOhm can be 13--26 mV.
    This retains genuinely weaker gratings while preventing a tiny ripple from
    being used merely to reach a configured count.
    """
    candidates = []
    channel_diagnostics = []
    for channel in allowed_channels:
        result = channel_results[int(channel)]
        noise = max(float(result.noise_sigma_v), 0.00025)
        valid_fits = tuple(fit for fit in result.fits if fit.valid)
        strongest = max(
            (max(float(fit.amplitude_v), 0.0) for fit in valid_fits),
            default=0.0,
        )
        amplitude_threshold = max(
            float(minimum_amplitude_v),
            float(minimum_relative_amplitude) * strongest,
            float(minimum_noise_multiple) * noise,
        )
        accepted_count = 0
        for fit in valid_fits:
            amplitude = max(float(fit.amplitude_v), 0.0)
            if amplitude < amplitude_threshold:
                continue
            rmse = max(float(fit.rmse_v), noise, 0.00025)
            quality = (
                amplitude
                * max(float(fit.r_squared), 0.05)
                / rmse
            )
            candidates.append({
                "channel": int(channel),
                "fit": fit,
                "quality": float(quality),
                "amplitude_threshold_v": float(amplitude_threshold),
            })
            accepted_count += 1
        channel_diagnostics.append({
            "channel": int(channel),
            "detected_candidate_count": int(result.candidate_count),
            "valid_fit_count": len(valid_fits),
            "accepted_peak_count": int(accepted_count),
            "rejected_as_tiny_count": int(len(valid_fits) - accepted_count),
            "noise_sigma_v": float(result.noise_sigma_v),
            "strongest_amplitude_v": float(strongest),
            "amplitude_threshold_v": float(amplitude_threshold),
        })

    candidates.sort(key=lambda item: item["fit"].center_nm)
    clusters = []
    for candidate in candidates:
        center = float(candidate["fit"].center_nm)
        if clusters and center - clusters[-1][-1]["fit"].center_nm <= 0.20:
            clusters[-1].append(candidate)
        else:
            clusters.append([candidate])
    merged = [max(cluster, key=lambda item: item["quality"]) for cluster in clusters]
    if expected_peaks is not None and len(merged) != int(expected_peaks):
        raise ValueError(
            f"自适应识别到{len(merged)}个有效峰，"
            f"与指定的{int(expected_peaks)}个峰不一致"
        )
    return tuple(merged), tuple(channel_diagnostics)


def _detect_measured_template_peaks(
    wavelength_nm,
    teacher_signal,
    *,
    channel,
    expected_peaks,
    minimum_amplitude_v=0.0,
    minimum_relative_amplitude=0.08,
    minimum_noise_multiple=10.0,
):
    """Locate complete non-Gaussian FBG regions on one measured CH1 template.

    A five-sample triangular smoother suppresses isolated ADC-code spikes but
    retains repeatable side lobes.  Prominence and minimum separation merge the
    lobes of one grating without fitting a Gaussian.  The result is accepted
    only when the measured count exactly matches the configured sensor.
    """

    x = np.asarray(wavelength_nm, dtype=float).reshape(-1)
    signal = np.asarray(teacher_signal, dtype=float).reshape(-1)
    if signal.shape != x.shape or x.size < 9:
        raise ValueError("实测模板的波长与CH1数据长度不一致")
    if not np.all(np.isfinite(signal)):
        raise ValueError("实测模板包含无效CH1数据")

    kernel = np.asarray((1.0, 2.0, 3.0, 2.0, 1.0), dtype=float)
    kernel /= float(np.sum(kernel))
    smoothed = np.convolve(np.pad(signal, (2, 2), mode="edge"), kernel, mode="valid")
    differences = np.diff(smoothed)
    difference_median = float(np.median(differences))
    robust_noise = (
        1.4826
        * float(np.median(np.abs(differences - difference_median)))
        / math.sqrt(2.0)
    )
    # One 12-bit ADC code is about 0.61 mV.  A 0.25 mV sigma floor keeps a
    # perfectly flat/quantized baseline from making every one-code ripple a
    # grating, while still accepting the measured 11--26 mV low-gain peaks.
    noise_sigma = max(robust_noise, 0.00025)
    baseline = float(np.quantile(smoothed, 0.20))
    strongest = max(0.0, float(np.max(smoothed)) - baseline)
    prominence_threshold = max(
        float(minimum_amplitude_v),
        float(minimum_relative_amplitude) * strongest,
        float(minimum_noise_multiple) * noise_sigma,
    )
    step_nm = float(np.median(np.diff(x)))
    minimum_separation_nm = 0.80
    minimum_distance_samples = max(
        1, int(math.ceil(minimum_separation_nm / step_nm))
    )
    raw_indices, _ = find_peaks(smoothed, distance=minimum_distance_samples)
    peak_indices, properties = find_peaks(
        smoothed,
        distance=minimum_distance_samples,
        prominence=prominence_threshold,
    )
    detected = int(len(peak_indices))
    if expected_peaks is not None and detected != int(expected_peaks):
        raise ValueError(
            f"实测CH{int(channel)}模板识别到{detected}个有效峰，"
            f"与指定的{int(expected_peaks)}个峰不一致；"
            "已禁止覆盖旧正式点表"
        )
    if detected <= 0:
        raise ValueError("实测CH1模板未识别到有效光栅峰")

    widths, _height, left_ips, right_ips = peak_widths(
        smoothed, peak_indices, rel_height=0.5
    )
    sample_axis = np.arange(x.size, dtype=float)
    centres = []
    for index in peak_indices:
        index = int(index)
        offset = 0.0
        if 0 < index < x.size - 1:
            denominator = (
                smoothed[index - 1]
                - 2.0 * smoothed[index]
                + smoothed[index + 1]
            )
            if abs(float(denominator)) > np.finfo(float).eps:
                offset = 0.5 * (
                    smoothed[index - 1] - smoothed[index + 1]
                ) / denominator
                offset = float(np.clip(offset, -0.5, 0.5))
        centres.append(float(np.interp(index + offset, sample_axis, x)))

    peaks = []
    for peak_number, index in enumerate(peak_indices):
        index = int(index)
        centre = centres[peak_number]
        neighbour_limits = []
        if peak_number:
            neighbour_limits.append(0.42 * (centre - centres[peak_number - 1]))
        if peak_number + 1 < detected:
            neighbour_limits.append(0.42 * (centres[peak_number + 1] - centre))
        half_span = min((0.80, *neighbour_limits))
        region_start = max(float(x[0]), centre - half_span)
        region_stop = min(float(x[-1]), centre + half_span)
        region_size = int(
            np.count_nonzero((x >= region_start) & (x <= region_stop))
        )
        if region_size < 5:
            raise ValueError(
                f"第{peak_number + 1}个峰周围不足5个2001点实测样本"
            )
        left_nm = float(np.interp(left_ips[peak_number], sample_axis, x))
        right_nm = float(np.interp(right_ips[peak_number], sample_axis, x))
        sigma_nm = max(step_nm, (right_nm - left_nm) / 2.354820045)
        prominence = float(properties["prominences"][peak_number])
        signal_to_noise = max(prominence / noise_sigma, 1.0)
        centre_std_pm = step_nm * 1000.0 / math.sqrt(signal_to_noise)
        descriptor = MeasuredTemplatePeak(
            center_nm=centre,
            amplitude_v=prominence,
            sigma_nm=float(sigma_nm),
            rmse_v=float(noise_sigma),
            center_std_pm=float(centre_std_pm),
            peak_index=index,
            region_start_nm=float(region_start),
            region_stop_nm=float(region_stop),
        )
        peaks.append(
            {
                "channel": int(channel),
                "fit": descriptor,
                "quality": float(prominence / max(prominence_threshold, 1e-12)),
                "amplitude_threshold_v": float(prominence_threshold),
                "region": (float(region_start), float(region_stop)),
            }
        )

    diagnostics = ({
        "channel": int(channel),
        "detector": "measured_template_prominence_v1",
        "detected_candidate_count": int(len(raw_indices)),
        "valid_fit_count": detected,
        "accepted_peak_count": detected,
        "rejected_as_tiny_count": max(0, int(len(raw_indices)) - detected),
        "noise_sigma_v": float(noise_sigma),
        "strongest_amplitude_v": float(strongest),
        "amplitude_threshold_v": float(prominence_threshold),
        "minimum_separation_nm": float(minimum_separation_nm),
        "smoothing": "triangular_5_samples",
    },)
    return tuple(peaks), diagnostics


def _select_fullband_rows_from_template_plan(
    points,
    plan,
    peaks,
    voltage_by_channel,
):
    """Render one optimizer plan using exact rows from the audited 2001 table."""

    if len(plan.peaks) != len(peaks):
        raise ValueError("实测模板峰数与稀疏选点计划不一致")
    voltages = np.asarray(voltage_by_channel, dtype=float)
    if voltages.shape != (4, len(points)) or not np.all(np.isfinite(voltages)):
        raise ValueError("自动选点原始ADC数组必须为完整4×2001点")
    rows = []
    for peak_number, (template, candidate) in enumerate(
        zip(plan.peaks, peaks, strict=True), start=1
    ):
        selected = tuple(int(index) for index in template.selected_indices)
        fit = candidate["fit"]
        core_offsets = [
            abs(float(points[index].measured_nm) - float(fit.center_nm))
            for index in selected[1:-1]
        ]
        for local_index, fullband_index in enumerate(selected):
            point = points[fullband_index]
            if int(point.index) != fullband_index:
                raise ValueError("2001点审计表索引与选点索引不一致")
            if local_index == 0:
                selection_role = "left_guard"
                template_role = "left_baseline_anchor"
            elif local_index == len(selected) - 1:
                selection_role = "right_guard"
                template_role = "right_baseline_anchor"
            else:
                selection_role = "peak_body"
                if fullband_index == int(template.peak_index):
                    template_role = "peak_amplitude_anchor"
                elif fullband_index < int(template.peak_index):
                    template_role = "left_shift_sensitive"
                else:
                    template_role = "right_shift_sensitive"
            source_channel = int(candidate["channel"])
            source_values = [
                float(voltages[channel, fullband_index]) for channel in range(4)
            ]
            rows.append({
                "peak_number": int(peak_number),
                "source_channel": source_channel,
                "fullband_index": fullband_index,
                "target_wavelength_nm": float(point.target_nm),
                "measured_wavelength_nm": float(point.measured_nm),
                "fitted_center_nm": float(fit.center_nm),
                "selection_half_span_nm": float(
                    max(
                        fit.center_nm - fit.region_start_nm,
                        fit.region_stop_nm - fit.center_nm,
                    )
                ),
                "selection_core_half_span_nm": float(max(core_offsets, default=0.0)),
                "selection_role": selection_role,
                "template_role": template_role,
                "selection_method": "measured_template_fisher_v1",
                "offset_sigma": float(
                    (float(point.measured_nm) - float(fit.center_nm))
                    / max(float(fit.sigma_nm), 1e-12)
                ),
                "template_signal_v": float(
                    template.selected_template_signal[local_index]
                ),
                "template_slope_v_per_nm": float(
                    template.selected_template_slope[local_index]
                ),
                "template_reliability_weight": float(
                    template.selected_reliability_weight[local_index]
                ),
                "projected_shift_information": float(
                    template.projected_shift_information
                ),
                "source_adc_v_by_channel": source_values,
                "source_adc_v": source_values[source_channel],
                # Never interpolate or synthesize a PI11210 current tuple.
                "codes": [int(code) for code in point.codes],
            })
    rows.sort(key=lambda row: int(row["fullband_index"]))
    expected_count = int(plan.total_points)
    indices = [int(row["fullband_index"]) for row in rows]
    if (
        len(rows) != expected_count
        or len(set(indices)) != expected_count
        or indices != sorted(indices)
    ):
        raise ValueError("实测模板稀疏点表不完整或存在重复点")
    return rows


def _allocate_mode_point_counts(peaks, total_points, minimum_per_peak):
    """Distribute a fixed frame budget across every automatically found peak."""
    peak_count = len(peaks)
    if peak_count <= 0:
        raise ValueError("未识别到达到门限的有效光栅峰")
    if peak_count * int(minimum_per_peak) > int(total_points):
        maximum = int(total_points) // int(minimum_per_peak)
        raise ValueError(
            f"识别到{peak_count}个有效峰，但{int(total_points)}点帧预算"
            f"在每峰至少{int(minimum_per_peak)}点时最多支持{maximum}个峰"
        )
    base, remainder = divmod(int(total_points), peak_count)
    counts = [base] * peak_count
    # Extra samples go to the broadest peaks, where they provide the greatest
    # improvement to the two-sided fit while preserving the exact frame size.
    widest_first = sorted(
        range(peak_count),
        key=lambda index: float(peaks[index]["fit"].sigma_nm),
        reverse=True,
    )
    for index in widest_first[:remainder]:
        counts[index] += 1
    return tuple(counts)


def _peak_sampling_targets(fit, count, neighbour_half_span_nm=None):
    """Place dense fit points on the peak body with one guard on each side.

    The former fixed 0.38 nm minimum half-span put roughly half of the point
    budget four to five sigma away from the narrow gratings measured on this
    instrument.  Those rows were valid laser rows, but their reflected voltage
    was naturally almost zero.  Keep only two outer baseline/shift guards and
    spend every other row on the peak top and its information-rich slopes.
    """
    count = int(count)
    if count < 5:
        raise ValueError("每个光栅峰至少需要5个选点")
    sigma_nm = float(fit.sigma_nm)
    if not math.isfinite(sigma_nm) or sigma_nm <= 0.0:
        raise ValueError("光栅峰宽无效，无法自适应选点")

    guard_half_span = float(np.clip(2.70 * sigma_nm, 0.10, 0.38))
    if neighbour_half_span_nm is not None:
        guard_half_span = min(
            guard_half_span,
            max(0.08, float(neighbour_half_span_nm)),
        )
    core_half_span = min(1.90 * sigma_nm, 0.82 * guard_half_span)
    core_count = count - 2
    core = np.linspace(
        float(fit.center_nm) - core_half_span,
        float(fit.center_nm) + core_half_span,
        core_count,
    )
    targets = [(float(fit.center_nm) - guard_half_span, "left_guard")]
    targets.extend((float(value), "peak_body") for value in core)
    targets.append((float(fit.center_nm) + guard_half_span, "right_guard"))
    return tuple(targets), guard_half_span, core_half_span


def _select_fullband_rows_for_peaks(
    points,
    peaks,
    counts,
    voltage_by_channel=None,
):
    """Select calibrated rows concentrated on each peak body and two guards."""
    if len(peaks) != len(counts):
        raise ValueError("峰数与每峰点数不匹配")
    wavelengths = np.asarray([point.measured_nm for point in points], dtype=float)
    requests = []
    centers = np.asarray(
        [float(candidate["fit"].center_nm) for candidate in peaks], dtype=float
    )
    for peak_index, (candidate, count) in enumerate(zip(peaks, counts, strict=True)):
        fit = candidate["fit"]
        neighbour_limits = []
        if peak_index:
            neighbour_limits.append(0.42 * (centers[peak_index] - centers[peak_index - 1]))
        if peak_index + 1 < len(centers):
            neighbour_limits.append(0.42 * (centers[peak_index + 1] - centers[peak_index]))
        neighbour_half_span = min(neighbour_limits) if neighbour_limits else None
        targets, guard_half_span, core_half_span = _peak_sampling_targets(
            fit,
            count,
            neighbour_half_span,
        )
        for target, role in targets:
            requests.append((
                float(target),
                int(peak_index),
                candidate,
                float(guard_half_span),
                float(core_half_span),
                str(role),
            ))

    selected_indices = set()
    selected = []
    voltages = None
    if voltage_by_channel is not None:
        voltages = np.asarray(voltage_by_channel, dtype=float)
        if voltages.shape != (4, len(points)) or not np.all(np.isfinite(voltages)):
            raise ValueError("自动选点原始ADC数组必须为完整4×2001点")
    for (
        target,
        peak_index,
        candidate,
        guard_half_span,
        core_half_span,
        role,
    ) in sorted(requests):
        insertion = int(np.searchsorted(wavelengths, target))
        chosen = None
        for radius in range(len(wavelengths)):
            options = (insertion - radius, insertion + radius)
            valid = [
                index for index in options
                if 0 <= index < len(wavelengths) and index not in selected_indices
            ]
            if valid:
                chosen = min(valid, key=lambda index: abs(wavelengths[index] - target))
                break
        if chosen is None:
            raise ValueError("无法为模式表分配唯一的2001点索引")
        selected_indices.add(chosen)
        point = points[chosen]
        fit = candidate["fit"]
        row = {
            "peak_number": peak_index + 1,
            "source_channel": int(candidate["channel"]),
            "fullband_index": int(point.index),
            "target_wavelength_nm": float(point.target_nm),
            "measured_wavelength_nm": float(point.measured_nm),
            "fitted_center_nm": float(fit.center_nm),
            "selection_half_span_nm": float(guard_half_span),
            "selection_core_half_span_nm": float(core_half_span),
            "selection_role": role,
            "offset_sigma": float(
                (float(point.measured_nm) - float(fit.center_nm))
                / float(fit.sigma_nm)
            ),
            "codes": [int(code) for code in point.codes],
        }
        if voltages is not None:
            source_values = [float(voltages[channel, chosen]) for channel in range(4)]
            row["source_adc_v_by_channel"] = source_values
            row["source_adc_v"] = source_values[int(candidate["channel"])]
        selected.append(row)
    selected.sort(key=lambda row: row["measured_wavelength_nm"])
    if len(selected) != sum(int(count) for count in counts):
        raise ValueError("模式表选点数量不完整")
    measured = np.asarray(
        [row["measured_wavelength_nm"] for row in selected], dtype=float
    )
    if len(set(selected_indices)) != len(selected) or np.any(np.diff(measured) <= 0.0):
        raise ValueError("模式表存在重复或非递增波长点")
    return selected


def select_mode_points_from_dense_spectrum(
    wavelength_nm,
    voltage_by_channel,
    points,
    *,
    expected_peaks=FINGER_EXPECTED_PEAK_COUNT,
    fbg_channels=FINGER_FBG_CHANNELS,
    stress_point_count=FINGER_STRESS_POINT_COUNT,
    temperature_point_count=FINGER_TEMPERATURE_POINT_COUNT,
    stress_minimum_per_peak=MINIMUM_POINTS_PER_PEAK["stress"],
    temperature_minimum_per_peak=MINIMUM_POINTS_PER_PEAK["temperature"],
    minimum_amplitude_v=0.0,
    minimum_relative_amplitude=0.08,
    minimum_noise_multiple=10.0,
    feedback_selectors=(PD_FEEDBACK_DEFAULT_SELECTOR,) * 2,
    sample_noise_std=None,
    stable_mask=None,
):
    """Derive stress/temperature candidate tables from one measured 2001-point scan.

    The current thumb defaults to exactly nine peaks on CH1.  A caller may pass
    another explicit channel/peak/budget contract for a different sensor.  Every
    emitted row comes directly from the wavelength-meter-audited 2001-point DAC
    table; this routine never invents or interpolates DAC currents.
    """
    x = np.asarray(wavelength_nm, dtype=float).reshape(-1)
    values = np.asarray(voltage_by_channel, dtype=float)
    if len(points) != x.size or values.shape != (4, x.size):
        raise ValueError("自动选点需要4通道完整2001点光谱")
    if x.size != FULLBAND_ACCURACY_POINT_COUNT or np.any(np.diff(x) <= 0.0):
        raise ValueError("自动选点只接受严格递增的完整2001点波长轴")
    if not np.all(np.isfinite(values)):
        raise ValueError("自动选点前必须完成所有2001点ADC采集")
    for fullband_index, point in enumerate(points):
        if int(point.index) != fullband_index:
            raise ValueError("2001点审计表索引与扫描顺序不一致")
        if not math.isclose(
            float(point.measured_nm),
            float(x[fullband_index]),
            rel_tol=0.0,
            abs_tol=5e-7,
        ):
            raise ValueError("ADC光谱波长轴与2001点审计表不一致")

    try:
        fbg_channels = tuple(int(channel) for channel in fbg_channels)
    except (TypeError, ValueError):
        raise ValueError("光栅分析通道配置无效") from None
    if (
        not fbg_channels
        or len(set(fbg_channels)) != len(fbg_channels)
        or any(channel < 0 or channel > 3 for channel in fbg_channels)
    ):
        raise ValueError("光栅分析通道必须是不重复的CH0～CH3")
    threshold_values = (
        float(minimum_amplitude_v),
        float(minimum_relative_amplitude),
        float(minimum_noise_multiple),
    )
    if (
        not all(math.isfinite(value) for value in threshold_values)
        or threshold_values[0] < 0.0
        or not 0.0 <= threshold_values[1] <= 1.0
        or threshold_values[2] < 0.0
    ):
        raise ValueError("光栅峰幅值/噪声门限配置无效")

    feedback_selectors = tuple(int(value) for value in feedback_selectors)
    if len(feedback_selectors) != 2 or any(
        value < 0 or value >= len(PD_FEEDBACK_KOHM_BY_SELECTOR)
        for value in feedback_selectors
    ):
        raise ValueError("CH0/CH1跨阻选择码必须均为0~3")
    # The measured-template optimizer works on one physical reflection path.
    # Keep the older multi-channel merger available for non-thumb legacy
    # reports, but the current nine-grating thumb is deliberately CH1-only.
    use_template_optimizer = len(fbg_channels) == 1
    if use_template_optimizer:
        source_channel = int(fbg_channels[0])
        stress_peaks, stress_detection = _detect_measured_template_peaks(
            x,
            values[source_channel],
            channel=source_channel,
            expected_peaks=expected_peaks,
            minimum_amplitude_v=minimum_amplitude_v,
            minimum_relative_amplitude=minimum_relative_amplitude,
            minimum_noise_multiple=minimum_noise_multiple,
        )
        temperature_peaks = stress_peaks
        temperature_detection = stress_detection
    else:
        channel_results = {
            channel: fit_dense_reflection_spectrum(x, values[channel])
            for channel in fbg_channels
        }
        stress_peaks, stress_detection = _merge_dense_peak_candidates(
            channel_results,
            fbg_channels,
            expected_peaks,
            minimum_amplitude_v=minimum_amplitude_v,
            minimum_relative_amplitude=minimum_relative_amplitude,
            minimum_noise_multiple=minimum_noise_multiple,
        )
        temperature_peaks, temperature_detection = _merge_dense_peak_candidates(
            channel_results,
            fbg_channels,
            expected_peaks,
            minimum_amplitude_v=minimum_amplitude_v,
            minimum_relative_amplitude=minimum_relative_amplitude,
            minimum_noise_multiple=minimum_noise_multiple,
        )
    if len(stress_peaks) != len(temperature_peaks):
        raise ValueError(
            "有效峰数不一致："
            f"应力通道识别{len(stress_peaks)}个，"
            f"温度通道识别{len(temperature_peaks)}个；"
            "请检查光路通道或跨阻档位"
        )
    for peak_number, (stress_peak, temperature_peak) in enumerate(
        zip(stress_peaks, temperature_peaks, strict=True), start=1
    ):
        separation_nm = abs(
            float(stress_peak["fit"].center_nm)
            - float(temperature_peak["fit"].center_nm)
        )
        if separation_nm > 0.20:
            raise ValueError(
                f"第{peak_number}个峰在应力与温度通道中的中心差"
                f"{separation_nm * 1000.0:.1f} pm，无法确认为同一光栅"
            )

    detected_peak_count = len(temperature_peaks)
    stress_point_count, stress_minimum_per_peak = validate_mode_point_budget(
        "stress",
        stress_point_count,
        detected_peak_count,
        stress_minimum_per_peak,
    )
    temperature_point_count, temperature_minimum_per_peak = validate_mode_point_budget(
        "temperature",
        temperature_point_count,
        detected_peak_count,
        temperature_minimum_per_peak,
    )
    stress_counts = _allocate_mode_point_counts(
        stress_peaks, stress_point_count, stress_minimum_per_peak
    )
    temperature_counts = _allocate_mode_point_counts(
        temperature_peaks, temperature_point_count, temperature_minimum_per_peak
    )

    stress_plan = None
    temperature_plan = None
    if use_template_optimizer:
        peak_regions = tuple(candidate["region"] for candidate in stress_peaks)
        stress_plan = optimize_ch1_sampling(
            x,
            values[source_channel],
            peak_regions,
            points_per_peak=stress_counts,
            expected_peak_count=detected_peak_count,
            sample_noise_std=sample_noise_std,
            stable_mask=stable_mask,
        )
        temperature_plan = optimize_ch1_sampling(
            x,
            values[source_channel],
            peak_regions,
            points_per_peak=temperature_counts,
            expected_peak_count=detected_peak_count,
            sample_noise_std=sample_noise_std,
            stable_mask=stable_mask,
        )
        stress_rows = _select_fullband_rows_from_template_plan(
            points, stress_plan, stress_peaks, values
        )
        temperature_rows = _select_fullband_rows_from_template_plan(
            points, temperature_plan, temperature_peaks, values
        )
    else:
        stress_rows = _select_fullband_rows_for_peaks(
            points,
            stress_peaks,
            stress_counts,
            values,
        )
        temperature_rows = _select_fullband_rows_for_peaks(
            points,
            temperature_peaks,
            temperature_counts,
            values,
        )
    if (
        len(stress_rows) != stress_point_count
        or len(temperature_rows) != temperature_point_count
    ):
        raise ValueError("自适应选点未生成完整的动态模式表")

    def peak_payload(peaks, plan=None):
        payload = []
        for index, candidate in enumerate(peaks):
            fit = candidate["fit"]
            item = {
                "peak_number": index + 1,
                "source_channel": int(candidate["channel"]),
                "center_nm": float(fit.center_nm),
                "amplitude_v": float(fit.amplitude_v),
                "sigma_nm": float(fit.sigma_nm),
                "r_squared": float(getattr(fit, "r_squared", 0.0)),
                "rmse_v": float(fit.rmse_v),
                "center_std_pm": float(fit.center_std_pm),
                "quality_score": float(candidate["quality"]),
            }
            if plan is not None:
                template = plan.peaks[index]
                item.update({
                    "fit_model": "measured_template_not_gaussian",
                    "region_start_nm": float(fit.region_start_nm),
                    "region_stop_nm": float(fit.region_stop_nm),
                    "teacher_peak_fullband_index": int(template.peak_index),
                    "selected_fullband_indices": [
                        int(value) for value in template.selected_indices
                    ],
                    "dense_wavelength_nm": [
                        float(value) for value in template.dense_wavelength_nm
                    ],
                    "dense_template_signal_v": [
                        float(value) for value in template.dense_signal
                    ],
                    "selected_template_signal_v": [
                        float(value) for value in template.selected_template_signal
                    ],
                    "selected_reliability_weight": [
                        float(value)
                        for value in template.selected_reliability_weight
                    ],
                    "projected_shift_information": float(
                        template.projected_shift_information
                    ),
                })
            payload.append(item)
        return payload

    strict_thumb_contract = (
        expected_peaks is not None
        and int(expected_peaks) == FINGER_EXPECTED_PEAK_COUNT
        and fbg_channels == FINGER_FBG_CHANNELS
        and use_template_optimizer
    )
    report = {
        "schema": (
            "equal_interval_auto_mode_selection_v4"
            if strict_thumb_contract
            else "equal_interval_auto_mode_selection_v3"
        ),
        "source_point_count": int(x.size),
        # Keep the legacy field so older viewers can still open this report;
        # it now records the automatically detected count rather than a target.
        "expected_peak_count": int(detected_peak_count),
        "detected_peak_count": int(detected_peak_count),
        "configured_peak_count": (
            None if expected_peaks is None else int(expected_peaks)
        ),
        "fbg_channels": list(fbg_channels),
        "selection": {
            "method": (
                "measured_template_fisher_v1"
                if use_template_optimizer
                else "legacy_gaussian_target_spacing"
            ),
            "uses_measured_non_gaussian_shape": bool(use_template_optimizer),
            "uses_dense_stability_mask": stable_mask is not None,
            "uses_dense_noise_weights": sample_noise_std is not None,
            "dac_row_policy": "copy_exact_fullband_2001_audited_row",
        },
        "detection": {
            "minimum_amplitude_v": float(minimum_amplitude_v),
            "minimum_relative_amplitude": float(minimum_relative_amplitude),
            "minimum_noise_multiple": float(minimum_noise_multiple),
            "merge_tolerance_nm": 0.20,
            "stress_channels": list(stress_detection),
            "temperature_channels": list(temperature_detection),
        },
        "analog_feedback": {
            "ch0_selector": feedback_selectors[0],
            "ch1_selector": feedback_selectors[1],
            "ch0_kohm": PD_FEEDBACK_KOHM_BY_SELECTOR[feedback_selectors[0]],
            "ch1_kohm": PD_FEEDBACK_KOHM_BY_SELECTOR[feedback_selectors[1]],
            "digital_gain": 1.0,
        },
        "stress": {
            "point_count": len(stress_rows),
            "minimum_points_per_peak": int(stress_minimum_per_peak),
            "points_per_peak": list(stress_counts),
            "peaks": peak_payload(stress_peaks, stress_plan),
            "rows": stress_rows,
        },
        "temperature": {
            "point_count": len(temperature_rows),
            "minimum_points_per_peak": int(temperature_minimum_per_peak),
            "points_per_peak": list(temperature_counts),
            "peaks": peak_payload(temperature_peaks, temperature_plan),
            "rows": temperature_rows,
        },
    }
    if strict_thumb_contract:
        report["sparse_sequence_validation"] = {
            "status": "pending",
            "validation_method": "hardware_sparse_order_replay",
            "stress_table_fingerprint_sha256": sparse_stress_table_fingerprint(
                stress_rows
            ),
            "validated_point_count": 0,
            "validated_repeats": 0,
            "validated_fullband_indices": [],
            "per_point_results": [],
            "failed_mode_indices": [],
            "certified_hidden_predecessors": [],
            "requires_hardware_sparse_order_replay": True,
            "message": (
                "45点顺序尚未在真实激光器上复现；"
                "验证通过前不允许覆盖旧正式点表"
            ),
        }
    return report


def decode_single_value_monitor_frame(frame):
    """Return PDT, PDR and CH0..CH3 codes from one EXTRA monitor frame."""
    if len(frame) != SINGLE_VALUE_FRAME_SIZE:
        return None
    if frame[0:4] != b"\xFF\xFF\x02\x02":
        return None
    return tuple(
        (frame[4 + channel * 2] << 8) | frame[5 + channel * 2]
        for channel in range(6)
    )


def build_extra_feedback_command(ch0_selector, ch1_selector):
    """Build an EXTRA-mode CH0/CH1 analogue feedback selection command."""
    selectors = (int(ch0_selector), int(ch1_selector))
    if any(
        value < 0 or value >= len(PD_FEEDBACK_KOHM_BY_SELECTOR)
        for value in selectors
    ):
        raise ValueError("RS2255跨阻选择码必须为0~3")
    command = bytearray(tx_size)
    command[0:4] = bytes((0xFF, 0xFF, 0x01, 0x06))
    command[4] = selectors[0]
    command[5] = selectors[1]
    return bytes(command)


def build_stress_feedback_command(ch0_selector, ch1_selector):
    """Arm fixed CH0/CH1 feedback selectors for the next stress session."""
    selectors = (int(ch0_selector), int(ch1_selector))
    if any(
        value < 0 or value >= len(PD_FEEDBACK_KOHM_BY_SELECTOR)
        for value in selectors
    ):
        raise ValueError("RS2255跨阻选择码必须为0~3")
    command = bytearray(tx_size)
    command[0:4] = bytes((0xFF, 0xFF, 0x01, 0x07))
    command[4] = selectors[0]
    command[5] = selectors[1]
    return bytes(command)


def decode_extra_feedback_status(frame):
    """Return CH0/CH1 IO readback selectors plus the ACK status byte."""
    if len(frame) != SINGLE_VALUE_FRAME_SIZE:
        return None
    if frame[0:4] != b"\xFF\xFF\x02\x03":
        return None
    if frame[4] not in range(4) or frame[5] not in range(4):
        return None
    if frame[6] not in (ACK_VALUE, ACK_ERROR_VALUE):
        return None
    return (int(frame[4]), int(frame[5])), int(frame[6])


def decode_stress_feedback_status(frame):
    """Return stress CH0/CH1 selector IO readback and ACK status."""
    if len(frame) != SINGLE_VALUE_FRAME_SIZE:
        return None
    if frame[0:4] != b"\xFF\xFF\x02\x04":
        return None
    if frame[4] not in range(4) or frame[5] not in range(4):
        return None
    if frame[6] not in (ACK_VALUE, ACK_ERROR_VALUE):
        return None
    return (int(frame[4]), int(frame[5])), int(frame[6])


def set_stress_feedback_and_verify(port, selectors, timeout_s=1.0):
    """Arm fixed stress feedback and return the physical choseA/B readback."""
    expected = tuple(int(value) for value in selectors)
    command = build_stress_feedback_command(*expected)
    rx = bytearray()
    last_actual = None
    port.reset_input_buffer()
    for _attempt in range(2):
        port.write(command)
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            waiting = int(getattr(port, "in_waiting", 0) or 0)
            data = port.read(min(max(waiting, SINGLE_VALUE_FRAME_SIZE), 4096))
            if not data:
                time.sleep(0.001)
                continue
            rx.extend(data)
            for frame in extract_single_value_frames(rx):
                result = decode_stress_feedback_status(frame)
                if result is None:
                    continue
                actual, status = result
                last_actual = actual
                if status == ACK_ERROR_VALUE:
                    raise RuntimeError("单片机报告应力跨阻IO核验失败")
                if status == ACK_VALUE and actual == expected:
                    return actual
        rx.clear()
    if last_actual is None:
        raise RuntimeError("未收到应力模式跨阻回读")
    raise RuntimeError(
        "应力跨阻回读与设定不一致："
        f"设定{expected[0]}/{expected[1]}，实际{last_actual[0]}/{last_actual[1]}"
    )


def set_stress_multirate_and_verify(
    port,
    configuration=MultirateConfiguration(),
    timeout_s=1.0,
    *, schedule_version=CURRENT_SCHEDULE_VERSION,
):
    """Arm/disarm the negotiated scheduler and require its exact ACK."""

    configuration.validate()
    command = (
        build_multirate_arm_command(configuration, schedule_version=schedule_version)
        if configuration.enabled
        else build_multirate_disarm_command(schedule_version=schedule_version)
    )
    rx = bytearray()
    last_ack = None
    port.reset_input_buffer()
    for _attempt in range(2):
        port.write(command)
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            waiting = int(getattr(port, "in_waiting", 0) or 0)
            data = port.read(min(max(waiting, SINGLE_VALUE_FRAME_SIZE), 4096))
            if not data:
                time.sleep(0.001)
                continue
            rx.extend(data)
            for frame in extract_single_value_frames(rx):
                ack = decode_multirate_ack(frame)
                if ack is None:
                    continue
                last_ack = ack
                try:
                    return (
                        validate_current_arm_ack(ack, configuration, requested_version=schedule_version)
                        if configuration.enabled
                        else validate_current_disarm_ack(ack, requested_version=schedule_version)
                    )
                except ValueError as exc:
                    raise RuntimeError(f"自适应扫描协议核验失败：{exc}") from exc
        rx.clear()
    if last_ack is None:
        raise RuntimeError("未收到自适应扫描02/05回读")
    raise RuntimeError("自适应扫描回读与设定不一致")


def _scan_frame_board_extension(frame):
    """Return the B5/4D payload from one length-validated scan frame."""

    if len(frame) < 8 or frame[:2] != b"\xEE\xEE":
        return None
    point_count = int.from_bytes(frame[2:4], "big")
    position = 4 + point_count * 8
    if position >= len(frame) or frame[position] != 0xAB:
        return None
    position += 1
    for _channel in range(4):
        if position >= len(frame):
            return None
        peak_count = int(frame[position])
        position += 1 + peak_count * 4
        if position > len(frame):
            return None
    position += 4  # temperature
    extension_header = position + 4  # four legacy gain-mask bytes
    if extension_header + 4 > len(frame) - 2:
        return None
    if frame[extension_header : extension_header + 3] != b"\xB5\x4D\x01":
        return None
    extension_length = int(frame[extension_header + 3])
    start = extension_header + 4
    end = start + extension_length
    if end > len(frame) - 2:
        return None
    return frame[start:end]


def enter_stress_multirate_and_verify_first_map(
    port,
    configuration=MultirateConfiguration(),
    timeout_s=4.0,
    *, schedule_version=CURRENT_SCHEDULE_VERSION,
):
    """Enter STRESS only after the current arm ACK and retain its first MAP."""

    if not configuration.enabled:
        raise ValueError("应力实时会话不允许关闭自适应调度")
    set_stress_multirate_and_verify(port, configuration, schedule_version=schedule_version)
    port.reset_input_buffer()
    port.write(build_work_mode_command(0))
    buffer = bytearray()
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        waiting = int(getattr(port, "in_waiting", 0) or 0)
        data = port.read(min(max(waiting, 128), 8192))
        if not data:
            time.sleep(0.001)
            continue
        buffer.extend(data)
        for frame in extract_raw_scan_frames(buffer):
            extension = _scan_frame_board_extension(frame)
            if extension is None:
                raise RuntimeError("首帧缺少板端调度状态")
            try:
                schedule = decode_board_frame_schedule(extension)
            except ValueError as exc:
                raise RuntimeError(f"首帧自适应调度无效：{exc}") from exc
            if schedule is None:
                raise RuntimeError("板卡未进入经确认的自适应扫描会话")
            if schedule.version != schedule_version:
                raise RuntimeError(
                    f"自适应扫描版本不一致："
                    f"板端v{schedule.version}/上位机v{schedule_version}"
                )
            if not schedule.is_complete_map:
                raise RuntimeError("自适应会话首帧不是真实45点MAP")
            if schedule.bandwidth_discontinuity != (configuration.map_period_frames == REFERENCE_MAP_PERIOD):
                raise RuntimeError("首帧带宽标记与完整谱参考/动态采集请求不一致")
            frames_queue.put(frame)
            return schedule
    raise RuntimeError("等待自适应会话首个45点MAP超时")


def set_soa_shutter_and_verify(port, timeout_s=3.0):
    """Enter EXTRA and prove that PI11210's hardware SOA gate is closed.

    A successful serial write is not sufficient for this safety operation.
    The function accepts only the exact five returned zero codes, the shutter
    mode byte, and an online/identified/initialised PI11210 status report.
    """

    if not getattr(port, "is_open", False):
        raise RuntimeError("串口已关闭，无法核验SOA安全关光")
    port.write(build_work_mode_command(2))
    try:
        port.flush()
    except (AttributeError, serial.SerialException):
        pass
    time.sleep(AP_MODE_SWITCH_SETTLE_S)
    port.reset_input_buffer()
    shutter_codes = (0, 0, 0, 0, 0)
    port.write(
        build_single_value_dac_command(
            shutter_codes,
            soa_mode=PI11210_SOA_SHUTTER_MODE,
        )
    )

    deadline = time.monotonic() + float(timeout_s)
    buffer = bytearray()
    while time.monotonic() < deadline:
        waiting = int(getattr(port, "in_waiting", 0) or 0)
        data = port.read(min(max(waiting, SINGLE_VALUE_FRAME_SIZE), 4096))
        if not data:
            time.sleep(0.001)
            continue
        buffer.extend(data)
        for frame in extract_single_value_frames(buffer):
            if frame[:4] != b"\xFF\xFF\x02\x00":
                continue
            returned = tuple(
                (frame[4 + channel * 2] << 8)
                | frame[5 + channel * 2]
                for channel in range(5)
            )
            if returned != shutter_codes:
                continue
            flags = int(frame[15])
            raw_status = (int(frame[16]) << 8) | int(frame[17])
            i2c_errors = (int(frame[18]) << 8) | int(frame[19])
            if int(frame[14]) != PI11210_SOA_SHUTTER_MODE:
                raise RuntimeError("SOA关光状态回读不一致")
            if not (flags & 0x80):
                raise RuntimeError("当前单片机未回报SOA硬件关光状态")
            if flags & 0x03 != 0x03:
                raise RuntimeError(
                    f"PI11210离线或器件ID不符（状态0x{raw_status:04X}）"
                )
            if not (flags & 0x04):
                raise RuntimeError(
                    f"PI11210关光写入失败（累计I²C错误{i2c_errors}次）"
                )
            return True
    raise RuntimeError("等待SOA安全关光回读超时")


def decode_extra_feedback_ack(frame):
    """Return successfully verified CH0/CH1 feedback selectors, or ``None``."""
    result = decode_extra_feedback_status(frame)
    if result is None or result[1] != ACK_VALUE:
        return None
    return result[0]


def fullband_accuracy_statistics(samples):
    """Return robust point statistics and a conservative settled flag.

    Columns are PDT, PDR, CH0, CH1, CH2 and CH3.  Only the newest samples are
    considered so an early transient does not poison an otherwise settled
    point.  The gate combines scaled MAD (noise) with first/last-window median
    drift, using absolute ADC-code floors at low signal.
    """
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("稳定判定需要N×6路ADC样本")
    if len(values) == 0:
        nan = np.full(6, np.nan, dtype=float)
        return nan, nan.copy(), nan.copy(), False
    values = values[-FULLBAND_ACCURACY_STAT_WINDOW:]
    median = np.median(values, axis=0)
    sigma = 1.4826 * np.median(np.abs(values - median), axis=0)
    span = max(3, len(values) // 3)
    drift = np.abs(
        np.median(values[-span:], axis=0)
        - np.median(values[:span], axis=0)
    )
    mad_limit = np.maximum(
        FULLBAND_ACCURACY_MAD_FLOOR_CODES,
        0.003 * np.maximum(64.0, median),
    )
    drift_limit = np.maximum(
        FULLBAND_ACCURACY_DRIFT_FLOOR_CODES,
        0.004 * np.maximum(64.0, median),
    )
    settled = bool(
        len(values) >= FULLBAND_ACCURACY_MIN_SAMPLES
        and np.all(sigma <= mad_limit)
        and np.all(drift <= drift_limit)
    )
    return median, sigma, drift, settled

def extract_single_value_frames(rx_buffer):
    """Extract aligned 20-byte EXTRA-mode frames from an arbitrary USB stream."""
    frames = []
    marker = b"\xFF\xFF"
    while True:
        marker_index = rx_buffer.find(marker)
        if marker_index < 0:
            if rx_buffer and rx_buffer[-1] == 0xFF:
                rx_buffer[:] = b"\xFF"
            else:
                rx_buffer.clear()
            break
        if marker_index:
            del rx_buffer[:marker_index]
        if len(rx_buffer) < SINGLE_VALUE_FRAME_SIZE:
            break
        # Resynchronise only at a complete, valid nested EXTRA header.  The
        # PI11210 I2C error counter may legitimately reach 0xFFFF, so a bare
        # marker in bytes 18..19 is no longer enough to declare truncation.
        nested_marker = -1
        search_from = 2
        while True:
            candidate_marker = rx_buffer.find(
                marker, search_from, SINGLE_VALUE_FRAME_SIZE
            )
            if candidate_marker < 0:
                break
            if (
                candidate_marker + 3 < len(rx_buffer)
                and rx_buffer[candidate_marker + 2] == 2
                and rx_buffer[candidate_marker + 3] in (0, 1, 2, 3, 4, 5)
            ):
                nested_marker = candidate_marker
                break
            search_from = candidate_marker + 2
        if nested_marker > 0:
            del rx_buffer[:nested_marker]
            continue
        candidate = bytes(rx_buffer[:SINGLE_VALUE_FRAME_SIZE])
        if candidate[2] != 2 or candidate[3] not in (0, 1, 2, 3, 4, 5):
            del rx_buffer[0]
            continue
        del rx_buffer[:SINGLE_VALUE_FRAME_SIZE]
        frames.append(candidate)
    return frames


def extract_manual_scan_frames(rx_buffer):
    """Extract aligned 20-byte MANUAL-mode status/ACK frames.

    MANUAL firmware used to emit temperature frames at a very high rate.  The
    desktop must therefore parse all complete frames already present in one
    large serial read instead of performing one blocking 20-byte read and one
    Qt signal emission per frame.
    """
    frames = []
    marker = b"\xFF\xFF"
    while True:
        marker_index = rx_buffer.find(marker)
        if marker_index < 0:
            rx_buffer[:] = b"\xFF" if rx_buffer[-1:] == b"\xFF" else b""
            break
        if marker_index:
            del rx_buffer[:marker_index]
        if len(rx_buffer) < SINGLE_VALUE_FRAME_SIZE:
            break
        candidate = bytes(rx_buffer[:SINGLE_VALUE_FRAME_SIZE])
        if candidate[2] != 1 or candidate[3] not in (0, 1):
            del rx_buffer[0]
            continue
        del rx_buffer[:SINGLE_VALUE_FRAME_SIZE]
        frames.append(candidate)
    return frames


def manual_scan_ack_status(frame):
    """Return 0x21/0xE1 for a valid MANUAL DAC response, else ``None``."""
    if len(frame) != SINGLE_VALUE_FRAME_SIZE:
        return None
    if frame[0:4] != b"\xFF\xFF\x01\x00":
        return None
    status = int(frame[4])
    return status if status in (ACK_VALUE, ACK_ERROR_VALUE) else None


def build_single_value_dac_command(
    codes,
    soa_mode=PI11210_SOA_SOURCE_MODE,
    *,
    fullband_2001=False,
):
    """Build an EXTRA-mode command with explicit SOA source/shutter state."""
    raw_values = list(codes)
    if len(raw_values) != 5:
        raise ValueError("single-value DAC command requires five codes")
    source_limits = (
        _fullband_accuracy_code_limits()
        if fullband_2001
        else _pi11210_table_code_limits()
    )
    normalizer = (
        parse_pi11210_dac_code
        if fullband_2001
        else normalize_pi11210_calibration_code
    )
    values = [normalizer(code, limit) for code, limit in zip(raw_values, source_limits)]
    soa_mode = parse_pi11210_dac_code(soa_mode, PI11210_SOA_SHUTTER_MODE)
    if soa_mode not in (PI11210_SOA_SOURCE_MODE, PI11210_SOA_SHUTTER_MODE):
        raise ValueError("invalid PI11210 SOA mode")
    if soa_mode == PI11210_SOA_SHUTTER_MODE:
        values[1] = PI11210_SOA_SHUTTER_CODE
    limits = list(source_limits)
    if soa_mode == PI11210_SOA_SHUTTER_MODE:
        limits[1] = PI11210_SOA_SHUTTER_CODE
    if any(code < 0 or code > limit for code, limit in zip(values, limits)):
        raise ValueError("single-value DAC code exceeds the configured safety limit")

    command = bytearray(tx_size)
    command[0:4] = bytes((0xFF, 0xFF, 0x00, PI11210_PROTOCOL_ID))
    for channel, code in enumerate(values):
        command[4 + channel * 2] = (code >> 8) & 0xFF
        command[5 + channel * 2] = code & 0xFF
    command[14] = soa_mode
    return bytes(command)


def decode_single_value_rt_frame(frame):
    """Return CH0..CH3 ADC codes from one valid EXTRA-mode monitor frame."""
    values = decode_single_value_monitor_frame(frame)
    return None if values is None else values[2:]


def channels_with_valid_peaks(peak_fits, visible_channels, fbg_channels):
    """Select only visible FBG channels that currently contain a valid peak."""
    candidates = set(visible_channels) & set(fbg_channels)
    return {
        channel for channel in candidates
        if channel < len(peak_fits)
        and any(getattr(fit, "valid", False) for fit in peak_fits[channel])
    }

def pd_adc_code_to_current_ma(adc_code):
    """Convert a PDR/PDT ADC code to photodiode current in mA."""
    adc_voltage = float(adc_code) * PD_ADC_REFERENCE_V / PD_ADC_CODE_COUNT
    current_ma = (PD_TIA_BIAS_V - adc_voltage) * 1000.0 / PD_TIA_FEEDBACK_OHM
    return max(0.0, current_ma), adc_voltage

rx_queue = deque(maxlen=4000)

rx_buffer = bytearray()

frames_queue = LatestFrameBuffer(maxlen=200)

def get_desktop_path():
    return str(Path.home() / "Desktop")

def serial_write(info: bytes):
    ser.write(info)

def build_work_mode_command(mode: int):
    """Build one fixed-length MCU work-mode command."""
    if mode not in (0, 1, 2, 3):
        raise ValueError(f"Unsupported MCU work mode: {mode}")

    command_frame = bytearray(tx_size)
    command_frame[0] = 0xFF
    command_frame[1] = 0xFF
    command_frame[2] = 0x01
    command_frame[3] = 0x02
    command_frame[8] = mode
    return bytes(command_frame)


def send_work_mode_command(mode: int):
    """Send the selected MCU work mode after the serial port is open."""
    serial_write(build_work_mode_command(mode))

def try_write(inst, cmd):
    try:
        inst.write(cmd)
        return True
    except Exception:
        return False

def parse_arr(reply: str):
    if reply is None:
        return []
    s = str(reply).strip().replace("\r", "").replace("\n", "")
    if not s:
        return []
    out = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(float(p))
        except Exception:
            pass
    return out

def trunc6(x):
    """mW 建议保留 6 位小数更有意义"""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    try:
        return f"{float(x):.6f}"
    except Exception:
        return "nan"

def trunc4(x):
    """mW 建议保留 6 位小数更有意义"""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    try:
        return f"{float(x):.4f}"
    except Exception:
        return "nan"

def trunc3(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    try:
        return f"{math.trunc(float(x) * 1000) / 1000.0:.3f}"
    except Exception:
        return "nan"

def dbm_to_mw(dbm: float) -> float:
    # mW = 10^(dBm/10)
    return 10 ** (dbm / 10.0)

def get_top2_peaks_from_arrays(wav_list, pow_list_dbm):
    n = min(len(wav_list), len(pow_list_dbm))
    wav_list = wav_list[:n]
    pow_list_dbm = pow_list_dbm[:n]

    def is_small_int(x):
        return isinstance(x, (int, float)) and abs(x - round(x)) < 1e-9 and 0 <= x <= 200

    # 去掉头部计数 N
    if n >= 2 and is_small_int(wav_list[0]) and is_small_int(pow_list_dbm[0]):
        w1 = wav_list[1]
        if (500 <= w1 <= 20000) or (1e-9 <= w1 <= 1e-3):
            wav_list = wav_list[1:]
            pow_list_dbm = pow_list_dbm[1:]
            n -= 1

    # 单位判断：m->nm / nm直接用
    wav_nm = []
    for w in wav_list:
        w = float(w)
        if 1e-9 <= w <= 1e-3:
            wav_nm.append(w * 1e9)
        else:
            wav_nm.append(w)

    valid = [i for i in range(n) if 500 <= wav_nm[i] <= 20000]
    if not valid:
        return float("nan"), float("nan"), float("nan"), float("nan")

    valid.sort(key=lambda i: float(pow_list_dbm[i]), reverse=True)

    i1 = valid[0]
    wav1_nm = wav_nm[i1]
    pow1_mw = dbm_to_mw(float(pow_list_dbm[i1]))

    wav2_nm = pow2_mw = float("nan")
    if len(valid) >= 2:
        i2 = valid[1]
        wav2_nm = wav_nm[i2]
        pow2_mw = dbm_to_mw(float(pow_list_dbm[i2]))

    return wav1_nm, pow1_mw, wav2_nm, pow2_mw

def read_two_peaks_stable(inst):
    inst.write(":INIT")
    inst.write("*TRG")
    wav_reply = inst.query(":FETC:ARR:POW:WAV?")
    pow_reply = inst.query(":FETC:ARR:POW?")  # dBm
    wav_list = parse_arr(wav_reply)
    pow_list_dbm = parse_arr(pow_reply)
    return get_top2_peaks_from_arrays(wav_list, pow_list_dbm)

class MQComboBox(QtWidgets.QComboBox):
    def __init__(self):
        super().__init__()
        self.refresh_ports()

    def showPopup(self):
        self.refresh_ports()
        super().showPopup()

    def refresh_ports(self):
        current_text = self.currentText()

        self.blockSignals(True)
        self.clear()

        for p in list_ports.comports():
            text = f"{p.device} {p.description}"
            self.addItem(text, p.device)

        if current_text:
            index = self.findText(current_text)
            if index >= 0:
                self.setCurrentIndex(index)
        
        self.blockSignals(False)

class LogWidget(QPlainTextEdit):
    def __init__(self, max_lines=1000, readOnly=True):
        super().__init__()

        self.setReadOnly(readOnly)
        self.max_lines = max_lines

        self.setStyleSheet("""
        QPlainTextEdit{
            background-color: #1e1e1e;
            color: #dddddd;
            font-family: Consolas;
            font-size: 12px
        }
        """)

    def log(self, message, level="INFO"):
        time_str = datetime.now().strftime("%H:%M:%S")
        
        if level == "ERROR":
            text = f'<span style="color:#ff5555">[{time_str}] {message}</span>'
        elif level == "WARNING":
            text = f'<span style="color:#ffaa00">[{time_str}] {message}</span>'
        else:
            text = f'<span style="color:#dddddd">[{time_str}] {message}</span>'

        self.appendHtml(text)

        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

        if self.blockCount() > self.max_lines:
            cursor = self.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.select(cursor.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()

def read_available_scan_bytes(port):
    """Block for the first byte only, then drain the available serial backlog.

    A fixed 64-byte read can wait until timeout with a complete short tail
    already buffered. Adaptive reads also drain bursts without hundreds of
    repeated driver calls. Keep the batch bounded for stop responsiveness.
    """
    waiting = int(port.in_waiting or 0)
    return port.read(min(max(waiting, 1), 65536))


class peakWorker(QThread):
    temp_signal = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.running = True
        self.perx_size = 20

    def run(self):
        def process_buffer(buf):
            if not self.running:
                return
            for frame in extract_raw_scan_frames(buf):
                frames_queue.put(frame)

        def process_temperature(buf):
            while self.running:
                idx = buf.find(b'\xff\xff')
                if idx<0:
                    break

                temp_frame = buf[idx:idx+self.perx_size]
                if len(temp_frame)<self.perx_size:
                    break
                del buf[idx:idx+self.perx_size]

                temp_int_high = temp_frame[4]
                temp_int_low = temp_frame[5]
                temp_dec_high = temp_frame[6]
                temp_dec_low = temp_frame[7]
                temperature = (temp_int_high<<8) + temp_int_low + \
                                ((temp_dec_high<<8)+temp_dec_low)*0.0001
                self.temp_signal.emit(temperature)

        while self.running:
            with ser_cond:
                while self.running and not ser_open:
                    ser_cond.wait()
                    rx_buffer.clear()
                if not self.running:
                    break
            if ser.is_open:
                try:
                    data = read_available_scan_bytes(ser)
                except Exception as e:
                    if not self.running:
                        break
                    print("Serial Error:",e)
                    continue
                rx_buffer.extend(data)
                # print(rx_buffer)
                # process_temperature(rx_buffer)
                process_buffer(rx_buffer)


class StableSingleReferenceWorker(QThread):
    """Temporarily own the serial stream and acquire a stable single-value scan."""

    progress = pyqtSignal(int, int)

    def __init__(self, port, dac_rows, channels, feedback_selectors=(1, 1), parent=None):
        super().__init__(parent)
        self.port = port
        self.dac_rows = tuple(tuple(row) for row in dac_rows)
        self.channels = frozenset(int(channel) for channel in channels)
        self.feedback_selectors = tuple(int(value) for value in feedback_selectors)
        self.running = True
        self.result = None
        self.error_message = None
        self.cancelled = False
        self._rx_buffer = bytearray()

    @staticmethod
    def _ack_codes(frame):
        if len(frame) != SINGLE_VALUE_FRAME_SIZE or frame[0:4] != b"\xFF\xFF\x02\x00":
            return None
        return tuple(
            (frame[4 + channel * 2] << 8) | frame[5 + channel * 2]
            for channel in range(5)
        )

    def _write_and_measure(self, codes, settle_s, minimum_frames):
        expected = tuple(int(code) for code in codes)
        self.port.write(build_single_value_dac_command(expected))
        ack_deadline = time.monotonic() + 2.0
        settle_deadline = None
        samples = []

        while self.running:
            now = time.monotonic()
            if settle_deadline is None and now >= ack_deadline:
                raise RuntimeError("2秒内未收到单片机DAC回读")
            if settle_deadline is not None and now >= settle_deadline:
                if len(samples) >= minimum_frames:
                    if not samples:
                        return None
                    tail_count = min(len(samples), STABLE_REFERENCE_MAX_MEDIAN_FRAMES)
                    return np.median(
                        np.asarray(samples[-tail_count:], dtype=float), axis=0
                    )
                if now >= settle_deadline + 1.0:
                    raise RuntimeError(
                        f"稳定单值监测帧不足（{len(samples)}/{minimum_frames}）"
                    )

            waiting = int(getattr(self.port, "in_waiting", 0) or 0)
            data = self.port.read(min(max(waiting, 64), 4096))
            if not data:
                continue
            self._rx_buffer.extend(data)
            for frame in extract_single_value_frames(self._rx_buffer):
                returned = self._ack_codes(frame)
                if returned is not None:
                    if returned != expected:
                        continue
                    settle_deadline = time.monotonic() + float(settle_s)
                    samples.clear()
                    continue
                values = decode_single_value_rt_frame(frame)
                if values is not None and settle_deadline is not None:
                    samples.append(values)

        self.cancelled = True
        return None

    def run(self):
        try:
            if not self.dac_rows:
                raise RuntimeError("未找到应力扫描DAC表")
            self.port.reset_input_buffer()
            send_work_mode_command(2)
            time.sleep(0.12)
            self.port.reset_input_buffer()
            self._rx_buffer.clear()
            set_stress_feedback_and_verify(
                self.port, self.feedback_selectors
            )
            self.port.reset_input_buffer()
            self._rx_buffer.clear()

            # The final table point is the real predecessor of point zero in a
            # cyclic stress scan.  Every later point naturally inherits the
            # previous target, matching the laser's normal traversal direction.
            self._write_and_measure(
                self.dac_rows[-1], STABLE_REFERENCE_PRECONDITION_S, 0
            )

            measured = np.full((4, len(self.dac_rows)), np.nan, dtype=float)
            for index, codes in enumerate(self.dac_rows):
                if not self.running:
                    self.cancelled = True
                    break
                adc_codes = self._write_and_measure(
                    codes,
                    STABLE_REFERENCE_SETTLE_S,
                    STABLE_REFERENCE_MIN_FRAMES,
                )
                if adc_codes is None:
                    self.cancelled = True
                    break
                for channel in self.channels:
                    measured[channel, index] = (
                        float(adc_codes[channel])
                        * PD_ADC_REFERENCE_V / PD_ADC_CODE_COUNT
                    )
                self.progress.emit(index + 1, len(self.dac_rows))

            if not self.cancelled:
                self.result = measured
        except Exception as exc:
            if self.running:
                self.error_message = str(exc)
            else:
                self.cancelled = True
        finally:
            # Stay in EXTRA.  The owning GraphWindow must re-arm the volatile
            # v3 scheduler and verify its first MAP before restarting the live
            # reader; entering STRESS here would create an unverified legacy
            # full-map session.
            try:
                if self.port.is_open:
                    send_work_mode_command(2)
                    time.sleep(AP_MODE_SWITCH_SETTLE_S)
                    self.port.reset_input_buffer()
            except Exception as exc:
                if self.error_message is None and not self.cancelled:
                    self.error_message = f"返回安全单值状态失败：{exc}"


class APWorker(QThread):
    log_signal = pyqtSignal(str,str)
    temp_signal = pyqtSignal(float)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.aprx_size = 20
        self.running = True
        self.temperature = 0
        self.flag_queue = Queue()
        self.excel_code_columns = None

    def log(self, msg, level="info"):
        self.log_signal.emit(msg, level)

    def configure_excel_columns(self, header_row):
        self.excel_code_columns = resolve_pi11210_excel_code_columns(header_row)
        return self.excel_code_columns

    def stop(self):
        """Request a prompt, cooperative stop from either ACK or VISA work."""
        self.running = False
        self.requestInterruption()
        # Wake a queue wait immediately; the sentinel is never interpreted as
        # an ACK and avoids making the GUI wait for the serial timeout.
        self.flag_queue.put(None)

    def _interruptible_wait(self, duration_s):
        deadline = time.monotonic() + max(0.0, float(duration_s))
        while self.running:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return True
            time.sleep(min(ACK_WAIT_SLICE_S, remaining))
        return False

    def _send_command_with_ack(self, command):
        """Send one row with bounded retries and return its MANUAL ACK frame."""
        for attempt in range(ACK_MAX_ATTEMPTS):
            if not self.running:
                return None
            serial_write(command)
            deadline = time.monotonic() + ACK_ATTEMPT_TIMEOUT_S
            while self.running:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                try:
                    frame = self.flag_queue.get(
                        timeout=min(ACK_WAIT_SLICE_S, remaining)
                    )
                except Empty:
                    continue
                if frame is None:
                    continue
                status = manual_scan_ack_status(frame)
                if status == ACK_ERROR_VALUE:
                    raise RuntimeError(
                        "单片机报告PI11210写入失败（ACK=0xE1）"
                    )
                if status == ACK_VALUE:
                    return frame
            if self.running and attempt + 1 < ACK_MAX_ATTEMPTS:
                self.log("当前点DAC回读超时，正在有限重发", "WARNING")
        if not self.running:
            return None
        raise RuntimeError(
            f"{ACK_MAX_ATTEMPTS}次写入后仍未收到单片机DAC回读，已停止扫描"
        )

    def _safe_shutdown_laser(self):
        """Return to EXTRA and verify a bounded zero-current SOA shutter ACK."""
        if not ser.is_open:
            self.log("串口已关闭，无法验证SOA安全关光", "ERROR")
            return False

        try:
            # MANUAL has no shutter field.  Move to EXTRA first, then use the
            # firmware's explicit CLR zero-current gate.  All other codes are
            # zeroed only after the firmware has asserted that hardware gate.
            send_work_mode_command(2)
            try:
                ser.flush()
            except (AttributeError, serial.SerialException):
                pass
            time.sleep(AP_MODE_SWITCH_SETTLE_S)
            ser.reset_input_buffer()
            shutter_codes = (0, 0, 0, 0, 0)
            serial_write(
                build_single_value_dac_command(
                    shutter_codes, soa_mode=PI11210_SOA_SHUTTER_MODE
                )
            )

            deadline = time.monotonic() + AP_SAFE_SHUTDOWN_TIMEOUT_S
            buffer = bytearray()
            while time.monotonic() < deadline:
                waiting = int(getattr(ser, "in_waiting", 0) or 0)
                data = ser.read(min(max(waiting, 64), 4096))
                if not data:
                    continue
                buffer.extend(data)
                for frame in extract_single_value_frames(buffer):
                    if frame[0:4] != b"\xFF\xFF\x02\x00":
                        continue
                    returned = tuple(
                        (frame[4 + channel * 2] << 8)
                        | frame[5 + channel * 2]
                        for channel in range(5)
                    )
                    if returned != shutter_codes:
                        continue
                    if not (frame[15] & 0x80):
                        raise RuntimeError("当前单片机固件未回报PI11210安全状态")
                    flags = int(frame[15])
                    raw_status = (int(frame[16]) << 8) | int(frame[17])
                    i2c_errors = (int(frame[18]) << 8) | int(frame[19])
                    if int(frame[14]) != PI11210_SOA_SHUTTER_MODE:
                        raise RuntimeError("SOA关光状态回读不一致")
                    if flags & 0x03 != 0x03:
                        raise RuntimeError(
                            f"PI11210离线或器件ID不符（状态0x{raw_status:04X}）"
                        )
                    if not flags & 0x04:
                        raise RuntimeError(
                            f"PI11210关光写入失败（累计I²C错误{i2c_errors}次）"
                        )
                    self.log("已切回单值安全态，SOA CLR零电流关光已回读确认")
                    return True
            raise RuntimeError("等待SOA安全关光回读超时")
        except Exception as exc:
            self.log(f"SOA安全关光未能验证：{exc}", "ERROR")
            return False

    def excel_operate(self, iter_excel):
        """
        返回：(cmd_bytes, info_str)
        """
        try:
            row = next(iter_excel)
        except StopIteration:
            return None, None
        try:
            if self.excel_code_columns is None:
                raise ValueError("尚未根据Excel表头识别五路DAC码列")
            raw_codes = tuple(row[index] for index in self.excel_code_columns)
            limits = _pi11210_table_code_limits()
            codes = tuple(
                normalize_pi11210_calibration_code(value, maximum)
                for value, maximum in zip(raw_codes, limits)
            )
            readGain, readSOA, readPhase, readwaveA, readwaveB = codes
            cmd = build_single_value_dac_command(codes)
            info = f"Gain={readGain}, SOA={readSOA}, phase={readPhase}, waveA={readwaveA}, waveB={readwaveB}"
            return cmd, info
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"Excel行中的五路DAC码无效：{exc}") from exc

    def serial_recv_loop(self):
        buffer = bytearray()
        latest_temperature = None
        last_temperature_emit = 0.0
        while self.running:
            try:
                # Drain the native-USB stream in batches.  Older MANUAL
                # firmware can queue thousands of temperature frames before a
                # DAC ACK, so a single-frame read would leave the ACK buried.
                data = ser.read(4096)
            except Exception as exc:
                if self.running:
                    self.log(f"串口接收失败：{exc}", "ERROR")
                break
            if not data:
                continue
            buffer.extend(data)
            for frame in extract_manual_scan_frames(buffer):
                b = list(frame)
                if manual_scan_ack_status(frame) is not None:
                    # Queue both success and explicit PI11210 failure.  The
                    # latter used to be discarded and caused endless resend.
                    self.flag_queue.put(frame)
                elif b[3] == 0x01:
                    temp_int_high = b[4]
                    temp_int_low = b[5]
                    temp_dec_high = b[6]
                    temp_dec_low = b[7]
                    self.temperature = (temp_int_high<<8) + temp_int_low + \
                                    ((temp_dec_high<<8)+temp_dec_low)*0.0001
                    latest_temperature = self.temperature

            # Publish at most 20 UI updates/s regardless of MCU frame rate.
            # ACK frames above are never throttled or dropped.
            now = time.monotonic()
            if (
                latest_temperature is not None
                and now - last_temperature_emit >= 0.05
            ):
                self.temp_signal.emit(latest_temperature)
                latest_temperature = None
                last_temperature_emit = now

    def connect_aq6150(self, rm):
        # 只看资源信息，不真正打开设备
        infos = rm.list_resources_info()

        # 只保留 GPIB INSTR，完全跳过 ASRL/USB/TCPIP
        gpib_resources = [
            name for name, info in infos.items()
            if info.interface_type == InterfaceType.gpib
            and info.resource_class == "INSTR"
        ]

        for res in gpib_resources:
            inst = None
            try:
                inst = rm.open_resource(res)
                inst.timeout = VISA_TIMEOUT_MS
                inst.read_termination = "\n"
                inst.write_termination = "\n"

                idn = inst.query("*IDN?").strip()
                if is_supported_aq6150_identity(idn):
                    self.log(f"找到 AQ6150: {res} -> {idn}")
                    return inst

                inst.close()

            except Exception:
                if inst is not None:
                    try:
                        inst.close()
                    except:
                        pass

        self.log("未找到目标 AQ6150", "ERROR")

    def run(self):
        recv_thread = None
        try:
            XLSX_BASENAME = f"AQ6150B_log_{time.strftime('%H%M%S')}.xlsx"
            excel_path = self.file_path
            if not excel_path:
                self.log("未选择文件", "WARNING")
                return
            self.log(f"执行文件: {excel_path}")

            wb_in = load_workbook(excel_path, read_only=True, data_only=True)
            ws_in = wb_in.active
            header_row = next(
                ws_in.iter_rows(min_row=1, max_row=1, values_only=True),
                (),
            )
            code_columns = self.configure_excel_columns(header_row)
            self.log(
                "已识别五路DAC码列："
                + ", ".join(
                    f"{channel}=第{index + 1}列"
                    for channel, index in zip(
                        PI11210_EXCEL_CODE_CHANNELS, code_columns
                    )
                )
            )
            iter_excel = ws_in.iter_rows(min_row=2, values_only=True)

            rm = pyvisa.ResourceManager()
            inst = self.connect_aq6150(rm)
            if inst is None:
                rm.close()
                raise RuntimeError("未找到指定的 AQ6150 波长计")
            recv_thread = threading.Thread(target=self.serial_recv_loop, daemon=True)
            recv_thread.start()
            wb_out = None
            xlsx_path = None

            try:
                # 设置单位仍然用 dBm（我们在软件里转 mW）
                try_write(inst, ":INIT:CONT OFF")
                try_write(inst, ":TRIG:SOUR BUS")
                try_write(inst, ":FORM:NDAT 0NM")
                try_write(inst, ":UNIT:POW DBM")

                desktop = get_desktop_path()
                xlsx_path = Path(desktop) / XLSX_BASENAME
                # file_exists = os.path.isfile(xlsx_path)

                # with open(xlsx_path, "a", newline="", encoding="utf-8") as f:
                #     writer = csv.writer(f)
                wb_init = openpyxl.Workbook()
                wb_init.save(xlsx_path)
                wb_out = load_workbook(xlsx_path)
                ws_out = wb_out.active

                head_data = ["timestamp_iso",
                        "peak1_wavelength_nm", "peak1_power_mW",
                        "peak2_wavelength_nm", "peak2_power_mW", "temperature",
                        "pdr", "pdt", "ratio_rt", "r_sat_sa"]
                gap_len = len(head_data)+1

                self.log(
                    "开始记录（每点等待DAC回读；超时有限重发后安全停止）。"
                )

                count = 0
                since_flush = 0
                loop_time = -1

                # 缓存当前行（ACK 成功后才读取下一行）
                current_cmd = None
                current_info = None
                last_current = True

                while self.running:
                    # 循环写入表头
                    if last_current:
                        count=0
                        loop_time += 1
                        iter_excel = ws_in.iter_rows(min_row=2, values_only=True)
                        for i, value in enumerate(head_data):
                            ws_out.cell(row=1, column=i+1+loop_time*gap_len, value=value)
                        last_current = False
                    
                    # 只有当当前没有待发送行时，才取 Excel 下一行
                    if current_cmd is None:
                        cmd, info = self.excel_operate(iter_excel)
                        if cmd is None:
                            last_current = True
                            continue
                        current_cmd, current_info = cmd, info

                    # 打印 Excel 读取内容（可控频率）
                    if PRINT_EXCEL_EVERY > 0 and ((count + 1) % PRINT_EXCEL_EVERY == 0):
                        self.log("----excel----")
                        self.log(current_info)
                        self.log("----excel----")

                    while True:
                        try:
                            self.flag_queue.get_nowait()
                        except Empty:
                            break

                    # 发送并等待可取消、有总时限的ACK。
                    flag = self._send_command_with_ack(current_cmd)
                    if flag is None:
                        break

                    # ACK 成功：计数并“推进到下一行”
                    count += 1
                    self.log(f"循环:{loop_time+1}  计数:{count}")
                    current_cmd = None
                    current_info = None

                    # 读 PDT和PDR
                    pdt = (flag[5]<<8)+flag[6]
                    pdr = (flag[8]<<8)+flag[9]

                    pdt_sa,pdr_sa = flag[7],flag[10]

                    pdt = pdt*2.5/4096
                    pdr = pdr*2.5/4096

                    pdt = (1.25-pdt)/2
                    pdr = (1.25-pdr)/2

                    # 读 AQ（失败短重试）
                    ts = datetime.now().isoformat(timespec="milliseconds")
                    wav1_nm = pow1_mw = wav2_nm = pow2_mw = float("nan")
                    for _ in range(VISA_RETRY + 1):
                        if not self.running:
                            break
                        try:
                            wav1_nm, pow1_mw, wav2_nm, pow2_mw = read_two_peaks_stable(inst)
                            break
                        except Exception:
                            if not self.running:
                                break
                            try:
                                inst.close()
                            except Exception:
                                pass
                            inst = self.connect_aq6150(rm)
                            if inst is None:
                                raise RuntimeError("波长计重连失败")
                            if not self._interruptible_wait(1.0):
                                break

                    if not self.running:
                        break

                    # 写（波长保留3位，功率mW保留6位）
                    ratio = pdr / pdt if abs(pdt) > 1e-12 else float("nan")
                    cur_row = [ts, trunc3(wav1_nm), trunc6(pow1_mw), trunc3(wav2_nm), trunc6(pow2_mw), trunc4(self.temperature), trunc4(pdr), trunc4(pdt), trunc3(ratio), f"{pdr_sa}{pdt_sa}"]
                    for i, value in enumerate(cur_row):
                        ws_out.cell(row=count+1, column=i+1+loop_time*gap_len, value=value)

                    # 清缓冲到硬盘
                    since_flush += 1
                    if since_flush >= FLUSH_EVERY_N:
                        wb_out.save(xlsx_path)
                        since_flush = 0

            except KeyboardInterrupt:
                self.log("\n用户中断, 已停止.")

            finally:
                try:
                    if inst is not None:
                        inst.close()
                finally:
                    rm.close()
                if wb_out is not None and xlsx_path is not None:
                    wb_out.save(xlsx_path)
                    wb_out.close()
                self.log("结束.")

        except Exception as e:
            self.log(str(e), "ERROR")
        finally:
            self.running = False
            if recv_thread is not None and recv_thread.is_alive():
                recv_thread.join(timeout=0.6)
            self._safe_shutdown_laser()


class UnlimitedAccuracyWorker(QThread):
    """Replay the verified 2001-point table and publish one settled point."""

    point_ready = pyqtSignal(object)
    scan_started = pyqtSignal(int)
    scan_completed = pyqtSignal(int, float, int)
    status_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, port, points, guards, repeat_scans=False, parent=None):
        super().__init__(parent)
        self.port = port
        self.points = tuple(points)
        self.guards = dict(guards)
        self.repeat_scans = bool(repeat_scans)
        self.running = True
        self.error_message = None
        self.cancelled = False
        self.completed_scans = 0
        self._rx_buffer = bytearray()
        self._last_codes = (0, 0, 0, 0, 0)

    def stop(self):
        self.running = False
        self.requestInterruption()

    def _interruptible_wait(self, duration_s):
        deadline = time.monotonic() + max(0.0, float(duration_s))
        while self.running:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return True
            time.sleep(min(0.02, remaining))
        self.cancelled = True
        return False

    @staticmethod
    def _ack_codes(frame):
        if len(frame) != SINGLE_VALUE_FRAME_SIZE:
            return None
        if frame[0:4] != b"\xFF\xFF\x02\x00":
            return None
        return tuple(
            (frame[4 + channel * 2] << 8) | frame[5 + channel * 2]
            for channel in range(5)
        )

    @staticmethod
    def _validate_ack_status(frame, requested_mode):
        if not (frame[15] & 0x80):
            raise RuntimeError("当前单片机固件缺少PI11210状态回读")
        flags = int(frame[15])
        raw_status = (int(frame[16]) << 8) | int(frame[17])
        i2c_errors = (int(frame[18]) << 8) | int(frame[19])
        if int(frame[14]) != int(requested_mode):
            raise RuntimeError("SOA正向/关光状态与请求不一致")
        if flags & 0x03 != 0x03:
            raise RuntimeError(
                f"PI11210离线或器件ID不符（状态0x{raw_status:04X}）"
            )
        if not flags & 0x04:
            raise RuntimeError(
                f"PI11210写入失败（累计I²C错误{i2c_errors}次）"
            )
        if flags & 0x30:
            raise RuntimeError(f"PI11210过温保护（状态0x{raw_status:04X}）")

    def _read_available_frames(self):
        waiting = int(getattr(self.port, "in_waiting", 0) or 0)
        data = self.port.read(min(max(waiting, 64), 4096))
        if not data:
            return ()
        self._rx_buffer.extend(data)
        return tuple(extract_single_value_frames(self._rx_buffer))

    def _command_and_ack(self, codes, soa_mode=PI11210_SOA_SOURCE_MODE):
        command = build_single_value_dac_command(
            codes, soa_mode=soa_mode, fullband_2001=True
        )
        expected = tuple(
            (command[4 + channel * 2] << 8) | command[5 + channel * 2]
            for channel in range(5)
        )
        last_returned = None

        for attempt in range(2):
            if not self.running:
                self.cancelled = True
                return None
            self.port.write(command)
            deadline = time.monotonic() + 2.0
            while self.running and time.monotonic() < deadline:
                for frame in self._read_available_frames():
                    returned = self._ack_codes(frame)
                    if returned is None:
                        continue
                    last_returned = returned
                    if returned != expected:
                        continue
                    self._validate_ack_status(frame, soa_mode)
                    self._last_codes = expected
                    return time.monotonic()
            if self.running and attempt == 0:
                self.status_signal.emit("DAC回读超时，正在重发当前点")
        if not self.running:
            self.cancelled = True
            return None
        labels = ("GAIN", "SOA", "PHASE", "WAVE_A", "WAVE_B")
        requested_text = ", ".join(
            f"{name}={value}" for name, value in zip(labels, expected)
        )
        if last_returned is None:
            detail = "未收到任何DAC回读帧"
        else:
            returned_text = ", ".join(
                f"{name}={value}" for name, value in zip(labels, last_returned)
            )
            detail = f"最近回读：{returned_text}"
        raise RuntimeError(
            f"重试后仍未收到匹配的单片机DAC回读；请求：{requested_text}；{detail}"
        )

    def _apply_guard(self, guard):
        self.status_signal.emit(
            f"{guard.target_nm:.2f} nm：执行{len(guard.precondition_codes)}级过渡保护"
        )
        for stage, codes in enumerate(guard.precondition_codes, start=1):
            if self._command_and_ack(codes) is None:
                return False
            self.status_signal.emit(
                f"{guard.target_nm:.2f} nm：过渡保护 {stage}/{len(guard.precondition_codes)}"
            )
            if not self._interruptible_wait(guard.hold_s_per_stage):
                return False
        return True

    def _prepare_forward_scan(self):
        first = self.points[0]
        self.status_signal.emit(
            "正在建立1525 nm正向扫描初始状态（首点额外稳定500 ms）"
        )
        if self._command_and_ack(first.codes) is None:
            return False
        return self._interruptible_wait(FULLBAND_ACCURACY_FIRST_POINT_SETTLE_S)

    def _acquire_point(self, scan_id, point):
        started = time.monotonic()
        ack_time = self._command_and_ack(point.codes)
        if ack_time is None:
            return None
        if not self._interruptible_wait(FULLBAND_ACCURACY_MIN_SETTLE_S):
            return None

        # Discard monitor frames generated during the fixed laser-settling
        # interval.  Samples accepted below therefore all belong to the stable
        # observation window after the matching DAC acknowledgement.
        self.port.reset_input_buffer()
        self._rx_buffer.clear()
        samples = []
        median = sigma = drift = np.full(6, np.nan, dtype=float)
        settled = False
        deadline = time.monotonic() + FULLBAND_ACCURACY_POINT_TIMEOUT_S
        while self.running and time.monotonic() < deadline:
            for frame in self._read_available_frames():
                values = decode_single_value_monitor_frame(frame)
                if values is None or any(value < 0 or value > 4095 for value in values):
                    continue
                samples.append(values)
                median, sigma, drift, settled = fullband_accuracy_statistics(samples)
                if settled or len(samples) >= FULLBAND_ACCURACY_MAX_SAMPLES:
                    break
            if settled or len(samples) >= FULLBAND_ACCURACY_MAX_SAMPLES:
                break

        if not self.running:
            self.cancelled = True
            return None
        if not samples:
            raise RuntimeError(
                f"{point.target_nm:.2f} nm在稳定等待后未收到ADC监测数据"
            )
        median, sigma, drift, settled = fullband_accuracy_statistics(samples)
        saturated_channels = tuple(
            channel for channel, value in enumerate(median[2:])
            if value >= 4080.0
        )
        if saturated_channels:
            settled = False
        code_to_volt = PD_ADC_REFERENCE_V / PD_ADC_CODE_COUNT
        return {
            "scan_id": int(scan_id),
            "index": int(point.index),
            "total": len(self.points),
            "target_nm": float(point.target_nm),
            "wavelength_nm": float(point.measured_nm),
            "codes": tuple(point.codes),
            "adc_codes": tuple(float(value) for value in median[2:]),
            "voltages": tuple(float(value * code_to_volt) for value in median[2:]),
            "sigma_codes": tuple(float(value) for value in sigma[2:]),
            "sigma_volts": tuple(float(value * code_to_volt) for value in sigma[2:]),
            "monitor_codes": tuple(float(value) for value in median[:2]),
            "monitor_sigma_codes": tuple(float(value) for value in sigma[:2]),
            "maximum_drift_codes": float(np.nanmax(drift)),
            "stable": bool(settled),
            "saturated_channels": saturated_channels,
            "sample_count": len(samples),
            "elapsed_s": time.monotonic() - started,
        }

    def _safe_shutter(self):
        try:
            if self.port.is_open:
                self.port.write(
                    build_single_value_dac_command(
                        (0, 0, 0, 0, 0),
                        soa_mode=PI11210_SOA_SHUTTER_MODE,
                        fullband_2001=True,
                    )
                )
                self._last_codes = (0, 0, 0, 0, 0)
        except Exception:
            pass

    def run(self):
        try:
            if len(self.points) != FULLBAND_ACCURACY_POINT_COUNT:
                raise RuntimeError("无限准确模式没有加载完整2001点表")
            self.port.reset_input_buffer()
            self._rx_buffer.clear()
            send_work_mode_command(2)
            if not self._interruptible_wait(0.12):
                return
            self.port.reset_input_buffer()
            self._rx_buffer.clear()

            scan_id = 1
            while self.running:
                if not self._prepare_forward_scan():
                    break
                scan_started_at = time.monotonic()
                unstable_count = 0
                self.scan_started.emit(scan_id)
                for scan_index, point in enumerate(self.points):
                    if not self.running:
                        self.cancelled = True
                        break
                    guard = self.guards.get(point.index)
                    if guard is not None and not self._apply_guard(guard):
                        break
                    self.status_signal.emit(
                        f"正在稳定第{point.index + 1}/{len(self.points)}点 · {point.target_nm:.2f} nm"
                    )
                    result = self._acquire_point(scan_id, point)
                    if result is None:
                        break
                    result["scan_index"] = scan_index
                    if not result["stable"]:
                        unstable_count += 1
                    self.point_ready.emit(result)

                if not self.running:
                    break
                self.completed_scans += 1
                self.scan_completed.emit(
                    scan_id,
                    time.monotonic() - scan_started_at,
                    unstable_count,
                )
                if not self.repeat_scans:
                    break
                scan_id += 1
        except Exception as exc:
            self.error_message = str(exc)
            self.error_signal.emit(self.error_message)
        finally:
            self._safe_shutter()


class EqualIntervalWorker(UnlimitedAccuracyWorker):
    """Acquire exactly one fresh ADC frame after a user-defined laser delay."""

    feedback_confirmed = pyqtSignal(int, int)
    feedback_readback = pyqtSignal(int, int, bool)

    def __init__(
        self,
        port,
        points,
        guards,
        settle_s,
        feedback_selectors=(PD_FEEDBACK_DEFAULT_SELECTOR,) * 2,
        repeat_scans=False,
        parent=None,
    ):
        super().__init__(port, points, guards, repeat_scans, parent)
        self.settle_s = float(settle_s)
        self.feedback_selectors = tuple(int(value) for value in feedback_selectors)
        if len(self.feedback_selectors) != 2 or any(
            value < 0 or value >= len(PD_FEEDBACK_KOHM_BY_SELECTOR)
            for value in self.feedback_selectors
        ):
            raise ValueError("CH0/CH1模拟跨阻状态必须各提供一个值")
        if not math.isfinite(self.settle_s) or self.settle_s < 0.0:
            raise ValueError("激光器稳定时间必须是非负有限数")

    def _feedback_command_and_ack(self):
        expected = self.feedback_selectors
        command = build_extra_feedback_command(*expected)
        last_returned = None
        for attempt in range(2):
            if not self.running:
                self.cancelled = True
                return False
            self.port.write(command)
            deadline = time.monotonic() + 1.0
            while self.running and time.monotonic() < deadline:
                for frame in self._read_available_frames():
                    response = decode_extra_feedback_status(frame)
                    if response is None:
                        continue
                    returned, status = response
                    last_returned = returned
                    accepted = status == ACK_VALUE and returned == expected
                    self.feedback_readback.emit(*returned, accepted)
                    if status == ACK_ERROR_VALUE:
                        actual = (
                            f"CH0={PD_FEEDBACK_KOHM_BY_SELECTOR[returned[0]]} kΩ, "
                            f"CH1={PD_FEEDBACK_KOHM_BY_SELECTOR[returned[1]]} kΩ"
                        )
                        raise RuntimeError(
                            f"choseA/B IO核验失败（IO实际：{actual}）"
                        )
                    if returned == expected:
                        self.feedback_confirmed.emit(*returned)
                        return True
            if self.running and attempt == 0:
                self.status_signal.emit("跨阻状态回读超时，正在重发")
        if not self.running:
            self.cancelled = True
            return False
        actual = "无回读" if last_returned is None else (
            f"CH0={PD_FEEDBACK_KOHM_BY_SELECTOR[last_returned[0]]} kΩ, "
            f"CH1={PD_FEEDBACK_KOHM_BY_SELECTOR[last_returned[1]]} kΩ"
        )
        raise RuntimeError(f"未收到匹配的跨阻状态回读（{actual}）")

    def _acquire_point(self, scan_id, point):
        started = time.monotonic()
        if self._command_and_ack(point.codes) is None:
            return None
        if not self._interruptible_wait(self.settle_s):
            return None

        # The MCU streams monitor frames continuously.  Everything queued
        # during the requested laser delay belongs to the transient interval,
        # so clear it and accept exactly the first complete frame afterwards.
        self.port.reset_input_buffer()
        self._rx_buffer.clear()
        sample = None
        deadline = time.monotonic() + FULLBAND_ACCURACY_POINT_TIMEOUT_S
        while self.running and time.monotonic() < deadline:
            for frame in self._read_available_frames():
                values = decode_single_value_monitor_frame(frame)
                if values is None or any(value < 0 or value > 4095 for value in values):
                    continue
                sample = tuple(values)
                break
            if sample is not None:
                break
            time.sleep(0.001)

        if not self.running:
            self.cancelled = True
            return None
        if sample is None:
            raise RuntimeError(
                f"{point.target_nm:.2f} nm等待{self.settle_s:.3f}秒后未收到ADC数据"
            )

        adc_codes = tuple(float(value) for value in sample[2:])
        saturated_channels = tuple(
            channel for channel, value in enumerate(adc_codes) if value >= 4080.0
        )
        code_to_volt = PD_ADC_REFERENCE_V / PD_ADC_CODE_COUNT
        return {
            "scan_id": int(scan_id),
            "index": int(point.index),
            "total": len(self.points),
            "target_nm": float(point.target_nm),
            "wavelength_nm": float(point.measured_nm),
            "codes": tuple(point.codes),
            "adc_codes": adc_codes,
            "voltages": tuple(value * code_to_volt for value in adc_codes),
            "sigma_codes": (float("nan"),) * 4,
            "sigma_volts": (float("nan"),) * 4,
            "monitor_codes": tuple(float(value) for value in sample[:2]),
            "monitor_sigma_codes": (float("nan"),) * 2,
            "maximum_drift_codes": float("nan"),
            "stable": True,
            "single_sample": True,
            "saturated_channels": saturated_channels,
            "sample_count": 1,
            "settle_s": self.settle_s,
            "elapsed_s": time.monotonic() - started,
        }

    def run(self):
        try:
            if not self.points:
                raise RuntimeError("等间隔模式没有可扫描的标定点")
            self.port.reset_input_buffer()
            self._rx_buffer.clear()
            self.port.write(build_work_mode_command(2))
            if not self._interruptible_wait(0.12):
                return
            self.port.reset_input_buffer()
            self._rx_buffer.clear()
            self.status_signal.emit("正在设置并确认CH0/CH1模拟跨阻…")
            if not self._feedback_command_and_ack():
                return
            if not self._interruptible_wait(0.005):
                return
            self.port.reset_input_buffer()
            self._rx_buffer.clear()

            scan_id = 1
            while self.running:
                scan_started_at = time.monotonic()
                self.scan_started.emit(scan_id)
                for scan_index, point in enumerate(self.points):
                    if not self.running:
                        self.cancelled = True
                        break
                    if (
                        scan_index > 0
                        and scan_index % EQUAL_INTERVAL_FEEDBACK_VERIFY_EVERY_POINTS == 0
                    ):
                        self.status_signal.emit(
                            f"第{scan_index + 1}点前正在复核choseA/B IO实际档位…"
                        )
                        if not self._feedback_command_and_ack():
                            break
                    guard = self.guards.get(point.index)
                    if guard is not None and not self._apply_guard(guard):
                        break
                    self.status_signal.emit(
                        f"第{scan_index + 1}/{len(self.points)}点 · "
                        f"{point.target_nm:.2f} nm · 等待{self.settle_s:.3f}秒"
                    )
                    result = self._acquire_point(scan_id, point)
                    if result is None:
                        break
                    result["scan_index"] = scan_index
                    self.point_ready.emit(result)

                if not self.running:
                    break
                self.completed_scans += 1
                self.scan_completed.emit(
                    scan_id,
                    time.monotonic() - scan_started_at,
                    0,
                )
                if not self.repeat_scans:
                    break
                scan_id += 1
        except Exception as exc:
            self.error_message = str(exc)
            self.error_signal.emit(self.error_message)
        finally:
            self._safe_shutter()


class EqualIntervalFeedbackApplyWorker(EqualIntervalWorker):
    """Apply an idle equal-interval feedback selection and verify its IO pins."""

    def __init__(self, port, feedback_selectors, parent=None):
        super().__init__(
            port,
            (),
            {},
            settle_s=0.0,
            feedback_selectors=feedback_selectors,
            repeat_scans=False,
            parent=parent,
        )

    def run(self):
        try:
            self.port.reset_input_buffer()
            self._rx_buffer.clear()
            self.port.write(build_work_mode_command(2))
            if not self._interruptible_wait(0.12):
                return
            self.port.reset_input_buffer()
            self._rx_buffer.clear()
            self.status_signal.emit("正在下发跨阻并核验choseA/B IO实际电平…")
            self._feedback_command_and_ack()
        except Exception as exc:
            self.error_message = str(exc)
            self.error_signal.emit(self.error_message)


class ModeTableBuildWorker(QtCore.QThread):
    """Promote or restore mode tables and build the matching OTA package."""

    progress = QtCore.pyqtSignal(str)
    completed = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, action, report=None, candidate_path=None, parent=None):
        super().__init__(parent)
        self.action = str(action)
        self.report = report
        self.candidate_path = candidate_path

    def run(self):
        try:
            if self.action == "apply":
                self.progress.emit("正在校验2001点来源并备份上一版正式点表…")
                result = apply_candidate_and_build(
                    self.report,
                    candidate_path=self.candidate_path,
                )
            elif self.action == "restore":
                self.progress.emit("正在恢复上一版正式点表并重新生成固件…")
                result = restore_previous_and_build()
            else:
                raise ValueError("未知的点表操作")
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.completed.emit(result)


class UnlimitedAccuracyWindow(QtWidgets.QWidget):
    """Incremental 2001-point spectrum page for accuracy-first local scans."""

    CHANNEL_COLORS = ("#F2C94C", "#FF6B72", "#38C793", "#69A7FF")

    def __init__(self):
        super().__init__()
        self.setObjectName("graphPage")
        self.mode_display_name = "无限准确模式"
        self.export_prefix = "无限准确光谱"
        self.worker = None
        self.last_error = None
        self.current_scan_id = 0
        self.scan_started_at = None
        self.completed_points = 0
        self.stable_points = 0
        self.unstable_points = 0
        self.load_error = None
        try:
            self.points = load_fullband_accuracy_table()
            self.guards = load_fullband_transition_guards(self.points)
        except Exception as exc:
            self.points = ()
            self.guards = {}
            self.load_error = str(exc)

        page_layout = QtWidgets.QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        self.page_scroll = ResponsiveScrollArea()
        self.page_scroll.setObjectName("unlimitedPageScroll")
        self.page_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.page_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        content = ScrollContentWidget()
        content.setObjectName("unlimitedPageContent")
        root = QtWidgets.QVBoxLayout(content)
        root.setContentsMargins(10, 10, 10, 8)
        root.setSpacing(10)
        self.page_scroll.setWidget(content)
        page_layout.addWidget(self.page_scroll)

        controls = QtWidgets.QFrame()
        controls.setObjectName("controlGroup")
        controls_layout = FlowLayout(
            controls,
            margin=10,
            horizontal_spacing=9,
            vertical_spacing=8,
        )
        self.controls_layout = controls_layout

        self.mode_label = QtWidgets.QLabel("无限准确模式 · 2001点")
        self.mode_label.setObjectName("metricBadge")
        self.mode_label.setMinimumWidth(170)
        self.dac_label = QtWidgets.QLabel("DAC: PI11210 · 五通道全量写入")
        self.dac_label.setObjectName("metricBadge")
        self.dac_label.setMinimumWidth(220)
        self.axis_label = QtWidgets.QLabel("横轴：波长计标定实测值")
        self.axis_label.setObjectName("metricBadge")
        self.axis_label.setMinimumWidth(190)
        self.gain_label = QtWidgets.QLabel(
            "模拟跨阻：CH0/CH1 40 kΩ（相对2 kΩ为×20），"
            "CH2/CH3/PDT/PDR 2 kΩ（×1） · 数字显示：CH0~3 ×1.00"
        )
        self.gain_label.setObjectName("metricBadge")
        self.gain_label.setMinimumWidth(420)
        self.gain_label.setWordWrap(True)
        controls_layout.addWidget(self.mode_label)
        controls_layout.addWidget(self.dac_label)
        controls_layout.addWidget(self.axis_label)
        controls_layout.addWidget(self.gain_label)
        self.repeat_checkbox = QtWidgets.QCheckBox("完成后自动重扫")
        self.repeat_checkbox.setToolTip(
            "默认只完成一张高稳定光谱；勾选后从1525 nm重新建立正向状态"
        )
        controls_layout.addWidget(self.repeat_checkbox)
        self.scan_btn = QtWidgets.QPushButton("开始逐点扫描")
        self.scan_btn.setProperty("role", "primary")
        self.scan_btn.setMinimumWidth(132)
        self.clear_btn = QtWidgets.QPushButton("清空曲线")
        self.save_btn = QtWidgets.QPushButton("保存当前光谱")
        self.save_btn.setEnabled(False)
        controls_layout.addWidget(self.scan_btn)
        controls_layout.addWidget(self.clear_btn)
        controls_layout.addWidget(self.save_btn)
        root.addWidget(controls)

        progress_card = QtWidgets.QFrame()
        progress_card.setObjectName("statusCard")
        progress_layout = QtWidgets.QVBoxLayout(progress_card)
        progress_layout.setContentsMargins(14, 10, 14, 10)
        progress_layout.setSpacing(8)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, FULLBAND_ACCURACY_POINT_COUNT)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("0 / 2001")
        progress_layout.addWidget(self.progress_bar)

        self.current_label = QtWidgets.QLabel("当前波长：等待开始")
        self.current_label.setObjectName("metricBadge")
        self.quality_label = QtWidgets.QLabel("稳定点：0 · 需复核：0")
        self.quality_label.setObjectName("metricBadge")
        self.noise_label = QtWidgets.QLabel("CH0~3 σ：—")
        self.noise_label.setObjectName("metricBadge")
        self.eta_label = QtWidgets.QLabel("预计剩余：—")
        self.eta_label.setObjectName("metricBadge")
        metrics_widget = QtWidgets.QWidget()
        metrics_layout = FlowLayout(
            metrics_widget, horizontal_spacing=9, vertical_spacing=7
        )
        self.metrics_layout = metrics_layout
        metrics_layout.addWidget(self.current_label)
        metrics_layout.addWidget(self.quality_label)
        metrics_layout.addWidget(self.noise_label)
        metrics_layout.addWidget(self.eta_label)
        progress_layout.addWidget(metrics_widget)
        root.addWidget(progress_card)

        self.plot = pg.PlotWidget()
        self.plot.setObjectName("measurementPlot")
        self.plot.setLabel("bottom", "波长 (nm)")
        self.plot.setLabel("left", "ADC 电压 (V)")
        self.plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.plot.getAxis("left").enableAutoSIPrefix(False)
        self.plot.showGrid(x=True, y=True, alpha=0.18)
        self.plot.setXRange(1525.0, 1565.0, padding=0.01)
        self.plot.setYRange(0.0, PD_ADC_REFERENCE_V, padding=0.04)
        self.plot.addLegend(offset=(12, 10))
        self.curves = []
        for channel, color in enumerate(self.CHANNEL_COLORS):
            self.curves.append(
                self.plot.plot(
                    [], [],
                    pen=pg.mkPen(color, width=1.7),
                    name=f"CH{channel}",
                    connect="finite",
                )
            )
        root.addWidget(self.plot, 1)

        self.status_label = QtWidgets.QLabel(
            "严格正向扫描 · 每点至少稳定200 ms · 多帧中位数/MAD/漂移联合判定"
        )
        self.status_label.setObjectName("softHint")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        self.scan_btn.clicked.connect(self.on_scan_clicked)
        self.clear_btn.clicked.connect(self.clear_spectrum)
        self.save_btn.clicked.connect(self.save_spectrum)

        self._configure_scan_points(self.points)
        if self.load_error:
            self.scan_btn.setEnabled(False)
            self.status_label.setText(f"2001点标定表加载失败：{self.load_error}")

    def set_dac_label(self, _dac_type):
        self.dac_label.setText("DAC: PI11210 · 标定原始码全量写入")

    def reload_calibration_table(self):
        """Reload a newly certified runtime table without restarting the app."""
        if self.worker is not None and self.worker.isRunning():
            raise RuntimeError("2001点扫描运行中，不能替换标定表")
        try:
            points = load_fullband_accuracy_table()
            guards = load_fullband_transition_guards(points)
        except Exception as exc:
            self.load_error = str(exc)
            self.scan_btn.setEnabled(False)
            self.status_label.setText(f"2001点标定表加载失败：{self.load_error}")
            raise
        self.load_error = None
        self.guards = guards
        self._configure_scan_points(points)
        self.scan_btn.setEnabled(True)
        self.status_label.setText("新的145 mA、2001点认证标定表已载入")
        return True

    @staticmethod
    def _format_duration(seconds):
        if not np.isfinite(seconds) or seconds < 0:
            return "—"
        total = int(round(seconds))
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours}小时{minutes:02d}分"
        if minutes:
            return f"{minutes}分{secs:02d}秒"
        return f"{secs}秒"

    def _configure_scan_points(self, points):
        self.points = tuple(points)
        point_count = len(self.points)
        self.wavelengths = np.asarray(
            [point.measured_nm for point in self.points], dtype=float
        )
        self.values = np.full((4, point_count), np.nan, dtype=float)
        self.sigmas = np.full((4, point_count), np.nan, dtype=float)
        self.stable_flags = np.zeros(point_count, dtype=bool)
        self.sample_counts = np.zeros(point_count, dtype=np.uint16)
        self.point_elapsed = np.full(point_count, np.nan, dtype=float)
        self.progress_bar.setRange(0, max(point_count, 1))
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(f"0 / {point_count}")
        if point_count:
            left = float(self.wavelengths[0])
            right = float(self.wavelengths[-1])
            if right <= left:
                right = left + 0.02
            self.plot.setXRange(left, right, padding=0.015)

    def _scan_table_error(self):
        if self.load_error:
            return self.load_error
        if len(self.points) != FULLBAND_ACCURACY_POINT_COUNT:
            return "没有完整2001点标定表"
        return None

    def _reset_arrays(self, scan_id=0):
        self.current_scan_id = int(scan_id)
        self.completed_points = 0
        self.stable_points = 0
        self.unstable_points = 0
        self.values.fill(np.nan)
        self.sigmas.fill(np.nan)
        self.stable_flags.fill(False)
        self.sample_counts.fill(0)
        self.point_elapsed.fill(np.nan)
        for curve in self.curves:
            curve.setData([], [])
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(f"0 / {len(self.points)}")
        self.current_label.setText("当前波长：等待数据")
        self.quality_label.setText("稳定点：0 · 需复核：0")
        self.noise_label.setText("CH0~3 σ：—")
        self.eta_label.setText("预计剩余：—")
        self.save_btn.setEnabled(False)

    def clear_spectrum(self):
        self._reset_arrays(self.current_scan_id)
        if self.worker is not None and self.worker.isRunning():
            self.status_label.setText("曲线已清空；当前扫描继续，后续点仍会实时追加")
        else:
            self.status_label.setText(
                "严格正向扫描 · 每点至少稳定200 ms · 多帧稳健统计"
            )

    def on_scan_clicked(self):
        if self.worker is not None and self.worker.isRunning():
            self.scan_btn.setEnabled(False)
            self.scan_btn.setText("正在安全停止…")
            self.status_label.setText("正在停止，并关闭SOA光输出…")
            self.worker.stop()
            return
        self.start_scan()

    def _create_worker(self):
        return UnlimitedAccuracyWorker(
            ser,
            self.points,
            self.guards,
            repeat_scans=self.repeat_checkbox.isChecked(),
            parent=self,
        )

    def _set_scan_parameters_enabled(self, enabled):
        del enabled

    def start_scan(self):
        global ser_open
        global switch_mode_enable
        scan_table_error = self._scan_table_error()
        if scan_table_error:
            QMessageBox.critical(
                self, "标定表不可用", scan_table_error
            )
            return
        if not switch_mode_enable:
            QMessageBox.warning(self, "当前流程仍在运行", "请先停止当前采集流程")
            return
        try:
            ensure_serial_open()
            ser.reset_input_buffer()
        except Exception as exc:
            QMessageBox.critical(
                self, "串口错误", f"{self.mode_display_name}启动失败：\n{exc}"
            )
            return

        with ser_cond:
            ser_open = True
            ser_cond.notify_all()
        switch_mode_enable = False
        self.last_error = None
        self._reset_arrays(1)
        self.scan_started_at = time.monotonic()
        self.repeat_checkbox.setEnabled(False)
        self._set_scan_parameters_enabled(False)
        self.clear_btn.setEnabled(False)
        self.scan_btn.setEnabled(True)
        self.scan_btn.setText("停止扫描")
        self.worker = self._create_worker()
        self.worker.scan_started.connect(self._on_scan_started)
        self.worker.point_ready.connect(self._on_point_ready)
        self.worker.scan_completed.connect(self._on_scan_completed)
        self.worker.status_signal.connect(self.status_label.setText)
        self.worker.error_signal.connect(self._on_worker_error)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    def _on_scan_started(self, scan_id):
        self._reset_arrays(scan_id)
        self.scan_started_at = time.monotonic()
        self.status_label.setText(f"第{scan_id}轮正向扫描已开始")

    def _on_point_ready(self, result):
        scan_id = int(result["scan_id"])
        if scan_id != self.current_scan_id:
            self._reset_arrays(scan_id)
            self.scan_started_at = time.monotonic()
        index = int(result.get("scan_index", result["index"]))
        if index < 0 or index >= len(self.points):
            return
        self.values[:, index] = np.asarray(result["voltages"], dtype=float)
        self.sigmas[:, index] = np.asarray(result["sigma_volts"], dtype=float)
        self.stable_flags[index] = bool(result["stable"])
        self.sample_counts[index] = int(result["sample_count"])
        self.point_elapsed[index] = float(result["elapsed_s"])
        self.completed_points = max(self.completed_points, index + 1)
        self.stable_points = int(np.count_nonzero(self.stable_flags[:self.completed_points]))
        self.unstable_points = self.completed_points - self.stable_points

        x = self.wavelengths[:index + 1]
        for channel, curve in enumerate(self.curves):
            curve.setData(x, self.values[channel, :index + 1], connect="finite")
        self.progress_bar.setValue(index + 1)
        self.progress_bar.setFormat(f"{index + 1} / {len(self.points)}")
        self.current_label.setText(
            f"目标 {result['target_nm']:.2f} nm · 标定 {result['wavelength_nm']:.5f} nm"
        )
        single_sample = bool(result.get("single_sample", False))
        if single_sample:
            self.quality_label.setText(f"已采集：{self.completed_points} · 每点1帧ADC")
            self.noise_label.setText("原始单帧：未做均值、滤波或数字缩放")
        else:
            self.quality_label.setText(
                f"稳定点：{self.stable_points} · 需复核：{self.unstable_points}"
            )
            sigma_mv = np.asarray(result["sigma_volts"], dtype=float) * 1000.0
            self.noise_label.setText(
                "CH0~3 σ：" + "/".join(f"{value:.2f}" for value in sigma_mv) + " mV"
            )
        finite_elapsed = self.point_elapsed[:index + 1]
        finite_elapsed = finite_elapsed[np.isfinite(finite_elapsed)]
        if len(finite_elapsed):
            remaining = (len(self.points) - index - 1) * float(np.mean(finite_elapsed))
            self.eta_label.setText(f"预计剩余：{self._format_duration(remaining)}")
        saturated = tuple(result.get("saturated_channels", ()))
        if saturated:
            state = "ADC饱和：" + "/".join(f"CH{channel}" for channel in saturated)
        else:
            state = (
                "单帧ADC已采集"
                if single_sample
                else "稳定" if result["stable"] else "达到上限，已标记需复核"
            )
        self.status_label.setText(
            f"第{index + 1}点已绘制 · {state} · {result['sample_count']}个ADC样本"
        )
        self.save_btn.setEnabled(True)

    def _on_scan_completed(self, scan_id, elapsed_s, unstable_count):
        self.status_label.setText(
            f"第{scan_id}轮2001点完成 · 用时{self._format_duration(elapsed_s)}"
            f" · 需复核{unstable_count}点"
        )

    def _on_worker_error(self, message):
        self.last_error = str(message)
        self.status_label.setText(f"采集异常：{self.last_error}")

    def _release_acquisition_lock(self):
        global ser_open
        global switch_mode_enable
        with ser_cond:
            ser_open = False
            ser_cond.notify_all()
        switch_mode_enable = True
        try:
            release_serial_if_allowed()
        except Exception:
            pass

    def _on_worker_finished(self):
        worker = self.worker
        self._release_acquisition_lock()
        self.repeat_checkbox.setEnabled(True)
        self._set_scan_parameters_enabled(True)
        self.clear_btn.setEnabled(True)
        self.scan_btn.setEnabled(True)
        self.scan_btn.setText("重新开始扫描" if self.completed_points else "开始逐点扫描")
        if worker is not None and worker.cancelled:
            self.status_label.setText(
                f"扫描已停止，保留前{self.completed_points}个点；SOA已关闭"
            )
        elif self.last_error:
            QMessageBox.warning(
                self, f"{self.mode_display_name}已停止", self.last_error
            )
        elif self.completed_points == len(self.points):
            self.status_label.setText(
                f"2001点高稳定光谱已完成 · 稳定{self.stable_points}点"
                f" · 需复核{self.unstable_points}点 · SOA已关闭"
            )
        if worker is not None:
            worker.deleteLater()
        self.worker = None

    def save_spectrum(self):
        if self.completed_points <= 0:
            return
        default = Path(get_desktop_path()) / (
            self.export_prefix + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "保存当前光谱", str(default), "CSV 文件 (*.csv)"
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow([
                "index", "target_wavelength_nm", "calibrated_wavelength_nm",
                "ch0_v", "ch1_v", "ch2_v", "ch3_v",
                "ch0_sigma_v", "ch1_sigma_v", "ch2_sigma_v", "ch3_sigma_v",
                "stable", "sample_count", "point_elapsed_s",
                "gain_code", "soa_code", "phase_code", "wavelength_a_code",
                "wavelength_b_code",
            ])
            for index in range(self.completed_points):
                point = self.points[index]
                writer.writerow([
                    point.index,
                    f"{point.target_nm:.6f}",
                    f"{point.measured_nm:.8f}",
                    *[f"{value:.9f}" for value in self.values[:, index]],
                    *[f"{value:.9f}" for value in self.sigmas[:, index]],
                    int(self.stable_flags[index]),
                    int(self.sample_counts[index]),
                    f"{self.point_elapsed[index]:.6f}",
                    *point.codes,
                ])
        self.status_label.setText(f"当前光谱已保存：{path}")

    def shutdown(self):
        worker = self.worker
        if worker is not None and worker.isRunning():
            worker.stop()
            worker.wait(2500)
        self._release_acquisition_lock()


class EqualIntervalWindow(UnlimitedAccuracyWindow):
    """Raw one-frame spectrum page with an explicit DAC-to-ADC delay."""

    def __init__(self):
        super().__init__()
        self.feedback_apply_worker = None
        self.feedback_apply_error = None
        self.feedback_display_state = "pending"
        self.confirmed_feedback_selectors = None
        self.observed_feedback_selectors = None
        self.full_points = tuple(self.points)
        self.full_guards = dict(self.guards)
        self.active_requested_range = (
            (
                float(self.full_points[0].target_nm),
                float(self.full_points[-1].target_nm),
            )
            if self.full_points
            else (float("nan"), float("nan"))
        )
        self.mode_display_name = "等间隔模式"
        self.export_prefix = "等间隔原始光谱"
        self.mode_label.setText("等间隔模式 · 2001点")
        self.dac_label.setText("DAC: PI11210 · 每点回读确认")
        self.axis_label.setText("横轴：波长计标定实测值")

        self.settle_label = QtWidgets.QLabel("激光器稳定时间")
        self.settle_label.setObjectName("metricBadge")
        self.settle_spin = QtWidgets.QDoubleSpinBox()
        self.settle_spin.setObjectName("settleTimeSpin")
        self.settle_spin.setDecimals(3)
        self.settle_spin.setRange(0.0, 10.0)
        self.settle_spin.setSingleStep(0.010)
        self.settle_spin.setValue(0.200)
        self.settle_spin.setSuffix(" s")
        self.settle_spin.setMinimumWidth(108)
        self.settle_spin.setToolTip(
            "从匹配的DAC回读开始计时；到时清空旧帧，只采集随后第一帧ADC"
        )
        self.controls_layout.addWidget(self.settle_label)
        self.controls_layout.addWidget(self.settle_spin)

        self.range_label = QtWidgets.QLabel("扫描范围")
        self.range_label.setObjectName("metricBadge")
        self.start_wavelength_spin = QtWidgets.QDoubleSpinBox()
        self.start_wavelength_spin.setObjectName("equalIntervalStartWavelength")
        self.end_wavelength_spin = QtWidgets.QDoubleSpinBox()
        self.end_wavelength_spin.setObjectName("equalIntervalEndWavelength")
        minimum_nm = float(self.full_points[0].target_nm) if self.full_points else 1525.0
        maximum_nm = float(self.full_points[-1].target_nm) if self.full_points else 1565.0
        for spin, prefix, value in (
            (self.start_wavelength_spin, "开始 ", minimum_nm),
            (self.end_wavelength_spin, "结束 ", maximum_nm),
        ):
            spin.setDecimals(3)
            spin.setRange(minimum_nm, maximum_nm)
            spin.setSingleStep(0.020)
            spin.setValue(value)
            spin.setPrefix(prefix)
            spin.setSuffix(" nm")
            spin.setMinimumWidth(148)
            spin.setToolTip("按标定表的目标波长闭区间选点，每点间隔0.02 nm")
        self.controls_layout.addWidget(self.range_label)
        self.controls_layout.addWidget(self.start_wavelength_spin)
        self.controls_layout.addWidget(self.end_wavelength_spin)
        self.start_wavelength_spin.valueChanged.connect(self._update_range_preview)
        self.end_wavelength_spin.valueChanged.connect(self._update_range_preview)
        self._update_range_preview()

        self.ch0_feedback_btn = QtWidgets.QPushButton("CH0：40 kΩ")
        self.ch1_feedback_btn = QtWidgets.QPushButton("CH1：40 kΩ")
        for button, channel in (
            (self.ch0_feedback_btn, "CH0"),
            (self.ch1_feedback_btn, "CH1"),
        ):
            button.setProperty("feedbackSelector", PD_FEEDBACK_DEFAULT_SELECTOR)
            button.setMinimumWidth(112)
            button.setToolTip(
                f"点击循环切换{channel}的模拟跨阻：40/20/5/2 kΩ。"
                "空闲时立即下发并核验choseA/B IO电平；扫描期间锁定并周期复核。"
            )
            self.controls_layout.addWidget(button)
        self.feedback_status_label = QtWidgets.QLabel("跨阻：待单片机确认")
        self.feedback_status_label.setObjectName("metricBadge")
        self.controls_layout.addWidget(self.feedback_status_label)
        self.ch0_feedback_btn.clicked.connect(
            lambda _checked=False: self._cycle_feedback_selector(self.ch0_feedback_btn)
        )
        self.ch1_feedback_btn.clicked.connect(
            lambda _checked=False: self._cycle_feedback_selector(self.ch1_feedback_btn)
        )
        self._update_feedback_display("pending")
        self.repeat_checkbox.setToolTip("完成2001点后按相同等待时间重新扫描")
        self.status_label.setText(
            "每点：DAC回读确认 → 等待设定时间 → 丢弃旧数据 → 取1帧ADC → 实时绘制"
        )
        self.quality_label.setText("已采集：0 · 每点1帧ADC")
        self.noise_label.setText("原始单帧：未做均值、滤波或数字缩放")
        self.latest_adc_label = QtWidgets.QLabel("当前ADC码 CH0~3：—")
        self.latest_adc_label.setObjectName("metricBadge")
        self.metrics_layout.addWidget(self.latest_adc_label)
        self.precise_fit_btn = QtWidgets.QPushButton("精确拟合反射谱")
        self.precise_fit_btn.setObjectName("preciseReflectionFitButton")
        self.precise_fit_btn.setProperty("role", "primary")
        self.precise_fit_btn.setToolTip(
            "完成当前波长范围后，自动寻找各通道的布拉格峰并亚采样拟合中心波长"
        )
        self.precise_fit_btn.setEnabled(False)
        self.controls_layout.addWidget(self.precise_fit_btn)
        self.precise_fit_btn.clicked.connect(self.fit_reflection_spectrum)
        self.auto_select_btn = QtWidgets.QPushButton("查看自动选点")
        self.auto_select_btn.setObjectName("autoModeSelectionButton")
        self.auto_select_btn.setToolTip(
            f"仅分析CH1并严格确认{FINGER_EXPECTED_PEAK_COUNT}个有效光栅峰，"
            f"再生成应力{FINGER_STRESS_POINT_COUNT}点与"
            f"温度{FINGER_TEMPERATURE_POINT_COUNT}点候选表"
        )
        self.auto_select_btn.setEnabled(False)
        self.controls_layout.addWidget(self.auto_select_btn)
        self.auto_select_btn.clicked.connect(self._show_auto_mode_selection)
        self.auto_selection_label = QtWidgets.QLabel("自动选点：等待完整2001点")
        self.auto_selection_label.setObjectName("metricBadge")
        self.metrics_layout.addWidget(self.auto_selection_label)
        self.precise_fit_results = [() for _ in range(4)]
        self.precise_fit_items = []
        self.precise_fit_dialog = None
        self.auto_selection_report = None
        self.auto_selection_path = None
        self.mode_table_worker = None
        self.mode_table_progress = None
        # The unified shell installs this callback so a freshly built package
        # can be loaded directly into its existing OTA page.
        self.ota_package_ready = None

    def _cycle_feedback_selector(self, button, apply_hardware=True):
        cycle = (1, 3, 2, 0)  # 40 -> 20 -> 5 -> 2 kOhm
        current = int(button.property("feedbackSelector"))
        button.setProperty("feedbackSelector", cycle[(cycle.index(current) + 1) % len(cycle)])
        self.confirmed_feedback_selectors = None
        self.observed_feedback_selectors = None
        self._update_feedback_display("pending")
        if apply_hardware:
            self._start_feedback_apply()

    def _selected_feedback_selectors(self):
        return (
            int(self.ch0_feedback_btn.property("feedbackSelector")),
            int(self.ch1_feedback_btn.property("feedbackSelector")),
        )

    @staticmethod
    def _feedback_resistances(selectors):
        return tuple(PD_FEEDBACK_KOHM_BY_SELECTOR[value] for value in selectors)

    def _update_feedback_display(self, state=None):
        if state is not None:
            self.feedback_display_state = str(state)
        requested = self._selected_feedback_selectors()
        requested_resistances = self._feedback_resistances(requested)
        self.ch0_feedback_btn.setText(f"CH0：{requested_resistances[0]} kΩ")
        self.ch1_feedback_btn.setText(f"CH1：{requested_resistances[1]} kΩ")

        observed = self.observed_feedback_selectors
        if self.feedback_display_state == "confirmed" and observed == requested:
            resistances = self._feedback_resistances(observed)
            status = (
                f"跨阻：IO已核验 CH0 {resistances[0]} kΩ / CH1 {resistances[1]} kΩ"
            )
            source = "IO实际"
        elif observed is not None:
            resistances = self._feedback_resistances(observed)
            status = (
                f"跨阻不一致：设定 {requested_resistances[0]}/{requested_resistances[1]} kΩ；"
                f"IO实际 {resistances[0]}/{resistances[1]} kΩ"
            )
            source = "IO实际（不一致）"
        else:
            resistances = requested_resistances
            if self.feedback_display_state == "applying":
                status = "跨阻：正在下发并核验choseA/B IO…"
            elif self.feedback_display_state == "error":
                status = "跨阻：下发或IO核验失败"
            else:
                status = "跨阻：设定已改变，待IO核验"
            source = "设定，待IO核验"
        self.feedback_status_label.setText(status)

        gains = tuple(value / 2.0 for value in resistances)
        self.gain_label.setText(
            f"模拟跨阻（{source}）：CH0 {resistances[0]} kΩ（×{gains[0]:g}），"
            f"CH1 {resistances[1]} kΩ（×{gains[1]:g}），"
            "CH2/CH3/PDT/PDR 2 kΩ（×1） · 数字显示：CH0~3 ×1.00"
        )

    def _on_feedback_readback(self, ch0_selector, ch1_selector, accepted):
        actual = (int(ch0_selector), int(ch1_selector))
        self.observed_feedback_selectors = actual
        if bool(accepted) and actual == self._selected_feedback_selectors():
            self.confirmed_feedback_selectors = actual
            self._update_feedback_display("confirmed")
        else:
            self.confirmed_feedback_selectors = None
            self._update_feedback_display("mismatch")

    def _on_feedback_confirmed(self, ch0_selector, ch1_selector):
        actual = (int(ch0_selector), int(ch1_selector))
        self.confirmed_feedback_selectors = actual
        self.observed_feedback_selectors = actual
        self._update_feedback_display("confirmed")
        resistances = self._feedback_resistances(actual)
        self.status_label.setText(
            f"模拟跨阻IO已核验：CH0 {resistances[0]} kΩ，CH1 {resistances[1]} kΩ"
        )

    def _start_feedback_apply(self):
        global ser_open
        global switch_mode_enable
        if self.worker is not None and self.worker.isRunning():
            return
        if self.feedback_apply_worker is not None and self.feedback_apply_worker.isRunning():
            return
        with ser_cond:
            if ser_open or not switch_mode_enable:
                self._update_feedback_display("error")
                self.status_label.setText("跨阻未下发：当前有其他采集流程占用设备")
                return
            ser_open = True
            ser_cond.notify_all()
        switch_mode_enable = False
        try:
            ensure_serial_open()
            ser.reset_input_buffer()
        except Exception as exc:
            self._release_acquisition_lock()
            self._update_feedback_display("error")
            self.status_label.setText(f"跨阻下发失败：{exc}")
            return

        self.feedback_apply_error = None
        self._update_feedback_display("applying")
        self._set_scan_parameters_enabled(False)
        self.scan_btn.setEnabled(False)
        worker = EqualIntervalFeedbackApplyWorker(
            ser,
            self._selected_feedback_selectors(),
            parent=self,
        )
        self.feedback_apply_worker = worker
        worker.feedback_readback.connect(self._on_feedback_readback)
        worker.feedback_confirmed.connect(self._on_feedback_confirmed)
        worker.status_signal.connect(self.status_label.setText)
        worker.error_signal.connect(self._on_feedback_apply_error)
        worker.finished.connect(self._on_feedback_apply_finished)
        worker.start()

    def _on_feedback_apply_error(self, message):
        self.feedback_apply_error = str(message)
        if self.observed_feedback_selectors is None:
            self._update_feedback_display("error")
        self.status_label.setText(f"跨阻下发失败：{self.feedback_apply_error}")

    def _on_feedback_apply_finished(self):
        worker = self.feedback_apply_worker
        self._release_acquisition_lock()
        self._set_scan_parameters_enabled(True)
        self.scan_btn.setEnabled(True)
        if self.feedback_apply_error:
            QMessageBox.warning(self, "模拟跨阻未生效", self.feedback_apply_error)
        if worker is not None:
            worker.deleteLater()
        self.feedback_apply_worker = None

    def set_dac_label(self, _dac_type):
        self.dac_label.setText("DAC: PI11210 · 每点回读确认")

    def reload_calibration_table(self):
        """Refresh the full source table and rebuild the selected range."""
        super().reload_calibration_table()
        self.full_points = tuple(self.points)
        self.full_guards = dict(self.guards)
        minimum_nm = float(self.full_points[0].target_nm)
        maximum_nm = float(self.full_points[-1].target_nm)
        for spin in (self.start_wavelength_spin, self.end_wavelength_spin):
            spin.setRange(minimum_nm, maximum_nm)
        self.start_wavelength_spin.setValue(minimum_nm)
        self.end_wavelength_spin.setValue(maximum_nm)
        self._update_range_preview()
        return True

    def _create_worker(self):
        worker = EqualIntervalWorker(
            ser,
            self.points,
            self.guards,
            settle_s=self.settle_spin.value(),
            feedback_selectors=self._selected_feedback_selectors(),
            repeat_scans=self.repeat_checkbox.isChecked(),
            parent=self,
        )
        worker.feedback_readback.connect(self._on_feedback_readback)
        worker.feedback_confirmed.connect(self._on_feedback_confirmed)
        return worker

    def _scan_table_error(self):
        if self.load_error:
            return self.load_error
        if not self.points:
            return "当前波长范围内没有标定点"
        return None

    def _update_range_preview(self, _value=None):
        try:
            selected = self._selected_range_points()
        except ValueError:
            self.mode_label.setText("等间隔模式 · 范围无效")
            return
        self.mode_label.setText(f"等间隔模式 · {len(selected)}点")

    def _selected_range_points(self):
        return select_fullband_target_range(
            self.full_points,
            self.start_wavelength_spin.value(),
            self.end_wavelength_spin.value(),
        )

    def start_scan(self):
        if self.feedback_apply_worker is not None and self.feedback_apply_worker.isRunning():
            QMessageBox.information(self, "正在设置跨阻", "请等待choseA/B IO核验完成")
            return
        try:
            selected = self._selected_range_points()
        except ValueError as exc:
            QMessageBox.warning(self, "波长范围无效", str(exc))
            return
        self.active_requested_range = (
            float(self.start_wavelength_spin.value()),
            float(self.end_wavelength_spin.value()),
        )
        self._configure_scan_points(selected)
        self.guards = self.full_guards
        count = len(selected)
        self.mode_label.setText(f"等间隔模式 · {count}点")
        self.repeat_checkbox.setToolTip(
            f"完成当前{count}点范围后按相同等待时间重新扫描"
        )
        self.confirmed_feedback_selectors = None
        self.observed_feedback_selectors = None
        self._update_feedback_display("pending")
        super().start_scan()

    def _set_scan_parameters_enabled(self, enabled):
        feedback_busy = (
            self.feedback_apply_worker is not None
            and self.feedback_apply_worker.isRunning()
        )
        enabled = bool(enabled) and not feedback_busy
        if hasattr(self, "settle_spin"):
            self.settle_spin.setEnabled(enabled)
            self.start_wavelength_spin.setEnabled(enabled)
            self.end_wavelength_spin.setEnabled(enabled)
        if hasattr(self, "ch0_feedback_btn"):
            self.ch0_feedback_btn.setEnabled(enabled)
            self.ch1_feedback_btn.setEnabled(enabled)

    def _clear_precise_fit(self):
        dialog = getattr(self, "precise_fit_dialog", None)
        if dialog is not None:
            dialog.close()
            self.precise_fit_dialog = None
        for item in getattr(self, "precise_fit_items", ()):
            try:
                self.plot.removeItem(item)
            except Exception:
                pass
        self.precise_fit_items = []
        self.precise_fit_results = [() for _ in range(4)]
        if hasattr(self, "precise_fit_btn"):
            self.precise_fit_btn.setEnabled(False)

    def _reset_arrays(self, scan_id=0):
        super()._reset_arrays(scan_id)
        if hasattr(self, "precise_fit_items"):
            self._clear_precise_fit()
        if hasattr(self, "auto_select_btn"):
            self.auto_selection_report = None
            self.auto_selection_path = None
            self.auto_select_btn.setEnabled(False)
            if len(self.points) == FULLBAND_ACCURACY_POINT_COUNT:
                text = "自动选点：等待完整2001点"
            else:
                text = "自动选点：局部范围仅用于光谱/拟合"
            self.auto_selection_label.setText(text)

    def clear_spectrum(self):
        self._clear_precise_fit()
        super().clear_spectrum()
        if self.worker is None or not self.worker.isRunning():
            self.status_label.setText(
                "每点：DAC回读确认 → 等待设定时间 → 丢弃旧数据 → 取1帧ADC → 实时绘制"
            )
            self.quality_label.setText("已采集：0 · 每点1帧ADC")
            self.noise_label.setText("原始单帧：未做均值、滤波或数字缩放")
            self.latest_adc_label.setText("当前ADC码 CH0~3：—")

    def _on_scan_started(self, scan_id):
        self._clear_precise_fit()
        super()._on_scan_started(scan_id)
        self.quality_label.setText("已采集：0 · 每点1帧ADC")
        self.noise_label.setText("原始单帧：未做均值、滤波或数字缩放")
        self.latest_adc_label.setText("当前ADC码 CH0~3：—")
        self.status_label.setText(
            f"第{scan_id}轮开始 · 每点等待{self.settle_spin.value():.3f}秒后采1帧ADC"
        )

    def _on_point_ready(self, result):
        super()._on_point_ready(result)
        codes = tuple(int(round(value)) for value in result.get("adc_codes", ()))
        volts = tuple(float(value) for value in result.get("voltages", ()))
        if len(codes) == 4 and len(volts) == 4:
            self.latest_adc_label.setText(
                "当前ADC码 CH0~3："
                + "/".join(str(value) for value in codes)
                + " · 电压："
                + "/".join(f"{value:.4f}" for value in volts)
                + " V"
            )

    def _on_scan_completed(self, scan_id, elapsed_s, _unused):
        self.precise_fit_btn.setEnabled(
            self.completed_points == len(self.points)
            and not self.repeat_checkbox.isChecked()
        )
        selection_message = self._generate_auto_mode_selection(scan_id)
        self.status_label.setText(
            f"第{scan_id}轮{len(self.points)}点完成 · 用时{self._format_duration(elapsed_s)}"
            f" · 原始单帧数据已全部绘制 · {selection_message}"
        )

    def _on_worker_finished(self):
        worker = self.worker
        cancelled = bool(worker is not None and worker.cancelled)
        had_error = bool(self.last_error)
        super()._on_worker_finished()
        if (
            not cancelled
            and not had_error
            and self.completed_points == len(self.points)
        ):
            self.precise_fit_btn.setEnabled(True)
            if self.auto_selection_report is not None:
                peak_count = int(
                    self.auto_selection_report["detected_peak_count"]
                )
                stress_count = int(self.auto_selection_report["stress"]["point_count"])
                temperature_count = int(
                    self.auto_selection_report["temperature"]["point_count"]
                )
                selection = (
                    f"已识别{peak_count}个有效峰 · "
                    f"应力{stress_count}点/温度{temperature_count}点候选表已生成"
                    + (
                        " · 45点稀疏顺序待复现"
                        if str(
                            self.auto_selection_report.get(
                                "sparse_sequence_validation", {}
                            ).get("status", "")
                        ).lower()
                        == "pending"
                        else ""
                    )
                )
            else:
                selection = self.auto_selection_label.text().replace("自动选点：", "")
            self.status_label.setText(
                f"{len(self.points)}点等间隔原始光谱已完成 · 每点1帧ADC · "
                f"{selection} · SOA已关闭"
            )

    def _generate_auto_mode_selection(self, scan_id):
        if self.completed_points != len(self.points):
            return "自动选点未执行：光谱不完整"
        if len(self.points) != FULLBAND_ACCURACY_POINT_COUNT:
            self.auto_selection_report = None
            self.auto_selection_path = None
            self.auto_select_btn.setEnabled(False)
            self.auto_selection_label.setText("自动选点：局部范围仅用于光谱/拟合")
            return "局部范围不生成全波段模式点表"
        feedback = (
            self.confirmed_feedback_selectors
            or self._selected_feedback_selectors()
        )
        try:
            ch1_noise = np.asarray(self.sigmas[1], dtype=float)
            usable_ch1_noise = np.isfinite(ch1_noise) & (ch1_noise > 0.0)
            report = select_mode_points_from_dense_spectrum(
                self.wavelengths,
                self.values,
                self.points,
                expected_peaks=FINGER_EXPECTED_PEAK_COUNT,
                fbg_channels=FINGER_FBG_CHANNELS,
                stress_point_count=FINGER_STRESS_POINT_COUNT,
                temperature_point_count=FINGER_TEMPERATURE_POINT_COUNT,
                feedback_selectors=feedback,
                sample_noise_std=(
                    ch1_noise if np.any(usable_ch1_noise) else None
                ),
                stable_mask=self.stable_flags,
            )
            report["created_at"] = datetime.now().isoformat(timespec="seconds")
            report["scan_id"] = int(scan_id)
            report["settle_s"] = float(self.settle_spin.value())
            output_dir = output_path()
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / (
                "equal_interval_auto_mode_selection_"
                + datetime.now().strftime("%Y%m%d_%H%M%S")
                + ".json"
            )
            path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
            self.auto_selection_report = report
            self.auto_selection_path = path
            self.auto_select_btn.setEnabled(True)
            peak_count = int(report["detected_peak_count"])
            stress_count = int(report["stress"]["point_count"])
            temperature_count = int(report["temperature"]["point_count"])
            sequence_status = str(
                report.get("sparse_sequence_validation", {}).get(
                    "status", "not_required"
                )
            ).lower()
            sequence_text = (
                " · 稀疏顺序待复现"
                if sequence_status == "pending"
                else ""
            )
            self.auto_selection_label.setText(
                f"自动选点：CH1已识别{peak_count}峰 · "
                f"应力{stress_count}点 · 温度{temperature_count}点"
                f"{sequence_text}"
            )
            return (
                f"CH1严格识别{peak_count}个有效峰，"
                f"已生成应力{stress_count}点与温度{temperature_count}点候选表；"
                + (
                    "45点稀疏顺序复现通过前不会覆盖旧表"
                    if sequence_status == "pending"
                    else ""
                )
            )
        except Exception as exc:
            self.auto_selection_report = None
            self.auto_selection_path = None
            self.auto_select_btn.setEnabled(False)
            message = str(exc)
            self.auto_selection_label.setText(f"自动选点：未通过 · {message}")
            return f"自动选点未通过：{message}"

    def _show_auto_mode_selection(self):
        report = self.auto_selection_report
        if not report:
            return
        # A separate hardware replay tool may certify the saved candidate
        # after this scan.  Reload only the same fingerprint; final validation
        # still happens inside mode_table_manager before any file is changed.
        try:
            saved = json.loads(
                Path(self.auto_selection_path).read_text(encoding="utf-8")
            )
            current_fingerprint = report.get(
                "sparse_sequence_validation", {}
            ).get("stress_table_fingerprint_sha256")
            saved_fingerprint = saved.get(
                "sparse_sequence_validation", {}
            ).get("stress_table_fingerprint_sha256")
            if current_fingerprint and current_fingerprint == saved_fingerprint:
                report = saved
                self.auto_selection_report = saved
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("应力/温度模式自动选点结果")
        dialog.resize(860, 560)
        layout = QtWidgets.QVBoxLayout(dialog)
        peak_count = int(report["detected_peak_count"])
        stress_point_count = int(report["stress"]["point_count"])
        temperature_point_count = int(report["temperature"]["point_count"])
        stress_counts = tuple(int(value) for value in report["stress"]["points_per_peak"])
        temperature_counts = tuple(
            int(value) for value in report["temperature"]["points_per_peak"]
        )

        def count_description(counts):
            low, high = min(counts), max(counts)
            return f"每峰{low}点" if low == high else f"每峰{low}～{high}点"

        sequence_validation = report.get("sparse_sequence_validation")
        sequence_passed = (
            not isinstance(sequence_validation, dict)
            or str(sequence_validation.get("status", "")).lower() == "passed"
        )
        sequence_summary = (
            "45点真实稀疏顺序已复现通过。"
            if sequence_passed
            else "45点真实稀疏顺序待复现，目前只能查看候选表。"
        )
        summary = QtWidgets.QLabel(
            f"已从本轮2001点原始ADC光谱中自适应定位{peak_count}个有效峰；"
            f"光栅数据源为CH1，应力表{stress_point_count}点，"
            f"温度表{temperature_point_count}点；"
            f"应力模式{count_description(stress_counts)}，"
            f"温度模式{count_description(temperature_counts)}，"
            "选点依据实测非高斯模板的位移灵敏度。"
            "所有DAC码均直接来自已标定2001点表；"
            + sequence_summary
        )
        summary.setWordWrap(True)
        summary.setObjectName("softHint")
        layout.addWidget(summary)
        stress_peaks = report["stress"]["peaks"]
        temperature_peaks = report["temperature"]["peaks"]
        table = QtWidgets.QTableWidget(len(stress_peaks), 6, dialog)
        table.setHorizontalHeaderLabels(
            ("峰", "应力中心/nm", "应力源通道", "温度中心/nm", "温度源通道", "温度中心±/pm")
        )
        for row, (stress_peak, temperature_peak) in enumerate(
            zip(stress_peaks, temperature_peaks, strict=True)
        ):
            entries = (
                str(row + 1),
                f"{stress_peak['center_nm']:.6f}",
                f"CH{stress_peak['source_channel']}",
                f"{temperature_peak['center_nm']:.6f}",
                f"CH{temperature_peak['source_channel']}",
                f"{temperature_peak['center_std_pm']:.3f}",
            )
            for column, value in enumerate(entries):
                table.setItem(row, column, QtWidgets.QTableWidgetItem(value))
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        layout.addWidget(table)
        path_label = QtWidgets.QLabel(f"候选表：{self.auto_selection_path}")
        path_label.setWordWrap(True)
        layout.addWidget(path_label)
        button_row = QtWidgets.QWidget(dialog)
        button_layout = QtWidgets.QHBoxLayout(button_row)
        button_layout.setContentsMargins(0, 0, 0, 0)
        restore_btn = QtWidgets.QPushButton("恢复上一版点表")
        restore_btn.setToolTip("恢复最近一次自动覆盖前的应力/温度表，并生成可上传的OTA包")
        restore_btn.setEnabled(latest_mode_table_backup() is not None)
        restore_btn.clicked.connect(
            lambda _checked=False: self._confirm_mode_table_action("restore", dialog)
        )
        apply_btn = QtWidgets.QPushButton("应用为正式点表并生成 OTA 包")
        apply_btn.setProperty("role", "primary")
        apply_btn.setEnabled(sequence_passed)
        apply_btn.setToolTip(
            f"先备份当前正式点表，再写入应力{stress_point_count}点"
            f"和温度{temperature_point_count}点；"
            + (
                "编译失败会自动恢复，生成包后仍需人工确认OTA上传"
                if sequence_passed
                else "需先用真实激光器完整复现45点稀疏顺序"
            )
        )
        apply_btn.clicked.connect(
            lambda _checked=False: self._confirm_mode_table_action("apply", dialog)
        )
        close_btn = QtWidgets.QPushButton("关闭")
        close_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(restore_btn)
        button_layout.addStretch(1)
        button_layout.addWidget(close_btn)
        button_layout.addWidget(apply_btn)
        layout.addWidget(button_row)
        dialog.exec_()

    def _confirm_mode_table_action(self, action, dialog=None):
        worker = self.mode_table_worker
        if worker is not None and worker.isRunning():
            QMessageBox.information(self, "点表正在生成", "请等待当前编译和打包完成。")
            return
        scan_worker = self.worker
        if scan_worker is not None and scan_worker.isRunning():
            QMessageBox.information(self, "扫描仍在运行", "请先停止等间隔扫描再更换正式点表。")
            return

        if action == "apply":
            validation = self.auto_selection_report.get(
                "sparse_sequence_validation"
            )
            if (
                isinstance(validation, dict)
                and str(validation.get("status", "")).lower() != "passed"
            ):
                QMessageBox.warning(
                    self,
                    "稀疏顺序尚未复现",
                    "45点候选顺序尚未在真实激光器上完整验证，"
                    "当前正式点表保持不变。",
                )
                return
            peak_count = int(self.auto_selection_report["detected_peak_count"])
            stress_point_count = int(
                self.auto_selection_report["stress"]["point_count"]
            )
            temperature_point_count = int(
                self.auto_selection_report["temperature"]["point_count"]
            )
            title = "确认覆盖正式点表"
            message = (
                f"将使用本轮识别出的{peak_count}个有效峰：\n"
                f"• 覆盖应力模式{stress_point_count}点和"
                f"温度模式{temperature_point_count}点\n"
                "• 自动备份当前正式点表\n"
                "• 重新编译并生成OTA包\n\n"
                "此步骤不会自动上传板卡，仍需你在OTA页面确认。是否继续？"
            )
        else:
            if latest_mode_table_backup() is None:
                QMessageBox.information(self, "没有备份", "当前没有可恢复的上一版正式点表。")
                return
            title = "确认恢复上一版点表"
            message = (
                "将恢复最近一次覆盖前的应力和温度点表，并重新生成OTA包。\n"
                "此步骤同样不会自动上传板卡。是否继续？"
            )
        answer = QMessageBox.question(
            self,
            title,
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        if dialog is not None:
            dialog.accept()
        self._start_mode_table_action(action)

    def _start_mode_table_action(self, action):
        self.mode_table_progress = QtWidgets.QProgressDialog(
            "正在准备正式点表…", "", 0, 0, self
        )
        self.mode_table_progress.setWindowTitle("生成正式点表固件")
        self.mode_table_progress.setCancelButton(None)
        self.mode_table_progress.setWindowModality(QtCore.Qt.WindowModal)
        self.mode_table_progress.setMinimumDuration(0)
        self.mode_table_progress.show()
        self.mode_table_worker = ModeTableBuildWorker(
            action,
            report=self.auto_selection_report if action == "apply" else None,
            candidate_path=self.auto_selection_path if action == "apply" else None,
            parent=self,
        )
        self.mode_table_worker.progress.connect(self._mode_table_progress_changed)
        self.mode_table_worker.completed.connect(self._mode_table_action_completed)
        self.mode_table_worker.failed.connect(self._mode_table_action_failed)
        self.mode_table_worker.finished.connect(self._mode_table_action_finished)
        self.mode_table_worker.start()

    def _mode_table_progress_changed(self, message):
        if self.mode_table_progress is not None:
            self.mode_table_progress.setLabelText(str(message))

    def _mode_table_action_completed(self, result):
        if self.mode_table_progress is not None:
            self.mode_table_progress.close()
        if result.action == "apply":
            self.auto_selection_label.setText(
                f"正式点表：已写入{result.detected_peak_count}峰候选 · OTA包已生成"
            )
            title = "正式点表已生成"
            detail = (
                f"应力{result.stress_point_count}点和"
                f"温度{result.temperature_point_count}点已覆盖到固件工程，"
                "并已备份上一版。\n"
                f"OTA版本：{result.version}\n"
                f"固件包：{result.package_path}\n\n"
                "OTA成功前桌面仍保持旧波长轴，避免与板卡错配。\n"
                "要现在转到局域网OTA页面吗？"
            )
        else:
            self.auto_selection_label.setText("正式点表：上一版已恢复 · OTA包已生成")
            title = "上一版点表已恢复"
            detail = (
                f"恢复固件已经生成。\nOTA版本：{result.version}\n"
                f"固件包：{result.package_path}\n\n"
                "要现在转到局域网OTA页面吗？"
            )
        answer = QMessageBox.question(
            self,
            title,
            detail,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer == QMessageBox.Yes and self.ota_package_ready is not None:
            self.ota_package_ready(str(result.package_path))

    def _mode_table_action_failed(self, message):
        if self.mode_table_progress is not None:
            self.mode_table_progress.close()
        QMessageBox.warning(
            self,
            "正式点表未更改",
            f"{message}\n\n如果写入或编译中断，程序已经自动恢复原来的正式点表。",
        )

    def _mode_table_action_finished(self):
        if self.mode_table_progress is not None:
            self.mode_table_progress.close()
            self.mode_table_progress.deleteLater()
        self.mode_table_progress = None
        worker = self.mode_table_worker
        self.mode_table_worker = None
        if worker is not None:
            worker.deleteLater()

    def fit_reflection_spectrum(self):
        """Fit every resolvable Bragg peak in the completed dense scan."""

        if self.completed_points != len(self.points):
            QMessageBox.information(
                self,
                "精确反射谱拟合",
                f"请先完成当前范围的{len(self.points)}点等间隔扫描。",
            )
            return
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(
                self,
                "精确反射谱拟合",
                "扫描仍在运行，请先停止自动重扫后再拟合。",
            )
            return

        self._clear_precise_fit()
        channel_summaries = []
        total_valid = 0
        total_rejected = 0
        for channel, color in enumerate(self.CHANNEL_COLORS):
            result = fit_dense_reflection_spectrum(
                self.wavelengths,
                self.values[channel],
            )
            self.precise_fit_results[channel] = result.fits
            total_valid += len(result.fits)
            total_rejected += result.rejected_count
            channel_summaries.append(result)
            for peak_number, fit in enumerate(result.fits, start=1):
                half_width = max(3.2 * float(fit.sigma_nm), 0.16)
                dense_x = np.linspace(
                    max(float(self.wavelengths[0]), fit.center_nm - half_width),
                    min(float(self.wavelengths[-1]), fit.center_nm + half_width),
                    161,
                )
                fitted_curve = pg.PlotDataItem(
                    dense_x,
                    fit.evaluate(dense_x),
                    pen=pg.mkPen(color, width=2.8),
                    connect="finite",
                )
                self.plot.addItem(fitted_curve)
                self.precise_fit_items.append(fitted_curve)

                peak_y = float(fit.evaluate(np.asarray((fit.center_nm,)))[0])
                marker = pg.ScatterPlotItem(
                    x=[fit.center_nm],
                    y=[peak_y],
                    size=11,
                    symbol="d",
                    pen=pg.mkPen("#FFFFFF", width=1.2),
                    brush=pg.mkBrush(color),
                )
                self.plot.addItem(marker)
                self.precise_fit_items.append(marker)
                label = pg.TextItem(
                    text=f"CH{channel}-P{peak_number}\n{fit.center_nm:.5f} nm",
                    color=color,
                    anchor=(0.5, 1.12),
                    border=pg.mkPen(color),
                    fill=pg.mkBrush(15, 19, 26, 205),
                )
                label.setPos(fit.center_nm, peak_y)
                self.plot.addItem(label)
                self.precise_fit_items.append(label)

        self.precise_fit_btn.setEnabled(True)
        if total_valid == 0:
            self.status_label.setText(
                "精确拟合完成：没有找到满足显著度和拟合质量要求的布拉格峰"
            )
            QMessageBox.information(
                self,
                "精确反射谱拟合",
                "没有找到可信布拉格峰。可适当延长激光器稳定时间后重新扫描。",
            )
            return

        self.status_label.setText(
            f"精确拟合完成：找到{total_valid}个可信布拉格峰"
            f" · 排除{total_rejected}个低质量候选峰"
        )
        self._show_precise_fit_results(channel_summaries)

    def _show_precise_fit_results(self, channel_summaries):
        if self.precise_fit_dialog is not None:
            self.precise_fit_dialog.close()
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("布拉格光栅精确反射谱拟合结果")
        dialog.resize(940, 520)
        layout = QtWidgets.QVBoxLayout(dialog)
        summary = QtWidgets.QLabel(
            "中心波长由本次范围内的实测波长坐标进行单峰高斯亚采样拟合；"
            "±值为本轮拟合估计的中心标准不确定度。"
        )
        summary.setWordWrap(True)
        summary.setObjectName("softHint")
        layout.addWidget(summary)

        rows = sum(len(result.fits) for result in channel_summaries)
        table = QtWidgets.QTableWidget(rows, 8, dialog)
        table.setHorizontalHeaderLabels(
            (
                "通道",
                "峰序号",
                "中心波长/nm",
                "中心±/pm",
                "峰高/V",
                "FWHM/nm",
                "R²",
                "RMSE/mV",
            )
        )
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        row = 0
        for channel, result in enumerate(channel_summaries):
            for peak_number, fit in enumerate(result.fits, start=1):
                values = (
                    f"CH{channel}",
                    str(peak_number),
                    f"{fit.center_nm:.6f}",
                    f"{fit.center_std_pm:.3f}",
                    f"{fit.amplitude_v:.5f}",
                    f"{2.354820045 * fit.sigma_nm:.5f}",
                    f"{fit.r_squared:.5f}",
                    f"{fit.rmse_v * 1000.0:.3f}",
                )
                for column, value in enumerate(values):
                    table.setItem(row, column, QtWidgets.QTableWidgetItem(value))
                row += 1
        table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents
        )
        table.horizontalHeader().setStretchLastSection(True)
        table.setSortingEnabled(True)
        layout.addWidget(table, 1)
        close_buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        close_buttons.rejected.connect(dialog.close)
        layout.addWidget(close_buttons)
        self.precise_fit_dialog = dialog
        dialog.show()

    def shutdown(self):
        table_worker = self.mode_table_worker
        if table_worker is not None and table_worker.isRunning():
            # Source replacement and firmware linking are transactional and must
            # not be terminated between backup and package validation.
            table_worker.wait(180000)
        worker = self.feedback_apply_worker
        if worker is not None and worker.isRunning():
            worker.stop()
            worker.wait(2500)
        self.feedback_apply_worker = None
        super().shutdown()


class MainWindow(QMainWindow):
    def __init__(self, auto_acquire_local=True):
        super().__init__()

        self.auto_acquire_local = bool(auto_acquire_local)

        self.setWindowTitle("可调谐激光器")
        self.resize(1400, 800)

        menubar = self.menuBar()

        self.menu_port = menubar.addMenu("端口")
        self.menu_baud = menubar.addMenu("波特率")
        self.menu_page = menubar.addMenu("工作模式")
        # Legacy MS5614T/U_DAC selection menu is intentionally disabled.

        self.baud_rates = [9600, 115200, 2000000, 3000000, 4000000, 6000000]
        self.baud = 2000000
        self.baud_act = None

        self.ports = []
        listp = list_ports.comports()
        for p in listp:
            self.ports.append([p.device, p.description])
        self.port = find_native_stm32_cdc_port(listp)
        self.port_act = None

        self.MCU_mode = 3
        self.mode_act = None

        global ser
        ser.port = self.port
        ser.baudrate = self.baud

        self.stack = QStackedWidget()
        # self.stack.currentChanged.connect(self.on_switch_mode)

        self.page_peak = GraphWindow(precision_mode=False)
        self.page_temperature_peak = GraphWindow(precision_mode=True)
        self.page_ap6150 = FullbandAutoCalibrationWindow(
            prepare_session=self._prepare_fullband_calibration_session,
            finish_session=self._finish_fullband_calibration_session,
            parent=self,
        )
        self.page_ap6150.calibration_promoted.connect(
            self._reload_certified_fullband_table
        )
        self.page_extra = extraWindow()
        self.page_unlimited_accuracy = UnlimitedAccuracyWindow()
        self.page_equal_interval = EqualIntervalWindow()
        from temporary_test_widget import TemporaryTestWindow
        self.page_temporary_test = TemporaryTestWindow(self)

        self.stack.addWidget(self.page_peak)
        self.stack.addWidget(self.page_temperature_peak)
        self.stack.addWidget(self.page_ap6150)
        self.stack.addWidget(self.page_extra)
        self.stack.addWidget(self.page_unlimited_accuracy)
        self.stack.addWidget(self.page_equal_interval)
        self.stack.addWidget(self.page_temporary_test)

        self.setCentralWidget(self.stack)

        self.init_menu()
        self.update_dac_labels()
        self.stack.setCurrentIndex(self.MCU_mode)
        if self.auto_acquire_local:
            QtCore.QTimer.singleShot(0, self.acquire_local_control)

    def init_menu(self):
        # 端口
        self.menu_port.aboutToShow.connect(self.update_ports_menu)

        # 波特率
        for baud in self.baud_rates:
            action = QAction(str(baud), self)
            action.setCheckable(True)
            action.setData(baud)
            action.triggered.connect(self.set_baudrate)
            self.menu_baud.addAction(action)
            if baud == self.baud:
                action.setChecked(True)
                self.baud_act = action

        # 工作模式
        work_mode = [
            "应力寻峰模式",
            "温度寻峰模式",
            "自动校准模式",
            "单值模式",
            "无限准确模式",
            "等间隔模式",
            "临时测试模式",
        ]
        for i, mode_str in enumerate(work_mode):
            action = QAction(mode_str,self)
            action.setCheckable(True)
            action.setData(i)
            action.triggered.connect(self.set_page)
            self.menu_page.addAction(action)
            if i == self.MCU_mode:
                self.mode_act = action
                action.setChecked(True)

        # Legacy MS5614T/U_DAC selector removed: PI11210 is always used.
                
    def update_dac_labels(self):
        self.page_peak.set_dac_label(dac_type)
        self.page_temperature_peak.set_dac_label(dac_type)
        self.page_ap6150.set_dac_label(dac_type)
        self.page_extra.set_dac_label(dac_type)
        self.page_unlimited_accuracy.set_dac_label(dac_type)
        self.page_equal_interval.set_dac_label(dac_type)

    def reload_mode_tables(self):
        """Reload desktop wavelength axes after a matching OTA succeeds."""
        self.page_peak.reload_wavelength_table()
        self.page_temperature_peak.reload_wavelength_table()

    def reload_machine_calibration(self, machine_id):
        """Switch every 2001-point consumer to one physical machine's table."""

        temporary_page = getattr(self, "page_temporary_test", None)
        if temporary_page is not None:
            temporary_page.before_machine_switch()
        set_runtime_machine_id(machine_id)
        errors = []
        for page in (self.page_unlimited_accuracy, self.page_equal_interval):
            try:
                page.reload_calibration_table()
            except Exception as exc:
                errors.append(str(exc))
        try:
            self.page_extra.reload_calibration_table()
        except Exception as exc:
            errors.append(str(exc))
        if temporary_page is not None:
            temporary_page.reload_machine_profile(
                runtime_machine_label(machine_id),
                runtime_parameter_note(machine_id),
            )
        return errors

    def _prepare_fullband_calibration_session(self):
        """Shutter the laser and hand the CDC port to the calibration process."""
        global ap_open
        global switch_mode_enable
        if ap_open or not switch_mode_enable:
            raise RuntimeError("当前有其他采集或配置流程占用设备")
        ensure_serial_open()
        set_soa_shutter_and_verify(ser)
        port = str(ser.port or self.port or "COM6")
        ser.close()
        with ap_cond:
            ap_open = True
        switch_mode_enable = False
        return port

    def _finish_fullband_calibration_session(self, _success):
        """Reclaim the CDC port and independently confirm final shutter state."""
        global ap_open
        global switch_mode_enable
        error = None
        try:
            ensure_serial_open()
            set_soa_shutter_and_verify(ser)
        except Exception as exc:
            error = exc
        finally:
            with ap_cond:
                ap_open = False
            switch_mode_enable = True
        if error is not None:
            raise error

    def _reload_certified_fullband_table(self, _output_path):
        self.page_unlimited_accuracy.reload_calibration_table()
        self.page_equal_interval.reload_calibration_table()

    def start_stress_debug(self):
        """Select and start the local stress page after the main window opens."""
        global switch_mode_enable
        if not switch_mode_enable:
            return
        self.MCU_mode = 0
        self.stack.setCurrentIndex(0)
        for action in self.menu_page.actions():
            selected = action.data() == 0
            action.setChecked(selected)
            if selected:
                self.mode_act = action
        if not ser_open:
            self.page_peak.on_open_changed()

    def start_temperature_debug(self):
        """Select and start the local full-table precision page."""
        global switch_mode_enable
        if not switch_mode_enable:
            return
        self.MCU_mode = 1
        self.stack.setCurrentIndex(1)
        for action in self.menu_page.actions():
            selected = action.data() == 1
            action.setChecked(selected)
            if selected:
                self.mode_act = action
        if not ser_open:
            self.page_temperature_peak.on_open_changed()

    def acquire_local_control(self):
        """Claim the selected CDC port and report whether ownership succeeded."""
        global local_control_session
        try:
            local_control_session = True
            ensure_serial_open()
            ser.reset_input_buffer()
            return True
        except Exception as exc:
            local_control_session = False
            try:
                release_serial_if_allowed()
            except Exception:
                pass
            QMessageBox.warning(
                self,
                "本地控制未启用",
                f"无法打开 {self.port or 'STM32 USB'}，板卡仍保持远端控制：\n{exc}",
            )
            return False

    def update_ports_menu(self):
        menu = self.sender()

        menu.clear()
        self.ports.clear()

        for p in list_ports.comports():
            self.ports.append([p.device, p.description])
            action = QAction(f"{p.device} {p.description}", self)
            action.setCheckable(True)
            action.setData(p.device)
            action.triggered.connect(self.set_com_port)
            menu.addAction(action)
            if p.device == self.port:
                action.setChecked(True)
                self.port_act = action

    def set_dac(self):
        # Legacy entry point retained only for compatibility.  DAC switching
        # is disabled because the hardware now uses PI11210 exclusively.
        return
        
    def set_page(self):
        action = self.sender()
        if not switch_mode_enable:
            self.mode_act.setChecked(True)
            action.setChecked(False)
            QMessageBox.warning(self, "警告", "先停止当前页面工作流再切换模式")
            return

        self.MCU_mode = action.data()
        self.stack.setCurrentIndex(self.MCU_mode)
        self.mode_act.setChecked(False)
        action.setChecked(True)
        self.mode_act = action
        # if self.MCU_mode == 2:
        #     self.page_extra.monitor_btn.click()

    def set_com_port(self):
        global ser
        action = self.sender()
        if switch_mode_enable:
            was_open = ser.is_open
            if was_open:
                ser.close()
            self.port = action.data()
            ser.port = self.port
            if local_control_session:
                ensure_serial_open()
            if self.port_act is not None:
                self.port_act.setChecked(False)
            action.setChecked(True)
            self.port_act = action
        else:
            action.setChecked(False)
            QMessageBox.warning(self, "警告", "先停止当前页面工作流停止再修改端口")

    def set_baudrate(self):
        global ser
        action = self.sender()
        if switch_mode_enable:
            self.baud = action.data()
            ser.baudrate = self.baud
            self.baud_act.setChecked(False)
            action.setChecked(True)
            self.baud_act = action
        else:
            action.setChecked(False)
            QMessageBox.warning(self, "警告", "先停止当前页面工作流停止再修改波特率")

    def closeEvent(self, a0):
        global ser_open
        global local_control_session
        temporary_page = getattr(self, "page_temporary_test", None)
        temporary_worker = getattr(temporary_page, "worker", None)
        if temporary_worker is not None and temporary_worker.isRunning():
            temporary_page.stop()
            if not getattr(self, "_temporary_close_pending", False):
                self._temporary_close_pending = True
                temporary_worker.finished.connect(self.close)
            a0.ignore()
            return
        # Keep the event loop responsive while the sweep worker leaves a
        # possible VISA query and verifies SOA shutter.  Retrying close from
        # ``finished`` guarantees that neither the QThread nor the shared CDC
        # handle is destroyed underneath that safety sequence.
        ap_worker = getattr(self.page_ap6150, "worker", None)
        if ap_worker is not None and ap_worker.isRunning():
            ap_worker.stop()
            if not getattr(self, "_ap_close_pending", False):
                self._ap_close_pending = True
                ap_worker.finished.connect(self._retry_close_after_ap_worker)
            a0.ignore()
            return
        if (temporary_page is not None
                and not temporary_page.confirm_save_before_close(self)):
            a0.ignore()
            return
        self.page_unlimited_accuracy.shutdown()
        self.page_equal_interval.shutdown()
        for page in (self.page_peak, self.page_temperature_peak):
            worker = getattr(page, "worker", None)
            if worker is not None:
                worker.running = False
            reference_worker = getattr(page, "stable_reference_worker", None)
            if reference_worker is not None:
                reference_worker.running = False
            page.frame_timer.stop()
            page.update_timer.stop()
            page.fps_timer.stop()
        with ser_cond:
            ser_open = False
            ser_cond.notify_all()
        for page in (self.page_peak, self.page_temperature_peak):
            worker = getattr(page, "worker", None)
            if worker is not None and worker.isRunning():
                worker.wait(1000)
            reference_worker = getattr(page, "stable_reference_worker", None)
            if reference_worker is not None and reference_worker.isRunning():
                reference_worker.wait(1500)
            finger_window = getattr(page, "finger_3d_window", None)
            if finger_window is not None:
                finger_window.close()
        shutdown_error = None
        if ser.is_open:
            try:
                self.page_peak._stop_optical_session_safely()
            except Exception as exc:
                shutdown_error = str(exc)
                print("Application-exit optical safety error:", exc)
        self.page_extra.stop_recv_thread()
        local_control_session = False
        release_serial_if_allowed()
        if shutdown_error is not None:
            QMessageBox.critical(
                self,
                "退出安全核验失败",
                shutdown_error,
            )
        return super().closeEvent(a0)

    def _retry_close_after_ap_worker(self):
        self._ap_close_pending = False
        QtCore.QTimer.singleShot(0, self.close)

class GraphWindow(QtWidgets.QWidget):
    @staticmethod
    def detect_wave_gaps(waves, gap_ratio=1.75, local_radius=4):
        """Return indices after which the YAML wavelength sequence is discontinuous."""
        return detect_wavelength_gaps(waves)

    def __init__(self, precision_mode=False):
        super().__init__()
        self.setObjectName("graphPage")
        self.precision_mode = bool(precision_mode)
        self.mcu_work_mode = 3 if self.precision_mode else 0
        self.mode_name = "温度寻峰模式" if self.precision_mode else "应力寻峰模式"
        # The populated analogue front end uses 40 kOhm feedback on CH0/CH1
        # and 2 kOhm on CH2/CH3.  Temperature mode deliberately uses only the
        # two high-gain channels.  Stress-mode channels are no longer assigned
        # a hard-coded subset of grating numbers: every active optical channel
        # independently reports every peak it actually contains.
        self.active_fbg_channels = {0, 1} if self.precision_mode else {0, 1, 2}
        self.visible_plot_channels = {0, 1} if self.precision_mode else {0, 1, 2, 3}
        self.active_channel_mask = 0x03 if self.precision_mode else 0x0F
        self.default_fbg_channels = self.active_fbg_channels.copy()
        self.paused = False
        self.process_down = True
        self.pending_frame_received_ns = None
        self.frame_age_samples_ms = deque(maxlen=300)
        self.last_frame_received_monotonic_ns = None
        self.last_frame_age_ms = math.nan
        self.board_frame_sequence = None
        self.board_acquisition_duration_us = None
        self.board_table_crc32 = None
        self.board_boot_id = None
        self.board_uptime_ms = None
        self.board_sequence_tracker = BoardSequenceTracker()
        self.board_frame_schedule = None
        self.multirate_configuration = MultirateConfiguration()
        self.multirate_armed = False
        self.bandwidth_discontinuity = False
        # Public per-frame acquisition metadata used by the unified UI and by
        # recorded training rows.  A legacy frame is a genuine full MAP: old
        # firmware cannot enter reduced sampling without the new arm command.
        self.acquisition_profile = "MAP"
        self.fresh_point_indices = tuple(range(MULTIRATE_POINT_COUNT))
        self.fresh_point_mask = (True,) * MULTIRATE_POINT_COUNT
        self.point_age_frames = (0.0,) * MULTIRATE_POINT_COUNT
        self.sample_offset_us = (None,) * MULTIRATE_POINT_COUNT
        self.map_age_frames = 0
        self.frame_start_device_ms = None
        self.multirate_schedule_version = None
        self.current_frame_is_complete_map = True
        self.last_complete_map_channels = None
        self.contact_localizer = None
        self.contact_tracker = None
        self.contact_estimate = None
        self.contact_localizer_error = ""
        self.contact_baseline_adc_codes = None
        self.contact_baseline_transimpedance_ohm = None
        self.contact_baseline_capture_transimpedance_ohm = None
        self.contact_baseline_samples = []
        self.contact_baseline_capture_active = False
        self.contact_baseline_rejected_frames = 0
        self._bound_contact_3d_window = None
        self.fps_window_started_ns = time.perf_counter_ns()
        self.peaks_lines = [[] for _ in range(4)]
        self.visible_lines = [False for _ in range(4)]
        self.us_scatter_items = []
        self.filter_scatter_items = []
        self.stable_reference_values = None
        self.stable_reference_fits = [[] for _ in range(4)]
        self.stable_reference_channels = set()
        self.stable_reference_worker = None
        self.reference_capture_active = False
        self.finger_3d_window = None
        # The unified studio embeds the 3D view as a main-window tab.  The
        # callback lets this legacy page select that tab without opening a
        # second top-level window.  Standalone use keeps the original behavior.
        self.finger_3d_open_handler = None
        self.stress_reference_offset = None
        self.stress_reference_fast_frames = deque(
            maxlen=STRESS_REFERENCE_FAST_CALIBRATION_FRAMES
        )
        self.stress_reference_calibration_pending = False

        page_layout = QtWidgets.QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        self.page_scroll = ResponsiveScrollArea()
        self.page_scroll.setObjectName("graphPageScroll")
        self.page_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.page_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        content = ScrollContentWidget()
        content.setObjectName("graphPageContent")
        layout = QtWidgets.QGridLayout(content)
        layout.setContentsMargins(8, 8, 8, 6)
        layout.setVerticalSpacing(9)
        self.page_scroll.setWidget(content)
        page_layout.addWidget(self.page_scroll)

        self.ctrl_panel = QtWidgets.QFrame()
        self.ctrl_panel.setObjectName("controlGroup")
        self.ctrl_layout = FlowLayout(
            margin=10,
            horizontal_spacing=9,
            vertical_spacing=8,
        )
        self.ctrl_panel.setLayout(self.ctrl_layout)

        self.fps_label = QtWidgets.QLabel(
            "刷新: 0.0 Hz"
            if self.precision_mode
            else "压力场 0.0 Hz · 丢帧 0 · 帧龄 —"
        )
        self.fps_label.setMinimumWidth(100)
        self.fps_label.setAlignment(QtCore.Qt.AlignCenter)
        self.fps_label.setObjectName("metricBadge")
        if not self.precision_mode:
            self.fps_label.setMinimumWidth(390)
            self.fps_label.setToolTip(
                "压力场刷新率按完成拟合和3D更新的帧计数；"
                "跳帧是电脑为保持实时性而略过的旧完整帧；"
                "板采是STM32从扫描入口到ADC/温度/状态完成的时长；"
                "帧龄从完整帧进入电脑解析器算到本次显示更新完成。"
            )

        self.dac_label = QtWidgets.QLabel("DAC: PI11210")
        self.dac_label.setMinimumWidth(100)
        self.dac_label.setAlignment(QtCore.Qt.AlignCenter)
        self.dac_label.setObjectName("metricBadge")

        lab_temperature = QtWidgets.QLabel("温度(℃):")
        
        self.temperature_text = QtWidgets.QLineEdit("0")
        self.temperature_text.setReadOnly(True)
        self.temperature_text.setMinimumWidth(82)
        self.temperature = 0

        diff_threshold_label = QtWidgets.QLabel("最小峰高(V):")
        self.diff_threshold_text = QtWidgets.QLineEdit("0.03")
        self.diff_threshold_text.setMinimumWidth(82)
        self.diff_threshold_text.textChanged.connect(self.on_threshold_changed)

        self.voltage_scalars = [1.0, 1.0, 1.0, 1.0]
        self.digital_gain_spins = []

        # 原始点只用于核查，中心波长始终由稳健拟合计算。
        self.points_toggle = QtWidgets.QRadioButton("显示原始采样点")
        self.points_toggle.setChecked(False)
        self.points_toggle.setAutoExclusive(False)
        self.points_toggle.toggled.connect(self.on_toggle_points)

        # 仅控制圆润曲线显示，不改变中心波长计算结果。
        self.filter_toggle = QtWidgets.QRadioButton("圆润拟合显示")
        self.filter_toggle.setChecked(True)
        self.filter_toggle.setAutoExclusive(False)
        self.filter_toggle.toggled.connect(self.on_toggle_filter)

        self.com_btn = QtWidgets.QPushButton("打开")
        self.com_btn.setMinimumWidth(100)
        self.com_btn.setProperty("role", "primary")

        self.clear_btn = QtWidgets.QPushButton("清空数据")
        self.clear_btn.setMinimumWidth(100)
        self.clear_btn.setProperty("role", "danger")

        self.stable_reference_btn = QtWidgets.QPushButton("显示稳定单值曲线")
        self.stable_reference_btn.setMinimumWidth(160)
        self.stable_reference_btn.setCheckable(True)
        self.stable_reference_btn.setEnabled(False)
        self.stable_reference_btn.setVisible(not self.precision_mode)

        self.finger_3d_btn = QtWidgets.QPushButton("机械手指3D应力定位")
        self.finger_3d_btn.setMinimumWidth(175)
        self.finger_3d_btn.setVisible(not self.precision_mode)

        self.com_btn.clicked.connect(self.on_open_changed)
        self.clear_btn.clicked.connect(self.on_clear_chart)
        self.stable_reference_btn.toggled.connect(
            self.on_stable_reference_toggled
        )
        self.finger_3d_btn.clicked.connect(self.open_finger_3d)

        self.ctrl_layout.addWidget(self.fps_label)
        self.ctrl_layout.addWidget(self.dac_label)
        self.ctrl_layout.addWidget(
            compact_field(lab_temperature.text(), self.temperature_text)
        )
        self.ctrl_layout.addWidget(
            compact_field(diff_threshold_label.text(), self.diff_threshold_text)
        )
        digital_gain_field = QtWidgets.QWidget()
        digital_gain_layout = QtWidgets.QHBoxLayout(digital_gain_field)
        digital_gain_layout.setContentsMargins(0, 0, 0, 0)
        digital_gain_layout.setSpacing(6)
        digital_gain_layout.addWidget(QtWidgets.QLabel("数字倍率"))
        digital_channels = range(2) if self.precision_mode else range(4)
        for channel in digital_channels:
            spin = QtWidgets.QDoubleSpinBox()
            spin.setObjectName(f"ch{channel}DigitalGainSpin")
            spin.setDecimals(2)
            spin.setRange(0.01, 64.0)
            spin.setSingleStep(0.10)
            spin.setValue(1.0)
            spin.setPrefix(f"CH{channel} ×")
            spin.setMinimumWidth(102)
            spin.setToolTip("仅改变曲线显示倍率，原始ADC数据和峰中心计算始终保留")
            spin.valueChanged.connect(
                lambda value, index=channel: self._on_digital_gain_changed(
                    index, value
                )
            )
            self.digital_gain_spins.append(spin)
            digital_gain_layout.addWidget(spin)
        self.ctrl_layout.addWidget(digital_gain_field)

        self.ctrl_layout.addWidget(self.points_toggle)
        self.ctrl_layout.addWidget(self.filter_toggle)
        self.auto_scale_label = QtWidgets.QLabel(
            "模拟跨阻：继承等间隔选点时的CH0/CH1档位；"
            "数字显示：等待各波段首次有效峰"
            if self.precision_mode else
            "模拟跨阻：CH0/CH1手动锁定40 kΩ，CH2/CH3硬件固定2 kΩ；"
            "数字显示：CH0~3均为手动×1.00"
        )
        self.auto_scale_label.setMinimumWidth(220)
        self.auto_scale_label.setAlignment(QtCore.Qt.AlignCenter)
        self.auto_scale_label.setObjectName("softHint")
        self.auto_scale_label.setWordWrap(True)
        self.ctrl_layout.addWidget(self.auto_scale_label)
        self.stress_feedback_buttons = []
        self.stress_fixed_feedback_buttons = []
        self.stress_feedback_status_label = None
        self.multirate_status_label = None
        if not self.precision_mode:
            for channel in range(2):
                button = QtWidgets.QPushButton(f"CH{channel}模拟：40 kΩ")
                button.setObjectName(f"stressCh{channel}FeedbackButton")
                button.setProperty("feedbackSelector", PD_FEEDBACK_DEFAULT_SELECTOR)
                button.setMinimumWidth(126)
                button.setToolTip(
                    f"手动选择CH{channel}模拟跨阻：40/20/5/2 kΩ；"
                    "开始应力采集前下发，通过choseA/B IO回读后锁定"
                )
                button.clicked.connect(
                    lambda _checked=False, target=button:
                    self._cycle_stress_feedback_selector(target)
                )
                self.stress_feedback_buttons.append(button)
                self.ctrl_layout.addWidget(button)
            for channel in (2, 3):
                button = QtWidgets.QPushButton(f"CH{channel}模拟：固定2 kΩ")
                button.setObjectName(f"stressCh{channel}FixedFeedbackButton")
                button.setEnabled(False)
                button.setMinimumWidth(126)
                button.setToolTip(
                    f"CH{channel}硬件没有跨阻选择IO，模拟跨阻固定为2 kΩ"
                )
                self.stress_fixed_feedback_buttons.append(button)
                self.ctrl_layout.addWidget(button)
            self.stress_feedback_status_label = QtWidgets.QLabel(
                "应力跨阻：开始时下发并核验IO"
            )
            self.stress_feedback_status_label.setObjectName("metricBadge")
            self.ctrl_layout.addWidget(self.stress_feedback_status_label)
            self.multirate_status_label = QtWidgets.QLabel(
                "自适应扫描：启动时核验采样配置"
            )
            self.multirate_status_label.setObjectName("metricBadge")
            self.multirate_status_label.setToolTip(
                "每帧9个监测点覆盖全部光栅；检测到变化先用13点跟踪，"
                "单一区域连续确认后自动改为11点；45点用于完整形状复核。"
                "缓存点不参与当帧定位或训练。"
            )
            self.ctrl_layout.addWidget(self.multirate_status_label)
            self._update_stress_feedback_controls(confirmed=False)
        self.ctrl_layout.addWidget(self.com_btn)
        self.ctrl_layout.addWidget(self.clear_btn)

        layout.addWidget(self.ctrl_panel, 0, 0)

        self.board_status_panel = QtWidgets.QFrame()
        self.board_status_panel.setObjectName("statusCard")
        self.board_status_layout = FlowLayout(
            self.board_status_panel,
            margin=7,
            horizontal_spacing=9,
            vertical_spacing=7,
        )
        self.pi11210_status_label = QtWidgets.QLabel("PI11210：等待器件状态")
        self.fan_status_label = QtWidgets.QLabel("风扇：等待板温控制状态")
        self.pi11210_status_label.setWordWrap(True)
        self.fan_status_label.setWordWrap(True)
        self.board_status_layout.addWidget(self.pi11210_status_label)
        self.board_status_layout.addWidget(self.stable_reference_btn)
        self.board_status_layout.addWidget(self.finger_3d_btn)
        self.board_status_layout.addWidget(self.fan_status_label)
        layout.addWidget(self.board_status_panel, 1, 0)

        self.plot1 = pg.PlotWidget()
        self.plot1.setObjectName("measurementPlot")

        layout.addWidget(self.plot1, 2, 0)

        self.adc = [deque(maxlen=array_size) for _ in range(4)]
        self.data = [deque(maxlen=array_size) for _ in range(4)]
        self.filts = [np.array(list()) for _ in range(4)]
        self.ori_filts = [np.array(list()) for _ in range(4)]
        self.usdata = [list() for _ in range(4)]
        self.precision_frame_history = deque(maxlen=5)
        self.feedback_gain_masks = [0, 0]
        self.feedback_selectors = [None, None]
        self.digital_display_scales = np.ones(2, dtype=float)
        self.digital_display_scale_locked = np.ones(2, dtype=bool)

        self.waves = [[0 for _ in range(15)] for _ in range(4)]

        self.plot_legend = self.plot1.addLegend()
        self.color_list = ['yellow','#F57171','#8BFA7A','#8FC0FF']
        curve_names = ['CH0 光栅（模拟跨阻手动）', 'CH1 光栅（模拟跨阻手动）',
                       'CH2 光栅（2 kΩ）', 'CH3 直通/波长计（2 kΩ）']
        if self.precision_mode:
            curve_names[2] = None
            curve_names[3] = None
        self.curve1 = self.plot1.plot(pen=self.color_list[0], name=curve_names[0])
        self.curve2 = self.plot1.plot(pen=self.color_list[1], name=curve_names[1])
        self.curve3 = self.plot1.plot(pen=self.color_list[2], name=curve_names[2])
        self.curve4 = self.plot1.plot(pen=self.color_list[3], name=curve_names[3])
        self.stable_reference_curves = []
        for channel in range(4):
            name = "稳定单值参考（白色）" if channel == 0 else None
            reference_curve = self.plot1.plot(
                pen=pg.mkPen("white", width=2), name=name
            )
            reference_curve.setVisible(False)
            reference_curve.setZValue(4)
            self.stable_reference_curves.append(reference_curve)

        for ch_color in ['red','yellow','magenta','cyan']:
            scatter = pg.ScatterPlotItem([], [], pen=pg.mkPen(ch_color), brush=pg.mkBrush(ch_color), symbol='o', size=8)
            scatter.setVisible(False)
            self.plot1.addItem(scatter)
            self.us_scatter_items.append(scatter)

        for ch_color in ['yellow','yellow','yellow','yellow']:
            scatter = pg.ScatterPlotItem([], [], pen=pg.mkPen(255,255,0), brush=pg.mkBrush(255,255,0), symbol='o', size=6)
            scatter.setVisible(False)
            self.plot1.addItem(scatter)
            self.filter_scatter_items.append(scatter)

        self.num_panel = QtWidgets.QWidget()
        self.num_layout = QtWidgets.QGridLayout()
        self.num_layout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.num_panel.setLayout(self.num_layout)

        header_style = "font-weight:bold; font-size:14pt; padding:3pt;"

        self.check_boxs = []

        for r in range(4):
            cb = QtWidgets.QCheckBox()
            cb.setChecked(False)
            cb.stateChanged.connect(lambda state, i=r: self.toggle_line(i, state))
            self.check_boxs.append(cb)
            self.num_layout.addWidget(cb, r+1, 0)

        # 左侧表头（行：1~4）
        role_names = ["CH0 光栅（手动跨阻）", "CH1 光栅（手动跨阻）",
                      "CH2 光栅（2k）", "CH3 直通（2k）"]
        self.channel_role_labels = []
        for r in range(4):
            lab = QtWidgets.QLabel(role_names[r])
            lab.setMinimumWidth(95)
            lab.setAlignment(QtCore.Qt.AlignCenter)
            lab.setStyleSheet(header_style)
            self.channel_role_labels.append(lab)
            self.num_layout.addWidget(lab, r+1, 1)

        # 顶部表头（最多15峰；运行时只显示当前检测到的数量）
        self.peak_header_labels = []
        for c in range(15):
            lab = QtWidgets.QLabel(str(c+1))
            lab.setMinimumWidth(70)
            lab.setAlignment(QtCore.Qt.AlignCenter)
            lab.setStyleSheet(header_style)
            lab.hide()
            self.peak_header_labels.append(lab)
            self.num_layout.addWidget(lab, 0, c+2)

        self.num_labels = [[None for _ in range(15)] for _ in range(4)]

        for r in range(4):
            for c in range(15):
                lab = QtWidgets.QLabel("0")
                lab.setMinimumWidth(70)
                lab.setAlignment(QtCore.Qt.AlignCenter)
                lab.setStyleSheet("font-size:14pt; padding:2pt;")
                lab.hide()
                self.num_labels[r][c] = lab
                self.num_layout.addWidget(lab, r+1, c+2)

        if self.precision_mode:
            for channel in (2, 3):
                self.check_boxs[channel].hide()
                self.channel_role_labels[channel].hide()
                for label in self.num_labels[channel]:
                    label.hide()

        self.num_scroll = QtWidgets.QScrollArea()
        self.num_scroll.setObjectName("peakValueScroll")
        self.num_scroll.setWidgetResizable(True)
        self.num_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.num_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.num_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.num_scroll.setMinimumHeight(130)
        self.num_scroll.setMaximumHeight(250)
        self.num_scroll.setWidget(self.num_panel)
        layout.addWidget(self.num_scroll, 3, 0)

        self.filter_diff_threshold = float(self.diff_threshold_text.text())
        self.filter_diff_indices = [[] for _ in range(4)]
        self.last_peak_focus_indices = None
        # self.diff_send_interval = 50
        # self.update_count = 0
        self.max_diff_points = 100  # 每个通道最多 100 个差异点
        self.diff_indices_log_path = Path(__file__).resolve().parent / "filter_diff_indices.log"

        self.worker = peakWorker()
        self.worker.temp_signal.connect(self.update_temp)
        # self.worker.start()
        
        self.frame_timer = QtCore.QTimer()
        self.frame_timer.timeout.connect(self.process_frame)

        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self.update_plot)

        # FPS 统计：在 process_frame 中每成功处理一帧 +1，fps_timer 每秒结算并重置
        self.fps_frame_count = 0
        self.fps_timer = QtCore.QTimer()
        self.fps_timer.timeout.connect(self._refresh_fps)

        self.set_dac_label(dac_type)

        wavelength_file = Path(__file__).resolve().with_name(
            "wave_const.yaml" if self.precision_mode
            else "stress_wave_const.yaml"
        )
        with open(wavelength_file, 'r', encoding="utf-8") as file:
            yaml_data = yaml.safe_load(file)
            # Both axes are generated from the installed report and may use a
            # different safe point budget after a future calibration.
            self.table_start_index = 0
            self.yaml = yaml_data['Wave_DATA'][self.table_start_index:]
            self.wave_const = [num[0]+num[1]*0.001 for num in self.yaml]
            self.wave_gap_after = self.detect_wave_gaps(self.wave_const)
            self.wave_segments = build_segments(
                len(self.wave_const), self.wave_gap_after
            )
            self.precision_segment_display_scales = np.ones(
                (2, len(self.wave_segments)), dtype=float
            )
            self.precision_segment_scale_locked = np.zeros(
                (2, len(self.wave_segments)), dtype=bool
            )
            self.plot1.setXRange(int(min(self.wave_const)), math.ceil(max(self.wave_const)))

        x_extend = 5
        self.voltage_range = 5
        self.plot1.setYRange(-0.5,2.5)
        self.plot1.getViewBox().setLimits(xMin=self.wave_const[0]-x_extend,xMax=self.wave_const[-1]+x_extend,
                                          yMin=-self.voltage_range/self.voltage_range,yMax=self.voltage_range)
        self.plot1.showGrid(x=True, y=True)

        self.visual_index = 0
        self.visual_y = 0
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('r'))
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('r'))
        self.plot1.addItem(self.vLine, ignoreBounds=True)
        self.plot1.addItem(self.hLine, ignoreBounds=True)

        self.label = pg.TextItem(color='w')
        self.plot1.addItem(self.label)

        self.plot1.proxy = pg.SignalProxy(
            self.plot1.scene().sigMouseMoved,
            rateLimit=60,
            slot=self.mouseMoved
        )

        self.initials_length = 15
        self.peak_interval = 3
        self.peak_threshold = 0.10
        self.peak_min_prominence = float(self.diff_threshold_text.text())
        if self.precision_mode:
            self.peak_trackers = [
                TemperaturePeakTracker(len(self.wave_segments)) for _ in range(4)
            ]
        else:
            # In stress mode a wavelength movement is the measurement itself.
            # Do not wait for a second frame or retain an older cleaner peak.
            self.peak_trackers = [
                AdaptivePeakTracker(
                    len(self.wave_segments), immediate_response=True
                )
                for _ in range(4)
            ]
        self.peak_fits = [[] for _ in range(4)]
        self.display_peak_fits = [
            [None for _ in self.wave_segments] for _ in range(4)
        ]
        self.fbg_channels = self.active_fbg_channels.copy()
        self.direct_monitor_channel = 3
        # A/B tests showed four-sample CH2 reconstruction increased weak-peak
        # outliers, so keep the command available but use the faster two-sample
        # path by default.
        self.selective_ch2_focus_enabled = False
        self.display_normalizer = PeakDisplayNormalizer(channel_count=4)
        self.auto_display_scales = np.ones(4, dtype=float)
        self.frame_generation = 0
        self.new_frame_pending = False

        self.start_wave_index = 0
        self.end_wave_index = len(self.wave_const)

        self.filter_nor = 1

        # 控制是否显示点（filter_visual 和 update_us_point）
        self.show_points = False

        if not self.precision_mode:
            self._initialize_contact_localizer()

    def reload_wavelength_table(self):
        """Reload the report-sized mode axis after the board accepts a new table."""
        worker = getattr(self, "worker", None)
        if worker is not None and worker.isRunning():
            raise RuntimeError(f"{self.mode_name}仍在采集，不能重新载入点表")
        wavelength_file = Path(__file__).resolve().with_name(
            "wave_const.yaml" if self.precision_mode else "stress_wave_const.yaml"
        )
        payload = yaml.safe_load(wavelength_file.read_text(encoding="utf-8")) or {}
        rows = payload.get("Wave_DATA")
        mode_key = "temperature" if self.precision_mode else "stress"
        lower, upper = MODE_POINT_COUNT_LIMITS[mode_key]
        if not isinstance(rows, list) or not lower <= len(rows) <= upper:
            raise ValueError(
                f"{self.mode_name}波长表应为{lower}～{upper}点，实际为"
                f"{len(rows) if isinstance(rows, list) else 0}点"
            )
        waves = [float(row[0]) + float(row[1]) * 0.001 for row in rows]
        if not all(
            math.isfinite(value) and (index == 0 or value > waves[index - 1])
            for index, value in enumerate(waves)
        ):
            raise ValueError(f"{self.mode_name}波长表不是严格递增的有限数列")

        gaps = self.detect_wave_gaps(waves)
        segments = build_segments(len(waves), gaps)
        self.yaml = rows
        self.wave_const = waves
        self.wave_gap_after = gaps
        self.wave_segments = segments
        self.end_wave_index = len(waves)
        self.precision_segment_display_scales = np.ones(
            (2, len(segments)), dtype=float
        )
        self.precision_segment_scale_locked = np.zeros(
            (2, len(segments)), dtype=bool
        )
        if self.precision_mode:
            self.peak_trackers = [
                TemperaturePeakTracker(len(segments)) for _ in range(4)
            ]
        else:
            self.peak_trackers = [
                AdaptivePeakTracker(len(segments), immediate_response=True)
                for _ in range(4)
            ]
        self.peak_fits = [[] for _ in range(4)]
        self.display_peak_fits = [
            [None for _ in segments] for _ in range(4)
        ]
        self.on_clear_chart()
        if not self.precision_mode:
            self._initialize_contact_localizer()
        self.plot1.setXRange(int(min(waves)), math.ceil(max(waves)))
        self.plot1.getViewBox().setLimits(
            xMin=waves[0] - 5,
            xMax=waves[-1] + 5,
            yMin=-1,
            yMax=self.voltage_range,
        )

    def mouseMoved(self, evt):
        pos = evt[0]
        if self.plot1.sceneBoundingRect().contains(pos):

            mousePoint = self.plot1.plotItem.vb.mapSceneToView(pos)
            x = mousePoint.x()
            y = mousePoint.y()

            # 找最近的x索引
            index = np.abs(np.array(self.wave_const)-x).argmin()
            if index>=array_size or index<0 or len(self.adc[0])==0:
                return
            
            self.visual_index = index
            self.visual_y = y
            self.update_crosshair(index, y)

    def update_crosshair(self, index, y):
        if index >= array_size or index < 0 \
        or y > self.voltage_range or y < -self.voltage_range \
        or len(self.adc[0]) == 0:
            return
    
        visible_channels = sorted(self.visible_plot_channels)
        if not visible_channels:
            return
        adc = np.array([self.adc[channel][index] for channel in visible_channels])

        indey = np.abs(adc-y).argmin()

        x_snap = self.wave_const[index]
        y_snap = adc[indey]

        self.vLine.setPos(x_snap)
        self.hLine.setPos(y_snap)

        self.label.setText(f"x={x_snap:.3f}\ny={y_snap:.3f}")
        self.label.setPos(x_snap, y_snap)

    def _clear_stable_reference(self):
        self.stable_reference_values = None
        self.stable_reference_fits = [[] for _ in range(4)]
        self.stable_reference_channels.clear()
        self.stress_reference_offset = None
        self.stress_reference_fast_frames.clear()
        self.stress_reference_calibration_pending = False
        for curve in self.stable_reference_curves:
            curve.setData([], [])
            curve.setVisible(False)
        self.stable_reference_btn.blockSignals(True)
        self.stable_reference_btn.setChecked(False)
        self.stable_reference_btn.setText("显示稳定单值曲线")
        self.stable_reference_btn.blockSignals(False)

    def _initialize_contact_localizer(self):
        """Create the independent CH1 9x5 localizer without touching hardware."""

        self.contact_localizer = None
        self.contact_tracker = None
        self.contact_estimate = None
        self.contact_localizer_error = ""
        if len(self.wave_const) != MULTIRATE_POINT_COUNT:
            self.contact_localizer_error = (
                f"CH1九峰定位需要{MULTIRATE_POINT_COUNT}点，"
                f"当前为{len(self.wave_const)}点"
            )
            return
        try:
            self.contact_localizer = CH1NinePeakLocalizer.from_project_defaults()
            self.contact_tracker = MultiRateContactTracker(
                self.contact_localizer,
                signal_bandwidth_hz=15.0,
            )
        except Exception as exc:
            # Keep the existing spectrum page operational even when an
            # optional certified template is missing or malformed.
            self.contact_localizer_error = str(exc)

    def _set_current_frame_schedule(self, schedule, point_count):
        """Publish one frame's freshness contract for UI/training consumers."""

        point_count = int(point_count)
        self.board_frame_schedule = schedule
        if schedule is None:
            self.acquisition_profile = "MAP"
            self.fresh_point_indices = tuple(range(point_count))
            if point_count == MULTIRATE_POINT_COUNT:
                self.fresh_point_mask = (True,) * MULTIRATE_POINT_COUNT
                self.point_age_frames = (0.0,) * MULTIRATE_POINT_COUNT
                self.sample_offset_us = (None,) * MULTIRATE_POINT_COUNT
            else:
                self.fresh_point_mask = tuple(True for _ in range(point_count))
                self.point_age_frames = tuple(0.0 for _ in range(point_count))
                self.sample_offset_us = tuple(None for _ in range(point_count))
            self.map_age_frames = 0
            self.frame_start_device_ms = None
            self.multirate_schedule_version = None
            self.bandwidth_discontinuity = False
            self.current_frame_is_complete_map = True
            return

        if point_count != MULTIRATE_POINT_COUNT:
            raise ValueError("multirate schedule requires the 45-point stress table")
        fresh_set = frozenset(int(index) for index in schedule.fresh_indices)
        self.acquisition_profile = MULTIRATE_PROFILE_NAMES[schedule.profile]
        self.fresh_point_indices = tuple(sorted(fresh_set))
        self.fresh_point_mask = tuple(
            index in fresh_set for index in range(MULTIRATE_POINT_COUNT)
        )
        self.map_age_frames = int(schedule.map_age_frames)
        self.point_age_frames = tuple(
            0.0 if is_fresh else float(self.map_age_frames)
            for is_fresh in self.fresh_point_mask
        )
        self.sample_offset_us = tuple(schedule.sample_offset_us)
        self.frame_start_device_ms = int(schedule.frame_start_ms)
        self.multirate_schedule_version = int(schedule.version)
        self.bandwidth_discontinuity = bool(
            getattr(schedule, "bandwidth_discontinuity", False)
        )
        self.current_frame_is_complete_map = bool(schedule.is_complete_map)

    def _current_ch1_transimpedance_ohm(self):
        selector = self.feedback_selectors[1]
        if selector is None and len(self.stress_feedback_buttons) >= 2:
            selector = self.stress_feedback_buttons[1].property("feedbackSelector")
        try:
            selector = int(selector)
            return float(PD_FEEDBACK_KOHM_BY_SELECTOR[selector]) * 1000.0
        except (TypeError, ValueError, IndexError):
            return float(PD_FEEDBACK_KOHM_BY_SELECTOR[PD_FEEDBACK_DEFAULT_SELECTOR]) * 1000.0

    def _bind_contact_3d_window(self):
        window = self.finger_3d_window
        if window is None or window is self._bound_contact_3d_window:
            return
        if hasattr(window, "raw_baseline_capture_handler"):
            window.raw_baseline_capture_handler = self.begin_contact_baseline_capture
            window.raw_baseline_clear_handler = self.clear_contact_baseline
        if hasattr(window, "channel_combo"):
            window.channel_combo.setCurrentIndex(1)
            window.channel_combo.setEnabled(False)
            window.channel_combo.setToolTip("CH1上的9个光栅用于手指接触定位")
        self._bound_contact_3d_window = window

    def begin_contact_baseline_capture(self):
        """Arm 20 consecutive stable, full-MAP raw CH1 frames."""

        if self.precision_mode:
            return
        self.contact_baseline_samples = []
        self.contact_baseline_capture_active = True
        self.contact_baseline_rejected_frames = 0
        self.contact_baseline_adc_codes = None
        self.contact_baseline_transimpedance_ohm = None
        self.contact_baseline_capture_transimpedance_ohm = None
        self.contact_estimate = None
        if self.contact_tracker is not None:
            self.contact_tracker.reset()
        window = self.finger_3d_window
        if window is not None and hasattr(window, "set_raw_baseline_progress"):
            window.set_raw_baseline_progress(
                0,
                CONTACT_BASELINE_FRAMES,
                message="请保持手指无应力且静止；只接受稳定的完整MAP帧",
            )

    def clear_contact_baseline(self):
        self.contact_baseline_samples = []
        self.contact_baseline_capture_active = False
        self.contact_baseline_rejected_frames = 0
        self.contact_baseline_adc_codes = None
        self.contact_baseline_transimpedance_ohm = None
        self.contact_baseline_capture_transimpedance_ohm = None
        self.contact_estimate = None
        if self.contact_tracker is not None:
            self.contact_tracker.reset()
        window = self.finger_3d_window
        if window is not None and hasattr(window, "set_raw_baseline_cleared"):
            window.set_raw_baseline_cleared(
                message="原始CH1无应力基准已清除；等待重新采集20帧稳定MAP"
            )

    def _collect_contact_baseline_frame(self, raw_ch1_codes):
        """Consume a candidate baseline frame; return whether capture finished."""

        if not self.contact_baseline_capture_active:
            return False
        if not self.current_frame_is_complete_map:
            window = self.finger_3d_window
            if window is not None and hasattr(window, "set_raw_baseline_progress"):
                window.set_raw_baseline_progress(
                    len(self.contact_baseline_samples),
                    CONTACT_BASELINE_FRAMES,
                    message=(
                        f"基准等待：当前{self.acquisition_profile}不是完整MAP，"
                        "缓存点不计入"
                    ),
                )
            return False
        candidate = np.asarray(raw_ch1_codes, dtype=float).reshape(-1)
        if candidate.size != MULTIRATE_POINT_COUNT or not np.all(np.isfinite(candidate)):
            return False
        if np.any(candidate < 0.0) or np.any(candidate >= PD_ADC_CODE_COUNT):
            return False

        current_tia = self._current_ch1_transimpedance_ohm()
        if self.contact_baseline_capture_transimpedance_ohm is None:
            self.contact_baseline_capture_transimpedance_ohm = current_tia
        elif current_tia != self.contact_baseline_capture_transimpedance_ohm:
            self.contact_baseline_samples = []
            self.contact_baseline_capture_transimpedance_ohm = current_tia
            self.contact_baseline_rejected_frames += 1

        if self.contact_baseline_samples:
            centre = np.median(np.stack(self.contact_baseline_samples), axis=0)
            delta = np.abs(candidate - centre)
            stable = (
                float(np.median(delta)) <= CONTACT_BASELINE_MEDIAN_DELTA_CODES
                and float(np.percentile(delta, 95))
                <= CONTACT_BASELINE_P95_DELTA_CODES
            )
            if not stable:
                # Stability means consecutive frames.  Keep the newest frame as
                # the first candidate of a fresh run instead of silently mixing
                # pre/post-motion spectra.
                self.contact_baseline_samples = [candidate.copy()]
                self.contact_baseline_rejected_frames += 1
            else:
                self.contact_baseline_samples.append(candidate.copy())
        else:
            self.contact_baseline_samples.append(candidate.copy())

        completed = len(self.contact_baseline_samples)
        window = self.finger_3d_window
        if completed >= CONTACT_BASELINE_FRAMES:
            baseline = np.rint(
                np.median(np.stack(self.contact_baseline_samples), axis=0)
            ).astype(np.int16)
            self.contact_baseline_adc_codes = tuple(int(value) for value in baseline)
            self.contact_baseline_transimpedance_ohm = (
                self.contact_baseline_capture_transimpedance_ohm
            )
            self.contact_baseline_samples = []
            self.contact_baseline_capture_active = False
            if self.contact_tracker is not None:
                self.contact_tracker.reset()
            if window is not None and hasattr(window, "set_raw_baseline_progress"):
                window.set_raw_baseline_progress(
                    CONTACT_BASELINE_FRAMES,
                    CONTACT_BASELINE_FRAMES,
                    message=(
                        "CH1原始45点无应力基准已建立；"
                        f"丢弃不稳定段 {self.contact_baseline_rejected_frames} 次"
                    ),
                )
            return True

        if window is not None and hasattr(window, "set_raw_baseline_progress"):
            window.set_raw_baseline_progress(
                completed,
                CONTACT_BASELINE_FRAMES,
                message=(
                    f"正在采集稳定MAP原始基准 {completed}/{CONTACT_BASELINE_FRAMES}"
                ),
            )
        return False

    def _update_contact_pipeline(self, raw_ch1_codes, timestamp_ns):
        """Update raw baseline/localizer from one parsed CH1 composite frame."""

        if self.precision_mode or self.contact_localizer is None:
            return
        if self.contact_baseline_capture_active:
            self._collect_contact_baseline_frame(raw_ch1_codes)
        if self.contact_baseline_adc_codes is None:
            return

        kwargs = {
            "baseline_adc_codes": self.contact_baseline_adc_codes,
            "transimpedance_ohm": self._current_ch1_transimpedance_ohm(),
            "baseline_transimpedance_ohm": self.contact_baseline_transimpedance_ohm,
            "timestamp_ns": int(timestamp_ns),
            "profile": self.acquisition_profile,
            "point_age_frames": self.point_age_frames,
            "map_age_frames": self.map_age_frames,
        }
        if self.board_frame_schedule is not None and any(
            value is not None for value in self.sample_offset_us
        ):
            # The decoder exposes microseconds; the localizer's wire-audit API
            # intentionally accepts raw 2 us ticks.  Cached entries retain the
            # protocol sentinel and therefore can never become fresh by error.
            kwargs["sample_offsets_2us"] = tuple(
                0xFFFF if value is None else int(value) // 2
                for value in self.sample_offset_us
            )
        else:
            kwargs["fresh_point_indices"] = self.fresh_point_indices
        try:
            evidence = self.contact_localizer.extract_spectrum_evidence(
                self.wave_const,
                raw_ch1_codes,
                **kwargs,
            )
            sequence = self.board_frame_sequence
            if sequence is None:
                sequence = self.frame_generation
            try:
                self.contact_estimate = self.contact_tracker.update_evidence(
                    evidence,
                    sequence=sequence,
                )
            except ValueError as exc:
                if "duplicate or reversed" not in str(exc):
                    raise
                # A board reboot or transport change starts a new causal
                # session; never splice its velocity state onto the old one.
                self.contact_tracker.reset()
                self.contact_estimate = self.contact_tracker.update_evidence(
                    evidence,
                    sequence=sequence,
                )
            self.contact_localizer_error = ""
        except Exception as exc:
            self.contact_localizer_error = str(exc)

    def open_finger_3d(self):
        """Open the live fingertip view fed by this page's fitted ADC spectrum."""
        if self.precision_mode:
            return
        if callable(self.finger_3d_open_handler):
            self.finger_3d_open_handler()
            self._update_finger_3d()
            return
        if self.finger_3d_window is None:
            try:
                from mechanical_finger_3d import MechanicalFinger3DWindow

                self.finger_3d_window = MechanicalFinger3DWindow(parent=self)
            except Exception as exc:
                QMessageBox.critical(
                    self,
                    "3D应力界面启动失败",
                    "无法建立3D机械手指界面。请确认PyOpenGL依赖和"
                    f"finger_sensor_layout.yaml配置：\n{exc}",
                )
                self.finger_3d_window = None
                return
        self.finger_3d_window.show()
        self.finger_3d_window.raise_()
        self.finger_3d_window.activateWindow()
        self._update_finger_3d()

    def _update_finger_3d(self):
        if self.precision_mode or self.finger_3d_window is None:
            return
        self._bind_contact_3d_window()
        # The historical Gaussian display is only a full-MAP product.  A
        # reduced frame intentionally leaves it untouched; the causal raw45
        # contact estimate below still updates from the explicitly fresh rows.
        if self.current_frame_is_complete_map:
            self.finger_3d_window.update_from_spectrum(
                self.waves, self.frame_generation
            )
        if (
            self.contact_estimate is not None
            and hasattr(self.finger_3d_window, "update_from_contact_estimate")
        ):
            self.finger_3d_window.update_from_contact_estimate(
                self.contact_estimate,
                self.frame_generation,
            )

    def _update_stable_reference_curves(self):
        show = (
            not self.precision_mode
            and self.stable_reference_btn.isChecked()
            and self.stable_reference_values is not None
        )
        for channel, curve in enumerate(self.stable_reference_curves):
            if (
                not show
                or channel not in self.stable_reference_channels
                or channel not in self.visible_plot_channels
            ):
                curve.setData([], [])
                curve.setVisible(False)
                continue

            display_scale = (
                self.auto_display_scales[channel]
                * self.voltage_scalars[channel]
            )
            scaled = scale_segment_samples(
                self.wave_const,
                self.stable_reference_values[channel],
                self.wave_segments,
                self.stable_reference_fits[channel],
                display_scale,
            )
            plot_x, plot_y = self.curve_data_with_gaps(scaled)
            curve.setData(plot_x, plot_y, connect="finite")
            curve.setVisible(True)

    def on_stable_reference_toggled(self, checked):
        if self.precision_mode:
            return
        if not checked:
            self.stable_reference_btn.setText("显示稳定单值曲线")
            self._update_stable_reference_curves()
            return
        if self.reference_capture_active:
            return
        if self.stable_reference_values is not None:
            self.stable_reference_btn.setText("隐藏稳定单值曲线")
            self._update_stable_reference_curves()
            return

        peak_channels = channels_with_valid_peaks(
            self.peak_fits, self.visible_plot_channels, self.fbg_channels
        )
        if not peak_channels:
            self.stable_reference_btn.setChecked(False)
            QMessageBox.information(
                self, "稳定单值参考", "当前还没有检测到有效光栅峰通道。"
            )
            return
        dac_rows = load_stress_table_dac_rows()
        if len(dac_rows) != len(self.wave_const):
            self.stable_reference_btn.setChecked(False)
            QMessageBox.warning(
                self,
                "稳定单值参考",
                f"DAC表与应力波长表点数不一致：{len(dac_rows)}/{len(self.wave_const)}",
            )
            return
        if not ser_open or not ser.is_open or not self.worker.isRunning():
            self.stable_reference_btn.setChecked(False)
            QMessageBox.information(
                self, "稳定单值参考", "请先启动应力寻峰采集。"
            )
            return

        self.reference_capture_active = True
        self.stable_reference_btn.setEnabled(False)
        self.stable_reference_btn.setText(f"采集稳定单值 0/{len(dac_rows)}")
        self.com_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self.frame_timer.stop()
        self.update_timer.stop()
        self.fps_timer.stop()
        self.worker.running = False
        self.worker.wait(1000)
        frames_queue.clear()
        rx_buffer.clear()
        try:
            ser.reset_input_buffer()
        except Exception:
            pass

        self.stable_reference_worker = StableSingleReferenceWorker(
            ser,
            dac_rows,
            peak_channels,
            self._selected_stress_feedback_selectors(),
            self,
        )
        self.stable_reference_worker.progress.connect(
            self._on_stable_reference_progress
        )
        self.stable_reference_worker.finished.connect(
            self._on_stable_reference_finished
        )
        self.stable_reference_worker.start()

    def _on_stable_reference_progress(self, completed, total):
        self.stable_reference_btn.setText(
            f"采集稳定单值 {completed}/{total}"
        )

    def _restart_stress_reader(self):
        if not ser_open or not ser.is_open:
            return False
        try:
            self._reset_realtime_frame_stats(clear_buffer=True)
            rx_buffer.clear()
            ser.reset_input_buffer()
            self._configure_stress_feedback_hardware()
            self._configure_stress_multirate_hardware()
            self._apply_active_channel_mask(0)
        except Exception as exc:
            self.multirate_armed = False
            QMessageBox.critical(
                self,
                "应力采集恢复失败",
                f"稳定单值参考结束后无法重建自适应会话：\n{exc}",
            )
            return False
        self.worker = peakWorker()
        self.worker.temp_signal.connect(self.update_temp)
        self.worker.start()
        self.frame_timer.start(2)
        self.update_timer.start(10)
        self.fps_timer.start(1000)
        return True

    def _on_stable_reference_finished(self):
        worker = self.stable_reference_worker
        if worker is None:
            return
        result = worker.result
        error_message = worker.error_message
        cancelled = worker.cancelled
        if error_message:
            result = None
        channels = set(worker.channels)
        worker.deleteLater()
        self.stable_reference_worker = None
        self.reference_capture_active = False

        restarted = self._restart_stress_reader()
        self.com_btn.setEnabled(True)
        self.clear_btn.setEnabled(not ser_open)
        self.stable_reference_btn.setEnabled(
            bool(restarted and ser_open and ser.is_open)
        )

        if result is not None:
            self.stable_reference_values = result
            self.stable_reference_channels = channels
            self.stress_reference_offset = None
            self.stress_reference_fast_frames.clear()
            self.stress_reference_calibration_pending = not self.precision_mode
            self.stable_reference_fits = [[] for _ in range(4)]
            for channel in channels:
                fits = fit_channel_segments(
                    self.wave_const,
                    result[channel],
                    self.wave_segments,
                    scan_min_prominence(
                        self.peak_min_prominence, channel, False
                    ),
                    allow_edge_peak=True,
                )
                mask_unconnected_fbg_segments(
                    fits, channel, False, stress_all_segments=True
                )
                self.stable_reference_fits[channel] = fits
            self.stable_reference_btn.setText("隐藏稳定单值曲线")
            self._update_stable_reference_curves()
        else:
            self.stable_reference_btn.setChecked(False)
            if error_message and not cancelled and ser_open:
                QMessageBox.warning(
                    self,
                    "稳定单值参考采集失败",
                    error_message,
                )

    def on_open_changed(self):
        global ser_open
        global ser_cond
        global ser
        global switch_mode_enable

        do_open = False
        do_close = False

        with ser_cond: 
            if ser_open == False:
                do_open = True
                ser_open = True 
                ser_cond.notify_all() 
                
            else: 
                do_close = True
                ser_open = False

        if do_open:
            try:
                ensure_serial_open()
                self._reset_realtime_frame_stats(clear_buffer=True)
                ser.reset_input_buffer()
                if not self.precision_mode:
                    self._configure_stress_feedback_hardware()
                    self._configure_stress_multirate_hardware()
                    self._clear_stable_reference()
                    # Multirate is an explicit CH1-only contract. Hide every
                    # curve until the retained first MAP publishes 0x02.
                    self._apply_active_channel_mask(0)
                else:
                    send_work_mode_command(self.mcu_work_mode)
            except Exception as e:
                print("Serial Error:",e)
                if ser.is_open:
                    try:
                        set_soa_shutter_and_verify(ser)
                    except Exception as shutter_exc:
                        print("SOA shutter verification error:", shutter_exc)
                    release_serial_if_allowed()
                with ser_cond:
                    ser_open = False
                self.multirate_armed = False
                QMessageBox.critical(
                    self, "串口错误", f"{self.mode_name}启动失败:\n{str(e)}"
                )
                for button in self.stress_feedback_buttons:
                    button.setEnabled(True)
                return
            
            switch_mode_enable = False
            self.clear_btn.setEnabled(False)

            self.worker = peakWorker()
            self.worker.temp_signal.connect(self.update_temp)
            self.worker.start()

            self.frame_timer.start(2)
            self.update_timer.start(10)
            self.precision_frame_history.clear()
            for tracker in self.peak_trackers:
                tracker.reset()
            self.digital_display_scales = np.ones(2, dtype=float)
            self.digital_display_scale_locked = np.ones(2, dtype=bool)
            self.precision_segment_display_scales[:] = 1.0
            self.precision_segment_scale_locked[:] = False
            self.fps_timer.start(1000)

            self.com_btn.setText("关闭")
            self.stable_reference_btn.setEnabled(not self.precision_mode)
            for button in self.stress_feedback_buttons:
                button.setEnabled(False)
        elif do_close:
            self.worker.running = False
            self.frame_timer.stop()
            self.update_timer.stop()
            self.fps_timer.stop()
            self.worker.wait(1200)

            shutdown_error = None
            try:
                if ser.is_open:
                    self._stop_optical_session_safely()
            except Exception as e:
                shutdown_error = str(e)
                print("Serial safety shutdown error:", e)
            finally:
                try:
                    release_serial_if_allowed()
                except Exception as e:
                    if shutdown_error is None:
                        shutdown_error = str(e)
                    print("Serial Error:", e)

            switch_mode_enable = True
            self.clear_btn.setEnabled(True)

            self.com_btn.setText("打开")
            self.stable_reference_btn.setEnabled(False)
            self.stable_reference_btn.setChecked(False)
            for button in self.stress_feedback_buttons:
                button.setEnabled(True)
            if not self.precision_mode:
                self._update_stress_feedback_controls(confirmed=False)
            if shutdown_error is not None:
                QMessageBox.critical(
                    self,
                    "停止安全核验失败",
                    shutdown_error,
                )

    def on_clear_chart(self):
        self._clear_stable_reference()
        if not self.precision_mode:
            self.clear_contact_baseline()
        curves = [self.curve1, self.curve2, self.curve3, self.curve4]
        for idx in range(4):
            curves[idx].setData([])
            self.data[idx].clear()
            self.adc[idx].clear()
            self.temperature = 0
            self.us_scatter_items[idx].setData([], [])
            self.filter_scatter_items[idx].setData([], [])

            for l in self.peaks_lines[idx]:
                l.setVisible(False)
                self.plot1.removeItem(l)
            self.peaks_lines[idx].clear()
            for label in self.num_labels[idx]:
                label.setText("0")
        self.temperature_text.setText("0")
        self.fan_status_label.setText("风扇：等待板温控制状态")
        for tracker in self.peak_trackers:
            tracker.reset()
        self.display_peak_fits = [
            [None for _ in self.wave_segments] for _ in range(4)
        ]
        self.display_normalizer.reset()
        self.auto_display_scales = np.ones(4, dtype=float)
        self.feedback_gain_masks = [0, 0]
        self.feedback_selectors = [None, None]
        self.digital_display_scales = np.ones(2, dtype=float)
        self.digital_display_scale_locked = np.ones(2, dtype=bool)
        self.precision_segment_display_scales[:] = 1.0
        self.precision_segment_scale_locked[:] = False
        self._update_gain_annotations()
        self._update_stable_reference_curves()
        self.precision_frame_history.clear()
        self.new_frame_pending = False
        self._reset_realtime_frame_stats(clear_buffer=True)

    def _reset_realtime_frame_stats(self, *, clear_buffer):
        if clear_buffer:
            frames_queue.clear(reset_counters=True)
        self.process_down = True
        self.new_frame_pending = False
        self.pending_frame_received_ns = None
        self.frame_age_samples_ms.clear()
        self.last_frame_received_monotonic_ns = None
        self.last_frame_age_ms = math.nan
        self.board_frame_sequence = None
        self.board_acquisition_duration_us = None
        self.board_table_crc32 = None
        self.board_boot_id = None
        self.board_uptime_ms = None
        self._set_current_frame_schedule(None, len(self.wave_const))
        self.bandwidth_discontinuity = False
        self.last_complete_map_channels = None
        self.contact_estimate = None
        if self.contact_tracker is not None:
            self.contact_tracker.reset()
        self.board_sequence_tracker.reset()
        self.fps_frame_count = 0
        self.fps_window_started_ns = time.perf_counter_ns()
        self.fps_label.setText(
            "刷新: 0.0 Hz"
            if self.precision_mode
            else "压力场 0.0 Hz · 丢帧 0 · 帧龄 —"
        )
        if not self.precision_mode:
            self.fps_label.setToolTip(
                "压力场刷新率按完成拟合和3D更新的帧计数；"
                "丢帧是电脑为保持实时性而略过的旧完整帧；"
                "帧龄从完整帧进入电脑解析器算到本次显示更新完成。"
            )

    def on_threshold_changed(self):
        try:
            value = float(self.diff_threshold_text.text())
            self.filter_diff_threshold = value
            self.peak_min_prominence = max(value, 0.001)
            self.new_frame_pending = True
        except Exception:
            pass

    def _refresh_fps(self):
        now_ns = time.perf_counter_ns()
        elapsed_s = max(
            (now_ns - self.fps_window_started_ns) / 1_000_000_000.0,
            1e-6,
        )
        refresh_hz = self.fps_frame_count / elapsed_s
        self.fps_window_started_ns = now_ns
        board_acquisition_text = (
            f"{self.board_acquisition_duration_us / 1000.0:.1f} ms"
            if self.board_acquisition_duration_us is not None
            else "—"
        )
        board_sequence_stats = self.board_sequence_tracker.stats()
        if self.precision_mode:
            self.fps_label.setText(
                f"刷新: {refresh_hz:.1f} Hz · 板采 {board_acquisition_text}"
            )
        else:
            stats = frames_queue.stats()
            if self.frame_age_samples_ms:
                frame_age_p95_ms = float(
                    np.percentile(self.frame_age_samples_ms, 95)
                )
                frame_age_text = (
                    f"{self.last_frame_age_ms:.1f}/{frame_age_p95_ms:.1f} ms"
                )
            else:
                frame_age_text = "—"
            self.fps_label.setText(
                f"压力场 {refresh_hz:.1f} Hz · 丢帧 {stats.dropped} "
                f"· {self.acquisition_profile} {len(self.fresh_point_indices)}/"
                f"{len(self.fresh_point_mask)} · 板采 {board_acquisition_text} "
                f"· 帧龄 {frame_age_text}"
                + (" · 带宽空档" if self.bandwidth_discontinuity else "")
            )
            self.fps_label.setToolTip(
                f"完整帧接收 {stats.received}；为保持实时性主动跳过 "
                f"{stats.consumer_skipped}；队列溢出 {stats.overflow_dropped}；"
                f"待处理 {stats.pending}。\n"
                f"板端帧序号 {self.board_frame_sequence if self.board_frame_sequence is not None else '—'}；"
                f"序号缺帧 {board_sequence_stats.missing}；"
                f"重复 {board_sequence_stats.duplicates}；"
                f"重启 {board_sequence_stats.resets}。\n"
                f"当前调度 {self.acquisition_profile}；新鲜点 "
                f"{len(self.fresh_point_indices)}/{len(self.fresh_point_mask)}；"
                f"MAP年龄 {self.map_age_frames} 帧；"
                f"带宽空档标记 {int(self.bandwidth_discontinuity)}；"
                f"板端起始 {self.frame_start_device_ms if self.frame_start_device_ms is not None else '—'} ms。\n"
                "帧龄显示最近一帧/近期 P95，起点是完整帧进入电脑解析器，"
                "终点是拟合和 3D 更新完成。"
            )
        self.fps_frame_count = 0

    def _on_digital_gain_changed(self, channel, value):
        """Apply one explicit per-channel display/fitting multiplier."""
        if 0 <= int(channel) < len(self.voltage_scalars):
            self.voltage_scalars[int(channel)] = float(value)
            self.new_frame_pending = True
            self._update_gain_annotations()

    def _selected_stress_feedback_selectors(self):
        if self.precision_mode or len(self.stress_feedback_buttons) != 2:
            return (PD_FEEDBACK_DEFAULT_SELECTOR, PD_FEEDBACK_DEFAULT_SELECTOR)
        return tuple(
            int(button.property("feedbackSelector"))
            for button in self.stress_feedback_buttons
        )

    def _cycle_stress_feedback_selector(self, button):
        if self.precision_mode or ser_open:
            return
        cycle = (1, 3, 2, 0)  # 40 -> 20 -> 5 -> 2 kOhm
        current = int(button.property("feedbackSelector"))
        button.setProperty(
            "feedbackSelector", cycle[(cycle.index(current) + 1) % len(cycle)]
        )
        self._update_stress_feedback_controls(confirmed=False)

    def _update_stress_feedback_controls(self, confirmed=False):
        if self.precision_mode:
            return
        selectors = self._selected_stress_feedback_selectors()
        resistances = tuple(PD_FEEDBACK_KOHM_BY_SELECTOR[item] for item in selectors)
        for channel, button in enumerate(self.stress_feedback_buttons):
            button.setText(f"CH{channel}模拟：{resistances[channel]} kΩ")
        if self.stress_feedback_status_label is not None:
            self.stress_feedback_status_label.setText(
                (
                    f"应力跨阻：IO已核验并锁定 "
                    f"CH0 {resistances[0]} kΩ / CH1 {resistances[1]} kΩ"
                )
                if confirmed
                else (
                    f"应力跨阻：待启动核验 "
                    f"CH0 {resistances[0]} kΩ / CH1 {resistances[1]} kΩ"
                )
            )

    def _configure_stress_feedback_hardware(self):
        """Enter EXTRA, arm fixed stress feedback and verify physical IO."""
        send_work_mode_command(2)
        time.sleep(0.12)
        actual = set_stress_feedback_and_verify(
            ser, self._selected_stress_feedback_selectors()
        )
        self.feedback_selectors = [int(actual[0]), int(actual[1])]
        self.feedback_gain_masks = [
            1 if int(selector) == 3 else 0 for selector in actual
        ]
        self._update_stress_feedback_controls(confirmed=True)
        return actual

    def _configure_stress_multirate_hardware(self):
        """Arm the current contract and require its first true MAP."""

        schedule = enter_stress_multirate_and_verify_first_map(
            ser,
            self.multirate_configuration,
            schedule_version=getattr(self, "requested_multirate_version", CURRENT_SCHEDULE_VERSION),
        )
        self.multirate_armed = True
        if self.multirate_status_label is not None:
            if self.multirate_configuration.map_period_frames == REFERENCE_MAP_PERIOD:
                self.multirate_status_label.setText(
                    "完整谱参考：每帧45点实采 · 非稳定单值教师 · 不作15 Hz验收"
                )
                return schedule
            map_policy = (
                "连续测量（仅首帧MAP）"
                if self.multirate_configuration.map_period_frames == 0
                else f"静稳{self.multirate_configuration.map_period_frames}帧复核MAP"
            )
            self.multirate_status_label.setText(
                f"自适应扫描：v{schedule.version}已核验 · "
                f"全域9点 / 单区11点 / 宽域13点 / {map_policy}"
            )
        return schedule

    def _stop_optical_session_safely(self):
        """Disarm volatile scheduling and finish with an exact SOA shutter."""

        errors = []
        if not self.precision_mode and self.multirate_armed:
            try:
                ser.write(build_work_mode_command(2))
                try:
                    ser.flush()
                except (AttributeError, serial.SerialException):
                    pass
                time.sleep(AP_MODE_SWITCH_SETTLE_S)
                set_stress_multirate_and_verify(
                    ser,
                    MultirateConfiguration(enabled=False),
                )
                self.multirate_armed = False
            except Exception as exc:
                errors.append(f"自适应调度解除失败：{exc}")
        try:
            set_soa_shutter_and_verify(ser)
        except Exception as exc:
            errors.append(f"SOA安全关光未确认：{exc}")
        if self.multirate_status_label is not None:
            self.multirate_status_label.setText(
                "自适应扫描：已解除，SOA已关光"
                if not errors
                else "自适应扫描：停止安全核验失败"
            )
        if errors:
            raise RuntimeError("\n".join(errors))

    def find_filter_diff_indices(self, adc_vec, fil_vec):
        if len(adc_vec) == 0 or len(fil_vec) == 0:
            return []
        diff = np.abs(np.array(adc_vec) - fil_vec)
        indices = np.nonzero(diff > self.filter_diff_threshold)[0]
        return indices.tolist()[:self.max_diff_points]

    def send_filter_diff_indices(self):
        if not ser.is_open:
            return

        channel_points = [ids[:self.max_diff_points] for ids in self.filter_diff_indices]

        command_frame = bytearray(tx_size)
        command_frame[0] = 0xFF
        command_frame[1] = 0xFF
        command_frame[2] = 0x01
        command_frame[3] = 0x04

        # 按通道打包：len0 [idx_hi idx_lo]*len0, len1 [idx_hi idx_lo]*len1, ... len3 [idx_hi idx_lo]*len3
        pos = 4
        for idx_list in channel_points:
            command_frame[pos] = len(idx_list) & 0xFF
            pos += 1
            for idx in idx_list:
                if pos + 1 >= tx_size:
                    break
                command_frame[pos] = (idx >> 8) & 0xFF
                command_frame[pos + 1] = idx & 0xFF
                pos += 2

        try:
            ser.write(bytes(command_frame))
        except Exception as e:
            print("Serial send peak focus indices error:", e)

    def update_peak_focus_indices(self):
        """Ask the MCU to oversample only the low-gain CH2 peak neighborhoods."""
        channel_points = [[], [], [], []]
        focus = set()
        for segment, fit in zip(self.wave_segments, self.peak_fits[2]):
            if not self.selective_ch2_focus_enabled:
                break
            if not fit.valid or not np.isfinite(fit.center_nm):
                continue
            section_indices = np.arange(segment.start, segment.stop)
            nearest = int(
                section_indices[
                    np.argmin(
                        np.abs(
                            np.asarray(self.wave_const)[section_indices]
                            - fit.center_nm
                        )
                    )
                ]
            )
            for index in range(nearest - 2, nearest + 3):
                if segment.start <= index < segment.stop:
                    focus.add(index)
        channel_points[2] = sorted(focus)
        focus_signature = tuple(tuple(points) for points in channel_points)
        if focus_signature == self.last_peak_focus_indices:
            return
        self.filter_diff_indices = channel_points
        self.send_filter_diff_indices()
        self.last_peak_focus_indices = focus_signature

    def send_wave_range(self):
        if not ser.is_open:
            return

        command_frame = bytearray(tx_size)
        command_frame[0] = 0xFF
        command_frame[1] = 0xFF
        command_frame[2] = 0x01
        command_frame[3] = 0x05

        command_frame[4] = (self.start_wave_index >> 8) & 0xFF
        command_frame[5] = self.start_wave_index & 0xFF
        command_frame[6] = (self.end_wave_index >> 8) & 0xFF
        command_frame[7] = self.end_wave_index & 0xFF

        try:
            ser.write(bytes(command_frame))
        except Exception as e:
            print("Serial send wave range error:", e)

    def log_filter_diff_indices(self, channel_points):
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with self.diff_indices_log_path.open("a", encoding="utf-8") as f:
                f.write(f"{timestamp} threshold={self.filter_diff_threshold}\n")
                for ch, idx_list in enumerate(channel_points):
                    if not idx_list:
                        continue
                    index_str = ",".join(str(self.wave_const[idx]) for idx in idx_list)
                    write_log = f"CH{ch}: {index_str}\n"
                    f.write(write_log)
                f.write("\n")
        except Exception as e:
            print("Log write error:", e)

    def process_frame(self):
        raw = b""
        try:
            if self.process_down is False or self.new_frame_pending:
                # print("no valid frame")
                return
            queued_frame = frames_queue.take(
                latest_only=not self.precision_mode
            )
            if queued_frame is None:
                return
            raw = queued_frame.payload
            self.pending_frame_received_ns = queued_frame.received_ns
            # Parsing mutates the channel buffers. An invalid/partial parse
            # must not leave their former receipt timestamp looking valid.
            self.last_frame_received_monotonic_ns = None
            self.process_down = False
            single_size = 8
            previous_waves = [list(channel) for channel in self.waves]

            # print(raw)
            if raw[0]!=0xEE or raw[1]!=0xEE:
                print("pack head error")
                self.pending_frame_received_ns = None
                self.process_down = True
                # print(raw)
                return 
            
            global array_size
            _array_size = (raw[2]<<8)+raw[3]
            if not _array_size == array_size:
                array_size = _array_size
                # print(array_size)

                self.adc = [deque(maxlen=array_size) for _ in range(4)]
                self.data = [deque(maxlen=array_size) for _ in range(4)]
                self.precision_frame_history.clear()

            self.usdata = [list() for _ in range(4)]

            com_index = 0

            valid_data_count = self.end_wave_index - self.start_wave_index
            # print(valid_data_count, array_size)
            for i in range(4, array_size*8+4, single_size):
                com_input = raw[i:i+single_size]

                ch1 = (com_input[0]<<8)+com_input[1]
                # ust1 = com_input[2]
                ch2 = (com_input[2]<<8)+com_input[3]
                # ust2 = com_input[5]
                ch3 = (com_input[4]<<8)+com_input[5]
                # ust3 = com_input[8]
                ch4 = (com_input[6]<<8)+com_input[7]
                # ust4 = com_input[11]

                v1 = ch1*PD_ADC_REFERENCE_V/PD_ADC_CODE_COUNT
                v2 = ch2*PD_ADC_REFERENCE_V/PD_ADC_CODE_COUNT
                v3 = ch3*PD_ADC_REFERENCE_V/PD_ADC_CODE_COUNT
                v4 = ch4*PD_ADC_REFERENCE_V/PD_ADC_CODE_COUNT

                # print(f"{ch1},{ch2},{ch3},{ch4}")

                self.adc[0].append(v1)
                self.adc[1].append(v2)
                self.adc[2].append(v3)
                self.adc[3].append(v4)

                self.data[0].append(ch1)
                self.data[1].append(ch2)
                self.data[2].append(ch3)
                self.data[3].append(ch4)

                # if ust1 ==1:
                #     self.usdata[0].append(com_index)
                # if ust2 ==1:
                #     self.usdata[1].append(com_index)
                # if ust3 ==1:
                #     self.usdata[2].append(com_index)
                # if ust4 ==1:
                #     self.usdata[3].append(com_index)

                com_index+=1

            if raw[array_size*8+4]!=0xAB:
                print("wave head error")
                print(hex(raw[array_size*8+4]))
                self.pending_frame_received_ns = None
                self.process_down = True
                return
            parsed_waves = [[0 for _ in range(15)] for _ in range(4)]
            frame_count = array_size*8+4
            data_len = 0
            for i in range(0, 4):
                frame_count += 1

                data_len = raw[frame_count]

                for j in range(data_len):
                    frame_count+=1
                    int_high = raw[frame_count]
                    frame_count+=1
                    int_low = raw[frame_count]
                    frame_count+=1
                    dec_high = raw[frame_count]
                    frame_count+=1
                    dec_low = raw[frame_count]
                    get_wave = (int_high<<8)+int_low+((dec_high<<8)+dec_low)*0.001

                    parsed_waves[i][j] = get_wave
            
            frame_count+=1
            temp_int_high = raw[frame_count]
            frame_count+=1
            temp_int_low = raw[frame_count]
            frame_count+=1
            temp_dec_high = raw[frame_count]
            frame_count+=1
            temp_dec_low = raw[frame_count]
            temperature = (temp_int_high<<8)+temp_int_low+((temp_dec_high<<8)+temp_dec_low)*0.0001
            self.update_temp(temperature)

            # New firmware appends two big-endian uint16 masks (nine sections)
            # before FF EF.  Retain support for the former two-byte frames.
            trailer_start = frame_count + 1
            trailer_end = len(raw) - 2
            trailer_length = trailer_end - trailer_start
            previous_gain_masks = tuple(self.feedback_gain_masks)
            previous_feedback_selectors = tuple(self.feedback_selectors)
            self.board_frame_sequence = None
            self.board_acquisition_duration_us = None
            self.board_table_crc32 = None
            self.board_boot_id = None
            self.board_uptime_ms = None
            self._set_current_frame_schedule(None, _array_size)
            if trailer_length >= 4:
                self.feedback_gain_masks[0] = (
                    (int(raw[trailer_start]) << 8)
                    | int(raw[trailer_start + 1])
                )
                self.feedback_gain_masks[1] = (
                    (int(raw[trailer_start + 2]) << 8)
                    | int(raw[trailer_start + 3])
                )
            elif trailer_length >= 2:
                self.feedback_gain_masks[0] = int(raw[trailer_start])
                self.feedback_gain_masks[1] = int(raw[trailer_start + 1])
            else:
                self.feedback_gain_masks[0] = 0
                self.feedback_gain_masks[1] = 0

            # Board-status extension B5 4D, version 1.  Its first 28 bytes keep
            # the established power/thermal/channel/PI11210/version layout;
            # timing-aware firmware appends sequence and acquisition_us at
            # bytes 28..35.  Old 28-byte recordings therefore stay readable.
            extension_start = trailer_start + 4
            if (
                trailer_length >= 24
                and raw[extension_start] == 0xB5
                and raw[extension_start + 1] == 0x4D
                and raw[extension_start + 2] == 1
            ):
                extension_length = int(raw[extension_start + 3])
                extension_end = extension_start + 4 + extension_length
                if extension_length < 16 or extension_end > trailer_end:
                    raise ValueError("invalid board-status extension length")
                ext = raw[extension_start + 4 : extension_end]
                pd_flags = int(ext[0])
                pd_protocol = int(ext[1])
                pd_voltage_mv = (int(ext[2]) << 8) | int(ext[3])
                pd_current_ma = (int(ext[4]) << 8) | int(ext[5])
                pd_power_mw = (
                    (int(ext[6]) << 24)
                    | (int(ext[7]) << 16)
                    | (int(ext[8]) << 8)
                    | int(ext[9])
                )
                fan_rpm = (int(ext[10]) << 8) | int(ext[11])
                fan_duty_permille = (int(ext[12]) << 8) | int(ext[13])
                thermal_flags = (int(ext[14]) << 8) | int(ext[15])
                if len(ext) >= 17:
                    self._apply_active_channel_mask(int(ext[16]))
                pi_flags = None
                pi_raw_status = None
                pi_i2c_errors = None
                if len(ext) >= 22:
                    pi_flags = int(ext[17])
                    pi_raw_status = (int(ext[18]) << 8) | int(ext[19])
                    pi_i2c_errors = (int(ext[20]) << 8) | int(ext[21])
                if len(ext) >= 24:
                    selectors = (int(ext[22]), int(ext[23]))
                    if all(
                        0 <= value < len(PD_FEEDBACK_KOHM_BY_SELECTOR)
                        for value in selectors
                    ):
                        self.feedback_selectors[:] = selectors
                firmware_version = None
                if len(ext) >= 28:
                    firmware_version = (
                        (int(ext[24]) << 24)
                        | (int(ext[25]) << 16)
                        | (int(ext[26]) << 8)
                        | int(ext[27])
                    )
                board_timing = decode_board_frame_timing(ext)
                if board_timing is not None:
                    self.board_frame_sequence = board_timing.sequence
                    self.board_acquisition_duration_us = (
                        board_timing.acquisition_duration_us
                    )
                    self.board_sequence_tracker.observe(board_timing.sequence)
                runtime_identity = decode_board_runtime_identity(ext)
                if runtime_identity is not None:
                    self.board_table_crc32 = runtime_identity.table_crc32
                    self.board_boot_id = runtime_identity.boot_id
                    self.board_uptime_ms = runtime_identity.uptime_ms
                self._set_current_frame_schedule(
                    decode_board_frame_schedule(ext),
                    _array_size,
                )
                self._update_board_status(
                    pd_flags,
                    pd_protocol,
                    pd_voltage_mv,
                    pd_current_ma,
                    pd_power_mw,
                    fan_rpm,
                    fan_duty_permille,
                    thermal_flags,
                    pi_flags,
                    pi_raw_status,
                    pi_i2c_errors,
                    firmware_version,
                )

            self.waves = (
                parsed_waves
                if self.current_frame_is_complete_map
                else previous_waves
            )

            if self.precision_mode:
                # Do not median-fuse spectra captured with different hardware
                # feedback resistors.  A newly learned 20 kOhm section starts
                # a clean five-frame history from its first consistent frame.
                if (
                    tuple(self.feedback_gain_masks) != previous_gain_masks
                    or tuple(self.feedback_selectors)
                    != previous_feedback_selectors
                ):
                    self.precision_frame_history.clear()
                precision_frame = np.asarray(
                    [list(channel_values) for channel_values in self.adc],
                    dtype=float,
                )
                if precision_frame.shape == (4, array_size):
                    fused = precision_median_fuse(
                        self.precision_frame_history,
                        precision_frame,
                        range(4),
                    )
                    self.adc = [
                        deque(fused[channel].tolist(), maxlen=array_size)
                        for channel in range(4)
                    ]

            # Preserve the serial parser's receipt time with this generation;
            # update_plot clears only the transient pending-frame timestamp.
            # Never replace an old queued frame's receipt with UI render time.
            self.last_frame_received_monotonic_ns = self.pending_frame_received_ns
            self.frame_generation += 1
            self.new_frame_pending = True
            self.process_down = True
        except Exception as e:
            print(e)
            print(len(raw))
            self.pending_frame_received_ns = None
            self.process_down = True
            return

    def update_temp(self, _temperature):
        self.temperature = _temperature
        self.temperature_text.setText(f"{_temperature}")

    def _apply_active_channel_mask(self, channel_mask):
        """Lock stress plotting to the channels selected by the MCU discovery frame."""
        allowed_mask = 0x03 if self.precision_mode else 0x0F
        channel_mask = int(channel_mask) & allowed_mask
        if channel_mask == self.active_channel_mask:
            return

        previous_channels = set(self.visible_plot_channels)
        active_channels = {
            channel for channel in range(4)
            if channel_mask & (1 << channel)
        }
        self.active_channel_mask = channel_mask
        self.visible_plot_channels = active_channels
        self.active_fbg_channels = active_channels & self.default_fbg_channels
        self.fbg_channels = self.active_fbg_channels.copy()

        curves = [self.curve1, self.curve2, self.curve3, self.curve4]
        for channel in range(4):
            enabled = channel in active_channels
            self.check_boxs[channel].setEnabled(enabled)
            if not enabled:
                curves[channel].setData([], [])
                self.filts[channel] = np.array([], dtype=float)
                self.ori_filts[channel] = np.array([], dtype=float)
                self.us_scatter_items[channel].setData([], [])
                self.filter_scatter_items[channel].setData([], [])
                self.waves[channel] = [0.0] * self.initials_length
                self.peak_fits[channel] = []
                self.display_peak_fits[channel] = [
                    None for _ in self.wave_segments
                ]
                for line in self.peaks_lines[channel]:
                    line.setVisible(False)
                for label in self.num_labels[channel]:
                    label.setText("0")
            elif channel not in previous_channels:
                self.peak_trackers[channel].reset()

        self._update_gain_annotations()
        self._update_stable_reference_curves()

    def _update_board_status(
        self,
        pd_flags,
        pd_protocol,
        pd_voltage_mv,
        pd_current_ma,
        pd_power_mw,
        fan_rpm,
        fan_duty_permille,
        thermal_flags,
        pi_flags=None,
        pi_raw_status=None,
        pi_i2c_errors=None,
        firmware_version=None,
    ):
        duty = min(max(fan_duty_permille / 10.0, 0.0), 100.0)
        if thermal_flags & 0x0004:
            control = "温度传感器故障，安全满速"
        elif thermal_flags & 0x0002:
            control = "≥33℃，连续 PID 增速"
        else:
            control = "33℃ 连续 PID"
        tach = "，转速反馈异常" if thermal_flags & 0x0008 else ""
        self.fan_status_label.setText(
            f"风扇：{duty:.1f}% / {fan_rpm} RPM，{control}{tach}"
        )
        firmware_text = ""
        if firmware_version is not None:
            version = int(firmware_version)
            firmware_text = (
                f"固件 v{(version >> 16) & 0xFF}."
                f"{(version >> 8) & 0xFF}.{version & 0xFF} · "
            )
        if pi_flags is None:
            self.pi11210_status_label.setText(
                f"{firmware_text}PI11210：旧固件未上报器件状态"
            )
        elif pi_flags & 0x07 == 0x07:
            revision = (int(pi_raw_status) >> 4) & 0x0F
            mode = "CLR零电流关光" if pi_flags & 0x80 else "正向出光"
            warnings = []
            if pi_flags & 0x20:
                warnings.append("过温保护")
            elif pi_flags & 0x10:
                warnings.append("过温")
            elif pi_flags & 0x08:
                warnings.append("高温预警")
            if pi_flags & 0x40:
                warnings.append("I²C已恢复")
            warning_text = "，" + "、".join(warnings) if warnings else ""
            self.pi11210_status_label.setText(
                f"{firmware_text}PI11210：在线/ID正确 Rev{revision}，{mode}，"
                f"I²C错误{int(pi_i2c_errors)}次{warning_text}"
            )
        else:
            self.pi11210_status_label.setText(
                f"{firmware_text}PI11210：离线或器件ID不符"
                f"（状态0x{int(pi_raw_status or 0):04X}）"
            )

    def set_dac_label(self, dac_type):
        self.dac_label.setText("DAC: PI11210")

    def _feedback_resistance_text(self, channel):
        mask = int(self.feedback_gain_masks[channel])
        if not self.precision_mode:
            selector = self.feedback_selectors[channel]
            if selector is None:
                selector = self._selected_stress_feedback_selectors()[channel]
            return f"{PD_FEEDBACK_KOHM_BY_SELECTOR[int(selector)]} kΩ"
        selector = self.feedback_selectors[channel]
        if selector is not None:
            return f"{PD_FEEDBACK_KOHM_BY_SELECTOR[int(selector)]} kΩ"
        relevant = (1 << len(self.wave_segments)) - 1
        active = mask & relevant
        if active == 0:
            return "40 kΩ"
        if active == relevant:
            return "20 kΩ"
        return "40/20 kΩ"

    def _update_precision_display_gains(self, fits_by_channel):
        """Acquire one immutable digital gain for each valid CH0/CH1 section."""
        if not self.precision_mode:
            return
        for channel in range(2):
            channel_mask = int(self.feedback_gain_masks[channel])
            selector = self.feedback_selectors[channel]
            fits = fits_by_channel[channel]
            for segment_index, fit in enumerate(fits):
                if self.precision_segment_scale_locked[channel, segment_index]:
                    continue
                if fit is None or not fit.valid:
                    continue
                hardware_is_maximum = (
                    int(selector) == PD_FEEDBACK_DEFAULT_SELECTOR
                    if selector is not None
                    else not bool(channel_mask & (1 << segment_index))
                )
                self.precision_segment_display_scales[
                    channel, segment_index
                ] = required_display_gain(
                    fit.display_baseline_v,
                    fit.display_amplitude_v,
                    hardware_is_maximum=hardware_is_maximum,
                    target_peak_v=1.05,
                    maximum_gain=64.0,
                    minimum_signal_v=0.003,
                )
                self.precision_segment_scale_locked[channel, segment_index] = True

            locked_scales = self.precision_segment_display_scales[channel][
                self.precision_segment_scale_locked[channel]
            ]
            self.digital_display_scales[channel] = (
                float(np.max(locked_scales)) if locked_scales.size else 1.0
            )

    def _update_gain_annotations(self):
        curves = [self.curve1, self.curve2, self.curve3, self.curve4]
        resistances = [self._feedback_resistance_text(0), self._feedback_resistance_text(1)]
        total_scales = np.asarray(self.auto_display_scales, dtype=float) * np.asarray(
            self.voltage_scalars, dtype=float
        )
        precision_gain_text = [None, None]
        if self.precision_mode:
            for channel in range(2):
                locked = self.precision_segment_scale_locked[channel]
                scales = (
                    self.precision_segment_display_scales[channel][locked]
                    * self.voltage_scalars[channel]
                )
                if not scales.size:
                    precision_gain_text[channel] = "等待有效峰"
                else:
                    minimum = float(np.min(scales))
                    maximum = float(np.max(scales))
                    precision_gain_text[channel] = (
                        f"×{maximum:.2f}"
                        if abs(maximum - minimum) < 0.005
                        else f"×{minimum:.2f}~{maximum:.2f}"
                    )
        names = [
            f"CH0 光栅（模拟{resistances[0]}，数字"
            f"{precision_gain_text[0] if self.precision_mode else f'×{total_scales[0]:.2f}'}）",
            f"CH1 光栅（模拟{resistances[1]}，数字"
            f"{precision_gain_text[1] if self.precision_mode else f'×{total_scales[1]:.2f}'}）",
            f"CH2 光栅（模拟2 kΩ，数字×{total_scales[2]:.2f}）",
            f"CH3 直通/波长计（模拟2 kΩ，数字×{total_scales[3]:.2f}）",
        ]
        for channel in range(4):
            if channel not in self.visible_plot_channels:
                names[channel] = f"CH{channel}（无波形，采集关闭）"
        if hasattr(self.plot_legend, "getLabel"):
            for curve, name in zip(curves, names):
                label = self.plot_legend.getLabel(curve)
                if label is not None:
                    label.setText(name)
        self.channel_role_labels[0].setText(
            f"CH0 光栅\n模拟{resistances[0]}  数字"
            f"{precision_gain_text[0] if self.precision_mode else f'×{total_scales[0]:.2f}'}"
        )
        self.channel_role_labels[1].setText(
            f"CH1 光栅\n模拟{resistances[1]}  数字"
            f"{precision_gain_text[1] if self.precision_mode else f'×{total_scales[1]:.2f}'}"
        )
        self.channel_role_labels[2].setText(
            f"CH2 光栅\n模拟2 kΩ  数字×{total_scales[2]:.2f}"
        )
        self.channel_role_labels[3].setText(
            f"CH3 直通/波长计\n模拟2 kΩ  数字×{total_scales[3]:.2f}"
        )

        if self.precision_mode:
            summaries = []
            for channel in range(2):
                locked = self.precision_segment_scale_locked[channel]
                scales = (
                    self.precision_segment_display_scales[channel][locked]
                    * self.voltage_scalars[channel]
                )
                if not scales.size:
                    summaries.append("等待有效峰")
                else:
                    minimum = float(np.min(scales))
                    maximum = float(np.max(scales))
                    gain_text = precision_gain_text[channel]
                    summaries.append(
                        f"{gain_text}，已锁定{int(np.count_nonzero(locked))}/"
                        f"{len(self.wave_segments)}段"
                    )
            self.auto_scale_label.setText(
                f"模拟跨阻：CH0 {resistances[0]}，CH1 {resistances[1]}；"
                f"数字显示：CH0 {summaries[0]}，CH1 {summaries[1]}；"
                "CH2/CH3采集关闭"
            )
        else:
            self.auto_scale_label.setText(
                f"模拟跨阻（手动）：CH0 {resistances[0]}，CH1 {resistances[1]}，"
                "CH2/CH3 固定2 kΩ；"
                f"数字显示（手动）：CH0 ×{total_scales[0]:.2f}，"
                f"CH1 ×{total_scales[1]:.2f}，"
                f"CH2 ×{total_scales[2]:.2f}，CH3 ×{total_scales[3]:.2f}"
            )

        if not self.precision_mode:
            active_text = "/".join(
                f"CH{channel}" for channel in sorted(self.visible_plot_channels)
            ) or "无"
            self.auto_scale_label.setText(
                f"{self.auto_scale_label.text()}   自动采集：{active_text}"
            )
            for channel in range(4):
                if channel not in self.visible_plot_channels:
                    self.channel_role_labels[channel].setText(
                        f"CH{channel}\n无波形，采集关闭"
                    )

    def update_us_point(self):
        if not self.show_points:
            for item in self.us_scatter_items:
                item.setData([], [])
                item.setVisible(False)
            return

        for ch in range(4):
            if ch not in self.visible_plot_channels:
                self.us_scatter_items[ch].setData([], [])
                self.us_scatter_items[ch].setVisible(False)
                continue
            if len(self.filts[ch]) != len(self.wave_const):
                self.us_scatter_items[ch].setData([], [])
                self.us_scatter_items[ch].setVisible(False)
                continue

            self.us_scatter_items[ch].setData(
                x=self.wave_const,
                y=self.filts[ch],
            )
            self.us_scatter_items[ch].setVisible(True)

    def curve_data_with_gaps(self, y_data):
        """Insert NaN separators so YAML wavelength gaps are not connected."""
        plot_x = []
        plot_y = []
        for index, (wave, value) in enumerate(zip(self.wave_const, y_data)):
            plot_x.append(wave)
            plot_y.append(value)
            if index in self.wave_gap_after:
                plot_x.append(np.nan)
                plot_y.append(np.nan)
        return np.asarray(plot_x, dtype=float), np.asarray(plot_y, dtype=float)

    def update_plot(self):
        if self.process_down == False or not self.new_frame_pending:
            return
        if not len(self.wave_const) == len(self.adc[0]):
            self.new_frame_pending = False
            self.pending_frame_received_ns = None
            return
        self.process_down = False
        self.new_frame_pending = False
        try:
            raw_channels = [np.asarray(values, dtype=float) for values in self.adc]
            if not self.precision_mode and len(self.data[1]) == MULTIRATE_POINT_COUNT:
                contact_timestamp_ns = (
                    time.monotonic_ns()
                    if self.pending_frame_received_ns is None
                    else int(self.pending_frame_received_ns)
                )
                self._update_contact_pipeline(
                    tuple(int(value) for value in self.data[1]),
                    contact_timestamp_ns,
                )
            if (
                not self.precision_mode
                and self.stress_reference_calibration_pending
                and self.stable_reference_values is not None
                and self.current_frame_is_complete_map
            ):
                frame_matrix = np.stack(raw_channels, axis=0)
                if frame_matrix.shape == self.stable_reference_values.shape:
                    self.stress_reference_fast_frames.append(frame_matrix.copy())
                    if (
                        len(self.stress_reference_fast_frames)
                        >= STRESS_REFERENCE_FAST_CALIBRATION_FRAMES
                    ):
                        self.stress_reference_offset = (
                            compute_fixed_stress_reference_offset(
                                tuple(self.stress_reference_fast_frames),
                                self.stable_reference_values,
                                self.stable_reference_channels,
                            )
                        )
                        self.stress_reference_calibration_pending = False

            analysis_channels = [values.copy() for values in raw_channels]
            if (
                not self.precision_mode
                and self.stress_reference_offset is not None
                and self.stress_reference_offset.shape
                    == (len(analysis_channels), len(self.wave_const))
            ):
                for channel in self.stable_reference_channels:
                    if 0 <= channel < len(analysis_channels):
                        analysis_channels[channel] = np.clip(
                            analysis_channels[channel]
                            - self.stress_reference_offset[channel],
                            0.0,
                            PD_ADC_REFERENCE_V,
                        )

            if self.current_frame_is_complete_map:
                self.last_complete_map_channels = [
                    values.copy() for values in analysis_channels
                ]
                for channel, values in enumerate(analysis_channels):
                    self.ori_filts[channel] = values
                    if channel not in self.visible_plot_channels:
                        self.peak_fits[channel] = []
                        self.waves[channel] = [0.0] * self.initials_length
                        self.display_peak_fits[channel] = [
                            None for _ in self.wave_segments
                        ]
                        continue
                    if self.precision_mode and channel not in self.active_fbg_channels:
                        self.peak_fits[channel] = []
                        self.waves[channel] = [0.0] * self.initials_length
                        continue
                    channel_threshold = scan_min_prominence(
                        self.peak_min_prominence, channel, self.precision_mode
                    )
                    self.peak_fits[channel] = fit_channel_segments(
                        self.wave_const,
                        values,
                        self.wave_segments,
                        channel_threshold,
                        allow_edge_peak=not self.precision_mode,
                    )
                    mask_unconnected_fbg_segments(
                        self.peak_fits[channel],
                        channel,
                        self.precision_mode,
                        stress_all_segments=not self.precision_mode,
                    )
                    if channel in self.fbg_channels:
                        tracker = self.peak_trackers[channel]
                        tracked = tracker.update(self.peak_fits[channel])
                        for segment_index, (fit, center) in enumerate(
                            zip(self.peak_fits[channel], tracked)
                        ):
                            if (
                                fit.valid
                                and tracker.measurement_accepted[segment_index]
                                and np.isfinite(center)
                            ):
                                self.display_peak_fits[channel][segment_index] = replace(
                                    fit, center_nm=float(center)
                                )
                    else:
                        for fit in self.peak_fits[channel]:
                            fit.valid = False
                        tracked = np.full(len(self.wave_segments), np.nan)
                    peaks = [float(value) if np.isfinite(value) else 0.0
                             for value in tracked]
                    self.waves[channel] = (
                        peaks[:self.initials_length]
                        + [0.0] * max(0, self.initials_length - len(peaks))
                    )

                self.update_peak_focus_indices()
                display_analysis_channels = analysis_channels
            else:
                # Reduced frames contain a mix of fresh samples and historical
                # cache.  They are valid input to the explicit-freshness CH1
                # localizer above, but never to the legacy full-profile fit.
                display_analysis_channels = (
                    self.last_complete_map_channels
                    if self.last_complete_map_channels is not None
                    else analysis_channels
                )
                for channel, values in enumerate(display_analysis_channels):
                    self.ori_filts[channel] = values
            if self.precision_mode:
                self.auto_display_scales = np.ones(4, dtype=float)
                self.digital_display_scale_locked[:] = True
                self._update_precision_display_gains(self.peak_fits)
            else:
                # Stress mode has no automatic digital normalization.  The
                # four operator-entered multipliers are the only display gain.
                self.auto_display_scales = np.ones(4, dtype=float)
                self.digital_display_scales[:] = 1.0
                self.digital_display_scale_locked[:] = True
            self._update_gain_annotations()

            curves = [self.curve1, self.curve2, self.curve3, self.curve4]
            for channel, curve in enumerate(curves):
                if channel not in self.visible_plot_channels:
                    self.filts[channel] = np.array([], dtype=float)
                    curve.setData([], [])
                    self.us_scatter_items[channel].setData([], [])
                    self.filter_scatter_items[channel].setData([], [])
                    continue
                auto_scale = self.auto_display_scales[channel]
                if self.precision_mode and channel < 2:
                    display_scale = (
                        self.precision_segment_display_scales[channel]
                        * self.voltage_scalars[channel]
                    )
                else:
                    display_scale = auto_scale * self.voltage_scalars[channel]
                self.filts[channel] = scale_segment_samples(
                    self.wave_const,
                    display_analysis_channels[channel],
                    self.wave_segments,
                    self.peak_fits[channel],
                    display_scale,
                )
                plot_x, plot_y = rounded_display_curve(
                    self.wave_const,
                    display_analysis_channels[channel],
                    self.wave_segments,
                    [
                        display_fit if display_fit is not None else raw_fit
                        for display_fit, raw_fit in zip(
                            self.display_peak_fits[channel],
                            self.peak_fits[channel],
                        )
                    ],
                    display_scale=display_scale,
                    fitted=bool(self.filter_nor),
                    normalize_segment_heights=channel >= 2,
                )
                curve.setData(plot_x, plot_y, connect='finite')

            self._update_stable_reference_curves()

            self.update_us_point()
            self.update_crosshair(self.visual_index, self.visual_y)

            self._update_peak_value_table()
            for channel in range(4):
                if self.visible_lines[channel]:
                    self.cal_peaks_line(channel)
                else:
                    for line in self.peaks_lines[channel]:
                        line.setVisible(False)
            self._update_finger_3d()
            if self.pending_frame_received_ns is not None:
                self.fps_frame_count += 1
                self.last_frame_age_ms = max(
                    0.0,
                    (
                        time.monotonic_ns()
                        - self.pending_frame_received_ns
                    )
                    / 1_000_000.0,
                )
                self.frame_age_samples_ms.append(self.last_frame_age_ms)
        finally:
            self.pending_frame_received_ns = None
            self.process_down = True

    def _update_peak_value_table(self):
        """Compact the table to the peaks retained by the live tracker.

        ``self.waves`` stays indexed by wavelength segment for 3D mapping and
        tracking.  A recently valid peak therefore survives a few malformed
        frames instead of making the displayed count flicker between seven and
        eight.  The tracker removes it after a sustained real loss.  Only the
        presentation is compacted, so an empty section never creates a zero
        column or changes a grating's identity.
        """
        displayed = []
        for channel in range(4):
            channel_peaks = []
            if channel in self.visible_plot_channels and channel in self.fbg_channels:
                for segment_index in range(len(self.wave_segments)):
                    if (
                        segment_index < len(self.waves[channel])
                        and self.waves[channel][segment_index]
                    ):
                        channel_peaks.append(float(self.waves[channel][segment_index]))
            displayed.append(channel_peaks[:self.initials_length])

        visible_column_count = max((len(peaks) for peaks in displayed), default=0)
        for column, header in enumerate(self.peak_header_labels):
            header.setVisible(column < visible_column_count)
        for channel, peaks in enumerate(displayed):
            for column, label in enumerate(self.num_labels[channel]):
                if column < len(peaks):
                    label.setText(f"{peaks[column]:.4f}")
                    label.show()
                else:
                    label.setText("")
                    label.hide()
        self.num_panel.updateGeometry()

    def calculate_peaks(self, _data, i):
        if len(_data) == 0:
            return [0]*self.initials_length
        return self.find_peaks(_data, i)
        
    def cal_peaks_line(self, i):
        if len(self.data[i]) == 0:
            return
        for l in self.peaks_lines[i]:
            l.setVisible(False)

        while len(self.peaks_lines[i]) < self.initials_length:
            vline = pg.InfiniteLine(
                angle=90,
                pen=pg.mkPen(self.color_list[i], width=1,
                             style=QtCore.Qt.DashLine),
            )
            self.plot1.addItem(vline)
            self.peaks_lines[i].append(vline)

        for idx,p in enumerate(self.waves[i]):
            if p==0:
                self.peaks_lines[i][idx].setVisible(False)
            else:
                self.peaks_lines[i][idx].setPos(p)
                self.peaks_lines[i][idx].setVisible(True)

    def toggle_line(self, i, state):
        visible = (state == QtCore.Qt.Checked)
        self.visible_lines[i] = visible
        if visible:
            self.cal_peaks_line(i)
        else:
            for line in self.peaks_lines[i]:
                line.setVisible(False)

    def find_initial(self, data_vec, adc_length, adc_index):
        initials = [0]*self.initials_length
        ma = max(data_vec)
        mi = min(data_vec)
        gap = ma-mi
        data_norvec = list(np.array(data_vec)-mi)
        mean = sum(data_norvec)
        
        # if gap*100 < 10 or 100*mean>10*adc_length*gap:
        if gap*100<10:
            # print(f"{adc_index}:", mean/(adc_length*gap))
            return initials
        it = 0
        i = 1
        # for i in range(1,adc_length-2):
        while i<(adc_length-2):
            if it>=self.initials_length: break
            start = i-self.peak_interval if i-self.peak_interval>=0 else 0
            end = i+self.peak_interval if i+self.peak_interval<adc_length else adc_length

            # print(data_norvec)
            front = list(data_norvec)[start:i]
            back = list(data_norvec)[i+1:end]
            # print(front)
            # print(back)
            if data_norvec[i]>=max(front) and data_norvec[i]>=max(back) and data_norvec[i]>self.peak_threshold:
                # if 10*(data_norvec[i]-data_norvec[i-1])>gap*5 or 10*(data_norvec[i]-data_norvec[i-1])>gap*5:
                # #    self.adc[adc_index][i] = 0
                #    continue
                # print(data_norvec[i])
                # print(i)
                initials[it] = i
                i+=self.peak_interval
                it+=1
            i+=1
        # print(initials)
        return initials

    def find_peaks(self, data_vec, adc_index):
        # initials = [0]*self.initials_length
        peaks_vec = [0]*self.initials_length
        adc_length = array_size

        initials = self.find_initial(data_vec, adc_length, adc_index)

        self.start_wave_index = initials[0]-2*self.peak_interval if initials[0]-2*self.peak_interval>=0 else 0
        
        for i in range(self.initials_length):
            if initials[i]==0:
                if i > 0:
                    self.end_wave_index = initials[i-1]+2*self.peak_interval \
                        if initials[i-1]+2*self.peak_interval<len(self.wave_const) else len(self.wave_const)
                else:
                    self.end_wave_index = len(self.wave_const)
                break
            start = initials[i]-self.peak_interval if initials[i]-self.peak_interval>=0 else 0
            end = initials[i]+self.peak_interval if initials[i]+self.peak_interval<adc_length else adc_length
            sumy=0
            sumxy=0

            peak_max = max(data_vec[start:end])
            threshold = peak_max * 0.5

            for j in range(start, end):
                if data_vec[j] < threshold:
                    continue

                sumxy+=data_vec[j]*self.wave_const[j]
                sumy+=data_vec[j]
            peaks_vec[i] = sumxy/sumy
            # print(peaks_vec[i])

        # print(peaks_vec)
        # print(self.start_wave_index, self.end_wave_index)
        return peaks_vec
    
    def adc_filter(self, adc_vec, channel):
        if len(adc_vec) == 0:
            return np.array([])

        arr = np.asarray(adc_vec, dtype=float)

        #--------------------------#
        # 滤波
        if self.filter_nor:
            y = medfilt(arr, kernel_size=3)
            y = savgol_filter(y, window_length=3, polyorder=2)
            y = np.array(y)
        # 不滤波
        else:
            y = np.array(adc_vec)
        #--------------------------#

        if self.show_points:
            self.filter_visual(channel, arr, y)
        else:
            self.filter_scatter_items[channel].setData([], [])

        return y
    
    def filter_visual(self, channel, adc_vec, fil_vec):
        x_data = []
        y_data = []
        for i,(fi,adc) in enumerate(zip(fil_vec, adc_vec)):
            if abs(fi-adc) >= self.filter_diff_threshold:
                x_data.append(self.wave_const[i])
                y_data.append(adc)

        self.filter_scatter_items[channel].setData(x=x_data, y=y_data)
        self.filter_scatter_items[channel].setVisible(self.show_points and bool(x_data))

    def on_toggle_filter(self, checked):
        self.filter_nor = 1 if checked else 0
        self.new_frame_pending = True
        self.update_plot()

    def on_toggle_points(self, checked):
        self.show_points = checked
        for item in self.filter_scatter_items + self.us_scatter_items:
            item.setVisible(False)
            item.setData([], [])
        self.new_frame_pending = True
        self.update_plot()
        

class ap6150bWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        page_layout = QtWidgets.QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        self.page_scroll = ResponsiveScrollArea()
        self.page_scroll.setObjectName("apPageScroll")
        self.page_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.page_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        content = ScrollContentWidget()
        content.setObjectName("apPageContent")
        layout = QtWidgets.QGridLayout(content)
        self.page_scroll.setWidget(content)
        page_layout.addWidget(self.page_scroll)

        self.ctrl_panel = QtWidgets.QWidget()
        self.ctrl_layout = QtWidgets.QVBoxLayout()
        self.ctrl_layout.setContentsMargins(5,5,5,5)
        self.ctrl_layout.setSpacing(10)
        self.ctrl_panel.setLayout(self.ctrl_layout)

        self.fileText_edit = QLineEdit()
        self.fileText_edit.setPlaceholderText("选择波长数据文件")
        self.fileText_edit.setMinimumWidth(180)
        self.file_button = QPushButton("...")
        self.file_path = ""

        self.dac_label = QtWidgets.QLabel("DAC: PI11210")
        self.dac_label.setMinimumWidth(100)
        self.dac_label.setAlignment(QtCore.Qt.AlignCenter)
        self.gain_label = QtWidgets.QLabel(
            "模拟跨阻：CH0/CH1 40 kΩ（×20），CH2/CH3/PDT/PDR 2 kΩ（×1）"
            " · 数字显示：×1.00（未缩放）"
        )
        self.gain_label.setObjectName("metricBadge")
        self.gain_label.setMinimumWidth(380)
        self.gain_label.setWordWrap(True)

        lab_temperature = QtWidgets.QLabel("温度(℃):")
        self.temperature_text = QtWidgets.QLineEdit("0")
        self.temperature_text.setReadOnly(True)
        self.temperature_text.setMinimumWidth(82)
        self.temperature = 0

        self.com_btn = QtWidgets.QPushButton("开始")
        self.com_btn.setMinimumWidth(100)

        self.clear_btn = QtWidgets.QPushButton("清空日志")
        self.clear_btn.setMinimumWidth(100)

        self.file_button.clicked.connect(self.select_file)
        self.com_btn.clicked.connect(self.ap_thread)
        self.clear_btn.clicked.connect(self.on_clear)

        file_row = QtWidgets.QWidget()
        file_layout = QtWidgets.QHBoxLayout(file_row)
        file_layout.setContentsMargins(0, 0, 0, 0)
        file_layout.setSpacing(8)
        file_layout.addWidget(self.fileText_edit, 1)
        file_layout.addWidget(self.file_button)
        self.ctrl_layout.addWidget(file_row)

        actions_row = QtWidgets.QWidget()
        actions_layout = FlowLayout(
            actions_row, horizontal_spacing=9, vertical_spacing=8
        )
        actions_layout.addWidget(self.dac_label)
        actions_layout.addWidget(self.gain_label)
        actions_layout.addWidget(
            compact_field(lab_temperature.text(), self.temperature_text)
        )
        actions_layout.addWidget(self.com_btn)
        actions_layout.addWidget(self.clear_btn)
        self.ctrl_layout.addWidget(actions_row)

        layout.addWidget(self.ctrl_panel, 0, 0)

        self.printf_area = LogWidget()
        layout.addWidget(self.printf_area)

        self.worker =  APWorker(self.file_path)
        self.worker.log_signal.connect(self.printf_area.log)
        self.worker.temp_signal.connect(self.update_temp)

        self.set_dac_label(dac_type)

    def select_file(self):
        self.file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择Excel文件",
            get_desktop_path(),
            "ALL Files (*);;Text Files (*.xlsx)"
        )
        self.fileText_edit.setText(self.file_path)

    def ap_thread(self):
        global ap_open
        global switch_mode_enable

        if self.worker is not None and self.worker.isRunning():
            self.com_btn.setEnabled(False)
            self.com_btn.setText("正在关闭…")
            self.printf_area.log("正在取消当前点并收尾，窗口仍可操作")
            self.worker.stop()
            return

        if ap_open or not switch_mode_enable:
            QMessageBox.warning(self, "警告", "请先停止当前页面的工作流")
            return

        self.file_path = self.fileText_edit.text().strip()
        if not self.file_path:
            QMessageBox.warning(self, "未选择文件", "请先选择扫波长Excel文件")
            return

        try:
            ensure_serial_open()
            ser.reset_input_buffer()
            send_work_mode_command(1)
        except Exception as e:
            print("Serial Error:", str(e))
            if ser.is_open:
                release_serial_if_allowed()
            with ap_cond:
                ap_open = False
            QMessageBox.critical(self, "串口错误", f"扫波长模式启动失败:\n{str(e)}")
            return

        with ap_cond:
            ap_open = True
        switch_mode_enable = False

        self.worker = APWorker(self.file_path)
        self.worker.log_signal.connect(self.printf_area.log)
        self.worker.temp_signal.connect(self.update_temp)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

        self.com_btn.setEnabled(True)
        self.com_btn.setText("关闭")

    def _on_worker_finished(self):
        global ap_open
        global switch_mode_enable

        with ap_cond:
            ap_open = False
        switch_mode_enable = True
        try:
            release_serial_if_allowed()
        except Exception as exc:
            self.printf_area.log(f"串口收尾失败：{exc}", "ERROR")
        self.com_btn.setEnabled(True)
        self.com_btn.setText("开始")

    def shutdown(self, wait_ms=9000):
        """Bounded non-GUI fallback for callers that cannot defer closing.

        The normal windows use the asynchronous ``finished`` path.  Nine
        seconds covers both possible three-second VISA queries, serial-reader
        exit, and the bounded SOA-shutter verification.
        """
        if self.worker is None or not self.worker.isRunning():
            return True
        self.worker.stop()
        return bool(self.worker.wait(int(wait_ms)))

    def on_clear(self):
        self.printf_area.clear()

    def update_temp(self, _temperature):
        self.temperature = _temperature
        self.temperature_text.setText(f"{_temperature}")

    def set_dac_label(self, dac_type):
        self.dac_label.setText("DAC: PI11210")


class extraWindow(QtWidgets.QWidget):
    temp_signal = pyqtSignal(float)
    rt_signal = pyqtSignal(int,int)
    volite_signal = pyqtSignal(int,int, int, int)
    error_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.confirmed_feedback_selectors = None
        self.observed_feedback_selectors = None
        self.feedback_selection_dirty = False
        self.fullband_points = ()
        self.fullband_table_error = None
        try:
            self.fullband_points = load_fullband_accuracy_table()
        except (OSError, ValueError, csv.Error) as exc:
            self.fullband_table_error = str(exc)
        self.fullband_dac_rows = tuple(
            tuple(point.codes) for point in self.fullband_points
        )
        self.fullband_dac_row_index = {
            row: index for index, row in enumerate(self.fullband_dac_rows)
        }
        self.auto_wavelength_point = None
        self._auto_output_fullband_index = None

        page_layout = QtWidgets.QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        self.page_scroll = ResponsiveScrollArea()
        self.page_scroll.setObjectName("singleValueScroll")
        self.page_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.page_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        content = ScrollContentWidget()
        content.setObjectName("singleValueContent")
        layout = QtWidgets.QVBoxLayout(content)
        layout.setSpacing(15)
        layout.setContentsMargins(20,20,20,20)
        self.page_scroll.setWidget(content)
        page_layout.addWidget(self.page_scroll)

        top_layout = QtWidgets.QVBoxLayout()

        wavelength_box = QtWidgets.QGroupBox("按波长自动输出（2001点单模标定表）")
        wavelength_layout = QtWidgets.QGridLayout(wavelength_box)
        wavelength_layout.setHorizontalSpacing(10)
        wavelength_layout.setVerticalSpacing(8)

        wavelength_title = QtWidgets.QLabel("目标波长")
        self.auto_wavelength_spin = QtWidgets.QDoubleSpinBox()
        self.auto_wavelength_spin.setDecimals(6)
        self.auto_wavelength_spin.setSingleStep(0.020)
        self.auto_wavelength_spin.setSuffix(" nm")
        self.auto_wavelength_spin.setKeyboardTracking(False)
        self.auto_wavelength_spin.setMinimumWidth(190)
        self.auto_wavelength_spin.setMinimumHeight(40)
        self.auto_wavelength_spin.setRange(
            float(FULLBAND_ACCURACY_START_NM),
            float(
                FULLBAND_ACCURACY_START_NM
                + FULLBAND_ACCURACY_STEP_NM
                * (FULLBAND_ACCURACY_POINT_COUNT - 1)
            ),
        )
        self.auto_wavelength_spin.setValue(float(FULLBAND_ACCURACY_START_NM))

        self.auto_wavelength_btn = QtWidgets.QPushButton("自动匹配并输出")
        self.auto_wavelength_btn.setProperty("role", "primary")
        self.auto_wavelength_btn.setMinimumHeight(40)
        self.auto_wavelength_btn.setMinimumWidth(150)
        self.auto_wavelength_btn.clicked.connect(self.output_by_wavelength)

        self.auto_wavelength_match_label = QtWidgets.QLabel()
        self.auto_wavelength_match_label.setObjectName("metricBadge")
        self.auto_wavelength_match_label.setWordWrap(True)
        self.auto_wavelength_note = QtWidgets.QLabel(
            "输入完成后会自动填入最接近的波长计实测单模点；"
            "不对五路DAC码做线性插值，避免跨调谐分支或双峰。"
        )
        self.auto_wavelength_note.setWordWrap(True)

        wavelength_layout.addWidget(wavelength_title, 0, 0)
        wavelength_layout.addWidget(self.auto_wavelength_spin, 0, 1)
        wavelength_layout.addWidget(self.auto_wavelength_btn, 0, 2)
        wavelength_layout.setColumnStretch(3, 1)
        wavelength_layout.addWidget(self.auto_wavelength_match_label, 1, 0, 1, 4)
        wavelength_layout.addWidget(self.auto_wavelength_note, 2, 0, 1, 4)
        self.auto_wavelength_spin.editingFinished.connect(
            self.preview_wavelength_match
        )

        head_box = QtWidgets.QGroupBox("从Excel复制五路原始DAC码")
        head_layout = QtWidgets.QHBoxLayout()

        self.excel_text = QtWidgets.QLineEdit()
        self.excel_text.setPlaceholderText(
            "从标定表复制：GAIN码\tSOA码\tPHASE码\tWAVE_A码\tWAVE_B码"
        )
        self.excel_text.setMinimumHeight(40)
        self.excel_text.setMinimumWidth(180)

        self.write_btn = QtWidgets.QPushButton("写入")
        self.write_btn.clicked.connect(self.write_data_MCU)
        self.write_btn.setMinimumHeight(40)
        self.write_btn.setMinimumWidth(120)

        head_layout.addWidget(self.excel_text)
        head_layout.addWidget(self.write_btn)

        head_box.setLayout(head_layout)

        button_box = QtWidgets.QGroupBox("工作流控制")
        button_layout = QtWidgets.QVBoxLayout()

        self.dac_label = QtWidgets.QLabel("DAC: PI11210")
        self.dac_label.setAlignment(QtCore.Qt.AlignHCenter)
        self.dac_label.setMinimumHeight(25)
        self.gain_label = QtWidgets.QLabel(
            "模拟跨阻：CH0/CH1 40 kΩ（×20），CH2/CH3/PDT/PDR 2 kΩ（×1）"
            " · 数字显示：×1.00（未缩放）"
        )
        self.gain_label.setObjectName("metricBadge")
        self.gain_label.setAlignment(QtCore.Qt.AlignCenter)
        self.gain_label.setWordWrap(True)

        feedback_controls = QtWidgets.QHBoxLayout()
        feedback_controls.setSpacing(8)
        feedback_controls.addWidget(QtWidgets.QLabel("手动模拟跨阻"))
        self.ch0_feedback_btn = QtWidgets.QPushButton("CH0：40 kΩ")
        self.ch1_feedback_btn = QtWidgets.QPushButton("CH1：40 kΩ")
        for button, channel in (
            (self.ch0_feedback_btn, "CH0"),
            (self.ch1_feedback_btn, "CH1"),
        ):
            button.setProperty("feedbackSelector", PD_FEEDBACK_DEFAULT_SELECTOR)
            button.setMinimumHeight(36)
            button.setMinimumWidth(116)
            button.setToolTip(
                f"点击循环切换{channel}模拟跨阻：40/20/5/2 kΩ；"
                "每次均由单片机回读choseA/B实际电平确认。"
            )
            feedback_controls.addWidget(button)
        feedback_controls.addStretch(1)
        self.feedback_status_label = QtWidgets.QLabel("跨阻：待手动设置或启动时核验")
        self.feedback_status_label.setObjectName("metricBadge")
        self.feedback_status_label.setWordWrap(True)
        self.ch0_feedback_btn.clicked.connect(
            lambda _checked=False: self._cycle_feedback_selector(
                self.ch0_feedback_btn
            )
        )
        self.ch1_feedback_btn.clicked.connect(
            lambda _checked=False: self._cycle_feedback_selector(
                self.ch1_feedback_btn
            )
        )

        self.monitor_btn = QtWidgets.QPushButton("开始")
        self.monitor_btn.clicked.connect(self.on_work)
        self.monitor_btn.setMinimumHeight(40)
        self.monitor_btn.setMinimumWidth(120)

        button_layout.addWidget(self.dac_label)
        button_layout.addWidget(self.gain_label)
        button_layout.addLayout(feedback_controls)
        button_layout.addWidget(self.feedback_status_label)
        button_layout.addWidget(self.monitor_btn)

        button_box.setLayout(button_layout)

        top_layout.addWidget(wavelength_box)
        top_layout.addWidget(head_box)
        top_layout.addWidget(button_box)

        layout.addLayout(top_layout)

        self.body_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.body_splitter.setChildrenCollapsible(False)

        param_box = QtWidgets.QGroupBox("参数")
        form_left = QtWidgets.QFormLayout()
        configure_form_layout(form_left)
        form_left.setVerticalSpacing(18)

        def create_param(name, maximum_code):
            edit = QtWidgets.QLineEdit("0")
            edit.setMinimumWidth(100)
            edit.setValidator(QtGui.QIntValidator(0, maximum_code, edit))
            edit.setToolTip(f"直接输入十进制DAC码，允许范围：0～{maximum_code}")

            dac_name = QtWidgets.QLabel("待写入/回读DAC码:")

            dac_text = QtWidgets.QLabel(f"{0}")
            dac_text.setMinimumWidth(80)
            dac_text.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

            state = QtWidgets.QLabel("● 未写入")
            state.setStyleSheet("color:#d97706")

            widget = QtWidgets.QWidget()
            h = QtWidgets.QGridLayout(widget)
            h.setContentsMargins(0, 0, 0, 0)
            h.setHorizontalSpacing(7)
            h.setVerticalSpacing(4)
            h.addWidget(edit, 0, 0, 1, 3)
            h.addWidget(dac_name, 1, 0)
            h.addWidget(dac_text, 1, 1)
            h.addWidget(state, 1, 2)
            h.setColumnStretch(0, 1)

            form_left.addRow(name, widget)

            return edit, state, dac_text
        
        (self.gain_text,self.gain_state,self.gain_dac_text),self.gain = create_param(
            "GAIN(DAC码)", PI11210_GAIN_MAX_CODE
        ),0
        self.soa_mode_combo = QtWidgets.QComboBox()
        self.soa_mode_combo.addItem("正向出光（0～135 mA）", PI11210_SOA_SOURCE_MODE)
        self.soa_mode_combo.addItem(
            "负值语义：关光（安全零电流）", PI11210_SOA_SHUTTER_MODE
        )
        self.soa_mode_combo.setToolTip(
            "负值只表示关光；为满足3 V反压上限，固件使用CLR零电流Gate"
        )
        form_left.addRow("SOA工作状态", self.soa_mode_combo)
        (self.soa_text,self.soa_state,self.soa_dac_text),self.soa = create_param(
            "SOA(DAC码)", PI11210_SOA_SOURCE_MAX_CODE
        ),0
        (self.phase_text,self.phase_state,self.phase_dac_text),self.phase = create_param(
            "PHASE(DAC码)", PI11210_PHASE_MAX_CODE
        ),0
        (self.wavelena_text,self.wavelena_state,self.wavelena_dac_text),self.wavelena = create_param(
            "WAVE_A(DAC码)", PI11210_WAVELENGTH_MAX_CODE
        ),0
        (self.wavelenb_text,self.wavelenb_state,self.wavelenb_dac_text),self.wavelenb = create_param(
            "WAVE_B(DAC码)", PI11210_WAVELENGTH_MAX_CODE
        ),0

        param_box.setLayout(form_left)

        monitor_box = QtWidgets.QGroupBox("监测")
        form_right = QtWidgets.QFormLayout()
        configure_form_layout(form_right)
        form_right.setVerticalSpacing(20)

        def read_only(name):
            e = QtWidgets.QLineEdit(f"{0:.4f}")

            e.setReadOnly(True)
            e.setMinimumWidth(100)

            form_right.addRow(name, e)

            return e

        self.temperature_text,self.temperature = read_only("温度(℃)"),0

        def create_monitor(name):
            e = QtWidgets.QLineEdit("0")
            e.setReadOnly(True)
            e.setMinimumWidth(100)

            v_name = QtWidgets.QLabel("ADC电压:")

            v_text = QtWidgets.QLabel(f"{0:>5}V")

            widget = QtWidgets.QWidget()
            h = QtWidgets.QGridLayout(widget)
            h.setContentsMargins(0, 0, 0, 0)
            h.setHorizontalSpacing(7)
            h.setVerticalSpacing(4)
            h.addWidget(e, 0, 0, 1, 2)
            h.addWidget(v_name, 1, 0)
            h.addWidget(v_text, 1, 1)

            form_right.addRow(name, widget)

            return e, v_text
        
        (self.pdr_text,self.pdr_v),self.pdr = create_monitor("PDR(mA)"),0
        (self.pdt_text,self.pdt_v),self.pdt = create_monitor("PDT(mA)"),0
        self.ratio_rt_text,self.ratio_rt = read_only("PDR/PDT"),0
        self.ch1_text,self.v1 = read_only("CH0"),0
        self.ch2_text,self.v2 = read_only("CH1"),0
        self.ch3_text,self.v3 = read_only("CH2"),0
        self.ch4_text,self.v4 = read_only("CH3"),0

        monitor_box.setLayout(form_right)

        self.body_splitter.addWidget(param_box)
        self.body_splitter.addWidget(monitor_box)
        self.body_splitter.setStretchFactor(0, 2)
        self.body_splitter.setStretchFactor(1, 1)
        layout.addWidget(self.body_splitter, 1)

        self.excel_text.textChanged.connect(self.excel_adc_value)

        self.gain_text.textChanged.connect(self.gain_transfer)
        self.soa_text.textChanged.connect(self.soa_transfer)
        self.soa_mode_combo.currentIndexChanged.connect(self.soa_mode_changed)
        self.phase_text.textChanged.connect(self.phase_transfer)
        self.wavelena_text.textChanged.connect(self.wavelena_transfer)
        self.wavelenb_text.textChanged.connect(self.wavelenb_transfer)

        self.temp_signal.connect(
            lambda x:self.temperature_text.setText(f"{x:.4f}")
        )
        self.rt_signal.connect(self.rt_transfer)
        self.volite_signal.connect(self.volite_transfer)
        self.error_signal.connect(self.show_error)

        self.res_queue = Queue()
        self.feedback_res_queue = Queue()
        self.exrx_size = 20
        self.running = True
        self.recv_thread = None
        self.stress_dac_rows = load_stress_table_dac_rows()
        self.stress_dac_row_index = {
            row: index for index, row in enumerate(self.stress_dac_rows)
        }
        if self.fullband_table_error:
            self.auto_wavelength_btn.setEnabled(False)
            self.auto_wavelength_spin.setEnabled(False)
            self.auto_wavelength_match_label.setText(
                f"2001点标定表不可用：{self.fullband_table_error}"
            )
        else:
            self.preview_wavelength_match()
        QtCore.QTimer.singleShot(0, self._update_body_orientation)

    def reload_calibration_table(self):
        """Reload single-value wavelength matching after a machine switch."""

        try:
            points = load_fullband_accuracy_table()
        except (OSError, ValueError, csv.Error, json.JSONDecodeError) as exc:
            self.fullband_points = ()
            self.fullband_dac_rows = ()
            self.fullband_dac_row_index = {}
            self.fullband_table_error = str(exc)
            self.auto_wavelength_point = None
            self._auto_output_fullband_index = None
            if hasattr(self, "auto_wavelength_match_label"):
                self.auto_wavelength_match_label.setText(
                    f"2001点标定表不可用：{self.fullband_table_error}"
                )
            raise
        self.fullband_points = tuple(points)
        self.fullband_dac_rows = tuple(
            tuple(point.codes) for point in self.fullband_points
        )
        self.fullband_dac_row_index = {
            row: index for index, row in enumerate(self.fullband_dac_rows)
        }
        self.fullband_table_error = None
        self.auto_wavelength_point = None
        self._auto_output_fullband_index = None
        if hasattr(self, "auto_wavelength_match_label"):
            self.preview_wavelength_match()
        return True

    def _selected_feedback_selectors(self):
        return (
            int(self.ch0_feedback_btn.property("feedbackSelector")),
            int(self.ch1_feedback_btn.property("feedbackSelector")),
        )

    @staticmethod
    def _feedback_resistances(selectors):
        return tuple(PD_FEEDBACK_KOHM_BY_SELECTOR[value] for value in selectors)

    def _update_feedback_display(self, state="pending"):
        requested = self._selected_feedback_selectors()
        requested_resistances = self._feedback_resistances(requested)
        self.ch0_feedback_btn.setText(f"CH0：{requested_resistances[0]} kΩ")
        self.ch1_feedback_btn.setText(f"CH1：{requested_resistances[1]} kΩ")

        observed = self.observed_feedback_selectors
        if state == "confirmed" and observed == requested:
            resistances = self._feedback_resistances(observed)
            source = "IO实际"
            status = (
                f"跨阻：IO已核验 CH0 {resistances[0]} kΩ / "
                f"CH1 {resistances[1]} kΩ"
            )
        elif observed is not None:
            resistances = self._feedback_resistances(observed)
            source = "IO实际（与设定不一致）"
            status = (
                f"跨阻不一致：设定 CH0/CH1 "
                f"{requested_resistances[0]}/{requested_resistances[1]} kΩ；"
                f"IO实际 {resistances[0]}/{resistances[1]} kΩ"
            )
        else:
            resistances = requested_resistances
            source = "设定，待IO核验"
            if state == "applying":
                status = "跨阻：正在下发并核验choseA/B IO实际电平…"
            elif state == "error":
                status = "跨阻：下发或IO核验失败"
            elif state == "stopped":
                status = "跨阻：监测已停止，下次启动时重新核验"
            else:
                status = "跨阻：设定已改变，待IO核验"
        self.feedback_status_label.setText(status)
        gains = tuple(value / 2.0 for value in resistances)
        self.gain_label.setText(
            f"模拟跨阻（{source}）：CH0 {resistances[0]} kΩ（×{gains[0]:g}），"
            f"CH1 {resistances[1]} kΩ（×{gains[1]:g}），"
            "CH2/CH3/PDT/PDR 2 kΩ（×1） · 数字显示：CH0~3 ×1.00"
        )

    def _cycle_feedback_selector(self, button):
        cycle = (1, 3, 2, 0)  # 40 -> 20 -> 5 -> 2 kOhm
        current = int(button.property("feedbackSelector"))
        button.setProperty(
            "feedbackSelector", cycle[(cycle.index(current) + 1) % len(cycle)]
        )
        self.confirmed_feedback_selectors = None
        self.observed_feedback_selectors = None
        self.feedback_selection_dirty = True
        self._update_feedback_display("pending")
        self._apply_single_value_feedback()

    def _apply_single_value_feedback(self, ensure_monitor=True, show_error=True):
        """Apply CH0/CH1 feedback selectors and verify physical choseA/B IO."""

        if ensure_monitor and not self._start_single_value_monitor():
            return False
        expected = self._selected_feedback_selectors()
        while True:
            try:
                self.feedback_res_queue.get_nowait()
            except Empty:
                break
        self._update_feedback_display("applying")
        last_actual = None
        try:
            for _attempt in range(2):
                serial_write(build_extra_feedback_command(*expected))
                try:
                    actual, status = self.feedback_res_queue.get(timeout=1.0)
                except Empty:
                    continue
                actual = tuple(int(value) for value in actual)
                last_actual = actual
                self.observed_feedback_selectors = actual
                if status == ACK_ERROR_VALUE:
                    raise RuntimeError("单片机报告choseA/B IO核验失败")
                if status == ACK_VALUE and actual == expected:
                    self.confirmed_feedback_selectors = actual
                    self.feedback_selection_dirty = False
                    self._update_feedback_display("confirmed")
                    return True
            if last_actual is None:
                raise RuntimeError("2秒内未收到模拟跨阻状态回读")
            actual_resistances = self._feedback_resistances(last_actual)
            raise RuntimeError(
                "模拟跨阻回读与设定不一致："
                f"CH0 {actual_resistances[0]} kΩ / "
                f"CH1 {actual_resistances[1]} kΩ"
            )
        except (OSError, RuntimeError, ValueError) as exc:
            self.confirmed_feedback_selectors = None
            self.feedback_selection_dirty = True
            self._update_feedback_display("error")
            self.feedback_status_label.setText(f"跨阻下发失败：{exc}")
            if show_error:
                QMessageBox.warning(self, "模拟跨阻未生效", str(exc))
            return False

    @staticmethod
    def _codes_to_currents_ma(codes):
        full_scales = (
            PI11210_GAIN_FS_MA,
            PI11210_SOA_SOURCE_FS_MA,
            PI11210_PHASE_FS_MA,
            PI11210_WAVELENGTH_FS_MA,
            PI11210_WAVELENGTH_FS_MA,
        )
        return tuple(
            float(code) * float(full_scale) / 65536.0
            for code, full_scale in zip(codes, full_scales)
        )

    def _wavelength_match_text(self, point, suffix=""):
        requested = float(self.auto_wavelength_spin.value())
        error_pm = (float(point.measured_nm) - requested) * 1000.0
        currents = self._codes_to_currents_ma(point.codes)
        return (
            f"匹配第{int(point.index) + 1}/"
            f"{FULLBAND_ACCURACY_POINT_COUNT}点 · "
            f"表目标 {float(point.target_nm):.6f} nm · "
            f"波长计标定 {float(point.measured_nm):.8f} nm · "
            f"与输入相差 {error_pm:+.3f} pm\n"
            f"电流：GAIN {currents[0]:.3f} mA · SOA {currents[1]:.3f} mA · "
            f"PHASE {currents[2]:.4f} mA · WAVE A {currents[3]:.4f} mA · "
            f"WAVE B {currents[4]:.4f} mA{suffix}"
        )

    def preview_wavelength_match(self):
        """Populate all five channels from the nearest audited wavelength row."""

        if not self.fullband_points:
            return None
        try:
            point = nearest_fullband_accuracy_point(
                self.fullband_points, self.auto_wavelength_spin.value()
            )
        except ValueError as exc:
            self.auto_wavelength_point = None
            self.auto_wavelength_match_label.setText(str(exc))
            return None

        self.auto_wavelength_point = point
        self.soa_mode_combo.setCurrentIndex(0)
        self.excel_text.setText("\t".join(str(code) for code in point.codes))
        self.auto_wavelength_match_label.setText(
            self._wavelength_match_text(point)
        )
        return point

    def output_by_wavelength(self):
        """Match, populate and safely write one calibrated wavelength row."""

        point = self.preview_wavelength_match()
        if point is None:
            QMessageBox.warning(
                self, "波长自动输出", "没有可用的2001点单模标定数据。"
            )
            return
        self._auto_output_fullband_index = int(point.index)
        try:
            self.write_data_MCU()
        finally:
            self._auto_output_fullband_index = None
        states = (
            self.gain_state,
            self.soa_state,
            self.phase_state,
            self.wavelena_state,
            self.wavelenb_state,
        )
        if all("已写入" in state.text() for state in states):
            self.auto_wavelength_match_label.setText(
                self._wavelength_match_text(point, " · 已输出")
            )

    def _update_body_orientation(self):
        if not hasattr(self, "body_splitter"):
            return
        orientation = (
            QtCore.Qt.Horizontal if self.width() >= 980 else QtCore.Qt.Vertical
        )
        if self.body_splitter.orientation() != orientation:
            self.body_splitter.setOrientation(orientation)
            self.body_splitter.setSizes((2, 1))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_body_orientation()

    def set_dac_label(self, dac_type):
        self.dac_label.setText("DAC: PI11210（原始码；表内点自动预置路径）")

    # def showEvent(self, a0):
    #     self.serial_open()
    #     self.start_recv_thread()
    #     return super().showEvent(a0)

    def start_recv_thread(self):
        if self.recv_thread and self.recv_thread.is_alive():
            return
        
        self.running = True
        self.recv_thread = threading.Thread(target=self.serial_recv, daemon=True)
        self.recv_thread.start()

        self.gain_text.textChanged.emit(self.gain_text.text())
        self.soa_text.textChanged.emit(self.soa_text.text())
        self.phase_text.textChanged.emit(self.phase_text.text())
        self.wavelena_text.textChanged.emit(self.wavelena_text.text())
        self.wavelenb_text.textChanged.emit(self.wavelenb_text.text())

    def _single_value_monitor_is_active(self):
        """Return whether this page owns a live single-value receiver."""
        return bool(
            ap_open
            and self.recv_thread is not None
            and self.recv_thread.is_alive()
        )

    def _start_single_value_monitor(self):
        """Start EXTRA mode and its receiver, returning success to the caller.

        The single-value DAC ACK is consumed by ``serial_recv``.  Keeping this
        startup in one place prevents the Write button from sending a command
        before any receiver exists, while also ensuring a failed startup never
        falls through to a DAC write.
        """
        global ap_open
        global switch_mode_enable

        if self._single_value_monitor_is_active():
            return True
        if ap_open or not switch_mode_enable:
            QMessageBox.warning(
                self,
                "单值模式未启动",
                "当前有其他工作流正在运行，请先停止后再写入DAC码。",
            )
            return False

        try:
            ensure_serial_open()
            ser.reset_input_buffer()
            send_work_mode_command(2)
            self.start_recv_thread()
            if self.recv_thread is None or not self.recv_thread.is_alive():
                raise RuntimeError("单值模式接收线程未成功启动")
        except Exception as exc:
            self.running = False
            try:
                if self.recv_thread is not None and self.recv_thread.is_alive():
                    self.stop_recv_thread()
            except Exception:
                pass
            try:
                release_serial_if_allowed()
            except Exception:
                pass
            with ap_cond:
                ap_open = False
            switch_mode_enable = True
            self.monitor_btn.setText("开始")
            QMessageBox.critical(
                self, "串口错误", f"单值模式启动失败:\n{str(exc)}"
            )
            return False

        with ap_cond:
            ap_open = True
        switch_mode_enable = False
        self.monitor_btn.setText("停止")
        return True

    def stop_recv_thread(self):
        self.running = False

        if self.recv_thread:
            self.recv_thread.join(timeout=0.6)
            if self.recv_thread.is_alive():
                self.error_signal.emit("串口接收线程未能在超时前停止")
                return False
            self.recv_thread = None
        return True

    def _stop_single_value_output_safely(self):
        """Force all five DAC channels to zero before releasing single-value IO."""
        global ap_open
        global switch_mode_enable

        if self._single_value_monitor_is_active():
            self._write_direct_dac_codes(
                (0, 0, 0, 0, 0),
                soa_mode=PI11210_SOA_SHUTTER_MODE,
            )
            if not self.stop_recv_thread():
                raise RuntimeError("单值模式接收线程未能在全零关光后停止")
        with ap_cond:
            ap_open = False
        switch_mode_enable = True
        self.monitor_btn.setText("开始")
        self.confirmed_feedback_selectors = None
        self._update_feedback_display("stopped")
        self.dac_label.setText("DAC: PI11210（五路全零，已确认关光）")
        return True

    def on_work(self):
        global ser
        global ap_open
        global switch_mode_enable

        if not ap_open:
            if self._start_single_value_monitor():
                self._apply_single_value_feedback(
                    ensure_monitor=False, show_error=True
                )
        else:
            try:
                self._stop_single_value_output_safely()
                release_serial_if_allowed()
            except Exception as e:
                self.error_signal.emit(f"单值模式全零关光失败：{e}")
                return

    def serial_recv(self):
        # self.serial_open()
        rx_buffer = bytearray()
        latest_temperature = None
        latest_rt = None
        latest_voltages = None
        last_gui_emit = 0.0
        while self.running:
            try:
                # Batch the high-rate USB/CDC monitor stream.  At low traffic
                # pyserial still returns after its 0.2 s timeout, while at the
                # normal continuous rate this avoids thousands of tiny reads.
                data = ser.read(4096)
            except Exception as exc:
                if self.running:
                    self.error_signal.emit(f"串口接收失败：{exc}")
                break
            if not data:
                continue

            rx_buffer.extend(data)
            for frame in extract_single_value_frames(rx_buffer):
                v = list(frame)
                if v[3] == 0x00:
                    self.res_queue.put(v)
                elif v[3] == 0x01:
                    temp_int_high = v[4]
                    temp_int_low = v[5]
                    temp_dec_high = v[6]
                    temp_dec_low = v[7]
                    self.temperature = (temp_int_high<<8) + temp_int_low + \
                                    ((temp_dec_high<<8)+temp_dec_low)*0.0001
                    latest_temperature = self.temperature
                elif v[3] == 0x02:
                    _pdt = (v[4]<<8)+v[5]# 先采的ch6是pdt
                    _pdr = (v[6]<<8)+v[7]# 后采的ch7是pdr

                    ch1 = (v[8]<<8)+v[9]
                    ch2 = (v[10]<<8)+v[11]
                    ch3 = (v[12]<<8)+v[13]
                    ch4 = (v[14]<<8)+v[15]

                    latest_rt = (_pdr, _pdt)
                    latest_voltages = (ch1, ch2, ch3, ch4)
                elif v[3] == 0x03:
                    feedback_status = decode_extra_feedback_status(frame)
                    if feedback_status is not None:
                        self.feedback_res_queue.put(feedback_status)

            # The MCU can report monitor frames much faster than Qt can paint
            # them.  Preserve every DAC ACK above, but publish only the newest
            # monitor snapshot at 20 Hz so the UI event queue stays responsive.
            now = time.monotonic()
            if now - last_gui_emit >= 0.05:
                if latest_temperature is not None:
                    self.temp_signal.emit(latest_temperature)
                    latest_temperature = None
                if latest_rt is not None:
                    self.rt_signal.emit(*latest_rt)
                    latest_rt = None
                if latest_voltages is not None:
                    self.volite_signal.emit(*latest_voltages)
                    latest_voltages = None
                last_gui_emit = now

        self.serial_close()

    def write_data_MCU(self):
        invalid_fields = self._refresh_direct_dac_inputs()
        if invalid_fields:
            QMessageBox.warning(
                self,
                "DAC码输入错误",
                "以下通道不是有效的原始DAC码："
                + "、".join(invalid_fields)
                + "\n请直接复制标定表中的五个‘码’列，不要复制mA列。",
            )
            return
        if not self._start_single_value_monitor():
            return
        if (
            self.feedback_selection_dirty
            and self.confirmed_feedback_selectors
            != self._selected_feedback_selectors()
            and not self._apply_single_value_feedback(
                ensure_monitor=False, show_error=True
            )
        ):
            return
        soa_mode = self._selected_soa_mode()
        target_codes = (
            self.gain, self.soa, self.phase, self.wavelena, self.wavelenb
        )
        while True:
            try:
                self.res_queue.get_nowait()
            except Empty:
                break
        fullband_table_index = self._auto_output_fullband_index
        if (
            soa_mode != PI11210_SOA_SOURCE_MODE
            or fullband_table_index is None
            or not 0 <= int(fullband_table_index) < len(self.fullband_dac_rows)
            or self.fullband_dac_rows[int(fullband_table_index)] != target_codes
        ):
            fullband_table_index = None
        stress_table_index = (
            self.stress_dac_row_index.get(target_codes)
            if soa_mode == PI11210_SOA_SOURCE_MODE else None
        )
        try:
            if fullband_table_index is not None and len(self.fullband_dac_rows) > 1:
                predecessor_index = (
                    fullband_table_index - 1
                ) % len(self.fullband_dac_rows)
                self._write_direct_dac_codes(
                    self.fullband_dac_rows[predecessor_index]
                )
                time.sleep(SINGLE_VALUE_TABLE_PRECONDITION_S)
                self.dac_label.setText(
                    f"DAC: PI11210（先预置2001点表第{predecessor_index + 1}点，"
                    f"再写第{fullband_table_index + 1}点）"
                )
            elif stress_table_index is not None and len(self.stress_dac_rows) > 1:
                predecessor_index = (
                    stress_table_index - 1
                ) % len(self.stress_dac_rows)
                self._write_direct_dac_codes(
                    self.stress_dac_rows[predecessor_index]
                )
                time.sleep(SINGLE_VALUE_TABLE_PRECONDITION_S)
                self.dac_label.setText(
                    f"DAC: PI11210（先预置应力表第{predecessor_index + 1}点，"
                    f"再写第{stress_table_index + 1}点）"
                )
            else:
                self.dac_label.setText("DAC: PI11210（自定义原始码直接写入）")
            response = self._write_direct_dac_codes(target_codes, soa_mode=soa_mode)
            if soa_mode == PI11210_SOA_SHUTTER_MODE:
                self.dac_label.setText(
                    "DAC: PI11210（SOA关光：CLR安全零电流）"
                )
        except (Empty, RuntimeError, ValueError) as exc:
            self.error_signal.emit(f"DAC写入失败：{exc}")
            return

        res_gain = (response[4]<<8)+response[5]
        res_soa = (response[6]<<8)+response[7]
        res_phase = (response[8]<<8)+response[9]
        res_wavelena = (response[10]<<8)+response[11]
        res_wavelenb = (response[12]<<8)+response[13]

        res_list, self_list, state_list = [res_gain,res_soa,res_phase,res_wavelena,res_wavelenb],\
                            list(target_codes),\
                            [self.gain_state, self.soa_state,self.phase_state,self.wavelena_state,self.wavelenb_state]

        for i in range(len(res_list)):
            if res_list[i]==self_list[i]:
                state_list[i].setText("● 已写入")
                state_list[i].setStyleSheet("color:#16a34a;")
            else:
                state_list[i].setText("● 回读不一致")
                state_list[i].setStyleSheet("color:#dc2626;")

    def _write_direct_dac_codes(
        self, codes, soa_mode=PI11210_SOA_SOURCE_MODE
    ):
        raw_codes = tuple(codes)
        if len(raw_codes) != 5:
            raise ValueError("single-value DAC command requires five codes")
        expected = list(
            normalize_pi11210_calibration_code(code, maximum)
            for code, maximum in zip(raw_codes, _pi11210_table_code_limits())
        )
        requested_mode = parse_pi11210_dac_code(
            soa_mode, PI11210_SOA_SHUTTER_MODE
        )
        if requested_mode == PI11210_SOA_SHUTTER_MODE:
            expected[1] = PI11210_SOA_SHUTTER_CODE
        expected = tuple(expected)
        serial_write(
            build_single_value_dac_command(expected, soa_mode=requested_mode)
        )
        try:
            response = self.res_queue.get(timeout=2.0)
        except Empty:
            raise RuntimeError("2秒内未收到单片机DAC回读") from None
        if len(response) < 14:
            raise RuntimeError("单片机回读帧长度不足")
        returned = tuple(
            (response[4 + channel * 2] << 8)
            | response[5 + channel * 2]
            for channel in range(5)
        )
        if returned != expected:
            raise RuntimeError(
                "单片机回读不一致：" + ",".join(str(value) for value in returned)
            )
        if len(response) >= 20 and response[15] & 0x80:
            actual_mode = int(response[14])
            flags = int(response[15])
            raw_status = (int(response[16]) << 8) | int(response[17])
            i2c_errors = (int(response[18]) << 8) | int(response[19])
            if actual_mode != requested_mode:
                raise RuntimeError("SOA正向/关光状态与请求不一致")
            if flags & 0x03 != 0x03:
                raise RuntimeError(
                    f"PI11210离线或器件ID不符（状态0x{raw_status:04X}）"
                )
            if not flags & 0x04:
                raise RuntimeError(f"PI11210写入失败（累计I²C错误{i2c_errors}次）")
            if flags & 0x30:
                raise RuntimeError(f"PI11210过温保护（状态0x{raw_status:04X}）")
        elif requested_mode == PI11210_SOA_SHUTTER_MODE:
            raise RuntimeError("当前单片机固件不支持安全的SOA关光命令")
        return response

    def excel_adc_value(self, str):
        cell_list = str.rstrip("\n").split("\t")
        if not len(cell_list) == 5:
            return
        # Calibration rows always describe the normal positive-SOA path.
        self.soa_mode_combo.setCurrentIndex(0)
        self_text_list =[
            self.gain_text,self.soa_text,self.phase_text,self.wavelena_text,self.wavelenb_text
        ]
        for i in range(len(self_text_list)):
            self_text_list[i].setText(cell_list[i])

    def _refresh_direct_dac_inputs(self):
        soa_maximum = (
            PI11210_SOA_SHUTTER_CODE
            if self._selected_soa_mode() == PI11210_SOA_SHUTTER_MODE
            else PI11210_SOA_SOURCE_MAX_CODE
        )
        fields = (
            ("GAIN", self.gain_text, self.gain_dac_text, self.gain_state,
             "gain", PI11210_GAIN_MAX_CODE),
            ("SOA", self.soa_text, self.soa_dac_text, self.soa_state,
             "soa", soa_maximum),
            ("PHASE", self.phase_text, self.phase_dac_text, self.phase_state,
             "phase", PI11210_PHASE_MAX_CODE),
            ("WAVE_A", self.wavelena_text, self.wavelena_dac_text,
             self.wavelena_state, "wavelena", PI11210_WAVELENGTH_MAX_CODE),
            ("WAVE_B", self.wavelenb_text, self.wavelenb_dac_text,
             self.wavelenb_state, "wavelenb", PI11210_WAVELENGTH_MAX_CODE),
        )
        invalid_fields = []
        for name, edit, code_label, state_label, attribute, maximum in fields:
            try:
                code = normalize_pi11210_calibration_code(edit.text(), maximum)
            except ValueError:
                invalid_fields.append(name)
                code_label.setText("--")
                state_label.setText("● 输入无效")
                state_label.setStyleSheet("color:#dc2626;")
                continue
            setattr(self, attribute, code)
            code_label.setText(str(code))
            state_label.setText("● 未写入")
            state_label.setStyleSheet("color:#d97706;")
        return invalid_fields

    def _selected_soa_mode(self):
        mode = self.soa_mode_combo.currentData()
        return (
            PI11210_SOA_SHUTTER_MODE
            if int(mode) == PI11210_SOA_SHUTTER_MODE
            else PI11210_SOA_SOURCE_MODE
        )

    def soa_mode_changed(self, _index):
        shutter = self._selected_soa_mode() == PI11210_SOA_SHUTTER_MODE
        if shutter:
            self.soa_text.setText(str(PI11210_SOA_SHUTTER_CODE))
            self.soa_text.setEnabled(False)
            self.soa_text.setToolTip(
                "负值表示关光；硬件输出由固件固定为安全零电流"
            )
        else:
            self.soa_text.setEnabled(True)
            self.soa_text.setValidator(
                QtGui.QIntValidator(0, PI11210_SOA_SOURCE_MAX_CODE, self.soa_text)
            )
            self.soa_text.setToolTip(
                f"正向SOA原始DAC码：0～{PI11210_SOA_SOURCE_MAX_CODE}"
            )
        self._refresh_direct_dac_inputs()

    def gain_transfer(self, _text):
        self._refresh_direct_dac_inputs()

    def soa_transfer(self, _text):
        self._refresh_direct_dac_inputs()

    def phase_transfer(self, _text):
        self._refresh_direct_dac_inputs()

    def wavelena_transfer(self, _text):
        self._refresh_direct_dac_inputs()

    def wavelenb_transfer(self, _text):
        self._refresh_direct_dac_inputs()

    def rt_transfer(self, r, t):
        self.pdr, v7 = pd_adc_code_to_current_ma(r)
        self.pdt, v6 = pd_adc_code_to_current_ma(t)
        if self.pdt > PD_RATIO_MIN_CURRENT_MA:
            self.ratio_rt = self.pdr / self.pdt
            ratio_text = f"{self.ratio_rt:.3f}"
        else:
            self.ratio_rt = float("nan")
            ratio_text = "--"

        self.pdr_v.setText(f"{v7:.2f}V")
        self.pdt_v.setText(f"{v6:.2f}V")
        self.pdr_text.setText(f"{self.pdr:.4f}")
        self.pdt_text.setText(f"{self.pdt:.4f}")
        self.ratio_rt_text.setText(ratio_text)

    def volite_transfer(self, ch1, ch2, ch3, ch4):
        self.v1 = ch1*PD_ADC_REFERENCE_V/PD_ADC_CODE_COUNT
        self.v2 = ch2*PD_ADC_REFERENCE_V/PD_ADC_CODE_COUNT
        self.v3 = ch3*PD_ADC_REFERENCE_V/PD_ADC_CODE_COUNT
        self.v4 = ch4*PD_ADC_REFERENCE_V/PD_ADC_CODE_COUNT

        self.ch1_text.setText(f"{self.v1:.4f}V")
        self.ch2_text.setText(f"{self.v2:.4f}V")
        self.ch3_text.setText(f"{self.v3:.4f}V")
        self.ch4_text.setText(f"{self.v4:.4f}V")

    def reset_write_state(self):
        states = [self.gain_state, self.soa_state,self.phase_state,self.wavelena_state,self.wavelenb_state]
        for s in states:
            s.setText("● 未写入")
            s.setStyleSheet("color:#d97706")

    def show_error(self, str):
        QMessageBox.critical(self, "crash", f"异常:\n{str}")

    def serial_open(self):
        try:
            if not ser.is_open:
                ser.open()
            ser_open = True
        except Exception as e:
            self.error_signal.emit(str(e))
            return
        
    def serial_close(self):
        try:
            release_serial_if_allowed()
            ser_open = False
        except Exception as e:
            self.error_signal.emit(str(e))


class KalmanFilter1D:
    def __init__(self, Q=1e-5, R=0.1**2):
        # Q: 过程噪声（模型不确定性）
        # R: 测量噪声（传感器噪声）
        self.Q = Q
        self.R = R

        self.x = 0.0   # 状态估计
        self.P = 1.0   # 估计误差协方差

    def update(self, z):
        # 1. 预测
        x_pred = self.x
        P_pred = self.P + self.Q

        # 2. 卡尔曼增益
        K = P_pred / (P_pred + self.R)

        # 3. 更新
        self.x = x_pred + K * (z - x_pred)
        self.P = (1 - K) * P_pred

        return self.x
    
def hampel_filter(x, window_size=3, n_sigma=3):
    x = np.asarray(x).copy()
    n = len(x)

    for i in range(window_size, n - window_size):
        window = x[i-window_size:i+window_size+1]

        med = np.median(window)
        mad = np.median(np.abs(window - med))

        if mad == 0:
            continue

        threshold = n_sigma * 1.4826 * mad

        if abs(x[i] - med) > threshold:
            x[i] = med

    return x

if __name__=="__main__":
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet("""
        QPushButton, QLabel {
            font-size: 12pt;
            font-family: Microsoft YaHei;
        }
    """)
    win = MainWindow()
    win.show()
    if "--stress-autostart" in sys.argv:
        QtCore.QTimer.singleShot(500, win.start_stress_debug)
    elif "--temperature-autostart" in sys.argv:
        QtCore.QTimer.singleShot(500, win.start_temperature_debug)
    sys.exit(app.exec_())


