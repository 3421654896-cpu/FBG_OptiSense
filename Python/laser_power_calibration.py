"""Safe AQ6150B-assisted calibration for the PI11210/JDSU laser table.

The tool tunes SOA for optical power and then makes a small local PHASE
correction to recover the requested wavelength.  It never changes GAIN,
WAVE_A or WAVE_B, and it enforces the laser-current limits before a frame is
sent.  Progress is checkpointed after every wavelength so an interrupted run
can be resumed.

The firmware source is not overwritten.  A proposed calibrated C file and a
JSON report are written next to this script for review.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pyvisa
import serial
import yaml

from laser_dac_safety import PI11210_SAFE_CODE_LIMITS, validate_pi11210_codes


SERIAL_FRAME_SIZE = 808
PI11210_FULL_SCALE_MA = (150.0, 150.0, 20.0, 80.0, 80.0)
# The laser datasheet gives 150 mA absolute maxima for GAIN and SOA.  The
# user-authorized operating ceiling for the new high-power map is 135 mA.
# Keep a second margin below the absolute rating and enforce it on every
# calibration/probe command.
LASER_MAX_MA = (135.0, 135.0, 10.0, 30.0, 30.0)

ADC_REFERENCE_V = 2.5
ADC_CODE_COUNT = 4096.0
PD_BIAS_V = 1.25
PD_FEEDBACK_OHM = 2000.0

DEFAULT_TARGET_POWER_MW = 0.5
DEFAULT_POWER_TOLERANCE_MW = 0.006
DEFAULT_WAVELENGTH_TOLERANCE_PM = 3.0
DEFAULT_PHASE_SLOPE_PM_PER_MA = -75.0
MAX_PHASE_OFFSET_MA = 0.65
MAX_SOA_STEP_MA = 20.0
MIN_VALID_POWER_MW = 0.02

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent / "JDSU"
DAC_SOURCE = PROJECT / "Core" / "Src" / "dac_const.c"
WAVE_YAML = HERE / "wave_const.yaml"
CHECKPOINT = HERE / "laser_calibration_checkpoint.json"
REPORT = HERE / "laser_calibration_report.json"
PROPOSED_C = HERE / "dac_const_calibrated.c"


def clamp(value: float, low: float, high: float) -> float:
    return min(max(float(value), low), high)


def current_to_code(current_ma: float, channel: int) -> int:
    current_ma = clamp(current_ma, 0.0, LASER_MAX_MA[channel])
    code = round(current_ma / PI11210_FULL_SCALE_MA[channel] * 65536.0)
    # Never let DAC rounding cross a configured safety ceiling.
    maximum_code = math.floor(
        LASER_MAX_MA[channel] / PI11210_FULL_SCALE_MA[channel] * 65536.0
    )
    return min(max(int(code), 0), min(0xFFFF, int(maximum_code)))


def code_to_current(code: int, channel: int) -> float:
    return int(code) * PI11210_FULL_SCALE_MA[channel] / 65536.0


def pd_code_to_current_ma(code: int) -> float:
    voltage = float(code) * ADC_REFERENCE_V / ADC_CODE_COUNT
    return max(0.0, (PD_BIAS_V - voltage) * 1000.0 / PD_FEEDBACK_OHM)


def parse_dac_rows(source: Path) -> list[list[int]]:
    rows: list[list[int]] = []
    row_re = re.compile(r"\{\s*([^{}]+?)\s*\},?")
    for match in row_re.finditer(source.read_text(encoding="utf-8")):
        fields = [part.strip() for part in match.group(1).split(",")]
        if len(fields) < 3:
            continue
        try:
            values = [int(field, 0) for field in fields]
        except ValueError:
            continue
        if values[:3] == [0xFFFF, 0xFFFF, 0xFFFF]:
            break
        if len(values) == 5:
            rows.append(values)
    if not rows:
        raise RuntimeError(f"No valid five-channel DAC rows found in {source}")
    return rows


def load_wavelengths(source: Path) -> list[float]:
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    rows = data.get("Wave_DATA", data)
    return [float(integer) + float(fraction) / 1000.0 for integer, fraction in rows]


def parse_number_array(reply: str) -> list[float]:
    values: list[float] = []
    for item in str(reply).strip().replace("\r", "").replace("\n", "").split(","):
        try:
            values.append(float(item.strip()))
        except (TypeError, ValueError):
            pass
    return values


@dataclass
class Reading:
    wavelength_nm: float
    wavelength_error_pm: float
    power_mw: float
    power_dbm: float
    peak_count: int
    secondary_power_mw: float | None = None
    side_mode_suppression_db: float | None = None
    pdt_code: int | None = None
    pdr_code: int | None = None
    pdt_ma: float | None = None
    pdr_ma: float | None = None
    pdr_pdt_ratio: float | None = None


class LaserSerial:
    def __init__(self, port: str, *, code_limits=PI11210_SAFE_CODE_LIMITS):
        self.code_limits = tuple(int(value) for value in code_limits)
        if len(self.code_limits) != 5:
            raise ValueError("PI11210 safety limit set must contain five channels")
        self.serial = serial.Serial(port, 115200, timeout=0.05, write_timeout=1.0)
        self._rx = bytearray()
        self.set_extra_mode()

    def close(self) -> None:
        self.serial.close()

    def set_extra_mode(self) -> None:
        frame = bytearray(SERIAL_FRAME_SIZE)
        frame[0:4] = bytes((0xFF, 0xFF, 0x01, 0x02))
        frame[8] = 2
        self.serial.write(frame)
        self.serial.flush()
        time.sleep(0.08)
        self.serial.reset_input_buffer()
        self._rx.clear()

    @staticmethod
    def validate_codes(
        codes: list[int], limits=PI11210_SAFE_CODE_LIMITS
    ) -> tuple[int, int, int, int, int]:
        """Reject negative, fractional and over-limit codes before serialization."""

        return validate_pi11210_codes(codes, limits)

    def set_codes(self, codes: list[int]) -> None:
        safe_codes = self.validate_codes(codes, self.code_limits)
        frame = bytearray(SERIAL_FRAME_SIZE)
        frame[0:4] = bytes((0xFF, 0xFF, 0x00, 0x01))
        for channel, value in enumerate(safe_codes):
            frame[4 + channel * 2] = (value >> 8) & 0xFF
            frame[5 + channel * 2] = value & 0xFF
        self.serial.write(frame)
        self.serial.flush()

    def latest_pd_reading(self) -> tuple[int, int] | None:
        waiting = self.serial.in_waiting
        if waiting:
            self._rx.extend(self.serial.read(waiting))
        marker = bytes((0xFF, 0xFF, 0x02, 0x02))
        index = self._rx.rfind(marker)
        if index < 0 or len(self._rx) < index + 8:
            return None
        pdt = (self._rx[index + 4] << 8) | self._rx[index + 5]
        pdr = (self._rx[index + 6] << 8) | self._rx[index + 7]
        if len(self._rx) > 4096:
            del self._rx[:-1024]
        return pdt, pdr


class AQ6150B:
    def __init__(self, resource: str, *, average_count: int = 1):
        average_count = int(average_count)
        if not 1 <= average_count <= 100:
            raise ValueError("AQ6150B average_count must be between 1 and 100")
        self.manager = pyvisa.ResourceManager()
        self.instrument = self.manager.open_resource(resource)
        self.instrument.timeout = 5000
        identity = str(self.instrument.query("*IDN?")).strip()
        if "YOKOGAWA" not in identity.upper() or "AQ6150" not in identity.upper():
            raise RuntimeError(f"Unexpected wavelength meter at {resource}: {identity}")
        for command in (
            "*CLS",
            ":INIT:CONT OFF",
            ":FORM:NDAT 0NM",
            ":UNIT:POW DBM",
            # Make wavelength/SMSR evidence independent of whatever settings
            # were last selected on the AQ6150B front panel.  NORMAL is the
            # specified high-accuracy update mode; NARROW is the CW-laser
            # correction profile.  A 40 dB relative search threshold ensures
            # a side mode near the 20 dB acceptance boundary is not hidden.
            ":SENS:CORR:DEV NARR",
            ":SENS:URAT NORM",
            ":CALC2:ASE ON",
            f":CALC2:COUN {average_count}",
            ":CALC2:PTHR:MOD REL",
            ":CALC2:PTHR 40",
            ":CALC2:PEXC 10",
        ):
            self.instrument.write(command)
        self.measurement_profile = {
            "device_type": str(self.instrument.query(":SENS:CORR:DEV?")).strip(),
            "update_rate": str(self.instrument.query(":SENS:URAT?")).strip(),
            "auto_peak_search": str(self.instrument.query(":CALC2:ASE?")).strip(),
            "average_count": int(float(self.instrument.query(":CALC2:COUN?"))),
            "peak_threshold_mode": str(
                self.instrument.query(":CALC2:PTHR:MOD?")
            ).strip(),
            "peak_threshold_relative_db": float(
                self.instrument.query(":CALC2:PTHR?")
            ),
            "peak_excursion_db": float(self.instrument.query(":CALC2:PEXC?")),
        }
        if (
            not self.measurement_profile["device_type"].upper().startswith("NARR")
            or not self.measurement_profile["update_rate"].upper().startswith("NORM")
            or self.measurement_profile["auto_peak_search"] not in {"1", "ON"}
            or self.measurement_profile["average_count"] != average_count
            or not self.measurement_profile["peak_threshold_mode"].upper().startswith("REL")
            or abs(self.measurement_profile["peak_threshold_relative_db"] - 40.0) > 0.01
            or abs(self.measurement_profile["peak_excursion_db"] - 10.0) > 0.01
        ):
            raise RuntimeError(
                "AQ6150B measurement profile did not accept the required settings: "
                f"{self.measurement_profile}"
            )
        self.identity = identity

    def close(self) -> None:
        self.instrument.close()
        self.manager.close()

    def measure(self, target_nm: float, select_main_peak: bool = False) -> Reading:
        self.instrument.write(":INIT")
        # AQ6150B FETCh queries are overlapping commands: when issued during a
        # measurement the instrument holds the response until that measurement
        # is complete.  The VISA timeout is the fail-closed bound, so an extra
        # fixed delay here only slows every closed-loop probe and can never
        # prove completion.
        wavelengths = parse_number_array(self.instrument.query(":FETC:ARR:POW:WAV?"))
        powers = parse_number_array(self.instrument.query(":FETC:ARR:POW?"))
        if not wavelengths or not powers:
            raise RuntimeError("AQ6150B returned an empty measurement")
        count = min(int(wavelengths[0]), int(powers[0]))
        if count <= 0:
            raise RuntimeError("AQ6150B returned an empty measurement")
        wavelengths = wavelengths[1 : 1 + count]
        powers = powers[1 : 1 + count]
        if not wavelengths or len(wavelengths) != len(powers):
            raise RuntimeError("AQ6150B wavelength/power arrays are inconsistent")
        if select_main_peak:
            index = max(range(len(powers)), key=lambda item: powers[item])
        else:
            index = min(
                range(len(wavelengths)),
                key=lambda item: abs(wavelengths[item] * 1e9 - target_nm),
            )
        wavelength_nm = wavelengths[index] * 1e9
        power_dbm = powers[index]
        other_powers = [value for item, value in enumerate(powers) if item != index]
        secondary_dbm = max(other_powers) if other_powers else None
        return Reading(
            wavelength_nm=wavelength_nm,
            wavelength_error_pm=(wavelength_nm - target_nm) * 1000.0,
            power_mw=10.0 ** (power_dbm / 10.0),
            power_dbm=power_dbm,
            peak_count=count,
            secondary_power_mw=(10.0 ** (secondary_dbm / 10.0)
                                if secondary_dbm is not None else None),
            side_mode_suppression_db=(power_dbm - secondary_dbm
                                      if secondary_dbm is not None else None),
        )


def enrich_pd(reading: Reading, laser: LaserSerial) -> Reading:
    try:
        raw = laser.latest_pd_reading()
    except (serial.SerialException, OSError):
        # PDT/PDR telemetry enriches the AQ result but is not the wavelength or
        # power authority.  A transient USB status-query failure must not throw
        # away an otherwise valid AQ6150B measurement/checkpoint.
        return reading
    if raw is None:
        return reading
    pdt_code, pdr_code = raw
    pdt_ma = pd_code_to_current_ma(pdt_code)
    pdr_ma = pd_code_to_current_ma(pdr_code)
    reading.pdt_code = pdt_code
    reading.pdr_code = pdr_code
    reading.pdt_ma = pdt_ma
    reading.pdr_ma = pdr_ma
    reading.pdr_pdt_ratio = pdr_ma / pdt_ma if pdt_ma > 1e-6 else None
    return reading


def take_reading(
    laser: LaserSerial,
    meter: AQ6150B,
    codes: list[int],
    target_nm: float,
    settle_s: float = 0.30,
    select_main_peak: bool = False,
) -> Reading:
    laser.set_codes(codes)
    time.sleep(settle_s)
    return enrich_pd(meter.measure(target_nm, select_main_peak=select_main_peak), laser)


def bounded_phase(original_ma: float, requested_ma: float) -> float:
    low = max(0.0, original_ma - MAX_PHASE_OFFSET_MA)
    high = min(LASER_MAX_MA[2], original_ma + MAX_PHASE_OFFSET_MA)
    return clamp(requested_ma, low, high)


def calibrate_row(
    laser: LaserSerial,
    meter: AQ6150B,
    index: int,
    target_nm: float,
    original_codes: list[int],
    target_power_mw: float,
    power_tolerance_mw: float,
    wavelength_tolerance_pm: float,
) -> dict:
    codes = list(original_codes)
    original_phase_ma = code_to_current(codes[2], 2)
    soa_ma = code_to_current(codes[1], 1)
    phase_ma = original_phase_ma
    history: list[dict] = []

    reading = take_reading(laser, meter, codes, target_nm, settle_s=0.34)
    history.append({"action": "baseline", "codes": list(codes), "reading": asdict(reading)})

    phase_slope = DEFAULT_PHASE_SLOPE_PM_PER_MA
    last_phase_sample: tuple[float, float] | None = None

    for iteration in range(12):
        power_ok = abs(reading.power_mw - target_power_mw) <= power_tolerance_mw
        wavelength_ok = abs(reading.wavelength_error_pm) <= wavelength_tolerance_pm
        if power_ok and wavelength_ok:
            break

        if not power_ok:
            if reading.power_mw < MIN_VALID_POWER_MW:
                raise RuntimeError(f"Optical power collapsed to {reading.power_mw:.6f} mW")
            requested = soa_ma * target_power_mw / reading.power_mw
            requested = clamp(requested, soa_ma - MAX_SOA_STEP_MA, soa_ma + MAX_SOA_STEP_MA)
            soa_ma = clamp(requested, 0.0, LASER_MAX_MA[1])
            codes[1] = current_to_code(soa_ma, 1)
            reading = take_reading(laser, meter, codes, target_nm)
            history.append(
                {"action": f"soa_{iteration}", "codes": list(codes), "reading": asdict(reading)}
            )
            continue

        if last_phase_sample is not None:
            previous_phase, previous_error = last_phase_sample
            delta_phase = phase_ma - previous_phase
            if abs(delta_phase) > 0.005:
                measured_slope = (reading.wavelength_error_pm - previous_error) / delta_phase
                if -600.0 <= measured_slope <= -5.0:
                    phase_slope = measured_slope
        last_phase_sample = (phase_ma, reading.wavelength_error_pm)

        requested_phase = phase_ma - reading.wavelength_error_pm / phase_slope
        # A 0.2 mA phase step can cross tens of picometres on this device.
        # Probe locally in <=0.08 mA increments, then use the measured slope.
        requested_phase = clamp(requested_phase, phase_ma - 0.08, phase_ma + 0.08)
        phase_ma = bounded_phase(original_phase_ma, requested_phase)
        codes[2] = current_to_code(phase_ma, 2)
        next_reading = take_reading(laser, meter, codes, target_nm)
        if next_reading.power_mw < MIN_VALID_POWER_MW:
            codes[2] = current_to_code(last_phase_sample[0], 2)
            take_reading(laser, meter, codes, target_nm, settle_s=0.12)
            raise RuntimeError("PHASE correction crossed a mode boundary and was reverted")
        reading = next_reading
        history.append(
            {"action": f"phase_{iteration}", "codes": list(codes), "reading": asdict(reading)}
        )

    success = (
        abs(reading.power_mw - target_power_mw) <= power_tolerance_mw
        and abs(reading.wavelength_error_pm) <= wavelength_tolerance_pm
    )
    return {
        "index": index,
        "target_nm": target_nm,
        "success": success,
        "original_codes": original_codes,
        "calibrated_codes": codes if success else original_codes,
        "gain_ma": code_to_current((codes if success else original_codes)[0], 0),
        "soa_ma": code_to_current((codes if success else original_codes)[1], 1),
        "phase_ma": code_to_current((codes if success else original_codes)[2], 2),
        "final": asdict(reading),
        "history": history,
        "failure": None if success else "tolerance_not_reached",
    }


def save_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(20):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def merge_validated_overrides(
    payload: dict,
    override_path: Path,
    wavelengths: list[float],
    original_rows: list[list[int]],
) -> None:
    """Merge separately repeated measurements, but only for the same wavelength table."""
    overrides = json.loads(override_path.read_text(encoding="utf-8"))
    for key, override in overrides.get("results", {}).items():
        index = int(key)
        if not 0 <= index < min(len(wavelengths), len(original_rows)):
            raise ValueError(f"Override index out of range: {index}")
        target_nm = float(override["target_nm"])
        if not math.isclose(target_nm, wavelengths[index], abs_tol=0.0001):
            raise ValueError(
                f"Override {index} belongs to {target_nm:.3f} nm, not {wavelengths[index]:.3f} nm"
            )
        codes = [int(value) for value in override["calibrated_codes"]]
        LaserSerial.validate_codes(codes)
        final = dict(override["final"])
        if abs(float(final["wavelength_error_pm"])) > 5.0:
            raise ValueError(f"Override {index} exceeds the 5 pm wavelength limit")
        if abs(float(final["power_mw"]) - float(payload["target_power_mw"])) > 0.006:
            raise ValueError(f"Override {index} exceeds the optical-power limit")
        payload["results"][key] = {
            "index": index,
            "target_nm": target_nm,
            "success": True,
            "original_codes": original_rows[index],
            "calibrated_codes": codes,
            "gain_ma": code_to_current(codes[0], 0),
            "soa_ma": code_to_current(codes[1], 1),
            "phase_ma": code_to_current(codes[2], 2),
            "final": final,
            "history": [],
            "failure": None,
            "validation": override.get("validation", "repeated_directed_scan"),
        }


def build_outputs(
    payload: dict,
    original_rows: list[list[int]],
    dac_source: Path = DAC_SOURCE,
    checkpoint: Path = CHECKPOINT,
    report: Path = REPORT,
    proposed_c: Path = PROPOSED_C,
) -> tuple[int, int]:
    calibrated_rows = [list(row) for row in original_rows]
    for key, result in payload["results"].items():
        index = int(key)
        if result.get("success") and index < len(calibrated_rows):
            calibrated_rows[index] = list(result["calibrated_codes"])
    save_json(checkpoint, payload)
    save_json(report, payload)
    write_proposed_c(dac_source, calibrated_rows, proposed_c)
    success_count = sum(bool(item.get("success")) for item in payload["results"].values())
    return success_count, len(payload["results"])


def write_proposed_c(original_source: Path, rows: list[list[int]], destination: Path) -> None:
    text = original_source.read_text(encoding="utf-8")
    row_re = re.compile(r"\{\s*([^{}]+?)\s*\},?")
    valid_index = 0

    def replace(match: re.Match) -> str:
        nonlocal valid_index
        fields = [part.strip() for part in match.group(1).split(",")]
        try:
            values = [int(field, 0) for field in fields]
        except ValueError:
            return match.group(0)
        if len(values) == 5 and values[:3] != [0xFFFF, 0xFFFF, 0xFFFF] and valid_index < len(rows):
            row = rows[valid_index]
            valid_index += 1
            return "{" + ", ".join(str(value) for value in row) + "},"
        return match.group(0)

    destination.write_text(row_re.sub(replace, text), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--gpib", default="GPIB0::7::INSTR")
    parser.add_argument("--start", type=int, default=0, help="inclusive table index")
    parser.add_argument("--stop", type=int, default=None, help="exclusive table index")
    parser.add_argument("--target-power", type=float, default=DEFAULT_TARGET_POWER_MW)
    parser.add_argument("--power-tolerance", type=float, default=DEFAULT_POWER_TOLERANCE_MW)
    parser.add_argument(
        "--wavelength-tolerance-pm", type=float, default=DEFAULT_WAVELENGTH_TOLERANCE_PM
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dac-source", type=Path, default=DAC_SOURCE)
    parser.add_argument("--wave-yaml", type=Path, default=WAVE_YAML)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--proposed-c", type=Path, default=PROPOSED_C)
    parser.add_argument(
        "--validated-overrides",
        type=Path,
        default=None,
        help="merge separately repeated measurements after verifying wavelength and limits",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="rebuild report and proposed C table from the checkpoint without opening hardware",
    )
    args = parser.parse_args()

    if not (0.0 < args.target_power <= 2.0):
        raise SystemExit("Refusing an implausible target optical power")

    original_rows = parse_dac_rows(args.dac_source)
    wavelengths = load_wavelengths(args.wave_yaml)
    count = min(len(original_rows), len(wavelengths))
    stop = min(args.stop if args.stop is not None else count, count)
    start = clamp(args.start, 0, stop)
    start = int(start)

    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "meter": None,
        "target_power_mw": args.target_power,
        "power_tolerance_mw": args.power_tolerance,
        "wavelength_tolerance_pm": args.wavelength_tolerance_pm,
        "limits_ma": {
            "gain": LASER_MAX_MA[0],
            "soa_source": LASER_MAX_MA[1],
            "phase": LASER_MAX_MA[2],
            "wave_a": LASER_MAX_MA[3],
            "wave_b": LASER_MAX_MA[4],
        },
        "results": {},
    }
    if (args.resume or args.build_only) and args.checkpoint.exists():
        old = json.loads(args.checkpoint.read_text(encoding="utf-8"))
        if math.isclose(float(old.get("target_power_mw", -1)), args.target_power):
            payload["results"].update(old.get("results", {}))

    if args.validated_overrides is not None:
        merge_validated_overrides(payload, args.validated_overrides, wavelengths, original_rows)

    if args.build_only:
        success_count, result_count = build_outputs(
            payload,
            original_rows,
            args.dac_source,
            args.checkpoint,
            args.report,
            args.proposed_c,
        )
        print(
            f"Build complete: {success_count}/{result_count} successful. "
            f"Report={args.report} proposed_table={args.proposed_c}",
            flush=True,
        )
        return 0

    laser = LaserSerial(args.port)
    meter = AQ6150B(args.gpib)
    payload["meter"] = meter.identity
    park_codes = original_rows[44] if len(original_rows) > 44 else original_rows[0]
    try:
        for index in range(start, stop):
            key = str(index)
            if args.resume and key in payload["results"] and payload["results"][key].get("success"):
                print(f"[{index + 1:03d}/{stop:03d}] resume: already calibrated", flush=True)
                continue
            try:
                result = calibrate_row(
                    laser,
                    meter,
                    index,
                    wavelengths[index],
                    original_rows[index],
                    args.target_power,
                    args.power_tolerance,
                    args.wavelength_tolerance_pm,
                )
            except Exception as error:
                laser.set_codes(original_rows[index])
                result = {
                    "index": index,
                    "target_nm": wavelengths[index],
                    "success": False,
                    "original_codes": original_rows[index],
                    "calibrated_codes": original_rows[index],
                    "failure": str(error),
                }
            payload["results"][key] = result
            save_json(args.checkpoint, payload)
            final = result.get("final", {})
            print(
                f"[{index + 1:03d}/{stop:03d}] {wavelengths[index]:.3f} nm "
                f"success={result['success']} power={final.get('power_mw', float('nan')):.6f} mW "
                f"error={final.get('wavelength_error_pm', float('nan')):+.3f} pm "
                f"SOA={result.get('soa_ma', float('nan')):.3f} mA "
                f"PHASE={result.get('phase_ma', float('nan')):.3f} mA",
                flush=True,
            )
    finally:
        try:
            laser.set_codes(park_codes)
            time.sleep(0.08)
        finally:
            laser.close()
            meter.close()

    success_count, result_count = build_outputs(
        payload,
        original_rows,
        args.dac_source,
        args.checkpoint,
        args.report,
        args.proposed_c,
    )
    print(
        f"Complete: {success_count}/{result_count} successful. "
        f"Report={args.report} proposed_table={args.proposed_c}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
