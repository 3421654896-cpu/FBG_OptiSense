"""Offline train/evaluate/deploy model for the nine-FBG mechanical finger.

The model consumes the strictly causal, three-frame feature vector produced by
``finger_features.CausalFeatureWindow.model_vector``.  A saved bundle is pinned
to CH1, nine gratings, five wavelengths per grating (45 points), the feature
layout, ADC provenance (direct vs firmware RC), and a wavelength-table CRC.
Version-1 bundles did not pin ADC provenance and must be retrained, never
silently migrated by guessing their input semantics. The module contains no UI,
serial, network, or machine-control code.

Area is an optional target.  It is *not* fitted when the training labels cannot
distinguish area (notably a single 8 mm flat indenter with one nominal area).
In that case predictions return ``contact_area_mm2=None`` and retain an audit
reason in the bundle/report instead of publishing a misleading constant.
"""

from __future__ import annotations

import math
import os
import platform
import tempfile
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit

from finger_dataset import (
    DEFAULT_GRATING_COUNT,
    DEFAULT_POINTS_PER_PEAK,
    DEFAULT_SENSOR_CHANNEL,
)
from finger_features import (
    CAUSAL_FRAME_COUNT,
    FEATURE_NAMES,
    CausalFeatureWindow,
    PeakQuality,
)

MODEL_BUNDLE_VERSION = "fbg-finger-localizer/v3"
DEFAULT_POINT_COUNT = DEFAULT_GRATING_COUNT * DEFAULT_POINTS_PER_PEAK
MODEL_FEATURE_DIM = (
    DEFAULT_GRATING_COUNT * len(FEATURE_NAMES) * CAUSAL_FRAME_COUNT
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return None
    return float(np.percentile(array, percentile))


class FingerModelError(RuntimeError):
    """Base exception for training/deployment failures."""


class ModelCompatibilityError(FingerModelError):
    """The runtime acquisition layout does not match the trained bundle."""


class InsufficientGroupedDataError(FingerModelError):
    """A leakage-free grouped train/test split cannot be made."""


class AreaNotIdentifiableError(FingerModelError):
    """Raised only by callers that explicitly require an area model."""


@dataclass(frozen=True)
class FingerRuntimeSignature:
    """Acquisition contract that must match before inference is allowed."""

    table_crc32: int | None
    channel: int = DEFAULT_SENSOR_CHANNEL
    grating_count: int = DEFAULT_GRATING_COUNT
    points_per_peak: int = DEFAULT_POINTS_PER_PEAK
    causal_frame_count: int = CAUSAL_FRAME_COUNT
    feature_names: tuple[str, ...] = FEATURE_NAMES
    adc_value_kind: str = "unknown"

    def __post_init__(self) -> None:
        crc = self.table_crc32
        if crc is not None and not 0 <= int(crc) <= 0xFFFFFFFF:
            raise ValueError("table_crc32 must fit uint32 or be None")
        object.__setattr__(self, "table_crc32", None if crc is None else int(crc))
        object.__setattr__(self, "channel", int(self.channel))
        object.__setattr__(self, "grating_count", int(self.grating_count))
        object.__setattr__(self, "points_per_peak", int(self.points_per_peak))
        object.__setattr__(
            self, "causal_frame_count", int(self.causal_frame_count)
        )
        object.__setattr__(
            self, "feature_names", tuple(str(value) for value in self.feature_names)
        )

    @property
    def point_count(self) -> int:
        return self.grating_count * self.points_per_peak

    @property
    def feature_dim(self) -> int:
        return self.grating_count * len(self.feature_names) * self.causal_frame_count

    def validate_finger_layout(self) -> None:
        problems = []
        if self.adc_value_kind not in {"direct_adc", "firmware_rc_estimate"}:
            problems.append("adc_value_kind is unknown; declare direct ADC or firmware RC estimates")
        if self.channel != DEFAULT_SENSOR_CHANNEL:
            problems.append(f"channel CH{self.channel} (expected CH1)")
        if self.grating_count != DEFAULT_GRATING_COUNT:
            problems.append(
                f"{self.grating_count} gratings (expected {DEFAULT_GRATING_COUNT})"
            )
        if self.points_per_peak != DEFAULT_POINTS_PER_PEAK:
            problems.append(
                f"{self.points_per_peak} points/peak "
                f"(expected {DEFAULT_POINTS_PER_PEAK})"
            )
        if self.causal_frame_count != CAUSAL_FRAME_COUNT:
            problems.append(
                f"{self.causal_frame_count} causal frames "
                f"(expected {CAUSAL_FRAME_COUNT})"
            )
        if self.feature_names != FEATURE_NAMES:
            problems.append("feature-name/order mismatch")
        if self.feature_dim != MODEL_FEATURE_DIM:
            problems.append(
                f"feature dimension {self.feature_dim} (expected {MODEL_FEATURE_DIM})"
            )
        if problems:
            raise ModelCompatibilityError("; ".join(problems))

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["point_count"] = self.point_count
        result["feature_dim"] = self.feature_dim
        return result


@dataclass(frozen=True)
class FingerTrainingSample:
    """One labelled causal window.

    ``event_id`` identifies a complete physical press/release event, never an
    individual frame.  ``session_id`` identifies one acquisition session.
    Splits use one of these group levels and therefore cannot leak adjacent
    frames from an event into both train and test.
    """

    features: Sequence[float]
    session_id: str
    event_id: str
    contact: bool
    model_x_mm: float | None = None
    model_y_mm: float | None = None
    contact_area_mm2: float | None = None
    area_label_source: str | None = None
    area_label_traceable: bool = False
    area_label_independent: bool = False
    area_measurement_id: str | None = None
    area_calibration_id: str | None = None
    indenter_diameter_mm: float | None = None

    def __post_init__(self) -> None:
        vector = np.asarray(self.features, dtype=float).reshape(-1)
        if vector.size != MODEL_FEATURE_DIM:
            raise ValueError(
                f"feature vector has {vector.size} values; expected {MODEL_FEATURE_DIM}"
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("feature vector must contain only finite values")
        if not str(self.session_id).strip() or not str(self.event_id).strip():
            raise ValueError("session_id and event_id must not be empty")
        for name in ("model_x_mm", "model_y_mm", "contact_area_mm2"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite or None")
        if self.contact_area_mm2 is not None and float(self.contact_area_mm2) < 0.0:
            raise ValueError("contact_area_mm2 must be non-negative")
        for name in ("area_label_traceable", "area_label_independent"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        measurement_id = (
            None
            if self.area_measurement_id is None
            else str(self.area_measurement_id).strip()
        )
        calibration_id = (
            None
            if self.area_calibration_id is None
            else str(self.area_calibration_id).strip()
        )
        if self.area_label_traceable and (not measurement_id or not calibration_id):
            raise ValueError(
                "traceable area labels require measurement and calibration IDs"
            )
        diameter = self.indenter_diameter_mm
        if diameter is not None and (
            not math.isfinite(float(diameter)) or float(diameter) <= 0.0
        ):
            raise ValueError("indenter_diameter_mm must be finite and positive")

        object.__setattr__(self, "features", vector)
        object.__setattr__(self, "session_id", str(self.session_id))
        object.__setattr__(self, "event_id", str(self.event_id))
        object.__setattr__(self, "contact", bool(self.contact))
        for name in ("model_x_mm", "model_y_mm", "contact_area_mm2"):
            value = getattr(self, name)
            object.__setattr__(self, name, None if value is None else float(value))
        object.__setattr__(
            self,
            "area_label_source",
            None if self.area_label_source is None else str(self.area_label_source),
        )
        object.__setattr__(self, "area_measurement_id", measurement_id)
        object.__setattr__(self, "area_calibration_id", calibration_id)
        object.__setattr__(
            self,
            "indenter_diameter_mm",
            None if diameter is None else float(diameter),
        )

    @property
    def event_group(self) -> str:
        return f"{self.session_id}::{self.event_id}"

    @classmethod
    def from_window(
        cls,
        window: CausalFeatureWindow,
        *,
        session_id: str,
        event_id: str,
        contact: bool,
        model_x_mm: float | None = None,
        model_y_mm: float | None = None,
        contact_area_mm2: float | None = None,
        area_label_source: str | None = None,
        area_label_traceable: bool = False,
        area_label_independent: bool = False,
        area_measurement_id: str | None = None,
        area_calibration_id: str | None = None,
        indenter_diameter_mm: float | None = None,
    ) -> FingerTrainingSample:
        if not window.ready:
            raise ValueError("three causal frames are required before training")
        return cls(
            window.model_vector(),
            session_id=session_id,
            event_id=event_id,
            contact=contact,
            model_x_mm=model_x_mm,
            model_y_mm=model_y_mm,
            contact_area_mm2=contact_area_mm2,
            area_label_source=area_label_source,
            area_label_traceable=area_label_traceable,
            area_label_independent=area_label_independent,
            area_measurement_id=area_measurement_id,
            area_calibration_id=area_calibration_id,
            indenter_diameter_mm=indenter_diameter_mm,
        )


@dataclass(frozen=True)
class FingerPrediction:
    contact_probability: float
    contact_detected: bool
    model_x_mm: float | None
    model_y_mm: float | None
    contact_area_mm2: float | None
    area_model_available: bool
    area_unavailable_reason: str | None


@dataclass(frozen=True)
class ModelTrainingConfig:
    # Three independently predicted targets mean three tree traversals on a
    # contact frame.  Sixty-four trees retains ensemble robustness while
    # leaving substantially more of the 66.7 ms frame budget for ADC/DAC/UI.
    n_estimators: int = 64
    min_samples_leaf: int = 2
    test_size: float = 0.25
    random_state: int = 20260905
    contact_threshold: float = 0.50
    split_level: str = "auto"  # auto, session, event
    split_candidates: int = 128
    inference_benchmark_repeats: int = 300
    min_position_train_samples: int = 12
    min_position_train_groups: int = 3

    def __post_init__(self) -> None:
        if int(self.n_estimators) < 8:
            raise ValueError("n_estimators must be at least 8")
        if int(self.min_samples_leaf) < 1:
            raise ValueError("min_samples_leaf must be positive")
        if not 0.05 <= float(self.test_size) <= 0.50:
            raise ValueError("test_size must be within 0.05..0.50")
        if not 0.0 < float(self.contact_threshold) < 1.0:
            raise ValueError("contact_threshold must be within 0..1")
        if self.split_level not in {"auto", "session", "event"}:
            raise ValueError("split_level must be auto, session, or event")
        if int(self.inference_benchmark_repeats) < 1:
            raise ValueError("inference_benchmark_repeats must be positive")


@dataclass
class FingerModelBundle:
    """Joblib-serializable inference bundle with a strict acquisition contract."""

    signature: FingerRuntimeSignature
    contact_model: ExtraTreesClassifier
    position_model: ExtraTreesRegressor
    area_model: ExtraTreesRegressor | None
    contact_threshold: float
    area_unavailable_reason: str | None
    training_report: dict[str, Any]
    bundle_version: str = MODEL_BUNDLE_VERSION
    created_utc: str = field(default_factory=_utc_now)
    library_versions: dict[str, str] = field(
        default_factory=lambda: {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        }
    )

    @property
    def area_model_available(self) -> bool:
        return self.area_model is not None

    def _validate_estimators(self) -> None:
        expected = self.signature.feature_dim
        for name, estimator in (
            ("contact", self.contact_model),
            ("position", self.position_model),
            ("area", self.area_model),
        ):
            if estimator is None:
                continue
            actual = getattr(estimator, "n_features_in_", None)
            if actual != expected:
                raise ModelCompatibilityError(
                    f"{name} estimator expects {actual!r} features; "
                    f"bundle signature requires {expected}"
                )
        if getattr(self.position_model, "n_outputs_", None) != 2:
            raise ModelCompatibilityError(
                "position estimator must output model_x_mm and model_y_mm"
            )

    def assert_compatible(
        self,
        runtime: FingerRuntimeSignature,
        *,
        allow_unpinned_table: bool = False,
    ) -> None:
        if self.bundle_version != MODEL_BUNDLE_VERSION:
            raise ModelCompatibilityError(
                f"bundle version {self.bundle_version!r} is unsupported"
            )
        self.signature.validate_finger_layout()
        self._validate_estimators()
        runtime.validate_finger_layout()
        comparisons = (
            ("adc_value_kind", self.signature.adc_value_kind, runtime.adc_value_kind),
            ("channel", self.signature.channel, runtime.channel),
            ("grating_count", self.signature.grating_count, runtime.grating_count),
            (
                "points_per_peak",
                self.signature.points_per_peak,
                runtime.points_per_peak,
            ),
            (
                "causal_frame_count",
                self.signature.causal_frame_count,
                runtime.causal_frame_count,
            ),
            ("feature_names", self.signature.feature_names, runtime.feature_names),
        )
        differences = [
            f"{name}: model={expected!r}, runtime={actual!r}"
            for name, expected, actual in comparisons
            if expected != actual
        ]
        if differences:
            raise ModelCompatibilityError("; ".join(differences))

        expected_crc = self.signature.table_crc32
        runtime_crc = runtime.table_crc32
        if expected_crc is None:
            if not allow_unpinned_table:
                raise ModelCompatibilityError(
                    "model is not pinned to a wavelength-table CRC; deployment refused"
                )
        elif runtime_crc is None:
            raise ModelCompatibilityError(
                f"runtime did not provide table CRC; model expects 0x{expected_crc:08X}"
            )
        elif runtime_crc != expected_crc:
            raise ModelCompatibilityError(
                f"table CRC mismatch: model=0x{expected_crc:08X}, "
                f"runtime=0x{runtime_crc:08X}"
            )

    def _contact_probability(self, matrix: np.ndarray) -> np.ndarray:
        probabilities = self.contact_model.predict_proba(matrix)
        classes = list(self.contact_model.classes_)
        if True not in classes and 1 not in classes:
            raise FingerModelError("contact classifier has no positive class")
        index = classes.index(True) if True in classes else classes.index(1)
        return np.asarray(probabilities[:, index], dtype=float)

    def predict_vector(
        self,
        features: Sequence[float],
        runtime: FingerRuntimeSignature,
        *,
        allow_unpinned_table: bool = False,
    ) -> FingerPrediction:
        self.assert_compatible(
            runtime, allow_unpinned_table=allow_unpinned_table
        )
        vector = np.asarray(features, dtype=float).reshape(-1)
        if vector.size != self.signature.feature_dim:
            raise ModelCompatibilityError(
                f"feature vector has {vector.size} values; "
                f"model expects {self.signature.feature_dim}"
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("inference features must be finite")
        matrix = vector.reshape(1, -1)
        probability = float(self._contact_probability(matrix)[0])
        detected = probability >= self.contact_threshold
        if detected:
            xy = np.asarray(self.position_model.predict(matrix)[0], dtype=float)
            x_value, y_value = float(xy[0]), float(xy[1])
            area_value = (
                None
                if self.area_model is None
                else max(0.0, float(self.area_model.predict(matrix)[0]))
            )
        else:
            x_value = y_value = area_value = None
        return FingerPrediction(
            contact_probability=min(1.0, max(0.0, probability)),
            contact_detected=detected,
            model_x_mm=x_value,
            model_y_mm=y_value,
            contact_area_mm2=area_value,
            area_model_available=self.area_model is not None,
            area_unavailable_reason=(
                None if self.area_model is not None else self.area_unavailable_reason
            ),
        )

    def predict_window(
        self,
        window: CausalFeatureWindow,
        runtime: FingerRuntimeSignature,
    ) -> FingerPrediction:
        if not window.ready:
            raise ValueError("inference requires a full three-frame causal window")
        fatal = int(
            PeakQuality.NONFINITE_SAMPLE
            | PeakQuality.ADC_OUT_OF_RANGE
            | PeakQuality.NONMONOTONIC_WAVELENGTH
            | PeakQuality.LOW_AMPLITUDE
            | PeakQuality.BASELINE_INVALID
            | PeakQuality.FRAME_CRC_FAILED
            | PeakQuality.TABLE_CRC_MISMATCH
            | PeakQuality.TEMPLATE_FIT_REJECTED
        )
        history_masks = np.asarray(window.quality_masks, dtype=np.uint16)
        expected_quality_shape = (CAUSAL_FRAME_COUNT, DEFAULT_GRATING_COUNT)
        if history_masks.shape != expected_quality_shape:
            raise ModelCompatibilityError(
                f"quality mask shape is {history_masks.shape}; "
                f"expected {expected_quality_shape}"
            )
        valid_peaks = int(np.count_nonzero((history_masks & fatal) == 0))
        expected_valid = DEFAULT_GRATING_COUNT * CAUSAL_FRAME_COUNT
        if valid_peaks < expected_valid:
            raise FingerModelError(
                f"only {valid_peaks}/{expected_valid} causal peak observations are "
                "valid; inference refused"
            )
        return self.predict_vector(window.model_vector(), runtime)

    def save(self, destination: os.PathLike[str] | str) -> Path:
        """Atomically save the bundle.  Load only bundles from trusted sources."""

        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        try:
            joblib.dump(self, temporary, compress=3)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    @classmethod
    def load(cls, source: os.PathLike[str] | str) -> FingerModelBundle:
        """Load a trusted joblib bundle and validate its schema/layout."""

        bundle = joblib.load(Path(source))
        if not isinstance(bundle, cls):
            raise ModelCompatibilityError("joblib file is not a FingerModelBundle")
        if bundle.bundle_version != MODEL_BUNDLE_VERSION:
            raise ModelCompatibilityError(
                f"unsupported model bundle version: {bundle.bundle_version!r}"
            )
        bundle.signature.validate_finger_layout()
        bundle._validate_estimators()
        return bundle


def _split_is_valid(
    samples: Sequence[FingerTrainingSample], train: np.ndarray, test: np.ndarray
) -> bool:
    train_contact = {samples[index].contact for index in train}
    test_contact = {samples[index].contact for index in test}
    if train_contact != {False, True} or test_contact != {False, True}:
        return False
    train_position = sum(
        samples[index].contact
        and samples[index].model_x_mm is not None
        and samples[index].model_y_mm is not None
        for index in train
    )
    test_position = sum(
        samples[index].contact
        and samples[index].model_x_mm is not None
        and samples[index].model_y_mm is not None
        for index in test
    )
    return train_position >= 3 and test_position >= 1


def _try_grouped_split(
    samples: Sequence[FingerTrainingSample],
    groups: np.ndarray,
    config: ModelTrainingConfig,
) -> tuple[np.ndarray, np.ndarray] | None:
    if len(np.unique(groups)) < 2:
        return None
    labels = np.asarray([sample.contact for sample in samples], dtype=bool)
    splitter = GroupShuffleSplit(
        n_splits=config.split_candidates,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for train, test in splitter.split(np.zeros(len(samples)), labels, groups):
        if not _split_is_valid(samples, train, test):
            continue
        train_groups = set(groups[train])
        test_groups = set(groups[test])
        if train_groups & test_groups:
            raise AssertionError("GroupShuffleSplit leaked a group")
        overall_ratio = float(np.mean(labels))
        score = (
            abs(len(test) / len(samples) - config.test_size)
            + abs(float(np.mean(labels[train])) - overall_ratio)
            + abs(float(np.mean(labels[test])) - overall_ratio)
        )
        if best is None or score < best[0]:
            best = score, train, test
    return None if best is None else (best[1], best[2])


def _grouped_split(
    samples: Sequence[FingerTrainingSample], config: ModelTrainingConfig
) -> tuple[np.ndarray, np.ndarray, str, np.ndarray]:
    session_groups = np.asarray([sample.session_id for sample in samples], dtype=object)
    event_groups = np.asarray([sample.event_group for sample in samples], dtype=object)
    levels = (
        ("session", session_groups),
        ("event", event_groups),
    )
    if config.split_level != "auto":
        levels = tuple(item for item in levels if item[0] == config.split_level)
    for level, groups in levels:
        result = _try_grouped_split(samples, groups, config)
        if result is not None:
            return result[0], result[1], level, groups
    raise InsufficientGroupedDataError(
        "cannot form a leakage-free grouped holdout containing contact and "
        "no-contact samples in both train and test; collect more complete "
        "press events/sessions"
    )


def assess_area_identifiability(
    samples: Sequence[FingerTrainingSample], indices: Sequence[int]
) -> tuple[bool, str | None, np.ndarray]:
    """Conservatively decide whether true area has distinguishable labels."""

    labelled = np.asarray(
        [
            index
            for index in indices
            if samples[index].contact
            and samples[index].contact_area_mm2 is not None
            and samples[index].area_label_source is not None
        ],
        dtype=int,
    )
    if len(labelled) < 12:
        return (
            False,
            "fewer than 12 contact frames have traceable area labels",
            labelled,
        )

    untraceable = [
        index
        for index in labelled
        if not (
            samples[index].area_label_traceable
            and samples[index].area_label_independent
            and samples[index].area_measurement_id
            and samples[index].area_calibration_id
        )
    ]
    if untraceable:
        return (
            False,
            (
                "area values exist but are not independently linked to a "
                "traceable measurement and calibration record"
            ),
            labelled,
        )

    # Evaluate independent press-event medians, not per-frame jitter.  This
    # prevents sensor noise around one fixed label from masquerading as area
    # variation.
    group_values: dict[str, list[float]] = {}
    diameters = []
    for index in labelled:
        sample = samples[index]
        group_values.setdefault(sample.event_group, []).append(
            float(sample.contact_area_mm2)
        )
        if sample.indenter_diameter_mm is not None:
            diameters.append(float(sample.indenter_diameter_mm))
    event_areas = np.asarray(
        [float(np.median(values)) for values in group_values.values()], dtype=float
    )
    distinct_areas = np.unique(np.round(event_areas, decimals=3))
    if distinct_areas.size < 3:
        diameter_text = ""
        if diameters and np.unique(np.round(diameters, decimals=3)).size == 1:
            diameter_text = f" (single {diameters[0]:g} mm indenter)"
        return (
            False,
            (
                "contact events contain fewer than three distinguishable area "
                f"labels{diameter_text}; true area is not identifiable"
            ),
            labelled,
        )
    span = float(np.ptp(event_areas))
    median = max(float(np.median(event_areas)), 1e-9)
    if span < max(0.5, 0.05 * median):
        return (
            False,
            "area-label span is too small to distinguish from labelling noise",
            labelled,
        )
    sources = {samples[index].area_label_source.lower() for index in labelled}
    trusted = sources <= {
        "measured",
        "calibrated",
        "pressure_film",
        "camera_contact_patch_calibrated",
        "known_indenter_contact",
    }
    if not trusted:
        return (
            False,
            (
                "area labels vary but lack measured/calibrated contact-area "
                "provenance; nominal tool-face area is not treated as true contact area"
            ),
            labelled,
        )
    return True, None, labelled


def _model_options(config: ModelTrainingConfig) -> dict[str, Any]:
    # One inference sample at a time is faster and more deterministic with
    # n_jobs=1; tree construction remains acceptably quick for this small input.
    return {
        "n_estimators": config.n_estimators,
        "min_samples_leaf": config.min_samples_leaf,
        "max_features": "sqrt",
        "random_state": config.random_state,
        "n_jobs": 1,
    }


def _evaluate_and_benchmark(
    bundle: FingerModelBundle,
    samples: Sequence[FingerTrainingSample],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    split_level: str,
    split_groups: np.ndarray,
    config: ModelTrainingConfig,
) -> dict[str, Any]:
    x_test = np.stack([samples[index].features for index in test_indices])
    y_test = np.asarray([samples[index].contact for index in test_indices], dtype=bool)
    probability = bundle._contact_probability(x_test)
    predicted = probability >= bundle.contact_threshold
    contact_metrics = {
        "sample_count": len(test_indices),
        "f1": float(f1_score(y_test, predicted, zero_division=0)),
        "precision": float(precision_score(y_test, predicted, zero_division=0)),
        "recall": float(recall_score(y_test, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, predicted)),
        "threshold": float(bundle.contact_threshold),
    }

    position_indices = np.asarray(
        [
            index
            for index in test_indices
            if samples[index].contact
            and samples[index].model_x_mm is not None
            and samples[index].model_y_mm is not None
        ],
        dtype=int,
    )
    position_errors: list[float] = []
    if len(position_indices):
        matrix = np.stack([samples[index].features for index in position_indices])
        truth = np.asarray(
            [
                (samples[index].model_x_mm, samples[index].model_y_mm)
                for index in position_indices
            ],
            dtype=float,
        )
        estimate = np.asarray(bundle.position_model.predict(matrix), dtype=float)
        position_errors = np.linalg.norm(estimate - truth, axis=1).tolist()
    position_metrics = {
        "sample_count": len(position_errors),
        "median_error_mm": _percentile(position_errors, 50),
        "p95_error_mm": _percentile(position_errors, 95),
    }

    area_errors: list[float] = []
    if bundle.area_model is not None:
        area_indices = np.asarray(
            [
                index
                for index in test_indices
                if samples[index].contact
                and samples[index].contact_area_mm2 is not None
            ],
            dtype=int,
        )
        if len(area_indices):
            matrix = np.stack([samples[index].features for index in area_indices])
            truth = np.asarray(
                [samples[index].contact_area_mm2 for index in area_indices],
                dtype=float,
            )
            estimate = np.maximum(
                0.0, np.asarray(bundle.area_model.predict(matrix), dtype=float)
            )
            area_errors = np.abs(estimate - truth).tolist()
    area_metrics = {
        "enabled": bundle.area_model is not None,
        "unavailable_reason": bundle.area_unavailable_reason,
        "sample_count": len(area_errors),
        "mae_mm2": None if not area_errors else float(np.mean(area_errors)),
        "p95_abs_error_mm2": _percentile(area_errors, 95),
    }

    # Benchmark the exact single-window deployment path without compatibility
    # checking overhead.  Compatibility is checked once when a stream starts.
    benchmark_vectors = x_test
    repetitions = config.inference_benchmark_repeats
    for index in range(min(10, repetitions)):
        matrix = benchmark_vectors[index % len(benchmark_vectors)].reshape(1, -1)
        bundle._contact_probability(matrix)
        bundle.position_model.predict(matrix)
        if bundle.area_model is not None:
            bundle.area_model.predict(matrix)
    durations_ms = []
    for index in range(repetitions):
        matrix = benchmark_vectors[index % len(benchmark_vectors)].reshape(1, -1)
        start = time.perf_counter_ns()
        bundle._contact_probability(matrix)
        bundle.position_model.predict(matrix)
        if bundle.area_model is not None:
            bundle.area_model.predict(matrix)
        durations_ms.append((time.perf_counter_ns() - start) / 1_000_000.0)
    mean_ms = float(np.mean(durations_ms))
    inference_metrics = {
        "repetitions": repetitions,
        "mean_ms": mean_ms,
        "median_ms": _percentile(durations_ms, 50),
        "p95_ms": _percentile(durations_ms, 95),
        "p99_ms": _percentile(durations_ms, 99),
        "max_ms": max(durations_ms),
        "estimated_inferences_per_second": 1000.0 / mean_ms,
        "meets_15_hz_inference_budget": bool(
            _percentile(durations_ms, 99) <= 1000.0 / 15.0
        ),
        "budget_ms_at_15_hz": 1000.0 / 15.0,
        "scope": "model inference only; acquisition and UI latency are separate",
    }

    train_groups = sorted({str(value) for value in split_groups[train_indices]})
    test_groups = sorted({str(value) for value in split_groups[test_indices]})
    split_report = {
        "level": split_level,
        "train_sample_count": len(train_indices),
        "test_sample_count": len(test_indices),
        "train_group_count": len(train_groups),
        "test_group_count": len(test_groups),
        "train_groups": train_groups,
        "test_groups": test_groups,
        "group_overlap": sorted(set(train_groups) & set(test_groups)),
        "random_frame_split_used": False,
    }
    return {
        "contact": contact_metrics,
        "position": position_metrics,
        "area": area_metrics,
        "inference_timing": inference_metrics,
        "split": split_report,
    }


def train_finger_model(
    samples: Iterable[FingerTrainingSample],
    signature: FingerRuntimeSignature,
    *,
    config: ModelTrainingConfig | None = None,
    require_area_model: bool = False,
) -> tuple[FingerModelBundle, dict[str, Any]]:
    """Train contact, continuous XY, and optionally identifiable contact area."""

    signature.validate_finger_layout()
    config = config or ModelTrainingConfig()
    records = tuple(samples)
    if len(records) < 24:
        raise InsufficientGroupedDataError(
            "at least 24 labelled causal windows are required"
        )
    train_indices, test_indices, split_level, groups = _grouped_split(records, config)
    x_train = np.stack([records[index].features for index in train_indices])
    y_train = np.asarray(
        [records[index].contact for index in train_indices], dtype=bool
    )
    options = _model_options(config)
    contact_model = ExtraTreesClassifier(
        **options, class_weight="balanced"
    ).fit(x_train, y_train)

    position_indices = np.asarray(
        [
            index
            for index in train_indices
            if records[index].contact
            and records[index].model_x_mm is not None
            and records[index].model_y_mm is not None
        ],
        dtype=int,
    )
    position_groups = {records[index].event_group for index in position_indices}
    if (
        len(position_indices) < config.min_position_train_samples
        or len(position_groups) < config.min_position_train_groups
    ):
        raise InsufficientGroupedDataError(
            "not enough independently grouped contact samples with model_x/model_y "
            "labels to train continuous position"
        )
    position_x = np.stack([records[index].features for index in position_indices])
    position_y = np.asarray(
        [
            (records[index].model_x_mm, records[index].model_y_mm)
            for index in position_indices
        ],
        dtype=float,
    )
    position_model = ExtraTreesRegressor(**options).fit(position_x, position_y)

    area_ok, area_reason, area_indices = assess_area_identifiability(
        records, train_indices
    )
    area_model = None
    if area_ok:
        area_x = np.stack([records[index].features for index in area_indices])
        area_y = np.asarray(
            [records[index].contact_area_mm2 for index in area_indices], dtype=float
        )
        area_model = ExtraTreesRegressor(**options).fit(area_x, area_y)
    elif require_area_model:
        raise AreaNotIdentifiableError(area_reason or "area is not identifiable")

    bundle = FingerModelBundle(
        signature=signature,
        contact_model=contact_model,
        position_model=position_model,
        area_model=area_model,
        contact_threshold=config.contact_threshold,
        area_unavailable_reason=area_reason,
        training_report={},
    )
    metrics = _evaluate_and_benchmark(
        bundle,
        records,
        train_indices,
        test_indices,
        split_level,
        groups,
        config,
    )
    report = {
        "bundle_version": MODEL_BUNDLE_VERSION,
        "created_utc": bundle.created_utc,
        "signature": signature.to_dict(),
        "dataset": {
            "sample_count": len(records),
            "session_count": len({sample.session_id for sample in records}),
            "event_count": len({sample.event_group for sample in records}),
            "contact_sample_count": sum(sample.contact for sample in records),
        },
        "metrics": metrics,
        "area_policy": {
            "trained": area_model is not None,
            "reason_if_not_trained": area_reason,
            "prediction_when_unavailable": "null",
            "constant_area_model_forbidden": True,
        },
        "training_config": asdict(config),
        "library_versions": dict(bundle.library_versions),
    }
    if metrics["split"]["group_overlap"]:
        raise AssertionError("group leakage detected after training")
    bundle.training_report = report
    return bundle, report


def save_finger_model(
    bundle: FingerModelBundle, destination: os.PathLike[str] | str
) -> Path:
    return bundle.save(destination)


def load_finger_model(source: os.PathLike[str] | str) -> FingerModelBundle:
    return FingerModelBundle.load(source)


__all__ = [
    "DEFAULT_POINT_COUNT",
    "MODEL_BUNDLE_VERSION",
    "MODEL_FEATURE_DIM",
    "AreaNotIdentifiableError",
    "FingerModelBundle",
    "FingerModelError",
    "FingerPrediction",
    "FingerRuntimeSignature",
    "FingerTrainingSample",
    "InsufficientGroupedDataError",
    "ModelCompatibilityError",
    "ModelTrainingConfig",
    "assess_area_identifiability",
    "load_finger_model",
    "save_finger_model",
    "train_finger_model",
]
