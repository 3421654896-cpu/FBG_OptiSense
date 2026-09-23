"""Causal realtime inference for the locked all-fresh 45-point CH1 stream.

The classifier is the exact rule selected on r1/r2 and accepted once on the
untouched r3 round.  It reports a calibrated Mach3 work-plane grid position at
the validated 0.30 mm indentation condition.  The 31 Hz sampling-rate gate is
reported separately from physical 15 Hz response, which is not yet verified.
The known 8 mm circular tool face is reported as a nominal geometric area; it
is kept separate from the unknown deformation-dependent physical contact patch.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np

from evaluate_coverage_position import _feature_variants


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = (
    HERE / "artifacts" / "ch1_fast_fullmap_position_extratrees_v2.joblib"
)
POINT_COUNT = 45
SENSOR_COUNT = 9
CADENCE_WINDOW = 32
FIFTEEN_HZ_NYQUIST_INTERVAL_US = 1_000_000.0 / 30.0
PRESS_FACE_DIAMETER_MM = 8.0
NOMINAL_TOOL_FACE_AREA_MM2 = math.pi * (PRESS_FACE_DIAMETER_MM / 2.0) ** 2


@dataclass(frozen=True)
class FastFullMapRealtimeResult:
    sequence: int
    board_time_us: int
    state: str
    baseline_ready: bool
    baseline_frozen: bool
    contact_detected: bool
    predicted_point: int | None
    provisional_x_mm: float | None
    provisional_y_mm: float | None
    response_rms_codes: float | None
    activation_threshold_codes: float
    response_strength: tuple[float, ...]
    first_delta_codes: tuple[float | None, ...]
    confidence: float
    confidence_kind: str
    cadence_hz: float | None
    sampling_gate_for_15hz: bool
    position_coordinate_frame: str = "mach3_work_xy_mm"
    position_accuracy_verified: bool = False
    static_position_accuracy_validated_at_0p30mm: bool = True
    physical_15hz_verified: bool = False
    physical_contact_area_mm2: None = None
    nominal_tool_face_area_mm2: float = NOMINAL_TOOL_FACE_AREA_MM2
    press_footprint_area_mm2: float = 0.0
    press_footprint_basis: str = "detected_contact_x_fixed_8mm_tool_face"
    contact_area_verified: bool = False
    quality_flags: tuple[str, ...] = ()
    model_schema: str = "ch1_fast_fullmap_position_model_v1"
    static_validation_trial_majority_exact_percent: float = 100.0
    static_validation_frame_exact_percent: float = 100.0
    static_validation_frame_within_4mm_percent: float = 100.0

    def to_dict(self) -> dict:
        return {
            "protocol": "F45",
            "sequence": self.sequence,
            "board_time_us": self.board_time_us,
            "state": self.state,
            "baseline_ready": self.baseline_ready,
            "baseline_frozen": self.baseline_frozen,
            "contact_detected": self.contact_detected,
            "predicted_point": self.predicted_point,
            "position": {
                "x_mm": self.provisional_x_mm,
                "y_mm": self.provisional_y_mm,
                "coordinate_frame": self.position_coordinate_frame,
                "accuracy_verified": self.position_accuracy_verified,
                "static_grid_validated_at_0p30mm": (
                    self.static_position_accuracy_validated_at_0p30mm
                ),
            },
            "response_rms_codes": self.response_rms_codes,
            "activation_threshold_codes": self.activation_threshold_codes,
            "response_strength": list(self.response_strength),
            "first_delta_codes": list(self.first_delta_codes),
            "confidence": self.confidence,
            "confidence_kind": self.confidence_kind,
            "model_schema": self.model_schema,
            "static_validation": {
                "trial_majority_exact_percent": (
                    self.static_validation_trial_majority_exact_percent
                ),
                "frame_exact_percent": self.static_validation_frame_exact_percent,
                "frame_within_4mm_percent": (
                    self.static_validation_frame_within_4mm_percent
                ),
            },
            "cadence_hz": self.cadence_hz,
            "sampling_gate_for_15hz": self.sampling_gate_for_15hz,
            "physical_15hz_verified": self.physical_15hz_verified,
            "physical_contact_area_mm2": None,
            "nominal_tool_face_area_mm2": self.nominal_tool_face_area_mm2,
            "press_footprint_area_mm2": self.press_footprint_area_mm2,
            "press_footprint_basis": self.press_footprint_basis,
            "press_face_diameter_mm": PRESS_FACE_DIAMETER_MM,
            "contact_area_verified": False,
            "quality_flags": list(self.quality_flags),
        }


class FastFullMapRealtimeLocalizer:
    """Learn one no-contact baseline, freeze it, then classify every frame."""

    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path).resolve()
        if self.model_path.suffix.lower() == ".joblib":
            payload = joblib.load(self.model_path)
        else:
            payload = json.loads(self.model_path.read_text("utf-8"))
        self.model_schema = str(payload.get("schema"))
        if self.model_schema not in {
            "ch1_fast_fullmap_position_model_v1",
            "ch1_fast_fullmap_position_model_v2",
        }:
            raise ValueError("unsupported fast full-map model")
        protocol = payload.get("protocol") or {}
        if (
            protocol.get("name") != "F45"
            or tuple(protocol.get("route_rows") or ()) != tuple(range(POINT_COUNT))
            or int(protocol.get("spacing_us", -1)) != 50
            or tuple(protocol.get("feedback_selectors") or ()) != (0, 2)
        ):
            raise ValueError("model protocol does not match the qualified F45 route")
        validation = payload.get("static_validation") or {}
        if not (validation.get("gate") or {}).get("passed"):
            raise ValueError("model has no passed untouched static validation")
        classifier = payload.get("classifier") or {}
        self._tree_classifier = None
        self.coordinates_by_point: dict[int, tuple[float, float]] = {}
        if self.model_schema == "ch1_fast_fullmap_position_model_v1":
            self.feature = str(classifier.get("feature"))
            if not self.feature.startswith("first_"):
                raise ValueError("realtime F45 model must be based on first-code samples")
            self.feature_variant = self.feature.removeprefix("first_")
            self.scaling = str(classifier.get("scaling"))
            self.metric = str(classifier.get("metric"))
            if self.scaling != "robust" or self.metric not in {"euclidean", "cosine"}:
                raise ValueError("unsupported locked classifier preprocessing")
            self.centre = np.asarray(classifier.get("centre"), dtype=float)
            self.scale = np.asarray(classifier.get("scale"), dtype=float)
            prototypes = classifier.get("prototypes") or []
            self.prototype_vectors = np.asarray(
                [item["vector"] for item in prototypes], dtype=float
            )
            feature_width = int(self.prototype_vectors.shape[1])
            if (
                self.prototype_vectors.shape[0] != 34
                or self.centre.shape != (feature_width,)
                or self.scale.shape != (feature_width,)
                or not np.all(np.isfinite(self.prototype_vectors))
                or not np.all(np.isfinite(self.centre))
                or not np.all(np.isfinite(self.scale))
                or np.any(self.scale <= 0.0)
            ):
                raise ValueError("malformed realtime prototype matrix")
            self.prototype_points = np.asarray(
                [int(item["point_index"]) for item in prototypes], dtype=int
            )
            self.prototype_xy = np.asarray(
                [item["work_xy_mm"] for item in prototypes], dtype=float
            )
            if (
                set(self.prototype_points.tolist()) != set(range(1, 18))
                or self.prototype_xy.shape != (34, 2)
                or not np.all(np.isfinite(self.prototype_xy))
            ):
                raise ValueError("model does not contain two physical prototypes per point")
            for point in range(1, 18):
                if int(np.count_nonzero(self.prototype_points == point)) != 2:
                    raise ValueError("model must contain two prototypes for every point")
        else:
            feature = payload.get("feature") or {}
            if feature != {
                "name": "pair_transition_delta45",
                "dimensions": 135,
                "preprocessing": "none",
            }:
                raise ValueError("runtime model does not use the locked 135-D feature")
            self.feature = str(feature["name"])
            self._tree_classifier = classifier
            if (
                not hasattr(classifier, "predict")
                or not hasattr(classifier, "predict_proba")
                or tuple(int(value) for value in classifier.classes_)
                != tuple(range(1, 18))
                or int(getattr(classifier, "n_features_in_", -1)) != 135
            ):
                raise ValueError("malformed ExtraTrees runtime classifier")
            # One observation arrives per frame. Parallel prediction costs
            # more than tree traversal and can create a UI backlog.
            self._tree_classifier.n_jobs = 1
            self._tree_classes = np.asarray(classifier.classes_, dtype=int)
            self._tree_arrays = tuple(
                (
                    estimator.tree_.children_left,
                    estimator.tree_.children_right,
                    estimator.tree_.feature,
                    estimator.tree_.threshold,
                    estimator.tree_.value[:, 0, :],
                )
                for estimator in classifier.estimators_
            )
            if len(self._tree_arrays) != 400:
                raise ValueError("runtime model must contain all 400 locked trees")
            raw_coordinates = payload.get("coordinates_by_point") or {}
            self.coordinates_by_point = {
                int(point): tuple(float(value) for value in xy)
                for point, xy in raw_coordinates.items()
            }
            if (
                set(self.coordinates_by_point) != set(range(1, 18))
                or any(len(xy) != 2 or not np.isfinite(xy).all() for xy in self.coordinates_by_point.values())
            ):
                raise ValueError("runtime model has invalid calibrated coordinates")
        self.static_validation = dict(validation)
        baseline = payload.get("baseline") or {}
        self.minimum_warmup_frames = int(baseline.get("minimum_laser_warmup_frames", -1))
        self.baseline_frames = int(baseline.get("no_contact_frames", -1))
        self.activation_threshold_codes = float(
            (payload.get("activation") or {}).get("threshold_codes", math.nan)
        )
        if (
            self.minimum_warmup_frames != 16
            or self.baseline_frames != 64
            or not math.isfinite(self.activation_threshold_codes)
            or self.activation_threshold_codes <= 0.0
        ):
            raise ValueError("invalid baseline or activation contract")
        self.reset()

    def reset(self) -> None:
        self._baseline_samples: list[np.ndarray] = []
        self._baseline: np.ndarray | None = None
        self._baseline_second: np.ndarray | None = None
        self._baseline_sigma: np.ndarray | None = None
        self._last_sequence: int | None = None
        self._last_board_time_us: int | None = None
        self._intervals_us: deque[int] = deque(maxlen=CADENCE_WINDOW)

    @property
    def baseline_ready(self) -> bool:
        return self._baseline is not None

    @property
    def baseline_frozen(self) -> bool:
        return self._baseline is not None

    def _validate_frame(
        self, frame: Mapping
    ) -> tuple[int, int, np.ndarray, np.ndarray]:
        sequence = int(frame["sequence"])
        board_time_us = int(frame["cycle_start_us"])
        if self._last_sequence is not None and sequence != self._last_sequence + 1:
            self.reset()
            raise ValueError("missing, duplicated or reordered realtime F45 frame")
        records = list(frame.get("records") or ())
        if (
            len(records) != POINT_COUNT
            or tuple(int(item["row"]) for item in records) != tuple(range(POINT_COUNT))
            or int(frame.get("spacing_us", -1)) != 50
            or tuple(frame.get("feedback_selectors", (0, 2))) != (0, 2)
        ):
            self.reset()
            raise ValueError("realtime frame does not match all-fresh F45")
        values = np.asarray([item["first_code"] for item in records], dtype=float)
        if values.shape != (POINT_COUNT,) or not np.all(np.isfinite(values)):
            self.reset()
            raise ValueError("invalid F45 first-code vector")
        if self.model_schema == "ch1_fast_fullmap_position_model_v2":
            second = np.asarray(
                [item.get("second_code", math.nan) for item in records], dtype=float
            )
            if second.shape != (POINT_COUNT,) or not np.all(np.isfinite(second)):
                self.reset()
                raise ValueError("v2 F45 frame requires a physical second-code vector")
            if not bool(frame.get("second_code_is_physical_sample", True)):
                self.reset()
                raise ValueError("v2 F45 frame second-code samples are not physical")
        else:
            second = values.copy()
        if self._last_board_time_us is not None:
            interval = board_time_us - self._last_board_time_us
            if interval <= 0:
                self.reset()
                raise ValueError("nonmonotonic realtime board clock")
            self._intervals_us.append(interval)
        self._last_sequence = sequence
        self._last_board_time_us = board_time_us
        return sequence, board_time_us, values, second

    def _cadence(self) -> tuple[float | None, bool]:
        if len(self._intervals_us) < CADENCE_WINDOW:
            return None, False
        intervals = np.asarray(self._intervals_us, dtype=float)
        return 1_000_000.0 / float(np.median(intervals)), bool(
            float(np.max(intervals)) <= FIFTEEN_HZ_NYQUIST_INTERVAL_US
        )

    def _predict_extra_trees(self, feature: np.ndarray) -> tuple[int, float]:
        """Exact single-row ExtraTrees vote without sklearn batch overhead."""

        vector = np.asarray(feature, dtype=float).reshape(-1)
        if vector.shape != (135,) or not np.all(np.isfinite(vector)):
            raise ValueError("invalid 135-D realtime feature")
        probabilities = np.zeros(len(self._tree_classes), dtype=float)
        for left, right, indices, thresholds, values in self._tree_arrays:
            node = 0
            while left[node] != right[node]:
                node = (
                    int(left[node])
                    if vector[int(indices[node])] <= thresholds[node]
                    else int(right[node])
                )
            leaf = np.asarray(values[node], dtype=float)
            total = float(np.sum(leaf))
            if total > 0.0:
                probabilities += leaf / total
        probabilities /= float(len(self._tree_arrays))
        winner = int(np.argmax(probabilities))
        ordered = np.sort(probabilities)
        return int(self._tree_classes[winner]), float(ordered[-1] - ordered[-2])

    def update(self, frame: Mapping) -> FastFullMapRealtimeResult:
        sequence, board_time_us, values, second = self._validate_frame(frame)
        cadence_hz, sampling_gate = self._cadence()
        flags: list[str] = []
        response_rms = None
        response_strength = np.zeros(SENSOR_COUNT, dtype=float)
        delta_tuple: tuple[float | None, ...] = (None,) * POINT_COUNT
        contact = False
        predicted_point = None
        x_mm = y_mm = None
        confidence = 0.0
        confidence_kind = (
            "extratrees_probability_margin"
            if self.model_schema == "ch1_fast_fullmap_position_model_v2"
            else "nearest_label_distance_margin_not_probability"
        )

        if not bool(frame.get("minimum_warmup_complete", False)):
            state = "minimum_laser_warmup"
        elif self._baseline is None:
            self._baseline_samples.append(np.stack((values, second)))
            state = "learning_no_contact_baseline"
            if len(self._baseline_samples) == self.baseline_frames:
                matrix = np.stack(self._baseline_samples)
                self._baseline = np.median(matrix[:, 0, :], axis=0)
                self._baseline_second = np.median(matrix[:, 1, :], axis=0)
                self._baseline_sigma = 1.4826 * np.median(
                    np.abs(matrix[:, 1, :] - self._baseline_second), axis=0
                )
                state = "baseline_ready_and_frozen"
        else:
            state = "contact_monitoring"
            delta = values - self._baseline
            delta_tuple = tuple(float(value) for value in delta)
            segment_rms = np.sqrt(np.mean(delta.reshape(SENSOR_COUNT, 5) ** 2, axis=1))
            response_rms = float(np.sqrt(np.mean(delta**2)))
            maximum = float(np.max(segment_rms))
            if maximum > 0.0:
                response_strength = np.clip(segment_rms / maximum, 0.0, 1.0)
            contact = response_rms >= self.activation_threshold_codes
            if contact:
                if self.model_schema == "ch1_fast_fullmap_position_model_v2":
                    first_delta = values - self._baseline
                    second_delta = second - self._baseline_second
                    transition = (second - values) - (
                        self._baseline_second - self._baseline
                    )
                    feature = np.concatenate(
                        (first_delta, second_delta, transition)
                    )
                    predicted_point, confidence = self._predict_extra_trees(feature)
                    x_mm, y_mm = self.coordinates_by_point[predicted_point]
                    confidence_kind = "extratrees_probability_margin"
                else:
                    feature = _feature_variants(
                        self._baseline, values, self._baseline_sigma
                    )[self.feature_variant]
                    vector = (feature - self.centre) / self.scale
                    if self.metric == "cosine":
                        vector /= max(float(np.linalg.norm(vector)), 1e-12)
                        distances = 1.0 - self.prototype_vectors @ vector
                    else:
                        distances = np.linalg.norm(
                            self.prototype_vectors - vector, axis=1
                        )
                    winner = int(np.argmin(distances))
                    predicted_point = int(self.prototype_points[winner])
                    x_mm, y_mm = (float(value) for value in self.prototype_xy[winner])
                    label_distances = sorted(
                        min(
                            float(distances[index])
                            for index in np.flatnonzero(self.prototype_points == point)
                        )
                        for point in range(1, 18)
                    )
                    confidence = max(
                        0.0,
                        min(
                            1.0,
                            (label_distances[1] - label_distances[0])
                            / max(label_distances[1], 1e-12),
                        ),
                    )
                    confidence_kind = "nearest_label_distance_margin_not_probability"
            else:
                flags.append("below_0p30mm_training_activation_threshold")
                confidence_kind = (
                    "extratrees_probability_margin"
                    if self.model_schema == "ch1_fast_fullmap_position_model_v2"
                    else "nearest_label_distance_margin_not_probability"
                )

        if not sampling_gate:
            flags.append("15hz_sampling_gate_not_ready_or_failed")
        return FastFullMapRealtimeResult(
            sequence=sequence,
            board_time_us=board_time_us,
            state=state,
            baseline_ready=self.baseline_ready,
            baseline_frozen=self.baseline_frozen,
            contact_detected=contact,
            predicted_point=predicted_point,
            provisional_x_mm=x_mm,
            provisional_y_mm=y_mm,
            response_rms_codes=response_rms,
            activation_threshold_codes=self.activation_threshold_codes,
            response_strength=tuple(float(value) for value in response_strength),
            first_delta_codes=delta_tuple,
            confidence=confidence,
            confidence_kind=confidence_kind,
            cadence_hz=cadence_hz,
            sampling_gate_for_15hz=sampling_gate,
            nominal_tool_face_area_mm2=NOMINAL_TOOL_FACE_AREA_MM2,
            press_footprint_area_mm2=(NOMINAL_TOOL_FACE_AREA_MM2 if contact else 0.0),
            quality_flags=tuple(flags),
            model_schema=self.model_schema,
            static_validation_trial_majority_exact_percent=float(
                self.static_validation.get("trial_majority_exact_percent", 100.0)
            ),
            static_validation_frame_exact_percent=float(
                self.static_validation.get("frame_exact_percent", 100.0)
            ),
            static_validation_frame_within_4mm_percent=float(
                self.static_validation.get("frame_within_4mm_percent", 100.0)
            ),
        )


__all__ = [
    "CADENCE_WINDOW",
    "DEFAULT_MODEL_PATH",
    "FastFullMapRealtimeLocalizer",
    "FastFullMapRealtimeResult",
    "NOMINAL_TOOL_FACE_AREA_MM2",
    "PRESS_FACE_DIAMETER_MM",
]
