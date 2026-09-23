"""Temporary selected-channel spectra: detected peaks x 5 calibrated rows.

Protocol v3 never modifies installed tables. Automatic selections are equally
spaced; operator-edited points may be irregular. Plotted wavelengths are the
actual meter calibration of those rows. Sparse ADC points are measurements,
not a reconstructed full-band spectrum.
"""
from __future__ import annotations

import binascii
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

from laser_dac_safety import FULLBAND_2001_CODE_LIMITS

# T45 consumes rows from the dedicated 145 mA / 2001-point calibration.  Keep
# this opt-in local to protocol v3; legacy candidate/stress routes remain on
# their lower ceiling.
LIMITS = FULLBAND_2001_CODE_LIMITS
POINTS_PER_PEAK = 5
MAX_PEAKS = 9
MAX_POINTS = POINTS_PER_PEAK * MAX_PEAKS
# Compatibility name retained for older tests/tools that import POINTS.
POINTS = MAX_POINTS
FEEDBACK_KOHM = (2, 40, 5, 20)
SIGNAL_CHANNELS = ('CH0', 'CH1', 'CH2', 'CH3')
SELECTION_MIN_R2 = .95
SELECTION_MAX_RMSE_FRACTION = .10
SELECTION_CENTER_SMOOTH_SIGMA_NM = .04
SELECTION_DBM_SCORE_FRACTION = .27
SELECTION_DBM_AMBIGUITY_RATIO = .65
SELECTION_LINEAR_REFINEMENT_NM = .15


def assess_five_point_shape(wavelength_nm, values):
    """Stricter-than-display gate used when accepting a fixed five-point route.

    The realtime fitter always displays a numerical estimate for finite input,
    while its ``valid`` flag identifies estimates without quality warnings. A
    reusable route still needs extra margin: point three must be the sampled
    apex and the reference fit must be comfortably inside the residual limit.
    """

    from temporary_peak_fit import fit_five_points

    x = np.asarray(wavelength_nm, dtype=float)
    y = np.asarray(values, dtype=float)
    height = float(np.ptp(y)) if y.shape == (5,) and np.all(np.isfinite(y)) else 0.
    # Dense references may be expressed as ADC codes, volts, or normalized
    # test data.  Shape fitting is affine-scale invariant, while the realtime
    # fitter's raw-data amplitude/saturation gates are intentionally not.
    normalized = (100. + 1000. * (y - np.min(y)) / height) if height > 0 else y
    fit = fit_five_points(x, normalized)
    r2 = fit.get('r_squared')
    rmse = fit.get('rmse_adc')
    center_is_third = bool(y.shape == (5,) and height > 0
                           and np.max(y) - y[2] <= .03 * height)
    normalized_height = float(np.ptp(normalized)) if normalized.shape == (5,) else 0.
    rmse_fraction = (float(rmse / normalized_height)
                     if rmse is not None and normalized_height > 0 else math.inf)
    passed = bool(fit.get('valid') and center_is_third and r2 is not None
                  and r2 >= SELECTION_MIN_R2
                  and rmse_fraction <= SELECTION_MAX_RMSE_FRACTION)
    return dict(
        passed=passed,
        reason='固定五点峰形余量通过' if passed else (
            '第三点偏离峰顶' if not center_is_third else
            fit.get('reason', '参考峰形不合格') if not fit.get('valid') else
            '参考峰形拟合余量不足'
        ),
        center_is_third=center_is_third,
        fit_valid=bool(fit.get('valid')),
        fitted_center_nm=fit.get('center_nm'),
        r_squared=float(r2) if r2 is not None else None,
        rmse_fraction=rmse_fraction if math.isfinite(rmse_fraction) else None,
        reference_values=y.tolist() if y.shape == (5,) else [],
        minimum_r_squared=SELECTION_MIN_R2,
        maximum_rmse_fraction=SELECTION_MAX_RMSE_FRACTION,
    )


def _route_jump_score(table, indices, previous_row=None):
    """Dimensionless DAC travel for a sparse route, including its entry jump."""

    codes = []
    if previous_row is not None:
        codes.append(np.asarray(previous_row['codes'], dtype=float))
    codes.extend(np.asarray(table[index].codes, dtype=float) for index in indices)
    limits = np.asarray(LIMITS, dtype=float)
    return float(sum(np.sum(np.abs(right - left) / limits)
                     for left, right in zip(codes, codes[1:])))


def _plan_schema(point_count):
    return ('temporary_test_45_plan_v1' if point_count == MAX_POINTS
            else 'temporary_test_variable_plan_v2')


def _validate_group_count(groups):
    if not isinstance(groups, (list, tuple)) or not 1 <= len(groups) <= MAX_PEAKS:
        raise ValueError('临时路线需要1～9个峰，每峰5点')
    return len(groups) * POINTS_PER_PEAK


def build_selected_plan(table, groups, reference_values, *, source='validated_live_reference'):
    """Build a fixed route from explicitly validated peak×5 physical indices."""

    table = tuple(table)
    values = np.asarray(reference_values, dtype=float)
    point_count = _validate_group_count(groups)
    if (len(table) != 2001 or values.shape != (point_count,)
            or not np.all(np.isfinite(values))):
        raise ValueError(f'验证路线需要完整2001点表、{len(groups)}组索引和{point_count}个有限参考值')
    rows, checks, peaks = [], [], []
    previous = -1
    for group, indices in enumerate(groups, 1):
        if (len(indices) != 5 or any(type(index) is not int for index in indices)
                or indices[0] <= previous or len(set(np.diff(indices))) != 1):
            raise ValueError('验证路线必须是9组严格递增的等间隔五点')
        selected = []
        for index in indices:
            if not 0 <= index < len(table):
                raise ValueError('验证路线索引越界')
            point = table[index]
            selected.append(dict(group=group, index=int(point.index),
                                 target_nm=float(point.target_nm),
                                 measured_nm=float(point.measured_nm),
                                 codes=list(point.codes)))
        start = (group - 1) * POINTS_PER_PEAK
        check = assess_five_point_shape(
            [row['measured_nm'] for row in selected], values[start:start + 5]
        )
        check.update(group=group, peak_center_index=indices[2],
                     spacing_nm=(indices[1] - indices[0]) * .02)
        if not check['passed']:
            raise ValueError(f"验证路线第{group}峰不合格：{check['reason']}")
        checks.append(check)
        peaks.append(float(check['fitted_center_nm']))
        rows.extend(selected)
        previous = indices[-1]
    command(rows, 32, 1)
    return dict(schema=_plan_schema(point_count), rows=rows, peaks_nm=peaks,
                peak_count=len(groups), point_count=point_count,
                reference_kind='validated_sparse_CH1', reference_source=source,
                optical_accuracy_verified=False, discarded_weak_peaks=0,
                spacing_basis='nominal_calibrated_table', hidden_points=0,
                baseline_fraction=.1, soft_baseline=True, endpoint_checks=[],
                selection_fit_checks=checks, selection_validation_required=False,
                validated_reference_values=values.tolist())


def build_manual_dense_plan(table, groups, wavelengths, values, *, source='manual_dense_selection'):
    """Rebuild an operator-edited peak×5 route against one dense CH1 spectrum.

    Automatic selection remains equally spaced, but an operator may move each
    displayed point independently. Manual groups only have to remain strictly
    increasing on the physical 2001-point table. A weak dense-shape preview
    does not reject the edit: the route is kept disabled until the normal live
    45-point reference independently passes all nine groups.
    """

    table = tuple(table)
    x = np.asarray(wavelengths, dtype=float)
    y = np.asarray(values, dtype=float)
    point_count = _validate_group_count(groups)
    if (len(table) != 2001 or x.ndim != 1
            or y.shape != x.shape or len(x) < point_count
            or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))
            or np.any(np.diff(x) <= 0)):
        raise ValueError('手动选点需要完整2001点表和有序密集光谱')

    rows, checks, peaks = [], [], []
    measured_axis = np.asarray([point.measured_nm for point in table], dtype=float)
    previous = -1
    for group, indices in enumerate(groups, 1):
        if (len(indices) != 5 or any(type(index) is not int for index in indices)
                or indices[0] <= previous or np.any(np.diff(indices) <= 0)
                or indices[0] < 0 or indices[-1] >= len(table)):
            raise ValueError(f'手动路线必须是{len(groups)}组互不重叠且波长递增的五点')
        selected = []
        for index in indices:
            point = table[index]
            selected.append(dict(group=group, index=int(point.index),
                                 target_nm=float(point.target_nm),
                                 measured_nm=float(point.measured_nm),
                                 codes=list(point.codes)))
        sampled = [float(np.interp(measured_axis[index], x, y)) for index in indices]
        check = assess_five_point_shape(
            [row['measured_nm'] for row in selected], sampled
        )
        check.update(group=group, peak_center_index=indices[2], spacing_nm=None,
                     point_spacings_nm=(np.diff(indices) * .02).tolist())
        fitted_center = check.get('fitted_center_nm')
        peaks.append(float(fitted_center) if fitted_center is not None
                     and math.isfinite(fitted_center) else selected[2]['measured_nm'])
        checks.append(check)
        rows.extend(selected)
        previous = indices[-1]

    command(rows, 32, 1)
    return dict(
        schema=_plan_schema(point_count), rows=rows, peaks_nm=peaks,
        peak_count=len(groups), point_count=point_count,
        reference_kind='dense_CH1', reference_source=source,
        reference_wavelengths_nm=x.tolist(), reference_values=y.tolist(),
        optical_accuracy_verified=False, discarded_weak_peaks=0,
        spacing_basis='nominal_calibrated_table', hidden_points=0,
        baseline_fraction=None, soft_baseline=True, endpoint_checks=[],
        selection_fit_checks=checks,
        selection_validation_required=not all(check['passed'] for check in checks),
        manual_selection=True,
    )


def _validate_plan(plan):
    rows = plan.get('rows', [])
    command(rows, 32, 1)
    peaks = plan.get('peaks_nm', [])
    peak_count = len(rows) // POINTS_PER_PEAK
    if (len(rows) % POINTS_PER_PEAK or not 1 <= peak_count <= MAX_PEAKS
            or len(peaks) != peak_count or not all(math.isfinite(v) for v in peaks)):
        raise ValueError('临时点表峰数或每峰五点结构无效')
    for i, row in enumerate(rows):
        if (row.get('group') != i // 5 + 1 or
                not math.isfinite(row.get('target_nm', math.nan)) or
                not math.isfinite(row.get('measured_nm', math.nan)) or
                abs(row['target_nm'] - (1525 + .02 * row['index'])) > 1e-6 or
                abs(row['measured_nm'] - row['target_nm']) > .00201):
            raise ValueError('临时点表的分组或标定坐标无效')
    plan['optical_accuracy_verified'] = False
    plan['peak_count'] = peak_count
    plan['point_count'] = len(rows)
    checks = plan.get('selection_fit_checks')
    plan['selection_validation_required'] = not (
        isinstance(checks, list) and len(checks) == peak_count
        and all(isinstance(check, dict) and check.get('passed') for check in checks)
    )
    return plan


def validate_plan_payload(plan):
    """Validate an in-memory saved plan before restoring it into the UI."""

    if not isinstance(plan, dict):
        raise ValueError('保存的临时45点计划格式无效')
    # Validation mutates only derived safety flags; isolate the caller's data.
    return _validate_plan(json.loads(json.dumps(plan)))


def load_saved_plan(path):
    """Load one validated, selection-only 45-point plan without live I/O."""

    path = Path(path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('schema') not in (
            'temporary_test_45_plan_v1', 'temporary_test_variable_plan_v2'):
        raise ValueError('不是临时选点文件')
    plan = _validate_plan(payload)
    plan['selection_plan_source'] = str(path.resolve())
    return plan


def load_calibrated_plan(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    plan = payload.get('plan', {})
    results = payload.get('rows', [])
    if (not payload.get('complete') or len(results) not in range(5, 46, 5)
            or not all(r.get('success') for r in results)):
        raise ValueError('需要完整成功的每峰五点实时标定文件')
    _validate_plan(plan)
    for i, (selected, calibrated) in enumerate(zip(plan['rows'],results)):
        if (selected['index'] != calibrated['index'] or selected['group'] != i//5+1 or
            selected['codes'] != calibrated['calibrated_codes'] or
            not math.isfinite(selected['target_nm']) or
            not math.isfinite(selected['measured_nm']) or
            abs(selected['target_nm']-(1525+.02*selected['index'])) > 1e-6 or
            abs(selected['measured_nm']-selected['target_nm']) > .00201):
            raise ValueError('临时标定记录不一致或波长误差超限')
    plan['calibration_source'] = str(Path(path).resolve())
    plan['optical_accuracy_verified'] = False  # Held calibration is not transient verification.
    return plan


def select_points(table, wavelengths, values, *, spacing_nm=None,
                  baseline_fraction=None, soft_baseline=False,
                  feedback_selector=2):
    """Select five equally spaced calibrated rows inside every FWHM.

    ``spacing_nm``, ``baseline_fraction`` and ``soft_baseline`` remain in the
    call signature so older saved UI configuration can still be opened.  The
    automatic temporary-mode rule is now deliberately fixed: all five points
    must lie at or above the prominence-referenced half-height of the smoothed
    measured peak.  Operator-dragged routes continue to use
    :func:`build_manual_dense_plan` and may be nonuniform.
    """
    x, y = np.asarray(wavelengths, float), np.asarray(values, float)
    if x.ndim != 1 or y.shape != x.shape or len(x) < POINTS_PER_PEAK:
        raise ValueError("需要至少5个有序密集光谱点")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.any(np.diff(x) <= 0):
        raise ValueError("参考光谱存在空值或波长顺序错误")
    if np.max(np.diff(x)) > .1 + 1e-8:
        raise ValueError("参考光谱必须连续，步长不大于0.1 nm")
    step = float(np.median(np.diff(x)))
    # Dense scans can contain isolated route-dependent spikes even when every
    # individual point passes its within-window stability gate.  Find peaks in
    # estimated optical-power dBm: logarithmic compression keeps a weak late
    # FBG visible when the first peaks would otherwise set an overly high
    # global linear-ADC prominence threshold.  dBm also makes baseline ripple
    # look prominent, so rank candidates by dBm prominence * sqrt(dBm width)
    # and retain only the dominant cluster.  The original linear ADC envelope
    # remains authoritative for FWHM geometry, exported samples, and fitting.
    center_curve = gaussian_filter1d(
        y, sigma=max(.5, SELECTION_CENTER_SMOOTH_SIGMA_NM / step), mode='nearest'
    )
    from temporary_optical_units import adc_code_to_dbm

    if type(feedback_selector) is not int or feedback_selector not in range(4):
        raise ValueError('Invalid analog feedback selector')
    # Real ADC data has a half-LSB optical floor.  The relative floor keeps
    # normalized engineering/test spectra from turning tiny Gaussian tails
    # into hundreds of artificial negative dB and does not affect real codes.
    detection_floor_code = (
        .5 if float(np.max(y)) >= 50.0
        else max(float(np.max(y)) * 1e-4, np.finfo(float).tiny)
    )
    detection_curve = gaussian_filter1d(
        np.asarray(adc_code_to_dbm(
            np.maximum(y, detection_floor_code), feedback_selector
        ), dtype=float),
        sigma=max(.5, SELECTION_CENTER_SMOOTH_SIGMA_NM / step),
        mode='nearest',
    )
    detection_prominence = max(float(np.ptp(detection_curve)) * .08, 1e-9)
    detected_peaks, properties = find_peaks(
        detection_curve,
        prominence=detection_prominence,
        distance=max(1, round(1.0 / step)),
    )
    if not len(detected_peaks):
        raise ValueError('dBm光强谱未识别到有效峰')
    detection_width_samples = peak_widths(
        detection_curve, detected_peaks, rel_height=.5
    )[0]
    detection_width_nm = np.maximum(detection_width_samples * step, step)
    detection_scores = (
        np.asarray(properties['prominences'], dtype=float)
        * np.sqrt(detection_width_nm)
    )
    # A clipped (saturated) FBG reflection has a flattened top, which inflates
    # its dBm half-height width and therefore its score far beyond the physical
    # peak population.  Normalising against that inflated maximum raises the
    # keep threshold and silently discards the weakest real gratings.  Use the
    # second-highest score as the reference instead: a single clipped giant
    # cannot raise it, yet it stays far above ripple/baseline bumps, which the
    # same fraction rule continues to exclude.
    if len(detection_scores) >= 2:
        detection_score_reference = float(
            np.sort(detection_scores)[-2]
        )
    else:
        detection_score_reference = float(np.max(detection_scores))
    dominant = np.flatnonzero(
        detection_scores >= detection_score_reference * SELECTION_DBM_SCORE_FRACTION
    )
    if not len(dominant):
        dominant = np.asarray([int(np.argmax(detection_scores))], dtype=int)
    rank = dominant[np.argsort(detection_scores[dominant])[::-1]]
    if len(rank) > MAX_PEAKS:
        # Never silently discard a tenth broad dBm peak whose combined score
        # is comparable to the weakest of the selected nine.
        if (detection_scores[rank[MAX_PEAKS]]
                >= SELECTION_DBM_AMBIGUITY_RATIO
                * detection_scores[rank[MAX_PEAKS - 1]]):
            raise ValueError('主峰与次峰无法可靠区分，请检查参考光谱；未生成临时点表')
        rank = rank[:MAX_PEAKS]
    selected_detection_indices = np.sort(rank)
    selected_detection_peaks = detected_peaks[selected_detection_indices]
    discarded = len(detected_peaks) - len(selected_detection_peaks)

    # Smoothing after a logarithm can shift an apex by a few samples. Refine
    # each selected dBm candidate to the local maximum of the smoothed linear
    # ADC envelope before computing the physical half-height crossings.
    refinement_radius = max(1, round(SELECTION_LINEAR_REFINEMENT_NM / step))
    refined_peaks = []
    for detected_peak in selected_detection_peaks:
        left = max(0, int(detected_peak) - refinement_radius)
        right = min(len(center_curve), int(detected_peak) + refinement_radius + 1)
        refined_peaks.append(left + int(np.argmax(center_curve[left:right])))
    peaks = np.asarray(refined_peaks, dtype=int)
    if len(set(peaks.tolist())) != len(peaks):
        raise ValueError('dBm主峰在线性光谱中发生重叠，请检查参考光谱')
    if not 1 <= len(peaks) <= MAX_PEAKS:
        raise ValueError(f"参考光谱识别到{len(peaks)}个有效峰，无法生成每峰五点路线")
    if spacing_nm is not None and (
            not math.isfinite(float(spacing_nm)) or float(spacing_nm) < 0):
        raise ValueError("点间隔必须为非负数")
    if baseline_fraction is not None and not 0 < float(baseline_fraction) < .5:
        raise ValueError('基线附近比例必须在0和0.5之间')
    # scipy's half-height is referenced to each peak's measured prominence,
    # rather than to ADC zero.  This is the correct FWHM definition when the
    # local detector baseline is nonzero.
    widths = peak_widths(center_curve, peaks, rel_height=.5)
    axis = np.asarray([p.target_nm for p in table], float)
    if len(axis) != 2001 or not np.allclose(np.diff(axis), .02, atol=1e-7):
        raise ValueError("需要完整0.02 nm标定表")
    measured_axis = np.asarray([p.measured_nm for p in table], float)
    if not np.all(np.isfinite(measured_axis)) or np.any(np.diff(measured_axis) <= 0):
        raise ValueError("实测标定波长必须有限且递增")
    rows, centers, endpoint_checks, selection_fit_checks = [], [], [], []
    sample_axis = np.arange(len(x), dtype=float)
    for group, (peak, half_height, left, right) in enumerate(
            zip(peaks, widths[1], widths[2], widths[3]), 1):
        peak_nm = float(x[peak])
        left_nm = float(np.interp(left, sample_axis, x))
        right_nm = float(np.interp(right, sample_axis, x))
        threshold = float(half_height)
        nominal_center = int(np.argmin(abs(measured_axis - peak_nm)))

        # The physical table is nominally uniform at 0.02 nm.  Search the
        # nearest few possible middle rows and all feasible integer strides.
        # This keeps the third point at the measured apex while guaranteeing
        # exact equal spacing and keeping every selected point inside the
        # interpolated half-height crossings.
        candidate_centers = range(
            max(0, nominal_center - 2), min(len(table), nominal_center + 3)
        )
        choices = []
        for candidate_center in candidate_centers:
            if abs(float(measured_axis[candidate_center]) - peak_nm) > .05:
                continue
            for candidate_stride in range(1, len(table) // 4):
                indices = [
                    candidate_center + offset * candidate_stride
                    for offset in (-2, -1, 0, 1, 2)
                ]
                if indices[0] < 0 or indices[-1] >= len(table):
                    break
                if measured_axis[indices[0]] < left_nm or measured_axis[indices[-1]] > right_nm:
                    break
                if rows and indices[0] <= rows[-1]['index']:
                    continue
                envelope_values = np.interp(measured_axis[indices], x, center_curve)
                # A small numerical tolerance only covers interpolation at
                # the crossing; it never admits a visibly sub-half-height row.
                if np.any(envelope_values < threshold - 1e-9):
                    continue
                sampled = [float(np.interp(measured_axis[j], x, y)) for j in indices]
                check = assess_five_point_shape(measured_axis[indices], sampled)
                choices.append(dict(
                    center=candidate_center,
                    stride=candidate_stride,
                    indices=indices,
                    sampled=sampled,
                    envelope_values=envelope_values.tolist(),
                    check=check,
                    center_error_nm=abs(float(measured_axis[candidate_center]) - peak_nm),
                    uncovered_nm=(float(measured_axis[indices[0]]) - left_nm)
                                 + (right_nm - float(measured_axis[indices[-1]])),
                    route_jump_score=_route_jump_score(
                        table, indices, rows[-1] if rows else None
                    ),
                ))
        if not choices:
            raise ValueError(
                f'第{group}峰在半高宽内没有峰顶居中、等间隔且可拟合的五点窗口'
            )
        # A noisy dense forward scan can contain route-dependent single-row
        # spikes even though its smoothed FBG envelope is valid.  Prefer a raw
        # five-point preview that already fits, but do not let that preview
        # veto geometrically valid half-height points: the independently
        # acquired slow sparse reference remains the authoritative 9/9 gate.
        fitting_choices = [choice for choice in choices if choice['check']['passed']]
        choice_pool = fitting_choices or choices
        # First cover as much of the FWHM as the 0.02 nm table permits.  The
        # following keys only resolve grid/fit ties and do not move a point
        # below half height.
        chosen = min(choice_pool, key=lambda choice: (
            -choice['stride'],
            choice['center_error_nm'],
            (choice['check']['rmse_fraction']
             if choice['check']['rmse_fraction'] is not None else math.inf),
            (-choice['check']['r_squared']
             if choice['check']['r_squared'] is not None else math.inf),
            choice['uncovered_nm'],
            choice['route_jump_score'],
        ))
        center = chosen['center']
        stride = chosen['stride']
        indices = chosen['indices']
        centers.append(float(x[peak]))
        sampled = chosen['sampled']
        fit_check = chosen['check']
        fit_check.update(group=group, peak_center_index=center, spacing_nm=stride * .02)
        selection_fit_checks.append(fit_check)
        endpoint_checks.append(dict(
            group=group,
            selection_rule='at_or_above_prominence_half_height',
            half_height_adc=threshold,
            fwhm_left_nm=left_nm,
            fwhm_right_nm=right_nm,
            fwhm_nm=right_nm - left_nm,
            spacing_nm=stride * .02,
            peak_center_index=center,
            selected_envelope_adc=chosen['envelope_values'],
            all_selected_at_or_above_half_height=True,
            route_jump_score=chosen['route_jump_score'],
        ))
        for index in indices:
            point = table[index]
            rows.append(dict(group=group, index=int(point.index), target_nm=float(point.target_nm),
                             measured_nm=float(point.measured_nm), codes=list(point.codes)))
    command(rows, 32, 1)  # Same validation as the transport boundary.
    detection_checks = []
    dominant_set = set(int(value) for value in dominant)
    selected_detection_set = set(int(value) for value in selected_detection_indices)
    for candidate_index, peak_index in enumerate(detected_peaks):
        detection_checks.append(dict(
            wavelength_nm=float(x[peak_index]),
            prominence_db=float(properties['prominences'][candidate_index]),
            width_nm=float(detection_width_nm[candidate_index]),
            score=float(detection_scores[candidate_index]),
            dominant=bool(candidate_index in dominant_set),
            selected=bool(candidate_index in selected_detection_set),
        ))
    return dict(schema=_plan_schema(len(rows)), rows=rows, peaks_nm=centers,
                peak_count=len(centers), point_count=len(rows),
                reference_kind="dense_CH1", optical_accuracy_verified=False,
                discarded_weak_peaks=discarded,
                spacing_basis="nominal_calibrated_table", hidden_points=0,
                selection_center_method="dbm_prominence_width_rank_then_linear_fwhm_inner_equal_spacing",
                peak_detection_domain="estimated_optical_power_dbm",
                peak_detection_feedback_selector=feedback_selector,
                peak_detection_candidate_count=len(detected_peaks),
                peak_detection_floor_adc=float(detection_floor_code),
                peak_detection_dominant_score_fraction=SELECTION_DBM_SCORE_FRACTION,
                peak_detection_checks=detection_checks,
                selection_center_smooth_sigma_nm=SELECTION_CENTER_SMOOTH_SIGMA_NM,
                selection_height_fraction=.5,
                selection_rule="five_equal_interval_points_at_or_above_half_height",
                baseline_fraction=None,soft_baseline=False,
                endpoint_checks=endpoint_checks,
                selection_fit_checks=selection_fit_checks,
                selection_validation_required=not all(
                    check['passed'] for check in selection_fit_checks
                ))


def _feedback_pair(adc_channel, feedback_selector):
    if adc_channel == 0:
        return feedback_selector, 0
    if adc_channel == 1:
        return 0, feedback_selector
    return 0, 0


def command(rows, cycles, tag, first_delay_us=600, spacing_us=50,
            boundary_extra_us=0, feedback_selector=2, point_delays_us=None,
            adc_channel=1):
    if type(feedback_selector) is not int or feedback_selector not in range(4):
        raise ValueError('Invalid analog feedback selector')
    if type(adc_channel) is not int or adc_channel not in range(4):
        raise ValueError('Invalid ADC channel')
    if adc_channel >= 2:
        # CH2/CH3 have fixed 2 kOhm feedback; keep the encoded provenance exact.
        feedback_selector = 0
    if type(cycles) is not int or not (cycles == 0 or 32 <= cycles <= 512):
        raise ValueError("cycles must be 0 (continuous) or 32..512")
    point_count = len(rows)
    if (type(tag) is not int or not 1 <= tag <= 0xffffffff
            or point_count not in range(POINTS_PER_PEAK, MAX_POINTS + 1,
                                        POINTS_PER_PEAK)):
        raise ValueError("5～45 rows (a multiple of five) and a nonzero uint32 tag required")
    if any(type(v) is not int or not 50 <= v <= limit or v % 25 for v, limit in ((first_delay_us, 850), (spacing_us, 600))) or first_delay_us + spacing_us > 900:
        raise ValueError("首次等待须为50～850 µs、双采样间隔须为50～600 µs，均为25 µs整数倍且合计不超过900 µs")
    if type(boundary_extra_us) is not int or not 0 <= boundary_extra_us <= 3000 or boundary_extra_us % 25:
        raise ValueError('跨峰额外等待须为0～3000 µs的25 µs整数倍；总等待不限，允许低于15 Hz')
    packet = bytearray(808)
    protocol_version = (
        3 if adc_channel == 1 and point_count == MAX_POINTS else
        4 if adc_channel == 1 else
        5
    )
    packet[:6] = bytes((0xff, 0xff, 3, 10, protocol_version, point_count))
    packet[6:8] = first_delay_us.to_bytes(2, 'big')
    packet[8:12] = tag.to_bytes(4, 'big')
    packet[12:16] = (b'T45!' if protocol_version == 3 else
                     b'TVR!' if protocol_version == 4 else b'TVC!')
    packet[16:18] = cycles.to_bytes(2, 'big')
    packet[18] = 16
    packet[19] = ((0 if feedback_selector == 2 else feedback_selector + 1)
                  if protocol_version in (3, 4)
                  else (adc_channel << 4) | feedback_selector)
    packet[20:22] = spacing_us.to_bytes(2, 'big')
    packet[22:24] = boundary_extra_us.to_bytes(2,'big')
    previous = -1
    for i, row in enumerate(rows):
        index, codes = row['index'], tuple(row['codes'])
        if type(index) is not int or not previous < index <= 2000 or len(codes) != 5:
            raise ValueError("Invalid physical table order")
        if any(type(c) is not int or not 0 <= c <= limit for c, limit in zip(codes, LIMITS)):
            raise ValueError("DAC current limit exceeded")
        previous = index
        packet[24+i*12:36+i*12] = b''.join(v.to_bytes(2, 'big') for v in (index, *codes))
    packet[564:568] = binascii.crc32(packet[:564]).to_bytes(4, 'big')
    if point_delays_us is not None:
        if boundary_extra_us or len(point_delays_us) != point_count or any(
                type(v) is not int or not 50 <= v <= 15000 or v % 25 for v in point_delays_us):
            raise ValueError(f'逐点等待须为{point_count}个50～15000 µs的25 µs整数倍，跨峰额外等待须为0')
        packet[568:572] = b'PWT1'
        encoded = b''.join(v.to_bytes(2, 'big') for v in point_delays_us)
        packet[572:572 + len(encoded)] = encoded
        packet[662:666] = binascii.crc32(packet[:662]).to_bytes(4, 'big')
    return bytes(packet)


def extract_frames(buffer):
    while True:
        start = buffer.find(b'\xd9\x9d')
        if start < 0:
            if len(buffer) > 1:
                del buffer[:-1]
            return
        del buffer[:start]
        if len(buffer) < 4:
            return
        kind, status = buffer[2:4]
        # Include the old rejection type so old firmware fails visibly.  Data
        # frames carry their variable point count at byte 29.
        if kind in (10, 14) and status == 127:
            length = 20
        elif kind == 14:
            if len(buffer) < 30:
                return
            point_count = buffer[29]
            if point_count not in range(POINTS_PER_PEAK, MAX_POINTS + 1,
                                        POINTS_PER_PEAK):
                del buffer[0]
                continue
            length = 46 + point_count * (13 if status & 2 else 9)
        elif kind == 15:
            length = 42
        else:
            length = 0
        if not length:
            del buffer[0]
            continue
        if len(buffer) < length:
            return
        wire = bytes(buffer[:length])
        del buffer[:length]
        yield wire


def decode(wire, tag, route_crc, first_delay_us=600, spacing_us=50,
           boundary_extra_us=0, feedback_selector=2, point_delays_us=None,
           point_count=MAX_POINTS, adc_channel=1):
    u16 = lambda at: int.from_bytes(wire[at:at+2], 'big')
    u32 = lambda at: int.from_bytes(wire[at:at+4], 'big')
    if len(wire) == 20 and wire[:2] == b'\xd9\x9d' and wire[2] in (10, 14) and wire[3] == 127:
        if u32(4) != tag or wire[-2:] != b'\x9d\xd9':
            raise ValueError("Rejection identity mismatch")
        raise RuntimeError(f"板卡拒绝临时{point_count}点协议（{u16(8):04x}），请使用配套固件并检查设备状态")
    expected_narrow = 46 + point_count * 9
    expected_wide = 46 + point_count * 13
    if len(wire) not in (expected_narrow, expected_wide, 42) or wire[:2] != b'\xd9\x9d' or wire[-2:] != b'\x9d\xd9' or u32(len(wire)-6) != binascii.crc32(wire[:-6]):
        raise ValueError("Frame CRC/length/trailer mismatch")
    if u32(4) != tag:
        raise ValueError("Session tag mismatch")
    if len(wire) == 42 and wire[2] == 15:
        feedback_pair = _feedback_pair(adc_channel, feedback_selector)
        if (u32(32) != route_crc
                or wire[18:22] != bytes((1, 1, *feedback_pair))):
            raise ValueError("Terminal route/shutter/feedback mismatch")
        return dict(kind='terminal', status=wire[3], completed=u16(12), requested=u16(14),
                    stop_reason=u16(16), i2c_errors=u16(22), spi_errors=u32(24), firmware=u32(8),
                    adc_channel=adc_channel,
                    signal_channel_name=SIGNAL_CHANNELS[adc_channel])
    status = wire[3]
    wide = bool(status & 2)
    encoded_channel = (status >> 2) & 3
    permitted_mask = 3 if adc_channel == 1 else 15
    if (wire[2] != 14 or len(wire) != (expected_wide if wide else expected_narrow)
            or status & ~permitted_mask or encoded_channel != (0 if adc_channel == 1 else adc_channel)
            or u32(24) != route_crc):
        raise ValueError("Data identity mismatch")
    sequence = u32(12)
    order_end = 40 + point_count
    feedback_pair = _feedback_pair(adc_channel, feedback_selector)
    if (bool(status & 1) != (sequence >= 16)
            or wire[28:34] != bytes((first_delay_us//25, point_count, 16,
                                    spacing_us//25, *feedback_pair))
            or wire[40:order_end] != bytes(range(point_count))):
        raise ValueError("Data timing/count/order mismatch")
    records, previous = [], -1
    for i in range(point_count):
        at = order_end + i*(12 if wide else 8)
        first, second = u16(at), u16(at+2)
        write, end = (u32(at+4), u32(at+8)) if wide else (u16(at+4), u16(at+6))
        expected_delay = first_delay_us+spacing_us+(boundary_extra_us if i%5==0 else 0)
        if point_delays_us is not None:
            expected_delay = point_delays_us[i] + spacing_us
        if max(first, second) > 4095 or not previous < write < end <= u32(20)+1 or end-write < expected_delay:
            raise ValueError("ADC/timestamp validation failed")
        previous = end
        records.append(dict(first_code=first, second_code=second, write_end_us=write, second_end_us=end))
    return dict(kind='data', sequence=sequence, cycle_start_us=u32(16), elapsed_us=u32(20),
                records=records, i2c_errors=u16(34), spi_errors=u32(36), firmware=u32(8),
                feedback_selector=feedback_selector, adc_channel=adc_channel,
                signal_channel_name=SIGNAL_CHANNELS[adc_channel])


def quality(frame):
    pairs = np.asarray([[r['first_code'], r['second_code']] for r in frame['records']], float)
    values = pairs[:, 1]
    clipped = bool(np.any(pairs >= 4080))
    settling = bool(np.any(abs(pairs[:, 1]-pairs[:, 0]) > np.maximum(8, abs(values)*.02)))
    if len(values) % POINTS_PER_PEAK:
        raise ValueError('Frame point count is not divisible into five-point peaks')
    edge = any(int(np.argmax(group)) in (0, 4)
               for group in values.reshape(-1, POINTS_PER_PEAK))
    return dict(warmed_up=frame['sequence'] >= 16, clipped=clipped, unsettled=settling,
                pair_difference_large=settling, stability_verified=False,
                peak_at_edge=edge, optical_accuracy_verified=False)


def execute_device(device, plan, output, *, cycles=256, on_frame=None, should_stop=lambda: False,
                   first_delay_us=600, spacing_us=50, boundary_extra_us=0,
                   feedback_selector=2, point_delays_us=None, adc_channel=1):
    import app_JDSU as app
    from benchmark_stress_realtime import _safe_disarm, _close_shutter
    from capture_ch1_teacher_spectrum import save_checkpoint
    output = Path(output)
    if type(adc_channel) is not int or adc_channel not in range(4):
        raise ValueError('Invalid ADC channel')
    if adc_channel >= 2:
        feedback_selector = 0
    if output.exists():
        raise FileExistsError(output)
    tag = (time.time_ns() & 0xffffffff) or 1
    point_delays_us = list(point_delays_us) if point_delays_us is not None else None
    point_count = len(plan['rows'])
    peak_count = point_count // POINTS_PER_PEAK
    packet = command(plan['rows'], cycles, tag, first_delay_us, spacing_us,
                     boundary_extra_us, feedback_selector, point_delays_us,
                     adc_channel)
    crc = binascii.crc32(packet[:662] if point_delays_us is not None else packet[24:564])
    report = dict(schema=('temporary_test_45_capture_v1' if point_count == MAX_POINTS
                          else 'temporary_test_variable_capture_v2'),
                  plan=plan, frames=[], complete=False,
                  point_count=point_count, peak_count=peak_count,
                  first_delay_us=first_delay_us, spacing_us=spacing_us,
                  boundary_extra_us=boundary_extra_us,
                  point_delays_us=point_delays_us,
                  signal_channel=adc_channel,
                  signal_channel_name=SIGNAL_CHANNELS[adc_channel],
                  feedback_selector=feedback_selector,
                  ch1_feedback_selector=(feedback_selector if adc_channel == 1 else 0),
                  signal_feedback_kohm=(FEEDBACK_KOHM[feedback_selector]
                                        if adc_channel < 2 else 2),
                  optical_accuracy_verified=False, rate_over_15hz_verified=False)
    old_timeout = device.timeout
    continuous = cycles == 0
    frame_log = None
    received_count = 0
    report['continuous'] = continuous
    report['stopped_by_user'] = False
    try:
        if continuous:
            output.parent.mkdir(parents=True, exist_ok=True)
            log_path = output.with_suffix('.frames.jsonl')
            frame_log = log_path.open('x', encoding='utf-8')
            report['frame_log'] = str(log_path)
            save_checkpoint(output, report)
        device.timeout = .03
        device.dtr = True
        if not _safe_disarm(device) or not _close_shutter(device):
            raise RuntimeError("Initial shutdown not confirmed")
        worker = app.EqualIntervalWorker(
            device, (), {}, settle_s=0,
            feedback_selectors=_feedback_pair(adc_channel, feedback_selector),
        )
        if not worker._feedback_command_and_ack():
            raise RuntimeError("Feedback ACK missing")
        if should_stop():
            raise InterruptedError("已取消")
        device.reset_input_buffer()
        if device.write(packet) != len(packet):
            raise RuntimeError("Incomplete command write")
        device.flush()
        buffer, terminal = bytearray(), None
        wait_us = (sum(point_delays_us) if point_delays_us is not None else
                   point_count*first_delay_us + peak_count*boundary_extra_us) + point_count*spacing_us
        frame_timeout_s = max(3., wait_us/1e6 + 1.)
        deadline = time.monotonic() + (frame_timeout_s if continuous else
                                     max(43., cycles*(wait_us/1e6 + .1) + 10.))
        while time.monotonic() < deadline and terminal is None:
            if should_stop():
                if continuous:
                    report['stopped_by_user'] = True
                    break
                raise InterruptedError("已停止")
            buffer.extend(device.read(8192))
            for wire in extract_frames(buffer):
                frame = decode(wire, tag, crc, first_delay_us, spacing_us,
                               boundary_extra_us, feedback_selector,
                               point_delays_us, point_count, adc_channel)
                if frame['i2c_errors'] or frame['spi_errors']:
                    raise RuntimeError("Board I2C/SPI error")
                if frame['kind'] == 'terminal':
                    terminal = frame
                    break
                expected_sequence = received_count & 0xffffffff
                if frame['sequence'] != expected_sequence:
                    raise ValueError(
                        "Missing/reordered frame: "
                        f"expected={expected_sequence}, received={frame['sequence']}"
                    )
                raw_start = frame['cycle_start_us']
                if report['frames']:
                    previous = report['frames'][-1]
                    delta = (raw_start - (previous['cycle_start_us'] & 0xffffffff)) & 0xffffffff
                    if not 0 < delta < 3_000_000 or frame['firmware'] != previous['firmware']:
                        raise ValueError("Board clock/firmware changed")
                    frame['cycle_start_us'] = previous['cycle_start_us'] + delta
                received_count += 1
                frame['host_received_s'] = time.monotonic()
                frame['quality'] = quality(frame)
                report['frames'].append(frame)
                if continuous:
                    frame_log.write(json.dumps(frame, ensure_ascii=False) + '\n')
                    if received_count % 16 == 0:
                        frame_log.flush()
                    del report['frames'][:-512]
                    deadline = time.monotonic() + frame_timeout_s
                if on_frame is not None:
                    on_frame(frame)
        if terminal is None and not report['stopped_by_user']:
            raise TimeoutError(f"{point_count}点扫描未收到结束确认")
        report['terminal'] = terminal
        if terminal is not None and (continuous or terminal['status'] or terminal['stop_reason'] or terminal['completed'] != cycles or terminal['requested'] != cycles or received_count != cycles):
            if terminal['stop_reason'] & 16:
                raise RuntimeError('板卡触发计时或会话超时保护，扫描不完整；请核对配套长等待固件及设备状态')
            raise RuntimeError("扫描未完整完成")
        times = [f['cycle_start_us'] for f in report['frames'] if f['sequence'] >= 16]
        gaps = np.diff(times)
        report['mean_hz'] = float(1e6 / np.mean(gaps)) if len(gaps) else 0.
        report['p99_period_ms'] = float(np.percentile(gaps, 99) / 1000) if len(gaps) else 0.
        report['rate_statistics_window'] = 'last_512_frames' if continuous else 'all_post_warmup_frames'
        report['rate_over_15hz_verified'] = report['mean_hz'] > 15 and report['p99_period_ms'] < 1000/15
        report['complete'] = True
    except BaseException as exc:
        report['error'] = str(exc)
        raise
    finally:
        errors = []
        for key, action in (('disarm_ack', _safe_disarm), ('shutter_ack', _close_shutter)):
            try:
                report[key] = bool(action(device))
            except Exception as exc:
                report[key] = False
                errors.append(str(exc))
        device.timeout = old_timeout
        report['cleanup_errors'] = errors
        report['received_frames'] = received_count
        if frame_log is not None:
            frame_log.close()
        save_checkpoint(output, report)
        if not report.get('disarm_ack') or not report.get('shutter_ack'):
            raise RuntimeError("停光未确认，请检查板卡；" + str(report.get('error', '')))
    return report
