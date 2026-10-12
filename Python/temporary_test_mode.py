"""Temporary selected-channel spectra: five-point legacy or seven-point routes.

Temporary protocols never modify installed tables. New automatic selections
pair the left/right fitted optical heights, not wavelength distances; operator
edits may be irregular. Plotted wavelengths are the
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
from scipy.interpolate import Akima1DInterpolator
from scipy.signal import find_peaks, peak_widths

from laser_dac_safety import FULLBAND_2001_CODE_LIMITS

# T45 consumes rows from the dedicated 145 mA / 2001-point calibration.  Keep
# this opt-in local to protocol v3; legacy candidate/stress routes remain on
# their lower ceiling.
LIMITS = FULLBAND_2001_CODE_LIMITS
POINTS_PER_PEAK = 5
MAX_PEAKS = 9
MAX_POINTS = POINTS_PER_PEAK * MAX_PEAKS
AUTO_POINTS_PER_PEAK = 7
MAX_AUTO_POINTS = AUTO_POINTS_PER_PEAK * MAX_PEAKS
V6_CRC_OFFSET = 796
V6_FIELD_BITS = (11, 16, 16, 15, 15, 15, 10)
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
AUTO_SELECTION_PROMINENCE_DB = 5.0
AUTO_SELECTION_HEIGHT_FRACTION = .5


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
    count = len(groups[0])
    if count not in (5, 7) or any(len(group) != count for group in groups):
        raise ValueError('每峰必须统一为5点或7点')
    return len(groups) * count


def points_per_peak_for_rows(rows):
    """Infer uniform contiguous groups; old ungrouped diagnostic rows stay x5."""
    if not rows or not all('group' in row for row in rows):
        return 5
    groups = [row['group'] for row in rows]
    count = sum(group == 1 for group in groups)
    if count not in (5, 7) or len(rows) % count or len(rows) // count > MAX_PEAKS:
        raise ValueError('点表必须包含1～9个完整的五点或七点峰')
    if groups != [i // count + 1 for i in range(len(rows))]:
        raise ValueError('峰分组必须连续且按波长排序')
    return count


def assess_peak_shape(wavelength_nm, values):
    if len(values) == 5:
        return assess_five_point_shape(wavelength_nm, values)
    from temporary_peak_fit_makima import fit_peak_points
    fit = fit_peak_points(wavelength_nm, values)
    return dict(passed=bool(fit.get('valid')), reason=fit.get('reason'),
                fitted_center_nm=fit.get('center_nm'), r_squared=fit.get('r_squared'),
                reference_values=list(values))


def build_selected_plan(table, groups, reference_values, *, source='validated_live_reference'):
    """Build a fixed route from explicitly validated peak×5 physical indices."""

    table = tuple(table)
    values = np.asarray(reference_values, dtype=float)
    point_count = _validate_group_count(groups)
    group_size = len(groups[0])
    if (len(table) != 2001 or values.shape != (point_count,)
            or not np.all(np.isfinite(values))):
        raise ValueError(f'验证路线需要完整2001点表、{len(groups)}组索引和{point_count}个有限参考值')
    rows, checks, peaks = [], [], []
    previous = -1
    for group, indices in enumerate(groups, 1):
        if (len(indices) != group_size or any(type(index) is not int for index in indices)
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
        start = (group - 1) * group_size
        check = assess_peak_shape(
            [row['measured_nm'] for row in selected], values[start:start + group_size]
        )
        check.update(group=group, peak_center_index=indices[group_size // 2],
                     spacing_nm=(indices[1] - indices[0]) * .02)
        if not check['passed']:
            raise ValueError(f"验证路线第{group}峰不合格：{check['reason']}")
        checks.append(check)
        peaks.append(float(check['fitted_center_nm']))
        rows.extend(selected)
        previous = indices[-1]
    command(rows, 32, 1)
    return dict(schema=_plan_schema(point_count), rows=rows, peaks_nm=peaks,
                peak_count=len(groups), point_count=point_count, points_per_peak=group_size,
                reference_kind='validated_sparse_CH1', reference_source=source,
                optical_accuracy_verified=False, discarded_weak_peaks=0,
                spacing_basis='nominal_calibrated_table', hidden_points=0,
                baseline_fraction=.1, soft_baseline=True, endpoint_checks=[],
                selection_fit_checks=checks, selection_validation_required=False,
                validated_reference_values=values.tolist())


def _three_height_pairs(left_dbm, right_dbm, peak_dbm, drop_db, *, anchor_outer=False):
    """Jointly choose ordered, distinct real pairs near three optical levels.

    Each flank is ordered from apex outward. The optical-pair mismatch has
    four times the squared-error weight of an individual target-level error.
    A two-dimensional prefix minimum enforces distinct/outward rows without
    imposing mirrored wavelengths or enumerating every six-row combination.
    With anchor_outer, the lowest pair is fixed to the closest allowed height
    above the cutoff on each flank, leaving at least two real rows inward.
    """
    left, right = np.asarray(left_dbm, float), np.asarray(right_dbm, float)
    if (left.ndim != 1 or right.ndim != 1 or len(left) < 3 or len(right) < 3
            or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right))
            or not math.isfinite(peak_dbm) or not math.isfinite(drop_db) or drop_db <= 0):
        raise ValueError('峰左右各需至少三个真实采样点')
    levels = np.linspace(drop_db / 3, drop_db, 3)
    mismatch = 4 * (left[:, None] - right[None, :]) ** 2
    outer = None
    if anchor_outer:
        threshold = peak_dbm - drop_db
        endpoints = []
        for flank in (left, right):
            eligible = [i for i in range(2, len(flank)) if flank[i] >= threshold - 1e-8
                        and np.count_nonzero(flank[:i] >= flank[i]) >= 2]
            if not eligible:
                raise ValueError('半高线以上缺少最低点及两个更高的内侧真实点')
            endpoints.append(min(eligible, key=lambda i: abs(flank[i] - threshold)))
        outer = tuple(endpoints)
    stages = []
    for stage, level in enumerate(levels):
        cost = ((peak_dbm - left[:, None] - level) ** 2
                + (peak_dbm - right[None, :] - level) ** 2 + mismatch)
        if stages:
            prefix = np.minimum.accumulate(np.minimum.accumulate(stages[-1], axis=0), axis=1)
            prior = np.full_like(cost, np.inf)
            prior[1:, 1:] = prefix[:-1, :-1]
            cost += prior
        if outer is not None:
            if stage == 2:
                anchored = np.full_like(cost, np.inf)
                anchored[outer] = cost[outer]
                cost = anchored
            else:
                # A noisy flank must not put an inner selected point below
                # the anchored "lowest" point.
                cost[(left[:, None] < left[outer[0]]) | (right[None, :] < right[outer[1]])] = np.inf
        stages.append(cost)
    at = np.unravel_index(int(np.argmin(stages[-1])), stages[-1].shape)
    pairs = []
    for stage in range(2, -1, -1):
        pairs.append(tuple(int(index) for index in at))
        if stage:
            prior = stages[stage - 1][:at[0], :at[1]]
            at = np.unravel_index(int(np.argmin(prior)), prior.shape)
    return list(reversed(pairs))


def auto_select_dense_plan(table, wavelengths, values, *, feedback_selector=2,
                           source='auto_dense_selection'):
    """Select seven height-layered real rows in each peak's visual dBm upper half.

    Keep upstream smoothed dBm detection/ranking. Fit the raw dense ADC with
    the default Makima interpolant, just like the displayed sparse peak, then
    convert its apex to dBm. The two local smoothed dBm valleys define a peak
    base; use the higher valley, then the dBm midpoint to the fitted apex.
    This is visual log-height, NOT half of linear power or a fixed -5 dB band.
    Retained raw ADC and physical DAC codes are never changed; a new sparse
    reference remains necessary.
    """
    table = tuple(table)
    x, y = np.asarray(wavelengths, float), np.asarray(values, float)
    if (len(table) != 2001 or x.ndim != 1 or y.shape != x.shape or len(x) < MAX_AUTO_POINTS
            or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))
            or np.any(np.diff(x) <= 0) or np.max(np.diff(x)) > .105):
        raise ValueError('自动选点需要完整2001点表和连续有序密集光谱')
    if type(feedback_selector) is not int or feedback_selector not in range(4):
        raise ValueError('Invalid analog feedback selector')
    measured_axis = np.asarray([point.measured_nm for point in table], float)
    if not np.all(np.isfinite(measured_axis)) or np.any(np.diff(measured_axis) <= 0):
        raise ValueError('标定波长必须有限且递增')
    # Dense samples can be a 751-point subrange with stride 2, not the full
    # 2001-point table. Map each actual sample to its calibrated physical row.
    physical_indices = np.asarray([
        int(np.argmin(abs(measured_axis - wavelength))) for wavelength in x
    ], dtype=int)
    if (np.any(np.diff(physical_indices) <= 0)
            or np.any(abs(measured_axis[physical_indices] - x) > .01001)):
        raise ValueError('密集光谱无法唯一映射到当前标定表')
    step = float(np.median(np.diff(x)))
    sigma = max(.5, SELECTION_CENTER_SMOOTH_SIGMA_NM / step)
    from temporary_optical_units import adc_code_to_dbm
    floor = .5 if float(np.max(y)) >= 50 else max(float(np.max(y)) * 1e-4, np.finfo(float).tiny)
    detection_curve = gaussian_filter1d(
        np.asarray(adc_code_to_dbm(np.maximum(y, floor), feedback_selector), float),
        sigma=sigma, mode='nearest')
    detected, properties = find_peaks(
        detection_curve, prominence=AUTO_SELECTION_PROMINENCE_DB,
        distance=max(1, round(1.0 / step)))
    if len(detected) < MAX_PEAKS:
        raise ValueError(f'只识别到{len(detected)}个有效峰，未生成63点；可继续手动选点')
    widths = np.maximum(peak_widths(detection_curve, detected, rel_height=.5)[0] * step, step)
    scores = np.asarray(properties['prominences'], float) * np.sqrt(widths)
    selected = np.sort(detected[np.argsort(scores)[::-1][:MAX_PEAKS]])
    # Default fitting is Makima in linear ADC; the dBm display is a conversion
    # of that fitted curve, not a new Gaussian or a fit of the smoothed detector.
    spline = Akima1DInterpolator(x, y, method='makima', extrapolate=False)
    stationary = np.asarray(spline.derivative().roots(extrapolate=False), float)
    stationary = stationary[np.isfinite(stationary)]
    maxima = stationary[np.asarray(spline.derivative(2)(stationary)) < 0]
    fitted_adc = np.asarray(spline(measured_axis[physical_indices]), float)
    sample_by_index = {int(index): order for order, index in enumerate(physical_indices)}
    groups, symmetry, bands = [], [], []
    for peak in selected:
        position = int(np.searchsorted(detected, peak))
        lower = (float((x[detected[position - 1]] + x[peak]) / 2)
                 if position else float(x[0]))
        upper = (float((x[peak] + x[detected[position + 1]]) / 2)
                 if position + 1 < len(detected) else float(x[-1]))
        candidates = maxima[(maxima >= max(lower, x[peak] - .35))
                            & (maxima <= min(upper, x[peak] + .35))]
        if not len(candidates):
            raise ValueError(f'第{len(groups) + 1}峰未找到内部Makima拟合峰顶，请检查密集光谱')
        apex_nm = float(max(candidates, key=lambda wavelength: float(spline(wavelength))))
        peak_adc = float(spline(apex_nm))
        if not math.isfinite(peak_adc) or peak_adc <= 0:
            raise ValueError('拟合峰值无效，无法计算dBm选点下限')
        peak_dbm = float(adc_code_to_dbm(peak_adc, feedback_selector))
        left_base_samples = np.flatnonzero((x >= lower) & (x < apex_nm))
        right_base_samples = np.flatnonzero((x > apex_nm) & (x <= upper))
        if not len(left_base_samples) or not len(right_base_samples):
            raise ValueError('拟合峰两侧缺少局部峰底，无法确定dBm上半段')
        left_base = int(left_base_samples[np.argmin(detection_curve[left_base_samples])])
        right_base = int(right_base_samples[np.argmin(detection_curve[right_base_samples])])
        left_base_dbm, right_base_dbm = float(detection_curve[left_base]), float(detection_curve[right_base])
        baseline_dbm = max(left_base_dbm, right_base_dbm)
        if not math.isfinite(baseline_dbm) or baseline_dbm >= peak_dbm:
            raise ValueError('局部峰底与拟合峰顶无有效dBm高度差')
        threshold_dbm = baseline_dbm + AUTO_SELECTION_HEIGHT_FRACTION * (peak_dbm - baseline_dbm)
        drop_db = peak_dbm - threshold_dbm
        threshold_adc = peak_adc * 10 ** (-drop_db / 10)
        crossings = np.asarray(spline.solve(threshold_adc, extrapolate=False), float)
        left_crossings = crossings[(crossings >= lower) & (crossings < apex_nm)]
        right_crossings = crossings[(crossings > apex_nm) & (crossings <= upper)]
        if not len(left_crossings) or not len(right_crossings):
            raise ValueError(f'第{len(groups) + 1}峰的dBm视觉上半段范围不完整，'
                             '请扩大扫描范围')
        band_left, band_right = float(np.max(left_crossings)), float(np.min(right_crossings))
        center = int(physical_indices[int(np.argmin(abs(x - apex_nm)))])
        # Rows must be real captured samples, inside the connected fitted band,
        # and above the same peak-relative dBm cutoff in raw and fitted values.
        within = ((x >= band_left) & (x <= band_right)
                  & (measured_axis[physical_indices] >= band_left)
                  & (measured_axis[physical_indices] <= band_right)
                  & (y >= threshold_adc - 1e-9) & (fitted_adc >= threshold_adc - 1e-9))
        available = set(int(index) for index in physical_indices[within])
        if center not in available:
            raise ValueError('拟合峰顶附近没有符合dBm上半段下限的真实中心点，'
                             '请减小采样间隔')
        # Pair optical heights, not physical-index offsets. Keep three distinct
        # captured rows strictly on each fitted flank and in wavelength order.
        left_indices = sorted((index for index in available if index < center
                               and measured_axis[index] < apex_nm
                               and x[sample_by_index[index]] < apex_nm), reverse=True)
        right_indices = sorted(index for index in available if index > center
                               and measured_axis[index] > apex_nm
                               and x[sample_by_index[index]] > apex_nm)
        if len(left_indices) < 3 or len(right_indices) < 3:
            raise ValueError(f'第{len(groups) + 1}峰的dBm上半段内'
                             '左右不足三个实测点，请减小采样间隔')
        left_heights = np.asarray(adc_code_to_dbm(
            fitted_adc[[sample_by_index[index] for index in left_indices]], feedback_selector), float)
        right_heights = np.asarray(adc_code_to_dbm(
            fitted_adc[[sample_by_index[index] for index in right_indices]], feedback_selector), float)
        pairs = _three_height_pairs(left_heights, right_heights, peak_dbm, drop_db,
                                    anchor_outer=True)
        paired_left = [left_indices[left] for left, _ in pairs]
        paired_right = [right_indices[right] for _, right in pairs]
        indices = list(reversed(paired_left)) + [center] + paired_right
        groups.append(indices)
        sample_orders = [sample_by_index[index] for index in indices]
        bands.append(dict(group=len(groups), fitted_peak_nm=apex_nm,
                          fitted_peak_dbm=peak_dbm, threshold_dbm=threshold_dbm,
                          baseline_dbm=baseline_dbm, left_base_dbm=left_base_dbm,
                          right_base_dbm=right_base_dbm,
                          left_base_nm=float(x[left_base]), right_base_nm=float(x[right_base]),
                          peak_drop_db=drop_db, height_fraction=AUTO_SELECTION_HEIGHT_FRACTION,
                          band_left_nm=band_left, band_right_nm=band_right,
                          fitted_peak_adc=peak_adc, threshold_adc=threshold_adc,
                          selected_fitted_dbm=np.asarray(adc_code_to_dbm(
                              fitted_adc[sample_orders], feedback_selector), float).tolist(),
                          selected_raw_dbm=np.asarray(adc_code_to_dbm(
                              y[sample_orders], feedback_selector), float).tolist(),
                          all_selected_at_or_above_threshold=True,
                          center_sample_error_pm=1000 * (x[sample_by_index[center]] - apex_nm)))
        selected_fit = bands[-1]['selected_fitted_dbm']
        selected_raw = bands[-1]['selected_raw_dbm']
        bands[-1]['boundary_height_gap_db'] = [selected_fit[0] - threshold_dbm,
                                               selected_fit[6] - threshold_dbm]
        paired_fit_left, paired_fit_right = selected_fit[2::-1], selected_fit[4:]
        symmetry.append(dict(group=len(groups), center_index=center,
                             center_target_nm=float(table[center].target_nm),
                             left_indices=paired_left, right_indices=paired_right,
                             target_drop_db=np.linspace(drop_db / 3, drop_db, 3).tolist(),
                             target_height_dbm=(peak_dbm - np.linspace(drop_db/3, drop_db, 3)).tolist(),
                             fitted_left_dbm=paired_fit_left, fitted_right_dbm=paired_fit_right,
                             fitted_pair_difference_db=[abs(left - right) for left, right
                                                        in zip(paired_fit_left, paired_fit_right)],
                             raw_pair_difference_db=[abs(left - right) for left, right
                                                     in zip(selected_raw[2::-1], selected_raw[4:])]))
    plan = build_manual_dense_plan(table, groups, x, y, source=source)
    plan.update(
        manual_selection_method='auto_makima_height_paired_7point', automatic_selection=True,
        points_per_peak=7, symmetry_checks=symmetry,
        symmetry_basis='fitted_dbm_height',
        selection_feedback_selector=feedback_selector,
        selection_prominence_db=AUTO_SELECTION_PROMINENCE_DB,
        selection_height_region='dbm_visual_upper_half',
        selection_height_fraction=AUTO_SELECTION_HEIGHT_FRACTION,
        selection_baseline_basis='higher_of_two_local_smoothed_dbm_valleys',
        selection_boundary_rule='closest_real_height_above_half_with_two_inner_rows',
        selection_fit_method='makima_piecewise_cubic_dense_adc',
        selection_threshold_unit='dbm', selection_band_checks=bands,
        selection_rule='seven_real_uniform_height_paired_points_in_dbm_visual_upper_half',
        automatic_selection_upstream_commit='fca441009c225cac1d1c60b0afd651f1b5d96b25',
        discarded_weak_peaks=len(detected) - MAX_PEAKS,
    )
    return plan


def build_manual_dense_plan(table, groups, wavelengths, values, *, source='manual_dense_selection'):
    """Rebuild an operator-edited peak×5 route against one dense CH1 spectrum.

    Automatic selection pairs optical heights, but an operator may move each
    displayed point independently. Manual groups only have to remain strictly
    increasing on the physical 2001-point table. A weak dense-shape preview
    does not reject the edit: the route is kept disabled until the normal live
    45-point reference independently passes all nine groups.
    """

    table = tuple(table)
    x = np.asarray(wavelengths, dtype=float)
    y = np.asarray(values, dtype=float)
    point_count = _validate_group_count(groups)
    group_size = len(groups[0])
    if (len(table) != 2001 or x.ndim != 1
            or y.shape != x.shape or len(x) < point_count
            or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))
            or np.any(np.diff(x) <= 0)):
        raise ValueError('手动选点需要完整2001点表和有序密集光谱')

    rows, checks, peaks = [], [], []
    measured_axis = np.asarray([point.measured_nm for point in table], dtype=float)
    previous = -1
    for group, indices in enumerate(groups, 1):
        if (len(indices) != group_size or any(type(index) is not int for index in indices)
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
        check = assess_peak_shape(
            [row['measured_nm'] for row in selected], sampled
        )
        middle = group_size // 2
        check.update(group=group, peak_center_index=indices[middle], spacing_nm=None,
                     point_spacings_nm=(np.diff(indices) * .02).tolist())
        fitted_center = check.get('fitted_center_nm')
        peaks.append(float(fitted_center) if fitted_center is not None
                     and math.isfinite(fitted_center) else selected[middle]['measured_nm'])
        checks.append(check)
        rows.extend(selected)
        previous = indices[-1]

    command(rows, 32, 1)
    return dict(
        schema=_plan_schema(point_count), rows=rows, peaks_nm=peaks,
        peak_count=len(groups), point_count=point_count, points_per_peak=group_size,
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
    group_size = points_per_peak_for_rows(rows)
    peak_count = len(rows) // group_size
    if (len(rows) % group_size or not 1 <= peak_count <= MAX_PEAKS
            or len(peaks) != peak_count or not all(math.isfinite(v) for v in peaks)):
        raise ValueError('临时点表峰数或每峰五点结构无效')
    for i, row in enumerate(rows):
        if (row.get('group') != i // group_size + 1 or
                not math.isfinite(row.get('target_nm', math.nan)) or
                not math.isfinite(row.get('measured_nm', math.nan)) or
                abs(row['target_nm'] - (1525 + .02 * row['index'])) > 1e-6 or
                abs(row['measured_nm'] - row['target_nm']) > .00201):
            raise ValueError('临时点表的分组或标定坐标无效')
    if (plan.get('automatic_selection') and group_size == 7
            and plan.get('symmetry_basis') == 'fitted_dbm_height'):
        _validate_height_pair_metadata(plan, peak_count)
    elif plan.get('automatic_selection') and group_size == 7:
        # Legacy wavelength-mirrored routes retain their original validation;
        # new height-paired routes must not be forced back onto a mirrored grid.
        for start in range(0, len(rows), 7):
            indices = [row['index'] for row in rows[start:start + 7]]
            if any(left + right != 2 * indices[3] for left, right in zip(indices, reversed(indices))):
                raise ValueError('自动七点路线左右不对称')
    plan['optical_accuracy_verified'] = False
    plan['peak_count'] = peak_count
    plan['point_count'] = len(rows)
    if plan.get('points_per_peak', group_size) != group_size:
        raise ValueError('保存的每峰点数与路线分组不一致')
    plan['points_per_peak'] = group_size
    checks = plan.get('selection_fit_checks')
    plan['selection_validation_required'] = not (
        isinstance(checks, list) and len(checks) == peak_count
        and all(isinstance(check, dict) and check.get('passed') for check in checks)
    )
    return plan


def _validate_height_pair_metadata(plan, peak_count):
    """Check height-pair identities/diagnostics without claiming optical accuracy."""
    checks, bands = plan.get('symmetry_checks'), plan.get('selection_band_checks')
    region = plan.get('selection_height_region')
    visual_half = region == 'dbm_visual_upper_half'
    drop = plan.get('selection_peak_drop_db')
    if (not isinstance(checks, list) or len(checks) != peak_count
            or not isinstance(bands, list) or len(bands) != peak_count
            or region not in (None, 'dbm_visual_upper_half')):
        raise ValueError('保存的高度配对明细不完整')
    if visual_half:
        if (plan.get('selection_height_fraction') != .5
                or plan.get('selection_baseline_basis') != 'higher_of_two_local_smoothed_dbm_valleys'
                or plan.get('selection_boundary_rule') != 'closest_real_height_above_half_with_two_inner_rows'):
            raise ValueError('保存的dBm视觉半高规则无效')
    elif type(drop) not in (int, float) or not math.isfinite(drop) or drop <= 0:
        # Old fixed-relative-cutoff records retain their original meaning.
        raise ValueError('保存的高度配对明细不完整')
    try:
        for group, (check, band) in enumerate(zip(checks, bands), 1):
            rows = plan['rows'][(group - 1)*7:group*7]
            indices = [row['index'] for row in rows]
            if (check['group'] != group or band['group'] != group
                    or check['center_index'] != indices[3]
                    or check['left_indices'] != indices[2::-1]
                    or check['right_indices'] != indices[4:]):
                raise ValueError('高度配对行号与保存路线不一致')
            fit = np.asarray(band['selected_fitted_dbm'], float)
            raw = np.asarray(band['selected_raw_dbm'], float)
            if visual_half:
                baseline, left_base, right_base = (float(band[key]) for key in
                    ('baseline_dbm', 'left_base_dbm', 'right_base_dbm'))
                drop = band['fitted_peak_dbm'] - band['threshold_dbm']
                if (not all(math.isfinite(value) for value in (baseline, left_base, right_base, drop))
                        or drop <= 0 or band['height_fraction'] != .5
                        or abs(baseline - max(left_base, right_base)) > 1e-8
                        or abs(band['threshold_dbm'] - .5*(band['fitted_peak_dbm'] + baseline)) > 1e-8
                        or abs(band['peak_drop_db'] - drop) > 1e-8):
                    raise ValueError('保存的dBm视觉半高下限不一致')
            if (fit.shape != (7,) or raw.shape != (7,)
                    or not np.all(np.isfinite(fit)) or not np.all(np.isfinite(raw))
                    or not math.isfinite(band['fitted_peak_dbm'])
                    or not math.isfinite(band['threshold_dbm'])
                    or abs(band['threshold_dbm'] - (band['fitted_peak_dbm'] - drop)) > 1e-8
                    or np.any(fit < band['threshold_dbm'] - 1e-8)
                    or np.any(raw < band['threshold_dbm'] - 1e-8)
                    or not rows[2]['measured_nm'] < band['fitted_peak_nm'] < rows[4]['measured_nm']):
                raise ValueError('保存的高度配对光强或峰顶范围无效')
            expected = [
                ('target_drop_db', np.linspace(drop/3, drop, 3)),
                ('fitted_left_dbm', fit[2::-1]), ('fitted_right_dbm', fit[4:]),
                ('fitted_pair_difference_db', abs(fit[2::-1] - fit[4:])),
                ('raw_pair_difference_db', abs(raw[2::-1] - raw[4:])),
            ]
            if visual_half:
                expected.append(('target_height_dbm', band['fitted_peak_dbm'] - np.linspace(drop/3, drop, 3)))
                gap = np.asarray(band['boundary_height_gap_db'], float)
                if gap.shape != (2,) or not np.allclose(gap, fit[[0, 6]] - band['threshold_dbm'],
                                                       rtol=0, atol=1e-8):
                    raise ValueError('保存的半高边界偏差与明细不一致')
                if np.any(fit[1:3] < fit[0] - 1e-8) or np.any(fit[4:6] < fit[6] - 1e-8):
                    raise ValueError('保存的半高边界点不是该侧最低点')
            for key, values in expected:
                saved = np.asarray(check[key], float)
                if saved.shape != (3,) or not np.allclose(saved, values, rtol=0, atol=1e-8):
                    raise ValueError('保存的高度配对差值与明细不一致')
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError('保存的高度配对明细无效') from exc


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
    if (not payload.get('complete') or len(results) not in (set(range(5, 46, 5)) | set(range(7, 64, 7)))
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
    dominant = np.flatnonzero(
        detection_scores
        >= float(np.max(detection_scores)) * SELECTION_DBM_SCORE_FRACTION
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
            adc_channel=1, single_sample=False, fast_dac_writes=False):
    if type(single_sample) is not bool or type(fast_dac_writes) is not bool:
        raise ValueError('Sampling/DAC options must be boolean')
    if fast_dac_writes and not single_sample:
        raise ValueError('Fast DAC writes require the Beta single-sample protocol')
    if single_sample:
        spacing_us = 0  # Exactly one wait and one physical ADC read per point.
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
    group_size = points_per_peak_for_rows(rows)
    if (type(tag) is not int or not 1 <= tag <= 0xffffffff
            or point_count not in range(group_size, group_size * MAX_PEAKS + 1,
                                        group_size)):
        raise ValueError("5～45 rows (a multiple of five) and a nonzero uint32 tag required")
    if (type(first_delay_us) is not int or not 50 <= first_delay_us <= 850 or first_delay_us % 25
            or (not single_sample and (type(spacing_us) is not int or not 50 <= spacing_us <= 600
                                       or spacing_us % 25 or first_delay_us + spacing_us > 900))):
        raise ValueError("首次等待须为50～850 µs、双采样间隔须为50～600 µs，均为25 µs整数倍且合计不超过900 µs")
    if type(boundary_extra_us) is not int or not 0 <= boundary_extra_us <= 3000 or boundary_extra_us % 25:
        raise ValueError('跨峰额外等待须为0～3000 µs的25 µs整数倍；总等待不限，允许低于15 Hz')
    packet = bytearray(808)
    protocol_version = (
        7 if single_sample else
        6 if group_size == 7 else
        3 if adc_channel == 1 and point_count == MAX_POINTS else
        4 if adc_channel == 1 else
        5
    )
    packet[:6] = bytes((0xff, 0xff, 3, 10, protocol_version, point_count))
    packet[6:8] = first_delay_us.to_bytes(2, 'big')
    packet[8:12] = tag.to_bytes(4, 'big')
    packet[12:16] = (b'T45!' if protocol_version == 3 else
                     b'TVR!' if protocol_version == 4 else
                     b'T7S!' if protocol_version == 6 else
                     b'T1S!' if protocol_version == 7 else b'TVC!')
    packet[16:18] = cycles.to_bytes(2, 'big')
    packet[18] = 16
    packet[19] = ((0 if feedback_selector == 2 else feedback_selector + 1)
                  if protocol_version in (3, 4)
                  else (adc_channel << 4) | feedback_selector)
    packet[20:22] = spacing_us.to_bytes(2, 'big')
    packet[22:24] = boundary_extra_us.to_bytes(2,'big')
    if single_sample:
        packet[22:24] = bytes((group_size, int(fast_dac_writes)))
    previous = -1
    for i, row in enumerate(rows):
        index, codes = row['index'], tuple(row['codes'])
        if type(index) is not int or not previous < index <= 2000 or len(codes) != 5:
            raise ValueError("Invalid physical table order")
        if any(type(c) is not int or not 0 <= c <= limit for c, limit in zip(codes, LIMITS)):
            raise ValueError("DAC current limit exceeded")
        previous = index
        if protocol_version not in (6, 7):
            packet[24+i*12:36+i*12] = b''.join(v.to_bytes(2, 'big') for v in (index, *codes))
    if protocol_version in (6, 7):
        # Fixed 808-byte transport stays unchanged. Lossless bit packing uses
        # the existing code ceilings, not rounding/truncation of DAC values.
        waits = ([first_delay_us + (boundary_extra_us if i % group_size == 0 else 0)
                  for i in range(point_count)] if point_delays_us is None
                 else list(point_delays_us))
        if (len(waits) != point_count or (point_delays_us is not None and boundary_extra_us)
                or any(type(v) is not int or not 50 <= v <= 15000 or v % 25 for v in waits)):
            raise ValueError('七点路线逐点等待须为50～15000 µs的25 µs整数倍')
        bit = 24 * 8
        for row, wait in zip(rows, waits):
            for value, width in zip((row['index'], *row['codes'], wait // 25), V6_FIELD_BITS):
                for shift in range(width - 1, -1, -1):
                    packet[bit // 8] |= ((value >> shift) & 1) << (7 - bit % 8)
                    bit += 1
        packet[V6_CRC_OFFSET:V6_CRC_OFFSET + 4] = binascii.crc32(packet[:V6_CRC_OFFSET]).to_bytes(4, 'big')
        return bytes(packet)
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
        if kind in (10, 14, 16) and status == 127:
            length = 20
        elif kind in (14, 16):
            if len(buffer) < 30:
                return
            point_count = buffer[29]
            if point_count not in (set(range(5, 46, 5)) | set(range(7, 64, 7))):
                del buffer[0]
                continue
            length = 46 + point_count * (11 if kind == 16 else 13 if status & 2 else 9)
        elif kind in (15, 17):
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
           point_count=MAX_POINTS, adc_channel=1, points_per_peak=5,
           single_sample=False, fast_dac_writes=False):
    if single_sample:
        spacing_us = 0
    u16 = lambda at: int.from_bytes(wire[at:at+2], 'big')
    u32 = lambda at: int.from_bytes(wire[at:at+4], 'big')
    if len(wire) == 20 and wire[:2] == b'\xd9\x9d' and wire[2] in (10, 14, 16) and wire[3] == 127:
        if u32(4) != tag or wire[-2:] != b'\x9d\xd9':
            raise ValueError("Rejection identity mismatch")
        raise RuntimeError(f"板卡拒绝临时{point_count}点协议（{u16(8):04x}），请使用配套固件并检查设备状态")
    expected_narrow = 46 + point_count * 9
    expected_wide = 46 + point_count * (11 if single_sample else 13)
    if len(wire) not in (expected_narrow, expected_wide, 42) or wire[:2] != b'\xd9\x9d' or wire[-2:] != b'\x9d\xd9' or u32(len(wire)-6) != binascii.crc32(wire[:-6]):
        raise ValueError("Frame CRC/length/trailer mismatch")
    if u32(4) != tag:
        raise ValueError("Session tag mismatch")
    if len(wire) == 42 and wire[2] == (17 if single_sample else 15):
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
    if (wire[2] != (16 if single_sample else 14) or (single_sample and not wide)
            or len(wire) != (expected_wide if wide else expected_narrow)
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
        at = order_end + i*(10 if single_sample else 12 if wide else 8)
        first = u16(at)
        second = first if single_sample else u16(at+2)
        write, end = ((u32(at+2), u32(at+6)) if single_sample else
                      (u32(at+4), u32(at+8)) if wide else (u16(at+4), u16(at+6)))
        expected_delay = first_delay_us+spacing_us+(boundary_extra_us if i%points_per_peak==0 else 0)
        if point_delays_us is not None:
            expected_delay = point_delays_us[i] + spacing_us
        if max(first, second) > 4095 or not previous < write < end <= u32(20)+1 or end-write < expected_delay:
            raise ValueError("ADC/timestamp validation failed")
        previous = end
        records.append(dict(adc_code=first, write_end_us=write, sample_end_us=end)
                       if single_sample else dict(first_code=first, second_code=second,
                                                  write_end_us=write, second_end_us=end))
    return dict(kind='data', sequence=sequence, cycle_start_us=u32(16), elapsed_us=u32(20),
                records=records, i2c_errors=u16(34), spi_errors=u32(36), firmware=u32(8),
                feedback_selector=feedback_selector, adc_channel=adc_channel,
                signal_channel_name=SIGNAL_CHANNELS[adc_channel], points_per_peak=points_per_peak,
                single_sample=single_sample, samples_per_point=(1 if single_sample else 2),
                fast_dac_writes=fast_dac_writes)


def frame_adc_values(frame):
    key = 'adc_code' if frame.get('single_sample') else 'second_code'
    return [record[key] for record in frame['records']]


def quality(frame):
    single = bool(frame.get('single_sample'))
    pairs = (np.asarray(frame_adc_values(frame), float).reshape(-1, 1) if single else
             np.asarray([[r['first_code'], r['second_code']] for r in frame['records']], float))
    values = pairs[:, 0 if single else 1]
    clipped = bool(np.any(pairs >= 4080))
    settling = False if single else bool(np.any(abs(pairs[:, 1]-pairs[:, 0]) > np.maximum(8, abs(values)*.02)))
    group_size = frame.get('points_per_peak', 5)
    if len(values) % group_size:
        raise ValueError('Frame point count is not divisible into five-point peaks')
    edge = any(int(np.argmax(group)) in (0, group_size - 1)
               for group in values.reshape(-1, group_size))
    return dict(warmed_up=frame['sequence'] >= 16, clipped=clipped, unsettled=settling,
                pair_difference_large=(None if single else settling),
                difference_check_performed=not single, stability_verified=False,
                peak_at_edge=edge, optical_accuracy_verified=False)


def execute_device(device, plan, output, *, cycles=256, on_frame=None, should_stop=lambda: False,
                   first_delay_us=600, spacing_us=50, boundary_extra_us=0,
                   feedback_selector=2, point_delays_us=None, adc_channel=1,
                   single_sample=False, fast_dac_writes=False):
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
    group_size = points_per_peak_for_rows(plan['rows'])
    peak_count = point_count // group_size
    packet = command(plan['rows'], cycles, tag, first_delay_us, spacing_us,
                     boundary_extra_us, feedback_selector, point_delays_us,
                     adc_channel, single_sample, fast_dac_writes)
    spacing_us = 0 if single_sample else spacing_us
    crc = binascii.crc32(packet[:V6_CRC_OFFSET] if packet[4] in (6, 7) else
                        packet[:662] if point_delays_us is not None else packet[24:564])
    report = dict(schema=('temporary_test_45_capture_v1' if point_count == MAX_POINTS
                          else 'temporary_test_variable_capture_v2'),
                  plan=plan, frames=[], complete=False,
                  point_count=point_count, peak_count=peak_count, points_per_peak=group_size,
                  first_delay_us=first_delay_us, spacing_us=spacing_us,
                  single_sample=single_sample, samples_per_point=(1 if single_sample else 2),
                  fast_dac_writes=fast_dac_writes, protocol_version=packet[4],
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
                               point_delays_us, point_count, adc_channel, group_size,
                               single_sample, fast_dac_writes)
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
                frame['host_received_s'] = time.perf_counter()
                arrival_lookup = getattr(device, 'take_frame_arrival_info', None)
                if callable(arrival_lookup):
                    arrival = arrival_lookup(wire)
                    if (isinstance(arrival, tuple) and len(arrival) == 2
                            and isinstance(arrival[0], float)
                            and isinstance(arrival[1], int)):
                        frame['socket_arrival_s'] = arrival[0]
                        frame['socket_receive_batch'] = arrival[1]
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
        # Shutdown helpers clear the serial input buffer.  The LAN adapter
        # also clears its counters there, so retain the acquisition counters
        # before sending the mandatory disarm/shutter commands.
        stats = getattr(device, 'transport_stats', None)
        if isinstance(stats, dict):
            report['transport_stats_before_shutdown'] = dict(stats)
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
