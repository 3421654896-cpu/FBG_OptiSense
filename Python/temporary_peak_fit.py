"""Five measured points -> modified-Akima cubic peak estimate.

The realtime route always contains exactly five measured points per grating.
A symmetric Gaussian forces both tails toward one constant baseline and can
therefore draw artificial platforms when the real FBG peak is asymmetric.
This module instead uses a modified Akima (Makima) piecewise-cubic interpolant:
it passes through all five physical ADC samples, permits different left/right
flanks, limits cubic overshoot, and derives the centre from the interpolant's
local maximum.  No extra wavelength points are acquired or invented as input.

Shape checks are quality warnings rather than destructive gates.  The stored
wavelength axis remains the existing board calibration, not live wavelength-
meter truth.
"""

import numpy as np
from scipy.interpolate import Akima1DInterpolator


FIT_METHOD = 'makima_piecewise_cubic_5point'
_RENDER_POINT_COUNT = 401


def _half_height_width(curve_x, curve_y, center_nm, baseline_adc, peak_adc):
    """Return effective FWHM and whether either crossing left the 5-point window."""

    half_level = float(baseline_adc + 0.5 * (peak_adc - baseline_adc))
    center_index = int(np.argmin(np.abs(curve_x - center_nm)))

    left_candidates = np.flatnonzero(curve_y[:center_index + 1] <= half_level)
    right_candidates = np.flatnonzero(curve_y[center_index:] <= half_level)
    limited = not len(left_candidates) or not len(right_candidates)

    if len(left_candidates):
        left_low = int(left_candidates[-1])
        left_high = min(left_low + 1, center_index)
        if left_low == left_high or curve_y[left_high] == curve_y[left_low]:
            left_nm = float(curve_x[left_low])
        else:
            left_nm = float(np.interp(
                half_level,
                [curve_y[left_low], curve_y[left_high]],
                [curve_x[left_low], curve_x[left_high]],
            ))
    else:
        left_nm = float(curve_x[0])

    if len(right_candidates):
        right_high = center_index + int(right_candidates[0])
        right_low = max(center_index, right_high - 1)
        if right_low == right_high or curve_y[right_low] == curve_y[right_high]:
            right_nm = float(curve_x[right_high])
        else:
            # The right flank is descending; reverse both arrays for np.interp.
            right_nm = float(np.interp(
                half_level,
                [curve_y[right_high], curve_y[right_low]],
                [curve_x[right_high], curve_x[right_low]],
            ))
    else:
        right_nm = float(curve_x[-1])

    return max(0.0, right_nm - left_nm), bool(limited)


def _gaussian_shape_diagnostics(x, y):
    """Return legacy residual metrics without controlling centre or rendering.

    An exact cubic interpolant has zero sample residual by construction, so its
    R² cannot distinguish a clean one-summit peak from five irregular points.
    Keep the historical bounded Gaussian only as a hidden shape-consistency
    diagnostic.  The reported centre and displayed curve always come from the
    Makima cubic; this helper cannot create the former platform artefact.
    """

    origin = float(x[2])
    scale = float(np.median(np.diff(x)))
    u = (x - origin) / scale
    height = float(np.ptp(y))
    if height <= 1e-12:
        return 0.0, 0.0
    low, high = float(u[0] + 0.25), float(u[-1] - 0.25)
    sigma_low = 0.35
    sigma_high = float(u[-1] - u[0])

    centers, widths = np.meshgrid(
        np.linspace(low, high, 61),
        np.linspace(sigma_low, sigma_high, 32),
        indexing='ij',
    )
    centers = centers.ravel()
    widths = widths.ravel()
    gaussian = np.exp(
        -0.5 * ((u[None, :] - centers[:, None]) / widths[:, None]) ** 2
    )
    centered = gaussian - gaussian.mean(axis=1)[:, None]
    centered_y = y - y.mean()
    amplitude = (centered @ centered_y) / np.maximum(
        np.sum(centered**2, axis=1), 1e-12
    )
    baseline = y.mean() - amplitude * gaussian.mean(axis=1)
    negative = baseline < 0.0
    baseline[negative] = 0.0
    amplitude[negative] = (
        np.sum(gaussian[negative] * y, axis=1)
        / np.maximum(np.sum(gaussian[negative] ** 2, axis=1), 1e-12)
    )
    predicted = baseline[:, None] + amplitude[:, None] * gaussian
    sse = np.sum((predicted - y) ** 2, axis=1)
    sse[(amplitude < 0.0) | ~np.isfinite(sse)] = np.inf
    best_sse = float(np.min(sse))
    if not np.isfinite(best_sse):
        return 0.0, float('inf')
    variance = float(np.sum((y - y.mean()) ** 2))
    r_squared = 0.0 if variance <= 1e-12 else 1.0 - best_sse / variance
    rmse = float(np.sqrt(best_sse / 5.0))
    return float(r_squared), rmse


def fit_five_points(wavelength_nm, adc):
    def unavailable(reason):
        return dict(
            valid=False,
            fit_available=False,
            quality_warning=True,
            quality_issues=[reason],
            reason=f'无法拟合：{reason}',
            center_nm=None,
            r_squared=None,
            fit_method=FIT_METHOD,
            input_point_count=5,
            optical_accuracy_verified=False,
        )

    x = np.asarray(wavelength_nm, dtype=float)
    y = np.asarray(adc, dtype=float)
    if (x.shape != (5,) or y.shape != (5,)
            or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))):
        return unavailable('数据不完整')
    if np.any(y < 0):
        return unavailable('ADC无效')

    issues = []

    def warn(message):
        if message not in issues:
            issues.append(message)

    order = np.argsort(x, kind='stable')
    if not np.array_equal(order, np.arange(5)):
        x, y = x[order], y[order]
        warn('波长点顺序已重排')
    if np.any(np.diff(x) <= 0):
        return unavailable('波长坐标重复或无效')

    if np.max(y) >= 4080:
        warn('ADC接近饱和')
    height = float(np.ptp(y))
    if height < 24:
        warn('幅值过低，中心不充分约束')
    apex = int(np.argmax(y))
    if apex in (0, 4):
        warn('离散峰值位于窗口边缘')
    tolerance = max(8.0, height * 0.03)
    rises_to_apex = apex > 0 and not np.any(
        np.diff(y[:apex + 1]) < -tolerance
    )
    falls_after_apex = apex < 4 and not np.any(
        np.diff(y[apex:]) > tolerance
    )
    if not (rises_to_apex and falls_after_apex):
        warn('五点未形成清晰的先升后降单峰')

    if height <= 1e-12:
        center_nm = float(x[2])
        peak_adc = float(y[2])
        dense_x = np.unique(np.concatenate((
            np.linspace(x[0], x[-1], _RENDER_POINT_COUNT), x
        )))
        dense_y = np.full_like(dense_x, float(np.mean(y)))
        baseline_adc = float(np.mean(y))
        fwhm_nm = float(x[-1] - x[0])
        width_limited = True
    else:
        try:
            spline = Akima1DInterpolator(
                x, y, method='makima', extrapolate=False
            )
            stationary = np.asarray(
                spline.derivative().roots(extrapolate=False), dtype=float
            )
            maxima = [
                float(value) for value in stationary
                if (np.isfinite(value) and x[0] <= value <= x[-1]
                    and float(spline.derivative(2)(value)) < 0.0)
            ]
            if maxima:
                center_nm = max(maxima, key=lambda value: float(spline(value)))
                if len(maxima) > 1:
                    warn('三次样条出现多个局部峰')
            else:
                center_nm = float(x[apex])
                warn('三次样条未找到内部极大值，使用离散峰顶')

            dense_x = np.unique(np.concatenate((
                np.linspace(x[0], x[-1], _RENDER_POINT_COUNT), x,
                np.asarray((center_nm,)),
            )))
            dense_y = np.asarray(spline(dense_x), dtype=float)
            if not np.all(np.isfinite(dense_y)):
                raise FloatingPointError('Makima输出包含非有限值')
            peak_adc = float(spline(center_nm))
            baseline_adc = max(0.0, float(min(y[0], y[-1])))
            fwhm_nm, width_limited = _half_height_width(
                dense_x, dense_y, center_nm, baseline_adc, peak_adc
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            return unavailable('三次样条计算失败')

    amplitude_adc = max(0.0, peak_adc - baseline_adc)
    effective_sigma_nm = float(
        fwhm_nm / 2.354820045 if fwhm_nm > 0 else 0.0
    )

    # Makima is an exact interpolant, so sample residual is zero by definition.
    # Use the historical bounded-Gaussian residual only as an independent
    # shape-consistency score; it never controls the centre or rendered curve.
    r2, rmse = _gaussian_shape_diagnostics(x, y)
    if r2 < 0.90:
        warn(f'五点峰形一致性偏低({r2:.3f})')
    if height <= 1e-12 or rmse / max(height, 1e-12) > 0.15:
        warn('五点峰形残差相对幅值偏大')
    # The five points may intentionally stop slightly above the half-height
    # crossing.  This limits the reported effective width but does not make the
    # spline centre invalid; expose it as metadata instead of rejecting an
    # otherwise clean five-point peak.
    if center_nm <= x[0] or center_nm >= x[-1]:
        warn('拟合中心贴近采样窗口边界')
    if 0 < apex < 4 and not x[apex - 1] <= center_nm <= x[apex + 1]:
        warn('拟合中心与离散峰顶不一致')
    overshoot = max(0.0, peak_adc - float(np.max(y)))
    if height > 0 and overshoot > 0.15 * height:
        warn('三次样条峰顶过冲')
    if height > 0 and float(np.min(dense_y)) < -0.05 * height:
        warn('三次样条边缘下冲')

    valid = not issues
    reason = (
        '质量通过·Makima三次样条·五点估计·精度未验证'
        if valid else '⚠ 可能有问题：' + '；'.join(issues)
    )
    return dict(
        valid=valid,
        fit_available=True,
        quality_warning=not valid,
        quality_issues=list(issues),
        reason=reason,
        center_nm=float(center_nm),
        sigma_nm=effective_sigma_nm,
        fwhm_nm=float(fwhm_nm),
        fwhm_limited_by_five_point_window=bool(width_limited),
        baseline_adc=float(baseline_adc),
        amplitude_adc=float(amplitude_adc),
        peak_adc=float(peak_adc),
        r_squared=r2,
        rmse_adc=rmse,
        quality_metric='bounded_gaussian_residual_diagnostic_only',
        curve_x_nm=dense_x.tolist(),
        curve_adc=dense_y.tolist(),
        fit_method=FIT_METHOD,
        input_point_count=5,
        optical_accuracy_verified=False,
    )


def fit_temporary_frame(rows, values, warmed_up=True):
    if (len(rows) != len(values) or len(rows) not in range(5, 46, 5)):
        raise ValueError('实时拟合需要1～9个完整五点峰')
    peak_count = len(rows) // 5
    if not warmed_up:
        return [dict(valid=False, fit_available=False, quality_warning=True,
                     reason='预热中', center_nm=None, r_squared=None,
                     fit_method=FIT_METHOD, input_point_count=5)
                for _ in range(peak_count)]
    return [fit_five_points(
                [r['measured_nm'] for r in rows[i:i + 5]],
                values[i:i + 5],
            )
            for i in range(0, len(rows), 5)]
