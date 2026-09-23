"""Five measured points -> bounded Gaussian+constant estimate, never repaired data.

Every finite five-point arrangement gets a numerical fit. Shape checks are
reported as quality warnings instead of suppressing the estimated center.
The wavelength axis is the stored calibration, not live wavelength-meter truth.
Quality gates are display heuristics, not an optical accuracy certification.
"""
import numpy as np


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
            optical_accuracy_verified=False,
        )

    x = np.asarray(wavelength_nm, dtype=float)
    y = np.asarray(adc, dtype=float)
    if x.shape != (5,) or y.shape != (5,) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
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
    tolerance = max(8., height*.03)
    rises_to_apex = apex > 0 and not np.any(
        np.diff(y[:apex + 1]) < -tolerance
    )
    falls_after_apex = apex < 4 and not np.any(
        np.diff(y[apex:]) > tolerance
    )
    if not (rises_to_apex and falls_after_apex):
        warn('五点未形成清晰的先升后降单峰')

    origin = x[2]
    scale = float(np.median(np.diff(x)))
    u = (x-origin)/scale
    low, high = u[0]+.25, u[-1]-.25
    sigma_low, sigma_high = .35, float(u[-1]-u[0])

    def solve(centers, widths):
        centers, widths = np.meshgrid(centers, widths, indexing='ij')
        centers, widths = centers.ravel(), widths.ravel()
        g = np.exp(-.5*((u[None, :]-centers[:, None])/widths[:, None])**2)
        centered = g-g.mean(axis=1)[:, None]
        amplitude = np.sum(centered*(y-y.mean()), axis=1)/np.maximum(np.sum(centered**2, axis=1), 1e-12)
        baseline = y.mean()-amplitude*g.mean(axis=1)
        # Nonnegative baseline least squares boundary; no changes to raw y.
        negative = baseline < 0
        baseline[negative] = 0
        amplitude[negative] = np.sum(g[negative]*y, axis=1)/np.sum(g[negative]**2, axis=1)
        prediction = baseline[:, None]+amplitude[:, None]*g
        sse = np.sum((prediction-y)**2, axis=1)
        sse[(amplitude < 0) | ~np.isfinite(sse)] = np.inf
        best = int(np.argmin(sse))
        return centers[best], widths[best], amplitude[best], baseline[best], sse[best]

    if height <= 1e-12:
        center = float(u[2])
        sigma = max(sigma_low, float(u[-1] - u[0]) / 4.0)
        amplitude = 0.0
        baseline = float(np.mean(y))
        sse = 0.0
    else:
        center_grid = np.linspace(low, high, 61)
        width_grid = np.linspace(sigma_low, sigma_high, 32)
        coarse = solve(center_grid, width_grid)
        dc, ds = center_grid[1]-center_grid[0], width_grid[1]-width_grid[0]
        center, sigma, amplitude, baseline, sse = solve(
            np.linspace(max(low, coarse[0]-dc), min(high, coarse[0]+dc), 41),
            np.linspace(max(sigma_low, coarse[1]-ds), min(sigma_high, coarse[1]+ds), 41))
        if not np.all(np.isfinite([center, sigma, amplitude, baseline, sse])):
            center = float(u[apex] if height > 0 else u[2])
            sigma = max(sigma_low, float(u[-1] - u[0]) / 4.0)
            baseline = max(0.0, float(np.min(y)))
            amplitude = max(0.0, float(np.max(y) - baseline))
            prediction = baseline + amplitude * np.exp(
                -.5 * ((u - center) / sigma) ** 2
            )
            sse = float(np.sum((prediction - y) ** 2))
            warn('使用保底高斯估计')

    rmse = float(np.sqrt(sse/5))
    variance = float(np.sum((y-y.mean())**2))
    r2 = 0.0 if variance <= 1e-12 else float(1-sse/variance)
    if r2 < .90:
        warn(f'R²偏低({r2:.3f})')
    if height <= 1e-12 or rmse/max(height, 1e-12) > .15:
        warn('拟合残差相对幅值偏大')
    if center <= low+.01 or center >= high-.01:
        warn('拟合中心贴近采样窗口边界')
    if sigma <= sigma_low+.01 or sigma >= sigma_high-.01:
        warn('峰宽受到五点窗口边界限制')
    if height > 1e-12 and amplitude > 2*height:
        warn('拟合幅值异常放大')
    if 0 < apex < 4 and not u[apex-1] <= center <= u[apex+1]:
        warn('拟合中心与离散峰顶不一致')

    center_nm = float(origin+center*scale)
    sigma_nm = float(sigma*scale)
    dense_x = np.linspace(x[0], x[-1], 101)
    dense_y = baseline+amplitude*np.exp(-.5*((dense_x-center_nm)/sigma_nm)**2)
    valid = not issues
    reason = (
        '质量通过·五点估计·精度未验证'
        if valid else '⚠ 可能有问题：' + '；'.join(issues)
    )
    return dict(
        valid=valid,
        fit_available=True,
        quality_warning=not valid,
        quality_issues=list(issues),
        reason=reason,
        center_nm=center_nm,
        sigma_nm=sigma_nm,
        baseline_adc=float(baseline),
        amplitude_adc=float(amplitude),
        r_squared=r2,
        rmse_adc=rmse,
        curve_x_nm=dense_x.tolist(),
        curve_adc=dense_y.tolist(),
        optical_accuracy_verified=False,
    )


def fit_temporary_frame(rows, values, warmed_up=True):
    if (len(rows) != len(values) or len(rows) not in range(5, 46, 5)):
        raise ValueError('实时拟合需要1～9个完整五点峰')
    peak_count = len(rows) // 5
    if not warmed_up:
        return [dict(valid=False, fit_available=False, quality_warning=True,
                     reason='预热中', center_nm=None, r_squared=None)
                for _ in range(peak_count)]
    return [fit_five_points([r['measured_nm'] for r in rows[i:i+5]], values[i:i+5])
            for i in range(0, len(rows), 5)]
