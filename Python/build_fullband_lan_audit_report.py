"""Export a compact CSV, Markdown summary and plot from a LAN AQ6150B audit."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def quantile(values, fraction):
    ordered = sorted(float(value) for value in values)
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def smsr(row):
    value = row["reading"].get("side_mode_suppression_db")
    return 99.0 if value is None else float(value)


def classify(row):
    if abs(float(row["wavelength_drift_from_calibration_pm"])) > 20.0:
        return "mode_hop"
    if smsr(row) < 20.0:
        return "low_smsr"
    if (
        abs(float(row["wavelength_error_pm"])) > 2.0
        or abs(float(row["power_relative_error_percent"])) > 0.5
    ):
        return "calibration_drift"
    return "pass"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--prefix", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = sorted(payload["rows"].values(), key=lambda row: float(row["target_nm"]))
    for row in rows:
        row["audit_status"] = classify(row)

    args.prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.prefix.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        fieldnames = (
            "index", "target_nm", "original_measured_nm", "current_measured_nm",
            "wavelength_error_pm", "wavelength_drift_from_original_pm",
            "original_power_mw", "current_power_mw",
            "power_drift_from_original_percent", "smsr_db", "peak_count",
            "audit_status",
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "index": row["index"],
                "target_nm": f"{float(row['target_nm']):.6f}",
                "original_measured_nm": f"{float(row['calibration_reference']['measured_wavelength_nm']):.8f}",
                "current_measured_nm": f"{float(row['reading']['wavelength_nm']):.8f}",
                "wavelength_error_pm": f"{float(row['wavelength_error_pm']):.3f}",
                "wavelength_drift_from_original_pm": f"{float(row['wavelength_drift_from_calibration_pm']):.3f}",
                "original_power_mw": f"{float(row['calibration_reference']['measured_power_mw']):.9f}",
                "current_power_mw": f"{float(row['reading']['power_mw']):.9f}",
                "power_drift_from_original_percent": f"{float(row['power_drift_from_calibration_percent']):.4f}",
                "smsr_db": "single_peak" if row["reading"].get("side_mode_suppression_db") is None else f"{smsr(row):.4f}",
                "peak_count": row["reading"]["peak_count"],
                "audit_status": row["audit_status"],
            })

    normal = [row for row in rows if row["audit_status"] not in ("mode_hop", "low_smsr")]
    mode_hops = [row for row in rows if row["audit_status"] == "mode_hop"]
    low_smsr = [row for row in rows if row["audit_status"] == "low_smsr"]
    drift = [float(row["wavelength_drift_from_calibration_pm"]) for row in normal]
    target_error = [float(row["wavelength_error_pm"]) for row in normal]
    power = [float(row["reading"]["power_mw"]) for row in normal]
    power_drift = [float(row["power_drift_from_calibration_percent"]) for row in normal]

    width, height = 1600, 1000
    x_values = [float(row["target_nm"]) for row in rows]
    x_min, x_max = min(x_values), max(x_values)
    panels = (
        (
            "Wavelength drift from old calibration (pm; clipped)",
            [max(-10.0, min(10.0, float(row["wavelength_drift_from_calibration_pm"]))) for row in rows],
            -10.0, 10.0, 0.0,
        ),
        (
            "AQ6150B optical power (mW)",
            [float(row["reading"]["power_mw"]) for row in rows],
            0.5, 1.72, 1.53,
        ),
        (
            "Side-mode suppression ratio (dB; clipped)",
            [min(40.0, smsr(row)) for row in rows],
            0.0, 40.0, 20.0,
        ),
    )
    left, right = 95, width - 35
    panel_height = 265
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="70" y="34" font-family="Arial" font-size="18" fill="#0f172a">2001-point LAN replay / AQ6150B audit (2026-09-04)</text>',
    ]
    for panel_index, (title, values, y_min, y_max, reference) in enumerate(panels):
        top = 62 + panel_index * 305
        bottom = top + panel_height
        svg.append(
            f'<rect x="{left}" y="{top}" width="{right-left}" height="{panel_height}" fill="#ffffff" stroke="#cbd5e1"/>'
        )
        for grid in range(1, 5):
            y = top + panel_height * grid / 5.0
            svg.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#e2e8f0"/>')
        reference_y = bottom - (reference - y_min) / (y_max - y_min) * panel_height
        svg += [
            f'<line x1="{left}" y1="{reference_y:.2f}" x2="{right}" y2="{reference_y:.2f}" stroke="#f59e0b" stroke-width="1.4" stroke-dasharray="7 5"/>',
            f'<text x="{left}" y="{top-9}" font-family="Arial" font-size="14" fill="#0f172a">{title}</text>',
            f'<text x="15" y="{top+12}" font-family="Arial" font-size="12" fill="#334155">{y_max:g}</text>',
            f'<text x="15" y="{bottom}" font-family="Arial" font-size="12" fill="#334155">{y_min:g}</text>',
        ]
        points = []
        for row, x_value, y_value in zip(rows, x_values, values):
            px = left + (x_value - x_min) / (x_max - x_min) * (right - left)
            py = bottom - (y_value - y_min) / (y_max - y_min) * panel_height
            points.append(f"{px:.2f},{py:.2f}")
            if row["audit_status"] in ("mode_hop", "low_smsr"):
                svg.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4" fill="#e03131"/>')
        svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#60a5fa" stroke-width="0.8"/>')
        if panel_index == 2:
            for tick in range(9):
                value = x_min + tick * (x_max - x_min) / 8.0
                px = left + tick * (right - left) / 8.0
                svg.append(f'<text x="{px-22:.2f}" y="{bottom+22}" font-family="Arial" font-size="12" fill="#334155">{value:.0f}</text>')
            svg.append(f'<text x="{(left+right)//2-75}" y="{bottom+48}" font-family="Arial" font-size="13" fill="#334155">Target wavelength (nm)</text>')
    svg.append('</svg>')
    plot_path = args.prefix.with_suffix(".svg")
    plot_path.write_text("\n".join(svg), encoding="utf-8")

    focus_targets = (1528.40, 1528.44, 1528.46, 1538.62)
    focus_rows = [min(rows, key=lambda row: abs(float(row["target_nm"]) - target)) for target in focus_targets]
    report = [
        "# 2001 点局域网复测报告（2026-09-04）",
        "",
        f"- 完整实测：{len(rows)}/2001 点；结束时 SOA 关光确认：{bool(payload.get('soa_shutter_confirmed'))}。",
        f"- 正常波长支路且 SMSR≥20 dB：{len(normal)} 点；明显错模：{len(mode_hops)} 点；SMSR<20 dB：{len(low_smsr)} 点。",
        f"- 正常点相对旧标定的波长漂移：中位数 {statistics.median(drift):+.3f} pm，5%～95% 为 {quantile(drift, .05):+.3f}～{quantile(drift, .95):+.3f} pm，最大绝对值 {max(abs(value) for value in drift):.3f} pm。",
        f"- 正常点相对目标波长：中位误差 {statistics.median(target_error):+.3f} pm；±2/±3/±4 pm 内分别为 {sum(abs(value)<=2 for value in target_error)}/{sum(abs(value)<=3 for value in target_error)}/{sum(abs(value)<=4 for value in target_error)} 点。",
        f"- 正常点当前功率：最小 {min(power):.4f} mW，中位 {statistics.median(power):.4f} mW，最大 {max(power):.4f} mW。",
        f"- 相对旧标定功率变化：中位 {statistics.median(power_drift):+.3f}%，95% 绝对变化≤{quantile([abs(value) for value in power_drift], .95):.3f}%。",
        "",
        "## 明显错模",
        "",
        "| 目标/nm | 当前主峰/nm | 相对旧标定漂移/pm | 功率/mW | SMSR/dB |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in mode_hops:
        report.append(
            f"| {float(row['target_nm']):.2f} | {float(row['reading']['wavelength_nm']):.8f} | "
            f"{float(row['wavelength_drift_from_calibration_pm']):+.2f} | "
            f"{float(row['reading']['power_mw']):.4f} | {smsr(row):.2f} |"
        )
    report += [
        "",
        "## 重点点位",
        "",
        "| 目标/nm | 当前主峰/nm | 相对旧标定漂移/pm | 功率/mW | SMSR/dB |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in focus_rows:
        report.append(
            f"| {float(row['target_nm']):.2f} | {float(row['reading']['wavelength_nm']):.8f} | "
            f"{float(row['wavelength_drift_from_calibration_pm']):+.2f} | "
            f"{float(row['reading']['power_mw']):.4f} | {smsr(row):.2f} |"
        )
    report += [
        "",
        "结论：旧表的主要变化是约 +2.4 pm 的整体长波漂移和功率整体上升；不能直接用一个全局偏置覆盖，因为存在少量错模/低 SMSR 点，必须先逐点修复并重复验证。",
        "",
    ]
    report_path = args.prefix.with_suffix(".md")
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({
        "csv": str(csv_path), "plot": str(plot_path), "report": str(report_path),
        "normal": len(normal), "mode_hops": len(mode_hops), "low_smsr": len(low_smsr),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
