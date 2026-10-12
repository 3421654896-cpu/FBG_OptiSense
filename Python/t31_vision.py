"""T31 marker/probe vision for CNC touch calibration.

The module deliberately uses the packages already bundled with the desktop
application (PyQt5, NumPy and SciPy).  It does not open or manipulate any
Windows UI.  A frame is supplied by ``t31_snapshot.exe`` and is analysed as a
normal image file.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtGui
from scipy import ndimage


@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    area: int
    bbox: tuple[int, int, int, int]


def load_rgb(path: Path) -> tuple[np.ndarray, QtGui.QImage]:
    image = QtGui.QImage(str(path))
    if image.isNull():
        raise ValueError(f"无法读取图像：{path}")
    image = image.convertToFormat(QtGui.QImage.Format_RGB888)
    width, height = image.width(), image.height()
    bits = image.bits()
    bits.setsize(height * image.bytesPerLine())
    rgb = np.frombuffer(bits, np.uint8).reshape(height, image.bytesPerLine())
    rgb = rgb[:, : width * 3].reshape(height, width, 3).copy()
    return rgb, image


def _components(mask: np.ndarray, minimum_area: int) -> list[Detection]:
    labels, count = ndimage.label(mask)
    detections: list[Detection] = []
    for label in range(1, count + 1):
        ys, xs = np.where(labels == label)
        if xs.size < minimum_area:
            continue
        detections.append(
            Detection(
                x=float(xs.mean()),
                y=float(ys.mean()),
                area=int(xs.size),
                bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
            )
        )
    return detections


def _merge_touching_vertical_fragments(items: list[Detection]) -> list[Detection]:
    """Merge one painted mark split by a thin horizontal glare band."""

    merged = list(items)
    changed = True
    while changed:
        changed = False
        for first_index, first in enumerate(merged):
            for second_index in range(first_index + 1, len(merged)):
                second = merged[second_index]
                horizontal_overlap = min(first.bbox[2], second.bbox[2]) - max(
                    first.bbox[0], second.bbox[0]
                )
                vertical_gap = max(
                    0,
                    max(first.bbox[1], second.bbox[1])
                    - min(first.bbox[3], second.bbox[3]),
                )
                # Strong glare can cut one painted mark into fragments with
                # an 8..10 pixel blank band.  Neighbouring grid rows are more
                # than 60 pixels apart, so a 12-pixel merge bound remains
                # conservative while preserving the nine logical markers.
                if horizontal_overlap < 1 or vertical_gap > 12:
                    continue
                area = first.area + second.area
                combined = Detection(
                    x=(first.x * first.area + second.x * second.area) / area,
                    y=(first.y * first.area + second.y * second.area) / area,
                    area=area,
                    bbox=(
                        min(first.bbox[0], second.bbox[0]),
                        min(first.bbox[1], second.bbox[1]),
                        max(first.bbox[2], second.bbox[2]),
                        max(first.bbox[3], second.bbox[3]),
                    ),
                )
                merged[first_index] = combined
                del merged[second_index]
                changed = True
                break
            if changed:
                break
    return merged


def detect_red_markers(rgb: np.ndarray) -> list[Detection]:
    height, width, _ = rgb.shape
    red = rgb[:, :, 0].astype(np.float32)
    green = rgb[:, :, 1].astype(np.float32)
    blue = rgb[:, :, 2].astype(np.float32)
    yy, xx = np.indices((height, width))

    # The broad spatial gate rejects the coloured CNC shaft while retaining
    # all nine marks even if the camera is moved slightly.
    mask = (
        (red > 120)
        & (red > green * 1.22 + 16)
        & (red > blue * 1.12 + 8)
        & (xx > int(width * 0.29))
        & (xx < int(width * 0.51))
        & (yy > int(height * 0.12))
        & (yy < int(height * 0.82))
    )
    mask = ndimage.binary_opening(mask, iterations=1)
    mask = ndimage.binary_closing(mask, iterations=3)
    markers = _components(mask, minimum_area=max(300, width * height // 15000))
    markers = _merge_touching_vertical_fragments(markers)
    if len(markers) != 9:
        raise RuntimeError(f"红点检测数量为 {len(markers)}，要求恰好为 9；禁止自动运动")

    # The photographed grid is oblique.  Sorting by image y, taking groups of
    # three and then sorting each row by x gives a stable top-left ->
    # bottom-right logical index without assuming equal spacing.
    by_y = sorted(markers, key=lambda item: item.y)
    ordered: list[Detection] = []
    for start in range(0, 9, 3):
        ordered.extend(sorted(by_y[start : start + 3], key=lambda item: item.x))
    return ordered


def detect_probe_tip(
    rgb: np.ndarray,
    markers: list[Detection],
    *,
    expected_x: float | None = None,
) -> Detection:
    height, width, _ = rgb.shape
    # First locate the coloured metal shaft.  Pool all saturated pixels in a
    # broad right-hand ROI: individual rainbow bands can split into several
    # connected components as the axes move, but their median centreline is
    # stable.
    sx0, sx1 = int(width * 0.50), int(width * 0.92)
    # Cover the complete right-hand column of requested targets.  The former
    # 0.35..0.61 band was suitable only for the centre target and correctly
    # failed once the probe was aligned to the upper marker.
    sy0, sy1 = int(height * 0.20), int(height * 0.80)
    shaft_crop = rgb[sy0:sy1, sx0:sx1].astype(np.int16)
    shaft_spread = shaft_crop.max(axis=2) - shaft_crop.min(axis=2)
    shaft_mask = (shaft_spread > 42) & (shaft_crop.max(axis=2) > 88)
    shaft_mask = ndimage.binary_opening(shaft_mask, iterations=1)
    ys, xs = np.where(shaft_mask)
    if xs.size < 350:
        raise RuntimeError("未检测到压头金属杆；禁止自动运动")
    shaft_y = float(np.median(ys) + sy0)

    # The hard leading edge is the strongest dark-to-light transition along
    # that centreline.  Unlike a fixed shaft-to-pad offset, this remains valid
    # when perspective changes during a multi-axis move.
    x0 = max(int(width * 0.46), int(max(item.x for item in markers) + width * 0.012))
    x1 = min(int(width * 0.72), int(np.percentile(xs, 40) + sx0))
    y0, y1 = int(round(shaft_y - 10)), int(round(shaft_y + 11))
    luminance = rgb[y0:y1, x0:x1].astype(np.float32).mean(axis=(0, 2))
    if luminance.size < 60:
        raise RuntimeError("白色压头前缘搜索窗口不足；禁止自动运动")
    smooth = np.convolve(luminance, np.ones(9, dtype=np.float32) / 9.0, mode="same")
    gradient = np.gradient(smooth)
    guard = 12
    inner = gradient[guard:-guard]
    candidate_local = np.flatnonzero(
        (inner[1:-1] >= inner[:-2])
        & (inner[1:-1] >= inner[2:])
        & (inner[1:-1] >= 3.0)
    ) + guard + 1
    if candidate_local.size == 0:
        raise RuntimeError("白色压头前缘没有可信候选；禁止自动运动")
    if expected_x is None:
        peak = int(candidate_local[np.argmax(gradient[candidate_local])])
    else:
        candidate_global = candidate_local + x0
        nearest = int(np.argmin(np.abs(candidate_global - float(expected_x))))
        if abs(float(candidate_global[nearest]) - float(expected_x)) > 65.0:
            raise RuntimeError("白色压头前缘偏离运动预测；禁止自动运动")
        peak = int(candidate_local[nearest])
    if float(gradient[peak]) < 3.0:
        raise RuntimeError("白色压头前缘对比度不足；禁止自动运动")
    tip_x = float(x0 + peak)
    tip_y = shaft_y
    return Detection(
        x=tip_x,
        y=tip_y,
        area=int(xs.size),
        bbox=(
            int(round(tip_x)),
            y0,
            int(xs.max() + sx0),
            y1,
        ),
    )


def detect_surface_x(rgb: np.ndarray, marker: Detection, probe: Detection) -> float:
    """Locate the finger's right-hand surface at the target marker height."""

    height, width, _ = rgb.shape
    y = int(round(marker.y))
    y0, y1 = max(0, y - 12), min(height, y + 13)
    x0 = max(0, int(round(marker.x + width * 0.015)))
    x1 = min(width, int(round(probe.x - width * 0.015)))
    if x1 - x0 < 30:
        raise RuntimeError("压头已进入表面安全邻域，边界窗口不足；停止自动运动")
    luminance = np.median(rgb[y0:y1, x0:x1].astype(np.float32).mean(axis=2), axis=0)
    smooth = np.convolve(luminance, np.ones(11, dtype=np.float32) / 11.0, mode="same")
    gradient = np.gradient(smooth)
    guard = 10
    local = gradient[guard:-guard]
    edge = int(np.argmin(local) + guard)
    if float(gradient[edge]) > -5.0:
        raise RuntimeError("手指表面边界置信度不足；停止自动运动")
    return float(x0 + edge)


def detect_close_contact(
    rgb: np.ndarray,
    marker: Detection,
    probe_y: float,
    *,
    expected_front_x: float | None = None,
) -> dict[str, float | bool | None]:
    """Measure the real soft-tip clearance without locking onto its rear edge.

    The white pad contains several strong internal transitions.  The original
    temporal detector could therefore follow a rear seam instead of the
    leading tip.  For contact work, first locate the finger's strong
    bright-to-dark surface edge, then search only the physically plausible
    20..110 pixel clearance band for the first dark-to-bright pad edge.
    When that distinct edge disappears, the soft pad has merged with the
    surface and is treated as contact rather than extrapolated through it.
    """

    height, width, _ = rgb.shape
    # Clearance must be measured on the probe centreline.  Averaging this
    # with the target marker height gives a row that crosses neither the tip
    # nor the intended contact surface while the axes are still aligning.
    y = int(round(probe_y))
    y0, y1 = max(0, y - 10), min(height, y + 11)
    row = rgb[y0:y1].astype(np.float32).mean(axis=(0, 2))
    smooth = np.convolve(row, np.ones(9, dtype=np.float32) / 9.0, mode="same")
    gradient = np.gradient(smooth)

    surface_lo = max(0, int(round(marker.x + width * 0.012)))
    surface_hi = min(width, int(round(marker.x + width * 0.055)))
    if surface_hi - surface_lo < 30:
        raise RuntimeError("手指表面搜索窗口不足；禁止自动运动")
    surface_local = gradient[surface_lo:surface_hi]
    surface_x = int(np.argmin(surface_local) + surface_lo)
    surface_strength = float(gradient[surface_x])
    if surface_strength > -5.0:
        raise RuntimeError("手指表面边界置信度不足；禁止自动运动")

    # During ordinary acquisition, ignore the first 20 pixels because the
    # surface edge itself produces a small positive rebound.  During a
    # controlled approach, a temporal prediction is available and the real
    # fuzzy tip is allowed to enter this near-surface band.
    tip_lo = min(width, surface_x + (5 if expected_front_x is not None else 20))
    tip_hi = min(width, surface_x + 111)
    local = gradient[tip_lo:tip_hi]
    candidates = np.flatnonzero(
        (local[1:-1] >= local[:-2])
        & (local[1:-1] >= local[2:])
        & (local[1:-1] >= (1.5 if expected_front_x is not None else 3.0))
    ) + tip_lo + 1
    if candidates.size == 0:
        return {
            "surface_x": float(surface_x),
            "surface_strength": surface_strength,
            "probe_front_x": None,
            "clearance_px": None,
            "contact": None,
        }

    if expected_front_x is None:
        probe_front_x = int(candidates[0])
    else:
        nearest = int(np.argmin(np.abs(candidates - float(expected_front_x))))
        probe_front_x = int(candidates[nearest])
        if abs(probe_front_x - float(expected_front_x)) > 30.0:
            return {
                "surface_x": float(surface_x),
                "surface_strength": surface_strength,
                "probe_front_x": None,
                "clearance_px": None,
                "contact": None,
            }
    clearance = float(probe_front_x - surface_x)
    return {
        "surface_x": float(surface_x),
        "surface_strength": surface_strength,
        "probe_front_x": float(probe_front_x),
        "clearance_px": clearance,
        "contact": clearance <= 12.0,
    }


def analyse(
    path: Path,
    *,
    expected_probe_x: float | None = None,
) -> tuple[list[Detection], Detection, QtGui.QImage]:
    rgb, image = load_rgb(path)
    markers = detect_red_markers(rgb)
    probe = detect_probe_tip(rgb, markers, expected_x=expected_probe_x)
    return markers, probe, image


def annotate(
    image: QtGui.QImage,
    markers: list[Detection],
    probe: Detection,
    output: Path,
) -> None:
    canvas = image.copy()
    painter = QtGui.QPainter(canvas)
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
    font = QtGui.QFont("Microsoft YaHei UI", 18, QtGui.QFont.Bold)
    painter.setFont(font)
    for index, marker in enumerate(markers, 1):
        painter.setPen(QtGui.QPen(QtGui.QColor(35, 255, 130), 5))
        painter.drawEllipse(QtCore.QPointF(marker.x, marker.y), 28, 28)
        painter.drawText(int(marker.x + 34), int(marker.y + 7), str(index))
    painter.setPen(QtGui.QPen(QtGui.QColor(0, 210, 255), 6))
    painter.drawLine(int(probe.x - 30), int(probe.y), int(probe.x + 30), int(probe.y))
    painter.drawLine(int(probe.x), int(probe.y - 30), int(probe.x), int(probe.y + 30))
    painter.drawText(int(probe.x + 38), int(probe.y - 12), "TIP")
    painter.end()
    if not canvas.save(str(output), "PNG"):
        raise RuntimeError(f"无法保存标注图像：{output}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="T31 九红点与白色压头检测")
    parser.add_argument("image", type=Path)
    parser.add_argument("--annotated", type=Path)
    parser.add_argument("--target-index", type=int, default=6, choices=range(1, 10))
    parser.add_argument("--expected-probe-x", type=float)
    parser.add_argument("--expected-front-x", type=float)
    args = parser.parse_args(argv)

    markers, probe, image = analyse(
        args.image,
        expected_probe_x=args.expected_probe_x,
    )
    target = markers[args.target_index - 1]
    rgb, _ = load_rgb(args.image)
    contact = detect_close_contact(
        rgb,
        target,
        probe.y,
        expected_front_x=args.expected_front_x,
    )
    if args.annotated:
        annotate(image, markers, probe, args.annotated)
    print(
        json.dumps(
            {
                "image": str(args.image.resolve()),
                "markers": [asdict(item) for item in markers],
                "probe": asdict(probe),
                "target_index": args.target_index,
                "surface_x": contact["surface_x"],
                "surface_gap_px": contact["clearance_px"],
                "probe_front_x": contact["probe_front_x"],
                "contact": contact["contact"],
                "surface_edge_strength": contact["surface_strength"],
                "target_vertical_error_px": probe.y - target.y,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
