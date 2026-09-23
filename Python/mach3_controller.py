"""Safety-constrained, command-based control for the Mach3 XYZ stage.

Mach3 is a 32-bit application, while the FBG desktop program normally runs in
64-bit Python.  A persistent 32-bit PowerShell child hosts Mach3's documented
COM automation object and exchanges line-delimited JSON with this module.
No Windows UI automation is used for motion.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

APP_DIR = Path(__file__).resolve().parent
DEFAULT_BRIDGE = APP_DIR / "mach3_bridge.ps1"
POWERSHELL_X86 = (
    Path(os.environ.get("WINDIR", r"C:\Windows"))
    / "SysWOW64"
    / "WindowsPowerShell"
    / "v1.0"
    / "powershell.exe"
)
MAX_COMMISSIONING_STEP_MM = 1.0
MAX_COMMISSIONING_FEED_MM_MIN = 30.0
MAX_CONFIRMED_TOTAL_MM = 10.0
MAX_SMOOTH_XY_DELTA_MM = 30.0
MAX_SMOOTH_XY_FEED_MM_MIN = 120.0


class Mach3CommandError(RuntimeError):
    """Mach3 rejected a request or could not confirm its result."""


def validate_micro_move(
    axis: str,
    delta_mm: float,
    feed_mm_min: float,
    *,
    safe_zone_confirmed: bool,
) -> tuple[str, float, float]:
    axis = str(axis).strip().upper()
    if axis not in {"X", "Y", "Z"}:
        raise ValueError("只允许控制 X、Y、Z 轴")
    delta = float(delta_mm)
    feed = float(feed_mm_min)
    if not math.isfinite(delta) or not 0 < abs(delta) <= MAX_COMMISSIONING_STEP_MM:
        raise ValueError("分段移动的单步必须大于 0 且不超过 1 mm")
    if not math.isfinite(feed) or not 0 < feed <= MAX_COMMISSIONING_FEED_MM_MIN:
        raise ValueError("首次调试速度必须大于 0 且不超过 30 mm/min")
    if not safe_zone_confirmed:
        raise ValueError("尚未人工确认运动方向和 0.1 mm 行程内无遮挡")
    return axis, delta, feed


class Mach3Controller:
    """Persistent direct-command channel with a deliberately tiny API."""

    def __init__(
        self,
        *,
        bridge_path: Path = DEFAULT_BRIDGE,
        process_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
    ):
        self.bridge_path = Path(bridge_path)
        self._process_factory = process_factory
        self._process: subprocess.Popen | None = None
        self._lock = threading.RLock()

    def _start(self) -> subprocess.Popen:
        if self._process is not None and self._process.poll() is None:
            return self._process
        if not POWERSHELL_X86.is_file():
            raise Mach3CommandError(f"未找到 32 位 PowerShell：{POWERSHELL_X86}")
        if not self.bridge_path.is_file():
            raise Mach3CommandError(f"未找到 Mach3 指令桥：{self.bridge_path}")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._process = self._process_factory(
            [
                str(POWERSHELL_X86),
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(self.bridge_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # Do not use utf-8-sig here: the same TextIO wrapper is used for
            # stdin, and a BOM before the first JSON object breaks
            # ConvertFrom-Json in Windows PowerShell 5.1.
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )
        return self._process

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            process = self._start()
            assert process.stdin is not None and process.stdout is not None
            process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            process.stdin.flush()
            line = process.stdout.readline()
            if not line:
                details = ""
                if process.stderr is not None:
                    details = process.stderr.read().strip()
                self._process = None
                raise Mach3CommandError(details or "Mach3 指令桥意外退出")
            try:
                response = json.loads(line)
            except json.JSONDecodeError as exc:
                raise Mach3CommandError(f"Mach3 返回了无效响应：{line.strip()}") from exc
            if not response.get("ok"):
                raise Mach3CommandError(str(response.get("error") or "Mach3 指令失败"))
            return dict(response.get("result") or {})

    def status(self) -> dict[str, Any]:
        return self._request({"op": "status"})

    def position_sample(self) -> dict[str, Any]:
        """Read-only DRO timing sample; not a safety snapshot or encoder proof."""
        return self._request({"op": "position_sample"})

    def emergency_stop(self) -> dict[str, Any]:
        """Stop first, then latch E-stop only if it is not already latched."""

        return self._request({"op": "estop"})

    def release_emergency_stop(
        self, *, operator_confirmed: bool = False
    ) -> dict[str, Any]:
        """Release Reset/E-stop only after explicit operator confirmation."""

        if not operator_confirmed:
            raise ValueError("尚未确认机床周围安全，不能解除急停")
        return self._request(
            {"op": "release_estop", "operator_confirmed": True}
        )

    def move_relative_microstep(
        self,
        axis: str,
        delta_mm: float,
        *,
        feed_mm_min: float = 20.0,
        safe_zone_confirmed: bool = False,
    ) -> dict[str, Any]:
        axis, delta, feed = validate_micro_move(
            axis,
            delta_mm,
            feed_mm_min,
            safe_zone_confirmed=safe_zone_confirmed,
        )
        return self._request(
            {
                "op": "move",
                "axis": axis,
                "delta_mm": delta,
                "feed_mm_min": feed,
                "safe_zone_confirmed": True,
            }
        )

    def move_relative_staged(
        self,
        axis: str,
        total_mm: float,
        *,
        step_mm: float = 1.0,
        feed_mm_min: float = 20.0,
        safe_zone_confirmed: bool = False,
    ) -> list[dict[str, Any]]:
        """Move up to 10 mm as independently verified one-millimetre steps."""

        axis = str(axis).strip().upper()
        total = float(total_mm)
        step = float(step_mm)
        if axis not in {"X", "Y", "Z"}:
            raise ValueError("只允许控制 X、Y、Z 轴")
        if not math.isfinite(total) or not 0 < abs(total) <= MAX_CONFIRMED_TOTAL_MM:
            raise ValueError("单次确认的总行程必须大于 0 且不超过 10 mm")
        if not math.isfinite(step) or not 0 < step <= MAX_COMMISSIONING_STEP_MM:
            raise ValueError("分段步长必须大于 0 且不超过 1 mm")
        if not safe_zone_confirmed:
            raise ValueError("尚未人工确认整个行程无遮挡且手已放在物理急停上")

        direction = 1.0 if total > 0 else -1.0
        remaining = abs(total)
        results = []
        while remaining > 1e-9:
            delta = direction * min(step, remaining)
            try:
                result = self.move_relative_microstep(
                    axis,
                    delta,
                    feed_mm_min=feed_mm_min,
                    safe_zone_confirmed=True,
                )
                before = result.get("before", {})
                after = result.get("after", {})
                for other_axis in {"X", "Y", "Z"} - {axis}:
                    key = f"work_{other_axis.lower()}"
                    if abs(float(after[key]) - float(before[key])) > 0.002:
                        raise Mach3CommandError(
                            f"{other_axis} 轴在 {axis} 移动时发生异常位移"
                        )
                results.append(result)
                remaining -= abs(delta)
            except Exception:
                try:
                    self.emergency_stop()
                finally:
                    raise
        return results

    def move_relative_continuous_axis(
        self,
        axis: str,
        delta_mm: float,
        *,
        feed_mm_min: float = 60.0,
        safe_zone_confirmed: bool = False,
    ) -> dict[str, Any]:
        """Send one continuous axis command after the full path is confirmed."""

        axis = str(axis).strip().upper()
        delta = float(delta_mm)
        feed = float(feed_mm_min)
        if axis not in {"X", "Y", "Z"}:
            raise ValueError("只允许控制 X、Y、Z 轴")
        if not math.isfinite(delta) or not 0.003 < abs(delta) <= 10:
            raise ValueError("连续行程必须大于 0.003 mm 且不超过 10 mm")
        if not math.isfinite(feed) or not 0 < feed <= 60:
            raise ValueError("连续速度必须大于 0 且不超过 60 mm/min")
        if not safe_zone_confirmed:
            raise ValueError("尚未人工确认完整行程和限位余量")
        return self._request(
            {
                "op": "long_axis_move",
                "axis": axis,
                "delta_mm": delta,
                "feed_mm_min": feed,
                "safe_zone_confirmed": True,
            }
        )

    def move_absolute_xy_continuous(
        self,
        target_x: float,
        target_y: float,
        *,
        feed_mm_min: float = 60.0,
        safe_zone_confirmed: bool = False,
    ) -> dict[str, Any]:
        """Move X and Y together with one guarded absolute G1 command."""

        target_x = float(target_x)
        target_y = float(target_y)
        feed = float(feed_mm_min)
        if not all(math.isfinite(value) for value in (target_x, target_y, feed)):
            raise ValueError("XY目标和速度必须为有限数值")
        if not 0 < feed <= MAX_SMOOTH_XY_FEED_MM_MIN:
            raise ValueError("XY连续速度必须大于0且不超过120 mm/min")
        if not safe_zone_confirmed:
            raise ValueError("尚未确认XY完整路径位于安全高度且限位余量充足")
        return self._request(
            {
                "op": "xy_move",
                "target_x": target_x,
                "target_y": target_y,
                "feed_mm_min": feed,
                "max_axis_delta_mm": MAX_SMOOTH_XY_DELTA_MM,
                "safe_zone_confirmed": True,
            }
        )

    def move_relative_continuous_x(
        self,
        delta_mm: float,
        *,
        feed_mm_min: float = 60.0,
        safe_zone_confirmed: bool = False,
    ) -> dict[str, Any]:
        """Backward-compatible X-only alias."""

        return self.move_relative_continuous_axis(
            "X",
            delta_mm,
            feed_mm_min=feed_mm_min,
            safe_zone_confirmed=safe_zone_confirmed,
        )

    def close(self) -> None:
        with self._lock:
            process = self._process
            self._process = None
            if process is None:
                return
            if process.poll() is None:
                try:
                    assert process.stdin is not None
                    process.stdin.write('{"op":"quit"}\n')
                    process.stdin.flush()
                    process.wait(timeout=1.0)
                except Exception:
                    process.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Mach3 XYZ 安全指令控制")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--status", action="store_true", help="只读查询状态")
    group.add_argument("--estop", action="store_true", help="停止并锁定急停")
    group.add_argument(
        "--release-estop", action="store_true", help="人工确认安全后解除急停"
    )
    group.add_argument("--move", choices=("X", "Y", "Z"), help="首次 0.1 mm 微动轴")
    parser.add_argument("--delta-mm", type=float, default=0.1)
    parser.add_argument(
        "--step-mm", type=float, default=1.0, help="总行程大于1 mm时的分段步长"
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="仅对X轴发送一条连续长行程命令，不拆分",
    )
    parser.add_argument("--feed-mm-min", type=float, default=20.0)
    parser.add_argument(
        "--confirm-safe-zone",
        action="store_true",
        help="确认该方向 0.1 mm 内无遮挡且手已放在物理急停上",
    )
    args = parser.parse_args(argv)
    try:
        with Mach3Controller() as controller:
            if args.status:
                result = controller.status()
            elif args.estop:
                result = controller.emergency_stop()
            elif args.release_estop:
                result = controller.release_emergency_stop(
                    operator_confirmed=args.confirm_safe_zone
                )
            else:
                if args.continuous:
                    result = controller.move_relative_continuous_axis(
                        args.move,
                        args.delta_mm,
                        feed_mm_min=args.feed_mm_min,
                        safe_zone_confirmed=args.confirm_safe_zone,
                    )
                elif abs(args.delta_mm) <= MAX_COMMISSIONING_STEP_MM:
                    result = controller.move_relative_microstep(
                        args.move,
                        args.delta_mm,
                        feed_mm_min=args.feed_mm_min,
                        safe_zone_confirmed=args.confirm_safe_zone,
                    )
                else:
                    result = controller.move_relative_staged(
                        args.move,
                        args.delta_mm,
                        step_mm=args.step_mm,
                        feed_mm_min=args.feed_mm_min,
                        safe_zone_confirmed=args.confirm_safe_zone,
                    )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (Mach3CommandError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
