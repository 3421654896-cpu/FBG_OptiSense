import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent


class QtStartupEnvironmentTests(unittest.TestCase):
    def test_stale_platform_plugin_path_is_replaced_before_qapplication(self):
        script = textwrap.dedent(
            """
            import importlib
            import os
            import sys
            from pathlib import Path

            importlib.import_module(sys.argv[1])
            expected = (
                Path(sys.prefix) / "Lib" / "site-packages" / "PyQt5" /
                "Qt5" / "plugins" / "platforms"
            ).resolve()
            actual = Path(os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"]).resolve()
            if actual != expected:
                raise RuntimeError(f"plugin path was not repaired: {actual} != {expected}")
            from PyQt5 import QtWidgets
            app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
            print("qt-startup-ok")
            """
        )
        for module_name in (
            "app_JDSU",
            "fbg_unified_app",
            "fullband_calibration_progress",
        ):
            with self.subTest(module=module_name):
                environment = os.environ.copy()
                environment["QT_QPA_PLATFORM"] = "offscreen"
                environment["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(
                    APP_DIR / "definitely-not-a-qt-plugin-directory"
                )
                result = subprocess.run(
                    [sys.executable, "-c", script, module_name],
                    cwd=APP_DIR,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                self.assertEqual(
                    result.returncode,
                    0,
                    f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
                )
                self.assertIn("qt-startup-ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
