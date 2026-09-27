"""Force-display separation and rendering tests."""

import os
import sys
import unittest

sys.path.insert(0, ".")
qt_platform_plugins = (
    sys.prefix + r"\Lib\site-packages\PyQt5\Qt5\plugins\platforms"
)
if os.path.isdir(qt_platform_plugins):
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = qt_platform_plugins
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_OPENGL", "desktop")

from PyQt5 import QtWidgets  # noqa: E402

from manual_stress_capture import FingerSurfaceMap, ForceVectorWidget  # noqa: E402


class FingerModelSeparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_finger_surface_contains_no_force_visualization(self):
        surface = FingerSurfaceMap()
        try:
            surface.show()
            self.app.processEvents()
            self.assertFalse(hasattr(surface, "force_arrows"))
            self.assertFalse(hasattr(surface, "update_force_arrows"))
        finally:
            surface.close()


class ForceVectorPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_panel_draws_connected_and_disconnected(self):
        widget = ForceVectorWidget()
        widget.resize(320, 430)
        widget.set_force(6.0, -3.5, 18.0, connected=True)
        widget.show()
        self.app.processEvents()
        self.assertGreater(widget.grab().size().width(), 300)
        widget.set_force(None, None, None, connected=False)
        self.app.processEvents()
        self.assertGreater(widget.grab().size().height(), 320)
        widget.close()

    def test_panel_uses_colored_vectors_and_has_no_black_frame(self):
        widget = ForceVectorWidget()
        widget.resize(320, 430)
        widget.set_force(60.0, -60.0, 90.0, connected=True)
        widget.show()
        self.app.processEvents()
        image = widget.grab().toImage()
        pixels = [
            image.pixelColor(x, y)
            for y in range(image.height())
            for x in range(image.width())
        ]
        for red, green, blue in widget.AXIS_COLORS.values():
            self.assertTrue(
                any(
                    abs(color.red() - red) < 12
                    and abs(color.green() - green) < 12
                    and abs(color.blue() - blue) < 12
                    for color in pixels
                )
            )
        black_outer_pixels = 0
        for x in range(image.width()):
            for y in (2, 3, image.height() - 3, image.height() - 4):
                color = image.pixelColor(x, y)
                if color.red() < 20 and color.green() < 20 and color.blue() < 20:
                    black_outer_pixels += 1
        self.assertEqual(black_outer_pixels, 0)
        widget.close()


if __name__ == "__main__":
    unittest.main()
