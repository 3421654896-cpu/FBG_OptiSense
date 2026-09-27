from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent.parent


class OpenGLLauncherTests(unittest.TestCase):
    def test_dev_launcher_does_not_force_incompatible_software_opengl(self):
        text = (ROOT / 'start_dev.ps1').read_text(encoding='utf-8-sig')
        self.assertNotIn('$env:QT_OPENGL = "software"', text)
        self.assertIn('$env:QT_OPENGL = "desktop"', text)

    def test_demo_launcher_does_not_force_incompatible_software_opengl(self):
        text = (ROOT / 'start_demo.ps1').read_text(encoding='utf-8-sig')
        self.assertNotIn('$env:QT_OPENGL = "software"', text)
        self.assertIn('$env:QT_OPENGL = "desktop"', text)


if __name__ == '__main__':
    unittest.main()
