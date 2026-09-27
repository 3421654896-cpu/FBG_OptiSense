"""Tests for manual touch annotation and 45-point sequence recording."""

import unittest

from manual_touch_dataset import (
    ManualTouchLabel,
    decode_native_ch1_stress_frame,
    trial_frame_quality,
)


def native_45_frame(ch1_codes):
    if len(ch1_codes) != 45:
        raise ValueError
    frame = bytearray(b"\xEE\xEE" + (45).to_bytes(2, "big"))
    for point, ch1 in enumerate(ch1_codes):
        channels = (point, ch1, 1000 + point, 2000 + point)
        for value in channels:
            frame.extend(int(value).to_bytes(2, "big"))
    return bytes(frame)


class NativeFrameDecodeTests(unittest.TestCase):
    def test_decode_exactly_45_ch1_codes(self):
        expected = [100 + point * 3 for point in range(45)]
        decoded = decode_native_ch1_stress_frame(native_45_frame(expected))

        self.assertEqual(decoded["point_count"], 45)
        self.assertEqual(decoded["ch1_adc_codes"], expected)
        self.assertEqual(decoded["all_channels"][0], [0, 100, 1000, 2000])
        self.assertEqual(decoded["all_channels"][-1], [44, 232, 1044, 2044])

    def test_reject_non_45_point_frame(self):
        payload = b"\xEE\xEE" + (9).to_bytes(2, "big") + bytes(9 * 8)
        with self.assertRaisesRegex(ValueError, "45"):
            decode_native_ch1_stress_frame(payload)


class ManualLabelTests(unittest.TestCase):
    def test_location_and_user_stress_label_are_kept_on_every_frame(self):
        label = ManualTouchLabel(
            location_id="tip_left",
            location_name="指尖左侧",
            model_x=-0.25,
            model_y=0.58,
            stress_label="中等按压",
        )

        quality = trial_frame_quality(
            label,
            trial_id="trial-001",
            phase="pressed",
            frame_index=12,
        )

        self.assertEqual(quality["manual_location_id"], "tip_left")
        self.assertEqual(quality["manual_location_name"], "指尖左侧")
        self.assertEqual(quality["manual_model_xy"], [-0.25, 0.58])
        self.assertEqual(quality["manual_stress_label"], "中等按压")
        self.assertEqual(quality["manual_phase"], "pressed")
        self.assertEqual(quality["trial_id"], "trial-001")
        self.assertEqual(quality["trial_frame_index"], 12)

    def test_label_rejects_position_outside_normalized_model(self):
        with self.assertRaises(ValueError):
            ManualTouchLabel(
                location_id="bad",
                location_name="outside",
                model_x=1.5,
                model_y=0.0,
                stress_label="",
            )


if __name__ == "__main__":
    unittest.main()
