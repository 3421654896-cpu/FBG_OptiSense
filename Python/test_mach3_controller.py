import unittest
from unittest import mock

from mach3_controller import Mach3Controller, validate_micro_move


class MicroMoveSafetyTests(unittest.TestCase):
    def test_only_xyz_are_allowed(self):
        with self.assertRaises(ValueError):
            validate_micro_move("A", 0.1, 20, safe_zone_confirmed=True)

    def test_step_cannot_exceed_one_mm(self):
        with self.assertRaises(ValueError):
            validate_micro_move("X", 1.0001, 20, safe_zone_confirmed=True)

    def test_feed_is_limited(self):
        with self.assertRaises(ValueError):
            validate_micro_move("Y", -0.1, 31, safe_zone_confirmed=True)

    def test_safe_zone_confirmation_is_mandatory(self):
        with self.assertRaises(ValueError):
            validate_micro_move("Z", 0.05, 10, safe_zone_confirmed=False)

    def test_valid_negative_microstep_is_normalized(self):
        self.assertEqual(
            validate_micro_move(" x ", -0.1, 20, safe_zone_confirmed=True),
            ("X", -0.1, 20.0),
        )


class ControllerProtocolTests(unittest.TestCase):
    def test_emergency_stop_has_no_reset_or_confirmation_argument(self):
        controller = Mach3Controller()
        controller._request = mock.Mock(return_value={"estop": True, "moving": False})
        result = controller.emergency_stop()
        self.assertTrue(result["estop"])
        controller._request.assert_called_once_with({"op": "estop"})

    def test_release_emergency_stop_requires_operator_confirmation(self):
        controller = Mach3Controller()
        controller._request = mock.Mock()
        with self.assertRaises(ValueError):
            controller.release_emergency_stop(operator_confirmed=False)
        controller._request.assert_not_called()

    def test_release_emergency_stop_sends_explicit_confirmation(self):
        controller = Mach3Controller()
        controller._request = mock.Mock(
            return_value={"spindle_command": "M5", "after": {"estop": False}}
        )
        result = controller.release_emergency_stop(operator_confirmed=True)
        self.assertEqual(result["spindle_command"], "M5")
        controller._request.assert_called_once_with(
            {"op": "release_estop", "operator_confirmed": True}
        )

    def test_move_sends_only_validated_fields(self):
        controller = Mach3Controller()
        controller._request = mock.Mock(return_value={"command": "G21 G90 G1 X0.1"})
        controller.move_relative_microstep(
            "X", 0.1, feed_mm_min=20, safe_zone_confirmed=True
        )
        controller._request.assert_called_once_with(
            {
                "op": "move",
                "axis": "X",
                "delta_mm": 0.1,
                "feed_mm_min": 20.0,
                "safe_zone_confirmed": True,
            }
        )

    def test_ten_mm_move_is_split_into_ten_verified_steps(self):
        controller = Mach3Controller()
        positions = iter(float(index) for index in range(11))

        def step_result(_payload):
            before_x = next(positions)
            return {
                "before": {"work_x": before_x, "work_y": 2.0, "work_z": 3.0},
                "after": {"work_x": before_x + 1.0, "work_y": 2.0, "work_z": 3.0},
            }

        controller._request = mock.Mock(side_effect=step_result)
        results = controller.move_relative_staged(
            "X", 10.0, step_mm=1.0, feed_mm_min=20, safe_zone_confirmed=True
        )
        self.assertEqual(len(results), 10)
        self.assertEqual(controller._request.call_count, 10)

    def test_confirmed_continuous_x_move_is_one_bridge_command(self):
        controller = Mach3Controller()
        controller._request = mock.Mock(return_value={"command": "G21 G90 G1 X-95"})
        controller.move_relative_continuous_x(
            -10.0, feed_mm_min=60, safe_zone_confirmed=True
        )
        controller._request.assert_called_once_with(
            {
                "op": "long_axis_move",
                "axis": "X",
                "delta_mm": -10.0,
                "feed_mm_min": 60.0,
                "safe_zone_confirmed": True,
            }
        )

    def test_continuous_x_move_rejects_unconfirmed_or_oversized_path(self):
        controller = Mach3Controller()
        with self.assertRaises(ValueError):
            controller.move_relative_continuous_x(-10.0)
        with self.assertRaises(ValueError):
            controller.move_relative_continuous_x(
                -10.1, safe_zone_confirmed=True
            )

    def test_continuous_y_and_z_are_supported(self):
        controller = Mach3Controller()
        controller._request = mock.Mock(return_value={"command": "ok"})
        controller.move_relative_continuous_axis(
            "Y", 10, safe_zone_confirmed=True
        )
        controller.move_relative_continuous_axis(
            "Z", 10, safe_zone_confirmed=True
        )
        self.assertEqual(controller._request.call_args_list[0].args[0]["axis"], "Y")
        self.assertEqual(controller._request.call_args_list[1].args[0]["axis"], "Z")

    def test_sub_millimetre_continuous_press_is_one_command(self):
        controller = Mach3Controller()
        controller._request = mock.Mock(return_value={"command": "ok"})
        controller.move_relative_continuous_axis(
            "Z", -0.2, feed_mm_min=15, safe_zone_confirmed=True
        )
        controller._request.assert_called_once_with(
            {
                "op": "long_axis_move",
                "axis": "Z",
                "delta_mm": -0.2,
                "feed_mm_min": 15.0,
                "safe_zone_confirmed": True,
            }
        )

    def test_absolute_xy_move_is_one_coordinated_command(self):
        controller = Mach3Controller()
        controller._request = mock.Mock(return_value={"command": "ok"})
        controller.move_absolute_xy_continuous(
            -4.5, 12.25, feed_mm_min=80, safe_zone_confirmed=True
        )
        controller._request.assert_called_once_with(
            {
                "op": "xy_move",
                "target_x": -4.5,
                "target_y": 12.25,
                "feed_mm_min": 80.0,
                "max_axis_delta_mm": 30.0,
                "safe_zone_confirmed": True,
            }
        )

    def test_absolute_xy_move_requires_confirmation_and_speed_limit(self):
        controller = Mach3Controller()
        with self.assertRaises(ValueError):
            controller.move_absolute_xy_continuous(1.0, 2.0)
        with self.assertRaises(ValueError):
            controller.move_absolute_xy_continuous(
                1.0, 2.0, feed_mm_min=120.1, safe_zone_confirmed=True
            )


if __name__ == "__main__":
    unittest.main()
