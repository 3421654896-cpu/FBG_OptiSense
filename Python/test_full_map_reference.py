"""Reference-only acquisition must never be mistaken for a dynamic sample."""
import unittest
from pathlib import Path

from frame_telemetry import decode_board_frame_schedule
from stress_multirate_protocol import MultirateConfiguration, build_multirate_arm_command, select_current_timed_samples
from stress_multirate_reference import Scheduler, Profile
from test_stress_multirate import _extension


class FullMapReferenceTests(unittest.TestCase):
    def test_full_fresh_reference_ignores_quiet_and_active_states(self):
        scheduler = Scheduler(map_period_frames=1, single_track_enabled=True)
        for sequence in range(150):
            # Exercise forced activity, quiet grey band and stale ROI choices.
            scheduler.activity_latched = bool(sequence % 2)
            scheduler.baseline_quiet_frames = 0
            scheduler.active_pair = (3, 7)
            scheduler.single_track_active = True
            plan = scheduler.begin_frame()
            self.assertEqual(plan.sequence, sequence)
            self.assertEqual(plan.profile, Profile.MAP)
            self.assertEqual(plan.fresh_indices, tuple(range(45)))
            self.assertIsNone(plan.primary_segment)
            self.assertIsNone(plan.secondary_segment)
            self.assertTrue(plan.bandwidth_discontinuity)
            values = [100 + p + sequence for p in range(45)]
            scheduler.record_plan(values)
            scheduler.end_frame()
            self.assertEqual(scheduler.cache, values)

    def test_reference_gap_rejects_dynamic_reconstruction(self):
        extension = _extension(0, tuple(range(45)), bandwidth_discontinuity=True)
        schedule = decode_board_frame_schedule(extension)
        self.assertTrue(schedule.is_complete_map)
        self.assertTrue(schedule.has_point_timing)
        with self.assertRaisesRegex(ValueError, "bandwidth"):
            select_current_timed_samples([100]*45, extension=extension)

    def test_reference_is_opt_in_and_uses_unchanged_dac_table_protocol(self):
        normal = build_multirate_arm_command()
        reference = build_multirate_arm_command(MultirateConfiguration(map_period_frames=1))
        self.assertEqual([i for i, (a, b) in enumerate(zip(normal, reference)) if a != b], [5])
        self.assertEqual(normal[5], 0)
        self.assertEqual(reference[5], 1)
        scheduler = Scheduler()
        initial = scheduler.begin_frame()
        self.assertEqual(initial.profile, Profile.MAP)
        self.assertFalse(initial.bandwidth_discontinuity)
        scheduler.record_plan([100]*45)
        scheduler.end_frame()
        self.assertEqual(scheduler.begin_frame().profile, Profile.SURVEY)

    def test_firmware_reference_branch_precedes_cache_activity_and_quiet(self):
        root = Path(__file__).resolve().parents[1]/"JDSU/Core"
        source = (root/"Src/stress_multirate.c").read_text("utf-8")
        begin = source.split("const StressMultiratePlan_t *StressMultirate_BeginFrame(", 1)[1]
        branch = begin.split("else if(!state->enabled", 1)[0]
        self.assertIn("state->enabled &&", branch)
        self.assertIn("state->mapPeriodFrames == STRESS_MULTIRATE_REFERENCE_MAP_PERIOD", branch)
        self.assertIn("buildMapPlan(plan);", branch)
        self.assertIn("plan->bandwidthDiscontinuity = 1U;", branch)
        command = (root/"Src/ms5614t.c").read_text("utf-8")
        self.assertIn("mapPeriod == STRESS_MULTIRATE_REFERENCE_MAP_PERIOD ||", command)
        self.assertIn("mapPeriodFrames != STRESS_MULTIRATE_REFERENCE_MAP_PERIOD &&", source)


if __name__ == "__main__":
    unittest.main()
