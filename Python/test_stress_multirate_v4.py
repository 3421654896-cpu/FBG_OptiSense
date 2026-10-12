"""Deterministic policy tests; these do not claim hardware bandwidth validation."""
import unittest

from stress_multirate_reference import (
    ADC_MAX_CODE, STARTUP_STABLE_FRAMES, BaselineStatus, Profile, Scheduler,
    sentinel_index, track_single_indices, triplet_indices,
)


class AdaptiveV4Tests(unittest.TestCase):
    def setUp(self):
        self.baseline = [100, 300, 600, 300, 100] * 9

    def frame(self, scheduler, values, omit=None):
        plan = scheduler.begin_frame()
        for point in plan.fresh_indices:
            if point != omit:
                scheduler.record(point, values[point])
        scheduler.end_frame()
        return plan

    def ready(self, *, v4=True):
        scheduler = Scheduler(map_period_frames=0, single_track_enabled=v4)
        self.frame(scheduler, self.baseline)
        for _ in range(STARTUP_STABLE_FRAMES):
            self.frame(scheduler, self.baseline)
        self.assertEqual(scheduler.baseline_status, BaselineStatus.READY)
        return scheduler

    def contact(self, *segments):
        values = self.baseline.copy()
        for segment in segments:
            values[sentinel_index(segment)] += 100
        return values

    def qualify_single(self, scheduler, segment=4):
        values = self.contact(segment)
        self.assertEqual(self.frame(scheduler, values).profile, Profile.SURVEY)
        for _ in range(8):
            self.assertEqual(self.frame(scheduler, values).profile, Profile.TRACK_WIDE)
        self.assertTrue(scheduler.single_track_active)
        return values

    def test_first_trigger_is_wide_then_eight_confirmations_enable_single(self):
        scheduler = self.ready()
        values = self.qualify_single(scheduler)
        plan = self.frame(scheduler, values)
        self.assertEqual(plan.profile, Profile.TRACK_SINGLE)
        self.assertEqual(len(plan.fresh_indices), 11)
        self.assertEqual(plan.primary_segment, 4)
        self.assertIsNone(plan.secondary_segment)
        self.assertEqual(plan.fresh_indices, track_single_indices(4))
        self.assertTrue(all(sentinel_index(g) in plan.fresh_indices for g in range(9)))

    def test_second_contact_expands_on_the_very_next_frame(self):
        scheduler = self.ready()
        self.qualify_single(scheduler)
        values = self.contact(4, 7)
        self.assertEqual(self.frame(scheduler, values).profile, Profile.TRACK_SINGLE)
        plan = self.frame(scheduler, values)
        self.assertEqual(plan.profile, Profile.TRACK_WIDE)
        self.assertEqual(set((plan.primary_segment, plan.secondary_segment)), {4, 7})
        self.assertEqual(len(plan.fresh_indices), 13)

    def test_flat_or_saturated_roi_cannot_qualify_single(self):
        for code in (400, ADC_MAX_CODE):
            with self.subTest(code=code):
                scheduler = self.ready()
                values = self.contact(4)
                for index in triplet_indices(4):
                    values[index] = code
                for _ in range(20):
                    self.assertNotEqual(self.frame(scheduler, values).profile, Profile.TRACK_SINGLE)

    def test_missing_or_invalid_sentinel_cancels_single_without_relearning(self):
        for invalid in (False, True):
            with self.subTest(invalid=invalid):
                scheduler = self.ready()
                values = self.qualify_single(scheduler)
                prior = scheduler.baseline.copy()
                missing = sentinel_index(0)
                if invalid:
                    values[missing] = ADC_MAX_CODE + 1
                self.frame(scheduler, values, omit=None if invalid else missing)
                self.assertTrue(scheduler.activity_latched)
                self.assertFalse(scheduler.single_track_active)
                self.assertEqual(scheduler.baseline, prior)
                self.assertEqual(scheduler.begin_frame().profile, Profile.TRACK_WIDE)

    def test_non_sentinel_roi_outlier_does_not_bias_spatial_ranking(self):
        scheduler = self.ready()
        values = self.contact(4)
        self.frame(scheduler, values)
        values[triplet_indices(0)[1]] = 4000
        self.frame(scheduler, values)
        self.assertEqual(scheduler.active_pair[0], 4)
        self.assertEqual(scheduler.last_scores[0], 0)

    def test_ranking_compensates_certified_sentinel_sensitivity_only_in_v4(self):
        values = self.baseline.copy()
        values[sentinel_index(1)] += 100  # 100*79
        values[sentinel_index(3)] += 60   # 60*205: larger equivalent shift
        for v4, expected in ((True, 3), (False, 1)):
            scheduler = self.ready(v4=v4)
            self.frame(scheduler, values)
            self.assertEqual(scheduler.active_pair[0], expected)

    def test_alternating_contact_and_quiet_frames_do_not_release_or_go_single(self):
        scheduler = self.ready()
        contact = self.contact(4)
        self.frame(scheduler, contact)
        # Alternating samples approximate a fastest-resolvable oscillation,
        # not a physical 15 Hz actuator test.
        for frame in range(40):
            plan = self.frame(scheduler, self.baseline if frame % 2 == 0 else contact)
            self.assertEqual(plan.profile, Profile.TRACK_WIDE)
            self.assertTrue(scheduler.activity_latched)

    def test_v3_capability_never_emits_profile_three(self):
        scheduler = self.ready(v4=False)
        for _ in range(30):
            self.assertNotEqual(self.frame(scheduler, self.contact(4)).profile, Profile.TRACK_SINGLE)

    def test_four_quiet_frames_release_single(self):
        scheduler = self.ready()
        self.qualify_single(scheduler)
        for _ in range(4):
            self.frame(scheduler, self.baseline)
        self.assertFalse(scheduler.activity_latched)
        self.assertEqual(scheduler.begin_frame().profile, Profile.SURVEY)


if __name__ == "__main__":
    unittest.main()
