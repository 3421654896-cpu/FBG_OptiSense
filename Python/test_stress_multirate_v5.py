"""Opt-in G8-right optical-path experiment; no pressure accuracy claims."""
import json
from pathlib import Path
import unittest

from frame_telemetry import (
    MULTIRATE_EXTENSION_VERSION, sentinel_offsets_for_version,
    decode_board_frame_schedule, decode_board_frame_start_ms,
)
from stress_multirate_protocol import (
    MultirateConfiguration, build_multirate_arm_command, decode_multirate_ack,
    validate_current_arm_ack,
)
from stress_multirate_reference import Scheduler, Profile, track_indices, track_single_indices
from test_frame_telemetry import _v4_schedule_extension
from test_capture_sparse_held_trajectory import fixture as held_fixture
from test_capture_preserved_ch1_raw import fixture as raw_fixture
from capture_sparse_held_trajectory import decode as decode_held
from capture_preserved_ch1_raw import decode as decode_raw
from benchmark_stress_realtime import (
    _require_initial_current_map, _summarize_schedule_freshness,
    summarize_sentinel_signal_health,
)
from frame_telemetry import decode_board_frame_timing


class V5ContractTests(unittest.TestCase):
    def test_defaults_are_unchanged_and_v5_requires_exact_ack(self):
        self.assertEqual(MULTIRATE_EXTENSION_VERSION,4)
        self.assertEqual(build_multirate_arm_command()[10],4)
        cmd = build_multirate_arm_command(schedule_version=5)
        self.assertEqual(cmd[10],5)
        ack = bytearray(20); ack[:4] = b'\xff\xff\x02\x05'
        ack[4:10] = cmd[4:10]; ack[10:12] = bytes((0x21,5))
        decoded = decode_multirate_ack(bytes(ack))
        validate_current_arm_ack(decoded,requested_version=5)
        with self.assertRaises(ValueError): validate_current_arm_ack(decoded)
        ack[11] = 4
        with self.assertRaises(ValueError):
            validate_current_arm_ack(decode_multirate_ack(bytes(ack)),requested_version=5)

    def test_only_g8_sentinel_changes_all_profiles_and_rois(self):
        old = sentinel_offsets_for_version(4); new = sentinel_offsets_for_version(5)
        self.assertEqual([g for g in range(9) if old[g] != new[g]],[7])
        for profile in (0,1,2,3):
            pairs = [(255,255)] if profile < 2 else (
                [(g,255) for g in range(9)] if profile == 3 else
                [(a,b) for a in range(9) for b in range(9) if a != b])
            for primary,secondary in pairs:
                points = (tuple(range(45)) if profile == 0 else
                          tuple(g*5+o for g,o in enumerate(new)) if profile == 1 else
                          track_indices(primary,secondary,g8_right=True) if profile == 2 else
                          track_single_indices(primary,g8_right=True))
                ext = _v4_schedule_extension(profile,points,primary=primary,secondary=secondary,version=5)
                schedule = decode_board_frame_schedule(ext)
                self.assertEqual(schedule.fresh_indices,points)
                self.assertEqual(decode_board_frame_start_ms(ext),1234)
                if profile == 0:
                    _require_initial_current_map(schedule,decode_board_frame_timing(ext),requested_version=5)
                    with self.assertRaises(RuntimeError):
                        _require_initial_current_map(schedule,decode_board_frame_timing(ext))
                if profile == 1 or (profile >= 2 and 7 not in (primary,secondary)):
                    wrong = bytearray(ext); wrong[36] = 4
                    with self.assertRaises(ValueError): decode_board_frame_schedule(bytes(wrong))

    def test_reference_uses_new_sentinel_for_baseline_and_contact(self):
        s = Scheduler(map_period_frames=0,single_track_enabled=True,g8_right=True)
        values = [100,300,600,300,100]*9
        def frame():
            plan=s.begin_frame()
            for p in plan.fresh_indices: s.record(p,values[p])
            s.end_frame(); return plan
        self.assertEqual(frame().profile,Profile.MAP)
        for _ in range(8):
            plan=frame(); self.assertIn(38,plan.fresh_indices); self.assertNotIn(36,plan.fresh_indices)
        values[36]=4000  # cached/unused cannot create G8 contact
        frame(); self.assertFalse(s.activity_latched)
        values[38]=400
        frame(); self.assertTrue(s.activity_latched); self.assertEqual(s.active_pair[0],7)
        self.assertEqual(s.last_scores[7],100)
        self.assertEqual(s._rank_score(7),16500)
        self.assertEqual(frame().profile,Profile.TRACK_WIDE)
        s.start_session(); self.assertTrue(s.g8_right)

    def test_g8_weight_preserves_relative_template_sensitivity(self):
        p=Path(__file__).parent/'artifacts/ch1_measured_template_selection_v4_certified_20260905.json'
        slopes=json.loads(p.read_text('utf-8'))['stress']['peaks'][7]['selected_template_slope_v_per_nm']
        self.assertEqual(round(71*abs(slopes[1]/slopes[3])),165)

    def test_held_capture_requires_new_prefix_and_explicit_requested_version(self):
        b=bytearray(held_fixture()); b[14]=5; b[8:12]=(0x1001d).to_bytes(4,'big')
        b[38+36//8] &= ~(1<<(36%8)); b[38+38//8] |= 1<<(38%8)
        r=decode_held(bytes(b),36,99,2,5)
        self.assertEqual(r['schedule_version'],5)
        self.assertFalse(r['target_would_normally_be_sampled'])
        with self.assertRaises(ValueError): decode_held(bytes(b),36,99,2)
        b[38+36//8] |= 1<<(36%8); b[38+38//8] &= ~(1<<(38%8))
        with self.assertRaises(ValueError): decode_held(bytes(b),36,99,2,5)

    def test_preserved_pairs_reject_old_bitmap_and_keep_unread_rows_empty(self):
        b=bytearray(raw_fixture()); b[22]=5; b[36:40]=(0x1001d).to_bytes(4,'big')
        with self.assertRaises(ValueError): decode_raw(bytes(b),11,2,5)
        b[28+36//8] &= ~(1<<(36%8)); b[28+38//8] |= 1<<(38%8)
        a=40+36*13; c=40+38*13
        b[c:c+13]=b[a:a+13]; b[a:a+13]=bytes(9)+b'\xff'*4
        r=decode_raw(bytes(b),11,2,5)
        self.assertIsNone(r['records'][36]['first_direct_adc'])
        self.assertEqual(r['records'][38]['first_direct_adc'],10)
        with self.assertRaises(ValueError): decode_raw(bytes(b),11,2)

    def test_benchmark_counts_only_requested_version_and_new_sentinel(self):
        points=tuple(g*5+o for g,o in enumerate(sentinel_offsets_for_version(5)))
        schedule=decode_board_frame_schedule(_v4_schedule_extension(1,points,version=5))
        r=_summarize_schedule_freshness([schedule],requested_version=5)
        self.assertEqual(r['schedule_version'],5)
        self.assertEqual(r['current_version_frames'],1)
        self.assertEqual(_summarize_schedule_freshness([schedule])['current_version_frames'],0)
        trace=[{'fresh_ch1_codes':{'36':0,'38':50}}]*4
        h=summarize_sentinel_signal_health(trace,requested_version=5)
        self.assertEqual(h['per_grating']['G8']['point_index'],38)
        self.assertFalse(h['per_grating']['G8']['needs_review'])


if __name__ == '__main__': unittest.main()
