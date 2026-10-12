import copy
from dataclasses import asdict
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import screen_ch1_reduced_pairs as screen
from test_capture_certified_ch1_transients import fixture as transient_wire


def fixture(gratings=(1,), variant='two_flanks'):
    rows, hidden = screen.load_installed_rows()
    cases = screen.build_cases(rows, gratings, variant)
    report = dict(schema='ch1_reduced_edge_pair_screen_v1', complete=True, repeats=1,
                  installed_rows=rows, cases=cases, captures=[], candidate_variant=variant,
                  candidate_indices=list(screen.candidate_indices(variant)))
    raw = b'\xff\xff\x02\x02' + b'\x00\x64'*6 + bytes(4)
    tail = dict(monitor_wire_hex=[raw.hex()]*15, monitor_adc=[[100]*6]*15,
                ch1_adc_code=100., stable=True)
    for case in cases:
        for path, source in case['sources'].items():
            tag = len(report['captures']) + 1
            wire = bytearray(transient_wire())
            wire[4:6], wire[6:8] = source.to_bytes(2, 'big'), case['target'].to_bytes(2, 'big')
            wire[14:18] = tag.to_bytes(4, 'big')
            for index in range(33):
                wire[36+index*16:38+index*16] = (100+index).to_bytes(2, 'big')
            decoded = screen.decode(bytes(wire), source, case['target'], tag)
            reading = screen.Reading(case['target_nm'], 0., 1.7, 2.3, 1)
            report['captures'].append(dict(target=case['target'], source=source, path=path, repeat=0,
                target_codes=case['target_codes'], source_codes=rows[source]['codes'], transient=decoded,
                before_meter=copy.deepcopy(tail), after_meter=copy.deepcopy(tail),
                meter=asdict(reading), review=screen.optical_review(reading, case['target_nm']),
                meter_start_host_ns=1, meter_end_host_ns=2))
    return report


class ReducedPairsTests(unittest.TestCase):
    def test_eighteen_candidate_keeps_both_flanks_each_grating_and_wrap(self):
        rows, _ = screen.load_installed_rows()
        cases = screen.build_cases(rows)
        self.assertEqual(len(cases), 18)
        self.assertEqual(cases[0]['sources'], dict(adjacent_held=0, reduced18_held=43))
        self.assertEqual(cases[1]['sources'], dict(adjacent_held=2, reduced18_held=1))
        for grating in range(1, 10):
            self.assertEqual([c['target']%5 for c in cases if c['grating']==grating], [1, 3])

    def test_invalid_bounds_and_existing_output_never_open_hardware(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(screen.serial, 'Serial') as device:
            output = Path(directory)/'existing.json'
            output.touch()
            with self.assertRaises(FileExistsError): screen.execute(output)
            for repeats in (0, 3, True, 1.5):
                with self.assertRaises(ValueError): screen.execute(Path(directory)/'new.json', repeats=repeats)
            for gratings in ((), (0,), (10,), (1,1), (True,)):
                with self.assertRaises(ValueError): screen.execute(Path(directory)/'new.json', gratings=gratings)
            for variant in ('unknown', '', None, [17, 18], True):
                with self.assertRaises(ValueError): screen.execute(Path(directory)/'new.json', variant=variant)
            device.assert_not_called()

    def test_peak_right_variant_keeps_exact_predecessors_without_changing_other_gratings(self):
        rows, _ = screen.load_installed_rows()
        full = screen.build_cases(rows, variant='g4_g8_peak_right')
        self.assertEqual(len(full), 18)
        selected = screen.build_cases(rows, (4, 8), 'g4_g8_peak_right')
        self.assertEqual([(c['target'], c['sources']['adjacent_held'], c['sources']['reduced18_held'])
                          for c in selected], [(17, 16, 13), (18, 17, 17), (37, 36, 33), (38, 37, 37)])
        base = screen.build_cases(rows)
        self.assertEqual([c for c in full if c['grating'] not in (4, 8)],
                         [c for c in base if c['grating'] not in (4, 8)])
        result = screen.summarize(fixture((4, 8), 'g4_g8_peak_right'))
        self.assertEqual(result['candidate_variant'], 'g4_g8_peak_right')
        self.assertEqual(len(result['trials']), 8)
        self.assertFalse(result['training_eligible'])

    def test_legacy_default_report_still_redecodes_but_variant_map_mismatch_is_rejected(self):
        report = fixture()
        del report['candidate_variant']
        self.assertEqual(screen.summarize(report)['candidate_variant'], 'two_flanks')
        del report['candidate_indices']
        self.assertEqual(screen.summarize(report)['candidate_variant'], 'two_flanks')
        for kind in ('variant', 'indices', 'case_duplicate', 'wrong_variant'):
            report = fixture((4, 8), 'g4_g8_peak_right')
            if kind == 'variant': report['candidate_variant'] = 'unknown'
            elif kind == 'indices': report['candidate_indices'][0] = 2
            elif kind == 'case_duplicate': report['cases'].append(copy.deepcopy(report['cases'][0]))
            elif kind == 'wrong_variant': report['candidate_variant'] = 'two_flanks'
            with self.subTest(kind=kind), self.assertRaises(ValueError): screen.summarize(report)

    def test_reverse_g8_keeps_original_rows_but_audits_changed_g9_entry(self):
        rows, _ = screen.load_installed_rows()
        full = screen.build_cases(rows, variant='g8_right_left')
        self.assertEqual(set(c['target'] for c in full), set(screen.REDUCED_INDICES))
        selected = screen.build_cases(rows, (8, 9), 'g8_right_left')
        self.assertEqual([(c['target'], c['sources']['reduced18_held']) for c in selected],
                         [(38, 33), (36, 38), (41, 36), (43, 41)])
        report = fixture((8, 9), 'g8_right_left')
        self.assertEqual(len(screen.summarize(report)['trials']), 8)
        report['cases'][2]['sources']['reduced18_held'] = 38
        with self.assertRaises(ValueError): screen.summarize(report)

    def test_raw_redecode_and_causal_group_completion(self):
        result = screen.summarize(fixture())
        self.assertEqual(len(result['trials']), 4)
        self.assertFalse(result['all_nine_gratings_covered'])
        self.assertFalse(result['continuous_path_qualified'])
        self.assertFalse(result['physical_15hz_verified'])
        # Group at 505 us has no completion bound before 650 us; preceding
        # group at 405 us does, because the next group starts at 505 us.
        self.assertEqual(result['trials'][0]['six_adc_group_completed_by_cutoff_codes']['650'], 108)

    def test_missing_duplicate_codes_raw_tail_and_meter_fail_closed(self):
        for kind in ('missing', 'duplicate', 'source', 'codes', 'raw', 'tail', 'nan', 'time', 'tag'):
            report = fixture()
            row = report['captures'][0]
            if kind == 'missing': report['captures'].pop()
            elif kind == 'duplicate': report['captures'][-1] = copy.deepcopy(row)
            elif kind == 'source': row['source'] = 1
            elif kind == 'codes': row['target_codes'] = [1]*5
            elif kind == 'raw': row['transient']['direct_adc'][0][1] = 2
            elif kind == 'tail': row['before_meter']['monitor_adc'][0][3] = 2
            elif kind == 'nan': row['meter']['wavelength_nm'] = float('nan')
            elif kind == 'time': row['meter_end_host_ns'] = 0
            elif kind == 'tag': report['captures'][1]['transient']['request_tag'] = row['transient']['request_tag']
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                screen.summarize(report)

    def test_meter_setup_failure_still_shutters(self):
        port = Mock()
        port.__enter__ = Mock(return_value=port)
        port.__exit__ = Mock(return_value=False)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(screen.serial, 'Serial', return_value=port), \
                patch.object(screen, '_safe_disarm', return_value=True) as disarm, \
                patch.object(screen, '_close_shutter', return_value=True) as shutter, \
                patch.object(screen.app, 'EqualIntervalWorker') as worker, \
                patch.object(screen, 'AQ6150B', side_effect=RuntimeError('meter missing')):
            worker.return_value._feedback_command_and_ack.return_value = True
            output = Path(directory)/'failed.json'
            with self.assertRaisesRegex(RuntimeError, 'meter missing'):
                screen.execute(output, repeats=1, gratings=(1,))
            self.assertEqual(shutter.call_count, 2)
            self.assertEqual(disarm.call_count, 2)
            saved = json.loads(output.read_text('utf-8'))
            self.assertFalse(saved['complete'])
            self.assertTrue(saved['safety_cleanup']['exact_soa_shutter_ack'])

    def test_no_motion_or_install_entry_points_and_tail_has_no_dac_rewrite(self):
        source = inspect.getsource(screen.execute)
        for unsafe in ('Mach3Controller', '.move_', 'install_', 'set_firmware'):
            self.assertNotIn(unsafe, source)
        between = source.split("record['transient'] = capture", 1)[1].split('save_checkpoint', 1)[0]
        self.assertNotIn('_command_and_ack', between)
        self.assertNotIn('.write(', between)

    def test_meter_failure_after_laser_output_keeps_partial_and_shutters(self):
        port = Mock()
        port.__enter__ = Mock(return_value=port)
        port.__exit__ = Mock(return_value=False)
        ordering = []
        def shut(_):
            ordering.append('shutter')
            return True
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(screen.serial, 'Serial', return_value=port), \
                patch.object(screen, '_safe_disarm', return_value=True), \
                patch.object(screen, '_close_shutter', side_effect=shut), \
                patch.object(screen.app, 'EqualIntervalWorker') as worker, \
                patch.object(screen, 'capture', return_value=fixture()['captures'][0]['transient']), \
                patch.object(screen, 'capture_stable_tail', return_value=fixture()['captures'][0]['before_meter']), \
                patch.object(screen, 'AQ6150B') as meter:
            worker.return_value._command_and_ack.return_value = 1.
            meter.return_value.identity = 'test meter'
            meter.return_value.measure.side_effect = RuntimeError('query timeout')
            meter.return_value.close.side_effect = lambda: ordering.append('meter close')
            output = Path(directory)/'partial.json'
            with self.assertRaisesRegex(RuntimeError, 'query timeout'):
                screen.execute(output, repeats=1, gratings=(1,))
            saved = json.loads(output.read_text('utf-8'))
            self.assertFalse(saved['complete'])
            self.assertEqual(len(saved['captures']), 1)
            self.assertIn('transient', saved['captures'][0])
            self.assertNotIn('after_meter', saved['captures'][0])
            self.assertTrue(saved['safety_cleanup']['exact_soa_shutter_ack'])
            self.assertEqual(ordering, ['shutter', 'shutter', 'meter close'])


if __name__ == '__main__':
    unittest.main()
