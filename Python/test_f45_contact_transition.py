import unittest
from audit_f45_contact_transition import transition, validated_gain_identity, require_fresh_capture
from test_capture_fast_fullmap_stream import data_frame
from test_capture_fast_flank_stream import end_frame
from capture_fast_fullmap_stream import decode_data
from test_capture_fast_fullmap_stream import _finish_crc

HISTORICAL_FIRMWARE = 0x10036


def historical_data_frame(ch1_selector=2):
    raw = bytearray(data_frame())
    raw[8:12] = HISTORICAL_FIRMWARE.to_bytes(4, 'big')
    raw[33] = ch1_selector
    return _finish_crc(raw)


def historical_end_frame(ch1_selector=2):
    raw = bytearray(end_frame())
    raw[8:12] = HISTORICAL_FIRMWARE.to_bytes(4, 'big')
    raw[21] = ch1_selector
    return _finish_crc(raw)


class TransitionTests(unittest.TestCase):
    def test_timing_metadata_corruption_rejected(self):
        frame = decode_data(historical_data_frame(), tag=123, table_crc32=456,
                            spacing_us=75, expected_firmware=HISTORICAL_FIRMWARE)
        data = {'terminal':{'wire_hex':historical_end_frame().hex()},'frames':[frame], 'table_codes_crc32':456}
        frame['cycle_start_us'] += 1
        with self.assertRaisesRegex(ValueError, 'timing'): validated_gain_identity(data)

    def test_prospective_validation_rejects_old_or_training_data(self):
        payload={'created_utc':'2026-09-08T00:00:00+00:00','captures':[{'sha256':'trained'}]}
        data={'created_utc':'2026-09-08T00:01:00+00:00'}
        require_fresh_capture(payload,data,'new')
        with self.assertRaises(ValueError): require_fresh_capture(payload,data,'trained')
        with self.assertRaises(ValueError): require_fresh_capture(payload,{'created_utc':payload['created_utc']},'new')

    def test_explicit_alternate_gain_still_checked(self):
        frame=decode_data(historical_data_frame(3),tag=123,table_crc32=456,
                          spacing_us=75,ch1_selector=3,
                          expected_firmware=HISTORICAL_FIRMWARE)
        data={'terminal':{'wire_hex':historical_end_frame(3).hex()},'frames':[frame],
              'feedback_selectors':[0,3],'table_codes_crc32':456}
        self.assertEqual(validated_gain_identity(data,(0,3))['selectors'],[0,3])
        with self.assertRaises(ValueError): validated_gain_identity(data)

    def test_legacy_gain_recovered_and_corruption_rejected(self):
        frame = decode_data(historical_data_frame(), tag=123, table_crc32=456,
                            spacing_us=75, expected_firmware=HISTORICAL_FIRMWARE)
        data = {'terminal':{'wire_hex':historical_end_frame().hex()},'frames':[frame], 'table_codes_crc32':456}
        self.assertEqual(validated_gain_identity(data)['selectors'],[0,2])
        frame['records'][0]['first_code'] += 1
        with self.assertRaises(ValueError): validated_gain_identity(data)

    def test_initial_misses_not_hidden_by_settling(self):
        events = dict(contact_descent_started=1,contact_reached_commanded_depth=1_000_000_000,
                      release_started=2_000_000_000,safe_height_reached=3_000_000_000)
        result = transition([1_010_000_000,1_100_000_000,1_300_000_000,3_010_000_000],
                            [0,0,17,0],events,17)
        self.assertEqual(result['first_200ms_after_depth']['correct'],0)
        self.assertEqual(result['whole_depth_hold']['first_correct_receipt_after_event_ms'],300)
        self.assertEqual(result['after_safe_retraction']['first_correct_receipt_after_event_ms'],10)


if __name__ == '__main__': unittest.main()
