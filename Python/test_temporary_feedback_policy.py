"""Evaluate the actual firmware feedback guards without board I/O.

The host must not silently force CH0 to 2 kOhm to bypass a firmware bug.
These tests evaluate the C predicates read from ms5614t.c, not a duplicate
Python policy, including the legacy 0x0200 diagnostic bit.
"""
import re
import unittest
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / 'JDSU/Core/Src/ms5614t.c'


def condition_before_reject(source, bit):
    marker = f'reject |= {bit}U;'
    assert source.count(marker) == 1, marker
    end = source.index(marker)
    start = source.rfind('if(', 0, end) + 3
    assert start >= 3
    depth = 1
    cursor = start
    while depth:
        char = source[cursor]
        depth += (char == '(') - (char == ')')
        cursor += 1
    assert not source[cursor:end].strip()
    return source[start:cursor - 1]


def evaluate_c_condition(expression, **values):
    expression = re.sub(r'PD_ReadFeedbackSelector\(([01])U\)',
                        lambda m: f'actual{m[1]}', expression)
    expression = re.sub(r'\b(\d+)U\b', r'\1', expression)
    expression = ' '.join(expression.replace('&&', ' and ').replace('||', ' or ').split())
    # Only these simple boolean comparisons are expected. Never execute C or
    # arbitrary Python / hardware imports while checking the guards.
    names = set(re.findall(r'\b[A-Za-z_]\w*\b', expression)) - {'and', 'or'}
    assert names <= set(values), names
    assert re.fullmatch(r'[A-Za-z0-9_\s()!<>=&|]+', expression), expression
    return bool(eval(expression, {'__builtins__': {}}, values))


class TemporaryFeedbackPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text(encoding='utf-8')
        cls.legacy_guard = condition_before_reject(cls.source, 512)
        cls.channel_guard = condition_before_reject(cls.source, 1024)

    def rejection(self, channel, requested, actual0, actual1):
        expected0 = requested if channel == 0 else 0
        expected1 = requested if channel == 1 else 0
        values = dict(selectedAdcChannel=channel, expectedSelector0=expected0,
                      expectedSelector1=expected1, actual0=actual0, actual1=actual1)
        return ((512 if evaluate_c_condition(self.legacy_guard, **values) else 0) |
                (1024 if evaluate_c_condition(self.channel_guard, **values) else 0))

    def test_ch0_5k_regression_matches_reported_0200_failure(self):
        self.assertEqual(self.rejection(0, 2, 2, 0), 0)

    def test_all_switchable_ch0_ch1_feedback_settings_are_accepted(self):
        for channel in (0, 1):
            for selector in range(4):
                with self.subTest(channel=channel, selector=selector):
                    actual = (selector, 0) if channel == 0 else (0, selector)
                    self.assertEqual(self.rejection(channel, selector, *actual), 0)

    def test_fixed_ch2_ch3_keep_both_switchable_frontends_at_2k(self):
        for channel in (2, 3):
            with self.subTest(channel=channel):
                self.assertEqual(self.rejection(channel, 0, 0, 0), 0)
                self.assertTrue(self.rejection(channel, 0, 2, 0) & 512)
                self.assertTrue(self.rejection(channel, 0, 0, 2) & 1024)

    def test_wrong_selected_or_unselected_feedback_is_still_rejected(self):
        for channel in (0, 1):
            for selector in range(4):
                for wrong in range(4):
                    if wrong == selector:
                        continue
                    actual = (wrong, 0) if channel == 0 else (0, wrong)
                    self.assertTrue(self.rejection(channel, selector, *actual) & 1024)
                actual = (selector, 1) if channel == 0 else (1, selector)
                self.assertTrue(self.rejection(channel, selector, *actual) & 1024)

    def test_invalid_selector_and_legacy_ch1_safety_remain_rejected(self):
        self.assertTrue(self.rejection(0, 4, 4, 0) & 1024)
        self.assertTrue(self.rejection(1, 4, 0, 4) & 1024)
        self.assertTrue(self.rejection(1, 2, 2, 2) & 512)

    def test_guards_follow_selected_channel_and_precede_source_writes(self):
        self.assertRegex(self.source,
                         r'selectedAdcChannel\s*=\s*\(temporary45Stream && \(route == 5U \|\| route == 6U \|\| route == 7U\)\)'
                         r'\s*\? candidateRoute\.adcChannel : 1U;')
        self.assertRegex(self.source,
                         r'selectedFeedback\s*=\s*temporary45Stream'
                         r'\s*\? candidateRoute\.feedbackSelector : requestedStreamSelector1;')
        feedback_end = self.source.index('reject |= 1024U;')
        shutter_guard = self.source.index('reject |= 2048U;', feedback_end)
        rejection_return = self.source.index('return;', shutter_guard)
        source_write = self.source.index('PI11210_ApplyCalibrationCodes(', rejection_return)
        self.assertLess(feedback_end, shutter_guard)
        self.assertLess(rejection_return, source_write)
        self.assertLess(self.source.index('uint8_t selectedAdcChannel ='),
                        self.source.index('reject |= 512U;'))


if __name__ == '__main__':
    unittest.main()
