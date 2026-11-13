"""Birth-date checks must agree with the scalar parsing used by the model."""
import sys
from pathlib import Path
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from validate_data import validate_fighters


class FighterProfileValidationTests(unittest.TestCase):
    def profiles(self, dates):
        return pd.DataFrame([
            dict(name=f'Fighter {i}', url=f'http://ufcstats.test/fighter/{i}',
                 height='5\' 9"', reach='71"', stance='Orthodox', dob=dob)
            for i, dob in enumerate(dates)
        ])

    def test_may_first_does_not_invalidate_abbreviated_months(self):
        dates = ['May 13, 2000', 'Oct 16, 1998', 'Sep 30, 1999', 'February 10, 1996']
        for order in (dates, list(reversed(dates))):
            with self.subTest(order=order):
                self.assertEqual(validate_fighters(self.profiles(order)).errors, [])

    def test_invalid_date_identifies_fighter_without_duplicate_age_error(self):
        report = validate_fighters(self.profiles(['Feb 30, 1996']))
        self.assertEqual(len(report.errors), 1)
        self.assertIn('invalid known DOB', report.errors[0])
        self.assertIn('Fighter 0', report.errors[0])
        self.assertIn('http://ufcstats.test/fighter/0', report.errors[0])

    def test_implausible_dates_still_fail(self):
        for dob in ['Jan 01, 1920', 'Jan 01, 2099']:
            with self.subTest(dob=dob):
                report = validate_fighters(self.profiles([dob]))
                self.assertEqual(len(report.errors), 1)
                self.assertIn('implausible known DOB', report.errors[0])

    def test_unknown_and_empty_dates_remain_allowed(self):
        self.assertEqual(validate_fighters(self.profiles(['--', None, '', 'N/A'])).errors, [])


if __name__ == '__main__':
    unittest.main()
