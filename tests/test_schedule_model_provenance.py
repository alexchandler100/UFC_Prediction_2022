import sys
from pathlib import Path
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from refresh_schedule_model_provenance import model_id, reverse_recorded_schedules


class ScheduleModelProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame({'fight_url': ['https://ufcstats.com/fight-details/a'] * 2,
                                 'time_format': ['5 Rnd (5-5-5-5-5)'] * 2,
                                 'result': ['W', 'L'], 'sig_strikes_landed': [11, 12]})
        self.change = {'fight_id': 'a', 'changed_side_cells': 2,
                       'time_format': '5 Rnd (5-5-5-5-5)'}

    def test_reconstruction_changes_only_recorded_schedules(self):
        before = self.raw.copy(deep=True)
        recovered = reverse_recorded_schedules(self.raw, [self.change])
        self.assertEqual(recovered.time_format.tolist(), ['', ''])
        pd.testing.assert_frame_equal(self.raw, before)
        pd.testing.assert_frame_equal(recovered.drop(columns='time_format'), before.drop(columns='time_format'))

    def test_rejects_unexpected_schedule_duplicate_and_missing_side(self):
        for changes in ([{**self.change, 'time_format': '3 Rnd (5-5-5)'}],
                        [self.change, self.change], [{**self.change, 'fight_id': 'missing'}]):
            with self.assertRaises(ValueError):
                reverse_recorded_schedules(self.raw, changes)
        with self.assertRaises(ValueError):
            reverse_recorded_schedules(self.raw.iloc[:1], [self.change])

    def test_new_identity_binds_new_state_without_changing_parameters(self):
        old = {'coefficients': [.1, .2], 'state_fingerprint_sha256': 'old'}
        old['model_id'] = model_id(old)
        updated = {**old, 'state_fingerprint_sha256': 'new'}
        self.assertNotEqual(model_id(updated), old['model_id'])
        self.assertEqual(updated['coefficients'], old['coefficients'])
        self.assertEqual(model_id({**old, 'model_id': 'ignored'}), old['model_id'])


if __name__ == '__main__':
    unittest.main()
