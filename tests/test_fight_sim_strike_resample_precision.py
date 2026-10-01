import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fight_sim.strike_resample_precision import (
    DESIGN_PATH, FINAL_STREAMS, V2, _card_worker, card_weights, decision, differential_precision, member_seeds, verify_seed_storage,
)


class StrikeResamplePrecisionTests(unittest.TestCase):
    def test_larger_seed_sequence_preserves_original_ten_members(self):
        first = member_seeds(2903, "card", "2024-01-01T00:00:00+00:00", 10)
        extended = member_seeds(2903, "card", "2024-01-01T00:00:00+00:00", 100)
        other = member_seeds(20261001, "card", "2024-01-01T00:00:00+00:00", 100)
        np.testing.assert_array_equal(first, extended[:10])
        self.assertFalse(np.array_equal(extended, other))

    def test_sampling_repeats_whole_cards_and_rejects_future_cards(self):
        frame = pd.DataFrame({"event_id": ["a"]*4+["b"]*2+["c"]*6,
                              "date": pd.to_datetime(["2024-01-01"]*12, utc=True)})
        weights = card_weights(frame, 123, "2024-01-02")
        grouped = frame.assign(weight=weights).groupby("event_id").weight
        self.assertTrue(grouped.nunique().eq(1).all())
        self.assertEqual(grouped.first().sum(), 3)
        reordered = frame.iloc[::-1]
        recovered = pd.Series(card_weights(reordered, 123, "2024-01-02"), index=reordered.index).sort_index()
        np.testing.assert_array_equal(weights, recovered.to_numpy())
        frame.loc[0, "date"] = pd.Timestamp("2024-01-02", tz="UTC")
        with self.assertRaisesRegex(ValueError, "strictly earlier"):
            card_weights(frame, 123, "2024-01-02")

    def test_worker_preserves_uint64_seed_digits_when_members_are_concatenated(self):
        training = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]*2, utc=True),
                                 "event_id": ["old", "old"], "fight_id": ["old-fight", "old-fight"]})
        test = pd.DataFrame({"fight_id": ["fight", "fight"], "fighter_id": ["r", "b"],
                             "event_id": ["card", "card"], "date": pd.to_datetime(["2024-01-02"]*2, utc=True),
                             "side": ["red", "blue"], "winner": ["red", "red"], "fight_seconds": [900,900],
                             "sig_strikes_attempts": [100,100], "sig_strikes_landed": [50,50]})
        cutoff = pd.Timestamp("2024-01-02", tz="UTC")
        prediction = {"strike_pace": np.array([7.,7.]), "strike_accuracy": np.array([.5,.5])}
        job = {"training": training, "test": test, "cutoff": cutoff,
               "selected": {("strike_pace", "opponent_adjusted"): 10, ("strike_accuracy", "opponent_adjusted"): 10},
               "design": {"streams": {"independent_seed": 20261001}, "members_per_stream": 3},
               "reference": np.array([[7,.5],[7,.5]]), "event_id": "card", "deadline": time.monotonic()+60,
               "inner_ids": (), "old_prefix": {}}
        with patch("fight_sim.strike_resample_precision.forecast_arm", return_value=prediction):
            worker = _card_worker(job)
            result = worker["members"]
        expected = member_seeds(20261001, "card", cutoff, 3)
        self.assertTrue(result.member_seed.map(lambda value: isinstance(value, str)).all())
        self.assertEqual(result.groupby("member").member_seed.first().tolist(), [str(int(seed)) for seed in expected])
        self.assertEqual(verify_seed_storage(result, [worker["provenance"]], job["design"])["exact_seed_strings"], 6)
        with self.assertRaisesRegex(ValueError, "lost integer precision"):
            verify_seed_storage(result.assign(member_seed=result.member_seed.astype(float)), [worker["provenance"]], job["design"])

    def test_differential_uncertainty_preserves_red_blue_member_covariance(self):
        rows = []
        for member, common in enumerate((10., 100., 1000.)):
            for side, rate in (("red", common), ("blue", common-2)):
                rows.append({"fight_id": "fight", "event_id": "card", "stream": "sample",
                             "member": member, "side": side, "landed_rate": rate})
        result = differential_precision(pd.DataFrame(rows)).iloc[0]
        self.assertEqual(result["mean"], 2)
        self.assertEqual(result["standard_error"], 0)
        self.assertFalse(result["sign_unresolved_by_normal_95_mc_interval"])

    def passing_inputs(self):
        design = json.loads(DESIGN_PATH.read_text())
        scores = {
            V2: {"pace_nll": 18., "accuracy_nll": 5.},
            "original_marginal": {"pace_nll": 17., "accuracy_nll": 4.7},
            **{stream: {"pace_nll": 16., "accuracy_nll": 4.5, "strike_leader_wins": 120,
                         "decisive_fights": 228} for stream in FINAL_STREAMS},
        }
        stable = {"strike_leader_agreement": 1., "accuracy_rms_difference": 0., "pace_relative_rms_difference": 0.}
        return design, scores, stable

    def test_both_streams_must_pass_even_if_pooled_or_one_seed_is_good(self):
        design, scores, stable = self.passing_inputs()
        self.assertTrue(decision(scores, stable, design)["candidate_advances_to_simulation_screen"])
        for stream in FINAL_STREAMS:
            failed = copy.deepcopy(scores)
            failed[stream]["strike_leader_wins"] = 119
            failed["pooled_200_descriptive"] = dict(scores[FINAL_STREAMS[0]], strike_leader_wins=140)
            self.assertFalse(decision(failed, stable, design)["candidate_advances_to_simulation_screen"])

    def test_performance_does_not_override_instability_or_numerical_rounding(self):
        design, scores, stable = self.passing_inputs()
        for field, value in (("strike_leader_agreement", .949), ("accuracy_rms_difference", .011),
                             ("pace_relative_rms_difference", .021)):
            self.assertFalse(decision(scores, dict(stable, **{field: value}), design)["candidate_advances_to_simulation_screen"])
        scores[FINAL_STREAMS[0]]["accuracy_nll"] = scores["original_marginal"]["accuracy_nll"]-1e-9
        self.assertFalse(decision(scores, stable, design)["candidate_advances_to_simulation_screen"])


if __name__ == "__main__":
    unittest.main()
