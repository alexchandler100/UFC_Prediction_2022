from dataclasses import asdict, replace
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fight_sim.domain import BoutConfig, FighterParameters, FighterSnapshot, Side, SimulationRunSpec, SimulatorConfig
from fight_sim.opponent_audit import OpponentAdjustmentAuditConfig, _context_predictions, _fit_target, _linear_prediction
from fight_sim.strike_bridge_audit import integrate, neutral_phase_rates
from fight_sim.strike_handoff import STRIKE_TARGETS, context_occupancy, fit_strike_effects, map_matchup_strikes, weighted_observation_context
from fight_sim.strike_handoff_audit import forecast_arm


def training_rows():
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01"]*4 + ["2020-01-08"]*4, utc=True),
        "division": ["Lightweight", "Lightweight", "Heavyweight", "Heavyweight"]*2,
        "era": ["2020-2024"]*8,
        "fighter_id": ["a", "b", "c", "d", "a", "b", "c", "d"],
        "opponent_id": ["b", "a", "d", "c", "b", "a", "d", "c"],
        "fight_seconds": [900, 900, 125, 125, 900, 900, 725, 725],
        "sig_strikes_attempts": [150, 105, 21, 14, 165, 87, 67, 88],
        "sig_strikes_landed": [75, 30, 12, 7, 93, 36, 20, 26],
    })


def spec():
    red = FighterSnapshot("red", "Red", "2024-01-01T00:00:00+00:00", "Lightweight",
                          FighterParameters(strike_rate_distance=12, strike_rate_clinch=4,
                                            strike_rate_ground=9, strike_defense=.7))
    blue = replace(red, fighter_id="blue", fighter_name="Blue", parameters=replace(
        red.parameters, strike_rate_distance=4, strike_rate_clinch=10, strike_rate_ground=16, strike_defense=.3))
    return SimulationRunSpec(bout=BoutConfig("test", "red", "blue"), red=red, blue=blue,
                             root_seed="fixed", parameter_artifact_id="historical", simulator=SimulatorConfig(
                                 distance_strike_hazard_multiplier=1.5, clinch_strike_hazard_multiplier=.5,
                                 ground_strike_hazard_multiplier=.75))


class StrikeHandoffTests(unittest.TestCase):
    def test_unit_weights_reproduce_original_context_and_effects(self):
        frame = training_rows()
        settings = OpponentAdjustmentAuditConfig()
        for target in STRIKE_TARGETS:
            original = _fit_target(frame, target, settings)
            context = weighted_observation_context(frame, target, np.ones(len(frame)))
            baseline = _context_predictions(frame, context)
            np.testing.assert_allclose(baseline, _context_predictions(frame, original.context), rtol=1e-14)
            effect = fit_strike_effects(frame, target, baseline, np.ones(len(frame)), 10, cutoff="2020-01-09")
            expected = [_linear_prediction(base, actor, opponent, target, original, 10, "opponent_adjusted")
                        for base, actor, opponent in zip(baseline, frame.fighter_id, frame.opponent_id)]
            np.testing.assert_allclose(effect.predict(baseline, frame.fighter_id, frame.opponent_id), expected, rtol=1e-14)

    def test_card_multiplicity_matches_repeated_bouts(self):
        frame = training_rows()
        weights = np.array([0, 0, 0, 0, 3, 3, 2, 2])
        expanded = frame.iloc[np.repeat(np.arange(len(frame)), weights)].reset_index(drop=True)
        for target in STRIKE_TARGETS:
            weighted = weighted_observation_context(frame, target, weights)
            repeated = weighted_observation_context(expanded, target, np.ones(len(expanded)))
            self.assertEqual(weighted, repeated)
            first = fit_strike_effects(frame, target, _context_predictions(frame, weighted), weights, 10, cutoff="2020-01-09")
            second = fit_strike_effects(expanded, target, _context_predictions(expanded, repeated), np.ones(len(expanded)), 10, cutoff="2020-01-09")
            np.testing.assert_allclose(first.predict(_context_predictions(frame, weighted), frame.fighter_id, frame.opponent_id),
                                       second.predict(_context_predictions(frame, repeated), frame.fighter_id, frame.opponent_id), rtol=1e-14)

    def test_same_card_and_future_training_are_rejected(self):
        frame = training_rows()
        for cutoff in ("2020-01-08", "2019-12-31"):
            with self.assertRaisesRegex(ValueError, "strictly before"):
                fit_strike_effects(frame, STRIKE_TARGETS[0], np.ones(len(frame))*7, np.ones(len(frame)), 10, cutoff=cutoff)
        frame.loc[0, "date"] = pd.NaT
        with self.assertRaisesRegex(ValueError, "strictly before"):
            fit_strike_effects(frame, STRIKE_TARGETS[0], np.ones(len(frame))*7, np.ones(len(frame)), 10, cutoff="2020-01-09")

    def test_candidate_forecasts_ignore_test_fight_outcomes_and_duration(self):
        training = training_rows()
        test = training.iloc[:4].copy()
        test["date"] = pd.Timestamp("2020-01-09", tz="UTC")
        selected = {(target.name, "opponent_adjusted"): 10 for target in STRIKE_TARGETS}
        before = forecast_arm(training, test, np.ones(len(training)), selected, "2020-01-09")
        test["sig_strikes_attempts"] = 999999
        test["sig_strikes_landed"] = 0
        test["fight_seconds"] = 1
        test["result"] = "changed"
        after = forecast_arm(training, test, np.ones(len(training)), selected, "2020-01-09")
        for target in STRIKE_TARGETS:
            np.testing.assert_array_equal(before[target.name], after[target.name])

    def test_mapping_preserves_both_sides_and_non_strike_state(self):
        original = spec()
        weights = context_occupancy({"distance_phase_share": .65, "clinch_phase_share": .1, "ground_phase_share": .25})
        occupancy = {side: weights for side in Side}
        targets = {Side.RED: (8.2, .57), Side.BLUE: (5.1, .34)}
        mapped = map_matchup_strikes(original, targets, occupancy)
        self.assertEqual(mapped.simulator, original.simulator)
        self.assertEqual(mapped.bout, original.bout)
        self.assertEqual(mapped.root_seed, original.root_seed)
        strike_fields = {"strike_accuracy", "strike_defense", "strike_rate_distance", "strike_rate_clinch", "strike_rate_ground"}
        for side in Side:
            pace, landed = integrate(neutral_phase_rates(mapped, side), occupancy[side])
            np.testing.assert_allclose([pace, landed/pace], targets[side], rtol=1e-10)
            before = asdict(getattr(original, side.value).parameters)
            after = asdict(getattr(mapped, side.value).parameters)
            self.assertEqual({key: value for key, value in before.items() if key not in strike_fields},
                             {key: value for key, value in after.items() if key not in strike_fields})
            self.assertAlmostEqual(after["strike_rate_distance"]/after["strike_rate_ground"],
                                   before["strike_rate_distance"]/before["strike_rate_ground"])

    def test_unreachable_engine_targets_are_rejected(self):
        occupancy = {side: {"distance": 1.0} for side in Side}
        for target in ((7, .01), (7, .99), (1000, .5)):
            with self.assertRaises(ValueError):
                map_matchup_strikes(spec(), {side: target for side in Side}, occupancy)

    def test_mapping_is_symmetric_under_fighter_swap(self):
        original = spec()
        swapped = replace(original, red=replace(original.blue, fighter_id="red"), blue=replace(original.red, fighter_id="blue"))
        occupancy = {side: {"distance": .7, "clinch": .1, "ground_top": .1, "ground_bottom": .1} for side in Side}
        targets = {Side.RED: (8.2, .57), Side.BLUE: (5.1, .34)}
        first = map_matchup_strikes(original, targets, occupancy)
        second = map_matchup_strikes(swapped, {side: targets[side.opponent] for side in Side}, occupancy)
        self.assertEqual(first.red.parameters, second.blue.parameters)
        self.assertEqual(first.blue.parameters, second.red.parameters)


if __name__ == "__main__":
    unittest.main()
