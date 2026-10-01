from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import hashlib

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fight_sim.domain import BoutConfig, FighterParameters, FighterSnapshot, Side, SimulationRunSpec, SimulatorConfig
from fight_sim.strike_bridge_audit import accuracy_sensitivity, card_interval, historical_frame, integrate, neutral_phase_rates


def spec():
    parameters = FighterParameters(strike_rate_distance=10, strike_rate_clinch=8, strike_rate_ground=6,
                                   strike_accuracy=.5, strike_defense=.5)
    red = FighterSnapshot(fighter_id="red", fighter_name="Red", as_of_utc="2024-01-01T00:00:00+00:00",
                          division="Lightweight", parameters=parameters)
    return SimulationRunSpec(bout=BoutConfig("test", "red", "blue"), red=red,
                             blue=replace(red, fighter_id="blue", fighter_name="Blue"), root_seed="test", parameter_artifact_id="bridge-test",
                             simulator=SimulatorConfig(distance_strike_hazard_multiplier=1.5,
                                                       clinch_strike_hazard_multiplier=.5,
                                                       ground_strike_hazard_multiplier=.75))


class StrikeBridgeTests(unittest.TestCase):
    def test_neutral_engine_units_and_bottom_suppression(self):
        rates = neutral_phase_rates(spec(), Side.RED)
        self.assertAlmostEqual(rates["distance"][0], 15)
        self.assertAlmostEqual(rates["distance"][1], 7.5)
        self.assertAlmostEqual(rates["clinch"][0], 4)
        self.assertAlmostEqual(rates["ground_top"][0], 4.5)
        self.assertAlmostEqual(rates["ground_bottom"][0], 4.5 * .18)
        self.assertEqual(rates["scramble"], (0, 0))
        self.assertEqual(rates, neutral_phase_rates(spec(), Side.BLUE))

    def test_opponent_log_odds_transfer_is_weakened_not_reversed(self):
        actor, opponent = accuracy_sensitivity(spec(), Side.RED,
                                               {"strike_accuracy": .5, "strike_defense": .5}, 0, 0)
        self.assertAlmostEqual(actor, 1, places=7)
        # At p=.5, dq/dv=.5*.75*.25; dlogit(q)/dv=.375, not 1.
        self.assertAlmostEqual(opponent, .375, places=7)

    def test_phase_allocation_requires_real_probability_weights(self):
        rates = neutral_phase_rates(spec(), Side.RED)
        self.assertEqual(integrate(rates, {"distance": 1}), rates["distance"])
        with self.assertRaises(ValueError):
            integrate(rates, {"distance": 2})
        with self.assertRaises(ValueError):
            integrate(rates, {"distance": 1.1, "clinch": -.1})

    def test_uncertainty_resamples_whole_cards(self):
        rows = pd.DataFrame({"event_id": ["a", "a", "b", "b"], "delta": [-1., -1., 1., 1.]})
        original = card_interval(rows, "delta")
        duplicated = card_interval(pd.concat([rows] * 10), "delta")
        self.assertEqual(original, duplicated)
        self.assertEqual(original["card_95_interval"], [-1, 1])

    def test_historical_sources_allow_line_endings_but_reject_data_changes(self):
        payload = b"fighter,value\na,12\n"
        expected = hashlib.sha256(payload.replace(b"\n", b"\r\n")).hexdigest()
        with patch("fight_sim.strike_bridge_audit.git_bytes", return_value=payload):
            self.assertEqual(historical_frame("test.csv", expected, "frozen").value.iloc[0], 12)
        with patch("fight_sim.strike_bridge_audit.git_bytes", return_value=payload.replace(b"12", b"13")):
            with self.assertRaisesRegex(ValueError, "source hash mismatch"):
                historical_frame("test.csv", expected, "frozen")


if __name__ == "__main__":
    unittest.main()
