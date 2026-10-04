from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fight_semantics import SCHEDULE_CONTRACT_VERSION
from fight_predictor.outcome_model import DiscreteTimeOutcomeModel
from fight_predictor.outcome_publication import build_outcome_forecast_publication
from fight_predictor.joint_grid import build_joint_grid, validate_joint_grid, archive_joint_publication, validate_joint_archive


class JointGridTests(unittest.TestCase):
    def setUp(self):
        self.model = DiscreteTimeOutcomeModel(["skill_diff"])
        self.model.schedule_contract_version = SCHEDULE_CONTRACT_VERSION
        self.model.pipeline = SimpleNamespace(
            named_steps={"model": SimpleNamespace(classes_=["continue", "fighter_ko_tko", "opponent_decision"])},
            predict_proba=lambda rows: np.tile([.5, .3, .2], (len(rows), 1)))
        self.prediction = self.model.predict({"skill_diff": 0}, 3)
        self.totals = {str(line): self.prediction.probability_over_seconds(int(line * 300)) for line in [.5, 1.5, 2.5]}

    def publication(self):
        builder = SimpleNamespace(matchup_features=lambda *args: pd.DataFrame([{"skill_diff": 0}]))
        return build_outcome_forecast_publication(self.model, builder,
            pd.DataFrame([{"fighter id": "a", "opponent id": "b", "fighter name": "A", "opponent name": "B", "division": "Lightweight"}]),
            {"event_id": "event", "event_url": "http://ufcstats.com/event-details/event", "date": "2026-10-10", "title": "UFC Test"},
            selected_c=.1, training_input_sha256="a" * 64, model_trained_through="2026-09-26",
            forecast_issued_at_utc="2026-10-04T12:00:00Z", source_commit_sha="b" * 40)

    def test_joint_grid_recovers_existing_marginals_without_changing_them(self):
        grid = build_joint_grid(self.prediction)
        validate_joint_grid(grid, self.prediction.terminal_probabilities, self.totals, 3)
        self.assertAlmostEqual(self.prediction.terminal_probabilities["fighter_ko_tko"], .590625)
        self.assertAlmostEqual(self.prediction.terminal_probabilities["opponent_decision"], .409375)
        self.assertAlmostEqual(self.totals["1.5"], .125)
        self.assertAlmostEqual(sum(row["probability"] for row in grid["scenarios"]), 1)
        self.assertGreater(grid["decision_mass_before_scheduled_end"], 0)
        self.assertFalse(grid["settlement_ready"])

    def test_marginal_disagreement_and_unjustified_promotion_are_rejected(self):
        grid = build_joint_grid(self.prediction)
        for edit in [lambda g: g.update(settlement_ready=True),
                     lambda g: g["scenarios"][0].update(probability=.01),
                     lambda g: g.update(decision_mass_before_scheduled_end=0),
                     lambda g: g["scenarios"].append(g["scenarios"][0]),
                     lambda g: g["scenarios"][0].update(start_seconds=-1)]:
            changed = deepcopy(grid)
            edit(changed)
            with self.assertRaises(ValueError):
                validate_joint_grid(changed, self.prediction.terminal_probabilities, self.totals, 3)

    def test_legacy_marginal_only_predictions_do_not_gain_invented_joint_rows(self):
        self.assertIsNone(build_joint_grid(SimpleNamespace()))

    def test_archive_is_prospective_deduplicated_and_validated(self):
        publication = self.publication()
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(archive_joint_publication(directory, publication, "2026-10-04T13:00:00Z"))
            path = next(Path(directory).glob("*.json"))
            before = path.read_bytes()
            self.assertFalse(archive_joint_publication(directory, publication, "2026-10-05T13:00:00Z"))
            self.assertEqual(before, path.read_bytes())
            self.assertEqual(validate_joint_archive(directory), 1)
            self.assertFalse(archive_joint_publication(directory, publication, "2026-10-10T00:00:00Z"))
            with self.assertRaises(ValueError):
                archive_joint_publication(directory, publication, "2026-10-04T10:00:00Z")
            with self.assertRaises(ValueError):
                archive_joint_publication(directory, publication, "2026-10-04T13:00:00")

    def test_archive_rejects_tampering(self):
        publication = self.publication()
        with tempfile.TemporaryDirectory() as directory:
            archive_joint_publication(directory, publication, "2026-10-04T13:00:00Z")
            path = next(Path(directory).glob("*.json"))
            path.write_text(path.read_text().replace('"settlement_ready": false', '"settlement_ready": true'))
            with self.assertRaises(ValueError):
                validate_joint_archive(directory)


if __name__ == "__main__":
    unittest.main()
