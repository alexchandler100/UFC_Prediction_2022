from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fight_sim.parameters import canonical_sha256  # noqa: E402
from fight_sim.research import _frozen_cohort_selection  # noqa: E402


class FrozenSimulationCohortTests(unittest.TestCase):
    def test_fight_identity_checksum_prevents_a_moving_cohort(self):
        physical = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2024-01-01", tz="UTC"),
                    "event_id": "event-a",
                    "fight_id": "fight-a",
                    "red_prior_ufc_fights": 3,
                    "blue_prior_ufc_fights": 4,
                },
                {
                    "date": pd.Timestamp("2024-01-01", tz="UTC"),
                    "event_id": "event-a",
                    "fight_id": "excluded-low-exposure",
                    "red_prior_ufc_fights": 2,
                    "blue_prior_ufc_fights": 4,
                },
            ]
        )
        source_hashes = {"raw": "raw", "profiles": None, "round_stats": None}
        manifest = {
            "schema_version": 1,
            "selection_contract": {
                "min_prior_ufc_fights": 3,
                "source_sha256": source_hashes,
            },
            "cohorts": {
                "development": {
                    "eligible_fights": 1,
                    "fight_ids_sha256": canonical_sha256(["fight-a"]),
                    "events": [{"date": "2024-01-01", "event_id": "event-a"}],
                }
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cohorts.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            selected, events, counts, metadata = _frozen_cohort_selection(
                physical,
                manifest_path=path,
                cohort_name="development",
                min_prior_ufc_fights=3,
                source_sha256=source_hashes,
            )
            self.assertEqual(selected["fight_id"].tolist(), ["fight-a"])
            self.assertEqual(events[0]["excluded_low_exposure"], 1)
            self.assertEqual(counts["eligible_fights"], 1)
            self.assertEqual(metadata["fight_ids_sha256"], canonical_sha256(["fight-a"]))

            changed = pd.concat(
                [
                    physical,
                    pd.DataFrame(
                        [
                            {
                                "date": pd.Timestamp("2024-01-01", tz="UTC"),
                                "event_id": "event-a",
                                "fight_id": "new-fight",
                                "red_prior_ufc_fights": 3,
                                "blue_prior_ufc_fights": 3,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            with self.assertRaisesRegex(ValueError, "fight identities changed"):
                _frozen_cohort_selection(
                    changed,
                    manifest_path=path,
                    cohort_name="development",
                    min_prior_ufc_fights=3,
                    source_sha256=source_hashes,
                )


if __name__ == "__main__":
    unittest.main()
