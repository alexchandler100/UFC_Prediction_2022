import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from external_mma import (
    ExternalBoutObservation,
    ExternalMmaStore,
    KaggleProMmaAdapter,
    build_auxiliary_doubled,
    load_approved_auxiliary,
    propose_ufcstats_crosswalk,
)
from fight_predictor import PointInTimeDatasetBuilder


def make_profiles(*fighter_ids):
    return pd.DataFrame(
        [
            {
                "name": fighter_id.upper(),
                "height": "5' 10\"",
                "reach": '70"',
                "stance": "Orthodox",
                "dob": "Jan 01, 1990",
                "url": f"https://ufcstats.test/fighter-details/{fighter_id}",
            }
            for fighter_id in fighter_ids
        ]
    )


def make_ufc_fight(fight_id, date, fighter_id, opponent_id, result="W"):
    common = {
        "date": date,
        "fight_url": f"https://ufcstats.test/fight-details/{fight_id}",
        "event_url": f"https://ufcstats.test/event-details/event-{fight_id}",
        "division": "Lightweight",
        "method": "U-DEC",
        "round": 3,
        "time": "5:00",
        "time_format": "3 Rnd (5-5)",
        "total_fight_time": 900,
        "source_card_index": 0,
        "bout_order": 0,
        "knockdowns": 0,
        "sig_strikes_landed": 20,
        "sig_strikes_attempts": 40,
        "total_strikes_landed": 25,
        "total_strikes_attempts": 50,
        "takedowns_landed": 0,
        "takedowns_attempts": 1,
        "sub_attempts": 0,
        "reversals": 0,
        "control": 0,
    }
    return [
        {
            **common,
            "fighter": fighter_id.upper(),
            "opponent": opponent_id.upper(),
            "fighter_url": f"https://ufcstats.test/fighter-details/{fighter_id}",
            "opponent_url": f"https://ufcstats.test/fighter-details/{opponent_id}",
            "result": result,
        },
        {
            **common,
            "fighter": opponent_id.upper(),
            "opponent": fighter_id.upper(),
            "fighter_url": f"https://ufcstats.test/fighter-details/{opponent_id}",
            "opponent_url": f"https://ufcstats.test/fighter-details/{fighter_id}",
            "result": "L" if result == "W" else "W",
        },
    ]


def make_observation(
    *, promotion="Bellator MMA", date="2019-01-01", result="W",
    fighter_source_id="/fighter/a-source", fighter_name="A",
    opponent_source_id="/fighter/x-source", opponent_name="X",
    bout_id="bout-1",
):
    return ExternalBoutObservation.create(
        source="fixture",
        snapshot_sha256="a" * 64,
        source_bout_id=bout_id,
        source_event_id="event-1",
        source_url="https://provider.test/event-1",
        event_date=date,
        event_name="Fixture Event",
        promotion=promotion,
        fighter_source_id=fighter_source_id,
        fighter_name=fighter_name,
        opponent_source_id=opponent_source_id,
        opponent_name=opponent_name,
        result=result,
        method="KO/TKO",
        finish_round=1,
        finish_clock_seconds=120,
    )


class ExternalMmaTests(unittest.TestCase):
    def test_observation_orientation_is_stable_and_inverts_result(self):
        observation = make_observation(
            fighter_source_id="z", opponent_source_id="a", result="W"
        )
        self.assertEqual(observation.fighter_source_id, "a")
        self.assertEqual(observation.opponent_source_id, "z")
        self.assertEqual(observation.result, "L")

    def test_kaggle_adapter_uses_participants_to_disambiguate_match_number(self):
        header = (
            "url,event_title,organisation,date,location,match_nr,fighter1_url,"
            "fighter2_url,fighter1_name,fighter2_name,fighter1_result,"
            "fighter2_result,win_method,win_details,referee,round,time\n"
        )
        rows = (
            "/events/e,Event,One Championship,Aug 6 2019,X,12,/fighter/a,"
            "/fighter/b,A,B,win,loss,TKO,Punches,Ref,1,4:11\n"
            "/events/e,Event,One Championship,Aug 6 2019,X,12,/fighter/c,"
            "/fighter/d,C,D,win,loss,Submission,Choke,Ref,0,00:00\n"
        )
        result = KaggleProMmaAdapter().convert(
            (header + rows).encode("utf-8"), "b" * 64
        )
        self.assertEqual(result.total_rows, 2)
        self.assertEqual(len(result.observations), 2)
        self.assertEqual(len({row.source_bout_id for row in result.observations}), 2)
        self.assertEqual([row.source_bout_order for row in result.observations], [12, 12])
        self.assertIsNone(result.observations[1].finish_round)

    def test_store_is_idempotent_and_source_attributed(self):
        csv_text = (
            "url,event_title,organisation,date,location,match_nr,fighter1_url,"
            "fighter2_url,fighter1_name,fighter2_name,fighter1_result,"
            "fighter2_result,win_method,win_details,referee,round,time\n"
            "/events/e,Event,Bellator MMA,Aug 6 2019,X,1,/fighter/a,"
            "/fighter/b,A,B,win,loss,Decision,Unanimous,Ref,3,5:00\n"
        ).encode("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source_registry.json").write_text(
                json.dumps(
                    {
                        "sources": [
                            {
                                "key": "kaggle_pro_mma_fights_v1",
                                "collection_status": "manual_import",
                                "license": "CC0",
                                "source_page": "https://example.test",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            store = ExternalMmaStore(root)
            first = store.import_bytes(KaggleProMmaAdapter(), csv_text)
            second = store.import_bytes(KaggleProMmaAdapter(), csv_text)
            self.assertEqual(first["added"], 1)
            self.assertEqual(second["status"], "already_imported")
            self.assertEqual(store.validate()["observations"], 1)

    def test_exact_historical_ufc_bout_builds_approved_crosswalk(self):
        raw = pd.DataFrame(make_ufc_fight("fight-1", "2020-01-01", "a", "b"))
        observation = make_observation(
            promotion="Ultimate Fighting Championship (UFC)",
            date="2020-01-01",
            fighter_source_id="/fighter/a-source",
            fighter_name="A",
            opponent_source_id="/fighter/b-source",
            opponent_name="B",
        )
        approved, review = propose_ufcstats_crosswalk([observation], raw)
        self.assertEqual(len(approved), 2)
        self.assertTrue(approved["status"].eq("approved").all())
        self.assertEqual(set(approved["canonical_fighter_id"]), {"a", "b"})
        self.assertTrue(review.empty)

    def test_external_history_updates_state_but_never_adds_training_label(self):
        raw = pd.DataFrame(make_ufc_fight("fight-1", "2020-01-01", "a", "c"))
        identities = {("fixture", "/fighter/a-source"): "a"}
        auxiliary = build_auxiliary_doubled([make_observation()], identities)
        baseline = PointInTimeDatasetBuilder(raw, make_profiles("a", "c")).build()
        enriched_builder = PointInTimeDatasetBuilder(
            raw, make_profiles("a", "c"), auxiliary_fights=auxiliary
        )
        enriched = enriched_builder.build()

        self.assertEqual(len(enriched), len(baseline))
        self.assertEqual(enriched.iloc[0]["fight_id"], "fight-1")
        self.assertEqual(enriched.iloc[0]["has_history_diff"], 1.0)
        self.assertGreater(enriched.iloc[0]["career_fights_log_diff"], 0.0)
        self.assertEqual(enriched.iloc[0]["career_sig_landed_per15_diff"], 0.0)
        self.assertEqual(enriched_builder.history_count("a"), 2)

    def test_production_loader_requires_explicit_matching_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            auxiliary_path = root / "auxiliary.csv"
            policy_path = root / "policy.json"
            auxiliary = build_auxiliary_doubled([make_observation()], {})
            auxiliary.to_csv(auxiliary_path, index=False)
            policy_path.write_text(
                json.dumps(
                    {"schema_version": 1, "enable_auxiliary_replay": False}
                ),
                encoding="utf-8",
            )
            self.assertIsNone(load_approved_auxiliary(auxiliary_path, policy_path))
            policy_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "enable_auxiliary_replay": True,
                        "approved_auxiliary_sha256": "0" * 64,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "approved hash"):
                load_approved_auxiliary(auxiliary_path, policy_path)


if __name__ == "__main__":
    unittest.main()
