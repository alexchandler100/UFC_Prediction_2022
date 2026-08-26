from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fight_sim.domain import FighterParameters  # noqa: E402
from fight_sim.parameters import (  # noqa: E402
    CausalParameterFitter,
    ParameterFitConfig,
)


def _side(
    *,
    date: str,
    event: str,
    fight: str,
    fighter: str,
    opponent: str,
    result: str,
    head: int,
    body: int,
    leg: int,
    reversals: int,
    takedowns_landed: int = 10,
) -> dict[str, object]:
    landed = head + body + leg
    attempted = landed + 30
    return {
        "date": date,
        "event_url": f"http://ufcstats.com/event-details/{event}",
        "fight_url": f"http://ufcstats.com/fight-details/{fight}",
        "fighter_url": f"http://ufcstats.com/fighter-details/{fighter}",
        "opponent_url": f"http://ufcstats.com/fighter-details/{opponent}",
        "fighter": fighter.title(),
        "opponent": opponent.title(),
        "division": "Lightweight",
        "result": result,
        "method": "U-DEC",
        "round": 3,
        "total_fight_time": 900,
        "knockdowns": 0,
        "sig_strikes_landed": landed,
        "sig_strikes_attempts": attempted,
        "head_strikes_landed": head,
        "body_strikes_landed": body,
        "leg_strikes_landed": leg,
        "takedowns_landed": takedowns_landed,
        "takedowns_attempts": takedowns_landed + 2,
        "sub_attempts": 0,
        "reversals": reversals,
        "control": 120,
        "distance_strikes_attempts": attempted - 20,
        "clinch_strikes_attempts": 10,
        "ground_strikes_attempts": 10,
    }


def _bout(
    *,
    date: str,
    event: str,
    fight: str,
    first: str = "head",
    second: str = "leg",
    first_targets: tuple[int, int, int] = (80, 5, 5),
    second_targets: tuple[int, int, int] = (5, 5, 80),
    first_reversals: int = 8,
    second_reversals: int = 0,
) -> list[dict[str, object]]:
    return [
        _side(
            date=date,
            event=event,
            fight=fight,
            fighter=first,
            opponent=second,
            result="W",
            head=first_targets[0],
            body=first_targets[1],
            leg=first_targets[2],
            reversals=first_reversals,
        ),
        _side(
            date=date,
            event=event,
            fight=fight,
            fighter=second,
            opponent=first,
            result="L",
            head=second_targets[0],
            body=second_targets[1],
            leg=second_targets[2],
            reversals=second_reversals,
        ),
    ]


def _history(count: int = 6) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(count):
        rows.extend(
            _bout(
                date=f"2020-{index + 1:02d}-01",
                event=f"event-{index}",
                fight=f"fight-{index}",
            )
        )
    return pd.DataFrame(rows)


def _profiles() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "url": f"http://ufcstats.com/fighter-details/{fighter}",
                "name": fighter.upper(),
                "dob": dob,
            }
            for fighter, dob in (
                ("head", "1990-01-01"),
                ("leg", "1991-01-01"),
            )
        ]
    )


def _config(*, members: int = 1) -> ParameterFitConfig:
    return ParameterFitConfig(
        bootstrap_members=members,
        random_seed=37,
        rate_prior_fights=2.0,
        probability_prior_attempts=20.0,
        rare_event_prior_opportunities=10.0,
        division_prior_fights=5.0,
    )


class ObservableParameterContractTests(unittest.TestCase):
    def test_target_share_validation_and_roundtrip(self):
        parameters = FighterParameters(
            head_target_share=0.70,
            body_target_share=0.20,
            leg_target_share=0.10,
            reversal_after_escape=0.31,
        )

        self.assertEqual(
            FighterParameters.from_dict(parameters.to_dict()), parameters
        )
        with self.assertRaisesRegex(ValueError, "target shares must sum to one"):
            FighterParameters(
                head_target_share=0.70,
                body_target_share=0.20,
                leg_target_share=0.20,
            )
        with self.assertRaisesRegex(ValueError, "head_target_share"):
            FighterParameters(
                head_target_share=1.01,
                body_target_share=0.0,
                leg_target_share=-0.01,
            )

    def test_fitted_target_profiles_are_coherent_and_reversals_shrink(self):
        artifact = CausalParameterFitter(_history(), _profiles()).fit(
            "2021-01-01",
            config=_config(),
            created_at_utc="2021-01-02T00:00:00Z",
        )
        member = artifact.members[0]
        head = member.fighter_parameters["head"]
        leg = member.fighter_parameters["leg"]

        for values in (head, leg):
            self.assertAlmostEqual(
                values["head_target_share"]
                + values["body_target_share"]
                + values["leg_target_share"],
                1.0,
            )
        self.assertGreater(head["head_target_share"], head["leg_target_share"])
        self.assertGreater(leg["leg_target_share"], leg["head_target_share"])
        self.assertGreater(
            head["head_target_share"], leg["head_target_share"]
        )
        self.assertGreater(leg["leg_target_share"], head["leg_target_share"])

        # Observed reversal ratios are 8/10 and 0/10. Strong pooling must
        # preserve their direction without returning either raw endpoint.
        self.assertGreater(head["reversal_after_escape"], leg["reversal_after_escape"])
        self.assertLess(head["reversal_after_escape"], 0.8)
        self.assertGreater(leg["reversal_after_escape"], 0.0)


class CausalParameterInvariantTests(unittest.TestCase):
    def test_appending_future_fights_cannot_change_earlier_artifact_or_snapshot(self):
        history = _history(4)
        future = pd.DataFrame(
            _bout(
                date="2022-01-01",
                event="future-event",
                fight="future-fight",
                first_targets=(0, 0, 90),
                second_targets=(90, 0, 0),
                first_reversals=0,
                second_reversals=10,
            )
        )
        config = _config(members=2)
        baseline_fitter = CausalParameterFitter(history, _profiles())
        appended_fitter = CausalParameterFitter(
            pd.concat([history, future], ignore_index=True), _profiles()
        )
        baseline = baseline_fitter.fit(
            "2021-01-01",
            config=config,
            created_at_utc="2021-01-02T00:00:00Z",
        )
        appended = appended_fitter.fit(
            "2021-01-01",
            config=config,
            created_at_utc="2021-01-02T00:00:00Z",
        )

        self.assertEqual(appended.to_dict(), baseline.to_dict())
        self.assertEqual(
            appended_fitter.snapshot_for(
                appended,
                "head",
                division="Lightweight",
                member_index=0,
            ).to_dict(),
            baseline_fitter.snapshot_for(
                baseline,
                "head",
                division="Lightweight",
                member_index=0,
            ).to_dict(),
        )

    def test_same_card_is_excluded_and_current_career_summaries_are_not_features(self):
        history = _history(4)
        same_card_rows = [
            *_bout(
                date="2021-01-01",
                event="cutoff-card",
                fight="cutoff-fight-1",
                first_targets=(0, 0, 90),
                second_targets=(90, 0, 0),
                first_reversals=0,
                second_reversals=10,
            ),
            *_bout(
                date="2021-01-01",
                event="cutoff-card",
                fight="cutoff-fight-2",
                first="reserve-a",
                second="reserve-b",
                first_targets=(90, 0, 0),
                second_targets=(0, 0, 90),
            ),
        ]
        config = _config()
        baseline_fitter = CausalParameterFitter(history, _profiles())
        same_card_fitter = CausalParameterFitter(
            pd.concat([history, pd.DataFrame(same_card_rows)], ignore_index=True),
            _profiles(),
        )
        baseline = baseline_fitter.fit(
            "2021-01-01",
            config=config,
            created_at_utc="2021-01-02T00:00:00Z",
        )
        same_card = same_card_fitter.fit(
            "2021-01-01",
            config=config,
            created_at_utc="2021-01-02T00:00:00Z",
        )
        self.assertEqual(same_card.to_dict(), baseline.to_dict())

        leaky = history.copy()
        leaky["career_wins"] = 999
        leaky["career_losses"] = 0
        leaky["career_fights"] = 999
        leaky["career_sig_strikes_landed_per_minute"] = 999.0
        leaky_fitter = CausalParameterFitter(leaky, _profiles())
        leaky_artifact = leaky_fitter.fit(
            "2021-01-01",
            config=config,
            created_at_utc="2021-01-02T00:00:00Z",
        )
        # Unknown present-day summaries may be retained in source provenance,
        # but they must never enter sufficient statistics or snapshots.
        self.assertEqual(leaky_artifact.members, baseline.members)
        baseline_snapshot = baseline_fitter.snapshot_for(
            baseline,
            "head",
            division="Lightweight",
            member_index=0,
        ).to_dict()
        leaky_snapshot = leaky_fitter.snapshot_for(
            leaky_artifact,
            "head",
            division="Lightweight",
            member_index=0,
        ).to_dict()
        baseline_snapshot.pop("source_hash")
        leaky_snapshot.pop("source_hash")
        self.assertEqual(leaky_snapshot, baseline_snapshot)
        self.assertEqual(leaky_snapshot["experience_fights"], 4)


if __name__ == "__main__":
    unittest.main()
