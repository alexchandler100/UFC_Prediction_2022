from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fight_sim.evaluation import (  # noqa: E402
    BacktestConfig,
    evaluate_simulation_ledger,
    event_card_paired_interval,
    load_backtest_report,
    run_chronological_backtest,
    write_backtest_report,
)
from fight_sim.parameters import (  # noqa: E402
    CausalParameterFitter,
    ParameterFitConfig,
    canonical_sha256,
    load_parameter_artifact,
    save_parameter_artifact,
)
from fight_sim.publication import (  # noqa: E402
    build_shadow_forecast_publication,
    compact_shadow_aggregate,
    validate_shadow_forecast_publication,
    write_shadow_forecast_publication,
)
from fight_sim.research import (  # noqa: E402
    BASELINE_WARNINGS_ATTR,
    _borderline_joint_comparisons,
    _compact_evaluation_forecast,
    _recent_complete_event_selection,
    attach_chronological_model_baselines,
    attach_timestamped_market_baselines,
    causal_joint_baseline_forecasts,
    execute_posterior_backtest,
    physical_backtest_frame,
)
from market_tracker import (  # noqa: E402
    ForecastCapture,
    QuoteSnapshot,
    TotalRoundsQuoteSnapshot,
    matchup_id_for,
)


def _side(
    date: str,
    event: str,
    fight: str,
    fighter: str,
    opponent: str,
    result: str,
    *,
    method: str,
    sig_landed: int,
    sig_attempted: int,
    td_landed: int = 0,
    td_attempted: int = 0,
    sub_attempts: int = 0,
    knockdowns: int = 0,
    control: int = 0,
) -> dict[str, object]:
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
        "method": method,
        "round": 3,
        "total_fight_time": 900,
        "knockdowns": knockdowns,
        "sig_strikes_landed": sig_landed,
        "sig_strikes_attempts": sig_attempted,
        "takedowns_landed": td_landed,
        "takedowns_attempts": td_attempted,
        "sub_attempts": sub_attempts,
        "reversals": 0,
        "control": control,
        "distance_strikes_attempts": int(sig_attempted * 0.75),
        "clinch_strikes_attempts": int(sig_attempted * 0.15),
        "ground_strikes_attempts": sig_attempted
        - int(sig_attempted * 0.75)
        - int(sig_attempted * 0.15),
    }


def _raw() -> pd.DataFrame:
    rows = []
    fights = (
        ("2020-01-01", "event-1", "fight-1", "a", "b", "KO/TKO", 80, 120, 45, 100),
        ("2021-01-01", "event-2", "fight-2", "a", "c", "U-DEC", 100, 170, 90, 180),
        ("2022-01-01", "event-3", "fight-3", "b", "c", "SUB", 55, 100, 40, 90),
    )
    for date, event, fight, winner, loser, method, wl, wa, ll, la in fights:
        rows.append(
            _side(
                date,
                event,
                fight,
                winner,
                loser,
                "W",
                method=method,
                sig_landed=wl,
                sig_attempted=wa,
                td_landed=2,
                td_attempted=5,
                sub_attempts=int(method == "SUB"),
                knockdowns=int(method == "KO/TKO"),
                control=120,
            )
        )
        rows.append(
            _side(
                date,
                event,
                fight,
                loser,
                winner,
                "L",
                method=method,
                sig_landed=ll,
                sig_attempted=la,
                td_landed=1,
                td_attempted=4,
                sub_attempts=0,
                control=40,
            )
        )
    return pd.DataFrame(rows)


def _profiles() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"url": f"http://ufcstats.com/fighter-details/{fighter}", "name": fighter.upper(), "dob": dob}
            for fighter, dob in (("a", "1990-01-01"), ("b", "1991-01-01"), ("c", "1992-01-01"))
        ]
    )


def _forecast(matchup_id: str, red_count: int, blue_count: int) -> dict[str, object]:
    total = red_count + blue_count
    return {
        "schema_version": "fight-sim.v1",
        "matchup_id": matchup_id,
        "scheduled_rounds": 3,
        "total_paths": total,
        "bootstrap_members": 2,
        "outcome_counts": {
            "red_decision": red_count,
            "blue_decision": blue_count,
        },
        "outcome_probabilities": {
            "red_decision": red_count / total,
            "blue_decision": blue_count / total,
        },
        "bootstrap_outcome_counts": [],
        "duration_bins": [{"upper_seconds": 900, "count": total}],
        "method_round_counts": [],
        "total_lines": [
            {
                "half_rounds": 2.5,
                "threshold_seconds": 750,
                "over": total,
                "under": 0,
                "push": 0,
                "no_action": 0,
            }
        ],
        "statistic_summaries": [],
        "uncertainty": [
            {
                "metric": "red_win_probability",
                "estimate": red_count / total,
                "process_mcse": 0.01,
                "parameter_p025": 0.2,
                "parameter_median": red_count / total,
                "parameter_p975": 0.8,
                "conditional_probabilities": {},
            }
        ],
        "survival": [
            {"seconds": 0.0, "probability": 1.0},
            {"seconds": 900.0, "probability": 0.0},
        ],
    }


class FightSimulationResearchTests(unittest.TestCase):
    def test_posterior_backtest_resumes_per_fight_and_reuses_shared_fit_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "raw.csv"
            profiles_path = root / "profiles.csv"
            missing_rounds = root / "rounds-missing.csv"
            cache = root / "fit-cache"
            _raw().to_csv(raw_path, index=False)
            _profiles().to_csv(profiles_path, index=False)

            common = {
                "raw_path": raw_path,
                "profiles_path": profiles_path,
                "round_path": missing_rounds,
                "last_events": 1,
                "min_prior_ufc_fights": 0,
                "bootstrap_members": 1,
                "paths_per_matchup": 1,
                "seed_repeats": 1,
                "min_training_fights": 1,
                "workers": 1,
                "chunk_size": 1,
                "fit_cache_dir": cache,
                "fidelity": "screen",
            }
            first_dir = root / "first"
            _, first = execute_posterior_backtest(
                output_dir=first_dir,
                **common,
            )
            self.assertEqual(first["runtime"]["fit_cache_misses"], 1)
            self.assertEqual(
                first["runtime"]["computed_fight_seed_pairs_this_invocation"],
                1,
            )

            _, resumed = execute_posterior_backtest(
                output_dir=first_dir,
                resume=True,
                **common,
            )
            self.assertEqual(resumed["runtime"]["resumed_fight_seed_pairs"], 1)
            self.assertEqual(
                resumed["runtime"]["computed_fight_seed_pairs_this_invocation"],
                0,
            )
            self.assertEqual(resumed["runtime"]["simulation_seconds"], 0.0)

            _, cached = execute_posterior_backtest(
                output_dir=root / "second",
                **common,
            )
            self.assertEqual(cached["runtime"]["fit_cache_hits"], 1)
            self.assertEqual(cached["runtime"]["fit_cache_misses"], 0)
            self.assertEqual(
                cached["aggregate"], first["aggregate"]
            )

    def test_recent_event_selection_keeps_whole_cards_then_excludes_low_exposure(self):
        physical = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp(date, tz="UTC"),
                    "event_id": event,
                    "fight_id": fight,
                    "red_prior_ufc_fights": red,
                    "blue_prior_ufc_fights": blue,
                }
                for date, event, fight, red, blue in (
                    ("2026-01-01", "old", "old-1", 10, 10),
                    ("2026-02-01", "middle", "middle-1", 3, 3),
                    ("2026-02-01", "middle", "middle-2", 2, 9),
                    ("2026-03-01", "new", "new-1", 5, 4),
                    ("2026-03-01", "new", "new-2", 0, 0),
                )
            ]
        )

        selected, manifest, counts = _recent_complete_event_selection(
            physical, last_events=2, min_prior_ufc_fights=3
        )

        self.assertEqual(selected["fight_id"].tolist(), ["middle-1", "new-1"])
        self.assertEqual([item["event_id"] for item in manifest], ["middle", "new"])
        self.assertEqual(
            [(item["card_fights"], item["eligible_fights"]) for item in manifest],
            [(2, 1), (2, 1)],
        )
        self.assertEqual(
            counts,
            {
                "selected_card_fights": 4,
                "eligible_fights": 2,
                "excluded_low_exposure": 2,
            },
        )
        reserved, reserved_manifest, _ = _recent_complete_event_selection(
            physical,
            last_events=1,
            min_prior_ufc_fights=3,
            skip_latest_events=1,
        )
        self.assertEqual(reserved["fight_id"].tolist(), ["middle-1"])
        self.assertEqual(
            [item["event_id"] for item in reserved_manifest], ["middle"]
        )

    def test_experience_band_uses_only_strictly_earlier_fights(self):
        baseline = physical_backtest_frame(_raw()).set_index("fight_id")
        self.assertEqual(baseline.loc["fight-1", "red_prior_ufc_fights"], 0)
        self.assertEqual(baseline.loc["fight-1", "blue_prior_ufc_fights"], 0)
        self.assertEqual(
            baseline.loc["fight-1", "experience_band"], "debutant_in_matchup"
        )
        self.assertEqual(
            baseline.loc["fight-2", "experience_band"], "debutant_in_matchup"
        )
        self.assertEqual(
            baseline.loc["fight-3", "experience_band"], "both_1_to_2_prior"
        )

        def predict(train, test, cutoff):
            del train, cutoff
            return pd.DataFrame(
                [
                    {
                        "fight_id": row["fight_id"],
                        "forecast": {
                            "total_paths": 100,
                            "outcome_counts": {row["actual_outcome"]: 100},
                            "outcome_probabilities": {
                                row["actual_outcome"]: 1.0
                            },
                        },
                    }
                    for row in test.to_dict("records")
                ]
            )

        _, report = run_chronological_backtest(
            baseline.reset_index(),
            predict,
            config=BacktestConfig(
                first_test_year=2022,
                last_test_year=2022,
                min_training_fights=2,
                card_bootstrap_replicates=10,
            ),
        )
        self.assertIn("experience_band", report.slices)
        self.assertIn("both_1_to_2_prior", report.slices["experience_band"])

        future = pd.DataFrame(
            [
                _side(
                    "2023-01-01",
                    "event-4",
                    "fight-4",
                    "a",
                    "b",
                    "W",
                    method="U-DEC",
                    sig_landed=80,
                    sig_attempted=120,
                ),
                _side(
                    "2023-01-01",
                    "event-4",
                    "fight-4",
                    "b",
                    "a",
                    "L",
                    method="U-DEC",
                    sig_landed=70,
                    sig_attempted=115,
                ),
            ]
        )
        appended = physical_backtest_frame(
            pd.concat([_raw(), future], ignore_index=True)
        ).set_index("fight_id")
        columns = [
            "red_prior_ufc_fights",
            "blue_prior_ufc_fights",
            "experience_band",
        ]
        pd.testing.assert_frame_equal(
            baseline.loc[["fight-1", "fight-2", "fight-3"], columns],
            appended.loc[["fight-1", "fight-2", "fight-3"], columns],
        )

    def test_borderline_precision_includes_competing_risk_joint(self):
        report = SimpleNamespace(
            comparisons={
                "competing_risk_joint": {
                    "paired_event_card_interval": {
                        "interval_p025": -0.01,
                        "interval_p975": 0.02,
                    }
                },
                "population_joint": {
                    "paired_event_card_interval": {
                        "interval_p025": -0.03,
                        "interval_p975": -0.01,
                    }
                },
            }
        )
        self.assertEqual(
            _borderline_joint_comparisons(report),
            ("competing_risk_joint",),
        )

    def test_parameter_fit_is_strictly_causal_and_round_trips(self):
        config = ParameterFitConfig(bootstrap_members=3, random_seed=7)
        fitter = CausalParameterFitter(_raw(), _profiles())
        artifact = fitter.fit(
            "2022-01-01",
            config=config,
            created_at_utc="2022-01-02T00:00:00Z",
        )
        changed = _raw()
        changed.loc[changed["date"].eq("2022-01-01"), "sig_strikes_attempts"] = 100000
        changed_artifact = CausalParameterFitter(changed, _profiles()).fit(
            "2022-01-01",
            config=config,
            created_at_utc="2022-01-02T00:00:00Z",
        )
        self.assertEqual(artifact.artifact_sha256, changed_artifact.artifact_sha256)
        self.assertEqual(artifact.observed_fights, 2)
        snapshot = fitter.snapshot_for(
            artifact, "a", division="Lightweight", member_index=0
        )
        self.assertEqual(snapshot.fighter_id, "a")
        self.assertEqual(snapshot.experience_fights, 2)
        self.assertEqual(snapshot.source_hash, artifact.artifact_sha256)
        self.assertGreater(snapshot.parameters.strike_rate_distance, 0)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parameters.json.gz"
            save_parameter_artifact(path, artifact)
            loaded = load_parameter_artifact(path)
        self.assertEqual(loaded.to_dict(), artifact.to_dict())

    def test_snapshot_refuses_cutoff_mismatch(self):
        fitter = CausalParameterFitter(_raw(), _profiles())
        artifact = fitter.fit(
            "2022-01-01",
            config=ParameterFitConfig(bootstrap_members=1),
            created_at_utc="2022-01-02T00:00:00Z",
        )
        with self.assertRaisesRegex(ValueError, "exactly match"):
            fitter.snapshot_for(
                artifact,
                "a",
                division="Lightweight",
                member_index=0,
                as_of="2021-01-01",
            )

    def test_evaluation_scores_coherent_ledger_and_card_paired_difference(self):
        rows = []
        for index, (actual, red) in enumerate(
            (("red_decision", 80), ("blue_decision", 20), ("red_decision", 70), ("blue_decision", 30))
        ):
            rows.append(
                {
                    "event_id": f"event-{index // 2}",
                    "fight_id": f"fight-{index}",
                    "date": f"2023-01-{index + 1:02d}",
                    "actual_outcome": actual,
                    "actual_duration_seconds": 900,
                    "forecast": _forecast(f"matchup-{index}", red, 100 - red),
                    "baseline": 0.5,
                }
            )
        ledger = pd.DataFrame(rows)
        metrics = evaluate_simulation_ledger(ledger)
        self.assertEqual(metrics["n_fights"], 4)
        self.assertAlmostEqual(metrics["duration_crps_seconds"], 0.0)
        self.assertLess(metrics["winner"]["log_loss"], 0.5)
        ledger["challenger"] = [0.8, 0.2, 0.7, 0.3]
        interval = event_card_paired_interval(
            ledger, "challenger", "baseline", replicates=100, random_seed=4
        )
        self.assertLess(interval["challenger_minus_baseline_log_loss"], 0)

    def test_chronological_runner_never_exposes_test_rows_as_training(self):
        rows = []
        for year in range(2019, 2023):
            for item in range(2):
                fight_id = f"{year}-{item}"
                rows.append(
                    {
                        "date": f"{year}-06-{item + 1:02d}",
                        "event_id": f"event-{year}",
                        "fight_id": fight_id,
                        "actual_outcome": "red_decision" if item == 0 else "blue_decision",
                        "actual_duration_seconds": 900,
                    }
                )
        seen = []

        def predict(train, test, cutoff):
            self.assertTrue((train["date"] < cutoff).all())
            self.assertTrue((test["date"] >= cutoff).all())
            seen.append(cutoff.year)
            return pd.DataFrame(
                [
                    {"fight_id": row.fight_id, "forecast": _forecast(row.fight_id, 50, 50)}
                    for row in test.itertuples()
                ]
            )

        ledger, report = run_chronological_backtest(
            pd.DataFrame(rows),
            predict,
            config=BacktestConfig(
                first_test_year=2020,
                min_training_fights=2,
                card_bootstrap_replicates=20,
            ),
        )
        self.assertEqual(seen, [2020, 2021, 2022])
        self.assertEqual(len(ledger), 6)
        self.assertTrue(report.candidate_only)
        self.assertFalse(report.production_enabled)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.json"
            write_backtest_report(path, report)
            loaded = load_backtest_report(path)
        self.assertEqual(loaded.to_dict(), report.to_dict())

    def test_bounded_test_filter_preserves_complete_prior_training_history(self):
        rows = []
        for year in (2019, 2020, 2021):
            for item in range(3):
                rows.append(
                    {
                        "date": f"{year}-06-{item + 1:02d}",
                        "event_id": f"event-{year}",
                        "fight_id": f"{year}-{item}",
                        "actual_outcome": "red_decision",
                        "actual_duration_seconds": 900,
                        "selected": item == 0,
                    }
                )
        training_sizes = []

        def predict(train, test, _cutoff):
            training_sizes.append(len(train))
            self.assertEqual(len(test), 1)
            return pd.DataFrame(
                [
                    {
                        "fight_id": row.fight_id,
                        "forecast": _forecast(row.fight_id, 50, 50),
                    }
                    for row in test.itertuples()
                ]
            )

        ledger, report = run_chronological_backtest(
            pd.DataFrame(rows),
            predict,
            config=BacktestConfig(
                first_test_year=2020,
                last_test_year=2021,
                min_training_fights=1,
                card_bootstrap_replicates=20,
            ),
            test_filter_column="selected",
        )
        self.assertEqual(training_sizes, [3, 6])
        self.assertEqual(len(ledger), 2)
        self.assertEqual([fold["training_fights"] for fold in report.folds], [3, 6])
        self.assertEqual([fold["test_fights"] for fold in report.folds], [1, 1])

    def test_timestamped_baselines_validate_schema_alignment_and_pre_event_timing(self):
        event_id = "event-2020"
        fight_id = "fight-2020"
        matchup_id = matchup_id_for(event_id, "a", "b")
        physical = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2020-06-01", tz="UTC"),
                    "event_id": event_id,
                    "fight_id": fight_id,
                    "red_fighter_id": "a",
                    "blue_fighter_id": "b",
                    "division": "Lightweight",
                    "scheduled_rounds": 3,
                }
            ]
        )
        forecast = ForecastCapture.create(
            capture_id="capture-forecast",
            matchup_id=matchup_id,
            fight_id=fight_id,
            event_id=event_id,
            fighter_id="a",
            opponent_id="b",
            event_date="2020-06-01",
            timing_precision="date",
            event_start_utc=None,
            forecast_issued_at_utc="2020-05-30T12:00:00Z",
            model_probability=0.7,
            model_id="incumbent",
            model_version="test-v1",
            model_trained_through="2020-05-01",
            model_training_cutoff_precision="date",
            source_commit_sha="a" * 40,
        )
        quotes = [
            QuoteSnapshot.create(
                capture_id="capture-market",
                matchup_id=matchup_id,
                fight_id=fight_id,
                event_id=event_id,
                fighter_id="a",
                opponent_id="b",
                event_date="2020-06-01",
                timing_precision="date",
                event_start_utc=None,
                observed_at_utc="2020-05-30T13:00:00Z",
                source="fixture",
                book=f"Book {index}",
                fighter_moneyline=line,
                opponent_moneyline=opponent_line,
                source_payload_sha256="b" * 64,
            )
            for index, (line, opponent_line) in enumerate(
                ((-150, 130), (-140, 120), (-160, 135))
            )
        ]
        totals = [
            TotalRoundsQuoteSnapshot.create(
                capture_id="capture-total",
                matchup_id=matchup_id,
                fight_id=fight_id,
                event_id=event_id,
                fighter_id="a",
                opponent_id="b",
                event_date="2020-06-01",
                timing_precision="date",
                event_start_utc=None,
                observed_at_utc="2020-05-30T14:00:00Z",
                source="fixture",
                source_event_id="source-event",
                source_book_key=f"book-{index}",
                source_quote_updated_at_utc="2020-05-30T13:59:00Z",
                source_commence_time_utc="2020-06-01T12:00:00Z",
                book=f"Book {index}",
                line=2.5,
                over_moneyline=line,
                under_moneyline=under_line,
                source_payload_sha256=f"{index + 11:064x}",
            )
            for index, (line, under_line) in enumerate(((-110, -110), (-120, 100)))
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            forecast_rows = [forecast.to_mapping()]
            post_event = dict(forecast.to_mapping())
            post_event["forecast_issued_at_utc"] = "2020-06-02T00:00:00Z"
            forecast_rows.append(post_event)
            pd.DataFrame(forecast_rows, columns=ForecastCapture.FIELDNAMES).to_csv(
                root / "forecast_captures.csv", index=False
            )
            pd.DataFrame(
                [item.to_mapping() for item in quotes],
                columns=QuoteSnapshot.FIELDNAMES,
            ).to_csv(root / "quote_snapshots.csv", index=False)
            pd.DataFrame(
                [item.to_mapping() for item in totals],
                columns=TotalRoundsQuoteSnapshot.FIELDNAMES,
            ).to_csv(root / "total_round_quote_snapshots.csv", index=False)
            attached = attach_timestamped_market_baselines(physical, root)

        self.assertAlmostEqual(attached.loc[0, "production_red_win_probability"], 0.7)
        self.assertIn(
            "market_red_win_probability", attached.columns, str(attached.attrs)
        )
        self.assertAlmostEqual(
            attached.loc[0, "market_red_win_probability"],
            sum(item.no_vig_fighter_probability for item in quotes) / len(quotes),
        )
        self.assertEqual(attached.loc[0, "market_total_line_rounds"], 2.5)
        self.assertAlmostEqual(
            attached.loc[0, "market_total_over_probability"],
            sum(sorted(item.no_vig_over_probability for item in totals)) / 2.0,
        )
        self.assertIn(
            "forecast_captures_invalid_rows:1",
            attached.attrs[BASELINE_WARNINGS_ATTR],
        )

    def test_optional_baseline_files_with_no_supported_coverage_add_no_score_column(self):
        physical = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2020-06-01", tz="UTC"),
                    "event_id": "event",
                    "fight_id": "fight",
                    "red_fighter_id": "a",
                    "blue_fighter_id": "b",
                    "division": "Lightweight",
                    "scheduled_rounds": 3,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "forecast_captures.csv",
                "quote_snapshots.csv",
                "total_round_quote_snapshots.csv",
            ):
                (root / name).write_text("unsupported,value\n1,2\n", encoding="utf-8")
            attached = attach_timestamped_market_baselines(physical, root)
        self.assertNotIn("production_red_win_probability", attached)
        self.assertNotIn("market_red_win_probability", attached)
        self.assertNotIn("market_total_over_probability", attached)
        warnings = attached.attrs[BASELINE_WARNINGS_ATTR]
        self.assertIn("forecast_captures_unsupported_schema", warnings)
        self.assertIn("moneyline_quotes_unsupported_schema", warnings)
        self.assertIn("total_round_quotes_unsupported_schema", warnings)

    def test_chronological_model_failures_are_isolated_by_fold(self):
        physical = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2020-06-01", tz="UTC"),
                    "event_id": "event-2020",
                    "fight_id": "test-2020",
                    "red_fighter_id": "a",
                    "blue_fighter_id": "b",
                    "division": "Lightweight",
                    "scheduled_rounds": 3,
                    "_backtest_selected": True,
                },
                {
                    "date": pd.Timestamp("2021-06-01", tz="UTC"),
                    "event_id": "event-2021",
                    "fight_id": "test-2021",
                    "red_fighter_id": "a",
                    "blue_fighter_id": "b",
                    "division": "Lightweight",
                    "scheduled_rounds": 3,
                    "_backtest_selected": True,
                },
            ]
        )
        historical = []
        dates = pd.date_range("2010-01-01", periods=1000, freq="D")
        for index, date in enumerate(dates):
            historical.append(
                {
                    "date": date,
                    "event_id": f"history-event-{index}",
                    "fight_id": f"history-{index}",
                    "fighter_id": f"h-{index}-a",
                    "opponent_id": f"h-{index}-b",
                    "target": index % 2,
                    "bout_order": 0,
                    "label_method": "U-DEC",
                    "x_diff": 0.0,
                }
            )
        tests = [
            {
                "date": "2020-06-01",
                "event_id": "event-2020",
                "fight_id": "test-2020",
                "fighter_id": "a",
                "opponent_id": "b",
                "target": 1,
                "bout_order": 0,
                "label_method": "U-DEC",
                "x_diff": 1.0,
            },
            {
                "date": "2021-06-01",
                "event_id": "event-2021",
                "fight_id": "test-2021",
                "fighter_id": "a",
                "opponent_id": "b",
                "target": 0,
                "bout_order": 0,
                "label_method": "SUB",
                "x_diff": -1.0,
            },
        ]

        class FakeBuilder:
            def __init__(self, *_args, **_kwargs):
                pass

        class FakeTemporal:
            def __init__(self, point, _builder):
                self.point = point

            def walk_forward_predictions(self, years):
                year = years[0]
                if year == 2021:
                    raise RuntimeError("fixture fold failure")
                row = self.point.loc[self.point["fight_id"].eq("test-2020")].iloc[0]
                return pd.DataFrame(
                    [
                        {
                            **row.to_dict(),
                            "model_probability": 0.7,
                            "training_through": "2019-12-31",
                        }
                    ]
                )

        class FakeOutcomePrediction:
            terminal_probabilities = {
                "fighter_ko_tko": 0.1,
                "fighter_submission": 0.1,
                "fighter_decision": 0.2,
                "fighter_other": 0.0,
                "opponent_ko_tko": 0.1,
                "opponent_submission": 0.1,
                "opponent_decision": 0.4,
                "opponent_other": 0.0,
            }

        class FakeOutcomeModel:
            def predict(self, _source, _rounds):
                return FakeOutcomePrediction()

        outcome_calls = 0

        def fake_evaluate(*_args, **_kwargs):
            nonlocal outcome_calls
            outcome_calls += 1
            if outcome_calls == 1:
                raise RuntimeError("fixture outcome fold failure")
            return FakeOutcomeModel(), {}

        with tempfile.TemporaryDirectory() as directory:
            point_path = Path(directory) / "point.csv"
            pd.DataFrame([*historical, *tests]).to_csv(point_path, index=False)
            with (
                patch(
                    "fight_predictor.point_in_time.PointInTimeDatasetBuilder",
                    FakeBuilder,
                ),
                patch(
                    "fight_predictor.point_in_time.TemporalFightPredictor",
                    FakeTemporal,
                ),
                patch(
                    "fight_predictor.outcome_model.evaluate_outcome_model",
                    side_effect=fake_evaluate,
                ),
            ):
                attached = attach_chronological_model_baselines(
                    physical,
                    raw=pd.DataFrame(),
                    profiles=pd.DataFrame(),
                    point_in_time_path=point_path,
                    years=(2020, 2021),
                )
        self.assertIn(
            "production_red_win_probability", attached.columns, str(attached.attrs)
        )
        self.assertAlmostEqual(attached.loc[0, "production_red_win_probability"], 0.7)
        self.assertTrue(pd.isna(attached.loc[1, "production_red_win_probability"]))
        self.assertTrue(pd.isna(attached.loc[0, "outcome_model_red_win_probability"]))
        self.assertAlmostEqual(attached.loc[1, "outcome_model_red_win_probability"], 0.4)
        warnings = attached.attrs[BASELINE_WARNINGS_ATTR]
        self.assertIn("incumbent_fold_2021_failed:RuntimeError", warnings)
        self.assertIn("outcome_fold_2020_failed:RuntimeError", warnings)

    def test_causal_joint_baselines_are_smoothed_and_swap_symmetric(self):
        train = pd.DataFrame(
            [
                {"actual_outcome": "red_ko_tko", "division": "Lightweight"},
                {"actual_outcome": "blue_ko_tko", "division": "Lightweight"},
                {"actual_outcome": "red_decision", "division": "Welterweight"},
                {"actual_outcome": "draw", "division": "Welterweight"},
                {"actual_outcome": "no_contest", "division": "Lightweight"},
            ]
        )
        test = pd.DataFrame(
            [
                {"fight_id": "one", "division": "Lightweight"},
                {"fight_id": "two", "division": "Unseen"},
            ]
        )
        forecasts = causal_joint_baseline_forecasts(train, test)
        for column in ("population_forecast", "division_forecast"):
            for forecast in forecasts[column]:
                probabilities = forecast["outcome_probabilities"]
                self.assertAlmostEqual(sum(probabilities.values()), 1.0)
                self.assertTrue(all(value > 0.0 for value in probabilities.values()))
                for method in ("ko_tko", "submission", "decision", "other"):
                    self.assertEqual(
                        probabilities[f"red_{method}"],
                        probabilities[f"blue_{method}"],
                    )
        unseen_division = forecasts.loc[1, "division_forecast"][
            "outcome_probabilities"
        ]
        population = forecasts.loc[1, "population_forecast"]["outcome_probabilities"]
        for outcome in population:
            self.assertAlmostEqual(unseen_division[outcome], population[outcome])

    def test_shadow_publication_stays_separate_and_paper_only(self):
        fitter = CausalParameterFitter(_raw(), _profiles())
        artifact = fitter.fit(
            "2022-01-01",
            config=ParameterFitConfig(bootstrap_members=2),
            created_at_utc="2022-01-02T00:00:00Z",
        )
        event_id = "event-upcoming"
        matchup_id = matchup_id_for(event_id, "a", "b")
        publication = build_shadow_forecast_publication(
            {matchup_id: _forecast(matchup_id, 60, 40)},
            [
                {
                    "red_fighter_id": "a",
                    "blue_fighter_id": "b",
                    "red_fighter_name": "A",
                    "blue_fighter_name": "B",
                    "division": "Lightweight",
                }
            ],
            {
                "event_id": event_id,
                "event_url": "http://ufcstats.com/event-details/event-upcoming",
                "date": "2022-02-01T00:00:00Z",
                "title": "UFC Test",
            },
            artifact,
            forecast_issued_at_utc="2022-01-15T00:00:00Z",
            source_commit_sha="a" * 40,
        )
        validated = validate_shadow_forecast_publication(publication)
        self.assertTrue(validated["candidate_only"])
        self.assertTrue(validated["paper_only"])
        self.assertFalse(validated["execution_enabled"])
        self.assertEqual(validated["production_influence"], "none")
        aggregate = validated["matchups"][0]["aggregate"]
        self.assertEqual(aggregate["detail_level"], "compact_shadow_v1")
        self.assertEqual(
            aggregate["omitted_local_authority_fields"],
            ["bootstrap_statistic_distributions"],
        )

    def test_shadow_compaction_hashes_but_does_not_publish_local_member_histograms(self):
        matchup_id = "matchup-test"
        full = _forecast(matchup_id, 60, 40)
        full["statistic_distributions"] = [
            {
                "statistic": "red_significant_strikes",
                "total_paths": 100,
                "counts": [{"value": 50.0, "count": 100}],
            }
        ]
        full["bootstrap_statistic_distributions"] = [
            {
                "bootstrap_member": 0,
                "statistic": "red_significant_strikes",
                "paths": 50,
                "counts": [{"value": 50.0, "count": 50}],
            }
        ]
        compact = compact_shadow_aggregate(full)
        self.assertEqual(compact["local_aggregate_sha256"], canonical_sha256(full))
        self.assertNotIn("bootstrap_statistic_distributions", compact)
        self.assertEqual(compact["statistic_distributions"], full["statistic_distributions"])
        self.assertEqual(compact_shadow_aggregate(compact), compact)

    def test_backtest_ledger_omits_only_member_statistic_histograms(self):
        full = _forecast("matchup-test", 60, 40)
        full["bootstrap_statistic_distributions"] = [{"large": "local-only"}]
        full["statistic_distributions"] = [{"aggregate": "retained"}]
        compact = _compact_evaluation_forecast(full)
        self.assertNotIn("bootstrap_statistic_distributions", compact)
        self.assertEqual(compact["statistic_distributions"], full["statistic_distributions"])
        self.assertEqual(compact["local_aggregate_sha256"], canonical_sha256(full))

    def test_shadow_writer_withholds_oversized_publication(self):
        fitter = CausalParameterFitter(_raw(), _profiles())
        artifact = fitter.fit(
            "2022-01-01",
            config=ParameterFitConfig(bootstrap_members=1),
            created_at_utc="2022-01-02T00:00:00Z",
        )
        publication = build_shadow_forecast_publication(
            {},
            [],
            {
                "event_id": "event-upcoming",
                "event_url": "http://ufcstats.com/event-details/event-upcoming",
                "date": "2022-02-01T00:00:00Z",
                "title": "UFC Test",
            },
            artifact,
            forecast_issued_at_utc="2022-01-15T00:00:00Z",
            source_commit_sha="a" * 40,
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch("fight_sim.publication.MAX_SHADOW_PUBLICATION_BYTES", 32):
                with self.assertRaisesRegex(ValueError, "withholding"):
                    write_shadow_forecast_publication(
                        Path(directory) / "shadow.json", publication
                    )


if __name__ == "__main__":
    unittest.main()
