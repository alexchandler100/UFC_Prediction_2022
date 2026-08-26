from __future__ import annotations

from dataclasses import replace
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fight_sim.aggregate import aggregate_paths  # noqa: E402
from fight_sim.domain import (  # noqa: E402
    BoutConfig,
    DecisionType,
    EventType,
    FightResult,
    FighterParameters,
    FighterSnapshot,
    FighterStats,
    OutcomeMethod,
    Phase,
    Side,
    SimulationPath,
    SimulationRunSpec,
    SimulatorConfig,
)
from fight_sim.engine import simulate_fight, simulate_indices  # noqa: E402
from fight_sim.markets import (  # noqa: E402
    Settlement,
    coherent_market_probabilities,
    settle_total,
)
from fight_sim.monte_carlo import run_nested  # noqa: E402
from fight_sim.reducer import (  # noqa: E402
    ReplayError,
    initial_state,
    reduce_event,
    reduce_events,
    state_hash,
)
from fight_sim.replay import diff_event_streams, replay_trace, stochastic_replay  # noqa: E402
from fight_sim.telemetry import (  # noqa: E402
    select_trace_paths,
    trace_digest,
    trace_from_dict,
    trace_to_dict,
)


def _snapshot(fighter_id: str, parameters: FighterParameters | None = None) -> FighterSnapshot:
    return FighterSnapshot(
        fighter_id=fighter_id,
        fighter_name=fighter_id.upper(),
        as_of_utc="2026-08-01T00:00:00+00:00",
        division="Lightweight",
        parameters=parameters or FighterParameters(),
        experience_fights=12,
        observed_fight_seconds=2_400,
        observed_rounds=8,
        data_quality="observed",
    )


def _spec(
    *,
    bootstrap_member: int = 0,
    root_seed: str = "core-test-seed",
    round_seconds: int = 30,
    red_parameters: FighterParameters | None = None,
    blue_parameters: FighterParameters | None = None,
) -> SimulationRunSpec:
    red = _snapshot("red-id", red_parameters)
    blue = _snapshot("blue-id", blue_parameters)
    return SimulationRunSpec(
        bout=BoutConfig(
            matchup_id="red-id--blue-id",
            red_fighter_id=red.fighter_id,
            blue_fighter_id=blue.fighter_id,
            scheduled_rounds=1,
            round_seconds=round_seconds,
            dynamics_seconds=5,
            division="Lightweight",
        ),
        red=red,
        blue=blue,
        root_seed=root_seed,
        parameter_artifact_id="artifact-sha256",
        bootstrap_member=bootstrap_member,
    )


def _zero_activity_parameters() -> FighterParameters:
    return FighterParameters(
        strike_rate_distance=0,
        strike_rate_clinch=0,
        strike_rate_ground=0,
        clinch_entry_rate=0,
        clinch_exit_rate=0,
        takedown_attempt_rate=0,
        ground_control_rate=0,
        escape_rate=0,
        submission_attempt_rate=0,
    )


class FightSimDomainTests(unittest.TestCase):
    def test_contract_roundtrip_and_stat_partitions(self):
        spec = _spec()
        self.assertEqual(SimulationRunSpec.from_dict(spec.to_dict()), spec)
        with self.assertRaisesRegex(ValueError, "target strike partitions"):
            FighterStats(
                strike_attempts=1,
                strikes_landed=1,
                significant_strike_attempts=1,
                significant_strikes_landed=1,
            )
        with self.assertRaisesRegex(ValueError, "distinct"):
            BoutConfig("same", "fighter", "fighter")
        with self.assertRaisesRegex(ValueError, "ko_tko_finish_probability_multiplier"):
            SimulatorConfig(ko_tko_finish_probability_multiplier=-0.01)

    def test_zero_hazard_reaches_bell_and_one_terminal_event(self):
        zero = _zero_activity_parameters()
        path = simulate_fight(
            _spec(red_parameters=zero, blue_parameters=zero),
            0,
            telemetry="full",
        )
        kinds = [event.event_type for event in path.events]
        self.assertEqual(path.result.fight_time_us, 30_000_000)
        self.assertEqual(kinds.count(EventType.ROUND_SCORE), 1)
        self.assertEqual(kinds.count(EventType.ROUND_BELL), 1)
        self.assertEqual(kinds.count(EventType.TERMINATION), 1)
        self.assertLess(kinds.index(EventType.ROUND_BELL), kinds.index(EventType.TERMINATION))
        self.assertEqual(path.red_stats.significant_strike_attempts, 0)
        self.assertEqual(path.blue_stats.significant_strike_attempts, 0)

    def test_research_mechanics_multiplier_changes_only_configured_hazard(self):
        strike_only = replace(
            _zero_activity_parameters(),
            strike_rate_distance=30.0,
        )
        base = _spec(red_parameters=strike_only, blue_parameters=strike_only)
        active = simulate_fight(base, 0, telemetry="none")
        disabled = simulate_fight(
            replace(
                base,
                simulator=SimulatorConfig(distance_strike_hazard_multiplier=0.0),
            ),
            0,
            telemetry="none",
        )
        self.assertGreater(
            active.red_stats.significant_strike_attempts
            + active.blue_stats.significant_strike_attempts,
            0,
        )
        self.assertEqual(disabled.red_stats.significant_strike_attempts, 0)
        self.assertEqual(disabled.blue_stats.significant_strike_attempts, 0)

    def test_global_rare_no_contest_process_is_supported(self):
        zero = _zero_activity_parameters()
        base = _spec(red_parameters=zero, blue_parameters=zero, round_seconds=300)
        path = simulate_fight(
            replace(
                base,
                simulator=SimulatorConfig(no_contest_rate_per_minute=100.0),
            ),
            0,
            telemetry="full",
        )
        self.assertIs(path.result.method, OutcomeMethod.NO_CONTEST)
        self.assertIsNone(path.result.winner)
        self.assertEqual(
            sum(event.event_type is EventType.TERMINATION for event in path.events),
            1,
        )


class FightSimDeterminismTests(unittest.TestCase):
    def test_full_telemetry_and_lean_execution_have_golden_parity(self):
        spec = _spec(round_seconds=60)
        lean = simulate_fight(spec, 17)
        traced = simulate_fight(spec, 17, telemetry="full")
        self.assertEqual(lean.result, traced.result)
        self.assertEqual(lean.red_stats, traced.red_stats)
        self.assertEqual(lean.blue_stats, traced.blue_stats)
        self.assertEqual(lean.final_state_hash, traced.final_state_hash)
        self.assertEqual(
            replay_trace(traced, scheduled_rounds=spec.bout.scheduled_rounds),
            traced.final_state_hash,
        )
        self.assertEqual(
            trace_digest(traced),
            "1c3b265931b5b973867b7957384405c6c34970e7903c1868cf6406c46cae11ac",
        )
        replayed = stochastic_replay(spec, 17, expected=traced)
        self.assertEqual(trace_digest(replayed), trace_digest(traced))
        loaded = trace_from_dict(trace_to_dict(traced))
        self.assertEqual(loaded, traced)

    def test_direct_indices_are_order_and_chunk_independent(self):
        spec = _spec()
        forward = simulate_indices(spec, (2, 7, 11))
        reverse = simulate_indices(spec, (11, 2, 7))
        by_index = {path.simulation_index: path for path in reverse}
        for path in forward:
            other = by_index[path.simulation_index]
            self.assertEqual(path.result, other.result)
            self.assertEqual(path.final_state_hash, other.final_state_hash)

    def test_hash_corruption_and_first_divergence_are_detected(self):
        spec = _spec()
        path = simulate_fight(spec, 5, telemetry="full")
        tampered_event = replace(path.events[3], state_hash_after="0" * 64)
        tampered = (*path.events[:3], tampered_event, *path.events[4:])
        with self.assertRaises(ReplayError):
            reduce_events(initial_state(path.matchup_id, 1), tampered)
        difference = diff_event_streams(path.events, tampered)
        self.assertIsNotNone(difference)
        self.assertEqual(difference.event_index, 3)
        self.assertEqual(difference.field_path, "state_hash_after")

    def test_reducer_rejects_semantically_illegal_and_incomplete_streams(self):
        spec = _spec(round_seconds=60)
        path = simulate_fight(spec, 5, telemetry="full")
        state = initial_state(path.matchup_id, spec.bout.scheduled_rounds)
        checked_illegal_action = False
        for event in path.events:
            if event.event_type is EventType.ACTION_ATTEMPT and state.phase is not Phase.GROUND:
                with self.assertRaisesRegex(ReplayError, "illegal from phase"):
                    reduce_event(
                        state,
                        replace(event, action="submission"),
                        verify_hashes=False,
                    )
                checked_illegal_action = True
                break
            state = reduce_event(state, event)
        self.assertTrue(checked_illegal_action)
        with self.assertRaisesRegex(ReplayError, "exactly one final termination"):
            reduce_events(
                initial_state(path.matchup_id, spec.bout.scheduled_rounds),
                path.events[:-1],
            )

    def test_final_hash_is_the_terminal_reduced_state(self):
        spec = _spec(round_seconds=60)
        path = simulate_fight(spec, 9, telemetry="full")
        reduced = reduce_events(initial_state(path.matchup_id, 1), path.events)
        self.assertIsNotNone(reduced.result)
        self.assertEqual(state_hash(reduced), path.final_state_hash)

    def test_three_and_five_round_runs_share_their_predecision_prefix(self):
        zero = _zero_activity_parameters()
        three = _spec(red_parameters=zero, blue_parameters=zero, round_seconds=30)
        three = replace(
            three,
            bout=replace(three.bout, scheduled_rounds=3),
        )
        five = replace(
            three,
            bout=replace(three.bout, scheduled_rounds=5),
        )
        short = simulate_fight(three, 12, telemetry="full")
        long = simulate_fight(five, 12, telemetry="full")

        def prefix(path):
            rows = []
            for event in path.events:
                if event.event_type is EventType.TERMINATION:
                    break
                rows.append(
                    (
                        event.event_type,
                        event.round_number,
                        event.fight_time_us,
                        event.actor,
                        event.target,
                        event.action,
                        event.payload,
                        event.rng_draws,
                    )
                )
                if event.event_type is EventType.ROUND_BELL and event.round_number == 3:
                    break
            return rows

        self.assertEqual(prefix(short), prefix(long))


class FightSimAggregationTests(unittest.TestCase):
    def test_nested_counts_uncertainty_markets_and_trace_selection_are_coherent(self):
        specs = (_spec(bootstrap_member=0), _spec(bootstrap_member=1))
        result = run_nested(
            specs,
            12,
            chunk_size=5,
            workers=1,
            max_traces=6,
            winner_mcse_target=1.0,
            parameter_quantile_tolerance=1.0,
        )
        forecast = result.forecast
        self.assertEqual(forecast.total_paths, 24)
        self.assertEqual(sum(item.count for item in forecast.outcome_counts), 24)
        self.assertEqual(sum(item.count for item in forecast.duration_bins), 24)
        self.assertTrue(all(
            line.over + line.under + line.push + line.no_action == 24
            for line in forecast.total_lines
        ))
        self.assertTrue(all(
            left.probability >= right.probability
            for left, right in zip(forecast.survival, forecast.survival[1:])
        ))
        markets = coherent_market_probabilities(forecast)
        winner = markets["winner"]
        self.assertAlmostEqual(
            winner["red"] + winner["blue"] + winner["draw"] + winner["no_contest"],
            1.0,
        )
        self.assertLessEqual(len(result.traces), 6)
        self.assertIsNotNone(result.trace_manifest)
        self.assertEqual(len(result.traces), len(result.trace_manifest.trace_hashes))
        for trace in result.traces:
            replay_trace(trace, scheduled_rounds=1)

        run_id = result.trace_manifest.run_id
        selected_forward = select_trace_paths(result.paths, run_id=run_id, max_traces=6)
        selected_reverse = select_trace_paths(reversed(result.paths), run_id=run_id, max_traces=6)
        self.assertEqual(selected_forward, selected_reverse)

    def test_aggregate_is_input_order_invariant(self):
        paths = simulate_indices(_spec(), range(10))
        forward = aggregate_paths(paths, 1).to_dict()
        reverse = aggregate_paths(reversed(paths), 1).to_dict()
        self.assertEqual(forward, reverse)

    def test_worker_and_chunk_configuration_do_not_change_results(self):
        specs = (_spec(bootstrap_member=0), _spec(bootstrap_member=1))
        serial = run_nested(
            specs,
            10,
            workers=1,
            chunk_size=3,
            max_traces=0,
            winner_mcse_target=1.0,
            parameter_quantile_tolerance=1.0,
        )
        parallel = run_nested(
            specs,
            10,
            workers=2,
            chunk_size=7,
            max_traces=0,
            winner_mcse_target=1.0,
            parameter_quantile_tolerance=1.0,
        )
        self.assertEqual(serial.forecast.to_dict(), parallel.forecast.to_dict())
        self.assertEqual(
            [path.final_state_hash for path in serial.paths],
            [path.final_state_hash for path in parallel.paths],
        )

    def test_total_settlement_has_explicit_boundary_and_no_action(self):
        base = FightResult(
            winner=Side.RED,
            method=OutcomeMethod.KO_TKO,
            round_number=1,
            fight_time_us=150_000_000,
            round_time_us=150_000_000,
            reason="test",
        )
        self.assertIs(settle_total(base, 0.5), Settlement.PUSH)
        self.assertIs(
            settle_total(replace(base, fight_time_us=149_999_999, round_time_us=149_999_999), 0.5),
            Settlement.UNDER,
        )
        self.assertIs(
            settle_total(replace(base, fight_time_us=150_000_001, round_time_us=150_000_001), 0.5),
            Settlement.OVER,
        )
        no_contest = FightResult(
            winner=None,
            method=OutcomeMethod.NO_CONTEST,
            round_number=1,
            fight_time_us=100,
            round_time_us=100,
            reason="test",
        )
        self.assertIs(settle_total(no_contest, 0.5), Settlement.NO_ACTION)

    def test_identical_profiles_are_statistically_symmetric(self):
        paths = simulate_indices(_spec(round_seconds=60), range(1200))
        red = sum(path.result.winner is Side.RED for path in paths)
        blue = sum(path.result.winner is Side.BLUE for path in paths)
        decisive = red + blue
        self.assertGreater(decisive, 0)
        self.assertLess(abs(red / decisive - 0.5), 0.06)

    def test_directional_archetype_has_expected_effect(self):
        strong = FighterParameters(
            strike_rate_distance=12.0,
            strike_accuracy=0.65,
            strike_power=0.85,
            knockdown_rate_per_landed=0.08,
            finish_after_knockdown=0.55,
        )
        weak = FighterParameters(
            strike_rate_distance=3.5,
            strike_accuracy=0.30,
            strike_defense=0.35,
            strike_power=0.25,
            ko_resistance=0.30,
        )
        paths = simulate_indices(
            _spec(round_seconds=120, red_parameters=strong, blue_parameters=weak),
            range(800),
        )
        red = sum(path.result.winner is Side.RED for path in paths)
        blue = sum(path.result.winner is Side.BLUE for path in paths)
        self.assertGreater(red, blue)

    def test_swapping_profiles_swaps_win_probability(self):
        strong = FighterParameters(
            strike_rate_distance=10.0,
            strike_accuracy=0.58,
            strike_defense=0.70,
            strike_power=0.70,
            knockdown_rate_per_landed=0.05,
            finish_after_knockdown=0.45,
        )
        weak = FighterParameters(
            strike_rate_distance=5.0,
            strike_accuracy=0.35,
            strike_defense=0.42,
            strike_power=0.30,
            ko_resistance=0.40,
        )
        forward = simulate_indices(
            _spec(
                root_seed="swap-symmetry",
                round_seconds=120,
                red_parameters=strong,
                blue_parameters=weak,
            ),
            range(1200),
        )
        reverse = simulate_indices(
            _spec(
                root_seed="swap-symmetry",
                round_seconds=120,
                red_parameters=weak,
                blue_parameters=strong,
            ),
            range(1200),
        )
        forward_red = sum(path.result.winner is Side.RED for path in forward)
        forward_blue = sum(path.result.winner is Side.BLUE for path in forward)
        reverse_red = sum(path.result.winner is Side.RED for path in reverse)
        reverse_blue = sum(path.result.winner is Side.BLUE for path in reverse)
        self.assertGreater(forward_red + forward_blue, 0)
        self.assertGreater(reverse_red + reverse_blue, 0)
        forward_probability = forward_red / (forward_red + forward_blue)
        reverse_probability = reverse_blue / (reverse_red + reverse_blue)
        self.assertLess(abs(forward_probability - reverse_probability), 0.05)


if __name__ == "__main__":
    unittest.main()
