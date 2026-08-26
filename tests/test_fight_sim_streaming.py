from __future__ import annotations

from dataclasses import replace
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fight_sim.aggregate import ForecastAccumulator, aggregate_paths  # noqa: E402
from fight_sim.domain import (  # noqa: E402
    BoutConfig,
    FighterParameters,
    FighterSnapshot,
    SimulationRunSpec,
    SimulatorConfig,
)
from fight_sim.engine import simulate_indices  # noqa: E402
from fight_sim.monte_carlo import (  # noqa: E402
    convergence_diagnostics,
    NestedSimulationBatchError,
    run_adaptive_nested,
    run_nested,
)
from fight_sim.telemetry import trace_digest  # noqa: E402
from fight_sim.research import execute_run  # noqa: E402


def _snapshot(fighter_id: str) -> FighterSnapshot:
    return FighterSnapshot(
        fighter_id=fighter_id,
        fighter_name=fighter_id.upper(),
        as_of_utc="2026-08-01T00:00:00+00:00",
        division="Lightweight",
        parameters=FighterParameters(),
        experience_fights=12,
        observed_fight_seconds=2_400,
        observed_rounds=8,
        data_quality="observed",
    )


def _spec(member: int = 0) -> SimulationRunSpec:
    red = _snapshot("red-id")
    blue = _snapshot("blue-id")
    return SimulationRunSpec(
        bout=BoutConfig(
            matchup_id="red-id--blue-id",
            red_fighter_id=red.fighter_id,
            blue_fighter_id=blue.fighter_id,
            scheduled_rounds=1,
            round_seconds=30,
            dynamics_seconds=5,
            division="Lightweight",
        ),
        red=red,
        blue=blue,
        root_seed="streaming-test-seed",
        parameter_artifact_id="artifact-sha256",
        bootstrap_member=member,
    )


class StreamingAccumulatorTests(unittest.TestCase):
    def test_accumulator_merge_matches_path_reducer_exactly(self):
        paths = simulate_indices(_spec(), range(18))
        expected = aggregate_paths(paths, 1).to_dict()
        left = ForecastAccumulator(1)
        right = ForecastAccumulator(1)
        for path in paths[::2]:
            left.add_path(path)
        for path in reversed(paths[1::2]):
            right.add_path(path)

        left.merge(right)

        self.assertEqual(left.forecast().to_dict(), expected)
        self.assertEqual(len(left.duration_values_us), 18)
        self.assertEqual(
            len(left.duration_values_us) * left.duration_values_us.itemsize,
            18 * 8,
        )

    def test_streaming_equals_retained_forecast_convergence_and_traces(self):
        specs = (_spec(0), _spec(1))
        retained = run_nested(
            specs,
            12,
            workers=1,
            chunk_size=7,
            max_traces=6,
            retain_paths=True,
            winner_mcse_target=1.0,
            parameter_quantile_tolerance=1.0,
        )
        streamed = run_nested(
            reversed(specs),
            12,
            workers=1,
            chunk_size=5,
            max_traces=6,
            retain_paths=False,
            winner_mcse_target=1.0,
            parameter_quantile_tolerance=1.0,
        )

        self.assertEqual(streamed.paths, ())
        self.assertFalse(streamed.paths_retained)
        self.assertEqual(streamed.forecast.to_dict(), retained.forecast.to_dict())
        self.assertEqual(streamed.convergence, retained.convergence)
        self.assertEqual(streamed.trace_manifest, retained.trace_manifest)
        self.assertEqual(
            [trace_digest(path) for path in streamed.traces],
            [trace_digest(path) for path in retained.traces],
        )
        self.assertEqual(
            convergence_diagnostics(
                retained.paths,
                retained.forecast,
                winner_mcse_target=1.0,
                parameter_quantile_tolerance=1.0,
            ),
            streamed.convergence[0],
        )
        self.assertIsNotNone(streamed.ledger)
        self.assertTrue(streamed.ledger.streaming)
        self.assertEqual(streamed.ledger.total_paths, 24)
        self.assertEqual(streamed.ledger.retained_paths, 0)
        self.assertEqual(streamed.ledger.packed_duration_bytes, 24 * 8)
        self.assertLessEqual(streamed.ledger.max_in_flight_paths, 5)

    def test_streaming_is_worker_and_chunk_invariant_with_bounded_window(self):
        specs = (_spec(0), _spec(1))
        serial = run_nested(
            specs,
            10,
            workers=1,
            chunk_size=3,
            max_traces=5,
            retain_paths=False,
            winner_mcse_target=1.0,
            parameter_quantile_tolerance=1.0,
        )
        parallel = run_nested(
            specs,
            10,
            workers=2,
            chunk_size=7,
            max_traces=5,
            retain_paths=False,
            winner_mcse_target=1.0,
            parameter_quantile_tolerance=1.0,
        )

        self.assertEqual(serial.forecast.to_dict(), parallel.forecast.to_dict())
        self.assertEqual(serial.convergence, parallel.convergence)
        self.assertEqual(serial.trace_manifest, parallel.trace_manifest)
        self.assertEqual(
            [trace_digest(path) for path in serial.traces],
            [trace_digest(path) for path in parallel.traces],
        )
        self.assertLessEqual(parallel.ledger.max_in_flight_paths, 2 * 2 * 7)

    def test_auto_streaming_does_not_retain_paths(self):
        result = run_nested(
            (_spec(),),
            6,
            workers=1,
            chunk_size=2,
            max_traces=0,
            path_retention_limit=5,
            winner_mcse_target=1.0,
            parameter_quantile_tolerance=1.0,
        )

        self.assertEqual(result.paths, ())
        self.assertTrue(result.ledger.streaming)
        self.assertEqual(result.ledger.total_paths, 6)
        self.assertEqual(result.ledger.invariant_failure_count, 0)
        self.assertEqual(result.invariant_failures, ())

    def test_real_event_cap_failure_is_durable_and_withholds_the_batch(self):
        failing = replace(_spec(), simulator=SimulatorConfig(max_events=1))

        with self.assertRaises(NestedSimulationBatchError) as raised:
            run_nested(
                (failing,),
                6,
                workers=1,
                chunk_size=2,
                max_traces=0,
                retain_paths=False,
            )

        error = raised.exception
        self.assertEqual(len(error.failures), 1)
        failure = error.failures[0]
        self.assertEqual(failure.bootstrap_member, 0)
        self.assertEqual(failure.simulation_index, 0)
        self.assertIn("event cap 1", failure.failures[0])
        self.assertGreaterEqual(len(failure.events), 2)
        payload = error.to_dict()
        self.assertEqual(payload["status"], "failed_invariant")
        self.assertFalse(payload["complete"])
        self.assertFalse(payload["published"])
        self.assertEqual(
            len(payload["failures"][0]["events"]),
            len(failure.events),
        )

    def test_execute_run_atomically_persists_invariant_failure(self):
        failing = replace(_spec(), simulator=SimulatorConfig(max_events=1))
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "failed-run"
            with (
                patch(
                    "fight_sim.research.load_parameter_artifact",
                    return_value=object(),
                ),
                patch(
                    "fight_sim.research.load_research_inputs",
                    return_value=(object(), None, None),
                ),
                patch(
                    "fight_sim.research.CausalParameterFitter",
                    return_value=object(),
                ),
                patch(
                    "fight_sim.research.build_specs",
                    return_value=(failing,),
                ),
            ):
                with self.assertRaises(NestedSimulationBatchError) as raised:
                    execute_run(
                        parameter_path="ignored-parameters",
                        raw_path="ignored-raw",
                        profiles_path="ignored-profiles",
                        round_path="ignored-rounds",
                        red_fighter_id="red-id",
                        blue_fighter_id="blue-id",
                        division="Lightweight",
                        output_dir=destination,
                        initial_paths_per_member=2,
                        max_paths_per_member=2,
                        chunk_size=2,
                        max_traces=0,
                    )

            failure_path = destination / "invariant-failure.json"
            self.assertEqual(raised.exception.failure_path, failure_path)
            self.assertEqual(
                json.loads(failure_path.read_text(encoding="utf-8")),
                raised.exception.to_dict(),
            )
            self.assertTrue((destination / "specs.json").is_file())
            self.assertFalse((destination / "aggregate.json").exists())
            self.assertFalse((destination / "convergence.json").exists())
            self.assertFalse((destination / "analysis.html").exists())

    def test_adaptive_chunk_merges_equal_one_shot_final_range(self):
        specs = (_spec(0), _spec(1))
        adaptive = run_adaptive_nested(
            specs,
            initial_paths_per_member=4,
            max_paths_per_member=8,
            workers=1,
            chunk_size=3,
            max_traces=0,
            retain_paths=False,
            winner_mcse_target=1e-12,
            parameter_quantile_tolerance=0.0,
        )
        one_shot = run_nested(
            specs,
            8,
            workers=1,
            chunk_size=5,
            max_traces=0,
            retain_paths=False,
            winner_mcse_target=1e-12,
            parameter_quantile_tolerance=0.0,
        )

        self.assertEqual(len(adaptive.convergence), 2)
        self.assertEqual(adaptive.forecast.to_dict(), one_shot.forecast.to_dict())
        self.assertEqual(adaptive.convergence[-1], one_shot.convergence[-1])
        self.assertEqual(adaptive.ledger.packed_duration_bytes, 16 * 8)


if __name__ == "__main__":
    unittest.main()
