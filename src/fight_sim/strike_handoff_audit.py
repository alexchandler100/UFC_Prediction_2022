"""Fixed, causal estimator-handoff comparisons; no new fight trajectories.

Run: PYTHONPATH=src python -m fight_sim.strike_handoff_audit
"""
from __future__ import annotations

import hashlib
import json
import math
import time

import numpy as np
import pandas as pd

from .domain import Side, SimulatorConfig
from .opponent_audit import (
    ChronologicalOpponentRidgeSelector, _context_predictions, _negative_log_likelihood,
)
from .parameters import CausalParameterFitter, ParameterFitConfig, canonical_sha256, load_parameter_artifact
from .research import _fit_cache_contract, build_specs
from .strike_bridge_audit import (
    ROOT, OUTPUT, PHASES, card_interval, git_bytes, historical_frame,
    integrate, load_runs, neutral_phase_rates,
)
from .strike_handoff import (
    STRIKE_TARGETS, context_occupancy, covariate_multiplier, fit_strike_effects,
    map_matchup_strikes, weighted_observation_context,
)

LOCAL = ROOT / "artifacts/simulations/strike-handoff-20261001"
CANDIDATE = "observation_bootstrap"
BOOTSTRAP_ARMS = (
    CANDIDATE, "simulator_bootstrap", "simulator_bootstrap_covariates",
    "observation_bootstrap_covariates",
)


def simulator_baseline(frame, member, target):
    contexts = [member.context_parameters.get(
        f"{str(division).strip() or 'Unknown'}|{era}", member.context_parameters["__global__"]
    ) for division, era in zip(frame.division, frame.era)]
    if target.name == "strike_pace":
        return np.array([math.fsum(context[f"strike_rate_{phase}"] * context[f"{phase}_phase_share"]
                                   for phase in PHASES) for context in contexts])
    return np.array([context["strike_accuracy"] for context in contexts])


def forecast_arm(training, test, weights, selected, cutoff, member=None,
                 *, context_source="observation", use_covariates=False):
    output = {}
    for target in STRIKE_TARGETS:
        if context_source == "observation":
            context = weighted_observation_context(training, target, weights)
            train_base = _context_predictions(training, context)
            test_base = _context_predictions(test, context)
        else:
            train_base = simulator_baseline(training, member, target)
            test_base = simulator_baseline(test, member, target)
        if use_covariates and target.name == "strike_pace":
            train_base *= covariate_multiplier(training, member.covariate_effects)
            test_base *= covariate_multiplier(test, member.covariate_effects)
        effects = fit_strike_effects(training, target, train_base, weights,
                                     selected[(target.name, "opponent_adjusted")], cutoff=cutoff)
        output[target.name] = effects.predict(test_base, test.fighter_id, test.opponent_id)
    return output


def validate_saved_inputs():
    prior = json.loads(git_bytes("2bc8e378:audit/simulation/2026-10-01/strike_bridge_results.json"))
    for name, expected in prior["input_file_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"prior input changed: {name}")
    old_member_path = ROOT / "artifacts/simulations/strike-bridge-20261001/member_rows.csv"
    if hashlib.sha256(old_member_path.read_bytes()).hexdigest() != prior["member_rows_sha256"]:
        raise ValueError("prior bridge member rows changed")
    for name in ("parameters.py", "opponent_audit.py", "engine.py", "domain.py"):
        old = git_bytes(f"59a16cd2:src/fight_sim/{name}").replace(b"\r\n", b"\n")
        if old != (ROOT / "src/fight_sim" / name).read_bytes().replace(b"\r\n", b"\n"):
            raise ValueError(f"original implementation changed: {name}")
    old = pd.read_csv(old_member_path, dtype={key: str for key in ("fight_id", "event_id", "fighter_id")})
    original = pd.read_csv(ROOT / "artifacts/simulations/opponent-adjustment-bout-clustered-predictions-20260827.csv",
                           dtype={key: str for key in ("fight_id", "event_id", "fighter_id")})
    return prior, old, original


def maximum_reproduction_error(actual, expected, label):
    if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=1e-9, atol=1e-9):
        raise ValueError(f"failed to reproduce {label}; max difference {np.max(np.abs(actual-expected))}")
    return float(np.max(np.abs(actual - expected)))


def run_comparisons(design, manifests, records, prior, old, original):
    started = time.monotonic()
    contract = manifests["v2"]["run_contract"]
    hashes = contract["source_sha256"]
    fitter = CausalParameterFitter(
        historical_frame("ufc_fights_reported_doubled.csv", hashes["raw"], "59a16cd2"),
        historical_frame("fighter_stats.csv", hashes["profiles"], "59a16cd2"),
        historical_frame("ufc_fight_round_stats_doubled.csv", hashes["round_stats"], "59a16cd2"),
        use_takedown_control_association=True,
    )
    selector = ChronologicalOpponentRidgeSelector(fitter.raw_fights)
    frame = fitter._attach_age(selector.frame)
    event_dates = frame.groupby("event_id").date.max()
    prior_cards = {row["event_id"]: row for row in prior["causal_checks"]}
    simulator = SimulatorConfig(**contract["simulation"]["simulator_config"])
    rows, provenance, mappings = [], [], []
    maximum_errors = {"observation_reference": 0.0, "old_v2_conditional": 0.0}
    original_lookup = original.pivot(index=["fight_id", "fighter_id"], columns="target", values="opponent_adjusted_prediction")
    old_lookup = old.set_index(["fight_id", "fighter_id", "member"])
    events = pd.DataFrame(records["v2"].values()).groupby(["causal_cutoff_utc", "event_id"], sort=True)
    for position, ((timestamp, event), fights) in enumerate(events, 1):
        if time.monotonic() - started > design["maximum_runtime_seconds"]:
            raise TimeoutError("handoff audit exceeded its frozen time budget")
        cutoff = pd.Timestamp(timestamp)
        training = frame[frame.date.lt(cutoff)].copy()
        test = frame[frame.fight_id.isin(fights.fight_id)].sort_values(["fight_id", "fighter_id"]).copy()
        if len(test) != 2 * len(fights) or not test.date.eq(cutoff).all() or not test.event_id.eq(event).all():
            raise ValueError("outer-card membership mismatch")
        # Use the exact pre-cutoff snapshot covariates, never target outcomes.
        metadata = pd.DataFrame([fitter._snapshot_metadata(identity, cutoff) for identity in test.fighter_id], index=test.index)
        for field in ("age_years", "experience_fights", "layoff_days"):
            test[field] = metadata[field]
        selected, inner_ids = selector.selected_for_cutoff(cutoff)
        if len(inner_ids) != 8 or any(event_dates.loc[inner] >= cutoff for inner in inner_ids):
            raise ValueError("inner selection violates causal cutoff")
        saved_selection = prior_cards[event]["member_selection"][0]
        if list(inner_ids) != saved_selection["inner_event_ids"]:
            raise ValueError("inner selection cards changed")
        for target in STRIKE_TARGETS:
            if selected[(target.name, "opponent_adjusted")] != saved_selection[f"selected_{target.name}_ridge"]:
                raise ValueError("selected shrinkage changed")
        reference = forecast_arm(training, test, np.ones(len(training)), selected, cutoff)
        indices = pd.MultiIndex.from_frame(test[["fight_id", "fighter_id"]])
        expected = original_lookup.reindex(indices)
        error = maximum_reproduction_error(
            np.column_stack([reference[target.name] for target in STRIKE_TARGETS]),
            expected[[target.name for target in STRIKE_TARGETS]].to_numpy(), "original observation predictor",
        )
        maximum_errors["observation_reference"] = max(error, maximum_errors["observation_reference"])
        event_seed = 2903 + int(canonical_sha256({"event_id": event, "date": cutoff.isoformat()})[:8], 16)
        config = ParameterFitConfig.historical(bootstrap_members=10, random_seed=event_seed)
        cache_contract = _fit_cache_contract(cutoff=cutoff, config=config, source_sha256=hashes, parameter_model=contract["parameter_model"])
        cache = ROOT / "artifacts/simulations/causal-fit-cache" / f"fit-{canonical_sha256(cache_contract)}.json.gz"
        if hashlib.sha256(cache.read_bytes()).hexdigest() != prior_cards[event]["cache_sha256"]:
            raise ValueError("historical cached member file changed")
        artifact = load_parameter_artifact(cache)
        if artifact.artifact_sha256 != prior_cards[event]["artifact_sha256"] or artifact.config != config:
            raise ValueError("historical member contract mismatch")
        if pd.to_datetime(artifact.trained_through, utc=True) >= cutoff or pd.Timestamp(artifact.as_of_utc) != cutoff:
            raise ValueError("parameter fit is not causal")
        full_specs = {fight["fight_id"]: build_specs(
            fitter, artifact, red_fighter_id=fight["red_fighter_id"], blue_fighter_id=fight["blue_fighter_id"],
            division=fight["division"], scheduled_rounds=fight["scheduled_rounds"], event_id=event,
            root_seed=f"posterior:2903:{fight['fight_id']}", simulator_base=simulator,
            snapshot_parameter_mode="full", _artifact_validated=True,
        ) for fight in fights.to_dict("records")}
        event_ids = np.array(sorted(training.event_id.unique()), dtype=object)
        index_lookup = {(str(row.fight_id), str(row.fighter_id)): index for index, row in enumerate(test.itertuples())}
        for member in artifact.members:
            if member.sampled_event_count != len(event_ids):
                raise ValueError("training event set changed")
            rng = np.random.Generator(np.random.PCG64DXSM(member.bootstrap_seed))
            unique, counts = np.unique(rng.choice(event_ids, size=len(event_ids), replace=True), return_counts=True)
            weights = training.event_id.map(dict(zip(unique, counts))).fillna(0).to_numpy(float)
            arms = {"observation_reference": reference}
            for arm in BOOTSTRAP_ARMS:
                arms[arm] = forecast_arm(training, test, weights, selected, cutoff, member,
                                          context_source=arm.split("_")[0], use_covariates=arm.endswith("covariates"))
            prior_rows = old_lookup.reindex(pd.MultiIndex.from_tuples(
                [(fight, identity, member.member_index) for fight, identity in indices], names=old_lookup.index.names
            ))
            error = maximum_reproduction_error(
                np.column_stack([arms["simulator_bootstrap_covariates"][target.name] for target in STRIKE_TARGETS]),
                np.column_stack([prior_rows.conditional_attempts / prior_rows.actual_minutes, prior_rows.conditional_accuracy]),
                "old v2 conditional predictor",
            )
            maximum_errors["old_v2_conditional"] = max(error, maximum_errors["old_v2_conditional"])
            mapped_predictions = {target.name: np.empty(len(test)) for target in STRIKE_TARGETS}
            for fight in fights.to_dict("records"):
                spec = full_specs[fight["fight_id"]][member.member_index]
                context = member.context_parameters.get(f"{fight['division']}|{fitter._era_key(cutoff, 5)}", member.context_parameters["__global__"])
                occupancy = {side: context_occupancy(context) for side in Side}
                targets = {side: tuple(arms[CANDIDATE][target.name][index_lookup[(fight["fight_id"], fight[f"{side.value}_fighter_id"])]]
                                      for target in STRIKE_TARGETS) for side in Side}
                mapped = map_matchup_strikes(spec, targets, occupancy)
                for side in Side:
                    index = index_lookup[(fight["fight_id"], fight[f"{side.value}_fighter_id"])]
                    pace, landed = integrate(neutral_phase_rates(mapped, side), occupancy[side])
                    mapped_predictions["strike_pace"][index] = pace
                    mapped_predictions["strike_accuracy"][index] = landed / pace
                    mappings.append({"fight_id": fight["fight_id"], "event_id": event, "member": member.member_index,
                                     "side": side.value, "rate_absolute_error": abs(pace-targets[side][0]),
                                     "accuracy_absolute_error": abs(landed/pace-targets[side][1]),
                                     "strike_accuracy_parameter": getattr(mapped, side.value).parameters.strike_accuracy,
                                     **{f"strike_rate_{phase}": getattr(getattr(mapped, side.value).parameters, f"strike_rate_{phase}") for phase in PHASES}})
            arms["mapped_candidate"] = mapped_predictions
            for arm, prediction in arms.items():
                for index, observed in enumerate(test.to_dict("records")):
                    fight = records["v2"][observed["fight_id"]]
                    side = "red" if observed["fighter_id"] == fight["red_fighter_id"] else "blue"
                    rows.append({"fight_id": observed["fight_id"], "fighter_id": observed["fighter_id"], "event_id": event,
                                 "date": str(cutoff.date()), "member": member.member_index, "arm": arm, "side": side,
                                 "winner": fight["actual_outcome"].split("_")[0], "actual_minutes": observed["fight_seconds"]/60,
                                 "actual_attempts": observed["sig_strikes_attempts"], "actual_landed": observed["sig_strikes_landed"],
                                 "pace": prediction["strike_pace"][index], "accuracy": prediction["strike_accuracy"][index],
                                 "landed_rate": prediction["strike_pace"][index]*prediction["strike_accuracy"][index]})
        provenance.append({"event_id": event, "cutoff": cutoff.isoformat(), "training_latest": training.date.max().isoformat(),
                           "inner_cards": list(inner_ids), "artifact_sha256": artifact.artifact_sha256,
                           "selected_ridge": {target.name: selected[(target.name, "opponent_adjusted")] for target in STRIKE_TARGETS}})
        print(f"Card {position}/30: {len(fights)} fights; {time.monotonic()-started:.1f}s", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(mappings), provenance, maximum_errors, time.monotonic()-started


def score_rows(members, original):
    keys = ["fight_id", "fighter_id", "event_id", "date", "side", "winner", "arm"]
    sides = members.groupby(keys, as_index=False).mean(numeric_only=True).drop(columns="member")
    marginal = sides[sides.arm.eq("observation_reference")].copy()
    lookup = original.pivot(index=["fight_id", "fighter_id"], columns="target", values="marginal_prediction")
    matched = lookup.reindex(pd.MultiIndex.from_frame(marginal[["fight_id", "fighter_id"]]))
    marginal["pace"] = matched.strike_pace.to_numpy()
    marginal["accuracy"] = matched.strike_accuracy.to_numpy()
    marginal["landed_rate"] = marginal.pace * marginal.accuracy
    marginal["arm"] = "original_marginal"
    sides = pd.concat([sides, marginal], ignore_index=True)
    sides["pace_nll"] = _negative_log_likelihood(sides.actual_attempts.to_numpy(), sides.actual_minutes.to_numpy(), sides.pace.to_numpy(), kind="rate")
    sides["accuracy_nll"] = _negative_log_likelihood(sides.actual_landed.to_numpy(), sides.actual_attempts.to_numpy(), sides.accuracy.to_numpy(), kind="probability")
    summaries, ranking_frames = {}, {}
    for arm, group in sides.groupby("arm"):
        ranking = group.pivot(index=["fight_id", "event_id", "winner"], columns="side", values="landed_rate").reset_index()
        ranking = ranking[ranking.winner.isin(["red", "blue"])].copy()
        difference = ranking.red-ranking.blue
        ranking["correct"] = np.where(difference.eq(0), .5, ((difference > 0) == ranking.winner.eq("red")).astype(float))
        ranking_frames[arm] = ranking
        summaries[arm] = {"pace_nll": float(group.pace_nll.mean()), "accuracy_nll": float(group.accuracy_nll.mean()),
                          "attempt_bias_per_side": float((group.pace*group.actual_minutes-group.actual_attempts).mean()),
                          "landed_bias_per_side": float((group.landed_rate*group.actual_minutes-group.actual_landed).mean()),
                          "strike_leader_wins": float(ranking.correct.sum()), "decisive_fights": len(ranking),
                          "ranking_card_interval": card_interval(ranking, "correct")}
    comparisons = {}
    pairs = ((CANDIDATE, "observation_reference"), ("simulator_bootstrap", CANDIDATE),
             ("simulator_bootstrap_covariates", "simulator_bootstrap"), ("observation_bootstrap_covariates", CANDIDATE),
             (CANDIDATE, "simulator_bootstrap_covariates"), (CANDIDATE, "original_marginal"))
    for candidate, baseline in pairs:
        candidate_rows = sides[sides.arm.eq(candidate)].set_index(["fight_id", "fighter_id"])
        baseline_rows = sides[sides.arm.eq(baseline)].set_index(["fight_id", "fighter_id"]).reindex(candidate_rows.index)
        comparison = {}
        for metric in ("pace_nll", "accuracy_nll"):
            delta = pd.DataFrame({"event_id": candidate_rows.event_id, "difference": candidate_rows[metric]-baseline_rows[metric]})
            comparison[metric] = card_interval(delta, "difference")
        ranks = ranking_frames[candidate].set_index("fight_id")
        other = ranking_frames[baseline].set_index("fight_id").reindex(ranks.index)
        delta = pd.DataFrame({"event_id": ranks.event_id, "difference": ranks.correct-other.correct})
        comparison["ranking"] = card_interval(delta, "difference")
        comparisons[f"{candidate}_minus_{baseline}"] = comparison
    candidate = summaries[CANDIDATE]
    gates = {f"lower_{metric}_than_{baseline}": candidate[metric] < summaries[baseline][metric]
             for baseline in ("simulator_bootstrap_covariates", "original_marginal") for metric in ("pace_nll", "accuracy_nll")}
    gates["preserves_original_winner_ordering"] = candidate["strike_leader_wins"] >= summaries["observation_reference"]["strike_leader_wins"]
    return sides, {"scores": summaries, "paired_card_intervals": comparisons, "performance_gates": gates}


def main():
    design_path = OUTPUT / "strike_handoff_design.json"
    design = json.loads(design_path.read_text())
    if design["candidate"] != CANDIDATE or design["cohort"] != "development_2024":
        raise ValueError("unexpected frozen experiment contract")
    started = time.monotonic()
    prior, old, original = validate_saved_inputs()
    manifests, records = load_runs()
    members, mappings, provenance, errors, calculation_seconds = run_comparisons(design, manifests, records, prior, old, original)
    if len(members) != 229*2*10*6 or members.duplicated(["fight_id", "fighter_id", "member", "arm"]).any() or len(mappings) != 4580:
        raise ValueError("incomplete handoff comparison")
    sides, report = score_rows(members, original)
    LOCAL.mkdir(parents=True, exist_ok=True)
    for name, frame in (("member_rows", members), ("mapping_rows", mappings), ("side_rows", sides)):
        frame.to_csv(LOCAL / f"{name}.csv", index=False)
    report.update(schema_version=1, research_only=True, candidate=CANDIDATE, production_changed=False,
                  new_trajectories=0, locked_cohorts_opened=[], fights=229, cards=30, member_side_rows=4580,
                  comparison_rows=len(members), calculation_seconds=calculation_seconds,
                  total_seconds=time.monotonic()-started, design_sha256=canonical_sha256(design),
                  source_sha256=manifests["v2"]["run_contract"]["source_sha256"],
                  prior_bridge_report_sha256=canonical_sha256(prior), reproduction_maximum_errors=errors,
                  mapping_maximum_errors={name: float(mappings[name].max()) for name in ("rate_absolute_error", "accuracy_absolute_error")},
                  causal_checks=provenance, limitations=design["limitations"],
                  output_sha256={f"{name}.csv": hashlib.sha256((LOCAL/f"{name}.csv").read_bytes()).hexdigest() for name in ("member_rows", "mapping_rows", "side_rows")})
    report["candidate_advances_to_simulation_screen"] = all(report["performance_gates"].values())
    report["report_sha256"] = canonical_sha256(report)
    (OUTPUT / "strike_handoff_results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: report[key] for key in ("scores", "performance_gates", "reproduction_maximum_errors", "mapping_maximum_errors", "candidate_advances_to_simulation_screen")}, indent=2))


if __name__ == "__main__":
    main()
