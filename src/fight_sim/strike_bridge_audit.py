"""Offline diagnostic for the frozen 229-fight development study; no trajectories.

Run with PYTHONPATH=src: python -m fight_sim.strike_bridge_audit.
Historical source bytes and cached posterior members are mandatory. No fitting
fallback, holdout evaluation, mechanics changes, or betting decisions occur here.
"""
from __future__ import annotations

from dataclasses import replace
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from .domain import Phase, Side, SimulatorConfig, TelemetryLevel
from .engine import STRIKES_PER_EXCHANGE, _Runtime, _hazards, _strike_probability
from .opponent_audit import _negative_log_likelihood
from .parameters import CausalParameterFitter, ParameterFitConfig, canonical_sha256, load_parameter_artifact
from .posterior_predictive import _crps
from .research import _fit_cache_contract, build_specs

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "audit/simulation/2026-10-01"
LOCAL = ROOT / "artifacts/simulations/strike-bridge-20261001"
RUNS = {
    "full": "two-route-v2-1-development-100paths-20260827",
    "v2": "two-route-v2-1-opponent-adjusted-v2-strikes-development-100paths-20260827",
}
PHASES = ("distance", "clinch", "ground")


def git_bytes(spec: str) -> bytes:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "show", spec], cwd=ROOT
    )


def historical_frame(name: str, expected: str, revision: str) -> pd.DataFrame:
    payload = git_bytes(f"{revision}:src/content/data/processed/{name}")
    normalized = payload.replace(b"\r\n", b"\n")
    candidates = (payload, normalized, normalized.replace(b"\n", b"\r\n"))
    if not any(hashlib.sha256(value).hexdigest() == expected for value in candidates):
        raise ValueError(f"historical source hash mismatch: {name}")
    return pd.read_csv(io.BytesIO(payload), low_memory=False)


def neutral_phase_rates(spec, side: Side) -> dict[str, tuple[float, float]]:
    """Exact engine attempts/landed per minute at full stamina and zero hurt."""
    runtime = _Runtime(spec, 0, TelemetryLevel.NONE)
    result = {}
    for label, phase, top in (
        ("distance", Phase.DISTANCE, None),
        ("clinch", Phase.CLINCH, None),
        ("ground_top", Phase.GROUND, side),
        ("ground_bottom", Phase.GROUND, side.opponent),
        ("scramble", Phase.SCRAMBLE, None),
    ):
        runtime.state.phase = phase
        runtime.state.top_position = top
        attempts = math.fsum(
            rate * 60 * STRIKES_PER_EXCHANGE
            for action, actor, rate in _hazards(runtime)
            if action == "strike" and actor == side
        )
        result[label] = (attempts, attempts * _strike_probability(runtime, side))
    return result


def integrate(rates, weights) -> tuple[float, float]:
    if any(value < -1e-8 for value in weights.values()) or not math.isclose(
        sum(weights.values()), 1.0, abs_tol=1e-8
    ):
        raise ValueError("phase weights must be nonnegative and sum to one")
    return tuple(math.fsum(rates[key][i] * value for key, value in weights.items()) for i in (0, 1))


def accuracy_sensitivity(spec, side, context, offense, vulnerability):
    """Local engine log-odds response; the observation formula has slope 1."""
    h = 1e-5
    def probability(actor_delta, opponent_delta):
        actor = getattr(spec, side.value)
        opponent = getattr(spec, side.opponent.value)
        actor = replace(actor, parameters=replace(
            actor.parameters, strike_accuracy=float(expit(logit(context["strike_accuracy"]) + offense + actor_delta))
        ))
        opponent = replace(opponent, parameters=replace(
            opponent.parameters, strike_defense=float(expit(logit(context["strike_defense"]) - vulnerability - opponent_delta))
        ))
        runtime = _Runtime(replace(spec, **{side.value: actor, side.opponent.value: opponent}), 0, TelemetryLevel.NONE)
        return float(logit(_strike_probability(runtime, side)))
    return (
        (probability(h, 0) - probability(-h, 0)) / (2 * h),
        (probability(0, h) - probability(0, -h)) / (2 * h),
    )


def rate_covariate(snapshot, member):
    effects = member.covariate_effects
    age = ((snapshot.age_years - 30) / 10 if snapshot.age_years is not None else effects.get("age_center", 0))
    experience = math.log1p(max(float(snapshot.experience_fights or 0), 0))
    layoff = (math.log1p(max(snapshot.layoff_days, 0) / 365.25) if snapshot.layoff_days is not None else effects.get("log_layoff_years_center", 0))
    adjustment = (
        effects.get("age_per_decade", 0) * (age - effects.get("age_center", 0))
        + effects.get("log_experience", 0) * (experience - effects.get("log_experience_center", 0))
        + effects.get("log_layoff_years", 0) * (layoff - effects.get("log_layoff_years_center", 0))
    )
    return float(np.clip(math.exp(adjustment), .6, 1.4))


def card_interval(values: pd.DataFrame, column: str):
    grouped = values.groupby("event_id")[column].agg(["sum", "count"])
    rng = np.random.Generator(np.random.PCG64DXSM(20261001))
    choices = rng.integers(0, len(grouped), size=(4000, len(grouped)))
    samples = grouped["sum"].to_numpy()[choices].sum(axis=1) / grouped["count"].to_numpy()[choices].sum(axis=1)
    return {"mean": float(values[column].mean()), "card_95_interval": np.quantile(samples, [.025, .975]).tolist()}


def load_runs():
    manifests, records = {}, {}
    for name, folder in RUNS.items():
        path = ROOT / "artifacts/simulations" / folder
        manifest = json.loads((path / "run-manifest.json").read_text())
        contract = manifest["run_contract"]
        if canonical_sha256(contract) != manifest["run_contract_sha256"]:
            raise ValueError("run contract hash mismatch")
        if contract["selection"]["cohort_name"] != "development_2024":
            raise ValueError("only development_2024 may be read")
        manifests[name] = manifest
        with gzip.open(path / "forecast-ledger.jsonl.gz", "rt") as stream:
            rows = [json.loads(line) for line in stream]
        records[name] = {row["fight_id"]: row for row in rows}
        if len(rows) != 229 or len(records[name]) != 229 or len({row["event_id"] for row in rows}) != 30:
            raise ValueError("wrong or duplicate development cohort")
    if records["full"].keys() != records["v2"].keys():
        raise ValueError("unpaired saved runs")
    for key in ("source_sha256", "simulation", "parameter_model"):
        if manifests["full"]["run_contract"][key] != manifests["v2"]["run_contract"][key]:
            raise ValueError(f"run comparison mismatch: {key}")
    return manifests, records


def member_statistics(row, index):
    return {item["statistic"]: float(item["conditional_means"][str(index)]) for item in row["forecast"]["statistic_uncertainty"]}


def build_member_rows(design, manifests, records):
    start = time.monotonic()
    revision = design["historical_source_revision"]
    for name in ("parameters.py", "opponent_audit.py", "engine.py", "domain.py"):
        old = git_bytes(f"{revision}:src/fight_sim/{name}").replace(b"\r\n", b"\n")
        if old != (ROOT / "src/fight_sim" / name).read_bytes().replace(b"\r\n", b"\n"):
            raise ValueError(f"historical implementation changed: {name}")
    contract = manifests["v2"]["run_contract"]
    hashes = contract["source_sha256"]
    fitter = CausalParameterFitter(
        historical_frame("ufc_fights_reported_doubled.csv", hashes["raw"], revision),
        historical_frame("fighter_stats.csv", hashes["profiles"], revision),
        historical_frame("ufc_fight_round_stats_doubled.csv", hashes["round_stats"], revision),
        use_takedown_control_association=True,
    )
    simulator = SimulatorConfig(**contract["simulation"]["simulator_config"])
    rows, causal = [], []
    grouped = pd.DataFrame(records["v2"].values()).groupby(["causal_cutoff_utc", "event_id"], sort=True)
    for position, ((timestamp, event), fights) in enumerate(grouped, 1):
        if time.monotonic() - start > design["maximum_runtime_seconds"]:
            raise TimeoutError("bridge audit exceeded its frozen runtime cap")
        cutoff = pd.Timestamp(timestamp)
        event_seed = 2903 + int(canonical_sha256({"event_id": event, "date": cutoff.isoformat()})[:8], 16)
        config = ParameterFitConfig.historical(bootstrap_members=10, random_seed=event_seed)
        cache_contract = _fit_cache_contract(cutoff=cutoff, config=config, source_sha256=hashes, parameter_model=contract["parameter_model"])
        cache = ROOT / "artifacts/simulations/causal-fit-cache" / f"fit-{canonical_sha256(cache_contract)}.json.gz"
        if not cache.is_file():
            raise FileNotFoundError(f"exact historical fit required: {cache}")
        artifact = load_parameter_artifact(cache)
        if artifact.config != config or pd.Timestamp(artifact.as_of_utc) != cutoff or pd.to_datetime(artifact.trained_through, utc=True) >= cutoff:
            raise ValueError("cached fit violates historical cutoff/configuration")
        event_dates = fitter.raw_fights.groupby("event_id")["date"].max()
        for fight in fights.to_dict("records"):
            specs = build_specs(
                fitter, artifact, red_fighter_id=fight["red_fighter_id"], blue_fighter_id=fight["blue_fighter_id"],
                division=fight["division"], scheduled_rounds=fight["scheduled_rounds"], event_id=event,
                root_seed=f"posterior:2903:{fight['fight_id']}", simulator_base=simulator,
                snapshot_parameter_mode="opponent_adjusted_v2", _artifact_validated=True,
            )
            for spec in specs:
                index = spec.bootstrap_member
                member = artifact.members[index]
                context = member.context_parameters.get(f"{fight['division']}|{fitter._era_key(cutoff, config.era_years)}", member.context_parameters["__global__"])
                effects = fitter._opponent_adjusted_v2_effects(artifact, index, cutoff)
                diag = fitter._opponent_adjusted_v2_diagnostics_cache[(artifact.artifact_sha256, index)]
                if any(event_dates.loc[inner] >= cutoff for inner in diag["inner_event_ids"]):
                    raise ValueError("ridge selection used an outer/future card")
                saved = {name: member_statistics(values[fight["fight_id"]], index) for name, values in records.items()}
                for side in Side:
                    identity = fight[f"{side.value}_fighter_id"]
                    other = fight[f"{side.opponent.value}_fighter_id"]
                    defaults = {key: 0.0 for key in ("strike_pace_offense", "strike_pace_vulnerability", "strike_accuracy_offense", "strike_accuracy_vulnerability")}
                    actor = effects.get(identity, defaults)
                    opponent = effects.get(other, defaults)
                    snapshot = getattr(spec, side.value)
                    params = snapshot.parameters
                    minutes = fight["actual_duration_seconds"] / 60
                    actual_attempts = fight[f"actual_{side.value}_significant_strike_attempts"]
                    actual_landed = fight[f"actual_{side.value}_significant_strikes"]
                    row = {"fight_id": fight["fight_id"], "event_id": event, "date": str(cutoff.date()), "member": index,
                           "side": side.value, "fighter_id": identity, "winner": fight["actual_outcome"].split("_")[0],
                           "actual_attempts": actual_attempts, "actual_landed": actual_landed, "actual_minutes": minutes}
                    conditional_rate = sum(context[f"strike_rate_{phase}"] * context[f"{phase}_phase_share"] for phase in PHASES) * rate_covariate(snapshot, member) * math.exp(actor["strike_pace_offense"])
                    conditional_accuracy = float(expit(logit(context["strike_accuracy"]) + actor["strike_accuracy_offense"] + opponent["strike_accuracy_vulnerability"]))
                    row.update(conditional_attempts=conditional_rate * minutes, conditional_accuracy=conditional_accuracy,
                               conditional_landed=conditional_rate * minutes * conditional_accuracy)
                    row["snapshot_attempts"] = sum(getattr(params, f"strike_rate_{phase}") * context[f"{phase}_phase_share"] for phase in PHASES) * minutes
                    row["snapshot_landed"] = row["snapshot_attempts"] * conditional_accuracy
                    row["snapshot_accuracy"] = conditional_accuracy
                    row["actor_accuracy_slope"], row["opponent_accuracy_slope"] = accuracy_sensitivity(spec, side, context, actor["strike_accuracy_offense"], opponent["strike_accuracy_vulnerability"])
                    row["context_reweight_ratio"] = sum(context[f"strike_rate_{phase}"] * getattr(params, f"{phase}_phase_share") for phase in PHASES) / sum(context[f"strike_rate_{phase}"] * context[f"{phase}_phase_share"] for phase in PHASES)
                    phase_rates = neutral_phase_rates(spec, side)
                    proxy = {"distance": context["distance_phase_share"], "clinch": context["clinch_phase_share"],
                             "ground_top": context["ground_phase_share"] / 2, "ground_bottom": context["ground_phase_share"] / 2, "scramble": 0}
                    attempts, landed = integrate(phase_rates, proxy)
                    row.update(engine_proxy_attempts=attempts * minutes, engine_proxy_landed=landed * minutes, engine_proxy_accuracy=landed / attempts)
                    for name, stats in saved.items():
                        duration = sum(stats[f"{phase.value}_time_seconds"] for phase in Phase)
                        if duration <= 0:
                            raise ValueError("empty saved member exposure")
                        row[f"{name}_duration_seconds"] = duration
                        for quantity, statistic in (("attempts", "significant_strike_attempts"), ("landed", "significant_strikes")):
                            row[f"{name}_{quantity}"] = stats[f"{side.value}_{statistic}"]
                            row[f"{name}_duration_standardized_{quantity}"] = row[f"{name}_{quantity}"] * fight["actual_duration_seconds"] / duration
                        row[f"{name}_accuracy"] = row[f"{name}_landed"] / max(row[f"{name}_attempts"], 1e-12)
                        row[f"{name}_duration_standardized_accuracy"] = row[f"{name}_accuracy"]
                    stats = saved["v2"]
                    duration = row["v2_duration_seconds"]
                    control = stats["red_control_seconds"] + stats["blue_control_seconds"]
                    # Ledger summaries round each path's seconds independently.
                    # Preserve its top/bottom ratio and its ground-time total.
                    if abs(control - stats["ground_time_seconds"]) > 1.5:
                        raise ValueError("saved control/ground exposure differs beyond rounding")
                    top_share = stats[f"{side.value}_control_seconds"] / control if control > 0 else .5
                    top = top_share * stats["ground_time_seconds"]
                    row["control_rounding_difference_seconds"] = control - stats["ground_time_seconds"]
                    observed_weights = {"distance": stats["distance_time_seconds"] / duration, "clinch": stats["clinch_time_seconds"] / duration,
                                        "ground_top": top / duration, "ground_bottom": (stats["ground_time_seconds"] - top) / duration,
                                        "scramble": stats["scramble_time_seconds"] / duration}
                    attempts, landed = integrate(phase_rates, observed_weights)
                    row.update(engine_saved_phase_attempts=attempts * minutes, engine_saved_phase_landed=landed * minutes, engine_saved_phase_accuracy=landed / attempts)
                    rows.append(row)
        causal.append({"event_id": event, "cutoff": cutoff.isoformat(), "trained_through": artifact.trained_through,
                       "artifact_sha256": artifact.artifact_sha256, "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
                       "member_selection": [fitter._opponent_adjusted_v2_diagnostics_cache[(artifact.artifact_sha256, index)] for index in range(10)]})
        print(f"Card {position}/30: {len(fights)} fights, {time.monotonic()-start:.1f}s", flush=True)
    return pd.DataFrame(rows), causal, time.monotonic() - start


def summarize(members, records):
    keys = ["fight_id", "event_id", "date", "side", "fighter_id", "winner"]
    sides = members.groupby(keys, as_index=False).mean(numeric_only=True).drop(columns="member")
    stages = ["conditional", "snapshot", "engine_proxy", "engine_saved_phase", "v2_duration_standardized", "v2", "full_duration_standardized", "full"]
    old = pd.read_csv(ROOT / "artifacts/simulations/opponent-adjustment-bout-clustered-predictions-20260827.csv")
    original_report = json.loads((ROOT / "artifacts/simulations/opponent-adjustment-bout-clustered-audit-20260827.json").read_text())
    saved_hash = original_report.pop("report_sha256")
    if canonical_sha256(original_report) != saved_hash:
        raise ValueError("original observation report hash mismatch")
    if original_report["source_sha256"] != json.loads((ROOT / "artifacts/simulations" / RUNS["v2"] / "run-manifest.json").read_text())["run_contract"]["source_sha256"]:
        raise ValueError("observation audit source mismatch")
    for label in ("context", "marginal", "opponent"):
        column = "opponent_adjusted_prediction" if label == "opponent" else f"{label}_prediction"
        prediction = old.pivot(index=["fight_id", "fighter_id"], columns="target", values=column)
        selected = prediction.reindex(pd.MultiIndex.from_frame(sides[["fight_id", "fighter_id"]]))
        if selected[["strike_pace", "strike_accuracy"]].isna().any().any():
            raise ValueError("saved observation audit does not cover all fighter sides")
        sides[f"original_{label}_attempts"] = selected["strike_pace"].to_numpy() * sides["actual_minutes"]
        sides[f"original_{label}_accuracy"] = selected["strike_accuracy"].to_numpy()
        sides[f"original_{label}_landed"] = sides[f"original_{label}_attempts"] * sides[f"original_{label}_accuracy"]
        stages.append(f"original_{label}")
    scores = {}
    for stage in stages:
        sides[f"{stage}_pace_nll"] = _negative_log_likelihood(sides.actual_attempts.to_numpy(), np.ones(len(sides)), sides[f"{stage}_attempts"].to_numpy(), kind="rate")
        sides[f"{stage}_accuracy_nll"] = _negative_log_likelihood(sides.actual_landed.to_numpy(), sides.actual_attempts.to_numpy(), sides[f"{stage}_accuracy"].to_numpy(), kind="binomial")
        fights = sides.pivot(index=["fight_id", "event_id", "winner"], columns="side", values=f"{stage}_landed").reset_index()
        difference = fights.red - fights.blue
        fights["correct"] = np.where(difference == 0, .5, ((difference > 0) == fights.winner.eq("red")).astype(float))
        fights = fights[fights.winner.isin(["red", "blue"])]
        scores[stage] = {"attempt_bias_per_side": float((sides[f"{stage}_attempts"] - sides.actual_attempts).mean()),
                         "landed_bias_per_side": float((sides[f"{stage}_landed"] - sides.actual_landed).mean()),
                         "pace_nll_per_side": float(sides[f"{stage}_pace_nll"].mean()), "accuracy_nll_per_side": float(sides[f"{stage}_accuracy_nll"].mean()),
                         "winner_ranking": {"fights": len(fights), "correct_with_half_credit_ties": float(fights.correct.sum()), **card_interval(fights, "correct")}}
    comparisons = {}
    for candidate, baseline in (("conditional", "original_opponent"), ("snapshot", "conditional"), ("engine_proxy", "conditional"), ("engine_saved_phase", "engine_proxy"), ("v2_duration_standardized", "engine_saved_phase"), ("v2", "v2_duration_standardized"), ("v2", "full"), ("original_opponent", "original_marginal")):
        comparisons[f"{candidate}_minus_{baseline}"] = {}
        for metric in ("attempts", "landed", "pace_nll", "accuracy_nll"):
            differences = sides[["event_id"]].copy()
            differences["difference"] = sides[f"{candidate}_{metric}"] - sides[f"{baseline}_{metric}"]
            comparisons[f"{candidate}_minus_{baseline}"][metric] = card_interval(differences, "difference")
        ranks = sides.pivot(index=["fight_id", "event_id", "winner"], columns="side", values=[f"{candidate}_landed", f"{baseline}_landed"])
        valid = ranks.index.get_level_values("winner").isin(["red", "blue"])
        ranks = ranks[valid]
        winner = ranks.index.get_level_values("winner") == "red"
        def correct(label):
            diff = ranks[(f"{label}_landed", "red")] - ranks[(f"{label}_landed", "blue")]
            return np.where(diff == 0, .5, ((diff > 0) == winner).astype(float))
        differences = pd.DataFrame({"event_id": ranks.index.get_level_values("event_id"), "difference": correct(candidate)-correct(baseline)})
        comparisons[f"{candidate}_minus_{baseline}"]["winner_ranking"] = card_interval(differences, "difference")
    distributions = {}
    for name, values in records.items():
        summary = {quantity: [] for quantity in ("attempts", "landed")}
        for record in values.values():
            lookup = {item["statistic"]: item for item in record["forecast"]["statistic_distributions"]}
            for quantity, statistic in (("attempts", "total_significant_strike_attempts"), ("landed", "total_significant_strikes")):
                distribution = lookup[statistic]
                support = np.array([item["value"] for item in distribution["counts"]])
                mass = np.array([item["count"] for item in distribution["counts"]]) / distribution["total_paths"]
                if not math.isclose(mass.sum(), 1.0):
                    raise ValueError("saved distribution has invalid mass")
                observed = record[f"actual_{statistic}"]
                summary[quantity].append({"event_id": record["event_id"], "bias": float(support @ mass - observed), "crps": _crps(observed, support, mass)})
        distributions[name] = {quantity: {metric: card_interval(pd.DataFrame(values), metric) for metric in ("bias", "crps")} for quantity, values in summary.items()}
    members = members.assign(snapshot_to_conditional_rate_ratio=members.snapshot_attempts / members.conditional_attempts)
    return sides, {"stage_scores": scores, "paired_card_intervals": comparisons, "saved_pooled_distributions": distributions,
                   "observation_report_sha256": saved_hash,
                   "mapping": {key: {"mean": float(members[key].mean()), "p05": float(members[key].quantile(.05)), "p95": float(members[key].quantile(.95))} for key in ("actor_accuracy_slope", "opponent_accuracy_slope", "context_reweight_ratio", "snapshot_to_conditional_rate_ratio")},
                   "duration_bias_seconds": {name: float((sides[f"{name}_duration_seconds"] - sides.actual_minutes*60).mean()) for name in RUNS}}


def main():
    design = json.loads((OUTPUT / "strike_bridge_design.json").read_text())
    manifests, records = load_runs()
    members, causal, seconds = build_member_rows(design, manifests, records)
    if len(members) != 229 * 2 * 10 or members.duplicated(["fight_id", "side", "member"]).any():
        raise ValueError("member coverage incomplete")
    LOCAL.mkdir(parents=True, exist_ok=True)
    member_path = LOCAL / "member_rows.csv"
    members.to_csv(member_path, index=False)
    sides, report = summarize(members, records)
    sides.to_csv(LOCAL / "side_rows.csv", index=False)
    report.update(schema_version=1, research_only=True, production_changed=False, new_trajectories=0,
                  cohort="development_2024", fights=229, cards=30, member_side_rows=len(members),
                  runtime_seconds=seconds, design_sha256=canonical_sha256(design),
                  source_run_contracts={key: value["run_contract_sha256"] for key, value in manifests.items()},
                  member_rows_sha256=hashlib.sha256(member_path.read_bytes()).hexdigest(),
                  source_sha256=manifests["v2"]["run_contract"]["source_sha256"],
                  historical_source_revision=design["historical_source_revision"], causal_checks=causal,
                  limitations=design["limitations"], locked_cohorts_opened=[])
    report["input_file_sha256"] = {
        str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [ROOT / "artifacts/simulations/opponent-adjustment-bout-clustered-predictions-20260827.csv"]
        + [ROOT / "artifacts/simulations" / folder / "forecast-ledger.jsonl.gz" for folder in RUNS.values()]
    }
    (OUTPUT / "strike_bridge_results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"fights":229, "cards":30, "seconds":seconds, "mapping":report["mapping"]}, indent=2))


if __name__ == "__main__":
    main()
