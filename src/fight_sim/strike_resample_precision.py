"""Frozen two-stream training-resample precision check; no fight trajectories.

Run with PYTHONPATH=src: python -m fight_sim.strike_resample_precision.
Each card can run in a separate ordinary Python process. All estimator code,
sources, random streams, sample counts and decision rules are pinned up front.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import time

import numpy as np
import pandas as pd

from .opponent_audit import ChronologicalOpponentRidgeSelector, _negative_log_likelihood
from .parameters import canonical_sha256
from .strike_bridge_audit import ROOT, OUTPUT, card_interval, git_bytes, historical_frame, load_runs
from .strike_handoff import STRIKE_TARGETS
from .strike_handoff_audit import forecast_arm, maximum_reproduction_error, validate_saved_inputs

LOCAL = ROOT / "artifacts/simulations/strike-resample-precision-20261001"
DESIGN_PATH = OUTPUT / "strike_resample_precision_design.json"
REFERENCE = "observation_reference"
V2 = "simulator_bootstrap_covariates"
FINAL_STREAMS = ("original_seed_100", "independent_seed_100")


def member_seeds(root, event_id, cutoff, count):
    event_hash = int(canonical_sha256({"event_id": str(event_id), "date": pd.Timestamp(cutoff).isoformat()})[:8], 16)
    return np.random.SeedSequence(int(root) + event_hash).generate_state(count, dtype=np.uint64)


def card_weights(training, seed, cutoff):
    dates = pd.to_datetime(training.date, utc=True)
    if dates.isna().any() or dates.ge(pd.to_datetime(cutoff, utc=True)).any():
        raise ValueError("training cards must be strictly earlier")
    events = np.array(sorted(training.event_id.unique()), dtype=object)
    if not len(events):
        raise ValueError("no training cards")
    rng = np.random.Generator(np.random.PCG64DXSM(int(seed)))
    unique, counts = np.unique(rng.choice(events, size=len(events), replace=True), return_counts=True)
    return training.event_id.map(dict(zip(unique, counts))).fillna(0).to_numpy(float)


def _card_worker(job):
    training, test = job["training"], job["test"]
    cutoff, selected, design = job["cutoff"], job["selected"], job["design"]
    prediction = forecast_arm(training, test, np.ones(len(training)), selected, cutoff)
    reference_error = maximum_reproduction_error(
        np.column_stack([prediction[target.name] for target in STRIKE_TARGETS]),
        job["reference"], "original observation reference",
    )
    identity = test[["fight_id", "fighter_id", "event_id", "date", "side", "winner"]].copy()
    identity["date"] = identity.date.dt.strftime("%Y-%m-%d")
    identity["actual_minutes"] = test.fight_seconds.to_numpy()/60
    identity["actual_attempts"] = test.sig_strikes_attempts.to_numpy()
    identity["actual_landed"] = test.sig_strikes_landed.to_numpy()
    output, seed_hashes = [], {}
    prefix_error = 0.0
    for stream, root in design["streams"].items():
        seeds = member_seeds(root, job["event_id"], cutoff, design["members_per_stream"])
        seed_hashes[stream] = canonical_sha256([int(seed) for seed in seeds])
        for member, seed in enumerate(seeds):
            if time.monotonic() > job["deadline"]:
                raise TimeoutError("fixed precision-check runtime cap reached")
            weights = card_weights(training, seed, cutoff)
            prediction = forecast_arm(training, test, weights, selected, cutoff)
            pace, accuracy = prediction["strike_pace"], prediction["strike_accuracy"]
            if stream == "original_seed" and member < 10:
                prefix_error = max(prefix_error, maximum_reproduction_error(
                    np.column_stack([pace, accuracy, pace*accuracy]), job["old_prefix"][member],
                    f"original ten-member prefix, member {member}",
                ))
            frame = identity.copy()
            # Mixing signed/unsigned uint64 columns during concatenation can
            # silently round seeds through float64. Preserve decimal text.
            frame["stream"], frame["member"], frame["member_seed"] = stream, member, str(int(seed))
            frame["pace"], frame["accuracy"], frame["landed_rate"] = pace, accuracy, pace*accuracy
            output.append(frame)
    return {
        "members": pd.concat(output, ignore_index=True),
        "reference_error": reference_error, "prefix_error": prefix_error,
        "provenance": {
            "event_id": job["event_id"], "cutoff": cutoff.isoformat(),
            "training_latest": training.date.max().isoformat(), "training_cards": int(training.event_id.nunique()),
            "training_fights": int(training.fight_id.nunique()), "inner_cards": list(job["inner_ids"]),
            "selected_ridge": {target.name: selected[(target.name, "opponent_adjusted")] for target in STRIKE_TARGETS},
            "member_seed_hashes": seed_hashes,
        },
    }


def validate_inputs():
    prior, _old_bridge, original = validate_saved_inputs()
    handoff = json.loads(git_bytes("b19e6280:audit/simulation/2026-10-01/strike_handoff_results.json"))
    content = {key: value for key, value in handoff.items() if key != "report_sha256"}
    if canonical_sha256(content) != handoff["report_sha256"]:
        raise ValueError("handoff report hash invalid")
    for filename in ("strike_handoff.py", "strike_handoff_audit.py"):
        pinned = git_bytes(f"b19e6280:src/fight_sim/{filename}").replace(b"\r\n", b"\n")
        if pinned != (ROOT / "src/fight_sim" / filename).read_bytes().replace(b"\r\n", b"\n"):
            raise ValueError("the precision experiment must not alter the estimator")
    base = ROOT / "artifacts/simulations/strike-handoff-20261001"
    for filename, expected in handoff["output_sha256"].items():
        if hashlib.sha256((base / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"handoff output changed: {filename}")
    dtypes = {key: str for key in ("fight_id", "fighter_id", "event_id")}
    old_members = pd.read_csv(base / "member_rows.csv", dtype=dtypes)
    old_sides = pd.read_csv(base / "side_rows.csv", dtype=dtypes)
    return prior, handoff, original, old_members, old_sides


def prepare_jobs(design, prior, original, old_members, records, deadline):
    raw = historical_frame("ufc_fights_reported_doubled.csv", prior["source_sha256"]["raw"], "59a16cd2")
    selector = ChronologicalOpponentRidgeSelector(raw)
    frame = selector.frame
    event_dates = frame.groupby("event_id").date.max()
    saved_cards = {row["event_id"]: row for row in prior["causal_checks"]}
    reference = original.pivot(index=["fight_id", "fighter_id"], columns="target", values="opponent_adjusted_prediction")
    old = old_members[old_members.arm.eq("observation_bootstrap")].set_index(["fight_id", "fighter_id", "member"])
    jobs = []
    events = pd.DataFrame(records["v2"].values()).groupby(["causal_cutoff_utc", "event_id"], sort=True)
    for (timestamp, event), fights in events:
        if time.monotonic() > deadline:
            raise TimeoutError("precision-check preparation exceeded runtime cap")
        cutoff = pd.Timestamp(timestamp)
        training = frame[frame.date.lt(cutoff)].copy()
        test = frame[frame.fight_id.isin(fights.fight_id)].sort_values(["fight_id", "fighter_id"]).copy()
        if len(test) != 2*len(fights) or not test.date.eq(cutoff).all() or not test.event_id.eq(event).all():
            raise ValueError("unexpected evaluation cohort membership")
        selected, inner_ids = selector.selected_for_cutoff(cutoff)
        saved = saved_cards[event]["member_selection"][0]
        if list(inner_ids) != saved["inner_event_ids"] or any(event_dates.loc[inner] >= cutoff for inner in inner_ids):
            raise ValueError("changed or noncausal inner selection cards")
        for target in STRIKE_TARGETS:
            if selected[(target.name, "opponent_adjusted")] != saved[f"selected_{target.name}_ridge"]:
                raise ValueError("shrinkage selection changed")
        test["side"] = ["red" if identity == records["v2"][fight]["red_fighter_id"] else "blue"
                        for identity, fight in zip(test.fighter_id, test.fight_id)]
        test["winner"] = [records["v2"][fight]["actual_outcome"].split("_")[0] for fight in test.fight_id]
        indices = pd.MultiIndex.from_frame(test[["fight_id", "fighter_id"]])
        prefix = {member: old.reindex(pd.MultiIndex.from_tuples(
            [(fight, identity, member) for fight, identity in indices], names=old.index.names
        ))[["pace", "accuracy", "landed_rate"]].to_numpy() for member in range(10)}
        jobs.append({"event_id": event, "cutoff": cutoff, "training": training, "test": test,
                     "selected": selected, "inner_ids": inner_ids, "design": design, "deadline": deadline,
                     "reference": reference.reindex(indices)[[target.name for target in STRIKE_TARGETS]].to_numpy(),
                     "old_prefix": prefix})
    print(f"Prepared {len(jobs)} causal card cutoffs and verified all saved inner selections.", flush=True)
    return jobs


def summarize_prefixes(members, old_sides, design):
    keys = ["fight_id", "fighter_id", "event_id", "date", "side", "winner"]
    values = ["actual_minutes", "actual_attempts", "actual_landed", "pace", "accuracy", "landed_rate"]
    frames = []
    for stream in design["streams"]:
        for count in design["diagnostic_prefix_sizes"]:
            frame = members[members.stream.eq(stream) & members.member.lt(count)].groupby(keys, as_index=False)[values].mean()
            frame["arm"] = f"{stream}_{count:03d}"
            frames.append(frame)
    pooled = members.groupby(keys, as_index=False)[values].mean()
    pooled["arm"] = "pooled_200_descriptive"
    frames.append(pooled)
    baseline_arms = (REFERENCE, "original_marginal", V2, "observation_bootstrap")
    for arm in baseline_arms:
        frames.append(old_sides[old_sides.arm.eq(arm)][keys + values + ["arm"]].copy())
    sides = pd.concat(frames, ignore_index=True)
    sides["pace_nll"] = _negative_log_likelihood(sides.actual_attempts.to_numpy(), sides.actual_minutes.to_numpy(), sides.pace.to_numpy(), kind="rate")
    sides["accuracy_nll"] = _negative_log_likelihood(sides.actual_landed.to_numpy(), sides.actual_attempts.to_numpy(), sides.accuracy.to_numpy(), kind="probability")
    scores, rankings = {}, {}
    for arm, group in sides.groupby("arm"):
        ranking = group.pivot(index=["fight_id", "event_id", "winner"], columns="side", values="landed_rate").reset_index()
        ranking = ranking[ranking.winner.isin(["red", "blue"])].copy()
        ranking["differential"] = ranking.red-ranking.blue
        ranking["choice"] = np.sign(ranking.differential).astype(int)
        ranking["correct"] = np.where(ranking.choice.eq(0), .5, (ranking.choice.eq(1) == ranking.winner.eq("red")).astype(float))
        rankings[arm] = ranking
        scores[arm] = {"pace_nll": float(group.pace_nll.mean()), "accuracy_nll": float(group.accuracy_nll.mean()),
                       "attempt_bias_per_side": float((group.pace*group.actual_minutes-group.actual_attempts).mean()),
                       "strike_leader_wins": float(ranking.correct.sum()), "decisive_fights": len(ranking),
                       "ranking_card_interval": card_interval(ranking, "correct")}
    comparisons = {}
    for candidate in (*FINAL_STREAMS, "pooled_200_descriptive"):
        current = sides[sides.arm.eq(candidate)].set_index(["fight_id", "fighter_id"])
        for baseline in baseline_arms:
            other = sides[sides.arm.eq(baseline)].set_index(["fight_id", "fighter_id"]).reindex(current.index)
            comparison = {}
            for metric in ("pace_nll", "accuracy_nll"):
                differences = pd.DataFrame({"event_id": current.event_id, "difference": current[metric]-other[metric]})
                comparison[metric] = card_interval(differences, "difference")
            ranking = rankings[candidate].set_index("fight_id")
            old_ranking = rankings[baseline].set_index("fight_id").reindex(ranking.index)
            differences = pd.DataFrame({"event_id": ranking.event_id, "difference": ranking.correct-old_ranking.correct})
            comparison["ranking"] = card_interval(differences, "difference")
            comparisons[f"{candidate}_minus_{baseline}"] = comparison
    first = sides[sides.arm.eq(FINAL_STREAMS[0])].set_index(["fight_id", "fighter_id"])
    second = sides[sides.arm.eq(FINAL_STREAMS[1])].set_index(["fight_id", "fighter_id"]).reindex(first.index)
    reference = sides[sides.arm.eq(REFERENCE)].set_index(["fight_id", "fighter_id"]).reindex(first.index)
    first_rank = rankings[FINAL_STREAMS[0]].set_index("fight_id")
    second_rank = rankings[FINAL_STREAMS[1]].set_index("fight_id").reindex(first_rank.index)
    stability = {
        "strike_leader_agreement": float(first_rank.choice.eq(second_rank.choice).mean()),
        "strike_leader_disagreements": int(first_rank.choice.ne(second_rank.choice).sum()),
        "accuracy_rms_difference": float(np.sqrt(np.mean(np.square(first.accuracy-second.accuracy)))),
        "pace_relative_rms_difference": float(np.sqrt(np.mean(np.square((first.pace-second.pace)/reference.pace)))),
    }
    return sides, scores, comparisons, stability


def decision(scores, stability, design):
    rule = design["stability_rule"]
    stable = {
        "strike_leader_agreement": stability["strike_leader_agreement"] >= rule["minimum_strike_leader_agreement"],
        "accuracy_rms_difference": stability["accuracy_rms_difference"] <= rule["maximum_accuracy_rms_difference"],
        "pace_relative_rms_difference": stability["pace_relative_rms_difference"] <= rule["maximum_pace_relative_rms_difference"],
    }
    performance = {}
    for stream in FINAL_STREAMS:
        candidate = scores[stream]
        performance[stream] = {
            "preserves_reference_winner_count": candidate["strike_leader_wins"] >= design["performance_rule"]["minimum_correct_strike_leaders"],
            "complete_decisive_cohort": candidate["decisive_fights"] == design["performance_rule"]["decisive_fights"],
            **{f"lower_{metric}_than_{baseline}": candidate[metric] < scores[baseline][metric] - 1e-6
               for baseline in (V2, "original_marginal") for metric in ("pace_nll", "accuracy_nll")},
        }
    advance = all(stable.values()) and all(all(values.values()) for values in performance.values())
    return {"stability_checks": stable, "performance_checks": performance,
            "candidate_advances_to_simulation_screen": advance}


def differential_precision(members):
    paired = members.pivot(index=["fight_id", "event_id", "stream", "member"], columns="side", values="landed_rate")
    paired["difference"] = paired.red-paired.blue
    grouped = paired.reset_index().groupby(["fight_id", "event_id", "stream"]).difference.agg(["mean", "std", "count"]).reset_index()
    grouped["standard_error"] = grouped["std"]/np.sqrt(grouped["count"])
    grouped["sign_unresolved_by_normal_95_mc_interval"] = grouped["mean"].abs() <= 1.96*grouped.standard_error
    return grouped


def verify_seed_storage(members, provenance, design):
    expected = {}
    hashes = 0
    for card in provenance:
        for stream, root in design["streams"].items():
            seeds = [int(value) for value in member_seeds(root, card["event_id"], card["cutoff"], design["members_per_stream"])]
            if canonical_sha256(seeds) != card["member_seed_hashes"][stream]:
                raise ValueError("member seed sequence hash changed")
            hashes += 1
            for member, seed in enumerate(seeds):
                expected[(card["event_id"], stream, member)] = str(seed)
    for event, stream, member, value in members[["event_id", "stream", "member", "member_seed"]].itertuples(index=False, name=None):
        if not isinstance(value, str) or value != expected[(event, stream, member)]:
            raise ValueError("member seed lost integer precision")
    return {"exact_seed_strings": len(members), "seed_sequence_hashes_verified": hashes}


def main():
    start = time.monotonic()
    design = json.loads(DESIGN_PATH.read_text())
    if design["cohort"] != "development_2024" or design["members_per_stream"] != 100 or design["streams"] != {"original_seed": 2903, "independent_seed": 20261001}:
        raise ValueError("unexpected frozen precision-check design")
    deadline = start + design["maximum_runtime_seconds"]
    prior, handoff, original, old_members, old_sides = validate_inputs()
    manifests, records = load_runs()
    jobs = prepare_jobs(design, prior, original, old_members, records, deadline)
    frames, provenance = [], []
    errors = {"original_reference": 0.0, "original_ten_member_prefix": 0.0}
    with ProcessPoolExecutor(max_workers=design["workers"]) as pool:
        futures = [pool.submit(_card_worker, job) for job in jobs]
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            frames.append(result["members"])
            provenance.append(result["provenance"])
            errors["original_reference"] = max(errors["original_reference"], result["reference_error"])
            errors["original_ten_member_prefix"] = max(errors["original_ten_member_prefix"], result["prefix_error"])
            print(f"Completed {completed}/30 cards at 100 members in both streams; {time.monotonic()-start:.1f}s", flush=True)
    members = pd.concat(frames, ignore_index=True).sort_values(["date", "event_id", "fight_id", "fighter_id", "stream", "member"], kind="stable")
    if len(members) != 229*2*100*2 or members.duplicated(["fight_id", "fighter_id", "stream", "member"]).any():
        raise ValueError("incomplete or duplicated precision study")
    seed_storage = verify_seed_storage(members, provenance, design)
    sides, scores, comparisons, stability = summarize_prefixes(members, old_sides, design)
    precision = differential_precision(members)
    report = decision(scores, stability, design)
    LOCAL.mkdir(parents=True, exist_ok=True)
    for name, frame in (("member_rows", members), ("side_rows", sides), ("differential_precision", precision)):
        frame.to_csv(LOCAL / f"{name}.csv", index=False)
    report.update(schema_version=1, research_only=True, production_changed=False, new_trajectories=0,
                  locked_cohorts_opened=[], cohort="development_2024", fights=229, cards=30,
                  members_per_stream=100, streams=design["streams"], member_side_rows=len(members),
                  design_sha256=canonical_sha256(design), handoff_report_sha256=handoff["report_sha256"],
                  historical_source_sha256=prior["source_sha256"],
                  elapsed_seconds=time.monotonic()-start, reproduction_maximum_errors=errors,
                  seed_storage_verification=seed_storage,
                  scores=scores, paired_card_intervals=comparisons, stability=stability,
                  differential_precision={stream: {"fights": len(group), "mean_standard_error": float(group.standard_error.mean()),
                                                   "unresolved_signs": int(group.sign_unresolved_by_normal_95_mc_interval.sum())}
                                          for stream, group in precision.groupby("stream")},
                  causal_checks=sorted(provenance, key=lambda row: (row["cutoff"], row["event_id"])),
                  limitations=design["limitations"],
                  output_sha256={f"{name}.csv": hashlib.sha256((LOCAL / f"{name}.csv").read_bytes()).hexdigest()
                                 for name in ("member_rows", "side_rows", "differential_precision")})
    report["report_sha256"] = canonical_sha256(report)
    (OUTPUT / "strike_resample_precision_results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: report[key] for key in ("scores", "stability", "stability_checks", "performance_checks", "candidate_advances_to_simulation_screen", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
