"""Frozen small-family nonlinear experiment; never edits live model artifacts.

Run: .venv/Scripts/python.exe src/evaluate_nonlinear_features.py
2021-22: descriptive residuals. 2023-24: choose one feature family.
2025-26: evaluate only that frozen choice against the baseline.
"""
import copy
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fight_predictor.nonlinear_research import GROUPS, NonlinearResearchBuilder
from fight_predictor.point_in_time import PointInTimeDatasetBuilder, TemporalFightPredictor
from evaluate_style_matchup_challenger import (
    _align_predictions, _metric, _validate_candidate_baseline_features,
    event_block_difference_interval,
)
from market_tracker._storage import atomic_write_text

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src/content/data"
SPEC = {
    "version": 1, "production_enabled": False,
    "discovery_years": [2021, 2022], "selection_years": [2023, 2024],
    "test_years": [2025, 2026], "groups": GROUPS,
    "selection_rule": "Lowest selection log loss among families improving both log loss and Brier in EACH selection year; otherwise baseline. Ties within 1e-6 favor fewer terms, then name.",
    "training": "Same production ten-year window, five C values, four chronological internal splits, and symmetric calibration; all settings trained on earlier fights only.",
    "seed": 48, "bootstrap_replicates": 4000,
    "limitations": [
        "These historical years have been examined in earlier project research; this is not a pristine project-wide holdout or a prospective betting record.",
        "Historical features are replayed from current corrected sources and profiles, not archived as-of snapshots.",
        "Residual bins are descriptive, with no p-values or claims of independent discoveries. Four prespecified families compete on selection years; only the chosen family reaches the final comparison.",
        "Striking absorption rates proxy style and exposure, not cumulative medical damage. Reach/activity interactions are not direct opponent-style measurements.",
        "Probability improvements alone do not establish an edge over bookmaker odds or profitability.",
    ],
}


def write_json(path, value):
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def metrics(frame, column="model_probability"):
    result = _metric(frame, column)
    p = np.clip(frame[column].to_numpy(), 1e-8, 1 - 1e-8)
    model = LogisticRegression(C=1e6, max_iter=10000).fit(
        np.log(p / (1 - p)).reshape(-1, 1), frame.target)
    result.update(calibration_intercept=float(model.intercept_[0]),
                  calibration_slope=float(model.coef_[0, 0]))
    return result


def select_family(comparisons):
    baseline = comparisons["baseline"]
    eligible = []
    for name, item in comparisons.items():
        if name == "baseline":
            continue
        if all(item["per_year"][str(y)][m] < baseline["per_year"][str(y)][m]
               for y in SPEC["selection_years"] for m in ("log_loss", "brier")):
            eligible.append(name)
    if not eligible:
        return "baseline"
    best = min(comparisons[n]["overall"]["log_loss"] for n in eligible)
    tied = [n for n in eligible if comparisons[n]["overall"]["log_loss"] <= best + 1e-6]
    return min(tied, key=lambda n: (len(GROUPS[n]), n))


def residual_bins(predictions, points, columns):
    frame = predictions.merge(points[["fight_id", *columns]], on="fight_id", validate="one_to_one")
    frame["residual"] = frame.target - frame.model_probability
    result = {}
    for column in ["model_probability", *columns]:
        bins = pd.qcut(frame[column], 8, duplicates="drop")
        rows = []
        for _, group in frame.groupby(bins, observed=True):
            rows.append({"mean_input": float(group[column].mean()), "fights": len(group),
                         "mean_residual": float(group.residual.mean()),
                         "mean_probability": float(group.model_probability.mean()),
                         "win_rate": float(group.target.mean())})
        result[column] = rows
    return result


def evaluate():
    paths = [DATA / "processed" / n for n in (
        "ufc_fights_reported_doubled.csv", "fighter_stats.csv", "ufc_fights_point_in_time.csv")]
    paths += [Path(__file__), ROOT / "src/fight_predictor/nonlinear_research.py",
              ROOT / "src/fight_predictor/point_in_time.py", ROOT / "src/evaluate_style_matchup_challenger.py"]
    hashes = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in paths}
    key = sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    output = ROOT / "artifacts/nonlinear_research" / key
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "protocol.json", {**SPEC, "source_sha256": hashes})

    def cached(name, produce):
        path = output / (name + ".csv")
        digest = path.with_suffix(".sha256")
        if path.exists() and digest.exists() and sha256(path.read_bytes()).hexdigest() == digest.read_text():
            return pd.read_csv(path, low_memory=False)
        frame = produce()
        atomic_write_text(path, frame.to_csv(index=False, float_format="%.17g"))
        atomic_write_text(digest, sha256(path.read_bytes()).hexdigest())
        return frame

    raw, profiles, base_points = [pd.read_csv(p, low_memory=False) for p in paths[:3]]
    base = PointInTimeDatasetBuilder(raw, profiles)
    extended = NonlinearResearchBuilder(raw, profiles)
    print("Replaying pre-fight histories and checking all 82 original features", flush=True)
    points = cached("features", extended.build)
    _validate_candidate_baseline_features(base_points, points, base.feature_columns)
    if points.fight_id.duplicated().any():
        raise ValueError("Each fight must appear exactly once")

    def predict(name, years):
        builder = copy.copy(base)
        builder.feature_columns = tuple(base.feature_columns) + tuple(
            term + "_diff" for term in GROUPS.get(name, ()))
        frames = []
        for year in years:
            print(f"Evaluating {name}: {year}", flush=True)
            frame = cached(f"{name}_{year}", lambda: TemporalFightPredictor(points, builder).walk_forward_predictions((year,)))
            if not (pd.to_datetime(frame.training_through) < pd.to_datetime(frame.date)).all():
                raise ValueError("Training must precede every evaluated fight")
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    discovery = predict("baseline", SPEC["discovery_years"])
    terms = [t + "_diff" for values in GROUPS.values() for t in values]
    diagnostics = residual_bins(discovery, points,
                               ["age_diff", "reach_diff", "career_fights_log_diff", "days_since_fight_log_diff", *terms])
    write_json(output / "discovery_residual_bins.json", diagnostics)
    selection = {}
    for name in ["baseline", *GROUPS]:
        frame = predict(name, SPEC["selection_years"])
        selection[name] = {"overall": metrics(frame), "fights": len(frame),
                           "per_year": {str(y): metrics(g) for y, g in frame.groupby("evaluation_year")}}
    chosen = select_family(selection)
    # Persist the decision before any final-test prediction is requested.
    write_json(output / "frozen_selection.json", {"chosen": chosen, "selection": selection})
    print(f"Frozen choice before final evaluation: {chosen}", flush=True)
    baseline_test = predict("baseline", SPEC["test_years"])
    selected_test = baseline_test if chosen == "baseline" else predict(chosen, SPEC["test_years"])
    paired = _align_predictions(baseline_test, selected_test, probability_name="candidate_probability")
    atomic_write_text(output / "final_predictions.csv", paired.to_csv(index=False, float_format="%.17g"))
    interval = event_block_difference_interval(paired, "candidate_probability", "model_probability", seed=48, replicates=4000)
    subgroup_frame = paired.merge(points[["fight_id", "age_diff", "career_fights_log_diff"]],
                                  on="fight_id", validate="one_to_one")
    subgroup_masks = {
        "age_gap_under_5_years": subgroup_frame.age_diff.abs() < 5,
        "age_gap_at_least_5_years": subgroup_frame.age_diff.abs() >= 5,
        "experience_log_gap_under_1": subgroup_frame.career_fights_log_diff.abs() < 1,
        "experience_log_gap_at_least_1": subgroup_frame.career_fights_log_diff.abs() >= 1,
    }
    subgroups = {name: {"fights": int(mask.sum()),
                       "baseline": metrics(subgroup_frame[mask]),
                       "candidate": metrics(subgroup_frame[mask], "candidate_probability")}
                 for name, mask in subgroup_masks.items() if mask.sum() >= 30}
    report = {**SPEC, "source_sha256": hashes, "artifact_directory": str(output.relative_to(ROOT)),
              "chosen": chosen, "selection": selection,
              "test": {"fights": len(paired), "events": int(paired.event_id.nunique()),
                       "baseline": metrics(paired), "candidate": metrics(paired, "candidate_probability"),
                       "paired_log_loss": interval,
                       "descriptive_subgroups_not_selection_criteria": subgroups,
                       "per_year": {str(y): {"fights": len(g), "baseline": metrics(g),
                                              "candidate": metrics(g, "candidate_probability")}
                                    for y, g in paired.groupby("evaluation_year")}},
              "decision": "Keep production unchanged. " + (
                  "Candidate merits prospective recording; historical improvement still needs confirmation."
                  if chosen != "baseline" and interval["ci_95_upper"] < 0
                  else "No convincing evidence to add these features.")}
    write_json(output / "report.json", report)
    write_json(DATA / "model_research/nonlinear_feature_comparison.json", report)
    plot_diagnostics(diagnostics, paired, selection, output)
    print(json.dumps({"chosen": chosen, "test": report["test"], "output": str(output)}, indent=2), flush=True)
    return report


def plot_diagnostics(diagnostics, paired, selection, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 3, figsize=(15, 14), constrained_layout=True)
    for ax, (name, rows) in zip(axes.flat, diagnostics.items()):
        # Tied values can leave quantile bins with one or two fights. Do not
        # turn an individual upset into an apparent residual cliff.
        visible = [r for r in rows if r["fights"] >= 30]
        ax.plot([r["mean_input"] for r in visible], [r["mean_residual"] for r in visible], "o-")
        for row in visible:
            ax.annotate(str(row["fights"]), (row["mean_input"], row["mean_residual"]),
                        xytext=(0, 6), textcoords="offset points", fontsize=7)
        ax.axhline(0, color="black", linewidth=.7)
        ax.set_title(name.replace("_diff", "").replace("_", " "), fontsize=10)
        ax.set_ylabel("Actual win rate minus prediction")
    fig.suptitle("2021–22 baseline errors: descriptive only; labels = fight counts; bins under 30 omitted")
    fig.savefig(output / "residuals.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    names = list(selection)
    base_loss = selection["baseline"]["overall"]["log_loss"]
    axes[0].barh(names, [selection[n]["overall"]["log_loss"] - base_loss for n in names])
    axes[0].axvline(0, color="black", linewidth=.7)
    axes[0].set_xlabel("2023–24 change in log loss (negative is better)")
    for column, label in [("model_probability", "Baseline"), ("candidate_probability", "Frozen choice")]:
        bins = pd.cut(paired[column], np.linspace(0, 1, 11), include_lowest=True)
        grouped = paired.groupby(bins, observed=True)
        enough = grouped.size() >= 30
        axes[1].plot(grouped[column].mean()[enough], grouped.target.mean()[enough], "o-", label=label)
    axes[1].plot([0, 1], [0, 1], "k--")
    axes[1].set(xlabel="2025–26 predicted probability (bins with 30+ fights)", ylabel="Actual win rate")
    axes[1].legend()
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    evaluate()
