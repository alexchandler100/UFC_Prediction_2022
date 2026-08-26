"""Observed-versus-simulated diagnostics for one completed physical fight.

The validator consumes exact marginal Monte Carlo counts from a completed run
and the repository's mirrored UFCStats bout totals.  It deliberately reports
definition alignment and marginal diagnostics separately: a marginal tail
probability is not a joint likelihood for the complete observed fight vector.
"""

from __future__ import annotations

from html import escape
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd


VALIDATION_SCHEMA_VERSION = 1


STATISTIC_TARGETS: dict[str, tuple[str, str, str, str]] = {
    "significant_strikes": (
        "sig_strikes_landed",
        "Significant strikes landed",
        "count",
        "exact",
    ),
    "significant_strike_attempts": (
        "sig_strikes_attempts",
        "Significant strikes attempted",
        "count",
        "exact",
    ),
    "head_strikes_landed": (
        "head_strikes_landed",
        "Head significant strikes landed",
        "count",
        "exact",
    ),
    "body_strikes_landed": (
        "body_strikes_landed",
        "Body significant strikes landed",
        "count",
        "exact",
    ),
    "leg_strikes_landed": (
        "leg_strikes_landed",
        "Leg significant strikes landed",
        "count",
        "exact",
    ),
    "distance_strikes_landed": (
        "distance_strikes_landed",
        "Distance significant strikes landed",
        "count",
        "exact",
    ),
    "distance_strike_attempts": (
        "distance_strikes_attempts",
        "Distance significant strikes attempted",
        "count",
        "exact",
    ),
    "clinch_strikes_landed": (
        "clinch_strikes_landed",
        "Clinch significant strikes landed",
        "count",
        "exact",
    ),
    "clinch_strike_attempts": (
        "clinch_strikes_attempts",
        "Clinch significant strikes attempted",
        "count",
        "exact",
    ),
    "ground_strikes_landed": (
        "ground_strikes_landed",
        "Ground significant strikes landed",
        "count",
        "exact",
    ),
    "ground_strike_attempts": (
        "ground_strikes_attempts",
        "Ground significant strikes attempted",
        "count",
        "exact",
    ),
    "knockdowns": ("knockdowns", "Knockdowns", "count", "exact"),
    "takedowns": ("takedowns_landed", "Takedowns landed", "count", "exact"),
    "takedown_attempts": (
        "takedowns_attempts",
        "Takedowns attempted",
        "count",
        "exact",
    ),
    "submission_attempts": (
        "sub_attempts",
        "Submission attempts",
        "count",
        "exact",
    ),
    # The simulator accrues ground top-position time. UFCStats publishes a
    # broader control field that can include non-ground control. Preserve the
    # useful comparison while refusing to call the definitions identical.
    "control_seconds": (
        "control",
        "Control seconds",
        "seconds",
        "partial",
    ),
}


def _atomic_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, newline="\n"
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)
    return path


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _fight_id(frame: pd.DataFrame) -> pd.Series:
    if "fight_id" in frame:
        return frame["fight_id"].astype(str)
    if "fight_url" not in frame:
        raise ValueError("observed fights require fight_id or fight_url")
    return frame["fight_url"].astype(str).str.rstrip("/").str.rsplit("/", n=1).str[-1]


def _fighter_id(frame: pd.DataFrame) -> pd.Series:
    if "fighter_id" in frame:
        return frame["fighter_id"].astype(str)
    if "fighter_url" not in frame:
        raise ValueError("observed fights require fighter_id or fighter_url")
    return frame["fighter_url"].astype(str).str.rstrip("/").str.rsplit("/", n=1).str[-1]


def _distribution(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, int]:
    counts = row.get("counts")
    if not isinstance(counts, list) or not counts:
        raise ValueError("statistic distribution has no exact counts")
    support = np.asarray([float(item["value"]) for item in counts], dtype=float)
    frequency = np.asarray([int(item["count"]) for item in counts], dtype=float)
    if np.any(~np.isfinite(support)) or np.any(frequency < 0) or frequency.sum() <= 0:
        raise ValueError("statistic distribution contains invalid values")
    order = np.argsort(support, kind="stable")
    total = int(frequency.sum())
    return support[order], frequency[order] / total, total


def _weighted_quantile(support: np.ndarray, mass: np.ndarray, probability: float) -> float:
    index = int(np.searchsorted(np.cumsum(mass), probability, side="left"))
    return float(support[min(index, len(support) - 1)])


def _crps(observed: float, support: np.ndarray, mass: np.ndarray) -> float:
    first = float(np.sum(mass * np.abs(support - observed)))
    cumulative_probability = 0.0
    cumulative_value = 0.0
    half_pair_distance = 0.0
    for value, probability in zip(support, mass, strict=True):
        half_pair_distance += float(probability) * (
            float(value) * cumulative_probability - cumulative_value
        )
        cumulative_probability += float(probability)
        cumulative_value += float(probability) * float(value)
    return first - half_pair_distance


def score_observation(
    *, statistic: str,
    label: str,
    observed: float,
    support: np.ndarray,
    mass: np.ndarray,
    unit: str,
    definition_alignment: str,
    predictive_sample_size: int,
) -> dict[str, Any]:
    """Score one observed value against an exact discrete predictive marginal."""

    if predictive_sample_size <= 0:
        raise ValueError("predictive_sample_size must be positive")
    mean = float(np.sum(support * mass))
    variance = float(np.sum(np.square(support - mean) * mass))
    standard_deviation = math.sqrt(max(0.0, variance))
    less = float(mass[support < observed].sum())
    equal = float(mass[support == observed].sum())
    greater = float(mass[support > observed].sum())
    lower_inclusive = less + equal
    upper_inclusive = greater + equal
    two_sided_tail = min(1.0, 2.0 * min(lower_inclusive, upper_inclusive))
    zero_tail_upper_95 = (
        min(1.0, 2.0 * (1.0 - math.pow(0.05, 1.0 / predictive_sample_size)))
        if two_sided_tail == 0.0
        else None
    )
    intervals: dict[str, dict[str, float | bool]] = {}
    for coverage in (0.50, 0.80, 0.90, 0.95):
        tail = (1.0 - coverage) / 2.0
        lower = _weighted_quantile(support, mass, tail)
        upper = _weighted_quantile(support, mass, 1.0 - tail)
        intervals[f"p{int(coverage * 100)}"] = {
            "lower": lower,
            "upper": upper,
            "contains_observed": lower <= observed <= upper,
        }
    crps = _crps(observed, support, mass)
    return {
        "statistic": statistic,
        "label": label,
        "unit": unit,
        "definition_alignment": definition_alignment,
        "observed": observed,
        "mean": mean,
        "standard_deviation": standard_deviation,
        "median": _weighted_quantile(support, mass, 0.5),
        "p05": _weighted_quantile(support, mass, 0.05),
        "p95": _weighted_quantile(support, mass, 0.95),
        "pit_lower": less,
        "pit_upper": less + equal,
        "mid_pit_percentile": less + 0.5 * equal,
        "lower_tail_probability": lower_inclusive,
        "upper_tail_probability": upper_inclusive,
        "two_sided_tail_probability": two_sided_tail,
        "two_sided_tail_monte_carlo_resolution": min(
            1.0, 2.0 / predictive_sample_size
        ),
        "two_sided_tail_upper_95_when_zero": zero_tail_upper_95,
        "predictive_sample_size": predictive_sample_size,
        "exact_predictive_probability": equal,
        "standardized_residual": (
            (observed - mean) / standard_deviation
            if standard_deviation > 0
            else (0.0 if observed == mean else None)
        ),
        "crps": crps,
        "standardized_crps": crps / standard_deviation if standard_deviation > 0 else None,
        "central_intervals": intervals,
    }


def _actual_outcome_key(red: Mapping[str, Any], blue: Mapping[str, Any]) -> str:
    result = str(red.get("result") or "").strip().upper()
    winner = "red" if result == "W" else "blue" if result == "L" else ""
    method = str(red.get("method") or "").strip().upper()
    if result in {"D", "DRAW"}:
        return "draw"
    if result in {"NC", "N/C"}:
        return "no_contest"
    if "DEC" in method:
        bucket = "decision"
    elif "SUB" in method:
        bucket = "submission"
    elif "KO" in method:
        bucket = "ko_tko"
    else:
        bucket = "other"
    return f"{winner}_{bucket}" if winner else bucket


def validate_completed_fight(
    run_path: str | Path,
    *,
    observed_path: str | Path,
    fight_id: str,
) -> dict[str, Any]:
    run = Path(run_path)
    if run.is_dir():
        aggregate_document = _load_json(run / "aggregate.json")
        specs_document = _load_json(run / "specs.json")
    else:
        raise ValueError("completed-fight validation requires a run directory")
    aggregate = aggregate_document.get("aggregate", aggregate_document)
    if not isinstance(aggregate, dict):
        raise ValueError("run aggregate is invalid")
    specs = specs_document.get("specs")
    if not isinstance(specs, list) or not specs:
        raise ValueError("run specs are missing")
    first_spec = specs[0]
    red_spec = first_spec["red"]
    blue_spec = first_spec["blue"]
    red_id = str(red_spec["fighter_id"])
    blue_id = str(blue_spec["fighter_id"])

    observed_frame = pd.read_csv(observed_path, low_memory=False)
    observed_frame = observed_frame.assign(
        _fight_id=_fight_id(observed_frame),
        _fighter_id=_fighter_id(observed_frame),
    )
    bout = observed_frame.loc[observed_frame["_fight_id"].eq(str(fight_id))].copy()
    if len(bout) != 2 or set(bout["_fighter_id"]) != {red_id, blue_id}:
        raise ValueError(
            "observed fight must contain exactly the red and blue fighters from the run"
        )
    red = bout.loc[bout["_fighter_id"].eq(red_id)].iloc[0].to_dict()
    blue = bout.loc[bout["_fighter_id"].eq(blue_id)].iloc[0].to_dict()

    distributions = {
        str(item["statistic"]): item
        for item in aggregate.get("statistic_distributions", [])
        if isinstance(item, Mapping) and item.get("statistic")
    }
    scores: list[dict[str, Any]] = []
    missing_distributions: list[str] = []
    for side, actual in (("red", red), ("blue", blue)):
        for suffix, (source, label, unit, alignment) in STATISTIC_TARGETS.items():
            statistic = f"{side}_{suffix}"
            if statistic not in distributions:
                missing_distributions.append(statistic)
                continue
            observed_value = pd.to_numeric(pd.Series([actual.get(source)]), errors="coerce").iloc[0]
            if pd.isna(observed_value):
                continue
            support, mass, sample_size = _distribution(distributions[statistic])
            score = score_observation(
                statistic=statistic,
                label=f"{actual.get('fighter', side.title())}: {label}",
                observed=float(observed_value),
                support=support,
                mass=mass,
                unit=unit,
                definition_alignment=alignment,
                predictive_sample_size=sample_size,
            )
            scores.append(score)

    # Duration is published in five-second upper-bound bins rather than in the
    # exact marginal statistic ledger. Score it with that declared resolution.
    duration_bins = aggregate.get("duration_bins", [])
    if duration_bins:
        support = np.asarray(
            [float(item["upper_seconds"]) for item in duration_bins], dtype=float
        )
        counts = np.asarray([int(item["count"]) for item in duration_bins], dtype=float)
        observed_duration = float(red["total_fight_time"])
        scores.append(
            score_observation(
                statistic="duration_seconds",
                label="Fight duration (five-second bins)",
                observed=observed_duration,
                support=support,
                mass=counts / counts.sum(),
                unit="seconds",
                definition_alignment="binned",
                predictive_sample_size=int(counts.sum()),
            )
        )

    outcome_probabilities = aggregate.get("outcome_probabilities", {})
    actual_outcome = _actual_outcome_key(red, blue)
    outcome_counts = aggregate.get("outcome_counts", {})
    actual_outcome_count = int(outcome_counts.get(actual_outcome, 0))
    total_paths = int(aggregate.get("total_paths", 0))
    actual_outcome_probability = float(
        outcome_probabilities.get(actual_outcome, 0.0)
    )
    actual_outcome_mcse = (
        math.sqrt(
            actual_outcome_probability * (1.0 - actual_outcome_probability) / total_paths
        )
        if total_paths > 0
        else None
    )
    outliers = [
        item["statistic"]
        for item in scores
        if item["definition_alignment"] in {"exact", "binned"}
        and item["two_sided_tail_probability"] < 0.05
    ]
    exact_scores = [
        item
        for item in scores
        if item["definition_alignment"] in {"exact", "binned"}
    ]
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "candidate_only": True,
        "production_enabled": False,
        "execution_enabled": False,
        "run_directory": str(run.resolve()),
        "matchup_id": aggregate.get("matchup_id"),
        "fight_id": str(fight_id),
        "event_date": str(red.get("date") or ""),
        "red": {"fighter_id": red_id, "fighter_name": red_spec.get("fighter_name")},
        "blue": {"fighter_id": blue_id, "fighter_name": blue_spec.get("fighter_name")},
        "total_paths": total_paths,
        "bootstrap_members": int(aggregate.get("bootstrap_members", 0)),
        "actual_outcome": actual_outcome,
        "actual_outcome_count": actual_outcome_count,
        "actual_outcome_probability": actual_outcome_probability,
        "actual_outcome_process_mcse": actual_outcome_mcse,
        "statistics": scores,
        "summary": {
            "scored_marginals": len(scores),
            "exact_or_binned_marginals": len(exact_scores),
            "two_sided_tail_outliers_p_lt_0_05": outliers,
            "p95_coverage_rate": (
                float(
                    np.mean(
                        [item["central_intervals"]["p95"]["contains_observed"] for item in exact_scores]
                    )
                )
                if exact_scores
                else None
            ),
            "median_absolute_standardized_residual": (
                float(
                    np.median(
                        [
                            abs(item["standardized_residual"])
                            for item in exact_scores
                            if item["standardized_residual"] is not None
                        ]
                    )
                )
                if exact_scores
                else None
            ),
        },
        "coverage_warnings": [
            "marginal_tail_probabilities_are_not_a_joint_fight_likelihood",
            "ufcstats_control_is_broader_than_simulated_ground_top_control",
            "ufcstats_distance_strikes_include_punches_and_kicks",
            *(
                ["single_bootstrap_member_excludes_parameter_model_uncertainty"]
                if int(aggregate.get("bootstrap_members", 0)) == 1
                else []
            ),
            *(
                ["run_does_not_publish_all_aligned_statistic_distributions"]
                if missing_distributions
                else []
            ),
        ],
        "missing_statistic_distributions": missing_distributions,
        "joint_distribution_available": False,
    }


def render_validation_html(report: Mapping[str, Any]) -> str:
    rows: list[str] = []
    for item in report.get("statistics", []):
        interval = item["central_intervals"]["p90"]
        residual = item.get("standardized_residual")
        tail_upper = item.get("two_sided_tail_upper_95_when_zero")
        tail_text = (
            f"0 observed; &lt;{float(tail_upper):.3%} (95% MC upper)"
            if tail_upper is not None
            else f"{item['two_sided_tail_probability']:.2%}"
        )
        rows.append(
            "<tr>"
            f"<td>{escape(str(item['label']))}</td>"
            f"<td>{item['observed']:,.2f}</td>"
            f"<td>{item['mean']:,.2f}</td>"
            f"<td>{item['median']:,.2f}</td>"
            f"<td>{interval['lower']:,.2f}–{interval['upper']:,.2f}</td>"
            f"<td>{item['mid_pit_percentile']:.1%}</td>"
            f"<td>{tail_text}</td>"
            f"<td>{'—' if residual is None else f'{residual:+.2f}'}</td>"
            f"<td>{item['crps']:,.3f}</td>"
            f"<td>{escape(str(item['definition_alignment']))}</td>"
            "</tr>"
        )
    warnings = "".join(
        f"<li>{escape(str(value).replace('_', ' '))}</li>"
        for value in report.get("coverage_warnings", [])
        if value
    )
    red = report["red"]["fighter_name"]
    blue = report["blue"]["fighter_name"]
    embedded = escape(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Posterior-predictive fight validation</title>
<style>
:root{{color-scheme:dark;background:#0b1020;color:#e8edf7;font-family:Inter,system-ui,sans-serif}}
body{{margin:0;padding:28px}}main{{max-width:1500px;margin:auto}}.panel{{background:#121a2c;border:1px solid #29344d;border-radius:12px;padding:18px;margin:16px 0;overflow:auto}}
h1,h2{{margin:.2em 0 .7em}}.metric{{display:inline-block;margin:6px 18px 6px 0}}.metric b{{display:block;font-size:1.3rem;color:#71d7a5}}
table{{border-collapse:collapse;width:100%;font-size:.88rem}}th,td{{padding:8px 9px;border-bottom:1px solid #29344d;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}th{{color:#9fb0cc;position:sticky;top:0;background:#121a2c}}
.warn{{color:#ffd28a}}code{{white-space:pre-wrap;word-break:break-word}}</style></head>
<body><main><h1>{escape(str(red))} vs. {escape(str(blue))}</h1>
<p>Observed fight versus exact Monte Carlo marginal distributions. Candidate-only research.</p>
<section class="panel"><span class="metric">Paths<b>{int(report['total_paths']):,}</b></span><span class="metric">Actual outcome<b>{escape(str(report['actual_outcome']).replace('_',' ').title())}</b></span><span class="metric">Simulated probability of actual outcome<b>{float(report['actual_outcome_probability']):.2%}</b></span><span class="metric">95% marginal coverage<b>{float(report['summary']['p95_coverage_rate'] or 0):.1%}</b></span></section>
<section class="panel"><h2>Observed values inside simulated distributions</h2><table><thead><tr><th>Statistic</th><th>Observed</th><th>Mean</th><th>Median</th><th>Central 90%</th><th>Percentile</th><th>Two-sided tail</th><th>Z residual</th><th>CRPS</th><th>Alignment</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
<section class="panel warn"><h2>Interpretation limits</h2><ul>{warnings}</ul></section>
<details class="panel"><summary>Embedded authoritative JSON</summary><code>{embedded}</code></details>
</main></body></html>"""


def write_validation_report(
    report: Mapping[str, Any], *, json_path: str | Path, html_path: str | Path
) -> tuple[Path, Path]:
    json_output = _atomic_text(
        Path(json_path), json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    html_output = _atomic_text(Path(html_path), render_validation_html(report))
    return json_output, html_output
