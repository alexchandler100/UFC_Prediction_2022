"""Self-contained local HTML reports for fight-simulation research.

This module deliberately has no web-server or JavaScript-package dependency.
Reports are written below the ignored ``artifacts/`` tree by the CLI and embed
the exact JSON used to render them so a chart can always be audited.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from html import escape
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping


REPORT_SCHEMA_VERSION = 1


def _mapping(value: object | None) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict"):
        converted = value.to_dict()  # type: ignore[attr-defined]
        if isinstance(converted, dict):
            return converted
    if is_dataclass(value):
        converted = asdict(value)
        if isinstance(converted, dict):
            return converted
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"expected a mapping-compatible value, got {type(value).__name__}")


def _number(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _percent(value: object) -> str:
    return f"{_number(value):.1%}"


def _pretty(value: object) -> str:
    if value is None:
        return "Unavailable"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:,.4f}"
    return str(value)


def _outcome_probabilities(aggregate: Mapping[str, Any]) -> dict[str, float]:
    probabilities = aggregate.get("outcome_probabilities")
    if isinstance(probabilities, Mapping):
        return {
            str(key): _number(value)
            for key, value in probabilities.items()
            if _number(value) >= 0.0
        }
    counts = aggregate.get("outcome_counts")
    total = max(1.0, _number(aggregate.get("total_paths"), 1.0))
    if isinstance(counts, Mapping):
        return {str(key): _number(value) / total for key, value in counts.items()}
    return {}


def _bar_chart(values: Mapping[str, float]) -> str:
    rows = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    if not rows:
        return '<p class="muted">No outcome distribution was supplied.</p>'
    width = 780
    label_width = 210
    plot_width = width - label_width - 80
    row_height = 31
    height = 24 + row_height * len(rows)
    elements = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Outcome probability bar chart">'
    ]
    for index, (label, probability) in enumerate(rows):
        y = 19 + index * row_height
        bar_width = max(0.0, min(1.0, probability)) * plot_width
        elements.extend(
            [
                f'<text x="0" y="{y + 15}" class="chart-label">{escape(label.replace("_", " ").title())}</text>',
                f'<rect x="{label_width}" y="{y}" width="{plot_width}" height="20" rx="4" class="track"/>',
                f'<rect x="{label_width}" y="{y}" width="{bar_width:.2f}" height="20" rx="4" class="bar"/>',
                f'<text x="{label_width + plot_width + 10}" y="{y + 15}" class="chart-value">{probability:.1%}</text>',
            ]
        )
    elements.append("</svg>")
    return "".join(elements)


def _survival_chart(points: object) -> str:
    if not isinstance(points, list) or not points:
        return '<p class="muted">No duration survival curve was supplied.</p>'
    parsed: list[tuple[float, float]] = []
    for point in points:
        if not isinstance(point, Mapping):
            continue
        seconds = _number(point.get("seconds"))
        probability = _number(point.get("probability"))
        if seconds >= 0 and 0 <= probability <= 1:
            parsed.append((seconds, probability))
    parsed.sort()
    if not parsed:
        return '<p class="muted">No valid duration points were supplied.</p>'
    width, height = 780, 250
    left, top, right, bottom = 58, 18, 18, 42
    max_seconds = max(seconds for seconds, _ in parsed) or 1.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    coordinates = [
        (
            left + seconds / max_seconds * plot_width,
            top + (1.0 - probability) * plot_height,
        )
        for seconds, probability in parsed
    ]
    polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in coordinates)
    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Fight duration survival curve">'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" class="axis"/>'
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" class="axis"/>'
        f'<polyline points="{polyline}" class="survival"/>'
        f'<text x="4" y="{top + 4}" class="chart-value">100%</text>'
        f'<text x="20" y="{top + plot_height}" class="chart-value">0%</text>'
        f'<text x="{left}" y="{height - 10}" class="chart-value">0:00</text>'
        f'<text x="{left + plot_width - 50}" y="{height - 10}" class="chart-value">{int(max_seconds // 60)}:{int(max_seconds % 60):02d}</text>'
        "</svg>"
    )


def _table(headers: Iterable[str], rows: Iterable[Iterable[object]]) -> str:
    head = "".join(f"<th>{escape(str(header))}</th>" for header in headers)
    body_rows = []
    for row in rows:
        body_rows.append(
            "<tr>"
            + "".join(f"<td>{escape(_pretty(value))}</td>" for value in row)
            + "</tr>"
        )
    if not body_rows:
        column_count = len(tuple(headers))
        body_rows.append(
            f'<tr><td colspan="{column_count}" class="muted">No records supplied.</td></tr>'
        )
    return f"<div class=\"table-wrap\"><table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"


def _uncertainty_rows(aggregate: Mapping[str, Any]) -> list[list[object]]:
    rows: list[list[object]] = []
    values = aggregate.get("uncertainty", [])
    if not isinstance(values, list):
        return rows
    for item in values:
        if not isinstance(item, Mapping):
            continue
        rows.append(
            [
                item.get("metric"),
                _percent(item.get("estimate")),
                _percent(item.get("process_mcse")),
                _percent(item.get("parameter_p025")),
                _percent(item.get("parameter_median")),
                _percent(item.get("parameter_p975")),
            ]
        )
    return rows


def _flatten_metric_rows(
    value: object,
    *,
    category: str,
    prefix: str = "",
) -> list[list[object]]:
    """Flatten scalar evaluation metrics without hiding nested comparisons."""

    rows: list[list[object]] = []
    if isinstance(value, Mapping):
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            name = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(
                _flatten_metric_rows(item, category=category, prefix=name)
            )
    elif isinstance(value, (str, int, float, bool)) or value is None:
        rows.append([category, prefix, value])
    return rows


def _trace_sections(traces: Iterable[object]) -> str:
    sections: list[str] = []
    for trace_index, raw_trace in enumerate(traces):
        trace = _mapping(raw_trace)
        events = trace.get("events", [])
        if not isinstance(events, list):
            events = []
        rows = []
        for event in events:
            if not isinstance(event, Mapping):
                continue
            clock = event.get("clock", {})
            if not isinstance(clock, Mapping):
                clock = {}
            rows.append(
                [
                    event.get("sequence"),
                    event.get("event_type", event.get("kind")),
                    clock.get("round", event.get("round_number")),
                    clock.get(
                        "fight_seconds",
                        event.get(
                            "fight_seconds",
                            _number(event.get("fight_time_us")) / 1_000_000,
                        ),
                    ),
                    event.get("actor"),
                    event.get("action"),
                    event.get("state_before_hash", event.get("state_hash_before")),
                    event.get("state_after_hash", event.get("state_hash_after")),
                ]
            )
        label = trace.get("simulation_index", trace_index)
        reasons = trace.get("selection_reasons", [])
        sections.append(
            '<details class="trace">'
            f'<summary>Simulation {escape(str(label))} · {len(events)} events · {escape(", ".join(map(str, reasons)))}</summary>'
            + _table(
                ("Seq", "Event", "Round", "Fight seconds", "Actor", "Action", "Before hash", "After hash"),
                rows,
            )
            + "</details>"
        )
    return "".join(sections) or '<p class="muted">No diagnostic traces were supplied.</p>'


def render_analysis_report(
    aggregate: object,
    *,
    run_spec: object | None = None,
    traces: Iterable[object] = (),
    evaluation: object | None = None,
    title: str | None = None,
) -> str:
    """Return a fully self-contained, auditable HTML analysis report."""

    aggregate_map = _mapping(aggregate)
    spec_map = _mapping(run_spec)
    evaluation_map = _mapping(evaluation)
    report_title = title or f"Fight simulation · {aggregate_map.get('matchup_id', 'unknown matchup')}"
    outcomes = _outcome_probabilities(aggregate_map)

    total_rows = []
    for item in aggregate_map.get("total_lines", []):
        if not isinstance(item, Mapping):
            continue
        denominator = sum(
            int(_number(item.get(key))) for key in ("over", "under", "push", "no_action")
        )
        denominator = max(1, denominator)
        total_rows.append(
            [
                item.get("half_rounds"),
                item.get("threshold_seconds"),
                _percent(_number(item.get("over")) / denominator),
                _percent(_number(item.get("under")) / denominator),
                _percent(_number(item.get("push")) / denominator),
                _percent(_number(item.get("no_action")) / denominator),
            ]
        )

    statistic_uncertainty = {
        str(item.get("statistic")): item
        for item in aggregate_map.get("statistic_uncertainty", [])
        if isinstance(item, Mapping) and item.get("statistic")
    }
    statistic_rows = []
    for item in aggregate_map.get("statistic_summaries", []):
        if isinstance(item, Mapping):
            uncertainty = statistic_uncertainty.get(str(item.get("statistic")), {})
            statistic_rows.append(
                [
                    item.get("statistic"),
                    item.get("mean"),
                    item.get("p05"),
                    item.get("median"),
                    item.get("p95"),
                    uncertainty.get("process_mcse_mean"),
                    uncertainty.get("parameter_model_p025"),
                    uncertainty.get("parameter_model_median"),
                    uncertainty.get("parameter_model_p975"),
                ]
            )

    baseline_rows: list[list[object]] = []
    if isinstance(evaluation_map.get("aggregate"), Mapping):
        baseline_rows.extend(
            _flatten_metric_rows(
                evaluation_map["aggregate"], category="simulation"
            )
        )
    metrics = evaluation_map.get("metrics")
    if isinstance(metrics, Mapping):
        baseline_rows.extend(
            _flatten_metric_rows(metrics, category="evaluation")
        )
    comparisons = evaluation_map.get("comparisons")
    if isinstance(comparisons, Mapping):
        baseline_rows.extend(
            _flatten_metric_rows(comparisons, category="baseline comparison")
        )

    stack_rows: list[list[object]] = []
    stack_fold_rows: list[list[object]] = []
    stack = (
        comparisons.get("production_simulation_stack")
        if isinstance(comparisons, Mapping)
        else None
    )
    if isinstance(stack, Mapping):
        stack_metrics = stack.get("stack")
        production_metrics = stack.get("production_same_fights")
        paired = stack.get("paired_event_card_interval_vs_production")
        stack_metrics = stack_metrics if isinstance(stack_metrics, Mapping) else {}
        production_metrics = (
            production_metrics if isinstance(production_metrics, Mapping) else {}
        )
        paired = paired if isinstance(paired, Mapping) else {}
        stack_rows = [
            ["Status", stack.get("status")],
            ["Covered fights", stack.get("n_covered")],
            ["Stack log loss", stack_metrics.get("log_loss")],
            ["Production log loss (same fights)", production_metrics.get("log_loss")],
            ["Stack Brier", stack_metrics.get("brier")],
            ["Production Brier (same fights)", production_metrics.get("brier")],
            ["Paired log-loss difference", paired.get("challenger_minus_baseline_log_loss")],
            ["Paired interval 2.5%", paired.get("interval_p025")],
            ["Paired interval 97.5%", paired.get("interval_p975")],
            ["Candidate freeze recommended", stack.get("candidate_freeze_recommended")],
        ]
        for fold in stack.get("folds", []):
            if not isinstance(fold, Mapping):
                continue
            stack_fold_rows.append(
                [
                    fold.get("test_year"),
                    fold.get("status"),
                    fold.get("training_fights"),
                    fold.get("test_fights"),
                    fold.get("beta_model"),
                    fold.get("beta_sim"),
                ]
            )
    stack_section = ""
    if stack_rows:
        stack_section = (
            '<section class="panel"><h2>Production + simulation winner stack</h2>'
            '<p class="muted">Weights are zero-intercept, nonnegative, and fitted '
            "only on earlier out-of-fold fights. Negative paired differences favor "
            "the stack.</p>"
            + _table(("Measure", "Value"), stack_rows)
            + _table(
                (
                    "Test year",
                    "Status",
                    "Prior OOF fights",
                    "Test fights",
                    "Model weight",
                    "Simulation weight",
                ),
                stack_fold_rows,
            )
            + "</section>"
        )

    warnings = []
    for source in (spec_map, aggregate_map, evaluation_map):
        values = source.get("warnings", source.get("coverage_warnings", []))
        if isinstance(values, list):
            warnings.extend(str(value) for value in values)

    convergence_rows = []
    convergence = evaluation_map.get(
        "history",
        evaluation_map.get("convergence", spec_map.get("convergence", [])),
    )
    if isinstance(convergence, Mapping) and isinstance(
        convergence.get("history"), list
    ):
        convergence = convergence["history"]
    if isinstance(convergence, Mapping):
        convergence = [convergence]
    if isinstance(convergence, list):
        for index, item in enumerate(convergence, start=1):
            if not isinstance(item, Mapping):
                continue
            convergence_rows.append(
                [
                    index,
                    item.get("paths_per_member"),
                    item.get("total_paths"),
                    _percent(item.get("winner_process_mcse")),
                    _percent(item.get("split_estimate_difference")),
                    _percent(item.get("parameter_quantile_max_shift")),
                    item.get("mcse_within_target"),
                    item.get("headline_batches_stable"),
                    item.get("parameter_quantiles_stable"),
                ]
            )

    embedded = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "aggregate": aggregate_map,
        "run_spec": spec_map,
        "evaluation": evaluation_map,
    }
    embedded_json = json.dumps(
        embedded, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).replace("</", "<\\/")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(report_title)}</title>
<style>
:root {{ color-scheme: dark; --bg:#0b1017; --panel:#131b26; --line:#2b3a4d; --text:#eef3f8; --muted:#9eafc2; --accent:#53d6a4; --accent2:#5fa8ff; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.5 system-ui,sans-serif; }}
main {{ width:min(1120px,calc(100% - 28px)); margin:28px auto 72px; }} h1 {{ font-size:clamp(24px,4vw,40px); margin-bottom:4px; }} h2 {{ margin-top:0; }}
.lede,.muted {{ color:var(--muted); }} .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:16px; }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:18px; margin:16px 0; overflow:hidden; }}
.facts {{ display:flex; flex-wrap:wrap; gap:10px; }} .fact {{ background:#0e1620; border:1px solid var(--line); border-radius:8px; padding:8px 12px; }}
.chart {{ width:100%; max-height:420px; overflow:visible; }} .track {{ fill:#233041; }} .bar {{ fill:var(--accent); }} .axis {{ stroke:#73859b; stroke-width:1; }} .survival {{ fill:none; stroke:var(--accent2); stroke-width:3; }}
.chart-label,.chart-value {{ fill:var(--text); font-size:13px; }} .table-wrap {{ overflow:auto; }} table {{ border-collapse:collapse; width:100%; }} th,td {{ border-bottom:1px solid var(--line); padding:7px 9px; text-align:left; white-space:nowrap; }} th {{ color:var(--muted); font-weight:600; }}
.warning {{ border-left:3px solid #f4c95d; padding:8px 12px; background:#231f13; }} details.trace {{ border-top:1px solid var(--line); padding:10px 0; }} summary {{ cursor:pointer; font-weight:650; }}
code {{ color:#a9d4ff; }}
</style>
</head>
<body><main>
<h1>{escape(report_title)}</h1>
<p class="lede">Candidate-only local research report. Process Monte Carlo error and bootstrap parameter/model uncertainty are separate; illustrative traces are not probability estimates.</p>
<section class="panel"><div class="facts">
<div class="fact"><strong>Paths</strong><br>{int(_number(aggregate_map.get('total_paths'))):,}</div>
<div class="fact"><strong>Bootstrap members</strong><br>{int(_number(aggregate_map.get('bootstrap_members'))):,}</div>
<div class="fact"><strong>Scheduled rounds</strong><br>{escape(str(aggregate_map.get('scheduled_rounds', 'Unknown')))}</div>
<div class="fact"><strong>Matchup ID</strong><br><code>{escape(str(aggregate_map.get('matchup_id', 'Unknown')))}</code></div>
</div></section>
{''.join(f'<p class="warning">{escape(warning)}</p>' for warning in warnings)}
<div class="grid">
<section class="panel"><h2>Joint terminal outcomes</h2>{_bar_chart(outcomes)}</section>
<section class="panel"><h2>Duration survival</h2>{_survival_chart(aggregate_map.get('survival'))}</section>
</div>
<section class="panel"><h2>Nested uncertainty</h2>{_table(("Metric","Estimate","Process MCSE","Parameter 2.5%","Parameter median","Parameter 97.5%"), _uncertainty_rows(aggregate_map))}</section>
<section class="panel"><h2>Convergence</h2>{_table(("Batch","Paths/member","Total paths","Winner MCSE","Split difference","Parameter quantile shift","MCSE gate","Batch gate","Parameter gate"), convergence_rows)}</section>
<section class="panel"><h2>Full-fight totals</h2>{_table(("Line","Threshold seconds","Over","Under","Push","No action"), total_rows)}</section>
<section class="panel"><h2>Projected fight statistics</h2>{_table(("Statistic","Mean","P05","Median","P95","Process MCSE of mean","Parameter/model 2.5%","Parameter/model median","Parameter/model 97.5%"), statistic_rows)}</section>
{stack_section}
<section class="panel"><h2>Evaluation and baselines</h2>{_table(("Category","Metric","Value"), baseline_rows)}</section>
<section class="panel"><h2>Selected deterministic traces</h2>{_trace_sections(traces)}</section>
<section class="panel"><h2>Audit payload</h2><p class="muted">The exact aggregate, run specification, and evaluation payload are embedded below.</p><details><summary>Show canonical report data</summary><pre id="audit-data">{escape(json.dumps(embedded, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))}</pre></details></section>
<script type="application/json" id="fight-sim-report-data">{embedded_json}</script>
</main></body></html>"""


def write_analysis_report(
    path: str | Path,
    aggregate: object,
    *,
    run_spec: object | None = None,
    traces: Iterable[object] = (),
    evaluation: object | None = None,
    title: str | None = None,
) -> Path:
    """Atomically write a local report and return its resolved path."""

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_analysis_report(
        aggregate,
        run_spec=run_spec,
        traces=traces,
        evaluation=evaluation,
        title=title,
    )
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(rendered)
        handle.flush()
    temporary.replace(target)
    return target


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value
