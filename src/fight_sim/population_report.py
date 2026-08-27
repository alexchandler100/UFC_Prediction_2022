"""Self-contained population posterior-predictive HTML report."""

from __future__ import annotations

from html import escape
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping


def _number(value: object, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return escape(str(value))


def _percent(value: object) -> str:
    if value is None:
        return "—"
    return f"{float(value):.1%}"


def _pit_bars(values: object) -> str:
    counts = list(values) if isinstance(values, list) else []
    maximum = max([int(value) for value in counts] or [1])
    return '<div class="spark">' + "".join(
        f'<span title="bin {index + 1}: {int(value)}" style="height:{max(2, int(value) / maximum * 34):.1f}px"></span>'
        for index, value in enumerate(counts)
    ) + "</div>"


def render_population_report(report: Mapping[str, object]) -> str:
    aggregate = dict(report.get("aggregate") or {})
    winner = dict(aggregate.get("winner") or {})
    selection = dict(report.get("selection") or {})
    runtime = dict(report.get("runtime") or {})
    comparisons = dict(report.get("comparisons") or {})
    checks = dict(aggregate.get("posterior_predictive_checks") or {})
    check_rows = []
    ordered = sorted(
        (
            (str(name), dict(value))
            for name, value in checks.items()
            if isinstance(value, Mapping)
        ),
        key=lambda pair: (
            float(pair[1].get("pit_cvm_nominal_iid_pvalue") or 1.0),
            pair[0],
        ),
    )
    for name, item in ordered:
        check_rows.append(
            "<tr>"
            f"<td>{escape(name.replace('_', ' ').title())}</td>"
            f"<td>{int(item.get('n', 0))}</td>"
            f"<td>{_number(item.get('predictive_minus_observed_mean'))}</td>"
            f"<td>{_number(item.get('mean_crps'))}</td>"
            f"<td>{_percent(item.get('interval_50_coverage'))}</td>"
            f"<td>{_percent(item.get('interval_80_coverage'))}</td>"
            f"<td>{_percent(item.get('interval_90_coverage'))}</td>"
            f"<td>{_percent(item.get('interval_95_coverage'))}</td>"
            f"<td>{_percent(item.get('randomized_tail_below_0_05_rate'))}</td>"
            f"<td>{_number(item.get('pit_cvm_nominal_iid_pvalue'), 4)}</td>"
            f"<td>{_pit_bars(item.get('pit_histogram_10'))}</td>"
            "</tr>"
        )
    event_rows = []
    for source in list(report.get("event_summaries") or []):
        item = dict(source)
        event_rows.append(
            "<tr>"
            f"<td>{escape(str(item.get('date')))}</td>"
            f"<td><code>{escape(str(item.get('event_id')))}</code></td>"
            f"<td>{int(item.get('fights', 0))}</td>"
            f"<td>{_number(item.get('joint_log_loss'))}</td>"
            f"<td>{_number(item.get('winner_log_loss'))}</td>"
            f"<td>{_number(item.get('duration_crps_seconds'), 1)}</td>"
            "</tr>"
        )
    embedded = escape(json.dumps(report, sort_keys=True, ensure_ascii=False))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>UFC simulator population posterior checks</title>
<style>
:root{{color-scheme:dark;font-family:Inter,Segoe UI,system-ui,sans-serif;background:#0b0f17;color:#e7edf7}}
body{{margin:0;padding:28px}}main{{max-width:1600px;margin:auto}}h1{{margin-bottom:4px}}h2{{margin-top:28px}}
.muted{{color:#9ba8ba}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:20px 0}}
.card,.panel{{background:#141b27;border:1px solid #2a3548;border-radius:10px;padding:16px}}.value{{font-size:25px;font-weight:700;margin-top:5px}}
.table{{overflow:auto;border:1px solid #2a3548;border-radius:10px}}table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{padding:8px 10px;border-bottom:1px solid #273143;text-align:right;white-space:nowrap}}th{{position:sticky;top:0;background:#1b2433;color:#b9c6d8}}th:first-child,td:first-child,td:nth-child(2){{text-align:left}}
.spark{{display:flex;align-items:flex-end;gap:2px;height:36px;min-width:110px}}.spark span{{display:block;width:9px;background:#43d17a;border-radius:2px 2px 0 0}}code{{color:#8ecbff}}details{{margin-top:24px}}
</style></head><body><main>
<h1>UFC simulator population posterior checks</h1>
<p class="muted">Causal event-cutoff research · candidate only · no production influence</p>
<div class="cards">
 <div class="card"><div class="muted">Eligible fights</div><div class="value">{int(selection.get('eligible_fights', 0))}</div></div>
 <div class="card"><div class="muted">Low-exposure excluded</div><div class="value">{int(selection.get('excluded_low_exposure', 0))}</div></div>
 <div class="card"><div class="muted">Winner log loss</div><div class="value">{_number(winner.get('log_loss'))}</div></div>
 <div class="card"><div class="muted">Winner Brier</div><div class="value">{_number(winner.get('brier'))}</div></div>
 <div class="card"><div class="muted">Side × method log loss</div><div class="value">{_number(aggregate.get('primary_joint_side_method_log_loss'))}</div></div>
 <div class="card"><div class="muted">Duration CRPS</div><div class="value">{_number(aggregate.get('duration_crps_seconds'),1)}s</div></div>
 <div class="card"><div class="muted">Runtime</div><div class="value">{_number(float(runtime.get('elapsed_seconds', 0))/60,1)} min</div></div>
</div>
<section class="panel"><h2>Outcome baseline comparison</h2>
<p>Simulation joint log loss: <strong>{_number(comparisons.get('simulation_joint_log_loss'))}</strong> · population baseline: <strong>{_number(comparisons.get('population_joint_log_loss'))}</strong> · division baseline: <strong>{_number(comparisons.get('division_joint_log_loss'))}</strong>.</p>
</section>
<h2>Posterior-predictive calibration</h2>
<p class="muted">Well-calibrated columns approach nominal coverage (50/80/90/95%), a 5% tail rate, and a flat ten-bin PIT histogram. Nominal PIT p-values are diagnostic only and do not correct for card clustering or multiple testing.</p>
<div class="table"><table><thead><tr><th>Statistic</th><th>N</th><th>Mean bias</th><th>CRPS</th><th>50%</th><th>80%</th><th>90%</th><th>95%</th><th>Tail &lt;5%</th><th>CvM p</th><th>PIT histogram</th></tr></thead><tbody>{''.join(check_rows)}</tbody></table></div>
<h2>Event summaries</h2><div class="table"><table><thead><tr><th>Date</th><th>Event</th><th>Fights</th><th>Joint loss</th><th>Winner loss</th><th>Duration CRPS</th></tr></thead><tbody>{''.join(event_rows)}</tbody></table></div>
<details><summary>Embedded authoritative summary JSON</summary><pre>{embedded}</pre></details>
</main></body></html>"""


def write_population_report(path: str | Path, report: Mapping[str, object]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = render_population_report(report)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination
