"""Optional Qt desktop explorer for local fight-simulation run artifacts."""

from __future__ import annotations

import csv
from pathlib import Path
import sys
from typing import Any, Mapping

from .gui_data import (
    DistributionSeries,
    RunBundle,
    RunBundleError,
    TraceTimeline,
    load_run_bundle,
    load_trace_timeline,
    pretty_metric,
)


class GuiDependencyError(RuntimeError):
    pass


try:  # Kept optional so simulation and server environments never require Qt.
    from PySide6 import QtCore, QtGui, QtWidgets
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    from matplotlib.figure import Figure
except ModuleNotFoundError:  # pragma: no cover - exercised on minimal installations
    QtCore = QtGui = QtWidgets = None  # type: ignore[assignment]


_RED = "#56b4e9"
_BLUE = "#e69f00"
_GREEN = "#43d17a"
_PINK = "#ff6b8a"
_GRID = "#39404d"
_TEXT = "#e8edf4"
_MUTED = "#9da8b6"


def _pct(value: object) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "—"


def _fighter_outcomes(bundle: RunBundle) -> tuple[float, float, float]:
    probabilities = bundle.aggregate.get("outcome_probabilities", {})
    if not isinstance(probabilities, Mapping):
        return 0.0, 0.0, 0.0
    red = sum(float(value) for key, value in probabilities.items() if str(key).startswith("red_"))
    blue = sum(float(value) for key, value in probabilities.items() if str(key).startswith("blue_"))
    return red, blue, max(0.0, 1.0 - red - blue)


if QtWidgets is not None:

    class FigurePanel(QtWidgets.QWidget):
        def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
            super().__init__(parent)
            self.figure = Figure(figsize=(8, 5), constrained_layout=True)
            self.canvas = FigureCanvasQTAgg(self.figure)
            self.toolbar = NavigationToolbar2QT(self.canvas, self)
            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.toolbar)
            layout.addWidget(self.canvas, 1)

        def axes(self, rows: int = 1, columns: int = 1, **kwargs: Any):
            self.figure.clear()
            self.figure.set_facecolor("#171a21")
            axes = self.figure.subplots(rows, columns, **kwargs)
            for axis in getattr(axes, "flat", [axes]):
                axis.set_facecolor("#20252e")
                axis.tick_params(colors=_MUTED)
                for spine in axis.spines.values():
                    spine.set_color(_GRID)
                axis.xaxis.label.set_color(_TEXT)
                axis.yaxis.label.set_color(_TEXT)
                axis.title.set_color(_TEXT)
                axis.grid(True, color=_GRID, alpha=0.35, linewidth=0.7)
            return axes

        def redraw(self) -> None:
            self.canvas.draw_idle()

        def save_png(self, path: str | Path) -> None:
            self.figure.savefig(path, dpi=180, facecolor=self.figure.get_facecolor())


    class OverviewTab(QtWidgets.QWidget):
        def __init__(self, bundle: RunBundle) -> None:
            super().__init__()
            self.bundle = bundle
            layout = QtWidgets.QVBoxLayout(self)
            header = QtWidgets.QLabel()
            header.setObjectName("hero")
            layout.addWidget(header)
            self.figure = FigurePanel()
            layout.addWidget(self.figure, 2)
            self.table = QtWidgets.QTableWidget()
            self.table.setColumnCount(6)
            self.table.setHorizontalHeaderLabels(
                ["Statistic", "Mean", "5%", "Median", "95%", "Observed"]
            )
            self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
            self.table.setAlternatingRowColors(True)
            layout.addWidget(self.table, 3)
            red, blue, other = _fighter_outcomes(bundle)
            status = "converged" if bundle.convergence.get("converged") else "research output"
            header.setText(
                f"{bundle.red_name}  vs  {bundle.blue_name}\n"
                f"{bundle.total_paths:,} paths · {status} · {len(bundle.trace_paths)} full traces"
            )
            axis = self.figure.axes()
            labels = [bundle.red_name, bundle.blue_name, "Draw / NC / other"]
            bars = axis.barh(labels[::-1], [red, blue, other][::-1], color=[_RED, _BLUE, _MUTED][::-1])
            axis.set_xlim(0, 1)
            axis.set_xlabel("Probability")
            axis.set_title("Winner distribution")
            axis.xaxis.set_major_formatter(lambda value, position: f"{value:.0%}")
            axis.bar_label(bars, labels=[f"{value:.1%}" for value in [red, blue, other][::-1]], color=_TEXT, padding=5)
            self.figure.redraw()
            observed_by_key = {
                series.key: series.observed for series in bundle.distributions
            }
            summaries = bundle.aggregate.get("statistic_summaries", [])
            rows = [item for item in summaries if isinstance(item, Mapping)]
            self.table.setRowCount(len(rows))
            for row, item in enumerate(rows):
                values = [
                    pretty_metric(str(item.get("statistic", ""))),
                    item.get("mean"), item.get("p05"), item.get("median"), item.get("p95"),
                    observed_by_key.get(str(item.get("statistic", ""))),
                ]
                for column, value in enumerate(values):
                    text = "—" if value is None else (f"{value:,.2f}" if isinstance(value, float) else str(value))
                    self.table.setItem(row, column, QtWidgets.QTableWidgetItem(text))


    class DistributionTab(QtWidgets.QWidget):
        def __init__(self, bundle: RunBundle) -> None:
            super().__init__()
            self.bundle = bundle
            self.series = bundle.distributions
            layout = QtWidgets.QVBoxLayout(self)
            controls = QtWidgets.QHBoxLayout()
            self.category = QtWidgets.QComboBox()
            self.category.addItems(["All", "Fight length", "Position time", "Strikes", "Grappling", "Knockdowns", "Red fighter", "Blue fighter"])
            self.metric = QtWidgets.QComboBox()
            self.view = QtWidgets.QComboBox()
            self.view.addItems(["Probability mass", "Cumulative probability", "Exceedance probability"])
            controls.addWidget(QtWidgets.QLabel("Category"))
            controls.addWidget(self.category)
            controls.addWidget(QtWidgets.QLabel("Metric"))
            controls.addWidget(self.metric, 1)
            controls.addWidget(QtWidgets.QLabel("View"))
            controls.addWidget(self.view)
            layout.addLayout(controls)
            self.figure = FigurePanel()
            layout.addWidget(self.figure, 1)
            self.summary = QtWidgets.QLabel()
            self.summary.setWordWrap(True)
            self.summary.setObjectName("summary")
            layout.addWidget(self.summary)
            self.category.currentTextChanged.connect(self._filter)
            self.metric.currentIndexChanged.connect(self._plot)
            self.view.currentIndexChanged.connect(self._plot)
            self._filter()

        def _matches(self, series: DistributionSeries, category: str) -> bool:
            key = series.key
            return {
                "All": True,
                "Fight length": key == "duration_seconds",
                "Position time": key.endswith("_time_seconds") or key.endswith("_control_seconds"),
                "Strikes": "strike" in key,
                "Grappling": any(word in key for word in ("control", "takedown", "submission", "reversal")),
                "Knockdowns": "knockdown" in key,
                "Red fighter": key.startswith("red_"),
                "Blue fighter": key.startswith("blue_"),
            }.get(category, True)

        def _filter(self) -> None:
            selected = self.metric.currentData()
            self.metric.blockSignals(True)
            self.metric.clear()
            for series in self.series:
                if self._matches(series, self.category.currentText()):
                    self.metric.addItem(series.label, series.key)
            index = self.metric.findData(selected)
            self.metric.setCurrentIndex(max(0, index))
            self.metric.blockSignals(False)
            self._plot()

        def _plot(self) -> None:
            key = self.metric.currentData()
            if not key:
                return
            series = self.bundle.distribution(str(key))
            axis = self.figure.axes()
            values = list(series.values)
            probabilities = list(series.probabilities)
            view = self.view.currentText()
            if view == "Cumulative probability":
                running = 0.0
                y = []
                for probability in probabilities:
                    running += probability
                    y.append(running)
                axis.step(values, y, where="post", color=_GREEN, linewidth=2)
                axis.set_ylabel("P(X ≤ x)")
            elif view == "Exceedance probability":
                running = 1.0
                y = []
                for probability in probabilities:
                    y.append(running)
                    running -= probability
                axis.step(values, y, where="post", color=_PINK, linewidth=2)
                axis.set_ylabel("P(X ≥ x)")
            else:
                width = max(0.8, (values[1] - values[0]) * 0.9) if len(values) > 1 else 0.8
                axis.bar(values, probabilities, width=width, color=_RED, alpha=0.85)
                axis.set_ylabel("Probability")
            if series.observed is not None:
                axis.axvline(series.observed, color="#ffffff", linestyle="--", linewidth=2, label=f"Observed: {series.observed:g}")
                axis.legend(facecolor="#20252e", labelcolor=_TEXT)
            axis.set_xlabel("Seconds" if series.unit == "seconds" else "Count")
            axis.set_title(series.label)
            if series.key == "duration_seconds":
                axis.xaxis.set_major_formatter(lambda value, position: f"{int(value)//60}:{int(value)%60:02d}")
            self.figure.redraw()
            validation = series.validation or {}
            details = [f"Mean {series.mean:,.2f}", f"paths {series.total:,}"]
            if series.observed is not None:
                details.extend([
                    f"observed {series.observed:,.2f}",
                    f"percentile {_pct(validation.get('mid_pit_percentile'))}",
                    f"two-sided tail {_pct(validation.get('two_sided_tail_probability'))}",
                    f"CRPS {float(validation.get('crps', 0)):,.3f}",
                ])
            self.summary.setText("  ·  ".join(details))


    class MarketsTab(QtWidgets.QWidget):
        def __init__(self, bundle: RunBundle) -> None:
            super().__init__()
            layout = QtWidgets.QVBoxLayout(self)
            self.figure = FigurePanel()
            layout.addWidget(self.figure, 2)
            splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
            self.totals = QtWidgets.QTableWidget()
            self.totals.setColumnCount(7)
            self.totals.setHorizontalHeaderLabels(["Line", "Seconds", "Over", "Under", "Push", "No action", "Fair O/U odds"])
            self.outcomes = QtWidgets.QTableWidget()
            self.outcomes.setColumnCount(3)
            self.outcomes.setHorizontalHeaderLabels(["Outcome", "Probability", "Fair decimal"])
            splitter.addWidget(self.outcomes)
            splitter.addWidget(self.totals)
            splitter.setStretchFactor(1, 2)
            layout.addWidget(splitter, 1)
            self._populate(bundle)

        def _populate(self, bundle: RunBundle) -> None:
            axes = self.figure.axes(1, 2)
            survival = [item for item in bundle.aggregate.get("survival", []) if isinstance(item, Mapping)]
            axes[0].step([float(item["seconds"]) for item in survival], [float(item["probability"]) for item in survival], where="post", color=_GREEN, linewidth=2)
            axes[0].set_title("Fight survival")
            axes[0].set_xlabel("Fight seconds")
            axes[0].set_ylabel("P(fight still active)")
            method_rows = [item for item in bundle.aggregate.get("method_round_counts", []) if isinstance(item, Mapping)]
            methods = sorted({str(item.get("method")) for item in method_rows})
            rounds = range(1, int(bundle.aggregate.get("scheduled_rounds", 3)) + 1)
            matrix = [[0.0 for _ in rounds] for _ in methods]
            for item in method_rows:
                if str(item.get("method")) in methods and int(item.get("round_number", 0)) in rounds:
                    matrix[methods.index(str(item["method"]))][int(item["round_number"]) - 1] = int(item.get("count", 0)) / max(1, bundle.total_paths)
            image = axes[1].imshow(matrix, aspect="auto", cmap="viridis")
            axes[1].set_title("Method × finish round")
            axes[1].set_xticks(list(range(len(list(rounds)))), [str(value) for value in rounds])
            axes[1].set_yticks(list(range(len(methods))), [pretty_metric(value) for value in methods])
            axes[1].set_xlabel("Round")
            self.figure.figure.colorbar(image, ax=axes[1], label="Probability")
            self.figure.redraw()
            probabilities = bundle.aggregate.get("outcome_probabilities", {})
            rows = sorted(probabilities.items(), key=lambda item: -float(item[1])) if isinstance(probabilities, Mapping) else []
            self.outcomes.setRowCount(len(rows))
            for row, (name, probability) in enumerate(rows):
                values = [pretty_metric(str(name)), _pct(probability), f"{1 / float(probability):.2f}" if float(probability) else "—"]
                for column, value in enumerate(values):
                    self.outcomes.setItem(row, column, QtWidgets.QTableWidgetItem(value))
            totals = [item for item in bundle.aggregate.get("total_lines", []) if isinstance(item, Mapping)]
            self.totals.setRowCount(len(totals))
            for row, item in enumerate(totals):
                actionable = max(1, int(item.get("over", 0)) + int(item.get("under", 0)))
                over = int(item.get("over", 0)) / actionable
                under = int(item.get("under", 0)) / actionable
                values = [
                    f"O/U {item.get('half_rounds')}", item.get("threshold_seconds"),
                    f"{over:.1%}", f"{under:.1%}", item.get("push", 0), item.get("no_action", 0),
                    f"{1 / over:.2f} / {1 / under:.2f}" if over and under else "—",
                ]
                for column, value in enumerate(values):
                    self.totals.setItem(row, column, QtWidgets.QTableWidgetItem(str(value)))
            for table in (self.outcomes, self.totals):
                table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)


    class ConvergenceTab(QtWidgets.QWidget):
        def __init__(self, bundle: RunBundle) -> None:
            super().__init__()
            layout = QtWidgets.QVBoxLayout(self)
            self.figure = FigurePanel()
            layout.addWidget(self.figure, 2)
            self.table = QtWidgets.QTableWidget()
            self.table.setColumnCount(7)
            self.table.setHorizontalHeaderLabels(["Metric", "Estimate", "MCSE", "Param 2.5%", "Median", "Param 97.5%", "Members"])
            layout.addWidget(self.table, 1)
            history = bundle.convergence.get("convergence", [])
            rows = [item for item in history if isinstance(item, Mapping)] if isinstance(history, list) else []
            axes = self.figure.axes(1, 2)
            axes[0].plot([item.get("total_paths", 0) for item in rows], [item.get("winner_process_mcse", 0) for item in rows], marker="o", color=_GREEN)
            axes[0].set_title("Winner Monte Carlo error")
            axes[0].set_xlabel("Total paths")
            axes[0].set_ylabel("MCSE")
            axes[1].plot([item.get("total_paths", 0) for item in rows], [item.get("parameter_quantile_max_shift", 0) for item in rows], marker="o", color=_PINK)
            axes[1].set_title("Parameter quantile stability")
            axes[1].set_xlabel("Total paths")
            axes[1].set_ylabel("Maximum shift")
            self.figure.redraw()
            uncertainty = [item for item in bundle.aggregate.get("uncertainty", []) if isinstance(item, Mapping)]
            self.table.setRowCount(len(uncertainty))
            for row, item in enumerate(uncertainty):
                conditional = item.get("conditional_probabilities", {})
                values = [item.get("metric"), item.get("estimate"), item.get("process_mcse"), item.get("parameter_p025"), item.get("parameter_median"), item.get("parameter_p975"), len(conditional) if isinstance(conditional, Mapping) else 0]
                for column, value in enumerate(values):
                    text = f"{float(value):.5f}" if isinstance(value, float) else str(value)
                    self.table.setItem(row, column, QtWidgets.QTableWidgetItem(text))
            self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)


    class TraceTab(QtWidgets.QWidget):
        def __init__(self, bundle: RunBundle) -> None:
            super().__init__()
            self.bundle = bundle
            self.timeline: TraceTimeline | None = None
            layout = QtWidgets.QVBoxLayout(self)
            controls = QtWidgets.QHBoxLayout()
            self.trace = QtWidgets.QComboBox()
            for path in bundle.trace_paths:
                self.trace.addItem(path.stem, str(path))
            self.mode = QtWidgets.QComboBox()
            self.mode.addItems(["Dynamics", "Cumulative strikes", "Grappling", "Position / phase"])
            self.filter = QtWidgets.QLineEdit()
            self.filter.setPlaceholderText("Filter events by type, actor, or action…")
            controls.addWidget(QtWidgets.QLabel("Trace"))
            controls.addWidget(self.trace, 1)
            controls.addWidget(QtWidgets.QLabel("Plot"))
            controls.addWidget(self.mode)
            controls.addWidget(self.filter, 1)
            layout.addLayout(controls)
            self.result = QtWidgets.QLabel("No full traces captured for this run.")
            self.result.setObjectName("summary")
            layout.addWidget(self.result)
            splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
            self.figure = FigurePanel()
            self.events = QtWidgets.QTableWidget()
            self.events.setColumnCount(8)
            self.events.setHorizontalHeaderLabels(["Seq", "Time", "Round", "Event", "Actor", "Action", "Phase", "RNG draws"])
            self.events.horizontalHeader().setSectionResizeMode(5, QtWidgets.QHeaderView.Stretch)
            splitter.addWidget(self.figure)
            splitter.addWidget(self.events)
            splitter.setStretchFactor(0, 2)
            layout.addWidget(splitter, 1)
            self.trace.currentIndexChanged.connect(self._load)
            self.mode.currentIndexChanged.connect(self._plot)
            self.filter.textChanged.connect(self._events)
            self._load()

        def _load(self) -> None:
            path = self.trace.currentData()
            if not path:
                return
            self.timeline = load_trace_timeline(str(path))
            result = self.timeline.result
            duration = float(result.get("fight_time_us", 0)) / 1_000_000
            self.result.setText(
                f"Simulation {self.timeline.simulation_index:,} · member {self.timeline.bootstrap_member} · "
                f"winner {result.get('winner') or 'none'} · {pretty_metric(str(result.get('method', 'unknown')))} · "
                f"round {result.get('round_number', '?')} at {int(duration)//60}:{int(duration)%60:02d} · {len(self.timeline.events)} events"
            )
            self._plot()
            self._events()

        def _plot(self) -> None:
            timeline = self.timeline
            if timeline is None:
                return
            axis = self.figure.axes()
            x = timeline.seconds
            mode = self.mode.currentText()
            if mode == "Dynamics":
                axis.plot(x, timeline.red_stamina, color=_RED, label=f"{self.bundle.red_name} stamina")
                axis.plot(x, timeline.blue_stamina, color=_BLUE, label=f"{self.bundle.blue_name} stamina")
                axis.plot(x, timeline.red_hurt, color=_PINK, linestyle="--", label=f"{self.bundle.red_name} hurt")
                axis.plot(x, timeline.blue_hurt, color="#ffd166", linestyle="--", label=f"{self.bundle.blue_name} hurt")
                axis.plot(x, timeline.red_damage, color=_RED, linestyle=":", label=f"{self.bundle.red_name} damage")
                axis.plot(x, timeline.blue_damage, color=_BLUE, linestyle=":", label=f"{self.bundle.blue_name} damage")
                axis.set_ylim(-0.03, 1.03)
                axis.set_ylabel("Bounded state")
            elif mode == "Cumulative strikes":
                for side, stats, color, name in (("red", timeline.red_stats, _RED, self.bundle.red_name), ("blue", timeline.blue_stats, _BLUE, self.bundle.blue_name)):
                    axis.step(x, stats["significant_strike_attempts"], where="post", color=color, linestyle="--", alpha=0.6, label=f"{name} attempted")
                    axis.step(x, stats["significant_strikes_landed"], where="post", color=color, linewidth=2, label=f"{name} landed")
                axis.set_ylabel("Cumulative count")
            elif mode == "Grappling":
                for stats, color, name in ((timeline.red_stats, _RED, self.bundle.red_name), (timeline.blue_stats, _BLUE, self.bundle.blue_name)):
                    axis.step(x, stats["takedowns_landed"], where="post", color=color, label=f"{name} takedowns")
                    axis.step(x, stats["submission_attempts"], where="post", color=color, linestyle="--", label=f"{name} submissions")
                    axis.step(x, [value / 1_000_000 for value in stats["control_time_us"]], where="post", color=color, linestyle=":", label=f"{name} control sec")
                axis.set_ylabel("Cumulative count / seconds")
            else:
                phase_number = {"distance": 0, "clinch": 1, "scramble": 2, "ground": 3, "red_top": 4, "blue_top": 5}
                axis.step(x, [phase_number.get(value, -1) for value in timeline.positions], where="post", color=_GREEN, linewidth=2)
                axis.set_yticks(
                    [0, 1, 2, 3, 4, 5],
                    ["Distance", "Clinch", "Scramble", "Ground", f"{self.bundle.red_name} top", f"{self.bundle.blue_name} top"],
                )
                axis.set_ylabel("Phase")
            axis.set_title(f"Single-path timeline · simulation {timeline.simulation_index}")
            axis.set_xlabel("Fight time (seconds; rests excluded)")
            if mode != "Position / phase":
                axis.legend(facecolor="#20252e", labelcolor=_TEXT, ncols=2, fontsize=8)
            self.figure.redraw()

        def _events(self) -> None:
            if self.timeline is None:
                return
            needle = self.filter.text().strip().lower()
            events = [event for event in self.timeline.events if not needle or needle in " ".join(str(event.get(key, "")) for key in ("event_type", "actor", "action", "phase_after")).lower()]
            self.events.setRowCount(len(events))
            for row, event in enumerate(events):
                seconds = float(event.get("fight_time_us", 0)) / 1_000_000
                values = [event.get("sequence"), f"{int(seconds)//60}:{seconds%60:04.1f}", event.get("round_number"), event.get("event_type"), event.get("actor") or "—", event.get("action") or "—", event.get("phase_after"), len(event.get("rng_draws", []))]
                for column, value in enumerate(values):
                    self.events.setItem(row, column, QtWidgets.QTableWidgetItem(str(value)))


    class SimulationExplorer(QtWidgets.QMainWindow):
        def __init__(self, run_directory: str | Path) -> None:
            super().__init__()
            self.bundle = load_run_bundle(run_directory)
            self.setWindowTitle(f"UFC Simulation Explorer — {self.bundle.red_name} vs {self.bundle.blue_name}")
            self.resize(1380, 900)
            self.tabs = QtWidgets.QTabWidget()
            self.tabs.addTab(OverviewTab(self.bundle), "Overview")
            self.tabs.addTab(DistributionTab(self.bundle), "Distributions")
            self.tabs.addTab(MarketsTab(self.bundle), "Betting markets")
            self.tabs.addTab(ConvergenceTab(self.bundle), "Convergence")
            self.tabs.addTab(TraceTab(self.bundle), "Single-run traces")
            self.setCentralWidget(self.tabs)
            self.statusBar().showMessage(str(self.bundle.directory))
            self._menus()

        def _menus(self) -> None:
            file_menu = self.menuBar().addMenu("File")
            open_action = file_menu.addAction("Open run…")
            open_action.setShortcut(QtGui.QKeySequence.Open)
            open_action.triggered.connect(self._open)
            export_action = file_menu.addAction("Export current chart…")
            export_action.setShortcut("Ctrl+E")
            export_action.triggered.connect(self._export_chart)
            data_action = file_menu.addAction("Export all distribution data…")
            data_action.triggered.connect(self._export_data)
            file_menu.addSeparator()
            file_menu.addAction("Quit", self.close, QtGui.QKeySequence.Quit)

        def _open(self) -> None:
            directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Open simulation run", str(self.bundle.directory.parent))
            if directory:
                try:
                    replacement = SimulationExplorer(directory)
                except RunBundleError as error:
                    QtWidgets.QMessageBox.critical(self, "Cannot open run", str(error))
                    return
                replacement.show()
                self.close()

        def _export_chart(self) -> None:
            tab = self.tabs.currentWidget()
            panel = getattr(tab, "figure", None)
            if not isinstance(panel, FigurePanel):
                QtWidgets.QMessageBox.information(self, "Nothing to export", "This tab has no chart.")
                return
            default = self.bundle.directory / f"{self.tabs.tabText(self.tabs.currentIndex()).lower().replace(' ', '-')}.png"
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export chart", str(default), "PNG image (*.png)")
            if path:
                panel.save_png(path)
                self.statusBar().showMessage(f"Exported {path}", 5000)

        def _export_data(self) -> None:
            default = self.bundle.directory / "simulation-distributions.csv"
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export distributions", str(default), "CSV file (*.csv)"
            )
            if path:
                write_distribution_csv(self.bundle, path)
                self.statusBar().showMessage(f"Exported {path}", 5000)


    _STYLE = """
    QMainWindow, QWidget { background: #171a21; color: #e8edf4; }
    QTabWidget::pane { border: 1px solid #39404d; }
    QTabBar::tab { background: #242a34; padding: 10px 18px; margin-right: 2px; }
    QTabBar::tab:selected { background: #334155; color: #ffffff; }
    QComboBox, QLineEdit { background: #242a34; border: 1px solid #4b5563; border-radius: 4px; padding: 6px; }
    QTableWidget { background: #1d222b; alternate-background-color: #242a34; gridline-color: #39404d; }
    QHeaderView::section { background: #2c3440; color: #e8edf4; padding: 6px; border: 0; }
    QToolBar, QMenuBar, QMenu, QStatusBar { background: #20252e; color: #e8edf4; }
    QLabel#hero { font-size: 22px; font-weight: 600; padding: 10px; }
    QLabel#summary { background: #20252e; border: 1px solid #39404d; border-radius: 5px; padding: 8px; }
    QSplitter::handle { background: #39404d; height: 4px; }
    """


def launch_gui(run_directory: str | Path) -> int:
    """Launch the local desktop explorer for a completed run directory."""

    if QtWidgets is None:
        raise GuiDependencyError(
            "the optional GUI dependencies are not installed; run "
            "`python -m pip install -r requirements-gui.txt`"
        )
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("UFC Simulation Explorer")
    app.setStyle("Fusion")
    app.setStyleSheet(_STYLE)
    window = SimulationExplorer(run_directory)
    window.show()
    # Keep a reference when embedded in an existing QApplication.
    setattr(app, "_fight_sim_window", window)
    return int(app.exec())


def write_distribution_csv(bundle: RunBundle, path: str | Path) -> Path:
    """Export every authoritative PMF point for offline inspection."""

    destination = Path(path)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["statistic", "value", "count", "probability", "observed"])
        for series in bundle.distributions:
            for value, count, probability in zip(series.values, series.counts, series.probabilities):
                writer.writerow([series.key, value, count, probability, series.observed])
    return destination
