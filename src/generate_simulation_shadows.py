"""Generate one candidate-only simulation shadow after production publication.

This entry point is intentionally separate from ``update_and_rebuild_model``.
The production updater commits its validated data first; a missing gate,
non-converged simulation, timeout, or invariant failure here cannot prevent or
roll back that production publication.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path

from git import Repo

from data_handler import DataHandler
from fight_sim.shadow import maybe_generate_weekly_shadows


SRC_ROOT = Path(__file__).resolve().parent
SIMULATION_ROOT = SRC_ROOT / "content/data/simulation"
CARD_PATH = SRC_ROOT / "content/data/external/card_info.json"


def _source_revision() -> str:
    revision = Repo(SRC_ROOT.parents[0]).head.commit.hexsha.strip().lower()
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise RuntimeError("could not resolve the simulator source revision")
    return revision


def main() -> int:
    data = DataHandler()
    upcoming = data.get("vegas_odds", filetype="json")
    if upcoming is None or upcoming.empty:
        raise RuntimeError("published upcoming-card rows are unavailable")
    card = json.loads(CARD_PATH.read_text(encoding="utf-8"))
    issued = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    path = maybe_generate_weekly_shadows(
        simulation_directory=SIMULATION_ROOT,
        raw_fights=data.get("ufc_fights_reported_doubled"),
        fighter_profiles=data.get("fighter_stats"),
        round_stats=data.get("ufc_fight_round_stats_doubled"),
        upcoming=upcoming,
        card=card,
        source_commit_sha=_source_revision(),
        forecast_issued_at_utc=issued,
    )
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    message = (
        "Simulation shadows are disabled; no reviewed frozen gate was found."
        if path is None
        else f"Appended candidate-only simulation shadow: {path}"
    )
    print(message)
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as output:
            output.write(f"\n## Simulation shadow\n\n- {message}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
