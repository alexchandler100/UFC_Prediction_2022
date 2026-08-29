"""Print human-readable progress from a posterior study summary."""

from __future__ import annotations

import json
from pathlib import Path
import sys


def main(summary_path: str) -> int:
    report = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    selection = report["selection"]
    runtime = report["runtime"]
    print(
        "Progress: {}/{} fights and {}/{} fight/seed pairs checkpointed.".format(
            selection["completed_fights"],
            selection["eligible_fights"],
            runtime["completed_fight_seed_pairs"],
            runtime["planned_fight_seed_pairs"],
        )
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: simulation_study_progress.py SUMMARY.json")
    raise SystemExit(main(sys.argv[1]))
