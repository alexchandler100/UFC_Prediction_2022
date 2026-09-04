"""Bounded offline audit runner. Writes audit outputs, never production inputs.

Run with Python 3.11 and the existing project environment; Node is needed for
the exact website replay. No collection, training, fitting search, or API calls.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def protected_inputs() -> dict[str, str]:
    paths = [ROOT / name for name in ('script.js', 'index.html', 'style.css')]
    paths.extend((ROOT / '.github/workflows').glob('*.yml'))
    for folder, directories, names in os.walk(ROOT / 'src'):
        directories[:] = [name for name in directories
                          if name not in ('content', '__pycache__')]
        paths.extend(Path(folder) / name for name in names if name.endswith('.py'))
    # Pin all authoritative market ledgers and the key model inputs/outputs.
    paths.extend((ROOT / 'src/content/data/market').glob('*'))
    for folder in ('processed', 'external'):
        paths.extend((ROOT / f'src/content/data/{folder}').glob('*.csv'))
        paths.extend((ROOT / f'src/content/data/{folder}').glob('*.json'))
    return {str(path.relative_to(ROOT)).replace('\\', '/'): digest(path)
            for path in sorted(set(paths)) if path.is_file()}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'audit/profitability')
    parser.add_argument('--analysis-dir', type=Path,
                        help='Override the existing local historical analysis directory.')
    parser.add_argument('--skip-components', action='store_true',
                        help='Only regenerate website replay and input manifest.')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / 'audit'):
        parser.error('Output must be inside this repository\'s audit directory.')
    before = protected_inputs()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    commands = []
    environment = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
    failure = None
    try:
        if not args.skip_components:
            for component in ('duration', 'board', 'history'):
                command = [sys.executable, '-B',
                           str(ROOT / f'scripts/audit_profitability_{component}.py'),
                           '--output-dir', str(output / component)]
                if component == 'history' and args.analysis_dir:
                    command.extend(['--analysis-dir', str(args.analysis_dir)])
                result = subprocess.run(command, cwd=ROOT, env=environment,
                                        capture_output=True, text=True,
                                        encoding='utf-8', timeout=180)
                commands.append({'component': component, 'returncode': result.returncode,
                                 'stdout': result.stdout, 'stderr': result.stderr})
                if result.returncode:
                    raise RuntimeError(f'{component} failed: {result.stderr}')
                print(f'{component}: complete', flush=True)
        website = subprocess.run(['node', str(ROOT / 'scripts/audit_profitability_website.cjs')],
                                 cwd=ROOT, capture_output=True, text=True,
                                 encoding='utf-8', check=True, timeout=30)
        replay = json.loads(website.stdout)
        (output / 'website_replay.json').write_text(
            json.dumps(replay, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        write_csv(output / 'website_replay_summary.csv', replay['summaries'])
        write_csv(output / 'website_replay_bets.csv', replay['rows'])
        print(f"website: {len(replay['summaries'])} existing-view/payout combinations", flush=True)
    except Exception as error:
        failure = error
    finally:
        after = protected_inputs()
        changed = sorted(key for key in before.keys() | after.keys()
                         if before.get(key) != after.get(key))
        manifest = {'schema_version': 1, 'offline': True,
                    'status': 'failed' if failure or changed else 'complete',
                    'production_inputs_unchanged': not changed,
                    'changed_production_inputs': changed,
                    'protected_source_hashes': before,
                    'runtime_seconds': round(time.monotonic() - started, 3),
                    'component_results': commands,
                    'failure': str(failure) if failure else None}
        (output / 'manifest.json').write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    if changed:
        raise RuntimeError(f'Production input changed during audit: {changed}')
    if failure:
        raise failure
    print(f"Audit completed in {manifest['runtime_seconds']}s; production inputs unchanged.")


if __name__ == '__main__':
    main()
