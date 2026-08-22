"""Atomic, append-only storage for external source snapshots and observations."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable

from .adapters import AdapterResult
from .schema import ExternalBoutObservation, ExternalDataError


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ExternalDataError(f"invalid JSONL at {path}:{line_number}: {error}") from error
        if not isinstance(value, dict):
            raise ExternalDataError(f"JSONL row at {path}:{line_number} is not an object")
        rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    lines = [_canonical_json(row) for row in rows]
    _atomic_write(path, "".join(f"{line}\n" for line in lines))


class ExternalMmaStore:
    """Own the local raw-snapshot ledger and normalized observation ledger."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.observations_path = self.root / "bouts.jsonl"
        self.snapshots_path = self.root / "snapshots.jsonl"
        self.rejections_path = self.root / "rejections.jsonl"
        self.raw_root = self.root / "raw"
        self.registry_path = self.root / "source_registry.json"

    def source_registry(self) -> dict[str, dict[str, object]]:
        try:
            value = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ExternalDataError(f"could not read source registry: {error}") from error
        if not isinstance(value, dict) or not isinstance(value.get("sources"), list):
            raise ExternalDataError("source_registry.json must contain a sources array")
        sources = {}
        for row in value["sources"]:
            if not isinstance(row, dict) or not str(row.get("key", "")).strip():
                raise ExternalDataError("every source registry row needs a key")
            key = str(row["key"]).strip()
            if key in sources:
                raise ExternalDataError(f"duplicate source registry key {key!r}")
            sources[key] = row
        return sources

    def observations(self) -> list[ExternalBoutObservation]:
        rows = [ExternalBoutObservation.from_mapping(row) for row in _read_jsonl(self.observations_path)]
        self._validate_observation_set(rows)
        return rows

    @staticmethod
    def _validate_observation_set(rows: list[ExternalBoutObservation]) -> None:
        seen: dict[str, str] = {}
        source_bouts: dict[tuple[str, str], str] = {}
        for observation in rows:
            payload = _canonical_json(observation.to_dict())
            if observation.observation_id in seen:
                previous = seen[observation.observation_id]
                qualifier = "conflicting" if previous != payload else "duplicate"
                raise ExternalDataError(
                    f"{qualifier} observation_id {observation.observation_id}"
                )
            seen[observation.observation_id] = payload
            bout_key = (observation.source, observation.source_bout_id)
            if bout_key in source_bouts:
                raise ExternalDataError(f"source bout key is not unique: {bout_key}")
            source_bouts[bout_key] = observation.observation_id

    def validate(self) -> dict[str, int]:
        registry = self.source_registry()
        observations = self.observations()
        snapshots = _read_jsonl(self.snapshots_path)
        known_snapshots = {str(row.get("snapshot_sha256", "")) for row in snapshots}
        for observation in observations:
            if observation.source not in registry:
                raise ExternalDataError(f"unknown source {observation.source!r}")
            if observation.snapshot_sha256 not in known_snapshots:
                raise ExternalDataError(
                    f"observation references missing snapshot {observation.snapshot_sha256}"
                )
        for row in snapshots:
            digest = str(row.get("snapshot_sha256", ""))
            if len(digest) != 64:
                raise ExternalDataError("snapshot ledger contains an invalid SHA-256")
            raw_path = str(row.get("raw_path", ""))
            if raw_path:
                path = self.root / raw_path
                if not path.exists() or sha256(path.read_bytes()).hexdigest() != digest:
                    raise ExternalDataError(f"raw snapshot hash mismatch: {path}")
        return {
            "sources": len(registry),
            "snapshots": len(snapshots),
            "observations": len(observations),
            "rejections": len(_read_jsonl(self.rejections_path)),
        }

    def import_bytes(
        self,
        adapter,
        content: bytes,
        *,
        retrieved_at_utc: str | None = None,
        store_raw: bool = False,
        original_filename: str = "source.csv",
    ) -> dict[str, object]:
        registry = self.source_registry()
        if adapter.source_key not in registry:
            raise ExternalDataError(
                f"source {adapter.source_key!r} is not in source_registry.json"
            )
        source_config = registry[adapter.source_key]
        if source_config.get("collection_status") not in {"manual_import", "licensed_export"}:
            raise ExternalDataError(
                f"source {adapter.source_key!r} is not approved for collection"
            )
        digest = sha256(content).hexdigest()
        result: AdapterResult = adapter.convert(content, digest)
        self._validate_observation_set(result.observations)
        retrieved = retrieved_at_utc or datetime.now(timezone.utc).replace(
            microsecond=0
        ).isoformat()
        parsed_retrieved = datetime.fromisoformat(retrieved.replace("Z", "+00:00"))
        if parsed_retrieved.tzinfo is None:
            raise ExternalDataError("retrieved_at_utc must include an offset")
        retrieved = parsed_retrieved.astimezone(timezone.utc).replace(
            microsecond=0
        ).isoformat()

        snapshots = _read_jsonl(self.snapshots_path)
        if any(
            str(row.get("source")) == adapter.source_key
            and str(row.get("snapshot_sha256")) == digest
            for row in snapshots
        ):
            return {
                "status": "already_imported",
                "snapshot_sha256": digest,
                "source_rows": result.total_rows,
                "accepted": len(result.observations),
                "rejected": len(result.rejected),
            }

        existing = {row.observation_id: row for row in self.observations()}
        added = 0
        for observation in result.observations:
            previous = existing.get(observation.observation_id)
            if previous is not None and previous.to_dict() != observation.to_dict():
                # Snapshot hash is part of the row; repeat observations from a
                # newer export are allowed only when all substantive fields agree.
                old = previous.to_dict()
                new = observation.to_dict()
                old.pop("snapshot_sha256", None)
                new.pop("snapshot_sha256", None)
                if old != new:
                    raise ExternalDataError(
                        f"source changed immutable bout {observation.source_bout_id}"
                    )
                continue
            if previous is None:
                existing[observation.observation_id] = observation
                added += 1

        raw_relative = ""
        if store_raw:
            safe_name = Path(original_filename).name or "source.csv"
            raw_path = self.raw_root / adapter.source_key / digest / safe_name
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            if raw_path.exists() and sha256(raw_path.read_bytes()).hexdigest() != digest:
                raise ExternalDataError(f"refusing to overwrite mismatched raw snapshot {raw_path}")
            if not raw_path.exists():
                raw_path.write_bytes(content)
            raw_relative = raw_path.relative_to(self.root).as_posix()

        snapshot_row = {
            "schema_version": 1,
            "source": adapter.source_key,
            "snapshot_sha256": digest,
            "retrieved_at_utc": retrieved,
            "source_rows": result.total_rows,
            "accepted_rows": len(result.observations),
            "rejected_rows": len(result.rejected),
            "license": source_config.get("license", ""),
            "source_page": source_config.get("source_page", ""),
            "raw_path": raw_relative,
        }
        rejected = _read_jsonl(self.rejections_path)
        rejected.extend(
            {
                "schema_version": 1,
                "source": adapter.source_key,
                "snapshot_sha256": digest,
                **row,
            }
            for row in result.rejected
        )
        sorted_observations = sorted(
            (row.to_dict() for row in existing.values()),
            key=lambda row: (row["event_date"], row["source"], row["source_bout_id"]),
        )
        _write_jsonl(self.observations_path, sorted_observations)
        _write_jsonl(self.snapshots_path, [*snapshots, snapshot_row])
        _write_jsonl(self.rejections_path, rejected)
        self.validate()
        return {
            "status": "imported",
            "snapshot_sha256": digest,
            "source_rows": result.total_rows,
            "accepted": len(result.observations),
            "added": added,
            "rejected": len(result.rejected),
            "raw_stored": bool(raw_relative),
        }
