"""Deterministic local performance measurements for the bulk simulator."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
from statistics import median
import time
from typing import Iterable

from .monte_carlo import run_nested
from .parameters import canonical_sha256
from .research import atomic_write_json, load_specs


def compiled_backend_readiness() -> dict[str, object]:
    compilers = {
        name: shutil.which(name)
        for name in ("cl", "g++", "clang++")
    }
    available_compilers = {
        name: path for name, path in compilers.items() if path is not None
    }
    packages = {
        name: importlib.util.find_spec(name) is not None
        for name in ("numba", "Cython", "pybind11")
    }
    return {
        "compiler_commands": available_compilers,
        "optional_packages": packages,
        "prototype_supported": bool(available_compilers)
        or packages["numba"],
        "recommended_backend": (
            "numba" if packages["numba"] else (
                "cpp_batch_extension" if available_compilers else None
            )
        ),
    }


def execute_benchmark(
    spec_path: str | Path,
    *,
    paths_per_member: int = 128,
    worker_counts: Iterable[int] = (1,),
    chunk_size: int = 64,
    repeats: int = 1,
    output: str | Path | None = None,
) -> dict[str, object]:
    """Measure complete streaming runs while checking worker invariance."""

    workers = tuple(dict.fromkeys(int(value) for value in worker_counts))
    if paths_per_member <= 0 or chunk_size <= 0 or repeats <= 0:
        raise ValueError("benchmark paths, chunk size, and repeats must be positive")
    if not workers or any(value <= 0 for value in workers):
        raise ValueError("benchmark worker counts must be positive")
    specs = load_specs(spec_path)
    rows: list[dict[str, object]] = []
    expected_forecast_sha256: str | None = None
    baseline_seconds: float | None = None
    for worker_count in workers:
        durations: list[float] = []
        forecast_sha256: str | None = None
        for _ in range(repeats):
            started = time.perf_counter()
            result = run_nested(
                specs,
                paths_per_member,
                workers=worker_count,
                chunk_size=chunk_size,
                max_traces=0,
                retain_paths=False,
            )
            durations.append(time.perf_counter() - started)
            current_hash = canonical_sha256(result.forecast.to_dict())
            if forecast_sha256 is not None and current_hash != forecast_sha256:
                raise RuntimeError("repeated benchmark forecasts are not deterministic")
            forecast_sha256 = current_hash
        if expected_forecast_sha256 is None:
            expected_forecast_sha256 = forecast_sha256
        elif forecast_sha256 != expected_forecast_sha256:
            raise RuntimeError("benchmark forecast changed with worker count")
        elapsed = float(median(durations))
        if baseline_seconds is None:
            baseline_seconds = elapsed
        total_paths = len(specs) * paths_per_member
        rows.append(
            {
                "workers": worker_count,
                "chunk_size": chunk_size,
                "repeats": repeats,
                "paths_per_member": paths_per_member,
                "bootstrap_members": len(specs),
                "total_paths": total_paths,
                "median_seconds": elapsed,
                "paths_per_second": total_paths / elapsed,
                "speedup_vs_first": baseline_seconds / elapsed,
                "forecast_sha256": forecast_sha256,
            }
        )
    body: dict[str, object] = {
        "schema_version": 1,
        "benchmark_type": "complete_streaming_nested_simulation",
        "spec_path": str(Path(spec_path)),
        "results": rows,
        "worker_invariant": True,
        "compiled_backend_readiness": compiled_backend_readiness(),
    }
    body["benchmark_sha256"] = canonical_sha256(body)
    if output is not None:
        atomic_write_json(output, body)
    return body
