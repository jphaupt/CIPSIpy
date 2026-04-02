#!/usr/bin/env python3
"""Run the simple end-to-end benchmark and optionally write a metadata-rich JSON file.

Examples:
    python benchmarks/run_whole_program_benchmark.py
    python benchmarks/run_whole_program_benchmark.py --include-slow \
      --output-json benchmarks/results/head.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        capture_output=True,
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_output_json(repo_root: Path, hostname: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    commit = _resolve_head_sha(repo_root)[:12]
    safe_host = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in hostname)
    return repo_root / "benchmarks" / "results" / f"whole-program-{timestamp}-{commit}-{safe_host}.json"


def _resolve_head_sha(repo_root: Path) -> str:
    return _run(["git", "rev-parse", "HEAD"], cwd=repo_root).stdout.strip()


def _resolve_branch(repo_root: Path) -> str:
    try:
        return _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root).stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def _parse_benchmark_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    benches = payload.get("benchmarks", [])
    if not benches:
        raise RuntimeError(f"No benchmark entries found in {path}")

    return {
        "benchmark_count": len(benches),
        "mean_seconds": sum(float(b["stats"]["mean"]) for b in benches) / len(benches),
        "median_seconds": sum(float(b["stats"]["median"]) for b in benches) / len(benches),
        "min_seconds": min(float(b["stats"]["min"]) for b in benches),
        "max_seconds": max(float(b["stats"]["max"]) for b in benches),
        "rounds": sum(int(b["stats"]["rounds"]) for b in benches),
        "tests": [
            {
                "name": b.get("name"),
                "fullname": b.get("fullname"),
                "param": b.get("param"),
                "mean_seconds": float(b["stats"]["mean"]),
                "median_seconds": float(b["stats"]["median"]),
                "min_seconds": float(b["stats"]["min"]),
                "max_seconds": float(b["stats"]["max"]),
                "stddev_seconds": float(b["stats"]["stddev"]),
                "rounds": int(b["stats"]["rounds"]),
            }
            for b in benches
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run end-to-end CIPSI benchmark with metadata output")
    parser.add_argument(
        "--include-slow",
        action="store_true",
        help="Include slow benchmark case (LiH/6-31gstar)",
    )
    parser.add_argument(
        "--hostname",
        default=None,
        help=(
            "Override hostname stored in results. "
            "Defaults to $CIPSI_BENCH_HOSTNAME, then machine hostname."
        ),
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=6,
        help="Minimum benchmark rounds per test (default: 6)",
    )
    parser.add_argument(
        "--output-json",
        default="auto",
        help=(
            "Path to write benchmark results and metadata as JSON. "
            "Default: auto-generate under benchmarks/results/."
        ),
    )
    parser.add_argument(
        "--no-output-json",
        action="store_true",
        help="Disable JSON output file generation.",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    hostname = args.hostname or os.environ.get("CIPSI_BENCH_HOSTNAME") or socket.gethostname()

    with tempfile.TemporaryDirectory(prefix="cipsipy-bench-json-") as tmp_dir:
        pytest_json = Path(tmp_dir) / "pytest-benchmark.json"

        cmd = [
            "pytest",
            "benchmarks/test_cipsi_performance.py",
            "--benchmark-only",
            "--benchmark-min-rounds",
            str(max(1, args.rounds)),
            "--benchmark-disable-gc",
            "--benchmark-json",
            str(pytest_json),
        ]
        if args.include_slow:
            cmd.extend(["-m", "benchmark or slow"])
        else:
            cmd.extend(["-m", "benchmark and not slow"])

        print("Running whole-program benchmark...", flush=True)
        print(f"Command: {' '.join(cmd)}", flush=True)

        started = datetime.now(timezone.utc)
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(repo_root), check=False, text=True)
        elapsed = time.perf_counter() - t0
        finished = datetime.now(timezone.utc)

        if proc.returncode != 0:
            return proc.returncode

        stats = _parse_benchmark_json(pytest_json)

    payload = {
        "run_started_at": started.isoformat(),
        "run_finished_at": finished.isoformat(),
        "wall_time_seconds": elapsed,
        "git": {
            "branch": _resolve_branch(repo_root),
            "commit": _resolve_head_sha(repo_root),
        },
        "host": {
            "hostname": hostname,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        },
        "command": " ".join(cmd),
        "benchmark": stats,
    }

    print("\nRun metadata")
    print("=" * 60)
    print(f"commit:   {payload['git']['commit']}")
    print(f"branch:   {payload['git']['branch']}")
    print(f"host:     {payload['host']['hostname']}")
    print(f"wall[s]:  {payload['wall_time_seconds']:.4f}")
    print(f"mean[s]:  {payload['benchmark']['mean_seconds']:.4f}")
    print(f"tests:    {payload['benchmark']['benchmark_count']}")

    if not args.no_output_json:
        if args.output_json == "auto":
            output_path = _default_output_json(repo_root, hostname)
        else:
            output_path = Path(args.output_json)
            if not output_path.is_absolute():
                output_path = repo_root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        print(f"wrote:    {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
