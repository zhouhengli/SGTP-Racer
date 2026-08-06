#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence


REQUIRED_RENDER_COLUMNS: List[str] = [
    "case_id",
    "map_name",
    "trace_path",
    "trace_meta_path",
    "planned_trajs_path",
    "boundary_offsets_path",
    "refline_xy_path",
    "artifact_dir",
]

OUTPUT_COLUMNS: List[str] = REQUIRED_RENDER_COLUMNS + ["status"]


def load_raw_records(run_dir: Path) -> List[Dict[str, str]]:
    """Load only the manifest fields consumed by the replay-video renderer."""
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "metrics" / "raw_episode_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Raw manifest does not exist: {manifest_path}")

    with manifest_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        missing = [name for name in REQUIRED_RENDER_COLUMNS if name not in fieldnames]
        if missing:
            raise ValueError(
                f"Raw manifest is missing render columns {missing}: {manifest_path}"
            )

        return [
            {
                **{name: str(row.get(name, "")) for name in REQUIRED_RENDER_COLUMNS},
                "status": str(row.get("status", "ok")),
            }
            for row in reader
        ]


def write_episode_table(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    """Write the minimal episode table accepted by the replay-video renderer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in OUTPUT_COLUMNS})


def recompute_run_outputs(run_dir: Path) -> Dict[str, object]:
    """Create one renderer-ready episode table without loading episode artifacts or computing metrics."""
    run_dir = Path(run_dir).resolve()
    output_path = run_dir / "post_metrics" / "episode_metrics.csv"
    rows = load_raw_records(run_dir)
    write_episode_table(output_path, rows)
    return {
        "run_dir": str(run_dir),
        "episode_table": str(output_path),
        "num_episodes": len(rows),
    }


def find_raw_run_dirs(batched_run_dir: Path) -> List[Path]:
    """Find first-level run directories containing a raw episode manifest."""
    batched_run_dir = Path(batched_run_dir).resolve()
    if not batched_run_dir.is_dir():
        raise FileNotFoundError(f"Batched run directory does not exist: {batched_run_dir}")

    return sorted(
        child.resolve()
        for child in batched_run_dir.iterdir()
        if child.is_dir() and (child / "metrics" / "raw_episode_manifest.csv").is_file()
    )


def recompute_batched_run_outputs(batched_run_dir: Path) -> Dict[str, object]:
    """Create renderer-ready episode tables for every first-level raw run directory."""
    batched_run_dir = Path(batched_run_dir).resolve()
    run_dirs = find_raw_run_dirs(batched_run_dir)
    if not run_dirs:
        raise FileNotFoundError(
            f"No first-level raw run directories found under: {batched_run_dir}"
        )

    run_outputs = [recompute_run_outputs(run_dir) for run_dir in run_dirs]
    return {
        "batched_run_dir": str(batched_run_dir),
        "num_run_dirs": len(run_outputs),
        "num_episodes": sum(int(item["num_episodes"]) for item in run_outputs),
        "run_outputs": run_outputs,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse input paths and write the minimal renderer episode tables."""
    parser = argparse.ArgumentParser(
        description="Create renderer-ready episode tables from raw episode manifests."
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--batched-run-dir", type=Path)
    args = parser.parse_args(argv)

    if args.batched_run_dir is not None:
        result = recompute_batched_run_outputs(args.batched_run_dir)
    elif args.run_dir is not None:
        result = recompute_run_outputs(args.run_dir)
    else:
        raise SystemExit("Provide --run-dir or --batched-run-dir")

    print(result)


if __name__ == "__main__":
    main()
