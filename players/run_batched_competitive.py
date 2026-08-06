#!/usr/bin/env python3
"""Run batched competitive evaluations and save raw artifacts only.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from players.utils.common import (
    RAW_MANIFEST_COLUMNS,
    clean_id,
    format_duration,
    json_ready,
    load_yaml,
    now_tag,
    rel_to,
    require_cfg,
    require_key,
    save_yaml,
    write_csv,
    write_json,
)


TRACE_CHANNELS: List[str] = [
    "t",
    "poses_x",
    "poses_y",
    "poses_theta",
    "action_steer",
    "action_speed",
]

EVAL_SEED: int = 6300


@dataclass(frozen=True)
class EvalCase:
    """Resolved evaluation case."""

    case_id: str
    split: str
    map_name: str
    raceline: str
    opp_raceline: str
    ego_start_slot_idx: int
    interval_idx: int
    rand_seed: int
    sim_duration: float
    v_global_limit: float
    opp_speed_scale: float
    num_agents: int


def map_names(split_cfg: Dict[str, Any], split_key: str) -> List[str]:
    """Read map names from one split section."""
    names: List[str] = []
    for entry in split_cfg[split_key]:
        names.append(entry if isinstance(entry, str) else str(entry["map_name"]))
    return names


def waypoint_csv_path(root: Path, cfg: Dict[str, Any], map_name: str, raceline: str) -> Path:
    """Build the waypoint CSV path for one map/raceline."""
    return root / require_cfg(cfg, "map_root") / map_name / f"{map_name}_{raceline}.csv"


def load_num_waypoints(root: Path, cfg: Dict[str, Any], map_name: str, raceline: str) -> int:
    """Load waypoint count for start-point construction."""
    arr = np.loadtxt(
        waypoint_csv_path(root, cfg, map_name, raceline),
        delimiter=require_cfg(cfg, "waypoint_delimiter"),
        skiprows=int(require_cfg(cfg, "waypoint_skiprows")),
    )
    return int(arr.shape[0])


def uniform_indices(num_wp: int, num_points: int) -> List[int]:
    """Return uniformly spaced waypoint indices."""
    return np.linspace(0, int(num_wp), int(num_points), endpoint=False).astype(int).tolist()


def trace_to_numpy(trace: Sequence[Dict[str, Any]]) -> np.ndarray:
    """Convert rollout trace dictionaries into a dense numpy tensor."""
    if not trace:
        raise ValueError("rollout_trace is empty; cannot save trace tensor")

    num_steps = len(trace)
    num_agents = len(trace[0]["poses_x"])
    tensor = np.full((num_steps, num_agents, len(TRACE_CHANNELS)), np.nan, dtype=np.float32)
    channel_index = {name: idx for idx, name in enumerate(TRACE_CHANNELS)}

    for step_idx, step in enumerate(trace):
        tensor[step_idx, :, channel_index["t"]] = float(step["t"])
        for key in ["poses_x", "poses_y", "poses_theta"]:
            tensor[step_idx, :, channel_index[key]] = np.asarray(step[key], dtype=np.float64)
        action = np.asarray(step["action"], dtype=np.float64)
        tensor[step_idx, :, channel_index["action_steer"]] = action[:, 0]
        tensor[step_idx, :, channel_index["action_speed"]] = action[:, 1]
    return tensor


def save_trace_npy(trace_path: Path, meta_path: Path, trace: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Save one dense trace tensor and metadata."""
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    tensor = trace_to_numpy(trace)
    np.save(trace_path, tensor, allow_pickle=False)
    meta = {
        "channels": TRACE_CHANNELS,
    }
    write_json(meta_path, meta)
    return meta


def save_planned_trajs_npz(
    path: Path,
    planned_trajs: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Save only the ego per-planning-cycle best trajectories needed for video replay."""
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays: Dict[str, np.ndarray] = {}
    records_meta: List[Dict[str, Any]] = []

    for record_idx, record in enumerate(planned_trajs):
        trajs = list(record.get("trajs", []))
        ego_traj = trajs[0] if trajs else None
        key = None
        if ego_traj is not None:
            arr = np.asarray(ego_traj, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] > 2:
                arr = arr[:, :2]
            key = f"r{record_idx:06d}_a00"
            arrays[key] = arr

        records_meta.append(
            {
                "trace_start_index": int(record.get("trace_start_index", 0)),
                "agents": [{"agent_idx": 0, "key": key}],
            }
        )

    metadata = {
        "records": records_meta,
    }

    arrays["metadata_json"] = np.asarray(json.dumps(json_ready(metadata)), dtype=np.str_)
    np.savez_compressed(path, **arrays)
    return metadata


def save_render_artifact_json(path: Path, case: EvalCase, args: SimpleNamespace) -> Dict[str, Any]:
    """Save the tiny metadata file expected by trace_dashborad.render_case_video.

    Batch evaluation still runs with ``save_video=False``.  This file is only
    a replay descriptor: it lets the offline dashboard locate map assets,
    vehicle dimensions, frame stride, fps, and HUD limits.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    map_yaml_path = Path("MapZoo") / str(case.map_name) / f"{case.map_name}.yaml"

    artifact = {
        "render_config": {
            "out_fps": int(getattr(args, "out_fps", 10)),
            "video_capture_every": int(getattr(args, "video_capture_every", 1)),
            "vehicle_length": float(getattr(args, "length", 0.58)),
            "vehicle_width": float(getattr(args, "width", 0.31)),
        },
        "hud_limits": {
            "speed_max": float(getattr(args, "v_max")) * float(getattr(args, "v_global_limit", 1.0)),
            "steer_limit": float(getattr(args, "delta_max")),
        },
        "assets": {
            "map_yaml_path": str(map_yaml_path),
        },
    }
    write_json(path, artifact)
    return artifact


def save_array_npy(path: Path, value: Any, fmt: str, run_dir: Path) -> Dict[str, Any]:
    """Save one numpy array artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(value, dtype=np.float32)
    np.save(path, arr, allow_pickle=False)
    return {"format": fmt, "path": rel_to(path, run_dir), "shape": list(arr.shape), "dtype": str(arr.dtype)}


class CompetitiveRawEvalRunner:
    """Run competitive evaluation and save raw artifacts only."""

    def __init__(self, eval_config: Dict[str, Any], project_root: Path, sim_duration: float, num_start_points_per_map: int) -> None:
        """Initialize output paths from config and command-line overrides."""
        self.cfg = copy.deepcopy(eval_config)
        self.root = project_root
        self.sim_duration = float(sim_duration)
        self.num_start_points_per_map = int(num_start_points_per_map)

        method_name = clean_id(self.cfg["algorithm"]["method"])
        self.run_dir = self.root / require_cfg(self.cfg, "output_root") / f"{method_name}_raw_eval_{now_tag()}"
        self.config_dir = self.run_dir / "configs"
        self.case_dir = self.run_dir / "cases"
        self.trace_dir = self.run_dir / "traces"
        self.artifact_dir = self.run_dir / "artifacts"
        self.metric_dir = self.run_dir / "metrics"

    def evaluate(self) -> Dict[str, Any]:
        """Run all cases, save raw artifacts, and write raw manifest/summary."""
        start = time.perf_counter()
        self.prepare_run()
        cases = self.build_cases()
        if not cases:
            raise RuntimeError("no evaluation cases were built")

        records = self.run_cases(cases)
        elapsed = time.perf_counter() - start
        self.write_raw_manifest(records)
        summary = self.build_summary(records, elapsed)

        print(f"[EVAL] raw data done cases={len(records)} out={self.run_dir}", flush=True)
        print(f"[EVAL] raw manifest out={self.metric_dir / 'raw_episode_manifest.csv'}", flush=True)
        print("[EVAL] run metrics separately: python -m players.metrics.competitive_metrics --run-dir <run_dir> --project-root .", flush=True)
        return summary

    def prepare_run(self) -> None:
        """Create output directories and save config snapshots."""
        for path in [self.config_dir, self.case_dir, self.trace_dir, self.artifact_dir, self.metric_dir]:
            path.mkdir(parents=True, exist_ok=True)
        save_yaml(self.config_dir / "eval_config.yaml", self.cfg)
        save_yaml(self.config_dir / "algorithm_config.yaml", require_cfg(self.cfg, "algorithm"))
        save_yaml(self.config_dir / "rollout_config.yaml", require_cfg(self.cfg, "rollout"))
        save_yaml(self.config_dir / "split_snapshot.yaml", load_yaml(self.root / require_cfg(self.cfg, "split_path")))

    def build_cases(self) -> List[EvalCase]:
        """Build all resolved evaluation cases from the split/config."""
        split_key = str(require_cfg(self.cfg, "split_key"))
        split_cfg = load_yaml(self.root / require_cfg(self.cfg, "split_path"))
        maps = map_names(split_cfg, split_key)
        seed = int(EVAL_SEED)
        num_start_points = int(self.num_start_points_per_map)
        raceline = str(require_cfg(self.cfg, "raceline"))
        opp_raceline = str(require_cfg(self.cfg, "opp_raceline"))
        rollout = require_cfg(self.cfg, "rollout")
        algorithm = require_cfg(self.cfg, "algorithm")

        cases: List[EvalCase] = []
        for map_name in maps:
            num_wp = load_num_waypoints(self.root, self.cfg, map_name, raceline)
            for start_id, wp_idx in enumerate(uniform_indices(num_wp, num_start_points)):
                cases.append(
                    EvalCase(
                        case_id=f"{clean_id(map_name)}_{split_key}_{start_id:03d}_seed{seed}",
                        split=split_key,
                        map_name=map_name,
                        raceline=raceline,
                        opp_raceline=opp_raceline,
                        ego_start_slot_idx=int(wp_idx) % num_wp,
                        interval_idx=int(require_key(rollout, "interval_idx")),
                        rand_seed=seed,
                        sim_duration=float(self.sim_duration),
                        v_global_limit=float(require_key(rollout, "v_global_limit")),
                        opp_speed_scale=float(require_key(rollout, "opp_speed_scale")),
                        num_agents=int(require_key(algorithm, "num_agents")),
                    )
                )
        save_yaml(self.case_dir / "resolved_cases.yaml", [asdict(case) for case in cases])
        return cases

    def run_cases(self, cases: Sequence[EvalCase]) -> List[Dict[str, Any]]:
        """Run all cases with the configured parallel backend."""
        requested = int(require_cfg(self.cfg, "num_workers"))
        if requested < 1:
            raise ValueError("num_workers must be >= 1")
        max_workers = min(requested, len(cases))
        backend = str(require_cfg(self.cfg, "parallel_backend")).lower()
        if backend not in {"thread", "process"}:
            raise ValueError(f"unsupported parallel_backend: {backend}")

        total_cases = len(cases)
        eval_start_time = time.perf_counter()
        if max_workers == 1:
            records: List[Dict[str, Any]] = []
            for finished, case in enumerate(cases, start=1):
                records.append(self.run_case(case))
                self.print_progress(finished, total_cases, eval_start_time)
            return records

        executor_cls = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor
        records: List[Dict[str, Any]] = []
        with executor_cls(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _run_single_case_worker,
                    {
                        "cfg": self.cfg,
                        "root": str(self.root),
                        "run_dir": str(self.run_dir),
                        "case": asdict(case),
                        "sim_duration": self.sim_duration,
                        "num_start_points_per_map": self.num_start_points_per_map,
                    },
                )
                for case in cases
            ]
            for finished, future in enumerate(as_completed(futures), start=1):
                records.append(future.result())
                self.print_progress(finished, total_cases, eval_start_time)
        return sorted(records, key=lambda r: str(r["case"]["case_id"]))

    def print_progress(self, finished: int, total_cases: int, eval_start_time: float) -> None:
        """Print evaluation progress and estimated remaining wall time."""
        elapsed = time.perf_counter() - eval_start_time
        avg_time = elapsed / finished
        remain = avg_time * (total_cases - finished)
        print(f"[EVAL] progress {finished}/{total_cases} remain={format_duration(remain)} elapsed={format_duration(elapsed)}", flush=True)

    def make_args(self, case: EvalCase) -> SimpleNamespace:
        """Create planner arguments for one case."""
        args = copy.deepcopy(require_cfg(self.cfg, "algorithm"))
        args.update(asdict(case))
        args["ego_idx"] = int(case.ego_start_slot_idx)
        del args["case_id"]
        del args["split"]

        # Batched raw evaluation never uses online rendering/video. Pairwise rows
        # are recomputed offline; this flag remains true only for compatibility
        # with planners that expect the attribute to exist.
        args.update({"render": False, "save_video": False, "collect_pairwise_rows": True})

        for key in ["planner_family", "method", "interaction_mode", "mppi_bias_mode_ego", "mppi_bias_mode_opp", "ibr_time"]:
            require_key(args, key)
        return SimpleNamespace(**args)

    def run_case(self, case: EvalCase) -> Dict[str, Any]:
        """Run one case and save only artifacts required by offline metrics."""
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        from players.planner.competitive_planner import run_competitive_players

        case_artifact_dir = self.artifact_dir / clean_id(case.map_name) / clean_id(case.case_id)
        case_artifact_dir.mkdir(parents=True, exist_ok=True)

        args = self.make_args(case)
        args.eval_case_id = case.case_id
        # Kept only for planner/generator compatibility. This script does not
        # build plots, dashboards, or videos.
        args.plot_data_dir = str(case_artifact_dir)
        args.offline_plot_data_dir = str(case_artifact_dir)

        case_start_time = time.perf_counter()
        metrics = run_competitive_players(args, return_metrics=True)
        wall_time = float(time.perf_counter() - case_start_time)

        trace_path = self.trace_dir / f"{case.case_id}.npy"
        trace_meta_path = self.trace_dir / f"{case.case_id}_meta.json"
        planned_trajs_path = case_artifact_dir / "planned_trajs.npz"
        boundary_offsets_path = case_artifact_dir / "boundary_offsets.npy"
        refline_xy_path = case_artifact_dir / "refline_xy.npy"
        render_artifact_path = case_artifact_dir / "render_artifact.json"

        save_trace_npy(trace_path, trace_meta_path, metrics["rollout_trace"])
        save_planned_trajs_npz(planned_trajs_path, metrics.get("planned_trajs", []))
        save_array_npy(boundary_offsets_path, metrics["boundary_offsets"], "boundary_offsets_npy_v1", self.run_dir)
        save_array_npy(refline_xy_path, np.asarray(metrics["refline_xy"])[:, :2], "refline_waypoints_npy_v1", self.run_dir)

        save_render_artifact_json(render_artifact_path, case, args)

        return {
            "case": asdict(case),
            "status": "ok",
            "error": "",
            "wall_time": wall_time,
            "trace_path": rel_to(trace_path, self.run_dir),
            "trace_meta_path": rel_to(trace_meta_path, self.run_dir),
            "planned_trajs_path": rel_to(planned_trajs_path, self.run_dir),
            "boundary_offsets_path": rel_to(boundary_offsets_path, self.run_dir),
            "refline_xy_path": rel_to(refline_xy_path, self.run_dir),
            "render_artifact_path": rel_to(render_artifact_path, self.run_dir),
            "artifact_dir": rel_to(case_artifact_dir, self.run_dir),
        }

    def write_raw_manifest(self, records: Sequence[Dict[str, Any]]) -> None:
        """Write the raw-run manifest consumed by the offline metric script."""
        rows: List[Dict[str, Any]] = []
        for record in records:
            case = record["case"]
            rows.append(
                {
                    "case_id": case["case_id"],
                    "map_name": case["map_name"],
                    "trace_path": record["trace_path"],
                    "trace_meta_path": record["trace_meta_path"],
                    "planned_trajs_path": record.get("planned_trajs_path", ""),
                    "boundary_offsets_path": record["boundary_offsets_path"],
                    "refline_xy_path": record["refline_xy_path"],
                    "artifact_dir": record["artifact_dir"],
                }
            )
        write_csv(self.metric_dir / "raw_episode_manifest.csv", rows, RAW_MANIFEST_COLUMNS)

    def build_summary(self, records: Sequence[Dict[str, Any]], elapsed: float) -> Dict[str, Any]:
        """Build a raw-run summary without metrics/dashboard outputs."""
        return {
            "run_dir": str(self.run_dir),
            "elapsed": float(elapsed),
            "num_cases": len(records),
            "raw_artifact_only": True,
            "metrics_computed": False,
            "dashboard_built": False,
            "video_rendered": False,
            "next_step": "python -m players.metrics.competitive_metrics --run-dir <run_dir> --project-root .",
            "files": {
                "raw_episode_manifest": str(self.metric_dir / "raw_episode_manifest.csv"),
                "traces": str(self.trace_dir),
                "planned_trajs": "artifacts/<map>/<case>/planned_trajs.npz",
                "render_artifacts": "artifacts/<map>/<case>/render_artifact.json",
                "artifacts": str(self.artifact_dir),
            },
        }

def _run_single_case_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run one case inside a worker process."""
    runner = CompetitiveRawEvalRunner(
        payload["cfg"],
        Path(payload["root"]),
        float(payload["sim_duration"]),
        int(payload["num_start_points_per_map"]),
    )
    runner.run_dir = Path(payload["run_dir"])
    runner.config_dir = runner.run_dir / "configs"
    runner.case_dir = runner.run_dir / "cases"
    runner.trace_dir = runner.run_dir / "traces"
    runner.artifact_dir = runner.run_dir / "artifacts"
    runner.metric_dir = runner.run_dir / "metrics"
    return runner.run_case(EvalCase(**payload["case"]))


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse arguments and run raw evaluation."""
    parser = argparse.ArgumentParser(description="Run competitive raw-data evaluation only")
    parser.add_argument("--eval-config", type=str, default="players/config/eval_config.yaml")
    parser.add_argument("--project-root", type=str, default=".")
    parser.add_argument("--sim-duration", type=float, required=True)
    parser.add_argument("--num-start-points-per-map", type=int, required=True)
    args = parser.parse_args(argv)

    config_path = Path(args.eval_config).resolve()
    project_root = Path(args.project_root).resolve()
    summary = CompetitiveRawEvalRunner(
        load_yaml(config_path),
        project_root,
        float(args.sim_duration),
        int(args.num_start_points_per_map),
    ).evaluate()
    print(json.dumps(json_ready(summary), indent=2))


if __name__ == "__main__":
    main()
