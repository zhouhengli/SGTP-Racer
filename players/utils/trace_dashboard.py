#!/usr/bin/env python3
"""Render replay videos from saved evaluation artifacts."""

from __future__ import annotations

import json
import os
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np


os.environ.setdefault("MPLBACKEND", "Agg")


def clean_name(text: str) -> str:
    """Convert text into a file-name-safe string."""
    for old, new in [("/", "_"), ("\\", "_"), (" ", "_"), (":", "_"), (";", "_"), (",", "_")]:
        text = text.replace(old, new)
    return text


def load_json(path: Path) -> Dict[str, Any]:
    """Load one JSON mapping."""
    with path.open("r") as f:
        return dict(json.load(f))


def resolve_run_path(value: Any, run_dir: Path) -> Path:
    """Resolve one path relative to the run directory."""
    path = Path(str(value))
    return path if path.is_absolute() else run_dir / path


def resolve_project_path(value: Any, project_root: Path) -> Path:
    """Resolve one path relative to the project root."""
    path = Path(str(value))
    return path if path.is_absolute() else project_root / path


@contextmanager
def suppress_renderer_deprecation_warnings():
    """Suppress dependency deprecation warnings emitted by the renderer."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*ImageIO v3.*", category=DeprecationWarning)
        warnings.filterwarnings("ignore", message=".*mode.*deprecated.*", category=DeprecationWarning)
        yield


def import_renderer(project_root: Path):
    """Import the offline renderer from the project root."""
    resolved_root = project_root.resolve()
    if str(resolved_root) not in sys.path:
        sys.path.insert(0, str(resolved_root))
    with suppress_renderer_deprecation_warnings():
        import players.utils.offline_save_video as offline_video
    try:
        import imageio.v2 as imageio_v2

        offline_video.imageio = imageio_v2
    except Exception:
        pass
    return offline_video.OfflineRaceVideoRenderer, offline_video.VehicleState


def load_planned_trajs(planned_trajs_path: Path) -> Tuple[Dict[str, Any], np.lib.npyio.NpzFile]:
    """Load planned-trajectory metadata and arrays."""
    data = np.load(planned_trajs_path, allow_pickle=False)
    metadata = json.loads(str(data["metadata_json"].item()))
    return metadata, data


def planned_trajs_for_frame(
    planned_meta: Dict[str, Any],
    planned_npz: np.lib.npyio.NpzFile,
    frame_idx: int,
) -> List[Optional[np.ndarray]]:
    """Return the latest planned trajectories available for one trace frame."""
    records = planned_meta["records"]
    if not records:
        return [None]
    selected_record = records[0]
    for record in records:
        if int(record["trace_start_index"]) <= int(frame_idx):
            selected_record = record
        else:
            break

    trajs: List[Optional[np.ndarray]] = []
    for agent_meta in selected_record["agents"]:
        key = agent_meta["key"]
        trajs.append(None if key is None else np.asarray(planned_npz[str(key)]))
    return trajs


def load_renderer_limits(artifact: Dict[str, Any]) -> Tuple[float, float]:
    """Load the speed and steering limits required by the HUD."""
    hud_limits = artifact["hud_limits"]
    return float(hud_limits["speed_max"]), float(hud_limits["steer_limit"])


def render_info_for_frame(
    trace: np.ndarray,
    channels: Dict[str, int],
    map_name: str,
    frame_idx: int,
) -> Dict[str, Any]:
    """Build the HUD values used for one rendered frame."""
    return {
        "ego_steer": float(trace[frame_idx, 0, channels["action_steer"]]),
        "ego_speed": float(trace[frame_idx, 0, channels["action_speed"]]),
        "track_name": map_name,
    }


def video_frame_indices(num_steps: int, frame_step: int) -> List[int]:
    """Return trace indices matching the configured video-capture cadence."""
    if int(num_steps) <= 0:
        return []
    step = max(1, int(frame_step))
    indices = list(range(step - 1, int(num_steps), step))
    return indices if indices else [int(num_steps) - 1]


def render_case_video(
    row: Mapping[str, Any],
    run_dir: Path,
    out_dir: Path,
    project_root: Path,
) -> Path:
    """Render one replay video from the saved episode artifacts."""
    artifact_dir = resolve_run_path(row["artifact_dir"], run_dir)
    artifact = load_json(artifact_dir / "render_artifact.json")
    hud_speed_max, steer_limit = load_renderer_limits(artifact)

    trace_meta = load_json(resolve_run_path(row["trace_meta_path"], run_dir))
    channels = {name: index for index, name in enumerate(trace_meta["channels"])}
    trace = np.load(resolve_run_path(row["trace_path"], run_dir), allow_pickle=False)
    refline_xy = np.load(resolve_run_path(row["refline_xy_path"], run_dir), allow_pickle=False)
    boundary_offsets = np.load(resolve_run_path(row["boundary_offsets_path"], run_dir), allow_pickle=False)
    planned_meta, planned_npz = load_planned_trajs(resolve_run_path(row["planned_trajs_path"], run_dir))

    render_config = artifact["render_config"]
    map_yaml_path = resolve_project_path(artifact["assets"]["map_yaml_path"], project_root)
    OfflineRaceVideoRenderer, VehicleState = import_renderer(project_root)

    with suppress_renderer_deprecation_warnings():
        renderer = OfflineRaceVideoRenderer(
            map_yaml_path=str(map_yaml_path),
            refline_xy=refline_xy[:, :2],
            boundary_offsets=boundary_offsets,
            output_fps=int(render_config["out_fps"]),
            vehicle_length=float(render_config["vehicle_length"]),
            vehicle_width=float(render_config["vehicle_width"]),
            hud_speed_max=hud_speed_max,
            steer_limit=steer_limit,
        )

    num_steps = int(trace.shape[0])
    num_agents = int(trace.shape[1])
    frame_step = int(render_config["video_capture_every"])
    map_name = str(row["map_name"])

    for frame_idx in video_frame_indices(num_steps, frame_step):
        vehicle_states = [
            VehicleState(
                x=float(trace[frame_idx, agent_idx, channels["poses_x"]]),
                y=float(trace[frame_idx, agent_idx, channels["poses_y"]]),
                heading=float(trace[frame_idx, agent_idx, channels["poses_theta"]]),
            )
            for agent_idx in range(num_agents)
        ]
        renderer.capture_multi(
            vehicle_states=vehicle_states,
            sim_time=float(trace[frame_idx, 0, channels["t"]]),
            vehicle_trajs=planned_trajs_for_frame(planned_meta, planned_npz, frame_idx),
            render_info=render_info_for_frame(trace, channels, map_name, frame_idx),
        )

    video_path = out_dir / f"{clean_name(str(row['case_id']))}_random_replay.mp4"
    try:
        with suppress_renderer_deprecation_warnings():
            renderer.save(str(video_path))
    finally:
        renderer.close()
        planned_npz.close()
    return video_path
