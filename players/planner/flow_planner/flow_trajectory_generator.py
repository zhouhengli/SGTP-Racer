from __future__ import annotations

import copy
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from numba import njit
import torch

from players.utils.common import load_config, get_map_paths, project_point_to_centerline
from players.planner.controller.pure_pursuit import PurePursuitPlanner

MPPI_POST_SELECT = True

@dataclass
class PlannerRuntime:
    """Store the flow model runtime and the tracker-facing planner attributes."""
    runner: Any
    tracker: Any
    map_path: str
    map_directory: str
    raceline_path: str


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _smooth_velocity(v: np.ndarray, alpha: float = 0.8) -> np.ndarray:
    """Smooth a one-dimensional velocity signal with an exponential moving average."""
    v_smooth = np.copy(v)
    for i in range(1, len(v_smooth)):
        v_smooth[i] = alpha * v_smooth[i - 1] + (1.0 - alpha) * v_smooth[i]
    return v_smooth


def traj_xytheta_to_xyv(traj: np.ndarray, mpc_dt: Optional[float] = None, smooth: bool = True) -> np.ndarray:
    """Convert an xytheta trajectory into an xyv trajectory for tracker consumption."""
    if mpc_dt is None or float(mpc_dt) <= 0.0:
        raise ValueError("mpc_dt must be a positive value")
    traj = np.asarray(traj, dtype=float)
    if traj.ndim != 2 or traj.shape[1] < 2:
        raise ValueError(f"traj must have shape [T, >=2], got {traj.shape}")
    x = traj[:, 0]
    y = traj[:, 1]
    dx = np.diff(x)
    dy = np.diff(y)
    v = np.sqrt(dx * dx + dy * dy) / float(mpc_dt)
    v = np.concatenate([v, [v[-1] if len(v) else 0.0]])
    if smooth:
        v = _smooth_velocity(v)
    return np.stack([x, y, v], axis=1)


def load_racing_line_fields(process_config_path: str) -> Tuple[List[str], Dict[str, Any]]:
    """Load flow conditioning field names and preprocessing metadata."""
    with open(process_config_path, "r", encoding="utf-8") as f:
        init_data = json.load(f)
    return list(init_data["racing_line"]["fields"]), init_data


def _wpnts_to_array(data: Any, key: str, fields: List[str]) -> np.ndarray:
    """Convert a global waypoint JSON payload into a dense numeric array."""

    payload = data.get(key, data) if isinstance(data, dict) else data
    if isinstance(payload, dict) and "wpnts" in payload:
        payload = payload["wpnts"]
    rows = []
    for item in payload:
        if isinstance(item, dict):
            rows.append([float(item[name]) for name in fields])
        else:
            rows.append([float(getattr(item, name)) for name in fields])
    return np.asarray(rows, dtype=float)


def build_openloop_runner(config_path: str, ckpt_path: str, device: str = "cuda") -> Any:
    """Create and initialize the FlowRacingPlanner inference runtime."""
    from players.planner.flow_planner.planner import FlowRacingPlanner

    planner = FlowRacingPlanner(
        config_path=config_path,
        ckpt_path=ckpt_path,
        enable_ema=True,
        device=device,
        use_cfg=True,
    )
    planner.initialize()
    return SimpleNamespace(
        planner=planner,
        tracker=None,
        config_path=config_path,
        ckpt_path=ckpt_path,
        device=device,
    )


def setup_runtime(
    args,
    map_name: str,
    raceline_file: str,
    config_path: Optional[str] = None,
    v_scale: Optional[float] = None,
    ocp_conf: Optional[Dict[str, Any]] = None,
    game_block_conf: Optional[Dict[str, Any]] = None,
    biased_type: Optional[str] = None,
) -> Tuple[PlannerRuntime, str]:
    """Create the ego flow runtime while matching the existing MPPI setup interface."""
    del game_block_conf, biased_type
    config = load_config(config_path)
    if ocp_conf is None:
        with open(args.ocp_config, "r") as f:
            ocp_conf = yaml.safe_load(f)

    map_directory, map_path = get_map_paths(map_name)
    raceline_path = os.path.join(map_directory, f"{map_name}_{raceline_file}.csv")

    flow_config_path = str(getattr(args, "flow_config_path", "checkpoint/config.yaml"))
    flow_ckpt_path = str(getattr(args, "flow_ckpt_path", "checkpoint/latest.pth"))
    flow_device = getattr(args, "flow_device", None)
    if flow_device is None:
        flow_device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

    runner = build_openloop_runner(config_path=flow_config_path, ckpt_path=flow_ckpt_path, device=str(flow_device))
    tracker = PurePursuitPlanner(config, raceline_path, wb=args.wheel_base)
    runner.tracker = tracker
    runner.planner.tracker = tracker
    runner.planner.goal_grid = None
    runner.planner.best_traj = None

    runtime = PlannerRuntime(
        runner=runner,
        tracker=tracker,
        map_path=map_path,
        map_directory=map_directory,
        raceline_path=raceline_path,
    )

    waypoints_raw = np.loadtxt(raceline_path, delimiter=";", skiprows=1)
    runtime.waypoints = np.vstack(
        (
            waypoints_raw[:, 3],
            waypoints_raw[:, 4],
            waypoints_raw[:, 9],
            waypoints_raw[:, 7],
            waypoints_raw[:, 1],
        )
    ).T
    if v_scale is not None:
        runtime.waypoints[:, 2] *= float(v_scale)
    runtime.d_right_left = np.vstack((waypoints_raw[:, 5], waypoints_raw[:, 6]))
    runtime.tracker_steps = int(args.tracker_steps)
    runtime.mppi_N = int(ocp_conf.get("N"))
    runtime.mppi_duration = float(ocp_conf.get("horizon"))
    runtime.mppi_dt = float(runtime.mppi_duration / runtime.mppi_N)
    runtime.N = runtime.mppi_N
    runtime.duration = runtime.mppi_duration
    runtime.dt = runtime.mppi_dt

    return runtime, map_directory


def load_racing_line_dict(track_name: str, process_config_path: str) -> Dict[str, Any]:
    """Load the racing-line dictionary used to build flow conditioning tensors."""
    fields, init_data = load_racing_line_fields(process_config_path)
    query_forward = init_data["environment"]["query_forward_m"]
    query_back = init_data["environment"]["query_back_m"]
    racing_line_points = init_data["racing_line"]["racing_line_points"]
    track_file = Path("MapZoo") / track_name / "global_waypoints.json"

    with track_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    racing_line_array = _wpnts_to_array(data, "global_traj_wpnts_iqp", fields)
    field_to_idx = {name: i for i, name in enumerate(fields)}

    return {
        "track_length": float(racing_line_array[-1, field_to_idx["s_m"]]),
        "x_ref": racing_line_array[:, field_to_idx["x_m"]],
        "y_ref": racing_line_array[:, field_to_idx["y_m"]],
        "psi_ref": racing_line_array[:, field_to_idx["psi_rad"]],
        "s_ref": racing_line_array[:, field_to_idx["s_m"]],
        "vx_ref": racing_line_array[:, field_to_idx.get("vx_mps", 9)],
        "whole_array": racing_line_array,
        "field_to_idx": field_to_idx,
        "x_idx": field_to_idx["x_m"],
        "y_idx": field_to_idx["y_m"],
        "psi_idx": field_to_idx["psi_rad"],
        "s_idx": field_to_idx["s_m"],
        "d_right_idx": field_to_idx["d_right"],
        "d_left_idx": field_to_idx["d_left"],
        "vx_idx": field_to_idx.get("vx_mps", 9),
        "query_forward": float(query_forward),
        "query_back": float(query_back),
        "racing_line_points": int(racing_line_points),
        "opp_radius": init_data["environment"].get("opp_radius"),
        "fut_time_horizon_sec": float(init_data["time"]["fut_time_horizon_sec"]),
        "num_fut_poses_cond": int(init_data["time"]["num_fut_poses_cond"]),
    }


def build_racing_segment(
    racing_line_dict: Dict[str, Any],
    s_ego: float,
    back: Optional[float] = None,
    forward: Optional[float] = None,
    num_points: Optional[int] = None,
) -> np.ndarray:
    """Build the fixed-length racing-line segment consumed by the flow model."""
    track_length = float(racing_line_dict["track_length"])
    s_ref = np.asarray(racing_line_dict["s_ref"], dtype=float)
    arr = np.asarray(racing_line_dict["whole_array"], dtype=float)
    back = float(racing_line_dict["query_back"] if back is None else back)
    forward = float(racing_line_dict["query_forward"] if forward is None else forward)
    num_points = int(racing_line_dict["racing_line_points"] if num_points is None else num_points)

    d_fwd = (s_ref - float(s_ego)) % track_length
    distance_buffer = 1.0
    mask = (d_fwd <= (forward + distance_buffer)) | (d_fwd >= track_length - (back + distance_buffer))
    if np.count_nonzero(mask) < 2:
        mask = np.ones_like(s_ref, dtype=bool)

    segment = arr[mask][:, [
        racing_line_dict["x_idx"],
        racing_line_dict["y_idx"],
        racing_line_dict["d_right_idx"],
        racing_line_dict["d_left_idx"],
        racing_line_dict["vx_idx"],
        racing_line_dict["psi_idx"],
    ]]
    s_seg = arr[mask][:, racing_line_dict["s_idx"]]

    def interpolate_segment_fixed_length_no_extrap(
            segment: np.ndarray,
            s_seg: np.ndarray,
            s_ego: float,
            track_length: float,
            K: int,
            back_dist: float = None,
            forward_dist: float = None,
        ):
            """
            Args:
                segment: (N, 6), [x, y, d_right, d_left, vx, psi]
                s_seg: (N,)
                s_ego: scalar
                track_length: scalar
                K: output length

            Returns:
                segment_interp: (K, 6)
            """
            segment = np.asarray(segment, dtype=float)
            s_seg = np.asarray(s_seg, dtype=float)

            if segment.shape[0] != s_seg.shape[0]:
                raise ValueError(
                    f"segment and s_seg must have the same length, got {segment.shape[0]} vs {s_seg.shape[0]}"
                )
            if segment.shape[0] < 2:
                raise ValueError("Need at least 2 points to interpolate without extrapolation.")

            s_rel = s_seg - s_ego
            s_rel = (s_rel + 0.5 * track_length) % track_length - 0.5 * track_length

            order = np.argsort(s_rel)
            s_rel = s_rel[order]
            segment = segment[order]

            d_rel_u, idx = np.unique(s_rel, return_index=True)
            segment_u = segment[idx]

            if d_rel_u.size < 2:
                raise ValueError("Not enough unique s_rel samples to interpolate.")

            d_target = np.linspace(-back_dist, forward_dist, K)

            eps = 1e-9
            if d_rel_u[0] > d_target[0] + eps or d_rel_u[-1] < d_target[-1] - eps:
                raise ValueError("Interpolation would require extrapolation.")

            D = segment_u.shape[1]
            theta_idx = D - 1 

            out_cols = []
            for i in range(D):
                if i == theta_idx:
                    continue
                out_cols.append(np.interp(d_target, d_rel_u, segment_u[:, i]))
            out = np.stack(out_cols, axis=1) if out_cols else np.zeros((K, 0), dtype=float)

            theta = segment_u[:, theta_idx]
            c = np.cos(theta)
            s = np.sin(theta)
            c_i = np.interp(d_target, d_rel_u, c)
            s_i = np.interp(d_target, d_rel_u, s)
            theta_i = np.arctan2(s_i, c_i)

            if out.shape[1] == 0:
                segment_interp = theta_i.reshape(-1, 1)
            else:
                segment_interp = np.concatenate([out, theta_i.reshape(-1, 1)], axis=1)

            return segment_interp

    
    return interpolate_segment_fixed_length_no_extrap(
        segment=segment,
        s_seg=s_seg,
        s_ego=float(s_ego),
        track_length=track_length,
        K=num_points,
        back_dist=back,
        forward_dist=forward,
    )

def build_model_sample(
    obs: Dict[str, Any],
    racing_line_dict: Dict[str, Any],
    agent_idx: int,
    other_idx: int,
    query_forward: float,
    query_back: float,
    racing_line_points: int,
) -> Dict[str, Any]:
    """Build one flow-model sample from the current gym observation and raceline context."""
    raceline_xy = np.stack([racing_line_dict["x_ref"], racing_line_dict["y_ref"]], axis=1)
    progress, nearest_idx = project_point_to_centerline(
        np.array([obs["poses_x"][agent_idx], obs["poses_y"][agent_idx]], dtype=float),
        raceline_xy,
    )
    seg = build_racing_segment(
        racing_line_dict,
        progress,
        back=query_back,
        forward=query_forward,
        num_points=racing_line_points,
    )

    return {
        "ego_current_state": np.array([
            obs["poses_x"][agent_idx],
            obs["poses_y"][agent_idx],
            obs["poses_theta"][agent_idx],
            obs["linear_vels_x"][agent_idx],
            obs["poses_x"][other_idx],
            obs["poses_y"][other_idx],
            obs["poses_theta"][other_idx],
            obs["linear_vels_x"][other_idx],
        ], dtype=np.float32),
        "ego_agent_future": np.zeros((40, 5), dtype=np.float32),
        "neighbor_agents_past": np.zeros((3, 10, 5), dtype=np.float32),
        "neighbor_agents_future": np.zeros((3, 15, 5), dtype=np.float32),
        "racing_line_seg": seg.astype(np.float32),
        "nearest_idx": int(nearest_idx),
    }


def infer_traj(runtime: PlannerRuntime, sample: Dict[str, Any], args) -> np.ndarray:
    """Run the flow model and return raw xytheta candidate trajectories."""
    inputs = runtime.runner.planner.planner_input_to_model_inputs(sample, int(args.flow_batch))
    outputs = runtime.runner.planner.infer_planner_trajectory(inputs)
    path = runtime.runner.planner.outputs_to_trajectory(outputs)
    return np.asarray(path, dtype=np.float32)


def _prepare_opponent_future_trajectory(opp_heading: np.ndarray, opp_traj_xyv: np.ndarray) -> np.ndarray:
    """Convert an opponent xyv trajectory and heading vector into xy-heading-v format."""
    opp_heading = np.asarray(opp_heading, dtype=float)
    opp_traj_xyv = np.asarray(opp_traj_xyv, dtype=float)
    return np.concatenate([opp_traj_xyv[:, :2], opp_heading[:, None], opp_traj_xyv[:, 2:3]], axis=1)


def _process_opponent_trajectory(
    opp_traj: np.ndarray,
    target_duration: float,
    target_len: int,
    opp_duration: float,
    interp_kind: str = "linear",
    vy_threshold: float = 0.05,
) -> np.ndarray:
    """Resample an opponent xy-heading-v trajectory into the flow future-condition format."""
    if interp_kind != "linear":
        raise ValueError("Only linear interpolation is supported in this self-contained flow module")
    opp_traj = np.asarray(opp_traj, dtype=float)
    if opp_traj.ndim != 2 or opp_traj.shape[1] != 4:
        raise ValueError(f"opp_traj must have shape [N, 4], got {opp_traj.shape}")
    if opp_traj.shape[0] < 2 or int(target_len) < 2:
        raise ValueError("Need at least two source and target points")
    if float(target_duration) > float(opp_duration):
        raise ValueError(f"target_duration={target_duration} exceeds opp_duration={opp_duration}")

    n = opp_traj.shape[0]
    t_src = np.linspace(0.0, float(opp_duration), n)
    t_query = np.linspace(0.0, float(target_duration), int(target_len))
    heading_unwrapped = np.unwrap(opp_traj[:, 2])

    x_new = np.interp(t_query, t_src, opp_traj[:, 0])
    y_new = np.interp(t_query, t_src, opp_traj[:, 1])
    heading_new = _wrap_to_pi(np.interp(t_query, t_src, heading_unwrapped))

    dt = float(target_duration) / (int(target_len) - 1)
    vx_world = np.gradient(x_new, dt)
    vy_world = np.gradient(y_new, dt)
    cos_h = np.cos(heading_new)
    sin_h = np.sin(heading_new)
    vx_body = cos_h * vx_world + sin_h * vy_world
    vy_body = -sin_h * vx_world + cos_h * vy_world
    vy_body[np.abs(vy_body) < float(vy_threshold)] = 0.0
    return np.stack([x_new, y_new, heading_new, vx_body, vy_body], axis=-1)


@njit(cache=True)
def flow_resample_batch_and_make_state_njit(flow_xytheta, source_dt, target_dt, target_steps):
    """Resample flow xytheta trajectories and convert them into xythetav state batches."""
    K = flow_xytheta.shape[0]
    T = flow_xytheta.shape[1]
    flow_xytheta_resampled = np.empty((K, target_steps, 3), dtype=np.float64)
    flow_state_batch = np.empty((K, target_steps, 4), dtype=np.float64)
    two_pi = 2.0 * math.pi
    src_t_end = (T - 1) * source_dt

    for k in range(K):
        heading_unwrapped = np.empty(T, dtype=np.float64)
        heading_unwrapped[0] = flow_xytheta[k, 0, 2]
        offset = 0.0
        prev_raw = flow_xytheta[k, 0, 2]
        for i in range(1, T):
            raw = flow_xytheta[k, i, 2]
            delta = raw - prev_raw
            if delta > math.pi:
                offset -= two_pi
            elif delta < -math.pi:
                offset += two_pi
            heading_unwrapped[i] = raw + offset
            prev_raw = raw

        for j in range(target_steps):
            t = j * target_dt
            if t > src_t_end:
                t = src_t_end
            idx = int(t / source_dt)
            if idx >= T - 1:
                idx = T - 2
                alpha = 1.0
            else:
                alpha = (t - idx * source_dt) / source_dt
            x = flow_xytheta[k, idx, 0] + alpha * (flow_xytheta[k, idx + 1, 0] - flow_xytheta[k, idx, 0])
            y = flow_xytheta[k, idx, 1] + alpha * (flow_xytheta[k, idx + 1, 1] - flow_xytheta[k, idx, 1])
            heading = heading_unwrapped[idx] + alpha * (heading_unwrapped[idx + 1] - heading_unwrapped[idx])
            theta = math.atan2(math.sin(heading), math.cos(heading))
            flow_xytheta_resampled[k, j, 0] = x
            flow_xytheta_resampled[k, j, 1] = y
            flow_xytheta_resampled[k, j, 2] = theta
            flow_state_batch[k, j, 0] = x
            flow_state_batch[k, j, 1] = y
            flow_state_batch[k, j, 2] = theta
            flow_state_batch[k, j, 3] = 0.0

        for j in range(1, target_steps):
            dx = flow_xytheta_resampled[k, j, 0] - flow_xytheta_resampled[k, j - 1, 0]
            dy = flow_xytheta_resampled[k, j, 1] - flow_xytheta_resampled[k, j - 1, 1]
            flow_state_batch[k, j, 3] = math.sqrt(dx * dx + dy * dy) / target_dt
        flow_state_batch[k, 0, 3] = flow_state_batch[k, 1, 3]

    return flow_xytheta_resampled, flow_state_batch



def _ensure_flow_runtime(args, planner, agent_idx: int) -> None:
    """Create or reuse the cached flow runtime attached to the planner."""
    if int(agent_idx) == 0:
        raceline_file = args.raceline
        v_scale = args.v_global_limit
    else:
        raceline_file = args.opp_raceline
        v_scale = args.opp_speed_scale * args.v_global_limit

    runtime_cache_key = (
        str(args.map_name),
        str(raceline_file),
        str(args.config),
        str(getattr(args, "ocp_config", "")),
        float(v_scale),
        float(args.wheel_base),
        str(getattr(args, "flow_config_path", "checkpoint/config.yaml")),
        str(getattr(args, "flow_ckpt_path", "checkpoint/latest.pth")),
    )
    cached_runtime = getattr(planner, "flow_runtime", None)
    cached_key = getattr(planner, "_flow_runtime_cache_key", None)
    if cached_runtime is not None and cached_key == runtime_cache_key:
        if not hasattr(planner, "d_right_left") and hasattr(cached_runtime, "d_right_left"):
            planner.d_right_left = cached_runtime.d_right_left
        return

    planner.flow_runtime, _ = setup_runtime(
        args=args,
        map_name=args.map_name,
        raceline_file=raceline_file,
        config_path=args.config,
        v_scale=v_scale,
    )
    planner._flow_runtime_cache_key = runtime_cache_key
    planner.d_right_left = planner.flow_runtime.d_right_left


def num_agents_from_obs(obs: Dict[str, Any]) -> int:
    """Return the number of agents encoded in a gym observation."""
    return len(obs["poses_x"])


def collect_other_poses_from_obs(obs: Dict[str, Any], agent_idx: int) -> np.ndarray:
    """Return current poses of all non-controlled agents as [M, 3]."""
    return np.asarray(
        [
            [float(obs["poses_x"][j]), float(obs["poses_y"][j]), float(obs["poses_theta"][j])]
            for j in range(num_agents_from_obs(obs))
            if j != int(agent_idx)
        ],
        dtype=float,
    )


def _nearest_opponent_indices(obs: Dict[str, Any], ego_idx: int, max_opps: int) -> List[int]:
    """Select nearest opponents by Euclidean distance to ego."""
    ego_xy = np.asarray([obs["poses_x"][ego_idx], obs["poses_y"][ego_idx]], dtype=float)
    order = []
    for j in range(num_agents_from_obs(obs)):
        if j == ego_idx:
            continue
        opp_xy = np.asarray([obs["poses_x"][j], obs["poses_y"][j]], dtype=float)
        order.append((float(np.linalg.norm(ego_xy - opp_xy)), j))
    order.sort(key=lambda item: item[0])
    return [j for _, j in order[: int(max_opps)]]


def build_flow_context(args) -> Dict[str, Any]:
    """Build per-episode flow context that should be merged into params_dict."""
    raceline_dict = load_racing_line_dict(args.map_name, args.flow_cond_config)
    return {
        "raceline_dict": raceline_dict,
        "query_forward": raceline_dict["query_forward"],
        "query_back": raceline_dict["query_back"],
        "racing_line_points": raceline_dict["racing_line_points"],
        "fut_time_horizon_sec": raceline_dict["fut_time_horizon_sec"],
        "num_fut_poses_cond": raceline_dict["num_fut_poses_cond"],
    }


def generate_flow_candidates_batch(
    args,
    ego_planner,
    obs: Dict[str, Any],
    params_dict: Dict[str, Any],
    opponent_prediction_trajectory_by_agent_idx: Dict[int, np.ndarray],
    opponent_prediction_heading_by_agent_idx: Dict[int, np.ndarray],
) -> Dict[str, Any]:
    """Generate ego flow candidates conditioned on MPPI opponent predictions."""
    t0 = time.time()

    ego_idx = int(params_dict.get("ego_agent_idx"))
    flow_batch = int(args.flow_batch)
    selected_opps = _nearest_opponent_indices(obs, ego_idx, int(args.flow_max_cond_opps))
    if not selected_opps:
        raise ValueError("Flow generator requires at least one opponent")
    primary_opp = selected_opps[0]

    # t0 = time.time()
    base_sample = build_model_sample(
        obs,
        params_dict["raceline_dict"],
        ego_idx,
        primary_opp,
        params_dict["query_forward"],
        params_dict["query_back"],
        params_dict["racing_line_points"],
    )

    ego_sample = copy.deepcopy(base_sample)
    neighbor_base = np.asarray(ego_sample["neighbor_agents_future"])
    num_slots, num_future_steps, num_future_dims = neighbor_base.shape
    neighbor_future_batch = np.zeros((flow_batch, num_slots, num_future_steps, num_future_dims), dtype=neighbor_base.dtype)

    for slot, opp_idx in enumerate(selected_opps[:num_slots]):
        if opp_idx not in opponent_prediction_trajectory_by_agent_idx:
            continue
        opp_planner = params_dict["agent_planners"][opp_idx]
        opp_traj = np.asarray(opponent_prediction_trajectory_by_agent_idx[opp_idx], dtype=float)
        opp_heading = np.asarray(opponent_prediction_heading_by_agent_idx[opp_idx], dtype=float)
        opp_duration = float(getattr(opp_planner, "duration"))
        opponent_condition = _process_opponent_trajectory(
            opp_traj=_prepare_opponent_future_trajectory(opp_heading, opp_traj),
            target_duration=float(params_dict["fut_time_horizon_sec"]),
            target_len=int(params_dict["num_fut_poses_cond"]),
            opp_duration=opp_duration,
            interp_kind="linear",
            vy_threshold=0.05,
        )
        neighbor_future_batch[:, slot, :, :] = opponent_condition[None, :, :]

    ego_sample["neighbor_agents_future"] = neighbor_future_batch
    flow_xytheta = np.asarray(infer_traj(ego_planner.flow_runtime, ego_sample, args=args))
    target_dt = float(getattr(ego_planner, "dt"))
    target_steps = int(getattr(ego_planner, "N")) + 1
    _, flow_state_batch = flow_resample_batch_and_make_state_njit(
        np.ascontiguousarray(flow_xytheta, dtype=np.float64),
        float(args.mpc_dt),
        target_dt,
        target_steps,
    )
    infer_time = time.time() - t0

    return {
        "flow_state_batch": flow_state_batch,
        "infer_time": infer_time,
    }


def flow_based_generator(args, ego_planner, opp_planner, obs: Dict[str, Any], params_dict: Dict[str, Any]):
    """Return the ego flow trajectory and the same opponent MPPI trajectories used as flow conditions."""
    ego_idx = int(params_dict["ego_agent_idx"])
    # t0 = time.time()
    if ego_idx != 0:
        raise ValueError(f"Flow-based generator currently supports ego_idx=0, got {ego_idx}")
    # this function is to prevent redundant runtime initialization when the generator is called multiple times per episode, which can happen when the planner is used in a multi-agent rollout setting with shared planners
    _ensure_flow_runtime(args, ego_planner, ego_idx)

    agent_planners = params_dict.get("agent_planners")
    opp_indices = [agent_idx for agent_idx in range(num_agents_from_obs(obs)) if agent_idx != ego_idx]
    if not opp_indices:
        raise ValueError("Flow generator requires at least one opponent")

    
    opponent_trajs: Dict[int, np.ndarray] = {}
    opponent_headings: Dict[int, np.ndarray] = {}
    for opp_idx in opp_indices:
        opp_planner_i = agent_planners[opp_idx] if agent_planners is not None else opp_planner
        opp_traj, opp_heading = opp_planner_i.plan(
            float(obs["poses_x"][opp_idx]),
            float(obs["poses_y"][opp_idx]),
            float(obs["poses_theta"][opp_idx]),
            collect_other_poses_from_obs(obs, opp_idx),
            float(obs["linear_vels_x"][opp_idx]),
            opp_pred_traj=None,
            post_select=MPPI_POST_SELECT,
            bias_mode="none",
        )
        opponent_trajs[int(opp_idx)] = np.asarray(opp_traj, dtype=float)
        opponent_headings[int(opp_idx)] = np.asarray(opp_heading, dtype=float)

    
    final_pack = generate_flow_candidates_batch(args, ego_planner, obs, params_dict, opponent_trajs, opponent_headings)
    # print(f"Flow inference and processing took {time.time() - t0:.3f} seconds")
    infer_time = final_pack["infer_time"]

    ego_planner.infer_time = final_pack["infer_time"]
    best_state = np.asarray(final_pack["flow_state_batch"][0], dtype=float)
    ego_traj = np.stack([best_state[:, 0], best_state[:, 1], best_state[:, 3]], axis=-1)
    return ego_traj, opponent_trajs, infer_time

