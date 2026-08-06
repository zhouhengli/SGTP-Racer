"""Generate End2Race ego actions and MPPI opponent trajectories."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from players.planner.End2Race.end2race_policy import End2RacePlanner


def collect_other_poses_from_obs(obs: Dict, agent_idx: int) -> np.ndarray:
    """Return current poses of all agents except the controlled agent."""
    return np.asarray(
        [
            [float(obs["poses_x"][j]), float(obs["poses_y"][j]), float(obs["poses_theta"][j])]
            for j in range(len(obs["poses_x"]))
            if j != agent_idx
        ],
        dtype=float,
    )


def setup_end2race_planner(args, map_name, raceline, config_path, v_scale, ocp_conf=None, game_block_conf=None, biased_type=None):
    """Create an End2Race planner while borrowing MPPI map geometry for simulator compatibility."""
    del game_block_conf, biased_type
    from players.utils.planner_registry import planners
    geometry_planner, config_directory = planners["mppi"][0](
        args,
        map_name,
        raceline,
        config_path=config_path,
        v_scale=v_scale,
        ocp_conf=ocp_conf,
        game_block_conf=None,
        biased_type=getattr(args, "mppi_bias_mode_ego"),
    )
    model_path = "players/planner/End2Race/pretrained/pretrained.pth"
    return End2RacePlanner(args=args, geometry_planner=geometry_planner, model_path=model_path), config_directory


def plan_mppi_opponent(args, planner, obs: Dict, agent_idx: int, opp_pred_traj: Optional[List[np.ndarray]] = None) -> np.ndarray:
    """Plan one MPPI trajectory for one opponent agent."""
    result = planner.plan(
        float(obs["poses_x"][agent_idx]),
        float(obs["poses_y"][agent_idx]),
        float(obs["poses_theta"][agent_idx]),
        collect_other_poses_from_obs(obs, agent_idx),
        float(obs["linear_vels_x"][agent_idx]),
        opp_pred_traj=opp_pred_traj,
        post_select=True,
        bias_mode=str(getattr(args, "mppi_bias_mode_opp", "none")).lower(),
    )
    return result[0] if isinstance(result, tuple) else result


def end2race_mppi_generator(args, agent_planners: List, obs: Dict, params_dict: Dict):
    """Run direct End2Race policy inference for ego and MPPI planning for all opponents."""
    del params_dict
    t0 = time.time()
    best_trajs = [None for _ in agent_planners]
    ego_planner = agent_planners[0]
    ego_planner.policy_plan(obs, agent_idx=0)
    infer_time = time.time() - t0
    for agent_idx in range(1, len(agent_planners)):
        best_trajs[agent_idx] = plan_mppi_opponent(args, agent_planners[agent_idx], obs, agent_idx, opp_pred_traj=None)
    return best_trajs, infer_time


def compute_direct_ego_mppi_opp_actions(obs: Dict, agent_planners: List, best_trajs: List, steer_limits) -> np.ndarray:
    """Convert the direct ego action and planned opponent trajectories into simulator actions."""
    actions = np.zeros((len(agent_planners), 2), dtype=float)
    for agent_idx, planner in enumerate(agent_planners):
        if getattr(planner, "direct_action", False):
            actions[agent_idx] = np.asarray(planner.last_action, dtype=float).reshape(2)
        else:
            steer, speed = planner.tracker.plan(
                obs["poses_x"][agent_idx],
                obs["poses_y"][agent_idx],
                obs["poses_theta"][agent_idx],
                obs["linear_vels_x"][agent_idx],
                best_trajs[agent_idx],
            )
            actions[agent_idx] = [float(np.clip(steer, steer_limits[0], steer_limits[1])), float(speed)]
    return actions
