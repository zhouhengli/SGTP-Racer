import time
import numpy as np


def _collect_other_poses_from_obs(obs, agent_idx):
    rows = []
    num_agents = len(obs["poses_x"])
    for j in range(num_agents):
        if j == agent_idx:
            continue
        rows.append([
            float(obs["poses_x"][j]),
            float(obs["poses_y"][j]),
            float(obs["poses_theta"][j]),
            float(obs["linear_vels_x"][j]),
            float(obs["linear_vels_y"][j]),
        ])
    return np.asarray(rows, dtype=float)


def spliner_plan_once(planner, obs, agent_idx):
    traj, _ = planner.plan(
        pose_x=float(obs["poses_x"][agent_idx]),
        pose_y=float(obs["poses_y"][agent_idx]),
        pose_theta=float(obs["poses_theta"][agent_idx]),
        velocity=float(obs["linear_vels_x"][agent_idx]),
        opp_poses=_collect_other_poses_from_obs(obs, agent_idx),
    )
    return traj


def spliner_based_multi_generator(args, agent_planners, obs, params_dict):
    del params_dict

    mode = str(args.interaction_mode).lower()
    if mode != "nonreactive":
        raise ValueError("Spliner planner only supports interaction_mode='nonreactive'.")

    t0 = time.perf_counter()
    best_trajs = [
        spliner_plan_once(agent_planners[i], obs, i)
        for i in range(len(agent_planners))
    ]
    infer_time = time.perf_counter() - t0
    return best_trajs, infer_time
