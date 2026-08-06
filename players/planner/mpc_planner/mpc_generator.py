import time

import numpy as np
from players.planner.planner_generators import get_interaction_mode, collect_other_trajs


def collect_other_poses_from_obs(obs, agent_idx):
    rows = []
    for j in range(len(obs["poses_x"])):
        if j == agent_idx:
            continue
        vx = float(obs.get("linear_vels_x", [0.0] * len(obs["poses_x"]))[j])
        vy = float(obs.get("linear_vels_y", [0.0] * len(obs["poses_x"]))[j])
        rows.append([
            float(obs["poses_x"][j]),
            float(obs["poses_y"][j]),
            float(obs["poses_theta"][j]),
            vx,
            vy,
        ])
    return np.asarray(rows, dtype=float)


def mpc_plan_once(planner, obs, agent_idx, opp_pred_trajs):
    # t0 = time.time()
    traj, _ = planner.plan(
        pose_x=float(obs["poses_x"][agent_idx]),
        pose_y=float(obs["poses_y"][agent_idx]),
        pose_theta=float(obs["poses_theta"][agent_idx]),
        velocity=float(obs["linear_vels_x"][agent_idx]),
        opp_poses=collect_other_poses_from_obs(obs, agent_idx),
        opp_pred_trajs=opp_pred_trajs,
    )
    # print(f"MPC plan time for agent {agent_idx}: {time.time() - t0:.3f}s")
    return traj


def mpc_based_multi_generator(args, agent_planners, obs, params_dict):
    
    mode = get_interaction_mode(args)
    ego_idx = 0
    num_agents = len(agent_planners)

    # get initial plan for all agents without opponent trajectory information as warm starts
    curr = [
        mpc_plan_once(agent_planners[i], obs, i, opp_pred_trajs=None)
        for i in range(num_agents)
    ]

    t0 = time.time()
    if mode == "nonreactive":
        curr = [
            mpc_plan_once(agent_planners[i], obs, i, collect_other_trajs(curr, i))
            for i in range(num_agents)
        ]
        return curr, time.time() - t0


    if mode == "opp-reactive":
        out = list(curr)
        for i in range(num_agents):
            if i != ego_idx:
                out[i] = mpc_plan_once(agent_planners[i], obs, i, collect_other_trajs(curr, i))
        infer_time = time.time() - t0
        return out, infer_time

    if mode != "ibr":
        raise ValueError(f"Unsupported MPC interaction mode: {mode!r}")

    num_ibr_iters = int(params_dict.get("num_ibr_iters"))
    # print(f"Running MPC-based iterative best response for {num_ibr_iters} iterations...")
    for _ in range(num_ibr_iters):
        prev = curr
        curr = [
            mpc_plan_once(agent_planners[i], obs, i, collect_other_trajs(prev, i))
            for i in range(num_agents)
        ]

    curr[ego_idx] = mpc_plan_once(
        agent_planners[ego_idx],
        obs,
        ego_idx,
        collect_other_trajs(curr, ego_idx),
    )

    infer_time = time.time() - t0
    # print(f"MPC-based IBR inference time: {infer_time:.3f}s for {num_agents} agents, {num_ibr_iters} IBR iterations.")
    return curr, infer_time
