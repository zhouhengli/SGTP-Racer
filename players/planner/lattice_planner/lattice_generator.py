import numpy as np
import time

def get_lattice_interaction_mode(args):
    mode = getattr(args, "interaction_mode", None)
    if mode is None:
        return "nonreactive"
    return str(mode).lower()


def collect_opp_poses(obs, agent_idx):
    poses = []
    num_agents = len(obs["poses_x"])
    for j in range(num_agents):
        if j == agent_idx:
            continue
        vx = float(obs["linear_vels_x"][j]) if "linear_vels_x" in obs else 0.0
        vy = float(obs["linear_vels_y"][j]) if "linear_vels_y" in obs else 0.0
        poses.append([
            float(obs["poses_x"][j]),
            float(obs["poses_y"][j]),
            float(obs["poses_theta"][j]),
            vx,
            vy,
        ])
    return np.asarray(poses, dtype=np.float64)


def lattice_plan_once(planner, obs, agent_idx):
    all_traj, all_clothoid, ego_pose, ref_col_idx = planner.generate_candidates(
        float(obs["poses_x"][agent_idx]),
        float(obs["poses_y"][agent_idx]),
        float(obs["poses_theta"][agent_idx]),
        float(obs["linear_vels_x"][agent_idx]),
    )
    opp_poses = collect_opp_poses(obs, agent_idx)
    return planner.select_best_response(
        all_traj,
        all_clothoid,
        ego_pose,
        opp_poses,
        ref_col_idx,
    )


def lattice_multi_generator(args, agent_planners, obs, params_dict):
    mode = get_lattice_interaction_mode(args)
    if mode != "nonreactive":
        raise ValueError("This generator only implements interaction_mode='nonreactive'.")
    t0 = time.time()
    curr = [
        lattice_plan_once(agent_planners[i], obs, i)
        for i in range(len(agent_planners))
    ]
    infer_time = time.time() - t0
    return curr, infer_time

