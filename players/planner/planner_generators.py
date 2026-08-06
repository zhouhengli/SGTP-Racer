import time
from typing import Dict, List
from players.planner.flow_planner.flow_trajectory_generator import flow_based_generator

import numpy as np


MPPI_POST_SELECT = True  # Whether to enable feasibility selection on ego trajectories.
# MPPI_POST_SELECT = False  # Whether to enable feasibility selection on ego trajectories.

USE_OPP_PRED = True  # Whether to pass predicted opponent trajectories to the planner.
# USE_OPP_PRED = False  # Disable opponent-trajectory predictions; _other_trajs returns None.

# ---------------------------------------------------------------------------
# Inlined generator utilities
# ---------------------------------------------------------------------------

def resolve_current_method_mode(method: str) -> str:
    """Resolve method suffix into the interaction mode used by the generator."""
    method = str(method)
    if method.endswith("_ibr"):
        return "ibr"
    if method.endswith("_opp-reactive"):
        return "opp-reactive"
    if method.endswith("_nonreactive"):
        return "nonreactive"
    raise ValueError(
        f"Could not resolve interaction mode from method name '{method}'. "
        "Please specify --interaction_mode explicitly or use method names with "
        "suffixes like '_ibr' or '_opp-reactive'."
    )

def get_interaction_mode(args) -> str:
    """Return nonreactive, opp-reactive, or ibr."""
    mode = args.interaction_mode
    return resolve_current_method_mode(args.method) if mode is None else str(mode).lower()

def collect_other_poses_from_obs(obs: Dict, agent_idx: int) -> np.ndarray:
    """Return current poses of all non-controlled agents as [M, 3]."""

    def num_agents_from_obs(obs: Dict) -> int:
        return len(obs["poses_x"])

    return np.asarray(
        [
            [
                float(obs["poses_x"][j]),
                float(obs["poses_y"][j]),
                float(obs["poses_theta"][j]),
            ]
            for j in range(num_agents_from_obs(obs))
            if j != agent_idx
        ],
        dtype=float,
    )


def collect_other_trajs(trajs: List[np.ndarray], agent_idx: int) -> List[np.ndarray]:
    """Return predicted trajectories of all agents except agent_idx."""
    return [
        trajs[j]
        for j in range(len(trajs))
        if j != agent_idx and trajs[j] is not None
    ]


# Short local helpers used below; these keep the IBR loop visually compact.
_other_poses = collect_other_poses_from_obs


def _other_trajs(trajs: List[np.ndarray], agent_idx: int):
    """Return opponent predictions, or None when opponent prediction is disabled."""
    if not USE_OPP_PRED:
        return None
    return collect_other_trajs(trajs, agent_idx)


def ocp_based_multi_generator(args, agent_planners, obs, params_dict):
    """Multi-agent MPPI generator with compact synchronous IBR.

    The loop is intentionally explicit:
        selfish prediction -> L synchronous best-response updates.
    """
    mode = get_interaction_mode(args)
    ego_idx = 0

    def get_agent_bias_mode(agent_idx):
        if agent_idx == ego_idx:
            return str(args.mppi_bias_mode_ego).lower()
        return str(args.mppi_bias_mode_opp).lower()

    def plan_once(agent_idx, opp_pred_traj, post_select, round_name, bias_override):
        del round_name  # Kept in the signature for readable call sites.

        if bias_override is not None:
            bias_mode = str(bias_override).lower()
        else:
            bias_mode = get_agent_bias_mode(agent_idx)

        if bias_mode not in {"none", "ancillary"}:
            raise ValueError(
                "Unsupported MPPI bias_mode after cleanup: "
                f"{bias_mode!r}. Use 'none' or 'ancillary'."
            )

        planner = agent_planners[agent_idx]
        return planner.plan(
            float(obs["poses_x"][agent_idx]),
            float(obs["poses_y"][agent_idx]),
            float(obs["poses_theta"][agent_idx]),
            _other_poses(obs, agent_idx),
            float(obs["linear_vels_x"][agent_idx]),
            opp_pred_traj=opp_pred_traj,
            post_select=post_select,
            bias_mode=bias_mode,
        )

    initial_plan_result_by_agent_idx = [None for _ in range(len(agent_planners))]

    # First build unbiased/selfish predictions for every agent. These trajectories
    # are then used by the interaction modes below as opponent predictions.
    for agent_idx in range(len(agent_planners)):
        initial_plan_result_by_agent_idx[agent_idx] = plan_once(
            agent_idx=agent_idx,
            opp_pred_traj=None, # No opponent predictions for the initial round, to get truly selfish trajectories regardless of the MPPI bias settings for each agent.
            post_select=MPPI_POST_SELECT,
            round_name="initial_none",
            bias_override="none", # Explicitly override bias_mode to "none" for the initial prediction round, to get truly unbiased/selfish trajectories regardless of the MPPI bias settings for each agent.
        )

    curr = [
        initial_plan_result_by_agent_idx[i][0]
        for i in range(len(agent_planners))
    ]

    t0 = time.time()
    # nonreactive: compute selfish trajectories, then run one
    # safety-aware replanning pass against those fixed predictions.
    if mode == "nonreactive":
        out = list(curr)
        for i in range(len(agent_planners)):
            out[i] = plan_once(
                agent_idx=i,
                opp_pred_traj=_other_trajs(curr, i), # Use the initial selfish predictions as opponent predictions.
                post_select=MPPI_POST_SELECT,
                round_name="nonreactive",
                bias_override=None, # Use the default bias mode, which can be configured separately for ego and opponents.
            )[0]
        infer_time = time.time() - t0
        return out, infer_time

    if mode == "opp-reactive":
        out = list(curr)
        for i in range(len(agent_planners)):
            if i != ego_idx:
                out[i] = plan_once(
                    agent_idx=i,
                    opp_pred_traj=_other_trajs(curr, i),
                    post_select=MPPI_POST_SELECT,
                    round_name="opp-reactive",
                    bias_override=None, # Use the default bias mode for the opponent agents, which can be configured separately for ego and opponents.
                )[0]
        infer_time = time.time() - t0
        return out, infer_time

    if mode != "ibr":
        raise ValueError(f"Unsupported interaction mode: {mode!r}")

    # IBR: every agent responds to the same clean trajectory bank
    # from the previous round.
    for ibr_iter in range(int(params_dict["num_ibr_iters"])):
        prev = curr
        curr = [
            plan_once(
                agent_idx=i,
                opp_pred_traj=_other_trajs(prev, i),
                post_select=MPPI_POST_SELECT,
                round_name=f"ibr_{ibr_iter}",
                bias_override=None, # Use the default bias mode for all agents in the IBR rounds, which can be configured separately for ego and opponents.
            )[0]
            for i in range(len(agent_planners))
        ]

    curr[ego_idx] = plan_once(
        agent_idx=ego_idx,
        opp_pred_traj=_other_trajs(curr, ego_idx),
        post_select=MPPI_POST_SELECT,
        round_name="final_ego_response",
        bias_override=None, # Use the default bias mode for the ego agent in the final response, which can be configured separately for ego and opponents.
    )[0]

    infer_time = time.time() - t0
    # print(f"IBR inference time: {infer_time:.3f}s for {len(agent_planners)} agents, {params_dict['num_ibr_iters']} IBR iterations.")
    return curr, infer_time
