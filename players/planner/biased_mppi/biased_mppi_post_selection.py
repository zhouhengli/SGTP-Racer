import numpy as np
import torch
import time

from numba import njit
from typing import Optional

from .biased_mppi_planner import BiasedMPPIPlanner

@njit(cache=True)
def _boundary_validity_numba(
    state_seq_batch,
    waypoint_xy,
    waypoint_psi,
    d_right_left,
    half_width,
    safety_buffer,
    boundary_margin,
):
    sample_count = state_seq_batch.shape[0]
    horizon_len = state_seq_batch.shape[1]
    waypoint_count = waypoint_xy.shape[0]

    valid_boundary = np.ones(sample_count, dtype=np.bool_)
    min_boundary_clearance = np.empty(sample_count, dtype=state_seq_batch.dtype)

    for k in range(sample_count):
        min_clearance_k = 1.0e9

        for t in range(horizon_len):
            px = state_seq_batch[k, t, 0]
            py = state_seq_batch[k, t, 1]

            if not np.isfinite(px) or not np.isfinite(py):
                min_clearance_k = -1.0e9
                continue

            best_idx = 0
            best_dist2 = 1.0e18

            for j in range(waypoint_count):
                dx_wp = px - waypoint_xy[j, 0]
                dy_wp = py - waypoint_xy[j, 1]
                dist2 = dx_wp * dx_wp + dy_wp * dy_wp

                if dist2 < best_dist2:
                    best_dist2 = dist2
                    best_idx = j

            wx = waypoint_xy[best_idx, 0]
            wy = waypoint_xy[best_idx, 1]
            wpsi = waypoint_psi[best_idx]

            dx = px - wx
            dy = py - wy

            cos_psi = np.cos(wpsi)
            sin_psi = np.sin(wpsi)

            lateral_d = -dx * sin_psi + dy * cos_psi

            width_right = d_right_left[0, best_idx]
            width_left = d_right_left[1, best_idx]

            clear_right = width_right + lateral_d
            clear_left = width_left - lateral_d

            clear_right_eff = clear_right - half_width - safety_buffer
            clear_left_eff = clear_left - half_width - safety_buffer

            if clear_left_eff < clear_right_eff:
                clear_eff = clear_left_eff
            else:
                clear_eff = clear_right_eff

            if clear_eff < min_clearance_k:
                min_clearance_k = clear_eff

        min_boundary_clearance[k] = min_clearance_k
        valid_boundary[k] = min_clearance_k >= boundary_margin

    return valid_boundary, min_boundary_clearance


@njit(cache=True)
def _collision_validity_one_opp_numba(
    state_seq_batch,
    opp_traj,
    collision_dist,
    valid_collision,
    min_obs_dist,
):
    sample_count = state_seq_batch.shape[0]
    horizon_len = state_seq_batch.shape[1]

    compare_len = horizon_len
    if opp_traj.shape[0] < compare_len:
        compare_len = opp_traj.shape[0]

    for k in range(sample_count):
        min_dist2_k = 1.0e18

        for t in range(compare_len):
            dx = state_seq_batch[k, t, 0] - opp_traj[t, 0]
            dy = state_seq_batch[k, t, 1] - opp_traj[t, 1]

            dist2 = dx * dx + dy * dy

            if dist2 < min_dist2_k:
                min_dist2_k = dist2

        min_dist_k = np.sqrt(min_dist2_k)

        if min_dist_k < min_obs_dist[k]:
            min_obs_dist[k] = min_dist_k

        if min_dist_k < collision_dist:
            valid_collision[k] = False

@njit(cache=True)
def _select_index_numba(
    costs,
    valid_boundary,
    valid_collision,
    min_boundary_clearance,
    min_obs_dist,
    collision_dist,
    boundary_margin,
):
    sample_count = costs.shape[0]

    best_valid_idx = -1
    best_valid_cost = 1.0e18

    for k in range(sample_count):
        if valid_boundary[k] and valid_collision[k] and np.isfinite(costs[k]):
            if costs[k] < best_valid_cost:
                best_valid_cost = costs[k]
                best_valid_idx = k

    if best_valid_idx >= 0:
        return best_valid_idx

    best_idx = 0
    best_violation = 1.0e18

    for k in range(sample_count):
        collision_violation = collision_dist - min_obs_dist[k]
        if not np.isfinite(collision_violation) or collision_violation < 0.0:
            collision_violation = 0.0
        

        boundary_violation = boundary_margin - min_boundary_clearance[k]
        if not np.isfinite(boundary_violation) or boundary_violation < 0.0:
            boundary_violation = 0.0
        # if boundary_violation > 0.0:
        #     print(f"Collision violation for sample {k}: {collision_violation}")
        #     print(f"Boundary violation for sample {k}: {float(boundary_violation):.3f}")

        safety_violation = collision_violation + boundary_violation

        if safety_violation < best_violation:
            best_violation = safety_violation
            best_idx = k

    return best_idx


class BiasedMPPIPostSelectionPlanner(BiasedMPPIPlanner):
    COLLISION_DIST = 0.9
    BOUNDARY_MARGIN = 0.06
    BOUNDARY_SAFETY_BUFFER = 0.30

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.dtype == torch.float64:
            self._np_dtype = np.float64
        else:
            self._np_dtype = np.float32

        self._init_boundary_arrays()

    # ------------------------------------------------------------------
    # Boundary utilities
    # ------------------------------------------------------------------

    def _init_boundary_arrays(self):
        self._waypoints_xy_np = np.ascontiguousarray(
            self.waypoints[:, :2],
            dtype=self._np_dtype,
        )

        self._waypoints_psi_np = np.ascontiguousarray(
            self.waypoints[:, 3],
            dtype=self._np_dtype,
        )

        self._d_right_left_np = np.ascontiguousarray(
            self.d_right_left,
            dtype=self._np_dtype,
        )

    def _compute_boundary_validity(self, state_seq_batch_np):
        half_width = 0.5 * float(self.args.width)

        return _boundary_validity_numba(
            state_seq_batch_np,
            self._waypoints_xy_np,
            self._waypoints_psi_np,
            self._d_right_left_np,
            half_width,
            self.BOUNDARY_SAFETY_BUFFER,
            self.BOUNDARY_MARGIN,
        )

    # ------------------------------------------------------------------
    # Collision utilities
    # ------------------------------------------------------------------

    def _to_numpy(self, value):
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()

        return np.ascontiguousarray(
            value,
            dtype=self._np_dtype,
        )

    def _opponent_traj_list(self, opp_pred_traj):
        if opp_pred_traj is None:
            return []

        if isinstance(opp_pred_traj, list):
            return opp_pred_traj

        if isinstance(opp_pred_traj, tuple):
            return list(opp_pred_traj)

        if torch.is_tensor(opp_pred_traj):
            if opp_pred_traj.ndim == 3:
                return [opp_pred_traj[i] for i in range(opp_pred_traj.shape[0])]
            return [opp_pred_traj]

        opp_arr = np.asarray(opp_pred_traj)

        if opp_arr.ndim == 3:
            return [opp_arr[i] for i in range(opp_arr.shape[0])]

        return [opp_pred_traj]

    def _compute_collision_validity(self, state_seq_batch_np, opp_pred_traj):
        sample_count = int(state_seq_batch_np.shape[0])

        valid_collision = np.ones(sample_count, dtype=np.bool_)

        min_obs_dist = np.full(
            sample_count,
            np.inf,
            dtype=self._np_dtype,
        )

        opp_trajs = self._opponent_traj_list(opp_pred_traj)

        for one_opp_traj in opp_trajs:
            opp_np = self._to_numpy(one_opp_traj)

            _collision_validity_one_opp_numba(
                state_seq_batch_np,
                opp_np,
                self.COLLISION_DIST,
                valid_collision,
                min_obs_dist,
            )

        return valid_collision, min_obs_dist

    # ------------------------------------------------------------------
    # MPPI batch access and commit
    # ------------------------------------------------------------------

    def _get_mppi_candidate_batches_cpu(self):
        state_seq_batch = self.mppi._state_seq_batch.detach().cpu().numpy()
        action_seq_batch = self.mppi._perturbed_action_seqs.detach().cpu().numpy()
        costs = self.mppi._last_costs.detach().view(-1).cpu().numpy()
        # print(f"MPPI candidate batch shapes - state_seq_batch: {state_seq_batch.shape}, action_seq_batch: {action_seq_batch.shape}, costs: {costs.shape}")


        state_seq_batch = np.ascontiguousarray(
            state_seq_batch,
            dtype=self._np_dtype,
        )

        action_seq_batch = np.ascontiguousarray(
            action_seq_batch,
            dtype=self._np_dtype,
        )

        costs = np.ascontiguousarray(
            costs,
            dtype=self._np_dtype,
        )

        return state_seq_batch, action_seq_batch, costs

    def _commit_selected_action_sequence(
        self,
        selected_action_seq_np,
        selected_state_seq_np=None,
    ):
        selected_action_seq = torch.as_tensor(
            selected_action_seq_np,
            device=self.mppi._previous_action_seq.device,
            dtype=self.mppi._previous_action_seq.dtype,
        )

        self.mppi._previous_action_seq = selected_action_seq.detach().clone()

        if (
            hasattr(self.mppi, "_actions_history_for_sg")
            and self.mppi._actions_history_for_sg is not None
            and self.mppi._actions_history_for_sg.shape[0] > 0
        ):
            self.mppi._actions_history_for_sg[-1, :] = selected_action_seq[0].detach()

        if selected_state_seq_np is not None and hasattr(self.mppi, "_optimal_state_seq"):
            selected_state_seq = torch.as_tensor(
                selected_state_seq_np,
                device=self.device,
                dtype=self.dtype,
            )

            self.mppi._optimal_state_seq = selected_state_seq.detach().clone()

    # ------------------------------------------------------------------
    # Post-selection core
    # ------------------------------------------------------------------

    def _select_candidate_index(
        self,
        costs_np,
        valid_boundary_np,
        valid_collision_np,
        min_boundary_clearance_np,
        min_obs_dist_np,
    ):
        return int(
            _select_index_numba(
                costs_np,
                valid_boundary_np,
                valid_collision_np,
                min_boundary_clearance_np,
                min_obs_dist_np,
                self.COLLISION_DIST,
                self.BOUNDARY_MARGIN,
            )
        )

    def _post_select_mppi_output(self, opp_pred_traj):
        state_seq_batch_np, action_seq_batch_np, costs_np = (
            self._get_mppi_candidate_batches_cpu()
        )

        # boundary feasibility checks are performed on the CPU for efficiency
        valid_boundary_np, min_boundary_clearance_np = self._compute_boundary_validity(
            state_seq_batch_np,
        )

        # inner-vehicle collision feasibility checks are performed on the CPU for efficiency
        valid_collision_np, min_obs_dist_np = self._compute_collision_validity(
            state_seq_batch_np,
            opp_pred_traj,
        )

        # choose the most feasible and lowest-cost candidate from the MPPI batch, with a fallback to the least-violating candidate if no feasible candidates exist
        selected_idx = self._select_candidate_index(
            costs_np=costs_np,
            valid_boundary_np=valid_boundary_np,
            valid_collision_np=valid_collision_np,
            min_boundary_clearance_np=min_boundary_clearance_np,
            min_obs_dist_np=min_obs_dist_np,
        )

        selected_state_seq_np = np.ascontiguousarray(
            state_seq_batch_np[selected_idx].copy(),
            dtype=self._np_dtype,
        )

        selected_action_seq_np = np.ascontiguousarray(
            action_seq_batch_np[selected_idx].copy(),
            dtype=self._np_dtype,
        )

        # commit the selected state and control sequence to the planner's internal state for warm starting the next planning iteration
        self._commit_selected_action_sequence(
            selected_action_seq_np=selected_action_seq_np,
            selected_state_seq_np=selected_state_seq_np,
        )

        return selected_state_seq_np, selected_action_seq_np

    # ------------------------------------------------------------------
    # Public planner interface
    # ------------------------------------------------------------------

    def plan(
        self,
        pose_x,
        pose_y,
        pose_theta,
        opp_poses,
        velocity,
        opp_pred_traj=None,
        post_select: Optional[bool] = None,
        bias_mode: Optional[str] = None,
    ):
        if post_select is None:
            raise ValueError("post_select flag must be explicitly provided to plan() in BiasedMPPIPostSelectionPlanner.")

        traj, heading = super().plan(
            pose_x=pose_x,
            pose_y=pose_y,
            pose_theta=pose_theta,
            opp_poses=opp_poses,
            velocity=velocity,
            opp_pred_traj=opp_pred_traj,
            bias_mode=bias_mode,
        )

        # use normal MPPI output
        if not post_select:
            return traj, heading

        # if opp_pred_traj is None:
        #     print(f"MPPI post-selection with opponent prediction of shape {opp_pred_traj.shape if torch.is_tensor(opp_pred_traj) else 'N/A'}")


        # enforce feasibility selection for smooth and safe transitions between diverse racing behaviors
        # commit the selected state and control sequence to the planner's internal state for warm starting the next planning iteration
        selected_state_seq_np, _ = self._post_select_mppi_output(opp_pred_traj=opp_pred_traj)
        traj, heading = self._state_seq_to_traj_heading(selected_state_seq_np)
        self.best_traj = traj
        return traj, heading


__all__ = ["BiasedMPPIPostSelectionPlanner"]