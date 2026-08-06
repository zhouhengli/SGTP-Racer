import yaml
import copy
import numpy as np
import torch
import time

from typing import Dict, Optional

from players.planner.controller.pure_pursuit import PurePursuitPlanner
from .biased_mppi_core import BiasedMPPI
from players.utils.game_block_cost import GameBlockCost
from players.utils.common import nearest_point


class BiasedMPPIPlanner:
    """Base MPPI planner.

    This class implements standard MPPI, tracking cost, and game-block cost.
    Ancillary prior sampling is implemented in a separate subclass in
    biased_mppi_ancillary.py.
    """

    def __init__(
        self,
        args,
        conf,
        map_path,
        wpt_path,
        wb=None,
        v_scale=None,
        ocp_conf=None,
        game_block_conf=None,
        biased_type=None,
    ):
        self.args = args
        self.conf = conf
        self.map_path = map_path
        self.map_ext = '.png'
        self.wheelbase = wb

        waypoints = np.loadtxt(wpt_path, delimiter=';', skiprows=1)
        self.waypoints = np.vstack(
            (
                waypoints[:, 3], # x
                waypoints[:, 4], # y
                waypoints[:, 9], # v_ref
                waypoints[:, 7], # heading
                waypoints[:, 1], # s
            )
        ).T
        self.waypoints[:, 2] *= v_scale
        self.d_right_left = np.vstack((waypoints[:, 5], waypoints[:, 6]))

        self.state_t = 0.0
        self.best_traj = None
        self.prev_traj_local = None
        self.goal_grid = None
        self.tracker = PurePursuitPlanner(conf, wpt_path, wb=wb)

        if ocp_conf is None:
            with open(args.ocp_config, 'r') as f:
                ocp_conf = yaml.safe_load(f)
        else:
            ocp_conf = copy.deepcopy(ocp_conf)

        if game_block_conf is None:
            with open(args.game_config, 'r') as f:
                game_yaml = yaml.safe_load(f)
            game_block_conf = game_yaml['game_block_cost']
        else:
            game_block_conf = copy.deepcopy(game_block_conf)

        self.ocp_conf = copy.deepcopy(ocp_conf)
        self.game_block_conf = copy.deepcopy(game_block_conf)

        # Fixed scalar used only for numerical balancing against tracking/control costs.
        self.game_cost_weight = float(self.game_block_conf.get('game_cost_weight'))
        # print(f"Game cost weight: {self.game_cost_weight}")

        self.duration = ocp_conf.get('horizon')
        self.N = ocp_conf.get('N')
        self.dt = float(self.duration / self.N)
        self.L = wb
        self.v_max = ocp_conf.get('v_max')
        self.a_max = ocp_conf.get('a_max')
        self.delta_max = float(args.delta_max)
        self.num_samples = ocp_conf.get('num_samples')
        self.lambda_ = ocp_conf.get('lambda_')
        self.sigma_a = ocp_conf.get('sigma')[0]
        self.sigma_delta = ocp_conf.get('sigma')[1]
        # print(f"MPPI config: num_samples={self.num_samples}")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32

        self.mppi_bias_mode = str(biased_type or 'none').lower()
        self.keep_mean_sample = True


        self.game_block_cost = GameBlockCost(
            contest_s_gap=game_block_conf['contest_s_gap'],
            longitudinal_weight=game_block_conf['longitudinal_weight'],
            contest_weight=game_block_conf['contest_weight'],
            block_weight=game_block_conf['block_weight'],
            tail_ratio=game_block_conf['tail_ratio'],
            role_s_margin=game_block_conf['role_s_margin'],
            safety_weight=game_block_conf['safety_weight'],
            safe_dist=game_block_conf['safe_dist'],
        )
        cost_conf = ocp_conf.get('cost_weights')
        self.opp_cost_weight = cost_conf.get('opp_cost_weight')
        running_cost_conf = cost_conf.get('running')
        self.cost_running_track = float(running_cost_conf.get('track'))
        self.cost_running_heading = float(running_cost_conf.get('heading'))
        self.cost_running_speed = float(running_cost_conf.get('speed'))
        self.cost_running_u_accel = float(running_cost_conf.get('u_accel'))
        self.cost_running_u_steer = float(running_cost_conf.get('u_steer'))
        self.cost_running_du_accel = float(running_cost_conf.get('du_accel'))
        self.cost_running_du_steer = float(running_cost_conf.get('du_steer'))
        self.ref_traj = None
        self._build_extended_waypoints()
        self._build_mppi_core()

    def _effective_sampling_bias_mode(self, requested_mode: Optional[str] = None) -> str:
        """Base planner supports pure MPPI only."""
        # another _effective_sampling_bias_mode is overridden in AncillaryBiasedMPPIPlanner to support ancillary sampling; the base class always returns 'none' regardless of requested_mode or self.mppi_bias_mode.
        mode = str(requested_mode or self.mppi_bias_mode or 'none').lower()

        if mode != 'none':
            raise ValueError(
                "Base BiasedMPPIPlanner supports only bias_mode='none'. "
                "Use AncillaryBiasedMPPIPlanner for bias_mode='ancillary'."
            )

        return mode

    def _prior_sampling_config(self, mode: str):
        """Hook for subclasses that reserve MPPI samples for prior actions."""
        return None, 0, 0.0

    def set_parameters(self, parameters):
        if not isinstance(parameters, dict):
            return
        rebuild_needed = False
        rebuild_keys = {
            'N',
            'dt',
            'L',
            'v_max',
            'a_max',
            'delta_max',
            'num_samples',
            'lambda_',
            'sigma_a',
            'sigma_delta',
            'keep_mean_sample',
        }
        for key, value in parameters.items():
            setattr(self, key, value)
            if key in rebuild_keys:
                rebuild_needed = True
        if rebuild_needed:
            self._build_extended_waypoints()
            self._build_mppi_core()

    def _as_opp_pose_tensor_list(self, opp_poses):
        if opp_poses is None:
            return (None, None)
        if torch.is_tensor(opp_poses):
            pose_tensor = opp_poses.to(device=self.device, dtype=self.dtype)
        else:
            try:
                pose_tensor = torch.tensor(opp_poses, device=self.device, dtype=self.dtype)
            except Exception:
                return (None, None)
        if pose_tensor.numel() == 0:
            return (None, None)
        if pose_tensor.ndim == 1:
            if pose_tensor.shape[0] < 2:
                return (None, None)
            pose_tensor = pose_tensor.view(1, -1)
        if pose_tensor.ndim != 2 or pose_tensor.shape[1] < 2:
            return (None, None)
        pose_xy = pose_tensor[:, :2]
        finite_mask = torch.isfinite(pose_xy).all(dim=1)
        nonzero_mask = ~((pose_xy[:, 0] == 0.0) & (pose_xy[:, 1] == 0.0))
        valid_mask = finite_mask & nonzero_mask
        if torch.count_nonzero(valid_mask).item() == 0:
            return (None, None)
        pose_xy = pose_xy[valid_mask]
        opp_pose_torch = pose_xy[0]
        opp_poses_torch = pose_xy
        return (opp_pose_torch, opp_poses_torch)

    def _as_opp_traj_list(self, opp_pred_traj):
        """Return only valid opponent prediction trajectories.
            valid opp_pred_traj present -> trajectory-level game cost is enabled;
            no valid opp_pred_traj      -> no game cost, use current-pose obstacle cost.
        """
        def _valid_one(traj):
            if traj is None:
                return None

            if torch.is_tensor(traj):
                if traj.ndim != 2 or traj.shape[0] <= 0 or traj.shape[1] < 2:
                    return None
                xy = traj[:, :2]
                if not bool(torch.isfinite(xy).all().item()):
                    return None
                if not bool(torch.any((xy[:, 0] != 0.0) | (xy[:, 1] != 0.0)).item()):
                    return None
                return traj

            try:
                arr = np.asarray(traj, dtype=float)
            except Exception:
                return None

            if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] < 2:
                return None
            xy = arr[:, :2]
            if not np.isfinite(xy).all():
                return None
            if not np.any((xy[:, 0] != 0.0) | (xy[:, 1] != 0.0)):
                return None
            return arr

        if opp_pred_traj is None:
            return []

        if torch.is_tensor(opp_pred_traj):
            if opp_pred_traj.ndim == 2:
                out = _valid_one(opp_pred_traj)
                return [] if out is None else [out]
            if opp_pred_traj.ndim == 3:
                return [t for t in (_valid_one(opp_pred_traj[i]) for i in range(opp_pred_traj.shape[0])) if t is not None]
            return []

        try:
            arr = np.asarray(opp_pred_traj, dtype=float)
            if arr.ndim == 2:
                out = _valid_one(arr)
                return [] if out is None else [out]
            if arr.ndim == 3:
                return [t for t in (_valid_one(arr[i]) for i in range(arr.shape[0])) if t is not None]
        except Exception:
            pass

        if isinstance(opp_pred_traj, (list, tuple)):
            return [t for t in (_valid_one(traj) for traj in opp_pred_traj) if t is not None]

        return []

    def _state_seq_to_traj_heading(self, state_seq_np: np.ndarray):
        traj = np.zeros((state_seq_np.shape[0], 3), dtype=float)
        traj[:, 0] = state_seq_np[:, 0]
        traj[:, 1] = state_seq_np[:, 1]
        traj[:, 2] = state_seq_np[:, 3]
        heading = state_seq_np[:, 2]
        return (traj, heading)

    def _cost_func_torch(self, state, action, info):
        # Computes tracking and control-effort costs. When no valid opponent
        # prediction is available, it also adds current-pose obstacle costs.
        # the game cost is computed in _trajectory_game_cost_torch
        ref_traj = info['ref_traj']
        t = int(info['t'])
        is_terminal = bool(info.get('is_terminal_step', torch.count_nonzero(action).item() == 0))
        ref_idx = min(t + 1, self.N) if is_terminal else min(t, self.N)
        ref = ref_traj[:, ref_idx]
        rx, ry, rpsi, rv = (ref[0], ref[1], ref[2], ref[3])
        px = state[:, 0]
        py = state[:, 1]
        psi = state[:, 2]
        v = state[:, 3]
        ex = px - rx
        ey = py - ry
        epsi = torch.atan2(torch.sin(psi - rpsi), torch.cos(psi - rpsi))
        ev = v - rv
        cost_track = self.cost_running_track * (ex ** 2 + ey ** 2)
        cost_heading = self.cost_running_heading * epsi ** 2
        cost_speed = self.cost_running_speed * ev ** 2
        cost = cost_track + cost_heading + cost_speed
        if not is_terminal:
            prev_action = info['prev_action']
            a = action[:, 0]
            delta = action[:, 1]
            da = action[:, 0] - prev_action[:, 0]
            ddelta = action[:, 1] - prev_action[:, 1]
            cost_u = self.cost_running_u_accel * a ** 2 + self.cost_running_u_steer * delta ** 2
            cost_du = self.cost_running_du_accel * da ** 2 + self.cost_running_du_steer * ddelta ** 2
            cost = cost + cost_u + cost_du
        if len(self._as_opp_traj_list(info.get('opp_pred_traj', None))) > 0:
            return cost

        def _ellipse_opponent_cost(px, py, ox, oy):
            dx = px - ox
            dy = py - oy
            a_long = 2.0
            a_lat = 1.6
            ellipse_value = (dx / a_long) ** 2 + (dy / a_lat) ** 2
            return self.opp_cost_weight * (torch.relu(1.0 - ellipse_value) * 7.5) ** 2
        opp_poses = info.get('opp_poses', None)
        if opp_poses is not None:
            if torch.is_tensor(opp_poses):
                opp_pose_tensor = opp_poses.to(device=self.device, dtype=self.dtype)
            else:
                opp_pose_tensor = torch.tensor(opp_poses, device=self.device, dtype=self.dtype)
            if opp_pose_tensor.ndim == 1:
                opp_pose_tensor = opp_pose_tensor.view(1, -1)
            if opp_pose_tensor.ndim == 2 and opp_pose_tensor.shape[1] >= 2:
                for k in range(opp_pose_tensor.shape[0]):
                    ox = opp_pose_tensor[k, 0]
                    oy = opp_pose_tensor[k, 1]
                    if not (ox == 0.0 and oy == 0.0):
                        cost = cost + _ellipse_opponent_cost(px, py, ox, oy)
                return cost
        opp_pose = info.get('opp_pose', None)
        if opp_pose is not None:
            ox, oy = (opp_pose[0], opp_pose[1])
            if not (ox == 0.0 and oy == 0.0):
                cost = cost + _ellipse_opponent_cost(px, py, ox, oy)
        return cost

    def _build_extended_waypoints(self):
        s_total = self.waypoints[-1, 4]
        wp_extended = np.vstack([self.waypoints, self.waypoints.copy()])
        wp_extended[len(self.waypoints):, 4] += s_total
        self.wp_s = wp_extended[:, 4]
        self.wp_x = wp_extended[:, 0]
        self.wp_y = wp_extended[:, 1]
        self.wp_psi = wp_extended[:, 3]
        self.wp_v = wp_extended[:, 2]
        self.s_total = s_total

    def _build_mppi_core(self):
        """Build MPPI core.

        The base class passes no prior sampler.  Subclasses can override
        _prior_sampling_config() to reserve prior sample slots.
        """
        effective_bias_mode = self._effective_sampling_bias_mode()
        prior_sampler, num_prior_samples, prior_noise_scale = (
            self._prior_sampling_config(effective_bias_mode)
        )

        self._u_min_torch = torch.tensor(
            [-self.a_max, -self.delta_max],
            device=self.device,
            dtype=self.dtype,
        )
        self._u_max_torch = torch.tensor(
            [self.a_max, self.delta_max],
            device=self.device,
            dtype=self.dtype,
        )
        sigmas = torch.tensor(
            [self.sigma_a, self.sigma_delta],
            device=self.device,
            dtype=self.dtype,
        )

        self.mppi = BiasedMPPI(
            horizon=self.N,
            num_samples=self.num_samples,
            dim_state=4, # x, y, psi, v
            dim_control=2, # not for underlying control which is done by pp controller
            dynamics=self._dynamics_torch,
            cost_func=self._cost_func_torch,
            u_min=self._u_min_torch,
            u_max=self._u_max_torch,
            sigmas=sigmas,
            lambda_=float(self.lambda_),
            device=self.device,
            dtype=self.dtype,
            sampling_bias_mode=effective_bias_mode,
            prior_sampler=prior_sampler,
            num_prior_samples=num_prior_samples,
            prior_noise_scale=prior_noise_scale,
            keep_mean_sample=self.keep_mean_sample,
            trajectory_cost_func=self._trajectory_game_cost_torch,
        )

    def _dynamics_torch(self, state, action):
        # value from f1tenth_gym/gym/f110_gym/envs/f110_env.py
        # models are from the paper: The Kinematic Bicycle Model: a Consistent Model for Planning  Feasible Trajectories for Autonomous Vehicles?
        lf = 0.162
        lr = 0.145
        px = state[:, 0]
        py = state[:, 1]
        psi = state[:, 2]
        v = state[:, 3]
        a = action[:, 0]
        delta = torch.clamp(action[:, 1], -self.delta_max, self.delta_max)
        v = torch.clamp(v, 0.0, self.v_max)
        beta = torch.atan((lr / (lf + lr)) * torch.tan(delta))
        px_next = px + v * torch.cos(psi + beta) * self.dt
        py_next = py + v * torch.sin(psi + beta) * self.dt
        psi_next = psi + (v / lr) * torch.tan(beta) * torch.cos(beta) * self.dt
        psi_next = torch.atan2(torch.sin(psi_next), torch.cos(psi_next))
        v_next = torch.clamp(v + a * self.dt, 0.0, self.v_max)

        return torch.stack([px_next, py_next, psi_next, v_next], dim=1)

    def build_reference(self, x, y):
        # here is the reference trajectory for tracking cost, which is built from waypoints and current ego position
        _, _, _, idx = nearest_point(np.array([x, y]), self.waypoints[:, :2])
        s_curr = self.waypoints[idx, 4]
        ref_traj = np.zeros((4, self.N + 1), dtype=float)
        v_ref_base = self.wp_v[idx]
        for k in range(self.N + 1):
            s_target = s_curr + v_ref_base * (k * self.dt)
            ref_traj[0, k] = np.interp(s_target, self.wp_s, self.wp_x)
            ref_traj[1, k] = np.interp(s_target, self.wp_s, self.wp_y)
            ref_traj[2, k] = np.interp(s_target, self.wp_s, self.wp_psi)
            ref_traj[3, k] = np.interp(s_target, self.wp_s, self.wp_v)
        self.ref_traj = ref_traj
        return ref_traj

    def _trajectory_game_cost_torch(
        self,
        state_seq_batch: torch.Tensor,
        action_seq_batch: torch.Tensor,
        info: Dict,
    ) -> Optional[torch.Tensor]:
        """Prediction-based trajectory game cost.

        The switch is intentionally minimal:
            - no valid opp_pred_traj -> no GameBlockCost
            - valid opp_pred_traj    -> GameBlockCost enabled

        The raw GameBlockCost is multiplied by self.game_cost_weight for
        numerical balancing with tracking/control costs.
        """
        opp_pred_trajs = self._as_opp_traj_list(info.get('opp_pred_traj', None))
        if len(opp_pred_trajs) == 0:
            return None

        ego_xyv = torch.stack(
            [
                state_seq_batch[:, :, 0],
                state_seq_batch[:, :, 1],
                state_seq_batch[:, :, 3],
            ],
            dim=-1,
        ).detach().cpu().numpy().astype(np.float64)

        total_game_cost = np.zeros(ego_xyv.shape[0], dtype=np.float64)
        waypoints_np = np.asarray(self.waypoints, dtype=np.float64)

        for one_opp_pred_traj in opp_pred_trajs:
            if torch.is_tensor(one_opp_pred_traj):
                opp_np = one_opp_pred_traj.detach().cpu().numpy().astype(np.float64)
            else:
                opp_np = np.asarray(one_opp_pred_traj, dtype=np.float64)

            if opp_np.shape[1] < 3:
                opp_xyv = np.zeros((opp_np.shape[0], 3), dtype=np.float64)
                opp_xyv[:, :2] = opp_np[:, :2]
            else:
                opp_xyv = opp_np[:, :3]

            horizon_len = min(ego_xyv.shape[1], opp_xyv.shape[0])
            if horizon_len <= 0:
                continue

            game_cost_i = self.game_block_cost(
                ego_xyv[:, :horizon_len, :],
                opp_xyv[:horizon_len, :],
                waypoints_np,
            ).reshape(-1)
            total_game_cost += game_cost_i

        total_game_cost *= self.game_cost_weight
        return torch.tensor(total_game_cost, device=self.device, dtype=self.dtype)

    def plan(
        self,
        pose_x,
        pose_y,
        pose_theta,
        opp_poses,
        velocity,
        opp_pred_traj=None,
        bias_mode: Optional[str] = None,
    ):
        """Run one MPPI planning call without hard post-selection, which is called by 'players/planner/planner_generators.py'

        Passing a valid opp_pred_traj is the only switch that enables trajectory
        game cost.  Passing None keeps the planner in current-pose-based obstacle avoidance.
        """
        x0 = torch.tensor(
            [pose_x, pose_y, pose_theta, velocity],
            device=self.device,
            dtype=self.dtype,
        )

        runtime_bias_mode = self._effective_sampling_bias_mode(bias_mode)


        ref_np = self.build_reference(pose_x, pose_y)
        ref_torch = torch.tensor(ref_np, device=self.device, dtype=self.dtype)
        opp_pose_torch, opp_poses_torch = self._as_opp_pose_tensor_list(opp_poses)

        info = {
            'ref_traj': ref_torch,
            'opp_pose': opp_pose_torch,
            'opp_poses': opp_poses_torch,
            'opp_pred_traj': opp_pred_traj,
            'sampling_bias_mode': runtime_bias_mode,
        }


        # start_time = time.time()
        with torch.no_grad():
            _, x_opt = self.mppi(x0, info)
        # print(f"MPPI optimization time: {time.time() - start_time:.3f}s")

        x_opt_np = x_opt[0].detach().cpu().numpy()
        traj, heading = self._state_seq_to_traj_heading(x_opt_np)
        self.best_traj = traj
        return traj, heading

__all__ = ['BiasedMPPIPlanner']
