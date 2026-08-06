import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from players.planner.controller.pure_pursuit import PurePursuitPlanner
from players.utils.common import get_map_paths, load_config

# ref: https://github.com/ForzaETH/race_stack/tree/main/planner/spliner
# =============================================================================
# Spliner parameters
# =============================================================================

SPLINER_STATE_GBFREE = "GBFree"
SPLINER_STATE_TRAILING = "Trailing"
SPLINER_STATE_OVERTAKE = "Overtake"

SPLINER_LOOKAHEAD_M = 10.0
SPLINER_TRAJ_LENGTH_M = 12.0
SPLINER_RESOLUTION_M = 0.10

SPLINER_PRE_APEX_M = np.array([-4.0, -3.0, -1.5], dtype=float)
SPLINER_POST_APEX_M = np.array([2.0, 3.0, 4.0], dtype=float)
SPLINER_APEX_OFFSETS_M = np.array(
    [-4.0, -3.0, -1.5, 0.0, 2.0, 3.0, 4.0],
    dtype=float,
)

SPLINER_EVASION_DIST_M = 0.65
SPLINER_OBS_TRAJ_THRESH_M = 0.30
SPLINER_BOUND_MINDIST_M = 0.20
SPLINER_OPP_WIDTH_M = 0.31
SPLINER_OVERTAKE_SPEED_SCALE = 0.90

SPLINER_TRAILING_TARGET_GAP_M = 2.0
SPLINER_TRAILING_KP = 1.0
SPLINER_TRAILING_KD = 0.2
SPLINER_TRAILING_MIN_SPEED_MPS = 1.5

SPLINER_NUM_CURVATURE_CHECK_POINTS = 20
SPLINER_VELOCITY_LENGTH_SCALER_LIMIT = 1.5


@dataclass
class SplinerObstacle:
    agent_idx: int
    s_center: float
    d_center: float
    gap_s: float
    speed: float

    @property
    def d_left(self) -> float:
        return self.d_center + 0.5 * SPLINER_OPP_WIDTH_M

    @property
    def d_right(self) -> float:
        return self.d_center - 0.5 * SPLINER_OPP_WIDTH_M


class SplinerPlanner:
    """Finite-state spliner local planner.

    The implementation deliberately keeps the existing project interface:
    ``plan(...)`` returns a local waypoint trajectory, then the existing Pure
    Pursuit controller tracks it.
    """

    def __init__(self, conf, map_path: str, wpt_path: str, wb: float, v_scale: float):
        self.conf = conf
        self.map_path = map_path
        self.map_ext = ".png"
        self.wheelbase = float(wb)
        self.v_scale = float(v_scale)

        raw = np.loadtxt(wpt_path, delimiter=";", skiprows=1)
        self.waypoints = np.vstack((
            raw[:, 3],  # x
            raw[:, 4],  # y
            raw[:, 9],  # v
            raw[:, 7],  # heading
            raw[:, 1],  # s
        )).T.astype(float)
        self.waypoints[:, 2] *= self.v_scale

        self.d_right_left = np.vstack((raw[:, 5], raw[:, 6])).astype(float)
        self.s_max = float(self.waypoints[-1, 4])
        self.wp_count = int(self.waypoints.shape[0])
        self.ds_nominal = float(np.median(np.diff(self.waypoints[:, 4])))
        self.v_max = float(np.max(self.waypoints[:, 2]))

        self.state = SPLINER_STATE_GBFREE
        self.last_obstacle = None
        self.debug_frenet_traj = None
        self.best_traj = None
        self.tracker = PurePursuitPlanner(conf, wpt_path, wb=wb)

    # ---------------------------------------------------------------------
    # Frenet conversion
    # ---------------------------------------------------------------------
    def cartesian_to_frenet(self, x: float, y: float) -> Tuple[float, float, int]:
        pts = self.waypoints[:, :2]
        distances = np.sum((pts - np.array([x, y], dtype=float)) ** 2, axis=1)
        idx = int(np.argmin(distances))

        xr = float(self.waypoints[idx, 0])
        yr = float(self.waypoints[idx, 1])
        psi = float(self.waypoints[idx, 3])
        s_ref = float(self.waypoints[idx, 4])

        dx = float(x) - xr
        dy = float(y) - yr
        ds = dx * np.cos(psi) + dy * np.sin(psi)
        d = -dx * np.sin(psi) + dy * np.cos(psi)
        s = (s_ref + ds) % self.s_max
        return float(s), float(d), idx

    def frenet_to_cartesian(self, s_query, d_query):
        s_arr = np.asarray(s_query, dtype=float)
        d_arr = np.asarray(d_query, dtype=float)
        s_mod = np.mod(s_arr, self.s_max)

        s_base = self.waypoints[:, 4]
        s_ext = np.concatenate([s_base, s_base[1:] + self.s_max])
        x_ext = np.concatenate([self.waypoints[:, 0], self.waypoints[1:, 0]])
        y_ext = np.concatenate([self.waypoints[:, 1], self.waypoints[1:, 1]])
        v_ext = np.concatenate([self.waypoints[:, 2], self.waypoints[1:, 2]])
        sin_ext = np.concatenate([np.sin(self.waypoints[:, 3]), np.sin(self.waypoints[1:, 3])])
        cos_ext = np.concatenate([np.cos(self.waypoints[:, 3]), np.cos(self.waypoints[1:, 3])])
        right_ext = np.concatenate([self.d_right_left[0], self.d_right_left[0, 1:]])
        left_ext = np.concatenate([self.d_right_left[1], self.d_right_left[1, 1:]])

        x_ref = np.interp(s_mod, s_ext, x_ext)
        y_ref = np.interp(s_mod, s_ext, y_ext)
        v_ref = np.interp(s_mod, s_ext, v_ext)
        sin_ref = np.interp(s_mod, s_ext, sin_ext)
        cos_ref = np.interp(s_mod, s_ext, cos_ext)
        psi_ref = np.arctan2(sin_ref, cos_ref)
        d_right = np.interp(s_mod, s_ext, right_ext)
        d_left = np.interp(s_mod, s_ext, left_ext)

        x = x_ref - d_arr * np.sin(psi_ref)
        y = y_ref + d_arr * np.cos(psi_ref)
        return x, y, v_ref, psi_ref, d_right, d_left

    # ---------------------------------------------------------------------
    # State logic
    # ---------------------------------------------------------------------
    def _global_segment(self, ego_s: float, speed_cap: float) -> np.ndarray:
        s_samples = np.arange(
            ego_s,
            ego_s + SPLINER_TRAJ_LENGTH_M + SPLINER_RESOLUTION_M,
            SPLINER_RESOLUTION_M,
        )
        d_samples = np.zeros_like(s_samples)
        x, y, v, psi, _, _ = self.frenet_to_cartesian(s_samples, d_samples)
        v = np.minimum(v, float(speed_cap))
        traj = np.column_stack([x, y, v, psi])
        self.debug_frenet_traj = np.column_stack([np.mod(s_samples, self.s_max), d_samples])
        return traj

    def _collect_front_obstacles(self, ego_s: float, opp_poses: np.ndarray) -> List[SplinerObstacle]:
        obstacles: List[SplinerObstacle] = []
        for j, row in enumerate(np.asarray(opp_poses, dtype=float)):
            s_opp, d_opp, _ = self.cartesian_to_frenet(float(row[0]), float(row[1]))
            gap_s = (s_opp - ego_s) % self.s_max
            if gap_s < SPLINER_LOOKAHEAD_M and abs(d_opp) < SPLINER_OBS_TRAJ_THRESH_M:
                obstacles.append(
                    SplinerObstacle(
                        agent_idx=j,
                        s_center=float(s_opp),
                        d_center=float(d_opp),
                        gap_s=float(gap_s),
                        speed=float(row[3]),
                    )
                )
        return obstacles

    def _choose_overtake_side(self, obs: SplinerObstacle) -> Tuple[str, float]:
        _, _, _, _, d_right_track, d_left_track = self.frenet_to_cartesian(
            np.array([obs.s_center]),
            np.array([0.0]),
        )
        d_right_track = float(d_right_track[0])
        d_left_track = float(d_left_track[0])

        left_space = d_left_track - obs.d_left
        right_space = obs.d_right + d_right_track
        min_space = SPLINER_EVASION_DIST_M + SPLINER_BOUND_MINDIST_M

        d_apex_left = obs.d_left + SPLINER_EVASION_DIST_M
        d_apex_right = obs.d_right - SPLINER_EVASION_DIST_M

        if d_apex_left < 0.0:
            d_apex_left = 0.0
        if d_apex_right > 0.0:
            d_apex_right = 0.0

        if right_space > min_space and left_space < min_space:
            return "right", float(d_apex_right)
        if left_space > min_space and right_space < min_space:
            return "left", float(d_apex_left)
        if abs(d_apex_left) <= abs(d_apex_right):
            return "left", float(d_apex_left)
        return "right", float(d_apex_right)

    def _local_curvature_side(self, s_apex_unwrapped: float) -> str:
        s_probe = s_apex_unwrapped + np.arange(SPLINER_NUM_CURVATURE_CHECK_POINTS) * self.ds_nominal
        _, _, _, psi_probe, _, _ = self.frenet_to_cartesian(s_probe, np.zeros_like(s_probe))
        psi_unwrapped = np.unwrap(psi_probe)
        curvature_sign = float(np.sum(np.diff(psi_unwrapped)))
        return "left" if curvature_sign < 0.0 else "right"

    def _make_overtake_spline(self, ego_s: float, obs: SplinerObstacle) -> Tuple[bool, np.ndarray]:
        s_apex = ego_s + obs.gap_s
        overtake_side, d_apex = self._choose_overtake_side(obs)
        outside_side = self._local_curvature_side(s_apex)

        speed_scale = np.clip(
            1.0 + SPLINER_VELOCITY_LENGTH_SCALER_LIMIT * 0.0 + self.last_ego_speed / self.v_max,
            1.0,
            SPLINER_VELOCITY_LENGTH_SCALER_LIMIT,
        )

        s_points = []
        d_points = []
        for offset in SPLINER_APEX_OFFSETS_M:
            offset_scaled = float(offset) * speed_scale
            if outside_side == overtake_side:
                offset_scaled *= 1.75
            s_points.append(s_apex + offset_scaled)
            d_points.append(d_apex if offset == 0.0 else 0.0)

        s_points = np.asarray(s_points, dtype=float)
        d_points = np.asarray(d_points, dtype=float)
        spline = InterpolatedUnivariateSpline(s_points, d_points, k=3)

        s_samples = np.arange(s_points[0], s_points[-1] + SPLINER_RESOLUTION_M, SPLINER_RESOLUTION_M)
        d_samples = spline(s_samples)
        if d_apex < 0.0:
            d_samples = np.clip(d_samples, d_apex, 0.0)
        else:
            d_samples = np.clip(d_samples, 0.0, d_apex)

        x, y, v, psi, d_right, d_left = self.frenet_to_cartesian(s_samples, d_samples)
        valid_left = d_samples <= d_left - SPLINER_BOUND_MINDIST_M
        valid_right = d_samples >= -d_right + SPLINER_BOUND_MINDIST_M
        valid = bool(np.all(valid_left & valid_right))

        v = v * SPLINER_OVERTAKE_SPEED_SCALE
        traj = np.column_stack([x, y, v, psi])
        self.debug_frenet_traj = np.column_stack([np.mod(s_samples, self.s_max), d_samples])
        return valid, traj

    def _trailing_speed(self, ego_s: float, ego_v: float, obs: SplinerObstacle) -> float:
        gap = obs.gap_s
        e_gap = SPLINER_TRAILING_TARGET_GAP_M - gap
        delta_vs = float(ego_v) - obs.speed
        v_des = obs.speed - (SPLINER_TRAILING_KP * e_gap + SPLINER_TRAILING_KD * delta_vs)
        return float(max(SPLINER_TRAILING_MIN_SPEED_MPS, v_des))

    def plan(
        self,
        pose_x: float,
        pose_y: float,
        pose_theta: float,
        velocity: float,
        opp_poses: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        del pose_theta
        self.last_ego_speed = float(velocity)
        ego_s, _, _ = self.cartesian_to_frenet(float(pose_x), float(pose_y))
        front_obstacles = self._collect_front_obstacles(ego_s, opp_poses)

        if len(front_obstacles) == 0:
            self.state = SPLINER_STATE_GBFREE
            traj = self._global_segment(ego_s, speed_cap=self.v_max)
        else:
            target_obs = min(front_obstacles, key=lambda item: item.gap_s)
            self.last_obstacle = target_obs
            spline_valid, spline_traj = self._make_overtake_spline(ego_s, target_obs)
            if spline_valid:
                self.state = SPLINER_STATE_OVERTAKE
                traj = spline_traj
            else:
                self.state = SPLINER_STATE_TRAILING
                speed_cap = self._trailing_speed(ego_s, velocity, target_obs)
                traj = self._global_segment(ego_s, speed_cap=speed_cap)

        self.best_traj = traj
        return traj, traj[:, 3]


def setup_spliner_planner(
    args,
    map_name,
    raceline_file,
    config_path,
    v_scale,
    ocp_conf,
    game_block_conf,
    biased_type,
):
    del ocp_conf, game_block_conf, biased_type

    config = load_config(config_path)
    map_directory, map_path = get_map_paths(map_name)
    raceline_path = os.path.join(map_directory, f"{map_name}_{raceline_file}.csv")

    planner = SplinerPlanner(
        config,
        map_path,
        raceline_path,
        wb=args.wheel_base,
        v_scale=v_scale,
    )
    return planner, map_directory
