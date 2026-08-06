
import logging
import os

import numpy as np
import yaml
from numba import njit
from PIL import Image
from pyclothoids import Clothoid
from scipy.ndimage import distance_transform_edt as edt
    
from players.planner.controller.pure_pursuit import PurePursuitPlanner
from .collision_checks import collision, get_vertices
from players.utils.common import nearest_point, intersect_point, get_rotation_matrix, zero_2_2pi, map_collision

logger = logging.getLogger(__name__)

def sample_traj(clothoid, npts, v):
    # traj (m, 5)
    traj = np.empty((npts, 5))
    k0 = clothoid.Parameters[3]
    dk = clothoid.Parameters[4]

    for i in range(npts):
        s = i * (clothoid.length / max(npts - 1, 1))
        traj[i, 0] = clothoid.X(s)
        traj[i, 1] = clothoid.Y(s)
        traj[i, 2] = v
        traj[i, 3] = clothoid.Theta(s)
        traj[i, 4] = np.sqrt(clothoid.XDD(s) ** 2 + clothoid.YDD(s) ** 2)
    return traj

class LatticePlanner:
    def __init__(self, args, conf, map_path, wpt_path, wb, v_scale=1.0):
        self.wheelbase = wb
        self.map_path = map_path
        self.map_ext = ".png"

        load_waypoints = np.loadtxt(wpt_path, delimiter=";", skiprows=1)
        self.d_right_left = np.vstack((load_waypoints[:, 5], load_waypoints[:, 6]))
        self.waypoints = np.vstack((
            load_waypoints[:, 3],  # x
            load_waypoints[:, 4],  # y
            load_waypoints[:, 9],  # vx/reference speed
            load_waypoints[:, 7],  # heading
            load_waypoints[:, 1],  # s
        )).T
        self.waypoints[:, 2] *= v_scale
        self.s_max = self.waypoints[-1, 4]

        self.lh_grid_rows = 30
        self.traj_num = self.lh_grid_rows * 11
        self.lh_grid_lb = conf.lh_grid_lb
        self.lh_grid_ub = conf.lh_grid_ub
        self.traj_points = conf.traj_points
        self.traj_v_scale = conf.traj_v_scale
        self.v_lattice_span = np.linspace(conf.traj_v_span_min, conf.traj_v_span_max, conf.traj_v_span_num)
        self.v_lattice_num = conf.traj_v_span_num

        self.params_num = conf.params_num
        self.params_name = conf.params_name
        self.params_idx = {}
        count = 0
        for name, num in zip(self.params_name, self.params_num):
            self.params_idx[name] = count
            count += num

        try:
            self.cost_weights_num = conf.weights_num
        except Exception:
            self.cost_weights_num = 4
        self.set_cost_weights(self.cost_weights_num)

        self.sample_func = sample_lookahead_square
        self.shape_cost_funcs = [get_follow_optim_cost]
        self.constant_cost_funcs = [get_map_collision]
        self.selection_func = np.argmin

        self.best_traj = None
        self.best_traj_ref_v = 0.0
        self.best_traj_idx = 0
        self.prev_traj_local = np.zeros((self.traj_points, 2))
        self.prev_opp_pose = np.zeros((0, 2))
        self.goal_grid = None
        self.state_i = None
        self.state_t = None
        self.step_all_cost = {}
        self.all_costs = None
        self.last_s = 0.0

        self.tracker = PurePursuitPlanner(conf, wpt_path, wb=wb)
        self.conf = conf
        self.step = 0

        map_img_path = os.path.splitext(self.map_path)[0] + self.map_ext
        self.map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM)).astype(np.float64)
        self.map_img[self.map_img <= 128.0] = 0.0
        self.map_img[self.map_img > 128.0] = 255.0
        self.map_height = self.map_img.shape[0]
        self.map_width = self.map_img.shape[1]

        with open(self.map_path + ".yaml", "r") as yaml_stream:
            map_metadata = yaml.safe_load(yaml_stream)
            self.map_resolution = map_metadata["resolution"]
            self.origin = map_metadata["origin"]

        self.orig_x = self.origin[0]
        self.orig_y = self.origin[1]
        self.orig_s = np.sin(self.origin[2])
        self.orig_c = np.cos(self.origin[2])
        self.dt = self.map_resolution * edt(self.map_img)
        self.map_metainfo = (
            self.orig_x,
            self.orig_y,
            self.orig_c,
            self.orig_s,
            self.map_height,
            self.map_width,
            self.map_resolution,
        )
        self.collision_thres = 0.35

    def set_parameters(self, parameters, v_scale=6.0):
        if isinstance(parameters, np.ndarray):
            for name, num in zip(self.params_name, self.params_num):
                start = self.params_idx[name]
                if name == "cost_weights":
                    self.set_cost_weights(parameters[start:start + num])
                else:
                    setattr(self, name, parameters[start])
        else:
            for key, value in parameters.items():
                if key == "cost_weights":
                    self.set_cost_weights(value)
                else:
                    setattr(self, key, value)

    def set_cost_weights(self, cost_weights):
        if isinstance(cost_weights, int):
            n = cost_weights
            self.cost_weights = np.array([1.0 / n] * n)
            return
        if len(cost_weights) != self.cost_weights_num:
            raise ValueError("Length of cost weights must match number of cost functions.")
        self.cost_weights = np.asarray(cost_weights, dtype=np.float64)

    def generate_candidates(self, pose_x, pose_y, pose_theta, velocity, waypoints=None):
        waypoints = self.waypoints
        ego_pose = np.array([pose_x, pose_y, pose_theta])
        _, _, t, nearest_i = nearest_point(ego_pose[:2], waypoints[:, 0:2])
        self.state_i = nearest_i
        self.state_t = t

        nearest_s = waypoints[nearest_i, -1]
        min_L = self.tracker.get_L(velocity)
        lh_grid = np.linspace(min_L + self.lh_grid_lb, min_L + self.lh_grid_ub, self.lh_grid_rows)
        self.goal_grid, ref_col_idx, _ = self.sample(pose_x, pose_y, pose_theta, velocity, waypoints, lh_grid, nearest_s)

        all_traj = []
        all_traj_clothoid = []
        for point in self.goal_grid:
            clothoid = Clothoid.G1Hermite(pose_x, pose_y, pose_theta, point[0], point[1], point[2])
            all_traj.append(sample_traj(clothoid, self.traj_points, point[3]))
            all_traj_clothoid.append(np.array(clothoid.Parameters))

        return np.array(all_traj), np.array(all_traj_clothoid), ego_pose, ref_col_idx

    def select_best_response(self, all_traj, all_traj_clothoid, ego_pose, opp_poses, ref_col_idx):
        self.step_all_cost = {}
        all_costs = self.eval(all_traj, all_traj_clothoid, opp_poses, ego_pose, ref_col_idx)
        self.all_costs = all_costs

        best_traj_idx = int(self.selection_func(all_costs))
        row_idx, col_idx = divmod(best_traj_idx, self.v_lattice_num)
        self.best_traj_idx = best_traj_idx

        best_traj = all_traj[row_idx].copy()
        self.best_traj_ref_v = best_traj[-1, 2]
        best_traj[:, 2] *= self.v_lattice_span[col_idx] * self.traj_v_scale

        self.best_traj = best_traj
        self.prev_traj_local = traj_global2local(ego_pose, best_traj[:, :2])
        if opp_poses is not None and len(opp_poses) > 0:
            self.prev_opp_pose = opp_poses[:, :2].copy()
        return best_traj

    def sample(self, pose_x, pose_y, pose_theta, velocity, waypoints, lh_grid, nearest_s):
        s_dist = waypoints[:, -1]
        track_len = s_dist[-1]
        query_s = nearest_s + lh_grid

        s_ext = np.concatenate([s_dist, s_dist[1:] + track_len])
        right_ext = np.concatenate([self.d_right_left[0], self.d_right_left[0][1:]])
        left_ext = np.concatenate([self.d_right_left[1], self.d_right_left[1][1:]])
        d_right_left_grid = np.vstack((-np.interp(query_s, s_ext, right_ext), np.interp(query_s, s_ext, left_ext)))

        return self.sample_func(
            pose_x,
            pose_y,
            pose_theta,
            velocity,
            waypoints,
            lh_grid,
            d_right_left_grid=d_right_left_grid,
        )

    def eval(self, all_traj, all_traj_clothoid, opp_poses, ego_pose, ref_col_idx):
        cost_weights = self.cost_weights
        n, k = self.traj_num, self.v_lattice_num
        mean_k, _ = get_curvature(all_traj, all_traj_clothoid)
        cost = np.zeros(self.traj_num)

        for i, func in enumerate(self.shape_cost_funcs):
            cur_cost = func(
                all_traj,
                all_traj_clothoid,
                opp_poses,
                ego_pose,
                self.prev_traj_local,
                self.dt,
                self.map_metainfo,
                ref_col_idx,
            )
            cur_cost = cost_weights[i] * cur_cost
            self.step_all_cost[func.__name__] = cur_cost
            cost += cur_cost

        for func in self.constant_cost_funcs:
            cur_cost = func(
                all_traj,
                all_traj_clothoid,
                opp_poses,
                ego_pose,
                self.prev_traj_local,
                self.dt,
                self.map_metainfo,
                self.collision_thres,
            )
            self.step_all_cost[func.__name__] = cur_cost
            cost += cur_cost

        mean_k_lattice = np.repeat(mean_k, k).reshape(n, k)
        all_traj_v = all_traj[:, -1, 2]
        traj_v_lattice = np.repeat(all_traj_v, k).reshape(n, k) * self.v_lattice_span * self.traj_v_scale
        abs_v_cost = (
            -cost_weights[-3] * np.log(1.0 + traj_v_lattice)
            + cost_weights[-2] * (mean_k_lattice - np.min(mean_k)) * traj_v_lattice
        )
        self.step_all_cost["abs_v_cost"] = abs_v_cost

        collision_cost = cost_weights[-1] * get_obstacle_collision_cost(
            all_traj,
            traj_v_lattice,
            opp_poses,
        )
        self.step_all_cost["obstacle_collision_cost"] = collision_cost

        return np.repeat(cost, k).reshape(n, k) + abs_v_cost + collision_cost


@njit(cache=True)
def sample_lookahead_square(pose_x, pose_y, pose_theta, velocity, waypoints, lookahead_distances=None, d_right_left_grid=None):
    grid_num = 11
    position = np.array([pose_x, pose_y])
    nearest_p, nearest_dist, t, nearest_i = nearest_point(position, waypoints[:, 0:2])
    xy_grid = np.zeros((2, 1))
    theta_grid = np.zeros((len(lookahead_distances), 1))
    v_grid = np.zeros((len(lookahead_distances), 1))
    ref_col_idx = np.zeros(len(lookahead_distances), dtype=np.int64)
    ref_global_idx = np.zeros(len(lookahead_distances), dtype=np.int64)

    for i, d in enumerate(lookahead_distances):
        widths_i = np.linspace(d_right_left_grid[0, i], d_right_left_grid[1, i], num=grid_num)
        ref_col_idx[i] = np.argmin(np.abs(widths_i))
        ref_global_idx[i] = i * grid_num + ref_col_idx[i]
        local_span_i = np.vstack((np.zeros_like(widths_i), widths_i))
        lh_pt, i2, t2 = intersect_point(np.ascontiguousarray(nearest_p), d, waypoints[:, 0:2], t + nearest_i, wrap=True)
        i2 = int(i2)
        lh_pt_theta = waypoints[i2, 3]
        lh_pt_v = waypoints[i2, 2]
        lh_span_points = get_rotation_matrix(lh_pt_theta) @ local_span_i + lh_pt.reshape(2, -1)
        xy_grid = np.hstack((xy_grid, lh_span_points))
        theta_grid[i] = zero_2_2pi(lh_pt_theta)
        v_grid[i] = lh_pt_v

    xy_grid = xy_grid[:, 1:]
    theta_grid = np.repeat(theta_grid, grid_num).reshape(1, -1)
    v_grid = np.repeat(v_grid, grid_num).reshape(1, -1)
    return np.vstack((xy_grid, theta_grid, v_grid)).T, ref_col_idx, ref_global_idx


@njit(cache=True)
def traj_global2local(ego_pose, traj):
    new_traj = np.zeros_like(traj)
    pose_x, pose_y, pose_theta = ego_pose
    c = np.cos(pose_theta)
    s = np.sin(pose_theta)
    new_traj[..., 0] = c * (traj[..., 0] - pose_x) + s * (traj[..., 1] - pose_y)
    new_traj[..., 1] = -s * (traj[..., 0] - pose_x) + c * (traj[..., 1] - pose_y)
    return new_traj


@njit(cache=True)
def get_follow_optim_cost(traj, traj_clothoid, opp_poses=None, ego_pose=None, prev_traj=None, dt=None, map_metainfo=None, ref_col_idx=None):
    n = traj.shape[0]
    cols = 11
    cost = np.zeros(n, dtype=np.float64)
    for traj_idx in range(n):
        row_idx = traj_idx // cols
        col_idx = traj_idx % cols
        cost[traj_idx] = (col_idx - ref_col_idx[row_idx]) ** 2
    return cost


def get_curvature(traj, traj_clothoid):
    k0 = traj_clothoid[:, 3].reshape(-1, 1)
    dk = traj_clothoid[:, 4].reshape(-1, 1)
    s = traj_clothoid[:, -1]
    s_pts = np.linspace(np.zeros_like(s), s, num=traj.shape[1]).T
    traj_k = k0 + dk * s_pts
    traj_k_abs = np.abs(traj_k)
    traj_steer = np.arctan(0.307 * traj_k)
    max_steer = np.max(np.abs(traj_steer), axis=1)
    mean_k = np.mean(traj_k_abs, axis=1)
    for i in range(len(mean_k)):
        if max_steer[i] > 0.4:
            mean_k[i] *= 2.0
    return mean_k, np.max(traj_k_abs, axis=1)


@njit(cache=True)
def get_map_collision(traj, traj_clothoid, opp_poses=None, ego_pose=None, prev_traj=None, dt=None, map_metainfo=None, collision_thres=0.35):
    all_traj_pts = np.ascontiguousarray(traj).reshape(-1, 5)
    collisions = map_collision(all_traj_pts[:, 0:2], dt, map_metainfo, eps=collision_thres)
    collisions = collisions.reshape(len(traj), -1)
    cost = np.zeros(len(traj), dtype=np.float64)
    for i in range(len(traj)):
        if np.any(collisions[i]):
            cost[i] = 3000.0
    return cost


@njit(cache=True)
def get_obstacle_collision_cost(traj, v_lattice, opp_poses):
    max_cost = 40.0
    min_cost = 10.0
    width = 0.31
    length = 0.58
    safety_width = 0.15
    safety_length = 0.2

    n, m, _ = traj.shape
    k = v_lattice.shape[1]
    base_cost = np.zeros(n, dtype=np.float64)

    if opp_poses is None or len(opp_poses) == 0:
        return np.repeat(base_cost, k).reshape(n, k)

    for opp_i in range(len(opp_poses)):
        opp_pose = np.zeros(3, dtype=np.float64)
        opp_pose[0] = float(opp_poses[opp_i, 0])
        opp_pose[1] = float(opp_poses[opp_i, 1])
        opp_pose[2] = float(opp_poses[opp_i, 2])
        opp_box = get_vertices(opp_pose, length + safety_length, width + safety_width)

        for i in range(n):
            tr = traj[i]
            for j in range(m):
                ego_box = get_vertices(tr[j], length + safety_length, width + safety_width)
                if collision(opp_box, ego_box):
                    collision_cost = max_cost - j * (max_cost - min_cost) / m
                    if collision_cost > base_cost[i]:
                        base_cost[i] = collision_cost
                    break

    return np.repeat(base_cost, k).reshape(n, k)
