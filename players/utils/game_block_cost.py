import numpy as np
from numba import njit


@njit(cache=True)
def _point_to_frenet_d_s(x, y, waypoints):
    best_i = 0
    best_dist2 = 1e18

    for i in range(waypoints.shape[0]):
        dx = x - waypoints[i, 0]
        dy = y - waypoints[i, 1]
        dist2 = dx * dx + dy * dy

        if dist2 < best_dist2:
            best_dist2 = dist2
            best_i = i

    wx = waypoints[best_i, 0]
    wy = waypoints[best_i, 1]
    theta = waypoints[best_i, 3]
    s_ref = waypoints[best_i, 4]

    dx = x - wx
    dy = y - wy

    d = -dx * np.sin(theta) + dy * np.cos(theta)
    return d, s_ref


@njit(cache=True)
def _traj_tail_mean_d_and_last_s(traj, waypoints, tail_ratio):
    m = traj.shape[0]
    start_idx = int((1.0 - tail_ratio) * m)

    if start_idx < 0:
        start_idx = 0
    if start_idx > m - 1:
        start_idx = m - 1

    d_sum = 0.0
    count = 0
    last_s = 0.0

    for k in range(start_idx, m):
        d_k, s_k = _point_to_frenet_d_s(traj[k, 0], traj[k, 1], waypoints)
        d_sum += d_k
        count += 1
        last_s = s_k

    return d_sum / count, last_s


@njit(cache=True)
def _min_aligned_distance(ego_traj, opp_traj):
    m = min(ego_traj.shape[0], opp_traj.shape[0])
    dmin = 1e18

    for k in range(m):
        dx = ego_traj[k, 0] - opp_traj[k, 0]
        dy = ego_traj[k, 1] - opp_traj[k, 1]
        d = np.sqrt(dx * dx + dy * dy)

        if d < dmin:
            dmin = d

    return dmin


@njit(cache=True)
def _compute_game_block_cost(
    all_traj,
    opp_traj,
    waypoints,
    contest_s_gap,
    longitudinal_weight,
    contest_weight,
    block_weight,
    tail_ratio,
    role_s_margin,
    safety_weight,
    safe_dist,
):
    """

    c_game = c_long + c_contest + c_block + c_safety

    all_traj:   [K, T, 3] = x, y, v
    opp_traj:   [T, 3]    = x, y, v
    waypoints:  [M, >=5]  = x, y, v_ref, heading, s
    """
    K = all_traj.shape[0]
    cost = np.zeros((K, 1), dtype=np.float64)
    def circular_signed_gap(ego_s, opp_s, track_len):
        return ((ego_s - opp_s + 0.5 * track_len) % track_len) - 0.5 * track_len
    track_len = waypoints[-1, 4]

    opp_mean_d, opp_last_s = _traj_tail_mean_d_and_last_s(
        opp_traj,
        waypoints,
        tail_ratio,
    )
    _, opp_s0 = _point_to_frenet_d_s(
        opp_traj[0, 0],
        opp_traj[0, 1],
        waypoints,
    )

    eps = 1e-6

    for i in range(K):
        ego_traj = all_traj[i]

        ego_mean_d, ego_last_s = _traj_tail_mean_d_and_last_s(
            ego_traj,
            waypoints,
            tail_ratio,
        )
        _, ego_s0 = _point_to_frenet_d_s(
            ego_traj[0, 0],
            ego_traj[0, 1],
            waypoints,
        )

        current_s_gap = circular_signed_gap(ego_s0, opp_s0, track_len)
        future_s_gap = circular_signed_gap(ego_last_s, opp_last_s, track_len)

        # 1) Longitudinal future-advantage reward.
        # The reward is stronger when the two vehicles are close enough to interact.
        contest_alpha = 1.0 / (1.0 + np.abs(current_s_gap) / (contest_s_gap + eps))
        cost[i, 0] -= longitudinal_weight * contest_alpha * future_s_gap

        # 2) Contest-state reward.
        contest_score = 0.0
        if np.abs(future_s_gap) < contest_s_gap:
            contest_score += 1.0
        cost[i, 0] -= contest_weight * contest_score

        # 3) Blocking alignment reward.
        if current_s_gap > role_s_margin and current_s_gap < contest_s_gap:
            lateral_gap = np.abs(ego_mean_d - opp_mean_d)
            cost[i, 0] -= block_weight / (1.0 + lateral_gap)

        # 4) Safety penalty.
        dmin = _min_aligned_distance(ego_traj, opp_traj)
        if dmin < safe_dist:
            violation = safe_dist - dmin
            cost[i, 0] += safety_weight * violation * violation


    return cost


class GameBlockCost:
    def __init__(
        self,
        contest_s_gap,
        longitudinal_weight,
        contest_weight,
        block_weight,
        tail_ratio,
        role_s_margin,
        safety_weight,
        safe_dist,
    ):
        self.contest_s_gap = contest_s_gap
        self.longitudinal_weight = longitudinal_weight
        self.contest_weight = contest_weight
        self.block_weight = block_weight
        self.tail_ratio = tail_ratio
        self.role_s_margin = role_s_margin
        self.safety_weight = safety_weight
        self.safe_dist = safe_dist

    def __call__(self, all_traj, opp_traj, waypoints):
        return _compute_game_block_cost(
            np.ascontiguousarray(all_traj[:, :, :3], dtype=np.float64),
            np.ascontiguousarray(opp_traj[:, :3], dtype=np.float64),
            np.ascontiguousarray(waypoints, dtype=np.float64),
            self.contest_s_gap,
            self.longitudinal_weight,
            self.contest_weight,
            self.block_weight,
            self.tail_ratio,
            self.role_s_margin,
            self.safety_weight,
            self.safe_dist,
        )
