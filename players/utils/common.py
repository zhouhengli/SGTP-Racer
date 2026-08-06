import numpy as np
import math
from numba import njit
import yaml
from types import SimpleNamespace as Namespace
import os
from typing import Any, Tuple, Dict, List, Optional, Sequence
import csv
import json
from datetime import datetime
from pathlib import Path


RAW_MANIFEST_COLUMNS: List[str] = [
    "case_id",
    "map_name",
    "trace_path",
    "trace_meta_path",
    "planned_trajs_path",
    "boundary_offsets_path",
    "refline_xy_path",
    "artifact_dir",
]


def now_tag() -> str:
    """Return a compact wall-clock tag."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def clean_id(x: Any) -> str:
    """Convert text into a file-system-safe identifier."""
    return str(x).replace("/", "_").replace(" ", "_").replace(":", "_")


def json_ready(x: Any) -> Any:
    """Convert common numpy/path objects into JSON-serializable values."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): json_ready(v) for k, v in x.items()}
    if isinstance(x, list):
        return [json_ready(v) for v in x]
    if isinstance(x, tuple):
        return [json_ready(v) for v in x]
    return x


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load one YAML mapping."""
    path = Path(path)
    with path.open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return data


def save_yaml(path: Path, obj: Any) -> None:
    """Write one YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(json_ready(obj), f, sort_keys=False)


def load_json(path: Path) -> Dict[str, Any]:
    """Load one JSON mapping."""
    path = Path(path)
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"JSON root must be a mapping: {path}")
    return data


def write_json(path: Path, obj: Any, indent: int = 2) -> None:
    """Write one JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(json_ready(obj), f, indent=indent)


def require_key(mapping: Dict[str, Any], key: str) -> Any:
    """Read one required dictionary key."""
    return mapping[key]


def require_cfg(cfg: Dict[str, Any], dotted_path: str) -> Any:
    """Read one required dotted configuration path."""
    cur: Any = cfg
    for key in dotted_path.split("."):
        cur = cur[key]
    return cur


def format_duration(seconds: float) -> str:
    """Format seconds as a compact duration string."""
    seconds_i = max(0, int(round(seconds)))
    hours, rem = divmod(seconds_i, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def rel_to(path: Path, base: Path) -> str:
    """Return one POSIX path relative to a base directory."""
    return Path(path).resolve().relative_to(Path(base).resolve()).as_posix()


def csv_cell(value: Any) -> Any:
    """Convert one value for CSV output."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        return json.dumps(json_ready(value))
    return value


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> None:
    """Write dictionaries to CSV with explicit columns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_cell(row.get(key, "")) for key in columns})


def columns_for_rows(rows: Sequence[Dict[str, Any]], preferred: Sequence[str]) -> List[str]:
    """Return stable CSV columns from preferred names and row keys."""
    seen = set()
    columns: List[str] = []
    for key in preferred:
        if key not in seen:
            columns.append(key)
            seen.add(key)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                columns.append(key)
                seen.add(key)
    return columns


def parse_float(value: Any) -> Optional[float]:
    """Parse numeric, bool, and bool-like strings as finite floats."""
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    text = str(value).strip()
    if text == "":
        return None
    lowered = text.lower()
    if lowered in {"true", "yes"}:
        return 1.0
    if lowered in {"false", "no"}:
        return 0.0
    try:
        number = float(text)
    except ValueError:
        return None
    if not np.isfinite(number):
        return None
    return number


def aggregate_rows(rows: Sequence[Dict[str, Any]], group_key: str) -> List[Dict[str, Any]]:
    """Aggregate all scalar numeric metrics by one group key."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[group_key]), []).append(row)

    out_rows: List[Dict[str, Any]] = []
    for name, group_rows in sorted(groups.items()):
        out: Dict[str, Any] = {"group": name, group_key: name, "num_rows": len(group_rows)}
        metric_keys = columns_for_rows(group_rows, [])
        for key in metric_keys:
            values: List[float] = []
            for row in group_rows:
                if key not in row:
                    continue
                value = parse_float(row[key])
                if value is not None:
                    values.append(value)
            arr = np.asarray(values, dtype=float)
            if arr.size:
                out[f"{key}_mean"] = float(np.mean(arr))
                out[f"{key}_std"] = float(np.std(arr))
        out_rows.append(out)
    return out_rows


def aggregate_overall(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate all scalar numeric metrics over all rows."""
    out: Dict[str, Any] = {"group": "overall", "num_rows": len(rows)}
    metric_keys = columns_for_rows(rows, [])
    for key in metric_keys:
        values: List[float] = []
        for row in rows:
            if key not in row:
                continue
            value = parse_float(row[key])
            if value is not None:
                values.append(value)
        arr = np.asarray(values, dtype=float)
        if arr.size:
            out[f"{key}_mean"] = float(np.mean(arr))
            out[f"{key}_std"] = float(np.std(arr))
    return out


def _point_to_frenet_d(x: float, y: float, waypoints: np.ndarray) -> float:
    """
    Return signed lateral offset d to the nearest waypoint frame.
    waypoints format: [x, y, vx, heading, s]
    """
    dx = waypoints[:, 0] - x
    dy = waypoints[:, 1] - y
    idx = int(np.argmin(dx * dx + dy * dy))

    wx = float(waypoints[idx, 0])
    wy = float(waypoints[idx, 1])
    wpsi = float(waypoints[idx, 3])

    rx = x - wx
    ry = y - wy

    c = np.cos(wpsi)
    s = np.sin(wpsi)

    d = -rx * s + ry * c
    return float(d)

def _safe_ratio(num: float, den: float) -> float:
    return float(num) / max(float(den), 1.0)

@njit(cache=True)
def xy_2_rc(x, y, orig_x, orig_y, orig_c, orig_s, height, width, resolution):
    """
    Translate (x, y) coordinate into (r, c) in the matrix
        Args:
            x (float): coordinate in x (m)
            y (float): coordinate in y (m)
            orig_x (float): x coordinate of the map origin (m)
            orig_y (float): y coordinate of the map origin (m)

        Returns:
            r (int): row number in the transform matrix of the given point
            c (int): column number in the transform matrix of the given point
    """
    # translation
    x_trans = x - orig_x
    y_trans = y - orig_y

    # rotation
    x_rot = x_trans * orig_c + y_trans * orig_s
    y_rot = -x_trans * orig_s + y_trans * orig_c

    # clip the state to be a cell
    if x_rot < 0 or x_rot >= width * resolution or y_rot < 0 or y_rot >= height * resolution:
        c = -1
        r = -1
    else:
        c = int(x_rot / resolution)
        r = int(y_rot / resolution)

    return r, c


@njit(cache=True)
def map_collision(points, dt, map_metainfo, eps=0.4):
    """
    Check wheter a point is in collision with the map

    Args:
        points (numpy.ndarray(N, 2)): points to check
        dt (numpy.ndarray(n, m)): the map distance transform
        map_metainfo (tuple (x, y, c, s, h, w, resol)): map metainfo
        eps (float, default=0.1): collision threshold
    Returns:
        collisions (numpy.ndarray (N, )): boolean vector of wheter input points are in collision

    """
    orig_x, orig_y, orig_c, orig_s, height, width, resolution = map_metainfo
    collisions = np.empty((points.shape[0],))
    for i in range(points.shape[0]):
        if dt[xy_2_rc(points[i, 0], points[i, 1], orig_x, orig_y, orig_c, orig_s, height, width, resolution)] <= eps:
            collisions[i] = True
        else:
            collisions[i] = False
    return np.ascontiguousarray(collisions)


@njit(cache=True)
def zero_2_2pi(angle):
    if angle > 2 * math.pi:
        return angle - 2.0 * math.pi
    if angle < 0:
        return angle + 2.0 * math.pi

    return angle

@njit(cache=True)
def get_rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.ascontiguousarray(np.array([[c, -s], [s, c]]))

@njit(cache=True)
def intersect_point(point, radius, trajectory, t=0.0, wrap=False):
    """
    starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.

    Assumes that the first segment passes within a single radius of the point

    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    """
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0] - 1):
        start = trajectory[i, :]
        end = trajectory[i + 1, :] + 1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V, V)
        b = 2.0 * np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point, point) - 2.0 * np.dot(start, point) - radius * radius
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0 * a)
        t2 = (-b + discriminant) / (2.0 * a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0], :]
            end = trajectory[(i + 1) % trajectory.shape[0], :] + 1e-6
            V = end - start

            a = np.dot(V, V)
            b = 2.0 * np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point, point) - 2.0 * np.dot(start, point) - radius * radius
            discriminant = b * b - 4 * a * c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0 * a)
            t2 = (-b + discriminant) / (2.0 * a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t

def initialize_agents_positions(
    ego_planner: None,
    opp_planner: None,
    ego_idx: int,
    interval_idx: int,
    rng: np.random.Generator
) -> np.ndarray:
    # Ego position setup
    ego_waypoints_xytheta = np.hstack((
        ego_planner.waypoints[:, :2],
        ego_planner.waypoints[:, 3].reshape(-1, 1)
    ))
    ego_pos, _ = random_position(
        ego_waypoints_xytheta, 1, rng, 0.0, 0.0, ego_idx, 0
    )
    
    # Opponent position setup with interval mapping
    opp_waypoints_xytheta = np.hstack((
        opp_planner.waypoints[:, :2],
        opp_planner.waypoints[:, 3].reshape(-1, 1)
    ))
    
    ego_waypoint = ego_waypoints_xytheta[ego_idx]
    ego_map_idx = find_corresponding_waypoint(ego_waypoint, opp_waypoints_xytheta)
    opp_idx = (ego_map_idx + interval_idx) % len(opp_waypoints_xytheta)
    
    opp_pos, _ = random_position(
        opp_waypoints_xytheta, 1, rng, 0.0, 0.0, opp_idx, 0
    )
    
    return np.vstack([ego_pos, opp_pos]), opp_idx

def initialize_multi_agents_positions(agent_planners: List, ego_idx: int, interval_idx: int, rng=None) -> Tuple[np.ndarray, int]:
    """
    Minimal N-agent initializer.

    For N=2, reuse existing initialize_agents_positions exactly.
    For N>2, place agents along the first planner's raceline with interval_idx.
    """
    num_agents = len(agent_planners)

    if num_agents == 2:
        return initialize_agents_positions(
            agent_planners[0],
            agent_planners[1],
            ego_idx,
            interval_idx,
            rng,
        )

    waypoints = agent_planners[0].waypoints
    num_wp = len(waypoints)

    poses = np.zeros((num_agents, 3), dtype=float)

    for i in range(num_agents):
        idx = int((ego_idx + i * interval_idx) % num_wp)

        x = float(waypoints[idx, 0])
        y = float(waypoints[idx, 1])

        # Do not assume waypoint column 2 is heading.
        # Estimate heading from neighboring waypoint geometry.
        prev_idx = (idx - 1) % num_wp
        next_idx = (idx + 1) % num_wp
        dx = float(waypoints[next_idx, 0] - waypoints[prev_idx, 0])
        dy = float(waypoints[next_idx, 1] - waypoints[prev_idx, 1])
        theta = float(np.arctan2(dy, dx))

        poses[i] = np.array([x, y, theta], dtype=float)

    # Keep old output contract: second return value was opp_idx.
    # For N-agent, use 1 as the compatibility opponent index.
    return poses, 1

def compute_control_actions(
    obs: Dict,
    ego_planner: None,
    opp_planner: None,
    ego_best_traj: np.ndarray,
    opp_best_traj: np.ndarray,
    steer_limits: Tuple[float, float] = (None, None)
) -> np.ndarray:
    # Ego control
    ego_steer, ego_speed = ego_planner.tracker.plan(
        obs['poses_x'][0],
        obs['poses_y'][0],
        obs['poses_theta'][0],
        obs['linear_vels_x'][0],
        ego_best_traj
    )
    ego_steer = np.clip(ego_steer, steer_limits[0], steer_limits[1])
    
    # Opponent control
    opp_steer, opp_speed = opp_planner.tracker.plan(
        obs['poses_x'][1],
        obs['poses_y'][1],
        obs['poses_theta'][1],
        obs['linear_vels_x'][1],
        opp_best_traj
    )
    opp_steer = np.clip(opp_steer, steer_limits[0], steer_limits[1])
    
    return np.array([[ego_steer, ego_speed], [opp_steer, opp_speed]])

def compute_multi_control_actions(
        obs: Dict, agent_planners: List, 
        best_trajs: List[np.ndarray], 
        nearest_agent_indices: np.ndarray, 
        steer_limits: Tuple[float, float]
) -> np.ndarray:
    """
    Reuse existing two-agent compute_control_actions by slicing each agent's
    nearest-opponent pair.

    For each real agent i:
      pair index 0 = i
      pair index 1 = nearest_agent_indices[i]

    We only take pair_action[0].
    """
    num_agents = len(agent_planners)
    action = np.zeros((num_agents, 2), dtype=float)

    for i in range(num_agents):
        j = int(nearest_agent_indices[i])
        pair_obs = _slice_obs_for_pair(obs, i, j)

        pair_action = compute_control_actions(
            pair_obs,
            agent_planners[i],
            agent_planners[j],
            best_trajs[i],
            best_trajs[j],
            steer_limits=steer_limits,
        )
        # only to compile with 2 agents version, TODO: fit to dynamic input process
        action[i] = pair_action[0]

    return action


@njit(cache=True)
def nearest_point(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.

    Args:
        point (numpy.ndarray, (2, )): (x, y) of current pose
        trajectory (numpy.ndarray, (N, 2)): array of (x, y) trajectory waypoints
            NOTE: points in trajectory must be unique. If they are not unique, a divide by 0 error will destroy the world

    Returns:
        nearest_point (numpy.ndarray, (2, )): nearest point on the trajectory to the point
        nearest_dist (float): distance to the nearest point
        t (float): nearest point's location as a segment between 0 and 1 on the vector formed by the closest two points on the trajectory. (p_i---*-------p_i+1)
        i (int): index of nearest point in the array of trajectory waypoints
    """
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    projections = trajectory[:-1, :] + (t * diffs.T).T
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment


"""
Geometry utilities
"""


@njit(cache=True)
def zero_2_2pi(angle):
    if angle > 2 * math.pi:
        return angle - 2.0 * math.pi
    if angle < 0:
        return angle + 2.0 * math.pi

    return angle

# (x0, y0, theta0, k0, dk, arc_length)


@njit(cache=True)
def get_actuation_PD(pose_theta, lookahead_point, position, lookahead_distance, wheelbase, prev_error, P, D):
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2] - position)
    speed = lookahead_point[2]
    error = 2.0 * waypoint_y / lookahead_distance ** 2
    if np.abs(waypoint_y) < 1e-4:
        return speed, 0., error
    steering_angle = P * error + D * (error-prev_error)
    return speed, steering_angle, error

@njit(cache=True)
def project_point_to_centerline(point, centerline):
    """
    Project a point onto a centerline and return progress along centerline.
    
    Args:
        point (np.ndarray (2,)): [x, y] position to project
        centerline (np.ndarray (N, 2)): centerline waypoints
        
    Returns:
        progress (float): Distance along centerline from start (meters)
        nearest_idx (int): Index of nearest centerline segment
    """
    # Find nearest point on centerline
    nearest_p, nearest_dist, t, nearest_idx = nearest_point(point, centerline)
    
    # Calculate cumulative distance up to nearest segment
    progress = 0.0
    for i in range(nearest_idx):
        segment_length = np.linalg.norm(centerline[i+1] - centerline[i])
        progress += segment_length
    
    # Add fractional progress within current segment
    if nearest_idx < len(centerline) - 1:
        segment_vec = centerline[nearest_idx + 1] - centerline[nearest_idx]
        segment_length = np.linalg.norm(segment_vec)
        progress += t * segment_length
    
    return progress, nearest_idx


def random_position(waypoints_xytheta, sampled_number=1, rng=None, xy_noise=0.0, theta_noise=0.0, 
                   ego_idx=100, interval_idx=20):
    """Generate random starting positions along waypoints with optional noise"""
    for i in range(sampled_number):
        starting_idx = (ego_idx + i * interval_idx) % len(waypoints_xytheta)
        x, y, theta = waypoints_xytheta[starting_idx][0], waypoints_xytheta[starting_idx][1], waypoints_xytheta[starting_idx][2]
        x = x + rng.random(size=1)[0] * xy_noise
        y = y + rng.random(size=1)[0] * xy_noise
        theta = zero_2_2pi(theta) + rng.random(size=1)[0] * theta_noise 
        if i == 0:
            res = np.array([[x, y, theta]])
        else:
            res = np.vstack((res, np.array([[x, y, theta]])))
    return res, ego_idx


@njit(cache=True)
def _find_corresponding_waypoint_njit(ego_waypoint, opp_waypoints):
    best_idx = 0
    best_dist = 1e30
    ex = ego_waypoint[0]
    ey = ego_waypoint[1]
    for i in range(opp_waypoints.shape[0]):
        dx = opp_waypoints[i, 0] - ex
        dy = opp_waypoints[i, 1] - ey
        dist = dx * dx + dy * dy
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx

def find_corresponding_waypoint(ego_waypoint, opp_waypoints):
    """Find the waypoint on opponent raceline closest to ego waypoint spatially."""
    return int(_find_corresponding_waypoint_njit(np.ascontiguousarray(ego_waypoint, dtype=np.float64), np.ascontiguousarray(opp_waypoints, dtype=np.float64)))

def load_config(config_path=None):
    """Load lattice planner configuration from YAML file"""
    with open(config_path) as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)
    return Namespace(**config_dict)

def get_map_paths(map_name):
    """Generate map-related paths for a given map name"""
    map_directory = os.path.join('MapZoo', map_name)
    map_path = os.path.join(map_directory, f'{map_name}')
    return map_directory, map_path

# ============================================================================
# Competitive planner shared utilities
# ============================================================================

def _slice_obs_for_pair(obs, ego_idx: int, opp_idx: int):
    """Build a two-agent observation view where pair index 0 is ego_idx and pair index 1 is opp_idx."""
    pair_indices = [int(ego_idx), int(opp_idx)]
    max_idx = max(pair_indices)
    pair_obs = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] > max_idx:
            pair_obs[key] = value[pair_indices].copy()
        elif isinstance(value, list) and len(value) > max_idx:
            pair_obs[key] = [value[k] for k in pair_indices]
        else:
            pair_obs[key] = value
    return pair_obs

@njit(cache=True)
def _compute_raw_progresses_from_arrays_njit(poses_x, poses_y, refline):
    num_agents = poses_x.shape[0]
    progresses = np.zeros(num_agents, dtype=np.float64)
    point = np.empty(2, dtype=np.float64)
    for i in range(num_agents):
        point[0] = poses_x[i]
        point[1] = poses_y[i]
        s, _ = project_point_to_centerline(point, refline)
        progresses[i] = s
    return progresses

def _compute_raw_progresses(obs, refline) -> np.ndarray:
    """Project every agent position to the reference line."""
    return _compute_raw_progresses_from_arrays_njit(
            np.ascontiguousarray(obs["poses_x"], dtype=np.float64), 
            np.ascontiguousarray(obs["poses_y"], dtype=np.float64), 
            np.ascontiguousarray(refline, dtype=np.float64)
            )

@njit(cache=True)
def _nearest_agents_by_ref_projection(raw_progresses, refline_total_length: float) -> np.ndarray:
    """For each agent, choose the closest other agent by circular reference-line projection distance."""
    num_agents = raw_progresses.shape[0]
    nearest = np.empty(num_agents, dtype=np.int64)
    for i in range(num_agents):
        best_j = -1
        best_dist = 1e30
        for j in range(num_agents):
            if i == j:
                continue
            ds = abs(raw_progresses[i] - raw_progresses[j])
            loop_ds = refline_total_length - ds
            if loop_ds < ds:
                ds = loop_ds
            if ds < best_dist:
                best_dist = ds
                best_j = j
        nearest[i] = best_j
    return nearest

@njit(cache=True)
def _update_multi_progresses_arrays_njit(raw_progresses, prev_raw_progresses, lap_counts, refline_total_length):
    n = raw_progresses.shape[0]
    new_lap_counts = lap_counts.copy()
    unwrapped_progresses = np.empty(n, dtype=np.float64)
    half_len = refline_total_length / 2.0
    for i in range(n):
        delta = raw_progresses[i] - prev_raw_progresses[i]
        if delta < -half_len:
            new_lap_counts[i] += 1
        elif delta > half_len:
            new_lap_counts[i] -= 1
        unwrapped_progresses[i] = raw_progresses[i] + new_lap_counts[i] * refline_total_length
    return raw_progresses, unwrapped_progresses, new_lap_counts

def _update_multi_progresses(obs, refline, refline_total_length: float, prev_raw_progresses: np.ndarray, lap_counts: np.ndarray):
    """Return raw progress, unwrapped progress, and updated lap counts for all agents."""
    raw_progresses = _compute_raw_progresses(obs, refline)
    return _update_multi_progresses_arrays_njit(
                raw_progresses, 
                np.ascontiguousarray(prev_raw_progresses, dtype=np.float64), 
                np.ascontiguousarray(lap_counts, dtype=np.int64), 
                float(refline_total_length)
            )
