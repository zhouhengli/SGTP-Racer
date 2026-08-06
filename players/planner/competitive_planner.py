import contextlib
import os
import time
with open(os.devnull, "w") as devnull:
    with contextlib.redirect_stderr(devnull):
        import gym
import f110_gym
import numpy as np
from typing import List
from tqdm import tqdm
from pathlib import Path
import yaml

import warnings

from players.planner.spliner.spliner_generator import spliner_based_multi_generator
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained since 2022.*")

from players.planner.mpc_planner.mpc_generator import mpc_based_multi_generator
from players.utils.offline_save_video import OfflineRaceVideoRenderer, VehicleState
from players.planner.End2Race.utils import create_planner_render_callback
from players.utils.planner_registry import planners
from players.planner.planner_generators import ocp_based_multi_generator

from players.planner.flow_planner.flow_trajectory_generator import build_flow_context, flow_based_generator
from players.planner.lattice_planner.lattice_generator import lattice_multi_generator
from players.planner.End2Race.end2race_generator import end2race_mppi_generator, compute_direct_ego_mppi_opp_actions

from players.utils.common import (
    _compute_raw_progresses,
    _nearest_agents_by_ref_projection,
    _update_multi_progresses,
    compute_multi_control_actions,
    compute_control_actions,
    initialize_multi_agents_positions,
)
from players.utils.gym_helper import (
    save_data,
    initialize_refline,
    generate_output_filename,
)

# ============================================================================
# Render State
# ============================================================================

RENDER_INFO_TEMPLATE = {
    "ego_steer": 0.0,
    "ego_speed": 0.0,
    "opp_steer": 0.0,
    "opp_speed": 0.0,
    "track_name": None,
}

render_info = RENDER_INFO_TEMPLATE.copy()
draw_grid_pts = []
draw_traj_pts = []
draw_sample_traj_pts = []
draw_select_sample_traj_pts = []

# Kept as globals for render/debug compatibility with create_planner_render_callback.
ego_planner = None
opp_planner = None

def _align_initial_progresses_to_ego(raw_progresses, refline_total_length):
    """Align initial multi-agent progress values to ego with signed circular offsets."""
    raw = np.asarray(raw_progresses, dtype=float)
    current = raw.copy()
    lap_counts = np.zeros(raw.shape[0], dtype=int)
    if raw.shape[0] <= 1:
        return current, lap_counts

    track_len = float(refline_total_length)
    ego_s = float(raw[0])

    for i in range(1, raw.shape[0]):
        signed_delta = (
            (float(raw[i]) - ego_s + 0.5 * track_len) % track_len
        ) - 0.5 * track_len
        current[i] = ego_s + signed_delta
        lap_counts[i] = int(round((current[i] - float(raw[i])) / track_len))

    return current, lap_counts


# ============================================================================
# End2Race Data Collection Helpers
# ============================================================================
# End2Race open-source demonstration.py records ego-only demonstrations at 10 Hz:
#   time, steer, desired_speed, lidar_0, ..., lidar_359
# The final file writing is intentionally left to players.utils.gym_helper.save_data.
END2RACE_EGO_AGENT_IDX = 0
END2RACE_DEFAULT_SAMPLE_INTERVAL = 0.1
END2RACE_DEFAULT_NUM_LIDAR_BEAMS = 360

def _downsample_lidar_for_end2race(scan, target_points=END2RACE_DEFAULT_NUM_LIDAR_BEAMS):
    
    scan = np.asarray(scan, dtype=np.float32).reshape(-1)
    target_points = int(target_points)

    if target_points <= 0 or scan.size == 0:
        raise ValueError("target_points must be positive and scan must not be empty")
    if scan.size == target_points:
        return scan

    # For 1440 -> 360 this is equivalent to evenly selecting one ray per degree.
    indices = np.linspace(0, scan.size - 1, target_points).round().astype(np.int64)
    return scan[indices].astype(np.float32, copy=False)


def _append_end2race_sample( collected_data: List,
    *,
    record_time: float,
    obs,
    action,
    ego_agent_idx: int = END2RACE_EGO_AGENT_IDX,
    target_lidar_points: int = END2RACE_DEFAULT_NUM_LIDAR_BEAMS,
) -> None:
    
    if "scans" not in obs:
        raise KeyError("obs does not contain 'scans'; End2Race data collection needs ego LiDAR")

    lidar_ego = _downsample_lidar_for_end2race(
        obs["scans"][ego_agent_idx],
        target_points=target_lidar_points,
    )

    ego_action = np.asarray(action[ego_agent_idx], dtype=float).reshape(-1)
    steer = float(ego_action[0])
    desired_speed = float(ego_action[1])

    collected_data.append(
        [round(float(record_time), 4), steer, desired_speed]
        + lidar_ego.astype(float).tolist()
    )

# ============================================================================
# Main Simulation Function
# ============================================================================
def run_competitive_players(args, return_metrics=False) -> None:
    global ego_planner, opp_planner


    if "flow" in str(args.method).lower():
        with open("players/config/flow_config.yaml", "r") as f:
            flow_cfg = yaml.safe_load(f)

        for k, v in flow_cfg.items():
            if not hasattr(args, k):
                setattr(args, k, v)

    rng = np.random.default_rng(args.rand_seed)
    steer_limits = (-args.delta_max, args.delta_max)

    # ========================================================================
    # Setup Phase
    # ========================================================================
    try:
        planner_func = planners[args.planner_family]
    except KeyError:
        raise ValueError(f"Unsupported method: {args.planner_family}")

    # Only used to initialize opponent MPPI solvers; it is not the IBR process itself.
    if "lattice" in args.method:
        opp_planner_func = planners["lattice_planner"]
    elif "mpc" in args.method:
        opp_planner_func = planners["mpc"]
    elif "spliner" in args.method:
        opp_planner_func = planners["spliner"]
    else:
        opp_planner_func = planners["mppi"]

    render_info.update({"track_name": args.map_name})

    rollout_trace = []
    planned_traj_records = []


    # --------------------- planner config selection ---------------------
    # Normal run: read planner defaults from config files.
    ego_ocp_conf = None
    opp_ocp_conf = None
    ego_game_conf = None
    opp_game_conf = None
    # --------------------- planner config selection ---------------------


    num_agents = int(getattr(args, "num_agents"))

    agent_planners = []
    config_directory = None

    for agent_idx in range(num_agents):
        if agent_idx == 0:
            raceline_i = args.raceline
            v_scale_i = args.v_global_limit
            ocp_conf_i = ego_ocp_conf
            game_conf_i = ego_game_conf

            # Ego follows args.method.
            setup_fn_i = planner_func[0]
            biased_type_i = args.mppi_bias_mode_ego
        else:
            raceline_i = args.opp_raceline
            v_scale_i = args.opp_speed_scale * args.v_global_limit
            ocp_conf_i = opp_ocp_conf
            game_conf_i = opp_game_conf

            # All opponents are initialized as BiasedMPPIPlanner.
            setup_fn_i = opp_planner_func[0]
            biased_type_i = args.mppi_bias_mode_opp

        planner_i, config_directory_i = setup_fn_i(
            args,
            args.map_name,
            raceline_i,
            config_path=args.config,
            v_scale=v_scale_i,
            ocp_conf=ocp_conf_i,
            game_block_conf=game_conf_i,
            biased_type=biased_type_i,
        )

        if config_directory is None:
            config_directory = config_directory_i

        agent_planners.append(planner_i)

    # Keep old globals for render/debug compatibility.
    ego_planner = agent_planners[0]
    opp_planner = agent_planners[1] if num_agents > 1 else None

    # ========================================================================
    # Environment and Rendering Setup
    # ========================================================================
    env = gym.make(
        "f110-v0",
        map=ego_planner.map_path,
        map_ext=".png",
        timestep=args.dt,
        num_agents=num_agents,
    )

    if args.render:
        def get_ego_planner():
            return ego_planner

        render_callback = create_planner_render_callback(
            render_info,
            get_ego_planner,
            draw_grid_pts,
            draw_traj_pts,
            draw_sample_traj_pts,
            draw_select_sample_traj_pts,
            margin=args.render_margin,
        )
        env.add_render_callback(render_callback)

    # Initialize refline.
    refline, refline_total_length = initialize_refline(
        config_directory,
        args.map_name,
        refline=args.raceline,
    )

    # Initialize agent positions.
    agent_poses, opp_idx = initialize_multi_agents_positions(
        agent_planners,
        args.ego_idx,
        args.interval_idx,
        rng,
    )
    if num_agents == 1:
        opp_idx = 0

    # Reset environment.
    obs, _, done, _ = env.reset(poses=agent_poses)

    # Render initial state. This only works for gym-based rendering,
    # not for OfflineRaceVideoRenderer.
    if args.render:
        env.render()

    video_renderer = None
    map_yaml_path = f"{ego_planner.map_path}.yaml"
    if args.save_video:
        raceline_xy = np.stack(
            [ego_planner.waypoints[:, 0], ego_planner.waypoints[:, 1]],
            axis=1,
        )
        video_renderer = OfflineRaceVideoRenderer(
            map_yaml_path=map_yaml_path,
            refline_xy=raceline_xy,
            boundary_offsets=ego_planner.d_right_left,
            output_fps=args.out_fps,
            vehicle_length=args.length,
            vehicle_width=args.width,
            vehicle_ref_offset=0.0,
            hud_speed_max=args.v_max*args.v_global_limit,
            steer_limit=args.delta_max,
        )

    # ========================================================================
    # State Tracking Initialization
    # ========================================================================
    initial_raw_progresses = _compute_raw_progresses(obs, refline)  # shape: [N_agents]
    current_progresses, lap_counts = _align_initial_progresses_to_ego(
        initial_raw_progresses,
        refline_total_length,
    )
    prev_raw_progresses = initial_raw_progresses.copy()

    current_ego_progress = float(current_progresses[0])

    initial_state = "multi_agent_tracking" if num_agents > 1 else "single_agent_tracking"
    current_state = initial_state
    final_state = initial_state

    num_ibr_iters = args.ibr_time

    # ========================================================================
    # Simulation Loop Setup
    # ========================================================================
    laptime = 0.0
    collected_data: List = []

    # End2Race-compatible demonstration collection.
    # Default is off so normal evaluation remains unchanged unless explicitly enabled.
    collect_end2race_data = args.end2race
    end2race_sample_interval = END2RACE_DEFAULT_SAMPLE_INTERVAL
    end2race_num_lidar_beams = END2RACE_DEFAULT_NUM_LIDAR_BEAMS
    next_end2race_record_time = end2race_sample_interval

    video_frames: List = []
    collision_occurred = False
    tracker_steps = (
        ego_planner.conf.tracker_steps
        if hasattr(ego_planner, "conf") and hasattr(ego_planner.conf, "tracker_steps")
        else ego_planner.tracker_steps
    )
    video_step_count = 0

    pbar = tqdm(
        total=args.sim_duration,
        desc="Simulation",
        unit="s",
        dynamic_ncols=True,
        bar_format=(
            "{l_bar}{bar}| "
            "{n:.2f}/{total:.1f} "
            "[{elapsed}<{remaining}, {rate_fmt}] "
            "{postfix}"
        ),
        mininterval=5.0,
        miniters=1,
        disable=args.end2race or args.collect_pairwise_rows, 
    ) 

    flow_context = None
    if "flow" in str(args.method):
        flow_context = build_flow_context(args)

    while not done and laptime < args.sim_duration:
        # ====================================================================
        # Planning Phase (executed once per tracker cycle)
        # ====================================================================
        raw_progresses = _compute_raw_progresses(obs, refline)
        if num_agents > 1:
            nearest_agent_indices = _nearest_agents_by_ref_projection(
                raw_progresses,
                refline_total_length,
            )
        else:
            nearest_agent_indices = np.zeros(1, dtype=int)

        params_dict = {
            "num_ibr_iters": int(num_ibr_iters),
            "agent_progresses": current_progresses.copy(), # with lap counting and wraparound handled, shape: [N_agents]
            "agent_raw_progresses": raw_progresses.copy(),
            "nearest_agent_indices": nearest_agent_indices.copy(),
            "agent_planners": agent_planners,
            "ego_agent_idx": 0,
        }

        if flow_context is not None:
            params_dict.update(flow_context)

        generator = planner_func[1]

        if generator is lattice_multi_generator \
            or generator is ocp_based_multi_generator \
            or generator is mpc_based_multi_generator \
            or generator is spliner_based_multi_generator \
            or generator is end2race_mppi_generator:
            
            best_trajs, infer_time = generator(
                args=args,
                agent_planners=agent_planners,
                obs=obs,
                params_dict=params_dict,
            )

        elif generator is flow_based_generator:
            # print(f"[INFO] Running flow-based generator...")
            ego_best_traj, opponent_trajs_by_agent_idx, infer_time = generator(
                args=args,
                ego_planner=ego_planner,
                opp_planner=opp_planner,
                obs=obs,
                params_dict=params_dict,
            )

            best_trajs = [None for _ in agent_planners]
            best_trajs[0] = ego_best_traj

            for agent_idx, traj in opponent_trajs_by_agent_idx.items():
                best_trajs[int(agent_idx)] = traj

        else:
            raise ValueError(f"Unsupported planner generator: {generator}")

        del infer_time

        ego_record_traj = None if best_trajs[0] is None else np.asarray(best_trajs[0], dtype=np.float32)
        if ego_record_traj is not None and ego_record_traj.ndim == 2 and ego_record_traj.shape[1] > 2:
            ego_record_traj = ego_record_traj[:, :2]
        planned_traj_records.append({
            "trace_start_index": int(len(rollout_trace)),
            "trajs": [None if ego_record_traj is None else ego_record_traj.copy()],
        })

        ego_best_traj = best_trajs[0]
        opp_best_traj = best_trajs[1] if num_agents > 1 else None

        # ====================================================================
        # Tracking Phase (multiple steps following the planned trajectory)
        # ====================================================================
        for tracker_count in range(tracker_steps):
            if done or laptime >= args.sim_duration:
                break
            
            if getattr(ego_planner, "direct_action", False):
                action = compute_direct_ego_mppi_opp_actions(obs, agent_planners, best_trajs, steer_limits)
            elif int(num_agents) == 1: # for only 1 agent
                steer, speed = ego_planner.tracker.plan(
                    obs["poses_x"][0],
                    obs["poses_y"][0],
                    obs["poses_theta"][0],
                    obs["linear_vels_x"][0],
                    ego_best_traj,
                )
                steer = np.clip(steer, steer_limits[0], steer_limits[1])
                action = np.asarray([[steer, speed]], dtype=float)
            elif int(num_agents) > 2: # for > 2 agents
                action = compute_multi_control_actions(
                    obs=obs,
                    agent_planners=agent_planners,
                    best_trajs=best_trajs,
                    nearest_agent_indices=nearest_agent_indices,
                    steer_limits=steer_limits,
                )
            else: # for 2 agents
                action = compute_control_actions(
                    obs,
                    ego_planner,
                    opp_planner,
                    ego_best_traj,
                    opp_best_traj,
                    steer_limits=steer_limits,
                )

            # Update render information.
            if args.render or args.save_video:
                render_info.update({
                    "ego_steer": action[0, 0],
                    "ego_speed": action[0, 1],
                    "opp_steer": action[1, 0] if num_agents > 1 else 0.0,
                    "opp_speed": action[1, 1] if num_agents > 1 else 0.0,
                })

            # Step environment.
            obs, timestep, done, _ = env.step(action)

            video_step_count += 1
            if (
                args.save_video
                and video_renderer is not None
                and video_step_count % args.video_capture_every == 0
            ):
                num_render_agents = len(obs["poses_x"])

                vehicle_states = [
                    VehicleState(
                        x=float(obs["poses_x"][i]),
                        y=float(obs["poses_y"][i]),
                        heading=float(obs["poses_theta"][i]),
                    )
                    for i in range(num_render_agents)
                ]

                video_renderer.capture_multi(
                    vehicle_states=vehicle_states,
                    sim_time=laptime,
                    title="Closed-loop Racing",
                    vehicle_trajs=best_trajs,
                    extra_text=None,
                    render_info=render_info.copy(),
                    follow_vehicle_idx=0,
                )

            # Update simulation time.
            prev_laptime = laptime
            laptime += timestep
            if laptime > args.sim_duration:
                laptime = args.sim_duration

            pbar.update(laptime - prev_laptime)

            # End2Race demonstration collection.
            if collect_end2race_data:
                # here while is used to strictly make sure the time stamps of recorded samples align with the specified interval, even if there are occasional longer delays in the loop due to environment stepping or planning.
                while laptime >= next_end2race_record_time:
                    _append_end2race_sample(
                        collected_data,
                        record_time=next_end2race_record_time,
                        obs=obs,
                        action=action,
                        ego_agent_idx=END2RACE_EGO_AGENT_IDX,
                        target_lidar_points=end2race_num_lidar_beams,
                    )
                    next_end2race_record_time += end2race_sample_interval

            # Handle wraparound. _update_multi_progresses updates lap counting
            # and wraparound internally, so it is the single source of progress.
            raw_progresses, current_progresses, lap_counts = _update_multi_progresses(
                obs=obs,
                refline=refline,
                refline_total_length=refline_total_length,
                prev_raw_progresses=prev_raw_progresses,
                lap_counts=lap_counts,
            )
            prev_raw_progresses = raw_progresses.copy()

            # Compatibility variable for existing debug/render output.
            current_ego_progress = float(current_progresses[0])

            rollout_trace.append({
                "t": float(laptime),
                "poses_x": np.asarray(obs["poses_x"], dtype=float).tolist(),
                "poses_y": np.asarray(obs["poses_y"], dtype=float).tolist(),
                "poses_theta": np.asarray(obs["poses_theta"], dtype=float).tolist(),
                "action": np.asarray(action, dtype=float).tolist(),
            })
            
            # Keep final_state independent from any selected opponent.
            current_state = "multi_agent_tracking" if num_agents > 1 else "single_agent_tracking"
            final_state = current_state

            # Collision detection.
            collisions = np.asarray(obs["collisions"], dtype=bool).ravel()
            collision_detected = bool(collisions[0])
            collision_msg = "[WARNING] Collision detected for ego!"

            if collision_detected:
                done = True
                collision_occurred = True
                print(collision_msg)

            # Update progress bar postfix info.
            pbar.set_postfix({
                "state": current_state,
                "ego_s": f"{current_ego_progress:.2f}",
                "opp_s": f"{current_progresses[nearest_agent_indices[0]]:.2f}" if num_agents > 1 else "N/A",
                "ego_v": f"{obs['linear_vels_x'][0]:.2f}",
            })

            # Rendering.
            if args.render:
                frame = env.render(mode="rgb_array")
                if collect_end2race_data and args.save_gym:
                    video_frames.append(frame)

    pbar.close()

    # ========================================================================
    # Post-Simulation Processing
    # ========================================================================
    print("=" * 80, flush=True)
    print("[ARGS]", flush=True)
    for key, value in sorted(vars(args).items()):
        print(f"{key}: {value}", flush=True)
    print("=" * 80, flush=True)

    refline_xy = np.asarray(ego_planner.waypoints, dtype=np.float32)
    boundary_offsets = np.asarray(ego_planner.d_right_left, dtype=np.float32)
    metrics = {
        "rollout_trace": rollout_trace,
        "planned_trajs": planned_traj_records,
        "refline_xy": refline_xy,
        "boundary_offsets": boundary_offsets,
    }

    output_filename = ""
    if args.end2race or args.save_video or not return_metrics:
        output_filename = generate_output_filename(
            args,
            final_state,
            args.opp_raceline,
            args.ego_idx,
            opp_idx,
            args.opp_speed_scale,
        )

    if args.end2race:
        save_data(
            args,
            collected_data,
            video_frames,
            collision_occurred,
            final_state,
            output_filename,
            laptime,
            opp_idx,
        )

    if args.save_video and video_renderer is not None:
        video_path = os.path.join("results", f"{output_filename}.mp4")
        video_renderer.save(video_path)
        video_renderer.close()

    if return_metrics:
        return metrics

    print(f"[INFO] Results saved with filename: {output_filename}")


if __name__ == "__main__":
    # This would be called from trainer/main script with args.
    pass
