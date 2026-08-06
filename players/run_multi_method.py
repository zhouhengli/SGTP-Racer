import argparse
import time
import numpy as np
import yaml
import warnings

from players.planner.competitive_planner import run_competitive_players

warnings.filterwarnings("ignore")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Multi-Agent Planner Runner")
    parser.add_argument("--map_name", type=str, default="Nuerburgring") # default map name, all maps are in `config/data_split_final.yaml`

    parser.add_argument("--mppi_bias_mode_ego", type=str, default="none", choices=["none", "ancillary"]) # for biased-mppi
    parser.add_argument("--mppi_bias_mode_opp", type=str, default="none", choices=["none", "ancillary"],) # for biased-mppi

    parser.add_argument("--ego_idx", type=int, default=1100) # Start index of the ego vehicle
    parser.add_argument("--num_agents", type=int, default=3) # Number of agents in the simulation
    parser.add_argument("--sim_duration", type=float, default=1.0) # Duration of the simulation in seconds
    parser.add_argument("--save_video", type=bool, default=True) # Whether to save the simulation video
    parser.add_argument("--interval_idx", type=int, default=20) # Interval index for vehicles gap

    return parser.parse_args()


def load_yaml_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def merge_args_with_config(args, cfg):
    args.eval_case_id = cfg.get("eval_case_id")
    args.collect_pairwise_rows = cfg.get("collect_pairwise_rows", False)
    args.interaction_mode = cfg.get("interaction_mode")
    args.opp_speed_scale = cfg.get("opp_speed_scale")
    args.render = cfg.get("render", False)
    args.tracker_steps = cfg["tracker_steps"]
    args.save_gym = cfg["save_gym"]
    args.out_fps = cfg["out_fps"]
    args.dt = cfg["dt"]
    args.v_global_limit = cfg["v_global_limit"]
    args.end2race = cfg["end2race"]
    args.rand_seed = cfg["rand_seed"]
    args.render_margin = cfg["render_margin"]
    args.video_capture_every = cfg["video_capture_every"]

    args.wheel_base = cfg["wheel_base"]
    args.length = cfg["length"]
    args.width = cfg["width"]
    args.planner_family = cfg["planner_family"]

    args.v_max = cfg["v_max"]
    args.a_max = cfg["a_max"]
    args.delta_max = cfg["delta_max"]
    args.config = cfg["config"]
    args.raceline = cfg["raceline"]
    args.opp_raceline = cfg["raceline"]
    args.ibr_time = cfg["ibr_time"]
    args.ocp_config = cfg["ocp_config"]
    args.game_config = cfg["game_config"]

    return args


def main():
    args = parse_arguments()
    cfg = load_yaml_config("players/config/run_config.yaml")
    args = merge_args_with_config(args, cfg)
    args.method = f"{args.planner_family}_{args.interaction_mode}"

    if args.end2race:
        args.save_video = False

    start_time = time.perf_counter()
    metrics = run_competitive_players(args, return_metrics=True)
    wall_time = time.perf_counter() - start_time
    print(f"[INFO] Run completed in {wall_time:.2f} seconds")


if __name__ == "__main__":
    main()
