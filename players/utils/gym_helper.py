import os
import csv
import json
from datetime import datetime
import imageio
from typing import Tuple, Dict, List, Optional
from players.utils.common import *

def generate_output_filename(
    args,
    final_state: str,
    opp_raceline: str,
    ego_idx: int,
    opp_idx: int,
    opp_speed_scale: float
) -> str:
    state_prefix = "o" if final_state == "overtaking" else "f"
    opp_raceline_num = opp_raceline.replace('raceline', '').replace('.csv', '')
    
    return f"{args.method}_{state_prefix}_ol{opp_raceline_num}_e{ego_idx}_o{opp_idx}_s{opp_speed_scale}"




def detect_state_transition(
    ego_progress: float,
    opp_progress: float
) -> str:
    return "overtaking" if ego_progress > opp_progress else "following"


def initialize_refline(config_directory: str, map_name: str, refline: str) -> Tuple[np.ndarray, float]:
    """
    Load and initialize refline waypoints and compute total length.
    
    Args:
        config_directory: Directory containing refline file
        map_name: Name of the map
    
    Returns:
        Tuple of (centerline_points, total_length)
    """
    refline_path = os.path.join(
        config_directory,
        f"{map_name}_{refline}.csv"
    )
    refline_wp = np.loadtxt(refline_path, delimiter=';', skiprows=1)
    refline = np.vstack((refline_wp[:, 3], refline_wp[:, 4])).T
    
    # Compute total refline length
    total_length = 0.0
    for i in range(len(refline) - 1):
        total_length += np.linalg.norm(refline[i + 1] - refline[i])
    
    return refline, total_length


def save_data(args, collected_data, video_frames, collision_occurred, 
              final_state, base_filename, laptime, opp_idx):
    """Save data with unified format"""
    dir_timestamp = datetime.now().strftime("%Y%m%d")
    dataset_dir = f"results/end2race/dataset_{dir_timestamp}"
    os.makedirs(dataset_dir, exist_ok=True)
    
    if collision_occurred:
        collision_dir = os.path.join(dataset_dir, "collision")
        os.makedirs(collision_dir, exist_ok=True)
        
        # Multi-agent collision metadata
        collision_metadata = {
            'mode': 'multi_agent',
            'ego_raceline': str(args.raceline),
            'ego_idx': int(args.ego_idx),
            'opp_raceline': str(args.opp_raceline),
            'opp_idx': int(opp_idx),
            'speed_scale': float(args.opp_speed_scale),
            'interval_idx': int(args.interval_idx),
            'simulation_time': float(laptime),
            'final_state': str(final_state)
        }
        
        metadata_path = os.path.join(collision_dir, f"{base_filename}.json")
        with open(metadata_path, 'w') as f:
            json.dump(collision_metadata, f, indent=2)
        
        if args.render and video_frames and args.save_gym:
            video_filename = os.path.join(collision_dir, f"{base_filename}.mp4")
            imageio.mimwrite(video_filename, video_frames, fps=100, macro_block_size=1)
            print(f"Collision video saved to {video_filename}")
        
        print(f"Collision metadata saved to {metadata_path}")
    else:
        success_dir = os.path.join(dataset_dir, "success")
        os.makedirs(success_dir, exist_ok=True)
        
        csv_filename = os.path.join(success_dir, f"{base_filename}.csv")
        
        # Modified header
        header = ["time", "steer", "desired_speed"] + [f"lidar_{i}" for i in range(360)]
        
        with open(csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(collected_data)
        
        print(f"Multi-agent data saved to {csv_filename}")
        
        if args.render and video_frames and args.save_gym:
            video_filename = os.path.join(success_dir, f"{base_filename}.mp4")
            imageio.mimwrite(video_filename, video_frames, fps=100, macro_block_size=1)
            print(f"Video saved to {video_filename}")
