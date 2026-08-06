"""Wrap the current End2Race model as a direct-action planner."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from players.planner.End2Race.model import End2Race


class End2RacePlanner:
    """Use the current End2Race model to infer ego steering and speed from LiDAR and previous speed."""

    direct_action = True

    def __init__(self, args, geometry_planner, model_path: str) -> None:
        """Initialize the policy model and copy geometry fields required by the simulator."""
        self.geometry_planner = geometry_planner
        self.map_path = geometry_planner.map_path
        self.waypoints = geometry_planner.waypoints
        self.d_right_left = geometry_planner.d_right_left
        self.tracker_steps = 1
        self.num_lidar = 360
        self.noise = 0.0
        self.delta_max = float(getattr(args, "delta_max"))
        self.v_max = float(getattr(args, "v_max"))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = End2Race(mask_prob=0.1, hidden_scale=4).to(self.device)
        self._load_model(model_path)
        self.model.eval()
        self.hidden_state: Optional[torch.Tensor] = None
        self.prev_speed: Optional[float] = None
        self.last_action = np.zeros(2, dtype=float)

    def _load_model(self, model_path: str) -> None:
        """Load model weights saved by the original End2Race training script."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"End2Race model not found: {path}")
        state = torch.load(path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=False)

    def reset(self, initial_speed: Optional[float] = None) -> None:
        """Reset recurrent state before a new episode."""
        self.hidden_state = torch.zeros((1, 1, self.model.gru.hidden_size), device=self.device)
        self.prev_speed = None if initial_speed is None else float(initial_speed)
        self.last_action[:] = 0.0

    def _scan360(self, scan: np.ndarray) -> np.ndarray:
        """Convert simulator LiDAR into the 360-beam format used by End2Race."""
        scan = np.asarray(scan, dtype=np.float32).reshape(-1)
        if scan.size == self.num_lidar:
            return scan
        idx = np.linspace(0, scan.size - 1, self.num_lidar).round().astype(np.int64)
        return scan[idx].astype(np.float32, copy=False)

    def _apply_noise(self, scan: np.ndarray) -> np.ndarray:
        """Mask a fixed fraction of LiDAR beams with zeros."""
        if self.noise <= 0.0:
            return scan
        out = scan.copy()
        n = int(out.size * self.noise)
        if n > 0:
            out[np.random.choice(out.size, min(n, out.size), replace=False)] = 0.0
        return out

    def policy_plan(self, obs, agent_idx: int = 0) -> np.ndarray:
        """Run one End2Race inference step and return [steer, speed]."""
        if self.hidden_state is None:
            self.reset(initial_speed=float(obs["linear_vels_x"][agent_idx]))
        scan = self._apply_noise(self._scan360(obs["scans"][agent_idx]))
        prev_speed = float(obs["linear_vels_x"][agent_idx]) if self.prev_speed is None else float(self.prev_speed)
        start = torch.cuda.Event(enable_timing=True) if self.device == "cuda" else None
        end = torch.cuda.Event(enable_timing=True) if self.device == "cuda" else None
        if start is not None:
            start.record()
        with torch.no_grad():
            lidar_tensor = torch.tensor(scan, dtype=torch.float32, device=self.device).view(1, 1, -1)
            speed_tensor = torch.tensor([[[prev_speed]]], dtype=torch.float32, device=self.device)
            action_seq, self.hidden_state = self.model(lidar_tensor, speed_tensor, self.hidden_state)
            action = action_seq[:, -1, :].detach().cpu().numpy()[0].astype(float)
        if end is not None:
            end.record()
            torch.cuda.synchronize()
        action[0] = float(np.clip(action[0], -self.delta_max, self.delta_max))
        action[1] = float(np.clip(action[1], 0.0, self.v_max))
        self.prev_speed = float(obs["linear_vels_x"][agent_idx])
        self.last_action = action
        return action
