# Note: Some comments may be outdated. If comments differ from the implementation, debug the call flow or contact the author: Zhouheng Li (https://zhouhengli.github.io/).

import numpy as np
from torch.utils.data import Dataset
from torch.nn import functional as F
import os
from players.planner.flow_planner.data.dataset.utils import (
    RacingDataSample,
    openjson,
    opendata,
)
from bisect import bisect_right
import copy
import torch
import time
        
g = 9.81
ay_lim = 0.6 * g # accept up to 0.6g lateral acceleration
    
class RacingDataset(Dataset):
    def __init__(self, data_dir, data_list, past_neighbor_num, predicted_neighbor_num, future_len, cond_time_len, future_downsampling_method):
        super().__init__()

        self.data_dir = data_dir
        self.data_list_json = openjson(data_list)  # List of npz file names (e.g., length ~2000)

        self._past_neighbor_num = past_neighbor_num 
        self._future_len = future_len
        self._condition_future_neighbor_len = cond_time_len
        self._condition_neighbor_num = past_neighbor_num

        self._wheel_base = 0.307  # use f1tenth vehicle wheel base

        # Build prefix sums for global indexing:
        # prefix[i] = total number of samples in files [0, i)
        self.prefix = [0]

        start_time = time.time()
        for fname in self.data_list_json:
            fpath = os.path.join(self.data_dir, fname)
            # Only read the length metadata to minimize memory usage
            data = opendata(fpath)
            n = int(data["ego_agent_future"].shape[0])

            self.prefix.append(self.prefix[-1] + n)
        print(f"Finished building dataset index in {time.time() - start_time:.2f} seconds.")

        # Total number of samples across all files
        self.total_len = self.prefix[-1]
        print(f"Total dataset length is {self.total_len}")

    def __len__(self):
        # Return the total number of samples across all sub-files
        return self.total_len

    def _locate(self, idx: int):
        """
        Map a global index `idx` to (file_id, local_idx).

        file_id: which npz file
        local_idx: which row within that npz file
        """
        # Support negative indexing
        if idx < 0:
            idx += self.total_len

        if idx < 0 or idx >= self.total_len:
            raise IndexError(f"Index {idx} out of range [0, {self.total_len})")

        # Find the largest file_id such that prefix[file_id] <= idx
        # bisect_right returns insertion position; subtract 1 to get file_id
        file_id = bisect_right(self.prefix, idx) - 1
        local_idx = idx - self.prefix[file_id]
        return file_id, local_idx

    def __getitem__(self, idx) -> RacingDataSample:
        """
        Load only the needed sample from the corresponding npz file.

        This is memory-efficient because we do NOT keep any sub-file fully loaded in RAM.
        """
        
        file_id, local_idx = self._locate(idx)
        fpath = os.path.join(self.data_dir, self.data_list_json[file_id])
        data = opendata(fpath)
        

        # raw shapes
        ego_current_state_raw = data['ego_current_state'][local_idx]                    # (8,)
        ego_agent_future_raw = data['ego_agent_future'][local_idx]                      # (40, 5)
        neighbor_agents_past_raw = data['neighbor_agents_past'][local_idx]              # (4, 25, 5)
        neighbor_agents_future_raw = data['neighbor_agents_future'][local_idx]          # (4, 25, 5)
        racing_line_seg_raw = data['racing_line_seg'][local_idx]                        # (100, 5)

        
        (
            ego_current_state,                                                          # (10,)
            ego_agent_future,                                                           # (40, 5)
            neighbor_agents_past,                                                       # (2, 25, 6)
            neighbor_agents_future,                                                     # (2, 25, 6)
            racing_line_seg,                                                            # (10, 10, 7)
        ) = self.remap_sample(
            ego_current_state_raw=ego_current_state_raw,
            ego_agent_future_raw=ego_agent_future_raw,
            neighbor_agents_past_raw=neighbor_agents_past_raw,
            neighbor_agents_future_raw=neighbor_agents_future_raw,
            racing_line_seg_raw=racing_line_seg_raw,
            past_neighbor_num=self._past_neighbor_num,
            future_len=self._future_len,
            condition_future_neighbor_len=self._condition_future_neighbor_len,
            condition_neighbor_num=self._condition_neighbor_num,
        )
        
        s1 = time.time()
        def to_tensor(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).float()
            return x

        ego_agent_future = ego_agent_future[:, :3] # only x,y,heading
        
        ego_current_state      = to_tensor(ego_current_state)
        ego_agent_future       = to_tensor(ego_agent_future)
        neighbor_agents_past   = to_tensor(neighbor_agents_past)
        neighbor_agents_future = to_tensor(neighbor_agents_future)
        racing_line_seg        = to_tensor(racing_line_seg)

        data = RacingDataSample(
            batched=False,
            ego_current=ego_current_state,
            ego_future=ego_agent_future,
            neighbor_past=neighbor_agents_past,
            neighbor_future=neighbor_agents_future,
            racing_lines=racing_line_seg,
        )
        
        return data

    @staticmethod
    def body_vel_to_world(cos_t, sin_t, vx_body, vy_body):
        vx_world = vx_body * cos_t - vy_body * sin_t
        vy_world = vx_body * sin_t + vy_body * cos_t
        return vx_world, vy_world

    @staticmethod
    def remap_sample(
        ego_current_state_raw: np.ndarray,
        ego_agent_future_raw: np.ndarray,
        neighbor_agents_past_raw: np.ndarray,
        neighbor_agents_future_raw: np.ndarray,
        racing_line_seg_raw: np.ndarray,
        past_neighbor_num: int,
        future_len: int,
        condition_future_neighbor_len: int,
        condition_neighbor_num: int,
    ):
        """
        Remap raw sample into model-ready features.
        neighbor_agents_future_raw and neighbor_agents_past_raw_rank: (A, H, 5) with A agents, H future steps, 5 state features [x, y, theta, vx, vy], but only [x, y, theta] is finally used
        ego_current_state_raw: [x, y, theta, vx, vy, delta, yaw_rate, ax]

        Returns:
            ego_current_state: (10,) [x, y, cos(theta), sin(theta), vx, vy, ax, ay delta, yaw_rate]
            ego_agent_future: (H, 5) [x, y, theta, vx, vy]
            neighbor_agents_past: (N, T, 6)  [x, y, cos(theta), sin(theta), vx, vy]
            neighbor_agents_future: (N, H, 6) [x, y, cos(theta), sin(theta), vx, vy]
            racing_line_seg: (10, 20, 6) [x, y, cos(phi), sin(phi), d_right, d_left]
        """

        # ---- ego current state ----
        # ego_current_state_raw:  [x, y, theta, vx, vy, delta, yaw_rate, ax] from rosbag
        ego_state = RacingDataset.remap_state_kinematic_no_acc(ego_current_state_raw[:5])  # (6,) [x, y, cos(theta), sin(theta), vx, vy]
        # insert ax, ay placeholders at position 6, but actually not used in final model condition, but we keep it for flow augmentation
        # (10,) ego_state final: [x, y, cos(theta), sin(theta), vx, vy, ax, ay, delta, yaw_rate], it's what we need for flow augmentation
        ax = ego_current_state_raw[..., -1]
        ay = np.sqrt(ego_current_state_raw[3]**2 + ego_current_state_raw[4]**2) * ego_current_state_raw[..., -2]  # v * yaw_rate
        if abs(ay) < 0.05:
            ay = 0.0  # filter small ay value
        ay = np.clip(ay, -ay_lim, ay_lim)  # limit unreasonable large value
        ego_state = np.concatenate(
            [
                ego_state[..., :5],                # x, y, cos, sin, vx
                ego_current_state_raw[..., 4:5],   # vy
                ax[..., None],                     # ax
                ay[..., None],                     # ay
                ego_current_state_raw[..., 5:6],   # delta
                ego_current_state_raw[..., -2:-1], # yaw_rate
            ],
            axis=-1
        )
        vx_world, vy_world = RacingDataset.body_vel_to_world(
            ego_state[..., 2], # cos(theta)
            ego_state[..., 3], # sin(theta)
            ego_state[..., 4], # vx_body
            ego_state[..., 5], # vy_body
        )
        ego_state[..., 4] = vx_world
        ego_state[..., 5] = vy_world

        # ---- ego future ----
        # neighbor_agents_future_raw:  [x, y, theta, vx, vy]
        # Do the cos(theta) sin(theta) in function forward, it's original implementation in flow planner, wierd but I keep it
        ego_agent_future = ego_agent_future_raw.copy()[:future_len, :]  # (H, 5) [x, y, theta, vx, vy]

        # ---- neighbor future for condition ----
        neighbor_agents_future = RacingDataset.remap_state_kinematic_no_acc(neighbor_agents_future_raw[:condition_neighbor_num, :condition_future_neighbor_len, :])  # (N, H, 6) [x, y, cos(theta), sin(theta), vx, vy], include current state at index 0
        vx_world, vy_world = RacingDataset.body_vel_to_world(
            neighbor_agents_future[..., 2], # cos(theta)
            neighbor_agents_future[..., 3], # sin(theta)
            neighbor_agents_future[..., 4], # vx_body
            neighbor_agents_future[..., 5], # vy_body
        )
        neighbor_agents_future[..., 4] = vx_world
        neighbor_agents_future[..., 5] = vy_world
        
        # ---- neighbor past ----
        # (T, A, 6), include neighbor current state
        # (6,) feature: [x, y, cos(theta), sin(theta), vx, vy]
        neighbor_agents_past = RacingDataset.remap_state_kinematic_no_acc(neighbor_agents_past_raw[:condition_neighbor_num, :condition_future_neighbor_len, :])  # (N, T, 6) [x, y, cos(theta), sin(theta), vx, vy]
        neighbor_agents_past = neighbor_agents_past[:past_neighbor_num, :, :]
        vx_world, vy_world = RacingDataset.body_vel_to_world(
            neighbor_agents_past[..., 2], # cos(theta)
            neighbor_agents_past[..., 3], # sin(theta)
            neighbor_agents_past[..., 4], # vx_body
            neighbor_agents_past[..., 5], # vy_body
        )
        neighbor_agents_past[..., 4] = vx_world
        neighbor_agents_past[..., 5] = vy_world

        # ---- racing line ----
        racing_line_seg = RacingDataset.convert_racing_line_seg(racing_line_seg_raw)  # (100, 7)
        line_seg = 10
        racing_line_seg = racing_line_seg.reshape(line_seg, int(racing_line_seg.shape[0]/line_seg), racing_line_seg.shape[-1])

        return (
            ego_state.astype(np.float32),
            ego_agent_future.astype(np.float32),
            neighbor_agents_past.astype(np.float32),
            neighbor_agents_future.astype(np.float32),
            racing_line_seg.astype(np.float32),
        )

    @staticmethod
    def convert_racing_line_seg(seg):
        """
        seg: (N, 5) -> [x, y, d_right, d_left, vx, phi]
        return: (N, 6) -> [x, y, cos(phi), sin(phi), d_right, d_left, vx]
        """
        out = np.zeros((seg.shape[0], 7), dtype=seg.dtype)
        out[:, 0:2] = seg[:, 0:2]
        out[:, 2] = np.cos(seg[:, -1])
        out[:, 3] = np.sin(seg[:, -1])
        out[:, 4:6] = seg[:, 2:4]
        out[:, -1] = seg[:, -2]
        return out
    
    @staticmethod
    def remap_state_kinematic_no_acc(state_raw: np.ndarray) -> np.ndarray:
        """
        state_raw: (..., 5) with order:
            [x, y, theta, vx, vy]
        return: (..., 6) with order:
            [x, y, cos(theta), sin(theta), vx, vy]
        """
        if state_raw.shape[-1] != 5:
            raise ValueError(f"Expected last dim = 8, got {state_raw.shape[-1]}")

        out = np.zeros(state_raw.shape[:-1] + (6,), dtype=np.float32)

        out[..., 0:2] = state_raw[..., 0:2]                # x, y
        theta = state_raw[..., 2]
        out[..., 2] = np.cos(theta).astype(np.float32)     # cos(theta)
        out[..., 3] = np.sin(theta).astype(np.float32)     # sin(theta)

        out[..., 4] = state_raw[..., 3]                    # vx
        out[..., 5] = state_raw[..., 4]                    # vy

        return out