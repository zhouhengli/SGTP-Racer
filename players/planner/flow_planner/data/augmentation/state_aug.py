import torch
from typing import List, Union

import numpy as np
import torch
from players.planner.flow_planner.data.dataset.dataloader_racing_ros import RacingDataSample

NUM_REFINE = 10
REFINE_HORIZON = 0.5
TIME_INTERVAL = 0.05

def vector_transform(vector, transform_mat, bias=None):
    """
    vector: (B, ..., 2)
    transform_mat: (B, 2, 2)
    bias: (B, ..., 2)
    """
    shape = vector.shape
    B = vector.shape[0]
    nexpand = vector.ndim - 2
    if bias is not None:
        vector = vector - bias.reshape(B, *([1] * nexpand), -1)
    vector = vector.reshape(B, -1, 2).permute(0, 2, 1) # (B, 2, N1 * N2 ...)
    return torch.bmm(transform_mat, vector).permute(0, 2, 1).reshape(*shape) # (B, ..., 2)

def heading_transform(heading, transform_mat):
    """
    heading: (B, ...)
    transform_mat: (B, 2, 2)
    """
    B = heading.shape[0]
    shape = heading.shape
    nexpand = heading.ndim - 1
    heading = heading.reshape(B, -1)
    transform_mat = transform_mat.reshape(B, 1, 2, 2)
    return torch.atan2(
        torch.cos(heading) * transform_mat[..., 1, 0] + torch.sin(heading) * transform_mat[..., 1, 1],
        torch.cos(heading) * transform_mat[..., 0, 0] + torch.sin(heading) * transform_mat[..., 0, 1]
    ).reshape(*shape)

class StatePerturbation():
    """
    Data augmentation that perturbs the current ego position and generates a feasible trajectory history that satisfies a set of kinematic constraints.    
    """

    def __init__(
        self,
        # device,
        augment_prob,
        low: List[float] = [-0., -0.75, -0.2, -1, -0.5, -0.2, -0.1, 0., -0.],
        high: List[float] = [0., 0.75, 0.2, 1, 0.5, 0.2, 0.1, 0., 0.],
        # augment_prob: float = 0.5,
        normalize=True,
    ) -> None:
        """
        Initialize the augmentor,
        state: [x, y, yaw, vx, vy, ax, ay, steering angle, yaw rate],
        :param dt: Time interval between trajectory points.
        :param low: Parameter to set lower bound vector of the Uniform noise on [x, y, yaw]. Used only if use_uniform_noise == True.
        :param high: Parameter to set upper bound vector of the Uniform noise on [x, y, yaw]. Used only if use_uniform_noise == True.
        :param augment_prob: probability between 0 and 1 of applying the data augmentation
        :param use_uniform_noise: Parameter to decide to use uniform noise instead of gaussian noise if true.S
        """
        self._augment_prob = augment_prob
        self._normalize = normalize
        self._low = torch.tensor(low)
        self._high = torch.tensor(high)
        self._wheel_base = 0.307  # use f1tenth vehicle wheel base
        
        self.refine_horizon = REFINE_HORIZON
        self.num_refine = NUM_REFINE
        self.time_interval = TIME_INTERVAL
        
        T = REFINE_HORIZON
        self.coeff_matrix = torch.linalg.inv(torch.tensor([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [1, T, T**2, T**3, T**4, T**5],
            [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
            [0, 0, 2, 6*T, 12*T**2, 20*T**3]
        ], dtype=torch.float32))
        self.t_matrix = torch.pow(
            torch.linspace(TIME_INTERVAL, REFINE_HORIZON, NUM_REFINE).unsqueeze(1), 
            torch.arange(6).unsqueeze(0)
        )  # shape (B, N+1)
        
    def __call__(self, data) -> RacingDataSample:
        aug_flag, aug_ego_current_state = self.augment(data)
        refine_ego_future = self.refine_future_trajectory(aug_ego_current_state, data.ego_future) # [4, P+1:, 3=(x,y,heading)]

        # use safe augment
        use_aug_flag = aug_flag
        data.ego_current[use_aug_flag] = aug_ego_current_state[use_aug_flag]
        data.ego_future[..., 1:, :][use_aug_flag] = refine_ego_future[use_aug_flag]

        return self.centric_transform(data).to(torch.float32)
    
    def move_to(self, device):
        self._low = self._low.to(device)
        self._high = self._high.to(device)
        self.coeff_matrix = self.coeff_matrix.to(device)
        self.t_matrix = self.t_matrix.to(device)
        
    def augment(
        self,
        data
    ):
        # Only aug current state
        self._device = data.ego_current.device
        self.move_to(self._device)
        
        ego_current_state = data.ego_current.clone()

        B = ego_current_state.shape[0]
        aug_flag = (torch.rand(B) <= self._augment_prob).bool().to(self._device) & ~(abs(ego_current_state[:, 4]) < 2.0)

        random_tensor = torch.rand(B, len(self._low)).to(self._device)
        scaled_random_tensor = self._low + (self._high - self._low) * random_tensor

        new_state = torch.zeros((B, 9), dtype=torch.float32).to(self._device)
        new_state[:, 3:] = ego_current_state[:, 4:10] # x, y, h is 0 because of ego-centric, update vx, vy, ax, ay, steering angle, yaw rate
        new_state = new_state + scaled_random_tensor
        new_state[:, 3] = torch.max(new_state[:, 3], torch.tensor(0.0, device=new_state.device))
        new_state[:, -1] = torch.clip(new_state[:, -1], -0.85, 0.85)


        ego_current_state[:, :2] = new_state[:, :2]
        ego_current_state[:, 2] = torch.cos(new_state[:, 2])
        ego_current_state[:, 3] = torch.sin(new_state[:, 2])
        ego_current_state[:, 4:8] = new_state[:, 3:7]
        ego_current_state[:, 8:10] = new_state[:, -2:] # steering angle, yaw rate

        # update steering angle and yaw rate
        cur_velocity = ego_current_state[:, 4]
        yaw_rate = ego_current_state[:, 9] 

        steering_angle = torch.zeros_like(cur_velocity)
        new_yaw_rate = torch.zeros_like(yaw_rate)

        mask = torch.abs(cur_velocity) < 0.2
        not_mask = ~mask
        steering_angle[not_mask] = torch.atan(yaw_rate[not_mask] * self._wheel_base / torch.abs(cur_velocity[not_mask]))
        steering_angle[not_mask] = torch.clamp(steering_angle[not_mask], -2 / 3 * np.pi, 2 / 3 * np.pi)
        new_yaw_rate[not_mask] = yaw_rate[not_mask]


        ego_current_state[:, 8] = steering_angle
        ego_current_state[:, 9] = new_yaw_rate

        return aug_flag, ego_current_state


    def normalize_angle(self, angle: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return (angle + np.pi) % (2 * np.pi) - np.pi
    

    def get_transform_matrix_batch(self, cur_state):
        processed_input = torch.column_stack(
            (
                cur_state[:, 2],  # cos
                cur_state[:, 3],  # sin
            )
        )

        reshaping_tensor = torch.tensor(
            [
                [1, 0, 0, 1],
                [0, 1, -1, 0],
            ], dtype=torch.float32
        ).to(processed_input.device)
        return (processed_input @ reshaping_tensor).reshape(-1, 2, 2)
        
    def centric_transform(
        self,
        data: RacingDataSample,
    ):
        # In this function and flow planer project, all the vx and vy means world frame
        # the data from ros-mpc, vx vy means ego frame. So we need to transfrom them into world frame first. It's done in players.planner/flow_planner/data/dataset/dataloader_racing_ros.py
        cur_state = data.ego_current.clone()
        center_xy = cur_state[:, :2]
        # world frame to ego frame
        transform_matrix = self.get_transform_matrix_batch(cur_state)

        # ego xy
        data.ego_current[..., :2] = vector_transform(data.ego_current[..., :2], transform_matrix, center_xy)
        # ego cos sin
        data.ego_current[..., 2:4] = vector_transform(data.ego_current[..., 2:4], transform_matrix)
        # ego vx, vy
        data.ego_current[..., 4:6] = vector_transform(data.ego_current[..., 4:6], transform_matrix)
        # ego ax, ay, not used
        data.ego_current[..., 6:8] = vector_transform(data.ego_current[..., 6:8], transform_matrix)
        numerical_type = data.ego_current.dtype
        
        # ego future xy
        data.ego_future[..., :, :2] = vector_transform(data.ego_future[..., :, :2].to(numerical_type), transform_matrix, center_xy)
        # ego future heading
        data.ego_future[..., :, 2] = heading_transform(data.ego_future[..., :, 2].to(numerical_type), transform_matrix)

        # neighbor past xy, [x, y, cos(theta), sin(theta), vx, vy]
        x = data.neighbor_past[..., :6].to(numerical_type)
        mask = (
            torch.all(x[..., :2] == 0, dim=-1) &
            torch.all(x[..., 3:] == 0, dim=-1) &
            (x[..., 2] == 1) # during dataset generation, [x=0,y=0,heading=0] -> [x=0,y=0,cos=1,sin=0], it's also invalid data
        )
        data.neighbor_past[..., :2] = vector_transform(data.neighbor_past[..., :2].to(numerical_type), transform_matrix, center_xy)
        # neighbor past cos sin
        data.neighbor_past[..., 2:4] = vector_transform(data.neighbor_past[..., 2:4].to(numerical_type), transform_matrix)
        # neighbor past vx, vy
        data.neighbor_past[..., 4:6] = vector_transform(data.neighbor_past[..., 4:6].to(numerical_type), transform_matrix)
        data.neighbor_past[mask] = 0.

        # neighbor future xy, [x, y, cos(theta), sin(theta), vx, vy]
        x = data.neighbor_future[..., :6].to(numerical_type)
        mask = (
            torch.all(x[..., :2] == 0, dim=-1) &
            torch.all(x[..., 3:] == 0, dim=-1) &
            (x[..., 2] == 1) # during dataset generation, [x=0,y=0,heading=0] -> [x=0,y=0,cos=1,sin=0], it's also invalid data
        )
        data.neighbor_future[..., :2] = vector_transform(data.neighbor_future[..., :2].to(numerical_type), transform_matrix, center_xy)
        # neighbor future cos sin
        data.neighbor_future[..., 2:4] = vector_transform(data.neighbor_future[..., 2:4].to(numerical_type), transform_matrix)
        # neighbor future vx, vy
        data.neighbor_future[..., 4:6] = vector_transform(data.neighbor_future[..., 4:6].to(numerical_type), transform_matrix)
        data.neighbor_future[mask] = 0.

        # racing_lines
        mask = torch.sum(torch.ne(data.racing_lines[..., :6], 0), dim=-1) == 0 # actually, it wouldn't be triggered. We always have racing lines
        data.racing_lines[..., :2] = vector_transform(data.racing_lines[..., :2], transform_matrix, center_xy)
        data.racing_lines[..., 2:4] = vector_transform(data.racing_lines[..., 2:4], transform_matrix)
        data.racing_lines[mask] = 0.

        return data
    
    def refine_future_trajectory(self, aug_current_state, ego_future):
        """
        refine future trajectory with quintic spline interpolation
        This function implicitly assumes that ego_future[0] corresponds to the current state, and the actual future trajectory starts from index 1.
        
        Args:
            aug_current_state: (B, 16) current state of the ego vehicle after augmentation
            ego_future:        (B, 80, 3) future trajectory of the ego vehicle
            
        Returns:
            ego_future: refined future trajectory of the ego vehicle with velocity and acceleration
        """
        
        P = self.num_refine
        dt = self.time_interval
        B = aug_current_state.shape[0]
        M_t = self.t_matrix.unsqueeze(0).expand(B, -1, -1)
        A = self.coeff_matrix.unsqueeze(0).expand(B, -1, -1)
        
        # state: [x, y, heading, velocity, acceleration, yaw_rate]
        x0, y0, theta0, v0, a0, omega0 = (
            aug_current_state[:, 0], 
            aug_current_state[:, 1], 
            torch.atan2(
                (ego_future[:, int(P/2), 1] - aug_current_state[:, 1]), 
                (ego_future[:, int(P/2), 0] - aug_current_state[:, 0])
            ), 
            torch.norm(aug_current_state[:, 4:6], dim=-1), 
            torch.norm(aug_current_state[:, 6:8], dim=-1), 
            aug_current_state[:, 9]
        )
        
        xT, yT, thetaT, vT, aT, omegaT = (
            ego_future[:, P, 0],
            ego_future[:, P, 1],
            ego_future[:, P, 2],
            torch.norm(ego_future[:, P, :2] - ego_future[:, P - 1, :2], dim=-1) / dt,
            torch.norm(ego_future[:, P, :2] - 2 * ego_future[:, P - 1, :2] + ego_future[:, P - 2, :2], dim=-1) / dt**2,
            self.normalize_angle(ego_future[:, P, 2] - ego_future[:, P - 1, 2]) / dt
        )

        # Boundary conditions
        sx = torch.stack([
            x0, 
            v0*torch.cos(theta0), 
            a0*torch.cos(theta0) - v0*torch.sin(theta0)*omega0, 
            xT, 
            vT*torch.cos(thetaT), 
            aT*torch.cos(thetaT) - vT*torch.sin(thetaT)*omegaT
        ], dim=-1)
        
        sy = torch.stack([
            y0, 
            v0*torch.sin(theta0), 
            a0*torch.sin(theta0) + v0*torch.cos(theta0)*omega0, 
            yT, 
            vT*torch.sin(thetaT), 
            aT*torch.sin(thetaT) + vT*torch.cos(thetaT)*omegaT
        ], dim=-1)
        ax = A @ sx[:, :, None].to(torch.float32)
        ay = A @ sy[:, :, None].to(torch.float32)

        # Position interpolation
        traj_x = M_t @ ax
        traj_y = M_t @ ay
        
        # Compute heading from velocity direction
        traj_heading = torch.cat([
            torch.atan2(traj_y[:, :1, 0] - y0.unsqueeze(-1), traj_x[:, :1, 0] - x0.unsqueeze(-1)),
            torch.atan2(traj_y[:, 1:, 0] - traj_y[:, :-1, 0], traj_x[:, 1:, 0] - traj_x[:, :-1, 0])
        ], dim=1)

        return torch.concatenate([torch.cat([traj_x, traj_y, traj_heading[..., None]], axis=-1), 
                            ego_future[:, P+1:, :]], axis=1)
    