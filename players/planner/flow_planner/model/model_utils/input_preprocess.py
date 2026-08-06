import torch
import math
from players.planner.flow_planner.data.dataset.dataloader_racing_ros import RacingDataSample

class ModelInputProcessor:
    def __init__(
        self,
        future_len,
        obs_normalizer,
        state_normalizer,
        neighbor_pred_num
    ):
        self.future_len = future_len
        self.obs_normalizer = obs_normalizer
        self.state_normalizer = state_normalizer
        self.neighbor_pred_num = neighbor_pred_num

    def state_preprocess(self, x):
        return self.state_normalizer(x) if self.state_normalizer is not None else x
    
    def state_postprocess(self, x):
        return self.state_normalizer.inverse(x) if self.state_normalizer is not None else x

    def x_differentiate(self, x_future, x_current):
        x_all = torch.cat([x_current, x_future], dim=-2)
        return x_all[..., 1:, :] - x_all[..., :-1, :]

    def x_integral(self, dx_future, x_current):
        v_all = torch.cat([x_current, dx_future], dim=-2)
        return torch.cumsum(v_all, dim=-2)[..., 1:, :]

    def sample_to_model_input(
        self,
        data: RacingDataSample,
        device,
        kinematic,
        is_training: bool=False
    ):
                
        if self.obs_normalizer is not None:
            data = self.obs_normalizer(data)

        ego_future = data.ego_future
        if ego_future.numel() != 0:
            ego_future = ego_future[..., -self.future_len:, :] # (x, y, heading, v, yaw rate)

        model_inputs = {}
        model_inputs['neighbor_past'] = data.neighbor_past.to(device)
        model_inputs['neighbor_future'] = data.neighbor_future.to(device)
        model_inputs['racing_lines'] = data.racing_lines.to(device)
        
        ego_current_state = data.ego_current
        model_inputs['ego_current'] = ego_current_state
        ego_current_xy_cos_sin = ego_current_state[..., :4] # (x, y, cos(heading), sin(heading))
        ego_current = torch.cat([
            ego_current_xy_cos_sin[..., :2], # x, y
            torch.atan2(ego_current_xy_cos_sin[..., 3:4], ego_current_xy_cos_sin[..., 2:3]), # heading
        ], dim=-1) # (B, state_dim=[x,y,yaw])

        current_states = ego_current[:, None] # (batch, 1, state_dim=[x,y,yaw,vx,vy])
        
        if is_training:
            gt_future = ego_future[:, None, :, :] # (B,1,T,5)

            gt_with_current = torch.cat([
                    current_states[:, :, None, :],
                    gt_future
                ], dim=2) # (B,1,T+1,5=[x,y,heading]), adding current state at beginning for calculate velocity/acceleration

            gt_with_current.to(device)
        else:
            gt_with_current = current_states[:, :, None, :].repeat(1, 1, self.future_len + 1, 1) # (B,1,T+1,5)

        # Only 'waypoints' is used for racing flow planner
        if kinematic == 'waypoints':
            gt_with_current = torch.cat([
                gt_with_current[..., :2],
                torch.cat([
                    gt_with_current[..., 2:3].cos(),
                    gt_with_current[..., 2:3].sin()
                ], dim=-1),
            ], dim=-1)
            gt_with_current[..., 1:, :] = self.state_normalizer(gt_with_current[..., 1:, :]) # (B,1,T+1,4=[x,y,cos,sin]), the first one will be removed later
        elif kinematic == 'velocity': # NOT USED
            future_velocity = self.x_differentiate(gt_with_current[..., 1:, :], gt_with_current[..., :1, :])
            gt_with_current = torch.cat([gt_with_current[..., :1, :], future_velocity], dim=-2)
        elif kinematic == 'acceleration': # NOT USED
            future_velocity = self.x_differentiate(gt_with_current[..., 1:, :], gt_with_current[..., :1, :])
            current_velocity = torch.cat([ego_current_state[..., 4:6], ego_current_state[..., 9:10]], dim=-1)[:, None, None, :]
            future_acc = self.x_differentiate(future_velocity, current_velocity)
            gt_with_current = torch.cat([current_velocity, future_acc], dim=-2)
        
        return model_inputs, gt_with_current