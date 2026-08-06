from random import sample
from typing import Dict
from matplotlib.pylab import sample
import torch
from torch import nn
from flow_matching.utils import ModelWrapper
from flow_matching.path.scheduler.scheduler import Scheduler
from flow_matching.path.scheduler.schedule_transform import ScheduleTransformedModel
from flow_matching.path.affine import AffineProbPath
from players.planner.flow_planner.model.model_utils.traj_tool import traj_chunking, assemble_actions

class VelocityModel(nn.Module):

    def __init__(self, model_fn, path, pred_transform_func: str, correct_xt_fn=None, use_cfg=True, cfg_weight=None):
        super().__init__()
        self.model_fn = model_fn
        self.path = path
        self.pred_transform_func = pred_transform_func
        self.correct_xt_fn = correct_xt_fn
        self.use_cfg = use_cfg
        self.cfg_weight = cfg_weight

    def forward(self, x, t, opp_fut=None, **model_extras):
        # TODO: add wrapper for guidance etc.
        '''
        :params
            x: sampled_trajectories for model input
            t: sampled time for model input
            s0: initial state for correcting x_t if needed
            pre_x: previous x for correction if needed
            opp_fut: opponent future trajectories for guidance
            **model_extras: other model params used for prediction 
        '''
        B, P, _, _ = x.shape
        # x_base = x
        t = t.unsqueeze(0).to(x.device)
        
        if self.use_cfg:
            x = x.repeat(2, *[1] * (x.dim()-1))
        
        pred = self.model_fn(x, t, **model_extras)
        
        # x0=pred, xt=xclean, t=t
        u = self.pred_transform_func(pred, x, t)
        # Add guidance here

        if self.use_cfg:
            u_cond, u_uncond = torch.chunk(u, 2) # the conditioned batch is the first half
            u = (1 - self.cfg_weight) * u_uncond + self.cfg_weight * u_cond
        
        return u