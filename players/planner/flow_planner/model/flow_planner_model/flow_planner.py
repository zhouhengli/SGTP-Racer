import re
import os
import sys
from typing import Literal, Callable, Any, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from players.planner.flow_planner.model.model_base import DiffusionADPlanner
from players.planner.flow_planner.model.model_utils.input_preprocess import ModelInputProcessor
from players.planner.flow_planner.model.model_utils.traj_tool import traj_chunking, assemble_actions
from players.planner.flow_planner.data.dataset.dataloader_racing_ros import RacingDataSample

class FlowPlanner(DiffusionADPlanner):

    def __init__(
        self,
        model_encoder,
        model_decoder,
        
        flow_ode,
        
        model_type: Literal['x_start', 'noise', 'velocity'] = 'x_start',
        
        kinematic: Literal["waypoints", "velocity", "acceleration"] = 'waypoints',
    
        assemble_method='linear',
        
        data_processor: ModelInputProcessor = None,
        
        device='cuda',
        
        **planner_params
    ):
        
        super(FlowPlanner, self).__init__()
        self.model_encoder = model_encoder
        self.model_decoder = model_decoder
        self._model_type = model_type
        # self.device = device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.flow_ode = flow_ode # including flow matching path and ode solver
        self.cfg_prob = planner_params['cfg_prob']
        self.cfg_weight = planner_params['cfg_weight']
        self.cfg_type = planner_params['cfg_type']

        self.kinematic = kinematic
        
        self.assemble_method = assemble_method
        
        self.data_processor = data_processor

        self.planner_params = planner_params # including the action_len, future_len etc.
        self.action_num = (self.planner_params['future_len'] - self.planner_params['action_overlap']) // (self.planner_params['action_len'] - self.planner_params['action_overlap'])
        
        self.basic_loss = nn.MSELoss(reduction='none')
        self.pre_traj = None
        self.path = flow_ode.path
        
    def prepare_model_input(self, cfg_flags, data: RacingDataSample, use_cfg, is_training):
        B = data.ego_current.shape[0]

        if is_training:
            # modify the data sample according to cfg_flags
            cfg_type = self.cfg_type
            if cfg_type == 'neighbors':
                neighbor_num = self.planner_params['neighbor_num']
                cfg_neighbor_num = min(self.planner_params['cfg_neighbor_num'], neighbor_num)
                mask_flags = cfg_flags.view(B, *([1] * (data.neighbor_past.dim()-1))).repeat(1, neighbor_num, 1, 1)
                mask_flags[:, cfg_neighbor_num:, :] = 1
                # !!!!!!!!!!!!! must do in-place masking for cfg here !!!!!!!!!!!!!
                data.neighbor_past *= mask_flags
                data.neighbor_future *= mask_flags
            elif cfg_type == 'lanes':
                data.lanes = data.lanes * cfg_flags.view(B, *([1] * (data.lanes.dim()-1)))

        else:
            if use_cfg:
                data = data.repeat(2) # For cfg inference, repeat the batch
                cfg_type = self.cfg_type
                if cfg_type == 'neighbors':
                    neighbor_num = self.planner_params['neighbor_num']
                    cfg_neighbor_num = min(self.planner_params['cfg_neighbor_num'], neighbor_num)
                    # mask_flags = cfg_flags.view(B * 2, *([1] * (data.neighbor_past.dim()-1))).repeat(1, neighbor_num, 1, 1) # [1] * (4 - 1) -> [1, 1, 1]->(B*2, neighbor_num=4, 1, 1)
                    mask_flags = cfg_flags[:, None, None, None].expand(-1, neighbor_num, 1, 1)
                    mask_flags[:, cfg_neighbor_num:, :] = 1
                    # !!!!!!!!!!!!! must do in-place masking for cfg here !!!!!!!!!!!!!
                    data.neighbor_past *= mask_flags
                    data.neighbor_future *= mask_flags
                elif cfg_type == 'lanes':
                    data.lanes = data.lanes * cfg_flags.view(B * 2, *([1] * (data.lanes.dim()-1)))
           
        # Here decides what's used by the model, only ["neighbor_past", "racing_lines", "ego_current"]
        model_inputs, gt = self.data_processor.sample_to_model_input(
            data, device=self.device, kinematic=self.kinematic, is_training=is_training
        ) # gt shape: (B, 1, T+1, D=[x,y,cos,sin])
            
        model_inputs.update({'cfg_flags': cfg_flags})
        
        return model_inputs, gt
        
    def extract_encoder_inputs(self, inputs):
        
        encoder_inputs = {
            'neighbors': inputs['neighbor_future'],
            'racing_lines': inputs['racing_lines'],
        }
        return encoder_inputs
    
    def extract_decoder_inputs(self, encoder_outputs, inputs):
        model_extra = dict(cfg_flags=inputs['cfg_flags'] if 'cfg_flags' in inputs.keys() else None,)
        model_extra.update(encoder_outputs)
        return model_extra
    
    def encoder(self, **encoder_inputs):
        return self.model_encoder(**encoder_inputs)
    
    def decoder(self, x, t, **model_extra):
        return self.model_decoder(x, t, **model_extra)
        
    def forward(self, data: RacingDataSample, mode='train', **params):
        if mode == 'train':
            return self.forward_train(data)
        elif mode == 'inference':
            return self.forward_inference(data, params['use_cfg'], params['cfg_weight'])
    
    def forward_train(self, data: RacingDataSample):
        '''
        Forward a training step and compute the training loss.
        1. generate cfg_flags
        2. preprocess (masking) according to the cfg_flags
        3. model forward
        4. compute basic mse loss
        
        Return:
            prediction: the raw prediction of the model, specified by model.prediction_type;
            loss_dict: a dict of loss containing unreduced mse loss, consistency loss and neighbor prediction loss (if one exists).
        '''
        B = data.ego_current.shape[0]
        roll_dice = torch.rand((B, 1))
        cfg_flags = (roll_dice > self.cfg_prob).to(torch.int32).to(self.device) # NOTE: 1 for conditioned (unmasked), 0 for unconditioned (masked)
        # gt shape: (B, 1, T+1, D=[x,y,cos,sin])
        model_inputs, gt = self.prepare_model_input(cfg_flags, data, use_cfg=False, is_training=True) # note that the cfg_flags are packed into the model_inputs
        
        encoder_inputs = self.extract_encoder_inputs(model_inputs)
        encoder_outputs = self.encoder(**encoder_inputs)

        decoder_model_extra = self.extract_decoder_inputs(encoder_outputs, model_inputs)
        B, P, T_, D = gt.shape
        
        # gt[:, :, 1:, :].shape is (batch, 1 future len=40, state_dim=6), noised_traj shape [B,1,T=40,D=5], target shape the same as noised_traj, t shape (B,)
        noised_traj, target, t = self.flow_ode.sample(gt[:, :, 1:, :], self._model_type) # Here, drop gt[0] since current state is repeated in sample_to_model_input
        noised_traj_tokens = traj_chunking(noised_traj, self.planner_params['action_len'], self.planner_params['action_overlap'])
        noised_traj_tokens = torch.cat(noised_traj_tokens, dim=1)
        target_tokens = traj_chunking(target, self.planner_params['action_len'], self.planner_params['action_overlap'])
        target_tokens = torch.cat(target_tokens, dim=1)
        
        prediction = self.decoder(noised_traj_tokens, t, **decoder_model_extra)
        
        loss_dict = {}
        batch_loss = self.basic_loss(prediction, target_tokens)
        loss_dict['batch_loss'] = batch_loss
        
        loss = torch.sum(batch_loss, dim=-1) # (B, action_num, action_length, dim)
        loss_dict['ego_planning_loss'] = loss.mean()

        if self.planner_params['action_overlap'] > 0 and prediction.shape[1] > 3:
            consistency_loss = [torch.mean(torch.sum(self.basic_loss(prediction[:, i:i+1, -self.planner_params['action_overlap']:, :], prediction[:, i+1:i+2, :self.planner_params['action_overlap'], :]), dim=-1)) for i in range(0, prediction.shape[1]-2)]
            loss_dict['consistency_loss'] = sum(consistency_loss) / len(consistency_loss)
        else:
            loss_dict['consistency_loss'] = torch.tensor(0.0, device=loss.device)
        
        assert not torch.isnan(loss).sum(), f"loss is NaN"
        
        return prediction, loss_dict
    
    def forward_inference(self, data: RacingDataSample, use_cfg=True, cfg_weight=None):

        renoise_t = 0.1 # (1-renoise_t)*x0 + renoise_t*xt, x0 is noise, xt is clean traj
        B = data.ego_current.shape[0]
        if use_cfg:
            cfg_flags = torch.cat([torch.ones((B,), device=self.device), torch.zeros((B,), device=self.device)], dim=0).to(torch.int32)
        else:
            cfg_flags = torch.ones((B,), device=self.device).to(torch.int32)
        
        model_inputs, _ = self.prepare_model_input(cfg_flags, data, use_cfg, is_training=False)
    
        encoder_inputs = self.extract_encoder_inputs(model_inputs)
        encoder_outputs = self.encoder(**encoder_inputs)

        decoder_model_extra = self.extract_decoder_inputs(encoder_outputs, model_inputs)
        
        opp_fut = model_inputs["neighbor_future"][:, :, :, :2] # (B, T, 2)
        if self.pre_traj is None:
            x_init = torch.randn((B, self.action_num, self.planner_params['action_len'], self.planner_params['state_dim']), device=self.device) # (batch=1, action_num=7, action_len=10, state_dim=4)
        else:
            # x_init = self.pre_traj
            # t = self.time_sampler.sample(B).to(self.device)
            t = torch.full((B,), renoise_t, device=self.device)
            x_0 = torch.randn_like(self.pre_traj, device=self.device)
            path_sample = self.path.sample(x_0=x_0, x_1=self.pre_traj, t=t)
            # x_1 is clean data
            x_init = path_sample.x_t
        
        sample = self.flow_ode.generate(self.pre_traj, x_init, self.decoder, self._model_type, start_t=renoise_t, use_cfg=use_cfg, cfg_weight=cfg_weight, opp_fut=opp_fut, **decoder_model_extra) 
        self.pre_traj = sample

        sample = assemble_actions(sample, self.planner_params['future_len'], self.planner_params['action_len'], self.planner_params['action_overlap'], self.planner_params['state_dim'], self.assemble_method) 
        
        sample = self.data_processor.state_postprocess(sample) # inverse normalization, transform back to original state space
        
        return sample

    @property
    def model_type(self,):
        return self._model_type
    
    def get_optimizer_params(self):
        return [
            {'params': self.model_encoder.parameters()},
            {'params': self.model_decoder.parameters()}
        ]