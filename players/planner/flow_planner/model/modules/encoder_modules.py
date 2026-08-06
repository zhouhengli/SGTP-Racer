import math
import torch
import torch.nn as nn
from timm.layers import Mlp

from functools import partial
from players.planner.flow_planner.model.modules.decoder_modules import MixerBlock, SelfAttentionBlock
from players.planner.flow_planner.model.model_utils.tool_func import lanes_to_route_mask


class AgentFusionEncoder(nn.Module):
    def __init__(self, cond_time_len, drop_path_rate=0.3, hidden_dim=192, layer_num=3, tokens_mlp_dim=64, channels_mlp_dim=128):
        super().__init__()

        self._hidden_dim = hidden_dim
        self._channel = channels_mlp_dim

        self.type_emb = nn.Linear(2, channels_mlp_dim)

        self.channel_pre_project = Mlp(in_features=6+1,  hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.token_pre_project = Mlp(in_features=cond_time_len, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.)

        self.blocks = nn.ModuleList([MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate) for i in range(layer_num)])

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x):
        '''
        x: B, N, H, F (x, y, cos, sin, vx, vy)
        '''
        F_dim = x.shape[-1]
        x = x[..., :F_dim] # x, y, cos, sin, vx, vy

        pos = x[:, :, 0, :F_dim].clone() # x, y, cos, sin, take the current position as the extra feature
        # neighbor: [1,0]
        pos[..., -2:] = 0.0
        pos[..., -2] = 1.0
        
        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :F_dim], 0), dim=-1).to(x.device) == 0 # for mask_v==0, this indicates that the corresponding x is padded with 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        x = torch.cat([x, (~mask_v).float().unsqueeze(-1)], dim=-1) # Here cat the mask dimension, so the in_features of self.channel_pre_project is F_dim+1
        x = x.view(B * P, V, -1)

        valid_indices = ~mask_p.view(-1) 
        x = x[valid_indices] 
        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)  

        # pooling
        x = torch.mean(x, dim=1)
        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B * P, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x  # Fill in valid parts
        
        return x_result.view(B, P, -1) , mask_p.reshape(B, -1), pos.view(B, P, -1)
  
class LaneFusionEncoder(nn.Module):
    def __init__(self, lane_points_num, drop_path_rate=0.3, hidden_dim=192, layer_num=3, tokens_mlp_dim=64, channels_mlp_dim=128):
        super().__init__()
        self._lane_points_num = lane_points_num
        self._channel = channels_mlp_dim

        self.channel_pre_project = Mlp(in_features=6, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.token_pre_project = Mlp(in_features=lane_points_num, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.)

        self.blocks = nn.ModuleList([MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate) for i in range(layer_num)])

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x):
        '''
        x: B, P, V, D (x, y, cos(phi), sin(phi), dright, dleft)
        '''
        x = x[..., :6]

        pos = x[:, :, int(self._lane_points_num / 2), :6].clone() # x, y, cos(phi), sin(phi)
        # lane: [0,1]
        pos[..., -2:] = 0.0
        pos[..., -1] = 1.0

        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :6], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        x = x.view(B * P, V, -1)

        valid_indices = ~mask_p.view(-1) 
        x = x[valid_indices].type(torch.float32)

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)

        for block in self.blocks:
            x = block(x)  

        x = torch.mean(x, dim=1)
        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B * P, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x  # Fill in valid parts
        
        return x_result.view(B, P, -1) , mask_p.reshape(B, -1), pos.view(B, P, -1)

class FusionEncoder(nn.Module):
    def __init__(self, hidden_dim=192, num_heads=6, drop_path_rate=0.3, layer_num=3):
        super().__init__()

        dpr = drop_path_rate

        self.blocks = nn.ModuleList(
            [SelfAttentionBlock(hidden_dim, num_heads, dropout=dpr) for i in range(layer_num)]
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask):

        mask[:, 0] = False

        for b in self.blocks:
            x = b(x, mask)

        return self.norm(x)