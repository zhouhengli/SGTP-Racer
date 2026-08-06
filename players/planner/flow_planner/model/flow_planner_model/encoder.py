import torch
from torch import nn
from players.planner.flow_planner.model.modules.encoder_modules import *

class FlowPlannerEncoder(nn.Module):
    def __init__(self, 
                 encoder_hidden_dim,
                 with_ego_history,
                 neighbor_encoder: AgentFusionEncoder,
                 lane_encoder: LaneFusionEncoder,
                 action_length: int, # 10
                 action_overlap: int, # 5
                 future_len: int=24,
                 lane_num=10, 
                 lane_dim=6,
                 neighbor_agent_num=4,
                 neighbor_pred_num=0,
                 ):
        super().__init__()

        self.with_ego_history = with_ego_history
        self.lane_dim = lane_dim

        self.neighbor_encoder = neighbor_encoder

        self.lane_encoder = lane_encoder

        self.future_len = future_len
        self.neighbor_agent_num = neighbor_agent_num
        self.lane_num = lane_num
        self.neighbor_pred_num = neighbor_pred_num

        self.token_num = self.neighbor_agent_num + self.lane_num

        self.lane_pos_emb = nn.Linear(6, encoder_hidden_dim)
        self.agent_pos_emb = nn.Linear(6, encoder_hidden_dim)
        self.hidden_dim = encoder_hidden_dim

        action_num = (self.future_len - action_overlap) // (action_length - action_overlap)
        self.action_num = int(action_num)

        self.initialize_weights()


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        self.apply(_basic_init)

        # Initialize embedding MLP:
        # nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.normal_(self.lane_pos_emb.weight, std=0.02)
        nn.init.normal_(self.agent_pos_emb.weight, std=0.02)
        nn.init.normal_(self.neighbor_encoder.type_emb.weight, std=0.02)

    def forward(self, neighbors, racing_lines):
        B = neighbors.shape[0]

        encoding_neighbors, neighbors_mask, neighbor_pos = self.neighbor_encoder(neighbors)
        encoding_lanes, lanes_mask, lane_pos = self.lane_encoder(racing_lines)

        lanes_loc = racing_lines[:, :, int(self.lane_encoder._lane_points_num / 2), :2].clone()
        neighbors_loc = neighbors[:, :, -1, :2].clone()
        ego_loc = torch.tensor([-0.5, 0], device=neighbors.device)[None, None, :].repeat(B, self.action_num, 1) # 0,0 after normalization becomes -0.5,0
        pred_neighbor_loc = neighbors[:, :self.neighbor_pred_num, -1, :2].clone() # only include number of self.neighbor_pred_num neighbors, the shape MUST be (B, 0, 2) here
        all_loc = torch.cat([neighbors_loc, lanes_loc, ego_loc, pred_neighbor_loc], dim=-2) # cat at the dimension of number
        token_dist = torch.norm(all_loc[:, None, :, :] - all_loc[:, :, None, :], dim=-1)

        def encoding_process(encoding, mask, pos, pos_emb):
            token_num = encoding.shape[1]
            pos = pos.view(B * token_num, -1).type(torch.float32)
            mask = mask.view(-1)
            encoding_pos = pos_emb(pos[~mask])
            encoding_pos_result = torch.zeros((B * token_num, self.hidden_dim), device=encoding_pos.device)
            encoding_pos_result[~mask] = encoding_pos
            encoding = encoding + encoding_pos_result.view(B, token_num, -1)
            return encoding
        
        neighbors_encoding = encoding_process(encoding_neighbors, neighbors_mask, neighbor_pos, self.agent_pos_emb)
        lanes_encoding = encoding_process(encoding_lanes, lanes_mask, lane_pos, self.lane_pos_emb)

        encoder_outputs=dict(encodings=(neighbors_encoding, lanes_encoding),
                              masks=(~neighbors_mask, ~lanes_mask),
                              token_dist=token_dist)

        return encoder_outputs