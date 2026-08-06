import time
import warnings
import torch
import numpy as np
from typing import Dict
import hydra
from hydra.utils import instantiate
import omegaconf
import copy

from players.planner.flow_planner.data.dataset.utils import convert_to_model_inputs

warnings.filterwarnings("ignore")

from players.planner.flow_planner.data.dataset.utils import RacingDataSample
from players.planner.flow_planner.data.dataset.dataloader_racing_ros import RacingDataset
from players.planner.flow_planner.data.augmentation.state_aug import StatePerturbation

class FlowRacingPlanner:
    def __init__(
            self,
            config_path,
            ckpt_path: str,
            enable_ema: bool = None,
            device: str = None,
            use_cfg: bool = None,
        ):

        assert device in ["cpu", "cuda"], f"device {device} not supported"
        if device == "cuda":
            assert torch.cuda.is_available(), "cuda is not available"

        config = omegaconf.OmegaConf.load(config_path)
        self._config = config
        self._ckpt_path = ckpt_path

        self._ema_enabled = enable_ema
        self._device = device
        self._history_points_num = config.model.model_encoder.neighbor_encoder.cond_time_len
        self._agent_num = config.model.neighbor_num
        self._future_len = config.model.future_len

        self._planner = instantiate(config.model)
        self._cond_time_len = config.model.model_encoder.neighbor_encoder.cond_time_len

        self.core = instantiate(config.core)

        self.use_cfg = use_cfg

        self.cfg_weight = config.model.cfg_weight
        self.centralize_fn = StatePerturbation(augment_prob=0.0)  # only centralization, no noise
        self.ego_current_xy = None

    def initialize(self) -> None:

        if self._ckpt_path is not None:
            state_dict = torch.load(self._ckpt_path, weights_only=True, map_location=self._device)

            if self._ema_enabled:
                state_dict = state_dict['ema_state_dict']
            else:
                if "model" in state_dict.keys():
                    state_dict = state_dict['model']
            # use for ddp
            # model_state_dict = {k[len("module."):]: v for k, v in state_dict.items() if k.startswith("module.")}
            model_state_dict = state_dict
            self._planner.load_state_dict(model_state_dict)
        else:
            print("load random model")

        self._planner = self._planner.to(self._device)
        self._planner.eval()

    def _remap_one_raw_sample(
        self,
        ego_current_state_raw,
        ego_agent_future_raw,
        neighbor_agents_past_raw,
        neighbor_agents_future_raw,
        racing_line_seg_raw,
    ):
        """Remap one raw planner sample into the model-side feature format."""
        return RacingDataset.remap_sample(
            ego_current_state_raw=ego_current_state_raw,
            ego_agent_future_raw=ego_agent_future_raw,
            neighbor_agents_past_raw=neighbor_agents_past_raw,
            neighbor_agents_future_raw=neighbor_agents_future_raw,
            racing_line_seg_raw=racing_line_seg_raw,
            past_neighbor_num=self._history_points_num,
            future_len=self._future_len,
            condition_future_neighbor_len=self._cond_time_len,
            condition_neighbor_num=self._agent_num,
        )


    def _make_racing_data_from_normed_inputs(self, normed_inputs, batched: bool):
        """Build RacingDataSample from normalized tensor inputs."""
        return RacingDataSample(
            batched=batched,
            ego_current=normed_inputs["ego_current_state"],
            ego_future=normed_inputs["ego_agent_future"],
            neighbor_past=normed_inputs["neighbor_agents_past"],
            racing_lines=normed_inputs["racing_lines"],
            neighbor_future=normed_inputs["neighbor_agents_future"],
        )


    def planner_input_to_model_inputs(self, sample, num_candidates):
        """Convert planner sample into model inputs.

        Supported opponent future condition formats:
        - [A, T, D]; single condition, repeated to num_candidates.
        - [B, A, T, D]; batched conditions, one condition per candidate.

        The second mode is the key path for multi-batch flow conditioning.
        """
        ego_current_state_raw = np.asarray(sample["ego_current_state"]).copy()
        ego_agent_future_raw = np.asarray(sample["ego_agent_future"]).copy()
        neighbor_agents_past_raw = np.asarray(sample["neighbor_agents_past"]).copy()
        neighbor_agents_future_raw = np.asarray(sample["neighbor_agents_future"]).copy()
        racing_line_seg_raw = np.asarray(sample["racing_line_seg"]).copy()

        assert ego_current_state_raw.shape == (8,), (
            f"ego_current_state expected (8,), got {ego_current_state_raw.shape}"
        )

        assert racing_line_seg_raw.shape == (150, 6), (
            f"racing_line_seg expected (150,6), got {racing_line_seg_raw.shape}"
        )

        batch_size = int(neighbor_agents_future_raw.shape[0])
        num_candidates = int(num_candidates)

        if batch_size != num_candidates:
            raise ValueError(
                "Batched neighbor_agents_future requires "
                f"batch_size == num_candidates. "
                f"Got batch_size={batch_size}, num_candidates={num_candidates}."
            )

        normed_list = []

        # This loop is only preprocessing. It does not run flow inference.
        for batch_idx in range(batch_size):
            (
                ego_current_state,
                ego_agent_future,
                neighbor_agents_past,
                neighbor_agents_future,
                racing_line_seg,
            ) = self._remap_one_raw_sample(
                ego_current_state_raw=ego_current_state_raw,
                ego_agent_future_raw=ego_agent_future_raw,
                neighbor_agents_past_raw=neighbor_agents_past_raw,
                neighbor_agents_future_raw=neighbor_agents_future_raw[batch_idx],
                racing_line_seg_raw=racing_line_seg_raw,
            )

            data_dict = {
                "racing_lines": racing_line_seg,
                "neighbor_agents_past": neighbor_agents_past,
                "ego_current_state": ego_current_state,
                "ego_agent_future": ego_agent_future,
                "neighbor_agents_future": neighbor_agents_future,
            }

            normed_list.append(convert_to_model_inputs(data_dict, self._device))

        normed_inputs = {
            key: torch.cat([one[key] for one in normed_list], dim=0)
            for key in normed_list[0].keys()
        }

        data = self._make_racing_data_from_normed_inputs(
            normed_inputs=normed_inputs,
            batched=True,
        )

        self.ego_current_xy = (
            copy.deepcopy(data.ego_current[:, :4])
            .to("cpu")
            .numpy()
            .astype(np.float32)
        )

        data_norm = self.centralize_fn.centric_transform(data).to(self._device)

        return data_norm

    def traj_ego_to_world(self, traj_xyyaw_ego, ego_xy_cossin_world):
        """Transform ego-frame trajectories back to world-frame trajectories."""
        def wrap_to_pi(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi

        traj_xyyaw_ego = np.asarray(traj_xyyaw_ego)
        ego_xy_cossin_world = np.asarray(ego_xy_cossin_world)

        if traj_xyyaw_ego.ndim not in (2, 3) or traj_xyyaw_ego.shape[-1] < 3:
            raise ValueError(
                f"Expected traj_xyyaw_ego shape [T,3] or [N,T,3], "
                f"got {traj_xyyaw_ego.shape}."
            )

        x_e = traj_xyyaw_ego[..., 0]
        y_e = traj_xyyaw_ego[..., 1]
        yaw_e = traj_xyyaw_ego[..., 2]

        # Single trajectory path.
        if traj_xyyaw_ego.ndim == 2:
            if ego_xy_cossin_world.shape == (4,):
                ego_x, ego_y, c, s = ego_xy_cossin_world
            elif ego_xy_cossin_world.shape == (1, 4):
                ego_x, ego_y, c, s = ego_xy_cossin_world[0]
            elif ego_xy_cossin_world.shape == (4, 1):
                ego_x, ego_y, c, s = ego_xy_cossin_world[:, 0]
            else:
                raise ValueError(
                    "Invalid ego pose shape for single trajectory: "
                    f"{ego_xy_cossin_world.shape}."
                )

        # Batched trajectory path.
        else:
            if ego_xy_cossin_world.shape == (4,):
                ego_x, ego_y, c, s = ego_xy_cossin_world
                ego_x = np.asarray([ego_x])
                ego_y = np.asarray([ego_y])
                c = np.asarray([c])
                s = np.asarray([s])
            elif ego_xy_cossin_world.ndim == 2 and ego_xy_cossin_world.shape[0] == 4:
                ego_x, ego_y, c, s = ego_xy_cossin_world
            elif ego_xy_cossin_world.ndim == 2 and ego_xy_cossin_world.shape[1] == 4:
                ego_x, ego_y, c, s = ego_xy_cossin_world.T
            else:
                raise ValueError(
                    "Invalid ego pose shape for batched trajectories: "
                    f"{ego_xy_cossin_world.shape}."
                )

            ego_x = ego_x[:, None]
            ego_y = ego_y[:, None]
            c = c[:, None]
            s = s[:, None]

        x_w = c * x_e - s * y_e + ego_x
        y_w = s * x_e + c * y_e + ego_y

        heading_world = np.arctan2(s, c)
        yaw_w = wrap_to_pi(yaw_e + heading_world)

        return np.stack([x_w, y_w, yaw_w], axis=-1)
    
    def outputs_to_trajectory(self, outputs: Dict[str, torch.Tensor]):    

        predictions = outputs[:, 0, :, :].detach().cpu().numpy().astype(np.float32) # T, 4
        heading = np.arctan2(predictions[:, :, 3], predictions[:, :, 2])[..., None] 
        predictions = np.concatenate([predictions[..., :2], heading], axis=-1) 

        # transform back to global coordinate
        traj_world = self.traj_ego_to_world(predictions, self.ego_current_xy.T) # (T, 3)

        return traj_world

    def infer_planner_trajectory(self, inputs):

        # In flow matching, obs_normalizer is done by sample_to_model_input()
        # And state inverse is done by forward_inference()
        # start_time = time.time()
        with torch.inference_mode():
            outputs = self.core.inference(self._planner, inputs, use_cfg=self.use_cfg, cfg_weight=self.cfg_weight)
            # print(f"Inference time: {time.time() - start_time:.6f} s")

        # Here we output trajectory for openloop evaluation
        return outputs
