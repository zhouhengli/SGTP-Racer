"""
Biased MPPI extension of Kohei Honda's mppi_core.

Key changes vs the original core:
1. Keeps the same MPPI API.
2. Adds optional external prior sampling without knowing the prior type.

This file is intended as a drop-in replacement proposal for mppi_core.py.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from scipy.optimize import brentq, minimize_scalar
from torch.distributions.multivariate_normal import MultivariateNormal


PriorSampler = Callable[
    [torch.Tensor, Dict, torch.Tensor, int], Optional[torch.Tensor]
]

TrajectoryCostFunc = Callable[
    [torch.Tensor, torch.Tensor, Dict], Optional[torch.Tensor]
]

class BiasedMPPI(nn.Module):
    """Model Predictive Path Integral Control (MPPI) solver.

    Extended with optional Biased-MPPI sampling.
    """

    def __init__(
        self,
        horizon: int,
        num_samples: int,
        dim_state: int,
        dim_control: int,
        dynamics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        cost_func: Callable[[torch.Tensor, torch.Tensor, Dict], torch.Tensor],
        u_min: torch.Tensor,
        u_max: torch.Tensor,
        sigmas: torch.Tensor,
        lambda_: float | str,
        lbps_delta: float = 0.01,
        essps_target_ess: Optional[float] = None,
        lambda_min: float = 0.01,
        lambda_max: float = 10.0,
        exploration: float = 0.0,
        use_sg_filter: bool = False,
        sg_window_size: int = 5,
        sg_poly_order: int = 3,
        device=torch.device("cuda"),
        dtype=torch.float32,
        seed: int = 42,
        # -------- unified biased rollout-sampling modes --------
        sampling_bias_mode: str = "none",  # "none" | "prior"

        # Prior mode
        prior_sampler: Optional[PriorSampler] = None,
        num_prior_samples: int = 0,
        prior_noise_scale: float = 0.0,


        keep_mean_sample: bool = True,
        trajectory_cost_func: Optional[TrajectoryCostFunc] = None,
    ) -> None:
        super().__init__()

        torch.manual_seed(seed)

        assert u_min.shape == (dim_control,)
        assert u_max.shape == (dim_control,)
        assert sigmas.shape == (dim_control,)

        if torch.cuda.is_available() and device == torch.device("cuda"):
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        self._dtype = dtype

        self._horizon = horizon
        self._num_samples = num_samples
        self._dim_state = dim_state
        self._dim_control = dim_control
        self._dynamics = dynamics
        self._cost_func = cost_func
        self._u_min = u_min.clone().detach().to(self._device, self._dtype)
        self._u_max = u_max.clone().detach().to(self._device, self._dtype)
        self._sigmas = sigmas.clone().detach().to(self._device, self._dtype)
        self._exploration = exploration
        self._use_sg_filter = use_sg_filter
        self._sg_window_size = sg_window_size
        self._sg_poly_order = sg_poly_order

        # sampling configuration
        self._sampling_bias_mode = str(sampling_bias_mode).lower()
        if self._sampling_bias_mode not in {"none", "prior"}:
            raise ValueError(
                "sampling_bias_mode must be 'none' or 'prior'. "
                f"Got {sampling_bias_mode}."
            )

        self._prior_sampler = prior_sampler
        self._num_prior_samples = int(num_prior_samples)
        self._prior_noise_scale = float(prior_noise_scale)


        self._keep_mean_sample = keep_mean_sample
        self._trajectory_cost_func = trajectory_cost_func

        self._covariance = torch.zeros(
            self._horizon,
            self._dim_control,
            self._dim_control,
            device=self._device,
            dtype=self._dtype,
        )
        self._covariance[:, :, :] = torch.diag(sigmas**2).to(self._device, self._dtype)
        self._inv_covariance = torch.zeros_like(
            self._covariance, device=self._device, dtype=self._dtype
        )
        for t in range(1, self._horizon):
            self._inv_covariance[t] = torch.inverse(self._covariance[t])

        zero_mean = torch.zeros(dim_control, device=self._device, dtype=self._dtype)
        self._noise_distribution = MultivariateNormal(
            loc=zero_mean, covariance_matrix=self._covariance
        )

        self._sample_shape = torch.Size([self._num_samples])
        self._action_noises = self._noise_distribution.rsample(
            sample_shape=self._sample_shape
        )

        zero_mean_seq = torch.zeros(
            self._horizon, self._dim_control, device=self._device, dtype=self._dtype
        )
        self._perturbed_action_seqs = torch.clamp(
            zero_mean_seq + self._action_noises, self._u_min, self._u_max
        )
        self._previous_action_seq = zero_mean_seq

        self._coeffs = self._savitzky_golay_coeffs(
            self._sg_window_size, self._sg_poly_order
        )
        self._actions_history_for_sg = torch.zeros(
            self._horizon - 1, self._dim_control, device=self._device, dtype=self._dtype
        )

        self._state_seq_batch = torch.zeros(
            self._num_samples,
            self._horizon + 1,
            self._dim_state,
            device=self._device,
            dtype=self._dtype,
        )
        self._weights = torch.zeros(
            self._num_samples, device=self._device, dtype=self._dtype
        )
        self._optimal_state_seq = torch.zeros(
            self._horizon + 1, self._dim_state, device=self._device, dtype=self._dtype
        )
        self._last_costs = torch.zeros(
            self._num_samples, device=self._device, dtype=self._dtype
        )

        self._lambda: float | str = lambda_
        self._lbps_delta = lbps_delta
        self._essps_target_ess = (
            essps_target_ess if essps_target_ess is not None else num_samples / 10
        )
        self._lambda_min = lambda_min
        self._lambda_max = lambda_max

        if self._lambda == "MPO":
            self._auto_lambda = "MPO"
            self._lambda = 1.0
            self._mpo_epsilon = 0.1
            self.log_temperature = torch.nn.Parameter(
                torch.log(
                    torch.tensor([self._lambda], device=self._device, dtype=self._dtype)
                )
            )
            self.optimizer = torch.optim.Adam([self.log_temperature], lr=0.2)
        elif self._lambda == "LBPS":
            self._auto_lambda = "LBPS"
        elif self._lambda == "ESSPS":
            self._auto_lambda = "ESSPS"
        elif isinstance(self._lambda, float):
            self._auto_lambda = None
        else:
            raise ValueError(
                "lambda_ must be 'MPO', 'LBPS', 'ESSPS', or a float value."
            )
        
    def reset(self):
        self._previous_action_seq = torch.zeros(
            self._horizon, self._dim_control, device=self._device, dtype=self._dtype
        )
        self._actions_history_for_sg = torch.zeros(
            self._horizon - 1, self._dim_control, device=self._device, dtype=self._dtype
        )

    def set_prior_sampler(
        self,
        prior_sampler: Optional[PriorSampler],
        num_prior_samples: Optional[int] = None,
        prior_noise_scale: Optional[float] = None,
    ) -> None:
        self._prior_sampler = prior_sampler
        if num_prior_samples is not None:
            self._num_prior_samples = int(num_prior_samples)
        if prior_noise_scale is not None:
            self._prior_noise_scale = float(prior_noise_scale)

    def _prior_slot_bounds(self) -> Tuple[int, int]:
        start = 1 if self._keep_mean_sample else 0
        end = min(start + max(self._num_prior_samples, 0), self._num_samples)
        return start, end

    def _select_prior_slot_policies(
        self,
        state: torch.Tensor,
        info: Dict,
    ) -> List[str]:
        if self._active_bias_mode(info) != "prior":
            return []

        if (
            self._prior_sampler is None
            or self._num_prior_samples <= 0
        ):
            return []

        start, end = self._prior_slot_bounds()
        budget = max(0, end - start)
        if budget <= 0:
            return []

        if hasattr(self._prior_sampler, "select_policies"):
            slot_names = self._prior_sampler.select_policies(state, info, budget)
            if slot_names is None:
                return []
            return list(slot_names)[:budget]

        return []

    def _active_bias_mode(self, info: Dict) -> str:
        """Return the effective sampling mode for this forward pass.

        A zero prior budget is a hard off switch.  The core falls back to
        pure MPPI when no prior sampler or no prior slots are available.
        """
        mode = str(info.get("sampling_bias_mode", self._sampling_bias_mode)).lower()
        if mode not in {"none", "prior"}:
            raise ValueError(
                "sampling_bias_mode must be 'none' or 'prior'. "
                f"Got {mode}."
            )

        if (
            mode == "prior"
            and (
                self._prior_sampler is None
                or self._num_prior_samples <= 0
            )
        ):
            return "none"

        return mode


    def _compute_batch_costs(
        self,
        state_seq_batch: torch.Tensor,
        action_seq_batch: torch.Tensor,
        info: Dict,
    ) -> torch.Tensor:
        """Compute the same BiasedMPPI cost on an arbitrary batch size."""
        batch_size = int(action_seq_batch.shape[0])

        costs = torch.zeros(
            batch_size,
            self._horizon,
            device=self._device,
            dtype=self._dtype,
        )

        cost_info = dict(info)

        for t in range(self._horizon):
            prev_index = t - 1 if t > 0 else 0
            cost_info["prev_action"] = action_seq_batch[:, prev_index, :]
            cost_info["t"] = t
            cost_info["is_terminal_step"] = False

            costs[:, t] = self._cost_func(
                state_seq_batch[:, t, :],
                action_seq_batch[:, t, :],
                cost_info,
            )

        cost_info["is_terminal_step"] = True

        zero_action = torch.zeros(
            batch_size,
            self._dim_control,
            device=self._device,
            dtype=self._dtype,
        )

        terminal_costs = self._cost_func(
            state_seq_batch[:, -1, :],
            zero_action,
            cost_info,
        )

        total_costs = torch.sum(costs, dim=1) + terminal_costs

        if self._trajectory_cost_func is not None:
            extra_costs = self._trajectory_cost_func(
                state_seq_batch,
                action_seq_batch,
                cost_info,
            )
            if extra_costs is not None:
                extra_costs = extra_costs.to(self._device, self._dtype).view(-1)
                if extra_costs.shape[0] != batch_size:
                    raise ValueError(
                        "trajectory_cost_func returned wrong batch size: "
                        f"{extra_costs.shape[0]} vs {batch_size}."
                    )
                # print(f"Adding extra trajectory costs: {extra_costs.cpu().numpy().sum()}, total before: {total_costs.cpu().numpy().sum()}")
                total_costs = total_costs + extra_costs

        return total_costs


    def _update_auto_lambda(self, costs: torch.Tensor) -> None:
        if self._auto_lambda == "LBPS":
            result = minimize_scalar(
                lambda lambda_: self._lbps_objective(lambda_, costs.detach()),
                bounds=(self._lambda_min, self._lambda_max),
                method="bounded",
            )
            self._lambda = result.x

        elif self._auto_lambda == "ESSPS":
            ess_at_min = self._compute_ess(
                torch.softmax(-costs.detach() / self._lambda_min, dim=0)
            )
            ess_at_max = self._compute_ess(
                torch.softmax(-costs.detach() / self._lambda_max, dim=0)
            )

            if self._essps_target_ess <= ess_at_min:
                self._lambda = self._lambda_min
            elif self._essps_target_ess >= ess_at_max:
                self._lambda = self._lambda_max
            else:
                self._lambda = brentq(
                    lambda lambda_: self._essps_objective(lambda_, costs.detach()),
                    self._lambda_min,
                    self._lambda_max,
                )


    def _finalize_action_sequence(
        self,
        state: torch.Tensor,
        optimal_action_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Shared finalization for none / prior modes."""
        if self._use_sg_filter:
            prolonged_action_seq = torch.cat(
                [self._actions_history_for_sg, optimal_action_seq],
                dim=0,
            )
            filtered_action_seq = torch.zeros_like(
                prolonged_action_seq,
                device=self._device,
                dtype=self._dtype,
            )
            for i in range(self._dim_control):
                filtered_action_seq[:, i] = self._apply_savitzky_golay(
                    prolonged_action_seq[:, i],
                    self._coeffs,
                )
            optimal_action_seq = filtered_action_seq[-self._horizon :]

        optimal_state_seq = self._states_prediction(
            state,
            optimal_action_seq.unsqueeze(0),
        )

        self._previous_action_seq = optimal_action_seq.detach().clone()
        self._optimal_state_seq = optimal_state_seq[0].detach().clone()

        optimal_action = optimal_action_seq[0]
        self._actions_history_for_sg = torch.cat(
            [self._actions_history_for_sg[1:], optimal_action.view(1, -1)]
        )

        return optimal_action_seq, optimal_state_seq

    def _sample_action_sequences(
        self,
        state: torch.Tensor,
        info: Dict,
        mean_action_seq: torch.Tensor,
    ) -> torch.Tensor:
        """Base sampling stage.

        This now only builds the Gaussian / inherited sample set and optionally
        keeps the warm-start sample. Prior actions are injected later inside
        the rollout loop, one time step at a time.
        """
        self._action_noises = self._noise_distribution.rsample(
            sample_shape=self._sample_shape
        )

        threshold = int(self._num_samples * (1 - self._exploration))
        inherited_samples = mean_action_seq + self._action_noises[:threshold]
        gaussian_samples = torch.cat(
            [inherited_samples, self._action_noises[threshold:]],
            dim=0,
        )
        gaussian_samples = torch.clamp(gaussian_samples, self._u_min, self._u_max)

        samples = gaussian_samples.clone()
        if self._keep_mean_sample:
            samples[0] = mean_action_seq

        return samples

    def forward(
        self,
        state: torch.Tensor,
        info: Dict = {},
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert state.shape == (self._dim_state,)

        if not torch.is_tensor(state):
            state = torch.tensor(state, device=self._device, dtype=self._dtype)
        else:
            if state.device != self._device or state.dtype != self._dtype:
                state = state.to(self._device, self._dtype)

        info = dict(info) if info is not None else {}

        mean_action_seq = self._previous_action_seq.clone().detach()

        mode = self._active_bias_mode(info)
        info["sampling_bias_mode"] = mode

        # Prior mode only overwrites reserved sample slots during rollout.
        self._perturbed_action_seqs = self._sample_action_sequences(
            state,
            info,
            mean_action_seq,
        )

        prior_slot_names = self._select_prior_slot_policies(state, info)
        prior_start, _ = self._prior_slot_bounds()
        prior_count = len(prior_slot_names)
        prior_end = prior_start + prior_count

        self._state_seq_batch[:, 0, :] = state.repeat(self._num_samples, 1)

        for t in range(self._horizon):
            u_t = self._perturbed_action_seqs[:, t, :].clone()

            if (
                mode == "prior"
                and prior_count > 0
                and self._prior_sampler is not None
                and hasattr(self._prior_sampler, "sample_with_policies")
            ):
                prior_states_t = self._state_seq_batch[prior_start:prior_end, t, :]
                prior_actions_t = self._prior_sampler.sample_with_policies(
                    prior_states_t,
                    info,
                    t,
                    prior_slot_names,
                )

                if prior_actions_t is not None:
                    prior_actions_t = prior_actions_t.to(self._device, self._dtype)
                    u_t[prior_start:prior_end, :] = prior_actions_t
                    self._perturbed_action_seqs[prior_start:prior_end, t, :] = prior_actions_t

            self._state_seq_batch[:, t + 1, :] = self._dynamics(
                self._state_seq_batch[:, t, :],
                u_t,
            )

        costs = self._compute_batch_costs(
            self._state_seq_batch,
            self._perturbed_action_seqs,
            info,
        )

        self._last_costs = costs.detach().clone()

        self._update_auto_lambda(costs)

        self._weights = torch.softmax(-costs / self._lambda, dim=0)

        optimal_action_seq = torch.sum(
            self._weights.view(self._num_samples, 1, 1)
            * self._perturbed_action_seqs,
            dim=0,
        )

        if self._auto_lambda == "MPO":
            for _ in range(1):
                self.optimizer.zero_grad()
                temperature = torch.nn.functional.softplus(self.log_temperature)
                cost_logsumexp = torch.logsumexp(-costs / temperature, dim=0)
                loss = temperature * (self._mpo_epsilon + torch.mean(cost_logsumexp))
                loss.backward()
                self.optimizer.step()
            self._lambda = torch.exp(self.log_temperature).item()

        return self._finalize_action_sequence(state, optimal_action_seq)


    def _states_prediction(
        self, state: torch.Tensor, action_seqs: torch.Tensor
    ) -> torch.Tensor:
        state_seqs = torch.zeros(
            action_seqs.shape[0],
            self._horizon + 1,
            self._dim_state,
            device=self._device,
            dtype=self._dtype,
        )
        state_seqs[:, 0, :] = state
        for t in range(self._horizon):
            state_seqs[:, t + 1, :] = self._dynamics(
                state_seqs[:, t, :], action_seqs[:, t, :]
            )
        return state_seqs

    def _compute_ess(self, weights: torch.Tensor) -> float:
        return 1.0 / torch.sum(weights**2).item()

    def _lbps_objective(self, lambda_: float, costs: torch.Tensor) -> float:
        weights = torch.softmax(-costs / lambda_, dim=0)
        ess = self._compute_ess(weights)
        expected_return = -torch.sum(weights * costs).item()
        cost_range = (costs.max() - costs.min()).item()
        penalty = (
            cost_range
            * math.sqrt((1 - self._lbps_delta) / self._lbps_delta)
            / math.sqrt(ess)
        )
        return -(expected_return - penalty)

    def _essps_objective(self, lambda_: float, costs: torch.Tensor) -> float:
        weights = torch.softmax(-costs / lambda_, dim=0)
        ess = self._compute_ess(weights)
        return ess - self._essps_target_ess

    def _savitzky_golay_coeffs(self, window_size: int, poly_order: int) -> torch.Tensor:
        if window_size % 2 == 0 or window_size <= poly_order:
            raise ValueError("window_size must be odd and greater than poly_order.")
        half_window = (window_size - 1) // 2
        indices = torch.arange(
            -half_window, half_window + 1, dtype=self._dtype, device=self._device
        )
        A = torch.vander(indices, N=poly_order + 1, increasing=True)
        pseudo_inverse = torch.linalg.pinv(A)
        coeffs = pseudo_inverse[0]
        return coeffs

    def _apply_savitzky_golay(
        self, y: torch.Tensor, coeffs: torch.Tensor
    ) -> torch.Tensor:
        pad_size = len(coeffs) // 2
        y_padded = torch.cat([y[:pad_size].flip(0), y, y[-pad_size:].flip(0)])
        y_filtered = torch.conv1d(
            y_padded.view(1, 1, -1), coeffs.view(1, 1, -1), padding="valid"
        )
        return y_filtered.view(-1)
