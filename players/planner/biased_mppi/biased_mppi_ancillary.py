import math
from typing import Dict, List, Optional

import torch

from .biased_mppi_post_selection import BiasedMPPIPostSelectionPlanner


class RacingAncillaryControllers:
    """Closed-loop ancillary controllers for Biased-MPPI.

    Each prior slot is bound to one controller name for the whole MPPI call.
    During rollout, the controller is queried at each time step with the
    slot's current state and returns a single action [accel, steer].
    """

    def __init__(self, planner):
        self.p = planner

    def _heading_error(self, a, b):
        return torch.atan2(torch.sin(a - b), torch.cos(a - b))

    def _opponent_relevant(
        self,
        state: torch.Tensor,
        opp_pose: Optional[torch.Tensor],
    ) -> bool:
        if opp_pose is None:
            return False
        dx = opp_pose[0] - state[0]
        dy = opp_pose[1] - state[1]
        dist = torch.sqrt(dx * dx + dy * dy)
        heading_to_opp = torch.atan2(dy, dx)
        rel = torch.abs(self._heading_error(heading_to_opp, state[2]))
        return bool(
            (dist < self.p.opp_bias_distance).item()
            and (rel < self.p.opp_bias_fov).item()
        )

    def select_policies(
        self,
        state: torch.Tensor,
        info: Dict,
        budget: int,
    ) -> List[str]:
        """Choose which ancillary controllers occupy the reserved prior slots.

        This is called once per MPPI solve, using the initial state.
        """
        if budget <= 0:
            return []

        opp_pose = info.get("opp_pose", None)

        candidates: List[str] = []
        candidates.append("track_reference")
        candidates.append("recover_align")
        candidates.append("brake_follow")

        if self._opponent_relevant(state, opp_pose):
            candidates.append("opp_brake_or_evade")
            candidates.append("evade_left")
            candidates.append("evade_right")
        else:
            candidates.append("fast_follow")
            candidates.append("evade_left")
            candidates.append("evade_right")

        return candidates[:budget]

    def sample_with_policies(
        self,
        states: torch.Tensor,
        info: Dict,
        t: int,
        slot_policies: List[str],
    ) -> Optional[torch.Tensor]:
        """Return one action per reserved prior slot at rollout step t.
        """
        if states.ndim == 1:
            states = states.unsqueeze(0)

        if states.shape[0] == 0 or len(slot_policies) == 0:
            return None

        actions: List[torch.Tensor] = []
        for i, policy_name in enumerate(slot_policies):
            actions.append(self._policy_step(policy_name, states[i], info, t))

        out = torch.stack(actions, dim=0)
        return torch.clamp(out, self.p._u_min_torch, self.p._u_max_torch)

    def _policy_step(
        self,
        policy_name: str,
        state: torch.Tensor,
        info: Dict,
        t: int,
    ) -> torch.Tensor:
        if policy_name == "track_reference":
            return self.track_reference_step(state, info["ref_traj"], t)
        if policy_name == "fast_follow":
            return self.fast_follow_step(state, info["ref_traj"], t)
        if policy_name == "recover_align":
            return self.recover_align_step(state, info["ref_traj"], t)
        if policy_name == "brake_follow":
            return self.brake_follow_step(state, info["ref_traj"], t)
        if policy_name == "evade_left":
            return self.evade_step(state, info["ref_traj"], t, direction=1.0)
        if policy_name == "evade_right":
            return self.evade_step(state, info["ref_traj"], t, direction=-1.0)
        if policy_name == "opp_brake_or_evade":
            opp_pose = info.get("opp_pose", None)
            if opp_pose is None:
                return self.brake_follow_step(state, info["ref_traj"], t)
            return self.opp_brake_or_evade_step(
                state,
                info["ref_traj"],
                opp_pose,
                t,
            )

        raise ValueError(f"Unknown ancillary policy: {policy_name}")

    def track_reference_step(
        self,
        state: torch.Tensor,
        ref: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        ref_idx = min(t + 1, self.p.N)
        rx, ry, rpsi, rv = (
            ref[0, ref_idx],
            ref[1, ref_idx],
            ref[2, ref_idx],
            ref[3, ref_idx],
        )
        heading_to_ref = torch.atan2(ry - state[1], rx - state[0])
        psi_err = self._heading_error(heading_to_ref, state[2])
        path_err = self._heading_error(rpsi, state[2])
        steer = self.p.k_track_pos * psi_err + self.p.k_track_heading * path_err
        accel = self.p.k_track_speed * (rv - state[3])
        return torch.stack([accel, steer])

    def fast_follow_step(
        self,
        state: torch.Tensor,
        ref: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        u = self.track_reference_step(state, ref, t).clone()
        u[0] = u[0] + self.p.fast_follow_accel_bias
        return u

    def recover_align_step(
        self,
        state: torch.Tensor,
        ref: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        ref_idx = min(t + 1, self.p.N)
        rpsi = ref[2, ref_idx]
        psi_err = self._heading_error(rpsi, state[2])
        steer = self.p.k_recover_heading * psi_err
        accel = state.new_tensor(self.p.recover_accel_bias)
        return torch.stack([accel, steer])

    def brake_follow_step(
        self,
        state: torch.Tensor,
        ref: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        ref_idx = min(t + 1, self.p.N)
        rx, ry, rpsi = ref[0, ref_idx], ref[1, ref_idx], ref[2, ref_idx]
        heading_to_ref = torch.atan2(ry - state[1], rx - state[0])
        steer = (
            0.6 * self.p.k_track_pos * self._heading_error(heading_to_ref, state[2])
            + 0.8 * self.p.k_track_heading * self._heading_error(rpsi, state[2])
        )
        accel = state.new_tensor(self.p.brake_accel)
        return torch.stack([accel, steer])

    def evade_step(
        self,
        state: torch.Tensor,
        ref: torch.Tensor,
        t: int,
        direction: float,
    ) -> torch.Tensor:
        ref_idx = min(t + 1, self.p.N)
        rpsi = ref[2, ref_idx]
        steer = (
            self.p.k_evade_heading * self._heading_error(rpsi, state[2])
            + state.new_tensor(direction * self.p.evade_steer_bias)
        )
        accel = state.new_tensor(self.p.evade_accel)
        return torch.stack([accel, steer])

    def opp_brake_or_evade_step(
        self,
        state: torch.Tensor,
        ref: torch.Tensor,
        opp_pose: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        dx = opp_pose[0] - state[0]
        dy = opp_pose[1] - state[1]
        side = torch.sign(torch.sin(torch.atan2(dy, dx) - state[2]))
        if side.item() == 0.0:
            side = state.new_tensor(1.0)
        return self.evade_step(
            state,
            ref,
            t,
            direction=float(-side.item()),
        )


__all__ = ["RacingAncillaryControllers"]


# ---------------------------------------------------------------------------
# Fixed ancillary-prior settings
# ---------------------------------------------------------------------------
# Set this to 0 to force pure MPPI sampling.  In that case the core receives
# sampling_bias_mode='none', prior_sampler=None, and num_prior_samples=0.
ANCILLARY_NUM_SAMPLES = 6
ANCILLARY_NOISE_SCALE = 0.0 # set to 0 to disable noise around ancillary samples
ANCILLARY_KEEP_MEAN_SAMPLE = True

ANCILLARY_K_TRACK_POS = 1.4
ANCILLARY_K_TRACK_HEADING = 1.0
ANCILLARY_K_TRACK_SPEED = 2.0

ANCILLARY_K_RECOVER_HEADING = 1.6
ANCILLARY_K_EVADE_HEADING = 0.8

ANCILLARY_FAST_FOLLOW_ACCEL_BIAS = 0.8
ANCILLARY_RECOVER_ACCEL_BIAS = -1.0
ANCILLARY_BRAKE_ACCEL = -4.5
ANCILLARY_EVADE_ACCEL = -1.5
ANCILLARY_EVADE_STEER_BIAS = 0.2

ANCILLARY_OPP_BIAS_DISTANCE = 4.5
ANCILLARY_OPP_BIAS_FOV_DEG = 60.0


class AncillaryBiasedMPPIPlanner(BiasedMPPIPostSelectionPlanner):
    """MPPI planner with hand-written ancillary prior samples.

    The base planner is unchanged.  This subclass only decides how many prior
    slots to reserve and supplies RacingAncillaryControllers to the MPPI core.
    """

    def __init__(
        self,
        args,
        conf,
        map_path,
        wpt_path,
        wb=None,
        v_scale=None,
        ocp_conf=None,
        game_block_conf=None,
        biased_type=None,
    ):
        self.num_prior_samples = max(0, int(ANCILLARY_NUM_SAMPLES))
        self.prior_noise_scale = float(ANCILLARY_NOISE_SCALE)
        self.ancillary_sampler = RacingAncillaryControllers(self)

        # Gains used by RacingAncillaryControllers.
        self.k_track_pos = float(ANCILLARY_K_TRACK_POS)
        self.k_track_heading = float(ANCILLARY_K_TRACK_HEADING)
        self.k_track_speed = float(ANCILLARY_K_TRACK_SPEED)
        self.k_recover_heading = float(ANCILLARY_K_RECOVER_HEADING)
        self.k_evade_heading = float(ANCILLARY_K_EVADE_HEADING)

        self.fast_follow_accel_bias = float(ANCILLARY_FAST_FOLLOW_ACCEL_BIAS)
        self.recover_accel_bias = float(ANCILLARY_RECOVER_ACCEL_BIAS)
        self.brake_accel = float(ANCILLARY_BRAKE_ACCEL)
        self.evade_accel = float(ANCILLARY_EVADE_ACCEL)
        self.evade_steer_bias = float(ANCILLARY_EVADE_STEER_BIAS)

        self.opp_bias_distance = float(ANCILLARY_OPP_BIAS_DISTANCE)
        self.opp_bias_fov = math.radians(float(ANCILLARY_OPP_BIAS_FOV_DEG))

        # Build once as a normal post-selection planner, then rebuild with
        # ancillary prior enabled.  This avoids putting ancillary logic into
        # the base planner constructor.
        super().__init__(
            args,
            conf,
            map_path,
            wpt_path,
            wb=wb,
            v_scale=v_scale,
            ocp_conf=ocp_conf,
            game_block_conf=game_block_conf,
            biased_type='none',
        )

        self.mppi_bias_mode = 'ancillary'
        self.keep_mean_sample = bool(ANCILLARY_KEEP_MEAN_SAMPLE)
        self.num_prior_samples = self._fixed_num_prior_samples()
        self._build_mppi_core()

    def _fixed_num_prior_samples(self) -> int:
        return min(max(0, int(ANCILLARY_NUM_SAMPLES)), int(self.num_samples))

    def _effective_sampling_bias_mode(self, requested_mode=None) -> str:
        mode = str(requested_mode or self.mppi_bias_mode or 'none').lower()

        if mode == 'none':
            return 'none'

        if mode in {'ancillary', 'prior'}:
            return 'prior' if self.num_prior_samples > 0 else 'none'

        raise ValueError("Ancillary planner supports only bias_mode='none' or 'ancillary'.")

    def _prior_sampling_config(self, mode: str):
        if mode != 'prior' or self.num_prior_samples <= 0:
            return None, 0, 0.0

        return self.ancillary_sampler, self.num_prior_samples, self.prior_noise_scale

    def set_parameters(self, parameters):
        super().set_parameters(parameters)
        self.num_prior_samples = self._fixed_num_prior_samples()
        self._build_mppi_core()


__all__ = [
    'RacingAncillaryControllers',
    'AncillaryBiasedMPPIPlanner',
]
