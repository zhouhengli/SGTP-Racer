import time

import numpy as np
import casadi as ca

from players.utils.common import nearest_point
from players.planner.controller.pure_pursuit import PurePursuitPlanner


# =============================================================================
# Racing MPC parameters
# =============================================================================

# Horizon / discretization.
MPC_N = 12
MPC_HORIZON = 1.20
MPC_DT = MPC_HORIZON / MPC_N

MPC_V_MAX = 10.0
MPC_A_MAX = 3.0
MPC_DELTA_MAX = 0.40

MPC_LF = 0.162
MPC_LR = 0.145

# Track-boundary hard constraints.
MPC_VEHICLE_HALF_WIDTH = 0.17
MPC_BOUNDARY_BUFFER = 0.05
MPC_BOUNDARY_START_INDEX = 0
MPC_LAG_BOUND = 1.50

# Hard collision constraints.
MPC_SAFE_DIST = 0.5
MPC_COLLISION_START_INDEX = 1
MPC_COLLISION_BIG_M = 1.0e4

# Progress-state constraints.
MPC_VTHETA_MAX = 12.0
MPC_PROGRESS_WINDOW = 16.4
MPC_TRACK_REPEAT_LAPS = 3
MPC_MIN_INITIAL_GUESS_SPEED = 0.50

# MPCC / racing objective weights.
MPC_W_CONTOUR = 40.0
MPC_W_LAG = 4.0
MPC_W_HEADING = 2.0
MPC_W_SPEED = 0.30
MPC_W_PROGRESS = 20.0
MPC_W_VTHETA = 0.50
MPC_W_ACCEL = 0.05
MPC_W_STEER = 0.10
MPC_W_DACCEL = 0.02
MPC_W_DSTEER = 0.20

# IPOPT options
MPC_IPOPT_PRINT_TIME = 0
MPC_IPOPT_PRINT_LEVEL = 0
MPC_IPOPT_MAX_ITER = 3000
MPC_IPOPT_TOL = 5.0e-5
MPC_IPOPT_ACCEPTABLE_TOL = 5.0e-5
MPC_IPOPT_MAX_CPU_TIME = 5.0
MPC_IPOPT_SB = "yes"

# Fallback/debug behavior.
MPC_USE_DEBUG_VALUE_ON_FAILURE = True


class MPCPlanner:
    """EVO-MPCC style racing MPCC planner.
    https://github.com/zhouhengli/EVO-MPCC
    """

    def __init__(self, conf, map_path, wpt_path, wb, v_scale, max_opps):
        """Initialize the planner, load raceline data, and build the NLP solver."""
        self.conf = conf
        self.map_path = map_path
        self.map_ext = ".png"
        self.wheelbase = float(wb)
        self.max_opps = int(max_opps)

        raw = np.loadtxt(wpt_path, delimiter=";", skiprows=1)
        self.waypoints = np.vstack((
            raw[:, 3],  # x reference/raceline [m]
            raw[:, 4],  # y reference/raceline [m]
            raw[:, 9],  # reference speed [m/s]
            raw[:, 7],  # reference heading/tangent [rad]
            raw[:, 1],  # arc length s [m]
        )).T
        self.waypoints[:, 2] *= float(v_scale)

        # d_right_left[0] = distance from reference line to right boundary [m].
        # d_right_left[1] = distance from reference line to left boundary [m].
        # With lateral contouring error e_c, in-track means:
        #     -d_right <= e_c <= d_left.
        self.d_right_left = np.vstack((raw[:, 5], raw[:, 6])).astype(float)
        self.s_max = float(self.waypoints[-1, 4])

        self.N = int(MPC_N)
        self.duration = float(MPC_HORIZON)
        self.dt = float(MPC_DT)
        self.L = self.wheelbase
        self.v_max = float(MPC_V_MAX)
        self.a_max = float(MPC_A_MAX)
        self.delta_max = float(MPC_DELTA_MAX)

        self.best_traj = None
        self.last_state_traj = None
        self.prev_X = None
        self.prev_U = None
        self.prev_S = None
        self.prev_VS = None
        self.ref_traj = None
        self.debug_ref_traj = None
        self.debug_predicted_traj = None
        self.debug_s_traj = None
        self.debug_last_solver_failed = False

        self.debug_s_current = None

        self.tracker = PurePursuitPlanner(conf, wpt_path, wb=wb)

        self._build_periodic_track_interpolants()
        self.build_solver()

    def _build_periodic_track_interpolants(self):
        """Build CasADi linear interpolants over repeated laps."""
        s = np.asarray(self.waypoints[:, 4], dtype=float)
        x = np.asarray(self.waypoints[:, 0], dtype=float)
        y = np.asarray(self.waypoints[:, 1], dtype=float)
        v = np.asarray(self.waypoints[:, 2], dtype=float)
        psi = np.asarray(self.waypoints[:, 3], dtype=float)
        right = np.asarray(self.d_right_left[0], dtype=float)
        left = np.asarray(self.d_right_left[1], dtype=float)

        # Keep the interpolation grid strictly increasing.
        keep = np.ones(len(s), dtype=bool)
        keep[1:] = np.diff(s) > 1.0e-8
        s = s[keep]
        x = x[keep]
        y = y[keep]
        v = v[keep]
        psi = psi[keep]
        right = right[keep]
        left = left[keep]

        pieces_s = [s]
        pieces_x = [x]
        pieces_y = [y]
        pieces_v = [v]
        pieces_cos = [np.cos(psi)]
        pieces_sin = [np.sin(psi)]
        pieces_right = [right]
        pieces_left = [left]

        for lap in range(1, int(MPC_TRACK_REPEAT_LAPS)):
            pieces_s.append(s[1:] + lap * self.s_max)
            pieces_x.append(x[1:])
            pieces_y.append(y[1:])
            pieces_v.append(v[1:])
            pieces_cos.append(np.cos(psi[1:]))
            pieces_sin.append(np.sin(psi[1:]))
            pieces_right.append(right[1:])
            pieces_left.append(left[1:])

        self.s_ext_np = np.concatenate(pieces_s).astype(float)
        self.x_ext_np = np.concatenate(pieces_x).astype(float)
        self.y_ext_np = np.concatenate(pieces_y).astype(float)
        self.v_ext_np = np.concatenate(pieces_v).astype(float)
        self.cos_ext_np = np.concatenate(pieces_cos).astype(float)
        self.sin_ext_np = np.concatenate(pieces_sin).astype(float)
        self.right_ext_np = np.concatenate(pieces_right).astype(float)
        self.left_ext_np = np.concatenate(pieces_left).astype(float)

        suffix = str(id(self))
        self.track_x_fun = ca.interpolant(f"track_x_{suffix}", "linear", [self.s_ext_np], self.x_ext_np)
        self.track_y_fun = ca.interpolant(f"track_y_{suffix}", "linear", [self.s_ext_np], self.y_ext_np)
        self.track_v_fun = ca.interpolant(f"track_v_{suffix}", "linear", [self.s_ext_np], self.v_ext_np)
        self.track_cos_fun = ca.interpolant(f"track_cos_{suffix}", "linear", [self.s_ext_np], self.cos_ext_np)
        self.track_sin_fun = ca.interpolant(f"track_sin_{suffix}", "linear", [self.s_ext_np], self.sin_ext_np)
        self.track_right_fun = ca.interpolant(f"track_right_{suffix}", "linear", [self.s_ext_np], self.right_ext_np)
        self.track_left_fun = ca.interpolant(f"track_left_{suffix}", "linear", [self.s_ext_np], self.left_ext_np)

    def dynamics(self, x, u):
        """Apply the discrete CG kinematic bicycle model with slip angle beta.
        paper: The kinematic bicycle model: A consistent model for planning feasible trajectories for autonomous vehicles?
        """
        px = x[0]
        py = x[1]
        psi = x[2]
        v = x[3]
        a = u[0]
        delta = u[1]

        beta = ca.atan((MPC_LR / (MPC_LF + MPC_LR)) * ca.tan(delta))
        return ca.vertcat(
            px + v * ca.cos(psi + beta) * self.dt,
            py + v * ca.sin(psi + beta) * self.dt,
            psi + (v / float(MPC_LR)) * ca.tan(beta) * ca.cos(beta) * self.dt,
            v + a * self.dt,
        )

    def _track_frame_expr(self, s_abs):
        """Return the interpolated track frame and bounds at progress s_abs."""
        cx = self.track_x_fun(s_abs)
        cy = self.track_y_fun(s_abs)
        cpsi_cos = self.track_cos_fun(s_abs)
        cpsi_sin = self.track_sin_fun(s_abs)

        # cos/sin are linearly interpolated separately to avoid angle wrapping.
        norm = ca.sqrt(cpsi_cos * cpsi_cos + cpsi_sin * cpsi_sin + 1.0e-9)
        cpsi_cos = cpsi_cos / norm
        cpsi_sin = cpsi_sin / norm

        v_ref = self.track_v_fun(s_abs)
        d_right = self.track_right_fun(s_abs)
        d_left = self.track_left_fun(s_abs)
        return cx, cy, cpsi_cos, cpsi_sin, v_ref, d_right, d_left

    def _contouring_errors_expr(self, x_state, s_abs):
        """Compute contouring, lag, heading, speed, and boundary terms."""
        cx, cy, cpsi_cos, cpsi_sin, v_ref, d_right, d_left = self._track_frame_expr(s_abs)
        dx = x_state[0] - cx
        dy = x_state[1] - cy

        # Reference tangent t = [cos(psi_ref), sin(psi_ref)].
        # Reference normal  n = [-sin(psi_ref), cos(psi_ref)].
        e_contour = -cpsi_sin * dx + cpsi_cos * dy
        e_lag = cpsi_cos * dx + cpsi_sin * dy

        ego_cos = ca.cos(x_state[2])
        ego_sin = ca.sin(x_state[2])
        cos_heading_error = ego_cos * cpsi_cos + ego_sin * cpsi_sin

        return e_contour, e_lag, cos_heading_error, v_ref, d_right, d_left

    def build_solver(self):
        """Build the CasADi Opti problem for the MPCC planner."""
        N = self.N
        self.opti = ca.Opti()

        # Physical state and control semantics.
        self.X = self.opti.variable(4, N + 1)  # [x, y, psi, v]
        self.U = self.opti.variable(2, N)      # [a, delta]

        # MPCC progress state and virtual progress speed.
        self.S = self.opti.variable(1, N + 1)   # absolute progress coordinate [m]
        self.VS = self.opti.variable(1, N)      # virtual ds/dt [m/s]

        self.X0 = self.opti.parameter(4)
        self.S0 = self.opti.parameter(1)

        self.OppXY = [self.opti.parameter(2, N + 1) for _ in range(self.max_opps)]
        self.OppActive = self.opti.parameter(self.max_opps)

        cost = 0.0
        track_margin = float(MPC_VEHICLE_HALF_WIDTH + MPC_BOUNDARY_BUFFER)
        safe2 = float(MPC_SAFE_DIST * MPC_SAFE_DIST)

        self.opti.subject_to(self.X[:, 0] == self.X0)
        self.opti.subject_to(self.S[0, 0] == self.S0)

        self.opti.subject_to(self.opti.bounded(0.0, self.X[3, :], self.v_max))
        self.opti.subject_to(self.opti.bounded(-self.a_max, self.U[0, :], self.a_max))
        self.opti.subject_to(self.opti.bounded(-self.delta_max, self.U[1, :], self.delta_max))
        self.opti.subject_to(self.opti.bounded(0.0, self.VS, float(MPC_VTHETA_MAX)))

        for k in range(N):
            self.opti.subject_to(self.X[:, k + 1] == self.dynamics(self.X[:, k], self.U[:, k]))
            self.opti.subject_to(self.S[0, k + 1] == self.S[0, k] + self.VS[0, k] * self.dt)
            self.opti.subject_to(self.S[0, k] >= self.S0)
            self.opti.subject_to(self.S[0, k] <= self.S0 + float(MPC_PROGRESS_WINDOW))

            e_c, e_lag, cos_epsi, v_ref, _, _ = self._contouring_errors_expr(self.X[:, k], self.S[0, k])

            cost += float(MPC_W_CONTOUR) * e_c * e_c
            cost += float(MPC_W_LAG) * e_lag * e_lag
            cost += float(MPC_W_HEADING) * (1.0 - cos_epsi)
            cost += float(MPC_W_VTHETA) * (self.VS[0, k] - self.X[3, k]) * (self.VS[0, k] - self.X[3, k])
            cost += float(MPC_W_ACCEL) * self.U[0, k] * self.U[0, k]
            cost += float(MPC_W_STEER) * self.U[1, k] * self.U[1, k]

            if k > 0:
                da = self.U[0, k] - self.U[0, k - 1]
                ddelta = self.U[1, k] - self.U[1, k - 1]
                cost += float(MPC_W_DACCEL) * da * da
                cost += float(MPC_W_DSTEER) * ddelta * ddelta

        # we adopt tvc mode of mpcc for overtaking: https://github.com/zhouhengli/EVO-MPCC/blob/main/src/evo-mpcc_planner/src/nonlinear_mpc_casadi/scripts/Nonlinear_MPC.py
        cost += float(MPC_W_SPEED) * (self.X[3, N] - v_ref) * (self.X[3, N] - v_ref)
        self.opti.subject_to(self.S[0, N] >= self.S0)
        self.opti.subject_to(self.S[0, N] <= self.S0 + float(MPC_PROGRESS_WINDOW))
        cost += -float(MPC_W_PROGRESS) * (self.S[0, N] - self.S0)

        # Hard track-boundary constraints in the reference-line Frenet frame.
        for k in range(max(0, int(MPC_BOUNDARY_START_INDEX)), N + 1):
            e_c, e_lag, _, _, d_right, d_left = self._contouring_errors_expr(self.X[:, k], self.S[0, k])
            self.opti.subject_to(e_c >= -d_right + track_margin)
            self.opti.subject_to(e_c <= d_left - track_margin)
            self.opti.subject_to(self.opti.bounded(-float(MPC_LAG_BOUND), e_lag, float(MPC_LAG_BOUND)))

        # Hard collision constraints against predicted opponent positions.
        start_k = min(max(0, int(MPC_COLLISION_START_INDEX)), N)
        for j in range(self.max_opps):
            for k in range(start_k, N + 1):
                dx = self.X[0, k] - self.OppXY[j][0, k]
                dy = self.X[1, k] - self.OppXY[j][1, k]
                dist2 = dx * dx + dy * dy
                self.opti.subject_to(
                    dist2 + (1.0 - self.OppActive[j]) * float(MPC_COLLISION_BIG_M) >= safe2 # decide how many opponents are active (we are using all agents in the game, but we can also use a subset of them)
                )

        self.opti.minimize(cost)
        self.opti.solver(
            "ipopt",
            {
                "print_time": int(MPC_IPOPT_PRINT_TIME),
                "ipopt.print_level": int(MPC_IPOPT_PRINT_LEVEL),
                "ipopt.max_iter": int(MPC_IPOPT_MAX_ITER),
                "ipopt.tol": float(MPC_IPOPT_TOL),
                "ipopt.acceptable_tol": float(MPC_IPOPT_ACCEPTABLE_TOL),
                "ipopt.acceptable_iter": 15,
                "ipopt.max_cpu_time": float(MPC_IPOPT_MAX_CPU_TIME),
                "ipopt.sb": str(MPC_IPOPT_SB),
            },
        )

    def _nearest_s(self, x, y):
        """Return nearest one-lap progress as the current NLP S0 without cross-call unwrapping."""
        _, _, t, idx = nearest_point(np.array([x, y], dtype=float), self.waypoints[:, :2])
        idx_next = min(idx + 1, len(self.waypoints) - 1)
        s0 = self.waypoints[idx, 4]
        s1 = self.waypoints[idx_next, 4]
        s_curr = float(s0 + t * (s1 - s0))
        s_curr = float(np.clip(s_curr, self.s_ext_np[0], self.s_ext_np[-1]))
        self.debug_s_current = s_curr
        return s_curr

    def build_reference(self, x, y, velocity):
        """Build a nominal trajectory for warm-starting, debugging, and metrics."""
        s_curr = self._nearest_s(x, y)
        v_nom = float(velocity)
        v_nom = max(float(MPC_MIN_INITIAL_GUESS_SPEED), min(self.v_max, v_nom))

        ref = np.zeros((4, self.N + 1), dtype=float)
        for k in range(self.N + 1):
            s_k = s_curr + v_nom * k * self.dt
            s_k = np.clip(s_k, self.s_ext_np[0], self.s_ext_np[-1])
            ref[0, k] = np.interp(s_k, self.s_ext_np, self.x_ext_np)
            ref[1, k] = np.interp(s_k, self.s_ext_np, self.y_ext_np)
            c = np.interp(s_k, self.s_ext_np, self.cos_ext_np)
            s = np.interp(s_k, self.s_ext_np, self.sin_ext_np)
            ref[2, k] = np.arctan2(s, c)
            ref[3, k] = np.interp(s_k, self.s_ext_np, self.v_ext_np)
        return ref, s_curr

    def _opp_xy_from_traj(self, traj):
        """Convert an opponent predicted trajectory into a fixed horizon xy array."""
        xy = np.zeros((2, self.N + 1), dtype=float)
        arr = np.asarray(traj, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] == 0:
            return xy
        last = min(self.N + 1, arr.shape[0])
        xy[:, :last] = arr[:last, :2].T
        if last < self.N + 1 and last > 0:
            xy[:, last:] = xy[:, last - 1:last]
        return xy

    def _opp_xy_from_pose(self, pose):
        """Convert a static opponent pose into a fixed horizon xy array."""
        xy = np.zeros((2, self.N + 1), dtype=float)
        xy[0, :] = float(pose[0])
        xy[1, :] = float(pose[1])
        return xy

    def _set_opponent_parameters(self, opp_pred_trajs, opp_poses):
        """Set opponent activity flags and predicted xy parameters for the solver."""
        active = np.zeros(self.max_opps, dtype=float)
        slots = [np.zeros((2, self.N + 1), dtype=float) for _ in range(self.max_opps)]

        if opp_pred_trajs is not None and len(opp_pred_trajs) > 0:
            for j, traj in enumerate(opp_pred_trajs[:self.max_opps]):
                slots[j] = self._opp_xy_from_traj(traj)
                active[j] = 1.0
        elif opp_poses is not None and len(opp_poses) > 0:
            for j, pose in enumerate(opp_poses[:self.max_opps]):
                slots[j] = self._opp_xy_from_pose(pose)
                active[j] = 1.0

        self.opti.set_value(self.OppActive, active)
        for j in range(self.max_opps):
            self.opti.set_value(self.OppXY[j], slots[j])

    def _default_progress_guess(self, x0, s_curr):
        """Create a monotone fallback progress warm start from current speed."""
        vs_guess = max(float(MPC_MIN_INITIAL_GUESS_SPEED), float(x0[3]))
        vs_guess = min(vs_guess, float(MPC_VTHETA_MAX))
        s_guess = s_curr + np.linspace(0.0, vs_guess * self.duration, self.N + 1)
        s_guess = np.minimum(s_guess, s_curr + float(MPC_PROGRESS_WINDOW))
        s_guess = np.clip(s_guess, float(self.s_ext_np[0]), float(self.s_ext_np[-1]))
        return s_guess.reshape(1, -1)

    def _set_initial_guess(self, x0, ref, s_curr):
        """Warm-start the NLP by shifting the previous solution and holding its terminal sample."""
        # ---- Physical state X ------------------------------------------------
        if self.prev_X is not None and np.shape(self.prev_X) == (4, self.N + 1):
            X_guess = np.empty((4, self.N + 1), dtype=float)
            X_guess[:, :-1] = self.prev_X[:, 1:]
            X_guess[:, -1] = X_guess[:, -2]  # hold previous terminal state
            X_guess[:, 0] = x0
            X_guess[3, :] = np.clip(X_guess[3, :], 0.0, self.v_max)
            self.opti.set_initial(self.X, X_guess)
        else:
            init_X = ref.copy()
            init_X[:, 0] = x0
            init_X[3, :] = np.clip(init_X[3, :], 0.0, self.v_max)
            self.opti.set_initial(self.X, init_X)

        # ---- Control U -------------------------------------------------------
        if self.prev_U is not None and np.shape(self.prev_U) == (2, self.N):
            U_guess = np.empty((2, self.N), dtype=float)
            U_guess[:, :-1] = self.prev_U[:, 1:]
            U_guess[:, -1] = self.prev_U[:, -1]  # hold previous terminal input
            U_guess[0, :] = np.clip(U_guess[0, :], -self.a_max, self.a_max)
            U_guess[1, :] = np.clip(U_guess[1, :], -self.delta_max, self.delta_max)
            self.opti.set_initial(self.U, U_guess)
        else:
            self.opti.set_initial(self.U, np.zeros((2, self.N), dtype=float))

        # ---- Progress S / virtual speed VS ----------------------------------
        if self.prev_S is not None and np.shape(self.prev_S) == (1, self.N + 1):
            S_guess = np.empty((1, self.N + 1), dtype=float)
            S_guess[:, :-1] = self.prev_S[:, 1:]
            S_guess[:, -1] = S_guess[:, -2]  # hold previous terminal progress

            # Re-align the shifted trajectory to the current S0.
            S_guess += float(s_curr) - float(S_guess[0, 0])
            S_guess[0, :] = np.maximum.accumulate(S_guess[0, :])
            S_guess[0, :] = np.minimum(S_guess[0, :], float(s_curr) + float(MPC_PROGRESS_WINDOW))
            S_guess[0, :] = np.clip(S_guess[0, :], float(self.s_ext_np[0]), float(self.s_ext_np[-1]))
            S_guess[0, 0] = float(s_curr)
        else:
            S_guess = self._default_progress_guess(x0, s_curr)

        self.opti.set_initial(self.S, S_guess)

        # Keep VS consistent with the S warm start.  Since the terminal S sample
        # is held, the last VS initial value may be zero; this is acceptable for
        # an initial guess and remains within the imposed VS bounds.
        VS_guess = np.diff(S_guess[0, :]) / self.dt
        VS_guess = np.clip(VS_guess, 0.0, float(MPC_VTHETA_MAX)).reshape(1, -1)
        self.opti.set_initial(self.VS, VS_guess)

    def _fallback_solution(self, x0, ref, s_curr):
        """Return a conservative nominal solution when the NLP solution is unavailable."""
        X_sol = ref.copy()
        X_sol[:, 0] = x0
        U_sol = np.zeros((2, self.N), dtype=float)
        s_sol = s_curr + np.linspace(
            0.0,
            max(float(MPC_MIN_INITIAL_GUESS_SPEED), float(x0[3])) * self.duration,
            self.N + 1,
        )
        vs_sol = np.full(
            (1, self.N),
            max(float(MPC_MIN_INITIAL_GUESS_SPEED), float(x0[3])),
            dtype=float,
        )
        return X_sol, U_sol, s_sol.reshape(1, -1), vs_sol

    def plan(self, pose_x, pose_y, pose_theta, velocity, opp_poses, opp_pred_trajs=None):
        """Solve one MPCC planning step and return the predicted trajectory and headings."""
        x0 = np.array([pose_x, pose_y, pose_theta, velocity], dtype=float)
        ref, s_curr = self.build_reference(pose_x, pose_y, velocity)
        self.ref_traj = ref
        self.debug_ref_traj = ref

        self.opti.set_value(self.X0, x0)
        self.opti.set_value(self.S0, np.array([s_curr], dtype=float))
        self._set_opponent_parameters(opp_pred_trajs, opp_poses)
        self._set_initial_guess(x0, ref, s_curr)

        self.debug_last_solver_failed = False
        try:
            sol = self.opti.solve()
            X_sol = sol.value(self.X)
            U_sol = sol.value(self.U)
            S_sol = sol.value(self.S)
            VS_sol = sol.value(self.VS)
        except RuntimeError:
            self.debug_last_solver_failed = True
            if bool(MPC_USE_DEBUG_VALUE_ON_FAILURE):
                try:
                    X_sol = self.opti.debug.value(self.X)
                    U_sol = self.opti.debug.value(self.U)
                    S_sol = self.opti.debug.value(self.S)
                    VS_sol = self.opti.debug.value(self.VS)
                except Exception:
                    X_sol, U_sol, S_sol, VS_sol = self._fallback_solution(x0, ref, s_curr)
            else:
                X_sol, U_sol, S_sol, VS_sol = self._fallback_solution(x0, ref, s_curr)

        self.prev_X = X_sol
        self.prev_U = U_sol
        self.prev_S = S_sol
        self.prev_VS = VS_sol
        self.debug_predicted_traj = X_sol
        self.debug_s_traj = S_sol
        self.last_state_traj = X_sol.T.copy()

        traj = np.zeros((self.N + 1, 4), dtype=float)
        traj[:, 0] = X_sol[0, :]
        traj[:, 1] = X_sol[1, :]
        traj[:, 2] = X_sol[3, :]
        traj[:, 3] = X_sol[2, :]
        heading = X_sol[2, :]
        self.best_traj = traj
        return traj, heading
