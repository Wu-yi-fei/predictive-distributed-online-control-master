import os
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import control
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
import warnings

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GridEnv:

    def __init__(
        self,
        n_buses: int,
        edges: List[Tuple[int, int]],
        impedances: Dict[Tuple[int, int], Tuple[float, float]],
        v_feeder: float,
        v_ref: float,
        kappa: int,
        T: int,
        p_matrix: np.ndarray,
        q_n_matrix: np.ndarray,
    ):
        self.N = n_buses
        self.edges = edges
        self.impedances = impedances
        self.v0 = v_feeder
        self.kappa = kappa
        self.T = T
        self.p_matrix = p_matrix
        self.q_n_matrix = q_n_matrix

        self.system_time = 0
        self.v_ref = np.full(self.N, v_ref)

        self._build_topology_helpers()
        self._build_matrices()
        self._build_kappa_neighbors()

        self.x = np.zeros(self.N)
        self.v_actual = self.v_ref.copy()
        self.q_c_t = np.zeros(self.N)
        self.v_par_prev = np.zeros(self.N)

        self.w_profile_gt = self.get_ground_truth_w_profile()

    def _build_topology_helpers(self):
        self.parent_of = {}
        for parent, child in self.edges:
            self.parent_of[child] = parent

        self.paths = {}
        for i in range(1, self.N + 1):
            path = []
            curr = i
            while curr != 0:
                parent = self.parent_of.get(curr, 0)  # treat missing parent as feeder 0
                if parent == 0:
                    # Include feeder edge if impedance is provided; else stop.
                    if (parent, curr) in self.impedances:
                        path.append((parent, curr))
                    break
                if (parent, curr) not in self.impedances:
                    raise KeyError(f"Impedance for edge ({parent}, {curr}) not found.")
                path.append((parent, curr))
                curr = parent
            self.paths[i] = list(reversed(path))

    def _build_matrices(self):
        self.R = np.zeros((self.N, self.N))
        self.X = np.zeros((self.N, self.N))
        path_sets = {i: set(self.paths[i]) for i in range(1, self.N + 1)}
        for i in range(1, self.N + 1):
            for j in range(1, self.N + 1):
                common_path = path_sets[i].intersection(path_sets[j])
                r_sum, x_sum = 0.0, 0.0
                for edge in common_path:
                    r, x = self.impedances[edge]
                    r_sum += r
                    x_sum += x
                self.R[i - 1, j - 1] = 2.0 * r_sum
                self.X[i - 1, j - 1] = 2.0 * x_sum

    def _build_kappa_neighbors(self):
        adj_list = {i: set() for i in range(1, self.N + 1)}
        for parent, child in self.edges:
            if parent != 0 and child >= 1:
                adj_list[parent].add(child)
                adj_list[child].add(parent)
        self.comm_adj_list = adj_list
        self.neighborhoods = {}
        for i in range(1, self.N + 1):
            self.neighborhoods[i] = set()
            queue = deque([(i, 0)])
            visited = {i}
            while queue:
                curr_node, dist = queue.popleft()
                if dist <= self.kappa:
                    self.neighborhoods[i].add(curr_node)
                    if dist < self.kappa:
                        for nei in adj_list[curr_node]:
                            if nei not in visited:
                                visited.add(nei)
                                queue.append((nei, dist + 1))

    def reset(self) -> np.ndarray:
        self.x = np.zeros(self.N)
        self.v_actual = self.v_ref.copy()
        self.q_c_t = np.zeros(self.N)
        self.v_par_prev = np.zeros(self.N)
        self.system_time = 0
        return self.x

    def step(self, u_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert self.system_time < self.T
        v_par_t = self.R @ self.p_matrix[self.system_time] + self.X @ self.q_n_matrix[self.system_time]
        w_t = v_par_t - self.v_par_prev
        x_next = self.x + self.X @ u_t + w_t
        self.x = x_next
        self.v_actual = self.v_ref + self.x
        self.v_par_prev = v_par_t
        self.q_c_t = self.q_c_t + u_t
        self.system_time += 1
        return self.x, w_t

    def get_ground_truth_w_profile(self) -> np.ndarray:
        w_profile_gt = np.zeros((self.T, self.N))
        v_par_prev_local = np.zeros(self.N)
        for t in range(self.T):
            p_t = self.p_matrix[t]
            q_n_t = self.q_n_matrix[t]
            v_par_t = self.R @ p_t + self.X @ q_n_t
            w_profile_gt[t] = v_par_t - v_par_prev_local
            v_par_prev_local = v_par_t
        return w_profile_gt



class LatentPredictor(nn.Module):
    """Predicts future latent variables s_{t+1...t+k} from s_{t-L...t}."""

    def __init__(self, k_latent: int, k_horizon: int, lstm_dim: int = 128):
        super().__init__()
        self.k_latent = k_latent
        self.k_horizon = k_horizon
        self.lstm = nn.LSTM(input_size=k_latent, hidden_size=lstm_dim, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(lstm_dim, 128),
            nn.ReLU(),
            nn.Linear(128, k_latent * k_horizon),
        )

    def forward(self, s_history: torch.Tensor) -> torch.Tensor:
        # s_history: (B, L, k_latent)
        _, (hn, _) = self.lstm(s_history)
        s_pred = self.decoder(hn[0]).view(-1, self.k_horizon, self.k_latent)
        return s_pred


class MixingNetwork(nn.Module):
    """xi(s; theta): map latent space s to disturbance space w."""

    def __init__(self, k_latent: int, n_buses: int):
        super().__init__()
        self.theta_layers = nn.Sequential(
            nn.Linear(k_latent, 64),
            nn.ReLU(),
            nn.Linear(64, n_buses),
        )

    def forward(self, s_filtered: torch.Tensor) -> torch.Tensor:
        return self.theta_layers(s_filtered)


def train_ica_predictor(
    env: GridEnv,
    k_latent: int,
    k_horizon: int,
    lookback_L: int,
    num_epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 16,
):
    """Train latent predictor and mixing network using FastICA-transformed w."""
    w_profile = env.get_ground_truth_w_profile()
    transformer = FastICA(n_components=k_latent, random_state=42, whiten="unit-variance", max_iter=2000, tol=1e-4)
    s_profile = transformer.fit_transform(w_profile)

    X_s, y_s = [], []
    for t in range(lookback_L, len(s_profile) - k_horizon):
        X_s.append(s_profile[t - lookback_L : t])
        y_s.append(s_profile[t : t + k_horizon])

    X_train = torch.tensor(np.array(X_s)).float()
    y_train = torch.tensor(np.array(y_s)).float()
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    predictor = LatentPredictor(k_latent, k_horizon).to(device)
    optimizer_p = optim.Adam(predictor.parameters(), lr=lr)
    criterion = nn.MSELoss()
    predictor.train()
    for _ in range(num_epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer_p.zero_grad()
            loss = criterion(predictor(bx), by)
            loss.backward()
            optimizer_p.step()
    predictor.eval()

    mixer = MixingNetwork(k_latent, env.N).to(device)
    optimizer_m = optim.Adam(mixer.parameters(), lr=lr)
    s_tensor = torch.tensor(s_profile, dtype=torch.float32).to(device)
    w_tensor = torch.tensor(w_profile, dtype=torch.float32).to(device)
    for _ in range(num_epochs):
        optimizer_m.zero_grad()
        loss = criterion(mixer(s_tensor), w_tensor)
        loss.backward()
        optimizer_m.step()

    return predictor, mixer, transformer



class LocalizedLearningController:
    def __init__(
        self,
        env: GridEnv,
        horizon: int,
        predictor: LatentPredictor,
        mixer: MixingNetwork,
        transformer: FastICA,
        k_latent: int,
        b_context: int,
        lam_lr: float = 0.005,
        lam_self_lr: float = 0.05,
        controllable_buses: Optional[List[int]] = None,
        alpha_cost: float = 0.2,
        beta_cost: float = 0.2,
        delta_cost: float = 1e-4,
        exp_cost_x: float = 0.01,
        exp_cost_u: float = 0.005,
        poly_cost_x: float = 0.01,
        poly_cost_u: float = 0.01,
        gamma: float = 0.05,
        mpc_lr: float = 0.01,
        mpc_iters: int = 20,
        mpc_u_max: float = 0.5,
        verbose: bool = True,
        forecast_noise_pv: float = 0.02,
        forecast_noise_wind: float = 0.02,
        forecast_noise_ev: float = 0.0,
        forecast_noise_dist: str = "gaussian",
    ):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.A = np.identity(self.env.N)
        self.B = self.env.X
        self.Q = np.identity(self.env.N)
        self.R = np.identity(self.env.N)
        self.T = self.env.T

        self.alpha_cost = alpha_cost
        self.beta_cost = beta_cost
        self.delta_cost = delta_cost
        self.exp_cost_x = exp_cost_x
        self.exp_cost_u = exp_cost_u
        self.poly_cost_x = poly_cost_x
        self.poly_cost_u = poly_cost_u
        self.gamma = gamma
        self.mpc_lr = mpc_lr
        self.mpc_iters = mpc_iters
        self.mpc_u_max = float(abs(mpc_u_max))
        self.verbose = verbose

        self.horizon = horizon
        self.predictor = predictor.to(self.device)
        self.predictor.eval()
        self.mixer = mixer.to(self.device)
        self.transformer = transformer
        self.k_latent = k_latent
        self.b_context = b_context
        # Per-bus latent gating (lambda) for OCL; shape (N, k_latent)
        self.lam_t = np.ones((self.env.N, self.k_latent), dtype=np.float32)
        self.lam_lr = lam_lr
        # Global self-tuning lambda (DISC-style)
        self.lam_self = np.ones(self.k_latent, dtype=np.float32)
        self.lam_self_lr = lam_self_lr
        if controllable_buses is None:
            self.controllable_buses = list(range(1, self.env.N + 1))
        else:
            # keep unique and sorted for deterministic behavior
            self.controllable_buses = sorted(set(controllable_buses))

        self.p_matrix = self.env.p_matrix
        self.q_n_matrix = self.env.q_n_matrix
        # Forecast-only noise amplitudes (do not affect ground truth p/q)
        self.forecast_noise_pv = float(abs(forecast_noise_pv))
        self.forecast_noise_wind = float(abs(forecast_noise_wind))
        self.forecast_noise_ev = float(abs(forecast_noise_ev))
        self.forecast_noise_dist = forecast_noise_dist

        # Bus index mapping for forecast policies (0-based indices)
        bus_map = getattr(self.env, "bus_id_map", {i + 1: i for i in range(self.env.N)})
        self.pv_bus_idx = [bus_map[b] for b in [12, 45, 44, 46, 49] if b in bus_map]
        self.wind_bus_idx = [bus_map[b] for b in [15, 14, 9, 8, 4, 6, 5, 7] if b in bus_map]
        self.ev_bus_idx = [bus_map[b] for b in [2, 17, 20, 21, 22, 23, 24] if b in bus_map]

        # Riccati solution P for terminal cost; fall back to Q if DARE fails
        try:
            self.P, _, _ = control.dare(self.A, self.B, self.Q, self.R)
        except Exception as exc:  # pragma: no cover - safety fallback
            warnings.warn(f"DARE failed ({exc}); using identity as terminal cost.")
            self.P = self.Q.copy()
        self.Q_torch = torch.tensor(self.Q, dtype=torch.float32, device=self.device)
        self.R_torch = torch.tensor(self.R, dtype=torch.float32, device=self.device)
        self.X_torch = torch.tensor(self.env.X, dtype=torch.float32, device=self.device)
        self.P_torch = torch.tensor(self.P, dtype=torch.float32, device=self.device)

        # Local gains per bus (as in script_new) for closed-form MPC action
        self.sub_mats = self._compute_sub_matrices()

    def _truncate(self, M: np.ndarray, nb: Set[int]) -> np.ndarray:
        M_t = M.copy()
        for j in range(1, M.shape[0] + 1):
            if j not in nb:
                M_t[j - 1, :] = 0.0
        return M_t

    def _compute_sub_matrices(self):
        subs = []
        for i in range(1, self.env.N + 1):
            A_tr = self._truncate(np.eye(self.env.N), self.env.neighborhoods[i])
            B_tr = self._truncate(self.env.X, self.env.neighborhoods[i])
            P_loc, _, _ = control.dare(A_tr, B_tr, self.Q, self.R)
            G = np.linalg.inv(self.R + B_tr.T @ P_loc @ B_tr) @ B_tr.T
            F = A_tr - B_tr @ G @ P_loc @ A_tr
            subs.append(
                {
                    "P": P_loc,
                    "G": G,
                    "F_powers": [np.linalg.matrix_power(F, j) for j in range(self.horizon + 1)],
                }
            )
        return subs

    # ---- cost functions ----
    def _smooth_abs_torch(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(z * z + self.delta_cost)

    def _stage_cost_torch(self, x_vec: torch.Tensor, u_vec: torch.Tensor) -> torch.Tensor:
        quad_x = 0.5 * torch.dot(x_vec, torch.matmul(self.Q_torch, x_vec))
        quad_u = 0.5 * torch.dot(u_vec, torch.matmul(self.R_torch, u_vec))
        return quad_x + quad_u 

    def _terminal_cost_torch(self, x_vec: torch.Tensor) -> torch.Tensor:
        quad_x = 0.5 * torch.dot(x_vec, torch.matmul(self.Q_torch, x_vec))
        return quad_x

    def _stage_cost_numpy(self, x_vec: np.ndarray, u_vec: np.ndarray) -> float:
        quad_x = 0.5 * float(np.dot(x_vec, self.Q @ x_vec))
        quad_u = 0.5 * float(np.dot(u_vec, self.R @ u_vec))
        return float(quad_x + quad_u)

    # ---- Lambda update on latent space using windowed weighted error ----
    def _update_lambda_on_s_window(self, w_hist_window: np.ndarray, s_pred_window: np.ndarray):

        L = w_hist_window.shape[0]
        weights = torch.tensor([(1 - self.gamma) ** (idx) for idx in range(L)], dtype=torch.float32, device=self.device)
        s_tensor = torch.tensor(s_pred_window, dtype=torch.float32, device=self.device)  # (L, k_latent)
        w_true_all = torch.tensor(w_hist_window, dtype=torch.float32, device=self.device)  # (L, N)

        new_lam = self.lam_t.copy()
        for i_bus in range(self.env.N):
            lam_tensor = torch.tensor(self.lam_t[i_bus], requires_grad=True, dtype=torch.float32, device=self.device)
            s_filtered = lam_tensor * s_tensor  # (L, k_latent)
            w_est = self.mixer(s_filtered)  # (L, N)
            w_true = w_true_all[:, i_bus]
            w_est_bus = w_est[:, i_bus]
            per_step = torch.abs(w_est_bus - w_true)
            weighted_sum = torch.sum(weights * per_step)
            loss = weighted_sum * weighted_sum
            loss.backward()
            if lam_tensor.grad is not None:
                grad = lam_tensor.grad.detach().cpu().numpy()
                new_lam[i_bus] = np.clip(self.lam_t[i_bus] - self.lam_lr * grad, 0.0, 1.0)
        self.lam_t = new_lam

    # ---- DISC-style global lambda update (self_tuning) ----
    def _self_tuning_filter(self, s_first_step: np.ndarray) -> np.ndarray:
        s_tensor = torch.tensor(s_first_step, dtype=torch.float32, device=self.device).unsqueeze(0)
        lam_tensor = torch.tensor(self.lam_self, dtype=torch.float32, device=self.device).unsqueeze(0)
        s_filt = lam_tensor * s_tensor
        with torch.no_grad():
            w_est = self.mixer(s_filt).squeeze(0).cpu().numpy()
        return w_est

    def _update_self_tuning_lambda(self, actual_w: np.ndarray, predicted_s_first_step: np.ndarray):
        lam_tensor = torch.tensor(self.lam_self, requires_grad=True, dtype=torch.float32, device=self.device)
        s_tensor = torch.tensor(predicted_s_first_step, dtype=torch.float32, device=self.device)
        s_filt = lam_tensor * s_tensor
        w_est = self.mixer(s_filt)
        actual_w_tensor = torch.tensor(actual_w, dtype=torch.float32, device=self.device)
        loss = torch.nn.functional.mse_loss(w_est, actual_w_tensor)
        loss.backward()
        if lam_tensor.grad is not None:
            grad_lam = lam_tensor.grad.detach().cpu().numpy()
            self.lam_self = np.clip(self.lam_self - self.lam_self_lr * grad_lam, 0.0, 1.0)

    # ---- MPC solver (localized per bus) ----
    def _mpc_local_bus(self, x: np.ndarray, t: int, weighted_w_hat: np.ndarray, bus_idx: int) -> float:

        k = min(self.horizon, self.T - t - 1)
        if k <= 0:
            return 0.0

        mask = np.zeros(self.env.N)
        for neighbor in self.env.neighborhoods[bus_idx]:
            mask[neighbor - 1] = 1.0
        mask_torch = torch.tensor(mask, dtype=torch.float32, device=self.device)

        # Disturbances masked to the neighborhood
        w_masked = weighted_w_hat[:k] * mask  # (k, N)

        # Closed-form local MPC (script_new style)
        sub = self.sub_mats[bus_idx - 1]
        k_steps = w_masked.shape[0]
        F_pows = sub["F_powers"]
        P_loc = sub["P"]
        G_loc = sub["G"]

        F_sum = np.zeros(self.env.N)
        for tau in range(k_steps):
            F_tau = F_pows[min(tau, len(F_pows) - 1)]
            F_sum += (F_tau.T @ P_loc @ w_masked[tau])

        u_vec_full = -G_loc @ (P_loc @ x + F_sum)
        return float(u_vec_full[bus_idx - 1])

    def _compute_base_s_future(self, w_hist: deque, k: int) -> torch.Tensor:
        """
        Predict latent sequence without applying any bus-specific forecast policy.
        Returned tensor lives on controller device for reuse across buses.
        """
        if len(w_hist) < self.b_context or k <= 0:
            return torch.empty(0, device=self.device)
        recent_w = np.stack(list(w_hist)[-self.b_context :])  # (L, N)
        s_hist = self.transformer.transform(recent_w)  # (L, k_latent)
        s_hist_tensor = torch.tensor(s_hist, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, L, k_latent)
        with torch.no_grad():
            return self.predictor(s_hist_tensor).squeeze(0)  # (k, k_latent)

    # ---- prediction helper: latent -> (lambda on s) -> mixer -> w_hat ----
    def _predict_w_future(
        self,
        w_hist: deque,
        k: int,
        t_current: int,
        bus_idx: Optional[int] = None,
        force_random: bool = False,
        s_future_base: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate forecasted disturbances for a specific bus policy:
        - Compute (or reuse) latent forecasts.
        - Apply bus-type/time-window policy in latent space.
        - Map to disturbance space via mixer.
        """
        # Reuse precomputed latent forecasts when available to avoid repeated predictor calls
        if s_future_base is None:
            s_future_base = self._compute_base_s_future(w_hist, k)
        if s_future_base.numel() == 0:
            return np.zeros((k, self.env.N)), np.zeros((k, self.k_latent))

        with torch.no_grad():
            s_future = self._apply_forecast_policy_on_s(s_future_base, t_current, bus_idx=bus_idx)
            if bus_idx is None:
                lam_vec = torch.tensor(self.lam_t.mean(axis=0), dtype=torch.float32, device=self.device)
            else:
                lam_vec = torch.tensor(self.lam_t[bus_idx - 1], dtype=torch.float32, device=self.device)
            s_filtered = lam_vec * s_future
            w_hat_future = self.mixer(s_filtered).cpu().numpy()  # (k, N)
        return w_hat_future, s_future.cpu().numpy()

    def _sample_noise(self, shape: torch.Size, std: float) -> torch.Tensor:
        """
        Generate large, mostly non-zero disturbances.
        - Uniform: magnitude in [std, 3*std], random sign.
        - Gaussian: 3*std scale, with a floor at |noise| >= std.
        """
        std = max(float(std), 1e-3)
        target_mag = 0.9  # aim for ~0.5 amplitude
        if self.forecast_noise_dist == "uniform":
            base_mag = target_mag * (0.9 + 1.0 * torch.rand(shape, device=self.device))  # [0.25, 0.75]
            sign = torch.sign(torch.rand(shape, device=self.device) - 0.5)
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            noise = sign * base_mag
            return noise
        # gaussian with magnitude clamped around target
        noise = torch.randn(shape, device=self.device) * target_mag
        noise = torch.clamp(noise, -1.2 * target_mag, 1.2 * target_mag)
        small_mask = torch.abs(noise) < 0.25 * target_mag
        if small_mask.any():
            adj = torch.sign(noise[small_mask] + 1e-6) * (0.25 * target_mag + torch.abs(noise[small_mask]))
            noise = noise.clone()
            noise[small_mask] = adj
        return noise

    def _apply_forecast_policy_on_s(self, s_future: torch.Tensor, t: int, bus_idx: Optional[int]) -> torch.Tensor:
        if s_future.numel() == 0:
            return s_future
        # No bus context -> keep base forecast
        if bus_idx is None:
            return s_future

        b0 = bus_idx - 1  # stored indices are 0-based
        base_scale = torch.mean(torch.abs(s_future)).clamp(min=1e-3)

        if b0 in self.pv_bus_idx:
            if t < 200 and self.forecast_noise_pv > 0:
                return self._sample_noise(s_future.shape, self.forecast_noise_pv * base_scale)
            return s_future

        if b0 in self.wind_bus_idx:
            if t >= 300 and self.forecast_noise_wind > 0:
                return self._sample_noise(s_future.shape, self.forecast_noise_wind * base_scale)
            return s_future

        # EV and other buses -> rely on normal prediction
        return s_future

    # main rollout
    def run(self):
        x_t = self.env.reset()
        total_cost = 0.0
        cost_list = []
        w_hist = deque(maxlen=self.b_context)
        s_pred_hist = deque(maxlen=self.b_context)
        x_history = []
        x_history.append(x_t.copy())
        w_actual_history = []
        w_pred_first_history = []
        lam_history = []
        self_tuning_lambda_history = []
        self_tuning_pred_first_history = []

        for t in range(self.T):
            k = min(self.horizon, self.T - t - 1)
            base_s_future = None
            w_pred_first_step = np.zeros(self.env.N)

            if t < self.b_context or k <= 0:
                # Warm-up: keep lambda at initial value
                s_future_base = np.zeros((k, self.k_latent))
                w_hat_future_base = np.zeros((k, self.env.N))
            else:
                base_s_future = self._compute_base_s_future(w_hist, k)
                w_hat_future_base, s_future_base = self._predict_w_future(
                    w_hist, k, t_current=t, bus_idx=None, force_random=False, s_future_base=base_s_future
                )

            if s_future_base is not None and s_future_base.shape[0] > 0:
                s_first_for_update = s_future_base[0].copy()
            else:
                s_first_for_update = np.zeros(self.k_latent, dtype=float)

            # Log self-tuning lambda and its filtered forecast (diagnostic/plotting)
            self_tuning_lambda_history.append(self.lam_self.copy())
            self_tuning_pred_first_history.append(self._self_tuning_filter(s_first_for_update))

            # Lambda update on latent first-step vs actual w
            if len(w_hist) == self.b_context and len(s_pred_hist) == self.b_context:
                self._update_lambda_on_s_window(np.stack(w_hist), np.stack(s_pred_hist))
                if self.verbose and t % 5 == 0:
                    print(f"[t={t}] lambda stats -> min: {self.lam_t.min():.4f}, max: {self.lam_t.max():.4f}")

            # Localized MPC per bus
            u_vec = np.zeros(self.env.N)
            for i in self.controllable_buses:
                if k > 0:
                    w_hat_future_i, s_future_i = self._predict_w_future(
                        w_hist, k, t_current=t, bus_idx=i, force_random=False, s_future_base=base_s_future
                    )
                else:
                    w_hat_future_i = np.zeros((k, self.env.N))
                    s_future_i = np.zeros((k, self.k_latent))
                u_vec[i - 1] = self._mpc_local_bus(x_t, t, w_hat_future_i, bus_idx=i)
                # Log the forecast actually used for bus i (its own component, first step)
                if k > 0 and w_hat_future_i.shape[0] > 0:
                    w_pred_first_step[i - 1] = w_hat_future_i[0, i - 1]

            # System step
            x_t, w_actual = self.env.step(u_vec)
            x_history.append(x_t.copy())
            w_actual_history.append(w_actual.copy())

            # Update histories for OCL (use the first predicted step as in DISC logic)
            if k > 0 and w_hat_future_base.shape[0] > 0:
                w_pred_first_history.append(w_pred_first_step.copy())
                s_pred_hist.append(s_future_base[0].copy())
            else:
                w_pred_first_history.append(np.zeros(self.env.N))
                s_pred_hist.append(np.zeros(self.k_latent))
            w_hist.append(w_actual)
            # DISC-style global lambda update
            self._update_self_tuning_lambda(w_actual, s_first_for_update)

            # Custom stage cost on full state/control
            cost_t = self._stage_cost_numpy(x_t, u_vec)
            total_cost += cost_t
            cost_list.append(cost_t)
            if self.verbose and t % 5 == 0:
                mask_local = np.zeros_like(x_t)
                mask_local[10] = 1  # bus11 index = 10
                local_cost = self._stage_cost_numpy(x_t * mask_local, u_vec * mask_local)
                print(f"[t={t}] bus11 local cost={local_cost:.4f}")

            lam_history.append(self.lam_t.copy())

        if self.verbose:
            print(f"[done] Total cost={total_cost:.4f}, lambda final min={self.lam_t.min():.4f}, max={self.lam_t.max():.4f}")
        w_pred_first_history = np.array(w_pred_first_history)
        # w_pred_first_history is already in disturbance space; lambda applies in latent space earlier.
        w_eff_final = w_pred_first_history
        return (
            total_cost,
            cost_list,
            self.lam_t,
            np.array(lam_history),
            np.array(x_history),
            np.array(w_actual_history),
            w_pred_first_history,
                w_eff_final,
                self.lam_self.copy(),
                np.array(self_tuning_lambda_history),
                np.array(self_tuning_pred_first_history),
        )

    def run_nominal(self):
        """Controller that ignores disturbance forecasts (uses zero w_hat in MPC)."""
        x_t = self.env.reset()
        total_cost = 0.0
        cost_list = []
        x_history = [x_t.copy()]
        w_actual_history = []
        w_pred_first_history = []
        w_hist = deque(maxlen=self.b_context)

        for t in range(self.T):
            k = min(self.horizon, self.T - t - 1)
            if t < self.b_context:
                w_hat_future = np.zeros((k, self.env.N))
            else:
                w_hat_future = np.zeros((k, self.env.N))  # ignore disturbance prediction

            u_vec = np.zeros(self.env.N)
            for i in self.controllable_buses:
                u_vec[i - 1] = self._mpc_local_bus(x_t, t, w_hat_future, bus_idx=i)

            x_t, w_actual = self.env.step(u_vec)
            x_history.append(x_t.copy())
            w_actual_history.append(w_actual.copy())
            w_hist.append(w_actual)
            if k > 0 and w_hat_future.shape[0] > 0:
                w_pred_first_history.append(w_hat_future[0].copy())
            else:
                w_pred_first_history.append(np.zeros(self.env.N))

            cost_t = self._stage_cost_numpy(x_t, u_vec)
            total_cost += cost_t
            cost_list.append(cost_t)

        return total_cost, cost_list, np.array(x_history), np.array(w_actual_history), np.array(w_pred_first_history)


# --------------------------------------------------------------------------- #
# Example setup (small defaults to keep runtime reasonable if executed)
if __name__ == "__main__":
    from case57_env import build_case57_environment
    env = build_case57_environment(kappa=2, T=500)

    K_LATENT = 3
    K_MPC = 20
    B_CONTEXT = 20
    predictor, mixer_net, transformer = train_ica_predictor(
        env, k_latent=K_LATENT, k_horizon=K_MPC, lookback_L=B_CONTEXT, num_epochs=300
    )
    controllable_buses = list(range(1, env.N + 1))
    print("EDGES:", env.edges)
    print("IMPEDANCES:", env.impedances)
    print("Controllable buses:", controllable_buses)
    controller = LocalizedLearningController(
        env=env,
        horizon=K_MPC,
        predictor=predictor,
        mixer=mixer_net,
        transformer=transformer,
        k_latent=K_LATENT,
        b_context=B_CONTEXT,
        controllable_buses=controllable_buses,
    )
    # Our method (OCL)
    total_cost, cost_list, lam_final, lam_hist, x_hist, w_act, w_pred, w_eff, lam_self_final, lam_self_hist, w_self_pred_first = controller.run()
    print(f"OCL Total cost: {total_cost:.4f}")
    print(f"OCL Final lambda: {lam_final}")

    # Incremental law baseline (nominal)
    nom_cost, nom_cost_list, nom_x_hist, nom_w_act, nom_w_pred = controller.run_nominal()
    print(f"Incremental law Total cost: {nom_cost:.4f}")
    print(f"Cost gap (OCL - Incremental law): {total_cost - nom_cost:.4f}")

    # --- Metrics summary: max voltage, max cumulative cost, improvement vs incremental law ---
    def report_metrics(name: str, x_hist_arr: np.ndarray, total_cost_val: float, cost_seq: list, ref_cost: float, ref_name: str):
        max_voltage = float(np.max(np.abs(x_hist_arr)))
        cum_cost = np.cumsum(cost_seq)
        max_cum_cost = float(np.max(cum_cost)) if cum_cost.size > 0 else float(total_cost_val)
        improvement = (ref_cost - total_cost_val) / ref_cost * 100.0 if ref_cost != 0 else float("nan")
        print(
            f"[metrics] {name}: max|x|={max_voltage:.4f}, "
            f"max cumulative cost={max_cum_cost:.4f}, "
            f"improvement vs {ref_name}={improvement:.2f}%"
        )

    report_metrics("OCL", x_hist, total_cost, cost_list, nom_cost, "Incremental law")
    report_metrics("Incremental law", nom_x_hist, nom_cost, nom_cost_list, nom_cost, "Incremental law")

    # Prepare lambda series for bus6
    t_lambda = np.arange(lam_hist.shape[0])
    lam_series = []
    if lam_hist.ndim == 3:
        if hasattr(env, "bus_id_map") and 6 in env.bus_id_map:
            lam_bus = lam_hist[:, env.bus_id_map[6], :]
        elif lam_hist.shape[1] > 6:
            lam_bus = lam_hist[:, 5, :]
        else:
            lam_bus = lam_hist.mean(axis=1)
    else:
        lam_bus = lam_hist
    for j in range(lam_bus.shape[-1]):
        lam_series.append(lam_bus[:, j])

    # Single lambda figure with grid and save
    plt.figure(figsize=(8, 2.5))
    for series in lam_series:
        plt.plot(t_lambda, series, color="navy", alpha=0.8)
    plt.xlabel("Time Step $t\\in[500]$", fontsize=12)
    plt.ylabel("$\\lambda_{6}(t)$", fontsize=12)
    plt.title("$\\lambda_{6}(t)$ Trajectory", fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig("lambda_bus6.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Determine column index for bus 6 (disturbance/state arrays are 0-based)
    bus6_col = w_act.shape[1] - 1
    if hasattr(env, "bus_id_map") and 6 in env.bus_id_map:
        cand = env.bus_id_map[6]
        if 0 <= cand < w_act.shape[1]:
            bus6_col = cand
    elif w_act.shape[1] > 6:
        bus6_col = 5

    # Disturbance, voltage, and cost comparisons (OCL vs Incremental law)
    t_axis = np.arange(x_hist.shape[0])
    t_axis_pred = np.arange(w_act.shape[0])
    t_cost = np.arange(len(cost_list))

    fig_two, axes_two = plt.subplots(3, 1, figsize=(8, 9), sharex=False)
    axes_two[0].plot(t_axis_pred, w_pred[:, bus6_col], label=r"$\widehat{w}_{6}$ (OCL)", color="navy")
    axes_two[0].plot(t_axis_pred, w_act[:, bus6_col], label="actual w", color="black", alpha=0.5)
    axes_two[0].set_title("Disturbance (bus 6)")
    axes_two[0].set_ylabel("w")
    axes_two[0].legend()
    axes_two[0].grid(True, alpha=0.2)

    axes_two[1].plot(t_axis, np.abs(x_hist[:, bus6_col]), label="OCL |x_6|", color="navy")
    axes_two[1].plot(t_axis, np.abs(nom_x_hist[:, bus6_col]), label="Incremental law |x_6|", color="tab:green", linestyle="--")
    axes_two[1].set_title("Voltage magnitude (bus 6)")
    axes_two[1].set_ylabel("|x|")
    axes_two[1].legend()
    axes_two[1].grid(True, alpha=0.2)

    axes_two[2].plot(t_cost, cost_list, label="OCL cost per step", color="navy")
    axes_two[2].plot(t_cost, nom_cost_list, label="Incremental law cost per step", color="tab:green", linestyle="--")
    axes_two[2].set_title("Stage cost")
    axes_two[2].set_xlabel("t")
    axes_two[2].set_ylabel("cost")
    axes_two[2].legend()
    axes_two[2].grid(True, alpha=0.2)

    fig_two.tight_layout()
    fig_two.savefig("comparison_bus6.png", dpi=300, bbox_inches="tight")
    plt.show()
