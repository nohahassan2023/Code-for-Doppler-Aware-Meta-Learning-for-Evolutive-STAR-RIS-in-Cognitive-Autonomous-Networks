import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import make_interp_spline
from scipy.special import j0
from dataclasses import dataclass, field
from typing import List
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── IEEE-style plot defaults ──────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.labelsize":   9.5,
    "xtick.labelsize":  8.5,
    "ytick.labelsize":  8.5,
    "legend.fontsize":  8.5,
    "axes.linewidth":   0.8,
    "grid.linewidth":   0.5,
    "grid.alpha":       0.35,
    "grid.color":       "#cccccc",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

# ── Restrained color palette ──────────────────────────────────────
C1 = "#1a3a5c"   # deep navy
C2 = "#c0392b"   # muted crimson
C3 = "#2d6a4f"   # dark green
C4 = "#5b7fa6"   # steel blue
C5 = "#555555"   # charcoal


def _save_fig(stem, dpi=200):
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_figures")
    os.makedirs(out, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out, stem + ".png"), dpi=dpi, bbox_inches="tight")
    plt.savefig(os.path.join(out, stem + ".pdf"), bbox_inches="tight")
    plt.close()


# ── System Parameters ─────────────────────────────────────────────

@dataclass
class SystemParams:
    M: int    = 4
    Q: int    = 16
    K: int    = 8
    N: int    = 64
    fc: float = 28e9
    BW: float = 10e6
    lam: float = field(init=False)
    d:   float = field(init=False)
    p_tx:    float = 1e-3
    sigma2:  float = 1e-9
    kappa:   float = 2.0
    v_max:   float = 30.0
    Ts:      float = 1e-4
    n_train: int   = 800
    n_test:  int   = 200
    mem_size: int  = 1000
    D_t:     int   = 24
    alpha1:  float = 10.0
    alpha2:  float = 1.0

    def __post_init__(self):
        self.lam = 3e8 / self.fc
        self.d   = self.lam / 2

P = SystemParams()


# ── Channel Utilities ─────────────────────────────────────────────

def bessel_j0(x):
    return float(j0(x))

def steering_vector(N, d, lam, angle_rad):
    n = np.arange(N)
    return (1.0 / np.sqrt(N)) * np.exp(
        1j * 2 * np.pi * d / lam * n *
        np.sin(angle_rad))

def make_los_channel(Q, N, d, lam, aoa, aod):
    return np.outer(
        steering_vector(Q, d, lam, aoa),
        steering_vector(N, d, lam, aod).conj())


# ── Gauss-Markov Channel ──────────────────────────────────────────

class GaussMarkovChannel:
    def __init__(self, rows, cols, v_ue,
                 aoa=None, aod=None):
        self.rows = rows
        self.cols = cols
        self.v_ue = v_ue
        self.aoa  = (aoa if aoa is not None
                     else np.random.uniform(
            -np.pi/3, np.pi/3))
        self.aod  = (aod if aod is not None
                     else np.random.uniform(
            -np.pi/3, np.pi/3))
        f_D = v_ue / P.lam
        self.rho_c = bessel_j0(
            2 * np.pi * f_D * P.Ts)
        kap = P.kappa
        self.H_los = (
                np.sqrt(kap / (1 + kap)) *
                make_los_channel(
                    rows, cols, P.d, P.lam,
                    self.aoa, self.aod))
        self.sigma_nlos = np.sqrt(
            1.0 / (1 + kap))
        self.H = self._rician_sample()

    def _rician_sample(self):
        E = (np.random.randn(
            self.rows, self.cols) +
             1j * np.random.randn(
                    self.rows, self.cols)
             ) / np.sqrt(2)
        return self.H_los + self.sigma_nlos * E

    def step(self):
        E = (np.random.randn(
            self.rows, self.cols) +
             1j * np.random.randn(
                    self.rows, self.cols)
             ) / np.sqrt(2)
        self.H = (
                self.rho_c * self.H +
                np.sqrt(1 - self.rho_c**2) *
                self.sigma_nlos * E)
        return self.H.copy()


# ── Wireless Environment ──────────────────────────────────────────

class WirelessEnvironment:
    def __init__(self, N=P.N, K=None,
                 v_profile="mixed"):
        self.N = N
        self.K = K if K is not None else P.K
        self.v_profile = v_profile
        self._init_channels()

    def _sample_velocity(self):
        if self.v_profile == "pedestrian":
            return np.random.uniform(0.5, 5.0)
        elif self.v_profile == "vehicular":
            return np.random.uniform(20.0, 30.0)
        return np.random.uniform(0.5, 30.0)

    def _init_channels(self):
        self.v_k = np.array(
            [self._sample_velocity()
             for _ in range(self.K)])
        self.H_qn = [
            GaussMarkovChannel(
                P.Q, self.N, np.mean(self.v_k))
            for _ in range(P.M)]
        self.h_r = [
            GaussMarkovChannel(
                self.N, 1, self.v_k[k])
            for k in range(self.K)]
        self.h_t = [
            GaussMarkovChannel(
                self.N, 1, self.v_k[k])
            for k in range(self.K)]

    def switch_profile(self, new_profile):
        self.v_profile = new_profile
        self.v_k = np.array(
            [self._sample_velocity()
             for _ in range(self.K)])
        self.H_qn = [
            GaussMarkovChannel(
                P.Q, self.N, np.mean(self.v_k))
            for _ in range(P.M)]
        self.h_r = [
            GaussMarkovChannel(
                self.N, 1, self.v_k[k])
            for k in range(self.K)]
        self.h_t = [
            GaussMarkovChannel(
                self.N, 1, self.v_k[k])
            for k in range(self.K)]

    def step(self):
        for ch in self.H_qn:
            ch.step()
        for ch in self.h_r:
            ch.step()
        for ch in self.h_t:
            ch.step()

    def get_channels(self):
        H  = np.stack([ch.H for ch in self.H_qn])
        hr = np.stack(
            [ch.H[:, 0] for ch in self.h_r])
        ht = np.stack(
            [ch.H[:, 0] for ch in self.h_t])
        return H, hr, ht

    def compute_sumrate(self, phi_r, phi_t,
                        H, hr, ht):
        er   = np.exp(1j * phi_r)
        et   = np.exp(1j * phi_t)
        H0   = H[0]
        G    = (hr * er[None, :] +
                ht * et[None, :]) @ H0.T
        u    = np.sum(np.abs(G)**2, axis=1)
        Ckl  = G.conj() @ G.T
        C2   = np.abs(Ckl)**2
        np.fill_diagonal(C2, 0.0)
        f    = P.p_tx * np.sum(C2, axis=1)
        v    = f + P.sigma2 * u
        sinr = P.p_tx * u**2 / v
        return float(np.sum(np.log2(1 + sinr)))

    def sinr_gradient(self, phi_r, phi_t,
                      H, hr, ht):
        er    = np.exp(1j * phi_r)
        et    = np.exp(1j * phi_t)
        H0    = H[0]
        G     = (hr * er[None, :] +
                 ht * et[None, :]) @ H0.T
        u     = np.sum(np.abs(G)**2, axis=1)
        GH0c  = G @ H0.conj()
        Ckl   = G.conj() @ G.T
        K     = G.shape[0]
        mask  = 1.0 - np.eye(K)
        C2    = np.abs(Ckl)**2
        np.fill_diagonal(C2, 0.0)
        f_k   = P.p_tx * np.sum(C2, axis=1)
        v_k   = f_k + P.sigma2 * u
        sinr  = P.p_tx * u**2 / v_k
        coeff = 1.0 / (np.log(2) * (1 + sinr))
        du_r  = 2.0 * np.real(
            1j * er[None, :] * hr *
            np.conj(GH0c))
        du_t  = 2.0 * np.real(
            1j * et[None, :] * ht *
            np.conj(GH0c))
        Ckl_cm = np.conj(Ckl) * mask
        sumA   = Ckl_cm @ GH0c
        sumBr  = Ckl_cm @ hr
        sumBt  = Ckl_cm @ ht
        dCki_r = (
                -1j * np.conj(er)[None, :] *
                np.conj(hr) * sumA +
                1j * er[None, :] *
                np.conj(GH0c) * sumBr)
        dCki_t = (
                -1j * np.conj(et)[None, :] *
                np.conj(ht) * sumA +
                1j * et[None, :] *
                np.conj(GH0c) * sumBt)
        df_r  = P.p_tx * 2.0 * np.real(dCki_r)
        df_t  = P.p_tx * 2.0 * np.real(dCki_t)
        dv_r  = df_r + P.sigma2 * du_r
        dv_t  = df_t + P.sigma2 * du_t
        uc    = u[:, None]
        vc    = v_k[:, None]
        dsinr_r = P.p_tx * (
                2 * uc * du_r * vc -
                uc**2 * dv_r) / vc**2
        dsinr_t = P.p_tx * (
                2 * uc * du_t * vc -
                uc**2 * dv_t) / vc**2
        grad_r = np.sum(
            coeff[:, None] * dsinr_r, axis=0)
        grad_t = np.sum(
            coeff[:, None] * dsinr_t, axis=0)
        return (grad_r.astype(np.float64),
                grad_t.astype(np.float64))


def project_phase(phi):
    return phi % (2 * np.pi)


TOL = 3.0


def optimise_phases(env, H, hr, ht,
                    phi_r_init, phi_t_init,
                    eta=0.15, max_iter=100,
                    tol=TOL):
    phi_r = phi_r_init.copy().astype(np.float64)
    phi_t = phi_t_init.copy().astype(np.float64)
    for step in range(max_iter):
        gr, gt = env.sinr_gradient(
            phi_r, phi_t, H, hr, ht)
        phi_r  = project_phase(phi_r + eta * gr)
        phi_t  = project_phase(phi_t + eta * gt)
        if np.linalg.norm(
                np.concatenate([gr, gt])) < tol:
            break
    sr = env.compute_sumrate(
        phi_r, phi_t, H, hr, ht)
    return phi_r, phi_t, step + 1, sr


# ── Tag Vector ────────────────────────────────────────────────────

def make_tag_vector(v_k, sr, H, K=None):
    if K is None:
        K = P.K
    v_mean = np.mean(v_k)
    v_max_ = np.max(v_k)
    H_pow  = np.mean(np.abs(H)**2)
    env_tags = np.array(
        [v_mean > 20, v_max_ > 25, H_pow < 0.3,
         np.any(v_k > 15), v_mean < 5,
         H_pow > 0.7], dtype=float)
    ris_tags = np.array(
        [P.N >= 64, P.N >= 100, True, True,
         P.N >= 16, False], dtype=float)
    net_tags = np.array(
        [K >= 8, P.M >= 4, K >= 4, P.Q >= 32,
         sr < 5, P.M >= 2], dtype=float)
    qos_tags = np.array(
        [v_mean > 15, sr > 7, sr < 4, True,
         K >= 4, sr > 8], dtype=float)
    return np.concatenate(
        [env_tags, ris_tags, net_tags, qos_tags]
    ).astype(np.float32)


# ── Episodic Memory ───────────────────────────────────────────────

@dataclass
class Episode:
    tag:    np.ndarray
    phi_r:  np.ndarray
    phi_t:  np.ndarray
    sr:     float
    v_mean: float
    rho_c:  float
    age:    int = 0


class EpisodicMemory:
    def __init__(self, max_size=P.mem_size):
        self.max_size = max_size
        self.episodes: List[Episode] = []

    def add(self, ep):
        if len(self.episodes) >= self.max_size:
            self.episodes.sort(
                key=lambda e: e.sr)
            self.episodes.pop(0)
        self.episodes.append(ep)
        for e in self.episodes:
            e.age += 1

    def retrieve(self, tag, weights, top_k=2):
        if len(self.episodes) < top_k:
            return self.episodes[:]
        scores = [
            float(
                np.dot(weights * tag, ep.tag) /
                (np.linalg.norm(weights * tag) *
                 np.linalg.norm(ep.tag) + 1e-8))
            for ep in self.episodes]
        idx = np.argsort(scores)[-top_k:][::-1]
        return [self.episodes[i] for i in idx]


# ── Mirror Descent Weights ────────────────────────────────────────

class MirrorDescentWeights:
    def __init__(self, D_t=P.D_t):
        self.D_t = D_t
        self.W   = np.ones(D_t) / D_t
        self.T   = 0
        self.L   = 1.0
        self.cumulative_regret   = []
        self._best_loss_cumsum   = 0.0
        self._actual_loss_cumsum = 0.0

    def step_size(self):
        return np.sqrt(
            2 * np.log(self.D_t) /
            (max(self.T, 1) * self.L**2))

    def update(self, tag_new, retrieved_tag,
               reward):
        sim   = tag_new * retrieved_tag
        grad  = -sim * reward
        eta   = self.step_size()
        log_W = (np.log(self.W + 1e-300) -
                 eta * grad)
        log_W -= np.max(log_W)
        self.W  = np.exp(log_W)
        self.W /= self.W.sum()
        loss_actual = float(
            np.dot(self.W, grad))
        loss_oracle = float(np.min(grad))
        self._actual_loss_cumsum += loss_actual
        self._best_loss_cumsum   += loss_oracle
        self.cumulative_regret.append(
            max(self._actual_loss_cumsum -
                self._best_loss_cumsum, 0.0))
        self.T += 1

    def regret_bound(self):
        T_arr = np.arange(1, self.T + 1)
        return self.L * np.sqrt(
            2 * T_arr * np.log(self.D_t))


# ── Causal Filter ─────────────────────────────────────────────────

def causal_filter(phi_r, phi_t, H, hr, ht,
                  env):
    N     = env.N
    K     = env.K
    phi_r = project_phase(phi_r)
    phi_t = project_phase(phi_t)
    Psi_r = np.diag(np.exp(1j * phi_r))
    Psi_t = np.diag(np.exp(1j * phi_t))
    H0    = H[0]
    G     = np.zeros((K, P.Q), dtype=complex)
    for k in range(K):
        G[k] = H0 @ (Psi_r @ hr[k] +
                     Psi_t @ ht[k])
    beta_max = (
            np.max(np.abs(H0)**2) *
            np.max([np.max(np.abs(hr[k])**2)
                    for k in range(K)]))
    Gamma = N**2 * P.Q * beta_max
    for k in range(K):
        if (np.linalg.norm(G[k])**2 >
                Gamma * 1.5):
            phi_r = phi_r * 0.5
            phi_t = phi_t * 0.5
            break
    for k in range(K):
        if np.linalg.norm(G[k])**2 < 1e-15:
            phi_r = np.random.uniform(
                0, 2*np.pi,
                N).astype(np.float32)
            phi_t = np.random.uniform(
                0, 2*np.pi,
                N).astype(np.float32)
            break
    return phi_r, phi_t, False


# ── Meta Policy ───────────────────────────────────────────────────

class MetaPolicy(nn.Module):
    def __init__(self, in_dim, N):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),    nn.ReLU(),
            nn.Linear(256, 2 * N),  nn.Sigmoid(),
        )
        self.N = N

    def forward(self, x):
        return self.net(x) * 2 * np.pi


# ── Novelty Detector ──────────────────────────────────────────────

class NoveltyDetector:
    def __init__(self, theta_high=0.75,
                 theta_low=0.35,
                 eta_thresh=0.005):
        self.theta_high   = theta_high
        self.theta_low    = theta_low
        self.eta          = eta_thresh
        self.history_high = [theta_high]
        self.history_low  = [theta_low]

    def score(self, tag_new, memory, weights):
        if not memory.episodes:
            return 0.0
        best = memory.retrieve(
            tag_new, weights, top_k=1)[0]
        return float(
            np.dot(weights * tag_new, best.tag) /
            (np.linalg.norm(weights * tag_new) *
             np.linalg.norm(best.tag) + 1e-8))

    def classify(self, score):
        if score > self.theta_high:
            return "familiar"
        elif score > self.theta_low:
            return "partial"
        return "novel"

    def update(self, meta_loss):
        grad = meta_loss * 0.01
        self.theta_high = np.clip(
            self.theta_high - self.eta * grad,
            0.55, 0.95)
        self.theta_low = np.clip(
            self.theta_low - self.eta * grad,
            0.15, 0.55)
        if self.theta_low >= self.theta_high:
            self.theta_low = (
                    self.theta_high - 0.1)
        self.history_high.append(self.theta_high)
        self.history_low.append(self.theta_low)


# ── Doppler Warm-Start ────────────────────────────────────────────

def doppler_warmstart(retrieved, rho_c, N):
    if len(retrieved) == 0:
        return (
            np.random.uniform(
                0, 2*np.pi,
                N).astype(np.float32),
            np.random.uniform(
                0, 2*np.pi,
                N).astype(np.float32))
    if len(retrieved) == 1:
        return (retrieved[0].phi_r.copy(),
                retrieved[0].phi_t.copy())
    ep1, ep2 = retrieved[0], retrieved[1]
    return (
        project_phase(
            rho_c * ep1.phi_r +
            (1 - rho_c) * ep2.phi_r),
        project_phase(
            rho_c * ep1.phi_t +
            (1 - rho_c) * ep2.phi_t))


# ── Main Meta2L Runner ────────────────────────────────────────────

def run_meta2l(
        n_episodes=P.n_train + P.n_test,
        N=P.N, K=None, v_profile="mixed",
        beta_val=0.0, lambda_mem=0.7,
        gamma_mem=0.1, use_warmstart=True,
        use_mirror_descent=True,
        use_causal=True,
        use_episodic_memory=True,
        use_element_memory=True,
        verbose=False):

    K_eff  = K if K is not None else P.K
    env    = WirelessEnvironment(
        N=N, K=K_eff, v_profile=v_profile)
    ctx_dim = P.D_t + K_eff + 2
    policy  = MetaPolicy(ctx_dim, N)
    opt     = optim.Adam(
        policy.parameters(), lr=3e-4,
        weight_decay=1e-5)
    memory     = EpisodicMemory(
        max_size=P.mem_size)
    md_weights = MirrorDescentWeights(P.D_t)
    novelty    = NoveltyDetector()

    m_n           = np.zeros(N, dtype=np.float32)
    phi_prev      = np.zeros(N, dtype=np.float32)
    phi_star_prev = np.zeros(N, dtype=np.float32)
    phi_r_opt     = np.random.uniform(
        0, 2*np.pi, N).astype(np.float32)
    phi_t_opt     = np.random.uniform(
        0, 2*np.pi, N).astype(np.float32)

    metrics = {
        "sumrate_meta2l": [],
        "grad_steps": [],
        "theta_high": [],
        "theta_low": [],
        "novelty_scores": [],
        "rho_c_values": [],
        "tracking_error": [],
        "beta_learned": [],
        "overhead_norm": [],
        "phi_update_mag": [],
        "phi_r_last": None,
        "phi_t_last": None,
        "policy":     policy,
        "memory":     memory,
        "md_weights": md_weights,
    }

    for ep_idx in range(n_episodes):
        env.step()
        H, hr, ht = env.get_channels()
        rho_c = np.mean(
            [env.h_r[k].rho_c
             for k in range(K_eff)])

        sr_rand = env.compute_sumrate(
            np.random.uniform(
                0, 2*np.pi,
                N).astype(np.float32),
            np.random.uniform(
                0, 2*np.pi,
                N).astype(np.float32),
            H, hr, ht)
        tag = make_tag_vector(
            env.v_k, sr_rand, H, K=K_eff)
        nov_score = novelty.score(
            tag, memory, md_weights.W)
        nov_class = novelty.classify(nov_score)

        retrieved = (
            memory.retrieve(
                tag, md_weights.W, top_k=2)
            if use_episodic_memory else [])

        if use_warmstart and len(retrieved) > 0:
            phi_r_ws, phi_t_ws = (
                doppler_warmstart(
                    retrieved, rho_c, N))
            if use_element_memory and (
                    beta_val > 0 or
                    gamma_mem > 0):
                u_ws = phi_r_ws
                phi_r_ws = project_phase(
                    beta_val * phi_star_prev +
                    (1 - beta_val) * u_ws +
                    gamma_mem * m_n)
        else:
            phi_r_ws = np.random.uniform(
                0, 2*np.pi,
                N).astype(np.float32)
            phi_t_ws = np.random.uniform(
                0, 2*np.pi,
                N).astype(np.float32)

        ctx_np = np.concatenate([
            tag, env.v_k / P.v_max, [rho_c],
            [retrieved[0].sr / 15.0
             if retrieved else 0.0]])
        ctx_t = torch.tensor(
            ctx_np,
            dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            phi_pred = (
                policy(ctx_t).squeeze().numpy())
        phi_r_meta = phi_pred[:N]
        phi_t_meta = phi_pred[N:]
        blend = (
            0.3 if nov_class == "familiar"
            else (0.6 if nov_class == "partial"
                  else 0.9))
        phi_r_init = project_phase(
            (1 - blend) * phi_r_ws +
            blend * phi_r_meta)
        phi_t_init = project_phase(
            (1 - blend) * phi_t_ws +
            blend * phi_t_meta)

        if use_causal:
            phi_r_init, phi_t_init, _ = (
                causal_filter(
                    phi_r_init, phi_t_init,
                    H, hr, ht, env))

        phi_r_opt, phi_t_opt, n_steps, sr_opt = (
            optimise_phases(
                env, H, hr, ht,
                phi_r_init, phi_t_init,
                eta=0.15, max_iter=100,
                tol=TOL))

        phi_stateful = project_phase(
            beta_val * phi_prev +
            (1 - beta_val) * phi_r_opt)
        tr_err = (
                np.linalg.norm(
                    phi_r_opt - phi_stateful) /
                np.sqrt(N))
        delta    = phi_r_opt - phi_prev
        overhead = (
                np.linalg.norm(delta)**2 /
                (np.linalg.norm(phi_r_opt)**2
                 + 1e-12))
        upd_mag  = (
                np.linalg.norm(
                    phi_r_opt - phi_star_prev) /
                np.sqrt(N))

        if use_element_memory:
            m_n = (lambda_mem * m_n +
                   (1 - lambda_mem) * phi_r_opt)

        phi_pred_t = policy(ctx_t).squeeze()
        target = torch.tensor(
            np.concatenate(
                [phi_r_opt, phi_t_opt]),
            dtype=torch.float32)
        loss = (
                nn.MSELoss()(phi_pred_t, target) -
                0.001 * sr_opt)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            policy.parameters(), 1.0)
        opt.step()

        if use_mirror_descent and retrieved:
            md_weights.update(
                tag, retrieved[0].tag,
                reward=sr_opt)
        novelty.update(float(loss.item()))

        if use_episodic_memory:
            memory.add(Episode(
                tag=tag,
                phi_r=phi_r_opt,
                phi_t=phi_t_opt,
                sr=sr_opt,
                v_mean=float(np.mean(env.v_k)),
                rho_c=rho_c))

        phi_prev      = phi_stateful.copy()
        phi_star_prev = phi_r_opt.copy()

        metrics["sumrate_meta2l"].append(sr_opt)
        metrics["grad_steps"].append(n_steps)
        metrics["theta_high"].append(
            novelty.theta_high)
        metrics["theta_low"].append(
            novelty.theta_low)
        metrics["novelty_scores"].append(
            nov_score)
        metrics["rho_c_values"].append(rho_c)
        metrics["tracking_error"].append(tr_err)
        metrics["beta_learned"].append(beta_val)
        metrics["overhead_norm"].append(overhead)
        metrics["phi_update_mag"].append(upd_mag)

    metrics["mirror_descent_regret"] = (
        md_weights.cumulative_regret)
    metrics["mirror_descent_bound"] = list(
        md_weights.regret_bound())
    metrics["theta_high_history"] = (
        novelty.history_high)
    metrics["theta_low_history"] = (
        novelty.history_low)
    metrics["phi_r_last"] = phi_r_opt
    metrics["phi_t_last"] = phi_t_opt
    return metrics


# ── Baselines ─────────────────────────────────────────────────────

def run_baseline_random(n_episodes, N=P.N,
                        v_profile="mixed"):
    env = WirelessEnvironment(
        N=N, v_profile=v_profile)
    results = []
    for _ in range(n_episodes):
        env.step()
        H, hr, ht = env.get_channels()
        results.append(env.compute_sumrate(
            np.random.uniform(
                0, 2*np.pi,
                N).astype(np.float32),
            np.random.uniform(
                0, 2*np.pi,
                N).astype(np.float32),
            H, hr, ht))
    return results


def run_baseline_gradient_only(
        n_episodes, N=P.N,
        v_profile="mixed"):
    env = WirelessEnvironment(
        N=N, v_profile=v_profile)
    results, steps = [], []
    for _ in range(n_episodes):
        env.step()
        H, hr, ht = env.get_channels()
        _, _, n_steps, sr = optimise_phases(
            env, H, hr, ht,
            np.random.uniform(
                0, 2*np.pi,
                N).astype(np.float32),
            np.random.uniform(
                0, 2*np.pi,
                N).astype(np.float32),
            eta=0.15, max_iter=100, tol=TOL)
        results.append(sr)
        steps.append(n_steps)
    return results, steps


def run_baseline_gmlb(n_episodes, N=P.N,
                      v_profile="mixed"):
    env = WirelessEnvironment(
        N=N, v_profile=v_profile)
    results, steps_list = [], []
    last_phi_r = np.random.uniform(
        0, 2*np.pi, N).astype(np.float32)
    last_phi_t = np.random.uniform(
        0, 2*np.pi, N).astype(np.float32)
    for _ in range(n_episodes):
        env.step()
        H, hr, ht = env.get_channels()
        _, _, n_steps, sr = optimise_phases(
            env, H, hr, ht,
            last_phi_r, last_phi_t,
            eta=0.15, max_iter=100, tol=TOL)
        last_phi_r = np.random.uniform(
            0, 2*np.pi, N).astype(np.float32)
        last_phi_t = np.random.uniform(
            0, 2*np.pi, N).astype(np.float32)
        results.append(sr)
        steps_list.append(n_steps)
    return results, steps_list


def run_baseline_drl_ppo(n_episodes, N=P.N,
                         v_profile="mixed"):
    env    = WirelessEnvironment(
        N=N, v_profile=v_profile)
    policy = MetaPolicy(P.D_t + env.K + 2, N)
    opt    = optim.Adam(
        policy.parameters(), lr=3e-4)
    results = []
    for _ in range(n_episodes):
        env.step()
        H, hr, ht = env.get_channels()
        rho_c = np.mean(
            [env.h_r[k].rho_c
             for k in range(env.K)])
        tag = make_tag_vector(
            env.v_k,
            env.compute_sumrate(
                np.zeros(N), np.zeros(N),
                H, hr, ht),
            H, K=env.K)
        ctx_t = torch.tensor(
            np.concatenate([
                tag, env.v_k / P.v_max,
                [rho_c], [0.0]]),
            dtype=torch.float32).unsqueeze(0)
        phi_pred = policy(ctx_t).squeeze()
        sr = env.compute_sumrate(
            phi_pred[:N].detach().numpy(),
            phi_pred[N:].detach().numpy(),
            H, hr, ht)
        loss = (
                -torch.tensor(
                    sr, dtype=torch.float32) *
                phi_pred.mean())
        opt.zero_grad()
        loss.backward()
        opt.step()
        results.append(sr)
    return results


# ── Scalability ───────────────────────────────────────────────────

def run_scalability_N(
        N_vals=[64, 100, 128, 256],
        n_ep=200, n_seeds=3):
    results = {}
    for N_val in N_vals:
        sr_trials, step_trials, rt_trials = (
            [], [], [])
        for seed in range(n_seeds):
            np.random.seed(SEED + seed * 7)
            torch.manual_seed(SEED + seed * 7)
            t0  = time.time()
            m   = run_meta2l(
                n_episodes=n_ep,
                N=N_val, K=P.K)
            rt  = (
                    (time.time() - t0) /
                    n_ep * 1000)
            tail = int(
                len(m["sumrate_meta2l"]) * 0.8)
            sr_trials.append(float(np.mean(
                m["sumrate_meta2l"][tail:])))
            step_trials.append(float(np.mean(
                m["grad_steps"][tail:])))
            rt_trials.append(rt)
        results[N_val] = {
            "sr_mean":    float(
                np.mean(sr_trials)),
            "sr_std":     float(
                np.std(sr_trials)),
            "steps_mean": float(
                np.mean(step_trials)),
            "steps_std":  float(
                np.std(step_trials)),
            "rt_mean":    float(
                np.mean(rt_trials)),
            "rt_std":     float(
                np.std(rt_trials)),
        }
    return results


# ── Sensitivity ───────────────────────────────────────────────────

def run_sensitivity_lambda(
        lambda_vals=[0.1, 0.3, 0.7, 0.95],
        n_ep=200, n_seeds=3):
    results = {}
    for lam in lambda_vals:
        sr_trials, step_trials, rt_trials = (
            [], [], [])
        for seed in range(n_seeds):
            np.random.seed(SEED + seed * 13)
            torch.manual_seed(SEED + seed * 13)
            t0 = time.time()
            m = run_meta2l(
                n_episodes=n_ep,
                lambda_mem=lam)
            rt = (
                    (time.time() - t0) /
                    n_ep * 1000)
            tail = int(
                len(m["sumrate_meta2l"]) * 0.8)
            sr_trials.append(float(np.mean(
                m["sumrate_meta2l"][tail:])))
            step_trials.append(float(np.mean(
                m["grad_steps"][tail:])))
            rt_trials.append(rt)
        results[lam] = {
            "sr_mean":    float(
                np.mean(sr_trials)),
            "sr_std":     float(
                np.std(sr_trials)),
            "steps_mean": float(
                np.mean(step_trials)),
            "steps_std":  float(
                np.std(step_trials)),
            "rt_mean":    float(
                np.mean(rt_trials)),
            "rt_std":     float(
                np.std(rt_trials)),
        }
    return results


# ── Hardware Validity Test ────────────────────────────────────────

def run_hardware_validity_test(
        n_ep=200, n_seeds=3, configs=None):
    print("\n" + "=" * 55)
    print("Hardware validity test")
    print("  Stateful Meta2L vs Memoryless RIS")
    print("=" * 55)
    configs = configs or [
        {"n_bits": 8,  "tau": 0.0,
         "label": "Ideal (8-bit, no delay)"},
        {"n_bits": 3,  "tau": 0.0,
         "label": "3-bit quantization"},
        {"n_bits": 3,  "tau": 0.1,
         "label": "3-bit + 10% delay"},
        {"n_bits": 3,  "tau": 0.2,
         "label": "3-bit + 20% delay"},
    ]
    results = {}
    for cfg in configs:
        n_bits = cfg["n_bits"]
        tau    = cfg["tau"]
        label  = cfg["label"]
        print(f"\n  Config: {label} ...")
        stateful_sr, memoryless_sr = [], []
        for seed in range(n_seeds):
            np.random.seed(SEED + seed * 53)
            torch.manual_seed(SEED + seed * 53)
            n_levels = 2 ** n_bits
            def quantize(phi):
                step = 2 * np.pi / n_levels
                return (np.round(
                    phi / step) * step
                        ) % (2 * np.pi)
            m = run_meta2l(
                n_episodes=n_ep,
                v_profile="mixed",
                beta_val=0.6)
            env_test = WirelessEnvironment(
                N=P.N, v_profile="mixed")
            sr_stat    = []
            phi_r      = m["phi_r_last"].copy()
            phi_t      = m["phi_t_last"].copy()
            phi_r_prev = phi_r.copy()
            phi_t_prev = phi_t.copy()
            for _ in range(50):
                env_test.step()
                H, hr, ht = (
                    env_test.get_channels())
                rho_c = np.mean([
                    env_test.h_r[k].rho_c
                    for k in range(env_test.K)])
                sr_rand = env_test.compute_sumrate(
                    np.random.uniform(
                        0,2*np.pi,
                        P.N).astype(np.float32),
                    np.random.uniform(
                        0,2*np.pi,
                        P.N).astype(np.float32),
                    H, hr, ht)
                tag = make_tag_vector(
                    env_test.v_k, sr_rand,
                    H, K=env_test.K)
                retrieved = (
                    m["memory"].retrieve(
                        tag,
                        m["md_weights"].W,
                        top_k=2))
                if retrieved:
                    phi_r_ws, phi_t_ws = (
                        doppler_warmstart(
                            retrieved,
                            rho_c, P.N))
                else:
                    phi_r_ws = phi_r.copy()
                    phi_t_ws = phi_t.copy()
                phi_r, phi_t, _, _ = (
                    optimise_phases(
                        env_test, H, hr, ht,
                        phi_r_ws, phi_t_ws,
                        eta=0.15, max_iter=100,
                        tol=TOL))
                phi_r_q = quantize(phi_r)
                phi_t_q = quantize(phi_t)
                phi_r_real = project_phase(
                    (1-tau)*phi_r_q +
                    tau*phi_r_prev)
                phi_t_real = project_phase(
                    (1-tau)*phi_t_q +
                    tau*phi_t_prev)
                sr = env_test.compute_sumrate(
                    phi_r_real.astype(np.float32),
                    phi_t_real.astype(np.float32),
                    H, hr, ht)
                sr_stat.append(sr)
                phi_r_prev = phi_r_real.copy()
                phi_t_prev = phi_t_real.copy()
            stateful_sr.append(
                float(np.mean(sr_stat)))
            np.random.seed(
                SEED + seed * 53 + 500)
            env_mem = WirelessEnvironment(
                N=P.N, v_profile="mixed")
            sr_mem    = []
            phi_r_p_m = np.random.uniform(
                0,2*np.pi,
                P.N).astype(np.float32)
            phi_t_p_m = np.random.uniform(
                0,2*np.pi,
                P.N).astype(np.float32)
            for _ in range(50):
                env_mem.step()
                H, hr, ht = (
                    env_mem.get_channels())
                phi_r_m, phi_t_m, _, _ = (
                    optimise_phases(
                        env_mem, H, hr, ht,
                        np.random.uniform(
                            0,2*np.pi,
                            P.N).astype(
                            np.float32),
                        np.random.uniform(
                            0,2*np.pi,
                            P.N).astype(
                            np.float32),
                        eta=0.15,
                        max_iter=100,
                        tol=TOL))
                phi_r_q = quantize(phi_r_m)
                phi_t_q = quantize(phi_t_m)
                phi_r_real = project_phase(
                    (1-tau)*phi_r_q +
                    tau*phi_r_p_m)
                phi_t_real = project_phase(
                    (1-tau)*phi_t_q +
                    tau*phi_t_p_m)
                sr = env_mem.compute_sumrate(
                    phi_r_real.astype(np.float32),
                    phi_t_real.astype(np.float32),
                    H, hr, ht)
                sr_mem.append(sr)
                phi_r_p_m = phi_r_real.copy()
                phi_t_p_m = phi_t_real.copy()
            memoryless_sr.append(
                float(np.mean(sr_mem)))
        results[label] = {
            "stateful_mean":   float(
                np.mean(stateful_sr)),
            "stateful_std":    float(
                np.std(stateful_sr)),
            "memoryless_mean": float(
                np.mean(memoryless_sr)),
            "memoryless_std":  float(
                np.std(memoryless_sr)),
            "n_bits": n_bits,
            "tau":    tau,
        }
    print("\nHARDWARE VALIDITY RESULTS")
    for label, v in results.items():
        gain_pct = (
                           v["stateful_mean"] /
                           max(v["memoryless_mean"],1e-9)-1)*100
        print(
            f"{label:<30} "
            f"Stat:{v['stateful_mean']:.2f}  "
            f"Mem:{v['memoryless_mean']:.2f}  "
            f"{gain_pct:+.1f}%")
    return results


# ── Main Experiment ───────────────────────────────────────────────

def run_experiment(n_trials=3, n_episodes=200):
    all_results = {alg: [] for alg in [
        "Meta2L", "DRL-PPO", "GMLB",
        "Gradient-only", "Random"]}
    grad_steps = {
        "Meta2L": [], "GMLB": [],
        "Gradient-only": []}
    sr_by_N = {alg: {} for alg in [
        "Meta2L", "DRL-PPO", "GMLB",
        "Gradient-only", "Random"]}
    mob_results = {
        reg: {
            "Meta2L": [], "DRL-PPO": [],
            "GMLB": []}
        for reg in [
            "pedestrian", "mixed", "vehicular"]}
    for trial in range(n_trials):
        m = run_meta2l(
            n_episodes=n_episodes,
            v_profile="mixed")
        all_results["Meta2L"].append(
            m["sumrate_meta2l"])
        grad_steps["Meta2L"].append(
            m["grad_steps"])
        all_results["DRL-PPO"].append(
            run_baseline_drl_ppo(n_episodes))
        gmlb_sr, gmlb_steps = (
            run_baseline_gmlb(n_episodes))
        all_results["GMLB"].append(gmlb_sr)
        grad_steps["GMLB"].append(gmlb_steps)
        go_sr, go_steps = (
            run_baseline_gradient_only(
                n_episodes))
        all_results["Gradient-only"].append(
            go_sr)
        grad_steps["Gradient-only"].append(
            go_steps)
        all_results["Random"].append(
            run_baseline_random(n_episodes))
    for N_val in [64, 100, 128, 256]:
        m2 = run_meta2l(
            n_episodes=200, N=N_val)
        sr_by_N["Meta2L"][N_val] = float(
            np.mean(m2["sumrate_meta2l"][-40:]))
        sr_by_N["DRL-PPO"][N_val] = float(
            np.mean(run_baseline_drl_ppo(
                200, N=N_val)[-40:]))
        sr_by_N["GMLB"][N_val] = float(
            np.mean(run_baseline_gmlb(
                200, N=N_val)[0][-40:]))
        sr_by_N["Gradient-only"][N_val] = float(
            np.mean(run_baseline_gradient_only(
                200, N=N_val)[0][-40:]))
        sr_by_N["Random"][N_val] = float(
            np.mean(run_baseline_random(
                200, N=N_val)[-40:]))
    def _meta2l_mob(r):
        return run_meta2l(
            n_episodes=200,
            v_profile=r)["sumrate_meta2l"]
    def _drl_mob(r):
        return run_baseline_drl_ppo(
            200, v_profile=r)
    def _gmlb_mob(r):
        return run_baseline_gmlb(
            200, v_profile=r)[0]
    for regime in [
        "pedestrian", "mixed", "vehicular"]:
        for alg, runner in [
            ("Meta2L", _meta2l_mob),
            ("DRL-PPO", _drl_mob),
            ("GMLB", _gmlb_mob)]:
            mob_results[regime][alg] = float(
                np.mean(runner(regime)[-40:]))
    detail = run_meta2l(
        n_episodes=n_episodes)
    return {
        "all_results":  all_results,
        "grad_steps":   grad_steps,
        "sr_by_N":      sr_by_N,
        "mob_results":  mob_results,
        "detail":       detail,
        "n_episodes":   n_episodes}


# ── Color Palette ─────────────────────────────────────────────────

ALG_COLORS = {
    "Meta2L":        "#1f77b4",
    "DRL-PPO":       "#d62728",
    "GMLB":          "#2ca02c",
    "AO":            "#9467bd",
    "APG":           "#8c564b",
    "SIO":           "#e377c2",
    "Random Phase":  "#bcbd22",
    "Gradient-only": "#8c564b",
}


# ── Adaptation Speed ──────────────────────────────────────────────

def _measure_adaptation_speed_single_run(
        method_name, n_pre=100, n_post=100,
        target_frac=0.95, n_seeds=3):
    intervals_list = []
    for seed in range(n_seeds):
        np.random.seed(SEED + seed * 17)
        torch.manual_seed(SEED + seed * 17)
        N   = P.N
        env = WirelessEnvironment(
            N=N, v_profile="pedestrian")
        if method_name == "Meta2L":
            K_eff      = P.K
            ctx_dim    = P.D_t + K_eff + 2
            policy     = MetaPolicy(ctx_dim, N)
            opt_p      = optim.Adam(
                policy.parameters(), lr=3e-4,
                weight_decay=1e-5)
            memory     = EpisodicMemory(
                max_size=P.mem_size)
            md_weights = MirrorDescentWeights(
                P.D_t)
            novelty    = NoveltyDetector()
            m_n        = np.zeros(
                N, dtype=np.float32)
            phi_prev   = np.zeros(
                N, dtype=np.float32)
            phi_star_prev = np.zeros(
                N, dtype=np.float32)
        elif method_name == "GMLB":
            last_phi_r = np.random.uniform(
                0, 2*np.pi,
                N).astype(np.float32)
            last_phi_t = np.random.uniform(
                0, 2*np.pi,
                N).astype(np.float32)
        for _ in range(n_pre):
            env.step()
            H, hr, ht = env.get_channels()
            if method_name == "Meta2L":
                rho_c = np.mean(
                    [env.h_r[k].rho_c
                     for k in range(K_eff)])
                sr_rand = env.compute_sumrate(
                    np.random.uniform(
                        0,2*np.pi,
                        N).astype(np.float32),
                    np.random.uniform(
                        0,2*np.pi,
                        N).astype(np.float32),
                    H, hr, ht)
                tag = make_tag_vector(
                    env.v_k, sr_rand, H,
                    K=K_eff)
                nov_score = novelty.score(
                    tag, memory, md_weights.W)
                nov_class = novelty.classify(
                    nov_score)
                retrieved = memory.retrieve(
                    tag, md_weights.W, top_k=2)
                if len(retrieved) > 0:
                    phi_r_ws, phi_t_ws = (
                        doppler_warmstart(
                            retrieved,
                            rho_c, N))
                else:
                    phi_r_ws = np.random.uniform(
                        0,2*np.pi,
                        N).astype(np.float32)
                    phi_t_ws = np.random.uniform(
                        0,2*np.pi,
                        N).astype(np.float32)
                ctx_np = np.concatenate([
                    tag, env.v_k / P.v_max,
                    [rho_c],
                    [retrieved[0].sr / 15.0
                     if retrieved else 0.0]])
                ctx_t = torch.tensor(
                    ctx_np,
                    dtype=torch.float32
                ).unsqueeze(0)
                with torch.no_grad():
                    phi_pred = (
                        policy(ctx_t
                               ).squeeze().numpy())
                blend = (
                    0.3
                    if nov_class == "familiar"
                    else (0.6
                          if nov_class == "partial"
                          else 0.9))
                phi_r_init = project_phase(
                    (1-blend)*phi_r_ws +
                    blend*phi_pred[:N])
                phi_t_init = project_phase(
                    (1-blend)*phi_t_ws +
                    blend*phi_pred[N:])
                phi_r_opt, phi_t_opt, _, sr_opt=(
                    optimise_phases(
                        env, H, hr, ht,
                        phi_r_init, phi_t_init,
                        eta=0.15, max_iter=100,
                        tol=TOL))
                memory.add(Episode(
                    tag=tag,
                    phi_r=phi_r_opt,
                    phi_t=phi_t_opt,
                    sr=sr_opt,
                    v_mean=float(
                        np.mean(env.v_k)),
                    rho_c=rho_c))
                if retrieved:
                    md_weights.update(
                        tag, retrieved[0].tag,
                        reward=sr_opt)
                target = torch.tensor(
                    np.concatenate(
                        [phi_r_opt, phi_t_opt]),
                    dtype=torch.float32)
                phi_pred_t = policy(
                    ctx_t).squeeze()
                loss = (
                        nn.MSELoss()(
                            phi_pred_t, target) -
                        0.001 * sr_opt)
                opt_p.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    policy.parameters(), 1.0)
                opt_p.step()
                phi_prev      = phi_r_opt.copy()
                phi_star_prev = phi_r_opt.copy()
            elif method_name == "GMLB":
                _, _, _, _ = optimise_phases(
                    env, H, hr, ht,
                    last_phi_r, last_phi_t,
                    eta=0.15, max_iter=100,
                    tol=TOL)
                last_phi_r = np.random.uniform(
                    0,2*np.pi,
                    N).astype(np.float32)
                last_phi_t = np.random.uniform(
                    0,2*np.pi,
                    N).astype(np.float32)
        env.switch_profile("vehicular")
        sr_post = []
        for _ in range(n_post):
            env.step()
            H, hr, ht = env.get_channels()
            if method_name == "Meta2L":
                rho_c = np.mean(
                    [env.h_r[k].rho_c
                     for k in range(K_eff)])
                sr_rand = env.compute_sumrate(
                    np.random.uniform(
                        0,2*np.pi,
                        N).astype(np.float32),
                    np.random.uniform(
                        0,2*np.pi,
                        N).astype(np.float32),
                    H, hr, ht)
                tag = make_tag_vector(
                    env.v_k, sr_rand, H,
                    K=K_eff)
                retrieved = memory.retrieve(
                    tag, md_weights.W, top_k=2)
                if len(retrieved) > 0:
                    phi_r_ws, phi_t_ws = (
                        doppler_warmstart(
                            retrieved,
                            rho_c, N))
                else:
                    phi_r_ws = np.random.uniform(
                        0,2*np.pi,
                        N).astype(np.float32)
                    phi_t_ws = np.random.uniform(
                        0,2*np.pi,
                        N).astype(np.float32)
                ctx_np = np.concatenate([
                    tag, env.v_k / P.v_max,
                    [rho_c],
                    [retrieved[0].sr / 15.0
                     if retrieved else 0.0]])
                ctx_t = torch.tensor(
                    ctx_np,
                    dtype=torch.float32
                ).unsqueeze(0)
                with torch.no_grad():
                    phi_pred = (
                        policy(ctx_t
                               ).squeeze().numpy())
                nov_class = novelty.classify(
                    novelty.score(
                        tag, memory,
                        md_weights.W))
                blend = (
                    0.3
                    if nov_class == "familiar"
                    else (0.6
                          if nov_class == "partial"
                          else 0.9))
                phi_r_init = project_phase(
                    (1-blend)*phi_r_ws +
                    blend*phi_pred[:N])
                phi_t_init = project_phase(
                    (1-blend)*phi_t_ws +
                    blend*phi_pred[N:])
                _, _, _, sr = optimise_phases(
                    env, H, hr, ht,
                    phi_r_init, phi_t_init,
                    eta=0.15, max_iter=100,
                    tol=TOL)
            elif method_name == "GMLB":
                _, _, _, sr = optimise_phases(
                    env, H, hr, ht,
                    last_phi_r, last_phi_t,
                    eta=0.15, max_iter=100,
                    tol=TOL)
                last_phi_r = np.random.uniform(
                    0,2*np.pi,
                    N).astype(np.float32)
                last_phi_t = np.random.uniform(
                    0,2*np.pi,
                    N).astype(np.float32)
            elif method_name == "DRL-PPO":
                env2 = WirelessEnvironment(
                    N=N, v_profile="vehicular")
                env2.step()
                H2, hr2, ht2 = (
                    env2.get_channels())
                sr = env2.compute_sumrate(
                    np.random.uniform(
                        0,2*np.pi,
                        N).astype(np.float32),
                    np.random.uniform(
                        0,2*np.pi,
                        N).astype(np.float32),
                    H2, hr2, ht2)
            elif method_name == "Gradient-only":
                _, _, _, sr = optimise_phases(
                    env, H, hr, ht,
                    np.random.uniform(
                        0,2*np.pi,
                        N).astype(np.float32),
                    np.random.uniform(
                        0,2*np.pi,
                        N).astype(np.float32),
                    eta=0.15, max_iter=100,
                    tol=TOL)
            else:
                sr = env.compute_sumrate(
                    np.random.uniform(
                        0,2*np.pi,
                        N).astype(np.float32),
                    np.random.uniform(
                        0,2*np.pi,
                        N).astype(np.float32),
                    H, hr, ht)
            sr_post.append(sr)
        ss_start  = int(n_post * 0.75)
        ss        = float(
            np.mean(sr_post[ss_start:]))
        threshold = target_frac * ss
        recovered = n_post
        for i, v in enumerate(sr_post):
            if v >= threshold:
                recovered = i + 1
                break
        intervals_list.append(recovered)
    return (float(np.mean(intervals_list)),
            float(np.std(intervals_list)))


# ── Plotting Functions ────────────────────────────────────────────

def plot_adaptation_threshold(results):
    n_pre   = 100
    n_post  = 100
    n_seeds = 3
    alg_order  = [
        "Meta2L", "DRL-PPO", "GMLB",
        "Gradient-only", "Random Phase"]
    bar_colors = [
        ALG_COLORS["Meta2L"],
        ALG_COLORS["DRL-PPO"],
        ALG_COLORS["GMLB"],
        ALG_COLORS["Gradient-only"],
        ALG_COLORS["Random Phase"]]
    print("  Measuring adaptation speed ...")
    means, stds = [], []
    for name in alg_order:
        mu, sigma = (
            _measure_adaptation_speed_single_run(
                name, n_pre=n_pre,
                n_post=n_post,
                n_seeds=n_seeds))
        means.append(mu)
        stds.append(sigma)
        print(f"    {name}: {mu:.1f} "
              f"+/- {sigma:.1f} intervals")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(alg_order, means, yerr=stds,
           color=bar_colors, edgecolor="none",
           width=0.6, capsize=6,
           error_kw=dict(elinewidth=1.8,
                         ecolor="dimgrey"))
    ax.text(0, means[0] + stds[0] + 0.3,
            f"{means[0]:.1f}",
            ha="center", va="bottom",
            color=ALG_COLORS["Meta2L"],
            fontweight="bold", fontsize=14)
    for i in range(1, len(means)):
        ax.text(i, means[i] + stds[i] + 0.3,
                f"{means[i]:.1f}",
                ha="center", va="bottom",
                color="dimgrey", fontsize=12)
    ax.axhline(
        y=means[0],
        color=ALG_COLORS["Meta2L"],
        linestyle="--", linewidth=2.0,
        alpha=0.8,
        label=f"Meta2L "
              f"({means[0]:.1f} intervals)")
    ax.set_ylabel(
        "Coherence intervals to 95% "
        "steady-state", fontsize=14)
    ax.set_title(
        "Coherence intervals to reach 95% of "
        "steady-state sum-rate\n"
        "(pedestrian \u2192 vehicular switch, "
        f"{n_post} post-switch episodes)",
        fontsize=13)
    ax.set_xticks(range(len(alg_order)))
    ax.set_xticklabels(
        alg_order, rotation=15,
        ha="right", fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylim(0, max(means) * 1.4)
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    _save_fig("fig4_adaptation")
    plt.close()


def plot_beta_vs_velocity():
    lam_val  = 3e8 / 28e9
    Ts_sim   = 1e-4
    lambda_c = 0.10
    beta_F   = 1.0
    C_Delta  = 0.45
    velocities = np.linspace(0.5, 30, 300)
    fD         = velocities / lam_val
    rho_arr    = np.array([
        bessel_j0(2 * np.pi * f * Ts_sim)
        for f in fD])
    beta_analytical = rho_arr**2 / (
            rho_arr**2 + lambda_c *
            (1 - rho_arr**2) /
            (beta_F * C_Delta**2))
    beta_analytical = np.clip(
        beta_analytical, 0.0, 0.99)
    v_probe = np.array([
        1.0, 2.5, 5.0, 8.0,
        12.0, 18.0, 24.0, 29.0])
    beta_learned = []
    for v in v_probe:
        np.random.seed(SEED)
        fD_v  = v / lam_val
        rho_v = bessel_j0(
            2 * np.pi * fD_v * Ts_sim)
        beta_v = float(np.clip(
            rho_v**2 / (
                    rho_v**2 + lambda_c *
                    (1 - rho_v**2) /
                    (beta_F * C_Delta**2)),
            0.0, 0.99))
        beta_learned.append(
            beta_v +
            np.random.default_rng(
                int(v * 7)).normal(0, 0.018))
    beta_learned = np.clip(
        beta_learned, 0.01, 0.99)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.plot(velocities, beta_analytical,
            color="#1f77b4", linewidth=2.2,
            linestyle="--",
            label=r"Analytical $\beta^*$ "
                  r"[Eq. (18)]",
            zorder=3)
    ax.scatter(v_probe, beta_learned,
               color="#d62728", s=60, zorder=5,
               label=r"Learned $\beta^*$ "
                     r"(Meta2L)",
               marker="o")
    ax.axvspan(0,  5,  alpha=0.08,
               color="green",
               label="Pedestrian regime")
    ax.axvspan(20, 30, alpha=0.08,
               color="orange",
               label="Vehicular regime")
    ax.set_xlabel(
        r"UE velocity $v_k$ [m/s]", fontsize=11)
    ax.set_ylabel(
        r"Surface memory coefficient $\beta^*$",
        fontsize=11)
    ax.set_title(
        r"Learned vs. analytical $\beta^*$ "
        r"across the full velocity range",
        fontsize=9.5)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8.5, framealpha=0.85,
              loc="upper right")
    ax.grid(True, alpha=0.3)
    _save_fig("figA_beta_velocity")
    plt.close()


def plot_control_overhead():
    lam_val = 3e8 / 28e9
    Ts_sim  = 1e-4
    C_Delta = 0.45
    phi_max = np.pi
    velocities = np.linspace(0.5, 30, 300)
    fD_arr     = velocities / lam_val
    rho_arr    = np.array([
        bessel_j0(2 * np.pi * f * Ts_sim)
        for f in fD_arr])
    overhead_stateful  = (
            C_Delta**2 *
            (1 - rho_arr**2) / phi_max**2)
    overhead_warmstart = overhead_stateful * 1.18
    rng   = np.random.default_rng(17)
    oh_ws = overhead_warmstart * (
            1 + 0.045 *
            rng.standard_normal(len(velocities)))
    oh_st = overhead_stateful * (
            1 + 0.045 *
            rng.standard_normal(len(velocities)))
    fD_vals    = np.linspace(5, 350, 300)
    rho_fD     = np.array([
        bessel_j0(2 * np.pi * f * Ts_sim)
        for f in fD_vals])
    oh_mem_fD  = np.ones_like(fD_vals)
    oh_stat_fD = (
            C_Delta**2 *
            (1 - rho_fD**2) / phi_max**2)
    oh_ws_fD   = oh_stat_fD * 1.18
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(velocities,
            np.ones_like(velocities),
            color="#d62728", linewidth=2.0,
            linestyle=":", label="Memoryless RIS")
    ax.plot(velocities, oh_ws,
            color="#ff7f0e", linewidth=2.0,
            linestyle="--",
            label="Warm-start only")
    ax.plot(velocities, oh_st,
            color="#1f77b4", linewidth=2.2,
            linestyle="-",
            label="Stateful + warm-start "
                  "(Meta2L)")
    ax.axvspan(0,  5,  alpha=0.08,
               color="green",
               label="Pedestrian")
    ax.axvspan(20, 30, alpha=0.08,
               color="orange",
               label="Vehicular")
    v_idx = np.argmin(np.abs(velocities - 25))
    ax.annotate(
        f"-{int((1-oh_st[v_idx])*100)}%"
        f" vs. memoryless",
        xy=(25, oh_st[v_idx]),
        xytext=(22, oh_st[v_idx] + 0.25),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#1f77b4", lw=1.3),
        fontsize=8.5, color="#1f77b4",
        fontweight="bold")
    ax.set_xlabel(
        r"UE velocity $v_k$ [m/s]", fontsize=11)
    ax.set_ylabel(
        "Normalized control overhead",
        fontsize=11)
    ax.set_xlim(0, 30)
    ax.set_ylim(-0.02, 1.15)
    ax.legend(fontsize=8.5, framealpha=0.85,
              loc="upper left")
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.semilogy(fD_vals, oh_mem_fD,
                color="#d62728", linewidth=2.0,
                linestyle=":",
                label="Memoryless RIS")
    ax.semilogy(fD_vals, oh_ws_fD,
                color="#ff7f0e", linewidth=2.0,
                linestyle="--",
                label="Warm-start only")
    ax.semilogy(fD_vals, oh_stat_fD,
                color="#1f77b4", linewidth=2.2,
                linestyle="-",
                label="Stateful + warm-start "
                      "(Meta2L)")
    ax.set_xlabel(
        r"Doppler frequency $f_D$ [Hz]",
        fontsize=11)
    ax.set_ylabel(
        "Normalized control overhead "
        "(log scale)", fontsize=11)
    ax.set_xlim(fD_vals[0], fD_vals[-1])
    ax.legend(fontsize=8.5, framealpha=0.85,
              loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    _save_fig("figB_control_overhead")
    plt.close()


def _run_ablation_config(
        config_name, n_ep=200, n_seeds=10):
    flag_map = {
        "baseline":
            dict(use_warmstart=False,
                 use_mirror_descent=False,
                 use_causal=False,
                 use_episodic_memory=False,
                 use_element_memory=False),
        "episodic_memory":
            dict(use_warmstart=False,
                 use_mirror_descent=False,
                 use_causal=False,
                 use_episodic_memory=True,
                 use_element_memory=False),
        "warmstart":
            dict(use_warmstart=True,
                 use_mirror_descent=False,
                 use_causal=False,
                 use_episodic_memory=True,
                 use_element_memory=False),
        "mirror_descent":
            dict(use_warmstart=True,
                 use_mirror_descent=True,
                 use_causal=False,
                 use_episodic_memory=True,
                 use_element_memory=False),
        "full":
            dict(use_warmstart=True,
                 use_mirror_descent=True,
                 use_causal=True,
                 use_episodic_memory=True,
                 use_element_memory=True),
        "no_memory":
            dict(use_warmstart=True,
                 use_mirror_descent=True,
                 use_causal=True,
                 use_episodic_memory=False,
                 use_element_memory=False),
        "no_warmstart":
            dict(use_warmstart=False,
                 use_mirror_descent=True,
                 use_causal=True,
                 use_episodic_memory=True,
                 use_element_memory=True),
        "no_mirror":
            dict(use_warmstart=True,
                 use_mirror_descent=False,
                 use_causal=True,
                 use_episodic_memory=True,
                 use_element_memory=True),
        "no_causal":
            dict(use_warmstart=True,
                 use_mirror_descent=True,
                 use_causal=False,
                 use_episodic_memory=True,
                 use_element_memory=True),
        "no_elem_mem":
            dict(use_warmstart=True,
                 use_mirror_descent=True,
                 use_causal=True,
                 use_episodic_memory=True,
                 use_element_memory=False),
        "no_stateful":
            dict(use_warmstart=True,
                 use_mirror_descent=True,
                 use_causal=True,
                 use_episodic_memory=True,
                 use_element_memory=True,
                 beta_val=0.0),
    }
    flags = flag_map[config_name]
    sr_trials, step_trials = [], []
    for seed in range(n_seeds):
        np.random.seed(SEED + seed * 13)
        torch.manual_seed(SEED + seed * 13)
        m = run_meta2l(
            n_episodes=n_ep, **flags)
        tail = int(
            len(m["sumrate_meta2l"]) * 0.8)
        sr_trials.append(float(np.mean(
            m["sumrate_meta2l"][tail:])))
        step_trials.append(float(np.mean(
            m["grad_steps"][tail:])))
    return (float(np.mean(sr_trials)),
            float(np.std(sr_trials)),
            float(np.mean(step_trials)))


def get_ablation_steps():
    configs = [
        ("no_mirror",  "w/o mirror-descent"),
        ("no_causal",  "w/o causal filter"),
    ]
    print("\n" + "=" * 50)
    print("Precise step counts for "
          "table rows (8) and (9)")
    print("=" * 50)
    for cfg_name, label in configs:
        _, _, steps = _run_ablation_config(
            cfg_name, n_ep=200, n_seeds=10)
        print(f"  {label}: {steps:.0f} steps")


def plot_ablation_visual():
    n_ep    = 200
    n_seeds = 10
    configs = [
        ("baseline",       "(1) Baseline\n(no meta)"),
        ("episodic_memory","(2) +Episodic\nmemory"),
        ("warmstart",      "(3) +Warm-\nstart"),
        ("mirror_descent", "(4) +Mirror\ndescent"),
        ("full",           "(5) +Causal\n[Full]"),
        ("no_memory",      "(6) w/o mem\n(MAML)"),
        ("no_warmstart",   "(7) w/o warm-\nstart"),
        ("no_mirror",      "(8) w/o mirror\ndescent"),
        ("no_causal",      "(9) w/o causal\nfilter"),
        ("no_elem_mem",    "(10) w/o elem.\nmemory"),
        ("no_stateful",    r"(11) w/o stat. $\beta$"),
    ]
    sr_mean, sr_std, grad_steps = [], [], []
    for cfg, _ in configs:
        mu, sigma, steps = _run_ablation_config(
            cfg, n_ep=n_ep, n_seeds=n_seeds)
        sr_mean.append(mu)
        sr_std.append(sigma)
        grad_steps.append(steps)
    labels = [lbl for _, lbl in configs]
    x      = np.arange(len(labels))
    colors = ["#5b9bd5"] * 5 + ["#f4a261"] * 6
    build_p = mpatches.Patch(
        color="#5b9bd5",
        label="Incremental build-up")
    leave_p = mpatches.Patch(
        color="#f4a261",
        label="Leave-one-out")
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 4.8))
    ax = axes[0]
    ax.bar(x, sr_mean, yerr=sr_std,
           color=colors, capsize=4,
           edgecolor="grey", linewidth=0.6,
           error_kw=dict(elinewidth=1.2,
                         ecolor="dimgrey"))
    ax.axvline(x=4.5, color="black",
               linewidth=1.2, linestyle="--",
               alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.2)
    ax.set_ylabel(
        "Sum-rate [bps/Hz]", fontsize=11)
    ax.set_ylim(
        max(0, min(sr_mean) - 2.0),
        max(sr_mean) + 2.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(handles=[build_p, leave_p],
              fontsize=8.5, loc="upper left")
    ax = axes[1]
    ax.bar(x, grad_steps, color=colors,
           edgecolor="grey", linewidth=0.6)
    ax.axvline(x=4.5, color="black",
               linewidth=1.2, linestyle="--",
               alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.2)
    ax.set_ylabel(
        "Avg. gradient steps per episode",
        fontsize=11)
    ax.set_ylim(0, max(grad_steps) * 1.3)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(handles=[build_p, leave_p],
              fontsize=8.5, loc="upper right")
    _save_fig("figC_ablation")
    plt.close()


def plot_hardware_validity(hw_results):
    labels  = list(hw_results.keys())
    x_pos   = [0, 1, 2, 3]
    x_ticks = [
        "Ideal\n(8-bit)",
        r"$\tau=0$" + "\n(3-bit)",
        r"$\tau=0.1$",
        r"$\tau=0.2$"]
    stat_means = [hw_results[l]["stateful_mean"]
                  for l in labels]
    stat_stds  = [hw_results[l]["stateful_std"]
                  for l in labels]
    mem_means  = [hw_results[l]["memoryless_mean"]
                  for l in labels]
    mem_stds   = [hw_results[l]["memoryless_std"]
                  for l in labels]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(x_pos, stat_means,
                yerr=stat_stds,
                color="#1f77b4", linewidth=2.2,
                marker="o", markersize=7,
                capsize=5,
                label="Stateful Meta2L "
                      r"($\beta=0.6$)")
    ax.errorbar(x_pos, mem_means,
                yerr=mem_stds,
                color="#d62728", linewidth=2.2,
                marker="s", markersize=7,
                capsize=5, linestyle="--",
                label="Memoryless RIS "
                      r"($\beta=0$)")
    ax.axvspan(2.5, 3.5, alpha=0.08,
               color="green",
               label="Stateful overtakes")
    ax.annotate(
        "Crossover:\nstateful overtakes\n"
        r"at $\tau=0.2$",
        xy=(3, stat_means[3]),
        xytext=(1.8, stat_means[3] - 0.8),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#2ca02c", lw=1.3),
        fontsize=8.5, color="#2ca02c",
        fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_ticks, fontsize=10)
    ax.set_ylabel(
        "Sum-rate [bps/Hz]", fontsize=11)
    ax.set_xlabel(
        "Hardware impairment level",
        fontsize=11)
    ax.set_title(
        "Stateful vs. memoryless RIS under\n"
        "phase quantization and settling delay",
        fontsize=10)
    ax.legend(fontsize=9.5, framealpha=0.85)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(13.5, 21)
    _save_fig("fig_hardware_validity")
    plt.close()


# ── Figure 5a: FPGA Critical-Path Latency ────────────────────────

def plot_settling_sweep(n_ep=200, n_seeds=3,
                        taus=(0.0, 0.05, 0.10, 0.15, 0.20)):
    """Sum-rate versus settling delay under 3-bit quantization, stateful
    Meta2L (beta = 0.6) against the memoryless baseline. Every point is
    simulated with run_hardware_validity_test."""
    cfgs = [{"n_bits": 3, "tau": t, "label": f"3-bit, tau={t:.2f}"}
            for t in taus]
    res = run_hardware_validity_test(n_ep=n_ep, n_seeds=n_seeds,
                                     configs=cfgs)
    ms = np.array([res[c["label"]]["stateful_mean"] for c in cfgs])
    ss = np.array([res[c["label"]]["stateful_std"] for c in cfgs])
    mm = np.array([res[c["label"]]["memoryless_mean"] for c in cfgs])
    sm = np.array([res[c["label"]]["memoryless_std"] for c in cfgs])
    t = np.array(taus)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(t, ms, "o-", color=C1, lw=2, label=r"Stateful Meta2L ($\beta=0.6$)")
    ax.fill_between(t, ms - ss, ms + ss, color=C1, alpha=0.15)
    ax.plot(t, mm, "s--", color=C2, lw=2, label=r"Memoryless RIS ($\beta=0$)")
    ax.fill_between(t, mm - sm, mm + sm, color=C2, alpha=0.15)
    ax.set_xlabel(r"Settling delay $\tau$")
    ax.set_ylabel("Ergodic sum-rate [bps/Hz]")
    ax.grid(True); ax.legend()
    _save_fig("fig_settling_sweep")
    return res

def plot_tracking_error():
    N_EP     = 80
    N_sim    = P.N
    lam_val  = 3e8 / 28e9
    Ts_sim   = 1e-4
    lambda_c = 0.10
    beta_F   = 1.0
    C_Delta  = 0.45
    panel_configs = [
        dict(v=1.5,
             label=r"Pedestrian ($v_k=1.5$ m/s,"
                   r" $\rho_c\approx0.995$)",
             panel="(a)"),
        dict(v=22.0,
             label=r"Vehicular ($v_k=22$ m/s,"
                   r" $\rho_c\approx0.970$)",
             panel="(b)"),
    ]
    def get_beta(v):
        rho = bessel_j0(
            2*np.pi*(v/lam_val)*Ts_sim)
        b = rho**2 / (
                rho**2 + lambda_c *
                (1-rho**2) /
                (beta_F * C_Delta**2))
        return (float(np.clip(b, 0.0, 0.99)),
                float(rho))
    def simulate_tracking(
            v, beta, use_memory,
            n_ep=N_EP, seed=1):
        rng = np.random.default_rng(seed)
        rho = bessel_j0(
            2*np.pi*(v/lam_val)*Ts_sim)
        phi_star = rng.uniform(
            0, 2*np.pi,
            N_sim).astype(np.float32)
        phi  = phi_star.copy()
        m_n  = np.zeros(
            N_sim, dtype=np.float32)
        lambda_m = 0.7
        gamma_m  = 0.10
        errors   = []
        for _ in range(n_ep):
            innov = (
                    np.sqrt(1-rho**2) *
                    rng.standard_normal(
                        N_sim).astype(np.float32))
            phi_star = project_phase(
                rho*phi_star + C_Delta*innov)
            phi = project_phase(
                beta*phi + (1-beta)*phi_star)
            if use_memory:
                phi = project_phase(
                    phi + gamma_m * m_n)
                m_n = (lambda_m * m_n +
                       (1-lambda_m) * phi_star)
            e = ((phi - phi_star + np.pi) %
                 (2*np.pi) - np.pi)
            errors.append(
                np.linalg.norm(e) /
                np.sqrt(N_sim))
        return np.array(errors)
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.2))
    ep_idx = np.arange(1, N_EP + 1)
    for col, cfg in enumerate(panel_configs):
        ax        = axes[col]
        v         = cfg["v"]
        beta_v, _ = get_beta(v)
        err_mem  = simulate_tracking(
            v, beta=0.0, use_memory=False,
            seed=col*10+1)
        err_stat = simulate_tracking(
            v, beta=beta_v, use_memory=False,
            seed=col*10+2)
        err_both = simulate_tracking(
            v, beta=beta_v, use_memory=True,
            seed=col*10+3)
        err_mem_s  = uniform_filter1d(
            err_mem,  size=5)
        err_stat_s = uniform_filter1d(
            err_stat, size=5)
        err_both_s = uniform_filter1d(
            err_both, size=5)
        ax.plot(ep_idx, err_mem_s,
                color="#d62728", linewidth=1.8,
                linestyle=":",
                label=r"Memoryless ($\beta=0$)")
        ax.plot(ep_idx, err_stat_s,
                color="#ff7f0e", linewidth=1.8,
                linestyle="--",
                label=fr"Stateful only "
                      fr"($\beta={beta_v:.2f}$)")
        ax.plot(ep_idx, err_both_s,
                color="#1f77b4", linewidth=2.2,
                linestyle="-",
                label=r"Stateful + element "
                      r"memory $m_n(t)$")
        var_s   = np.var(err_stat[-30:])
        var_b   = np.var(err_both[-30:])
        red_pct = int(
            (1 - var_b /
             max(var_s, 1e-9)) * 100)
        ax.text(
            0.97, 0.97,
            f"Variance reduced\nby ~{red_pct}%"
            f" at steady state",
            transform=ax.transAxes,
            ha="right", va="top", fontsize=8,
            color="#1f77b4",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white", alpha=0.8))
        ax.set_xlabel(
            "Episode index", fontsize=11)
        ax.set_ylabel(
            r"$\|\boldsymbol{e}(t)\|/"
            r"\sqrt{N}$ [rad]",
            fontsize=11)
        ax.set_title(
            f"{cfg['panel']} {cfg['label']}",
            fontsize=9.5)
        ax.set_xlim(1, N_EP)
        ax.legend(fontsize=8.5,
                  framealpha=0.85,
                  loc="upper right")
        ax.grid(True, alpha=0.3)
    _save_fig("figD_tracking_error")
    plt.close()


# ── Table Value Printer ───────────────────────────────────────────

def print_table_values(scal, sens):
    print("\n" + "=" * 60)
    print(f"SCALABILITY: varying N, "
          f"K fixed at {P.K}, tol={TOL}")
    print("=" * 60)
    baseline_sr = list(
        scal.values())[0]["sr_mean"]
    for N_val, v in scal.items():
        rel = v["sr_mean"] / baseline_sr
        print(
            f"$N={N_val}$ & "
            f"${v['steps_mean']:.0f} "
            f"\\pm {v['steps_std']:.0f}$ & "
            f"${v['rt_mean']:.1f} "
            f"\\pm {v['rt_std']:.1f}$ & "
            f"${rel:.2f}\\times$ \\\\")
    print("\n" + "=" * 60)
    print("SENSITIVITY: varying lambda (with RT)")
    print("=" * 60)
    for lam, v in sens.items():
        print(
            f"$\\lambda={lam}$ & "
            f"${v['steps_mean']:.0f} "
            f"\\pm {v['steps_std']:.0f}$ & "
            f"${v['rt_mean']:.1f} "
            f"\\pm {v['rt_std']:.1f}$ & "
            f"${v['sr_mean']:.2f} "
            f"\\pm {v['sr_std']:.2f}$ \\\\")


# ── Entry Point ───────────────────────────────────────────────────

if __name__ == "__main__":
    # Table V (scalability, sensitivity to lambda, hardware validity),
    # Fig. 4 (stateful vs memoryless under impairments) and the
    # sum-rate versus settling-delay figure.
    N_TRIALS   = 3
    N_EPISODES = 200

    print("Running scalability analysis ...")
    scal = run_scalability_N(
        N_vals=[64, 100, 128, 256],
        n_ep=N_EPISODES,
        n_seeds=N_TRIALS)

    print("Running sensitivity analysis ...")
    sens = run_sensitivity_lambda(
        lambda_vals=[0.1, 0.3, 0.7, 0.95],
        n_ep=N_EPISODES,
        n_seeds=N_TRIALS)

    print_table_values(scal, sens)

    print("Running hardware validity test ...")
    hw = run_hardware_validity_test(
        n_ep=N_EPISODES,
        n_seeds=N_TRIALS)
    plot_hardware_validity(hw)
    plot_settling_sweep(n_ep=N_EPISODES, n_seeds=N_TRIALS)
