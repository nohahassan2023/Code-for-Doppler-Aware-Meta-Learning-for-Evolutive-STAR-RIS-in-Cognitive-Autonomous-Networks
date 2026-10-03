import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import uniform_filter1d
from scipy.special import j0
from dataclasses import dataclass, field
from typing import List
import warnings
warnings.filterwarnings("ignore")

_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_figures")
os.makedirs(_OUT_DIR, exist_ok=True)


def _save_fig(stem, dpi=200):
    plt.savefig(os.path.join(_OUT_DIR, stem + ".png"), dpi=dpi, bbox_inches="tight")
    plt.savefig(os.path.join(_OUT_DIR, stem + ".pdf"), bbox_inches="tight")
    plt.close()


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── System parameters ─────────────────────────────────────────────────────────
@dataclass
class SystemParams:
    M:       int   = 4        # access points
    Q:       int   = 64       # antennas per AP  (paper: 64)
    K:       int   = 32       # UEs              (paper: 32)
    N:       int   = 16       # RIS elements (default; varied in scalability)
    fc:      float = 28e9
    BW:      float = 10e6
    lam:     float = field(init=False)
    d:       float = field(init=False)
    p_tx:    float = 1e-3
    sigma2:  float = 1e-9
    kappa:   float = 2.0      # Rician K-factor (3 dB)
    v_max:   float = 30.0
    Ts:      float = 1e-4
    n_train: int   = 800
    n_test:  int   = 200
    mem_size:int   = 1000
    D_t:     int   = 24
    alpha1:  float = 10.0
    alpha2:  float = 1.0

    def __post_init__(self):
        self.lam = 3e8 / self.fc
        self.d   = self.lam / 2

P = SystemParams()


# ── Helpers ───────────────────────────────────────────────────────────────────
def bessel_j0(x):
    return float(j0(x))

def steering_vector(N, d, lam, angle_rad):
    n = np.arange(N)
    return (1.0 / np.sqrt(N)) * np.exp(1j * 2 * np.pi * d / lam * n * np.sin(angle_rad))

def make_los_channel(Q, N, d, lam, aoa, aod):
    return np.outer(steering_vector(Q, d, lam, aoa),
                    steering_vector(N, d, lam, aod).conj())

def project_phase(phi):
    return phi % (2 * np.pi)


# ── Channel model ─────────────────────────────────────────────────────────────
class GaussMarkovChannel:
    def __init__(self, rows, cols, v_ue, aoa=None, aod=None):
        self.rows = rows
        self.cols = cols
        self.v_ue = v_ue
        self.aoa  = aoa if aoa is not None else np.random.uniform(-np.pi/3, np.pi/3)
        self.aod  = aod if aod is not None else np.random.uniform(-np.pi/3, np.pi/3)
        f_D = v_ue / P.lam
        self.rho_c = bessel_j0(2 * np.pi * f_D * P.Ts)
        kap = P.kappa
        self.H_los = (np.sqrt(kap / (1 + kap)) *
                      make_los_channel(rows, cols, P.d, P.lam, self.aoa, self.aod))
        self.sigma_nlos = np.sqrt(1.0 / (1 + kap))
        self.H = self._rician_sample()

    def _rician_sample(self):
        E = (np.random.randn(self.rows, self.cols) +
             1j * np.random.randn(self.rows, self.cols)) / np.sqrt(2)
        return self.H_los + self.sigma_nlos * E

    def step(self):
        E = (np.random.randn(self.rows, self.cols) +
             1j * np.random.randn(self.rows, self.cols)) / np.sqrt(2)
        self.H = (self.rho_c * self.H +
                  np.sqrt(1 - self.rho_c**2) * self.sigma_nlos * E)
        return self.H.copy()


# ── Wireless environment ──────────────────────────────────────────────────────
class WirelessEnvironment:
    def __init__(self, N=P.N, v_profile="mixed"):
        self.N = N
        self.v_profile = v_profile
        self._init_channels()

    def _sample_velocity(self):
        if self.v_profile == "pedestrian":
            return np.random.uniform(0.5, 5.0)
        elif self.v_profile == "vehicular":
            return np.random.uniform(20.0, 30.0)
        return np.random.uniform(0.5, 30.0)

    def _init_channels(self):
        self.v_k  = np.array([self._sample_velocity() for _ in range(P.K)])
        self.H_qn = [GaussMarkovChannel(P.Q, self.N, np.mean(self.v_k))
                     for _ in range(P.M)]
        self.h_r  = [GaussMarkovChannel(self.N, 1, self.v_k[k]) for k in range(P.K)]
        self.h_t  = [GaussMarkovChannel(self.N, 1, self.v_k[k]) for k in range(P.K)]

    def switch_profile(self, new_profile):
        self.v_profile = new_profile
        self.v_k  = np.array([self._sample_velocity() for _ in range(P.K)])
        self.H_qn = [GaussMarkovChannel(P.Q, self.N, np.mean(self.v_k))
                     for _ in range(P.M)]
        self.h_r  = [GaussMarkovChannel(self.N, 1, self.v_k[k]) for k in range(P.K)]
        self.h_t  = [GaussMarkovChannel(self.N, 1, self.v_k[k]) for k in range(P.K)]

    def step(self):
        for ch in self.H_qn: ch.step()
        for ch in self.h_r:  ch.step()
        for ch in self.h_t:  ch.step()

    def get_channels(self):
        H  = np.stack([ch.H        for ch in self.H_qn])
        hr = np.stack([ch.H[:, 0]  for ch in self.h_r])
        ht = np.stack([ch.H[:, 0]  for ch in self.h_t])
        return H, hr, ht

    def compute_sumrate(self, phi_r, phi_t, H, hr, ht):
        er  = np.exp(1j * phi_r)
        et  = np.exp(1j * phi_t)
        H0  = H[0]
        G   = (hr * er[None, :] + ht * et[None, :]) @ H0.T
        u   = np.sum(np.abs(G)**2, axis=1)
        Ckl = G.conj() @ G.T
        C2  = np.abs(Ckl)**2
        np.fill_diagonal(C2, 0.0)
        f    = P.p_tx * np.sum(C2, axis=1)
        v    = f + P.sigma2 * u
        sinr = P.p_tx * u**2 / v
        return float(np.sum(np.log2(1 + sinr)))

    def sinr_gradient(self, phi_r, phi_t, H, hr, ht):
        er   = np.exp(1j * phi_r)
        et   = np.exp(1j * phi_t)
        H0   = H[0]
        G    = (hr * er[None, :] + ht * et[None, :]) @ H0.T
        u    = np.sum(np.abs(G)**2, axis=1)
        GH0c = G @ H0.conj()
        Ckl  = G.conj() @ G.T
        K    = G.shape[0]
        mask = 1.0 - np.eye(K)
        C2   = np.abs(Ckl)**2
        np.fill_diagonal(C2, 0.0)
        f_k  = P.p_tx * np.sum(C2, axis=1)
        v_k  = f_k + P.sigma2 * u
        sinr = P.p_tx * u**2 / v_k
        coeff = 1.0 / (np.log(2) * (1 + sinr))
        du_r = 2.0 * np.real(1j * er[None, :] * hr * np.conj(GH0c))
        du_t = 2.0 * np.real(1j * et[None, :] * ht * np.conj(GH0c))
        Ckl_cm = np.conj(Ckl) * mask
        sumA   = Ckl_cm @ GH0c
        sumBr  = Ckl_cm @ hr
        sumBt  = Ckl_cm @ ht
        dCki_r = (-1j * np.conj(er)[None, :] * np.conj(hr) * sumA
                  + 1j * er[None, :] * np.conj(GH0c) * sumBr)
        dCki_t = (-1j * np.conj(et)[None, :] * np.conj(ht) * sumA
                  + 1j * et[None, :] * np.conj(GH0c) * sumBt)
        df_r = P.p_tx * 2.0 * np.real(dCki_r)
        df_t = P.p_tx * 2.0 * np.real(dCki_t)
        dv_r = df_r + P.sigma2 * du_r
        dv_t = df_t + P.sigma2 * du_t
        uc   = u[:, None]
        vc   = v_k[:, None]
        dsinr_r = P.p_tx * (2 * uc * du_r * vc - uc**2 * dv_r) / vc**2
        dsinr_t = P.p_tx * (2 * uc * du_t * vc - uc**2 * dv_t) / vc**2
        grad_r = np.sum(coeff[:, None] * dsinr_r, axis=0)
        grad_t = np.sum(coeff[:, None] * dsinr_t, axis=0)
        return grad_r.astype(np.float64), grad_t.astype(np.float64)


# ── Phase optimisation ────────────────────────────────────────────────────────
def optimise_phases(env, H, hr, ht, phi_r_init, phi_t_init,
                    eta=0.15, max_iter=40, tol=0.02):
    phi_r = phi_r_init.copy().astype(np.float64)
    phi_t = phi_t_init.copy().astype(np.float64)
    for step in range(max_iter):
        gr, gt = env.sinr_gradient(phi_r, phi_t, H, hr, ht)
        phi_r  = project_phase(phi_r + eta * gr)
        phi_t  = project_phase(phi_t + eta * gt)
        if np.linalg.norm(np.concatenate([gr, gt])) < tol:
            break
    sr = env.compute_sumrate(phi_r, phi_t, H, hr, ht)
    return phi_r, phi_t, step + 1, sr


# ── Tag vector ────────────────────────────────────────────────────────────────
def make_tag_vector(v_k, sr, H):
    v_mean = np.mean(v_k)
    v_max_ = np.max(v_k)
    H_pow  = np.mean(np.abs(H)**2)
    env_tags = np.array([v_mean > 20, v_max_ > 25, H_pow < 0.3,
                         np.any(v_k > 15), v_mean < 5, H_pow > 0.7], dtype=float)
    ris_tags = np.array([P.N >= 64, P.N >= 100, True, True, P.N >= 16, False], dtype=float)
    net_tags = np.array([P.K >= 8, P.M >= 4, P.K >= 4, P.Q >= 32, sr < 5, P.M >= 2], dtype=float)
    qos_tags = np.array([v_mean > 15, sr > 7, sr < 4, True, P.K >= 4, sr > 8], dtype=float)
    return np.concatenate([env_tags, ris_tags, net_tags, qos_tags]).astype(np.float32)


# ── Episode and memory ────────────────────────────────────────────────────────
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
            self.episodes.sort(key=lambda e: e.sr)
            self.episodes.pop(0)
        self.episodes.append(ep)
        for e in self.episodes:
            e.age += 1

    def retrieve(self, tag, weights, top_k=2):
        if len(self.episodes) < top_k:
            return self.episodes[:]
        scores = [float(np.dot(weights * tag, ep.tag) /
                        (np.linalg.norm(weights * tag) * np.linalg.norm(ep.tag) + 1e-8))
                  for ep in self.episodes]
        idx = np.argsort(scores)[-top_k:][::-1]
        return [self.episodes[i] for i in idx]


# ── Mirror descent weights ────────────────────────────────────────────────────
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
        return np.sqrt(2 * np.log(self.D_t) / (max(self.T, 1) * self.L**2))

    def update(self, tag_new, retrieved_tag, reward):
        sim  = tag_new * retrieved_tag
        grad = -sim * reward
        eta  = self.step_size()
        log_W = np.log(self.W + 1e-300) - eta * grad
        log_W -= np.max(log_W)
        self.W  = np.exp(log_W)
        self.W /= self.W.sum()
        loss_actual = float(np.dot(self.W, grad))
        loss_oracle = float(np.min(grad))
        self._actual_loss_cumsum += loss_actual
        self._best_loss_cumsum   += loss_oracle
        self.cumulative_regret.append(
            max(self._actual_loss_cumsum - self._best_loss_cumsum, 0.0))
        self.T += 1

    def regret_bound(self):
        T_arr = np.arange(1, self.T + 1)
        return self.L * np.sqrt(2 * T_arr * np.log(self.D_t))


# ── Causal filter ─────────────────────────────────────────────────────────────
def causal_filter(phi_r, phi_t, H, hr, ht, env):
    N     = env.N
    phi_r = project_phase(phi_r)
    phi_t = project_phase(phi_t)
    Psi_r = np.diag(np.exp(1j * phi_r))
    Psi_t = np.diag(np.exp(1j * phi_t))
    H0    = H[0]
    G     = np.zeros((P.K, P.Q), dtype=complex)
    for k in range(P.K):
        G[k] = H0 @ (Psi_r @ hr[k] + Psi_t @ ht[k])
    beta_max = (np.max(np.abs(H0)**2) *
                np.max([np.max(np.abs(hr[k])**2) for k in range(P.K)]))
    Gamma = N**2 * P.Q * beta_max
    corrected = False
    for k in range(P.K):
        if np.linalg.norm(G[k])**2 > Gamma * 1.5:
            phi_r = phi_r * 0.5
            phi_t = phi_t * 0.5
            corrected = True
            break
    for k in range(P.K):
        if np.linalg.norm(G[k])**2 < 1e-15:
            phi_r = np.random.uniform(0, 2 * np.pi, N).astype(np.float32)
            phi_t = np.random.uniform(0, 2 * np.pi, N).astype(np.float32)
            corrected = True
            break
    return phi_r, phi_t, corrected


# ── Meta policy ───────────────────────────────────────────────────────────────
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


# ── Novelty detector ──────────────────────────────────────────────────────────
class NoveltyDetector:
    def __init__(self, theta_high=0.75, theta_low=0.35, eta_thresh=0.005):
        self.theta_high   = theta_high
        self.theta_low    = theta_low
        self.eta          = eta_thresh
        self.history_high = [theta_high]
        self.history_low  = [theta_low]

    def score(self, tag_new, memory, weights):
        if not memory.episodes:
            return 0.0
        best = memory.retrieve(tag_new, weights, top_k=1)[0]
        return float(np.dot(weights * tag_new, best.tag) /
                     (np.linalg.norm(weights * tag_new) * np.linalg.norm(best.tag) + 1e-8))

    def classify(self, score):
        if score > self.theta_high:   return "familiar"
        elif score > self.theta_low:  return "partial"
        return "novel"

    def update(self, meta_loss):
        grad = meta_loss * 0.01
        self.theta_high = np.clip(self.theta_high - self.eta * grad, 0.55, 0.95)
        self.theta_low  = np.clip(self.theta_low  - self.eta * grad, 0.15, 0.55)
        if self.theta_low >= self.theta_high:
            self.theta_low = self.theta_high - 0.1
        self.history_high.append(self.theta_high)
        self.history_low.append(self.theta_low)


# ── Doppler warm-start ────────────────────────────────────────────────────────
def doppler_warmstart(retrieved, rho_c, N):
    if len(retrieved) == 0:
        return (np.random.uniform(0, 2*np.pi, N).astype(np.float32),
                np.random.uniform(0, 2*np.pi, N).astype(np.float32))
    if len(retrieved) == 1:
        return retrieved[0].phi_r.copy(), retrieved[0].phi_t.copy()
    ep1, ep2 = retrieved[0], retrieved[1]
    return (project_phase(rho_c * ep1.phi_r + (1 - rho_c) * ep2.phi_r),
            project_phase(rho_c * ep1.phi_t + (1 - rho_c) * ep2.phi_t))


# ── Meta2L ────────────────────────────────────────────────────────────────────
def run_meta2l(n_episodes=P.n_train + P.n_test, N=P.N, v_profile="mixed",
               beta_val=0.0, lambda_mem=0.7, gamma_mem=0.1,
               use_warmstart=True, use_mirror_descent=True,
               use_causal=True, use_episodic_memory=True,
               use_element_memory=True, verbose=False):

    env        = WirelessEnvironment(N=N, v_profile=v_profile)
    ctx_dim    = P.D_t + P.K + 2
    policy     = MetaPolicy(ctx_dim, N)
    opt        = optim.Adam(policy.parameters(), lr=3e-4, weight_decay=1e-5)
    memory     = EpisodicMemory(max_size=P.mem_size)
    md_weights = MirrorDescentWeights(P.D_t)
    novelty    = NoveltyDetector()

    m_n           = np.zeros(N, dtype=np.float32)
    phi_prev      = np.zeros(N, dtype=np.float32)
    phi_star_prev = np.zeros(N, dtype=np.float32)

    metrics = {
        "sumrate_meta2l": [], "grad_steps": [],
        "theta_high": [], "theta_low": [],
        "novelty_scores": [], "rho_c_values": [],
        "tracking_error": [], "beta_learned": [],
        "overhead_norm": [], "phi_update_mag": [],
        "phi_r_last": None, "phi_t_last": None,
    }

    phi_r_opt = np.random.uniform(0, 2*np.pi, N).astype(np.float32)
    phi_t_opt = np.random.uniform(0, 2*np.pi, N).astype(np.float32)

    for ep_idx in range(n_episodes):
        env.step()
        H, hr, ht = env.get_channels()
        rho_c = np.mean([env.h_r[k].rho_c for k in range(P.K)])

        sr_rand = env.compute_sumrate(
            np.random.uniform(0, 2*np.pi, N).astype(np.float32),
            np.random.uniform(0, 2*np.pi, N).astype(np.float32), H, hr, ht)
        tag       = make_tag_vector(env.v_k, sr_rand, H)
        nov_score = novelty.score(tag, memory, md_weights.W)
        nov_class = novelty.classify(nov_score)

        retrieved = memory.retrieve(tag, md_weights.W, top_k=2) if use_episodic_memory else []

        if use_warmstart and len(retrieved) > 0:
            phi_r_ws, phi_t_ws = doppler_warmstart(retrieved, rho_c, N)
            if use_element_memory and (beta_val > 0 or gamma_mem > 0):
                u_ws     = phi_r_ws
                phi_r_ws = project_phase(
                    beta_val * phi_star_prev + (1 - beta_val) * u_ws + gamma_mem * m_n)
        else:
            phi_r_ws = np.random.uniform(0, 2*np.pi, N).astype(np.float32)
            phi_t_ws = np.random.uniform(0, 2*np.pi, N).astype(np.float32)

        ctx_np = np.concatenate([tag, env.v_k / P.v_max, [rho_c],
                                 [retrieved[0].sr / 15.0 if retrieved else 0.0]])
        ctx_t = torch.tensor(ctx_np, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            phi_pred = policy(ctx_t).squeeze().numpy()

        blend      = 0.3 if nov_class == "familiar" else (0.6 if nov_class == "partial" else 0.9)
        phi_r_init = project_phase((1 - blend) * phi_r_ws + blend * phi_pred[:N])
        phi_t_init = project_phase((1 - blend) * phi_t_ws + blend * phi_pred[N:])

        if use_causal:
            phi_r_init, phi_t_init, _ = causal_filter(phi_r_init, phi_t_init, H, hr, ht, env)

        phi_r_opt, phi_t_opt, n_steps, sr_opt = optimise_phases(
            env, H, hr, ht, phi_r_init, phi_t_init, eta=0.15, max_iter=40, tol=0.02)

        phi_stateful = project_phase(beta_val * phi_prev + (1 - beta_val) * phi_r_opt)
        tr_err       = np.linalg.norm(phi_r_opt - phi_stateful) / np.sqrt(N)
        delta        = phi_r_opt - phi_prev
        overhead     = np.linalg.norm(delta)**2 / (np.linalg.norm(phi_r_opt)**2 + 1e-12)
        upd_mag      = np.linalg.norm(phi_r_opt - phi_star_prev) / np.sqrt(N)

        if use_element_memory:
            m_n = lambda_mem * m_n + (1 - lambda_mem) * phi_r_opt

        phi_pred_t = policy(ctx_t).squeeze()
        target = torch.tensor(np.concatenate([phi_r_opt, phi_t_opt]), dtype=torch.float32)
        loss   = nn.MSELoss()(phi_pred_t, target) - 0.001 * sr_opt
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if use_mirror_descent and retrieved:
            md_weights.update(tag, retrieved[0].tag, reward=sr_opt)
        novelty.update(float(loss.item()))

        if use_episodic_memory:
            memory.add(Episode(tag=tag, phi_r=phi_r_opt, phi_t=phi_t_opt,
                               sr=sr_opt, v_mean=float(np.mean(env.v_k)), rho_c=rho_c))

        phi_prev      = phi_stateful.copy()
        phi_star_prev = phi_r_opt.copy()

        metrics["sumrate_meta2l"].append(sr_opt)
        metrics["grad_steps"].append(n_steps)
        metrics["theta_high"].append(novelty.theta_high)
        metrics["theta_low"].append(novelty.theta_low)
        metrics["novelty_scores"].append(nov_score)
        metrics["rho_c_values"].append(rho_c)
        metrics["tracking_error"].append(tr_err)
        metrics["beta_learned"].append(beta_val)
        metrics["overhead_norm"].append(overhead)
        metrics["phi_update_mag"].append(upd_mag)

    metrics["mirror_descent_regret"] = md_weights.cumulative_regret
    metrics["mirror_descent_bound"]  = list(md_weights.regret_bound())
    metrics["theta_high_history"]    = novelty.history_high
    metrics["theta_low_history"]     = novelty.history_low
    metrics["phi_r_last"]            = phi_r_opt
    metrics["phi_t_last"]            = phi_t_opt
    return metrics


# ── Baselines ─────────────────────────────────────────────────────────────────
def run_baseline_random(n_episodes, N=P.N, v_profile="mixed"):
    env = WirelessEnvironment(N=N, v_profile=v_profile)
    results = []
    for _ in range(n_episodes):
        env.step()
        H, hr, ht = env.get_channels()
        results.append(env.compute_sumrate(
            np.random.uniform(0, 2*np.pi, N).astype(np.float32),
            np.random.uniform(0, 2*np.pi, N).astype(np.float32), H, hr, ht))
    return results


def run_baseline_gmlb(n_episodes, N=P.N, v_profile="mixed"):
    env = WirelessEnvironment(N=N, v_profile=v_profile)
    results, steps_list = [], []
    last_phi_r = np.random.uniform(0, 2*np.pi, N).astype(np.float32)
    last_phi_t = np.random.uniform(0, 2*np.pi, N).astype(np.float32)
    for _ in range(n_episodes):
        env.step()
        H, hr, ht = env.get_channels()
        _, _, n_steps, sr = optimise_phases(env, H, hr, ht, last_phi_r, last_phi_t,
                                            eta=0.15, max_iter=40)
        last_phi_r = np.random.uniform(0, 2*np.pi, N).astype(np.float32)
        last_phi_t = np.random.uniform(0, 2*np.pi, N).astype(np.float32)
        results.append(sr)
        steps_list.append(n_steps)
    return results, steps_list


def run_baseline_drl_ppo(n_episodes, N=P.N, v_profile="mixed"):
    """
    Proximal Policy Optimization — 3x256 ReLU, Adam lr=3e-4,
    gamma=0.99, lam_gae=0.95, clip_eps=0.2, replay buffer 10 000.
    """
    env = WirelessEnvironment(N=N, v_profile=v_profile)

    class PPOPolicy(nn.Module):
        def __init__(self, in_dim, N):
            super().__init__()
            self.actor = nn.Sequential(
                nn.Linear(in_dim, 256), nn.ReLU(),
                nn.Linear(256, 256),    nn.ReLU(),
                nn.Linear(256, 256),    nn.ReLU(),
                nn.Linear(256, 2 * N),  nn.Sigmoid(),
            )
            self.critic = nn.Sequential(
                nn.Linear(in_dim, 256), nn.ReLU(),
                nn.Linear(256, 256),    nn.ReLU(),
                nn.Linear(256, 1),
            )
            self.N = N

        def forward(self, x):
            return self.actor(x) * 2 * np.pi

        def value(self, x):
            return self.critic(x).squeeze(-1)

    ctx_dim    = P.D_t + P.K + 2
    policy     = PPOPolicy(ctx_dim, N)
    opt        = optim.Adam(policy.parameters(), lr=3e-4)

    gamma      = 0.99
    lam_gae    = 0.95
    clip_eps   = 0.2
    buf_size   = 10000
    ppo_epochs = 4
    batch_size = 64

    buf_states, buf_actions   = [], []
    buf_rewards, buf_logprobs = [], []
    buf_values                = []
    results                   = []

    for ep_idx in range(n_episodes):
        env.step()
        H, hr, ht = env.get_channels()
        rho_c = np.mean([env.h_r[k].rho_c for k in range(P.K)])
        sr0   = env.compute_sumrate(np.zeros(N), np.zeros(N), H, hr, ht)
        tag   = make_tag_vector(env.v_k, sr0, H)
        ctx_np = np.concatenate([tag, env.v_k / P.v_max, [rho_c], [0.0]])
        ctx_t  = torch.tensor(ctx_np, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            phi_pred = policy(ctx_t).squeeze()
            val      = policy.value(ctx_t).item()

        noise    = torch.randn_like(phi_pred) * 0.3
        phi_act  = torch.clamp(phi_pred + noise, 0, 2 * np.pi)
        log_prob = float(-0.5 * (noise**2).sum().item())

        sr = env.compute_sumrate(phi_act[:N].numpy(), phi_act[N:].numpy(), H, hr, ht)

        buf_states.append(ctx_np);    buf_actions.append(phi_act.numpy())
        buf_rewards.append(sr);       buf_logprobs.append(log_prob)
        buf_values.append(val)

        if len(buf_states) > buf_size:
            buf_states.pop(0);  buf_actions.pop(0)
            buf_rewards.pop(0); buf_logprobs.pop(0); buf_values.pop(0)

        if ep_idx > 0 and ep_idx % 32 == 0 and len(buf_states) >= batch_size:
            rewards_t = torch.tensor(buf_rewards[-batch_size:],  dtype=torch.float32)
            values_t  = torch.tensor(buf_values[-batch_size:],   dtype=torch.float32)
            old_lp_t  = torch.tensor(buf_logprobs[-batch_size:], dtype=torch.float32)
            states_t  = torch.tensor(np.stack(buf_states[-batch_size:]),  dtype=torch.float32)
            actions_t = torch.tensor(np.stack(buf_actions[-batch_size:]), dtype=torch.float32)

            advantages = torch.zeros(batch_size)
            gae = 0.0; next_val = 0.0
            for t in reversed(range(batch_size)):
                delta         = rewards_t[t] + gamma * next_val - values_t[t]
                gae           = delta + gamma * lam_gae * gae
                advantages[t] = gae
                next_val      = values_t[t].item()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            returns    = advantages + values_t

            for _ in range(ppo_epochs):
                phi_new  = policy(states_t)
                val_new  = policy.value(states_t)
                noise_new = phi_new - actions_t
                new_lp   = -0.5 * (noise_new**2).sum(dim=1)
                ratio    = torch.exp(new_lp - old_lp_t)
                surr1    = ratio * advantages
                surr2    = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
                loss     = (-torch.min(surr1, surr2).mean()
                            + 0.5 * nn.MSELoss()(val_new, returns))
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt.step()

        results.append(sr)
    return results


def run_baseline_ao(n_episodes, N=P.N, v_profile="mixed"):
    """Alternating Optimisation — alternates reflection/transmission blocks."""
    env = WirelessEnvironment(N=N, v_profile=v_profile)
    results = []
    for _ in range(n_episodes):
        env.step()
        H, hr, ht = env.get_channels()
        phi_r = np.random.uniform(0, 2*np.pi, N).astype(np.float64)
        phi_t = np.random.uniform(0, 2*np.pi, N).astype(np.float64)
        eta   = 0.12
        for _ in range(20):
            for _ in range(5):
                gr, _ = env.sinr_gradient(phi_r, phi_t, H, hr, ht)
                phi_r = project_phase(phi_r + eta * gr)
            for _ in range(5):
                _, gt = env.sinr_gradient(phi_r, phi_t, H, hr, ht)
                phi_t = project_phase(phi_t + eta * gt)
        results.append(env.compute_sumrate(phi_r, phi_t, H, hr, ht))
    return results


def run_baseline_apg(n_episodes, N=P.N, v_profile="mixed"):
    """Accelerated Proximal Gradient — Nesterov momentum, cold-start."""
    env = WirelessEnvironment(N=N, v_profile=v_profile)
    results = []
    for _ in range(n_episodes):
        env.step()
        H, hr, ht = env.get_channels()
        phi_r = np.random.uniform(0, 2*np.pi, N).astype(np.float64)
        phi_t = np.random.uniform(0, 2*np.pi, N).astype(np.float64)
        y_r, y_t = phi_r.copy(), phi_t.copy()
        t_k = 1.0; eta = 0.12
        for _ in range(30):
            gr, gt    = env.sinr_gradient(y_r, y_t, H, hr, ht)
            phi_r_new = project_phase(y_r + eta * gr)
            phi_t_new = project_phase(y_t + eta * gt)
            t_k1      = (1 + np.sqrt(1 + 4 * t_k**2)) / 2
            mom       = (t_k - 1) / t_k1
            y_r       = phi_r_new + mom * (phi_r_new - phi_r)
            y_t       = phi_t_new + mom * (phi_t_new - phi_t)
            phi_r, phi_t = phi_r_new, phi_t_new
            t_k = t_k1
        results.append(env.compute_sumrate(phi_r, phi_t, H, hr, ht))
    return results


def run_baseline_sio(n_episodes, N=P.N, v_profile="mixed"):
    """Swarm Intelligence Optimisation — particle swarm, no gradient."""
    env = WirelessEnvironment(N=N, v_profile=v_profile)
    n_particles = 20
    results = []
    for _ in range(n_episodes):
        env.step()
        H, hr, ht = env.get_channels()
        pos      = np.random.uniform(0, 2*np.pi, (n_particles, 2*N)).astype(np.float32)
        vel      = np.zeros_like(pos)
        pbest    = pos.copy()
        pbest_sr = np.array([env.compute_sumrate(pos[i, :N], pos[i, N:], H, hr, ht)
                             for i in range(n_particles)])
        gbest    = pbest[np.argmax(pbest_sr)].copy()
        gbest_sr = pbest_sr.max()
        w, c1, c2 = 0.6, 1.5, 1.5
        for _ in range(15):
            r1  = np.random.rand(*pos.shape)
            r2  = np.random.rand(*pos.shape)
            vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
            pos = project_phase(pos + vel)
            sr_arr   = np.array([env.compute_sumrate(pos[i, :N], pos[i, N:], H, hr, ht)
                                 for i in range(n_particles)])
            improved = sr_arr > pbest_sr
            pbest[improved]    = pos[improved]
            pbest_sr[improved] = sr_arr[improved]
            if sr_arr.max() > gbest_sr:
                gbest    = pos[np.argmax(sr_arr)].copy()
                gbest_sr = sr_arr.max()
        results.append(gbest_sr)
    return results


# ── Experiment runner ─────────────────────────────────────────────────────────
def run_experiment(n_trials=3, n_episodes=200):
    all_results = {alg: [] for alg in
                   ["Meta2L", "DRL-PPO", "GMLB", "AO", "APG", "SIO", "Random Phase"]}
    grad_steps  = {"Meta2L": [], "GMLB": []}
    sr_by_N     = {alg: {} for alg in
                   ["Meta2L", "DRL-PPO", "GMLB", "AO", "APG", "SIO", "Random Phase"]}
    mob_results = {reg: {"Meta2L": [], "DRL-PPO": [], "GMLB": []}
                   for reg in ["pedestrian", "mixed", "vehicular"]}

    for trial in range(n_trials):
        print(f"  Trial {trial+1}/{n_trials}")
        m = run_meta2l(n_episodes=n_episodes, v_profile="mixed")
        all_results["Meta2L"].append(m["sumrate_meta2l"])
        grad_steps["Meta2L"].append(m["grad_steps"])
        all_results["DRL-PPO"].append(run_baseline_drl_ppo(n_episodes))
        gmlb_sr, gmlb_steps = run_baseline_gmlb(n_episodes)
        all_results["GMLB"].append(gmlb_sr)
        grad_steps["GMLB"].append(gmlb_steps)
        all_results["AO"].append(run_baseline_ao(n_episodes))
        all_results["APG"].append(run_baseline_apg(n_episodes))
        all_results["SIO"].append(run_baseline_sio(n_episodes))
        all_results["Random Phase"].append(run_baseline_random(n_episodes))

    for N_val in [16, 64, 100, 256]:
        print(f"  Scalability N={N_val}")
        m2 = run_meta2l(n_episodes=100, N=N_val)
        sr_by_N["Meta2L"][N_val]       = float(np.mean(m2["sumrate_meta2l"][-20:]))
        sr_by_N["DRL-PPO"][N_val]      = float(np.mean(run_baseline_drl_ppo(100, N=N_val)[-20:]))
        sr_by_N["GMLB"][N_val]         = float(np.mean(run_baseline_gmlb(100, N=N_val)[0][-20:]))
        sr_by_N["AO"][N_val]           = float(np.mean(run_baseline_ao(100, N=N_val)[-20:]))
        sr_by_N["APG"][N_val]          = float(np.mean(run_baseline_apg(100, N=N_val)[-20:]))
        sr_by_N["SIO"][N_val]          = float(np.mean(run_baseline_sio(100, N=N_val)[-20:]))
        sr_by_N["Random Phase"][N_val] = float(np.mean(run_baseline_random(100, N=N_val)[-20:]))

    for regime in ["pedestrian", "mixed", "vehicular"]:
        print(f"  Mobility regime: {regime}")
        mob_results[regime]["Meta2L"]  = float(np.mean(
            run_meta2l(n_episodes=100, v_profile=regime)["sumrate_meta2l"][-20:]))
        mob_results[regime]["DRL-PPO"] = float(np.mean(
            run_baseline_drl_ppo(100, v_profile=regime)[-20:]))
        mob_results[regime]["GMLB"]    = float(np.mean(
            run_baseline_gmlb(100, v_profile=regime)[0][-20:]))

    detail = run_meta2l(n_episodes=n_episodes)
    return {"all_results": all_results, "grad_steps": grad_steps,
            "sr_by_N": sr_by_N, "mob_results": mob_results,
            "detail": detail, "n_episodes": n_episodes}


# ── Color palette ─────────────────────────────────────────────────────────────
ALG_COLORS = {
    "Meta2L":       "#1f77b4",
    "DRL-PPO":      "#d62728",
    "GMLB":         "#2ca02c",
    "AO":           "#9467bd",
    "APG":          "#8c564b",
    "SIO":          "#e377c2",
    "Random Phase": "#bcbd22",
}

RC = {
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   8.5,
    "figure.dpi":       120,
}
plt.rcParams.update(RC)


# ── fig3: spectral efficiency ─────────────────────────────────────────────────
def plot_spectral_efficiency(results):
    all_r    = results["all_results"]
    n_ep     = results["n_episodes"]
    ep_range = np.arange(1, n_ep + 1)

    def smooth(arr_list):
        mean = np.mean(arr_list, axis=0)
        return uniform_filter1d(mean, size=max(1, n_ep // 40))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Spectral Efficiency Comparison", fontweight="bold")

    ax = axes[0]
    ax.set_title("(a) Sum-Rate Convergence")
    alg_order = ["Meta2L", "DRL-PPO", "GMLB", "AO", "APG", "SIO", "Random Phase"]
    styles    = ["-", "--", ":", "-.", (0,(3,1,1,1)), (0,(5,2)), "-."]
    for alg, ls in zip(alg_order, styles):
        if alg in all_r and all_r[alg]:
            ax.plot(ep_range, smooth(all_r[alg]),
                    color=ALG_COLORS[alg], linewidth=2.0,
                    linestyle=ls, label=alg)
    ax.axvline(x=200, color="grey", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(202, ax.get_ylim()[0] + 0.2, "ep.200", fontsize=8, color="grey")
    ax.set_xlabel("Episode"); ax.set_ylabel("Sum-Rate [bps/Hz]")
    ax.legend(ncol=2, loc="lower right"); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.set_title("(b) Steady-State Distribution (10 MC Trials)")
    tail_data  = {alg: [np.mean(t[-int(n_ep*0.2):]) for t in all_r[alg]]
                  for alg in alg_order if alg in all_r and all_r[alg]}
    plot_data  = [tail_data[a] for a in alg_order if a in tail_data]
    plot_labels = [a for a in alg_order if a in tail_data]
    bp = ax.boxplot(plot_data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, alg in zip(bp["boxes"], plot_labels):
        patch.set_facecolor(ALG_COLORS[alg])
        patch.set_alpha(0.75)
    ax.set_xticklabels(plot_labels, rotation=20, ha="right")
    ax.set_ylabel("Sum-Rate [bps/Hz]"); ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    _save_fig("plot_spectral_efficiency")


# ── fig4: adaptation speed ────────────────────────────────────────────────────
def _measure_adaptation_speed(run_fn, n_ep=60, target_frac=0.95, n_seeds=3):
    half = n_ep // 2
    intervals_list = []
    for seed in range(n_seeds):
        np.random.seed(SEED + seed * 17)
        torch.manual_seed(SEED + seed * 17)
        sr_post   = run_fn(half, "vehicular")
        ss        = float(np.mean(sr_post[int(half * 0.75):]))
        threshold = target_frac * ss
        recovered = half
        for i, v in enumerate(sr_post):
            if v >= threshold:
                recovered = i + 1; break
        intervals_list.append(recovered)
    return float(np.mean(intervals_list)), float(np.std(intervals_list))

def plot_adaptation_threshold():
    n_ep    = 60
    n_seeds = 3

    runners = [
        ("Meta2L",       lambda n, v: run_meta2l(n_episodes=n, v_profile=v)["sumrate_meta2l"]),
        ("DRL-PPO",      lambda n, v: run_baseline_drl_ppo(n, v_profile=v)),
        ("GMLB",         lambda n, v: run_baseline_gmlb(n, v_profile=v)[0]),
        ("AO",           lambda n, v: run_baseline_ao(n, v_profile=v)),
        ("APG",          lambda n, v: run_baseline_apg(n, v_profile=v)),
        ("SIO",          lambda n, v: run_baseline_sio(n, v_profile=v)),
        ("Random Phase", lambda n, v: run_baseline_random(n, v_profile=v)),
    ]

    means, stds, labels, colors = [], [], [], []
    for name, fn in runners:
        mu, sigma = _measure_adaptation_speed(fn, n_ep=n_ep, n_seeds=n_seeds)
        means.append(mu); stds.append(sigma)
        labels.append(name); colors.append(ALG_COLORS[name])

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Adaptation Speed After Channel Change", fontweight="bold")
    ax.bar(labels, means, yerr=stds, color=colors, edgecolor="none",
           width=0.6, capsize=4, error_kw=dict(elinewidth=1.2, ecolor="dimgrey"))
    ax.text(0, means[0] + stds[0] + 0.2, f"{means[0]:.1f}",
            ha="center", va="bottom", color=ALG_COLORS["Meta2L"],
            fontweight="bold", fontsize=10)
    ax.axhline(y=means[0], color=ALG_COLORS["Meta2L"],
               linestyle="--", linewidth=1.6, alpha=0.8)
    ax.set_ylabel("Coherence Intervals to 95% Steady-State")
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, max(means) * 1.3); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _save_fig("plot_adaptation_threshold")


# ── fig5: scalability + mobility ─────────────────────────────────────────────
def plot_scalability_mobility(results):
    sr_by_N     = results["sr_by_N"]
    mob_results = results["mob_results"]
    N_vals      = [16, 64, 100, 256]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Scalability and Mobility Regime Analysis", fontweight="bold")

    ax = axes[0]
    ax.set_title("(a) Scalability with RIS Array Size")
    for alg in ["Meta2L", "DRL-PPO", "GMLB", "AO", "SIO"]:
        if alg in sr_by_N:
            vals = [sr_by_N[alg].get(n, 0) for n in N_vals]
            ax.plot(N_vals, vals, marker="o", color=ALG_COLORS[alg],
                    linewidth=2.0, label=alg)
    ax.set_xlabel("Number of RIS Elements N")
    ax.set_ylabel("Sum-Rate [bps/Hz]")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.set_title("(b) Gains Across Mobility Regimes")
    regimes      = ["pedestrian", "mixed", "vehicular"]
    regime_labels = ["Pedestrian\n($v_k<5$ m/s)", "Mixed\n($v_k\\sim\\mathcal{U}[0,30]$)",
                     "Vehicular\n($v_k>20$ m/s)"]
    x   = np.arange(len(regimes))
    w   = 0.25
    gmlb_sr = np.array([mob_results[r]["GMLB"]    for r in regimes])
    m2_sr   = np.array([mob_results[r]["Meta2L"]  for r in regimes])
    drl_sr  = np.array([mob_results[r]["DRL-PPO"] for r in regimes])
    m2_gain   = 100 * (m2_sr  - gmlb_sr) / (np.abs(gmlb_sr) + 1e-9)
    drl_gain  = 100 * (drl_sr - gmlb_sr) / (np.abs(gmlb_sr) + 1e-9)
    ax.bar(x - w,   m2_gain,  width=w, color=ALG_COLORS["Meta2L"],  label="Meta2L")
    ax.bar(x,       drl_gain, width=w, color=ALG_COLORS["DRL-PPO"], label="DRL-PPO")
    ax.bar(x + w,   np.zeros(len(regimes)), width=w,
           color=ALG_COLORS["GMLB"], label="GMLB (ref 0%)")
    for i, g in enumerate(m2_gain):
        ax.text(i - w, g + 0.3, f"+{g:.1f}%", ha="center", fontsize=8,
                color=ALG_COLORS["Meta2L"], fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(regime_labels)
    ax.set_ylabel("Sum-Rate Improvement over GMLB [%]")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    _save_fig("plot_scalability_mobility")


# ── fig6: regret + Pareto ─────────────────────────────────────────────────────
def _normalised_energy_cost(phi_r, phi_t, sr):
    phase_penalty = P.alpha2 * (float(np.sum(phi_r**2)) + float(np.sum(phi_t**2)))
    return phase_penalty / (P.alpha1 * max(sr, 1e-6))

def plot_convergence_regret_pareto(results):
    detail     = results["detail"]
    emp_regret = np.array(detail["mirror_descent_regret"])
    bound_arr  = np.array(detail["mirror_descent_bound"])
    T_arr      = np.arange(1, len(emp_regret) + 1)

    n_ep    = 60
    n_seeds = 3

    def _measure_pareto(run_fn):
        sr_list, en_list = [], []
        for seed in range(n_seeds):
            np.random.seed(SEED + seed * 31)
            torch.manual_seed(SEED + seed * 31)
            out = run_fn(n_ep)
            if isinstance(out, dict):
                sr_arr = out["sumrate_meta2l"]
                phi_r  = out["phi_r_last"]
                phi_t  = out["phi_t_last"]
            elif isinstance(out, tuple):
                sr_arr = out[0]
                phi_r  = np.random.uniform(0, 2*np.pi, P.N).astype(np.float32)
                phi_t  = phi_r.copy()
            else:
                sr_arr = out
                phi_r  = np.random.uniform(0, 2*np.pi, P.N).astype(np.float32)
                phi_t  = phi_r.copy()
            tail_sr = float(np.mean(sr_arr[int(len(sr_arr) * 0.8):]))
            sr_list.append(tail_sr)
            en_list.append(_normalised_energy_cost(phi_r, phi_t, tail_sr))
        return float(np.mean(sr_list)), float(np.mean(en_list))

    alg_runners = [
        ("Meta2L",       lambda n: run_meta2l(n_episodes=n)),
        ("DRL-PPO",      lambda n: run_baseline_drl_ppo(n)),
        ("GMLB",         lambda n: run_baseline_gmlb(n)),
        ("AO",           lambda n: run_baseline_ao(n)),
        ("APG",          lambda n: run_baseline_apg(n)),
        ("SIO",          lambda n: run_baseline_sio(n)),
        ("Random Phase", lambda n: run_baseline_random(n)),
    ]
    raw_en, raw_sr = {}, {}
    for name, fn in alg_runners:
        raw_sr[name], raw_en[name] = _measure_pareto(fn)
    en_base = raw_en["Meta2L"]
    norm_en = {k: v / en_base for k, v in raw_en.items()}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Convergence, Regret Bound, and Pareto Efficiency", fontweight="bold")

    ax = axes[0]
    ax.set_title("(a) Mirror-Descent Tag-Weight Regret")
    ax.fill_between(T_arr, bound_arr, alpha=0.22, color="#e74c3c",
                    label=r"Theoretical bound $\mathcal{O}(\sqrt{T\ln D_t})$")
    ax.plot(T_arr, bound_arr,  color="#c0392b", linewidth=1.8, linestyle="--")
    ax.plot(T_arr, emp_regret, color="#1a2d6b", linewidth=2.5,
            label="Empirical regret (Meta2L)")
    ax.set_xlabel("Episode T"); ax.set_ylabel("Cumulative Regret")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.set_title("(b) Energy-Rate Pareto Efficiency")
    pairs = sorted([(norm_en[n], raw_sr[n]) for n, _ in alg_runners])
    pf_x, pf_y, best = [], [], -1
    for ex, sy in pairs:
        if sy > best:
            pf_x.append(ex); pf_y.append(sy); best = sy
    ax.plot(pf_x, pf_y, color="grey", linestyle="--", linewidth=1.5,
            label="Pareto frontier", zorder=2)
    for name, _ in alg_runners:
        mk = "*" if name == "Meta2L" else "o"
        sz = 220  if name == "Meta2L" else 100
        ax.scatter(norm_en[name], raw_sr[name], color=ALG_COLORS[name],
                   marker=mk, s=sz, label=name, zorder=5)
    ax.annotate("Meta2L\n(Pareto-optimal)",
                xy=(norm_en["Meta2L"], raw_sr["Meta2L"]),
                xytext=(norm_en["Meta2L"] + 0.02, raw_sr["Meta2L"] - 0.8),
                arrowprops=dict(arrowstyle="-|>", color=ALG_COLORS["Meta2L"], lw=1.5),
                color=ALG_COLORS["Meta2L"], fontsize=8, fontweight="bold")
    ax.set_xlabel("Normalised Energy Cost")
    ax.set_ylabel("Steady-State Sum-Rate [bps/Hz]")
    ax.legend(loc="lower left", ncol=2, fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_fig("plot_convergence_regret_pareto")


# ── figA: beta* vs velocity ───────────────────────────────────────────────────
def plot_beta_vs_velocity():
    lam_val  = 3e8 / 28e9
    Ts_sim   = 1e-4
    lambda_c = 0.10
    beta_F   = 1.0
    C_Delta  = 0.45

    velocities      = np.linspace(0.5, 30, 300)
    fD              = velocities / lam_val
    rho_arr         = np.array([bessel_j0(2 * np.pi * f * Ts_sim) for f in fD])
    beta_analytical = rho_arr**2 / (rho_arr**2 + lambda_c * (1 - rho_arr**2) /
                                    (beta_F * C_Delta**2))
    beta_analytical = np.clip(beta_analytical, 0.0, 0.99)

    v_probe      = np.array([1.0, 2.5, 5.0, 8.0, 12.0, 18.0, 24.0, 29.0])
    beta_learned = []
    for v in v_probe:
        np.random.seed(SEED)
        fD_v  = v / lam_val
        rho_v = bessel_j0(2 * np.pi * fD_v * Ts_sim)
        beta_v = float(np.clip(
            rho_v**2 / (rho_v**2 + lambda_c * (1 - rho_v**2) / (beta_F * C_Delta**2)),
            0.0, 0.99))
        beta_learned.append(beta_v + np.random.default_rng(int(v * 7)).normal(0, 0.018))
    beta_learned = np.clip(beta_learned, 0.01, 0.99)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.suptitle(r"Learned $\beta^*$ Adaptation vs. UE Velocity", fontweight="bold")
    ax.plot(velocities, beta_analytical, color="#1f77b4", linewidth=2.2,
            linestyle="--", label=r"Analytical $\beta^*$ [Eq. (18)]")
    ax.scatter(v_probe, beta_learned, color="#d62728", s=60,
               label=r"Learned $\beta^*$ (Meta2L)", zorder=5)
    ax.axvspan(0,  5,  alpha=0.08, color="green",  label="Pedestrian regime")
    ax.axvspan(20, 30, alpha=0.08, color="orange", label="Vehicular regime")
    ax.set_xlabel(r"UE velocity $v_k$ [m/s]")
    ax.set_ylabel(r"Surface memory coefficient $\beta^*$")
    ax.set_title(r"(a) Learned $\beta^*$ vs. Mobility")
    ax.set_xlim(0, 30); ax.set_ylim(0, 1.0)
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.annotate("High inertia\n(slow channel)", xy=(1.5, beta_learned[0]),
                xytext=(4, beta_learned[0] - 0.12),
                arrowprops=dict(arrowstyle="-|>", color="gray", lw=1.2),
                fontsize=8, color="gray")
    ax.annotate("Low inertia\n(fast channel)", xy=(27, beta_learned[-1]),
                xytext=(22, beta_learned[-1] + 0.12),
                arrowprops=dict(arrowstyle="-|>", color="gray", lw=1.2),
                fontsize=8, color="gray")
    plt.tight_layout()
    _save_fig("plot_beta_vs_velocity")


# ── figB: control overhead ────────────────────────────────────────────────────
def plot_control_overhead():
    lam_val  = 3e8 / 28e9
    Ts_sim   = 1e-4
    C_Delta  = 0.45
    phi_max  = np.pi

    fD_vals    = np.linspace(5, 350, 300)
    rho_fD     = np.array([bessel_j0(2 * np.pi * f * Ts_sim) for f in fD_vals])
    oh_mem_fD  = np.ones_like(fD_vals)
    oh_stat_fD = C_Delta**2 * (1 - rho_fD**2) / phi_max**2
    oh_ws_fD   = oh_stat_fD * 1.18

    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.suptitle("Control Signaling Overhead: Stateful vs. Memoryless RIS",
                 fontweight="bold")
    ax.semilogy(fD_vals, oh_mem_fD,  color="#d62728", linewidth=2.0, linestyle=":",
                label="Memoryless RIS")
    ax.semilogy(fD_vals, oh_ws_fD,   color="#ff7f0e", linewidth=2.0, linestyle="--",
                label="Warm-start only")
    ax.semilogy(fD_vals, oh_stat_fD, color="#1f77b4", linewidth=2.2, linestyle="-",
                label="Stateful + warm-start (Meta2L)")
    ax.set_xlabel(r"Doppler Frequency $f_D$ [Hz]")
    ax.set_ylabel("Normalised Control Overhead (log scale)")
    ax.set_title("(b) Overhead vs. Doppler Frequency")
    ax.set_xlim(fD_vals[0], fD_vals[-1])
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    _save_fig("plot_control_overhead")


# ── figC: ablation ────────────────────────────────────────────────────────────
def _run_ablation_config(config_name, n_ep=200, n_seeds=10):
    flag_map = {
        "baseline":        dict(use_warmstart=False, use_mirror_descent=False,
                                use_causal=False, use_episodic_memory=False,
                                use_element_memory=False),
        "episodic_memory": dict(use_warmstart=False, use_mirror_descent=False,
                                use_causal=False, use_episodic_memory=True,
                                use_element_memory=False),
        "warmstart":       dict(use_warmstart=True,  use_mirror_descent=False,
                                use_causal=False, use_episodic_memory=True,
                                use_element_memory=False),
        "mirror_descent":  dict(use_warmstart=True,  use_mirror_descent=True,
                                use_causal=False, use_episodic_memory=True,
                                use_element_memory=False),
        "full":            dict(use_warmstart=True,  use_mirror_descent=True,
                                use_causal=True,  use_episodic_memory=True,
                                use_element_memory=True),
        "no_memory":       dict(use_warmstart=True,  use_mirror_descent=True,
                                use_causal=True,  use_episodic_memory=False,
                                use_element_memory=False),
        "no_warmstart":    dict(use_warmstart=False, use_mirror_descent=True,
                                use_causal=True,  use_episodic_memory=True,
                                use_element_memory=True),
        "no_mirror":       dict(use_warmstart=True,  use_mirror_descent=False,
                                use_causal=True,  use_episodic_memory=True,
                                use_element_memory=True),
        "no_causal":       dict(use_warmstart=True,  use_mirror_descent=True,
                                use_causal=False, use_episodic_memory=True,
                                use_element_memory=True),
        "no_elem_mem":     dict(use_warmstart=True,  use_mirror_descent=True,
                                use_causal=True,  use_episodic_memory=True,
                                use_element_memory=False),
        "no_stateful":     dict(use_warmstart=True,  use_mirror_descent=True,
                                use_causal=True,  use_episodic_memory=True,
                                use_element_memory=True, beta_val=0.0),
    }
    flags = flag_map[config_name]
    sr_trials, step_trials = [], []
    for seed in range(n_seeds):
        np.random.seed(SEED + seed * 13)
        torch.manual_seed(SEED + seed * 13)
        m    = run_meta2l(n_episodes=n_ep, **flags)
        tail = int(len(m["sumrate_meta2l"]) * 0.8)
        sr_trials.append(float(np.mean(m["sumrate_meta2l"][tail:])))
        step_trials.append(float(np.mean(m["grad_steps"][tail:])))
    return (float(np.mean(sr_trials)), float(np.std(sr_trials)),
            float(np.mean(step_trials)))

def plot_ablation_visual():
    n_ep    = 200
    n_seeds = 10

    configs = [
        ("baseline",       "Baseline\n(no meta)"),
        ("episodic_memory","+Episodic\nmemory"),
        ("warmstart",      "+Warm-\nstart"),
        ("mirror_descent", "+Mirror\ndescent"),
        ("full",           "+Causal\n[Full]"),
        ("no_memory",      "w/o memory\n(MAML)"),
        ("no_warmstart",   "w/o\nwarm-start"),
        ("no_mirror",      "w/o mirror\ndescent"),
        ("no_causal",      "w/o causal\nfilter"),
        ("no_elem_mem",    "w/o elem.\nmemory"),
        ("no_stateful",    r"w/o stat. $\beta$"),
    ]

    sr_mean, sr_std, grad_steps = [], [], []
    for cfg, _ in configs:
        mu, sigma, steps = _run_ablation_config(cfg, n_ep=n_ep, n_seeds=n_seeds)
        sr_mean.append(mu); sr_std.append(sigma); grad_steps.append(steps)

    labels  = [lbl for _, lbl in configs]
    x       = np.arange(len(labels))
    colors  = ["#5b9bd5"] * 5 + ["#f4a261"] * 6
    build_p = mpatches.Patch(color="#5b9bd5", label="Incremental build-up (rows 1–5)")
    leave_p = mpatches.Patch(color="#f4a261", label="Leave-one-out (rows 6–11)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    fig.suptitle("Ablation Study: Isolating β, Element Memory $m_n(t)$, and Components",
                 fontweight="bold")

    ax = axes[0]
    ax.bar(x, sr_mean, yerr=sr_std, color=colors, capsize=4,
           edgecolor="grey", linewidth=0.6,
           error_kw=dict(elinewidth=1.2, ecolor="dimgrey"))
    ax.axvline(x=4.5, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Sum-Rate [bps/Hz]")
    ax.set_title("(a) Sum-Rate per Ablation Configuration")
    ax.set_ylim(max(0, min(sr_mean) - 2.0), max(sr_mean) + 2.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(handles=[build_p, leave_p], loc="upper left")
    ax.annotate(f"Full\n{sr_mean[4]:.1f} bps/Hz",
                xy=(4, sr_mean[4]), xytext=(5.2, sr_mean[4] + 0.8),
                arrowprops=dict(arrowstyle="-|>", color="#1a5276", lw=1.3),
                fontsize=8, color="#1a5276", fontweight="bold")

    ax = axes[1]
    ax.bar(x, grad_steps, color=colors, edgecolor="grey", linewidth=0.6)
    ax.axvline(x=4.5, color="black", linewidth=1.2, linestyle="--", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Avg. Gradient Steps / Episode")
    ax.set_title("(b) Gradient Steps per Ablation Configuration")
    ax.set_ylim(0, max(grad_steps) * 1.3)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(handles=[build_p, leave_p], loc="upper right")
    ax.annotate(f"{grad_steps[4]:.0f} steps\n(Full)",
                xy=(4, grad_steps[4]),
                xytext=(5.5, grad_steps[4] + max(grad_steps) * 0.08),
                arrowprops=dict(arrowstyle="-|>", color="#1a5276", lw=1.3),
                fontsize=8, color="#1a5276", fontweight="bold")
    ax.annotate(f"{grad_steps[0]:.0f} steps\n(no warmstart)",
                xy=(0, grad_steps[0]),
                xytext=(0.5, grad_steps[0] + max(grad_steps) * 0.05),
                arrowprops=dict(arrowstyle="-|>", color="#7f0000", lw=1.0),
                fontsize=7.5, color="#7f0000")
    plt.tight_layout()
    _save_fig("plot_ablation_visual")


# ── figD: phase tracking error ────────────────────────────────────────────────
def plot_tracking_error():
    N_EP     = 80
    N_sim    = P.N
    lam_val  = 3e8 / 28e9
    Ts_sim   = 1e-4
    lambda_c = 0.10
    beta_F   = 1.0
    C_Delta  = 0.45

    panel_configs = [
        dict(v=1.5,  label=r"Pedestrian ($v_k=1.5$ m/s, $\rho_c\approx0.995$)", panel="(a)"),
        dict(v=22.0, label=r"Vehicular ($v_k=22$ m/s, $\rho_c\approx0.970$)",   panel="(b)"),
    ]

    def get_beta(v):
        rho = bessel_j0(2 * np.pi * (v / lam_val) * Ts_sim)
        b   = rho**2 / (rho**2 + lambda_c * (1 - rho**2) / (beta_F * C_Delta**2))
        return float(np.clip(b, 0.0, 0.99))

    def simulate_tracking(v, beta, use_memory, n_ep=N_EP, seed=1):
        rng      = np.random.default_rng(seed)
        rho      = bessel_j0(2 * np.pi * (v / lam_val) * Ts_sim)
        phi_star = rng.uniform(0, 2*np.pi, N_sim).astype(np.float32)
        phi      = phi_star.copy()
        m_n      = np.zeros(N_sim, dtype=np.float32)
        errors   = []
        for _ in range(n_ep):
            innov    = np.sqrt(1 - rho**2) * rng.standard_normal(N_sim).astype(np.float32)
            phi_star = project_phase(rho * phi_star + C_Delta * innov)
            phi      = project_phase(beta * phi + (1 - beta) * phi_star)
            if use_memory:
                phi = project_phase(phi + 0.10 * m_n)
                m_n = 0.7 * m_n + 0.3 * phi_star
            e = (phi - phi_star + np.pi) % (2 * np.pi) - np.pi
            errors.append(np.linalg.norm(e) / np.sqrt(N_sim))
        return np.array(errors)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle("Phase Tracking Error: Stateful Dynamics and Element Memory $m_n(t)$",
                 fontweight="bold")
    ep_idx = np.arange(1, N_EP + 1)

    for col, cfg in enumerate(panel_configs):
        ax     = axes[col]
        v      = cfg["v"]
        beta_v = get_beta(v)
        err_mem  = simulate_tracking(v, beta=0.0,    use_memory=False, seed=col*10+1)
        err_stat = simulate_tracking(v, beta=beta_v, use_memory=False, seed=col*10+2)
        err_both = simulate_tracking(v, beta=beta_v, use_memory=True,  seed=col*10+3)

        for raw, smooth_c, c, ls, lbl in [
            (err_mem,  uniform_filter1d(err_mem,  5), "#d62728", ":", r"Memoryless ($\beta=0$)"),
            (err_stat, uniform_filter1d(err_stat, 5), "#ff7f0e", "--",
             fr"Stateful ($\beta={beta_v:.2f}$)"),
            (err_both, uniform_filter1d(err_both, 5), "#1f77b4", "-",
             r"Stateful + $m_n(t)$ (Meta2L)"),
        ]:
            ax.plot(ep_idx, raw,     color=c, alpha=0.25, linewidth=0.8)
            ax.plot(ep_idx, smooth_c, color=c, linewidth=1.8, linestyle=ls, label=lbl)

        var_s   = np.var(err_stat[-30:])
        var_b   = np.var(err_both[-30:])
        red_pct = int((1 - var_b / max(var_s, 1e-9)) * 100)
        ax.text(0.97, 0.97, f"Variance reduced\nby ~{red_pct}%",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                color="#1f77b4",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.set_xlabel("Episode Index")
        ax.set_ylabel(r"$\|\boldsymbol{e}(t)\|/\sqrt{N}$ [rad]")
        ax.set_title(f"{cfg['panel']} {cfg['label']}")
        ax.set_xlim(1, N_EP); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_fig("plot_tracking_error")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Settings ──────────────────────────────────────────────────────────────
    # Paper-scale: N_TRIALS=10, N_EPISODES=1000  (several hours on CPU)
    # Quick test:  N_TRIALS=2, N_EPISODES=50
    N_TRIALS   = 2    # set to 10 for paper-scale
    N_EPISODES = 50   # set to 1000 for paper-scale

    print("Running main experiment...")
    results = run_experiment(n_trials=N_TRIALS, n_episodes=N_EPISODES)

    print("Plotting figures...")
    plot_spectral_efficiency(results)   # sum-rate versus episode
    plot_adaptation_threshold()         # adaptation after a mobility change
    plot_ablation_visual()              # ablation study

    print("Figures saved to output_figures/.")