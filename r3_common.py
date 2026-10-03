"""
r3_common.py

Shared utilities for META2L_R3_BASELINES.py and the Table VIII experiments.

The module reuses the simulator in META2L_SENSITIVITY4.py without modifying it.
It adds:

- SwitchEnv: starts with a pedestrian velocity profile and switches to a
  vehicular profile after N_PRE intervals.
- adaptation_intervals: measures how many intervals are needed for the
  sum-rate to reach 95% of the post-switch steady-state value.
- run_simple: evaluates the cold-start, previous-solution, and nearest-neighbor
  baselines used in the temporal-reuse experiments.
"""

import importlib.util
import os

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch

torch.set_num_threads(1)


# Load the main simulator from the same directory.
HERE = os.path.dirname(os.path.abspath(__file__))

simulator_path = os.path.join(
    HERE,
    "META2L_SENSITIVITY4.py",
)

spec = importlib.util.spec_from_file_location(
    "meta2l",
    simulator_path,
)

M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

P = M.P


# Experiment settings.
SEED = 42
N_EP = 200
TAIL = 0.8

N_PRE = 100
N_POST = 100

MAX_ITER = 100


# Reuse the environment implementation from the main simulator.
BaseEnv = M.WirelessEnvironment


class SwitchEnv(BaseEnv):
    """
    Environment that starts with pedestrian motion and switches to
    vehicular motion after N_PRE intervals.
    """

    def __init__(self, N=P.N, K=None, v_profile="mixed"):
        super().__init__(
            N=N,
            K=K,
            v_profile="pedestrian",
        )

        self._count = 0

    def step(self):
        self._count += 1

        if self._count == N_PRE + 1:
            self.switch_profile("vehicular")

        super().step()


def adaptation_intervals(sr_post, frac=0.95):
    """
    Return the first interval at which the sum-rate reaches a given
    fraction of the post-switch steady-state value.
    """

    tail_start = int(0.75 * len(sr_post))
    steady_state = float(np.mean(sr_post[tail_start:]))

    target = frac * steady_state

    for i, sum_rate in enumerate(sr_post):
        if sum_rate >= target:
            return i + 1

    # The target was not reached within the available intervals.
    return len(sr_post)


def seed_all(seed):
    """Set the NumPy and PyTorch random seeds."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def rand_phases(N):
    """Generate independent random phase vectors for the two surfaces."""
    phi_r = np.random.uniform(0, 2 * np.pi, N)
    phi_t = np.random.uniform(0, 2 * np.pi, N)

    return phi_r, phi_t


def run_simple(method, n_ep, env, K=None):
    """
    Run one of the simple temporal-reuse baselines.

    Methods:
        cold:
            Start each interval from a new random phase configuration.

        prev:
            Start from the optimized solution obtained in the previous
            interval.

        nn:
            Retrieve the closest stored episode using uniform tag weights.
            The retrieved phase configuration is used as the initialization.

    The nearest-neighbor baseline does not use Doppler interpolation,
    a meta-policy, mirror descent, or a causal filter.
    """

    N = env.N
    Keff = env.K

    memory = M.EpisodicMemory(
        max_size=P.mem_size
    )

    # Uniform weights are used for the tag-distance calculation.
    tag_weights = np.ones(P.D_t) / P.D_t

    previous = rand_phases(N)

    sum_rate_history = []
    step_history = []

    for _ in range(n_ep):
        env.step()

        H, hr, ht = env.get_channels()

        if method == "cold":
            # No temporal reuse.
            init = rand_phases(N)

        elif method == "prev":
            # Reuse the solution from the previous interval.
            init = previous

        elif method == "nn":
            # Build a tag from a random-phase operating point and retrieve
            # the closest stored episode.
            random_phases = rand_phases(N)

            sr_random = env.compute_sumrate(
                *random_phases,
                H,
                hr,
                ht,
            )

            tag = M.make_tag_vector(
                env.v_k,
                sr_random,
                H,
                K=Keff,
            )

            retrieved = memory.retrieve(
                tag,
                tag_weights,
                top_k=1,
            )

            if retrieved:
                init = (
                    retrieved[0].phi_r.copy(),
                    retrieved[0].phi_t.copy(),
                )
            else:
                init = random_phases

        else:
            raise ValueError(f"Unknown method: {method}")

        # Refine the selected initialization using gradient ascent.
        phi_r, phi_t, n_iter, sum_rate = M.optimise_phases(
            env,
            H,
            hr,
            ht,
            init[0],
            init[1],
            eta=0.15,
            max_iter=MAX_ITER,
            tol=M.TOL,
        )

        # Store the optimized solution for later nearest-neighbor retrieval.
        if method == "nn":
            episode = M.Episode(
                tag=tag,
                phi_r=phi_r,
                phi_t=phi_t,
                sr=sum_rate,
                v_mean=float(np.mean(env.v_k)),
                rho_c=0.0,
            )

            memory.add(episode)

        previous = (phi_r, phi_t)

        sum_rate_history.append(sum_rate)
        step_history.append(n_iter)

    return sum_rate_history, step_history