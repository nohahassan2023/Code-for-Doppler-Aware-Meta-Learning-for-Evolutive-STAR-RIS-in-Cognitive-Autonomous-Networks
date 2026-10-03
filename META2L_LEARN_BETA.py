"""
META2L_LEARN_BETA.py

Fig. 2: Learned versus optimal surface memory coefficient.

For each UE velocity and random seed, the script:

1. Generates a sequence of coherence intervals. At each interval, the
   controller target u(t) = phi*(t) is obtained using projected gradient
   ascent, initialized from the solution at the previous interval.

2. Applies the stateful surface model from (8):
       phi(t) = phi*(t) + beta * wrap(phi(t-1) - phi*(t))

3. Evaluates the per-interval cost from the stateful objective in (12):
       J_t(beta) =
           omega1 [R(phi*(t)) - R(phi(t))]
           + lambda_c ||wrap(phi(t) - phi(t-1))||_2^2

4. Finds the optimal beta by an exhaustive search over a fine beta grid,
   replaying the same trajectory. This is an offline oracle used only for
   comparison.

5. Learns beta online in a single pass through the same trajectory. The
   gradient is computed from the recurrent sensitivity

       s(t) = d phi(t) / d beta
            = wrap(phi(t-1) - phi*(t)) + beta s(t-1)

   and Adam updates the logit of beta, with beta = sigmoid(z).

The online learner does not use future channel realizations, the beta grid,
or the closed-form beta expression.
"""

import importlib.util
import os

import matplotlib
matplotlib.use("Agg")

import numpy as np
from scipy.special import j0


# Load the main simulator without modifying it.
HERE = os.path.dirname(os.path.abspath(__file__))
SIMULATOR = os.path.join(HERE, "META2L_SENSITIVITY4.py")

spec = importlib.util.spec_from_file_location(
    "meta2l",
    SIMULATOR,
)

M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

P = M.P


# Parameters from Table II and the experiment setup.
OMEGA1 = 10.0
LAMBDA_C = 0.10

T_INTERVALS = 600
BURN_IN = 20

LR = 0.03

BETA_GRID = np.linspace(0.0, 0.99, 100)

V_PROBE = [
    1.0,
    2.5,
    5.0,
    8.0,
    12.0,
    15.0,
    18.0,
    21.0,
    24.0,
    27.0,
    29.0,
]

N_SEEDS = 3


def wrap(x):
    """Wrap phase differences to the interval [-pi, pi)."""
    return (x + np.pi) % (2 * np.pi) - np.pi


class FixedVelocityEnv(M.WirelessEnvironment):
    """Environment in which all UEs move at the same fixed velocity."""

    def __init__(self, v, N=P.N, K=None):
        self.v_fixed = v

        super().__init__(
            N=N,
            K=K,
            v_profile="mixed",
        )

    def _sample_velocity(self):
        return self.v_fixed


def make_trajectory(v, seed, T=T_INTERVALS):
    """
    Generate the channel and optimized-phase trajectory for one velocity
    and one random seed.
    """

    np.random.seed(seed)

    env = FixedVelocityEnv(v)
    N = env.N

    # Obtain an initial channel realization and phase solution.
    env.step()

    H, hr, ht = env.get_channels()

    phi_r, phi_t, _, _ = M.optimise_phases(
        env,
        H,
        hr,
        ht,
        np.random.uniform(0, 2 * np.pi, N),
        np.random.uniform(0, 2 * np.pi, N),
    )

    trajectory = []

    for _ in range(T):
        env.step()

        H, hr, ht = env.get_channels()

        # Warm-start the phase optimization from the previous solution.
        phi_r, phi_t, _, sum_rate = M.optimise_phases(
            env,
            H,
            hr,
            ht,
            phi_r,
            phi_t,
        )

        trajectory.append(
            (
                H,
                hr,
                ht,
                phi_r.copy(),
                phi_t.copy(),
                sum_rate,
            )
        )

    return env, trajectory


def interval_cost(env, step, phi_prev):
    """
    Return the stateful objective for one interval as a function of the
    current surface phase vector.
    """

    H, hr, ht, phi_r_star, phi_t_star, rate_star = step
    N = env.N

    def cost(phi):
        rate = env.compute_sumrate(
            phi[:N],
            phi[N:],
            H,
            hr,
            ht,
        )

        phase_change = wrap(phi - phi_prev)

        rate_loss = OMEGA1 * (rate_star - rate)
        control_cost = LAMBDA_C * float(phase_change @ phase_change)

        return rate_loss + control_cost

    return cost


def replay_cost(env, trajectory, beta):
    """
    Evaluate a fixed beta over a previously generated trajectory.

    This is the offline exhaustive-search evaluation used to obtain the
    reference beta*. The channel trajectory itself is not regenerated for
    different beta values.
    """

    N = env.N

    phi = np.concatenate(
        [
            trajectory[0][3],
            trajectory[0][4],
        ]
    )

    costs = []

    for t, step in enumerate(trajectory[1:], start=1):
        target = np.concatenate(
            [
                step[3],
                step[4],
            ]
        )

        phase_error = wrap(phi - target)

        new_phi = M.project_phase(
            target + beta * phase_error
        )

        if t >= BURN_IN:
            cost = interval_cost(
                env,
                step,
                phi,
            )(new_phi)

            costs.append(cost)

        phi = new_phi

    return float(np.mean(costs))


def learn_online(env, trajectory):
    """
    Learn beta in one forward pass using the recurrent gradient and Adam.

    The optimization variable is z, with beta = sigmoid(z).
    """

    N = env.N

    # Adam state for the scalar logit z.
    z = 0.0
    m1 = 0.0
    m2 = 0.0
    iteration = 0

    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    # Start from the optimized phase configuration in the first interval.
    phi = np.concatenate(
        [
            trajectory[0][3],
            trajectory[0][4],
        ]
    )

    # Recurrent sensitivity d phi(t-1) / d beta.
    sensitivity = np.zeros(2 * N)

    beta_history = []

    for t, (H, hr, ht, phi_r_star, phi_t_star, _) in enumerate(
        trajectory[1:],
        start=1,
    ):
        beta = 1.0 / (1.0 + np.exp(-z))

        target = np.concatenate(
            [
                phi_r_star,
                phi_t_star,
            ]
        )

        phase_error = wrap(phi - target)

        new_phi = M.project_phase(
            target + beta * phase_error
        )

        # Recurrent derivative of phi(t) with respect to beta.
        new_sensitivity = (
            phase_error
            + beta * sensitivity
        )

        if t >= BURN_IN:
            # Gradient of the sum-rate with respect to the surface phases.
            grad_r, grad_t = env.sinr_gradient(
                new_phi[:N],
                new_phi[N:],
                H,
                hr,
                ht,
            )

            rate_gradient = np.concatenate(
                [
                    grad_r,
                    grad_t,
                ]
            )

            d_rate = float(
                rate_gradient @ new_sensitivity
            )

            phase_change = wrap(new_phi - phi)

            d_phase_cost = (
                2.0
                * float(
                    phase_change
                    @ (new_sensitivity - sensitivity)
                )
            )

            d_cost = (
                -OMEGA1 * d_rate
                + LAMBDA_C * d_phase_cost
            )

            # Chain rule through beta = sigmoid(z).
            gradient = (
                d_cost
                * beta
                * (1.0 - beta)
            )

            # Adam update.
            iteration += 1

            m1 = (
                beta1 * m1
                + (1.0 - beta1) * gradient
            )

            m2 = (
                beta2 * m2
                + (1.0 - beta2) * gradient**2
            )

            m1_hat = m1 / (1.0 - beta1**iteration)
            m2_hat = m2 / (1.0 - beta2**iteration)

            z -= (
                LR
                * m1_hat
                / (np.sqrt(m2_hat) + eps)
            )

            # Keep the logit in a reasonable numerical range.
            z = float(np.clip(z, -5.0, 5.0))

        phi = new_phi
        sensitivity = new_sensitivity

        beta_history.append(
            1.0 / (1.0 + np.exp(-z))
        )

    # Use the last quarter of the trajectory to summarize the learned beta.
    tail_length = len(beta_history) // 4

    learned_beta = float(
        np.mean(beta_history[-tail_length:])
    )

    return learned_beta, beta_history


def rho_of(v):
    """Compute the channel correlation coefficient for a given velocity."""
    return float(
        j0(
            2 * np.pi * (v / P.lam) * P.Ts
        )
    )


if __name__ == "__main__":

    rows = []

    for v in V_PROBE:
        optimal_betas = []
        learned_betas = []

        for seed_index in range(N_SEEDS):
            seed = 42 + 11 * seed_index

            env, trajectory = make_trajectory(
                v,
                seed=seed,
            )

            # Exhaustive-search reference.
            costs = [
                replay_cost(
                    env,
                    trajectory,
                    beta,
                )
                for beta in BETA_GRID
            ]

            optimal_beta = float(
                BETA_GRID[int(np.argmin(costs))]
            )

            # Online learned beta.
            learned_beta, _ = learn_online(
                env,
                trajectory,
            )

            optimal_betas.append(optimal_beta)
            learned_betas.append(learned_beta)

        rho = rho_of(v)

        row = (
            v,
            rho,
            np.mean(optimal_betas),
            np.std(optimal_betas),
            np.mean(learned_betas),
            np.std(learned_betas),
        )

        rows.append(row)

        print(
            f"v={v:5.1f}  "
            f"rho_c={rho:.3f}  "
            f"optimal beta*={row[2]:.3f} +/- {row[3]:.3f}   "
            f"learned={row[4]:.3f} +/- {row[5]:.3f}",
            flush=True,
        )

    rows = np.array(rows)


    # Save the results used by the plotting script.
    results_dir = os.path.join(
        HERE,
        "results",
    )

    os.makedirs(
        results_dir,
        exist_ok=True,
    )

    output_file = os.path.join(
        results_dir,
        "learned_beta.csv",
    )

    np.savetxt(
        output_file,
        rows,
        delimiter=",",
        header=(
            "v,rho_c,beta_opt_mean,beta_opt_std,"
            "beta_learned_mean,beta_learned_std"
        ),
        comments="",
    )

    mean_error = np.mean(
        np.abs(rows[:, 4] - rows[:, 2])
    )

    print(
        "mean |learned - optimal| =",
        float(mean_error),
    )