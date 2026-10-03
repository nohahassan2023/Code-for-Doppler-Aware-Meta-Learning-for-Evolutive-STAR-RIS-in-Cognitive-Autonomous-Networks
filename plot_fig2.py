import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import j0


# Keep the plotting style consistent with the other figures.
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
})


# Paths
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "output_figures")
os.makedirs(OUT, exist_ok=True)


# Load the beta-learning results.
data = np.loadtxt(
    os.path.join(RESULTS, "learned_beta.csv"),
    delimiter=",",
    skiprows=1,
)

v, rho, beta_opt, std_opt, beta_learned, std_learned = data.T


# Closed-form expression for beta* from Eq. (49).
def beta_star(q):
    return ((q + 2.0) - np.sqrt(q * (q + 4.0))) / 2.0


# Fit kappa using the exhaustive-search beta values.
def fitting_error(log_kappa):
    kappa = np.exp(log_kappa)
    q = kappa * (1.0 - rho**2)
    return np.sum((beta_star(q) - beta_opt) ** 2)


fit = minimize_scalar(
    fitting_error,
    bounds=(-5, 10),
    method="bounded",
)

kappa = float(np.exp(fit.x))


# Wavelength at the carrier frequency.
c = 3e8
fc = 28e9
wavelength = c / fc


# Evaluate the analytical expression over the full velocity range.
velocity = np.linspace(0.5, 30.0, 300)

rho_curve = j0(
    2.0 * np.pi * velocity / wavelength * 1e-4
)

beta_curve = beta_star(
    kappa * (1.0 - rho_curve**2)
)


# Create the figure.
fig, ax = plt.subplots(figsize=(6.4, 4.6))

ax.axvspan(
    0,
    5,
    color="#eaf4ea",
    zorder=0,
    label="Pedestrian regime",
)

ax.axvspan(
    20,
    30,
    color="#fdf3e6",
    zorder=0,
    label="Vehicular regime",
)

ax.plot(
    velocity,
    beta_curve,
    "--",
    color="#1f77b4",
    lw=2.2,
    zorder=2,
    label=r"Analytical $\beta^*$ [Eq. (49)]",
)

ax.plot(
    v,
    beta_opt,
    "s",
    mfc="none",
    mec="black",
    ms=7,
    mew=1.2,
    zorder=4,
    label=r"Optimal $\beta^*$ (exhaustive search)",
)

ax.errorbar(
    v,
    beta_learned,
    yerr=std_learned,
    fmt="o",
    color="#d62728",
    ecolor="#d62728",
    ms=7,
    capsize=3,
    zorder=5,
    label=r"Learned $\beta$ (Meta2L)",
)


# Highlight the change in surface memory with velocity.
ax.annotate(
    "High inertia\n(slow channel)",
    xy=(2.5, beta_learned[1]),
    xytext=(6.5, 0.84),
    fontsize=8.5,
    color="grey",
    arrowprops=dict(
        arrowstyle="->",
        color="grey",
    ),
)

ax.annotate(
    "Low inertia\n(fast channel)",
    xy=(24, beta_learned[8]),
    xytext=(20.8, 0.34),
    fontsize=8.5,
    color="grey",
    arrowprops=dict(
        arrowstyle="->",
        color="grey",
    ),
)


ax.set_title(
    r"Learned vs. analytical $\beta^*$ across the full velocity range",
    fontsize=10,
)

ax.set_xlabel(r"UE velocity $v_k$ [m/s]")
ax.set_ylabel(r"Surface memory coefficient $\beta^*$")

ax.set_xlim(0, 30)
ax.set_ylim(0, 1.0)

ax.grid(
    True,
    color="#dddddd",
    lw=0.6,
    zorder=0,
)

ax.legend(
    fontsize=8.5,
    loc="upper right",
    framealpha=1.0,
)

plt.tight_layout()


# Save the figure in the formats used in the paper.
for ext in ("eps", "pdf"):
    fig.savefig(
        os.path.join(OUT, f"fig_learned_beta.{ext}"),
        format=ext,
    )

fig.savefig(
    os.path.join(OUT, "fig_learned_beta.png"),
    dpi=220,
)


# Report the fit and the difference between the learned and exhaustive-search
# values.
beta_pred = beta_star(
    kappa * (1.0 - rho**2)
)

rmse_opt = np.sqrt(
    np.mean((beta_pred - beta_opt) ** 2)
)

rmse_learned = np.sqrt(
    np.mean((beta_pred - beta_learned) ** 2)
)

mean_abs_error = np.mean(
    np.abs(beta_learned - beta_opt)
)

print(
    f"kappa={kappa:.2f}  "
    f"RMSE(opt)={rmse_opt:.3f}  "
    f"RMSE(learned)={rmse_learned:.3f}  "
    f"mean|learned-opt|={mean_abs_error:.3f}"
)