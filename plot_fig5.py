# Fig. 5: Analytical critical-path latency for the Zynq UltraScale+ MPSoC
# at 200 MHz, with N = 64 RIS elements.
#
# Cycle counts:
#   - Tag retrieval: T_E * D_t / 8 = 3000 cycles
#   - Warm-start initialization: N cycles
#   - Gradient ascent: T_grad * N cycles
#   - Stateful update and memory write: 2N cycles
#
# At 200 MHz, one cycle corresponds to 5 ns = 0.005 microseconds.

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
})


# System and timing parameters.
N = 64
T_E = 1000
D_t = 24
cycle_time = 5e-3  # microseconds

T_warm = 24
T_cold = 96


def latency_cycles(T_grad):
    """Return the cycle count for each stage of the critical path."""
    return np.array([
        T_E * D_t / 8,
        N,
        T_grad * N,
        2 * N,
    ])


# Compute the latency of each stage for the two initialization cases.
warm_latency = latency_cycles(T_warm) * cycle_time
cold_latency = latency_cycles(T_cold) * cycle_time

warm_latency = np.append(warm_latency, warm_latency.sum())
cold_latency = np.append(cold_latency, cold_latency.sum())


stage_names = [
    "Tag retrieval\n(×8 DSP48E2 slices)",
    "Warm-start\ninitialization",
    "Gradient ascent\n($T_{\\mathrm{grad}}$ steps)",
    "Stateful update\n+ memory write",
    "Total critical\npath",
]


# Plot the analytical latency estimates.
y = np.arange(len(stage_names))
bar_height = 0.30

fig, ax = plt.subplots(figsize=(9, 5.5))

ax.barh(
    y + bar_height / 2,
    warm_latency,
    height=bar_height,
    color="#1a3a5c",
    label=f"Warm-start ($T_{{\\mathrm{{grad}}}}$={T_warm})",
    zorder=3,
)

ax.barh(
    y - bar_height / 2,
    cold_latency,
    height=bar_height,
    color="#e6a4a0",
    label=f"Cold-start ($T_{{\\mathrm{{grad}}}}$={T_cold})",
    zorder=3,
)


# Reference coherence window.
ax.axvline(
    100,
    color="#2d6a4f",
    lw=1.8,
    ls="--",
    zorder=5,
    label="Coherence window (100 μs)",
)


# Add the latency values next to the bars.
for i in range(len(stage_names)):
    ax.text(
        warm_latency[i] + 1,
        i + bar_height / 2,
        f"{warm_latency[i]:.1f} μs",
        va="center",
        fontsize=8,
        color="#1a3a5c",
        fontweight="bold",
    )

    ax.text(
        cold_latency[i] + 1,
        i - bar_height / 2,
        f"{cold_latency[i]:.1f} μs",
        va="center",
        fontsize=8,
        color="#b03a2e",
        fontweight="bold",
    )


# Show the available margin relative to the 100 μs coherence window.
warm_margin = 100 / warm_latency[-1]
cold_margin = 100 / cold_latency[-1]

ax.text(
    warm_latency[-1] + 12,
    4 + bar_height / 2 + 0.12,
    f"{warm_margin:.1f}× margin",
    fontsize=8.5,
    color="#1a3a5c",
    fontweight="bold",
)

ax.text(
    cold_latency[-1] + 12,
    4 - bar_height / 2 - 0.12,
    f"{cold_margin:.1f}× margin",
    fontsize=8.5,
    color="#b03a2e",
    fontweight="bold",
)


ax.set_yticks(y)
ax.set_yticklabels(stage_names)

ax.set_xlabel("Latency [μs] (analytical cycle-count estimate)")

ax.set_xlim(0, 118)

ax.grid(
    True,
    axis="x",
    color="#dddddd",
)

ax.set_title(
    f"Critical-path latency estimate, Zynq UltraScale+ MPSoC @ 200 MHz, N={N}\n"
    f"Warm-start ($T_{{\\mathrm{{grad}}}}$={T_warm}) "
    f"vs. cold-start ($T_{{\\mathrm{{grad}}}}$={T_cold})"
)

ax.legend(
    loc="lower right",
    framealpha=1,
)

plt.tight_layout()


# Save the figure in the formats used in the paper.
output_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "output_figures",
)

os.makedirs(output_dir, exist_ok=True)

for ext in ("eps", "pdf"):
    fig.savefig(
        os.path.join(output_dir, f"fig_fpga_latency.{ext}")
    )

fig.savefig(
    os.path.join(output_dir, "fig_fpga_latency.png"),
    dpi=200,
)


# Print the values used in the figure.
print(
    "warm",
    warm_latency.round(2),
    "cold",
    cold_latency.round(2),
    "margins",
    round(warm_margin, 2),
    round(cold_margin, 2),
)


# Also report the total latency for several RIS sizes.
for n_elements in (16, 64, 128, 256):
    total_cycles = (
        3000
        + n_elements
        + T_warm * n_elements
        + 2 * n_elements
    )

    total_latency_ms = total_cycles * cycle_time / 1000

    print(
        n_elements,
        round(total_latency_ms, 4),
        "ms",
    )