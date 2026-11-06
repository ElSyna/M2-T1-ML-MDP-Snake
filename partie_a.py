"""Generate the figures requested for Part A of the Snake MDP assignment."""

import random

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from simulation import simulate


GAMMAS = (0, 0.1, 0.5, 0.9, 1)
SEED = 42


def plot_mdp_schema(path="figures/ex2_mdp_schema.png"):
    """Draw a four-state cycle; the fruit remains at (0, 0) in each state."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    nodes = {
        "s0": (0, 1.4, "s0\n tête (2,2), direction →"),
        "s1": (3, 1.4, "s1\n tête (3,2), direction ↓"),
        "s2": (3, -1.4, "s2\n tête (3,1), direction ←"),
        "s3": (0, -1.4, "s3\n tête (2,1), direction ↑"),
    }

    for x, y, label in nodes.values():
        ax.text(
            x,
            y,
            f"{label}\nfruit (0,0)",
            ha="center",
            va="center",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.55", "fc": "#e9f2fb", "ec": "#3973ac"},
        )

    arrows = [
        ("s0", "s1", "bas, r=0", (0, 0.18)),
        ("s1", "s2", "gauche, r=0", (0.25, 0)),
        ("s2", "s3", "haut, r=0", (0, -0.18)),
        ("s3", "s0", "droite, r=0", (-0.25, 0)),
    ]
    for source, target, label, offset in arrows:
        x1, y1, _ = nodes[source]
        x2, y2, _ = nodes[target]
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=15,
                linewidth=1.5,
                color="#555555",
                shrinkA=58,
                shrinkB=58,
            )
        )
        ax.text(
            (x1 + x2) / 2 + offset[0],
            (y1 + y2) / 2 + offset[1],
            label,
            ha="center",
            va="center",
            fontsize=9,
            color="#333333",
            bbox={"fc": "white", "ec": "none", "alpha": 0.9, "pad": 2},
        )

    ax.set_title("Exercice 2.7 — Cycle d'états du MDP (fruit fixé en (0,0))")
    ax.set_xlim(-1.5, 4.5)
    ax.set_ylim(-2.2, 2.2)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_gamma_comparison(
    runs=100,
    steps=1_000,
    size=5,
    eps=0.1,
    seed=SEED,
    path="figures/ex3_12_comparaison_gamma.png",
):
    """Compare mean reward for the requested discount factors."""
    fig, ax = plt.subplots(figsize=(9, 5))
    summaries = {}
    window = 50

    for gamma in GAMMAS:
        random.seed(seed)
        rewards = simulate(
            runs=runs,
            steps=steps,
            size=size,
            length=1,
            eps=eps,
            gamma=gamma,
        )
        mean_reward = rewards.mean(axis=0)
        smoothed = np.convolve(
            mean_reward, np.ones(window) / window, mode="valid"
        )
        ax.plot(
            np.arange(window, steps + 1),
            smoothed,
            label=f"γ = {gamma}",
        )
        summaries[gamma] = (
            float(mean_reward[:100].mean()),
            float(mean_reward[-100:].mean()),
        )

    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_title(
        f"Exercice 3.12 — Effet de γ "
        f"({runs} parties, grille {size}×{size}, ε = {eps})"
    )
    ax.set_xlabel("Étape")
    ax.set_ylabel("Récompense moyenne (moyenne glissante sur 50 étapes)")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return summaries


if __name__ == "__main__":
    plot_mdp_schema()
    results = plot_gamma_comparison()
    print("γ | moyenne des 100 premières | moyenne des 100 dernières")
    for gamma, (early, late) in results.items():
        print(f"{gamma:g} | {early:+.3f} | {late:+.3f}")
