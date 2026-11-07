"""Partie C du TP Snake : nombre de relances (3.11), recherche du meilleur
facteur d'actualisation (3.13), passage à 10 000 étapes (3.14) et serpents plus
longs (4.7).

Usage : python partie_c.py [relances] [gamma] [longueurs]   (sans argument : tout)
"""

import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from players import GreedyPlayer
from simulation import play, simulate
from snake import ShortSnake

SEED = 42
SIZE = 5
STEPS = 1_000
EPS = 0.1
GAMMA = 0.5
WINDOW = 50

RUN_COUNTS = (1, 5, 10, 20, 50, 100, 200, 500)
BATCHES = 10  # lots indépendants par nombre de relances


def moving_average(x, w=WINDOW):
    return np.convolve(x, np.ones(w) / w, mode="valid")


def mean_curve(runs, seed):
    """Courbe moyenne sur `runs` parties indépendantes (graine donnée)."""
    random.seed(seed)
    rewards = simulate(runs, STEPS, size=SIZE, eps=EPS, gamma=GAMMA)
    return rewards.mean(axis=0)


def study_runs(run_counts=RUN_COUNTS, batches=BATCHES, seed=SEED):
    """Pour chaque nombre de relances n, tire `batches` lots indépendants de
    n parties et mesure à quel point la courbe moyenne lissée change d'un lot
    à l'autre.

    bruit(n) = écart-type, d'un lot à l'autre, de la courbe lissée à chaque
    étape, moyenné quadratiquement sur les étapes.
    """
    results = {}
    for n in run_counts:
        start = time.perf_counter()
        curves = np.array([
            moving_average(mean_curve(n, seed + 1000 * n + b))
            for b in range(batches)
        ])
        elapsed = (time.perf_counter() - start) / batches
        noise = float(np.sqrt(curves.var(axis=0, ddof=1).mean()))
        results[n] = {"noise": noise, "time": elapsed, "curves": curves}
    return results


def learning_signal(runs=500, seed=SEED):
    """Amélioration moyenne (100 dernières - 100 premières étapes) à mesurer."""
    m = mean_curve(runs, seed)
    return float(m[-100:].mean() - m[:100].mean())


def plot_study(results, signal, chosen, path="figures/ex3_11_nombre_de_relances.png"):
    counts = list(results)
    noise = [results[n]["noise"] for n in counts]
    steps = np.arange(WINDOW, STEPS + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    ax1.loglog(counts, noise, "o-", color="#2a78d6", label="bruit mesuré")
    ref = noise[0] * np.sqrt(counts[0] / np.array(counts))
    ax1.loglog(counts, ref, "--", color="#888888", label="décroissance en 1/√n")
    ax1.axhline(signal / 4, color="#d1495b", linewidth=1,
                label=f"seuil = signal/4 = {signal / 4:.2f}")
    ax1.axvline(chosen, color="#1baf7a", linewidth=1, label=f"choix : {chosen}")
    ax1.set_xlabel("Nombre de relances n")
    ax1.set_ylabel("Bruit de la courbe moyenne lissée\n(écart-type d'un lot à l'autre)")
    ax1.set_title("Bruit en fonction du nombre de relances")
    ax1.legend(frameon=False)
    ax1.grid(True, color="#e5e5e5")

    for n, color in zip((1, 10, chosen), ("#b0b0b0", "#eb6834", "#2a78d6")):
        if n not in results:
            continue
        for b, curve in enumerate(results[n]["curves"][:3]):
            ax2.plot(steps, curve, color=color, linewidth=1.1,
                     label=f"n = {n}" if b == 0 else None)
    ax2.axhline(0, color="#555555", linewidth=0.8)
    ax2.set_xlabel("Étape")
    ax2.set_ylabel(f"Récompense moyenne (moyenne glissante {WINDOW})")
    ax2.set_title("3 lots indépendants par nombre de relances")
    ax2.legend(frameon=False, loc="lower right")
    ax2.grid(True, color="#e5e5e5")

    fig.suptitle(f"Ex 3.11 — Choix du nombre de relances "
                 f"(grille {SIZE}×{SIZE}, γ = {GAMMA}, ε = {EPS}, {STEPS} étapes)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def choose_runs(results, signal, ratio=4):
    """Plus petit n dont le bruit est inférieur à signal / ratio."""
    for n, r in results.items():
        if r["noise"] < signal / ratio:
            return n
    return max(results)


# --- 3.13 / 3.14 : recherche du meilleur facteur d'actualisation -------------

GAMMA_GRID = (0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1)


def gamma_search(steps, runs, last, gammas=GAMMA_GRID, seed=SEED):
    """Évalue chaque γ sur `runs` parties de `steps` étapes.

    La partie r utilise la même graine pour tous les γ : les comparaisons sont
    appariées (mêmes fruits, mêmes tirages ε tant que les décisions coïncident),
    ce qui réduit fortement le bruit de la différence entre deux γ.

    Pour chaque γ on garde, par partie, la récompense moyenne sur toute la
    durée (clé "all") et sur les `last` dernières étapes (clé "last"), plus la
    courbe moyenne sur les parties (clé "curve").
    """
    jobs = [(g, steps, runs, last, seed) for g in gammas]
    with ProcessPoolExecutor() as pool:  # un processus par γ, résultats identiques
        return dict(zip(gammas, pool.map(_evaluate_gamma, jobs)))


def _evaluate_gamma(job):
    gamma, steps, runs, last, seed = job
    mean_all = np.zeros(runs)
    mean_last = np.zeros(runs)
    curve = np.zeros(steps)
    for r in range(runs):
        random.seed(seed + r)
        game = ShortSnake(SIZE)
        player = GreedyPlayer(game, eps=EPS, gamma=gamma)
        rewards = play(game, player, steps)
        mean_all[r] = rewards.mean()
        mean_last[r] = rewards[-last:].mean()
        curve += rewards
    return {"all": mean_all, "last": mean_last, "curve": curve / runs}


def summarize(results, key):
    """Moyenne ± erreur standard, et écart apparié avec le meilleur γ.

    Un γ est « indistinguable du meilleur » si son écart apparié avec lui est
    inférieur à 2 erreurs standard de cet écart.
    """
    means = {g: float(r[key].mean()) for g, r in results.items()}
    best = max(means, key=means.get)
    rows = {}
    for g, r in results.items():
        n = len(r[key])
        se = float(r[key].std(ddof=1) / np.sqrt(n))
        diff = r[key] - results[best][key]
        d_mean = float(diff.mean())
        d_se = float(diff.std(ddof=1) / np.sqrt(n))
        rows[g] = {"mean": means[g], "se": se, "diff": d_mean, "diff_se": d_se,
                   "tie": d_mean >= -2 * d_se}
    return best, rows


def print_summary(results, title, last):
    print(title)
    for key, label in (("all", "toute la durée"), ("last", f"{last} dernières étapes")):
        best, rows = summarize(results, key)
        print(f"  métrique : récompense moyenne sur {label} ; meilleur γ = {best}")
        print(f"  {'γ':>5} | {'moyenne':>8} ± {'ES':>5} | {'écart/meilleur':>14} ± {'ES':>5} | indistinguable")
        for g, row in rows.items():
            print(f"  {g:>5g} | {row['mean']:+8.3f} ± {row['se']:.3f} | "
                  f"{row['diff']:+14.3f} ± {row['diff_se']:.3f} | {'oui' if row['tie'] else 'non'}")


def plot_gamma_search(results, steps, last, window, title, path):
    gammas = list(results)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    for key, color, label in (("all", "#2a78d6", "toute la durée"),
                              ("last", "#eb6834", f"{last} dernières étapes")):
        means = np.array([results[g][key].mean() for g in gammas])
        ses = np.array([results[g][key].std(ddof=1) / np.sqrt(len(results[g][key]))
                        for g in gammas])
        ax1.errorbar(gammas, means, yerr=2 * ses, marker="o", capsize=3,
                     color=color, label=label)
    ax1.set_xlabel("Facteur d'actualisation γ")
    ax1.set_ylabel("Récompense moyenne par étape (± 2 erreurs standard)")
    ax1.set_title("Récompense moyenne selon γ")
    ax1.legend(frameon=False)
    ax1.grid(True, color="#e5e5e5")

    shown = [g for g in (0, 0.1, 0.5, 1) if g in results]
    x = np.arange(window, steps + 1)
    for g, color, style in zip(shown, ("#2a78d6", "#eb6834", "#1baf7a", "#d1495b"),
                               ("-", "-", "--", ":")):
        ax2.plot(x, moving_average(results[g]["curve"], window), color=color,
                 linestyle=style, label=f"γ = {g:g}")
    ax2.axhline(0, color="#555555", linewidth=0.8)
    ax2.set_xlabel("Étape")
    ax2.set_ylabel(f"Récompense moyenne (moyenne glissante {window})")
    ax2.set_title("Courbes moyennes")
    ax2.legend(frameon=False, loc="upper left")
    ax2.grid(True, color="#e5e5e5")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_relances():
    signal = learning_signal()
    results = study_runs()
    chosen = choose_runs(results, signal)
    plot_study(results, signal, chosen)

    print(f"signal à mesurer (fin - début) ≈ {signal:+.3f}")
    print(f"{'n':>5} | {'bruit':>7} | {'temps/lot (s)':>13}")
    for n, r in results.items():
        print(f"{n:>5} | {r['noise']:7.3f} | {r['time']:13.2f}")
    print(f"nombre de relances retenu : {chosen}")


def run_gamma():
    for steps, runs, last, window, name, q in (
        (1_000, 1_000, 200, 50, "ex3_13_recherche_gamma", "3.13"),
        (10_000, 300, 1_000, 200, "ex3_14_gamma_10000", "3.14"),
    ):
        start = time.perf_counter()
        results = gamma_search(steps, runs, last)
        title = (f"Ex {q} — Effet de γ ({runs} parties de {steps} étapes, "
                 f"grille {SIZE}×{SIZE}, ε = {EPS})")
        print_summary(results, f"{q} : {runs} parties, {steps} étapes "
                      f"({time.perf_counter() - start:.0f} s)", last)
        plot_gamma_search(results, steps, last, window, title, f"figures/{name}.png")


# --- 4.7 : serpents plus longs ------------------------------------------------

LENGTHS_BUDGET = (1, 2, 3, 4, 5, 6, 8, 10)  # même budget : 10 000 étapes
LENGTHS_LONG = (1, 2, 3, 4, 5, 6)           # entraînement long : 1 000 000 étapes
LENGTH_RUNS = {"budget": 100, "long": 6}
LENGTH_WORKERS = {"budget": 8, "long": 3}  # l'entraînement long garde de gros tableaux Q


def _long_run(job):
    length, steps, block, seed, gamma = job
    random.seed(seed)
    game = ShortSnake(SIZE, length=length)
    player = GreedyPlayer(game, eps=EPS, gamma=gamma)
    rewards = np.zeros(steps, dtype=np.int8)
    self_hits = np.zeros(steps, dtype=bool)
    for t in range(steps):
        action = player.get_action()
        value = game.move(action)
        player.reward(action, value)
        rewards[t] = value
        # même test que dans ShortSnake.move : auto-intersection si longueur >= 5
        self_hits[t] = length >= 5 and game.snake_pos in game.body[1:]
    pairs = len(player.eval_count)
    revisited = sum(c > 1 for c in player.eval_count.values())
    return {
        "reward": rewards.reshape(-1, block).mean(axis=1),
        "hits": self_hits.reshape(-1, block).mean(axis=1),
        "pairs": pairs,
        "revisited": revisited,
    }


def length_study(lengths, runs, steps, block, workers, gamma=GAMMA, seed=SEED):
    """Pour chaque longueur, `runs` parties de `steps` étapes (mêmes graines
    pour toutes les longueurs). Renvoie, par longueur, des tableaux (runs, blocs)
    de récompense moyenne et de taux d'auto-intersection par bloc d'étapes,
    ainsi que le nombre de paires (état, action) vues à la fin."""
    jobs = [(L, steps, block, seed + r, gamma) for L in lengths for r in range(runs)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        outputs = list(pool.map(_long_run, jobs))
    results = {}
    for k, L in enumerate(lengths):
        part = outputs[k * runs:(k + 1) * runs]
        results[L] = {
            "reward": np.array([o["reward"] for o in part]),
            "hits": np.array([o["hits"] for o in part]),
            "pairs": np.array([o["pairs"] for o in part]),
            "revisited": np.array([o["revisited"] for o in part]),
        }
    return results


def pairs_order_of_magnitude(length, size=SIZE):
    """Ordre de grandeur du nombre de paires (état, action) possibles :
    3 actions × 4·N·M·3^(L-2)·(N·M - L) états pour L >= 2 (4.2 pour L = 2) ;
    la formule de 2.3 pour L = 1. Pour L >= 5 c'est une approximation, car le
    corps peut se recouvrir."""
    cells = size * size
    if length == 1:
        return 3 * 4 * cells * (cells - 1)
    return 3 * 4 * cells * 3 ** (length - 2) * (cells - length)


def se(values, axis=0):
    return values.std(axis=axis, ddof=1) / np.sqrt(values.shape[axis])


def print_length_summary(results, steps, block, title):
    nblocks = steps // block
    tail = max(1, nblocks // 10)
    print(title)
    print(f"  {'L':>3} | {'1er bloc':>9} | {'derniers 10 %':>14} ± {'ES':>5} | "
          f"{'auto-inters.':>12} | {'paires vues':>11} | {'revisitées':>10} | bloc moyen > 0")
    for L, r in results.items():
        mean_curve_ = r["reward"].mean(axis=0)
        last = r["reward"][:, -tail:].mean(axis=1)
        positive = np.flatnonzero(mean_curve_ > 0)
        first_pos = f"{(positive[0] + 1) * block}" if len(positive) else "jamais"
        print(f"  {L:>3} | {mean_curve_[0]:+9.2f} | {last.mean():+14.2f} ± {se(last):.2f} | "
              f"{r['hits'][:, -tail:].mean():12.3f} | {r['pairs'].mean():11.0f} | "
              f"{(r['revisited'] / r['pairs']).mean():10.1%} | {first_pos}")


def plot_length_study(results, steps, block, title, path):
    nblocks = steps // block
    tail = max(1, nblocks // 10)
    x = (np.arange(nblocks) + 0.5) * block
    lengths = list(results)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(lengths)))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.6))
    for L, color in zip(lengths, colors):
        m = results[L]["reward"].mean(axis=0)
        ax1.plot(x, m, color=color, label=f"L = {L}")
    ax1.axhline(0, color="#555555", linewidth=0.8)
    ax1.set_xlabel("Étape")
    ax1.set_ylabel(f"Récompense moyenne (blocs de {block} étapes)")
    ax1.set_title("Courbes d'apprentissage")
    ax1.legend(frameon=False, ncol=2, fontsize=8)
    ax1.grid(True, color="#e5e5e5")

    last = [results[L]["reward"][:, -tail:].mean(axis=1) for L in lengths]
    ax2.errorbar(lengths, [v.mean() for v in last], yerr=[2 * se(v) for v in last],
                 marker="o", capsize=3, color="#2a78d6")
    ax2.axhline(0, color="#555555", linewidth=0.8)
    ax2.set_xlabel("Longueur du serpent L")
    ax2.set_ylabel(f"Récompense moyenne sur les {tail * block} dernières étapes\n(± 2 ES)")
    ax2.set_title("Niveau atteint")
    ax2.grid(True, color="#e5e5e5")

    ax3.semilogy(lengths, [results[L]["pairs"].mean() for L in lengths], "o-",
                 color="#eb6834", label="paires (état, action) vues")
    ax3.semilogy(lengths, [pairs_order_of_magnitude(L) for L in lengths], "--",
                 color="#888888", label="paires possibles (ordre de grandeur)")
    ax3.axhline(steps, color="#d1495b", linewidth=1, label=f"{steps} étapes")
    ax3.set_xlabel("Longueur du serpent L")
    ax3.set_ylabel("Nombre de paires")
    ax3.set_title("Taille de l'espace d'états")
    ax3.legend(frameon=False, fontsize=8)
    ax3.grid(True, color="#e5e5e5")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_longueurs():
    experiments = (
        ("budget", LENGTHS_BUDGET, 10_000, 500, "ex4_7_budget_fixe", "même budget"),
        ("long", LENGTHS_LONG, 1_000_000, 20_000, "ex4_7_entrainement_long", "entraînement long"),
    )
    for key, lengths, steps, block, name, label in experiments:
        runs = LENGTH_RUNS[key]
        start = time.perf_counter()
        results = length_study(lengths, runs, steps, block, LENGTH_WORKERS[key])
        title = (f"Ex 4.7 — {label} : {runs} parties de {steps} étapes "
                 f"(grille {SIZE}×{SIZE}, γ = {GAMMA}, ε = {EPS})")
        print_length_summary(results, steps, block,
                             f"4.7 {label} : {runs} parties, {steps} étapes "
                             f"({time.perf_counter() - start:.0f} s)")
        plot_length_study(results, steps, block, title, f"figures/{name}.png")


if __name__ == "__main__":
    todo = set(sys.argv[1:]) or {"relances", "gamma", "longueurs"}
    if "relances" in todo:
        run_relances()
    if "gamma" in todo:
        run_gamma()
    if "longueurs" in todo:
        run_longueurs()
