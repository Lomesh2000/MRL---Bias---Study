"""
RQ3 — Does MRL concentrate bias in early dimensions?

From notes (IMG_6968):
    "Since MRL enforces I(f^(d1)(w); Y) to be high,
     bias might be stronger in early dim, weaker in later dim."

    C(d) = Var(b^(d)(w)) / Var(b^(D)(w))

Three regimes (Sir's LaTeX Section 5):
    Amplification : C(d) high at small d  --> bias packed into early dims by MRL
    Dilution      : C(d) grows with d     --> bias pushed to later dims
    Redistribution: C(d) roughly flat     --> bias moves but total is constant
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os

from src.bias.bias_subspace import BiasSubspace
from src.bias.bias_metrics import (
    compute_bias_concentration, weat_effect_size, classify_bias_regime
)


def run_rq3(embeddings_baseline: Dict[str, np.ndarray],
            embeddings_mrl: Dict[str, np.ndarray],
            gender_pairs: List[Tuple[str, str]],
            prefix_dims: List[int],
            weat_sets: Dict = None,
            k: int = 10,
            save_dir: str = "results/") -> pd.DataFrame:
    """
    Compute C(d) for baseline and MRL embeddings across all prefix dims.
    Also compute WEAT effect size at each prefix dim.
    Classify each model into bias regime.
    """
    results = []

    for emb_name, embeddings in [("baseline", embeddings_baseline),
                                   ("mrl", embeddings_mrl)]:
        print(f"\n=== RQ3: {emb_name} ===")

        # Fit bias subspace on full dim
        bs_full = BiasSubspace(k=k)
        bs_full.fit(embeddings, gender_pairs)

        # C(d) concentration
        concentration = compute_bias_concentration(
            embeddings, prefix_dims, bs_full, fit_per_dim=False
        )

        # WEAT at each prefix dim
        for d in prefix_dims:
            embs_d = {w: embeddings[w][:d] for w in embeddings}
            weat_d = None
            if weat_sets is not None:
                try:
                    X = [embs_d[w] for w in weat_sets["X"] if w in embs_d]
                    Y = [embs_d[w] for w in weat_sets["Y"] if w in embs_d]
                    A = [embs_d[w] for w in weat_sets["A"] if w in embs_d]
                    B = [embs_d[w] for w in weat_sets["B"] if w in embs_d]
                    if X and Y and A and B:
                        weat_d = weat_effect_size(X, Y, A, B)
                except Exception:
                    pass

            results.append({
                "model": emb_name,
                "dim_d": d,
                "C_d": concentration[d],
                "weat_d": weat_d,
            })
            print(f"  d={d:4d} | C(d)={concentration[d]:.4f}"
                  + (f" | WEAT={weat_d:.4f}" if weat_d is not None else ""))

        regime = classify_bias_regime(concentration)
        print(f"  --> Bias Regime for {emb_name}: {regime.upper()}")

    df = pd.DataFrame(results)

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "bias_scores", "rq3_concentration.csv")
    df.to_csv(csv_path, index=False)

    _plot_rq3(df, save_dir)
    return df


def _plot_rq3(df: pd.DataFrame, save_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # C(d) plot
    ax = axes[0]
    for model in df["model"].unique():
        sub = df[df["model"] == model].sort_values("dim_d")
        ax.plot(sub["dim_d"], sub["C_d"], marker="o", label=model, linewidth=2)
    ax.axhline(1.0, color="grey", linestyle="--", alpha=0.5, label="full dim baseline")
    ax.set_xlabel("Prefix dimension d")
    ax.set_ylabel("C(d) = Var(b^(d)) / Var(b^(D))")
    ax.set_title("RQ3: Bias Concentration C(d) — Baseline vs MRL")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # WEAT plot
    ax = axes[1]
    for model in df["model"].unique():
        sub = df[df["model"] == model].dropna(subset=["weat_d"]).sort_values("dim_d")
        if len(sub):
            ax.plot(sub["dim_d"], sub["weat_d"].abs(), marker="s",
                    label=model, linewidth=2)
    ax.set_xlabel("Prefix dimension d")
    ax.set_ylabel("|WEAT effect size d|")
    ax.set_title("RQ3: WEAT Bias at Each Prefix Level")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "figures", "rq3_bias_concentration.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved RQ3 plot -> {path}")
    plt.close()
