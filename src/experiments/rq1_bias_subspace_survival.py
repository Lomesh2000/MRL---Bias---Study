"""
RQ1 — Does the bias subspace B^(D) survive prefix truncation?

From notes (IMG_6967 / IMG_6965):
    "Rakshit method computes P_B on full embedding R^D.
     But our problem: when we truncate to prefix f^(d)(w),
     does bias subspace B still exist within the prefix?
     Does Pi_d(B^(D)) still capture the same bias subspace as B^(d)?"

Formally:
    Define prefix projection Pi_d: R^D -> R^d
    Question: Is Pi_d(B^(D)) a good approximation of B^(d)?

    Measure: subspace alignment score between Pi_d(B^(D)) and B^(d)
    
    Alignment = || P_{B^(d)} - Pi_d P_{B^(D)} Pi_d^T ||_F / k

    If alignment ≈ 0  --> B^(D) survives truncation (good)
    If alignment >> 0 --> bias subspace CHANGES at each prefix level
                         (Rakshit's P_B is not valid at smaller d)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") # for headless environments (e.g. Colab, servers)
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os

from src.bias.bias_subspace import BiasSubspace
from src.bias.bias_metrics import compute_bias_concentration, classify_bias_regime


def run_rq1(embeddings_baseline: Dict[str, np.ndarray],
            embeddings_mrl: Dict[str, np.ndarray],
            gender_pairs: List[Tuple[str, str]],
            prefix_dims: List[int],
            k: int = 10,
            save_dir: str = "results/") -> pd.DataFrame:
    """
    Main experiment for RQ1.

    For each prefix dim d:
      1. Fit B^(D) on full embeddings (Rakshit's approach)
      2. Fit B^(d) on truncated embeddings (prefix-level subspace)
      3. Compute subspace alignment between Pi_d(B^(D)) and B^(d)
      4. Compare for baseline vs MRL embeddings

    Returns: DataFrame with alignment scores per dim.
    """
    D = next(iter(embeddings_baseline.values())).shape[0]
    results = []

    for emb_name, embeddings in [("baseline", embeddings_baseline),
                                   ("mrl", embeddings_mrl)]:
        print(f"\n--- RQ1: {emb_name} embeddings ---")

        # Fit bias subspace on FULL dimension D (Rakshit's method)
        bs_full = BiasSubspace(k=k)
        bs_full.fit(embeddings, gender_pairs)
        P_B_full = bs_full.P_B   # (D, D)

        for d in prefix_dims:
            # B^(d): fit bias subspace at prefix dimension d
            bs_d = BiasSubspace(k=k)
            truncated = {w: embeddings[w][:d] for w in embeddings}
            try:
                bs_d.fit(truncated, gender_pairs)
            except ValueError:
                continue

            P_B_d = bs_d.P_B    # (d, d)

            # Pi_d(B^(D)): project full P_B down to first d dims
            # This is the top-left (d x d) block of P_B_full
            P_B_full_truncated = P_B_full[:d, :d]

            # Subspace alignment: Frobenius distance (normalised)
            alignment_err = np.linalg.norm(P_B_d - P_B_full_truncated, 'fro')
            alignment_err /= k   # normalise by subspace dimension

            # Also measure: how much bias variance is captured
            # at this prefix level (C(d) from bias concentration)
            words = list(embeddings.keys())
            scores_full = [bs_full.bias_score(embeddings[w]) for w in words]
            scores_d = [bs_d.bias_score(truncated[w]) for w in words]

            var_full = float(np.var(scores_full))
            var_d = float(np.var(scores_d))
            C_d = var_d / var_full if var_full > 0 else 0.0

            results.append({
                "model": emb_name,
                "dim_d": d,
                "alignment_error": alignment_err,
                "C_d": C_d,
                "var_full": var_full,
                "var_d": var_d
            })
            print(f"  d={d:4d} | alignment_err={alignment_err:.4f} | C(d)={C_d:.4f}")

    df = pd.DataFrame(results)

    # Save
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "bias_scores", "rq1_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved RQ1 results -> {csv_path}")

    # Plot
    _plot_rq1(df, save_dir)
    return df


def _plot_rq1(df: pd.DataFrame, save_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, ylabel, title in zip(
        axes,
        ["alignment_error", "C_d"],
        ["Subspace Alignment Error (lower=B survives)", "C(d) Bias Concentration"],
        ["RQ1: Does B^(D) survive truncation?",
         "RQ1: Bias concentration at prefix level d"]
    ):
        for model in df["model"].unique():
            sub = df[df["model"] == model]
            ax.plot(sub["dim_d"], sub[metric], marker="o", label=model)
        ax.set_xlabel("Prefix dimension d")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "figures", "rq1_bias_subspace_survival.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved RQ1 plot -> {path}")
    plt.close()
