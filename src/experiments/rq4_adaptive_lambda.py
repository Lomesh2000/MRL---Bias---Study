"""
RQ4 — Should the debiasing trade-off lambda_2 depend on prefix dimension d?

From notes (IMG_6968):
    "Should lambda trade-off depend on prefix dimension?
     Maybe optimal. lambda_2^(d) needs to scale with d."

Intuition:
    - MRL concentrates task info in early dims (RQ3 result)
    - Bias may also concentrate in early dims
    - Stronger debiasing (larger lambda_2) may be needed at small d
      to counteract the amplification effect
    - But too large lambda_2 at small d may destroy task info

Experiment:
    Grid search over lambda_2 values at each prefix dim d.
    Measure trade-off: bias reduction (L2^(d) decrease) vs
                       semantic preservation (cosine sim to original).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from typing import Dict, List, Tuple
import os

from src.bias.bias_subspace import BiasSubspace


def soft_debias_with_lambda(embeddings: Dict[str, np.ndarray],
                            bs: BiasSubspace,
                            lambda_2: float) -> Dict[str, np.ndarray]:
    """
    Simple soft debiasing at a given lambda_2:
        f̃(w) = f(w) - lambda_2 * P_B f(w)
              = (I - lambda_2 * P_B) f(w)

    lambda_2 = 0  --> no debiasing
    lambda_2 = 1  --> full projection (hard debias)
    lambda_2 in (0,1) --> soft partial debiasing
    """
    debiased = {}
    for w, v in embeddings.items():
        bias_comp = bs.P_B @ v
        debiased[w] = v - lambda_2 * bias_comp
    return debiased


def evaluate_tradeoff(truncated_original: Dict[str, np.ndarray],
                      truncated_debiased: Dict[str, np.ndarray],
                      bs_d: BiasSubspace,
                      d: int) -> Tuple[float, float]:
    """
    Returns:
        bias_score   : mean L2 bias at prefix d (lower is better)
        semantic_sim : mean cosine similarity to original (higher is better)    
    """
    from src.bias.bias_metrics import cosine_similarity
    words = list(truncated_original.keys())

    bias_scores = []
    cos_sims = []
    for w in words:
        v_orig = truncated_original[w]
        v_deb = truncated_debiased[w]
        bias_scores.append(float(np.linalg.norm(bs_d.project(v_deb))))
        cos_sims.append(cosine_similarity(v_orig, v_deb))

    return float(np.mean(bias_scores)), float(np.mean(cos_sims))


def run_rq4(embeddings: Dict[str, np.ndarray],
            gender_pairs: List[Tuple[str, str]],
            prefix_dims: List[int],
            lambda_grid: List[float] = None,
            k: int = 10,
            save_dir: str = "results/") -> pd.DataFrame:
    """
    Grid search over lambda_2 x prefix_dims.
    Find optimal lambda_2 per dim d.
    """
    if lambda_grid is None:
        lambda_grid = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]

    results = []
    for d, lam in product(prefix_dims, lambda_grid):
        # Truncate orig embeddings to prefix d
        truncated_orig = {w: embeddings[w][:d] for w in embeddings}
        
        # Fit bias subspace at prefix d
        bs_d = BiasSubspace(k=k)
        try:
            bs_d.fit(truncated_orig, gender_pairs)
        except ValueError:
            continue

        # Debias natively inside this d-dimensional truncated space
        debiased_d = soft_debias_with_lambda(truncated_orig, bs_d, lam)

        bias_score, sem_sim = evaluate_tradeoff(
            truncated_orig, debiased_d, bs_d, d)

        results.append({
            "dim_d": d,
            "lambda_2": lam,
            "mean_bias": bias_score,
            "semantic_sim": sem_sim,
            # Combined score: low bias AND high semantic sim
            "combined": sem_sim - bias_score
        })

    df = pd.DataFrame(results)

    # For each dim, find optimal lambda
    optimal = df.loc[df.groupby("dim_d")["combined"].idxmax()][
        ["dim_d", "lambda_2", "mean_bias", "semantic_sim"]
    ]
    print("\nOptimal lambda_2 per prefix dim:")
    print(optimal.to_string(index=False))

    os.makedirs(os.path.join(save_dir, "bias_scores"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "figures"), exist_ok=True)

    df.to_csv(os.path.join(save_dir, "bias_scores", "rq4_lambda_grid.csv"), index=False)
    optimal.to_csv(os.path.join(save_dir, "bias_scores", "rq4_optimal_lambda.csv"), index=False)

    # Plot optimal lambda vs d
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(optimal["dim_d"], optimal["lambda_2"], marker="o", linewidth=2, color="purple")
    ax.set_xlabel("Prefix dimension d")
    ax.set_ylabel("Optimal lambda_2^(d)")
    ax.set_title("RQ4: Does optimal lambda_2 depend on prefix dim?")
    ax.grid(True, alpha=0.3)

    # Heatmap of combined score
    ax = axes[1]
    pivot = df.pivot(index="lambda_2", columns="dim_d", values="combined")
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{l:.1f}" for l in pivot.index])
    ax.set_xlabel("Prefix dimension d")
    ax.set_ylabel("lambda_2")
    ax.set_title("Trade-off: sem_sim - mean_bias")
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "figures", "rq4_adaptive_lambda.png"), dpi=150)
    plt.close()

    return df, optimal
