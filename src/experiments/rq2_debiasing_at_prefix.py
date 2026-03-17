"""
RQ2 — After DeepSoftDebias (DSD) on full embeddings,
       does bias removal hold at prefix level?

From notes (IMG_6968):
    "After DSD, P_{B^(D)} f̃(w) ≈ 0.
     But what about P_{B^(d)} f̃^(d)(w)?
     Is L2^(d) = sum_w || P_{B^(d)} f̃^(d)(w) ||^2 still small?"

Experiment:
    1. Debias using Rakshit method (full dim D)
    2. At each prefix d, measure residual bias L2^(d)
    3. Compare: debiased_full vs raw at each prefix level
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os

from src.bias.bias_subspace import BiasSubspace


def residual_bias_at_prefix(embeddings_raw: Dict[str, np.ndarray],
                             embeddings_eval: Dict[str, np.ndarray],
                             prefix_dims: List[int],
                             gender_pairs: List[Tuple[str, str]],
                             k: int) -> Dict[int, float]:
    """
    For each prefix dim d, compute:
        L2^(d) = sum_w || P_{B_{raw}^(d)} * f_{eval}^(d)(w) ||^2

    This tells us how much bias remains inside the prefix evaluated against
    the original unmodified bias subspace.
    """
    residuals = {}
    for d in prefix_dims:
        # Fit bias subspace at prefix level d using RAW embeddings
        bs_d = BiasSubspace(k=k)
        truncated_raw = {w: embeddings_raw[w][:d] for w in embeddings_raw}
        truncated_eval = {w: embeddings_eval[w][:d] for w in embeddings_eval}
        
        try:
            bs_d.fit(truncated_raw, gender_pairs)
        except ValueError:
            residuals[d] = float("nan")
            continue

        # L2^(d) = sum_w || P_{B_{raw}^(d)} f_{eval}^(d)(w) ||^2
        L2_d = sum(
            float(np.linalg.norm(bs_d.project(truncated_eval[w])) ** 2)
            for w in truncated_eval
        )
        # Normalise by number of words
        residuals[d] = L2_d / len(truncated_eval)

    return residuals


def run_rq2(embeddings_raw: Dict[str, np.ndarray],
            embeddings_debiased: Dict[str, np.ndarray],
            gender_pairs: List[Tuple[str, str]],
            prefix_dims: List[int],
            k: int = 10,
            save_dir: str = "results/") -> pd.DataFrame:
    """
    Compare residual bias L2^(d) before and after DSD,
    for both baseline and MRL embeddings.
    """
    residuals_raw = residual_bias_at_prefix(
        embeddings_raw, embeddings_raw, prefix_dims, gender_pairs, k)
    residuals_deb = residual_bias_at_prefix(
        embeddings_raw, embeddings_debiased, prefix_dims, gender_pairs, k)

    rows = []
    for d in prefix_dims:
        rows.append({
            "dim_d": d,
            "L2_raw": residuals_raw.get(d, float("nan")),
            "L2_debiased": residuals_deb.get(d, float("nan")),
        })
        print(f"  d={d:4d} | L2_raw={residuals_raw.get(d,0):.4f}"
              f" | L2_debiased={residuals_deb.get(d,0):.4f}")

    df = pd.DataFrame(rows)
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, "bias_scores", "rq2_residual_bias.csv"), index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["dim_d"], df["L2_raw"], marker="o", label="Before DSD (raw)")
    ax.plot(df["dim_d"], df["L2_debiased"], marker="s", label="After DSD")
    ax.set_xlabel("Prefix dimension d")
    ax.set_ylabel("L2^(d) = mean ||P_{B^(d)} f^(d)(w)||^2")
    ax.set_title("RQ2: Does debiasing hold at prefix level?")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "figures", "rq2_residual_bias.png"), dpi=150)
    plt.close()

    return df
