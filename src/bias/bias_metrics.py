"""
Bias Metrics:
  1. WEAT (Word Embedding Association Test) -- Caliskan et al. 2017
  2. C(d) Bias Concentration -- core metric for this paper
  3. Association score s(w, A, B)

WEAT Math:
    s(w, A, B) = mean_{a in A} cos(w,a) - mean_{b in B} cos(w,b)
    WEAT(X, Y, A, B) = sum_{x in X} s(x,A,B) - sum_{y in Y} s(y,A,B)
    d = [mean_x s(x,A,B) - mean_y s(y,A,B)] / std_{w in XuY} s(w,A,B)

Bias Concentration (our new metric):
    b^(d)(w) = <f^(d)(w), v_bias^(d)>
    C(d) = Var(b^(d)(w)) / Var(b^(D)(w))
    C(d) high at small d --> Bias Amplification
    C(d) low at small d  --> Bias Dilution
    C(d) flat            --> Bias Redistribution
"""

import numpy as np
from typing import Dict, List, Optional
# from scipy.stats import pearsonr  <-- Removed to fix hang on import


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    return float(np.dot(u, v) / denom) if denom > 0 else 0.0


def association_score(w: np.ndarray, A: List[np.ndarray],
                      B: List[np.ndarray]) -> float:
    """
    s(w, A, B) = mean_{a in A} cos(w,a) - mean_{b in B} cos(w,b)
    """
    mean_A = np.mean([cosine_similarity(w, a) for a in A])
    mean_B = np.mean([cosine_similarity(w, b) for b in B])
    return float(mean_A - mean_B)


def weat_effect_size(X: List[np.ndarray], Y: List[np.ndarray],
                     A: List[np.ndarray], B: List[np.ndarray]) -> float:
    """
    WEAT effect size d.
    d ≈ 0 --> no bias; |d| > 0.5 --> strong bias.
    """
    scores_X = [association_score(x, A, B) for x in X]
    scores_Y = [association_score(y, A, B) for y in Y]
    all_scores = scores_X + scores_Y

    mean_diff = np.mean(scores_X) - np.mean(scores_Y)
    std_all = np.std(all_scores, ddof=1)

    return float(mean_diff / std_all) if std_all > 0 else 0.0


def compute_bias_concentration(
        embeddings: Dict[str, np.ndarray],
        prefix_dims: List[int],
        bias_subspace_full,          # BiasSubspace fitted on full dim D
        fit_per_dim: bool = True     # if True: refit B at each d (for RQ1)
) -> Dict[int, float]:
    """
    Compute C(d) = Var(b^(d)(w)) / Var(b^(D)(w)) for each prefix dim d.

    Args:
        embeddings       : {word: vector(D)}
        prefix_dims      : list of d values e.g. [8,16,32,64,128,300]
        bias_subspace_full: fitted BiasSubspace on full D
        fit_per_dim      : if True, refit B separately at each d (RQ1 analysis)

    Returns:
        {d: C(d)} concentration ratio for each prefix level
    """
    words = list(embeddings.keys())
    D = next(iter(embeddings.values())).shape[0]

    # Variance at full dimension D (denominator)
    scores_full = [bias_subspace_full.bias_score(embeddings[w]) for w in words]
    var_full = float(np.var(scores_full))

    concentration = {}
    for d in prefix_dims:
        # Truncate embeddings to first d dims
        embs_d = {w: embeddings[w][:d] for w in words}

        # Project with full P_B then truncate result to d dims
        P_B_full = bias_subspace_full.P_B   # (D, D)
        scores_d = []
        for w in words:
            z = embeddings[w]               # (D,)
            z_B = P_B_full @ z              # (D,)
            scores_d.append(float(np.linalg.norm(z_B[:d])))

        var_d = float(np.var(scores_d))
        concentration[d] = var_d / var_full if var_full > 0 else 0.0

    return concentration


def classify_bias_regime(concentration: Dict[int, float]) -> str:
    """
    Given C(d) values, classify the bias regime per Sir's LaTeX (Section 5).

    Returns one of:
        'amplification'    -- C(d) high at small d
        'dilution'         -- C(d) increases with d
        'redistribution'   -- C(d) roughly flat, near constant total
    """
    dims = sorted(concentration.keys())
    values = [concentration[d] for d in dims]

    # Pearson correlation of C(d) with d
    if len(dims) < 3:
        return "unknown"

    # Avoid scipy dependency for simple correlation check
    if np.std(values) == 0:
        corr = 0.0
    else:
        corr = np.corrcoef(dims, values)[0, 1]

    # High C at small dims (negative correlation with d) -> amplification
    if corr < -0.4 and values[0] > 0.7:
        return "amplification"
    # C grows with d -> dilution (bias pushed to later dims)
    elif corr > 0.4:
        return "dilution"
    # Neither -> redistribution (flat, bias just moved)
    else:
        return "redistribution"
