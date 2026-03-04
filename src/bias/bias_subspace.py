"""
Bias Subspace Computation.

From Rakshit et al. (2024) / Bolukbasi et al. (2016):

Given n gender pairs (he/she, man/woman, ...):
    s_i = f(w_i^+) - f(w_i^-)           difference vectors
    C   = (1/n) sum_i s_i s_i^T         covariance matrix
    PCA: C v_i = lambda_i v_i
    B   = span(v_1, ..., v_k)            k-dimensional bias subspace
    P_B = sum_{i=1}^k v_i v_i^T         projection matrix (d x d)

Key difference from hard debiasing:
    - Hard debiasing uses only k=1 (single direction)
    - This method uses top-k PCs -> captures full bias SUBSPACE
"""

import numpy as np
from typing import Dict, List, Tuple


class BiasSubspace:
    """
    Computes and stores the bias subspace B for a given embedding space.

    Args:
        k: number of top principal components to retain (dimension of B)
    """

    def __init__(self, k: int = 10):
        self.k = k
        self.components_ = None    # shape (k, d): the v_1 ... v_k basis vectors
        self.P_B = None            # shape (d, d): projection matrix
        self.eigenvalues_ = None

    def fit(self, embeddings: Dict[str, np.ndarray],
            gender_pairs: List[Tuple[str, str]]) -> "BiasSubspace":
        """
        Fit the bias subspace from gender word pairs.

        Args:
            embeddings  : {word: vector}  -- the full embedding lookup
            gender_pairs: [(w_pos, w_neg)] e.g. [("he","she"), ("man","woman")]

        Returns:
            self (fitted)
        """
        diff_vectors = []
        for w_pos, w_neg in gender_pairs:
            if w_pos in embeddings and w_neg in embeddings:
                # s_i = f(w^+) - f(w^-)
                s_i = embeddings[w_pos] - embeddings[w_neg]
                diff_vectors.append(s_i)

        if len(diff_vectors) < 2:
            raise ValueError(f"Need >=2 valid pairs, got {len(diff_vectors)}")

        S = np.stack(diff_vectors)       # (n_pairs, d)

        # Covariance matrix C = (1/n) S^T S  (d x d)
        C = (S.T @ S) / len(diff_vectors)

        # PCA: eigendecompose C
        eigenvalues, eigenvectors = np.linalg.eigh(C)

        # Sort descending by eigenvalue
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]   # columns are eigenvectors

        # Take top k
        self.k = min(self.k, len(eigenvalues))
        self.components_ = eigenvectors[:, :self.k].T    # (k, d)
        self.eigenvalues_ = eigenvalues[:self.k]

        # Projection matrix P_B = sum_i v_i v_i^T
        V = self.components_.T                            # (d, k)
        self.P_B = V @ V.T                               # (d, d)

        return self

    def project(self, w: np.ndarray) -> np.ndarray:
        """
        f_B(w) = P_B * w   (biased component)
        """
        return self.P_B @ w

    def debias(self, w: np.ndarray) -> np.ndarray:
        """
        f_perp(w) = (I - P_B) * w   (bias-free component)
        """
        return w - self.project(w)

    def bias_score(self, w: np.ndarray) -> float:
        """
        Scalar bias score: magnitude of projection onto bias subspace.
        b(w) = || P_B w ||_2
        """
        return float(np.linalg.norm(self.project(w)))

    # ------------------------------------------------------------------
    # RQ1 helper: compute bias subspace at a specific PREFIX dimension d
    # ------------------------------------------------------------------
    def fit_at_prefix(self, embeddings: Dict[str, np.ndarray],
                      gender_pairs: List[Tuple[str, str]],
                      d: int) -> "BiasSubspace":
        """
        Fit bias subspace using only the first d dimensions of embeddings.
        Used to answer RQ1: Does B^(D) survive truncation to B^(d)?
        """
        truncated = {w: v[:d] for w, v in embeddings.items()}
        bs_d = BiasSubspace(k=self.k)
        bs_d.fit(truncated, gender_pairs)
        return bs_d
