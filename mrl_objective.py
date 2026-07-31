"""
MRL Nested Loss — Multi-task loss over nested embedding prefixes.

Reference: Kusupati et al., "Matryoshka Representation Learning", NeurIPS 2022.

Math:
    L_MRL = sum_{l=1}^{L} c_l * L(W_l * Pi_{m_l}(f(x)), y)

Key insight: dimension j receives gradient from ALL loss terms l where m_l >= j.
Early dimensions get L times more gradient signal --> concentration of information.
"""

import torch
import torch.nn as nn
from typing import List


class MRLLoss(nn.Module):
    """
    Nested MRL loss across multiple prefix dimensions.

    Args:
        dims  : list of prefix sizes, e.g. [8, 16, 32, 64, 128, 300]
        num_classes: number of output classes
        weights: optional per-dim loss weights c_l (default: uniform=1)
    """

    def __init__(self, dims: List[int], num_classes: int, weights: List[float] = None):
        super().__init__()
        self.dims = sorted(dims)                         # d1 < d2 < ... < D
        self.weights = weights or [1.0] * len(dims)     # c_l coefficients
        # Separate classifier head for each prefix size
        self.heads = nn.ModuleList([
            nn.Linear(d, num_classes) for d in self.dims
        ])
        self.ce = nn.CrossEntropyLoss()

    def forward(self, z: torch.Tensor, y: torch.Tensor):
        """
        Args:
            z : full embedding [batch, D]
            y : class labels   [batch]
        Returns:
            total scalar loss, dict of per-dim losses
        """
        total_loss = 0.0
        per_dim_losses = {}

        for l, (d, w, head) in enumerate(zip(self.dims, self.weights, self.heads)):
            # Pi_{m_l}: truncate to first d dimensions
            z_prefix = z[:, :d]                         # [batch, d]
            logits = head(z_prefix)
            loss_l = self.ce(logits, y)
            total_loss = total_loss + w * loss_l
            per_dim_losses[f"loss_d{d}"] = loss_l.item()

        return total_loss, per_dim_losses


def prefix_project(z: torch.Tensor, d: int) -> torch.Tensor:
    """
    Pi_d: R^D -> R^d  (truncate to first d dims).
    Used during evaluation.
    """
    return z[:, :d]
