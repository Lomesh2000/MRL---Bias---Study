"""
Comparison of MRL vs Non-MRL Models for Bias Analysis.

Models:
- With MRL: tomaarsen/mpnet-base-nli-matryoshka
- Without MRL: tomaarsen/mpnet-base-nli
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import matplotlib
matplotlib.use("Agg")

# Add src to path
sys.path.append(os.getcwd())

from src.experiments.rq1_bias_subspace_survival import run_rq1
from src.experiments.rq3_bias_concentration import run_rq3
from src.experiments.rq2_debiasing_at_prefix import residual_bias_at_prefix
from src.experiments.rq4_adaptive_lambda import run_rq4, soft_debias_with_lambda
from src.bias.bias_subspace import BiasSubspace

# Configuration
PREFIX_DIMS = [64, 128, 256, 384, 512, 768]
K = 10
SAVE_DIR = "results_comparison/"
MODEL_MRL = "tomaarsen/mpnet-base-nli-matryoshka"
MODEL_NO_MRL = "tomaarsen/mpnet-base-nli"

os.makedirs(os.path.join(SAVE_DIR, "bias_scores"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "figures"), exist_ok=True)

# -----------------------------------------------------------------------
# Step 1: Data Preparation
# -----------------------------------------------------------------------
GENDER_PAIRS = [
    ("he", "she"), ("man", "woman"), ("king", "queen"),
    ("boy", "girl"), ("his", "her"), ("him", "her"),
    ("father", "mother"), ("son", "daughter"), ("brother", "sister"),
    ("male", "female"), ("guy", "gal"), ("uncle", "aunt")
]

VOCAB_LIST = [
    "he", "she", "man", "woman", "king", "queen", "boy", "girl", "his", "her",
    "him", "father", "mother", "son", "daughter", "brother", "sister",
    "male", "female", "guy", "gal", "uncle", "aunt",
    "doctor", "nurse", "engineer", "teacher", "coder", "artist", "mechanic", "dancer",
    "scientist", "secretary", "manager", "assistant", "pilot", "steward",
    "smart", "caring", "strong", "weak", "dominant", "sensitive",
    "leader", "supporter", "rational", "emotional", "brave", "gentle",
    "time", "year", "person", "way", "day", "thing", "world", "life", "hand", "part",
    "child", "eye", "woman", "place", "work", "week", "case", "point", "government", "company"
]
VOCAB_LIST = list(set(VOCAB_LIST))

def get_embeddings(model_name, vocab):
    print(f"Loading {model_name}...", flush=True)
    model = SentenceTransformer(model_name, trust_remote_code=True)
    print("Generating embeddings...", flush=True)
    embeddings_matrix = model.encode(vocab, convert_to_numpy=True)
    return {w: v for w, v in zip(vocab, embeddings_matrix)}

print("--- Step 1: Load Models & Embeddings ---")
embeddings_mrl = get_embeddings(MODEL_MRL, VOCAB_LIST)
embeddings_no_mrl = get_embeddings(MODEL_NO_MRL, VOCAB_LIST)

# -----------------------------------------------------------------------
# Step 2: RQ1 (Subspace Survival)
# -----------------------------------------------------------------------
print("\n--- Running RQ1: Bias Subspace Survival ---")
# 'embeddings_baseline' here refers to the Non-MRL model for comparison
df_rq1 = run_rq1(
    embeddings_baseline=embeddings_no_mrl,
    embeddings_mrl=embeddings_mrl,
    gender_pairs=GENDER_PAIRS,
    prefix_dims=PREFIX_DIMS,
    k=min(K, len(GENDER_PAIRS)),
    save_dir=SAVE_DIR
)

# -----------------------------------------------------------------------
# Step 3: RQ3 (Bias Concentration)
# -----------------------------------------------------------------------
print("\n--- Running RQ3: Bias Concentration ---")
run_rq3(
    embeddings_baseline=embeddings_no_mrl,
    embeddings_mrl=embeddings_mrl,
    gender_pairs=GENDER_PAIRS,
    prefix_dims=PREFIX_DIMS,
    k=min(K, len(GENDER_PAIRS)),
    save_dir=SAVE_DIR
)

# -----------------------------------------------------------------------
# Step 4: RQ2 (Debiasing at Prefix) - Custom implementation for comparison
# -----------------------------------------------------------------------
print("\n--- Running RQ2: Residual Bias at Prefix ---")

def calc_residual_curve(embeddings, label):
    # Fit bias subspace on full embeddings
    bs_full = BiasSubspace(k=min(K, len(GENDER_PAIRS)))
    bs_full.fit(embeddings, GENDER_PAIRS)
    
    # Debias FULL embeddings (lambda=1.0)
    debiased_full = soft_debias_with_lambda(embeddings, bs_full, lambda_2=1.0)
    
    # Calculate residual bias at each prefix for the DEBIASED embeddings
    # We want to see if removing bias at D=768 removes it at d=64
    res_scores = residual_bias_at_prefix(
        debiased_full, bs_full, PREFIX_DIMS, GENDER_PAIRS
    )
    return res_scores

res_no_mrl = calc_residual_curve(embeddings_no_mrl, "Non-MRL")
res_mrl = calc_residual_curve(embeddings_mrl, "MRL")

# Save & Plot RQ2
df_rq2 = pd.DataFrame({
    "dim_d": PREFIX_DIMS,
    "Non-MRL (Residual)": [res_no_mrl[d] for d in PREFIX_DIMS],
    "MRL (Residual)": [res_mrl[d] for d in PREFIX_DIMS]
})
csv_path = os.path.join(SAVE_DIR, "bias_scores", "rq2_comparison.csv")
df_rq2.to_csv(csv_path, index=False)
print(f"Saved RQ2 comparison to {csv_path}")

plt.figure(figsize=(8, 5))
plt.plot(df_rq2["dim_d"], df_rq2["Non-MRL (Residual)"], marker="o", label="Non-MRL (Debiased)")
plt.plot(df_rq2["dim_d"], df_rq2["MRL (Residual)"], marker="s", label="MRL (Debiased)")
plt.xlabel("Prefix dimension d")
plt.ylabel("Residual Bias L2")
plt.title("RQ2: Residual Bias after Full-Dim Debiasing")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(SAVE_DIR, "figures", "rq2_comparison.png"), dpi=150)
plt.close()

# -----------------------------------------------------------------------
# Step 5: RQ4 (Optimal Lambda)
# -----------------------------------------------------------------------
print("\n--- Running RQ4: Optimal Lambda ---")
# Run grid search for both
print("Grid search for MRL model...")
df_rq4_mrl, _ = run_rq4(
    embeddings_mrl, GENDER_PAIRS, PREFIX_DIMS,
    lambda_grid=[0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
    save_dir=os.path.join(SAVE_DIR, "mrl")
)

print("Grid search for Non-MRL model...")
df_rq4_no_mrl, _ = run_rq4(
    embeddings_no_mrl, GENDER_PAIRS, PREFIX_DIMS,
    lambda_grid=[0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
    save_dir=os.path.join(SAVE_DIR, "non_mrl")
)

print(f"\nAll experiments complete. Results in {SAVE_DIR}")
