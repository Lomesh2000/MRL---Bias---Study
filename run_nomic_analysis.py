"""
Script to analyze MRL properties of Nomic Embed using the existing pipeline.
"""

import sys
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import matplotlib
matplotlib.use("Agg")

# Add src to path
sys.path.append(os.getcwd())

from src.experiments.rq1_bias_subspace_survival import run_rq1
from src.experiments.rq3_bias_concentration import run_rq3
from src.experiments.rq2_debiasing_at_prefix import residual_bias_at_prefix
from src.experiments.rq4_adaptive_lambda import run_rq4
from src.bias.bias_subspace import BiasSubspace

# Configuration
# Nomic v1.5 supports MRL with flexible dimensionality.
# Common dims: 64, 128, 256, 512, 768
PREFIX_DIMS = [64, 128, 256, 512, 768]
K = 10
SAVE_DIR = "results_nomic/"
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

os.makedirs(os.path.join(SAVE_DIR, "bias_scores"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "figures"), exist_ok=True)

# -----------------------------------------------------------------------
# Step 1: Data Preparation (Vocabulary & Gender Pairs)
# -----------------------------------------------------------------------
GENDER_PAIRS = [
    ("he", "she"), ("man", "woman"), ("king", "queen"),
    ("boy", "girl"), ("his", "her"), ("him", "her"),
    ("father", "mother"), ("son", "daughter"), ("brother", "sister"),
    ("male", "female"), ("guy", "gal"), ("uncle", "aunt")
]

# Expanded vocabulary for robust PC analysis and projection
VOCAB_LIST = [
    # Gendered words
    "he", "she", "man", "woman", "king", "queen", "boy", "girl", "his", "her",
    "him", "father", "mother", "son", "daughter", "brother", "sister",
    "male", "female", "guy", "gal", "uncle", "aunt",
    # Professions (often stereotyped)
    "doctor", "nurse", "engineer", "teacher", "coder", "artist", "mechanic", "dancer",
    "scientist", "secretary", "manager", "assistant", "pilot", "steward",
    # Adjectives
    "smart", "caring", "strong", "weak", "dominant", "sensitive",
    "leader", "supporter", "rational", "emotional", "brave", "gentle",
    # Common Nouns
    "time", "year", "person", "way", "day", "thing", "world", "life", "hand", "part",
    "child", "eye", "woman", "place", "work", "week", "case", "point", "government", "company"
]

# Ensure unique
VOCAB_LIST = list(set(VOCAB_LIST))

print(f"Loading model: {MODEL_NAME}...")
print("Note: If this is the first time, it will download the model (~300MB). Please wait...")
# trust_remote_code=True is needed for Nomic, as it uses custom code
model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

print("Generating embeddings...")
# Nomic outputs normalized embeddings by default, Matryoshka happens via slicing
embeddings_matrix = model.encode(VOCAB_LIST, convert_to_numpy=True)
embeddings_nomic = {w: v for w, v in zip(VOCAB_LIST, embeddings_matrix)}

# Create a "random" baseline for contrast in plots (simulating non-MRL / no structure)
# This helps us see if Nomic's MRL structure is actually doing something different
print("Generating random baseline for contrast...")
dim = embeddings_matrix.shape[1]
embeddings_random = {
    w: np.random.randn(dim).astype(np.float32) 
    for w in VOCAB_LIST
}
# Normalize random embeddings
for w in embeddings_random:
    embeddings_random[w] /= np.linalg.norm(embeddings_random[w])

# -----------------------------------------------------------------------
# Step 2: Run RQ1 (Subspace Survival)
# -----------------------------------------------------------------------
print("\n--- Running RQ1: Bias Subspace Survival ---")
# We compare Nomic (MRL) against Random (Baseline)
# If Nomic is MRL, its bias subspace should survive better (lower alignment error) at low dims? 
# Or actually, MRL packs info, so bias might be preserved in prefix.
df_rq1 = run_rq1(
    embeddings_baseline=embeddings_random,
    embeddings_mrl=embeddings_nomic,
    gender_pairs=GENDER_PAIRS,
    prefix_dims=PREFIX_DIMS,
    k=min(K, len(GENDER_PAIRS)), # can't have k > num_pairs for definition usually, but PCA works on differences
    save_dir=SAVE_DIR
)

# -----------------------------------------------------------------------
# Step 3: Run RQ3 (Bias Concentration)
# -----------------------------------------------------------------------
# We expect Nomic to show "Amplification" (high C(d) at low d) or at least different from random
print("\n--- Running RQ3: Bias Concentration ---")
run_rq3(
    embeddings_baseline=embeddings_random,
    embeddings_mrl=embeddings_nomic,
    gender_pairs=GENDER_PAIRS,
    prefix_dims=PREFIX_DIMS,
    k=min(K, len(GENDER_PAIRS)),
    save_dir=SAVE_DIR
)

# -----------------------------------------------------------------------
# Step 4: Run RQ2 (Debiasing at Prefix)
# -----------------------------------------------------------------------
# We check if debiasing Nomic at full dim (768) cleans up the prefixes
print("\n--- Running RQ2: Residual Bias at Prefix ---")
# Fit bias subspace on full Nomic
bs_full = BiasSubspace(k=min(K, len(GENDER_PAIRS)))
bs_full.fit(embeddings_nomic, GENDER_PAIRS)

# Get residual bias L2 scores
res_scores = residual_bias_at_prefix(
    embeddings_nomic, bs_full, PREFIX_DIMS, GENDER_PAIRS
)

print("RQ2 Results (Nomic):")
for d, score in res_scores.items():
    print(f"  d={d}: L2_residual={score:.4f}")

# -----------------------------------------------------------------------
# Step 5: Run RQ4 (Optimal Lambda)
# -----------------------------------------------------------------------
print("\n--- Running RQ4: Optimal Lambda ---")
# Using a subset of words/pairs to keep it fast
# We need to adapt optimal_lambda_grid_search arguments if needed
# It expects embeddings, gender_pairs, prefix_dims
run_rq4(
    embeddings=embeddings_nomic,
    gender_pairs=GENDER_PAIRS,
    prefix_dims=PREFIX_DIMS,
    lambda_grid=[0.0, 0.2, 0.5, 0.8, 1.0, 2.0, 5.0],
    k=min(K, len(GENDER_PAIRS)),
    save_dir=SAVE_DIR
)

print(f"\nAnalysis complete. Results saved in {SAVE_DIR}")
