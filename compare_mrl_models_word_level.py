"""
Comparison of MRL vs Non-MRL Models for Bias Analysis.

Models:
- With MRL: tomaarsen/mpnet-base-nli-matryoshka
- Without MRL: tomaarsen/mpnet-base-nli
"""

import sys
import gc
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import matplotlib
matplotlib.use("Agg")
import torch
torch.set_num_threads(2)

# Add src to path
sys.path.append(os.getcwd())

from src.experiments.rq1_bias_subspace_survival import run_rq1
from src.experiments.rq3_bias_concentration import run_rq3
from src.experiments.rq2_debiasing_at_prefix import residual_bias_at_prefix
from src.experiments.rq4_adaptive_lambda import run_rq4, soft_debias_with_lambda
from src.bias.bias_subspace import BiasSubspace
from src.bias.stereoset_last_hidden_state import evaluate_stereoset

# Configuration
PREFIX_DIMS = [64, 128, 256, 384, 512, 768]
K = 10
SAVE_DIR = "results_comparison_word_level/"
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

    model = SentenceTransformer(model_name, trust_remote_code=True, device="cpu")

    tokenizer = model.tokenizer
    transformer = model._first_module().auto_model

    embeddings = {}

    print("Generating token-level embeddings...", flush=True)

    for word in vocab:
        inputs = tokenizer(word, return_tensors="pt")

        with torch.no_grad():
            outputs = transformer(**inputs)

        tokens = outputs.last_hidden_state[0]

        # remove CLS and SEP
        word_tokens = tokens[1:-1]

        # average if multiple tokens
        vec = word_tokens.mean(dim=0).cpu().numpy()

        embeddings[word] = vec

    del model
    gc.collect()

    return embeddings

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


# -----------------------------------------------------------------------
# Step 6: StereoSet Evaluation (SS, LMS, ICAT)
# -----------------------------------------------------------------------
print("\n--- Running Step 6: StereoSet Evaluation ---")

# Load models again? It's better to pass the models if they weren't loaded
# But 'get_embeddings' didn't return the model object, only embeddings.
# We need the model object for sentence encoding of new sentences.

def run_stereoset_for_model(model_name, label):
    print(f"Loading {model_name} for StereoSet...", flush=True)
    model = SentenceTransformer(model_name, trust_remote_code=True)
    
    results = []
    # Test at different prefix dimensions
    # MRL model supports truncation. Non-MRL technically doesn't but we can truncate manually.
    dims_to_test = PREFIX_DIMS  # [64, 128, ...]
    
    for d in dims_to_test:
        print(f"  Evaluating {label} at d={d}...", flush=True)
        scores = evaluate_stereoset(
            model, 
            dataset_path="data/stereoset_dev.json",
            prefix_dim=d,
            tokenizer=model.tokenizer,
            transformer=model._first_module().auto_model,
            use_last_hidden_state=True
        )
        scores["dim_d"] = d
        scores["model"] = label
        results.append(scores)
        print(f"    d={d} | SS={scores['ss']:.2f} | LMS={scores['lms']:.2f} | ICAT={scores['icat']:.2f}")
    
    del model
    gc.collect()
    return results

# Run for MRL
res_mrl_ss = run_stereoset_for_model(MODEL_MRL, "MRL")

# Run for Non-MRL
res_no_mrl_ss = run_stereoset_for_model(MODEL_NO_MRL, "Non-MRL")

# Combine and save
df_ss = pd.DataFrame(res_mrl_ss + res_no_mrl_ss)
csv_ss_path = os.path.join(SAVE_DIR, "bias_scores", "stereoset_results.csv")
df_ss.to_csv(csv_ss_path, index=False)
print(f"Saved StereoSet results to {csv_ss_path}")

# Plot StereoSet
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = ["ss", "lms", "icat"]
titles = ["Stereotype Score (SS) - Ideal 50", "Language Modeling (LMS) - Ideal 100", "ICAT Score - Ideal 100"]

for ax, met, tit in zip(axes, metrics, titles):
    for label in ["MRL", "Non-MRL"]:
        sub = df_ss[df_ss["model"] == label]
        ax.plot(sub["dim_d"], sub[met], marker="o", label=label)
    
    ax.set_title(tit)
    ax.set_xlabel("Dimension d")
    ax.set_ylabel(met.upper())
    
    # Add ideal lines
    if "ss" in met:
        ax.axhline(50, color='r', linestyle='--', alpha=0.5, label="Ideal (50)")
    elif "lms" in met or "irat" in met:
        ax.axhline(100, color='r', linestyle='--', alpha=0.5, label="Ideal (100)")
        
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "figures", "stereoset_comparison.png"), dpi=150)
plt.close()

print(f"\nAll experiments complete. Results in {SAVE_DIR}")
