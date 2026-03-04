"""
Master experiment runner.
Follows the exact pipeline from Sir's LaTeX (Section 4) + your RQs.

PIPELINE:
    Step 0: Load config
    Step 1: Prepare data (small Reddit subset or toy data)
    Step 2: Train baseline Word2Vec + MRL Word2Vec
    Step 3: Load gender pairs and WEAT word lists
    Step 4: RQ1 -- bias subspace survival under truncation
    Step 5: RQ3 -- bias concentration C(d) baseline vs MRL
    Step 6: Apply debiasing (Rakshit method)
    Step 7: RQ2 -- does debiasing hold at prefix level?
    Step 8: RQ4 -- optimal lambda_2 per prefix dim
"""

import matplotlib
matplotlib.use("Agg") # for headless environments (e.g. Colab, servers
import yaml
import numpy as np
import os

# -----------------------------------------------------------------------
# Step 0: Config
# -----------------------------------------------------------------------
with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

PREFIX_DIMS = cfg["embeddings"]["mrl_dims"]
K = cfg["bias_subspace"]["k"]
SAVE_DIR = cfg["results"]["save_dir"]
os.makedirs(os.path.join(SAVE_DIR, "bias_scores"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "figures"), exist_ok=True)

print("=" * 60)
print("MRL Bias Experiment Pipeline")
print(f"Prefix dims: {PREFIX_DIMS}")
print(f"Bias subspace k: {K}")
print("=" * 60)

# -----------------------------------------------------------------------
# Step 1: Data preparation (use toy data if real corpus not available yet)
# -----------------------------------------------------------------------
def load_or_create_toy_data():
    """
    Creates small synthetic embeddings for quick pipeline testing.
    Replace with real Word2Vec trained on Reddit-L2 once data is ready.
    """
    np.random.seed(cfg["embeddings"]["seed"])
    D = cfg["embeddings"]["dim"]
    words = (
        ["he", "she", "man", "woman", "king", "queen",
         "boy", "girl", "his", "her", "him"]
        + [f"word_{i}" for i in range(500)]
    )
    embeddings = {w: np.random.randn(D).astype(np.float32) for w in words}

    # Inject synthetic gender bias: add gender direction to male/female words
    gender_dir = np.zeros(D)
    # bias lives in first 20 dims so that d=8 and d=16 cut it off
    # This lets us test if our metrics detect "partial" bias survival
    gender_dir[:20] = 1.0
    gender_dir /= np.linalg.norm(gender_dir)

    for w in ["he", "man", "king", "boy", "his", "him"]:
        embeddings[w] = embeddings[w] + 2.0 * gender_dir
    for w in ["she", "woman", "queen", "girl", "her"]:
        embeddings[w] = embeddings[w] - 2.0 * gender_dir

    return embeddings

print("\nStep 1: Loading embeddings...")
embeddings_baseline = load_or_create_toy_data()

# MRL embeddings: same structure but bias slightly redistributed
# (in real experiment: train separately with MRL objective)
embeddings_mrl = {w: v + 0.1 * np.random.randn(*v.shape)
                  for w, v in embeddings_baseline.items()}

# -----------------------------------------------------------------------
# Step 3: Gender pairs
# -----------------------------------------------------------------------
GENDER_PAIRS = [
    ("he", "she"), ("man", "woman"), ("king", "queen"),
    ("boy", "girl"), ("his", "her"), ("him", "her"),
]

# -----------------------------------------------------------------------
# Step 4: RQ1
# -----------------------------------------------------------------------
print("\nStep 4: RQ1 -- Bias Subspace Survival Under Truncation")
print("About to import RQ1...", flush=True)
from src.experiments.rq1_bias_subspace_survival import run_rq1
print("Import done.", flush=True)
# from src.experiments.rq1_bias_subspace_survival import run_rq1
df_rq1 = run_rq1(
    embeddings_baseline, embeddings_mrl,
    GENDER_PAIRS, PREFIX_DIMS, k=K, save_dir=SAVE_DIR
)

# -----------------------------------------------------------------------
# Step 5: RQ3
# -----------------------------------------------------------------------
print("\nStep 5: RQ3 -- Bias Concentration C(d)")
from src.experiments.rq3_bias_concentration import run_rq3

# Simple WEAT sets (career vs family, male vs female)
WEAT_SETS = {
    "X": ["he", "man", "king"],         # male target
    "Y": ["she", "woman", "queen"],     # female target
    "A": [f"word_{i}" for i in range(10)],   # placeholder attribute A
    "B": [f"word_{i}" for i in range(10, 20)],  # placeholder attribute B
}
df_rq3 = run_rq3(
    embeddings_baseline, embeddings_mrl,
    GENDER_PAIRS, PREFIX_DIMS,
    weat_sets=WEAT_SETS, k=K, save_dir=SAVE_DIR
)

# -----------------------------------------------------------------------
# Step 6: Apply Rakshit debiasing
# -----------------------------------------------------------------------
print("\nStep 6: Applying Rakshit debiasing (parity-enforcing)")
from src.bias.bias_subspace import BiasSubspace
bs = BiasSubspace(k=K)
bs.fit(embeddings_baseline, GENDER_PAIRS)
embeddings_debiased = {w: bs.debias(v) for w, v in embeddings_baseline.items()}

# -----------------------------------------------------------------------
# Step 7: RQ2
# -----------------------------------------------------------------------
print("\nStep 7: RQ2 -- Does debiasing hold at prefix level?")
from src.experiments.rq2_debiasing_at_prefix import run_rq2
df_rq2 = run_rq2(
    embeddings_baseline, embeddings_debiased,
    GENDER_PAIRS, PREFIX_DIMS, k=K, save_dir=SAVE_DIR
)

# -----------------------------------------------------------------------
# Step 8: RQ4
# -----------------------------------------------------------------------
print("\nStep 8: RQ4 -- Optimal lambda_2 per prefix dim")
from src.experiments.rq4_adaptive_lambda import run_rq4
df_rq4, optimal_lambda = run_rq4(
    embeddings_baseline, GENDER_PAIRS, PREFIX_DIMS,
    lambda_grid=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    k=K, save_dir=SAVE_DIR
)

print("\n" + "=" * 60)
print("All experiments complete. Results saved to:", SAVE_DIR)
print("=" * 60)
