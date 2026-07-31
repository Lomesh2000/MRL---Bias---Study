import matplotlib
matplotlib.use("Agg") # For headless environments
import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from datasets import load_dataset

from src.bias.deep_soft_debias import DeepSoftDebias

# -----------------------------------------------------------------------
# Step 0: Config & Output Setup
# -----------------------------------------------------------------------
PREFIX_DIMS = [64, 128, 256, 384, 512, 768]
K = 10
SAVE_DIR = "results_dsd_comprehensive/"

os.makedirs(os.path.join(SAVE_DIR, "figures"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "tables"), exist_ok=True)

print("=" * 60)
print("Evaluating DSD: MRL vs Non-MRL (Bias & Utility)")
print("=" * 60)

# -----------------------------------------------------------------------
# Step 1: Prepare Vocabulary Data
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

MODEL_MRL = "tomaarsen/mpnet-base-nli-matryoshka"
MODEL_NO_MRL = "tomaarsen/mpnet-base-nli"

def get_vocab_embeddings(model_name, vocab):
    print(f"Loading {model_name} for vocab...", flush=True)
    model = SentenceTransformer(model_name, trust_remote_code=True, device='cpu')
    embeddings_matrix = model.encode(vocab, convert_to_numpy=True)
    del model
    gc.collect() 
    return {w: v for w, v in zip(vocab, embeddings_matrix)}

print("Generating vocab embeddings...")
vocab_embs_non_mrl = get_vocab_embeddings(MODEL_NO_MRL, VOCAB_LIST)
vocab_embs_mrl = get_vocab_embeddings(MODEL_MRL, VOCAB_LIST)

actual_K = min(K, len(GENDER_PAIRS))

# -----------------------------------------------------------------------
# Step 2: Prepare STS-B Data (For Semantic Utility)
# -----------------------------------------------------------------------
print("\nLoading STS-B dataset for utility evaluation...")
stsb = load_dataset("glue", "stsb", split="validation")
sts_sentences1 = stsb['sentence1']
sts_sentences2 = stsb['sentence2']
sts_human_scores = stsb['label']

def get_sts_embeddings(model_name, sentences1, sentences2):
    print(f"Loading {model_name} for STS-B...", flush=True)
    model = SentenceTransformer(model_name, trust_remote_code=True, device='cpu')
    embs1 = model.encode(sentences1, convert_to_numpy=True)
    embs2 = model.encode(sentences2, convert_to_numpy=True)
    del model
    gc.collect()
    return embs1, embs2

print("Generating STS-B embeddings...")
sts_e1_non_mrl, sts_e2_non_mrl = get_sts_embeddings(MODEL_NO_MRL, sts_sentences1, sts_sentences2)
sts_e1_mrl, sts_e2_mrl = get_sts_embeddings(MODEL_MRL, sts_sentences1, sts_sentences2)

# -----------------------------------------------------------------------
# Step 3: Define Metrics & Helpers
# -----------------------------------------------------------------------
MALE_WORDS = [pair[0] for pair in GENDER_PAIRS if pair[0] in VOCAB_LIST]
FEMALE_WORDS = [pair[1] for pair in GENDER_PAIRS if pair[1] in VOCAB_LIST]
TARGET_WORDS = [w for w in VOCAB_LIST if w not in MALE_WORDS and w not in FEMALE_WORDS]

def compute_mac_bias(emb_dict, male_words, female_words, target_words):
    if not target_words: return 0.0
    M_embs = np.array([emb_dict[w] for w in male_words])
    F_embs = np.array([emb_dict[w] for w in female_words])
    T_embs = np.array([emb_dict[w] for w in target_words])
    
    sim_to_male = cosine_similarity(T_embs, M_embs).mean(axis=1)
    sim_to_female = cosine_similarity(T_embs, F_embs).mean(axis=1)
    return np.abs(sim_to_male - sim_to_female).mean()

def evaluate_sts(embs1, embs2, labels):
    cos_sims = [cosine_similarity([e1], [e2])[0][0] for e1, e2 in zip(embs1, embs2)]
    correlation, _ = spearmanr(cos_sims, labels)
    return correlation

def apply_dsd_to_array(dsd_model, array_data):
    """Helper to apply a dictionary-based DSD transform to a numpy array."""
    dummy_dict = {i: vec for i, vec in enumerate(array_data)}
    transformed_dict = dsd_model.transform(dummy_dict)
    return np.array([transformed_dict[i] for i in range(len(array_data))])

# -----------------------------------------------------------------------
# Step 4: Main Evaluation Loop (With Lambda Sweep)
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("Applying DSD: Sweeping Dimensions and Lambda_2")
print("=" * 60)

LAMBDAS = [0.1, 0.5, 0.8, 1.0] # Keeping 1.0 as the baseline failure point
results = []

for d in PREFIX_DIMS:
    print(f"\nProcessing Dimension: {d} ...", flush=True)
    
    # --- A. Slice Raw Embeddings FIRST ---
    v_non_mrl_d = {w: v[:d] for w, v in vocab_embs_non_mrl.items()}
    v_mrl_d = {w: v[:d] for w, v in vocab_embs_mrl.items()}
    
    s1_non_mrl_d = sts_e1_non_mrl[:, :d]
    s2_non_mrl_d = sts_e2_non_mrl[:, :d]
    s1_mrl_d = sts_e1_mrl[:, :d]
    s2_mrl_d = sts_e2_mrl[:, :d]
    
    # --- B. Compute BEFORE Metrics (Only once per dimension) ---
    mac_non_mrl_pre = compute_mac_bias(v_non_mrl_d, MALE_WORDS, FEMALE_WORDS, TARGET_WORDS)
    mac_mrl_pre = compute_mac_bias(v_mrl_d, MALE_WORDS, FEMALE_WORDS, TARGET_WORDS)
    sts_non_mrl_pre = evaluate_sts(s1_non_mrl_d, s2_non_mrl_d, sts_human_scores)
    sts_mrl_pre = evaluate_sts(s1_mrl_d, s2_mrl_d, sts_human_scores)
    
    for l2 in LAMBDAS:
        print(f"  -> Testing lambda_2 = {l2}", flush=True)
        
        # --- C. Fit DSD on Vocab Embeddings ---
        dsd_non_mrl = DeepSoftDebias(lambda_2=l2, k=actual_K, hidden_dim=d, lr=1e-3, epochs=100)
        dsd_non_mrl.fit(v_non_mrl_d, GENDER_PAIRS)
        v_non_mrl_debiased = dsd_non_mrl.transform(v_non_mrl_d)
        
        dsd_mrl = DeepSoftDebias(lambda_2=l2, k=actual_K, hidden_dim=d, lr=1e-3, epochs=100)
        dsd_mrl.fit(v_mrl_d, GENDER_PAIRS)
        v_mrl_debiased = dsd_mrl.transform(v_mrl_d)
        
        # --- D. Apply DSD to STS Embeddings ---
        s1_non_mrl_debiased = apply_dsd_to_array(dsd_non_mrl, s1_non_mrl_d)
        s2_non_mrl_debiased = apply_dsd_to_array(dsd_non_mrl, s2_non_mrl_d)
        
        s1_mrl_debiased = apply_dsd_to_array(dsd_mrl, s1_mrl_d)
        s2_mrl_debiased = apply_dsd_to_array(dsd_mrl, s2_mrl_d)
        
        # --- E. Compute AFTER Metrics ---
        mac_non_mrl_post = compute_mac_bias(v_non_mrl_debiased, MALE_WORDS, FEMALE_WORDS, TARGET_WORDS)
        mac_mrl_post = compute_mac_bias(v_mrl_debiased, MALE_WORDS, FEMALE_WORDS, TARGET_WORDS)
        
        sts_non_mrl_post = evaluate_sts(s1_non_mrl_debiased, s2_non_mrl_debiased, sts_human_scores)
        sts_mrl_post = evaluate_sts(s1_mrl_debiased, s2_mrl_debiased, sts_human_scores)
        
        # --- F. Store ---
        results.append({
            "Prefix_Dim": d,
            "Lambda_2": l2,
            "Non-MRL_MAC_Before": mac_non_mrl_pre,
            "Non-MRL_MAC_After": mac_non_mrl_post,
            "MRL_MAC_Before": mac_mrl_pre,
            "MRL_MAC_After": mac_mrl_post,
            "Non-MRL_STS_Before": sts_non_mrl_pre,
            "Non-MRL_STS_After": sts_non_mrl_post,
            "MRL_STS_Before": sts_mrl_pre,
            "MRL_STS_After": sts_mrl_post
        })

# -----------------------------------------------------------------------
# Step 5: Save Results (Simplified for multi-lambda output)
# -----------------------------------------------------------------------
df = pd.DataFrame(results)
csv_path = os.path.join(SAVE_DIR, "tables", "dsd_lambda_sweep_results.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved numerical results to -> {csv_path}")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

print("\nSnapshot of high-dimension (d=768) behavior across lambdas:")
print(df[df["Prefix_Dim"] == 768][["Lambda_2", "MRL_MAC_After", "MRL_STS_After"]].to_string(index=False))

print("Done!")
# Plot 1: Non-MRL Bias
axes[0, 0].plot(df["Prefix_Dim"], df["Non-MRL_MAC_Before"], marker="o", linestyle=":", label="Before DSD")
axes[0, 0].plot(df["Prefix_Dim"], df["Non-MRL_MAC_After"], marker="s", label="After DSD")
axes[0, 0].set_title("Non-MRL Model: MAC Bias (Lower is better)")
axes[0, 0].set_xlabel("Prefix Dimension d")
axes[0, 0].set_ylabel("MAC Score")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: MRL Bias
axes[0, 1].plot(df["Prefix_Dim"], df["MRL_MAC_Before"], marker="o", linestyle=":", label="Before DSD")
axes[0, 1].plot(df["Prefix_Dim"], df["MRL_MAC_After"], marker="s", label="After DSD")
axes[0, 1].set_title("MRL Model: MAC Bias (Lower is better)")
axes[0, 1].set_xlabel("Prefix Dimension d")
axes[0, 1].set_ylabel("MAC Score")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Non-MRL Utility
axes[1, 0].plot(df["Prefix_Dim"], df["Non-MRL_STS_Before"], marker="o", linestyle=":", color="green", label="Before DSD")
axes[1, 0].plot(df["Prefix_Dim"], df["Non-MRL_STS_After"], marker="s", color="purple", label="After DSD")
axes[1, 0].set_title("Non-MRL Model: Semantic Utility (Higher is better)")
axes[1, 0].set_xlabel("Prefix Dimension d")
axes[1, 0].set_ylabel("STS-B Spearman Corr")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: MRL Utility
axes[1, 1].plot(df["Prefix_Dim"], df["MRL_STS_Before"], marker="o", linestyle=":", color="green", label="Before DSD")
axes[1, 1].plot(df["Prefix_Dim"], df["MRL_STS_After"], marker="s", color="purple", label="After DSD")
axes[1, 1].set_title("MRL Model: Semantic Utility (Higher is better)")
axes[1, 1].set_xlabel("Prefix Dimension d")
axes[1, 1].set_ylabel("STS-B Spearman Corr")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(SAVE_DIR, "figures", "dsd_bias_utility_tradeoff.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Saved comprehensive plot to -> {plot_path}")

print("Done!")