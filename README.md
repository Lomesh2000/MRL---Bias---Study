# MRL Bias Distribution — Experiment Project

> **Paper:** "On the Distribution of Social Bias in Matryoshka Representation Learning"  
> Based on Sir's LaTeX outline + Rakshit et al. (2024) debiasing method.

---

## Research Questions (from your notes)

| RQ | Question | Key Formula |
|----|----------|-------------|
| **RQ1** | Does B^(D) survive prefix truncation Π_d? | Alignment(Π_d(B^(D)), B^(d)) |
| **RQ2** | After DSD, does L2^(d) stay small at prefix d? | L2^(d) = Σ‖P_{B^(d)} f̃^(d)(w)‖² |
| **RQ3** | Does MRL concentrate bias in early dims? | C(d) = Var(b^(d)) / Var(b^(D)) |
| **RQ4** | Should λ₂ scale with prefix dim d? | Grid search λ₂^(d) |

---

## Folder Structure

```
mrl_bias_project/
├── config/config.yaml          ← all hyperparameters here
├── data/
│   ├── gender_pairs/           ← he-she, man-woman word pairs
│   └── weat_wordlists/         ← WEAT attribute/target word sets
├── src/
│   ├── mrl/mrl_objective.py    ← MRL nested loss (NeurIPS 2022)
│   ├── bias/
│   │   ├── bias_subspace.py    ← B = span(v1..vk) via PCA
│   │   └── bias_metrics.py     ← WEAT, C(d), regime classifier
│   ├── embeddings/
│   │   └── train_word2vec.py   ← baseline + MRL Word2Vec
│   └── experiments/
│       ├── rq1_*.py            ← subspace survival
│       ├── rq2_*.py            ← debiasing at prefix
│       ├── rq3_*.py            ← bias concentration C(d)
│       └── rq4_*.py            ← adaptive lambda
├── results/
│   ├── bias_scores/            ← CSV files with C(d), WEAT, etc.
│   └── figures/                ← plots for paper
└── run_experiments.py          ← MASTER RUNNER
```

---

## Step-by-Step Start (following Sir's LaTeX Section 4)

### Phase 1 — Word Embeddings (controlled setting)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full pipeline with toy data (sanity check)
python run_experiments.py

# 3. Check results
ls results/figures/
ls results/bias_scores/
```

### Phase 2 — Real Data

```python
# Download small Reddit-L2 subset
from datasets import load_dataset
ds = load_dataset("reddit", split="train[:50000]")
sentences = [s.split() for s in ds["body"] if s]

# Train real Word2Vec
from src.embeddings.train_word2vec import train_baseline_word2vec
model = train_baseline_word2vec(sentences, dim=300, epochs=10)
```

### Phase 3 — LLM Embeddings (after word embeddings baseline)

Use a model that natively supports MRL:
```python
from sentence_transformers import SentenceTransformer
# nomic-embed-text supports MRL with truncate_dim
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
emb = model.encode(["doctor", "nurse"], convert_to_numpy=True,
                   truncate_dim=64)  # prefix dim d=64
```

---

## Key Math Summary

**Bias Subspace:**
```
s_i = f(w_i^+) - f(w_i^-)
C = (1/n) Σ s_i s_i^T
B = span(v_1, ..., v_k)   [top-k eigenvectors of C]
P_B = Σ v_i v_i^T
```

**Bias Concentration C(d):**
```
C(d) = Var(b^(d)(w)) / Var(b^(D)(w))
C(d) >> 1 at small d → Amplification (MRL packs bias early)
C(d) << 1 at small d → Dilution (MRL separates bias from task)
C(d) ≈ flat           → Redistribution (bias moves, doesn't vanish)
```
