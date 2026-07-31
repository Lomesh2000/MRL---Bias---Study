from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.losses import MatryoshkaLoss
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.trainer import SentenceTransformerTrainer
from datasets import load_dataset

# Load dataset
dataset = load_dataset("sentence-transformers/stsb", split="train")

# more data
snli = load_dataset("snli", split="train")
mnli = load_dataset("multi_nli", split="train")

dataset = snli + mnli

train_data = [
    {"sentence1": row["sentence1"], "sentence2": row["sentence2"]}
    for row in dataset
]

# Load MPNet base
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Base contrastive loss
base_loss = MultipleNegativesRankingLoss(model)

# Prefix dimensions for Matryoshka
matryoshka_dims = [64, 128, 256, 512, 768]

# MRL loss
loss = MatryoshkaLoss(
    model=model,
    loss=base_loss,
    matryoshka_dims=matryoshka_dims
)

# Training args
args = SentenceTransformerTrainingArguments(
    output_dir="mpnet_mrl_model",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    warmup_ratio=0.1
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_data,
    loss=loss,
)

trainer.train()

model.save("mpnet_mrl_model")