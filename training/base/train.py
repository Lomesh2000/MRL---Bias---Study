from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.trainer import SentenceTransformerTrainer
from datasets import load_dataset

# Load STS dataset
dataset = load_dataset("sentence-transformers/stsb", split="train")

# Convert dataset format
train_data = [
    {"sentence1": row["sentence1"], "sentence2": row["sentence2"]}
    for row in dataset
]

# Load MPNet model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Standard contrastive loss
loss = MultipleNegativesRankingLoss(model)

# Training configuration
args = SentenceTransformerTrainingArguments(
    output_dir="mpnet_baseline_model",
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

model.save("mpnet_baseline_model")