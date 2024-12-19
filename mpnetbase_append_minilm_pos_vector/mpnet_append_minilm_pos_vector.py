import logging
import json
import os
from datetime import datetime

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from transformers import AutoTokenizer
from sentence_transformers.losses import CoSENTLoss
from typing import Literal
import spacy
import torch
import torch.nn as nn

# Set the log level to INFO to get more information
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)

nlp = spacy.load("en_core_web_sm")

modelA = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model_name = "sentence-transformers/all-mpnet-base-v2"
train_batch_size = 16
num_epochs = 4
output_dir = (
    "output/training_stsbenchmark_"
    + model_name.replace("/", "-")
    + "-"
    + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)


class EnsembleEmbeddingsSentenceTransformer(SentenceTransformer):
    def __init__(self, model):
        super().__init__(model)

    def encode(
        sentences: list[str] | str,
        prompt_name: str | None = None,
        prompt: str | None = None,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        output_value: Literal["sentence_embedding", "token_embeddings"]
        | None = "sentence_embedding",
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
        convert_to_numpy: Literal[False] = True,
        convert_to_tensor: Literal[False] = False,
        device: str = None,
        normalize_embeddings: bool = False,
        **kwargs,
    ) -> list[torch.Tensor]:
        doc = nlp(sentences)

        processed = " ".join([str(token.pos_) for token in doc])

        embeddingAProcessed = modelA.encode(sentences, convert_to_tensor=True)
        posA = modelA.encode(processed, convert_to_tensor=True)

        embedding = torch.cat(
            (embeddingAProcessed, posA),
            dim=0,
        )

        return [embedding]


class DecayMeanPooling(nn.Module):
    def __init__(self, dimension: int, decay: float = 0.95) -> None:
        super(DecayMeanPooling, self).__init__()
        self.dimension = dimension
        self.decay = decay

    def forward(
        self, features: dict[str, torch.Tensor], **kwargs
    ) -> dict[str, torch.Tensor]:
        token_embeddings = features["token_embeddings"]
        attention_mask = features["attention_mask"].unsqueeze(-1)

        # Apply the attention mask to filter away padding tokens
        token_embeddings = token_embeddings * attention_mask
        # Calculate mean of token embeddings
        sentence_embeddings = token_embeddings.sum(1) / attention_mask.sum(1)
        # Apply exponential decay
        importance_per_dim = self.decay ** torch.arange(
            sentence_embeddings.size(1), device=sentence_embeddings.device
        )
        features["sentence_embedding"] = sentence_embeddings * importance_per_dim
        return features

    def get_config_dict(self) -> dict[str, float]:
        return {"dimension": self.dimension, "decay": self.decay}

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def save(self, save_dir: str, **kwargs) -> None:
        with open(os.path.join(save_dir, "config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut, indent=4)

    def load(load_dir: str, **kwargs) -> "DecayMeanPooling":
        with open(os.path.join(load_dir, "config.json")) as fIn:
            config = json.load(fIn)

        return DecayMeanPooling(**config)


# 1. Here we define our SentenceTransformer model.
tokenizer = AutoTokenizer.from_pretrained(model_name)
transformer = EnsembleEmbeddingsSentenceTransformer(model_name)
pooling = decay_mean_pooling = DecayMeanPooling(1024, decay=0.99)
normalize = models.Normalize()
model = SentenceTransformer(modules=[transformer, pooling, normalize])

loss = CoSENTLoss(model)

# 2. Load the STSB dataset: https://huggingface.co/datasets/sentence-transformers/stsb
train_dataset = load_dataset("sentence-transformers/stsb", split="train")
eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
test_dataset = load_dataset("sentence-transformers/stsb", split="test")
logging.info(train_dataset)

# 3. Define our training loss
# CosineSimilarityLoss (https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) needs two text columns and one
# similarity score column (between 0 and 1)
train_loss = losses.CosineSimilarityLoss(model=model)
# train_loss = losses.CoSENTLoss(model=model)

# 4. Define an evaluator for use during training. This is useful to keep track of alongside the evaluation loss.
dev_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=eval_dataset["sentence1"],
    sentences2=eval_dataset["sentence2"],
    scores=eval_dataset["score"],
    main_similarity=SimilarityFunction.COSINE,
    name="sts-dev",
)

# 5. Define the training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=output_dir,
    # Optional training parameters:
    num_train_epochs=num_epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=100,
    run_name="sts",  # Will be used in W&B if `wandb` is installed
)

# 6. Create the trainer & start training
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=train_loss,
    evaluator=dev_evaluator,
)
trainer.train()

# 7. Evaluate the model performance on the STS Benchmark test dataset
test_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=test_dataset["sentence1"],
    sentences2=test_dataset["sentence2"],
    scores=test_dataset["score"],
    main_similarity=SimilarityFunction.COSINE,
    name="sts-test",
)
test_evaluator(model)

model.save_pretrained("models/sentence-transformers-embedding-ensemble/final")
