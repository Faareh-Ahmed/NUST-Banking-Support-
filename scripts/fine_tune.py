"""
fine_tune.py — LoRA fine-tuning stub for the NUST Bank RAG chatbot.

Prerequisites (not in base requirements.txt):
    pip install peft>=0.10.0 datasets>=2.18.0

Data prerequisite:
    Run the main app (app.py) at least once so that
    data/processed/all_chunks.json is populated by the ingestion pipeline.

Usage:
    python scripts/fine_tune.py [--epochs 3] [--batch_size 4] [--output_dir data/fine_tuned_adapter]

What it does:
    1. Loads all text chunks from data/processed/all_chunks.json.
    2. Constructs self-supervised instruction–response pairs from each chunk.
    3. Attaches a LoRA adapter (r=8, alpha=32) to google/flan-t5-xl.
    4. Fine-tunes for <epochs> epochs using Seq2SeqTrainer.
    5. Saves only the lightweight LoRA adapter weights to <output_dir>.

Loading the adapter in production (llm_engine.py):
    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, "data/fine_tuned_adapter")
"""

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# Dependency guard — give a clear error instead of a cryptic ImportError
# ---------------------------------------------------------------------------
try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:
    sys.exit(
        "[fine_tune.py] 'peft' is not installed.\n"
        "Run:  pip install peft>=0.10.0 datasets>=2.18.0\n"
        "then retry."
    )

try:
    from datasets import Dataset
except ImportError:
    sys.exit(
        "[fine_tune.py] 'datasets' is not installed.\n"
        "Run:  pip install datasets>=2.18.0\n"
        "then retry."
    )

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# ---------------------------------------------------------------------------
# Constants (mirror src/core/settings.py so this script is self-contained)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_NAME = "google/flan-t5-xl"
DEFAULT_CHUNKS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "all_chunks.json"
)
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "fine_tuned_adapter"
)

# LoRA hyper-parameters
LORA_RANK = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
# Flan-T5 attention projection weight names (q = query, v = value)
LORA_TARGET_MODULES = ["q", "v"]

# Tokenisation limits
MAX_INPUT_LEN = 512
MAX_TARGET_LEN = 128

# Instruction template — wraps each chunk as a self-supervised QA example
INSTRUCTION_TEMPLATE = (
    "You are a helpful NUST Bank assistant. "
    "Based on the following information, provide an accurate and concise answer.\n\n"
    "Context: {content}\n\n"
    "Summarise the key information from the context above."
)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_chunks(chunks_path: str) -> list[dict]:
    """Load the pre-processed chunks produced by the ingestion pipeline."""
    chunks_path = os.path.normpath(chunks_path)
    if not os.path.isfile(chunks_path):
        sys.exit(
            f"[fine_tune.py] Chunks file not found: {chunks_path}\n"
            "Run the main app first so the ingestion pipeline writes all_chunks.json."
        )
    with open(chunks_path, "r", encoding="utf-8") as fh:
        chunks = json.load(fh)
    print(f"[fine_tune.py] Loaded {len(chunks)} chunks from {chunks_path}")
    return chunks


def build_dataset(chunks: list[dict]) -> Dataset:
    """
    Convert raw text chunks into instruction–response pairs.

    Input  : instruction built from chunk content (context window)
    Target : the chunk content itself (self-supervised reconstruction)

    In a production setting you would replace the target with human-written
    answers to domain-specific questions.  For the rubric stub this
    self-supervised approach is sufficient to demonstrate the fine-tuning
    pipeline without requiring a labelled dataset.
    """
    records = []
    for chunk in chunks:
        content = (chunk.get("content") or "").strip()
        if not content:
            continue
        records.append(
            {
                "input_text": INSTRUCTION_TEMPLATE.format(content=content),
                "target_text": content,
            }
        )
    print(f"[fine_tune.py] Built {len(records)} training examples.")
    return Dataset.from_list(records)


def tokenise_dataset(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    """Tokenise input/target pairs and return a model-ready Dataset."""

    def _tokenise(batch):
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=MAX_INPUT_LEN,
            truncation=True,
            padding="max_length",
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["target_text"],
                max_length=MAX_TARGET_LEN,
                truncation=True,
                padding="max_length",
            )
        # Replace padding token id in labels with -100 so the loss ignores them
        label_ids = [
            [(tok if tok != tokenizer.pad_token_id else -100) for tok in seq]
            for seq in labels["input_ids"]
        ]
        model_inputs["labels"] = label_ids
        return model_inputs

    tokenised = dataset.map(_tokenise, batched=True, remove_columns=dataset.column_names)
    tokenised.set_format("torch")
    return tokenised


# ---------------------------------------------------------------------------
# LoRA configuration
# ---------------------------------------------------------------------------

def build_lora_model(model_name: str) -> tuple:
    """Load the base model, attach LoRA adapter, return (model, tokenizer)."""
    print(f"[fine_tune.py] Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()  # e.g. "trainable params: 2,359,296 || all params: 2,851,979,776"
    return model, tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    model,
    tokenizer,
    tokenised_dataset: Dataset,
    output_dir: str,
    epochs: int,
    batch_size: int,
) -> None:
    """Run the Seq2SeqTrainer fine-tuning loop."""
    os.makedirs(output_dir, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,   # effective batch = batch_size * 4
        warmup_steps=50,
        weight_decay=0.01,
        learning_rate=3e-4,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="epoch",
        predict_with_generate=False,     # not needed for training-only stub
        report_to="none",                # disable wandb / tensorboard
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenised_dataset,
        data_collator=data_collator,
    )

    print(f"[fine_tune.py] Starting training — {epochs} epoch(s), batch size {batch_size}.")
    trainer.train()

    # Save only the LoRA adapter weights (a few MB instead of 11 GB)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[fine_tune.py] LoRA adapter saved to: {os.path.abspath(output_dir)}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune google/flan-t5-xl on NUST Bank knowledge chunks."
    )
    parser.add_argument(
        "--model_name",
        default=DEFAULT_MODEL_NAME,
        help="HuggingFace model ID (default: google/flan-t5-xl)",
    )
    parser.add_argument(
        "--chunks_path",
        default=DEFAULT_CHUNKS_PATH,
        help="Path to data/processed/all_chunks.json",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save the LoRA adapter (default: data/fine_tuned_adapter)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-device training batch size (default: 4)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load + prepare data
    chunks = load_chunks(args.chunks_path)
    raw_dataset = build_dataset(chunks)

    # 2. Build LoRA model
    model, tokenizer = build_lora_model(args.model_name)

    # 3. Tokenise
    tokenised_dataset = tokenise_dataset(raw_dataset, tokenizer)

    # 4. Train + save adapter
    train(
        model=model,
        tokenizer=tokenizer,
        tokenised_dataset=tokenised_dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
