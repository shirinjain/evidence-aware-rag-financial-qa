"""
Fine-tunes cross-encoder/ms-marco-MiniLM-L-6-v2 on hard-negative data
via RankNetLoss (pairwise ranking) - same setup validated in the
original project's smoke test.

Running LOCALLY (not Colab) since this needs to complete autonomously
without requiring manual Colab steps. Dataset here is much smaller
(83 rows, 874 docs vs the original project's 327/6113) so local CPU
training should be far more tractable time-wise even without GPU.

Loss curve is printed and saved so convergence can be checked directly
before trusting the result - the original project's first attempt
skipped this and shipped a model that turned out not to have learned a
real signal.
"""
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "0"  # need to download the base model once
os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json

from datasets import Dataset
from sentence_transformers.cross_encoder import (
    CrossEncoder,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.losses import RankNetLoss

PAIRS_PATH = ROOT / "data" / "processed" / "finetune_pairwise.jsonl"
OUT_DIR = ROOT / "data" / "processed" / "cross_encoder_finetuned"
FINAL_PATH = ROOT / "data" / "processed" / "cross_encoder_finetuned_final"
LOSS_LOG_PATH = ROOT / "data" / "processed" / "finetune_loss_log.json"


class LossLogger:
    def __init__(self):
        self.logs = []

    def __call__(self, logs, **kwargs):
        pass


def main():
    rows = [json.loads(line) for line in open(PAIRS_PATH)]
    dataset = Dataset.from_list(rows).train_test_split(test_size=0.15, seed=0)
    print(dataset)

    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", num_labels=1)
    # explicit mini_batch_size caps how many (query,doc) pairs go through
    # one forward pass at once - the previous run's escalating step time
    # (82s -> 251s) may have been driven by uncontrolled memory growth
    # from variable per-question doc-list sizes; capping this bounds it
    loss = RankNetLoss(model, mini_batch_size=8)

    args = CrossEncoderTrainingArguments(
        output_dir=str(OUT_DIR),
        num_train_epochs=2,  # time-boxed retry - fewer epochs
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=[],
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        loss=loss,
    )
    result = trainer.train()

    # save the full loss history so convergence can be verified directly
    history = trainer.state.log_history
    with open(LOSS_LOG_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print("\nloss history:")
    for entry in history:
        print(entry)

    model.save_pretrained(str(FINAL_PATH))
    print(f"\nsaved fine-tuned model to {FINAL_PATH}")


if __name__ == "__main__":
    main()
