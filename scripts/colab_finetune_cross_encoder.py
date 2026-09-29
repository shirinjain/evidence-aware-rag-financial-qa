"""
Run this in Google Colab (GPU runtime) - local MPS fine-tuning showed a
repeated escalating per-step slowdown (82s->251s in one attempt, then
2s->65s+ in a retry with tighter memory controls) across two attempts,
not fixed by hyperparameter changes - looks like genuine hardware/MPS
degradation under sustained load, same category of issue that
justified routing the embedding step to Colab earlier this project.

Setup in Colab, before running:
  1. Runtime -> Change runtime type -> T4 GPU
  2. Upload data/processed/finetune_pairwise.jsonl to your Google
     Drive, e.g. MyDrive/evidence_aware_rag_2/finetune_pairwise.jsonl

Paste each ### CELL ### block below into its own Colab cell, in order.
"""

### CELL 1 - install + mount Drive ###
"""
!pip install -q sentence-transformers datasets

from google.colab import drive
drive.mount('/content/drive')
"""

### CELL 2 - config ###
"""
DRIVE_DIR = '/content/drive/MyDrive/evidence_aware_rag_2'
PAIRS_PATH = f'{DRIVE_DIR}/finetune_pairwise.jsonl'
OUTPUT_DIR = f'{DRIVE_DIR}/cross_encoder_finetuned'
"""

### CELL 3 - load data + model, train ###
"""
import json
from datasets import Dataset
from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.losses import RankNetLoss

rows = [json.loads(line) for line in open(PAIRS_PATH)]
dataset = Dataset.from_list(rows).train_test_split(test_size=0.15, seed=0)
print(dataset)

model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', num_labels=1, device='cuda')
print('device:', model.model.device)
loss = RankNetLoss(model)

# NOTE: save_total_limit removed and load_best_model_at_end=False -
# a previous run's loss curve looked perfectly normal (steadily
# decreasing train/eval loss) but the resulting "best by eval_loss"
# checkpoint still made downstream retrieval WORSE, with inverted
# score preferences on spot-check. Loss alone isn't a reliable
# selector - keeping every epoch's checkpoint so they can each be
# evaluated on the actual downstream reranking task afterward, and
# picking whichever one genuinely helps, not whichever had lowest loss.
args = CrossEncoderTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    eval_strategy='epoch',
    save_strategy='epoch',
    logging_steps=5,
    load_best_model_at_end=False,
    report_to=[],
)

trainer = CrossEncoderTrainer(model=model, args=args, train_dataset=dataset['train'], eval_dataset=dataset['test'], loss=loss)
trainer.train()

for entry in trainer.state.log_history:
    print(entry)
"""

### CELL 4 - zip ALL epoch checkpoints for download ###
"""
import shutil, os
print('checkpoints saved:', os.listdir(OUTPUT_DIR))
shutil.make_archive(f'{DRIVE_DIR}/cross_encoder_all_checkpoints', 'zip', OUTPUT_DIR)
print('saved:', f'{DRIVE_DIR}/cross_encoder_all_checkpoints.zip')
"""
