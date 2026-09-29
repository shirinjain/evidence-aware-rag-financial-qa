"""
Run this in Google Colab (GPU runtime) - local MPS embedding stabilized
at ~18-19 chunks/sec (~2 hour ETA for all 146,821 chunks), the same
range that justified moving to Colab in the original project.

Setup in Colab, before running:
  1. Runtime -> Change runtime type -> T4 GPU
  2. Upload data/processed/child_chunks.parquet to your Google Drive,
     e.g. MyDrive/evidence_aware_rag_2/child_chunks.parquet

Paste each ### CELL ### block below into its own Colab cell, in order.
Output is written to your Drive (embeddings array is large, so this
avoids the browser download widget).

Checkpointing: saves progress every CHECKPOINT_EVERY batches, so a
dropped Colab session resumes instead of restarting.
"""

### CELL 1 - install + mount Drive ###
"""
!pip install -q sentence-transformers pandas pyarrow

from google.colab import drive
drive.mount('/content/drive')
"""

### CELL 2 - config: EDIT THIS PATH to match where you put the file in Drive ###
"""
DRIVE_DIR = '/content/drive/MyDrive/evidence_aware_rag_2'
INPUT_PARQUET = f'{DRIVE_DIR}/child_chunks.parquet'
OUTPUT_EMBEDDINGS = f'{DRIVE_DIR}/mpnet_embeddings.npy'
OUTPUT_CHUNK_IDS = f'{DRIVE_DIR}/mpnet_chunk_ids.csv'
CHECKPOINT_PATH = f'{DRIVE_DIR}/mpnet_embeddings_checkpoint.npy'
CHECKPOINT_PROGRESS_PATH = f'{DRIVE_DIR}/mpnet_checkpoint_progress.txt'

BATCH_SIZE = 256
CHECKPOINT_EVERY_N_BATCHES = 20
"""

### CELL 3 - load model + data ###
"""
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import os, time

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cuda')
print('device:', model.device)

df = pd.read_parquet(INPUT_PARQUET)

def build_embedding_text(row):
    parts = [str(row['entity']), 'FY' + str(row['fiscal_period']), str(row['section_title'] or '')]
    return ', '.join(p for p in parts if p) + ': ' + row['raw_text']

texts = df.apply(build_embedding_text, axis=1).tolist()
chunk_ids = df['chunk_id'].tolist()
print(f'{len(texts)} chunks to embed')
"""

### CELL 4 - embed with checkpointing (resumable) ###
"""
start_idx = 0
all_embeddings = []
if os.path.exists(CHECKPOINT_PATH) and os.path.exists(CHECKPOINT_PROGRESS_PATH):
    all_embeddings = [np.load(CHECKPOINT_PATH)]
    start_idx = int(open(CHECKPOINT_PROGRESS_PATH).read().strip())
    print(f'resuming from checkpoint: {start_idx}/{len(texts)} already done')

t_start = time.time()
for batch_num, i in enumerate(range(start_idx, len(texts), BATCH_SIZE)):
    batch = texts[i:i + BATCH_SIZE]
    embs = model.encode(batch, show_progress_bar=False)
    all_embeddings.append(embs)

    done = i + len(batch)
    if batch_num % CHECKPOINT_EVERY_N_BATCHES == 0 or done >= len(texts):
        elapsed = time.time() - t_start
        rate = (done - start_idx) / elapsed if elapsed > 0 else 0
        remaining = (len(texts) - done) / rate if rate > 0 else 0
        print(f'{done}/{len(texts)} embedded | {rate:.1f} chunks/sec | ~{remaining/60:.1f} min remaining')

        combined = np.vstack(all_embeddings)
        np.save(CHECKPOINT_PATH, combined)
        with open(CHECKPOINT_PROGRESS_PATH, 'w') as f:
            f.write(str(done))

print('done embedding')
"""

### CELL 5 - final save ###
"""
final_embeddings = np.vstack(all_embeddings)
np.save(OUTPUT_EMBEDDINGS, final_embeddings)
pd.DataFrame({'chunk_id': chunk_ids}).to_csv(OUTPUT_CHUNK_IDS, index=False)
print('saved:', OUTPUT_EMBEDDINGS, final_embeddings.shape)
print('saved:', OUTPUT_CHUNK_IDS)

import os
if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)
if os.path.exists(CHECKPOINT_PROGRESS_PATH):
    os.remove(CHECKPOINT_PROGRESS_PATH)
"""
