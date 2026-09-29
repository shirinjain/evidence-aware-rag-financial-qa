"""
Computes mpnet embeddings for all chunks - same model as the original
project (all-mpnet-base-v2, chosen there after a real quality
comparison against alternatives). This corpus is ~1/4 the size
(146,821 vs 594,267 chunks), so at the previously-measured local MPS
rate (~32.5 chunks/sec) this should take ~75 minutes locally - well
under the multi-hour threshold that required routing to Colab before.

Embeds a lightweight contextual text (entity + year + section : raw_text),
same reasoning as build_bm25_index.py's index text - so a company/year
mention in the corpus text itself contributes to the embedding, not just
raw_text alone.

Resumable via checkpointing (same pattern as colab_embed_mpnet.py),
since a 75-minute local run risks the machine sleeping/being interrupted.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "processed" / "child_chunks.parquet"
EMBEDDINGS_PATH = ROOT / "data" / "processed" / "mpnet_embeddings.npy"
CHUNK_IDS_PATH = ROOT / "data" / "processed" / "mpnet_chunk_ids.csv"
CHECKPOINT_PATH = ROOT / "data" / "processed" / "mpnet_embeddings_checkpoint.npy"
CHECKPOINT_PROGRESS_PATH = ROOT / "data" / "processed" / "mpnet_checkpoint_progress.txt"

BATCH_SIZE = 64
CHECKPOINT_EVERY_N_BATCHES = 20


def _build_embedding_text(row) -> str:
    parts = [str(row["entity"]), f"FY{row['fiscal_period']}", str(row["section_title"] or "")]
    return f"{', '.join(p for p in parts if p)}: {row['raw_text']}"


def main():
    df = pd.read_parquet(CHUNKS_PATH)
    texts = df.apply(_build_embedding_text, axis=1).tolist()
    chunk_ids = df["chunk_id"].tolist()
    print(f"{len(texts)} chunks to embed")

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="mps")

    start_idx = 0
    all_embeddings = []
    if CHECKPOINT_PATH.exists() and CHECKPOINT_PROGRESS_PATH.exists():
        all_embeddings = [np.load(CHECKPOINT_PATH)]
        start_idx = int(CHECKPOINT_PROGRESS_PATH.read_text().strip())
        print(f"resuming from checkpoint: {start_idx}/{len(texts)} already done")

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
            print(f"{done}/{len(texts)} embedded | {rate:.1f} chunks/sec | ~{remaining/60:.1f} min remaining", flush=True)

            combined = np.vstack(all_embeddings)
            np.save(CHECKPOINT_PATH, combined)
            CHECKPOINT_PROGRESS_PATH.write_text(str(done))

    final_embeddings = np.vstack(all_embeddings)
    np.save(EMBEDDINGS_PATH, final_embeddings)
    pd.DataFrame({"chunk_id": chunk_ids}).to_csv(CHUNK_IDS_PATH, index=False)
    print("saved:", EMBEDDINGS_PATH, final_embeddings.shape)

    CHECKPOINT_PATH.unlink(missing_ok=True)
    CHECKPOINT_PROGRESS_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
