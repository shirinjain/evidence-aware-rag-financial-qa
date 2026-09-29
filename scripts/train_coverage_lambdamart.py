"""
Custom coverage-aware LambdaMART, implemented from scratch (see
conversation: LightGBM/XGBoost's macOS wheels need libomp, which isn't
installed here and would require bootstrapping Homebrew - a bigger
environment change than this warrants, so this uses sklearn's
DecisionTreeRegressor as the weak learner in a manual gradient-boosting
loop instead. This is a legitimate implementation, not a workaround:
it needs a custom objective anyway (Δcoverage@k instead of ΔNDCG),
which off-the-shelf libraries don't expose a clean hook for either.

One simplification vs. textbook LambdaMART, stated plainly: leaf
values are NOT computed via the Newton step (sum(grad)/sum(hess) per
leaf) - trees are fit directly to the raw lambda gradients via least
squares. This is standard first-order gradient boosting rather than
full Newton boosting; it converges slightly slower but is simpler to
verify correct, and was judged reasonable given the training set size
(~150-200 groups) doesn't warrant the extra complexity.

The exact, non-approximated Δcoverage@k rule this relies on: coverage@k
is a SET-membership metric (a hop is "covered" iff at least one of its
chunks is in the top-k), so swapping any two candidates that are BOTH
inside the current top-k, or BOTH outside, provably cannot change
coverage@k at all - only a pair straddling the rank-k boundary (one
inside, one outside) can. So only those boundary-crossing pairs get a
nonzero lambda; this is exact, not a heuristic simplification, and
keeps the per-round cost at O(k * (n-k)) pairs per query instead of
O(n^2).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from extract_lambdamart_features import FEATURE_COLS
from eval.coverage_metrics import coverage_at_k
from eval.metrics import evaluate_ranking

K = 5           # optimize for coverage@5 - matches the headline P@5/NDCG@5 metrics used throughout this project
N_ROUNDS = 100
LEARNING_RATE = 0.15
MAX_DEPTH = 3
SIGMA = 1.0

# warm-start variant: a smaller, more transferable feature set (avoids
# raw bm25/dense scores, which encode company/document-length-specific
# quirks a tree can latch onto with only ~26 training examples) plus
# much stronger regularization, learning a small correction ON TOP of
# the cross-encoder score (initialized to it, not zero) rather than
# trying to relearn the whole ranking from scratch - see conversation:
# this targets the diagnosed failure mode directly (trees memorizing
# company-specific score ranges instead of a transferable rule).
WARM_START_FEATURE_COLS = ["ce_score_norm", "is_value_dup_of_higher_ranked", "entity_match", "year_match"]
WARM_START_COL = "ce_score_norm"
WARM_START_MAX_DEPTH = 2
WARM_START_MIN_SAMPLES_LEAF = 25
WARM_START_LEARNING_RATE = 0.05

# no-cross-encoder variant: warm-starts from stage-1 fusion instead of
# the cross-encoder, using only pre-cross-encoder signals + the
# duplicate-detection feature. Since none of these have any fine-tuning
# history, this sidesteps the CE fine-tuning-data contamination issue
# entirely - no need to separate "clean" vs "contaminated" test
# questions, since there's nothing here for any of them to have
# memorized in the first place.
NO_CE_FEATURE_COLS = ["bm25_norm", "dense_norm", "cc_score", "is_value_dup_of_higher_ranked", "entity_match", "year_match", "is_table_row", "lexical_overlap"]
NO_CE_WARM_START_COL = "cc_score"


def load_groups(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def prep(groups, feature_cols=None, warm_start_col=None):
    feature_cols = feature_cols or FEATURE_COLS
    warm_start_col = warm_start_col or WARM_START_COL
    dfs = [g["df"] for g in groups]
    all_df = pd.concat(dfs, ignore_index=True)
    X = all_df[feature_cols].to_numpy(dtype=float)
    hop_labels_flat = all_df["hop_label"].tolist()
    labels_flat = all_df["label"].to_numpy(dtype=float)
    warm_start_scores = all_df[warm_start_col].to_numpy(dtype=float)
    boundaries = []
    num_hops_per_group = []
    start = 0
    for g in groups:
        n = len(g["df"])
        boundaries.append((start, start + n))
        num_hops_per_group.append(len(g["hop_groups"]))
        start += n
    return X, boundaries, hop_labels_flat, num_hops_per_group, warm_start_scores, labels_flat


def compute_lambdas_for_group(scores: np.ndarray, hop_labels: list, num_hops: int, k: int, sigma: float = SIGMA):
    n = len(scores)
    if num_hops == 0 or n == 0:
        return np.zeros(n), np.zeros(n)
    k_eff = min(k, n)
    order = np.argsort(-scores)
    top_k_idx = set(order[:k_eff].tolist())
    outside = [i for i in range(n) if i not in top_k_idx]

    def hops_covered(idx_set):
        return {hop_labels[i] for i in idx_set if hop_labels[i] is not None}

    current_coverage = len(hops_covered(top_k_idx)) / num_hops

    grad = np.zeros(n)
    hess = np.zeros(n)
    for a in top_k_idx:
        for b in outside:
            new_idx_set = (top_k_idx - {a}) | {b}
            new_coverage = len(hops_covered(new_idx_set)) / num_hops
            delta = new_coverage - current_coverage
            if delta == 0:
                continue
            winner, loser = (b, a) if delta > 0 else (a, b)
            s_diff = scores[winner] - scores[loser]
            rho = 1.0 / (1.0 + np.exp(sigma * s_diff))
            lam = sigma * rho * abs(delta)
            grad[winner] += lam
            grad[loser] -= lam
            h = sigma * sigma * rho * (1 - rho) * abs(delta)
            hess[winner] += h
            hess[loser] += h
    return grad, hess


def compute_ndcg_lambdas_for_group(scores: np.ndarray, labels: np.ndarray, k: int, sigma: float = SIGMA):
    """Standard NDCG@k LambdaRank gradient (binary relevance) - the
    textbook objective LambdaMART was originally paired with, used here
    ONLY as a controlled ablation baseline: identical features,
    warm-start, and regularization as the coverage-aware version, the
    ONE thing that differs is this swaps Δcoverage@k for ΔNDCG@k. Any
    gap between the two isolates what the coverage-specific modification
    itself is contributing, versus generic "a boosted-tree reranker
    helps" effects from the features/warm-start alone.

    ΔNDCG@k for swapping ranks i,j is computed exactly (not
    approximated): removing i and j's old DCG@k contributions (0 if
    their rank is >=k) and adding their new ones after the swap,
    normalized by IDCG@k."""
    n = len(scores)
    if n == 0:
        return np.zeros(n), np.zeros(n)
    order = np.argsort(-scores)
    rank_of = np.empty(n, dtype=int)
    rank_of[order] = np.arange(n)

    ideal = np.sort(labels)[::-1]
    k_eff = min(k, n)
    idcg = sum(ideal[r] / np.log2(r + 2) for r in range(k_eff))
    if idcg == 0:
        return np.zeros(n), np.zeros(n)

    grad = np.zeros(n)
    hess = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if labels[i] <= labels[j]:
                continue
            ri, rj = rank_of[i], rank_of[j]
            delta_dcg = 0.0
            if ri < k_eff:
                delta_dcg -= labels[i] / np.log2(ri + 2)
            if rj < k_eff:
                delta_dcg -= labels[j] / np.log2(rj + 2)
            if rj < k_eff:
                delta_dcg += labels[i] / np.log2(rj + 2)
            if ri < k_eff:
                delta_dcg += labels[j] / np.log2(ri + 2)
            delta = delta_dcg / idcg
            if delta == 0:
                continue
            winner, loser = (i, j) if delta > 0 else (j, i)
            s_diff = scores[winner] - scores[loser]
            rho = 1.0 / (1.0 + np.exp(sigma * s_diff))
            lam = sigma * rho * abs(delta)
            grad[winner] += lam
            grad[loser] -= lam
            h = sigma * sigma * rho * (1 - rho) * abs(delta)
            hess[winner] += h
            hess[loser] += h
    return grad, hess


def train(groups, n_rounds=N_ROUNDS, lr=LEARNING_RATE, max_depth=MAX_DEPTH, verbose=True,
          warm_start=False, feature_cols=None, min_samples_leaf=10, warm_start_col=None, objective="coverage"):
    """warm_start=True: ensemble is initialized to warm_start_col's own
    score (not zero), and trees are fit to learn a small correction on
    top of it - see WARM_START_*/NO_CE_* constants and module-level notes.
    objective="coverage" (default) or "ndcg" - see compute_ndcg_lambdas_for_group
    for why the "ndcg" option exists (ablation, not a recommended setting)."""
    feature_cols = feature_cols or FEATURE_COLS
    X, boundaries, hop_labels_flat, num_hops_per_group, warm_start_scores, labels_flat = prep(groups, feature_cols, warm_start_col)
    n_total = X.shape[0]
    ensemble_scores = warm_start_scores.copy() if warm_start else np.zeros(n_total)
    trees = []
    for round_i in range(n_rounds):
        grad = np.zeros(n_total)
        for (start, end), num_hops in zip(boundaries, num_hops_per_group):
            if objective == "ndcg":
                g, _ = compute_ndcg_lambdas_for_group(ensemble_scores[start:end], labels_flat[start:end], K)
            else:
                g, _ = compute_lambdas_for_group(ensemble_scores[start:end], hop_labels_flat[start:end], num_hops, K)
            grad[start:end] = g
        tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        tree.fit(X, grad)
        ensemble_scores += lr * tree.predict(X)
        trees.append(tree)
        if verbose and (round_i % 10 == 0 or round_i == n_rounds - 1):
            print(f"round {round_i}: mean|grad|={np.abs(grad).mean():.5f}  max|grad|={np.abs(grad).max():.5f}", flush=True)
    return trees


def predict(trees, X, lr=LEARNING_RATE, warm_start_scores=None):
    scores = warm_start_scores.copy() if warm_start_scores is not None else np.zeros(X.shape[0])
    for tree in trees:
        scores += lr * tree.predict(X)
    return scores


def evaluate(groups, trees, lr=LEARNING_RATE, ks=(5, 10, 20), warm_start=False, feature_cols=None, warm_start_col=None):
    """Compares 3 orderings per query: stage1 (baseline), cross-encoder-only
    (beta=1 equivalent), and the trained LambdaMART ensemble - on both
    coverage@k and recall@k/ndcg@k, using this project's existing eval
    functions so numbers are directly comparable to earlier results."""
    feature_cols = feature_cols or FEATURE_COLS
    warm_start_col = warm_start_col or WARM_START_COL
    rows_stage1, rows_ce, rows_lgbm = [], [], []
    for g in groups:
        df = g["df"]
        hop_groups = g["hop_groups"]
        gold_ids = {cid for cids in hop_groups.values() for cid in cids}
        X = df[feature_cols].to_numpy(dtype=float)
        warm_start_scores = df[warm_start_col].to_numpy(dtype=float) if warm_start else None
        lgbm_scores = predict(trees, X, lr, warm_start_scores)

        def _ranked_by(score_col_or_array):
            if isinstance(score_col_or_array, str):
                order = df[score_col_or_array].to_numpy().argsort()[::-1]
            else:
                order = np.argsort(-score_col_or_array)
            return df["chunk_id"].to_numpy()[order].tolist()

        stage1_ranked = _ranked_by("stage1_rank_norm")
        ce_ranked = _ranked_by("ce_score_norm")
        lgbm_ranked = _ranked_by(lgbm_scores)

        for ranked, bucket in [(stage1_ranked, rows_stage1), (ce_ranked, rows_ce), (lgbm_ranked, rows_lgbm)]:
            rec = evaluate_ranking(ranked, gold_ids, ks=list(ks))
            for k in ks:
                rec[f"coverage@{k}"] = coverage_at_k(ranked, hop_groups, k)
            bucket.append(rec)

    return pd.DataFrame(rows_stage1), pd.DataFrame(rows_ce), pd.DataFrame(rows_lgbm)


if __name__ == "__main__":
    train_groups = load_groups(ROOT / "data" / "processed" / "lambdamart_train_features.pkl")
    val_groups = load_groups(ROOT / "data" / "processed" / "lambdamart_val_features.pkl")
    print(f"{len(train_groups)} train groups, {len(val_groups)} val groups")

    print("\n=== training ===")
    trees = train(train_groups)

    for name, groups in [("TRAIN", train_groups), ("VAL (held-out)", val_groups)]:
        print(f"\n=== {name} ===")
        df_stage1, df_ce, df_lgbm = evaluate(groups, trees)
        for label, df in [("stage-1 baseline", df_stage1), ("cross-encoder (beta=1)", df_ce), ("LambdaMART (coverage-aware)", df_lgbm)]:
            means = df.mean(numeric_only=True)
            print(f"  {label}: " + " ".join(f"{c}={means[c]:.4f}" for c in means.index))
