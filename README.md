# Evidence-Aware RAG for Multi-Hop Financial QA

## The problem, with a real example

Imagine you ask a system: *"What was 3M's cash and cash equivalents, cost of sales, total current assets and total current liabilities in FY2018?"*

To answer this, the system has to find four separate numbers, scattered across different tables in a 200-page SEC filing. We tried this with a standard search setup (the kind most "search your PDFs" tools use), and it completely failed to find one of the four numbers - **cash and cash equivalents didn't show up anywhere in the top 100 search results**, even though the exact right table row was sitting right there in the document. When we asked for that one number by itself ("What was 3M's cash and cash equivalents in FY2018?"), the system found it instantly, ranked #1.

**Why did this happen?** Search systems typically work by matching words in your question to words in the document. When you cram four different topics into one question, the system gets confused about which words matter for which fact, and the search results end up favoring whichever table happens to overlap with the *most* words in the question - not necessarily the table with the *right* answer. We found a fix for this (explained below), and it's one of two real improvements this project made.

The second problem is subtler. Financial filings often repeat the same number in more than one place - a company's revenue might appear once in a table and again in a press release that came out a few months earlier. If your system finds two copies of that revenue number but misses a *different* required fact entirely, standard scoring will say "great, I found most of what you asked for!" - even though it actually failed the question. We built a way to measure and fix this too.

This project is about diagnosing problems like these in financial document search, one real example at a time, and testing fixes properly (on data the system has never seen) rather than assuming a fix works.

## How the system works, in plain terms

1. **Search** - given a question, find candidate passages from the filings using two methods at once: keyword matching (like a smarter version of Ctrl+F) and meaning-based matching (which can find a match even if the words don't line up exactly, e.g. "profit" vs "net income"). Combine both into one ranked list.
2. **Re-rank** - take the top candidates from step 1 and use a more careful (but slower) model to re-score them, one at a time, looking closely at the question *and* the candidate together rather than separately.
3. **Coverage check** (for multi-fact questions specifically) - a second re-ranking pass that explicitly tries to avoid the "found two copies of one fact, missed another fact entirely" problem described above.

Everything below is about how well each of these steps actually works, tested on real questions the system never saw during development.

## Research Question 1: Does a more careful "second look" actually help?

**The question in plain terms**: after the first search pass finds ~50 candidate passages, does spending more compute to carefully re-read and re-score each one (the "re-rank" step above) actually make things better, or is it not worth it?

**What we did**: we fine-tuned a small language model specifically to get better at this domain, by showing it real mistakes our own search system was making - not made-up wrong answers, but the *actual* wrong passages the system was ranking highly by mistake. This is the training data a generic, off-the-shelf model wouldn't have.

**Did it work?** Yes, clearly - tested on 185 questions the model never saw during training, built from facts it was never trained on either:

| | Before re-ranking | After re-ranking |
|---|---|---|
| How often the right answer is in the top 10 results | 74% | **94%** |
| Ranking quality score (0=bad, 1=perfect) | 0.52 | **0.84** |

**An honest catch we found along the way**: while building this, we once checked a model whose training looked completely normal - the error numbers went down steadily like they're supposed to - but when we actually looked at what it was doing, it was scoring *wrong* answers higher than *right* answers. The training metrics looked fine; the model was actually broken. The lesson: you can't just trust that "the numbers went down," you have to check what the model actually does on real examples. We check this on every model in this project now.

**Another honest finding**: the re-ranking model works great on questions phrased like a simple fact-lookup ("what was X's revenue"), but when we tested it on real analyst-style questions that require *calculating* something ("what's the 3-year average capex as a percent of revenue"), it sometimes made things *worse* than not re-ranking at all. So we built a dial (called `beta`) for how much to trust the re-ranker, and the honest answer is: trust it fully if your questions look like simple fact lookups, trust it less if they're more analytical.

## Research Question 2: What about the "found two copies of one fact, missed another" problem?

**A real example.** A company's FY2023 10-K and their earnings press release (issued a few months earlier) both report the same net income figure. A question asking for 4 different facts about that company might get: 2 copies of net income (one from each document), 0 copies of a different required fact. A standard scoring system sees "4 out of 4 slots are gold chunks" and says great job - but the system actually only found 1 out of the 2 distinct facts needed. It just found the *same one* twice.

**How we measured this properly**: we built a metric that counts *distinct facts found*, not *total correct passages found*. On a set of questions specifically designed to have this repeated-fact problem, the two metrics told very different stories about the exact same search results:

| | "Total correct passages found" (standard metric) | "Distinct facts found" (our metric) |
|---|---|---|
| Plain search, no re-ranking | 24% | 42% |

That's a huge gap on the *same* ranked list - proof that the standard way of measuring "did we find the answer" can be seriously misleading for questions with more than one required fact.

**Did we find a fix?** A modest one. We trained a second re-ranking model specifically taught to notice "this candidate repeats a fact I already have" and prefer a candidate covering a still-missing fact instead. It's a smaller, cheaper model than the one from RQ1 (no expensive fine-tuning needed), and here's what it achieved on questions it never saw during training:

| | Plain search | + our fact-coverage-aware model |
|---|---|---|
| Distinct facts found | 42% | **49%** |

That's a real, honest, modestly-sized improvement (+16%) - not a breakthrough, but genuine and properly tested. We also checked: does a version of this model *without* the fact-coverage-specific training do just as well, using the exact same setup otherwise? No - it actually performed **worse than not re-ranking at all**. That comparison confirms the fact-coverage-specific idea is what's doing the work, not just "any model helps a bit."

**The fix that mattered more than expected**: remember the opening example, where bundling 4 facts into one question made the search miss one of them entirely? We found that simply **splitting a multi-fact question into separate single-fact questions**, searching for each one individually, and combining the results back together, recovered that missing fact completely - and more broadly, improved how many distinct facts get found by about 20%, with zero cases where it made things worse across 40 test questions. This was a bigger, cleaner win than any of the fancier re-ranking approaches we tried, which is itself a useful lesson: sometimes the simplest fix (ask one question at a time) beats a more sophisticated model.

## What actually worked, summarized honestly

| Approach | What it's for | Result |
|---|---|---|
| Fine-tuned re-ranking model | Simple fact-lookup and multi-fact questions | **Strong, reliable improvement** (74%→94% top-10 hit rate) |
| Splitting multi-fact questions into separate searches | Multi-fact questions specifically | **Strong, reliable improvement**, and the simplest fix we tried |
| Fact-coverage-aware re-ranking model | Questions with repeated/restated facts | **Real but modest improvement** (+16%), honestly small |
| A from-scratch attempt using a similar idea but a different math formula | Same as above | **Failed** - worse than doing nothing, a useful negative result showing the *specific* design choices mattered, not just the general idea |

We also tried a couple of more exotic ideas borrowed from academic research on "diverse search results" (explained in full in `RQ2_RESULTS.md` and `PROJECT_SUMMARY.md` for anyone who wants the deeper technical detail) - they didn't beat the simpler approach above, which is itself worth knowing before spending time on something fancier.

## Where things stand

- The fine-tuned re-ranker (RQ1) is the strongest, most reliable part of this project.
- Splitting multi-fact questions before searching is a simple trick that helped more than expected.
- The repeated-fact problem (RQ2) is real and measurable, and we made modest, honest progress on it - not a solved problem.
- Every number above was checked on questions the relevant model never saw during training, specifically to avoid fooling ourselves - see `PROJECT_SUMMARY.md` for the full discipline behind that.

## For more technical detail

- [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md) - full technical writeup tying both research questions together
- [`RQ1_RESULTS.md`](RQ1_RESULTS.md) - the re-ranking model: architecture, training details, full results
- [`RQ2_RESULTS.md`](RQ2_RESULTS.md) - the fact-coverage model: the math behind it, full results, and everything that didn't work

## Repository structure

```
scripts/          the search pipeline, data preparation, model training
eval/              the scoring code (including the "distinct facts found" metric)
data/processed/    question sets, answer keys, generated prompts
```
