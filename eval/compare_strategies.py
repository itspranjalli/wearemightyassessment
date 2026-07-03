"""Head-to-head comparison of retrieval strategies over the labeled dataset.

Runs three strategies as an ablation, so each stage's contribution is visible:

  A. single-step top-1        — the literal "closest document" baseline
  B. single-step top-k+thresh — multiple docs by raw similarity cutoff,
                                no query expansion, no LLM grading
  C. multi-step (full)        — expansion + multi-query search + LLM grading
                                (the shipped pipeline: app.retrieval.retrieve)

All three see the same ChromaDB collection and the same 8 labeled queries
(eval/dataset.py). We report set-retrieval quality (recall / precision / F1),
top-1 accuracy, and a single-doc vs multi-doc query breakdown — the last one is
where single-step structurally cannot compete.

Run:  python -m eval.compare_strategies
"""
from statistics import mean

from app import config
from app.ingest import embed_texts, get_collection, ingest
from app.retrieval import retrieve

from .dataset import EVAL_SET


# ---- strategies -----------------------------------------------------------

def strat_single_top1(query: str, _emb) -> set[str]:
    col = get_collection()
    hit = col.query(query_embeddings=[_emb], n_results=1)
    return {hit["ids"][0][0]}


def strat_single_topk_threshold(query: str, _emb) -> set[str]:
    """Retrieve up to TOP_K docs, keep those above the similarity threshold.
    A doc is always kept if it is the single closest (avoids empty results)."""
    col = get_collection()
    k = min(config.TOP_K_PER_QUERY, col.count())
    hit = col.query(query_embeddings=[_emb], n_results=k)
    ids, dists = hit["ids"][0], hit["distances"][0]
    kept = {doc_id for doc_id, dist in zip(ids, dists)
            if (1.0 - dist) >= config.RELEVANCE_THRESHOLD}
    return kept or {ids[0]}


def strat_multi_step(query: str, _emb) -> set[str]:
    result = retrieve(query)
    return {e["id"] for e in result["evidence"]}


STRATEGIES = [
    ("A. single-step top-1", strat_single_top1),
    ("B. single-step top-k + threshold", strat_single_topk_threshold),
    ("C. multi-step (full pipeline)", strat_multi_step),
]


# ---- metrics --------------------------------------------------------------

def score(relevant: set[str], retrieved: set[str]) -> dict:
    tp = len(relevant & retrieved)
    recall = tp / len(relevant) if relevant else 0.0
    precision = tp / len(retrieved) if retrieved else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"recall": recall, "precision": precision, "f1": f1, "retrieved": len(retrieved)}


def main():
    print("Ingestion:", ingest(), "\n")

    # Pre-embed each query once; single-step strategies reuse the vector.
    queries = [item["query"] for item in EVAL_SET]
    embeddings = embed_texts(queries)

    # Run every strategy once per query; cache retrieved sets for reuse.
    # results[strategy_name] = list of per-query metric dicts (with is_multi flag)
    # retrieved_sets[strategy_name] = list of retrieved id sets, aligned to EVAL_SET
    results = {name: [] for name, _ in STRATEGIES}
    retrieved_sets = {name: [] for name, _ in STRATEGIES}
    for item, emb in zip(EVAL_SET, embeddings):
        relevant = item["relevant"]
        is_multi = len(relevant) > 1
        for name, fn in STRATEGIES:
            retrieved = fn(item["query"], emb)
            retrieved_sets[name].append(retrieved)
            m = score(relevant, retrieved)
            m["is_multi"] = is_multi
            results[name].append(m)

    def agg(rows, key, subset=None):
        sel = [r for r in rows if subset is None or r["is_multi"] == subset]
        return mean(r[key] for r in sel) if sel else 0.0

    n_multi = sum(len(i["relevant"]) > 1 for i in EVAL_SET)
    n_single = len(EVAL_SET) - n_multi

    print(f"Dataset: {len(EVAL_SET)} queries ({n_single} single-doc, {n_multi} multi-doc)\n")
    print(f"{'Strategy':<36}{'Recall':>8}{'Prec':>8}{'F1':>8}{'AvgRet':>8}")
    print("-" * 68)
    for name, _ in STRATEGIES:
        rows = results[name]
        print(f"{name:<36}{agg(rows,'recall'):>7.0%}{agg(rows,'precision'):>8.0%}"
              f"{agg(rows,'f1'):>8.2f}{mean(r['retrieved'] for r in rows):>8.1f}")

    print(f"\nRecall on MULTI-document queries only (n={n_multi}):")
    for name, _ in STRATEGIES:
        print(f"  {name:<36}{agg(results[name],'recall',subset=True):>6.0%}")
    print(f"\nRecall on SINGLE-document queries only (n={n_single}):")
    for name, _ in STRATEGIES:
        print(f"  {name:<36}{agg(results[name],'recall',subset=False):>6.0%}")

    print("\nCost / call profile (per query):")
    print(f"  {'A. single-step top-1':<36}1 embed, 0 LLM calls")
    print(f"  {'B. single-step top-k + threshold':<36}1 embed, 0 LLM calls")
    print(f"  {'C. multi-step (full pipeline)':<36}~2 embed batches + 2 LLM calls (expand + grade)")

    print("\nPer-query recall (relevant set → each strategy's retrieved set):")
    for i, item in enumerate(EVAL_SET):
        print(f"\n  Q: {item['query']}")
        print(f"     relevant: {sorted(item['relevant'])}")
        for name, _ in STRATEGIES:
            got = sorted(retrieved_sets[name][i])
            hit = retrieved_sets[name][i] & item["relevant"]
            print(f"     {name:<36}{got}  (recall {len(hit)/len(item['relevant']):.0%})")


if __name__ == "__main__":
    main()
