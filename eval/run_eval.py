"""Run the full evaluation suite: retrieval → report → deterministic checks
→ LLM-as-judge, followed by the adversarial security suite. Prints summary
tables and writes eval/results.json.

Usage:
  python -m eval.run_eval                  # functional eval + security suite
  python -m eval.run_eval --skip-judge     # deterministic checks + security
  python -m eval.run_eval --skip-security  # functional eval only
  python -m eval.run_eval --security-only  # security suite only
"""
import argparse
import json
import sys
from pathlib import Path

from app.ingest import ingest, load_case_files
from app.report import generate_report
from app.retrieval import retrieve

from .checks import citation_metrics, retrieval_metrics
from .dataset import EVAL_SET
from .judges import JUDGE_MODEL, judge_report
from .security_tests import main as run_security_suite

RESULTS_PATH = Path(__file__).parent / "results.json"

MIN_RETRIEVAL_RECALL = 0.95
MIN_DISTRACTOR_REJECTION = 0.75
MIN_TOP1_RATE = 0.90
MIN_CITATION_COVERAGE = 0.75


def fmt_pct(x):
    return "  n/a" if x is None else f"{x * 100:4.0f}%"


def main(skip_judge: bool = False):
    print("Ingestion:", ingest())
    corpus = {d["id"]: d["text"] for d in load_case_files()}

    results = []
    for i, item in enumerate(EVAL_SET, 1):
        query, relevant = item["query"], item["relevant"]
        print(f"\n[{i}/{len(EVAL_SET)}] {query}")

        retrieval = retrieve(query)
        report = generate_report(query, retrieval["evidence"])

        row = {
            "query": query,
            "relevant": sorted(relevant),
            "evidence": [(e["id"], e["confidence"]) for e in retrieval["evidence"]],
            "excluded": [(e["id"], e["confidence"]) for e in retrieval["excluded"]],
            "retrieval": retrieval_metrics(relevant, retrieval["evidence"], retrieval["excluded"]),
            "citations": citation_metrics(report, retrieval["evidence"], retrieval["excluded"]),
            "report": report,
        }
        r, c = row["retrieval"], row["citations"]
        print(f"    retrieval: recall {fmt_pct(r['recall'])}  distractor-rejection {fmt_pct(r['distractor_rejection'])}  top1-relevant {r['top1_relevant']}")
        print(f"    citations: coverage {fmt_pct(c['citation_coverage'])}  cites-excluded {c['cites_excluded_docs']}  cites-unknown {c['cites_unknown_docs']}")

        if not skip_judge:
            verdict = judge_report(query, report, corpus, relevant)
            row["judge"] = verdict
            f = verdict["faithfulness"]
            print(f"    judge ({JUDGE_MODEL}): faithfulness {f['supported']}/{f['total']} claims"
                  f"  relevance {verdict['answer_relevance']['score']}/5"
                  f"  completeness {verdict['completeness']['score']}/5")
            for claim in f["unsupported_claims"]:
                print(f"      !! unsupported: {claim.get('claim', '')[:90]}")
        results.append(row)

    # Aggregate summary
    n = len(results)
    print("\n" + "=" * 72)
    print("AGGREGATE")
    avg_recall = sum(r["retrieval"]["recall"] for r in results) / n
    avg_distractor_rejection = sum(r["retrieval"]["distractor_rejection"] for r in results) / n
    top1_rate = sum(r["retrieval"]["top1_relevant"] for r in results) / n
    avg_citation_coverage = sum(r["citations"]["citation_coverage"] for r in results) / n
    reports_citing_excluded = sum(bool(r["citations"]["cites_excluded_docs"]) for r in results)
    hallucinated_sources = sum(bool(r["citations"]["cites_unknown_docs"]) for r in results)
    reports_missing_citations = sum(not r["citations"]["has_citations"] for r in results)

    print(f"  retrieval recall:        {fmt_pct(avg_recall)}")
    print(f"  distractor rejection:    {fmt_pct(avg_distractor_rejection)}")
    print(f"  top-1 relevant:          {sum(r['retrieval']['top1_relevant'] for r in results)}/{n}")
    print(f"  citation coverage:       {fmt_pct(avg_citation_coverage)}")
    print(f"  reports missing cites:   {reports_missing_citations}/{n}")
    print(f"  reports citing excluded: {reports_citing_excluded}/{n}")
    print(f"  hallucinated sources:    {hallucinated_sources}/{n}")
    if not skip_judge:
        scored = [r for r in results if r.get("judge", {}).get("faithfulness", {}).get("score") is not None]
        total_claims = sum(r["judge"]["faithfulness"]["total"] for r in scored)
        supported = sum(r["judge"]["faithfulness"]["supported"] for r in scored)
        print(f"  faithfulness:            {supported}/{total_claims} claims supported ({fmt_pct(supported / total_claims if total_claims else None).strip()})")
        print(f"  answer relevance (avg):  {sum(r['judge']['answer_relevance']['score'] for r in scored) / len(scored):.1f}/5")
        print(f"  completeness (avg):      {sum(r['judge']['completeness']['score'] for r in scored) / len(scored):.1f}/5")

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nFull results written to {RESULTS_PATH}")

    functional_ok = (
        avg_recall >= MIN_RETRIEVAL_RECALL
        and avg_distractor_rejection >= MIN_DISTRACTOR_REJECTION
        and top1_rate >= MIN_TOP1_RATE
        and avg_citation_coverage >= MIN_CITATION_COVERAGE
        and reports_missing_citations == 0
        and reports_citing_excluded == 0
        and hallucinated_sources == 0
    )
    print("\nFUNCTIONAL GATE:", "PASS" if functional_ok else "FAIL")
    print(
        "  thresholds:"
        f" recall>={MIN_RETRIEVAL_RECALL:.0%},"
        f" distractor-rejection>={MIN_DISTRACTOR_REJECTION:.0%},"
        f" top1>={MIN_TOP1_RATE:.0%},"
        f" citation-coverage>={MIN_CITATION_COVERAGE:.0%},"
        " no missing/invalid citations"
    )
    return functional_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-judge", action="store_true", help="skip the LLM-as-judge stage")
    parser.add_argument("--skip-security", action="store_true", help="skip the adversarial security suite")
    parser.add_argument("--security-only", action="store_true", help="run only the security suite")
    args = parser.parse_args()

    functional_ok = True
    security_ok = True
    if not args.security_only:
        functional_ok = main(skip_judge=args.skip_judge)
    if not args.skip_security or args.security_only:
        print("\n" + "#" * 72)
        print("# ADVERSARIAL SECURITY SUITE")
        print("#" * 72 + "\n")
        security_ok = run_security_suite()

    # Non-zero exit if the functional or security gate fails, so this can gate CI.
    sys.exit(0 if (functional_ok and security_ok) else 1)
