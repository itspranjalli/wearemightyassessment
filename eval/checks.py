"""Deterministic evaluation checks — no LLM involved, fully reproducible.

Retrieval metrics compare the pipeline's evidence/excluded split against the
human answer key. Citation metrics are string checks on the generated report.
"""
import re

CITATION_PATTERN = re.compile(r"\[([\w\-. ]+\.txt)\]")


def retrieval_metrics(relevant: set[str], evidence: list[dict], excluded: list[dict]) -> dict:
    evidence_ids = {e["id"] for e in evidence}
    distractors = {e["id"] for e in evidence + excluded} - relevant

    recall = len(relevant & evidence_ids) / len(relevant)
    rejected = distractors - evidence_ids
    distractor_rejection = len(rejected) / len(distractors) if distractors else 1.0
    top1_relevant = bool(evidence) and evidence[0]["id"] in relevant
    return {
        "recall": recall,
        "distractor_rejection": distractor_rejection,
        "top1_relevant": top1_relevant,
        "missed": sorted(relevant - evidence_ids),
        "leaked_distractors": sorted(distractors & evidence_ids),
    }


def citation_metrics(report: str, evidence: list[dict], excluded: list[dict]) -> dict:
    cited = set(CITATION_PATTERN.findall(report))
    evidence_sources = {e["source"] for e in evidence}
    excluded_sources = {e["source"] for e in excluded}

    valid = cited & evidence_sources
    cites_excluded = sorted(cited & excluded_sources)
    cites_unknown = sorted(cited - evidence_sources - excluded_sources)

    # Paragraphs that state facts should carry a citation; headings and
    # empty lines don't count.
    paragraphs = [
        p for p in report.split("\n\n")
        if p.strip() and not p.strip().startswith("#")
    ]
    cited_paragraphs = sum(1 for p in paragraphs if CITATION_PATTERN.search(p))
    return {
        "has_citations": bool(cited),
        "valid_citations": sorted(valid),
        "cites_excluded_docs": cites_excluded,   # red-herring leakage into the report
        "cites_unknown_docs": cites_unknown,     # hallucinated sources
        "citation_coverage": cited_paragraphs / len(paragraphs) if paragraphs else 0.0,
    }
