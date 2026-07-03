"""Offline mock verification for the RAG pipeline.

This does not call OpenAI or ChromaDB. It monkeypatches the app's integration
points with deterministic fakes so we can verify that the declared workflow is
wired correctly: multi-query retrieval, confidence ranking, excluded documents
with reasons, citation validation/retry, and deterministic eval checks.

Run:
  python -m eval.mock_system_check
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from app import report, retrieval

from .checks import citation_metrics, retrieval_metrics


MOCK_DOCS = {
    "case_access": {
        "source": "case_access.txt",
        "text": "The attacker used a phishing email and fake login page to steal credentials.",
    },
    "case_funds": {
        "source": "case_funds.txt",
        "text": "The stolen funds moved to a Solana wallet and then through Tornado Cash.",
    },
    "case_noise": {
        "source": "case_noise.txt",
        "text": "This unrelated report describes office Wi-Fi maintenance and lunch orders.",
    },
}


class FakeCollection:
    def count(self) -> int:
        return len(MOCK_DOCS)

    def query(self, query_embeddings, n_results: int):
        q = query_embeddings[0].lower()
        if "launder" in q or "fund" in q or "mixer" in q or "tornado" in q:
            ranked = [
                ("case_funds", 0.12),
                ("case_noise", 0.68),
                ("case_access", 0.74),
            ]
        elif "access" in q or "phishing" in q:
            ranked = [
                ("case_access", 0.10),
                ("case_noise", 0.70),
                ("case_funds", 0.76),
            ]
        else:
            ranked = [
                ("case_noise", 0.60),
                ("case_access", 0.72),
                ("case_funds", 0.78),
            ]
        ranked = ranked[:n_results]
        return {
            "ids": [[doc_id for doc_id, _ in ranked]],
            "documents": [[MOCK_DOCS[doc_id]["text"] for doc_id, _ in ranked]],
            "metadatas": [[{"source": MOCK_DOCS[doc_id]["source"]} for doc_id, _ in ranked]],
            "distances": [[distance for _, distance in ranked]],
        }


class FakeOpenAI:
    calls = 0

    def __init__(self, api_key: str | None = None):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create_completion)
        )

    def _create_completion(self, **kwargs):
        FakeOpenAI.calls += 1
        system = kwargs["messages"][0]["content"]
        if "Additional output constraint" not in system:
            content = "The funds moved through Tornado Cash according to case_funds.txt."
        else:
            content = (
                "## Executive Summary\n\n"
                "The stolen funds moved through Tornado Cash after reaching a Solana wallet "
                "[case_funds.txt].\n\n"
                "## Evidence Analysis\n\n"
                "The evidence supports a laundering path involving Solana and Tornado Cash "
                "[case_funds.txt].\n\n"
                "## Assessment & Leads\n\n"
                "Investigators should trace the Solana wallet and downstream mixer outputs "
                "[case_funds.txt].\n\n"
                "## Evidence Gaps\n\n"
                "The final cash-out exchange is not identified in the mock evidence "
                "[case_funds.txt]."
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def fake_expand_query(query: str) -> list[str]:
    return [
        "stolen funds laundering path",
        "crypto mixer Tornado Cash evidence",
        "Solana wallet fund movement",
    ]


def fake_embed_texts(texts: list[str]) -> list[str]:
    return texts


def fake_grade_relevance(query: str, candidates: list[dict]) -> dict[str, dict]:
    grades = {}
    for c in candidates:
        relevant = "Tornado Cash" in c["text"] or "Solana wallet" in c["text"]
        grades[c["id"]] = {
            "relevant": relevant,
            "reason": (
                "Contains laundering evidence involving Solana and Tornado Cash."
                if relevant
                else "Does not contain evidence relevant to the detective question."
            ),
        }
    return grades


def main() -> None:
    original = {
        "expand_query": retrieval.expand_query,
        "embed_texts": retrieval.embed_texts,
        "get_collection": retrieval.get_collection,
        "grade_relevance": retrieval.grade_relevance,
        "openai": report.OpenAI,
        "api_key": report.config.require_api_key,
    }
    try:
        retrieval.expand_query = fake_expand_query
        retrieval.embed_texts = fake_embed_texts
        retrieval.get_collection = lambda: FakeCollection()
        retrieval.grade_relevance = fake_grade_relevance
        report.OpenAI = FakeOpenAI
        report.config.require_api_key = lambda: "mock-key"

        result = retrieval.retrieve("How were the stolen funds laundered?")
        generated = report.generate_report(result["query"], result["evidence"])

        retrieval_check = retrieval_metrics(
            {"case_funds"},
            result["evidence"],
            result["excluded"],
        )
        citation_check = citation_metrics(
            generated,
            result["evidence"],
            result["excluded"],
        )

        summary = {
            "query": result["query"],
            "sub_queries": result["sub_queries"],
            "evidence": [
                {
                    "id": e["id"],
                    "source": e["source"],
                    "confidence": e["confidence"],
                    "reason": e["reason"],
                }
                for e in result["evidence"]
            ],
            "excluded": [
                {
                    "id": e["id"],
                    "source": e["source"],
                    "confidence": e["confidence"],
                    "reason": e["reason"],
                }
                for e in result["excluded"]
            ],
            "retrieval_metrics": retrieval_check,
            "citation_metrics": citation_check,
            "report_retry_used": FakeOpenAI.calls == 2,
            "report": generated,
        }

        assert result["sub_queries"], "query expansion did not produce sub-queries"
        assert result["evidence"][0]["id"] == "case_funds", "wrong top evidence"
        assert result["evidence"][0]["confidence"] == 0.88, "confidence conversion failed"
        assert any(e["id"] == "case_noise" for e in result["excluded"]), "noise doc not excluded"
        assert retrieval_check["recall"] == 1.0, "mock recall should be perfect"
        assert citation_check["has_citations"], "report should include bracket citations"
        assert not citation_check["cites_unknown_docs"], "report cited unknown docs"
        assert not citation_check["cites_excluded_docs"], "report cited excluded docs"
        assert summary["report_retry_used"], "citation retry should have been exercised"

        out = Path("eval/mock_results.json")
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        print(f"\nMOCK SYSTEM CHECK: PASS ({out})")
    finally:
        retrieval.expand_query = original["expand_query"]
        retrieval.embed_texts = original["embed_texts"]
        retrieval.get_collection = original["get_collection"]
        retrieval.grade_relevance = original["grade_relevance"]
        report.OpenAI = original["openai"]
        report.config.require_api_key = original["api_key"]


if __name__ == "__main__":
    main()
