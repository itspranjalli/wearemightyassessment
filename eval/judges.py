"""LLM-as-judge evaluation for generated investigation reports.

Judged dimensions (per report):
  - faithfulness: every factual claim checked against the doc it cites (binary
    per claim, aggregated) — strict rubric, defaults to unsupported.
  - answer_relevance: does the report address the detective's question? (1-5)
  - completeness: given the ground-truth relevant docs, which key facts are
    missing from the report? (1-5 + list)

Design notes: the judge sees the FULL corpus plus the answer key — privileged
information the generator never had — and uses a stronger model than the
generator (JUDGE_MODEL, default gpt-4o) to reduce self-preference bias.
The judging prompt is case-agnostic; all case knowledge arrives as data.
"""
import json
import os

from openai import OpenAI

from app import config

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o")

JUDGE_SYSTEM_PROMPT = """\
You are a strict evaluation judge for an investigation-report system.
You will receive: a detective's question, the FULL corpus of source documents,
the ids of documents a human labeled as relevant to the question, and the
system's generated report.

Perform three evaluations and respond with JSON only.

1. FAITHFULNESS — Extract every factual claim in the report (a statement about
   the case that could be true or false; skip hedged meta-commentary and the
   Evidence Gaps section). For each claim, check the source document it cites.
   A claim is supported only if that cited document states it. If a claim
   cites no document, check the whole corpus, but note it as uncited. When
   uncertain, mark the claim unsupported.

2. ANSWER RELEVANCE — Does the report directly address the detective's
   question (5 = fully on-point, 1 = off-topic or evasive)?

3. COMPLETENESS — Considering ONLY the human-labeled relevant documents, list
   any key facts in them that the report failed to mention
   (5 = nothing material missing, 1 = most key facts missing).

JSON schema:
{
  "claims": [{"claim": "...", "cited_source": "case_N.txt or null",
               "supported": true/false, "note": "one short sentence"}],
  "answer_relevance": {"score": 1-5, "reason": "..."},
  "completeness": {"score": 1-5, "missed_facts": ["..."], "reason": "..."}
}"""


def judge_report(query: str, report: str, corpus: dict[str, str], relevant: set[str]) -> dict:
    corpus_block = "\n\n".join(
        f"--- Document id: {doc_id} ---\n{text}" for doc_id, text in corpus.items()
    )
    user_content = (
        f"Detective's question:\n{query}\n\n"
        f"Human-labeled relevant document ids: {sorted(relevant)}\n\n"
        f"Full corpus:\n\n{corpus_block}\n\n"
        f"Generated report to evaluate:\n\n{report}"
    )
    client = OpenAI(api_key=config.require_api_key())
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    verdict = json.loads(response.choices[0].message.content)

    claims = verdict.get("claims", [])
    supported = sum(1 for c in claims if c.get("supported"))
    verdict["faithfulness"] = {
        "supported": supported,
        "total": len(claims),
        "score": supported / len(claims) if claims else None,
        "unsupported_claims": [c for c in claims if not c.get("supported")],
    }
    return verdict
