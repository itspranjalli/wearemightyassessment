"""Evaluation dataset: test queries with human-labeled ground truth.

This is TEST DATA, not application logic — the app's prompts stay fully
case-agnostic. To evaluate a different corpus, replace this file's labels
with ones for that corpus.

`relevant`: doc ids a human investigator judged as answering the query.
Everything else in the corpus counts as a distractor for that query.
"""

EVAL_SET = [
    {
        "query": "How did the hacker gain access to the exchange?",
        "relevant": {"case_1", "case_3"},
    },
    {
        "query": "How were the stolen funds laundered?",
        "relevant": {"case_2", "case_5"},
    },
    {
        "query": "Where did the stolen cryptocurrency go after the theft?",
        "relevant": {"case_2", "case_5"},
    },
    {
        "query": "Was there any extortion or ransom demand after the attack?",
        "relevant": {"case_7"},
    },
    {
        "query": "What phishing techniques were used against employees?",
        "relevant": {"case_1"},
    },
    {
        "query": "What do we know about the attacker's location or origin?",
        "relevant": {"case_3"},
    },
    {
        "query": "What suspicious login activity happened before the theft?",
        "relevant": {"case_3"},
    },
    {
        "query": "Which mixing services were used to hide the money trail?",
        "relevant": {"case_5"},
    },
]
