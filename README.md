# Case Evidence Retrieval System

Setup and run guide for the AI-powered case evidence retrieval app.

For architecture, design decisions, evaluation rationale, and system design
round material, see:

- [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md)
- [docs/system_design_presentation_report.md](docs/system_design_presentation_report.md)
- [docs/system_design_presentation_report.pdf](docs/system_design_presentation_report.pdf)

## Prerequisites

- Python 3.10+
- An OpenAI API key

## 1. Install Dependencies

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you do not want a virtual environment, install directly:

```bash
pip install -r requirements.txt
```

## 2. Configure Environment

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and set your API key:

```bash
OPENAI_API_KEY=your_api_key_here
```

Optional model overrides:

```bash
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini
```

## 3. Add or Replace Case Files

Place case files in the `data/` directory as `.txt` files:

```text
data/
  case_1.txt
  case_2.txt
  ...
```

The app ingests all non-empty `.txt` files in `data/`.

Embeddings are stored in `chroma_db/`. The system uses a content fingerprint, so
embeddings are regenerated only when case-file text changes. Filename-only
changes refresh stored metadata without calling the embedding API.

## 4. Run the App

Start the FastAPI server:

```bash
uvicorn app.main:app --port 8000
```

Open the UI:

[http://localhost:8000]

On startup, the app attempts to ingest the case files if `OPENAI_API_KEY` is
configured.

## 5. API Quick Check

List ingested cases:

```bash
curl http://localhost:8000/api/cases
```

Ask an investigation question:

```bash
curl -X POST http://localhost:8000/api/investigate \
  -H "Content-Type: application/json" \
  -d '{"query":"How were the stolen funds laundered?"}'
```

## 6. Manual Ingestion

To rebuild the vector store manually:

```bash
python -m app.ingest
```

This respects the content fingerprint and only calls the embedding API when
case-file text has changed.

To force regeneration even when content is unchanged:

```bash
python -m app.ingest --force
```

## 7. Verification Commands

Offline mock verification, no API calls:

```bash
python -m eval.mock_system_check
```

Functional evaluation, requires API access:

```bash
python -m eval.run_eval --skip-judge --skip-security
```

Full evaluation, requires API access:

```bash
python -m eval.run_eval
```

Security suite only, requires API access:

```bash
python -m eval.security_tests
```

Retrieval-strategy comparison, requires API access:

```bash
python -m eval.compare_strategies
```

## 8. Troubleshooting

`OPENAI_API_KEY is not set`

Make sure `.env` exists in the project root and contains:

```bash
OPENAI_API_KEY=your_api_key_here
```

`No .txt case files found`

Add at least one non-empty `.txt` file to `data/`.

Stale or unexpected retrieval results

The app fingerprints corpus contents automatically. If you still want a clean
manual rebuild, delete `chroma_db/` and run:

```bash
python -m app.ingest
```

Network or DNS errors during evaluation

The live eval commands call OpenAI APIs. Use the offline mock check when network
access is unavailable:

```bash
python -m eval.mock_system_check
```

## Project Files

- `app/` - application code
- `data/` - case files
- `static/` - browser UI
- `eval/` - verification and evaluation scripts
- `docs/` - system design and presentation documents
- `requirements.txt` - Python dependencies
