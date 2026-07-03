"""Central configuration. All values are case-agnostic and tunable."""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Paths
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
CORPUS_MANIFEST_PATH = CHROMA_DIR / "corpus_manifest.json"
STATIC_DIR = PROJECT_ROOT / "static"

# Vector store
COLLECTION_NAME = "case_files"

# Models
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# Retrieval tuning
NUM_SUB_QUERIES = 4          # how many focused sub-queries the LLM expands into
TOP_K_PER_QUERY = 4          # docs fetched per (sub-)query
RELEVANCE_THRESHOLD = 0.25   # cosine similarity below this = "low relevance"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


def require_api_key() -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to the .env file in the project root."
        )
    return OPENAI_API_KEY
