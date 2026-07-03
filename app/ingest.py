"""Load case files from data/, embed them with OpenAI, and store in ChromaDB.

Generic by design: any set of .txt files dropped into data/ becomes the
searchable evidence corpus. Runs idempotently at API startup or manually via
`python -m app.ingest`.
"""
import hashlib
import json
import argparse

import chromadb
from openai import OpenAI

from . import config


def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def load_case_files() -> list[dict]:
    files = sorted(config.DATA_DIR.glob("*.txt"))
    return [
        {"id": f.stem, "source": f.name, "text": f.read_text(encoding="utf-8").strip()}
        for f in files
        if f.stat().st_size > 0
    ]


def corpus_manifest(docs: list[dict]) -> dict:
    """Create stable fingerprints for content and source metadata.

    Embeddings depend only on document text, so the content fingerprint ignores
    filenames. The full fingerprint still records filenames for traceability.
    """
    entries = [
        {
            "id": d["id"],
            "source": d["source"],
            "sha256": hashlib.sha256(d["text"].encode("utf-8")).hexdigest(),
        }
        for d in docs
    ]
    content_entries = sorted(e["sha256"] for e in entries)
    content_payload = json.dumps(content_entries, separators=(",", ":"))
    full_payload = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return {
        "content_fingerprint": hashlib.sha256(content_payload.encode("utf-8")).hexdigest(),
        "fingerprint": hashlib.sha256(full_payload.encode("utf-8")).hexdigest(),
        "documents": entries,
    }


def stored_manifest() -> dict | None:
    try:
        return json.loads(config.CORPUS_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def manifest_content_fingerprint(manifest: dict | None) -> str | None:
    if not manifest:
        return None
    if manifest.get("content_fingerprint"):
        return manifest["content_fingerprint"]
    documents = manifest.get("documents", [])
    if not isinstance(documents, list):
        return None
    hashes = sorted(
        d.get("sha256", "")
        for d in documents
        if isinstance(d, dict) and d.get("sha256")
    )
    if len(hashes) != len(documents):
        return None
    payload = json.dumps(hashes, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_manifest(manifest: dict) -> None:
    config.CORPUS_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.CORPUS_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def refresh_metadata_without_embedding(collection: chromadb.Collection, docs: list[dict]) -> None:
    """Refresh ids/source names when content is unchanged, reusing embeddings."""
    existing = collection.get(include=["documents", "embeddings"])
    by_hash: dict[str, list[list[float]]] = {}
    for text, embedding in zip(existing["documents"], existing["embeddings"]):
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        by_hash.setdefault(digest, []).append(
            embedding.tolist() if hasattr(embedding, "tolist") else embedding
        )

    embeddings = []
    for d in docs:
        digest = hashlib.sha256(d["text"].encode("utf-8")).hexdigest()
        embeddings.append(by_hash[digest].pop())

    if existing["ids"]:
        collection.delete(ids=existing["ids"])
    collection.add(
        ids=[d["id"] for d in docs],
        embeddings=embeddings,
        documents=[d["text"] for d in docs],
        metadatas=[{"source": d["source"]} for d in docs],
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    client = OpenAI(api_key=config.require_api_key())
    response = client.embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def ingest(force: bool = False) -> dict:
    """Embed and store all case files. Skips work if the collection is current."""
    docs = load_case_files()
    if not docs:
        raise RuntimeError(f"No .txt case files found in {config.DATA_DIR}")

    manifest = corpus_manifest(docs)
    previous_manifest = stored_manifest()
    collection = get_collection()
    if (
        not force
        and collection.count() == len(docs)
        and previous_manifest
        and manifest_content_fingerprint(previous_manifest) == manifest["content_fingerprint"]
    ):
        if previous_manifest != manifest:
            refresh_metadata_without_embedding(collection, docs)
            write_manifest(manifest)
        return {
            "status": "up-to-date",
            "documents": collection.count(),
            "fingerprint": manifest["fingerprint"],
            "content_fingerprint": manifest["content_fingerprint"],
        }

    embeddings = embed_texts([d["text"] for d in docs])
    existing = collection.get()["ids"]
    if existing:
        collection.delete(ids=existing)
    collection.add(
        ids=[d["id"] for d in docs],
        embeddings=embeddings,
        documents=[d["text"] for d in docs],
        metadatas=[{"source": d["source"]} for d in docs],
    )
    write_manifest(manifest)
    return {
        "status": "ingested",
        "documents": len(docs),
        "fingerprint": manifest["fingerprint"],
        "content_fingerprint": manifest["content_fingerprint"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="regenerate embeddings even if content is unchanged")
    args = parser.parse_args()
    print(ingest(force=args.force))
