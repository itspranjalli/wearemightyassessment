"""FastAPI app: evidence-retrieval API + static detective UI."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config
from .ingest import get_collection, ingest
from .report import generate_report
from .retrieval import retrieve


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the vector store on startup if the corpus isn't ingested yet.
    if config.OPENAI_API_KEY:
        try:
            print("Ingestion:", ingest())
        except Exception as exc:  # keep the app up; /api endpoints report the error
            print(f"Ingestion failed at startup: {exc}")
    else:
        print("OPENAI_API_KEY not set — add it to .env; ingestion deferred.")
    yield


app = FastAPI(title="Case Evidence Retrieval System", lifespan=lifespan)


class InvestigateRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)


@app.post("/api/investigate")
def investigate(request: InvestigateRequest):
    try:
        ingest()  # no-op when the collection is current
        retrieval = retrieve(request.query)
        report = generate_report(request.query, retrieval["evidence"])
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {**retrieval, "report": report}


@app.get("/api/cases")
def list_cases():
    collection = get_collection()
    data = collection.get()
    cases = [
        {"id": doc_id, "source": meta["source"], "preview": doc[:160]}
        for doc_id, meta, doc in zip(data["ids"], data["metadatas"], data["documents"])
    ]
    return {"count": len(cases), "cases": sorted(cases, key=lambda c: c["id"])}


@app.get("/")
def index():
    return FileResponse(config.STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")
