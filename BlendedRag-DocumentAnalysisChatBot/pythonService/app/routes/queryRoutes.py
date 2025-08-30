from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from app.retrieval.queryRefiner import expandQuery, basicPreprocess
from app.retrieval.retriever import retrieveTopK
from app.embeddings.embeddingClient import EmbeddingClient
from app.storage.documentStore import documentStore
from app.utils.logger import getLogger

# Implementing a max-score fusion [to be later by RRF or weighted sum]
router = APIRouter()
logger = getLogger(__name__)
EmbeddingClient = EmbeddingClient()

class QueryRequest(BaseModel):
    docId: str
    query: str
    topK: int =5
    refine: bool = True

class RetrievedChunk(BaseModel):
    chunkIndex: int
    text: str
    score: float

class QueryResponse(BaseModel)
    docId: str
    query: str
    refinedQueries: List[str]
    results: List[RetrievedChunk]

@router.post("/api/query", response_model=QueryResponse)
def queryEndpoint(req: QueryRequest):
    doc = documentStore.getDocument(req.docId)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if req.refine:
        refinedQueries = expandQuery(req.query)
    else:
        refinedQueries = [basicPreprocess(req.query)]

    # retrieve with each refined query, then fuse by max score (simple fusion)
    scoreMap = {}  # chunkIndex -> best (score, text)
    for q in refinedQueries:
        results = retrieveTopK(req.docId, q, topK=req.topK)
        for r in results:
            idx = int(r["chunkIndex"])
            score = float(r["score"])
            if idx not in scoreMap or score > scoreMap[idx]["score"]:
                scoreMap[idx] = {"score": score, "text": r["text"]}

    # convert scoreMap to sorted list
    fused = sorted(
        [{"chunkIndex": i, "text": v["text"], "score": v["score"]} for i, v in scoreMap.items()],
        key=lambda x: -x["score"]
    )[:req.topK]

    logger.info(f"Query for docId={req.docId} returned {len(fused)} chunks")

    return QueryResponse(
        docId=req.docId,
        query=req.query,
        refinedQueries=refinedQueries,
        results=fused
    )