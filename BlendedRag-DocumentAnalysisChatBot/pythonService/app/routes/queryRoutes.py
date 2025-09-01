# app/routes/queryRoutes.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import re
from app.retrieval.queryRefiner import expandQuery, basicPreprocess
from app.retrieval.retriever import retrieveTopK
from app.embeddings.embeddingClient import EmbeddingClient
from app.storage.documentStore import documentStore
from app.utils.logger import getLogger

router = APIRouter()
logger = getLogger(__name__)
embedding_client = EmbeddingClient()  # singleton client

# Request and Response models
class QueryRequest(BaseModel):
    docId: str
    query: str
    topK: int = 5
    refine: bool = True

class RetrievedChunk(BaseModel):
    chunkIndex: int
    text: str
    score: float
    snippet: str = ""  # top sentences snippet

class QueryResponse(BaseModel):
    docId: str
    query: str
    refinedQueries: List[str]
    results: List[RetrievedChunk]
    mergedBlocks: List[str]  # merged blocks for context window

# --- Helper functions ---
def fuseResultsRRF(resultsList: list[list[dict]], k: int = 60):
    """Fuse multiple query results using Reciprocal Rank Fusion (RRF)."""
    scoreMap = {}
    for results in resultsList:
        for rank, r in enumerate(results, start=1):
            idx = r["chunkIndex"]
            text = r["text"]
            score = 1 / (k + rank)
            if idx in scoreMap:
                scoreMap[idx]["score"] += score
            else:
                scoreMap[idx] = {"score": score, "text": text}
    fused = sorted(
        [{"chunkIndex": idx, "text": v["text"], "score": v["score"]} for idx, v in scoreMap.items()],
        key=lambda x: -x["score"]
    )
    return fused

def mergeTopChunks(chunks: list[dict], maxTokens: int = 500):
    """Merge chunks sequentially until maxTokens is reached."""
    merged = []
    current_block = ""
    current_tokens = 0
    for c in chunks:
        tokens = len(c["text"].split())
        if current_tokens + tokens > maxTokens:
            if current_block:
                merged.append(current_block)
            current_block = c["text"]
            current_tokens = tokens
        else:
            current_block += " " + c["text"]
            current_tokens += tokens
    if current_block:
        merged.append(current_block)
    return merged

def getTopSentences(text: str, query: str, top_n: int = 3):
    """Return top-n sentences containing query terms."""
    sentences = re.split(r'(?<=[.!?]) +', text)
    query_terms = set(query.lower().split())
    scores = [(len(set(s.lower().split()) & query_terms), s) for s in sentences]
    scores.sort(reverse=True)
    top_sentences = [s for _, s in scores[:top_n]]
    return " ".join(top_sentences)

# --- API Endpoint ---
@router.post("/api/query", response_model=QueryResponse)
def queryEndpoint(req: QueryRequest):
    doc = documentStore.getDocument(req.docId)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    refinedQueries = expandQuery(req.query) if req.refine else [basicPreprocess(req.query)]

    # Retrieve top-K chunks for each refined query
    resultsList = [retrieveTopK(req.docId, q, topK=req.topK) for q in refinedQueries]

    # Fuse results using RRF
    fusedChunks = fuseResultsRRF(resultsList, k=60)[:req.topK]

    # Add snippet highlighting
    for chunk in fusedChunks:
        chunk["snippet"] = getTopSentences(chunk["text"], req.query, top_n=3)

    # Merge top chunks for context window
    mergedBlocks = mergeTopChunks(fusedChunks, maxTokens=500)

    logger.info(f"Query for docId={req.docId} returned {len(fusedChunks)} chunks and {len(mergedBlocks)} merged blocks")

    return QueryResponse(
        docId=req.docId,
        query=req.query,
        refinedQueries=refinedQueries,
        results=[
            RetrievedChunk(
                chunkIndex=item["chunkIndex"],
                text=item["text"],
                score=item["score"],
                snippet=item["snippet"]
            )
            for item in fusedChunks
        ],
        mergedBlocks=mergedBlocks
    )



# from fastapi import APIRouter, HTTPException
# from pydantic import BaseModel
# from typing import List
# from app.retrieval.queryRefiner import expandQuery, basicPreprocess
# from app.retrieval.retriever import retrieveTopK
# from app.embeddings.embeddingClient import EmbeddingClient
# from app.storage.documentStore import documentStore
# from app.utils.logger import getLogger

# # Implementing a max-score fusion [to be later by RRF or weighted sum]
# router = APIRouter()
# logger = getLogger(__name__)
# EmbeddingClient = EmbeddingClient()

# class QueryRequest(BaseModel):
#     docId: str
#     query: str
#     topK: int =5
#     refine: bool = True

# class RetrievedChunk(BaseModel):
#     chunkIndex: int
#     text: str
#     score: float

# class QueryResponse(BaseModel)
#     docId: str
#     query: str
#     refinedQueries: List[str]
#     results: List[RetrievedChunk]

# @router.post("/api/query", response_model=QueryResponse)
# def queryEndpoint(req: QueryRequest):
#     doc = documentStore.getDocument(req.docId)
#     if not doc:
#         raise HTTPException(status_code=404, detail="Document not found")

#     if req.refine:
#         refinedQueries = expandQuery(req.query)
#     else:
#         refinedQueries = [basicPreprocess(req.query)]

#     # retrieve with each refined query, then fuse by max score (simple fusion)
#     scoreMap = {}  # chunkIndex -> best (score, text)
#     for q in refinedQueries:
#         results = retrieveTopK(req.docId, q, topK=req.topK)
#         for r in results:
#             idx = int(r["chunkIndex"])
#             score = float(r["score"])
#             if idx not in scoreMap or score > scoreMap[idx]["score"]:
#                 scoreMap[idx] = {"score": score, "text": r["text"]}

#     # convert scoreMap to sorted list
#     fused = sorted(
#         [{"chunkIndex": i, "text": v["text"], "score": v["score"]} for i, v in scoreMap.items()],
#         key=lambda x: -x["score"]
#     )[:req.topK]

#     logger.info(f"Query for docId={req.docId} returned {len(fused)} chunks")

#     return QueryResponse(
#         docId=req.docId,
#         query=req.query,
#         refinedQueries=refinedQueries,
#         results=fused
#     )