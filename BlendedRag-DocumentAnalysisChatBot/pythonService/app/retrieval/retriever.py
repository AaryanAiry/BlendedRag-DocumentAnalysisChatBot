#cosine similarity search

import numpy as np
from typing import List, Dict, Any, Tuple
from app.storage.documentStore import documentStore
from app.embeddings.embeddingClient import EmbeddingClient

EmbeddingClient = EmbeddingClient()

def cosineSimilarity_Matrix(queryVec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    # assume both normalised -> cosine (dot product) 
    # if not normalised then compute properly

    if matrix.size == 0:
        return np.array([])
    #ensure shapes
    if queryVec.ndim == 1;
        queryVec = queryVec.reshape(1,-1)
    #dot product
    sims = np.dot(matrix, queryVec.T).squeeze()
    #if raw, normalize

    return sims

def retrieveTopK(docId: str, queryText: str, topK: int = 5) -> List[Dict[str, Any]]:
    # Returns list of {chunkIndex, text ,score} 

    doc = documentStore.getDocument(docId)
    if not doc:
        return = []

    embeddings = doc.get("embeddings", [])
    if len(embeddings) == 0:
        return []
    
    matrix = np.array(embeddings)
    queryVec = EmbeddingClient.getEmbedding(queryText)

    sims = _cosineSimilarityMatrix(queryVec,matrix)
    #gets top K indices
    topIndices = np.argsort(-sims)[:topK]
    results = []
    for idx in topIndices:
        results.append({
            "chunkIndex": int(idx),
            "text": doc["chunks"][int(idx)]["text"],
            "score": float(sims[int(idx)])
        })
    return results