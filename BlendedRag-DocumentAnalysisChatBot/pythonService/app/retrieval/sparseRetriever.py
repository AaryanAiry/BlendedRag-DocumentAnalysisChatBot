# app/retrievers/sparseRetriever.py
import os
import pickle
from rank_bm25 import BM25Okapi
from typing import List, Dict
from app.utils.logger import getLogger

logger = getLogger(__name__)

CACHE_DIR = "pythonService/data/cache/bm25"
os.makedirs(CACHE_DIR, exist_ok=True)

class SparseRetriever:
    def __init__(self):
        self.indices = {}  # in-memory cache {doc_id: BM25Okapi}

    def _get_cache_path(self, doc_id: str) -> str:
        return os.path.join(CACHE_DIR, f"{doc_id}.pkl")

    def indexDocument(self, doc_id: str, chunks: List[str]):
        """
        Build BM25 index for a document and cache it.
        """
        tokenized_chunks = [chunk.lower().split() for chunk in chunks]
        bm25 = BM25Okapi(tokenized_chunks)

        self.indices[doc_id] = bm25

        with open(self._get_cache_path(doc_id), "wb") as f:
            pickle.dump({"chunks": chunks, "bm25": bm25}, f)

        logger.info(f"BM25 index built and cached for document {doc_id}")

    def _load_index(self, doc_id: str):
        if doc_id in self.indices:
            return self.indices[doc_id]

        path = self._get_cache_path(doc_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No BM25 cache found for doc_id={doc_id}")

        with open(path, "rb") as f:
            data = pickle.load(f)
            bm25 = data["bm25"]
            self.indices[doc_id] = bm25
            self._cached_chunks = data["chunks"]
            return bm25

    def query(self, doc_id: str, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve top chunks for a query using BM25.
        Returns list of dicts: [{"chunk": str, "score": float}, ...]
        """
        bm25 = self._load_index(doc_id)
        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)

        # Load chunks from cache file
        with open(self._get_cache_path(doc_id), "rb") as f:
            data = pickle.load(f)
            chunks = data["chunks"]

        ranked = sorted(
            [{"chunk": c, "score": s} for c, s in zip(chunks, scores)],
            key=lambda x: x["score"],
            reverse=True
        )

        return ranked[:top_k]

# Singleton instance
sparseRetriever = SparseRetriever()
