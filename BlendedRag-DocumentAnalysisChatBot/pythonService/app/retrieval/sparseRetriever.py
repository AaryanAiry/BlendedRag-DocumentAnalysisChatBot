# app/retrieval/sparseRetriever.py
from rank_bm25 import BM25Okapi
from typing import List, Dict
import re

class SparseRetriever:
    def __init__(self, documents: List[Dict]):
        """
        documents: list of {"id": ..., "text": ...}
        """
        self.docs = documents
        self.corpus = [self._tokenize(d["text"]) for d in documents]
        self.bm25 = BM25Okapi(self.corpus)

    def _tokenize(self, text: str) -> List[str]:
        return [t.lower() for t in re.split(r"\W+", text) if t]

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.docs[i] for i in ranked_indices]
