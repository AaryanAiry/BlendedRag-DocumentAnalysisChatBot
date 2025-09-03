# app/retrievers/blendedRetriever.py
from typing import List, Dict
from app.retrievers.denseRetriever import denseRetriever
from app.retrievers.sparseRetriever import sparseRetriever
from app.utils.logger import getLogger

logger = getLogger(__name__)

class BlendedRetriever:
    def __init__(self, alpha: float = 0.6):
        """
        alpha: weight for dense retriever (0.6 = 60% dense, 40% sparse)
        """
        self.alpha = alpha

    def _normalize(self, scores: List[float]) -> List[float]:
        if not scores:
            return []
        min_s, max_s = min(scores), max(scores)
        if max_s - min_s == 0:
            return [0.5] * len(scores)  # neutral if all scores same
        return [(s - min_s) / (max_s - min_s) for s in scores]

    def query(self, doc_id: str, query: str, top_k: int = 5) -> List[Dict]:
        """
        Blends dense (semantic) and sparse (keyword) scores.
        Returns ranked chunks.
        """
        dense_results = denseRetriever.query(doc_id, query, top_k=top_k)
        sparse_results = sparseRetriever.query(doc_id, query, top_k=top_k)

        # Extract scores & chunks
        dense_scores = self._normalize([r["score"] for r in dense_results])
        sparse_scores = self._normalize([r["score"] for r in sparse_results])

        combined = {}

        # Merge dense first
        for i, r in enumerate(dense_results):
            combined[r["chunk"]] = self.alpha * dense_scores[i]

        # Merge sparse (add weighted)
        for i, r in enumerate(sparse_results):
            if r["chunk"] in combined:
                combined[r["chunk"]] += (1 - self.alpha) * sparse_scores[i]
            else:
                combined[r["chunk"]] = (1 - self.alpha) * sparse_scores[i]

        ranked = sorted(
            [{"chunk": c, "score": s} for c, s in combined.items()],
            key=lambda x: x["score"],
            reverse=True
        )

        return ranked[:top_k]

# Singleton instance
blendedRetriever = BlendedRetriever()
