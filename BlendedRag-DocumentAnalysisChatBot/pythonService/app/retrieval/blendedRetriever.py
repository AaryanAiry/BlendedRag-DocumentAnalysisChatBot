# app/retrievers/blendedRetriever.py
from typing import List, Dict
from app.retrieval.denseRetriever import DenseRetriever
from app.retrieval.sparseRetriever import SparseRetriever
from app.utils.logger import getLogger
from app.chromaClient import chromaClient
from app.embeddings.embeddingClient import EmbeddingClient

logger = getLogger(__name__)


class BlendedRetriever:
    def __init__(self, alpha: float = 0.6):
        """
        alpha: weight for dense retriever (0.6 = 60% dense, 40% sparse)
        """
        self.alpha = alpha
        embedding_client = EmbeddingClient()
        self.dense = DenseRetriever(
            chroma_client=chromaClient,
            embedding_fn=embedding_client.generateEmbedding
        )
        self.sparse = SparseRetriever()  # adjust if needs params

    def _normalize(self, scores: List[float]) -> List[float]:
        if not scores:
            return []
        min_s, max_s = min(scores), max(scores)
        if max_s - min_s == 0:
            return [0.5] * len(scores)  # neutral if all scores are the same
        return [(s - min_s) / (max_s - min_s) for s in scores]

    def query(self, doc_id: str, query: str, top_k: int = 5) -> List[Dict]:
        """
        Blends dense (semantic) and sparse (keyword) scores.
        Returns ranked chunks.
        """
        dense_results = self.dense.query(doc_id, query, top_k=top_k)
        sparse_results = self.sparse.query(doc_id, query, top_k=top_k)

        # Extract scores & chunks
        dense_scores = self._normalize([r["score"] for r in dense_results])
        sparse_scores = self._normalize([r["score"] for r in sparse_results])

        combined = {}

        # Merge dense first
        for i, r in enumerate(dense_results):
            combined[r["chunk"]["id"]] = {
                "chunk": r["chunk"],
                "score": self.alpha * dense_scores[i]
            }

        # Merge sparse (add weighted)
        for i, r in enumerate(sparse_results):
            chunk_data = r.get("chunk")
            if isinstance(chunk_data, dict):
                cid = chunk_data.get("id")
                text = chunk_data.get("text", "")
            else:  # it's a string
                cid = None
                text = chunk_data
            if cid in combined:
                combined[cid]["score"] += (1 - self.alpha) * sparse_scores[i]
            else:
                combined[cid] = {
                    "chunk": r["chunk"],
                    "score": (1 - self.alpha) * sparse_scores[i]
                }

        ranked = sorted(
            combined.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        return ranked[:top_k]


# Singleton instance
blendedRetriever = BlendedRetriever()
