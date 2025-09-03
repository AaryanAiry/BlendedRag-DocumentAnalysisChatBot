# app/retrieval/blendedRetriever.py
from typing import List, Dict, Any, Tuple
import math

class BlendedRetriever:
    def __init__(self, chroma_client, embedding_fn, bm25_store=None):
        """
        chroma_client: Your existing Chroma client instance.
        embedding_fn: Callable to generate embeddings for queries.
        bm25_store: Optional BM25 index object (if using local or on-disk).
        """
        self.chroma = chroma_client
        self.embedding_fn = embedding_fn
        self.bm25_store = bm25_store

    def _reciprocal_rank_fusion(self, results: Dict[str, List[Tuple[str, float]]], weights: Dict[str, float]) -> List[Tuple[str, float]]:
        """
        Fuse BM25 and Dense scores using Reciprocal Rank Fusion (RRF) with weights.
        results: {"bm25": [(doc_id, score), ...], "dense": [(doc_id, score), ...]}
        weights: {"bm25": float, "dense": float}
        """
        fused = {}
        k = 60  # RRF constant
        for source, docs in results.items():
            w = weights.get(source, 0.5)
            for rank, (doc_id, _) in enumerate(docs, start=1):
                score = w * (1 / (k + rank))
                fused[doc_id] = fused.get(doc_id, 0) + score
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)

    def retrieve(self, doc_id: str, collection_name: str, queries: List[str], keywords: List[str], top_k_each: int = 20, top_k_final: int = 10) -> List[Dict[str, Any]]:
        """
        Blended retrieval.
        - queries: list of query variants (from RQ)
        - keywords: high-value tokens (from RQ)
        """
        results = {"bm25": [], "dense": []}

        # --- BM25 retrieval (if available) ---
        if self.bm25_store:
            bm25_docs = self.bm25_store.search(doc_id, keywords or queries, top_k=top_k_each)
            results["bm25"] = [(doc["id"], doc["score"]) for doc in bm25_docs]

        # --- Dense retrieval (Chroma) ---
        embeddings = [self.embedding_fn(q) for q in queries]
        dense_docs = self.chroma.query(
            collection_name=collection_name,
            query_embeddings=embeddings,
            n_results=top_k_each
        )
        dense_pairs = []
        for doc in dense_docs["documents"]:
            for d in doc:
                dense_pairs.append((d["id"], d["score"]))
        results["dense"] = dense_pairs

        # --- Fuse ---
        weights = {"bm25": 0.5, "dense": 0.5}  # can be overridden by RQ weightingHint
        fused = self._reciprocal_rank_fusion(results, weights)

        # --- Retrieve metadata ---
        final_docs = []
        seen = set()
        for doc_id, score in fused[:top_k_final]:
            if doc_id not in seen:
                meta = self.chroma.get_document_metadata(collection_name, doc_id)
                final_docs.append({"id": doc_id, "score": score, "metadata": meta})
                seen.add(doc_id)
        return final_docs
