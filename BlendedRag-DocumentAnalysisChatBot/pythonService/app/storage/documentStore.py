from typing import Dict, Any, List
import threading
from app.chromaClient import chromaClient
from app.utils.logger import getLogger

logger = getLogger(__name__)

EMBEDDING_DIM = 384  # must match the embedding model

class DocumentStore:
    def __init__(self):
        self.lock = threading.Lock()
        # Chroma collection for all document chunks
        self.collection = chromaClient.get_or_create_collection(
            name="documents",
            metadata={"description": "PDF chunks with embeddings"}
        )
        # In-memory metadata for quick access and listing
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def saveDocument(self, docId: str, data: Dict[str, Any]) -> None:
        """Save document metadata + chunks + embeddings."""
        with self.lock:
            chunks = data["chunks"]
            embeddings = data.get("embeddings")
            if embeddings is not None and len(chunks) != len(embeddings):
                raise ValueError("Number of chunks and embeddings must match")

            # Normalize chunks to dict format if needed
            if chunks and isinstance(chunks[0], str):
                chunks = [{"text": c} for c in chunks]

            # Prepare vector data for Chroma
            if embeddings is not None:
                ids = [f"{docId}_{i}" for i in range(len(chunks))]
                metadatas = [{"docId": docId, "chunkIndex": i, "text": chunk["text"]}
                             for i, chunk in enumerate(chunks)]
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings.tolist(),
                    metadatas=metadatas,
                    documents=[c["text"] for c in chunks]
                )

            # Store basic document metadata in-memory
            self._metadata[docId] = {
                "fileName": data.get("fileName"),
                "pageCount": data.get("pageCount"),
                "numChunks": len(chunks)
            }

    def getDocument(self, docId: str) -> Dict[str, Any] | None:
        """Retrieve document chunks and metadata from Chroma."""
        try:
            # Use dummy embedding of correct dimension
            dummy_embedding = [0.0] * EMBEDDING_DIM

            result = self.collection.query(
                query_embeddings=[dummy_embedding],
                n_results=1000,
                where={"docId": docId}
            )
            if result and len(result.get("documents", [])) > 0:
                chunks = [
                    {
                        "chunkIndex": md["chunkIndex"],
                        "text": md["text"],
                        "score": 0.0
                    }
                    for md in result["metadatas"][0]
                ]
                metadata = self._metadata.get(docId, {})
                return {
                    "docId": docId,
                    "fileName": metadata.get("fileName"),
                    "pageCount": metadata.get("pageCount"),
                    "chunks": chunks
                }
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve document {docId}: {e}")
            return None

    def listDocuments(self) -> List[Dict[str, Any]]:
        """Return shallow metadata list of all documents."""
        with self.lock:
            return [{"docId": docId, **meta} for docId, meta in self._metadata.items()]

    def deleteDocument(self, docId: str) -> bool:
        """Delete all chunks and metadata for a document."""
        with self.lock:
            if docId in self._metadata:
                del self._metadata[docId]
            # delete from Chroma
            self.collection.delete(where={"docId": docId})
            return True

# Singleton instance
documentStore = DocumentStore()
