# app/storage/documentStore.py
from typing import Dict, Any, List
import threading
import numpy as np
from app.storage.chromaClient import chromaClient

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
        """
        Save document metadata + chunks + embeddings.
        Chunks should be a list of dicts {"text": str}.
        Embeddings should be a np.ndarray matching chunks length.
        """
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
                    metadatas=metadatas
                )

            # Store basic document metadata in-memory
            self._metadata[docId] = {
                "fileName": data.get("fileName"),
                "pageCount": data.get("pageCount"),
                "numChunks": len(data.get("chunks", []))
            }

    def getDocument(self, docId: str) -> Dict[str, Any] | None:
        """
        Retrieve document metadata + chunks from Chroma
        """
        with self.lock:
            if docId not in self._metadata:
                return None

            # Retrieve chunks from Chroma
            result = self.collection.query(
                query_embeddings=[[0.0]],  # dummy query to filter by metadata
                n_results=1000,
                where={"docId": docId}
            )
            chunks = [{"text": md["text"]} for md in result['metadatas'] if "chunkIndex" in md]

            metadata = self._metadata[docId]
            return {
                "docId": docId,
                "fileName": metadata.get("fileName"),
                "pageCount": metadata.get("pageCount"),
                "chunks": chunks
            }

    def listDocuments(self) -> List[Dict[str, Any]]:
        """Return shallow metadata list of all documents"""
        with self.lock:
            return [
                {"docId": docId, **meta} for docId, meta in self.metadataIndex.items()
            ]

    def deleteDocument(self, docId: str) -> bool:
        """Delete all chunks and metadata for a document"""
        with self.lock:
            if docId in self.metadataIndex:
                del self.metadataIndex[docId]
            # delete from Chroma
            self.collection.delete(where={"docId": docId})
            return True

# Singleton
documentStore = DocumentStore()
