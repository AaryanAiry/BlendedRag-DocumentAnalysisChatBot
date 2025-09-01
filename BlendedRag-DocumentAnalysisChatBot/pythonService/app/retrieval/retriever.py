# app/retrieval/retriever.py
# Chroma-based semantic search

from typing import List, Dict, Any
from app.storage.documentStore import documentStore
from app.embeddings.embeddingClient import EmbeddingClient

# Singleton embedding client
embeddingClient = EmbeddingClient()

def retrieveTopK(docId: str, queryText: str, topK: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve top-K most similar chunks for a given query from a stored document using Chroma.
    
    Args:
        docId (str): ID of the document to query.
        queryText (str): User query text.
        topK (int): Number of top chunks to return.
    
    Returns:
        List[Dict[str, Any]]: List of chunks with 'chunkIndex', 'text', and 'score'.
    """
    # Get document from Chroma
    doc = documentStore.getDocument(docId)
    if not doc or len(doc.get("chunks", [])) == 0:
        return []

    # Generate embedding for the query
    queryVec = embeddingClient.generateEmbedding(queryText)

    # Query Chroma collection
    collection = documentStore.collection
    results = collection.query(
        query_embeddings=[queryVec.tolist()],
        n_results=topK,
        where={"docId": docId}
    )

    # Process results
    top_chunks = []
    for i, metadata in enumerate(results['metadatas'][0]):
        top_chunks.append({
            "chunkIndex": metadata["chunkIndex"],
            "text": metadata["text"],
            "score": float(results['distances'][0][i])
        })

    return top_chunks
