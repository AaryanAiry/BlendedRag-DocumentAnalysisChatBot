# testQueryEmbedding.py
import asyncio
from app.routes import queryRoutes

async def test_query():
    # Replace this with a docId that you ingested earlier
    doc_id = "7e5e5b9a-3c72-49f7-a214-b13e3bedeef8"

    # Test query
    test_query = "What is the summary of this document?"

    # Generate embedding using your client
    embedding = queryRoutes.embedding_client.generateEmbedding(test_query)
    print("Query embedding shape:", embedding.shape)  # Should be (384,)

    # Use chromaRetrieveTopK to see if retrieval works
    chunks = queryRoutes.chromaRetrieveTopK(doc_id, test_query, topK=2)
    print("Retrieved chunks:", chunks)

if __name__ == "__main__":
    asyncio.run(test_query())
