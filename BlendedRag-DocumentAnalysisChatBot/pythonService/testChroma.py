from app.storage.chromaClient import chromaClient, collection

# Test adding a dummy vector
dummy_texts = ["Hello world!", "Chroma test"]
dummy_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]  # small vectors

collection.add(
    ids=["1", "2"],
    documents=dummy_texts,
    embeddings=dummy_embeddings,
    metadatas=[{"info": "test1"}, {"info": "test2"}]
)

# Query
results = collection.query(
    query_embeddings=[[0.1, 0.2, 0.3]],
    n_results=1
)

print("Query results:", results)
