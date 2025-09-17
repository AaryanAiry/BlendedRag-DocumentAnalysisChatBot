# inspectChroma2.py
import chromadb

client = chromadb.PersistentClient(path="./data/chroma")
collection = client.get_collection("documents")

print(f"Total documents: {collection.count()}")
target_doc_id = "3b254394-e4b9-4489-b7b4-1841b4e7abe6"  # Your docId
results = collection.get(where={"docId": target_doc_id}, include=["metadatas", "documents"])
for meta, doc in zip(results["metadatas"], results["documents"]):
    print(f"\nFull Metadata: {meta}")
    print(f"Doc ID: {meta.get('docId', 'N/A')}")
    print(f"Chunk Index: {meta.get('chunkIndex', 'N/A')}")
    print(f"File: {meta.get('fileName', 'N/A')}")
    print(f"Page Count: {meta.get('pageCount', 'N/A')}")
    print(f"Text (first 100 chars): {doc[:100] + '...' if doc else 'No text available'}")