# inspectChroma.py
import chromadb

client = chromadb.PersistentClient(path="/mnt/MyLinuxSpace/FYP/BlendedRag-DocumentAnalysisChatBot/pythonService/data/chroma")
collection = client.get_collection("documents")

print(f"Total documents: {collection.count()}")
target_doc_id = "3b254394-e4b9-4489-b7b4-1841b4e7abe6"  # Your recent upload
results = collection.get(include=["metadatas", "documents"])
for meta, doc in zip(results["metadatas"], results["documents"]):
    if meta and meta.get("docId"):
        if meta["docId"] == target_doc_id:
            print(f"\nDoc ID: {meta['docId']}")
            print(f"Chunk Index: {meta['chunkIndex']}")
            print(f"File: {meta.get('fileName', 'N/A')}")
            print(f"Page Count: {meta.get('pageCount', 'N/A')}")
            print(f"Text (first 100 chars): {doc[:100] + '...' if doc else 'No text available'}")